// The native device's life cycle and window: slots 10, 11, 12, 13 and 22.
// Copyright (C) 2026 WarcraftXL
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.

#include "native/Slots.hpp"

#include "native/Caps.hpp"
#include "native/Engine.hpp"
#include "native/Native.hpp"
#include "native/Present.hpp"
#include "shaders/Translated.hpp"
#include "vulkan/Context.hpp"

#include "wxl/Script.hpp"
#include "wxl/game/Binding.hpp"
#include "wxl/offsets/engine/Gx.hpp"
#include "wxl/offsets/engine/GxDevice.hpp"

#include <windows.h>

#include <cstdint>
#include <cstring>

// Each of these four is the D3D9 backend's own function (read in the decompilation: DeviceCreate
// 0x00690750, DeviceDestroy 0x006905F0, DeviceSetFormat 0x006904D0 with ICreateD3dDevice 0x0068F3D0
// and IDestroyD3d, DeviceWM 0x00690230) with every D3D call taken out and the Vulkan context put
// where the IDirect3DDevice9 was. What remains is the engine's own work -- its window class, its
// window, its window procedure, its render-state and light defaults, its cursor, its placeholder
// texture, its UI shaders -- done by the engine's own functions in the engine's own order, so the
// object ends slot 13 in the state D3D would have left it in, minus the D3D fields.
//
// The base DeviceSetFormat (0x006840F0) runs last, as D3D calls it: after the device is up, on the
// format ICreateWindow may have rewritten (the fullscreen-window styles resize it in place).
//
// Fullscreen is a borderless popup over the monitor at the desktop's mode: Vulkan 1.0 has no
// exclusive fullscreen. The engine keeps rendering at the format's resolution and the present pass
// scales that back buffer onto the swapchain. The device runs single-sampled whatever the format
// asks (caps say so; render state 19 is ignored).
namespace
{
    namespace gxdev   = wxl::offsets::engine::gxdevice;
    namespace gx      = wxl::offsets::engine::gx;
    namespace slots   = wxlvk::device::slots;
    namespace context = wxlvk::vk::context;
    namespace present = wxlvk::native::present;

    using wxl::game::At;
    using wxlvk::device::CGxFormat;
    using wxlvk::device::CGxGammaRamp;
    using wxlvk::device::CGxShader;
    using wxlvk::device::CGxTex;
    using wxlvk::native::Base;
    using wxlvk::native::Engine;
    using wxlvk::native::Own;

    bool g_resizePending = false;

    uint64_t g_formatSets;       ///< slot 13 calls that succeeded (the first is the device's creation)
    uint64_t g_formatFailures;
    uint64_t g_resizes;          ///< slot 22 size events with the device live (swapchain rebuilds asked)
    uint64_t g_fullscreenRects;  ///< slot 22 size rects replaced by the fullscreen render size
    uint64_t g_displayChanges;   ///< slot 22 event 1
    uint64_t g_focusChanges;     ///< slot 22 events 3 / 4
    uint64_t g_minimised;        ///< slot 22 size events reporting minimised / hidden
    uint64_t g_gammaSets;        ///< slots 15 / 16
    uint64_t g_mipLevelSets;     ///< slot 14
    uint64_t g_texturesRecreated;

    /// "Shaders\Vertex" and "Shaders\Pixel", as ICreateD3dDevice passes them. Ours rather than the
    /// engine's literals because those were not located; ShaderCreate copies the directory into the
    /// path it builds and keeps no pointer to it.
    constexpr const char* kUiVertexDir = "Shaders\\Vertex";
    constexpr const char* kUiPixelDir  = "Shaders\\Pixel";

    uint8_t* Bytes(void* device) { return static_cast<uint8_t*>(device); }

    /// The format the device holds (+0x1BC) is fullscreen: windowed byte 0 (CGxDevice::IDevIsWindowed).
    bool Fullscreen(void* device) { return At<uint8_t>(device, gxdev::kWindowed) == 0; }

    /**
     * Moves a fullscreen popup over the whole monitor it is on. ICreateWindow made it at the format's
     * size, which is the screen only after D3D's display-mode change; the engine keeps rendering at
     * the format's resolution (slot 22 hands it that size), scaled by the present pass.
     */
    void CoverMonitor(HWND hwnd)
    {
        if (!hwnd) return;
        MONITORINFO mi = {};
        mi.cbSize = sizeof mi;
        if (!GetMonitorInfoA(MonitorFromWindow(hwnd, MONITOR_DEFAULTTOPRIMARY), &mi)) return;
        const RECT& r = mi.rcMonitor;
        SetWindowPos(hwnd, HWND_TOP, r.left, r.top, r.right - r.left, r.bottom - r.top,
                     SWP_FRAMECHANGED | SWP_SHOWWINDOW);
    }

    /// The size rect slot 22 hands DeviceSetDefWindow in fullscreen: the format's, not the window's.
    void FormatRect(void* device, float rect[4])
    {
        rect[0] = 0.0f;
        rect[1] = 0.0f;
        rect[2] = static_cast<float>(At<int32_t>(device, gxdev::kFormat + gxdev::kFmtHeight));
        rect[3] = static_cast<float>(At<int32_t>(device, gxdev::kFormat + gxdev::kFmtWidth));
    }

    /// Re-applies the gamma ramp the device holds (+0x354) to the present pass, for the format now set.
    void ApplyGamma(void* device)
    {
        wxlvk::native::present::SetGamma(reinterpret_cast<const uint16_t*>(Bytes(device) + gxdev::kGammaRamp),
                                          reinterpret_cast<const uint16_t*>(Bytes(device) + gxdev::kGammaRampDesktop),
                                          Fullscreen(device));
    }

    /**
     * What IDestroyD3d does that is not D3D: the two UI shader globals through slot 69, the cursor
     * through slot 3, the placeholder texture through slot 58. Plus the native half: the presenter's
     * swapchain objects, then the swapchain and the surface (the window is about to go).
     *
     * @param void* device : the CGxDevice
     */
    void DestroyNative(void* device)
    {
        CGxShader** vertex = reinterpret_cast<CGxShader**>(gxdev::kUiVertexShader);
        for (unsigned i = 0; i < 2; ++i)
            if (vertex[i]) Own<slots::kShaderDestroy>(device)(device, nullptr, &vertex[i]);
        CGxShader** pixel = reinterpret_cast<CGxShader**>(gxdev::kUiPixelShader);
        if (*pixel) Own<slots::kShaderDestroy>(device)(device, nullptr, pixel);

        Own<slots::kICursorDestroy>(device)(device, nullptr);

        CGxTex*& placeholder = At<CGxTex*>(device, gxdev::kD3dPlaceholderTex);
        if (placeholder)
        {
            Own<slots::kTexDestroy>(device)(device, nullptr, placeholder);
            placeholder = nullptr;
        }

        present::ReleaseSwapchain();
        context::ReleaseWindow();
        wxlvk::native::impl::CursorRelease();   // before the window it shows on

        // D3D leaves +0xF58 alone here and rewrites it in ICreateD3dDevice. A native device with no
        // surface has no context, so the gate says so -- slot 38 and the slot-42 stub then do nothing
        // between here and the next slot 13.
        At<uint32_t>(device, gxdev::kContext) = 0;
    }

    /**
     * ICreateD3dDevice without D3D: the Vulkan device on the window ICreateWindow just made, then
     * everything else ICreateD3dDevice does, in its order.
     *
     * @param void* device : the CGxDevice
     * @param void* format : the local copy of the format, as ICreateWindow left it
     * @return bool ok : false when the Vulkan device could not be brought up
     */
    bool CreateNativeDevice(void* device, const void* format)
    {
        // The core's ImGui overlay renderer: registered once.
        wxlvk::native::impl::OverlayRegister();

        // 0. The manifest and the MD5 provider, before anything creates a shader (step 7 does):
        //    slot 72 decides validity by the bytes' hash. Needs no Vulkan, so a device that comes
        //    up late does not leave the UI shaders judged without it.
        wxlvk::shaders::translated::Install();

        // 1. The device. Where D3D9 CreateDevice ran. Then the draw path on it (memory, textures,
        //    shader modules, pipelines), before the placeholder texture and the UI shaders below,
        //    which are the first things that need it.
        if (!wxlvk::native::impl::BringVulkan(device)) return false;
        wxlvk::native::present::InitDrawPath();

        // 1b. One black frame now. The engine's first present follows the DBC load, some eight
        //     seconds of a window that would otherwise show whatever was under it (white, on a
        //     fresh window). Once per session: a format change keeps its previous picture until
        //     its own first present.
        wxlvk::native::present::PresentBlank();

        // 2. "+0xF58 = 1", D3D's line right after a successful CreateDevice.
        At<uint32_t>(device, gxdev::kContext) = 1;

        // 3. ISetCaps.
        wxlvk::native::caps::Fill(device);

        // 4. IStateSetD3DDefaults minus its D3D calls (0x006A3A60): force every render state dirty
        //    and sync -- slot 1 is a no-op, so this leaves the shadow equal to the app values, which
        //    is the state D3D leaves it in -- then the two "everything changed" markers, the lights,
        //    and ISceneBegin's non-D3D half, which is the first frame beginning.
        Engine<gxdev::IRsForceUpdateFn>(gxdev::kIRsForceUpdate)(device, nullptr);
        Engine<gxdev::IRsSyncFn>(gxdev::kIRsSync)(device, nullptr, 0);
        At<uint32_t>(device, gxdev::kAttribChanged)   = 0xFFFFFFFFu;
        At<uint32_t>(device, gxdev::kIndexBufChanged) = 0;
        Engine<gxdev::InitLightsFn>(gxdev::kInitLights)(device, nullptr);
        Engine<gxdev::ShaderConstantsClearFn>(gxdev::kShaderConstantsClear)();

        // 5. Slot 15 with the live ramp, slot 2 with the format -- through the table, as D3D does.
        //    15 hands the ramp to the present pass (applied in fullscreen, as D3D); 2 is the base's
        //    choice of cursor (caps +0x310 is 1: hardware when gxCursor asks for it), then the
        //    hardware cursor's first build.
        Own<slots::kDeviceSetGammaRamp>(device)(
            device, nullptr, reinterpret_cast<const CGxGammaRamp*>(Bytes(device) + gxdev::kGammaRamp));
        Own<slots::kICursorCreate>(device)(device, nullptr, static_cast<const CGxFormat*>(format));

        // 6. The 8x8 placeholder into +0x3B58: GxTexCreate + an immediate GxTexUpdate, the engine's
        //    own fill. Flags as CGxTexFlags(1,0,0,0,0,0,1,0,0,0) builds them: filter 1, max
        //    anisotropy 1 capped by caps +0x300 -- 0x201 on any adapter with anisotropy at all.
        const uint32_t aniso = At<uint32_t>(device, gxdev::kCapsMaxAnisotropy) >= 1u ? 1u : 0u;
        const uint32_t flags = 1u | (aniso << gxdev::kTexFlagAnisoShift);
        void** placeholder = reinterpret_cast<void**>(Bytes(device) + gxdev::kD3dPlaceholderTex);
        Engine<gxdev::GxTexCreateFn>(gxdev::kGxTexCreate)(
            8, 8, 2, flags, reinterpret_cast<void*>(static_cast<uintptr_t>(gxdev::kPlaceholderTexColor)),
            reinterpret_cast<gxdev::TexFillFn>(gxdev::kPlaceholderTexFill), placeholder);
        if (*placeholder) Engine<gxdev::GxTexUpdateFn>(gxdev::kGxTexUpdate)(*placeholder, 0, 0, 0, 0, 1);

        // 7. The UI shaders, through slot 68 of the device, into the engine's two globals.
        const char* ui = reinterpret_cast<const char*>(gxdev::kUiShaderName);
        Own<slots::kShaderCreate>(device)(device, nullptr,
                                          reinterpret_cast<CGxShader**>(gxdev::kUiVertexShader),
                                          gxdev::kShTargetVertex, kUiVertexDir, ui, 2);
        Own<slots::kShaderCreate>(device)(device, nullptr,
                                          reinterpret_cast<CGxShader**>(gxdev::kUiPixelShader),
                                          gxdev::kShTargetPixel, kUiPixelDir, ui, 1);
        return true;
    }

    /// Clips the cursor to the window, as D3D does in fullscreen (format byte +0x1C3 == 0).
    void ClipIfFullscreen(void* device)
    {
        if (At<uint8_t>(device, gxdev::kWindowed) != 0) return;
        RECT r = {};
        if (GetWindowRect(At<HWND>(device, gxdev::kWindow), &r)) ClipCursor(&r);
    }
}

namespace wxlvk::native::impl
{
    bool BringVulkan(void* device)
    {
        const HWND hwnd = At<HWND>(device, gxdev::kWindow);
        if (!context::Live())
        {
            // Synchronous, on the render thread, on the client's window: Bring() finds the window
            // through the same field (+0x3968, window::Locate).
            const context::Attempt attempt = context::Bring();
            if (attempt == context::Attempt::Declined)
            {
                wxl::ScriptMgr::Log(WXL_LOG_ERROR,
                                    "native: the Vulkan device could not be brought up on HWND 0x%08X "
                                    "(the lines above say why); slot 13 fails, so the engine reports "
                                    "it could not create its device",
                                    static_cast<unsigned>(reinterpret_cast<uintptr_t>(hwnd)));
                return false;
            }
            if (attempt == context::Attempt::Retry)
                wxl::ScriptMgr::Log(WXL_LOG_WARN,
                                    "native: the client window is not measurable yet (minimised?); "
                                    "the Vulkan device comes up on the first present that finds it");
            return true;
        }

        // A device that is up and a window that is new: the format changed, or the engine
        // restarted its graphics. Rebind() sees the HWND differ and rebuilds surface and swapchain;
        // the instance and the VkDevice stay.
        context::Rebind();
        return true;
    }

    void RequestResize() { g_resizePending = true; }

    bool TakeResize()
    {
        const bool pending = g_resizePending;
        g_resizePending = false;
        return pending;
    }

    // --- slot 10 ---------------------------------------------------------------------------------------

    int __fastcall DeviceCreate(void* device, void* edx, WNDPROC wndProc, const CGxFormat* format)
    {
        // D3D 0x00690750 minus ICreateD3d (Direct3DCreate9 and the adapter checks).
        At<uint32_t>(device, gxdev::kOwnsWindow) = 1;

        // The desktop gamma ramp, saved so it can be put back, and copied as the live one.
        HDC dc = GetDC(nullptr);
        if (dc)
        {
            if (GetDeviceGammaRamp(dc, Bytes(device) + gxdev::kGammaRampDesktop))
                std::memcpy(Bytes(device) + gxdev::kGammaRamp, Bytes(device) + gxdev::kGammaRampDesktop,
                            gxdev::kGammaRampBytes);
            ReleaseDC(nullptr, dc);
        }

        const uint16_t atom = Engine<gxdev::WindowClassCreateFn>(gxdev::kWindowClassCreate)();
        At<uint16_t>(device, gxdev::kWindowClassAtom) = atom;
        if (!atom)
        {
            wxl::ScriptMgr::Log(WXL_LOG_ERROR, "native: DeviceCreate: WindowClassCreate failed (%lu)",
                                static_cast<unsigned long>(GetLastError()));
            Own<slots::kDeviceDestroy>(device)(device, edx);
            return 0;
        }

        // The base half (0x00682CB0): stores the WNDPROC at +0xF54 and returns slot 13's answer --
        // called through the table, so slot 13 is ours.
        if (Base<slots::kDeviceCreate>()(device, edx, wndProc, format)) return 1;

        Own<slots::kDeviceDestroy>(device)(device, edx);
        return 0;
    }

    // --- slot 11 ---------------------------------------------------------------------------------------

    void __fastcall DeviceDestroy(void* device, void* edx)
    {
        // D3D 0x006905F0, reordered in one place: the surface has to go before the window it was
        // made from, so the native IDestroy runs before DestroyWindow rather than after it.
        Base<slots::kDeviceDestroy>()(device, edx);
        wxlvk::native::OnDeviceDestroy();
        DestroyNative(device);

        HWND& hwnd = At<HWND>(device, gxdev::kWindow);
        if (hwnd && At<uint32_t>(device, gxdev::kOwnsWindow))
        {
            DestroyWindow(hwnd);
            hwnd = nullptr;
        }
        uint16_t& atom = At<uint16_t>(device, gxdev::kWindowClassAtom);
        if (atom)
        {
            UnregisterClassA(reinterpret_cast<LPCSTR>(static_cast<uintptr_t>(atom)), GetModuleHandleA(nullptr));
            atom = 0;
        }

        // The instance and the VkDevice are deliberately kept: a slot 10 may follow (a graphics
        // restart builds a new CGxDevice), and at exit the process takes them --
        // a vkDestroyDevice from here would be safe, but one from DLL detach is not, and keeping one
        // rule for both is simpler than being right twice.
    }

    // --- slot 12 ---------------------------------------------------------------------------------------

    void __fastcall DeviceEvictResources(void* /*device*/, void* /*edx*/)
    {
        // CMap::Unload's call, on a zone change: D3D asks the runtime to evict managed resources.
        // Native has nothing resident to evict, but it is the moment the engine has just destroyed
        // the old zone's pools and textures, so the retirements their completed frames allow are
        // destroyed here (a fence poll, no wait) rather than at the next frame's open.
        present::Collect();
    }

    // --- slot 13 ---------------------------------------------------------------------------------------

    int __fastcall DeviceSetFormat(void* device, void* edx, const CGxFormat* format)
    {
        const uint8_t* f = reinterpret_cast<const uint8_t*>(format);
        wxl::ScriptMgr::Log(WXL_LOG_DEBUG,
                            "native: DeviceSetFormat %dx%d, %s, depth format %d, %d sample(s), vsync %d",
                            At<int32_t>(f, gxdev::kFmtWidth), At<int32_t>(f, gxdev::kFmtHeight),
                            At<uint8_t>(f, gxdev::kFmtWindowed) ? "windowed" : "fullscreen",
                            At<int32_t>(f, gxdev::kFmtDepthFormat), At<int32_t>(f, gxdev::kFmtSamples),
                            At<int32_t>(f, gxdev::kFmtVSync));

        // The old window and everything on it, as D3D: hide, destroy the device's objects, destroy
        // the window.
        HWND& hwnd = At<HWND>(device, gxdev::kWindow);
        if (hwnd) ShowWindow(hwnd, SW_HIDE);
        DestroyNative(device);
        if (hwnd) DestroyWindow(hwnd);
        hwnd = nullptr;

        std::memcpy(Bytes(device) + gxdev::kFormat, format, gxdev::kFormatSize);
        // A local copy, because ICreateWindow rewrites the size and position of the fullscreen-window
        // styles in place, and the base half below copies what it ends with.
        uint8_t local[gxdev::kFormatSize];
        std::memcpy(local, format, sizeof local);

        // Before the swapchain is built: the format's vsync picks its present mode.
        context::SetVsync(At<int32_t>(local, gxdev::kFmtVSync) != 0);

        // The back buffer's size. Fullscreen renders at the format's resolution on a window covering
        // the monitor; windowed follows the window (0 x 0).
        const bool fullscreen = At<uint8_t>(local, gxdev::kFmtWindowed) == 0;
        const int32_t fmtW = At<int32_t>(local, gxdev::kFmtWidth);
        const int32_t fmtH = At<int32_t>(local, gxdev::kFmtHeight);
        present::SetRenderExtent(fullscreen && fmtW > 0 ? uint32_t(fmtW) : 0u,
                                 fullscreen && fmtH > 0 ? uint32_t(fmtH) : 0u);

        bool ok = Engine<gxdev::CreateWindowFn>(gxdev::kCreateWindow)(device, edx, local);
        if (!ok)
            wxl::ScriptMgr::Log(WXL_LOG_ERROR, "native: DeviceSetFormat: ICreateWindow failed");
        if (ok)
        {
            // The window's WM_SETCURSOR for the hardware cursor, then fullscreen over the monitor.
            // The resize this makes reaches slot 22 with the context still 0: only defWindow moves.
            impl::CursorWindowCreated(device, hwnd);
            if (fullscreen) CoverMonitor(hwnd);
        }
        ok = ok && CreateNativeDevice(device, local);
        ok = ok && Base<slots::kDeviceSetFormat>()(device, edx, reinterpret_cast<const CGxFormat*>(local)) != 0;

        if (!ok)
        {
            ++g_formatFailures;
            wxl::ScriptMgr::Log(WXL_LOG_ERROR, "native: DeviceSetFormat: unable to set format");
            DestroyNative(device);
            if (hwnd) DestroyWindow(hwnd);
            hwnd = nullptr;
            return 0;
        }

        At<uint32_t>(device, gxdev::kHasFocus)         = 1;
        At<uint32_t>(device, gxdev::kD3dCursorDirty)   = 1;
        ClipIfFullscreen(device);

        // The ramp again, now that the format the base copied says windowed or fullscreen.
        ApplyGamma(device);

        ++g_formatSets;
        if (g_formatSets > 1)
        {
            // A format change. D3D's IDestroyD3d forced every texture's recreation, which ends in
            // slot 6 (texture-recreation callbacks: UpdateAllPortraits 0x00617070 and
            // CShadowCache::NotifyTexturesRecreate); native keeps every texture, but the targets the
            // engine sized by the screen are stale, so the callbacks run as D3D runs them, and the
            // device-restored ones too (slot 5: UpdateAllPortraits, and 0x007E7FE0, the FFX passes'
            // re-create flag), which D3D reaches after its next Reset.
            Own<slots::kNotifyOnTextureRecreation>(device)(device, edx);
            Own<slots::kNotifyOnDeviceRestored>(device)(device, edx);
        }

        wxl::ScriptMgr::Log(WXL_LOG_DEBUG,
                            "native: DeviceSetFormat done on HWND 0x%08X: Vulkan %s, context up, caps "
                            "filled, defaults set, cursor, placeholder and UI shaders created",
                            static_cast<unsigned>(reinterpret_cast<uintptr_t>(hwnd)),
                            context::Live() ? "device up" : "device pending (window not measurable)");
        return 1;
    }

    // --- slot 22 ---------------------------------------------------------------------------------------

    void __fastcall DeviceWM(void* device, void* edx, uint32_t event, uintptr_t p1, uintptr_t p2)
    {
        // D3D 0x00690230, event by event. WindowProcD3d calls this for size (0), display change (1),
        // and focus (2/4 lost, 3 gained). It must work before the Vulkan device exists: the first
        // WM_SIZE arrives inside CreateWindowExA, in the middle of slot 13.
        switch (event)
        {
            case 0:
            {
                // p2: 0 restored / resized, 1 minimised (or a fullscreen window deactivated), 2
                // hidden by another window maximising, 3 a fullscreen window activated again.
                if (p2 == 1 || p2 == 2)
                {
                    ++g_minimised;
                    At<uint32_t>(device, gxdev::kSizeTracking) = 0;
                    return;
                }
                At<uint32_t>(device, gxdev::kSizeTracking) = 1;
                // Fullscreen renders at the format's resolution on a monitor-sized window; the
                // engine's defWindow (its render size) is the format's, the present pass scales.
                float fsRect[4];
                const float* rect = reinterpret_cast<const float*>(p1);
                if (Fullscreen(device))
                {
                    FormatRect(device, fsRect);
                    rect = fsRect;
                    ++g_fullscreenRects;
                }
                Engine<gxdev::DeviceSetDefWindowFn>(gx::kDeviceSetDefWindow)(device, edx, rect);
                if (At<uint32_t>(device, gxdev::kContext) && context::Live())
                {
                    // Where D3D Resets. Nothing native is lost on a resize, so the only work is the
                    // swapchain, and that waits for the next present (one rebuild per frame rather
                    // than one per WM_SIZE of a drag). The device-restored callbacks still fire, as
                    // D3D fires them, because engine code resizes its own targets there (the FFX
                    // passes' flag 0x007E7FE0, UpdateAllPortraits).
                    ++g_resizes;
                    RequestResize();
                    Own<slots::kNotifyOnDeviceRestored>(device)(device, edx);
                }
                At<uint32_t>(device, gx::kViewportDirty) = 1;
                return;
            }
            case 1:
                // A display change while fullscreen. D3D re-reads the size; the native popup is moved
                // back over its monitor (its size may be the new desktop's) and keeps the format's
                // render size.
                ++g_displayChanges;
                if (Fullscreen(device))
                {
                    CoverMonitor(At<HWND>(device, gxdev::kWindow));
                    if (At<uint32_t>(device, gxdev::kContext) && context::Live()) RequestResize();
                }
                if (At<uint32_t>(device, gxdev::kSizeTracking))
                {
                    float fsRect[4];
                    const float* rect = reinterpret_cast<const float*>(p1);
                    if (Fullscreen(device))
                    {
                        FormatRect(device, fsRect);
                        rect = fsRect;
                    }
                    Engine<gxdev::DeviceSetDefWindowFn>(gx::kDeviceSetDefWindow)(device, edx, rect);
                    At<uint32_t>(device, gx::kViewportDirty) = 1;
                }
                return;
            case 2:
            case 4:
                ++g_focusChanges;
                At<uint32_t>(device, gxdev::kHasFocus) = 0;
                // A borderless fullscreen window stays on screen behind the one Alt-Tab chose; the
                // cursor clip slot 13 / event 3 set must not follow the user there.
                if (event == 4 && Fullscreen(device)) ClipCursor(nullptr);
                return;
            case 3:
                ++g_focusChanges;
                At<uint32_t>(device, gxdev::kHasFocus)       = 1;
                At<uint32_t>(device, gxdev::kD3dCursorDirty) = 1;
                ClipIfFullscreen(device);
                return;
            default:
                return;
        }
    }

    // --- slots 15 / 16 -----------------------------------------------------------------------------------

    void __fastcall DeviceSetGammaRamp(void* device, void* edx, const CGxGammaRamp* ramp)
    {
        // D3D 0x0069FE80: the base copies the ramp to +0x354, then SetGammaRamp in fullscreen. Native:
        // the ramp is applied by the present pass, fullscreen only, as D3D.
        Base<slots::kDeviceSetGammaRamp>()(device, edx, ramp);
        ++g_gammaSets;
        ApplyGamma(device);
    }

    void __fastcall DeviceSetGammaLevel(void* device, void* edx, float level)
    {
        // D3D 0x0068E4C0: the base builds the ramp from the level into +0x354 (CGxGammaRamp::Set).
        Base<slots::kDeviceSetGammaLevel>()(device, edx, level);
        ++g_gammaSets;
        ApplyGamma(device);
    }

    void NoteTexturesRecreated(uint64_t count)
    {
        ++g_mipLevelSets;
        g_texturesRecreated += count;
    }

    void LogDeviceStats()
    {
        wxlvk::native::present::Stats ps = {};
        wxlvk::native::present::GetStats(ps);
        const VkExtent2D sw = wxlvk::native::present::SwapExtent();
        wxl::ScriptMgr::Log(WXL_LOG_INFO,
                            "native: device: %llu format change(s) (%llu failed), %llu resize(s), %llu "
                            "minimise / deactivate event(s), %llu fullscreen size rect(s) replaced, %llu display "
                            "change(s), %llu focus change(s); gamma %llu set(s), %s; base mip level %llu set(s), "
                            "%llu texture(s) recreated",
                            static_cast<unsigned long long>(g_formatSets ? g_formatSets - 1 : 0),
                            static_cast<unsigned long long>(g_formatFailures),
                            static_cast<unsigned long long>(g_resizes), static_cast<unsigned long long>(g_minimised),
                            static_cast<unsigned long long>(g_fullscreenRects),
                            static_cast<unsigned long long>(g_displayChanges),
                            static_cast<unsigned long long>(g_focusChanges),
                            static_cast<unsigned long long>(g_gammaSets),
                            wxlvk::native::present::GammaActive() ? "applied" : "not applied",
                            static_cast<unsigned long long>(g_mipLevelSets),
                            static_cast<unsigned long long>(g_texturesRecreated));
        wxl::ScriptMgr::Log(WXL_LOG_INFO,
                            "native: present: swapchain %ux%u; %llu present pass(es) (%llu scaled, %llu with "
                            "gamma, %llu LUT upload(s), %llu fell back to a copy), %llu frame(s) with nothing "
                            "acquired (window hidden)",
                            sw.width, sw.height, static_cast<unsigned long long>(ps.presentPasses),
                            static_cast<unsigned long long>(ps.scaledPresents),
                            static_cast<unsigned long long>(ps.gammaPresents),
                            static_cast<unsigned long long>(ps.lutUploads),
                            static_cast<unsigned long long>(ps.presentPassRefused),
                            static_cast<unsigned long long>(ps.hiddenFrames));
    }
}
