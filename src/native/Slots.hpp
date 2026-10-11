// The native device's slot implementations, declared once so src/native/Table.cpp can install them
// with the slot table's own type check.
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

#pragma once

#include "device/Slots.hpp"
#include "vulkan/Texture.hpp"
#include "vulkan/Vk.hpp"

#include <windows.h>

#include <cstdint>

// Every signature here must be the one src/device/Slots.hpp states for the slot -- Table.cpp's Put<>
// takes `SlotTraits<Slot>::Fn`, so a mismatch is a build error, not a `ret N` desync.
namespace wxlvk::native::impl
{
    using namespace wxlvk::device;

    // --- the slots, in slot order (the file that implements each in brackets) ---------------------
    void   __fastcall ITexMarkAsUpdated(void* device, void* edx, CGxTex* tex);                      // 0  (Textures)
    void   __fastcall IRsSendToHw(void* device, void* edx, uint32_t state);                         // 1  (Draw)
    void   __fastcall ICursorCreate(void* device, void* edx, const CGxFormat* format);              // 2  (Cursor)
    void   __fastcall ICursorDraw(void* device, void* edx);                                         // 4  (Scene)
    int    __fastcall DeviceCreateOnWindow(void* device, void* edx, HWND window, const CGxFormat* format); // 9 (Stubs)
    int    __fastcall DeviceCreate(void* device, void* edx, WNDPROC wndProc, const CGxFormat* format); // 10 (Device)
    void   __fastcall DeviceDestroy(void* device, void* edx);                                       // 11 (Device)
    void   __fastcall DeviceEvictResources(void* device, void* edx);                                // 12 (Device)
    int    __fastcall DeviceSetFormat(void* device, void* edx, const CGxFormat* format);            // 13 (Device)
    void   __fastcall DeviceSetBaseMipLevel(void* device, void* edx, uint32_t level);               // 14 (Textures)
    void   __fastcall DeviceSetGammaRamp(void* device, void* edx, const CGxGammaRamp* ramp);        // 15 (Device)
    void   __fastcall DeviceSetGammaLevel(void* device, void* edx, float level);                    // 16 (Device)
    void   __fastcall DeviceReadPixels(void* device, void* edx, CiRect* rect, TSGrowableArray* out); // 20 (Targets)
    void   __fastcall DeviceWM(void* device, void* edx, uint32_t event, uintptr_t p1, uintptr_t p2); // 22 (Device)
    void   __fastcall DeviceSetRenderTarget(void* device, void* edx, uint32_t buffer, CGxTex* tex,
                                            uint32_t face);                                         // 23 (Targets)
    void   __fastcall CaptureDepthStencilSurface(void* device, void* edx);                          // 24 (Targets)
    void   __fastcall BindDepthStencilSurface(void* device, void* edx);                             // 25 (Targets)
    void   __fastcall DeviceResolveDepthBuffer(void* device, void* edx, CGxTex* tex);               // 26 (Targets)
    void   __fastcall DeviceCopyTex(void* device, void* edx, CGxTex* src, uint32_t srcFace, CGxTex* dst,
                                    uint32_t dstFace);                                              // 27 (Targets)
    void   __fastcall LogCrashInfo(void* device, void* edx, char* buffer, uint32_t size);           // 37 (Stubs)
    void   __fastcall ScenePresent(void* device, void* edx);                                        // 38 (Scene)
    void   __fastcall SceneClear(void* device, void* edx, uint32_t mask, uint32_t color);           // 39 (Scene)
    void   __fastcall Draw(void* device, void* edx, CGxBatch* batch, int indexed);                  // 42 (Draw)
    void   __fastcall PoolSizeSet(void* device, void* edx, CGxPool* pool, uint32_t size);           // 52 (Pools)
    void   __fastcall PoolDestroy(void* device, void* edx, CGxPool* pool);                          // 53 (Pools)
    char*  __fastcall BufLock(void* device, void* edx, CGxBuf* buf);                                // 54 (Pools)
    int    __fastcall BufUnlock(void* device, void* edx, CGxBuf* buf, uint32_t bytesWritten);       // 55 (Pools)
    void   __fastcall BufData(void* device, void* edx, CGxBuf* buf, const void* data, uint32_t size,
                              uint32_t offset);                                                     // 56 (Pools)
    void   __fastcall TexDestroy(void* device, void* edx, CGxTex* tex);                             // 58 (Textures)
    bool   __fastcall TexCopy(void* device, void* edx, CGxTex* tex, C2iVector* dstPos, C2iVector* srcPos,
                              C2iVector* size, uint32_t level, uint32_t face);                      // 59 (Targets)
    bool   __fastcall TexStretch(void* device, void* edx, CGxTex* tex, CiRect* dst, CiRect* src,
                                 uint32_t level, uint32_t face);                                    // 60 (Targets)
    void   __fastcall QueryCreate(void* device, void* edx, CGxQuery** out, uint32_t type);          // 62 (Queries)
    void   __fastcall QueryDestroy(void* device, void* edx, CGxQuery** query);                      // 63 (Queries)
    bool   __fastcall QueryBegin(void* device, void* edx, CGxQuery* query);                         // 64 (Queries)
    uint32_t __fastcall QueryEnd(void* device, void* edx, CGxQuery* query);                         // 65 (Queries)
    int    __fastcall QueryGetParam(void* device, void* edx, CGxQuery* query, uint32_t param,
                                    uint32_t* out);                                                 // 66 (Queries)
    bool   __fastcall QueryGetData(void* device, void* edx, CGxQuery* query, void* out);            // 67 (Queries)
    void   __fastcall ShaderCreate(void* device, void* edx, CGxShader** out, uint32_t target,
                                   const char* directory, const char* name, uint32_t permutations); // 68 (Shaders)
    void   __fastcall ShaderDestroy(void* device, void* edx, CGxShader** shader);                   // 69 (Shaders)
    int    __fastcall IShaderReload(void* device, void* edx, CGxShader* shader, const char* directory,
                                    const char* name);                                              // 71 (Stubs)
    void   __fastcall IShaderCreate(void* device, void* edx, CGxShader* shader);                    // 72 (Shaders)
    void   __fastcall CursorSetVisible(void* device, void* edx, int visible);                       // 73 (Cursor)

    // --- the draw's resolver (Pools.cpp, Textures.cpp, Shaders.cpp) ----------------------------------

    /// A CGxPool's current VkBuffer, stamped as used by the frame being recorded.
    bool ResolvePool(const void* pool, VkBuffer& out);
    /// Set by ResolvePool() when the pool it resolved is a dynamic one (the only kind with drawn
    /// ranges); slot 42 clears it before the reader runs and notes the draw's bufs only when it
    /// is set afterwards (Pools.cpp).
    extern bool g_dynamicPoolResolved;
    /// A CGxTex's image, stamped as used by the frame being recorded.
    bool ResolveTexture(const void* tex, const wxlvk::vk::texture::Image*& image);
    /// Bumped whenever a CGxShader is created or destroyed (Shaders.cpp): the draw reader's
    /// per-stage memo of the bound shaders (name, validity, slice class) is keyed on it, since a
    /// create rewrites +0x20 / +0x2C and a destroyed shader's address may be reused.
    extern uint32_t g_shaderGeneration;

    // --- shared between the files ----------------------------------------------------------------------

    /// What ISetTexture's call of slot 0 does at bind: the lazy create / upload of a bound texture.
    void PrepareTexture(void* device, void* tex);

    /// A CGxBuf a draw just bound: a dynamic pool remembers the byte range, so a later lock of it
    /// in the same frame copies on write instead of changing what the draw sees (Pools.cpp).
    void NoteBufDrawn(const void* buf);

    /// Draw calls slot 42 received so far.
    uint64_t DrawsTotal();

    /**
     * Brings the Vulkan device up on the client window if it is not up yet, or moves the surface to
     * the window the device holds now when it is (Device.cpp).
     *
     * @param void* device : the CGxDevice
     * @return bool usable : false only for a permanent decline (no loader, no adapter, a create
     *         failed); a window that is not measurable yet is true, and retried from slot 38
     */
    bool BringVulkan(void* device);

    /**
     * D3D's Clear rectangle (cgxdevice 9.1 / 9.4): the viewport in pixels of the target by
     * IXformSetViewport's rounding, intersected with the scissor (IStateSyncScissorRect's) when
     * render state 20 is on, against `extent`. Also what a draw's viewport and scissor are placed by
     * (Draw.cpp).
     *
     * @param void* device : the CGxDevice
     * @param VkExtent2D extent : the target
     * @param VkViewport& viewport : receives the viewport (integer edges, depth range from +0xF80)
     * @param VkRect2D& scissor : receives the scissor (the viewport rect when state 20 is off)
     * @return bool any : false when the rectangle is empty
     */
    bool PlaceD3d(const void* device, const VkExtent2D& extent, VkViewport& viewport, VkRect2D& scissor);

    /// D3D9's pixel centres on Vulkan's: every draw's viewport moves this much right and down
    /// (Draw.cpp explains the 1/128).
    constexpr float kD3d9CentreShift = 0.5f - 1.0f / 128.0f;

    /**
     * The viewport and scissor a draw recorded now would get: PlaceD3d, then D3D9's pixel-centre
     * shift on the viewport, exactly as slot 42 places the engine's draws. The rendering API hands
     * it to its passes so their geometry lands on the engine's pixels and depth range.
     *
     * @return bool any : false when the rectangle is empty
     */
    bool PlaceDraw(const void* device, const VkExtent2D& extent, VkViewport& viewport, VkRect2D& scissor);

    /// Slot 22 saw a size change; slot 38 rebuilds the swapchain once after its next present
    /// instead of once per WM_SIZE of a drag.
    void RequestResize();
    bool TakeResize();

    /// A texture's image as slots 23 / 26 / 27 / 59 / 60 need it (Textures.cpp).
    struct TargetImage
    {
        wxlvk::vk::texture::Image* image = nullptr;
        VkImageView                view  = VK_NULL_HANDLE;  ///< the attachment view (with `attach`)
        uint64_t*                  stamp = nullptr;         ///< the record's reference serial
        bool                       depth = false;           ///< a format-12 depth image
        bool                       renderTarget = false;    ///< CGxTexFlags bit 7
    };

    /**
     * Creates the texture's image when it has none (D3D's "create only" from these slots), then
     * hands it; with `attach`, also its attachment view for `face` (level 0, one layer; a depth
     * image's own view).
     *
     * @return bool ok : false when the texture has no image, or (with `attach`) cannot be attached
     */
    bool TargetImageOf(void* device, void* tex, uint32_t face, bool attach, TargetImage& out);

    /// Slot 4's hardware branch: rebuilds the Windows cursor from +0x2960 when +0x3B4C is set.
    void HardwareCursorDraw(void* device);
    /// Subclasses a window slot 13 just made, for WM_SETCURSOR with the hardware cursor.
    void CursorWindowCreated(void* device, HWND hwnd);
    /// Destroys the hardware cursor before its window goes (slots 11 and 13).
    void CursorRelease();

    /// The overlay renderer the core's ImGui host draws through when no IDirect3DDevice9 exists
    /// (Overlay.cpp): registered with the core's wxl.ui.overlay at the first slot 13.
    void OverlayRegister();

    /// Slot 14 (Textures.cpp) recreated this many textures (Device.cpp counts them).
    void NoteTexturesRecreated(uint64_t count);

    // --- the stats lines (src/native/Stats.cpp calls them with WXL_VK_STATS) ---------------------------

    void LogSceneStats();
    void LogDrawStats();
    void LogPoolStats();
    void LogTextureStats();
    void LogShaderStats();
    void LogTargetStats();
    void LogQueryStats();
    void LogDeviceStats();
    void LogCursorStats();
    void LogOverlayStats();
    void LogStubStats();
}
