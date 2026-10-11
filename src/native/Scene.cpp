// The native device's scene slots: 38 (present), 39 (clear) and 4 (the software cursor).
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

#include "native/Engine.hpp"
#include "native/Native.hpp"
#include "native/Present.hpp"
#include "vulkan/Context.hpp"
#include "vulkan/Loader.hpp"

#include "wxl/Script.hpp"
#include "wxl/game/Binding.hpp"
#include "wxl/offsets/engine/GxDevice.hpp"

#include <windows.h>

#include <cstdint>

namespace
{
    namespace gxdev   = wxl::offsets::engine::gxdevice;
    namespace slots   = wxlvk::device::slots;
    namespace context = wxlvk::vk::context;
    namespace present = wxlvk::native::present;

    using wxl::game::At;
    using wxlvk::native::Base;
    using wxlvk::native::Engine;
    using wxlvk::native::Own;

    uint64_t g_presents;
    uint64_t g_clears;
    uint64_t g_colourClears;
    uint64_t g_clearsSkipped;    ///< clears with no pass to record into
    uint64_t g_clearsInTarget;   ///< clears recorded while a texture was the colour target
    uint64_t g_clearsFolded;     ///< clears that became (part of) the load op of the pass they began

    /// A slot-39 clear's rectangle on a target of `extent`: D3D's (Draw.cpp PlaceD3d).
    bool PlaceClear(const void* device, const VkExtent2D& extent, VkRect2D& rect)
    {
        VkViewport viewport = {};
        return wxlvk::native::impl::PlaceD3d(device, extent, viewport, rect);
    }

    uint64_t g_cursorDraws;      ///< slot 4 calls that ran the base software-cursor draw
    uint64_t g_cursorHardware;   ///< slot 4 calls with +0x2954 set (the hardware cursor, Cursor.cpp)

    /// True when there is nothing to present into: no window, or a minimised / zero-size one. An
    /// acquire on a zero-extent surface is OUT_OF_DATE every frame and a rebuild every frame.
    bool WindowHidden(void* device)
    {
        const HWND hwnd = At<HWND>(device, gxdev::kWindow);
        if (!hwnd || IsIconic(hwnd)) return true;
        RECT r = {};
        return !GetClientRect(hwnd, &r) || r.right <= r.left || r.bottom <= r.top;
    }
}

namespace wxlvk::native::impl
{
    void __fastcall ScenePresent(void* device, void* edx)
    {
        // D3D 0x006A3450, with the Vulkan present where Present was.
        if (At<uint32_t>(device, gxdev::kContext))
        {
            // Read before the base half clears it, as D3D does.
            const uint32_t screenshot = At<uint32_t>(device, gxdev::kScreenshotPending);

            // Frame counter, screenshot flag, the poisoned-frame flag.
            Base<slots::kScenePresent>()(device, edx);

            // The software cursor, through the table. D3D draws it here, after EndScene -- inside its
            // own Begin/EndScene pair; here into the frame still open on the back buffer.
            Own<slots::kICursorDraw>(device)(device, edx);

            // Format byte +0x1C2 ("fix lag") asks D3D for a GPU sync at this point. The presenter
            // already waits for the frame-in-flight slot it reuses, one frame behind, which is the
            // same bound; nothing more is done for it.

            if (screenshot) Engine<gxdev::DeviceScreenShotFn>(gxdev::kDeviceScreenShot)(device, edx);

            Engine<gxdev::WaitForFpsCapFn>(gxdev::kWaitForFpsCap)(device, edx);

            // Hidden frames (minimised, zero-size) are submitted with nothing acquired, so a
            // zero-size surface never reaches the swapchain; the restore's WM_SIZE rebuilds it.
            const bool visible = !WindowHidden(device);
            if (visible && !context::Live())
            {
                // A window that was not measurable at slot 13 gets its device now, and the draw
                // path with it.
                BringVulkan(device);
                present::InitDrawPath();
            }
            // A resize marks the swapchain stale; the frame rebuilds it once, after this present.
            if (TakeResize()) present::RequestRebuild();
            // Submitted whether or not the window can show it: the frame's transfers must run.
            present::Present(visible);

            ++g_presents;
            wxlvk::native::OnPresent(g_presents);
        }

        // ISceneBegin's half that is not D3D (0x006A3350): the next frame begins with every constant
        // marked for re-upload -- the engine relies on that reset once a frame.
        Engine<gxdev::ShaderConstantsClearFn>(gxdev::kShaderConstantsClear)();
    }

    void __fastcall ICursorDraw(void* device, void* edx)
    {
        // D3D 0x0068E810. With the hardware cursor (+0x2954 set: caps +0x310 offers one and the
        // format's gxCursor asks for it) D3D's SetCursorProperties branch becomes a Windows cursor
        // made from the same bits and hotspot (src/native/Cursor.cpp), rebuilt when +0x3B4C says
        // they changed. Otherwise it is ISceneBegin, the base draw, ISceneEnd: the base (0x00687A90)
        // checks +0x2950 (visible), +0x2954 (hardware) and the mouse inside the window itself, then
        // draws the 32x32 cursor texture (+0x3960, created by slot 2, filled by its ICursorUpdate
        // callback through slot 0) as one strip through slot 42 -- fixed function, blend 1, no
        // depth, with the half-pixel shift of api 1/2. Engine code only: no D3D object is touched.
        if (!At<uint32_t>(device, gxdev::kContext)) return;
        if (At<uint32_t>(device, gxdev::kCursorHardware))
        {
            ++g_cursorHardware;
            HardwareCursorDraw(device);
            return;
        }
        ++g_cursorDraws;
        Base<slots::kICursorDraw>()(device, edx);
    }

    void __fastcall SceneClear(void* device, void* /*edx*/, uint32_t mask, uint32_t color)
    {
        // D3D 0x006A74B0: Clear(0, NULL, flags, color, 1.0f, 0) -- which clears only the viewport
        // rectangle, and the scissor rectangle with state 20 on (cgxdevice 9.1). Native records
        // vkCmdClearAttachments at the clear's place in the frame's pass, over the same rectangle
        // (PlaceD3d, Draw.cpp), with depth to 1.0. Like D3D, not affected by the colour mask.
        if (!At<uint32_t>(device, gxdev::kContext)) return;
        ++g_clears;
        if (mask & 1u) ++g_colourClears;
        if ((mask & 3u) == 0) return;

        // The pass is on the current target (slot 23's), so a clear while a texture is the colour
        // target clears that texture, inside its viewport, as D3D's does.
        if (At<const void*>(device, gxdev::kRenderTargets + gxdev::kBufferColor * gxdev::kRenderTargetStride
                                        + gxdev::kRenderTargetTex) != nullptr)
            ++g_clearsInTarget;

        // The CImVector dword is D3DCOLOR 0xAARRGGBB; the attachment's float clear is r g b a.
        present::PassClear pending;
        pending.mask      = mask & 3u;
        pending.colour[0] = static_cast<float>((color >> 16) & 0xFFu) / 255.0f;
        pending.colour[1] = static_cast<float>((color >> 8) & 0xFFu) / 255.0f;
        pending.colour[2] = static_cast<float>(color & 0xFFu) / 255.0f;
        pending.colour[3] = static_cast<float>(color >> 24) / 255.0f;
        pending.device    = device;
        pending.place     = &PlaceClear;
        // A clear that begins the pass over its whole area is the pass's load op; `folded` says
        // which aspects that took, and only the others are recorded below.
        uint32_t        folded = 0;
        VkCommandBuffer cmd    = present::PassClearing(pending, folded);
        if (!cmd)
        {
            ++g_clearsSkipped;
            return;
        }
        if (folded) ++g_clearsFolded;
        VkViewport viewport = {};
        VkRect2D   rect     = {};
        if (!PlaceD3d(device, present::Extent(), viewport, rect)) return;

        VkClearAttachment a[2] = {};
        uint32_t          n    = 0;
        if ((mask & 1u) && !(folded & 1u))
        {
            a[n].aspectMask      = VK_IMAGE_ASPECT_COLOR_BIT;
            a[n].colorAttachment = 0;
            for (uint32_t i = 0; i < 4; ++i) a[n].clearValue.color.float32[i] = pending.colour[i];
            ++n;
        }
        if ((mask & 2u) && !(folded & 2u) && present::HasDepth())
        {
            a[n].aspectMask                      = VK_IMAGE_ASPECT_DEPTH_BIT;
            a[n].clearValue.depthStencil.depth   = 1.0f;
            a[n].clearValue.depthStencil.stencil = 0;
            ++n;
        }
        if (n == 0) return;
        VkClearRect r = {};
        r.rect       = rect;
        r.layerCount = 1;
        context::Get()->dapi->CmdClearAttachments(cmd, n, a, 1, &r);
    }

    void LogSceneStats()
    {
        wxl::ScriptMgr::Log(WXL_LOG_INFO,
                            "native: scene: %llu present(s); %llu clear(s) (%llu with colour, %llu in a "
                            "texture target, %llu with no pass, %llu folded into a load op); cursor: %llu "
                            "software draw(s), %llu hardware",
                            static_cast<unsigned long long>(g_presents),
                            static_cast<unsigned long long>(g_clears),
                            static_cast<unsigned long long>(g_colourClears),
                            static_cast<unsigned long long>(g_clearsInTarget),
                            static_cast<unsigned long long>(g_clearsSkipped),
                            static_cast<unsigned long long>(g_clearsFolded),
                            static_cast<unsigned long long>(g_cursorDraws),
                            static_cast<unsigned long long>(g_cursorHardware));
    }
}
