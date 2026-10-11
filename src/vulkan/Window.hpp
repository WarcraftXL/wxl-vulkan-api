// The client's own render window, read out of the engine rather than guessed at.
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

#include "vulkan/Vk.hpp"

#include "wxl/Script.hpp"
#include "wxl/game/Gx.hpp"

// The graphics-device field map is reached straight from the core: restating an offset in this
// repository would be a second copy to keep in step with the binary.
#include "wxl/offsets/engine/GxDevice.hpp"

namespace wxlvk::vk::window
{
    namespace gxdev = wxl::offsets::engine::gxdevice;

    /// The window class every D3D backend registers, from CGxDeviceD3d::ICreateWindow (0x0068EBB0).
    /// Used only by the fallback below, and only to confirm what the primary path already found.
    constexpr const char* kClassName = "GxWindowClassD3d";

    /// What one lookup produced, with enough context for the log line to explain itself.
    struct Found
    {
        HWND     hwnd    = nullptr;
        unsigned width   = 0;      ///< client area, from GetClientRect
        unsigned height  = 0;
        unsigned curWidth  = 0;    ///< the engine's live render resolution (curWindow)
        unsigned curHeight = 0;
        bool     windowed = false; ///< the active CGxFormat's windowed byte
        bool     fromEngine = false; ///< true via the device field, false via the class-name fallback
    };

    /**
     * The HWND the client renders into.
     *
     * Primary path, and the only one that is evidence rather than inference:
     * `CGxDeviceD3d::ICreateWindow` (0x0068EBB0) stores the HWND it creates at device + 0x3968, and
     * the core names that offset as `gxdevice::kWindow`. The device object itself is the
     * `gx::kGxDevicePtr` (0x00C5DF88) singleton, reached through the core's public
     * `wxl::game::gx::RawGraphicsDevice()`. Reference doc sections 9.5 and 9.6 both place the HWND
     * at +0x3968, and the core's own comment records that the field sits at the same place in every
     * stock backend -- which matters here because the object was built by the engine's factory
     * from this extension rather than by the engine itself, and the field map is identical for
     * exactly that reason.
     *
     * Fallback: FindWindowExA on the class name above, restricted to a window owned by this process.
     * It exists for the case where the device pointer is not up yet but a window already is -- and
     * it is a fallback rather than the primary path because a class-name search cannot tell the
     * render window from another window of the same class, whereas the device field is the engine's
     * own answer to the question.
     *
     * The window is validated, not trusted: IsWindow, then GetWindowThreadProcessId against our own
     * process id, then a non-empty client rect. A zero-size client area is reported as `hwnd` set
     * with width and height 0 so the caller can retry rather than treat a minimised window as a
     * failure -- `vkCreateSwapchainKHR` with a 0x0 extent is invalid usage.
     *
     * @return Found found : hwnd null when nothing usable was found
     */
    inline Found Locate()
    {
        Found found;

        if (void* device = wxl::game::gx::RawGraphicsDevice())
        {
            found.hwnd       = wxl::game::At<HWND>(device, gxdev::kWindow);
            found.fromEngine = true;

            // kWindowed is already the absolute device offset (kFormat + the format's windowed
            // byte), per the core's own static_assert chain against gx::kFormatWidth/Height.
            found.windowed = wxl::game::At<unsigned char>(device, gxdev::kWindowed) != 0;

            // curWindow is a rect of four floats with width at +0xC and height at +0x8; the core
            // names the two reads directly. This is the engine's live render resolution, which is
            // what a swapchain extent should agree with when the backbuffer is bound.
            const float curW = wxl::game::At<float>(device, wxl::offsets::engine::gx::kCurWindowWidth);
            const float curH = wxl::game::At<float>(device, wxl::offsets::engine::gx::kCurWindowHeight);
            if (curW > 0.0f && curW < 65536.0f) found.curWidth  = static_cast<unsigned>(curW);
            if (curH > 0.0f && curH < 65536.0f) found.curHeight = static_cast<unsigned>(curH);
        }

        if (!found.hwnd)
        {
            // Walk the windows of our own class until one belongs to this process. FindWindowExA
            // with a null parent and a null child walks the top-level list.
            const DWORD self = ::GetCurrentProcessId();
            HWND candidate = nullptr;
            while ((candidate = ::FindWindowExA(nullptr, candidate, kClassName, nullptr)) != nullptr)
            {
                DWORD owner = 0;
                ::GetWindowThreadProcessId(candidate, &owner);
                if (owner == self) break;
            }
            found.hwnd       = candidate;
            found.fromEngine = false;
        }

        if (!found.hwnd) return Found();

        if (!::IsWindow(found.hwnd))
        {
            wxl::ScriptMgr::Log(WXL_LOG_WARN,
                                "vulkan: window: %s produced 0x%08X, which is not a window; "
                                "ignoring it",
                                found.fromEngine ? "the device field at +0x3968"
                                                 : "the GxWindowClassD3d search",
                                static_cast<unsigned>(reinterpret_cast<uintptr_t>(found.hwnd)));
            return Found();
        }

        DWORD owner = 0;
        ::GetWindowThreadProcessId(found.hwnd, &owner);
        if (owner != ::GetCurrentProcessId())
        {
            wxl::ScriptMgr::Log(WXL_LOG_WARN,
                                "vulkan: window: 0x%08X belongs to process %lu, not to us (%lu); "
                                "ignoring it",
                                static_cast<unsigned>(reinterpret_cast<uintptr_t>(found.hwnd)),
                                owner, ::GetCurrentProcessId());
            return Found();
        }

        RECT client = {};
        if (::GetClientRect(found.hwnd, &client))
        {
            found.width  = static_cast<unsigned>(client.right - client.left);
            found.height = static_cast<unsigned>(client.bottom - client.top);
        }
        return found;
    }
}
