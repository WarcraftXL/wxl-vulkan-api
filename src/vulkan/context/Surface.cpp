// The window's surface.
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

#include "vulkan/context/Internal.hpp"

namespace wxlvk::vk::context::detail
{
    // --- surface ----------------------------------------------------------------------------------

    /**
     * Creates the Win32 surface on an HWND.
     *
     * The HINSTANCE is read off the window itself (GWLP_HINSTANCE) rather than assumed: it is the
     * module that registered the window's class -- Wow.exe for "GxWindowClassD3d"
     * (CGxDeviceD3d::IWindowClassCreate, 0x0068EB20) -- and the window's own answer cannot drift
     * from it. The spec only requires a valid HINSTANCE, but a layer or driver may look the class up
     * with it.
     *
     * @param HWND hwnd : the window to present into
     * @return bool ok
     */
    bool CreateSurface(HWND hwnd)
    {
        VkWin32SurfaceCreateInfoKHR info = {};
        info.sType     = VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR;
        info.hinstance = reinterpret_cast<HINSTANCE>(
            static_cast<uintptr_t>(::GetWindowLongPtrA(hwnd, GWLP_HINSTANCE)));
        if (!info.hinstance) info.hinstance = ::GetModuleHandleA(nullptr);
        info.hwnd      = hwnd;

        const VkResult r = g.iapi.CreateWin32SurfaceKHR(g.instance, &info, nullptr, &g.surface);
        if (r != VK_SUCCESS)
        {
            wxl::ScriptMgr::Log(WXL_LOG_WARN, "vulkan: vkCreateWin32SurfaceKHR failed: %s",
                                ResultName(r));
            g.surface = VK_NULL_HANDLE;
            return false;
        }
        g.hwnd = hwnd;
        return true;
    }
}
