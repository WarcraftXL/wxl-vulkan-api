// The Vulkan objects this extension owns: instance, physical device, logical device, queues,
// Win32 surface and swapchain. Created at runtime, destroyed in reverse order.
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

using namespace wxlvk::vk::context::detail;

namespace wxlvk::vk::context
{
    Attempt Bring()
    {
        if (g.settled) return g.device ? Attempt::Done : Attempt::Declined;

        if (++g.attempts > kMaxAttempts)
        {
            wxl::ScriptMgr::Log(WXL_LOG_WARN,
                                "vulkan: gave up after %u frames waiting for a usable window; "
                                "declining", kMaxAttempts);
            g.settled = true;
            Teardown();
            return Attempt::Declined;
        }

        // The window is checked first, before anything is created, so a client that is still putting
        // its window up costs a GetClientRect per attempt and nothing else.
        const window::Found win = window::Locate();

        if (!win.hwnd || win.width == 0 || win.height == 0)
        {
            // Only the first wait is logged; the rest would be one line per frame.
            if (g.attempts == 1)
                wxl::ScriptMgr::Log(WXL_LOG_DEBUG,
                                    "vulkan: no usable window yet (%s); waiting",
                                    win.hwnd ? "zero-size client area" : "no HWND");
            return Attempt::Retry;
        }

        g.baseline = footprint::Take();
        wxl::ScriptMgr::Log(WXL_LOG_DEBUG,
                            "vulkan: bring-up starting on HWND 0x%08X (%ux%u client, %s, from %s); "
                            "address space before: committed %u MiB, reserved %u MiB, largest free "
                            "block %u MiB",
                            static_cast<unsigned>(reinterpret_cast<uintptr_t>(win.hwnd)),
                            win.width, win.height, win.windowed ? "windowed" : "fullscreen",
                            win.fromEngine ? "the device field at +0x3968" : "a GxWindowClassD3d search",
                            footprint::Mib(g.baseline.committed), footprint::Mib(g.baseline.reserved),
                            footprint::Mib(g.baseline.largestFree));

        // Each step is final: a failure here is a permanent decline, not a retry, because none of
        // these failures is a function of time. Teardown() unwinds whatever did get created.
        g.settled = true;

        if (!loader::Load())                 { Teardown(); return Attempt::Declined; }
        g.afterLoader = footprint::Take();
        LogFootprint("loading vulkan-1.dll", g.afterLoader);

        if (!CreateInstance())               { Teardown(); return Attempt::Declined; }
        g.afterInstance = footprint::Take();
        LogFootprint("vkCreateInstance", g.afterInstance);

        // The surface before the physical-device choice, because present support on THIS surface is
        // one of the three requirements that choice is made on.
        if (!CreateSurface(win.hwnd))        { Teardown(); return Attempt::Declined; }
        if (!SelectPhysicalDevice())         { Teardown(); return Attempt::Declined; }
        if (!CreateDevice())                 { Teardown(); return Attempt::Declined; }
        g.afterDevice = footprint::Take();
        LogFootprint("vkCreateDevice", g.afterDevice);

        if (!CreateSwapchain(win))
        {
            // The instance, device and surface are good; only the swapchain failed. They are kept:
            // the next rebuild (a resize, a present that finds it missing) tries again.
            wxl::ScriptMgr::Log(WXL_LOG_WARN,
                                "vulkan: instance, device and surface stand; the swapchain does not");
        }

        const footprint::Snapshot now = footprint::Take();
        LogFootprint("the whole bring-up", now);
        return Attempt::Done;
    }

    void ReleaseWindow()
    {
        if (!g.device) return;
        DropSwapchainObject();
        DropSurface();
    }

    void Rebind()
    {
        if (!g.device) return;

        const window::Found win = window::Locate();
        if (!win.hwnd)
        {
            wxl::ScriptMgr::Log(WXL_LOG_WARN,
                                "vulkan: no client window can be found; the surface and swapchain "
                                "stay down");
            DropSwapchainObject();
            DropSurface();
            return;
        }

        // A format change or a graphics restart destroys the window and makes a new one, so the
        // surface's HWND is the thing to compare -- not the extent.
        if (win.hwnd != g.hwnd)
        {
            wxl::ScriptMgr::Log(WXL_LOG_DEBUG,
                                "vulkan: the client window changed (0x%08X -> 0x%08X); rebuilding the "
                                "surface as well as the swapchain",
                                static_cast<unsigned>(reinterpret_cast<uintptr_t>(g.hwnd)),
                                static_cast<unsigned>(reinterpret_cast<uintptr_t>(win.hwnd)));
            DropSwapchainObject();
            DropSurface();
            if (!CreateSurface(win.hwnd)) return;

            // The adapter is not re-chosen. Present support is re-verified instead, because that is
            // the one property of the choice that a new window can invalidate -- a window moved to a
            // display driven by the other adapter. Re-running the whole selection would mean a new
            // VkDevice, and every object the device holds would go with it.
            VkBool32 present = VK_FALSE;
            if (g.iapi.GetPhysicalDeviceSurfaceSupportKHR(g.physical, g.presentFamily, g.surface,
                                                          &present) != VK_SUCCESS
                || !present)
            {
                wxl::ScriptMgr::Log(WXL_LOG_WARN,
                                    "vulkan: queue family %u cannot present to the new window; the "
                                    "surface stands but no swapchain is built. The chosen adapter is "
                                    "no longer the one driving this window.",
                                    g.presentFamily);
                return;
            }
        }
        else
        {
            DropSwapchainObject();
        }

        if (win.width == 0 || win.height == 0)
        {
            wxl::ScriptMgr::Log(WXL_LOG_DEBUG,
                                "vulkan: the window is zero-size; the swapchain stays down until the "
                                "next rebuild");
            return;
        }
        CreateSwapchain(win);
    }

    void LogValidationSummary()
    {
        AcquireSRWLockShared(&g_validation.lock);
        if (g_validation.idCount || g_validation.untracked)
        {
            wxl::ScriptMgr::Log(g_validation.errors ? WXL_LOG_WARN : WXL_LOG_INFO,
                                "vulkan: validation: %llu error(s), %llu warning(s) over %u message id(s)%s",
                                static_cast<unsigned long long>(g_validation.errors),
                                static_cast<unsigned long long>(g_validation.warnings), g_validation.idCount,
                                g_validation.untracked ? " (more ids than the table holds, all logged)" : "");
            for (uint32_t i = 0; i < g_validation.idCount; ++i)
            {
                const ValidationId& e = g_validation.ids[i];
                if (e.count <= kValidationRepeats) continue;
                wxl::ScriptMgr::Log(e.level == WXL_LOG_ERROR ? WXL_LOG_WARN : WXL_LOG_INFO,
                                    "vulkan: validation: [%s] (id 0x%08X) %llu time(s), the first %u logged",
                                    e.name, static_cast<unsigned>(e.id), static_cast<unsigned long long>(e.count),
                                    kValidationRepeats);
            }
        }
        ReleaseSRWLockShared(&g_validation.lock);
    }

    void Teardown()
    {
        const bool hadVulkan = g.instance || g.device || g.surface != VK_NULL_HANDLE;

        // Reverse order of creation, and each step guarded on its own handle so Teardown() is also
        // the unwinder for a half-finished Bring().
        if (g.device)
        {
            if (g.dapi.DeviceWaitIdle) g.dapi.DeviceWaitIdle(g.device);
            DropSwapchainObject();
            // Unconditionally, even though DropSwapchainObject has already bumped it when there was
            // a swapchain to drop. The generation's contract is "anything you derived from the
            // swapchain is stale", and destroying the device makes that true whether a swapchain
            // existed or not. A double bump costs one rebuild that was going to happen anyway;
            // a missing one costs a renderer holding handles on a destroyed device.
            ++g.swapGen;
            if (g.dapi.DestroyDevice) g.dapi.DestroyDevice(g.device, nullptr);
            g.device        = nullptr;
            g.graphicsQueue = nullptr;
            g.presentQueue  = nullptr;
            g.physical      = nullptr;
        }
        else
        {
            // No device means no swapchain either, but clear the handle rather than trust that,
            // and bump the generation anyway: its contract is "anything you derived from the
            // swapchain is stale", and that is true here whether or not a swapchain ever existed.
            g.swapchain = VK_NULL_HANDLE;
            ++g.swapGen;
        }

        DropSurface();

        if (g.messenger != VK_NULL_HANDLE && g.iapi.DestroyDebugUtilsMessengerEXT)
        {
            g.iapi.DestroyDebugUtilsMessengerEXT(g.instance, g.messenger, nullptr);
            g.messenger = VK_NULL_HANDLE;
        }

        if (g.instance && g.iapi.DestroyInstance)
        {
            g.iapi.DestroyInstance(g.instance, nullptr);
            g.instance = nullptr;
        }

        // vulkan-1.dll is deliberately left loaded. The one path that reaches Teardown() is a
        // half-finished bring-up, where the ICD may not have initialised at all; FreeLibrary there
        // reclaims little and risks the loader's own detach-order hazards.
        if (hadVulkan) wxl::ScriptMgr::Log(WXL_LOG_DEBUG, "vulkan: torn down");
    }

    bool Live() { return g.instance != nullptr && g.device != nullptr; }

    void SetVsync(bool on) { g.vsync = on; }

    const Handles* Get()
    {
        if (!g.instance || !g.device) return nullptr;

        // Refilled on every call rather than kept in step with the dozen sites that assign to State.
        // The copy is thirteen words on a path that runs once a frame; a cached view that somebody
        // forgot to refresh after a resize is a bug that shows up once an hour and looks like a
        // driver problem.
        //
        // One static, handed out by pointer, and therefore single-threaded by contract: every caller
        // is on the client's render thread, the thread the device slots run on. A caller that wants
        // these from another thread must copy the struct, not keep the pointer.
        static Handles view;
        view.instance       = g.instance;
        view.physical       = g.physical;
        view.device         = g.device;
        view.graphicsQueue  = g.graphicsQueue;
        view.presentQueue   = g.presentQueue;
        view.graphicsFamily = g.graphicsFamily;
        view.presentFamily  = g.presentFamily;
        view.swapchain      = g.swapchain;
        view.format         = g.surfaceFormat.format;
        view.extent         = g.extent;
        view.imageCount     = g.imageCount;
        view.swapchainUsage = g.usage;
        view.textureCompressionBC = g.bc;
        view.largePoints    = g.largePoints;
        view.samplerAnisotropy = g.anisotropy;
        view.occlusionQueryPrecise = g.preciseOcclusion;
        view.independentBlend      = g.independentBlend;
        view.iapi           = &g.iapi;
        view.dapi           = &g.dapi;
        return &view;
    }

    uint32_t SwapchainGeneration() { return g.swapGen; }
}
