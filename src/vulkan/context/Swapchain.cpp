// The swapchain: its format, its present mode, its images, its rebuild.
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
    // --- swapchain --------------------------------------------------------------------------------

    /**
     * Picks the surface format, and says why.
     *
     * Preference: B8G8R8A8_UNORM with SRGB_NONLINEAR, then R8G8B8A8_UNORM, then whatever the surface
     * offered first.
     *
     * B8G8R8A8 first because it is what the client already produces. The engine's back buffer is a
     * CGxFormat colour format of X8R8G8B8 or A8R8G8B8 (EGxFormat 1 and 2, reference doc 9.5), and a
     * D3D `A8R8G8B8` is blue-green-red-alpha in memory -- byte for byte a Vulkan B8G8R8A8. Choosing
     * the matching order makes the back buffer's present a copy rather than a swizzle, and the back
     * buffer's own format (the swapchain's) the engine's.
     *
     * UNORM rather than SRGB for the same kind of reason: the engine applies gamma through the ramp
     * at device +0x354 (reference doc 9.6) and its shaders write values that are already encoded, so
     * an _SRGB view would encode them a second time.
     *
     * @param vector formats : what vkGetPhysicalDeviceSurfaceFormatsKHR returned
     * @param string& why : receives a literal explaining the pick
     * @return VkSurfaceFormatKHR format
     */
    VkSurfaceFormatKHR PickFormat(const std::vector<VkSurfaceFormatKHR>& formats, const char*& why)
    {
        // The legacy "no preferred format" convention: a single entry of VK_FORMAT_UNDEFINED means
        // the surface accepts anything. Rare now, but cheap to honour and a crash to ignore.
        if (formats.size() == 1 && formats[0].format == VK_FORMAT_UNDEFINED)
        {
            why = "the surface has no preferred format, so our own first choice was used";
            return { VK_FORMAT_B8G8R8A8_UNORM, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR };
        }

        for (const VkSurfaceFormatKHR& f : formats)
            if (f.format == VK_FORMAT_B8G8R8A8_UNORM
                && f.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR)
            {
                why = "it matches the engine's A8R8G8B8 back buffer byte for byte, and UNORM leaves "
                      "the engine's own gamma ramp as the only encoder";
                return f;
            }

        for (const VkSurfaceFormatKHR& f : formats)
            if (f.format == VK_FORMAT_R8G8B8A8_UNORM
                && f.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR)
            {
                why = "B8G8R8A8_UNORM was not offered; this is the same 8-bit UNORM at the cost of a "
                      "channel swizzle later";
                return f;
            }

        why = "none of our preferred formats was offered, so the surface's first entry was taken";
        return formats[0];
    }

    /**
     * Creates the swapchain on the existing surface.
     *
     * @param Found& win : the located window, for the fallback extent
     * @return bool ok : false on a real failure; a zero extent is reported as false with a log line
     *         that says it is transient
     */
    bool CreateSwapchain(const window::Found& win)
    {
        VkSurfaceCapabilitiesKHR caps = {};
        VkResult r = g.iapi.GetPhysicalDeviceSurfaceCapabilitiesKHR(g.physical, g.surface, &caps);
        if (r != VK_SUCCESS)
        {
            wxl::ScriptMgr::Log(WXL_LOG_WARN,
                                "vulkan: vkGetPhysicalDeviceSurfaceCapabilitiesKHR failed: %s",
                                ResultName(r));
            return false;
        }

        uint32_t formatCount = 0;
        g.iapi.GetPhysicalDeviceSurfaceFormatsKHR(g.physical, g.surface, &formatCount, nullptr);
        if (formatCount == 0)
        {
            wxl::ScriptMgr::Log(WXL_LOG_WARN, "vulkan: the surface offers no format; declining the "
                                              "swapchain");
            return false;
        }
        std::vector<VkSurfaceFormatKHR> formats(formatCount);
        g.iapi.GetPhysicalDeviceSurfaceFormatsKHR(g.physical, g.surface, &formatCount,
                                                  formats.data());
        formats.resize(formatCount);

        uint32_t modeCount = 0;
        g.iapi.GetPhysicalDeviceSurfacePresentModesKHR(g.physical, g.surface, &modeCount, nullptr);
        std::vector<VkPresentModeKHR> modes(modeCount);
        if (modeCount)
            g.iapi.GetPhysicalDeviceSurfacePresentModesKHR(g.physical, g.surface, &modeCount,
                                                           modes.data());
        modes.resize(modeCount);

        // --- extent ---
        //
        // currentExtent of 0xFFFFFFFF is the surface saying "the swapchain's size decides the
        // window's", which on Win32 does not happen for an ordinary HWND -- but it is the documented
        // signal and the fallback has to exist. Otherwise currentExtent is authoritative and must be
        // used verbatim: choosing anything else on Windows yields VK_ERROR_OUT_OF_DATE_KHR straight
        // away.
        VkExtent2D extent = caps.currentExtent;
        const char* extentFrom = "the surface's currentExtent";
        if (extent.width == 0xFFFFFFFFu || extent.height == 0xFFFFFFFFu)
        {
            extentFrom = "the window's client rect, clamped to the surface's limits";
            extent.width  = win.width;
            extent.height = win.height;
            if (extent.width  < caps.minImageExtent.width)  extent.width  = caps.minImageExtent.width;
            if (extent.height < caps.minImageExtent.height) extent.height = caps.minImageExtent.height;
            if (extent.width  > caps.maxImageExtent.width)  extent.width  = caps.maxImageExtent.width;
            if (extent.height > caps.maxImageExtent.height) extent.height = caps.maxImageExtent.height;
        }

        if (extent.width == 0 || extent.height == 0)
        {
            // Not a failure. A minimised window reports a 0x0 client area, and vkCreateSwapchainKHR
            // with a zero extent is invalid usage, so declining is correct. Bring()'s own window
            // check catches this before anything is created; reaching it here means the surface
            // reported a zero currentExtent against a non-empty client rect, a mid-resize race. The
            // next rebuild (Rebind()) retries.
            wxl::ScriptMgr::Log(WXL_LOG_DEBUG,
                                "vulkan: the surface's extent is 0x0 (minimised or mid-resize); no "
                                "swapchain this time, the next rebuild retries");
            return false;
        }

        // --- image count ---
        //
        // minImageCount + 1. At exactly minImageCount the CPU cannot start a frame until the
        // compositor has released the only spare image, so the two serialise and the frame time
        // becomes the sum of CPU and GPU instead of the larger of the two. One extra image buys that
        // overlap back.
        uint32_t imageCount = caps.minImageCount + 1;
        if (caps.maxImageCount != 0 && imageCount > caps.maxImageCount) imageCount = caps.maxImageCount;

        // --- present mode ---
        //
        // The engine's own vsync setting decides, as it decides D3D's presentation interval. On:
        // FIFO, which the spec guarantees for every presentable surface. Off: IMMEDIATE,
        // D3DPRESENT_INTERVAL_IMMEDIATE's equivalent (tearing allowed); MAILBOX when the surface
        // lacks it (no tearing, no cap, newest frame wins); FIFO when it offers neither.
        VkPresentModeKHR presentMode = VK_PRESENT_MODE_FIFO_KHR;
        const char* modeWhy = "the engine's format has vsync on";
        bool fifoOffered = false;
        for (VkPresentModeKHR m : modes) if (m == VK_PRESENT_MODE_FIFO_KHR) fifoOffered = true;
        if (!fifoOffered && !modes.empty())
        {
            // Should be unreachable: a surface that reports no FIFO is out of spec. Handled anyway,
            // because declining here would be worse than using what the driver does offer.
            presentMode = modes[0];
            modeWhy     = "the surface did not offer FIFO, which is out of spec; its first mode was "
                          "taken instead";
        }

        if (!g.vsync)
        {
            bool immediate = false, mailbox = false;
            for (VkPresentModeKHR m : modes)
            {
                if (m == VK_PRESENT_MODE_IMMEDIATE_KHR) immediate = true;
                if (m == VK_PRESENT_MODE_MAILBOX_KHR)   mailbox   = true;
            }
            if (immediate)
            {
                presentMode = VK_PRESENT_MODE_IMMEDIATE_KHR;
                modeWhy     = "the engine's format has vsync off (IMMEDIATE, as D3D's immediate "
                              "presentation interval)";
            }
            else if (mailbox)
            {
                presentMode = VK_PRESENT_MODE_MAILBOX_KHR;
                modeWhy     = "the engine's format has vsync off and the surface offers no "
                              "IMMEDIATE (MAILBOX, uncapped without tearing)";
            }
            else
                modeWhy = "the engine's format has vsync off but the surface offers only FIFO";
        }

        const char* formatWhy = "";
        const VkSurfaceFormatKHR format = PickFormat(formats, formatWhy);

        // --- usage, transform, alpha ---
        if ((caps.supportedUsageFlags & VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT) == 0)
        {
            wxl::ScriptMgr::Log(WXL_LOG_WARN,
                                "vulkan: the surface does not support COLOR_ATTACHMENT usage "
                                "(flags 0x%08X); declining the swapchain",
                                caps.supportedUsageFlags);
            return false;
        }

        if (caps.supportedCompositeAlpha == 0)
        {
            wxl::ScriptMgr::Log(WXL_LOG_WARN, "vulkan: the surface supports no composite alpha "
                                              "mode; declining the swapchain");
            return false;
        }
        VkCompositeAlphaFlagBitsKHR alpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
        if ((caps.supportedCompositeAlpha & VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR) == 0)
            // The lowest set bit. OPAQUE is what a game window wants and is what every Windows
            // driver reports, so this is a correctness guard rather than an expected path.
            alpha = static_cast<VkCompositeAlphaFlagBitsKHR>(
                caps.supportedCompositeAlpha & (~caps.supportedCompositeAlpha + 1u));

        const uint32_t families[2] = { g.graphicsFamily, g.presentFamily };
        const bool concurrent = g.graphicsFamily != g.presentFamily;

        VkSwapchainCreateInfoKHR info = {};
        info.sType            = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
        info.surface          = g.surface;
        info.minImageCount    = imageCount;
        info.imageFormat      = format.format;
        info.imageColorSpace  = format.colorSpace;
        info.imageExtent      = extent;
        info.imageArrayLayers = 1;
        // COLOR_ATTACHMENT is what the present pass and the direct path render through and is
        // required of every surface. TRANSFER_DST is what the back buffer's copy writes and is
        // optional, so it is requested only when offered and the frame reads the result off
        // Handles::swapchainUsage before it copies. Asking for a usage the surface does not support
        // fails the create; copying into an image made without the bit is invalid usage that drivers
        // perform anyway, and the second is the worse failure because it is the one nobody sees.
        VkImageUsageFlags usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
        if (caps.supportedUsageFlags & VK_IMAGE_USAGE_TRANSFER_DST_BIT)
            usage |= VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        else
            wxl::ScriptMgr::Log(WXL_LOG_WARN,
                                "vulkan: the surface does not offer TRANSFER_DST usage (flags "
                                "0x%08X); the swapchain is created without it and the frame renders "
                                "straight into its images", caps.supportedUsageFlags);
        info.imageUsage       = usage;
        info.imageSharingMode = concurrent ? VK_SHARING_MODE_CONCURRENT : VK_SHARING_MODE_EXCLUSIVE;
        info.queueFamilyIndexCount = concurrent ? 2u : 0u;
        info.pQueueFamilyIndices   = concurrent ? families : nullptr;
        // currentTransform, not IDENTITY: passing a transform the surface does not support is
        // invalid, and on a rotated display currentTransform is the only one guaranteed to be.
        info.preTransform     = caps.currentTransform;
        info.compositeAlpha   = alpha;
        info.presentMode      = presentMode;
        info.clipped          = VK_TRUE;
        info.oldSwapchain     = VK_NULL_HANDLE;

        r = g.dapi.CreateSwapchainKHR(g.device, &info, nullptr, &g.swapchain);
        if (r != VK_SUCCESS)
        {
            wxl::ScriptMgr::Log(WXL_LOG_WARN, "vulkan: vkCreateSwapchainKHR failed: %s",
                                ResultName(r));
            g.swapchain = VK_NULL_HANDLE;
            return false;
        }

        // The driver may hand out more images than were asked for; the real number is read back
        // rather than assumed.
        uint32_t actual = 0;
        g.dapi.GetSwapchainImagesKHR(g.device, g.swapchain, &actual, nullptr);

        g.surfaceFormat = format;
        g.presentMode   = presentMode;
        g.extent        = extent;
        g.imageCount    = actual ? actual : imageCount;
        g.usage         = usage;
        ++g.swapGen;

        wxl::ScriptMgr::Log(WXL_LOG_DEBUG,
                            "vulkan: swapchain created %ux%u, %s / %s, %s, %u image(s) "
                            "(asked %u, surface allows %u..%u), sharing %s; extent from %s (engine "
                            "curWindow %ux%u); format chosen because %s; present mode chosen because %s",
                            extent.width, extent.height, FormatName(format.format),
                            ColorSpaceName(format.colorSpace), PresentModeName(presentMode),
                            g.imageCount, imageCount, caps.minImageCount, caps.maxImageCount,
                            concurrent ? "CONCURRENT" : "EXCLUSIVE", extentFrom, win.curWidth,
                            win.curHeight, formatWhy, modeWhy);
        return true;
    }

    /// Destroys the swapchain if there is one. Safe to call repeatedly.
    void DropSwapchainObject()
    {
        if (g.swapchain == VK_NULL_HANDLE) return;

        // The spec requires that no work still references the swapchain's images.
        if (g.dapi.DeviceWaitIdle) g.dapi.DeviceWaitIdle(g.device);

        g.dapi.DestroySwapchainKHR(g.device, g.swapchain, nullptr);
        g.swapchain = VK_NULL_HANDLE;
        g.extent    = VkExtent2D{ 0, 0 };
        g.imageCount = 0;
        g.usage      = 0;
        ++g.swapGen;
    }

    /// Destroys the surface if there is one. The swapchain must already be gone.
    void DropSurface()
    {
        if (g.surface == VK_NULL_HANDLE) return;
        g.iapi.DestroySurfaceKHR(g.instance, g.surface, nullptr);
        g.surface = VK_NULL_HANDLE;
        g.hwnd    = nullptr;
    }
}
