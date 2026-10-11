// The logical device, its features and its queue.
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
    // --- logical device ---------------------------------------------------------------------------

    /**
     * Creates the logical device with one queue per distinct family, and resolves the device level.
     *
     * Three features are requested, each only when the adapter reports it: textureCompressionBC,
     * robustBufferAccess and largePoints. Nothing else, because an unused enabled feature is an
     * unused cost -- some drivers allocate differently for robustBufferAccess and the like -- and
     * because asking for a feature the adapter lacks fails the create.
     *
     * No device layers either: enabledLayerCount on VkDeviceCreateInfo has been deprecated and
     * ignored since Vulkan 1.0.x, and instance layers already cover the device.
     *
     * @return bool ok
     */
    bool CreateDevice()
    {
        const float priority = 1.0f;

        VkDeviceQueueCreateInfo queues[2] = {};
        uint32_t queueCount = 0;

        queues[queueCount].sType            = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        queues[queueCount].queueFamilyIndex = g.graphicsFamily;
        queues[queueCount].queueCount       = 1;
        queues[queueCount].pQueuePriorities = &priority;
        ++queueCount;

        // One VkDeviceQueueCreateInfo per family, never two for the same family: the spec forbids a
        // duplicate queueFamilyIndex in the array, and the unified case is exactly when that would
        // happen.
        if (g.presentFamily != g.graphicsFamily)
        {
            queues[queueCount].sType            = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
            queues[queueCount].queueFamilyIndex = g.presentFamily;
            queues[queueCount].queueCount       = 1;
            queues[queueCount].pQueuePriorities = &priority;
            ++queueCount;
        }

        // The swapchain, then what the optional capabilities add by name (Caps.cpp PlanCaps: the memory budget,
        // gl_Layer from the vertex stage, FP16 on a 1.1 device); WXL_HOST appends its interop extensions after them.
        constexpr uint32_t kMaxBase = 6;
        const char* extensions[kMaxBase] = { VK_KHR_SWAPCHAIN_EXTENSION_NAME };
        uint32_t    extensionCount = 1;
        // Features: ask only for what the adapter reports, because asking for anything else fails
        // the create. textureCompressionBC: nearly every texture the client loads is a
        // DXT-compressed BLP, and uploading those as BC1/BC2/BC3 is only legal with the feature on.
        // Every desktop GPU has it; the query exists so that a machine without it gets decoded
        // RGBA8 images and a log line instead of a device that fails to come up.
        VkPhysicalDeviceFeatures features = {};
        {
            VkPhysicalDeviceFeatures have = {};
            g.iapi.GetPhysicalDeviceFeatures(g.physical, &have);
            features.textureCompressionBC = have.textureCompressionBC;
            g.bc = have.textureCompressionBC == VK_TRUE;
            // robustBufferAccess: an out-of-bounds buffer read returns zero (or in-buffer bytes)
            // instead of faulting the device. The translated shaders index the bone palette with
            // a register the engine fills (`regs.v[31 + a0.x]`), a vertex buffer is bound whole and
            // addressed by the engine's own offsets: every one of those is a read this feature turns
            // from a lost device into a wrong pixel. Every desktop driver supports it; the cost is a
            // bounds check the hardware does anyway.
            features.robustBufferAccess = have.robustBufferAccess;
            // largePoints: the weather sprites (`sand`, `snowpoint`) are point lists
            // whose vertex module writes gl_PointSize from the engine's oPts, and without the
            // feature every size above 1.0 is clamped to one pixel. Same pattern as the two
            // above: asked for only when reported, so a device without it still comes up and
            // the snow is one pixel wide rather than absent. Vulkan clamps to pointSizeRange,
            // which src/vulkan/Pipelines.cpp logs at Init in place of D3D's POINTSIZE_MIN/MAX.
            features.largePoints = have.largePoints;
            g.largePoints        = have.largePoints == VK_TRUE;
            // samplerAnisotropy: the engine's filter 5 is D3D's ANISOTROPIC min/mag filter with the
            // texture's own MAXANISOTROPY (CGxTexFlags bits 9-13, cgxdevice 2.7 / 4.7); without the
            // feature such a sampler is trilinear, as it is on an adapter without the D3D cap.
            features.samplerAnisotropy = have.samplerAnisotropy;
            g.anisotropy               = have.samplerAnisotropy == VK_TRUE;
            // occlusionQueryPrecise: D3D9's occlusion query returns the exact count of samples that
            // passed, and its one user (the sun / moon glare, DayNight::DNGlare) divides that count
            // by the quad's projected area to scale the glare. A query begun without
            // VK_QUERY_CONTROL_PRECISE_BIT may return any non-zero value for "some passed", which
            // reads as an almost hidden sun.
            features.occlusionQueryPrecise = have.occlusionQueryPrecise;
            g.preciseOcclusion             = have.occlusionQueryPrecise == VK_TRUE;
            // independentBlend: the G-buffer pass class blends the frame with the draw's own mode and masks
            // or zeroes its three other attachments, which is per-attachment blend state. Without the
            // feature the device makes no G-buffer (native/present/Images.cpp) and every pipeline keeps
            // one blend state for all its attachments.
            features.independentBlend = have.independentBlend;
            g.independentBlend        = have.independentBlend == VK_TRUE;
            if (!g.bc)
                wxl::ScriptMgr::Log(WXL_LOG_WARN,
                                    "vulkan: the adapter has no textureCompressionBC; DXT textures "
                                    "are decoded to RGBA8, four to eight times larger");
            if (!g.preciseOcclusion)
                wxl::ScriptMgr::Log(WXL_LOG_WARN,
                                    "vulkan: the adapter has no occlusionQueryPrecise; occlusion queries "
                                    "may count any passing draw as one sample (the sun glare dims)");
            if (!g.largePoints)
                wxl::ScriptMgr::Log(WXL_LOG_WARN,
                                    "vulkan: the adapter has no largePoints; point sprites (the "
                                    "weather's sand and snow) will draw one pixel wide");
        }

        // The optional capabilities (WXL_VK_CAPS, Caps.cpp): every optional feature the adapter reports, the
        // extension features chained, their extensions named. Nothing with the key off.
        const VkPhysicalDeviceFeatures plainFeatures = features;
        CapsChain capsChain;
        PlanCaps(features, capsChain, nullptr, extensions, extensionCount, kMaxBase);
        const bool richer = capsChain.head != nullptr || extensionCount > 1
                            || std::memcmp(&plainFeatures, &features, sizeof features) != 0;

        // WXL_HOST=1 adds the host link's interop extensions and the timeline feature (src/host/), its pNext
        // chained after the capabilities'; otherwise this is the swapchain extension (and the capabilities') alone.
        const wxlvk::host::DeviceExtensions enabled =
            wxlvk::host::ExtendDevice(g.iapi, g.instance, g.physical, extensions, extensionCount);
        if (enabled.pNext)
        {
            // The capabilities' chain ends on the host's struct (theirs were planned with a null tail).
            VkBaseOutStructure* last = static_cast<VkBaseOutStructure*>(capsChain.head);
            while (last && last->pNext) last = last->pNext;
            if (last) last->pNext = static_cast<VkBaseOutStructure*>(const_cast<void*>(enabled.pNext));
            else capsChain.head = const_cast<void*>(enabled.pNext);
        }
        VkDeviceCreateInfo info = {};
        info.sType                   = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        info.pNext                   = capsChain.head;
        info.queueCreateInfoCount    = queueCount;
        info.pQueueCreateInfos       = queues;
        info.enabledExtensionCount   = enabled.count;
        info.ppEnabledExtensionNames = enabled.names;
        info.pEnabledFeatures        = &features;

        VkResult r = g.iapi.CreateDevice(g.physical, &info, nullptr, &g.device);
        bool chainUsed = richer;
        if (r != VK_SUCCESS && richer)
        {
            // The richer device refused (a driver that reports a feature it will not enable): the device as it was,
            // its six features, the swapchain and the host link's extensions only.
            wxl::ScriptMgr::Log(WXL_LOG_WARN, "vulkan: vkCreateDevice with the optional capabilities failed (%s); "
                                "creating the plain device (set %s=0 to skip the attempt)", ResultName(r), settings::kCapsKey);
            g.device = nullptr;
            const char* plainNames[kMaxBase] = { VK_KHR_SWAPCHAIN_EXTENSION_NAME };
            const wxlvk::host::DeviceExtensions plain =
                wxlvk::host::ExtendDevice(g.iapi, g.instance, g.physical, plainNames, 1);
            features = plainFeatures;
            info.pNext                   = plain.pNext;
            info.enabledExtensionCount   = plain.count;
            info.ppEnabledExtensionNames = plain.names;
            r = g.iapi.CreateDevice(g.physical, &info, nullptr, &g.device);
            chainUsed = false;
        }
        if (r != VK_SUCCESS)
        {
            wxl::ScriptMgr::Log(WXL_LOG_WARN, "vulkan: vkCreateDevice failed: %s", ResultName(r));
            g.device = nullptr;
            return false;
        }
        CapsCreated(chainUsed);

        if (!loader::LoadDevice(g.iapi, g.device, g.dapi))
        {
            if (g.dapi.DestroyDevice) g.dapi.DestroyDevice(g.device, nullptr);
            g.device = nullptr;
            return false;
        }

        g.dapi.GetDeviceQueue(g.device, g.graphicsFamily, 0, &g.graphicsQueue);
        g.dapi.GetDeviceQueue(g.device, g.presentFamily, 0, &g.presentQueue);

        wxl::ScriptMgr::Log(WXL_LOG_DEBUG,
                            "vulkan: logical device created with %u queue(s): graphics family %u"
                            "%s, present family %u%s. Extensions: %u (%s first)",
                            queueCount, g.graphicsFamily,
                            g.graphicsQueue ? "" : " (queue handle null!)",
                            g.presentFamily,
                            g.presentFamily == g.graphicsFamily ? " (same family)" : "",
                            info.enabledExtensionCount, VK_KHR_SWAPCHAIN_EXTENSION_NAME);
        return true;
    }
}
