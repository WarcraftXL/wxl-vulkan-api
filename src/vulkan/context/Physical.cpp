// Choosing the physical device and its queue, and what it supports.
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
    // --- physical device --------------------------------------------------------------------------

    /// deviceType as a rank: the only criterion that is a real preference rather than a tie-break.
    unsigned TypeRank(VkPhysicalDeviceType t)
    {
        switch (t)
        {
            case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:   return 4;
            case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU: return 3;
            case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU:    return 2;
            case VK_PHYSICAL_DEVICE_TYPE_CPU:            return 1;
            default:                                     return 0;
        }
    }

    /**
     * Why `a` is preferred over `b`, or null when it is not.
     *
     * A lexicographic comparison on named criteria rather than a single numeric score, for one
     * reason: a score cannot say *which* term decided, and "do not silently take index 0" is only
     * honoured if the log can name the criterion. The order is:
     *
     *  1. device type. A discrete GPU beats an integrated one, and both beat a software device.
     *     This is the criterion that matters on a laptop with switchable graphics, where index 0 is
     *     routinely the integrated part.
     *  2. one queue family covering both graphics and present. Two families mean every future
     *     present needs an ownership transfer or a CONCURRENT swapchain, both of which cost
     *     something for nothing on hardware that does not need them.
     *  3. the higher supported API version.
     *  4. the larger total DEVICE_LOCAL heap, as a proxy for "the bigger GPU" among two of the same
     *     type -- a proxy, not a measurement of anything, which is why it ranks below the three
     *     facts above it.
     *  5. the lower enumeration index, so the choice is deterministic rather than arbitrary when
     *     two adapters are genuinely indistinguishable. Reaching this rule is the only case in which
     *     taking index 0 is correct, and the log says so.
     *
     * @param Candidate& a : the incumbent challenger
     * @param Candidate& b : the current best
     * @return string reason : null when `a` is not better
     */
    const char* WhyBetter(const Candidate& a, const Candidate& b)
    {
        if (TypeRank(a.props.deviceType) != TypeRank(b.props.deviceType))
            return TypeRank(a.props.deviceType) > TypeRank(b.props.deviceType)
                       ? "it is the better device type" : nullptr;
        if (a.unified != b.unified)
            return a.unified ? "one queue family covers graphics and present" : nullptr;
        if (a.props.apiVersion != b.props.apiVersion)
            return a.props.apiVersion > b.props.apiVersion ? "it supports a higher API version"
                                                           : nullptr;
        if (a.deviceLocal != b.deviceLocal)
            return a.deviceLocal > b.deviceLocal ? "it has more device-local memory" : nullptr;
        if (a.index != b.index)
            return a.index < b.index ? "the two are indistinguishable, so the lower index wins"
                                     : nullptr;
        return nullptr;
    }

    /**
     * Fills in one candidate and decides whether it is usable at all.
     *
     * Three hard requirements, each of which is checked rather than assumed:
     *  - VK_KHR_swapchain in its device extension list. Without it there is no swapchain, and a
     *    headless adapter (a compute-only ICD, or a software device) legitimately lacks it.
     *  - a queue family with VK_QUEUE_GRAPHICS_BIT.
     *  - a queue family reporting present support on OUR surface, via
     *    vkGetPhysicalDeviceSurfaceSupportKHR. This is the check that cannot be replaced by
     *    guesswork: on a hybrid laptop the adapter that can render is not always the one wired to
     *    the display the window is on.
     *
     * @param Candidate& c : handle and index already set; everything else filled here
     */
    void Examine(Candidate& c)
    {
        g.iapi.GetPhysicalDeviceProperties(c.handle, &c.props);

        uint32_t extCount = 0;
        g.iapi.EnumerateDeviceExtensionProperties(c.handle, nullptr, &extCount, nullptr);
        std::vector<VkExtensionProperties> exts(extCount);
        if (extCount)
            g.iapi.EnumerateDeviceExtensionProperties(c.handle, nullptr, &extCount, exts.data());
        exts.resize(extCount);
        if (!HasExtension(exts, VK_KHR_SWAPCHAIN_EXTENSION_NAME))
        {
            c.reject = "no " VK_KHR_SWAPCHAIN_EXTENSION_NAME;
            return;
        }

        VkPhysicalDeviceMemoryProperties mem = {};
        g.iapi.GetPhysicalDeviceMemoryProperties(c.handle, &mem);
        for (uint32_t h = 0; h < mem.memoryHeapCount; ++h)
            if (mem.memoryHeaps[h].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT)
                c.deviceLocal += mem.memoryHeaps[h].size;

        uint32_t familyCount = 0;
        g.iapi.GetPhysicalDeviceQueueFamilyProperties(c.handle, &familyCount, nullptr);
        std::vector<VkQueueFamilyProperties> families(familyCount);
        if (familyCount)
            g.iapi.GetPhysicalDeviceQueueFamilyProperties(c.handle, &familyCount, families.data());
        families.resize(familyCount);

        for (uint32_t f = 0; f < familyCount; ++f)
        {
            if (families[f].queueCount == 0) continue;

            const bool graphics = (families[f].queueFlags & VK_QUEUE_GRAPHICS_BIT) != 0;

            VkBool32 present = VK_FALSE;
            if (g.iapi.GetPhysicalDeviceSurfaceSupportKHR(c.handle, f, g.surface, &present)
                != VK_SUCCESS)
                present = VK_FALSE;

            // A family that does both ends the search: that is the preferred shape, and taking it
            // the moment it appears is also what makes `unified` meaningful.
            if (graphics && present)
            {
                c.graphicsFamily = f;
                c.presentFamily  = f;
                c.unified        = true;
                break;
            }
            if (graphics && c.graphicsFamily == UINT32_MAX) c.graphicsFamily = f;
            if (present  && c.presentFamily  == UINT32_MAX) c.presentFamily  = f;
        }

        if (c.graphicsFamily == UINT32_MAX) { c.reject = "no graphics queue family"; return; }
        if (c.presentFamily  == UINT32_MAX) { c.reject = "no queue family can present to our window"; }
    }

    /**
     * Enumerates the adapters, logs every one, and picks the best by the rules in WhyBetter.
     *
     * @return bool ok : false when nothing usable was found
     */
    bool SelectPhysicalDevice()
    {
        uint32_t count = 0;
        VkResult r = g.iapi.EnumeratePhysicalDevices(g.instance, &count, nullptr);
        if (r != VK_SUCCESS && r != VK_INCOMPLETE)
        {
            wxl::ScriptMgr::Log(WXL_LOG_WARN, "vulkan: vkEnumeratePhysicalDevices failed: %s",
                                ResultName(r));
            return false;
        }
        if (count == 0)
        {
            wxl::ScriptMgr::Log(WXL_LOG_WARN,
                                "vulkan: the loader reports no physical device; there is a "
                                "vulkan-1.dll but no usable ICD. Declining.");
            return false;
        }

        std::vector<VkPhysicalDevice> handles(count);
        r = g.iapi.EnumeratePhysicalDevices(g.instance, &count, handles.data());
        if (r != VK_SUCCESS && r != VK_INCOMPLETE)
        {
            wxl::ScriptMgr::Log(WXL_LOG_WARN, "vulkan: vkEnumeratePhysicalDevices failed: %s",
                                ResultName(r));
            return false;
        }
        handles.resize(count);

        std::vector<Candidate> candidates(count);
        int best = -1;
        // Replaced by WhyBetter's answer whenever a later adapter displaces the incumbent, so it
        // always names the criterion that actually decided. The initial wording is what stands when
        // the first usable adapter is never displaced -- which is the common case, and is not the
        // same statement as "it was index 0".
        const char* bestReason = "it was the first usable adapter and nothing beat it";

        for (uint32_t i = 0; i < count; ++i)
        {
            Candidate& c = candidates[i];
            c.handle = handles[i];
            c.index  = i;
            Examine(c);

            char apiText[32];
            char driverText[32];
            if (c.reject)
            {
                wxl::ScriptMgr::Log(WXL_LOG_DEBUG,
                                    "vulkan:   adapter %u '%s' (%s, VID %04x DID %04x) rejected: %s",
                                    i, c.props.deviceName, DeviceTypeName(c.props.deviceType),
                                    c.props.vendorID, c.props.deviceID, c.reject);
                continue;
            }

            wxl::ScriptMgr::Log(WXL_LOG_DEBUG,
                                "vulkan:   adapter %u '%s' (%s, VID %04x DID %04x) api %s driver %s "
                                "device-local %u MiB queues gfx=%u present=%u%s",
                                i, c.props.deviceName, DeviceTypeName(c.props.deviceType),
                                c.props.vendorID, c.props.deviceID,
                                ApiVersionText(c.props.apiVersion, apiText, sizeof apiText),
                                DriverVersionText(c.props.vendorID, c.props.driverVersion,
                                                  driverText, sizeof driverText),
                                footprint::Mib(c.deviceLocal),
                                c.graphicsFamily, c.presentFamily,
                                c.unified ? " (unified)" : "");

            if (best < 0) { best = static_cast<int>(i); continue; }

            if (const char* why = WhyBetter(c, candidates[static_cast<size_t>(best)]))
            {
                best       = static_cast<int>(i);
                bestReason = why;
            }
        }

        if (best < 0)
        {
            wxl::ScriptMgr::Log(WXL_LOG_WARN,
                                "vulkan: %u adapter(s) enumerated, none of them usable; declining",
                                count);
            return false;
        }

        const Candidate& chosen = candidates[static_cast<size_t>(best)];
        g.physical       = chosen.handle;
        g.graphicsFamily = chosen.graphicsFamily;
        g.presentFamily  = chosen.presentFamily;

        char apiText[32];
        char driverText[32];
        wxl::ScriptMgr::Log(WXL_LOG_DEBUG,
                            "vulkan: chose adapter %u '%s' out of %u: %s. %s, api %s, driver %s, "
                            "VID %04x DID %04x",
                            chosen.index, chosen.props.deviceName, count,
                            count == 1 ? "it is the only adapter" : bestReason,
                            DeviceTypeName(chosen.props.deviceType),
                            ApiVersionText(chosen.props.apiVersion, apiText, sizeof apiText),
                            DriverVersionText(chosen.props.vendorID, chosen.props.driverVersion,
                                              driverText, sizeof driverText),
                            chosen.props.vendorID, chosen.props.deviceID);
        return true;
    }
}
