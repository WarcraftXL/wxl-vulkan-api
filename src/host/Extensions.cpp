// wxl-vulkan-api: the interop extensions, enabled only with WXL_HOST=1.
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

#include "host/Host.hpp"

#include "host/Settings.hpp"
#include "host/State.hpp"

#include "wxl/Script.hpp"

#include <cstring>

// The device stays a Vulkan 1.0 device: the interop is the KHR extensions of 1.1/1.2 core, each
// asked for by name, so nothing about the device changes for a user without WXL_HOST. Either every
// extension below is there and all are enabled, or none is and the log says which one is missing --
// the first P0 measurement on the client side.
namespace wxlvk::host
{
    namespace
    {
        constexpr const char* kInstance[] = {
            VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME,
            VK_KHR_EXTERNAL_MEMORY_CAPABILITIES_EXTENSION_NAME,
            VK_KHR_EXTERNAL_SEMAPHORE_CAPABILITIES_EXTENSION_NAME,
        };

        constexpr const char* kDevice[] = {
            VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME,
            VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME,
            VK_KHR_EXTERNAL_SEMAPHORE_EXTENSION_NAME,
            VK_KHR_EXTERNAL_SEMAPHORE_WIN32_EXTENSION_NAME,
            VK_KHR_TIMELINE_SEMAPHORE_EXTENSION_NAME,
            VK_KHR_GET_MEMORY_REQUIREMENTS_2_EXTENSION_NAME,
            VK_KHR_DEDICATED_ALLOCATION_EXTENSION_NAME,
        };

        constexpr uint32_t kMaxDevice = 16;

        bool                                         g_instanceOk = false;
        const char*                                  g_names[kMaxDevice] = {};
        VkPhysicalDeviceTimelineSemaphoreFeaturesKHR g_timeline{};

        bool Has(const std::vector<VkExtensionProperties>& available, const char* name)
        {
            for (const VkExtensionProperties& e : available)
                if (std::strcmp(e.extensionName, name) == 0) return true;
            return false;
        }

        std::vector<VkExtensionProperties> DeviceAvailable(const vk::loader::Instance& iapi, VkPhysicalDevice physical)
        {
            uint32_t count = 0;
            iapi.EnumerateDeviceExtensionProperties(physical, nullptr, &count, nullptr);
            std::vector<VkExtensionProperties> out(count);
            if (count) iapi.EnumerateDeviceExtensionProperties(physical, nullptr, &count, out.data());
            out.resize(count);
            return out;
        }

        bool TimelineSupported(VkInstance instance, VkPhysicalDevice physical)
        {
            const auto features2 = reinterpret_cast<PFN_vkGetPhysicalDeviceFeatures2KHR>(
                vk::loader::Api().GetInstanceProcAddr(instance, "vkGetPhysicalDeviceFeatures2KHR"));
            if (!features2) return false;
            VkPhysicalDeviceTimelineSemaphoreFeaturesKHR timeline{};
            timeline.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES_KHR;
            VkPhysicalDeviceFeatures2KHR all{};
            all.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2_KHR;
            all.pNext = &timeline;
            features2(physical, &all);
            return timeline.timelineSemaphore == VK_TRUE;
        }
    }

    void ExtendInstance(const std::vector<VkExtensionProperties>& available, std::vector<const char*>& extensions)
    {
        if (!settings::Wanted()) return;
        for (const char* name : kInstance)
            if (!Has(available, name))
            {
                wxl::ScriptMgr::Log(WXL_LOG_WARN, "host: the loader has no %s; the host link is off", name);
                return;
            }
        for (const char* name : kInstance) extensions.push_back(name);
        g_instanceOk = true;
    }

    DeviceExtensions ExtendDevice(const vk::loader::Instance& iapi, VkInstance instance, VkPhysicalDevice physical,
                                  const char* const* base, uint32_t baseCount)
    {
        DeviceExtensions out{ base, baseCount, nullptr };
        if (!settings::Wanted() || !g_instanceOk) return out;
        const std::vector<VkExtensionProperties> available = DeviceAvailable(iapi, physical);
        for (const char* name : kDevice)
            if (!Has(available, name))
            {
                wxl::ScriptMgr::Log(WXL_LOG_WARN, "host: the 32-bit driver has no %s; the host link is off", name);
                return out;
            }
        if (!TimelineSupported(instance, physical) || baseCount + std::size(kDevice) > kMaxDevice)
        {
            wxl::ScriptMgr::Log(WXL_LOG_WARN, "host: the 32-bit driver has no timeline semaphores; the host link is off");
            return out;
        }
        uint32_t n = 0;
        for (uint32_t i = 0; i < baseCount; ++i) g_names[n++] = base[i];
        for (const char* name : kDevice) g_names[n++] = name;
        g_timeline                   = {};
        g_timeline.sType             = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES_KHR;
        g_timeline.timelineSemaphore = VK_TRUE;
        G().interop                  = true;
        wxl::ScriptMgr::Log(WXL_LOG_INFO, "host: interop enabled on the native device: external memory and "
                                          "semaphores (Win32), timeline semaphores, dedicated allocations");
        return { g_names, n, &g_timeline };
    }
}
