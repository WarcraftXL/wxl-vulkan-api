// wxl-vulkan-api: the wxl.vk.hostlink table, and the host link's Install.
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

#include "host/Compose.hpp"
#include "host/Import.hpp"
#include "host/Settings.hpp"
#include "host/State.hpp"
#include "native/Present.hpp"
#include "vulkan/Context.hpp"

#include "wxl/Script.hpp"

#include <cstring>

namespace wxlvk::host
{
    namespace
    {
        namespace context = wxlvk::vk::context;
        namespace present = wxlvk::native::present;

        int __cdecl GetAdapterIds(WXL_VkAdapterIds* out)
        {
            if (!out || !Load()) return 0;
            const context::Handles*         h = context::Get();
            VkPhysicalDeviceIDPropertiesKHR id{};
            id.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES_KHR;
            VkPhysicalDeviceProperties2KHR props{};
            props.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2_KHR;
            props.pNext = &id;
            G().fn.getPhysicalDeviceProperties2KHR(h->physical, &props);
            *out = {};
            out->structSize = sizeof *out;
            std::memcpy(out->deviceUUID, id.deviceUUID, 16);
            std::memcpy(out->driverUUID, id.driverUUID, 16);
            std::memcpy(out->deviceLUID, id.deviceLUID, 8);
            out->luidValid     = id.deviceLUIDValid;
            out->vendorID      = props.properties.vendorID;
            out->deviceID      = props.properties.deviceID;
            out->driverVersion = props.properties.driverVersion;
            out->apiVersion    = props.properties.apiVersion;
            out->interop       = WXL_VK_INTEROP_MEMORY_WIN32 | WXL_VK_INTEROP_SEMAPHORE_WIN32 | WXL_VK_INTEROP_TIMELINE
                                 | WXL_VK_INTEROP_IDS;
            std::memcpy(out->name, props.properties.deviceName, sizeof out->name - 1);
            return 1;
        }

        int __cdecl GetFrameExtent(uint32_t* width, uint32_t* height)
        {
            VkFormat   format{};
            VkExtent2D extent{};
            if (!width || !height || !present::BackBufferInfo(format, extent)) return 0;
            *width  = extent.width;
            *height = extent.height;
            return extent.width && extent.height;
        }

        int __cdecl Import(const WXL_VkSharedWorld* world, char* why, uint32_t cap)
        {
            return world && ImportWorld(*world, why, cap) ? 1 : 0;
        }

        void __cdecl Release() { ReleaseWorld(); }

        void __cdecl SetCallbacks(const WXL_VkHostLinkCallbacks* callbacks)
        {
            G().callbacks = callbacks ? *callbacks : WXL_VkHostLinkCallbacks{};
        }

        void __cdecl GetStats(WXL_VkHostLinkStats* out)
        {
            if (!out || out->structSize < sizeof *out) return;
            WXL_VkHostLinkStats s = G().stats;
            s.structSize          = sizeof s;
            s.completedSerial     = present::Completed();
            *out                  = s;
        }

        const WXL_VkHostLinkApi g_api = {
            sizeof(WXL_VkHostLinkApi), WXL_VK_HOST_LINK_API_VERSION, &GetAdapterIds, &GetFrameExtent, &Import,
            &Release, &SetCallbacks, &GetStats,
        };
    }

    void Install()
    {
        if (!settings::Wanted()) return;
        const WXL_Api* api = wxl::ScriptMgr::Api();
        if (!api) return;
        api->PublishInterface(WXL_VK_HOST_LINK_API_NAME, WXL_VK_HOST_LINK_API_VERSION, const_cast<WXL_VkHostLinkApi*>(&g_api));
        const bool seam = AttachCompose();
        wxl::ScriptMgr::Log(seam ? WXL_LOG_INFO : WXL_LOG_WARN, "host: %s published%s", WXL_VK_HOST_LINK_API_NAME,
                            seam ? "; the world image is composed at Gx.WorldOnRender's return"
                                 : ", but Gx.WorldOnRender could not be attached: nothing will be composed");
    }
}
