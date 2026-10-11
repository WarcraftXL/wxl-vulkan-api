// wxl-vulkan-api: the host link's seams in the device -- the only names the rest of the device sees.
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

#include "vulkan/Loader.hpp"
#include "vulkan/Vk.hpp"

#include <cstdint>
#include <vector>

// src/host/, by responsibility:
//
//   Host         these seams: what Extension.cpp, vulkan/Context.cpp and native/Present.cpp call
//   Settings     WXL_HOST
//   State        what the files below share
//   Extensions   the interop instance and device extensions, enabled only with WXL_HOST=1
//   Import       the shared world's images and timelines into this device, and their retirement
//   Compose      the copy of the host's world image into the back buffer at the end of the world frame
//   Submit       the timeline wait and signal added to the submit that carries a composite
//   Api          the wxl.vk.hostlink table (include/wxl/VkHostLinkApi.h)
//
// With WXL_HOST unset every seam is a no-op: no extension, no hook, no publication, the submits as
// they were.
namespace wxlvk::host
{
    /// Publishes wxl.vk.hostlink and attaches the composite seam. From WXL_Load, native device only.
    void Install();

    /// Instance creation: appends the interop instance extensions the loader offers (WXL_HOST=1 only).
    void ExtendInstance(const std::vector<VkExtensionProperties>& available, std::vector<const char*>& extensions);

    /// What the device is created with: the device's own extensions, plus the interop ones when wanted.
    struct DeviceExtensions
    {
        const char* const* names = nullptr;
        uint32_t           count = 0;
        const void*        pNext = nullptr;   ///< the timeline-semaphore feature, or null
    };

    /**
     * Device creation: the device's list unchanged (WXL_HOST off, or an extension missing: logged),
     * or that list plus every interop device extension and the timeline feature.
     */
    DeviceExtensions ExtendDevice(const vk::loader::Instance& iapi, VkInstance instance, VkPhysicalDevice physical,
                                  const char* const* base, uint32_t baseCount);

    /// Storage the submit decoration points into; lives on the caller's stack until the submit returns.
    struct SubmitExtras
    {
        VkSemaphore          waits[2]{};
        VkPipelineStageFlags stages[2]{};
        uint64_t             waitValues[2]{};
        VkSemaphore          signals[2]{};
        uint64_t             signalValues[2]{};
        VkTimelineSemaphoreSubmitInfoKHR timeline{};
        uint32_t             slot = 0;
        uint64_t             waitValue = 0;
        uint64_t             signalValue = 0;
        bool                 used = false;
    };

    /// Adds the pending composite's wait and signal to `submit` (nothing when none is pending).
    void DecorateSubmit(VkSubmitInfo& submit, SubmitExtras& extras);

    /// Tells the client whether the decorated submit went through.
    void AfterSubmit(const SubmitExtras& extras, bool ok);
}
