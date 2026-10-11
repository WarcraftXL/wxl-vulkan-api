// wxl-vulkan-api: what the host link's files share.
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

#include "wxl/VkHostLinkApi.h"

#include <cstdint>

namespace wxlvk::host
{
    /// The entry points the link uses, resolved from the live device on first use.
    struct Functions
    {
        PFN_vkCreateImage                   createImage                   = nullptr;
        PFN_vkDestroyImage                  destroyImage                  = nullptr;
        PFN_vkGetImageMemoryRequirements    getImageMemoryRequirements    = nullptr;
        PFN_vkAllocateMemory                allocateMemory                = nullptr;
        PFN_vkFreeMemory                    freeMemory                    = nullptr;
        PFN_vkBindImageMemory               bindImageMemory               = nullptr;
        PFN_vkCreateSemaphore               createSemaphore               = nullptr;
        PFN_vkDestroySemaphore              destroySemaphore              = nullptr;
        PFN_vkImportSemaphoreWin32HandleKHR importSemaphoreWin32HandleKHR = nullptr;
        PFN_vkCmdPipelineBarrier            cmdPipelineBarrier            = nullptr;
        PFN_vkCmdBlitImage                  cmdBlitImage                  = nullptr;
        PFN_vkCmdCopyImage                  cmdCopyImage                  = nullptr;
        PFN_vkGetPhysicalDeviceProperties2KHR getPhysicalDeviceProperties2KHR = nullptr;
    };

    /// One imported generation.
    struct Imported
    {
        uint32_t       generation = 0;
        uint32_t       slots      = 0;
        VkImage        images[WXL_VK_SHARED_SLOTS]{};
        VkDeviceMemory memory[WXL_VK_SHARED_SLOTS]{};
        VkFormat       format = VK_FORMAT_UNDEFINED;
        VkExtent2D     extent{};
        VkSemaphore    hostTimeline   = VK_NULL_HANDLE;
        VkSemaphore    clientTimeline = VK_NULL_HANDLE;
        uint64_t       lastSerial = 0;   ///< the last frame that named any of it
    };

    struct State
    {
        bool      interop = false;   ///< every interop extension was enabled at device creation
        bool      loaded  = false;
        Functions fn;
        Imported  live;
        Imported  retiring[2];       ///< generations waiting for their last frame to complete
        WXL_VkHostLinkCallbacks callbacks{};
        bool            pending = false;   ///< a composite is recorded and its submit not yet made
        WXL_VkHostFrame pendingFrame{};
        VkSemaphore     pendingHost   = VK_NULL_HANDLE;   ///< the recorded composite's timelines: they
        VkSemaphore     pendingClient = VK_NULL_HANDLE;   ///< outlive a release until its frame completes
        WXL_VkHostLinkStats stats{};
    };

    /// The link's one state; render thread only.
    State& G();

    /// Resolves Functions from the live device. False without a device or without the interop extensions.
    bool Load();

    /// Destroys every retiring generation whose last frame completed.
    void CollectRetired();
}
