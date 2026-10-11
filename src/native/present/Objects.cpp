// The device's own objects: command pools and buffers, fences, semaphores.
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

#include "native/present/Internal.hpp"

namespace wxlvk::native::present::detail
{
    // --- device objects -------------------------------------------------------------------------------

    bool EnsureDeviceObjects(const context::Handles& h)
    {
        if (s.pool != VK_NULL_HANDLE && s.device == h.device) return true;
        if (s.pool != VK_NULL_HANDLE)
        {
            // A new VkDevice: the old one is gone, so its objects are forgotten, never destroyed.
            wxl::ScriptMgr::Log(WXL_LOG_WARN, "native: frame: the Vulkan device changed; the frame's "
                                "old objects are forgotten, not destroyed");
            s = State();
        }
        const loader::Device& d = *h.dapi;
        s.device = h.device;
        s.dapi   = h.dapi;

        VkCommandPoolCreateInfo pool = {};
        pool.sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        // RESET_COMMAND_BUFFER so each vkBeginCommandBuffer resets implicitly: the loader table has
        // no vkResetCommandBuffer and needs none with this flag.
        pool.flags            = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        pool.queueFamilyIndex = h.graphicsFamily;
        VkResult r = d.CreateCommandPool(h.device, &pool, nullptr, &s.pool);
        if (r != VK_SUCCESS)
        {
            s.pool = VK_NULL_HANDLE;
            Fail("vkCreateCommandPool", r);
            return false;
        }

        VkCommandBuffer buffers[2 * kFramesInFlight] = {};
        VkCommandBufferAllocateInfo alloc = {};
        alloc.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        alloc.commandPool        = s.pool;
        alloc.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        alloc.commandBufferCount = 2 * kFramesInFlight;
        r = d.AllocateCommandBuffers(h.device, &alloc, buffers);
        if (r != VK_SUCCESS)
        {
            Fail("vkAllocateCommandBuffers", r);
            return false;
        }

        VkSemaphoreCreateInfo sem = {};
        sem.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
        VkFenceCreateInfo fence = {};
        fence.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        fence.flags = VK_FENCE_CREATE_SIGNALED_BIT;   // the first wait on each slot returns at once
        for (uint32_t i = 0; i < kFramesInFlight; ++i)
        {
            s.slots[i].main = buffers[2 * i];
            s.slots[i].pre  = buffers[2 * i + 1];
            r = d.CreateSemaphore(h.device, &sem, nullptr, &s.slots[i].imageAvailable);
            if (r == VK_SUCCESS) r = d.CreateFence(h.device, &fence, nullptr, &s.slots[i].fence);
            if (r != VK_SUCCESS)
            {
                Fail("vkCreateSemaphore / vkCreateFence", r);
                return false;
            }
        }
        return true;
    }
}
