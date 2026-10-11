// The pipelines' state, and the teardown of its layouts, pools and rings.
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

#include "vulkan/pipelines/Internal.hpp"

namespace wxlvk::vk::pipelines::detail
{
    State s;
    SetCache g_setCache[wxlvk::vk::pipelines::kFrameSlots];

    /**
     * Logs a failed Vulkan call with its numeric result.
     *
     * @param string what : the call that failed, by name
     * @param VkResult r : what it returned
     */
    void Fail(const char* what, VkResult r)
    {
        wxl::ScriptMgr::Log(WXL_LOG_WARN, "vulkan: pipelines: %s failed: %s (%d)", what,
                            ResultName(r), static_cast<int>(r));
    }

    /**
     * Confirms every entry point this file calls is non-null, naming the ones that are not.
     *
     * @param loader::Device& d : the device table
     * @return bool ok : false when at least one is null, each logged by name
     */
    bool HaveEntryPoints(const loader::Device& d)
    {
        struct Named { const void* fn; const char* name; };
        const Named entries[] = {
            { reinterpret_cast<const void*>(d.DeviceWaitIdle),             "vkDeviceWaitIdle" },
            { reinterpret_cast<const void*>(d.CreateDescriptorSetLayout),  "vkCreateDescriptorSetLayout" },
            { reinterpret_cast<const void*>(d.DestroyDescriptorSetLayout), "vkDestroyDescriptorSetLayout" },
            { reinterpret_cast<const void*>(d.CreatePipelineLayout),       "vkCreatePipelineLayout" },
            { reinterpret_cast<const void*>(d.DestroyPipelineLayout),      "vkDestroyPipelineLayout" },
            { reinterpret_cast<const void*>(d.CreateGraphicsPipelines),    "vkCreateGraphicsPipelines" },
            { reinterpret_cast<const void*>(d.DestroyPipeline),            "vkDestroyPipeline" },
            { reinterpret_cast<const void*>(d.CreateDescriptorPool),       "vkCreateDescriptorPool" },
            { reinterpret_cast<const void*>(d.DestroyDescriptorPool),      "vkDestroyDescriptorPool" },
            { reinterpret_cast<const void*>(d.ResetDescriptorPool),        "vkResetDescriptorPool" },
            { reinterpret_cast<const void*>(d.AllocateDescriptorSets),     "vkAllocateDescriptorSets" },
            { reinterpret_cast<const void*>(d.UpdateDescriptorSets),       "vkUpdateDescriptorSets" },
            { reinterpret_cast<const void*>(d.CreateSampler),              "vkCreateSampler" },
            { reinterpret_cast<const void*>(d.DestroySampler),             "vkDestroySampler" },
            { reinterpret_cast<const void*>(d.CmdBindPipeline),            "vkCmdBindPipeline" },
            { reinterpret_cast<const void*>(d.CmdBindVertexBuffers),       "vkCmdBindVertexBuffers" },
            { reinterpret_cast<const void*>(d.CmdBindIndexBuffer),         "vkCmdBindIndexBuffer" },
            { reinterpret_cast<const void*>(d.CmdBindDescriptorSets),      "vkCmdBindDescriptorSets" },
            { reinterpret_cast<const void*>(d.CmdPushConstants),           "vkCmdPushConstants" },
            { reinterpret_cast<const void*>(d.CmdSetViewport),             "vkCmdSetViewport" },
            { reinterpret_cast<const void*>(d.CmdSetScissor),              "vkCmdSetScissor" },
            { reinterpret_cast<const void*>(d.CmdDrawIndexed),             "vkCmdDrawIndexed" },
            { reinterpret_cast<const void*>(d.CmdDraw),                    "vkCmdDraw" },
        };

        unsigned missing = 0;
        for (const Named& e : entries)
        {
            if (!e.fn)
            {
                wxl::ScriptMgr::Log(WXL_LOG_WARN, "vulkan: pipelines: entry point missing: %s",
                                    e.name);
                ++missing;
            }
        }

        if (missing)
            wxl::ScriptMgr::Log(WXL_LOG_WARN,
                                "vulkan: pipelines: %u entry point(s) missing; declining", missing);
        return missing == 0;
    }

    /**
     * Destroys the layouts and the pools, the ring's descriptor pool among them. The pipelines,
     * the samplers and the ring buffers are handled separately so that a swapchain rebuild can
     * flush the first without touching anything else.
     *
     * @param loader::Device& d : the device table
     * @param VkDevice device : the device they were created on
     */
    void DropLayoutsAndPools(const loader::Device& d, VkDevice device)
    {
        for (VkDescriptorPool& p : s.pools)
        {
            if (p != VK_NULL_HANDLE) d.DestroyDescriptorPool(device, p, nullptr);
            p = VK_NULL_HANDLE;
        }
        for (auto& slot : s.poolSpill)
            for (VkDescriptorPool& p : slot)
            {
                if (p != VK_NULL_HANDLE) d.DestroyDescriptorPool(device, p, nullptr);
                p = VK_NULL_HANDLE;
            }
        if (s.ringSpillPool != VK_NULL_HANDLE) d.DestroyDescriptorPool(device, s.ringSpillPool, nullptr);
        s.ringSpillPool = VK_NULL_HANDLE;
        for (auto& slot : s.ringSpillSets)
            for (VkDescriptorSet& set : slot) set = VK_NULL_HANDLE;
        // The pool frees its sets.
        if (s.regsPool != VK_NULL_HANDLE) d.DestroyDescriptorPool(device, s.regsPool, nullptr);
        s.regsPool = VK_NULL_HANDLE;
        for (VkDescriptorSet& set : s.regsSets) set = VK_NULL_HANDLE;

        if (s.bonesPool != VK_NULL_HANDLE) d.DestroyDescriptorPool(device, s.bonesPool, nullptr);
        s.bonesPool = VK_NULL_HANDLE;
        for (VkDescriptorSet& set : s.bonesSets) set = VK_NULL_HANDLE;

        for (uint32_t i = 0; i < s.layoutCount; ++i)
        {
            LayoutEntry& l = s.layouts[i];
            if (l.bones != VK_NULL_HANDLE) d.DestroyPipelineLayout(device, l.bones, nullptr);
            if (l.wind != VK_NULL_HANDLE) d.DestroyPipelineLayout(device, l.wind, nullptr);
            if (l.windBones != VK_NULL_HANDLE) d.DestroyPipelineLayout(device, l.windBones, nullptr);
            if (l.windShadow != VK_NULL_HANDLE) d.DestroyPipelineLayout(device, l.windShadow, nullptr);
            if (l.windShadowBones != VK_NULL_HANDLE) d.DestroyPipelineLayout(device, l.windShadowBones, nullptr);
            if (l.pipeline != VK_NULL_HANDLE) d.DestroyPipelineLayout(device, l.pipeline, nullptr);
            if (l.set0 != VK_NULL_HANDLE) d.DestroyDescriptorSetLayout(device, l.set0, nullptr);
            l = LayoutEntry();
        }
        s.layoutCount   = 0;
        s.stats.layouts = 0;

        if (s.regsLayout != VK_NULL_HANDLE) d.DestroyDescriptorSetLayout(device, s.regsLayout, nullptr);
        s.regsLayout = VK_NULL_HANDLE;
        if (s.bonesLayout != VK_NULL_HANDLE) d.DestroyDescriptorSetLayout(device, s.bonesLayout, nullptr);
        s.bonesLayout = VK_NULL_HANDLE;
        s.bonesReady  = false;
    }

    /**
     * Destroys the ring's buffers. The caller has waited for the device, and the descriptors that
     * point at them go with the pool in DropLayoutsAndPools().
     */
    void DropRing()
    {
        for (memory::Buffer& b : s.ring) memory::DestroyBuffer(b); // a zeroed buffer is a no-op
        for (auto& slot : s.ringSpill)
            for (memory::Buffer& b : slot) memory::DestroyBuffer(b);
        for (memory::Buffer& b : s.boneRing) memory::DestroyBuffer(b);
        for (uint32_t& c : s.boneCursor) c = 0;
        s.bonesReady = false;
        for (uint32_t& c : s.ringCursor) c = 0;
        for (uint32_t& a : s.ringActive) a = 0;
        s.ringAlign = 1;
    }
}
