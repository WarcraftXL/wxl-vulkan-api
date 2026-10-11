// The set-0 and pipeline layouts, and the bone-palette and tree-wind layouts.
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
    // --- layouts ------------------------------------------------------------------------------

    /**
     * The set-0 layout and pipeline layout for a set of sampler slots, created on first use.
     *
     * Binding N is slot N, a combined image sampler in the fragment stage; the pipeline layout is
     * that set, the ring's set 1, and the UI's push range. A mask of 0 (a pixel module with no
     * sampler) is a layout with no bindings, which Vulkan allows and which keeps set 1 at index 1
     * for every pipeline.
     *
     * @param uint32 mask : bit N for slot N
     * @param uint32& index : receives the table index
     * @return bool ok : false when the table is full or a create failed (logged)
     */
    bool LayoutFor(uint32_t mask, uint32_t& index)
    {
        for (uint32_t i = 0; i < s.layoutCount; ++i)
        {
            if (s.layouts[i].mask == mask)
            {
                index = i;
                return true;
            }
        }
        if (s.layoutCount >= kMaxLayouts)
        {
            if (!s.loggedLayoutsFull)
            {
                s.loggedLayoutsFull = true;
                wxl::ScriptMgr::Log(WXL_LOG_WARN,
                                    "vulkan: pipelines: %u distinct sampler-slot sets (kMaxLayouts); "
                                    "mask 0x%04X and later are refused (logged once)",
                                    kMaxLayouts, mask);
            }
            return false;
        }

        const loader::Device& d = *s.dapi;

        VkDescriptorSetLayoutBinding bindings[Cmd::kTextureStages] = {};
        uint32_t count = 0;
        for (uint32_t slot = 0; slot < Cmd::kTextureStages; ++slot)
        {
            if (!(mask & (1u << slot))) continue;
            VkDescriptorSetLayoutBinding& b = bindings[count++];
            b.binding         = slot;
            b.descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            b.descriptorCount = 1;
            b.stageFlags      = VK_SHADER_STAGE_FRAGMENT_BIT;
        }

        VkDescriptorSetLayoutCreateInfo setLayout = {};
        setLayout.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        setLayout.bindingCount = count;
        setLayout.pBindings    = count ? bindings : nullptr;

        LayoutEntry e;
        e.mask = mask;
        VkResult r = d.CreateDescriptorSetLayout(s.device, &setLayout, nullptr, &e.set0);
        if (r != VK_SUCCESS)
        {
            Fail("vkCreateDescriptorSetLayout (set 0)", r);
            return false;
        }

        // The UI's four constant registers, pushed per draw. 64 bytes is far under the 128 the
        // spec guarantees for maxPushConstantsSize, so no limit is queried. The ring pairs do not
        // declare the range; a layout with a range a stage never reads is fine.
        VkPushConstantRange push = {};
        push.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
        push.offset     = 0;
        push.size       = kPushBytes;

        const VkDescriptorSetLayout sets[2] = { e.set0, s.regsLayout };

        VkPipelineLayoutCreateInfo layout = {};
        layout.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        layout.setLayoutCount         = 2;
        layout.pSetLayouts            = sets;
        layout.pushConstantRangeCount = 1;
        layout.pPushConstantRanges    = &push;

        r = d.CreatePipelineLayout(s.device, &layout, nullptr, &e.pipeline);
        if (r != VK_SUCCESS)
        {
            Fail("vkCreatePipelineLayout", r);
            d.DestroyDescriptorSetLayout(s.device, e.set0, nullptr);
            return false;
        }

        index = s.layoutCount;
        s.layouts[s.layoutCount++] = e;
        s.stats.layouts = s.layoutCount;
        wxl::ScriptMgr::Log(WXL_LOG_DEBUG,
                            "vulkan: pipelines: layout #%u for sampler slots 0x%04X (%u binding%s)",
                            s.layoutCount, mask, count, count == 1 ? "" : "s");
        return true;
    }

    /**
     * The pipeline layout of a set-0 layout with the palette's set 2 behind set 1, made on first use.
     *
     * @param uint32 index : the layout table entry
     * @return VkPipelineLayout layout : VK_NULL_HANDLE when it could not be made (logged)
     */
    VkPipelineLayout BonesLayoutFor(uint32_t index)
    {
        LayoutEntry& e = s.layouts[index];
        if (e.bones != VK_NULL_HANDLE || !s.bonesReady) return e.bones;
        VkPushConstantRange push = {};
        push.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
        push.offset     = 0;
        push.size       = kPushBytes;
        const VkDescriptorSetLayout sets[3] = { e.set0, s.regsLayout, s.bonesLayout };
        VkPipelineLayoutCreateInfo info = {};
        info.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        info.setLayoutCount         = 3;
        info.pSetLayouts            = sets;
        info.pushConstantRangeCount = 1;
        info.pPushConstantRanges    = &push;
        const VkResult r = s.dapi->CreatePipelineLayout(s.device, &info, nullptr, &e.bones);
        if (r != VK_SUCCESS)
        {
            e.bones = VK_NULL_HANDLE;
            Fail("vkCreatePipelineLayout (bone palette)", r);
        }
        return e.bones;
    }

    /**
     * The pipeline layout of a set-0 layout with tree wind's set 3 behind an empty set 2, made on
     * first use (src/vulkan/Wind.hpp).
     *
     * @param uint32 index : the layout table entry
     * @return VkPipelineLayout layout : VK_NULL_HANDLE until the wind's resources exist, or when it
     *         could not be made (logged)
     */
    VkPipelineLayout WindLayoutFor(uint32_t index, bool bones, bool shadow)
    {
        LayoutEntry& e = s.layouts[index];
        VkPipelineLayout& out = shadow ? (bones ? e.windShadowBones : e.windShadow) : (bones ? e.windBones : e.wind);
        if (out != VK_NULL_HANDLE || !wxlvk::vk::wind::Ready()) return out;
        if (bones && BonesLayoutFor(index) == VK_NULL_HANDLE) return VK_NULL_HANDLE;
        VkPushConstantRange push = {};
        push.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
        push.offset     = 0;
        // The shadow replay's: the wind's values (16 bytes), then the light's matrix (shadowgen.py).
        push.size       = shadow ? static_cast<uint32_t>(sizeof(wxlvk::vk::wind::Draw) + 64) : kPushBytes;
        const VkDescriptorSetLayout sets[4] = { e.set0, s.regsLayout,
                                                bones ? s.bonesLayout : wxlvk::vk::wind::EmptyLayout(),
                                                wxlvk::vk::wind::SetLayout() };
        VkPipelineLayoutCreateInfo info = {};
        info.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        info.setLayoutCount         = 4;
        info.pSetLayouts            = sets;
        info.pushConstantRangeCount = 1;
        info.pPushConstantRanges    = &push;
        const VkResult r = s.dapi->CreatePipelineLayout(s.device, &info, nullptr, &out);
        if (r != VK_SUCCESS)
        {
            out = VK_NULL_HANDLE;
            Fail("vkCreatePipelineLayout (tree wind)", r);
        }
        return out;
    }

}
