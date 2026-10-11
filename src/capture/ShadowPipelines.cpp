// The shadow replay's pipelines: one per captured pipeline entry (and alpha test), depth only.
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

#include "capture/ShadowPipelines.hpp"

#include "capture/ShadowPass.hpp"
#include "vulkan/Context.hpp"
#include "vulkan/Pipelines.hpp"
#include "vulkan/Shaders.hpp"

#include "wxl/Script.hpp"

namespace
{
    namespace context   = wxlvk::vk::context;
    namespace pipelines = wxlvk::vk::pipelines;
    using wxlvk::capture::shadowpipes::Pipe;

    /// An open-addressed table over (entry << 1 | alphaTest): a zone has a few hundred entries.
    constexpr uint32_t kSlots = 4096;

    struct Slot
    {
        uint32_t key    = 0xFFFFFFFFu;
        Pipe     pipe;
        bool     failed = false;
    };

    struct State
    {
        VkDevice device = nullptr;
        Slot     slots[kSlots];
        uint32_t built  = 0;
        uint32_t failed = 0;
        bool     full   = false;
    };
    State g;

    VkPipeline Build(const pipelines::ShadowSource& src, VkRenderPass pass, const context::Handles& h)
    {
        namespace shaders = wxlvk::vk::shaders;
        const VkShaderModule vert = shaders::GetPath(src.vertPath);
        const VkShaderModule frag = src.fragPath[0] ? shaders::GetPath(src.fragPath) : VK_NULL_HANDLE;
        if (vert == VK_NULL_HANDLE || (src.fragPath[0] && frag == VK_NULL_HANDLE)) return VK_NULL_HANDLE;
        VkPipelineShaderStageCreateInfo stages[2] = {};
        stages[0].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stages[0].stage  = VK_SHADER_STAGE_VERTEX_BIT;
        stages[0].module = vert;
        stages[0].pName  = "main";
        stages[1]        = stages[0];
        stages[1].stage  = VK_SHADER_STAGE_FRAGMENT_BIT;
        stages[1].module = frag;

        VkPipelineVertexInputStateCreateInfo input = {};
        input.sType                           = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        input.vertexBindingDescriptionCount   = src.bindingCount;
        input.pVertexBindingDescriptions      = src.bindings;
        input.vertexAttributeDescriptionCount = src.attrCount;
        input.pVertexAttributeDescriptions    = src.attrs;
        VkPipelineInputAssemblyStateCreateInfo assembly = {};
        assembly.sType    = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        assembly.topology = src.topology;
        VkPipelineViewportStateCreateInfo viewport = {};
        viewport.sType         = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        viewport.viewportCount = 1;
        viewport.scissorCount  = 1;
        VkPipelineRasterizationStateCreateInfo raster = {};
        raster.sType           = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        raster.polygonMode     = VK_POLYGON_MODE_FILL;
        raster.cullMode        = VK_CULL_MODE_NONE;
        raster.frontFace       = VK_FRONT_FACE_COUNTER_CLOCKWISE;
        raster.depthBiasEnable = VK_TRUE;
        raster.lineWidth       = 1.0f;
        VkPipelineMultisampleStateCreateInfo multisample = {};
        multisample.sType                = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
        VkPipelineDepthStencilStateCreateInfo depth = {};
        depth.sType            = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
        depth.depthTestEnable  = VK_TRUE;
        depth.depthWriteEnable = VK_TRUE;
        depth.depthCompareOp   = VK_COMPARE_OP_LESS_OR_EQUAL;
        depth.maxDepthBounds   = 1.0f;
        VkPipelineColorBlendStateCreateInfo blend = {};
        blend.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        const VkDynamicState dynamics[3] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR, VK_DYNAMIC_STATE_DEPTH_BIAS };
        VkPipelineDynamicStateCreateInfo dynamic = {};
        dynamic.sType             = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
        dynamic.dynamicStateCount = 3;
        dynamic.pDynamicStates    = dynamics;

        VkGraphicsPipelineCreateInfo info = {};
        info.sType               = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        info.stageCount          = frag != VK_NULL_HANDLE ? 2u : 1u;
        info.pStages             = stages;
        info.pVertexInputState   = &input;
        info.pInputAssemblyState = &assembly;
        info.pViewportState      = &viewport;
        info.pRasterizationState = &raster;
        info.pMultisampleState   = &multisample;
        info.pDepthStencilState  = &depth;
        info.pColorBlendState    = &blend;
        info.pDynamicState       = &dynamic;
        info.layout              = src.layout;
        info.renderPass          = pass;
        info.basePipelineIndex   = -1;
        VkPipeline pipe = VK_NULL_HANDLE;
        if (h.dapi->CreateGraphicsPipelines(h.device, pipelines::PipelineCache(), 1, &info, nullptr, &pipe) != VK_SUCCESS)
            return VK_NULL_HANDLE;
        return pipe;
    }

    Slot* Find(uint32_t key, bool insert)
    {
        uint32_t i = (key * 2654435761u) & (kSlots - 1);
        for (uint32_t probe = 0; probe < kSlots; ++probe, i = (i + 1) & (kSlots - 1))
        {
            if (g.slots[i].key == key) return &g.slots[i];
            if (g.slots[i].key == 0xFFFFFFFFu) return insert ? &g.slots[i] : nullptr;
        }
        return nullptr;
    }

    void Fill(Slot& slot, uint32_t entry, bool alphaTest, const context::Handles& h)
    {
        pipelines::ShadowSource src;
        const VkRenderPass pass = wxlvk::capture::shadowpass::Get();
        if (pass == VK_NULL_HANDLE || !pipelines::ShadowSourceFor(entry, alphaTest, src)) return;
        slot.pipe.pipeline = Build(src, pass, h);
        slot.pipe.layout   = src.layout;
        slot.pipe.bones    = src.bones;
        slot.pipe.wind     = src.wind;
    }
}

namespace wxlvk::capture::shadowpipes
{
    bool Get(uint32_t entry, bool alphaTest, Pipe& out)
    {
        const context::Handles* h = context::Get();
        if (!h || !h->device) return false;
        if (g.device != h->device)
        {
            // A new device: the old one's pipelines are gone with it, never destroyed here.
            for (Slot& s : g.slots) s = Slot();
            g.built  = 0;
            g.failed = 0;
            g.full   = false;
            g.device = h->device;
        }
        const uint32_t key = (entry << 1) | (alphaTest ? 1u : 0u);
        Slot* slot = Find(key, true);
        if (!slot)
        {
            if (!g.full)
            {
                g.full = true;
                wxl::ScriptMgr::Log(WXL_LOG_WARN, "capture: the shadow pipeline table is full (%u); later "
                                    "casters are not replayed (logged once)", kSlots);
            }
            return false;
        }
        if (slot->key != key)
        {
            slot->key = key;
            Fill(*slot, entry, alphaTest, *h);
            slot->failed = slot->pipe.pipeline == VK_NULL_HANDLE;
            if (slot->failed) ++g.failed;
            else ++g.built;
        }
        if (slot->failed) return false;
        out = slot->pipe;
        return true;
    }

    void Counts(uint32_t& built, uint32_t& failed)
    {
        built  = g.built;
        failed = g.failed;
    }
}
