// The shadow replay's depth-only render pass class.
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

#include "capture/ShadowPass.hpp"

#include "vulkan/Context.hpp"

#include "wxl/Script.hpp"

namespace
{
    namespace context = wxlvk::vk::context;

    struct State
    {
        bool         tried  = false;
        VkRenderPass pass   = VK_NULL_HANDLE;
        VkRenderPass load   = VK_NULL_HANDLE;   ///< the same class, LOAD on load
        VkFormat     format = VK_FORMAT_UNDEFINED;
    };
    State g;

    VkFormat PickFormat(const context::Handles& h)
    {
        const VkFormatFeatureFlags need = VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT;
        const VkFormat candidates[2] = { VK_FORMAT_D32_SFLOAT, VK_FORMAT_D16_UNORM };
        for (VkFormat f : candidates)
        {
            VkFormatProperties fp = {};
            h.iapi->GetPhysicalDeviceFormatProperties(h.physical, f, &fp);
            if ((fp.optimalTilingFeatures & need) == need) return f;
        }
        return VK_FORMAT_UNDEFINED;
    }

    /// The pass class; `load` keeps what the image holds (it must then be in SHADER_READ_ONLY_OPTIMAL).
    VkRenderPass Make(const context::Handles& h, VkFormat format, bool load)
    {
        VkAttachmentDescription a = {};
        a.format         = format;
        a.samples        = VK_SAMPLE_COUNT_1_BIT;
        a.loadOp         = load ? VK_ATTACHMENT_LOAD_OP_LOAD : VK_ATTACHMENT_LOAD_OP_CLEAR;
        a.storeOp        = VK_ATTACHMENT_STORE_OP_STORE;
        a.stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        a.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        a.initialLayout  = load ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL : VK_IMAGE_LAYOUT_UNDEFINED;
        a.finalLayout    = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        VkAttachmentReference ref = { 0, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL };
        VkSubpassDescription sub = {};
        sub.pipelineBindPoint       = VK_PIPELINE_BIND_POINT_GRAPHICS;
        sub.pDepthStencilAttachment = &ref;
        const VkPipelineStageFlags reads = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
        const VkPipelineStageFlags tests = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
        VkSubpassDependency dep[2] = {};
        dep[0].srcSubpass    = VK_SUBPASS_EXTERNAL;
        dep[0].dstSubpass    = 0;
        dep[0].srcStageMask  = reads | tests;
        dep[0].srcAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        dep[0].dstStageMask  = tests;
        dep[0].dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        dep[1].srcSubpass    = 0;
        dep[1].dstSubpass    = VK_SUBPASS_EXTERNAL;
        dep[1].srcStageMask  = tests;
        dep[1].srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        dep[1].dstStageMask  = reads;
        dep[1].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        VkRenderPassCreateInfo info = {};
        info.sType           = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        info.attachmentCount = 1;
        info.pAttachments    = &a;
        info.subpassCount    = 1;
        info.pSubpasses      = &sub;
        info.dependencyCount = 2;
        info.pDependencies   = dep;
        VkRenderPass pass = VK_NULL_HANDLE;
        if (h.dapi->CreateRenderPass(h.device, &info, nullptr, &pass) != VK_SUCCESS) return VK_NULL_HANDLE;
        return pass;
    }

    void Ensure()
    {
        if (g.tried) return;
        const context::Handles* h = context::Get();
        if (!h || !h->device || !h->iapi || !h->dapi) return;
        g.tried  = true;
        g.format = PickFormat(*h);
        if (g.format != VK_FORMAT_UNDEFINED)
        {
            g.pass = Make(*h, g.format, false);
            g.load = Make(*h, g.format, true);
        }
        wxl::ScriptMgr::Log(g.pass != VK_NULL_HANDLE ? WXL_LOG_DEBUG : WXL_LOG_WARN,
                            g.pass != VK_NULL_HANDLE ? "capture: shadow pass class made (%s, depth only)"
                                                     : "capture: no shadow pass class (no sampled depth format %s); "
                                                       "the world is never replayed",
                            wxlvk::vk::FormatName(g.format));
    }
}

namespace wxlvk::capture::shadowpass
{
    VkRenderPass Get()
    {
        Ensure();
        return g.pass;
    }

    VkRenderPass GetLoad()
    {
        Ensure();
        return g.load;
    }

    VkFormat Format()
    {
        Ensure();
        return g.format;
    }
}
