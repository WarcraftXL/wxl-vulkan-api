// The core's ImGui overlay drawn natively: the renderer of the core's wxl.ui.overlay API.
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

#include "native/Slots.hpp"

#include "native/Present.hpp"
#include "vulkan/Context.hpp"
#include "vulkan/Loader.hpp"
#include "vulkan/Memory.hpp"
#include "vulkan/Pipelines.hpp"
#include "vulkan/Shaders.hpp"
#include "vulkan/Vk.hpp"

#include "wxl/OverlayApi.h"
#include "wxl/Script.hpp"

#include <cstdint>
#include <cstring>
#include <new>

// Without an IDirect3DDevice9 the core runs its overlay (F9: input, panels, the ImGui frame) on its
// backend-neutral frame event -- GxScenePresent, before slot 38 -- and hands the finished frame here
// as plain arrays (wxl/OverlayApi.h). This draws it into the frame still open on the back buffer,
// after everything the engine drew and before slot 38's software cursor, which is where D3D's
// EndScene put it: one small pipeline (overlay.vert / overlay.frag: position, uv, colour; alpha
// blend, no depth, a scissor per draw), its vertices and indices in a host-visible buffer per frame
// slot, one descriptor set per texture (the font atlas is the only one the core makes).
//
// The back buffer is the render size (fullscreen: the format's resolution) and the overlay's
// coordinates are the window's client pixels (the Win32 backend measures the window, and the mouse
// is in those pixels): the projection maps the display onto the whole pass, and the clip rectangles
// scale by the same ratio, so the overlay lands where the mouse is at any render size.
namespace
{
    namespace context = wxlvk::vk::context;
    namespace loader  = wxlvk::vk::loader;
    namespace memory  = wxlvk::vk::memory;
    namespace present = wxlvk::native::present;

    using wxlvk::vk::ResultName;

    constexpr uint32_t kFrameSlots   = 2;
    constexpr uint32_t kMaxPipelines = 4;
    constexpr uint32_t kMaxTextures  = 8;

    struct Tex
    {
        VkImage         image = VK_NULL_HANDLE;
        VkImageView     view  = VK_NULL_HANDLE;
        memory::Block   block;
        VkDescriptorSet set   = VK_NULL_HANDLE;
        uint32_t        width = 0, height = 0;
    };

    struct Ring
    {
        memory::Buffer vb;
        memory::Buffer ib;
        uint64_t       lastUsed = 0;
    };

    struct State
    {
        bool registered = false;
        bool apiMissing = false;
        bool failed     = false;

        VkDevice              device  = nullptr;
        const loader::Device* dapi    = nullptr;
        VkDescriptorSetLayout dsl     = VK_NULL_HANDLE;
        VkPipelineLayout      layout  = VK_NULL_HANDLE;
        VkSampler             sampler = VK_NULL_HANDLE;
        VkDescriptorPool      pool    = VK_NULL_HANDLE;
        struct { VkRenderPass pass; VkPipeline pipe; } pipes[kMaxPipelines] = {};
        uint32_t              pipeCount = 0;
        Ring                  rings[kFrameSlots];

        uint64_t calls;           ///< Render calls from the core (one per open overlay frame)
        uint64_t skippedEmpty;    ///< calls with no list, no vertex or a zero display size
        uint64_t frames;          ///< overlay frames drawn
        uint64_t draws;           ///< indexed draws recorded
        uint64_t skippedTarget;   ///< frames that found a texture target bound at present
        uint64_t skippedNoPass;   ///< frames with no pass to draw into (no swapchain, hidden window)
        uint64_t skippedOther;    ///< no pipeline, no buffer
        uint64_t textures;
    };
    State g;

    WXL_OverlayRenderer g_renderer = {};

    bool Fail(const char* what, VkResult r)
    {
        if (!g.failed)
            wxl::ScriptMgr::Log(WXL_LOG_WARN, "native: overlay: %s failed (%s); the overlay is not drawn",
                                what, ResultName(r));
        g.failed = true;
        return false;
    }

    /// The layout, sampler and descriptor pool, on the live device.
    bool EnsureObjects()
    {
        if (g.failed) return false;
        const context::Handles* h = context::Get();
        if (!h || !h->device || !h->dapi) return false;
        if (g.dsl != VK_NULL_HANDLE && g.device == h->device) return true;
        g.device = h->device;
        g.dapi   = h->dapi;
        const loader::Device& d = *g.dapi;

        VkDescriptorSetLayoutBinding b = {};
        b.binding         = 0;
        b.descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        b.descriptorCount = 1;
        b.stageFlags      = VK_SHADER_STAGE_FRAGMENT_BIT;
        VkDescriptorSetLayoutCreateInfo li = {};
        li.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        li.bindingCount = 1;
        li.pBindings    = &b;
        VkResult r = d.CreateDescriptorSetLayout(g.device, &li, nullptr, &g.dsl);
        if (r != VK_SUCCESS) return Fail("vkCreateDescriptorSetLayout", r);

        VkPushConstantRange push = { VK_SHADER_STAGE_VERTEX_BIT, 0, 4 * sizeof(float) };
        VkPipelineLayoutCreateInfo pl = {};
        pl.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pl.setLayoutCount         = 1;
        pl.pSetLayouts            = &g.dsl;
        pl.pushConstantRangeCount = 1;
        pl.pPushConstantRanges    = &push;
        r = d.CreatePipelineLayout(g.device, &pl, nullptr, &g.layout);
        if (r != VK_SUCCESS) return Fail("vkCreatePipelineLayout", r);

        VkSamplerCreateInfo si = {};
        si.sType        = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        si.magFilter    = VK_FILTER_LINEAR;
        si.minFilter    = VK_FILTER_LINEAR;
        si.mipmapMode   = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        si.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        si.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        r = d.CreateSampler(g.device, &si, nullptr, &g.sampler);
        if (r != VK_SUCCESS) return Fail("vkCreateSampler", r);

        VkDescriptorPoolSize size = { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, kMaxTextures };
        VkDescriptorPoolCreateInfo pi = {};
        pi.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        pi.maxSets       = kMaxTextures;
        pi.poolSizeCount = 1;
        pi.pPoolSizes    = &size;
        r = d.CreateDescriptorPool(g.device, &pi, nullptr, &g.pool);
        if (r != VK_SUCCESS) return Fail("vkCreateDescriptorPool", r);
        return true;
    }

    /// The pipeline for the pass class the back buffer is drawn with (pipelines key on the pass).
    VkPipeline PipelineFor(VkRenderPass pass)
    {
        for (uint32_t i = 0; i < g.pipeCount; ++i)
            if (g.pipes[i].pass == pass) return g.pipes[i].pipe;
        if (g.pipeCount == kMaxPipelines) return VK_NULL_HANDLE;

        const VkShaderModule vs = wxlvk::vk::shaders::Get("overlay_vert");
        const VkShaderModule fs = wxlvk::vk::shaders::Get("overlay_frag");
        if (vs == VK_NULL_HANDLE || fs == VK_NULL_HANDLE)
        {
            Fail("loading shaders/overlay_vert.spv / overlay_frag.spv", VK_ERROR_INITIALIZATION_FAILED);
            return VK_NULL_HANDLE;
        }
        VkPipelineShaderStageCreateInfo st[2] = {};
        st[0].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        st[0].stage  = VK_SHADER_STAGE_VERTEX_BIT;
        st[0].module = vs;
        st[0].pName  = "main";
        st[1]        = st[0];
        st[1].stage  = VK_SHADER_STAGE_FRAGMENT_BIT;
        st[1].module = fs;

        VkVertexInputBindingDescription bind = { 0, sizeof(WXL_OverlayVertex), VK_VERTEX_INPUT_RATE_VERTEX };
        VkVertexInputAttributeDescription attr[3] = {
            { 0, 0, VK_FORMAT_R32G32_SFLOAT, 0 },
            { 1, 0, VK_FORMAT_R32G32_SFLOAT, 8 },
            { 2, 0, VK_FORMAT_R8G8B8A8_UNORM, 16 },
        };
        VkPipelineVertexInputStateCreateInfo vi = {};
        vi.sType                           = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        vi.vertexBindingDescriptionCount   = 1;
        vi.pVertexBindingDescriptions      = &bind;
        vi.vertexAttributeDescriptionCount = 3;
        vi.pVertexAttributeDescriptions    = attr;
        VkPipelineInputAssemblyStateCreateInfo ia = {};
        ia.sType    = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        VkPipelineViewportStateCreateInfo vp = {};
        vp.sType         = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        vp.viewportCount = 1;
        vp.scissorCount  = 1;
        VkPipelineRasterizationStateCreateInfo rs = {};
        rs.sType       = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        rs.polygonMode = VK_POLYGON_MODE_FILL;
        rs.cullMode    = VK_CULL_MODE_NONE;
        rs.frontFace   = VK_FRONT_FACE_COUNTER_CLOCKWISE;
        rs.lineWidth   = 1.0f;
        // Single-sampled, as every native pass.
        VkPipelineMultisampleStateCreateInfo ms = {};
        ms.sType                = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
        VkPipelineDepthStencilStateCreateInfo ds = {};
        ds.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
        VkPipelineColorBlendAttachmentState cba = {};
        cba.blendEnable         = VK_TRUE;
        cba.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
        cba.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
        cba.colorBlendOp        = VK_BLEND_OP_ADD;
        cba.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
        cba.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
        cba.alphaBlendOp        = VK_BLEND_OP_ADD;
        cba.colorWriteMask      = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT |
                                  VK_COLOR_COMPONENT_A_BIT;
        VkPipelineColorBlendStateCreateInfo cb = {};
        cb.sType           = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        cb.attachmentCount = 1;
        cb.pAttachments    = &cba;
        const VkDynamicState dyn[2] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
        VkPipelineDynamicStateCreateInfo dy = {};
        dy.sType             = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
        dy.dynamicStateCount = 2;
        dy.pDynamicStates    = dyn;
        VkGraphicsPipelineCreateInfo gp = {};
        gp.sType               = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        gp.stageCount          = 2;
        gp.pStages             = st;
        gp.pVertexInputState   = &vi;
        gp.pInputAssemblyState = &ia;
        gp.pViewportState      = &vp;
        gp.pRasterizationState = &rs;
        gp.pMultisampleState   = &ms;
        gp.pDepthStencilState  = &ds;
        gp.pColorBlendState    = &cb;
        gp.pDynamicState       = &dy;
        gp.layout              = g.layout;
        gp.renderPass          = pass;
        VkPipeline pipe = VK_NULL_HANDLE;
        const VkResult r = g.dapi->CreateGraphicsPipelines(g.device, VK_NULL_HANDLE, 1, &gp, nullptr, &pipe);
        if (r != VK_SUCCESS)
        {
            Fail("vkCreateGraphicsPipelines", r);
            return VK_NULL_HANDLE;
        }
        g.pipes[g.pipeCount].pass = pass;
        g.pipes[g.pipeCount].pipe = pipe;
        ++g.pipeCount;
        return pipe;
    }

    void ImageBarrier(VkCommandBuffer cmd, VkImage image, VkImageLayout from, VkImageLayout to,
                      VkAccessFlags srcAccess, VkAccessFlags dstAccess, VkPipelineStageFlags srcStage,
                      VkPipelineStageFlags dstStage)
    {
        VkImageMemoryBarrier b = {};
        b.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        b.srcAccessMask       = srcAccess;
        b.dstAccessMask       = dstAccess;
        b.oldLayout           = from;
        b.newLayout           = to;
        b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.image               = image;
        b.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        b.subresourceRange.levelCount = 1;
        b.subresourceRange.layerCount = 1;
        g.dapi->CmdPipelineBarrier(cmd, srcStage, dstStage, 0, 0, nullptr, 0, nullptr, 1, &b);
    }

    void DestroyTex(Tex* t)
    {
        if (!t) return;
        if (g.dapi && g.device)
        {
            // The set stays in the pool (the loader table has no vkFreeDescriptorSets): textures are
            // made once a session, so the pool of kMaxTextures is never the limit in practice.
            if (t->view != VK_NULL_HANDLE) g.dapi->DestroyImageView(g.device, t->view, nullptr);
            if (t->image != VK_NULL_HANDLE) g.dapi->DestroyImage(g.device, t->image, nullptr);
        }
        memory::Free(t->block);
        delete t;
    }

    // --- the renderer the core calls -----------------------------------------------------------------

    void* __cdecl CreateTexture(void*, const uint8_t* rgba, uint32_t width, uint32_t height)
    {
        if (!rgba || !width || !height || !EnsureObjects()) return nullptr;
        const loader::Device& d = *g.dapi;
        Tex* t = new (std::nothrow) Tex();
        if (!t) return nullptr;
        t->width  = width;
        t->height = height;

        VkImageCreateInfo ii = {};
        ii.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        ii.imageType     = VK_IMAGE_TYPE_2D;
        ii.format        = VK_FORMAT_R8G8B8A8_UNORM;
        ii.extent        = VkExtent3D{ width, height, 1 };
        ii.mipLevels     = 1;
        ii.arrayLayers   = 1;
        ii.samples       = VK_SAMPLE_COUNT_1_BIT;
        ii.tiling        = VK_IMAGE_TILING_OPTIMAL;
        ii.usage         = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        ii.sharingMode   = VK_SHARING_MODE_EXCLUSIVE;
        ii.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        VkResult r = d.CreateImage(g.device, &ii, nullptr, &t->image);
        if (r != VK_SUCCESS)
        {
            t->image = VK_NULL_HANDLE;
            DestroyTex(t);
            Fail("vkCreateImage (overlay texture)", r);
            return nullptr;
        }
        VkMemoryRequirements req = {};
        d.GetImageMemoryRequirements(g.device, t->image, &req);
        if (!memory::Allocate(req, memory::Where::DeviceLocal, t->block)
            || d.BindImageMemory(g.device, t->image, t->block.memory, t->block.offset) != VK_SUCCESS)
        {
            DestroyTex(t);
            Fail("device memory for the overlay texture", VK_ERROR_OUT_OF_DEVICE_MEMORY);
            return nullptr;
        }
        VkImageViewCreateInfo vi = {};
        vi.sType    = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        vi.image    = t->image;
        vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
        vi.format   = VK_FORMAT_R8G8B8A8_UNORM;
        vi.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        vi.subresourceRange.levelCount = 1;
        vi.subresourceRange.layerCount = 1;
        r = d.CreateImageView(g.device, &vi, nullptr, &t->view);
        if (r != VK_SUCCESS)
        {
            t->view = VK_NULL_HANDLE;
            DestroyTex(t);
            Fail("vkCreateImageView (overlay texture)", r);
            return nullptr;
        }
        VkDescriptorSetAllocateInfo ai = {};
        ai.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        ai.descriptorPool     = g.pool;
        ai.descriptorSetCount = 1;
        ai.pSetLayouts        = &g.dsl;
        r = d.AllocateDescriptorSets(g.device, &ai, &t->set);
        if (r != VK_SUCCESS)
        {
            t->set = VK_NULL_HANDLE;
            DestroyTex(t);
            Fail("vkAllocateDescriptorSets (overlay texture)", r);
            return nullptr;
        }
        VkDescriptorImageInfo di = { g.sampler, t->view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
        VkWriteDescriptorSet w = {};
        w.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w.dstSet          = t->set;
        w.dstBinding      = 0;
        w.descriptorCount = 1;
        w.descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        w.pImageInfo      = &di;
        d.UpdateDescriptorSets(g.device, 1, &w, 0, nullptr);

        // The texels, into the frame's pre-pass buffer: submitted ahead of every draw of the frame.
        const VkDeviceSize bytes = VkDeviceSize(width) * height * 4;
        VkBuffer           buffer = VK_NULL_HANDLE;
        VkDeviceSize       offset = 0;
        void*              mapped = nullptr;
        VkCommandBuffer    cmd    = present::PrePass();
        if (!cmd || !present::Stage(bytes, buffer, offset, mapped))
        {
            DestroyTex(t);
            Fail("staging the overlay texture", VK_ERROR_OUT_OF_HOST_MEMORY);
            return nullptr;
        }
        std::memcpy(mapped, rgba, static_cast<size_t>(bytes));
        ImageBarrier(cmd, t->image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 0,
                     VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
        VkBufferImageCopy c = {};
        c.bufferOffset                = offset;
        c.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        c.imageSubresource.layerCount = 1;
        c.imageExtent                 = VkExtent3D{ width, height, 1 };
        d.CmdCopyBufferToImage(cmd, buffer, t->image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &c);
        ImageBarrier(cmd, t->image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                     VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                     VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);

        ++g.textures;
        wxl::ScriptMgr::Log(WXL_LOG_DEBUG, "native: overlay: texture %ux%u made for the core's ImGui overlay "
                            "(%llu KiB, uploaded in the pre-pass)", width, height,
                            static_cast<unsigned long long>(bytes / 1024));
        return t;
    }

    void __cdecl DestroyTexture(void*, void* texture)
    {
        if (!texture) return;
        if (g.dapi && g.device) g.dapi->DeviceWaitIdle(g.device);
        DestroyTex(static_cast<Tex*>(texture));
    }

    /// Room for `bytes` in one of the slot's buffers: grown (doubling) when short, the old one
    /// retired by the serial of the last frame that read it.
    bool Fit(memory::Buffer& buf, VkDeviceSize bytes, VkBufferUsageFlags usage, uint64_t lastUsed)
    {
        if (buf.handle != VK_NULL_HANDLE && buf.size >= bytes) return true;
        VkDeviceSize size = buf.size ? buf.size : 64 * 1024;
        while (size < bytes) size *= 2;
        if (buf.handle != VK_NULL_HANDLE) present::RetireBuffer(buf, lastUsed);
        if (!memory::CreateBuffer(size, usage, memory::Where::HostVisible, buf) || !buf.block.mapped)
        {
            memory::DestroyBuffer(buf);
            return false;
        }
        return true;
    }

    void __cdecl Render(void*, const WXL_OverlayFrame* f)
    {
        ++g.calls;
        if (!f || !f->listCount || !f->totalVertices || f->displaySize[0] <= 0.0f || f->displaySize[1] <= 0.0f)
        {
            ++g.skippedEmpty;
            return;
        }
        if (!EnsureObjects())
        {
            ++g.skippedOther;
            return;
        }
        // On the back buffer only: a texture target still bound at present is the engine's business.
        if (!present::OnBackBuffer())
        {
            ++g.skippedTarget;
            return;
        }
        VkCommandBuffer cmd = present::Pass();
        if (!cmd)
        {
            ++g.skippedNoPass;
            return;
        }
        const VkRenderPass pass   = present::RenderPass();
        const VkExtent2D   extent = present::Extent();
        const VkPipeline   pipe   = PipelineFor(pass);
        if (pipe == VK_NULL_HANDLE || extent.width == 0 || extent.height == 0)
        {
            ++g.skippedOther;
            return;
        }

        Ring&              ring  = g.rings[present::Slot() % kFrameSlots];
        const VkDeviceSize vb    = VkDeviceSize(f->totalVertices) * sizeof(WXL_OverlayVertex);
        const VkDeviceSize ib    = (VkDeviceSize(f->totalIndices) * sizeof(uint16_t) + 3) & ~VkDeviceSize(3);
        if (!Fit(ring.vb, vb, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, ring.lastUsed)
            || !Fit(ring.ib, ib ? ib : 4, VK_BUFFER_USAGE_INDEX_BUFFER_BIT, ring.lastUsed))
        {
            ++g.skippedOther;
            return;
        }
        // This slot's previous frame was waited for when the frame opened: the bytes are free.
        uint8_t* vdst = static_cast<uint8_t*>(ring.vb.block.mapped);
        uint8_t* idst = static_cast<uint8_t*>(ring.ib.block.mapped);
        for (uint32_t n = 0; n < f->listCount; ++n)
        {
            const WXL_OverlayList& l = f->lists[n];
            std::memcpy(vdst, l.vertices, size_t(l.vertexCount) * sizeof(WXL_OverlayVertex));
            std::memcpy(idst, l.indices, size_t(l.indexCount) * sizeof(uint16_t));
            vdst += size_t(l.vertexCount) * sizeof(WXL_OverlayVertex);
            idst += size_t(l.indexCount) * sizeof(uint16_t);
        }
        memory::FlushHost(ring.vb.block);
        memory::FlushHost(ring.ib.block);
        ring.lastUsed = present::Serial();

        const loader::Device& d = *g.dapi;
        // The engine's draws that follow in this pass (the software cursor) must bind afresh.
        wxlvk::vk::pipelines::InvalidateBindings();
        d.CmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
        const VkDeviceSize zero = 0;
        d.CmdBindVertexBuffers(cmd, 0, 1, &ring.vb.handle, &zero);
        d.CmdBindIndexBuffer(cmd, ring.ib.handle, 0, VK_INDEX_TYPE_UINT16);
        VkViewport viewport = {};
        viewport.width    = static_cast<float>(extent.width);
        viewport.height   = static_cast<float>(extent.height);
        viewport.maxDepth = 1.0f;
        d.CmdSetViewport(cmd, 0, 1, &viewport);
        // Display pixels to clip space over the whole pass; Vulkan's +y down is the display's.
        float push[4];
        push[0] = 2.0f / f->displaySize[0];
        push[1] = 2.0f / f->displaySize[1];
        push[2] = -1.0f - f->displayPos[0] * push[0];
        push[3] = -1.0f - f->displayPos[1] * push[1];
        d.CmdPushConstants(cmd, g.layout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof push, push);

        const float sx = static_cast<float>(extent.width) / f->displaySize[0];
        const float sy = static_cast<float>(extent.height) / f->displaySize[1];
        uint32_t globalVtx = 0, globalIdx = 0;
        const Tex* bound = nullptr;
        for (uint32_t n = 0; n < f->listCount; ++n)
        {
            const WXL_OverlayList& l = f->lists[n];
            for (uint32_t i = 0; i < l.cmdCount; ++i)
            {
                const WXL_OverlayCmd& c = l.cmds[i];
                const Tex* t = static_cast<const Tex*>(c.texture);
                if (!t || !c.elemCount) continue;
                float x0 = c.clip[0] * sx, y0 = c.clip[1] * sy, x1 = c.clip[2] * sx, y1 = c.clip[3] * sy;
                if (x0 < 0.0f) x0 = 0.0f;
                if (y0 < 0.0f) y0 = 0.0f;
                if (x1 > float(extent.width)) x1 = float(extent.width);
                if (y1 > float(extent.height)) y1 = float(extent.height);
                if (x1 <= x0 || y1 <= y0) continue;
                VkRect2D scissor = {};
                scissor.offset.x      = static_cast<int32_t>(x0);
                scissor.offset.y      = static_cast<int32_t>(y0);
                scissor.extent.width  = static_cast<uint32_t>(x1 - x0);
                scissor.extent.height = static_cast<uint32_t>(y1 - y0);
                if (!scissor.extent.width || !scissor.extent.height) continue;
                d.CmdSetScissor(cmd, 0, 1, &scissor);
                if (t != bound)
                {
                    d.CmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, g.layout, 0, 1, &t->set, 0, nullptr);
                    bound = t;
                }
                d.CmdDrawIndexed(cmd, c.elemCount, 1, c.idxOffset + globalIdx,
                                 static_cast<int32_t>(c.vtxOffset + globalVtx), 0);
                ++g.draws;
            }
            globalVtx += l.vertexCount;
            globalIdx += l.indexCount;
        }
        present::NoteDraw();
        ++g.frames;
    }
}

namespace wxlvk::native::impl
{
    void OverlayRegister()
    {
        if (g.registered || g.apiMissing) return;
        const WXL_OverlayApi* api =
            wxl::ScriptMgr::GetInterface<const WXL_OverlayApi>(WXL_OVERLAY_API_NAME, WXL_OVERLAY_API_VERSION);
        if (!api || api->structSize < sizeof(WXL_OverlayApi) || !api->RegisterRenderer)
        {
            g.apiMissing = true;
            wxl::ScriptMgr::Log(WXL_LOG_WARN,
                                "native: the core does not publish %s v%u (an older core): its ImGui overlay "
                                "stays dark, and its OnFrame / OnEndScene come only with a core that has "
                                "the GxScenePresent frame hook",
                                WXL_OVERLAY_API_NAME, WXL_OVERLAY_API_VERSION);
            return;
        }
        g_renderer.structSize     = sizeof g_renderer;
        g_renderer.user           = &g;
        g_renderer.CreateTexture  = &CreateTexture;
        g_renderer.DestroyTexture = &DestroyTexture;
        g_renderer.Render         = &Render;
        g.registered = api->RegisterRenderer(&g_renderer, "wxl-vulkan-api native device") != 0;
        wxl::ScriptMgr::Log(g.registered ? WXL_LOG_DEBUG : WXL_LOG_WARN,
                            g.registered ? "native: overlay renderer registered with the core (%s v%u): the "
                                           "core's OnFrame / OnEndScene fire from its GxScenePresent hook and "
                                           "its ImGui overlay (F9) is drawn through Vulkan"
                                         : "native: the core refused the overlay renderer (%s v%u)",
                            WXL_OVERLAY_API_NAME, WXL_OVERLAY_API_VERSION);
    }

    void LogOverlayStats()
    {
        if (!g.registered) return;
        wxl::ScriptMgr::Log(WXL_LOG_INFO,
                            "native: overlay: %llu Render call(s), %llu ImGui frame(s) drawn, %llu draw(s), "
                            "%llu texture(s); skipped %llu empty, %llu with a texture target bound, %llu with no "
                            "pass, %llu other%s",
                            static_cast<unsigned long long>(g.calls), static_cast<unsigned long long>(g.frames), static_cast<unsigned long long>(g.draws),
                            static_cast<unsigned long long>(g.textures),
                            static_cast<unsigned long long>(g.skippedEmpty),
                            static_cast<unsigned long long>(g.skippedTarget),
                            static_cast<unsigned long long>(g.skippedNoPass),
                            static_cast<unsigned long long>(g.skippedOther), g.failed ? " (FAILED, see above)" : "");
    }
}
