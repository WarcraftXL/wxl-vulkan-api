// The present pass (scale, gamma LUT), and finishing a frame.
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
    // --- the present pass -------------------------------------------------------------------------------
    //
    // The copy above is the present whenever the back buffer and the swapchain have one size and no
    // gamma is applied -- every windowed frame. Fullscreen renders at the format's
    // resolution on a desktop-sized borderless window (Vulkan 1.0 has no exclusive fullscreen), and
    // gamma is D3D's fullscreen-only ramp: both are this pass instead -- one full-screen triangle that
    // samples the back buffer (LINEAR, stretched over the whole swapchain image, as the engine's mouse
    // mapping normalises by the whole client rect) and, with gamma on, looks each channel up in a
    // 256-entry LUT.

    bool PresentFail(const char* what, VkResult r)
    {
        s.ppFailed = true;
        wxl::ScriptMgr::Log(WXL_LOG_WARN, "native: present: %s failed (%s); the present stays a copy "
                            "(no scaling, no gamma)", what, ResultName(r));
        return false;
    }

    /// The pass, its pipeline, the descriptor set and the LUT image. False: copies only (logged once).
    bool EnsurePresentObjects(VkFormat swapFormat)
    {
        if (s.ppFailed) return false;
        const loader::Device& d = *s.dapi;
        VkResult r = VK_SUCCESS;

        if (s.ppPass != VK_NULL_HANDLE && s.ppFormat != swapFormat)
        {
            // A new swapchain format: pass and pipeline are remade (the device is idle: a rebuild).
            if (s.ppPipe != VK_NULL_HANDLE) d.DestroyPipeline(s.device, s.ppPipe, nullptr);
            d.DestroyRenderPass(s.device, s.ppPass, nullptr);
            s.ppPipe = VK_NULL_HANDLE;
            s.ppPass = VK_NULL_HANDLE;
        }
        if (s.ppPass == VK_NULL_HANDLE)
        {
            VkAttachmentDescription a = {};
            a.format         = swapFormat;
            a.samples        = VK_SAMPLE_COUNT_1_BIT;
            a.loadOp         = VK_ATTACHMENT_LOAD_OP_DONT_CARE;   // every pixel is written
            a.storeOp        = VK_ATTACHMENT_STORE_OP_STORE;
            a.stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
            a.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
            a.initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED;
            a.finalLayout    = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
            VkAttachmentReference ref = { 0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL };
            VkSubpassDescription sub = {};
            sub.pipelineBindPoint    = VK_PIPELINE_BIND_POINT_GRAPHICS;
            sub.colorAttachmentCount = 1;
            sub.pColorAttachments    = &ref;
            // The layout transition waits for the acquire, which the submit waits for at this stage.
            VkSubpassDependency dep = {};
            dep.srcSubpass    = VK_SUBPASS_EXTERNAL;
            dep.dstSubpass    = 0;
            dep.srcStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
            dep.srcAccessMask = 0;
            dep.dstStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
            dep.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
            VkRenderPassCreateInfo info = {};
            info.sType           = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
            info.attachmentCount = 1;
            info.pAttachments    = &a;
            info.subpassCount    = 1;
            info.pSubpasses      = &sub;
            info.dependencyCount = 1;
            info.pDependencies   = &dep;
            r = d.CreateRenderPass(s.device, &info, nullptr, &s.ppPass);
            if (r != VK_SUCCESS)
            {
                s.ppPass = VK_NULL_HANDLE;
                return PresentFail("vkCreateRenderPass (present)", r);
            }
            s.ppFormat = swapFormat;
        }

        if (s.ppDsl == VK_NULL_HANDLE)
        {
            VkDescriptorSetLayoutBinding b[2] = {};
            for (uint32_t i = 0; i < 2; ++i)
            {
                b[i].binding         = i;
                b[i].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                b[i].descriptorCount = 1;
                b[i].stageFlags      = VK_SHADER_STAGE_FRAGMENT_BIT;
            }
            VkDescriptorSetLayoutCreateInfo li = {};
            li.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
            li.bindingCount = 2;
            li.pBindings    = b;
            r = d.CreateDescriptorSetLayout(s.device, &li, nullptr, &s.ppDsl);
            if (r != VK_SUCCESS)
            {
                s.ppDsl = VK_NULL_HANDLE;
                return PresentFail("vkCreateDescriptorSetLayout (present)", r);
            }
            VkPushConstantRange push = { VK_SHADER_STAGE_FRAGMENT_BIT, 0, 4 * sizeof(float) };
            VkPipelineLayoutCreateInfo pl = {};
            pl.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
            pl.setLayoutCount         = 1;
            pl.pSetLayouts            = &s.ppDsl;
            pl.pushConstantRangeCount = 1;
            pl.pPushConstantRanges    = &push;
            r = d.CreatePipelineLayout(s.device, &pl, nullptr, &s.ppLayout);
            if (r != VK_SUCCESS)
            {
                s.ppLayout = VK_NULL_HANDLE;
                return PresentFail("vkCreatePipelineLayout (present)", r);
            }
            VkDescriptorPoolSize size = { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 2 };
            VkDescriptorPoolCreateInfo pi = {};
            pi.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
            pi.maxSets       = 1;
            pi.poolSizeCount = 1;
            pi.pPoolSizes    = &size;
            r = d.CreateDescriptorPool(s.device, &pi, nullptr, &s.ppPool);
            if (r != VK_SUCCESS)
            {
                s.ppPool = VK_NULL_HANDLE;
                return PresentFail("vkCreateDescriptorPool (present)", r);
            }
            VkDescriptorSetAllocateInfo ai = {};
            ai.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
            ai.descriptorPool     = s.ppPool;
            ai.descriptorSetCount = 1;
            ai.pSetLayouts        = &s.ppDsl;
            r = d.AllocateDescriptorSets(s.device, &ai, &s.ppSet);
            if (r != VK_SUCCESS)
            {
                s.ppSet = VK_NULL_HANDLE;
                return PresentFail("vkAllocateDescriptorSets (present)", r);
            }
            VkSamplerCreateInfo si = {};
            si.sType        = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
            si.magFilter    = VK_FILTER_LINEAR;
            si.minFilter    = VK_FILTER_LINEAR;
            si.mipmapMode   = VK_SAMPLER_MIPMAP_MODE_NEAREST;
            si.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
            si.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
            si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
            si.maxLod       = 0.0f;
            r = d.CreateSampler(s.device, &si, nullptr, &s.ppSampler);
            if (r != VK_SUCCESS)
            {
                s.ppSampler = VK_NULL_HANDLE;
                return PresentFail("vkCreateSampler (present)", r);
            }
        }

        if (s.lutImage == VK_NULL_HANDLE)
        {
            if (!MakeRaw(VK_FORMAT_R8G8B8A8_UNORM, VkExtent2D{ 256, 1 },
                         VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, s.lutImage, s.lutView,
                         s.lutBlock, "gamma LUT"))
                return PresentFail("the gamma LUT image", VK_ERROR_OUT_OF_DEVICE_MEMORY);
            s.lutLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            s.lutDirty  = true;
            VkDescriptorImageInfo li = { s.ppSampler, s.lutView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
            VkWriteDescriptorSet w = {};
            w.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w.dstSet          = s.ppSet;
            w.dstBinding      = 1;
            w.descriptorCount = 1;
            w.descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            w.pImageInfo      = &li;
            d.UpdateDescriptorSets(s.device, 1, &w, 0, nullptr);
        }

        if (s.ppPipe == VK_NULL_HANDLE)
        {
            const VkShaderModule vs = wxlvk::vk::shaders::Get("present_vert");
            const VkShaderModule fs = wxlvk::vk::shaders::Get("present_frag");
            if (vs == VK_NULL_HANDLE || fs == VK_NULL_HANDLE)
                return PresentFail("loading shaders/present_vert.spv / present_frag.spv",
                                   VK_ERROR_INITIALIZATION_FAILED);
            VkPipelineShaderStageCreateInfo st[2] = {};
            st[0].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            st[0].stage  = VK_SHADER_STAGE_VERTEX_BIT;
            st[0].module = vs;
            st[0].pName  = "main";
            st[1]        = st[0];
            st[1].stage  = VK_SHADER_STAGE_FRAGMENT_BIT;
            st[1].module = fs;
            VkPipelineVertexInputStateCreateInfo vi = {};
            vi.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
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
            VkPipelineMultisampleStateCreateInfo ms = {};
            ms.sType                = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
            ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
            VkPipelineColorBlendAttachmentState cba = {};
            cba.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT |
                                 VK_COLOR_COMPONENT_A_BIT;
            VkPipelineColorBlendStateCreateInfo cb = {};
            cb.sType           = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
            cb.attachmentCount = 1;
            cb.pAttachments    = &cba;
            const VkDynamicState dyn[2] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
            VkPipelineDynamicStateCreateInfo ds = {};
            ds.sType             = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
            ds.dynamicStateCount = 2;
            ds.pDynamicStates    = dyn;
            VkGraphicsPipelineCreateInfo gp = {};
            gp.sType               = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
            gp.stageCount          = 2;
            gp.pStages             = st;
            gp.pVertexInputState   = &vi;
            gp.pInputAssemblyState = &ia;
            gp.pViewportState      = &vp;
            gp.pRasterizationState = &rs;
            gp.pMultisampleState   = &ms;
            gp.pColorBlendState    = &cb;
            gp.pDynamicState       = &ds;
            gp.layout              = s.ppLayout;
            gp.renderPass          = s.ppPass;
            r = d.CreateGraphicsPipelines(s.device, VK_NULL_HANDLE, 1, &gp, nullptr, &s.ppPipe);
            if (r != VK_SUCCESS)
            {
                s.ppPipe = VK_NULL_HANDLE;
                return PresentFail("vkCreateGraphicsPipelines (present)", r);
            }
        }
        return true;
    }

    /// The present pass onto the acquired swapchain image: false when it could not be recorded (the
    /// caller copies instead).
    bool RecordPresentPass(VkCommandBuffer cmd)
    {
        const context::Handles* h = context::Get();
        if (!h || !EnsurePresentObjects(h->format)) return false;
        const loader::Device& d = *s.dapi;

        if (s.ppFbs[s.image] == VK_NULL_HANDLE)
        {
            if (s.views[s.image] == VK_NULL_HANDLE) return false;
            VkFramebufferCreateInfo fi = {};
            fi.sType           = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
            fi.renderPass      = s.ppPass;
            fi.attachmentCount = 1;
            fi.pAttachments    = &s.views[s.image];
            fi.width           = s.swapExtent.width;
            fi.height          = s.swapExtent.height;
            fi.layers          = 1;
            const VkResult r = d.CreateFramebuffer(s.device, &fi, nullptr, &s.ppFbs[s.image]);
            if (r != VK_SUCCESS)
            {
                s.ppFbs[s.image] = VK_NULL_HANDLE;
                return false;
            }
        }

        // The LUT, when it changed. Staged first: a failure leaves nothing half recorded.
        if (s.lutDirty || s.lutLayout == VK_IMAGE_LAYOUT_UNDEFINED)
        {
            VkBuffer     buffer = VK_NULL_HANDLE;
            VkDeviceSize offset = 0;
            void*        mapped = nullptr;
            if (present::Stage(sizeof s.lut, buffer, offset, mapped))
            {
                std::memcpy(mapped, s.lut, sizeof s.lut);
                // The barrier's source scope (fragment reads) orders the copy after the frames in
                // flight that sample the LUT.
                Barrier(cmd, s.lutImage, VK_IMAGE_ASPECT_COLOR_BIT, 1, 1, s.lutLayout,
                        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, s.lutLayout == VK_IMAGE_LAYOUT_UNDEFINED);
                VkBufferImageCopy c = {};
                c.bufferOffset                = offset;
                c.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
                c.imageSubresource.layerCount = 1;
                c.imageExtent                 = VkExtent3D{ 256, 1, 1 };
                d.CmdCopyBufferToImage(cmd, buffer, s.lutImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &c);
                Barrier(cmd, s.lutImage, VK_IMAGE_ASPECT_COLOR_BIT, 1, 1, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, false);
                s.lutLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                s.lutDirty  = false;
                ++s.stats.lutUploads;
            }
            else if (s.lutLayout == VK_IMAGE_LAYOUT_UNDEFINED)
                return false;   // never uploaded: the set would name an undefined image
        }

        // The set names the back buffer that stands now. It changes only after a rebuild, which waited
        // for the device: no frame still in flight reads the old binding.
        if (s.ppSetBack != s.backView)
        {
            VkDescriptorImageInfo bi = { s.ppSampler, s.backView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
            VkWriteDescriptorSet w = {};
            w.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w.dstSet          = s.ppSet;
            w.dstBinding      = 0;
            w.descriptorCount = 1;
            w.descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            w.pImageInfo      = &bi;
            d.UpdateDescriptorSets(s.device, 1, &w, 0, nullptr);
            s.ppSetBack = s.backView;
        }

        if (s.backLayout == VK_IMAGE_LAYOUT_UNDEFINED)
        {
            // Never drawn into: black, as the copy path makes it.
            Barrier(cmd, s.backImage, VK_IMAGE_ASPECT_COLOR_BIT, 1, 1, VK_IMAGE_LAYOUT_UNDEFINED,
                    VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, true);
            VkClearColorValue black = {};
            black.float32[3] = 1.0f;
            VkImageSubresourceRange range = {};
            range.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            range.levelCount = 1;
            range.layerCount = 1;
            d.CmdClearColorImage(cmd, s.backImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &black, 1, &range);
            s.backLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        }
        if (s.backLayout != VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
            Barrier(cmd, s.backImage, VK_IMAGE_ASPECT_COLOR_BIT, 1, 1, s.backLayout,
                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, false);
        s.backLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        VkRenderPassBeginInfo begin = {};
        begin.sType             = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        begin.renderPass        = s.ppPass;
        begin.framebuffer       = s.ppFbs[s.image];
        begin.renderArea.extent = s.swapExtent;
        d.CmdBeginRenderPass(cmd, &begin, VK_SUBPASS_CONTENTS_INLINE);
        VkViewport viewport = {};
        viewport.width    = static_cast<float>(s.swapExtent.width);
        viewport.height   = static_cast<float>(s.swapExtent.height);
        viewport.maxDepth = 1.0f;
        VkRect2D scissor = {};
        scissor.extent = s.swapExtent;
        d.CmdSetViewport(cmd, 0, 1, &viewport);
        d.CmdSetScissor(cmd, 0, 1, &scissor);
        d.CmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, s.ppPipe);
        d.CmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, s.ppLayout, 0, 1, &s.ppSet, 0, nullptr);
        const float push[4] = { s.gamma ? 1.0f : 0.0f, 0.0f, 0.0f, 0.0f };
        d.CmdPushConstants(cmd, s.ppLayout, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof push, push);
        d.CmdDraw(cmd, 3, 1, 0, 0);
        d.CmdEndRenderPass(cmd);

        const bool scaled = s.extent.width != s.swapExtent.width || s.extent.height != s.swapExtent.height;
        ++s.stats.presentPasses;
        if (scaled) ++s.stats.scaledPresents;
        if (s.gamma) ++s.stats.gammaPresents;
        return true;
    }

    /// Ends, submits and (if an image was acquired) presents the open frame.
    void FinishFrame(bool visible)
    {
        if (!s.open) return;
        const context::Handles* h = context::Get();
        const loader::Device&   d = *s.dapi;
        FrameSlot&              slot = s.slots[s.fi];

        EndPass();
        // The frame's GPU span ends with its last pass: the present's copy and its wait for the image are not the frame's.
        FrameGpuEnd(slot.main, s.fi);
        if (s.activeRun)
        {
            // A query cannot span two submissions' pools: the engine's open one ends here.
            s.activeRun->active = false;
            s.activeRun = nullptr;
            ++s.stats.queryOpenAtPresent;
        }

        // A frame that drew nothing on the back buffer presents it as an earlier frame left it, as
        // D3D's would be.
        if (s.ownBack)
        {
            if (visible && Acquire() && s.backImage != VK_NULL_HANDLE)
            {
                // Scaled (fullscreen at the format's resolution) or with gamma: the present pass;
                // else, and when the pass cannot be had, the copy.
                const bool wantPass = s.gamma || s.extent.width != s.swapExtent.width
                                      || s.extent.height != s.swapExtent.height;
                if (!wantPass || !RecordPresentPass(slot.main))
                {
                    if (wantPass) ++s.stats.presentPassRefused;
                    RecordPresentCopy(slot.main);
                }
            }
        }
        else if (s.acquired)
        {
            // COLOR_ATTACHMENT_OPTIMAL (or never drawn) -> PRESENT_SRC; the present waits on the
            // render-finished semaphore, which carries the memory dependency.
            Barrier(slot.main, s.images[s.image], VK_IMAGE_ASPECT_COLOR_BIT, 1, 1, s.swapLayout,
                    VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, s.swapLayout == VK_IMAGE_LAYOUT_UNDEFINED);
            s.swapLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
        }

        VkCommandBuffer buffers[2];
        uint32_t        count = 0;
        if (s.preBegun)
        {
            EndPrePass(slot);
            VkResult r = d.EndCommandBuffer(slot.pre);
            if (r != VK_SUCCESS) return Fail("vkEndCommandBuffer (pre-pass)", r);
            buffers[count++] = slot.pre;
        }
        VkResult r = d.EndCommandBuffer(slot.main);
        if (r != VK_SUCCESS) return Fail("vkEndCommandBuffer", r);
        buffers[count++] = slot.main;
        FlushRings(slot);

        // Reset right before the submit that signals it: no early return can then leave it unsignalled.
        r = d.ResetFences(s.device, 1, &slot.fence);
        if (r != VK_SUCCESS) return Fail("vkResetFences", r);

        // The own back buffer's first use of the swapchain image is the copy or the present pass's
        // colour output, so both stages wait for the acquire; the direct path's is its render pass.
        const VkPipelineStageFlags wait = s.ownBack ? VK_PIPELINE_STAGE_TRANSFER_BIT |
                                                          VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT
                                                    : VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        VkSubmitInfo submit = {};
        submit.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit.commandBufferCount = count;
        submit.pCommandBuffers    = buffers;
        if (s.acquired)
        {
            submit.waitSemaphoreCount   = 1;
            submit.pWaitSemaphores      = &slot.imageAvailable;
            submit.pWaitDstStageMask    = &wait;
            submit.signalSemaphoreCount = 1;
            submit.pSignalSemaphores    = &s.renderFinished[s.image];
        }
        wxlvk::host::SubmitExtras hostLink;   // WXL_HOST: a composite's timeline wait and signal; else nothing
        wxlvk::host::DecorateSubmit(submit, hostLink);
        r = d.QueueSubmit(h->graphicsQueue, 1, &submit, slot.fence);
        wxlvk::host::AfterSubmit(hostLink, r == VK_SUCCESS);
        if (r != VK_SUCCESS) return Fail("vkQueueSubmit", r);
        if (!s.acquired) ++s.stats.hiddenFrames;
        slot.serial = s.serial;
        s.submitted = s.serial;
        ++s.stats.frames;
        if (s.breaksThisFrame)
        {
            ++s.stats.framesBroken;
            if (s.breaksThisFrame > s.stats.maxBreaks) s.stats.maxBreaks = s.breaksThisFrame;
        }

        if (s.acquired)
        {
            VkPresentInfoKHR present = {};
            present.sType              = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
            present.waitSemaphoreCount = 1;
            present.pWaitSemaphores    = &s.renderFinished[s.image];
            present.swapchainCount     = 1;
            present.pSwapchains        = &h->swapchain;
            present.pImageIndices      = &s.image;
            r = d.QueuePresentKHR(h->presentQueue, &present);
            ++s.presentedCount;
            ++s.stats.presented;
            if (r == VK_ERROR_OUT_OF_DATE_KHR || r == VK_SUBOPTIMAL_KHR) s.rebuild = true;
            else if (r != VK_SUCCESS) Fail("vkQueuePresentKHR", r);
            if (s.presentedCount == 1)
                wxl::ScriptMgr::Log(WXL_LOG_DEBUG, "native: frame: first frame presented on the client "
                                    "window (%ux%u, %u images, depth %s, %s)", s.extent.width,
                                    s.extent.height, s.imageCount,
                                    s.depthFormat == VK_FORMAT_D32_SFLOAT ? "D32" :
                                    s.depthFormat == VK_FORMAT_UNDEFINED ? "none" : "D24S8",
                                    s.ownBack ? "own back buffer" : "straight into the swapchain image");
        }

        if (s.drewThisFrame) s.everDrew = true;
        s.open = false;
        detail::serialNow = s.serial + 1;   // Serial() names the next frame until it opens
        s.fi   = (s.fi + 1) % kFramesInFlight;
    }

    bool SameTarget(const TargetDesc& a, const TargetDesc& b)
    {
        return a.colour == b.colour && a.colourView == b.colourView && a.depth == b.depth
               && a.depthImage == b.depthImage && a.depthView == b.depthView && a.extent.width == b.extent.width
               && a.extent.height == b.extent.height && a.invalid == b.invalid && a.colourStamp == b.colourStamp
               && a.depthStamp == b.depthStamp;
    }
}
