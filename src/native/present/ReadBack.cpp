// The present path's targets, copies and read-backs.
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

using namespace wxlvk::native::present::detail;

namespace wxlvk::native::present
{
    // --- targets, copies, read-back -------------------------------------------------------------------

    void SetTarget(const TargetDesc& t)
    {
        if (SameTarget(t, s.target)) return;
        EndPass();
        s.target = t;
    }

    const TargetDesc& Target() { return s.target; }

    bool OnBackBuffer() { return s.passOpen ? s.passColour.back : s.target.colour == nullptr; }

    void ForgetImage(const texture::Image* image)
    {
        if (!image) return;
        if (s.passOpen && (s.passColour.tex == image || (s.passHasDepth && s.passDepth.tex == image))) EndPass();
        if (s.target.colour == image || s.target.depthImage == image)
        {
            s.target.invalid     = true;
            s.target.colour      = s.target.colour == image ? nullptr : s.target.colour;
            s.target.depthImage  = s.target.depthImage == image ? nullptr : s.target.depthImage;
            s.target.colourStamp = nullptr;
            s.target.depthStamp  = nullptr;
        }
    }

    void ForgetView(VkImageView view, uint64_t lastUsed)
    {
        if (!s.dapi) return;
        DropFramebuffersNaming(view, lastUsed, false);
    }

    void RetireView(VkImageView view, uint64_t lastUsed)
    {
        if (view == VK_NULL_HANDLE || !s.dapi) return;
        DropFramebuffersNaming(view, lastUsed, false);
        Retired e;
        e.kind   = Kind::View;
        e.view   = view;
        e.serial = lastUsed;
        Retire(e);
    }

    VkFormat DepthTextureFormat()
    {
        const context::Handles* h = context::Get();
        if (!h || !h->iapi || !h->iapi->GetPhysicalDeviceFormatProperties) return VK_FORMAT_UNDEFINED;
        EnsureDepthFormat();
        const VkFormatFeatureFlags need = VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT |
                                          VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT;
        // The back depth's own format first (one pass class with the back buffer when a depth
        // texture is drawn with the back colour); single-aspect only, see texture::CreateTarget.
        const VkFormat candidates[] = { s.depthFormat, VK_FORMAT_D32_SFLOAT, VK_FORMAT_D16_UNORM };
        for (VkFormat f : candidates)
        {
            if (f != VK_FORMAT_D32_SFLOAT && f != VK_FORMAT_D16_UNORM) continue;
            VkFormatProperties fp = {};
            h->iapi->GetPhysicalDeviceFormatProperties(h->physical, f, &fp);
            if ((fp.optimalTilingFeatures & need) == need) return f;
        }
        return VK_FORMAT_UNDEFINED;
    }

    bool BackBufferReadable()
    {
        if (!s.ownBack || s.backImage == VK_NULL_HANDLE)
        {
            const context::Handles* h = context::Get();
            if (h && h->swapchain != VK_NULL_HANDLE && h->extent.width && h->extent.height && EnsureFrame())
                EnsureImageObjects(*h);
        }
        return s.ownBack && s.backImage != VK_NULL_HANDLE;
    }

    bool BackBuffer(VkCommandBuffer cmd, VkImageLayout layout, VkImage& image, VkFormat& format, VkExtent2D& extent)
    {
        if (!cmd || !BackBufferReadable()) return false;
        if (s.passOpen && s.passColour.back) EndPass();   // the caller's Outside() did this already
        if (s.backLayout == VK_IMAGE_LAYOUT_UNDEFINED)
        {
            // Never drawn since it was (re)made: a copy or a read-back of it would hand over whatever
            // the memory held. Black, as a first pass on it and the present copy make it.
            Barrier(cmd, s.backImage, VK_IMAGE_ASPECT_COLOR_BIT, 1, 1, VK_IMAGE_LAYOUT_UNDEFINED,
                    VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, true);
            VkClearColorValue black = {};
            black.float32[3] = 1.0f;
            VkImageSubresourceRange range = {};
            range.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            range.levelCount = 1;
            range.layerCount = 1;
            s.dapi->CmdClearColorImage(cmd, s.backImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &black, 1, &range);
            s.backLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        }
        if (s.backLayout != layout)
            Barrier(cmd, s.backImage, VK_IMAGE_ASPECT_COLOR_BIT, 1, 1, s.backLayout, layout,
                    s.backLayout == VK_IMAGE_LAYOUT_UNDEFINED);
        s.backLayout  = layout;
        s.backTouched = true;
        image  = s.backImage;
        format = s.colourFormat;
        extent = s.extent;
        return true;
    }

    bool BackBufferInfo(VkFormat& format, VkExtent2D& extent)
    {
        if (!s.ownBack || s.backImage == VK_NULL_HANDLE || s.noImage) return false;
        format = s.colourFormat;
        extent = s.extent;
        return true;
    }

    bool SceneDepth(VkCommandBuffer cmd, VkImageLayout layout, VkImageView& view, VkFormat& format)
    {
        if (!cmd || !s.depthSampled || s.depthView == VK_NULL_HANDLE) return false;
        // Not written this frame (the next pass would discard it) or never: nothing to read.
        if (s.depthFresh || s.depthLayout == VK_IMAGE_LAYOUT_UNDEFINED) return false;
        if (s.passOpen && s.passHasDepth && s.passDepth.back) EndPass();
        if (s.depthLayout != layout)
            Barrier(cmd, s.depthImage, AspectOf(s.depthFormat), 1, 1, s.depthLayout, layout, false);
        s.depthLayout = layout;
        s.backTouched = true;
        view   = s.depthView;
        format = s.depthFormat;
        return true;
    }

    void Transition(VkCommandBuffer cmd, texture::Image& image, VkImageLayout to)
    {
        if (!cmd || image.handle == VK_NULL_HANDLE || image.layout == to) return;
        Barrier(cmd, image.handle, AspectOf(image.format), image.levels, image.layers, image.layout, to,
                image.layout == VK_IMAGE_LAYOUT_UNDEFINED);
        image.layout = to;
    }

    bool FlushAndWait()
    {
        if (!s.open) return true;
        if (s.failed) return false;
        wxlvk::native::perf::Scope timed(wxlvk::native::perf::kFlush);
        if (s.acquired)
        {
            // The direct path: a submit now would have to consume the acquire's semaphore.
            ++s.stats.flushRefused;
            return false;
        }
        const context::Handles* h = context::Get();
        const loader::Device&   d = *s.dapi;
        FrameSlot&              slot = s.slots[s.fi];
        if (s.passOpen)
        {
            s.breakView = s.passColour.view;
            EndPass();
            ++s.stats.passBreaks;
            ++s.breaksThisFrame;
        }
        VkCommandBuffer buffers[2];
        uint32_t        count = 0;
        if (s.preBegun)
        {
            EndPrePass(slot);
            VkResult r = d.EndCommandBuffer(slot.pre);
            if (r != VK_SUCCESS)
            {
                Fail("vkEndCommandBuffer (pre-pass, flush)", r);
                return false;
            }
            buffers[count++] = slot.pre;
        }
        // Everything the GPU wrote is made visible to the host reads that follow the wait.
        VkMemoryBarrier m = {};
        m.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        m.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        m.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
        d.CmdPipelineBarrier(slot.main, VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                             VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &m, 0, nullptr, 0, nullptr);
        VkResult r = d.EndCommandBuffer(slot.main);
        if (r != VK_SUCCESS)
        {
            Fail("vkEndCommandBuffer (flush)", r);
            return false;
        }
        buffers[count++] = slot.main;
        FlushRings(slot);
        r = d.ResetFences(s.device, 1, &slot.fence);
        if (r != VK_SUCCESS)
        {
            Fail("vkResetFences (flush)", r);
            return false;
        }
        VkSubmitInfo submit = {};
        submit.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit.commandBufferCount = count;
        submit.pCommandBuffers    = buffers;
        wxlvk::host::SubmitExtras hostLink;   // WXL_HOST: a composite's timeline wait and signal; else nothing
        wxlvk::host::DecorateSubmit(submit, hostLink);
        r = d.QueueSubmit(h->graphicsQueue, 1, &submit, slot.fence);
        wxlvk::host::AfterSubmit(hostLink, r == VK_SUCCESS);
        if (r != VK_SUCCESS)
        {
            Fail("vkQueueSubmit (flush)", r);
            return false;
        }
        r = d.WaitForFences(s.device, 1, &slot.fence, VK_TRUE, UINT64_MAX);
        if (r != VK_SUCCESS)
        {
            Fail("vkWaitForFences (flush)", r);
            return false;
        }
        ++s.stats.flushes;
        // The same frame goes on: same serial, same slot, its pools and staging spans untouched (the
        // GPU is done with them, nothing reset them).
        VkCommandBufferBeginInfo begin = {};
        begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        r = d.BeginCommandBuffer(slot.main, &begin);
        if (r != VK_SUCCESS)
        {
            Fail("vkBeginCommandBuffer (after a flush)", r);
            return false;
        }
        s.preBegun = false;
        return true;
    }
}
