// Opening and closing the engine's pass, the pre-pass, the present copy.
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
    // --- the pass ------------------------------------------------------------------------------------

    bool GBufferWanted()
    {
        // The G-buffer class is for the opaque world's pass: a consumer asks, the setting allows, the
        // images were not refused, and the engine is between its world scene pass's start and
        // AfterOpaqueWorld. Before the world (the engine's clear), after the point (the transparents,
        // the consumers' own draws, the interface): the plain class, whose pipelines name one attachment.
        return s.gbufRequests > 0 && !s.gbufFailed && s.worldOpen && !s.opaqueDone && wxlvk::settings::GBufferAllowed();
    }

    void EndPass()
    {
        if (!s.passOpen) return;
        VkCommandBuffer cmd = s.slots[s.fi].main;
        if (s.activeRun) QueryEndVk(*s.activeRun);
        s.dapi->CmdEndRenderPass(cmd);
        s.passOpen = false;
        if (s.passGBuffer)
        {
            // The G-buffer's writes, visible to the consumers' reads (a MidPass, a stage: vertex, fragment
            // and compute) and ordered before the next pass's writes. The images stay in GENERAL.
            VkMemoryBarrier m = {};
            m.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
            m.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
            m.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
            s.dapi->CmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                                       VK_PIPELINE_STAGE_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT |
                                           VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                                       0, 1, &m, 0, nullptr, 0, nullptr);
            ++s.stats.barrierCalls;
            s.passGBuffer = false;
        }
        // A texture target is sampled next, or copied: SHADER_READ_ONLY is where everything finds it.
        // Both attachments in one barrier call.
        Barriers out;
        if (s.passColour.tex)
        {
            out.Add(s.passColour.image, s.passColour.aspect, s.passColour.levels, s.passColour.layers,
                    VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, false);
            *s.passColour.layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        }
        if (s.passHasDepth && s.passDepth.tex)
        {
            out.Add(s.passDepth.image, s.passDepth.aspect, s.passDepth.levels, s.passDepth.layers,
                    VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, false);
            *s.passDepth.layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        }
        out.Record(cmd);
        s.passColour = Side();
        s.passDepth  = Side();
    }

    /**
     * Begins a pass on the current target. Null when it cannot be drawn into this frame.
     *
     * The clears due at the pass's start become its load ops where they cover the whole area: a
     * first use (black, the far plane) always, and `pending` -- the engine's slot-39 clear that
     * asked for this pass -- when its rectangle is the pass area. CLEAR then replaces LOAD for that
     * attachment, with the value the clears recorded in order would have left (the engine's over a
     * first use). A depth format with stencil keeps its clears inside the pass (the load op would
     * need the stencil's too), and so does any clear of less than the area.
     *
     * @param PassClear* pending : the slot-39 clear, or null
     * @param uint32* folded : receives the aspects of `pending` the load op took (with pending)
     */
    VkCommandBuffer BeginPass(const present::PassClear* pending, uint32_t* folded)
    {
        if (folded) *folded = 0;
        const TargetDesc& t   = s.target;
        VkCommandBuffer   cmd = s.slots[s.fi].main;
        if (t.invalid)
        {
            ++s.stats.noTarget;
            return nullptr;
        }
        Side c, d;
        if (!t.colour)
        {
            if (!BackColour(c)) return nullptr;
        }
        else
        {
            if (t.colourView == VK_NULL_HANDLE || t.colour->handle == VK_NULL_HANDLE)
            {
                ++s.stats.noTarget;
                return nullptr;
            }
            TextureSide(t.colour, t.colourView, t.colourStamp, c);
        }
        VkExtent2D area = c.extent;
        if (t.extent.width && t.extent.height) area = Min(area, t.extent);

        bool hasDepth = false;
        switch (t.depth)
        {
            case TargetDesc::Depth::Back:      hasDepth = BackDepth(d); break;
            case TargetDesc::Depth::Offscreen: hasDepth = Offscreen(d, area); break;
            case TargetDesc::Depth::Texture:
                if (t.depthImage && t.depthImage->handle != VK_NULL_HANDLE && t.depthView != VK_NULL_HANDLE)
                {
                    TextureSide(t.depthImage, t.depthView, t.depthStamp, d);
                    hasDepth = true;
                }
                break;
            case TargetDesc::Depth::None: break;
        }
        if (hasDepth) area = Min(area, d.extent);
        if (area.width == 0 || area.height == 0)
        {
            ++s.stats.noTarget;
            return nullptr;
        }

        // The G-buffer class for the opaque world's pass on the back buffer, while a consumer asks and the
        // images can be had (docs/architecture.md "The G-buffer").
        const bool gbuffer = c.back && hasDepth && d.back && GBufferWanted() && EnsureGBufferImages();
        const VkRenderPass pass = ClassPass(c.format, hasDepth ? d.format : VK_FORMAT_UNDEFINED, gbuffer);
        if (pass == VK_NULL_HANDLE) return nullptr;
        const VkFramebuffer fb = FramebufferFor(pass, c.view, hasDepth ? d.view : VK_NULL_HANDLE, area);
        if (fb == VK_NULL_HANDLE) return nullptr;

        Barriers in;
        if (gbuffer && (s.gbufLayout == VK_IMAGE_LAYOUT_UNDEFINED || s.gbufFresh))
        {
            // The G-buffer's attachments stay in GENERAL: their first pass takes them from UNDEFINED (the
            // load op clears what they hold); the frame's first G-buffer pass orders itself after the reads
            // of the frame before -- a consumer's stage, the other frame in flight -- through the layout's
            // full scope. The frame's later G-buffer passes need nothing here: EndPass's barrier already
            // orders each one after the one before (the full scope would drain the GPU at every pass break).
            const bool first = s.gbufLayout == VK_IMAGE_LAYOUT_UNDEFINED;
            for (uint32_t i = 0; i < kGBufferAttachments; ++i)
                in.Add(s.gbufImage[i], VK_IMAGE_ASPECT_COLOR_BIT, 1, 1,
                       first ? VK_IMAGE_LAYOUT_UNDEFINED : VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL, first);
            s.gbufLayout = VK_IMAGE_LAYOUT_GENERAL;
        }
        if (*c.layout != VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL || c.discard)
            in.Add(c.image, c.aspect, c.levels, c.layers, *c.layout, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, c.discard);
        *c.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        if (hasDepth)
        {
            if (*d.layout != VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL || d.discard)
                in.Add(d.image, d.aspect, d.levels, d.layers, *d.layout, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                       d.discard);
            *d.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
            if (d.back) s.depthFresh = false;
        }
        in.Record(cmd);

        // The clears due at the start: first uses (black and the far plane, for a frame the engine
        // did not clear itself) and the engine's own clear that asked for this pass. Each becomes
        // the attachment's load op when it covers the area; what cannot is recorded in the pass,
        // first uses before the engine's, as they would have been.
        const bool depthPlain = hasDepth && (d.aspect & VK_IMAGE_ASPECT_STENCIL_BIT) == 0;
        // The clear values follow the class's attachment order: the colour, a G-buffer class's three
        // (zero: class 0, no normal), the depth. The G-buffer is cleared by the frame's first pass of
        // its class and kept by the later ones.
        const uint32_t depthAt = gbuffer ? 1 + kGBufferAttachments : 1;
        VkClearValue values[2 + kGBufferAttachments] = {};
        values[0].color.float32[3]         = 1.0f;
        values[depthAt].depthStencil.depth = 1.0f;
        bool clearColour  = c.clear;
        bool clearDepth   = hasDepth && d.clear && depthPlain;
        const bool clearGBuffer = gbuffer && s.gbufFresh;
        uint32_t engineFolded = 0;
        if (pending && pending->place && folded)
        {
            VkRect2D rect = {};
            if (pending->place(pending->device, area, rect) && rect.offset.x == 0 && rect.offset.y == 0
                && rect.extent.width == area.width && rect.extent.height == area.height)
            {
                if (pending->mask & 1u)
                {
                    clearColour = true;
                    for (uint32_t i = 0; i < 4; ++i) values[0].color.float32[i] = pending->colour[i];
                    engineFolded |= 1u;
                }
                if ((pending->mask & 2u) && depthPlain)
                {
                    clearDepth = true;
                    values[depthAt].depthStencil.depth = 1.0f;
                    engineFolded |= 2u;
                }
            }
        }
        const uint32_t which    = (clearColour ? 1u : 0u) | (clearDepth ? 2u : 0u) | (clearGBuffer ? 4u : 0u);
        bool           clearing = false;
        const VkRenderPass beginPass = which ? ClearingPass(pass, which, clearing) : pass;
        if (clearing)
        {
            // What the load op took: the caller records the engine's clear only for the rest.
            if (folded) *folded = engineFolded;
            if (c.clear) ++s.stats.firstUseFolded;
            if (hasDepth && d.clear && clearDepth) ++s.stats.firstUseFolded;
            s.stats.engineClearsFolded += (engineFolded & 1u) + ((engineFolded >> 1) & 1u);
        }

        VkRenderPassBeginInfo begin = {};
        begin.sType             = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        begin.renderPass        = beginPass;
        begin.framebuffer       = fb;
        begin.renderArea.extent = area;
        begin.clearValueCount   = clearing ? depthAt + (hasDepth ? 1u : 0u) : 0u;
        begin.pClearValues      = clearing ? values : nullptr;
        s.dapi->CmdBeginRenderPass(cmd, &begin, VK_SUBPASS_CONTENTS_INLINE);
        if (clearing) ++s.stats.clearLoadPasses;
        // A pass may follow a flush (a fresh command buffer) or the present pass's binds: the draws'
        // bind tracking starts over.
        pipelines::InvalidateBindings();

        // The first-use clears the load op did not take.
        VkClearAttachment a[2] = {};
        uint32_t          n    = 0;
        if (c.clear && !(clearing && clearColour))
        {
            a[n].aspectMask                  = VK_IMAGE_ASPECT_COLOR_BIT;
            a[n].colorAttachment             = 0;
            a[n].clearValue.color.float32[3] = 1.0f;
            ++n;
        }
        if (hasDepth && d.clear && !(clearing && clearDepth))
        {
            a[n].aspectMask                      = d.aspect;
            a[n].clearValue.depthStencil.depth   = 1.0f;
            a[n].clearValue.depthStencil.stencil = 0;
            ++n;
        }
        if (n)
        {
            VkClearRect r = {};
            r.rect.extent = area;
            r.layerCount  = 1;
            s.dapi->CmdClearAttachments(cmd, n, a, 1, &r);
        }
        if (s.breakView != VK_NULL_HANDLE)
        {
            // The pass after a break: on the same view it reloads what the break stored.
            if (s.breakView == c.view) ++s.stats.breaksResumed;
            s.breakView = VK_NULL_HANDLE;
        }

        if (c.stamp) *c.stamp = s.serial;
        if (hasDepth && d.stamp) *d.stamp = s.serial;
        if (c.back || (hasDepth && d.back)) s.backTouched = true;
        s.passColour   = c;
        s.passDepth    = hasDepth ? d : Side();
        s.passHasDepth = hasDepth;
        s.passGBuffer  = gbuffer;
        s.passClass    = pass;
        s.passExtent   = area;
        s.passOpen     = true;
        ++s.stats.passes;
        if (gbuffer)
        {
            s.gbufFresh  = false;
            s.gbufDrawn  = true;
            s.gbufSerial = s.serial;
            ++s.stats.gbufferPasses;
        }
        if (t.colour) ++s.stats.targetPasses;
        if (s.activeRun && s.activeRun->active) QueryBeginVk(*s.activeRun);
        return cmd;
    }

    /// The pre-pass buffer's end: its copies made visible to every reader in the main buffer.
    void EndPrePass(FrameSlot& slot)
    {
        VkMemoryBarrier m = {};
        m.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        m.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        m.dstAccessMask = VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT | VK_ACCESS_INDEX_READ_BIT |
                          VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT |
                          VK_ACCESS_TRANSFER_WRITE_BIT;
        s.dapi->CmdPipelineBarrier(slot.pre, VK_PIPELINE_STAGE_TRANSFER_BIT,
                                   VK_PIPELINE_STAGE_VERTEX_INPUT_BIT | VK_PIPELINE_STAGE_VERTEX_SHADER_BIT |
                                       VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                                   0, 1, &m, 0, nullptr, 0, nullptr);
    }

    /// The present's copy of the own back buffer onto the acquired swapchain image.
    void RecordPresentCopy(VkCommandBuffer cmd)
    {
        const VkImage swap = s.images[s.image];
        if (s.backLayout == VK_IMAGE_LAYOUT_UNDEFINED)
        {
            // Never drawn into: black.
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
        if (s.backLayout != VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL)
            Barrier(cmd, s.backImage, VK_IMAGE_ASPECT_COLOR_BIT, 1, 1, s.backLayout,
                    VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, false);
        s.backLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;

        // The swapchain image: its contents are discarded; the submit waits on the acquire at the
        // TRANSFER stage, which this barrier's source stage names, so the two chain.
        VkImageMemoryBarrier b = {};
        b.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        b.srcAccessMask       = 0;
        b.dstAccessMask       = VK_ACCESS_TRANSFER_WRITE_BIT;
        b.oldLayout           = VK_IMAGE_LAYOUT_UNDEFINED;
        b.newLayout           = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.image               = swap;
        b.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        b.subresourceRange.levelCount = 1;
        b.subresourceRange.layerCount = 1;
        s.dapi->CmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0,
                                   nullptr, 0, nullptr, 1, &b);

        // Same format and, between rebuilds, the same extent: a straight copy. Never past either
        // image (the present pass's fallback when the two differ crops rather than overruns).
        VkImageCopy copy = {};
        copy.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        copy.srcSubresource.layerCount = 1;
        copy.dstSubresource            = copy.srcSubresource;
        copy.extent                    = VkExtent3D{ s.extent.width < s.swapExtent.width ? s.extent.width : s.swapExtent.width,
                                                     s.extent.height < s.swapExtent.height ? s.extent.height : s.swapExtent.height, 1 };
        s.dapi->CmdCopyImage(cmd, s.backImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, swap,
                             VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);
        ++s.stats.presentCopies;

        b.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        b.dstAccessMask = 0;
        b.oldLayout     = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        b.newLayout     = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
        s.dapi->CmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0,
                                   0, nullptr, 0, nullptr, 1, &b);
    }
}
