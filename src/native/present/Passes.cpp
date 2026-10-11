// The render pass classes, the clearing passes and the framebuffers.
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
    // --- the passes -----------------------------------------------------------------------------------

    VkFormat PickDepth(const context::Handles& h)
    {
        if (!h.iapi || !h.iapi->GetPhysicalDeviceFormatProperties) return VK_FORMAT_UNDEFINED;
        // D32 first: nothing native touches stencil (the engine never clears it, cgxdevice 9.1),
        // and D32 is the format a Vulkan 1.0 device most commonly offers as an attachment.
        const VkFormat candidates[] = { VK_FORMAT_D32_SFLOAT, VK_FORMAT_D24_UNORM_S8_UINT };
        for (VkFormat f : candidates)
        {
            VkFormatProperties fp = {};
            h.iapi->GetPhysicalDeviceFormatProperties(h.physical, f, &fp);
            if (fp.optimalTilingFeatures & VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT) return f;
        }
        return VK_FORMAT_UNDEFINED;
    }

    void EnsureDepthFormat()
    {
        if (s.depthFormat != VK_FORMAT_UNDEFINED) return;
        static bool warned = false;
        if (const context::Handles* h = context::Get()) s.depthFormat = PickDepth(*h);
        if (s.depthFormat == VK_FORMAT_UNDEFINED && !warned)
        {
            warned = true;
            wxl::ScriptMgr::Log(WXL_LOG_WARN, "native: frame: no depth attachment format; passes have "
                                "no depth, depth clears are dropped and 3D draws will overdraw");
        }
    }

    /**
     * The render pass of one class. Every pass is LOAD / STORE with its attachments in their
     * attachment layouts at both ends: the engine clears explicitly (slot 39), first-use discards and
     * clears are recorded by BeginPass, and every layout change around a pass is a barrier of its own.
     * Load ops and layouts do not take part in render-pass compatibility, so this one pass per class
     * is also what pipelines of that class are built against.
     *
     * The external dependency orders the pass's attachment accesses after earlier attachment writes
     * and transfers on the same images (a pass resumed after an inline transfer, the other frame in
     * flight's depth writes) -- the cases where no layout change, so no barrier, sits in between.
     */
    VkRenderPass MakePass(VkFormat colour, VkFormat depth, bool clearColour, bool clearDepth, bool gbuffer, bool clearGBuffer)
    {
        // The attachment order of every class: the colour, a G-buffer class's three, the depth.
        VkAttachmentDescription a[2 + kGBufferAttachments] = {};
        a[0].format         = colour;
        a[0].samples        = VK_SAMPLE_COUNT_1_BIT;
        a[0].loadOp         = clearColour ? VK_ATTACHMENT_LOAD_OP_CLEAR : VK_ATTACHMENT_LOAD_OP_LOAD;
        a[0].storeOp        = VK_ATTACHMENT_STORE_OP_STORE;
        a[0].stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        a[0].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        a[0].initialLayout  = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        a[0].finalLayout    = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

        // A G-buffer class: its three attachments after the colour, in GENERAL for the device's life (the
        // consumers sample them between passes with no layout change; Pass.cpp orders the accesses).
        uint32_t n = 1;
        VkAttachmentReference colourRefs[1 + kGBufferAttachments] = {};
        colourRefs[0] = { 0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL };
        if (gbuffer)
        {
            for (uint32_t i = 0; i < kGBufferAttachments; ++i, ++n)
            {
                a[n].format         = s.gbufFormat[i];
                a[n].samples        = VK_SAMPLE_COUNT_1_BIT;
                a[n].loadOp         = clearGBuffer ? VK_ATTACHMENT_LOAD_OP_CLEAR : VK_ATTACHMENT_LOAD_OP_LOAD;
                a[n].storeOp        = VK_ATTACHMENT_STORE_OP_STORE;
                a[n].stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
                a[n].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
                a[n].initialLayout  = VK_IMAGE_LAYOUT_GENERAL;
                a[n].finalLayout    = VK_IMAGE_LAYOUT_GENERAL;
                colourRefs[1 + i]   = { n, VK_IMAGE_LAYOUT_GENERAL };
            }
        }

        const bool hasDepth = depth != VK_FORMAT_UNDEFINED;
        VkAttachmentReference depthRef = { n, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL };
        if (hasDepth)
        {
            a[n].format         = depth;
            a[n].samples        = VK_SAMPLE_COUNT_1_BIT;
            a[n].loadOp         = clearDepth ? VK_ATTACHMENT_LOAD_OP_CLEAR : VK_ATTACHMENT_LOAD_OP_LOAD;
            a[n].storeOp        = VK_ATTACHMENT_STORE_OP_STORE;
            a[n].stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_LOAD;
            a[n].stencilStoreOp = VK_ATTACHMENT_STORE_OP_STORE;
            a[n].initialLayout  = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
            a[n].finalLayout    = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
            ++n;
        }

        VkSubpassDescription sub = {};
        sub.pipelineBindPoint       = VK_PIPELINE_BIND_POINT_GRAPHICS;
        sub.colorAttachmentCount    = gbuffer ? 1 + kGBufferAttachments : 1;
        sub.pColorAttachments       = colourRefs;
        sub.pDepthStencilAttachment = hasDepth ? &depthRef : nullptr;

        const VkPipelineStageFlags stages = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                                            VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT |
                                            VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
        VkSubpassDependency dep = {};
        dep.srcSubpass    = VK_SUBPASS_EXTERNAL;
        dep.dstSubpass    = 0;
        dep.srcStageMask  = stages | VK_PIPELINE_STAGE_TRANSFER_BIT;
        dep.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT |
                            VK_ACCESS_TRANSFER_WRITE_BIT;
        dep.dstStageMask  = stages;
        dep.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                            VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
                            VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

        VkRenderPassCreateInfo info = {};
        info.sType           = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        info.attachmentCount = n;
        info.pAttachments    = a;
        info.subpassCount    = 1;
        info.pSubpasses      = &sub;
        info.dependencyCount = 1;
        info.pDependencies   = &dep;

        VkRenderPass pass = VK_NULL_HANDLE;
        const VkResult r = s.dapi->CreateRenderPass(s.device, &info, nullptr, &pass);
        if (r != VK_SUCCESS)
        {
            Fail("vkCreateRenderPass", r);
            return VK_NULL_HANDLE;
        }
        return pass;
    }

    /// The pass of a class, made on first use and kept for the device's life (a pipeline's cache
    /// key holds the handle, so a pass is never destroyed and its handle never reused).
    VkRenderPass ClassPass(VkFormat colour, VkFormat depth, bool gbuffer)
    {
        for (uint32_t i = 0; i < s.classCount; ++i)
            if (s.classes[i].colour == colour && s.classes[i].depth == depth && s.classes[i].gbuffer == gbuffer)
                return s.classes[i].pass;
        if (s.classCount == kMaxPassClasses)
        {
            static bool logged = false;
            if (!logged)
            {
                logged = true;
                wxl::ScriptMgr::Log(WXL_LOG_WARN, "native: frame: more than %u render pass classes; a "
                                    "target of %s / %s is not drawn (logged once)", kMaxPassClasses,
                                    wxlvk::vk::FormatName(colour), wxlvk::vk::FormatName(depth));
            }
            return VK_NULL_HANDLE;
        }
        const VkRenderPass pass = MakePass(colour, depth, false, false, gbuffer, false);
        if (pass == VK_NULL_HANDLE) return VK_NULL_HANDLE;
        PassClass c;
        c.colour      = colour;
        c.depth       = depth;
        c.gbuffer     = gbuffer;
        c.pass        = pass;
        c.clearing[0] = pass;
        s.classes[s.classCount++] = c;
        s.stats.passClasses = s.classCount;
        wxl::ScriptMgr::Log(WXL_LOG_DEBUG, "native: frame: render pass class #%u: colour %s, depth %s%s",
                            s.classCount, wxlvk::vk::FormatName(colour),
                            depth == VK_FORMAT_UNDEFINED ? "none" : wxlvk::vk::FormatName(depth),
                            gbuffer ? ", with the G-buffer's three attachments" : "");
        // The pipelines the last session built for this class, built now rather than at their
        // first draws (vulkan/Pipelines.cpp, pipeline-keys.bin).
        pipelines::PassCreated(pass, colour, depth, gbuffer);
        return pass;
    }

    bool IsGBufferPass(VkRenderPass pass)
    {
        for (uint32_t i = 0; i < s.classCount; ++i)
            if (s.classes[i].pass == pass) return s.classes[i].gbuffer;
        return false;
    }

    /**
     * The class pass with CLEAR load ops, made on first use: `which` bit 0 the colour, bit 1 the
     * depth. Falls back to the LOAD pass when it cannot be made, so the caller can always begin.
     *
     * @param VkRenderPass base : a class's LOAD pass (ClassPass)
     * @param uint32 which : 1 .. 7 (bit 2: the G-buffer's attachments, on a G-buffer class)
     * @param bool& clearing : receives whether the returned pass clears
     * @return VkRenderPass pass
     */
    VkRenderPass ClearingPass(VkRenderPass base, uint32_t which, bool& clearing)
    {
        clearing = false;
        for (uint32_t i = 0; i < s.classCount; ++i)
        {
            PassClass& c = s.classes[i];
            if (c.pass != base) continue;
            if (c.clearing[which] == VK_NULL_HANDLE)
                c.clearing[which] = MakePass(c.colour, c.depth, (which & 1u) != 0, (which & 2u) != 0, c.gbuffer, (which & 4u) != 0);
            if (c.clearing[which] == VK_NULL_HANDLE) return base;
            clearing = true;
            return c.clearing[which];
        }
        return base;
    }

    void RemoveFramebuffer(uint32_t i, uint64_t lastUsed, bool now)
    {
        Framebuffer& f = s.fbs[i];
        if (f.fb != VK_NULL_HANDLE)
        {
            if (now)
                s.dapi->DestroyFramebuffer(s.device, f.fb, nullptr);
            else
            {
                Retired e;
                e.kind   = Kind::Framebuffer;
                e.fb     = f.fb;
                e.serial = lastUsed > f.lastUsed ? lastUsed : f.lastUsed;
                Retire(e);
            }
        }
        s.fbs[i] = s.fbs[s.fbCount - 1];
        s.fbs[s.fbCount - 1] = Framebuffer();
        --s.fbCount;
        s.stats.framebuffers = s.fbCount;
    }

    /// Every framebuffer naming `view` goes: by serial, or at once when the device is idle.
    void DropFramebuffersNaming(VkImageView view, uint64_t lastUsed, bool now)
    {
        if (view == VK_NULL_HANDLE) return;
        for (uint32_t i = 0; i < s.fbCount;)
        {
            if (s.fbs[i].colour == view || s.fbs[i].depth == view)
                RemoveFramebuffer(i, lastUsed, now);
            else
                ++i;
        }
    }

    VkFramebuffer FramebufferFor(VkRenderPass pass, VkImageView colour, VkImageView depth, VkExtent2D extent)
    {
        for (uint32_t i = 0; i < s.fbCount; ++i)
        {
            Framebuffer& f = s.fbs[i];
            if (f.pass == pass && f.colour == colour && f.depth == depth && f.extent.width == extent.width
                && f.extent.height == extent.height)
            {
                f.lastUsed = s.serial;
                return f.fb;
            }
        }
        if (s.fbCount == kMaxFramebuffers)
        {
            // Full: the least recently used goes, by serial (it may be in this very frame).
            uint32_t oldest = 0;
            for (uint32_t i = 1; i < s.fbCount; ++i)
                if (s.fbs[i].lastUsed < s.fbs[oldest].lastUsed) oldest = i;
            RemoveFramebuffer(oldest, 0, false);
        }
        // The class's attachment order: the colour, a G-buffer class's three, the depth.
        VkImageView attachments[2 + kGBufferAttachments] = {};
        uint32_t    count = 0;
        attachments[count++] = colour;
        if (IsGBufferPass(pass))
            for (uint32_t i = 0; i < kGBufferAttachments; ++i) attachments[count++] = s.gbufView[i];
        if (depth != VK_NULL_HANDLE) attachments[count++] = depth;
        VkFramebufferCreateInfo info = {};
        info.sType           = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        info.renderPass      = pass;
        info.attachmentCount = count;
        info.pAttachments    = attachments;
        info.width           = extent.width;
        info.height          = extent.height;
        info.layers          = 1;
        VkFramebuffer fb = VK_NULL_HANDLE;
        const VkResult r = s.dapi->CreateFramebuffer(s.device, &info, nullptr, &fb);
        if (r != VK_SUCCESS)
        {
            Fail("vkCreateFramebuffer", r);
            return VK_NULL_HANDLE;
        }
        Framebuffer& f = s.fbs[s.fbCount++];
        f.pass     = pass;
        f.colour   = colour;
        f.depth    = depth;
        f.extent   = extent;
        f.fb       = fb;
        f.lastUsed = s.serial;
        ++s.stats.framebuffersMade;
        s.stats.framebuffers = s.fbCount;
        return fb;
    }
}
