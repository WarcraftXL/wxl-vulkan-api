// The swapchain-derived objects: the back buffer, the depth, the acquire.
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
    // --- swapchain-derived objects --------------------------------------------------------------------

    /// One device-local image and view of our own (the back buffer, the depths). False: nothing left.
    bool MakeRaw(VkFormat format, VkExtent2D extent, VkImageUsageFlags usage, VkImage& image, VkImageView& view,
                 memory::Block& block, const char* what)
    {
        const loader::Device& d = *s.dapi;
        VkImageCreateInfo info = {};
        info.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        info.imageType     = VK_IMAGE_TYPE_2D;
        info.format        = format;
        info.extent        = VkExtent3D{ extent.width, extent.height, 1 };
        info.mipLevels     = 1;
        info.arrayLayers   = 1;
        info.samples       = VK_SAMPLE_COUNT_1_BIT;
        info.tiling        = VK_IMAGE_TILING_OPTIMAL;
        info.usage         = usage;
        info.sharingMode   = VK_SHARING_MODE_EXCLUSIVE;
        info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        VkResult r = d.CreateImage(s.device, &info, nullptr, &image);
        if (r != VK_SUCCESS)
        {
            image = VK_NULL_HANDLE;
            wxl::ScriptMgr::Log(WXL_LOG_ERROR, "native: frame: vkCreateImage (%s) failed (%s)", what, ResultName(r));
            return false;
        }
        VkMemoryRequirements req = {};
        d.GetImageMemoryRequirements(s.device, image, &req);
        if (!memory::Allocate(req, memory::Where::DeviceLocal, block)
            || d.BindImageMemory(s.device, image, block.memory, block.offset) != VK_SUCCESS)
        {
            d.DestroyImage(s.device, image, nullptr);
            memory::Free(block);
            image = VK_NULL_HANDLE;
            block = memory::Block();
            wxl::ScriptMgr::Log(WXL_LOG_ERROR, "native: frame: no device memory for the %s", what);
            return false;
        }
        VkImageViewCreateInfo v = {};
        v.sType    = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        v.image    = image;
        v.viewType = VK_IMAGE_VIEW_TYPE_2D;
        v.format   = format;
        v.subresourceRange.aspectMask = AspectOf(format);
        v.subresourceRange.levelCount = 1;
        v.subresourceRange.layerCount = 1;
        r = d.CreateImageView(s.device, &v, nullptr, &view);
        if (r != VK_SUCCESS)
        {
            view = VK_NULL_HANDLE;
            d.DestroyImage(s.device, image, nullptr);
            memory::Free(block);
            image = VK_NULL_HANDLE;
            block = memory::Block();
            wxl::ScriptMgr::Log(WXL_LOG_ERROR, "native: frame: vkCreateImageView (%s) failed (%s)", what,
                                ResultName(r));
            return false;
        }
        return true;
    }

    /// Destroys what names the swapchain. The caller has waited for the device.
    void DropImageObjects()
    {
        if (!s.dapi || !s.device) return;
        const loader::Device& d = *s.dapi;
        for (uint32_t i = 0; i < kMaxImages; ++i)
        {
            DropFramebuffersNaming(s.views[i], 0, true);
            // The present pass's framebuffers name the swapchain views too.
            if (s.ppFbs[i] != VK_NULL_HANDLE) d.DestroyFramebuffer(s.device, s.ppFbs[i], nullptr);
            s.ppFbs[i] = VK_NULL_HANDLE;
            if (s.views[i] != VK_NULL_HANDLE) d.DestroyImageView(s.device, s.views[i], nullptr);
            if (s.renderFinished[i] != VK_NULL_HANDLE) d.DestroySemaphore(s.device, s.renderFinished[i], nullptr);
            s.views[i] = VK_NULL_HANDLE;
            s.renderFinished[i] = VK_NULL_HANDLE;
            s.imagesInFlight[i] = VK_NULL_HANDLE;
            s.images[i] = VK_NULL_HANDLE;
        }
        DropFramebuffersNaming(s.depthView, 0, true);
        DropFramebuffersNaming(s.backView, 0, true);
        if (s.depthView != VK_NULL_HANDLE) d.DestroyImageView(s.device, s.depthView, nullptr);
        if (s.depthImage != VK_NULL_HANDLE) d.DestroyImage(s.device, s.depthImage, nullptr);
        memory::Free(s.depthBlock);
        if (s.backView != VK_NULL_HANDLE) d.DestroyImageView(s.device, s.backView, nullptr);
        if (s.backImage != VK_NULL_HANDLE) d.DestroyImage(s.device, s.backImage, nullptr);
        memory::Free(s.backBlock);
        DropGBufferImages();   // sized with the back buffer; their framebuffers name the back view and went above
        s.depthView   = VK_NULL_HANDLE;
        s.depthImage  = VK_NULL_HANDLE;
        s.depthBlock  = memory::Block();
        s.depthLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        s.backView    = VK_NULL_HANDLE;
        s.backImage   = VK_NULL_HANDLE;
        s.backBlock   = memory::Block();
        s.backLayout  = VK_IMAGE_LAYOUT_UNDEFINED;
        s.imageCount  = 0;
        s.swapGen     = UINT32_MAX;
    }

    /// The image list, the per-image views and semaphores, the depth image and the back buffer (on
    /// the own-back-buffer path) for the swapchain that stands now.
    bool EnsureImageObjects(const context::Handles& h)
    {
        if (s.swapGen == context::SwapchainGeneration() && s.imageCount) return true;
        const loader::Device& d = *h.dapi;
        if (s.imageCount)
        {
            if (s.open && s.backTouched)
            {
                // The swapchain moved while this frame still names the old back buffer: nothing is
                // dropped under recorded commands; the frame presents nothing and rebuilds after.
                s.rebuild = true;
                return false;
            }
            // Earlier frames may still be presenting from these images.
            d.DeviceWaitIdle(s.device);
            s.completed = s.submitted;
            DropImageObjects();
        }
        EnsureDepthFormat();
        s.colourFormat = h.format;

        uint32_t count = 0;
        VkResult r = d.GetSwapchainImagesKHR(s.device, h.swapchain, &count, nullptr);
        if (r != VK_SUCCESS || count == 0 || count > kMaxImages)
        {
            wxl::ScriptMgr::Log(WXL_LOG_ERROR, "native: frame: the swapchain has %u image(s) (%s); at "
                                "most %u are tracked", count, ResultName(r), kMaxImages);
            s.failed = true;
            return false;
        }
        r = d.GetSwapchainImagesKHR(s.device, h.swapchain, &count, s.images);
        if (r != VK_SUCCESS)
        {
            Fail("vkGetSwapchainImagesKHR", r);
            return false;
        }
        // The own back buffer whenever the swapchain images can be copied into. Without TRANSFER_DST
        // the direct path renders straight into the acquired swapchain image, and the back-buffer
        // copies and read-back are refused.
        s.ownBack = (h.swapchainUsage & VK_IMAGE_USAGE_TRANSFER_DST_BIT) != 0;

        // The render size. Fullscreen asks for the format's resolution on a window at the
        // desktop's; the present pass scales. Only the own back buffer can differ from the swapchain.
        VkExtent2D render = h.extent;
        if (s.ownBack && s.renderWanted.width && s.renderWanted.height) render = s.renderWanted;

        // SAMPLED as well where the format allows it, so a post-process effect can read the scene's
        // depth (SceneDepth).
        VkImageUsageFlags depthUsage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
        s.depthSampled = false;
        if (s.depthFormat != VK_FORMAT_UNDEFINED && h.iapi && h.iapi->GetPhysicalDeviceFormatProperties)
        {
            VkFormatProperties fp = {};
            h.iapi->GetPhysicalDeviceFormatProperties(h.physical, s.depthFormat, &fp);
            // Depth-only formats: the view of a depth / stencil format names both aspects, which a
            // sampled view must not.
            s.depthSampled = (fp.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT) != 0
                             && AspectOf(s.depthFormat) == VK_IMAGE_ASPECT_DEPTH_BIT;
            if (s.depthSampled) depthUsage |= VK_IMAGE_USAGE_SAMPLED_BIT;
        }
        if (s.depthFormat != VK_FORMAT_UNDEFINED
            && !MakeRaw(s.depthFormat, render, depthUsage, s.depthImage, s.depthView, s.depthBlock, "back depth"))
        {
            s.failed = true;
            return false;
        }
        s.depthLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        if (!s.modeLogged)
        {
            s.modeLogged = true;
            wxl::ScriptMgr::Log(s.ownBack ? WXL_LOG_DEBUG : WXL_LOG_WARN,
                                s.ownBack ? "native: frame: rendering into an own back buffer (%s), copied "
                                            "onto the swapchain image at present"
                                          : "native: frame: the swapchain images cannot be copied into "
                                            "(no TRANSFER_DST); rendering straight into them (%s), and "
                                            "back-buffer copies / read-back are refused",
                                wxlvk::vk::FormatName(h.format));
        }
        if (s.ownBack)
        {
            // SAMPLED as well, for the present pass (scale, gamma).
            if (!MakeRaw(h.format, render,
                         VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                             VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                         s.backImage, s.backView, s.backBlock, "back buffer"))
            {
                s.failed = true;
                return false;
            }
            s.backLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        }

        VkSemaphoreCreateInfo sem = {};
        sem.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
        for (uint32_t i = 0; i < count; ++i)
        {
            // The direct path's attachment views, and the present pass's on the own back buffer path.
            {
                VkImageViewCreateInfo view = {};
                view.sType    = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
                view.image    = s.images[i];
                view.viewType = VK_IMAGE_VIEW_TYPE_2D;
                view.format   = h.format;
                view.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
                view.subresourceRange.levelCount = 1;
                view.subresourceRange.layerCount = 1;
                r = d.CreateImageView(s.device, &view, nullptr, &s.views[i]);
                if (r != VK_SUCCESS)
                {
                    s.views[i] = VK_NULL_HANDLE;
                    Fail("vkCreateImageView", r);
                    return false;
                }
            }
            // Per IMAGE, not per frame: only re-acquiring an image proves its present consumed it.
            r = d.CreateSemaphore(s.device, &sem, nullptr, &s.renderFinished[i]);
            if (r != VK_SUCCESS)
            {
                s.renderFinished[i] = VK_NULL_HANDLE;
                Fail("vkCreateSemaphore (render finished)", r);
                return false;
            }
        }
        s.imageCount = count;
        s.extent     = render;
        s.swapExtent = h.extent;
        s.swapGen    = context::SwapchainGeneration();
        if (render.width != h.extent.width || render.height != h.extent.height)
            wxl::ScriptMgr::Log(WXL_LOG_DEBUG,
                                "native: frame: back buffer %ux%u on a %ux%u swapchain (fullscreen at the "
                                "format's resolution on a desktop-sized window); the present pass scales",
                                render.width, render.height, h.extent.width, h.extent.height);
        return true;
    }

    /**
     * The G-buffer's three images at the back buffer's size (docs/architecture.md "The G-buffer"): a
     * swapchain-derived object, dropped and re-made with the back buffer. Made at the first G-buffer
     * pass a consumer's request brings; false when they cannot be had, and then refused for the session.
     */
    bool EnsureGBufferImages()
    {
        if (s.gbufImage[0] != VK_NULL_HANDLE) return true;
        if (s.gbufFailed || !s.ownBack || s.extent.width == 0 || s.extent.height == 0) return false;
        const context::Handles* h = context::Get();
        if (!h) return false;
        if (!h->independentBlend)
        {
            // The class's pipelines blend the frame one way and the G-buffer another: per-attachment state.
            s.gbufFailed = true;
            wxl::ScriptMgr::Log(WXL_LOG_WARN, "native: frame: the adapter has no independentBlend; the G-buffer is refused for the session");
            return false;
        }
        // The normal's format must be attachable: SNORM carries the precision, SFLOAT is the fallback.
        if (h->iapi && h->iapi->GetPhysicalDeviceFormatProperties)
        {
            VkFormatProperties fp = {};
            h->iapi->GetPhysicalDeviceFormatProperties(h->physical, s.gbufFormat[1], &fp);
            if (!(fp.optimalTilingFeatures & VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT))
                s.gbufFormat[1] = VK_FORMAT_R16G16B16A16_SFLOAT;
        }
        static const char* const kWhat[kGBufferAttachments] = { "G-buffer albedo", "G-buffer normal", "G-buffer light" };
        for (uint32_t i = 0; i < kGBufferAttachments; ++i)
        {
            if (MakeRaw(s.gbufFormat[i], s.extent, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                        s.gbufImage[i], s.gbufView[i], s.gbufBlock[i], kWhat[i]))
                continue;
            DropGBufferImages();
            s.gbufFailed = true;
            wxl::ScriptMgr::Log(WXL_LOG_WARN, "native: frame: the G-buffer cannot be made (%s); refused for the session",
                                kWhat[i]);
            return false;
        }
        s.gbufLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        uint64_t bytes = 0;
        for (const memory::Block& b : s.gbufBlock) bytes += b.size;
        wxl::ScriptMgr::Log(WXL_LOG_DEBUG, "native: frame: G-buffer %ux%u: albedo %s, normal %s, light %s (%llu MiB)",
                            s.extent.width, s.extent.height, wxlvk::vk::FormatName(s.gbufFormat[0]),
                            wxlvk::vk::FormatName(s.gbufFormat[1]), wxlvk::vk::FormatName(s.gbufFormat[2]),
                            static_cast<unsigned long long>(bytes >> 20));
        return true;
    }

    /// Destroys the G-buffer's images. The caller has waited for the device (DropImageObjects), or
    /// nothing has used them yet (a failed make).
    void DropGBufferImages()
    {
        if (!s.dapi || !s.device) return;
        const loader::Device& d = *s.dapi;
        for (uint32_t i = 0; i < kGBufferAttachments; ++i)
        {
            if (s.gbufView[i] != VK_NULL_HANDLE) d.DestroyImageView(s.device, s.gbufView[i], nullptr);
            if (s.gbufImage[i] != VK_NULL_HANDLE) d.DestroyImage(s.device, s.gbufImage[i], nullptr);
            memory::Free(s.gbufBlock[i]);
            s.gbufView[i]  = VK_NULL_HANDLE;
            s.gbufImage[i] = VK_NULL_HANDLE;
            s.gbufBlock[i] = memory::Block();
        }
        s.gbufLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        s.gbufDrawn  = false;
    }

    /// The acquire, once a frame: at the present on the own back buffer, at the first back-buffer
    /// pass on the direct path.
    bool Acquire()
    {
        if (s.acquired) return true;
        if (s.noImage) return false;
        const context::Handles* h = context::Get();
        FrameSlot&              slot = s.slots[s.fi];
        if (!h || h->swapchain == VK_NULL_HANDLE || h->extent.width == 0 || h->extent.height == 0
            || !EnsureImageObjects(*h))
        {
            s.noImage = true;
            ++s.stats.noImage;
            return false;
        }
        uint32_t image = 0;
        VkResult r = s.dapi->AcquireNextImageKHR(s.device, h->swapchain, UINT64_MAX, slot.imageAvailable,
                                                 VK_NULL_HANDLE, &image);
        if (r == VK_ERROR_OUT_OF_DATE_KHR)
        {
            // Nothing acquired, the semaphore untouched: this frame presents nothing, the swapchain
            // is rebuilt after its submit.
            s.noImage = true;
            s.rebuild = true;
            ++s.stats.noImage;
            return false;
        }
        if (r == VK_SUBOPTIMAL_KHR) s.rebuild = true;   // acquired: finish the frame, rebuild after
        else if (r != VK_SUCCESS)
        {
            Fail("vkAcquireNextImageKHR", r);
            return false;
        }
        if (image >= s.imageCount)
        {
            s.failed = true;
            return false;
        }
        if (s.imagesInFlight[image] != VK_NULL_HANDLE && s.imagesInFlight[image] != slot.fence)
            s.dapi->WaitForFences(s.device, 1, &s.imagesInFlight[image], VK_TRUE, UINT64_MAX);
        s.imagesInFlight[image] = slot.fence;
        s.image      = image;
        s.acquired   = true;
        s.swapLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        return true;
    }
}
