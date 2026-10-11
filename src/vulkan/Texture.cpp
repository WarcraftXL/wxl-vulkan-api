// Images: create them, fill them (uploads, clears, generated mip chains), and destroy them.
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


#include "vulkan/texture/Internal.hpp"

using namespace wxlvk::vk::texture::detail;

namespace wxlvk::vk::texture
{
    bool Init()
    {
        const context::Handles* h = context::Get();
        if (!h || !h->device || !h->dapi || !h->iapi || !h->physical)
        {
            wxl::ScriptMgr::Log(WXL_LOG_WARN, "vulkan: texture: Init() without a live context");
            return false;
        }

        if (s.ready)
        {
            if (s.device == h->device) return true;

            // A different VkDevice means ours is gone, and calling into it to destroy anything
            // would be a call into a dead device.
            wxl::ScriptMgr::Log(WXL_LOG_WARN,
                                "vulkan: texture: the context reports a different VkDevice than the "
                                "one this module was built on; forgetting %u image(s) and the upload "
                                "objects rather than destroying them",
                                s.ownedCount);
            FreeTables();
            s = State();
        }

        if (!HaveEntryPoints(*h->dapi, *h->iapi)) return false;

        s.owned = new (std::nothrow) Owned[kMaxImages];
        // The index; without it (the heap said no) FindOwned() scans.
        s.hash = new (std::nothrow) uint32_t[kHashSlots];
        if (s.hash) std::memset(s.hash, 0, sizeof(uint32_t) * kHashSlots);
        if (!s.owned)
        {
            wxl::ScriptMgr::Log(WXL_LOG_ERROR, "vulkan: texture: cannot allocate the image table");
            return false;
        }

        s.device   = h->device;
        s.physical = h->physical;
        s.queue    = h->graphicsQueue;
        s.dapi     = h->dapi;
        s.iapi     = h->iapi;

        VkPhysicalDeviceProperties props = {};
        s.iapi->GetPhysicalDeviceProperties(s.physical, &props);
        s.maxDimension     = props.limits.maxImageDimension2D;
        s.maxDimensionCube = props.limits.maxImageDimensionCube;
        // See the format check in CreateMipped(): the per-format TRANSFER_SRC/DST feature bits are
        // only reported by a 1.1+ driver, and on a 1.0 driver their absence means nothing.
        s.transferBits = props.apiVersion >= VK_MAKE_API_VERSION(0, 1, 1, 0);

        VkCommandPoolCreateInfo pool = {};
        pool.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        // A pool of our own for the synchronous paths (UploadRegion, ClearAll) rather than the
        // frame's: those buffers are re-recorded every frame and two of them may be in flight.
        // TRANSIENT because the single buffer is re-recorded for every upload; RESET_COMMAND_BUFFER
        // so vkBeginCommandBuffer on it is a legal implicit reset, without one more required entry
        // point for vkResetCommandBuffer.
        pool.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT |
                     VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        // The graphics family, because the upload is submitted to the graphics queue -- the same
        // queue the draws that read the image run on, which is what lets every barrier here use
        // VK_QUEUE_FAMILY_IGNORED. A dedicated transfer queue would be faster for bulk uploads and
        // would need a queue family ownership transfer on every image; not worth it for this path.
        pool.queueFamilyIndex = h->graphicsFamily;

        VkResult r = s.dapi->CreateCommandPool(s.device, &pool, nullptr, &s.pool);
        if (r != VK_SUCCESS)
        {
            s.pool = VK_NULL_HANDLE;
            Fail("vkCreateCommandPool", r);
            s = State();
            return false;
        }

        VkCommandBufferAllocateInfo alloc = {};
        alloc.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        alloc.commandPool        = s.pool;
        alloc.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        alloc.commandBufferCount = 1;

        r = s.dapi->AllocateCommandBuffers(s.device, &alloc, &s.cmd);
        if (r != VK_SUCCESS)
        {
            Fail("vkAllocateCommandBuffers", r);
            DropUploadObjects(*s.dapi, s.device);
            s = State();
            return false;
        }

        VkFenceCreateInfo fence = {};
        fence.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        // Unsignalled: it is reset immediately before every submit anyway (see UploadRegion), so the
        // initial state is never observed.

        r = s.dapi->CreateFence(s.device, &fence, nullptr, &s.fence);
        if (r != VK_SUCCESS)
        {
            s.fence = VK_NULL_HANDLE;
            Fail("vkCreateFence", r);
            DropUploadObjects(*s.dapi, s.device);
            s = State();
            return false;
        }

        s.ready = true;
        wxl::ScriptMgr::Log(WXL_LOG_DEBUG,
                            "vulkan: texture: ready -- upload pool on queue family %u, images up to "
                            "%ux%u, at most %u tracked at once",
                            h->graphicsFamily, s.maxDimension, s.maxDimension, kMaxImages);
        return true;
    }

    bool CreateMipped(uint32_t width, uint32_t height, uint32_t levels, VkFormat format, Image& out)
    {
        return CreateImage(width, height, levels, false, format, out);
    }

    bool CreateCube(uint32_t size, uint32_t levels, VkFormat format, Image& out)
    {
        // Square is a Vulkan requirement for CUBE_COMPATIBLE (and D3D's CreateCubeTexture takes
        // one edge length), and the cube limit is its own limit, often smaller than the 2D one.
        if (s.ready && size > s.maxDimensionCube)
        {
            wxl::ScriptMgr::Log(WXL_LOG_WARN,
                                "vulkan: texture: refusing a %u-texel cube map; this device allows "
                                "1..%u on a side",
                                size, s.maxDimensionCube);
            return false;
        }
        return CreateImage(size, size, levels, true, format, out);
    }

    bool CreateTarget(uint32_t width, uint32_t height, uint32_t levels, bool cube, VkFormat format,
                      bool depth, Image& out)
    {
        if (cube && (width != height || (s.ready && width > s.maxDimensionCube))) return false;
        if (depth && (cube || levels != 1)) return false;
        return CreateImage(width, height, levels, cube, format, out,
                           depth ? VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT
                                 : VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
                           depth);
    }


    bool ClearAll(Image& img, const float* rgba)
    {
        if (!s.ready || s.wedged || img.handle == VK_NULL_HANDLE) return false;
        const loader::Device& d = *s.dapi;

        VkCommandBufferBeginInfo begin = {};
        begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        VkResult r = d.BeginCommandBuffer(s.cmd, &begin);
        if (r != VK_SUCCESS) { Fail("vkBeginCommandBuffer (clear)", r); return false; }

        // From whatever layout it has: UNDEFINED discards the contents, which is the point.
        Transition(s.cmd, img.handle, img.levels, img.layers, VK_IMAGE_LAYOUT_UNDEFINED,
                   VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, 0,
                   VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT);

        // Mid grey, opaque, unless the caller names a colour: a level that stays this way is
        // recognisable, and neutral enough at a distance not to read as a bug on its own. Not an
        // engine value: D3D does not define the contents either.
        VkClearColorValue colour = {};
        colour.float32[0] = 0.5f;
        colour.float32[1] = 0.5f;
        colour.float32[2] = 0.5f;
        colour.float32[3] = 1.0f;
        if (rgba)
            for (unsigned i = 0; i < 4; ++i) colour.float32[i] = rgba[i];
        const VkImageSubresourceRange range = WholeColour(img.levels, img.layers);
        d.CmdClearColorImage(s.cmd, img.handle, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &colour, 1, &range);

        const VkImageLayout to = img.wanted != VK_IMAGE_LAYOUT_UNDEFINED
                                     ? img.wanted
                                     : VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        Transition(s.cmd, img.handle, img.levels, img.layers, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, to,
                   VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
                   VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                   VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT);

        r = d.EndCommandBuffer(s.cmd);
        if (r != VK_SUCCESS) { Fail("vkEndCommandBuffer (clear)", r); return false; }
        r = d.ResetFences(s.device, 1, &s.fence);
        if (r != VK_SUCCESS) { Fail("vkResetFences (clear)", r); return false; }

        VkSubmitInfo submit = {};
        submit.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit.commandBufferCount = 1;
        submit.pCommandBuffers    = &s.cmd;
        r = d.QueueSubmit(s.queue, 1, &submit, s.fence);
        if (r != VK_SUCCESS) { Fail("vkQueueSubmit (clear)", r); return false; }
        r = d.WaitForFences(s.device, 1, &s.fence, VK_TRUE, UINT64_MAX);
        if (r != VK_SUCCESS)
        {
            s.wedged = true;
            Fail("vkWaitForFences (clear)", r);
            return false;
        }
        img.layout = to;
        return true;
    }

    bool RegionBytes(const Image& img, uint32_t width, uint32_t height, uint64_t& rowBytes,
                     uint32_t& rowCount)
    {
        const FormatInfo fmt = Describe(img.format);
        if (fmt.unitBytes == 0 || width == 0 || height == 0) return false;
        rowBytes = RowBytes(fmt, width);
        rowCount = RowCount(fmt, height);
        return true;
    }

    bool CanGenerateMips(VkFormat format)
    {
        if (!s.ready || !s.iapi || !s.physical) return false;
        const FormatInfo fmt = Describe(format);
        // A blit cannot write a block-compressed level (BLIT_DST is never offered for BC), and the
        // engine never asks a DXT texture for generated levels anyway: its chains arrive whole.
        if (fmt.unitBytes == 0 || fmt.compressed) return false;
        VkFormatProperties props = {};
        s.iapi->GetPhysicalDeviceFormatProperties(s.physical, format, &props);
        const VkFormatFeatureFlags need = VK_FORMAT_FEATURE_BLIT_SRC_BIT | VK_FORMAT_FEATURE_BLIT_DST_BIT |
                                          VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT;
        return (props.optimalTilingFeatures & need) == need;
    }

    bool RecordGenerateMips(VkCommandBuffer cmd, Image& img)
    {
        if (!s.ready || !cmd || img.handle == VK_NULL_HANDLE || img.levels < 2) return false;
        const VkPipelineStageFlags readers = VK_PIPELINE_STAGE_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;

        // One subresource-range barrier, for the chain's per-level steps; Transition() covers the
        // whole image, which is the right shape only for the first and the last step.
        const auto levels = [&](uint32_t base, uint32_t count, VkImageLayout from, VkImageLayout to,
                                VkPipelineStageFlags srcStage, VkAccessFlags srcAccess,
                                VkPipelineStageFlags dstStage, VkAccessFlags dstAccess) {
            VkImageMemoryBarrier b = {};
            b.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            b.srcAccessMask       = srcAccess;
            b.dstAccessMask       = dstAccess;
            b.oldLayout           = from;
            b.newLayout           = to;
            b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            b.image               = img.handle;
            b.subresourceRange    = WholeColour(img.levels, img.layers);
            b.subresourceRange.baseMipLevel = base;
            b.subresourceRange.levelCount   = count;
            s.dapi->CmdPipelineBarrier(cmd, srcStage, dstStage, 0, 0, nullptr, 0, nullptr, 1, &b);
        };

        // Before: the whole image to TRANSFER_DST, contents kept. The first scope waits for every
        // earlier sampler read (this buffer's and earlier submissions') and makes the level-0
        // upload's transfer write available, whichever command recorded it.
        const VkImageLayout from = img.layout == VK_IMAGE_LAYOUT_UNDEFINED ? VK_IMAGE_LAYOUT_UNDEFINED : img.layout;
        Transition(cmd, img.handle, img.levels, img.layers, from, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                   readers | VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
                   VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT);

        // Level by level, every layer at once (a cube's six faces filter independently: a blit's
        // layer range maps layer i to layer i). Level l - 1 becomes the source once written.
        for (uint32_t l = 1; l < img.levels; ++l)
        {
            levels(l - 1, 1, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                   VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
                   VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_READ_BIT);
            VkImageBlit blit = {};
            blit.srcSubresource.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
            blit.srcSubresource.mipLevel       = l - 1;
            blit.srcSubresource.baseArrayLayer = 0;
            blit.srcSubresource.layerCount     = img.layers;
            blit.srcOffsets[1] = VkOffset3D{ static_cast<int32_t>(LevelSide(img.extent.width, l - 1)),
                                             static_cast<int32_t>(LevelSide(img.extent.height, l - 1)), 1 };
            blit.dstSubresource                = blit.srcSubresource;
            blit.dstSubresource.mipLevel       = l;
            blit.dstOffsets[1] = VkOffset3D{ static_cast<int32_t>(LevelSide(img.extent.width, l)),
                                             static_cast<int32_t>(LevelSide(img.extent.height, l)), 1 };
            s.dapi->CmdBlitImage(cmd, img.handle, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, img.handle,
                                 VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit, VK_FILTER_LINEAR);
        }

        // After: every level to the layout the image is tracked in -- the sources from
        // TRANSFER_SRC, the last level from TRANSFER_DST -- visible to the sampling stages.
        const VkImageLayout to = img.wanted == VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL
                                     ? VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL
                                     : VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        const bool toShader = to == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        const VkPipelineStageFlags dstStage  = toShader ? readers : VK_PIPELINE_STAGE_TRANSFER_BIT;
        const VkAccessFlags        dstAccess = toShader ? VK_ACCESS_SHADER_READ_BIT : VK_ACCESS_TRANSFER_READ_BIT;
        levels(0, img.levels - 1, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, to, VK_PIPELINE_STAGE_TRANSFER_BIT,
               VK_ACCESS_TRANSFER_READ_BIT, dstStage, dstAccess);
        levels(img.levels - 1, 1, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, to, VK_PIPELINE_STAGE_TRANSFER_BIT,
               VK_ACCESS_TRANSFER_WRITE_BIT, dstStage, dstAccess);
        img.layout = to;
        return true;
    }

    void DestroyIdle(Image& img)
    {
        if (img.handle == VK_NULL_HANDLE)
        {
            img = Image();
            return;
        }
        const uint32_t i = FindOwned(img.handle);
        if (i == kMaxImages || !s.dapi || !s.device)
        {
            img = Image();   // not one of ours: forgetting is the only safe answer
            return;
        }
        if (s.owned[i].view != VK_NULL_HANDLE) s.dapi->DestroyImageView(s.device, s.owned[i].view, nullptr);
        s.dapi->DestroyImage(s.device, s.owned[i].handle, nullptr);
        memory::Free(s.owned[i].block);
        RemoveOwned(i);
        img = Image();
    }

    uint32_t RecordRegions(VkCommandBuffer cmd, Image& img, VkBuffer src, const RegionCopy* regions,
                           uint32_t count)
    {
        if (!s.ready || !cmd || src == VK_NULL_HANDLE || img.handle == VK_NULL_HANDLE || !regions || count == 0)
            return 0;
        const FormatInfo fmt = Describe(img.format);
        if (fmt.unitBytes == 0) return 0;

        // UploadRegion's checks, per region, without its log lines; a refused one is left out of
        // the batch (the caller counts the shortfall from the return value).
        constexpr uint32_t kBatch = 64;
        VkBufferImageCopy  copies[kBatch];
        uint32_t           n = 0;
        uint32_t           recorded = 0;
        bool               began = false;
        const bool sampled = img.layout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        const VkImageLayout to = img.wanted == VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL
                                     ? VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL
                                     : VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        const auto flush = [&]() {
            if (n == 0) return;
            s.dapi->CmdCopyBufferToImage(cmd, src, img.handle, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, n, copies);
            recorded += n;
            n = 0;
        };
        for (uint32_t i = 0; i < count; ++i)
        {
            const RegionCopy& g = regions[i];
            if (g.level >= img.levels || g.layer >= img.layers) continue;
            const uint32_t levelW = LevelSide(img.extent.width, g.level);
            const uint32_t levelH = LevelSide(img.extent.height, g.level);
            if (g.width == 0 || g.height == 0 || uint64_t(g.x) + g.width > levelW
                || uint64_t(g.y) + g.height > levelH)
                continue;
            if (fmt.compressed)
            {
                const bool originOk = g.x % kBlock == 0 && g.y % kBlock == 0;
                const bool widthOk  = g.width % kBlock == 0 || g.x + g.width == levelW;
                const bool heightOk = g.height % kBlock == 0 || g.y + g.height == levelH;
                if (!originOk || !widthOk || !heightOk) continue;
            }
            if (g.srcOffset % 4 != 0 || g.srcOffset % fmt.unitBytes != 0) continue;
            if (!began)
            {
                // One barrier for the whole batch: every earlier read (this buffer's draws, the
                // frames in flight -- write-after-read needs only the execution dependency) and an
                // earlier copy or clear of the image (TRANSFER, with its writes) before the copies;
                // contents outside the regions kept, so the old layout is the tracked one.
                VkPipelineStageFlags srcStage =
                    sampled ? (VK_PIPELINE_STAGE_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT
                               | VK_PIPELINE_STAGE_TRANSFER_BIT)
                            : VK_PIPELINE_STAGE_TRANSFER_BIT;
                if (img.layout != VK_IMAGE_LAYOUT_UNDEFINED && !sampled
                    && img.layout != VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL
                    && img.layout != VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL)
                    srcStage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
                Transition(cmd, img.handle, img.levels, img.layers, img.layout,
                           VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, srcStage,
                           img.layout == VK_IMAGE_LAYOUT_UNDEFINED ? 0 : VK_ACCESS_TRANSFER_WRITE_BIT,
                           VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT);
                began = true;
            }
            VkBufferImageCopy& c = copies[n++];
            c = VkBufferImageCopy{};
            c.bufferOffset                    = g.srcOffset;
            c.bufferRowLength                 = 0;   // tightly packed: see UploadRegion
            c.bufferImageHeight               = 0;
            c.imageSubresource.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
            c.imageSubresource.mipLevel       = g.level;
            c.imageSubresource.baseArrayLayer = g.layer;
            c.imageSubresource.layerCount     = 1;
            c.imageOffset = VkOffset3D{ static_cast<int32_t>(g.x), static_cast<int32_t>(g.y), 0 };
            c.imageExtent = VkExtent3D{ g.width, g.height, 1 };
            if (n == kBatch) flush();
        }
        if (!began) return 0;
        flush();
        const bool toShader = to == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        Transition(cmd, img.handle, img.levels, img.layers, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, to,
                   VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
                   toShader ? (VK_PIPELINE_STAGE_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT
                               | VK_PIPELINE_STAGE_TRANSFER_BIT)
                            : VK_PIPELINE_STAGE_TRANSFER_BIT,
                   toShader ? (VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT)
                            : (VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT));
        img.layout = to;
        return recorded;
    }

    bool RecordClear(VkCommandBuffer cmd, Image& img, const float* rgba)
    {
        if (!s.ready || !cmd || img.handle == VK_NULL_HANDLE) return false;
        // ClearAll's commands, recorded: from UNDEFINED (the contents are discarded, which is the
        // point), the clear, then `wanted` with the clear made visible to samplers and transfers.
        // The source stage waits for every earlier use of the image at all stages: a recorded clear
        // may follow anything in the buffer, unlike ClearAll's own fresh submission.
        Transition(cmd, img.handle, img.levels, img.layers, VK_IMAGE_LAYOUT_UNDEFINED,
                   VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0,
                   VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT);
        VkClearColorValue colour = {};
        colour.float32[0] = 0.5f;
        colour.float32[1] = 0.5f;
        colour.float32[2] = 0.5f;
        colour.float32[3] = 1.0f;
        if (rgba)
            for (unsigned i = 0; i < 4; ++i) colour.float32[i] = rgba[i];
        const VkImageSubresourceRange range = WholeColour(img.levels, img.layers);
        s.dapi->CmdClearColorImage(cmd, img.handle, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &colour, 1, &range);
        const VkImageLayout to = img.wanted != VK_IMAGE_LAYOUT_UNDEFINED ? img.wanted
                                                                         : VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        Transition(cmd, img.handle, img.levels, img.layers, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, to,
                   VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
                   VK_PIPELINE_STAGE_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT
                       | VK_PIPELINE_STAGE_TRANSFER_BIT,
                   VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT);
        img.layout = to;
        return true;
    }
}
