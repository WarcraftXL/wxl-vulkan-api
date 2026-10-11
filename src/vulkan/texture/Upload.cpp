// A texture region upload: staged ahead of the frame, or inline.
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
    bool UploadRegion(Image& img, uint32_t level, uint32_t x, uint32_t y,
                      uint32_t width, uint32_t height, const void* rows, uint32_t rowPitch,
                      uint32_t layer)
    {
        if (!s.ready)
        {
            wxl::ScriptMgr::Log(WXL_LOG_WARN, "vulkan: texture: UploadRegion() before Init()");
            return false;
        }
        if (s.wedged)
        {
            wxl::ScriptMgr::Log(WXL_LOG_WARN,
                                "vulkan: texture: UploadRegion() refused: an earlier upload's wait "
                                "failed and its command buffer may still be pending");
            return false;
        }
        if (img.handle == VK_NULL_HANDLE || FindOwned(img.handle) == kMaxImages || !rows)
        {
            wxl::ScriptMgr::Log(WXL_LOG_WARN,
                                "vulkan: texture: UploadRegion() needs an image this module made "
                                "and a pixel pointer");
            return false;
        }

        const FormatInfo fmt = Describe(img.format);
        if (fmt.unitBytes == 0 || img.levels == 0)
        {
            // Cannot happen for an image this module created; a corrupted Image is refused rather
            // than used to size a copy.
            wxl::ScriptMgr::Log(WXL_LOG_WARN,
                                "vulkan: texture: UploadRegion() of an Image with format %s and %u "
                                "level(s), which this module did not create",
                                Name(img.format), img.levels);
            return false;
        }

        // --- the region against the level -----------------------------------------------------
        if (level >= img.levels)
        {
            wxl::ScriptMgr::Log(WXL_LOG_WARN,
                                "vulkan: texture: refusing an upload to level %u of an image with "
                                "%u level(s)",
                                level, img.levels);
            return false;
        }
        if (layer >= img.layers)
        {
            wxl::ScriptMgr::Log(WXL_LOG_WARN,
                                "vulkan: texture: refusing an upload to layer %u of an image with "
                                "%u layer(s)",
                                layer, img.layers);
            return false;
        }

        const uint32_t levelW = LevelSide(img.extent.width, level);
        const uint32_t levelH = LevelSide(img.extent.height, level);

        // In 64 bits so that x + width cannot wrap around to something small.
        if (width == 0 || height == 0 || uint64_t(x) + width > levelW ||
            uint64_t(y) + height > levelH)
        {
            wxl::ScriptMgr::Log(WXL_LOG_WARN,
                                "vulkan: texture: refusing a %ux%u region at (%u, %u) of level %u, "
                                "which is %ux%u",
                                width, height, x, y, level, levelW, levelH);
            return false;
        }

        if (fmt.compressed)
        {
            // A block-compressed level is addressed in texels but stored in 4x4 blocks, and a
            // copy may not start or stop inside a block: imageOffset must be a multiple of the
            // block size, and so must imageExtent -- except that an extent is allowed to run to
            // the level's edge, because the last level(s) of a chain can be narrower than a block
            // (a 2x2 level is one block, stored whole). Those are the spec's rules for
            // vkCmdCopyBufferToImage, and a driver does not report a violation; it copies garbage.
            const bool originOk = x % kBlock == 0 && y % kBlock == 0;
            const bool widthOk  = width % kBlock == 0 || x + width == levelW;
            const bool heightOk = height % kBlock == 0 || y + height == levelH;
            if (!originOk || !widthOk || !heightOk)
            {
                wxl::ScriptMgr::Log(WXL_LOG_WARN,
                                    "vulkan: texture: refusing a %ux%u region at (%u, %u) of a %s "
                                    "level %u (%ux%u): origin and extent must be multiples of %u, "
                                    "or the extent must reach the level's edge",
                                    width, height, x, y, Name(img.format), level, levelW, levelH,
                                    kBlock);
                return false;
            }
        }

        // --- the source rows against the region -----------------------------------------------
        const uint64_t rowBytes = RowBytes(fmt, width);
        const uint32_t rowCount = RowCount(fmt, height);
        const uint64_t total    = rowBytes * rowCount;

        if (rowPitch < rowBytes)
        {
            // A pitch shorter than a row means the rows overlap in the caller's memory, which is
            // never what anyone meant; the usual cause is a pitch in texels or in blocks.
            wxl::ScriptMgr::Log(WXL_LOG_WARN,
                                "vulkan: texture: refusing a row pitch of %u bytes for a %u-texel "
                                "wide %s region, whose rows are %llu bytes",
                                rowPitch, width, Name(img.format),
                                static_cast<unsigned long long>(rowBytes));
            return false;
        }

        // The last row's end, in bytes from `rows`, must be addressable in this 32-bit process;
        // the memcpy loop below indexes with size_t.
        const uint64_t span = uint64_t(rowCount - 1) * rowPitch + rowBytes;
        if (span > SIZE_MAX || total > SIZE_MAX)
        {
            wxl::ScriptMgr::Log(WXL_LOG_WARN,
                                "vulkan: texture: refusing a %ux%u region whose source spans %llu "
                                "bytes, more than this process can address",
                                width, height, static_cast<unsigned long long>(span));
            return false;
        }

        memory::Buffer* staging = nullptr;
        if (!memory::Staging(static_cast<VkDeviceSize>(total), staging) || !staging ||
            staging->handle == VK_NULL_HANDLE || !staging->block.mapped || staging->size < total)
        {
            wxl::ScriptMgr::Log(WXL_LOG_WARN,
                                "vulkan: texture: no mapped staging buffer of %llu bytes",
                                static_cast<unsigned long long>(total));
            return false;
        }

        // --- repack the pitched rows into a tight staging buffer ------------------------------
        //
        // The caller's rows are `rowPitch` bytes apart; the staging buffer gets them `rowBytes`
        // apart, one memcpy per row. The pitch is dropped here on purpose rather than passed to
        // vkCmdCopyBufferToImage as bufferRowLength: that field is in TEXELS, not bytes, and for a
        // compressed format in texels of the block grid, so a byte pitch there is wrong in two
        // different ways depending on the format -- and the driver reports neither; it shears the
        // picture. Repacking costs one extra pass over the bytes and removes the question.
        //
        // For a compressed format a "row" here is a block row: ceil(width / 4) blocks of
        // unitBytes, and there are ceil(height / 4) of them -- which is also how the BLP's DXT
        // payload is laid out, so the engine's pitch is simply the level's block-row size.
        //
        // The staging buffer is reused across uploads, which is safe because every upload waits
        // for its own copy to finish before it returns: nothing can still be reading the buffer
        // when the next memcpy overwrites it.
        {
            const uint8_t* src = static_cast<const uint8_t*>(rows);
            uint8_t*       dst = static_cast<uint8_t*>(staging->block.mapped);
            const size_t   n   = static_cast<size_t>(rowBytes);
            for (uint32_t i = 0; i < rowCount; ++i)
                std::memcpy(dst + size_t(i) * n, src + size_t(i) * rowPitch, n);
        }

        // Two separate things have to be true before the GPU reads these bytes, and they are
        // handled in two different places.
        //
        // Availability out of the CPU's caches: on a memory type without HOST_COHERENT the writes
        // above can sit where the GPU does not look, and the copy then uploads whatever was there
        // before -- silently, with no error anywhere. FlushHost() is the vkFlushMappedMemoryRanges
        // for that case (rounded to nonCoherentAtomSize, which only the memory module knows) and
        // returns at once on a coherent type. It belongs here, between the last write and the
        // submit, because that window is the only place a flush can be correct.
        //
        // Visibility to the device: no HOST -> TRANSFER barrier is recorded, because vkQueueSubmit
        // itself guarantees that host writes made available before the submit are visible to the
        // commands it submits. A barrier on HOST_WRITE would restate that guarantee.
        if (!memory::FlushHost(staging->block))
        {
            wxl::ScriptMgr::Log(WXL_LOG_WARN,
                                "vulkan: texture: flushing the staging buffer failed; not uploading "
                                "bytes the GPU might not see");
            return false;
        }

        const loader::Device& d = *s.dapi;

        VkCommandBufferBeginInfo begin = {};
        begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

        VkResult r = d.BeginCommandBuffer(s.cmd, &begin);
        if (r != VK_SUCCESS)
        {
            Fail("vkBeginCommandBuffer (upload)", r);
            return false;
        }

        // --- 1. the whole image to TRANSFER_DST_OPTIMAL -----------------------------------------
        //
        // The whole image and not the one level, because Image tracks one layout: see WholeColour.
        //
        // The old layout is UNDEFINED when nothing in the image is worth keeping: the first upload
        // (the tracked layout says UNDEFINED), or a copy that overwrites every texel of a 1-level,
        // 1-layer image. UNDEFINED is what lets the driver skip
        // preserving contents nobody will read. Any other case -- a region, one level of a chain,
        // or one face of a cube map -- keeps the tracked layout as the old one, so the texels the
        // copy does not touch survive.
        //
        // The source stage is TRANSFER and not TOP_OF_PIPE. On a first upload the two are
        // equivalent. On a re-upload they are not: the image may be the source of a copy in a frame
        // that has been submitted and not finished, and this transition is a write. A barrier's
        // first scope covers everything submitted earlier on the queue, but only at the stages it
        // names -- TOP_OF_PIPE names none, so the transition could overtake that copy's read.
        // TRANSFER makes it wait for the read. The access mask is 0 for a discard, since a
        // write-after-read hazard needs only the execution dependency, and TRANSFER_WRITE when
        // contents are kept: the last writes to this image were an earlier upload's copy, already
        // made available by that upload's own barrier, and naming them again is harmless.
        //
        // SHADER_READ_ONLY_OPTIMAL is the other layout this module leaves an image in (see
        // `wanted`), and its reader is the fragment shader of a draw a frame in flight may still
        // be running. The source stage is therefore FRAGMENT_SHADER -- the stage
        // whose reads this write must wait for -- and the access is 0 in the discard case: a read
        // has nothing to flush. When contents are kept the previous writes were an earlier upload's
        // copy, already made available by its own barrier and fenced before this one was recorded,
        // so TRANSFER_WRITE is named again for the same harmless reason as above, with TRANSFER
        // added to the stage so the mask is well-formed.
        //
        // A layout this module did not put the image in gets the conservative ALL_COMMANDS /
        // MEMORY_WRITE pair, because nothing here knows who wrote it.
        const bool discard = img.layout == VK_IMAGE_LAYOUT_UNDEFINED ||
                             (img.levels == 1 && img.layers == 1 && x == 0 && y == 0 &&
                              width == levelW && height == levelH);
        const bool ours    = img.layout == VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL ||
                             img.layout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        const bool sampled = img.layout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        const VkImageLayout from = discard ? VK_IMAGE_LAYOUT_UNDEFINED : img.layout;

        VkPipelineStageFlags srcStage  = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
        VkAccessFlags        srcAccess = VK_ACCESS_MEMORY_WRITE_BIT;
        if (sampled)
        {
            srcStage  = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT |
                        (discard ? 0 : VK_PIPELINE_STAGE_TRANSFER_BIT);
            srcAccess = discard ? 0 : VK_ACCESS_TRANSFER_WRITE_BIT;
        }
        else if (discard || ours)
        {
            srcStage  = VK_PIPELINE_STAGE_TRANSFER_BIT;
            srcAccess = discard ? 0 : VK_ACCESS_TRANSFER_WRITE_BIT;
        }

        Transition(s.cmd, img.handle, img.levels, img.layers, from,
                   VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, srcStage, srcAccess,
                   VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT);

        // --- 2. the copy ----------------------------------------------------------------------
        VkBufferImageCopy copy = {};
        // 0 is the VkBuffer's own first byte, not the VkDeviceMemory's: the buffer was bound at its
        // block's offset, and copy offsets are relative to the buffer. (The spec also wants it a
        // multiple of 4 and of the texel block size; 0 is both.)
        copy.bufferOffset = 0;
        // Both 0, and 0 does NOT mean "zero-length rows". It means "tightly packed: take the row
        // length from imageExtent.width and the image height from imageExtent.height". These two
        // fields are in TEXELS, not bytes, and exist for a buffer with padded rows; the repack
        // above is what makes 0 true here, whatever pitch the caller had. Writing a byte pitch
        // here (width * 4) is the classic mistake, and it shears the picture diagonally.
        copy.bufferRowLength                 = 0;
        copy.bufferImageHeight               = 0;
        copy.imageSubresource.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
        copy.imageSubresource.mipLevel       = level;
        // The cube face, for a cube map: layer i is face i, see CreateImage's view comment.
        copy.imageSubresource.baseArrayLayer = layer;
        copy.imageSubresource.layerCount     = 1;
        copy.imageOffset = VkOffset3D{ static_cast<int32_t>(x), static_cast<int32_t>(y), 0 };
        copy.imageExtent = VkExtent3D{ width, height, 1 };

        d.CmdCopyBufferToImage(s.cmd, staging->handle, img.handle,
                               VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);

        // --- 3. the whole image to the layout its owner wants -----------------------------------
        //
        // TRANSFER_WRITE made available either way: the copy's writes have to land before anything
        // reads the image. What they are made visible TO is the owner's choice, through `wanted`:
        //
        //  - TRANSFER_SRC_OPTIMAL, TRANSFER / TRANSFER_READ -- a copy source. The destination
        //    scope covers commands in later submissions too.
        //  - SHADER_READ_ONLY_OPTIMAL, FRAGMENT_SHADER / SHADER_READ -- a sampled texture. The
        //    image is then in the layout a combined-image-sampler descriptor names, so the draw
        //    path binds it with no barrier per frame; that is the whole point of the field.
        //
        // Anything else is not a layout this module knows how to make visible correctly, so it is
        // corrected to TRANSFER_SRC_OPTIMAL and said once, rather than guessed at with a mask that
        // would be silently wrong on some driver.
        VkImageLayout to = img.wanted;
        if (to != VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL &&
            to != VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
        {
            if (!s.wantedLogged)
            {
                s.wantedLogged = true;
                wxl::ScriptMgr::Log(WXL_LOG_WARN,
                                    "vulkan: texture: an upload asked for layout %d, which this "
                                    "module cannot leave an image in; using TRANSFER_SRC_OPTIMAL "
                                    "(logged once)",
                                    static_cast<int>(to));
            }
            to = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        }

        const bool toShader = to == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        Transition(s.cmd, img.handle, img.levels, img.layers, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                   to, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
                   toShader ? VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT : VK_PIPELINE_STAGE_TRANSFER_BIT,
                   toShader ? VK_ACCESS_SHADER_READ_BIT : VK_ACCESS_TRANSFER_READ_BIT);

        r = d.EndCommandBuffer(s.cmd);
        if (r != VK_SUCCESS)
        {
            Fail("vkEndCommandBuffer (upload)", r);
            return false;
        }

        // Reset immediately before the submit that will signal it: no early return can then leave
        // the fence unsignalled with a wait to come.
        r = d.ResetFences(s.device, 1, &s.fence);
        if (r != VK_SUCCESS)
        {
            Fail("vkResetFences (upload)", r);
            return false;
        }

        VkSubmitInfo submit = {};
        submit.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit.commandBufferCount = 1;
        submit.pCommandBuffers    = &s.cmd;

        r = d.QueueSubmit(s.queue, 1, &submit, s.fence);
        if (r != VK_SUCCESS)
        {
            Fail("vkQueueSubmit (upload)", r);
            return false;
        }

        // A fence, not vkQueueWaitIdle. Both stall the render thread, and either would be correct.
        // But the graphics queue is shared with the frame, and vkQueueWaitIdle would also wait out
        // every frame in flight -- a longer stall for no
        // additional guarantee, since the only thing this function needs finished is its own copy
        // (so the staging buffer and the command buffer may be reused). The fence waits for exactly
        // that submission and nothing else.
        r = d.WaitForFences(s.device, 1, &s.fence, VK_TRUE, UINT64_MAX);
        if (r != VK_SUCCESS)
        {
            // The command buffer may still be pending, and recording into a pending command buffer
            // is invalid usage. Refuse further uploads rather than find out.
            s.wedged = true;
            Fail("vkWaitForFences (upload)", r);
            return false;
        }

        img.layout = to;
        return true;
    }
}
