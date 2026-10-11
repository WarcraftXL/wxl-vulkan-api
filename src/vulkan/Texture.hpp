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

#pragma once

#include "vulkan/Memory.hpp"
#include "vulkan/Vk.hpp"

#include <cstddef>
#include <cstdint>

// Every image is OPTIMAL-tiled, TRANSFER_DST | TRANSFER_SRC | SAMPLED (plus an attachment usage for a
// render target), with a view made at creation. Uploads either submit and wait (UploadRegion,
// ClearAll: the synchronous paths) or are recorded into a command buffer the caller submits
// (RecordRegions, RecordClear, RecordGenerateMips: the frame's pre-pass or main buffer).
//
// Ownership: this module owns the images it creates, not the caller. The caller's Image is a view
// of one of them; a DestroyIdle() of an image this module does not hold is recognised and does
// nothing, rather than destroying a handle twice.
//
// Init() after memory::Init(): every image's memory is a memory::Block.
//
// Every function here must run on the render thread. UploadRegion() submits to the graphics queue,
// and a VkQueue is externally synchronised: two threads submitting to it at once is invalid usage,
// which a lock here would hide rather than prevent.
namespace wxlvk::vk::texture
{
    /// One image, with the layout it is currently in.
    struct Image
    {
        VkImage        handle = VK_NULL_HANDLE;
        /// A view of every level and every layer, in the image's own format, created with the
        /// image: VIEW_TYPE_2D for a 2D image, VIEW_TYPE_CUBE for a cube map. What a descriptor
        /// binds; a pipeline never sees the VkImage itself. A shader's sampler type must match
        /// the view type (sampler2D on a cube view is invalid usage), which is why `cube` is
        /// kept beside it for the draw path to check.
        VkImageView    view   = VK_NULL_HANDLE;
        /// This module's own number for the image, from 1 up, never given twice in a session: what
        /// a draw-path cache keys an image by across frames, since a destroyed image's handles
        /// (and its view's) are reused by the driver and its id is not. 0: not made here.
        uint32_t       id     = 0;
        memory::Block  block;
        VkFormat       format = VK_FORMAT_UNDEFINED;
        VkExtent2D     extent = { 0, 0 };                   ///< of mip level 0
        uint32_t       levels = 1;                          ///< mip levels the image was created with
        uint32_t       layers = 1;                          ///< array layers: 1, or 6 for a cube map
        bool           cube   = false;                      ///< created by CreateCube(); `view` is a cube view
        VkImageLayout  layout = VK_IMAGE_LAYOUT_UNDEFINED;  ///< tracked, so a barrier knows its source;
                                                            ///< one value for every level and layer (see Texture.cpp)
        /// The layout an upload or a clear leaves the image in. SHADER_READ_ONLY_OPTIMAL for every
        /// texture (set at creation, src/native/Textures.cpp), so it is already in the layout a
        /// descriptor needs and no per-frame barrier is spent on it; TRANSFER_SRC_OPTIMAL is the
        /// default for an image nothing has asked for. Only those two values are meaningful; see
        /// UploadRegion.
        VkImageLayout  wanted = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    };

    /**
     * Prepares the image table and the module's own upload command pool, command buffer and fence.
     * Idempotent.
     *
     * Needs a live context. Checks every entry point this module calls, once, here -- so nothing
     * further down tests a function pointer, and in particular nothing on the per-frame path does.
     *
     * @return bool ok : false when there is no context, an entry point is missing (each logged by
     *         name), or a create failed; on false nothing this call created is left behind
     */
    bool Init();

    /**
     * Creates an OPTIMAL-tiled, TRANSFER_DST | TRANSFER_SRC | SAMPLED 2D image with `levels` mip
     * levels, binds DeviceLocal memory to it, and creates its `view` (2D, the image's format,
     * every level, colour aspect). Level i has extent max(1, width >> i) x max(1, height >> i).
     *
     * Block-compressed formats (VK_FORMAT_BC1_RGBA_UNORM_BLOCK, BC2, BC3) are accepted when the
     * context reports `textureCompressionBC`, and refused with a log line when it does not. The
     * caller is responsible for not asking for more levels than the format allows: the engine never
     * goes below a 4x4 block for DXT, and this function only enforces the Vulkan ceiling of
     * floor(log2(max(width, height))) + 1.
     *
     * @param uint32 width : texels, 1 .. maxImageDimension2D
     * @param uint32 height : texels, 1 .. maxImageDimension2D
     * @param uint32 levels : mip levels, 1 .. floor(log2(max(width, height))) + 1
     * @param VkFormat format : one Describe() in Texture.cpp knows
     * @param Image& out : must be zeroed; receives the image in VK_IMAGE_LAYOUT_UNDEFINED
     * @return bool ok : on failure nothing is left created and `out` is untouched
     */
    bool CreateMipped(uint32_t width, uint32_t height, uint32_t levels, VkFormat format, Image& out);

    /**
     * Creates a cube map: as CreateMipped(size, size, levels, format, out), but with six array
     * layers, VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT, and a VK_IMAGE_VIEW_TYPE_CUBE view over all
     * six. `out.layers` is 6 and `out.cube` is true.
     *
     * Layer order is the API's: +X, -X, +Y, -Y, +Z, -Z (VkImageSubresourceRange layer i is cube
     * face i in Vulkan's own numbering, 16.5.4 "Cube map face selection"). D3D9's
     * D3DCUBEMAP_FACES enumerates the same six in the same order, so a caller copying D3D faces
     * uploads face i into layer i with no permutation; see UploadRegion's `layer`.
     *
     * @param uint32 size : texels on each side, 1 .. maxImageDimensionCube
     * @param uint32 levels : mip levels, 1 .. floor(log2(size)) + 1
     * @param VkFormat format : one Describe() in Texture.cpp knows
     * @param Image& out : must be zeroed; receives the image in VK_IMAGE_LAYOUT_UNDEFINED
     * @return bool ok : on failure nothing is left created and `out` is untouched
     */
    bool CreateCube(uint32_t size, uint32_t levels, VkFormat format, Image& out);

    /**
     * A render target: CreateMipped / CreateCube
     * plus COLOR_ATTACHMENT usage (the format's COLOR_ATTACHMENT feature checked), or, with `depth`,
     * a single-level 2D image with DEPTH_STENCIL_ATTACHMENT | SAMPLED usage whose `view` has the
     * depth aspect -- the view is both its attachment and what a draw samples, so `format` must be a
     * single-aspect depth format (D32_SFLOAT or D16_UNORM). A depth image must never be passed to
     * ClearAll() or an upload: those are colour-aspect paths.
     *
     * @param uint32 width, height : level-0 extent (equal for a cube)
     * @param uint32 levels : mip levels (1 for depth)
     * @param bool cube : six layers and a cube view (colour only)
     * @param VkFormat format : the storage format
     * @param bool depth : a depth target
     * @param Image& out : must be zeroed; receives the image in VK_IMAGE_LAYOUT_UNDEFINED
     * @return bool ok
     */
    bool CreateTarget(uint32_t width, uint32_t height, uint32_t levels, bool cube, VkFormat format,
                      bool depth, Image& out);

    /**
     * Fills every level of every layer of a freshly created image with one known colour and
     * leaves it in `img.wanted` (or SHADER_READ_ONLY_OPTIMAL when `wanted` is unset).
     *
     * A VkImage is born with undefined contents, and a level the engine never uploads keeps
     * them -- noise where a texture's lower mips are sampled by distance before anything wrote
     * them. This guarantees that whatever is sampled is a colour and not VRAM. One submit for all
     * levels, synchronous like UploadRegion; RecordClear() is the recorded form.
     *
     * @param Image& img : created by Create/CreateMipped, any layout
     * @param const float* rgba : the colour, r g b a; null for opaque mid grey (render targets are
     *        born transparent black)
     * @return bool ok : false when the upload objects are wedged or the submit failed
     */
    bool ClearAll(Image& img, const float* rgba = nullptr);

    /**
     * Uploads a rectangular region of one mip level from pitched source rows, through the staging
     * buffer, and leaves the WHOLE image in `img.wanted` -- TRANSFER_SRC_OPTIMAL for a copy
     * source, SHADER_READ_ONLY_OPTIMAL for a sampled one; any other value is treated as
     * TRANSFER_SRC_OPTIMAL and logged once. Submits and waits: the fallback when the frame's upload
     * ring has no room at all. Nothing is logged on success.
     *
     * The rows are repacked tightly into the staging buffer, so the caller's pitch never reaches
     * vkCmdCopyBufferToImage; see the copy in Texture.cpp for why that is the only safe route.
     *
     * For a block-compressed format the region is in texels but must sit on the 4x4 block grid:
     * `x` and `y` multiples of 4, `width` and `height` multiples of 4 or reaching exactly the
     * level's edge. Anything else is refused with a log line rather than recorded.
     *
     * @param Image& img : from Create*(); its `layout` is updated
     * @param uint32 level : the mip level, < img.levels
     * @param uint32 x, y : the region's origin in texels (BC: multiples of 4, or 0)
     * @param uint32 width, height : the region in texels, non-zero, within the level (BC: multiples
     *        of 4, or exactly the level's remaining extent)
     * @param const void* rows : the first row of the region in the caller's memory
     * @param uint32 rowPitch : bytes between consecutive source rows (BC: between block rows); at
     *        least the tight row size
     * @param uint32 layer : the array layer, < img.layers: 0 for a 2D image, the cube face
     *        0..5 (+X, -X, +Y, -Y, +Z, -Z) for one from CreateCube()
     * @return bool ok
     */
    bool UploadRegion(Image& img, uint32_t level, uint32_t x, uint32_t y,
                      uint32_t width, uint32_t height, const void* rows, uint32_t rowPitch,
                      uint32_t layer = 0);

    /**
     * The tightly packed size of a region of `img`'s format: bytes per row (per block row for a
     * BC format) and the row count. What RecordRegions() expects its source buffer to hold, so a
     * caller filling a staging span packs exactly what the copy will read.
     *
     * @param const Image& img : from Create*()
     * @param uint32 width, height : the region in texels
     * @param uint64& rowBytes : receives the tight row size
     * @param uint32& rowCount : receives the number of rows (block rows for BC)
     * @return bool ok : false for a format this module does not describe or an empty region
     */
    bool RegionBytes(const Image& img, uint32_t width, uint32_t height, uint64_t& rowBytes,
                     uint32_t& rowCount);

    /**
     * Destroys the view, then the image, and frees its block, for a caller that has PROVEN the GPU
     * is done with the image: the frame retires images by serial and calls this only once the
     * fence of the last frame that named the image has signalled. No device wait.
     *
     * @param Image& img : from Create*(); zeroed on return
     */
    void DestroyIdle(Image& img);

    /**
     * Whether an image of this format can have its mip chain generated by RecordGenerateMips():
     * optimal tiling with BLIT_SRC, BLIT_DST and linear filtering, not block-compressed. Asked
     * before a generate-mips texture (CGxTexFlags bit 6) is sized for its full chain.
     *
     * @param VkFormat format : the image's format
     * @return bool ok
     */
    bool CanGenerateMips(VkFormat format);

    /// One region of a batched upload (RecordRegions): as UploadRegion's arguments, with the
    /// packed source's offset in the batch's one buffer.
    struct RegionCopy
    {
        uint32_t     level  = 0;
        uint32_t     x      = 0;
        uint32_t     y      = 0;
        uint32_t     width  = 0;
        uint32_t     height = 0;
        uint32_t     layer  = 0;
        VkDeviceSize srcOffset = 0;
    };

    /**
     * UploadRegion()'s copies RECORDED into a command buffer the caller owns, for a batch -- every
     * region of one fill protocol run (each level, each face) of one image, from one source buffer
     * -- with ONE barrier before the copies and ONE after. The same per-region checks as
     * UploadRegion; a region that fails them is left out. The first barrier waits for earlier
     * sampler reads and earlier transfers of the image (a clear or a batch recorded before it in
     * the same buffer); the tracked layout ends at `img.wanted` as after UploadRegion. Outside a
     * render pass; the source buffer lives until the submission completes.
     *
     * @param VkCommandBuffer cmd : recording, outside a render pass
     * @param Image& img : from Create*(); its `layout` is updated
     * @param VkBuffer src : TRANSFER_SRC buffer holding every region, packed (RegionBytes)
     * @param const RegionCopy* regions : `count` regions
     * @param uint32 count : how many
     * @return uint32 recorded : regions recorded (0: nothing was recorded at all)
     */
    uint32_t RecordRegions(VkCommandBuffer cmd, Image& img, VkBuffer src, const RegionCopy* regions,
                           uint32_t count);

    /**
     * ClearAll()'s commands recorded into a command buffer the caller submits
     * (the native frame's pre-pass) instead of a submission of their own and a fence wait. Every
     * level and layer to one colour, then `img.wanted` (SHADER_READ_ONLY_OPTIMAL when unset), which
     * becomes the tracked layout. Outside a render pass.
     *
     * @param VkCommandBuffer cmd : recording
     * @param Image& img : created by Create*() (colour; never a depth target)
     * @param const float* rgba : as ClearAll's
     * @return bool ok : false when nothing was recorded
     */
    bool RecordClear(VkCommandBuffer cmd, Image& img, const float* rgba = nullptr);

    /**
     * Records the mip chain of `img` derived from its level 0 -- what D3D9's AUTOGENMIPMAP does
     * after a level-0 update -- as a vkCmdBlitImage per level (linear, every layer at once), with
     * the barriers between levels. Outside a render pass. The first barrier waits for every
     * earlier sampler read and transfer write of the image, so it may follow a level-0 upload
     * recorded in the same buffer or submitted before it; the image ends in `img.wanted`'s layout,
     * which becomes the tracked one. The caller has checked CanGenerateMips(img.format).
     *
     * @param VkCommandBuffer cmd : recording, outside a render pass
     * @param Image& img : from CreateMipped/CreateCube with levels >= 2
     * @return bool ok : false when nothing was recorded
     */
    bool RecordGenerateMips(VkCommandBuffer cmd, Image& img);
}
