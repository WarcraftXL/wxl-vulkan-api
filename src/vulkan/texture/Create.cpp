// The one create path behind every texture: 2D, mipped, cube, target.
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

namespace wxlvk::vk::texture::detail
{
    /**
     * The one create path behind CreateMipped(), CreateCube() and CreateTarget(): a 2D image with `levels` mip
     * levels and, for a cube, six array layers, CUBE_COMPATIBLE and a cube view. One function
     * rather than two so that the format, limit and memory checks cannot drift apart.
     *
     * @param uint32 width, height : level-0 extent (equal for a cube; CreateCube passes one size)
     * @param uint32 levels : mip levels, 1 .. floor(log2(max(width, height))) + 1
     * @param bool cube : six layers and a VIEW_TYPE_CUBE view, else one layer and VIEW_TYPE_2D
     * @param VkFormat format : one Describe() knows
     * @param Image& out : must be zeroed; receives the image in VK_IMAGE_LAYOUT_UNDEFINED
     * @return bool ok : on failure nothing is left created and `out` is untouched
     */
    bool CreateImage(uint32_t width, uint32_t height, uint32_t levels, bool cube, VkFormat format,
                     Image& out, VkImageUsageFlags extraUsage, bool depth)
    {
        if (!s.ready)
        {
            wxl::ScriptMgr::Log(WXL_LOG_WARN, "vulkan: texture: an image was asked for before Init()");
            return false;
        }
        if (out.handle != VK_NULL_HANDLE)
        {
            // Overwriting a live Image would leak the old one from the caller's point of view.
            wxl::ScriptMgr::Log(WXL_LOG_WARN,
                                "vulkan: texture: an image created into an Image that still holds "
                                "one; it must be destroyed first");
            return false;
        }
        if (width == 0 || height == 0 || width > s.maxDimension || height > s.maxDimension)
        {
            wxl::ScriptMgr::Log(WXL_LOG_WARN,
                                "vulkan: texture: refusing a %ux%u image; this device allows 1..%u "
                                "on each side",
                                width, height, s.maxDimension);
            return false;
        }
        if (levels == 0 || levels > MaxLevels(width, height))
        {
            // The Vulkan ceiling. Whether a DXT chain should stop earlier (at the 4x4 block) is
            // the caller's knowledge, not this module's: a 2x2 BC level is legal Vulkan.
            wxl::ScriptMgr::Log(WXL_LOG_WARN,
                                "vulkan: texture: refusing %u mip level(s) for a %ux%u image; "
                                "1..%u are possible",
                                levels, width, height, MaxLevels(width, height));
            return false;
        }

        const FormatInfo fmt = Describe(format);
        // A depth target is never uploaded, so its texel size is never needed.
        if (fmt.unitBytes == 0 && !depth)
        {
            wxl::ScriptMgr::Log(WXL_LOG_WARN,
                                "vulkan: texture: refusing format %s: its texel size is not known "
                                "here, so an upload's length could not be checked",
                                Name(format));
            return false;
        }

        if (fmt.compressed)
        {
            // The BC family is one optional feature, textureCompressionBC, which Context enables
            // when the device has it. Creating a BC image without it is invalid usage, not a
            // graceful failure; the caller decodes on the CPU instead (src/vulkan/TexFormat.hpp).
            const context::Handles* h = context::Get();
            if (!h || !h->textureCompressionBC)
            {
                wxl::ScriptMgr::Log(WXL_LOG_WARN,
                                    "vulkan: texture: refusing format %s: the device feature "
                                    "textureCompressionBC is not enabled",
                                    Name(format));
                return false;
            }
        }

        if (s.ownedCount >= kMaxImages)
        {
            wxl::ScriptMgr::Log(WXL_LOG_WARN,
                                "vulkan: texture: refusing a new image: %u are already tracked "
                                "(kMaxImages)",
                                s.ownedCount);
            return false;
        }

        // SAMPLED_IMAGE is required: it is a 1.0 bit, it is what the SAMPLED usage below needs,
        // and a format without it is not usable as a texture at all. It also carries the 1.0
        // transfer check: in Vulkan 1.0 there is no per-format TRANSFER_SRC/TRANSFER_DST feature
        // bit (those are VK_KHR_maintenance1, core only from 1.1), and any format the device
        // supports with optimal tiling at all supports transfers. On a 1.1+ driver the two bits
        // exist and are checked as well -- but only there, because on a 1.0 driver they are never
        // set and requiring them would refuse every format.
        VkFormatProperties fp = {};
        s.iapi->GetPhysicalDeviceFormatProperties(s.physical, format, &fp);

        VkFormatFeatureFlags needed = VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT;
        if (s.transferBits)
            needed |= VK_FORMAT_FEATURE_TRANSFER_SRC_BIT | VK_FORMAT_FEATURE_TRANSFER_DST_BIT;
        if (extraUsage & VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT) needed |= VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT;
        if (depth) needed |= VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT;

        if ((fp.optimalTilingFeatures & needed) != needed)
        {
            wxl::ScriptMgr::Log(WXL_LOG_WARN,
                                "vulkan: texture: format %s is not supported with optimal tiling "
                                "for sampling%s on this device (features 0x%08x)",
                                Name(format), s.transferBits ? " and transfer" : "",
                                static_cast<unsigned>(fp.optimalTilingFeatures));
            return false;
        }

        const uint32_t layers = cube ? 6u : 1u;

        VkImageCreateInfo info = {};
        info.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        // A cube map is a 2D image with six layers that a cube view may be made over; the flag is
        // what allows that view, and it is only legal on a square 2D image with layers >= 6.
        info.flags         = cube ? VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT : 0u;
        info.imageType     = VK_IMAGE_TYPE_2D;
        info.format        = format;
        info.extent        = VkExtent3D{ width, height, 1 };
        info.mipLevels     = levels;
        info.arrayLayers   = layers;
        info.samples       = VK_SAMPLE_COUNT_1_BIT;
        // OPTIMAL, not LINEAR. A LINEAR image could be mapped and written directly with no staging
        // buffer, but linear tiling is allowed to support almost nothing -- BLIT_SRC included, on
        // several drivers -- and is slow to read.
        info.tiling        = VK_IMAGE_TILING_OPTIMAL;
        // TRANSFER_DST for the uploads, TRANSFER_SRC for copies out of it and the mip chain's
        // blits, SAMPLED for the draws.
        info.usage         = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                             VK_IMAGE_USAGE_SAMPLED_BIT | extraUsage;
        info.sharingMode   = VK_SHARING_MODE_EXCLUSIVE;
        info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

        VkImage image = VK_NULL_HANDLE;
        VkResult r = s.dapi->CreateImage(s.device, &info, nullptr, &image);
        if (r != VK_SUCCESS)
        {
            Fail("vkCreateImage", r);
            return false;
        }

        VkMemoryRequirements req = {};
        s.dapi->GetImageMemoryRequirements(s.device, image, &req);

        // The block honours req.alignment; that is part of Allocate()'s contract. An OPTIMAL image
        // and a linear resource sharing one VkDeviceMemory also have to be bufferImageGranularity
        // apart, which is likewise the allocator's to enforce and not something this call can see.
        memory::Block block;
        if (!memory::Allocate(req, memory::Where::DeviceLocal, block))
        {
            wxl::ScriptMgr::Log(WXL_LOG_WARN,
                                "vulkan: texture: no device-local memory for a %ux%u %s %s "
                                "with %u level(s) (%llu bytes, alignment %llu)",
                                width, height, Name(format), cube ? "cube map" : "image", levels,
                                static_cast<unsigned long long>(req.size),
                                static_cast<unsigned long long>(req.alignment));
            s.dapi->DestroyImage(s.device, image, nullptr);
            return false;
        }

        r = s.dapi->BindImageMemory(s.device, image, block.memory, block.offset);
        if (r != VK_SUCCESS)
        {
            Fail("vkBindImageMemory", r);
            s.dapi->DestroyImage(s.device, image, nullptr);
            memory::Free(block);
            return false;
        }

        // The view, with the image rather than on first use: a draw binds it through a
        // descriptor, and the descriptor is written on the render thread's per-frame path, where
        // a create call -- and its failure handling -- does not belong. Every level, so the sampler
        // can read the whole chain; identity swizzle, so what was uploaded is what the
        // shader reads (the D3DCOLOR vertex attribute is where the channel order is undone, not
        // here).
        //
        // A cube view over the six layers, in the API's face order (layer i = face i: +X, -X,
        // +Y, -Y, +Z, -Z). samplerCube in a shader then selects the face from the direction
        // vector by the spec's own table, which is the same table D3D9's texCUBE uses.
        VkImageViewCreateInfo view = {};
        view.sType        = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        view.image        = image;
        view.viewType     = cube ? VK_IMAGE_VIEW_TYPE_CUBE : VK_IMAGE_VIEW_TYPE_2D;
        view.format       = format;
        view.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
        view.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
        view.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
        view.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
        view.subresourceRange = WholeColour(levels, layers);
        // A depth target's one view is both its attachment and what a draw samples: a single-
        // aspect format (the native callers pick D32 or D16, never D24S8).
        if (depth) view.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;

        VkImageView imageView = VK_NULL_HANDLE;
        r = s.dapi->CreateImageView(s.device, &view, nullptr, &imageView);
        if (r != VK_SUCCESS)
        {
            Fail("vkCreateImageView", r);
            s.dapi->DestroyImage(s.device, image, nullptr);
            memory::Free(block);
            return false;
        }

        s.owned[s.ownedCount].handle = image;
        s.owned[s.ownedCount].view   = imageView;
        s.owned[s.ownedCount].block  = block;
        HashInsert(s.ownedCount);
        ++s.ownedCount;

        out.handle = image;
        out.view   = imageView;
        out.id     = ++s.nextId;
        out.block  = block;
        out.format = format;
        out.extent = VkExtent2D{ width, height };
        out.levels = levels;
        out.layers = layers;
        out.cube   = cube;
        out.layout = VK_IMAGE_LAYOUT_UNDEFINED;

        return true;
    }
}
