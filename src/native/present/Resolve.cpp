// Resolving the current target into the pass's colour and depth sides.
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
    // --- resolving the current target ------------------------------------------------------------------

    bool BackColour(Side& out)
    {
        if (s.noImage) return false;
        const context::Handles* h = context::Get();
        if (!h || h->swapchain == VK_NULL_HANDLE || h->extent.width == 0 || h->extent.height == 0
            || !EnsureImageObjects(*h))
        {
            s.noImage = true;
            ++s.stats.noImage;
            return false;
        }
        out = Side();
        out.back   = true;
        out.format = s.colourFormat;
        out.extent = s.extent;
        if (s.ownBack)
        {
            out.image   = s.backImage;
            out.view    = s.backView;
            out.layout  = &s.backLayout;
            out.discard = s.backLayout == VK_IMAGE_LAYOUT_UNDEFINED;
            out.clear   = out.discard;   // a back buffer never drawn: black
        }
        else
        {
            if (!Acquire()) return false;
            out.image   = s.images[s.image];
            out.view    = s.views[s.image];
            out.layout  = &s.swapLayout;
            out.discard = s.swapLayout == VK_IMAGE_LAYOUT_UNDEFINED;
            out.clear   = out.discard;
        }
        return out.view != VK_NULL_HANDLE;
    }

    bool BackDepth(Side& out)
    {
        if (s.depthView == VK_NULL_HANDLE)
        {
            const context::Handles* h = context::Get();
            if (!h || h->swapchain == VK_NULL_HANDLE || !EnsureImageObjects(*h)) return false;
        }
        if (s.depthView == VK_NULL_HANDLE) return false;
        out = Side();
        out.back    = true;
        out.image   = s.depthImage;
        out.view    = s.depthView;
        out.format  = s.depthFormat;
        out.extent  = s.extent;
        out.layout  = &s.depthLayout;
        out.aspect  = AspectOf(s.depthFormat);
        // Discarded and cleared to the far plane at its first use in a frame.
        out.discard = s.depthFresh || s.depthLayout == VK_IMAGE_LAYOUT_UNDEFINED;
        out.clear   = out.discard;
        return true;
    }

    /// D3D's +0x3B38: one depth image for every colour target without a depth texture, grown to the
    /// largest target asked for, never shrunk.
    bool Offscreen(Side& out, VkExtent2D need)
    {
        EnsureDepthFormat();
        if (s.depthFormat == VK_FORMAT_UNDEFINED) return false;
        if (s.offImage == VK_NULL_HANDLE || s.offExtent.width < need.width || s.offExtent.height < need.height)
        {
            const VkExtent2D grow = { need.width > s.offExtent.width ? need.width : s.offExtent.width,
                                      need.height > s.offExtent.height ? need.height : s.offExtent.height };
            if (s.offImage != VK_NULL_HANDLE)
            {
                // Earlier passes of this very frame may have used it: retired by the open serial.
                DropFramebuffersNaming(s.offView, s.serial, false);
                Retired e;
                e.kind     = Kind::Raw;
                e.raw      = s.offImage;
                e.view     = s.offView;
                e.rawBlock = s.offBlock;
                e.serial   = s.serial;
                Retire(e);
                s.offImage = VK_NULL_HANDLE;
                s.offView  = VK_NULL_HANDLE;
                s.offBlock = memory::Block();
            }
            if (!MakeRaw(s.depthFormat, grow, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, s.offImage, s.offView,
                         s.offBlock, "offscreen depth"))
            {
                s.offExtent = VkExtent2D{ 0, 0 };
                return false;
            }
            s.offExtent = grow;
            s.offLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            ++s.stats.offscreenGrows;
            wxl::ScriptMgr::Log(WXL_LOG_DEBUG, "native: frame: offscreen depth (colour targets without a "
                                "depth texture) is now %ux%u %s", grow.width, grow.height,
                                wxlvk::vk::FormatName(s.depthFormat));
        }
        out = Side();
        out.image   = s.offImage;
        out.view    = s.offView;
        out.format  = s.depthFormat;
        out.extent  = s.offExtent;
        out.layout  = &s.offLayout;
        out.aspect  = AspectOf(s.depthFormat);
        out.discard = s.offLayout == VK_IMAGE_LAYOUT_UNDEFINED;
        out.clear   = out.discard;
        return true;
    }

    void TextureSide(texture::Image* img, VkImageView view, uint64_t* stamp, Side& out)
    {
        out = Side();
        out.image   = img->handle;
        out.view    = view;
        out.format  = img->format;
        out.extent  = img->extent;
        out.layout  = &img->layout;
        out.levels  = img->levels;
        out.layers  = img->layers;
        out.aspect  = AspectOf(img->format);
        out.tex     = img;
        out.stamp   = stamp;
        out.discard = img->layout == VK_IMAGE_LAYOUT_UNDEFINED;
        out.clear   = out.discard && IsDepth(img->format);
    }

    VkExtent2D Min(VkExtent2D a, VkExtent2D b)
    {
        return VkExtent2D{ a.width < b.width ? a.width : b.width, a.height < b.height ? a.height : b.height };
    }
}
