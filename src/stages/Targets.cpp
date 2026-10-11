// The post-process chain's working images: two frame-sized storage images in VRAM, shared by every
// effect as a ping-pong pair.
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

#include "stages/Targets.hpp"

#include "stages/Gpu.hpp"

#include "wxl/Script.hpp"

namespace
{
    namespace gpu = wxlvk::stages::gpu;

    constexpr VkImageUsageFlags kUsage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                                         VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;

    struct Working
    {
        gpu::OwnedImage img;
        uint64_t        lastUsed = 0;
    };
    Working g;

    bool Fits(VkExtent2D extent)
    {
        return g.img.img.image != VK_NULL_HANDLE && g.img.extent.width == extent.width && g.img.extent.height == extent.height;
    }
}

namespace wxlvk::stages::targets
{
    bool Get(VkExtent2D extent, VkCommandBuffer cmd, uint64_t serial, Image& out)
    {
        if (!gpu::Api()) return false;
        if (!Fits(extent))
        {
            gpu::RetireImage(g.img, g.lastUsed);
            if (!gpu::MakeImage(kFormat, extent, kUsage, g.img)) return false;
            gpu::Layout(cmd, g.img.img.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
            wxl::ScriptMgr::Log(WXL_LOG_DEBUG, "stages: the working image is %ux%u", extent.width, extent.height);
        }
        g.lastUsed = serial;
        out        = g.img.img;
        return true;
    }

    void Release(uint64_t lastUsed)
    {
        if (g.img.img.image == VK_NULL_HANDLE) return;
        gpu::RetireImage(g.img, lastUsed > g.lastUsed ? lastUsed : g.lastUsed);
    }

    uint64_t Bytes() { return g.img.block.size; }
}
