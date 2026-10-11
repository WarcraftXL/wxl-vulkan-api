// The textures' owned-image table, format descriptions, layouts and barriers.
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
    State s;

    /**
     * Logs a failed Vulkan call with its numeric result.
     *
     * @param string what : the call that failed, by name
     * @param VkResult r : what it returned
     */
    void Fail(const char* what, VkResult r)
    {
        wxl::ScriptMgr::Log(WXL_LOG_WARN, "vulkan: texture: %s failed: %s (%d)", what,
                            ResultName(r), static_cast<int>(r));
    }

    /**
     * The format table: every VkFormat this module accepts, in one place.
     *
     * A short allow-list rather than a full table of every VkFormat, because the number is used to
     * size the staging copy and to refuse an upload whose byte count is wrong -- and a guessed size
     * for a format nobody has tested is exactly how an upload reads past the caller's buffer.
     * Three-byte formats such as R8G8B8_UNORM are left out on purpose: they are rarely supported
     * with optimal tiling, and a caller should expand to four bytes rather than discover that on
     * some driver.
     *
     * The compressed entries are the three DXT variants the client's BLP files carry (DXT1, DXT3,
     * DXT5 in D3D9's names). They are only usable when the device has textureCompressionBC, which
     * CreateMipped() checks against the context.
     *
     * @param VkFormat f : the format
     * @return FormatInfo info : unitBytes is 0 for a format this module does not handle
     */
    FormatInfo Describe(VkFormat f)
    {
        switch (f)
        {
            case VK_FORMAT_R8_UNORM:                 return { 1,  false, "R8_UNORM" };
            case VK_FORMAT_R8G8_UNORM:               return { 2,  false, "R8G8_UNORM" };
            case VK_FORMAT_R8G8_SNORM:               return { 2,  false, "R8G8_SNORM" };
            case VK_FORMAT_R5G6B5_UNORM_PACK16:      return { 2,  false, "R5G6B5_UNORM_PACK16" };
            case VK_FORMAT_B5G6R5_UNORM_PACK16:      return { 2,  false, "B5G6R5_UNORM_PACK16" };
            case VK_FORMAT_A1R5G5B5_UNORM_PACK16:    return { 2,  false, "A1R5G5B5_UNORM_PACK16" };
            case VK_FORMAT_R8G8B8A8_UNORM:           return { 4,  false, "R8G8B8A8_UNORM" };
            case VK_FORMAT_R8G8B8A8_SRGB:            return { 4,  false, "R8G8B8A8_SRGB" };
            case VK_FORMAT_B8G8R8A8_UNORM:           return { 4,  false, "B8G8R8A8_UNORM" };
            case VK_FORMAT_B8G8R8A8_SRGB:            return { 4,  false, "B8G8R8A8_SRGB" };
            case VK_FORMAT_A2B10G10R10_UNORM_PACK32: return { 4,  false, "A2B10G10R10_UNORM_PACK32" };
            case VK_FORMAT_A2R10G10B10_UNORM_PACK32: return { 4,  false, "A2R10G10B10_UNORM_PACK32" };
            case VK_FORMAT_R16G16_SFLOAT:            return { 4,  false, "R16G16_SFLOAT" };
            case VK_FORMAT_R32_SFLOAT:               return { 4,  false, "R32_SFLOAT" };
            case VK_FORMAT_R16G16B16A16_SFLOAT:      return { 8,  false, "R16G16B16A16_SFLOAT" };
            case VK_FORMAT_R32G32B32A32_SFLOAT:      return { 16, false, "R32G32B32A32_SFLOAT" };
            case VK_FORMAT_BC1_RGBA_UNORM_BLOCK:     return { 8,  true,  "BC1_RGBA_UNORM_BLOCK" };
            case VK_FORMAT_BC2_UNORM_BLOCK:          return { 16, true,  "BC2_UNORM_BLOCK" };
            case VK_FORMAT_BC3_UNORM_BLOCK:          return { 16, true,  "BC3_UNORM_BLOCK" };
            default:                                 return {};
        }
    }

    /**
     * A format's name for the log: the table's, or Vk.hpp's fallback for a format not in it.
     *
     * @param VkFormat f : the format
     * @return string name
     */
    const char* Name(VkFormat f)
    {
        const FormatInfo info = Describe(f);
        return info.name ? info.name : FormatName(f);
    }

    /**
     * The side of a mip level: max(1, side >> level), which is how Vulkan defines it.
     *
     * @param uint32 side : the level-0 width or height
     * @param uint32 level : the mip level
     * @return uint32 side
     */
    uint32_t LevelSide(uint32_t side, uint32_t level)
    {
        const uint32_t v = level >= 32 ? 0 : side >> level;
        return v ? v : 1;
    }

    /**
     * The most mip levels an image of this size may have: floor(log2(max(w, h))) + 1, the level at
     * which both sides have reached 1.
     *
     * @param uint32 width : level-0 width
     * @param uint32 height : level-0 height
     * @return uint32 levels
     */
    uint32_t MaxLevels(uint32_t width, uint32_t height)
    {
        uint32_t side   = width > height ? width : height;
        uint32_t levels = 1;
        while (side > 1)
        {
            side >>= 1;
            ++levels;
        }
        return levels;
    }

    /**
     * Rows in a tightly packed region: texel rows, or block rows (ceil(height / 4)) for a
     * compressed format.
     *
     * @param FormatInfo& info : the format
     * @param uint32 height : the region's height in texels
     * @return uint32 rows
     */
    uint32_t RowCount(const FormatInfo& info, uint32_t height)
    {
        return info.compressed ? (height + kBlock - 1) / kBlock : height;
    }

    /**
     * Bytes in one tightly packed row: width texels, or ceil(width / 4) blocks for a compressed
     * format. In 64 bits because maxImageDimension2D * 16 does not fit in 32 on every device.
     *
     * @param FormatInfo& info : the format
     * @param uint32 width : the region's width in texels
     * @return uint64 bytes
     */
    uint64_t RowBytes(const FormatInfo& info, uint32_t width)
    {
        const uint64_t units = info.compressed ? (width + kBlock - 1) / kBlock : width;
        return units * info.unitBytes;
    }

    /**
     * Confirms every entry point this file calls is non-null, naming the ones that are not.
     *
     * @param loader::Device& d : the device table
     * @param loader::Instance& i : the instance table
     * @return bool ok : false when at least one is null, each logged by name
     */
    bool HaveEntryPoints(const loader::Device& d, const loader::Instance& i)
    {
        struct Entry { const void* fn; const char* name; };
        const Entry entries[] = {
            { reinterpret_cast<const void*>(d.DeviceWaitIdle),             "vkDeviceWaitIdle" },
            { reinterpret_cast<const void*>(d.CreateImage),                "vkCreateImage" },
            { reinterpret_cast<const void*>(d.DestroyImage),               "vkDestroyImage" },
            { reinterpret_cast<const void*>(d.GetImageMemoryRequirements), "vkGetImageMemoryRequirements" },
            { reinterpret_cast<const void*>(d.BindImageMemory),            "vkBindImageMemory" },
            { reinterpret_cast<const void*>(d.CreateImageView),            "vkCreateImageView" },
            { reinterpret_cast<const void*>(d.DestroyImageView),           "vkDestroyImageView" },
            { reinterpret_cast<const void*>(d.CreateCommandPool),          "vkCreateCommandPool" },
            { reinterpret_cast<const void*>(d.DestroyCommandPool),         "vkDestroyCommandPool" },
            { reinterpret_cast<const void*>(d.AllocateCommandBuffers),     "vkAllocateCommandBuffers" },
            { reinterpret_cast<const void*>(d.BeginCommandBuffer),         "vkBeginCommandBuffer" },
            { reinterpret_cast<const void*>(d.EndCommandBuffer),           "vkEndCommandBuffer" },
            { reinterpret_cast<const void*>(d.CmdPipelineBarrier),         "vkCmdPipelineBarrier" },
            { reinterpret_cast<const void*>(d.CmdCopyBufferToImage),       "vkCmdCopyBufferToImage" },
            { reinterpret_cast<const void*>(d.CmdClearColorImage),         "vkCmdClearColorImage" },
            { reinterpret_cast<const void*>(d.CmdBlitImage),               "vkCmdBlitImage" },
            { reinterpret_cast<const void*>(d.CreateFence),                "vkCreateFence" },
            { reinterpret_cast<const void*>(d.DestroyFence),               "vkDestroyFence" },
            { reinterpret_cast<const void*>(d.WaitForFences),              "vkWaitForFences" },
            { reinterpret_cast<const void*>(d.ResetFences),                "vkResetFences" },
            { reinterpret_cast<const void*>(d.QueueSubmit),                "vkQueueSubmit" },
            { reinterpret_cast<const void*>(i.GetPhysicalDeviceProperties),
              "vkGetPhysicalDeviceProperties" },
            { reinterpret_cast<const void*>(i.GetPhysicalDeviceFormatProperties),
              "vkGetPhysicalDeviceFormatProperties" },
        };

        unsigned missing = 0;
        for (const Entry& e : entries)
        {
            if (!e.fn)
            {
                wxl::ScriptMgr::Log(WXL_LOG_WARN, "vulkan: texture: entry point missing: %s", e.name);
                ++missing;
            }
        }

        if (missing)
            wxl::ScriptMgr::Log(WXL_LOG_WARN,
                                "vulkan: texture: %u entry point(s) missing; declining", missing);
        return missing == 0;
    }

    /**
     * Destroys the upload pool and fence. The command buffer goes with the pool.
     *
     * @param loader::Device& d : the device table
     * @param VkDevice device : the device they were created on
     */
    void DropUploadObjects(const loader::Device& d, VkDevice device)
    {
        if (s.fence != VK_NULL_HANDLE) d.DestroyFence(device, s.fence, nullptr);
        s.fence = VK_NULL_HANDLE;

        // vkDestroyCommandPool frees every buffer allocated from it.
        if (s.pool != VK_NULL_HANDLE) d.DestroyCommandPool(device, s.pool, nullptr);
        s.pool = VK_NULL_HANDLE;
        s.cmd  = nullptr;
    }

    /**
     * Finds an image in the ownership table.
     *
     * @param VkImage handle : the image
     * @return uint32 index : kMaxImages when it is not there
     */
    uint32_t FindOwned(VkImage handle)
    {
        if (!s.hash || handle == VK_NULL_HANDLE)
        {
            for (uint32_t i = 0; i < s.ownedCount; ++i)
                if (s.owned[i].handle == handle) return i;
            return kMaxImages;
        }
        for (uint32_t p = HashOf(handle);; p = (p + 1) & (kHashSlots - 1))
        {
            const uint32_t v = s.hash[p];
            if (v == 0) return kMaxImages;
            if (s.owned[v - 1].handle == handle) return v - 1;
        }
    }

    /// The hash slot holding dense index `index` (whose handle is s.owned[index].handle).
    uint32_t HashSlotOf(uint32_t index)
    {
        for (uint32_t p = HashOf(s.owned[index].handle);; p = (p + 1) & (kHashSlots - 1))
            if (s.hash[p] == index + 1 || s.hash[p] == 0) return p;
    }

    /// Files a new dense entry (already written at s.owned[index]) in the index.
    void HashInsert(uint32_t index)
    {
        if (!s.hash) return;
        uint32_t p = HashOf(s.owned[index].handle);
        while (s.hash[p] != 0) p = (p + 1) & (kHashSlots - 1);
        s.hash[p] = index + 1;
    }

    /**
     * Takes dense entry `i` out of the table, keeping it dense (swap-remove) and the index exact:
     * linear probing with backward-shift deletion, so no tombstones accumulate over a session.
     * The entry's objects are the caller's to have destroyed first.
     */
    void RemoveOwned(uint32_t i)
    {
        const uint32_t last = s.ownedCount - 1;
        if (s.hash)
        {
            uint32_t hole = HashSlotOf(i);
            s.hash[hole] = 0;
            for (uint32_t j = (hole + 1) & (kHashSlots - 1); s.hash[j] != 0; j = (j + 1) & (kHashSlots - 1))
            {
                const uint32_t home = HashOf(s.owned[s.hash[j] - 1].handle);
                // The entry at j may move into the hole unless its home lies cyclically in (hole, j].
                const bool stays = hole <= j ? (home > hole && home <= j) : (home > hole || home <= j);
                if (stays) continue;
                s.hash[hole] = s.hash[j];
                s.hash[j]    = 0;
                hole         = j;
            }
            if (i != last) s.hash[HashSlotOf(last)] = i + 1;
        }
        s.owned[i]    = s.owned[last];
        s.owned[last] = Owned();
        --s.ownedCount;
    }

    void FreeTables()
    {
        delete[] s.owned;
        delete[] s.hash;
        s.owned = nullptr;
        s.hash  = nullptr;
    }

    /**
     * The whole-image colour subresource range: every mip level and every array layer -- one
     * layer for what CreateMipped() makes, six for a CreateCube() image.
     *
     * Every barrier in this file goes through here, with the image's full level count, and that is
     * a decision rather than an accident. Image tracks ONE layout for the whole image. A per-level
     * layout was considered for UploadRegion(), which touches one level at a time, and rejected:
     * a barrier on one level would leave the others in a different layout than the tracked field
     * claims, making that field a lie -- and every later barrier reads it as its oldLayout, so the
     * lie would turn into invalid usage no driver reports.
     *
     * The same argument covers layers: an upload writes one face of a cube map, and a barrier
     * on that face alone would leave the other five in a layout the tracked field no longer
     * describes. Every barrier names all of them.
     *
     * @param uint32 levels : the image's mip level count (1 for a swapchain image)
     * @param uint32 layers : the image's array layer count (1, or 6 for a cube map)
     * @return VkImageSubresourceRange range
     */
    VkImageSubresourceRange WholeColour(uint32_t levels, uint32_t layers)
    {
        VkImageSubresourceRange range = {};
        range.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
        range.baseMipLevel   = 0;
        range.levelCount     = levels;
        range.baseArrayLayer = 0;
        range.layerCount     = layers;
        return range;
    }

    /**
     * One image layout transition, recorded as its own vkCmdPipelineBarrier.
     *
     * A helper and not an inline struct each time, because every barrier in this file differs only
     * in the six values passed here -- and the queue family indices, the one field that is
     * dangerous to forget, are then written exactly once. They are IGNORED because no barrier here
     * is a queue family ownership transfer: every image this module makes is EXCLUSIVE and only ever
     * used on the graphics queue.
     *
     * @param VkCommandBuffer cmd : recording
     * @param VkImage image : the image
     * @param uint32 levels : its mip level count; the barrier covers all of them (see WholeColour)
     * @param uint32 layers : its array layer count; likewise all of them
     * @param VkImageLayout from : its current layout, or UNDEFINED to discard its contents
     * @param VkImageLayout to : the layout wanted
     * @param VkPipelineStageFlags srcStage : what must finish before the transition
     * @param VkAccessFlags srcAccess : the writes to make available
     * @param VkPipelineStageFlags dstStage : what waits for the transition
     * @param VkAccessFlags dstAccess : the accesses to make the image visible to
     */
    void Transition(VkCommandBuffer cmd, VkImage image, uint32_t levels, uint32_t layers,
                    VkImageLayout from, VkImageLayout to, VkPipelineStageFlags srcStage,
                    VkAccessFlags srcAccess, VkPipelineStageFlags dstStage, VkAccessFlags dstAccess)
    {
        VkImageMemoryBarrier b = {};
        b.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        b.srcAccessMask       = srcAccess;
        b.dstAccessMask       = dstAccess;
        b.oldLayout           = from;
        b.newLayout           = to;
        b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.image               = image;
        b.subresourceRange    = WholeColour(levels, layers);

        s.dapi->CmdPipelineBarrier(cmd, srcStage, dstStage, 0, 0, nullptr, 0, nullptr, 1, &b);
    }
}
