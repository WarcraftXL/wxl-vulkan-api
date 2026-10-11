// The textures' shared internals: the owned-image table, the format descriptions, the state.
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

#include "vulkan/Texture.hpp"

#include "vulkan/Context.hpp"
#include "vulkan/Loader.hpp"
#include "vulkan/Memory.hpp"

#include "wxl/Script.hpp"

#include <cstdint>
#include <cstring>
#include <new>

// The textures live in src/vulkan/texture/ over this one header; their public face is src/vulkan/Texture.hpp
// (src/vulkan/Texture.cpp).
//
//   Registry  the owned-image table, the format descriptions, the layouts and barriers
//   Create    the one create path (2D, mipped, cube, target)
//   Upload    a region upload, staged or inline
namespace wxlvk::vk::texture::detail
{
    namespace context = wxlvk::vk::context;
    namespace loader  = wxlvk::vk::loader;
    namespace memory  = wxlvk::vk::memory;

    using wxlvk::vk::FormatName;
    using wxlvk::vk::ResultName;
    using Image = wxlvk::vk::texture::Image;

    /// The most images this module tracks at once.
    ///
    /// A fixed table rather than a std::vector: a bounded cost, and a full table is a logged
    /// refusal rather than silent growth. One image per engine texture with an image (render
    /// targets included); a zone holds a few thousand. Heap-allocated at Init() (~450 KiB) rather
    /// than static, so it does not ship in the DLL's .data.
    constexpr uint32_t kMaxImages = 16384;

    /// The index over the table, twice its size (a load factor of at most one half keeps
    /// linear probing short), a power of two for the mask.
    constexpr uint32_t kHashSlots = 2 * kMaxImages;
    static_assert((kHashSlots & (kHashSlots - 1)) == 0, "the probe masks with kHashSlots - 1");

    /// A VkImage handle (a 64-bit value on this 32-bit build, often an aligned pointer or a small
    /// counter) mixed down to a slot: the finaliser of MurmurHash3's 64-bit variant.
    inline uint32_t HashOf(VkImage handle)
    {
        uint64_t v = static_cast<uint64_t>(handle);
        v ^= v >> 33;
        v *= 0xFF51AFD7ED558CCDull;
        v ^= v >> 33;
        return static_cast<uint32_t>(v) & (kHashSlots - 1);
    }

    /// One image this module created and has not yet destroyed.
    struct Owned
    {
        VkImage       handle = VK_NULL_HANDLE;
        VkImageView   view   = VK_NULL_HANDLE; ///< destroyed before the image, on every path
        memory::Block block;
    };

    /// Everything this file owns, in one place.
    ///
    /// VkCommandBuffer, VkDevice, VkQueue and VkPhysicalDevice are dispatchable and therefore
    /// pointers; VkCommandPool, VkFence and VkImage are uint64_t on this 32-bit build and are
    /// compared against VK_NULL_HANDLE, never nullptr.
    struct State
    {
        VkDevice                device   = nullptr;
        VkPhysicalDevice        physical = nullptr;
        VkQueue                 queue    = nullptr;
        const loader::Device*   dapi     = nullptr;
        const loader::Instance* iapi     = nullptr;

        VkCommandPool   pool  = VK_NULL_HANDLE;
        VkCommandBuffer cmd   = nullptr;
        VkFence         fence = VK_NULL_HANDLE;

        uint32_t maxDimension     = 0; ///< VkPhysicalDeviceLimits::maxImageDimension2D
        uint32_t maxDimensionCube = 0; ///< VkPhysicalDeviceLimits::maxImageDimensionCube
        bool transferBits = false; ///< the driver is 1.1+, so FORMAT_FEATURE_TRANSFER_* are meaningful

        Owned*   owned      = nullptr; ///< kMaxImages entries, allocated at Init(), dense
        uint32_t ownedCount = 0;
        uint32_t nextId     = 0;       ///< Image::id of the last image made; never reused
        /// An open-addressed index over `owned` (kHashSlots entries, dense index + 1, 0 empty), so
        /// FindOwned() is a probe and not a scan of up to 16384 images: a login screen destroys
        /// hundreds of images and a zone change thousands.
        uint32_t* hash      = nullptr;

        bool wantedLogged  = false; ///< an upload has already complained about an odd `wanted`
        bool wedged        = false; ///< an upload's wait failed; the command buffer may be pending
        bool ready         = false;
    };
    extern State s;


    /// Everything this module knows about one VkFormat: the unit a row is made of, and its size.
    ///
    /// For an uncompressed format the unit is a texel; for a block-compressed one it is a 4x4
    /// block, and a "row" everywhere in this file is then a row of blocks covering four texel rows.
    struct FormatInfo
    {
        uint32_t    unitBytes  = 0;       ///< bytes per texel, or per 4x4 block; 0 = not accepted
        bool        compressed = false;   ///< the unit is a 4x4 block
        const char* name       = nullptr; ///< for the log; Vk.hpp's FormatName knows only a few
    };

    /// The side of a block, in texels, for every compressed format here. BC1/2/3 are all 4x4.
    constexpr uint32_t kBlock = 4;

    // Registry.cpp
    void Fail(const char* what, VkResult r);
    FormatInfo Describe(VkFormat f);
    const char* Name(VkFormat f);
    uint32_t LevelSide(uint32_t side, uint32_t level);
    uint32_t MaxLevels(uint32_t width, uint32_t height);
    uint32_t RowCount(const FormatInfo& info, uint32_t height);
    uint64_t RowBytes(const FormatInfo& info, uint32_t width);
    bool HaveEntryPoints(const loader::Device& d, const loader::Instance& i);
    void DropUploadObjects(const loader::Device& d, VkDevice device);
    uint32_t FindOwned(VkImage handle);
    uint32_t HashSlotOf(uint32_t index);
    void HashInsert(uint32_t index);
    void RemoveOwned(uint32_t i);
    void FreeTables();
    VkImageSubresourceRange WholeColour(uint32_t levels, uint32_t layers);
    void Transition(VkCommandBuffer cmd, VkImage image, uint32_t levels, uint32_t layers, VkImageLayout from, VkImageLayout to, VkPipelineStageFlags srcStage, VkAccessFlags srcAccess, VkPipelineStageFlags dstStage, VkAccessFlags dstAccess);

    // Create.cpp
    /// The shared create path behind CreateMipped(), CreateCube() and CreateTarget(). The two last
    /// parameters are CreateTarget()'s: extra usage bits and a depth image.
    bool CreateImage(uint32_t width, uint32_t height, uint32_t levels, bool cube, VkFormat format, Image& out,
                     VkImageUsageFlags extraUsage = 0, bool depth = false);
}
