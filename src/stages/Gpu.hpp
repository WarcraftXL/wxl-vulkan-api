// The Vulkan pieces the device's own compute builds from (the stages' working image, the depth history):
// compute pipelines, storage buffers and images, barriers, retirement.
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

#include "stages/Frame.hpp"
#include "vulkan/Loader.hpp"
#include "vulkan/Memory.hpp"
#include "vulkan/Vk.hpp"

#include <cstdint>

// Everything here is device-local: the chain's images and buffers are written and read by the GPU
// only, so none of them costs address space in this 32-bit process.
namespace wxlvk::stages::gpu
{
    /// The device table, or null without a live context.
    const wxlvk::vk::loader::Device* Api();

    /// The device, or null.
    VkDevice Device();

    /**
     * A compute pipeline from shaders/<stem>.spv (entry "main"), through the persisted pipeline cache.
     *
     * @return VkPipeline pipeline : VK_NULL_HANDLE on a failure, logged
     */
    VkPipeline ComputePipeline(const char* stem, VkPipelineLayout layout);

    /**
     * A descriptor set layout of `count` bindings 0 .. count-1, each one descriptor of its type,
     * visible to the compute stage.
     *
     * @return VkDescriptorSetLayout layout : VK_NULL_HANDLE on a failure, logged
     */
    VkDescriptorSetLayout ComputeSetLayout(const VkDescriptorType* types, uint32_t count);

    /// A pipeline layout of one set and one compute push range of `pushBytes`.
    VkPipelineLayout ComputeLayout(VkDescriptorSetLayout set, uint32_t pushBytes);

    /**
     * A descriptor pool holding exactly `count` sets of `layout`, and the sets: one per frame slot.
     *
     * @return bool ok : false on a failure, logged; `pool` is then whatever was made (destroy nothing:
     *         the effect is off for the session)
     */
    bool AllocateSets(VkDescriptorSetLayout layout, const VkDescriptorPoolSize* sizes, uint32_t sizeCount,
                      uint32_t count, VkDescriptorPool& pool, VkDescriptorSet* sets);

    /// A nearest, clamp-to-edge sampler for texelFetch reads (made once, kept).
    VkSampler PointSampler();

    /// A linear, clamp-to-edge sampler, one level (made once, kept).
    VkSampler LinearSampler();

    /// A device-local buffer with STORAGE | TRANSFER_DST and `extraUsage`.
    bool StorageBuffer(VkDeviceSize bytes, VkBufferUsageFlags extraUsage, wxlvk::vk::memory::Buffer& out);

    /// An image of ours: the chain's working images.
    struct OwnedImage
    {
        Image                    img;
        VkExtent2D               extent = { 0, 0 };
        wxlvk::vk::memory::Block block;
    };

    /// A device-local 2D image and its view, one level, `usage`.
    bool MakeImage(VkFormat format, VkExtent2D extent, VkImageUsageFlags usage, OwnedImage& out);

    /// Hands the image to the frame's retirement, to go once `lastUsed` completed. Zeroes `img`.
    void RetireImage(OwnedImage& img, uint64_t lastUsed);

    /// Hands the buffer to the frame's retirement. Zeroes `buf`.
    void RetireBuffer(wxlvk::vk::memory::Buffer& buf, uint64_t lastUsed);

    /// One global memory barrier.
    void Barrier(VkCommandBuffer cmd, VkPipelineStageFlags srcStage, VkAccessFlags srcAccess,
                 VkPipelineStageFlags dstStage, VkAccessFlags dstAccess);

    /// Compute writes made visible to the next compute pass (and to an indirect dispatch's args).
    void ComputeToCompute(VkCommandBuffer cmd);

    /// One image's layout change, whole image, colour aspect, with a full-pipeline scope: only for
    /// the rare transitions (an image's first use).
    void Layout(VkCommandBuffer cmd, VkImage image, VkImageLayout from, VkImageLayout to);

    /// Groups of `size` covering `n`.
    inline uint32_t Groups(uint32_t n, uint32_t size) { return (n + size - 1) / size; }
}
