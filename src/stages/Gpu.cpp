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

#include "stages/Gpu.hpp"

#include "native/Present.hpp"
#include "vulkan/Context.hpp"
#include "vulkan/Pipelines.hpp"
#include "vulkan/Shaders.hpp"

#include "wxl/Script.hpp"

namespace
{
    namespace context = wxlvk::vk::context;
    namespace memory  = wxlvk::vk::memory;

    using wxlvk::vk::ResultName;

    bool Failed(const char* what, VkResult r)
    {
        wxl::ScriptMgr::Log(WXL_LOG_ERROR, "stages: %s failed (%s)", what, ResultName(r));
        return false;
    }

    VkImageView MakeView(VkImage image, VkFormat format)
    {
        VkImageViewCreateInfo v = {};
        v.sType    = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        v.image    = image;
        v.viewType = VK_IMAGE_VIEW_TYPE_2D;
        v.format   = format;
        v.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        v.subresourceRange.levelCount = 1;
        v.subresourceRange.layerCount = 1;
        VkImageView view = VK_NULL_HANDLE;
        const VkResult r = wxlvk::stages::gpu::Api()->CreateImageView(wxlvk::stages::gpu::Device(), &v, nullptr, &view);
        if (r != VK_SUCCESS)
        {
            Failed("vkCreateImageView", r);
            return VK_NULL_HANDLE;
        }
        return view;
    }
}

namespace wxlvk::stages::gpu
{
    const wxlvk::vk::loader::Device* Api()
    {
        const context::Handles* h = context::Get();
        return h ? h->dapi : nullptr;
    }

    VkDevice Device()
    {
        const context::Handles* h = context::Get();
        return h ? h->device : nullptr;
    }

    VkPipeline ComputePipeline(const char* stem, VkPipelineLayout layout)
    {
        const VkShaderModule module = wxlvk::vk::shaders::Get(stem);
        if (module == VK_NULL_HANDLE)
        {
            wxl::ScriptMgr::Log(WXL_LOG_ERROR, "stages: shaders/%s.spv could not be loaded", stem);
            return VK_NULL_HANDLE;
        }
        VkComputePipelineCreateInfo info = {};
        info.sType        = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        info.stage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        info.stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
        info.stage.module = module;
        info.stage.pName  = "main";
        info.layout       = layout;
        VkPipeline pipe = VK_NULL_HANDLE;
        const VkResult r = Api()->CreateComputePipelines(Device(), wxlvk::vk::pipelines::PipelineCache(), 1, &info,
                                                         nullptr, &pipe);
        if (r != VK_SUCCESS)
        {
            wxl::ScriptMgr::Log(WXL_LOG_ERROR, "stages: vkCreateComputePipelines (%s) failed (%s)", stem, ResultName(r));
            return VK_NULL_HANDLE;
        }
        return pipe;
    }

    VkDescriptorSetLayout ComputeSetLayout(const VkDescriptorType* types, uint32_t count)
    {
        VkDescriptorSetLayoutBinding b[16] = {};
        if (count > 16) return VK_NULL_HANDLE;
        for (uint32_t i = 0; i < count; ++i)
        {
            b[i].binding         = i;
            b[i].descriptorType  = types[i];
            b[i].descriptorCount = 1;
            b[i].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
        }
        VkDescriptorSetLayoutCreateInfo info = {};
        info.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        info.bindingCount = count;
        info.pBindings    = b;
        VkDescriptorSetLayout layout = VK_NULL_HANDLE;
        const VkResult r = Api()->CreateDescriptorSetLayout(Device(), &info, nullptr, &layout);
        if (r != VK_SUCCESS) Failed("vkCreateDescriptorSetLayout", r);
        return r == VK_SUCCESS ? layout : VK_NULL_HANDLE;
    }

    VkPipelineLayout ComputeLayout(VkDescriptorSetLayout set, uint32_t pushBytes)
    {
        VkPushConstantRange push = { VK_SHADER_STAGE_COMPUTE_BIT, 0, pushBytes };
        VkPipelineLayoutCreateInfo info = {};
        info.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        info.setLayoutCount         = 1;
        info.pSetLayouts            = &set;
        info.pushConstantRangeCount = pushBytes ? 1u : 0u;
        info.pPushConstantRanges    = pushBytes ? &push : nullptr;
        VkPipelineLayout layout = VK_NULL_HANDLE;
        const VkResult r = Api()->CreatePipelineLayout(Device(), &info, nullptr, &layout);
        if (r != VK_SUCCESS) Failed("vkCreatePipelineLayout", r);
        return r == VK_SUCCESS ? layout : VK_NULL_HANDLE;
    }

    bool AllocateSets(VkDescriptorSetLayout layout, const VkDescriptorPoolSize* sizes, uint32_t sizeCount,
                      uint32_t count, VkDescriptorPool& pool, VkDescriptorSet* sets)
    {
        VkDescriptorPoolCreateInfo pi = {};
        pi.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        pi.maxSets       = count;
        pi.poolSizeCount = sizeCount;
        pi.pPoolSizes    = sizes;
        VkResult r = Api()->CreateDescriptorPool(Device(), &pi, nullptr, &pool);
        if (r != VK_SUCCESS) return Failed("vkCreateDescriptorPool", r);
        VkDescriptorSetLayout layouts[32];
        if (count > 32) return false;
        for (uint32_t i = 0; i < count; ++i) layouts[i] = layout;
        VkDescriptorSetAllocateInfo ai = {};
        ai.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        ai.descriptorPool     = pool;
        ai.descriptorSetCount = count;
        ai.pSetLayouts        = layouts;
        r = Api()->AllocateDescriptorSets(Device(), &ai, sets);
        return r == VK_SUCCESS ? true : Failed("vkAllocateDescriptorSets", r);
    }

    VkSampler LinearSampler()
    {
        static VkSampler sampler = VK_NULL_HANDLE;
        if (sampler != VK_NULL_HANDLE) return sampler;
        VkSamplerCreateInfo si = {};
        si.sType        = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        si.magFilter    = VK_FILTER_LINEAR;
        si.minFilter    = VK_FILTER_LINEAR;
        si.mipmapMode   = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        si.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        si.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        const VkResult r = Api()->CreateSampler(Device(), &si, nullptr, &sampler);
        if (r != VK_SUCCESS)
        {
            Failed("vkCreateSampler", r);
            sampler = VK_NULL_HANDLE;
        }
        return sampler;
    }

    VkSampler PointSampler()
    {
        static VkSampler sampler = VK_NULL_HANDLE;
        if (sampler != VK_NULL_HANDLE) return sampler;
        VkSamplerCreateInfo si = {};
        si.sType        = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        si.magFilter    = VK_FILTER_NEAREST;
        si.minFilter    = VK_FILTER_NEAREST;
        si.mipmapMode   = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        si.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        si.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        const VkResult r = Api()->CreateSampler(Device(), &si, nullptr, &sampler);
        if (r != VK_SUCCESS)
        {
            Failed("vkCreateSampler", r);
            sampler = VK_NULL_HANDLE;
        }
        return sampler;
    }

    bool StorageBuffer(VkDeviceSize bytes, VkBufferUsageFlags extraUsage, memory::Buffer& out)
    {
        const VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | extraUsage;
        if (memory::CreateBuffer(bytes, usage, memory::Where::DeviceLocal, out)) return true;
        wxl::ScriptMgr::Log(WXL_LOG_ERROR, "stages: no device memory for a %llu KiB buffer",
                            static_cast<unsigned long long>(bytes >> 10));
        return false;
    }

    bool MakeImage(VkFormat format, VkExtent2D extent, VkImageUsageFlags usage, OwnedImage& out)
    {
        const wxlvk::vk::loader::Device& d = *Api();
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
        OwnedImage img;
        VkResult r = d.CreateImage(Device(), &info, nullptr, &img.img.image);
        if (r != VK_SUCCESS) return Failed("vkCreateImage", r);
        VkMemoryRequirements req = {};
        d.GetImageMemoryRequirements(Device(), img.img.image, &req);
        if (!memory::Allocate(req, memory::Where::DeviceLocal, img.block)
            || d.BindImageMemory(Device(), img.img.image, img.block.memory, img.block.offset) != VK_SUCCESS)
        {
            d.DestroyImage(Device(), img.img.image, nullptr);
            memory::Free(img.block);
            wxl::ScriptMgr::Log(WXL_LOG_ERROR, "stages: no device memory for a %ux%u image", extent.width, extent.height);
            return false;
        }
        img.img.view = MakeView(img.img.image, format);
        if (img.img.view == VK_NULL_HANDLE)
        {
            d.DestroyImage(Device(), img.img.image, nullptr);
            memory::Free(img.block);
            return false;
        }
        img.img.format = format;
        img.extent     = extent;
        out            = img;
        return true;
    }

    void RetireImage(OwnedImage& img, uint64_t lastUsed)
    {
        if (img.img.image != VK_NULL_HANDLE)
            wxlvk::native::present::RetireRaw(img.img.image, img.img.view, img.block, lastUsed);
        img = OwnedImage();
    }

    void RetireBuffer(memory::Buffer& buf, uint64_t lastUsed)
    {
        wxlvk::native::present::RetireBuffer(buf, lastUsed);
    }

    void Barrier(VkCommandBuffer cmd, VkPipelineStageFlags srcStage, VkAccessFlags srcAccess,
                 VkPipelineStageFlags dstStage, VkAccessFlags dstAccess)
    {
        VkMemoryBarrier m = {};
        m.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        m.srcAccessMask = srcAccess;
        m.dstAccessMask = dstAccess;
        Api()->CmdPipelineBarrier(cmd, srcStage, dstStage, 0, 1, &m, 0, nullptr, 0, nullptr);
    }

    void ComputeToCompute(VkCommandBuffer cmd)
    {
        Barrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT,
                VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_INDIRECT_COMMAND_READ_BIT);
    }

    void Layout(VkCommandBuffer cmd, VkImage image, VkImageLayout from, VkImageLayout to)
    {
        VkImageMemoryBarrier b = {};
        b.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        b.srcAccessMask       = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
        b.dstAccessMask       = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
        b.oldLayout           = from;
        b.newLayout           = to;
        b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.image               = image;
        b.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        b.subresourceRange.levelCount = 1;
        b.subresourceRange.layerCount = 1;
        Api()->CmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0,
                                  nullptr, 0, nullptr, 1, &b);
    }
}
