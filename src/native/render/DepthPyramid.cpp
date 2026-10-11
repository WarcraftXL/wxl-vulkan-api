// The opaque depth's min / max pyramid for the rendering API's consumers (wxl.vk.render GetDepthPyramid).
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

#include "native/render/DepthPyramid.hpp"

#include "native/Present.hpp"
#include "native/Render.hpp"
#include "stages/Gpu.hpp"
#include "vulkan/Context.hpp"
#include "vulkan/Memory.hpp"
#include "vulkan/Pipelines.hpp"

#include "wxl/Script.hpp"

#include <cstring>

namespace
{
    namespace gpu     = wxlvk::stages::gpu;
    namespace present = wxlvk::native::present;
    namespace memory  = wxlvk::vk::memory;

    constexpr uint32_t kSlots     = wxlvk::vk::pipelines::kFrameSlots;
    constexpr uint32_t kMaxLevels = 14;   ///< level 0 up to 8192 texels a side

    struct State
    {
        bool                  tried = false;
        bool                  ok    = false;
        bool                  rgba  = false;   ///< the R32G32B32A32 build (no extended storage formats)
        VkFormat              format = VK_FORMAT_UNDEFINED;
        VkDescriptorSetLayout set    = VK_NULL_HANDLE;
        VkPipelineLayout      layout = VK_NULL_HANDLE;
        VkPipeline            pipe   = VK_NULL_HANDLE;
        VkDescriptorPool      pool   = VK_NULL_HANDLE;
        /// A set per level and frame slot: a slot's sets are rewritten only when that slot records, after its previous
        /// frame completed (a set a pending frame binds is never touched) -- level 0's every frame (the scene depth's view),
        /// the others when the image was remade since the slot last wrote them.
        VkDescriptorSet       sets[kSlots][kMaxLevels] = {};
        uint32_t              setsFor[kSlots] = {};   ///< the image generation each slot's upper sets name
        uint32_t              generation = 0;         ///< bumped at every (re)made image

        VkImage               image = VK_NULL_HANDLE;
        VkImageView           whole = VK_NULL_HANDLE;           ///< every level, for the consumers
        VkImageView           level[kMaxLevels] = {};           ///< one level each, the storage views
        memory::Block         block;
        VkExtent2D            source = { 0, 0 };                ///< the back buffer it was made for
        VkExtent2D            extent = { 0, 0 };                ///< level 0's
        uint32_t              levels = 0;
        bool                  fresh  = false;                   ///< made, never written: UNDEFINED until the first build

        uint64_t              builtSerial = 0;
        uint64_t              lastUsed    = 0;
        float                 depthMin = 0.0f, depthMax = 1.0f;
        uint32_t              requests = 0;
        uint64_t              builds   = 0;
    };
    State g;

    VkExtent2D Half(VkExtent2D e) { return VkExtent2D{ (e.width + 1) / 2, (e.height + 1) / 2 }; }

    bool MakeObjects()
    {
        const wxlvk::vk::context::Handles* h = wxlvk::vk::context::Get();
        if (!h || gpu::PointSampler() == VK_NULL_HANDLE) return false;
        // rg32f storage needs shaderStorageImageExtendedFormats (enabled when offered) and the format's
        // storage bit; rgba32f storage is core.
        VkFormatProperties fp = {};
        h->iapi->GetPhysicalDeviceFormatProperties(h->physical, VK_FORMAT_R32G32_SFLOAT, &fp);
        const bool rg = (wxlvk::vk::context::Caps().enabled & wxlvk::vk::kCapStorageImageExt) != 0
                        && (fp.optimalTilingFeatures & VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT) != 0;
        g.rgba   = !rg;
        g.format = rg ? VK_FORMAT_R32G32_SFLOAT : VK_FORMAT_R32G32B32A32_SFLOAT;
        const VkDescriptorType types[3] = { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                                            VK_DESCRIPTOR_TYPE_STORAGE_IMAGE };
        g.set = gpu::ComputeSetLayout(types, 3);
        if (g.set == VK_NULL_HANDLE) return false;
        g.layout = gpu::ComputeLayout(g.set, 4 * sizeof(int32_t));
        if (g.layout == VK_NULL_HANDLE) return false;
        g.pipe = gpu::ComputePipeline(g.rgba ? "depthpyramid4_comp" : "depthpyramid_comp", g.layout);
        if (g.pipe == VK_NULL_HANDLE) return false;
        // One pool for every slot's sets.
        constexpr uint32_t kSets = kSlots * kMaxLevels;
        const VkDescriptorPoolSize sizes[2] = {
            { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, kSets },
            { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 2 * kSets },
        };
        VkDescriptorSet all[kSets] = {};
        if (!gpu::AllocateSets(g.set, sizes, 2, kSets, g.pool, all)) return false;
        for (uint32_t i = 0; i < kSlots; ++i)
            for (uint32_t l = 0; l < kMaxLevels; ++l) g.sets[i][l] = all[i * kMaxLevels + l];
        wxl::ScriptMgr::Log(WXL_LOG_DEBUG, "render: depth pyramid: %s levels", g.rgba ? "R32G32B32A32_SFLOAT (no extended "
                            "storage formats)" : "R32G32_SFLOAT");
        return true;
    }

    void RetireImage(uint64_t lastUsed)
    {
        if (g.image == VK_NULL_HANDLE) return;
        for (uint32_t i = 0; i < g.levels; ++i)
            if (g.level[i] != VK_NULL_HANDLE) present::RetireView(g.level[i], lastUsed);
        present::RetireRaw(g.image, g.whole, g.block, lastUsed);
        g.image = VK_NULL_HANDLE;
        g.whole = VK_NULL_HANDLE;
        for (VkImageView& v : g.level) v = VK_NULL_HANDLE;
        g.block  = memory::Block();
        g.levels = 0;
        g.builtSerial = 0;
    }

    VkImageView MakeView(uint32_t base, uint32_t count)
    {
        VkImageViewCreateInfo vi = {};
        vi.sType    = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        vi.image    = g.image;
        vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
        vi.format   = g.format;
        vi.subresourceRange.aspectMask   = VK_IMAGE_ASPECT_COLOR_BIT;
        vi.subresourceRange.baseMipLevel = base;
        vi.subresourceRange.levelCount   = count;
        vi.subresourceRange.layerCount   = 1;
        VkImageView view = VK_NULL_HANDLE;
        return gpu::Api()->CreateImageView(gpu::Device(), &vi, nullptr, &view) == VK_SUCCESS ? view : VK_NULL_HANDLE;
    }

    /// The image at the back buffer's size (half of it, rounded up, and its levels).
    bool EnsureImage(VkExtent2D source)
    {
        if (g.image != VK_NULL_HANDLE && g.source.width == source.width && g.source.height == source.height) return true;
        // The old image's last frame may still read it: retired by serial, not destroyed.
        RetireImage(g.lastUsed);
        const VkExtent2D e = Half(source);
        uint32_t levels = 1;
        for (uint32_t w = e.width, h = e.height; (w > 1 || h > 1) && levels < kMaxLevels; ++levels)
        {
            w = (w + 1) / 2;
            h = (h + 1) / 2;
        }
        VkImageCreateInfo ii = {};
        ii.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        ii.imageType     = VK_IMAGE_TYPE_2D;
        ii.format        = g.format;
        ii.extent        = VkExtent3D{ e.width, e.height, 1 };
        ii.mipLevels     = levels;
        ii.arrayLayers   = 1;
        ii.samples       = VK_SAMPLE_COUNT_1_BIT;
        ii.tiling        = VK_IMAGE_TILING_OPTIMAL;
        ii.usage         = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
        ii.sharingMode   = VK_SHARING_MODE_EXCLUSIVE;
        ii.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        const wxlvk::vk::loader::Device& d = *gpu::Api();
        if (d.CreateImage(gpu::Device(), &ii, nullptr, &g.image) != VK_SUCCESS)
        {
            g.image = VK_NULL_HANDLE;
            return false;
        }
        VkMemoryRequirements req = {};
        d.GetImageMemoryRequirements(gpu::Device(), g.image, &req);
        if (!memory::Allocate(req, memory::Where::DeviceLocal, g.block)
            || d.BindImageMemory(gpu::Device(), g.image, g.block.memory, g.block.offset) != VK_SUCCESS)
        {
            d.DestroyImage(gpu::Device(), g.image, nullptr);
            memory::Free(g.block);
            g.image = VK_NULL_HANDLE;
            g.block = memory::Block();
            return false;
        }
        g.levels = levels;
        g.whole  = MakeView(0, levels);
        bool views = g.whole != VK_NULL_HANDLE;
        for (uint32_t i = 0; i < levels && views; ++i)
        {
            g.level[i] = MakeView(i, 1);
            views      = g.level[i] != VK_NULL_HANDLE;
        }
        if (!views)
        {
            RetireImage(0);
            return false;
        }
        g.source = source;
        g.extent = e;
        g.fresh  = true;
        ++g.generation;
        wxl::ScriptMgr::Log(WXL_LOG_DEBUG, "render: depth pyramid: %ux%u, %u level(s)", e.width, e.height, levels);
        return true;
    }

    /// A slot's upper levels' sets for the current image: the level below in, the level out; binding 0 holds the whole
    /// image (never read there, but a statically used binding must be valid).
    void WriteUpper(uint32_t slot)
    {
        if (g.setsFor[slot] == g.generation) return;
        g.setsFor[slot] = g.generation;
        const wxlvk::vk::loader::Device& d = *gpu::Api();
        for (uint32_t i = 1; i < g.levels; ++i)
        {
            const VkDescriptorImageInfo wholeInfo = { gpu::PointSampler(), g.whole, VK_IMAGE_LAYOUT_GENERAL };
            const VkDescriptorImageInfo src       = { VK_NULL_HANDLE, g.level[i - 1], VK_IMAGE_LAYOUT_GENERAL };
            const VkDescriptorImageInfo dst       = { VK_NULL_HANDLE, g.level[i], VK_IMAGE_LAYOUT_GENERAL };
            VkWriteDescriptorSet w[3] = {};
            const VkDescriptorImageInfo* infos[3] = { &wholeInfo, &src, &dst };
            for (uint32_t b = 0; b < 3; ++b)
            {
                w[b].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                w[b].dstSet          = g.sets[slot][i];
                w[b].dstBinding      = b;
                w[b].descriptorCount = 1;
                w[b].descriptorType  = b == 0 ? VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER : VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
                w[b].pImageInfo      = infos[b];
            }
            d.UpdateDescriptorSets(gpu::Device(), 3, w, 0, nullptr);
        }
    }

    void WriteFirst(uint32_t slot, VkImageView depthView)
    {
        const VkDescriptorImageInfo depth = { gpu::PointSampler(), depthView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
        const VkDescriptorImageInfo src   = { VK_NULL_HANDLE, g.level[0], VK_IMAGE_LAYOUT_GENERAL };   // never read at level 0
        const VkDescriptorImageInfo dst   = { VK_NULL_HANDLE, g.level[0], VK_IMAGE_LAYOUT_GENERAL };
        const VkDescriptorImageInfo* infos[3] = { &depth, &src, &dst };
        VkWriteDescriptorSet w[3] = {};
        for (uint32_t b = 0; b < 3; ++b)
        {
            w[b].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w[b].dstSet          = g.sets[slot][0];
            w[b].dstBinding      = b;
            w[b].descriptorCount = 1;
            w[b].descriptorType  = b == 0 ? VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER : VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            w[b].pImageInfo      = infos[b];
        }
        gpu::Api()->UpdateDescriptorSets(gpu::Device(), 3, w, 0, nullptr);
    }
}

namespace wxlvk::native::render::depthpyramid
{
    bool Wanted() { return g.requests > 0; }

    void Build(VkCommandBuffer cmd, VkImageView depthView)
    {
        if (!g.requests || !cmd || depthView == VK_NULL_HANDLE) return;
        VkFormat   format = VK_FORMAT_UNDEFINED;
        VkExtent2D source = {};
        if (!present::BackBufferInfo(format, source) || source.width == 0 || source.height == 0) return;
        WXL_VkSceneConstants scene = {};
        VkViewport           viewport = {};
        float                jitter[2];
        if (!wxlvk::native::render::SceneNow(source, scene, viewport, jitter)) return;
        if (!g.tried)
        {
            g.tried = true;
            g.ok    = MakeObjects();
            if (!g.ok) wxl::ScriptMgr::Log(WXL_LOG_ERROR, "render: depth pyramid: its pipeline could not be made; it stays off");
        }
        if (!g.ok || !EnsureImage(source)) return;
        const uint32_t slot = present::Slot();
        if (slot >= kSlots) return;
        WriteFirst(slot, depthView);
        WriteUpper(slot);

        const wxlvk::vk::loader::Device& d = *gpu::Api();
        // The previous frame's reads of the one image come before this frame's writes (and a new image leaves UNDEFINED).
        VkImageMemoryBarrier b = {};
        b.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        b.srcAccessMask       = g.fresh ? 0 : VK_ACCESS_SHADER_READ_BIT;
        b.dstAccessMask       = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT;
        b.oldLayout           = g.fresh ? VK_IMAGE_LAYOUT_UNDEFINED : VK_IMAGE_LAYOUT_GENERAL;
        b.newLayout           = VK_IMAGE_LAYOUT_GENERAL;
        b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.image               = g.image;
        b.subresourceRange    = { VK_IMAGE_ASPECT_COLOR_BIT, 0, VK_REMAINING_MIP_LEVELS, 0, 1 };
        d.CmdPipelineBarrier(cmd,
                             g.fresh ? VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT
                                     : VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_VERTEX_SHADER_BIT |
                                           VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1, &b);
        g.fresh = false;

        present::GpuLaneBegin(cmd, present::kGpuPyramid);
        d.CmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, g.pipe);
        VkExtent2D src = source;
        VkExtent2D dst = g.extent;
        for (uint32_t i = 0; i < g.levels; ++i)
        {
            const VkDescriptorSet set = g.sets[slot][i];
            const int32_t push[4] = { static_cast<int32_t>(src.width), static_cast<int32_t>(src.height), i == 0 ? 1 : 0, 0 };
            d.CmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, g.layout, 0, 1, &set, 0, nullptr);
            d.CmdPushConstants(cmd, g.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof push, push);
            d.CmdDispatch(cmd, gpu::Groups(dst.width, 8), gpu::Groups(dst.height, 8), 1);
            // The level just written is the next one's input.
            if (i + 1 < g.levels) gpu::Barrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
                                               VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT);
            src = dst;
            dst = Half(dst);
        }
        present::GpuLaneEnd(cmd, present::kGpuPyramid);
        // Visible to every stage that may read it after: the consumers' MidPasses, Records and stages.
        gpu::Barrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
                     VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_VERTEX_SHADER_BIT |
                         VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                     VK_ACCESS_SHADER_READ_BIT);
        g.builtSerial = present::Serial();
        g.lastUsed    = g.builtSerial;
        g.depthMin    = viewport.minDepth;
        g.depthMax    = viewport.maxDepth;
        ++g.builds;
    }

    void Request(bool wanted)
    {
        if (wanted) ++g.requests;
        else if (g.requests > 0) --g.requests;
        if (!g.requests) RetireImage(g.lastUsed > present::Serial() ? g.lastUsed : present::Serial());
    }

    bool Get(WXL_VkDepthPyramid& out)
    {
        if (g.image == VK_NULL_HANDLE || g.builtSerial != present::Serial() || out.structSize < sizeof(uint32_t)) return false;
        WXL_VkDepthPyramid p = {};
        p.structSize = out.structSize;
        p.valid      = 1;
        p.serial     = g.builtSerial;
        p.image      = g.image;
        p.view       = g.whole;
        p.format     = g.format;
        p.layout     = VK_IMAGE_LAYOUT_GENERAL;
        p.extent     = g.extent;
        p.levels     = g.levels;
        p.depthMin   = g.depthMin;
        p.depthMax   = g.depthMax;
        const size_t n = out.structSize < sizeof p ? out.structSize : sizeof p;
        std::memcpy(&out, &p, n);
        out.structSize = static_cast<uint32_t>(n);
        return true;
    }

    uint64_t Builds() { return g.builds; }
}
