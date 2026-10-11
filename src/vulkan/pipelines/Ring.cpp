// The uniform ring and the bone-palette ring, and their spills.
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

#include "vulkan/pipelines/Internal.hpp"

namespace wxlvk::vk::pipelines::detail
{
    // --- the ring -----------------------------------------------------------------------------

    /// Logs a spill that could not be made, once.
    void SpillFailed(const char* what, VkResult r)
    {
        if (s.loggedSpillFailed) return;
        s.loggedSpillFailed = true;
        wxl::ScriptMgr::Log(WXL_LOG_WARN, "vulkan: pipelines: %s could not be created (%s); the "
                            "frame's further draws are counted as exhausted (logged once)",
                            what, ResultName(r));
    }

    /**
     * Moves a slot whose ring is full to its next spill buffer, made on first use
     * (one host-visible buffer of kRingBytes and its own set-1 descriptor, from a small pool of
     * its own). The cursor restarts at 0 in it; BeginFrame() moves the slot back to its own ring.
     *
     * @param uint32 slot : the frame slot
     * @return bool ok : false past kRingSpill, or when a create failed
     */
    bool SpillRing(uint32_t slot)
    {
        if (s.ringActive[slot] >= kRingSpill) return false;
        const uint32_t i = s.ringActive[slot];   // the spill index the slot moves to
        const loader::Device& d = *s.dapi;
        if (s.ringSpill[slot][i].handle == VK_NULL_HANDLE)
        {
            if (s.ringSpillPool == VK_NULL_HANDLE)
            {
                VkDescriptorPoolSize size = {};
                size.type            = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;
                size.descriptorCount = wxlvk::vk::pipelines::kFrameSlots * kRingSpill;
                VkDescriptorPoolCreateInfo pool = {};
                pool.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
                pool.maxSets       = wxlvk::vk::pipelines::kFrameSlots * kRingSpill;
                pool.poolSizeCount = 1;
                pool.pPoolSizes    = &size;
                const VkResult r = d.CreateDescriptorPool(s.device, &pool, nullptr, &s.ringSpillPool);
                if (r != VK_SUCCESS)
                {
                    s.ringSpillPool = VK_NULL_HANDLE;
                    SpillFailed("the ring spill's descriptor pool", r);
                    return false;
                }
            }
            memory::Buffer& b = s.ringSpill[slot][i];
            if (!memory::CreateBuffer(kRingBytes, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, memory::Where::Stream, b)
                || !b.block.mapped)
            {
                memory::DestroyBuffer(b);
                SpillFailed("a ring spill buffer", VK_ERROR_OUT_OF_DEVICE_MEMORY);
                return false;
            }
            std::memset(b.block.mapped, 0, kRingBytes);   // as the slot's own ring (CreateRing)
            VkDescriptorSetAllocateInfo alloc = {};
            alloc.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
            alloc.descriptorPool     = s.ringSpillPool;
            alloc.descriptorSetCount = 1;
            alloc.pSetLayouts        = &s.regsLayout;
            const VkResult r = d.AllocateDescriptorSets(s.device, &alloc, &s.ringSpillSets[slot][i]);
            if (r != VK_SUCCESS)
            {
                s.ringSpillSets[slot][i] = VK_NULL_HANDLE;
                memory::DestroyBuffer(b);
                SpillFailed("a ring spill descriptor set", r);
                return false;
            }
            VkDescriptorBufferInfo info = {};
            info.buffer = b.handle;
            info.offset = 0;
            info.range  = kRingRange;
            VkWriteDescriptorSet write = {};
            write.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            write.dstSet          = s.ringSpillSets[slot][i];
            write.dstBinding      = 0;
            write.descriptorCount = 1;
            write.descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;
            write.pBufferInfo     = &info;
            d.UpdateDescriptorSets(s.device, 1, &write, 0, nullptr);
            ++s.stats.ringSpillBuffers;
            wxl::ScriptMgr::Log(WXL_LOG_DEBUG, "vulkan: pipelines: frame slot %u spilled into ring buffer "
                                "#%u (%u KiB more uniform room per frame)", slot, i + 2, kRingBytes / 1024);
        }
        s.ringActive[slot] = i + 1;
        s.ringCursor[slot] = 0;
        ++s.stats.ringSpills;
        if (s.ringActive[slot] + 1 > s.stats.ringMostBuffers) s.stats.ringMostBuffers = s.ringActive[slot] + 1;
        return true;
    }

    /**
     * Moves a slot whose descriptor pool is exhausted to its next spill pool, made
     * on first use with the same size as the slot's own. BeginFrame() resets them all.
     *
     * @param uint32 slot : the frame slot
     * @return bool ok : false past kPoolSpill, or when the create failed
     */
    bool SpillPool(uint32_t slot)
    {
        if (s.poolActive[slot] >= kPoolSpill) return false;
        const uint32_t i = s.poolActive[slot];
        if (s.poolSpill[slot][i] == VK_NULL_HANDLE)
        {
            VkDescriptorPoolSize size = {};
            size.type            = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            size.descriptorCount = kSetsPerPool * 4;
            VkDescriptorPoolCreateInfo pool = {};
            pool.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
            pool.maxSets       = kSetsPerPool;
            pool.poolSizeCount = 1;
            pool.pPoolSizes    = &size;
            const VkResult r = s.dapi->CreateDescriptorPool(s.device, &pool, nullptr, &s.poolSpill[slot][i]);
            if (r != VK_SUCCESS)
            {
                s.poolSpill[slot][i] = VK_NULL_HANDLE;
                SpillFailed("a spill descriptor pool", r);
                return false;
            }
            ++s.stats.poolSpillPools;
        }
        s.poolActive[slot] = i + 1;
        ++s.stats.poolSpills;
        return true;
    }

    /**
     * Creates the ring: the alignment from the device's limits, one buffer per slot, one
     * descriptor pool and one set per slot pointing at that slot's buffer with range = the larger
     * slice class. s.regsLayout must exist.
     *
     * @param context::Handles& h : the live context, for the physical device and its limits
     * @return bool ok : on false, whatever was created is left for the caller's cleanup
     *         (DropRing() and DropLayoutsAndPools())
     */
    bool CreateRing(const context::Handles& h)
    {
        const loader::Device& d = *s.dapi;

        if (!h.iapi || !h.iapi->GetPhysicalDeviceProperties || !h.physical)
        {
            wxl::ScriptMgr::Log(WXL_LOG_WARN,
                                "vulkan: pipelines: no vkGetPhysicalDeviceProperties to read the "
                                "uniform offset alignment from; declining");
            return false;
        }

        VkPhysicalDeviceProperties props = {};
        h.iapi->GetPhysicalDeviceProperties(h.physical, &props);

        // The spec promises an alignment that is a power of two and at most 256, and a range of
        // at least 16384; both are checked anyway, since a slice that does not fit the range is a
        // refusal, not an assumption.
        const VkDeviceSize align = props.limits.minUniformBufferOffsetAlignment;
        if (align == 0 || align > 0x10000 || (align & (align - 1)) != 0
            || kRingRange > props.limits.maxUniformBufferRange)
        {
            wxl::ScriptMgr::Log(WXL_LOG_WARN,
                                "vulkan: pipelines: minUniformBufferOffsetAlignment %llu or "
                                "maxUniformBufferRange %u cannot hold a %u-byte slice; declining",
                                static_cast<unsigned long long>(align),
                                props.limits.maxUniformBufferRange, kRingRange);
            return false;
        }
        s.ringAlign = static_cast<uint32_t>(align);
        s.padded64  = (kSliceBytes64 + s.ringAlign - 1) / s.ringAlign * s.ringAlign;
        s.padded256 = (kSliceBytes256 + s.ringAlign - 1) / s.ringAlign * s.ringAlign;

        // The point-size clamp. D3D clamps a vertex shader's oPts to the
        // POINTSIZE_MIN/MAX render states (81/82), which are not carried; Vulkan clamps
        // gl_PointSize to this range instead, and to 1.0 whole when largePoints is off. Logged
        // once so a snowflake that comes out the wrong size can be read against the limit.
        wxl::ScriptMgr::Log(WXL_LOG_DEBUG,
                            "vulkan: pipelines: point sprites: largePoints %s, pointSizeRange "
                            "[%g, %g] (granularity %g); a point_size vertex module's gl_PointSize "
                            "is clamped to that in place of D3D's POINTSIZE_MIN/MAX",
                            h.largePoints ? "on" : "OFF (every point draws one pixel)",
                            static_cast<double>(props.limits.pointSizeRange[0]),
                            static_cast<double>(props.limits.pointSizeRange[1]),
                            static_cast<double>(props.limits.pointSizeGranularity));

        for (uint32_t i = 0; i < wxlvk::vk::pipelines::kFrameSlots; ++i)
        {
            // Device-local host-visible memory when there is some (ReBAR or the 256 MiB BAR window,
            // memory::Where::Stream), so the GPU reads the ring from VRAM.
            if (!memory::CreateBuffer(kRingBytes, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                                      memory::Where::Stream, s.ring[i]))
            {
                wxl::ScriptMgr::Log(WXL_LOG_WARN,
                                    "vulkan: pipelines: the uniform ring for slot %u (%u KiB) "
                                    "could not be created; declining",
                                    i, kRingBytes / 1024);
                return false;
            }
            if (!s.ring[i].block.mapped)
            {
                wxl::ScriptMgr::Log(WXL_LOG_WARN,
                                    "vulkan: pipelines: the uniform ring for slot %u came back "
                                    "unmapped; declining",
                                    i);
                return false;
            }
            s.ringCursor[i] = 0;
            // Zeroed once, so the high rows a 256-row slice does not copy (above the
            // engine's high-water register) read as D3D's initial zero until something writes them.
            std::memset(s.ring[i].block.mapped, 0, kRingBytes);
        }

        // One tiny pool for the kFrameSlots ring sets, separate from the per-slot pools: those are
        // reset every frame, and these sets live for the session.
        VkDescriptorPoolSize size = {};
        size.type            = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;
        size.descriptorCount = wxlvk::vk::pipelines::kFrameSlots;

        VkDescriptorPoolCreateInfo pool = {};
        pool.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        pool.flags         = 0;
        pool.maxSets       = wxlvk::vk::pipelines::kFrameSlots;
        pool.poolSizeCount = 1;
        pool.pPoolSizes    = &size;

        VkResult r = d.CreateDescriptorPool(s.device, &pool, nullptr, &s.regsPool);
        if (r != VK_SUCCESS)
        {
            s.regsPool = VK_NULL_HANDLE;
            Fail("vkCreateDescriptorPool (ring)", r);
            return false;
        }

        VkDescriptorSetLayout layouts[wxlvk::vk::pipelines::kFrameSlots];
        for (VkDescriptorSetLayout& l : layouts) l = s.regsLayout;

        VkDescriptorSetAllocateInfo alloc = {};
        alloc.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        alloc.descriptorPool     = s.regsPool;
        alloc.descriptorSetCount = wxlvk::vk::pipelines::kFrameSlots;
        alloc.pSetLayouts        = layouts;

        r = d.AllocateDescriptorSets(s.device, &alloc, s.regsSets);
        if (r != VK_SUCCESS)
        {
            for (VkDescriptorSet& set : s.regsSets) set = VK_NULL_HANDLE;
            Fail("vkAllocateDescriptorSets (ring)", r);
            return false;
        }

        // Each set: the whole buffer at offset 0 with range = the larger class; the per-draw
        // dynamic offset selects the slice. A 64-row shader reads the first 1296 bytes of a range
        // that covers 4368, which is allowed -- the range bounds the binding, the block bounds the
        // shader -- and is what lets one descriptor serve both classes.
        VkDescriptorBufferInfo infos[wxlvk::vk::pipelines::kFrameSlots] = {};
        VkWriteDescriptorSet   writes[wxlvk::vk::pipelines::kFrameSlots] = {};
        for (uint32_t i = 0; i < wxlvk::vk::pipelines::kFrameSlots; ++i)
        {
            infos[i].buffer = s.ring[i].handle;
            infos[i].offset = 0;
            infos[i].range  = kRingRange;

            writes[i].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[i].dstSet          = s.regsSets[i];
            writes[i].dstBinding      = 0;
            writes[i].dstArrayElement = 0;
            writes[i].descriptorCount = 1;
            writes[i].descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;
            writes[i].pBufferInfo     = &infos[i];
        }
        d.UpdateDescriptorSets(s.device, wxlvk::vk::pipelines::kFrameSlots, writes, 0, nullptr);

        const uint32_t padded64  = (kSliceBytes64 + s.ringAlign - 1) / s.ringAlign * s.ringAlign;
        const uint32_t padded256 = (kSliceBytes256 + s.ringAlign - 1) / s.ringAlign * s.ringAlign;
        wxl::ScriptMgr::Log(WXL_LOG_DEBUG,
                            "vulkan: pipelines: uniform ring: %u KiB per slot x %u slot(s), "
                            "host-visible and mapped; slices of %u bytes (64-row class, %u padded, "
                            "%u per slot) or %u bytes (256-row class, %u padded, %u per slot) at "
                            "minUniformBufferOffsetAlignment %u; descriptor range %u",
                            kRingBytes / 1024, wxlvk::vk::pipelines::kFrameSlots,
                            kSliceBytes64, padded64, (kRingBytes - kRingRange) / padded64 + 1,
                            kSliceBytes256, padded256, (kRingBytes - kRingRange) / padded256 + 1,
                            s.ringAlign, kRingRange);
        return true;
    }

    /**
     * Creates the bone-palette ring (kBoneRingBytes): set 2's layout, one storage buffer per slot,
     * one set per slot over the whole buffer with range = b[]. Not fatal: on failure everything it
     * made is destroyed and the 256-row draws keep the 256-row modules.
     *
     * @param context::Handles& h : the live context, for the storage offset alignment
     */
    void CreateBones(const context::Handles& h)
    {
        const loader::Device& d = *s.dapi;
        VkPhysicalDeviceProperties props = {};
        h.iapi->GetPhysicalDeviceProperties(h.physical, &props);
        const VkDeviceSize align = props.limits.minStorageBufferOffsetAlignment;
        if (align == 0 || align > 0x10000 || (align & (align - 1)) != 0 || kBoneRange > props.limits.maxStorageBufferRange)
        {
            wxl::ScriptMgr::Log(WXL_LOG_DEBUG, "vulkan: pipelines: storage offset alignment %llu does not suit the "
                                "palette ring; 256-row draws keep the 256-row modules",
                                static_cast<unsigned long long>(align));
            return;
        }
        s.boneAlign = static_cast<uint32_t>(align);

        const auto undo = [&](const char* what, VkResult r) {
            Fail(what, r);
            for (memory::Buffer& b : s.boneRing) memory::DestroyBuffer(b);
            if (s.bonesPool != VK_NULL_HANDLE) d.DestroyDescriptorPool(s.device, s.bonesPool, nullptr);
            if (s.bonesLayout != VK_NULL_HANDLE) d.DestroyDescriptorSetLayout(s.device, s.bonesLayout, nullptr);
            s.bonesPool   = VK_NULL_HANDLE;
            s.bonesLayout = VK_NULL_HANDLE;
            for (VkDescriptorSet& set : s.bonesSets) set = VK_NULL_HANDLE;
        };

        VkDescriptorSetLayoutBinding binding = {};
        binding.binding         = 0;
        binding.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC;
        binding.descriptorCount = 1;
        binding.stageFlags      = VK_SHADER_STAGE_VERTEX_BIT;
        VkDescriptorSetLayoutCreateInfo layout = {};
        layout.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layout.bindingCount = 1;
        layout.pBindings    = &binding;
        VkResult r = d.CreateDescriptorSetLayout(s.device, &layout, nullptr, &s.bonesLayout);
        if (r != VK_SUCCESS)
        {
            s.bonesLayout = VK_NULL_HANDLE;
            undo("vkCreateDescriptorSetLayout (set 2)", r);
            return;
        }

        for (uint32_t i = 0; i < kFrameSlots; ++i)
        {
            if (!memory::CreateBuffer(kBoneRingBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, memory::Where::Stream,
                                      s.boneRing[i])
                || !s.boneRing[i].block.mapped)
            {
                undo("the palette ring buffer", VK_ERROR_OUT_OF_DEVICE_MEMORY);
                return;
            }
            std::memset(s.boneRing[i].block.mapped, 0, kBoneRingBytes);
            s.boneCursor[i] = 0;
        }

        VkDescriptorPoolSize size = {};
        size.type            = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC;
        size.descriptorCount = kFrameSlots;
        VkDescriptorPoolCreateInfo pool = {};
        pool.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        pool.maxSets       = kFrameSlots;
        pool.poolSizeCount = 1;
        pool.pPoolSizes    = &size;
        r = d.CreateDescriptorPool(s.device, &pool, nullptr, &s.bonesPool);
        if (r != VK_SUCCESS)
        {
            s.bonesPool = VK_NULL_HANDLE;
            undo("vkCreateDescriptorPool (set 2)", r);
            return;
        }
        VkDescriptorSetLayout layouts[kFrameSlots];
        for (VkDescriptorSetLayout& l : layouts) l = s.bonesLayout;
        VkDescriptorSetAllocateInfo alloc = {};
        alloc.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        alloc.descriptorPool     = s.bonesPool;
        alloc.descriptorSetCount = kFrameSlots;
        alloc.pSetLayouts        = layouts;
        r = d.AllocateDescriptorSets(s.device, &alloc, s.bonesSets);
        if (r != VK_SUCCESS)
        {
            undo("vkAllocateDescriptorSets (set 2)", r);
            return;
        }
        VkDescriptorBufferInfo infos[kFrameSlots]  = {};
        VkWriteDescriptorSet   writes[kFrameSlots] = {};
        for (uint32_t i = 0; i < kFrameSlots; ++i)
        {
            infos[i].buffer = s.boneRing[i].handle;
            infos[i].offset = 0;
            infos[i].range  = kBoneRange;
            writes[i].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[i].dstSet          = s.bonesSets[i];
            writes[i].dstBinding      = 0;
            writes[i].descriptorCount = 1;
            writes[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC;
            writes[i].pBufferInfo     = &infos[i];
        }
        d.UpdateDescriptorSets(s.device, kFrameSlots, writes, 0, nullptr);
        s.bonesReady = true;
        wxl::ScriptMgr::Log(WXL_LOG_DEBUG,
                            "vulkan: pipelines: bone-palette ring: %u KiB per slot x %u slot(s), mapped; "
                            "palettes of up to %u bytes at minStorageBufferOffsetAlignment %u",
                            kBoneRingBytes / 1024, kFrameSlots, kBoneRange, s.boneAlign);
    }
}
