// Device memory: the memory-type choice, a sub-allocator over a few large blocks, and buffers.
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


#include "vulkan/memory/Internal.hpp"

using namespace wxlvk::vk::memory::detail;

namespace wxlvk::vk::memory
{
    bool Init()
    {
        if (s.ready) return true;

        const context::Handles* h = context::Get();
        if (!h || !h->iapi || !h->dapi)
        {
            wxl::ScriptMgr::Log(WXL_LOG_WARN, "vulkan: memory: Init() without a device; declining");
            return false;
        }
        if (!HaveEntryPoints(*h->iapi, *h->dapi)) return false;

        s.device = h->device;
        s.dapi   = h->dapi;

        h->iapi->GetPhysicalDeviceMemoryProperties(h->physical, &s.props);

        VkPhysicalDeviceProperties props = {};
        h->iapi->GetPhysicalDeviceProperties(h->physical, &props);
        // Both are powers of two by spec; the floor of 1 only protects RoundUp() from a zero a
        // broken driver might report, which would turn every mask into "align to everything".
        s.granularity    = props.limits.bufferImageGranularity ? props.limits.bufferImageGranularity : 1;
        s.atom           = props.limits.nonCoherentAtomSize ? props.limits.nonCoherentAtomSize : 1;
        s.maxAllocations = props.limits.maxMemoryAllocationCount;

        LogTable();

        // Resolved against "any resource" here, for the log and for the decline. Each Allocate()
        // resolves again against its own memoryTypeBits, because a resource is allowed to exclude
        // the type that suits most others, and the answer for it may then differ.
        bool ok = true;
        const Where wheres[] = { Where::DeviceLocal, Where::HostVisible, Where::Stream };
        for (Where w : wheres)
        {
            const char*    why  = "";
            const uint32_t type = Resolve(0xFFFFFFFFu, w, why);
            if (type == kNoType)
            {
                wxl::ScriptMgr::Log(WXL_LOG_WARN, "vulkan: memory: %s -> no usable memory type; declining",
                                    WhereName(w));
                ok = false;
                continue;
            }

            char text[128];
            const VkMemoryType& t = s.props.memoryTypes[type];
            const bool fallback   = std::strncmp(why, "FALLBACK", 8) == 0;
            wxl::ScriptMgr::Log(fallback ? WXL_LOG_WARN : WXL_LOG_DEBUG,
                                "vulkan: memory: %s -> type %u (heap %u, %llu MiB, %s): %s", WhereName(w), type,
                                t.heapIndex, MiB(s.props.memoryHeaps[t.heapIndex].size),
                                PropertyText(t.propertyFlags, text, sizeof text), why);
        }

        if (!ok)
        {
            s.device = nullptr;
            s.dapi   = nullptr;
            return false;
        }

        s.ready = true;
        return true;
    }

    bool Allocate(const VkMemoryRequirements& req, Where where, Block& out)
    {
        if (!s.ready)
        {
            wxl::ScriptMgr::Log(WXL_LOG_WARN, "vulkan: memory: Allocate() before Init()");
            return false;
        }
        if (req.size == 0)
        {
            wxl::ScriptMgr::Log(WXL_LOG_WARN, "vulkan: memory: Allocate() of zero bytes");
            return false;
        }

        const char*    why  = "";
        const uint32_t type = Resolve(req.memoryTypeBits, where, why);
        if (type == kNoType)
        {
            wxl::ScriptMgr::Log(WXL_LOG_WARN, "vulkan: memory: no %s type in memoryTypeBits 0x%08X",
                                WhereName(where), req.memoryTypeBits);
            return false;
        }

        // Alignment. req.alignment is the offset the resource must be bound at, and it is not a
        // suggestion: bind a buffer at a misaligned offset and vkBindBufferMemory is invalid usage
        // that most drivers accept silently, after which the GPU reads from the rounded-down
        // address -- a texture shifted by a few texels, a vertex buffer whose first vertices are
        // the previous resource's tail. Nothing errors; the picture is just wrong.
        //
        // bufferImageGranularity. Allocate() is not told whether the caller is binding a buffer or
        // an image, so the alternative of separate block lists per resource kind is not available
        // through this API. Instead no two sub-allocations may ever share a granularity page:
        // every start is aligned to the granularity and every size padded to it. Then a linear
        // buffer and an optimal-tiling image can never be neighbours within one page, whichever
        // the caller made. On NVIDIA the granularity is 1 and this costs nothing; where it is
        // larger (1 KiB to 64 KiB on some adapters) it costs at most one page of slack per
        // allocation, logged at Init() so the waste is explainable.
        //
        // nonCoherentAtomSize is not folded in: FlushHost() rounds its own range to the atom and
        // clamps to the block, so it needs no help from placement.
        const VkDeviceSize align  = req.alignment > s.granularity ? req.alignment : s.granularity;
        const VkDeviceSize padded = RoundUp(req.size, s.granularity);

        uint32_t     slot   = kMaxSlots;
        VkDeviceSize offset = 0;

        if (padded > BlockSize(where))
        {
            // Bigger than a block: its own VkDeviceMemory, sized to the request rather than rounded
            // to a block multiple, because the slack of a dedicated block is never reused.
            slot = NewSlot(type, where, padded, true);
            if (slot == kMaxSlots) return false;
            if (!Carve(s.slots[slot], padded, align, offset))
            {
                DropSlot(*s.dapi, slot);
                return false;
            }
        }
        else
        {
            for (uint32_t i = 0; i < kMaxSlots; ++i)
            {
                Slot& candidate = s.slots[i];
                if (candidate.memory == VK_NULL_HANDLE || candidate.dedicated) continue;
                if (candidate.where != where || candidate.typeIndex != type) continue;
                if (candidate.capacity - candidate.inUse < padded) continue;
                if (Carve(candidate, padded, align, offset))
                {
                    slot = i;
                    break;
                }
            }

            if (slot == kMaxSlots)
            {
                slot = NewSlot(type, where, BlockSize(where), false);
                if (slot == kMaxSlots) return false;
                if (!Carve(s.slots[slot], padded, align, offset))
                {
                    // Unreachable for a fresh block -- padded <= capacity and offset 0 is aligned to
                    // anything -- but cheap to keep honest.
                    DropSlot(*s.dapi, slot);
                    return false;
                }
            }
        }

        const Slot& owner = s.slots[slot];
        out.memory = owner.memory;
        out.offset = offset;
        out.size   = req.size;
        out.mapped = owner.base ? static_cast<char*>(owner.base) + static_cast<size_t>(offset) : nullptr;
        out.owner  = slot;
        out.index  = s.generation[slot];
        return true;
    }

    void Free(const Block& block)
    {
        if (block.memory == VK_NULL_HANDLE || block.size == 0) return;
        if (!s.ready || block.owner >= kMaxSlots) return;

        Slot& slot = s.slots[block.owner];
        if (slot.memory != block.memory || s.generation[block.owner] != block.index)
        {
            // A block from a dedicated slot since released and reused.
            // Returning its span would hand someone else's live bytes to the next Allocate().
            wxl::ScriptMgr::Log(WXL_LOG_WARN, "vulkan: memory: Free() of a stale block ignored");
            return;
        }

        // The same padding Allocate() applied; see the bufferImageGranularity comment there.
        Release(slot, block.offset, RoundUp(block.size, s.granularity));

        // A dedicated block is released as soon as it empties: its size was the one resource's, so
        // it is unlikely to be reused, and holding it -- mapped, if host-visible -- would keep its
        // address space for nothing. An ordinary block that empties waits for Tick().
        if (slot.dedicated && slot.inUse == 0)
        {
            DropSlot(*s.dapi, block.owner);
            ++s.dedicatedFreed;
        }
    }

    void Tick()
    {
        if (!s.ready) return;
        ++s.tick;
        // A block that has stayed empty for kReturnAfterTicks presents goes back to the driver,
        // except the kSpareBlocks lowest-indexed empty ones per (type, class): the reserved figure
        // follows what is in use, a host-visible block's address space comes back, and a class
        // that refills at once still finds a block. A dedicated block never waits here (Free()
        // releases it the moment it empties).
        for (uint32_t i = 0; i < kMaxSlots; ++i)
        {
            const Slot& slot = s.slots[i];
            if (slot.memory == VK_NULL_HANDLE || slot.dedicated || slot.inUse != 0) continue;
            if (s.tick - slot.emptySince < kReturnAfterTicks) continue;

            uint32_t sparesBefore = 0;
            for (uint32_t k = 0; k < i; ++k)
            {
                const Slot& o = s.slots[k];
                if (o.memory != VK_NULL_HANDLE && !o.dedicated && o.inUse == 0 && o.typeIndex == slot.typeIndex
                    && o.where == slot.where)
                    ++sparesBefore;
            }
            if (sparesBefore < kSpareBlocks) continue;

            const VkDeviceSize bytes = slot.capacity;
            DropSlot(*s.dapi, i);
            ++s.returnedBlocks;
            s.returnedBytes += bytes;
        }
    }

    bool CreateBuffer(VkDeviceSize size, VkBufferUsageFlags usage, Where where, Buffer& out)
    {
        if (!s.ready)
        {
            wxl::ScriptMgr::Log(WXL_LOG_WARN, "vulkan: memory: CreateBuffer() before Init()");
            return false;
        }
        if (size == 0)
        {
            wxl::ScriptMgr::Log(WXL_LOG_WARN, "vulkan: memory: CreateBuffer() of zero bytes");
            return false;
        }

        const loader::Device& d = *s.dapi;

        VkBufferCreateInfo info = {};
        info.sType       = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        info.size        = size;
        info.usage       = usage;
        // EXCLUSIVE: every user is on the graphics family. A separate transfer queue would need
        // either CONCURRENT here or an ownership transfer, and neither exists yet to need it.
        info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        VkBuffer handle = VK_NULL_HANDLE;
        VkResult r = d.CreateBuffer(s.device, &info, nullptr, &handle);
        if (r != VK_SUCCESS) return Fail("vkCreateBuffer", r);

        VkMemoryRequirements req = {};
        d.GetBufferMemoryRequirements(s.device, handle, &req);

        Block block;
        if (!Allocate(req, where, block))
        {
            d.DestroyBuffer(s.device, handle, nullptr);
            wxl::ScriptMgr::Log(WXL_LOG_WARN, "vulkan: memory: no memory for a %llu-byte %s buffer",
                                static_cast<unsigned long long>(size), WhereName(where));
            return false;
        }

        r = d.BindBufferMemory(s.device, handle, block.memory, block.offset);
        if (r != VK_SUCCESS)
        {
            Free(block);
            d.DestroyBuffer(s.device, handle, nullptr);
            return Fail("vkBindBufferMemory", r);
        }

        out.handle = handle;
        out.block  = block;
        out.size   = size;
        return true;
    }

    void DestroyBuffer(Buffer& buf)
    {
        if (buf.handle == VK_NULL_HANDLE && buf.block.memory == VK_NULL_HANDLE) return;

        if (buf.handle != VK_NULL_HANDLE)
        {
            // Through the context, and only while vulkan-1.dll is still mapped: a buffer destroyed
            // from a static destructor at process exit must not call into an unloaded loader.
            const context::Handles* h = context::Get();
            if (h && h->dapi && h->dapi->DestroyBuffer && loader::StillMapped())
                h->dapi->DestroyBuffer(h->device, buf.handle, nullptr);
        }

        Free(buf.block);
        buf = Buffer();
    }

    bool Staging(VkDeviceSize size, Buffer*& out)
    {
        if (s.staging.handle != VK_NULL_HANDLE && s.staging.size >= size)
        {
            out = &s.staging;
            return true;
        }
        if (!s.ready)
        {
            wxl::ScriptMgr::Log(WXL_LOG_WARN, "vulkan: memory: Staging() before Init()");
            return false;
        }

        // Grow geometrically while that stays modest, so a sequence of slightly larger uploads does
        // not regrow on every call; past kStagingLarge, grow to exactly what is asked, because
        // doubling a mapped buffer there would double an address-space cost nobody needed.
        VkDeviceSize want = RoundUp(size ? size : 1, kMiB);
        if (want < kStagingFloor) want = kStagingFloor;
        const VkDeviceSize doubled = s.staging.size * 2;
        if (doubled > want && doubled <= kStagingLarge) want = doubled;

        Buffer fresh;
        if (!CreateBuffer(want, VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                          Where::HostVisible, fresh))
        {
            // The old buffer, if any, is left as it was: a failed growth does not take away the
            // capacity a smaller upload could still use.
            wxl::ScriptMgr::Log(WXL_LOG_WARN, "vulkan: memory: staging growth to %llu MiB failed", MiB(want));
            return false;
        }

        const VkDeviceSize was = s.staging.size;
        DestroyBuffer(s.staging);
        s.staging = fresh;

        wxl::ScriptMgr::Log(want > kStagingLarge ? WXL_LOG_WARN : WXL_LOG_DEBUG,
                            "vulkan: memory: staging buffer %llu -> %llu MiB (persistently mapped: that much "
                            "address space for the session)",
                            MiB(was), MiB(want));
        out = &s.staging;
        return true;
    }

    bool FlushHost(const Block& block)
    {
        if (block.memory == VK_NULL_HANDLE || !s.ready || block.owner >= kMaxSlots) return true;

        const Slot& slot = s.slots[block.owner];
        if (slot.memory != block.memory || !slot.base) return true;
        if (slot.flags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) return true;

        // The range rules for a non-coherent flush: offset a multiple of nonCoherentAtomSize, and
        // size either a multiple of it or reaching exactly the end of the allocation. Round out to
        // the atom, and use VK_WHOLE_SIZE when rounding would pass the end -- a block's capacity
        // need not itself be a multiple of the atom.
        const VkDeviceSize start = RoundDown(block.offset, s.atom);
        const VkDeviceSize end   = RoundUp(block.offset + block.size, s.atom);

        VkMappedMemoryRange range = {};
        range.sType  = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
        range.memory = slot.memory;
        range.offset = start;
        range.size   = end >= slot.capacity ? VK_WHOLE_SIZE : end - start;

        const VkResult r = s.dapi->FlushMappedMemoryRanges(s.device, 1, &range);
        if (r != VK_SUCCESS) return Fail("vkFlushMappedMemoryRanges", r);
        return true;
    }

    void Stats(uint64_t& vkAllocateMemoryCalls, uint64_t& bytesReserved, uint64_t& bytesInUse)
    {
        vkAllocateMemoryCalls = s.allocateCalls;
        bytesReserved         = s.reserved;
        bytesInUse            = s.inUse;
    }

    void GetDetail(Detail& out)
    {
        out = Detail();
        out.reservedPeak   = s.reservedPeak;
        out.mapped         = s.mapped;
        out.mappedPeak     = s.mappedPeak;
        out.returnedBlocks = s.returnedBlocks;
        out.returnedBytes  = s.returnedBytes;
        out.dedicatedFreed = s.dedicatedFreed;
        out.spanGrowths    = s.spanGrowths;
        out.spanPeak       = s.spanPeak;
        for (const Slot& slot : s.slots)
        {
            if (slot.memory == VK_NULL_HANDLE) continue;
            ++out.blocks;
            if (slot.dedicated) ++out.dedicatedBlocks;
            else if (slot.inUse == 0) ++out.emptyBlocks;
            if (slot.where == Where::DeviceLocal) out.deviceLocal += slot.capacity;
            if (slot.typeIndex < s.props.memoryTypeCount)
            {
                const uint32_t heap = s.props.memoryTypes[slot.typeIndex].heapIndex;
                if (heap < VK_MAX_MEMORY_HEAPS) out.heapReserved[heap] += slot.capacity;
            }
        }
    }
}
