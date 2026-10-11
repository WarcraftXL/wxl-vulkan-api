// The allocator's blocks and their spans: carve, release, grow, drop.
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

namespace wxlvk::vk::memory::detail
{
    /**
     * Doubles a slot's span list (kFirstSpans for a slot that has none), keeping its entries.
     *
     * @param Slot& slot : the block
     * @return bool ok : false when the heap refused
     */
    bool GrowSpans(Slot& slot)
    {
        const uint32_t cap   = slot.spanCap ? slot.spanCap * 2 : kFirstSpans;
        Span*          fresh = new (std::nothrow) Span[cap];
        if (!fresh) return false;
        for (uint32_t i = 0; i < slot.spanCount; ++i) fresh[i] = slot.spans[i];
        delete[] slot.spans;
        slot.spans   = fresh;
        slot.spanCap = cap;
        if (cap > kFirstSpans) ++s.spanGrowths;
        return true;
    }

    /**
     * Takes `size` bytes at `align` out of a slot's free spans, first fit.
     *
     * The leftovers on both sides of the placed range go back into the list, so the alignment slack
     * in front of an allocation is not lost -- which is what lets Free() reconstruct the span it
     * returns from nothing but the caller's offset and size.
     *
     * @param Slot& slot : the block
     * @param VkDeviceSize size : padded size
     * @param VkDeviceSize align : effective alignment, a power of two
     * @param VkDeviceSize& offset : receives the placed offset
     * @return bool ok : false when no span fits (or the list could not grow for a split)
     */
    bool Carve(Slot& slot, VkDeviceSize size, VkDeviceSize align, VkDeviceSize& offset)
    {
        for (uint32_t i = 0; i < slot.spanCount; ++i)
        {
            const Span   sp    = slot.spans[i];
            const VkDeviceSize start = RoundUp(sp.offset, align);
            const VkDeviceSize end   = sp.offset + sp.size;
            if (start < sp.offset || start + size > end) continue;

            const VkDeviceSize front = start - sp.offset;
            const VkDeviceSize back  = end - (start + size);

            if (front && back)
            {
                if (slot.spanCount == slot.spanCap && !GrowSpans(slot)) continue;
                for (uint32_t k = slot.spanCount; k > i + 1; --k) slot.spans[k] = slot.spans[k - 1];
                slot.spans[i]     = Span{ sp.offset, front };
                slot.spans[i + 1] = Span{ start + size, back };
                ++slot.spanCount;
                if (slot.spanCount > s.spanPeak) s.spanPeak = slot.spanCount;
            }
            else if (front)
                slot.spans[i] = Span{ sp.offset, front };
            else if (back)
                slot.spans[i] = Span{ start + size, back };
            else
            {
                for (uint32_t k = i; k + 1 < slot.spanCount; ++k) slot.spans[k] = slot.spans[k + 1];
                --slot.spanCount;
            }

            offset = start;
            slot.inUse += size;
            s.inUse    += size;
            return true;
        }
        return false;
    }

    /**
     * Returns [offset, offset + size) to a slot's free list, merging with either neighbour.
     *
     * @param Slot& slot : the block
     * @param VkDeviceSize offset : start of the span
     * @param VkDeviceSize size : padded size
     */
    void Release(Slot& slot, VkDeviceSize offset, VkDeviceSize size)
    {
        slot.inUse -= size;
        s.inUse    -= size;

        uint32_t p = 0;
        while (p < slot.spanCount && slot.spans[p].offset < offset) ++p;

        const bool joinPrev = p > 0 && slot.spans[p - 1].offset + slot.spans[p - 1].size == offset;
        const bool joinNext = p < slot.spanCount && offset + size == slot.spans[p].offset;

        if (joinPrev && joinNext)
        {
            slot.spans[p - 1].size += size + slot.spans[p].size;
            for (uint32_t k = p; k + 1 < slot.spanCount; ++k) slot.spans[k] = slot.spans[k + 1];
            --slot.spanCount;
        }
        else if (joinPrev)
            slot.spans[p - 1].size += size;
        else if (joinNext)
        {
            slot.spans[p].offset = offset;
            slot.spans[p].size  += size;
        }
        else
        {
            if (slot.spanCount == slot.spanCap && !GrowSpans(slot))
            {
                // Only when the heap refuses a few KiB: the span stays unusable until the block is
                // released rather than being handed out twice. Logged once.
                if (!s.spanGrowWarned)
                {
                    s.spanGrowWarned = true;
                    wxl::ScriptMgr::Log(WXL_LOG_WARN,
                                        "vulkan: memory: a block's free list could not grow past %u "
                                        "spans; %llu byte(s) unusable until the block is released "
                                        "(logged once)",
                                        slot.spanCap, static_cast<unsigned long long>(size));
                }
                return;
            }
            for (uint32_t k = slot.spanCount; k > p; --k) slot.spans[k] = slot.spans[k - 1];
            slot.spans[p] = Span{ offset, size };
            ++slot.spanCount;
            if (slot.spanCount > s.spanPeak) s.spanPeak = slot.spanCount;
        }
        if (slot.inUse == 0) slot.emptySince = s.tick;
    }

    /**
     * Gives a slot's memory back to the driver and empties the slot, bumping its generation.
     *
     * @param loader::Device& d : the device table
     * @param uint32 i : the slot
     */
    void DropSlot(const loader::Device& d, uint32_t i)
    {
        Slot& slot = s.slots[i];
        if (slot.memory != VK_NULL_HANDLE)
        {
            if (slot.base)
            {
                d.UnmapMemory(s.device, slot.memory);
                s.mapped -= slot.capacity;
            }
            d.FreeMemory(s.device, slot.memory, nullptr);
            s.reserved -= slot.capacity;
            s.inUse    -= slot.inUse;
        }
        delete[] slot.spans;
        slot = Slot();
        ++s.generation[i];
    }

    /**
     * Allocates a new block into a free slot and, for the host-visible class, maps it.
     *
     * Mapped once, here, for the block's whole life, and never per allocation. Per-allocation
     * map/unmap would be slower (each is a kernel transition on most drivers) and is not even
     * legal: a VkDeviceMemory may have only one active mapping at a time, so two live
     * sub-allocations in one block could not both be mapped. The price is the one this file keeps
     * coming back to -- the whole block's size in address space from now for the session.
     *
     * A DeviceLocal block whose type happens to be host-visible (an integrated adapter, or the
     * fallback) is deliberately not mapped: nobody asked to write it from the CPU, and mapping it
     * would spend address space for no reader.
     *
     * @param uint32 type : the memory type
     * @param Where where : the class
     * @param VkDeviceSize capacity : bytes
     * @param bool dedicated : a one-resource block, released when it empties
     * @return uint32 slot : the index, or kMaxSlots on failure (logged)
     */
    uint32_t NewSlot(uint32_t type, Where where, VkDeviceSize capacity, bool dedicated)
    {
        uint32_t i = 0;
        while (i < kMaxSlots && s.slots[i].memory != VK_NULL_HANDLE) ++i;
        if (i == kMaxSlots)
        {
            wxl::ScriptMgr::Log(WXL_LOG_WARN, "vulkan: memory: all %u block slots in use (%llu MiB reserved)",
                                kMaxSlots, MiB(s.reserved));
            return kMaxSlots;
        }

        const loader::Device& d = *s.dapi;

        VkMemoryAllocateInfo info = {};
        info.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        info.allocationSize  = capacity;
        info.memoryTypeIndex = type;

        VkDeviceMemory memory = VK_NULL_HANDLE;
        VkResult r = d.AllocateMemory(s.device, &info, nullptr, &memory);
        if (r != VK_SUCCESS)
        {
            wxl::ScriptMgr::Log(WXL_LOG_WARN,
                                "vulkan: memory: vkAllocateMemory of %llu MiB (type %u, %s) failed: %s; "
                                "%llu MiB already reserved",
                                MiB(capacity), type, WhereName(where), ResultName(r), MiB(s.reserved));
            return kMaxSlots;
        }
        ++s.allocateCalls;

        void* base = nullptr;
        if (where != Where::DeviceLocal)
        {
            r = d.MapMemory(s.device, memory, 0, VK_WHOLE_SIZE, 0, &base);
            if (r != VK_SUCCESS || !base)
            {
                d.FreeMemory(s.device, memory, nullptr);
                // VK_ERROR_MEMORY_MAP_FAILED here on this process usually means address space, not
                // device memory: there was no contiguous run of `capacity` bytes left to map into.
                wxl::ScriptMgr::Log(WXL_LOG_WARN,
                                    "vulkan: memory: vkMapMemory of a %llu MiB block failed: %s "
                                    "(likely no contiguous address space left)",
                                    MiB(capacity), ResultName(r));
                return kMaxSlots;
            }
        }

        Slot& slot = s.slots[i];
        if (!slot.spans && !GrowSpans(slot))
        {
            if (base) d.UnmapMemory(s.device, memory);
            d.FreeMemory(s.device, memory, nullptr);
            wxl::ScriptMgr::Log(WXL_LOG_WARN, "vulkan: memory: no heap for a block's free list");
            return kMaxSlots;
        }
        slot.memory    = memory;
        slot.base      = base;
        slot.capacity  = capacity;
        slot.inUse     = 0;
        slot.flags     = s.props.memoryTypes[type].propertyFlags;
        slot.typeIndex = type;
        slot.where     = where;
        slot.dedicated = dedicated;
        slot.spanCount  = 1;
        slot.spans[0]   = Span{ 0, capacity };
        slot.emptySince = s.tick;

        s.reserved += capacity;
        if (base) s.mapped += capacity;
        if (s.reserved > s.reservedPeak) s.reservedPeak = s.reserved;
        if (s.mapped > s.mappedPeak) s.mappedPeak = s.mapped;

        uint32_t live = 0;
        for (const Slot& other : s.slots) live += other.memory != VK_NULL_HANDLE ? 1u : 0u;

        wxl::ScriptMgr::Log(WXL_LOG_DEBUG,
                            "vulkan: memory: new %s%s block: %llu MiB, type %u%s; now %u block(s), "
                            "%llu MiB reserved, %llu of %u allocation(s) used",
                            dedicated ? "dedicated " : "", WhereName(where), MiB(RoundUp(capacity, kMiB)), type,
                            base ? ", mapped (costs that much address space)" : "", live, MiB(s.reserved),
                            static_cast<unsigned long long>(s.allocateCalls), s.maxAllocations);
        return i;
    }
}
