// The memory budget (wxl.vk.render GetMemoryBudget, MayKeepResident) and its log line.
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

#include "native/render/Budget.hpp"

#include "native/Present.hpp"
#include "vulkan/Context.hpp"
#include "vulkan/Footprint.hpp"
#include "vulkan/Memory.hpp"

#include "wxl/Script.hpp"

#include <windows.h>

#include <cstdio>
#include <cstring>

namespace
{
    namespace context   = wxlvk::vk::context;
    namespace memory    = wxlvk::vk::memory;
    namespace footprint = wxlvk::vk::footprint;

    constexpr uint64_t kMiB = 1024ull * 1024ull;

    /// MayKeepResident's floor under the device-local budget, beside WXL_VK_RESIDENT_MARGIN of it: what the engine's
    /// own textures of the next zone and a module's resize need without an eviction.
    constexpr uint64_t kResidentFloor = 256ull * kMiB;
    /// The address space a mapped request must leave free, and the slack its largest free run must have over it.
    constexpr uint64_t kAddressFloor = 512ull * kMiB;
    constexpr uint64_t kAddressSlack = 64ull * kMiB;
    /// The budget's line once in a session, this many presents in (the first zone's textures loaded).
    constexpr uint64_t kFirstLinePresents = 600;

    struct State
    {
        bool               taken = false;
        WXL_VkMemoryBudget b     = {};
        bool               saidFirst = false;
    };
    State g;

    /// The heap's budget and usage from the driver, or the estimate (80 % of the heap; the allocator's bytes).
    void FillHeap(WXL_VkMemoryHeapBudget& h, const VkPhysicalDeviceMemoryProperties& props, uint32_t heap, bool driver,
                  const VkDeviceSize* budget, const VkDeviceSize* usage, const memory::Detail& md)
    {
        h = WXL_VkMemoryHeapBudget();
        if (heap >= props.memoryHeapCount) return;
        h.size     = props.memoryHeaps[heap].size;
        h.reserved = md.heapReserved[heap];
        if (driver && budget[heap])
        {
            h.budget = budget[heap];
            h.usage  = usage[heap];
        }
        else
        {
            h.budget = h.size / 5 * 4;
            h.usage  = h.reserved;
        }
    }

    void Refresh(bool walk)
    {
        const context::Handles* hd = context::Get();
        if (!hd) return;
        VkPhysicalDeviceMemoryProperties props = {};
        VkDeviceSize budget[VK_MAX_MEMORY_HEAPS] = {}, usage[VK_MAX_MEMORY_HEAPS] = {};
        const bool driver = context::MemoryHeaps(props, budget, usage);
        memory::Detail md;
        memory::GetDetail(md);
        uint64_t allocs = 0, reserved = 0, inUse = 0;
        memory::Stats(allocs, reserved, inUse);

        // The heaps: the largest DEVICE_LOCAL one (the card), the largest other one (system memory), and a DEVICE_LOCAL
        // heap of its own holding the host-visible device-local types (the BAR window without Resizable BAR).
        uint32_t card = UINT32_MAX, sys = UINT32_MAX, bar = UINT32_MAX;
        for (uint32_t i = 0; i < props.memoryHeapCount; ++i)
        {
            const bool local = (props.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) != 0;
            uint32_t& pick = local ? card : sys;
            if (pick == UINT32_MAX || props.memoryHeaps[i].size > props.memoryHeaps[pick].size) pick = i;
        }
        for (uint32_t t = 0; t < props.memoryTypeCount; ++t)
        {
            const VkMemoryPropertyFlags f = props.memoryTypes[t].propertyFlags;
            const uint32_t heap = props.memoryTypes[t].heapIndex;
            if ((f & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) && (f & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) && heap != card) bar = heap;
        }

        WXL_VkMemoryBudget b = {};
        b.structSize = sizeof b;
        b.flags      = driver ? WXL_VK_BUDGET_DRIVER : 0u;
        b.serial     = wxlvk::native::present::Serial();
        FillHeap(b.deviceLocal, props, card, driver, budget, usage, md);
        FillHeap(b.hostVisible, props, sys, driver, budget, usage, md);
        FillHeap(b.bar, props, bar, driver, budget, usage, md);
        b.allocatorReserved = reserved;
        b.allocatorInUse    = inUse;
        b.allocatorPeak     = md.reservedPeak;
        b.mapped            = md.mapped;

        MEMORYSTATUSEX ms = {};
        ms.dwLength = sizeof ms;
        if (GlobalMemoryStatusEx(&ms))
        {
            b.addressFree  = ms.ullAvailVirtual;
            b.addressTotal = ms.ullTotalVirtual;
        }
        // The walk is the costly part: its figure is kept between walks.
        b.addressLargestFree = walk || !g.taken ? footprint::Take().largestFree : g.b.addressLargestFree;

        const uint64_t margin = static_cast<uint64_t>(static_cast<double>(b.deviceLocal.budget) * WXL_VK_RESIDENT_MARGIN);
        const uint64_t keep   = margin > kResidentFloor ? margin : kResidentFloor;
        b.residentHeadroom    = b.deviceLocal.budget > b.deviceLocal.usage + keep ? b.deviceLocal.budget - b.deviceLocal.usage - keep : 0;
        g.b     = b;
        g.taken = true;
    }

    unsigned long long Mib(uint64_t v) { return static_cast<unsigned long long>(v / kMiB); }
}

namespace wxlvk::native::render::budget
{
    void OnPresent(uint64_t presents)
    {
        if (!context::Get()) return;
        if (!g.taken || presents % kRefreshPresents == 0) Refresh(presents % kWalkPresents == 0);
        if (!g.saidFirst && presents >= kFirstLinePresents)
        {
            g.saidFirst = true;
            Log("in session");
        }
    }

    bool Get(WXL_VkMemoryBudget& out)
    {
        if (!g.taken) Refresh(true);
        if (!g.taken) return false;
        const size_t n = out.structSize < sizeof g.b ? out.structSize : sizeof g.b;
        std::memcpy(&out, &g.b, n);
        out.structSize = static_cast<uint32_t>(n);
        return true;
    }

    bool MayKeep(uint64_t extraBytes, uint32_t memoryClass)
    {
        if (!g.taken) Refresh(true);
        if (!g.taken) return false;
        const WXL_VkMemoryBudget& b = g.b;
        if (memoryClass == WXL_VK_MEMORY_DEVICE_LOCAL) return extraBytes <= b.residentHeadroom;
        // The mapped classes: the address space first (a mapping costs its size there), then their heap (system memory,
        // or the BAR window's for the stream class while it has one).
        if (b.addressFree < extraBytes + kAddressFloor || b.addressLargestFree < extraBytes + kAddressSlack) return false;
        const WXL_VkMemoryHeapBudget& h = memoryClass == WXL_VK_MEMORY_STREAM && b.bar.size ? b.bar : b.hostVisible;
        return h.usage + extraBytes <= h.budget;
    }

    void Log(const char* when)
    {
        Refresh(true);   // a line is rare: its figures fresh
        if (!g.taken) return;
        const WXL_VkMemoryBudget& b = g.b;
        char bar[160] = "";
        if (b.bar.size)
            std::snprintf(bar, sizeof bar, "; the BAR window %llu MiB, budget %llu, used %llu (the device's rings %llu)",
                          Mib(b.bar.size), Mib(b.bar.budget), Mib(b.bar.usage), Mib(b.bar.reserved));
        wxl::ScriptMgr::Log(WXL_LOG_INFO,
                            "wxl-vulkan-api: memory budget (%s, %s): video memory %llu MiB, budget %llu, used %llu (the device's "
                            "allocator %llu of it), room to keep %llu more resident; system memory %llu MiB, budget %llu, used "
                            "%llu%s; the allocator %llu MiB reserved (peak %llu), %llu in use, %llu mapped; address space %llu of "
                            "%llu MiB free, largest run %llu",
                            when, (b.flags & WXL_VK_BUDGET_DRIVER) ? "the driver's figures" : "estimated: no VK_EXT_memory_budget",
                            Mib(b.deviceLocal.size), Mib(b.deviceLocal.budget), Mib(b.deviceLocal.usage), Mib(b.deviceLocal.reserved),
                            Mib(b.residentHeadroom), Mib(b.hostVisible.size), Mib(b.hostVisible.budget), Mib(b.hostVisible.usage),
                            bar, Mib(b.allocatorReserved), Mib(b.allocatorPeak), Mib(b.allocatorInUse), Mib(b.mapped),
                            Mib(b.addressFree), Mib(b.addressTotal), Mib(b.addressLargestFree));
    }
}
