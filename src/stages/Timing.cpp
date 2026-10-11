// GPU time of each point's stage run and of each stage, from timestamp queries.
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

#include "stages/Timing.hpp"

#include "stages/Gpu.hpp"
#include "vulkan/Context.hpp"
#include "vulkan/Pipelines.hpp"

#include "wxl/Script.hpp"

namespace
{
    namespace context = wxlvk::vk::context;
    using wxlvk::stages::timing::kLanes;
    using wxlvk::stages::timing::Lane;

    constexpr uint32_t kSlots = wxlvk::vk::pipelines::kFrameSlots;

    struct SlotQueries
    {
        VkQueryPool pool    = VK_NULL_HANDLE;
        uint64_t    serial  = 0;   ///< the frame whose run last reset the pool
        uint32_t    written = 0;   ///< lanes (bits) both of whose timestamps were recorded
    };

    struct State
    {
        bool        tried     = false;
        bool        available = false;
        double      nsPerTick = 1.0;
        uint64_t    mask      = ~0ull;   ///< the valid bits of a timestamp
        SlotQueries slots[kSlots];
        uint32_t    current   = 0;       ///< the slot of the run being recorded
        Lane        lanes[kLanes];
    };
    State g;

    /// The queue's timestamp bits and the tick length, and one pool per slot.
    bool Init()
    {
        g.tried = true;
        const context::Handles* h = context::Get();
        if (!h || !h->iapi || !h->iapi->GetPhysicalDeviceProperties || !h->iapi->GetPhysicalDeviceQueueFamilyProperties)
            return false;
        VkPhysicalDeviceProperties props = {};
        h->iapi->GetPhysicalDeviceProperties(h->physical, &props);
        uint32_t count = 0;
        h->iapi->GetPhysicalDeviceQueueFamilyProperties(h->physical, &count, nullptr);
        VkQueueFamilyProperties families[16] = {};
        if (count > 16) count = 16;
        h->iapi->GetPhysicalDeviceQueueFamilyProperties(h->physical, &count, families);
        const uint32_t bits = h->graphicsFamily < count ? families[h->graphicsFamily].timestampValidBits : 0;
        if (bits == 0 || props.limits.timestampPeriod <= 0.0f)
        {
            wxl::ScriptMgr::Log(WXL_LOG_DEBUG, "stages: the graphics queue has no timestamps; GPU times unknown");
            return false;
        }
        g.mask      = bits >= 64 ? ~0ull : ((1ull << bits) - 1);
        g.nsPerTick = props.limits.timestampPeriod;
        VkQueryPoolCreateInfo info = {};
        info.sType      = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
        info.queryType  = VK_QUERY_TYPE_TIMESTAMP;
        info.queryCount = 2 * kLanes;
        for (SlotQueries& s : g.slots)
            if (h->dapi->CreateQueryPool(h->device, &info, nullptr, &s.pool) != VK_SUCCESS) return false;
        return true;
    }

    void Note(Lane& lane, double ms)
    {
        lane.lastMs = ms;
        lane.sumMs += ms;
        if (ms > lane.maxMs) lane.maxMs = ms;
        ++lane.samples;
    }

    /// Reads the lanes the slot's previous run wrote, without waiting.
    void Harvest(SlotQueries& s)
    {
        const wxlvk::vk::loader::Device& d = *wxlvk::stages::gpu::Api();
        for (uint32_t lane = 0; lane < kLanes; ++lane)
        {
            if (!(s.written & (1u << lane))) continue;
            uint64_t t[2] = {};
            const VkResult r = d.GetQueryPoolResults(wxlvk::stages::gpu::Device(), s.pool, 2 * lane, 2, sizeof t, t,
                                                     sizeof t[0], VK_QUERY_RESULT_64_BIT);
            if (r != VK_SUCCESS) continue;
            const uint64_t ticks = (t[1] - t[0]) & g.mask;
            Note(g.lanes[lane], static_cast<double>(ticks) * g.nsPerTick / 1.0e6);
        }
        s.written = 0;
    }
}

namespace wxlvk::stages::timing
{
    bool Open(VkCommandBuffer cmd, uint32_t slot, uint64_t serial)
    {
        if (!g.tried) g.available = Init();
        if (!g.available || slot >= kSlots) return false;
        g.current = slot;
        SlotQueries& s = g.slots[slot];
        if (s.serial == serial) return true;
        Harvest(s);
        gpu::Api()->CmdResetQueryPool(cmd, s.pool, 0, 2 * kLanes);
        s.serial = serial;
        return true;
    }

    void Start(VkCommandBuffer cmd, uint32_t lane)
    {
        if (!g.available || lane >= kLanes) return;
        // BOTTOM_OF_PIPE at the start as well: the timestamp is taken once every earlier command has
        // finished, so the span does not include the tail of the engine's draws still in flight
        // (TOP_OF_PIPE did, which is what made the first figures 2 to 10 times too high).
        gpu::Api()->CmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, g.slots[g.current].pool, 2 * lane);
    }

    void Stop(VkCommandBuffer cmd, uint32_t lane)
    {
        if (!g.available || lane >= kLanes) return;
        SlotQueries& s = g.slots[g.current];
        gpu::Api()->CmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, s.pool, 2 * lane + 1);
        s.written |= 1u << lane;
    }

    const Lane& Get(uint32_t lane) { return g.lanes[lane < kLanes ? lane : 0]; }

    void Reset()
    {
        for (Lane& l : g.lanes)
        {
            l.sumMs   = 0.0;
            l.maxMs   = 0.0;
            l.samples = 0;
        }
    }

    bool Available() { return g.available; }
}
