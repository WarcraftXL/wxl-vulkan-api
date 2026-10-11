// The frame's GPU time and the device's shared work, from timestamp queries (wxl.vk.render GetFrameStats).
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

#include "native/present/Internal.hpp"

// One query pool per frame slot, kQueries timestamps: the main buffer's start (TOP_OF_PIPE, at the frame's open), the
// frame's end (BOTTOM_OF_PIPE, after its last pass: the present's copy and its wait for the swapchain image are left out,
// so vsync does not read as GPU time), the pre-pass's start (the pre-pass is submitted first; it resets its own query,
// since a reset recorded in the main buffer would run after it), and two per lane. A slot's results are read at its next
// open, after its fence: they are there, and a result that is not is skipped, never waited for. The frame's span is from
// the earlier start to the end -- what the GPU spent on the frame's submission, the modules' work included; a mid-frame
// flush (a read-back) puts its CPU wait into that frame's span, which the line's maximum then shows. Kept for the device's
// life, as the stages' pools (src/stages/Timing.cpp): nothing Vulkan runs at exit.
namespace wxlvk::native::present::detail
{
    namespace
    {
        enum Query : uint32_t
        {
            kMainStart = 0,
            kFrameEnd,
            kLaneFirst,                                   ///< lane L's ends at kLaneFirst + 2 L, + 1
            kPreStart = kLaneFirst + 2 * kGpuLaneCount,   ///< reset and written in the pre-pass
            kQueries
        };

        struct GpuSlot
        {
            VkQueryPool pool     = VK_NULL_HANDLE;
            bool        open     = false;   ///< the frame's start was written (kMainStart)
            bool        ended    = false;   ///< kFrameEnd written
            bool        pre      = false;   ///< kPreStart written
            uint32_t    lanes    = 0;       ///< lanes both of whose ends were written
            uint32_t    breaks   = 0;
        };

        struct Ring
        {
            double   ms[kGpuWindow] = {};
            uint32_t count = 0, next = 0;
            double   last  = -1.0;
            uint64_t at    = 0;   ///< the harvest that noted the latest value
            void Note(double v, uint64_t harvest)
            {
                at = harvest;
                ms[next] = v;
                next     = (next + 1) % kGpuWindow;
                if (count < kGpuWindow) ++count;
                last = v;
            }
            double Avg() const
            {
                if (!count) return -1.0;
                double sum = 0.0;
                for (uint32_t i = 0; i < count; ++i) sum += ms[i];
                return sum / count;
            }
            double Max() const
            {
                if (!count) return -1.0;
                double m = 0.0;
                for (uint32_t i = 0; i < count; ++i) m = ms[i] > m ? ms[i] : m;
                return m;
            }
        };

        struct State
        {
            bool     tried     = false;
            bool     available = false;
            double   nsPerTick = 1.0;
            uint64_t mask      = ~0ull;
            GpuSlot  slots[kFramesInFlight];
            uint32_t current   = 0;     ///< the slot of the frame being recorded
            bool     recording = false; ///< a frame is open (its main start written)
            Ring     frame;
            Ring     lanes[kGpuLaneCount];
            uint32_t breaks    = 0;
            uint64_t harvests  = 0;   ///< frames read so far
            double   sessionSum = 0.0, sessionMax = 0.0;
            uint64_t sessionCount = 0;
        };
        State t;

        /// A lane that has not run for a whole window reads as not run (-1), not as its old figures.
        bool Stale(const Ring& r) { return r.count == 0 || t.harvests > r.at + kGpuWindow; }

        bool Init()
        {
            t.tried = true;
            const context::Handles* h = context::Get();
            if (!h || !h->iapi || !h->iapi->GetPhysicalDeviceProperties || !h->dapi->CreateQueryPool
                || !h->dapi->CmdWriteTimestamp || !h->dapi->GetQueryPoolResults)
                return false;
            VkPhysicalDeviceProperties props = {};
            h->iapi->GetPhysicalDeviceProperties(h->physical, &props);
            uint32_t count = 0;
            h->iapi->GetPhysicalDeviceQueueFamilyProperties(h->physical, &count, nullptr);
            VkQueueFamilyProperties families[16] = {};
            if (count > 16) count = 16;
            h->iapi->GetPhysicalDeviceQueueFamilyProperties(h->physical, &count, families);
            const uint32_t bits = h->graphicsFamily < count ? families[h->graphicsFamily].timestampValidBits : 0;
            if (bits == 0 || props.limits.timestampPeriod <= 0.0f) return false;
            t.mask      = bits >= 64 ? ~0ull : ((1ull << bits) - 1);
            t.nsPerTick = props.limits.timestampPeriod;
            VkQueryPoolCreateInfo info = {};
            info.sType      = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
            info.queryType  = VK_QUERY_TYPE_TIMESTAMP;
            info.queryCount = kQueries;
            for (GpuSlot& sl : t.slots)
                if (h->dapi->CreateQueryPool(h->device, &info, nullptr, &sl.pool) != VK_SUCCESS)
                {
                    sl.pool = VK_NULL_HANDLE;
                    return false;
                }
            return true;
        }

        bool Read(const GpuSlot& sl, uint32_t first, uint32_t n, uint64_t* out)
        {
            return s.dapi->GetQueryPoolResults(s.device, sl.pool, first, n, n * sizeof(uint64_t), out, sizeof(uint64_t),
                                               VK_QUERY_RESULT_64_BIT) == VK_SUCCESS;
        }

        double Ms(uint64_t a, uint64_t b) { return static_cast<double>((b - a) & t.mask) * t.nsPerTick / 1.0e6; }

        /// The slot's previous frame, read (its fence was just waited).
        void Harvest(GpuSlot& sl)
        {
            ++t.harvests;
            if (sl.open && sl.ended)
            {
                uint64_t ends[2] = {};
                uint64_t pre     = 0;
                if (Read(sl, kMainStart, 2, ends))
                {
                    uint64_t start = ends[0];
                    // The pre-pass ran first: its start, when it is the earlier one (a wrap of the counter aside).
                    if (sl.pre && Read(sl, kPreStart, 1, &pre) && ((ends[0] - pre) & t.mask) < (t.mask >> 1)) start = pre;
                    const double ms = Ms(start, ends[1]);
                    t.frame.Note(ms, t.harvests);
                    t.sessionSum += ms;
                    t.sessionMax = ms > t.sessionMax ? ms : t.sessionMax;
                    ++t.sessionCount;
                    t.breaks = sl.breaks;
                }
            }
            for (uint32_t lane = 0; lane < kGpuLaneCount; ++lane)
            {
                if (!(sl.lanes & (1u << lane))) continue;
                uint64_t ends[2] = {};
                if (Read(sl, kLaneFirst + 2 * lane, 2, ends)) t.lanes[lane].Note(Ms(ends[0], ends[1]), t.harvests);
            }
            sl.open = sl.ended = sl.pre = false;
            sl.lanes = 0;
        }
    }

    void FrameGpuOpen(VkCommandBuffer main, uint32_t slot, uint64_t /*serial*/)
    {
        t.recording = false;
        if (!t.tried) t.available = Init();
        if (!t.available || slot >= kFramesInFlight || !main) return;
        GpuSlot& sl = t.slots[slot];
        Harvest(sl);
        t.current = slot;
        // Every query but the pre-pass's, which the pre-pass resets itself.
        s.dapi->CmdResetQueryPool(main, sl.pool, 0, kPreStart);
        s.dapi->CmdWriteTimestamp(main, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, sl.pool, kMainStart);
        sl.open     = true;
        t.recording = true;
    }

    void FrameGpuPrePass(VkCommandBuffer pre, uint32_t slot)
    {
        if (!t.recording || slot != t.current || !pre) return;
        GpuSlot& sl = t.slots[slot];
        s.dapi->CmdResetQueryPool(pre, sl.pool, kPreStart, 1);
        s.dapi->CmdWriteTimestamp(pre, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, sl.pool, kPreStart);
        sl.pre = true;
    }

    void FrameGpuEnd(VkCommandBuffer main, uint32_t slot)
    {
        if (!t.recording || slot != t.current || !main) return;
        GpuSlot& sl = t.slots[slot];
        s.dapi->CmdWriteTimestamp(main, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, sl.pool, kFrameEnd);
        sl.ended    = true;
        sl.breaks   = s.breaksThisFrame;
        t.recording = false;
    }
}

namespace wxlvk::native::present
{
    void GpuLaneBegin(VkCommandBuffer cmd, uint32_t lane)
    {
        using namespace detail;
        if (!t.recording || lane >= kGpuLaneCount || !cmd) return;
        s.dapi->CmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, t.slots[t.current].pool, kLaneFirst + 2 * lane);
    }

    void GpuLaneEnd(VkCommandBuffer cmd, uint32_t lane)
    {
        using namespace detail;
        if (!t.recording || lane >= kGpuLaneCount || !cmd) return;
        GpuSlot& sl = t.slots[t.current];
        s.dapi->CmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, sl.pool, kLaneFirst + 2 * lane + 1);
        sl.lanes |= 1u << lane;
    }

    void GetGpuTimes(GpuTimes& out)
    {
        using namespace detail;
        out = GpuTimes();
        out.timed = t.available;
        if (!t.available) return;
        out.frameLast = t.frame.last;
        out.frameAvg  = t.frame.Avg();
        out.frameMax  = t.frame.Max();
        out.frameSessionCount = t.sessionCount;
        if (t.sessionCount)
        {
            out.frameSessionAvg = t.sessionSum / static_cast<double>(t.sessionCount);
            out.frameSessionMax = t.sessionMax;
        }
        for (uint32_t i = 0; i < kGpuLaneCount; ++i)
        {
            if (Stale(t.lanes[i])) continue;
            out.laneAvg[i]  = t.lanes[i].Avg();
            out.laneLast[i] = t.lanes[i].last;
        }
        out.passBreaks = t.breaks;
    }
}
