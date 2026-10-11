// Occlusion queries inside the engine's pass.
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

namespace wxlvk::native::present::detail
{
    // --- queries in the pass -------------------------------------------------------------------------

    void QueryBeginVk(QueryRun& run)
    {
        if (run.inPass || !s.passOpen || run.slot != s.fi) return;
        const VkQueryPool pool = s.queryPools[s.fi];
        if (pool == VK_NULL_HANDLE || run.count >= present::kQuerySplits || s.queryNext[s.fi] >= kQueriesPerSlot)
        {
            if (!run.overflow) ++s.stats.queryOverflow;
            run.overflow = true;
            return;
        }
        const uint32_t index = s.queryNext[s.fi]++;
        s.dapi->CmdBeginQuery(s.slots[s.fi].main, pool, index, s.queryControl);
        run.index[run.count++] = index;
        run.inPass = true;
        if (run.count > 1) ++s.stats.querySplits;
    }

    void QueryEndVk(QueryRun& run)
    {
        if (!run.inPass) return;
        if (s.passOpen) s.dapi->CmdEndQuery(s.slots[s.fi].main, s.queryPools[run.slot], run.index[run.count - 1]);
        run.inPass = false;
    }

    void Unregister(QueryRun& run)
    {
        if (!run.registered) return;
        run.registered = false;
        const uint32_t slot = run.slot < kFramesInFlight ? run.slot : 0;
        for (uint32_t i = 0; i < s.runCount[slot]; ++i)
            if (s.runs[slot][i] == &run)
            {
                s.runs[slot][i] = s.runs[slot][s.runCount[slot] - 1];
                s.runs[slot][--s.runCount[slot]] = nullptr;
                return;
            }
    }

    /// Sums a run's Vulkan queries; false while one is not ready (only without `wait`).
    bool Harvest(QueryRun& run, bool wait)
    {
        uint64_t sum = 0;
        for (uint32_t i = 0; i < run.count; ++i)
        {
            uint32_t value = 0;
            const VkResult r = s.dapi->GetQueryPoolResults(s.device, s.queryPools[run.slot], run.index[i], 1,
                                                           sizeof value, &value, sizeof value,
                                                           wait ? VK_QUERY_RESULT_WAIT_BIT : 0u);
            if (r == VK_NOT_READY) return false;
            if (r == VK_SUCCESS) sum += value;
        }
        run.samples   = sum;
        run.harvested = true;
        Unregister(run);
        return true;
    }
}
