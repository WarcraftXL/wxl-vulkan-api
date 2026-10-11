// The present path's occlusion query API.
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

using namespace wxlvk::native::present::detail;

namespace wxlvk::native::present
{
    // --- queries ---------------------------------------------------------------------------------------

    bool QueryStart(QueryRun& run)
    {
        if (!EnsureFrame()) return false;
        QueryForget(run);
        if (s.queryPools[0] == VK_NULL_HANDLE)
        {
            VkQueryPoolCreateInfo info = {};
            info.sType      = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
            info.queryType  = VK_QUERY_TYPE_OCCLUSION;
            info.queryCount = kQueriesPerSlot;
            for (uint32_t i = 0; i < kFramesInFlight; ++i)
            {
                const VkResult r = s.dapi->CreateQueryPool(s.device, &info, nullptr, &s.queryPools[i]);
                if (r != VK_SUCCESS)
                {
                    s.queryPools[i] = VK_NULL_HANDLE;
                    wxl::ScriptMgr::Log(WXL_LOG_ERROR, "native: queries: vkCreateQueryPool failed (%s); "
                                        "occlusion queries answer 0", ResultName(r));
                    return false;
                }
            }
            // D3D9's occlusion query is an exact sample count, and the glare divides it by the quad's
            // area: precise whenever the device allows it.
            const context::Handles* h = context::Get();
            s.queryControl = h && h->occlusionQueryPrecise ? VK_QUERY_CONTROL_PRECISE_BIT : 0u;
            // This frame's pool was not reset at its open (it did not exist): reset ahead of the frame.
            // The other slot's is reset at its own next open.
            VkCommandBuffer pre = PrePass();
            if (!pre) return false;
            s.dapi->CmdResetQueryPool(pre, s.queryPools[s.fi], 0, kQueriesPerSlot);
            wxl::ScriptMgr::Log(WXL_LOG_DEBUG, "native: queries: occlusion query pools made (%u per frame "
                                "slot, %s); the first engine query began at serial %llu", kQueriesPerSlot,
                                s.queryControl ? "precise" : "not precise: the device has no occlusionQueryPrecise",
                                static_cast<unsigned long long>(s.serial));
        }
        if (s.activeRun && s.activeRun != &run)
        {
            QueryStop(*s.activeRun);
            ++s.stats.queryNested;
        }
        run = QueryRun();
        run.serial = s.serial;
        run.slot   = s.fi;
        run.active = true;
        if (s.runCount[s.fi] < kMaxRunsPerSlot)
        {
            s.runs[s.fi][s.runCount[s.fi]++] = &run;
            run.registered = true;
        }
        else
            run.overflow = true;   // not harvested at the slot's reopen: read before then, or lost
        s.activeRun = &run;
        ++s.stats.queriesBegun;
        if (s.passOpen) QueryBeginVk(run);
        return true;
    }

    void QueryStop(QueryRun& run)
    {
        if (!run.active) return;
        QueryEndVk(run);
        run.active = false;
        if (s.activeRun == &run) s.activeRun = nullptr;
    }

    void QueryForget(QueryRun& run)
    {
        if (run.active) QueryStop(run);
        if (s.activeRun == &run) s.activeRun = nullptr;
        Unregister(run);
    }

    bool QueryReady(QueryRun& run)
    {
        if (run.serial == 0) return true;   // never issued: D3D's new query is signalled
        if (run.active) return false;
        if (run.harvested) return true;
        if (!run.registered && s.serial >= run.serial + kFramesInFlight)
        {
            // Not on a harvest list (more runs than it holds) and its slot reopened since: lost.
            run.samples   = 0;
            run.harvested = true;
            return true;
        }
        if (run.serial > s.completed) Poll();
        if (run.serial > s.completed) return false;
        return Harvest(run, false);
    }

    bool QueryActive() { return s.activeRun != nullptr && s.activeRun->active; }

    void NoteQueryDraw(bool drawn)
    {
        if (!s.activeRun) return;
        ++s.activeRun->draws;
        if (drawn) ++s.activeRun->drawn;
    }

    bool QueryInPass() { return s.activeRun != nullptr && s.activeRun->inPass; }

    uint32_t PassDepthKind(VkFormat& format)
    {
        format = VK_FORMAT_UNDEFINED;
        if (!s.passOpen || !s.passHasDepth) return 0;
        format = s.passDepth.format;
        if (s.passDepth.back) return 1;
        if (s.passDepth.tex) return 2;
        return 3;
    }

    bool QueryWait(QueryRun& run)
    {
        if (QueryReady(run)) return true;
        if (run.active) return false;
        if (s.open && run.serial == s.serial)
        {
            if (!FlushAndWait()) return false;
            return Harvest(run, true);
        }
        if (!WaitSerial(run.serial)) return false;
        return Harvest(run, false) || Harvest(run, true);
    }
}
