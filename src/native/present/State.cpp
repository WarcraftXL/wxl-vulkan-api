// The present path's state, its frame times and its retirement room.
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
    State s;
    uint64_t serialNow = 1;

    void NoteFrameTime(wxlvk::native::present::FrameTimes& t, double ms)
    {
        if (t.frames == 0 || ms < t.minMs) t.minMs = ms;
        if (ms > t.maxMs) t.maxMs = ms;
        t.sumMs += ms;
        ++t.frames;
        if (ms > kSlowFrameMs) ++t.slow;
        if (ms > kHitchFrameMs) ++t.hitches;
    }

    /// Learns, without blocking, which submitted frames have finished: one fence status per slot.
    void Poll()
    {
        if (!s.dapi || !s.device || !s.dapi->GetFenceStatus) return;
        for (FrameSlot& slot : s.slots)
            if (slot.serial > s.completed && slot.fence != VK_NULL_HANDLE
                && s.dapi->GetFenceStatus(s.device, slot.fence) == VK_SUCCESS)
                s.completed = slot.serial;
    }

    /// Room for one more retirement: grows the list, false when it is at kMaxRetired or the heap said no.
    bool RetireRoom()
    {
        if (s.retiredCount < s.retiredCap) return true;
        if (s.retiredCap >= kMaxRetired) return false;
        const uint32_t cap   = s.retiredCap ? s.retiredCap * 2 : kRetiredFirst;
        Retired*       fresh = new (std::nothrow) Retired[cap];
        if (!fresh) return false;
        for (uint32_t i = 0; i < s.retiredCount; ++i) fresh[i] = s.retired[i];
        delete[] s.retired;
        s.retired    = fresh;
        s.retiredCap = cap;
        return true;
    }

    void Fail(const char* what, VkResult r)
    {
        s.failed = true;
        wxl::ScriptMgr::Log(WXL_LOG_ERROR, "native: frame: %s failed (%s); the frame stops, the client "
                            "keeps running with nothing on its window", what, ResultName(r));
    }
}
