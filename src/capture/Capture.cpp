// The frame's opaque world draws, captured as the pipelines record them, for the shadow replay.
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

#include "capture/Capture.hpp"

#include "native/Present.hpp"

namespace
{
    using wxlvk::capture::Draw;
    using wxlvk::capture::kMaxDraws;

    struct State
    {
        uint32_t requests = 0;
        bool     inWorld  = false;
        bool     closed   = false;   ///< AfterOpaqueWorld fired in the frame being captured
        uint64_t serial   = 0;       ///< the frame the list belongs to
        uint32_t count    = 0;
        Draw     draws[kMaxDraws];
        wxlvk::capture::Stats stats = {};
    };
    State g;

    /// A new frame's list starts empty.
    void Roll()
    {
        const uint64_t now = wxlvk::native::present::Serial();
        if (g.serial == now) return;
        if (g.count) ++g.stats.frames;
        g.serial          = now;
        g.count           = 0;
        g.closed          = false;
        g.stats.lastFrame = 0;
    }
}

namespace wxlvk::capture
{
    void Request(bool wanted)
    {
        if (wanted) ++g.requests;
        else if (g.requests) --g.requests;
    }

    void BeginWorld()
    {
        if (!g.requests) return;
        Roll();
        g.inWorld = true;
    }

    void EndWorld() { g.inWorld = false; }

    void CloseOpaque()
    {
        if (g.inWorld) g.closed = true;
    }

    bool Open()
    {
        if (!g.requests || !g.inWorld || g.closed || !native::present::OnBackBuffer()) return false;
        Roll();
        if (g.count < kMaxDraws) return true;
        ++g.stats.overflow;
        return false;
    }

    void Note(const Draw& d)
    {
        g.draws[g.count++] = d;
        g.stats.lastFrame = g.count;
        ++g.stats.taken;
        if (d.live) ++g.stats.movers;
    }

    const Draw* Draws(uint32_t& count)
    {
        count = g.serial == native::present::Serial() ? g.count : 0;
        return g.draws;
    }

    uint64_t Serial() { return g.serial; }

    void GetStats(Stats& out) { out = g.stats; }
}
