// The world's sub-pixel jitter, for a temporal resolve.
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

#include "native/render/Jitter.hpp"

#include "native/Present.hpp"

namespace
{
    constexpr uint32_t kSamples = 8;

    /// The radical inverse of `i` in `base`: the Halton sequence.
    float Halton(uint32_t i, uint32_t base)
    {
        float f = 1.0f, r = 0.0f;
        for (; i > 0; i /= base)
        {
            f /= static_cast<float>(base);
            r += f * static_cast<float>(i % base);
        }
        return r;
    }

    struct State
    {
        bool     inWorld  = false;
        bool     active   = false;
        uint32_t requests = 0;   ///< consumers asking for the jitter (RequestJitter)
        uint32_t index    = 0;
        float    x        = 0.0f;
        float    y        = 0.0f;
        uint64_t jittered = 0;
        uint64_t skipped  = 0;
    };
    State g;
}

namespace wxlvk::native::render::jitter
{
    void BeginWorld()
    {
        g.inWorld = true;
        g.active = g.requests > 0;
        if (!g.active)
        {
            g.x = g.y = 0.0f;
            return;
        }
        g.index = g.index % kSamples + 1;   // 1 .. 8: Halton(0) is 0 in both bases
        g.x = Halton(g.index, 2) - 0.5f;
        g.y = Halton(g.index, 3) - 0.5f;
    }

    void EndWorld() { g.inWorld = false; }

    void Apply(VkViewport& viewport)
    {
        if (!g.inWorld || !g.active || !wxlvk::native::present::OnBackBuffer()) return;
        if (viewport.maxDepth >= 0.999f)
        {
            ++g.skipped;
            return;
        }
        viewport.x += g.x;
        viewport.y += g.y;
        ++g.jittered;
    }

    void Current(float& x, float& y)
    {
        x = g.active ? g.x : 0.0f;
        y = g.active ? g.y : 0.0f;
    }

    bool Active() { return g.active; }

    void Request(bool wanted)
    {
        if (wanted) ++g.requests;
        else if (g.requests > 0) --g.requests;
    }

    void Counts(uint64_t& jittered, uint64_t& skipped)
    {
        jittered = g.jittered;
        skipped  = g.skipped;
    }
}
