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

#pragma once

#include "vulkan/Vk.hpp"

#include <cstdint>

// The jitter moves a draw's Vulkan viewport origin by a sub-pixel offset (an 8-sample Halton 2/3
// sequence, one sample per world frame), which shifts everything the draw rasterises by exactly
// that much and changes nothing else -- no projection is touched, so no shader constant, no
// culling and no engine state sees it. A draw is jittered when all three hold:
//
//   - it is recorded inside the world scene pass (between the entry and the return of the core's
//     Gx.WorldOnRender, src/native/render/WorldPass.cpp): the world text and the interface come after it;
//   - its pass is on the back buffer: portraits and other texture targets keep still;
//   - its viewport's depth range ends below 1: the world draws in 0 .. 0.94, and the engine's own
//     full-screen passes inside the world pass (glow composite, screen effects), which resample an
//     image already drawn, use 0 .. 1 and are left alone.
//
// It is on only while a consumer asks for it (wxl.vk.render RequestJitter: wxl-vulkan-pp's TAA); the
// counters let a log confirm the split.
namespace wxlvk::native::render::jitter
{
    /// A consumer asks for the jitter (`wanted`) or stops asking. Counted; it applies from the next
    /// world pass.
    void Request(bool wanted);

    /// The world scene pass begins: the next jitter sample, while any consumer asks for it.
    void BeginWorld();

    /// The world scene pass ended.
    void EndWorld();

    /// Moves a draw's viewport by the jitter when the draw qualifies (see above). From every
    /// place a draw's viewport is made (src/native/Draw.cpp).
    void Apply(VkViewport& viewport);

    /// The jitter of the latest world pass, in pixels (0, 0 when it was off).
    void Current(float& x, float& y);

    /// Whether the latest world pass was jittered.
    bool Active();

    /// Draws jittered, and draws inside the world pass on the back buffer left alone for their
    /// depth range, since the start.
    void Counts(uint64_t& jittered, uint64_t& skipped);
}
