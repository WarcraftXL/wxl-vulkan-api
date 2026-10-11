// The engine's batched-doodad draw entry, bracketed: what is drawn inside it is a placed doodad.
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

// Not the movers' rule (taking the unbatched doodads for movers made them flicker); the bracket stays
// for the figures. A mover is a unit's draw, by its texture's folder: vulkan/pipelines/Record.cpp.

// The M2 scene draws the map's placed doodads through its hardware-instanced batch path
// (M2.DrawBatchDoodad, include/wxl/offsets/engine/Gx.hpp kDrawBatchDoodad) and everything else one
// draw at a time: the units, the players, the creatures, the mounts, the doodads its batch path refused
// (a fading one, a special material). The shadow capture (src/capture/) reads this bracket to tell what
// moves from what stands: a single M2 draw is taken as a mover whether or not it uses the bone palette,
// because a character of few bones per batch does not, and a cached shadow map must never hold it.
namespace wxlvk::native::render::doodadbatch
{
    /// Attaches the bracket; false when the point could not be hooked (every M2 draw then counts as a
    /// mover: correct shadows, the far caches without the placed doodads).
    bool Install();

    /// Whether the engine is inside its batched-doodad draw (the render thread).
    bool Inside();

    /// Whether the bracket is attached.
    bool Hooked();
}
