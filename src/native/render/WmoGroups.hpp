// The engine's map object group renderers, bracketed: which draws light an interior.
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

#include "wxl/VkRenderApi.h"

#include <cstdint>

// The interior split. The engine draws every batch of a map object group from one of two leaves
// (Wmo.ExtRender, Wmo.IntRender: __thiscall(root, group, flag)). While one runs, this bracket knows the group
// and its MOGP flags: a group with 0x2000 (INTERIOR) and without 0x40 (EXTERIOR_LIT) is lit by the engine's
// interior light (its baked vertex colours, the WMO's ambient), never by the sun's bands. The draw reader
// (draw/Reader.cpp WriteMoverRow) writes that into the row p[14].y, which the G-buffer pixel variants copy
// into the albedo's bit 5 (WXL_VK_GBUFFER_INTERIOR_BIT), and hands the draw's own transform here once per
// group, so the frame's interior groups can be handed over as boxes in view space (WXL_VkGBuffer::interiors):
// the models and units standing inside, whose batched draws know nothing of the group, are told by them.
// Every INDOOR group (0x2000) hands its box over, the exterior-lit ones (0x40) flagged
// (boundsMin[3] = WXL_VK_INTERIOR_EXTERIOR_LIT): their air is enclosed (the fog keeps the outdoor air out of a
// mine's or a cave's exterior-lit tunnels), their light the sun's bands (the light relights them as exteriors).
// Each box also says which group it is (WXL_VkGBuffer::interiorGroups): the WMO's model key (its root's name)
// and the group's index in the root's group array, so a lighting consumer can keep an indoor light to its own groups and
// those one portal away (no light from the floor below through the boards).
namespace wxlvk::native::render::wmogroups
{
    /// Attaches the bracket; false when either leaf could not be hooked (no pixel is then marked interior).
    bool Install();

    /// Whether the draw being read is a batch of an interior-lit group (the render thread, inside a leaf).
    bool InteriorDraw();

    /// Whether the draw being read is a batch of an indoor group, exterior-lit or not.
    bool EnclosedDraw();

    /// The draw being read is an indoor group's batch with this vertex register file (c31..c33: the
    /// map object's space to view space): the group's box is recorded for the frame, once per group and
    /// placement, flagged when the group is exterior-lit. Call only when EnclosedDraw() said so.
    void NoteInteriorDraw(const float (*vertexRows)[4]);

    /// The bounding sphere, in view space (xyz, radius), of the map object group whose batch is being read with this
    /// vertex register file (c31..c33: the map object's space to view space): its MOGP box through the transform. For
    /// the shadow replay's culling (capture::Draw::bound). False (out untouched) outside a group renderer, or for a box
    /// that is not finite.
    bool DrawBound(const float (*vertexRows)[4], float out[4]);

    /// The interior groups the frame `serial` drew so far (in its view space): their count, *out the array
    /// (valid until the next frame records its first one). 0 for another frame.
    uint32_t Volumes(uint64_t serial, const WXL_VkInteriorVolume** out);

    /// Which group each of those volumes is (the WMO's model key, the group's index in its root, its MOGP flags), the same
    /// order; valid with the array Volumes handed out.
    const WXL_VkInteriorGroup* Groups();

    struct Stats
    {
        bool     hooked = false;
        uint64_t leaves = 0;         ///< leaf calls seen
        uint64_t interiorLeaves = 0; ///< of them, an interior group's
        uint64_t rootMismatch = 0;   ///< leaf calls whose group did not name the leaf's root (not trusted)
        uint32_t lastVolumes = 0;    ///< interior groups the last finished frame recorded
        uint64_t dropped = 0;        ///< groups past WXL_VK_GBUFFER_MAX_INTERIORS, dropped
    };
    const Stats& GetStats();
}
