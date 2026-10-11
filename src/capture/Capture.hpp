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

#pragma once

#include "vulkan/Vk.hpp"

#include <cstdint>

// While a consumer asks for it (wxl.vk.render RequestDrawCapture), every draw pipelines::Record
// makes inside the world scene pass, on the back buffer, before AfterOpaqueWorld, that writes depth
// with blend opaque or alpha-key, and whose vertex module has a shadow variant, appends one record:
// what the replay binds again (src/capture/Replay.cpp). Everything a record names -- the uniform
// ring's slice, the descriptor sets, the bone palette, the pool buffers -- lives until the frame's
// fence, so the replay later in the same frame can bind it as it was. No register is copied.
namespace wxlvk::capture
{
    /// The most draws one frame keeps; the rest are counted and not replayed.
    constexpr uint32_t kMaxDraws = 16384;   ///< with every frustum of the engine widened towards the sun (wxl-light), a frame holds more

    /// One captured draw.
    struct Draw
    {
        uint32_t        entry;          ///< the pipeline entry (pipelines::ShadowPipeline)
        VkDescriptorSet set0;           ///< its textures (read only when alpha-tested)
        VkDescriptorSet ring;           ///< set 1 and its dynamic offset: the slice it drew with
        uint32_t        ringOffset;
        VkDescriptorSet bones;          ///< set 2 and its offset; VK_NULL_HANDLE without a palette
        uint32_t        bonesOffset;
        VkBuffer        vbs[4];
        uint32_t        vbCount;
        VkBuffer        ib;
        uint32_t        firstIndex;
        uint32_t        count;          ///< index count (vertex count when not indexed)
        int32_t         vertexOffset;
        uint8_t         indexed;
        uint8_t         alphaTest;      ///< its pixel module's alpha test matters (leaves, cards)
        uint8_t         wind;           ///< a tree-wind draw: set 3 and the model's push values below
        uint8_t         live;           ///< a unit's draw (a character's, a creature's or an item's texture): a mover, never cached
        uint8_t         terrain;        ///< a terrain draw (a Terrain vertex family): what a TERRAIN_ONLY replay keeps alone
        uint8_t         doodad;         ///< a static M2 doodad's draw (a Diffuse_ vertex family, no unit's, no tree-wind draw): WITH_DOODADS
        uint8_t         mapobj;         ///< a map object's draw (a MapObj vertex family): WITH_MAPOBJECTS
        VkDescriptorSet windSet;        ///< set 3 of its frame slot (src/vulkan/Wind.hpp)
        int32_t         windPush[4];    ///< the model's wind::Draw, as the draw pushed it
        /// Its bounding sphere in the frame's VIEW space (xyz, radius), for the replay's culling; radius < 0 when it
        /// has none (the terrain, a map object's batch drawn outside the group renderers, a model whose rows say
        /// nothing): never culled. A map object's is its group's box (native/render/WmoGroups.cpp DrawBound).
        float           bound[4];
    };

    /// A consumer asks for the capture or stops asking (counted).
    void Request(bool wanted);

    /// The world scene pass begins / ends (src/native/render/WorldPass.cpp's detour).
    void BeginWorld();
    void EndWorld();

    /// AfterOpaqueWorld has fired: the opaque world is complete; nothing more is taken this frame.
    void CloseOpaque();

    /**
     * Whether the draw being recorded may be taken: asked, in the world pass before the opaque
     * world closed, on the back buffer, in a frame not yet full. The draw's own state (opaque,
     * depth-writing, a shadow variant) is the caller's to test (pipelines::Record).
     */
    bool Open();

    /// Appends one record (the caller has checked Open() and the draw's state).
    void Note(const Draw& d);

    /// The frame's records; valid until the next frame opens.
    const Draw* Draws(uint32_t& count);

    /// The frame being captured (its serial), so a stale list is never replayed.
    uint64_t Serial();

    /// Counters for the stats line.
    struct Stats
    {
        uint64_t frames;      ///< frames with a capture
        uint64_t taken;       ///< draws captured
        uint64_t movers;      ///< of them, taken as movers (the M2 scene's single draws, native/render/DoodadBatch.cpp)
        uint64_t overflow;    ///< draws past kMaxDraws
        uint32_t lastFrame;   ///< draws the latest frame captured
    };
    void GetStats(Stats& out);
}
