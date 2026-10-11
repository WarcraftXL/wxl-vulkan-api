// The shadow replay's pipelines: one per captured pipeline entry (and alpha test), depth only.
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

// The entry's vertex input and layout (pipelines::ShadowSourceFor), its vertex module's shadow
// variant (shadowgen.py: the clip position re-projected by the push range's matrix), its pixel module
// only when the draw is alpha-tested (the module ends with that test), no colour attachment, no
// culling (both faces cast: thin geometry has no back), depth test and write, the bias dynamic.
// Built on first use against the shadow pass class, through the persisted pipeline cache; a build
// that fails is remembered and its draws skipped.
namespace wxlvk::capture::shadowpipes
{
    struct Pipe
    {
        VkPipeline       pipeline = VK_NULL_HANDLE;
        VkPipelineLayout layout   = VK_NULL_HANDLE;
        bool             bones    = false;   ///< binds set 2 (the palette)
        bool             wind     = false;   ///< binds set 3 and pushes the wind's values, the matrix at offset 16
    };

    /// The pipeline for (entry, alphaTest); false when it cannot be had.
    bool Get(uint32_t entry, bool alphaTest, Pipe& out);

    /// Pipelines built and builds that failed, for the stats.
    void Counts(uint32_t& built, uint32_t& failed);
}
