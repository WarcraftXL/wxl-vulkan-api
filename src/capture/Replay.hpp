// The shadow replay: the frame's captured opaque world drawn again, depth only, from a light.
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

// Each captured draw is bound as it drew -- its vertex and index buffers, its ring slice (set 1, the
// constants it drew with), its palette (set 2), its textures (set 0, only for an alpha-tested caster)
// -- with its shadow pipeline (src/capture/ShadowPipelines.cpp), and one matrix in the push range:
// inverse(the frame's view-projection) x the light's view-projection, which takes the engine's clip
// position, whatever the module made it from, into the light's clip space. Then the SHADOW_CASTERS
// passes draw the consumers' own casters into the same pass.
namespace wxlvk::capture
{
    /// wxl.vk.render ReplayWorld.
    bool Replay(const WXL_VkReplayDesc& desc, WXL_VkReplayStats* stats);

    /// wxl.vk.render ReplayWorldViews: several views in one walk of the capture and one render pass (the LOAD class).
    bool ReplayViews(const WXL_VkReplayViewsDesc& desc, WXL_VkReplayStats* stats);

    /// The stats lines: captures, replays, draws, shadow pipelines.
    void LogStats();
}
