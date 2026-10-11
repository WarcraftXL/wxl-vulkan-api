// The opaque depth's min / max pyramid, built for the rendering API's consumers while one asks (wxl.vk.render
// RequestDepthPyramid, GetDepthPyramid).
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
#include "wxl/VkRenderApi.h"

#include <cstdint>

// Built at AfterOpaqueWorld in the MidPass break (src/native/api/Points.cpp), right after the opaque depth copy and
// before the consumers' MidPasses, from the scene depth as it is there (SHADER_READ_ONLY_OPTIMAL): the opaque world,
// no consumer's geometry -- the same values as GetOpaqueDepth. One image (R32G32_SFLOAT, or R32G32B32A32_SFLOAT on a
// device without the extended storage formats), its levels down to 1 x 1, one compute dispatch a level
// (src/shaders/glsl/depthpyramid.comp) with a barrier between; its writes made visible to the compute, vertex and fragment
// stages of everything after it, its earlier frame's reads waited for first (one image: this frame overwrites the last).
// GENERAL layout for its life. Nothing is made, recorded or kept while nobody asks; the last asker's withdrawal retires it.
namespace wxlvk::native::render::depthpyramid
{
    /// Whether any consumer asks for it (the MidPass break is then made at AfterOpaqueWorld).
    bool Wanted();

    /// Records this frame's pyramid into `cmd` (outside a pass) from the scene depth `depthView`.
    void Build(VkCommandBuffer cmd, VkImageView depthView);

    /// wxl.vk.render RequestDepthPyramid: counted.
    void Request(bool wanted);

    /// wxl.vk.render GetDepthPyramid: this frame's, filled up to out.structSize. False when none was built this frame.
    bool Get(WXL_VkDepthPyramid& out);

    /// Pyramids built since the start, for the stats line.
    uint64_t Builds();
}
