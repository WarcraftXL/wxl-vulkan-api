// Last frame's opaque scene depth, kept for the rendering API's consumers (the foliage's occlusion
// culling builds its Hi-Z pyramid from it in Prepare, before the world is drawn).
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

// While a consumer has asked for it (WXL_VkRenderApi::RequestPreviousDepth): at AfterOpaqueWorld,
// BEFORE the consumers' Record (src/native/Render.cpp), so the depth holds terrain, map objects and
// models but not what the consumers draw (the foliage must not occlude itself), one compute pass
// (src/shaders/glsl/depthcopy.comp) copies the depth, as it is, into one of two R32F images,
// alternating by frame, with the frame's camera and viewport beside it. The next frame's Prepare
// reads the one written last. The stages read the depth later, after the consumers.
//
// Synchronisation: the copy is followed by a barrier making its writes visible to the compute,
// vertex and fragment stages of everything recorded or submitted later (the next frame's pre-pass
// included), and preceded by one that waits for those stages' reads of the image it overwrites --
// the one read two frames back. The images stay in GENERAL layout.
namespace wxlvk::native::render::depthhistory
{
    /// Whether any consumer asks for the copy.
    bool Wanted();

    /**
     * Records this frame's copy into `cmd` (outside a pass), from the scene depth in
     * SHADER_READ_ONLY_OPTIMAL (present::SceneDepth).
     */
    void Capture(VkCommandBuffer cmd, VkImageView depthView);

    /// A consumer asks for the copy (`wanted` non-zero) or stops asking. Counted: the copy runs
    /// while any consumer wants it.
    void Request(bool wanted);

    /**
     * The latest copy made before the frame being recorded, described for a consumer.
     *
     * @param WXL_VkPreviousDepth& out : filled up to out.structSize
     * @return bool any : false when no copy exists (none asked for yet, or none made)
     */
    bool Previous(WXL_VkPreviousDepth& out);

    /// Copies made since the start, for the stats line.
    uint64_t Copies();

    /// A consumer asks for THIS frame's copy (wxl.vk.render RequestOpaqueDepth) or stops asking. Counted
    /// with Request's askers: the one copy runs while either kind wants it.
    void RequestOpaque(bool wanted);

    /**
     * The copy made in the frame being recorded (wxl.vk.render GetOpaqueDepth): the same image the next frame's
     * Previous() hands over, from the MidPass at AfterOpaqueWorld on.
     *
     * @return bool made : false when this frame made none (yet)
     */
    bool Opaque(WXL_VkPreviousDepth& out);
}
