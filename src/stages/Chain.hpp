// The stages: consumers' compute over the frame (wxl.vk.render RegisterStage), run in order at their
// point of the engine's frame.
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

#include "stages/Frame.hpp"
#include "wxl/VkRenderApi.h"

#include <cstdint>

// A run, at one point of one frame, when at least one stage of that point is wanted:
//
//   1. the open render pass is ended (a pass break; the next draw resumes the pass with LOAD);
//   2. when a wanted stage writes colour, the back buffer is copied into the working image;
//   3. each wanted stage runs, in its order, on that image, timed;
//   4. when the image was copied in, it is copied back onto the back buffer.
//
// Once per point per frame. Nothing runs, and nothing is touched, when no stage of the point is
// wanted: the frame is then the same command for command as without the stages. The device has no
// effect of its own here: anti-aliasing, ambient occlusion and lighting are the consumers' stages.
namespace wxlvk::stages::chain
{
    /// The most stages there can be (the timing lanes past the points').
    constexpr uint32_t kMaxStages = 16;

    /**
     * Adds a stage at its point and order (lower runs first; an equal order goes after). At any time: a
     * frame never sees it half-added (everything runs on the render thread).
     *
     * @return void* stage : the handle for Remove / Stats, or null when the table is full
     */
    void* Add(const WXL_VkStageDesc& desc, const char* name);

    /// Stops running a stage (its slot in the chain is kept).
    void Remove(void* stage);

    /// A stage's figures (wxl.vk.render GetStageStats); false for an unknown handle.
    bool Stats(void* stage, WXL_VkStageStats& out);

    /// The world's scene, saved at AfterOpaqueWorld for AfterLiquids and BeforeUi (Run does it at that point).
    void NoteWorld();

    /// A point of the frame was reached (src/native/render/WorldPass.cpp, src/native/Render.cpp).
    void Run(Point point);

    /// The stats lines: runs, skips, GPU times, working memory.
    void LogStats();
}
