// The frame's figures for the rendering API's consumers (wxl.vk.render GetFrameStats) and the device's frame line.
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

// The CPU side is measured here, at each present (the interval since the previous one, the draws received and recorded
// in between); the GPU side by src/native/present/FrameGpu.cpp (the frame's submission and the device's shared work, from
// timestamps); slot 42's split by src/native/Perf.cpp (WXL_VK_STATS only). GetFrameStats composes the three. The frame
// line -- one INFO line, every kLinePresents presents while WXL_VK_STATS is on, and once at the session's end whatever
// it says -- is what a before / after comparison reads: "wxl-vulkan-api: frame: cpu A ms (max M), gpu G ms (max X) ...".
namespace wxlvk::native::render::framestats
{
    /// Once per present (src/native/Stats.cpp OnPresent).
    void OnPresent(uint64_t presents);

    /// wxl.vk.render GetFrameStats: filled up to out.structSize. False before the first present.
    bool Get(WXL_VkFrameStats& out);

    /// The frame line (INFO).
    void Log(const char* when);
}
