// The shadow replay's depth-only render pass class.
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

// One depth attachment, no colour: CLEAR on load (the replay draws every caster again), STORE,
// UNDEFINED -> SHADER_READ_ONLY_OPTIMAL. Its external dependencies order it after the previous
// frames' fragment and compute reads of the same image, and its writes before the fragment and
// compute reads that follow -- so a consumer samples the map with no barrier of its own. Made on
// first use, kept for the device's life (pipelines are built against it).
namespace wxlvk::capture::shadowpass
{
    /// The pass, or VK_NULL_HANDLE when no format supports a sampled depth attachment.
    VkRenderPass Get();

    /// The same class with LOAD on load, from SHADER_READ_ONLY_OPTIMAL (WXL_VK_REPLAY_LOAD).
    VkRenderPass GetLoad();

    /// Its depth format: D32_SFLOAT where it can be sampled, else D16_UNORM.
    VkFormat Format();
}
