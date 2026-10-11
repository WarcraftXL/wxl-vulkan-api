// What a run of a point's stages works on: the points, the working image, the frame's description.
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

namespace wxlvk::stages
{
    /// Where in the engine's frame the stages run (the wxl.vk.render stage points).
    enum class Point : uint8_t
    {
        /// The opaque world is drawn (terrain, map objects, opaque models, ground clutter, the
        /// consumers' passes); the transparent models, liquids, weather and interface are not. The
        /// scene depth is complete for what is opaque.
        AfterOpaqueWorld,
        /// The world is drawn, the engine's own post-processing (glow, screen effects) included; the
        /// world text and the interface are not.
        BeforeUi,
        /// (Appended 2026-10-11.) The engine's late liquid pass has returned: the water is drawn and its
        /// depth written; the weather and the transparent models over the water follow (with the camera in
        /// no liquid). The scene is the world's as AfterOpaqueWorld saw it.
        AfterLiquids,
        Count,
    };

    /// One image of a run: a storage image in GENERAL layout for the whole of the run.
    struct Image
    {
        VkImage     image  = VK_NULL_HANDLE;
        VkImageView view   = VK_NULL_HANDLE;
        VkFormat    format = VK_FORMAT_UNDEFINED;
    };

    /// The frame a run works on. Valid only during the run.
    struct Io
    {
        VkCommandBuffer cmd    = nullptr;
        uint32_t        slot   = 0;      ///< the frame-in-flight slot (0 .. pipelines::kFrameSlots - 1)
        uint64_t        serial = 0;      ///< the frame's serial: what resources it names are retired by
        VkExtent2D      extent = { 0, 0 };
        /// The frame: R8G8B8A8_UNORM storage holding the back buffer's bytes as they are -- with a BGRA
        /// back buffer, .r is blue (`swapRB`). Null at a point no wanted stage writes colour at.
        Image           frame;
        bool            swapRB = false;
        /// The scene depth (SHADER_READ_ONLY_OPTIMAL), when a wanted stage reads it; null when it cannot
        /// be had this frame.
        VkImageView     depthView   = VK_NULL_HANDLE;
        VkFormat        depthFormat = VK_FORMAT_UNDEFINED;
        /// The engine's scene at the point (camera, projection, fog), and the viewport a draw would get
        /// now: its rectangle (D3D9's pixel centres included) and its depth range, which the world draws
        /// in (0 .. 0.94, not 0 .. 1).
        WXL_VkSceneConstants scene = {};
        VkViewport           viewport = {};
        bool                 sceneValid = false;
        /// The sub-pixel jitter the world was drawn with this frame (src/native/render/Jitter.hpp), in
        /// pixels; `viewport` includes it. 0, 0 when the jitter is off.
        float                jitter[2] = { 0.0f, 0.0f };
    };
}
