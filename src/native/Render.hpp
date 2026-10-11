// The rendering API the native device publishes (wxl/VkRenderApi.h): insertion points, resources,
// raw device access and draw filters for other extensions.
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

// The device knows the engine's frame (it is the engine's device), its consumers do not: the points
// are named for what the frame has drawn when they fire, and are reached from engine seams here.
// With no pass registered at a point and no draw filter, the device behaves as without this module:
// the seam's detour calls through and the draw path tests one counter.
namespace wxlvk::native::render
{
    namespace detail
    {
        /// Live draw filters; read by the draw path on every draw, so it stays a plain counter.
        extern uint32_t filterCount;

        /// Forgets what FilterDraw() remembered about the last bound shaders: the native device
        /// calls it when a CGxShader is created or destroyed, since the memo is keyed on the
        /// shader's address (src/native/Shaders.cpp).
        void ForgetFilterCache();
    }

    /**
     * Publishes wxl.vk.render and attaches the engine seams the insertion points fire from. Called
     * from WXL_Load when the native device is wanted, before the core enables the extensions' hooks.
     */
    void Install();

    /// Whether the late liquid pass's seam is attached (AfterLiquids can fire): RegisterStage refuses that
    /// point when it is not, so a consumer falls back to AfterOpaqueWorld.
    bool AfterLiquidsLive();

    /// Whether any draw filter is live: the draw path's only cost while none is.
    inline bool AnyDrawFilter() { return detail::filterCount != 0; }

    /**
     * Whether a filter suppresses the draw about to be made: the bound vertex and pixel shaders'
     * names (render states 77, 78) against the live filters.
     *
     * @param uintptr table : the device's render-state array (kRsTable's target)
     * @return bool suppressed : the draw is dropped (counted against the matching filter)
     */
    bool FilterDraw(uintptr_t table);

    /**
     * The scene as the engine's world sees it now -- the block the insertion points hand their
     * passes -- and the viewport a draw recorded now would get (D3D9's pixel centres included, and
     * TAA's jitter when it applies now), for the post-process chain's reconstruction from depth.
     *
     * @param float jitter[2] : receives the jitter `viewport` carries (0, 0 when none was applied),
     *        measured against the same viewport made without it
     * @return bool ok : false without the engine's device
     */
    bool SceneNow(VkExtent2D extent, WXL_VkSceneConstants& scene, VkViewport& viewport, float jitter[2]);

    /**
     * The SHADOW_CASTERS passes' Records, inside a shadow cascade's pass (src/capture/Replay.cpp):
     * their context carries the light's view-projection as the scene's viewProj and projection
     * (view identity), the cascade's viewport and the shadow pass class.
     *
     * @return uint32 called : the passes called
     */
    uint32_t RecordShadowCasters(VkCommandBuffer cmd, VkRenderPass pass, const VkViewport& viewport, const VkRect2D& area,
                                 const float lightViewProj[16], uint32_t cascade);

    /// The stats lines: points reached, passes called, filters' counts, resources made.
    void LogStats();
}
