// The world scene pass's seam: where BeforeUi fires, where the capture and the jitter open and close.
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

// The core's hook point "Gx.WorldOnRender" (CGWorldFrame's render pass) wraps the world scene: the
// terrain, the models, the transparents, the liquids, the weather, the engine's glow and screen
// effects; its caller then draws the world text batch (names, combat text) and the interface follows.
// It runs once per world frame and never on the glue screens, which have no world. Around it:
//
//   entry    the jitter's next sample (src/native/render/Jitter.hpp) and the opaque world's draw
//            capture (src/capture/Capture.hpp) open;
//   return   both close, and the BeforeUi stages run (src/stages/Chain.hpp).
//
// AfterOpaqueWorld fires from the rendering API's own seam (src/native/Render.cpp), after its
// consumers' passes, so the depth a stage reads includes their geometry.
namespace wxlvk::native::render::worldpass
{
    /// Attaches the seam. From WXL_Load, when the native device is wanted; false (logged) when the hook
    /// point could not be attached: no BeforeUi stage, no capture, no jitter.
    bool Install();
}
