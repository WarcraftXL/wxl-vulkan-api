// The core's ImGui overlay, drawn natively (src/native/Overlay.cpp): the vertex stage.
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
#version 450

// wxl/OverlayApi.h's vertex, 20 bytes: position in display pixels, uv, colour R8G8B8A8_UNORM.
layout(location = 0) in vec2 inPos;
layout(location = 1) in vec2 inUV;
layout(location = 2) in vec4 inColor;

/// Display pixels to clip space: scale (2 / size) and translate (-1 - pos * scale). The display's
/// +y is down, as Vulkan's clip space, so nothing is flipped.
layout(push_constant) uniform Consts { vec2 scale; vec2 translate; } pc;

layout(location = 0) out vec4 vColor;
layout(location = 1) out vec2 vUV;

void main()
{
    vColor      = inColor;
    vUV         = inUV;
    gl_Position = vec4(inPos * pc.scale + pc.translate, 0.0, 1.0);
}
