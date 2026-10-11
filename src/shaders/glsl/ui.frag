// The client's own UI pixel shader ("UI"), re-expressed for Vulkan.
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

// The engine's "UI" pixel shader is tex2D(s0, uv) * color and nothing else; so is this.
//
// No alpha test here yet. The UI draws with alphaRef 1 and blend 2 (SRCALPHA / INVSRCALPHA), so a
// texel the test would drop is one the blend already leaves invisible; the picture is the same
// either way. When a material that needs it arrives, the test is one `discard` on
// `color.a * 255.0 < alphaRef` behind a second push constant -- cheap, and a one-line change in
// both this file and Pipelines.cpp's push range.
layout(set = 0, binding = 0) uniform sampler2D tex;

layout(location = 0) in vec4 vColor;
layout(location = 1) in vec2 vUV;

layout(location = 0) out vec4 outColor;

void main()
{
    outColor = texture(tex, vUV) * vColor;
}
