// The client's own UI vertex shader ("UI:0"), re-expressed for Vulkan.
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

// What the engine's bytecode does, read out of it: four dp4 against c0..c3, then a pass-through of
// the colour and the first texture coordinate. c0..c3 hold transpose(View * ProjNative) -- so c_i
// is column i of View*ProjNative under the engine's row-vector convention -- which is exactly why
// the four dots below are written out one by one and not as a mat4 multiply: a mat4 built from
// the four registers invites a transpose in one direction or the other, and the dots cannot.
//
// Vertex layout PCT, stride 24: position float3 at 0, D3DCOLOR at 12 (B,G,R,A in memory, which
// the pipeline's B8G8R8A8_UNORM attribute undoes), texcoord float2 at 16.
layout(location = 0) in vec3 inPos;
layout(location = 1) in vec4 inColor;
layout(location = 2) in vec2 inUV;

/// The four constant registers the engine's shader reads, pushed per draw (64 bytes).
layout(push_constant) uniform Consts { vec4 c[4]; } pc;

layout(location = 0) out vec4 vColor;
layout(location = 1) out vec2 vUV;

void main()
{
    vec4 p = vec4(inPos, 1.0);
    gl_Position = vec4(dot(pc.c[0], p), dot(pc.c[1], p), dot(pc.c[2], p), dot(pc.c[3], p));

    // The engine's clip space is D3D's: +y is UP. Vulkan's is +y DOWN. A Vulkan 1.0 target cannot
    // flip the viewport (a negative height needs VK_KHR_maintenance1), so the flip lives here, on
    // the one value it concerns. z needs nothing: ProjNative already remaps depth into [0, 1] on
    // the D3D API id, which is also Vulkan's range.
    gl_Position.y = -gl_Position.y;

    vColor = inColor;
    vUV    = inUV;
}
