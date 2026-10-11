// The client's own "Desaturate" UI pixel shader, re-expressed for Vulkan.
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

// Shaders\Pixel\ps_3_0\Desaturate.bls, one permutation, 128 bytes; the image-mode shader
// CSimpleTexture::Init picks for a desaturated UI texture, paired with UI:0 on format 8 like "UI".
// Its reconstruction (a reconstructed HLSL source (not part of this repository)) compiles back to the
// shipped token stream byte for byte.
//
// The shipped bytecode, whole:
//
//   def c0, 0.299, 0.587, 0.114, 0
//   dcl_color v0.w                    only the vertex ALPHA is declared; the vertex rgb is unread
//   dcl_texcoord v1.xy
//   texld r0, v1, s0
//   dp3 r0.x, r0, c0                  luminance of the texel, Rec. 601 weights
//   mul r0.y, r0.w, v0.w              texel alpha * vertex alpha
//   mov oC0, r0.xxxy
//
// So the vertex colour does not tint the result at all -- a desaturated texture drawn with a red
// vertex colour comes out grey, not dark red -- and the alpha is the same product the UI shader
// makes. c0 is def'd in the bytecode and is not part of the contract; nothing is uploaded, which is
// why this shader, like ui.frag, reads no constant block.
layout(set = 0, binding = 0) uniform sampler2D tex;

layout(location = 0) in vec4 vColor;
layout(location = 1) in vec2 vUV;

layout(location = 0) out vec4 outColor;

void main()
{
    vec4  t = texture(tex, vUV);
    float l = dot(t.rgb, vec3(0.299, 0.587, 0.114));
    outColor = vec4(l, l, l, t.a * vColor.a);
}
