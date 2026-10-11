// The client's own modulate-combiner pixel shader ("Combiners_Mod:4"), re-expressed for Vulkan.
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

// Shaders\Pixel\ps_3_0\Combiners_Mod.bls permutation 4 (shadow tier 0, group 1, no texkill;
// byte-identical to permutation 0). The pixel half of both Effect(Particle) and
// Effect(Particle_Unlit), after Color_T1:0 or CDiffuse_T1:1. Translated from
// a reconstructed HLSL source (not part of this repository), which fxc compiles back to the shipped token
// stream byte for byte.
//
// The shipped bytecode, whole:
//
//   texld r0, v1, s0
//   mul oC0.w, r0.w, v0.w             AlphaOp0(Mod): texture alpha * vertex alpha
//   mad r0.xyz, r0, v0, -c2           ColorOp0(Mod): texture * vertex colour, then vertex fog:
//   mad oC0.xyz, v2.x, r0, c2         fog * (rgb - fogColour) + fogColour = mix(fogColour, rgb, fog)
//
// The registers it reads (pixel target 4, slot 70): c2 = (fog.r, fog.g, fog.b, alphaRef) from
// CShaderEffect::SetFogParams 0x00873210, .xyz in R, G, B order. s0 is the particle texture (state 21).
//
// The alpha test is NOT in the engine's shader. This permutation has no texkill; the client runs the
// fixed-function test of render state 7 (AlphaRef) instead: D3D ALPHAFUNC GREATEREQUAL, pass when
// alpha * 255 >= AlphaRef, off when AlphaRef < 1 (cgxdevice 6.6; contract section 2c). The discard
// below reproduces that state from misc.x, which src/vulkan/Pipelines.cpp fills from Cmd::alphaRef
// as ref / 255 (0 when the test is off). c2.w is the engine's shadow of the reference, but in this
// mode SetAlphaRef writes the render state and leaves c2.w alone, so regs.p[2].w may be stale and is
// deliberately not used.
layout(set = 0, binding = 0) uniform sampler2D tex;

/// One slice of the uniform ring; the same block as the two particle vertex shaders.
layout(std140, set = 1, binding = 0) uniform Regs {
    vec4 p[16];   // pixel  c0..c15, exactly Cmd::pixelConst
    vec4 misc;      // misc.x = alphaRef / 255.0 (0 = test off); .yzw spare
    vec4 v[64];   // vertex c0..c63, exactly Cmd::vertexConst (the 64-row slice class)
} regs;

layout(location = 0) in vec4  vColor;   // COLOR0
layout(location = 1) in vec2  vUV;      // TEXCOORD0
layout(location = 2) in float vFog;     // FOG, 1 = unfogged

layout(location = 0) out vec4 outColor;

void main()
{
    vec4  t   = texture(tex, vUV);
    float a   = t.a * vColor.a;
    vec3  rgb = mix(regs.p[2].rgb, t.rgb * vColor.rgb, vFog);

    // Render state 7, as a discard: a * 255 < ref  <=>  a < ref / 255. Blending is the pipeline's.
    if (regs.misc.x > 0.0 && a < regs.misc.x) discard;

    outColor = vec4(rgb, a);
}
