// The client's own unlit particle vertex shader ("Color_T1:0"), re-expressed for Vulkan.
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

// Shaders\Vertex\vs_3_0\Color_T1.bls permutation 0 (byte-identical to CDiffuse_T1 permutation 0 and
// to Color_T1 2/4/6/8). Effect(Particle_Unlit) in shaders\Effects\Particle.wfx pairs it with
// Combiners_Mod; CParticleEmitter2::RenderParticles picks it for an emitter without the lit bit, with
// vertex format 8 PCT. Translated line for line from
// a reconstructed HLSL source (not part of this repository), which fxc compiles back to the shipped token stream
// byte for byte.
//
// The registers it reads, and which engine setter fills each (all vertex target 0, slot 70):
//
//   c2..c5    transpose(ProjNative) rows          CShaderEffect::UpdateProjMatrix      0x00872C10
//   c6, c7    texture matrix unit 0, COLUMNS 0, 1 CShaderEffect::SetTexMtx_Identity   0x00873480
//             c6 = (m00 m10 m20 m30), c7 = (m01 m11 m21 m31); identity for every particle batch
//   c30       fog (-1/(end-start), end/(end-start), exponent, 0)
//             (0, 1, 1, 0) when fog is off          CShaderEffect::SetFogEnabled        0x00873390
//   c31..c33  transpose(World*View) rows 0..2      CShaderEffect::UpdateWorldViewMatrix 0x00872B00
//             identity for a particle batch: the vertices were pre-multiplied on the CPU
//
// c0 is def'd (1, 0, 0, 0) in the bytecode and is not part of the contract. Nothing in c0..c39 that
// the shader does not read is touched; the whole window comes in as one uniform slice so that
// `regs.v[N]` is the engine's `cN`, and src/vulkan/Pipelines.cpp writes it without knowing which
// shader it is for.
//
// Vertex layout PCT, stride 24: position float3 at 0, D3DCOLOR at 12 (B,G,R,A in memory, which the
// pipeline's B8G8R8A8_UNORM attribute undoes), texcoord float2 at 16.
layout(location = 0) in vec3 inPos;
layout(location = 1) in vec4 inColor;
layout(location = 2) in vec2 inUV;

/// One slice of the uniform ring: the engine's pixel cache c0..c15, one row of backend state,
/// then its vertex cache c0..c63. Every shader that reads the ring declares exactly this block,
/// and the translated modules (docs/architecture.md) declare the same one with v[64] or
/// v[256] -- p and misc come FIRST so a pixel module reads them at the same offset whatever the
/// vertex module's class.
layout(std140, set = 1, binding = 0) uniform Regs {
    vec4 p[16];   // pixel  c0..c15, exactly Cmd::pixelConst
    vec4 misc;      // misc.x = alphaRef / 255.0 (0 = test off); .yzw spare
    vec4 v[64];   // vertex c0..c63, exactly Cmd::vertexConst (the 64-row slice class)
} regs;

layout(location = 0) out vec4  vColor;   // COLOR0    -> ps v0
layout(location = 1) out vec2  vUV;      // TEXCOORD0 -> ps v1.xy
layout(location = 2) out float vFog;     // FOG       -> ps v2.x, 1 = unfogged

void main()
{
    // dp4 r0.x, c31, v0 / dp4 r0.y, c32, v0 / mov r0.w, c0.x / dp4 r0.z, c33, v0: the position
    // through the three transposed World*View rows, w forced to 1. The dots are written out one by
    // one, as in ui.vert, so no mat4 is built and no transpose can creep in.
    vec4 P = vec4(inPos, 1.0);
    vec4 p;
    p.x = dot(regs.v[31], P);
    p.y = dot(regs.v[32], P);
    p.w = 1.0;
    p.z = dot(regs.v[33], P);

    // dp4 o0.xyzw, c2..c5, r0: the projection.
    vec4 clip = vec4(dot(regs.v[2], p), dot(regs.v[3], p), dot(regs.v[4], p), dot(regs.v[5], p));

    // The engine's clip space is D3D's: +y is UP. Vulkan's is +y DOWN. A Vulkan 1.0 target cannot
    // flip the viewport (a negative height needs VK_KHR_maintenance1), so the flip lives here, on
    // the one value it concerns. z needs nothing: ProjNative already remaps depth into [0, 1] on
    // the D3D API id, which is also Vulkan's range.
    clip.y = -clip.y;
    gl_Position = clip;

    // mad / max / pow / min: (end - z) / (end - start), clamped below at 0, raised to the fog
    // exponent, clamped above at 1. z is view depth, +z forward. Kept in the shipped order and
    // with both clamps: f is never negative when pow sees it, and fog-off (c30 = (0, 1, 1, 0))
    // gives exactly 1.
    float f = p.z * regs.v[30].x + regs.v[30].y;
    f = max(f, 0.0);
    f = pow(f, regs.v[30].z);
    vFog = min(f, 1.0);

    // mad r0.xyz, v2.xyxw, c0.xxyw, c0.yyxw -> (u, v, 1); dp3 against c6.xyww / c7.xyww: the two
    // texture-matrix columns, with the translation row taken from .w (the matrix is 4x4 and the
    // shader reads m00, m10, m30 / m01, m11, m31).
    vec3 t = vec3(inUV, 1.0);
    vUV = vec2(dot(regs.v[6].xyw, t), dot(regs.v[7].xyw, t));

    // mov o1, v1: the colour untouched -- no 2x, no saturate.
    vColor = inColor;
}
