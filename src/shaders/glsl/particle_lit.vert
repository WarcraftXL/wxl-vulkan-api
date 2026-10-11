// The client's own lit particle vertex shader ("CDiffuse_T1:1"), re-expressed for Vulkan.
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

// Shaders\Vertex\vs_3_0\CDiffuse_T1.bls permutation 1: lit, no local lights, no skinning, shadow
// tier 0. Effect(Particle) in shaders\Effects\Particle.wfx pairs it with Combiners_Mod;
// CParticleEmitter2::RenderParticles picks it for an emitter with the lit bit, with vertex format 4
// PNCT. Translated line for line from the reconstruction in
// 3.Tools\bls_tools\out\hlsl\CDiffuse_T1_p1.hlsl, which fxc compiles back to the shipped token
// stream byte for byte; the contract is 3.Tools\bls_tools\out\m2_login_shader_contracts.md.
//
// The registers it reads, and which engine setter fills each (all vertex target 0, slot 70):
//
//   c2..c5    transpose(ProjNative) rows          CShaderEffect::UpdateProjMatrix      0x00872C10
//   c6, c7    texture matrix unit 0, COLUMNS 0, 1 CShaderEffect::SetTexMtx_Identity   0x00873480
//   c10       .xyz directional light diffuse, each channel <= 1
//                                                 CShaderEffect::SetLocalLighting     0x00873CA0
//   c11       .xyz ambient colour                 same
//   c12       .xyz normalised light direction, VIEW space, pointing from the light INTO the scene
//                                                 same; built by CM2Lighting::SetupSunlight
//   c29       emissive, float4 in 0..1, (0,0,0,0) for a particle batch
//                                                 CShaderEffect::SetEmissive          0x00873A50
//   c30       fog (-1/(end-start), end/(end-start), exponent, 0); (0, 1, 1, 0) when fog is off
//                                                 CShaderEffect::SetFogEnabled        0x00873390
//   c31..c33  transpose(World*View) rows 0..2     CShaderEffect::UpdateWorldViewMatrix 0x00872B00
//
// c0 is def'd (1, 0, 0, 0) in the bytecode. The .w of c10, c11 and c12 overlap the next upload or
// are stale and are not read. The block below is the whole c0..c39 window, see particle_unlit.vert.
//
// Vertex layout PNCT, stride 36: position float3 at 0, normal float3 at 12, D3DCOLOR at 24
// (B,G,R,A in memory, undone by the B8G8R8A8_UNORM attribute), texcoord float2 at 28. The normal is
// a per-batch constant (row 2 of the emitter-to-camera matrix): lit particles are lit as if facing up.
layout(location = 0) in vec3 inPos;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec4 inColor;
layout(location = 3) in vec2 inUV;

/// One slice of the uniform ring; the same block as particle_unlit.vert and combiners_mod.frag.
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
    // dp4 r0.x, c31, v0 / dp4 r0.y, c32, v0 / mov r0.w, c0.x / dp4 r0.z, c33, v0.
    vec4 P = vec4(inPos, 1.0);
    vec4 p;
    p.x = dot(regs.v[31], P);
    p.y = dot(regs.v[32], P);
    p.w = 1.0;
    p.z = dot(regs.v[33], P);

    // dp4 o0.xyzw, c2..c5, r0.
    vec4 clip = vec4(dot(regs.v[2], p), dot(regs.v[3], p), dot(regs.v[4], p), dot(regs.v[5], p));

    // The engine's clip space is D3D's: +y is UP. Vulkan's is +y DOWN. A Vulkan 1.0 target cannot
    // flip the viewport (a negative height needs VK_KHR_maintenance1), so the flip lives here, on
    // the one value it concerns. z needs nothing: ProjNative already remaps depth into [0, 1] on
    // the D3D API id, which is also Vulkan's range.
    clip.y = -clip.y;
    gl_Position = clip;

    // mad / max / pow / min: the fog factor, both clamps kept, in the shipped order.
    float f = p.z * regs.v[30].x + regs.v[30].y;
    f = max(f, 0.0);
    f = pow(f, regs.v[30].z);
    vFog = min(f, 1.0);

    // dp3 r0.xyz, c31..c33, v1; nrm r1.xyz, r0: the normal through the World*View 3x3 as a row
    // vector, then normalised.
    vec3 n;
    n.x = dot(regs.v[31].xyz, inNormal);
    n.y = dot(regs.v[32].xyz, inNormal);
    n.z = dot(regs.v[33].xyz, inNormal);
    n = normalize(n);

    // dp3_sat r0.x, -c12, r1: c12 points INTO the scene, so the lambert term is against -c12,
    // saturated. mad_sat r0.xyz, r0.x, c10, c11: N.L * diffuse + ambient, saturated. mov r0.w, c0.x.
    // mad_sat o1, v2, r0, c29: vertex colour * lighting + emissive, saturated -- so alpha is
    // saturate(colour.a * 1 + emissive.a). No 2x scaling anywhere (that is the Mod2x family).
    float ndl = clamp(dot(-regs.v[12].xyz, n), 0.0, 1.0);
    vec4 lit;
    lit.xyz = clamp(ndl * regs.v[10].xyz + regs.v[11].xyz, 0.0, 1.0);
    lit.w   = 1.0;
    vColor  = clamp(inColor * lit + regs.v[29], 0.0, 1.0);

    // (u, v, 1) against the two texture-matrix columns c6.xyw / c7.xyw.
    vec3 t = vec3(inUV, 1.0);
    vUV = vec2(dot(regs.v[6].xyw, t), dot(regs.v[7].xyw, t));
}
