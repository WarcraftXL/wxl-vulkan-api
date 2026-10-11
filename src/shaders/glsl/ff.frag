// The fixed-function texture cascade (render state 78 null), stage 0, re-expressed for Vulkan.
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

// What the D3D backend does for a draw with no pixel shader, from cgxdevice 6.1, 6.8 and the
// EGxTexOp table in 2.7: CURRENT starts as the interpolated diffuse; stage 0, if it has a texture,
// combines the texel with CURRENT by its ColorOp (state 37) and AlphaOp (state 45); the alpha test
// of render state 7; then vertex fog on rgb. Stage 0 is what the 2D paths that reach this shader
// use (6.7: the glue screen's quads, the cursor, text fallback: one stage, Mod/Mod); stage 1
// (states 38/46, its UV set from state 70) is run by the two-sampler build, which the rows for
// the two-UV layouts (formats 6, 9, and the textured format 7) use, since the M2 models'
// fixed-function path is where a second stage appears. Stage 1's selectors travel in v[11] = (ColorOp1, AlphaOp1, stage-1 UV
// set, stage-0 UV set), 6 again meaning "no texture, the cascade ended". Stages 2+ are not run.
//
// The texture coordinates arrive per STAGE, not per set: ff.vert has already routed
// the vertex set each stage reads (TEXCOORDINDEX) and run its texgen and texture transform, so
// vUV is stage 0's coordinate and vUV1 stage 1's, as vec4 with the components past the
// transform's COUNT zero. One piece of D3D's rule is left to this side because the rasteriser
// does it per pixel: TEXTURETRANSFORMFLAGS PROJECTED (v[46].y / .w bit 0x100) divides the
// coordinate by the last counted component after interpolation, as a perspective-correct
// projective lookup needs.
//
// Under WXL_FF_LIT the vertex shader's specular output (D3D's COLOR1) is added to the result of
// the cascade, which is where D3D adds it when SPECULARENABLE is on (before fog and the alpha
// test take their turn); ff.vert sends (0, 0, 0) when it is off.
//
// The op table, as the engine's s_colorOp / s_colorArgs set it (arg1, arg2 per op):
//
//   0 Mod      MODULATE (TEXTURE, CURRENT)             tex * cur
//   1 Mod2x    MODULATE2X                              2 * tex * cur
//   2 Add      ADD                                     tex + cur
//   3 PassThru SELECTARG2                              cur
//   4 Decal    BLENDCURRENTALPHA (CURRENT, TEXTURE)    cur * cur.a + tex * (1 - cur.a)
//   5 Fade     BLENDDIFFUSEALPHA (TEXTURE, CURRENT)    tex * diff.a + cur * (1 - diff.a)
//
// Colour and alpha use the same table; at stage 0 CURRENT and DIFFUSE are the same value, and D3D
// saturates each stage's result. The selectors arrive in misc.z / misc.w, 6 (draw::Cmd::kFfOpNone)
// meaning "stage 0 has no texture, the cascade ends before it".
//
// Compiled per sampler count and lit-ness (shared.cmake): WXL_FF_TEXTURED=1 is ff_frag, with the
// one sampler, for formats 8 and 10; WXL_FF_TEXTURED=0 is ff_color_frag for the UNTEXTURED
// format-7 draws (PC: no texcoord), which declares no sampler at all so that set 0's descriptor --
// which the pipeline side leaves unwritten for a row that does not sample -- is never statically
// used; WXL_FF_TEXTURED=2 declares s0 and s1 (ff_frag2 for format 9, and for the textured format-7
// draws -- the ground decals, both stages' coordinates from TexGen -- with ff_pc2_vert); WXL_FF_LIT=1 adds the specular
// input the lit vertex builds write (ff_lit_frag for format 4, ff_lit2_frag for format 6). A
// two-sampler build with an untextured stage 1 still has a valid binding 1: the pipeline side
// binds stage 0's image there, and v[11].x = 6 keeps the shader from reading it.
#ifndef WXL_FF_LIT
#   define WXL_FF_LIT 0
#endif
#if WXL_FF_TEXTURED >= 1
layout(set = 0, binding = 0) uniform sampler2D tex;
#endif
#if WXL_FF_TEXTURED >= 2
layout(set = 0, binding = 1) uniform sampler2D tex1;
#endif

/// One slice of the uniform ring; the same block as every ring shader.
layout(std140, set = 1, binding = 0) uniform Regs {
    vec4 p[16];   // pixel  c0..c15, exactly Cmd::pixelConst
    vec4 misc;      // .x alphaRef/255 (0 = test off), .y colour mode, .z stage-0 colour op, .w stage-0 alpha op
    vec4 v[64];   // vertex c0..c63, exactly Cmd::vertexConst (the 64-row slice class)
} regs;

layout(location = 0) in vec4  vColor;   // CURRENT at stage 0 = DIFFUSE
layout(location = 1) in vec4  vUV;      // stage 0's texture coordinate
layout(location = 2) in float vFog;     // 1 = unfogged
#if WXL_FF_TEXTURED >= 2
layout(location = 3) in vec4  vUV1;     // stage 1's texture coordinate
#endif
#if WXL_FF_LIT
layout(location = 4) in vec3  vSpecular;
#endif

layout(location = 0) out vec4 outColor;

#if WXL_FF_TEXTURED >= 1
/**
 * The 2D sampling coordinate of a stage: the interpolated texcoord, divided by its last counted
 * component when the stage's transform is PROJECTED (D3D's rasteriser-side divide). COUNT2
 * PROJECTED would divide u by v, which the engine never sends (its state 2 is COUNT3 |
 * PROJECTED); it is left undivided rather than guessed.
 *
 * @param vec4 t : the stage's coordinate from ff.vert
 * @param int flags : the stage's TEXTURETRANSFORMFLAGS from v[46]
 * @return vec2 uv : what the sampler reads
 */
vec2 StageUV(vec4 t, int flags)
{
    if ((flags & 0x100) != 0)
    {
        int   count = flags & 0xFF;
        float d     = count == 3 ? t.z : (count == 4 ? t.w : 0.0);
        if (d != 0.0) return t.xy / d;
    }
    return t.xy;
}

/// One EGxTexOp on all four channels; the caller keeps rgb from the colour op and a from the
/// alpha op. Decal blends by CURRENT's alpha, Fade by DIFFUSE's: the same value at stage 0, not
/// at stage 1, so both arrive.
vec4 Op(int op, vec4 t, vec4 cur, float diffA)
{
    if (op == 0) return t * cur;
    if (op == 1) return 2.0 * t * cur;
    if (op == 2) return t + cur;
    if (op == 3) return cur;
    if (op == 4) return cur * cur.a + t * (1.0 - cur.a);
    if (op == 5) return t * diffA + cur * (1.0 - diffA);
    return cur;        // an op outside the table: the stage is left out, as D3D leaves a DISABLE stage out
}
#endif

void main()
{
    vec4 cur = vColor;

#if WXL_FF_TEXTURED >= 1
    int cop = int(regs.misc.z + 0.5);
    int aop = int(regs.misc.w + 0.5);
    if (cop <= 5)
    {
        vec4 t = texture(tex, StageUV(vUV, int(regs.v[46].y + 0.5)));
        cur = clamp(vec4(Op(cop, t, cur, vColor.a).rgb, Op(aop, t, cur, vColor.a).a), 0.0, 1.0);
#if WXL_FF_TEXTURED >= 2
        int cop1 = int(regs.v[11].x + 0.5);
        int aop1 = int(regs.v[11].y + 0.5);
        if (cop1 <= 5)
        {
            vec4 t1  = texture(tex1, StageUV(vUV1, int(regs.v[46].w + 0.5)));
            cur = clamp(vec4(Op(cop1, t1, cur, vColor.a).rgb, Op(aop1, t1, cur, vColor.a).a), 0.0, 1.0);
        }
#endif
    }
#endif
#if WXL_FF_LIT
    // D3D's specular add, after the cascade and before fog; saturated like a stage result.
    cur.rgb = clamp(cur.rgb + vSpecular, 0.0, 1.0);
#endif

    // Render state 7, as combiners_mod.frag does it: a * 255 < ref  <=>  a < ref / 255.
    if (regs.misc.x > 0.0 && cur.a < regs.misc.x) discard;

    // Vertex fog after the stages, on rgb only (6.5). Fog off arrives as vFog = 1.
    cur.rgb = mix(regs.v[10].rgb, cur.rgb, vFog);

    outColor = cur;
}
