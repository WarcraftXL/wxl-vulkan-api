// Fixed-function vertex processing (render state 77 null), re-expressed for Vulkan.
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

// What the D3D backend does for a draw with no vertex shader, from cgxdevice 5.4 and 6.8: the
// position goes through the device's own matrices -- World top, View top, ProjNative -- which
// IStateSyncXforms sends as D3D transforms, not as shader constants; the colour is the vertex
// colour, the material, or lit (6.3/6.4); vertex fog is linear in view depth (6.5). The engine
// uploads NO constant for such a draw, so src/draw/Reader.cpp writes a register file of its own
// into the first rows of the slice (the layout is draw::Cmd's kFf* constants):
//
//   v[0..3]  World * View, the engine's row-vector product, rows as they are: pos' = pos * M
//   v[4..7]  ProjNative (device +0xFC8), the projection in D3D's depth range
//   v[8]     the constant colour, when misc.y says to use it
//   v[9]     fog as (scale, bias): f = z_view * scale + bias, (0, 1) when fog is off
//   v[10]    the fog colour (read by ff.frag)
//   v[11]    stage 1's ops (read by ff.frag); .zw the UV set per stage, which v[35] carries too
//   misc.y   colour mode: 0 vertex colour (white when the format has none), 1 v[8], 2 v[8].rgb with
//            the vertex alpha, 3 lit (WXL_FF_LIT builds only; the rows below)
//
// The lit rows (cgxdevice 6.3/6.4), written only under mode 3:
//
//   v[12..14] the inverse transpose of (World * View)'s 3x3, rows as the engine composes them: n' = n * M
//   v[15]     material diffuse r g b a, which is also the material AMBIENT (ISetMaterial, 6.3)
//   v[16]     material emissive r g b, .a = specular power
//   v[17]     material specular r g b, .a = 1 when SPECULARENABLE
//   v[18]     (colour-source row 0..3 of 6.3's table, NormalizeNormals, enabled light count, 0)
//   v[19..34] four lights of four rows: (position / direction in VIEW space, w 1 point 0 directional),
//             (ambient rgb, att0), (diffuse rgb, att1), (specular rgb, att2); the enabled ones first
//
// The texture-coordinate rows (cgxdevice 5.4 / 6.2), written for every draw:
//
//   v[35..36] eight floats: the vertex set output texcoord n reads (TEXCOORDINDEX, states 69..76)
//   v[38..41] stage 0's D3D texture matrix, as IXformSetTex composes it from the texgen and
//             application stacks; v[42..45] stage 1's. Row-major, t' = t * M.
//   v[46]     (stage-0 TexGen mode, stage-0 transform flags, stage-1 mode, stage-1 flags); the flags
//             are D3D's TEXTURETRANSFORMFLAGS: COUNT 0 (DISABLE) / 2 / 3 / 4, | 0x100 PROJECTED
//   v[12].w   NormalizeNormals (state 5), for the modes that read the camera-space normal; the
//             rows v[12..14] are then written whether or not the draw is lit
//
// What D3D's fixed-function vertex pipeline does with them, which StageCoord() below reproduces:
// the stage's input is the vertex set (mode 0, D3DTSS_TCI_PASSTHRU), the camera-space position
// (modes 1..3, D3DTSS_TCI_CAMERASPACEPOSITION; the texgen matrix is where the three differ), the
// camera-space reflection vector (modes 4 and 6, CAMERASPACEREFLECTIONVECTOR: reflect(normalize(e),
// n), the sphere map's 0.5 scale being in its matrix) or the camera-space normal (mode 5,
// CAMERASPACENORMAL). With the transform DISABLED the input goes out as it is. Otherwise the input
// is padded to four components with 1 in the FIRST missing one and 0 after -- (u, v) becomes
// (u, v, 1, 0), (x, y, z) becomes (x, y, z, 1) -- multiplied as a row vector by the matrix, and
// the first COUNT components are the output; PROJECTED divides them by the last of those, which
// ff.frag does per pixel, as the rasteriser does. The padding rule is Wine's reading of D3D in
// wined3d's compute_texture_matrix ("Direct3D passes the default 1.0 in the 2nd coord, while GL
// passes it in the 4th": for a two-float set it moves matrix row 3 into row 4 so GL's (u, v, 0, 1)
// meets D3D's (u, v, 1, 0)), and the engine confirms it on its side: for COUNT2 it copies the
// C44Matrix translation (row 3) into row 2, which is only needed if D3D reads a 2D translation
// from row 2 -- under a (u, v, 1, 1) padding that copy would double the translation.
//
// What D3D's fixed function computes from those (the "Mathematics of Lighting" the engine relies
// on; the engine leaves every D3D lighting default except the ones 6.4 names): in camera space,
//
//   diffuse  = saturate(E + A * sum(La_i * att_i) + D.rgb * sum(Ld_i * max(N.L_i, 0) * att_i)),  alpha = D.a
//   specular = saturate(S * sum(Ls_i * (N.H_i)^P * att_i))   when SPECULARENABLE, H = norm(L + norm(-pos))
//
// with E, A, D, S the material emissive, ambient, diffuse, specular after the colour-source table
// (one of them replaced by the vertex colour when it has one), the global AMBIENT 0 (6.4), att 1
// for a directional light and 1 / (a0 + a1 d + a2 d^2) for a point light within D3DLIGHT9.Range
// = 10000 (6.4), L the unit vector from the vertex TOWARD the light (a directional light's stored
// direction points the other way, as D3DLIGHT9.Direction does), N the transformed normal,
// normalised only when state 5 says so (D3D's NORMALIZENORMALS). The specular is a separate
// output: D3D adds it to the pixel after the texture cascade, so it travels as its own varying.
//
// The products are written out as row-vector sums rather than as a mat4 multiply for the reason
// ui.vert gives: a mat4 built from four registers invites a transpose, and the sums cannot.
//
// One source for every fixed-function layout a row draws, selected by WXL_FF_FORMAT at compile
// time (shared.cmake), because the transform is the same for all and only the attribute list
// differs. The locations follow src/vulkan/Pipelines.cpp's VertexLayoutFor, which numbers
// attributes in the order they appear in the vertex:
//
//   4  PNCT  stride 36: position @0 (0), normal float3 @12 (1), D3DCOLOR @24 (2), texcoord @28 (3)
//   6  PNCT2 stride 44: as 4, then texcoord1 float2 @36 (4)
//   7  PC    stride 16: position float3 @0 (0), D3DCOLOR @12 (1)
//   8  PCT   stride 24: position float3 @0 (0), D3DCOLOR @12 (1), texcoord float2 @16 (2)
//   9  PCT2  stride 32: as 8, then texcoord1 float2 @24 (3)
//   10 PT    stride 20: position float3 @0 (0), texcoord float2 @12 (1)
//
// WXL_FF_LIT is set for the formats with a normal (4, 6): the lit path needs one, and a build
// without the define keeps mode 3 off its switch so the normal-less layouts never read v[12..34].
//
// WXL_FF_SEMANTIC_VARYINGS (ff_sem_vert) is the build that pairs with a TRANSLATED pixel
// module instead of ff.frag: the engine's full-screen effects draw with no vertex shader and a
// ps_2_0 FFX* pixel shader, which reads t0..t3 and v0 -- the D3D fixed-function vertex pipeline
// feeding the programmable pixel stage. The inputs are then by semantic at the translated
// contract's locations (position0 0, color0 2, texcoord0 4, texcoord1 5, and texcoord2..7 at
// 8..13, this side's extension for the custom layouts), over any layout: the pipeline side
// describes every input, and v[37] (draw::Cmd::kFfSemanticCtl) says which the vertex really
// carries -- .x the texcoord sets as a bit mask, .w 1 when it has a colour -- so an input the
// layout lacks is never forwarded. The outputs are the translated varying layout
// (docs/architecture.md, "Shaders"): color0 at 0, texcoord0..7 at 2..9 as vec4 with zw = 0, fog0
// at 10. Output texcoord n carries input set v[35 + n/4][n%4] (kFfTexRoute: D3D's TEXCOORDINDEX
// per stage, states 69..76), zero when that set is missing, plus v[37].yz, a texcoord offset the
// reader leaves at zero (the native viewport sits on D3D9's pixel centres instead). Unlit: mode 3
// reads v[8], the emissive, as every build without WXL_FF_LIT.
#ifndef WXL_FF_LIT
#   define WXL_FF_LIT 0
#endif
#ifndef WXL_FF_SEMANTIC_VARYINGS
#   define WXL_FF_SEMANTIC_VARYINGS 0
#endif
#if WXL_FF_SEMANTIC_VARYINGS
#   define WXL_HAS_NORMAL 0
#   define WXL_HAS_COLOR 1
#   define WXL_COLOR_LOCATION 2
#   define WXL_HAS_UV    1
#   define WXL_UV_LOCATION 4
#   define WXL_HAS_UV1   1
#   define WXL_UV1_LOCATION 5
#elif WXL_FF_FORMAT == 4
#   define WXL_HAS_NORMAL 1
#   define WXL_HAS_COLOR 1
#   define WXL_COLOR_LOCATION 2
#   define WXL_HAS_UV    1
#   define WXL_UV_LOCATION 3
#   define WXL_HAS_UV1   0
#elif WXL_FF_FORMAT == 6
#   define WXL_HAS_NORMAL 1
#   define WXL_HAS_COLOR 1
#   define WXL_COLOR_LOCATION 2
#   define WXL_HAS_UV    1
#   define WXL_UV_LOCATION 3
#   define WXL_HAS_UV1   1
#   define WXL_UV1_LOCATION 4
#elif WXL_FF_FORMAT == 7
#   define WXL_HAS_NORMAL 0
#   define WXL_HAS_COLOR 1
#   define WXL_COLOR_LOCATION 1
#   define WXL_HAS_UV    0
#   define WXL_HAS_UV1   0
#elif WXL_FF_FORMAT == 8
#   define WXL_HAS_NORMAL 0
#   define WXL_HAS_COLOR 1
#   define WXL_COLOR_LOCATION 1
#   define WXL_HAS_UV    1
#   define WXL_UV_LOCATION 2
#   define WXL_HAS_UV1   0
#elif WXL_FF_FORMAT == 9
#   define WXL_HAS_NORMAL 0
#   define WXL_HAS_COLOR 1
#   define WXL_COLOR_LOCATION 1
#   define WXL_HAS_UV    1
#   define WXL_UV_LOCATION 2
#   define WXL_HAS_UV1   1
#   define WXL_UV1_LOCATION 3
#elif WXL_FF_FORMAT == 10
#   define WXL_HAS_NORMAL 0
#   define WXL_HAS_COLOR 0
#   define WXL_HAS_UV    1
#   define WXL_UV_LOCATION 1
#   define WXL_HAS_UV1   0
#else
#   error "WXL_FF_FORMAT must be 4, 6, 7, 8, 9 or 10"
#endif
#if WXL_FF_LIT && !WXL_HAS_NORMAL
#   error "WXL_FF_LIT needs a format with a normal"
#endif
// WXL_FF_STAGE1: the build writes stage 1's coordinate (vUV1) for the two-sampler ff.frag. On by
// default exactly when the layout has a second UV set; ff_pc2_vert turns it on for format 7 (PC),
// whose textured draws -- the ground decals: the selection circle, spell markers -- have no UV set
// at all and generate BOTH stages' coordinates (TexGen camera-space position through each stage's
// texture matrix). A stage that reads a set the layout lacks reads set 0's stand-in.
#ifndef WXL_FF_STAGE1
#   define WXL_FF_STAGE1 WXL_HAS_UV1
#endif

layout(location = 0) in vec3 inPos;
#if WXL_HAS_NORMAL
layout(location = 1) in vec3 inNormal;
#endif
#if WXL_HAS_COLOR
layout(location = WXL_COLOR_LOCATION) in vec4 inColor;
#endif
#if WXL_HAS_UV
layout(location = WXL_UV_LOCATION) in vec2 inUV;
#endif
#if WXL_HAS_UV1
layout(location = WXL_UV1_LOCATION) in vec2 inUV1;
#endif
#if WXL_FF_SEMANTIC_VARYINGS
layout(location = 8) in vec2 inUV27[6];  // texcoord sets 2..7 at 8..13
#endif

/// One slice of the uniform ring; the same block as every ring shader.
layout(std140, set = 1, binding = 0) uniform Regs {
    vec4 p[16];   // pixel  c0..c15, exactly Cmd::pixelConst
    vec4 misc;      // .x alphaRef/255, .y colour mode, .z stage-0 colour op, .w stage-0 alpha op
    vec4 v[64];   // vertex c0..c63, exactly Cmd::vertexConst (the 64-row slice class)
} regs;

#if WXL_FF_SEMANTIC_VARYINGS
// The translated varying layout, so any pixel module of the manifest can read these.
layout(location = 0)  out vec4  vColor;   // color0: the stage-0 CURRENT (diffuse)
layout(location = 2)  out vec4  vTex[8];  // texcoord0..7, .zw = 0
layout(location = 10) out float vFog;     // fog0: 1 = unfogged
#else
layout(location = 0) out vec4  vColor;   // the stage-0 CURRENT (diffuse)
layout(location = 1) out vec4  vUV;      // stage 0's texture coordinate, after its routing, texgen and transform; zero when the format has no set
layout(location = 2) out float vFog;     // 1 = unfogged
#if WXL_FF_STAGE1
layout(location = 3) out vec4  vUV1;     // stage 1's, for the two-sampler ff.frag
#endif
#if WXL_FF_LIT
layout(location = 4) out vec3  vSpecular; // D3D's COLOR1: added after the cascade by the WXL_FF_LIT ff.frag
#endif
#endif

/**
 * The camera-space normal for a texgen mode: the vertex normal through v[12..14], normalised
 * when state 5 says so (v[12].w), as D3D's NORMALIZENORMALS applies to the normal before any
 * use. Zero in a build without a normal input, which is D3D's reading of a layout without one.
 *
 * @return vec3 n : the camera-space normal
 */
vec3 CameraNormal()
{
#if WXL_HAS_NORMAL
    vec3 n = inNormal.x * regs.v[12].xyz + inNormal.y * regs.v[13].xyz + inNormal.z * regs.v[14].xyz;
    return regs.v[12].w > 0.5 ? normalize(n) : n;
#else
    return vec3(0.0);
#endif
}

/**
 * One stage's texture coordinate as D3D's fixed-function vertex pipeline forms it (the header's
 * rule): the texgen input, then the texture transform with D3D's padding and count.
 *
 * @param int stage : 0 or 1; selects v[46]'s pair and the matrix rows v[38 + 4 * stage ..]
 * @param vec2 uv : the vertex set the stage's TEXCOORDINDEX routed to it
 * @param vec3 e : the camera-space position
 * @return vec4 t : the output texcoord; the components past COUNT are zero, PROJECTED is left for
 *                  the fragment stage (ff.frag divides by the last counted component)
 */
vec4 StageCoord(int stage, vec2 uv, vec3 e)
{
    int gen   = int(regs.v[46][2 * stage] + 0.5);
    int flags = int(regs.v[46][2 * stage + 1] + 0.5);

    vec3 g;
    int  comps;
    if (gen == 0)      { g = vec3(uv, 0.0); comps = 2; }
    else if (gen == 5) { g = CameraNormal(); comps = 3; }
    else if (gen == 4 || gen == 6) { g = reflect(normalize(e), CameraNormal()); comps = 3; }
    else               { g = e; comps = 3; }

    int count = flags & 0xFF;
    if (count == 0) return vec4(g, 0.0);                        // DISABLE: the input as it is

    vec4 t = comps == 2 ? vec4(g.xy, 1.0, 0.0) : vec4(g, 1.0);  // D3D's padding
    int  b = 38 + 4 * stage;
    t = t.x * regs.v[b] + t.y * regs.v[b + 1] + t.z * regs.v[b + 2] + t.w * regs.v[b + 3];
    if (count < 4) t.w = 0.0;                                   // COUNTn: only n components go out
    if (count < 3) t.z = 0.0;
    return t;
}

#if WXL_FF_LIT
/**
 * D3D's fixed-function vertex lighting on the rows the reader wrote (the header's formula).
 *
 * @param vec3 pos : the view-space position
 * @param vec4 vin : the vertex colour (opaque white when the format has none)
 * @param out vec3 specular : receives the specular output
 * @return vec4 diffuse : the lit diffuse, alpha from the diffuse source
 */
vec4 Lit(vec3 pos, vec4 vin, out vec3 specular)
{
    vec3 n = inNormal.x * regs.v[12].xyz + inNormal.y * regs.v[13].xyz + inNormal.z * regs.v[14].xyz;
    if (regs.v[18].y > 0.5) n = normalize(n);

    // The colour-source table of 6.3: which material colour the vertex colour stands in for.
    int  src = int(regs.v[18].x + 0.5);
    vec4 D   = src == 1 ? vin : regs.v[15];
    vec3 A   = D.rgb;                                     // material ambient = material diffuse
    vec3 E   = src == 3 ? vin.rgb : regs.v[16].rgb;
    vec3 S   = src == 2 ? vin.rgb : regs.v[17].rgb;
    float P  = regs.v[16].a;
    bool specOn = regs.v[17].a > 0.5;

    vec3 amb = vec3(0.0), dif = vec3(0.0), spc = vec3(0.0);
    vec3 V = normalize(-pos);                             // LOCALVIEWER: the camera is at the origin
    int count = int(regs.v[18].z + 0.5);
    for (int i = 0; i < 4; ++i)
    {
        if (i >= count) break;
        vec4 L0 = regs.v[19 + 4 * i];
        vec4 L1 = regs.v[20 + 4 * i];
        vec4 L2 = regs.v[21 + 4 * i];
        vec4 L3 = regs.v[22 + 4 * i];
        vec3  L;
        float att = 1.0;
        if (L0.w == 1.0)
        {
            vec3  d    = L0.xyz - pos;
            float dist = length(d);
            if (dist > 10000.0 || dist == 0.0) continue; // past Range: no contribution at all
            L = d / dist;
            float den = L1.w + L2.w * dist + L3.w * dist * dist;
            att = den > 0.0 ? 1.0 / den : 1.0;
        }
        else
            L = -normalize(L0.xyz);
        amb += L1.rgb * att;
        float ndl = dot(n, L);
        if (ndl > 0.0)
        {
            dif += L2.rgb * ndl * att;
            if (specOn)
            {
                float ndh = dot(n, normalize(L + V));
                if (ndh > 0.0) spc += L3.rgb * pow(ndh, P) * att;
            }
        }
    }
    specular = specOn ? clamp(S * spc, 0.0, 1.0) : vec3(0.0);
    return vec4(clamp(E + A * amb + D.rgb * dif, 0.0, 1.0), D.a);
}
#endif

void main()
{
    // pos * (World * View): view-space position, w stays 1 for an affine product.
    vec4 P = vec4(inPos, 1.0);
    vec4 e = P.x * regs.v[0] + P.y * regs.v[1] + P.z * regs.v[2] + regs.v[3];

    // * ProjNative.
    vec4 clip = e.x * regs.v[4] + e.y * regs.v[5] + e.z * regs.v[6] + e.w * regs.v[7];

    // The engine's clip space is D3D's: +y is UP. Vulkan's is +y DOWN. A Vulkan 1.0 target cannot
    // flip the viewport (a negative height needs VK_KHR_maintenance1), so the flip lives here, on
    // the one value it concerns. z needs nothing: ProjNative already remaps depth into [0, 1] on
    // the D3D API id, which is also Vulkan's range.
    clip.y = -clip.y;
    gl_Position = clip;

#if WXL_FF_SEMANTIC_VARYINGS
    // The colour input is always declared; whether the vertex has one is v[37].w (the pipeline
    // side points a missing input at the vertex's first bytes, which must not be read as colour).
    vec4 vin = regs.v[37].w > 0.5 ? inColor : vec4(1.0);
#elif WXL_HAS_COLOR
    vec4 vin = inColor;
#else
    vec4 vin = vec4(1.0);   // D3D's default diffuse when the vertex carries none (6.4)
#endif
    int mode = int(regs.misc.y + 0.5);
#if WXL_FF_LIT
    if (mode == 3)
    {
        vec3 spec;
        vColor    = Lit(e.xyz, vin, spec);
        vSpecular = spec;
    }
    else
    {
        vColor    = mode == 0 ? vin : (mode == 1 ? regs.v[8] : vec4(regs.v[8].rgb, vin.a));
        vSpecular = vec3(0.0);
    }
#else
    // Mode 3 cannot reach a build without the define (the reader sets it only with a normal); the
    // constant branch is still the right reading for it, since v[8] then holds the emissive.
    vColor = mode == 0 ? vin : (mode == 1 ? regs.v[8] : vec4(regs.v[8].rgb, vin.a));
#endif

#if WXL_FF_SEMANTIC_VARYINGS
    // The vertex's texcoord sets, zero where v[37].x says the layout has none, then D3D's
    // routing: output n carries input set TEXCOORDINDEX[n] (v[35..36]), and the texcoord
    // offset (v[37].yz, zero).
    int  present = int(regs.v[37].x + 0.5);
    vec2 sets[8];
    sets[0] = (present & 1)   != 0 ? inUV       : vec2(0.0);
    sets[1] = (present & 2)   != 0 ? inUV1      : vec2(0.0);
    sets[2] = (present & 4)   != 0 ? inUV27[0]  : vec2(0.0);
    sets[3] = (present & 8)   != 0 ? inUV27[1]  : vec2(0.0);
    sets[4] = (present & 16)  != 0 ? inUV27[2]  : vec2(0.0);
    sets[5] = (present & 32)  != 0 ? inUV27[3]  : vec2(0.0);
    sets[6] = (present & 64)  != 0 ? inUV27[4]  : vec2(0.0);
    sets[7] = (present & 128) != 0 ? inUV27[5]  : vec2(0.0);
    // Outputs 0 and 1 then go through their stage's texgen and texture transform; the offset is
    // on the vertex set, before the transform. Outputs 2..7 have no matrix rows: a transform on
    // them is not run and they go out plain.
    vec2 halfPixel = regs.v[37].yz;
    for (int n = 0; n < 8; ++n)
    {
        int src = int(regs.v[35 + n / 4][n % 4] + 0.5);
        vec2 uv = (src >= 0 && src < 8 && (present & (1 << src)) != 0) ? sets[src] + halfPixel : vec2(0.0);
        vTex[n] = n < 2 ? StageCoord(n, uv, e.xyz) : vec4(uv, 0.0, 0.0);
    }
#else
    // D3D's routing (TEXCOORDINDEX, v[35].xy for stages 0 and 1): a set this layout lacks reads
    // set 0, the only one a one-UV build has to offer; then the stage's texgen and transform.
#if WXL_HAS_UV
    vec2 set0 = inUV;
#else
    vec2 set0 = vec2(0.0);
#endif
#if WXL_HAS_UV1
    vec2 set1 = inUV1;
#else
    vec2 set1 = set0;
#endif
    vUV = StageCoord(0, int(regs.v[35].x + 0.5) == 1 ? set1 : set0, e.xyz);
#if WXL_FF_STAGE1
    vUV1 = StageCoord(1, int(regs.v[35].y + 0.5) == 1 ? set1 : set0, e.xyz);
#endif
#endif

    // Linear vertex fog on the view-space depth (range fog is never enabled, 6.5), clamped the
    // way D3D clamps the per-vertex factor. Fog off is (scale 0, bias 1): exactly 1.
    vFog = clamp(e.z * regs.v[9].x + regs.v[9].y, 0.0, 1.0);
}
