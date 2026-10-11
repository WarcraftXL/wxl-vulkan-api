// One engine draw, read off the device at slot 42 and ready to be issued through Vulkan.
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

#include "common/Simd.hpp"
#include "vulkan/Texture.hpp"
#include "vulkan/Vk.hpp"

#include <cstddef>
#include <cstdint>

// The boundary between the two halves of the draw path. src/draw/Reader.cpp fills one of these per
// engine draw by reading the CGxDevice once the state sync has run; src/vulkan/Pipelines.cpp
// consumes one by binding everything it names and issuing the draw. Neither side includes the
// other: this header is the whole contract, and anything a draw needs that is not in here is a
// change to this file first.
//
// Everything is already resolved to Vulkan objects or plain numbers. No engine pointer crosses the
// boundary except `vs`/`ps`, which are the shader *names* the engine interned ("UI:0", "UI") and
// which the pipeline side uses only as a key, and `vertexConstHigh`, which points into the native
// device's register file and is read before the slot returns.
namespace wxlvk::draw
{
    /// What one draw binds and how it is issued. Plain data; memcpy-able; zero is "nothing".
    ///
    /// No field has a default member initialiser: a `Cmd` is built on the stack for every draw and
    /// the reader resets it itself (Reset(), below) with sixteen-byte stores, so that no compiler-
    /// generated field-by-field construction or `rep stos` runs on the draw path. The defaults a
    /// field documents ("3", "true", "0xF" ...) are what Reset() writes.
    struct Cmd
    {
        // --- geometry: the pool buffers, bound at offset 0, addressed the way D3D does ------------
        VkBuffer     vertexBuffer;            ///< VK_NULL_HANDLE: unresolved
        uint32_t     vertexStride;            ///< bytes per vertex (24 for the UI's PCT)
        int32_t      vertexOffset;            ///< D3D BaseVertexIndex: buf.offset / buf.stride
        VkBuffer     indexBuffer;
        uint32_t     firstIndex;              ///< (ib.offset >> 1) + batch.start, in 16-bit indices
        uint32_t     indexBase;               ///< ib.offset >> 1: where the index buf starts in its pool
        uint32_t     indexCount;              ///< batch.count
        uint32_t     topology;                ///< EGxPrim: 0 points, 1 lines, 2 strip, 3 tris, 4 tristrip, 5 fan; 3 by default
        bool         indexed;                 ///< true by default

        // --- the material -------------------------------------------------------------------
        //
        // The names are bounded at kNameChars: the longest the client uses is
        // "Combiners_Opaque_AddAlpha_Alpha:15" (34), and a truncated name would be a key the
        // translated registry (src/shaders/Translated.cpp) could not resolve.
        static constexpr uint32_t kNameChars = 40;
        uint32_t     vertexFormat;            ///< EGxVertexBufferFormat (8 = PCT), 14 = custom

        // --- the custom vertex layout (vertexFormat == kVertexFormatCustom) ----------------------
        //
        // The engine has no fifteenth format: for a layout outside its fourteen, GxPrimVertexPtr's
        // callers fill the device's attribute slots directly (cgxdevice 3.4: +0x2790 + a*0x10 =
        // {attrib, dataFormat, offset, stride} for attribute a, +0x28A8 the enabled mask) and write
        // 14 at +0x28B0, and the D3D backend builds a D3DVERTEXELEMENT9 list from the enabled slots
        // at draw time (IStateSyncVertexPtrs, "one stream per distinct CGxBuf"). The reader copies
        // the enabled slots here, in the engine's own encodings -- EGxVertexAttrib (3.5: 0 position,
        // 1 blend weight, 2 blend indices, 3 normal, 4 colour 0, 5 colour 1, 6 + n texcoord n) and
        // the attribute data format (3.5: 0 D3DCOLOR, 1 UBYTE4, 2 UBYTE4N, 3 FLOAT2, 4 FLOAT3, 5
        // SHORT2, 6 FLOAT1) -- and the pipeline side turns them into the vertex input and keys the
        // pipeline on them, since two custom layouts share the format number and nothing else. Eight
        // elements: the client's own custom layouts are the full-screen-effect quads (position,
        // colour, up to four texcoord sets); a layout with more is refused rather than truncated.
        static constexpr uint32_t kVertexFormatCustom = 14;
        static constexpr uint32_t kMaxCustomElements  = 8;
        struct VertexElement
        {
            uint8_t attrib;   ///< EGxVertexAttrib 0..13
            uint8_t format;   ///< the attribute data format 0..6
            uint8_t offset;   ///< bytes from the vertex's start (below 256 for any layout that fits a stride)
            uint8_t stream;   ///< 0: vertexBuffer; n: extraBuffer[n - 1]
        };
        uint32_t      customCount;                      ///< 0 unless vertexFormat == kVertexFormatCustom
        VertexElement custom[kMaxCustomElements];       ///< in attribute order, which is the slots' order

        // --- the multi-stream custom layout --------------------------------------------------------
        //
        // D3D's IStateSyncVertexPtrs gives every distinct CGxBuf of a custom layout its own stream
        // (cgxdevice 3.4). With caps +0x224 at 0 -- the stream-offset mode D3D reports and the native
        // caps keep -- every stream is set at byte offset 0 and the draw's BaseVertexIndex (attribute
        // 0's buf.offset / its stride) addresses all of them alike (cgxdevice 7.2, step 4). So an
        // extra stream is its pool's buffer bound at offset 0 with its own buf's stride, nothing more.
        static constexpr uint32_t kMaxStreams = 4;
        uint32_t     extraStreams;                               ///< streams beyond vertexBuffer
        VkBuffer     extraBuffer[kMaxStreams - 1];               ///< stream n + 1's pool buffer, bound at offset 0
        uint32_t     extraStride[kMaxStreams - 1];               ///< stream n + 1's buf item size
        /// rs(77)->name and rs(78)->name, "" for fixed-function vertex processing / the
        /// fixed-function cascade. Always NUL-padded to the full kNameChars: the pipeline key
        /// copies the whole array and compares keys byte for byte.
        char         vs[kNameChars];
        char         ps[kNameChars];

        /// The 16 texture stages, kRsTexture0 + stage, each resolved to its image (null when the
        /// stage has no texture) and its CGxTex+0x2C flags (bits 0-2 filter, 3 wrapU, 4 wrapV, 9-13
        /// aniso). The hand-written pairs read stage 0; a translated pixel module reads the stages
        /// its manifest row's `sampler` lines name -- D3D sampler slot s<N> is texture stage N -- so
        /// the pipeline side writes one descriptor per declared slot from this array.
        static constexpr uint32_t kTextureStages = 16;
        const wxlvk::vk::texture::Image* texture[kTextureStages];
        uint32_t     texFlags[kTextureStages];
        uint32_t     engineTexMask;           ///< bit N: the engine had a texture bound on stage N, resolved to an image or not (the skip census)

        // --- fixed state, the engine's own encodings --------------------------------------------
        uint32_t     blend;                   ///< EGxBlend 0..13
        uint32_t     alphaRef;                ///< 0..255; below 1 the test is off
        bool         depthTest;
        uint32_t     depthFunc;               ///< 0 LE, 1 EQ, 2 GE, 3 LT
        bool         depthWrite;
        uint32_t     cull;                    ///< 0 none, 1 CW, 2 CCW
        uint32_t     colorWrite;              ///< engine mask: bit0 R, bit1 B, bit2 G, bit3 A; 0xF by default
        /// Render state 0 as D3D sends it, D3DRS_DEPTHBIAS = -value in depth-range units, when caps
        /// +0x304 offers it; and render state 84 as D3D sends it, BLENDFACTOR = round(v * 255) in all
        /// four channels, here / 255, read by blend mode 11 only.
        float        depthBias;
        float        blendFactor;

        // --- constants: the register files, as far as the shaders read them ----------------------
        //
        // The vertex rows c0..c63 and the pixel rows c0..c15 at the moment of the draw. 16 pixel rows
        // cover every pixel shader (the engine uploads c0..c12, c13/c14 are def'd), and 64 vertex rows
        // cover every vertex shader that does not index the bone palette from c31 or read the shadow
        // matrices at c224..c235. For those, `vertexConstHigh` points at rows 64..255.
        //
        // The pixel rows are copied into `pixelConst` (the reader writes the fog row over them). The
        // vertex rows are read through `vertexRows`: the native device's register file itself for a
        // shader draw, `vertexConst` -- a copy with the fixed-function file written over it -- for a
        // draw with no vertex shader. Both arrays sit at the end of the struct, so the reader resets
        // a Cmd without touching them (draw::BuildCmd).
        //
        // One fixed layout for every shader rather than a per-shader packing, so the pipeline side
        // writes one uniform slice per draw without knowing which shader it is for -- which is what
        // the translated shaders want: `regs.v[N]` is the engine's `cN`.
        static constexpr uint32_t kVertexRegs     = 64;
        static constexpr uint32_t kVertexRegsHigh = 256; ///< the whole file: the 256-row slice class
        static constexpr uint32_t kPixelRegs      = 16;
        /// The one pixel row the reader writes over the registers, for EVERY draw: (FogColor.rgb /
        /// 255, 1 when the master fog bit and render state 12 are both on, else 0). Safe because no
        /// shipped pixel module reads p[13..15] -- the engine uploads c0..c12 at most and c13/c14 are
        /// def'd literals -- and the hand-written fragment shaders read p[] not at all. Its reader is
        /// the generated `fixed_fog` ff companion of the weather vertex modules, which blends
        /// mix(p[15].rgb, rgb, fog0) while p[15].w > 0.
        static constexpr uint32_t kFogRow         = 15;
        static_assert(kFogRow < kPixelRegs, "the fog row must be inside the pixel window");
        /// The other row the reader writes for EVERY draw: p[14].x is 1 for a UNIT's draw -- a skinned model
        /// (its vertex module takes bone weights) wearing a character's, a creature's or an item's texture,
        /// or a composited, nameless skin (vulkan/Wind.cpp UnitModel, UnnamedTexture) -- and 0 otherwise.
        /// The G-buffer pixel variants (tools/gbuffer) copy it into the class byte's bit 6
        /// (WXL_VK_GBUFFER_MOVER_BIT), so a lighting consumer tells a unit from the world's models (no
        /// snow on a creature); the shadow capture takes its movers from it. Safe as the fog row is.
        /// p[14].y is 1 for a batch of an interior map object group (MOGP 0x2000 without 0x40: lit by the
        /// engine's interior light; native/render/WmoGroups.cpp): the variants copy it into the class byte's
        /// bit 5 (WXL_VK_GBUFFER_INTERIOR_BIT).
        static constexpr uint32_t kMoverRow       = 14;
        static_assert(kMoverRow < kFogRow, "the mover row precedes the fog row");
        /// The 64 vertex rows the slice carries: the register file, or `vertexConst`. Read before the
        /// slot returns, like `vertexConstHigh`.
        const float (*vertexRows)[4];

        /// The uniform slice class this draw's vertex shader needs: kVertexRegs (64) or
        /// kVertexRegsHigh (256), from the manifest record the native device filed on the bound
        /// vertex CGxShader when it created it (src/native/Shaders.cpp). 64 for every hand-written
        /// pair, the fixed function and an unknown name. The pipeline side writes this many rows
        /// of regs.v[].
        uint32_t     vconst;
        /// Rows 64..255 of the vertex register file when `vconst` is 256, null otherwise.
        const float (*vertexConstHigh)[4];
        /// How many of the 192 rows behind `vertexConstHigh` the ring slice needs: the rows above c63
        /// the engine has ever written (the rest of D3D's register file is still its initial zero).
        /// All 192 by default.
        uint32_t     highRows;

        /// The bone-palette path of a 256-row draw (a module's `bones` variant, which reads every
        /// row past c63 from a storage buffer): the register file from c31 (`paletteRows`), how many
        /// of its rows the engine has ever written (`paletteCount`), and a serial that changes
        /// whenever a constant flush wrote any row from c31 up -- the pipeline side uploads the
        /// rows only when the serial moved. Null / 0 for every other draw.
        static constexpr uint32_t kPaletteBase = 31;
        const float (*paletteRows)[4];
        uint32_t     paletteCount;
        uint64_t     paletteSerial;

        // --- fixed function: vs == "" (the register file) ------------------------------------
        //
        // The D3D backend draws such a Cmd with the device's own transforms, material and
        // texture-stage states (cgxdevice 5.4, 6.1-6.8), none of which goes through a shader
        // constant -- so for these draws the registers above hold a stale copy of whatever the last
        // shader draw uploaded. The reader therefore writes a register file of its own into
        // vertexConst at the rows named here, and the three selectors below travel in the slice's
        // misc row as (alphaRef / 255, ffColorMode, ffColorOp, ffAlphaOp). The pipeline side still
        // copies one slice per draw without knowing the shader; only ff.vert / ff.frag read these
        // rows, and a shader-driven draw leaves the selectors at their defaults.
        //
        // The other null, vs != "" with ps == "" (the vs_2_0 weather records, drawn through the
        // fixed-function PIXEL cascade): the vertex registers are live -- rows 0..21 are the module's
        // own c0..c21 -- so the reader writes NONE of the rows below and fills only what the generated
        // ff companion consumes: ffColorOp / ffAlphaOp (misc.zw, states 37 / 45 or kFfOpNone), the
        // alpha reference (misc.x) and kFogRow. `fixedFunction` is false for them: it means the rows
        // were written.
        static constexpr uint32_t kFfWorldView  = 0;  ///< rows 0..3: World * View, row-major, as the engine composes it
        static constexpr uint32_t kFfProjection = 4;  ///< rows 4..7: ProjNative, the projection in the API's depth range
        static constexpr uint32_t kFfColor      = 8;  ///< the constant colour ffColorMode selects, r g b a
        static constexpr uint32_t kFfFog        = 9;  ///< (scale, bias, 0, 0): fog = z_view * scale + bias; (0, 1) when off
        static constexpr uint32_t kFfFogColor   = 10; ///< the fog colour, r g b

        // The second texture stage and the lit path (cgxdevice 6.1, 6.3, 6.4). Rows kFfNormalXf.. are
        // written only when ffColorMode is kFfColorLit (or a texgen needs the normal); the rest of the
        // file is written for every fixed-function draw, so a row's content never depends on the format.
        static constexpr uint32_t kFfStage1     = 11; ///< (ColorOp1, AlphaOp1, stage-1 UV set, stage-0 UV set): states 38/46 or kFfOpNone, states 70/69; ff.vert applies the sets from kFfTexRoute
        static constexpr uint32_t kFfNormalXf   = 12; ///< rows 12..14: the inverse transpose of World * View's 3x3, rows as the engine composes them: n' = n * M; row 12's .w = NormalizeNormals (state 5). Also written, unlit, when a stage-0/1 TexGen needs the camera-space normal
        static constexpr uint32_t kFfMatDiffuse = 15; ///< material diffuse = material ambient (state 1), r g b a
        static constexpr uint32_t kFfMatEmissive = 16; ///< material emissive (state 2) r g b, .a = specular power (state 4)
        static constexpr uint32_t kFfMatSpecular = 17; ///< material specular (state 3) r g b, .a = 1 when SPECULARENABLE (power > 0 and specular != 0)
        static constexpr uint32_t kFfLightCtl   = 18; ///< (colour-source row of 6.3's table: 0 all material, 1 ambient+diffuse from Color0, 2 specular from Color0, 3 emissive from Color0; NormalizeNormals; enabled light count; 0)
        static constexpr uint32_t kFfLight0     = 19; ///< rows 19..34: the enabled light slots, packed first, kFfLightRows each
        static constexpr uint32_t kFfLightRows  = 4;  ///< per light: (position or direction in VIEW space, w 1 point / 0 directional), (ambient rgb, attenuation0), (diffuse rgb, attenuation1), (specular rgb, attenuation2)
        static constexpr uint32_t kFfLightCount = 4;  ///< the device's four slots (gxdev::kLightCount)
        static_assert(kFfLight0 + kFfLightRows * kFfLightCount <= kVertexRegs, "the fixed-function file must fit the vertex window");

        // The rows the fixed-function vertex stage reads when it feeds a TRANSLATED pixel module
        // (vs == "", ps a known module: the full-screen FFX passes). ff.vert under
        // WXL_FF_SEMANTIC_VARYINGS writes the translated varying layout -- color0 at 0, texcoord 0..7
        // at 2..9, fog0 at 10 -- and routes the vertex's texcoord sets to the outputs the way D3D's
        // fixed-function vertex pipeline does: output texcoord n carries input set TEXCOORDINDEX[n]
        // (render states 69 + n, the low 16 bits; a texgen in the high bits is not run). Written for
        // every fixed-function draw, like the rest of the file.
        static constexpr uint32_t kFfTexRoute    = 35; ///< rows 35..36: eight floats, the input texcoord set each output texcoord 0..7 reads
        static constexpr uint32_t kFfSemanticCtl = 37; ///< (bit n set when the layout carries texcoord set n, as a float; texcoord offset u; texcoord offset v; 1 when the layout carries colour 0). The offsets are 0: the native viewport sits on D3D9's pixel centres instead
        static_assert(kFfSemanticCtl < kVertexRegs, "the fixed-function file must fit the vertex window");

        // Texture coordinate generation and texture transforms on stages 0 and 1 (cgxdevice 5.4
        // IXformSetTex, 6.2 ISetTexGen). The D3D backend sends, per stage, ONE texture matrix
        // (D3DTS_TEXTURE0 + s) and its TEXTURETRANSFORMFLAGS, composed from the TexTransform state
        // (61 + s) and two of the device's matrix stacks -- the backend-internal texgen matrix (stack
        // 11 + s, written by ISetTexGen from the TexGen mode) and the application texture matrix
        // (stack s). The reader composes the same matrix from the same stack tops (identity flags
        // honoured) and delivers it here, with the stage's TexGen mode (53 + s: 0 the vertex set,
        // 1..3 the camera-space position -- the texgen matrix is where they differ -- 4 and 6 the
        // camera-space reflection vector, 5 the camera-space normal) and the flags in D3D's own
        // encoding, so that ff.vert can run D3D's rule: the input is padded to four components (1 in
        // the first missing one, 0 after), multiplied as a row vector, the first COUNT components
        // kept, and PROJECTED divides by the last of them in the fragment stage. Written for every
        // fixed-function draw (identity, mode 0, DISABLE when nothing is set).
        static constexpr uint32_t kFfTexXf0       = 38;  ///< rows 38..41: stage 0's texture matrix, row-major, as IXformSetTex composes it: t' = t * M
        static constexpr uint32_t kFfTexXf1       = 42;  ///< rows 42..45: stage 1's
        static constexpr uint32_t kFfTexXfRows    = 4;
        static constexpr uint32_t kFfTexXfStages  = 2;   ///< stages carried; a texgen or transform on a textured stage 2+ is not run
        static constexpr uint32_t kFfTexCtl       = 46;  ///< (stage-0 TexGen mode, stage-0 flags, stage-1 TexGen mode, stage-1 flags); flags = COUNT (0 DISABLE, 2, 3, 4) | kFfTtfProjected
        static constexpr uint32_t kFfTtfProjected = 0x100; ///< D3DTTFF_PROJECTED, as D3D encodes it
        static_assert(kFfTexXf0 + kFfTexXfRows * kFfTexXfStages <= kFfTexCtl, "the texture matrices precede their control row");
        static_assert(kFfTexCtl < kVertexRegs, "the fixed-function file must fit the vertex window");

        static constexpr uint32_t kFfColorVertex      = 0; ///< the vertex colour (opaque white when the format has none)
        static constexpr uint32_t kFfColorConstant    = 1; ///< v[kFfColor] whole
        static constexpr uint32_t kFfColorConstantRgb = 2; ///< v[kFfColor].rgb with the vertex alpha
        static constexpr uint32_t kFfColorLit         = 3; ///< D3D vertex lighting from the rows above (ff.vert under WXL_FF_LIT; a shader built without it reads v[kFfColor], which then holds the emissive)
        static constexpr uint32_t kFfOpNone           = 6; ///< EGxTexOp is 0..5; 6 = the stage has no texture, the cascade ends

        bool         fixedFunction;           ///< the vertex shader is null and the rows above were written (NOT set for vs != "" with ps == "")
        uint32_t     ffColorMode;             ///< kFfColor* : where stage 0's CURRENT comes from; kFfColorVertex by default
        uint32_t     ffColorOp;               ///< stage 0 ColorOp (state 37), or kFfOpNone (the default)
        uint32_t     ffAlphaOp;               ///< stage 0 AlphaOp (state 45), or kFfOpNone (the default)

        /// Stage 0's CGxTex as the engine bound it -- the one engine pointer besides the names, read
        /// by src/vulkan/Wind.cpp only (the texture's file path, to find tree models) and only while
        /// a consumer watches for them. Null when the stage has none.
        const void*  engineTex0;
        /// The vertex buf's own vertex count (CGxBuf +0x10): src/vulkan/Wind.cpp reads a tree model's
        /// whole buffer back, not only the range the batches it has seen reach.
        uint32_t     vertexItems;
        uintptr_t    vertexBufId;      ///< the vertex CGxBuf's address: an identity only, never dereferenced (tree wind's stale-key check)
        uint32_t     vertexBufBytes;   ///< the vertex buf's bytes (CGxBuf +0x14)
        uint32_t     vertexPoolBytes;  ///< its pool's, the VkBuffer's size (CGxPool +0x10)
        uint32_t     indexPoolBytes;   ///< the index pool's

        // --- placement on the current target ---------------------------------------------------
        VkViewport   viewport;                ///< D3D's integer viewport on the target, y-down, moved onto D3D9's pixel centres
        VkRect2D     scissor;                 ///< the viewport rect, intersected with the scissor when state 20 is on

        // --- the register copies (see "constants" above) -----------------------------------------
        //
        // Last: Reset() leaves them as they are, and the reader writes them before anything reads
        // them.
        float        pixelConst[kPixelRegs][4];   ///< written whole for every draw
        float        vertexConst[kVertexRegs][4]; ///< written only for a fixed-function draw, which points vertexRows here
    };

    /**
     * Every field to its default, except the two register arrays at the end, which the reader
     * writes before anything reads them: the head zeroed in sixteen-byte rows (the last row may
     * reach into pixelConst, written whole right after), then the few non-zero defaults.
     *
     * @param Cmd& c : the draw to reset
     */
    inline void Reset(Cmd& c)
    {
        constexpr size_t kHead = offsetof(Cmd, pixelConst);
        static_assert(kHead % 16 == 0 || sizeof(Cmd::pixelConst) >= 16, "the zeroing's spill lands in pixelConst");
        wxlvk::simd::ZeroRows(&c, (kHead + 15) / 16);
        c.topology    = 3;
        c.indexed     = true;
        c.colorWrite  = 0xF;
        c.vconst      = Cmd::kVertexRegs;
        c.highRows    = Cmd::kVertexRegsHigh - Cmd::kVertexRegs;
        c.ffColorMode = Cmd::kFfColorVertex;
        c.ffColorOp   = Cmd::kFfOpNone;
        c.ffAlphaOp   = Cmd::kFfOpNone;
    }
}
