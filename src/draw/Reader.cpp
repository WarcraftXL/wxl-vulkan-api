// Reading a draw off the CGxDevice: geometry, material, fixed state, constants, fixed function.
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

#include "draw/Reader.hpp"

#include "common/Simd.hpp"
#include "device/Slots.hpp"
#include "draw/Cmd.hpp"
#include "native/render/WmoGroups.hpp"
#include "shaders/Translated.hpp"
#include "vulkan/Manifest.hpp"
#include "vulkan/Texture.hpp"
#include "vulkan/Vk.hpp"
#include "vulkan/Wind.hpp"

#include "wxl/game/Binding.hpp"
#include "wxl/offsets/engine/Gx.hpp"
#include "wxl/offsets/engine/GxDevice.hpp"

#include <intrin.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>

// --- how the D3D backend draws, read out of the binary --------------------------------------------
//
// CGxDeviceD3d::Draw 0x006A3620 is the whole path, and every read below is one it makes:
//
//   if (*(dev + kContext) == 0 || *(dev + kFramePoisoned) != 0) return;   -- the two gates
//   IStateSync()                      -- creates lazy shaders, flushes dirty states, uploads textures
//   vb = *(dev + kAttribBufs + 0*4)   -- attribute 0's CGxBuf; equals kVertexBuf on a fixed format
//   ib = *(dev + kIndexBuf)
//   if indexed:
//       DrawIndexedPrimitive(prim, vb.offset / vb.stride, batch.minVert, batch.maxVert - minVert + 1,
//                            ib.offset / 2 + batch.start, primitives(prim, batch.count))
//   else:
//       DrawPrimitive(prim, vb.offset / vb.stride, primitives(prim, batch.count))  -- start ignored
//
// So a vertex buf's offset is a BaseVertexIndex (D3D adds it to every index), and an index buf's
// offset is a 16-bit index count (the engine has no 32-bit index path: gxdev::kIndexSize). Stream
// locks place a buf at a stride-aligned offset, which is what makes the division exact.
//
// The render states are read off the app state array behind kRsTable -- kRenderStateCount entries of
// kRsStride bytes, value first -- because that is what the backend itself reads at this moment; the
// hardware shadow at kRsShadow is what it last SENT, which is the same thing but without the
// master-enable override applied, and the pipelines want the engine's own encoding, not D3D's.
namespace
{
    namespace gx       = wxl::offsets::engine::gx;
    namespace gxdev    = wxl::offsets::engine::gxdevice;
    namespace texture  = wxlvk::vk::texture;
    namespace manifest = wxlvk::vk::manifest;

    using wxlvk::draw::Cmd;
    using wxlvk::draw::Resolver;
    using wxlvk::draw::Stages;

    /// The vertex attribute whose buf is the draw's vertex stream: attribute 0, position. The
    /// backend reads kAttribBufs + 0, and on a fixed-format draw that is also kVertexBuf.
    constexpr unsigned kVertexStreamAttrib = 0;

    /// The texture stages a Cmd carries: all sixteen, kRsTexture0 + 0..15. The hand-written pairs
    /// read stage 0; a translated pixel module reads whichever D3D sampler slots its manifest row
    /// declares, and slot N is stage N.
    constexpr unsigned kTextureStages = Cmd::kTextureStages;
    static_assert(kTextureStages == 16, "the engine has sixteen texture stages");

    static_assert(Cmd::kVertexRegsHigh == gx::kVsConstRegisters, "the 256-row class is the whole register file");
    static_assert(sizeof(Cmd::vertexConst[0]) == 4 * sizeof(float), "one register is four floats");
    static_assert(Cmd::kPixelRegs <= gx::kVsConstRegisters, "the pixel file is the same size as the vertex one");

    /// The highest render-state id this file reads, checked against the engine's array length so a
    /// wrong id is a compile error and not a read past the table.
    constexpr unsigned kHighestRsRead = gxdev::kRsBlendFactor;
    static_assert(kHighestRsRead < gxdev::kRenderStateCount, "render state id outside the engine's array");
    static_assert(gxdev::kRsTexture0 + kTextureStages - 1 < gxdev::kRenderStateCount, "texture stage id out of range");

    /// A `wxl::game::At<Batch>` of the CGxBatch, so the copy is one memcpy and every later read is a
    /// named field. Pinned to the core's layout.
    struct Batch
    {
        uint32_t prim;
        uint32_t start;
        uint32_t count;
        uint16_t minVert;
        uint16_t maxVert;
    };
    static_assert(sizeof(Batch) == gxdev::kBatchSize, "CGxBatch is 0x10 bytes");
    static_assert(offsetof(Batch, prim) == gxdev::kBatchPrim, "CGxBatch::prim");
    static_assert(offsetof(Batch, start) == gxdev::kBatchStart, "CGxBatch::start");
    static_assert(offsetof(Batch, count) == gxdev::kBatchCount, "CGxBatch::count");
    static_assert(offsetof(Batch, minVert) == gxdev::kBatchMinVert, "CGxBatch::minVert");
    static_assert(offsetof(Batch, maxVert) == gxdev::kBatchMaxVert, "CGxBatch::maxVert");

    // --- small helpers ---------------------------------------------------------------------------

    /**
     * The first dword of one render state's value. The caller has already checked the pointer.
     *
     * @param uintptr table : the state array
     * @param unsigned id : an EGxRenderState below kRenderStateCount
     * @return uint32 value
     */
    inline uint32_t Rs(uintptr_t table, unsigned id)
    {
        return wxl::game::At<const uint32_t>(table + id * gxdev::kRsStride, gxdev::kRsValue);
    }

    /// A render state's value as a float: FogStart and FogEnd (states 8 and 9) are floats.
    inline float RsF(uintptr_t table, unsigned id)
    {
        return wxl::game::At<const float>(table + id * gxdev::kRsStride, gxdev::kRsValue);
    }

    /// A float held to [lo, hi].
    inline float Clamp(float v, float lo, float hi)
    {
        return v < lo ? lo : (v > hi ? hi : v);
    }

    /**
     * What the reader last read off the shader bound on one stage: the name as a Cmd carries it,
     * NUL-padded to the field's full width, and -- vertex stage -- the slice class. A frame is runs
     * of draws on one shader, so the name walk, the validity read and the record read happen once
     * per run and the other draws of the run cost one compare and one fixed 40-byte copy.
     *
     * Trusted while the memo's shader address, its +0x2C word and the resolver's generation are
     * the ones it was taken under: the native device bumps the generation at every create and
     * destroy (a destroyed CGxShader's address may be reused by another), and +0x2C is what a
     * create writes last.
     */
    struct ShaderMemo
    {
        uintptr_t shader     = 0;                       ///< the CGxShader (0: none yet)
        uint32_t  valid      = 0;                       ///< its +0x2C word
        uint32_t  generation = 0;                       ///< Resolver::shaderGeneration
        uint32_t  vconst     = Cmd::kVertexRegs;        ///< vertex stage: the slice class
        char      name[Cmd::kNameChars] = {};           ///< NUL-padded
        /// translated::Resolve(name) for the vertex stage, taken once per memo (WriteMoverRow, 2026-10-11): the registry
        /// it answers from changes only at a shader create, which moves the generation and so refreshes the memo.
        const manifest::Record* resolved = nullptr;
        bool      resolvedKnown = false;
    };
    ShaderMemo g_shaderMemo[2];   ///< [0] vertex, [1] pixel

    /**
     * The name of the shader a render state holds, into a Cmd field, and the vertex slice class.
     *
     * An invalid shader (+0x2C == 0) is bound as NULL, which is what D3D's IBindVertexShader /
     * IBindPixelShader do with one (cgxdevice 8.4): the draw is fixed function, the name is "".
     * The slice class is the `vconst` of the manifest record the native device filed at +0x20 when
     * it created the shader (src/native/Shaders.cpp): 64 when there is none (a hand-written row, an
     * unknown shader, a null state).
     *
     * The name is copied rather than kept as a pointer: a CGxShader may be rebuilt under us
     * (IShaderReload), and the name is an SStrDupA the engine owns. At most kNameChars - 1
     * characters, the rest of the field zero, so the pipeline key can copy and compare the whole
     * field.
     *
     * @param Resolver& r : the lookups
     * @param uintptr table : the state array
     * @param unsigned id : kRsVertexShader or kRsPixelShader
     * @param ShaderMemo& m : the stage's memo
     * @param char (&dst)[kNameChars] : receives the name; already zero (Reset) for a null state
     * @return uint32 vconst : the slice class, kVertexRegs or kVertexRegsHigh
     */
    inline uint32_t ReadShader(const Resolver& r, uintptr_t table, unsigned id, ShaderMemo& m,
                               char (&dst)[Cmd::kNameChars])
    {
        const uintptr_t shader = Rs(table, id);
        if (!shader) return Cmd::kVertexRegs;
        const uint32_t valid      = wxl::game::At<const uint32_t>(shader, gxdev::kShaderValid);
        const uint32_t generation = *r.shaderGeneration;
        if (shader != m.shader || valid != m.valid || generation != m.generation)
        {
            m.shader     = shader;
            m.valid      = valid;
            m.generation = generation;
            m.vconst     = Cmd::kVertexRegs;
            m.resolved      = nullptr;
            m.resolvedKnown = false;
            for (char& ch : m.name) ch = '\0';
            if (valid)
            {
                if (const char* src = wxl::game::At<const char* const>(shader, gxdev::kShaderName))
                    for (size_t k = 0; k + 1 < Cmd::kNameChars && src[k]; ++k) m.name[k] = src[k];
                const manifest::Record* row = wxl::game::At<const manifest::Record* const>(shader, gxdev::kShaderApiObject);
                if (row) m.vconst = row->vconst;
            }
        }
        std::memcpy(dst, m.name, sizeof m.name);   // a fixed 40 bytes: inlined, no CRT call
        return m.vconst;
    }

    // --- geometry ----------------------------------------------------------------------------------

    /**
     * The extra streams of a custom layout, each its pool's buffer at offset 0, addressed by
     * attribute 0's BaseVertexIndex: D3D's stream-offset mode 0 (caps +0x224 = 0), cgxdevice 7.2
     * step 4. A buf whose own offset / stride is another vertex is bound at 0 all the same, since
     * D3D ignores that offset as well.
     *
     * @param Resolver& r : the lookups
     * @param const void* const* streams : the distinct bufs, [0] = attribute 0's
     * @param uint32 count : how many
     * @param Cmd& c : receives extraStreams, extraBuffer, extraStride
     * @return bool ok : false when a pool had no buffer
     */
    bool ReadExtraStreams(const Resolver& r, const void* const* streams, uint32_t count, Cmd& c)
    {
        for (uint32_t i = 1; i < count; ++i)
        {
            const void* const buf  = streams[i];
            const void* const pool = wxl::game::At<const void* const>(buf, gxdev::kBufPool);
            VkBuffer          vk   = VK_NULL_HANDLE;
            if (!pool || !r.pool(pool, vk) || vk == VK_NULL_HANDLE) return false;
            c.extraBuffer[i - 1] = vk;
            c.extraStride[i - 1] = wxl::game::At<const uint32_t>(buf, gxdev::kBufItemSize);
        }
        c.extraStreams = count - 1;
        return true;
    }

    /**
     * Reads a custom vertex layout (device format 14) off the attribute slots into the Cmd.
     *
     * cgxdevice 3.4: slot a at kAttribSlots + a * kAttribSlotStride is {attrib, dataFormat, offset,
     * stride}, attribute a's buf is at kAttribBufs + a * 4, and kAttribEnabled says which slots the
     * draw uses. The D3D backend builds its declaration from exactly these (IStateSyncVertexPtrs,
     * one stream per distinct CGxBuf), so this is the whole description. Each distinct buf becomes a
     * stream, up to Cmd::kMaxStreams; a slot the Cmd cannot carry is refused, never guessed.
     *
     * @param Resolver& r : the lookups
     * @param const void* dev : the CGxDevice
     * @param const void* vb : attribute 0's CGxBuf, the draw's first stream
     * @param uint32 stride : that buf's item size, the stride the draw is addressed by
     * @param Cmd& c : receives customCount, custom[] and the extra streams
     * @return bool ok : false when refused
     */
    bool ReadCustomLayout(const Resolver& r, const void* dev, const void* vb, uint32_t stride, Cmd& c)
    {
        const uint32_t enabled = wxl::game::At<const uint32_t>(dev, gxdev::kAttribEnabled);
        c.customCount = 0;
        const void* streams[Cmd::kMaxStreams] = { vb };
        uint32_t    streamCount = 1;
        for (unsigned a = 0; a < gxdev::kAttribCount; ++a)
        {
            if (!(enabled & (1u << a))) continue;
            const void* const buf = wxl::game::At<const void* const>(dev, gxdev::kAttribBufs + a * sizeof(void*));
            uint32_t stream = 0;
            while (stream < streamCount && streams[stream] != buf) ++stream;
            if (stream == streamCount)
            {
                if (!buf || streamCount == Cmd::kMaxStreams) return false;
                streams[streamCount++] = buf;
            }
            // Each stream's stride is its own buf's item size; attribute 0's is `stride`.
            const uint32_t want = stream == 0 ? stride : wxl::game::At<const uint32_t>(buf, gxdev::kBufItemSize);
            const size_t   slot    = gxdev::kAttribSlots + a * gxdev::kAttribSlotStride;
            const uint32_t attrib  = wxl::game::At<const uint32_t>(dev, slot + gxdev::kAttribSlotAttrib);
            const uint32_t format  = wxl::game::At<const uint32_t>(dev, slot + gxdev::kAttribSlotFormat);
            const uint32_t offset  = wxl::game::At<const uint32_t>(dev, slot + gxdev::kAttribSlotOffset);
            const uint32_t sstride = wxl::game::At<const uint32_t>(dev, slot + gxdev::kAttribSlotStrideB);
            // A slot stride of 0 is tolerated as "the buf's": the fixed formats' descriptors carry
            // one, a caller filling the slots by hand may not.
            if (attrib != a || format > gxdev::kAttribFmtFloat1 || offset > 0xFFu || want == 0 ||
                (sstride != 0 && sstride != want))
                return false;
            if (c.customCount >= Cmd::kMaxCustomElements) return false;
            Cmd::VertexElement& e = c.custom[c.customCount++];
            e.attrib = static_cast<uint8_t>(attrib);
            e.format = static_cast<uint8_t>(format);
            e.offset = static_cast<uint8_t>(offset);
            e.stream = static_cast<uint8_t>(stream);
        }
        // No position: nothing the stock Draw could have addressed either.
        if (c.customCount == 0 || c.custom[0].attrib != gxdev::kAttribPosition) return false;
        return streamCount == 1 || ReadExtraStreams(r, streams, streamCount, c);
    }

    /**
     * Resolves the draw's geometry: the vertex and index bufs, their pools, the pool buffers.
     *
     * @param Resolver& r : the lookups
     * @param const void* dev : the CGxDevice
     * @param const Batch& b : the copied CGxBatch
     * @param int indexed : the slot's second argument
     * @param Cmd& c : receives the geometry fields
     * @return bool ok : false when anything was missing
     */
    bool ReadGeometry(const Resolver& r, const void* dev, const Batch& b, int indexed, Cmd& c)
    {
        const void* vb = wxl::game::At<const void* const>(dev, gxdev::kAttribBufs + kVertexStreamAttrib * sizeof(void*));
        if (!vb) return false;
        const uint32_t stride      = wxl::game::At<const uint32_t>(vb, gxdev::kBufItemSize);
        const uint32_t offsetBytes = wxl::game::At<const uint32_t>(vb, gxdev::kBufOffset);
        if (stride == 0) return false;

        // The format is geometry too: 14 means the attribute slots, not a table, say what the vertex
        // holds, and that is read here, where the bufs they name are known.
        c.vertexFormat = wxl::game::At<const uint32_t>(dev, gxdev::kVertexFormat);
        if (c.vertexFormat == Cmd::kVertexFormatCustom && !ReadCustomLayout(r, dev, vb, stride, c))
            return false;

        VkBuffer vertex = VK_NULL_HANDLE;
        if (!r.pool(wxl::game::At<const void* const>(vb, gxdev::kBufPool), vertex) || vertex == VK_NULL_HANDLE)
            return false;
        c.vertexBuffer = vertex;
        c.vertexStride = stride;
        c.vertexItems  = wxl::game::At<const uint32_t>(vb, gxdev::kBufItemCount);
        c.vertexBufId  = reinterpret_cast<uintptr_t>(vb);
        c.vertexBufBytes = wxl::game::At<const uint32_t>(vb, gxdev::kBufBytes);
        {
            const void* pool = wxl::game::At<const void* const>(vb, gxdev::kBufPool);
            c.vertexPoolBytes = pool ? wxl::game::At<const uint32_t>(pool, gxdev::kPoolSize) : 0u;
        }
        // Exact: a stream lock places the buf at a stride multiple, which is what lets D3D take it
        // as a BaseVertexIndex at all. Narrowed to the signed field D3D itself uses.
        c.vertexOffset = static_cast<int32_t>(offsetBytes / stride);

        c.topology   = b.prim;
        c.indexed    = indexed != 0;
        c.indexCount = b.count;   // indices when indexed, vertices when not: the engine's own meaning

        if (!c.indexed)
        {
            // DrawPrimitive: batch.start is not read by the stock code, so it is not kept.
            c.indexBuffer = VK_NULL_HANDLE;
            c.firstIndex  = 0;
            return true;
        }

        const void* ib = wxl::game::At<const void* const>(dev, gxdev::kIndexBuf);
        if (!ib) return false;
        VkBuffer index = VK_NULL_HANDLE;
        if (!r.pool(wxl::game::At<const void* const>(ib, gxdev::kBufPool), index) || index == VK_NULL_HANDLE)
            return false;
        c.indexBuffer = index;
        c.indexPoolBytes = wxl::game::At<const uint32_t>(wxl::game::At<const void* const>(ib, gxdev::kBufPool), gxdev::kPoolSize);
        c.indexBase   = wxl::game::At<const uint32_t>(ib, gxdev::kBufOffset) / gxdev::kIndexSize;
        c.firstIndex  = c.indexBase + b.start;
        return true;
    }

    // --- material and fixed state ------------------------------------------------------------------

    /**
     * Reads the material: the two shader names (and the vertex slice class), every bound texture
     * stage and its flags.
     *
     * A texture the resolver does not know leaves its stage untextured; the pipelines decide whether
     * the draw can go without it. Every image is born asking to end its uploads in the sampling
     * layout (Image::wanted, src/native/Textures.cpp); the texture module owns the barrier.
     *
     * @param Resolver& r : the lookups
     * @param uintptr table : the state array
     * @param Stages& st : the stages the slot found bound
     * @param Cmd& c : receives the material fields and `vconst`
     */
    void ReadMaterial(const Resolver& r, uintptr_t table, const Stages& st, Cmd& c)
    {
        c.vconst = ReadShader(r, table, gxdev::kRsVertexShader, g_shaderMemo[0], c.vs);
        ReadShader(r, table, gxdev::kRsPixelShader, g_shaderMemo[1], c.ps);

        // The bound stages, and only those: a translated pixel module may sample any slot up to
        // s8 (terrain binds nine), and the Cmd's defaults already say "no texture" for the rest.
        c.engineTexMask = st.mask;
        if (st.mask & 1u) c.engineTex0 = st.tex[0];
        for (uint32_t mask = st.mask; mask; mask &= mask - 1)
        {
            unsigned long stage = 0;
            _BitScanForward(&stage, mask);
            const void* const     tex   = st.tex[stage];
            const texture::Image* image = nullptr;
            if (!r.texture(tex, image)) continue;
            c.texture[stage]  = image;
            c.texFlags[stage] = wxl::game::At<const uint32_t>(tex, gxdev::kTexFlags);
        }
    }

    /**
     * Reads the fixed state, with the master enables applied.
     *
     * A master bit that is off forces its state to the disabled value whatever the app state says
     * (slot 51's contract, gx::kMasterEnableField). The app array holds the value the engine asked
     * for; the backend applies the override on the way to the hardware, so a Cmd built from the app
     * array has to apply it too, or a draw the engine ran without depth test would get one.
     *
     * @param const void* dev : the CGxDevice
     * @param uintptr table : the state array
     * @param Cmd& c : receives the state fields
     */
    void ReadFixedState(const void* dev, uintptr_t table, Cmd& c)
    {
        const uint32_t master = wxl::game::At<const uint32_t>(dev, gx::kMasterEnableField);
        const auto on = [master](unsigned bit) { return (master & (1u << bit)) != 0; };

        c.blend      = Rs(table, gxdev::kRsBlend);
        c.alphaRef   = Rs(table, gxdev::kRsAlphaRef);
        c.depthTest  = on(gxdev::kMasterDepthTest) && Rs(table, gxdev::kRsDepthTest) != 0;
        c.depthFunc  = Rs(table, gxdev::kRsDepthFunc);
        c.depthWrite = on(gxdev::kMasterDepthWrite) && Rs(table, gxdev::kRsDepthWrite) != 0;
        c.colorWrite = on(gxdev::kMasterColorWrite) ? Rs(table, gxdev::kRsColorWrite) : 0u;
        c.cull       = on(gxdev::kMasterCulling) ? Rs(table, gxdev::kRsCulling) : gxdev::kCullNone;
    }

    // --- fixed function ------------------------------------------------------------------------

    /// A CImVector, 0xAARRGGBB as the dword the state array holds, as r g b a in 0..1.
    inline void ImVector(uint32_t v, float out[4])
    {
        out[0] = static_cast<float>((v >> 16) & 0xFFu) / 255.0f;
        out[1] = static_cast<float>((v >> 8) & 0xFFu) / 255.0f;
        out[2] = static_cast<float>(v & 0xFFu) / 255.0f;
        out[3] = static_cast<float>(v >> 24) / 255.0f;
    }

    /**
     * Whether D3D's fog blend runs for this draw: the master fog bit and render state 12 (Fog) both
     * on (cgxdevice 6.5). Shared by the vertex-side fog rows and the pixel-side fog row, so the two
     * cannot drift.
     */
    inline bool FogEnabled(const void* dev, uintptr_t table)
    {
        const uint32_t master = wxl::game::At<const uint32_t>(dev, gx::kMasterEnableField);
        return (master & (1u << gxdev::kMasterFog)) != 0 && Rs(table, gxdev::kRsFog) != 0;
    }

    /**
     * The fog colour and enable for the pixel side, in pixelConst[kFogRow]: (FogColor.rgb / 255,
     * fogOn ? 1 : 0), written for every draw after the pixel registers are copied.
     *
     * Safe because no shipped pixel module reads p[13..15] -- the engine uploads c0..c12 at most and
     * c13/c14 are def'd literals -- and the hand-written fragment shaders read p[] not at all. The
     * one reader is the generated `fixed_fog` companion, which blends mix(p[15].rgb, rgb, fog0) while
     * p[15].w > 0: the post-shader fog D3D applies to a programmable vertex shader's oFog only while
     * FOGENABLE is on.
     */
    inline void WriteFogRow(const void* dev, uintptr_t table, Cmd& c)
    {
        float* row = c.pixelConst[Cmd::kFogRow];
        ImVector(Rs(table, gxdev::kRsFogColor), row);
        row[3] = FogEnabled(dev, table) ? 1.0f : 0.0f;
    }

    /**
     * The mover row (Cmd::kMoverRow): p[14].x is 1 for a unit's draw. A unit's draw is a model draw -- its
     * vertex module takes bone weights, which the terrain and the map objects have not -- whose model wears a
     * unit's texture (vulkan/Wind.cpp: a character's, a creature's, an item's) or a composited, nameless skin
     * (a character's body). After the geometry and the material are read (the buffers, the stage-0 texture and
     * the vertex shader's name are what the two tests look at).
     */
    inline void WriteMoverRow(Cmd& c)
    {
        float* row = c.pixelConst[Cmd::kMoverRow];
        row[0] = row[1] = row[2] = row[3] = 0.0f;
        // p[14].y: a batch of an interior-lit map object group (native/render/WmoGroups.cpp, the engine's group
        // renderers bracketed). Every indoor group's MapObj vertex module (exterior-lit ones too) places the
        // group's box in view space through its c31..c33.
        if (wxlvk::native::render::wmogroups::InteriorDraw()) row[1] = 1.0f;
        if (wxlvk::native::render::wmogroups::EnclosedDraw() && std::strncmp(c.vs, "MapObj", 6) == 0)
            wxlvk::native::render::wmogroups::NoteInteriorDraw(c.vertexRows);
        if (c.vs[0] == '\0') return;
        // c.vs is the vertex memo's name (ReadMaterial ran first): its Resolve is memoised with it.
        ShaderMemo& memo = g_shaderMemo[0];
        const wxlvk::vk::manifest::Record* vr = nullptr;
        if (std::memcmp(memo.name, c.vs, sizeof memo.name) == 0)
        {
            if (!memo.resolvedKnown)
            {
                memo.resolved      = wxlvk::shaders::translated::Resolve(c.vs, false);
                memo.resolvedKnown = true;
            }
            vr = memo.resolved;
        }
        else
            vr = wxlvk::shaders::translated::Resolve(c.vs, false);
        if (!vr || (vr->inputMask & (1u << wxlvk::vk::manifest::kBlendWeight0)) == 0u) return;
        if (wxlvk::vk::wind::MoverDraw(c)) row[0] = 1.0f;
    }

    /**
     * Stage 0's combiner ops into the Cmd's selectors: ColorOp0 / AlphaOp0 (states 37 / 45) while the
     * stage has a texture, kFfOpNone otherwise -- the D3D stage state is `stageHasTexture ? op :
     * DISABLE` (6.1), and the cascade ends at the first DISABLE. Shared by the fixed-function file
     * and the companion read, so the two encode the selectors identically.
     *
     * @return bool textured : whether stage 0 has a texture
     */
    inline bool ReadStage0Ops(uintptr_t table, Cmd& c)
    {
        if (c.engineTexMask & 1u)   // ReadMaterial ran first
        {
            c.ffColorOp = Rs(table, gxdev::kRsColorOp0);
            c.ffAlphaOp = Rs(table, gxdev::kRsAlphaOp0);
            return true;
        }
        c.ffColorOp = Cmd::kFfOpNone;
        c.ffAlphaOp = Cmd::kFfOpNone;
        return false;
    }

    /// One C44Matrix: 16 floats, row-major, translation in 12..14.
    constexpr size_t kMatrixBytes = 16 * sizeof(float);
    static_assert(kMatrixBytes == gx::kDeviceViewStride, "a matrix stack level is one C44Matrix");

    /**
     * The matrix at the top of one of the device's CGxMatrixStack objects.
     *
     * The top's identity flag is honoured rather than the bytes trusted: it is what the D3D backend
     * itself consults (IStateSyncXforms skips the world transform when both the last and the current
     * top are flagged identity), so a level the engine flagged without rewriting is identity by that
     * contract, whatever the bytes say.
     *
     * @param const void* dev : the CGxDevice
     * @param unsigned xform : the stack, gxdev::kXformTex0 + s, kXformWorld, kXformView or kXformTexGen0 + s
     * @param float out[16] : receives the matrix
     * @return bool identity : the top carried the identity flag (out is the identity then)
     */
    bool StackTop(const void* dev, unsigned xform, float out[16])
    {
        const uintptr_t stack = reinterpret_cast<uintptr_t>(dev) + gxdev::kMatrixStacks
                                + xform * gxdev::kMatrixStackStride;
        uint32_t top = wxl::game::At<const uint32_t>(stack, gxdev::kMatrixStackTop);
        if (top >= gxdev::kMatrixStackDepth) top = gxdev::kMatrixStackDepth - 1;

        const uint32_t flags = wxl::game::At<const uint32_t>(stack, gxdev::kMatrixStackFlags + top * sizeof(uint32_t));
        if (flags & gxdev::kMatrixStackFlagIdentity)
        {
            for (unsigned i = 0; i < 16; ++i) out[i] = (i % 5 == 0) ? 1.0f : 0.0f;
            return true;
        }
        std::memcpy(out, &wxl::game::At<const float>(stack, gxdev::kMatrixStackBase + top * kMatrixBytes), kMatrixBytes);
        return false;
    }

    /// out = a * b, row-major: under the engine's `v' = v * M` convention, World * View in that
    /// order is the matrix that takes a vertex to view space.
    void Mul(const float a[16], const float b[16], float out[16])
    {
        for (unsigned r = 0; r < 4; ++r)
            for (unsigned c = 0; c < 4; ++c)
                out[r * 4 + c] = a[r * 4 + 0] * b[0 * 4 + c] + a[r * 4 + 1] * b[1 * 4 + c]
                               + a[r * 4 + 2] * b[2 * 4 + c] + a[r * 4 + 3] * b[3 * 4 + c];
    }

    /**
     * The inverse transpose of a C44Matrix's upper 3x3, as three rows of four (w 0), for the normal:
     * under `v' = v * M` a normal goes through (M^-1)^T so that it stays perpendicular under a
     * non-uniform scale, which is what D3D's fixed function does with NORMAL0 before lighting. A
     * singular 3x3 (a degenerate World, which D3D would light with garbage anyway) falls back to the
     * 3x3 itself.
     *
     * @param const float m[16] : World * View
     * @param float out[3][4] : receives the rows
     */
    void NormalXform(const float m[16], float out[3][4])
    {
        const float a = m[0], b = m[1], c = m[2];
        const float d = m[4], e = m[5], f = m[6];
        const float g = m[8], h = m[9], i = m[10];
        // Cofactors; the inverse is adj / det, adj = cofactor^T, so inverse^T = cofactor / det.
        const float c00 = e * i - f * h, c01 = f * g - d * i, c02 = d * h - e * g;
        const float c10 = c * h - b * i, c11 = a * i - c * g, c12 = b * g - a * h;
        const float c20 = b * f - c * e, c21 = c * d - a * f, c22 = a * e - b * d;
        const float det = a * c00 + b * c01 + c * c02;
        if (std::fabs(det) < 1e-12f)
        {
            for (unsigned r = 0; r < 3; ++r)
                for (unsigned k = 0; k < 3; ++k) out[r][k] = m[r * 4 + k];
        }
        else
        {
            const float inv = 1.0f / det;
            out[0][0] = c00 * inv; out[0][1] = c01 * inv; out[0][2] = c02 * inv;
            out[1][0] = c10 * inv; out[1][1] = c11 * inv; out[1][2] = c12 * inv;
            out[2][0] = c20 * inv; out[2][1] = c21 * inv; out[2][2] = c22 * inv;
        }
        out[0][3] = out[1][3] = out[2][3] = 0.0f;
    }

    /**
     * Reads the device's light slots (cgxdevice 6.4) into the Cmd's kFfLight* rows, the enabled ones
     * packed first, each moved into view space the way D3D moves a D3DLIGHT9: the engine sets the
     * lights in world space and D3D lights in camera space, so a position goes through the View top
     * whole and a direction through its 3x3 only. The slots stay world-space on the device.
     *
     * @param const void* dev : the CGxDevice
     * @param const float view[16] : the View stack top
     * @param Cmd& c : receives the light rows
     * @return unsigned count : enabled lights written, 0..kFfLightCount
     */
    unsigned ReadLights(const void* dev, const float view[16], Cmd& c)
    {
        unsigned n = 0;
        for (unsigned i = 0; i < gxdev::kLightCount && i < Cmd::kFfLightCount; ++i)
        {
            const uintptr_t slot = reinterpret_cast<uintptr_t>(dev) + gxdev::kLights + i * gxdev::kLightStride;
            if (wxl::game::At<const uint32_t>(slot, gxdev::kLightEnabled) == 0) continue;

            float (*rows)[4] = &c.vertexConst[Cmd::kFfLight0 + n * Cmd::kFfLightRows];
            const float* p   = &wxl::game::At<const float>(slot, gxdev::kLightPosition);
            const bool point = p[3] == 1.0f;
            for (unsigned k = 0; k < 3; ++k)
            {
                rows[0][k] = p[0] * view[0 * 4 + k] + p[1] * view[1 * 4 + k] + p[2] * view[2 * 4 + k]
                           + (point ? view[3 * 4 + k] : 0.0f);
            }
            rows[0][3] = point ? 1.0f : 0.0f;

            const float* amb = &wxl::game::At<const float>(slot, gxdev::kLightAmbient);
            const float* dif = &wxl::game::At<const float>(slot, gxdev::kLightDiffuse);
            const float* spc = &wxl::game::At<const float>(slot, gxdev::kLightSpecular);
            const float* att = &wxl::game::At<const float>(slot, gxdev::kLightAttenuation);
            for (unsigned k = 0; k < 3; ++k)
            {
                rows[1][k] = amb[k];
                rows[2][k] = dif[k];
                rows[3][k] = spc[k];
            }
            rows[1][3] = att[0];
            rows[2][3] = att[1];
            rows[3][3] = att[2];
            ++n;
        }
        return n;
    }

    /**
     * One stage's texture matrix and texture-coordinate control, as the D3D backend sends them
     * (cgxdevice 5.4 IXformSetTex, 6.2 ISetTexGen).
     *
     * The backend composes D3DTS_TEXTURE0 + s and its TEXTURETRANSFORMFLAGS from the TexTransform
     * state (61 + s) and two stack tops: the texgen matrix (stack 11 + s, which ISetTexGen filled from
     * the TexGen mode -- src/native/Draw.cpp SetTexGen) and the application texture matrix (stack s):
     *
     *   state 0: texgen alone; COUNT3 when it is not identity, DISABLE when it is
     *   state 1: texgen * appTex; TexGen 0: COUNT2 with row 3 (the C44Matrix translation) copied into
     *            row 2, D3D's 2D convention -- a two-component input is padded (u, v, 1, 0), so row 2
     *            is the row a 2D translation must sit in; else COUNT3
     *   state 2: texgen * appTex; COUNT3 | PROJECTED
     *
     * This writes the same product from the same tops into kFfTexXf0 + 4 * stage, and the TexGen mode
     * with the flags into kFfTexCtl; ff.vert runs the multiply and ff.frag the projective divide.
     *
     * @param const void* dev : the CGxDevice
     * @param uintptr table : the state array
     * @param unsigned stage : 0 or 1
     * @param Cmd& c : receives the rows
     */
    void ReadTexXform(const void* dev, uintptr_t table, unsigned stage, Cmd& c)
    {
        uint32_t gen        = Rs(table, gxdev::kRsTexGen0 + stage);
        const uint32_t mode = Rs(table, gxdev::kRsTexTransform0 + stage);

        float texgen[16], app[16];
        float* m = c.vertexConst[Cmd::kFfTexXf0 + Cmd::kFfTexXfRows * stage];
        const bool texgenIdentity = StackTop(dev, gxdev::kXformTexGen0 + stage, texgen);
        uint32_t flags = 0;
        switch (mode)
        {
        case 0:
            std::memcpy(m, texgen, kMatrixBytes);
            flags = texgenIdentity ? 0u : 3u;
            break;
        case 1:
            StackTop(dev, gxdev::kXformTex0 + stage, app);
            Mul(texgen, app, m);
            if (gen == 0)
            {
                flags = 2u;
                for (unsigned k = 0; k < 4; ++k) m[8 + k] = m[12 + k];
            }
            else
                flags = 3u;
            break;
        case 2:
            StackTop(dev, gxdev::kXformTex0 + stage, app);
            Mul(texgen, app, m);
            flags = 3u | Cmd::kFfTtfProjected;
            break;
        default:
            // Not a value the engine's table has: the stage runs untransformed.
            for (unsigned i = 0; i < 16; ++i) m[i] = (i % 5 == 0) ? 1.0f : 0.0f;
            break;
        }

        // Past EGxTexGen: the vertex set goes through plain.
        if (gen > 6) gen = 0;

        float* ctl = c.vertexConst[Cmd::kFfTexCtl];
        ctl[2 * stage]     = static_cast<float>(gen);
        ctl[2 * stage + 1] = static_cast<float>(flags);
    }

    /**
     * Reads what a draw with no vertex shader needs: the device's own transforms, the colour source,
     * the fog and the stage ops (cgxdevice 5.4, 6.1-6.5, 6.8).
     *
     * The engine uploads no constant for such a draw, so the rows this writes (Cmd::kFf*) replace a
     * register copy that holds, at best, the previous shader draw's values. Both halves are read
     * whenever the VERTEX shader is null, because the D3D backend syncs its fixed-function state that
     * way (IStateSync step 3). A draw with a programmable vertex shader and no pixel shader keeps its
     * registers and takes ReadStage0Ops() alone.
     *
     * @param const void* dev : the CGxDevice
     * @param uintptr table : the state array
     * @param Cmd& c : vs/ps already read; receives the fixed-function fields and rows
     */
    void ReadFixedFunction(const void* dev, uintptr_t table, Cmd& c)
    {
        c.fixedFunction = true;
        const uint32_t master = wxl::game::At<const uint32_t>(dev, gx::kMasterEnableField);
        const auto on = [master](unsigned bit) { return (master & (1u << bit)) != 0; };

        // --- transforms: pos * World * View * ProjNative, the three IStateSyncXforms sends -------
        float world[16], view[16], worldView[16];
        StackTop(dev, gxdev::kXformWorld, world);
        StackTop(dev, gxdev::kXformView, view);
        Mul(world, view, worldView);
        std::memcpy(c.vertexConst[Cmd::kFfWorldView], worldView, kMatrixBytes);
        std::memcpy(c.vertexConst[Cmd::kFfProjection],
                    &wxl::game::At<const float>(dev, gxdev::kApiProjection), kMatrixBytes);

        // --- the colour source (6.3, 6.4) --------------------------------------------------------
        const uint32_t enabled = wxl::game::At<const uint32_t>(dev, gxdev::kAttribEnabled);
        const bool hasColor  = (enabled & (1u << gxdev::kAttribBitColor0)) != 0;
        const bool hasNormal = (enabled & (1u << gxdev::kAttribBitNormal)) != 0;
        const bool lit = on(gxdev::kMasterLighting) && Rs(table, gxdev::kRsLighting) != 0;
        const bool normalizeNormals = Rs(table, gxdev::kRsNormalizeNormals) != 0;
        const uint32_t matDiffuse = Rs(table, gxdev::kRsMatDiffuse);
        float diffuse[4], emissive[4];
        ImVector(matDiffuse, diffuse);
        ImVector(Rs(table, gxdev::kRsMatEmissive), emissive);
        float* colour = c.vertexConst[Cmd::kFfColor];

        if (!lit)
        {
            if (hasColor || matDiffuse == 0xFFFFFFFFu)
            {
                // LIGHTING off: Color0, or D3D's opaque white when the vertex has none.
                c.ffColorMode = Cmd::kFfColorVertex;
            }
            else
            {
                // The engine's tint for unlit, uncoloured geometry: LIGHTING on with every light off
                // and a white AMBIENT, so ambient (= MatDiffuse) * white + emissive, with the
                // material's diffuse alpha.
                c.ffColorMode = Cmd::kFfColorConstant;
                for (unsigned i = 0; i < 3; ++i) colour[i] = Clamp(diffuse[i] + emissive[i], 0.0f, 1.0f);
                colour[3] = diffuse[3];
            }
        }
        else
        {
            // Lit. Without a normal D3D's diffuse and specular terms are zero and the engine sends
            // AMBIENT 0 in this mode, so the emissive is all that survives: that is what the
            // normal-less formats (7, 8, 10) get, and what kFfColor holds in every lit case. The
            // alpha follows the diffuse source -- COLOR1 when the vertex has one and ColorMaterial
            // (85) is 0, the material otherwise (6.3).
            const uint32_t colorMaterial = Rs(table, gxdev::kRsColorMaterial);
            const bool alphaFromVertex = hasColor && colorMaterial == 0;
            c.ffColorMode = alphaFromVertex ? Cmd::kFfColorConstantRgb : Cmd::kFfColorConstant;
            for (unsigned i = 0; i < 3; ++i) colour[i] = emissive[i];
            colour[3] = diffuse[3];

            if (hasNormal)
            {
                // With a normal, the lights apply (6.4): ff.vert under WXL_FF_LIT forms D3D's vertex
                // lighting from the material, the colour-source row of 6.3's table and the enabled
                // light slots moved into view space. The material's ambient IS its diffuse (state 1,
                // ISetMaterial), and the global AMBIENT is 0 in this mode, so no ambient row travels.
                // The specular term is kept separate by the shader -- D3D adds it after the texture
                // cascade -- and is on only while SPECULARENABLE would be: power > 0 and a non-zero
                // specular colour.
                c.ffColorMode = Cmd::kFfColorLit;

                float nx[3][4];
                NormalXform(worldView, nx);
                nx[0][3] = normalizeNormals ? 1.0f : 0.0f;
                std::memcpy(c.vertexConst[Cmd::kFfNormalXf], nx, sizeof nx);

                const uint32_t matSpecular = Rs(table, gxdev::kRsMatSpecular);
                const float    power       = RsF(table, gxdev::kRsMatSpecularExp);
                float specular[4];
                ImVector(matSpecular, specular);
                std::memcpy(c.vertexConst[Cmd::kFfMatDiffuse], diffuse, sizeof diffuse);
                float* em = c.vertexConst[Cmd::kFfMatEmissive];
                for (unsigned i = 0; i < 3; ++i) em[i] = emissive[i];
                em[3] = power;
                float* sp = c.vertexConst[Cmd::kFfMatSpecular];
                for (unsigned i = 0; i < 3; ++i) sp[i] = specular[i];
                sp[3] = (power > 0.0f && matSpecular != 0u) ? 1.0f : 0.0f;

                // 6.3's table: no Color0 -> every source is the material (row 0); with one,
                // ColorMaterial 0/1/2 picks which source the vertex colour replaces (rows 1..3).
                float* ctl = c.vertexConst[Cmd::kFfLightCtl];
                ctl[0] = hasColor ? static_cast<float>((colorMaterial <= 2 ? colorMaterial : 0) + 1) : 0.0f;
                ctl[1] = normalizeNormals ? 1.0f : 0.0f;
                ctl[2] = static_cast<float>(ReadLights(dev, view, c));
                ctl[3] = 0.0f;
            }
        }

        // --- fog (6.5): linear in view depth, off as (0, 1) so the factor is exactly 1 ----------
        float* fog = c.vertexConst[Cmd::kFfFog];
        fog[0] = 0.0f;
        fog[1] = 1.0f;
        fog[2] = fog[3] = 0.0f;
        if (FogEnabled(dev, table))
        {
            const float start = RsF(table, gxdev::kRsFogStart);
            const float end   = RsF(table, gxdev::kRsFogEnd);
            // (end - z) / (end - start) = z * scale + bias. Equal bounds would divide by zero, which
            // D3D leaves undefined; no fog is the harmless reading.
            if (end != start)
            {
                fog[0] = -1.0f / (end - start);
                fog[1] = end / (end - start);
            }
        }
        ImVector(Rs(table, gxdev::kRsFogColor), c.vertexConst[Cmd::kFfFogColor]);

        // --- stages 0 and 1 (6.1): the ops only while the stage has a texture; DISABLE otherwise.
        //
        // The cascade ends at the first untextured stage, so stage 1 counts only behind a textured
        // stage 0. Its ops travel in v[kFfStage1] with the UV set each stage reads (states 69/70,
        // default = the stage); the fragment shader built with two samplers (the rows for formats 6,
        // 7 and 9) runs it. Format 7 (PC) has no UV set at all, so its textured draws generate both
        // stages' coordinates (the ground decals) and run stage 1 as the two-UV layouts do.
        float* stage1 = c.vertexConst[Cmd::kFfStage1];
        stage1[0] = stage1[1] = static_cast<float>(Cmd::kFfOpNone);
        stage1[2] = static_cast<float>(Rs(table, gxdev::kRsTexCoord0 + 1));
        stage1[3] = static_cast<float>(Rs(table, gxdev::kRsTexCoord0));
        const bool stage0Textured = ReadStage0Ops(table, c);
        if (stage0Textured && (c.engineTexMask & 2u))
        {
            // The formats whose row runs stage 1: 6, 9 and 7 by their own rows, and the ones the
            // pipelines draw through those rows (CoverRow): 0 through 7's, 5, 11, 12 and 13
            // through the two-UV rows 6 and 9.
            const uint32_t f   = c.vertexFormat;
            const bool twoUv = f == 6 || f == 9 || f == 7 || f == 0 || f == 5 || f == 11 || f == 12 || f == 13;
            if (twoUv)
            {
                stage1[0] = static_cast<float>(Rs(table, gxdev::kRsColorOp0 + 1));
                stage1[1] = static_cast<float>(Rs(table, gxdev::kRsAlphaOp0 + 1));
            }
        }

        // --- TexGen and the texture transform on stages 0 and 1 (5.4, 6.2) ----------------------
        //
        // Written for both stages on every draw, textured or not, since the D3D backend sends the
        // texture matrices by the stage's dirty bytes, not by its texture, and the semantic ff.vert
        // transforms output texcoords 0 and 1 for a pixel module whatever the cascade would have
        // sampled. The normal-reading modes (4..6) need the camera-space normal rows, which the lit
        // path writes already; an unlit draw asking for them gets them here. The semantic build
        // declares no normal input, so for a draw feeding a pixel module those modes run with a zero
        // normal.
        const bool normalUsable = hasNormal && c.ps[0] == '\0';
        for (unsigned st = 0; st < Cmd::kFfTexXfStages; ++st) ReadTexXform(dev, table, st, c);
        if (normalUsable && c.ffColorMode != Cmd::kFfColorLit)
        {
            const float* tc = c.vertexConst[Cmd::kFfTexCtl];
            if (tc[0] >= 4.0f || tc[2] >= 4.0f)
            {
                float nx[3][4];
                NormalXform(worldView, nx);
                nx[0][3] = normalizeNormals ? 1.0f : 0.0f;
                std::memcpy(c.vertexConst[Cmd::kFfNormalXf], nx, sizeof nx);
            }
        }

        // --- what the semantic ff.vert needs to feed a translated pixel module -------------------
        //
        // D3D's fixed-function vertex pipeline routes input texcoord set TEXCOORDINDEX[n] to output
        // texcoord n (states 69..76, the low 16 bits; the high bits select a texgen, which is not run
        // here), and a ps_2_0 module reads output n as t<n>. Which sets and whether a colour the
        // vertex carries come from the device's enabled attribute mask, which GxPrimVertexPtr sets
        // for a fixed format and the caller for a custom one, so one read serves both. The two
        // half-pixel components stay 0: the native viewport is moved onto D3D9's pixel centres
        // instead (src/native/Draw.cpp), which serves every draw, the full-screen FFX passes included.
        float* route = c.vertexConst[Cmd::kFfTexRoute];
        for (unsigned n = 0; n < 8; ++n)
            route[n] = static_cast<float>(Rs(table, gxdev::kRsTexCoord0 + n) & 0xFFu);
        float* ctl = c.vertexConst[Cmd::kFfSemanticCtl];
        ctl[0] = static_cast<float>((enabled >> gxdev::kAttribTexCoord0) & 0xFFu);
        ctl[1] = 0.0f;
        ctl[2] = 0.0f;
        ctl[3] = (enabled & (1u << gxdev::kAttribBitColor0)) ? 1.0f : 0.0f;
    }
}

namespace wxlvk::draw
{
    bool BuildCmd(const void* device, const wxlvk::device::CGxBatch* batch, int indexed,
                  const Resolver& resolver, const Stages& stages, Cmd& c)
    {
        if (!device || !batch || !resolver.pool || !resolver.texture || !resolver.shaderGeneration
            || !resolver.vertexRegs || !resolver.pixelRegs)
            return false;

        Batch b;
        std::memcpy(&b, batch, sizeof b);
        // Every field to its default except the two register arrays at the end, which are written
        // below before anything reads them (pixelConst always, vertexConst for a fixed-function
        // draw).
        Reset(c);

        if (!ReadGeometry(resolver, device, b, indexed, c)) return false;
        const uintptr_t table = wxl::game::At<const uintptr_t>(device, gxdev::kRsTable);
        if (!table) return false;

        ReadMaterial(resolver, table, stages, c);
        ReadFixedState(device, table, c);

        const uint8_t* vregs = reinterpret_cast<const uint8_t*>(resolver.vertexRegs);
        c.vertexRows = resolver.vertexRegs;
        wxlvk::simd::CopyRows(c.pixelConst, resolver.pixelRegs, Cmd::kPixelRegs);
        WriteFogRow(device, table, c);
        WriteMoverRow(c);

        // The slice class came with the vertex shader (ReadMaterial): 256 when its module indexes
        // the bone palette or reads the shadow matrices, 64 for every other shader, the
        // hand-written pairs and the fixed function among them. The high rows are the live
        // registers; the pipelines copy them into the ring before the slot returns.
        c.vertexConstHigh = c.vconst == Cmd::kVertexRegsHigh
                                ? reinterpret_cast<const float (*)[4]>(vregs + Cmd::kVertexRegs * 16u)
                                : nullptr;

        // The two states D3D's state handler (0x006A4C30) sends this way (cgxdevice 2.5): state 0 as
        // D3DRS_DEPTHBIAS = -value when caps +0x304 offers it, and state 84 as BLENDFACTOR =
        // round(v * 255) in every channel, which only blend mode 11 reads.
        if (wxl::game::At<const uint32_t>(device, gxdev::kCapsDepthBias) != 0)
        {
            const float offset = RsF(table, gxdev::kRsPolygonOffset);
            if (offset != 0.0f) c.depthBias = -offset;
        }
        if (c.blend == 11)
        {
            const float v = Clamp(RsF(table, gxdev::kRsBlendFactor), 0.0f, 1.0f);
            c.blendFactor = std::floor(v * 255.0f + 0.5f) / 255.0f;
        }

        // A draw with a null VERTEX shader gets the device's own state in place of stale registers:
        // a copy of the registers with the fixed-function file written over its rows. A null pixel
        // shader under a real vertex shader (the vs_2_0 weather records, drawn through the generated
        // `ff` companion) is the other way round: the vertex registers are that module's c0..c21 and
        // must not be touched, so only the stage-0 selectors are read.
        if (c.vs[0] == '\0')
        {
            wxlvk::simd::CopyRows(c.vertexConst, vregs, Cmd::kVertexRegs);
            c.vertexRows = c.vertexConst;
            ReadFixedFunction(device, table, c);
        }
        else if (c.ps[0] == '\0')
            ReadStage0Ops(table, c);
        return true;
    }
}
