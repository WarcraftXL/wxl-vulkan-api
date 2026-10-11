// The native device's draw: slot 1 IRsSendToHw, slot 42 Draw, the D3D constant registers, and the
// D3D viewport / scissor rectangle the draw and the clear (slot 39) share.
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

#include "native/Slots.hpp"

#include "common/Simd.hpp"
#include "native/Engine.hpp"
#include "native/Perf.hpp"
#include "native/Present.hpp"
#include "native/Render.hpp"
#include "native/render/Jitter.hpp"
#include "draw/Cmd.hpp"
#include "draw/Reader.hpp"
#include "vulkan/Memory.hpp"
#include "vulkan/Wind.hpp"
#include "vulkan/Pipelines.hpp"

#include "wxl/Script.hpp"
#include "wxl/game/Binding.hpp"
#include "wxl/offsets/engine/Gx.hpp"
#include "wxl/offsets/engine/GxDevice.hpp"

#include <cstdint>
#include <cstring>

// D3D's Draw (0x006A3620) is: the two gates (+0xF58 context, +0xF5C poisoned frame), IStateSync
// (constants, IRsSync, fixed-function state, the vertex declaration and streams, the index buffer,
// the viewport), then DrawIndexedPrimitive / DrawPrimitive. Native keeps the gates and IRsSync(0)
// (so the dirty list and the shadow move as under D3D -- slot 1 is a no-op, the state is read at
// draw time), runs what the D3D binds would have run lazily (the shader creates of
// IBindVertexShader / IBindPixelShader while +0x30 is 0, slot 0 for each bound texture, as
// ISetTexture does), then builds a draw::Cmd on the stack (draw::BuildCmd) and records it at once
// through pipelines::Record into the frame's render pass on the current target.
//
// Placement: native draws into the target the engine sized its viewport for, so it uses D3D's own
// integer formulas (cgxdevice 9.4, IXformSetViewport and IStateSyncScissorRect) -- the same pixels
// D3D would touch -- with D3D's render-target rule when a texture is bound on buffer 0 or 1 (no y
// flip of the viewport's Y, the scissor's render-target rows).
namespace
{
    namespace gx        = wxl::offsets::engine::gx;
    namespace gxdev     = wxl::offsets::engine::gxdevice;
    namespace slots     = wxlvk::device::slots;
    namespace present   = wxlvk::native::present;
    namespace pipelines = wxlvk::vk::pipelines;

    using wxl::game::At;
    using wxlvk::device::CGxBatch;
    using wxlvk::device::CGxShader;
    using wxlvk::draw::Cmd;
    using wxlvk::native::Engine;
    using wxlvk::native::Own;

    /// Slot 42's time and its phases into the perf buckets kDraw and kDrawSync .. kDrawRecord,
    /// only with WXL_VK_STATS: one QueryPerformanceCounter at the entry and one per phase
    /// boundary -- the last boundary is the slot's end as well -- nothing at all without the key.
    struct Phases
    {
        bool     on    = false;
        bool     ended = false;   ///< End() ran: `at` is the slot's end
        uint64_t t0    = 0;       ///< the slot's entry
        uint64_t at    = 0;       ///< the last boundary
        void Start()
        {
            on = wxlvk::native::perf::Enabled();
            if (on) t0 = at = wxlvk::native::perf::Now();
        }
        void Mark(wxlvk::native::perf::Bucket b)
        {
            if (!on) return;
            const uint64_t now = wxlvk::native::perf::Now();
            wxlvk::native::perf::Add(b, now - at);
            at = now;
        }
        /// The last phase: its boundary is the slot's end.
        void End(wxlvk::native::perf::Bucket b)
        {
            Mark(b);
            ended = true;
        }
        /// The whole slot into kDraw (counted per call, as the per-draw line divides by it).
        void Finish()
        {
            if (!on) return;
            const uint64_t end = ended ? at : wxlvk::native::perf::Now();
            wxlvk::native::perf::Add(wxlvk::native::perf::kDraw, end - t0);
        }
    };

    uint64_t g_draws;
    uint64_t g_noBatch;
    uint64_t g_noTable;
    uint64_t g_drawsEarly;
    uint64_t g_gatedContext;
    uint64_t g_gatedPoisoned;
    uint64_t g_targetDraws;    ///< recorded while the pass was on a texture target
    uint64_t g_noDrawPath;
    uint64_t g_noPass;
    uint64_t g_unresolved;
    uint64_t g_emptyViewport;
    uint64_t g_filtered;       ///< dropped by a rendering-API draw filter (native/Render.cpp)
    uint64_t g_recorded;

    /// The draws an engine occlusion query counts (between slots 64 and 65), with WXL_VK_STATS: the
    /// first kQueryDrawLines are described in full, so one run says what the glare's quad met.
    constexpr uint32_t kQueryDrawLines = 8;
    uint64_t g_queryDraws;
    uint64_t g_queryDrawn;
    uint64_t g_queryNoVkQuery;   ///< recorded with no Vulkan query open in the pass (counted by none)
    uint32_t g_queryDrawLogged;

    /**
     * One line for a draw recorded inside an engine occlusion query: what decides whether its
     * samples pass and are counted (the pass it lands in and its depth attachment, the depth and
     * colour state, the viewport's depth range, the material), and whether the pipelines issued it.
     *
     * @param Cmd& c : the draw as recorded
     * @param bool drawn : the pipelines issued it
     * @param bool vkQuery : a Vulkan query was open in the pass when it was recorded
     */
    void LogQueryDraw(const Cmd& c, bool drawn, bool vkQuery)
    {
        VkFormat       depthFormat = VK_FORMAT_UNDEFINED;
        const uint32_t depthKind   = present::PassDepthKind(depthFormat);
        static constexpr const char* kDepthKinds[] = { "none", "back depth", "depth texture", "offscreen depth" };
        uint32_t textures = 0;
        for (uint32_t t = 0; t < Cmd::kTextureStages; ++t)
            if (c.texture[t]) textures |= 1u << t;
        const VkExtent2D extent = present::Extent();
        wxl::ScriptMgr::Log(WXL_LOG_INFO,
                            "native: query draw %u: %s, %s; target %s %ux%u, depth attachment %s (format %d); "
                            "vs=\"%s\" ps=\"%s\" format=%u prim=%u %u index(es); depth test %u func %u write %u, "
                            "colour write 0x%X, blend %u, alpha ref %u, cull %u; viewport %.1f,%.1f %.1fx%.1f "
                            "depth %.6f..%.6f; ff %s, stage-0 ops %u/%u; stage images 0x%04X, engine-bound 0x%04X",
                            g_queryDrawLogged + 1, drawn ? "drawn" : "SKIPPED (see the skip census)",
                            vkQuery ? "inside a Vulkan query" : "NO Vulkan query open in the pass",
                            present::OnBackBuffer() ? "back buffer" : "texture", extent.width, extent.height,
                            kDepthKinds[depthKind < 4 ? depthKind : 0], static_cast<int>(depthFormat), c.vs, c.ps,
                            c.vertexFormat, c.topology, c.indexCount, c.depthTest ? 1u : 0u, c.depthFunc,
                            c.depthWrite ? 1u : 0u, c.colorWrite, c.blend, c.alphaRef, c.cull,
                            static_cast<double>(c.viewport.x), static_cast<double>(c.viewport.y),
                            static_cast<double>(c.viewport.width), static_cast<double>(c.viewport.height),
                            static_cast<double>(c.viewport.minDepth), static_cast<double>(c.viewport.maxDepth),
                            c.fixedFunction ? "vertex" : "off", c.ffColorOp, c.ffAlphaOp, textures, c.engineTexMask);
        ++g_queryDrawLogged;
    }

    // D3D9's pixel centres. D3D9 puts a pixel's centre at integer window coordinates, Vulkan at
    // half-integers; the engine lays its 2D out for D3D9 (the software cursor's own half-pixel shift
    // on api 1/2, cgxdevice 10.6, is its word on it). Every native draw's viewport is therefore moved
    // half a pixel right and down, so a vertex lands where D3D9 puts it -- DXVK's d3d9 does the same,
    // 1/128 short of the half so an edge on an exact centre keeps D3D's tie-break. The clear
    // rectangle and the scissor are integer pixel rects that mean the same pixels in both APIs, so
    // PlaceD3d stays as it is and only the draw's viewport moves. The same shift serves the
    // full-screen FFX passes, which the engine lays out for D3D9's texel-centre convention.
    using wxlvk::native::impl::kD3d9CentreShift;

    // D3D's constant REGISTERS. The engine's shadows (gx::kVsConstCache, kPsConstCache) are not what
    // a D3D shader reads: ShaderConstantsClear refills them with FLT_MAX at every frame start
    // (ISceneBegin) and slot 70 only marks a dirty range, which D3D's IShaderConstantsFlush
    // (0x006A9FE0, the first step of every draw's IStateSync) uploads into the device and then
    // resets. A register the engine has not re-sent since the frame began therefore holds FLT_MAX in
    // the shadow and its LAST UPLOADED value in D3D's registers, and the engine relies on that
    // wherever it sets a constant once and draws with it in later frames (a portrait drawn from a Lua
    // event early in a frame reads the fog row c30 and the pixel fog colour p2 the world set the frame
    // before). Native keeps the registers itself, flushes into them exactly as D3D does, and builds
    // every draw from them. D3D9 starts its registers at zero.
    constexpr uint32_t kRegs = gx::kVsConstRegisters;
    float    g_vsRegs[kRegs][4];
    float    g_psRegs[kRegs][4];
    uint64_t g_constFlushes;   ///< flushes that uploaded a vertex or pixel range
    /// One past the highest vertex register any flush has written this session. Rows at and above it
    /// still hold D3D's initial zero; a 256-row ring slice copies only up to it.
    uint32_t g_vsHighWater;
    /// Bumped by every flush that wrote a vertex register from c31 (Cmd::kPaletteBase) up: the
    /// bone-palette storage buffer is uploaded again only when it moved (Cmd::paletteSerial).
    uint64_t g_paletteSerial = 1;

    /// IShaderConstantsFlush, into g_vsRegs / g_psRegs: the dirty ranges [start, end] of both
    /// shadows, then the ranges reset to "none" (start 0xFF, end 0) as D3D leaves them.
    void FlushConstants()
    {
        uint32_t& vs0 = *reinterpret_cast<uint32_t*>(gx::kVsDirtyRegStart);
        uint32_t& vs1 = *reinterpret_cast<uint32_t*>(gx::kVsDirtyRegEnd);
        uint32_t& ps0 = *reinterpret_cast<uint32_t*>(gx::kPsDirtyRegStart);
        uint32_t& ps1 = *reinterpret_cast<uint32_t*>(gx::kPsDirtyRegEnd);
        bool any = false;
        if (vs0 <= vs1 && vs1 < kRegs)
        {
            wxlvk::simd::CopyRows(g_vsRegs[vs0], reinterpret_cast<const float*>(gx::kVsConstCache) + vs0 * 4u,
                                  vs1 - vs0 + 1u);
            if (vs1 + 1u > g_vsHighWater) g_vsHighWater = vs1 + 1u;
            if (vs1 >= Cmd::kPaletteBase) ++g_paletteSerial;
            any = true;
        }
        vs0 = 0xFFu;
        vs1 = 0u;
        if (ps0 <= ps1 && ps1 < kRegs)
        {
            wxlvk::simd::CopyRows(g_psRegs[ps0], reinterpret_cast<const float*>(gx::kPsConstCache) + ps0 * 4u,
                                  ps1 - ps0 + 1u);
            any = true;
        }
        ps0 = 0xFFu;
        ps1 = 0u;
        if (any) ++g_constFlushes;
    }

    /// The bufs a recorded draw bound, for the dynamic pools' drawn ranges: every enabled
    /// attribute's (one per stream, deduplicated as they repeat) and the index buf. Called only
    /// when the draw resolved a dynamic pool (impl::g_dynamicPoolResolved): the other pools keep
    /// no drawn ranges, and most draws bind none.
    void NoteBufsDrawn(const void* device, int indexed)
    {
        const uint32_t enabled = At<const uint32_t>(device, gxdev::kAttribEnabled);
        const void*    last    = nullptr;
        for (unsigned a = 0; a < gxdev::kAttribCount; ++a)
        {
            if (!(enabled & (1u << a))) continue;
            const void* buf = At<const void* const>(device, gxdev::kAttribBufs + a * sizeof(void*));
            if (buf && buf != last) wxlvk::native::impl::NoteBufDrawn(buf);
            last = buf;
        }
        if (!enabled) wxlvk::native::impl::NoteBufDrawn(At<const void* const>(device, gxdev::kAttribBufs));
        if (indexed) wxlvk::native::impl::NoteBufDrawn(At<const void* const>(device, gxdev::kIndexBuf));
    }

    inline uint32_t Rs(uintptr_t table, unsigned id)
    {
        return At<const uint32_t>(table + id * gxdev::kRsStride, gxdev::kRsValue);
    }

    // D3D's ISetTexGen (0x006A4AF0), the one piece of D3D's IRsSendToHw a no-op slot 1 cannot drop.
    // It fills the backend-internal texgen stacks 11..18 (cgxdevice 5.3 / 6.2) that IXformSetTex
    // multiplies with the application texture matrix; without it a TexGen 1 / 2 draw (the ground
    // decals: TexGen 2 on both stages, WorldLinear) would feed the camera-space position straight into
    // a texture matrix built for world space. Written here, at the same moment D3D writes them (the
    // sync of state 53 + s, which slot 41 forces on every view change for the modes 1 and 2), from
    // the World / View tops of that moment.

    /// The top of one CGxMatrixStack, honouring its identity flag as the reader's StackTop does.
    void StackRead(void* device, unsigned xform, float out[16])
    {
        const uintptr_t stack = reinterpret_cast<uintptr_t>(device) + gxdev::kMatrixStacks + xform * gxdev::kMatrixStackStride;
        uint32_t top = At<uint32_t>(stack, gxdev::kMatrixStackTop);
        if (top >= gxdev::kMatrixStackDepth) top = gxdev::kMatrixStackDepth - 1;
        if (At<uint32_t>(stack, gxdev::kMatrixStackFlags + top * sizeof(uint32_t)) & gxdev::kMatrixStackFlagIdentity)
        {
            for (unsigned i = 0; i < 16; ++i) out[i] = (i % 5 == 0) ? 1.0f : 0.0f;
            return;
        }
        std::memcpy(out, &At<float>(stack, gxdev::kMatrixStackBase + top * 16 * sizeof(float)), 16 * sizeof(float));
    }

    /// The affine inverse of a row-vector C44Matrix (v' = v * M: rows 0..2 the linear part, row 3
    /// the translation). A singular 3x3 gives the identity, rather than infinities in a texcoord.
    void AffineInverse(const float m[16], float out[16])
    {
        const float a = m[0], b = m[1], c = m[2];
        const float d = m[4], e = m[5], f = m[6];
        const float g = m[8], h = m[9], k = m[10];
        const float det = a * (e * k - f * h) - b * (d * k - f * g) + c * (d * h - e * g);
        for (unsigned i = 0; i < 16; ++i) out[i] = (i % 5 == 0) ? 1.0f : 0.0f;
        if (det == 0.0f) return;
        const float r = 1.0f / det;
        out[0] = (e * k - f * h) * r;  out[1] = (c * h - b * k) * r;  out[2]  = (b * f - c * e) * r;
        out[4] = (f * g - d * k) * r;  out[5] = (a * k - c * g) * r;  out[6]  = (c * d - a * f) * r;
        out[8] = (d * h - e * g) * r;  out[9] = (b * g - a * h) * r;  out[10] = (a * e - b * d) * r;
        const float tx = m[12], ty = m[13], tz = m[14];
        for (unsigned col = 0; col < 3; ++col)
            out[12 + col] = -(tx * out[col] + ty * out[4 + col] + tz * out[8 + col]);
        out[3] = out[7] = out[11] = 0.0f;
        out[15] = 1.0f;
    }

    /// Row-major out = a * b (the engine's order: World * View takes a vertex to view space).
    void MatMul(const float a[16], const float b[16], float out[16])
    {
        for (unsigned r = 0; r < 4; ++r)
            for (unsigned col = 0; col < 4; ++col)
                out[r * 4 + col] = a[r * 4] * b[col] + a[r * 4 + 1] * b[4 + col] + a[r * 4 + 2] * b[8 + col]
                                 + a[r * 4 + 3] * b[12 + col];
    }

    /**
     * ISetTexGen(stage, gen) re-expressed: the texgen matrix of 6.2's table into stack 11 + stage's
     * top, flagged identity or not, and the stack marked dirty as the D3D side leaves it.
     *
     *   1 ObjectLinear: inverse(View) * inverse(World) = inverse(World * View), camera -> object
     *   2 WorldLinear:  affineInverse(View), camera -> world
     *   6 SphereMap:    xy * 0.5 + 0.5
     *   0, 3, 4, 5:     identity
     */
    void SetTexGen(void* device, unsigned stage, uint32_t gen)
    {
        float m[16];
        for (unsigned i = 0; i < 16; ++i) m[i] = (i % 5 == 0) ? 1.0f : 0.0f;
        bool identity = true;
        if (gen == 1 || gen == 2)
        {
            float view[16];
            StackRead(device, gxdev::kXformView, view);
            if (gen == 1)
            {
                float world[16], worldView[16];
                StackRead(device, gxdev::kXformWorld, world);
                MatMul(world, view, worldView);
                AffineInverse(worldView, m);
            }
            else
                AffineInverse(view, m);
            identity = false;
        }
        else if (gen == 6)
        {
            m[0] = m[5] = 0.5f;
            m[12] = m[13] = 0.5f;
            identity = false;
        }

        const uintptr_t stack = reinterpret_cast<uintptr_t>(device) + gxdev::kMatrixStacks
                                + (gxdev::kXformTexGen0 + stage) * gxdev::kMatrixStackStride;
        uint32_t top = At<uint32_t>(stack, gxdev::kMatrixStackTop);
        if (top >= gxdev::kMatrixStackDepth) top = gxdev::kMatrixStackDepth - 1;
        std::memcpy(&At<float>(stack, gxdev::kMatrixStackBase + top * 16 * sizeof(float)), m, sizeof m);
        uint32_t& flags = At<uint32_t>(stack, gxdev::kMatrixStackFlags + top * sizeof(uint32_t));
        flags = identity ? (flags | gxdev::kMatrixStackFlagIdentity) : (flags & ~gxdev::kMatrixStackFlagIdentity);
        At<uint8_t>(stack, gxdev::kMatrixStackDirty) = 1;
    }

    inline int32_t Trunc(float v) { return static_cast<int32_t>(v); }

    inline int32_t Clamp(int32_t v, int32_t lo, int32_t hi) { return v < lo ? lo : (v > hi ? hi : v); }

    const wxlvk::draw::Resolver kResolver = {
        &wxlvk::native::impl::ResolvePool,
        &wxlvk::native::impl::ResolveTexture,
        &wxlvk::native::impl::g_shaderGeneration,
        g_vsRegs,
        g_psRegs,
    };

    /// Slot 42 past the gates, with its phases marked; the slot itself is the thin wrapper below.
    void DrawBody(void* device, void* edx, CGxBatch* batch, int indexed, Phases& phase)
    {
        namespace perf = wxlvk::native::perf;
        ++g_draws;
        if (phase.on && present::Count() == 0) ++g_drawsEarly;
        if (!batch)
        {
            ++g_noBatch;
            return;
        }

        // D3D's two gates.
        if (!At<uint32_t>(device, gxdev::kContext))
        {
            ++g_gatedContext;
            return;
        }
        if (At<uint32_t>(device, gxdev::kFramePoisoned))
        {
            ++g_gatedPoisoned;
            return;
        }
        Engine<gxdev::IRsSyncFn>(gxdev::kIRsSync)(device, edx, 0);
        // D3D's IStateSync begins with the constant flush; the registers move with it, every draw
        // past the gates, whether or not this side can record it.
        FlushConstants();
        phase.Mark(perf::kDrawSync);

        if (!present::DrawPathUp())
        {
            ++g_noDrawPath;
            return;
        }
        const uintptr_t table = At<const uintptr_t>(device, gxdev::kRsTable);
        if (!table)
        {
            ++g_noTable;
            return;
        }
        // A consumer of the rendering API drawing this content itself: the engine's state sync has
        // run, so the engine sees the draw as made, and nothing of it reaches the frame.
        if (wxlvk::native::render::AnyDrawFilter() && wxlvk::native::render::FilterDraw(table))
        {
            ++g_filtered;
            return;
        }

        // What the D3D binds do lazily: the shader creates, then slot 0 per bound texture. Before
        // the pass is asked for, because an upload into an image this frame already drew from ends
        // the pass (the transfer rule) and the draw must come after it.
        const unsigned shaderStates[2] = { gxdev::kRsVertexShader, gxdev::kRsPixelShader };
        for (unsigned id : shaderStates)
        {
            CGxShader* sh = reinterpret_cast<CGxShader*>(static_cast<uintptr_t>(Rs(table, id)));
            if (sh && At<uint32_t>(sh, gxdev::kShaderCreateTried) == 0)
                Own<slots::kIShaderCreate>(device)(device, edx, sh);
        }
        // The sixteen stages, read once here and handed to the reader. ISetTexture's slot-0 call
        // (PrepareTexture) is made only for a texture that has no record yet or has an update
        // pending -- its own early-out, taken here without the call, since nearly every bind is
        // of a texture already uploaded.
        wxlvk::draw::Stages stages;
        stages.mask = 0;
        for (unsigned stage = 0; stage < Cmd::kTextureStages; ++stage)
        {
            const uintptr_t tex = Rs(table, gxdev::kRsTexture0 + stage);
            stages.tex[stage]   = reinterpret_cast<const void*>(tex);
            if (!tex) continue;
            stages.mask |= 1u << stage;
            if (!At<const void* const>(tex, gxdev::kTexApiObject2) || At<const uint8_t>(tex, gxdev::kTexUpdatePending))
                wxlvk::native::impl::PrepareTexture(device, reinterpret_cast<void*>(tex));
        }

        VkCommandBuffer cmd = present::Pass();
        phase.Mark(perf::kDrawPrepare);
        if (!cmd)
        {
            ++g_noPass;
            return;
        }

        Cmd c;   // no constructor runs: the reader resets it (draw::Reset)
        wxlvk::native::impl::g_dynamicPoolResolved = false;
        if (!wxlvk::draw::BuildCmd(device, batch, indexed, kResolver, stages, c))
        {
            ++g_unresolved;
            return;
        }
        // The 256-row class's ring slice copies only the rows the engine has ever written; so does
        // its bone-palette upload, which is made again only when a flush moved the serial.
        c.highRows = g_vsHighWater > Cmd::kVertexRegs ? g_vsHighWater - Cmd::kVertexRegs : 0u;
        if (c.vconst == Cmd::kVertexRegsHigh)
        {
            c.paletteRows   = &g_vsRegs[Cmd::kPaletteBase];
            c.paletteCount  = g_vsHighWater > Cmd::kPaletteBase ? g_vsHighWater - Cmd::kPaletteBase : 0u;
            c.paletteSerial = g_paletteSerial;
        }
        if (!wxlvk::native::impl::PlaceD3d(device, present::Extent(), c.viewport, c.scissor))
        {
            ++g_emptyViewport;   // D3D draws nothing visible either
            return;
        }
        // D3D9's pixel centres (kD3d9CentreShift): the viewport moves, the scissor does not.
        c.viewport.x += kD3d9CentreShift;
        c.viewport.y += kD3d9CentreShift;
        wxlvk::native::render::jitter::Apply(c.viewport);
        phase.Mark(perf::kDrawBuild);
        // A draw an engine occlusion query counts: whether the pipelines issued it, for 67's result.
        const bool     inQuery     = present::QueryActive();
        const bool     vkQuery     = inQuery && present::QueryInPass();
        const uint64_t drawnBefore = inQuery ? pipelines::DrawnCount() : 0;
        pipelines::Record(cmd, present::RenderPass(), present::Slot(), c);
        if (inQuery)
        {
            const bool drawn = pipelines::DrawnCount() != drawnBefore;
            present::NoteQueryDraw(drawn);
            ++g_queryDraws;
            if (drawn) ++g_queryDrawn;
            if (!vkQuery) ++g_queryNoVkQuery;
            if (phase.on && g_queryDrawLogged < kQueryDrawLines) LogQueryDraw(c, drawn, vkQuery);
        }
        present::NoteDraw();
        if (wxlvk::native::impl::g_dynamicPoolResolved) NoteBufsDrawn(device, indexed);
        ++g_recorded;
        if (phase.on && !present::OnBackBuffer()) ++g_targetDraws;
        phase.End(perf::kDrawRecord);
    }
}

namespace wxlvk::native::impl
{
    uint64_t DrawsTotal() { return g_draws; }

    bool PlaceDraw(const void* device, const VkExtent2D& extent, VkViewport& viewport, VkRect2D& scissor)
    {
        if (!PlaceD3d(device, extent, viewport, scissor)) return false;
        viewport.x += kD3d9CentreShift;
        viewport.y += kD3d9CentreShift;
        wxlvk::native::render::jitter::Apply(viewport);   // the world's sub-pixel jitter (src/native/render/Jitter.hpp)
        return true;
    }

    bool PlaceD3d(const void* device, const VkExtent2D& extent, VkViewport& viewport, VkRect2D& scissor)
    {
        const float W = static_cast<float>(extent.width);
        const float H = static_cast<float>(extent.height);
        const float minX = At<const float>(device, gxdev::kViewportMinX);
        const float maxX = At<const float>(device, gxdev::kViewportMaxX);
        const float minY = At<const float>(device, gxdev::kViewportMinY);
        const float maxY = At<const float>(device, gxdev::kViewportMaxY);

        // IXformSetViewport, the back-buffer branch (y flipped: engine y up, D3D y down).
        const int32_t X  = Trunc(minX * W + 0.5f);
        const int32_t Y  = Trunc((1.0f - maxY) * H + 0.5f);
        int32_t       vw = Trunc(maxX * W - static_cast<float>(X) + 0.5f);
        int32_t       vh = Trunc((1.0f - minY) * H - static_cast<float>(Y) + 0.5f);
        const int32_t iw = static_cast<int32_t>(extent.width);
        const int32_t ih = static_cast<int32_t>(extent.height);
        // With a texture bound on buffer 0 or 1, D3D takes Y = round(minY * H) and keeps the flipped
        // height (IXformSetViewport 0x006A99E0 tests the two surfaces, +0x2918 / +0x2924, which
        // native never fills: the textures at +0x2910 / +0x291C say the same thing).
        const bool target =
            At<const void* const>(device, gxdev::kRenderTargets + gxdev::kBufferColor * gxdev::kRenderTargetStride
                                              + gxdev::kRenderTargetTex) != nullptr
            || At<const void* const>(device, gxdev::kRenderTargets + gxdev::kBufferDepth * gxdev::kRenderTargetStride
                                                 + gxdev::kRenderTargetTex) != nullptr;
        const int32_t Yp = target ? Trunc(minY * H + 0.5f) : Y;
        const int32_t x0 = Clamp(X, 0, iw), y0 = Clamp(Yp, 0, ih);
        const int32_t x1 = Clamp(X + vw, x0, iw), y1 = Clamp(Yp + vh, y0, ih);
        vw = x1 - x0;
        vh = y1 - y0;

        float minZ = At<const float>(device, gxdev::kViewportMinZ);
        float maxZ = At<const float>(device, gxdev::kViewportMaxZ);
        minZ = minZ < 0.0f ? 0.0f : (minZ > 1.0f ? 1.0f : minZ);
        maxZ = maxZ < 0.0f ? 0.0f : (maxZ > 1.0f ? 1.0f : maxZ);
        viewport.x        = static_cast<float>(x0);
        viewport.y        = static_cast<float>(y0);
        viewport.width    = static_cast<float>(vw);
        viewport.height   = static_cast<float>(vh);
        viewport.minDepth = minZ;
        viewport.maxDepth = maxZ;

        int32_t l = x0, t = y0, r = x1, b = y1;
        const uintptr_t table = At<const uintptr_t>(device, gxdev::kRsTable);
        if (table && Rs(table, gxdev::kRsScissorTest) != 0)
        {
            // IStateSyncScissorRect, back buffer; the CRect is {minY, minX, maxY, maxX}. D3D clips
            // to the viewport anyway, so the scissor here is the intersection of the two.
            const float* rect = &At<const float>(device, gxdev::kScissorRect);
            const int32_t sl = Clamp(Trunc(rect[1] * W + 0.5f), 0, iw);
            const int32_t sr = Clamp(Trunc(rect[3] * W + 1.0f), 0, iw);
            // A render target's rows are not flipped (cgxdevice 9.4).
            const int32_t st = Clamp(target ? Trunc(rect[0] * H + 0.5f) : Trunc((1.0f - rect[2]) * H + 0.5f), 0, ih);
            const int32_t sb = Clamp(target ? Trunc(rect[2] * H + 1.0f) : Trunc((1.0f - rect[0]) * H + 1.0f), 0, ih);
            l = l > sl ? l : sl;
            t = t > st ? t : st;
            r = r < sr ? r : sr;
            b = b < sb ? b : sb;
        }
        if (r < l) r = l;
        if (b < t) b = t;
        scissor.offset.x      = l;
        scissor.offset.y      = t;
        scissor.extent.width  = static_cast<uint32_t>(r - l);
        scissor.extent.height = static_cast<uint32_t>(b - t);
        return vw > 0 && vh > 0 && r > l && b > t;
    }

    void __fastcall IRsSendToHw(void* device, void* /*edx*/, uint32_t state)
    {
        // IRsSync calls this for each state whose app value differs from the shadow and then copies
        // the value into the shadow itself, so a no-op leaves the engine's dirty list and shadow
        // exactly where D3D would. The draw reads the app array.
        //
        // Except TexGen (53..60): D3D's switch runs ISetTexGen there, which writes ENGINE state the
        // draw then reads -- the texgen matrix stacks 11..18 (SetTexGen above).
        if (device && state >= gxdev::kRsTexGen0 && state < gxdev::kRsTexGen0 + 8u)
        {
            const uintptr_t table = At<const uintptr_t>(device, gxdev::kRsTable);
            if (table) SetTexGen(device, state - gxdev::kRsTexGen0, Rs(table, state));
        }
    }

    void __fastcall Draw(void* device, void* edx, CGxBatch* batch, int indexed)
    {
        // The body is a plain function so that the slot has no object with a destructor on its
        // frame: the timing is two calls around it, both nothing without WXL_VK_STATS.
        Phases phase;
        phase.Start();
        DrawBody(device, edx, batch, indexed, phase);
        phase.Finish();
    }

    void LogDrawStats()
    {
        pipelines::Stats p = {};
        pipelines::GetStats(p);
        present::Stats f = {};
        present::GetStats(f);
        wxl::ScriptMgr::Log(WXL_LOG_INFO,
                            "native: draws: %llu received (%llu before the first present); gated: no "
                            "context %llu, poisoned frame %llu; no draw path %llu, no pass %llu, "
                            "unresolved geometry %llu, empty viewport %llu, draw filter %llu; %llu recorded (%llu into a "
                            "texture target): %llu drawn, skipped unsupported %llu / no texture %llu / "
                            "not ready %llu / sampler type %llu / descriptors %llu / uniforms %llu; %u "
                            "pipeline(s) (%u translated)",
                            static_cast<unsigned long long>(g_draws),
                            static_cast<unsigned long long>(g_drawsEarly),
                            static_cast<unsigned long long>(g_gatedContext),
                            static_cast<unsigned long long>(g_gatedPoisoned),
                            static_cast<unsigned long long>(g_noDrawPath),
                            static_cast<unsigned long long>(g_noPass),
                            static_cast<unsigned long long>(g_unresolved),
                            static_cast<unsigned long long>(g_emptyViewport),
                            static_cast<unsigned long long>(g_filtered),
                            static_cast<unsigned long long>(g_recorded),
                            static_cast<unsigned long long>(g_targetDraws),
                            static_cast<unsigned long long>(p.drawn),
                            static_cast<unsigned long long>(p.skippedUnsupported),
                            static_cast<unsigned long long>(p.skippedNoTexture),
                            static_cast<unsigned long long>(p.skippedNotReady),
                            static_cast<unsigned long long>(p.skippedSamplerType),
                            static_cast<unsigned long long>(p.descriptorExhausted),
                            static_cast<unsigned long long>(p.uniformExhausted), p.pipelines,
                            p.translatedPipelines);
        // Every draw received and not drawn lands in exactly one bucket above; the remainder says
        // whether that still holds (0 when it does).
        const uint64_t notDrawn = g_draws - p.drawn;
        const uint64_t before   = g_noBatch + g_gatedContext + g_gatedPoisoned + g_noDrawPath + g_noTable + g_noPass
                                + g_unresolved + g_emptyViewport + g_filtered;
        const uint64_t inRecord = p.skippedUnsupported + p.skippedNoTexture + p.skippedNotReady + p.skippedSamplerType
                                + p.descriptorExhausted + p.uniformExhausted;
        const uint64_t named    = before + inRecord;
        wxl::ScriptMgr::Log(WXL_LOG_INFO,
                            "native: draws not drawn: %llu = %llu before the record (no batch %llu, no "
                            "state table %llu, the rest above) + %llu skipped by the record; %llu not "
                            "accounted for",
                            static_cast<unsigned long long>(notDrawn), static_cast<unsigned long long>(before),
                            static_cast<unsigned long long>(g_noBatch), static_cast<unsigned long long>(g_noTable),
                            static_cast<unsigned long long>(inRecord),
                            static_cast<unsigned long long>(notDrawn > named ? notDrawn - named : 0));
        if (g_queryDraws)
            wxl::ScriptMgr::Log(WXL_LOG_INFO,
                                "native: draws inside occlusion queries: %llu recorded, %llu drawn, %llu with no "
                                "Vulkan query open in the pass",
                                static_cast<unsigned long long>(g_queryDraws),
                                static_cast<unsigned long long>(g_queryDrawn),
                                static_cast<unsigned long long>(g_queryNoVkQuery));
        // Which keys every skip bucket's draws were, with their draw counts (WXL_VK_STATS; nothing
        // when no draw was skipped).
        pipelines::LogUnsupported();
        const auto pct = [](uint64_t part, uint64_t whole) {
            return whole ? 100.0 * static_cast<double>(part) / static_cast<double>(whole) : 0.0;
        };
        const uint64_t setDraws = p.setsWritten + p.setsReused;
        wxl::ScriptMgr::Log(WXL_LOG_INFO,
                            "native: binds: descriptor sets %llu written (%llu image descriptors, %llu "
                            "outside the full cache) / %llu from the slot's cache (%.1f%%), cache peak %u, "
                            "%llu flush(es); pipeline binds %llu / %llu skipped (%.1f%%); set binds %llu / %llu "
                            "skipped; buffer binds %llu / %llu skipped; dynamic state %llu / %llu skipped; pushes "
                            "%llu / %llu skipped; %u sampler(s), %u anisotropic",
                            static_cast<unsigned long long>(p.setsWritten),
                            static_cast<unsigned long long>(p.descriptorWrites),
                            static_cast<unsigned long long>(p.setsUncached),
                            static_cast<unsigned long long>(p.setsReused), pct(p.setsReused, setDraws), p.setCachePeak,
                            static_cast<unsigned long long>(p.setCacheFlushes),
                            static_cast<unsigned long long>(p.pipelineBinds),
                            static_cast<unsigned long long>(p.pipelineBindsSkipped),
                            pct(p.pipelineBindsSkipped, p.pipelineBinds + p.pipelineBindsSkipped),
                            static_cast<unsigned long long>(p.setBinds),
                            static_cast<unsigned long long>(p.setBindsSkipped),
                            static_cast<unsigned long long>(p.bufferBinds),
                            static_cast<unsigned long long>(p.bufferBindsSkipped),
                            static_cast<unsigned long long>(p.dynamicSets),
                            static_cast<unsigned long long>(p.dynamicSkipped),
                            static_cast<unsigned long long>(p.pushes),
                            static_cast<unsigned long long>(p.pushesSkipped), p.samplers, p.anisoSamplers);
        wxl::ScriptMgr::Log(WXL_LOG_INFO,
                            "native: frame: %llu submitted, %llu presented; pass breaks %llu in %llu "
                            "frame(s) (at most %u in one); %llu pass request(s) with no image; staging "
                            "%llu KiB handed out in %u chunk(s), %llu refused; retired %llu image(s) "
                            "and %llu buffer(s) by serial (%llu at once, waiting list peak %u, %llu "
                            "wait(s) for a serial), %llu leaked (list full)",
                            static_cast<unsigned long long>(f.frames),
                            static_cast<unsigned long long>(f.presented),
                            static_cast<unsigned long long>(f.passBreaks),
                            static_cast<unsigned long long>(f.framesBroken), f.maxBreaks,
                            static_cast<unsigned long long>(f.noImage),
                            static_cast<unsigned long long>(f.stagingBytes / 1024), f.stagingChunks,
                            static_cast<unsigned long long>(f.stagingFailed),
                            static_cast<unsigned long long>(f.retiredImages),
                            static_cast<unsigned long long>(f.retiredBuffers),
                            static_cast<unsigned long long>(f.retiredAtOnce), f.retirePeak,
                            static_cast<unsigned long long>(f.serialWaits),
                            static_cast<unsigned long long>(f.retireLeaked));
        wxl::ScriptMgr::Log(WXL_LOG_INFO,
                            "native: passes: %llu begun (%llu on a texture target), %llu with a CLEAR "
                            "load op (%llu first-use and %llu engine clear aspect(s) folded); %llu of the "
                            "%llu break(s) resumed on the same target; %llu image barrier(s) in %llu "
                            "call(s)",
                            static_cast<unsigned long long>(f.passes),
                            static_cast<unsigned long long>(f.targetPasses),
                            static_cast<unsigned long long>(f.clearLoadPasses),
                            static_cast<unsigned long long>(f.firstUseFolded),
                            static_cast<unsigned long long>(f.engineClearsFolded),
                            static_cast<unsigned long long>(f.breaksResumed),
                            static_cast<unsigned long long>(f.passBreaks),
                            static_cast<unsigned long long>(f.barriers),
                            static_cast<unsigned long long>(f.barrierCalls));

        const uint64_t ring      = p.class64 + p.class256;
        const uint64_t ringDraws = p.slicesReused + p.slicesWritten;
        wxl::ScriptMgr::Log(WXL_LOG_INFO,
                            "native: constants: %llu flush(es) into the registers, vertex high-water "
                            "row c%u; slice classes %llu x 64 rows / %llu x 256 rows (%.1f%% class "
                            "256); %llu ring draw(s): %llu reused the previous slice (%.1f%%), %llu "
                            "wrote one (%llu KiB), %llu high row(s) not copied; ring spills %llu (%u "
                            "buffer(s), at most %u in one frame), descriptor pool spills %llu (%u "
                            "pool(s)); depth bias %llu, blend factor %llu, multi-stream %llu, fixed "
                            "function on a covered layout %llu draw(s), with a disabled stage on the "
                            "idle-stage image %llu draw(s)",
                            static_cast<unsigned long long>(g_constFlushes),
                            g_vsHighWater ? g_vsHighWater - 1u : 0u,
                            static_cast<unsigned long long>(p.class64),
                            static_cast<unsigned long long>(p.class256),
                            ring ? 100.0 * static_cast<double>(p.class256) / static_cast<double>(ring) : 0.0,
                            static_cast<unsigned long long>(ringDraws),
                            static_cast<unsigned long long>(p.slicesReused),
                            ringDraws ? 100.0 * static_cast<double>(p.slicesReused) / static_cast<double>(ringDraws) : 0.0,
                            static_cast<unsigned long long>(p.slicesWritten),
                            static_cast<unsigned long long>(p.sliceBytes / 1024),
                            static_cast<unsigned long long>(p.highRowsSkipped),
                            static_cast<unsigned long long>(p.ringSpills), p.ringSpillBuffers,
                            p.ringMostBuffers ? p.ringMostBuffers : 1u,
                            static_cast<unsigned long long>(p.poolSpills), p.poolSpillPools,
                            static_cast<unsigned long long>(p.depthBiasDraws),
                            static_cast<unsigned long long>(p.blendFactorDraws),
                            static_cast<unsigned long long>(p.multiStreamDraws),
                            static_cast<unsigned long long>(p.coveredDraws),
                            static_cast<unsigned long long>(p.idleStageDraws));
        {
            // The record split (WXL_VK_RECORD_SPLIT): the six phases of pipelines::Record as microseconds a draw.
            double   us[6] = {}, mhz = 0.0;
            uint64_t timed = 0;
            if (pipelines::RecordSplit(us, timed, mhz))
                wxl::ScriptMgr::Log(WXL_LOG_INFO,
                                    "native: record split (%llu draw(s), us a draw): key and pipeline %.3f, textures / slice / "
                                    "set %.3f, constants %.3f, binds %.3f, shadow capture %.3f, draw call %.3f; all %.3f (TSC %.0f MHz)",
                                    static_cast<unsigned long long>(timed), us[0], us[1], us[2], us[3], us[4], us[5],
                                    us[0] + us[1] + us[2] + us[3] + us[4] + us[5], mhz);
        }
        uint64_t buildTicks = 0, builds = 0;
        pipelines::BuildCounters(buildTicks, builds);
        LARGE_INTEGER qpf = {};
        QueryPerformanceFrequency(&qpf);
        const double tickMs = qpf.QuadPart ? 1000.0 / static_cast<double>(qpf.QuadPart) : 0.0;
        wxl::ScriptMgr::Log(WXL_LOG_INFO,
                            "native: pipeline builds: %llu at a draw, %llu pre-warmed at bring-up in %.1f ms "
                            "(%u key(s) listed, %llu not built); pre-warmed at their first draw: %llu "
                            "verified, %llu rebuilt; %.1f ms building in all",
                            static_cast<unsigned long long>(p.buildsAtDraw),
                            static_cast<unsigned long long>(p.prewarmed),
                            static_cast<double>(p.prewarmTicks) * tickMs, p.keysLoaded,
                            static_cast<unsigned long long>(p.prewarmSkipped),
                            static_cast<unsigned long long>(p.prewarmVerified),
                            static_cast<unsigned long long>(p.prewarmRebuilt),
                            static_cast<double>(buildTicks) * tickMs);
        wxlvk::vk::wind::LogStats();
        const uint64_t paletteDraws = p.paletteUploads + p.paletteReused;
        wxl::ScriptMgr::Log(WXL_LOG_INFO,
                            "native: bone palettes: %llu draw(s) through the bone-palette variants (counted "
                            "in the 64-row class above): %llu palette upload(s) (%llu KiB), %llu reused "
                            "the slot's last palette (%.1f%%); %llu 256-row draw(s) on the 256-row module "
                            "because the palette ring was full",
                            static_cast<unsigned long long>(p.bonesDraws),
                            static_cast<unsigned long long>(p.paletteUploads),
                            static_cast<unsigned long long>(p.paletteBytes / 1024),
                            static_cast<unsigned long long>(p.paletteReused),
                            paletteDraws ? 100.0 * static_cast<double>(p.paletteReused) / static_cast<double>(paletteDraws) : 0.0,
                            static_cast<unsigned long long>(p.bonesRingFull));
        if (p.gbufferDraws || p.gbufferPipelines)
            wxl::ScriptMgr::Log(WXL_LOG_INFO,
                                "native: G-buffer: %llu draw(s) through the modules' G-buffer variants (%u pipeline(s) "
                                "built with them); %llu depth-writing draw(s) with no record of their own (%u pipeline(s), "
                                "each named once in a 'writes depth on the G-buffer pass with NO record' line)",
                                static_cast<unsigned long long>(p.gbufferDraws), p.gbufferPipelines,
                                static_cast<unsigned long long>(p.gbufferNullDraws), p.gbufferNullPipelines);

        // Frame pacing, between two slot-38 presents (after the engine's WaitForFPSCap).
        present::FrameTimes window = {}, session = {};
        present::GetFrameTimes(window, session, true);
        const auto pacing = [](const char* what, const present::FrameTimes& t) {
            wxl::ScriptMgr::Log(WXL_LOG_INFO,
                                "native: frame times (%s): %llu frame(s), min %.2f / avg %.2f / max %.2f "
                                "ms, %llu over 33 ms, %llu over 100 ms",
                                what, static_cast<unsigned long long>(t.frames), t.frames ? t.minMs : 0.0,
                                t.frames ? t.sumMs / static_cast<double>(t.frames) : 0.0, t.maxMs,
                                static_cast<unsigned long long>(t.slow),
                                static_cast<unsigned long long>(t.hitches));
        };
        pacing("since the last line", window);
        pacing("session", session);

        uint64_t allocs = 0, reserved = 0, inUse = 0;
        wxlvk::vk::memory::Stats(allocs, reserved, inUse);
        wxlvk::vk::memory::Detail md;
        wxlvk::vk::memory::GetDetail(md);
        constexpr uint64_t kMiB = 1024 * 1024;
        wxl::ScriptMgr::Log(WXL_LOG_INFO,
                            "native: memory: %llu vkAllocateMemory call(s), %llu MiB reserved (peak "
                            "%llu; %llu device-local), %llu MiB in use; %llu MiB mapped = address space "
                            "(peak %llu); %u block(s) live (%u dedicated, %u empty), %llu returned to "
                            "the driver (%llu MiB), %llu dedicated released; free-span lists: longest "
                            "%u, %llu growth(s)",
                            static_cast<unsigned long long>(allocs),
                            static_cast<unsigned long long>(reserved / kMiB),
                            static_cast<unsigned long long>(md.reservedPeak / kMiB),
                            static_cast<unsigned long long>(md.deviceLocal / kMiB),
                            static_cast<unsigned long long>(inUse / kMiB),
                            static_cast<unsigned long long>(md.mapped / kMiB),
                            static_cast<unsigned long long>(md.mappedPeak / kMiB), md.blocks,
                            md.dedicatedBlocks, md.emptyBlocks,
                            static_cast<unsigned long long>(md.returnedBlocks),
                            static_cast<unsigned long long>(md.returnedBytes / kMiB),
                            static_cast<unsigned long long>(md.dedicatedFreed), md.spanPeak,
                            static_cast<unsigned long long>(md.spanGrowths));
    }
}
