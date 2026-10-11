// Recording one draw: its key and pipeline, what it reads (src/vulkan/pipelines/RecordInputs.cpp), its
// binds, the shadow capture, and the draw itself.
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

#include "vulkan/pipelines/Internal.hpp"

#include "native/render/WmoGroups.hpp"

#include <intrin.h>

#include <cmath>


using namespace wxlvk::vk::pipelines::detail;

// A draw goes through these phases, each of which may skip it (counted, and named once in the log):
//
//   Accept         the Cmd can be drawn at all, its key, its pipeline, the pipeline fits the Cmd
//   Textures       one image per sampler slot the pixel shader declares       (RecordInputs.cpp)
//   PlanSlice      the ring slice: reuse the previous one, or is there room   (RecordInputs.cpp)
//   DescriptorSet  set 0 from the frame's cache, or allocated and written      (RecordInputs.cpp)
//   WriteSlice / WritePalette   the constants and the bone palette written     (RecordInputs.cpp)
//   Bind           pipeline, dynamic state, buffers, sets, push values -- what changed since the
//                  previous draw only (State::Bound)
//   Capture        an opaque world draw noted for the shadow replay (src/capture/)
//   Issue          the draw
//
// Nothing is recorded into the command buffer before Bind, so a skip is whole.
namespace
{
    /// The Cmd can be drawn, and its pipeline is the one for it. False: skipped (counted).
    bool Accept(VkRenderPass pass, const Cmd& c, Draft& dr)
    {
        // Tree wind: the draw noted against the watched models; its push values when it is a windy one.
        dr.windDraw = wxlvk::vk::wind::Observe(c, dr.frameSlot);
        MakeKey(c, dr.frameSlot, dr.windDraw != nullptr, IsGBufferPass(pass), dr.key);
        Key& key = dr.key;

        // --- what cannot be drawn, decided before anything is bound -----------------------------
        if (c.vertexBuffer == VK_NULL_HANDLE || (c.indexed && c.indexBuffer == VK_NULL_HANDLE) || c.indexCount == 0
            || (!c.indexed && c.vertexOffset < 0) || !c.vertexRows)
        {
            if (!s.loggedNoBuffers)
            {
                s.loggedNoBuffers = true;
                wxl::ScriptMgr::Log(WXL_LOG_WARN, "vulkan: pipelines: a draw with no vertex buffer, no index buffer, no "
                                    "count, no vertex rows or a negative first vertex; skipped (logged once)");
            }
            ++s.stats.skippedUnsupported;
            CountUnsupported(key, c, "no vertex buffer, index buffer, count or vertex rows, or a negative first vertex");
            return false;
        }
        // Vulkan 1.0 wants a positive width and a positive height; a negative height is the
        // VK_KHR_maintenance1 flip this file does not use (the vertex shaders flip instead).
        if (c.viewport.width <= 0.0f || c.viewport.height <= 0.0f)
        {
            if (!s.loggedBadViewport)
            {
                s.loggedBadViewport = true;
                wxl::ScriptMgr::Log(WXL_LOG_WARN, "vulkan: pipelines: a draw with a %gx%g viewport; skipped (logged once)",
                                    static_cast<double>(c.viewport.width), static_cast<double>(c.viewport.height));
            }
            ++s.stats.skippedUnsupported;
            CountUnsupported(key, c, "an empty or negative viewport");
            return false;
        }

        // --- the pipeline ----------------------------------------------------------------------
        const char* failed = nullptr;
        dr.entry = Pipeline(key, pass, failed);
        if (!dr.entry && (key.bones & 2u))
        {
            // The wind key could not be had (no room, a failed build): the draw as it was, still.
            wxlvk::vk::wind::FellBack(failed);
            key.bones &= ~2u;
            dr.windDraw = nullptr;
            failed      = nullptr;
            dr.entry    = Pipeline(key, pass, failed);
        }
        if (!dr.entry && (key.bones & 4u))
        {
            // The G-buffer variant could not be had (no room in the module cache, a failed build): the pass
            // class's plain pipeline, the draw still drawn, its G-buffer record zeroed as any draw without
            // a variant (Build.cpp).
            static bool said = false;
            if (!said)
            {
                said = true;
                wxl::ScriptMgr::Log(WXL_LOG_WARN, "vulkan: pipelines: a G-buffer variant could not be had (%s); such draws take "
                                    "the class's plain pipeline (logged once)", failed ? failed : "?");
            }
            key.bones &= ~4u;
            failed   = nullptr;
            dr.entry = Pipeline(key, pass, failed);
        }
        if (!dr.entry)
        {
            CountUnsupported(key, c, failed);   // counted and logged by Pipeline()
            return false;
        }
        const PipelineEntry* entry = dr.entry;

        // A format this file lays out has one stride, and a Vulkan 1.0 pipeline bakes it in; a Cmd
        // whose stride disagrees cannot use the pipeline. Checked after the pipeline is resolved so
        // that an unknown key is named by Pipeline() and not lost here.
        if (c.vertexStride != entry->stride)
        {
            if (!s.loggedBadStride)
            {
                s.loggedBadStride = true;
                wxl::ScriptMgr::Log(WXL_LOG_WARN, "vulkan: pipelines: a format-%u draw with stride %u, not %u; the stride "
                                    "is baked into the pipeline, so it is skipped (logged once)",
                                    c.vertexFormat, c.vertexStride, entry->stride);
            }
            ++s.stats.skippedUnsupported;
            CountUnsupported(key, c, "the vertex stride is not the one the pipeline baked in", entry);
            return false;
        }

        // --- the slice class: the Cmd's rows must be the pipeline's class ------------------------
        //
        // Both come from the same registry lookup by vertex shader name, so a mismatch means the
        // registry answered differently at record time and at build time (a name resolved by the
        // manifest's name table rather than by a create). The draw is skipped rather than issued with
        // a stale palette.
        if (entry->vconst == Cmd::kVertexRegsHigh)
        {
            if (c.vconst != Cmd::kVertexRegsHigh)
            {
                if (!s.loggedClassMismatch)
                {
                    s.loggedClassMismatch = true;
                    wxl::ScriptMgr::Log(WXL_LOG_WARN, "vulkan: pipelines: vs=\"%s\" builds as the 256-row class but the "
                                        "reader recorded it as 64 rows; skipped (logged once)", c.vs);
                }
                ++s.stats.skippedUnsupported;
                CountUnsupported(key, c, "the pipeline is of the 256-row class and the draw of the 64-row one", entry);
                return false;
            }
            if (!c.vertexConstHigh)
            {
                if (!s.loggedNoHighRows)
                {
                    s.loggedNoHighRows = true;
                    wxl::ScriptMgr::Log(WXL_LOG_WARN, "vulkan: pipelines: a 256-row draw arrived without its high rows; "
                                        "skipped and counted under the ring (logged once)");
                }
                ++s.stats.uniformExhausted;
                CountSkip(kSkipUniforms, key, c, entry, -1, 0, "a 256-row draw without its high rows");
                return false;
            }
        }
        return true;
    }

    /**
     * Binds everything the draw needs, against State::Bound: a bind or a dynamic state equal to what
     * this command buffer already holds from the previous draw is not recorded again. The Vulkan rules
     * the tracking keeps:
     *   - binding a pipeline whose depth bias (blend constants) is static replaces that dynamic state,
     *     so the value must be set again for the next pipeline that has it dynamic; viewport and
     *     scissor are dynamic in every pipeline built here;
     *   - descriptor sets bound through another pipeline layout are disturbed (the set-0 layouts differ
     *     between layouts, and set 1 follows set 0), so a layout change rebinds both, and the push
     *     constants are pushed again with it.
     *
     * Offset 0 on every buffer, as the Cmd's contract says: the pool buffers are bound whole and
     * addressed the way D3D addresses them, by vertexOffset (BaseVertexIndex) and firstIndex (the index
     * buffer's byte offset in 16-bit units plus the batch start). A multi-stream layout's extra streams
     * too: D3D's stream-offset mode 0 (Cmd.hpp).
     *
     * @return bool ok : false when the draw is skipped after all (the streams disagree, a wind pipeline
     *         without its set)
     */
    bool Bind(VkCommandBuffer cmd, const Cmd& c, const Draft& dr, VkBuffer (&vbs)[Cmd::kMaxStreams + 1], uint32_t& vbCount)
    {
        const loader::Device&  d      = *s.dapi;
        const PipelineEntry*   entry  = dr.entry;
        const LayoutEntry&     layout = s.layouts[entry->layout];
        const uint32_t         frameSlot  = dr.frameSlot;
        const VkPipelineLayout pipeLayout = entry->bones ? (entry->wind ? layout.windBones : layout.bones)
                                                         : (entry->wind ? layout.wind : layout.pipeline);

        static const VkDeviceSize zeros[Cmd::kMaxStreams + 1] = {};
        vbs[0]  = c.vertexBuffer;
        vbCount = 1;
        for (uint32_t i = 0; i < c.extraStreams && vbCount < entry->streams; ++i) vbs[vbCount++] = c.extraBuffer[i];
        if (vbCount != entry->streams)
        {
            ++s.stats.skippedUnsupported;   // cannot happen: the key carries the strides
            CountUnsupported(dr.key, c, "the draw's streams are not the pipeline's", entry);
            return false;
        }
        // The constant stream after the draw's own (Build() left a binding for it).
        if (entry->constStream) vbs[vbCount++] = s.constVb.handle;

        State::Bound& b = s.bound;
        if (b.cmd != cmd)
        {
            b     = State::Bound();
            b.cmd = cmd;
        }

        if (b.pipeline != entry->pipeline)
        {
            d.CmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, entry->pipeline);
            b.pipeline = entry->pipeline;
            if (!entry->dynamicBias) b.bias = false;
            if (!entry->dynamicBlend) b.blend = false;
            ++s.stats.pipelineBinds;
        }
        else
            ++s.stats.pipelineBindsSkipped;

        if (!b.viewport || std::memcmp(&b.viewportValue, &c.viewport, sizeof c.viewport) != 0)
        {
            d.CmdSetViewport(cmd, 0, 1, &c.viewport);
            b.viewport      = true;
            b.viewportValue = c.viewport;
            ++s.stats.dynamicSets;
        }
        else
            ++s.stats.dynamicSkipped;
        if (!b.scissor || std::memcmp(&b.scissorValue, &c.scissor, sizeof c.scissor) != 0)
        {
            d.CmdSetScissor(cmd, 0, 1, &c.scissor);
            b.scissor      = true;
            b.scissorValue = c.scissor;
            ++s.stats.dynamicSets;
        }
        else
            ++s.stats.dynamicSkipped;
        if (entry->dynamicBias)
        {
            // D3DRS_DEPTHBIAS in Vulkan's r units (kDepthBiasUnits); no slope term, as D3D's
            // SLOPESCALEDEPTHBIAS is never set by the engine; no clamp (that needs a feature).
            const float bias = c.depthBias * kDepthBiasUnits;
            if (!b.bias || b.biasValue != bias)
            {
                d.CmdSetDepthBias(cmd, bias, 0.0f, 0.0f);
                b.bias      = true;
                b.biasValue = bias;
                ++s.stats.dynamicSets;
            }
            else
                ++s.stats.dynamicSkipped;
            ++s.stats.depthBiasDraws;
        }
        if (entry->dynamicBlend)
        {
            if (!b.blend || b.blendValue != c.blendFactor)
            {
                const float f[4] = { c.blendFactor, c.blendFactor, c.blendFactor, c.blendFactor };
                d.CmdSetBlendConstants(cmd, f);
                b.blend      = true;
                b.blendValue = c.blendFactor;
                ++s.stats.dynamicSets;
            }
            else
                ++s.stats.dynamicSkipped;
            ++s.stats.blendFactorDraws;
        }

        bool sameVbs = b.vbCount == vbCount;
        for (uint32_t i = 0; sameVbs && i < vbCount; ++i) sameVbs = b.vbs[i] == vbs[i];
        if (!sameVbs)
        {
            d.CmdBindVertexBuffers(cmd, 0, vbCount, vbs, zeros);
            b.vbCount = vbCount;
            for (uint32_t i = 0; i < vbCount; ++i) b.vbs[i] = vbs[i];
            ++s.stats.bufferBinds;
        }
        else
            ++s.stats.bufferBindsSkipped;
        if (entry->streams > 1) ++s.stats.multiStreamDraws;
        if (entry->constStream) ++s.stats.coveredDraws;
        if (dr.idleUsed) ++s.stats.idleStageDraws;

        if (b.layout != pipeLayout)
        {
            // Another layout: every set and the push range are rebound through it.
            b.layout = pipeLayout;
            b.set0   = VK_NULL_HANDLE;
            b.set1   = VK_NULL_HANDLE;
            b.set2   = VK_NULL_HANDLE;
            b.push   = false;
        }
        if (!entry->usesRing)
        {
            // The UI's path, as it shipped: set 0 and the first four registers pushed.
            if (b.set0 != dr.set)
            {
                d.CmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeLayout, 0, 1, &dr.set, 0, nullptr);
                b.set0 = dr.set;
                ++s.stats.setBinds;
            }
            else
                ++s.stats.setBindsSkipped;
            if (!b.push || !wxlvk::simd::EqualRows(b.pushValue, c.vertexRows, kPushBytes / 16))
            {
                d.CmdPushConstants(cmd, pipeLayout, VK_SHADER_STAGE_VERTEX_BIT, 0, kPushBytes, c.vertexRows);
                b.push = true;
                std::memcpy(b.pushValue, c.vertexRows, kPushBytes);
                ++s.stats.pushes;
            }
            else
                ++s.stats.pushesSkipped;
            return true;
        }

        const VkDescriptorSet ringSet = RingSet(frameSlot);
        if (b.set0 != dr.set)
        {
            // Set 0 and set 1 in one call; the one dynamic offset is set 1's slice.
            const VkDescriptorSet sets[2] = { dr.set, ringSet };
            d.CmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeLayout, 0, 2, sets, 1, &dr.dynamicOffset);
            ++s.stats.setBinds;
        }
        else if (b.set1 != ringSet || b.set1Offset != dr.dynamicOffset)
        {
            // The same textures: only the slice moves.
            d.CmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeLayout, 1, 1, &ringSet, 1, &dr.dynamicOffset);
            ++s.stats.setBinds;
        }
        else
            ++s.stats.setBindsSkipped;
        b.set0       = dr.set;
        b.set1       = ringSet;
        b.set1Offset = dr.dynamicOffset;

        if (entry->bones)
        {
            // Set 2, the palette: set 0 and 1 above leave it bound (same layout), so only a palette
            // change rebinds it.
            const VkDescriptorSet boneSet = s.bonesSets[frameSlot];
            if (b.set2 != boneSet || b.set2Offset != dr.paletteOffset)
            {
                d.CmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeLayout, 2, 1, &boneSet, 1,
                                        &dr.paletteOffset);
                b.set2       = boneSet;
                b.set2Offset = dr.paletteOffset;
                ++s.stats.setBinds;
            }
            else
                ++s.stats.setBindsSkipped;
            ++s.stats.bonesDraws;
        }
        if (entry->wind)
        {
            // Set 3 (the wind block and the per-vertex data) and the model's push values. The push
            // range is the UI's, unused by every other pair, so what the UI cached is stale. Never a
            // wind pipeline without its set: the draw is skipped rather than read nothing.
            const VkDescriptorSet windSet = wxlvk::vk::wind::Set(frameSlot);
            if (windSet == VK_NULL_HANDLE)
            {
                ++s.stats.skippedUnsupported;
                return false;
            }
            d.CmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeLayout, 3, 1, &windSet, 0, nullptr);
            const wxlvk::vk::wind::Draw still = {};
            d.CmdPushConstants(cmd, pipeLayout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(wxlvk::vk::wind::Draw),
                               dr.windDraw ? dr.windDraw : &still);
            b.push = false;
            if (dr.windDraw) wxlvk::vk::wind::Drawn();
        }
        return true;
    }

    /// A model-view 3 x 4 (three rows: view = (r0.v, r1.v, r2.v)) that is a real transform: its scale (the longest
    /// row) between 1/50 and 50 and a non-flat 3 x 3 (vulkan/Wind.cpp's test). 0 when it is not.
    float TransformScale(const float* r0, const float* r1, const float* r2)
    {
        const float s0 = r0[0] * r0[0] + r0[1] * r0[1] + r0[2] * r0[2];
        const float s1 = r1[0] * r1[0] + r1[1] * r1[1] + r1[2] * r1[2];
        const float s2 = r2[0] * r2[0] + r2[1] * r2[1] + r2[2] * r2[2];
        const float det = r0[0] * (r1[1] * r2[2] - r1[2] * r2[1]) - r0[1] * (r1[0] * r2[2] - r1[2] * r2[0])
                          + r0[2] * (r1[0] * r2[1] - r1[1] * r2[0]);
        const float big = s0 > s1 ? (s0 > s2 ? s0 : s2) : (s1 > s2 ? s1 : s2);
        if (!(big > 4e-4f && big < 2500.0f) || !(std::fabs(det) > 1e-6f)) return 0.0f;
        return std::sqrt(big);
    }

    /**
     * A model draw's bounding sphere in view space (the replay's culling, WXL_VK_REPLAY_CULL_FRUSTUM), conservative:
     * the model's own extent is not known here, so round where its transforms put the model's origin -- the static
     * model's c31..c33, or, skinned, every bone's rows (c31 + 3 i: the bone palette's high-water rows, stale ones
     * included, which only widens it) -- a reach of kModelReach yards times the largest scale (a doodad, a unit, a
     * mount, a large creature fit in it; a huge one past it would lose the part of its shadow that falls in a light
     * whose frustum misses its sphere). Radius < 0 (never culled) for a draw with no model rows or a sphere so wide the
     * test would be worth nothing.
     */
    void ModelBound(const Cmd& c, const PipelineEntry* entry, float out[4])
    {
        constexpr float    kModelReach = 30.0f;    ///< yd round a model's transforms its geometry is taken to stay within
        constexpr float    kMaxRadius  = 250.0f;   ///< past this the bound is dropped (unknown)
        constexpr uint32_t kMaxBones   = 64;
        out[0] = out[1] = out[2] = 0.0f;
        out[3] = -1.0f;
        if (!c.vertexRows || !entry->vrec) return;
        const bool skinned = (entry->vrec->inputMask & (1u << wxlvk::vk::manifest::kBlendWeight0)) != 0u;
        const float (*rows)[4] = c.vertexRows + Cmd::kPaletteBase;
        uint32_t bones = 1;
        if (skinned)
        {
            if (entry->bones && c.paletteRows && c.paletteCount >= 3)
            {
                rows  = c.paletteRows;
                bones = c.paletteCount / 3;
            }
            else
                bones = (Cmd::kVertexRegs - Cmd::kPaletteBase) / 3;
            if (bones > kMaxBones) bones = kMaxBones;
        }
        float lo[3] = { 3.0e38f, 3.0e38f, 3.0e38f }, hi[3] = { -3.0e38f, -3.0e38f, -3.0e38f };
        float scale = 0.0f;
        bool  any   = false;
        for (uint32_t b = 0; b < bones; ++b)
        {
            const float* r0 = rows[b * 3 + 0];
            const float* r1 = rows[b * 3 + 1];
            const float* r2 = rows[b * 3 + 2];
            const float  s  = TransformScale(r0, r1, r2);
            if (s <= 0.0f) continue;   // a hidden bone scaled to nothing, an unwritten row
            const float t[3] = { r0[3], r1[3], r2[3] };
            if (!std::isfinite(t[0]) || !std::isfinite(t[1]) || !std::isfinite(t[2])) continue;
            for (int k = 0; k < 3; ++k)
            {
                lo[k] = t[k] < lo[k] ? t[k] : lo[k];
                hi[k] = t[k] > hi[k] ? t[k] : hi[k];
            }
            scale = s > scale ? s : scale;
            any   = true;
        }
        if (!any) return;
        float half2 = 0.0f;
        for (int k = 0; k < 3; ++k)
        {
            out[k] = 0.5f * (lo[k] + hi[k]);
            const float h = 0.5f * (hi[k] - lo[k]);
            half2 += h * h;
        }
        const float radius = std::sqrt(half2) + kModelReach * (scale > 1.0f ? scale : 1.0f);
        if (radius < kMaxRadius) out[3] = radius;
    }

    /// The shadow replay's capture (src/capture/): an opaque world draw, as bound.
    void Capture(const Cmd& c, const Draft& dr, const VkBuffer* vbs, uint32_t vbCount)
    {
        const PipelineEntry* entry = dr.entry;
        const Key&           key   = dr.key;
        if (!wxlvk::capture::Open() || !entry->usesRing || !entry->translated || !entry->vrec) return;
        const bool variant = entry->wind ? (entry->bones ? entry->vrec->shadowWindBones : entry->vrec->shadowWind)
                                         : (entry->bones ? entry->vrec->shadowBones : entry->vrec->shadow);
        if (!variant || !key.depthWrite || !key.depthTest || key.blend > 1u || key.colorWrite == 0u || entry->pointList)
            return;
        wxlvk::capture::Draw cd = {};
        cd.entry       = static_cast<uint32_t>(entry - s.pipelines);
        cd.set0        = dr.set;
        cd.ring        = RingSet(dr.frameSlot);
        cd.ringOffset  = dr.dynamicOffset;
        cd.bones       = entry->bones ? s.bonesSets[dr.frameSlot] : VK_NULL_HANDLE;
        cd.bonesOffset = dr.paletteOffset;
        static_assert(sizeof cd.vbs / sizeof cd.vbs[0] >= Cmd::kMaxStreams, "every stream of a draw");
        for (uint32_t i = 0; i < vbCount && i < sizeof cd.vbs / sizeof cd.vbs[0]; ++i) cd.vbs[i] = vbs[i];
        cd.vbCount      = vbCount;
        cd.ib           = c.indexBuffer;
        cd.firstIndex   = c.firstIndex;
        cd.count        = c.indexCount;
        cd.vertexOffset = c.vertexOffset;
        cd.indexed      = c.indexed ? 1u : 0u;
        cd.alphaTest    = (key.blend == 1u || c.alphaRef != 0u) ? 1u : 0u;
        // A mover: a model draw (its vertex layout carries bone weights, which the terrain and the map
        // objects have not) of a UNIT -- a model one of whose draws wears a character's, a creature's or
        // an item's texture, which vulkan/Wind.cpp reads off every file texture's path (a body's own skin
        // is composited and has none) -- and not a swaying tree. The bone
        // palette marks the rest of what moves (capture/Replay.cpp). The world's doodads, batched or
        // not, are never movers: taking the unbatched ones for movers (the first rule of 2026-10-09)
        // kept them out of the cached maps and their shadows flickered with the camera. A skinned model
        // whose texture has no name at all -- a character's composited skin -- is a unit's too: its body's
        // draws come before its hair's in the frame and were taken for the world's (a snow cap on the
        // player's head, 2026-10-10).
        // The reader decided it (draw/Reader.cpp WriteMoverRow, the row the G-buffer variants read too).
        cd.live = c.pixelConst[Cmd::kMoverRow][0] > 0.5f && !entry->wind ? 1u : 0u;
        // The terrain's draws, for a replay that wants the ground alone (WXL_VK_REPLAY_TERRAIN_ONLY).
        cd.terrain = std::strncmp(c.vs, "Terrain", 7) == 0 ? 1u : 0u;
        // A static M2 doodad (the engine's M2 draws go through the Diffuse_ vertex family; vulkan/Wind.cpp reads them
        // the same way): what a DYNAMIC_ONLY replay adds with WXL_VK_REPLAY_WITH_DOODADS (wxl-vulkan-shadow's units'
        // map: the small doodads the light bake leaves out cast their own silhouette round the player).
        cd.doodad = cd.live == 0u && !entry->wind && std::strncmp(c.vs, "Diffuse_", 8) == 0 ? 1u : 0u;
        // A map object's draw (every MapObj vertex family, exterior and interior groups): what a DYNAMIC_ONLY replay adds
        // with WXL_VK_REPLAY_WITH_MAPOBJECTS (the units' map round the player: the roofs' real shadow, 2026-10-11).
        cd.mapobj = cd.live == 0u && !entry->wind && std::strncmp(c.vs, "MapObj", 6) == 0 ? 1u : 0u;
        // Its bound for the replay's culling: the units' and the M2 doodads' (and the swaying trees'); a map object's its
        // group's box (the engine's group renderer runs round this draw: native/render/WmoGroups.cpp); none for the terrain.
        cd.bound[3] = -1.0f;
        if (cd.live || cd.doodad || entry->wind)
        {
            // A bone-palette draw's bound is a function of its palette (the serial moves whenever a row from c31 up
            // changes, WritePalette's own test): a model's batches share it, so it is computed once for the run.
            const bool palette = entry->bones && c.paletteRows && c.paletteCount >= 3
                                 && (entry->vrec->inputMask & (1u << wxlvk::vk::manifest::kBlendWeight0)) != 0u;
            if (palette && s.boundValid && s.boundSerial == c.paletteSerial && s.boundCount == c.paletteCount)
                std::memcpy(cd.bound, s.boundValue, sizeof cd.bound);
            else
            {
                ModelBound(c, entry, cd.bound);
                s.boundValid = palette;
                if (palette)
                {
                    s.boundSerial = c.paletteSerial;
                    s.boundCount  = c.paletteCount;
                    std::memcpy(s.boundValue, cd.bound, sizeof cd.bound);
                }
            }
        }
        else if (cd.mapobj && c.vertexRows) wxlvk::native::render::wmogroups::DrawBound(c.vertexRows, cd.bound);
        if (entry->wind)
        {
            // A swaying tree's shadow sways with it: the same set 3 and the same model values.
            static_assert(sizeof cd.windPush == sizeof(wxlvk::vk::wind::Draw), "the wind's push values");
            const wxlvk::vk::wind::Draw still = {};
            cd.wind    = 1u;
            cd.windSet = wxlvk::vk::wind::Set(dr.frameSlot);
            std::memcpy(cd.windPush, dr.windDraw ? dr.windDraw : &still, sizeof cd.windPush);
        }
        wxlvk::capture::Note(cd);
    }

    /// The draw itself, and its counts.
    void Issue(VkCommandBuffer cmd, const Cmd& c, const PipelineEntry* entry)
    {
        const loader::Device& d = *s.dapi;
        State::Bound&         b = s.bound;
        if (c.indexed)
        {
            if (b.ib != c.indexBuffer)
            {
                d.CmdBindIndexBuffer(cmd, c.indexBuffer, 0, VK_INDEX_TYPE_UINT16);
                b.ib = c.indexBuffer;
                ++s.stats.bufferBinds;
            }
            else
                ++s.stats.bufferBindsSkipped;
            d.CmdDrawIndexed(cmd, c.indexCount, 1, c.firstIndex, c.vertexOffset, 0);
        }
        else
        {
            // indexCount is the vertex count for a non-indexed draw, and vertexOffset the first vertex;
            // its sign was checked by Accept().
            d.CmdDraw(cmd, c.indexCount, 1, static_cast<uint32_t>(c.vertexOffset), 0);
        }

        ++s.stats.drawn;
        if (entry->vconst == Cmd::kVertexRegsHigh) ++s.stats.class256;
        else                                        ++s.stats.class64;
        if (entry->companion) ++s.stats.companionDraws;
        if (entry->gbuf) ++s.stats.gbufferDraws;
        else if (entry->gbufNull) ++s.stats.gbufferNullDraws;
    }
}

namespace wxlvk::vk::pipelines
{
    void Record(VkCommandBuffer cmd, VkRenderPass pass, uint32_t frameSlot, const Cmd& c)
    {
        if (!s.ready || !cmd || pass == VK_NULL_HANDLE)
        {
            if (!s.loggedNotReady)
            {
                s.loggedNotReady = true;
                wxl::ScriptMgr::Log(WXL_LOG_WARN, "vulkan: pipelines: Record() %s; every draw is skipped (logged once)",
                                    !s.ready ? "before Init()" : "with a null command buffer or pass");
            }
            ++s.stats.skippedUnsupported;
            return;
        }
        if (frameSlot >= kFrameSlots)
        {
            ++s.stats.skippedUnsupported;   // BeginFrame has logged it
            return;
        }

        Draft dr;
        dr.frameSlot = frameSlot;
        if (s.split)
        {
            // The same phases, each timed with the time-stamp counter (WXL_VK_RECORD_SPLIT): what a draw's recording
            // costs, phase by phase, for the stats block. A skipped draw files what it spent.
            uint64_t t = __rdtsc(), n;
            const auto mark = [&t, &n](uint32_t phase) { n = __rdtsc(); s.splitTicks[phase] += n - t; t = n; };
            ++s.splitDraws;
            const bool accepted = Accept(pass, c, dr);
            mark(0);
            if (!accepted) return;
            const bool inputs = Textures(c, dr) && PlanSlice(c, dr) && DescriptorSet(c, dr);
            mark(1);
            if (!inputs) return;
            WriteSlice(dr);
            WritePalette(c, dr);
            mark(2);
            VkBuffer vbs[Cmd::kMaxStreams + 1] = {};
            uint32_t vbCount = 0;
            const bool bound = Bind(cmd, c, dr, vbs, vbCount);
            mark(3);
            if (!bound) return;
            Capture(c, dr, vbs, vbCount);
            mark(4);
            Issue(cmd, c, dr.entry);
            mark(5);
            return;
        }
        if (!Accept(pass, c, dr) || !Textures(c, dr) || !PlanSlice(c, dr) || !DescriptorSet(c, dr)) return;
        WriteSlice(dr);
        WritePalette(c, dr);

        VkBuffer vbs[Cmd::kMaxStreams + 1] = {};
        uint32_t vbCount = 0;
        if (!Bind(cmd, c, dr, vbs, vbCount)) return;
        Capture(c, dr, vbs, vbCount);
        Issue(cmd, c, dr.entry);
    }
}
