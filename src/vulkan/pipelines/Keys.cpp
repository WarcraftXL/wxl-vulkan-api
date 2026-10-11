// A draw's pipeline key, its hash, and the logs of what cannot be built.
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

namespace wxlvk::vk::pipelines::detail
{
    // --- the key ------------------------------------------------------------------------------

    /**
     * The pipeline key of a Cmd, built in place: every byte of the key is written, so the compare
     * (SameKey) and the hash (KeyHash) see no stale byte, and nothing is zeroed first.
     *
     * @param const Cmd& c : the draw
     * @param uint32 frameSlot : the slot it is recorded in (the palette ring's room)
     * @param bool windy : a tree-wind draw
     * @param bool gbuffer : the draw's pass is of the G-buffer class
     * @param Key& k : receives the key
     */
    void MakeKey(const Cmd& c, uint32_t frameSlot, bool windy, bool gbuffer, Key& k)
    {
        // The names are copied whole: the reader pads them with NUL to the field's full width
        // (src/draw/Reader.cpp ReadShader), and the last byte is always NUL there, so a name the
        // engine filled to the brim still logs cleanly. Fixed-size copies: inlined, no CRT call.
        static_assert(sizeof k.vs == sizeof c.vs && sizeof k.ps == sizeof c.ps, "the key's names are the Cmd's");
        std::memcpy(k.vs, c.vs, sizeof k.vs);
        std::memcpy(k.ps, c.ps, sizeof k.ps);
        k.vertexFormat = c.vertexFormat;
        k.blend        = c.blend;
        k.depthTest    = c.depthTest ? 1u : 0u;
        k.depthFunc    = c.depthFunc;
        k.depthWrite   = c.depthWrite ? 1u : 0u;
        k.cull         = c.cull;
        k.colorWrite   = c.colorWrite;
        k.topology     = c.topology;
        k.customStride = 0;
        k.customCount  = 0;
        for (uint32_t& w : k.custom) w = 0;
        for (uint32_t& w : k.streamStride) w = 0;
        if (c.vertexFormat == Cmd::kVertexFormatCustom)
        {
            k.customStride = c.vertexStride;
            k.customCount  = c.customCount <= Cmd::kMaxCustomElements ? c.customCount : Cmd::kMaxCustomElements;
            for (uint32_t i = 0; i < k.customCount; ++i)
                k.custom[i] = static_cast<uint32_t>(c.custom[i].attrib) | (static_cast<uint32_t>(c.custom[i].format) << 8) |
                              (static_cast<uint32_t>(c.custom[i].offset) << 16) |
                              (static_cast<uint32_t>(c.custom[i].stream) << 24);
            for (uint32_t i = 0; i < c.extraStreams && i < Cmd::kMaxStreams - 1; ++i)
                k.streamStride[i] = c.extraStride[i];
        }
        k.depthBias = c.depthBias != 0.0f ? 1u : 0u;
        // Format 0 (P) takes format 7's rows (CoverRow), white for the colour, so its textured
        // draws are keyed the same way.
        k.ffTextured = (c.fixedFunction && (c.vertexFormat == 7 || c.vertexFormat == 0) && c.ffColorOp < Cmd::kFfOpNone
                        && c.texture[0]) ? 1u : 0u;
        k.bones = 0;
        if (s.bonesReady && c.vconst == Cmd::kVertexRegsHigh && c.paletteRows)
        {
            if (PaletteFits(frameSlot, c)) k.bones = 1u;
            else                           ++s.stats.bonesRingFull;
        }
        if (windy) k.bones |= 2u;
        // The opaque world on a G-buffer class: a shader pair, depth tested and written, opaque or
        // alpha-keyed (blend 0 / 1), colour written -- the capture's own rule for a world draw.
        if (gbuffer && !c.fixedFunction && c.depthTest && c.depthWrite && c.blend <= 1u && c.colorWrite != 0u)
            k.bones |= 4u;
    }

    /**
     * Logs an unsupported key once, and counts the skip whether or not it was logged.
     *
     * @param Key& k : the key
     * @param string why : what about it cannot be drawn
     */
    void SkipUnsupported(const Key& k, const char* why)
    {
        ++s.stats.skippedUnsupported;

        for (uint32_t i = 0; i < s.skippedCount; ++i)
            if (SameKey(s.skipped[i], k)) return;

        if (s.skippedCount >= kMaxSkippedKeys)
        {
            if (!s.loggedSkippedFull)
            {
                s.loggedSkippedFull = true;
                wxl::ScriptMgr::Log(WXL_LOG_DEBUG,
                                    "vulkan: pipelines: %u distinct unsupported keys have been "
                                    "named; further ones are counted but not logged",
                                    kMaxSkippedKeys);
            }
            return;
        }
        s.skipped[s.skippedCount++] = k;

        wxl::ScriptMgr::Log(WXL_LOG_DEBUG,
                            "vulkan: pipelines: no pipeline for vs=\"%s\" ps=\"%s\" format=%u "
                            "blend=%u topology=%u cull=%u depth=%u/%u/%u write=0x%X: %s",
                            k.vs, k.ps, k.vertexFormat, k.blend, k.topology, k.cull, k.depthTest,
                            k.depthFunc, k.depthWrite, k.colorWrite, why);
    }

    /**
     * Logs an interface refusal of a translated pair once per (vs, ps, format).
     *
     * @param Key& k : the key
     * @param string what : the sentence
     * @param string semantic : the semantic it is about
     */
    void RefusePair(const Key& k, const char* what, const char* semantic)
    {
        for (uint32_t i = 0; i < s.refusalCount; ++i)
        {
            const PairRefusal& r = s.refusals[i];
            if (r.format == k.vertexFormat && std::strcmp(r.vs, k.vs) == 0 && std::strcmp(r.ps, k.ps) == 0)
                return;
        }
        if (s.refusalCount >= kMaxPairRefusals)
        {
            if (!s.loggedRefusalsFull)
            {
                s.loggedRefusalsFull = true;
                wxl::ScriptMgr::Log(WXL_LOG_WARN,
                                    "vulkan: pipelines: %u translated pairs have been refused by "
                                    "name; further refusals are counted only",
                                    kMaxPairRefusals);
            }
            return;
        }
        PairRefusal& r = s.refusals[s.refusalCount++];
        std::memcpy(r.vs, k.vs, sizeof r.vs);
        std::memcpy(r.ps, k.ps, sizeof r.ps);
        r.format = k.vertexFormat;
        wxl::ScriptMgr::Log(WXL_LOG_WARN,
                            "vulkan: pipelines: translated pair vs=\"%s\" ps=\"%s\" on format %u "
                            "refused: %s (%s)",
                            k.vs, k.ps, k.vertexFormat, what, semantic);
    }

    /**
     * Counts one skipped draw against its (bucket, key) in the census (State::unsupported),
     * with the reason and, for the pair's first draw, what the Cmd and the pipeline carried
     * beyond the key: the stages with an image against the stages the engine bound and the
     * slots the pipeline samples, the slot the skip was decided at, the fixed-function
     * selectors. Only with WXL_VK_STATS (State::census); the skip counters run regardless.
     *
     * @param uint32 bucket : the SkipBucket
     * @param Key& k : the key
     * @param Cmd& c : the draw
     * @param PipelineEntry* e : the pipeline, null when there is none
     * @param int32 slot : the slot the skip was decided at, -1 when none
     * @param uint32 detail : not ready: the image's layout; sampler type: 1 for a cube image
     * @param string why : the reason (static text); null reads as "unknown"
     */
    void CountSkip(uint32_t bucket, const Key& k, const Cmd& c, const PipelineEntry* e, int32_t slot,
                   uint32_t detail, const char* why)
    {
        if (!s.census) return;
        if (bucket >= kSkipBuckets) bucket = kSkipUnsupported;
        for (uint32_t i = 0; i < s.unsupportedCount; ++i)
        {
            State::Unsupported& u = s.unsupported[i];
            if (u.bucket == bucket && SameKey(u.key, k))
            {
                ++u.draws;
                return;
            }
        }
        if (s.unsupportedCount >= kMaxUnsupported)
        {
            ++s.unsupportedOther[bucket];
            return;
        }
        State::Unsupported& u = s.unsupported[s.unsupportedCount++];
        u.key    = k;
        u.why    = why ? why : "unknown";
        u.draws  = 1;
        u.bucket = bucket;
        u.ff     = (c.ffColorOp & 0xFFu) | ((c.ffAlphaOp & 0xFFu) << 8) | ((c.ffColorMode & 0xFFu) << 16)
                 | (c.fixedFunction ? 1u << 24 : 0u);
        if (c.fixedFunction)
        {
            // Stage 1's selectors as the reader wrote them (v[kFfStage1].xy), in bits 25..30.
            const float* st1 = c.vertexConst[Cmd::kFfStage1];
            const uint32_t cop1 = st1[0] >= 0.0f && st1[0] < 7.0f ? static_cast<uint32_t>(st1[0]) : 7u;
            const uint32_t aop1 = st1[1] >= 0.0f && st1[1] < 7.0f ? static_cast<uint32_t>(st1[1]) : 7u;
            u.ff |= (cop1 << 25) | (aop1 << 28);
        }
        else
            u.ff |= (7u << 25) | (7u << 28);
        u.textures = 0;
        for (uint32_t t = 0; t < Cmd::kTextureStages; ++t)
            if (c.texture[t]) u.textures |= 1u << t;
        u.engine   = c.engineTexMask;
        u.samplers = e ? e->writeMask : 0xFFFFFFFFu;
        u.cubes    = e ? e->cubeMask : 0u;
        u.slot     = slot;
        u.detail   = detail;
        u.source   = !e ? 4u : e->row ? 0u : e->companion ? 3u : e->ffVertex ? 2u : 1u;
    }

    /// The unsupported bucket: no pipeline, or one the draw cannot use.
    void CountUnsupported(const Key& k, const Cmd& c, const char* why, const PipelineEntry* e)
    {
        CountSkip(kSkipUnsupported, k, c, e, -1, 0, why);
    }
}
