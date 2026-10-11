// The pipeline of a key: the hashed table, the pre-warmed entries' check.
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
    /**
     * The first draw of a pre-warmed entry: the sources it was built from (by the persisted md5s)
     * against what Source() resolves for its key now, through the shaders this session created.
     * Equal: the entry is the one a draw-time build would have made, and is kept. Different (a
     * manifest or a name that moved since the list was written): the pipeline, never bound, is
     * destroyed and the key built as any other, so a draw never runs on a module it would not have
     * been given.
     *
     * @param uint32 i : the entry
     */
    void VerifyPrewarmed(uint32_t i)
    {
        PipelineEntry& e = s.pipelines[i];
        e.prewarmed = false;
        Sources     now;
        const char* why = "";
        if (Source(e.key, now.row, now.vr, now.pr, now.ffVertex, now.companion, why) && now.row == e.row
            && now.vr == e.vrec && now.pr == e.prec && now.ffVertex == e.ffVertex && now.companion == e.companion)
        {
            ++s.stats.prewarmVerified;
            return;
        }

        PipelineEntry built;
        LARGE_INTEGER t0 = {}, t1 = {};
        QueryPerformanceCounter(&t0);
        const bool ok = Build(e.key, e.pass, built, why, nullptr);
        QueryPerformanceCounter(&t1);
        s.buildTicks += static_cast<uint64_t>(t1.QuadPart - t0.QuadPart);
        ++s.buildCount;
        ++s.stats.prewarmRebuilt;
        if (e.pipeline != VK_NULL_HANDLE) s.dapi->DestroyPipeline(s.device, e.pipeline, nullptr);
        if (e.translated && s.translatedCount) --s.translatedCount;
        if (e.companion && s.companionCount) --s.companionCount;
        const Key          key  = e.key;
        const VkRenderPass pass = e.pass;
        if (ok)
        {
            e      = built;
            e.key  = key;
            e.pass = pass;
            if (e.translated) ++s.translatedCount;
            if (e.companion) ++s.companionCount;
        }
        else
        {
            e      = PipelineEntry();
            e.key  = key;
            e.pass = pass;
            e.why  = why;
            ++s.failedCount;
        }
        s.stats.pipelines = s.pipelineCount - s.failedCount;
        wxl::ScriptMgr::Log(WXL_LOG_DEBUG,
                            "vulkan: pipelines: pre-warmed vs=\"%s\" ps=\"%s\" format=%u resolved to other "
                            "sources at its first draw; rebuilt%s",
                            key.vs, key.ps, key.vertexFormat, ok ? "" : " and failed (skipped like any failed key)");
    }

    /**
     * The cached pipeline for a key and pass, built on first use.
     *
     * @param Key& k : the key
     * @param VkRenderPass pass : the pass
     * @return pointer entry : null when it cannot be built; the skip is counted and the key
     *         logged once
     */
    const PipelineEntry* Pipeline(const Key& k, VkRenderPass pass, const char*& failed)
    {
        // An in-world frame is thousands of draws over a few hundred keys, mostly in runs: one
        // compare for the run, the scan only at a change.
        if (s.pipelineLast < s.pipelineCount && s.pipelines[s.pipelineLast].pass == pass
            && SameKey(s.pipelines[s.pipelineLast].key, k))
        {
            if (s.pipelines[s.pipelineLast].prewarmed) VerifyPrewarmed(s.pipelineLast);
            if (s.pipelines[s.pipelineLast].pipeline == VK_NULL_HANDLE)
            {
                failed = s.pipelines[s.pipelineLast].why;
                SkipUnsupported(k, "the build failed before (see the first line for this key)");
                return nullptr;
            }
            return &s.pipelines[s.pipelineLast];
        }
        // The index instead of a scan of the table.
        const uint32_t home = KeyHash(k, pass);
        for (uint32_t p = home; s.pipeHash[p] != 0; p = (p + 1) & (kPipeHashSlots - 1))
        {
            const uint32_t i = s.pipeHash[p] - 1;
            if (s.pipelines[i].pass == pass && SameKey(s.pipelines[i].key, k))
            {
                s.pipelineLast = i;
                if (s.pipelines[i].prewarmed) VerifyPrewarmed(i);
                if (s.pipelines[i].pipeline == VK_NULL_HANDLE)
                {
                    // A key whose build failed is remembered, not rebuilt per draw.
                    failed = s.pipelines[i].why;
                    SkipUnsupported(k, "the build failed before (see the first line for this key)");
                    return nullptr;
                }
                return &s.pipelines[i];
            }
        }

        // Before building: a key the cache will never accept should not be built and then thrown
        // away, and the failure should say which key it was.
        if (s.pipelineCount >= kMaxPipelines)
        {
            if (!s.loggedPipelinesFull)
            {
                s.loggedPipelinesFull = true;
                wxl::ScriptMgr::Log(WXL_LOG_WARN,
                                    "vulkan: pipelines: the cache holds %u pipelines (kMaxPipelines); "
                                    "further keys are skipped (logged once)",
                                    kMaxPipelines);
            }
            failed = "the pipeline cache is full";
            SkipUnsupported(k, failed);
            return nullptr;
        }

        PipelineEntry built;
        const char*   why = "";
        LARGE_INTEGER t0 = {}, t1 = {};
        QueryPerformanceCounter(&t0);
        const bool ok = Build(k, pass, built, why, nullptr);
        QueryPerformanceCounter(&t1);
        s.buildTicks += static_cast<uint64_t>(t1.QuadPart - t0.QuadPart);
        ++s.buildCount;
        ++s.stats.buildsAtDraw;
        if (!ok)
        {
            failed = why;
            SkipUnsupported(k, why);
            // Remembered as a failed entry (no pipeline), so the next draw of the key is a lookup and
            // not another build attempt.
            PipelineEntry& f = s.pipelines[s.pipelineCount];
            f      = PipelineEntry();
            f.key  = k;
            f.pass = pass;
            f.why  = why;
            uint32_t p = home;
            while (s.pipeHash[p] != 0) p = (p + 1) & (kPipeHashSlots - 1);
            s.pipeHash[p] = ++s.pipelineCount;
            ++s.failedCount;
            return nullptr;
        }

        PipelineEntry& e = s.pipelines[s.pipelineCount++];
        e      = built;
        e.key  = k;
        e.pass = pass;
        {
            uint32_t p = home;
            while (s.pipeHash[p] != 0) p = (p + 1) & (kPipeHashSlots - 1);
            s.pipeHash[p] = s.pipelineCount;   // dense index + 1
        }
        s.stats.pipelines = s.pipelineCount - s.failedCount;
        if (e.translated) s.stats.translatedPipelines = ++s.translatedCount;
        if (e.companion)  s.stats.companionPipelines  = ++s.companionCount;
        if (e.gbuf)       ++s.stats.gbufferPipelines;
        if (e.gbufNull)   ++s.stats.gbufferNullPipelines;

        // The same two tables Build() consulted, so the line says what the pipeline was given
        // and not what the key said. Build() succeeded, so neither lookup can fail here.
        VkCullModeFlags cullMode  = VK_CULL_MODE_NONE;
        VkFrontFace     frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
        const char*     cullName  = "?";
        CullState(k.cull, cullMode, frontFace, cullName);
        VkCompareOp depthOp   = VK_COMPARE_OP_LESS_OR_EQUAL;
        const char* depthName = "?";
        DepthOp(k.depthFunc, depthOp, depthName);

        char source[256];
        if (e.companion)
            std::snprintf(source, sizeof source,
                          "translated vertex %s + ff companion %s (%s%s), class %u, samplers 0x%04X%s",
                          e.vrec->spv, e.prec->spv, manifest::FirstName(*e.prec),
                          e.prec->pointSprite ? ", point_sprite" : (e.prec->fixedFog ? ", fixed_fog" : ""),
                          e.vconst, e.writeMask,
                          e.pointList ? ", point list (gl_PointSize)" : "");
        else if (e.translated)
            std::snprintf(source, sizeof source,
                          "translated modules %s + %s, class %u, samplers 0x%04X (cube 0x%04X)%s%s",
                          e.bones ? (e.wind ? e.vrec->windBones : e.vrec->bones) : (e.wind ? e.vrec->wind : e.vrec->spv),
                          e.prec->spv, e.vconst,
                          e.writeMask, e.cubeMask,
                          e.bones ? ", bone palette in set 2"
                                  : (e.wind ? ", tree wind in set 3" : (e.pointList ? ", point list (gl_PointSize)" : "")),
                          e.gbuf ? ", G-buffer variants" : "");
        else if (e.ffVertex)
            std::snprintf(source, sizeof source,
                          "fixed-function vertex stage %s + translated pixel module %s, samplers 0x%04X (cube 0x%04X)",
                          kFfSemanticVertStem, e.prec->spv, e.writeMask, e.cubeMask);
        else
            std::snprintf(source, sizeof source, "hand-written row, constants via %s",
                          e.usesRing ? "the uniform ring" : "push constants");
        if (k.vertexFormat == Cmd::kVertexFormatCustom)
        {
            // The custom layout the pipeline was laid out for, element by element, in the
            // engine's encodings (Cmd::VertexElement): a<attrib>:fmt<format>@<offset>.
            size_t n = std::strlen(source);
            int    w = std::snprintf(source + n, sizeof source - n, ", custom layout stride %u:", k.customStride);
            if (w > 0) n += static_cast<size_t>(w) < sizeof source - n ? static_cast<size_t>(w) : sizeof source - n - 1;
            for (uint32_t i = 0; i < k.customCount && n < sizeof source - 1; ++i)
            {
                w = std::snprintf(source + n, sizeof source - n, " a%u:fmt%u@%u", k.custom[i] & 0xFFu,
                                  (k.custom[i] >> 8) & 0xFFu, (k.custom[i] >> 16) & 0xFFu);
                if (w > 0) n += static_cast<size_t>(w) < sizeof source - n ? static_cast<size_t>(w) : sizeof source - n - 1;
            }
        }

        wxl::ScriptMgr::Log(WXL_LOG_DEBUG,
                            "vulkan: pipelines: built #%u: vs=\"%s\" ps=\"%s\" format=%u blend=%u "
                            "topology=%u write=0x%X, %s, cull %u -> %s, depth test "
                            "%s func %s write %s%s%s, %u stream(s)",
                            s.pipelineCount, k.vs, k.ps, k.vertexFormat, k.blend, k.topology,
                            k.colorWrite, source, k.cull, cullName, k.depthTest ? "on" : "off",
                            depthName, k.depthWrite ? "on" : "off",
                            e.dynamicBias ? ", dynamic depth bias" : "",
                            e.dynamicBlend ? ", dynamic blend constant" : "", e.streams);
        return &e;
    }
}
