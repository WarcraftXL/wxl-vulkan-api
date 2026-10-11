// The Vulkan half of a draw: pipelines, samplers, descriptor sets, and the one function that binds
// everything a draw::Cmd names and issues it.
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

#include <intrin.h>

using namespace wxlvk::vk::pipelines::detail;

namespace wxlvk::vk::pipelines
{
    bool RecordSplit(double (&usPerDraw)[6], uint64_t& draws, double& tscMhz)
    {
        if (!s.split || !s.splitDraws) return false;
        LARGE_INTEGER q = {}, f = {};
        QueryPerformanceCounter(&q);
        QueryPerformanceFrequency(&f);
        const double seconds = f.QuadPart ? static_cast<double>(q.QuadPart - s.splitQpc0) / static_cast<double>(f.QuadPart) : 0.0;
        if (seconds < 1.0) return false;
        const double hz = static_cast<double>(__rdtsc() - s.splitTsc0) / seconds;
        if (hz <= 0.0) return false;
        for (uint32_t i = 0; i < 6; ++i)
            usPerDraw[i] = static_cast<double>(s.splitTicks[i]) * 1.0e6 / hz / static_cast<double>(s.splitDraws);
        draws  = s.splitDraws;
        tscMhz = hz / 1.0e6;
        return true;
    }

    void BuildCounters(uint64_t& ticks, uint64_t& builds)
    {
        ticks  = s.buildTicks;
        builds = s.buildCount;
    }

    void PassCreated(VkRenderPass pass, VkFormat colour, VkFormat depth, bool gbuffer)
    {
        if (pass == VK_NULL_HANDLE) return;
        // The class's description first, whatever the state: Build() reads the G-buffer flag for every
        // pipeline of the pass, and a pass made before Init() would otherwise be taken for a plain one.
        if (s.passInfoCount < sizeof s.passInfo / sizeof s.passInfo[0])
        {
            State::PassInfo& p = s.passInfo[s.passInfoCount++];
            p.pass    = pass;
            p.colour  = static_cast<uint32_t>(colour);
            p.depth   = static_cast<uint32_t>(depth);
            p.gbuffer = gbuffer ? 1u : 0u;
        }
        if (!s.ready) return;
        Prewarm(pass, static_cast<uint32_t>(colour), static_cast<uint32_t>(depth), gbuffer);
    }

    VkPipelineCache PipelineCache() { return s.ready ? s.cache : VK_NULL_HANDLE; }

    void SavePipelineCache()
    {
        if (s.ready && s.buildCount != s.keysSavedAt) SaveKeys();
        if (!s.ready || s.cache == VK_NULL_HANDLE || !s.dapi->GetPipelineCacheData) return;
        if (s.buildCount == s.cacheSavedAt) return;   // nothing new since the load or the last save
        size_t size = 0;
        VkResult r = s.dapi->GetPipelineCacheData(s.device, s.cache, &size, nullptr);
        if (r != VK_SUCCESS || size == 0 || size > kCacheMaxBytes) return;
        std::vector<uint8_t> data(size);
        r = s.dapi->GetPipelineCacheData(s.device, s.cache, &size, data.data());
        if ((r != VK_SUCCESS && r != VK_INCOMPLETE) || size == 0) return;
        const std::string path = wxlvk::path::Resolve(kCacheFile);
        const std::string tmp  = path + ".tmp";
        bool ok = false;
        if (FILE* f = std::fopen(tmp.c_str(), "wb"))
        {
            ok = std::fwrite(data.data(), 1, size, f) == size;
            ok = std::fclose(f) == 0 && ok;
        }
        ok = ok && MoveFileExA(tmp.c_str(), path.c_str(), MOVEFILE_REPLACE_EXISTING) != 0;
        if (!ok)
        {
            DeleteFileA(tmp.c_str());
            wxl::ScriptMgr::Log(WXL_LOG_WARN, "vulkan: pipelines: the pipeline cache could not be written to %s",
                                path.c_str());
            s.cacheSavedAt = s.buildCount;   // not retried every line
            return;
        }
        s.cacheSavedAt = s.buildCount;
        wxl::ScriptMgr::Log(WXL_LOG_DEBUG, "vulkan: pipelines: pipeline cache saved: %llu byte(s), %u pipeline(s) "
                            "live (%llu byte(s) were loaded at bring-up)",
                            static_cast<unsigned long long>(size), s.pipelineCount - s.failedCount,
                            static_cast<unsigned long long>(s.cacheLoaded));
    }

    bool Init()
    {
        const context::Handles* h = context::Get();
        if (!h || !h->device || !h->dapi)
        {
            wxl::ScriptMgr::Log(WXL_LOG_WARN, "vulkan: pipelines: Init() without a live context");
            return false;
        }

        if (s.ready)
        {
            if (s.device == h->device) return true;

            // As in texture::Init(): a different VkDevice means ours is gone.
            wxl::ScriptMgr::Log(WXL_LOG_WARN,
                                "vulkan: pipelines: the context reports a different VkDevice than "
                                "the one these objects were built on; forgetting %u pipeline(s) and "
                                "%u sampler(s) rather than destroying them",
                                s.pipelineCount, s.samplerCount);
            s = State();
        }

        if (!HaveEntryPoints(*h->dapi)) return false;

        s.device     = h->device;
        s.dapi       = h->dapi;
        s.mode   = settings::TranslatedMode();

        const loader::Device& d = *s.dapi;

        // Set 1, binding 0: one slice of the uniform ring, both stages, since the vertex shaders
        // read regs.v and the fragment shaders regs.p and regs.misc. It is created first because
        // every pipeline layout names it; set 0's layouts come from LayoutFor() as pairs ask for
        // them, the hand-written rows' {s0} one right here so Init's line can name it.
        VkDescriptorSetLayoutBinding regsBinding = {};
        regsBinding.binding         = 0;
        regsBinding.descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;
        regsBinding.descriptorCount = 1;
        regsBinding.stageFlags      = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;

        VkDescriptorSetLayoutCreateInfo regsLayout = {};
        regsLayout.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        regsLayout.bindingCount = 1;
        regsLayout.pBindings    = &regsBinding;

        VkResult r = d.CreateDescriptorSetLayout(s.device, &regsLayout, nullptr, &s.regsLayout);
        if (r != VK_SUCCESS)
        {
            s.regsLayout = VK_NULL_HANDLE;
            Fail("vkCreateDescriptorSetLayout (set 1)", r);
            s = State();
            return false;
        }

        uint32_t rowLayout = 0;
        if (!LayoutFor(kRowSamplerMask, rowLayout))
        {
            DropLayoutsAndPools(d, s.device);
            s = State();
            return false;
        }

        // One pool per frame-in-flight slot, each large enough for a full frame of draws with one
        // set each and, on average, four samplers per set (the UI binds one, terrain nine; a
        // frame that exceeds the average is counted as descriptorExhausted, not drawn wrong).
        // Reset whole in BeginFrame(). No FREE_DESCRIPTOR_SET_BIT: sets are never freed one at a
        // time, and a pool without the bit is allowed to be a bump allocator.
        VkDescriptorPoolSize size = {};
        size.type            = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        size.descriptorCount = kSetsPerPool * 4;

        VkDescriptorPoolCreateInfo pool = {};
        pool.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        pool.flags         = 0;
        pool.maxSets       = kSetsPerPool;
        pool.poolSizeCount = 1;
        pool.pPoolSizes    = &size;

        for (uint32_t i = 0; i < kFrameSlots; ++i)
        {
            r = d.CreateDescriptorPool(s.device, &pool, nullptr, &s.pools[i]);
            if (r != VK_SUCCESS)
            {
                s.pools[i] = VK_NULL_HANDLE;
                Fail("vkCreateDescriptorPool", r);
                DropLayoutsAndPools(d, s.device);
                s = State();
                return false;
            }
        }

        if (!CreateRing(*h))
        {
            DropRing();
            DropLayoutsAndPools(d, s.device);
            s = State();
            return false;
        }

        // The constant vertex stream (kConstVbBytes), written once. Not fatal: without it only the
        // fixed-function draws on a layout no row is declared for go unbuilt, as before it existed.
        if (memory::CreateBuffer(kConstVbBytes, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, memory::Where::HostVisible, s.constVb)
            && s.constVb.block.mapped)
        {
            uint8_t* bytes = static_cast<uint8_t*>(s.constVb.block.mapped);
            std::memset(bytes, 0, kConstVbBytes);
            std::memset(bytes + kConstWhiteOffset, 0xFF, 4);
            memory::FlushHost(s.constVb.block);
        }
        else
        {
            memory::DestroyBuffer(s.constVb);
            s.constVb = memory::Buffer();
            wxl::ScriptMgr::Log(WXL_LOG_WARN,
                                "vulkan: pipelines: the constant vertex stream could not be made; fixed-function "
                                "draws on a layout no row is declared for are skipped");
        }

        // The idle-stage image (kFillerColour), made and cleared once and kept for the device's
        // life like the constant stream. Not fatal: without it a fixed-function draw whose stage 0
        // is disabled on a row that declares a sampler is skipped as "no texture".
        {
            texture::Image filler;
            filler.wanted = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            if (texture::CreateMipped(1, 1, 1, VK_FORMAT_R8G8B8A8_UNORM, filler))
            {
                if (texture::ClearAll(filler, kFillerColour)) s.filler = filler;
                else texture::DestroyIdle(filler);
            }
            if (s.filler.handle == VK_NULL_HANDLE)
                wxl::ScriptMgr::Log(WXL_LOG_WARN,
                                    "vulkan: pipelines: the idle-stage image could not be made; fixed-function "
                                    "draws with a disabled stage 0 on a row that declares a sampler are skipped");
        }
        s.census = settings::StatsWanted();
        s.split  = settings::RecordSplitWanted();
        if (s.split)
        {
            LARGE_INTEGER q = {};
            QueryPerformanceCounter(&q);
            s.splitQpc0 = q.QuadPart;
            s.splitTsc0 = __rdtsc();
        }

        // Not fatal: without it every 256-row draw keeps its 256-row module.
        if (settings::BonesWanted()) CreateBones(*h);
        else wxl::ScriptMgr::Log(WXL_LOG_DEBUG, "vulkan: pipelines: %s=0: 256-row draws keep the 256-row modules",
                                 settings::kBonesKey);

        if (h->samplerAnisotropy && h->iapi && h->iapi->GetPhysicalDeviceProperties)
        {
            VkPhysicalDeviceProperties props = {};
            h->iapi->GetPhysicalDeviceProperties(h->physical, &props);
            s.anisoMax = props.limits.maxSamplerAnisotropy;
        }

        // The world variant of the samplers (Samplers.cpp): the device's own anisotropy, within the
        // adapter's limit (none without samplerAnisotropy), and mip LOD bias, for the world's mipmapped
        // textures only.
        {
            const float asked = settings::AnisotropyWanted();
            s.worldAniso  = asked > 1.0f ? (asked < s.anisoMax ? asked : s.anisoMax) : 0.0f;
            if (s.worldAniso <= 1.0f) s.worldAniso = 0.0f;
            s.worldBias   = settings::MipBiasWanted();
            s.worldFilter = s.worldAniso > 1.0f || s.worldBias != 0.0f;
            char aniso[96];
            if (s.worldAniso > 1.0f)
                std::snprintf(aniso, sizeof aniso, "anisotropy %.0fx (%s %.0f, the adapter's limit %.0f)",
                              static_cast<double>(s.worldAniso), settings::kAnisotropyKey, static_cast<double>(asked),
                              static_cast<double>(s.anisoMax));
            else if (asked > 1.0f)
                std::snprintf(aniso, sizeof aniso, "no anisotropy (%s %.0f; the adapter has no samplerAnisotropy)",
                              settings::kAnisotropyKey, static_cast<double>(asked));
            else
                std::snprintf(aniso, sizeof aniso, "no anisotropy of the device's own (%s off)", settings::kAnisotropyKey);
            wxl::ScriptMgr::Log(WXL_LOG_INFO,
                                "vulkan: pipelines: the world's mipmapped textures: %s, mip LOD bias %+.2f (%s); the "
                                "interface, the text and render targets keep the engine's sampling",
                                aniso, static_cast<double>(s.worldBias), settings::kMipBiasKey);
        }

        LoadPipelineCache(*h);
        LoadKeys();

        s.ready = true;

        const char* modeName = s.mode == settings::kTranslatedOff    ? "0, hand-written rows only"
                             : s.mode == settings::kTranslatedPrefer ? "2, translated modules preferred over the rows"
                                                                     : "1, translated modules fill in where no row exists";
        wxl::ScriptMgr::Log(WXL_LOG_DEBUG,
                            "vulkan: pipelines: ready -- set 0 (combined image samplers by slot, "
                            "fragment; layouts cached by slot set, %u max), set 1 (dynamic uniform "
                            "buffer, vertex + fragment), a %u-byte vertex push range, %u descriptor "
                            "pool(s) of %u sets, caches of %u pipelines and %u samplers, %u "
                            "hand-written shader pair(s); %s=%s; manifest %s",
                            kMaxLayouts, kPushBytes, kFrameSlots, kSetsPerPool,
                            kMaxPipelines, kMaxSamplers, kShaderRowCount, settings::kTranslatedKey,
                            modeName, manifest::Loaded() ? "loaded" : "absent");
        return true;
    }

    void BeginFrame(uint32_t frameSlot)
    {
        if (!s.ready) return;
        wxlvk::vk::wind::BeginFrame(frameSlot);

        if (frameSlot >= kFrameSlots)
        {
            if (!s.loggedBadSlot)
            {
                s.loggedBadSlot = true;
                wxl::ScriptMgr::Log(WXL_LOG_WARN,
                                    "vulkan: pipelines: BeginFrame(%u) but only %u slot(s) exist "
                                    "(kFrameSlots); Present.cpp's kFramesInFlight has drifted. No "
                                    "draw will be recorded for this slot (logged once)",
                                    frameSlot, kFrameSlots);
            }
            return;
        }

        // The slot's set-0 sets and their cache live across its frames (kSetCacheSlots): the pool
        // is reset whole, and the cache forgotten with it, only once the cache is nearly full or
        // the last frame of the slot spilled past the pool -- everything allocated from either
        // belongs to frames whose fences the frame has waited on. The ring's bytes for this slot
        // are free again for the same reason, every frame.
        SetCache& cache = g_setCache[frameSlot];
        if (s.poolActive[frameSlot] != 0 || cache.count >= kSetCacheFlushEntries || cache.used >= kSetCacheFlushWords)
        {
            const VkResult r = s.dapi->ResetDescriptorPool(s.device, s.pools[frameSlot], 0);
            if (r != VK_SUCCESS && !s.loggedPoolReset)
            {
                s.loggedPoolReset = true;
                Fail("vkResetDescriptorPool (logged once)", r);
            }
            for (uint32_t i = 0; i < s.poolActive[frameSlot] && i < kPoolSpill; ++i)
                if (s.poolSpill[frameSlot][i] != VK_NULL_HANDLE)
                    s.dapi->ResetDescriptorPool(s.device, s.poolSpill[frameSlot][i], 0);
            s.poolActive[frameSlot] = 0;
            // The cached sets went with the pool reset: a new generation forgets them all.
            if (++cache.gen == 0)
            {
                for (SetEntry& e : cache.table) e = SetEntry();
                cache.gen = 1;
            }
            cache.used  = 0;
            cache.count = 0;
            ++s.stats.setCacheFlushes;
        }
        s.ringActive[frameSlot] = 0;
        s.ringCursor[frameSlot] = 0;
        s.lastSlice[frameSlot].valid = false;   // its bytes are about to be rewritten
        s.boneCursor[frameSlot]  = 0;
        s.lastPalette[frameSlot] = State::LastPalette();
        // A new command buffer: nothing bound is trusted.
        s.bound = State::Bound();
    }

    void InvalidateBindings() { s.bound = State::Bound(); }

    void LogUnsupported()
    {
        if (s.unsupportedCount == 0) return;
        uint64_t other = 0;
        for (uint32_t b = 0; b < kSkipBuckets; ++b) other += s.unsupportedOther[b];
        wxl::ScriptMgr::Log(WXL_LOG_INFO,
                            "native: skipped keys: %u (bucket, key) pair(s) named (at most %u), %llu skip(s) of "
                            "further pairs (unsupported %llu / no texture %llu / not ready %llu / sampler type "
                            "%llu / descriptors %llu / uniforms %llu)",
                            s.unsupportedCount, kMaxUnsupported, static_cast<unsigned long long>(other),
                            static_cast<unsigned long long>(s.unsupportedOther[kSkipUnsupported]),
                            static_cast<unsigned long long>(s.unsupportedOther[kSkipNoTexture]),
                            static_cast<unsigned long long>(s.unsupportedOther[kSkipNotReady]),
                            static_cast<unsigned long long>(s.unsupportedOther[kSkipSamplerType]),
                            static_cast<unsigned long long>(s.unsupportedOther[kSkipDescriptors]),
                            static_cast<unsigned long long>(s.unsupportedOther[kSkipUniforms]));
        static constexpr const char* kSourceNames[] = { "row", "modules", "ff vertex + module",
                                                        "module + ff companion", "none" };
        for (uint32_t i = 0; i < s.unsupportedCount; ++i)
        {
            const State::Unsupported& u = s.unsupported[i];
            const Key&                k = u.key;
            // The custom layout as attrib:format@offset/stream, the way Key packs it.
            char layout[160] = "";
            if (k.vertexFormat == Cmd::kVertexFormatCustom)
            {
                int n = std::snprintf(layout, sizeof layout, " (stride %u:", k.customStride);
                for (uint32_t e = 0; e < k.customCount && n > 0 && n < static_cast<int>(sizeof layout); ++e)
                    n += std::snprintf(layout + n, sizeof layout - n, " %u:%u@%u/%u", k.custom[e] & 0xFFu,
                                       (k.custom[e] >> 8) & 0xFFu, (k.custom[e] >> 16) & 0xFFu, k.custom[e] >> 24);
                if (n > 0 && n < static_cast<int>(sizeof layout)) std::snprintf(layout + n, sizeof layout - n, ")");
            }
            char samplers[48] = "none (no pipeline)";
            if (u.samplers != 0xFFFFFFFFu)
                std::snprintf(samplers, sizeof samplers, "0x%04X (cube 0x%04X)", u.samplers, u.cubes);
            char at[48] = "";
            if (u.slot >= 0)
            {
                if (u.bucket == kSkipNotReady)
                    std::snprintf(at, sizeof at, ", at slot %d (layout %u)", static_cast<int>(u.slot), u.detail);
                else if (u.bucket == kSkipSamplerType)
                    std::snprintf(at, sizeof at, ", at slot %d (image %s)", static_cast<int>(u.slot),
                                  u.detail ? "cube" : "2D");
                else
                    std::snprintf(at, sizeof at, ", at slot %d", static_cast<int>(u.slot));
            }
            // Selectors: 0..5 EGxTexOp, 6 the stage is disabled (no texture), 7 not carried.
            wxl::ScriptMgr::Log(WXL_LOG_INFO,
                                "native: skipped %s: %llu draw(s) vs=\"%s\" ps=\"%s\" format=%u%s prim=%u "
                                "blend=%u cull=%u depth=%u/%u/%u write=0x%X bias=%u ffTextured=%u bones=%u; "
                                "source %s; ff %s, stage-0 ops %u/%u, stage-1 ops %u/%u, colour mode %u; stage "
                                "images 0x%04X, engine-bound 0x%04X, pipeline samplers %s%s: %s",
                                kSkipBucketNames[u.bucket], static_cast<unsigned long long>(u.draws), k.vs, k.ps,
                                k.vertexFormat, layout, k.topology, k.blend, k.cull, k.depthTest, k.depthFunc,
                                k.depthWrite, k.colorWrite, k.depthBias, k.ffTextured, k.bones,
                                kSourceNames[u.source < 5 ? u.source : 4], ((u.ff >> 24) & 1u) ? "vertex" : "off",
                                u.ff & 0xFFu, (u.ff >> 8) & 0xFFu, (u.ff >> 25) & 7u, (u.ff >> 28) & 7u,
                                (u.ff >> 16) & 0xFFu, u.textures, u.engine, samplers, at, u.why);
        }
    }


    bool HandWrittenRow(const char* name, bool pixel)
    {
        if (!name || !*name) return false;
        for (const ShaderRow& r : kShaderRows)
            if (std::strcmp(pixel ? r.ps : r.vs, name) == 0) return true;
        return false;
    }

    uint64_t DrawnCount() { return s.stats.drawn; }

    void GetStats(Stats& out)
    {
        out = s.stats;
        out.pipelines           = s.pipelineCount - s.failedCount;
        out.translatedPipelines = s.translatedCount;
        out.companionPipelines  = s.companionCount;
        out.samplers            = s.samplerCount;
        out.layouts             = s.layoutCount;
    }

    // --- the shadow replay's sources (src/capture/ShadowPipelines.cpp builds the pipelines) ---------

    bool ShadowSourceFor(uint32_t entryIndex, bool alphaTest, ShadowSource& out)
    {
        if (!s.ready || entryIndex >= s.pipelineCount) return false;
        const PipelineEntry& e = s.pipelines[entryIndex];
        const manifest::Record* vr = e.vrec;
        if (!e.translated || !vr || e.pipeline == VK_NULL_HANDLE || e.pointList) return false;
        if (e.wind ? (e.bones ? !vr->shadowWindBones : !vr->shadowWind) : (e.bones ? !vr->shadowBones : !vr->shadow))
            return false;
        out = ShadowSource();
        VertexLayout layout;
        uint32_t     missing = 0;
        const char*  why     = "";
        if (!TranslatedVertexLayout(e.key, vr->inputMask, layout, missing, why)) return false;
        if (!Topology(e.key.topology, out.topology)) return false;
        static_assert(sizeof out.attrs / sizeof out.attrs[0] >= sizeof layout.attrs / sizeof layout.attrs[0],
                      "every attribute a translated layout declares");
        for (uint32_t i = 0; i < layout.count; ++i) out.attrs[i] = layout.attrs[i];
        out.attrCount = layout.count;
        // The draw's own streams, as Build() declares them.
        out.bindings[0] = { 0, layout.stride, VK_VERTEX_INPUT_RATE_VERTEX };
        out.bindingCount = 1;
        for (uint32_t i = 0; i < Cmd::kMaxStreams - 1 && e.key.streamStride[i] != 0; ++i)
        {
            out.bindings[out.bindingCount] = { out.bindingCount, e.key.streamStride[i], VK_VERTEX_INPUT_RATE_VERTEX };
            ++out.bindingCount;
        }
        // "vertex/<md5>.spv" (or its `bones`, `wind`, `windbones` variant) -> the same with ".shadow.spv".
        const char*  spv = e.wind ? (e.bones ? vr->windBones : vr->wind) : (e.bones ? vr->bones : vr->spv);
        const size_t len = std::strlen(spv);
        if (len < 4 || len + 16 >= sizeof out.vertPath) return false;
        std::snprintf(out.vertPath, sizeof out.vertPath, "translated/%.*s.shadow.spv", static_cast<int>(len - 4), spv);
        if (alphaTest && e.prec)
        {
            // A module that reads stage 0 at texcoord0 (the leaves, the cards) is cut out by the
            // replay's one-texture cutout; any other keeps its own module (the alpha test is in it).
            const uint32_t tc0 = 1u << manifest::kTexcoord0;
            const LayoutEntry& lay = s.layouts[e.layout];
            const bool cutout = (e.prec->samplerMask & 1u) && !(e.prec->cubeMask & 1u) && (e.prec->readsMask & tc0)
                                && (vr->outputMask & tc0) && (lay.mask & 1u);
            if (cutout) std::snprintf(out.fragPath, sizeof out.fragPath, "shadow_cutout_frag.spv");
            else std::snprintf(out.fragPath, sizeof out.fragPath, "translated/%s", e.prec->spv);
        }
        const LayoutEntry& le = s.layouts[e.layout];
        out.layout = e.wind ? WindLayoutFor(e.layout, e.bones, true) : (e.bones ? le.bones : le.pipeline);
        out.bones  = e.bones;
        out.wind   = e.wind;
        return out.layout != VK_NULL_HANDLE;
    }
}
