// The persisted VkPipelineCache and key list, and the pre-warm at bring-up.
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
     * The VkPipelineCache, seeded from pipeline-cache.bin when the file's header
     * (VkPipelineCacheHeaderVersionOne) names this driver's vendor, device and pipelineCacheUUID.
     * A file from another driver or GPU is not handed over at all (a driver must reject it, but the
     * check costs nothing and makes the line say why). Without a cache pipelines are built all the same.
     */
    void LoadPipelineCache(const context::Handles& h)
    {
        if (!s.dapi->CreatePipelineCache || !h.iapi || !h.iapi->GetPhysicalDeviceProperties) return;
        VkPhysicalDeviceProperties props = {};
        h.iapi->GetPhysicalDeviceProperties(h.physical, &props);

        std::vector<uint8_t> data;
        const std::string    path = wxlvk::path::Resolve(kCacheFile);
        if (FILE* f = std::fopen(path.c_str(), "rb"))
        {
            std::fseek(f, 0, SEEK_END);
            const long size = std::ftell(f);
            std::fseek(f, 0, SEEK_SET);
            if (size > 0 && static_cast<size_t>(size) <= kCacheMaxBytes)
            {
                data.resize(static_cast<size_t>(size));
                if (std::fread(data.data(), 1, data.size(), f) != data.size()) data.clear();
            }
            std::fclose(f);
            if (data.empty()) s.cacheRejected = "unreadable or larger than the limit";
        }
        else
            s.cacheRejected = "no file yet";

        if (!data.empty())
        {
            uint32_t head[4] = {};
            if (data.size() < 16 + VK_UUID_SIZE)
                s.cacheRejected = "shorter than its header";
            else
            {
                std::memcpy(head, data.data(), sizeof head);
                if (head[0] < 16 + VK_UUID_SIZE || head[1] != VK_PIPELINE_CACHE_HEADER_VERSION_ONE)
                    s.cacheRejected = "not a version-one header";
                else if (head[2] != props.vendorID || head[3] != props.deviceID)
                    s.cacheRejected = "another vendor or device";
                else if (std::memcmp(data.data() + 16, props.pipelineCacheUUID, VK_UUID_SIZE) != 0)
                    s.cacheRejected = "another driver build (pipelineCacheUUID)";
            }
            if (s.cacheRejected) data.clear();
        }

        VkPipelineCacheCreateInfo info = {};
        info.sType           = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;
        info.initialDataSize = data.size();
        info.pInitialData    = data.empty() ? nullptr : data.data();
        VkResult r = s.dapi->CreatePipelineCache(s.device, &info, nullptr, &s.cache);
        if (r != VK_SUCCESS && !data.empty())
        {
            s.cacheRejected = "refused by the driver";
            data.clear();
            info.initialDataSize = 0;
            info.pInitialData    = nullptr;
            r = s.dapi->CreatePipelineCache(s.device, &info, nullptr, &s.cache);
        }
        if (r != VK_SUCCESS)
        {
            s.cache = VK_NULL_HANDLE;
            Fail("vkCreatePipelineCache (pipelines are built without one)", r);
            return;
        }
        s.cacheLoaded  = data.size();
        s.cacheSavedAt = s.buildCount;
        wxl::ScriptMgr::Log(WXL_LOG_DEBUG,
                            "vulkan: pipelines: pipeline cache %s (%s): %llu byte(s) loaded for %s, "
                            "device 0x%04X; saved there with the stats lines and at exit",
                            data.empty() ? "started empty" : "seeded from disk", path.c_str(),
                            static_cast<unsigned long long>(data.size()), props.deviceName, props.deviceID);
        if (data.empty() && s.cacheRejected)
            wxl::ScriptMgr::Log(WXL_LOG_DEBUG, "vulkan: pipelines: pipeline cache file not used: %s",
                                s.cacheRejected);
    }

    // --- the persisted key list (pipeline-keys.bin): the pre-warm's input -------------------------
    //
    // The VkPipelineCache makes a build cheap, not free: a zone's first frames still create every
    // pipeline they meet, a fraction of a millisecond each, and those land in the frames that are
    // already the slowest. So the keys this session built are written beside the cache -- each with
    // its pass class's formats and the md5s of the modules it was built from -- and the next
    // session builds them as each pass class is made (PassCreated), at bring-up. A pre-warmed
    // entry is checked at its first draw against Source() (VerifyPrewarmed), so a list written
    // against another manifest or another binding of a name costs a rebuild, never a wrong module.

    constexpr const char* kKeysFile    = "pipeline-keys.bin";
    constexpr uint32_t    kKeysMagic   = 0x4B505857u;   // "WXPK"
    constexpr uint32_t    kKeysVersion = 2;   ///< 2: the G-buffer class bit

    enum KeyKind : uint32_t
    {
        kKindRow       = 0,   ///< a hand-written row
        kKindModules   = 1,   ///< a translated vertex and pixel module
        kKindFfVertex  = 2,   ///< the semantic ff.vert and a translated pixel module
        kKindCompanion = 3,   ///< a translated vertex module and its generated ff companion
    };

    struct KeysHeader
    {
        uint32_t magic, version, keyBytes, entryBytes, count, mode, records, reserved;
    };

    struct PersistedKey
    {
        Key      key;
        uint32_t colour;
        uint32_t depth;
        uint32_t kind;
        uint32_t gbuffer;   ///< the class carried the G-buffer's attachments (pipelines::PassCreated)
        char     vmd5[33];
        char     pmd5[33];
        char     pad[2];
    };
    static_assert(sizeof(PersistedKey) % 4 == 0, "the persisted key is written as it is");

    /// The list read at Init; kept for the passes made later in the session.
    std::vector<PersistedKey> g_keys;

    uint32_t ManifestRecords()
    {
        manifest::Stats m = {};
        manifest::GetStats(m);
        return m.records;
    }

    /// Reads pipeline-keys.bin when its header matches this build's key, mode and manifest.
    void LoadKeys()
    {
        g_keys.clear();
        const std::string path = wxlvk::path::Resolve(kKeysFile);
        FILE* f = std::fopen(path.c_str(), "rb");
        if (!f) return;
        KeysHeader h = {};
        const char* why = nullptr;
        if (std::fread(&h, sizeof h, 1, f) != 1) why = "short header";
        else if (h.magic != kKeysMagic || h.version != kKeysVersion) why = "another format";
        else if (h.keyBytes != sizeof(Key) || h.entryBytes != sizeof(PersistedKey)) why = "another key layout";
        else if (h.mode != s.mode) why = "another WXL_VK_TRANSLATED mode";
        else if (h.records != ManifestRecords()) why = "another manifest";
        else if (h.count > kMaxPipelines) why = "too many entries";
        else
        {
            g_keys.resize(h.count);
            if (h.count && std::fread(g_keys.data(), sizeof(PersistedKey), h.count, f) != h.count)
            {
                g_keys.clear();
                why = "truncated";
            }
        }
        std::fclose(f);
        s.stats.keysLoaded = static_cast<uint32_t>(g_keys.size());
        wxl::ScriptMgr::Log(WXL_LOG_DEBUG, "vulkan: pipelines: %s: %u key(s) to pre-warm%s%s",
                            path.c_str(), s.stats.keysLoaded, why ? "; not used: " : "", why ? why : "");
    }

    /// Writes the built keys (failed ones excepted) through a temporary file and a rename.
    void SaveKeys()
    {
        std::vector<PersistedKey> out;
        out.reserve(s.pipelineCount);
        for (uint32_t i = 0; i < s.pipelineCount; ++i)
        {
            const PipelineEntry& e = s.pipelines[i];
            if (e.pipeline == VK_NULL_HANDLE) continue;
            const State::PassInfo* info = nullptr;
            for (uint32_t p = 0; p < s.passInfoCount; ++p)
                if (s.passInfo[p].pass == e.pass) info = &s.passInfo[p];
            if (!info) continue;
            PersistedKey k;
            std::memset(&k, 0, sizeof k);
            k.key    = e.key;
            k.colour = info->colour;
            k.depth  = info->depth;
            k.kind   = e.row ? kKindRow : e.companion ? kKindCompanion : e.ffVertex ? kKindFfVertex : kKindModules;
            k.gbuffer = info->gbuffer;
            if (e.vrec) std::memcpy(k.vmd5, e.vrec->md5, sizeof k.vmd5);
            if (e.prec) std::memcpy(k.pmd5, e.prec->md5, sizeof k.pmd5);
            out.push_back(k);
        }
        KeysHeader h = {};
        h.magic      = kKeysMagic;
        h.version    = kKeysVersion;
        h.keyBytes   = sizeof(Key);
        h.entryBytes = sizeof(PersistedKey);
        h.count      = static_cast<uint32_t>(out.size());
        h.mode       = s.mode;
        h.records    = ManifestRecords();
        const std::string path = wxlvk::path::Resolve(kKeysFile);
        const std::string tmp  = path + ".tmp";
        bool ok = false;
        if (FILE* f = std::fopen(tmp.c_str(), "wb"))
        {
            ok = std::fwrite(&h, sizeof h, 1, f) == 1
                 && (out.empty() || std::fwrite(out.data(), sizeof(PersistedKey), out.size(), f) == out.size());
            ok = std::fclose(f) == 0 && ok;
        }
        ok = ok && MoveFileExA(tmp.c_str(), path.c_str(), MOVEFILE_REPLACE_EXISTING) != 0;
        if (!ok)
        {
            DeleteFileA(tmp.c_str());
            wxl::ScriptMgr::Log(WXL_LOG_WARN, "vulkan: pipelines: the key list could not be written to %s", path.c_str());
        }
        s.keysSavedAt = s.buildCount;
    }

    /**
     * The sources a persisted key was built from, by md5.
     *
     * @return bool ok : false when a module is no longer in the manifest, or a row no longer exists
     */
    bool PersistedSources(const PersistedKey& p, Sources& out)
    {
        out = Sources();
        const auto byMd5 = [](const char* md5, manifest::Kind kind) -> const manifest::Record* {
            if (md5[0] == '\0') return nullptr;
            const manifest::Record* r = manifest::ByMd5(md5);
            return r && r->kind == kind ? r : nullptr;
        };
        switch (p.kind)
        {
            case kKindRow:
                out.row = RowFor(p.key);
                return out.row != nullptr;
            case kKindModules:
                out.vr = byMd5(p.vmd5, manifest::Kind::Vertex);
                out.pr = byMd5(p.pmd5, manifest::Kind::Pixel);
                return out.vr && out.pr;
            case kKindFfVertex:
                out.pr       = byMd5(p.pmd5, manifest::Kind::Pixel);
                out.ffVertex = true;
                return out.pr != nullptr;
            case kKindCompanion:
                out.vr        = byMd5(p.vmd5, manifest::Kind::Vertex);
                out.pr        = out.vr ? CompanionFor(*out.vr) : nullptr;
                out.companion = true;
                return out.pr && std::strcmp(out.pr->md5, p.pmd5) == 0;
            default:
                return false;
        }
    }

    /// Builds the persisted keys of one pass class (PassCreated).
    void Prewarm(VkRenderPass pass, uint32_t colour, uint32_t depth, bool gbuffer)
    {
        LARGE_INTEGER t0 = {}, t1 = {};
        QueryPerformanceCounter(&t0);
        uint32_t built = 0, skipped = 0;
        for (const PersistedKey& p : g_keys)
        {
            if (p.colour != colour || p.depth != depth || (p.gbuffer != 0) != gbuffer) continue;
            if (p.key.bones & 2u) continue;   // tree wind: built at its first draw, once its set 3 exists
            if (s.pipelineCount >= kMaxPipelines) break;

            const uint32_t home  = KeyHash(p.key, pass);
            bool           there = false;
            for (uint32_t h = home; s.pipeHash[h] != 0; h = (h + 1) & (kPipeHashSlots - 1))
            {
                const PipelineEntry& e = s.pipelines[s.pipeHash[h] - 1];
                if (e.pass == pass && SameKey(e.key, p.key)) there = true;
            }
            if (there) continue;

            Sources     src;
            PipelineEntry entry;
            const char* why = "";
            if (!PersistedSources(p, src) || !Build(p.key, pass, entry, why, &src))
            {
                ++skipped;
                continue;
            }
            PipelineEntry& e = s.pipelines[s.pipelineCount++];
            e           = entry;
            e.key       = p.key;
            e.pass      = pass;
            e.prewarmed = true;
            uint32_t h = home;
            while (s.pipeHash[h] != 0) h = (h + 1) & (kPipeHashSlots - 1);
            s.pipeHash[h] = s.pipelineCount;
            if (e.translated) ++s.translatedCount;
            if (e.companion) ++s.companionCount;
            if (e.gbuf) ++s.stats.gbufferPipelines;
            if (e.gbufNull) ++s.stats.gbufferNullPipelines;
            ++built;
        }
        QueryPerformanceCounter(&t1);
        const uint64_t ticks = static_cast<uint64_t>(t1.QuadPart - t0.QuadPart);
        s.buildTicks += ticks;
        s.buildCount += built;
        s.stats.prewarmTicks += ticks;
        s.stats.prewarmed += built;
        s.stats.prewarmSkipped += skipped;
        s.stats.pipelines = s.pipelineCount - s.failedCount;
        s.stats.translatedPipelines = s.translatedCount;
        s.stats.companionPipelines  = s.companionCount;
        if (built || skipped)
        {
            LARGE_INTEGER f = {};
            QueryPerformanceFrequency(&f);
            wxl::ScriptMgr::Log(WXL_LOG_DEBUG,
                                "vulkan: pipelines: pre-warmed %u pipeline(s) for pass class %s / %s in %.1f ms "
                                "(%u persisted key(s) not built)",
                                built, wxlvk::vk::FormatName(static_cast<VkFormat>(colour)),
                                depth ? wxlvk::vk::FormatName(static_cast<VkFormat>(depth)) : "none",
                                f.QuadPart ? static_cast<double>(ticks) * 1000.0 / static_cast<double>(f.QuadPart) : 0.0,
                                skipped);
        }
    }
}
