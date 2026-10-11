// The translated-module manifest: the parser and the two lookups.
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

#include "vulkan/Manifest.hpp"

#include "common/ModulePath.hpp"

#include "wxl/Script.hpp"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>
#include <string>
#include <vector>

namespace
{
    using namespace wxlvk::vk::manifest;

    /// The most records the table holds. The client's archives hold 1578 byte-distinct records across
    /// every archive; 2048 leaves room for Patch-4.MPQ and a re-run that splits a container.
    constexpr uint32_t kMaxRecords = 2048;

    /// The most name entries. 3105 names in the archives (every Name:perm plus the bare Name of each
    /// record 0), many folded onto shared records; 4096 is the bound.
    constexpr uint32_t kMaxNames = 4096;

    /// The manifest's place, relative to the DLL (src/common/ModulePath.hpp).
    constexpr const char* kManifestRelative = "shaders\\translated\\manifest.txt";

    constexpr uint16_t kNoName = 0xFFFF;

    /// The contract's semantic spellings, indexed by Semantic.
    constexpr const char* kSemanticNames[kSemanticCount] = {
        "position0", "normal0", "color0", "color1",
        "texcoord0", "texcoord1", "texcoord2", "texcoord3",
        "texcoord4", "texcoord5", "texcoord6", "texcoord7",
        "blendweight0", "blendindices0", "fog0", "psize0",
    };

    struct State
    {
        Record   records[kMaxRecords] = {};
        uint32_t recordCount          = 0;
        Name     names[kMaxNames]     = {};
        uint32_t nameCount            = 0;
        Stats    stats                = {};
        bool     attempted            = false;
        bool     loaded               = false;
    };
    State s;

    /**
     * Reads a whole file. Plain CRT I/O, once per session; the file is tens of kilobytes.
     *
     * @param string path : the file
     * @param string& out : receives the bytes
     * @return bool ok : false when the file could not be opened or read
     */
    bool ReadWhole(const std::string& path, std::string& out)
    {
        FILE* f = nullptr;
        if (fopen_s(&f, path.c_str(), "rb") != 0 || !f) return false;

        bool ok = false;
        do
        {
            if (std::fseek(f, 0, SEEK_END) != 0) break;
            const long size = std::ftell(f);
            if (size < 0) break;
            if (std::fseek(f, 0, SEEK_SET) != 0) break;
            try
            {
                out.resize(static_cast<size_t>(size));
            }
            catch (const std::bad_alloc&)
            {
                break;
            }
            if (size > 0 && std::fread(&out[0], 1, out.size(), f) != out.size()) break;
            ok = true;
        } while (false);

        std::fclose(f);
        return ok;
    }

    /**
     * Splits one line into at most `cap` tokens on spaces and tabs, dropping a '#' comment.
     * The line is modified in place (NULs written at token ends).
     *
     * @param char* line : NUL-terminated, writable
     * @param char** tokens : receives pointers into `line`
     * @param uint32 cap : capacity of `tokens`
     * @return uint32 count : tokens found (0 for a blank or comment-only line)
     */
    uint32_t Tokenise(char* line, char** tokens, uint32_t cap)
    {
        if (char* hash = std::strchr(line, '#')) *hash = '\0';

        uint32_t n = 0;
        char*    p = line;
        while (*p && n < cap)
        {
            while (*p == ' ' || *p == '\t' || *p == '\r') ++p;
            if (!*p) break;
            tokens[n++] = p;
            while (*p && *p != ' ' && *p != '\t' && *p != '\r') ++p;
            if (*p) *p++ = '\0';
        }
        return n;
    }

    /**
     * A semantic token's bit position.
     *
     * @param string token : e.g. "texcoord3"
     * @param uint32& out : receives the Semantic value
     * @return bool ok : false for a spelling outside the contract
     */
    bool ParseSemantic(const char* token, uint32_t& out)
    {
        for (uint32_t i = 0; i < kSemanticCount; ++i)
        {
            if (std::strcmp(token, kSemanticNames[i]) == 0)
            {
                out = i;
                return true;
            }
        }
        return false;
    }

    /**
     * Copies a bounded string, always terminating; refuses one that would not fit rather than
     * truncating, because a truncated md5 or path names a different file.
     */
    template <size_t N>
    bool CopyBounded(char (&dst)[N], const char* src)
    {
        const size_t len = std::strlen(src);
        if (len + 1 > N) return false;
        std::memcpy(dst, src, len + 1);
        return true;
    }

    /**
     * Whether a token is 32 lowercase hex characters -- the one spelling ByMd5() will be asked
     * with, so a manifest that spells a hash in uppercase is normalised here rather than missed.
     */
    bool NormaliseMd5(char* token)
    {
        if (std::strlen(token) != 32) return false;
        for (char* p = token; *p; ++p)
        {
            if (*p >= 'A' && *p <= 'F') *p = static_cast<char>(*p - 'A' + 'a');
            const bool hex = (*p >= '0' && *p <= '9') || (*p >= 'a' && *p <= 'f');
            if (!hex) return false;
        }
        return true;
    }

    /**
     * One `record` line into a fresh Record.
     *
     * @param char** t : the tokens, t[0] == "record"
     * @param uint32 n : how many
     * @param Record& r : receives the fields; untouched on failure
     * @return bool ok
     */
    bool ParseRecord(char** t, uint32_t n, Record& r)
    {
        if (n < 7) return false;

        Record out = {};
        out.firstName = kNoName;
        out.companion = kNoRecord;

        if (!NormaliseMd5(t[1]) || !CopyBounded(out.md5, t[1])) return false;

        if (std::strcmp(t[2], "vertex") == 0)     out.kind = Kind::Vertex;
        else if (std::strcmp(t[2], "pixel") == 0) out.kind = Kind::Pixel;
        else return false;

        // `ff` is a pixel record by construction (the generated fixed-function pixel stage); an
        // ff vertex record is a contradiction and is refused like any other bad profile.
        if (std::strcmp(t[3], "vs_3_0") == 0)      out.profile = Profile::Vs30;
        else if (std::strcmp(t[3], "ps_3_0") == 0) out.profile = Profile::Ps30;
        else if (std::strcmp(t[3], "vs_2_0") == 0) out.profile = Profile::Vs20;
        else if (std::strcmp(t[3], "ps_2_0") == 0) out.profile = Profile::Ps20;
        else if (std::strcmp(t[3], "ff") == 0 && out.kind == Kind::Pixel) out.profile = Profile::Ff;
        else return false;

        if (!CopyBounded(out.spv, t[4])) return false;

        // vconst= and pconst= may come in either order; both are required.
        bool haveV = false, haveP = false;
        for (uint32_t i = 5; i < n; ++i)
        {
            if (std::strncmp(t[i], "vconst=", 7) == 0)
            {
                const long v = std::strtol(t[i] + 7, nullptr, 10);
                if (v != 64 && v != 256) return false;
                out.vconst = static_cast<uint16_t>(v);
                haveV      = true;
            }
            else if (std::strncmp(t[i], "pconst=", 7) == 0)
            {
                const long p = std::strtol(t[i] + 7, nullptr, 10);
                if (p <= 0 || p > 16) return false;
                out.pconst = static_cast<uint16_t>(p);
                haveP      = true;
            }
        }
        if (!haveV || !haveP) return false;

        r = out;
        return true;
    }

    /**
     * One attribute line (name / sampler / input / output / reads / flag) applied to the current
     * record.
     *
     * @param char** t : the tokens
     * @param uint32 n : how many
     * @param Record& r : the record the line belongs to
     * @param uint32 recordIndex : its index, for the name table
     * @return bool ok : false for a line that did not parse (the record keeps what it had)
     */
    bool ParseAttribute(char** t, uint32_t n, Record& r, uint32_t recordIndex)
    {
        if (std::strcmp(t[0], "name") == 0)
        {
            if (n != 2) return false;
            if (s.nameCount >= kMaxNames)
            {
                ++s.stats.truncated;
                return true; // not a parse failure: the table is full
            }
            Name& e = s.names[s.nameCount];
            if (!CopyBounded(e.name, t[1])) return false;
            e.record = static_cast<uint16_t>(recordIndex);
            e.hash   = HashName(e.name);
            if (r.firstName == kNoName) r.firstName = static_cast<uint16_t>(s.nameCount);
            ++r.nameCount;
            ++s.nameCount;
            ++s.stats.names;
            return true;
        }
        if (std::strcmp(t[0], "sampler") == 0)
        {
            if (n != 3) return false;
            char*      end  = nullptr;
            const long slot = std::strtol(t[1], &end, 10);
            if (!end || *end || slot < 0 || slot > 15) return false;
            const bool cube = std::strcmp(t[2], "cube") == 0;
            if (!cube && std::strcmp(t[2], "2d") != 0) return false;
            r.samplerMask |= static_cast<uint16_t>(1u << slot);
            if (cube) r.cubeMask |= static_cast<uint16_t>(1u << slot);
            return true;
        }
        if (std::strcmp(t[0], "input") == 0)
        {
            // The register token (vN) is informative only: the location is fixed by semantic.
            if (n != 3) return false;
            uint32_t sem = 0;
            if (!ParseSemantic(t[1], sem)) return false;
            r.inputMask |= 1u << sem;
            return true;
        }
        if (std::strcmp(t[0], "output") == 0 || std::strcmp(t[0], "reads") == 0)
        {
            if (n != 2) return false;
            uint32_t sem = 0;
            if (!ParseSemantic(t[1], sem)) return false;
            if (t[0][0] == 'o') r.outputMask |= 1u << sem;
            else                r.readsMask  |= 1u << sem;
            return true;
        }
        if (std::strcmp(t[0], "flag") == 0)
        {
            // Any flag on any record: a flag the stage does not use on that kind is kept, not
            // refused, so a future emitter run that moves one does not skip the line.
            if (n != 2) return false;
            if (std::strcmp(t[1], "fixed_fog") == 0)         r.fixedFog = true;
            else if (std::strcmp(t[1], "point_size") == 0)   r.pointSize = true;
            else if (std::strcmp(t[1], "point_sprite") == 0) r.pointSprite = true;
            else return false;
            return true;
        }
        if (std::strcmp(t[0], "companion") == 0)
        {
            // Kept as the md5 for now: the companion blocks come after the vertex rows that name
            // them (the shipped file puts them last), so the index is resolved once the whole
            // file has been read (LinkCompanions). Only a vertex row can have one.
            if (n != 2 || r.kind != Kind::Vertex) return false;
            if (!NormaliseMd5(t[1]) || !CopyBounded(r.companionMd5, t[1])) return false;
            return true;
        }
        if (std::strcmp(t[0], "bones") == 0)
        {
            // Only a 256-row vertex row has one: the variant is that module on the 64-row block.
            if (n != 2 || r.kind != Kind::Vertex || r.vconst != 256) return false;
            if (!CopyBounded(r.bones, t[1])) return false;
            ++s.stats.bones;
            return true;
        }
        if (std::strcmp(t[0], "wind") == 0)
        {
            // A vertex record's (the bend) or an M2 pixel record's (the leaves' shimmer and light).
            if (n != 2) return false;
            if (!CopyBounded(r.wind, t[1])) return false;
            ++s.stats.wind;
            return true;
        }
        if (std::strcmp(t[0], "shadowwind") == 0 || std::strcmp(t[0], "shadowwindbones") == 0)
        {
            // The tree-wind variants' shadow variants (shadowgen.py). The attribute lines are sorted,
            // so these come BEFORE their `wind` / `windbones` lines: the flag is set here and checked
            // against the wind variant once the whole file is read (DropOrphanShadowWind).
            if (n != 2 || r.kind != Kind::Vertex) return false;
            if (t[0][10] == '\0') r.shadowWind = true;
            else                  r.shadowWindBones = true;
            return true;
        }
        if (std::strcmp(t[0], "shadow") == 0 || std::strcmp(t[0], "shadowbones") == 0)
        {
            // The draw replay's variants: names that follow from the module's own (shadowgen.py).
            if (n != 2 || r.kind != Kind::Vertex) return false;
            if (t[0][6] == '\0')
            {
                r.shadow = true;
                ++s.stats.shadow;
            }
            else
            {
                if (r.bones[0] == '\0') return false;
                r.shadowBones = true;
            }
            return true;
        }
        if (std::strncmp(t[0], "gbuf", 4) == 0)
        {
            // The G-buffer variants (3.Tools/gbuffer): names that follow from the module's own, like the
            // shadow ones. A vertex record's four, a pixel record's `gbuf` and `gbufwind`. The line may
            // precede the `bones` / `wind` one it derives from: checked once the file is read (DropOrphanGBuf).
            if (n != 2) return false;
            const char* v = t[0] + 4;
            if (*v == '\0')
            {
                r.gbuf = true;
                ++s.stats.gbuf;
            }
            else if (std::strcmp(v, "bones") == 0)
            {
                if (r.kind != Kind::Vertex) return false;
                r.gbufBones = true;
            }
            else if (std::strcmp(v, "wind") == 0)
                r.gbufWind = true;
            else if (std::strcmp(v, "windbones") == 0)
            {
                if (r.kind != Kind::Vertex) return false;
                r.gbufWindBones = true;
            }
            else
                return false;
            return true;
        }
        if (std::strcmp(t[0], "windbones") == 0)
        {
            if (n != 2 || r.kind != Kind::Vertex || r.vconst != 256) return false;
            if (!CopyBounded(r.windBones, t[1])) return false;
            return true;
        }
        return false;
    }

    /**
     * Clears a `shadowwind` / `shadowwindbones` flag whose record has no `wind` / `windbones`
     * variant to derive it from, after the whole file is in the table (the lines come in sorted
     * order, the shadow ones first).
     */
    void DropOrphanShadowWind()
    {
        for (uint32_t i = 0; i < s.recordCount; ++i)
        {
            Record& r = s.records[i];
            if (r.wind[0] == '\0')      r.shadowWind      = false;
            if (r.windBones[0] == '\0') r.shadowWindBones = false;
        }
    }

    /// Clears a G-buffer flag whose record has no variant to derive it from (a `gbufbones` without
    /// `bones`, a `gbufwind` without `wind`, a `gbufwindbones` without `windbones`).
    void DropOrphanGBuf()
    {
        for (uint32_t i = 0; i < s.recordCount; ++i)
        {
            Record& r = s.records[i];
            if (r.bones[0] == '\0')     r.gbufBones     = false;
            if (r.wind[0] == '\0')      r.gbufWind      = false;
            if (r.windBones[0] == '\0') r.gbufWindBones = false;
        }
    }

    /**
     * Resolves every `companion` md5 to a record index, after the whole file is in the table.
     * A md5 that names no record, or names one that is not an `ff` pixel record, leaves the
     * row without a companion (kNoRecord) and is counted and logged: such a vertex row's
     * `ps == ""` draws are then skipped and named by the pipeline side.
     */
    void LinkCompanions()
    {
        for (uint32_t i = 0; i < s.recordCount; ++i)
        {
            Record& r = s.records[i];
            r.companion = kNoRecord;
            if (r.kind != Kind::Vertex || r.companionMd5[0] == '\0') continue;

            for (uint32_t j = 0; j < s.recordCount; ++j)
            {
                const Record& c = s.records[j];
                if (c.kind == Kind::Pixel && c.profile == Profile::Ff
                    && std::memcmp(c.md5, r.companionMd5, 32) == 0)
                {
                    r.companion = static_cast<uint16_t>(j);
                    break;
                }
            }
            if (r.companion == kNoRecord)
            {
                ++s.stats.companionsUnresolved;
                wxl::ScriptMgr::Log(WXL_LOG_WARN,
                                    "vulkan: manifest: vertex record %s (%s) names companion %s, "
                                    "which is not an ff pixel record in this file; its draws with "
                                    "no pixel shader stay unsupported",
                                    r.md5, FirstName(r), r.companionMd5);
            }
            else
                ++s.stats.companions;
        }
    }
}

namespace wxlvk::vk::manifest
{
    uint32_t HashName(const char* str)
    {
        uint32_t h = 2166136261u;
        for (const char* p = str; *p; ++p) h = (h ^ static_cast<uint8_t>(*p)) * 16777619u;
        return h;
    }

    bool Load()
    {
        if (s.attempted) return s.loaded;
        s.attempted = true;

        const std::string path = wxlvk::path::Resolve(kManifestRelative);
        std::string       text;
        if (!ReadWhole(path, text))
        {
            // A deployment without the translator's output: only the hand-written rows can draw.
            wxl::ScriptMgr::Log(WXL_LOG_WARN,
                                "vulkan: manifest: no manifest at \"%s\"; no translated module "
                                "will be used, only the hand-written pipelines draw",
                                path.c_str());
            return false;
        }

        // Line by line, in place. A `record` line opens a record; every other line applies to the
        // record most recently opened. Blank lines are the contract's block separators and carry
        // no meaning here, so a manifest without them parses the same.
        Record*  current      = nullptr;
        uint32_t currentIndex = 0;
        bool     dropping     = false; // the current record did not fit: skip its attributes
        uint32_t lineNo       = 0;

        size_t pos = 0;
        while (pos < text.size())
        {
            size_t end = text.find('\n', pos);
            if (end == std::string::npos) end = text.size();
            std::string line = text.substr(pos, end - pos);
            pos = end + 1;
            ++lineNo;

            char* tokens[16];
            const uint32_t n = Tokenise(&line[0], tokens, 16);
            if (n == 0) continue;

            if (std::strcmp(tokens[0], "record") == 0)
            {
                dropping = false;
                current  = nullptr;
                if (s.recordCount >= kMaxRecords)
                {
                    ++s.stats.truncated;
                    dropping = true;
                    continue;
                }
                Record r = {};
                if (!ParseRecord(tokens, n, r))
                {
                    ++s.stats.skipped;
                    dropping = true;
                    continue;
                }
                currentIndex              = s.recordCount;
                s.records[s.recordCount++] = r;
                current                   = &s.records[currentIndex];
                ++s.stats.records;
                if (r.kind == Kind::Vertex) ++s.stats.vertex;
                else                        ++s.stats.pixel;
                if (r.profile == Profile::Ff) ++s.stats.ff;
                continue;
            }

            if (dropping) continue;
            if (!current)
            {
                ++s.stats.skipped; // an attribute before any record
                continue;
            }
            if (!ParseAttribute(tokens, n, *current, currentIndex))
            {
                ++s.stats.skipped;
                if (s.stats.skipped <= 8)
                    wxl::ScriptMgr::Log(WXL_LOG_WARN, "vulkan: manifest: line %u did not parse: \"%s\"",
                                        lineNo, tokens[0]);
            }
        }

        // The companion links and the shadow-wind check, once every record is in the table.
        LinkCompanions();
        DropOrphanShadowWind();
        DropOrphanGBuf();

        s.loaded = s.recordCount > 0;

        wxl::ScriptMgr::Log(s.loaded ? WXL_LOG_DEBUG : WXL_LOG_WARN,
                            "vulkan: manifest: \"%s\": %u record(s) (%u vertex, %u pixel, of which "
                            "%u generated ff), %u companion(s) linked (%u unresolved), %u bone-palette "
                            "variant(s), %u name(s), %u line(s) skipped, %u entr%s truncated (tables of "
                            "%u records and %u names)",
                            path.c_str(), s.stats.records, s.stats.vertex, s.stats.pixel, s.stats.ff,
                            s.stats.companions, s.stats.companionsUnresolved, s.stats.bones, s.stats.names,
                            s.stats.skipped, s.stats.truncated,
                            s.stats.truncated == 1 ? "y" : "ies", kMaxRecords, kMaxNames);
        if (s.stats.gbuf)
            wxl::ScriptMgr::Log(WXL_LOG_DEBUG, "vulkan: manifest: %u record(s) carry G-buffer variants (3.Tools/gbuffer)",
                                s.stats.gbuf);
        return s.loaded;
    }

    const Record* Companion(const Record& r)
    {
        if (!s.loaded || r.kind != Kind::Vertex || r.companion == kNoRecord) return nullptr;
        if (r.companion >= s.recordCount) return nullptr;
        return &s.records[r.companion];
    }

    const Record* FfStage(bool pointSprite)
    {
        if (!s.loaded) return nullptr;
        for (uint32_t i = 0; i < s.recordCount; ++i)
        {
            const Record& r = s.records[i];
            if (r.kind == Kind::Pixel && r.profile == Profile::Ff && r.pointSprite == pointSprite) return &r;
        }
        return nullptr;
    }

    bool Loaded() { return s.loaded; }

    const Record* ByMd5(const char* md5)
    {
        if (!s.loaded || !md5) return nullptr;
        for (uint32_t i = 0; i < s.recordCount; ++i)
            if (std::memcmp(s.records[i].md5, md5, 32) == 0) return &s.records[i];
        return nullptr;
    }

    const Record* ByName(const char* name, Kind kind)
    {
        if (!s.loaded || !name || !*name) return nullptr;
        const uint32_t h = HashName(name);
        for (uint32_t i = 0; i < s.nameCount; ++i)
        {
            const Name& e = s.names[i];
            if (e.hash != h || std::strcmp(e.name, name) != 0) continue;
            const Record& r = s.records[e.record];
            if (r.kind == kind) return &r;
        }
        return nullptr;
    }

    const char* FirstName(const Record& r)
    {
        if (r.firstName == kNoName || r.firstName >= s.nameCount) return r.md5;
        return s.names[r.firstName].name;
    }

    const char* SemanticName(uint32_t semantic)
    {
        return semantic < kSemanticCount ? kSemanticNames[semantic] : "?";
    }

    void GetStats(Stats& out) { out = s.stats; }
}
