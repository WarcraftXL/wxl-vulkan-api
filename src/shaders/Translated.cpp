// The translated-module registry: the MD5 of each created record, and the name table.
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

#include "shaders/Translated.hpp"

#include "common/Settings.hpp"
#include "vulkan/Manifest.hpp"

#include "wxl/Script.hpp"
#include "wxl/game/Binding.hpp"

// The core names the CGxShader fields: GxDevice.hpp the name and the target, Shader.hpp the code
// array (+0x4C length, +0x50 data).
#include "wxl/offsets/engine/GxDevice.hpp"
#include "wxl/offsets/engine/Shader.hpp"

#include <windows.h>
#include <wincrypt.h>

#include <cstddef>
#include <cstdint>
#include <cstring>

namespace
{
    namespace gxdev    = wxl::offsets::engine::gxdevice;
    namespace shx      = wxl::offsets::engine::shader;
    namespace settings = wxlvk::settings;
    namespace manifest = wxlvk::vk::manifest;

    /// The most distinct (name, stage) records the registry holds. The client's archives hold 1578
    /// distinct records and ~3100 names; a session creates far fewer, since the engine asks for
    /// `count` permutations of a container and not the container. 2048 is the manifest's own bound.
    constexpr uint32_t kMaxEntries = 2048;

    /// Names whose Resolve() fell back to the manifest's name table, remembered so the fallback
    /// is logged once per name and not once per pipeline build.
    constexpr uint32_t kMaxByNameLogged = 64;

    /// RFC 1321's second test vector: MD5("abc"). Checked once at Install(), because a wrong
    /// provider or a wrong buffer order would silently make every record "unknown".
    constexpr const char* kSelfTestInput  = "abc";
    constexpr const char* kSelfTestDigest = "900150983cd24fb0d6963f7d28e17f72";

    /// One record the engine created, by name and stage.
    struct Entry
    {
        uint32_t                hash;                       ///< manifest::HashName(name)
        char                    name[manifest::kNameChars]; ///< CGxShader +0x18
        bool                    pixel;
        bool                    known;                      ///< the manifest has `md5`
        char                    md5[33];                    ///< of the bytes at the last create
        const manifest::Record* row;                        ///< null when unknown
    };

    struct State
    {
        Entry    entries[kMaxEntries] = {};
        uint32_t count                = 0;

        uint32_t byNameLogged[kMaxByNameLogged] = {};
        uint32_t byNameLoggedCount               = 0;

        HCRYPTPROV provider   = 0;
        bool       armed      = false;
        bool       fullLogged = false;
    };
    State s;

    /**
     * MD5 of a byte range as 32 lowercase hex characters, through the Win32 cryptographic
     * provider rather than a hand-rolled digest: the provider is in every Windows this client runs
     * on, and a digest nobody wrote here is a digest nobody can get subtly wrong here.
     *
     * @param const void* data : the bytes
     * @param uint32 size : how many
     * @param char out[33] : receives the hex and a NUL
     * @return bool ok : false when the provider refused (logged by the caller)
     */
    bool Md5Hex(const void* data, uint32_t size, char out[33])
    {
        if (!s.provider) return false;

        HCRYPTHASH hash = 0;
        if (!CryptCreateHash(s.provider, CALG_MD5, 0, 0, &hash)) return false;

        bool ok = false;
        do
        {
            if (!CryptHashData(hash, static_cast<const BYTE*>(data), size, 0)) break;
            BYTE  digest[16];
            DWORD len = sizeof digest;
            if (!CryptGetHashParam(hash, HP_HASHVAL, digest, &len, 0) || len != 16) break;
            static const char hex[] = "0123456789abcdef";
            for (unsigned i = 0; i < 16; ++i)
            {
                out[2 * i]     = hex[digest[i] >> 4];
                out[2 * i + 1] = hex[digest[i] & 0xF];
            }
            out[32] = '\0';
            ok = true;
        } while (false);

        CryptDestroyHash(hash);
        return ok;
    }

    /**
     * Finds a registry entry by name and stage.
     *
     * @param string name : the record's name
     * @param uint32 hash : HashName(name), computed by the caller once
     * @param bool pixel : the stage
     * @return Entry* entry : null when never seen
     */
    Entry* Find(const char* name, uint32_t hash, bool pixel)
    {
        for (uint32_t i = 0; i < s.count; ++i)
        {
            Entry& e = s.entries[i];
            if (e.hash == hash && e.pixel == pixel && std::strcmp(e.name, name) == 0) return &e;
        }
        return nullptr;
    }

    /**
     * Hashes the bytes a record carries, looks them up, and files the answer under the record's name.
     *
     * One strcmp per registry entry and one MD5 over a few hundred bytes, at create time only --
     * device setup, a format change or an IShaderReload. Nothing here runs per draw.
     *
     * @param void* wrapper : the CGxShader
     * @param bool pixel : the stage
     * @return pointer row : the manifest record filed for the name, null when unknown
     */
    const manifest::Record* Register(void* wrapper, bool pixel)
    {
        if (!wrapper) return nullptr;

        // +0x18 is an SStrDupA the engine owns, non-null for every record ShaderCreate built; a
        // null is tolerated anyway, since a crash here would be unrecoverable.
        const char* const name = wxl::game::At<const char*>(wrapper, gxdev::kShaderName);
        if (!name || !*name) return nullptr;

        const uint32_t       size = wxl::game::At<uint32_t>(wrapper, shx::kCgxShaderByteLen);
        const uint8_t* const data = wxl::game::At<const uint8_t*>(wrapper, shx::kCgxShaderBytePtr);

        char md5[33] = {};
        const bool hashed = size != 0 && data != nullptr && Md5Hex(data, size, md5);

        const uint32_t hash = manifest::HashName(name);
        Entry* e = Find(name, hash, pixel);
        const bool fresh = e == nullptr;
        if (!e)
        {
            if (s.count >= kMaxEntries)
            {
                if (!s.fullLogged)
                {
                    s.fullLogged = true;
                    wxl::ScriptMgr::Log(WXL_LOG_WARN,
                                        "translated: the registry holds %u records (kMaxEntries); "
                                        "\"%s\" and anything after it are not registered (logged once)",
                                        kMaxEntries, name);
                }
                return nullptr;
            }
            e = &s.entries[s.count];
            std::memset(e, 0, sizeof *e);
            e->hash  = hash;
            e->pixel = pixel;
            std::strncpy(e->name, name, sizeof e->name - 1);
            // Published only now, complete: a draw-time reader scanning concurrently sees either
            // the old count or a finished entry.
            ++s.count;
        }

        // The same name created again with the same bytes (a format change) needs no
        // re-resolution; different bytes (an IShaderReload) are a new fact and are resolved again.
        const bool changed = !hashed || std::memcmp(e->md5, md5, 32) != 0;
        if (!fresh && !changed) return e->row;

        const manifest::Record* row = nullptr;
        if (hashed)
        {
            std::memcpy(e->md5, md5, sizeof e->md5);
            row = manifest::ByMd5(md5);
            // A hash hit for the wrong stage is a manifest bug, not a match.
            if (row && (row->kind == manifest::Kind::Pixel) != pixel) row = nullptr;
        }
        else
        {
            e->md5[0] = '\0';
        }
        e->row   = row;
        e->known = row != nullptr;

        if (!hashed)
            wxl::ScriptMgr::Log(WXL_LOG_WARN,
                                "translated: \"%s\" (%s): %u bytes at %p could not be hashed; "
                                "translated module unknown",
                                name, pixel ? "pixel" : "vertex", size, static_cast<const void*>(data));
        else if (row)
            wxl::ScriptMgr::Log(WXL_LOG_DEBUG,
                                "translated: \"%s\" (%s, %u bytes): known, md5 %s -> %s, vconst %u%s",
                                name, pixel ? "pixel" : "vertex", size, md5, row->spv, row->vconst,
                                fresh ? "" : " (re-resolved)");
        else
            wxl::ScriptMgr::Log(WXL_LOG_DEBUG,
                                "translated: \"%s\" (%s, %u bytes): unknown, md5 %s%s",
                                name, pixel ? "pixel" : "vertex", size, md5,
                                manifest::Loaded() ? "" : " (no manifest)");
        return row;
    }

    /**
     * Acquires the provider and proves it hashes: MD5("abc") against RFC 1321.
     *
     * @return bool ok : false when the provider could not be acquired or the vector mismatched
     */
    bool OpenProvider()
    {
        // CRYPT_VERIFYCONTEXT: no key container, no persistent state, which is all a hash needs
        // and the one mode that never prompts or touches the user's profile.
        if (!CryptAcquireContextA(&s.provider, nullptr, nullptr, PROV_RSA_FULL, CRYPT_VERIFYCONTEXT))
        {
            s.provider = 0;
            wxl::ScriptMgr::Log(WXL_LOG_ERROR,
                                "translated: CryptAcquireContext failed (%lu); no record can be "
                                "hashed, so no translated module can be matched",
                                static_cast<unsigned long>(GetLastError()));
            return false;
        }

        char digest[33] = {};
        if (!Md5Hex(kSelfTestInput, static_cast<uint32_t>(std::strlen(kSelfTestInput)), digest)
            || std::strcmp(digest, kSelfTestDigest) != 0)
        {
            wxl::ScriptMgr::Log(WXL_LOG_ERROR,
                                "translated: MD5 self-test failed: md5(\"abc\") = \"%s\", expected "
                                "\"%s\"; no translated module will be matched",
                                digest, kSelfTestDigest);
            CryptReleaseContext(s.provider, 0);
            s.provider = 0;
            return false;
        }
        return true;
    }
}

namespace wxlvk::shaders::translated
{
    bool Install()
    {
        if (s.armed) return true;
        if (settings::TranslatedMode() == settings::kTranslatedOff)
        {
            wxl::ScriptMgr::Log(WXL_LOG_DEBUG,
                                "translated: off (%s=0); only the hand-written rows can make a shader "
                                "valid",
                                settings::kTranslatedKey);
            return false;
        }
        const bool loaded = manifest::Load();
        if (!OpenProvider()) return false;
        s.armed = true;
        wxl::ScriptMgr::Log(WXL_LOG_DEBUG, "translated: armed; manifest %s; MD5 self-test passed",
                            loaded ? "loaded" : "absent, every record will be unknown");
        return true;
    }

    const manifest::Record* Create(void* shader, bool pixel)
    {
        if (!s.armed) return nullptr;
        return Register(shader, pixel);
    }

    const manifest::Record* Resolve(const char* name, bool pixel)
    {
        if (!name || !*name || !manifest::Loaded()) return nullptr;

        const uint32_t hash = manifest::HashName(name);
        if (const Entry* e = Find(name, hash, pixel)) return e->row;

        // Never created through slot 72: the name table is the weaker key, and said so once.
        const manifest::Record* row = manifest::ByName(name, pixel ? manifest::Kind::Pixel
                                                                   : manifest::Kind::Vertex);
        if (!row) return nullptr;

        bool logged = false;
        for (uint32_t i = 0; i < s.byNameLoggedCount; ++i)
            if (s.byNameLogged[i] == hash) logged = true;
        if (!logged)
        {
            if (s.byNameLoggedCount < kMaxByNameLogged) s.byNameLogged[s.byNameLoggedCount++] = hash;
            wxl::ScriptMgr::Log(WXL_LOG_WARN,
                                "translated: \"%s\" (%s) was never seen at a create; resolved by "
                                "name to %s, not by hash (logged once)",
                                name, pixel ? "pixel" : "vertex", row->spv);
        }
        return row;
    }
}
