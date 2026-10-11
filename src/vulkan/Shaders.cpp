// SPIR-V modules, loaded from the extension's own shaders/ folder and cached by name.
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

#include "vulkan/Shaders.hpp"

#include "common/ModulePath.hpp"
#include "vulkan/Context.hpp"
#include "vulkan/Loader.hpp"

#include "wxl/Script.hpp"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <new>
#include <string>
#include <vector>

namespace
{
    namespace context = wxlvk::vk::context;
    namespace loader  = wxlvk::vk::loader;

    using wxlvk::vk::ResultName;

    /// The most modules this cache holds. A fixed table for the reason every table in src/vulkan/
    /// is: address space is the scarce resource in this 32-bit process. The hand-written set is
    /// two dozen; the translated set a scene can ask for is a few hundred of the archives' 1578, each
    /// with up to three variants beside it (the bone palette, the tree wind, the G-buffer's), so 4096
    /// (about 320 KiB of table) is the bound. Beyond it a name is refused with one log line.
    constexpr uint32_t kMaxModules = 4096;

    /// The longest key accepted, so that "shaders\" + key + ".spv" fits a fixed buffer and the
    /// key can be compared with strncmp on a fixed field. A translated path,
    /// "translated/vertex/<32 hex>.spv", is 54; its tree-wind variant of a bone-palette variant,
    /// ".../<32 hex>.bones.wind.spv", 65. MAX_PATH-safe: the folder prefix is added in front.
    constexpr size_t kMaxName = 191;

    /// The first word of every SPIR-V module, in the host's byte order (the file is little-endian
    /// and so is this process; a big-endian module would read as 0x03022307 and be refused, which
    /// is correct -- vkCreateShaderModule wants the host's order).
    constexpr uint32_t kSpirvMagic = 0x07230203u;

    /// One name the cache has been asked for, whether or not it loaded.
    struct Entry
    {
        char           name[kMaxName + 1] = {};
        VkShaderModule module             = VK_NULL_HANDLE; ///< VK_NULL_HANDLE when `failed`
        bool           failed             = false;          ///< the load failed; do not retry
    };

    /// Everything this file owns, in one place.
    struct State
    {
        VkDevice              device = nullptr;
        const loader::Device* dapi   = nullptr;

        Entry    entries[kMaxModules] = {};
        uint32_t count                = 0;

        bool     fullLogged  = false;
        bool     ready       = false;
    };
    State s;

    /**
     * Logs a failed Vulkan call with its numeric result.
     *
     * @param string what : the call that failed, by name
     * @param VkResult r : what it returned
     */
    void Fail(const char* what, VkResult r)
    {
        wxl::ScriptMgr::Log(WXL_LOG_WARN, "vulkan: shaders: %s failed: %s (%d)", what,
                            ResultName(r), static_cast<int>(r));
    }

    /**
     * Confirms every entry point this file calls is non-null, naming the ones that are not.
     *
     * @param loader::Device& d : the device table
     * @return bool ok : false when at least one is null, each logged by name
     */
    bool HaveEntryPoints(const loader::Device& d)
    {
        struct Named { const void* fn; const char* name; };
        const Named entries[] = {
            { reinterpret_cast<const void*>(d.DeviceWaitIdle),      "vkDeviceWaitIdle" },
            { reinterpret_cast<const void*>(d.CreateShaderModule),  "vkCreateShaderModule" },
            { reinterpret_cast<const void*>(d.DestroyShaderModule), "vkDestroyShaderModule" },
        };

        unsigned missing = 0;
        for (const Named& e : entries)
        {
            if (!e.fn)
            {
                wxl::ScriptMgr::Log(WXL_LOG_WARN, "vulkan: shaders: entry point missing: %s", e.name);
                ++missing;
            }
        }

        if (missing)
            wxl::ScriptMgr::Log(WXL_LOG_WARN,
                                "vulkan: shaders: %u entry point(s) missing; declining", missing);
        return missing == 0;
    }

    /**
     * Finds a name in the cache.
     *
     * @param string name : the stem
     * @return Entry* entry : null when it has never been asked for
     */
    Entry* Find(const char* name)
    {
        for (uint32_t i = 0; i < s.count; ++i)
            if (std::strncmp(s.entries[i].name, name, kMaxName) == 0) return &s.entries[i];
        return nullptr;
    }

    /**
     * Reads a whole file into `out`. Plain CRT I/O: the file is a few kilobytes, read once per
     * session, and nothing about it wants a mapping or an async read.
     *
     * @param string path : the file
     * @param vector& out : receives the bytes; cleared on failure
     * @param string& why : receives the reason on failure, for the one log line
     * @return bool ok
     */
    bool ReadFile(const std::string& path, std::vector<uint8_t>& out, const char*& why)
    {
        out.clear();

        FILE* f = nullptr;
        if (fopen_s(&f, path.c_str(), "rb") != 0 || !f)
        {
            why = "the file could not be opened";
            return false;
        }

        bool ok = false;
        do
        {
            if (std::fseek(f, 0, SEEK_END) != 0) { why = "seek failed"; break; }
            const long size = std::ftell(f);
            if (size <= 0) { why = "the file is empty"; break; }
            if (std::fseek(f, 0, SEEK_SET) != 0) { why = "seek failed"; break; }

            try
            {
                out.resize(static_cast<size_t>(size));
            }
            catch (const std::bad_alloc&)
            {
                why = "no address space for the file";
                break;
            }

            if (std::fread(out.data(), 1, out.size(), f) != out.size())
            {
                why = "the read came up short";
                break;
            }
            ok = true;
        } while (false);

        std::fclose(f);
        if (!ok) out.clear();
        return ok;
    }
}

namespace
{
    /**
     * The shared load: one key, one path under shaders/, the cache, the checks and the create.
     *
     * @param string key : how the cache remembers it (a stem, or a relative path)
     * @param string relative : the file under the extension's folder, with "shaders\\" prefixed
     * @return VkShaderModule module : VK_NULL_HANDLE on any refusal, each logged once per key
     */
    VkShaderModule Load(const char* key, const char* relative)
    {
        if (Entry* e = Find(key)) return e->module; // VK_NULL_HANDLE when it failed before

        if (s.count >= kMaxModules)
        {
            if (!s.fullLogged)
            {
                s.fullLogged = true;
                wxl::ScriptMgr::Log(WXL_LOG_WARN,
                                    "vulkan: shaders: the cache holds %u names (kMaxModules); "
                                    "\"%s\" and anything after it are refused (logged once)",
                                    kMaxModules, key);
            }
            return VK_NULL_HANDLE;
        }

        // The entry is claimed before the load, so a failure is remembered by name and never
        // retried: a missing file is one open and one line per session, not one per draw.
        Entry& e = s.entries[s.count++];
        std::strncpy(e.name, key, kMaxName);
        e.name[kMaxName] = '\0';
        e.failed = true;

        const std::string path = wxlvk::path::Resolve(relative);

        std::vector<uint8_t> bytes;
        const char* why = "";
        if (!ReadFile(path, bytes, why))
        {
            wxl::ScriptMgr::Log(WXL_LOG_WARN, "vulkan: shaders: \"%s\": %s (\"%s\")", key, why,
                                path.c_str());
            return VK_NULL_HANDLE;
        }

        // Two checks the driver does not make. The size has to be a multiple of four because a
        // module is a sequence of 32-bit words and vkCreateShaderModule says so in its valid usage;
        // the magic number is the cheapest proof that the file is a module at all and not, say, the
        // GLSL source saved under the wrong name.
        if (bytes.size() < 4 || bytes.size() % 4 != 0)
        {
            wxl::ScriptMgr::Log(WXL_LOG_WARN,
                                "vulkan: shaders: \"%s\": %llu bytes is not a whole number of "
                                "SPIR-V words (\"%s\")",
                                key, static_cast<unsigned long long>(bytes.size()), path.c_str());
            return VK_NULL_HANDLE;
        }

        uint32_t magic = 0;
        std::memcpy(&magic, bytes.data(), sizeof magic);
        if (magic != kSpirvMagic)
        {
            wxl::ScriptMgr::Log(WXL_LOG_WARN,
                                "vulkan: shaders: \"%s\": first word is 0x%08X, not the SPIR-V magic "
                                "0x%08X (\"%s\")",
                                key, static_cast<unsigned>(magic), static_cast<unsigned>(kSpirvMagic),
                                path.c_str());
            return VK_NULL_HANDLE;
        }

        // pCode wants a uint32_t pointer, and a std::vector<uint8_t>'s storage is only guaranteed
        // to be aligned for uint8_t. In practice every allocator hands back something far better
        // aligned than four, but "in practice" is not a contract; the copy into a word vector is a
        // few kilobytes once per session and makes the alignment a fact.
        std::vector<uint32_t> words;
        try
        {
            words.resize(bytes.size() / 4);
        }
        catch (const std::bad_alloc&)
        {
            wxl::ScriptMgr::Log(WXL_LOG_WARN, "vulkan: shaders: \"%s\": no address space", key);
            return VK_NULL_HANDLE;
        }
        std::memcpy(words.data(), bytes.data(), bytes.size());

        VkShaderModuleCreateInfo info = {};
        info.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        info.codeSize = bytes.size();
        info.pCode    = words.data();

        VkShaderModule module = VK_NULL_HANDLE;
        const VkResult r = s.dapi->CreateShaderModule(s.device, &info, nullptr, &module);
        if (r != VK_SUCCESS)
        {
            Fail("vkCreateShaderModule", r);
            wxl::ScriptMgr::Log(WXL_LOG_WARN, "vulkan: shaders:   for \"%s\" (\"%s\")", key,
                                path.c_str());
            return VK_NULL_HANDLE;
        }

        e.module = module;
        e.failed = false;

        wxl::ScriptMgr::Log(WXL_LOG_DEBUG, "vulkan: shaders: loaded \"%s\": %llu bytes of SPIR-V",
                            key, static_cast<unsigned long long>(bytes.size()));
        return module;
    }
}

namespace wxlvk::vk::shaders
{
    bool Init()
    {
        const context::Handles* h = context::Get();
        if (!h || !h->device || !h->dapi)
        {
            wxl::ScriptMgr::Log(WXL_LOG_WARN, "vulkan: shaders: Init() without a live context");
            return false;
        }

        if (s.ready)
        {
            if (s.device == h->device) return true;

            // The same reasoning as texture::Init(): a different VkDevice means ours is gone, and
            // destroying modules on it would be a call into a dead device.
            wxl::ScriptMgr::Log(WXL_LOG_WARN,
                                "vulkan: shaders: the context reports a different VkDevice than the "
                                "one these modules were built on; forgetting %u of them rather than "
                                "destroying them",
                                s.count);
            s = State();
        }

        if (!HaveEntryPoints(*h->dapi)) return false;

        s.device = h->device;
        s.dapi   = h->dapi;
        s.ready  = true;

        wxl::ScriptMgr::Log(WXL_LOG_DEBUG,
                            "vulkan: shaders: ready -- modules are loaded from \"%s\" on first use, "
                            "at most %u cached",
                            wxlvk::path::Resolve("shaders\\").c_str(), kMaxModules);
        return true;
    }

    VkShaderModule Get(const char* name)
    {
        if (!s.ready || !name || !*name) return VK_NULL_HANDLE;

        if (std::strlen(name) > kMaxName)
        {
            // Refused rather than truncated: a truncated name is a different file, and the log
            // would then name a shader nobody asked for.
            wxl::ScriptMgr::Log(WXL_LOG_WARN,
                                "vulkan: shaders: refusing the name \"%s\": longer than %u", name,
                                static_cast<unsigned>(kMaxName));
            return VK_NULL_HANDLE;
        }

        char relative[8 + kMaxName + 5];
        std::snprintf(relative, sizeof relative, "shaders\\%s.spv", name);
        return Load(name, relative);
    }

    VkShaderModule GetPath(const char* path)
    {
        if (!s.ready || !path || !*path) return VK_NULL_HANDLE;

        if (std::strlen(path) > kMaxName)
        {
            wxl::ScriptMgr::Log(WXL_LOG_WARN,
                                "vulkan: shaders: refusing the path \"%s\": longer than %u", path,
                                static_cast<unsigned>(kMaxName));
            return VK_NULL_HANDLE;
        }

        // The manifest writes forward slashes; CreateFileA takes either, and the key keeps the
        // manifest's spelling so a log line matches the manifest line it came from.
        char relative[8 + kMaxName + 1];
        std::snprintf(relative, sizeof relative, "shaders\\%s", path);
        return Load(path, relative);
    }
}
