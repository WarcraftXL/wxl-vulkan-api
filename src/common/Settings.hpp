// The extension's configuration keys, read from the environment first and WarcraftXL.cfg second.
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

#pragma once

#include "wxl/CfgParse.hpp"

#include <windows.h>

#include <cstdlib>
#include <cstring>
#include <string>
#include <unordered_map>

// Every key lives here so WarcraftXL.cfg is parsed once, with the core's own precedence, behind one
// function-local static. Each reader resolves its key once and caches the answer.
namespace wxlvk::settings
{
    namespace detail
    {
        /**
         * The WarcraftXL.cfg entries, parsed once per DLL.
         *
         * The core's own reader looks for WarcraftXL.cfg in the working directory and falls back one
         * level up, because the client is sometimes launched from a subdirectory. That two-path
         * precedence cannot be expressed through wxl::ext::config::Raw (its File() helper caches
         * whichever path the first caller passed), so both lookups are done here on
         * wxl::cfg::ParseFile, the primitive both sides share: same line format, same comments.
         *
         * @return map entries
         */
        inline const std::unordered_map<std::string, std::string>& CfgFile()
        {
            static const std::unordered_map<std::string, std::string> entries = [] {
                auto map = wxl::cfg::ParseFile("WarcraftXL.cfg");
                if (map.empty()) map = wxl::cfg::ParseFile("..\\WarcraftXL.cfg");
                return map;
            }();
            return entries;
        }

        /**
         * One key's raw value: the environment first, then WarcraftXL.cfg -- the core's precedence.
         *
         * @param string name : the key
         * @param string buf : receives the NUL-terminated value
         * @param size_t cap : capacity of buf
         * @return bool found : false when neither source carries a non-empty value
         */
        inline bool Raw(const char* name, char* buf, size_t cap)
        {
            const DWORD n = GetEnvironmentVariableA(name, buf, static_cast<DWORD>(cap));
            if (n > 0 && n < cap) return true;

            const auto& cfg = CfgFile();
            const auto it = cfg.find(name);
            if (it == cfg.end() || it->second.empty() || it->second.size() + 1 > cap) return false;
            std::memcpy(buf, it->second.c_str(), it->second.size() + 1);
            return true;
        }

        /// The spellings that mean "off" for every key, so a key can be left in a .cfg and flipped
        /// rather than deleted. Matched before wxl::cfg::Truthy, which only rejects a leading
        /// 0 / n / f.
        inline bool OffWord(const char* value)
        {
            return _stricmp(value, "off") == 0 || _stricmp(value, "none") == 0
                   || _stricmp(value, "stock") == 0 || _stricmp(value, "no") == 0
                   || _stricmp(value, "false") == 0 || _stricmp(value, "0") == 0;
        }

        /**
         * One plain on/off key.
         *
         * @param string key : the key to read
         * @return bool wanted : false when unset or set to an off spelling
         */
        inline bool Flag(const char* key)
        {
            char value[32] = {};
            if (!Raw(key, value, sizeof value)) return false;
            if (OffWord(value)) return false;
            return wxl::cfg::Truthy(value, false);
        }
    }

    // --- WXL_VK_DEVICE ---------------------------------------------------------------------------

    /// The key that turns the extension on: `native` puts the native Vulkan device in place of the
    /// engine's D3D9 backend. Unset (or any other value), the extension registers nothing and hooks
    /// nothing, and the client runs its stock D3D9 device.
    constexpr const char* kDeviceKey = "WXL_VK_DEVICE";

    /// What WXL_VK_DEVICE says.
    enum class DeviceMode
    {
        Off,          ///< unset or an off spelling: the extension stays inert
        Native,       ///< `native`
        Unrecognised, ///< any other value: inert, and said so once at load
    };

    /**
     * What WXL_VK_DEVICE asked for. Resolved once.
     *
     * Only the word `native` turns the device on. The device is the one object in the client whose
     * failure means no picture at all, so owning it takes a deliberate, unambiguous act; a value the
     * extension does not know is reported rather than guessed at.
     *
     * @return DeviceMode mode
     */
    inline DeviceMode DeviceModeWanted()
    {
        static const DeviceMode mode = [] {
            char value[32] = {};
            if (!detail::Raw(kDeviceKey, value, sizeof value)) return DeviceMode::Off;
            if (_stricmp(value, "native") == 0) return DeviceMode::Native;
            if (detail::OffWord(value)) return DeviceMode::Off;
            return DeviceMode::Unrecognised;
        }();
        return mode;
    }

    // --- WXL_VK_TRANSLATED -----------------------------------------------------------------------

    /// Whether the pipelines may build a pair from the offline-translated SPIR-V under
    /// shaders/translated/, and how it ranks against the hand-written rows in
    /// src/vulkan/Pipelines.cpp:
    ///
    ///   0        never: hand-written rows only (the UI and the fixed function draw, the world does not)
    ///   1        fill in: a key with no hand-written row is built from the translated modules (default)
    ///   2        prefer: the translated modules win even where a hand-written row exists
    constexpr const char* kTranslatedKey = "WXL_VK_TRANSLATED";

    /// WXL_VK_TRANSLATED's three modes.
    enum : unsigned
    {
        kTranslatedOff    = 0,
        kTranslatedFill   = 1,
        kTranslatedPrefer = 2,
    };

    /**
     * What WXL_VK_TRANSLATED asked for. Resolved once.
     *
     * Unset is kTranslatedFill. The off spellings select 0; "2", "prefer", "force" and "all" select
     * 2; "1", "fill" and every other truthy spelling select 1. An unrecognised word falls back to
     * the default rather than to off, so a typo cannot silently remove the world.
     *
     * @return unsigned mode : one of kTranslated*
     */
    inline unsigned TranslatedMode()
    {
        static const unsigned mode = [] () -> unsigned {
            char value[32] = {};
            if (!detail::Raw(kTranslatedKey, value, sizeof value)) return kTranslatedFill;
            if (detail::OffWord(value)) return kTranslatedOff;
            if (_stricmp(value, "2") == 0 || _stricmp(value, "prefer") == 0
                || _stricmp(value, "force") == 0 || _stricmp(value, "all") == 0)
                return kTranslatedPrefer;
            if (_stricmp(value, "1") == 0 || _stricmp(value, "fill") == 0)
                return kTranslatedFill;
            return wxl::cfg::Truthy(value, true) ? kTranslatedFill : kTranslatedOff;
        }();
        return mode;
    }

    // --- WXL_VK_VALIDATION -----------------------------------------------------------------------

    /// Requests VK_LAYER_KHRONOS_validation and VK_EXT_debug_utils. Default off: the layer is a
    /// developer tool, and its allocations are not small in a 32-bit process.
    constexpr const char* kValidationKey = "WXL_VK_VALIDATION";

    /// Whether validation was asked for. Requested is not enabled: the instance only receives the
    /// layer when the loader reports it, and the log says which of the two happened.
    inline bool ValidationWanted()
    {
        static const bool wanted = detail::Flag(kValidationKey);
        return wanted;
    }

    // --- WXL_VK_STATS ----------------------------------------------------------------------------

    /// Turns on the detailed counters: one block of INFO lines about once a minute (draws, frame,
    /// pools, textures, targets, queries, CPU time, memory, constants) and a full block at device
    /// destruction. Default off: the log then holds errors, warnings and one summary line at
    /// shutdown.
    constexpr const char* kStatsKey = "WXL_VK_STATS";

    /// Whether the detailed counters are logged.
    inline bool StatsWanted()
    {
        static const bool wanted = detail::Flag(kStatsKey);
        return wanted;
    }

    // --- WXL_VK_BONES ----------------------------------------------------------------------------

    /// The bone-palette path (src/vulkan/Pipelines.cpp): a 256-row draw takes its module's `bones`
    /// variant and the palette ring. Default on; off keeps every 256-row draw on the 256-row
    /// modules, for an A/B comparison of the two paths.
    constexpr const char* kBonesKey = "WXL_VK_BONES";

    /// Whether the bone-palette path is allowed. Unset is on.
    inline bool BonesWanted()
    {
        static const bool wanted = [] () -> bool {
            char value[32] = {};
            if (!detail::Raw(kBonesKey, value, sizeof value)) return true;
            return !detail::OffWord(value);
        }();
        return wanted;
    }

    // --- WXL_VK_GBUFFER --------------------------------------------------------------------------

    /// The G-buffer (src/native/present/, docs/architecture.md "The G-buffer"): while a consumer asks
    /// for it (wxl.vk.render RequestGBuffer), the opaque world's pass carries the albedo, normal and
    /// light attachments and the modules' `gbuf` variants draw. Default allowed; off refuses every
    /// request, for an A/B of the attachments' cost.
    constexpr const char* kGBufferKey = "WXL_VK_GBUFFER";

    /// Whether the G-buffer may be made. Unset is allowed.
    inline bool GBufferAllowed()
    {
        static const bool allowed = [] () -> bool {
            char value[32] = {};
            if (!detail::Raw(kGBufferKey, value, sizeof value)) return true;
            return !detail::OffWord(value);
        }();
        return allowed;
    }

    // --- WXL_VK_ANISOTROPY, WXL_VK_MIP_BIAS --------------------------------------------------------

    /// The anisotropic filtering of the world's mipmapped textures (src/vulkan/pipelines/Samplers.cpp,
    /// the world variant): the most samples along a pixel's footprint, clamped to the adapter's
    /// limit, where it is more than the engine's own texture option asks for. 0 or 1 leaves the
    /// engine's filter. Default 16. Never for the interface, the text or a render target.
    constexpr const char* kAnisotropyKey = "WXL_VK_ANISOTROPY";

    /// The mip LOD bias of the same samplers, -1 .. 1: below 0 a sharper level is taken (the texture
    /// shimmers sooner), above 0 a softer one. Default -0.3.
    constexpr const char* kMipBiasKey = "WXL_VK_MIP_BIAS";

    /**
     * What WXL_VK_ANISOTROPY asked for. Resolved once.
     *
     * Unset is 16; an off spelling, or a number below 2, is no anisotropy of the device's own (0);
     * a number above 16 is 16; an unrecognised word falls back to the default.
     *
     * @return float samples : 0, or 2 .. 16 (the adapter's limit is applied by the caller)
     */
    inline float AnisotropyWanted()
    {
        static const float wanted = [] () -> float {
            char value[32] = {};
            if (!detail::Raw(kAnisotropyKey, value, sizeof value)) return 16.0f;
            if (detail::OffWord(value)) return 0.0f;
            char* end = nullptr;
            const double n = std::strtod(value, &end);
            if (end == value) return 16.0f;
            if (n < 2.0) return 0.0f;
            return static_cast<float>(n > 16.0 ? 16.0 : n);
        }();
        return wanted;
    }

    /**
     * What WXL_VK_MIP_BIAS asked for. Resolved once.
     *
     * Unset is -0.3; an off spelling is 0; a number is clamped to -1 .. 1 (within the 2 every Vulkan
     * device accepts); an unrecognised word falls back to the default.
     *
     * @return float bias
     */
    inline float MipBiasWanted()
    {
        static const float wanted = [] () -> float {
            char value[32] = {};
            if (!detail::Raw(kMipBiasKey, value, sizeof value)) return -0.3f;
            if (detail::OffWord(value)) return 0.0f;
            char* end = nullptr;
            const double n = std::strtod(value, &end);
            if (end == value) return -0.3f;
            return static_cast<float>(n < -1.0 ? -1.0 : (n > 1.0 ? 1.0 : n));
        }();
        return wanted;
    }

    // --- WXL_VK_CAPS -----------------------------------------------------------------------------

    /// The device's optional capabilities (src/vulkan/context/Caps.cpp): an instance and a device
    /// of Vulkan 1.1 or 1.2 when the loader and the adapter offer it, and the optional features the adapter
    /// reports enabled -- FP16 and INT8 arithmetic, 16-bit storage, the extended storage image formats,
    /// multiview, multi-draw indirect, VK_EXT_memory_budget, VK_EXT_shader_viewport_index_layer -- for the
    /// modules to use (wxl.vk.render GetDeviceCaps). Nothing the device itself draws changes with them.
    /// Default on; off is the Vulkan 1.0 device with the features it had before, for an A/B or a driver
    /// that misbehaves with the newer version.
    constexpr const char* kCapsKey = "WXL_VK_CAPS";

    // --- WXL_VK_RECORD_SPLIT ----------------------------------------------------------------------

    /// The draw's recording split into its phases (src/vulkan/pipelines/Record.cpp): the key and the
    /// pipeline, the textures / slice / descriptor set, the constants' copy, the binds, the shadow capture, the draw
    /// call -- each phase timed with the CPU's time-stamp counter (a few ns) and logged with the stats block as a cost
    /// per draw. Default off: the split adds six counter reads to every draw.
    constexpr const char* kRecordSplitKey = "WXL_VK_RECORD_SPLIT";

    /// Whether the record split is measured. Unset is off.
    inline bool RecordSplitWanted()
    {
        static const bool wanted = detail::Flag(kRecordSplitKey);
        return wanted;
    }

    /// Whether the optional capabilities are asked for. Unset is on.
    inline bool CapsWanted()
    {
        static const bool wanted = [] () -> bool {
            char value[32] = {};
            if (!detail::Raw(kCapsKey, value, sizeof value)) return true;
            return !detail::OffWord(value);
        }();
        return wanted;
    }
}
