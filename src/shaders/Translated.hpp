// The translated-module registry: which CGxShader records, by name, have a known translated module.
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

#include "vulkan/Manifest.hpp"

#include <cstdint>

// The bridge between the manifest (keyed by the MD5 of a record's bytecode) and the draw path (keyed
// by the record's name, which is all a draw::Cmd carries). The name is a stable key for this archive
// set, the hash a stable key for the bytes, so the hash is taken once per record when slot 72
// creates it. Nothing here runs per draw: the draw reader takes a bound shader's record from the
// CGxShader itself (+0x20, filed by slot 72) and the pipelines resolve by name at a build.
//
// Everything here runs on the render thread; there are no locks. Records are added only at create
// time, and an entry is complete before the count that publishes it is bumped.
namespace wxlvk::shaders::translated
{
    /**
     * Loads the manifest and proves the MD5 provider (RFC 1321's "abc" vector). Idempotent. Needs no
     * Vulkan device: the manifest is a text file and the hash is Win32. WXL_VK_TRANSLATED=0 refuses,
     * and then every record is unknown and only the hand-written rows can make a shader valid.
     *
     * @return bool armed : true when records can be hashed and looked up
     */
    bool Install();

    /**
     * Slot 72's half: hashes CGxShader +0x50 / +0x4C (the DX9 token stream, exactly the bytes the
     * translator hashed), looks the hash up in the manifest, and files the answer under the record's
     * name so Resolve() can answer for it.
     *
     * @param void* shader : the CGxShader
     * @param bool pixel : target 4 (pixel) or 0 (vertex)
     * @return pointer record : the manifest row for the bytes, null when unknown or not armed
     */
    const wxlvk::vk::manifest::Record* Create(void* shader, bool pixel);

    /**
     * The manifest record for a shader name, for one stage.
     *
     * The registry answers first: a record this session created is known or unknown by its bytes,
     * and an unknown record is null here even when the manifest carries the name, because the bytes
     * differ from the ones translated. A name the registry has never seen falls back to the
     * manifest's name table and is logged once, so a lookup resolved on the weaker key is visible.
     *
     * @param string name : rs(77)->name or rs(78)->name, as draw::Cmd carries it
     * @param bool pixel : which stage
     * @return pointer record : null when unknown, unregistered and unnamed, or nothing is loaded
     */
    const wxlvk::vk::manifest::Record* Resolve(const char* name, bool pixel);
}
