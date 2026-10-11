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

#pragma once

#include "vulkan/Vk.hpp"

// The .spv files are versioned under shaders/ and deployed beside the DLL as
// Extensions/wxl-vulkan-api/shaders/ (docs/architecture.md, "Shaders"), so a module is found
// relative to the DLL through src/common/ModulePath.hpp and never relative to the process. This file reads one, checks it is SPIR-V, and creates the VkShaderModule once; the
// pipeline cache in src/vulkan/Pipelines.cpp asks for modules by stem ("ui_vert") and never
// touches a file.
//
// Every function here must run on the render thread, like the rest of src/vulkan/: the cache is
// not synchronised, and the device calls it makes are not either.
namespace wxlvk::vk::shaders
{
    /**
     * Prepares the cache. Idempotent. Needs a live context, whose device every module is created
     * on; checks the two entry points this module calls, once, here.
     *
     * @return bool ok : false when there is no context or an entry point is missing (logged by
     *         name)
     */
    bool Init();

    /**
     * The module for <module dir>/shaders/<name>.spv, created on first use and cached by name.
     *
     * The file is read with plain CRT I/O; its size must be a multiple of four and its first word
     * the SPIR-V magic 0x07230203, else it is refused as not a module -- a corrupt or wrong file
     * fed to vkCreateShaderModule is undefined behaviour on a driver without validation, not an
     * error. A success is logged once per name with the byte count; a failure is logged once per
     * name and remembered, so a missing file costs one open per session and not one per draw.
     *
     * @param string name : the stem, without the folder or the extension, e.g. "ui_vert"; at most
     *        191 characters
     * @return VkShaderModule module : VK_NULL_HANDLE when the file is missing or invalid, when
     *         Init() has not succeeded, or when the cache of 512 names is full
     */
    VkShaderModule Get(const char* name);

    /**
     * The module for <module dir>/shaders/<relative>, the same cache keyed by the relative path.
     *
     * The translated modules live under shaders/translated/<vertex|pixel>/<md5>.spv, named by
     * the manifest (src/vulkan/Manifest.hpp) with forward slashes; this entry takes that path as
     * written, so a manifest row is loadable without a second spelling of it anywhere. Same
     * checks and the same once-per-name failure memory as Get().
     *
     * @param string relative : a path under shaders/, forward or back slashes, at most 191
     *        characters, e.g. "translated/vertex/0123...cdef.spv"
     * @return VkShaderModule module : as Get()
     */
    VkShaderModule GetPath(const char* relative);
}
