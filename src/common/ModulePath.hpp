// Where this extension's own files live, resolved from the DLL rather than from the process.
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

#include <windows.h>

#include <string>

// This extension deploys as a self-contained folder:
//
//     Extensions/wxl-vulkan-api/
//     +-- wxl-vulkan-api.dll
//     +-- shaders/*.spv
//
// so anything it loads at runtime has to be found relative to **the DLL**, never relative to the
// process. Two independent reasons, and either one on its own would be enough:
//
//  - the working directory is not reliably the client's folder. The core's own config reader already
//    compensates for this by looking for WarcraftXL.cfg one level up as well, which is evidence from
//    inside this project rather than a general worry.
//  - even when it is, the client's folder is the wrong place: our files are a directory deeper, and
//    the point of the layout is that the extension's folder can be dropped in or deleted whole.
//
// GetModuleHandleEx with FROM_ADDRESS on a function of ours is what makes this self-locating: it
// asks the loader which module a given address belongs to, so it works without the DllMain handle
// being plumbed anywhere and without naming our own file, which would be a second place to keep the
// DLL's name correct.
namespace wxlvk::path
{
    namespace detail
    {
        /// A function whose address is inside this DLL. Only its address is used.
        inline void Anchor() {}
    }

    /**
     * The directory holding this DLL, with a trailing backslash. Empty if the loader refused.
     *
     * Resolved once. UNPINNED deliberately: the flag set is FROM_ADDRESS without
     * UNCHANGED_REFCOUNT, so the handle is reference-counted for the duration -- but since the only
     * use is an immediate GetModuleFileNameA, the reference is released right after and nothing is
     * held. The core keeps extensions loaded for the life of the process anyway.
     *
     * @return string dir : e.g. "...\\1.Clients\\Extensions\\wxl-vulkan-api\\", or "" on failure
     */
    inline const std::string& Dir()
    {
        static const std::string dir = [] () -> std::string {
            HMODULE self = nullptr;
            if (!GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS,
                                    reinterpret_cast<LPCSTR>(&detail::Anchor), &self)
                || !self)
                return std::string();

            // MAX_PATH is not a guarantee on a long path, so grow until the call fits rather than
            // truncating: a silently truncated path would turn into a confusing "file not found".
            std::string buf(MAX_PATH, '\0');
            for (;;)
            {
                const DWORD n = GetModuleFileNameA(self, buf.data(),
                                                   static_cast<DWORD>(buf.size()));
                if (n == 0)
                {
                    FreeLibrary(self);
                    return std::string();
                }
                if (n < buf.size())
                {
                    buf.resize(n);
                    break;
                }
                if (buf.size() >= 32768)   // the NT path ceiling; give up rather than loop forever
                {
                    FreeLibrary(self);
                    return std::string();
                }
                buf.resize(buf.size() * 2);
            }
            FreeLibrary(self);

            const size_t slash = buf.find_last_of("\\/");
            if (slash == std::string::npos) return std::string();
            return buf.substr(0, slash + 1);
        }();
        return dir;
    }

    /**
     * A path to one of our deployed files.
     *
     * @param string relative : a path under the extension's folder, e.g. "shaders\\blit.vert.spv"
     * @return string path : the absolute path, or `relative` unchanged if Dir() failed -- which
     *         leaves the caller's own open failing with a sensible name instead of an empty one
     */
    inline std::string Resolve(const char* relative)
    {
        const std::string& dir = Dir();
        return dir.empty() ? std::string(relative) : dir + relative;
    }
}
