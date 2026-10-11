// The native device's stub slots: must not reach D3D, and have nothing real to do.
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

#include "native/Slots.hpp"

#include "native/Engine.hpp"

#include "wxl/Script.hpp"

#include <cstdint>
#include <cstdio>

// A stub answers what keeps the ENGINE's own bookkeeping in its steady state while nothing reaches a
// GPU: no-op, false, 0, "not valid" -- never a value that makes the engine retry in a loop or poison
// the frame. Each logs one DEBUG line the first time it is reached and counts every call, so a stub
// the engine does reach shows up in the stats.
namespace
{
    namespace slots = wxlvk::device::slots;

    using wxlvk::device::kSlotCount;
    using wxlvk::device::kSlotNames;

    bool     g_logged[kSlotCount];
    uint64_t g_calls[kSlotCount];

    void StubHit(unsigned slot, const char* answer)
    {
        ++g_calls[slot];
        if (g_logged[slot]) return;
        g_logged[slot] = true;
        wxl::ScriptMgr::Log(WXL_LOG_DEBUG, "native: stub slot %u %s reached; %s", slot,
                            kSlotNames[slot], answer);
    }
}

namespace wxlvk::native::impl
{
    int __fastcall DeviceCreateOnWindow(void* /*device*/, void* /*edx*/, HWND /*window*/,
                                        const CGxFormat* /*format*/)
    {
        // No caller is known; refusing is what makes GxDevCreate's caller fall back.
        StubHit(slots::kDeviceCreateOnWindow, "returns 0 (create on a foreign window is not done)");
        return 0;
    }

    void __fastcall LogCrashInfo(void* /*device*/, void* /*edx*/, char* buffer, uint32_t size)
    {
        StubHit(slots::kLogCrashInfo, "writes a one-line backend name");
        // D3D writes the adapter strings; an honest one-liner is more useful in a crash report than
        // an empty buffer.
        if (buffer && size) std::snprintf(buffer, size, "wxl-vulkan-api native device\n");
    }

    int __fastcall IShaderReload(void* /*device*/, void* /*edx*/, CGxShader* /*shader*/,
                                 const char* /*directory*/, const char* /*name*/)
    {
        // D3D's (0x006A5D50) returns 1 on every path, so 1 is the answer callers were written for.
        StubHit(slots::kIShaderReload, "returns 1, nothing reloaded");
        return 1;
    }

    void LogStubStats()
    {
        unsigned reached = 0;
        uint64_t total   = 0;
        for (unsigned i = 0; i < kSlotCount; ++i)
            if (g_calls[i])
            {
                ++reached;
                total += g_calls[i];
            }
        wxl::ScriptMgr::Log(WXL_LOG_INFO, "native: stubs: %u stub slot(s) reached, %llu call(s)",
                            reached, static_cast<unsigned long long>(total));
    }
}
