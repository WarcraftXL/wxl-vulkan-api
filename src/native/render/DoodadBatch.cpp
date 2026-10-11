// The engine's batched-doodad draw entry, bracketed.
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

#include "native/render/DoodadBatch.hpp"

#include "wxl/Script.hpp"

#include <cstdint>

namespace
{
    constexpr const char* kPoint = "M2.DrawBatchDoodad";

    /// __thiscall(this, elements, indices), ret 8 (Gx.hpp kDrawBatchDoodad), as this codebase's idiom
    /// for a hooked thiscall: fastcall with a dummy edx. Whatever it returns in eax is handed back.
    using Fn = uint32_t(__fastcall*)(void* ctx, void* edx, void* elements, void* indices);

    Fn       g_original = nullptr;
    uint32_t g_depth    = 0;
    bool     g_hooked   = false;

    uint32_t __fastcall Detour(void* ctx, void* edx, void* elements, void* indices)
    {
        ++g_depth;
        const uint32_t r = g_original(ctx, edx, elements, indices);
        --g_depth;
        return r;
    }
}

namespace wxlvk::native::render::doodadbatch
{
    bool Install()
    {
        static bool installed = false;
        if (installed) return g_hooked;
        installed = true;
        const WXL_Api* api = wxl::ScriptMgr::Api();
        if (!api) return false;
        g_hooked = api->HookAttachByName(kPoint, reinterpret_cast<void*>(&Detour), reinterpret_cast<void**>(&g_original),
                                         WXL_HOOK_DEFAULT_PRIORITY) != 0;
        if (!g_hooked)
            wxl::ScriptMgr::Log(WXL_LOG_WARN, "render: the hook point %s could not be attached: every model draw counts as a "
                                "mover for the shadow replay (the far caches lose the placed doodads)", kPoint);
        return g_hooked;
    }

    bool Inside() { return g_depth != 0; }
    bool Hooked() { return g_hooked; }
}
