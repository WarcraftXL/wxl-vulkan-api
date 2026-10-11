// The world scene pass's seam: where BeforeUi fires, where the capture and the jitter open and close.
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

#include "native/render/WorldPass.hpp"

#include "capture/Capture.hpp"
#include "native/Present.hpp"
#include "native/render/Jitter.hpp"
#include "stages/Chain.hpp"

#include "wxl/Script.hpp"

namespace
{
    constexpr const char* kWorldScenePoint = "Gx.WorldOnRender";
    using WorldSceneFn = void(__fastcall*)(void* worldFrame, void* edx);

    WorldSceneFn g_worldScene = nullptr;

    void __fastcall WorldScene(void* worldFrame, void* edx)
    {
        wxlvk::native::present::NoteWorld(true);   // the G-buffer class may be taken from here to AfterOpaqueWorld
        wxlvk::native::render::jitter::BeginWorld();
        wxlvk::capture::BeginWorld();
        g_worldScene(worldFrame, edx);
        wxlvk::capture::EndWorld();
        wxlvk::native::render::jitter::EndWorld();
        wxlvk::native::present::NoteWorld(false);
        wxlvk::stages::chain::Run(wxlvk::stages::Point::BeforeUi);
    }
}

namespace wxlvk::native::render::worldpass
{
    bool Install()
    {
        static bool installed = false, ok = false;
        if (installed) return ok;
        installed = true;
        const WXL_Api* api = wxl::ScriptMgr::Api();
        if (!api) return false;
        ok = api->HookAttachByName(kWorldScenePoint, reinterpret_cast<void*>(&WorldScene),
                                   reinterpret_cast<void**>(&g_worldScene), WXL_HOOK_DEFAULT_PRIORITY) != 0;
        if (!ok)
            wxl::ScriptMgr::Log(WXL_LOG_WARN, "render: the hook point %s could not be attached: no BeforeUi stage, no "
                                "draw capture, no jitter, no G-buffer", kWorldScenePoint);
        return ok;
    }
}
