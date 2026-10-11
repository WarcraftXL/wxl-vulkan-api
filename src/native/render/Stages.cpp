// wxl.vk.render's stages: the API's entries over src/stages/Chain.hpp.
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

#include "native/render/Stages.hpp"

#include "native/Render.hpp"
#include "native/render/Jitter.hpp"
#include "stages/Chain.hpp"

#include "wxl/Script.hpp"

namespace wxlvk::native::render::stages
{
    void* Register(const WXL_VkStageDesc* desc, const char* name)
    {
        if (!desc || desc->structSize < sizeof(WXL_VkStageDesc) || !desc->Run
            || (desc->point != WXL_VK_POINT_AFTER_OPAQUE_WORLD && desc->point != WXL_VK_POINT_BEFORE_UI
                && desc->point != WXL_VK_POINT_AFTER_LIQUIDS))
        {
            wxl::ScriptMgr::Log(WXL_LOG_WARN, "native: render: stage '%s' refused: a bad description or point",
                                name ? name : "?");
            return nullptr;
        }
        // AfterLiquids fires from its own seam: without it a stage there would never run.
        if (desc->point == WXL_VK_POINT_AFTER_LIQUIDS && !wxlvk::native::render::AfterLiquidsLive())
        {
            wxl::ScriptMgr::Log(WXL_LOG_WARN, "native: render: stage '%s' refused at AfterLiquids: the late liquid pass "
                                "is not detoured (see the log at load)", name ? name : "?");
            return nullptr;
        }
        void* stage = wxlvk::stages::chain::Add(*desc, name);
        if (!stage)
        {
            wxl::ScriptMgr::Log(WXL_LOG_WARN, "native: render: stage '%s' refused: %u stages already",
                                name ? name : "?", wxlvk::stages::chain::kMaxStages);
            return nullptr;
        }
        wxl::ScriptMgr::Log(WXL_LOG_INFO, "native: render: stage '%s' registered at %s, order %d%s%s", name ? name : "stage",
                            desc->point == WXL_VK_POINT_BEFORE_UI      ? "BeforeUi"
                            : desc->point == WXL_VK_POINT_AFTER_LIQUIDS ? "AfterLiquids"
                                                                         : "AfterOpaqueWorld",
                            desc->order,
                            (desc->flags & WXL_VK_STAGE_NEEDS_DEPTH) ? ", reads the depth" : "",
                            (desc->flags & WXL_VK_STAGE_WRITES_COLOR) ? ", writes the frame" : "");
        return stage;
    }

    void Unregister(void* stage) { wxlvk::stages::chain::Remove(stage); }

    int Stats(void* stage, WXL_VkStageStats* out)
    {
        if (!out || out->structSize < sizeof(uint32_t)) return 0;
        return wxlvk::stages::chain::Stats(stage, *out) ? 1 : 0;
    }

    void RequestJitter(int wanted) { wxlvk::native::render::jitter::Request(wanted != 0); }
}
