// The native device's log output: the shutdown summary, and the detailed counters of WXL_VK_STATS.
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

#include "native/Native.hpp"

#include "common/Settings.hpp"
#include "native/Perf.hpp"
#include "native/Present.hpp"
#include "native/Render.hpp"
#include "native/Slots.hpp"
#include "native/render/Budget.hpp"
#include "native/render/FrameStats.hpp"
#include "capture/Replay.hpp"
#include "stages/Chain.hpp"
#include "vulkan/Context.hpp"
#include "vulkan/Memory.hpp"
#include "vulkan/Pipelines.hpp"

#include "wxl/Script.hpp"

#include <cstdint>

// By default the extension logs errors, warnings and one summary line when the device is destroyed.
// WXL_VK_STATS=1 adds the detailed counters: one block about once a minute, and one at the end.
// Bring-up and one-off lines are DEBUG, visible with the core's WXL_LOG_LEVEL=debug.
namespace
{
    namespace present   = wxlvk::native::present;
    namespace pipelines = wxlvk::vk::pipelines;
    namespace impl      = wxlvk::native::impl;

    /// Presents between two saves of the pipeline cache (and two stats blocks): about a minute at
    /// 60 fps.
    constexpr uint64_t kStatsEvery = 3600;

    /// Every module's counters, in the order a reader of the log wants them.
    void LogStats()
    {
        impl::LogSceneStats();
        impl::LogDrawStats();
        impl::LogPoolStats();
        impl::LogTextureStats();
        impl::LogShaderStats();
        impl::LogTargetStats();
        impl::LogQueryStats();
        impl::LogDeviceStats();
        impl::LogCursorStats();
        impl::LogOverlayStats();
        wxlvk::native::render::LogStats();
        wxlvk::stages::chain::LogStats();
        wxlvk::capture::LogStats();
        impl::LogStubStats();
        wxlvk::native::render::framestats::Log("since the last line");   // before perf's: it reads perf's window
        wxlvk::native::perf::Log();
        wxlvk::native::render::budget::Log("stats block");
    }

    /// The one line every session ends with: frames, frame time, draws, memory.
    void LogSummary()
    {
        present::FrameTimes window = {}, session = {};
        present::GetFrameTimes(window, session, false);
        pipelines::Stats p = {};
        pipelines::GetStats(p);
        uint64_t allocs = 0, reserved = 0, inUse = 0;
        wxlvk::vk::memory::Stats(allocs, reserved, inUse);
        wxlvk::vk::memory::Detail md;
        wxlvk::vk::memory::GetDetail(md);
        wxl::ScriptMgr::Log(WXL_LOG_INFO,
                            "wxl-vulkan-api: %llu frame(s) presented, frame time avg %.2f / max %.2f "
                            "ms; %llu draw(s) received, %llu drawn, %u pipeline(s); memory %llu MiB "
                            "reserved (peak %llu, %llu mapped), %llu MiB in use (%llu allocation(s))",
                            static_cast<unsigned long long>(present::Count()),
                            session.frames ? session.sumMs / static_cast<double>(session.frames) : 0.0,
                            session.maxMs, static_cast<unsigned long long>(impl::DrawsTotal()),
                            static_cast<unsigned long long>(p.drawn), p.pipelines,
                            static_cast<unsigned long long>(reserved >> 20),
                            static_cast<unsigned long long>(md.reservedPeak >> 20),
                            static_cast<unsigned long long>(md.mapped >> 20),
                            static_cast<unsigned long long>(inUse >> 20),
                            static_cast<unsigned long long>(allocs));
    }
}

namespace wxlvk::native
{
    void OnPresent(uint64_t presents)
    {
        // The frame's figures and the memory budget (wxl.vk.render GetFrameStats, GetMemoryBudget), every present.
        wxlvk::native::render::framestats::OnPresent(presents);
        wxlvk::native::render::budget::OnPresent(presents);
        if (presents % kStatsEvery != 0) return;
        if (wxlvk::settings::StatsWanted())
        {
            LogStats();
            wxlvk::vk::context::LogValidationSummary();
        }
        pipelines::SavePipelineCache();
    }

    void OnDeviceDestroy()
    {
        if (wxlvk::settings::StatsWanted()) LogStats();
        wxlvk::vk::context::LogValidationSummary();
        LogSummary();
        // The session's frame times and memory, whatever WXL_VK_STATS says: the before / after lines.
        if (!wxlvk::settings::StatsWanted()) wxlvk::native::render::framestats::Log("session end");
        wxlvk::native::render::budget::Log("session end");
        pipelines::SavePipelineCache();
    }
}
