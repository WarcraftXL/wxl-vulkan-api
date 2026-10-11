// The frame's figures (wxl.vk.render GetFrameStats) and the frame line.
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

#include "native/render/FrameStats.hpp"

#include "native/Perf.hpp"
#include "native/Present.hpp"
#include "native/Slots.hpp"
#include "native/render/DepthHistory.hpp"
#include "native/render/DepthPyramid.hpp"
#include "vulkan/Pipelines.hpp"

#include "wxl/Script.hpp"

#include <windows.h>

#include <cstdio>
#include <cstring>

namespace
{
    namespace present = wxlvk::native::present;

    constexpr uint32_t kWindow = present::kGpuWindow;

    struct State
    {
        LARGE_INTEGER freq  = {};
        LARGE_INTEGER last  = {};
        double        ms[kWindow] = {};
        uint32_t      count = 0, next = 0;
        double        lastMs = 0.0;
        uint64_t      drawsAt = 0, drawnAt = 0;   ///< the counters at the previous present
        uint32_t      draws = 0, drawn = 0;       ///< the latest frame's
        uint64_t      frames = 0;
        double        sessionSum = 0.0, sessionMax = 0.0;   ///< every interval of the session
        uint64_t      sessionCount = 0;
        uint64_t      sessionDraws = 0;
    };
    State g;

    void Window(double& avg, double& max)
    {
        avg = max = 0.0;
        if (!g.count) return;
        double sum = 0.0;
        for (uint32_t i = 0; i < g.count; ++i)
        {
            sum += g.ms[i];
            max = g.ms[i] > max ? g.ms[i] : max;
        }
        avg = sum / g.count;
    }
}

namespace wxlvk::native::render::framestats
{
    void OnPresent(uint64_t presents)
    {
        LARGE_INTEGER now = {};
        if (!g.freq.QuadPart) QueryPerformanceFrequency(&g.freq);
        QueryPerformanceCounter(&now);
        if (g.last.QuadPart && g.freq.QuadPart)
        {
            g.lastMs = static_cast<double>(now.QuadPart - g.last.QuadPart) * 1000.0 / static_cast<double>(g.freq.QuadPart);
            g.ms[g.next] = g.lastMs;
            g.next = (g.next + 1) % kWindow;
            if (g.count < kWindow) ++g.count;
            g.sessionSum += g.lastMs;
            g.sessionMax = g.lastMs > g.sessionMax ? g.lastMs : g.sessionMax;
            ++g.sessionCount;
        }
        g.last = now;
        const uint64_t draws = wxlvk::native::impl::DrawsTotal();
        const uint64_t drawn = wxlvk::vk::pipelines::DrawnCount();
        g.draws   = static_cast<uint32_t>(draws - g.drawsAt);
        g.drawn   = static_cast<uint32_t>(drawn - g.drawnAt);
        if (g.sessionCount) g.sessionDraws += g.draws;
        g.drawsAt = draws;
        g.drawnAt = drawn;
        g.frames  = presents;
    }

    bool Get(WXL_VkFrameStats& out)
    {
        if (!g.frames || out.structSize < sizeof(uint32_t)) return false;
        WXL_VkFrameStats f = {};
        f.structSize = out.structSize;
        f.frames     = g.frames;
        double avg = 0.0, max = 0.0;
        Window(avg, max);
        f.cpuFrameMs    = static_cast<float>(g.lastMs);
        f.cpuFrameAvgMs = static_cast<float>(avg);
        f.cpuFrameMaxMs = static_cast<float>(max);
        present::GpuTimes gt;
        present::GetGpuTimes(gt);
        if (gt.timed) f.flags |= WXL_VK_FRAME_GPU_TIMED;
        f.gpuFrameMs    = static_cast<float>(gt.frameLast);
        f.gpuFrameAvgMs = static_cast<float>(gt.frameAvg);
        f.gpuFrameMaxMs = static_cast<float>(gt.frameMax);
        f.depthCopyMs   = static_cast<float>(gt.laneAvg[present::kGpuDepthCopy]);
        f.pyramidMs     = static_cast<float>(gt.laneAvg[present::kGpuPyramid]);
        f.draws         = g.draws;
        f.drawn         = g.drawn;
        f.passBreaks    = gt.passBreaks;
        double drawMs = -1.0, recordMs = -1.0;
        if (wxlvk::native::perf::WindowDrawMs(drawMs, recordMs)) f.flags |= WXL_VK_FRAME_CPU_SPLIT;
        f.drawCpuMs   = static_cast<float>(drawMs);
        f.recordCpuMs = static_cast<float>(recordMs);
        const size_t n = out.structSize < sizeof f ? out.structSize : sizeof f;
        std::memcpy(&out, &f, n);
        out.structSize = static_cast<uint32_t>(n);
        return true;
    }

    void Log(const char* when)
    {
        WXL_VkFrameStats f = {};
        f.structSize = sizeof f;
        if (!Get(f)) return;
        char gpu[160];
        if (f.flags & WXL_VK_FRAME_GPU_TIMED)
            std::snprintf(gpu, sizeof gpu, "gpu %.2f ms (max %.2f, the latest %.2f)", f.gpuFrameAvgMs, f.gpuFrameMaxMs, f.gpuFrameMs);
        else
            std::snprintf(gpu, sizeof gpu, "gpu unknown (no timestamps)");
        char copy[32] = "not made", pyramid[32] = "not built";
        if (f.depthCopyMs >= 0.0f) std::snprintf(copy, sizeof copy, "%.3f ms", f.depthCopyMs);
        if (f.pyramidMs >= 0.0f) std::snprintf(pyramid, sizeof pyramid, "%.3f ms", f.pyramidMs);
        char split[96] = "";
        if (f.flags & WXL_VK_FRAME_CPU_SPLIT)
            std::snprintf(split, sizeof split, "; draw slot %.3f ms a frame, recording %.3f", f.drawCpuMs, f.recordCpuMs);
        wxl::ScriptMgr::Log(WXL_LOG_INFO,
                            "wxl-vulkan-api: frame (%s, the last %u frames): cpu %.2f ms (max %.2f), %s; opaque depth copy %s, depth "
                            "pyramid %s; %u draw(s) received, "
                            "%u recorded, %u pass break(s) the latest frame%s",
                            when, g.count, f.cpuFrameAvgMs, f.cpuFrameMaxMs, gpu, copy, pyramid, f.draws, f.drawn, f.passBreaks, split);
        // The session's whole: what a before / after of two sessions over the same walk compares.
        present::GpuTimes gt;
        present::GetGpuTimes(gt);
        if (g.sessionCount)
            wxl::ScriptMgr::Log(WXL_LOG_INFO,
                                "wxl-vulkan-api: frame (%s, the session's %llu frames): cpu %.2f ms (max %.2f), gpu %.2f ms (max %.2f, "
                                "%llu timed), %.0f draw(s) a frame",
                                when, static_cast<unsigned long long>(g.sessionCount), g.sessionSum / static_cast<double>(g.sessionCount),
                                g.sessionMax, gt.frameSessionAvg, gt.frameSessionMax,
                                static_cast<unsigned long long>(gt.frameSessionCount),
                                static_cast<double>(g.sessionDraws) / static_cast<double>(g.sessionCount));
    }
}
