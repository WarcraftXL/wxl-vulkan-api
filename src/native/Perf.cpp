// The native device's CPU-time measurements.
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

#include "native/Perf.hpp"

#include "common/Settings.hpp"
#include "vulkan/Pipelines.hpp"

#include "wxl/Script.hpp"

#include <windows.h>

#include <cstdint>

namespace
{
    namespace perf = wxlvk::native::perf;

    constexpr double kSlowMs  = 33.0;
    constexpr double kHitchMs = 100.0;

    /// A slow frame is filed under a stall bucket when that bucket took at least this share of it.
    constexpr double kCauseShare = 1.0 / 3.0;

    /// Causes, in the order the line prints them: the four stall buckets, then "other".
    enum Cause : uint32_t
    {
        kCauseUpload = 0,
        kCausePipeline,
        kCauseFlush,
        kCauseFence,
        kCauseOther,
        kCauseCount
    };

    struct Agg
    {
        uint64_t frames = 0;
        double   sumMs[perf::kBucketCount] = {};
        double   maxMs[perf::kBucketCount] = {};
        uint64_t calls[perf::kBucketCount] = {};
        uint64_t slow = 0;
        uint64_t slowBy[kCauseCount] = {};
        uint64_t hitches = 0;
        uint64_t hitchBy[kCauseCount] = {};
    };

    struct State
    {
        double   tickMs = 0.0;   ///< milliseconds per QPC tick
        uint64_t open[perf::kBucketCount]  = {};   ///< ticks of the frame being accumulated
        uint64_t calls[perf::kBucketCount] = {};
        uint64_t buildTicks = 0;   ///< the pipelines' counter at the last EndFrame
        uint64_t buildCount = 0;
        Agg      window;
        Agg      session;
    };
    State g;

    double TickMs()
    {
        if (g.tickMs == 0.0)
        {
            LARGE_INTEGER f = {};
            QueryPerformanceFrequency(&f);
            g.tickMs = f.QuadPart ? 1000.0 / static_cast<double>(f.QuadPart) : 0.0;
        }
        return g.tickMs;
    }

    void File(Agg& a, const double ms[perf::kBucketCount], const uint64_t calls[perf::kBucketCount],
              double interval, Cause cause)
    {
        ++a.frames;
        for (uint32_t b = 0; b < perf::kBucketCount; ++b)
        {
            a.sumMs[b] += ms[b];
            if (ms[b] > a.maxMs[b]) a.maxMs[b] = ms[b];
            a.calls[b] += calls[b];
        }
        if (interval > kSlowMs)
        {
            ++a.slow;
            ++a.slowBy[cause];
        }
        if (interval > kHitchMs)
        {
            ++a.hitches;
            ++a.hitchBy[cause];
        }
    }

    void Line(const char* what, const Agg& a)
    {
        const double n = a.frames ? static_cast<double>(a.frames) : 1.0;
        const auto avg = [&](uint32_t b) { return a.sumMs[b] / n; };
        const auto per = [&](uint32_t b) { return static_cast<double>(a.calls[b]) / n; };
        wxl::ScriptMgr::Log(WXL_LOG_INFO,
                            "native: perf (%s): %llu frame(s); CPU ms per frame avg / max: slot 42 "
                            "%.3f / %.2f (%.0f calls), slot 54 %.3f / %.2f (%.0f), slot 55 %.3f / %.2f "
                            "(%.0f) [inclusive]; uploads %.3f / %.2f, pipeline builds %.3f / %.2f (%llu "
                            "built), flushes %.3f / %.2f (%llu), fence waits %.3f / %.2f",
                            what, static_cast<unsigned long long>(a.frames), avg(perf::kDraw),
                            a.maxMs[perf::kDraw], per(perf::kDraw), avg(perf::kBufLock), a.maxMs[perf::kBufLock],
                            per(perf::kBufLock), avg(perf::kBufUnlock), a.maxMs[perf::kBufUnlock],
                            per(perf::kBufUnlock), avg(perf::kUpload), a.maxMs[perf::kUpload],
                            avg(perf::kPipelineBuild), a.maxMs[perf::kPipelineBuild],
                            static_cast<unsigned long long>(a.calls[perf::kPipelineBuild]), avg(perf::kFlush),
                            a.maxMs[perf::kFlush], static_cast<unsigned long long>(a.calls[perf::kFlush]),
                            avg(perf::kFenceWait), a.maxMs[perf::kFenceWait]);
        // Per draw as well as per frame: the draw count moves with the scene, the cost per draw
        // is what the device's own work is measured by.
        const double draws = a.calls[perf::kDraw] ? static_cast<double>(a.calls[perf::kDraw]) : 1.0;
        const auto us = [&](uint32_t b) { return a.sumMs[b] * 1000.0 / draws; };
        wxl::ScriptMgr::Log(WXL_LOG_INFO,
                            "native: perf (%s): slot 42 per frame / per draw: sync %.3f ms / %.2f us, "
                            "prepare %.3f / %.2f, build %.3f / %.2f, record %.3f / %.2f; whole slot "
                            "%.3f / %.2f",
                            what, avg(perf::kDrawSync), us(perf::kDrawSync), avg(perf::kDrawPrepare),
                            us(perf::kDrawPrepare), avg(perf::kDrawBuild), us(perf::kDrawBuild),
                            avg(perf::kDrawRecord), us(perf::kDrawRecord), avg(perf::kDraw), us(perf::kDraw));
        wxl::ScriptMgr::Log(WXL_LOG_INFO,
                            "native: perf (%s): %llu frame(s) over 33 ms -- uploads %llu, pipeline "
                            "builds %llu, flushes %llu, fence waits %llu, other (engine CPU, cap, present) "
                            "%llu; %llu over 100 ms -- uploads %llu, pipeline builds %llu, flushes %llu, "
                            "fence waits %llu, other %llu",
                            what, static_cast<unsigned long long>(a.slow),
                            static_cast<unsigned long long>(a.slowBy[kCauseUpload]),
                            static_cast<unsigned long long>(a.slowBy[kCausePipeline]),
                            static_cast<unsigned long long>(a.slowBy[kCauseFlush]),
                            static_cast<unsigned long long>(a.slowBy[kCauseFence]),
                            static_cast<unsigned long long>(a.slowBy[kCauseOther]),
                            static_cast<unsigned long long>(a.hitches),
                            static_cast<unsigned long long>(a.hitchBy[kCauseUpload]),
                            static_cast<unsigned long long>(a.hitchBy[kCausePipeline]),
                            static_cast<unsigned long long>(a.hitchBy[kCauseFlush]),
                            static_cast<unsigned long long>(a.hitchBy[kCauseFence]),
                            static_cast<unsigned long long>(a.hitchBy[kCauseOther]));
    }
}

namespace wxlvk::native::perf
{
    namespace detail
    {
        bool enabled = false;
    }

    void Init() { detail::enabled = wxlvk::settings::StatsWanted(); }

    uint64_t Now()
    {
        LARGE_INTEGER t;
        QueryPerformanceCounter(&t);
        return static_cast<uint64_t>(t.QuadPart);
    }

    void Add(Bucket b, uint64_t ticks)
    {
        if (b >= kBucketCount) return;
        g.open[b] += ticks;
        ++g.calls[b];
    }

    void EndFrame(double intervalMs)
    {
        // The pipelines keep their own counter (src/vulkan/Pipelines.cpp knows nothing of this
        // file): its growth since the last frame is this frame's builds.
        uint64_t ticks = 0, count = 0;
        wxlvk::vk::pipelines::BuildCounters(ticks, count);
        if (ticks >= g.buildTicks && count >= g.buildCount)
        {
            g.open[kPipelineBuild]  += ticks - g.buildTicks;
            g.calls[kPipelineBuild] += count - g.buildCount;
        }
        g.buildTicks = ticks;
        g.buildCount = count;

        const double tick = TickMs();
        double ms[kBucketCount];
        for (uint32_t b = 0; b < kBucketCount; ++b) ms[b] = static_cast<double>(g.open[b]) * tick;

        Cause  cause = kCauseOther;
        double worst = intervalMs * kCauseShare;
        const struct { Bucket b; Cause c; } stalls[] = {
            { kUpload, kCauseUpload }, { kPipelineBuild, kCausePipeline }, { kFlush, kCauseFlush },
            { kFenceWait, kCauseFence },
        };
        for (const auto& st : stalls)
            if (ms[st.b] >= worst)
            {
                worst = ms[st.b];
                cause = st.c;
            }
        File(g.window, ms, g.calls, intervalMs, cause);
        File(g.session, ms, g.calls, intervalMs, cause);
        for (uint32_t b = 0; b < kBucketCount; ++b)
        {
            g.open[b]  = 0;
            g.calls[b] = 0;
        }
    }

    bool WindowDrawMs(double& drawMs, double& recordMs)
    {
        const Agg& a = g.window.frames ? g.window : g.session;
        if (!detail::enabled || !a.frames) return false;
        drawMs   = a.sumMs[kDraw] / static_cast<double>(a.frames);
        recordMs = a.sumMs[kDrawRecord] / static_cast<double>(a.frames);
        return true;
    }

    void Log()
    {
        Line("since the last line", g.window);
        Line("session", g.session);
        g.window = Agg();
    }
}
