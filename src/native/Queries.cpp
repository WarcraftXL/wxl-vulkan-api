// The native device's occlusion queries: slots 62-67 (cgxdevice 10.1).
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
#include "native/Present.hpp"

#include "common/Settings.hpp"

#include "wxl/Script.hpp"
#include "wxl/game/Binding.hpp"
#include "wxl/offsets/engine/GxDevice.hpp"

#include <cstdint>
#include <new>

// Only occlusion exists (EGxQueryType 0), and the only user found is the sun / moon glare
// (DayNight::DNGlare, decomp ~1186217): every frame it asks 66 param 1 ("ready?"); when ready it reads
// 67 and begins a new query (64), draws its quad, and ends it (65); when not, it keeps last frame's
// value and does not query. Native answers the same way D3D does:
//
//  - the CGxQuery's API word (+0x00) points at our record, made at 62 after the base allocation
//    and freed at 63 before the base frees the object;
//  - 64 / 65 begin and end a Vulkan occlusion query in the current pass (present::QueryStart /
//    QueryStop). A pass that ends inside an engine query ends its Vulkan query and the next pass
//    begins another; the result is their sum (Vulkan forbids a query across render pass instances);
//  - 66 param 0 is the result size (4), param 1 "ready": true once the frame that ran it completed
//    (a fence poll, never a wait) -- and true for a query never begun, as a new D3D query is
//    signalled;
//  - 67 waits as D3D's spin does: on the frame's fence, or, for a query of the frame being recorded,
//    by a mid-frame flush (present::FlushAndWait). The visible sample count, a uint32.
//
// Caps +0x314 offers them (src/native/Caps.cpp).
namespace
{
    namespace gxdev   = wxl::offsets::engine::gxdevice;
    namespace slots   = wxlvk::device::slots;
    namespace present = wxlvk::native::present;

    using wxl::game::At;
    using wxlvk::native::Base;

    struct NativeQuery
    {
        present::QueryRun run;
        uint32_t          type = 0;
    };

    struct Stats
    {
        uint64_t created;
        uint64_t destroyed;
        uint64_t begun;
        uint64_t beginFailed;
        uint64_t ended;
        uint64_t askedReady;
        uint64_t notReady;
        uint64_t read;
        uint64_t readFailed;
        uint64_t lastSamples;
        uint64_t maxSamples;
        uint64_t foreign;      ///< a CGxQuery with no record of ours (made before the device, or another type)
        uint64_t readNonZero;  ///< reads that returned a sample count above 0
        uint64_t readNoVk;     ///< reads of a query no Vulkan query ran for (no pass was open around its draws)
        uint64_t readNoDraw;   ///< reads of a query whose draws were all skipped, or that had none
        uint32_t readLogged;   ///< reads described in full (WXL_VK_STATS, kReadLines)
    };
    Stats g;

    /// With WXL_VK_STATS, the first reads are described in full: what the engine's query spanned.
    constexpr uint32_t kReadLines = 8;

    NativeQuery* Of(const void* query)
    {
        return query ? At<NativeQuery* const>(query, 0) : nullptr;
    }
}

namespace wxlvk::native::impl
{
    void __fastcall QueryCreate(void* device, void* edx, CGxQuery** out, uint32_t type)
    {
        Base<slots::kQueryCreate>()(device, edx, out, type);
        if (!out || !*out) return;
        NativeQuery* q = new (std::nothrow) NativeQuery();
        if (!q) return;
        q->type = type;
        At<NativeQuery*>(*out, 0) = q;
        ++g.created;
    }

    void __fastcall QueryDestroy(void* device, void* edx, CGxQuery** query)
    {
        if (query && *query)
            if (NativeQuery* q = Of(*query))
            {
                present::QueryForget(q->run);
                delete q;
                At<NativeQuery*>(*query, 0) = nullptr;
                ++g.destroyed;
            }
        Base<slots::kQueryDestroy>()(device, edx, query);
    }

    bool __fastcall QueryBegin(void* device, void* /*edx*/, CGxQuery* query)
    {
        NativeQuery* q = Of(query);
        if (!q || !At<uint32_t>(device, gxdev::kContext) || !present::DrawPathUp())
        {
            if (query && !q) ++g.foreign;
            ++g.beginFailed;
            return false;
        }
        if (!present::QueryStart(q->run))
        {
            ++g.beginFailed;
            return false;
        }
        ++g.begun;
        return true;
    }

    uint32_t __fastcall QueryEnd(void* device, void* /*edx*/, CGxQuery* query)
    {
        NativeQuery* q = Of(query);
        if (!q || !At<uint32_t>(device, gxdev::kContext)) return 0;
        present::QueryStop(q->run);
        ++g.ended;
        return 1;
    }

    int __fastcall QueryGetParam(void* device, void* /*edx*/, CGxQuery* query, uint32_t param, uint32_t* out)
    {
        // D3D 0x0068EA90: param 0 -> GetDataSize, returns 1; param 1 -> "GetData(NULL, 0, 0) == S_OK",
        // returns 0. Nothing without a context.
        NativeQuery* q = Of(query);
        if (!out) return 0;
        if (!q || !At<uint32_t>(device, gxdev::kContext))
        {
            *out = 0;
            return 0;
        }
        if (param == 0)
        {
            *out = sizeof(uint32_t);
            return 1;
        }
        ++g.askedReady;
        const bool ready = present::QueryReady(q->run);
        if (!ready) ++g.notReady;
        *out = ready ? 1u : 0u;
        return 0;
    }

    bool __fastcall QueryGetData(void* device, void* /*edx*/, CGxQuery* query, void* out)
    {
        NativeQuery* q = Of(query);
        if (!q || !out || !At<uint32_t>(device, gxdev::kContext) || !present::QueryWait(q->run))
        {
            ++g.readFailed;
            return false;
        }
        const uint64_t samples = q->run.samples;
        *static_cast<uint32_t*>(out) = samples > 0xFFFFFFFFull ? 0xFFFFFFFFu : static_cast<uint32_t>(samples);
        ++g.read;
        g.lastSamples = samples;
        if (samples > g.maxSamples) g.maxSamples = samples;
        if (samples) ++g.readNonZero;
        if (q->run.count == 0) ++g.readNoVk;
        if (q->run.drawn == 0) ++g.readNoDraw;
        if (g.readLogged < kReadLines && wxlvk::settings::StatsWanted())
        {
            ++g.readLogged;
            wxl::ScriptMgr::Log(WXL_LOG_INFO,
                                "native: query read %u: %llu sample(s) from %u Vulkan quer%s%s, frame %llu; %u draw(s) "
                                "inside, %u drawn",
                                g.readLogged, static_cast<unsigned long long>(samples), q->run.count,
                                q->run.count == 1 ? "y" : "ies", q->run.overflow ? " (overflowed: a lower bound)" : "",
                                static_cast<unsigned long long>(q->run.serial), q->run.draws, q->run.drawn);
        }
        return true;
    }

    void LogQueryStats()
    {
        present::Stats f = {};
        present::GetStats(f);
        wxl::ScriptMgr::Log(WXL_LOG_INFO,
                            "native: queries: %llu created / %llu destroyed; %llu begun (%llu refused), "
                            "%llu ended; ready asked %llu (%llu not ready); %llu read (%llu failed), last "
                            "%llu / max %llu sample(s), %llu above 0, %llu with no Vulkan query, %llu with no "
                            "draw drawn; Vulkan: %llu begun (%s), %llu split across a pass break, "
                            "%llu overflowed, %llu open at a present, %llu nested; %llu without a record",
                            static_cast<unsigned long long>(g.created),
                            static_cast<unsigned long long>(g.destroyed),
                            static_cast<unsigned long long>(g.begun),
                            static_cast<unsigned long long>(g.beginFailed),
                            static_cast<unsigned long long>(g.ended),
                            static_cast<unsigned long long>(g.askedReady),
                            static_cast<unsigned long long>(g.notReady),
                            static_cast<unsigned long long>(g.read),
                            static_cast<unsigned long long>(g.readFailed),
                            static_cast<unsigned long long>(g.lastSamples),
                            static_cast<unsigned long long>(g.maxSamples),
                            static_cast<unsigned long long>(g.readNonZero),
                            static_cast<unsigned long long>(g.readNoVk),
                            static_cast<unsigned long long>(g.readNoDraw),
                            static_cast<unsigned long long>(f.queriesBegun),
                            f.queryPrecise ? "precise" : "not precise",
                            static_cast<unsigned long long>(f.querySplits),
                            static_cast<unsigned long long>(f.queryOverflow),
                            static_cast<unsigned long long>(f.queryOpenAtPresent),
                            static_cast<unsigned long long>(f.queryNested),
                            static_cast<unsigned long long>(g.foreign));
    }
}
