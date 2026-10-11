// The rendering API the native device publishes: wxl.vk.render (wxl/VkRenderApi.h).
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


#include "native/api/Internal.hpp"

using namespace wxlvk::native::render::detail;

namespace wxlvk::native::render
{
    namespace detail
    {
        uint32_t filterCount = 0;
        bool     liquidsLive = false;   ///< the late liquid pass's seam is attached
    }

    bool AfterLiquidsLive() { return detail::liquidsLive; }

    void Install()
    {
        if (g.installed) return;
        g.installed = true;
        const WXL_Api* api = wxl::ScriptMgr::Api();
        if (!api) return;

        g_api.structSize       = sizeof g_api;
        g_api.apiVersion       = WXL_VK_RENDER_API_VERSION;
        g_api.GetDeviceInfo    = &GetDeviceInfo;
        g_api.RegisterPass     = &RegisterPass;
        g_api.UnregisterPass   = &UnregisterPass;
        g_api.AddDrawFilter    = &AddDrawFilter;
        g_api.RemoveDrawFilter = &RemoveDrawFilter;
        g_api.FilteredDraws    = &FilteredDraws;
        g_api.CreateBuffer     = &CreateBuffer;
        g_api.RetireBuffer     = &RetireBuffer;
        g_api.FlushBuffer      = &FlushBuffer;
        g_api.CreateImage      = &CreateImage;
        g_api.RetireImage      = &RetireImage;
        g_api.SetDrawFilterActive = &SetDrawFilterActive;
        g_api.RequestPreviousDepth = &RequestPreviousDepth;
        g_api.GetPreviousDepth     = &GetPreviousDepth;
        g_api.WatchTextures        = &WatchTextures;
        g_api.TakeWindModel        = &TakeWindModel;
        g_api.SetModelWind         = &SetModelWind;
        g_api.SetWind              = &SetWind;
        g_api.GetWindStats         = &GetWindStats;
        g_api.GetWindInstances     = &GetWindInstances;
        g_api.RegisterStage        = &RegisterStage;
        g_api.UnregisterStage      = &UnregisterStage;
        g_api.GetShadowPass        = &GetShadowPass;
        g_api.RequestDrawCapture   = &RequestDrawCapture;
        g_api.ReplayWorld          = &ReplayWorld;
        g_api.PublishSurface       = &wxlvk::native::render::surface::Publish;   // src/native/render/Surface.cpp
        g_api.GetSurface           = &wxlvk::native::render::surface::Get;
        g_api.SetModelWindEx       = &SetModelWindEx;
        g_api.PublishCanopy        = &wxlvk::native::render::canopy::Publish;   // src/native/render/Canopy.cpp
        g_api.GetCanopy            = &wxlvk::native::render::canopy::Get;
        g_api.GetStageStats        = &wxlvk::native::render::stages::Stats;           // src/native/render/Stages.cpp
        g_api.RequestJitter        = &wxlvk::native::render::stages::RequestJitter;
        g_api.RequestGBuffer       = &RequestGBuffer;   // src/native/api/Table.cpp over src/native/present/
        g_api.GetGBuffer           = &GetGBuffer;
        g_api.ReplayWorldViews     = &ReplayWorldViews;   // src/capture/Replay.cpp (a local light's cube in one pass)
        // The shared frame services: src/native/api/Table.cpp over src/vulkan/context/Caps.cpp and src/native/render/.
        g_api.GetDeviceCaps        = &GetDeviceCaps;
        g_api.GetMemoryBudget      = &GetMemoryBudget;
        g_api.MayKeepResident      = &MayKeepResident;
        g_api.RequestOpaqueDepth   = &RequestOpaqueDepth;
        g_api.GetOpaqueDepth       = &GetOpaqueDepth;
        g_api.RequestDepthPyramid  = &RequestDepthPyramid;
        g_api.GetDepthPyramid      = &GetDepthPyramid;
        g_api.GetFrameStats        = &GetFrameStats;

        const bool seam = api->HookAttachByName(kGroundEffectPoint, reinterpret_cast<void*>(&GroundEffectPass),
                                                reinterpret_cast<void**>(&g.groundEffectPass),
                                                WXL_HOOK_DEFAULT_PRIORITY) != 0;
        api->PublishInterface(WXL_VK_RENDER_API_NAME, WXL_VK_RENDER_API_VERSION, &g_api);
        wxl::ScriptMgr::Log(seam ? WXL_LOG_INFO : WXL_LOG_WARN,
                            seam ? "native: render: %s v%u published; AfterOpaqueWorld fires at the return "
                                   "of the engine's ground-effect pass (%s)"
                                 : "native: render: %s v%u published, but the hook point %s could not be "
                                   "attached: AfterOpaqueWorld never fires",
                            WXL_VK_RENDER_API_NAME, WXL_VK_RENDER_API_VERSION, kGroundEffectPoint);
        // AfterLiquids: the engine's late liquid pass, by address (offsets/engine/Liquid.hpp).
        liquidsLive = api->HookAttach(kLiquidPassName, wxl::offsets::engine::liquid::kWorldLiquidPass,
                                             reinterpret_cast<void*>(&LiquidPass), reinterpret_cast<void**>(&g.liquidPass),
                                             WXL_HOOK_DEFAULT_PRIORITY) != 0;
        wxl::ScriptMgr::Log(liquidsLive ? WXL_LOG_INFO : WXL_LOG_WARN,
                            liquidsLive ? "native: render: AfterLiquids fires at the return of the engine's late liquid pass "
                                      "(0x%08X)"
                                    : "native: render: the late liquid pass (0x%08X) could not be detoured: AfterLiquids "
                                      "never fires (its stages never run)",
                            static_cast<unsigned>(wxl::offsets::engine::liquid::kWorldLiquidPass));
    }

    uint32_t RecordShadowCasters(VkCommandBuffer cmd, VkRenderPass pass, const VkViewport& viewport, const VkRect2D& area,
                                 const float lightViewProj[16], uint32_t cascade)
    {
        PointState& ps = g.points[WXL_VK_POINT_SHADOW_CASTERS];
        if (ps.passes == 0) return 0;
        const void* dev = wxl::game::Read<const void*>(gx::kGxDevicePtr);
        if (!dev) return 0;
        WXL_VkSceneConstants scene;
        WXL_VkRenderContext  ctx = {};
        ctx.structSize      = sizeof ctx;
        ctx.point           = WXL_VK_POINT_SHADOW_CASTERS;
        ctx.serial          = present::Serial();
        ctx.completedSerial = present::Completed();
        ctx.frameSlot       = present::Slot();
        ctx.frameSlots      = pipelines::kFrameSlots;
        ctx.invocation      = cascade;
        ctx.hasDepth        = 1;
        ctx.cmd             = cmd;
        ctx.renderPass      = pass;
        ctx.extent          = area.extent;
        ctx.depthFormat     = wxlvk::capture::shadowpass::Format();
        ctx.colorFormat     = VK_FORMAT_UNDEFINED;
        ctx.viewport        = viewport;
        ctx.scissor         = area;
        ctx.viewportValid   = 1;
        ctx.scene           = &scene;
        // The camera's scene, with the light's matrices: a caster places its geometry with viewProj.
        BuildScene(dev, area.extent, scene);
        std::memcpy(scene.viewProj, lightViewProj, sizeof scene.viewProj);
        std::memcpy(scene.projection, lightViewProj, sizeof scene.projection);
        for (uint32_t i = 0; i < 16; ++i) scene.view[i] = (i % 5 == 0) ? 1.0f : 0.0f;
        uint32_t called = 0;
        for (Pass& p : g.passes)
        {
            if (!p.live || p.point != WXL_VK_POINT_SHADOW_CASTERS || !p.desc.Record) continue;
            p.desc.Record(p.desc.user, &ctx);
            ++p.records;
            ++called;
        }
        ++ps.fired;
        return called;
    }

    bool SceneNow(VkExtent2D extent, WXL_VkSceneConstants& scene, VkViewport& viewport, float jitter[2])
    {
        jitter[0] = jitter[1] = 0.0f;
        const void* dev = wxl::game::Read<const void*>(gx::kGxDevicePtr);
        if (!dev) return false;
        BuildScene(dev, extent, scene);
        VkRect2D   scissor = {};
        VkViewport plain   = {};
        if (!wxlvk::native::impl::PlaceDraw(dev, extent, viewport, scissor)) return false;
        // The same placement without the jitter: the difference is exactly what was applied, so the
        // caller never has to guess whether this moment was jittered.
        if (!wxlvk::native::impl::PlaceD3d(dev, extent, plain, scissor)) return true;
        jitter[0] = viewport.x - (plain.x + wxlvk::native::impl::kD3d9CentreShift);
        jitter[1] = viewport.y - (plain.y + wxlvk::native::impl::kD3d9CentreShift);
        return true;
    }

    bool FilterDraw(uintptr_t table)
    {
        const unsigned ids[2] = { gxdev::kRsVertexShader, gxdev::kRsPixelShader };
        for (unsigned stage = 0; stage < 2; ++stage)
        {
            const uintptr_t shader = wxl::game::At<const uint32_t>(table + ids[stage] * gxdev::kRsStride, gxdev::kRsValue);
            if (!shader) continue;
            int hit = -1;
            if (shader == g.lastShader[stage]) hit = g.lastFilter[stage];
            else
            {
                const char* name = wxl::game::At<const char* const>(shader, gxdev::kShaderName);
                if (name)
                    for (uint32_t i = 0; i < kMaxFilters; ++i)
                    {
                        const Filter& f = g.filters[i];
                        if (f.live && f.active && (f.stages & (1u << stage)) && Matches(f, name))
                        {
                            hit = static_cast<int>(i);
                            break;
                        }
                    }
                g.lastShader[stage] = shader;
                g.lastFilter[stage] = hit;
            }
            if (hit < 0) continue;
            Filter& f = g.filters[hit];
            ++f.hits;
            if (!f.saidFirst)
            {
                f.saidFirst = true;
                const char* name = wxl::game::At<const char* const>(shader, gxdev::kShaderName);
                wxl::ScriptMgr::Log(WXL_LOG_INFO, "native: render: first draw suppressed by filter '%s' (%s shader "
                                    "'%s', frame %llu)", f.name, stage == 0 ? "vertex" : "pixel",
                                    name ? name : "?", static_cast<unsigned long long>(present::Serial()));
            }
            return true;
        }
        return false;
    }

    void LogStats()
    {
        for (uint32_t p = 0; p < WXL_VK_POINT_COUNT; ++p)
        {
            const PointState& ps = g.points[p];
            if (ps.passes == 0 && ps.fired == 0) continue;
            wxl::ScriptMgr::Log(WXL_LOG_INFO,
                                "native: render: %s: %u pass(es), fired %llu, not fired %llu off the back "
                                "buffer / %llu without a frame, record skipped %llu (no pass), mid-pass skipped %llu "
                                "(no depth)",
                                kPointNames[p], ps.passes, static_cast<unsigned long long>(ps.fired),
                                static_cast<unsigned long long>(ps.offBackBuffer),
                                static_cast<unsigned long long>(ps.noFrame), static_cast<unsigned long long>(ps.noPass),
                                static_cast<unsigned long long>(ps.noDepth));
        }
        for (const Pass& p : g.passes)
            if (p.live)
                wxl::ScriptMgr::Log(WXL_LOG_INFO, "native: render: pass '%s' at %s: %llu prepare(s), %llu mid-pass(es), "
                                    "%llu record(s)", p.name, kPointNames[p.point],
                                    static_cast<unsigned long long>(p.prepares),
                                    static_cast<unsigned long long>(p.midPasses),
                                    static_cast<unsigned long long>(p.records));
        for (const Filter& f : g.filters)
            if (f.live)
                wxl::ScriptMgr::Log(WXL_LOG_INFO, "native: render: draw filter '%s' (%s): %llu draw(s) suppressed",
                                    f.name, f.owner, static_cast<unsigned long long>(f.hits));
        if (g.buffersMade || g.imagesMade || g.createFailed)
            wxl::ScriptMgr::Log(WXL_LOG_INFO,
                                "native: render: consumer resources: %llu buffer(s) made (%llu KiB), %llu retired; "
                                "%llu image(s) made, %llu retired; %llu create(s) failed",
                                static_cast<unsigned long long>(g.buffersMade),
                                static_cast<unsigned long long>(g.bufferBytes >> 10),
                                static_cast<unsigned long long>(g.buffersRetired),
                                static_cast<unsigned long long>(g.imagesMade),
                                static_cast<unsigned long long>(g.imagesRetired),
                                static_cast<unsigned long long>(g.createFailed));
    }
}
