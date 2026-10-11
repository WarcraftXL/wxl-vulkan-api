// The rendering API's shared internals: its passes, its filters, its resources, its state.
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

#include "native/Render.hpp"
#include "vulkan/Wind.hpp"

#include "native/Present.hpp"
#include "native/render/Stages.hpp"
#include "native/render/Canopy.hpp"
#include "native/render/Surface.hpp"
#include "native/render/WmoGroups.hpp"
#include "capture/Capture.hpp"
#include "capture/Replay.hpp"
#include "capture/ShadowPass.hpp"
#include "native/Slots.hpp"
#include "native/render/Budget.hpp"
#include "native/render/DepthHistory.hpp"
#include "native/render/DepthPyramid.hpp"
#include "native/render/FrameStats.hpp"
#include "stages/Chain.hpp"
#include "vulkan/Context.hpp"
#include "vulkan/Loader.hpp"
#include "vulkan/Memory.hpp"
#include "vulkan/Pipelines.hpp"
#include "vulkan/Vk.hpp"

#include "wxl/VkRenderApi.h"
#include "wxl/Script.hpp"
#include "wxl/game/Binding.hpp"
#include "wxl/offsets/engine/Camera.hpp"
#include "wxl/offsets/engine/Gx.hpp"
#include "wxl/offsets/engine/GxDevice.hpp"
#include "wxl/offsets/engine/Liquid.hpp"
#include "wxl/offsets/engine/Shader.hpp"

#include <windows.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <new>

// wxl.vk.render lives in src/native/api/ over this one header (its stages, depth history, jitter,
// surface and canopy are in src/native/render/). The public face is src/native/Render.hpp.
//
//   Scene   the scene constants at a point
//   Points  the insertion points' seams
//   Table   the published table and its entries
namespace wxlvk::native::render::detail
{
    namespace present   = wxlvk::native::present;
    namespace context   = wxlvk::vk::context;
    namespace loader    = wxlvk::vk::loader;
    namespace memory    = wxlvk::vk::memory;
    namespace pipelines = wxlvk::vk::pipelines;
    namespace gx        = wxl::offsets::engine::gx;
    namespace gxdev     = wxl::offsets::engine::gxdevice;
    namespace cam       = wxl::offsets::engine::camera;
    namespace shader    = wxl::offsets::engine::shader;

    constexpr uint32_t kMaxPasses   = 16;
    constexpr uint32_t kMaxFilters  = 16;
    constexpr size_t   kFilterChars = 64;

    /// The hook point AfterOpaqueWorld fires from, and its signature (no arguments, caller-cleaned).
    constexpr const char* kGroundEffectPoint = "Adt.RenderDetailDoodads";
    using GroundEffectPassFn = void(__cdecl*)();

    /// The engine's late liquid pass AfterLiquids fires from (the core's offsets: no named hook point yet),
    /// attached by address.
    constexpr const char* kLiquidPassName = "Liquid.WorldLiquidPass";

    const char* const kPointNames[WXL_VK_POINT_COUNT] = { "AfterOpaqueWorld", "ShadowCasters", "BeforeUi", "AfterLiquids" };

    struct Pass
    {
        bool                 live = false;
        uint32_t             point = 0;
        WXL_VkRenderPassDesc desc = {};
        const char*          name = "";
        uint64_t             prepares = 0;
        uint64_t             midPasses = 0;
        uint64_t             records  = 0;
    };

    struct Filter
    {
        bool        live = false;
        char        name[kFilterChars] = {};
        size_t      length = 0;
        uint32_t    stages = 0;
        const char* owner  = "";
        uint64_t    hits   = 0;
        bool        active = true;   ///< SetDrawFilterActive: a live filter that is paused drops nothing
        bool        saidFirst = false;
    };

    struct BufferRec
    {
        memory::Buffer buffer;
    };

    struct ImageRec
    {
        VkImage       image = VK_NULL_HANDLE;
        VkImageView   view  = VK_NULL_HANDLE;
        memory::Block block;
    };

    struct PointState
    {
        uint32_t passes = 0;        ///< live passes registered here
        uint64_t lastSerial = 0;    ///< the frame it last fired in
        uint32_t invocation = 0;    ///< fires in that frame so far
        uint64_t fired = 0;         ///< times it fired with passes to call
        uint64_t offBackBuffer = 0; ///< reached while a texture target was current (not fired)
        uint64_t noFrame = 0;       ///< reached with no frame or no draw path (not fired)
        uint64_t noPass = 0;        ///< Record skipped: no pass could be begun
        uint64_t noDepth = 0;       ///< MidPass / depth copy skipped: the depth could not be sampled
        bool     saidFirst = false;
    };

    struct State
    {
        bool               installed = false;
        GroundEffectPassFn groundEffectPass = nullptr;
        wxl::offsets::engine::liquid::WorldLiquidPassFn liquidPass = nullptr;   ///< the late liquid pass's trampoline
        Pass               passes[kMaxPasses];
        Filter             filters[kMaxFilters];
        PointState         points[WXL_VK_POINT_COUNT];

        // Filter lookups by shader object: one entry per stage, forgotten when the filters change.
        uintptr_t lastShader[2]  = {};
        int       lastFilter[2]  = { -1, -1 };

        LARGE_INTEGER qpcFrequency = {};
        LARGE_INTEGER qpcStart     = {};
        uint64_t      timeSerial   = 0;     ///< the frame `frameTime` was taken in
        double        frameTime    = 0.0;
        double        prevTime     = 0.0;

        uint64_t buffersMade = 0, buffersRetired = 0, bufferBytes = 0;
        uint64_t imagesMade = 0, imagesRetired = 0;
        uint64_t createFailed = 0;
    };
    extern State g;

    extern WXL_VkRenderApi g_api;


    // Scene.cpp
    void StackTop(const void* dev, unsigned xform, float out[16]);
    void Mul(const float a[16], const float b[16], float out[16]);
    void Eye(const float v[16], float out[4]);
    double Now();
    bool IsIdentity(const float m[16]);
    void BuildScene(const void* dev, VkExtent2D extent, WXL_VkSceneConstants& sc);

    // Points.cpp
    void WritesToDraws(VkCommandBuffer cmd);
    void Prepares(uint32_t point, WXL_VkRenderContext& ctx);
    void MidPasses(uint32_t point, WXL_VkRenderContext& ctx);
    void Fire(uint32_t point);
    void __cdecl GroundEffectPass();
    void __cdecl LiquidPass();

    // Table.cpp
    void* __cdecl RegisterStage(const WXL_VkStageDesc* desc, const char* name);
    void __cdecl UnregisterStage(void* stage);
    int __cdecl GetShadowPass(WXL_VkShadowPass* out);
    void __cdecl RequestDrawCapture(int wanted);
    void __cdecl RequestGBuffer(int wanted);
    int __cdecl GetGBuffer(WXL_VkGBuffer* out);
    int __cdecl ReplayWorld(const WXL_VkReplayDesc* desc, WXL_VkReplayStats* stats);
    int __cdecl ReplayWorldViews(const WXL_VkReplayViewsDesc* desc, WXL_VkReplayStats* stats);
    int __cdecl GetDeviceCaps(WXL_VkDeviceCaps* out);
    int __cdecl GetMemoryBudget(WXL_VkMemoryBudget* out);
    int __cdecl MayKeepResident(uint64_t extraBytes, uint32_t memoryClass);
    void __cdecl RequestOpaqueDepth(int wanted);
    int __cdecl GetOpaqueDepth(WXL_VkPreviousDepth* out);
    void __cdecl RequestDepthPyramid(int wanted);
    int __cdecl GetDepthPyramid(WXL_VkDepthPyramid* out);
    int __cdecl GetFrameStats(WXL_VkFrameStats* out);
    int __cdecl GetDeviceInfo(WXL_VkDeviceInfo* out);
    void* __cdecl RegisterPass(uint32_t point, const WXL_VkRenderPassDesc* desc, const char* name);
    void __cdecl UnregisterPass(void* handle);
    void ForgetFilterCache();
    void* __cdecl AddDrawFilter(const char* shaderName, uint32_t stages, const char* owner);
    void __cdecl RemoveDrawFilter(void* handle);
    void __cdecl SetDrawFilterActive(void* handle, int active);
    uint64_t __cdecl FilteredDraws(void* handle);
    void __cdecl RequestPreviousDepth(int wanted);
    int __cdecl GetPreviousDepth(WXL_VkPreviousDepth* out);
    void __cdecl WatchTextures(const char* const* substrings, uint32_t count);
    int __cdecl TakeWindModel(WXL_VkWindModel* out);
    int __cdecl SetModelWindEx(uint32_t id, const float* data, uint32_t vertexCount, uint32_t vec4sPerVertex);
    int __cdecl SetModelWind(uint32_t id, const float* perVertex, uint32_t vertexCount);
    void __cdecl SetWind(const WXL_VkWindParams* params);
    int __cdecl GetWindStats(WXL_VkWindStats* out);
    uint32_t __cdecl GetWindInstances(WXL_VkWindInstance* out, uint32_t max);
    memory::Where WhereOf(uint32_t cls);
    void* __cdecl CreateBuffer(const WXL_VkBufferDesc* desc, WXL_VkBufferInfo* out);
    void __cdecl RetireBuffer(void* handle, uint64_t lastSerial);
    void __cdecl FlushBuffer(void* handle);
    void* __cdecl CreateImage(const WXL_VkImageDesc* desc, WXL_VkImageInfo* out);
    void __cdecl RetireImage(void* handle, uint64_t lastSerial);
    bool Matches(const Filter& f, const char* name);
}
