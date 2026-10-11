// wxl.vk.render: the table the device publishes and its entries.
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

namespace wxlvk::native::render::detail
{
    WXL_VkRenderApi g_api = {};

    // --- the table ---------------------------------------------------------------------------------------

    void* __cdecl RegisterStage(const WXL_VkStageDesc* desc, const char* name)
    {
        return wxlvk::native::render::stages::Register(desc, name);
    }

    void __cdecl UnregisterStage(void* stage) { wxlvk::native::render::stages::Unregister(stage); }

    int __cdecl GetShadowPass(WXL_VkShadowPass* out)
    {
        if (!out || out->structSize < offsetof(WXL_VkShadowPass, loadPass)) return 0;
        out->renderPass  = wxlvk::capture::shadowpass::Get();
        out->depthFormat = wxlvk::capture::shadowpass::Format();
        if (out->structSize >= offsetof(WXL_VkShadowPass, loadPass) + sizeof out->loadPass)
            out->loadPass = wxlvk::capture::shadowpass::GetLoad();
        return out->renderPass != VK_NULL_HANDLE ? 1 : 0;
    }

    void __cdecl RequestDrawCapture(int wanted) { wxlvk::capture::Request(wanted != 0); }

    void __cdecl RequestGBuffer(int wanted) { wxlvk::native::present::RequestGBuffer(wanted != 0); }

    int __cdecl GetGBuffer(WXL_VkGBuffer* out)
    {
        if (!out || out->structSize < sizeof(uint32_t)) return 0;
        wxlvk::native::present::GBufferDesc g;
        if (!wxlvk::native::present::GBufferInfo(g)) return 0;
        WXL_VkGBuffer d = {};
        d.structSize   = out->structSize;
        d.valid        = g.valid ? 1u : 0u;
        d.serial       = g.serial;
        d.albedoView   = g.views[0];
        d.normalView   = g.views[1];
        d.lightView    = g.views[2];
        d.albedoFormat = g.formats[0];
        d.normalFormat = g.formats[1];
        d.lightFormat  = g.formats[2];
        d.layout       = g.layout;
        d.extent       = g.extent;
        d.interiorCount = wxlvk::native::render::wmogroups::Volumes(g.serial, &d.interiors);   // the interior groups drawn
        d.interiorGroups = d.interiorCount ? wxlvk::native::render::wmogroups::Groups() : nullptr;   // and which group each is
        // Only what the caller's structure has room for.
        const size_t n = out->structSize < sizeof d ? out->structSize : sizeof d;
        std::memcpy(out, &d, n);
        out->structSize = static_cast<uint32_t>(n);
        return 1;
    }

    int __cdecl ReplayWorld(const WXL_VkReplayDesc* desc, WXL_VkReplayStats* stats)
    {
        // The fields appended after `flags` (minIndexCount, cullSphere) are read only when the caller's size covers them:
        // a consumer built against an older header still replays.
        if (!desc || desc->structSize < offsetof(WXL_VkReplayDesc, minIndexCount)) return 0;
        return wxlvk::capture::Replay(*desc, stats) ? 1 : 0;
    }

    int __cdecl ReplayWorldViews(const WXL_VkReplayViewsDesc* desc, WXL_VkReplayStats* stats)
    {
        // Every field up to the stride is read: the struct was born whole (2026-10-11); a later field is read only when the
        // caller's size covers it.
        if (!desc || desc->structSize < offsetof(WXL_VkReplayViewsDesc, viewStride) + sizeof desc->viewStride) return 0;
        return wxlvk::capture::ReplayViews(*desc, stats) ? 1 : 0;
    }

    int __cdecl GetDeviceInfo(WXL_VkDeviceInfo* out)
    {
        if (!out) return 0;
        const context::Handles* h = context::Get();
        if (!h || !h->device || !h->iapi) return 0;
        std::memset(out, 0, sizeof *out);
        out->structSize          = sizeof *out;
        out->instance            = h->instance;
        out->physicalDevice      = h->physical;
        out->device              = h->device;
        out->queueFamily         = h->graphicsFamily;
        out->frameSlots          = pipelines::kFrameSlots;
        out->getInstanceProcAddr = loader::Api().GetInstanceProcAddr;
        out->getDeviceProcAddr   = h->iapi->GetDeviceProcAddr;
        out->pipelineCache       = pipelines::PipelineCache();
        return 1;
    }

    void* __cdecl RegisterPass(uint32_t point, const WXL_VkRenderPassDesc* desc, const char* name)
    {
        // A consumer built against the struct before MidPass was appended is taken as it is.
        const size_t oldest = offsetof(WXL_VkRenderPassDesc, MidPass);
        if (point >= WXL_VK_POINT_COUNT || point == WXL_VK_POINT_BEFORE_UI || point == WXL_VK_POINT_AFTER_LIQUIDS || !desc || desc->structSize < oldest)
        {
            wxl::ScriptMgr::Log(WXL_LOG_WARN, "native: render: pass '%s' refused: unknown point %u or an "
                                "older WXL_VkRenderPassDesc", name ? name : "?", point);
            return nullptr;
        }
        for (Pass& p : g.passes)
        {
            if (p.live) continue;
            p          = Pass();
            p.live     = true;
            p.point    = point;
            const size_t n = desc->structSize < sizeof p.desc ? desc->structSize : sizeof p.desc;
            std::memcpy(&p.desc, desc, n);   // what the consumer's struct lacks stays null
            p.name     = name ? name : "";
            ++g.points[point].passes;
            wxl::ScriptMgr::Log(WXL_LOG_INFO, "native: render: pass '%s' registered at %s (prepare %s, mid-pass %s, "
                                "record %s)", p.name, kPointNames[point], p.desc.Prepare ? "yes" : "no",
                                p.desc.MidPass ? "yes" : "no", p.desc.Record ? "yes" : "no");
            return &p;
        }
        wxl::ScriptMgr::Log(WXL_LOG_WARN, "native: render: pass '%s' refused: %u passes already registered",
                            name ? name : "?", kMaxPasses);
        return nullptr;
    }

    void __cdecl UnregisterPass(void* handle)
    {
        Pass* p = static_cast<Pass*>(handle);
        if (!p || p < g.passes || p >= g.passes + kMaxPasses || !p->live) return;
        p->live = false;
        --g.points[p->point].passes;
        wxl::ScriptMgr::Log(WXL_LOG_INFO, "native: render: pass '%s' withdrawn from %s", p->name,
                            kPointNames[p->point]);
    }

    void ForgetFilterCache()
    {
        g.lastShader[0] = g.lastShader[1] = 0;
        g.lastFilter[0] = g.lastFilter[1] = -1;
    }

    void* __cdecl AddDrawFilter(const char* shaderName, uint32_t stages, const char* owner)
    {
        const size_t length = shaderName ? std::strlen(shaderName) : 0;
        if (!length || length >= kFilterChars || (stages & 3u) == 0) return nullptr;
        for (Filter& f : g.filters)
        {
            if (f.live) continue;
            f        = Filter();
            f.live   = true;
            std::memcpy(f.name, shaderName, length + 1);
            f.length = length;
            f.stages = stages & 3u;
            f.owner  = owner ? owner : "";
            ++wxlvk::native::render::detail::filterCount;
            ForgetFilterCache();
            wxl::ScriptMgr::Log(WXL_LOG_INFO, "native: render: draw filter '%s' (%s%s%s) added by %s: the engine's "
                                "draws with that shader are dropped", f.name, (f.stages & 1u) ? "vertex" : "",
                                f.stages == 3u ? " and " : "", (f.stages & 2u) ? "pixel" : "", f.owner);
            return &f;
        }
        return nullptr;
    }

    void __cdecl RemoveDrawFilter(void* handle)
    {
        Filter* f = static_cast<Filter*>(handle);
        if (!f || f < g.filters || f >= g.filters + kMaxFilters || !f->live) return;
        f->live = false;
        if (f->active) --wxlvk::native::render::detail::filterCount;
        ForgetFilterCache();
        wxl::ScriptMgr::Log(WXL_LOG_INFO, "native: render: draw filter '%s' lifted (%llu draw(s) suppressed)",
                            f->name, static_cast<unsigned long long>(f->hits));
    }

    void __cdecl SetDrawFilterActive(void* handle, int active)
    {
        Filter* f = static_cast<Filter*>(handle);
        if (!f || f < g.filters || f >= g.filters + kMaxFilters || !f->live) return;
        const bool on = active != 0;
        if (on == f->active) return;
        f->active = on;
        if (on) ++wxlvk::native::render::detail::filterCount;
        else --wxlvk::native::render::detail::filterCount;
        ForgetFilterCache();
    }

    uint64_t __cdecl FilteredDraws(void* handle)
    {
        const Filter* f = static_cast<const Filter*>(handle);
        if (!f || f < g.filters || f >= g.filters + kMaxFilters) return 0;
        return f->hits;
    }

    void __cdecl RequestPreviousDepth(int wanted)
    {
        wxlvk::native::render::depthhistory::Request(wanted != 0);
    }

    int __cdecl GetPreviousDepth(WXL_VkPreviousDepth* out)
    {
        if (!out || out->structSize < sizeof(uint32_t)) return 0;
        return wxlvk::native::render::depthhistory::Previous(*out) ? 1 : 0;
    }

    // --- the optimisation pass's services (2026-10-11) ---------------------------------------------------------------

    int __cdecl GetDeviceCaps(WXL_VkDeviceCaps* out)
    {
        if (!out || out->structSize < sizeof(uint32_t) || !context::Get()) return 0;
        const wxlvk::vk::DeviceCaps& c = context::Caps();
        const context::Handles*      h = context::Get();
        WXL_VkDeviceCaps d = {};
        d.structSize        = out->structSize;
        d.apiVersion        = c.deviceVersion;
        d.instanceVersion   = c.instanceVersion;
        d.adapterVersion    = c.adapterVersion;
        d.enabled           = c.enabled;
        d.offered           = c.offered;
        d.subgroupSize      = c.deviceVersion >= VK_API_VERSION_1_1 ? c.subgroupSize : 0u;
        d.subgroupStages    = c.deviceVersion >= VK_API_VERSION_1_1 ? c.subgroupStages : 0u;
        d.subgroupOps       = c.deviceVersion >= VK_API_VERSION_1_1 ? c.subgroupOps : 0u;
        d.maxMultiviewViews = (c.enabled & wxlvk::vk::kCapMultiview) ? c.maxMultiviewViews : 0u;
        d.graphicsFamily    = h->graphicsFamily;
        d.graphicsQueues    = c.graphicsQueues;
        d.computeFamily     = c.computeFamily;
        d.computeQueues     = c.computeQueues;
        d.transferFamily    = c.transferFamily;
        d.transferQueues    = c.transferQueues;
        const size_t n = out->structSize < sizeof d ? out->structSize : sizeof d;
        std::memcpy(out, &d, n);
        out->structSize = static_cast<uint32_t>(n);
        return 1;
    }

    int __cdecl GetMemoryBudget(WXL_VkMemoryBudget* out)
    {
        if (!out || out->structSize < sizeof(uint32_t) || !context::Get()) return 0;
        return wxlvk::native::render::budget::Get(*out) ? 1 : 0;
    }

    int __cdecl MayKeepResident(uint64_t extraBytes, uint32_t memoryClass)
    {
        if (!context::Get()) return 0;
        return wxlvk::native::render::budget::MayKeep(extraBytes, memoryClass) ? 1 : 0;
    }

    void __cdecl RequestOpaqueDepth(int wanted) { wxlvk::native::render::depthhistory::RequestOpaque(wanted != 0); }

    int __cdecl GetOpaqueDepth(WXL_VkPreviousDepth* out)
    {
        if (!out || out->structSize < sizeof(uint32_t)) return 0;
        return wxlvk::native::render::depthhistory::Opaque(*out) ? 1 : 0;
    }

    void __cdecl RequestDepthPyramid(int wanted) { wxlvk::native::render::depthpyramid::Request(wanted != 0); }

    int __cdecl GetDepthPyramid(WXL_VkDepthPyramid* out)
    {
        if (!out) return 0;
        return wxlvk::native::render::depthpyramid::Get(*out) ? 1 : 0;
    }

    int __cdecl GetFrameStats(WXL_VkFrameStats* out)
    {
        if (!out) return 0;
        return wxlvk::native::render::framestats::Get(*out) ? 1 : 0;
    }

    // Tree wind (src/vulkan/Wind.hpp).
    void __cdecl WatchTextures(const char* const* substrings, uint32_t count) { wxlvk::vk::wind::Watch(substrings, count); }
    int __cdecl TakeWindModel(WXL_VkWindModel* out) { return wxlvk::vk::wind::TakeModel(out); }
    int __cdecl SetModelWindEx(uint32_t id, const float* data, uint32_t vertexCount, uint32_t vec4sPerVertex)
    {
        return wxlvk::vk::wind::SetModel(id, data, vertexCount, vec4sPerVertex);
    }

    int __cdecl SetModelWind(uint32_t id, const float* perVertex, uint32_t vertexCount)
    {
        return wxlvk::vk::wind::SetModel(id, perVertex, vertexCount);
    }
    void __cdecl SetWind(const WXL_VkWindParams* params) { wxlvk::vk::wind::SetParams(params); }
    int __cdecl GetWindStats(WXL_VkWindStats* out) { return wxlvk::vk::wind::Stats(out); }
    uint32_t __cdecl GetWindInstances(WXL_VkWindInstance* out, uint32_t max) { return wxlvk::vk::wind::Instances(out, max); }

    memory::Where WhereOf(uint32_t cls)
    {
        switch (cls)
        {
            case WXL_VK_MEMORY_HOST:   return memory::Where::HostVisible;
            case WXL_VK_MEMORY_STREAM: return memory::Where::Stream;
            default:                   return memory::Where::DeviceLocal;
        }
    }

    void* __cdecl CreateBuffer(const WXL_VkBufferDesc* desc, WXL_VkBufferInfo* out)
    {
        if (!desc || !out || desc->size == 0 || !context::Get()) return nullptr;
        BufferRec* rec = new (std::nothrow) BufferRec();
        if (!rec) return nullptr;
        if (!memory::CreateBuffer(desc->size, desc->usage, WhereOf(desc->memory), rec->buffer)
            || (desc->memory != WXL_VK_MEMORY_DEVICE_LOCAL && !rec->buffer.block.mapped))
        {
            memory::DestroyBuffer(rec->buffer);
            delete rec;
            ++g.createFailed;
            return nullptr;
        }
        out->buffer = rec->buffer.handle;
        out->size   = rec->buffer.size;
        out->mapped = rec->buffer.block.mapped;
        ++g.buffersMade;
        g.bufferBytes += desc->size;
        return rec;
    }

    void __cdecl RetireBuffer(void* handle, uint64_t lastSerial)
    {
        BufferRec* rec = static_cast<BufferRec*>(handle);
        if (!rec) return;
        present::RetireBuffer(rec->buffer, lastSerial);
        delete rec;
        ++g.buffersRetired;
    }

    void __cdecl FlushBuffer(void* handle)
    {
        const BufferRec* rec = static_cast<const BufferRec*>(handle);
        if (rec) memory::FlushHost(rec->buffer.block);
    }

    void* __cdecl CreateImage(const WXL_VkImageDesc* desc, WXL_VkImageInfo* out)
    {
        const context::Handles* h = context::Get();
        if (!desc || !out || !h || !h->device || desc->width == 0 || desc->height == 0) return nullptr;
        const loader::Device& d = *h->dapi;
        ImageRec* rec = new (std::nothrow) ImageRec();
        if (!rec) return nullptr;
        const uint32_t levels = desc->mipLevels ? desc->mipLevels : 1u;

        VkImageCreateInfo ii = {};
        ii.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        ii.imageType     = VK_IMAGE_TYPE_2D;
        ii.format        = desc->format;
        ii.extent        = VkExtent3D{ desc->width, desc->height, 1 };
        ii.mipLevels     = levels;
        ii.arrayLayers   = 1;
        ii.samples       = VK_SAMPLE_COUNT_1_BIT;
        ii.tiling        = VK_IMAGE_TILING_OPTIMAL;
        ii.usage         = desc->usage;
        ii.sharingMode   = VK_SHARING_MODE_EXCLUSIVE;
        ii.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        bool ok = d.CreateImage(h->device, &ii, nullptr, &rec->image) == VK_SUCCESS;
        if (ok)
        {
            VkMemoryRequirements req = {};
            d.GetImageMemoryRequirements(h->device, rec->image, &req);
            ok = memory::Allocate(req, memory::Where::DeviceLocal, rec->block)
                 && d.BindImageMemory(h->device, rec->image, rec->block.memory, rec->block.offset) == VK_SUCCESS;
        }
        if (ok)
        {
            VkImageViewCreateInfo vi = {};
            vi.sType    = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
            vi.image    = rec->image;
            vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
            vi.format   = desc->format;
            vi.subresourceRange.aspectMask = desc->aspect ? desc->aspect : VK_IMAGE_ASPECT_COLOR_BIT;
            vi.subresourceRange.levelCount = levels;
            vi.subresourceRange.layerCount = 1;
            ok = d.CreateImageView(h->device, &vi, nullptr, &rec->view) == VK_SUCCESS;
        }
        if (!ok)
        {
            if (rec->view != VK_NULL_HANDLE) d.DestroyImageView(h->device, rec->view, nullptr);
            if (rec->image != VK_NULL_HANDLE) d.DestroyImage(h->device, rec->image, nullptr);
            memory::Free(rec->block);
            delete rec;
            ++g.createFailed;
            return nullptr;
        }
        out->image = rec->image;
        out->view  = rec->view;
        ++g.imagesMade;
        return rec;
    }

    void __cdecl RetireImage(void* handle, uint64_t lastSerial)
    {
        ImageRec* rec = static_cast<ImageRec*>(handle);
        if (!rec) return;
        present::RetireRaw(rec->image, rec->view, rec->block, lastSerial);
        delete rec;
        ++g.imagesRetired;
    }

    /// Whether `name` is the filter's family: the name itself, or the name and a ':' suffix.
    bool Matches(const Filter& f, const char* name)
    {
        return std::strncmp(name, f.name, f.length) == 0 && (name[f.length] == '\0' || name[f.length] == ':');
    }
}
