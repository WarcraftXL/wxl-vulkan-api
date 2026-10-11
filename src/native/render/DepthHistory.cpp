// Last frame's opaque scene depth, kept for the rendering API's consumers (the foliage's occlusion
// culling builds its Hi-Z pyramid from it in Prepare, before the world is drawn).
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

#include "native/render/DepthHistory.hpp"

#include "native/Present.hpp"
#include "native/Render.hpp"
#include "common/CameraCut.hpp"
#include "stages/Gpu.hpp"
#include "vulkan/Pipelines.hpp"

#include "wxl/Script.hpp"

#include <cstddef>
#include <cstring>

namespace
{
    namespace gpu      = wxlvk::stages::gpu;
    namespace cut      = wxlvk::cut;
    namespace present  = wxlvk::native::present;

    /// The frame a copy is made from.
    struct Io
    {
        VkCommandBuffer      cmd    = nullptr;
        uint32_t             slot   = 0;
        uint64_t             serial = 0;
        VkExtent2D           extent = { 0, 0 };
        VkImageView          depthView = VK_NULL_HANDLE;
        WXL_VkSceneConstants scene = {};
        VkViewport           viewport = {};
        bool                 sceneValid = false;
        float                jitter[2] = { 0.0f, 0.0f };
    };

    constexpr uint32_t kSlots = wxlvk::vk::pipelines::kFrameSlots;

    /// One kept copy and the frame it describes.
    struct Copy
    {
        gpu::OwnedImage      image;
        uint64_t             serial = 0;
        bool                 made   = false;
        WXL_VkSceneConstants scene  = {};
        VkViewport           viewport = {};
        float                jitter[2] = { 0.0f, 0.0f };
        cut::Camera          camera;
    };

    class DepthHistory
    {
    public:
        /// The previous depth's consumers and this frame's (GetOpaqueDepth) share the one copy.
        bool Wanted() const { return requests_ > 0 || opaqueRequests_ > 0; }

        /// The copy of this frame, from the depth `io.depthView` holds now.
        void Capture(const Io& io)
        {
            if (!Ensure(io)) return;
            Record(io);
        }

        bool Ensure(const Io& io)
        {
            if (!io.sceneValid) return false;
            if (!tried_)
            {
                tried_ = true;
                ok_    = MakeObjects();
                if (!ok_) wxl::ScriptMgr::Log(WXL_LOG_ERROR, "render: depth history: its pipeline could not be made");
            }
            if (!ok_) return false;
            Copy& target = copies_[io.serial & 1];
            if (!EnsureImage(target, io)) return false;
            WriteSet(io, target);
            return true;
        }

        void Record(const Io& io)
        {
            Copy& target = copies_[io.serial & 1];
            const wxlvk::vk::loader::Device& d = *gpu::Api();
            // The reads of this image two frames back (any stage a consumer may use) come first.
            gpu::Barrier(io.cmd,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_VERTEX_SHADER_BIT |
                             VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                         VK_ACCESS_SHADER_READ_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT);
            present::GpuLaneBegin(io.cmd, present::kGpuDepthCopy);
            const int32_t push[4] = { static_cast<int32_t>(io.extent.width), static_cast<int32_t>(io.extent.height), 0, 0 };
            d.CmdBindDescriptorSets(io.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, layout_, 0, 1, &sets_[io.slot], 0, nullptr);
            d.CmdPushConstants(io.cmd, layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof push, push);
            d.CmdBindPipeline(io.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe_);
            d.CmdDispatch(io.cmd, gpu::Groups(io.extent.width, 8), gpu::Groups(io.extent.height, 8), 1);
            present::GpuLaneEnd(io.cmd, present::kGpuDepthCopy);
            // Visible to everything after it: this frame's later work and the next frame's pre-pass.
            gpu::Barrier(io.cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_VERTEX_SHADER_BIT |
                             VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                         VK_ACCESS_SHADER_READ_BIT);
            Describe(target, io);
            lastUsed_ = io.serial;
            ++copiesMade_;
        }

        void Release(uint64_t lastUsed)
        {
            const uint64_t at = lastUsed > lastUsed_ ? lastUsed : lastUsed_;
            for (Copy& c : copies_)
            {
                gpu::RetireImage(c.image, at);
                c.made = false;
            }
        }

        void Request(bool wanted)
        {
            if (wanted) ++requests_;
            else if (requests_ > 0) --requests_;
        }

        void RequestOpaque(bool wanted)
        {
            if (wanted) ++opaqueRequests_;
            else if (opaqueRequests_ > 0) --opaqueRequests_;
        }

        bool Previous(WXL_VkPreviousDepth& out) const;

        /// The copy made in the frame being recorded; null when none was (yet).
        const Copy* Current() const
        {
            const Copy& c = copies_[present::Serial() & 1];
            return c.made && c.serial == present::Serial() ? &c : nullptr;
        }

        uint64_t Copies() const { return copiesMade_; }

    private:
        bool MakeObjects()
        {
            const VkDescriptorType types[2] = { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE };
            if (gpu::PointSampler() == VK_NULL_HANDLE) return false;
            set_ = gpu::ComputeSetLayout(types, 2);
            if (set_ == VK_NULL_HANDLE) return false;
            layout_ = gpu::ComputeLayout(set_, 4 * sizeof(int32_t));
            if (layout_ == VK_NULL_HANDLE) return false;
            pipe_ = gpu::ComputePipeline("depthcopy_comp", layout_);
            if (pipe_ == VK_NULL_HANDLE) return false;
            const VkDescriptorPoolSize sizes[2] = {
                { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, kSlots },
                { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, kSlots },
            };
            return gpu::AllocateSets(set_, sizes, 2, kSlots, pool_, sets_);
        }

        /// The copy's image at the frame's size; a new one is not a valid history until it is written.
        bool EnsureImage(Copy& c, const Io& io)
        {
            if (c.image.img.image != VK_NULL_HANDLE && c.image.extent.width == io.extent.width
                && c.image.extent.height == io.extent.height)
                return true;
            gpu::RetireImage(c.image, lastUsed_);
            c.made = false;
            if (!gpu::MakeImage(VK_FORMAT_R32_SFLOAT, io.extent, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                                c.image))
                return false;
            gpu::Layout(io.cmd, c.image.img.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
            return true;
        }

        /// The slot's set: its last frame has completed, so it can be rewritten every run.
        void WriteSet(const Io& io, const Copy& target)
        {
            const VkDescriptorImageInfo depth = { gpu::PointSampler(), io.depthView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
            const VkDescriptorImageInfo out   = { VK_NULL_HANDLE, target.image.img.view, VK_IMAGE_LAYOUT_GENERAL };
            VkWriteDescriptorSet w[2] = {};
            for (uint32_t i = 0; i < 2; ++i)
            {
                w[i].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                w[i].dstSet          = sets_[io.slot];
                w[i].dstBinding      = i;
                w[i].descriptorCount = 1;
            }
            w[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            w[0].pImageInfo     = &depth;
            w[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            w[1].pImageInfo     = &out;
            gpu::Api()->UpdateDescriptorSets(gpu::Device(), 2, w, 0, nullptr);
        }

        void Describe(Copy& c, const Io& io)
        {
            c.serial    = io.serial;
            c.made      = true;
            c.scene     = io.scene;
            c.viewport  = io.viewport;
            c.jitter[0] = io.jitter[0];
            c.jitter[1] = io.jitter[1];
            c.camera    = cut::Of(io.serial, io.scene.view, io.scene.cameraPos);
        }

        /// The latest copy made before the frame being recorded; null when none.
        const Copy* Latest() const
        {
            const uint64_t now  = present::Serial();
            const Copy*    best = nullptr;
            for (const Copy& c : copies_)
                if (c.made && c.serial < now && (!best || c.serial > best->serial)) best = &c;
            return best;
        }

        bool                  tried_  = false;
        bool                  ok_     = false;
        VkDescriptorSetLayout set_    = VK_NULL_HANDLE;
        VkPipelineLayout      layout_ = VK_NULL_HANDLE;
        VkPipeline            pipe_   = VK_NULL_HANDLE;
        VkDescriptorPool      pool_   = VK_NULL_HANDLE;
        VkDescriptorSet       sets_[kSlots] = {};
        Copy                  copies_[2];
        uint32_t              requests_    = 0;
        uint32_t              opaqueRequests_ = 0;
        uint64_t              copiesMade_ = 0;
        uint64_t              lastUsed_    = 0;
    };

    /// Whether the copy describes the frame right before this one, at this size, with no camera cut.
    bool Usable(const Copy& c)
    {
        VkFormat   format = VK_FORMAT_UNDEFINED;
        VkExtent2D extent = {};
        if (!present::BackBufferInfo(format, extent) || extent.width != c.image.extent.width
            || extent.height != c.image.extent.height)
            return false;
        WXL_VkSceneConstants scene = {};
        VkViewport           viewport = {};
        float                jitter[2];
        if (!wxlvk::native::render::SceneNow(extent, scene, viewport, jitter)) return false;
        return !cut::Between(c.camera, cut::Of(present::Serial(), scene.view, scene.cameraPos));
    }

    /// A copy described for a consumer, `valid` given.
    void DescribeFor(const Copy* c, bool valid, WXL_VkPreviousDepth& out);

    bool DepthHistory::Previous(WXL_VkPreviousDepth& out) const
    {
        const Copy* c = Latest();
        if (!c) return false;
        DescribeFor(c, Usable(*c), out);
        return true;
    }

    void DescribeFor(const Copy* c, bool valid, WXL_VkPreviousDepth& out)
    {
        WXL_VkPreviousDepth d = {};
        d.structSize = out.structSize;
        d.valid      = valid ? 1u : 0u;
        d.serial     = c->serial;
        d.image      = c->image.img.image;
        d.imageView  = c->image.img.view;
        d.format     = VK_FORMAT_R32_SFLOAT;
        d.layout     = VK_IMAGE_LAYOUT_GENERAL;
        d.extent     = c->image.extent;
        std::memcpy(d.view, c->scene.view, sizeof d.view);
        std::memcpy(d.projection, c->scene.projection, sizeof d.projection);
        std::memcpy(d.viewProj, c->scene.viewProj, sizeof d.viewProj);
        d.depthMin  = c->viewport.minDepth;
        d.depthMax  = c->viewport.maxDepth;
        d.viewport  = c->viewport;
        d.jitter[0] = c->jitter[0];
        d.jitter[1] = c->jitter[1];
        // Only what the caller's structure has room for.
        const size_t n = out.structSize < sizeof d ? out.structSize : sizeof d;
        std::memcpy(&out, &d, n);
        out.structSize = static_cast<uint32_t>(n);
    }

    DepthHistory g_history;
}

namespace wxlvk::native::render::depthhistory
{
    bool Wanted() { return g_history.Wanted(); }

    void Capture(VkCommandBuffer cmd, VkImageView depthView)
    {
        Io io;
        VkFormat format = VK_FORMAT_UNDEFINED;
        if (!cmd || depthView == VK_NULL_HANDLE || !present::BackBufferInfo(format, io.extent)) return;
        io.cmd        = cmd;
        io.slot       = present::Slot();
        io.serial     = present::Serial();
        io.depthView  = depthView;
        io.sceneValid = wxlvk::native::render::SceneNow(io.extent, io.scene, io.viewport, io.jitter);
        g_history.Capture(io);
    }

    void Request(bool wanted)
    {
        g_history.Request(wanted);
        if (!g_history.Wanted()) g_history.Release(present::Serial());
    }
    bool Previous(WXL_VkPreviousDepth& out) { return g_history.Previous(out); }

    void RequestOpaque(bool wanted)
    {
        g_history.RequestOpaque(wanted);
        if (!g_history.Wanted()) g_history.Release(present::Serial());
    }

    bool Opaque(WXL_VkPreviousDepth& out)
    {
        const Copy* c = g_history.Current();
        if (!c) return false;
        DescribeFor(c, true, out);
        return true;
    }
    uint64_t Copies() { return g_history.Copies(); }
}
