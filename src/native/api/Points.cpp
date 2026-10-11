// The insertion points' seams: the consumers' passes fired where the engine's frame reaches them.
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
    // --- the points --------------------------------------------------------------------------------------

    /// Compute and transfer writes recorded so far made visible to the draws and compute after them.
    void WritesToDraws(VkCommandBuffer cmd)
    {
        VkMemoryBarrier m = {};
        m.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        m.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
        m.dstAccessMask = VK_ACCESS_INDIRECT_COMMAND_READ_BIT | VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT |
                          VK_ACCESS_INDEX_READ_BIT | VK_ACCESS_UNIFORM_READ_BIT | VK_ACCESS_SHADER_READ_BIT |
                          VK_ACCESS_SHADER_WRITE_BIT;
        context::Get()->dapi->CmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                                                 VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT | VK_PIPELINE_STAGE_VERTEX_INPUT_BIT |
                                                     VK_PIPELINE_STAGE_VERTEX_SHADER_BIT |
                                                     VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT |
                                                     VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                                 0, 1, &m, 0, nullptr, 0, nullptr);
    }

    /// The point's Prepares: the pre-pass buffer, between the two barriers.
    void Prepares(uint32_t point, WXL_VkRenderContext& ctx)
    {
        bool any = false;
        for (const Pass& p : g.passes)
            if (p.live && p.point == point && p.desc.Prepare) any = true;
        const bool wind = wxlvk::vk::wind::PrePassWanted();
        if (!any && !wind) return;
        VkCommandBuffer pre = present::PrePass();
        if (!pre) return;
        // Tree wind's read-backs: a model's indices or vertex range, once (src/vulkan/Wind.cpp).
        if (wind) wxlvk::vk::wind::PrePass(pre, ctx.serial, ctx.completedSerial);
        if (!any) return;
        context::Get()->dapi->CmdPipelineBarrier(pre,
                                                 VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT | VK_PIPELINE_STAGE_VERTEX_INPUT_BIT |
                                                     VK_PIPELINE_STAGE_VERTEX_SHADER_BIT |
                                                     VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT |
                                                     VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                                                 VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0,
                                                 nullptr, 0, nullptr, 0, nullptr);
        ctx.cmd        = pre;
        ctx.renderPass = VK_NULL_HANDLE;
        for (Pass& p : g.passes)
        {
            if (!p.live || p.point != point || !p.desc.Prepare) continue;
            p.desc.Prepare(p.desc.user, &ctx);
            ++p.prepares;
        }
        WritesToDraws(pre);
    }

    /**
     * Outside the engine's pass, before the Records: the previous-depth copy when it is asked for (so
     * the consumers' own geometry is not in it), then the MidPasses with this frame's depth. Nothing,
     * and no pass break, when neither is wanted.
     */
    void MidPasses(uint32_t point, WXL_VkRenderContext& ctx)
    {
        bool any = false;
        for (const Pass& p : g.passes)
            if (p.live && p.point == point && p.desc.MidPass) any = true;
        const bool copy    = wxlvk::native::render::depthhistory::Wanted();
        const bool pyramid = wxlvk::native::render::depthpyramid::Wanted();
        if (!any && !copy && !pyramid) return;
        VkCommandBuffer cmd = present::Outside();
        if (!cmd) return;
        VkImageView view   = VK_NULL_HANDLE;
        VkFormat    format = VK_FORMAT_UNDEFINED;
        if (!present::SceneDepth(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, view, format))
        {
            ++g.points[point].noDepth;
            return;
        }
        if (copy) wxlvk::native::render::depthhistory::Capture(cmd, view);
        // The min / max pyramid of the same depth (GetDepthPyramid), for the MidPasses below and everything after.
        if (pyramid) wxlvk::native::render::depthpyramid::Build(cmd, view);
        if (!any) return;
        ctx.cmd             = cmd;
        ctx.renderPass      = VK_NULL_HANDLE;
        ctx.depthView       = view;
        ctx.depthFormat     = format;
        ctx.depthViewLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        for (Pass& p : g.passes)
        {
            if (!p.live || p.point != point || !p.desc.MidPass) continue;
            p.desc.MidPass(p.desc.user, &ctx);
            ++p.midPasses;
        }
        ctx.depthView       = VK_NULL_HANDLE;
        ctx.depthViewLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        WritesToDraws(cmd);
        // The consumers' binds are not the draws' tracking's.
        pipelines::InvalidateBindings();
    }

    void Fire(uint32_t point)
    {
        PointState& ps = g.points[point];
        if (ps.passes == 0) return;
        if (!present::DrawPathUp() || !present::EnsureFrame())
        {
            ++ps.noFrame;
            return;
        }
        if (!present::OnBackBuffer())
        {
            ++ps.offBackBuffer;
            return;
        }
        const void* dev = wxl::game::Read<const void*>(gx::kGxDevicePtr);
        if (!dev) return;

        const uint64_t serial = present::Serial();
        if (ps.lastSerial == serial) ++ps.invocation;
        else
        {
            ps.lastSerial = serial;
            ps.invocation = 0;
        }
        ++ps.fired;

        const context::Handles* h = context::Get();
        if (!h || !h->dapi) return;

        WXL_VkSceneConstants scene;
        WXL_VkRenderContext  ctx = {};
        ctx.structSize      = sizeof ctx;
        ctx.point           = point;
        ctx.serial          = serial;
        ctx.completedSerial = present::Completed();
        ctx.frameSlot       = present::Slot();
        ctx.frameSlots      = pipelines::kFrameSlots;
        ctx.invocation      = ps.invocation;
        ctx.extent          = present::Extent();
        ctx.scene           = &scene;
        BuildScene(dev, ctx.extent, scene);
        ctx.viewportValid = wxlvk::native::impl::PlaceDraw(dev, ctx.extent, ctx.viewport, ctx.scissor) ? 1u : 0u;

        Prepares(point, ctx);
        MidPasses(point, ctx);
        // The opaque world is complete: the capture (the shadow replay's list) takes nothing more.
        if (point == WXL_VK_POINT_AFTER_OPAQUE_WORLD) wxlvk::capture::CloseOpaque();

        // --- Record: inside the pass the engine's world draws into ------------------------------------------
        VkCommandBuffer cmd = present::Pass();
        if (!cmd)
        {
            ++ps.noPass;
            return;
        }
        ctx.cmd        = cmd;
        ctx.renderPass = present::RenderPass();
        ctx.extent     = present::Extent();
        ctx.hasDepth   = present::HasDepth() ? 1u : 0u;
        ctx.occlusionQueryActive = present::QueryInPass() ? 1u : 0u;
        ctx.viewportValid = wxlvk::native::impl::PlaceDraw(dev, ctx.extent, ctx.viewport, ctx.scissor) ? 1u : 0u;
        present::PassFormats(ctx.colorFormat, ctx.depthFormat);
        scene.viewport[0] = static_cast<float>(ctx.extent.width);
        scene.viewport[1] = static_cast<float>(ctx.extent.height);
        scene.viewport[2] = ctx.extent.width ? 1.0f / static_cast<float>(ctx.extent.width) : 0.0f;
        scene.viewport[3] = ctx.extent.height ? 1.0f / static_cast<float>(ctx.extent.height) : 0.0f;

        uint32_t called = 0;
        for (Pass& p : g.passes)
        {
            if (!p.live || p.point != point || !p.desc.Record) continue;
            p.desc.Record(p.desc.user, &ctx);
            ++p.records;
            ++called;
        }
        // Whatever the passes bound, the engine's next draw binds everything again.
        pipelines::InvalidateBindings();
        if (called) present::NoteDraw();

        if (!ps.saidFirst)
        {
            ps.saidFirst = true;
            // The eye the view transform implies against the camera's own position: the two agree
            // when the block's view is the matrix the world was drawn with.
            float derived[4];
            Eye(scene.view, derived);
            wxl::ScriptMgr::Log(WXL_LOG_INFO,
                                "native: render: %s reached for the first time (frame %llu): %u pass(es), "
                                "pass %ux%u, depth %s, engine viewport %.1f,%.1f %.0fx%.0f depth %.4f..%.4f; scene eye %.2f, %.2f, %.2f (from the view %.2f, %.2f, "
                                "%.2f), projection [0] %.3f [5] %.3f [10] %.5f [14] %.5f, fog %s %.1f..%.1f",
                                kPointNames[point], static_cast<unsigned long long>(serial), ps.passes,
                                ctx.extent.width, ctx.extent.height, ctx.hasDepth ? "yes" : "no", ctx.viewport.x, ctx.viewport.y,
                                ctx.viewport.width, ctx.viewport.height, ctx.viewport.minDepth, ctx.viewport.maxDepth,
                                scene.cameraPos[0],
                                scene.cameraPos[1], scene.cameraPos[2], derived[0], derived[1], derived[2],
                                scene.projection[0], scene.projection[5], scene.projection[10], scene.projection[14],
                                scene.fogColor[3] > 0.5f ? "on" : "off", scene.fogRange[0], scene.fogRange[1]);
        }
    }

    void __cdecl GroundEffectPass()
    {
        g.groundEffectPass();
        // The opaque world is drawn: the passes from here on are plain ones (the G-buffer's attachments
        // keep what it wrote; the consumers' Records and the engine's transparents draw without them).
        present::NoteOpaqueEnd();
        Fire(WXL_VK_POINT_AFTER_OPAQUE_WORLD);
        // The stages' point at the same moment, after the consumers' passes, so a stage reading the
        // depth sees their geometry too (src/stages/Chain.hpp).
        wxlvk::stages::chain::Run(wxlvk::stages::Point::AfterOpaqueWorld);
    }

    void __cdecl LiquidPass()
    {
        g.liquidPass();
        // The water is drawn (and its depth written): the stages that read the depth find its surface. Stages
        // only: no pass of a consumer is fired here (the depth history and the capture belong to AfterOpaqueWorld).
        wxlvk::stages::chain::Run(wxlvk::stages::Point::AfterLiquids);
    }
}
