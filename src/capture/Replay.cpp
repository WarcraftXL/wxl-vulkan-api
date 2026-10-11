// The shadow replay: the frame's captured opaque world drawn again, depth only, from a light.
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

#include "capture/Replay.hpp"

#include "capture/Capture.hpp"
#include "capture/ShadowPass.hpp"
#include "capture/ShadowPipelines.hpp"
#include "native/Present.hpp"
#include "native/Render.hpp"
#include "vulkan/Context.hpp"
#include "vulkan/Pipelines.hpp"

#include "wxl/Script.hpp"

#include <climits>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <vector>

namespace
{
    namespace context  = wxlvk::vk::context;
    namespace pipes    = wxlvk::capture::shadowpipes;
    using wxlvk::capture::Draw;

    struct Totals
    {
        uint64_t replays  = 0;
        uint64_t drawn    = 0;
        uint64_t skipped  = 0;
        uint64_t alpha    = 0;
        uint64_t refused  = 0;   ///< calls that could not record (no pass, no scene)
        uint64_t culled   = 0;   ///< draws left out by their bounds (WXL_VK_REPLAY_CULL_FRUSTUM, cullSphere)
        uint64_t multi    = 0;   ///< ReplayWorldViews calls (their views counted in `replays`)
        bool     saidFirst = false;
    };
    Totals g;

    /// What the replay has bound, so a run of draws of one material binds once.
    struct Bound
    {
        VkPipeline       pipeline = VK_NULL_HANDLE;
        VkPipelineLayout layout   = VK_NULL_HANDLE;
        VkDescriptorSet  set0     = VK_NULL_HANDLE;
        VkDescriptorSet  ring     = VK_NULL_HANDLE;
        uint32_t         ringOffset = 0xFFFFFFFFu;
        VkDescriptorSet  bones    = VK_NULL_HANDLE;
        uint32_t         bonesOffset = 0xFFFFFFFFu;
        VkBuffer         vbs[4]   = {};
        uint32_t         vbCount  = 0;
        VkBuffer         ib       = VK_NULL_HANDLE;
    };

    /// The inverse of a 4x4 (row-major) in double, by Gauss-Jordan with partial pivoting.
    bool InvertDouble(const float m[16], double out[16])
    {
        double a[4][8];
        for (int r = 0; r < 4; ++r)
            for (int c = 0; c < 4; ++c)
            {
                a[r][c]     = m[r * 4 + c];
                a[r][c + 4] = r == c ? 1.0 : 0.0;
            }
        for (int col = 0; col < 4; ++col)
        {
            int pivot = col;
            for (int r = col + 1; r < 4; ++r)
                if (std::fabs(a[r][col]) > std::fabs(a[pivot][col])) pivot = r;
            if (std::fabs(a[pivot][col]) < 1e-30) return false;
            for (int c = 0; c < 8; ++c)
            {
                const double t = a[col][c];
                a[col][c]   = a[pivot][c];
                a[pivot][c] = t;
            }
            const double inv = 1.0 / a[col][col];
            for (int c = 0; c < 8; ++c) a[col][c] *= inv;
            for (int r = 0; r < 4; ++r)
            {
                if (r == col) continue;
                const double f = a[r][col];
                for (int c = 0; c < 8; ++c) a[r][c] -= f * a[col][c];
            }
        }
        for (int r = 0; r < 4; ++r)
            for (int c = 0; c < 4; ++c) out[r * 4 + c] = a[r][c + 4];
        return true;
    }

    /**
     * inverse(projection) x inverse(view) x the light's, in double, rounded once: false without the
     * frame's scene.
     *
     * Not inverse(viewProj): the frame's view-projection is a float product whose translation row
     * carries the world's coordinates (thousands of yards), and its float inverse is off by several
     * shadow texels once multiplied out. The draws form clip space from a view-space position and the
     * projection; undoing exactly those two, in double, lands within a hundredth of a texel
     * (a cascade probe measures 4.5 texels the first way, 0.003 this way).
     */
    bool SceneInverse(double t[16], double iv[16])
    {
        VkFormat   format = VK_FORMAT_UNDEFINED;
        VkExtent2D extent = {};
        if (!wxlvk::native::present::BackBufferInfo(format, extent)) return false;
        WXL_VkSceneConstants scene = {};
        VkViewport           viewport = {};
        float                jitter[2];
        if (!wxlvk::native::render::SceneNow(extent, scene, viewport, jitter)) return false;
        double ip[16];
        if (!InvertDouble(scene.projection, ip) || !InvertDouble(scene.view, iv)) return false;
        for (int r = 0; r < 4; ++r)
            for (int c = 0; c < 4; ++c)
            {
                double s = 0.0;
                for (int k = 0; k < 4; ++k) s += ip[r * 4 + k] * iv[k * 4 + c];
                t[r * 4 + c] = s;
            }
        return true;
    }

    /// inverse(projection) x inverse(view) (SceneInverse's `t`) x the light's, rounded once.
    void Compose(const double t[16], const float lightViewProj[16], float out[16])
    {
        for (int r = 0; r < 4; ++r)
            for (int c = 0; c < 4; ++c)
            {
                double s = 0.0;
                for (int k = 0; k < 4; ++k) s += t[r * 4 + k] * static_cast<double>(lightViewProj[k * 4 + c]);
                out[r * 4 + c] = static_cast<float>(s);
            }
    }

    bool ClipToLight(const float lightViewProj[16], float out[16], double iv[16])
    {
        double t[16];
        if (!SceneInverse(t, iv)) return false;
        Compose(t, lightViewProj, out);
        return true;
    }

    void Begin(const WXL_VkReplayDesc& desc, VkRenderPass pass, const wxlvk::vk::loader::Device& d)
    {
        VkClearValue clear = {};
        clear.depthStencil.depth = 1.0f;
        VkRenderPassBeginInfo begin = {};
        begin.sType           = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        begin.renderPass      = pass;
        begin.framebuffer     = desc.framebuffer;
        begin.renderArea      = desc.area;
        begin.clearValueCount = 1;
        begin.pClearValues    = &clear;
        d.CmdBeginRenderPass(desc.cmd, &begin, VK_SUBPASS_CONTENTS_INLINE);
        VkViewport vp = {};
        vp.x        = static_cast<float>(desc.area.offset.x);
        vp.y        = static_cast<float>(desc.area.offset.y);
        vp.width    = static_cast<float>(desc.area.extent.width);
        vp.height   = static_cast<float>(desc.area.extent.height);
        vp.maxDepth = 1.0f;
        d.CmdSetViewport(desc.cmd, 0, 1, &vp);
        d.CmdSetScissor(desc.cmd, 0, 1, &desc.area);
    }

    /// The draw's sets, its push matrix when the layout changed, its buffers. `alpha`: its textures bound (the
    /// alpha-tested casters a replay keeps as such); `windPush`: the wind values pushed (the draw's, or the still ones).
    void BindDraw(VkCommandBuffer cmd, const Draw& dr, bool alpha, const int32_t* windPush, const pipes::Pipe& p,
                  const float m[16], const WXL_VkReplayDesc& desc, Bound& b, const wxlvk::vk::loader::Device& d)
    {
        if (b.pipeline != p.pipeline)
        {
            d.CmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, p.pipeline);
            // Binding a pipeline with a dynamic bias keeps the value; set once per pipeline bind anyway.
            d.CmdSetDepthBias(cmd, desc.depthBias, 0.0f, desc.slopeBias);
            b.pipeline = p.pipeline;
        }
        if (b.layout != p.layout)
        {
            b = Bound{ b.pipeline, p.layout };
            // A tree-wind pipeline's matrix follows the wind's values (shadowgen.py).
            d.CmdPushConstants(cmd, p.layout, VK_SHADER_STAGE_VERTEX_BIT, p.wind ? sizeof dr.windPush : 0u,
                               16 * sizeof(float), m);
        }
        if (p.wind)
        {
            d.CmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, p.layout, 3, 1, &dr.windSet, 0, nullptr);
            d.CmdPushConstants(cmd, p.layout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof dr.windPush, windPush);
        }
        if (alpha && b.set0 != dr.set0)
        {
            const VkDescriptorSet sets[2] = { dr.set0, dr.ring };
            d.CmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, p.layout, 0, 2, sets, 1, &dr.ringOffset);
            b.set0       = dr.set0;
            b.ring       = dr.ring;
            b.ringOffset = dr.ringOffset;
        }
        else if (b.ring != dr.ring || b.ringOffset != dr.ringOffset)
        {
            d.CmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, p.layout, 1, 1, &dr.ring, 1, &dr.ringOffset);
            b.ring       = dr.ring;
            b.ringOffset = dr.ringOffset;
        }
        if (p.bones && (b.bones != dr.bones || b.bonesOffset != dr.bonesOffset))
        {
            d.CmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, p.layout, 2, 1, &dr.bones, 1, &dr.bonesOffset);
            b.bones       = dr.bones;
            b.bonesOffset = dr.bonesOffset;
        }
        bool same = b.vbCount == dr.vbCount;
        for (uint32_t i = 0; same && i < dr.vbCount; ++i) same = b.vbs[i] == dr.vbs[i];
        if (!same)
        {
            const VkDeviceSize zeros[4] = {};
            d.CmdBindVertexBuffers(cmd, 0, dr.vbCount, dr.vbs, zeros);
            b.vbCount = dr.vbCount;
            for (uint32_t i = 0; i < dr.vbCount; ++i) b.vbs[i] = dr.vbs[i];
        }
        if (dr.indexed && b.ib != dr.ib)
        {
            d.CmdBindIndexBuffer(cmd, dr.ib, 0, VK_INDEX_TYPE_UINT16);
            b.ib = dr.ib;
        }
    }

    /**
     * The culling of a replay: lightViewProj's side and far planes in world space (row vectors: clip = p M, the planes
     * w + x, w - x, w + y, w - y, w - z of its columns; never the near plane z >= 0, so a caster between the light and
     * it is kept as the depth clip treats it), and the caller's sphere. A draw's bound (capture::Draw::bound, the
     * frame's view space) is taken to world through the view's inverse (double: the world's coordinates are large).
     */
    struct Cull
    {
        bool   frustum = false, sphere = false;
        double planes[5][4] = {};
        double centre[3] = {}, radius = 0.0;
        double iv[16] = {};

        void Make(const WXL_VkReplayDesc& desc, const double inverseView[16])
        {
            const bool hasSphere = desc.structSize >= offsetof(WXL_VkReplayDesc, cullSphere) + sizeof desc.cullSphere;
            MakeView(desc.flags, desc.lightViewProj, hasSphere ? desc.cullSphere : nullptr, inverseView);
            if (desc.structSize >= offsetof(WXL_VkReplayDesc, excludeBoxCount) + sizeof desc.excludeBoxCount && desc.excludeBoxes
                && desc.excludeBoxCount)
            {
                boxes    = desc.excludeBoxes;
                boxCount = desc.excludeBoxCount;
                keep     = (desc.flags & WXL_VK_REPLAY_BOXES_KEEP) != 0;
            }
        }

        /// The planes of `lightViewProj` (with WXL_VK_REPLAY_CULL_FRUSTUM) and the sphere (`cullSphere`, may be null).
        void MakeView(uint32_t flags, const float lightViewProj[16], const float* cullSphere, const double inverseView[16])
        {
            std::memcpy(iv, inverseView, sizeof iv);
            frustum = sphere = false;
            if (flags & WXL_VK_REPLAY_CULL_FRUSTUM)
            {
                const float* M = lightViewProj;
                const int    pick[5][2] = { { 0, 1 }, { 0, -1 }, { 1, 1 }, { 1, -1 }, { 2, -1 } };
                frustum = true;
                for (int p = 0; p < 5; ++p)
                {
                    double v[4];
                    for (int i = 0; i < 4; ++i) v[i] = double(M[i * 4 + 3]) + pick[p][1] * double(M[i * 4 + pick[p][0]]);
                    const double len = std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
                    if (!(len > 1e-12))
                    {
                        frustum = false;   // a degenerate matrix: nothing culled by it
                        break;
                    }
                    for (int i = 0; i < 4; ++i) planes[p][i] = v[i] / len;
                }
            }
            if (cullSphere && cullSphere[3] > 0.0f && std::isfinite(cullSphere[0]) && std::isfinite(cullSphere[1])
                && std::isfinite(cullSphere[2]))
            {
                sphere = true;
                for (int k = 0; k < 3; ++k) centre[k] = cullSphere[k];
                radius = cullSphere[3];
            }
        }

        bool Active() const { return frustum || sphere; }

        /// The caller's boxes (WXL_VkReplayDesc::excludeBoxes): whether a draw that is no unit's and no map object's is
        /// left out by them -- its bound's centre inside one (outside them all with WXL_VK_REPLAY_BOXES_KEEP). A draw
        /// without a bound is never left out by an exclusion, and never kept by a KEEP.
        bool Excluded(const float b[4]) const
        {
            if (!boxCount) return false;
            if (!(b[3] > 0.0f)) return keep;
            double w[3];
            for (int j = 0; j < 3; ++j) w[j] = double(b[0]) * iv[0 * 4 + j] + double(b[1]) * iv[1 * 4 + j] + double(b[2]) * iv[2 * 4 + j] + iv[3 * 4 + j];
            bool inside = false;
            for (uint32_t i = 0; i < boxCount && !inside; ++i)
            {
                const float* x = boxes + i * 6;
                inside = w[0] >= x[0] && w[1] >= x[1] && w[2] >= x[2] && w[0] <= x[3] && w[1] <= x[4] && w[2] <= x[5];
            }
            return keep ? !inside : inside;
        }

        const float* boxes = nullptr;
        uint32_t     boxCount = 0;
        bool         keep = false;

        /// Whether a draw's bound lies wholly outside: its view-space sphere taken to world, against the planes and
        /// the sphere. A draw without a bound is never out.
        bool Out(const float b[4]) const
        {
            if (!(b[3] > 0.0f)) return false;
            double w[3];
            ToWorld(b, w);
            return OutWorld(w, b[3]);
        }

        /// A bound's view-space centre taken to world (the frame's inverse view).
        void ToWorld(const float b[4], double w[3]) const
        {
            for (int j = 0; j < 3; ++j) w[j] = double(b[0]) * iv[0 * 4 + j] + double(b[1]) * iv[1 * 4 + j] + double(b[2]) * iv[2 * 4 + j] + iv[3 * 4 + j];
        }

        /// Out() for a bound already in world space (`w` its centre, `r` its radius > 0).
        bool OutWorld(const double w[3], double r) const
        {
            if (sphere)
            {
                const double dx = w[0] - centre[0], dy = w[1] - centre[1], dz = w[2] - centre[2], reach = radius + r;
                if (dx * dx + dy * dy + dz * dz > reach * reach) return true;
            }
            if (frustum)
                for (int p = 0; p < 5; ++p)
                    if (planes[p][0] * w[0] + planes[p][1] * w[1] + planes[p][2] * w[2] + planes[p][3] < -r) return true;
            return false;
        }
    };

    /// What a replay's flags keep of the capture (one rule for ReplayWorld and ReplayWorldViews).
    struct Filter
    {
        bool     alphaWanted, asOpaque, dynamicOnly, noSkinned, noWind, windStill, terrainOnly, withDoodads, withMapObjs;
        uint32_t minCount;

        Filter(uint32_t flags, uint32_t minIndexCount)
        {
            alphaWanted = (flags & WXL_VK_REPLAY_ALPHA_TESTED) != 0;
            asOpaque    = !alphaWanted && (flags & WXL_VK_REPLAY_ALPHA_AS_OPAQUE) != 0;
            dynamicOnly = (flags & WXL_VK_REPLAY_DYNAMIC_ONLY) != 0;
            noSkinned   = (flags & WXL_VK_REPLAY_NO_SKINNED) != 0;
            noWind      = (flags & WXL_VK_REPLAY_NO_WIND) != 0;
            windStill   = (flags & WXL_VK_REPLAY_WIND_STILL) != 0;
            terrainOnly = (flags & WXL_VK_REPLAY_TERRAIN_ONLY) != 0;
            withDoodads = dynamicOnly && (flags & WXL_VK_REPLAY_WITH_DOODADS) != 0;
            withMapObjs = dynamicOnly && (flags & WXL_VK_REPLAY_WITH_MAPOBJECTS) != 0;
            minCount    = minIndexCount;
        }

        /// Whether the flags keep a draw (false: skipped).
        bool Keeps(const Draw& dr) const
        {
            // A mover: a unit's draw (a character's, a creature's or an item's texture: pipelines/Record.cpp)
            // -- what a cached map must never hold. The bone palette alone makes none: the world's animated
            // doodads (banners, lanterns, bushes) are skinned too, and a cached map holds them still, in the
            // pose they were captured in; kept out of the caches they were drawn from each frame's capture
            // alone, and their shadows popped with the camera.
            const bool moving = dr.live != 0u;
            // WITH_DOODADS: a static M2 doodad's opaque draw joins the movers; its alpha-tested draws (a crown's
            // leaves, a card) stay out whatever the flags say.
            const bool doodad = withDoodads && dr.doodad != 0u && !dr.alphaTest;
            // WITH_MAPOBJECTS: a map object's draw joins them too, alpha-tested ones under the flags' rule (a railing).
            const bool mapobj = withMapObjs && dr.mapobj != 0u;
            return !((dr.alphaTest && !alphaWanted && !asOpaque) || dr.count < minCount
                     || (dynamicOnly && !moving && !dr.wind && !doodad && !mapobj) || (noSkinned && moving) || (noWind && dr.wind)
                     || (terrainOnly && !dr.terrain));
        }
    };

    /// Records one kept draw: its state bound (the wind at rest when asked), the draw. The capture's record is read as
    /// it is (no copy of it a draw and a view).
    void Record(VkCommandBuffer cmd, const Draw& dr, bool alpha, const pipes::Pipe& p, const float m[16], bool windStill,
                const WXL_VkReplayDesc& desc, Bound& b, WXL_VkReplayStats& st, const wxlvk::vk::loader::Device& d)
    {
        static const int32_t kStill[4] = {};
        BindDraw(cmd, dr, alpha, dr.wind && windStill ? kStill : dr.windPush, p, m, desc, b, d);
        if (dr.indexed) d.CmdDrawIndexed(cmd, dr.count, 1, dr.firstIndex, dr.vertexOffset, 0);
        else d.CmdDraw(cmd, dr.count, 1, static_cast<uint32_t>(dr.vertexOffset), 0);
        ++st.replayed;
        if (alpha) ++st.alphaTested;
        if (dr.mapobj) ++st.mapObjects;
    }

    void DrawAll(const WXL_VkReplayDesc& desc, const float m[16], const Cull& cull, WXL_VkReplayStats& st,
                 const wxlvk::vk::loader::Device& d)
    {
        uint32_t    count = 0;
        const Draw* draws = wxlvk::capture::Draws(count);
        st.captured = count;
        const uint32_t minCount =
            desc.structSize >= offsetof(WXL_VkReplayDesc, minIndexCount) + sizeof desc.minIndexCount ? desc.minIndexCount : 0u;
        const Filter f(desc.flags, minCount);
        Bound b;
        for (uint32_t i = 0; i < count; ++i)
        {
            const Draw& dr = draws[i];
            const bool  alpha = dr.alphaTest && f.alphaWanted;
            const bool  moving = dr.live != 0u;
            if (!f.Keeps(dr))
            {
                ++st.skipped;
                continue;
            }
            if (cull.Active() && cull.Out(dr.bound))
            {
                ++st.culled;
                continue;
            }
            // The caller's boxes: a light's own model, what a unit carries that was not drawn as the unit's (a cached map's).
            if (!moving && !dr.mapobj && !dr.terrain && cull.Excluded(dr.bound))
            {
                ++st.excluded;
                continue;
            }
            pipes::Pipe p;
            if ((dr.wind && !dr.windSet) || !pipes::Get(dr.entry, alpha, p))
            {
                ++st.skipped;
                continue;
            }
            Record(desc.cmd, dr, alpha, p, m, f.windStill, desc, b, st, d);
        }
    }

    /// A draw ReplayWorldViews keeps, and the views it is drawn into (bit v: view v).
    struct Kept
    {
        uint32_t    index;
        uint32_t    views;
        pipes::Pipe pipe;
        bool        alpha;
    };

    uint32_t Bits(uint32_t x)
    {
        uint32_t n = 0;
        for (; x; x &= x - 1) ++n;
        return n;
    }
}

namespace wxlvk::capture
{
    bool Replay(const WXL_VkReplayDesc& desc, WXL_VkReplayStats* stats)
    {
        WXL_VkReplayStats st = {};
        st.structSize = sizeof st;
        const context::Handles* h = context::Get();
        const bool         load = (desc.flags & WXL_VK_REPLAY_LOAD) != 0;
        const VkRenderPass pass = load ? shadowpass::GetLoad() : shadowpass::Get();
        float  m[16];
        double iv[16];
        if (!h || !h->dapi || pass == VK_NULL_HANDLE || !desc.cmd || desc.framebuffer == VK_NULL_HANDLE
            || !ClipToLight(desc.lightViewProj, m, iv))
        {
            ++g.refused;
            return false;
        }
        const wxlvk::vk::loader::Device& d = *h->dapi;
        Cull cull;
        cull.Make(desc, iv);
        Begin(desc, pass, d);
        DrawAll(desc, m, cull, st, d);
        VkViewport vp = {};
        vp.x = static_cast<float>(desc.area.offset.x);
        vp.y = static_cast<float>(desc.area.offset.y);
        vp.width  = static_cast<float>(desc.area.extent.width);
        vp.height = static_cast<float>(desc.area.extent.height);
        vp.maxDepth = 1.0f;
        if (!(desc.flags & WXL_VK_REPLAY_NO_CASTERS))
            st.casters = native::render::RecordShadowCasters(desc.cmd, pass, vp, desc.area, desc.lightViewProj, desc.cascade);
        d.CmdEndRenderPass(desc.cmd);
        // The draws' bind tracking knows nothing of what was bound here.
        wxlvk::vk::pipelines::InvalidateBindings();

        ++g.replays;
        g.drawn   += st.replayed;
        g.skipped += st.skipped;
        g.alpha   += st.alphaTested;
        g.culled  += st.culled;
        if (!g.saidFirst)
        {
            g.saidFirst = true;
            wxl::ScriptMgr::Log(WXL_LOG_INFO, "capture: first shadow replay (frame %llu): %u of %u captured draw(s) "
                                "replayed (%u alpha-tested), %u skipped, %u caster pass(es)",
                                static_cast<unsigned long long>(Serial()), st.replayed, st.captured, st.alphaTested,
                                st.skipped, st.casters);
        }
        if (stats)
        {
            const uint32_t n = stats->structSize < sizeof st ? stats->structSize : static_cast<uint32_t>(sizeof st);
            std::memcpy(stats, &st, n);
            stats->structSize = n;
        }
        return true;
    }

    bool ReplayViews(const WXL_VkReplayViewsDesc& desc, WXL_VkReplayStats* stats)
    {
        WXL_VkReplayStats st = {};
        st.structSize = sizeof st;
        const context::Handles* h = context::Get();
        const VkRenderPass pass = shadowpass::GetLoad();
        const uint32_t     n = desc.views && desc.viewStride >= sizeof(WXL_VkReplayView) ? desc.viewCount : 0u;
        double t[16], iv[16];
        if (!h || !h->dapi || pass == VK_NULL_HANDLE || !desc.cmd || desc.framebuffer == VK_NULL_HANDLE || n == 0
            || n > WXL_VK_REPLAY_MAX_VIEWS || !SceneInverse(t, iv))
        {
            ++g.refused;
            return false;
        }
        const wxlvk::vk::loader::Device& d = *h->dapi;

        // Each view: its matrix from the engine's clip (the scene's inverse taken once), its culling, its rectangle; the
        // render area is the rectangles' bounding box (the LOAD class touches nothing it does not draw or clear).
        struct View
        {
            WXL_VkReplayView v;
            float            m[16];
            Cull             cull;
        };
        static View views[WXL_VK_REPLAY_MAX_VIEWS];   // the render thread's alone; Cull holds 300 bytes a view
        int32_t x0 = INT32_MAX, y0 = INT32_MAX, x1 = INT32_MIN, y1 = INT32_MIN;
        VkClearRect clears[WXL_VK_REPLAY_MAX_VIEWS];
        uint32_t    clearCount = 0;
        const uint8_t* at = reinterpret_cast<const uint8_t*>(desc.views);
        for (uint32_t k = 0; k < n; ++k, at += desc.viewStride)
        {
            View& w = views[k];
            std::memcpy(&w.v, at, sizeof w.v);
            Compose(t, w.v.lightViewProj, w.m);
            w.cull = Cull();
            w.cull.MakeView(desc.flags, w.v.lightViewProj, w.v.cullSphere, iv);
            const VkRect2D& r = w.v.area;
            x0 = r.offset.x < x0 ? r.offset.x : x0;
            y0 = r.offset.y < y0 ? r.offset.y : y0;
            x1 = r.offset.x + int32_t(r.extent.width) > x1 ? r.offset.x + int32_t(r.extent.width) : x1;
            y1 = r.offset.y + int32_t(r.extent.height) > y1 ? r.offset.y + int32_t(r.extent.height) : y1;
            if (w.v.clear) clears[clearCount++] = { r, 0, 1 };
        }
        // The boxes are every view's (view-independent).
        Cull boxes;
        boxes.MakeView(0u, views[0].v.lightViewProj, nullptr, iv);
        if (desc.excludeBoxes && desc.excludeBoxCount)
        {
            boxes.boxes    = desc.excludeBoxes;
            boxes.boxCount = desc.excludeBoxCount;
            boxes.keep     = (desc.flags & WXL_VK_REPLAY_BOXES_KEEP) != 0;
        }

        // ONE walk of the capture: each draw the flags keep is taken to world once and tested against every view, in the
        // order ReplayWorld tests it (the flags, the view's culling, the boxes, a pipeline) -- the counts add up as the
        // views' ReplayWorld calls would.
        uint32_t    count = 0;
        const Draw* draws = wxlvk::capture::Draws(count);
        st.captured = count;
        const Filter f(desc.flags, desc.minIndexCount);
        static std::vector<Kept> kept;   // the render thread's alone
        kept.clear();
        const uint32_t all = n >= 32 ? 0xFFFFFFFFu : ((1u << n) - 1u);
        for (uint32_t i = 0; i < count; ++i)
        {
            const Draw& dr = draws[i];
            if (!f.Keeps(dr))
            {
                st.skipped += n;
                continue;
            }
            uint32_t mask = all;
            if (dr.bound[3] > 0.0f)
            {
                double c[3];
                boxes.ToWorld(dr.bound, c);
                for (uint32_t k = 0; k < n; ++k)
                    if (views[k].cull.Active() && views[k].cull.OutWorld(c, dr.bound[3]))
                    {
                        mask &= ~(1u << k);
                        ++st.culled;
                    }
            }
            if (!mask) continue;
            const bool moving = dr.live != 0u;
            if (!moving && !dr.mapobj && !dr.terrain && boxes.Excluded(dr.bound))
            {
                st.excluded += Bits(mask);
                continue;
            }
            const bool  alpha = dr.alphaTest && f.alphaWanted;
            pipes::Pipe p;
            if ((dr.wind && !dr.windSet) || !pipes::Get(dr.entry, alpha, p))
            {
                st.skipped += Bits(mask);
                continue;
            }
            kept.push_back({ i, mask, p, alpha });
        }

        // ONE render pass over the views' bounding box: the views asked cleared, then each view's draws in the capture's
        // order (its viewport, its scissor, its matrix; the bindings started afresh, as a ReplayWorld of its own would).
        VkRenderPassBeginInfo begin = {};
        begin.sType       = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        begin.renderPass  = pass;
        begin.framebuffer = desc.framebuffer;
        begin.renderArea  = { { x0, y0 }, { static_cast<uint32_t>(x1 - x0), static_cast<uint32_t>(y1 - y0) } };
        d.CmdBeginRenderPass(desc.cmd, &begin, VK_SUBPASS_CONTENTS_INLINE);
        if (clearCount)
        {
            VkClearAttachment clear = {};
            clear.aspectMask              = VK_IMAGE_ASPECT_DEPTH_BIT;
            clear.clearValue.depthStencil = { 1.0f, 0 };
            d.CmdClearAttachments(desc.cmd, 1, &clear, clearCount, clears);
        }
        WXL_VkReplayDesc shared = {};   // what BindDraw reads: the command buffer and the bias
        shared.structSize = sizeof shared;
        shared.cmd        = desc.cmd;
        shared.depthBias  = desc.depthBias;
        shared.slopeBias  = desc.slopeBias;
        shared.flags      = desc.flags;
        for (uint32_t k = 0; k < n; ++k)
        {
            const View& w = views[k];
            VkViewport vp = {};
            vp.x        = static_cast<float>(w.v.area.offset.x);
            vp.y        = static_cast<float>(w.v.area.offset.y);
            vp.width    = static_cast<float>(w.v.area.extent.width);
            vp.height   = static_cast<float>(w.v.area.extent.height);
            vp.maxDepth = 1.0f;
            d.CmdSetViewport(desc.cmd, 0, 1, &vp);
            d.CmdSetScissor(desc.cmd, 0, 1, &w.v.area);
            Bound b;
            const uint32_t bit = 1u << k;
            for (const Kept& e : kept)
                if (e.views & bit) Record(desc.cmd, draws[e.index], e.alpha, e.pipe, w.m, f.windStill, shared, b, st, d);
            if (!(desc.flags & WXL_VK_REPLAY_NO_CASTERS))
                st.casters += native::render::RecordShadowCasters(desc.cmd, pass, vp, w.v.area, w.v.lightViewProj, w.v.cascade);
        }
        d.CmdEndRenderPass(desc.cmd);
        wxlvk::vk::pipelines::InvalidateBindings();

        g.replays += n;
        g.drawn   += st.replayed;
        g.skipped += st.skipped;
        g.alpha   += st.alphaTested;
        g.culled  += st.culled;
        ++g.multi;
        if (stats)
        {
            const uint32_t sz = stats->structSize < sizeof st ? stats->structSize : static_cast<uint32_t>(sizeof st);
            std::memcpy(stats, &st, sz);
            stats->structSize = sz;
        }
        return true;
    }

    void LogStats()
    {
        Stats cs;
        GetStats(cs);
        uint32_t built = 0, failed = 0;
        shadowpipes::Counts(built, failed);
        if (!cs.taken && !g.replays) return;
        wxl::ScriptMgr::Log(WXL_LOG_INFO,
                            "capture: %llu frame(s) captured, %llu draw(s) (%u last frame, %llu past the table, %llu movers); "
                            "%llu replay(s) (%llu several-view pass(es) among them): %llu draw(s), %llu alpha-tested, %llu skipped, "
                            "%llu culled by their bounds, %llu refused; shadow pipelines %u built, %u failed",
                            static_cast<unsigned long long>(cs.frames), static_cast<unsigned long long>(cs.taken),
                            cs.lastFrame, static_cast<unsigned long long>(cs.overflow), static_cast<unsigned long long>(cs.movers),
                            static_cast<unsigned long long>(g.replays), static_cast<unsigned long long>(g.multi),
                            static_cast<unsigned long long>(g.drawn),
                            static_cast<unsigned long long>(g.alpha), static_cast<unsigned long long>(g.skipped),
                            static_cast<unsigned long long>(g.culled), static_cast<unsigned long long>(g.refused), built, failed);
    }
}
