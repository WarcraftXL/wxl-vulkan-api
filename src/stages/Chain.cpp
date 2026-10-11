// The stages: consumers' compute over the frame (wxl.vk.render RegisterStage), run in order at their
// point of the engine's frame.
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

#include "stages/Chain.hpp"

#include "native/Present.hpp"
#include "native/Render.hpp"
#include "native/render/DepthHistory.hpp"
#include "native/render/Jitter.hpp"
#include "stages/Gpu.hpp"
#include "stages/Targets.hpp"
#include "stages/Timing.hpp"
#include "vulkan/Pipelines.hpp"

#include "wxl/Script.hpp"

#include <cstring>

namespace
{
    namespace present = wxlvk::native::present;
    namespace gpu     = wxlvk::stages::gpu;
    namespace targets = wxlvk::stages::targets;
    namespace timing  = wxlvk::stages::timing;

    using wxlvk::stages::Io;
    using wxlvk::stages::Point;
    using wxlvk::stages::chain::kMaxStages;

    constexpr uint32_t kPoints = static_cast<uint32_t>(Point::Count);
    static_assert(kPoints + kMaxStages <= timing::kLanes, "a timing lane per point and per stage");

    constexpr size_t kNameChars = 48;

    const char* const kPointNames[kPoints] = { "AfterOpaqueWorld", "BeforeUi", "AfterLiquids" };

    /// The public point of a stage point, and back (an unknown public point is taken as AfterOpaqueWorld,
    /// as before AfterLiquids existed; Register refuses the others).
    uint32_t ToPublic(Point p)
    {
        return p == Point::BeforeUi ? WXL_VK_POINT_BEFORE_UI
             : p == Point::AfterLiquids ? WXL_VK_POINT_AFTER_LIQUIDS
                                        : WXL_VK_POINT_AFTER_OPAQUE_WORLD;
    }

    Point FromPublic(uint32_t point)
    {
        return point == WXL_VK_POINT_BEFORE_UI ? Point::BeforeUi
             : point == WXL_VK_POINT_AFTER_LIQUIDS ? Point::AfterLiquids
                                                   : Point::AfterOpaqueWorld;
    }

    /// Why a point's latest run did or did not happen.
    enum class Outcome : uint8_t
    {
        Idle,        ///< no stage wanted
        Ran,
        NoFrame,     ///< no frame open, or the draw path is down
        OffBack,     ///< a point that needs the world's own pass found a texture target current
        NoBack,      ///< the direct path (no own back buffer) or no image this frame
        Format,      ///< a back buffer format the stages cannot copy
        NotReady,    ///< the working image could not be made
        NoDepth,     ///< the stages need the scene depth and it could not be had
        Count,
    };

    const char* const kOutcomeText[] = {
        "idle (nothing enabled)",
        "running",
        "waiting: no frame open",
        "skipped: a texture target is current",
        "unavailable: no own back buffer (direct path)",
        "unavailable: back buffer format",
        "failed: working image (see the log)",
        "skipped: no scene depth this frame",
    };
    static_assert(sizeof kOutcomeText / sizeof *kOutcomeText == static_cast<size_t>(Outcome::Count), "one text per outcome");

    struct PointState
    {
        uint64_t lastSerial = 0;
        Outcome  outcome    = Outcome::Idle;
        uint64_t counts[static_cast<size_t>(Outcome::Count)] = {};
    };

    struct Stage
    {
        WXL_VkStageDesc desc = {};
        Point    point = Point::AfterOpaqueWorld;
        char     name[kNameChars] = {};
        uint32_t lane  = 0;       ///< its timing lane, fixed at registration
        bool     live  = false;
        uint64_t runs  = 0;
        bool     wanted = false;  ///< the latest answer of its Wanted, this run
    };

    /// The world's scene as AfterOpaqueWorld saw it: what BeforeUi and AfterLiquids reconstruct with, since
    /// the engine's own full-screen passes (and the liquid pass's matrices) may have set their own by then.
    struct WorldScene
    {
        uint64_t             serial = 0;
        WXL_VkSceneConstants scene  = {};
        VkViewport           viewport = {};
        float                jitter[2] = { 0.0f, 0.0f };   ///< what `viewport` carries
        bool                 valid  = false;
    };

    struct State
    {
        WorldScene world;
        Stage      stages[kMaxStages];
        Stage*     order[kMaxStages] = {};   ///< the registered stages by order
        uint32_t   count = 0;
        PointState points[kPoints];
        uint64_t   lastColour   = 0;         ///< the last frame the working image was used
        bool       formatLogged = false;
        bool       firstLogged  = false;
    };
    State g;

    /// Asks every live stage of the point whether it runs this frame (once per run); true when any does.
    bool AskWanted(Point p, bool& colour, bool& depth)
    {
        bool any = false;
        colour = depth = false;
        for (uint32_t i = 0; i < g.count; ++i)
        {
            Stage& s = *g.order[i];
            s.wanted = false;
            if (!s.live || s.point != p) continue;
            s.wanted = !s.desc.Wanted || s.desc.Wanted(s.desc.user) != 0;
            if (!s.wanted) continue;
            any    = true;
            colour = colour || (s.desc.flags & WXL_VK_STAGE_WRITES_COLOR) != 0;
            depth  = depth || (s.desc.flags & WXL_VK_STAGE_NEEDS_DEPTH) != 0;
        }
        return any;
    }

    /// The back buffer's channel order: false for a format the stages cannot copy.
    bool ChannelOrder(VkFormat format, bool& swapRB)
    {
        swapRB = format == VK_FORMAT_B8G8R8A8_UNORM;
        if (format == VK_FORMAT_B8G8R8A8_UNORM || format == VK_FORMAT_R8G8B8A8_UNORM) return true;
        if (!g.formatLogged)
        {
            g.formatLogged = true;
            wxl::ScriptMgr::Log(WXL_LOG_WARN, "stages: the back buffer is %s; the stages copy only B8G8R8A8 / R8G8B8A8 "
                                "and stay off", wxlvk::vk::FormatName(format));
        }
        return false;
    }

    /// Whether the frame can be worked on now; `io` receives its extent and channel order.
    Outcome Preconditions(Point p, Io& io)
    {
        if (!present::DrawPathUp() || !present::FrameOpen()) return Outcome::NoFrame;
        // AfterOpaqueWorld's and AfterLiquids' stages read the depth of the pass the world draws into: only
        // that pass.
        if ((p == Point::AfterOpaqueWorld || p == Point::AfterLiquids) && !present::OnBackBuffer()) return Outcome::OffBack;
        VkFormat format = VK_FORMAT_UNDEFINED;
        if (!present::BackBufferInfo(format, io.extent)) return Outcome::NoBack;
        if (!ChannelOrder(format, io.swapRB)) return Outcome::Format;
        return Outcome::Ran;
    }

    void Copy(VkCommandBuffer cmd, VkImage src, VkImageLayout srcLayout, VkImage dst, VkImageLayout dstLayout,
              VkExtent2D extent)
    {
        VkImageCopy c = {};
        c.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        c.srcSubresource.layerCount = 1;
        c.dstSubresource            = c.srcSubresource;
        c.extent                    = VkExtent3D{ extent.width, extent.height, 1 };
        gpu::Api()->CmdCopyImage(cmd, src, srcLayout, dst, dstLayout, 1, &c);
    }

    /// The back buffer into the working image, ready for compute.
    bool CopyIn(Io& io)
    {
        if (!targets::Get(io.extent, io.cmd, io.serial, io.frame)) return false;
        VkImage back = VK_NULL_HANDLE;
        VkFormat format = VK_FORMAT_UNDEFINED;
        VkExtent2D extent = {};
        if (!present::BackBuffer(io.cmd, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, back, format, extent)) return false;
        // The working image's earlier readers and writers (the previous run) come first.
        gpu::Barrier(io.cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                     VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT,
                     VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT);
        Copy(io.cmd, back, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, io.frame.image, VK_IMAGE_LAYOUT_GENERAL, io.extent);
        gpu::Barrier(io.cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
                     VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
        g.lastColour = io.serial;
        return true;
    }

    /// The working image back onto the back buffer.
    void CopyOut(const Io& io)
    {
        gpu::Barrier(io.cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
                     VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_READ_BIT);
        VkImage back = VK_NULL_HANDLE;
        VkFormat format = VK_FORMAT_UNDEFINED;
        VkExtent2D extent = {};
        if (!present::BackBuffer(io.cmd, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, back, format, extent)) return;
        Copy(io.cmd, io.frame.image, VK_IMAGE_LAYOUT_GENERAL, back, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, io.extent);
    }

    WXL_VkStageContext Context(Point p, const Io& io)
    {
        WXL_VkStageContext ctx = {};
        ctx.structSize      = sizeof ctx;
        ctx.point           = ToPublic(p);
        ctx.serial          = io.serial;
        ctx.completedSerial = present::Completed();
        ctx.frameSlot       = io.slot;
        ctx.frameSlots      = wxlvk::vk::pipelines::kFrameSlots;
        ctx.cmd             = io.cmd;
        ctx.extent          = io.extent;
        ctx.colorImage      = io.frame.image;
        ctx.colorView       = io.frame.view;
        ctx.colorFormat     = io.frame.format;
        ctx.colorLayout     = VK_IMAGE_LAYOUT_GENERAL;
        ctx.swapRB          = io.swapRB ? 1u : 0u;
        ctx.depthView       = io.depthView;
        ctx.depthFormat     = io.depthFormat;
        ctx.depthLayout     = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        ctx.scene           = &io.scene;
        ctx.viewport        = io.viewport;
        ctx.sceneValid      = io.sceneValid ? 1u : 0u;
        ctx.jitter[0]       = io.jitter[0];
        ctx.jitter[1]       = io.jitter[1];
        return ctx;
    }

    /// Every wanted stage of the point, in order; how many ran. `depthMissing` is set when one was
    /// skipped for want of the depth.
    uint32_t RunStages(Point p, const Io& io, bool& depthMissing)
    {
        const WXL_VkStageContext ctx = Context(p, io);
        uint32_t ran = 0;
        for (uint32_t i = 0; i < g.count; ++i)
        {
            Stage& s = *g.order[i];
            if (!s.wanted || s.point != p) continue;
            if ((s.desc.flags & WXL_VK_STAGE_NEEDS_DEPTH) && io.depthView == VK_NULL_HANDLE)
            {
                depthMissing = true;
                continue;
            }
            timing::Start(io.cmd, s.lane);
            s.desc.Run(s.desc.user, &ctx);
            timing::Stop(io.cmd, s.lane);
            gpu::ComputeToCompute(io.cmd);
            ++s.runs;
            ++ran;
        }
        return ran;
    }

    void LogFirst(Point p, const Io& io)
    {
        if (g.firstLogged) return;
        g.firstLogged = true;
        wxl::ScriptMgr::Log(WXL_LOG_INFO, "stages: first run at %s (frame %llu, %ux%u, %s back buffer, GPU timing %s)",
                            kPointNames[static_cast<size_t>(p)], static_cast<unsigned long long>(io.serial),
                            io.extent.width, io.extent.height, io.swapRB ? "BGRA" : "RGBA",
                            timing::Available() ? "on" : "unavailable");
    }

    /// The run proper, preconditions met.
    Outcome Execute(Point p, bool colour, bool depth, Io& io)
    {
        io.cmd    = present::Outside();
        io.slot   = present::Slot();
        io.serial = present::Serial();
        if (!io.cmd) return Outcome::NoFrame;
        const uint32_t lane = static_cast<uint32_t>(p);
        timing::Open(io.cmd, io.slot, io.serial);
        timing::Start(io.cmd, lane);
        if (p != Point::AfterOpaqueWorld && g.world.valid && g.world.serial == io.serial)
        {
            io.scene      = g.world.scene;
            io.viewport   = g.world.viewport;
            io.jitter[0]  = g.world.jitter[0];
            io.jitter[1]  = g.world.jitter[1];
            io.sceneValid = true;
        }
        else
            io.sceneValid = wxlvk::native::render::SceneNow(io.extent, io.scene, io.viewport, io.jitter);
        if (colour && !CopyIn(io))
        {
            timing::Stop(io.cmd, lane);
            return Outcome::NotReady;
        }
        if (depth) present::SceneDepth(io.cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, io.depthView, io.depthFormat);
        bool depthMissing = false;
        const uint32_t ran = RunStages(p, io, depthMissing);
        if (colour) CopyOut(io);
        timing::Stop(io.cmd, lane);
        // The compute binds share nothing with the draws' tracking, but the next pass begins clean.
        wxlvk::vk::pipelines::InvalidateBindings();
        if (!ran) return depthMissing ? Outcome::NoDepth : Outcome::NotReady;
        LogFirst(p, io);
        return Outcome::Ran;
    }

    /// The working image goes back once no point has used it for longer than the frames in flight.
    void ReleaseIdle()
    {
        if (!present::FrameOpen()) return;
        if (present::Serial() > g.lastColour + wxlvk::vk::pipelines::kFrameSlots + 1) targets::Release(g.lastColour);
    }

    Stage* Find(void* handle)
    {
        Stage* s = static_cast<Stage*>(handle);
        return s >= g.stages && s < g.stages + kMaxStages && s->lane ? s : nullptr;
    }

    double Avg(const timing::Lane& t) { return t.samples ? t.sumMs / static_cast<double>(t.samples) : 0.0; }
}

namespace wxlvk::stages::chain
{
    void* Add(const WXL_VkStageDesc& desc, const char* name)
    {
        if (g.count >= kMaxStages) return nullptr;
        Stage& s = g.stages[g.count];
        s.desc  = desc;
        s.point = FromPublic(desc.point);
        std::strncpy(s.name, name ? name : "stage", kNameChars - 1);
        s.name[kNameChars - 1] = '\0';
        s.lane = kPoints + g.count;
        s.live = true;
        // Kept sorted by order (stable: an equal order goes after).
        uint32_t at = g.count;
        while (at > 0 && g.order[at - 1]->desc.order > desc.order)
        {
            g.order[at] = g.order[at - 1];
            --at;
        }
        g.order[at] = &s;
        ++g.count;
        return &s;
    }

    void Remove(void* stage)
    {
        if (Stage* s = Find(stage)) s->live = false;
    }

    bool Stats(void* stage, WXL_VkStageStats& out)
    {
        const Stage* s = Find(stage);
        if (!s) return false;
        WXL_VkStageStats st = {};
        st.structSize = out.structSize;
        st.point      = ToPublic(s->point);
        st.runs       = s->runs;
        const PointState& ps = g.points[static_cast<size_t>(s->point)];
        st.pointStatus = kOutcomeText[static_cast<size_t>(ps.outcome)];
        if (timing::Available())
        {
            const timing::Lane& t = timing::Get(s->lane);
            st.lastMs  = static_cast<float>(t.lastMs);
            st.avgMs   = static_cast<float>(Avg(t));
            st.maxMs   = static_cast<float>(t.maxMs);
            st.pointMs = static_cast<float>(timing::Get(static_cast<uint32_t>(s->point)).lastMs);
        }
        else
            st.lastMs = st.avgMs = st.maxMs = st.pointMs = -1.0f;
        // Only what the caller's structure has room for.
        const size_t n = out.structSize < sizeof st ? out.structSize : sizeof st;
        std::memcpy(&out, &st, n);
        out.structSize = static_cast<uint32_t>(n);
        return true;
    }

    void NoteWorld()
    {
        VkFormat   format = VK_FORMAT_UNDEFINED;
        VkExtent2D extent = {};
        g.world.valid = false;
        if (!present::FrameOpen() || !present::OnBackBuffer() || !present::BackBufferInfo(format, extent)) return;
        g.world.serial = present::Serial();
        g.world.valid  = wxlvk::native::render::SceneNow(extent, g.world.scene, g.world.viewport, g.world.jitter);
    }

    void Run(Point point)
    {
        if (point >= Point::Count) return;
        if (point == Point::AfterOpaqueWorld) NoteWorld();
        PointState& ps = g.points[static_cast<size_t>(point)];
        if (present::FrameOpen() && ps.lastSerial == present::Serial()) return;   // once per frame
        bool colour = false, depth = false;
        if (!AskWanted(point, colour, depth))
        {
            ps.outcome = Outcome::Idle;
            ReleaseIdle();
            return;
        }
        Io io;
        Outcome outcome = Preconditions(point, io);
        if (outcome == Outcome::Ran)
        {
            ps.lastSerial = present::Serial();
            outcome = Execute(point, colour, depth, io);
        }
        ps.outcome = outcome;
        ++ps.counts[static_cast<size_t>(outcome)];
        ReleaseIdle();
    }

    void LogStats()
    {
        for (uint32_t p = 0; p < kPoints; ++p)
        {
            const PointState& ps = g.points[p];
            const timing::Lane& t = timing::Get(p);
            uint64_t reached = 0;
            for (uint64_t c : ps.counts) reached += c;
            if (!reached) continue;
            wxl::ScriptMgr::Log(WXL_LOG_INFO,
                                "stages: %s: %llu run(s), skipped %llu no frame / %llu off the back buffer / %llu no back "
                                "buffer / %llu format / %llu failed / %llu no depth; GPU avg %.3f ms, max %.3f (copies "
                                "included); working image %llu MiB",
                                kPointNames[p], static_cast<unsigned long long>(ps.counts[static_cast<size_t>(Outcome::Ran)]),
                                static_cast<unsigned long long>(ps.counts[static_cast<size_t>(Outcome::NoFrame)]),
                                static_cast<unsigned long long>(ps.counts[static_cast<size_t>(Outcome::OffBack)]),
                                static_cast<unsigned long long>(ps.counts[static_cast<size_t>(Outcome::NoBack)]),
                                static_cast<unsigned long long>(ps.counts[static_cast<size_t>(Outcome::Format)]),
                                static_cast<unsigned long long>(ps.counts[static_cast<size_t>(Outcome::NotReady)]),
                                static_cast<unsigned long long>(ps.counts[static_cast<size_t>(Outcome::NoDepth)]), Avg(t),
                                t.maxMs, static_cast<unsigned long long>(targets::Bytes() >> 20));
        }
        for (uint32_t i = 0; i < g.count; ++i)
        {
            const Stage& s = *g.order[i];
            const timing::Lane& t = timing::Get(s.lane);
            if (!s.runs) continue;
            wxl::ScriptMgr::Log(WXL_LOG_INFO, "stages: '%s' at %s (order %d): %llu run(s), GPU avg %.3f ms, max %.3f",
                                s.name, kPointNames[static_cast<size_t>(s.point)], s.desc.order,
                                static_cast<unsigned long long>(s.runs), Avg(t), t.maxMs);
        }
        if (const uint64_t copies = wxlvk::native::render::depthhistory::Copies())
            wxl::ScriptMgr::Log(WXL_LOG_INFO, "stages: depth history: %llu copy(ies) of the opaque depth for the "
                                "rendering API's consumers", static_cast<unsigned long long>(copies));
        uint64_t jittered = 0, skipped = 0;
        wxlvk::native::render::jitter::Counts(jittered, skipped);
        if (jittered || skipped)
            wxl::ScriptMgr::Log(WXL_LOG_INFO, "stages: jitter: %llu world draw(s) jittered, %llu full-depth-range draw(s) "
                                "in the world pass left alone", static_cast<unsigned long long>(jittered),
                                static_cast<unsigned long long>(skipped));
        timing::Reset();
    }
}
