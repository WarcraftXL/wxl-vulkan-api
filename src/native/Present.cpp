// The native device's frame: one command buffer open across the slots, render passes on the current
// target (the own back buffer or a texture), the transfer rule, the queries, and the present.
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

#include "native/present/Internal.hpp"

using namespace wxlvk::native::present::detail;

namespace wxlvk::native::present
{
    bool InitDrawPath()
    {
        if (s.drawTried) return s.drawPath;
        if (!context::Live()) return false;   // retried once the device is up
        s.drawTried = true;
        s.drawPath  = memory::Init() && texture::Init() && wxlvk::vk::shaders::Init() && pipelines::Init();
        wxl::ScriptMgr::Log(s.drawPath ? WXL_LOG_DEBUG : WXL_LOG_ERROR,
                            s.drawPath ? "native: frame: the draw path is up (memory, textures, shader "
                                         "modules, pipelines); engine draws are recorded into the "
                                         "frame's render passes on the client window"
                                       : "native: frame: the draw path could not be brought up (the "
                                         "lines above say which module declined); the frame presents "
                                         "clears only");
        return s.drawPath;
    }

    bool DrawPathUp() { return s.drawPath; }

    bool EnsureFrame()
    {
        if (s.failed) return false;   // a hard failure stops the frame for good, open or not
        if (s.open) return true;
        const context::Handles* h = context::Get();
        if (!h || !h->device || !h->dapi) return false;
        if (!EnsureDeviceObjects(*h)) return false;

        const loader::Device& d    = *s.dapi;
        FrameSlot&            slot = s.slots[s.fi];
        VkResult r = VK_SUCCESS;
        {
            wxlvk::native::perf::Scope t(wxlvk::native::perf::kFenceWait);   // the GPU two frames behind
            r = d.WaitForFences(s.device, 1, &slot.fence, VK_TRUE, UINT64_MAX);
        }
        if (r != VK_SUCCESS)
        {
            Fail("vkWaitForFences (frame slot)", r);
            return false;
        }
        // One queue, submitted in order: this slot's last frame finishing proves every earlier one did.
        if (slot.serial > s.completed) s.completed = slot.serial;
        Poll();   // the other slot may have finished as well
        CollectRetired();
        ResetStaging(slot);

        // The queries this slot's last frame ran are read before its pool is reset below.
        for (uint32_t i = s.runCount[s.fi]; i > 0; --i)
            if (QueryRun* run = s.runs[s.fi][i - 1])
                if (!Harvest(*run, false))
                {
                    run->samples   = 0;
                    run->harvested = true;
                    Unregister(*run);
                }
        for (uint32_t i = 0; i < s.runCount[s.fi]; ++i)
            if (s.runs[s.fi][i]) s.runs[s.fi][i]->registered = false;
        s.runCount[s.fi]  = 0;
        s.queryNext[s.fi] = 0;

        s.serial = s.serial + 1;
        detail::serialNow = s.serial;   // Serial() answers the frame being opened from here on
        if (s.drawPath) pipelines::BeginFrame(s.fi);

        VkCommandBufferBeginInfo begin = {};
        begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        r = d.BeginCommandBuffer(slot.main, &begin);
        if (r != VK_SUCCESS)
        {
            Fail("vkBeginCommandBuffer", r);
            return false;
        }
        if (s.queryPools[s.fi] != VK_NULL_HANDLE)
            d.CmdResetQueryPool(slot.main, s.queryPools[s.fi], 0, kQueriesPerSlot);
        FrameGpuOpen(slot.main, s.fi, s.serial);   // the slot's last GPU times read, this frame's start (FrameGpu.cpp)
        s.open = true;
        s.preBegun = s.passOpen = s.acquired = s.noImage = s.drewThisFrame = s.backTouched = false;
        s.depthFresh      = true;
        s.breaksThisFrame = 0;
        s.breakView       = VK_NULL_HANDLE;
        s.opaqueDone      = false;
        s.gbufFresh       = true;
        s.gbufDrawn       = false;
        s.passGBuffer     = false;
        return true;
    }

    /// Whether the open pass, on the back buffer, is of the other class than one begun now would be:
    /// the G-buffer class is wanted from the world's first draw to AfterOpaqueWorld, nowhere else, and
    /// only while the images can be had -- the same decision BeginPass makes, or a frame on the direct
    /// path would break its pass at every draw.
    static bool ClassChanged()
    {
        if (!s.passColour.back || !s.passHasDepth || !s.passDepth.back) return false;
        return (GBufferWanted() && EnsureGBufferImages()) != s.passGBuffer;
    }

    /// Ends the open pass so the next one takes the class the draws want now: a break, counted.
    static void SwitchClass()
    {
        s.breakView = s.passColour.view;
        EndPass();
        ++s.stats.gbufferSwitches;
        ++s.stats.passBreaks;
        ++s.breaksThisFrame;
    }

    VkCommandBuffer Pass()
    {
        if (!EnsureFrame()) return nullptr;
        if (s.passOpen && !ClassChanged()) return s.slots[s.fi].main;
        if (s.passOpen) SwitchClass();
        return BeginPass();
    }

    VkCommandBuffer PassClearing(const PassClear& clear, uint32_t& folded)
    {
        folded = 0;
        if (!EnsureFrame()) return nullptr;
        if (s.passOpen && !ClassChanged()) return s.slots[s.fi].main;
        if (s.passOpen) SwitchClass();
        return BeginPass(&clear, &folded);
    }

    VkCommandBuffer Outside()
    {
        if (!EnsureFrame()) return nullptr;
        if (s.passOpen)
        {
            s.breakView = s.passColour.view;
            EndPass();
            ++s.stats.passBreaks;
            ++s.breaksThisFrame;
        }
        return s.slots[s.fi].main;
    }

    VkCommandBuffer PrePass()
    {
        if (!EnsureFrame()) return nullptr;
        FrameSlot& slot = s.slots[s.fi];
        if (s.preBegun) return slot.pre;
        VkCommandBufferBeginInfo begin = {};
        begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        const VkResult r = s.dapi->BeginCommandBuffer(slot.pre, &begin);
        if (r != VK_SUCCESS)
        {
            Fail("vkBeginCommandBuffer (pre-pass)", r);
            return nullptr;
        }
        // The copies about to be recorded write buffers the frame still in flight may be reading
        // (write-after-read across submissions): wait for those reads first.
        VkMemoryBarrier m = {};
        m.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        s.dapi->CmdPipelineBarrier(slot.pre,
                                   VK_PIPELINE_STAGE_VERTEX_INPUT_BIT | VK_PIPELINE_STAGE_VERTEX_SHADER_BIT |
                                       VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                                   VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &m, 0, nullptr, 0, nullptr);
        FrameGpuPrePass(slot.pre, s.fi);   // the pre-pass runs first: the frame's GPU span starts here
        s.preBegun = true;
        return slot.pre;
    }

    bool FrameOpen() { return s.open; }
    uint64_t Completed() { return s.completed; }
    bool UsedThisFrame(uint64_t lastUsed) { return s.open && lastUsed == s.serial; }
    uint32_t Slot() { return s.fi; }
    VkRenderPass RenderPass() { return s.passClass; }
    VkExtent2D Extent() { return s.passOpen ? s.passExtent : s.extent; }
    bool HasDepth() { return s.passOpen && s.passHasDepth; }

    void PassFormats(VkFormat& colour, VkFormat& depth)
    {
        colour = s.passOpen ? s.passColour.format : VK_FORMAT_UNDEFINED;
        depth  = s.passOpen && s.passHasDepth ? s.passDepth.format : VK_FORMAT_UNDEFINED;
    }

    void NoteDraw() { s.drewThisFrame = true; }

    void NoteWorld(bool open)
    {
        s.worldOpen = open;
        if (open) s.opaqueDone = false;
    }

    void NoteOpaqueEnd() { s.opaqueDone = true; }

    void RequestGBuffer(bool wanted)
    {
        if (wanted) ++s.gbufRequests;
        else if (s.gbufRequests) --s.gbufRequests;
    }

    bool GBufferInfo(GBufferDesc& out)
    {
        out = GBufferDesc();
        if (s.gbufImage[0] == VK_NULL_HANDLE) return false;
        // Valid once the frame's opaque world was drawn into them and that pass is over: never while a
        // pass of the class is still writing them.
        out.valid  = s.open && s.gbufDrawn && s.gbufSerial == s.serial && !s.passGBuffer;
        out.serial = s.gbufSerial;
        for (uint32_t i = 0; i < kGBufferAttachments; ++i)
        {
            out.views[i]   = s.gbufView[i];
            out.formats[i] = s.gbufFormat[i];
        }
        out.layout = VK_IMAGE_LAYOUT_GENERAL;
        out.extent = s.extent;
        return true;
    }

    bool Stage(VkDeviceSize bytes, VkBuffer& buffer, VkDeviceSize& offset, void*& mapped)
    {
        if (!EnsureFrame() || bytes == 0)
        {
            ++s.stats.stagingFailed;
            return false;
        }
        FrameSlot& slot = s.slots[s.fi];
        for (;;)
        {
            if (slot.chunkCursor < slot.chunkCount)
            {
                Chunk&             c  = slot.chunks[slot.chunkCursor];
                const VkDeviceSize at = (c.used + kStageAlign - 1) / kStageAlign * kStageAlign;
                if (at + bytes <= c.buffer.size)
                {
                    c.used   = at + bytes;
                    buffer   = c.buffer.handle;
                    offset   = at;
                    mapped   = static_cast<char*>(c.buffer.block.mapped) + static_cast<size_t>(at);
                    s.stats.stagingBytes += bytes;
                    return true;
                }
                ++slot.chunkCursor;
                continue;
            }
            if (slot.chunkCount == kMaxChunks)
            {
                ++s.stats.stagingFailed;
                return false;
            }
            Chunk& fresh = slot.chunks[slot.chunkCount];
            const VkDeviceSize size = bytes > kChunkBytes ? bytes : kChunkBytes;
            if (!memory::CreateBuffer(size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, memory::Where::HostVisible,
                                      fresh.buffer)
                || !fresh.buffer.block.mapped)
            {
                memory::DestroyBuffer(fresh.buffer);
                ++s.stats.stagingFailed;
                return false;
            }
            fresh.used = 0;
            ++slot.chunkCount;
            ++s.stats.stagingChunks;
        }
    }

    bool StageUpload(VkDeviceSize bytes, VkBuffer& buffer, VkDeviceSize& offset, void*& mapped, bool& full)
    {
        full = false;
        if (!EnsureFrame() || bytes == 0) return false;
        if (bytes > kUpChunkBytes)
        {
            ++s.stats.uploadOversize;   // a region larger than a chunk: the caller's synchronous path
            return false;
        }
        FrameSlot& slot = s.slots[s.fi];
        for (;;)
        {
            if (slot.upCursor < slot.upCount)
            {
                Chunk&             c  = slot.up[slot.upCursor];
                const VkDeviceSize at = (c.used + kStageAlign - 1) / kStageAlign * kStageAlign;
                if (at + bytes <= c.buffer.size)
                {
                    c.used   = at + bytes;
                    buffer   = c.buffer.handle;
                    offset   = at;
                    mapped   = static_cast<char*>(c.buffer.block.mapped) + static_cast<size_t>(at);
                    s.stats.uploadBytes += bytes;
                    return true;
                }
                ++slot.upCursor;
                continue;
            }
            if (slot.upCount == kMaxUpChunks)
            {
                full = true;   // the caller flushes (FlushUploads) and asks again
                return false;
            }
            Chunk& fresh = slot.up[slot.upCount];
            if (!memory::CreateBuffer(kUpChunkBytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, memory::Where::HostVisible,
                                      fresh.buffer)
                || !fresh.buffer.block.mapped)
            {
                memory::DestroyBuffer(fresh.buffer);
                // No memory for another chunk: a flush frees the ones there are, when there are any.
                full = slot.upCount != 0;
                if (!full) ++s.stats.stagingFailed;
                return false;
            }
            fresh.used = 0;
            ++slot.upCount;
            ++s.stats.uploadChunks;
            if (s.stats.uploadChunks > s.stats.uploadChunksPeak) s.stats.uploadChunksPeak = s.stats.uploadChunks;
        }
    }

    bool FlushUploads()
    {
        if (!s.open || s.failed) return false;
        if (!FlushAndWait()) return false;
        // Everything recorded so far ran and was waited for: every span of the slot's upload ring
        // is free (its staging ring is left alone -- a static lock's span may still be open there).
        FrameSlot& slot = s.slots[s.fi];
        for (uint32_t i = 0; i < slot.upCount; ++i) slot.up[i].used = 0;
        slot.upCursor = 0;
        ++s.stats.uploadFlushes;
        return true;
    }

    void RetireImage(texture::Image& image, uint64_t lastUsed)
    {
        if (image.handle == VK_NULL_HANDLE) return;
        Retired e;
        e.kind   = Kind::Image;
        e.img    = image;
        e.serial = lastUsed;
        image    = texture::Image();
        Retire(e);
    }

    void RetireBuffer(memory::Buffer& buffer, uint64_t lastUsed)
    {
        if (buffer.handle == VK_NULL_HANDLE) return;
        Retired e;
        e.kind   = Kind::Buffer;
        e.buf    = buffer;
        e.serial = lastUsed;
        buffer   = memory::Buffer();
        Retire(e);
    }

    void RetireRaw(VkImage image, VkImageView view, const memory::Block& block, uint64_t lastUsed)
    {
        if (image == VK_NULL_HANDLE && view == VK_NULL_HANDLE) return;
        Retired e;
        e.kind     = Kind::Raw;
        e.raw      = image;
        e.view     = view;
        e.rawBlock = block;
        e.serial   = lastUsed;
        Retire(e);
    }

    bool WaitSerial(uint64_t serial)
    {
        if (serial <= s.completed) return true;
        if (serial > s.submitted || !s.dapi) return false;   // the open frame: nothing to wait for yet
        for (FrameSlot& slot : s.slots)
        {
            // The slot that submitted `serial` still holds it; a slot that moved past it waited it.
            if (slot.serial < serial) continue;
            wxlvk::native::perf::Scope t(wxlvk::native::perf::kFlush);
            const VkResult r = s.dapi->WaitForFences(s.device, 1, &slot.fence, VK_TRUE, UINT64_MAX);
            if (r != VK_SUCCESS)
            {
                Fail("vkWaitForFences (a retired serial)", r);
                return false;
            }
            ++s.stats.serialWaits;
            if (slot.serial > s.completed) s.completed = slot.serial;
            break;
        }
        Poll();
        return serial <= s.completed;
    }

    void Collect()
    {
        Poll();
        CollectRetired();
    }

    void PresentBlank()
    {
        if (s.blankPresented || s.failed || s.presentedCount != 0) return;
        if (!EnsureFrame()) return;
        s.blankPresented = true;
        // Pass() begins the frame's first pass, whose first use clears to black (and the far plane);
        // with no draw in it, FinishFrame presents exactly that. The liveness colour is Present()'s.
        const bool drew = Pass() != nullptr;
        FinishFrame(true);
        wxl::ScriptMgr::Log(WXL_LOG_DEBUG,
                            "native: frame: one black frame %s right after the bring-up, so the "
                            "window is not left blank for the DBC load before the engine's first "
                            "present (serial %llu)",
                            drew && s.presentedCount ? "presented" : "submitted (not presentable yet)",
                            static_cast<unsigned long long>(s.serial));
    }

    void GetFrameTimes(FrameTimes& window, FrameTimes& session, bool resetWindow)
    {
        window  = s.window;
        session = s.session;
        if (resetWindow) s.window = FrameTimes();
    }

    void Present(bool visible)
    {
        if (s.failed) return;

        // The time since the previous present, at the moment slot 38 hands the frame over (after the
        // engine's own WaitForFPSCap, before this frame's submit and acquire).
        LARGE_INTEGER now = {};
        if (!s.qpcFrequency.QuadPart) QueryPerformanceFrequency(&s.qpcFrequency);
        QueryPerformanceCounter(&now);
        if (s.lastPresent.QuadPart && s.qpcFrequency.QuadPart)
        {
            const double ms = static_cast<double>(now.QuadPart - s.lastPresent.QuadPart) * 1000.0
                              / static_cast<double>(s.qpcFrequency.QuadPart);
            NoteFrameTime(s.window, ms);
            NoteFrameTime(s.session, ms);
            wxlvk::native::perf::EndFrame(ms);   // the frame's CPU buckets, and a slow one's cause
        }
        s.lastPresent = now;

        if (!EnsureFrame()) return;

        // Until the engine has drawn once, a slowly cycling colour over the whole image (the
        // liveness colour): the login screen's own clear is black, which is also what a hung window
        // looks like.
        if (visible && !s.everDrew && !s.drewThisFrame && !s.target.colour)
        {
            if (VkCommandBuffer cmd = Pass())
            {
                const float t = static_cast<float>(s.presentedCount) * kColourStep;
                VkClearAttachment a = {};
                a.aspectMask      = VK_IMAGE_ASPECT_COLOR_BIT;
                a.colorAttachment = 0;
                a.clearValue.color.float32[0] = 0.5f + 0.5f * std::sin(t);
                a.clearValue.color.float32[1] = 0.5f + 0.5f * std::sin(t + 2.0944f);
                a.clearValue.color.float32[2] = 0.5f + 0.5f * std::sin(t + 4.1888f);
                a.clearValue.color.float32[3] = 1.0f;
                VkClearRect rect = {};
                rect.rect.extent  = s.passExtent;
                rect.layerCount   = 1;
                s.dapi->CmdClearAttachments(cmd, 1, &a, 1, &rect);
            }
        }

        FinishFrame(visible);
        // Empty memory blocks go back to the driver after a few seconds (vulkan/Memory.cpp).
        memory::Tick();

        if (s.rebuild && !s.failed)
        {
            // Out of date, suboptimal, or a WM_SIZE: rebuilt once, here, between frames.
            s.rebuild = false;
            if (s.dapi) s.dapi->DeviceWaitIdle(s.device);
            s.completed = s.submitted;
            DropImageObjects();
            context::Rebind();
        }
    }

    void RequestRebuild() { s.rebuild = true; }

    void SetRenderExtent(uint32_t width, uint32_t height)
    {
        if (s.renderWanted.width == width && s.renderWanted.height == height) return;
        s.renderWanted = VkExtent2D{ width, height };
        // Images made for the old size go at the next rebuild; slot 13 has already dropped them.
        if (s.imageCount) s.rebuild = true;
    }

    void SetGamma(const uint16_t* ramp, const uint16_t* desktop, bool fullscreen)
    {
        if (!ramp) return;
        // A ramp is only used when it is one: non-decreasing per channel and spanning something.
        auto usable = [](const uint16_t* r) {
            for (uint32_t c = 0; c < 3; ++c)
            {
                const uint16_t* ch = r + c * 256;
                if (ch[255] < ch[0] + 0x4000u) return false;
                for (uint32_t k = 1; k < 256; ++k)
                    if (ch[k] < ch[k - 1]) return false;
            }
            return true;
        };
        // The engine's own ramp: clicking "windowed" in fullscreen handed slot 15 an all-zero ramp
        // while the format was still fullscreen, and the present pass turned the screen black until
        // the next ramp. D3D never shows such a ramp (it applies none in windowed, and the switch
        // reaches its Reset first). An unusable engine ramp turns the LUT off, as identity.
        if (!usable(ramp))
        {
            static unsigned warnedRamp = 0;
            if (warnedRamp < 4)
            {
                ++warnedRamp;
                wxl::ScriptMgr::Log(WXL_LOG_WARN,
                                    "native: gamma: the engine's ramp is not a usable ramp (R[0]=0x%04X "
                                    "R[128]=0x%04X R[255]=0x%04X, fullscreen %d); the present pass "
                                    "applies none",
                                    ramp[0], ramp[128], ramp[255], fullscreen ? 1 : 0);
            }
            s.gamma = false;
            return;
        }
        // The desktop ramp is inverted only when it is one too. GetDeviceGammaRamp can fail or
        // return zeros (some drivers, HDR displays) and leave the device's copy all zero; inverting
        // that maps every entry to 0 and presented a black screen, cursor included, on the first
        // fullscreen run. Treated as identity instead.
        if (desktop)
        {
            const bool valid = usable(desktop);
            static bool warned = false;
            if (!valid && !warned)
            {
                warned = true;
                wxl::ScriptMgr::Log(WXL_LOG_WARN,
                                    "native: gamma: the desktop ramp is not a usable ramp (R[0]=0x%04X "
                                    "R[128]=0x%04X R[255]=0x%04X); treated as identity",
                                    desktop[0], desktop[128], desktop[255]);
            }
            if (!valid) desktop = nullptr;
        }
        // Entry i of channel c: the value v whose display output D(v) is closest to the engine's
        // R(i) -- D is the desktop ramp the display applies after us, inverted by a search over its
        // entries. With an identity desktop ramp that is R(i) >> 8.
        bool identity = true;
        for (uint32_t c = 0; c < 3; ++c)
            for (uint32_t i = 0; i < 256; ++i)
            {
                const uint32_t want = ramp[c * 256 + i];
                uint32_t       v    = want >> 8;
                if (desktop)
                {
                    const uint16_t* dch     = desktop + c * 256;
                    uint32_t        bestErr = 0xFFFFFFFFu;
                    for (uint32_t k = 0; k < 256; ++k)
                    {
                        const uint32_t err = dch[k] > want ? dch[k] - want : want - dch[k];
                        if (err < bestErr)
                        {
                            bestErr = err;
                            v       = k;
                        }
                    }
                }
                s.lut[i * 4 + c] = static_cast<uint8_t>(v);
                if (v != i) identity = false;
            }
        for (uint32_t i = 0; i < 256; ++i) s.lut[i * 4 + 3] = 0xFF;
        s.lutDirty = true;
        const bool was = s.gamma;
        s.gamma = fullscreen && !identity;
        static unsigned logged = 0;
        if (logged < 12 && (was != s.gamma || logged == 0))
        {
            ++logged;
            wxl::ScriptMgr::Log(WXL_LOG_DEBUG,
                                "native: gamma: ramp set (R[128]=0x%04X G[128]=0x%04X B[128]=0x%04X, desktop "
                                "R[128]=0x%04X); %s",
                                ramp[128], ramp[256 + 128], ramp[512 + 128], desktop ? desktop[128] : 0x8080,
                                s.gamma ? "applied in the present pass (fullscreen)"
                                        : (fullscreen ? "equal to the desktop ramp, nothing to apply"
                                                      : "windowed: not applied, as D3D (fullscreen gamma only)"));
        }
    }

    bool GammaActive() { return s.gamma; }

    VkExtent2D SwapExtent() { return s.swapExtent; }

    void ReleaseSwapchain()
    {
        if (!s.dapi || !s.device || !context::Live()) return;
        FinishFrame(false);
        s.dapi->DeviceWaitIdle(s.device);
        s.completed = s.submitted;
        DropImageObjects();
        s.rebuild = false;
    }

    uint64_t Count() { return s.presentedCount; }

    void GetStats(Stats& out)
    {
        out         = s.stats;
        out.ownBack = s.ownBack;
        out.queryPrecise = s.queryControl != 0;
    }
}
