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

#pragma once

#include "vulkan/Memory.hpp"
#include "vulkan/Texture.hpp"
#include "vulkan/Vk.hpp"

#include <cstdint>

// The frame renders into an OWN BACK BUFFER (the swapchain's format; the format's resolution in
// fullscreen, else the swapchain's extent; COLOR_ATTACHMENT | TRANSFER_SRC | TRANSFER_DST | SAMPLED),
// put onto the acquired swapchain image at present by a copy, or by the present pass when it is
// scaled or carries the gamma LUT. A surface without TRANSFER_DST usage takes the DIRECT path
// instead: it renders into the swapchain image, acquired at the first pass, and refuses the
// back-buffer copies and read-backs.
//
// Passes are begun on the CURRENT TARGET (SetTarget, from slot 23): the back buffer or a texture's
// attachment view, with the depth slot 23's rules pick (the back depth, the shared offscreen depth
// grown to the largest target, or a depth texture). Every pass is LOAD/STORE with its attachments in
// their attachment layouts; layouts are tracked per image (texture::Image::layout for textures,
// fields here for the back buffer and the depths) and every change is an explicit barrier: a texture
// target goes back to SHADER_READ_ONLY when its pass ends, so what samples it next finds it there.
// One render pass per PASS CLASS (colour format, depth format) is what pipelines are built against,
// kept for the device's life; framebuffers are cached by (pass, colour view, depth view, extent) and
// retired by serial with the views they name.
//
// The frame, in the order things happen:
//
//   - it OPENS lazily, at the first thing that needs a command buffer: a draw, a clear, a buffer or
//     texture transfer, or the present itself. Opening waits that frame slot's fence (two frames
//     in flight), retires what the frames it proves finished named, rewinds the slot's staging and
//     upload rings and pipelines' per-slot pools, and begins the slot's main command buffer;
//   - a pass is begun at the first draw or clear on a target; a first use of an attachment in the
//     frame is discarded and cleared (black, the far plane), a pass resumed after a break LOADs;
//   - a TRANSFER goes either into the slot's pre-pass command buffer (submitted ahead of the main
//     one, so before every draw of the frame) or INLINE: the pass is ended, the copy recorded with
//     its barriers, and the next draw resumes the pass with LOAD. Inline exactly when the
//     destination was named by a command recorded earlier in this same frame -- the TRANSFER RULE;
//     Outside() counts each break;
//   - slot 38 PRESENTS: ends the pass, acquires, records the copy or the present pass, submits
//     {pre, main} and presents. Until the first frame that ever had a draw, a frame shows a slowly
//     cycling colour, so a live loop can still be told from a hung one before the engine draws.
//
// Frame serials: every opened frame takes the next serial; a resource a draw names is stamped with
// it; a slot remembers the serial it last submitted, and once its fence has been waited that serial
// (and every earlier one: one queue, submitted in order) is Completed(). Retirement is by serial: an
// image or a buffer is destroyed once the last frame that named it has completed, never with a
// device wait.
//
// Everything runs on the render thread, the thread that calls the device slots. Nothing Vulkan runs
// at DLL detach: the state is plain handles and counters with a trivial destructor.
namespace wxlvk::native::present
{
    /**
     * Brings up the modules the draw path needs on the live context: memory, textures, the
     * hand-written shader modules, the pipelines. Idempotent; a failure is logged once and leaves
     * the frame presenting clears only.
     *
     * @return bool up : the draw path can record draws
     */
    bool InitDrawPath();

    /// Whether InitDrawPath() succeeded.
    bool DrawPathUp();

    /**
     * Slot 38's present: finishes the open frame (opening one if nothing did), submits it and,
     * when an image was acquired, presents it. `visible` false (a minimised or zero-size window)
     * still submits the frame -- its transfers must execute -- but acquires nothing.
     *
     * @param bool visible : the window can be presented into
     */
    void Present(bool visible);

    /// Slot 22 saw a size change: the swapchain is rebuilt once after the next present, rather
    /// than once per WM_SIZE of a drag.
    void RequestRebuild();

    // --- the render size and the present pass -----------------------------------------------------

    /**
     * The back buffer's size, independent of the swapchain's: fullscreen renders at the format's
     * resolution on a borderless window at the desktop's, and the present scales one onto the other.
     * 0 x 0 (windowed) makes the back buffer follow the swapchain. Read when the back buffer is
     * next made: slot 13 sets it between ReleaseSwapchain() and the new swapchain.
     */
    void SetRenderExtent(uint32_t width, uint32_t height);

    /**
     * The engine's gamma ramp (+0x354, 3 x 256 WORD) as the present pass's LUT, composed with the
     * inverse of the desktop ramp (+0x954) the display applies on top of whatever is presented. D3D
     * applies the ramp only in fullscreen (SetGammaRamp with FULLSCREENGAMMA), so `fullscreen` false
     * turns it off, as does a ramp equal to the desktop's.
     *
     * @param pointer ramp : 768 WORDs, R then G then B
     * @param pointer desktop : the desktop ramp saved at slot 10, same layout; null for identity
     * @param bool fullscreen : the format is fullscreen
     */
    void SetGamma(const uint16_t* ramp, const uint16_t* desktop, bool fullscreen);

    /// Whether the present pass applies a LUT now (for the log lines).
    bool GammaActive();

    /// The swapchain's extent, as the present writes it (0 x 0 without one).
    VkExtent2D SwapExtent();

    /**
     * Before the swapchain or the window goes away (slots 11 and 13): finishes and submits an open
     * frame, waits for the device, and destroys everything sized by or naming the swapchain. The
     * command buffers, the passes and the retire list survive: they belong to the device.
     */
    void ReleaseSwapchain();

    /// Frames presented so far.
    uint64_t Count();

    // --- the frame, for the native slots --------------------------------------------------------

    /// Opens the frame if none is open. False when there is no context or the frame failed hard.
    bool EnsureFrame();

    /**
     * The main command buffer INSIDE the render pass on this frame's swapchain image: acquires and
     * begins the pass as needed. Null when there is nothing to draw into this frame (no swapchain,
     * the acquire said out of date, a hard failure); the caller counts the draw as skipped.
     */
    VkCommandBuffer Pass();

    /// A clear the pass about to begin may fold into its load op (slot 39, Scene.cpp).
    struct PassClear
    {
        uint32_t    mask   = 0;      ///< bit 0 colour, bit 1 depth
        float       colour[4] = {};  ///< r g b a
        const void* device = nullptr;
        /// The clear's rectangle on a target of `extent` (Draw.cpp PlaceD3d); false when empty.
        bool (*place)(const void* device, const VkExtent2D& extent, VkRect2D& rect) = nullptr;
    };

    /**
     * Pass(), with a clear that may become the load op of the pass when this call begins it and
     * the clear covers the whole pass area: CLEAR instead of LOAD for the aspects it names (the
     * depth only when the format has no stencil). Same pixels as the clear recorded inside the
     * pass, one command fewer.
     *
     * @param PassClear& clear : the clear
     * @param uint32& folded : receives the aspects (mask bits) the load op took; the caller
     *        records the rest as before
     * @return VkCommandBuffer cmd : as Pass()
     */
    VkCommandBuffer PassClearing(const PassClear& clear, uint32_t& folded);

    /**
     * The main command buffer OUTSIDE any render pass, for an inline transfer: ends the pass if
     * one is open, which is a pass break, counted. Null without a frame.
     */
    VkCommandBuffer Outside();

    /// The frame's pre-pass transfer command buffer, begun on first use. Null without a frame.
    VkCommandBuffer PrePass();

    /// Whether a frame is being recorded.
    bool FrameOpen();

    namespace detail
    {
        /// What Serial() answers now: the open frame's serial, or the next frame's while none is
        /// open. Kept up to date at the frame's open and close so the draw path's stamps (every
        /// pool and every texture a draw resolves) are one load and no call.
        extern uint64_t serialNow;
    }

    /// The serial of the frame being recorded (or of the next one when none is open).
    inline uint64_t Serial() { return detail::serialNow; }

    /// Every frame with a serial at or below this one has finished on the GPU.
    uint64_t Completed();

    /// True when `lastUsed` is the frame being recorded: a transfer into it must be inline.
    bool UsedThisFrame(uint64_t lastUsed);

    /// The frame-in-flight slot being recorded (pipelines::BeginFrame / Record take it).
    uint32_t Slot();

    /// The render pass pipelines are built against: the open pass's class (one per colour / depth
    /// format pair). Valid after Pass().
    VkRenderPass RenderPass();

    /// The open pass's render area (the target's, clamped to its attachments); with no pass
    /// open, the back buffer's extent (0x0 before the swapchain exists).
    VkExtent2D Extent();

    /// Whether the open pass has a depth attachment.
    bool HasDepth();

    /**
     * The open pass's attachment formats.
     *
     * @param VkFormat& colour : receives the colour format (UNDEFINED with no pass open)
     * @param VkFormat& depth : receives the depth format (UNDEFINED without depth)
     */
    void PassFormats(VkFormat& colour, VkFormat& depth);

    /// A draw was recorded in this frame: the liveness colour is not shown any more.
    void NoteDraw();

    // --- the G-buffer (docs/architecture.md "The G-buffer") ----------------------------------------

    /// The engine's world scene pass begins (true) or ends (false): between the two, until AfterOpaqueWorld
    /// fires, the draws on the back buffer may take the G-buffer class (src/native/render/WorldPass.cpp).
    void NoteWorld(bool open);

    /// AfterOpaqueWorld fired: the opaque world is drawn; every later pass of the frame is a plain one.
    void NoteOpaqueEnd();

    /// A consumer asks for the G-buffer (`wanted`) or stops asking. Counted.
    void RequestGBuffer(bool wanted);

    /// The G-buffer's images, as wxl.vk.render's GetGBuffer hands them out.
    struct GBufferDesc
    {
        bool          valid  = false;   ///< the frame being recorded drew its opaque world into them, and that pass is over
        uint64_t      serial = 0;       ///< the frame they describe
        VkImageView   views[3] = {};    ///< albedo, normal, light
        VkFormat      formats[3] = {};
        VkImageLayout layout = VK_IMAGE_LAYOUT_GENERAL;
        VkExtent2D    extent = { 0, 0 };
    };

    /// The G-buffer's images; false when they do not exist (nobody asked, refused, or not made yet).
    bool GBufferInfo(GBufferDesc& out);

    /**
     * A span of the current slot's staging ring: host-visible, mapped, TRANSFER_SRC, alive until
     * the slot's fence is next waited. 16-byte aligned, which every copy source here needs.
     *
     * @param VkDeviceSize bytes : how many
     * @param VkBuffer& buffer : receives the buffer the span is in
     * @param VkDeviceSize& offset : receives the span's offset in it
     * @param void*& mapped : receives the CPU pointer to the span
     * @return bool ok : false without a frame or when no memory could be had (counted)
     */
    bool Stage(VkDeviceSize bytes, VkBuffer& buffer, VkDeviceSize& offset, void*& mapped);

    /**
     * A span of the current slot's UPLOAD ring (texture regions), as Stage() but from a ring of
     * its own, bounded (4 x 8 MiB per slot): when it is full, `full` says so and the caller flushes
     * with FlushUploads() and asks again -- one wait per 32 MiB of uploads in a loading frame instead
     * of one per region. False without `full`: no frame, a region larger than one chunk (8 MiB), or
     * no memory at all; the caller takes its synchronous path.
     */
    bool StageUpload(VkDeviceSize bytes, VkBuffer& buffer, VkDeviceSize& offset, void*& mapped, bool& full);

    /**
     * FlushAndWait(), then the slot's upload ring rewound (every span in it was consumed by the
     * submission just waited for). False when the flush could not be done (the direct path once an
     * image was acquired, or a failure): the caller falls back to its synchronous path.
     */
    bool FlushUploads();

    /// Hands an image over to be destroyed once the frame with serial `lastUsed` has completed.
    void RetireImage(wxlvk::vk::texture::Image& image, uint64_t lastUsed);

    /// Hands a buffer over to be destroyed once the frame with serial `lastUsed` has completed.
    /// Both retirements destroy at once when that frame is already known complete (a fence poll
    /// first), and `lastUsed` must cover every command that names the object -- a copy recorded
    /// into the open frame counts as much as a draw (the pools' and textures' `lastRef`).
    void RetireBuffer(wxlvk::vk::memory::Buffer& buffer, uint64_t lastUsed);

    /// Hands an image that is not a texture-module image over to be destroyed, with its view and
    /// its memory, once the frame with serial `lastUsed` has completed. Same rule as above.
    void RetireRaw(VkImage image, VkImageView view, const wxlvk::vk::memory::Block& block, uint64_t lastUsed);

    /**
     * Blocks until the frame with serial `serial` has completed, when it has been submitted: the
     * stream ring's last resort when every chunk is still read. False for the open frame,
     * which cannot be waited for, and on a wait failure.
     */
    bool WaitSerial(uint64_t serial);

    /// Polls the frame fences and destroys every retirement they prove finished. Safe at any
    /// moment (slot 12, DeviceEvictResources, on a zone unload).
    void Collect();

    /**
     * One black frame, right after the bring-up (slot 13): the engine's first present comes only
     * after the DBC load, about eight seconds of a window with nothing on it. Once; logged.
     */
    void PresentBlank();

    /// The time between two presents (frame pacing).
    struct FrameTimes
    {
        uint64_t frames;   ///< intervals measured
        double   minMs;
        double   maxMs;
        double   sumMs;
        uint64_t slow;     ///< intervals above 33 ms
        uint64_t hitches;  ///< intervals above 100 ms
    };

    /**
     * @param FrameTimes& window : receives the intervals since the last call with resetWindow
     * @param FrameTimes& session : receives every interval
     * @param bool resetWindow : start a new window
     */
    void GetFrameTimes(FrameTimes& window, FrameTimes& session, bool resetWindow);

    /// The frame's counters, for the stats lines.
    struct Stats
    {
        uint64_t frames;          ///< frames submitted
        uint64_t presented;       ///< of which presented
        uint64_t passBreaks;      ///< render passes ended for an inline transfer (the transfer rule's cost)
        uint64_t framesBroken;    ///< frames with at least one break
        uint32_t maxBreaks;       ///< the most breaks one frame had
        uint64_t noImage;         ///< draws / clears that asked for a pass in a frame with no image
        uint64_t stagingBytes;    ///< bytes handed out of the staging rings
        uint64_t stagingFailed;   ///< staging requests that could not be met
        uint32_t stagingChunks;   ///< chunks alive across both slots
        uint64_t retiredImages;   ///< images destroyed by serial
        uint64_t retiredBuffers;  ///< buffers destroyed by serial
        uint64_t retireLeaked;    ///< retirements the full list could not take (leaked, never unsafe)
        uint64_t retiredAtOnce;   ///< of the two above, destroyed at the retirement (their frame had completed)
        uint32_t retirePeak;      ///< the longest the waiting list has been
        uint64_t serialWaits;     ///< WaitSerial() calls that blocked on a fence
        bool     ownBack;         ///< the frame renders into its own back buffer (else the direct path)
        uint64_t passes;          ///< render passes begun
        uint64_t targetPasses;    ///< of which on a texture target
        uint64_t presentCopies;   ///< back buffer -> swapchain copies
        uint64_t noTarget;        ///< pass requests while the engine's target had no attachment (dropped)
        uint64_t flushes;         ///< mid-frame submit-and-wait (read-back, a query waited for in its own frame)
        uint64_t flushRefused;    ///< flushes the direct path could not do (an image was acquired)
        uint32_t passClasses;     ///< distinct (colour, depth) render passes
        uint32_t framebuffers;    ///< live in the cache
        uint64_t framebuffersMade;
        uint64_t offscreenGrows;  ///< the shared offscreen depth re-made larger
        uint64_t queriesBegun;
        uint64_t querySplits;     ///< extra Vulkan queries because a pass ended inside an engine query
        uint64_t queryOverflow;   ///< a query ran out of indices (its count is a lower bound)
        uint64_t queryOpenAtPresent; ///< engine queries still open at a present (ended there)
        uint64_t queryNested;     ///< a query begun while another was open (the first was ended)
        bool     queryPrecise;    ///< the Vulkan queries are begun with VK_QUERY_CONTROL_PRECISE_BIT
        uint64_t hiddenFrames;       ///< frames submitted with nothing acquired: the window minimised or zero-size
        uint64_t presentPasses;      ///< presents through the present pass (scale and / or gamma) instead of the copy
        uint64_t scaledPresents;     ///< of which with the back buffer a different size than the swapchain
        uint64_t gammaPresents;      ///< of which with the gamma LUT applied
        uint64_t lutUploads;         ///< gamma LUT uploads
        uint64_t presentPassRefused; ///< presents that wanted the pass and fell back to a (cropping) copy
        uint64_t uploadBytes;        ///< bytes handed out of the upload rings
        uint64_t uploadFlushes;      ///< FlushUploads(): the upload ring of a frame was full
        uint64_t uploadOversize;     ///< upload regions larger than a ring chunk (synchronous path)
        uint32_t uploadChunks;       ///< upload ring chunks alive across both slots
        uint32_t uploadChunksPeak;   ///< the most there have been
        uint64_t breaksResumed;      ///< of passBreaks, followed by a pass on the same colour view (the break's real cost: one more load)
        uint64_t clearLoadPasses;    ///< passes begun with a CLEAR load op (a first-use or the engine's full clear folded in)
        uint64_t firstUseFolded;     ///< first-use clears folded into the load op (per aspect)
        uint64_t engineClearsFolded; ///< slot-39 clears of the whole pass area folded into the load op (per aspect)
        uint64_t barrierCalls;       ///< vkCmdPipelineBarrier calls for layout changes
        uint64_t barriers;           ///< image barriers they carried
        uint64_t gbufferPasses;      ///< passes begun on the G-buffer class (the opaque world's, while a consumer asks)
        uint64_t gbufferSwitches;    ///< passes ended because the class a draw wanted changed (into or out of the G-buffer class)
    };
    void GetStats(Stats& out);

    // --- targets, copies, read-back, queries -----------------------------------------------------

    /// What passes draw into, as src/native/Targets.cpp resolves the engine's target table.
    struct TargetDesc
    {
        enum class Depth : uint8_t
        {
            Back,       ///< the back buffer's depth
            Offscreen,  ///< the shared offscreen depth, grown to the target (D3D's +0x3B38)
            Texture,    ///< a depth texture (`depthImage`)
            None,
        };

        wxlvk::vk::texture::Image* colour      = nullptr;        ///< null: the back buffer
        VkImageView                colourView  = VK_NULL_HANDLE; ///< its attachment view (level 0, one layer)
        uint64_t*                  colourStamp = nullptr;        ///< receives the serial of each frame a pass on it runs in
        Depth                      depth       = Depth::Back;
        wxlvk::vk::texture::Image* depthImage  = nullptr;        ///< Depth::Texture
        VkImageView                depthView   = VK_NULL_HANDLE;
        uint64_t*                  depthStamp  = nullptr;
        VkExtent2D                 extent      = { 0, 0 };       ///< the engine's curWindow; 0x0: the colour's own
        bool                       invalid     = false;          ///< a target with no attachment: draws and clears are dropped
    };

    /// Makes `t` the current target; ends a pass open on another one (not counted as a break).
    void SetTarget(const TargetDesc& t);

    /// The current target.
    const TargetDesc& Target();

    /// The open pass (or the current target, with none open) draws into the back buffer.
    bool OnBackBuffer();

    /// A texture image is about to go away: a pass on it is ended, and a current target naming it
    /// becomes invalid until the next SetTarget.
    void ForgetImage(const wxlvk::vk::texture::Image* image);

    /// Retires, by serial, every cached framebuffer naming `view` (the view itself is the caller's).
    void ForgetView(VkImageView view, uint64_t lastUsed);

    /// ForgetView, then the view itself retired by serial.
    void RetireView(VkImageView view, uint64_t lastUsed);

    /// The format depth textures (EGxTexFormat 12) are made in: the back depth's when it can be
    /// sampled, else D16; UNDEFINED when neither can be.
    VkFormat DepthTextureFormat();

    /// Whether the back buffer can be copied from (the own back buffer exists).
    bool BackBufferReadable();

    /**
     * Moves the back buffer to `layout` (outside a pass: call Outside() first) and hands its image.
     *
     * @return bool ok : false on the direct path or before the back buffer exists
     */
    bool BackBuffer(VkCommandBuffer cmd, VkImageLayout layout, VkImage& image, VkFormat& format,
                    VkExtent2D& extent);

    /**
     * The own back buffer's format and extent, without touching its layout.
     *
     * @return bool ok : false on the direct path, before the back buffer exists, or in a frame
     *         that has no image
     */
    bool BackBufferInfo(VkFormat& format, VkExtent2D& extent);

    /**
     * Moves the back depth to `layout` (outside a pass: call Outside() first; a pass open on it is
     * ended) and hands its view, for an effect that reads the scene's depth.
     *
     * @return bool ok : false when the depth format cannot be sampled, or the depth holds nothing
     *         of this frame yet
     */
    bool SceneDepth(VkCommandBuffer cmd, VkImageLayout layout, VkImageView& view, VkFormat& format);

    /// A tracked texture image to `to` (whole image), outside a pass. A no-op when it is there.
    void Transition(VkCommandBuffer cmd, wxlvk::vk::texture::Image& image, VkImageLayout to);

    /**
     * Submits the frame recorded so far and waits for it, then reopens the main command buffer on the
     * same frame (same serial, same slot): what a D3D read-back's implicit flush is. A pass break.
     *
     * @return bool ok : false on the direct path once an image was acquired, or on a failure
     */
    bool FlushAndWait();

    /// The Vulkan side of one engine occlusion query (slots 62-67).
    constexpr uint32_t kQuerySplits = 8;
    struct QueryRun
    {
        uint64_t serial    = 0;      ///< frame it was begun in; 0 = never begun
        uint32_t slot      = 0;      ///< frame-in-flight slot whose pool holds its indices
        uint32_t index[kQuerySplits] = {};
        uint32_t count     = 0;      ///< Vulkan queries it spans (one per pass it ran through)
        bool     active    = false;  ///< between 64 and 65
        bool     inPass    = false;  ///< a Vulkan query of it is open in the current pass
        bool     harvested = false;  ///< `samples` is final
        bool     overflow  = false;  ///< ran out of indices: `samples` is a lower bound
        bool     registered = false; ///< in its slot's harvest list
        uint64_t samples   = 0;
        uint32_t draws     = 0;      ///< slot-42 draws recorded while it was active (NoteQueryDraw)
        uint32_t drawn     = 0;      ///< of those, issued by the pipelines (not skipped)
    };

    /// 64: begins `run` in the current frame (inside the pass now, or at the next pass begin).
    bool QueryStart(QueryRun& run);
    /// 65.
    void QueryStop(QueryRun& run);
    /// 63: forgets `run` (stops it, takes it off the harvest list).
    void QueryForget(QueryRun& run);
    /// 66 param 1: whether `run`'s result is known; harvests it when its frame has completed.
    bool QueryReady(QueryRun& run);
    /// 67: the result, waiting (or flushing the open frame) as D3D's spin does.
    bool QueryWait(QueryRun& run);
    /// Whether an engine occlusion query is between 64 and 65 (the draws it counts are noted).
    bool QueryActive();
    /// A draw inside the open engine query reached the pipelines: issued (`drawn`) or skipped.
    void NoteQueryDraw(bool drawn);
    /// Whether the open engine query has a Vulkan query open in the current pass.
    bool QueryInPass();
    /// The open pass's depth attachment, for the query lines: 0 none (or no pass), 1 the back
    /// depth, 2 a depth texture, 3 the shared offscreen depth; `format` receives its format.
    uint32_t PassDepthKind(VkFormat& format);

    // --- the frame's GPU times (src/native/present/FrameGpu.cpp, 2026-10-11) -------------------------------------

    /// The device's own shared work, timed in lanes beside the frame (wxl.vk.render GetFrameStats).
    enum GpuLane : uint32_t
    {
        kGpuDepthCopy = 0,   ///< the opaque depth copy (src/native/render/DepthHistory.cpp)
        kGpuPyramid,         ///< the depth pyramid (src/native/render/DepthPyramid.cpp)
        kGpuLaneCount
    };

    /// A lane's two ends, outside any pass, in the frame being recorded (both at BOTTOM_OF_PIPE: the span leaves out
    /// earlier work still draining). Nothing without timestamps on the queue.
    void GpuLaneBegin(VkCommandBuffer cmd, uint32_t lane);
    void GpuLaneEnd(VkCommandBuffer cmd, uint32_t lane);

    /// The harvested GPU figures, in milliseconds, over the last kGpuWindow frames (a lane's average counts only the
    /// frames that ran it). timed false: the queue has no timestamps.
    constexpr uint32_t kGpuWindow = 128;
    struct GpuTimes
    {
        bool   timed      = false;
        double frameLast  = -1.0;
        double frameAvg   = -1.0;
        double frameMax   = -1.0;
        double frameSessionAvg = -1.0;   ///< every harvested frame of the session
        double frameSessionMax = -1.0;
        uint64_t frameSessionCount = 0;
        double laneAvg[kGpuLaneCount] = { -1.0, -1.0 };
        double laneLast[kGpuLaneCount] = { -1.0, -1.0 };
        uint32_t passBreaks = 0;   ///< the latest submitted frame's
    };
    void GetGpuTimes(GpuTimes& out);
}
