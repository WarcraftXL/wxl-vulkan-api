// The present path's shared internals: its constants, its state and the helpers every part of
// src/native/present/ uses.
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


#include "native/Present.hpp"

#include "host/Host.hpp"
#include "common/Settings.hpp"
#include "native/Perf.hpp"
#include "vulkan/Context.hpp"
#include "vulkan/Loader.hpp"
#include "vulkan/Memory.hpp"
#include "vulkan/Pipelines.hpp"
#include "vulkan/Shaders.hpp"
#include "vulkan/Texture.hpp"
#include "vulkan/Vk.hpp"

#include "wxl/Script.hpp"

#include <windows.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <new>

// The present path lives in src/native/present/, one file per responsibility, over this one header.
// Nothing here is visible outside it: the public face is src/native/Present.hpp.
//
//   State        the state's definition, the frame times, the retirement room
//   Barriers     formats and image barriers
//   Retire       retirement by frame serial, the staging and upload rings
//   Objects      the device's own objects (pools, command buffers, fences, semaphores)
//   Passes       the render pass classes, the clearing passes, the framebuffers
//   Images       the swapchain-derived objects, the acquire
//   Resolve      resolving the current target into the pass's sides
//   PassQueries  occlusion queries inside the engine's pass
//   Pass         opening and closing the engine's pass, the pre-pass, the present copy
//   PresentPass  the present pass and finishing a frame
//   ReadBack     targets, copies and read-backs (public)
//   QueryApi     the occlusion query API (public)
namespace wxlvk::native::present::detail
{
    namespace context   = wxlvk::vk::context;
    namespace loader    = wxlvk::vk::loader;
    namespace memory    = wxlvk::vk::memory;
    namespace texture   = wxlvk::vk::texture;
    namespace pipelines = wxlvk::vk::pipelines;
    namespace present   = wxlvk::native::present;

    using wxlvk::vk::ResultName;
    using present::TargetDesc;
    using present::QueryRun;

    /// Two frames in flight, as pipelines::kFrameSlots: the CPU records frame N+1 while the GPU runs
    /// frame N, and no more, because each extra one is input latency.
    constexpr uint32_t kFramesInFlight = 2;
    static_assert(kFramesInFlight == pipelines::kFrameSlots, "pipelines keeps one pool per frame slot");

    /// A swapchain with more images than this is refused rather than half-tracked.
    constexpr uint32_t kMaxImages = 8;

    /// Radians per presented frame for the liveness colour (Present()).
    constexpr float kColourStep = 0.01f;

    /// The staging ring of one slot: chunks of this size, a larger request gets a chunk of its own.
    /// 4 MiB is a few glyph pages or a modest static vertex pool; the login screen's uploads fit
    /// one chunk a frame, a zone load grows the list and the trim below gives it back.
    constexpr VkDeviceSize kChunkBytes = 4ull * 1024 * 1024;
    constexpr uint32_t     kMaxChunks  = 32;
    /// Chunks a slot keeps through a frame that did not use them; the rest are destroyed then
    /// (safely: the slot's fence was just waited), so one loading frame does not pin its peak.
    constexpr uint32_t     kKeepChunks = 2;
    constexpr VkDeviceSize kStageAlign = 16;

    /// The UPLOAD ring of one slot, separate from the staging ring above: texture regions go
    /// here (StageUpload), so that when it is full the frame can be flushed and the ring rewound
    /// (FlushUploads) without touching a span a static pool lock still holds in the other ring.
    /// Bounded on purpose -- every chunk is mapped, and address space is this process's scarce
    /// resource: 4 x 8 MiB per slot (64 MiB for both), one kept through an idle frame. A loading
    /// frame that uploads more than 32 MiB flushes once per 32 MiB instead of once per region.
    constexpr VkDeviceSize kUpChunkBytes = 8ull * 1024 * 1024;
    constexpr uint32_t     kMaxUpChunks  = 4;
    constexpr uint32_t     kKeepUpChunks = 1;

    /// Retired objects waiting for their serial: a heap list that starts at kRetiredFirst entries
    /// and doubles up to kMaxRetired. A zone change destroys hundreds of pools and thousands of
    /// textures between two presents (each is held until its last frame completed), and
    /// most of them were last drawn frames ago, so they never reach the list at all (RetireImage
    /// destroys at once what a completed frame last named). Full means leaked (counted), never
    /// destroyed early.
    constexpr uint32_t kRetiredFirst = 4096;
    constexpr uint32_t kMaxRetired   = 65536;

    /// Frame times above these are counted: 33 ms is a frame under 30 fps, 100 ms a hitch.
    constexpr double kSlowFrameMs  = 33.0;
    constexpr double kHitchFrameMs = 100.0;

    /// Render passes by (colour, depth) format, framebuffers by attachments, occlusion queries per
    /// frame slot. A session's targets are a handful (7 seen in the world), so the caches are linear
    /// arrays.
    constexpr uint32_t kMaxPassClasses  = 12;   ///< the G-buffer class doubles the back buffer's
    static_assert(kMaxPassClasses == 12, "vulkan/pipelines/Internal.hpp mirrors this count in its passInfo[12]: change both");
    constexpr uint32_t kMaxFramebuffers = 128;
    constexpr uint32_t kQueriesPerSlot  = 64;
    constexpr uint32_t kMaxRunsPerSlot  = 64;

    /// The G-buffer (docs/architecture.md "The G-buffer"): three colour attachments after the colour on
    /// the opaque world's pass class -- the albedo, the view normal and its class, the engine's light term.
    constexpr uint32_t kGBufferAttachments  = 3;
    constexpr VkFormat kGBufferAlbedoFormat = VK_FORMAT_R8G8B8A8_UNORM;
    constexpr VkFormat kGBufferNormalFormat = VK_FORMAT_R16G16B16A16_SNORM;   ///< R16G16B16A16_SFLOAT when SNORM cannot be attached
    constexpr VkFormat kGBufferLightFormat  = VK_FORMAT_R8G8B8A8_UNORM;

    struct Chunk
    {
        memory::Buffer buffer;
        VkDeviceSize   used = 0;
    };

    struct FrameSlot
    {
        VkCommandBuffer main  = nullptr;
        VkCommandBuffer pre   = nullptr;
        VkSemaphore     imageAvailable = VK_NULL_HANDLE;
        VkFence         fence = VK_NULL_HANDLE;
        uint64_t        serial = 0;     ///< the frame this slot last submitted
        Chunk           chunks[kMaxChunks];
        uint32_t        chunkCount = 0;
        uint32_t        chunkCursor = 0;
        // The upload ring (kUpChunkBytes chunks).
        Chunk           up[kMaxUpChunks];
        uint32_t        upCount  = 0;
        uint32_t        upCursor = 0;
    };

    /// What a retirement holds: texture images, buffers, framebuffers, views (a target's attachment
    /// views) and raw images (the offscreen depth, which is not a texture-module image).
    enum class Kind : uint8_t
    {
        Image,
        Buffer,
        Framebuffer,
        View,
        Raw,
    };

    struct Retired
    {
        Kind           kind = Kind::Image;
        texture::Image img;
        memory::Buffer buf;
        VkFramebuffer  fb   = VK_NULL_HANDLE;
        VkImageView    view = VK_NULL_HANDLE;   ///< Kind::View, and Kind::Raw's view
        VkImage        raw  = VK_NULL_HANDLE;
        memory::Block  rawBlock;
        uint64_t       serial = 0;
    };

    struct PassClass
    {
        VkFormat     colour = VK_FORMAT_UNDEFINED;
        VkFormat     depth  = VK_FORMAT_UNDEFINED;
        bool         gbuffer = false;           ///< with the G-buffer's three attachments after the colour
        VkRenderPass pass   = VK_NULL_HANDLE;   ///< LOAD / LOAD: what pipelines and framebuffers are made against
        /// The same pass with CLEAR load ops, by the bits of what is cleared: 1 the colour, 2 the depth,
        /// 4 the G-buffer's attachments ([0] is `pass`). Load ops take no part in render-pass
        /// compatibility, so the class's pipelines and framebuffers serve these unchanged. Made on first use.
        VkRenderPass clearing[8] = {};
    };

    struct Framebuffer
    {
        VkRenderPass  pass   = VK_NULL_HANDLE;
        VkImageView   colour = VK_NULL_HANDLE;
        VkImageView   depth  = VK_NULL_HANDLE;
        VkExtent2D    extent = { 0, 0 };
        VkFramebuffer fb     = VK_NULL_HANDLE;
        uint64_t      lastUsed = 0;
    };

    /// One attachment of a pass, resolved: where its layout is tracked, and what the pass's end does.
    struct Side
    {
        VkImage            image  = VK_NULL_HANDLE;
        VkImageView        view   = VK_NULL_HANDLE;
        VkFormat           format = VK_FORMAT_UNDEFINED;
        VkExtent2D         extent = { 0, 0 };
        VkImageLayout*     layout = nullptr;
        uint32_t           levels = 1;
        uint32_t           layers = 1;
        VkImageAspectFlags aspect = VK_IMAGE_ASPECT_COLOR_BIT;
        texture::Image*    tex    = nullptr;   ///< a texture: back to SHADER_READ_ONLY at the pass's end
        uint64_t*          stamp  = nullptr;
        bool               discard = false;    ///< its contents are not kept (first use)
        bool               clear   = false;    ///< and are cleared after the pass begins
        bool               back    = false;    ///< the back buffer or the back depth
    };

    struct State
    {
        VkDevice              device = nullptr;
        const loader::Device* dapi   = nullptr;
        VkCommandPool         pool   = VK_NULL_HANDLE;
        FrameSlot             slots[kFramesInFlight];

        VkFormat colourFormat = VK_FORMAT_UNDEFINED;   ///< the swapchain's, which the back buffer has
        VkFormat depthFormat  = VK_FORMAT_UNDEFINED;

        // --- passes and framebuffers -------------------------------------------------------------------
        PassClass   classes[kMaxPassClasses];
        uint32_t    classCount = 0;
        Framebuffer fbs[kMaxFramebuffers];
        uint32_t    fbCount = 0;

        // --- swapchain-derived --------------------------------------------------------------------
        uint32_t      swapGen    = UINT32_MAX;
        uint32_t      imageCount = 0;
        VkExtent2D    extent     = { 0, 0 };
        VkImage       images[kMaxImages]         = {};
        VkImageView   views[kMaxImages]          = {};   ///< the direct path's attachment views, and the present pass's
        VkSemaphore   renderFinished[kMaxImages] = {};
        VkFence       imagesInFlight[kMaxImages] = {};
        VkImage        depthImage = VK_NULL_HANDLE;
        VkImageView    depthView  = VK_NULL_HANDLE;
        memory::Block  depthBlock;
        VkImageLayout  depthLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        bool           depthSampled = false;   ///< the back depth was made SAMPLED (SceneDepth)
        // The own back buffer.
        bool           ownBack     = false;
        bool           modeLogged  = false;
        VkImage        backImage   = VK_NULL_HANDLE;
        VkImageView    backView    = VK_NULL_HANDLE;
        memory::Block  backBlock;
        VkImageLayout  backLayout  = VK_IMAGE_LAYOUT_UNDEFINED;
        VkImageLayout  swapLayout  = VK_IMAGE_LAYOUT_UNDEFINED;   ///< the direct path: the acquired image's

        // --- the G-buffer: three colour attachments beside the back buffer (docs/architecture.md) -------
        uint32_t       gbufRequests = 0;        ///< consumers asking for it (RequestGBuffer)
        bool           gbufFailed   = false;    ///< its images could not be made: refused for the session
        VkImage        gbufImage[kGBufferAttachments] = {};   ///< albedo, normal, light
        VkImageView    gbufView[kGBufferAttachments]  = {};
        memory::Block  gbufBlock[kGBufferAttachments];
        VkFormat       gbufFormat[kGBufferAttachments] = { kGBufferAlbedoFormat, kGBufferNormalFormat, kGBufferLightFormat };
        VkImageLayout  gbufLayout   = VK_IMAGE_LAYOUT_UNDEFINED;   ///< UNDEFINED until their first pass, GENERAL from then on
        bool           worldOpen    = false;    ///< inside the engine's world scene pass (NoteWorld)
        bool           opaqueDone   = false;    ///< AfterOpaqueWorld fired this frame (NoteOpaqueEnd)
        bool           gbufFresh    = true;     ///< no G-buffer pass yet this frame: the first one clears the attachments
        bool           gbufDrawn    = false;    ///< a G-buffer pass ran this frame
        uint64_t       gbufSerial   = 0;        ///< the frame the attachments describe

        // --- the shared offscreen depth (colour targets without a depth texture) -----------------------
        VkImage        offImage  = VK_NULL_HANDLE;
        VkImageView    offView   = VK_NULL_HANDLE;
        memory::Block  offBlock;
        VkExtent2D     offExtent = { 0, 0 };
        VkImageLayout  offLayout = VK_IMAGE_LAYOUT_UNDEFINED;

        // --- the current target and the open pass ------------------------------------------------------
        TargetDesc   target;
        Side         passColour;
        Side         passDepth;
        bool         passHasDepth = false;
        bool         passGBuffer  = false;   ///< the open pass is of the G-buffer class
        VkRenderPass passClass    = VK_NULL_HANDLE;
        VkExtent2D   passExtent   = { 0, 0 };

        // --- queries -------------------------------------------------------------------------------------
        VkQueryPool queryPools[kFramesInFlight] = {};
        uint32_t    queryNext[kFramesInFlight]  = {};
        QueryRun*   runs[kFramesInFlight][kMaxRunsPerSlot] = {};
        uint32_t    runCount[kFramesInFlight] = {};
        QueryRun*   activeRun = nullptr;
        VkQueryControlFlags queryControl = 0; ///< PRECISE when the device has occlusionQueryPrecise (D3D's exact count)

        // --- the frame being recorded ---------------------------------------------------------------
        bool     open        = false;
        uint32_t fi          = 0;
        uint64_t serial      = 0;      ///< of the open frame; the last one opened otherwise
        uint64_t completed   = 0;
        uint64_t submitted   = 0;      ///< the last serial submitted
        bool     preBegun    = false;
        bool     passOpen    = false;
        bool     acquired    = false;
        bool     noImage     = false;  ///< the back buffer could not be had this frame: no back pass until the next
        bool     depthFresh  = false;  ///< the back depth is unused this frame: discarded and cleared at first use
        bool     backTouched = false;  ///< the back buffer or its depth was named this frame
        uint32_t image       = 0;
        bool     drewThisFrame = false;
        bool     everDrew    = false;
        uint32_t breaksThisFrame = 0;
        /// The colour view of the pass the last break ended, until the next pass begins: a pass
        /// begun on the same view is a resumed one (Stats::breaksResumed), on another a target
        /// change the engine made anyway.
        VkImageView breakView = VK_NULL_HANDLE;
        bool     rebuild     = false;

        bool     drawPath    = false;
        bool     drawTried   = false;
        bool     failed      = false;
        uint64_t presentedCount = 0;

        Retired* retired      = nullptr;   ///< heap, retiredCap entries
        uint32_t retiredCount = 0;
        uint32_t retiredCap   = 0;

        bool     blankPresented = false;   ///< the one black frame after the bring-up was shown

        // --- frame pacing: the time between two slot-38 presents -----------------------------------
        LARGE_INTEGER qpcFrequency = {};
        LARGE_INTEGER lastPresent  = {};
        wxlvk::native::present::FrameTimes window  = {};   ///< since the last stats line
        wxlvk::native::present::FrameTimes session = {};

        // --- the render size and the present pass (scale, gamma) -------------------------------------
        VkExtent2D            renderWanted = { 0, 0 };   ///< 0x0: the back buffer follows the swapchain
        VkExtent2D            swapExtent   = { 0, 0 };   ///< the swapchain's, which the present writes
        VkRenderPass          ppPass    = VK_NULL_HANDLE; ///< one colour attachment, the swapchain format
        VkFormat              ppFormat  = VK_FORMAT_UNDEFINED;
        VkFramebuffer         ppFbs[kMaxImages] = {};
        VkDescriptorSetLayout ppDsl     = VK_NULL_HANDLE;
        VkPipelineLayout      ppLayout  = VK_NULL_HANDLE;
        VkPipeline            ppPipe    = VK_NULL_HANDLE;
        VkDescriptorPool      ppPool    = VK_NULL_HANDLE;
        VkDescriptorSet       ppSet     = VK_NULL_HANDLE;
        VkSampler             ppSampler = VK_NULL_HANDLE;
        VkImageView           ppSetBack = VK_NULL_HANDLE; ///< the back view ppSet was last written with
        bool                  ppFailed  = false;          ///< the pass could not be built: copies only
        VkImage               lutImage  = VK_NULL_HANDLE; ///< 256 x 1 RGBA8, entry i = what value i becomes
        VkImageView           lutView   = VK_NULL_HANDLE;
        memory::Block         lutBlock;
        VkImageLayout         lutLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        uint8_t               lut[256 * 4] = {};
        bool                  lutDirty  = true;
        bool                  gamma     = false;           ///< the LUT is applied at present

        wxlvk::native::present::Stats stats = {};
    };
    extern State s;


    /// The pipeline stage and access a layout is used with (Barriers.cpp).
    void Scope(VkImageLayout l, VkPipelineStageFlags& stage, VkAccessFlags& access);

    /**
     * One layout change. The source scope is the one `from` implies even when `discard` makes the
     * old layout UNDEFINED: an image dropped and re-used (the back depth every frame, shared by both
     * frames in flight) still has to wait for its last writer.
     */
    /// Layout changes collected and recorded as one vkCmdPipelineBarrier: the colour and depth
    /// attachments of a pass change together at its begin and its end. The stage masks are the
    /// union of the members' (a member waits on no less than it would alone).
    struct Barriers
    {
        VkImageMemoryBarrier b[2 + kGBufferAttachments] = {};
        uint32_t             count    = 0;
        VkPipelineStageFlags srcStage = 0;
        VkPipelineStageFlags dstStage = 0;

        /**
         * One layout change. The source scope is the one `from` implies even when `discard` makes
         * the old layout UNDEFINED: an image dropped and re-used (the back depth every frame, shared
         * by both frames in flight) still has to wait for its last writer.
         */
        void Add(VkImage image, VkImageAspectFlags aspect, uint32_t levels, uint32_t layers, VkImageLayout from,
                 VkImageLayout to, bool discard)
        {
            VkPipelineStageFlags src = 0, dst = 0;
            VkAccessFlags        srcAccess = 0, dstAccess = 0;
            Scope(from, src, srcAccess);
            Scope(to, dst, dstAccess);
            if (from == VK_IMAGE_LAYOUT_UNDEFINED)
            {
                // Nothing tracked to wait for, but an attachment image may still be written by the
                // other frame in flight or read by a copy: the attachment and transfer stages,
                // conservatively.
                src       = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT |
                            VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT;
                srcAccess = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT |
                            VK_ACCESS_TRANSFER_WRITE_BIT;
            }
            VkImageMemoryBarrier& m = b[count++];
            m = VkImageMemoryBarrier{};
            m.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            m.srcAccessMask       = srcAccess;
            m.dstAccessMask       = dstAccess;
            m.oldLayout           = discard ? VK_IMAGE_LAYOUT_UNDEFINED : from;
            m.newLayout           = to;
            m.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            m.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            m.image               = image;
            m.subresourceRange.aspectMask = aspect;
            m.subresourceRange.levelCount = levels;
            m.subresourceRange.layerCount = layers;
            srcStage |= src;
            dstStage |= dst;
        }

        void Record(VkCommandBuffer cmd)
        {
            if (!count) return;
            s.dapi->CmdPipelineBarrier(cmd, srcStage, dstStage, 0, 0, nullptr, 0, nullptr, count, b);
            ++s.stats.barrierCalls;
            s.stats.barriers += count;
            count = 0;
            srcStage = dstStage = 0;
        }
    };

    // State.cpp
    void NoteFrameTime(wxlvk::native::present::FrameTimes& t, double ms);
    void Poll();
    bool RetireRoom();
    void Fail(const char* what, VkResult r);

    // Barriers.cpp
    bool IsDepth(VkFormat f);
    VkImageAspectFlags AspectOf(VkFormat f);
    void Scope(VkImageLayout l, VkPipelineStageFlags& stage, VkAccessFlags& access);
    void Barrier(VkCommandBuffer cmd, VkImage image, VkImageAspectFlags aspect, uint32_t levels, uint32_t layers, VkImageLayout from, VkImageLayout to, bool discard);

    // Retire.cpp
    void DestroyRetired(Retired& e);
    void CollectRetired();
    void Retire(Retired& e);
    void ResetUploads(FrameSlot& slot);
    void FlushRings(FrameSlot& slot);
    void ResetStaging(FrameSlot& slot);

    // Objects.cpp
    bool EnsureDeviceObjects(const context::Handles& h);

    // Passes.cpp
    VkFormat PickDepth(const context::Handles& h);
    void EnsureDepthFormat();
    VkRenderPass MakePass(VkFormat colour, VkFormat depth, bool clearColour = false, bool clearDepth = false, bool gbuffer = false, bool clearGBuffer = false);
    VkRenderPass ClassPass(VkFormat colour, VkFormat depth, bool gbuffer);
    bool IsGBufferPass(VkRenderPass pass);
    VkRenderPass ClearingPass(VkRenderPass base, uint32_t which, bool& clearing);
    void RemoveFramebuffer(uint32_t i, uint64_t lastUsed, bool now);
    void DropFramebuffersNaming(VkImageView view, uint64_t lastUsed, bool now);
    VkFramebuffer FramebufferFor(VkRenderPass pass, VkImageView colour, VkImageView depth, VkExtent2D extent);

    // Images.cpp
    bool MakeRaw(VkFormat format, VkExtent2D extent, VkImageUsageFlags usage, VkImage& image, VkImageView& view, memory::Block& block, const char* what);
    void DropImageObjects();
    bool EnsureImageObjects(const context::Handles& h);
    bool EnsureGBufferImages();
    void DropGBufferImages();
    bool Acquire();

    // Resolve.cpp
    bool BackColour(Side& out);
    bool BackDepth(Side& out);
    bool Offscreen(Side& out, VkExtent2D need);
    void TextureSide(texture::Image* img, VkImageView view, uint64_t* stamp, Side& out);
    VkExtent2D Min(VkExtent2D a, VkExtent2D b);

    // PassQueries.cpp
    void QueryBeginVk(QueryRun& run);
    void QueryEndVk(QueryRun& run);
    void Unregister(QueryRun& run);
    bool Harvest(QueryRun& run, bool wait);

    // Pass.cpp
    bool GBufferWanted();
    void EndPass();
    VkCommandBuffer BeginPass(const present::PassClear* pending = nullptr, uint32_t* folded = nullptr);
    void EndPrePass(FrameSlot& slot);
    void RecordPresentCopy(VkCommandBuffer cmd);

    // FrameGpu.cpp: the frame's GPU timestamps (wxl.vk.render GetFrameStats).
    /// At the frame's open, on its main buffer after its query reset: the slot's last results read, its pool reset, the
    /// frame's first timestamp.
    void FrameGpuOpen(VkCommandBuffer main, uint32_t slot, uint64_t serial);
    /// The pre-pass begun: its own start timestamp (the pre-pass runs first).
    void FrameGpuPrePass(VkCommandBuffer pre, uint32_t slot);
    /// After the frame's last pass, before the present copy: the frame's last timestamp.
    void FrameGpuEnd(VkCommandBuffer main, uint32_t slot);

    // PresentPass.cpp
    bool PresentFail(const char* what, VkResult r);
    bool EnsurePresentObjects(VkFormat swapFormat);
    bool RecordPresentPass(VkCommandBuffer cmd);
    void FinishFrame(bool visible);
    bool SameTarget(const TargetDesc& a, const TargetDesc& b);
}
