// The native device's textures: slot 0 ITexMarkAsUpdated (create + the fill protocol, run by us),
// slot 58 TexDestroy, and the texture half of the draw's resolver.
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

#include "native/Slots.hpp"

#include "native/Engine.hpp"
#include "native/Perf.hpp"
#include "native/Present.hpp"
#include "vulkan/Context.hpp"
#include "vulkan/TexFormat.hpp"
#include "vulkan/Texture.hpp"
#include "vulkan/Vk.hpp"

#include "wxl/Script.hpp"
#include "wxl/game/Binding.hpp"
#include "wxl/offsets/engine/GxDevice.hpp"

#include <cstdint>
#include <cstring>
#include <new>

// D3D's slot 0 (0x006A3070) is: if pending (+0x5A) and the context is up, ITexCreate (0x006A2C00)
// when +0x5B says so or there is no API object, ITexUpload (0x006A2D80) when a fill callback is set,
// then the base (0x00684900) clears the pending rect. This file is that, with a VkImage as the API
// object (a record behind +0x3C; +0x38 stays 0) and the upload protocol run by US, as its
// caller -- there is no D3D texture to lock, so where D3D LockRects a level and Blits the
// callback's bytes into it, the bytes go to a Vulkan image level instead:
//
//   cb(0, W, H, 0, 0, userArg, &pitch, &data)                once
//   (w, h, start, end) = ITexWHDStartEnd(tex)                 the engine's own 0x006A5EF0
//   for face (6 for a cube), for level in [start, end):       face-major
//       cb(1, w >> (level - start), ..., face, level, ...)    halving, minimum 1
//       if data and not a render target: the region of the update rect at this level, DXT-aligned,
//       from data + its offset (unless flag bit 15) -> image level (level - start), layer face
//   cb(2, W, H, 0, 0, ...)                                    once
//
// The region arithmetic (ComputeRegion / SourceAt) is ITexUpload's: the update rect shifted by the
// level, DXT-aligned, clamped.
//
// The format: dataFormat (+0x28, what the callback delivers) for an uploaded texture -- BC kept when
// the device has textureCompressionBC, decoded to RGBA8 when Vulkan 1.0 has no byte-identical format
// (Argb4444, Argb1555's channel order, V8U8 ...: texformat::Of says which), or when the device cannot
// sample the format. A render target (flag bit 7) gets an image of its storage format (+0x24), made
// with COLOR_ATTACHMENT usage (texture::CreateTarget), born transparent black and never uploaded;
// slot 23 renders into it through a per-face attachment view (level 0, one layer) this file owns.
// Format 12 (D24X8) becomes a depth image in DepthTextureFormat() (D32 on most adapters), usable as
// the depth attachment and sampled through its own view as a plain depth value. The fill protocol
// still runs for a render target -- the callback's cmd 1 is where the minimap
// (0x00581CD0) binds itself through slot 23 and draws its tiles -- but uploads nothing, as D3D.
// While a target is the colour attachment its tracked layout is COLOR_ATTACHMENT, so a draw that
// samples it there is skipped by the pipelines as "not ready" (a feedback loop on D3D too).
//
// THE TRANSFER RULE, for images: an upload into an image that no command of the frame's MAIN
// buffer has named yet (no draw sampled it, no pass or copy wrote it) is recorded into the frame's
// PRE-PASS buffer, submitted ahead of the main one, so every draw of the frame sees the new texels
// (no earlier draw of the frame read the old ones). An upload into an image the main buffer already
// named is recorded INLINE (pass ended, copy, pass resumed): the glyph cache updated while text is
// drawn is the case this exists for.
//
// Uploads are asynchronous. A fill protocol run stages every region into the frame slot's upload
// ring (present::StageUpload) and records them as ONE batch per image (texture::RecordRegions: one
// barrier, the copies, one barrier), and a new image's clear is recorded into the pre-pass as well
// (texture::RecordClear). Nothing waits. When the ring is full the batch so far is recorded, the
// frame flushed (present::FlushUploads: one submit-and-wait per 32 MiB) and the ring reused; a
// region larger than a ring chunk takes the general staging ring; only when neither can be had does
// a region take the synchronous path -- after a FlushAndWait, so everything recorded before it has
// executed and the tracked layout is the image's real one. The pre-pass records stamp `prePass`,
// not `lastRef`: lastRef is what the main buffer named (passes, copies, inline uploads --
// src/native/Targets.cpp writes it too), and the rule reads it.
//
// GENERATED MIPS (CGxTexFlags bit 6, D3D9's AUTOGENMIPMAP, which caps +0x234 offers): such a texture
// gets its full chain, the protocol's level 0 is uploaded and the rest is derived by a
// vkCmdBlitImage chain (texture::RecordGenerateMips) recorded under the same rule; a format that
// cannot be blitted (BC, or a driver without linear blits) keeps one level, counted.
//
// The REFERENCE STAMP: an inline upload or a mip chain names the image in the frame being recorded,
// so the image's retirement (slot 58, or a re-create) waits for that frame too, not only for the
// last frame that drew it.
namespace
{
    namespace gxdev     = wxl::offsets::engine::gxdevice;
    namespace slots     = wxlvk::device::slots;
    namespace texture   = wxlvk::vk::texture;
    namespace texformat = wxlvk::vk::texformat;
    namespace present   = wxlvk::native::present;

    using wxl::game::At;
    using wxlvk::native::Base;
    using wxlvk::native::Engine;
    using wxlvk::native::Own;

    constexpr uint32_t kCubeFaces    = 6;
    constexpr uint32_t kTargetCube   = 1;
    constexpr uint32_t kFormatD24X8  = 12;
    constexpr uint32_t kScratchMax   = 64u * 1024u * 1024u;

    struct TexRec
    {
        texture::Image image;          ///< VK_NULL_HANDLE when the texture cannot have one
        uint64_t       lastUsed   = 0; ///< serial of the last frame a draw named it in
        uint64_t       lastRef    = 0; ///< serial of the last frame a recorded copy or blit named it in (the main buffer)
        uint64_t       prePass    = 0; ///< serial of the last frame a pre-pass command (clear, upload, chain) named it in
        bool           genMips    = false; ///< flag bit 6 with a blittable format: level 0 uploaded, the chain derived
        uint32_t       start      = 0; ///< chain level of image level 0
        uint32_t       dataFormat = 0;
        bool           decode     = false;
        bool           renderTarget = false;
        bool           depthTarget  = false;              ///< format 12, a depth image
        bool           attachable   = false;              ///< made with COLOR_ATTACHMENT usage
        VkImageView    attach[kCubeFaces] = {};          ///< colour attachment views, by face (level 0)
    };

    struct Stats
    {
        uint64_t records;
        uint64_t images;
        uint64_t noImage;          ///< records whose texture has no image (depth target, unknown format, failure)
        uint64_t renderTargets;
        uint64_t uploads;          ///< protocol runs
        uint64_t regionsSync;      ///< regions through texture::UploadRegion, after a flush (no ring room at all)
        uint64_t regionsInline;    ///< regions recorded inline (drawn this frame: a pass break each)
        uint64_t regionsPrePass;   ///< regions recorded into the pre-pass (ahead of the frame, no wait)
        uint64_t batches;          ///< RecordRegions batches (one barrier pair each)
        uint64_t regionsLarge;     ///< regions larger than an upload ring chunk, staged in the general ring
        uint64_t clearsRecorded;   ///< new images cleared through the pre-pass
        uint64_t clearsSync;       ///< new images cleared by texture::ClearAll (no pre-pass to be had)
        uint64_t bytes;
        uint64_t refused;          ///< regions an upload refused (or the staging ring could not take)
        uint64_t decodeFailures;
        uint64_t genMipsTextures;  ///< images made with a generated chain (flag bit 6)
        uint64_t genMipsRefused;   ///< flag bit 6 on a format that cannot be blitted: one level
        uint64_t genMipsChains;    ///< chains recorded
        uint64_t genMipsInline;    ///< of which inline (the image was sampled earlier in the frame)
        uint64_t sampledFallback;  ///< formats the device cannot sample, decoded to RGBA8 instead
        uint64_t cubes;            ///< cube images made
        uint64_t sampledTarget;    ///< binds of a render target a draw resolved (sampled)
        uint64_t depthTargets;     ///< depth images made for format 12
        uint64_t depthRefused;     ///< format 12 with no sampleable depth format, or a failed create
        uint64_t targetFallback;   ///< a colour target whose format cannot be attached: sampled-only image
        uint64_t attachViews;      ///< attachment views made
        uint64_t reentered;        ///< slot 0 re-entered for the texture whose fill is running (skipped)
    };
    Stats g_stats;

    uint8_t* g_scratch     = nullptr;
    uint32_t g_scratchSize = 0;

    /// The texture whose fill protocol is running (the minimap's callback draws, and binds
    /// textures, inside it); slot 0 for that same texture is not re-entered.
    const void* g_filling = nullptr;

    TexRec*& RecOf(void* tex) { return At<TexRec*>(tex, gxdev::kTexApiObject2); }

    /// The last frame anything recorded named the image: a draw, an inline copy or a chain, or a
    /// pre-pass clear, upload or chain.
    uint64_t LastRef(const TexRec& r)
    {
        uint64_t last = r.lastUsed > r.lastRef ? r.lastUsed : r.lastRef;
        return r.prePass > last ? r.prePass : last;
    }

    /// The transfer rule: whether a command for the image must be recorded inline (the main
    /// buffer named it in this frame: a draw sampled it, or a pass, copy or inline upload wrote it)
    /// rather than into the pre-pass.
    bool MainNamed(const TexRec& r) { return present::UsedThisFrame(r.lastUsed) || present::UsedThisFrame(r.lastRef); }

    /// A record goes -- off the current target, its attachment views and framebuffers retired
    /// by the same serial as the image, before it.
    void DropRec(TexRec*& slot)
    {
        TexRec* r = slot;
        if (!r) return;
        present::ForgetImage(&r->image);
        const uint64_t last = LastRef(*r);
        for (VkImageView& v : r->attach)
        {
            present::RetireView(v, last);
            v = VK_NULL_HANDLE;
        }
        if (r->renderTarget) present::ForgetView(r->image.view, last);
        present::RetireImage(r->image, last);
        delete r;
        slot = nullptr;
    }

    /// Whether the device can sample a format from an optimally tiled image.
    bool Sampleable(VkFormat format)
    {
        const wxlvk::vk::context::Handles* h = wxlvk::vk::context::Get();
        if (!h || !h->iapi || !h->iapi->GetPhysicalDeviceFormatProperties) return true;   // not knowable: assume it can
        VkFormatProperties props = {};
        h->iapi->GetPhysicalDeviceFormatProperties(h->physical, format, &props);
        return (props.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT) != 0;
    }

    /// Levels of a full chain down to 1x1: floor(log2(max(w, h))) + 1.
    uint32_t FullChain(uint32_t w, uint32_t h)
    {
        uint32_t levels = 1;
        for (uint32_t side = w > h ? w : h; side > 1; side >>= 1) ++levels;
        return levels;
    }

    uint8_t* Scratch(uint64_t bytes)
    {
        if (bytes > kScratchMax) return nullptr;
        if (bytes <= g_scratchSize) return g_scratch;
        uint32_t want = g_scratchSize ? g_scratchSize : 256u * 1024u;
        while (want < bytes) want *= 2;
        uint8_t* fresh = new (std::nothrow) uint8_t[want];
        if (!fresh) return nullptr;
        delete[] g_scratch;
        g_scratch     = fresh;
        g_scratchSize = want;
        return g_scratch;
    }

    /// ITexCreate without D3D: the image for the size and level range ITexWHDStartEnd gives.
    void Create(void* device, void* tex)
    {
        TexRec*& slot = RecOf(tex);
        DropRec(slot);
        TexRec* r = new (std::nothrow) TexRec();
        if (!r) return;
        slot = r;
        ++g_stats.records;
        At<uint8_t>(tex, gxdev::kTexNeedsRecreate) = 0;   // as ITexCreate, whatever follows

        uint32_t w = 0, h = 0, start = 0, end = 0;
        Engine<gxdev::ITexWHDStartEndFn>(gxdev::kITexWHDStartEnd)(device, nullptr, tex, &w, &h, &start, &end);
        const uint32_t flags = At<uint32_t>(tex, gxdev::kTexFlags);
        r->renderTarget = (flags & gxdev::kTexFlagRenderTarget) != 0;
        r->start        = start;
        r->dataFormat   = At<uint32_t>(tex, r->renderTarget ? gxdev::kTexFormat : gxdev::kTexDataFormat);
        const bool cube = At<uint32_t>(tex, gxdev::kTexTarget) == kTargetCube;
        if (r->renderTarget) ++g_stats.renderTargets;

        const texformat::Info& info = texformat::Of(r->dataFormat);
        const wxlvk::vk::context::Handles* ctx = wxlvk::vk::context::Get();
        if (ctx && w && h && end > start && r->renderTarget && r->dataFormat == kFormatD24X8)
        {
            // A depth texture (the shadow map's, PortraitDepth on api 0/5). One level, 2D.
            const VkFormat depth = present::DepthTextureFormat();
            if (depth == VK_FORMAT_UNDEFINED || cube || !texture::CreateTarget(w, h, 1, false, depth, true, r->image))
            {
                r->image = texture::Image();
                ++g_stats.depthRefused;
                ++g_stats.noImage;
                return;
            }
            r->depthTarget = true;
            ++g_stats.depthTargets;
            ++g_stats.images;
            return;
        }
        if (!ctx || w == 0 || h == 0 || end <= start || r->dataFormat == kFormatD24X8 || !info.uploadable)
        {
            ++g_stats.noImage;
            return;
        }
        r->decode = info.needsDecode || (info.compressed && !ctx->textureCompressionBC);
        if (!r->decode && !r->renderTarget && info.format != VK_FORMAT_UNDEFINED && !Sampleable(info.format))
        {
            // Every texture format, on any adapter -- one the device cannot sample is decoded.
            r->decode = true;
            ++g_stats.sampledFallback;
        }
        const VkFormat format = r->decode ? VK_FORMAT_R8G8B8A8_UNORM : info.format;
        if (format == VK_FORMAT_UNDEFINED)
        {
            ++g_stats.noImage;
            return;
        }
        uint32_t levels = end - start;
        if ((flags & gxdev::kTexFlagGenerateMips) && !r->renderTarget)
        {
            // D3D's AUTOGENMIPMAP texture has one lockable level and a derived chain; so does this
            // one, when the format can be blitted.
            if (texture::CanGenerateMips(format))
            {
                const uint32_t full = FullChain(w, h);
                if (full > levels) levels = full;
                r->genMips = levels > 1;
                if (r->genMips) ++g_stats.genMipsTextures;
            }
            else
                ++g_stats.genMipsRefused;
        }
        bool made = false;
        if (r->renderTarget)
        {
            // An attachment. A format the device cannot render into keeps a sampled-only image
            // (slot 23 then reports the target as having no attachment, counted).
            made = texture::CreateTarget(w, h, levels, cube, format, false, r->image);
            if (made) r->attachable = true;
            else ++g_stats.targetFallback;
        }
        if (!made)
            made = cube ? (w == h && texture::CreateCube(w, levels, format, r->image))
                        : texture::CreateMipped(w, h, levels, format, r->image);
        if (made && cube) ++g_stats.cubes;
        if (!made)
        {
            r->image = texture::Image();
            ++g_stats.noImage;
            return;
        }
        // Born in the sampling layout, with known contents in every level and face: a level the
        // engine never uploads must sample as a colour, not as uninitialised VRAM. A render
        // target is born transparent black, not the uploads' grey: a target sampled before anything
        // drew into it (a glow chain's target on its first frame after a screen change, a portrait
        // whose model was not drawable yet) then adds nothing and shows nothing, which is what a
        // fresh D3D render target reads as on the drivers the engine was written against (and
        // what DXVK makes of one); the grey would flash.
        r->image.wanted = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        static const float kTargetBorn[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
        // Recorded into the pre-pass (a new image: nothing of this frame named it), no wait.
        // ClearAll's own submit-and-wait only when there is no pre-pass to be had.
        VkCommandBuffer pre = present::PrePass();
        if (pre && texture::RecordClear(pre, r->image, r->renderTarget ? kTargetBorn : nullptr))
        {
            r->prePass = present::Serial();
            ++g_stats.clearsRecorded;
        }
        else
        {
            texture::ClearAll(r->image, r->renderTarget ? kTargetBorn : nullptr);
            ++g_stats.clearsSync;
        }
        ++g_stats.images;
    }

    struct Region
    {
        uint32_t x, y, w, h;
    };

    /// ITexUpload's region of one level: the update rect >> level, DXT-aligned, clamped.
    bool ComputeRegion(void* tex, uint32_t flags, uint32_t level, uint32_t levelW, uint32_t levelH,
                       const texformat::Info& info, Region& out)
    {
        if (levelW == 0 || levelH == 0) return false;
        const int32_t* rect = &At<const int32_t>(tex, gxdev::kTexUpdateRect);
        const int32_t minY = rect[0], minX = rect[1], maxY = rect[2], maxX = rect[3];
        if (minX < 0 || minY < 0 || maxX <= minX || maxY <= minY)
        {
            // An empty or inverted rect: the whole level (the engine never uploads with one, but
            // the draw-time catch-up below does, after the base reset the rect).
            if (flags & gxdev::kTexFlagRectIsOrigin) return false;
            out = Region{ 0, 0, levelW, levelH };
            return true;
        }
        const uint32_t shift = level & 31u;
        uint32_t x0 = static_cast<uint32_t>(minX) >> shift;
        uint32_t y0 = static_cast<uint32_t>(minY) >> shift;
        uint32_t x1 = static_cast<uint32_t>(maxX) >> shift;
        uint32_t y1 = static_cast<uint32_t>(maxY) >> shift;
        if (x1 < x0 + 1) x1 = x0 + 1;
        if (y1 < y0 + 1) y1 = y0 + 1;
        if (info.compressed)
        {
            const uint32_t bw = info.blockWidth ? info.blockWidth : 1u;
            const uint32_t bh = info.blockHeight ? info.blockHeight : 1u;
            x0 -= x0 % bw;
            y0 -= y0 % bh;
            x1 = (x1 + bw - 1) / bw * bw;
            y1 = (y1 + bh - 1) / bh * bh;
        }
        if (x1 > levelW) x1 = levelW;
        if (y1 > levelH) y1 = levelH;
        if (x0 >= x1 || y0 >= y1) return false;
        out = Region{ x0, y0, x1 - x0, y1 - y0 };
        return true;
    }

    /// The region's first row in the callback's data.
    const uint8_t* SourceAt(uint32_t flags, const Region& g, const void* data, uint32_t pitch,
                            const texformat::Info& info)
    {
        const uint8_t* src = static_cast<const uint8_t*>(data);
        if (flags & gxdev::kTexFlagRectIsOrigin) return src;
        const uint32_t bw = info.blockWidth ? info.blockWidth : 1u;
        const uint32_t bh = info.blockHeight ? info.blockHeight : 1u;
        return src + static_cast<size_t>(g.y / bh) * pitch + static_cast<size_t>(g.x / bw) * info.blockBytes;
    }

    /// The regions of one fill protocol run staged so far, all in one source buffer.
    constexpr uint32_t kBatchMax = 128;   // 6 faces x 16 levels fit; a fuller batch is recorded and restarted
    struct Batch
    {
        VkBuffer           src   = VK_NULL_HANDLE;
        uint32_t           count = 0;
        texture::RegionCopy regions[kBatchMax];
    };

    /// Records the batch under the transfer rule (inline when the main buffer named the image
    /// this frame, else into the pre-pass) and empties it.
    void RecordBatch(TexRec& r, Batch& b)
    {
        if (b.count == 0) return;
        const bool      inl = MainNamed(r);
        VkCommandBuffer cmd = inl ? present::Outside() : present::PrePass();
        const uint32_t  n   = cmd ? texture::RecordRegions(cmd, r.image, b.src, b.regions, b.count) : 0u;
        if (n)
        {
            ++g_stats.batches;
            if (inl)
            {
                g_stats.regionsInline += n;
                r.lastRef = present::Serial();   // the main buffer names it
            }
            else
            {
                g_stats.regionsPrePass += n;
                r.prePass = present::Serial();
            }
        }
        g_stats.refused += b.count - n;
        b.count = 0;
        b.src   = VK_NULL_HANDLE;
    }

    /// One region into the image: staged into the upload ring and added to the batch, or, with
    /// no ring room at all, through texture::UploadRegion after a flush.
    void UploadRegion(TexRec& r, Batch& b, const Region& g, uint32_t vkLevel, uint32_t face, const uint8_t* rows,
                      uint32_t rowPitch)
    {
        uint64_t rowBytes = 0;
        uint32_t rowCount = 0;
        if (!texture::RegionBytes(r.image, g.w, g.h, rowBytes, rowCount) || rowBytes > rowPitch)
        {
            ++g_stats.refused;
            return;
        }
        const VkDeviceSize total = rowBytes * rowCount;
        VkBuffer     buf  = VK_NULL_HANDLE;
        VkDeviceSize off  = 0;
        void*        dst  = nullptr;
        bool         full = false;
        bool         got  = present::StageUpload(total, buf, off, dst, full);
        if (!got && full)
        {
            // The ring is full: what is staged is recorded, the frame flushed, the ring reused.
            RecordBatch(r, b);
            if (present::FlushUploads()) got = present::StageUpload(total, buf, off, dst, full);
        }
        if (!got && total > 0 && present::Stage(total, buf, off, dst))
        {
            got = true;   // larger than a ring chunk (or no ring chunk): the general staging ring
            ++g_stats.regionsLarge;
        }
        if (!got)
        {
            // No room anywhere: the synchronous path, once everything recorded before has executed
            // (its submission runs ahead of the frame, and the tracked layout must be the real one).
            RecordBatch(r, b);
            const bool flushed = !present::FrameOpen() || present::FlushAndWait();
            if (flushed && texture::UploadRegion(r.image, vkLevel, g.x, g.y, g.w, g.h, rows, rowPitch, face))
                ++g_stats.regionsSync;
            else
                ++g_stats.refused;
            return;
        }
        uint8_t* out = static_cast<uint8_t*>(dst);
        if (rowBytes == rowPitch)
            std::memcpy(out, rows, static_cast<size_t>(total));
        else
            for (uint32_t i = 0; i < rowCount; ++i)
                std::memcpy(out + static_cast<size_t>(i) * static_cast<size_t>(rowBytes),
                            rows + static_cast<size_t>(i) * rowPitch, static_cast<size_t>(rowBytes));
        g_stats.bytes += total;
        if (b.count && (b.src != buf || b.count == kBatchMax)) RecordBatch(r, b);
        b.src = buf;
        texture::RegionCopy& c = b.regions[b.count++];
        c.level     = vkLevel;
        c.x         = g.x;
        c.y         = g.y;
        c.width     = g.w;
        c.height    = g.h;
        c.layer     = face;
        c.srcOffset = off;
    }

    /// ITexUpload, run as the callback's caller.
    void Upload(void* device, void* tex, TexRec& r)
    {
        const gxdev::TexFillFn fill = At<gxdev::TexFillFn>(tex, gxdev::kTexFillCallback);
        if (!fill) return;
        void* const    userArg = At<void*>(tex, gxdev::kTexUserArg);
        const uint32_t W = At<uint32_t>(tex, gxdev::kTexWidth);
        const uint32_t H = At<uint32_t>(tex, gxdev::kTexHeight);
        const uint32_t flags = At<uint32_t>(tex, gxdev::kTexFlags);
        ++g_stats.uploads;

        uint32_t    pitch = 0;
        const void* data  = nullptr;
        fill(gxdev::kTexCmdBegin, W, H, 0, 0, userArg, &pitch, &data);

        uint32_t w = 0, h = 0, start = 0, end = 0;
        Engine<gxdev::ITexWHDStartEndFn>(gxdev::kITexWHDStartEnd)(device, nullptr, tex, &w, &h, &start, &end);
        const bool     cube  = At<uint32_t>(tex, gxdev::kTexTarget) == kTargetCube;
        const uint32_t faces = cube ? kCubeFaces : 1u;
        const texformat::Info& info = texformat::Of(r.dataFormat);
        const bool usable = r.image.handle != VK_NULL_HANDLE && !r.renderTarget && start == r.start;

        bool  levelZero = false;   // a level-0 region went in, so a generated chain is due
        Batch batch;               // the regions of this run, recorded together
        for (uint32_t face = 0; face < faces; ++face)
        {
            uint32_t lw = w, lh = h;
            for (uint32_t level = start; level < end; ++level)
            {
                data = nullptr;
                fill(gxdev::kTexCmdGetData, lw, lh, face, level, userArg, &pitch, &data);
                const uint32_t vkLevel = level - start;
                Region g;
                // A generated chain takes level 0 alone from the callback, as D3D's AUTOGENMIPMAP
                // texture can only be locked at level 0.
                if (data && usable && face < r.image.layers && vkLevel < r.image.levels
                    && (!r.genMips || vkLevel == 0)
                    && ComputeRegion(tex, flags, level, lw, lh, info, g))
                {
                    const uint8_t* src = SourceAt(flags, g, data, pitch, info);
                    if (r.decode)
                    {
                        uint8_t* rgba = Scratch(static_cast<uint64_t>(g.w) * g.h * 4u);
                        if (rgba && texformat::DecodeToRgba8(r.dataFormat, g.w, g.h, src, pitch, rgba))
                            UploadRegion(r, batch, g, vkLevel, face, rgba, g.w * 4u);
                        else
                            ++g_stats.decodeFailures;
                    }
                    else
                        UploadRegion(r, batch, g, vkLevel, face, src, pitch);
                    if (vkLevel == 0) levelZero = true;
                }
                lw = lw > 1 ? lw >> 1 : 1;
                lh = lh > 1 ? lh >> 1 : 1;
            }
        }
        fill(gxdev::kTexCmdEnd, W, H, 0, 0, userArg, &pitch, &data);
        RecordBatch(r, batch);   // before the chain, which derives from level 0

        if (r.genMips && levelZero)
        {
            // The chain under the transfer rule: inline when the main buffer named the image this
            // frame (the level-0 copy went inline too), else ahead of the frame's draws.
            const bool inlineChain = MainNamed(r);
            VkCommandBuffer cmd = inlineChain ? present::Outside() : present::PrePass();
            if (cmd && texture::RecordGenerateMips(cmd, r.image))
            {
                (inlineChain ? r.lastRef : r.prePass) = present::Serial();
                ++g_stats.genMipsChains;
                if (inlineChain) ++g_stats.genMipsInline;
            }
            else
                ++g_stats.refused;
        }
    }
}

namespace wxlvk::native::impl
{
    void __fastcall ITexMarkAsUpdated(void* device, void* edx, CGxTex* tex)
    {
        if (tex && tex == g_filling)
        {
            // The texture whose fill is running asked for itself again (its callback's draws or
            // binds); D3D would recurse, the record is already being filled.
            ++g_stats.reentered;
            return;
        }
        if (tex && At<uint8_t>(tex, gxdev::kTexUpdatePending) && At<uint32_t>(device, gxdev::kContext)
            && present::DrawPathUp())
        {
            wxlvk::native::perf::Scope timed(wxlvk::native::perf::kUpload);   // create, clear, fill
            if (At<uint8_t>(tex, gxdev::kTexNeedsRecreate) || !RecOf(tex)) Create(device, tex);
            if (TexRec* r = RecOf(tex))
            {
                const void* outer = g_filling;
                g_filling = tex;
                Upload(device, tex, *r);
                g_filling = outer;
            }
        }
        Base<slots::kITexMarkAsUpdated>()(device, edx, tex);
    }

    void __fastcall TexDestroy(void* device, void* edx, CGxTex* tex)
    {
        // D3D 0x006A2BB0 releases +0x38 first; here the image is retired by serial (the frame that
        // last sampled it may be on the GPU) and the base unlinks and frees the CGxTex.
        if (tex) DropRec(RecOf(tex));
        Base<slots::kTexDestroy>()(device, edx, tex);
    }

    void __fastcall DeviceSetBaseMipLevel(void* device, void* edx, uint32_t level)
    {
        // D3D 0x0068E4A0: ITexForceRecreation(1) (0x006A2AA0, read from the bytes), then the base
        // stores +0x350, which ITexWHDStartEnd reads when each texture is created again. Native's
        // force-recreation: for every texture on the device list (+0x290C, linked at +0x44) that has
        // an image, the image is retired by serial, +0x5B set, a whole update marked, the fill
        // callback told cmd 3 ("dropped"); the next bind creates it at the new base level and
        // refills it. One deviation: D3D calls the callback unconditionally; a texture with none
        // ("cannot update") cannot be refilled, so it keeps its image here.
        uint64_t recreated = 0;
        if (At<uint32_t>(device, gxdev::kContext))
        {
            uintptr_t link = At<uintptr_t>(device, gxdev::kTexListHead);
            while (link && !(link & 1u))
            {
                void* tex = reinterpret_cast<void*>(link);
                link      = At<uintptr_t>(tex, gxdev::kTexLinkNext);
                const gxdev::TexFillFn fill = At<gxdev::TexFillFn>(tex, gxdev::kTexFillCallback);
                const bool target = (At<uint32_t>(tex, gxdev::kTexFlags) & gxdev::kTexFlagRenderTarget) != 0;
                if (!RecOf(tex) || (!fill && !target)) continue;
                DropRec(RecOf(tex));
                At<uint8_t>(tex, gxdev::kTexNeedsRecreate) = 1;
                Engine<gxdev::GxTexUpdateFn>(gxdev::kGxTexUpdate)(tex, 0, 0, 0, 0, 0);
                if (fill)
                {
                    uint32_t    pitch = 0;
                    const void* data  = nullptr;
                    fill(gxdev::kTexCmdDropped, At<uint32_t>(tex, gxdev::kTexWidth), At<uint32_t>(tex, gxdev::kTexHeight),
                         0, 0, At<void*>(tex, gxdev::kTexUserArg), &pitch, &data);
                }
                ++recreated;
            }
        }
        Base<slots::kDeviceSetBaseMipLevel>()(device, edx, level);
        NoteTexturesRecreated(recreated);
        // D3D notifies when the device and its window exist.
        if (At<HWND>(device, gxdev::kWindow) && At<uint32_t>(device, gxdev::kContext))
            Own<slots::kNotifyOnTextureRecreation>(device)(device, edx);
        wxl::ScriptMgr::Log(WXL_LOG_DEBUG,
                            "native: DeviceSetBaseMipLevel %u: %llu texture(s) dropped for recreation at the new "
                            "base level (made again at their next bind), slot 6 notified", level,
                            static_cast<unsigned long long>(recreated));
    }

    void PrepareTexture(void* device, void* tex)
    {
        // ISetTexture (0x006A4900) calls slot 0 at every bind; the native IRsSendToHw is a no-op,
        // so the draw does it for every stage it binds. A texture that never got a record (its
        // update arrived while the device was not up, and the base cleared the pending flag) is
        // made pending again, whole, so it is not left blank for the session.
        if (!tex || !At<uint32_t>(device, gxdev::kContext)) return;
        if (!RecOf(tex))
        {
            if (!At<gxdev::TexFillFn>(tex, gxdev::kTexFillCallback)) return;
            At<uint8_t>(tex, gxdev::kTexUpdatePending) = 1;
        }
        if (!At<uint8_t>(tex, gxdev::kTexUpdatePending)) return;
        Own<slots::kITexMarkAsUpdated>(device)(device, nullptr, static_cast<CGxTex*>(tex));
    }

    bool ResolveTexture(const void* tex, const wxlvk::vk::texture::Image*& image)
    {
        TexRec* r = tex ? At<TexRec*>(tex, gxdev::kTexApiObject2) : nullptr;
        if (!r || r->image.handle == VK_NULL_HANDLE) return false;
        // A render target holds what slots 23 / 27 / 59 / 60 put in it, so it resolves like any
        // texture. One that is the colour attachment right now is in COLOR_ATTACHMENT and the
        // pipelines skip the draw as not ready.
        if (r->renderTarget) ++g_stats.sampledTarget;
        r->lastUsed = present::Serial();   // the transfer rule's stamp
        image = &r->image;
        return true;
    }

    void LogTextureStats()
    {
        const Stats& g = g_stats;
        wxl::ScriptMgr::Log(WXL_LOG_INFO,
                            "native: textures: %llu record(s), %llu image(s), %llu without one, %llu "
                            "render target(s); %llu upload(s): %llu region(s) ahead of the frame "
                            "(pre-pass, no wait), %llu inline, %llu synchronous after a flush (%llu KiB "
                            "staged), %llu refused, %llu decode failure(s)",
                            static_cast<unsigned long long>(g.records),
                            static_cast<unsigned long long>(g.images),
                            static_cast<unsigned long long>(g.noImage),
                            static_cast<unsigned long long>(g.renderTargets),
                            static_cast<unsigned long long>(g.uploads),
                            static_cast<unsigned long long>(g.regionsPrePass),
                            static_cast<unsigned long long>(g.regionsInline),
                            static_cast<unsigned long long>(g.regionsSync),
                            static_cast<unsigned long long>(g.bytes / 1024),
                            static_cast<unsigned long long>(g.refused),
                            static_cast<unsigned long long>(g.decodeFailures));
        present::Stats f = {};
        present::GetStats(f);
        wxl::ScriptMgr::Log(WXL_LOG_INFO,
                            "native: textures: %llu batch(es) (one barrier pair each), %llu region(s) "
                            "too large for the upload ring (general staging); new images cleared %llu in "
                            "the pre-pass / %llu synchronously; upload ring %llu KiB handed out, %llu "
                            "flush(es) for a full ring, %u chunk(s) of 8 MiB alive (peak %u)",
                            static_cast<unsigned long long>(g.batches),
                            static_cast<unsigned long long>(g.regionsLarge),
                            static_cast<unsigned long long>(g.clearsRecorded),
                            static_cast<unsigned long long>(g.clearsSync),
                            static_cast<unsigned long long>(f.uploadBytes / 1024),
                            static_cast<unsigned long long>(f.uploadFlushes), f.uploadChunks, f.uploadChunksPeak);
        wxl::ScriptMgr::Log(WXL_LOG_INFO,
                            "native: textures: %llu cube(s); generated mips: %llu texture(s), %llu "
                            "chain(s) recorded (%llu inline), %llu refused for the format (one level); "
                            "%llu format(s) the device cannot sample, decoded instead; %llu bind(s) of a "
                            "render target sampled; %llu depth target(s) made (%llu refused), %llu colour "
                            "target(s) without attachment usage, %llu attachment view(s), %llu slot-0 "
                            "re-entr%s for a texture being filled",
                            static_cast<unsigned long long>(g.cubes),
                            static_cast<unsigned long long>(g.genMipsTextures),
                            static_cast<unsigned long long>(g.genMipsChains),
                            static_cast<unsigned long long>(g.genMipsInline),
                            static_cast<unsigned long long>(g.genMipsRefused),
                            static_cast<unsigned long long>(g.sampledFallback),
                            static_cast<unsigned long long>(g.sampledTarget),
                            static_cast<unsigned long long>(g.depthTargets),
                            static_cast<unsigned long long>(g.depthRefused),
                            static_cast<unsigned long long>(g.targetFallback),
                            static_cast<unsigned long long>(g.attachViews),
                            static_cast<unsigned long long>(g.reentered), g.reentered == 1 ? "y" : "ies");
    }

    bool TargetImageOf(void* device, void* tex, uint32_t face, bool attach, TargetImage& out)
    {
        out = TargetImage();
        if (!tex || !At<uint32_t>(device, gxdev::kContext) || !present::DrawPathUp()) return false;
        // D3D's 23 / 26 / 27 / 59 / 60 create the texture first when it has no API object (cgxdevice
        // 4.5, "create only"): the same here, without the upload.
        if (At<uint8_t>(tex, gxdev::kTexNeedsRecreate) || !RecOf(tex)) Create(device, tex);
        TexRec* r = RecOf(tex);
        if (!r || r->image.handle == VK_NULL_HANDLE) return false;
        out.image = &r->image;
        out.stamp = &r->lastRef;
        out.depth = r->depthTarget;
        out.renderTarget = r->renderTarget;
        if (!attach) return true;
        if (!r->renderTarget) return false;
        if (r->depthTarget)
        {
            out.view = r->image.view;   // one view: attachment and sampled
            return true;
        }
        if (!r->attachable || face >= r->image.layers) return false;
        if (r->attach[face] == VK_NULL_HANDLE)
        {
            const wxlvk::vk::context::Handles* h = wxlvk::vk::context::Get();
            if (!h) return false;
            VkImageViewCreateInfo v = {};
            v.sType    = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
            v.image    = r->image.handle;
            v.viewType = VK_IMAGE_VIEW_TYPE_2D;
            v.format   = r->image.format;
            v.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
            v.subresourceRange.baseMipLevel   = 0;
            v.subresourceRange.levelCount     = 1;
            v.subresourceRange.baseArrayLayer = face;
            v.subresourceRange.layerCount     = 1;
            if (h->dapi->CreateImageView(h->device, &v, nullptr, &r->attach[face]) != VK_SUCCESS)
            {
                r->attach[face] = VK_NULL_HANDLE;
                return false;
            }
            ++g_stats.attachViews;
        }
        out.view = r->attach[face];
        return true;
    }
}
