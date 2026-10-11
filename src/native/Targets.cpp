// The native device's render targets and copies: slots 23 (DeviceSetRenderTarget), 24 / 25 (depth
// save / restore), 26 (DeviceResolveDepthBuffer), 27 (DeviceCopyTex), 59 (TexCopy), 60 (TexStretch),
// and slot 20 (DeviceReadPixels), the read-back of the back buffer.
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
#include "native/Present.hpp"
#include "vulkan/Context.hpp"
#include "vulkan/Memory.hpp"
#include "vulkan/Texture.hpp"
#include "vulkan/Vk.hpp"

#include "wxl/Script.hpp"
#include "wxl/game/Binding.hpp"
#include "wxl/offsets/engine/Gx.hpp"
#include "wxl/offsets/engine/GxDevice.hpp"

#include <cstdint>
#include <cstring>

// cgxdevice 9.3 / 10.2 / 10.3 is the D3D behaviour reproduced:
//
//  - 23: the base half (0x006842A0) keeps the target table (+0x2910 + buf*0xC) and curWindow right;
//    the native half resolves the table the way D3D's 0x0068F770 resolves it -- colour = the
//    texture's level-0 surface of the bound face, or the back buffer; depth = the depth texture
//    (+0x291C) if set, else with a colour override the shared offscreen depth (+0x3B38, grown to the
//    target), else the default depth -- and hands it to the frame (present::SetTarget), which ends
//    the open pass and begins the next one on it. Only on a change of {tex, face}, as D3D.
//  - 26 binds a depth texture as the current depth attachment until the next 23; 24 / 25 save and
//    restore the current depth choice (D3D's global 0xC6033C).
//  - 27 copies level 0 of one face to level 0 of another, 59 / 60 copy a rectangle of the BACK
//    BUFFER (whatever target is bound) into a level of a texture: outside the pass (a break), a
//    vkCmdCopyImage when formats and sizes match, else a vkCmdBlitImage (NEAREST for 27 / 59 and a
//    same-size 60, LINEAR for a stretched 60). Both images are top-down, as D3D's surfaces are.
//  - 20 copies the back buffer's top-left w x h (the rect origin NOT applied, as observed of D3D)
//    into a host buffer, flushes the frame and waits, and hands the bytes over as CImVector B G R A.
//
// RTAlphaSupported (0x00616DC0: Clear(3, 0), ReadPixels 64x64, "any alpha not 0xFF"): the back
// buffer here has a real alpha channel and the read-back keeps it, so it returns 1, as stock D3D does
// -- which is what makes UpdatePendingPortraits create its 64x64 portrait render target.
namespace
{
    namespace gx      = wxl::offsets::engine::gx;
    namespace gxdev   = wxl::offsets::engine::gxdevice;
    namespace slots   = wxlvk::device::slots;
    namespace memory  = wxlvk::vk::memory;
    namespace texture = wxlvk::vk::texture;
    namespace present = wxlvk::native::present;
    namespace impl    = wxlvk::native::impl;

    using wxl::game::At;
    using wxlvk::native::Base;
    using wxlvk::native::Engine;
    using present::TargetDesc;

    struct Stats
    {
        uint64_t setCalls;        ///< slot 23 calls
        uint64_t setUnchanged;    ///< of which with the same {tex, face}: nothing done, as D3D
        uint64_t setTexture;      ///< targets resolved to a texture colour attachment
        uint64_t setBack;         ///< targets resolved to the back buffer
        uint64_t setDepthTexture; ///< with a depth texture
        uint64_t setNoAttach;     ///< a bound texture with no attachment: its draws are dropped
        uint64_t depthResolves;   ///< slot 26
        uint64_t depthSaves;      ///< slot 24
        uint64_t depthRestores;   ///< slot 25
        uint64_t copyTex;         ///< slot 27 copies recorded
        uint64_t texCopy;         ///< slot 59 copies recorded
        uint64_t texStretch;      ///< slot 60 copies recorded
        uint64_t blits;           ///< of the three, through vkCmdBlitImage (format or size differ)
        uint64_t copyRefused;     ///< copies not done (no image, no back buffer, empty rect, format)
        uint64_t readPixels;      ///< slot 20 calls with a non-empty rect
        uint64_t readPixelsDone;  ///< of which really read back
        uint64_t readPixelsBytes;
        uint64_t readRefused;     ///< of which answered with 0xFF (no back buffer, flush refused)
    };
    Stats g;

    memory::Buffer g_readback;
    bool           g_noAttachLogged = false;

    /// Slots 24 / 25: the depth half of the current target.
    struct SavedDepth
    {
        bool                       valid = false;
        TargetDesc::Depth          depth = TargetDesc::Depth::Back;
        texture::Image*            image = nullptr;
        VkImageView                view  = VK_NULL_HANDLE;
        uint64_t*                  stamp = nullptr;
    };
    SavedDepth g_saved;

    void* const* TableEntry(void* device, uint32_t buffer)
    {
        return &At<void* const>(device, gxdev::kRenderTargets + buffer * gxdev::kRenderTargetStride
                                            + gxdev::kRenderTargetTex);
    }

    bool IsDepth(VkFormat f)
    {
        return f == VK_FORMAT_D16_UNORM || f == VK_FORMAT_D32_SFLOAT || f == VK_FORMAT_X8_D24_UNORM_PACK32
               || f == VK_FORMAT_D24_UNORM_S8_UINT || f == VK_FORMAT_D16_UNORM_S8_UINT
               || f == VK_FORMAT_D32_SFLOAT_S8_UINT;
    }

    /// D3D 0x0068F770's resolution of the target table, handed to the frame.
    void ApplyTargets(void* device)
    {
        void* const    colour = *TableEntry(device, gxdev::kBufferColor);
        const uint32_t face   = At<uint32_t>(device, gxdev::kRenderTargets + gxdev::kBufferColor * gxdev::kRenderTargetStride
                                                         + gxdev::kRenderTargetFace);
        void* const    depth  = *TableEntry(device, gxdev::kBufferDepth);

        TargetDesc t;
        const float w = At<const float>(device, gx::kCurWindowWidth);
        const float h = At<const float>(device, gx::kCurWindowHeight);
        t.extent = VkExtent2D{ w > 0.0f ? static_cast<uint32_t>(w + 0.5f) : 0u,
                               h > 0.0f ? static_cast<uint32_t>(h + 0.5f) : 0u };
        if (colour)
        {
            impl::TargetImage c;
            if (impl::TargetImageOf(device, colour, face, true, c) && !c.depth)
            {
                t.colour      = c.image;
                t.colourView  = c.view;
                t.colourStamp = c.stamp;
                ++g.setTexture;
            }
            else
            {
                t.invalid = true;
                ++g.setNoAttach;
                if (!g_noAttachLogged)
                {
                    g_noAttachLogged = true;
                    wxl::ScriptMgr::Log(WXL_LOG_WARN,
                                        "native: targets: a colour target has no attachment (CGxTex %p, "
                                        "%ux%u, storage format %u, flags 0x%X, face %u); its draws and "
                                        "clears are dropped until the next slot 23 (logged once)",
                                        colour, At<uint32_t>(colour, gxdev::kTexWidth),
                                        At<uint32_t>(colour, gxdev::kTexHeight),
                                        At<uint32_t>(colour, gxdev::kTexFormat),
                                        At<uint32_t>(colour, gxdev::kTexFlags), face);
                }
            }
        }
        else
            ++g.setBack;

        t.depth = colour ? TargetDesc::Depth::Offscreen : TargetDesc::Depth::Back;
        if (depth)
        {
            impl::TargetImage d;
            if (impl::TargetImageOf(device, depth, 0, true, d) && d.depth)
            {
                t.depth      = TargetDesc::Depth::Texture;
                t.depthImage = d.image;
                t.depthView  = d.view;
                t.depthStamp = d.stamp;
                ++g.setDepthTexture;
            }
        }
        present::SetTarget(t);
    }

    /**
     * One rectangle from one image to another, outside a pass, both already in TRANSFER_SRC /
     * TRANSFER_DST: a copy when formats and sizes match, else a blit the formats are checked for.
     */
    bool CopyRect(VkCommandBuffer cmd, VkImage src, VkFormat srcFormat, uint32_t srcLevel, uint32_t srcLayer,
                  int32_t sx0, int32_t sy0, int32_t sx1, int32_t sy1, VkImage dst, VkFormat dstFormat,
                  uint32_t dstLevel, uint32_t dstLayer, int32_t dx0, int32_t dy0, int32_t dx1, int32_t dy1,
                  bool linear)
    {
        const wxlvk::vk::context::Handles* h = wxlvk::vk::context::Get();
        if (!h || sx1 <= sx0 || sy1 <= sy0 || dx1 <= dx0 || dy1 <= dy0) return false;
        const bool srcDepth = IsDepth(srcFormat), dstDepth = IsDepth(dstFormat);
        if (srcDepth != dstDepth) return false;
        const VkImageAspectFlags aspect = srcDepth ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT;
        if (srcFormat == dstFormat && sx1 - sx0 == dx1 - dx0 && sy1 - sy0 == dy1 - dy0)
        {
            VkImageCopy c = {};
            c.srcSubresource = VkImageSubresourceLayers{ aspect, srcLevel, srcLayer, 1 };
            c.dstSubresource = VkImageSubresourceLayers{ aspect, dstLevel, dstLayer, 1 };
            c.srcOffset      = VkOffset3D{ sx0, sy0, 0 };
            c.dstOffset      = VkOffset3D{ dx0, dy0, 0 };
            c.extent         = VkExtent3D{ static_cast<uint32_t>(sx1 - sx0), static_cast<uint32_t>(sy1 - sy0), 1 };
            h->dapi->CmdCopyImage(cmd, src, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, dst,
                                  VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &c);
            return true;
        }
        VkFormatProperties sp = {}, dp = {};
        h->iapi->GetPhysicalDeviceFormatProperties(h->physical, srcFormat, &sp);
        h->iapi->GetPhysicalDeviceFormatProperties(h->physical, dstFormat, &dp);
        if (!(sp.optimalTilingFeatures & VK_FORMAT_FEATURE_BLIT_SRC_BIT)
            || !(dp.optimalTilingFeatures & VK_FORMAT_FEATURE_BLIT_DST_BIT))
            return false;
        if (srcDepth || !(sp.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT))
            linear = false;   // a depth blit must be NEAREST
        if (srcDepth && srcFormat != dstFormat) return false;
        VkImageBlit b = {};
        b.srcSubresource = VkImageSubresourceLayers{ aspect, srcLevel, srcLayer, 1 };
        b.dstSubresource = VkImageSubresourceLayers{ aspect, dstLevel, dstLayer, 1 };
        b.srcOffsets[0]  = VkOffset3D{ sx0, sy0, 0 };
        b.srcOffsets[1]  = VkOffset3D{ sx1, sy1, 1 };
        b.dstOffsets[0]  = VkOffset3D{ dx0, dy0, 0 };
        b.dstOffsets[1]  = VkOffset3D{ dx1, dy1, 1 };
        h->dapi->CmdBlitImage(cmd, src, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, dst, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                              1, &b, linear ? VK_FILTER_LINEAR : VK_FILTER_NEAREST);
        ++g.blits;
        return true;
    }

    int32_t Clamp(int32_t v, int32_t lo, int32_t hi) { return v < lo ? lo : (v > hi ? hi : v); }

    uint32_t LevelSide(uint32_t side, uint32_t level)
    {
        const uint32_t s = side >> (level & 31u);
        return s ? s : 1u;
    }

    /**
     * Slots 59 / 60: a rectangle of the back buffer into level `level`, layer `face` of `tex`.
     * Rects are {x0, y0, x1, y1}, the source's clamped to the back buffer, the destination's to the
     * level; a clamp that changes one and not the other keeps the copy a copy (59) by clamping both.
     */
    bool FromBackBuffer(void* device, void* tex, int32_t sx0, int32_t sy0, int32_t sx1, int32_t sy1, int32_t dx0,
                        int32_t dy0, int32_t dx1, int32_t dy1, uint32_t level, uint32_t face, bool stretch)
    {
        impl::TargetImage d;
        if (!impl::TargetImageOf(device, tex, face, false, d) || level >= d.image->levels || face >= d.image->layers
            || !present::BackBufferReadable())
        {
            ++g.copyRefused;
            return false;
        }
        VkCommandBuffer cmd = present::Outside();
        VkImage         back = VK_NULL_HANDLE;
        VkFormat        backFormat = VK_FORMAT_UNDEFINED;
        VkExtent2D      backExtent = {};
        if (!cmd || !present::BackBuffer(cmd, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, back, backFormat, backExtent))
        {
            ++g.copyRefused;
            return false;
        }
        const int32_t bw = static_cast<int32_t>(backExtent.width), bh = static_cast<int32_t>(backExtent.height);
        const int32_t lw = static_cast<int32_t>(LevelSide(d.image->extent.width, level));
        const int32_t lh = static_cast<int32_t>(LevelSide(d.image->extent.height, level));
        if (!stretch)
        {
            // A copy: one size, cut to what both sides have.
            int32_t w = sx1 - sx0, h = sy1 - sy0;
            if (sx0 < 0) { dx0 -= sx0; w += sx0; sx0 = 0; }
            if (sy0 < 0) { dy0 -= sy0; h += sy0; sy0 = 0; }
            if (dx0 < 0) { sx0 -= dx0; w += dx0; dx0 = 0; }
            if (dy0 < 0) { sy0 -= dy0; h += dy0; dy0 = 0; }
            if (sx0 + w > bw) w = bw - sx0;
            if (sy0 + h > bh) h = bh - sy0;
            if (dx0 + w > lw) w = lw - dx0;
            if (dy0 + h > lh) h = lh - dy0;
            sx1 = sx0 + w;
            sy1 = sy0 + h;
            dx1 = dx0 + w;
            dy1 = dy0 + h;
        }
        else
        {
            sx0 = Clamp(sx0, 0, bw); sx1 = Clamp(sx1, 0, bw); sy0 = Clamp(sy0, 0, bh); sy1 = Clamp(sy1, 0, bh);
            dx0 = Clamp(dx0, 0, lw); dx1 = Clamp(dx1, 0, lw); dy0 = Clamp(dy0, 0, lh); dy1 = Clamp(dy1, 0, lh);
        }
        present::Transition(cmd, *d.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        const bool linear = stretch && (sx1 - sx0 != dx1 - dx0 || sy1 - sy0 != dy1 - dy0);
        const bool ok = CopyRect(cmd, back, backFormat, 0, 0, sx0, sy0, sx1, sy1, d.image->handle, d.image->format,
                                 level, face, dx0, dy0, dx1, dy1, linear);
        present::Transition(cmd, *d.image, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        if (d.stamp) *d.stamp = present::Serial();
        if (!ok)
        {
            ++g.copyRefused;
            return false;
        }
        return true;
    }
}

namespace wxlvk::native::impl
{
    // --- slot 23 -------------------------------------------------------------------------------------

    void __fastcall DeviceSetRenderTarget(void* device, void* edx, uint32_t buffer, CGxTex* tex, uint32_t face)
    {
        ++g.setCalls;
        if (buffer <= gxdev::kBufferDepth && *TableEntry(device, buffer) == tex
            && At<uint32_t>(device, gxdev::kRenderTargets + buffer * gxdev::kRenderTargetStride
                                        + gxdev::kRenderTargetFace) == face)
        {
            ++g.setUnchanged;   // D3D 0x0068F770 acts only on a change
            return;
        }
        // The base half: the table, and curWindow = the texture's size or defWindow.
        Base<slots::kDeviceSetRenderTarget>()(device, edx, buffer, tex, face);
        if (!At<uint32_t>(device, gxdev::kContext) || !present::DrawPathUp()) return;
        g_saved.valid = false;   // D3D re-derives the depth surface here, overriding 24 / 26
        ApplyTargets(device);
        // As D3D: the viewport and the scissor are re-placed against the new curWindow.
        At<uint32_t>(device, gx::kViewportDirty) = 1;
        At<uint32_t>(device, gxdev::kScissorDirty) = 1;
    }

    // --- slots 24, 25, 26 -----------------------------------------------------------------------------

    void __fastcall CaptureDepthStencilSurface(void* /*device*/, void* /*edx*/)
    {
        ++g.depthSaves;
        const TargetDesc& t = present::Target();
        g_saved.valid = true;
        g_saved.depth = t.depth;
        g_saved.image = t.depthImage;
        g_saved.view  = t.depthView;
        g_saved.stamp = t.depthStamp;
    }

    void __fastcall BindDepthStencilSurface(void* /*device*/, void* /*edx*/)
    {
        ++g.depthRestores;
        if (!g_saved.valid) return;
        TargetDesc t = present::Target();
        t.depth      = g_saved.depth;
        t.depthImage = g_saved.image;
        t.depthView  = g_saved.view;
        t.depthStamp = g_saved.stamp;
        present::SetTarget(t);
    }

    void __fastcall DeviceResolveDepthBuffer(void* device, void* /*edx*/, CGxTex* tex)
    {
        // D3D 0x0068F900: create if needed, SetDepthStencilSurface(tex). Until the next slot 23.
        ++g.depthResolves;
        TargetImage d;
        if (!tex || !TargetImageOf(device, tex, 0, true, d) || !d.depth) return;
        TargetDesc t = present::Target();
        t.depth      = TargetDesc::Depth::Texture;
        t.depthImage = d.image;
        t.depthView  = d.view;
        t.depthStamp = d.stamp;
        present::SetTarget(t);
    }

    // --- slots 27, 59, 60 -----------------------------------------------------------------------------

    void __fastcall DeviceCopyTex(void* device, void* /*edx*/, CGxTex* src, uint32_t srcFace, CGxTex* dst,
                                  uint32_t dstFace)
    {
        // D3D 0x0068F950: level 0 to level 0, StretchRect(NONE).
        TargetImage a, b;
        if (!src || !dst || !TargetImageOf(device, src, srcFace, false, a) || !TargetImageOf(device, dst, dstFace, false, b)
            || a.image == b.image || srcFace >= a.image->layers || dstFace >= b.image->layers)
        {
            ++g.copyRefused;
            return;
        }
        VkCommandBuffer cmd = present::Outside();
        if (!cmd)
        {
            ++g.copyRefused;
            return;
        }
        present::Transition(cmd, *a.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
        present::Transition(cmd, *b.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        const bool ok = CopyRect(cmd, a.image->handle, a.image->format, 0, srcFace, 0, 0,
                                 static_cast<int32_t>(a.image->extent.width), static_cast<int32_t>(a.image->extent.height),
                                 b.image->handle, b.image->format, 0, dstFace, 0, 0,
                                 static_cast<int32_t>(b.image->extent.width), static_cast<int32_t>(b.image->extent.height),
                                 false);
        present::Transition(cmd, *a.image, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        present::Transition(cmd, *b.image, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        const uint64_t serial = present::Serial();
        if (a.stamp) *a.stamp = serial;
        if (b.stamp) *b.stamp = serial;
        if (ok) ++g.copyTex;
        else ++g.copyRefused;
    }

    bool __fastcall TexCopy(void* device, void* /*edx*/, CGxTex* tex, C2iVector* dstPos, C2iVector* srcPos,
                            C2iVector* size, uint32_t level, uint32_t face)
    {
        // D3D 0x006A30D0: StretchRect(back buffer, {srcPos, srcPos + size}, level surface,
        // {dstPos, dstPos + size}, NONE). C2iVector is {x, y}.
        if (!tex || !dstPos || !srcPos || !size || !At<uint32_t>(device, gxdev::kContext)) return false;
        const int32_t* d = reinterpret_cast<const int32_t*>(dstPos);
        const int32_t* s = reinterpret_cast<const int32_t*>(srcPos);
        const int32_t* z = reinterpret_cast<const int32_t*>(size);
        const bool ok = FromBackBuffer(device, tex, s[0], s[1], s[0] + z[0], s[1] + z[1], d[0], d[1], d[0] + z[0],
                                       d[1] + z[1], level, face, false);
        if (ok) ++g.texCopy;
        return ok;
    }

    bool __fastcall TexStretch(void* device, void* /*edx*/, CGxTex* tex, CiRect* dst, CiRect* src, uint32_t level,
                               uint32_t face)
    {
        // D3D 0x006A31E0: from the back buffer; a null rect is the whole surface; NONE when both
        // sizes match, LINEAR otherwise. CiRect {minY, minX, maxY, maxX} -> RECT {r[1], r[0], r[3], r[2]}.
        if (!tex || !At<uint32_t>(device, gxdev::kContext)) return false;
        int32_t sx0 = 0, sy0 = 0, sx1 = 0x7FFFFFFF, sy1 = 0x7FFFFFFF;
        int32_t dx0 = 0, dy0 = 0, dx1 = 0x7FFFFFFF, dy1 = 0x7FFFFFFF;
        if (src)
        {
            const int32_t* r = reinterpret_cast<const int32_t*>(src);
            sx0 = r[1]; sy0 = r[0]; sx1 = r[3]; sy1 = r[2];
        }
        // A null rect is the whole surface: the clamps in FromBackBuffer cut the open ends.
        if (dst)
        {
            const int32_t* r = reinterpret_cast<const int32_t*>(dst);
            dx0 = r[1]; dy0 = r[0]; dx1 = r[3]; dy1 = r[2];
        }
        const bool ok = FromBackBuffer(device, tex, sx0, sy0, sx1, sy1, dx0, dy0, dx1, dy1, level, face, true);
        if (ok) ++g.texStretch;
        return ok;
    }

    // --- slot 20 ---------------------------------------------------------------------------------------

    void __fastcall DeviceReadPixels(void* device, void* edx, CiRect* rect, TSGrowableArray* out)
    {
        if (!rect || !out) return;

        // The D3D path's first two steps, exactly (0x0068FED0): clamp to the window, then size the
        // array to w x h with the engine's own SetCount.
        int32_t* r = reinterpret_cast<int32_t*>(rect);
        Engine<gxdev::ClampRectToWindowFn>(gxdev::kClampRectToWindow)(device, edx, r);
        const int32_t h = r[2] - r[0];
        const int32_t w = r[3] - r[1];
        const uint32_t count = (w > 0 && h > 0) ? static_cast<uint32_t>(w) * static_cast<uint32_t>(h) : 0u;
        Engine<gxdev::CImVectorArraySetCountFn>(gxdev::kCImVectorArraySetCount)(out, edx, count);
        uint8_t* data = At<uint8_t*>(out, gxdev::kGrowableData);
        if (!data || !count) return;
        ++g.readPixels;

        bool ok = false;
        if (At<uint32_t>(device, gxdev::kContext) && present::DrawPathUp() && present::BackBufferReadable())
        {
            VkCommandBuffer cmd = present::Outside();
            VkImage         back = VK_NULL_HANDLE;
            VkFormat        format = VK_FORMAT_UNDEFINED;
            VkExtent2D      extent = {};
            if (cmd && present::BackBuffer(cmd, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, back, format, extent))
            {
                // From the surface's top-left corner, the rect origin not applied (observed of D3D).
                const uint32_t cw = static_cast<uint32_t>(w) < extent.width ? static_cast<uint32_t>(w) : extent.width;
                const uint32_t ch = static_cast<uint32_t>(h) < extent.height ? static_cast<uint32_t>(h) : extent.height;
                const VkDeviceSize bytes = static_cast<VkDeviceSize>(cw) * ch * 4u;
                if (bytes && g_readback.size < bytes)
                {
                    // Only ever read after a flush and a wait, so nothing in flight names it.
                    memory::DestroyBuffer(g_readback);
                    if (!memory::CreateBuffer(bytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT, memory::Where::HostVisible,
                                              g_readback)
                        || !g_readback.block.mapped)
                        memory::DestroyBuffer(g_readback);
                }
                if (bytes && g_readback.handle != VK_NULL_HANDLE)
                {
                    VkBufferImageCopy c = {};
                    c.imageSubresource = VkImageSubresourceLayers{ VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
                    c.imageExtent      = VkExtent3D{ cw, ch, 1 };
                    wxlvk::vk::context::Get()->dapi->CmdCopyImageToBuffer(cmd, back, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                                                         g_readback.handle, 1, &c);
                    if (present::FlushAndWait())
                    {
                        // Rows beyond what the back buffer has stay opaque white, as the stub was.
                        std::memset(data, 0xFF, static_cast<size_t>(count) * 4u);
                        const uint8_t* src = static_cast<const uint8_t*>(g_readback.block.mapped);
                        const bool     rgba = format == VK_FORMAT_R8G8B8A8_UNORM || format == VK_FORMAT_R8G8B8A8_SRGB;
                        for (uint32_t y = 0; y < ch; ++y)
                        {
                            uint8_t*       row = data + static_cast<size_t>(y) * static_cast<size_t>(w) * 4u;
                            const uint8_t* in  = src + static_cast<size_t>(y) * cw * 4u;
                            std::memcpy(row, in, static_cast<size_t>(cw) * 4u);
                            if (rgba)
                                for (uint32_t x = 0; x < cw; ++x)
                                {
                                    const uint8_t t = row[x * 4u];
                                    row[x * 4u]      = row[x * 4u + 2u];
                                    row[x * 4u + 2u] = t;
                                }
                        }
                        ok = true;
                        ++g.readPixelsDone;
                        g.readPixelsBytes += bytes;
                    }
                }
            }
        }
        if (!ok)
        {
            ++g.readRefused;
            std::memset(data, 0xFF, static_cast<size_t>(count) * 4u);
        }
    }

    void LogTargetStats()
    {
        present::Stats f = {};
        present::GetStats(f);
        wxl::ScriptMgr::Log(WXL_LOG_INFO,
                            "native: targets: %s; %llu pass(es) begun (%llu on a texture target), %u "
                            "pass class(es), %u framebuffer(s) live (%llu made), offscreen depth grown %llu "
                            "time(s), %llu pass request(s) dropped for a target without attachment; slot 23 "
                            "%llu call(s) (%llu unchanged): %llu texture / %llu back buffer, %llu with a depth "
                            "texture, %llu without attachment; slot 26 %llu, 24 / 25 %llu / %llu; G-buffer: %llu "
                            "pass(es) of its class, %llu class switch(es)",
                            f.ownBack ? "own back buffer" : "direct path (no own back buffer)",
                            static_cast<unsigned long long>(f.passes),
                            static_cast<unsigned long long>(f.targetPasses), f.passClasses, f.framebuffers,
                            static_cast<unsigned long long>(f.framebuffersMade),
                            static_cast<unsigned long long>(f.offscreenGrows),
                            static_cast<unsigned long long>(f.noTarget),
                            static_cast<unsigned long long>(g.setCalls),
                            static_cast<unsigned long long>(g.setUnchanged),
                            static_cast<unsigned long long>(g.setTexture),
                            static_cast<unsigned long long>(g.setBack),
                            static_cast<unsigned long long>(g.setDepthTexture),
                            static_cast<unsigned long long>(g.setNoAttach),
                            static_cast<unsigned long long>(g.depthResolves),
                            static_cast<unsigned long long>(g.depthSaves),
                            static_cast<unsigned long long>(g.depthRestores),
                            static_cast<unsigned long long>(f.gbufferPasses),
                            static_cast<unsigned long long>(f.gbufferSwitches));
        wxl::ScriptMgr::Log(WXL_LOG_INFO,
                            "native: copies: slot 27 %llu, slot 59 %llu, slot 60 %llu (%llu by blit), "
                            "%llu refused; slot 20 %llu call(s): %llu read back (%llu KiB), %llu answered "
                            "0xFF; %llu mid-frame flush(es) (%llu refused on the direct path); %llu present "
                            "copies",
                            static_cast<unsigned long long>(g.copyTex),
                            static_cast<unsigned long long>(g.texCopy),
                            static_cast<unsigned long long>(g.texStretch),
                            static_cast<unsigned long long>(g.blits),
                            static_cast<unsigned long long>(g.copyRefused),
                            static_cast<unsigned long long>(g.readPixels),
                            static_cast<unsigned long long>(g.readPixelsDone),
                            static_cast<unsigned long long>(g.readPixelsBytes / 1024),
                            static_cast<unsigned long long>(g.readRefused),
                            static_cast<unsigned long long>(f.flushes),
                            static_cast<unsigned long long>(f.flushRefused),
                            static_cast<unsigned long long>(f.presentCopies));
    }
}
