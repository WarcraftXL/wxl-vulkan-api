// The caps block (+0x214, 0x13C bytes): the native fill.
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

#include "native/Caps.hpp"

#include "vulkan/Context.hpp"
#include "vulkan/Vk.hpp"

#include "wxl/Script.hpp"
#include "wxl/game/Binding.hpp"
#include "wxl/offsets/engine/GxDevice.hpp"

#include <cstdint>
#include <cstring>

// Where each value of kD3dReference comes from. ISetCaps (0x0068EE20) was read whole; for each dword
// it either writes a constant (READ below), derives it from D3DCAPS9 / CheckDeviceFormat on the
// adapter (GUESS: what an NVIDIA RTX-class D3D9 driver reports), or does not write it at all,
// leaving the base constructor's zero (READ: not written).
//
//   0x214  texture units        min(8, MaxSimultaneousTextures)        GUESS 8
//   0x218                       constant 0                             READ
//   0x21C                       constant 1                             READ
//   0x220  vertex streams       MaxStreams                             GUESS 16
//   0x224  stream-offset mode   Caps2 & 1                              READ in effect: no D3DCAPS2 bit 0, so 0
//   0x228, 0x22C                not written by ISetCaps                READ 0 (the ctor's "api flags")
//   0x230  MaxVertexIndex       MaxVertexIndex                         GUESS 0xFFFFFF
//   0x234  auto-gen mips        Caps2 CANAUTOGENMIPMAP                 GUESS 1
//   0x238  format 0             constant 0                             READ
//   0x23C..0x268 formats 1-12   CheckDeviceFormat(texture)             GUESS 1 each
//   0x26C  2D                   constant 1                             READ
//   0x270  cube                 TextureCaps CUBEMAP                    GUESS 1
//   0x274                       constant 0                             READ
//   0x278  NPOT                 !(POW2 && !NONPOW2CONDITIONAL)          GUESS 1
//   0x27C  NPOT conditional     NONPOW2CONDITIONAL, forced 0 on NVIDIA READ 0 on this (NVIDIA) machine
//   0x280..0x28C max sizes      max(256, MaxTextureWidth), all four    GUESS 16384
//   0x290..0x2C4 RT formats     CheckDeviceFormat(render target) for f2, f5, f10, f11, (depth) f12;
//                               the others not written                 GUESS 1 for those five, READ 0 others
//   0x2C8..0x2DC levels         vs 0..3 at +0x2C8, ps 0..4 at +0x2D8, the rest 0
//                                                                      READ 3 / 4 (the client runs ps_3_0)
//   0x2E0  vs constants         MaxVertexShaderConst (0xC0 on GeForce FX)
//                                                                      GUESS 256
//   0x2E4..0x2F4                zeroed; ISetCaps never writes a pixel constant count
//                                                                      READ 0
//   0x2F8  trilinear            TextureFilterCaps MIPFLINEAR           GUESS 1
//   0x2FC  anisotropic          TextureFilterCaps MIN|MAGFANISOTROPIC, 0 if MaxAnisotropy < 2
//                                                                      GUESS 1
//   0x300  max anisotropy       MaxAnisotropy                          GUESS 16
//   0x304  depth bias           RasterCaps DEPTHBIAS                   GUESS 1
//   0x308  colour write         PrimitiveMiscCaps COLORWRITEENABLE     GUESS 1
//   0x30C  user clip planes     MaxUserClipPlanes                      GUESS 8
//   0x310  hardware cursor      CursorCaps & 1                         GUESS 1
//   0x314  occlusion query      CreateQuery(OCCLUSION) succeeds        GUESS 1
//   0x318 / 0x320 points        MaxPointSize > 1                       GUESS 1 / 1
//   0x31C  max point size       MaxPointSize (float)                   GUESS 8192.0
//   0x324  blend factor         SrcBlendCaps BLENDFACTOR               GUESS 1
//   0x328..0x338                constant 0                             READ
//   0x33C, 0x340                not written                            READ 0
//   0x344                       constant 1                             READ
//   0x348 / 0x34C               (ps level != 4)                        READ, derived
//
// Fill() then deviates from that table on purpose, one commented line per field.
namespace
{
    namespace gxdev   = wxl::offsets::engine::gxdevice;
    namespace context = wxlvk::vk::context;

    using wxl::game::At;

    constexpr size_t   kCapsBytes = 0x13C;
    constexpr unsigned kCapsWords = kCapsBytes / 4;   // 79
    static_assert(gxdev::kCapsBase + kCapsBytes == 0x350, "the caps block ends where kBaseMipLevel starts");

    /// What D3D writes on an NVIDIA RTX-class adapter, by dword from +0x214: one row of four per
    /// line, the offset of the first in the comment. See the provenance table above for which
    /// entries are read and which are guesses.
    constexpr uint32_t kD3dReference[kCapsWords] = {
        /* 0x214 */ 0x00000008u, 0x00000000u, 0x00000001u, 0x00000010u,
        /* 0x224 */ 0x00000000u, 0x00000000u, 0x00000001u, 0x00FFFFFFu,
        /* 0x234 */ 0x00000001u, 0x00000000u, 0x00000000u, 0x00000001u,
        /* 0x244 */ 0x00000001u, 0x00000001u, 0x00000001u, 0x00000001u,
        /* 0x254 */ 0x00000001u, 0x00000001u, 0x00000001u, 0x00000001u,
        /* 0x264 */ 0x00000001u, 0x00000001u, 0x00000001u, 0x00000001u,
        /* 0x274 */ 0x00000000u, 0x00000001u, 0x00000000u, 0x00004000u,
        /* 0x284 */ 0x00004000u, 0x00004000u, 0x00004000u, 0x00000000u,
        /* 0x294 */ 0x00000000u, 0x00000001u, 0x00000000u, 0x00000000u,
        /* 0x2A4 */ 0x00000001u, 0x00000000u, 0x00000000u, 0x00000000u,
        /* 0x2B4 */ 0x00000000u, 0x00000001u, 0x00000001u, 0x00000001u,
        /* 0x2C4 */ 0x00000000u, 0x00000003u, 0x00000000u, 0x00000000u,
        /* 0x2D4 */ 0x00000000u, 0x00000004u, 0x00000000u, 0x00000100u,
        /* 0x2E4 */ 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u,
        /* 0x2F4 */ 0x00000000u, 0x00000001u, 0x00000001u, 0x00000010u,
        /* 0x304 */ 0x00000001u, 0x00000001u, 0x00000008u, 0x00000001u,
        /* 0x314 */ 0x00000001u, 0x00000001u, 0x46000000u, 0x00000001u,
        /* 0x324 */ 0x00000001u, 0x00000000u, 0x00000000u, 0x00000000u,
        /* 0x334 */ 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u,
        /* 0x344 */ 0x00000001u, 0x00000000u, 0x00000000u,
    };

    /// The dword at a device offset inside the caps block.
    uint32_t& Word(void* device, size_t offset) { return At<uint32_t>(device, offset); }

    /// True when the format can be a sampled colour attachment with optimal tiling.
    bool RenderableColour(const context::Handles& h, VkFormat format)
    {
        VkFormatProperties props = {};
        h.iapi->GetPhysicalDeviceFormatProperties(h.physical, format, &props);
        const VkFormatFeatureFlags need =
            VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT | VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT;
        return (props.optimalTilingFeatures & need) == need;
    }

    /// True when the format can be a sampled depth attachment with optimal tiling.
    bool RenderableDepth(const context::Handles& h, VkFormat format)
    {
        VkFormatProperties props = {};
        h.iapi->GetPhysicalDeviceFormatProperties(h.physical, format, &props);
        const VkFormatFeatureFlags need =
            VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT;
        return (props.optimalTilingFeatures & need) == need;
    }

    uint32_t Min(uint32_t a, uint32_t b) { return a < b ? a : b; }

    /// Bit pattern of a float, for the one float field.
    uint32_t Bits(float f)
    {
        uint32_t u = 0;
        std::memcpy(&u, &f, sizeof u);
        return u;
    }

    float Float(uint32_t u)
    {
        float f = 0.0f;
        std::memcpy(&f, &u, sizeof f);
        return f;
    }
}

namespace wxlvk::native::caps
{
    void Fill(void* device)
    {
        // The reference first, whole: every field the code below does not mention keeps D3D's value.
        std::memcpy(static_cast<uint8_t*>(device) + gxdev::kCapsBase, kD3dReference, kCapsBytes);

        const context::Handles* h = context::Get();
        VkPhysicalDeviceProperties props = {};
        if (h) h->iapi->GetPhysicalDeviceProperties(h->physical, &props);

        // --- deliberate deviations, one per line -------------------------------------------------

        // Shader levels: vs_3_0 and ps_3_0, always. The translated modules exist for the engine's
        // vs_3_0 / ps_3_0 .bls records, so those are the only profile folders this device can draw;
        // the CGxFormat's max-profile fields, which would lower a level on D3D, are not applied.
        const int32_t vs = static_cast<int32_t>(gxdev::kVertexProfileVs30);
        const int32_t ps = static_cast<int32_t>(gxdev::kPixelProfilePs30);
        Word(device, gxdev::kCapsShaderTarget + gxdev::kShTargetVertex * 4u) = static_cast<uint32_t>(vs);
        Word(device, gxdev::kCapsShaderTarget + gxdev::kShTargetPixel * 4u)  = static_cast<uint32_t>(ps);
        // ISetCaps' last act, `caps 0x348 = caps 0x34C = (pixelLevel != 4)` (see gxdev::kCaps348):
        // 0 for ps_3_0.
        Word(device, gxdev::kCaps348) = 0;
        Word(device, gxdev::kCaps34C) = 0;

        // Clip planes: not implemented, so not offered; with 0 the engine takes no clip-plane path.
        // The hardware cursor (src/native/Cursor.cpp: slots 2, 4, 73): offered, as D3D's CursorCaps
        // does, so the format's gxCursor byte decides, as on D3D. Occlusion queries
        // (src/native/Queries.cpp): offered, so the sun / moon glare (DNGlare) runs, as D3D's
        // CreateQuery(OCCLUSION) succeeding offers them. Multisampling is not offered: the device
        // renders single-sampled, and anti-aliasing is left to a post-process.
        Word(device, gxdev::kCapsMaxClipPlanes)   = 0;
        Word(device, gxdev::kCapsHwCursor)        = 1;
        Word(device, gxdev::kCapsOcclusionQuery)  = 1;

        if (h)
        {
            // Texture sizes: D3D's value, capped by what a VkImage can be.
            const uint32_t max2D   = props.limits.maxImageDimension2D;
            const uint32_t maxCube = props.limits.maxImageDimensionCube;
            Word(device, gxdev::kCapsMaxTexWidth)  = Min(Word(device, gxdev::kCapsMaxTexWidth), max2D);
            Word(device, gxdev::kCapsMaxTexHeight) = Min(Word(device, gxdev::kCapsMaxTexHeight), max2D);
            Word(device, gxdev::kCapsMaxCubeSize)  = Min(Word(device, gxdev::kCapsMaxCubeSize), maxCube);
            Word(device, gxdev::kCapsMaxTexExtent) = Min(Word(device, gxdev::kCapsMaxTexExtent), max2D);

            // Anisotropy: the limit is a property of the adapter whether or not the feature is on.
            // The feature itself is not enabled on our VkDevice (filter 5 samples as trilinear,
            // src/vulkan/Pipelines.cpp), but reporting 0 would change what the ENGINE does:
            // GxTexCreate refuses a filter-5 texture outright when +0x2FC is 0 (0x00681CB0).
            const uint32_t aniso = static_cast<uint32_t>(props.limits.maxSamplerAnisotropy);
            Word(device, gxdev::kCapsMaxAnisotropy) = Min(Word(device, gxdev::kCapsMaxAnisotropy),
                                                          aniso ? aniso : 1u);
            if (Word(device, gxdev::kCapsMaxAnisotropy) < 2) Word(device, gxdev::kCapsTexFilterAniso) = 0;

            // Points: from the device's own range, and 1.0 when large points are off (Context
            // enables the feature whenever the adapter has it).
            const float maxPoint = h->largePoints ? props.limits.pointSizeRange[1] : 1.0f;
            Word(device, gxdev::kCapsMaxPointSize)     = Bits(maxPoint);
            Word(device, gxdev::kCapsPointParameters)  = maxPoint > 1.0f ? 1u : 0u;
            Word(device, gxdev::kCapsPointSprites)     = maxPoint > 1.0f ? 1u : 0u;

            // Render-target formats: what a VkImage of the matching format can be, asked of the
            // adapter the way D3D asks CheckDeviceFormat. Argb8888 is stored B8G8R8A8 (the back
            // buffer's own), Rgb565 R5G6B5, G16R16F R16G16_SFLOAT, R32F R32_SFLOAT; the depth texture
            // is D24 + stencil, or D32 where D24S8 is not offered (AMD).
            Word(device, gxdev::kCapsRtArgb8888) = RenderableColour(*h, VK_FORMAT_B8G8R8A8_UNORM) ? 1u : 0u;
            Word(device, gxdev::kCapsRtRgb565)   = RenderableColour(*h, VK_FORMAT_R5G6B5_UNORM_PACK16) ? 1u : 0u;
            Word(device, gxdev::kCapsRtRg16f)    = RenderableColour(*h, VK_FORMAT_R16G16_SFLOAT) ? 1u : 0u;
            Word(device, gxdev::kCapsRtR32f)     = RenderableColour(*h, VK_FORMAT_R32_SFLOAT) ? 1u : 0u;
            Word(device, gxdev::kCapsDepthTexD24S8) =
                (RenderableDepth(*h, VK_FORMAT_D24_UNORM_S8_UINT) || RenderableDepth(*h, VK_FORMAT_D32_SFLOAT))
                    ? 1u : 0u;
        }
        // Texture formats 1-12 stay 1 whether or not textureCompressionBC is on: the DXT formats are
        // decoded on upload where the device lacks BC, so the engine may ask for them.

        wxl::ScriptMgr::Log(WXL_LOG_DEBUG,
                            "native: caps: vs %d / ps %d, clip planes 0, hardware cursor 1, occlusion "
                            "1, single-sampled, max texture %u, aniso %u, point size %.1f, RT argb8888 "
                            "%u rgb565 %u rg16f %u r32f %u depth %u",
                            vs, ps, Word(device, gxdev::kCapsMaxTexWidth),
                            Word(device, gxdev::kCapsMaxAnisotropy),
                            static_cast<double>(Float(Word(device, gxdev::kCapsMaxPointSize))),
                            Word(device, gxdev::kCapsRtArgb8888), Word(device, gxdev::kCapsRtRgb565),
                            Word(device, gxdev::kCapsRtRg16f), Word(device, gxdev::kCapsRtR32f),
                            Word(device, gxdev::kCapsDepthTexD24S8));
        if (!h)
            wxl::ScriptMgr::Log(WXL_LOG_WARN,
                                "native: caps: no Vulkan device yet, so the size, anisotropy, point and "
                                "render-target rows are the reference table's unchecked values");
    }
}
