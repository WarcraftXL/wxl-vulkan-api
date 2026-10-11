// The engine's texture formats as Vulkan sees them: the VkFormat each one is, the size of a level,
// the length of a mip chain, and a decoder to RGBA8 for the ones a display path cannot show as is.
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

#include "vulkan/TexFormat.hpp"

// The enum values and their count. Included directly, as src/device/Slots.hpp does:
// wxl/game/GxDevice.hpp only aliases this header's namespace, so going through it would add a
// dependency on the SDK's binding helpers for no insulation.
#include "wxl/offsets/engine/GxDevice.hpp"

#include <cstddef>
#include <cstdint>
#include <cstring>

namespace
{
    namespace gxdev = wxl::offsets::engine::gxdevice;

    using wxlvk::vk::texformat::Info;

    // --- the table ----------------------------------------------------------------------------------
    //
    // One row per EGxTexFormat, in enum order, each worked out from the D3DFORMAT the engine's
    // s_GxTexFmtToD3dFmt maps it to. The reasoning is written out per row because the two APIs name
    // formats by the same convention and it is easy to assume that means the same names match:
    //
    //  - D3D9 names a format most-significant-first within its word. D3DFMT_A8R8G8B8 is the dword
    //    0xAARRGGBB; on this little-endian machine its BYTES in memory are B, G, R, A.
    //  - Vulkan's unpacked formats (R8G8B8A8, B8G8R8A8, R8G8, R16G16 ...) name components in MEMORY
    //    order, so the dword above is B8G8R8A8, not A8R8G8B8.
    //  - Vulkan's _PACK16 formats name components most-significant-first within a 16-bit word --
    //    the same convention as D3D -- so for those the question is whether the bit positions
    //    coincide, and for one of the engine's three (A4R4G4B4) they do not in any 1.0 core format.
    //
    // UNORM throughout, never sRGB: see the header. BC1/2/3 are only legal when the device was
    // created with textureCompressionBC, which is optional in 1.0 -- the context asks for it when
    // the adapter offers it, and `needsDecode` is NOT set for them, because whether they need
    // decoding is a property of the device, not of the format; the caller checks the feature and
    // calls DecodeToRgba8 itself (src/native/Textures.cpp).
    //
    // Aggregate order: format, blockWidth, blockHeight, blockBytes, compressed, needsDecode,
    // uploadable, name. Positional rather than designated so the file compiles as C++17 too, which
    // is what tests/texformat/ builds it as.
    constexpr Info kTable[] =
    {
        // 0 Unknown: D3DFMT_UNKNOWN. Nothing to create, nothing to upload.
        { VK_FORMAT_UNDEFINED, 1, 1, 0, false, false, false, "Unknown" },

        // 1 Abgr8888: D3DFMT_A8B8G8R8, the dword 0xAABBGGRR; bytes in memory R, G, B, A. That is
        //   Vulkan's memory-order R8G8B8A8. The format every BLP decodes to, and the one this
        //   decoder emits.
        { VK_FORMAT_R8G8B8A8_UNORM, 1, 1, 4, false, false, true, "Abgr8888" },

        // 2 Argb8888: D3DFMT_A8R8G8B8, the dword 0xAARRGGBB; bytes in memory B, G, R, A, which is
        //   Vulkan's B8G8R8A8 -- also the swapchain's usual format, so an upload in it blits with
        //   no conversion.
        { VK_FORMAT_B8G8R8A8_UNORM, 1, 1, 4, false, false, true, "Argb8888" },

        // 3 Argb4444: D3DFMT_A4R4G4B4, a 16-bit word with A in bits 15..12, R 11..8, G 7..4,
        //   B 3..0. Vulkan 1.0 core has two 4444 PACK16 formats and both put A in bits 3..0:
        //   R4G4B4A4 is R 15..12, G 11..8, B 7..4, A 3..0 and B4G4R4A4 is B 15..12, G 11..8,
        //   R 7..4, A 3..0. The one with A at the top, VK_FORMAT_A4R4G4B4_UNORM_PACK16, is
        //   VK_EXT_4444_formats, promoted in 1.3 -- not 1.0 core -- so this row is UNDEFINED and
        //   needsDecode.
        //
        //   Rejected alternative: upload as R4G4B4A4 and fix it with an image-view swizzle. Read as
        //   R4G4B4A4, the D3D word samples as (A, R, G, B), and a view swizzle of (G, B, A, R)
        //   does put it right -- but Info has nowhere to carry a swizzle, a copy of the image would
        //   not carry it either, and the engine only ever stores 4444 as a downgrade on an adapter
        //   without 8888 (textures are created in dataFormat, which is never 4444 from a BLP). A
        //   decode of the rare case costs less than a second table every consumer has to remember.
        { VK_FORMAT_UNDEFINED, 1, 1, 2, false, true, true, "Argb4444" },

        // 4 Argb1555: D3DFMT_A1R5G5B5, A in bit 15, R 14..10, G 9..5, B 4..0.
        //   VK_FORMAT_A1R5G5B5_UNORM_PACK16 is A in bit 15, R 10..14, G 5..9, B 0..4: identical.
        //   Not B5G5R5A1 or R5G5B5A1, which both put A in bit 0.
        { VK_FORMAT_A1R5G5B5_UNORM_PACK16, 1, 1, 2, false, false, true, "Argb1555" },

        // 5 Rgb565: D3DFMT_R5G6B5, R in bits 15..11, G 10..5, B 4..0.
        //   VK_FORMAT_R5G6B5_UNORM_PACK16 is R 11..15, G 5..10, B 0..4: identical. Not B5G6R5,
        //   which puts B at the top.
        { VK_FORMAT_R5G6B5_UNORM_PACK16, 1, 1, 2, false, false, true, "Rgb565" },

        // 6 Dxt1: D3DFMT_DXT1, 8-byte 4x4 blocks. BC1_RGBA rather than BC1_RGB because D3D9's DXT1
        //   has the colour-3-is-transparent rule in its c0 <= c1 mode and the engine relies on it
        //   for 1-bit-alpha BLPs; BC1_RGB decodes that same index as opaque black.
        { VK_FORMAT_BC1_RGBA_UNORM_BLOCK, 4, 4, 8, true, false, true, "Dxt1" },

        // 7 Dxt3: D3DFMT_DXT3, 16-byte blocks: 8 of explicit 4-bit alpha, 8 of BC1 colour.
        { VK_FORMAT_BC2_UNORM_BLOCK, 4, 4, 16, true, false, true, "Dxt3" },

        // 8 Dxt5: D3DFMT_DXT5, 16-byte blocks: 8 of interpolated alpha, 8 of BC1 colour.
        { VK_FORMAT_BC3_UNORM_BLOCK, 4, 4, 16, true, false, true, "Dxt5" },

        // 9 Uv88: D3DFMT_V8U8, the word with V in bits 15..8 and U in 7..0; bytes in memory U
        //   then V. Both SIGNED -- D3D samples V8U8 as a signed normalised bump-map delta, so the
        //   row is R8G8_SNORM with R = U and G = V, not R8G8_UNORM, which would shift every value
        //   by half the range.
        { VK_FORMAT_R8G8_SNORM, 1, 1, 2, false, false, true, "Uv88" },

        // 10 Gr1616F: D3DFMT_G16R16F, the dword with G in bits 31..16 and R in 15..0; in memory
        //    the R half-float then the G half-float, which is R16G16_SFLOAT.
        { VK_FORMAT_R16G16_SFLOAT, 1, 1, 4, false, false, true, "Gr1616F" },

        // 11 R32F: D3DFMT_R32F, one IEEE single.
        { VK_FORMAT_R32_SFLOAT, 1, 1, 4, false, false, true, "R32F" },

        // 12 D24X8: D3DFMT_D24X8, depth in bits 23..0 and 8 unused bits above it, which is
        //    VK_FORMAT_X8_D24_UNORM_PACK32 (D in 0..23, X in 24..31). Named so a depth texture can
        //    be created in it, but never uploaded -- the engine only ever renders into it -- and
        //    optional in 1.0: only D16 and one of D24S8 / D32S8 are guaranteed, so a caller checks
        //    vkGetPhysicalDeviceFormatProperties first.
        { VK_FORMAT_X8_D24_UNORM_PACK32, 1, 1, 4, false, false, false, "D24X8" },
    };

    constexpr uint32_t kRows = static_cast<uint32_t>(sizeof kTable / sizeof kTable[0]);
    static_assert(kRows == gxdev::kTexFormatCount, "one row per EGxTexFormat, in enum order");

    /// constexpr strcmp, for the row-name checks below.
    constexpr bool SameName(const char* a, const char* b)
    {
        while (*a && *a == *b) { ++a; ++b; }
        return *a == *b;
    }

    // The enum and the table are maintained apart, so pin a few rows to their enumerators: an
    // inserted row would otherwise shift every format after it and nothing would notice until a
    // DXT texture was uploaded as 2 bytes per texel.
    static_assert(SameName(kTable[gxdev::kTexFmtUnknown].name,  "Unknown"),  "row 0");
    static_assert(SameName(kTable[gxdev::kTexFmtAbgr8888].name, "Abgr8888"), "row 1");
    static_assert(SameName(kTable[gxdev::kTexFmtArgb4444].name, "Argb4444"), "row 3");
    static_assert(SameName(kTable[gxdev::kTexFmtDxt1].name,     "Dxt1"),     "row 6");
    static_assert(SameName(kTable[gxdev::kTexFmtDxt5].name,     "Dxt5"),     "row 8");
    static_assert(SameName(kTable[gxdev::kTexFmtU8V8].name,     "Uv88"),     "row 9");
    static_assert(SameName(kTable[gxdev::kTexFmtD24X8].name,    "D24X8"),    "row 12");
    static_assert(kTable[gxdev::kTexFmtDxt1].blockBytes == 8 && kTable[gxdev::kTexFmtDxt3].blockBytes == 16 &&
                  kTable[gxdev::kTexFmtDxt5].blockBytes == 16, "DXT block sizes");
    static_assert(!kTable[gxdev::kTexFmtUnknown].uploadable && !kTable[gxdev::kTexFmtD24X8].uploadable,
                  "Unknown and D24X8 are never uploaded");

    // --- integer helpers ------------------------------------------------------------------------------

    /// ceil(n / 4), the block count along one axis.
    constexpr uint32_t Blocks4(uint32_t n) { return (n + 3) / 4; }

    /// Widens an n-bit UNORM value to 8 bits by bit replication: the top bits are repeated into the
    /// low bits, so 0 maps to 0 and the all-ones value maps to 255 exactly. A plain shift maps the
    /// maximum 5-bit value to 248 and the maximum 6-bit one to 252, which is a visible grey tint
    /// on every white texel and is what hardware does not do.
    constexpr uint8_t Expand5(uint32_t v) { return static_cast<uint8_t>((v << 3) | (v >> 2)); }
    constexpr uint8_t Expand6(uint32_t v) { return static_cast<uint8_t>((v << 2) | (v >> 4)); }
    constexpr uint8_t Expand4(uint32_t v) { return static_cast<uint8_t>((v << 4) | v); }

    /// A little-endian 16-bit word, read without alignment assumptions: a block row inside a
    /// caller's buffer at an odd pitch is not guaranteed 2-aligned.
    inline uint32_t Read16(const uint8_t* p)
    {
        return static_cast<uint32_t>(p[0]) | (static_cast<uint32_t>(p[1]) << 8);
    }

    inline uint32_t Read32(const uint8_t* p)
    {
        return Read16(p) | (Read16(p + 2) << 16);
    }

    /// One RGB565 word to R, G, B bytes.
    inline void Unpack565(uint32_t c, uint8_t* rgb)
    {
        rgb[0] = Expand5((c >> 11) & 0x1F);
        rgb[1] = Expand6((c >> 5) & 0x3F);
        rgb[2] = Expand5(c & 0x1F);
    }

    // --- float helpers, for the two formats the decoder tone-maps ------------------------------------

    /// IEEE binary16 to binary32, bit-exact including subnormals, infinities and NaN. Written out
    /// rather than reached for in a library because nothing else in src/vulkan/ needs a half.
    inline float HalfToFloat(uint32_t h)
    {
        const uint32_t sign = (h >> 15) & 1;
        uint32_t exp  = (h >> 10) & 0x1F;
        uint32_t mant = h & 0x3FF;
        uint32_t bits;
        if (exp == 0)
        {
            if (mant == 0)
            {
                bits = sign << 31;                                 // +-0
            }
            else
            {
                // Subnormal: renormalise so the single-precision exponent carries the shift.
                exp = 127 - 15 + 1;
                while ((mant & 0x400) == 0) { mant <<= 1; --exp; }
                mant &= 0x3FF;
                bits = (sign << 31) | (exp << 23) | (mant << 13);
            }
        }
        else if (exp == 31)
        {
            bits = (sign << 31) | 0x7F800000u | (mant << 13);     // inf / NaN, payload kept
        }
        else
        {
            bits = (sign << 31) | ((exp + (127 - 15)) << 23) | (mant << 13);
        }
        float f;
        std::memcpy(&f, &bits, sizeof f);
        return f;
    }

    /// The crude tone map the header promises: clamp to [0, 1] and quantise. Values above 1 (an
    /// HDR render target) saturate, negatives and NaN go to 0. Anything smarter -- a Reinhard
    /// curve, an exposure -- would change the one range that matters on screen, 0..1, and
    /// make a 0.5 look like something other than mid-grey.
    inline uint8_t ToByte(float x)
    {
        if (!(x > 0.0f)) return 0;        // the negated test is what also catches NaN
        if (x >= 1.0f)   return 255;
        return static_cast<uint8_t>(x * 255.0f + 0.5f);
    }

    // --- block decoders -------------------------------------------------------------------------------
    //
    // Each decodes one whole 4x4 block into `texels`, 16 RGBA quads in raster order, and the caller
    // clips. Decoding whole blocks and clipping is simpler than teaching each decoder a partial
    // extent, and costs nothing that matters: a partial block exists only on the right and bottom
    // edges of a level whose size is not a multiple of 4.

    /// The four BC1 colours for a pair of 565 endpoints.
    ///
    /// Interpolants are rounded to nearest ((2a + b + 1) / 3, (a + b + 1) / 2), which is what a
    /// float lerp followed by a round gives and what the reference decoders in DirectXTex do. The
    /// specs tolerate a +-1 difference between implementations here, so a truncating decoder is
    /// not wrong, only different; the tests pick endpoints where both agree.
    ///
    /// @param bool fourColourAlways : BC2 and BC3 colour blocks are always in the four-colour mode,
    ///        whatever the endpoint order says -- the c0 <= c1 rule exists only in BC1, where it is
    ///        the one place a 1-bit alpha can be carried
    inline void Bc1Palette(uint32_t c0, uint32_t c1, bool fourColourAlways, uint8_t palette[4][4])
    {
        Unpack565(c0, palette[0]);
        Unpack565(c1, palette[1]);
        palette[0][3] = 255;
        palette[1][3] = 255;

        if (c0 > c1 || fourColourAlways)
        {
            for (int i = 0; i < 3; ++i)
            {
                const uint32_t a = palette[0][i];
                const uint32_t b = palette[1][i];
                palette[2][i] = static_cast<uint8_t>((2 * a + b + 1) / 3);
                palette[3][i] = static_cast<uint8_t>((a + 2 * b + 1) / 3);
            }
            palette[2][3] = 255;
            palette[3][3] = 255;
        }
        else
        {
            for (int i = 0; i < 3; ++i)
            {
                const uint32_t a = palette[0][i];
                const uint32_t b = palette[1][i];
                palette[2][i] = static_cast<uint8_t>((a + b + 1) / 2);
                palette[3][i] = 0;
            }
            palette[2][3] = 255;
            palette[3][3] = 0;        // colour 3 is transparent black: the 1-bit alpha
        }
    }

    /// The 8-byte BC1 colour block at `p`: two 565 endpoints, then 16 two-bit indices, texel 0 in
    /// the low bits of the first index byte.
    inline void DecodeColourBlock(const uint8_t* p, bool fourColourAlways, uint8_t texels[16][4])
    {
        uint8_t palette[4][4];
        Bc1Palette(Read16(p), Read16(p + 2), fourColourAlways, palette);
        const uint32_t indices = Read32(p + 4);
        for (uint32_t i = 0; i < 16; ++i)
        {
            const uint8_t* c = palette[(indices >> (2 * i)) & 3];
            texels[i][0] = c[0];
            texels[i][1] = c[1];
            texels[i][2] = c[2];
            texels[i][3] = c[3];
        }
    }

    void DecodeBc1(const uint8_t* p, uint8_t texels[16][4])
    {
        DecodeColourBlock(p, false, texels);
    }

    /// BC2: 8 bytes of 4-bit alpha, one 16-bit word per row with texel 0 in the low nibble, then
    /// the colour block.
    void DecodeBc2(const uint8_t* p, uint8_t texels[16][4])
    {
        DecodeColourBlock(p + 8, true, texels);
        for (uint32_t y = 0; y < 4; ++y)
        {
            const uint32_t row = Read16(p + 2 * y);
            for (uint32_t x = 0; x < 4; ++x)
                texels[4 * y + x][3] = Expand4((row >> (4 * x)) & 0xF);
        }
    }

    /// BC3: two alpha endpoints, 48 bits of 3-bit indices (texel 0 in the low bits), then the
    /// colour block. a0 > a1 selects eight interpolated alphas; otherwise six plus 0 and 255, the
    /// alpha analogue of BC1's two modes.
    void DecodeBc3(const uint8_t* p, uint8_t texels[16][4])
    {
        DecodeColourBlock(p + 8, true, texels);

        const uint32_t a0 = p[0];
        const uint32_t a1 = p[1];
        uint8_t alpha[8];
        alpha[0] = static_cast<uint8_t>(a0);
        alpha[1] = static_cast<uint8_t>(a1);
        if (a0 > a1)
        {
            for (uint32_t i = 1; i < 7; ++i)
                alpha[i + 1] = static_cast<uint8_t>(((7 - i) * a0 + i * a1 + 3) / 7);
        }
        else
        {
            for (uint32_t i = 1; i < 5; ++i)
                alpha[i + 1] = static_cast<uint8_t>(((5 - i) * a0 + i * a1 + 2) / 5);
            alpha[6] = 0;
            alpha[7] = 255;
        }

        // 48 bits as two 24-bit halves, each holding eight indices: a 64-bit read would run two
        // bytes past the block.
        for (uint32_t half = 0; half < 2; ++half)
        {
            const uint8_t* q = p + 2 + 3 * half;
            const uint32_t bits = static_cast<uint32_t>(q[0]) | (static_cast<uint32_t>(q[1]) << 8) |
                                  (static_cast<uint32_t>(q[2]) << 16);
            for (uint32_t i = 0; i < 8; ++i)
                texels[8 * half + i][3] = alpha[(bits >> (3 * i)) & 7];
        }
    }

    /// Copies the texels of one decoded block that fall inside the level.
    inline void WriteBlock(const uint8_t texels[16][4], uint32_t bx, uint32_t by, uint32_t width,
                           uint32_t height, uint8_t* dst)
    {
        for (uint32_t y = 0; y < 4; ++y)
        {
            const uint32_t py = 4 * by + y;
            if (py >= height) break;
            for (uint32_t x = 0; x < 4; ++x)
            {
                const uint32_t px = 4 * bx + x;
                if (px >= width) break;
                std::memcpy(dst + (static_cast<size_t>(py) * width + px) * 4, texels[4 * y + x], 4);
            }
        }
    }

    using BlockFn = void (*)(const uint8_t*, uint8_t[16][4]);

    /// Walks the block rows of a DXT level through `decode`.
    void DecodeBlocks(BlockFn decode, uint32_t blockBytes, uint32_t width, uint32_t height,
                      const uint8_t* src, uint32_t srcPitch, uint8_t* dst)
    {
        const uint32_t bw = Blocks4(width);
        const uint32_t bh = Blocks4(height);
        for (uint32_t by = 0; by < bh; ++by)
        {
            const uint8_t* row = src + static_cast<size_t>(by) * srcPitch;
            for (uint32_t bx = 0; bx < bw; ++bx)
            {
                uint8_t texels[16][4];
                decode(row + static_cast<size_t>(bx) * blockBytes, texels);
                WriteBlock(texels, bx, by, width, height, dst);
            }
        }
    }

    // --- uncompressed row decoders --------------------------------------------------------------------

    using RowFn = void (*)(const uint8_t* src, uint32_t width, uint8_t* dst);

    void RowAbgr8888(const uint8_t* src, uint32_t width, uint8_t* dst)
    {
        std::memcpy(dst, src, static_cast<size_t>(width) * 4);  // already R, G, B, A in memory
    }

    void RowArgb8888(const uint8_t* src, uint32_t width, uint8_t* dst)
    {
        for (uint32_t x = 0; x < width; ++x, src += 4, dst += 4)
        {
            dst[0] = src[2];   // memory is B, G, R, A
            dst[1] = src[1];
            dst[2] = src[0];
            dst[3] = src[3];
        }
    }

    void RowArgb4444(const uint8_t* src, uint32_t width, uint8_t* dst)
    {
        for (uint32_t x = 0; x < width; ++x, src += 2, dst += 4)
        {
            const uint32_t c = Read16(src);
            dst[0] = Expand4((c >> 8) & 0xF);
            dst[1] = Expand4((c >> 4) & 0xF);
            dst[2] = Expand4(c & 0xF);
            dst[3] = Expand4(c >> 12);
        }
    }

    void RowArgb1555(const uint8_t* src, uint32_t width, uint8_t* dst)
    {
        for (uint32_t x = 0; x < width; ++x, src += 2, dst += 4)
        {
            const uint32_t c = Read16(src);
            dst[0] = Expand5((c >> 10) & 0x1F);
            dst[1] = Expand5((c >> 5) & 0x1F);
            dst[2] = Expand5(c & 0x1F);
            dst[3] = static_cast<uint8_t>((c & 0x8000) ? 255 : 0);
        }
    }

    void RowRgb565(const uint8_t* src, uint32_t width, uint8_t* dst)
    {
        for (uint32_t x = 0; x < width; ++x, src += 2, dst += 4)
        {
            Unpack565(Read16(src), dst);
            dst[3] = 255;
        }
    }

    /// Signed U, V biased into the unsigned range, a flat normal in B: the usual way to look at a
    /// bump map. The sign is lost, which is one of the reasons the decode is display-only.
    void RowUv88(const uint8_t* src, uint32_t width, uint8_t* dst)
    {
        for (uint32_t x = 0; x < width; ++x, src += 2, dst += 4)
        {
            dst[0] = static_cast<uint8_t>(static_cast<int8_t>(src[0]) + 128);
            dst[1] = static_cast<uint8_t>(static_cast<int8_t>(src[1]) + 128);
            dst[2] = 128;
            dst[3] = 255;
        }
    }

    void RowGr1616F(const uint8_t* src, uint32_t width, uint8_t* dst)
    {
        for (uint32_t x = 0; x < width; ++x, src += 4, dst += 4)
        {
            dst[0] = ToByte(HalfToFloat(Read16(src)));
            dst[1] = ToByte(HalfToFloat(Read16(src + 2)));
            dst[2] = 0;
            dst[3] = 255;
        }
    }

    /// Grey rather than red: a single channel shown as red reads as "something is wrong with the
    /// texture", a grey ramp reads as the depth or height it usually is.
    void RowR32F(const uint8_t* src, uint32_t width, uint8_t* dst)
    {
        for (uint32_t x = 0; x < width; ++x, src += 4, dst += 4)
        {
            const uint32_t bits = Read32(src);
            float f;
            std::memcpy(&f, &bits, sizeof f);
            const uint8_t v = ToByte(f);
            dst[0] = v;
            dst[1] = v;
            dst[2] = v;
            dst[3] = 255;
        }
    }

    /// The uncompressed decoder for a format, or null.
    RowFn RowDecoder(uint32_t gxFormat)
    {
        switch (gxFormat)
        {
            case gxdev::kTexFmtAbgr8888: return RowAbgr8888;
            case gxdev::kTexFmtArgb8888: return RowArgb8888;
            case gxdev::kTexFmtArgb4444: return RowArgb4444;
            case gxdev::kTexFmtArgb1555: return RowArgb1555;
            case gxdev::kTexFmtRgb565:   return RowRgb565;
            case gxdev::kTexFmtU8V8:     return RowUv88;
            case gxdev::kTexFmtG16R16F:  return RowGr1616F;
            case gxdev::kTexFmtR32F:     return RowR32F;
            default:                     return nullptr;
        }
    }

    /// The block decoder for a format, or null.
    BlockFn BlockDecoder(uint32_t gxFormat)
    {
        switch (gxFormat)
        {
            case gxdev::kTexFmtDxt1: return DecodeBc1;
            case gxdev::kTexFmtDxt3: return DecodeBc2;
            case gxdev::kTexFmtDxt5: return DecodeBc3;
            default:                 return nullptr;
        }
    }
}

namespace wxlvk::vk::texformat
{
    const Info& Of(uint32_t gxFormat)
    {
        return kTable[gxFormat < kRows ? gxFormat : gxdev::kTexFmtUnknown];
    }

    uint32_t RowBytes(uint32_t gxFormat, uint32_t width)
    {
        const Info& f = Of(gxFormat);
        if (!f.uploadable || width == 0) return 0;
        return (f.compressed ? Blocks4(width) : width) * f.blockBytes;
    }

    uint32_t RowCount(uint32_t gxFormat, uint32_t height)
    {
        const Info& f = Of(gxFormat);
        if (!f.uploadable || height == 0) return 0;
        return f.compressed ? Blocks4(height) : height;
    }

    uint64_t LevelBytes(uint32_t gxFormat, uint32_t width, uint32_t height)
    {
        return static_cast<uint64_t>(RowBytes(gxFormat, width)) * RowCount(gxFormat, height);
    }

    uint32_t ChainLevels(uint32_t gxFormat, uint32_t width, uint32_t height)
    {
        if (width == 0 || height == 0) return 0;
        const Info& f = Of(gxFormat);

        if (!f.compressed)
        {
            // Levels until the larger side is 1: floor(log2(max)) + 1.
            uint32_t levels = 1;
            for (uint32_t side = width > height ? width : height; side > 1; side >>= 1) ++levels;
            return levels;
        }

        // Levels until the smaller side is 4, the level that still fills a block. The engine never
        // uploads the 2x2 and 1x1 tails of a DXT chain, so counting them would make the image two
        // levels longer than what arrives. Below 4 at level 0 there is still the one level.
        uint32_t levels = 0;
        for (uint32_t side = width < height ? width : height; side >= 4; side >>= 1) ++levels;
        return levels ? levels : 1;
    }

    bool DecodeToRgba8(uint32_t gxFormat, uint32_t width, uint32_t height, const void* src,
                       uint32_t srcPitch, uint8_t* dst)
    {
        if (!src || !dst || width == 0 || height == 0) return false;
        const Info& f = Of(gxFormat);
        if (!f.uploadable) return false;
        // A pitch shorter than a row would read the next row's bytes as this one's; refused rather
        // than read, because the usual way to get here is a pitch of 0 from an uninitialised
        // callback result, and a decode that "works" on that hides the bug.
        if (srcPitch < RowBytes(gxFormat, width)) return false;

        const auto* bytes = static_cast<const uint8_t*>(src);

        if (f.compressed)
        {
            const BlockFn decode = BlockDecoder(gxFormat);
            if (!decode) return false;
            DecodeBlocks(decode, f.blockBytes, width, height, bytes, srcPitch, dst);
            return true;
        }

        const RowFn row = RowDecoder(gxFormat);
        if (!row) return false;
        for (uint32_t y = 0; y < height; ++y)
            row(bytes + static_cast<size_t>(y) * srcPitch, width, dst + static_cast<size_t>(y) * width * 4);
        return true;
    }
}
