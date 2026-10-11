// Proof of src/vulkan/TexFormat.cpp: hand-built blocks with answers worked out on paper, every
// output byte compared, non-zero exit on the first disagreement.
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

// The extension's shared.cmake takes tests/ out of the DLL's sources. The guard stays as the
// second line of defence: WXL_EXTENSION is the define the loop puts on the DLL target and on
// nothing else (it is what PluginApi.h keys its exports on), so should this file ever be globbed
// into the DLL again it is an empty translation unit there, and the standalone CMakeLists.txt beside
// it, which never defines it, gets the whole program.
#ifndef WXL_EXTENSION

#include "vulkan/TexFormat.hpp"

#include "wxl/offsets/engine/GxDevice.hpp"

#include <cstdint>
#include <cstdio>
#include <cstring>

// Every expected value below was computed by hand from the format specifications, never by running
// the decoder: a test that compares the decoder with itself proves nothing. Where a specification
// leaves rounding to the implementation (the BC1 and BC3 interpolants), the endpoints are chosen so
// that the division is exact and every convention agrees.
namespace
{
    namespace gxdev = wxl::offsets::engine::gxdevice;
    namespace tf    = wxlvk::vk::texformat;

    int g_checks   = 0;
    int g_failures = 0;

    /// One named condition.
    void Check(bool ok, const char* what)
    {
        ++g_checks;
        if (!ok)
        {
            ++g_failures;
            std::printf("FAIL: %s\n", what);
        }
    }

    /// Compares a decoded image with its expected bytes, naming the first differing texel.
    void CheckPixels(const char* what, const uint8_t* got, const uint8_t* want, uint32_t width,
                     uint32_t height)
    {
        ++g_checks;
        const size_t bytes = static_cast<size_t>(width) * height * 4;
        for (size_t i = 0; i < bytes; ++i)
        {
            if (got[i] != want[i])
            {
                ++g_failures;
                const size_t texel = i / 4;
                std::printf("FAIL: %s: texel (%u, %u) channel %u: got %u, want %u\n", what,
                            static_cast<unsigned>(texel % width), static_cast<unsigned>(texel / width),
                            static_cast<unsigned>(i % 4), got[i], want[i]);
                return;
            }
        }
    }

    /// Fills 16 expected texels from a 4-entry palette and a per-texel index list.
    void Expect16(uint8_t out[16][4], const uint8_t palette[4][4], const uint8_t indices[16])
    {
        for (int i = 0; i < 16; ++i) std::memcpy(out[i], palette[indices[i]], 4);
    }

    void Put16(uint8_t* p, uint32_t v)
    {
        p[0] = static_cast<uint8_t>(v & 0xFF);
        p[1] = static_cast<uint8_t>(v >> 8);
    }

    void Put32(uint8_t* p, uint32_t v)
    {
        Put16(p, v & 0xFFFF);
        Put16(p + 2, v >> 16);
    }

    /// Packs 16 two-bit indices, texel 0 in the low bits.
    uint32_t PackIndices2(const uint8_t indices[16])
    {
        uint32_t v = 0;
        for (int i = 0; i < 16; ++i) v |= static_cast<uint32_t>(indices[i] & 3) << (2 * i);
        return v;
    }

    /// Writes 16 three-bit indices into the 6 bytes at p, texel 0 in the low bits.
    void PackIndices3(uint8_t* p, const uint8_t indices[16])
    {
        uint64_t v = 0;
        for (int i = 0; i < 16; ++i) v |= static_cast<uint64_t>(indices[i] & 7) << (3 * i);
        for (int i = 0; i < 6; ++i) p[i] = static_cast<uint8_t>((v >> (8 * i)) & 0xFF);
    }

    // --- the table ------------------------------------------------------------------------------------

    void TestTable()
    {
        const tf::Info& unknown = tf::Of(gxdev::kTexFmtUnknown);
        Check(unknown.format == VK_FORMAT_UNDEFINED && !unknown.uploadable, "Unknown row");
        Check(&tf::Of(gxdev::kTexFormatCount) == &unknown, "out-of-range maps to Unknown");
        Check(&tf::Of(0xFFFFFFFFu) == &unknown, "0xFFFFFFFF maps to Unknown");

        struct Row { uint32_t gx; VkFormat vk; uint32_t blockBytes; bool compressed; bool decode; bool up; const char* name; };
        const Row rows[] =
        {
            { gxdev::kTexFmtAbgr8888, VK_FORMAT_R8G8B8A8_UNORM,         4,  false, false, true,  "Abgr8888" },
            { gxdev::kTexFmtArgb8888, VK_FORMAT_B8G8R8A8_UNORM,         4,  false, false, true,  "Argb8888" },
            { gxdev::kTexFmtArgb4444, VK_FORMAT_UNDEFINED,              2,  false, true,  true,  "Argb4444" },
            { gxdev::kTexFmtArgb1555, VK_FORMAT_A1R5G5B5_UNORM_PACK16,  2,  false, false, true,  "Argb1555" },
            { gxdev::kTexFmtRgb565,   VK_FORMAT_R5G6B5_UNORM_PACK16,    2,  false, false, true,  "Rgb565" },
            { gxdev::kTexFmtDxt1,     VK_FORMAT_BC1_RGBA_UNORM_BLOCK,   8,  true,  false, true,  "Dxt1" },
            { gxdev::kTexFmtDxt3,     VK_FORMAT_BC2_UNORM_BLOCK,        16, true,  false, true,  "Dxt3" },
            { gxdev::kTexFmtDxt5,     VK_FORMAT_BC3_UNORM_BLOCK,        16, true,  false, true,  "Dxt5" },
            { gxdev::kTexFmtU8V8,     VK_FORMAT_R8G8_SNORM,             2,  false, false, true,  "Uv88" },
            { gxdev::kTexFmtG16R16F,  VK_FORMAT_R16G16_SFLOAT,          4,  false, false, true,  "Gr1616F" },
            { gxdev::kTexFmtR32F,     VK_FORMAT_R32_SFLOAT,             4,  false, false, true,  "R32F" },
            { gxdev::kTexFmtD24X8,    VK_FORMAT_X8_D24_UNORM_PACK32,    4,  false, false, false, "D24X8" },
        };
        for (const Row& r : rows)
        {
            const tf::Info& f = tf::Of(r.gx);
            char what[96];
            std::snprintf(what, sizeof what, "table row %s", r.name);
            Check(f.format == r.vk && f.blockBytes == r.blockBytes && f.compressed == r.compressed &&
                  f.needsDecode == r.decode && f.uploadable == r.up && std::strcmp(f.name, r.name) == 0 &&
                  f.blockWidth == (r.compressed ? 4u : 1u) && f.blockHeight == (r.compressed ? 4u : 1u),
                  what);
        }
        // Every format that has a VkFormat is a 1.0 core enumerator: the core range ends at 184.
        for (uint32_t i = 0; i < gxdev::kTexFormatCount; ++i)
            Check(static_cast<int>(tf::Of(i).format) <= 184, "VkFormat is 1.0 core");
    }

    // --- sizes and chains -----------------------------------------------------------------------------

    void TestSizes()
    {
        Check(tf::RowBytes(gxdev::kTexFmtArgb8888, 7) == 28, "Argb8888 row of 7");
        Check(tf::RowCount(gxdev::kTexFmtArgb8888, 5) == 5, "Argb8888 5 rows");
        Check(tf::LevelBytes(gxdev::kTexFmtArgb8888, 7, 5) == 140, "Argb8888 7x5");
        Check(tf::LevelBytes(gxdev::kTexFmtRgb565, 7, 5) == 70, "Rgb565 7x5");
        Check(tf::LevelBytes(gxdev::kTexFmtDxt1, 256, 256) == 32768, "Dxt1 256x256");
        Check(tf::RowBytes(gxdev::kTexFmtDxt1, 5) == 16, "Dxt1 row of 5 is two blocks");
        Check(tf::RowCount(gxdev::kTexFmtDxt1, 3) == 1, "Dxt1 3 rows is one block row");
        Check(tf::LevelBytes(gxdev::kTexFmtDxt1, 5, 3) == 16, "Dxt1 5x3");
        Check(tf::LevelBytes(gxdev::kTexFmtDxt5, 1, 1) == 16, "Dxt5 1x1 is a whole block");
        Check(tf::LevelBytes(gxdev::kTexFmtDxt3, 8, 4) == 32, "Dxt3 8x4");
        Check(tf::LevelBytes(gxdev::kTexFmtUnknown, 4, 4) == 0, "Unknown has no size");
        Check(tf::LevelBytes(gxdev::kTexFmtD24X8, 4, 4) == 0, "D24X8 has no upload size");
        Check(tf::LevelBytes(gxdev::kTexFmtArgb8888, 0, 4) == 0, "zero width");
        // 65536 x 65536 x 4 does not fit in 32 bits.
        Check(tf::LevelBytes(gxdev::kTexFmtArgb8888, 65536, 65536) == 17179869184ull, "64-bit size");

        Check(tf::ChainLevels(gxdev::kTexFmtArgb8888, 256, 256) == 9, "8888 256x256 chain");
        Check(tf::ChainLevels(gxdev::kTexFmtArgb8888, 256, 64) == 9, "8888 256x64 chain follows the larger side");
        Check(tf::ChainLevels(gxdev::kTexFmtArgb8888, 1, 1) == 1, "8888 1x1 chain");
        Check(tf::ChainLevels(gxdev::kTexFmtArgb8888, 5, 3) == 3, "8888 5x3 chain: 5, 2, 1");
        Check(tf::ChainLevels(gxdev::kTexFmtDxt1, 256, 256) == 7, "Dxt1 256x256 stops at 4x4");
        Check(tf::ChainLevels(gxdev::kTexFmtDxt1, 256, 64) == 5, "Dxt1 256x64 stops when the smaller side is 4");
        Check(tf::ChainLevels(gxdev::kTexFmtDxt5, 8, 8) == 2, "Dxt5 8x8");
        Check(tf::ChainLevels(gxdev::kTexFmtDxt1, 4, 4) == 1, "Dxt1 4x4");
        Check(tf::ChainLevels(gxdev::kTexFmtDxt1, 2, 2) == 1, "Dxt1 2x2 still has its one level");
        Check(tf::ChainLevels(gxdev::kTexFmtDxt1, 0, 8) == 0, "empty extent has no chain");
    }

    // --- BC1 ------------------------------------------------------------------------------------------

    void TestBc1()
    {
        uint8_t block[8];
        uint8_t got[16 * 4];
        uint8_t want[16][4];

        // Solid colour: c0 == c1 == pure red (0xF800), every index 0. 5-bit 31 must expand to 255.
        {
            const uint8_t idx[16] = { 0 };
            Put16(block, 0xF800); Put16(block + 2, 0xF800); Put32(block + 4, PackIndices2(idx));
            const uint8_t pal[4][4] = { { 255, 0, 0, 255 }, { 255, 0, 0, 255 }, { 255, 0, 0, 255 }, { 0, 0, 0, 0 } };
            Expect16(want, pal, idx);
            Check(tf::DecodeToRgba8(gxdev::kTexFmtDxt1, 4, 4, block, 8, got), "BC1 solid decodes");
            CheckPixels("BC1 solid red", got, &want[0][0], 4, 4);
        }

        // Four-colour mode: c0 = white 0xFFFF > c1 = black 0x0000. Colour 2 = 2/3 white = 170,
        // colour 3 = 1/3 white = 85, both exact (510/3, 255/3). Index pattern 0,1,2,3 per row.
        {
            const uint8_t idx[16] = { 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3 };
            Put16(block, 0xFFFF); Put16(block + 2, 0x0000); Put32(block + 4, PackIndices2(idx));
            const uint8_t pal[4][4] = { { 255, 255, 255, 255 }, { 0, 0, 0, 255 }, { 170, 170, 170, 255 }, { 85, 85, 85, 255 } };
            Expect16(want, pal, idx);
            Check(tf::DecodeToRgba8(gxdev::kTexFmtDxt1, 4, 4, block, 8, got), "BC1 four-colour decodes");
            CheckPixels("BC1 four-colour mode", got, &want[0][0], 4, 4);
        }

        // Three-colour mode: c0 = 0x0000 <= c1 = 0xC408 (r 24, g 32, b 8 -> 198, 130, 66 after
        // replication: 24<<3|24>>2 = 192|6, 32<<2|32>>4 = 128|2, 8<<3|8>>2 = 64|2). Colour 2 is the
        // exact half (99, 65, 33); colour 3 is transparent black.
        {
            const uint8_t idx[16] = { 3, 2, 1, 0, 0, 1, 2, 3, 3, 3, 3, 3, 2, 2, 2, 2 };
            Put16(block, 0x0000); Put16(block + 2, 0xC408); Put32(block + 4, PackIndices2(idx));
            const uint8_t pal[4][4] = { { 0, 0, 0, 255 }, { 198, 130, 66, 255 }, { 99, 65, 33, 255 }, { 0, 0, 0, 0 } };
            Expect16(want, pal, idx);
            Check(tf::DecodeToRgba8(gxdev::kTexFmtDxt1, 4, 4, block, 8, got), "BC1 three-colour decodes");
            CheckPixels("BC1 three-colour mode with transparent black", got, &want[0][0], 4, 4);
        }

        // Equal endpoints select the three-colour mode (c0 <= c1), so index 3 is transparent even
        // though both colours are the same opaque grey.
        {
            const uint8_t idx[16] = { 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3 };
            Put16(block, 0x8410); Put16(block + 2, 0x8410); Put32(block + 4, PackIndices2(idx));
            const uint8_t pal[4][4] = { { 0, 0, 0, 0 }, { 0, 0, 0, 0 }, { 0, 0, 0, 0 }, { 0, 0, 0, 0 } };
            Expect16(want, pal, idx);
            Check(tf::DecodeToRgba8(gxdev::kTexFmtDxt1, 4, 4, block, 8, got), "BC1 equal-endpoint decodes");
            CheckPixels("BC1 equal endpoints: index 3 is transparent", got, &want[0][0], 4, 4);
        }

        // Bit replication: 0x8410 is r 16, g 32, b 16 -> 132, 130, 132. A plain shift would give
        // 128, 128, 128.
        {
            const uint8_t idx[16] = { 0 };
            Put16(block, 0x8410); Put16(block + 2, 0x0000); Put32(block + 4, PackIndices2(idx));
            const uint8_t pal[4][4] = { { 132, 130, 132, 255 }, { 0, 0, 0, 255 }, { 0, 0, 0, 0 }, { 0, 0, 0, 0 } };
            Expect16(want, pal, idx);
            Check(tf::DecodeToRgba8(gxdev::kTexFmtDxt1, 4, 4, block, 8, got), "BC1 replication decodes");
            CheckPixels("565 bit replication", got, &want[0][0], 4, 4);
        }
    }

    // --- BC2 ------------------------------------------------------------------------------------------

    void TestBc2()
    {
        uint8_t block[16];
        uint8_t got[16 * 4];
        uint8_t want[16][4];

        // Row 0 alpha nibbles 0, 1, 2, 3 (word 0x3210: texel 0 in the low nibble) -> 0x00, 0x11,
        // 0x22, 0x33. Rows 1..3 alpha 0xF -> 255. Colour endpoints c0 = black < c1 = white, which
        // in BC1 would mean three-colour mode; in BC2 the colour block is always four-colour, so
        // index 2 is 1/3 white = 85 and index 3 is 2/3 white = 170, opaque.
        const uint8_t idx[16] = { 0, 1, 2, 3, 3, 3, 3, 3, 2, 2, 2, 2, 1, 1, 1, 1 };
        Put16(block, 0x3210); Put16(block + 2, 0xFFFF); Put16(block + 4, 0xFFFF); Put16(block + 6, 0xFFFF);
        Put16(block + 8, 0x0000); Put16(block + 10, 0xFFFF); Put32(block + 12, PackIndices2(idx));

        const uint8_t pal[4][4] = { { 0, 0, 0, 255 }, { 255, 255, 255, 255 }, { 85, 85, 85, 255 }, { 170, 170, 170, 255 } };
        Expect16(want, pal, idx);
        want[0][3] = 0x00; want[1][3] = 0x11; want[2][3] = 0x22; want[3][3] = 0x33;

        Check(tf::DecodeToRgba8(gxdev::kTexFmtDxt3, 4, 4, block, 16, got), "BC2 decodes");
        CheckPixels("BC2 explicit alpha and forced four-colour mode", got, &want[0][0], 4, 4);
    }

    // --- BC3 ------------------------------------------------------------------------------------------

    void TestBc3()
    {
        uint8_t block[16];
        uint8_t got[16 * 4];
        uint8_t want[16][4];

        // Eight-alpha mode: a0 = 224 > a1 = 0. 224 is 7 * 32, so the interpolants are exact:
        // index 2..7 = 192, 160, 128, 96, 64, 32. Colour: solid white.
        {
            const uint8_t aidx[16] = { 0, 1, 2, 3, 4, 5, 6, 7, 7, 6, 5, 4, 3, 2, 1, 0 };
            const uint8_t cidx[16] = { 0 };
            block[0] = 224; block[1] = 0;
            PackIndices3(block + 2, aidx);
            Put16(block + 8, 0xFFFF); Put16(block + 10, 0xFFFF); Put32(block + 12, PackIndices2(cidx));

            const uint8_t alpha[8] = { 224, 0, 192, 160, 128, 96, 64, 32 };
            for (int i = 0; i < 16; ++i) { want[i][0] = want[i][1] = want[i][2] = 255; want[i][3] = alpha[aidx[i]]; }
            Check(tf::DecodeToRgba8(gxdev::kTexFmtDxt5, 4, 4, block, 16, got), "BC3 eight-alpha decodes");
            CheckPixels("BC3 eight-alpha mode", got, &want[0][0], 4, 4);
        }

        // Six-alpha mode: a0 = 0 <= a1 = 200. Interpolants (4a0 + a1)/5 .. (a0 + 4a1)/5 = 40, 80,
        // 120, 160 exactly; index 6 is 0 and index 7 is 255 whatever the endpoints.
        {
            const uint8_t aidx[16] = { 0, 1, 2, 3, 4, 5, 6, 7, 7, 6, 5, 4, 3, 2, 1, 0 };
            const uint8_t cidx[16] = { 0 };
            block[0] = 0; block[1] = 200;
            PackIndices3(block + 2, aidx);
            Put16(block + 8, 0xFFFF); Put16(block + 10, 0xFFFF); Put32(block + 12, PackIndices2(cidx));

            const uint8_t alpha[8] = { 0, 200, 40, 80, 120, 160, 0, 255 };
            for (int i = 0; i < 16; ++i) { want[i][0] = want[i][1] = want[i][2] = 255; want[i][3] = alpha[aidx[i]]; }
            Check(tf::DecodeToRgba8(gxdev::kTexFmtDxt5, 4, 4, block, 16, got), "BC3 six-alpha decodes");
            CheckPixels("BC3 six-alpha mode", got, &want[0][0], 4, 4);
        }

        // The colour block of a BC3 ignores the c0 <= c1 rule: c0 = black, c1 = white, index 3
        // must be opaque 170, not transparent black.
        {
            const uint8_t aidx[16] = { 0 };
            const uint8_t cidx[16] = { 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3 };
            block[0] = 255; block[1] = 255;
            PackIndices3(block + 2, aidx);
            Put16(block + 8, 0x0000); Put16(block + 10, 0xFFFF); Put32(block + 12, PackIndices2(cidx));

            for (int i = 0; i < 16; ++i) { want[i][0] = want[i][1] = want[i][2] = 170; want[i][3] = 255; }
            Check(tf::DecodeToRgba8(gxdev::kTexFmtDxt5, 4, 4, block, 16, got), "BC3 colour decodes");
            CheckPixels("BC3 colour block is always four-colour", got, &want[0][0], 4, 4);
        }
    }

    // --- clipping and pitch ---------------------------------------------------------------------------

    void TestClipping()
    {
        // A 5x3 BC1 level: two blocks in one block row, at a pitch of 24 (8 bytes of padding
        // between block rows that must not be read, since there is only one row). Block 0 is
        // solid green, block 1 solid blue; the output is 5 wide, so only x = 4 comes from block 1,
        // and 3 high, so row 3 of each block is dropped.
        uint8_t src[24] = {};
        const uint8_t idx[16] = { 0 };
        Put16(src, 0x07E0); Put16(src + 2, 0x07E0); Put32(src + 4, PackIndices2(idx));
        Put16(src + 8, 0x001F); Put16(src + 10, 0x001F); Put32(src + 12, PackIndices2(idx));
        std::memset(src + 16, 0xAB, 8);

        // A guard after the 5 * 3 * 4 = 60 output bytes catches a write past the clipped extent.
        uint8_t got[60 + 16];
        std::memset(got, 0xCD, sizeof got);

        uint8_t want[60];
        for (uint32_t y = 0; y < 3; ++y)
        {
            for (uint32_t x = 0; x < 5; ++x)
            {
                uint8_t* p = want + (y * 5 + x) * 4;
                p[0] = 0; p[1] = (x < 4) ? 255 : 0; p[2] = (x < 4) ? 0 : 255; p[3] = 255;
            }
        }

        Check(tf::DecodeToRgba8(gxdev::kTexFmtDxt1, 5, 3, src, 24, got), "BC1 5x3 decodes");
        CheckPixels("BC1 5x3 clipped", got, want, 5, 3);
        bool guard = true;
        for (int i = 0; i < 16; ++i) guard = guard && got[60 + i] == 0xCD;
        Check(guard, "BC1 5x3 wrote nothing past the clipped extent");

        // A 1x1 BC3 level reads one whole block and writes one texel.
        uint8_t block[16] = {};
        block[0] = 77; block[1] = 0;
        Put16(block + 8, 0xF800); Put16(block + 10, 0xF800);
        uint8_t one[4 + 4];
        std::memset(one, 0xCD, sizeof one);
        const uint8_t wantOne[4] = { 255, 0, 0, 77 };
        Check(tf::DecodeToRgba8(gxdev::kTexFmtDxt5, 1, 1, block, 16, one), "BC3 1x1 decodes");
        CheckPixels("BC3 1x1", one, wantOne, 1, 1);
        Check(one[4] == 0xCD && one[7] == 0xCD, "BC3 1x1 wrote one texel");

        // Uncompressed pitch: a 2x2 Argb8888 at a pitch of 12 skips 4 bytes per row.
        const uint8_t rows[24] =
        {
            1, 2, 3, 4,   5, 6, 7, 8,   0xEE, 0xEE, 0xEE, 0xEE,
            9, 10, 11, 12, 13, 14, 15, 16, 0xEE, 0xEE, 0xEE, 0xEE,
        };
        const uint8_t wantRows[16] = { 3, 2, 1, 4, 7, 6, 5, 8, 11, 10, 9, 12, 15, 14, 13, 16 };
        uint8_t gotRows[16];
        Check(tf::DecodeToRgba8(gxdev::kTexFmtArgb8888, 2, 2, rows, 12, gotRows), "Argb8888 pitched decodes");
        CheckPixels("Argb8888 at pitch 12", gotRows, wantRows, 2, 2);

        // Refusals: a pitch shorter than a row, a null, an empty extent, a format with no decode.
        uint8_t sink[16] = {};
        Check(!tf::DecodeToRgba8(gxdev::kTexFmtArgb8888, 2, 2, rows, 7, sink), "short pitch refused");
        Check(!tf::DecodeToRgba8(gxdev::kTexFmtDxt1, 4, 4, src, 7, sink), "short block pitch refused");
        Check(!tf::DecodeToRgba8(gxdev::kTexFmtArgb8888, 2, 2, nullptr, 8, sink), "null source refused");
        Check(!tf::DecodeToRgba8(gxdev::kTexFmtArgb8888, 0, 2, rows, 8, sink), "empty extent refused");
        Check(!tf::DecodeToRgba8(gxdev::kTexFmtD24X8, 1, 1, rows, 4, sink), "D24X8 refused");
        Check(!tf::DecodeToRgba8(gxdev::kTexFmtUnknown, 1, 1, rows, 4, sink), "Unknown refused");
        Check(!tf::DecodeToRgba8(gxdev::kTexFormatCount, 1, 1, rows, 4, sink), "out-of-range refused");
    }

    // --- uncompressed formats -------------------------------------------------------------------------

    void TestUncompressed()
    {
        uint8_t got[8];

        // Abgr8888 is already R, G, B, A in memory: a copy.
        {
            const uint8_t src[8] = { 1, 2, 3, 4, 5, 6, 7, 8 };
            Check(tf::DecodeToRgba8(gxdev::kTexFmtAbgr8888, 2, 1, src, 8, got), "Abgr8888 decodes");
            CheckPixels("Abgr8888", got, src, 2, 1);
        }

        // Argb8888 is B, G, R, A in memory: R and B swap.
        {
            const uint8_t src[8]  = { 1, 2, 3, 4, 5, 6, 7, 8 };
            const uint8_t want[8] = { 3, 2, 1, 4, 7, 6, 5, 8 };
            Check(tf::DecodeToRgba8(gxdev::kTexFmtArgb8888, 2, 1, src, 8, got), "Argb8888 decodes");
            CheckPixels("Argb8888", got, want, 2, 1);
        }

        // Argb4444: 0xF12A is A = F, R = 1, G = 2, B = A -> each nibble doubled.
        {
            uint8_t src[4];
            Put16(src, 0xF12A); Put16(src + 2, 0x0000);
            const uint8_t want[8] = { 0x11, 0x22, 0xAA, 0xFF, 0, 0, 0, 0 };
            Check(tf::DecodeToRgba8(gxdev::kTexFmtArgb4444, 2, 1, src, 4, got), "Argb4444 decodes");
            CheckPixels("Argb4444", got, want, 2, 1);
        }

        // Argb1555: 0xFC10 is A = 1, R = 31, G = 0, B = 16 -> 255, 0, 132, 255; 0x7C00 is the same
        // red with A = 0.
        {
            uint8_t src[4];
            Put16(src, 0xFC10); Put16(src + 2, 0x7C00);
            const uint8_t want[8] = { 255, 0, 132, 255, 255, 0, 0, 0 };
            Check(tf::DecodeToRgba8(gxdev::kTexFmtArgb1555, 2, 1, src, 4, got), "Argb1555 decodes");
            CheckPixels("Argb1555", got, want, 2, 1);
        }

        // Rgb565: 0x8410 -> 132, 130, 132 (replication again); 0xFFFF -> white.
        {
            uint8_t src[4];
            Put16(src, 0x8410); Put16(src + 2, 0xFFFF);
            const uint8_t want[8] = { 132, 130, 132, 255, 255, 255, 255, 255 };
            Check(tf::DecodeToRgba8(gxdev::kTexFmtRgb565, 2, 1, src, 4, got), "Rgb565 decodes");
            CheckPixels("Rgb565", got, want, 2, 1);
        }

        // V8U8: bytes U then V, signed. (-128, 127) -> (0, 255, 128, 255); (0, 0) -> flat.
        {
            const uint8_t src[4]  = { 0x80, 0x7F, 0x00, 0x00 };
            const uint8_t want[8] = { 0, 255, 128, 255, 128, 128, 128, 255 };
            Check(tf::DecodeToRgba8(gxdev::kTexFmtU8V8, 2, 1, src, 4, got), "V8U8 decodes");
            CheckPixels("V8U8", got, want, 2, 1);
        }

        // G16R16F: R half then G half. (1.0, 0.5) -> (255, 128); (2.0, -1.0) saturates to
        // (255, 0). Half 1.0 is 0x3C00, 0.5 is 0x3800, 2.0 is 0x4000, -1.0 is 0xBC00.
        {
            uint8_t src[8];
            Put16(src, 0x3C00); Put16(src + 2, 0x3800); Put16(src + 4, 0x4000); Put16(src + 6, 0xBC00);
            const uint8_t want[8] = { 255, 128, 0, 255, 255, 0, 0, 255 };
            Check(tf::DecodeToRgba8(gxdev::kTexFmtG16R16F, 2, 1, src, 8, got), "G16R16F decodes");
            CheckPixels("G16R16F", got, want, 2, 1);
        }

        // R32F: 0.25 -> 64 grey; NaN -> 0.
        {
            uint8_t src[8];
            const float quarter = 0.25f;
            std::memcpy(src, &quarter, 4);
            Put32(src + 4, 0x7FC00000u);
            const uint8_t want[8] = { 64, 64, 64, 255, 0, 0, 0, 255 };
            Check(tf::DecodeToRgba8(gxdev::kTexFmtR32F, 2, 1, src, 8, got), "R32F decodes");
            CheckPixels("R32F", got, want, 2, 1);
        }
    }
}

int main()
{
    TestTable();
    TestSizes();
    TestBc1();
    TestBc2();
    TestBc3();
    TestClipping();
    TestUncompressed();

    std::printf("texformat: %d checks, %d failures\n", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}

#endif // WXL_EXTENSION
