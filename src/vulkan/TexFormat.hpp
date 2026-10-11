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

#pragma once

#include "vulkan/Vk.hpp"

#include <cstdint>

// Pure: no Vulkan call, no device, no allocation, no logging. Everything here is a function of its
// arguments and a constant table, which is what lets the decoder be proven by a standalone program
// (tests/texformat/) instead of by looking at a screen. The one Vulkan dependency is the VkFormat
// enum itself, reached through vulkan/Vk.hpp like every other file in src/vulkan/.
//
// The source of truth for the engine side is CGxDeviceD3d::s_GxTexFmtToD3dFmt, the table the D3D9
// backend indexes with an EGxTexFormat to get a D3DFORMAT. Each row of the table in TexFormat.cpp
// names that D3DFORMAT, works out its byte order, and says which VkFormat has the same one -- or
// that none does in Vulkan 1.0 core, which is what the `needsDecode` flag reports. Only 1.0 core
// formats are used, for the reason src/vulkan/Memory.hpp gives for its own restraint: the loader on
// the machine is promised to be 1.0 and nothing more.
//
// Every colour format is UNORM, never sRGB. D3D9 samples all of these as UNORM and the engine does
// its own gamma (the ramps at gxdevice::kGammaRamp), so an sRGB view would apply a second curve on
// top of the client's.
namespace wxlvk::vk::texformat
{
    /// What one engine format translates to.
    struct Info
    {
        VkFormat format;        ///< VK_FORMAT_UNDEFINED when there is no 1.0 core equivalent
        uint32_t blockWidth;    ///< 1 for uncompressed, 4 for DXT
        uint32_t blockHeight;   ///< 1 or 4
        uint32_t blockBytes;    ///< bytes per pixel for uncompressed, per 4x4 block for DXT
        bool     compressed;    ///< BC1/2/3
        bool     needsDecode;   ///< no usable VkFormat: the caller must decode to RGBA8 first
        bool     uploadable;    ///< false for Unknown and D24X8
        const char* name;       ///< "Abgr8888", "Dxt1", ...
    };

    /**
     * The table row for an engine format.
     *
     * A reference into a constexpr table rather than a copy, so a caller can keep the pointer for
     * the life of a texture without owning anything.
     *
     * @param uint32 gxFormat : an EGxTexFormat value
     * @return const Info& info : a static table row; out-of-range values map to the Unknown row
     */
    const Info& Of(uint32_t gxFormat);

    /**
     * Size of one mip level's tightly packed data, in bytes, for the format. For DXT this is
     * ceil(w/4) * ceil(h/4) * blockBytes.
     *
     * 64-bit on purpose: a 32-bit product of two 16-bit extents and a 4-byte texel already needs
     * 34 bits at the engine's maximum texture size, and a size that wraps is a size that passes a
     * bounds check it should have failed.
     *
     * @param uint32 gxFormat : an EGxTexFormat value
     * @param uint32 width : texels
     * @param uint32 height : texels
     * @return uint64 bytes : 0 for a non-uploadable format or an empty extent
     */
    uint64_t LevelBytes(uint32_t gxFormat, uint32_t width, uint32_t height);

    /**
     * Bytes per row of tightly packed data: for DXT a row of blocks, i.e. ceil(w/4) * blockBytes.
     *
     * @param uint32 gxFormat : an EGxTexFormat value
     * @param uint32 width : texels
     * @return uint32 bytes : 0 for a non-uploadable format or a zero width
     */
    uint32_t RowBytes(uint32_t gxFormat, uint32_t width);

    /**
     * Number of rows of data a level occupies: height for uncompressed, ceil(h/4) for DXT.
     *
     * @param uint32 gxFormat : an EGxTexFormat value
     * @param uint32 height : texels
     * @return uint32 rows : 0 for a non-uploadable format or a zero height
     */
    uint32_t RowCount(uint32_t gxFormat, uint32_t height);

    /**
     * The engine's rule for a full mip chain: levels down to 1x1 on the larger side, except that
     * DXT chains stop at the level where the SMALLER side reaches 4 (the engine does not upload
     * partial blocks). Returns the level count.
     *
     * A DXT texture that is already below 4 on a side at level 0 counts as one level: a chain of
     * zero levels is not a texture, and the one partial block its loader shipped is what the
     * engine uploads for it.
     *
     * @param uint32 gxFormat : an EGxTexFormat value
     * @param uint32 width : level 0 width in texels
     * @param uint32 height : level 0 height in texels
     * @return uint32 levels : 0 for an empty extent
     */
    uint32_t ChainLevels(uint32_t gxFormat, uint32_t width, uint32_t height);

    /**
     * Decodes one level of any uploadable colour format to tightly packed R8G8B8A8_UNORM
     * (memory order R, G, B, A), for display only. Handles BC1 (with the 1-bit alpha / colour-3
     * rule), BC2, BC3, and the uncompressed formats including Argb4444; expands V8U8 as
     * (u+128, v+128, 128, 255); Gr1616F and R32F are tone-mapped crudely to 8 bits and that is
     * documented as display-only.
     *
     * Display only, because the float formats lose everything above 1.0 and V8U8 loses its sign:
     * what comes out is something a person can look at on screen, not something a
     * shader may sample in place of the original. A backend that has to sample these formats
     * uploads them in their own VkFormat.
     *
     * No heap allocation: the DXT paths decode one block into a stack array and copy out the
     * texels that fall inside the level, which is also how a width or height that is not a
     * multiple of 4 is handled.
     *
     * @param uint32 gxFormat : an EGxTexFormat value
     * @param uint32 width : texels, non-zero
     * @param uint32 height : texels, non-zero
     * @param const void* src : the level data, `srcPitch` bytes per row (per block row for DXT)
     * @param uint32 srcPitch : bytes from one row (block row) to the next; at least RowBytes()
     * @param uint8* dst : width * height * 4 bytes
     * @return bool ok : false for a format it cannot decode (Unknown, D24X8), a null pointer, an
     *         empty extent or a pitch shorter than a row -- and then `dst` is untouched
     */
    bool DecodeToRgba8(uint32_t gxFormat, uint32_t width, uint32_t height,
                       const void* src, uint32_t srcPitch, uint8_t* dst);
}
