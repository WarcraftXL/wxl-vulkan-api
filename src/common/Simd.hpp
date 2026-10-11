// The draw path's 16-byte-row copies, clears and compares: SSE2, never the CRT.
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

#include <emmintrin.h>

#include <cstddef>
#include <cstdint>

// Everything the draw path copies is rows of sixteen bytes -- constant registers, a ring slice, a
// bone palette, a pipeline key -- and most of it lands in write-combined memory (the uniform and
// palette rings live in the BAR window). The CRT's memcpy / memset / memcmp are the wrong tools
// there: on this hardware a `rep movsb` into write-combined memory is five times slower than a
// loop of 16-byte stores (286 ns against 57 ns for one 64-row slice), a fast-string memset or
// memcpy issued shortly before such a copy adds about 100 ns to it, and MSVC inlines memcmp as a
// 4-byte loop. These helpers are the one implementation every file uses instead. Unaligned loads
// and stores throughout: the sources are 4-byte-aligned engine memory and the destinations are
// 16-byte-aligned ring offsets, and movups costs nothing extra on an aligned address.
namespace wxlvk::simd
{
    /**
     * Copies `rows` rows of sixteen bytes. Four rows per step: one 64-byte cache line, one
     * write-combining buffer, filled whole and in order.
     *
     * @param void* dst : the destination (may overlap nothing of `src`)
     * @param const void* src : the source
     * @param size_t rows : rows of 16 bytes
     */
    inline void CopyRows(void* dst, const void* src, size_t rows)
    {
        __m128i*       d = static_cast<__m128i*>(dst);
        const __m128i* s = static_cast<const __m128i*>(src);
        for (; rows >= 4; rows -= 4, d += 4, s += 4)
        {
            const __m128i a = _mm_loadu_si128(s);
            const __m128i b = _mm_loadu_si128(s + 1);
            const __m128i c = _mm_loadu_si128(s + 2);
            const __m128i e = _mm_loadu_si128(s + 3);
            _mm_storeu_si128(d, a);
            _mm_storeu_si128(d + 1, b);
            _mm_storeu_si128(d + 2, c);
            _mm_storeu_si128(d + 3, e);
        }
        for (; rows; --rows, ++d, ++s) _mm_storeu_si128(d, _mm_loadu_si128(s));
    }

    /**
     * Zeroes `rows` rows of sixteen bytes.
     *
     * @param void* dst : the destination
     * @param size_t rows : rows of 16 bytes
     */
    inline void ZeroRows(void* dst, size_t rows)
    {
        __m128i*      d = static_cast<__m128i*>(dst);
        const __m128i z = _mm_setzero_si128();
        for (; rows >= 4; rows -= 4, d += 4)
        {
            _mm_storeu_si128(d, z);
            _mm_storeu_si128(d + 1, z);
            _mm_storeu_si128(d + 2, z);
            _mm_storeu_si128(d + 3, z);
        }
        for (; rows; --rows, ++d) _mm_storeu_si128(d, z);
    }

    /**
     * Whether `rows` rows of sixteen bytes are byte for byte the same. Stops at the first row
     * that differs.
     *
     * @param const void* a : one
     * @param const void* b : the other
     * @param size_t rows : rows of 16 bytes
     * @return bool same
     */
    inline bool EqualRows(const void* a, const void* b, size_t rows)
    {
        const __m128i* x = static_cast<const __m128i*>(a);
        const __m128i* y = static_cast<const __m128i*>(b);
        for (; rows; --rows, ++x, ++y)
        {
            const __m128i eq = _mm_cmpeq_epi8(_mm_loadu_si128(x), _mm_loadu_si128(y));
            if (_mm_movemask_epi8(eq) != 0xFFFF) return false;
        }
        return true;
    }

    /**
     * The first row of `rows` that differs between `a` and `b`, or `rows` when none does: what a
     * caller that then copies from the difference onward wants to know.
     *
     * @return size_t row : the first differing row, `rows` when all are equal
     */
    inline size_t FirstDifferentRow(const void* a, const void* b, size_t rows)
    {
        const __m128i* x = static_cast<const __m128i*>(a);
        const __m128i* y = static_cast<const __m128i*>(b);
        for (size_t i = 0; i < rows; ++i)
        {
            const __m128i eq = _mm_cmpeq_epi8(_mm_loadu_si128(x + i), _mm_loadu_si128(y + i));
            if (_mm_movemask_epi8(eq) != 0xFFFF) return i;
        }
        return rows;
    }

    /**
     * Whether `n` 64-bit words are equal: a word loop, for the keys that are not whole rows.
     *
     * @return bool same
     */
    inline bool EqualWords64(const uint64_t* a, const uint64_t* b, size_t n)
    {
        for (size_t i = 0; i < n; ++i)
            if (a[i] != b[i]) return false;
        return true;
    }
}
