// 4x4 matrices as the engine holds them: row-major, row vectors (v' = v * M).
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

#include <cmath>

// Uploaded as they are, such a matrix reads in GLSL as its transpose, so `m * v` there is the row
// vector product v * M here: no transpose on either side.
namespace wxlvk::matrix
{
    /// out = a * b.
    inline void Mul(const float a[16], const float b[16], float out[16])
    {
        for (int r = 0; r < 4; ++r)
            for (int c = 0; c < 4; ++c)
                out[r * 4 + c] = a[r * 4 + 0] * b[0 * 4 + c] + a[r * 4 + 1] * b[1 * 4 + c] + a[r * 4 + 2] * b[2 * 4 + c]
                                 + a[r * 4 + 3] * b[3 * 4 + c];
    }

    /**
     * The inverse, by Gauss-Jordan elimination with partial pivoting (in double).
     *
     * @return bool ok : false for a singular matrix; `out` is then untouched
     */
    inline bool Invert(const float m[16], float out[16])
    {
        double a[4][8];
        for (int r = 0; r < 4; ++r)
            for (int c = 0; c < 4; ++c)
            {
                a[r][c]     = m[r * 4 + c];
                a[r][c + 4] = r == c ? 1.0 : 0.0;
            }
        for (int col = 0; col < 4; ++col)
        {
            int pivot = col;
            for (int r = col + 1; r < 4; ++r)
                if (std::fabs(a[r][col]) > std::fabs(a[pivot][col])) pivot = r;
            if (std::fabs(a[pivot][col]) < 1e-30) return false;
            for (int c = 0; c < 8; ++c)
            {
                const double t = a[col][c];
                a[col][c]   = a[pivot][c];
                a[pivot][c] = t;
            }
            const double inv = 1.0 / a[col][col];
            for (int c = 0; c < 8; ++c) a[col][c] *= inv;
            for (int r = 0; r < 4; ++r)
            {
                if (r == col) continue;
                const double f = a[r][col];
                for (int c = 0; c < 8; ++c) a[r][c] -= f * a[col][c];
            }
        }
        for (int r = 0; r < 4; ++r)
            for (int c = 0; c < 4; ++c) out[r * 4 + c] = static_cast<float>(a[r][c + 4]);
        return true;
    }
}
