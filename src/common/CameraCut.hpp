// Whether the camera jumped between two frames: a cut, a teleport, a loading screen.
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
#include <cstdint>

// One rule for everything that keeps last frame's image (TAA's motion, the depth history): the
// previous frame must be the one right before, and the eye must not have moved more than 10 yards
// or turned more than 60 degrees in between.
namespace wxlvk::cut
{
    constexpr float kDistance = 10.0f;   ///< yards in one frame
    constexpr float kCosine   = 0.5f;    ///< a turn past 60 degrees in one frame

    /// A frame's camera, as the test needs it.
    struct Camera
    {
        bool     valid      = false;
        uint64_t serial     = 0;
        float    eye[3]     = {};
        float    forward[3] = {};
    };

    /// The camera of a frame from its view matrix (row vectors: the forward axis in world space is
    /// the third column) and eye.
    inline Camera Of(uint64_t serial, const float view[16], const float eye[4])
    {
        Camera c;
        const float x = view[2], y = view[6], z = view[10];
        const float len = std::sqrt(x * x + y * y + z * z);
        c.forward[0] = len > 0.0f ? x / len : 0.0f;
        c.forward[1] = len > 0.0f ? y / len : 0.0f;
        c.forward[2] = len > 0.0f ? z / len : 1.0f;
        for (int i = 0; i < 3; ++i) c.eye[i] = eye[i];
        c.serial = serial;
        c.valid  = true;
        return c;
    }

    /// Whether `now` cannot use what was kept at `before`.
    inline bool Between(const Camera& before, const Camera& now)
    {
        if (!before.valid || !now.valid || before.serial + 1 != now.serial) return true;
        const float dx = now.eye[0] - before.eye[0], dy = now.eye[1] - before.eye[1], dz = now.eye[2] - before.eye[2];
        if (dx * dx + dy * dy + dz * dz > kDistance * kDistance) return true;
        return now.forward[0] * before.forward[0] + now.forward[1] * before.forward[1]
                   + now.forward[2] * before.forward[2] < kCosine;
    }
}
