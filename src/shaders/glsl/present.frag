// The present pass (src/native/Present.cpp): the back buffer, scaled, through the gamma LUT.
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
#version 450

layout(set = 0, binding = 0) uniform sampler2D backBuffer;
/// 256 x 1 RGBA8: entry i of a channel is what an 8-bit value i of that channel is presented as
/// (the engine's gamma ramp composed with the inverse of the desktop's).
layout(set = 0, binding = 1) uniform sampler2D lut;

/// x != 0: apply the LUT.
layout(push_constant) uniform Consts { vec4 flags; } pc;

layout(location = 0) in vec2 vUV;
layout(location = 0) out vec4 outColor;

void main()
{
    vec3 c = texture(backBuffer, vUV).rgb;
    if (pc.flags.x != 0.0)
    {
        // The centre of entry round(c * 255): a lookup, not a blend between entries.
        vec3 u = (floor(c * 255.0 + 0.5) + 0.5) / 256.0;
        c = vec3(texture(lut, vec2(u.r, 0.5)).r,
                 texture(lut, vec2(u.g, 0.5)).g,
                 texture(lut, vec2(u.b, 0.5)).b);
    }
    outColor = vec4(c, 1.0);
}
