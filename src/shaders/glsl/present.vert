// The present pass (src/native/Present.cpp): one full-screen triangle.
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

// Vertices 0, 1, 2 at uv (0,0), (2,0), (0,2): the triangle covers the viewport, and uv (0,0) is the
// top-left of both images -- the back buffer's row 0 is its top, as the copy path presents it.
layout(location = 0) out vec2 vUV;

void main()
{
    vec2 p      = vec2(float((gl_VertexIndex << 1) & 2), float(gl_VertexIndex & 2));
    vUV         = p;
    gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}
