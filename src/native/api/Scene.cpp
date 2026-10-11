// The scene as the device sees it at an insertion point: camera, projection, fog, viewport.
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


#include "native/api/Internal.hpp"

namespace wxlvk::native::render::detail
{
    State g;

    // --- the scene ------------------------------------------------------------------------------------

    /// One stack's top as the device draws with it: the identity flag wins over the bytes.
    void StackTop(const void* dev, unsigned xform, float out[16])
    {
        const uintptr_t stack = reinterpret_cast<uintptr_t>(dev) + gxdev::kMatrixStacks + xform * gxdev::kMatrixStackStride;
        uint32_t top = wxl::game::At<const uint32_t>(stack, gxdev::kMatrixStackTop);
        if (top >= gxdev::kMatrixStackDepth) top = gxdev::kMatrixStackDepth - 1;
        const uint32_t flags = wxl::game::At<const uint32_t>(stack, gxdev::kMatrixStackFlags + top * sizeof(uint32_t));
        if (flags & gxdev::kMatrixStackFlagIdentity)
        {
            for (unsigned i = 0; i < 16; ++i) out[i] = (i % 5 == 0) ? 1.0f : 0.0f;
            return;
        }
        std::memcpy(out, &wxl::game::At<const float>(stack, gxdev::kMatrixStackBase + top * 16 * sizeof(float)),
                    16 * sizeof(float));
    }

    void Mul(const float a[16], const float b[16], float out[16])
    {
        for (unsigned r = 0; r < 4; ++r)
            for (unsigned c = 0; c < 4; ++c)
                out[r * 4 + c] = a[r * 4 + 0] * b[0 * 4 + c] + a[r * 4 + 1] * b[1 * 4 + c]
                               + a[r * 4 + 2] * b[2 * 4 + c] + a[r * 4 + 3] * b[3 * 4 + c];
    }

    /// The eye of a row-vector view transform: the point that maps to the view-space origin.
    void Eye(const float v[16], float out[4])
    {
        // p * R + t = 0  ->  p = -t * R^-1, R the upper 3x3 (rows 0..2), t row 3.
        const float a = v[0], b = v[1], c = v[2];
        const float d = v[4], e = v[5], f = v[6];
        const float h = v[8], i = v[9], j = v[10];
        const float det = a * (e * j - f * i) - b * (d * j - f * h) + c * (d * i - e * h);
        out[0] = out[1] = out[2] = 0.0f;
        out[3] = 1.0f;
        if (std::fabs(det) < 1e-12f) return;
        const float k = 1.0f / det;
        const float inv[9] = {
            (e * j - f * i) * k, (c * i - b * j) * k, (b * f - c * e) * k,
            (f * h - d * j) * k, (a * j - c * h) * k, (c * d - a * f) * k,
            (d * i - e * h) * k, (b * h - a * i) * k, (a * e - b * d) * k,
        };
        const float tx = -v[12], ty = -v[13], tz = -v[14];
        for (unsigned col = 0; col < 3; ++col)
            out[col] = tx * inv[0 * 3 + col] + ty * inv[1 * 3 + col] + tz * inv[2 * 3 + col];
    }

    double Now()
    {
        if (g.qpcFrequency.QuadPart == 0)
        {
            QueryPerformanceFrequency(&g.qpcFrequency);
            QueryPerformanceCounter(&g.qpcStart);
        }
        LARGE_INTEGER t;
        QueryPerformanceCounter(&t);
        return static_cast<double>(t.QuadPart - g.qpcStart.QuadPart) / static_cast<double>(g.qpcFrequency.QuadPart);
    }

    bool IsIdentity(const float m[16])
    {
        for (unsigned i = 0; i < 16; ++i)
            if (m[i] != ((i % 5 == 0) ? 1.0f : 0.0f)) return false;
        return true;
    }

    /// The scene block as the engine's world shaders see it at the point.
    ///
    /// The world draws through shaders, which take their transforms as constants built from the
    /// engine's camera, not from the device's matrix stacks: in-world the View stack's top is identity
    /// at the point. So the view is the world camera's (camera::kView) and the eye its position
    /// (camera::kCameraPos); the View stack is the fallback where no camera is set (the glue screens).
    /// The projection is the device's API projection, the ProjNative the world shaders upload as
    /// c2..c5. The fog is the shader-effect fog block (shader::kEffectConstants: pixel c2 colour,
    /// vertex c30 scale / bias / exponent, factor = (z * scale + bias) ^ exponent with 1 unfogged),
    /// which is what the world's own pixels are fogged with; the fixed-function fog render states
    /// are not kept current on that path.
    void BuildScene(const void* dev, VkExtent2D extent, WXL_VkSceneConstants& sc)
    {
        std::memset(&sc, 0, sizeof sc);
        std::memcpy(sc.view, reinterpret_cast<const float*>(cam::kView), sizeof sc.view);
        const bool engineCamera = !IsIdentity(sc.view);
        if (!engineCamera) StackTop(dev, gxdev::kXformView, sc.view);
        std::memcpy(sc.projection, &wxl::game::At<const float>(dev, gxdev::kApiProjection), sizeof sc.projection);
        Mul(sc.view, sc.projection, sc.viewProj);
        if (engineCamera)
        {
            const float* eye = reinterpret_cast<const float*>(cam::kCameraPos);
            sc.cameraPos[0] = eye[0];
            sc.cameraPos[1] = eye[1];
            sc.cameraPos[2] = eye[2];
            sc.cameraPos[3] = 1.0f;
        }
        else
        {
            Eye(sc.view, sc.cameraPos);
        }

        const float* fogColour = reinterpret_cast<const float*>(shader::kEffectConstants + shader::kEffectConstFogColorAlphaRef);
        const float* fogParams = reinterpret_cast<const float*>(shader::kEffectConstants + shader::kEffectConstFogParams);
        const float scale = fogParams[0], bias = fogParams[1];
        sc.fogColor[0] = fogColour[0];
        sc.fogColor[1] = fogColour[1];
        sc.fogColor[2] = fogColour[2];
        if (scale < 0.0f)
        {
            // z * scale + bias is 1 at the start and 0 at the end.
            const float start = (1.0f - bias) / scale;
            const float end   = -bias / scale;
            sc.fogColor[3] = 1.0f;
            sc.fogRange[0] = start;
            sc.fogRange[1] = end;
            sc.fogRange[2] = end != start ? 1.0f / (end - start) : 0.0f;
            sc.fogRange[3] = fogParams[2] > 0.0f ? fogParams[2] : 1.0f;
        }

        sc.viewport[0] = static_cast<float>(extent.width);
        sc.viewport[1] = static_cast<float>(extent.height);
        sc.viewport[2] = extent.width ? 1.0f / static_cast<float>(extent.width) : 0.0f;
        sc.viewport[3] = extent.height ? 1.0f / static_cast<float>(extent.height) : 0.0f;

        // One clock reading per frame: every point of a frame sees the same time.
        const uint64_t serial = present::Serial();
        if (serial != g.timeSerial)
        {
            g.prevTime   = g.frameTime;
            g.frameTime  = Now();
            g.timeSerial = serial;
        }
        sc.time[0] = static_cast<float>(g.frameTime);
        sc.time[1] = static_cast<float>(g.frameTime - g.prevTime);
    }
}
