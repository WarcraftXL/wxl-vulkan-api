// The engine's map object group renderers, bracketed.
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

#include "native/render/WmoGroups.hpp"

#include "native/Present.hpp"

#include "wxl/Script.hpp"
#include "wxl/bake/Common.h"
#include "wxl/game/Binding.hpp"
#include "wxl/offsets/game/WMO.hpp"

#include <cmath>
#include <cstring>

namespace
{
    namespace off = wxl::offsets::game::wmo;

    constexpr const char* kExtPoint = "Wmo.ExtRender";
    constexpr const char* kIntPoint = "Wmo.IntRender";

    /// The register rows of the map object's space -> view space transform (every MapObj vertex module:
    /// `dp4 r0.x, c31, v0` .. `c33`, the rows the G-buffer variants take the view normal through).
    constexpr uint32_t kXformRow = 31;

    /// __thiscall(this = root, group, int flag), ret 8 (WMO.hpp Wmo_RenderLeafFn), as this codebase's idiom
    /// for a hooked thiscall: fastcall with a dummy edx.
    using Leaf = void(__fastcall*)(void* root, void* edx, void* group, int flag);

    /// What the leaf running now draws.
    struct Current
    {
        const void* group    = nullptr;
        const void* root     = nullptr;
        uint32_t    flags    = 0;       ///< the group's MOGP flags
        bool        interior = false;   ///< lit by the engine as an interior (0x2000 without 0x40): its pixels take bit 5
        bool        enclosed = false;   ///< an indoor group (0x2000), exterior-lit or not: its box keeps the outdoor air out
        bool        noted    = false;   ///< its box is recorded for this frame
    };

    struct State
    {
        Leaf     ext = nullptr, in = nullptr;
        Current  now;
        uint64_t serial = 0;   ///< the frame the volumes are for
        uint32_t count  = 0;
        WXL_VkInteriorVolume volumes[WXL_VK_GBUFFER_MAX_INTERIORS] = {};
        const void* groups[WXL_VK_GBUFFER_MAX_INTERIORS] = {};   ///< whose they are, for the dedup
        WXL_VkInteriorGroup infos[WXL_VK_GBUFFER_MAX_INTERIORS] = {};   ///< which group each is (WXL_VkGBuffer::interiorGroups)
        wxlvk::native::render::wmogroups::Stats stats;
    };
    State g;

    /// The leaf's group, judged: enclosed when MOGP has 0x2000 (INTERIOR), and lit as an interior when it has not
    /// 0x40 (EXTERIOR_LIT) too. A group that does not name the leaf's root back is not trusted (counted, taken as
    /// exterior).
    Current Judge(void* root, void* group)
    {
        Current c;
        // One line once the world has been drawn a while: the bracket's verdicts, for the log.
        if (++g.stats.leaves == 5000)
            wxl::ScriptMgr::Log(WXL_LOG_INFO, "render: map object groups: %llu leaf call(s), %llu of an interior group, %llu not trusted "
                                "(the group names another root), %u interior box(es) last frame",
                                static_cast<unsigned long long>(g.stats.leaves), static_cast<unsigned long long>(g.stats.interiorLeaves),
                                static_cast<unsigned long long>(g.stats.rootMismatch), g.stats.lastVolumes);
        if (!group) return c;
        if (wxl::game::At<void* const>(group, off::kOffGroupRoot) != root)
        {
            ++g.stats.rootMismatch;
            return c;
        }
        const uint32_t flags = wxl::game::At<const uint32_t>(group, off::kOffGroupFlags);
        c.group    = group;
        c.root     = root;
        c.flags    = flags;
        c.enclosed = (flags & off::kGroupFlagIndoor) != 0;
        c.interior = c.enclosed && (flags & off::kGroupFlagExteriorLit) == 0;
        if (c.interior) ++g.stats.interiorLeaves;
        return c;
    }

    void __fastcall ExtDetour(void* root, void* edx, void* group, int flag)
    {
        const Current saved = g.now;
        g.now = Judge(root, group);
        g.ext(root, edx, group, flag);
        g.now = saved;
    }

    void __fastcall IntDetour(void* root, void* edx, void* group, int flag)
    {
        const Current saved = g.now;
        g.now = Judge(root, group);
        g.in(root, edx, group, flag);
        g.now = saved;
    }

    /// A WMO root's model key: CRC-32 of its inline file name lower-cased with '\\' separators (an offline tool can
    /// compute it from the name alone); 0 without a name.
    uint32_t RootKey(const void* root)
    {
        if (!root) return 0;
        const char* name = static_cast<const char*>(root) + off::kOffNameInline;
        char        low[264];
        size_t      n = 0;
        for (; n < sizeof low - 1 && name[n]; ++n)
        {
            char c = name[n];
            if (c == '/') c = '\\';
            else if (c >= 'A' && c <= 'Z') c = static_cast<char>(c - 'A' + 'a');
            low[n] = c;
        }
        return n ? wxl::bake::Crc32(low, n) : 0u;
    }

    /// The group's index in its root's group array (the MOGI order, the `_NNN` of its file); WXL_VK_INTERIOR_GROUP_UNKNOWN
    /// when the array does not list it.
    uint32_t GroupIndex(const void* root, const void* group)
    {
        if (!root || !group) return WXL_VK_INTERIOR_GROUP_UNKNOWN;
        const uint32_t count = wxl::game::At<const uint32_t>(root, off::kOffGroupCount);
        const uint32_t n     = count < 512u ? count : 512u;
        for (uint32_t i = 0; i < n; ++i)
            if (wxl::game::At<void* const>(root, off::kOffGroupArray + 4u * i) == group) return i;
        return WXL_VK_INTERIOR_GROUP_UNKNOWN;
    }

    /// The inverse of an affine 3 x 4 (row i: out_i = A_i . p + t_i); false when singular.
    bool InvertAffine(const float m[3][4], float out[12])
    {
        const double a = m[0][0], b = m[0][1], c = m[0][2];
        const double d = m[1][0], e = m[1][1], f = m[1][2];
        const double h = m[2][0], i = m[2][1], k = m[2][2];
        const double c00 = e * k - f * i, c01 = f * h - d * k, c02 = d * i - e * h;
        const double det = a * c00 + b * c01 + c * c02;
        if (!(std::fabs(det) > 1e-12)) return false;
        const double inv = 1.0 / det;
        // B = A^-1 = adj(A) / det, adj = the cofactors transposed.
        const double B[3][3] = {
            { c00 * inv, (c * i - b * k) * inv, (b * f - c * e) * inv },
            { c01 * inv, (a * k - c * h) * inv, (c * d - a * f) * inv },
            { c02 * inv, (b * h - a * i) * inv, (a * e - b * d) * inv },
        };
        for (int r = 0; r < 3; ++r)
        {
            out[r * 4 + 0] = static_cast<float>(B[r][0]);
            out[r * 4 + 1] = static_cast<float>(B[r][1]);
            out[r * 4 + 2] = static_cast<float>(B[r][2]);
            out[r * 4 + 3] = static_cast<float>(-(B[r][0] * m[0][3] + B[r][1] * m[1][3] + B[r][2] * m[2][3]));
        }
        return true;
    }
}

namespace wxlvk::native::render::wmogroups
{
    bool Install()
    {
        static bool installed = false;
        if (installed) return g.stats.hooked;
        installed = true;
        const WXL_Api* api = wxl::ScriptMgr::Api();
        if (!api) return false;
        const bool ext = api->HookAttachByName(kExtPoint, reinterpret_cast<void*>(&ExtDetour), reinterpret_cast<void**>(&g.ext),
                                               WXL_HOOK_DEFAULT_PRIORITY) != 0;
        const bool in  = api->HookAttachByName(kIntPoint, reinterpret_cast<void*>(&IntDetour), reinterpret_cast<void**>(&g.in),
                                               WXL_HOOK_DEFAULT_PRIORITY) != 0;
        g.stats.hooked = ext && in;
        if (!g.stats.hooked)
            wxl::ScriptMgr::Log(WXL_LOG_WARN, "render: the hook points %s / %s could not be attached (%s, %s): no pixel is marked as an "
                                "interior's, the light relights the interiors from outside", kExtPoint, kIntPoint, ext ? "ok" : "failed",
                                in ? "ok" : "failed");
        return g.stats.hooked;
    }

    bool InteriorDraw() { return g.now.interior; }

    bool EnclosedDraw() { return g.now.enclosed; }

    void NoteInteriorDraw(const float (*rows)[4])
    {
        if (!g.now.enclosed || g.now.noted || !rows) return;
        g.now.noted = true;
        const uint64_t serial = wxlvk::native::present::Serial();
        if (serial != g.serial)
        {
            if (g.serial) g.stats.lastVolumes = g.count;
            g.serial = serial;
            g.count  = 0;
        }
        float m[3][4];
        std::memcpy(m, rows[kXformRow], sizeof m);
        // The same group at the same placement already this frame (a leaf run twice): nothing new.
        for (uint32_t i = 0; i < g.count; ++i)
            if (g.groups[i] == g.now.group)
            {
                // The box's centre through this draw's transform against the one recorded (the sphere's centre).
                const WXL_VkInteriorVolume& v = g.volumes[i];
                const float cx = 0.5f * (v.boundsMin[0] + v.boundsMax[0]), cy = 0.5f * (v.boundsMin[1] + v.boundsMax[1]),
                            cz = 0.5f * (v.boundsMin[2] + v.boundsMax[2]);
                const float px = m[0][0] * cx + m[0][1] * cy + m[0][2] * cz + m[0][3];
                const float py = m[1][0] * cx + m[1][1] * cy + m[1][2] * cz + m[1][3];
                const float pz = m[2][0] * cx + m[2][1] * cy + m[2][2] * cz + m[2][3];
                if (std::fabs(px - v.sphere[0]) + std::fabs(py - v.sphere[1]) + std::fabs(pz - v.sphere[2]) < 0.01f) return;
            }
        if (g.count >= WXL_VK_GBUFFER_MAX_INTERIORS)
        {
            ++g.stats.dropped;
            return;
        }
        WXL_VkInteriorVolume v = {};
        if (!InvertAffine(m, v.localFromView)) return;
        const float* box = &wxl::game::At<const float>(g.now.group, off::kOffGroupBbox);   // min xyz, max xyz
        for (int k = 0; k < 3; ++k)
        {
            v.boundsMin[k] = box[k] < box[k + 3] ? box[k] : box[k + 3];
            v.boundsMax[k] = box[k] < box[k + 3] ? box[k + 3] : box[k];
            if (!std::isfinite(v.boundsMin[k]) || !std::isfinite(v.boundsMax[k])) return;
        }
        // An exterior-lit indoor group (a cave's or a mine's tunnels, an open hall): its air is enclosed, its light the
        // sun's bands -- the fog keeps the outdoor air out of it, the light relights it as an exterior.
        v.boundsMin[3] = g.now.interior ? 0.0f : static_cast<float>(WXL_VK_INTERIOR_EXTERIOR_LIT);
        // The bounding sphere in view space: the box's centre through the transform, the farthest corner's reach.
        const float c[3] = { 0.5f * (v.boundsMin[0] + v.boundsMax[0]), 0.5f * (v.boundsMin[1] + v.boundsMax[1]),
                             0.5f * (v.boundsMin[2] + v.boundsMax[2]) };
        const float h[3] = { 0.5f * (v.boundsMax[0] - v.boundsMin[0]), 0.5f * (v.boundsMax[1] - v.boundsMin[1]),
                             0.5f * (v.boundsMax[2] - v.boundsMin[2]) };
        float r2 = 0.0f;
        for (int s = 0; s < 4; ++s)
        {
            const float q[3] = { h[0], (s & 1) ? -h[1] : h[1], (s & 2) ? -h[2] : h[2] };
            float l2 = 0.0f;
            for (int r = 0; r < 3; ++r)
            {
                const float t = m[r][0] * q[0] + m[r][1] * q[1] + m[r][2] * q[2];
                l2 += t * t;
            }
            if (l2 > r2) r2 = l2;
        }
        for (int r = 0; r < 3; ++r) v.sphere[r] = m[r][0] * c[0] + m[r][1] * c[1] + m[r][2] * c[2] + m[r][3];
        v.sphere[3] = std::sqrt(r2);
        g.volumes[g.count] = v;
        g.groups[g.count]  = g.now.group;
        g.infos[g.count]   = { RootKey(g.now.root), GroupIndex(g.now.root, g.now.group), g.now.flags, 0u };
        ++g.count;
    }

    bool DrawBound(const float (*rows)[4], float out[4])
    {
        if (!g.now.group || !rows) return false;
        float m[3][4];
        std::memcpy(m, rows[kXformRow], sizeof m);
        const float* box = &wxl::game::At<const float>(g.now.group, off::kOffGroupBbox);   // min xyz, max xyz
        float c[3], h[3];
        for (int k = 0; k < 3; ++k)
        {
            if (!std::isfinite(box[k]) || !std::isfinite(box[k + 3])) return false;
            c[k] = 0.5f * (box[k] + box[k + 3]);
            h[k] = 0.5f * std::fabs(box[k + 3] - box[k]);
        }
        // The farthest corner's reach through the transform (its scale included), the centre through all of it.
        float r2 = 0.0f;
        for (int s = 0; s < 4; ++s)
        {
            const float q[3] = { h[0], (s & 1) ? -h[1] : h[1], (s & 2) ? -h[2] : h[2] };
            float l2 = 0.0f;
            for (int r = 0; r < 3; ++r)
            {
                const float t = m[r][0] * q[0] + m[r][1] * q[1] + m[r][2] * q[2];
                l2 += t * t;
            }
            if (l2 > r2) r2 = l2;
        }
        float o[4];
        for (int r = 0; r < 3; ++r) o[r] = m[r][0] * c[0] + m[r][1] * c[1] + m[r][2] * c[2] + m[r][3];
        o[3] = std::sqrt(r2) + 0.5f;   // half a yard: the batch's own vertices may sit on the box's faces
        if (!std::isfinite(o[0]) || !std::isfinite(o[1]) || !std::isfinite(o[2]) || !(o[3] > 0.0f) || !(o[3] < 5000.0f)) return false;
        std::memcpy(out, o, sizeof o);
        return true;
    }

    uint32_t Volumes(uint64_t serial, const WXL_VkInteriorVolume** out)
    {
        if (out) *out = nullptr;
        if (serial != g.serial || g.count == 0) return 0;
        if (out) *out = g.volumes;
        return g.count;
    }

    const WXL_VkInteriorGroup* Groups() { return g.infos; }

    const Stats& GetStats() { return g.stats; }
}
