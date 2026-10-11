// wxl-vulkan-api native device: the published canopy density.
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

#include "native/render/Canopy.hpp"

#include "native/Present.hpp"

#include <cstddef>
#include <cstring>
#include <vector>

namespace
{
    constexpr uint32_t kMaxInstances = 4096;

    WXL_VkCanopyVolumes               g_volumes = {};
    std::vector<WXL_VkCanopyInstance> g_instances;
    bool                              g_held = false;
}

namespace wxlvk::native::render::canopy
{
    void __cdecl Publish(const WXL_VkCanopyVolumes* volumes)
    {
        if (!volumes || volumes->structSize < sizeof(WXL_VkCanopyVolumes) || !volumes->atlas) return;
        WXL_VkCanopyVolumes v = *volumes;
        v.structSize = sizeof v;
        const uint32_t n = v.instances ? (v.count < kMaxInstances ? v.count : kMaxInstances) : 0;
        g_instances.assign(v.instances, v.instances + n);
        v.count     = n;
        v.instances = g_instances.empty() ? nullptr : g_instances.data();
        g_volumes = v;
        g_held    = true;
    }

    const WXL_VkCanopyVolumes* For(uint64_t serial)
    {
        return g_held && g_volumes.serial == serial ? &g_volumes : nullptr;
    }

    int __cdecl Get(WXL_VkCanopyVolumes* out)
    {
        if (!out || out->structSize < sizeof(uint32_t)) return 0;
        const WXL_VkCanopyVolumes* v = For(wxlvk::native::present::Serial());
        if (!v) return 0;
        const uint32_t n = out->structSize < sizeof *v ? out->structSize : static_cast<uint32_t>(sizeof *v);
        std::memcpy(out, v, n);
        out->structSize = n;
        return 1;
    }
}
