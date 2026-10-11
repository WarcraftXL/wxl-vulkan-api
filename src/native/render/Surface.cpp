// wxl.vk.render's surfaces: the latest published, by frame serial.
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

#include "native/render/Surface.hpp"

#include "native/Present.hpp"

#include <cstddef>
#include <cstring>

namespace
{
    WXL_VkSurface g_surface = {};
    bool          g_held    = false;
}

namespace wxlvk::native::render::surface
{
    void __cdecl Publish(const WXL_VkSurface* surface)
    {
        if (!surface || surface->structSize < offsetof(WXL_VkSurface, velocityView) || !surface->view) return;
        WXL_VkSurface s = {};
        std::memcpy(&s, surface, surface->structSize < sizeof s ? surface->structSize : sizeof s);
        s.structSize = sizeof s;
        g_surface = s;
        g_held    = true;
    }

    const WXL_VkSurface* For(uint64_t serial)
    {
        return g_held && g_surface.serial == serial ? &g_surface : nullptr;
    }

    int __cdecl Get(WXL_VkSurface* out)
    {
        if (!out || out->structSize < sizeof(uint32_t)) return 0;
        const WXL_VkSurface* s = For(wxlvk::native::present::Serial());
        if (!s) return 0;
        const uint32_t n = out->structSize < sizeof *s ? out->structSize : static_cast<uint32_t>(sizeof *s);
        std::memcpy(out, s, n);
        out->structSize = n;
        return 1;
    }
}
