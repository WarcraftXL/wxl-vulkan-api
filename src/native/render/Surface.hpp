// wxl.vk.render's surfaces: a consumer's screen-space surface for the frame (the foliage's coverage,
// AO and motion), held for the post chain and the other consumers.
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

#include "wxl/VkRenderApi.h"

#include <cstdint>

namespace wxlvk::native::render::surface
{
    /// wxl.vk.render PublishSurface: copied, the latest of the frame wins.
    void __cdecl Publish(const WXL_VkSurface* surface);

    /// wxl.vk.render GetSurface: 1 and filled up to out->structSize when one was published for the
    /// frame being recorded.
    int __cdecl Get(WXL_VkSurface* out);

    /// The surface published for frame `serial`, or null (wxl.vk.render GetSurface reads it).
    const WXL_VkSurface* For(uint64_t serial);
}
