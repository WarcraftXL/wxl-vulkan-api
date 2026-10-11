// wxl-vulkan-api native device: the trees' canopy density a consumer publishes for this frame
// (wxl.vk.render PublishCanopy / GetCanopy), held for the volumetric readers.
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

namespace wxlvk::native::render::canopy
{
    /// wxl.vk.render PublishCanopy: copied with its instances, the latest of the frame wins.
    void __cdecl Publish(const WXL_VkCanopyVolumes* volumes);

    /// wxl.vk.render GetCanopy: 1 and filled up to out->structSize when one was published for the
    /// frame being recorded.
    int __cdecl Get(WXL_VkCanopyVolumes* out);

    /// The device's own readers: the volumes published for frame `serial`, or null.
    const WXL_VkCanopyVolumes* For(uint64_t serial);
}
