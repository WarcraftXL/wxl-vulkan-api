// wxl-vulkan-api: the shared world's images and timelines, imported into the native device.
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

#include "wxl/VkHostLinkApi.h"

#include <cstdint>

namespace wxlvk::host
{
    /**
     * Imports one generation in place of the live one (which retires until its frames complete).
     * Every handle in `world` is closed, imported or not.
     *
     * @return false with `why` filled when an import fails; nothing of the new generation is kept
     */
    bool ImportWorld(const WXL_VkSharedWorld& world, char* why, uint32_t cap);

    /// Stops composing and retires the live generation.
    void ReleaseWorld();
}
