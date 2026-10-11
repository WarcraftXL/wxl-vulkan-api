// wxl.vk.render's stages: the API's entries over src/stages/Chain.hpp.
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

// A stage is a consumer's compute over the frame: the device ends the engine's pass, copies the frame
// in when a wanted stage of the point writes colour, runs every wanted stage of the point in its order,
// copies the frame back and lets the next draw resume (src/stages/Chain.hpp). A stage gets the working
// image, the scene depth when it asks, the world's scene snapshot and the world's jitter.
namespace wxlvk::native::render::stages
{
    /// wxl.vk.render RegisterStage: null for a bad description or a full table.
    void* Register(const WXL_VkStageDesc* desc, const char* name);

    /// wxl.vk.render UnregisterStage.
    void Unregister(void* stage);

    /// wxl.vk.render GetStageStats.
    int Stats(void* stage, WXL_VkStageStats* out);

    /// wxl.vk.render RequestJitter.
    void RequestJitter(int wanted);
}
