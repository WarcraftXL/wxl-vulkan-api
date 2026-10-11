// The memory budget: the card's and the system's heaps against what the driver lets the process have, the device
// allocator's share, the 32-bit address space; the residency question the modules ask before keeping more.
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

// Refreshed every kRefreshPresents presents (OnPresent): vkGetPhysicalDeviceMemoryProperties2 with
// VkPhysicalDeviceMemoryBudgetPropertiesEXT when the device has VK_EXT_memory_budget (src/vulkan/context/Caps.cpp), the
// allocator's figures (src/vulkan/Memory.cpp) and GlobalMemoryStatusEx; the VirtualQuery walk of the address space for
// its largest free run (src/vulkan/Footprint.hpp, up to half a millisecond) only every kWalkPresents. Everything the API
// hands out (wxl.vk.render GetMemoryBudget, MayKeepResident) is read from that cache: no driver call on a consumer's
// question.
namespace wxlvk::native::render::budget
{
    /// Presents between two refreshes (about two seconds at 60 fps), and between two address-space walks (ten).
    constexpr uint64_t kRefreshPresents = 120;
    constexpr uint64_t kWalkPresents    = 600;

    /// Once per present (src/native/Stats.cpp OnPresent): refreshes every kRefreshPresents, and writes the budget's
    /// line once, a few seconds into the session.
    void OnPresent(uint64_t presents);

    /// The cached figures (refreshed first if none were taken yet). False before the device exists.
    bool Get(WXL_VkMemoryBudget& out);

    /// wxl.vk.render MayKeepResident: whether `extraBytes` more may stay resident in the class (WXL_VkMemoryClass).
    bool MayKeep(uint64_t extraBytes, uint32_t memoryClass);

    /// The budget's line (INFO): the stats block, the session's end.
    void Log(const char* when);
}
