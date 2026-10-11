// The device's optional capabilities: what the adapter offered, what was enabled, what the modules may use.
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

#include "vulkan/Vk.hpp"

#include <cstdint>

// The device was a Vulkan 1.0 one with six core features (src/vulkan/context/Logical.cpp). Since 2026-10-11
// (WXL_VK_CAPS, default on) the instance asks for the loader's version up to 1.2, the device is created at
// min(that, the adapter's), and the optional features below are enabled whenever the adapter reports them --
// each probed through vkGetPhysicalDeviceFeatures2 / Properties2 (core from 1.1), never assumed. None of them
// changes what the device itself draws: the translated modules and the hand-written shaders are SPIR-V 1.0
// that uses none. They are there for the modules (wxl.vk.render GetDeviceCaps, include/wxl/VkRenderApi.h
// WXL_VkDeviceCaps), which build a variant only where the flag says so.
namespace wxlvk::vk
{
    /// What one flag of DeviceCaps::enabled means; WXL_VK_CAP_* in VkRenderApi.h carries the same bits.
    enum CapBit : uint32_t
    {
        kCapFloat16          = 1u << 0,   ///< shaderFloat16: float16_t arithmetic
        kCapInt8             = 1u << 1,   ///< shaderInt8
        kCapStorage16        = 1u << 2,   ///< storageBuffer16BitAccess: 16-bit members in storage buffers
        kCapUniform16        = 1u << 3,   ///< uniformAndStorageBuffer16BitAccess
        kCapPush16           = 1u << 4,   ///< storagePushConstant16
        kCapInputOutput16    = 1u << 5,   ///< storageInputOutput16 (16-bit varyings; NVIDIA has none)
        kCapInt16            = 1u << 6,   ///< shaderInt16
        kCapStorageImageExt  = 1u << 7,   ///< shaderStorageImageExtendedFormats: R16F, RG16F, R11G11B10F ... storage images
        kCapMultiview        = 1u << 8,   ///< multiview: one draw into several layers, gl_ViewIndex
        kCapLayerFromVertex  = 1u << 9,   ///< VK_EXT_shader_viewport_index_layer: gl_Layer from a vertex shader
        kCapMemoryBudget     = 1u << 10,  ///< VK_EXT_memory_budget: the driver's per-heap budget and usage
        kCapMultiDrawIndirect = 1u << 11, ///< multiDrawIndirect: drawCount > 1 in one vkCmdDraw*Indirect
        kCapIndirectFirstInstance = 1u << 12,  ///< drawIndirectFirstInstance
        kCapSpirv13          = 1u << 13,  ///< a 1.1 device: SPIR-V 1.3, the subgroup operations (subgroupOps)
        kCapSpirv15          = 1u << 14,  ///< a 1.2 device: SPIR-V 1.5
        kCapMultiViewport    = 1u << 15,  ///< multiViewport: gl_ViewportIndex (with kCapLayerFromVertex, from a vertex shader)
    };

    /// What the probe found and the device was created with. Zeroed (a 1.0 device, nothing optional) until
    /// the device exists, and for good with WXL_VK_CAPS off.
    struct DeviceCaps
    {
        uint32_t loaderVersion   = VK_API_VERSION_1_0;   ///< vkEnumerateInstanceVersion (1.0 when the loader has none)
        uint32_t instanceVersion = VK_API_VERSION_1_0;   ///< what the instance was created with
        uint32_t adapterVersion  = VK_API_VERSION_1_0;   ///< the chosen adapter's VkPhysicalDeviceProperties::apiVersion
        uint32_t deviceVersion   = VK_API_VERSION_1_0;   ///< min(instanceVersion, adapterVersion): what the device may be used as
        uint32_t offered = 0;    ///< CapBit: what the adapter reports
        uint32_t enabled = 0;    ///< CapBit: what the device was created with (a subset of offered)
        uint32_t subgroupSize   = 0;   ///< VkPhysicalDeviceSubgroupProperties (a 1.1 device), 0 unknown
        uint32_t subgroupStages = 0;   ///< VkShaderStageFlags
        uint32_t subgroupOps    = 0;   ///< VkSubgroupFeatureFlags
        uint32_t maxMultiviewViews = 0;
        uint32_t computeFamily  = UINT32_MAX;   ///< a queue family with compute and no graphics (async compute)
        uint32_t computeQueues  = 0;
        uint32_t transferFamily = UINT32_MAX;   ///< a queue family with transfer alone (a copy engine)
        uint32_t transferQueues = 0;
        uint32_t graphicsQueues = 0;            ///< queues of the graphics family (the device takes one)
        char     driverName[64] = {};           ///< VkPhysicalDeviceDriverProperties (1.2), "" unknown
        char     driverInfo[64] = {};
        bool     probed = false;                ///< the probe ran (WXL_VK_CAPS on and the device came up)
    };
}
