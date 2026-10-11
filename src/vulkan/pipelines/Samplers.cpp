// The sampler cache.
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

#include "vulkan/pipelines/Internal.hpp"

namespace wxlvk::vk::pipelines::detail
{
    /**
     * The cached sampler for a masked texFlags value, built on first use.
     *
     * Filter (bits 0-2), the engine's EGxTexFilter:
     *
     *   0 nearest, no mip        NEAREST / NEAREST, mipmap NEAREST, maxLod 0
     *   1 linear, no mip         LINEAR / LINEAR,   mipmap NEAREST, maxLod 0
     *   2 nearest, mip nearest   NEAREST / NEAREST, mipmap NEAREST, maxLod unbounded
     *   3 linear, mip nearest    LINEAR / LINEAR,   mipmap NEAREST, maxLod unbounded
     *   4 linear, mip linear     LINEAR / LINEAR,   mipmap LINEAR,  maxLod unbounded
     *   5 anisotropic            as 4, with anisotropyEnable and maxAnisotropy = bits 9-13 when
     *                            src/vulkan/Context.cpp enabled samplerAnisotropy (D3D's
     *                            ANISOTROPIC min/mag filter with the texture's MAXANISOTROPY)
     *   6, 7                     not in the engine's enum; treated as 4
     *
     * "No mip" is mipmapMode NEAREST with maxLod 0: the level of detail is clamped to [0, 0], so
     * level 0 is the only level ever selected, and NEAREST means no blend toward a level the clamp
     * has already excluded. The other common spelling, mipmapMode LINEAR with maxLod 0.25, gets
     * the same picture by making the blend weight of level 1 round to nothing; it is a trick
     * where the clamp is a statement, so the clamp is used.
     *
     * Wrap: bit 3 (U) and bit 4 (V) set mean REPEAT, clear mean CLAMP_TO_EDGE -- the engine has
     * no border or mirrored modes. W is always CLAMP_TO_EDGE.
     *
     * The world variant (kSamplerWorldBit, set by the draw path for a world draw's mipmapped texture
     * through a linear mip filter, RecordInputs.cpp): `anisotropyEnable` with the device's own
     * anisotropy (WXL_VK_ANISOTROPY within the adapter's limit) where it is more than the flags ask
     * for, whatever the filter, and the device's mip LOD bias (WXL_VK_MIP_BIAS). The interface, the
     * text and the render targets never carry the bit, so they sample as the engine asked.
     *
     * @param uint32 key : texFlags & kSamplerKeyMask, with kSamplerWorldBit for the world variant
     * @return VkSampler sampler : VK_NULL_HANDLE when the cache is full or the create failed,
     *         logged once each
     */
    VkSampler Sampler(uint32_t key)
    {
        for (uint32_t i = 0; i < s.samplerCount; ++i)
            if (s.samplers[i].key == key) return s.samplers[i].sampler;

        if (s.samplerCount >= kMaxSamplers)
        {
            if (!s.loggedSamplersFull)
            {
                s.loggedSamplersFull = true;
                wxl::ScriptMgr::Log(WXL_LOG_WARN,
                                    "vulkan: pipelines: the sampler cache holds %u (kMaxSamplers); "
                                    "draws needing flags 0x%04X and later are skipped (logged once)",
                                    kMaxSamplers, key);
            }
            return VK_NULL_HANDLE;
        }

        const uint32_t filter = key & 0x7u;
        const bool     wrapU  = (key & (1u << 3)) != 0;
        const bool     wrapV  = (key & (1u << 4)) != 0;
        const bool     world  = (key & kSamplerWorldBit) != 0;
        // D3D's MAXANISOTROPY from bits 9-13 (cgxdevice 4.7), which only the ANISOTROPIC min/mag
        // filter of mode 5 reads; 0 or 1 is no anisotropy. The world variant raises it to the device's
        // own. Capped by the device's limit (the caps already clamp what the engine asks for to it,
        // Caps.cpp); without samplerAnisotropy the limit is 0 and nothing is anisotropic.
        uint32_t aniso = filter == 5 && s.anisoMax > 1.0f ? (key >> 9) & 0x1Fu : 0u;
        if (world && s.worldAniso > static_cast<float>(aniso)) aniso = static_cast<uint32_t>(s.worldAniso);
        if (static_cast<float>(aniso) > s.anisoMax) aniso = static_cast<uint32_t>(s.anisoMax);

        VkSamplerCreateInfo info = {};
        info.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;

        const bool linear = filter != 0 && filter != 2;
        info.magFilter = linear ? VK_FILTER_LINEAR : VK_FILTER_NEAREST;
        info.minFilter = info.magFilter;

        const bool noMip = filter < 2;
        info.mipmapMode = (filter >= 4) ? VK_SAMPLER_MIPMAP_MODE_LINEAR
                                        : VK_SAMPLER_MIPMAP_MODE_NEAREST;
        info.minLod     = 0.0f;
        info.maxLod     = noMip ? 0.0f : VK_LOD_CLAMP_NONE;
        info.mipLodBias = world && !noMip ? s.worldBias : 0.0f;

        info.addressModeU = wrapU ? VK_SAMPLER_ADDRESS_MODE_REPEAT : VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        info.addressModeV = wrapV ? VK_SAMPLER_ADDRESS_MODE_REPEAT : VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        info.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;

        info.anisotropyEnable        = aniso > 1 ? VK_TRUE : VK_FALSE;
        info.maxAnisotropy           = aniso > 1 ? static_cast<float>(aniso) : 1.0f;
        info.compareEnable           = VK_FALSE;
        info.compareOp               = VK_COMPARE_OP_ALWAYS;
        info.borderColor             = VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK;
        info.unnormalizedCoordinates = VK_FALSE;

        VkSampler sampler = VK_NULL_HANDLE;
        const VkResult r = s.dapi->CreateSampler(s.device, &info, nullptr, &sampler);
        if (r != VK_SUCCESS)
        {
            if (!s.loggedSamplerFailed)
            {
                s.loggedSamplerFailed = true;
                Fail("vkCreateSampler", r);
            }
            return VK_NULL_HANDLE;
        }

        SamplerEntry& e = s.samplers[s.samplerCount++];
        e.key     = key;
        e.sampler = sampler;
        s.samplerByKey[SamplerSlot(key)] = sampler;   // the draw path's lookup from here on
        s.stats.samplers = s.samplerCount;
        if (aniso > 1) ++s.stats.anisoSamplers;

        wxl::ScriptMgr::Log(WXL_LOG_DEBUG,
                            "vulkan: pipelines: sampler #%u for flags 0x%04X%s: filter %u -> %s/%s, "
                            "mip %s, anisotropy %u%s, LOD bias %+.2f, wrap %s/%s",
                            s.samplerCount, key & ~kSamplerWorldBit, world ? " (world)" : "", filter,
                            linear ? "LINEAR" : "NEAREST", linear ? "LINEAR" : "NEAREST",
                            noMip ? "none (maxLod 0)" : (filter >= 4 ? "LINEAR" : "NEAREST"),
                            aniso > 1 ? aniso : 1u,
                            filter == 5 && s.anisoMax <= 1.0f ? " (asked; samplerAnisotropy is not enabled)" : "",
                            static_cast<double>(info.mipLodBias), wrapU ? "REPEAT" : "CLAMP", wrapV ? "REPEAT" : "CLAMP");
        return sampler;
    }
}
