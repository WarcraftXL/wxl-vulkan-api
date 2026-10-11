// The device's optional capabilities: the instance version, the adapter's optional features probed, enabled
// and logged, the queue families a module may want to know about, the memory budget's query.
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

#include "vulkan/context/Internal.hpp"

#include <cstdio>

// The order of a bring-up with WXL_VK_CAPS on (the default):
//   1. CreateInstance asks InstanceVersion(): the loader's version (vkEnumerateInstanceVersion; a loader without it
//      is 1.0), patch dropped, capped at 1.2 -- nothing here wants 1.3.
//   2. SelectPhysicalDevice picks the adapter exactly as before (the version takes no new part in the choice).
//   3. CreateDevice asks PlanCaps(): the adapter's core features and, on a device of 1.1 or more, its
//      vkGetPhysicalDeviceFeatures2 / Properties2 chains, its extension list and its queue families. Every optional
//      feature it reports is asked for -- none is assumed -- and the create falls back to the device as it was
//      (six core features, no chain, no extension past the swapchain) if the driver refuses the richer one.
//   4. CapsCreated() notes what stands and writes ONE line (INFO) naming it all, and what was not offered.
// With WXL_VK_CAPS off every step is the 1.0 path, byte for byte: a 1.0 instance, no probe, no chain.
namespace wxlvk::vk::context::detail
{
    namespace
    {
        /// The highest version asked for: 1.2 holds every feature this probe enables (FP16 / INT8 core, the driver's
        /// name); 1.3 adds nothing a module here uses.
        constexpr uint32_t kHighest = VK_API_VERSION_1_2;

        /// What PlanCaps decided to ask for (CapBit), read back once the create answered.
        uint32_t g_asked = 0;

        uint32_t NoPatch(uint32_t v) { return VK_MAKE_API_VERSION(0, VK_API_VERSION_MAJOR(v), VK_API_VERSION_MINOR(v), 0); }
        uint32_t Lower(uint32_t a, uint32_t b) { return a < b ? a : b; }

        std::vector<VkExtensionProperties> DeviceExtensionList()
        {
            uint32_t n = 0;
            g.iapi.EnumerateDeviceExtensionProperties(g.physical, nullptr, &n, nullptr);
            std::vector<VkExtensionProperties> out(n);
            if (n) g.iapi.EnumerateDeviceExtensionProperties(g.physical, nullptr, &n, out.data());
            out.resize(n);
            return out;
        }

        template <class Fn> Fn InstanceFn(const char* name)
        {
            return reinterpret_cast<Fn>(loader::Api().GetInstanceProcAddr(g.instance, name));
        }

        /// The queue families a module might want: a compute family without graphics (async compute) and a
        /// transfer-only one (a copy engine), with their queue counts.
        void Families(DeviceCaps& c)
        {
            uint32_t n = 0;
            g.iapi.GetPhysicalDeviceQueueFamilyProperties(g.physical, &n, nullptr);
            std::vector<VkQueueFamilyProperties> f(n);
            if (n) g.iapi.GetPhysicalDeviceQueueFamilyProperties(g.physical, &n, f.data());
            f.resize(n);
            for (uint32_t i = 0; i < n; ++i)
            {
                const VkQueueFlags q = f[i].queueFlags;
                if (i == g.graphicsFamily) c.graphicsQueues = f[i].queueCount;
                if (f[i].queueCount == 0) continue;
                if ((q & VK_QUEUE_COMPUTE_BIT) && !(q & VK_QUEUE_GRAPHICS_BIT) && c.computeFamily == UINT32_MAX)
                {
                    c.computeFamily = i;
                    c.computeQueues = f[i].queueCount;
                }
                if ((q & VK_QUEUE_TRANSFER_BIT) && !(q & (VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT)) && c.transferFamily == UINT32_MAX)
                {
                    c.transferFamily = i;
                    c.transferQueues = f[i].queueCount;
                }
            }
        }

        /// Appends `text` to the list in `out` (", "-separated), bounded.
        void Add(char* out, size_t cap, const char* text)
        {
            const size_t n = std::strlen(out);
            if (n + 3 >= cap) return;
            std::snprintf(out + n, cap - n, "%s%s", n ? ", " : "", text);
        }

        /// The capability bits by name, for the line.
        struct Named { uint32_t bit; const char* name; };
        constexpr Named kNames[] = {
            { kCapFloat16, "FP16" },
            { kCapInt8, "INT8" },
            { kCapInt16, "INT16" },
            { kCapStorage16, "16-bit storage buffers" },
            { kCapUniform16, "16-bit uniforms" },
            { kCapPush16, "16-bit push constants" },
            { kCapInputOutput16, "16-bit varyings" },
            { kCapStorageImageExt, "extended storage image formats" },
            { kCapMultiview, "multiview" },
            { kCapLayerFromVertex, "gl_Layer from the vertex stage" },
            { kCapMultiViewport, "multi-viewport" },
            { kCapMultiDrawIndirect, "multi-draw indirect" },
            { kCapIndirectFirstInstance, "indirect first instance" },
            { kCapMemoryBudget, "memory budget" },
            { kCapSpirv13, "SPIR-V 1.3 and subgroups" },
            { kCapSpirv15, "SPIR-V 1.5" },
        };

        void SubgroupOps(uint32_t ops, char* out, size_t cap)
        {
            const Named names[] = {
                { VK_SUBGROUP_FEATURE_BASIC_BIT, "basic" }, { VK_SUBGROUP_FEATURE_VOTE_BIT, "vote" },
                { VK_SUBGROUP_FEATURE_ARITHMETIC_BIT, "arithmetic" }, { VK_SUBGROUP_FEATURE_BALLOT_BIT, "ballot" },
                { VK_SUBGROUP_FEATURE_SHUFFLE_BIT, "shuffle" }, { VK_SUBGROUP_FEATURE_SHUFFLE_RELATIVE_BIT, "relative shuffle" },
                { VK_SUBGROUP_FEATURE_CLUSTERED_BIT, "clustered" }, { VK_SUBGROUP_FEATURE_QUAD_BIT, "quad" },
            };
            out[0] = '\0';
            for (const Named& n : names)
                if (ops & n.bit) Add(out, cap, n.name);
            if (!out[0]) std::snprintf(out, cap, "none");
        }

        void Version(uint32_t v, char* out, size_t cap)
        {
            std::snprintf(out, cap, "%u.%u.%u", VK_API_VERSION_MAJOR(v), VK_API_VERSION_MINOR(v), VK_API_VERSION_PATCH(v));
        }
    }

    uint32_t InstanceVersion()
    {
        DeviceCaps& c = g.caps;
        c = DeviceCaps();
        uint32_t loaderVersion = VK_API_VERSION_1_0;
        const loader::Global& api = loader::Api();
        if (api.EnumerateInstanceVersion && api.EnumerateInstanceVersion(&loaderVersion) != VK_SUCCESS)
            loaderVersion = VK_API_VERSION_1_0;
        c.loaderVersion   = loaderVersion;
        c.instanceVersion = settings::CapsWanted() ? Lower(NoPatch(loaderVersion), kHighest) : VK_API_VERSION_1_0;
        if (c.instanceVersion < VK_API_VERSION_1_0) c.instanceVersion = VK_API_VERSION_1_0;
        return c.instanceVersion;
    }

    void PlanCaps(VkPhysicalDeviceFeatures& features, CapsChain& chain, void* tail, const char** names, uint32_t& count,
                  uint32_t room)
    {
        chain      = CapsChain();
        chain.head = tail;
        g_asked    = 0;
        DeviceCaps& c = g.caps;
        if (!settings::CapsWanted() || !g.physical) return;

        VkPhysicalDeviceProperties props = {};
        g.iapi.GetPhysicalDeviceProperties(g.physical, &props);
        c.adapterVersion = props.apiVersion;
        c.deviceVersion  = Lower(c.instanceVersion, NoPatch(props.apiVersion));
        c.probed         = true;
        const bool v11 = c.deviceVersion >= VK_API_VERSION_1_1;
        const bool v12 = c.deviceVersion >= VK_API_VERSION_1_2;
        const std::vector<VkExtensionProperties> exts = DeviceExtensionList();
        Families(c);

        // Core 1.0 features, optional: reported or not, as the six the device always asks for.
        VkPhysicalDeviceFeatures have = {};
        g.iapi.GetPhysicalDeviceFeatures(g.physical, &have);
        uint32_t offered = 0;
        if (have.shaderStorageImageExtendedFormats) offered |= kCapStorageImageExt;
        if (have.shaderInt16) offered |= kCapInt16;
        if (have.multiDrawIndirect) offered |= kCapMultiDrawIndirect;
        if (have.drawIndirectFirstInstance) offered |= kCapIndirectFirstInstance;
        if (have.multiViewport) offered |= kCapMultiViewport;
        if (v11) offered |= kCapSpirv13;
        if (v12) offered |= kCapSpirv15;

        // 1.1+: the extension features (16-bit storage and multiview are 1.1 core; FP16 / INT8 1.2 core or the KHR
        // extension on a 1.1 device). A struct is chained into the query only where the version or the extension
        // makes it known: an unknown sType there is invalid usage.
        const bool f16Known = v12 || (v11 && HasExtension(exts, VK_KHR_SHADER_FLOAT16_INT8_EXTENSION_NAME));
        const auto features2 = v11 ? InstanceFn<PFN_vkGetPhysicalDeviceFeatures2>("vkGetPhysicalDeviceFeatures2") : nullptr;
        if (features2)
        {
            VkPhysicalDevice16BitStorageFeatures      s16 = {};
            VkPhysicalDeviceMultiviewFeatures         mv  = {};
            VkPhysicalDeviceShaderFloat16Int8Features f16 = {};
            s16.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES;
            mv.sType  = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MULTIVIEW_FEATURES;
            f16.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES;
            s16.pNext = &mv;
            mv.pNext  = f16Known ? &f16 : nullptr;
            VkPhysicalDeviceFeatures2 all = {};
            all.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
            all.pNext = &s16;
            features2(g.physical, &all);
            if (s16.storageBuffer16BitAccess) offered |= kCapStorage16;
            if (s16.uniformAndStorageBuffer16BitAccess) offered |= kCapUniform16;
            if (s16.storagePushConstant16) offered |= kCapPush16;
            if (s16.storageInputOutput16) offered |= kCapInputOutput16;
            if (mv.multiview) offered |= kCapMultiview;
            if (f16Known && f16.shaderFloat16) offered |= kCapFloat16;
            if (f16Known && f16.shaderInt8) offered |= kCapInt8;
        }
        const auto properties2 = v11 ? InstanceFn<PFN_vkGetPhysicalDeviceProperties2>("vkGetPhysicalDeviceProperties2") : nullptr;
        if (properties2)
        {
            VkPhysicalDeviceSubgroupProperties sg  = {};
            VkPhysicalDeviceMultiviewProperties mvp = {};
            VkPhysicalDeviceDriverProperties    drv = {};
            sg.sType  = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES;
            mvp.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MULTIVIEW_PROPERTIES;
            drv.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES;
            sg.pNext  = &mvp;
            mvp.pNext = v12 ? &drv : nullptr;
            VkPhysicalDeviceProperties2 all = {};
            all.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
            all.pNext = &sg;
            properties2(g.physical, &all);
            c.subgroupSize      = sg.subgroupSize;
            c.subgroupStages    = sg.supportedStages;
            c.subgroupOps       = sg.supportedOperations;
            c.maxMultiviewViews = (offered & kCapMultiview) ? mvp.maxMultiviewViewCount : 0u;
            if (v12)
            {
                std::snprintf(c.driverName, sizeof c.driverName, "%s", drv.driverName);
                std::snprintf(c.driverInfo, sizeof c.driverInfo, "%s", drv.driverInfo);
            }
        }
        if (HasExtension(exts, VK_EXT_SHADER_VIEWPORT_INDEX_LAYER_EXTENSION_NAME)) offered |= kCapLayerFromVertex;
        // The budget's query (vkGetPhysicalDeviceMemoryProperties2) is 1.1 core: on a 1.0 instance it would need the
        // KHR instance extension, which only WXL_HOST asks for. A 1.0 device estimates instead (MemoryHeaps).
        if (v11 && HasExtension(exts, VK_EXT_MEMORY_BUDGET_EXTENSION_NAME)) offered |= kCapMemoryBudget;
        c.offered = offered;

        // Everything offered is asked for. Core features first.
        uint32_t asked = offered & (kCapSpirv13 | kCapSpirv15);
        auto core = [&](uint32_t bit, VkBool32& field) {
            if (offered & bit)
            {
                field = VK_TRUE;
                asked |= bit;
            }
        };
        core(kCapStorageImageExt, features.shaderStorageImageExtendedFormats);
        core(kCapInt16, features.shaderInt16);
        core(kCapMultiDrawIndirect, features.multiDrawIndirect);
        core(kCapIndirectFirstInstance, features.drawIndirectFirstInstance);
        core(kCapMultiViewport, features.multiViewport);

        // The chain, in front of the caller's (WXL_HOST's timeline struct when it is on). Separate structs, never the
        // VkPhysicalDeviceVulkan11/12Features aggregates: those may not stand beside the individual ones the host link
        // chains (the timeline semaphore's).
        void* head = tail;
        if (offered & kCapMultiview)
        {
            chain.multiview.sType     = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MULTIVIEW_FEATURES;
            chain.multiview.multiview = VK_TRUE;
            chain.multiview.pNext     = head;
            head = &chain.multiview;
            asked |= kCapMultiview;
        }
        const uint32_t any16 = kCapStorage16 | kCapUniform16 | kCapPush16 | kCapInputOutput16;
        if (offered & any16)
        {
            chain.storage16.sType                              = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES;
            chain.storage16.storageBuffer16BitAccess           = (offered & kCapStorage16) ? VK_TRUE : VK_FALSE;
            chain.storage16.uniformAndStorageBuffer16BitAccess = (offered & kCapUniform16) ? VK_TRUE : VK_FALSE;
            chain.storage16.storagePushConstant16              = (offered & kCapPush16) ? VK_TRUE : VK_FALSE;
            chain.storage16.storageInputOutput16               = (offered & kCapInputOutput16) ? VK_TRUE : VK_FALSE;
            chain.storage16.pNext = head;
            head = &chain.storage16;
            asked |= offered & any16;
        }
        // FP16 / INT8: core on a 1.2 device, the extension (asked for by name below) on a 1.1 one.
        const bool f16Ext   = !v12 && (offered & (kCapFloat16 | kCapInt8));
        const bool f16Room  = !f16Ext || count < room;
        if ((offered & (kCapFloat16 | kCapInt8)) && f16Room)
        {
            chain.float16.sType         = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES;
            chain.float16.shaderFloat16 = (offered & kCapFloat16) ? VK_TRUE : VK_FALSE;
            chain.float16.shaderInt8    = (offered & kCapInt8) ? VK_TRUE : VK_FALSE;
            chain.float16.pNext = head;
            head = &chain.float16;
            asked |= offered & (kCapFloat16 | kCapInt8);
            if (f16Ext) names[count++] = VK_KHR_SHADER_FLOAT16_INT8_EXTENSION_NAME;
        }
        chain.head = head;

        // The extensions, by name, while the caller's array has room.
        if ((offered & kCapMemoryBudget) && count < room)
        {
            names[count++] = VK_EXT_MEMORY_BUDGET_EXTENSION_NAME;
            asked |= kCapMemoryBudget;
        }
        if ((offered & kCapLayerFromVertex) && count < room)
        {
            names[count++] = VK_EXT_SHADER_VIEWPORT_INDEX_LAYER_EXTENSION_NAME;
            asked |= kCapLayerFromVertex;
        }
        g_asked = asked;
    }

    void CapsCreated(bool chainUsed)
    {
        DeviceCaps& c = g.caps;
        if (!chainUsed)
        {
            // The plain device: 1.0 semantics for everything a module may ask (the instance may still be newer).
            g_asked         = 0;
            c.enabled       = 0;
            c.deviceVersion = VK_API_VERSION_1_0;
        }
        else
            c.enabled = g_asked;
        g.memoryProperties2 = c.deviceVersion >= VK_API_VERSION_1_1
                                  ? InstanceFn<PFN_vkGetPhysicalDeviceMemoryProperties2>("vkGetPhysicalDeviceMemoryProperties2")
                                  : nullptr;
        if (!g.memoryProperties2) c.enabled &= ~static_cast<uint32_t>(kCapMemoryBudget);

        if (!settings::CapsWanted())
        {
            wxl::ScriptMgr::Log(WXL_LOG_INFO, "vulkan: device caps: %s is off: the Vulkan 1.0 device with its six core "
                                              "features, as before 2026-10-11", settings::kCapsKey);
            return;
        }
        char loaderText[24], adapterText[24], deviceText[24], instanceText[24];
        Version(c.loaderVersion, loaderText, sizeof loaderText);
        Version(c.adapterVersion, adapterText, sizeof adapterText);
        Version(c.deviceVersion, deviceText, sizeof deviceText);
        Version(c.instanceVersion, instanceText, sizeof instanceText);
        char on[512] = {}, missing[512] = {};
        for (const Named& n : kNames)
        {
            if (c.enabled & n.bit) Add(on, sizeof on, n.name);
            else if (!(c.offered & n.bit)) Add(missing, sizeof missing, n.name);
        }
        char ops[160];
        SubgroupOps(c.subgroupOps, ops, sizeof ops);
        char compute[96], transfer[96];
        if (c.computeFamily != UINT32_MAX)
            std::snprintf(compute, sizeof compute, "family %u, %u queue(s), not used yet", c.computeFamily, c.computeQueues);
        else
            std::snprintf(compute, sizeof compute, "none");
        if (c.transferFamily != UINT32_MAX)
            std::snprintf(transfer, sizeof transfer, "family %u, %u queue(s), not used", c.transferFamily, c.transferQueues);
        else
            std::snprintf(transfer, sizeof transfer, "none");
        wxl::ScriptMgr::Log(WXL_LOG_INFO,
                            "vulkan: device caps: a Vulkan %s device (loader %s, instance %s, adapter %s%s%s%s%s)%s; enabled: %s; "
                            "not offered: %s; subgroups %u lanes (stages 0x%X: %s); multiview %u view(s); queues: graphics family %u "
                            "(%u queue(s), 1 used), async compute %s, copy engine %s",
                            deviceText, loaderText, instanceText, adapterText, c.driverName[0] ? ", driver " : "", c.driverName,
                            c.driverInfo[0] ? " " : "", c.driverInfo,
                            chainUsed ? "" : " -- the driver refused the richer create, the plain one stands",
                            on[0] ? on : "nothing optional", missing[0] ? missing : "nothing", c.subgroupSize, c.subgroupStages, ops,
                            c.maxMultiviewViews, g.graphicsFamily, c.graphicsQueues, compute, transfer);
    }
}

namespace wxlvk::vk::context
{
    const DeviceCaps& Caps() { return detail::g.caps; }

    bool MemoryHeaps(VkPhysicalDeviceMemoryProperties& props, VkDeviceSize* budget, VkDeviceSize* usage)
    {
        using detail::g;
        if (!g.physical || !g.iapi.GetPhysicalDeviceMemoryProperties) return false;
        for (uint32_t h = 0; h < VK_MAX_MEMORY_HEAPS; ++h) budget[h] = usage[h] = 0;
        if (g.memoryProperties2 && (g.caps.enabled & kCapMemoryBudget))
        {
            VkPhysicalDeviceMemoryBudgetPropertiesEXT b = {};
            b.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT;
            VkPhysicalDeviceMemoryProperties2 all = {};
            all.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2;
            all.pNext = &b;
            g.memoryProperties2(g.physical, &all);
            props = all.memoryProperties;
            for (uint32_t h = 0; h < props.memoryHeapCount && h < VK_MAX_MEMORY_HEAPS; ++h)
            {
                budget[h] = b.heapBudget[h];
                usage[h]  = b.heapUsage[h];
            }
            return true;
        }
        g.iapi.GetPhysicalDeviceMemoryProperties(g.physical, &props);
        return false;
    }
}
