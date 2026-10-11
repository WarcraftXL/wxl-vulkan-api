// vulkan-1.dll and the three dispatch levels, all resolved at runtime.
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

#include "vulkan/Loader.hpp"

#include "wxl/Script.hpp"

namespace
{
    /// The system loader's file name. Bare, so the file-system redirector picks SysWOW64 for us.
    constexpr const char* kLoaderDll = "vulkan-1.dll";

    HMODULE                     g_module = nullptr;
    wxlvk::vk::loader::Global   g_global;
    uint32_t                    g_instanceVersion = 0;

    /**
     * One global-level entry point, by name.
     *
     * vkGetInstanceProcAddr with a null instance is the only legal way to ask for these four, and
     * the spec is explicit that asking for anything else that way returns null -- which is why this
     * helper is not reused for the instance level.
     *
     * @param string name : the function name
     * @return pointer fn : null when this loader does not have it
     */
    PFN_vkVoidFunction GlobalProc(const char* name)
    {
        return g_global.GetInstanceProcAddr ? g_global.GetInstanceProcAddr(nullptr, name) : nullptr;
    }
}

namespace wxlvk::vk::loader
{
    bool Load()
    {
        if (g_module) return g_global.CreateInstance != nullptr;

        g_module = ::LoadLibraryA(kLoaderDll);
        if (!g_module)
        {
            // Declining, not failing. A machine with no Vulkan driver has no vulkan-1.dll at all,
            // and that is the single most likely reason to be here -- so the line names the file and
            // the error and says what the consequence is, rather than reading as a defect.
            wxl::ScriptMgr::Log(WXL_LOG_WARN,
                                "vulkan: %s not present (LoadLibrary error %lu); the Vulkan "
                                "device cannot be created",
                                kLoaderDll, ::GetLastError());
            return false;
        }

        // The one symbol taken with GetProcAddress. Everything else comes through it, because the
        // loader exports only a 1.0-era subset by name and resolving e.g. vkCreateInstance with
        // GetProcAddress would silently bypass the layer chain.
        g_global.GetInstanceProcAddr = reinterpret_cast<PFN_vkGetInstanceProcAddr>(
            ::GetProcAddress(g_module, "vkGetInstanceProcAddr"));
        if (!g_global.GetInstanceProcAddr)
        {
            wxl::ScriptMgr::Log(WXL_LOG_WARN,
                                "vulkan: %s has no vkGetInstanceProcAddr; it is not a Vulkan loader. "
                                "Declining.", kLoaderDll);
            ::FreeLibrary(g_module);
            g_module = nullptr;
            return false;
        }

        g_global.CreateInstance = reinterpret_cast<PFN_vkCreateInstance>(
            GlobalProc("vkCreateInstance"));
        g_global.EnumerateInstanceLayerProperties =
            reinterpret_cast<PFN_vkEnumerateInstanceLayerProperties>(
                GlobalProc("vkEnumerateInstanceLayerProperties"));
        g_global.EnumerateInstanceExtensionProperties =
            reinterpret_cast<PFN_vkEnumerateInstanceExtensionProperties>(
                GlobalProc("vkEnumerateInstanceExtensionProperties"));

        // 1.1 and later. Null here means a 1.0 loader, which the spec says to read as
        // VK_API_VERSION_1_0 -- not as an error, and not as a reason to decline.
        g_global.EnumerateInstanceVersion = reinterpret_cast<PFN_vkEnumerateInstanceVersion>(
            GlobalProc("vkEnumerateInstanceVersion"));

        bool ok = true;
        if (!g_global.CreateInstance)
        {
            wxl::ScriptMgr::Log(WXL_LOG_WARN, "vulkan: global entry point missing: %s",
                                "vkCreateInstance");
            ok = false;
        }
        if (!g_global.EnumerateInstanceExtensionProperties)
        {
            wxl::ScriptMgr::Log(WXL_LOG_WARN, "vulkan: global entry point missing: %s",
                                "vkEnumerateInstanceExtensionProperties");
            ok = false;
        }
        // The layer enumeration is deliberately NOT required: without it, validation simply cannot
        // be offered, and that is a developer feature.
        if (!g_global.EnumerateInstanceLayerProperties)
            wxl::ScriptMgr::Log(WXL_LOG_DEBUG,
                                "vulkan: this loader has no vkEnumerateInstanceLayerProperties; "
                                "validation layers cannot be requested");

        if (!ok)
        {
            wxl::ScriptMgr::Log(WXL_LOG_WARN,
                                "vulkan: %s is too old; declining", kLoaderDll);
            ::FreeLibrary(g_module);
            g_module = nullptr;
            g_global = Global();
            return false;
        }

        g_instanceVersion = VK_API_VERSION_1_0;
        if (g_global.EnumerateInstanceVersion)
        {
            uint32_t version = 0;
            // The only documented failure is VK_ERROR_OUT_OF_HOST_MEMORY; on anything other than
            // success the 1.0 assumption above stands, which is the conservative direction.
            if (g_global.EnumerateInstanceVersion(&version) == VK_SUCCESS) g_instanceVersion = version;
        }

        char text[32];
        wxl::ScriptMgr::Log(WXL_LOG_DEBUG,
                            "vulkan: loaded %s at 0x%08X, instance API %s%s",
                            kLoaderDll, static_cast<unsigned>(reinterpret_cast<uintptr_t>(g_module)),
                            ApiVersionText(g_instanceVersion, text, sizeof text),
                            g_global.EnumerateInstanceVersion ? "" : " (assumed: no "
                                                                     "vkEnumerateInstanceVersion)");
        return true;
    }

    const Global& Api() { return g_global; }

    uint32_t InstanceVersion() { return g_instanceVersion; }

    bool StillMapped()
    {
        if (!g_module || !g_global.GetInstanceProcAddr) return false;

        // GetModuleHandleA does not take a reference, which is what makes it the right call here:
        // this runs on the exit path and must not change anybody's refcount.
        const HMODULE live = ::GetModuleHandleA(kLoaderDll);
        if (live != g_module) return false;

        // And the handle alone is not enough: a module handle is its base address, so a stale handle
        // could compare equal to a different mapping at the same base. Confirm the entry point we
        // actually hold still lies inside the image the headers describe.
        const auto* dos = reinterpret_cast<const IMAGE_DOS_HEADER*>(g_module);
        if (dos->e_magic != IMAGE_DOS_SIGNATURE) return false;
        const auto* nt = reinterpret_cast<const IMAGE_NT_HEADERS32*>(
            reinterpret_cast<const char*>(g_module) + dos->e_lfanew);
        if (nt->Signature != IMAGE_NT_SIGNATURE) return false;

        const uintptr_t base = reinterpret_cast<uintptr_t>(g_module);
        const uintptr_t end  = base + nt->OptionalHeader.SizeOfImage;
        const uintptr_t proc = reinterpret_cast<uintptr_t>(
            reinterpret_cast<void*>(g_global.GetInstanceProcAddr));
        return proc >= base && proc < end;
    }

    bool LoadInstance(VkInstance instance, bool debugUtils, Instance& out)
    {
        if (!instance || !g_global.GetInstanceProcAddr) return false;

        unsigned missing = 0;

        // A macro rather than twenty near-identical lines, and it does the one thing a hand-written
        // line gets wrong: it derives the string from the field name, so the name in the log and the
        // name passed to the loader cannot drift apart.
#define WXL_VK_INSTANCE_PROC(field)                                                               \
        do {                                                                                      \
            out.field = reinterpret_cast<PFN_vk##field>(                                          \
                g_global.GetInstanceProcAddr(instance, "vk" #field));                             \
            if (!out.field)                                                                       \
            {                                                                                     \
                wxl::ScriptMgr::Log(WXL_LOG_WARN, "vulkan: instance entry point missing: vk%s",   \
                                    #field);                                                      \
                ++missing;                                                                        \
            }                                                                                     \
        } while (false)

        WXL_VK_INSTANCE_PROC(DestroyInstance);
        WXL_VK_INSTANCE_PROC(EnumeratePhysicalDevices);
        WXL_VK_INSTANCE_PROC(GetPhysicalDeviceProperties);
        WXL_VK_INSTANCE_PROC(GetPhysicalDeviceFeatures);
        WXL_VK_INSTANCE_PROC(GetPhysicalDeviceMemoryProperties);
        WXL_VK_INSTANCE_PROC(GetPhysicalDeviceQueueFamilyProperties);
        WXL_VK_INSTANCE_PROC(EnumerateDeviceExtensionProperties);
        WXL_VK_INSTANCE_PROC(CreateDevice);
        WXL_VK_INSTANCE_PROC(GetDeviceProcAddr);

        // VK_KHR_surface / VK_KHR_win32_surface. These are only non-null because the instance was
        // created with those extensions enabled -- which is the whole reason the extension list is
        // checked against vkEnumerateInstanceExtensionProperties before vkCreateInstance rather than
        // being discovered by a null pointer here.
        WXL_VK_INSTANCE_PROC(CreateWin32SurfaceKHR);
        WXL_VK_INSTANCE_PROC(DestroySurfaceKHR);
        WXL_VK_INSTANCE_PROC(GetPhysicalDeviceFormatProperties);
        WXL_VK_INSTANCE_PROC(GetPhysicalDeviceSurfaceSupportKHR);
        WXL_VK_INSTANCE_PROC(GetPhysicalDeviceSurfaceCapabilitiesKHR);
        WXL_VK_INSTANCE_PROC(GetPhysicalDeviceSurfaceFormatsKHR);
        WXL_VK_INSTANCE_PROC(GetPhysicalDeviceSurfacePresentModesKHR);

        if (debugUtils)
        {
            WXL_VK_INSTANCE_PROC(CreateDebugUtilsMessengerEXT);
            WXL_VK_INSTANCE_PROC(DestroyDebugUtilsMessengerEXT);
        }

#undef WXL_VK_INSTANCE_PROC

        if (missing)
        {
            wxl::ScriptMgr::Log(WXL_LOG_WARN,
                                "vulkan: %u instance entry point(s) missing; declining", missing);
            return false;
        }
        return true;
    }

    bool LoadDevice(const Instance& api, VkDevice device, Device& out)
    {
        if (!device || !api.GetDeviceProcAddr) return false;

        unsigned missing = 0;

        // vkGetDeviceProcAddr, not vkGetInstanceProcAddr. Both would return something callable, but
        // the instance-level pointer goes through the loader's dispatch trampoline on every call,
        // and the device-level one is the driver's own entry for this device -- the faster one for
        // everything the frame calls thousands of times.
#define WXL_VK_DEVICE_PROC(field)                                                                 \
        do {                                                                                      \
            out.field = reinterpret_cast<PFN_vk##field>(                                          \
                api.GetDeviceProcAddr(device, "vk" #field));                                      \
            if (!out.field)                                                                       \
            {                                                                                     \
                wxl::ScriptMgr::Log(WXL_LOG_WARN, "vulkan: device entry point missing: vk%s",     \
                                    #field);                                                      \
                ++missing;                                                                        \
            }                                                                                     \
        } while (false)

        WXL_VK_DEVICE_PROC(DestroyDevice);
        WXL_VK_DEVICE_PROC(GetDeviceQueue);
        WXL_VK_DEVICE_PROC(DeviceWaitIdle);
        WXL_VK_DEVICE_PROC(CreateSwapchainKHR);
        WXL_VK_DEVICE_PROC(DestroySwapchainKHR);
        WXL_VK_DEVICE_PROC(GetSwapchainImagesKHR);

        // Everything below is what the frame and the modules call. It is resolved here, with the
        // rest, rather than lazily on the first frame: a missing entry point has to be a decline at
        // bring-up time with a name in the log, not a null call inside the frame.
        WXL_VK_DEVICE_PROC(CreateImageView);
        WXL_VK_DEVICE_PROC(DestroyImageView);
        WXL_VK_DEVICE_PROC(CreateRenderPass);
        WXL_VK_DEVICE_PROC(DestroyRenderPass);
        WXL_VK_DEVICE_PROC(CreateFramebuffer);
        WXL_VK_DEVICE_PROC(DestroyFramebuffer);

        WXL_VK_DEVICE_PROC(CreateCommandPool);
        WXL_VK_DEVICE_PROC(DestroyCommandPool);
        WXL_VK_DEVICE_PROC(AllocateCommandBuffers);
        WXL_VK_DEVICE_PROC(BeginCommandBuffer);
        WXL_VK_DEVICE_PROC(EndCommandBuffer);
        WXL_VK_DEVICE_PROC(CmdBeginRenderPass);
        WXL_VK_DEVICE_PROC(CmdEndRenderPass);

        // CreateSemaphore is the one field whose name collides with a windows.h macro;
        // src/vulkan/Loader.hpp undefines it, which is why the line below reads normally and the
        // string the macro builds is "vkCreateSemaphore" and not "vkCreateSemaphoreW".
        WXL_VK_DEVICE_PROC(CreateSemaphore);
        WXL_VK_DEVICE_PROC(DestroySemaphore);
        WXL_VK_DEVICE_PROC(CreateFence);
        WXL_VK_DEVICE_PROC(DestroyFence);
        WXL_VK_DEVICE_PROC(WaitForFences);
        WXL_VK_DEVICE_PROC(ResetFences);
        WXL_VK_DEVICE_PROC(GetFenceStatus);

        // The resource path: src/vulkan/Memory.cpp and src/vulkan/Texture.cpp. Resolved
        // here with everything else for the same reason the frame's entry points are -- a name in
        // the bring-up log beats a null pointer inside an upload.
        WXL_VK_DEVICE_PROC(AllocateMemory);
        WXL_VK_DEVICE_PROC(FreeMemory);
        WXL_VK_DEVICE_PROC(MapMemory);
        WXL_VK_DEVICE_PROC(UnmapMemory);
        WXL_VK_DEVICE_PROC(FlushMappedMemoryRanges);

        WXL_VK_DEVICE_PROC(CreateBuffer);
        WXL_VK_DEVICE_PROC(DestroyBuffer);
        WXL_VK_DEVICE_PROC(GetBufferMemoryRequirements);
        WXL_VK_DEVICE_PROC(BindBufferMemory);

        WXL_VK_DEVICE_PROC(CreateImage);
        WXL_VK_DEVICE_PROC(DestroyImage);
        WXL_VK_DEVICE_PROC(GetImageMemoryRequirements);
        WXL_VK_DEVICE_PROC(BindImageMemory);

        WXL_VK_DEVICE_PROC(CmdPipelineBarrier);
        WXL_VK_DEVICE_PROC(CmdCopyBuffer);
        WXL_VK_DEVICE_PROC(CmdCopyBufferToImage);
        WXL_VK_DEVICE_PROC(CmdClearColorImage);
        WXL_VK_DEVICE_PROC(CmdBlitImage);
        WXL_VK_DEVICE_PROC(CmdClearAttachments);


        // The draw path: src/vulkan/Shaders.cpp and src/vulkan/Pipelines.cpp. Required like the
        // rest, because every one of them is 1.0 core and a device cannot legally lack them.
        WXL_VK_DEVICE_PROC(CreateShaderModule);
        WXL_VK_DEVICE_PROC(DestroyShaderModule);

        WXL_VK_DEVICE_PROC(CreateDescriptorSetLayout);
        WXL_VK_DEVICE_PROC(DestroyDescriptorSetLayout);
        WXL_VK_DEVICE_PROC(CreatePipelineLayout);
        WXL_VK_DEVICE_PROC(DestroyPipelineLayout);
        WXL_VK_DEVICE_PROC(CreateGraphicsPipelines);
        WXL_VK_DEVICE_PROC(DestroyPipeline);
        WXL_VK_DEVICE_PROC(CreateDescriptorPool);
        WXL_VK_DEVICE_PROC(DestroyDescriptorPool);
        WXL_VK_DEVICE_PROC(ResetDescriptorPool);
        WXL_VK_DEVICE_PROC(AllocateDescriptorSets);
        WXL_VK_DEVICE_PROC(UpdateDescriptorSets);
        WXL_VK_DEVICE_PROC(CreateSampler);
        WXL_VK_DEVICE_PROC(DestroySampler);

        WXL_VK_DEVICE_PROC(CmdBindPipeline);
        WXL_VK_DEVICE_PROC(CmdBindVertexBuffers);
        WXL_VK_DEVICE_PROC(CmdBindIndexBuffer);
        WXL_VK_DEVICE_PROC(CmdBindDescriptorSets);
        WXL_VK_DEVICE_PROC(CmdPushConstants);
        WXL_VK_DEVICE_PROC(CmdSetViewport);
        WXL_VK_DEVICE_PROC(CmdSetScissor);
        WXL_VK_DEVICE_PROC(CmdDrawIndexed);
        WXL_VK_DEVICE_PROC(CmdDraw);
        WXL_VK_DEVICE_PROC(CmdSetDepthBias);
        WXL_VK_DEVICE_PROC(CmdSetBlendConstants);

        // Render targets, copies, read-back and queries; 1.0 core, so required like the rest.
        WXL_VK_DEVICE_PROC(CmdCopyImage);
        WXL_VK_DEVICE_PROC(CmdCopyImageToBuffer);
        WXL_VK_DEVICE_PROC(CreateQueryPool);
        WXL_VK_DEVICE_PROC(CmdResetQueryPool);
        WXL_VK_DEVICE_PROC(CmdBeginQuery);
        WXL_VK_DEVICE_PROC(CmdEndQuery);
        WXL_VK_DEVICE_PROC(GetQueryPoolResults);

        // The pipeline cache persisted beside the DLL. 1.0 core.
        WXL_VK_DEVICE_PROC(CreatePipelineCache);
        WXL_VK_DEVICE_PROC(GetPipelineCacheData);

        // The stages and the depth history (src/stages/, src/native/render/): compute, its indirect dispatches, buffer fills and
        // the GPU timestamps of the stats lines. 1.0 core.
        WXL_VK_DEVICE_PROC(CreateComputePipelines);
        WXL_VK_DEVICE_PROC(CmdDispatch);
        WXL_VK_DEVICE_PROC(CmdDispatchIndirect);
        WXL_VK_DEVICE_PROC(CmdFillBuffer);
        WXL_VK_DEVICE_PROC(CmdWriteTimestamp);

        WXL_VK_DEVICE_PROC(QueueSubmit);
        WXL_VK_DEVICE_PROC(AcquireNextImageKHR);
        WXL_VK_DEVICE_PROC(QueuePresentKHR);

#undef WXL_VK_DEVICE_PROC

        if (missing)
        {
            wxl::ScriptMgr::Log(WXL_LOG_WARN,
                                "vulkan: %u device entry point(s) missing; declining", missing);
            return false;
        }
        return true;
    }
}
