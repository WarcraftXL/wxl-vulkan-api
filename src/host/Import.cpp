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

#include "host/Import.hpp"

#include "host/State.hpp"
#include "native/Present.hpp"
#include "vulkan/Context.hpp"

#include "wxl/Script.hpp"

#include <cstdio>

// Imported memory is device-local and never mapped: a generation costs this 32-bit process no address
// space, only the driver's bookkeeping. Nothing imported is destroyed while a frame that named it may
// still run on the GPU: a replaced generation waits in `retiring` for its last serial to complete.
namespace wxlvk::host
{
    namespace
    {
        namespace context = wxlvk::vk::context;
        namespace present = wxlvk::native::present;

        constexpr VkExternalMemoryHandleTypeFlagBitsKHR    kMemory    = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT_KHR;
        constexpr VkExternalSemaphoreHandleTypeFlagBitsKHR kSemaphore = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT_KHR;

        HANDLE AsHandle(uint64_t h) { return reinterpret_cast<HANDLE>(static_cast<uintptr_t>(h)); }

        void CloseAll(const WXL_VkSharedWorld& w)
        {
            if (w.hostTimeline) CloseHandle(AsHandle(w.hostTimeline));
            if (w.clientTimeline) CloseHandle(AsHandle(w.clientTimeline));
            for (uint32_t i = 0; i < w.slotCount && i < WXL_VK_SHARED_SLOTS; ++i)
                if (w.color[i].handle) CloseHandle(AsHandle(w.color[i].handle));
        }

        bool Failed(char* why, uint32_t cap, const char* what, VkResult r)
        {
            if (why && cap) std::snprintf(why, cap, "%s failed: VkResult %d", what, static_cast<int>(r));
            return false;
        }

        uint32_t DeviceLocal(uint32_t bits)
        {
            const context::Handles* h = context::Get();
            VkPhysicalDeviceMemoryProperties m{};
            h->iapi->GetPhysicalDeviceMemoryProperties(h->physical, &m);
            for (uint32_t i = 0; i < m.memoryTypeCount; ++i)
                if ((bits & (1u << i)) && (m.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) return i;
            return UINT32_MAX;
        }

        bool ImportImage(const WXL_VkSharedImage& s, VkImage& image, VkDeviceMemory& memory, char* why, uint32_t cap)
        {
            const Functions& fn = G().fn;
            const VkDevice   device = context::Get()->device;
            VkExternalMemoryImageCreateInfoKHR ext{};
            ext.sType       = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO_KHR;
            ext.handleTypes = kMemory;
            VkImageCreateInfo info{};
            info.sType       = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
            info.pNext       = &ext;
            info.imageType   = VK_IMAGE_TYPE_2D;
            info.format      = static_cast<VkFormat>(s.format);
            info.extent      = { s.width, s.height, 1 };
            info.mipLevels   = 1;
            info.arrayLayers = 1;
            info.samples     = VK_SAMPLE_COUNT_1_BIT;
            info.tiling      = VK_IMAGE_TILING_OPTIMAL;
            info.usage       = s.usage;
            if (VkResult r = fn.createImage(device, &info, nullptr, &image); r != VK_SUCCESS)
                return Failed(why, cap, "vkCreateImage (import)", r);
            VkMemoryRequirements reqs{};
            fn.getImageMemoryRequirements(device, image, &reqs);

            VkImportMemoryWin32HandleInfoKHR import{};
            import.sType      = VK_STRUCTURE_TYPE_IMPORT_MEMORY_WIN32_HANDLE_INFO_KHR;
            import.handleType = kMemory;
            import.handle     = AsHandle(s.handle);
            VkMemoryDedicatedAllocateInfoKHR dedicated{};
            dedicated.sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO_KHR;
            dedicated.pNext = &import;
            dedicated.image = image;
            VkMemoryAllocateInfo alloc{};
            alloc.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
            alloc.pNext           = s.dedicated ? static_cast<const void*>(&dedicated) : &import;
            alloc.allocationSize  = s.allocationSize;
            alloc.memoryTypeIndex = DeviceLocal(reqs.memoryTypeBits);
            if (VkResult r = fn.allocateMemory(device, &alloc, nullptr, &memory); r != VK_SUCCESS)
                return Failed(why, cap, "vkAllocateMemory (import OPAQUE_WIN32)", r);
            if (VkResult r = fn.bindImageMemory(device, image, memory, 0); r != VK_SUCCESS)
                return Failed(why, cap, "vkBindImageMemory (import)", r);
            return true;
        }

        bool ImportTimeline(uint64_t handle, VkSemaphore& out, char* why, uint32_t cap)
        {
            const Functions& fn     = G().fn;
            const VkDevice   device = context::Get()->device;
            VkSemaphoreTypeCreateInfoKHR type{};
            type.sType         = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO_KHR;
            type.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE_KHR;
            VkSemaphoreCreateInfo info{};
            info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
            info.pNext = &type;
            if (VkResult r = fn.createSemaphore(device, &info, nullptr, &out); r != VK_SUCCESS)
                return Failed(why, cap, "vkCreateSemaphore (timeline)", r);
            VkImportSemaphoreWin32HandleInfoKHR import{};
            import.sType      = VK_STRUCTURE_TYPE_IMPORT_SEMAPHORE_WIN32_HANDLE_INFO_KHR;
            import.semaphore  = out;
            import.handleType = kSemaphore;
            import.handle     = AsHandle(handle);
            if (VkResult r = fn.importSemaphoreWin32HandleKHR(device, &import); r != VK_SUCCESS)
                return Failed(why, cap, "vkImportSemaphoreWin32HandleKHR (timeline OPAQUE_WIN32)", r);
            return true;
        }

        void Destroy(Imported& g)
        {
            const Functions& fn     = G().fn;
            const VkDevice   device = context::Get() ? context::Get()->device : VK_NULL_HANDLE;
            if (!device) return;
            for (uint32_t i = 0; i < WXL_VK_SHARED_SLOTS; ++i)
            {
                if (g.images[i]) fn.destroyImage(device, g.images[i], nullptr);
                if (g.memory[i]) fn.freeMemory(device, g.memory[i], nullptr);
            }
            if (g.hostTimeline) fn.destroySemaphore(device, g.hostTimeline, nullptr);
            if (g.clientTimeline) fn.destroySemaphore(device, g.clientTimeline, nullptr);
            g = {};
        }

        /// Moves the live generation to the retiring list (or destroys it when nothing used it).
        void Retire()
        {
            State& s = G();
            if (!s.live.generation && !s.live.hostTimeline) return;
            if (s.live.lastSerial == 0 || present::Completed() >= s.live.lastSerial)
            {
                Destroy(s.live);
                return;
            }
            for (Imported& r : s.retiring)
                if (!r.generation && !r.hostTimeline)
                {
                    r      = s.live;
                    s.live = {};
                    return;
                }
            // Both retiring places taken: the oldest one's frames are certainly done by now (two
            // generations ago); waiting on the device for it is the safe fallback.
            present::WaitSerial(s.retiring[0].lastSerial);
            Destroy(s.retiring[0]);
            s.retiring[0] = s.live;
            s.live        = {};
        }

        template <class F>
        bool Resolve(F& out, const char* name)
        {
            const context::Handles* h = context::Get();
            out = reinterpret_cast<F>(h->iapi->GetDeviceProcAddr(h->device, name));
            return out != nullptr;
        }
    }

    State& G()
    {
        static State state;
        return state;
    }

    bool Load()
    {
        State& s = G();
        if (s.loaded) return true;
        if (!s.interop || !context::Live()) return false;
        Functions& f = s.fn;
        bool ok = Resolve(f.createImage, "vkCreateImage") && Resolve(f.destroyImage, "vkDestroyImage")
                  && Resolve(f.getImageMemoryRequirements, "vkGetImageMemoryRequirements")
                  && Resolve(f.allocateMemory, "vkAllocateMemory") && Resolve(f.freeMemory, "vkFreeMemory")
                  && Resolve(f.bindImageMemory, "vkBindImageMemory") && Resolve(f.createSemaphore, "vkCreateSemaphore")
                  && Resolve(f.destroySemaphore, "vkDestroySemaphore")
                  && Resolve(f.importSemaphoreWin32HandleKHR, "vkImportSemaphoreWin32HandleKHR")
                  && Resolve(f.cmdPipelineBarrier, "vkCmdPipelineBarrier") && Resolve(f.cmdBlitImage, "vkCmdBlitImage")
                  && Resolve(f.cmdCopyImage, "vkCmdCopyImage");
        const context::Handles* h = context::Get();
        f.getPhysicalDeviceProperties2KHR = reinterpret_cast<PFN_vkGetPhysicalDeviceProperties2KHR>(
            vk::loader::Api().GetInstanceProcAddr(h->instance, "vkGetPhysicalDeviceProperties2KHR"));
        ok = ok && f.getPhysicalDeviceProperties2KHR;
        if (!ok) wxl::ScriptMgr::Log(WXL_LOG_WARN, "host: an interop entry point is missing; the host link is off");
        s.loaded  = ok;
        s.interop = ok;
        return ok;
    }

    void CollectRetired()
    {
        State& s = G();
        for (Imported& r : s.retiring)
            if ((r.generation || r.hostTimeline) && present::Completed() >= r.lastSerial) Destroy(r);
    }

    bool ImportWorld(const WXL_VkSharedWorld& world, char* why, uint32_t cap)
    {
        if (!Load())
        {
            CloseAll(world);
            if (why && cap) std::snprintf(why, cap, "the native device has no interop (WXL_HOST off at device creation, or a missing extension)");
            return false;
        }
        State& s = G();
        Retire();
        CollectRetired();
        Imported next;
        next.slots      = world.slotCount < WXL_VK_SHARED_SLOTS ? world.slotCount : WXL_VK_SHARED_SLOTS;
        next.format     = static_cast<VkFormat>(world.color[0].format);
        next.extent     = { world.color[0].width, world.color[0].height };
        bool ok = ImportTimeline(world.hostTimeline, next.hostTimeline, why, cap)
                  && ImportTimeline(world.clientTimeline, next.clientTimeline, why, cap);
        for (uint32_t i = 0; ok && i < next.slots; ++i) ok = ImportImage(world.color[i], next.images[i], next.memory[i], why, cap);
        CloseAll(world);   // an NT-handle import does not take ownership: the handles are ours to close
        if (!ok)
        {
            Destroy(next);
            return false;
        }
        next.generation      = world.generation;
        s.live               = next;
        s.stats.imported     = world.generation;
        return true;
    }

    void ReleaseWorld()
    {
        State& s = G();
        Retire();   // a composite already recorded keeps its timelines: they retire with its frame
        s.stats.imported = 0;
    }
}
