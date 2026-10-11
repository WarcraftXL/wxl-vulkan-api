// The Vulkan headers, included the one way this extension is allowed to include them, plus the
// value-to-text helpers every log line in src/vulkan/ needs.
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

// windows.h first, and not optional: third_party/vulkan-headers/vulkan/vulkan_win32.h names HWND
// and HINSTANCE in VkWin32SurfaceCreateInfoKHR without declaring them itself (checked -- the file
// has no #include at all), so the platform header has to already be in scope. Including it here
// rather than in each .cpp is also what keeps the two defines below from being a per-file ritual
// that one file eventually forgets.
#include <windows.h>

// VK_NO_PROTOTYPES before the header is what keeps this extension free of a Vulkan SDK and of a
// vulkan-1.lib import: it leaves the types, enums and PFN_ typedefs and drops the function
// declarations, so there is nothing to link and every entry point has to come from the loader.
#define VK_NO_PROTOTYPES
#define VK_USE_PLATFORM_WIN32_KHR
#include <vulkan/vulkan.h>

#include <cstdint>
#include <cstdio>

namespace wxlvk::vk
{
    // The headers are vendored, so these check that the vendored copy is the one being compiled
    // against and that it was parsed as 32-bit -- a handle size of 8 here would mean the whole
    // struct layout is wrong and every call would corrupt memory silently.
    //
    // They live here, in the one header every file under src/vulkan/ includes, because a 32-bit
    // parse is not a property of one file.
    static_assert(VK_HEADER_VERSION >= 363, "vendored third_party/vulkan-headers is older than pinned");
    static_assert(sizeof(VkInstance) == 4, "32-bit build: dispatchable handles must be 4 bytes");
    static_assert(sizeof(VkDeviceSize) == 8, "VkDeviceSize is 64-bit on every platform");

    // Non-dispatchable handles are uint64_t on a 32-bit build, not pointers. Worth asserting because
    // the usual mistake -- casting one through void* -- compiles on 64-bit and truncates here.
    static_assert(sizeof(VkSurfaceKHR) == 8, "non-dispatchable handles are 64-bit even on 32-bit");

    /**
     * A VkResult's enumerator name, for a log line.
     *
     * Only the results the bring-up can actually receive are named; anything else is reported as its
     * number, which is still enough to look up. A table of all ~60 enumerators would be mostly dead
     * text in a file that creates five objects.
     *
     * @param VkResult r : the result
     * @return string name : e.g. "VK_ERROR_INITIALIZATION_FAILED", or "VkResult(-1234)"
     */
    inline const char* ResultName(VkResult r)
    {
        switch (r)
        {
            case VK_SUCCESS:                        return "VK_SUCCESS";
            case VK_NOT_READY:                      return "VK_NOT_READY";
            case VK_TIMEOUT:                        return "VK_TIMEOUT";
            case VK_INCOMPLETE:                     return "VK_INCOMPLETE";
            case VK_ERROR_OUT_OF_HOST_MEMORY:       return "VK_ERROR_OUT_OF_HOST_MEMORY";
            case VK_ERROR_OUT_OF_DEVICE_MEMORY:     return "VK_ERROR_OUT_OF_DEVICE_MEMORY";
            case VK_ERROR_INITIALIZATION_FAILED:    return "VK_ERROR_INITIALIZATION_FAILED";
            case VK_ERROR_DEVICE_LOST:              return "VK_ERROR_DEVICE_LOST";
            case VK_ERROR_LAYER_NOT_PRESENT:        return "VK_ERROR_LAYER_NOT_PRESENT";
            case VK_ERROR_EXTENSION_NOT_PRESENT:    return "VK_ERROR_EXTENSION_NOT_PRESENT";
            case VK_ERROR_FEATURE_NOT_PRESENT:      return "VK_ERROR_FEATURE_NOT_PRESENT";
            case VK_ERROR_INCOMPATIBLE_DRIVER:      return "VK_ERROR_INCOMPATIBLE_DRIVER";
            case VK_ERROR_TOO_MANY_OBJECTS:         return "VK_ERROR_TOO_MANY_OBJECTS";
            case VK_ERROR_FORMAT_NOT_SUPPORTED:     return "VK_ERROR_FORMAT_NOT_SUPPORTED";
            case VK_ERROR_SURFACE_LOST_KHR:         return "VK_ERROR_SURFACE_LOST_KHR";
            case VK_ERROR_NATIVE_WINDOW_IN_USE_KHR: return "VK_ERROR_NATIVE_WINDOW_IN_USE_KHR";
            case VK_SUBOPTIMAL_KHR:                 return "VK_SUBOPTIMAL_KHR";
            case VK_ERROR_OUT_OF_DATE_KHR:          return "VK_ERROR_OUT_OF_DATE_KHR";
            case VK_ERROR_UNKNOWN:                  return "VK_ERROR_UNKNOWN";
            default:                                break;
        }
        // One static buffer, read immediately by the caller's log line. Single-threaded by
        // construction: every caller in src/vulkan/ runs on the render thread.
        static char text[32];
        std::snprintf(text, sizeof text, "VkResult(%d)", static_cast<int>(r));
        return text;
    }

    /// A VkPhysicalDeviceType as a word for the log.
    inline const char* DeviceTypeName(VkPhysicalDeviceType t)
    {
        switch (t)
        {
            case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU: return "integrated";
            case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:   return "discrete";
            case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU:    return "virtual";
            case VK_PHYSICAL_DEVICE_TYPE_CPU:            return "cpu";
            default:                                     return "other";
        }
    }

    /// A VkFormat as a name, for the handful the swapchain can land on.
    inline const char* FormatName(VkFormat f)
    {
        switch (f)
        {
            case VK_FORMAT_UNDEFINED:                 return "UNDEFINED";
            case VK_FORMAT_B8G8R8A8_UNORM:            return "B8G8R8A8_UNORM";
            case VK_FORMAT_B8G8R8A8_SRGB:             return "B8G8R8A8_SRGB";
            case VK_FORMAT_R8G8B8A8_UNORM:            return "R8G8B8A8_UNORM";
            case VK_FORMAT_R8G8B8A8_SRGB:             return "R8G8B8A8_SRGB";
            case VK_FORMAT_A2B10G10R10_UNORM_PACK32:  return "A2B10G10R10_UNORM_PACK32";
            case VK_FORMAT_A2R10G10B10_UNORM_PACK32:  return "A2R10G10B10_UNORM_PACK32";
            case VK_FORMAT_R16G16B16A16_SFLOAT:       return "R16G16B16A16_SFLOAT";
            case VK_FORMAT_R5G6B5_UNORM_PACK16:       return "R5G6B5_UNORM_PACK16";
            // The texture formats. Without these a log line said "VkFormat(137)" where it meant
            // BC3, which is a number somebody has to look up before the line means anything.
            case VK_FORMAT_A1R5G5B5_UNORM_PACK16:     return "A1R5G5B5_UNORM_PACK16";
            case VK_FORMAT_R8G8_SNORM:                return "R8G8_SNORM";
            case VK_FORMAT_R16G16_SFLOAT:             return "R16G16_SFLOAT";
            case VK_FORMAT_R32_SFLOAT:                return "R32_SFLOAT";
            case VK_FORMAT_BC1_RGBA_UNORM_BLOCK:      return "BC1_RGBA_UNORM_BLOCK";
            case VK_FORMAT_BC2_UNORM_BLOCK:           return "BC2_UNORM_BLOCK";
            case VK_FORMAT_BC3_UNORM_BLOCK:           return "BC3_UNORM_BLOCK";
            default:                                  break;
        }
        static char text[24];
        std::snprintf(text, sizeof text, "VkFormat(%d)", static_cast<int>(f));
        return text;
    }

    /// A VkColorSpaceKHR as a name.
    inline const char* ColorSpaceName(VkColorSpaceKHR c)
    {
        switch (c)
        {
            case VK_COLOR_SPACE_SRGB_NONLINEAR_KHR:    return "SRGB_NONLINEAR";
            case VK_COLOR_SPACE_EXTENDED_SRGB_LINEAR_EXT: return "EXTENDED_SRGB_LINEAR";
            case VK_COLOR_SPACE_HDR10_ST2084_EXT:      return "HDR10_ST2084";
            default:                                   break;
        }
        static char text[28];
        std::snprintf(text, sizeof text, "VkColorSpace(%d)", static_cast<int>(c));
        return text;
    }

    /// A VkPresentModeKHR as a name.
    inline const char* PresentModeName(VkPresentModeKHR m)
    {
        switch (m)
        {
            case VK_PRESENT_MODE_IMMEDIATE_KHR:    return "IMMEDIATE";
            case VK_PRESENT_MODE_MAILBOX_KHR:      return "MAILBOX";
            case VK_PRESENT_MODE_FIFO_KHR:         return "FIFO";
            case VK_PRESENT_MODE_FIFO_RELAXED_KHR: return "FIFO_RELAXED";
            default:                               break;
        }
        static char text[28];
        std::snprintf(text, sizeof text, "VkPresentMode(%d)", static_cast<int>(m));
        return text;
    }

    /**
     * An API version packed by VK_MAKE_API_VERSION, as "major.minor.patch".
     *
     * @param uint32 version : the packed version
     * @param char* buf : receives the text
     * @param size_t cap : capacity of buf
     * @return string buf, for use straight inside a log call
     */
    inline const char* ApiVersionText(uint32_t version, char* buf, size_t cap)
    {
        std::snprintf(buf, cap, "%u.%u.%u", VK_API_VERSION_MAJOR(version),
                      VK_API_VERSION_MINOR(version), VK_API_VERSION_PATCH(version));
        return buf;
    }

    /**
     * A driverVersion as text, decoded per vendor.
     *
     * `VkPhysicalDeviceProperties::driverVersion` is explicitly vendor-defined: the spec puts no
     * structure on it at all, so VK_API_VERSION_MAJOR and friends are only right by coincidence. Two
     * vendors are decoded here because they are the two that matter on this machine and on any
     * machine a 32-bit WoW client runs on:
     *
     *  - NVIDIA (0x10DE) packs 10/8/8/6 bits as major/minor/secondary-branch/tertiary-branch, which
     *    is what reproduces the "581.42" the control panel shows. Without this, an NVIDIA driver
     *    reads out as a nonsense "2325.168.0".
     *  - Intel on Windows (0x8086) packs 14/18 bits as major/minor.
     *
     * Everything else falls through to the Vulkan packing, which AMD and Mesa do follow. The decode
     * is a vendor convention documented by the vendors, not something measured here, so a wrong
     * number in the log means this table is wrong -- not that the device is.
     *
     * @param uint32 vendorId : VkPhysicalDeviceProperties::vendorID
     * @param uint32 version : VkPhysicalDeviceProperties::driverVersion
     * @param char* buf : receives the text
     * @param size_t cap : capacity of buf
     * @return string buf, for use straight inside a log call
     */
    inline const char* DriverVersionText(uint32_t vendorId, uint32_t version, char* buf, size_t cap)
    {
        if (vendorId == 0x10DE)
        {
            std::snprintf(buf, cap, "%u.%u.%u.%u", (version >> 22) & 0x3FF, (version >> 14) & 0x0FF,
                          (version >> 6) & 0x0FF, version & 0x03F);
            return buf;
        }
        if (vendorId == 0x8086)
        {
            std::snprintf(buf, cap, "%u.%u", version >> 14, version & 0x3FFF);
            return buf;
        }
        return ApiVersionText(version, buf, cap);
    }
}
