// The Vulkan context's shared internals: its state and the helpers every part of src/vulkan/context/
// uses.
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


#include "vulkan/Context.hpp"

#include "vulkan/DeviceCaps.hpp"

#include "vulkan/Footprint.hpp"
#include "vulkan/Loader.hpp"
#include "vulkan/Window.hpp"

#include "common/Settings.hpp"
#include "host/Host.hpp"

#include "wxl/Script.hpp"

#include <cstring>
#include <vector>

// The context lives in src/vulkan/context/, one file per stage of its bring-up, over this one header.
// Nothing here is visible outside it: the public face is src/vulkan/Context.hpp.
//
//   Debug      validation messages, extension lookups, footprint lines
//   Instance   the instance
//   Surface    the window's surface
//   Physical   the physical device and its queue
//   Logical    the logical device
//   Swapchain  the swapchain and its rebuild
namespace wxlvk::vk::context::detail
{
    namespace loader    = wxlvk::vk::loader;
    namespace footprint = wxlvk::vk::footprint;
    namespace window    = wxlvk::vk::window;
    namespace settings  = wxlvk::settings;

    using wxlvk::vk::ApiVersionText;
    using wxlvk::vk::ColorSpaceName;
    using wxlvk::vk::DeviceTypeName;
    using wxlvk::vk::DriverVersionText;
    using wxlvk::vk::FormatName;
    using wxlvk::vk::PresentModeName;
    using wxlvk::vk::ResultName;

    /// Everything this module owns, in one place so Teardown() cannot miss a field.
    ///
    /// Dispatchable handles (VkInstance, VkPhysicalDevice, VkDevice, VkQueue) are pointers;
    /// non-dispatchable ones (VkSurfaceKHR, VkSwapchainKHR, VkDebugUtilsMessengerEXT) are uint64_t
    /// on this build, because third_party/vulkan-headers/vulkan/vulkan_core.h sets
    /// VK_USE_64_BIT_PTR_DEFINES to 0 for anything that is not a 64-bit target. That is why the two
    /// groups are initialised differently below -- nullptr for the pointers, VK_NULL_HANDLE (which is
    /// 0ULL here) for the integers -- rather than uniformly, which would not compile.
    struct State
    {
        VkInstance               instance  = nullptr;
        VkPhysicalDevice         physical  = nullptr;
        VkDevice                 device    = nullptr;
        VkQueue                  graphicsQueue = nullptr;
        VkQueue                  presentQueue  = nullptr;

        VkSurfaceKHR             surface   = VK_NULL_HANDLE;
        VkSwapchainKHR           swapchain = VK_NULL_HANDLE;
        VkDebugUtilsMessengerEXT messenger = VK_NULL_HANDLE;

        uint32_t graphicsFamily = UINT32_MAX;
        uint32_t presentFamily  = UINT32_MAX;

        HWND               hwnd        = nullptr;
        VkSurfaceFormatKHR surfaceFormat = { VK_FORMAT_UNDEFINED, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR };
        VkPresentModeKHR   presentMode = VK_PRESENT_MODE_FIFO_KHR;
        VkExtent2D         extent      = { 0, 0 };
        uint32_t           imageCount  = 0;
        VkImageUsageFlags  usage       = 0;  ///< the imageUsage the swapchain was created with
        bool               bc          = false; ///< textureCompressionBC was enabled on the device
        bool               largePoints = false; ///< largePoints was enabled on the device (gl_PointSize above 1)
        bool               anisotropy  = false; ///< samplerAnisotropy was enabled on the device
        bool               preciseOcclusion = false; ///< occlusionQueryPrecise was enabled on the device
        bool               independentBlend = false; ///< independentBlend was enabled: per-attachment blend state (the G-buffer pass class)

        bool vsync      = true;    ///< the engine's format asks for vsync (SetVsync)

        /// The optional capabilities (Caps.cpp, WXL_VK_CAPS): what was probed, asked for and enabled.
        DeviceCaps caps;
        /// vkGetPhysicalDeviceMemoryProperties2 (a 1.1 instance) for the memory budget; null without it.
        PFN_vkGetPhysicalDeviceMemoryProperties2 memoryProperties2 = nullptr;

        bool debugUtils = false;   ///< VK_EXT_debug_utils was enabled on the instance
        bool validation = false;   ///< VK_LAYER_KHRONOS_validation was enabled on the instance

        loader::Instance iapi;
        loader::Device   dapi;

        footprint::Snapshot baseline;     ///< before LoadLibraryA
        footprint::Snapshot afterLoader;
        footprint::Snapshot afterInstance;
        footprint::Snapshot afterDevice;

        bool     settled  = false; ///< Bring() reached Done or Declined; stop calling it
        unsigned attempts = 0;

        /// Bumped on every swapchain create and destroy. See context::SwapchainGeneration().
        uint32_t swapGen = 0;
    };
    extern State g;

    /// How many times Bring() may answer Retry before it gives up and declines.
    ///
    /// Slot 13 calls it right after ICreateWindow, and slot 38 calls it again while the window is
    /// not measurable (minimised, or a zero-size client rect mid-resize). 600 is about ten seconds of
    /// presents at 60 fps: long enough to cover an alt-tab during loading, short enough that a
    /// permanently unusable window stops being retried.
    constexpr unsigned kMaxAttempts = 600;


    /// The validation messages, per message id: the first kValidationRepeats of an id are logged
    /// whole, the rest counted, so an error a draw raises does not write a line per draw (a frame is
    /// tens of thousands of them) and bury the first occurrence. LogValidationSummary() names the
    /// counted ones. The layer may call from any thread that calls Vulkan; the table is locked.
    constexpr uint32_t kValidationRepeats = 5;
    constexpr uint32_t kValidationIds     = 128;
    struct ValidationId
    {
        int32_t     id       = 0;
        int         level    = 0;
        uint64_t    count    = 0;
        char        name[96] = {};
    };
    struct ValidationTable
    {
        SRWLOCK      lock = SRWLOCK_INIT;
        ValidationId ids[kValidationIds];
        uint32_t     idCount   = 0;
        uint64_t     untracked = 0;   ///< messages of ids past the table, all logged
        uint64_t     errors    = 0;
        uint64_t     warnings  = 0;
    };
    extern ValidationTable g_validation;

    /// One enumerated adapter, with everything the choice is made on.
    struct Candidate
    {
        VkPhysicalDevice           handle = nullptr;
        VkPhysicalDeviceProperties props  = {};
        uint32_t index          = 0;
        uint32_t graphicsFamily = UINT32_MAX;
        uint32_t presentFamily  = UINT32_MAX;
        uint64_t deviceLocal    = 0;       ///< bytes across its DEVICE_LOCAL heaps
        bool     unified        = false;   ///< one family does both graphics and present
        const char* reject      = nullptr; ///< non-null when the adapter cannot be used
    };

    // Debug.cpp
    bool HasExtension(const std::vector<VkExtensionProperties>& list, const char* name);
    VKAPI_ATTR VkBool32 VKAPI_CALL DebugCallback( VkDebugUtilsMessageSeverityFlagBitsEXT severity, VkDebugUtilsMessageTypeFlagsEXT types, const VkDebugUtilsMessengerCallbackDataEXT* data, void* user);
    VkDebugUtilsMessengerCreateInfoEXT MessengerInfo();
    void LogFootprint(const char* phase, const footprint::Snapshot& now);

    // Instance.cpp
    bool CreateInstance();

    // Surface.cpp
    bool CreateSurface(HWND hwnd);

    // Physical.cpp
    unsigned TypeRank(VkPhysicalDeviceType t);
    const char* WhyBetter(const Candidate& a, const Candidate& b);
    void Examine(Candidate& c);
    bool SelectPhysicalDevice();

    // Logical.cpp
    bool CreateDevice();

    // Caps.cpp: the optional capabilities (WXL_VK_CAPS).
    /// The instance version to ask for: the loader's up to 1.2, or 1.0 with WXL_VK_CAPS off. Notes it in g.caps.
    uint32_t InstanceVersion();
    /// What the device is created with beyond the six core features: the extension feature structs (pNext,
    /// chained in front of `tail`), the extensions (appended to `names`, `count` updated, at most `room`),
    /// the core features (set in `features`). Probes the chosen adapter first; nothing with WXL_VK_CAPS off.
    struct CapsChain
    {
        VkPhysicalDevice16BitStorageFeatures      storage16 = {};
        VkPhysicalDeviceShaderFloat16Int8Features float16   = {};
        VkPhysicalDeviceMultiviewFeatures         multiview = {};
        void*                                     head      = nullptr;
    };
    void PlanCaps(VkPhysicalDeviceFeatures& features, CapsChain& chain, void* tail, const char** names, uint32_t& count,
                  uint32_t room);
    /// After vkCreateDevice: what was enabled, the budget's entry point, the log line.
    void CapsCreated(bool chainUsed);

    // Swapchain.cpp
    VkSurfaceFormatKHR PickFormat(const std::vector<VkSurfaceFormatKHR>& formats, const char*& why);
    bool CreateSwapchain(const window::Found& win);
    void DropSwapchainObject();
    void DropSurface();
}
