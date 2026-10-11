// The Vulkan instance: the loader, the layers, the extensions.
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

namespace wxlvk::vk::context::detail
{
    // --- instance ---------------------------------------------------------------------------------

    /**
     * Creates the instance, with the Win32 surface extensions and optionally validation.
     *
     * The device's own drawing needs only apiVersion 1.0: nothing it draws uses a function past
     * Vulkan 1.0 core plus VK_KHR_surface / VK_KHR_win32_surface / VK_KHR_swapchain. It asks for the
     * loader's own version capped at 1.2 (Caps.cpp InstanceVersion), because the optional capabilities
     * the modules may use -- FP16, 16-bit storage, subgroups, multiview, the memory budget -- are probed
     * and enabled through 1.1 core (vkGetPhysicalDeviceFeatures2 and its chains). A loader without
     * vkEnumerateInstanceVersion is 1.0 and gets a 1.0 instance, as before; WXL_VK_CAPS off asks for 1.0.
     *
     * @return bool ok
     */
    bool CreateInstance()
    {
        const loader::Global& api = loader::Api();

        uint32_t count = 0;
        if (api.EnumerateInstanceExtensionProperties(nullptr, &count, nullptr) != VK_SUCCESS)
            return false;
        std::vector<VkExtensionProperties> available(count);
        if (count && api.EnumerateInstanceExtensionProperties(nullptr, &count, available.data())
                         != VK_SUCCESS)
            return false;
        available.resize(count);

        // Both are hard requirements: without them there is no surface, and without a surface the
        // physical-device choice cannot be made on present support, which is the only defensible way
        // to make it.
        if (!HasExtension(available, VK_KHR_SURFACE_EXTENSION_NAME)
            || !HasExtension(available, VK_KHR_WIN32_SURFACE_EXTENSION_NAME))
        {
            wxl::ScriptMgr::Log(WXL_LOG_WARN,
                                "vulkan: the loader offers %u instance extension(s) but not both of "
                                "%s and %s; declining",
                                count, VK_KHR_SURFACE_EXTENSION_NAME,
                                VK_KHR_WIN32_SURFACE_EXTENSION_NAME);
            return false;
        }

        std::vector<const char*> extensions;
        extensions.push_back(VK_KHR_SURFACE_EXTENSION_NAME);
        extensions.push_back(VK_KHR_WIN32_SURFACE_EXTENSION_NAME);
        wxlvk::host::ExtendInstance(available, extensions);   // WXL_HOST only: the interop capabilities

        std::vector<const char*> layers;
        if (settings::ValidationWanted())
        {
            static const char* const kValidationLayer = "VK_LAYER_KHRONOS_validation";

            bool layerPresent = false;
            if (api.EnumerateInstanceLayerProperties)
            {
                uint32_t layerCount = 0;
                if (api.EnumerateInstanceLayerProperties(&layerCount, nullptr) == VK_SUCCESS)
                {
                    std::vector<VkLayerProperties> props(layerCount);
                    if (layerCount
                        && api.EnumerateInstanceLayerProperties(&layerCount, props.data()) == VK_SUCCESS)
                    {
                        props.resize(layerCount);
                        for (const VkLayerProperties& p : props)
                            if (std::strcmp(p.layerName, kValidationLayer) == 0) layerPresent = true;
                    }
                }
            }

            // Requested is not enabled. A layer that is not installed must be a log line, not a
            // VK_ERROR_LAYER_NOT_PRESENT from vkCreateInstance -- the user asked for validation, not
            // for the device to fail without it.
            if (layerPresent && HasExtension(available, VK_EXT_DEBUG_UTILS_EXTENSION_NAME))
            {
                layers.push_back(kValidationLayer);
                extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
                g.validation = true;
                g.debugUtils = true;
            }
            else
            {
                wxl::ScriptMgr::Log(WXL_LOG_WARN,
                                    "vulkan: %s asked for validation but %s; continuing without it",
                                    settings::kValidationKey,
                                    layerPresent ? "the loader has no " VK_EXT_DEBUG_UTILS_EXTENSION_NAME
                                                 : "VK_LAYER_KHRONOS_validation is not installed");
            }
        }

        VkApplicationInfo app = {};
        app.sType              = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        app.pApplicationName   = "World of Warcraft 3.3.5a (12340)";
        app.applicationVersion = VK_MAKE_API_VERSION(0, 3, 3, 5);
        app.pEngineName        = "WarcraftXL / wxl-vulkan-api";
        app.engineVersion      = VK_MAKE_API_VERSION(0, 0, 2, 0);
        app.apiVersion         = InstanceVersion();

        const VkDebugUtilsMessengerCreateInfoEXT messenger = MessengerInfo();

        VkInstanceCreateInfo info = {};
        info.sType                   = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        // Chaining the messenger here, and not only creating one afterwards, is what makes the
        // layers' complaints about vkCreateInstance itself visible -- they are emitted before any
        // messenger object can exist.
        info.pNext                   = g.debugUtils ? &messenger : nullptr;
        info.pApplicationInfo        = &app;
        info.enabledLayerCount       = static_cast<uint32_t>(layers.size());
        info.ppEnabledLayerNames     = layers.empty() ? nullptr : layers.data();
        info.enabledExtensionCount   = static_cast<uint32_t>(extensions.size());
        info.ppEnabledExtensionNames = extensions.data();

        VkResult r = api.CreateInstance(&info, nullptr, &g.instance);

        // One retry without layers, and only when layers were requested.
        //
        // Enumerating a layer is not the same as being able to load it, and on 32-bit Windows the
        // difference is routine rather than exotic: a 64-bit Vulkan SDK registers
        // VK_LAYER_KHRONOS_validation under HKLM\SOFTWARE\WOW6432Node\Khronos\Vulkan\ExplicitLayers
        // with a manifest whose library_path is the 64-bit DLL. vkEnumerateInstanceLayerProperties
        // reports it present, the availability check above passes, and vkCreateInstance then fails
        // with LAYER_NOT_PRESENT or INITIALIZATION_FAILED.
        //
        // Without this retry that is a permanent decline. Losing validation is a log line; losing
        // the device because validation is unloadable is an hour spent on the wrong question.
        if (r != VK_SUCCESS && !layers.empty())
        {
            wxl::ScriptMgr::Log(WXL_LOG_WARN,
                                "vulkan: vkCreateInstance failed with %u layer(s) requested: %s. "
                                "The layer enumerated but would not load -- on a 32-bit process "
                                "that usually means a 64-bit validation layer is registered. "
                                "Retrying without layers; validation will be off.",
                                static_cast<unsigned>(layers.size()), ResultName(r));

            info.enabledLayerCount   = 0;
            info.ppEnabledLayerNames = nullptr;
            g.instance               = nullptr;

            r = api.CreateInstance(&info, nullptr, &g.instance);
            if (r == VK_SUCCESS) g.validation = false;
        }

        if (r != VK_SUCCESS)
        {
            wxl::ScriptMgr::Log(WXL_LOG_WARN, "vulkan: vkCreateInstance failed: %s", ResultName(r));
            g.instance = nullptr;
            return false;
        }

        if (!loader::LoadInstance(g.instance, g.debugUtils, g.iapi))
        {
            // The instance exists but is unusable. Destroy it here rather than leaving it to
            // Teardown(), because the table it would be destroyed through is the one that just came
            // back incomplete -- DestroyInstance is the one entry point checked for first, so if it
            // is present it is safe to use, and if it is not there is nothing to be done anyway.
            if (g.iapi.DestroyInstance) g.iapi.DestroyInstance(g.instance, nullptr);
            g.instance = nullptr;
            return false;
        }

        if (g.debugUtils && g.iapi.CreateDebugUtilsMessengerEXT)
        {
            const VkResult m = g.iapi.CreateDebugUtilsMessengerEXT(g.instance, &messenger, nullptr,
                                                                   &g.messenger);
            if (m != VK_SUCCESS)
            {
                g.messenger = VK_NULL_HANDLE;
                wxl::ScriptMgr::Log(WXL_LOG_WARN,
                                    "vulkan: vkCreateDebugUtilsMessengerEXT failed: %s; the instance "
                                    "stands, validation messages after creation are lost",
                                    ResultName(m));
            }
        }

        wxl::ScriptMgr::Log(WXL_LOG_DEBUG,
                            "vulkan: instance created: api %u.%u requested, %u extension(s), "
                            "validation %s",
                            VK_API_VERSION_MAJOR(app.apiVersion), VK_API_VERSION_MINOR(app.apiVersion),
                            static_cast<unsigned>(extensions.size()),
                            g.validation ? "on" : "off");
        return true;
    }
}
