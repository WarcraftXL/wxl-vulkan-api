// wxl-vulkan-api: a native Vulkan CGxDevice for the 3.3.5a client.
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

#include "common/Settings.hpp"
#include "device/Backend.hpp"
#include "host/Host.hpp"
#include "native/Perf.hpp"
#include "native/Render.hpp"
#include "native/render/DoodadBatch.hpp"
#include "native/render/WmoGroups.hpp"
#include "native/render/WorldPass.hpp"

#include "wxl/Script.hpp"

WXL_DECLARE_EXTENSION("wxl-vulkan-api", 1)

/**
 * Registers the device factory when WXL_VK_DEVICE=native, and does nothing otherwise.
 *
 * No script is registered. The factory has to be in place before GxDevCreate runs, and the core
 * loads extensions from a detour on the engine-init seam, ahead of the body that eventually reaches
 * GxDevCreate -- so WXL_Load is the one moment that is early enough. Everything else the device does
 * happens inside its own vtable slots, from slot 13 on, and in the rendering API's insertion points
 * (wxl.vk.render), whose engine seams are attached here as well, while the core still enables hooks.
 *
 * With the key unset (or set to anything but `native`) nothing is registered and nothing is hooked:
 * the engine builds its stock D3D9 device exactly as it does without this extension installed.
 */
void AddScripts()
{
    using wxlvk::settings::DeviceMode;

    switch (wxlvk::settings::DeviceModeWanted())
    {
        case DeviceMode::Native:
            wxlvk::native::perf::Init();   // WXL_VK_STATS, before the first slot runs
            wxlvk::device::Install();
            wxlvk::native::render::Install();
            wxlvk::native::render::worldpass::Install();   // BeforeUi, the draw capture, the jitter
            wxlvk::native::render::doodadbatch::Install();   // the capture tells the units from the placed doodads
            wxlvk::native::render::wmogroups::Install();     // the G-buffer tells an interior group's batches
            wxlvk::host::Install();   // WXL_HOST only: wxl.vk.hostlink and its composite seam
            return;
        case DeviceMode::Unrecognised:
            wxl::ScriptMgr::Log(WXL_LOG_WARN,
                                "wxl-vulkan-api: %s is set to a value other than \"native\"; the "
                                "extension stays inert and the stock D3D9 device is used",
                                wxlvk::settings::kDeviceKey);
            return;
        default:
            return;
    }
}
