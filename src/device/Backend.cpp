// Building a real CGxDeviceD3d through the engine's own factory and giving it the native vtable.
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

#include "device/Backend.hpp"

#include "device/Slots.hpp"
#include "device/VTable.hpp"
#include "native/Native.hpp"
#include "offsets/GxDeviceD3d.hpp"

#include "wxl/GraphicsDeviceApi.h"
#include "wxl/Script.hpp"
#include "wxl/game/Binding.hpp"
#include "wxl/offsets/engine/Gx.hpp"
#include "wxl/offsets/engine/GxDevice.hpp"

#include <cstdint>

// Why the object is built by CGxDevice::NewD3d rather than assembled out of the core's
// InitBaseDevice() and GetBaseVTable():
//
// The native table keeps 46 of the stock D3D9 entries -- base slots and D3D overrides that never
// touch the IDirect3DDevice9 -- and those are CGxDeviceD3d methods reading CGxDeviceD3d fields. The
// base CGxDevice vtable has 24 __purecall entries, so a base-constructed object has nothing to keep
// for them, and the D3D methods read fields a base-constructed object does not have filled.
//
// So the object genuinely IS a CGxDeviceD3d, built the way the engine builds one. The engine's own
// factory does that in nine instructions, and calling it is both shorter and more faithful than
// reproducing it: same allocator, same size, same allocation tag and line, the same constructor, and
// therefore the same CGxDevice::DeviceCreatePools and DeviceCreateStreamBufs at the end of it --
// which the shader system's shutdown releases through, and whose absence crashes the client on exit.
//
// What is left for us is the vtable pointer the constructor writes at offset 0: it is overwritten
// with the native table (src/device/VTable.cpp).
namespace
{
    namespace d3d   = wxlvk::offsets::gxdevd3d;
    namespace gx    = wxl::offsets::engine::gx;
    namespace gxdev = wxl::offsets::engine::gxdevice;
    namespace slots = wxlvk::device::slots;

    /// What RegisterFactory records the factory under, and what the core names in its own logs.
    constexpr const char* kFactoryName = "wxl-vulkan-api/native";

    /**
     * Destroys a device through its own vtable slot 8 with the deleting flag set.
     *
     * The same thing GxDevCreate does when DeviceCreate fails: the scalar deleting destructor runs
     * CGxDeviceD3d's own teardown, then CGxDevice::~CGxDevice, then SMemFree. Called only on the
     * path where we built an object and then decided not to hand it over.
     *
     * @param void* device : the device, whose vtable pointer must still be one we trust
     */
    void DestroyThroughVTable(void* device)
    {
        void* const* table = *reinterpret_cast<void* const* const*>(device);
        const auto del = reinterpret_cast<wxlvk::device::ScalarDeletingDtorFn>(
            table[slots::kScalarDeletingDtor]);
        del(device, nullptr, 1);
    }

    /**
     * The core's WXL_DeviceCreateFn: construct the device, and nothing else.
     *
     * The core publishes whatever this returns as the engine device and then calls its slot 10
     * DeviceCreate with the wndProc and format -- which is why neither is touched here. Returning
     * null makes the core fall back to the engine's own CGxDevice::New*, so every refusal below
     * leaves the user with the stock client.
     *
     * A D3D9Ex request (api 2) is built as the D3D9 object (api 1): the ~30 engine branches on
     * CGxDevice::DeviceApi() treat 1 and 2 alike for everything the native device relies on, and one
     * object layout is one set of offsets to trust.
     *
     * @param int requested : an EGxApi, as the client's gxApi CVar resolved it
     * @param void* wndProc : the client WNDPROC; the core passes it on to slot 10
     * @param void* context : the CGxFormat; likewise
     * @return void* device : our CGxDeviceD3d, or null to let the engine build its own
     */
    void* __cdecl CreateDevice(int requested, void* /*wndProc*/, void* /*context*/)
    {
        if (requested != d3d::kApiD3d9 && requested != d3d::kApiD3d9Ex)
        {
            // api 0 is OpenGL: a different object size, constructor and vtable, none of which has
            // been read against. Decline rather than guess; the engine builds its own backend.
            wxl::ScriptMgr::Log(WXL_LOG_WARN,
                                "device: GxDevCreate asked for api %d, which is not one of the two "
                                "D3D backends; letting the engine build its own",
                                requested);
            return nullptr;
        }
        const int api = d3d::kApiD3d9;

        const void* const* table = wxlvk::device::VTable();
        if (!table) return nullptr; // VTable() logged why

        void* device = reinterpret_cast<gxdev::NewDeviceFn>(gxdev::kNewD3d)();
        if (!device)
        {
            wxl::ScriptMgr::Log(WXL_LOG_ERROR,
                                "device: the engine's own factory at 0x%08X returned null (the "
                                "allocation failed); letting the engine try again its way",
                                static_cast<unsigned>(gxdev::kNewD3d));
            return nullptr;
        }

        // Two things the engine's constructor has just done, both of which the native table depends
        // on: it installed the D3D9 vtable, and it wrote the api id the engine functions that branch
        // on CGxDevice::DeviceApi() read. If either is not what this code was written against, the
        // object is not the one we think it is and the offsets are wrong for this binary.
        const uintptr_t installed = *reinterpret_cast<const uintptr_t*>(device);
        const uint32_t  apiField  = wxl::game::At<uint32_t>(device, gxdev::kApi);

        if (installed != gx::kGxDeviceVTable || apiField != static_cast<uint32_t>(api))
        {
            wxl::ScriptMgr::Log(WXL_LOG_ERROR,
                                "device: the object the engine's factory built is not the one "
                                "expected (vtable 0x%08X, wanted 0x%08X; api field %u, wanted %d); "
                                "declining",
                                static_cast<unsigned>(installed),
                                static_cast<unsigned>(gx::kGxDeviceVTable), apiField, api);

            if (installed == gx::kGxDeviceVTable)
            {
                DestroyThroughVTable(device);
            }
            else
            {
                // The vtable pointer is not one we recognise, so slot 8 is not a function we are
                // willing to call. One leaked allocation at startup is the cheaper mistake.
                wxl::ScriptMgr::Log(WXL_LOG_WARN,
                                    "device: leaking the %u-byte allocation rather than calling slot "
                                    "8 through a vtable pointer we do not recognise",
                                    static_cast<unsigned>(gxdev::kD3d9ObjectSize));
            }
            return nullptr;
        }

        *reinterpret_cast<const void* const**>(device) = table;

        wxl::ScriptMgr::Log(WXL_LOG_DEBUG,
                            "device: CGxDeviceD3d at 0x%08X built by the engine's own factory (api %d "
                            "requested); native table: %u native, %u stub, %u stock slots (%u of %u "
                            "differ from stock)",
                            static_cast<unsigned>(reinterpret_cast<uintptr_t>(device)), requested,
                            wxlvk::native::CountOf(wxlvk::native::Role::Native),
                            wxlvk::native::CountOf(wxlvk::native::Role::Stub),
                            wxlvk::native::CountOf(wxlvk::native::Role::Stock),
                            wxlvk::device::OverriddenSlots(), wxlvk::device::kSlotCount);
        return device;
    }
}

namespace wxlvk::device
{
    bool Install()
    {
        // The core publishes this in its Boot phase, before any extension is loaded, so by the time
        // WXL_Load runs it is either there or the core's gx-device-create feature failed to install.
        const WXL_GraphicsDeviceApi* api =
            wxl::ScriptMgr::GetInterface<const WXL_GraphicsDeviceApi>(WXL_GRAPHICS_DEVICE_API_NAME,
                                                                      WXL_GRAPHICS_DEVICE_API_VERSION);
        if (!api)
        {
            wxl::ScriptMgr::Log(WXL_LOG_WARN,
                                "device: %s v%u is not published; there is no seam to supply a "
                                "CGxDevice through, so the stock device is used",
                                WXL_GRAPHICS_DEVICE_API_NAME, WXL_GRAPHICS_DEVICE_API_VERSION);
            return false;
        }

        if (!api->RegisterFactory(&CreateDevice, kFactoryName))
        {
            wxl::ScriptMgr::Log(WXL_LOG_ERROR,
                                "device: the core refused the factory registration; GxDevCreate will "
                                "build the engine's own backend");
            return false;
        }

        wxl::ScriptMgr::Log(WXL_LOG_DEBUG, "device: registered as %s", kFactoryName);
        return true;
    }
}
