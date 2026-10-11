// The native slot set: which of the 84 slots are ours, which are stubs, which stay stock.
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

#include "native/Native.hpp"

#include "native/Slots.hpp"
#include "offsets/GxDeviceD3d.hpp"

#include "wxl/Script.hpp"

#include <cstdint>

// The per-slot D3D dependency, read slot by slot in the decompilation: 38 stock D3D overrides reach
// the IDirect3DDevice9 directly or through ISceneBegin / ITexCreate / IReleaseD3dPool /
// IShaderCreate*, so in a device that never creates one they must not run -- 0, 1, 4, 9-14, 20,
// 22-27, 37-39, 42, 52-56, 58-60, 62-69, 71, 72. Four more are native because the native device
// offers what their stock versions do with D3D: 15 / 16 (gamma, applied in the present pass) and
// 2 / 73 (the hardware cursor, which caps +0x310 = 1 lets the engine ask for). Every other slot is
// either a base entry or one of the D3D overrides that are D3D-free (3, 8, 17, 21, 28, 35, 36,
// 40, 41, 75-83), and stays.
//
// The list is written out as data, an allow-list in effect, rather than derived: the one risk of
// this construction is a stock D3D slot left in the table that some path reaches, and the defence is
// that every slot's fate is a line someone wrote on purpose.
namespace
{
    using wxlvk::native::Role;
    using wxlvk::device::kSlotCount;
    using wxlvk::device::kSlotNames;
    using wxlvk::device::SlotTraits;

    namespace slots = wxlvk::device::slots;
    namespace impl  = wxlvk::native::impl;
    namespace gxdev = wxl::offsets::engine::gxdevice;
    namespace d3d   = wxlvk::offsets::gxdevd3d;

    /// The implemented slots.
    constexpr unsigned kNative[] = {
        slots::kDeviceCreate, slots::kDeviceDestroy, slots::kDeviceSetFormat, slots::kDeviceWM,
        slots::kScenePresent, slots::kSceneClear, slots::kDeviceEvictResources, slots::kICursorDraw,
        slots::kITexMarkAsUpdated, slots::kIRsSendToHw, slots::kDraw, slots::kPoolSizeSet,
        slots::kPoolDestroy, slots::kBufLock, slots::kBufUnlock, slots::kBufData, slots::kTexDestroy,
        slots::kShaderCreate, slots::kShaderDestroy, slots::kIShaderCreate,
        slots::kDeviceReadPixels, slots::kDeviceSetRenderTarget, slots::kCaptureDepthStencilSurface,
        slots::kBindDepthStencilSurface, slots::kDeviceResolveDepthBuffer, slots::kDeviceCopyTex,
        slots::kTexCopy, slots::kTexStretch,
        slots::kQueryCreate, slots::kQueryDestroy, slots::kQueryBegin, slots::kQueryEnd,
        slots::kQueryGetParam, slots::kQueryGetData,
        slots::kDeviceSetBaseMipLevel, slots::kDeviceSetGammaRamp, slots::kDeviceSetGammaLevel,
        slots::kICursorCreate, slots::kCursorSetVisible,
    };

    /// The rest of the must-not-reach-D3D set: stubs (src/native/Stubs.cpp). 71 was never reached;
    /// 9 has no caller.
    constexpr unsigned kStub[] = {
        slots::kDeviceCreateOnWindow, slots::kLogCrashInfo, slots::kIShaderReload,
    };

    static_assert(sizeof kNative / sizeof *kNative + sizeof kStub / sizeof *kStub == 42,
                  "the 38 D3D-dependent slots plus 2, 15, 16 and 73");

    /// The slots whose BASE entry a native slot or stub forwards to (native::Base<Slot>()).
    constexpr unsigned kBaseUsed[] = {
        slots::kITexMarkAsUpdated, slots::kDeviceCreate, slots::kDeviceDestroy,
        slots::kDeviceSetFormat, slots::kDeviceSetBaseMipLevel, slots::kDeviceSetRenderTarget,
        slots::kScenePresent, slots::kPoolDestroy, slots::kBufLock, slots::kBufUnlock,
        slots::kBufData, slots::kTexDestroy, slots::kQueryCreate, slots::kQueryDestroy,
        slots::kShaderCreate, slots::kShaderDestroy, slots::kICursorDraw,
        slots::kICursorCreate, slots::kCursorSetVisible, slots::kDeviceSetGammaRamp,
        slots::kDeviceSetGammaLevel,
    };

    /**
     * Writes one slot, with the compiler checking the signature against the slot table: Fn is
     * `SlotTraits<Slot>::Fn`, pop size included (src/device/Slots.hpp), so a function whose argument
     * list does not match the slot is a compile error rather than a stack desync at the first call.
     */
    template <unsigned Slot>
    void Put(void** table, typename SlotTraits<Slot>::Fn fn)
    {
        static_assert(Slot < kSlotCount, "slot index outside the engine's vtable");
        table[Slot] = reinterpret_cast<void*>(fn);
    }

    template <size_t N>
    bool Contains(const unsigned (&list)[N], unsigned slot)
    {
        for (unsigned s : list)
            if (s == slot) return true;
        return false;
    }

    Role RoleOf(unsigned slot)
    {
        if (Contains(kNative, slot)) return Role::Native;
        if (Contains(kStub, slot)) return Role::Stub;
        return Role::Stock;
    }

    /**
     * True when every base entry a native slot forwards to is real code.
     *
     * 24 base entries are __purecall; a native slot calling the base half of one of those would abort
     * the client. None of kBaseUsed is among them in the reference's table, and this makes it a
     * checked fact on the binary in front of us rather than a reading of a document.
     */
    bool BaseEntriesUsable()
    {
        const void* const* base = reinterpret_cast<const void* const*>(gxdev::kBaseVTable);
        for (unsigned slot : kBaseUsed)
        {
            const uintptr_t entry = reinterpret_cast<uintptr_t>(base[slot]);
            if (entry == gxdev::kPureCall || entry < d3d::kTextBegin || entry >= d3d::kTextEnd)
            {
                wxl::ScriptMgr::Log(WXL_LOG_ERROR,
                                    "native: the base vtable's slot %u (%s) holds 0x%08X, which is "
                                    "not a function the native device can forward to; declining",
                                    slot, kSlotNames[slot], static_cast<unsigned>(entry));
                return false;
            }
        }
        return true;
    }
}

namespace wxlvk::native
{
    bool Install(void** table, const void* const* stock)
    {
        if (!BaseEntriesUsable()) return false;

        Put<slots::kDeviceCreate>(table, &impl::DeviceCreate);
        Put<slots::kDeviceDestroy>(table, &impl::DeviceDestroy);
        Put<slots::kDeviceSetFormat>(table, &impl::DeviceSetFormat);
        Put<slots::kDeviceWM>(table, &impl::DeviceWM);
        Put<slots::kScenePresent>(table, &impl::ScenePresent);
        Put<slots::kSceneClear>(table, &impl::SceneClear);

        Put<slots::kITexMarkAsUpdated>(table, &impl::ITexMarkAsUpdated);
        Put<slots::kIRsSendToHw>(table, &impl::IRsSendToHw);
        Put<slots::kICursorCreate>(table, &impl::ICursorCreate);
        Put<slots::kICursorDraw>(table, &impl::ICursorDraw);
        Put<slots::kDeviceCreateOnWindow>(table, &impl::DeviceCreateOnWindow);
        Put<slots::kDeviceEvictResources>(table, &impl::DeviceEvictResources);
        Put<slots::kDeviceSetBaseMipLevel>(table, &impl::DeviceSetBaseMipLevel);
        Put<slots::kDeviceSetGammaRamp>(table, &impl::DeviceSetGammaRamp);
        Put<slots::kDeviceSetGammaLevel>(table, &impl::DeviceSetGammaLevel);
        Put<slots::kDeviceReadPixels>(table, &impl::DeviceReadPixels);
        Put<slots::kDeviceSetRenderTarget>(table, &impl::DeviceSetRenderTarget);
        Put<slots::kCaptureDepthStencilSurface>(table, &impl::CaptureDepthStencilSurface);
        Put<slots::kBindDepthStencilSurface>(table, &impl::BindDepthStencilSurface);
        Put<slots::kDeviceResolveDepthBuffer>(table, &impl::DeviceResolveDepthBuffer);
        Put<slots::kDeviceCopyTex>(table, &impl::DeviceCopyTex);
        Put<slots::kLogCrashInfo>(table, &impl::LogCrashInfo);
        Put<slots::kDraw>(table, &impl::Draw);
        Put<slots::kPoolSizeSet>(table, &impl::PoolSizeSet);
        Put<slots::kPoolDestroy>(table, &impl::PoolDestroy);
        Put<slots::kBufLock>(table, &impl::BufLock);
        Put<slots::kBufUnlock>(table, &impl::BufUnlock);
        Put<slots::kBufData>(table, &impl::BufData);
        Put<slots::kTexDestroy>(table, &impl::TexDestroy);
        Put<slots::kTexCopy>(table, &impl::TexCopy);
        Put<slots::kTexStretch>(table, &impl::TexStretch);
        Put<slots::kQueryCreate>(table, &impl::QueryCreate);
        Put<slots::kQueryDestroy>(table, &impl::QueryDestroy);
        Put<slots::kQueryBegin>(table, &impl::QueryBegin);
        Put<slots::kQueryEnd>(table, &impl::QueryEnd);
        Put<slots::kQueryGetParam>(table, &impl::QueryGetParam);
        Put<slots::kQueryGetData>(table, &impl::QueryGetData);
        Put<slots::kShaderCreate>(table, &impl::ShaderCreate);
        Put<slots::kShaderDestroy>(table, &impl::ShaderDestroy);
        Put<slots::kIShaderReload>(table, &impl::IShaderReload);
        Put<slots::kIShaderCreate>(table, &impl::IShaderCreate);
        Put<slots::kCursorSetVisible>(table, &impl::CursorSetVisible);

        // Counted from the written table rather than asserted, so a role list that names a slot
        // nobody Put<> is caught on the first run.
        unsigned mismatched = 0;
        for (unsigned i = 0; i < kSlotCount; ++i)
        {
            const bool ours = table[i] != stock[i];
            if (ours != (RoleOf(i) != Role::Stock)) ++mismatched;
        }
        if (mismatched)
            wxl::ScriptMgr::Log(WXL_LOG_ERROR,
                                "native: %u slot(s) of the table disagree with the role list", mismatched);
        return true;
    }

    unsigned CountOf(Role role)
    {
        unsigned n = 0;
        for (unsigned i = 0; i < kSlotCount; ++i)
            if (RoleOf(i) == role) ++n;
        return n;
    }
}
