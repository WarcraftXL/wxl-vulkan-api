// The native device's two ways of calling the engine: the base class's own slot entries, and a slot
// through the device's own vtable.
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

#include "device/Slots.hpp"

#include "wxl/offsets/engine/GxDevice.hpp"

#include <cstdint>

// Two helpers, and the reason there are two is the difference between "what the base class does for
// this slot" and "what this device does for this slot".
//
// Base<Slot>() is the entry the engine's own CGxDevice vtable (0x00A2DDC0) holds for a slot. Every
// D3D override that has a base half calls it first ("base 0x00682CB0", "base
// 0x006842A0", ...), and so does every native slot and stub that keeps the engine's bookkeeping.
// Reading it out of the base table rather than restating twelve addresses keeps one source for
// them -- the table the core already names as kBaseVTable -- and Slots.hpp's SlotTraits types the
// call, pop size included. native::BaseEntriesUsable() checks, before the first native table is
// installed, that every slot we take a base entry of really has one (24 are __purecall).
//
// Own<Slot>(device) goes through the device's OWN table, i.e. ours. Used wherever the engine
// itself calls a slot through `this` (D3D's DeviceCreate calls slot 11 on failure, its
// ICreateD3dDevice calls 15, 2 and 68, its ScenePresent calls 4): the call reaches whatever the
// table holds for the slot, exactly as the engine's own call would.
namespace wxlvk::native
{
    namespace gxdev = wxl::offsets::engine::gxdevice;

    /**
     * The base class's entry for one slot.
     *
     * @return Fn entry : typed by the slot's signature
     */
    template <unsigned Slot>
    inline typename device::SlotTraits<Slot>::Fn Base()
    {
        static_assert(Slot < device::kSlotCount, "slot index outside the engine's vtable");
        const void* const* table = reinterpret_cast<const void* const*>(gxdev::kBaseVTable);
        return reinterpret_cast<typename device::SlotTraits<Slot>::Fn>(table[Slot]);
    }

    /**
     * A slot of the device's own vtable, as the engine would reach it through `this`.
     *
     * @param void* device : the device
     * @return Fn entry
     */
    template <unsigned Slot>
    inline typename device::SlotTraits<Slot>::Fn Own(void* device)
    {
        static_assert(Slot < device::kSlotCount, "slot index outside the engine's vtable");
        void* const* table = *static_cast<void* const* const*>(device);
        return reinterpret_cast<typename device::SlotTraits<Slot>::Fn>(table[Slot]);
    }

    /// One engine function by address, typed at the call site.
    template <class Fn>
    inline Fn Engine(uintptr_t address)
    {
        return reinterpret_cast<Fn>(address);
    }
}
