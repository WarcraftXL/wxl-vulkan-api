// The EGxApi values the factory sees, and the bound a vtable copy is checked against.
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

#include <cstdint>

// Everything else about the device object and how it is built lives in the core
// (wxl/offsets/engine/GxDevice.hpp: the factories, constructors, allocation sizes, the base vtable and
// the field map; wxl/offsets/engine/Gx.hpp: the D3D9 vtable). Nothing of that is restated here -- an
// address in two places is an address that will eventually disagree with itself.
namespace wxlvk::offsets::gxdevd3d
{
    /// EGxApi values GxDevCreate dispatches on, and that the constructors write at gxdev::kApi.
    constexpr int kApiOpenGl = 0;
    constexpr int kApiD3d9   = 1;
    constexpr int kApiD3d9Ex = 2;

    // The client's `.text`, for checking that a copy of a vtable is a vtable: every one of the
    // gxdev::kVTableSlots entries has to be a code address in the image, and an entry that is not
    // means the table address is wrong for this build and the copy must not be installed.
    //
    // Read from Wow.exe's own PE section table rather than from the Ghidra memory map, which reports
    // the end rounded up to alignment (0x009DE3FF). The virtual size stops at 0x009DE3B2, and the
    // 0x4D bytes between the two are padding no function starts in -- so the tighter bound is both
    // the correct one and strictly better at catching a wrong address.
    constexpr uintptr_t kTextBegin = 0x00401000;
    constexpr uintptr_t kTextEnd   = 0x009DE3B3; // exclusive
}
