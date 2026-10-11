// The native device: what the rest of the extension needs to know about it.
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

// The native device is the engine-built CGxDeviceD3d carrying a table whose D3D-dependent slots never
// reach D3D: 39 slots are implemented here, 3 are stubs that answer a neutral value, and the 42 left
// are the stock entries that never touch the IDirect3DDevice9 (src/native/Table.cpp has the list).
namespace wxlvk::native
{
    /// What one slot of the native table is.
    enum class Role
    {
        Stock,  ///< the stock D3D9 table's entry, kept: base, or D3D code that never reaches D3D
        Native, ///< implemented here
        Stub,   ///< must not reach D3D and has nothing real to do: answers the neutral value
    };

    /**
     * Writes the native slot set over a copy of the stock table.
     *
     * Refuses -- writes nothing and returns false -- when the base vtable entries the native slots
     * forward to are not what this code was read against; the factory then declines and the engine
     * builds its own device.
     *
     * @param void** table : our copy of the stock table, kSlotCount entries
     * @param pointer stock : the stock table it was copied from, to check the result against
     * @return bool installed
     */
    bool Install(void** table, const void* const* stock);

    /// Slot counts by role, for the factory's log line.
    unsigned CountOf(Role role);

    /**
     * Slot 38, after each present: saves the pipeline cache about once a minute, and with
     * WXL_VK_STATS logs the detailed counters at the same moment.
     *
     * @param uint64 presents : presents done so far, this one included
     */
    void OnPresent(uint64_t presents);

    /// Slot 11: the one-line summary (always), the detailed counters (with WXL_VK_STATS), and a last
    /// pipeline-cache save.
    void OnDeviceDestroy();
}
