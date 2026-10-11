// The native device's vtable: a copy of the stock D3D9 table with the native slot set written over it.
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

namespace wxlvk::device
{
    /**
     * The native device's table, built once on first call and owned by this translation unit for
     * the life of the process: the stock D3D9 backend's 84 entries, copied, with the native slot set
     * (src/native/Table.cpp) written over them.
     *
     * A copy rather than a patch of the engine's own table, because the engine's table is shared by
     * every device of that api -- including one the engine builds itself after our factory declined.
     * Writing into it would reach devices we do not own; writing into a copy reaches exactly the
     * objects whose vtable pointer we set.
     *
     * @return pointer table : the 84 entries, or null when the stock table at the recorded address
     *                         did not look like a vtable or the native set refused to install
     */
    const void* const* VTable();

    /**
     * How many of the 84 slots our table no longer shares with the stock one, measured by comparing
     * the two tables entry by entry rather than counted at install time.
     *
     * @return unsigned count : 0 when the table was never built
     */
    unsigned OverriddenSlots();
}
