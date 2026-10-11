// Copying the stock D3D9 vtable, checking that it is one, and writing the native slot set over it.
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

#include "device/VTable.hpp"

#include "device/Slots.hpp"
#include "native/Native.hpp"
#include "offsets/GxDeviceD3d.hpp"

#include "wxl/Script.hpp"
#include "wxl/offsets/engine/Gx.hpp"

#include <cstdint>
#include <cstring>

namespace
{
    namespace d3d = wxlvk::offsets::gxdevd3d;
    namespace gx  = wxl::offsets::engine::gx;

    using wxlvk::device::kSlotCount;
    using wxlvk::device::kSlotNames;

    void* g_slot[kSlotCount];
    bool  g_built  = false;
    bool  g_usable = false;

    /// The stock D3D9 table the copy is taken from.
    const void* const* StockTable() { return reinterpret_cast<const void* const*>(gx::kGxDeviceVTable); }

    /**
     * True when every entry of a candidate table is a code address in the client image.
     *
     * Cheap and worth it: if the vtable address were wrong for the binary in front of us, the copy
     * would be 84 arbitrary dwords and installing it would hand the engine 84 jumps into data. A
     * vtable's entries are all in `.text` and nothing else about this check depends on the build.
     *
     * @param pointer table : the candidate, 84 dwords
     * @param unsigned& badSlot : receives the first entry that is not a code address
     * @return bool plausible
     */
    bool LooksLikeVTable(const void* const* table, unsigned& badSlot)
    {
        for (unsigned i = 0; i < kSlotCount; ++i)
        {
            const uintptr_t entry = reinterpret_cast<uintptr_t>(table[i]);
            if (entry < d3d::kTextBegin || entry >= d3d::kTextEnd)
            {
                badSlot = i;
                return false;
            }
        }
        return true;
    }
}

namespace wxlvk::device
{
    const void* const* VTable()
    {
        if (g_built) return g_usable ? g_slot : nullptr;
        g_built = true;

        const void* const* source = StockTable();
        unsigned badSlot = 0;
        if (!LooksLikeVTable(source, badSlot))
        {
            wxl::ScriptMgr::Log(WXL_LOG_ERROR,
                                "device: the table at 0x%08X is not a vtable in this image -- slot %u "
                                "(%s) holds 0x%08X, which is not code; refusing to copy it",
                                static_cast<unsigned>(gx::kGxDeviceVTable), badSlot, kSlotNames[badSlot],
                                static_cast<unsigned>(reinterpret_cast<uintptr_t>(source[badSlot])));
            return nullptr;
        }

        std::memcpy(g_slot, source, sizeof g_slot);
        // native::Install logs why it refuses. An unusable table makes the factory decline, which
        // is the engine building its own device -- the stock client, never a half-native one.
        if (!wxlvk::native::Install(g_slot, source)) return nullptr;
        g_usable = true;
        return g_slot;
    }

    unsigned OverriddenSlots()
    {
        if (!g_usable) return 0;
        const void* const* source = StockTable();
        unsigned count = 0;
        for (unsigned i = 0; i < kSlotCount; ++i)
            if (g_slot[i] != source[i]) ++count;
        return count;
    }
}
