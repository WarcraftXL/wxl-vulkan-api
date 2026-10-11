// What the bring-up costs in a 32-bit process's address space, measured rather than estimated.
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

#include <windows.h>

#include <cstdint>

// Why this file exists at all, and why it measures what it measures.
//
// Wow.exe 12340 is a 32-bit process. The number that kills it is not how much memory it uses but how
// much contiguous address space is left when the next large allocation arrives, and a Vulkan ICD is
// a large, eager consumer of exactly that -- the user's own WarcraftXL.cfg records an unused Vulkan
// driver costing about 230 MB.
//
// How much there IS to spend was measured rather than assumed, because the figure usually quoted for
// this client is wrong: `dumpbin /headers Wow.exe` reports PE characteristics 0x123, and 0x0020 of
// that is IMAGE_FILE_LARGE_ADDRESS_AWARE. So on a 64-bit host this process gets up to 4 GB of user
// address space, not the 2 GB a 32-bit binary is normally assumed to have. That makes the ICD's cost
// less alarming than it looks -- and it is exactly why Take() reads its upper bound from
// GetSystemInfo instead of stopping at 0x7FFFFFFF, since a walk that stopped there would silently
// under-report every figure below on the machine this actually runs on.
//
// The bring-up reports three figures and not one:
//
//  - committed bytes: pages the process has actually asked the OS to back. This is the closest thing
//    to "how much did it cost", and it is the figure to compare against the ~230 MB note.
//  - reserved bytes: address space taken, committed or not. An ICD that reserves a large arena and
//    commits a fraction of it costs the full reservation here, and that reservation is what a later
//    VirtualAlloc collides with. On a 32-bit process this matters MORE than the committed figure.
//  - largest free block: the single number that predicts the next large-allocation failure. A
//    process can have 600 MB free and still fail a 64 MB request.
//
// All three come from a VirtualQuery walk, which is deliberate: GetProcessMemoryInfo would need
// psapi on the link line and reports neither the reservation nor the fragmentation. VirtualQuery
// needs nothing, cannot fail in a way that matters, and is the same source Task Manager's committed
// figure is derived from.
//
// Cost of a walk: one VirtualQuery per region. A client in-world has on the order of a few thousand
// regions, so a walk is well under a millisecond and happens four times at bring-up and never again.
namespace wxlvk::vk::footprint
{
    /// One walk's result. All sizes in bytes.
    struct Snapshot
    {
        uint64_t reserved    = 0; ///< MEM_RESERVE + MEM_COMMIT region bytes
        uint64_t committed   = 0; ///< MEM_COMMIT region bytes
        uint64_t largestFree = 0; ///< biggest single MEM_FREE run
        uint32_t regions     = 0; ///< regions the walk saw, as a sanity check on the walk itself
    };

    /**
     * Walks the process's own user-mode address space.
     *
     * The upper bound comes from GetSystemInfo rather than from a hard-coded 0x7FFFFFFF: this client
     * is /LARGEADDRESSAWARE (see the file comment), so lpMaximumApplicationAddress is the only source
     * that is right on both a 64-bit host and a 32-bit one. Whatever the bound is, the walk reports
     * it in `reserved` consistently, so a delta between two snapshots is correct either way.
     *
     * @return Snapshot snapshot
     */
    inline Snapshot Take()
    {
        SYSTEM_INFO info = {};
        ::GetSystemInfo(&info);

        Snapshot snapshot;
        auto address = reinterpret_cast<uintptr_t>(info.lpMinimumApplicationAddress);
        const auto limit = reinterpret_cast<uintptr_t>(info.lpMaximumApplicationAddress);

        while (address < limit)
        {
            MEMORY_BASIC_INFORMATION region = {};
            if (::VirtualQuery(reinterpret_cast<LPCVOID>(address), &region, sizeof region) != sizeof region)
                break;

            // RegionSize of 0 would make this loop forever; it should not happen, but a diagnostic
            // helper must not be able to hang the render thread.
            if (region.RegionSize == 0) break;

            ++snapshot.regions;
            if (region.State == MEM_FREE)
            {
                if (region.RegionSize > snapshot.largestFree) snapshot.largestFree = region.RegionSize;
            }
            else
            {
                snapshot.reserved += region.RegionSize;
                if (region.State == MEM_COMMIT) snapshot.committed += region.RegionSize;
            }

            address += region.RegionSize;
        }
        return snapshot;
    }

    /// Bytes as whole MiB, for a log line that does not need the digits.
    inline unsigned Mib(uint64_t bytes) { return static_cast<unsigned>(bytes / (1024u * 1024u)); }

    /// A signed MiB delta, so a teardown that gives space back reads as a negative number.
    inline int MibDelta(uint64_t after, uint64_t before)
    {
        const int64_t delta = static_cast<int64_t>(after) - static_cast<int64_t>(before);
        return static_cast<int>(delta / (1024 * 1024));
    }
}
