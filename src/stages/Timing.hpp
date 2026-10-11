// GPU time of each point's stage run and of each stage, from timestamp queries.
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

#include "vulkan/Vk.hpp"

#include <cstdint>

// A LANE is one measured span: the stages' run at a point, or one stage. Each frame slot has its
// own query pool; a slot's results are read, without waiting, the next time that slot records a
// run -- its fence has been waited by then, so they are there (a result that is not is skipped,
// never waited for). Without timestamps on the graphics queue (timestampValidBits 0) nothing is
// recorded and every lane reads as unknown.
namespace wxlvk::stages::timing
{
    /// Lanes there are: the points first, then the stages in registration order.
    constexpr uint32_t kLanes = 20;

    /// One lane's figures.
    struct Lane
    {
        double   lastMs  = 0.0;  ///< the latest frame read
        double   sumMs   = 0.0;  ///< since the last Reset
        double   maxMs   = 0.0;
        uint64_t samples = 0;
    };

    /**
     * The start of a frame's first run in `slot`: reads that slot's previous results and resets its
     * pool. A second call for the same serial does nothing.
     *
     * @return bool timed : false when this device has no timestamps (Start / Stop then do nothing)
     */
    bool Open(VkCommandBuffer cmd, uint32_t slot, uint64_t serial);

    /// The two ends of a lane's span in the run being recorded.
    void Start(VkCommandBuffer cmd, uint32_t lane);
    void Stop(VkCommandBuffer cmd, uint32_t lane);

    /// A lane's figures.
    const Lane& Get(uint32_t lane);

    /// Starts every lane's window over (the stats line's sum and max).
    void Reset();

    /// Whether this device records timestamps.
    bool Available();
}
