// The native device's CPU-time measurements: time inside the hot slots and the stalls, per
// frame, and the cause of each slow frame.
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

// QueryPerformanceCounter around slot 42 (Draw), 54 (BufLock) and 55 (BufUnlock), and around the
// work that can stall a frame: texture uploads (the fill protocol run by slot 0, including any
// ring flush it needs), pipeline builds (src/vulkan/Pipelines.cpp's counter), flushes (a mid-frame
// submit-and-wait: read-back, query, an exhausted upload ring, a retired serial waited for) and the
// frame slot's fence wait at the frame's open (the GPU two frames behind). Accumulated per frame;
// slot 38 closes the frame with its present-to-present interval, and an interval over 33 ms is filed under the
// largest of the four stall buckets when that bucket is at least a third of the interval, else
// under "other" (the engine's own CPU, the FPS cap, the present).
//
// The slot buckets are INCLUSIVE: an upload or a pipeline build made from a draw is inside slot
// 42's time too. Everything runs on the render thread; no locks.
//
// Nothing is measured without WXL_VK_STATS: a QueryPerformanceCounter is 20 ns on this hardware,
// and slot 42 alone would spend two per draw on a number nobody reads. Enabled() is one load of a
// plain global, resolved once at load (Init()), so the hot path tests it and takes no timestamp.
namespace wxlvk::native::perf
{
    enum Bucket : uint32_t
    {
        kDraw = 0,       ///< slot 42
        kBufLock,        ///< slot 54
        kBufUnlock,      ///< slot 55
        kUpload,         ///< texture uploads (slot 0's fill protocol), ring flushes included
        kPipelineBuild,  ///< vkCreateGraphicsPipelines and its preparation (from the pipelines' counter)
        kFlush,          ///< mid-frame submit-and-wait, a retired serial waited for
        kFenceWait,      ///< the frame slot's fence at the frame's open
        // Slot 42 split into its phases, measured only with WXL_VK_STATS (Draw.cpp); exclusive of
        // each other, together slot 42 minus its gates and counters.
        kDrawSync,       ///< IRsSync(0) and the constant flush
        kDrawPrepare,    ///< the lazy shader creates and slot 0 per bound texture (uploads included)
        kDrawBuild,      ///< draw::BuildCmd and the placement
        kDrawRecord,     ///< pipelines::Record (pipeline builds included) and the draw's notes
        kBucketCount
    };

    namespace detail
    {
        /// WXL_VK_STATS, read once by Init(): a plain global so the per-draw test is one load.
        extern bool enabled;
    }

    /// Reads WXL_VK_STATS once, at extension load, before the first slot runs.
    void Init();

    /// Whether anything is measured at all (WXL_VK_STATS).
    inline bool Enabled() { return detail::enabled; }

    /// The QueryPerformanceCounter value.
    uint64_t Now();

    /// Adds `ticks` (QPC) to the open frame's bucket.
    void Add(Bucket b, uint64_t ticks);

    /// RAII: the scope's duration into one bucket. Nothing at all while Enabled() is false.
    struct Scope
    {
        Bucket   bucket;
        uint64_t start;
        explicit Scope(Bucket b) : bucket(b), start(detail::enabled ? Now() : 0) {}
        ~Scope()
        {
            if (detail::enabled) Add(bucket, Now() - start);
        }
        Scope(const Scope&) = delete;
        Scope& operator=(const Scope&) = delete;
    };

    /**
     * Slot 38: closes the frame whose interval just ended (`intervalMs` since the previous present)
     * and files its buckets.
     *
     * @param double intervalMs : the frame's interval, as present::FrameTimes measures it
     */
    void EndFrame(double intervalMs);

    /// The stats lines: since the last call, and the session.
    void Log();

    /**
     * Slot 42's CPU per frame and its record phase, averaged over the frames since the last stats line (wxl.vk.render
     * GetFrameStats). False without WXL_VK_STATS (nothing is measured) or before a frame was filed.
     */
    bool WindowDrawMs(double& drawMs, double& recordMs);
}
