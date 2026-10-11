// Retirement by frame serial, and the staging and upload rings.
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

#include "native/present/Internal.hpp"

namespace wxlvk::native::present::detail
{
    // --- retirement and the staging ring ---------------------------------------------------------------

    void DestroyRetired(Retired& e)
    {
        const loader::Device& d = *s.dapi;
        switch (e.kind)
        {
            case Kind::Image:
                texture::DestroyIdle(e.img);
                ++s.stats.retiredImages;
                break;
            case Kind::Buffer:
                memory::DestroyBuffer(e.buf);
                ++s.stats.retiredBuffers;
                break;
            case Kind::Framebuffer:
                if (e.fb != VK_NULL_HANDLE) d.DestroyFramebuffer(s.device, e.fb, nullptr);
                break;
            case Kind::View:
                if (e.view != VK_NULL_HANDLE) d.DestroyImageView(s.device, e.view, nullptr);
                break;
            case Kind::Raw:
                if (e.view != VK_NULL_HANDLE) d.DestroyImageView(s.device, e.view, nullptr);
                if (e.raw != VK_NULL_HANDLE) d.DestroyImage(s.device, e.raw, nullptr);
                memory::Free(e.rawBlock);
                break;
        }
    }

    void CollectRetired()
    {
        uint32_t keep = 0;
        for (uint32_t i = 0; i < s.retiredCount; ++i)
        {
            Retired& e = s.retired[i];
            if (e.serial <= s.completed)
            {
                DestroyRetired(e);
                continue;
            }
            if (keep != i) s.retired[keep] = e;
            ++keep;
        }
        for (uint32_t i = keep; i < s.retiredCount; ++i) s.retired[i] = Retired();
        s.retiredCount = keep;
    }

    /// The common half of every retirement: destroyed at once when its frame is known complete.
    void Retire(Retired& e)
    {
        // The open frame's serial is never completed, so a resource it named waits for it. The
        // fences are polled only when the serial was submitted -- for the open frame's (a zone
        // change retires thousands of objects inside one frame) no poll can help.
        if (e.serial > s.completed && e.serial <= s.submitted) Poll();
        if (e.serial <= s.completed)
        {
            DestroyRetired(e);
            ++s.stats.retiredAtOnce;
            return;
        }
        if (!RetireRoom())
        {
            ++s.stats.retireLeaked;
            return;
        }
        s.retired[s.retiredCount++] = e;
        if (s.retiredCount > s.stats.retirePeak) s.stats.retirePeak = s.retiredCount;
    }

    /// The upload ring's half of ResetStaging (same rule, its own keep count).
    void ResetUploads(FrameSlot& slot)
    {
        uint32_t keep = 0;
        for (uint32_t i = 0; i < slot.upCount; ++i)
        {
            Chunk& c = slot.up[i];
            if (i >= kKeepUpChunks && c.used == 0)
            {
                memory::DestroyBuffer(c.buffer);
                --s.stats.uploadChunks;
                continue;
            }
            c.used = 0;
            if (keep != i)
            {
                slot.up[keep] = c;
                c = Chunk();
            }
            ++keep;
        }
        slot.upCount  = keep;
        slot.upCursor = 0;
    }

    /// The CPU's writes into both rings made visible before a submit -- a no-op on a coherent
    /// type (every desktop adapter offers one), the flush the non-coherent fallback needs.
    void FlushRings(FrameSlot& slot)
    {
        for (uint32_t i = 0; i < slot.chunkCount; ++i)
            if (slot.chunks[i].used) memory::FlushHost(slot.chunks[i].buffer.block);
        for (uint32_t i = 0; i < slot.upCount; ++i)
            if (slot.up[i].used) memory::FlushHost(slot.up[i].buffer.block);
    }

    void ResetStaging(FrameSlot& slot)
    {
        ResetUploads(slot);
        // The slot's fence was just waited: nothing reads any of its chunks any more.
        uint32_t keep = 0;
        for (uint32_t i = 0; i < slot.chunkCount; ++i)
        {
            Chunk& c = slot.chunks[i];
            if (i >= kKeepChunks && c.used == 0)
            {
                memory::DestroyBuffer(c.buffer);
                --s.stats.stagingChunks;
                continue;
            }
            c.used = 0;
            if (keep != i)
            {
                slot.chunks[keep] = c;
                c = Chunk();
            }
            ++keep;
        }
        slot.chunkCount  = keep;
        slot.chunkCursor = 0;
    }
}
