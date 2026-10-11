// The native device's buffers: slots 52 PoolSizeSet, 53 PoolDestroy, 54 BufLock, 55 BufUnlock,
// 56 BufData, and the pool half of the draw's resolver.
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

#include "native/Slots.hpp"

#include "native/Engine.hpp"
#include "native/Perf.hpp"
#include "native/Present.hpp"
#include "vulkan/Context.hpp"
#include "vulkan/Memory.hpp"
#include "vulkan/Vk.hpp"

#include "wxl/Script.hpp"
#include "wxl/game/Binding.hpp"
#include "wxl/offsets/engine/GxDevice.hpp"

#include <cstdint>
#include <cstring>
#include <new>

// A CGxPool becomes VkBuffers owned by a record behind pool+0x14 (only slots 52-56 read that field,
// all native). Everything else -- the pool and buf objects, their lists, the stream
// cursor, the locked-buf table, the valid bytes -- stays the engine's: each slot runs its base
// entry first, exactly as the D3D overrides do (cgxdevice 3.3), and the lock reproduces D3D's
// IBufLock 0x0068FB10 step by step (read whole in the decompilation).
//
// The memory strategy, by usage (pool+0x0C):
//
//   2 stream   a RENAMED ring of chunks, each a host-visible mapped VkBuffer of pool+0x10 bytes.
//              Lock places the buf as D3D does (NOOVERWRITE while the cursor fits; at the wrap
//              CGxPool::Discard, offset 0) and writes in place; the wrap -- D3D's DISCARD, "new
//              memory, the old stays with the GPU" -- moves the pool to another chunk that no
//              unfinished frame named, or to a new one. A draw binds the chunk current at its
//              call, so a draw recorded before a wrap keeps reading the bytes it was given.
//   1 dynamic  a host-visible chunk the draws bind, behind a CPU SHADOW the lock hands out:
//              the unlock copies the written range into the chunk. NOOVERWRITE is the engine's
//              promise not to overwrite what the GPU may still read, and D3D keeps no more than
//              that promise -- but a D3D GPU runs a frame's draws while the frame is still being
//              issued, and a native frame runs them all after it, so "a byte drawn earlier in this
//              frame" is the one case where writing in place would show a draw bytes D3D never
//              showed it. The draw records the byte ranges it bound (NoteBufDrawn); a lock that
//              overlaps one moves the pool to another chunk first, a copy of the shadow (copy on
//              write), so the earlier draw keeps its bytes. Ranges drawn by an earlier frame still
//              in flight are D3D's own case and stay counted, not renamed.
//   0 static   a DEVICE-LOCAL buffer. Lock hands out a span of the frame's staging ring; Unlock
//              records a copy of it into the pool buffer under the TRANSFER RULE: into the
//              frame's pre-pass buffer (before every draw of the frame) when no draw of this frame
//              has named the pool, INLINE (the pass broken, the copy, the pass resumed) when one
//              has -- so each draw sees the bytes current at its call, which is D3D's guarantee
//              for a synchronised (flags 0) lock of a MANAGED buffer. No CPU shadow: D3D creates
//              these WRITEONLY and the engine writes the whole buf it locks.
//
// A pool gets its record at its first lock (or 52): the two stream pools exist before the device
// does, and slot 52 is never called on the login screen -- every other pool arrives sized by
// PoolCreate. With no context, or when memory cannot be had, the lock takes
// D3D's emergency path: the engine's own EmergencyMem block, and the unlock poisons the frame
// (+0xF5C), so the draws that would read the missing bytes are dropped until the next present.
namespace
{
    namespace gxdev   = wxl::offsets::engine::gxdevice;
    namespace slots   = wxlvk::device::slots;
    namespace memory  = wxlvk::vk::memory;
    namespace context = wxlvk::vk::context;
    namespace present = wxlvk::native::present;

    using wxl::game::At;
    using wxlvk::native::Base;
    using wxlvk::native::Engine;

    /// Chunks one pool may rename through. Two frames in flight need three at a steady one wrap
    /// per frame; a frame that wraps W times needs about 2 (W + 1). In-world frames wrap the
    /// 2.75 MiB vertex ring several times, so 32 -- and when even those are all still read, the
    /// rename waits for the oldest earlier frame instead of writing in place (FreeChunk).
    constexpr uint32_t kMaxChunks = 32;

    /// Byte ranges of a dynamic pool one frame's draws bound, before the list gives up and calls
    /// the whole pool drawn (every overlapping lock then renames).
    constexpr uint32_t kMaxDrawnRanges = 64;

    struct Range
    {
        uint32_t begin, end;
    };

    struct PoolRec
    {
        // The header the draw's resolver reads (usage, the current chunk and its stamp) is one
        // cache line; the chunk table follows.
        void*          pool       = nullptr;
        uint32_t       target     = 0;
        uint32_t       usage      = 0;
        uint32_t       size       = 0;
        uint32_t       chunkCount = 0;
        uint32_t       current    = 0;
        uint64_t       chunkUsed[kMaxChunks] = {};  ///< serial of the last frame a draw bound the chunk in
        memory::Buffer chunks[kMaxChunks];
        uint64_t       chunkRef[kMaxChunks]  = {};  ///< serial of the last frame any recorded command (a copy) named it

        // Dynamic pools: the shadow the lock hands out, and what this frame's draws bound.
        uint8_t*       shadow      = nullptr;   ///< `size` bytes; null when it could not be had (then written in place)
        uint64_t       drawnSerial = 0;
        uint32_t       drawnCount  = 0;
        bool           drawnAll    = false;     ///< the range list overflowed
        Range          drawn[kMaxDrawnRanges];

        ~PoolRec() { delete[] shadow; }
    };

    /// One outstanding lock per target, as the engine's locked-buf table has one (+0x2778).
    struct Lock
    {
        const void*  buf       = nullptr;
        PoolRec*     rec       = nullptr;
        bool         emergency = false;
        bool         staged    = false;   ///< static: the span below is where the engine writes
        bool         shadowed  = false;   ///< dynamic: the engine writes the shadow; the unlock copies
        VkBuffer     src       = VK_NULL_HANDLE;
        VkDeviceSize srcOffset = 0;
    };

    struct Stats
    {
        uint64_t locks[3];           ///< by usage
        uint64_t discards;           ///< stream wraps
        uint64_t renames;            ///< of which moved to another chunk
        uint64_t renameHazards;      ///< wraps with every chunk still unfinished: written in place, counted
        uint64_t chunksCreated;
        uint64_t staticPrePass;      ///< static copies recorded ahead of the frame's draws
        uint64_t staticInline;       ///< static copies recorded inline (a pass break each)
        uint64_t staticBytes;
        uint64_t dynamicThisFrame;   ///< dynamic locks of a pool a draw of this frame already bound
        uint64_t dynamicInFlight;    ///< dynamic locks of a pool an unfinished earlier frame bound
        uint64_t emergency;          ///< locks answered with EmergencyMem (no context / no memory)
        uint64_t records;            ///< pool records created
        uint64_t unresolved;         ///< draws whose pool had no buffer
        uint64_t renameWaits;        ///< stream wraps that waited for an earlier frame to free a chunk
        uint32_t mostChunks;         ///< the most chunks one pool has held
        uint64_t dynamicOverlap;     ///< dynamic locks overlapping a range a draw of this frame bound: the true hazard
        uint64_t dynamicRenames;     ///< of which renamed by copy on write
        uint64_t dynamicCowBytes;    ///< bytes copied for those
        uint64_t dynamicInPlace;     ///< of which written in place anyway (no chunk could be had)
        uint64_t shadowBytes;        ///< bytes of dynamic-pool shadows alive
        uint64_t copiesRefStamped;   ///< static copies that stamped their chunk for retirement
    };

    /// Outstanding locks per target. The engine's locked-buf table holds one per target, but
    /// BufData runs IBufLock / IBufUnlock itself, so a BufData inside an open lock of the same
    /// target (or two bufs locked at once) would overwrite a single record and lose the outer
    /// lock's staged or shadowed bytes (the dynamic shadow above all), so the records are a few
    /// deep and an unlock finds its own by buf.
    constexpr uint32_t kLockDepth = 4;
    Lock     g_locks[gxdev::kPoolTypeCount][kLockDepth];
    uint64_t g_lockOverflow;   ///< locks with every record of their target in use (the oldest is reused)

    /// The record for a new lock of `buf`: its own if it is already locked, else a free one.
    Lock& LockFor(uint32_t target, const void* buf)
    {
        Lock* row = g_locks[target < gxdev::kPoolTypeCount ? target : 0u];
        for (uint32_t i = 0; i < kLockDepth; ++i)
            if (row[i].buf == buf) return row[i];
        for (uint32_t i = 0; i < kLockDepth; ++i)
            if (!row[i].buf) return row[i];
        ++g_lockOverflow;
        return row[0];
    }

    /// The record of an outstanding lock of `buf`, or null.
    Lock* LockOf(uint32_t target, const void* buf)
    {
        Lock* row = g_locks[target < gxdev::kPoolTypeCount ? target : 0u];
        for (uint32_t i = 0; i < kLockDepth; ++i)
            if (row[i].buf == buf) return &row[i];
        return nullptr;
    }
    Stats g_stats;
    bool  g_loggedFail = false;

    PoolRec*& RecOf(void* pool) { return At<PoolRec*>(pool, gxdev::kPoolApiHandle); }

    bool CreateChunk(PoolRec& r)
    {
        if (r.chunkCount == kMaxChunks || r.size == 0) return false;
        const VkBufferUsageFlags use = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT |
                                       // TRANSFER_SRC: src/vulkan/Wind.cpp reads a tree model's
                                       // range back once.
                                       (r.usage == gxdev::kPoolUsageStatic
                                            ? VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT
                                            : VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
        // The stream and dynamic chunks prefer device-local host-visible memory (the BAR window
        // or ReBAR, memory::Where::Stream; plain host-visible when there is none or it is half
        // used), so the vertex fetch reads VRAM. The CPU only writes them (the dynamic pools write
        // through their shadow).
        const memory::Where where = r.usage == gxdev::kPoolUsageStatic ? memory::Where::DeviceLocal
                                                                       : memory::Where::Stream;
        memory::Buffer& b = r.chunks[r.chunkCount];
        if (!memory::CreateBuffer(r.size, use, where, b)
            || (where != memory::Where::DeviceLocal && !b.block.mapped))
        {
            memory::DestroyBuffer(b);
            if (!g_loggedFail)
            {
                g_loggedFail = true;
                wxl::ScriptMgr::Log(WXL_LOG_ERROR, "native: pools: no %s buffer of %u bytes for pool "
                                    "\"%s\"; its locks take the emergency path (logged once)",
                                    where != memory::Where::DeviceLocal ? "host-visible" : "device-local",
                                    r.size, At<const char*>(r.pool, gxdev::kPoolName)
                                                ? At<const char*>(r.pool, gxdev::kPoolName) : "?");
            }
            return false;
        }
        r.chunkUsed[r.chunkCount] = 0;
        r.chunkRef[r.chunkCount]  = 0;
        ++r.chunkCount;
        ++g_stats.chunksCreated;
        if (r.chunkCount > g_stats.mostChunks) g_stats.mostChunks = r.chunkCount;
        return true;
    }

    /// The last frame anything recorded named a chunk: a draw, or a copy into it.
    uint64_t LastRef(const PoolRec& r, uint32_t i)
    {
        return r.chunkUsed[i] > r.chunkRef[i] ? r.chunkUsed[i] : r.chunkRef[i];
    }

    void RetireAll(PoolRec& r)
    {
        for (uint32_t i = 0; i < r.chunkCount; ++i) present::RetireBuffer(r.chunks[i], LastRef(r, i));
        r.chunkCount = 0;
        r.current    = 0;
        r.drawnSerial = 0;
    }

    /// The dynamic pool's shadow, sized to the pool; dropped and remade when the size changes.
    void EnsureShadow(PoolRec& r)
    {
        if (r.usage != gxdev::kPoolUsageDynamic || r.size == 0) return;
        if (r.shadow) return;
        r.shadow = new (std::nothrow) uint8_t[r.size];
        if (r.shadow)
        {
            std::memset(r.shadow, 0, r.size);
            g_stats.shadowBytes += r.size;
        }
    }

    void DropShadow(PoolRec& r)
    {
        if (!r.shadow) return;
        g_stats.shadowBytes -= r.size;
        delete[] r.shadow;
        r.shadow = nullptr;
    }

    /// The pool's record, created and sized on first use. Null without a usable device.
    PoolRec* Ensure(void* device, void* pool)
    {
        if (!pool) return nullptr;
        PoolRec* r = RecOf(pool);
        if (r && r->chunkCount) return r;
        if (!At<uint32_t>(device, gxdev::kContext) || !present::DrawPathUp()) return nullptr;
        if (!r)
        {
            r = new (std::nothrow) PoolRec();
            if (!r) return nullptr;
            r->pool   = pool;
            r->target = At<uint32_t>(pool, gxdev::kPoolType);
            r->usage  = At<uint32_t>(pool, gxdev::kPoolUsage);
            if (r->usage > gxdev::kPoolUsageStream) r->usage = gxdev::kPoolUsageStatic;
            RecOf(pool) = r;
            ++g_stats.records;
        }
        const uint32_t size = At<uint32_t>(pool, gxdev::kPoolSize);
        if (size != r->size) DropShadow(*r);
        r->size = size;
        if (!CreateChunk(*r)) return nullptr;
        EnsureShadow(*r);
        return r;
    }

    /// The emergency answer of D3D's IBufLock: the engine's own scratch block for the pool type.
    char* EmergencyLock(void* device, void* edx, const void* buf, uint32_t target, Lock& l)
    {
        ++g_stats.emergency;
        l.emergency = true;
        void* const emergency = static_cast<uint8_t*>(device) + gxdev::kEmergencyLockBlock
                                + (target < gxdev::kPoolTypeCount ? target : 0u) * gxdev::kEmergencyLockStride;
        return static_cast<char*>(Engine<gxdev::EmergencyMemLockFn>(gxdev::kEmergencyMemLock)(
            emergency, edx, At<uint32_t>(buf, gxdev::kBufBytes)));
    }

    /// A chunk other than the current one that no unfinished frame named, or a new one; failing
    /// both, the chunk the oldest submitted frame named, once that frame is waited for (never the
    /// open frame's, which cannot be waited for). kMaxChunks when none can be had.
    uint32_t FreeChunk(PoolRec& r)
    {
        const uint64_t done = present::Completed();
        for (uint32_t i = 0; i < r.chunkCount; ++i)
            if (i != r.current && LastRef(r, i) <= done) return i;
        if (CreateChunk(r)) return r.chunkCount - 1;
        uint32_t oldest = kMaxChunks;
        for (uint32_t i = 0; i < r.chunkCount; ++i)
            if (i != r.current && !present::UsedThisFrame(LastRef(r, i))
                && (oldest == kMaxChunks || LastRef(r, i) < LastRef(r, oldest)))
                oldest = i;
        if (oldest != kMaxChunks && present::WaitSerial(LastRef(r, oldest)))
        {
            ++g_stats.renameWaits;
            return oldest;
        }
        return kMaxChunks;
    }

    /// The stream wrap's rename: a chunk no unfinished frame named, or a new one (FreeChunk).
    void Rename(PoolRec& r)
    {
        if (LastRef(r, r.current) <= present::Completed()) return;   // nothing in flight reads it: keep writing here
        const uint32_t next = FreeChunk(r);
        if (next == kMaxChunks)
        {
            ++g_stats.renameHazards;   // every chunk is read by this very frame: written in place
            return;
        }
        r.current = next;
        ++g_stats.renames;
    }

    /// Whether [begin, end) of a dynamic pool overlaps what a draw of this frame bound.
    bool DrawnThisFrame(const PoolRec& r, uint32_t begin, uint32_t end)
    {
        if (r.drawnSerial == 0 || !present::UsedThisFrame(r.drawnSerial)) return false;
        if (r.drawnAll) return true;
        for (uint32_t i = 0; i < r.drawnCount; ++i)
            if (begin < r.drawn[i].end && r.drawn[i].begin < end) return true;
        return false;
    }

    /**
     * The copy on write of a dynamic pool whose lock overlaps bytes a draw of this frame
     * bound. The pool moves to a chunk no unfinished frame reads, filled from the shadow (which
     * holds exactly what the chunk holds), so the earlier draw keeps its chunk and the bytes it
     * was given, and every later draw binds the new one -- D3D's "a draw sees the bytes as at its
     * call". The new chunk carries no drawn range yet.
     */
    void CopyOnWrite(PoolRec& r)
    {
        ++g_stats.dynamicOverlap;
        const uint32_t next = r.shadow ? FreeChunk(r) : kMaxChunks;
        if (next == kMaxChunks || !r.chunks[next].block.mapped)
        {
            ++g_stats.dynamicInPlace;
            return;
        }
        std::memcpy(r.chunks[next].block.mapped, r.shadow, r.size);
        memory::FlushHost(r.chunks[next].block);
        r.current     = next;
        r.drawnSerial = 0;
        r.drawnCount  = 0;
        r.drawnAll    = false;
        ++g_stats.dynamicRenames;
        g_stats.dynamicCowBytes += r.size;
    }

    /**
     * D3D's IBufLock, step by step, with native memory behind it.
     *
     * @return char* where the engine writes buf.size bytes
     */
    char* ILock(void* device, void* edx, void* buf)
    {
        void* const    pool   = At<void*>(buf, gxdev::kBufPool);
        const uint32_t target = pool ? At<uint32_t>(pool, gxdev::kPoolType) : 0u;
        Lock& l = LockFor(target, buf);
        l = Lock();
        l.buf = buf;

        // 1. No context: the emergency block.
        if (!At<uint32_t>(device, gxdev::kContext) || !pool) return EmergencyLock(device, edx, buf, target, l);

        // 2. The placement, which D3D does before it even has a buffer.
        const uint32_t usage = At<uint32_t>(pool, gxdev::kPoolUsage);
        const uint32_t bytes = At<uint32_t>(buf, gxdev::kBufBytes);
        bool wrapped = false;
        if (usage == gxdev::kPoolUsageStream)
        {
            uint32_t stride = At<uint32_t>(buf, gxdev::kBufItemSize);
            if (stride == 0) stride = 1;   // D3D would divide by zero; no engine buf has stride 0
            const uint32_t cursor  = At<uint32_t>(pool, gxdev::kPoolStreamCursor);
            const uint32_t aligned = (stride - 1 + cursor) - (stride - 1 + cursor) % stride;
            if (At<uint32_t>(pool, gxdev::kPoolSize) < bytes + aligned)
            {
                Engine<gxdev::PoolFn>(gxdev::kPoolDiscard)(pool);
                At<uint32_t>(buf, gxdev::kBufOffset)          = 0;
                At<uint32_t>(pool, gxdev::kPoolStreamCursor) = bytes;
                wrapped = true;
                ++g_stats.discards;
            }
            else
            {
                At<uint32_t>(buf, gxdev::kBufOffset)          = aligned;
                At<uint32_t>(pool, gxdev::kPoolStreamCursor) = aligned + bytes;
            }
        }
        if (usage <= gxdev::kPoolUsageStream) ++g_stats.locks[usage];

        // 3. The API buffer on first use (D3D's CreatePoolAPI).
        PoolRec* r = Ensure(device, pool);
        if (!r) return EmergencyLock(device, edx, buf, target, l);
        l.rec = r;
        if (wrapped) Rename(*r);

        // 4. The pointer.
        const uint32_t offset = At<uint32_t>(buf, gxdev::kBufOffset);
        if (static_cast<uint64_t>(offset) + bytes > r->size) return EmergencyLock(device, edx, buf, target, l);

        if (r->usage == gxdev::kPoolUsageStatic)
        {
            void* span = nullptr;
            if (!present::Stage(bytes ? bytes : 1u, l.src, l.srcOffset, span))
                return EmergencyLock(device, edx, buf, target, l);
            l.staged = true;
            return static_cast<char*>(span);
        }
        if (r->usage == gxdev::kPoolUsageDynamic)
        {
            const uint64_t used = r->chunkUsed[r->current];
            if (present::UsedThisFrame(used)) ++g_stats.dynamicThisFrame;
            else if (used > present::Completed()) ++g_stats.dynamicInFlight;
            if (DrawnThisFrame(*r, offset, offset + bytes)) CopyOnWrite(*r);
            if (r->shadow)
            {
                l.shadowed = true;
                return reinterpret_cast<char*>(r->shadow) + offset;
            }
        }
        return static_cast<char*>(r->chunks[r->current].block.mapped) + offset;
    }

    /**
     * The device-side half of D3D's IBufUnlock: for a static pool, the copy of [begin, begin +
     * length) of the staged span to buf.offset + begin under the transfer rule.
     */
    void RecordStaticCopy(PoolRec& r, uint32_t dstOffset, VkBuffer src, VkDeviceSize srcOffset,
                          uint32_t length)
    {
        if (length == 0) return;
        VkBufferCopy region = {};
        region.srcOffset = srcOffset;
        region.dstOffset = dstOffset;
        region.size      = length;

        const context::Handles* h = wxlvk::vk::context::Get();
        if (!h) return;
        const auto& d = *h->dapi;
        // The copy names the buffer in the frame being recorded, so a PoolSizeSet or a
        // PoolDestroy later in this frame (a zone load does both) must hold it until the frame
        // completes -- the draw stamp alone would let it go with an older serial.
        struct Stamp
        {
            PoolRec& r;
            ~Stamp() { if (present::FrameOpen()) { r.chunkRef[0] = present::Serial(); ++g_stats.copiesRefStamped; } }
        } stamp{ r };
        if (present::UsedThisFrame(r.chunkUsed[0]))
        {
            // A draw recorded earlier in this frame reads this pool: the copy must land after it.
            VkCommandBuffer cmd = present::Outside();
            if (!cmd) return;
            VkMemoryBarrier before = {};
            before.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;   // write-after-read: execution only
            d.CmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_VERTEX_INPUT_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
                                 1, &before, 0, nullptr, 0, nullptr);
            d.CmdCopyBuffer(cmd, src, r.chunks[0].handle, 1, &region);
            VkMemoryBarrier after = {};
            after.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
            after.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            after.dstAccessMask = VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT | VK_ACCESS_INDEX_READ_BIT;
            d.CmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_VERTEX_INPUT_BIT, 0,
                                 1, &after, 0, nullptr, 0, nullptr);
            ++g_stats.staticInline;
        }
        else
        {
            // No draw of this frame has named the pool: ahead of all of them. The pre-pass buffer's
            // own barriers order it after the previous frame's reads and before this frame's.
            VkCommandBuffer cmd = present::PrePass();
            if (!cmd) return;
            d.CmdCopyBuffer(cmd, src, r.chunks[0].handle, 1, &region);
            ++g_stats.staticPrePass;
        }
        g_stats.staticBytes += length;
    }

    /// D3D's IBufUnlock: the emergency case poisons the frame; otherwise "the unlock succeeded".
    void IUnlock(void* device, void* buf, uint32_t begin, uint32_t length)
    {
        void* const    pool   = At<void*>(buf, gxdev::kBufPool);
        const uint32_t target = pool ? At<uint32_t>(pool, gxdev::kPoolType) : 0u;
        Lock* found = LockOf(target, buf);
        if (!found || found->emergency || !found->rec)
        {
            // Including a buf we never saw locked: there is nothing behind it to make visible.
            At<uint8_t>(buf, gxdev::kBufValid)        = 0;
            At<uint32_t>(device, gxdev::kFramePoisoned) = 1;
            if (found) *found = Lock();
            return;
        }
        Lock& l = *found;
        PoolRec& r = *l.rec;
        if (l.staged)
            RecordStaticCopy(r, At<uint32_t>(buf, gxdev::kBufOffset) + begin, l.src, l.srcOffset + begin, length);
        else
        {
            if (l.shadowed && r.shadow && r.chunks[r.current].block.mapped)
            {
                // The written range, from the shadow into the chunk the draws will bind.
                const uint64_t at = static_cast<uint64_t>(At<uint32_t>(buf, gxdev::kBufOffset)) + begin;
                if (at + length <= r.size)
                    std::memcpy(static_cast<uint8_t*>(r.chunks[r.current].block.mapped) + at, r.shadow + at, length);
            }
            memory::FlushHost(r.chunks[r.current].block);   // a no-op on the coherent type
        }
        At<uint8_t>(buf, gxdev::kBufValid) = 1;
        l = Lock();
    }
}

namespace wxlvk::native::impl
{
    bool g_dynamicPoolResolved = false;

    void __fastcall PoolSizeSet(void* device, void* /*edx*/, CGxPool* pool, uint32_t size)
    {
        // D3D 0x0069FF80: IReleaseD3dPool (invalidate the bufs, release, cursor 0), the size, create.
        if (!pool) return;
        Engine<gxdev::PoolFn>(gxdev::kPoolInvalidate)(pool);
        if (PoolRec* r = RecOf(pool))
        {
            RetireAll(*r);
            DropShadow(*r);   // remade at the new size by Ensure
        }
        At<uint32_t>(pool, gxdev::kPoolStreamCursor) = 0;
        At<uint32_t>(pool, gxdev::kPoolSize)         = size;
        Ensure(device, pool);
    }

    void __fastcall PoolDestroy(void* device, void* edx, CGxPool* pool)
    {
        // D3D 0x0068E720: release, then the base unlinks and frees. Released by serial: the frame
        // that last bound a chunk may still be on the GPU, and destroying it now loses the device.
        if (pool)
            if (PoolRec* r = RecOf(pool))
            {
                RetireAll(*r);
                DropShadow(*r);
                delete r;
                RecOf(pool) = nullptr;
            }
        Base<slots::kPoolDestroy>()(device, edx, pool);
    }

    void NoteBufDrawn(const void* buf)
    {
        // The byte range of a dynamic pool a draw just bound (the header above, "dynamic").
        if (!buf) return;
        const void* const pool = At<const void* const>(buf, gxdev::kBufPool);
        PoolRec* r = pool ? At<PoolRec*>(pool, gxdev::kPoolApiHandle) : nullptr;
        if (!r || r->usage != gxdev::kPoolUsageDynamic) return;
        const uint64_t serial = present::Serial();
        if (r->drawnSerial != serial)
        {
            r->drawnSerial = serial;
            r->drawnCount  = 0;
            r->drawnAll    = false;
        }
        if (r->drawnAll) return;
        const uint32_t begin = At<const uint32_t>(buf, gxdev::kBufOffset);
        const uint32_t end   = begin + At<const uint32_t>(buf, gxdev::kBufBytes);
        for (uint32_t i = 0; i < r->drawnCount; ++i)
            if (begin <= r->drawn[i].end && r->drawn[i].begin <= end)
            {
                // Touching or overlapping: one range, so a run of neighbouring bufs stays one entry.
                if (begin < r->drawn[i].begin) r->drawn[i].begin = begin;
                if (end > r->drawn[i].end) r->drawn[i].end = end;
                return;
            }
        if (r->drawnCount == kMaxDrawnRanges)
        {
            r->drawnAll = true;
            return;
        }
        r->drawn[r->drawnCount++] = Range{ begin, end };
    }

    char* __fastcall BufLock(void* device, void* edx, CGxBuf* buf)
    {
        // D3D 0x0068FCE0: base (the dirty bytes, the locked-buf table), then IBufLock.
        wxlvk::native::perf::Scope timed(wxlvk::native::perf::kBufLock);
        Base<slots::kBufLock>()(device, edx, buf);
        if (!buf) return nullptr;
        return ILock(device, edx, buf);
    }

    int __fastcall BufUnlock(void* device, void* edx, CGxBuf* buf, uint32_t bytesWritten)
    {
        // D3D 0x0068FAE0: base, IBufUnlock, return 1. bytesWritten is meaningless on D3D (the
        // engine passes whatever); the whole buf is what the lock handed out.
        wxlvk::native::perf::Scope timed(wxlvk::native::perf::kBufUnlock);
        Base<slots::kBufUnlock>()(device, edx, buf, bytesWritten);
        if (buf) IUnlock(device, buf, 0, At<uint32_t>(buf, gxdev::kBufBytes));
        return 1;
    }

    void __fastcall BufData(void* device, void* edx, CGxBuf* buf, const void* data, uint32_t size,
                            uint32_t offset)
    {
        // D3D 0x0068FD00: base, p = IBufLock(buf), memcpy(p + offset, data, size), IBufUnlock --
        // the non-virtual pair, so slots 54/55 are not called. Only [offset, offset + size) is
        // copied for a static pool: the rest of the staged span was never written.
        Base<slots::kBufData>()(device, edx, buf, data, size, offset);
        if (!buf) return;
        char* p = ILock(device, edx, buf);
        if (p && data && size) std::memcpy(p + offset, data, size);
        IUnlock(device, buf, offset, size);
    }

    bool ResolvePool(const void* pool, VkBuffer& out)
    {
        PoolRec* r = pool ? At<PoolRec*>(pool, gxdev::kPoolApiHandle) : nullptr;
        if (!r || r->chunkCount == 0)
        {
            ++g_stats.unresolved;
            return false;
        }
        // The stamp the transfer rule and the stream rename read: this frame binds this chunk.
        r->chunkUsed[r->current] = present::Serial();
        if (r->usage == gxdev::kPoolUsageDynamic) g_dynamicPoolResolved = true;   // slot 42 notes the draw's bufs
        out = r->chunks[r->current].handle;
        return out != VK_NULL_HANDLE;
    }

    void LogPoolStats()
    {
        const Stats& g = g_stats;
        wxl::ScriptMgr::Log(WXL_LOG_INFO,
                            "native: pools: %llu record(s); locks static %llu / dynamic %llu / stream "
                            "%llu; stream wraps %llu (renamed %llu, in-place hazards %llu), %llu chunk(s) "
                            "made; static copies %llu pre-pass + %llu inline (%llu KiB); dynamic locks "
                            "of a pool drawn this frame %llu, in flight %llu; emergency locks %llu; "
                            "draws with an unresolved pool %llu",
                            static_cast<unsigned long long>(g.records),
                            static_cast<unsigned long long>(g.locks[0]),
                            static_cast<unsigned long long>(g.locks[1]),
                            static_cast<unsigned long long>(g.locks[2]),
                            static_cast<unsigned long long>(g.discards),
                            static_cast<unsigned long long>(g.renames),
                            static_cast<unsigned long long>(g.renameHazards),
                            static_cast<unsigned long long>(g.chunksCreated),
                            static_cast<unsigned long long>(g.staticPrePass),
                            static_cast<unsigned long long>(g.staticInline),
                            static_cast<unsigned long long>(g.staticBytes / 1024),
                            static_cast<unsigned long long>(g.dynamicThisFrame),
                            static_cast<unsigned long long>(g.dynamicInFlight),
                            static_cast<unsigned long long>(g.emergency),
                            static_cast<unsigned long long>(g.unresolved));
        wxl::ScriptMgr::Log(WXL_LOG_INFO,
                            "native: pools: %llu rename(s) that waited for an earlier frame, at most %u "
                            "chunk(s) in one pool (of %u); dynamic locks overlapping a range drawn this "
                            "frame %llu: %llu renamed by copy on write (%llu KiB), %llu written in place; "
                            "%llu KiB of dynamic shadows; %llu static copies stamped for retirement",
                            static_cast<unsigned long long>(g.renameWaits), g.mostChunks, kMaxChunks,
                            static_cast<unsigned long long>(g.dynamicOverlap),
                            static_cast<unsigned long long>(g.dynamicRenames),
                            static_cast<unsigned long long>(g.dynamicCowBytes / 1024),
                            static_cast<unsigned long long>(g.dynamicInPlace),
                            static_cast<unsigned long long>(g.shadowBytes / 1024),
                            static_cast<unsigned long long>(g.copiesRefStamped));
        if (g_lockOverflow)
            wxl::ScriptMgr::Log(WXL_LOG_WARN, "native: pools: %llu lock(s) found every record of their "
                                "target in use (more than %u bufs locked at once)",
                                static_cast<unsigned long long>(g_lockOverflow), kLockDepth);
    }
}
