// Device memory: the memory-type choice, a sub-allocator over a few large blocks, and buffers.
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

// Why a sub-allocator, and why it is this small one.
//
// vkAllocateMemory is a scarce call, not merely an expensive one: maxMemoryAllocationCount has a
// guaranteed floor of 4096 and real desktop drivers report exactly 4096. A backend that allocated
// per resource would run out of allocations long before it ran out of memory, and the failure would
// arrive as VK_ERROR_TOO_MANY_OBJECTS in the middle of a zone load. So memory comes from a handful
// of large blocks, carved by a first-fit free list (a fresh block is one free span, so the common
// case is a bump), and only a request larger than a block gets a VkDeviceMemory of its own.
//
// VMA was the obvious alternative and was rejected for the same reason the rest of src/vulkan/ is
// hand-rolled: it wants prototypes or a function table that includes 1.1+ entry points, and the
// vulkan-1.dll on the machine is only promised to be 1.0. Nothing here needs more than 1.0 core.
//
// The other constraint is the one peculiar to this process: it is 32-bit, and address space -- not
// VRAM -- is what runs out. A host-visible block is mapped for its whole life, and a mapping costs
// address space byte for byte. Every size in Memory.cpp that applies to the host-visible class is
// weighed against that, and Stats() exists so the cost is visible in the log rather than inferred.
//
// Single-threaded by contract, like everything else in src/vulkan/: every caller runs on the
// render thread the device slots run on.
namespace wxlvk::vk::memory
{
    /// Where a resource's memory should live.
    enum class Where
    {
        DeviceLocal, ///< DEVICE_LOCAL, not host-visible: the destination of an upload
        HostVisible, ///< HOST_VISIBLE (+ HOST_COHERENT when offered): the source of an upload
        /// Memory the CPU writes every frame and the GPU reads, preferring
        /// DEVICE_LOCAL | HOST_VISIBLE | HOST_COHERENT (the BAR window, all of VRAM with ReBAR) while
        /// this module holds less than half of that heap, else exactly HostVisible. Mapped, never
        /// read back by the CPU (write-combined either way).
        Stream,
    };

    /// One sub-allocation out of a block. The holder never frees `memory` itself.
    struct Block
    {
        VkDeviceMemory memory = VK_NULL_HANDLE;
        VkDeviceSize   offset = 0;
        VkDeviceSize   size   = 0;      ///< what the caller asked for, not the padded span
        void*          mapped = nullptr;///< the caller's own offset inside the mapping, or null
        uint32_t       owner  = 0;      ///< internal: which slot of the block list this came from
        uint32_t       index  = 0;      ///< internal: that slot's generation, so a stale Free is caught
    };

    /// A VkBuffer with its memory bound.
    struct Buffer
    {
        VkBuffer     handle = VK_NULL_HANDLE;
        Block        block;
        VkDeviceSize size   = 0;
    };

    /**
     * Reads the adapter's memory properties and logs them. Idempotent. Call after the device exists.
     *
     * Logs the whole memory-type table once -- per type its heap, its flags by name and the heap's
     * size -- then which type each Where resolved to and why. That pair of lines is what explains a
     * "the texture is garbage" report at a glance, so it is written even when everything is normal.
     *
     * @return bool ok : false when there is no context, a device entry point this module calls is
     *         null (each logged by name), or no usable memory type exists for either Where
     */
    bool Init();

    /**
     * Sub-allocates memory for a resource the caller has already created.
     *
     * Not logged: a backend calls this thousands of times. A new block is logged; a failure is.
     *
     * @param const VkMemoryRequirements& req : from vkGet*MemoryRequirements
     * @param Where where : which heap class
     * @param Block& out : receives the sub-allocation; untouched on failure
     * @return bool ok
     */
    bool Allocate(const VkMemoryRequirements& req, Where where, Block& out);

    /**
     * Gives a sub-allocation back. The span is reused by later Allocate() calls. A dedicated block
     * is released at once; an ordinary block that empties is kept until Tick() has seen it empty for
     * a few seconds, so a load/unload cycle does not free and re-allocate the same blocks.
     *
     * The caller must already have made sure the GPU is done with the resource; this module cannot
     * know, and does not wait.
     *
     * @param const Block& block : a block from Allocate(); a zeroed block is a no-op
     */
    void Free(const Block& block);

    /**
     * Once per present: returns the ordinary blocks that have stayed empty for about five seconds
     * to the driver, keeping one empty block per (memory type, class). An empty block holds no
     * resource, so no GPU work can still name it.
     */
    void Tick();

    /// The allocator's state beyond Stats(), for the stats line.
    struct Detail
    {
        uint64_t reservedPeak   = 0; ///< the highest reserved figure of the session
        uint64_t mapped         = 0; ///< reserved bytes that are mapped: this process's address space
        uint64_t mappedPeak     = 0;
        uint64_t deviceLocal    = 0; ///< reserved bytes of the DeviceLocal class
        uint64_t returnedBlocks = 0; ///< empty blocks given back by Tick()
        uint64_t returnedBytes  = 0;
        uint64_t dedicatedFreed = 0; ///< dedicated blocks released by Free()
        uint64_t spanGrowths    = 0; ///< free-span lists grown past their first size
        uint32_t spanPeak       = 0; ///< the longest free-span list seen
        uint32_t blocks         = 0; ///< live VkDeviceMemory objects
        uint32_t dedicatedBlocks = 0;
        uint32_t emptyBlocks    = 0; ///< ordinary blocks holding nothing now
        /// Reserved bytes per memory heap (VkPhysicalDeviceMemoryProperties' heap index): the budget's split
        /// (src/native/render/Budget.cpp).
        uint64_t heapReserved[VK_MAX_MEMORY_HEAPS] = {};
    };

    /// Fills `out` (see Detail).
    void GetDetail(Detail& out);

    /**
     * Creates a buffer and binds memory to it in one step.
     *
     * @param VkDeviceSize size : bytes; zero is refused, as vkCreateBuffer would refuse it
     * @param VkBufferUsageFlags usage : e.g. TRANSFER_SRC for staging, TRANSFER_DST | VERTEX for a
     *        device-local vertex buffer
     * @param Where where : which heap class
     * @param Buffer& out : receives the buffer
     * @return bool ok : on failure nothing is left created and `out` is untouched
     */
    bool CreateBuffer(VkDeviceSize size, VkBufferUsageFlags usage, Where where, Buffer& out);

    /**
     * Destroys the buffer and frees its block. Zeroes `buf`. A zeroed buffer is a no-op.
     *
     * @param Buffer& buf : a buffer from CreateBuffer()
     */
    void DestroyBuffer(Buffer& buf);

    /**
     * A host-visible staging buffer of at least `size`, owned by this module, reused across calls
     * and grown when a caller needs more. The returned pointer stays valid until the next call
     * that grows it.
     *
     * Growing destroys the old buffer, so no copy out of it may still be pending on the GPU -- the
     * one-shot upload pattern (write, copy, vkQueueWaitIdle) satisfies that by construction.
     *
     * After writing into `out->block.mapped` and before submitting the copy, call FlushHost() on
     * `out->block`. It costs nothing on coherent memory and it is what keeps a non-coherent adapter
     * from uploading stale bytes.
     *
     * @param VkDeviceSize size : the minimum capacity needed
     * @param Buffer*& out : receives the module's staging buffer
     * @return bool ok
     */
    bool Staging(VkDeviceSize size, Buffer*& out);

    /**
     * Makes host writes into a host-visible block visible to the device.
     *
     * Beyond the required set, and necessarily public. When the adapter offers no HOST_VISIBLE type
     * that is also HOST_COHERENT, Init() falls back to a non-coherent one, and then a CPU write can
     * sit in a cache the GPU does not snoop until vkFlushMappedMemoryRanges is called. Nothing inside
     * this module runs between the caller's last write and the caller's submit, so no internal
     * flush can be placed correctly -- only the caller knows when it has finished writing. On a
     * coherent type (the choice whenever one exists) this returns at once.
     *
     * @param const Block& block : a host-visible block from Allocate(), CreateBuffer() or Staging();
     *        a zeroed or device-local block is a no-op
     * @return bool ok : false only when the flush itself failed, which is logged
     */
    bool FlushHost(const Block& block);

    /**
     * What the allocator has done, for the log and for the address-space accounting.
     *
     * @param uint64 vkAllocateMemoryCalls : successful vkAllocateMemory calls since load, to hold
     *        against the driver's maxMemoryAllocationCount
     * @param uint64 bytesReserved : VkDeviceMemory bytes currently held, every block counted whole
     * @param uint64 bytesInUse : padded bytes currently handed out of those blocks
     */
    void Stats(uint64_t& vkAllocateMemoryCalls, uint64_t& bytesReserved, uint64_t& bytesInUse);
}
