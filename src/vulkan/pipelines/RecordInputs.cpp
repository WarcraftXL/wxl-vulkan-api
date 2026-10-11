// What one draw reads, decided and written before anything is bound: its textures, its descriptor set,
// its ring slice and its bone palette.
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

#include "vulkan/pipelines/Internal.hpp"

#include <intrin.h>

namespace wxlvk::vk::pipelines::detail
{
    /**
     * The textures, one per sampler slot the pixel shader declares.
     *
     * A pipeline whose pixel shader declares no sampler (the colour-only fixed-function row) draws
     * without one: set 0 is still allocated and bound (the layout has it) and nothing is written, which
     * is legal as long as no bound shader uses it. The pipeline is resolved first so that a textureless
     * draw of a key no source covers is named by Pipeline() instead of vanishing into skippedNoTexture.
     *
     * The fixed-function cascade (a hand-written ff row, or a vertex module's generated ff companion)
     * samples a binding only while its stage's selector is an EGxTexOp: the reader writes kFfOpNone for
     * a stage the engine left without a texture, which is the DISABLE the D3D backend sends for it
     * (cgxdevice 6.1), and both shaders then pass the diffuse through without reading the binding. Such a
     * binding must still hold a valid descriptor -- the layout declares it -- and gets the idle-stage
     * image (State::filler), which no texel of the draw ever comes from. A translated pixel module
     * samples what it declares unconditionally, so it gets no such stand-in.
     *
     * What this leaves in the Draft is one image descriptor per written slot and the set cache's key
     * word for it (the image's id and the sampler's key); the VkWriteDescriptorSets are built only
     * when the set is not in the cache (DescriptorSet).
     */
    bool Textures(const Cmd& c, Draft& dr)
    {
        const PipelineEntry* entry = dr.entry;
        const Key&           key   = dr.key;
        const bool cascade    = (entry->row && c.fixedFunction) || entry->companion;
        const bool stage0Idle = cascade && c.ffColorOp == Cmd::kFfOpNone;
        const wxlvk::vk::texture::Image* filler = s.filler.handle != VK_NULL_HANDLE ? &s.filler : nullptr;
        dr.writeCount = 0;
        dr.idleUsed   = false;
        dr.cacheable  = true;
        // The declared slots in ascending order, so a fallback to stage 0 below finds it checked.
        for (uint32_t mask = entry->writeMask; mask; mask &= mask - 1)
        {
            unsigned long slot = 0;
            _BitScanForward(&slot, mask);

            const wxlvk::vk::texture::Image* image = c.texture[slot];
            uint32_t                         flags = c.texFlags[slot];
            if (c.fixedFunction && slot > 0 && (!image || image->handle == VK_NULL_HANDLE))
            {
                // A fixed-function row declares as many samplers as its layout has UV sets, not as
                // many as this draw's cascade uses: stage 1 without a texture is the ordinary case
                // (6.1, the cascade ended at it), and its selector in the slice says so, so the shader
                // never samples the binding. It must still hold something valid, and stage 0's image
                // -- already checked, since the slots go in order -- is at hand; with stage 0 idle
                // too, the idle-stage image.
                image = c.texture[0];
                flags = c.texFlags[0];
                if (cascade && (!image || image->handle == VK_NULL_HANDLE) && filler)
                {
                    image = filler;
                    flags = 0;
                    dr.idleUsed = true;
                }
            }
            else if (slot == 0 && stage0Idle && (!image || image->handle == VK_NULL_HANDLE) && filler)
            {
                image = filler;
                flags = 0;
                dr.idleUsed = true;
            }
            if (!image || image->handle == VK_NULL_HANDLE || image->view == VK_NULL_HANDLE)
            {
                ++s.stats.skippedNoTexture;
                CountSkip(kSkipNoTexture, key, c, entry, static_cast<int32_t>(slot), 0,
                          (c.engineTexMask & (1u << slot))
                              ? "the engine bound a texture on the stage that resolved to no image"
                              : cascade ? "the stage has no texture and its selector samples it"
                                        : "the pixel module samples a stage the engine left without a texture");
                return false;
            }
            // The descriptor names SHADER_READ_ONLY_OPTIMAL, and a barrier cannot be recorded inside
            // the render pass to get there (a subpass self-dependency is the only barrier allowed in
            // one, and it is for attachments). Textures are born and uploaded into it (Image::wanted),
            // and a target goes back to it when its pass ends; an image still an attachment of the
            // open pass is not drawable from it.
            if (image->layout != VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
            {
                ++s.stats.skippedNotReady;
                CountSkip(kSkipNotReady, key, c, entry, static_cast<int32_t>(slot), static_cast<uint32_t>(image->layout),
                          "the image is not in SHADER_READ_ONLY_OPTIMAL (an attachment of the open pass, or mid-upload)");
                return false;
            }
            // The view's type against the shader's sampler type. A cube view in a sampler2D binding,
            // or a 2D view in a samplerCube one, is invalid usage (the descriptor's image view type
            // must match the OpTypeImage dimension), which a driver does not report and may read
            // garbage or fault on. The engine can do this legitimately -- a 2D texture bound on a stage
            // the material samples as an environment map, or the row fallback above copying a cube
            // from stage 0 -- so it is a per-draw skip with its own count, not a refusal of the pair.
            const bool wantCube = (entry->cubeMask & (1u << slot)) != 0;
            if (image->cube != wantCube)
            {
                ++s.stats.skippedSamplerType;
                CountSkip(kSkipSamplerType, key, c, entry, static_cast<int32_t>(slot), image->cube ? 1u : 0u,
                          wantCube ? "a 2D image on a samplerCube binding" : "a cube image on a sampler2D binding");
                return false;
            }
            // The anisotropy bits only key a filter-5 sampler: every other filter ignores them, and
            // keeping them would only split identical samplers. The table answers for every sampler
            // already made; Sampler() makes one on its first use and files it there.
            uint32_t samplerKey = flags & kSamplerKeyMask;
            if ((samplerKey & 0x7u) != 5u) samplerKey &= ~0x3E00u;
            // The world variant (the device's anisotropy and mip LOD bias, Samplers.cpp): a draw of the
            // world -- its depth range ends below 1, which the interface, the text and the engine's
            // full-screen passes never do -- reading an image with mip levels (a render target has one)
            // through a linear mip filter (3 .. 7). The bit is part of the set cache's key word below.
            if (s.worldFilter && image->levels > 1 && (samplerKey & 0x7u) >= 3u && c.viewport.maxDepth < kWorldDepthMax)
                samplerKey |= kSamplerWorldBit;
            VkSampler sampler = s.samplerByKey[SamplerSlot(samplerKey)];
            if (sampler == VK_NULL_HANDLE)
            {
                sampler = Sampler(samplerKey);
                if (sampler == VK_NULL_HANDLE)
                {
                    ++s.stats.skippedUnsupported;   // logged once by Sampler()
                    CountUnsupported(key, c, "no sampler for the texture's flags (see the sampler line)", entry);
                    return false;
                }
            }

            VkDescriptorImageInfo& img = dr.images[dr.writeCount];
            img.sampler     = sampler;
            img.imageView   = image->view;
            img.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            dr.bindings[dr.writeCount]     = static_cast<uint8_t>(slot);
            dr.setWords[1 + dr.writeCount] = static_cast<uint64_t>(image->id) | (static_cast<uint64_t>(samplerKey) << 32);
            if (image->id == 0) dr.cacheable = false;   // not one of the texture module's: no stable identity
            ++dr.writeCount;
        }
        return true;
    }

    /**
     * The ring slice: its sections, whether the previous ring draw's slice can be bound again, and
     * whether there is room for a new one -- decided before the set is allocated, so a skip is whole.
     *
     * The descriptor's range is the larger class, and the bind requires offset + range to stay inside
     * the buffer, so every slice -- of either class -- needs kRingRange bytes ahead of the cursor, not
     * merely its own size. The cursor advances by the slice's own padded size, so the last class-64
     * slices of a frame are not made to pay for a class they are not.
     *
     * The slice's sections are compared against the copy of the previous ring draw's slice in this
     * frame slot. Identical bytes (same class, same high-row count, same ring buffer) bind that slice
     * again: no ring room, no write. The 256-row class writes Cmd::highRows of its 192 high rows: the
     * rows the engine has ever written (the rest of D3D's register file is still zero and no shader
     * indexes it).
     */
    bool PlanSlice(const Cmd& c, Draft& dr)
    {
        const PipelineEntry* entry = dr.entry;
        const uint32_t frameSlot = dr.frameSlot;
        dr.padded = entry->vconst == Cmd::kVertexRegsHigh ? s.padded256 : s.padded64;

        dr.misc[0] = c.alphaRef >= 1u ? static_cast<float>(c.alphaRef) / 255.0f : 0.0f;
        dr.misc[1] = static_cast<float>(c.ffColorMode);
        dr.misc[2] = static_cast<float>(c.ffColorOp);
        dr.misc[3] = static_cast<float>(c.ffAlphaOp);
        constexpr uint32_t kHighMax = Cmd::kVertexRegsHigh - Cmd::kVertexRegs;
        dr.highRows = entry->vconst != Cmd::kVertexRegsHigh ? 0u : (c.highRows < kHighMax ? c.highRows : kHighMax);
        dr.sections[0] = { c.pixelConst, Cmd::kPixelRegs };
        dr.sections[1] = { dr.misc, 1u };
        dr.sections[2] = { c.vertexRows, Cmd::kVertexRegs };
        dr.sections[3] = { c.vertexConstHigh, dr.highRows };
        dr.reuse     = false;
        dr.firstDiff = 0;
        dr.diffRow   = 0;
        if (entry->usesRing)
        {
            const State::LastSlice& ls = s.lastSlice[frameSlot];
            if (ls.valid && ls.ring == s.ringActive[frameSlot] && ls.vconst == entry->vconst && ls.highRows == dr.highRows)
            {
                const char* copy = reinterpret_cast<const char*>(s.lastSliceData[frameSlot]);
                for (; dr.firstDiff < 4; ++dr.firstDiff)
                {
                    const SliceSection& sec = dr.sections[dr.firstDiff];
                    if (sec.rows && !wxlvk::simd::EqualRows(copy + dr.diffRow * 16u, sec.src, sec.rows)) break;
                    dr.diffRow += sec.rows;
                }
                dr.reuse = dr.firstDiff == 4;
            }
        }

        if (entry->usesRing && !dr.reuse && s.ringCursor[frameSlot] + kRingRange > kRingBytes && !SpillRing(frameSlot))
        {
            ++s.stats.uniformExhausted;
            CountSkip(kSkipUniforms, dr.key, c, entry, -1, 0, "the frame slot's uniform ring and its spills are full");
            if (!s.loggedUniformExhausted)
            {
                s.loggedUniformExhausted = true;
                wxl::ScriptMgr::Log(WXL_LOG_WARN,
                                    "vulkan: pipelines: the uniform ring for slot %u is full (%u KiB, kRingBytes); the "
                                    "frame has more ring draws than fit. Further exhaustions are counted only",
                                    frameSlot, kRingBytes / 1024);
            }
            return false;
        }
        return true;
    }

    /**
     * The descriptor set: from the slot's cache, or allocated and written.
     *
     * The key is the layout, the written-slot mask and every written slot's (image id, sampler key):
     * exactly what the set's contents are a function of (kSetCacheSlots). A hit is a set this frame
     * slot wrote for the same tuple, in this frame or an earlier one since its last flush; a miss
     * allocates from the slot's pool, writes, and remembers it.
     */
    bool DescriptorSet(const Cmd& c, Draft& dr)
    {
        const PipelineEntry*  entry = dr.entry;
        const uint32_t        frameSlot = dr.frameSlot;
        const loader::Device& d      = *s.dapi;
        const LayoutEntry&    layout = s.layouts[entry->layout];

        SetCache&       cache    = g_setCache[frameSlot];
        const uint32_t  keyCount = 1 + dr.writeCount;
        const uint64_t* keyWords = dr.setWords;
        dr.setWords[0] = static_cast<uint64_t>(entry->layout) | (static_cast<uint64_t>(entry->writeMask) << 32);
        uint32_t keyHash = 2166136261u;
        for (uint32_t i = 0; i < keyCount; ++i)
        {
            keyHash = (keyHash ^ static_cast<uint32_t>(keyWords[i])) * 16777619u;
            keyHash = (keyHash ^ static_cast<uint32_t>(keyWords[i] >> 32)) * 16777619u;
        }
        keyHash ^= keyHash >> 15;

        VkDescriptorSet set      = VK_NULL_HANDLE;
        uint32_t        freeSlot = kSetCacheSlots;
        if (dr.cacheable)
            for (uint32_t p = keyHash & (kSetCacheSlots - 1);; p = (p + 1) & (kSetCacheSlots - 1))
            {
                const SetEntry& e = cache.table[p];
                if (e.gen != cache.gen)
                {
                    freeSlot = p;
                    break;
                }
                if (e.hash == keyHash && e.words == keyCount
                    && wxlvk::simd::EqualWords64(&cache.arena[e.at], keyWords, keyCount))
                {
                    set = e.set;
                    break;
                }
            }
        if (set != VK_NULL_HANDLE)
        {
            ++s.stats.setsReused;
            dr.set = set;
            return true;
        }

        VkDescriptorSetAllocateInfo alloc = {};
        alloc.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        alloc.descriptorPool     = ActivePool(frameSlot);
        alloc.descriptorSetCount = 1;
        alloc.pSetLayouts        = &layout.set0;

        VkResult r = d.AllocateDescriptorSets(s.device, &alloc, &set);
        if ((r != VK_SUCCESS || set == VK_NULL_HANDLE) && SpillPool(frameSlot))
        {
            // The slot outgrew the pool; once more from the spill. The next BeginFrame() of the
            // slot flushes the pool and the cache together.
            alloc.descriptorPool = ActivePool(frameSlot);
            set = VK_NULL_HANDLE;
            r   = d.AllocateDescriptorSets(s.device, &alloc, &set);
        }
        if (r != VK_SUCCESS || set == VK_NULL_HANDLE)
        {
            // OUT_OF_POOL_MEMORY on a 1.1 driver, FRAGMENTED_POOL or OUT_OF_HOST_MEMORY on 1.0; all of
            // them mean the same thing here, since the pool is sized for a full frame.
            ++s.stats.descriptorExhausted;
            CountSkip(kSkipDescriptors, dr.key, c, entry, -1, 0, "the frame slot's descriptor pool and its spills are full");
            if (!s.loggedExhausted)
            {
                s.loggedExhausted = true;
                wxl::ScriptMgr::Log(WXL_LOG_WARN,
                                    "vulkan: pipelines: vkAllocateDescriptorSets returned %s; the frame has more "
                                    "draws than kSetsPerPool (%u) or more samplers than the pool's average, or the "
                                    "pool was not reset. Further exhaustions are counted only",
                                    ResultName(r), kSetsPerPool);
            }
            return false;
        }

        if (dr.writeCount)
        {
            // Built here, on the miss path only, field by field: no zeroing of the whole array.
            VkWriteDescriptorSet writes[Cmd::kTextureStages];
            for (uint32_t i = 0; i < dr.writeCount; ++i)
            {
                VkWriteDescriptorSet& w = writes[i];
                w.sType            = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                w.pNext            = nullptr;
                w.dstSet           = set;
                w.dstBinding       = dr.bindings[i];
                w.dstArrayElement  = 0;
                w.descriptorCount  = 1;
                w.descriptorType   = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                w.pImageInfo       = &dr.images[i];
                w.pBufferInfo      = nullptr;
                w.pTexelBufferView = nullptr;
            }
            d.UpdateDescriptorSets(s.device, dr.writeCount, writes, 0, nullptr);
            s.stats.descriptorWrites += dr.writeCount;
        }
        ++s.stats.setsWritten;

        // Remembered until the slot's next flush, while the table and the arena have room.
        if (dr.cacheable && freeSlot < kSetCacheSlots && cache.count < kSetCacheMax && cache.used + keyCount <= kSetCacheWords)
        {
            SetEntry& e = cache.table[freeSlot];
            e.set   = set;
            e.gen   = cache.gen;
            e.hash  = keyHash;
            e.at    = cache.used;
            e.words = keyCount;
            for (uint32_t i = 0; i < keyCount; ++i) cache.arena[cache.used + i] = keyWords[i];
            cache.used += keyCount;
            ++cache.count;
            if (cache.count > s.stats.setCachePeak) s.stats.setCachePeak = cache.count;
        }
        else
            ++s.stats.setsUncached;
        dr.set = set;
        return true;
    }

    /**
     * The ring slice, written (or the previous one bound again).
     *
     * The block's order: the pixel rows, the misc row, then the vertex rows -- the first 64 from the Cmd
     * and, for the 256-row class, the other 192 from vertexConstHigh. The alpha reference goes over as
     * ref / 255 so the fragment shader compares it against its own 0..1 alpha; 0 means render state 7 is
     * off (ref < 1). The other three are the fixed-function selectors draw::Cmd documents (colour mode,
     * stage-0 colour op, stage-0 alpha op), zero for a shader-driven draw and read by nothing then. The
     * flush is a no-op on a coherent type, which every desktop adapter offers; on the non-coherent
     * fallback it flushes the block once per draw, which is correct and slow, and the log line Memory.cpp
     * prints at Init() is where that fallback announces itself.
     *
     * The reused slice, or the sections from the first difference copied into the CPU copy (the ones
     * before it are already equal there) and the whole slice written to the ring in one sequential copy
     * of 16-byte rows -- the write-combined memory's best case, and five times the CRT's memcpy there.
     */
    void WriteSlice(Draft& dr)
    {
        dr.dynamicOffset = 0;
        if (!dr.entry->usesRing) return;
        const uint32_t frameSlot = dr.frameSlot;
        State::LastSlice& ls = s.lastSlice[frameSlot];
        if (dr.reuse)
        {
            dr.dynamicOffset = ls.offset;
            ++s.stats.slicesReused;
            return;
        }
        char*    copy = reinterpret_cast<char*>(s.lastSliceData[frameSlot]);
        uint32_t at   = dr.diffRow;
        for (uint32_t i = dr.firstDiff; i < 4; ++i)
        {
            if (dr.sections[i].rows) wxlvk::simd::CopyRows(copy + at * 16u, dr.sections[i].src, dr.sections[i].rows);
            at += dr.sections[i].rows;
        }
        dr.dynamicOffset = s.ringCursor[frameSlot];
        s.ringCursor[frameSlot] += dr.padded;
        wxlvk::simd::CopyRows(static_cast<char*>(RingBuffer(frameSlot).block.mapped) + dr.dynamicOffset, copy, at);
        memory::FlushHost(RingBuffer(frameSlot).block);
        ls.valid    = true;
        ls.offset   = dr.dynamicOffset;
        ls.ring     = s.ringActive[frameSlot];
        ls.vconst   = dr.entry->vconst;
        ls.highRows = dr.highRows;
        ++s.stats.slicesWritten;
        s.stats.sliceBytes += static_cast<uint64_t>(at) * 16u;
        constexpr uint32_t kHighMax = Cmd::kVertexRegsHigh - Cmd::kVertexRegs;
        if (dr.entry->vconst == Cmd::kVertexRegsHigh) s.stats.highRowsSkipped += kHighMax - dr.highRows;
    }

    /**
     * The bone palette of a `bones` variant pipeline: rows c31.. as far as the engine has ever written
     * them, into the slot's palette ring when the draw's serial differs from the palette the slot wrote
     * last; every draw until the next change binds that offset. MakeKey() only allowed the variant when
     * the ring had room.
     */
    void WritePalette(const Cmd& c, Draft& dr)
    {
        dr.paletteOffset = 0;
        if (!dr.entry->bones) return;
        const uint32_t frameSlot = dr.frameSlot;
        State::LastPalette& lp = s.lastPalette[frameSlot];
        if (lp.valid && lp.serial == c.paletteSerial && lp.count == c.paletteCount)
        {
            dr.paletteOffset = lp.offset;
            ++s.stats.paletteReused;
            return;
        }
        const uint32_t rows  = c.paletteCount < manifest::kBoneRows ? c.paletteCount : manifest::kBoneRows;
        const uint32_t bytes = rows * 16u;
        dr.paletteOffset = s.boneCursor[frameSlot];
        if (rows)
            wxlvk::simd::CopyRows(static_cast<char*>(s.boneRing[frameSlot].block.mapped) + dr.paletteOffset, c.paletteRows, rows);
        memory::FlushHost(s.boneRing[frameSlot].block);
        const uint32_t step = bytes ? bytes : 16u;
        s.boneCursor[frameSlot] += (step + s.boneAlign - 1) / s.boneAlign * s.boneAlign;
        lp.valid  = true;
        lp.serial = c.paletteSerial;
        lp.count  = c.paletteCount;
        lp.offset = dr.paletteOffset;
        ++s.stats.paletteUploads;
        s.stats.paletteBytes += bytes;
    }
}
