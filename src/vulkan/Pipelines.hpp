// The Vulkan half of a draw: pipelines, samplers, descriptor sets, and the one function that binds
// everything a draw::Cmd names and issues it.
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

#include "draw/Cmd.hpp"
#include "vulkan/Vk.hpp"

#include <cstdint>

// src/draw/Reader.cpp fills a draw::Cmd per engine draw; this file consumes one. It never reads
// the engine: everything it needs is in the Cmd by value, and anything it would need that is not
// there is a change to src/draw/Cmd.hpp first.
//
// What exists here, and why it is shaped the way it is:
//
//   - descriptor set layouts for set 0 (the samplers), one per distinct set of D3D sampler slots
//     a pixel module declares -- binding N is a combined image sampler for texture stage N,
//     fragment stage -- and a pipeline layout over each of them plus the one set-1 layout. Set 1,
//     binding 0 is a dynamic uniform buffer (vertex and fragment stages): one slice of the uniform
//     ring below, the constant registers as the Cmd carries them. Every pipeline layout also
//     keeps the UI's 64-byte vertex push-constant range: the UI pair reads c0..c3 from it and never
//     binds set 1; every other pair reads set 1 and never pushes. The hand-written rows all use the
//     layout for slot {0}; a translated pair uses the layout its pixel module's manifest row names.
//   - the uniform ring: per frame-in-flight slot, one persistently mapped buffer (device-local
//     host-visible when the adapter has some), byte-addressed, into which Record() writes one
//     slice per draw: the pixel rows, the misc row and then the vertex rows, std140, in one of two
//     classes -- 64 vertex rows (1296 bytes) or 256 (4368 bytes), decided by the draw's vertex
//     shader -- each padded to minUniformBufferOffsetAlignment. One set-1 descriptor per slot
//     points at the whole buffer with range = the larger class, so a draw costs a memcpy and a
//     vkCmdBindDescriptorSets with a dynamic offset; a draw whose slice is byte for byte the
//     previous one's binds it again. BeginFrame() rewinds the slot's cursor; a frame that runs the
//     ring dry spills into up to three more buffers.
//   - a descriptor pool per frame-in-flight slot for set 0, with spill pools behind it, and a
//     per-slot cache over it that lives across the slot's frames: a set is allocated and written
//     once per distinct (layout, images, samplers) tuple and bound from the cache after that, until
//     the cache or the pool fills and BeginFrame() resets both (a zone change). Record() also tracks
//     what the command buffer has bound and records only changes.
//   - a sampler cache keyed on the engine's own texture flags, and a pipeline cache keyed on the
//     material and fixed state a Cmd carries plus the render pass it is drawn in, through a
//     VkPipelineCache persisted beside the DLL. Which (vs, ps, vertex format) triples can be built
//     is a table of hand-written rows in Pipelines.cpp, and whatever pair the translated-module
//     manifest resolves (src/vulkan/Manifest.hpp through src/shaders/Translated.hpp): the vertex
//     input comes from the format and the vertex module's declared inputs, the set-0 layout from
//     the pixel module's samplers, and the slice class from the vertex module's constant window.
//     WXL_VK_TRANSLATED (src/common/Settings.hpp) says which of the two wins when both exist. A
//     translated vertex module drawn with NO pixel shader (the vs_2_0 weather records) pairs with
//     the generated `ff` companion its manifest row names; a vertex row flagged `point_size` draws
//     as a point list.
//
// Vulkan 1.0 core only: the render passes are src/native/Present.cpp's (one per pass class, kept for
// the device's life), the viewport is y-down and the vertex shaders flip y (see
// src/shaders/glsl/ui.vert), and there are no push descriptors. Every function here must run on the
// render thread, inside the frame's command buffer.
namespace wxlvk::vk::pipelines
{
    /// How many frame-in-flight slots the descriptor pools and rings are made for. Must equal the
    /// frame's kFramesInFlight (src/native/Present.cpp static_asserts it); BeginFrame() refuses a
    /// slot at or above this and logs it once.
    constexpr uint32_t kFrameSlots = 2;

    /**
     * Creates the layouts, the per-slot descriptor pools, the uniform ring and its per-slot
     * descriptors, and checks every entry point this module calls. Idempotent. Needs a live
     * context, memory::Init() and shaders::Init() to have succeeded.
     *
     * @return bool ok : false when there is no context, an entry point is missing (each logged by
     *         name), or a create failed; on false nothing this call created is left behind
     */
    bool Init();

    /**
     * Opens a frame on one frame-in-flight slot: rewinds that slot's uniform ring to its first byte,
     * and -- only when the slot's descriptor-set cache or its pool is nearly full -- resets the pool,
     * which frees every set Record() allocated from it, and forgets the cache. In a steady scene the
     * sets written in earlier frames of the slot are bound again from the cache.
     *
     * Call it once per frame, with the slot the frame is about to record -- after that slot's fence
     * has been waited on, so nothing in flight still reads the sets being freed.
     *
     * @param uint32 frameSlot : 0 .. kFrameSlots - 1
     */
    void BeginFrame(uint32_t frameSlot);

    /**
     * Binds everything `c` names and issues the draw: the pipeline for its material and fixed
     * state, its vertex and index buffers at offset 0, a descriptor set holding one texture per
     * sampler slot the pipeline's pixel shader declares, each with a cached sampler for that
     * stage's flags, its constants -- c0..c3 as push constants for the UI pair, the whole
     * pixelConst / alphaRef / vertexConst block as one ring slice (set 1, dynamic offset) for every
     * other pair -- and its viewport and scissor as dynamic state.
     *
     * Must be called inside the render pass `pass`, on `cmd`, between BeginFrame(frameSlot) and
     * the frame's vkEndCommandBuffer. A Cmd that cannot be drawn is skipped and counted, never
     * half-recorded: an unsupported vs/ps/format/blend/topology combination (logged once per key
     * at DEBUG), a declared sampler slot with no texture, a texture whose image is not in
     * SHADER_READ_ONLY_OPTIMAL, a descriptor pool with no set left, or a uniform ring with no room
     * for a slice.
     *
     * Nothing is logged per draw.
     *
     * @param VkCommandBuffer cmd : recording, inside `pass`
     * @param VkRenderPass pass : the pass the pipeline must be compatible with (present::RenderPass())
     * @param uint32 frameSlot : the slot BeginFrame() was called with
     * @param const Cmd& c : the draw
     */
    void Record(VkCommandBuffer cmd, VkRenderPass pass, uint32_t frameSlot, const wxlvk::draw::Cmd& c);

    /**
     * Forgets what Record() believes the command buffer has bound (pipeline, descriptor sets,
     * buffers, dynamic state, push constants), so the next draw binds everything. Called at every
     * render-pass begin and by anything else that binds into the frame's command buffer (the
     * overlay). Record() also forgets on its own when the command buffer handle changes.
     */
    void InvalidateBindings();

    /**
     * A render pass class was made (src/native/Present.cpp): remembers its formats, and builds
     * every pipeline pipeline-keys.bin lists for that class from the modules it was built from
     * last session (by md5, through the VkPipelineCache), so the draws that want them later find
     * them built. Each pre-warmed pipeline's first draw checks it against the sources a draw-time
     * build would use, and rebuilds it if they differ.
     *
     * @param VkRenderPass pass : the class's pass (LOAD / LOAD), as RenderPass() hands it to Record()
     * @param VkFormat colour : its colour format
     * @param VkFormat depth : its depth format, VK_FORMAT_UNDEFINED without depth
     * @param bool gbuffer : the class carries the G-buffer's three attachments after the colour (four
     *        colour attachments in all): its pipelines name four blend attachments, and the opaque
     *        world's draws may take the modules' `gbuf` variants
     */
    void PassCreated(VkRenderPass pass, VkFormat colour, VkFormat depth, bool gbuffer);

    /// What a captured draw's shadow pipeline is built from (src/capture/ShadowPipelines.cpp).
    struct ShadowSource
    {
        VkPipelineLayout                  layout = VK_NULL_HANDLE;  ///< the entry's (sets 0..2 and the push range)
        VkVertexInputBindingDescription   bindings[4] = {};
        uint32_t                          bindingCount = 0;
        VkVertexInputAttributeDescription attrs[16] = {};
        uint32_t                          attrCount = 0;
        VkPrimitiveTopology               topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        char                              vertPath[160] = {};   ///< under shaders/: the shadow variant
        char                              fragPath[160] = {};   ///< under shaders/: the pixel module, alpha-tested only; "" otherwise
        bool                              bones = false;
        bool                              wind = false;   ///< a tree-wind entry: set 3, its push values at 0, the matrix at 16
    };

    /**
     * The shadow variant's sources for pipeline entry `entryIndex` (a capture::Draw's `entry`):
     * its vertex input, topology, layout and modules.
     *
     * @param bool alphaTest : with the entry's pixel module (its alpha test), else no fragment stage
     * @return bool ok : false when the entry has no shadow variant or cannot be described
     */
    bool ShadowSourceFor(uint32_t entryIndex, bool alphaTest, ShadowSource& out);

    /**
     * Whether a shader name appears, on its side, in the hand-written row table. The native
     * device's slot 72 (src/native/Shaders.cpp) treats such a name as drawable even when the
     * manifest does not know its bytes, so the UI keeps its shipped pairs whatever the manifest
     * holds. Pure table lookup; no Vulkan state, callable before Init().
     *
     * @param string name : CGxShader +0x18, e.g. "UI:0", "Desaturate"
     * @param bool pixel : the pixel side (`ps` column) or the vertex side (`vs`)
     * @return bool listed
     */
    bool HandWrittenRow(const char* name, bool pixel);

    /// Counters since Init(). Every skipped draw lands in exactly one of the skip buckets.
    struct Stats
    {
        uint64_t drawn;               ///< draws issued
        uint64_t class64;             ///< of which through a 64-row slice (1296 bytes)
        uint64_t class256;            ///< of which through a 256-row slice (4368 bytes)
        uint64_t skippedUnsupported;  ///< a key with no pipeline
        uint64_t skippedNoTexture;    ///< a declared sampler slot had no texture, or it had no image
        uint64_t skippedNotReady;     ///< the image exists but is not in SHADER_READ_ONLY_OPTIMAL
        uint64_t skippedSamplerType;  ///< a 2D image bound where the pixel module declares a cube sampler, or a cube where it declares 2D
        uint64_t descriptorExhausted; ///< the slot's pool had no set left
        uint64_t uniformExhausted;    ///< the slot's uniform ring had no room, or the Cmd lacked its high rows
        uint64_t companionDraws;      ///< of `drawn`, issued through a translated vertex module and its generated ff companion
        uint64_t depthBiasDraws;      ///< of `drawn`, with a dynamic depth bias (render state 0)
        uint64_t blendFactorDraws;    ///< of `drawn`, with blend mode 11 and its dynamic constant (state 84)
        uint64_t multiStreamDraws;    ///< of `drawn`, binding more than one vertex stream of the draw's own
        uint64_t coveredDraws;        ///< of `drawn`, fixed function on a layout no row is declared for, the missing elements from the constant stream
        uint64_t idleStageDraws;      ///< of `drawn`, a fixed-function cascade with a disabled stage whose declared binding held the idle-stage image
        uint64_t ringSpills;          ///< frames-slot moves to a spill ring buffer (a frame outgrew 8 MiB of uniforms)
        uint64_t poolSpills;          ///< moves to a spill descriptor pool
        uint32_t ringSpillBuffers;    ///< spill ring buffers made (kept for the session)
        uint32_t poolSpillPools;      ///< spill descriptor pools made
        uint32_t ringMostBuffers;     ///< the most ring buffers one frame wrote into (1 = never spilled)
        uint32_t pipelines;           ///< live in the cache
        uint32_t translatedPipelines; ///< of which built from translated modules
        uint32_t companionPipelines;  ///< of the translated ones, built from a vertex module and its ff companion
        uint32_t samplers;            ///< live in the cache
        uint32_t anisoSamplers;       ///< of which anisotropic (filter 5, or the world variant, with samplerAnisotropy)
        uint32_t layouts;             ///< distinct set-0 layouts (sampler slot sets) live
        uint64_t slicesReused;        ///< ring draws that bound the previous draw's slice again (same bytes)
        uint64_t slicesWritten;       ///< ring draws that wrote a slice
        uint64_t sliceBytes;          ///< bytes those wrote into the ring
        uint64_t highRowsSkipped;     ///< 256-row draws' high rows not copied (above the engine's high-water row)
        uint64_t setsWritten;         ///< set-0 descriptor sets allocated and written
        uint64_t setsReused;          ///< draws that bound a set this frame slot had already written for the same tuple (this frame or an earlier one)
        uint64_t setsUncached;        ///< of setsWritten, not remembered (the slot's cache was full, or an image without an id)
        uint64_t descriptorWrites;    ///< image descriptors written
        uint32_t setCachePeak;        ///< the most sets one frame slot's cache held
        uint64_t setCacheFlushes;     ///< slot caches flushed with their pool at a BeginFrame (nearly full, or the pool spilled)
        uint64_t pipelineBinds;       ///< vkCmdBindPipeline recorded
        uint64_t pipelineBindsSkipped;///< draws whose pipeline was already bound
        uint64_t setBinds;            ///< vkCmdBindDescriptorSets recorded
        uint64_t setBindsSkipped;     ///< vkCmdBindDescriptorSets avoided: the sets and offsets were already bound
        uint64_t bonesDraws;          ///< of `drawn`, through a module's bone-palette variant (64-row slice, palette in set 2)
        uint64_t gbufferDraws;        ///< of `drawn`, through the modules' G-buffer variants: the pass's albedo, normal and light attachments written
        uint32_t gbufferPipelines;    ///< pipelines built with the G-buffer variants
        uint64_t gbufferNullDraws;    ///< of `drawn`, depth-writing draws on a G-buffer pass with no variant: no record of their own (Build's log names them)
        uint32_t gbufferNullPipelines;///< the pipelines those draws went through
        uint64_t paletteUploads;      ///< palettes written into the palette ring
        uint64_t paletteReused;       ///< bone-palette draws that bound the slot's last palette again
        uint64_t paletteBytes;        ///< bytes those uploads wrote
        uint64_t bonesRingFull;       ///< 256-row draws keyed to the 256-row module because the palette ring was full
        uint64_t buildsAtDraw;        ///< pipelines built because a draw asked for a key (the frame-time cost)
        uint64_t prewarmed;           ///< pipelines built at bring-up from pipeline-keys.bin
        uint64_t prewarmSkipped;      ///< persisted keys not pre-warmed (a module md5 no longer in the manifest, a failed build)
        uint64_t prewarmVerified;     ///< pre-warmed entries whose first draw resolved to the same sources
        uint64_t prewarmRebuilt;      ///< ... resolved to other sources and were rebuilt
        uint64_t prewarmTicks;        ///< QPC ticks the pre-warm took
        uint32_t keysLoaded;          ///< entries read from pipeline-keys.bin
        uint64_t bufferBinds;         ///< vertex / index buffer binds recorded
        uint64_t bufferBindsSkipped;  ///< ... not recorded, already bound
        uint64_t dynamicSets;         ///< viewport / scissor / depth bias / blend constant calls recorded
        uint64_t dynamicSkipped;      ///< ... not recorded, the value was current
        uint64_t pushes;              ///< push-constant updates recorded (the UI path)
        uint64_t pushesSkipped;       ///< ... not recorded, the bytes were current
    };

    /**
     * The counters.
     *
     * @param Stats& out : filled in whole
     */
    void GetStats(Stats& out);

    /// Stats::drawn alone, without the copy: slot 42 compares it around Record() for the draws an
    /// occlusion query counts.
    uint64_t DrawnCount();

    /**
     * Logs the census of skipped draws at INFO, for the stats block (recorded only with
     * WXL_VK_STATS): one `native: skipped <bucket>:` line per distinct (bucket, key) pair, at most
     * 32 across the six buckets (unsupported, no texture, not ready, sampler type, descriptors,
     * uniforms), with its draw count, the key's fields, the pipeline's source and sampler slots,
     * the stages with an image and the stages the engine bound, the fixed-function selectors, the
     * slot the skip was decided at, and the reason; nothing when no draw was skipped.
     */
    void LogUnsupported();

    /**
     * The time spent building pipelines (vkCreateGraphicsPipelines and what prepares it), as
     * QueryPerformanceCounter ticks, and the number of builds attempted. Monotonic; the CPU-time
     * measurements diff it per frame.
     */
    void BuildCounters(uint64_t& ticks, uint64_t& builds);

    /**
     * The record split (WXL_VK_RECORD_SPLIT): each phase's mean cost a draw since the start, in microseconds (the
     * time-stamp counter calibrated against the performance counter over the session), the draws timed and the counter's
     * rate. False when it is off or too early to calibrate. Phases: 0 the key and the pipeline, 1 the textures, the slice
     * and the descriptor set, 2 the constants and the palette copied, 3 the binds, 4 the shadow capture, 5 the draw call.
     */
    bool RecordSplit(double (&usPerDraw)[6], uint64_t& draws, double& tscMhz);

    /**
     * Writes the VkPipelineCache to the extension's folder (pipeline-cache.bin, through a
     * temporary file and a rename) when pipelines were built since the last save or the load, and
     * the list of built keys beside it (pipeline-keys.bin, the pre-warm's input). Cheap when
     * nothing changed.
     */
    void SavePipelineCache();

    /// The persisted VkPipelineCache, for pipelines built outside this module (the rendering API's
    /// consumers); VK_NULL_HANDLE before Init() or when it could not be made.
    VkPipelineCache PipelineCache();
}
