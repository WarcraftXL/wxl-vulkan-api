// The pipelines' shared internals: the constants, the tables, the state and the helpers every part of
// src/vulkan/pipelines/ uses.
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

#include "vulkan/Pipelines.hpp"

#include "common/ModulePath.hpp"
#include "common/Settings.hpp"
#include "common/Simd.hpp"
#include "shaders/Translated.hpp"

#include "vulkan/Context.hpp"
#include "vulkan/Loader.hpp"
#include "vulkan/Manifest.hpp"
#include "vulkan/Memory.hpp"
#include "vulkan/Shaders.hpp"
#include "vulkan/Texture.hpp"
#include "vulkan/Wind.hpp"
#include "capture/Capture.hpp"

#include "wxl/Script.hpp"

#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
// The pipelines live in src/vulkan/pipelines/, one file per responsibility, over this one header. Nothing
// here is visible outside them: the public face is src/vulkan/Pipelines.hpp (src/vulkan/Pipelines.cpp).
//
//   State        the state's definition, the teardown of layouts, pools and rings
//   Keys         a draw's key, its hash, the logs of what cannot be built
//   Encodings    the engine's blend, write mask, topology and fixed formats in Vulkan terms
//   VertexInput  the translated modules' vertex inputs, the custom layouts, the hand-written rows
//   Layouts      the set-0 / pipeline layouts, the bone-palette and tree-wind layouts
//   Sources      cull and depth state, what a key is built from
//   Build        building one pipeline
//   Lookup       the pipeline of a key (the hashed table, the pre-warm's check)
//   Samplers     the sampler cache
//   Ring         the uniform ring and the bone-palette ring, their spills
//   Cache        the persisted VkPipelineCache and key list, the pre-warm
//   Record       recording one draw
namespace wxlvk::vk::pipelines::detail
{
    namespace context    = wxlvk::vk::context;
    namespace loader     = wxlvk::vk::loader;
    namespace memory     = wxlvk::vk::memory;
    namespace texture    = wxlvk::vk::texture;
    namespace shaders    = wxlvk::vk::shaders;
    namespace manifest   = wxlvk::vk::manifest;
    namespace translated = wxlvk::shaders::translated;
    namespace settings   = wxlvk::settings;

    using wxlvk::draw::Cmd;
    using wxlvk::vk::ResultName;
    using wxlvk::vk::pipelines::kFrameSlots;

    /// The most pipelines the cache holds. The interface is a handful; a zone is a few hundred
    /// material and state combinations once the translated modules draw the world. Fixed, like
    /// every table in src/vulkan/, because address space is the scarce resource in this 32-bit
    /// process; a full cache refuses and counts.
    // An entry is a key, a handle and a few flags (a few hundred bytes); the persisted key list
    // pre-warms up to this many, and a world with tree wind keeps a second key per windy M2 state.
    constexpr uint32_t kMaxPipelines = 8192;

    /// Descriptor sets one frame slot's set-0 pool holds (with four combined image samplers per
    /// set on average), and so the draws a frame records before it spills into another pool. The
    /// UI alone is a few hundred draws per frame; an in-world frame a few thousand.
    constexpr uint32_t kSetsPerPool = 8192;

    /// The descriptor-set cache (SetCache, one per frame slot): a set-0 descriptor set is a pure
    /// function of (layout, the image and sampler of every slot it writes), and every frame binds
    /// the same few hundred tuples again and again -- the UI's glyph pages and panels, a model's
    /// batches, the terrain's layer textures. A tuple is allocated and written once and bound from
    /// the cache after that, ACROSS the slot's frames: the slot's pool is not reset per frame but
    /// only when the cache or the pool fills (BeginFrame(), after the slot's fence), which a
    /// steady scene never does and a zone change does once. The key names an image by its own
    /// never-reused id (texture::Image::id), not by a view handle the driver may give to a new
    /// image, and a sampler by its flags; a cached set whose image is gone is simply never matched
    /// again (a descriptor that is never bound may name a destroyed view). Bounded: kSetCacheSlots
    /// hashed entries (half of them filled at most, so a probe stays short) over an arena of
    /// kSetCacheWords key words; past either, a draw allocates as before, uncached, and the next
    /// BeginFrame() of the slot flushes the lot.
    constexpr uint32_t kSetCacheSlots = 4096;
    constexpr uint32_t kSetCacheMax   = kSetCacheSlots / 2;
    constexpr uint32_t kSetCacheWords = 32768;
    static_assert((kSetCacheSlots & (kSetCacheSlots - 1)) == 0, "the probe masks with kSetCacheSlots - 1");
    /// Past these at a slot's BeginFrame(), its cache and pool are flushed: room for a frame's
    /// worth of new tuples before the table fills mid-frame.
    constexpr uint32_t kSetCacheFlushEntries = kSetCacheMax * 3 / 4;
    constexpr uint32_t kSetCacheFlushWords   = kSetCacheWords * 3 / 4;

    /// The lookup index over the pipeline table (dense index + 1, 0 empty), twice its size, linearly
    /// probed: an in-world frame changes key a thousand times or more, and a scan of the table
    /// (hundreds of entries, a memcmp of the key each) per change is too slow.
    constexpr uint32_t kPipeHashSlots = 2 * kMaxPipelines;
    static_assert((kPipeHashSlots & (kPipeHashSlots - 1)) == 0, "the probe masks with kPipeHashSlots - 1");

    /// The pipeline cache file beside the DLL (src/common/ModulePath.hpp), and the
    /// largest blob it is trusted with (a session's few hundred pipelines are a few MiB at most).
    constexpr const char* kCacheFile     = "pipeline-cache.bin";
    constexpr size_t      kCacheMaxBytes = 64u * 1024u * 1024u;

    /// The most samplers the cache holds. The key below has 6 filters x 4 wrap combinations x the
    /// anisotropy bits, and the world variant of the mipmapped ones; the UI uses two or three of them,
    /// the scene perhaps a dozen and their world variants.
    constexpr uint32_t kMaxSamplers = 64;

    /// The most distinct set-0 layouts: one per distinct set of sampler slots a pixel module
    /// declares. The client's modules use {0}, {0,1}, {0..3}, {0..8} and a few others; sixteen is ample.
    constexpr uint32_t kMaxLayouts = 16;

    /// The most distinct unsupported keys that get their own log line. Past this the skips are
    /// still counted, just not named -- a scene that asks for more than this many materials this
    /// stage cannot draw has made its point.
    constexpr uint32_t kMaxSkippedKeys = 128;

    /// The most (skip bucket, key) pairs LogUnsupported() names with their draw counts, across every
    /// bucket; skips of further pairs are summed per bucket. Recorded only with WXL_VK_STATS.
    constexpr uint32_t kMaxUnsupported = 32;

    /// The skip buckets of Record(), in the order the draws line prints them (Stats::skipped*,
    /// descriptorExhausted, uniformExhausted). The census names its keys per bucket.
    enum SkipBucket : uint32_t
    {
        kSkipUnsupported = 0,
        kSkipNoTexture,
        kSkipNotReady,
        kSkipSamplerType,
        kSkipDescriptors,
        kSkipUniforms,
        kSkipBuckets
    };
    inline constexpr const char* kSkipBucketNames[kSkipBuckets] = {
        "unsupported", "no texture", "not ready", "sampler type", "descriptors", "uniforms"
    };

    /// The idle-stage image (State::filler): 1x1 R8G8B8A8, opaque black -- what D3D9 returns for a
    /// sampler with no texture. Bound only on a binding the fixed-function cascade provably does not
    /// read for the draw (its stage-0 selector is DISABLE), never in place of a texture a stage uses.
    inline constexpr float kFillerColour[4] = { 0.0f, 0.0f, 0.0f, 1.0f };

    /// The constant vertex stream (State::constVb): a binding of stride 0 that feeds an attribute a
    /// fixed-function row declares and the draw's layout lacks. Bytes 0..3 are a D3DCOLOR of opaque
    /// white -- D3D's diffuse for a vertex without one -- and the rest zero, for a missing normal or
    /// texture coordinate (D3D's default for an input the declaration does not carry).
    constexpr uint32_t kConstVbBytes      = 32;
    constexpr uint32_t kConstWhiteOffset  = 0;
    constexpr uint32_t kConstZeroOffset   = 16;

    /// The most translated pairs refused for an interface reason (an input the format lacks, a
    /// read the vertex module does not write) that get a line each.
    constexpr uint32_t kMaxPairRefusals = 64;

    /// Bits of CGxTex+0x2C a sampler depends on: filter (0-2), wrap U/V (3-4), anisotropy (9-13).
    /// Everything else in the word is not sampler state and must not fragment the cache.
    constexpr uint32_t kSamplerKeyMask = 0x3E1F;

    /// A sampler key's world variant (Samplers.cpp): a bit above the engine's flags, which the draw
    /// path sets for a world draw's mipmapped texture (RecordInputs.cpp). The sampler then takes the
    /// device's own anisotropy and mip LOD bias (WXL_VK_ANISOTROPY, WXL_VK_MIP_BIAS).
    constexpr uint32_t kSamplerWorldBit = 0x8000u;
    static_assert((kSamplerWorldBit & kSamplerKeyMask) == 0, "the world bit is not one of the engine's");

    /// A draw of the world: its viewport's depth range ends below this (the world's is 0 .. 0.94). The
    /// interface, the text and the engine's full-screen passes draw over the whole range -- the rule
    /// the jitter follows (src/native/render/Jitter.cpp).
    constexpr float kWorldDepthMax = 0.999f;

    /// The direct table from a masked key to its sampler (State::samplerByKey): the eleven key bits
    /// packed into an index, so the draw path's lookup is one load and no scan.
    constexpr uint32_t kSamplerSlots = 2048;
    constexpr uint32_t SamplerSlot(uint32_t key)
    {
        return (key & 0x1Fu) | ((key >> 4) & 0x3E0u) | ((key >> 5) & 0x400u);
    }
    static_assert(SamplerSlot(kSamplerKeyMask | kSamplerWorldBit) < kSamplerSlots, "every masked key has a slot");

    /// Push-constant range: the four vec4 registers ui.vert reads, c0..c3 -- the first 64 bytes of
    /// Cmd::vertexConst. The static_assert ties the shader's declaration to the Cmd's field: the
    /// field may grow, the push stays the UI's four rows at its front.
    constexpr uint32_t kPushBytes = 64;
    static_assert(sizeof(Cmd::vertexConst) >= kPushBytes, "ui.vert pushes c0..c3 of vertexConst");
    static_assert(sizeof(Cmd::vertexConst[0]) == 16, "one register is one vec4");

    // --- the uniform ring --------------------------------------------------------------------
    //
    // Every pair after the UI's reads its constants from one std140 block,
    //
    //   layout(std140, set = 1, binding = 0) uniform Regs {
    //       vec4 p[16];      // pixel  c0..c15, exactly Cmd::pixelConst
    //       vec4 misc;       // misc.x = alphaRef / 255.0 (0 = test off); .yzw the fixed-function selectors
    //       vec4 v[VCONST];  // vertex c0..c(VCONST-1): Cmd::vertexConst, then Cmd::vertexConstHigh
    //   } regs;
    //
    // p and misc FIRST, in that order, so that a pixel module -- which always declares v[64] --
    // reads them at the same offset whatever the vertex module's class (docs/architecture.md,
    // "Shaders"). VCONST is 64 for the hand-written pairs
    // and for every translated vertex module that neither indexes the bone palette nor reads the
    // shadow matrices, 256 for the ones that do; the manifest says which, and the Cmd carries the
    // answer as `vconst`. So a slice is 17 + 64 = 81 rows (1296 bytes) or 17 + 256 = 273 rows
    // (4368 bytes), before alignment.
    //
    // Per frame-in-flight slot there is one host-visible buffer of kRingBytes, byte-addressed,
    // and one descriptor set whose dynamic uniform binding points at that buffer with range = the
    // larger class. Record() writes the Cmd's rows into the slot's next aligned slice and binds
    // set 1 with the slice's byte offset as the dynamic offset. No allocation and no descriptor
    // write per draw; the alignment is queried once, in Init().

    /// Rows and bytes of the block's head: p[16] then misc.
    constexpr uint32_t kHeadRows  = Cmd::kPixelRegs + 1;
    constexpr uint32_t kHeadBytes = kHeadRows * 16;
    static_assert(sizeof(Cmd::pixelConst) == Cmd::kPixelRegs * 16, "regs.p[] is Cmd::pixelConst");
    static_assert(kHeadBytes == 272, "the GLSL block's head is 17 vec4s");

    /// Bytes of each slice class before alignment.
    constexpr uint32_t kSliceBytes64  = kHeadBytes + Cmd::kVertexRegs * 16;
    constexpr uint32_t kSliceBytes256 = kHeadBytes + Cmd::kVertexRegsHigh * 16;
    static_assert(sizeof(Cmd::vertexConst) == Cmd::kVertexRegs * 16, "regs.v[0..63] is Cmd::vertexConst");
    static_assert(kSliceBytes64 == 1296 && kSliceBytes256 == 4368, "the two slice classes");

    /// The ring's descriptor range: the larger class, so one descriptor serves both. A slice of
    /// either class therefore needs this much room ahead of the cursor, since the bind's valid
    /// usage is offset + range <= the buffer's size.
    constexpr uint32_t kRingRange = kSliceBytes256;

    /// Bytes per slot. The cost is address space, since a host-visible buffer is mapped for its
    /// life: 8 MiB per slot is one of src/vulkan/Memory.cpp's host-visible blocks, and holds 5461
    /// class-64 slices at the 256-byte alignment NVIDIA reports (1536 each), above a busy frame's
    /// ~4000 draws, or 1820 class-256 ones (4608 each). Identical consecutive slices are bound
    /// again rather than written (State::LastSlice), so a frame rarely needs all of it.
    constexpr uint32_t kRingBytes = 8u * 1024u * 1024u;

    /// How many more ring buffers, and how many more descriptor pools, one frame slot may spill
    /// into when a frame outgrows the first. Every draw of a frame is recorded as it arrives, so an
    /// in-world frame -- several thousand draws, about a third of them in the 256-row class -- would
    /// otherwise skip its tail. Made on first use, kept for the session, rewound with the slot.
    constexpr uint32_t kRingSpill = 3;
    constexpr uint32_t kPoolSpill = 3;

    // --- the bone-palette ring ----------------------------------------------------------------
    //
    // A 256-row vertex module's `bones` variant (manifest.hpp) reads every row past c63 from
    //
    //   layout(std430, set = 2, binding = 0) readonly buffer Bones { vec4 b[225]; } bones;  // b[i] = c(31 + i)
    //
    // and declares the 64-row uniform block, so its draws take the 64-row slice. Per frame slot one
    // persistently mapped storage buffer holds the palettes: Record() writes rows c31.. (as many as
    // the engine has ever written) at the cursor when the draw's palette serial differs from the
    // last one the slot wrote, and every draw until the next change binds that offset. The range is
    // the whole b[] (kBoneRange), so a write needs that much room ahead of the cursor; a frame that
    // runs the ring dry draws its remaining 256-row draws through the 256-row modules (the key's
    // `bones` bit is 0 then). 4 MiB per slot holds about 1100 palette changes a frame.
    constexpr uint32_t kBoneRingBytes = 4u * 1024u * 1024u;
    constexpr uint32_t kBoneRange     = manifest::kBoneRows * 16u;
    static_assert(Cmd::kPaletteBase == manifest::kBoneBase, "the reader's palette base is the variant's b[0]");

    /// D3DRS_DEPTHBIAS is in depth-range units (offset = bias, added to z); Vulkan's constant
    /// factor is in units of r, the minimum resolvable difference of the attachment. For
    /// D24_UNORM r is 2^-24 everywhere; for D32_SFLOAT it is 2^(e - 23) with e the exponent of the
    /// primitive's largest z, which is 2^-24 for every z in [0.5, 1) -- where a perspective
    /// scene's depth lives. So one factor serves both formats: constant = bias * 2^24. Closer than
    /// z = 0.5 a float attachment applies half the bias per binade, which is the direction in which
    /// D32 has the precision to spare.
    constexpr float kDepthBiasUnits = 16777216.0f;

    /**
     * One buildable (vertex shader, pixel shader, vertex format) triple: the engine's names as the
     * Cmd carries them, the SPIR-V stems src/vulkan/Shaders.cpp loads, and which constant path the
     * pair reads. A new pair is a new row.
     */
    struct ShaderRow
    {
        const char* vs;       ///< rs(77)->name, e.g. "Color_T1:0"
        const char* ps;       ///< rs(78)->name, e.g. "Combiners_Mod:4"
        uint32_t    format;   ///< EGxVertexBufferFormat the pair is declared for
        const char* vertSpv;  ///< stem under shaders/, without the extension
        const char* fragSpv;
        bool        usesRing; ///< true: set 1 (the ring); false: the UI's push range
        uint32_t    samplers; ///< bit N: the fragment shader samples set 0 binding N; a draw without that texture is skipped (0: no sampler at all)
        uint32_t    ffTextured = 0; ///< matches only keys with Key::ffTextured equal (the textured format-7 build)
    };

    /// The sampler-slot set most hand-written rows' fragment shaders are declared against: s0
    /// alone, at binding 0. The colour-only fixed-function row declares no sampler but is built
    /// on this layout with the binding left unwritten, which is legal while nothing uses it; the
    /// two-stage fixed-function rows declare s0 and s1.
    constexpr uint32_t kRowSamplerMask  = 0x1u;
    constexpr uint32_t kRowSamplerMask2 = 0x3u;

    /// The hand-written set. The UI pairs keep the push-constant path they shipped with; the two
    /// particle pairs (Effect(Particle_Unlit) and Effect(Particle) in Particle.wfx) read the ring,
    /// and so do the fixed-function rows: both shader names empty, one row per layout the engine
    /// draws that way (cgxdevice 6.7, and the M2 models' fixed-function fallback), the reader's
    /// register file in the slice's vertex rows and the stage-0 ops in misc. Format 7 (PC) has no
    /// texcoord: its untextured draws take the row whose fragment shader declares no sampler, its
    /// textured ones (the ground decals, whose TexGen makes both stages' coordinates;
    /// Key::ffTextured) the two-sampler row built on ff_pc2_vert; the two-UV formats 6 and 9 carry
    /// texture stage 1 and declare a second sampler; the formats with a normal, 4 and 6, are the lit
    /// builds. All read the 64-row class.
    inline constexpr ShaderRow kShaderRows[] = {
        { "UI:0",          "UI",              8,  "ui_vert",             "ui_frag",            false, kRowSamplerMask  },
        { "UI:0",          "Desaturate",      8,  "ui_vert",             "desaturate_frag",    false, kRowSamplerMask  },
        { "Color_T1:0",    "Combiners_Mod:4", 8,  "particle_unlit_vert", "combiners_mod_frag", true,  kRowSamplerMask  },
        { "CDiffuse_T1:1", "Combiners_Mod:4", 4,  "particle_lit_vert",   "combiners_mod_frag", true,  kRowSamplerMask  },
        { "",              "",                7,  "ff_pc_vert",          "ff_color_frag",      true,  0u               },
        { "",              "",                7,  "ff_pc2_vert",         "ff_frag2",           true,  kRowSamplerMask2, 1u },
        { "",              "",                8,  "ff_pct_vert",         "ff_frag",            true,  kRowSamplerMask  },
        { "",              "",                10, "ff_pt_vert",          "ff_frag",            true,  kRowSamplerMask  },
        { "",              "",                4,  "ff_pnct_vert",        "ff_lit_frag",        true,  kRowSamplerMask  },
        { "",              "",                6,  "ff_pnct2_vert",       "ff_lit2_frag",       true,  kRowSamplerMask2 },
        { "",              "",                9,  "ff_pct2_vert",        "ff_frag2",           true,  kRowSamplerMask2 },
    };
    constexpr uint32_t kShaderRowCount = sizeof kShaderRows / sizeof kShaderRows[0];

    /**
     * Everything about a draw that selects a pipeline. Compared byte for byte (SameKey), which is
     * why every field is a 4-byte integer or a 4-byte-multiple char array (no padding) and why
     * MakeKey() writes every byte of it (no stale bytes in the unused tail of a name).
     */
    struct Key
    {
        char     vs[Cmd::kNameChars];
        char     ps[Cmd::kNameChars];
        uint32_t vertexFormat;
        uint32_t blend;
        uint32_t depthTest;
        uint32_t depthFunc;
        uint32_t depthWrite;
        uint32_t cull;
        uint32_t colorWrite;
        uint32_t topology;
        // The custom layout itself, when vertexFormat is 14 (zero otherwise). Two custom
        // layouts share the format number and nothing else, and the vertex input is baked into
        // the pipeline, so the declaration is the key: the whole of it rather than a hash, so a
        // collision cannot hand a draw a pipeline laid out for another vertex. Each element is
        // Cmd::VertexElement packed as attrib | format << 8 | offset << 16.
        uint32_t customStride;
        uint32_t customCount;
        uint32_t custom[Cmd::kMaxCustomElements];
        // The depth bias is dynamic state -- its value changes per decal and per shadow caster, a
        // key per value would be a pipeline per value -- but its ENABLE is pipeline state, so the
        // key carries "this draw has one". The strides of a multi-stream custom layout's extra
        // streams, because a 1.0 pipeline bakes every binding's stride; each element's stream is in
        // its packed word (bits 24..31).
        uint32_t depthBias;
        uint32_t streamStride[Cmd::kMaxStreams - 1];
        // 1 for a fixed-function format-7 draw whose stage 0 has a texture (its ColorOp is an
        // EGxTexOp and the resolver gave an image): D3D samples it with the coordinate the stage's
        // TexGen makes, so the row with samplers draws it; 0 for every other key.
        uint32_t ffTextured;
        // Bit 0: a 256-row draw that may take the vertex module's bone-palette variant (the
        // palette storage ring of its frame slot has room or already holds the draw's palette).
        // The build takes the variant when the manifest has one, the 256-row module otherwise,
        // so the bit selects a path and never makes a key unbuildable.
        // Bit 1: a tree-wind draw (src/vulkan/Wind.hpp) that takes the vertex module's `wind`
        // variant when it has one; the same rule: a path, never an unbuildable key.
        // Bit 2: an opaque world draw on a G-buffer pass class (depth tested and written, opaque or
        // alpha-keyed, colour written, a shader pair) that takes the modules' `gbuf` variants when
        // both have one and writes the pass's three extra attachments; the same rule again.
        uint32_t bones;
    };
    static_assert(Cmd::kNameChars % 4 == 0, "the names must keep the key padding-free");
    static_assert(sizeof(Key) == 2 * Cmd::kNameChars + (13 + Cmd::kMaxCustomElements + Cmd::kMaxStreams - 1) * 4,
                  "Key must be padding-free for the byte compare");
    static_assert(sizeof(Key) % 16 == 0, "SameKey compares the key in 16-byte rows");

    /// The stem of ff.vert built with WXL_FF_SEMANTIC_VARYINGS (shared.cmake): the fixed-function
    /// vertex stage that pairs with any translated pixel module.
    constexpr const char* kFfSemanticVertStem = "ff_sem_vert";

    /// What that stage writes, as a mask of manifest::Semantic: colour 0 (white when the vertex
    /// has none), texcoord 0..7 (zero for a set the vertex lacks), fog 0. Not colour 1: it is
    /// unlit, and D3D's specular is a vertex-lighting product these draws never ask for.
    constexpr uint32_t kFfSemanticOutputs = (1u << manifest::kColor0) | (0xFFu << manifest::kTexcoord0) | (1u << manifest::kFog0);

    /// One set-0 layout (a set of sampler slots) and the pipeline layout over it and set 1.
    struct LayoutEntry
    {
        uint32_t              mask     = 0;              ///< bit N: binding N, texture stage N
        VkDescriptorSetLayout set0     = VK_NULL_HANDLE;
        VkPipelineLayout      pipeline = VK_NULL_HANDLE;
        VkPipelineLayout      bones    = VK_NULL_HANDLE; ///< set 0, set 1 and the palette's set 2; made on first use
        VkPipelineLayout      wind     = VK_NULL_HANDLE; ///< set 0, set 1, an empty set 2 and tree wind's set 3; made on first use
        VkPipelineLayout      windBones = VK_NULL_HANDLE; ///< set 0, set 1, the palette's set 2 and tree wind's set 3; made on first use
        /// The shadow replay's tree-wind layouts: as `wind` / `windBones` with an 80-byte push range
        /// (the wind's values, then the light's matrix); made on first use (src/capture/).
        VkPipelineLayout      windShadow      = VK_NULL_HANDLE;
        VkPipelineLayout      windShadowBones = VK_NULL_HANDLE;
    };

    /// One pipeline, for one key, built against one render pass.
    struct PipelineEntry
    {
        Key          key       = {};
        VkRenderPass pass      = VK_NULL_HANDLE;
        VkPipeline   pipeline  = VK_NULL_HANDLE;
        uint32_t     layout    = 0;     ///< index into the layout table
        uint32_t     writeMask = 0;     ///< sampler slots Record() must fill from Cmd::texture[]
        uint32_t     cubeMask  = 0;     ///< of writeMask, the slots declared `samplerCube` (0 for a row)
        uint32_t     stride    = 0;     ///< the vertex stride the pipeline baked in
        uint32_t     vconst    = Cmd::kVertexRegs; ///< the slice class: 64 or 256 vertex rows
        bool         usesRing  = false; ///< copied from the row, so Record() need not look it up
        bool         translated = false; ///< built from the manifest's modules, not a row
        bool         ffVertex   = false; ///< built from the semantic ff.vert and a translated pixel module
        bool         companion  = false; ///< of `translated`: the pixel module is the vertex record's generated ff companion
        bool         pointList  = false; ///< the vertex record is flagged point_size: built as VK_PRIMITIVE_TOPOLOGY_POINT_LIST whatever the key's prim
        bool         dynamicBias  = false; ///< depthBiasEnable and VK_DYNAMIC_STATE_DEPTH_BIAS: Record() sets the Cmd's bias
        bool         dynamicBlend = false; ///< blend mode 11 and VK_DYNAMIC_STATE_BLEND_CONSTANTS: Record() sets the Cmd's factor
        uint32_t     streams      = 1;     ///< vertex bindings the pipeline declares (1 + the Cmd's extraStreams)
        bool         bones        = false; ///< built from the vertex module's bone-palette variant: 64-row slice, palette in set 2
        bool         wind         = false; ///< built from the vertex module's tree-wind variant: set 3 and the push range
        bool         gbuf         = false; ///< built from the modules' G-buffer variants (3.Tools/gbuffer): the pass's three extra attachments written
        bool         gbufNull     = false; ///< a depth-writing pipeline on a G-buffer pass WITHOUT the variants: its pixels leave no record (Build)
        bool         constStream  = false; ///< a stride-0 binding at index `streams` feeds the attributes the layout lacks (State::constVb)
        bool         prewarmed    = false; ///< built at bring-up from the persisted key list, not yet checked against a draw (VerifyPrewarmed)
        const char*  why          = nullptr; ///< a failed entry: why the build failed (static text)
        const ShaderRow* row      = nullptr; ///< the hand-written row, when that was the source
        const manifest::Record* vrec = nullptr; ///< the translated vertex record, for the log
        const manifest::Record* prec = nullptr; ///< the translated pixel record, for the log
    };

    /// One sampler, for one masked texFlags value.
    struct SamplerEntry
    {
        uint32_t  key     = 0;
        VkSampler sampler = VK_NULL_HANDLE;
    };

    /// One vertex format as a Vulkan vertex input: the stride a 1.0 pipeline bakes in (there is no
    /// dynamic stride without VK_EXT_extended_dynamic_state) and the attributes at their offsets.
    struct VertexLayout
    {
        uint32_t                          stride = 0;
        uint32_t                          count  = 0;
        VkVertexInputAttributeDescription attrs[16] = {}; ///< the semantic ff.vert declares ten; a fixed format has at most six
    };

    /// A pair refused for its interface, remembered so the line is written once.
    struct PairRefusal
    {
        char vs[Cmd::kNameChars];
        char ps[Cmd::kNameChars];
        uint32_t format;
    };

    /// Everything this file owns, in one place so Teardown() cannot miss a field.
    ///
    /// As everywhere in src/vulkan/: VkDevice and VkCommandBuffer are dispatchable and therefore
    /// pointers; every other handle here is uint64_t on this 32-bit build and is compared against
    /// VK_NULL_HANDLE, never nullptr.
    struct State
    {
        VkDevice              device = nullptr;
        const loader::Device* dapi   = nullptr;

        VkDescriptorSetLayout regsLayout = VK_NULL_HANDLE; ///< set 1: the ring slice
        LayoutEntry           layouts[kMaxLayouts];        ///< set 0 and the pipeline layouts, by sampler mask
        uint32_t              layoutCount = 0;
        VkDescriptorPool      pools[wxlvk::vk::pipelines::kFrameSlots] = {};

        // The ring: one buffer, one set and one byte cursor per slot; the alignment is the device's.
        VkDescriptorPool regsPool                                     = VK_NULL_HANDLE;
        VkDescriptorSet  regsSets[wxlvk::vk::pipelines::kFrameSlots]  = {};
        memory::Buffer   ring[wxlvk::vk::pipelines::kFrameSlots];
        uint32_t         ringCursor[wxlvk::vk::pipelines::kFrameSlots] = {};
        uint32_t         ringAlign                                    = 1;
        uint32_t         padded64                                     = kSliceBytes64;  ///< a class-64 slice at ringAlign
        uint32_t         padded256                                    = kSliceBytes256; ///< a class-256 slice at ringAlign

        // What a frame spills into once its ring buffer or its descriptor pool
        // is full (kRingSpill / kPoolSpill). Index 0 of ringActive / poolActive is the slot's own.
        memory::Buffer   ringSpill[wxlvk::vk::pipelines::kFrameSlots][kRingSpill];
        VkDescriptorSet  ringSpillSets[wxlvk::vk::pipelines::kFrameSlots][kRingSpill] = {};
        VkDescriptorPool ringSpillPool                                = VK_NULL_HANDLE;
        uint32_t         ringActive[wxlvk::vk::pipelines::kFrameSlots] = {};
        VkDescriptorPool poolSpill[wxlvk::vk::pipelines::kFrameSlots][kPoolSpill] = {};
        uint32_t         poolActive[wxlvk::vk::pipelines::kFrameSlots] = {};

        // The bone-palette ring (kBoneRingBytes): set 2's layout, one buffer, one set and one
        // cursor per slot, and the palette each slot wrote last. bonesReady is false when any of it
        // could not be made; every 256-row draw then takes the 256-row modules.
        bool                  bonesReady  = false;
        VkDescriptorSetLayout bonesLayout = VK_NULL_HANDLE;
        VkDescriptorPool      bonesPool   = VK_NULL_HANDLE;
        VkDescriptorSet       bonesSets[wxlvk::vk::pipelines::kFrameSlots] = {};
        memory::Buffer        boneRing[wxlvk::vk::pipelines::kFrameSlots];
        uint32_t              boneCursor[wxlvk::vk::pipelines::kFrameSlots] = {};
        uint32_t              boneAlign   = 1;
        struct LastPalette
        {
            bool     valid  = false;
            uint64_t serial = 0;
            uint32_t count  = 0;
            uint32_t offset = 0;
        };
        LastPalette lastPalette[wxlvk::vk::pipelines::kFrameSlots];

        PipelineEntry pipelines[kMaxPipelines] = {};
        uint32_t      pipelineCount           = 0;
        uint32_t      pipelineLast            = 0; ///< the entry the previous draw used, tried first (runs of draws share a key)
        uint32_t      translatedCount         = 0;
        uint32_t      companionCount          = 0; ///< of translatedCount, built with an ff companion

        SamplerEntry samplers[kMaxSamplers] = {};
        uint32_t     samplerCount           = 0;
        float        anisoMax               = 0.0f; ///< maxSamplerAnisotropy with the feature enabled, 0 without
        float        worldAniso             = 0.0f; ///< the world variant's anisotropy (WXL_VK_ANISOTROPY within anisoMax), 0 none
        float        worldBias              = 0.0f; ///< the world variant's mip LOD bias (WXL_VK_MIP_BIAS)
        bool         worldFilter            = false; ///< either is on: the draw path keys world samplers
        VkSampler    samplerByKey[kSamplerSlots] = {}; ///< the draw path's lookup: by SamplerSlot(key), VK_NULL_HANDLE until made

        Key      skipped[kMaxSkippedKeys] = {};
        uint32_t skippedCount             = 0;

        /// The census of unsupported draws (CountUnsupported, printed by LogUnsupported): each
        /// distinct key with the reason, the draws it cost, and what the first such draw carried
        /// beyond the key.
        struct Unsupported
        {
            Key         key;
            const char* why;
            uint64_t    draws;
            uint32_t    bucket;   ///< SkipBucket
            uint32_t    ff;       ///< colour op | alpha op << 8 | colour mode << 16 | 1 << 24 for a fixed-function vertex stage
            uint32_t    textures; ///< bit N: stage N had an image
            uint32_t    engine;   ///< bit N: the engine had a texture bound on stage N (Cmd::engineTexMask)
            uint32_t    samplers; ///< the pipeline's sampler slots (writeMask); 0xFFFFFFFF: no pipeline
            uint32_t    cubes;    ///< of samplers, the samplerCube ones
            int32_t     slot;     ///< the slot the skip was decided at, -1 when none
            uint32_t    detail;   ///< not ready: the image's VkImageLayout; sampler type: 1 when the image is a cube
            uint32_t    source;   ///< 0 row, 1 modules, 2 ff vertex + module, 3 module + ff companion, 4 none
        };
        Unsupported unsupported[kMaxUnsupported] = {};
        uint32_t    unsupportedCount             = 0;
        uint64_t    unsupportedOther[kSkipBuckets] = {}; ///< skips of pairs past the table, per bucket
        bool        census                       = false; ///< settings::StatsWanted(), captured at Init
        /// The record split (WXL_VK_RECORD_SPLIT, Record.cpp): time-stamp-counter ticks per phase, the draws timed.
        bool        split                        = false;
        uint64_t    splitTicks[6]                = {};
        uint64_t    splitDraws                   = 0;
        uint64_t    splitTsc0                    = 0;   ///< the counters at Init: the time-stamp counter calibrated
        int64_t     splitQpc0                    = 0;   ///< against the performance counter over the session
        /// The last capture's model bound for a bone-palette draw (Record.cpp Capture): a model's batches share one
        /// palette (its serial and count), so its bound is the same; computed again only when they move.
        uint64_t    boundSerial                  = 0;
        uint32_t    boundCount                   = 0;
        bool        boundValid                   = false;
        float       boundValue[4]                = {};

        /// The constant vertex stream (kConstVbBytes); VK_NULL_HANDLE when it could not be made,
        /// and then a row that needs it is not built.
        memory::Buffer constVb;

        /// The idle-stage image (kFillerColour); handle VK_NULL_HANDLE when it could not be made,
        /// and then a fixed-function draw with a disabled stage 0 on a sampling row is skipped.
        texture::Image filler;

        PairRefusal refusals[kMaxPairRefusals] = {};
        uint32_t    refusalCount               = 0;

        unsigned mode = settings::kTranslatedFill; ///< settings::TranslatedMode(), captured at Init

        uint32_t pipeHash[kPipeHashSlots] = {};   ///< index over `pipelines`
        uint64_t buildTicks = 0;                  ///< QPC ticks spent building pipelines
        uint64_t buildCount = 0;
        uint32_t failedCount = 0;                 ///< entries remembering a failed build

        /// The persisted VkPipelineCache every build goes through (VK_NULL_HANDLE when it could not
        /// be created; pipelines are then built without one).
        VkPipelineCache cache           = VK_NULL_HANDLE;
        size_t          cacheLoaded     = 0;   ///< bytes of a valid file handed to the create
        uint64_t        cacheSavedAt    = 0;   ///< buildCount at the last save (or the load)
        const char*     cacheRejected   = nullptr; ///< why the file was not used, for the line

        /// The render passes Present.cpp made, with their formats (PassCreated): what the persisted
        /// key list names a pass by, since a pass handle does not outlive the process.
        struct PassInfo
        {
            VkRenderPass pass   = VK_NULL_HANDLE;
            uint32_t     colour = 0;
            uint32_t     depth  = 0;
            uint32_t     gbuffer = 0;   ///< the class carries the G-buffer's three attachments: four colour attachments in all
        };
        PassInfo passInfo[12];
        uint32_t passInfoCount = 0;
        uint64_t keysSavedAt   = 0;   ///< buildCount when pipeline-keys.bin was last written

        /// The slice the previous ring draw of each slot wrote, so a draw whose rows
        /// are byte for byte the same binds that slice again instead of writing a new one. `data`
        /// is a cached CPU copy (the ring itself is write-combined; reading it back is slow).
        struct LastSlice
        {
            bool     valid    = false;
            uint32_t offset   = 0;
            uint32_t ring     = 0;   ///< ringActive when it was written
            uint32_t vconst   = 0;
            uint32_t highRows = 0;
        };
        LastSlice lastSlice[wxlvk::vk::pipelines::kFrameSlots];
        float     lastSliceData[wxlvk::vk::pipelines::kFrameSlots][kRingRange / 16][4] = {};

        /// What the command buffer being recorded has bound through Record(), so a draw that binds
        /// what the previous one bound records nothing for it. Forgotten whenever something else may
        /// have bound into the buffer or the buffer may be new: InvalidateBindings() (every pass
        /// begin, the overlay) and a different command buffer handle.
        struct Bound
        {
            VkCommandBuffer  cmd       = nullptr;
            VkPipeline       pipeline  = VK_NULL_HANDLE;
            VkPipelineLayout layout    = VK_NULL_HANDLE;
            VkDescriptorSet  set0      = VK_NULL_HANDLE;
            VkDescriptorSet  set1      = VK_NULL_HANDLE;   ///< valid with set1Offset while not null
            uint32_t         set1Offset = 0;
            VkDescriptorSet  set2      = VK_NULL_HANDLE;   ///< the palette ring, valid with set2Offset while not null
            uint32_t         set2Offset = 0;
            bool             viewport  = false;
            bool             scissor   = false;
            bool             bias      = false;
            bool             blend     = false;
            bool             push      = false;
            VkViewport       viewportValue = {};
            VkRect2D         scissorValue  = {};
            float            biasValue     = 0.0f;
            float            blendValue    = 0.0f;
            float            pushValue[16] = {};
            uint32_t         vbCount   = 0;
            VkBuffer         vbs[Cmd::kMaxStreams] = {};
            VkBuffer         ib        = VK_NULL_HANDLE;
        };
        Bound bound;

        wxlvk::vk::pipelines::Stats stats = {};

        // Once-only log flags for the conditions that would otherwise repeat per draw.
        bool loggedNotReady         = false;
        bool loggedBadSlot          = false;
        bool loggedPoolReset        = false;
        bool loggedExhausted        = false;
        bool loggedUniformExhausted = false;
        bool loggedPipelinesFull    = false;
        bool loggedSamplersFull     = false;
        bool loggedSkippedFull      = false;
        bool loggedRefusalsFull     = false;
        bool loggedLayoutsFull      = false;
        bool loggedBadStride        = false;
        bool loggedBadViewport      = false;
        bool loggedNoBuffers        = false;
        bool loggedSamplerFailed    = false;
        bool loggedClassMismatch    = false;
        bool loggedNoHighRows       = false;
        bool loggedPointPrim        = false; ///< a point_size vertex record arrived on a prim other than 0
        bool loggedSpillFailed      = false; ///< a ring or pool spill could not be created

        bool ready = false;
    };
    extern State s;

    /// One frame slot's descriptor-set cache (kSetCacheSlots). An entry is live while its `gen` is
    /// the cache's: a flush (BeginFrame(), when the cache or the slot's pool is nearly full) bumps
    /// the generation instead of clearing the table. Outside State because State is rebuilt by
    /// assignment from a temporary, and these are 350 KiB per slot.
    struct SetEntry
    {
        VkDescriptorSet set   = VK_NULL_HANDLE;
        uint32_t        gen   = 0;
        uint32_t        hash  = 0;
        uint32_t        at    = 0;   ///< first key word in the arena
        uint32_t        words = 0;
    };
    struct SetCache
    {
        SetEntry table[kSetCacheSlots];
        uint64_t arena[kSetCacheWords];
        uint32_t used  = 0;          ///< arena words used this frame
        uint32_t count = 0;          ///< entries live this frame
        uint32_t gen   = 1;
    };
    extern SetCache g_setCache[wxlvk::vk::pipelines::kFrameSlots];

    VkPipelineLayout BonesLayoutFor(uint32_t index);
    VkPipelineLayout WindLayoutFor(uint32_t index, bool bones, bool shadow = false);

    // --- the fourteen engine formats, by semantic, for the translated modules -----------------
    //
    // cgxdevice.md 3.6: every fixed EGxVertexBufferFormat is one interleaved stream, and the table
    // there gives each element's byte offset. A translated vertex module declares its inputs by
    // semantic at fixed locations (docs/architecture.md, "Shaders"):
    //
    //   position0 0 R32G32B32_SFLOAT   normal0 1 R32G32B32_SFLOAT   color0 2 B8G8R8A8_UNORM
    //   color1 3 B8G8R8A8_UNORM        texcoord0 4 R32G32_SFLOAT    texcoord1 5 R32G32_SFLOAT
    //   blendweight0 6 R8G8B8A8_UNORM  blendindices0 7 R8G8B8A8_USCALED
    //
    // blendindices as USCALED and not UINT because the bytecode multiplies the index by 3 as a
    // float before mova (the design's trap table); the attribute must arrive as 0..255 in a float.
    // The builder declares only the elements the module's `input` lines name, so a format with an
    // element the module does not read costs nothing, and a module naming an element the format
    // lacks is refused by name.

    /// One element of a format.
    struct Element
    {
        uint8_t  semantic; ///< manifest::Semantic
        uint8_t  offset;   ///< bytes from the vertex's start
        VkFormat format;
        uint8_t  stream = 0; ///< the vertex binding it is read from; 0 for every fixed format
    };

    /// One format: its stride and up to six elements.
    struct FormatDesc
    {
        uint8_t stride;
        uint8_t count;
        Element e[6];
    };

    constexpr VkFormat kF3    = VK_FORMAT_R32G32B32_SFLOAT;
    constexpr VkFormat kF2    = VK_FORMAT_R32G32_SFLOAT;
    constexpr VkFormat kBgra  = VK_FORMAT_B8G8R8A8_UNORM;
    constexpr VkFormat kU4N   = VK_FORMAT_R8G8B8A8_UNORM;
    constexpr VkFormat kU4    = VK_FORMAT_R8G8B8A8_USCALED;

    constexpr uint8_t kP  = manifest::kPosition0;
    constexpr uint8_t kN  = manifest::kNormal0;
    constexpr uint8_t kC0 = manifest::kColor0;
    constexpr uint8_t kC1 = manifest::kColor1;
    constexpr uint8_t kT0 = manifest::kTexcoord0;
    constexpr uint8_t kT1 = manifest::kTexcoord0 + 1;
    constexpr uint8_t kBW = manifest::kBlendWeight0;
    constexpr uint8_t kBI = manifest::kBlendIndices0;

    inline constexpr FormatDesc kFormats[14] = {
        /*  0 P      */ { 12, 1, { { kP, 0, kF3 } } },
        /*  1 PN     */ { 24, 2, { { kP, 0, kF3 }, { kN, 12, kF3 } } },
        /*  2 PNC    */ { 28, 3, { { kP, 0, kF3 }, { kN, 12, kF3 }, { kC0, 24, kBgra } } },
        /*  3 PNT    */ { 32, 3, { { kP, 0, kF3 }, { kN, 12, kF3 }, { kT0, 24, kF2 } } },
        /*  4 PNCT   */ { 36, 4, { { kP, 0, kF3 }, { kN, 12, kF3 }, { kC0, 24, kBgra }, { kT0, 28, kF2 } } },
        /*  5 PNT2   */ { 40, 4, { { kP, 0, kF3 }, { kN, 12, kF3 }, { kT0, 24, kF2 }, { kT1, 32, kF2 } } },
        /*  6 PNCT2  */ { 44, 5, { { kP, 0, kF3 }, { kN, 12, kF3 }, { kC0, 24, kBgra }, { kT0, 28, kF2 }, { kT1, 36, kF2 } } },
        /*  7 PC     */ { 16, 2, { { kP, 0, kF3 }, { kC0, 12, kBgra } } },
        /*  8 PCT    */ { 24, 3, { { kP, 0, kF3 }, { kC0, 12, kBgra }, { kT0, 16, kF2 } } },
        /*  9 PCT2   */ { 32, 4, { { kP, 0, kF3 }, { kC0, 12, kBgra }, { kT0, 16, kF2 }, { kT1, 24, kF2 } } },
        /* 10 PT     */ { 20, 2, { { kP, 0, kF3 }, { kT0, 12, kF2 } } },
        /* 11 PT2    */ { 28, 3, { { kP, 0, kF3 }, { kT0, 12, kF2 }, { kT1, 20, kF2 } } },
        /* 12 PBNT2  */ { 48, 6, { { kP, 0, kF3 }, { kBW, 12, kU4N }, { kBI, 16, kU4 }, { kN, 20, kF3 }, { kT0, 32, kF2 }, { kT1, 40, kF2 } } },
        /* 13 PNC2T2 */ { 48, 6, { { kP, 0, kF3 }, { kN, 12, kF3 }, { kC0, 24, kBgra }, { kC1, 28, kBgra }, { kT0, 32, kF2 }, { kT1, 40, kF2 } } },
    };

    /// What a pipeline is built from, as Source() decides it.
    struct Sources
    {
        const ShaderRow*        row       = nullptr;
        const manifest::Record* vr        = nullptr;
        const manifest::Record* pr        = nullptr;
        bool                    ffVertex  = false;
        bool                    companion = false;
    };

    // State.cpp
    void Fail(const char* what, VkResult r);
    bool HaveEntryPoints(const loader::Device& d);
    void DropLayoutsAndPools(const loader::Device& d, VkDevice device);
    void DropRing();

    // Keys.cpp
    void MakeKey(const Cmd& c, uint32_t frameSlot, bool windy, bool gbuffer, Key& out);
    void SkipUnsupported(const Key& k, const char* why);
    void RefusePair(const Key& k, const char* what, const char* semantic);
    void CountSkip(uint32_t bucket, const Key& k, const Cmd& c, const PipelineEntry* e, int32_t slot, uint32_t detail,
                   const char* why);
    /// The unsupported bucket of the census: no pipeline, or one the draw cannot use.
    void CountUnsupported(const Key& k, const Cmd& c, const char* why, const PipelineEntry* e = nullptr);

    // Encodings.cpp
    bool BlendState(uint32_t blend, VkPipelineColorBlendAttachmentState& a);
    VkColorComponentFlags WriteMask(uint32_t engine);
    bool Topology(uint32_t prim, VkPrimitiveTopology& out);
    bool VertexLayoutFor(uint32_t format, VertexLayout& out);

    // VertexInput.cpp
    bool LocationFor(uint32_t semantic, uint32_t& location);
    bool SemanticOfAttrib(uint32_t attrib, uint32_t& semantic);
    bool FormatOfAttrib(uint32_t format, VkFormat& out);
    bool KeyElements(const Key& k, Element out[Cmd::kMaxCustomElements], uint32_t& count, uint32_t& stride, const char*& why);
    uint32_t SemanticMask(const Element* e, uint32_t count);
    uint32_t RowFormat(const Key& k);
    bool RowVertexLayout(const Key& k, const ShaderRow& row, VertexLayout& out, uint32_t constBinding, bool& needsConst);
    bool FfSemanticVertexLayout(const Key& k, VertexLayout& out, const char*& why);
    bool TranslatedVertexLayout(const Key& k, uint32_t inputMask, VertexLayout& out, uint32_t& missing, const char*& why);
    const ShaderRow* CoverRow(const Key& k);
    const ShaderRow* RowFor(const Key& k);

    // Layouts.cpp
    bool LayoutFor(uint32_t mask, uint32_t& index);

    // Sources.cpp
    bool CullState(uint32_t cull, VkCullModeFlags& mode, VkFrontFace& front, const char*& name);
    bool DepthOp(uint32_t func, VkCompareOp& op, const char*& name);
    const manifest::Record* CompanionFor(const manifest::Record& v);
    bool Source(const Key& k, const ShaderRow*& row, const manifest::Record*& vr, const manifest::Record*& pr, bool& ffVertex, bool& companion, const char*& why);

    // Build.cpp
    bool Build(const Key& k, VkRenderPass pass, PipelineEntry& out, const char*& why, const Sources* forced);

    // Lookup.cpp
    void VerifyPrewarmed(uint32_t i);
    const PipelineEntry* Pipeline(const Key& k, VkRenderPass pass, const char*& failed);

    // Samplers.cpp
    VkSampler Sampler(uint32_t key);

    // Ring.cpp
    void SpillFailed(const char* what, VkResult r);
    bool SpillRing(uint32_t slot);
    bool SpillPool(uint32_t slot);
    bool CreateRing(const context::Handles& h);
    void CreateBones(const context::Handles& h);

    // Cache.cpp
    void LoadPipelineCache(const context::Handles& h);
    void LoadKeys();
    void SaveKeys();
    void Prewarm(VkRenderPass pass, uint32_t colour, uint32_t depth, bool gbuffer);

    // --- one draw's record, in phases (Record.cpp, RecordInputs.cpp) -----------------------------

    /// One section of a ring slice, as the Cmd holds it: a run of 16-byte rows.
    struct SliceSection
    {
        const void* src;
        uint32_t    rows;
    };

    /**
     * What Record() has decided about one draw so far, handed from phase to phase. Every phase
     * returns false for a skipped draw, after counting it; nothing has been recorded into the
     * command buffer before the binds, so a skip is whole.
     *
     * The key, the images, the sections and the misc row carry no initialiser: each is written
     * whole by its phase, and a per-draw zeroing of them would be the `rep stos` the draw path
     * keeps away from its write-combined copies (common/Simd.hpp).
     */
    struct Draft
    {
        uint32_t                   frameSlot = 0;
        Key                        key;
        const PipelineEntry*       entry = nullptr;
        const wxlvk::vk::wind::Draw* windDraw = nullptr;   ///< the windy draw's push values, null otherwise

        // the textures (Textures): one entry per written slot, in slot order
        VkDescriptorImageInfo images[Cmd::kTextureStages];
        uint8_t               bindings[Cmd::kTextureStages];         ///< the slot each entry is for
        uint64_t              setWords[1 + Cmd::kTextureStages];     ///< the set cache's key: [0] the layout and mask (DescriptorSet), then image id | sampler key << 32 per entry
        uint32_t              writeCount = 0;
        bool                  idleUsed   = false;   ///< a binding took the idle-stage image
        bool                  cacheable  = true;    ///< every image has an id: the set may be cached across frames

        // the ring slice (PlanSlice, WriteSlice)
        float        misc[4];
        uint32_t     highRows = 0;
        uint32_t     padded   = 0;                 ///< the slice's size at the ring's alignment
        SliceSection sections[4];
        bool         reuse     = false;            ///< the previous ring draw's slice, bound again
        uint32_t     firstDiff = 0;                ///< the first section that differs from the copy (4: none)
        uint32_t     diffRow   = 0;                ///< its first row in the slice

        // what is bound (DescriptorSet, WriteSlice, WritePalette)
        VkDescriptorSet set           = VK_NULL_HANDLE;
        uint32_t        dynamicOffset = 0;
        uint32_t        paletteOffset = 0;

        Draft() = default;
        Draft(const Draft&) = delete;
        Draft& operator=(const Draft&) = delete;
    };

    // RecordInputs.cpp: what the draw reads, decided and written before anything is bound.
    bool Textures(const Cmd& c, Draft& dr);
    bool PlanSlice(const Cmd& c, Draft& dr);
    bool DescriptorSet(const Cmd& c, Draft& dr);
    void WriteSlice(Draft& dr);
    void WritePalette(const Cmd& c, Draft& dr);

    // --- per-draw helpers, inline: Record() calls them for every draw -----------------------------

    /**
     * Whether two keys are the same draw state: eleven 16-byte compares, not the CRT's memcmp
     * (which MSVC inlines as a 4-byte loop).
     *
     * @param Key& a : one
     * @param Key& b : the other
     * @return bool same
     */
    inline bool SameKey(const Key& a, const Key& b) { return wxlvk::simd::EqualRows(&a, &b, sizeof(Key) / 16); }

    /// An FNV-1a style mix over the key's 32-bit words (padding-free and fully written, see
    /// MakeKey) and the pass handle, folded to a slot. The words are read in place: a Key is
    /// 4-byte aligned and nothing but 4-byte fields.
    inline uint32_t KeyHash(const Key& k, VkRenderPass pass)
    {
        static_assert(sizeof(Key) % 4 == 0, "the key hashes by words");
        uint32_t        h = 2166136261u;
        const uint32_t* w = reinterpret_cast<const uint32_t*>(&k);
        for (uint32_t i = 0; i < sizeof(Key) / 4; ++i) h = (h ^ w[i]) * 16777619u;
        const uint64_t pv = static_cast<uint64_t>(pass);
        h = (h ^ static_cast<uint32_t>(pv)) * 16777619u;
        h = (h ^ static_cast<uint32_t>(pv >> 32)) * 16777619u;
        h ^= h >> 15;
        return h & (kPipeHashSlots - 1);
    }

    /// Whether `pass` is a G-buffer class (PassCreated said so): its pipelines name four colour
    /// attachments, and its opaque world draws may take the `gbuf` variants.
    inline bool IsGBufferPass(VkRenderPass pass)
    {
        for (uint32_t i = 0; i < s.passInfoCount; ++i)
            if (s.passInfo[i].pass == pass) return s.passInfo[i].gbuffer != 0;
        return false;
    }

    /// The ring buffer a slot is writing into now: its own, or the spill it moved to.
    inline memory::Buffer& RingBuffer(uint32_t slot)
    {
        return s.ringActive[slot] ? s.ringSpill[slot][s.ringActive[slot] - 1] : s.ring[slot];
    }

    /// The set-1 descriptor of RingBuffer(slot).
    inline VkDescriptorSet RingSet(uint32_t slot)
    {
        return s.ringActive[slot] ? s.ringSpillSets[slot][s.ringActive[slot] - 1] : s.regsSets[slot];
    }

    /// The descriptor pool a slot allocates set 0 from now.
    inline VkDescriptorPool ActivePool(uint32_t slot)
    {
        return s.poolActive[slot] ? s.poolSpill[slot][s.poolActive[slot] - 1] : s.pools[slot];
    }

    /// Whether a 256-row draw's palette fits this frame slot: already written, or room for one more.
    inline bool PaletteFits(uint32_t slot, const Cmd& c)
    {
        const State::LastPalette& lp = s.lastPalette[slot];
        if (lp.valid && lp.serial == c.paletteSerial && lp.count == c.paletteCount) return true;
        return s.boneCursor[slot] + kBoneRange <= kBoneRingBytes;
    }
}
