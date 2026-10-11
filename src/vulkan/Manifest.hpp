// The translated-module manifest: what the offline DX9 -> SPIR-V emitter produced, as fixed tables.
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

// The client's DX9 shader bytecode is translated offline (3.Tools/bls_tools/translate.py) into one
// SPIR-V module per byte-distinct record, and the DLL finds those modules through a manifest keyed by
// the MD5 of the record's bytes (docs/architecture.md, "Shaders"). This file is the DLL's side of
// that contract: it reads shaders/translated/manifest.txt next to the DLL once, into fixed tables,
// and answers two questions -- "which record has this hash?" (asked once per slot-72 create, by
// src/shaders/Translated.cpp) and "which record carries this name?" (the fallback when no create was
// ever seen for a name).
//
// The manifest's text format, verbatim from the contract with the emitter: blocks separated by a
// blank line, lines of space-separated tokens, '#' to end of line is a comment.
//
//   record <md5> <vertex|pixel> <vs_3_0|ps_3_0|vs_2_0|ps_2_0|ff> <spv path, forward slashes> vconst=<64|256> pconst=16
//   name <Name:perm>               repeatable; also the bare <Name> for record 0 of a container
//   sampler <slot 0..15> <2d|cube> pixel only, repeatable
//   input <semantic> <vN>          vertex only, repeatable
//   output <semantic>              vertex only, repeatable
//   reads <semantic>               pixel only, repeatable
//   flag <fixed_fog|point_size|point_sprite>  optional
//   companion <md5>                vertex only: the `ff` pixel record to bind when the draw's
//                                  pixel shader name is ""
//   bones <spv path>               vertex only, vconst=256: the bone-palette variant of the module,
//                                  reading every constant past c63 from set 2 binding 0 (a storage
//                                  buffer, b[i] = c(31 + i)) and declaring the 64-row uniform block
//   wind <spv path>                vertex only: the tree-wind variant, the position bent in model
//                                  space right before the module's own transform (set 3;
//                                  src/vulkan/Wind.hpp)
//   shadow <spv path>              vertex only: the draw replay's variant (shadowgen.py), the clip
//                                  position re-projected by a push-range matrix: <spv>.shadow.spv
//   shadowbones <spv path>         vertex only, with `bones`: the same of the bone-palette variant
//   shadowwind <spv path>          vertex only, with `wind`: the tree-wind variant's, <wind>.shadow.spv,
//                                  its matrix after the wind's push values (offset 16)
//   shadowwindbones <spv path>     vertex only, with `windbones`: the same of that variant
//   windbones <spv path>           vertex only, with `bones`: the same of the bone-palette variant
//                                  (a pixel record's `wind` is its leaves' half: bound only with a
//                                  vertex `wind` variant, which writes the varying it reads)
//   gbuf <spv path>                the G-buffer variant (3.Tools/gbuffer): a vertex record's, with the
//                                  view normal and the light term as extra varyings, or a pixel
//                                  record's, writing the albedo, normal and light attachments beside
//                                  the colour: <spv>.gbuf.spv
//   gbufbones / gbufwind / gbufwindbones <spv path>   vertex only: the same of its `bones` / `wind` /
//                                  `windbones` variant; a pixel record's `gbufwind` is the same of its
//                                  `wind` half
//
// Profile `ff` is a pixel record the emitter GENERATED rather than
// translated: the fixed-function pixel cascade, in the translated varying interface, for the
// vs_2_0 weather records the engine draws with no pixel shader at all. It has no bytecode (its
// md5 is of its GLSL text) and so is never matched at a create; the vertex record's `companion`
// line is the only way to reach it. Its names are "<vertex name>@ff", never an engine name. The
// flags: `fixed_fog` on a vertex row says it writes fog0 as D3D's oFog, on an ff row that it
// blends toward p[15].rgb while p[15].w > 0; `point_size` on a vertex row says it writes
// gl_PointSize and the draw is a point list; `point_sprite` on an ff row that it samples stage
// 0 at gl_PointCoord and reads no texcoord. The companion blocks come after the vertex rows
// that name them, so the link is resolved once the whole file is read.
//
// Semantics: position0 normal0 color0 color1 texcoord0..texcoord7 blendweight0 blendindices0 fog0
// psize0. They are kept as bit masks, because every consumer asks set questions of them: the
// pipeline builder declares the vertex attributes a module's inputs name and checks the pixel
// module's reads against the vertex module's outputs.
//
// No Vulkan, no engine, no device: this is a parser over a text file and can run at extension
// load, before any of those exist. Fixed tables rather than vectors, as everywhere in src/vulkan/,
// because address space is the scarce resource in this 32-bit process; the bounds are generous
// against the client's archives (1578 records, ~2700 names) and a manifest that exceeds them is truncated
// with a warning rather than refused.
namespace wxlvk::vk::manifest
{
    /// The semantics the contract names, as bit positions. The order is the contract's.
    enum Semantic : uint32_t
    {
        kPosition0     = 0,
        kNormal0       = 1,
        kColor0        = 2,
        kColor1        = 3,
        kTexcoord0     = 4,  ///< texcoord0..texcoord7 are 4..11
        kBlendWeight0  = 12,
        kBlendIndices0 = 13,
        kFog0          = 14,
        kPsize0        = 15,
        kSemanticCount = 16,
    };

    /// Which stage a record is for.
    enum class Kind : uint8_t
    {
        Vertex = 0,
        Pixel  = 1,
    };

    /// The DX9 profile the record was decoded from -- or Ff, the generated fixed-function pixel
    /// stage that decodes from nothing.
    enum class Profile : uint8_t
    {
        Vs30, Ps30, Vs20, Ps20, Ff,
    };

    /// "No record": the value of Record::companion on every row without a `companion` line.
    constexpr uint16_t kNoRecord = 0xFFFF;

    /// The longest `name` token kept, NUL included. The longest engine name is
    /// "Combiners_Opaque_AddAlpha_Alpha:15" (34); draw::Cmd bounds its names the same way.
    constexpr uint32_t kNameChars = 40;

    /// The longest spv path kept, NUL included: "pixel/" + 32 hex + ".spv" is 42.
    constexpr uint32_t kPathChars = 128;   ///< "vertex/<32 hex>.bones.wind.spv" is 54; room to spare

    /// One byte-distinct record the emitter translated.
    struct Record
    {
        char     md5[33];          ///< lowercase hex, NUL-terminated
        char     spv[kPathChars];  ///< relative to shaders/translated/, forward slashes
        Kind     kind;
        Profile  profile;
        uint16_t vconst;           ///< 64 or 256: rows of regs.v[] the vertex module declares
        uint16_t pconst;           ///< 16: rows of regs.p[]
        uint16_t samplerMask;      ///< pixel: bit s set for `sampler s ...`
        uint16_t cubeMask;         ///< pixel: bit s set when that sampler is a cube
        uint32_t inputMask;        ///< vertex: bit Semantic set for each `input`
        uint32_t outputMask;       ///< vertex: bit Semantic set for each `output`
        uint32_t readsMask;        ///< pixel: bit Semantic set for each `reads`
        bool     fixedFog;         ///< `flag fixed_fog`
        bool     pointSize;        ///< `flag point_size`: vertex row writes gl_PointSize; the draw is a point list
        bool     pointSprite;      ///< `flag point_sprite`: ff row samples stage 0 at gl_PointCoord
        uint16_t firstName;        ///< index of its first name entry, or 0xFFFF
        uint16_t nameCount;        ///< how many name entries it owns (not necessarily consecutive)
        uint16_t companion;        ///< vertex: index of its `ff` companion record, or kNoRecord
        char     companionMd5[33]; ///< vertex: the `companion` line's md5 as written, "" when none
        char     bones[kPathChars]; ///< vertex: the `bones` variant's spv path, "" when none
        char     wind[kPathChars];  ///< vertex: the `wind` variant's spv path, "" when none
        char     windBones[kPathChars]; ///< vertex: the `windbones` variant's (of `bones`), "" when none
        /// vertex: a `shadow` line (shadowgen.py) -- `spv` with ".shadow.spv" for ".spv" is the draw
        /// replay's variant; `shadowBones` the same of the `bones` variant. Flags, not paths: the
        /// names follow from the module's own, and the table is static.
        bool     shadow;
        bool     shadowBones;
        /// vertex: `shadowwind` / `shadowwindbones` -- `wind` / `windBones` with ".shadow.spv" for ".spv".
        bool     shadowWind;
        bool     shadowWindBones;
        /// The G-buffer variants (3.Tools/gbuffer): a vertex record's `gbuf` / `gbufbones` / `gbufwind` /
        /// `gbufwindbones` (`spv` / `bones` / `wind` / `windBones` with ".gbuf.spv" for ".spv"), a pixel
        /// record's `gbuf` / `gbufwind` (`spv` / `wind` likewise). Flags: the names follow from the module's own.
        bool     gbuf;
        bool     gbufBones;
        bool     gbufWind;
        bool     gbufWindBones;
    };

    /// The bone-palette variant's storage buffer: b[i] is vertex register c(kBoneBase + i), for
    /// kBoneRows rows (the translator's BONE_BASE / BONE_ROWS).
    constexpr uint32_t kBoneBase = 31;
    constexpr uint32_t kBoneRows = 256 - kBoneBase;

    /// One `name` line: an engine name ("Diffuse_T1:11", or the bare "Diffuse_T1") and its record.
    struct Name
    {
        char     name[kNameChars];
        uint16_t record;
        uint32_t hash; ///< FNV-1a of `name`, compared before the string
    };

    /**
     * Reads shaders/translated/manifest.txt next to the DLL, once. Idempotent: the second call
     * returns the first's answer without touching the file.
     *
     * A missing file is logged as "no manifest" (a warning); with no manifest every lookup below
     * answers null and only the hand-written pipelines draw. A malformed line is
     * skipped and counted; the file is never refused whole for one bad line, because a partial
     * manifest still names the records it does carry.
     *
     * @return bool loaded : true when at least one record was read
     */
    bool Load();

    /// Whether Load() found a manifest with at least one record.
    bool Loaded();

    /**
     * The record with this content hash.
     *
     * @param string md5 : 32 lowercase hex characters
     * @return pointer record : null when unknown or when nothing is loaded
     */
    const Record* ByMd5(const char* md5);

    /**
     * The record carrying this engine name for this stage. Vertex and pixel records can share a
     * name ("UI" is both), which is why the kind is part of the key.
     *
     * @param string name : as CGxShader +0x18 spells it
     * @param Kind kind : the stage
     * @return pointer record : null when no name entry matches or nothing is loaded
     */
    const Record* ByName(const char* name, Kind kind);

    /**
     * The `ff` companion a vertex record names: the generated fixed-function pixel stage to
     * bind when the draw's pixel shader name is "".
     *
     * @param Record& r : a vertex record
     * @return pointer record : its companion, or null when it has none or the md5 named no record
     */
    const Record* Companion(const Record& r);

    /**
     * A generated `ff` pixel stage by kind, for a vertex record with no `companion` line drawn
     * with no pixel shader: D3D runs the same fixed-function stage 0 after any vertex shader.
     *
     * @param bool pointSprite : the point-sprite stage (stage 0 at gl_PointCoord) or the
     *        texcoord0 one (fixed_fog)
     * @return pointer record : the first such record, or null when the manifest has none
     */
    const Record* FfStage(bool pointSprite);

    /**
     * A record's first name, for a log line.
     *
     * @param Record& r : the record
     * @return string name : its first `name` token, or its md5 when it has none
     */
    const char* FirstName(const Record& r);

    /**
     * A semantic's spelling in the contract, for a log line.
     *
     * @param uint32 semantic : a Semantic value
     * @return string name : e.g. "texcoord1"; "?" outside the table
     */
    const char* SemanticName(uint32_t semantic);

    /**
     * FNV-1a of a NUL-terminated string, the hash the name table and src/shaders/Translated.cpp's
     * registry both compare before the string. One definition so the two agree.
     *
     * @param string s : the string
     * @return uint32 hash
     */
    uint32_t HashName(const char* s);

    /// Counts, for the Install() line.
    struct Stats
    {
        uint32_t records;    ///< records read
        uint32_t vertex;     ///< of which vertex
        uint32_t pixel;      ///< of which pixel (the ff companions among them)
        uint32_t ff;         ///< of the pixel records, those of profile `ff` (generated companions)
        uint32_t names;      ///< name entries read
        uint32_t skipped;    ///< lines that did not parse
        uint32_t truncated;  ///< records or names dropped because a table was full
        uint32_t companions; ///< vertex records whose `companion` md5 resolved to an ff record
        uint32_t companionsUnresolved; ///< `companion` lines whose md5 named no record in the file
        uint32_t bones;      ///< vertex records with a `bones` variant
        uint32_t wind;       ///< vertex records with a `wind` variant
        uint32_t shadow;     ///< vertex records with a `shadow` variant (and `shadowbones`)
        uint32_t gbuf;       ///< records with a `gbuf` variant (vertex and pixel)
    };

    /// The counts since Load().
    void GetStats(Stats& out);
}
