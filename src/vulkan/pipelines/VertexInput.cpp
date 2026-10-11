// The translated modules' vertex inputs, the custom layouts and the hand-written rows.
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

namespace wxlvk::vk::pipelines::detail
{
    /**
     * The attribute location a translated module declares for a semantic.
     *
     * @param uint32 semantic : manifest::Semantic
     * @param uint32& location : receives it
     * @return bool ok : false for a semantic no vertex format carries (texcoord2+, fog, psize)
     */
    bool LocationFor(uint32_t semantic, uint32_t& location)
    {
        switch (semantic)
        {
            case manifest::kPosition0:     location = 0; return true;
            case manifest::kNormal0:       location = 1; return true;
            case manifest::kColor0:        location = 2; return true;
            case manifest::kColor1:        location = 3; return true;
            case manifest::kTexcoord0:     location = 4; return true;
            case manifest::kTexcoord0 + 1: location = 5; return true;
            case manifest::kBlendWeight0:  location = 6; return true;
            case manifest::kBlendIndices0: location = 7; return true;
            default:
                // Texcoord 2..7 at 8..13. Outside the emitter's contract (no fixed
                // format carries them, so no translated module names them); the semantic ff.vert
                // declares them, for the custom layouts that carry up to eight sets.
                if (semantic > manifest::kTexcoord0 + 1 && semantic < manifest::kBlendWeight0)
                {
                    location = 8 + (semantic - (manifest::kTexcoord0 + 2));
                    return true;
                }
                return false;
        }
    }

    // --- the custom layout, in the same terms -------------------------------------------------
    //
    // A key whose format is 14 carries its own elements (Key::custom, from Cmd::custom, from the
    // device's attribute slots: cgxdevice 3.4). The functions below put them into the Element
    // form the fixed table uses, so that everything downstream -- the translated input, the
    // semantic ff.vert's input, the hand-written rows' vertex-order input -- is written once
    // against KeyElements() and never asks which kind of format it has.

    /**
     * The manifest semantic of an engine vertex attribute (cgxdevice 3.5, Cmd::VertexElement).
     *
     * @param uint32 attrib : EGxVertexAttrib 0..13
     * @param uint32& semantic : receives the manifest::Semantic
     * @return bool ok : false past 13
     */
    bool SemanticOfAttrib(uint32_t attrib, uint32_t& semantic)
    {
        switch (attrib)
        {
            case 0: semantic = manifest::kPosition0;     return true;
            case 1: semantic = manifest::kBlendWeight0;  return true;
            case 2: semantic = manifest::kBlendIndices0; return true;
            case 3: semantic = manifest::kNormal0;       return true;
            case 4: semantic = manifest::kColor0;        return true;
            case 5: semantic = manifest::kColor1;        return true;
            default:
                if (attrib >= 6 && attrib <= 13)
                {
                    semantic = manifest::kTexcoord0 + (attrib - 6);
                    return true;
                }
                return false;
        }
    }

    /**
     * The Vulkan vertex format of an engine attribute data format (cgxdevice 3.5): what the D3D
     * backend's D3DDECLTYPE for it delivers to a shader, as the fixed table already encodes it
     * (D3DCOLOR as B8G8R8A8_UNORM, UBYTE4 as USCALED for the reason kFormats gives, SHORT2 as two
     * signed shorts expanded to floats).
     *
     * @param uint32 format : 0..6
     * @param VkFormat& out : receives it
     * @return bool ok : false past 6
     */
    bool FormatOfAttrib(uint32_t format, VkFormat& out)
    {
        switch (format)
        {
            case 0: out = VK_FORMAT_B8G8R8A8_UNORM;   return true; // ColorBGRA, D3DCOLOR
            case 1: out = VK_FORMAT_R8G8B8A8_USCALED; return true; // UByte4
            case 2: out = VK_FORMAT_R8G8B8A8_UNORM;   return true; // UByte4N
            case 3: out = VK_FORMAT_R32G32_SFLOAT;    return true; // Float2
            case 4: out = VK_FORMAT_R32G32B32_SFLOAT; return true; // Float3
            case 5: out = VK_FORMAT_R16G16_SSCALED;   return true; // Short2
            case 6: out = VK_FORMAT_R32_SFLOAT;       return true; // Float1
            default: return false;
        }
    }

    /**
     * The elements of a key's vertex layout: the fixed table's for formats 0..13, the key's own
     * for 14.
     *
     * @param Key& k : the key
     * @param Element out[] : receives up to Cmd::kMaxCustomElements elements
     * @param uint32& count : receives how many
     * @param uint32& stride : receives the stride
     * @param string& why : receives the reason on failure
     * @return bool ok
     */
    bool KeyElements(const Key& k, Element out[Cmd::kMaxCustomElements], uint32_t& count, uint32_t& stride,
                     const char*& why)
    {
        static_assert(sizeof(kFormats[0].e) / sizeof(kFormats[0].e[0]) <= Cmd::kMaxCustomElements,
                      "a fixed format must fit the element array");
        if (k.vertexFormat < 14)
        {
            const FormatDesc& f = kFormats[k.vertexFormat];
            for (uint32_t i = 0; i < f.count; ++i) out[i] = f.e[i];
            count  = f.count;
            stride = f.stride;
            return true;
        }
        if (k.vertexFormat != Cmd::kVertexFormatCustom || k.customCount == 0 ||
            k.customCount > Cmd::kMaxCustomElements || k.customStride == 0)
        {
            why = "the vertex format is neither one of the engine's fourteen fixed layouts nor a custom layout with elements";
            return false;
        }
        for (uint32_t i = 0; i < k.customCount; ++i)
        {
            uint32_t semantic = 0;
            VkFormat format   = VK_FORMAT_UNDEFINED;
            if (!SemanticOfAttrib(k.custom[i] & 0xFFu, semantic) || !FormatOfAttrib((k.custom[i] >> 8) & 0xFFu, format))
            {
                why = "the custom layout names an attribute or a data format outside the engine's tables";
                return false;
            }
            out[i].semantic = static_cast<uint8_t>(semantic);
            out[i].offset   = static_cast<uint8_t>((k.custom[i] >> 16) & 0xFFu);
            out[i].format   = format;
            out[i].stream   = static_cast<uint8_t>((k.custom[i] >> 24) & 0xFFu);
            if (out[i].stream >= Cmd::kMaxStreams || (out[i].stream && k.streamStride[out[i].stream - 1] == 0))
            {
                why = "the custom layout names a stream the draw carries no buffer for";
                return false;
            }
        }
        count  = k.customCount;
        stride = k.customStride;
        return true;
    }

    /**
     * The mask of semantics a layout carries, for matching a custom layout's element set against
     * the fixed formats'.
     *
     * @param const Element* e : the elements
     * @param uint32 count : how many
     * @return uint32 mask : bit manifest::Semantic
     */
    uint32_t SemanticMask(const Element* e, uint32_t count)
    {
        uint32_t mask = 0;
        for (uint32_t i = 0; i < count; ++i) mask |= 1u << e[i].semantic;
        return mask;
    }

    /**
     * The format a hand-written row is looked up by: the key's own for a fixed format, and for a
     * custom layout the fixed format whose element SET it equals (by semantic, whatever the
     * offsets and data formats), since the rows are declared per format and a row's shaders care
     * about which elements arrive, not where in the vertex they sit -- the vertex input carries
     * the custom offsets (RowVertexLayout). A custom set no fixed format has matches no row.
     *
     * @param Key& k : the key
     * @return uint32 format : 0..13, or 0xFFFFFFFF when none
     */
    uint32_t RowFormat(const Key& k)
    {
        if (k.vertexFormat != Cmd::kVertexFormatCustom) return k.vertexFormat;
        Element     e[Cmd::kMaxCustomElements];
        uint32_t    count = 0, stride = 0;
        const char* why   = "";
        if (!KeyElements(k, e, count, stride, why)) return 0xFFFFFFFFu;
        const uint32_t mask = SemanticMask(e, count);
        for (uint32_t f = 0; f < 14; ++f)
            if (SemanticMask(kFormats[f].e, kFormats[f].count) == mask) return f;
        return 0xFFFFFFFFu;
    }

    /**
     * The vertex input of a hand-written row: VertexLayoutFor's for a fixed format; for a custom
     * layout, the matched fixed format's vertex-order locations over the custom offsets and data
     * formats, so that the row's shader (compiled for the fixed format) reads each element from
     * where this vertex keeps it.
     *
     * A row that covers the key's layout rather than matching it (CoverRow) declares elements the
     * vertex lacks: each is read from the constant stream at binding `constBinding` (one past the
     * key's own streams), white for a colour and zero for anything else, and `needsConst` says so.
     *
     * @param Key& k : the key
     * @param ShaderRow& row : the row it is drawn with
     * @param VertexLayout& out : receives the layout
     * @param uint32 constBinding : the constant stream's binding, should one be needed
     * @param bool& needsConst : receives whether one is
     * @return bool ok : false when the row's format has no layout here
     */
    bool RowVertexLayout(const Key& k, const ShaderRow& row, VertexLayout& out, uint32_t constBinding,
                         bool& needsConst)
    {
        needsConst = false;
        if (k.vertexFormat != Cmd::kVertexFormatCustom && k.vertexFormat == row.format)
            return VertexLayoutFor(k.vertexFormat, out);
        if (row.format >= 14) return false;
        Element     e[Cmd::kMaxCustomElements];
        uint32_t    count = 0, stride = 0;
        const char* why   = "";
        if (!KeyElements(k, e, count, stride, why)) return false;
        const FormatDesc& fixed = kFormats[row.format];
        out.stride = stride;
        out.count  = 0;
        for (uint32_t i = 0; i < fixed.count; ++i)
        {
            const Element* match = nullptr;
            for (uint32_t j = 0; j < count; ++j)
                if (e[j].semantic == fixed.e[i].semantic) match = &e[j];
            VkVertexInputAttributeDescription& a = out.attrs[out.count++];
            a.location = i;
            if (match)
            {
                a.binding = match->stream;
                a.format  = match->format;
                a.offset  = match->offset;
            }
            else
            {
                a.binding  = constBinding;
                a.format   = fixed.e[i].format;
                a.offset   = fixed.e[i].semantic == manifest::kColor0 ? kConstWhiteOffset : kConstZeroOffset;
                needsConst = true;
            }
        }
        return true;
    }

    /**
     * The vertex input of the semantic ff.vert, over a key's layout.
     *
     * The shader declares every input it may need -- position0, color0, texcoord0..7 at the
     * semantic locations -- and a Vulkan pipeline must describe every input a vertex shader
     * declares, so an element the layout lacks is described too: as a float2 at offset 0, the
     * vertex's first eight bytes, in bounds for any layout with a position. The shader never
     * forwards such an input: kFfSemanticCtl says which sets and whether a colour the vertex
     * carries, and it writes zero (texcoord) or white (colour) for the rest.
     *
     * @param Key& k : the key
     * @param VertexLayout& out : receives the layout
     * @param string& why : receives the reason on failure
     * @return bool ok : false when the layout has no position or is not one this file reads
     */
    bool FfSemanticVertexLayout(const Key& k, VertexLayout& out, const char*& why)
    {
        Element  e[Cmd::kMaxCustomElements];
        uint32_t count = 0, stride = 0;
        if (!KeyElements(k, e, count, stride, why)) return false;

        static constexpr uint32_t kWanted[] = {
            manifest::kPosition0, manifest::kColor0,
            manifest::kTexcoord0, manifest::kTexcoord0 + 1, manifest::kTexcoord0 + 2, manifest::kTexcoord0 + 3,
            manifest::kTexcoord0 + 4, manifest::kTexcoord0 + 5, manifest::kTexcoord0 + 6, manifest::kTexcoord0 + 7,
        };
        static_assert(sizeof kWanted / sizeof kWanted[0] <= sizeof out.attrs / sizeof out.attrs[0],
                      "the semantic ff.vert's inputs must fit a VertexLayout");

        out.stride = stride;
        out.count  = 0;
        for (uint32_t sem : kWanted)
        {
            uint32_t location = 0;
            LocationFor(sem, location); // every semantic in kWanted has one
            const Element* match = nullptr;
            for (uint32_t j = 0; j < count; ++j)
                if (e[j].semantic == sem) match = &e[j];
            if (!match && sem == manifest::kPosition0)
            {
                why = "the layout has no position";
                return false;
            }
            VkVertexInputAttributeDescription& a = out.attrs[out.count++];
            a.location = location;
            a.binding  = match ? match->stream : 0u;
            a.format   = match ? match->format : VK_FORMAT_R32G32_SFLOAT;
            a.offset   = match ? match->offset : 0u;
        }
        return true;
    }

    /**
     * The vertex input of a translated pair: the format's elements the vertex module names, at
     * the semantic's location.
     *
     * @param Key& k : the key, for its format or its custom layout
     * @param uint32 inputMask : the vertex record's inputs
     * @param VertexLayout& out : receives the layout
     * @param uint32& missing : receives the first semantic the format lacks, on failure
     * @param string& why : receives the reason on failure
     * @return bool ok
     */
    bool TranslatedVertexLayout(const Key& k, uint32_t inputMask, VertexLayout& out,
                                uint32_t& missing, const char*& why)
    {
        Element  elements[Cmd::kMaxCustomElements];
        uint32_t count = 0;
        missing = manifest::kPosition0;
        if (!KeyElements(k, elements, count, out.stride, why)) return false;
        out.count = 0;

        for (uint32_t sem = 0; sem < manifest::kSemanticCount; ++sem)
        {
            if (!(inputMask & (1u << sem))) continue;

            uint32_t location = 0;
            if (!LocationFor(sem, location))
            {
                missing = sem;
                why     = "the vertex module reads a semantic no vertex layout carries";
                return false;
            }
            const Element* e = nullptr;
            for (uint32_t i = 0; i < count; ++i)
                if (elements[i].semantic == sem) e = &elements[i];
            if (!e)
            {
                missing = sem;
                why     = "the vertex module reads a semantic the format does not carry";
                return false;
            }
            VkVertexInputAttributeDescription& a = out.attrs[out.count++];
            a.location = location;
            a.binding  = e->stream;
            a.format   = e->format;
            a.offset   = e->offset;
        }
        return true;
    }

    /**
     * The fixed-function row (vs "" / ps "") for a layout no row is declared for: the formats the
     * engine's GxPrimVertexPtr can also produce (0 P, 1 PN, 2 PNC, 3 PNT, 5 PNT2, 11 PT2), the
     * skinned and two-colour ones (12, 13) and a custom set no fixed format has. D3D draws such a
     * vertex through the same fixed-function pipeline as any other, with white for a missing
     * diffuse and zero for a missing coordinate, so the row whose element set is the smallest
     * superset of the vertex's position, normal, colour 0 and first two UV sets draws it, the
     * elements it lacks fed by the constant stream (RowVertexLayout). Blend weights, indices and
     * colour 1 are not inputs of D3D's fixed function here (no vertex blending; the engine's
     * specular is lit, 6.4), so they take no part. A normal keeps the lit rows (4, 6); none of the
     * rows that match a layout exactly is ever replaced, since this runs only when none matched.
     *
     * @param Key& k : the key, vs and ps both ""
     * @return pointer row : the row, or null when the layout has no position or no row covers it
     */
    const ShaderRow* CoverRow(const Key& k)
    {
        Element     e[Cmd::kMaxCustomElements];
        uint32_t    count = 0, stride = 0;
        const char* why   = "";
        if (!KeyElements(k, e, count, stride, why)) return nullptr;
        constexpr uint32_t kInputs = (1u << manifest::kPosition0) | (1u << manifest::kNormal0) | (1u << manifest::kColor0)
                                   | (1u << manifest::kTexcoord0) | (1u << (manifest::kTexcoord0 + 1));
        const uint32_t want = SemanticMask(e, count) & kInputs;
        if (!(want & (1u << manifest::kPosition0))) return nullptr;

        const ShaderRow* best     = nullptr;
        uint32_t         bestSize = 0;
        for (const ShaderRow& r : kShaderRows)
        {
            if (r.vs[0] != '\0' || r.ps[0] != '\0' || r.ffTextured != k.ffTextured || r.format >= 14) continue;
            const uint32_t have = SemanticMask(kFormats[r.format].e, kFormats[r.format].count);
            if ((have & want) != want) continue;
            if (!best || kFormats[r.format].count < bestSize)
            {
                best     = &r;
                bestSize = kFormats[r.format].count;
            }
        }
        return best;
    }

    /**
     * The hand-written row that draws a key, or null: the row declared for its vs, ps and format,
     * else, for a fixed-function key, the row that covers its layout (CoverRow).
     *
     * @param Key& k : the key
     * @return pointer row : into kShaderRows, or null when no row matches or covers it
     */
    const ShaderRow* RowFor(const Key& k)
    {
        const uint32_t format = RowFormat(k);
        for (const ShaderRow& r : kShaderRows)
            if (format == r.format && std::strcmp(k.vs, r.vs) == 0 &&
                std::strcmp(k.ps, r.ps) == 0 && k.ffTextured == r.ffTextured)
                return &r;
        if (k.vs[0] == '\0' && k.ps[0] == '\0') return CoverRow(k);
        return nullptr;
    }
}
