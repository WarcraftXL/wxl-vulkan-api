// Cull and depth state, and what a key's pipeline is built from.
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
    // --- building -------------------------------------------------------------------------

    /**
     * The engine's cull mode as Vulkan's cull mode and front face -- and the derivation, because
     * the mapping is the identity and looks as if it should be the inversion.
     *
     * EGxCull, from the engine's s_cullMode table: 0 none, 1 culls clockwise (D3DCULL_CW), 2 culls
     * counter-clockwise (D3DCULL_CCW). Both APIs define a triangle's winding in WINDOW space,
     * y down, as the viewer sees it: D3D9's D3DCULL_CW culls "back faces with clockwise
     * vertices" in screen space, and Vulkan's area formula carries a leading minus sign exactly
     * so that VK_FRONT_FACE_COUNTER_CLOCKWISE means counter-clockwise on its y-down framebuffer.
     * So the only question is whether a triangle lands on the Vulkan screen with the winding it
     * had on the D3D screen -- and it does, vertex by vertex:
     *
     *   D3D9 viewport:      y_win = (1 - y_ndc) * H/2      NDC +1 is the top
     *   Vulkan viewport:    y_win = (1 + y_ndc') * H/2     NDC -1 is the top (1.0 core: positive
     *                                                       height, no VK_KHR_maintenance1)
     *   our vertex shaders: y_ndc' = -y_ndc                 gl_Position.y = -gl_Position.y
     *   therefore:          y_win = (1 - y_ndc) * H/2      the same pixel as D3D, for every vertex
     *
     * The shader flip and the viewport convention are two inversions and they cancel: the flip
     * exists precisely to put each vertex where D3D put it, and three vertices on the same pixels
     * wind the same way. The mistake to avoid is counting the flip as a winding change on its own.
     *
     * Hence 1 (cull CW) is front face COUNTER_CLOCKWISE, cull BACK; 2 (cull CCW) is front face
     * CLOCKWISE, cull BACK. The UI draws with 1 and is visible under D3D, so its quads are CCW
     * on screen and must stay visible here: that draw is the test of this table.
     *
     * @param uint32 cull : EGxCull
     * @param VkCullModeFlags& mode : receives the cull mode
     * @param VkFrontFace& front : receives the front face
     * @param string& name : receives a short spelling for the log
     * @return bool ok : false for a value the engine does not define
     */
    bool CullState(uint32_t cull, VkCullModeFlags& mode, VkFrontFace& front, const char*& name)
    {
        switch (cull)
        {
            case 0:
                mode  = VK_CULL_MODE_NONE;
                front = VK_FRONT_FACE_COUNTER_CLOCKWISE;
                name  = "NONE";
                return true;
            case 1:
                mode  = VK_CULL_MODE_BACK_BIT;
                front = VK_FRONT_FACE_COUNTER_CLOCKWISE;
                name  = "BACK with front CCW";
                return true;
            case 2:
                mode  = VK_CULL_MODE_BACK_BIT;
                front = VK_FRONT_FACE_CLOCKWISE;
                name  = "BACK with front CW";
                return true;
            default:
                return false;
        }
    }

    /**
     * The engine's depth function as a VkCompareOp. The engine's table (s_cmpFunc) is in the
     * D3D order: 0 LE, 1 EQ, 2 GE, 3 LT.
     *
     * @param uint32 func : the engine's depth function
     * @param VkCompareOp& op : receives the compare op
     * @param string& name : receives a short spelling for the log
     * @return bool ok : false for a value outside the table
     */
    bool DepthOp(uint32_t func, VkCompareOp& op, const char*& name)
    {
        switch (func)
        {
            case 0: op = VK_COMPARE_OP_LESS_OR_EQUAL;    name = "LE"; return true;
            case 1: op = VK_COMPARE_OP_EQUAL;            name = "EQ"; return true;
            case 2: op = VK_COMPARE_OP_GREATER_OR_EQUAL; name = "GE"; return true;
            case 3: op = VK_COMPARE_OP_LESS;             name = "LT"; return true;
            default: return false;
        }
    }

    /**
     * Decides which source a key is built from: a hand-written row, the translated modules, or
     * nothing. WXL_VK_TRANSLATED's three modes are exactly this function's three branches.
     *
     * @param Key& k : the key
     * @param const ShaderRow*& row : receives the row, when that is the source
     * @param const Record*& vr : receives the vertex record, when the modules are
     * @param const Record*& pr : receives the pixel record, when the modules are -- or when the
     *        source is the fixed-function vertex stage feeding that module (ffVertex)
     * @param bool& ffVertex : receives true for a null vertex shader on a known pixel module: the
     *        semantic ff.vert and the module
     * @param bool& companion : receives true for a null PIXEL shader on a known vertex module
     *        that names an ff companion: the module and its companion; vr and pr
     *        are both set then
     * @param string& why : receives the reason when nothing is available
     * @return bool ok : false when the key has no source
     */
    /**
     * The fixed-function pixel stage for a vertex module drawn with no pixel shader: the one its
     * manifest row names, else the generated stage of its kind -- the point-sprite one for a
     * point_size module, the texcoord0 one otherwise -- since D3D runs the same stage-0 cascade
     * after whichever vertex shader is bound. Build() still refuses the pair, by name, when the
     * stage reads a varying the module does not write.
     *
     * @param Record& v : the vertex record
     * @return pointer record : the ff pixel record, or null when the manifest has none
     */
    const manifest::Record* CompanionFor(const manifest::Record& v)
    {
        if (const manifest::Record* c = manifest::Companion(v)) return c;
        return manifest::FfStage(v.pointSize);
    }

    bool Source(const Key& k, const ShaderRow*& row, const manifest::Record*& vr,
                const manifest::Record*& pr, bool& ffVertex, bool& companion, const char*& why)
    {
        row       = RowFor(k);
        vr        = nullptr;
        pr        = nullptr;
        ffVertex  = false;
        companion = false;

        if (s.mode == settings::kTranslatedOff)
        {
            if (row) return true;
            why = "no hand-written shader pair for this vs/ps/format (translated modules are off)";
            return false;
        }

        // Fill: the row wins when it exists. Prefer: the modules win when BOTH resolve, and the
        // row is the fallback -- so a half-translated pair never mixes a module with a row.
        if (s.mode == settings::kTranslatedFill && row) return true;

        const manifest::Record* v = translated::Resolve(k.vs, false);
        const manifest::Record* p = translated::Resolve(k.ps, true);

        // A known vertex module drawn with NO pixel shader (the vs_2_0 weather records) is D3D's
        // programmable vertex stage feeding the fixed-function
        // pixel cascade. The emitter generated that cascade as an `ff` pixel record and the vertex
        // row names it (`companion`); it is bound in place of ff.frag, whose varyings and fog row
        // are the hand-written layout's and not this module's. Resolve("") is null by definition,
        // so the modules path below could never take this key; it comes after the row (none
        // exists, but the precedence is the rows') and before everything else. A draw whose ps is
        // a real shader never gets here with p null, so a vertex row with a companion keeps
        // pairing with its real pixel shader when the engine binds one.
        if (k.ps[0] == '\0' && v)
        {
            if (const manifest::Record* c = CompanionFor(*v))
            {
                row       = nullptr;
                vr        = v;
                pr        = c;
                companion = true;
                return true;
            }
        }

        if (v && p)
        {
            row = nullptr;
            vr  = v;
            pr  = p;
            return true;
        }
        if (row) return true;

        // No vertex shader and a known pixel module is the engine's full-screen-effect
        // shape (vs "", ps FFX*): the D3D backend runs its fixed-function vertex pipeline into
        // the programmable pixel stage, and so does this side, from the semantic build of ff.vert.
        // Resolve("") is null by definition, so this is the only source such a key can have; a
        // hand-written row for it (none exists) would have won above.
        if (k.vs[0] == '\0' && p)
        {
            pr       = p;
            ffVertex = true;
            return true;
        }

        if (!manifest::Loaded())
            why = "no hand-written shader pair for this vs/ps/format, and no manifest";
        else if (!v && !p)
            why = "no hand-written shader pair, and neither shader has a known translated module";
        else if (!v)
            why = "no hand-written shader pair, and the vertex shader has no known translated module";
        else if (k.ps[0] == '\0')
            why = "no hand-written shader pair, no pixel shader, and the vertex module's manifest row names no ff companion";
        else
            why = "no hand-written shader pair, and the pixel shader has no known translated module";
        return false;
    }
}
