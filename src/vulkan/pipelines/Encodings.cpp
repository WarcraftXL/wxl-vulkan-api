// The engine's blend, write mask, topology and fixed vertex formats, in Vulkan terms.
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
    // --- the engine's encodings, translated -------------------------------------------------

    /**
     * The colour blend state for one EGxBlend value, from the engine's own table.
     *
     *   0 Opaque        off                     7 InvSrcAlphaAdd    INVSRCALPHA / ONE
     *   1 AlphaKey      off (the test, not a blend)  8 InvSrcAlphaOpaque INVSRCALPHA / ZERO
     *   2 Alpha         SRCALPHA / INVSRCALPHA  9 SrcAlphaOpaque    SRCALPHA / ZERO
     *   3 Add           SRCALPHA / ONE         10 NoAlphaAdd        ONE / ONE
     *   4 Mod           DSTCOLOR / ZERO        11 ConstantAlpha     CONSTANT_COLOR / ONE_MINUS_CONSTANT_COLOR
     *   5 Mod2x         DSTCOLOR / SRCCOLOR    12, 13               not in the table read so far
     *   6 ModAdd        DSTCOLOR / ONE
     *
     * The same factors for alpha as for colour, op ADD on both: that is what D3D9 does when
     * SEPARATEALPHABLENDENABLE is off, which the engine never turns on. 11 reads the dynamic blend
     * constant Record() sets from the Cmd's factor (render state 84).
     *
     * @param uint32 blend : EGxBlend
     * @param VkPipelineColorBlendAttachmentState& a : receives the enable and factors
     * @return bool ok : false for a value that is not translated (12, 13)
     */
    bool BlendState(uint32_t blend, VkPipelineColorBlendAttachmentState& a)
    {
        a.blendEnable         = VK_TRUE;
        a.colorBlendOp        = VK_BLEND_OP_ADD;
        a.alphaBlendOp        = VK_BLEND_OP_ADD;

        VkBlendFactor src = VK_BLEND_FACTOR_ONE;
        VkBlendFactor dst = VK_BLEND_FACTOR_ZERO;
        switch (blend)
        {
            case 0:
            case 1:  a.blendEnable = VK_FALSE; break;
            case 2:  src = VK_BLEND_FACTOR_SRC_ALPHA;           dst = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA; break;
            case 3:  src = VK_BLEND_FACTOR_SRC_ALPHA;           dst = VK_BLEND_FACTOR_ONE;                 break;
            case 4:  src = VK_BLEND_FACTOR_DST_COLOR;           dst = VK_BLEND_FACTOR_ZERO;                break;
            case 5:  src = VK_BLEND_FACTOR_DST_COLOR;           dst = VK_BLEND_FACTOR_SRC_COLOR;           break;
            case 6:  src = VK_BLEND_FACTOR_DST_COLOR;           dst = VK_BLEND_FACTOR_ONE;                 break;
            case 7:  src = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA; dst = VK_BLEND_FACTOR_ONE;                 break;
            case 8:  src = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA; dst = VK_BLEND_FACTOR_ZERO;                break;
            case 9:  src = VK_BLEND_FACTOR_SRC_ALPHA;           dst = VK_BLEND_FACTOR_ZERO;                break;
            case 10: src = VK_BLEND_FACTOR_ONE;                 dst = VK_BLEND_FACTOR_ONE;                 break;
            case 11:
                // D3D's BLENDFACTOR / INVBLENDFACTOR with the factor grey in all four channels
                // (state 84), which is the dynamic blend constant Record() sets.
                src = VK_BLEND_FACTOR_CONSTANT_COLOR;
                dst = VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_COLOR;
                break;
            default: return false; // 12, 13
        }

        a.srcColorBlendFactor = src;
        a.dstColorBlendFactor = dst;
        a.srcAlphaBlendFactor = src;
        a.dstAlphaBlendFactor = dst;
        return true;
    }

    /**
     * The engine's colour write mask in Vulkan's bits. The engine's order is R, B, G, A -- bit 1
     * is BLUE and bit 2 is GREEN, the opposite of D3D's D3DCOLORWRITEENABLE_* and of Vulkan's --
     * so the two middle bits swap on the way through.
     *
     * @param uint32 engine : the mask as the Cmd carries it
     * @return VkColorComponentFlags mask
     */
    VkColorComponentFlags WriteMask(uint32_t engine)
    {
        VkColorComponentFlags m = 0;
        if (engine & 1u) m |= VK_COLOR_COMPONENT_R_BIT;
        if (engine & 2u) m |= VK_COLOR_COMPONENT_B_BIT;
        if (engine & 4u) m |= VK_COLOR_COMPONENT_G_BIT;
        if (engine & 8u) m |= VK_COLOR_COMPONENT_A_BIT;
        return m;
    }

    /**
     * An EGxPrim as a Vulkan topology.
     *
     * The whole of s_primitiveConversion (cgxdevice 2.7): every EGxPrim is a Vulkan 1.0 core
     * topology with D3D's vertex order and count (the fan is optional only under the portability
     * subset, which a Windows driver does not expose). The D3D probe saw only 3 and 4; 2 and 5
     * are here so that a rare caller is drawn rather than skipped.
     *
     * @param uint32 prim : EGxPrim
     * @param VkPrimitiveTopology& out : receives the topology
     * @return bool ok : false past 5
     */
    bool Topology(uint32_t prim, VkPrimitiveTopology& out)
    {
        switch (prim)
        {
            case 0: out = VK_PRIMITIVE_TOPOLOGY_POINT_LIST;     return true;
            case 1: out = VK_PRIMITIVE_TOPOLOGY_LINE_LIST;      return true;
            case 2: out = VK_PRIMITIVE_TOPOLOGY_LINE_STRIP;     return true;
            case 3: out = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;  return true;
            case 4: out = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP; return true;
            case 5: out = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_FAN;   return true;
            default: return false;
        }
    }

    /**
     * An EGxVertexBufferFormat as a Vulkan vertex input, for the formats a hand-written row
     * declares. One interleaved stream, binding 0, per-vertex rate, laid out as the engine lays it
     * out (cgxdevice 3.5/3.6) and as the GLSL declares it. The colour is a D3DCOLOR: four bytes
     * B, G, R, A in memory, which is exactly VK_FORMAT_B8G8R8A8_UNORM -- so the attribute fetch is
     * where the channel order is put right, and the shader sees r, g, b, a.
     *
     *   8 PCT,  stride 24: position float3 @0, colour @12, texcoord float2 @16
     *   4 PNCT, stride 36: position float3 @0, normal float3 @12, colour @24, texcoord float2 @28
     *   7 PC,   stride 16: position float3 @0, colour @12
     *  10 PT,   stride 20: position float3 @0, texcoord float2 @12
     *
     * Locations number the attributes in the order they appear in the vertex, which is what the
     * hand-written GLSL declares (ff.vert derives its locations from the same table). The
     * translated modules use the semantic-keyed table below instead.
     *
     * @param uint32 format : EGxVertexBufferFormat
     * @param VertexLayout& out : receives the layout
     * @return bool ok : false for a format no row uses
     */
    bool VertexLayoutFor(uint32_t format, VertexLayout& out)
    {
        VkVertexInputAttributeDescription* a = out.attrs;
        switch (format)
        {
            case 7:
                out.stride = 16;
                out.count  = 2;
                a[0].location = 0; a[0].binding = 0; a[0].format = VK_FORMAT_R32G32B32_SFLOAT; a[0].offset = 0;
                a[1].location = 1; a[1].binding = 0; a[1].format = VK_FORMAT_B8G8R8A8_UNORM;   a[1].offset = 12;
                return true;
            case 10:
                out.stride = 20;
                out.count  = 2;
                a[0].location = 0; a[0].binding = 0; a[0].format = VK_FORMAT_R32G32B32_SFLOAT; a[0].offset = 0;
                a[1].location = 1; a[1].binding = 0; a[1].format = VK_FORMAT_R32G32_SFLOAT;    a[1].offset = 12;
                return true;
            case 8:
                out.stride = 24;
                out.count  = 3;
                a[0].location = 0; a[0].binding = 0; a[0].format = VK_FORMAT_R32G32B32_SFLOAT; a[0].offset = 0;
                a[1].location = 1; a[1].binding = 0; a[1].format = VK_FORMAT_B8G8R8A8_UNORM;   a[1].offset = 12;
                a[2].location = 2; a[2].binding = 0; a[2].format = VK_FORMAT_R32G32_SFLOAT;    a[2].offset = 16;
                return true;
            case 4:
                out.stride = 36;
                out.count  = 4;
                a[0].location = 0; a[0].binding = 0; a[0].format = VK_FORMAT_R32G32B32_SFLOAT; a[0].offset = 0;
                a[1].location = 1; a[1].binding = 0; a[1].format = VK_FORMAT_R32G32B32_SFLOAT; a[1].offset = 12;
                a[2].location = 2; a[2].binding = 0; a[2].format = VK_FORMAT_B8G8R8A8_UNORM;   a[2].offset = 24;
                a[3].location = 3; a[3].binding = 0; a[3].format = VK_FORMAT_R32G32_SFLOAT;    a[3].offset = 28;
                return true;
            case 6: // PNCT2 (cgxdevice 3.6): format 4 and a second UV set
                out.stride = 44;
                out.count  = 5;
                a[0].location = 0; a[0].binding = 0; a[0].format = VK_FORMAT_R32G32B32_SFLOAT; a[0].offset = 0;
                a[1].location = 1; a[1].binding = 0; a[1].format = VK_FORMAT_R32G32B32_SFLOAT; a[1].offset = 12;
                a[2].location = 2; a[2].binding = 0; a[2].format = VK_FORMAT_B8G8R8A8_UNORM;   a[2].offset = 24;
                a[3].location = 3; a[3].binding = 0; a[3].format = VK_FORMAT_R32G32_SFLOAT;    a[3].offset = 28;
                a[4].location = 4; a[4].binding = 0; a[4].format = VK_FORMAT_R32G32_SFLOAT;    a[4].offset = 36;
                return true;
            case 9: // PCT2: format 8 and a second UV set
                out.stride = 32;
                out.count  = 4;
                a[0].location = 0; a[0].binding = 0; a[0].format = VK_FORMAT_R32G32B32_SFLOAT; a[0].offset = 0;
                a[1].location = 1; a[1].binding = 0; a[1].format = VK_FORMAT_B8G8R8A8_UNORM;   a[1].offset = 12;
                a[2].location = 2; a[2].binding = 0; a[2].format = VK_FORMAT_R32G32_SFLOAT;    a[2].offset = 16;
                a[3].location = 3; a[3].binding = 0; a[3].format = VK_FORMAT_R32G32_SFLOAT;    a[3].offset = 24;
                return true;
            default:
                return false;
        }
    }
}
