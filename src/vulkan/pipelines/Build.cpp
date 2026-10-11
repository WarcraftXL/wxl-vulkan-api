// Building one pipeline for a key against a render pass.
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
    /// "translated/<path>", or with ".gbuf" before its ".spv" for a G-buffer variant: the names follow
    /// from the module's own, as the shadow variants' do (3.Tools/gbuffer).
    static void VariantPath(char* out, size_t size, const char* path, bool gbuf)
    {
        const size_t n = std::strlen(path);
        if (gbuf && n > 4 && std::strcmp(path + n - 4, ".spv") == 0)
            std::snprintf(out, size, "translated/%.*s.gbuf.spv", static_cast<int>(n - 4), path);
        else
            std::snprintf(out, size, "translated/%s", path);
    }

    /**
     * Builds the pipeline for a key against a render pass.
     *
     * @param Key& k : the key
     * @param VkRenderPass pass : the pass it must be compatible with
     * @param PipelineEntry& out : receives the pipeline and everything Record() needs to use it
     * @param string& why : receives the reason on failure
     * @param Sources* forced : the sources to build from instead of Source()'s (the pre-warm), or null
     * @return bool ok
     */
    bool Build(const Key& k, VkRenderPass pass, PipelineEntry& out, const char*& why, const Sources* forced)
    {
        const ShaderRow*        row       = nullptr;
        const manifest::Record* vr        = nullptr;
        const manifest::Record* pr        = nullptr;
        bool                    ffVertex  = false;
        bool                    companion = false;
        if (forced)
        {
            // The pre-warm (Prewarm): the sources the persisted key was built from, by md5. The
            // first draw of the key checks them against Source() (VerifyPrewarmed).
            row       = forced->row;
            vr        = forced->vr;
            pr        = forced->pr;
            ffVertex  = forced->ffVertex;
            companion = forced->companion;
        }
        else if (!Source(k, row, vr, pr, ffVertex, companion, why))
            return false;
        // A companion pair IS a pair of translated modules (the vertex module and the generated
        // ff pixel module), so it takes the modules path everywhere below -- vertex input from
        // the vertex row, reads-subset-of-outputs, set 0 from the companion's samplers, the
        // vertex row's slice class -- and differs only in how it is logged and counted.
        const bool fromModules = row == nullptr && !ffVertex;

        // --- the vertex input, the interface check and the layout: where a pair is refused ----
        VertexLayout layout;
        uint32_t     samplerMask = kRowSamplerMask;
        // The draw's own streams: binding 0 and one per extra stream of a custom layout. A row
        // that covers a layout (CoverRow) may add the constant stream after them.
        uint32_t ownStreams = 1;
        while (ownStreams < Cmd::kMaxStreams && k.streamStride[ownStreams - 1] != 0) ++ownStreams;
        bool needsConst = false;
        if (fromModules || ffVertex)
        {
            // The vertex side's input and what it writes: the module's own, or the semantic
            // ff.vert's fixed set.
            uint32_t missing = 0;
            if (fromModules ? !TranslatedVertexLayout(k, vr->inputMask, layout, missing, why)
                            : !FfSemanticVertexLayout(k, layout, why))
            {
                RefusePair(k, why, manifest::SemanticName(missing));
                return false;
            }
            // reads subset-of outputs: the translator verified every engine pair satisfies it, so a
            // failure here is the emitter's and the draw is skipped rather than fed an undefined
            // varying. The fixed-function vertex stage writes a fixed set (kFfSemanticOutputs).
            const uint32_t unmet = pr->readsMask & ~(fromModules ? vr->outputMask : kFfSemanticOutputs);
            if (unmet)
            {
                uint32_t sem = 0;
                while (!(unmet & (1u << sem))) ++sem;
                why = fromModules ? "the pixel module reads a semantic the vertex module does not write"
                                  : "the pixel module reads a semantic the fixed-function vertex stage does not write";
                RefusePair(k, why, manifest::SemanticName(sem));
                return false;
            }
            // A `sampler N cube` line changes nothing about the set-0 layout: a samplerCube is a
            // combined image sampler at binding N like a sampler2D, and only the VIEW bound at
            // draw time differs (VIEW_TYPE_CUBE over six layers). The mask is kept on the entry
            // so that RecordOne() can refuse, per draw, an image whose view type is not the one
            // the module declared -- binding a 2D view to a samplerCube is invalid usage that no
            // driver reports, not a wrong picture.
            samplerMask = pr->samplerMask;
        }
        else if (!RowVertexLayout(k, *row, layout, ownStreams, needsConst))
        {
            why = "the row's vertex format has no layout here";
            return false;
        }
        else if (needsConst && (ownStreams >= Cmd::kMaxStreams || s.constVb.handle == VK_NULL_HANDLE))
        {
            why = "the fixed-function row needs the constant vertex stream, and there is no binding or no buffer for it";
            return false;
        }
        else
        {
            // A row with no sampler is still built on the one-sampler layout, with the binding
            // left unwritten (legal while nothing uses it), so that every row shares a layout
            // Init() made and a layout is never created for an empty set.
            samplerMask = row->samplers ? row->samplers : kRowSamplerMask;
        }

        uint32_t layoutIndex = 0;
        if (!LayoutFor(samplerMask, layoutIndex))
        {
            why = "no descriptor layout for the pixel module's sampler slots (see the line above)";
            return false;
        }

        VkPrimitiveTopology topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        if (!Topology(k.topology, topology))
        {
            why = "topology is not translated";
            return false;
        }
        // A vertex record flagged point_size writes gl_PointSize, which only a point
        // list reads, and the engine's draw for these is expected to be prim 0 (EGxPrim points),
        // which Topology() already maps to POINT_LIST. The flag is the bytecode's own statement
        // and wins over the key: a disagreement is logged once (the key still gets its own
        // pipeline, since the prim is in it) rather than drawn as triangles that would read a
        // sprite's single vertex three times.
        const bool pointList = fromModules && vr->pointSize;
        if (pointList && topology != VK_PRIMITIVE_TOPOLOGY_POINT_LIST)
        {
            if (!s.loggedPointPrim)
            {
                s.loggedPointPrim = true;
                wxl::ScriptMgr::Log(WXL_LOG_WARN,
                                    "vulkan: pipelines: vs=\"%s\" is flagged point_size but the "
                                    "engine drew it with prim %u, not 0 (points); built as a point "
                                    "list on the flag's word (logged once)",
                                    k.vs, k.topology);
            }
            topology = VK_PRIMITIVE_TOPOLOGY_POINT_LIST;
        }

        VkPipelineColorBlendAttachmentState blend = {};
        if (!BlendState(k.blend, blend))
        {
            why = "blend mode is not translated";
            return false;
        }
        blend.colorWriteMask = WriteMask(k.colorWrite);

        // The manifest's paths are relative to its own directory, shaders/translated/; the shader
        // loader resolves relative to shaders/.
        // A 256-row module with a bone-palette variant, on a key that allows it, is built from the
        // variant: the same math reading c64.. from set 2, on the layout that has set 2.
        const bool bones = (k.bones & 1u) && fromModules && vr->vconst == Cmd::kVertexRegsHigh && vr->bones[0] != '\0'
                           && BonesLayoutFor(layoutIndex) != VK_NULL_HANDLE;
        // A tree-wind draw on a module with a `wind` variant (its `windbones` one on the palette
        // path): that variant, on the layout with set 3. Without one, the module itself (still).
        const bool windVar = (k.bones & 2u) && fromModules && vr
                             && (bones ? vr->windBones[0] != '\0' : vr->wind[0] != '\0')
                             && WindLayoutFor(layoutIndex, bones) != VK_NULL_HANDLE;
        if ((k.bones & 2u) && fromModules && !windVar) wxlvk::vk::wind::NoVariant(k.vs);
        // With the vertex wind variant, the pixel module's own (the leaves' shimmer and light) when
        // it has one: it reads the varying only that variant writes.
        const bool windPix = windVar && pr && pr->wind[0] != '\0';
        // The G-buffer variants (3.Tools/gbuffer, the manifest's `gbuf` lines) on a G-buffer pass class:
        // an opaque world draw (the key's bit) whose vertex module -- in the variant chosen above: plain,
        // bones, wind, both -- and pixel module both have one takes them and writes the pass's three
        // extra attachments. Without both, the modules as they are, with those attachments masked off.
        const bool gbufPass = IsGBufferPass(pass);
        const bool gbufVar  = gbufPass && (k.bones & 4u) && fromModules && !pointList
                              && (bones ? (windVar ? vr->gbufWindBones : vr->gbufBones) : (windVar ? vr->gbufWind : vr->gbuf))
                              && (windPix ? pr->gbufWind : pr->gbuf);
        char vertPath[160], fragPath[160];   // "translated/" + a manifest path (kPathChars), ".gbuf" added
        VariantPath(vertPath, sizeof vertPath,
                    vr ? (bones ? (windVar ? vr->windBones : vr->bones) : (windVar ? vr->wind : vr->spv)) : "", gbufVar);
        VariantPath(fragPath, sizeof fragPath, pr ? (windPix ? pr->wind : pr->spv) : "", gbufVar);
        const VkShaderModule vert = fromModules ? shaders::GetPath(vertPath)
                                  : shaders::Get(ffVertex ? kFfSemanticVertStem : row->vertSpv);
        const VkShaderModule frag = (fromModules || ffVertex) ? shaders::GetPath(fragPath) : shaders::Get(row->fragSpv);
        if (vert == VK_NULL_HANDLE || frag == VK_NULL_HANDLE)
        {
            why = "a SPIR-V module did not load (see the shaders line above)";
            return false;
        }

        VkPipelineShaderStageCreateInfo stages[2] = {};
        stages[0].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stages[0].stage  = VK_SHADER_STAGE_VERTEX_BIT;
        stages[0].module = vert;
        stages[0].pName  = "main";
        stages[1].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stages[1].stage  = VK_SHADER_STAGE_FRAGMENT_BIT;
        stages[1].module = frag;
        stages[1].pName  = "main";

        // Binding 0 is the draw's vertex buffer; multi-stream custom layouts add one binding
        // per extra stream, each with its own buf's stride (the key carries them, zero past the last).
        VkVertexInputBindingDescription bindings[Cmd::kMaxStreams] = {};
        uint32_t bindingCount = 1;
        bindings[0].binding   = 0;
        bindings[0].stride    = layout.stride;
        bindings[0].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
        for (uint32_t i = 0; i < Cmd::kMaxStreams - 1 && k.streamStride[i] != 0; ++i)
        {
            bindings[bindingCount].binding   = bindingCount;
            bindings[bindingCount].stride    = k.streamStride[i];
            bindings[bindingCount].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
            ++bindingCount;
        }
        // The constant stream: stride 0, so every vertex reads the same bytes (Vulkan allows a
        // zero stride; Record() binds State::constVb here).
        const uint32_t realStreams = bindingCount;
        if (needsConst)
        {
            bindings[bindingCount].binding   = bindingCount;
            bindings[bindingCount].stride    = 0;
            bindings[bindingCount].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
            ++bindingCount;
        }

        VkPipelineVertexInputStateCreateInfo vertexInput = {};
        vertexInput.sType                           = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        vertexInput.vertexBindingDescriptionCount   = bindingCount;
        vertexInput.pVertexBindingDescriptions      = bindings;
        vertexInput.vertexAttributeDescriptionCount = layout.count;
        vertexInput.pVertexAttributeDescriptions    = layout.attrs;

        VkPipelineInputAssemblyStateCreateInfo assembly = {};
        assembly.sType                  = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        assembly.topology               = topology;
        assembly.primitiveRestartEnable = VK_FALSE;

        // Counts only; the values are dynamic state, set per draw from the Cmd.
        VkPipelineViewportStateCreateInfo viewport = {};
        viewport.sType         = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        viewport.viewportCount = 1;
        viewport.scissorCount  = 1;

        // Culling from the key, through the one table whose derivation is written down; see
        // CullState. Depth clamp off, like D3D9 without D3DRS_CLIPPING tricks: a fragment past
        // the far plane is clipped, not clamped.
        VkCullModeFlags cullMode  = VK_CULL_MODE_NONE;
        VkFrontFace     frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
        const char*     cullName  = "";
        if (!CullState(k.cull, cullMode, frontFace, cullName))
        {
            why = "cull mode is not one the engine defines";
            return false;
        }

        VkPipelineRasterizationStateCreateInfo raster = {};
        raster.sType                   = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        raster.depthClampEnable        = VK_FALSE;
        raster.rasterizerDiscardEnable = VK_FALSE;
        raster.polygonMode             = VK_POLYGON_MODE_FILL;
        raster.cullMode                = cullMode;
        raster.frontFace               = frontFace;
        // The bias is dynamic state, enabled only for a key that carries one (see Key).
        raster.depthBiasEnable         = k.depthBias ? VK_TRUE : VK_FALSE;
        raster.lineWidth               = 1.0f;

        VkPipelineMultisampleStateCreateInfo multisample = {};
        multisample.sType                = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

        // Depth from the key, against the pass class's depth attachment (src/native/Present.cpp).
        // The semantics line up with D3D9 without translation: D3DRS_ZENABLE is
        // depthTestEnable, D3DRS_ZWRITEENABLE is depthWriteEnable, and in both APIs a write with
        // the test off is no write at all (Vulkan: "depth writes are always disabled when
        // depthTestEnable is VK_FALSE"), so the UI's 0/0/0 is exactly "no depth". The compare op
        // is the engine's own function, through DepthOp.
        //
        // Depth RANGE needs nothing here. D3D9 and Vulkan both map clip z to [0, 1] -- the GL
        // [-1, 1] remap a translated shader would need does not arise -- and the viewport's
        // minDepth/maxDepth are the engine's own MinZ/MaxZ, set per draw with the viewport
        // (src/native/Draw.cpp PlaceD3d), as D3D9 applied them.
        //
        // A pass class without a depth attachment (no depth format, or a target without depth) has
        // its own render pass, and the spec ignores this whole structure for that subpass.
        VkCompareOp depthOp   = VK_COMPARE_OP_LESS_OR_EQUAL;
        const char* depthName = "";
        if (!DepthOp(k.depthFunc, depthOp, depthName))
        {
            why = "depth function is not one the engine defines";
            return false;
        }

        VkPipelineDepthStencilStateCreateInfo depthStencil = {};
        depthStencil.sType                 = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
        depthStencil.depthTestEnable       = k.depthTest ? VK_TRUE : VK_FALSE;
        depthStencil.depthWriteEnable      = k.depthWrite ? VK_TRUE : VK_FALSE;
        depthStencil.depthCompareOp        = depthOp;
        depthStencil.depthBoundsTestEnable = VK_FALSE;
        // No stencil: the engine never sets stencil state, and the attachment is D32_SFLOAT where
        // the adapter allows, which has no stencil aspect to test.
        depthStencil.stencilTestEnable     = VK_FALSE;
        depthStencil.minDepthBounds        = 0.0f;
        depthStencil.maxDepthBounds        = 1.0f;

        // A G-buffer class names four colour attachments: the draw's blend on the frame; on the three others
        // no blend and every channel for the G-buffer variants. A depth-writing pipeline WITHOUT a variant
        // (a module the tool has no family for, a fixed-function draw) writes the albedo attachment zeros --
        // a blend of two zero factors: its module has no output for the attachment, and whatever that
        // output holds is multiplied away -- so the class byte under it reads "none" and a consumer leaves
        // the pixel alone; its normal and light stay masked. A pipeline that writes no depth (a decal) masks
        // all three: the opaque surface under it keeps its record. (independentBlend makes the per-attachment
        // state legal; the device refuses the G-buffer without it.)
        //   2026-10-11: the zeros are NOT guaranteed. Vulkan leaves an attachment the fragment shader has no output for
        // undefined whatever the blend says, and the usual hardware simply skips the write: the albedo then keeps the
        // record of whatever was drawn behind (a Goldshire house's wall relit as the terrain behind it). What tells the
        // consumers apart is the variants' depth tag (VkRenderApi.h WXL_VK_GBUFFER_TAG_*: the record's clip w in the
        // normal's w, which these pipelines leave masked); each such pipeline is named in the log once, and its draws
        // counted (gbufferNullDraws).
        const uint32_t all = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        const bool gbufNull = gbufPass && !gbufVar && k.depthTest && k.depthWrite;
        if (gbufNull)
        {
            const char* reason = !(k.bones & 4u) ? "not an opaque world draw by its state (blended, colour masked or fixed-function)"
                               : ffVertex        ? "the fixed-function vertex stage"
                               : !fromModules    ? "a hand-written row"
                               : pointList       ? "a point list"
                                                 : "no G-buffer variant for this module pair in the manifest";
            wxl::ScriptMgr::Log(WXL_LOG_INFO,
                                "vulkan: G-buffer: vs=\"%s\" ps=\"%s\" blend %u colour write 0x%X writes depth on the "
                                "G-buffer pass with NO record (%s): its pixels keep the record of the surface behind, "
                                "which the consumers set aside by the depth tag",
                                k.vs, k.ps, k.blend, k.colorWrite, reason);
        }
        VkPipelineColorBlendAttachmentState blends[1 + 3] = { blend, {}, {}, {} };
        for (uint32_t i = 1; i < 4; ++i)
        {
            blends[i].blendEnable    = VK_FALSE;
            blends[i].colorWriteMask = gbufVar ? all : 0u;
        }
        if (gbufNull)
        {
            blends[1].blendEnable         = VK_TRUE;
            blends[1].srcColorBlendFactor = VK_BLEND_FACTOR_ZERO;
            blends[1].dstColorBlendFactor = VK_BLEND_FACTOR_ZERO;
            blends[1].colorBlendOp        = VK_BLEND_OP_ADD;
            blends[1].srcAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
            blends[1].dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
            blends[1].alphaBlendOp        = VK_BLEND_OP_ADD;
            blends[1].colorWriteMask      = all;
        }
        VkPipelineColorBlendStateCreateInfo colorBlend = {};
        colorBlend.sType           = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        colorBlend.logicOpEnable   = VK_FALSE;
        colorBlend.attachmentCount = gbufPass ? 4u : 1u;
        colorBlend.pAttachments    = blends;

        // Two more, each only for the pipelines that read it: the depth bias (dynamic rather than
        // keyed, Key says why) and the blend constant of mode 11 (dynamic because state 84 is a
        // float the engine animates -- a key per value would rebuild a pipeline per fade step).
        VkDynamicState dynamics[4] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
        uint32_t       dynamicCount = 2;
        if (k.depthBias) dynamics[dynamicCount++] = VK_DYNAMIC_STATE_DEPTH_BIAS;
        if (k.blend == 11) dynamics[dynamicCount++] = VK_DYNAMIC_STATE_BLEND_CONSTANTS;
        VkPipelineDynamicStateCreateInfo dynamic = {};
        dynamic.sType             = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
        dynamic.dynamicStateCount = dynamicCount;
        dynamic.pDynamicStates    = dynamics;

        VkGraphicsPipelineCreateInfo info = {};
        info.sType               = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        info.stageCount          = 2;
        info.pStages             = stages;
        info.pVertexInputState   = &vertexInput;
        info.pInputAssemblyState = &assembly;
        info.pViewportState      = &viewport;
        info.pRasterizationState = &raster;
        info.pMultisampleState   = &multisample;
        info.pDepthStencilState  = &depthStencil;
        info.pColorBlendState    = &colorBlend;
        info.pDynamicState       = &dynamic;
        info.layout              = bones ? (windVar ? s.layouts[layoutIndex].windBones : s.layouts[layoutIndex].bones)
                                 : (windVar ? s.layouts[layoutIndex].wind : s.layouts[layoutIndex].pipeline);
        info.renderPass          = pass;
        info.subpass             = 0;
        info.basePipelineHandle  = VK_NULL_HANDLE;
        info.basePipelineIndex   = -1;

        // Through the persisted cache (s.cache): a session after the first finds the driver's
        // compiled code for the pipelines it built before, and a zone's first frames do not compile
        // hundreds of them.
        VkPipeline pipeline = VK_NULL_HANDLE;
        const VkResult r = s.dapi->CreateGraphicsPipelines(s.device, s.cache, 1, &info,
                                                           nullptr, &pipeline);
        if (r != VK_SUCCESS)
        {
            Fail("vkCreateGraphicsPipelines", r);
            why = "vkCreateGraphicsPipelines failed (see the line above)";
            return false;
        }

        out.pipeline   = pipeline;
        out.row        = row;
        out.layout     = layoutIndex;
        out.stride     = layout.stride;
        out.dynamicBias  = k.depthBias != 0;
        out.dynamicBlend = k.blend == 11;
        out.streams      = realStreams;
        out.constStream  = needsConst;
        out.translated = fromModules;
        out.ffVertex   = ffVertex;
        out.companion  = companion;
        out.pointList  = pointList;
        out.gbufNull   = gbufNull;
        if (fromModules)
        {
            // The companion pair too: set 0 from the companion's one 2D sampler (stage 0 from
            // Cmd::texture[0], as for any module), the slice class from the vertex row (256 for
            // patter/rain, 64 for the sprites; the companion declares v[64] and reads none of it).
            out.usesRing  = true;
            out.writeMask = pr->samplerMask;
            out.cubeMask  = pr->cubeMask;
            // The variant declares the 64-row block: its slice is the 64-row class.
            out.vconst    = bones ? Cmd::kVertexRegs : vr->vconst;
            out.bones     = bones;
            out.wind      = windVar;
            out.gbuf      = gbufVar;
            out.vrec      = vr;
            out.prec      = pr;
        }
        else if (ffVertex)
        {
            // The ring, like every fixed-function row: the reader's register file in the vertex
            // rows, the module's p[] from the pixel cache, the 64-row class (the reader's answer for a null vertex shader).
            out.usesRing  = true;
            out.writeMask = pr->samplerMask;
            out.cubeMask  = pr->cubeMask;
            out.vconst    = Cmd::kVertexRegs;
            out.prec      = pr;
        }
        else
        {
            out.usesRing  = row->usesRing;
            out.writeMask = row->samplers;
            out.vconst    = Cmd::kVertexRegs;
        }
        return true;
    }
}
