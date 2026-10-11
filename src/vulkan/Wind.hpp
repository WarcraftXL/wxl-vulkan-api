// Tree wind: the engine's tree models found by their textures, read back once, and drawn through
// their vertex shader's `wind` variant with a consumer's per-vertex data (wxl.vk.render, TREE WIND).
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
#include "wxl/VkRenderApi.h"

#include <cstdint>

// Everything here runs on the render thread. Inert -- one bool tested per M2 draw -- until a consumer
// calls WatchTextures; the GPU resources are made at the first frame after that.
//
// A model is (pool buffer, base vertex, stride, index buffer): every instance of an M2 draws from the
// same bytes, so one model answers for all of them. Its life: Collecting (its batches, as draws name
// them) -> read back (indices, then the vertex range they reach; one model at a time, in the frame's
// pre-pass) -> Ready (handed to the consumer) -> Windy (its per-vertex data in the wind buffer) or
// Rejected. The variant (shaders/translated/vertex/<md5>.wind.spv) reads, in set 3, the wind block
// (binding 0) and the per-vertex data (binding 1), and in the push range {bias, start, count, on}:
// vertex gl_VertexIndex - bias of the model is data[start + that], when it is below count.
namespace wxlvk::vk::wind
{
    /// What a windy draw pushes (the variant's push block, 16 bytes at offset 0).
    struct Draw
    {
        int32_t  bias;    ///< the model's base vertex plus its lowest index
        uint32_t start;   ///< its first vec4 in the data buffer
        uint32_t count;   ///< its vertices
        uint32_t on;      ///< its layout: vec4s per vertex (1 the weights, 4 a Pivot Painter skeleton)
    };

    /**
     * The draw path's half, from pipelines::Record: notes the draw against the watched models and
     * says whether it is a windy one.
     *
     * @param Cmd& c : the draw
     * @return pointer draw : its push values when its model is windy and the wind is set, else null
     */
    const Draw* Observe(const wxlvk::draw::Cmd& c, uint32_t frameSlot);

    /// Whether an engine texture is a character's, a creature's or an item's (its path's first folder):
    /// the draw's model is a unit's, a mover for the shadow caches (pipelines/Record.cpp).
    bool UnitTexture(const void* tex);
    /// Whether an engine texture has no file name at all (a character's composited skin, a texture made at
    /// run time): on a skinned model, a unit's.
    bool UnnamedTexture(const void* tex);

    /// Whether the draw's model is a unit's: its texture is (UnitTexture), or an earlier draw of the same
    /// model's wore one -- a character's body is a composited texture with no path, its hair and its
    /// items have theirs, and the whole model moves together.
    bool UnitModel(const wxlvk::draw::Cmd& c);

    /**
     * UnitModel(c) || UnnamedTexture(c.engineTex0): the reader's mover test (draw/Reader.cpp WriteMoverRow), answered
     * from the previous draw's when the model (its buffer, offset, stride and buffer id) and the texture (its CGxTex,
     * user argument and fill) are the same -- a model's batches come in runs. Both tests give the same answer for the
     * same inputs, and the second call changes nothing UnitModel's first did not.
     */
    bool MoverDraw(const wxlvk::draw::Cmd& c);

    /// The frame's start (pipelines::BeginFrame): the slot's wind block from the last SetWind, and the
    /// GPU resources the first time after WatchTextures.
    void BeginFrame(uint32_t frameSlot);

    /// Set 3's layout and the empty set-2 layout a wind pipeline layout needs; VK_NULL_HANDLE until made.
    VkDescriptorSetLayout SetLayout();
    VkDescriptorSetLayout EmptyLayout();
    /// The slot's set 3.
    VkDescriptorSet Set(uint32_t frameSlot);
    /// Whether the resources exist (a wind pipeline can be built and bound).
    bool Ready();

    /// The pre-pass's half, from render's Prepares: the read-backs, one model at a time.
    bool PrePassWanted();
    void PrePass(VkCommandBuffer cmd, uint64_t serial, uint64_t completed);

    // --- the API (native/Render.cpp) -------------------------------------------------------------
    void Watch(const char* const* substrings, uint32_t count);
    int  TakeModel(WXL_VkWindModel* out);
    int  SetModel(uint32_t id, const float* perVertex, uint32_t vertexCount, uint32_t vec4sPerVertex = 1);
    void SetParams(const WXL_VkWindParams* params);

    /// The draw path's counts: a draw drawn with a wind pipeline; one whose wind key could not be had.
    void Drawn();
    void FellBack(const char* why);
    int  Stats(WXL_VkWindStats* out);
    uint32_t Instances(WXL_VkWindInstance* out, uint32_t max);

    /// A pipeline was asked for with the wind bit and its vertex record has no variant (logged once per name).
    void NoVariant(const char* vertexShader);

    /// Counters for the device's stats lines.
    void LogStats();
}
