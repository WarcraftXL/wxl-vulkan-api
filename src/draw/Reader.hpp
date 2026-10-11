// Reads one engine draw off the CGxDevice into a draw::Cmd, at slot 42.
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

#include "device/Slots.hpp"
#include "draw/Cmd.hpp"
#include "vulkan/Texture.hpp"
#include "vulkan/Vk.hpp"

// The engine half of the draw path. Slot 42 (src/native/Draw.cpp) calls BuildCmd() once the state
// sync and the lazy creates and uploads have run, and BuildCmd() reads off the CGxDevice everything
// that draw binds -- the vertex and index bufs and their pools, the custom vertex layout, the two
// shader names, the texture stages the slot found bound, the fixed state with the master enables
// applied, the constant registers, the fixed-function register file or the companion selectors,
// the placement -- into one draw::Cmd, every engine object resolved to a Vulkan one through the
// Resolver.
//
// Render thread only. Nothing here allocates, throws, logs or makes a Vulkan call.
namespace wxlvk::draw
{
    /// How the reader resolves the engine's objects: two functions, a generation and two register
    /// files, all required. The resolver is also the native device's moment to stamp what the draw
    /// uses (a pool or an image named by a draw of the current frame is what the transfer rule
    /// keys on), since it sees exactly what the draw binds.
    struct Resolver
    {
        /// A CGxPool to the VkBuffer the draw binds whole (offset 0, addressed by the Cmd's
        /// vertexOffset / firstIndex as D3D addresses the pool). False: not drawable.
        bool (*pool)(const void* pool, VkBuffer& out);
        /// A CGxTex to its image. False: the stage counts as untextured.
        bool (*texture)(const void* tex, const wxlvk::vk::texture::Image*& image);
        /// Bumped by the native device whenever a CGxShader's validity, manifest record or
        /// address may have changed (a create, a destroy): the reader memoises what it read off
        /// the bound shaders -- the name, whether it is valid, its slice class -- per stage, and
        /// trusts the memo only while this is the value it was taken under.
        const uint32_t* shaderGeneration;
        /// The register files the draw's constants are read from: 256 rows of four floats each,
        /// vertex and pixel -- the native device's copy of what D3D's registers hold. Not the
        /// engine's shadows (gx::kVsConstCache / kPsConstCache): those are refilled with FLT_MAX at
        /// every frame start, the hardware registers are not, so a register the engine has not
        /// re-sent this frame keeps its last value under D3D.
        const float (*vertexRegs)[4];
        const float (*pixelRegs)[4];
    };

    /// The texture stages as the slot found them bound (render states kRsTexture0 + N), read
    /// once there -- the slot walks them anyway for ISetTexture's lazy create / upload -- and
    /// handed to the reader so it does not read the sixteen states again.
    struct Stages
    {
        uint32_t    mask;                       ///< bit N: stage N has a texture
        const void* tex[Cmd::kTextureStages];   ///< the CGxTex of each stage in `mask`
    };

    /**
     * Builds one Cmd from the device as it stands now.
     *
     * The viewport and scissor are left for the caller, which places them with D3D's own integer
     * rules against its target (src/native/Draw.cpp PlaceD3d).
     *
     * @param void* device : the CGxDevice, after IRsSync(0) and the lazy texture uploads
     * @param CGxBatch* batch : the slot's batch
     * @param int indexed : the slot's second argument
     * @param Resolver resolver : the native lookups
     * @param Stages stages : the bound texture stages, as the slot read them
     * @param Cmd& out : receives the draw; reset first (Reset())
     * @return bool ok : false when the geometry could not be resolved
     */
    bool BuildCmd(const void* device, const wxlvk::device::CGxBatch* batch, int indexed,
                  const Resolver& resolver, const Stages& stages, Cmd& out);
}
