// The native device's shaders: slots 68 ShaderCreate, 69 ShaderDestroy, 72 IShaderCreate.
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

#include "native/Slots.hpp"

#include "native/Engine.hpp"
#include "native/Render.hpp"
#include "shaders/Translated.hpp"
#include "vulkan/Manifest.hpp"
#include "vulkan/Pipelines.hpp"

#include "wxl/Script.hpp"
#include "wxl/game/Binding.hpp"
#include "wxl/offsets/engine/GxDevice.hpp"

#include <cstdint>

// The validity contract, read in the decompilation (cgxdevice 8.2-8.4):
//
//   D3D ShaderCreate 0x006AA130   base 0x006897C0 (hash lookup, IShaderLoad reads the .bls records),
//                                 then, for count == 1 and (*out)+0x30 == 0, the create of target
//                                 0 (vertex) or 4 (pixel) at once;
//   D3D IShaderCreate 0x006A5E10  target 0 -> IShaderCreateVertex, 4 -> IShaderCreatePixel, any
//                                 other target: nothing at all (not even +0x30);
//   IShaderCreateVertex/Pixel     +0x2C = 0; with the context (+0xF58): +0x30 = 1, and when the
//   0x006AA0D0 / 0x006AA070       record has code (+0x4C != 0) and CreateVertexShader/CreatePixel-
//                                 Shader succeeds: +0x20 = the API object, +0x2C = 1;
//   D3D ShaderDestroy 0x006AA190  for target 0 or 4 at reference count 1: release +0x20, +0x20 = 0,
//                                 +0x30 = 0; then the base decrements and deletes.
//
// CGxShader::Valid (0x00689A50) calls slot 72 while +0x30 is 0 and returns +0x2C; the UI, the fonts
// and the M2 model code check it and fall back to fixed function when it is 0, and a bound invalid
// shader is bound as NULL (IBindVertexShader / IBindPixelShader), which the draw reader reproduces.
//
// Native keeps every line of that and replaces "CreateXShader succeeded" with "this device can draw
// the bytes": the MD5 of +0x50 / +0x4C is known to the offline translator's manifest
// (shaders/translated/manifest.txt beside the DLL, through src/shaders/Translated.cpp), OR the name
// is one of the hand-written rows (the UI pairs, the particle pairs), which the pipeline builder
// draws whatever the manifest says. +0x20 holds the manifest record (null for a row-only shader):
// the draw reader takes the vertex slice class from it (src/draw/Reader.cpp). An unknown shader is
// invalid, so the engine's own fixed-function fallback runs.
//
// THE GENERATION (impl::g_shaderGeneration): the draw reader memoises, per stage, what it read off
// the bound CGxShader -- the name, +0x2C, the record's class -- keyed on the shader's address, and
// the draw filters of the rendering API cache their answer the same way. A create rewrites what
// they read and a destroy frees an address another shader may get next, so both bump the
// generation here (the filters' cache is simply forgotten); nothing else touches a CGxShader.
namespace
{
    namespace gxdev      = wxl::offsets::engine::gxdevice;
    namespace slots      = wxlvk::device::slots;
    namespace translated = wxlvk::shaders::translated;
    namespace pipelines  = wxlvk::vk::pipelines;

    using wxl::game::At;
    using wxlvk::native::Base;

    uint64_t g_creates;
    uint64_t g_valid;
    uint64_t g_byRow;      ///< valid through a hand-written row only
    uint64_t g_invalid;
    uint64_t g_noCode;

    bool Drawable(uint32_t target) { return target == gxdev::kShTargetVertex || target == gxdev::kShTargetPixel; }

    /// A CGxShader changed under the draw path's memos: the generation moves, the filters forget.
    void Changed()
    {
        ++wxlvk::native::impl::g_shaderGeneration;
        wxlvk::native::render::detail::ForgetFilterCache();
    }

    /// IShaderCreateVertex / IShaderCreatePixel with "can this device draw it" for the API create.
    void Create(void* device, void* shader)
    {
        Changed();
        At<uint32_t>(shader, gxdev::kShaderValid) = 0;
        if (!At<uint32_t>(device, gxdev::kContext)) return;
        At<uint32_t>(shader, gxdev::kShaderCreateTried) = 1;
        ++g_creates;
        if (At<uint32_t>(shader, gxdev::kShaderCodeSize) == 0)
        {
            ++g_noCode;
            return;
        }
        const bool pixel = At<uint32_t>(shader, gxdev::kShaderTarget) == gxdev::kShTargetPixel;
        const wxlvk::vk::manifest::Record* row = translated::Create(shader, pixel);
        const char* const name = At<const char*>(shader, gxdev::kShaderName);
        const bool handWritten = pipelines::HandWrittenRow(name, pixel);
        if (!row && !handWritten)
        {
            ++g_invalid;
            return;
        }
        At<const void*>(shader, gxdev::kShaderApiObject) = row;
        At<uint32_t>(shader, gxdev::kShaderValid)        = 1;
        ++g_valid;
        if (!row) ++g_byRow;
    }
}

namespace wxlvk::native::impl
{
    uint32_t g_shaderGeneration = 1;

    void __fastcall ShaderCreate(void* device, void* edx, CGxShader** out, uint32_t target,
                                 const char* directory, const char* name, uint32_t permutations)
    {
        Base<slots::kShaderCreate>()(device, edx, out, target, directory, name, permutations);
        if (permutations == 1 && out && *out && At<uint32_t>(*out, gxdev::kShaderCreateTried) == 0
            && Drawable(target))
            Create(device, *out);
    }

    void __fastcall ShaderDestroy(void* device, void* edx, CGxShader** shader)
    {
        if (shader && *shader && Drawable(At<uint32_t>(*shader, gxdev::kShaderTarget))
            && At<uint32_t>(*shader, gxdev::kShaderRefCount) == 1)
        {
            // The module belongs to the shader cache, not to the record: nothing to release. The
            // base is about to free the object, whose address may be a new shader's tomorrow.
            Changed();
            At<const void*>(*shader, gxdev::kShaderApiObject) = nullptr;
            At<uint32_t>(*shader, gxdev::kShaderCreateTried)  = 0;
        }
        Base<slots::kShaderDestroy>()(device, edx, shader);
    }

    void __fastcall IShaderCreate(void* device, void* /*edx*/, CGxShader* shader)
    {
        if (shader && Drawable(At<uint32_t>(shader, gxdev::kShaderTarget))) Create(device, shader);
    }

    void LogShaderStats()
    {
        wxl::ScriptMgr::Log(WXL_LOG_INFO,
                            "native: shaders: %llu create(s): %llu valid (%llu through a hand-written "
                            "row only), %llu invalid (unknown bytes: the engine's fixed-function "
                            "fallback runs), %llu with no code",
                            static_cast<unsigned long long>(g_creates),
                            static_cast<unsigned long long>(g_valid),
                            static_cast<unsigned long long>(g_byRow),
                            static_cast<unsigned long long>(g_invalid),
                            static_cast<unsigned long long>(g_noCode));
    }
}
