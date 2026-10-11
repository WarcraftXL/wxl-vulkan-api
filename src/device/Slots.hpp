// The engine graphics device's 84 vtable slots: one row per slot, carrying its index, its name, the
// byte count its stock implementation pops, and its signature.
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

#include "wxl/offsets/engine/GxDevice.hpp"

#include <windows.h>

#include <cstdint>

// The engine's 84-slot CGxDevice vtable, slot by slot.
//
// Every slot is MSVC __thiscall: `this` in ECX, the arguments pushed right to left, and the CALLEE
// pops them (`ret N`, N = 4 * argument count). A free function reproduces that on x86 MSVC as
// `__fastcall(this, edx_dummy, args...)`, which is the shape every native slot is written in.
//
// That `ret N` is why the table below is written out slot by slot instead of being approximated. A
// thunk whose declared argument count disagrees with the callee's pop size moves ESP by the wrong
// amount on every call, and the crash lands wherever the corrupted frame is eventually used -- never
// at the slot that caused it. So each row states the pop size independently of the signature, and
// the static_asserts further down refuse to compile when the two disagree.
//
// Provenance of each column:
//   index, pop  read out of the binary for this project, not taken from the document: the four
//               device vtables (base 0x00A2DDC0, D3D9 0x00A2E718, D3D9Ex 0x00A2F500, OpenGL
//               0x00A2E198) were dumped from the Ghidra export's 09_program.bytes and each entry's
//               body scanned for its `ret` instruction. All four tables are exactly 84 entries long
//               -- the dword after slot 83 is string data in every one of them. 24 base entries are
//               __purecall (0x0040BAA5): slots 1, 17, 20, 21, 22, 35, 36, 52, 59, 60, 64-67, 71, 72
//               and 76-83.
//   name, args  the CGxDevice/CGxDeviceD3d reference, sections 1.0 and 1.2, cross-checked against
//               the Ghidra symbol name of each D3D9 entry. Where the reference marks a meaning
//               inferred rather than read, the row says so.
//
// Slot 0 gets its own warning because the obvious guess is wrong and the damage is silent: it is
// ITexMarkAsUpdated(CGxTex*), not a destructor. It is the engine's lazy texture-upload point,
// reached on every bind of a texture with a pending update. The destructor is slot 8.
namespace wxlvk::device
{
    namespace gxdev = wxl::offsets::engine::gxdevice;

    // Engine types the slots take by pointer. Incomplete on purpose: their fields are read through
    // the core's offsets (wxl/offsets/engine/GxDevice.hpp), and a layout guessed here would be a
    // second, unverified copy of sections 3, 4 and 8 of the reference.
    struct CGxTex;
    struct CGxPool;
    struct CGxBuf;
    struct CGxShader;
    struct CGxQuery;
    struct CGxFormat;
    struct CGxBatch;
    struct CGxGammaRamp;  ///< 0x600 bytes, 3 x 256 WORD (the device's copy is at +0x354)
    struct CImVector;     ///< 32-bit colour, bytes in memory B, G, R, A
    struct C44Matrix;     ///< 16 floats, row-major, row vectors
    struct CRect;         ///< 4 floats {minY, minX, maxY, maxX}
    struct CiRect;        ///< 4 ints, same order
    struct C2iVector;
    struct C2Vector;
    struct C3Vector;
    /// TSGrowableArray<T>, sized and filled by the read-back slots: CImVector for slot 20, float for 21.
    struct TSGrowableArray;

    /// What slots 29-34 add to and remove from. Slots 5-7 call them with no arguments at all:
    /// CGxDevice::NotifyOnDeviceRestored 0x006843B0 is `(**(code**)(this[0x63] + i * 4))()`.
    using DeviceNotifyFn = void(__cdecl*)();

    /// The `this` and the dead EDX every slot starts with.
    #define WXL_VK_SELF void* device, void* edx

    // --- the table ----------------------------------------------------------------------------------
    // X(index, Name, popBytes, ReturnType, (WXL_VK_SELF, args...))
    #define WXL_VK_DEVICE_SLOTS(X)                                                                      \
        /* internal: lazy texture upload, and the render-state flush --------------------------------*/ \
        X( 0, ITexMarkAsUpdated,               0x04, void,      (WXL_VK_SELF, CGxTex* tex))             \
        X( 1, IRsSendToHw,                     0x04, void,      (WXL_VK_SELF, uint32_t state))          \
        /* cursor, internal half --------------------------------------------------------------------*/ \
        X( 2, ICursorCreate,                   0x04, void,      (WXL_VK_SELF, const CGxFormat* format)) \
        X( 3, ICursorDestroy,                  0x00, void,      (WXL_VK_SELF))                          \
        X( 4, ICursorDraw,                     0x00, void,      (WXL_VK_SELF))                          \
        /* the three callback lists being fired (base only) -----------------------------------------*/ \
        X( 5, NotifyOnDeviceRestored,          0x00, void,      (WXL_VK_SELF))                          \
        X( 6, NotifyOnTextureRecreation,       0x00, void,      (WXL_VK_SELF))                          \
        X( 7, NotifyOnStereoChanged,           0x00, void,      (WXL_VK_SELF))                          \
        /* life cycle -------------------------------------------------------------------------------*/ \
        X( 8, ScalarDeletingDtor,              0x04, void*,     (WXL_VK_SELF, uint8_t flags))           \
        X( 9, DeviceCreateOnWindow,            0x08, int,       (WXL_VK_SELF, HWND window,              \
                                                                 const CGxFormat* format))              \
        X(10, DeviceCreate,                    0x08, int,       (WXL_VK_SELF, WNDPROC wndProc,          \
                                                                 const CGxFormat* format))              \
        X(11, DeviceDestroy,                   0x00, void,      (WXL_VK_SELF))                          \
        X(12, DeviceEvictResources,            0x00, void,      (WXL_VK_SELF))                          \
        X(13, DeviceSetFormat,                 0x04, int,       (WXL_VK_SELF, const CGxFormat* format)) \
        X(14, DeviceSetBaseMipLevel,           0x04, void,      (WXL_VK_SELF, uint32_t level))          \
        X(15, DeviceSetGammaRamp,              0x04, void,      (WXL_VK_SELF, const CGxGammaRamp* ramp))\
        X(16, DeviceSetGammaLevel,             0x04, void,      (WXL_VK_SELF, float level))             \
        X(17, DeviceWindow,                    0x00, HWND,      (WXL_VK_SELF))                          \
        /* screenshots and read-back ----------------------------------------------------------------*/ \
        X(18, DeviceTakeScreenShot,            0x00, void,      (WXL_VK_SELF))                          \
        X(19, DeviceReadScreenShot,            0x0C, void,      (WXL_VK_SELF, uint32_t* width,          \
                                                                 uint32_t* height, void** data))        \
        X(20, DeviceReadPixels,                0x08, void,      (WXL_VK_SELF, CiRect* rect,             \
                                                                 TSGrowableArray* out))                 \
        X(21, DeviceReadDepths,                0x08, void,      (WXL_VK_SELF, CiRect* rect,             \
                                                                 TSGrowableArray* out))                 \
        /* window messages, render targets, depth-stencil -------------------------------------------*/ \
        X(22, DeviceWM,                        0x0C, void,      (WXL_VK_SELF, uint32_t event,           \
                                                                 uintptr_t p1, uintptr_t p2))           \
        X(23, DeviceSetRenderTarget,           0x0C, void,      (WXL_VK_SELF, uint32_t buffer,          \
                                                                 CGxTex* tex, uint32_t face))           \
        X(24, CaptureDepthStencilSurface,      0x00, void,      (WXL_VK_SELF))                          \
        X(25, BindDepthStencilSurface,         0x00, void,      (WXL_VK_SELF))                          \
        X(26, DeviceResolveDepthBuffer,        0x04, void,      (WXL_VK_SELF, CGxTex* tex))             \
        X(27, DeviceCopyTex,                   0x10, void,      (WXL_VK_SELF, CGxTex* src,              \
                                                                 uint32_t srcFace, CGxTex* dst,         \
                                                                 uint32_t dstFace))                     \
        X(28, DeviceOverride,                  0x08, void,      (WXL_VK_SELF, int index,                \
                                                                 uint32_t value))                       \
        /* the three callback lists being edited (base only) -----------------------------------------*/\
        X(29, AddDeviceRestoredCallback,       0x04, void,      (WXL_VK_SELF, DeviceNotifyFn callback)) \
        X(30, RemoveDeviceRestoredCallback,    0x04, void,      (WXL_VK_SELF, DeviceNotifyFn callback)) \
        X(31, AddTextureRecreationCallback,    0x04, void,      (WXL_VK_SELF, DeviceNotifyFn callback)) \
        X(32, RemoveTextureRecreationCallback, 0x04, void,      (WXL_VK_SELF, DeviceNotifyFn callback)) \
        X(33, AddStereoChangedCallback,        0x04, void,      (WXL_VK_SELF, DeviceNotifyFn callback)) \
        X(34, RemoveStereoChangedCallback,     0x04, void,      (WXL_VK_SELF, DeviceNotifyFn callback)) \
        /* the window size the engine lays the UI out against ---------------------------------------*/ \
        X(35, CapsWindowSize,                  0x04, void,      (WXL_VK_SELF, CRect* out))              \
        X(36, CapsWindowSizeInScreenCoords,    0x04, void,      (WXL_VK_SELF, CRect* out))              \
        X(37, LogCrashInfo,                    0x08, void,      (WXL_VK_SELF, char* buffer,             \
                                                                 uint32_t size))                        \
        /* the frame --------------------------------------------------------------------------------*/ \
        X(38, ScenePresent,                    0x00, void,      (WXL_VK_SELF))                          \
        X(39, SceneClear,                      0x08, void,      (WXL_VK_SELF, uint32_t mask,            \
                                                                 uint32_t color))                       \
        X(40, XformSetProjection,              0x04, void,      (WXL_VK_SELF, const C44Matrix* matrix)) \
        X(41, XformSetView,                    0x04, void,      (WXL_VK_SELF, const C44Matrix* matrix)) \
        X(42, Draw,                            0x08, void,      (WXL_VK_SELF, CGxBatch* batch,          \
                                                                 int indexed))                          \
        /* immediate mode: base emulation on every stock backend bar OpenGL's 49 and 50 -------------*/ \
        X(43, PrimBegin,                       0x04, void,      (WXL_VK_SELF, uint32_t prim))           \
        X(44, PrimDrawElements,                0x00, void,      (WXL_VK_SELF))                          \
        X(45, PrimVertex,                      0x04, void,      (WXL_VK_SELF, const C3Vector* position))\
        X(46, PrimTexCoord,                    0x08, void,      (WXL_VK_SELF, uint32_t tmu,             \
                                                                 const C2Vector* coord))                \
        X(47, PrimNormal,                      0x04, void,      (WXL_VK_SELF, const C3Vector* normal))  \
        X(48, PrimColor,                       0x04, void,      (WXL_VK_SELF, const CImVector* color))  \
        /* 49 and 50 are inferred names: no-ops on D3D, glPointSize / glLineWidth on OpenGL --------*/  \
        X(49, PrimPointSize,                   0x04, void,      (WXL_VK_SELF, float size))              \
        X(50, PrimLineWidth,                   0x04, void,      (WXL_VK_SELF, float width))             \
        X(51, MasterEnableSet,                 0x08, void,      (WXL_VK_SELF, uint32_t enable, int on)) \
        /* pools and buffers ------------------------------------------------------------------------*/ \
        X(52, PoolSizeSet,                     0x08, void,      (WXL_VK_SELF, CGxPool* pool,            \
                                                                 uint32_t size))                        \
        X(53, PoolDestroy,                     0x04, void,      (WXL_VK_SELF, CGxPool* pool))           \
        X(54, BufLock,                         0x04, char*,     (WXL_VK_SELF, CGxBuf* buf))             \
        X(55, BufUnlock,                       0x08, int,       (WXL_VK_SELF, CGxBuf* buf,              \
                                                                 uint32_t bytesWritten))                \
        X(56, BufData,                         0x10, void,      (WXL_VK_SELF, CGxBuf* buf,              \
                                                                 const void* data, uint32_t size,       \
                                                                 uint32_t offset))                      \
        /* textures: 57 is base-only, and creates no API object (creation is lazy, slot 0) ----------*/ \
        X(57, TexCreate,                       0x2C, int,       (WXL_VK_SELF, uint32_t target,          \
                                                                 uint32_t width, uint32_t height,       \
                                                                 uint32_t depth, uint32_t format,       \
                                                                 uint32_t dataFormat, uint32_t flags,   \
                                                                 void* userArg, gxdev::TexFillFn fill,  \
                                                                 const char* name, CGxTex** out))       \
        X(58, TexDestroy,                      0x04, void,      (WXL_VK_SELF, CGxTex* tex))             \
        X(59, TexCopy,                         0x18, bool,      (WXL_VK_SELF, CGxTex* tex,              \
                                                                 C2iVector* dstPos, C2iVector* srcPos,  \
                                                                 C2iVector* size, uint32_t level,       \
                                                                 uint32_t face))                        \
        X(60, TexStretch,                      0x14, bool,      (WXL_VK_SELF, CGxTex* tex,              \
                                                                 CiRect* dst, CiRect* src,              \
                                                                 uint32_t level, uint32_t face))        \
        /* 61 is an inferred name: a byte count, no-op on D3D9, stored at +0x3C44 on D3D9Ex --------*/  \
        X(61, TexSetCacheSize,                 0x04, void,      (WXL_VK_SELF, uint32_t bytes))          \
        /* queries ----------------------------------------------------------------------------------*/ \
        X(62, QueryCreate,                     0x08, void,      (WXL_VK_SELF, CGxQuery** out,           \
                                                                 uint32_t type))                        \
        X(63, QueryDestroy,                    0x04, void,      (WXL_VK_SELF, CGxQuery** query))        \
        X(64, QueryBegin,                      0x04, bool,      (WXL_VK_SELF, CGxQuery* query))         \
        X(65, QueryEnd,                        0x04, uint32_t,  (WXL_VK_SELF, CGxQuery* query))         \
        X(66, QueryGetParam,                   0x0C, int,       (WXL_VK_SELF, CGxQuery* query,          \
                                                                 uint32_t param, uint32_t* out))        \
        X(67, QueryGetData,                    0x08, bool,      (WXL_VK_SELF, CGxQuery* query,          \
                                                                 void* out))                            \
        /* shaders: 68 and 71 take a DIRECTORY and a NAME, not a blob and a length -----------------*/  \
        X(68, ShaderCreate,                    0x14, void,      (WXL_VK_SELF, CGxShader** out,          \
                                                                 uint32_t target,                       \
                                                                 const char* directory,                 \
                                                                 const char* name,                      \
                                                                 uint32_t permutations))                \
        X(69, ShaderDestroy,                   0x04, void,      (WXL_VK_SELF, CGxShader** shader))      \
        X(70, ShaderConstantsSet,              0x10, void,      (WXL_VK_SELF, uint32_t target,          \
                                                                 uint32_t reg, const float* values,     \
                                                                 uint32_t vec4Count))                   \
        X(71, IShaderReload,                   0x0C, int,       (WXL_VK_SELF, CGxShader* shader,        \
                                                                 const char* directory,                 \
                                                                 const char* name))                     \
        X(72, IShaderCreate,                   0x04, void,      (WXL_VK_SELF, CGxShader* shader))       \
        /* cursor, public half ----------------------------------------------------------------------*/ \
        X(73, CursorSetVisible,                0x04, void,      (WXL_VK_SELF, int visible))             \
        X(74, CursorLock,                      0x00, uint32_t*, (WXL_VK_SELF))                          \
        X(75, CursorUnlock,                    0x08, void,      (WXL_VK_SELF, uint32_t hotspotX,        \
                                                                 uint32_t hotspotY))                    \
        /* stereo: the getters return in ST0 / AL --------------------------------------------------*/  \
        X(76, StereoSetConvergence,            0x04, void,      (WXL_VK_SELF, float convergence))       \
        X(77, StereoGetConvergence,            0x00, float,     (WXL_VK_SELF))                          \
        X(78, StereoSetSeparation,             0x04, void,      (WXL_VK_SELF, float separation))        \
        X(79, StereoGetSeparation,             0x00, float,     (WXL_VK_SELF))                          \
        X(80, StereoEnabled,                   0x00, bool,      (WXL_VK_SELF))                          \
        /* the three with no name and no caller: pure in the base, nullsubs in all three stock      */  \
        /* backends (0x00653A10 `ret 8`, 0x005EEB70 `ret 0`, 0x00653A10 again). The pop size is the */  \
        /* only thing known about them, so the arguments are typed as untouched dwords.            */  \
        X(81, Unknown81,                       0x08, void,      (WXL_VK_SELF, uint32_t p1, uint32_t p2))\
        X(82, Unknown82,                       0x00, void,      (WXL_VK_SELF))                          \
        X(83, Unknown83,                       0x08, void,      (WXL_VK_SELF, uint32_t p1, uint32_t p2))

    /// How many entries a stock table has. The core owns the number; this is where the two agree.
    constexpr unsigned kSlotCount = gxdev::kVTableSlots;

    /// Slot indices by name: `slots::kDraw`, `slots::kITexMarkAsUpdated`, ...
    namespace slots
    {
        #define WXL_VK_SLOT_ENUM(index, name, pop, ret, args) k##name = index,
        enum : unsigned
        {
            WXL_VK_DEVICE_SLOTS(WXL_VK_SLOT_ENUM)
        };
        #undef WXL_VK_SLOT_ENUM
    }

    /// One function-pointer type per slot: `ITexMarkAsUpdatedFn`, `DrawFn`, ...
    #define WXL_VK_SLOT_TYPEDEF(index, name, pop, ret, args) using name##Fn = ret(__fastcall*) args;
    WXL_VK_DEVICE_SLOTS(WXL_VK_SLOT_TYPEDEF)
    #undef WXL_VK_SLOT_TYPEDEF

    /**
     * How many stack arguments a slot signature declares, i.e. everything after `this` and the dead
     * EDX, and how many bytes the callee therefore has to pop.
     *
     * This exists so the pop size in each row of the table can be checked against the signature next
     * to it at compile time rather than trusted. Every argument any of the 84 slots takes is one
     * dword on x86 -- pointer, int, float, or a CImVector by value -- so `arguments * 4` is exactly
     * the `ret N` the stock implementation uses, and a disagreement is a stack desync waiting for
     * the first call.
     */
    template <typename Fn>
    struct Arity;

    template <typename Ret, typename... Args>
    struct Arity<Ret(__fastcall*)(void*, void*, Args...)>
    {
        static constexpr unsigned kArgs = sizeof...(Args);
        static constexpr unsigned kPopBytes = sizeof...(Args) * 4u;
    };

    /// Slot index -> its function type, pop size and name, for code that is generic over slots.
    template <unsigned Slot>
    struct SlotTraits;

    #define WXL_VK_SLOT_TRAITS(index, name, pop, ret, args)                                            \
        template <>                                                                                    \
        struct SlotTraits<index>                                                                       \
        {                                                                                              \
            using Fn = name##Fn;                                                                       \
            static constexpr unsigned kIndex = index;                                                  \
            static constexpr unsigned kPopBytes = pop;                                                 \
            static constexpr const char* kName = #name;                                                \
        };                                                                                             \
        static_assert(Arity<name##Fn>::kPopBytes == (pop),                                             \
                      "slot " #index " " #name ": its declared arguments do not add up to its ret N");
    WXL_VK_DEVICE_SLOTS(WXL_VK_SLOT_TRAITS)
    #undef WXL_VK_SLOT_TRAITS

    /// The slot names, indexed by slot, for a log line that has to name one.
    #define WXL_VK_SLOT_NAME(index, name, pop, ret, args) #name,
    constexpr const char* kSlotNames[] = { WXL_VK_DEVICE_SLOTS(WXL_VK_SLOT_NAME) };
    #undef WXL_VK_SLOT_NAME

    /// The byte count each slot's stock implementation pops, indexed by slot.
    #define WXL_VK_SLOT_POP(index, name, pop, ret, args) pop,
    constexpr unsigned kSlotPopBytes[] = { WXL_VK_DEVICE_SLOTS(WXL_VK_SLOT_POP) };
    #undef WXL_VK_SLOT_POP

    static_assert(sizeof kSlotNames / sizeof *kSlotNames == kSlotCount,
                  "the slot table has a different number of rows than the engine's vtable has entries");
    static_assert(sizeof kSlotPopBytes / sizeof *kSlotPopBytes == kSlotCount, "pop table length");

    #undef WXL_VK_SELF
}
