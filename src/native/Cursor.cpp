// The native device's hardware cursor: slots 2 and 73, and slot 4's hardware branch.
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
#include "native/Native.hpp"

#include "wxl/Script.hpp"
#include "wxl/game/Binding.hpp"
#include "wxl/offsets/engine/GxDevice.hpp"

#include <windows.h>

#include <cstdint>
#include <cstring>

// With caps +0x310 = 1 and the format's hardware-cursor byte (+5, the gxCursor CVar)
// set, the base ICursorCreate sets +0x2954 and makes no software cursor texture; the engine then
// writes the 32x32 ARGB image at +0x2960 through CursorLock (74) and CursorUnlock (75, which stores
// the hotspot and sets +0x3B4C, "re-apply the cursor"). D3D's slot 4 turns those bits into the
// D3D cursor with SetCursorProperties, and its window procedure shows or hides that cursor on
// WM_SETCURSOR through IDirect3DDevice9::ShowCursor -- a branch WindowProcD3d skips when +0x397C is
// null, as it is here.
//
// Native does what SetCursorProperties amounts to on Windows: a 32-bit alpha cursor made with
// CreateIconIndirect from the same bits and hotspot, installed as the window's cursor. The window
// is subclassed in slot 13 (every format change makes a new one) for the one message D3D's procedure
// answers itself: WM_SETCURSOR over the client area sets this cursor, or none while the engine has
// it hidden (+0x2950). Everywhere else the class cursor. With +0x2954 0 -- gxCursor 0, or
// a format without a hardware cursor -- nothing here runs and slot 4 draws the software cursor.
namespace
{
    namespace gxdev = wxl::offsets::engine::gxdevice;
    namespace slots = wxlvk::device::slots;

    using wxl::game::At;
    using wxlvk::native::Base;
    using wxlvk::native::Own;

    constexpr int kCursorSide = 32;

    void*   g_device    = nullptr;   ///< the device whose window is subclassed
    HWND    g_hwnd      = nullptr;
    WNDPROC g_prevProc  = nullptr;
    HCURSOR g_cursor    = nullptr;   ///< built from +0x2960, owned here

    uint64_t g_builds;               ///< hardware cursors built from the engine's bits
    uint64_t g_buildFailures;
    uint64_t g_setCursorMessages;    ///< WM_SETCURSOR answered with the hardware cursor (or none)
    uint64_t g_visibleCalls;         ///< slot 73 calls with the hardware cursor in use

    bool Hardware(void* device)
    {
        return device && At<uint32_t>(device, gxdev::kCursorHardware) != 0;
    }

    bool Visible(void* device)
    {
        return At<uint32_t>(device, gxdev::kCursorVisible) != 0;
    }

    /// The cursor the client area shows now: ours while visible, none while the engine hides it.
    HCURSOR Shown(void* device)
    {
        return Visible(device) ? g_cursor : nullptr;
    }

    bool CursorInClient(HWND hwnd)
    {
        POINT p = {};
        RECT  r = {};
        if (!hwnd || !GetCursorPos(&p) || !ScreenToClient(hwnd, &p) || !GetClientRect(hwnd, &r)) return false;
        return p.x >= r.left && p.x < r.right && p.y >= r.top && p.y < r.bottom;
    }

    LRESULT CALLBACK CursorProc(HWND hwnd, UINT msg, WPARAM wp, LPARAM lp)
    {
        if (msg == WM_SETCURSOR && LOWORD(lp) == HTCLIENT && hwnd == g_hwnd && Hardware(g_device))
        {
            // D3D's WindowProcD3d: SetCursor(NULL), then ShowCursor(hardware && visible).
            ++g_setCursorMessages;
            SetCursor(Shown(g_device));
            return TRUE;
        }
        return CallWindowProcA(g_prevProc, hwnd, msg, wp, lp);
    }

    /// A 32-bit alpha cursor from the 32 x 32 ARGB block at +0x2960 (rows of 0x80 bytes, as D3D's
    /// slot 4 copies them) with the hotspot at +0x2958 / +0x295C.
    HCURSOR Build(void* device)
    {
        BITMAPV5HEADER bi = {};
        bi.bV5Size        = sizeof bi;
        bi.bV5Width       = kCursorSide;
        bi.bV5Height      = -kCursorSide;   // top-down, as the engine's rows
        bi.bV5Planes      = 1;
        bi.bV5BitCount    = 32;
        bi.bV5Compression = BI_BITFIELDS;
        bi.bV5RedMask     = 0x00FF0000u;
        bi.bV5GreenMask   = 0x0000FF00u;
        bi.bV5BlueMask    = 0x000000FFu;
        bi.bV5AlphaMask   = 0xFF000000u;

        HDC     dc   = GetDC(nullptr);
        void*   bits = nullptr;
        HBITMAP colour = CreateDIBSection(dc, reinterpret_cast<BITMAPINFO*>(&bi), DIB_RGB_COLORS, &bits, nullptr, 0);
        if (dc) ReleaseDC(nullptr, dc);
        if (!colour || !bits)
        {
            if (colour) DeleteObject(colour);
            return nullptr;
        }
        std::memcpy(bits, static_cast<const uint8_t*>(device) + gxdev::kCursorImage,
                    kCursorSide * kCursorSide * sizeof(uint32_t));
        // The mask is ignored for a cursor whose colour bitmap carries alpha, but must exist.
        HBITMAP mask = CreateBitmap(kCursorSide, kCursorSide, 1, 1, nullptr);

        ICONINFO ii = {};
        ii.fIcon    = FALSE;
        ii.xHotspot = At<uint32_t>(device, gxdev::kCursorHotspotX);
        ii.yHotspot = At<uint32_t>(device, gxdev::kCursorHotspotY);
        ii.hbmMask  = mask;
        ii.hbmColor = colour;
        HCURSOR cursor = reinterpret_cast<HCURSOR>(CreateIconIndirect(&ii));
        DeleteObject(colour);
        if (mask) DeleteObject(mask);
        return cursor;
    }
}

namespace wxlvk::native::impl
{
    // --- slot 2 ----------------------------------------------------------------------------------------

    void __fastcall ICursorCreate(void* device, void* edx, const CGxFormat* format)
    {
        // D3D 0x0068E900 minus its texture: the base decides +0x2954 (format byte +5 and caps
        // +0x310) and makes the software texture when there is no hardware cursor; with one, D3D
        // creates its cursor surface, marks the cursor dirty and draws it through slot 4.
        Base<slots::kICursorCreate>()(device, edx, format);
        if (!Hardware(device)) return;
        At<uint32_t>(device, gxdev::kD3dCursorDirty) = 1;
        Own<slots::kICursorDraw>(device)(device, edx);
    }

    // --- slot 73 ---------------------------------------------------------------------------------------

    void __fastcall CursorSetVisible(void* device, void* edx, int visible)
    {
        // D3D 0x0068E750: the base stores +0x2950; with the hardware cursor and a context, and the
        // mouse inside the client rect, D3D's ShowCursor(visible) -- here SetCursor.
        Base<slots::kCursorSetVisible>()(device, edx, visible);
        if (!Hardware(device) || !At<uint32_t>(device, gxdev::kContext)) return;
        ++g_visibleCalls;
        const HWND hwnd = At<HWND>(device, gxdev::kWindow);
        if (CursorInClient(hwnd)) SetCursor(Shown(device));
    }

    // --- slot 4's hardware branch, and the window ------------------------------------------------------

    void HardwareCursorDraw(void* device)
    {
        // D3D's slot 4, hardware path: only when +0x3B4C says the image or the hotspot changed.
        if (!At<uint32_t>(device, gxdev::kD3dCursorDirty)) return;
        At<uint32_t>(device, gxdev::kD3dCursorDirty) = 0;

        HCURSOR fresh = Build(device);
        if (!fresh)
        {
            ++g_buildFailures;
            if (g_buildFailures <= 4)
                wxl::ScriptMgr::Log(WXL_LOG_WARN, "native: cursor: CreateIconIndirect failed (%lu); the "
                                    "previous hardware cursor stays", static_cast<unsigned long>(GetLastError()));
            return;
        }
        const HCURSOR old = g_cursor;
        g_cursor = fresh;
        ++g_builds;
        const HWND hwnd = At<HWND>(device, gxdev::kWindow);
        if (CursorInClient(hwnd)) SetCursor(Shown(device));
        if (old) DestroyCursor(old);
    }

    void CursorWindowCreated(void* device, HWND hwnd)
    {
        if (!hwnd) return;
        const WNDPROC prev = reinterpret_cast<WNDPROC>(
            SetWindowLongPtrA(hwnd, GWLP_WNDPROC, reinterpret_cast<LONG_PTR>(&CursorProc)));
        if (!prev)
        {
            wxl::ScriptMgr::Log(WXL_LOG_WARN, "native: cursor: the window could not be subclassed (%lu); a "
                                "hardware cursor will show the class cursor instead",
                                static_cast<unsigned long>(GetLastError()));
            return;
        }
        g_device   = device;
        g_hwnd     = hwnd;
        g_prevProc = prev;
    }

    void CursorRelease()
    {
        // The window it was shown on is about to go (slots 11 and 13); the image is rebuilt from
        // +0x2960 when the next slot 2 marks the cursor dirty.
        if (g_cursor)
        {
            SetCursor(LoadCursorA(nullptr, reinterpret_cast<LPCSTR>(IDC_ARROW)));
            DestroyCursor(g_cursor);
            g_cursor = nullptr;
        }
        g_hwnd = nullptr;
    }

    void LogCursorStats()
    {
        wxl::ScriptMgr::Log(WXL_LOG_INFO,
                            "native: cursor: hardware %s; %llu hardware cursor(s) built (%llu failed), "
                            "%llu WM_SETCURSOR answered, %llu slot-73 call(s) with the hardware cursor",
                            Hardware(g_device) ? "in use" : "not in use (software cursor, or gxCursor 0)",
                            static_cast<unsigned long long>(g_builds),
                            static_cast<unsigned long long>(g_buildFailures),
                            static_cast<unsigned long long>(g_setCursorMessages),
                            static_cast<unsigned long long>(g_visibleCalls));
    }
}
