// The caps block (+0x214, 0x13C bytes): what the native device fills it with.
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

namespace wxlvk::native::caps
{
    /**
     * Fills the device's caps block for the native device. Slot 13 calls it where ICreateD3dDevice
     * calls ISetCaps: after the Vulkan device exists, before the UI shaders are created (their
     * profile folder is read from it) and before slot 2 (the cursor reads +0x310).
     *
     * The rule: start from the values D3D writes on an NVIDIA RTX-class adapter (the reference table
     * in Caps.cpp) and deviate only on purpose -- every deviation is a commented line in Fill.
     *
     * @param void* device : the CGxDevice
     */
    void Fill(void* device);
}
