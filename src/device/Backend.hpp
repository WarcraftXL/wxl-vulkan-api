// The device factory the core calls in place of the engine's own CGxDevice::New*.
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

namespace wxlvk::device
{
    /**
     * Registers the native device's factory with the core's gx-device seam.
     *
     * Called only when WXL_VK_DEVICE=native. A core built without the seam, or one that refuses the
     * registration, is logged; GxDevCreate then builds the engine's own backend.
     *
     * @return bool registered
     */
    bool Install();
}
