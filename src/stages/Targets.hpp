// The post-process chain's working images: two frame-sized storage images in VRAM, shared by every
// effect as a ping-pong pair.
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

#include "stages/Frame.hpp"

#include <cstdint>

// The stages' working image: one frame-sized storage image in VRAM, shared by every stage of a run
// (they all work in place on it).
//
// R8G8B8A8_UNORM, whose STORAGE support Vulkan 1.0 makes mandatory, so the stages run on every device
// without a format query and without STORAGE usage on the back buffer itself (which some adapters pay
// for with framebuffer compression on every draw of the frame). The back buffer's bytes are copied in
// as they are (a BGRA back buffer and this format are copy-compatible); Io::swapRB says which channel is
// which.
namespace wxlvk::stages::targets
{
    /// The working image's format.
    constexpr VkFormat kFormat = VK_FORMAT_R8G8B8A8_UNORM;

    /**
     * The working image at `extent`, in GENERAL layout, made (or re-made at a new extent) as needed. The
     * image it replaces is retired by the serial of the last frame that used it.
     *
     * @param VkCommandBuffer cmd : the frame's main buffer, outside a pass (a new image's layout change
     *        is recorded into it)
     * @param uint64 serial : the frame being recorded: the image is stamped with it
     * @param Image& out : receives it
     * @return bool ok : false when no memory could be had (logged by the allocator)
     */
    bool Get(VkExtent2D extent, VkCommandBuffer cmd, uint64_t serial, Image& out);

    /// Retires the working image once the frame `lastUsed` has completed (no stage needs it now).
    void Release(uint64_t lastUsed);

    /// Bytes of VRAM the working image holds now.
    uint64_t Bytes();
}
