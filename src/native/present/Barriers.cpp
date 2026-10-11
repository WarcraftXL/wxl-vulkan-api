// Formats and image barriers.
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

#include "native/present/Internal.hpp"

namespace wxlvk::native::present::detail
{
    // --- formats, barriers ------------------------------------------------------------------------

    bool IsDepth(VkFormat f)
    {
        return f == VK_FORMAT_D16_UNORM || f == VK_FORMAT_X8_D24_UNORM_PACK32 || f == VK_FORMAT_D32_SFLOAT
               || f == VK_FORMAT_D16_UNORM_S8_UINT || f == VK_FORMAT_D24_UNORM_S8_UINT
               || f == VK_FORMAT_D32_SFLOAT_S8_UINT;
    }

    VkImageAspectFlags AspectOf(VkFormat f)
    {
        if (f == VK_FORMAT_D16_UNORM_S8_UINT || f == VK_FORMAT_D24_UNORM_S8_UINT || f == VK_FORMAT_D32_SFLOAT_S8_UINT)
            return VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT;
        return IsDepth(f) ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT;
    }

    /// The stages and accesses that use an image in a layout: a barrier's scope on either side.
    void Scope(VkImageLayout l, VkPipelineStageFlags& stage, VkAccessFlags& access)
    {
        switch (l)
        {
            case VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL:
                stage  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
                access = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
                return;
            case VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL:
                stage  = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
                access = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
                return;
            case VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL:
                // Compute too: the stages sample the back depth (src/stages/Chain.hpp).
                stage  = VK_PIPELINE_STAGE_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT |
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
                access = VK_ACCESS_SHADER_READ_BIT;
                return;
            case VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL:
                stage  = VK_PIPELINE_STAGE_TRANSFER_BIT;
                access = VK_ACCESS_TRANSFER_READ_BIT;
                return;
            case VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL:
                stage  = VK_PIPELINE_STAGE_TRANSFER_BIT;
                access = VK_ACCESS_TRANSFER_WRITE_BIT;
                return;
            case VK_IMAGE_LAYOUT_PRESENT_SRC_KHR:
                stage  = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
                access = 0;
                return;
            case VK_IMAGE_LAYOUT_UNDEFINED:
                stage  = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
                access = 0;
                return;
            default:
                stage  = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
                access = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
                return;
        }
    }



    /// One layout change, recorded at once (see Barriers::Add).
    void Barrier(VkCommandBuffer cmd, VkImage image, VkImageAspectFlags aspect, uint32_t levels, uint32_t layers,
                 VkImageLayout from, VkImageLayout to, bool discard)
    {
        Barriers one;
        one.Add(image, aspect, levels, layers, from, to, discard);
        one.Record(cmd);
    }
}
