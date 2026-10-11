// wxl-vulkan-api: the host's world image copied into the back buffer at the end of the world frame.
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

#include "host/Compose.hpp"

#include "host/State.hpp"
#include "native/Present.hpp"
#include "vulkan/Context.hpp"

#include "wxl/Script.hpp"

// The seam is the core's hook point "Gx.WorldOnRender" (CGWorldFrame's world pass), attached at a
// priority below the post-process chain's: this detour is outside post's, so it runs after post's
// BeforeUi chain returned. The host's image then replaces the whole world under the interface -- P0
// composes by copy; a depth-aware blend comes with the host's depth image.
//
// The copy is recorded into the frame's main command buffer, outside any pass. The GPU-side wait on
// the host's timeline is not recorded here: a wait belongs to a submit, and Submit.cpp adds it to the
// one that carries these commands.
namespace wxlvk::host
{
    namespace
    {
        namespace present = wxlvk::native::present;

        constexpr const char* kWorldScenePoint = "Gx.WorldOnRender";
        constexpr int         kPriority        = -10;   // outside post-process (0), inside wxl-host-client's camera (-20)

        using WorldSceneFn = void(__fastcall*)(void* worldFrame, void* edx);
        WorldSceneFn g_next = nullptr;

        /// The ownership transfer of a shared image between the host (EXTERNAL) and this queue.
        VkImageMemoryBarrier Ownership(VkImage image, uint32_t from, uint32_t to, VkAccessFlags src, VkAccessFlags dst)
        {
            VkImageMemoryBarrier b{};
            b.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            b.srcAccessMask       = src;
            b.dstAccessMask       = dst;
            b.oldLayout           = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
            b.newLayout           = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
            b.srcQueueFamilyIndex = from;
            b.dstQueueFamilyIndex = to;
            b.image               = image;
            b.subresourceRange    = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
            return b;
        }

        void Copy(VkCommandBuffer cmd, VkImage source, VkImage back, VkFormat backFormat, VkExtent2D backExtent)
        {
            const State& s = G();
            const VkExtent2D from = s.live.extent;
            if (s.live.format == backFormat && from.width == backExtent.width && from.height == backExtent.height)
            {
                VkImageCopy region{};
                region.srcSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
                region.dstSubresource = region.srcSubresource;
                region.extent         = { from.width, from.height, 1 };
                s.fn.cmdCopyImage(cmd, source, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, back,
                                  VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
                return;
            }
            VkImageBlit blit{};
            blit.srcSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
            blit.dstSubresource = blit.srcSubresource;
            blit.srcOffsets[1]  = { static_cast<int32_t>(from.width), static_cast<int32_t>(from.height), 1 };
            blit.dstOffsets[1]  = { static_cast<int32_t>(backExtent.width), static_cast<int32_t>(backExtent.height), 1 };
            s.fn.cmdBlitImage(cmd, source, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, back, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                              1, &blit, VK_FILTER_LINEAR);
        }

        /// Records the composite of `frame`. False when the device cannot take it this frame.
        bool Record(const WXL_VkHostFrame& frame)
        {
            State&          s   = G();
            VkCommandBuffer cmd = present::Outside();
            VkImage         back = VK_NULL_HANDLE;
            VkFormat        format{};
            VkExtent2D      extent{};
            if (!cmd || !present::BackBuffer(cmd, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, back, format, extent)) return false;
            const VkImage  source = s.live.images[frame.slot];
            const uint32_t family = vk::context::Get()->graphicsFamily;
            const VkImageMemoryBarrier acquire = Ownership(source, VK_QUEUE_FAMILY_EXTERNAL_KHR, family, 0, VK_ACCESS_TRANSFER_READ_BIT);
            s.fn.cmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0,
                                    nullptr, 1, &acquire);
            Copy(cmd, source, back, format, extent);
            const VkImageMemoryBarrier release = Ownership(source, family, VK_QUEUE_FAMILY_EXTERNAL_KHR, VK_ACCESS_TRANSFER_READ_BIT, 0);
            s.fn.cmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0, 0, nullptr, 0,
                                    nullptr, 1, &release);
            return true;
        }

        void Compose()
        {
            State& s = G();
            CollectRetired();
            if (s.pending && present::Serial() != s.live.lastSerial)
            {
                // The frame that recorded it ended without a submit (a failed frame): no wait, no
                // signal will happen, and the client must hand the slot back as not composed.
                s.pending = false;
                if (s.callbacks.FrameSubmitted) s.callbacks.FrameSubmitted(s.callbacks.user, &s.pendingFrame, 0);
            }
            if (!s.live.generation || !s.callbacks.SelectFrame || s.pending) return;
            if (!present::DrawPathUp() || !present::FrameOpen() || !present::OnBackBuffer())
            {
                ++s.stats.refused;
                return;
            }
            WXL_VkHostFrame frame{};
            if (!s.callbacks.SelectFrame(s.callbacks.user, &frame))
            {
                ++s.stats.skipped;
                return;
            }
            if (frame.slot >= s.live.slots || !Record(frame))
            {
                ++s.stats.refused;
                if (s.callbacks.FrameSubmitted) s.callbacks.FrameSubmitted(s.callbacks.user, &frame, 0);
                return;
            }
            s.pending       = true;
            s.pendingFrame  = frame;
            s.pendingHost   = s.live.hostTimeline;
            s.pendingClient = s.live.clientTimeline;
            s.live.lastSerial = present::Serial();
            s.stats.lastComposedSerial = s.live.lastSerial;
            ++s.stats.composed;
        }

        void __fastcall WorldScene(void* worldFrame, void* edx)
        {
            g_next(worldFrame, edx);
            Compose();
        }
    }

    bool AttachCompose()
    {
        const WXL_Api* api = wxl::ScriptMgr::Api();
        return api && api->HookAttachByName(kWorldScenePoint, reinterpret_cast<void*>(&WorldScene),
                                            reinterpret_cast<void**>(&g_next), kPriority) != 0;
    }
}
