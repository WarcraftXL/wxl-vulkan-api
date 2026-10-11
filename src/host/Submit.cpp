// wxl-vulkan-api: the timeline wait and signal added to the submit that carries a composite.
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

#include "host/Host.hpp"

#include "host/State.hpp"

// Two submits can carry a composite: the frame's own (native/Present.cpp Present) and a mid-frame
// flush (FlushAndWait) when the engine reads back after the world. Whichever comes first after the
// composite was recorded takes the wait and the signal; the existing binary semaphores keep their
// places in the arrays, with a timeline value of 0, which Vulkan ignores for a binary semaphore.
//
// The wait's stage is TRANSFER: the copy is a transfer. Every transfer of the batch waits with it --
// the frame's uploads included -- for a value the host has already submitted, so the delay is bounded
// by the host's GPU time for one frame, and nothing waits on the CPU.
namespace wxlvk::host
{
    void DecorateSubmit(VkSubmitInfo& submit, SubmitExtras& x)
    {
        State& s = G();
        x.used   = false;
        if (!s.pending || submit.waitSemaphoreCount > 1 || submit.signalSemaphoreCount > 1) return;

        uint32_t waits = 0, signals = 0;
        for (uint32_t i = 0; i < submit.waitSemaphoreCount; ++i, ++waits)
        {
            x.waits[waits]      = submit.pWaitSemaphores[i];
            x.stages[waits]     = submit.pWaitDstStageMask[i];
            x.waitValues[waits] = 0;
        }
        x.waits[waits]      = s.pendingHost;
        x.stages[waits]     = VK_PIPELINE_STAGE_TRANSFER_BIT;
        x.waitValues[waits] = s.pendingFrame.waitValue;
        ++waits;
        for (uint32_t i = 0; i < submit.signalSemaphoreCount; ++i, ++signals)
        {
            x.signals[signals]      = submit.pSignalSemaphores[i];
            x.signalValues[signals] = 0;
        }
        x.signals[signals]      = s.pendingClient;
        x.signalValues[signals] = s.pendingFrame.signalValue;
        ++signals;

        x.timeline                           = {};
        x.timeline.sType                     = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO_KHR;
        x.timeline.pNext                     = submit.pNext;
        x.timeline.waitSemaphoreValueCount   = waits;
        x.timeline.pWaitSemaphoreValues      = x.waitValues;
        x.timeline.signalSemaphoreValueCount = signals;
        x.timeline.pSignalSemaphoreValues    = x.signalValues;
        submit.pNext                = &x.timeline;
        submit.waitSemaphoreCount   = waits;
        submit.pWaitSemaphores      = x.waits;
        submit.pWaitDstStageMask    = x.stages;
        submit.signalSemaphoreCount = signals;
        submit.pSignalSemaphores    = x.signals;

        x.slot        = s.pendingFrame.slot;
        x.waitValue   = s.pendingFrame.waitValue;
        x.signalValue = s.pendingFrame.signalValue;
        x.used        = true;
        s.pending     = false;
    }

    void AfterSubmit(const SubmitExtras& x, bool ok)
    {
        if (!x.used) return;
        const State& s = G();
        if (!s.callbacks.FrameSubmitted) return;
        WXL_VkHostFrame frame{};
        frame.slot        = x.slot;
        frame.waitValue   = x.waitValue;
        frame.signalValue = x.signalValue;
        s.callbacks.FrameSubmitted(s.callbacks.user, &frame, ok ? 1 : 0);
    }
}
