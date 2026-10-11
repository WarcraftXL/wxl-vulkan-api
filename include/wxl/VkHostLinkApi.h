// The native device's host link: imports a world image another process renders and composes it.
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

#ifndef WXL_VK_HOST_LINK_API_H
#define WXL_VK_HOST_LINK_API_H

#include <stdint.h>

/*
 * The tiny interface between a host client extension (which owns the link to a 64-bit host process)
 * and the native device (which owns the GPU). Published as "wxl.vk.hostlink" by wxl-vulkan-api when it runs
 * the native device AND WXL_HOST is on; otherwise nothing is published, and nothing of the host link
 * exists in the device (no extension enabled, no wait or signal added to any submit).
 *
 * No Vulkan type crosses it: handles are the NT handles the host duplicated into this process, formats
 * and usages are the VkFormat / VkImageUsageFlags values as integers. The device does the Vulkan half:
 * the interop extensions at device creation, the imports, the GPU-side wait on the host's timeline,
 * the copy of the world image under the interface at the end of the world frame (the return of
 * "Gx.WorldOnRender", after the post-process chain), the signal of the client's timeline, and the
 * deferred destruction of everything imported.
 *
 * The device never waits on the CPU for the host and never waits on the GPU for a value the host has
 * not already submitted (the client hands the device only values the host published): a host that dies
 * leaves the device composing nothing, never hanging.
 *
 * Every function runs on the render thread (the engine's main thread).
 */

#ifdef __cplusplus
extern "C" {
#endif

#define WXL_VK_HOST_LINK_API_NAME    "wxl.vk.hostlink"
#define WXL_VK_HOST_LINK_API_VERSION 1

#define WXL_VK_SHARED_SLOTS 4

/** WXL_VkAdapterIds::interop bits. */
#define WXL_VK_INTEROP_MEMORY_WIN32    1u
#define WXL_VK_INTEROP_SEMAPHORE_WIN32 2u
#define WXL_VK_INTEROP_TIMELINE        4u
#define WXL_VK_INTEROP_IDS             8u

/** The device's adapter, as the host needs it to pick the same GPU. */
typedef struct WXL_VkAdapterIds
{
    uint32_t structSize;
    uint8_t  deviceUUID[16];
    uint8_t  driverUUID[16];
    uint8_t  deviceLUID[8];
    uint32_t luidValid;
    uint32_t vendorID;
    uint32_t deviceID;
    uint32_t driverVersion;
    uint32_t apiVersion;
    uint32_t interop;      ///< WXL_VK_INTEROP_*: what this device enabled
    char     name[64];
} WXL_VkAdapterIds;

/** One exported colour image, as the host describes it. */
typedef struct WXL_VkSharedImage
{
    uint64_t handle;          ///< NT handle in this process; the device closes it once imported
    uint64_t allocationSize;
    uint32_t format;          ///< VkFormat
    uint32_t width;
    uint32_t height;
    uint32_t usage;           ///< VkImageUsageFlags, identical to the exporter's
    uint32_t dedicated;       ///< non-zero: the memory is a dedicated allocation of the image
    uint32_t reserved;
} WXL_VkSharedImage;

/** One generation of the share: the slots' colour images and the two timelines. */
typedef struct WXL_VkSharedWorld
{
    uint32_t          structSize;
    uint32_t          generation;
    uint32_t          slotCount;      ///< at most WXL_VK_SHARED_SLOTS
    uint32_t          reserved;
    uint64_t          hostTimeline;   ///< NT handle: the host signals it when a slot is rendered
    uint64_t          clientTimeline; ///< NT handle: the device signals it when a composite is done
    WXL_VkSharedImage color[WXL_VK_SHARED_SLOTS];
} WXL_VkSharedWorld;

/** What one frame composes: filled by SelectFrame. */
typedef struct WXL_VkHostFrame
{
    uint32_t slot;          ///< the slot whose image is copied under the interface
    uint32_t reserved;
    uint64_t waitValue;     ///< the GPU waits hostTimeline >= this before the copy (an already submitted value)
    uint64_t signalValue;   ///< the device signals clientTimeline = this once the frame is done
} WXL_VkHostFrame;

/** The client's half, called by the device. */
typedef struct WXL_VkHostLinkCallbacks
{
    uint32_t structSize;
    void*    user;
    /**
     * At the composite point of a world frame: picks (and claims) the slot to compose.
     * @return non-zero with `out` filled to compose; zero to compose nothing this frame
     */
    int (__cdecl* SelectFrame)(void* user, WXL_VkHostFrame* out);
    /**
     * After the submit that carries the composite returned.
     * @param submitted  non-zero when the submit succeeded and will signal frame->signalValue
     */
    void (__cdecl* FrameSubmitted)(void* user, const WXL_VkHostFrame* frame, int submitted);
} WXL_VkHostLinkCallbacks;

typedef struct WXL_VkHostLinkStats
{
    uint32_t structSize;
    uint32_t imported;          ///< the generation imported, 0 when none
    uint64_t composed;          ///< frames composed
    uint64_t skipped;           ///< composite points with nothing selected
    uint64_t refused;           ///< composite points the device could not serve (no back buffer, a texture target)
    uint64_t lastComposedSerial;
    uint64_t completedSerial;   ///< frames finished on the GPU
} WXL_VkHostLinkStats;

typedef struct WXL_VkHostLinkApi
{
    uint32_t structSize;
    uint32_t apiVersion;

    /**
     * The device's adapter ids.
     * @return 1 once the device exists with every interop extension enabled; 0 before, or when the
     *         driver lacks one (the log says which)
     */
    int (__cdecl* GetAdapterIds)(WXL_VkAdapterIds* out);

    /// The back buffer's size, which the host renders at. 0 before the back buffer exists.
    int (__cdecl* GetFrameExtent)(uint32_t* width, uint32_t* height);

    /**
     * Imports one generation, replacing the one imported before (released once the GPU is done with
     * it). Closes every handle in `world`, imported or not.
     * @param why  receives the reason on failure (NUL-terminated within whyCap)
     * @return 1 on success
     */
    int (__cdecl* ImportWorld)(const WXL_VkSharedWorld* world, char* why, uint32_t whyCap);

    /// Stops composing and releases the imported generation once the GPU is done with it.
    void (__cdecl* ReleaseWorld)(void);

    /// Installs (or with NULL removes) the client's callbacks. Copied.
    void (__cdecl* SetCallbacks)(const WXL_VkHostLinkCallbacks* callbacks);

    /// The counters; set out->structSize first.
    void (__cdecl* GetStats)(WXL_VkHostLinkStats* out);
} WXL_VkHostLinkApi;

#ifdef __cplusplus
}
#endif

#endif // WXL_VK_HOST_LINK_API_H
