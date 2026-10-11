// The native device's rendering API: lets another extension record Vulkan work into the frame.
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

#ifndef WXL_VK_RENDER_API_H
#define WXL_VK_RENDER_API_H

#include <stdint.h>

// The consumer brings no Vulkan SDK either: the headers are the ones wxl-vulkan-api vendors
// (third_party/vulkan-headers), and every entry point comes from GetDeviceInfo's
// vkGetDeviceProcAddr. VK_NO_PROTOTYPES keeps the header from declaring functions nobody links.
#ifndef VK_NO_PROTOTYPES
#define VK_NO_PROTOTYPES
#endif
#include <vulkan/vulkan_core.h>

/*
 * wxl-vulkan-api publishes this interface with WarcraftXL's PublishInterface when it runs the native
 * device (WXL_VK_DEVICE=native); with the stock D3D9 device nothing is published and GetInterface
 * returns NULL. Extensions load in name order, so a consumer resolves it after load (its first
 * OnUpdate or OnWorldEnter), not in its WXL_Load.
 *
 * What a consumer gets:
 *
 *   - INSERTION POINTS. A pass registered at a point is called every time the engine's frame reaches
 *     it, with a context (WXL_VkRenderContext) describing the frame, the open render pass and the
 *     scene. Each point has two phases:
 *
 *       Prepare  recorded into the frame's PRE-PASS command buffer, outside any render pass. The GPU
 *                runs it before every command of the frame's main buffer -- before anything the engine
 *                draws this frame -- so it is the place for compute (culling, generation), fills and
 *                copies. It cannot read anything the frame renders.
 *       Record   recorded into the frame's main command buffer, INSIDE the render pass the engine's
 *                world draws into, at the moment the engine reaches the point.
 *
 *       MidPass  (appended to version 1) recorded into the frame's main command buffer at the point,
 *                OUTSIDE any render pass: the device has ended the engine's pass, and resumes it (with
 *                LOAD) before the Records. ctx->depthView is THIS frame's depth as the engine has drawn
 *                it so far -- before any consumer's Record at the point -- for compute that tests
 *                against it (a second, current-frame occlusion test). Costs a pass break, made only
 *                while a pass with a MidPass is registered or the previous depth is asked for.
 *
 *     After every Prepare of a point, and after the MidPasses, the device records one barrier making
 *     compute-shader and transfer writes visible to indirect-draw, vertex-input, vertex/fragment-shader
 *     and compute reads.
 *
 *   - RESOURCES through the device's allocator (address space is the scarce resource of this 32-bit
 *     process): buffers and 2D images, retired by frame serial.
 *
 *   - RAW DEVICE ACCESS: the VkDevice, vkGetDeviceProcAddr and the device's VkPipelineCache, for
 *     shader modules, pipelines, layouts, descriptor pools and query pools of the consumer's own.
 *
 *   - DRAW FILTERS: suppress engine draws by shader name.
 *
 *   - THE PREVIOUS FRAME'S DEPTH, on request: a sampled copy of the opaque world's depth as the last
 *     frame left it, with that frame's camera, for work that runs before this frame's world is drawn
 *     (occlusion culling against a Hi-Z pyramid built in Prepare).
 *
 * The rules a consumer follows:
 *
 *   1. Everything runs on the render thread (the engine's main thread). No call into this table and no
 *      Vulkan call on device objects from any other thread.
 *   2. Never submit, wait or present: no vkQueueSubmit, vkQueueWaitIdle, vkDeviceWaitIdle,
 *      vkWaitForFences on the device's work. The device owns the queue.
 *   3. In Record, stay inside the pass: no vkCmdBeginRenderPass / vkCmdEndRenderPass /
 *      vkCmdPipelineBarrier with image layout changes on the device's attachments. Bind and set what
 *      the draws need -- pipeline, descriptor sets, viewport and scissor (dynamic in every device
 *      pipeline), push constants, vertex and index buffers. The device forgets what it had bound when
 *      the callback returns, so the engine's next draw binds everything again.
 *   4. A resource a command names is in use until that frame's serial completes
 *      (ctx->completedSerial >= serial). Retire through RetireBuffer / RetireImage with the last serial
 *      that named it; destroy own objects (pipelines, pools) only once that serial completed.
 *   5. Per-frame data lives per frame slot (ctx->frameSlot): when a frame opens, the previous frame of
 *      the same slot has completed, so the slot's buffers are free to overwrite on the CPU.
 *   6. A point may fire more than once in a frame (ctx->invocation counts); a consumer whose Prepare
 *      writes per-slot data handles only the first.
 *   7. The pipelines a Record draws with are built against ctx->renderPass, which stays valid for the
 *      device's life; it is the open pass's compatibility class (colour and depth format), so keep
 *      one pipeline per distinct handle seen.
 *
 * Matrices are the engine's: 16 floats, row-major, for the row-vector convention clip = v * M. Copied
 * as they are into a std140 / std430 mat4, GLSL's `M * v` computes exactly that product. Clip space is
 * D3D's (depth 0..1, +y up); every device vertex shader negates gl_Position.y, and a consumer's must
 * too, because the device's viewports are y-down.
 */

#ifdef __cplusplus
extern "C" {
#endif

#define WXL_VK_RENDER_API_NAME    "wxl.vk.render"
#define WXL_VK_RENDER_API_VERSION 1

/// Where in the engine's frame a pass runs. Points are appended, never renumbered.
typedef enum WXL_VkRenderPoint
{
    /**
     * The engine's opaque world is drawn -- terrain, map objects, the opaque model pass and the
     * ground-effect (detail doodad) pass -- and its transparent model pass, liquids, weather and the
     * interface are not. The open pass is the back buffer with the world's depth. Fires only while
     * the engine draws the world into the back buffer.
     */
    WXL_VK_POINT_AFTER_OPAQUE_WORLD = 0,

    /**
     * (Appended to version 1.) Inside one shadow cascade's depth-only pass, while ReplayWorld draws
     * the frame's captured world into it: a Record callback here draws the consumer's own casters
     * (grass, trees) with ctx->scene's viewProj -- the light's -- into ctx->renderPass (the shadow
     * pass class, depth only, GetShadowPass), ctx->viewport the cascade's area. No Prepare, no
     * MidPass at this point; ctx->invocation is the cascade index.
     */
    WXL_VK_POINT_SHADOW_CASTERS = 1,

    /**
     * (Appended to version 1.) Stages only (RegisterStage): the world is drawn -- the transparents,
     * the liquids, the weather and the engine's own full-screen passes included -- and the world
     * text and the interface are not. RegisterPass refuses it.
     */
    WXL_VK_POINT_BEFORE_UI = 2,

    /**
     * (Appended to version 1.) Stages only (RegisterStage; RegisterPass refuses it): the engine's late liquid
     * pass has returned (the water and the ocean, its ripples). With the camera in no liquid the transparent
     * models UNDER the water (CM2Scene pass 2) are drawn before it, and the weather and the transparent models
     * over the water (pass 1) after it; with the camera in a liquid, pass 1 and the weather are drawn before
     * it and pass 2 after. The water writes the scene depth on this device (it offers no clip planes, so the
     * water material turns its depth write on): a stage that reads the depth here finds the water's surface
     * where the water is -- the end of a ray through the air. The stage context's scene and viewport are the
     * world's as AfterOpaqueWorld saw them (as at BeforeUi). Fires once per world frame, only while the world
     * draws into the back buffer. A device older than this point refuses a stage registered at it: register
     * at AfterOpaqueWorld then.
     */
    WXL_VK_POINT_AFTER_LIQUIDS = 3,

    WXL_VK_POINT_COUNT
} WXL_VkRenderPoint;

/// Memory classes for CreateBuffer.
typedef enum WXL_VkMemoryClass
{
    WXL_VK_MEMORY_DEVICE_LOCAL = 0, ///< not mapped: the GPU's own (compute outputs, uploads' destination)
    WXL_VK_MEMORY_HOST         = 1, ///< mapped, host-visible: staging and small read-backs
    WXL_VK_MEMORY_STREAM       = 2, ///< mapped, written by the CPU every frame and read by the GPU
} WXL_VkMemoryClass;

/**
 * The scene as the device sees it at the point, in std140-compatible rows (every member 16 bytes,
 * every matrix four rows), so a consumer can copy it whole into a uniform buffer.
 */
typedef struct WXL_VkSceneConstants
{
    float view[16];       ///< world -> view: the engine's world camera (the View stack where none is set)
    float projection[16]; ///< view -> clip: the projection the engine sent the device (D3D depth 0..1)
    float viewProj[16];   ///< view * projection
    float cameraPos[4];   ///< world-space eye position, w = 1
    float fogColor[4];    ///< rgb 0..1; a = 1 while the world's fog is on
    /// start, end (view-space depth), 1 / (end - start), exponent: the world shaders' fog factor is
    /// saturate((end - z) / (end - start)) ^ exponent, 1 unfogged
    float fogRange[4];
    float viewport[4];    ///< pass width, height, 1 / width, 1 / height
    float time[4];        ///< seconds since the device came up, seconds since the previous frame, 0, 0
} WXL_VkSceneConstants;

/**
 * The previous frame's opaque scene depth (GetPreviousDepth). Appended to version 1.
 *
 * The depth is copied at AfterOpaqueWorld after the Prepares and before every consumer's MidPass and
 * Record -- terrain, map objects, the opaque models and the engine's ground clutter drawn; no
 * consumer's geometry, transparents, liquids, weather or the engine's full-screen passes -- as raw
 * depth values: d in [depthMin, depthMax], the viewport depth range the world was drawn
 * with (0 .. 0.94), D3D's convention (0 near). With `projection` (row vectors, D3D depth 0 .. 1),
 * z' = (d - depthMin) / (depthMax - depthMin) is that matrix's clip z / w.
 *
 * Readable from any command recorded after the call that returned it, the pre-pass's included: its
 * writes were made visible to the compute, vertex and fragment stages when it was made. It is in
 * GENERAL layout, and the device leaves it there; a consumer samples it (a combined image sampler or
 * texelFetch, layout GENERAL) and never changes its layout or writes it. The image is replaced at a
 * size change: take `imageView` again from every call, never keep it across frames.
 */
typedef struct WXL_VkPreviousDepth
{
    uint32_t      structSize;   ///< the caller sets it; the device fills only what it covers
    /// 1 when the copy is of the frame right before the one being recorded, at the back buffer's
    /// current size, and the camera did not cut in between (an eye jump past 10 yards or a turn past
    /// 60 degrees). 0 on the first frame, after a resize, a loading screen or a cut: the copy is
    /// there but describes another view, and a consumer should not cull with it.
    uint32_t      valid;
    uint64_t      serial;       ///< the frame the copy was made in
    VkImage       image;
    VkImageView   imageView;
    VkFormat      format;       ///< VK_FORMAT_R32_SFLOAT
    VkImageLayout layout;       ///< VK_IMAGE_LAYOUT_GENERAL
    VkExtent2D    extent;       ///< the back buffer's at that frame
    float         view[16];     ///< that frame's world -> view (WXL_VkSceneConstants::view)
    float         projection[16];
    float         viewProj[16];
    float         depthMin;     ///< the world's viewport depth range that frame
    float         depthMax;
    /// The world's viewport that frame as its draws had it: pixels, D3D9's pixel-centre shift and the
    /// jitter included (`jitter`, in pixels, is the part that is the jitter: 0, 0 when none was asked for).
    VkViewport    viewport;
    float         jitter[2];
} WXL_VkPreviousDepth;

/// What a pass is called with. Valid only for the duration of the call.
typedef struct WXL_VkRenderContext
{
    uint32_t structSize;
    uint32_t point;            ///< a WXL_VkRenderPoint
    uint64_t serial;           ///< the frame being recorded; stamp what its commands name with it
    uint64_t completedSerial;  ///< every frame up to this one has finished on the GPU
    uint32_t frameSlot;        ///< 0 .. frameSlots - 1
    uint32_t frameSlots;
    uint32_t invocation;       ///< times this point already fired in this frame (0 the first)
    uint32_t hasDepth;         ///< the pass has a depth attachment (Record)
    VkCommandBuffer cmd;       ///< Prepare: the pre-pass buffer, no pass open. Record: the main buffer, pass open
    VkRenderPass    renderPass;///< the open pass's compatibility class; VK_NULL_HANDLE in Prepare
    VkExtent2D      extent;    ///< the pass's render area (Prepare: the back buffer's)
    VkFormat        colorFormat;
    VkFormat        depthFormat;   ///< VK_FORMAT_UNDEFINED without depth
    /// MidPass: a sampled view (depth aspect) of this frame's scene depth, in depthViewLayout. Prepare
    /// and Record: VK_NULL_HANDLE (the depth is the open pass's attachment; GetPreviousDepth has last
    /// frame's).
    VkImageView     depthView;
    const WXL_VkSceneConstants* scene;
    /// Record: an occlusion query of the device's own is active in the command buffer (the engine's
    /// sun glare), so a pass must not begin one of its own (Vulkan allows one active query per type).
    /// Appended to version 1: read it only when structSize covers it.
    uint32_t        occlusionQueryActive;
    /// Record: the viewport and scissor an engine draw made now would get -- the engine's viewport
    /// in pixels with D3D9's pixel-centre shift, and its DEPTH RANGE (the world may be drawn into
    /// part of 0..1). Geometry that must depth-test against the world sets exactly this viewport.
    /// viewportValid is 0 when the engine's rectangle is empty. Appended to version 1: read only
    /// when structSize covers it.
    VkViewport      viewport;
    VkRect2D        scissor;
    uint32_t        viewportValid;
    /// MidPass: depthView's layout (VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL), readable by the
    /// compute, vertex and fragment stages; never change it. Appended to version 1.
    VkImageLayout   depthViewLayout;
} WXL_VkRenderContext;

/// A pass's callbacks. Copied at registration; `user` is handed back to both.
typedef struct WXL_VkRenderPassDesc
{
    uint32_t    structSize;
    void*       user;
    /// Outside a render pass, on the pre-pass buffer (see the header comment). May be NULL.
    void (__cdecl* Prepare)(void* user, const WXL_VkRenderContext* ctx);
    /// Inside the pass, at the point. May be NULL.
    void (__cdecl* Record)(void* user, const WXL_VkRenderContext* ctx);
    /// Outside the pass, on the main buffer, at the point, before the Records (see the header
    /// comment). May be NULL. Appended to version 1: set structSize to sizeof the struct with it.
    void (__cdecl* MidPass)(void* user, const WXL_VkRenderContext* ctx);
} WXL_VkRenderPassDesc;

/// The device, for a consumer's own Vulkan objects. Fixed for the process once the device is up.
typedef struct WXL_VkDeviceInfo
{
    uint32_t                  structSize;
    VkInstance                instance;
    VkPhysicalDevice          physicalDevice;
    VkDevice                  device;
    uint32_t                  queueFamily;        ///< the one queue's family (graphics and compute)
    uint32_t                  frameSlots;
    PFN_vkGetInstanceProcAddr getInstanceProcAddr;
    PFN_vkGetDeviceProcAddr   getDeviceProcAddr;
    VkPipelineCache           pipelineCache;      ///< the device's persisted cache; VK_NULL_HANDLE when it has none
} WXL_VkDeviceInfo;

typedef struct WXL_VkBufferDesc
{
    VkDeviceSize       size;
    VkBufferUsageFlags usage;
    uint32_t           memory;   ///< a WXL_VkMemoryClass
} WXL_VkBufferDesc;

typedef struct WXL_VkBufferInfo
{
    VkBuffer     buffer;
    VkDeviceSize size;
    void*        mapped;         ///< the CPU pointer for the mapped classes, NULL for DEVICE_LOCAL
} WXL_VkBufferInfo;

/// A single-sampled 2D image, OPTIMAL tiling, born in VK_IMAGE_LAYOUT_UNDEFINED. Its layouts are
/// the consumer's to track and change (outside a pass).
typedef struct WXL_VkImageDesc
{
    VkFormat           format;
    uint32_t           width;
    uint32_t           height;
    uint32_t           mipLevels;   ///< 0 is taken as 1
    VkImageUsageFlags  usage;
    VkImageAspectFlags aspect;      ///< for the view; 0 is taken as COLOR
} WXL_VkImageDesc;

typedef struct WXL_VkImageInfo
{
    VkImage     image;
    VkImageView view;               ///< every level, one layer
} WXL_VkImageInfo;

/*
 * TREE WIND (appended to version 1). The engine's M2 draws whose stage-0 texture path matches a
 * consumer's patterns (WatchTextures) are grouped by model -- the pool buffer and base vertex they
 * draw from -- and, once a model's batches have stopped changing, its vertices and indices are read
 * back once and handed to the consumer (TakeWindModel). The consumer classifies and analyses it and
 * hands back one vec4 per vertex (SetModelWind): trunk, branch and leaf amplitudes in model units,
 * and a branch phase 0..1. From then on that model's draws use the vertex shader's `wind` variant,
 * which reads the vec4 by vertex and the wind of SetWind, and bends the model in its own space before
 * the engine's transform. A draw whose vertex shader has no wind variant (the skinned ones) is drawn
 * as before. Nothing of this runs while no consumer has called WatchTextures.
 */

/// One batch (one engine draw) of a watched model.
typedef struct WXL_VkWindBatch
{
    uint32_t    firstIndex;   ///< into WXL_VkWindModel::indices
    uint32_t    indexCount;
    uint32_t    alphaTested;  ///< 1: an alpha-keyed or alpha-tested texture (leaves), 0: opaque (bark)
    uint32_t    blend;        ///< EGxBlend
    const char* texture;      ///< its stage-0 texture's path ("" when the engine gave none)
} WXL_VkWindBatch;

/// A watched model's mesh, as its draws read it. Valid until the next TakeWindModel or SetModelWind.
typedef struct WXL_VkWindModel
{
    uint32_t               structSize;   ///< the caller sets it
    uint32_t               id;           ///< for SetModelWind
    uint32_t               vertexCount;
    uint32_t               vertexStride; ///< bytes; the position is three floats at offset 0
    const uint8_t*         vertices;
    uint32_t               indexCount;
    const uint16_t*        indices;      ///< 0 .. vertexCount - 1, triangle lists, per batch
    uint32_t               batchCount;
    const WXL_VkWindBatch* batches;
    const char*            vertexShader; ///< the draws' vertex shader name ("Diffuse_T1:1")
} WXL_VkWindModel;

/// The wind the variant reads, set once a frame (it applies from the next frame's draws on).
typedef struct WXL_VkWindParams
{
    uint32_t structSize;
    uint32_t pad[3];
    float    viewInv[16];  ///< view -> world, row vectors (WXL_VkSceneConstants convention)
    float    view[16];     ///< world -> view
    float    wind[4];      ///< direction xy (unit), strength, gust frequency (1 / yd)
    float    phase[4];     ///< gust phase xy, ripple phase zw (noise units)
    float    shape[4];     ///< ripple, flutter, the noise anchor xy (world)
    float    time[4];      ///< clock (s, wraps every 60), strength scale, on (1), the branches' sway (1 as designed)
    /// The artist's scales, 1 as designed: the canopy's sway, the edge rustle, the leaves' texture
    /// shimmer, the canopy's occlusion (a leaf deep in the canopy darker; the wind variant's pixel
    /// half). Appended: a shorter struct leaves them 1.
    float    look[4];
    /// (Appended.) Towards the sun xyz (world), the leaves' translucency (0 none).
    float    sun[4];
    /// (Appended.) The direct light's colour rgb, 0.
    float    sunColour[4];
    /// (Appended.) The far crossfade to a consumer's impostors: from (yd from the eye), to, on (1), 0. A
    /// tree's fragment between them is dropped by a screen-space dither (interleaved gradient noise of
    /// the pixel, kept where it is >= the share smoothstep(from, to, distance)), the impostor keeping the
    /// complement; past `to` the tree draws nothing.
    float    fade[4];
    /// (Appended.) The terrain's wind modifier around the eye: the world x, y of the grid's corner at
    /// its lowest x and y, yards a cell, cells a side (WXL_VK_WIND_GRID_EDGE; 0: no grid -- every tree
    /// takes the global wind).
    float    gridInfo[4];
    /// (Appended.) The grid, row-major along +x then +y: speed scale (1 as the field), the deflected
    /// way xy (world, unit), 1 where known.
    float    grid[256][4];
} WXL_VkWindParams;

/// The wind grid's edge (WXL_VkWindParams::grid, 16 x 16 cells).
#define WXL_VK_WIND_GRID_EDGE 16u

/// One windy tree instance drawn this frame (GetWindInstances): its model and its model -> view rows.
typedef struct WXL_VkWindInstance
{
    uint32_t model;       ///< the id TakeWindModel handed out
    uint32_t bone;        ///< whose rows these are (0 for a static model; a skinned one's first real bone)
    uint32_t pad[2];
    float    rows[12];    ///< three rows, model -> view, as the draw's c31..c33 (or its one bone)
} WXL_VkWindInstance;

/// Tree wind's counters (GetWindStats): the last full frame's draws and the models' states.
typedef struct WXL_VkWindStats
{
    uint32_t    structSize;
    uint32_t    models, collecting, windyModels, rejected;
    uint32_t    windyDraws;   ///< drawn with a wind pipeline, the last full frame
    uint32_t    fallbacks;    ///< windy draws whose wind pipeline could not be had (drawn still), the last frame
    const char* fallbackWhy;  ///< the last such reason ("" none)
    uint32_t    noVariant;    ///< vertex shaders met with no wind variant
    uint32_t    ready, paramsSet, dataKiB;
    float       wind[4];      ///< the wind block as last set
    float       time[4];
} WXL_VkWindStats;

/* --- stages: the frame as an image, at a point, outside any pass (appended to version 1) --------- */

#define WXL_VK_STAGE_NEEDS_DEPTH   0x1u   ///< the stage reads the scene depth (ctx->depthView)
#define WXL_VK_STAGE_WRITES_COLOR  0x2u   ///< the stage writes the frame (ctx->colorView): it is copied back

/**
 * What a stage runs with. The device ends the engine's pass, copies the back buffer into a working
 * image (when any stage of the point writes colour), runs the point's stages in their order, copies the
 * image back, and the engine's next draw resumes its pass with LOAD.
 * Valid only during the call.
 */
typedef struct WXL_VkStageContext
{
    uint32_t        structSize;
    uint32_t        point;            ///< WXL_VK_POINT_AFTER_OPAQUE_WORLD, _AFTER_LIQUIDS or _BEFORE_UI
    uint64_t        serial;
    uint64_t        completedSerial;
    uint32_t        frameSlot;
    uint32_t        frameSlots;
    VkCommandBuffer cmd;              ///< the frame's main buffer, outside any pass
    VkExtent2D      extent;           ///< the frame's
    /// The frame: a storage image in GENERAL layout, R8G8B8A8_UNORM holding the back buffer's bytes
    /// as they are -- with swapRB set, .r is blue. Every earlier write to it is visible to compute. It
    /// is made SAMPLED too: a stage may read it through a sampler (in GENERAL), for filtered taps.
    VkImage         colorImage;
    VkImageView     colorView;
    VkFormat        colorFormat;
    VkImageLayout   colorLayout;
    uint32_t        swapRB;
    /// The scene depth (SHADER_READ_ONLY_OPTIMAL, depth aspect) for a stage that asked for it;
    /// VK_NULL_HANDLE when it cannot be had this frame.
    VkImageView     depthView;
    VkFormat        depthFormat;
    VkImageLayout   depthLayout;
    /// The world's scene and viewport (pixel rect with D3D9's centre shift, the world's DEPTH RANGE)
    /// as AfterOpaqueWorld saw them (at every point); sceneValid 0 when the engine had none.
    const WXL_VkSceneConstants* scene;
    VkViewport      viewport;
    uint32_t        sceneValid;
    /// (Appended: read when structSize covers it.) The sub-pixel jitter `viewport` carries, in pixels: the
    /// world's draws were offset by it this frame (RequestJitter); 0, 0 when the world was not jittered.
    float           jitter[2];
} WXL_VkStageContext;

/// A stage's description, copied at registration.
typedef struct WXL_VkStageDesc
{
    uint32_t structSize;
    void*    user;
    uint32_t point;      ///< WXL_VK_POINT_AFTER_OPAQUE_WORLD, _AFTER_LIQUIDS (appended) or _BEFORE_UI
    /// Where in the point's chain; lower runs first. The device has none of its own: the consumers agree on
    /// their orders (for instance ambient occlusion before a deferred relight, the relight before the fog,
    /// anti-aliasing last).
    int32_t  order;
    uint32_t flags;      ///< WXL_VK_STAGE_*
    /// Whether the stage runs this frame (asked once per point per frame). May be NULL (always).
    int  (__cdecl* Wanted)(void* user);
    void (__cdecl* Run)(void* user, const WXL_VkStageContext* ctx);
} WXL_VkStageDesc;

/// One stage's figures (GetStageStats). Appended to version 1.
typedef struct WXL_VkStageStats
{
    uint32_t    structSize;    ///< the caller sets it; the device fills only what it covers
    uint32_t    point;         ///< the stage's WXL_VkRenderPoint
    uint64_t    runs;          ///< times its Run was called
    /// GPU time of its latest run, its mean and its maximum since the device's last stats line, in
    /// milliseconds; -1 when the queue has no timestamps.
    float       lastMs;
    float       avgMs;
    float       maxMs;
    /// The point's whole latest run (every stage of it and the frame's copies), in milliseconds; -1
    /// unknown.
    float       pointMs;
    /// Why the point's latest run did or did not happen ("running", "idle (nothing enabled)", ...).
    const char* pointStatus;
} WXL_VkStageStats;

/* --- shadows: the frame's world replayed from a light (appended to version 1) ------------------- */

/**
 * The device's depth-only pass class for the replay: one D32 (or D16) attachment, CLEAR on load,
 * STORE, UNDEFINED -> SHADER_READ_ONLY_OPTIMAL, its end made visible to fragment and compute reads.
 * A consumer makes its framebuffers against it (layers 1, its own depth image's view).
 */
typedef struct WXL_VkShadowPass
{
    uint32_t     structSize;
    VkRenderPass renderPass;
    VkFormat     depthFormat;
    /// (Appended: filled when structSize covers it.) The same class with LOAD on load and
    /// SHADER_READ_ONLY_OPTIMAL -> SHADER_READ_ONLY_OPTIMAL: a replay with WXL_VK_REPLAY_LOAD draws into
    /// what the image holds (a cached map accumulating casters). Compatible with renderPass: the same
    /// framebuffers and pipelines serve both.
    VkRenderPass loadPass;
} WXL_VkShadowPass;

#define WXL_VK_REPLAY_ALPHA_TESTED 0x1u   ///< also the alpha-tested casters (leaves, cards): their textures are read
/// The alpha-tested casters drawn as opaque (no texture read): a far cascade's tree canopies as solid
/// shapes, at a fraction of the cost. Ignored with WXL_VK_REPLAY_ALPHA_TESTED.
#define WXL_VK_REPLAY_ALPHA_AS_OPAQUE 0x2u
/// Only the moving casters: the draws of a unit's model (a character's, a creature's or an item's
/// texture: players, creatures, mounts, what they carry -- whether or not they use the bone palette)
/// and the tree-wind draws; the terrain, the map objects and the doodads, animated or not, are left out.
#define WXL_VK_REPLAY_DYNAMIC_ONLY 0x4u
/// The movers are left out: the units' draws (a cached map of what does not move; an animated doodad
/// is in it, in its captured pose).
#define WXL_VK_REPLAY_NO_SKINNED 0x8u
/// The tree-wind draws are left out.
#define WXL_VK_REPLAY_NO_WIND 0x10u
/// The SHADOW_CASTERS passes are not called (the consumers' own casters: grass, which moves).
#define WXL_VK_REPLAY_NO_CASTERS 0x20u
/// Draws into what the image holds: WXL_VkShadowPass::loadPass, nothing cleared, the area only sets the
/// viewport and the scissor. The image must be in SHADER_READ_ONLY_OPTIMAL (a pass's end leaves it so).
#define WXL_VK_REPLAY_LOAD 0x40u
/// The tree-wind draws at rest: their sway is left out (a cached map does not follow the wind).
#define WXL_VK_REPLAY_WIND_STILL 0x80u
/// The terrain alone: the map objects, the doodads and the movers are left out (a consumer's view of the
/// ground under everything: a snow layer lying on it). A device without this flag replays the
/// whole static world (the flag is appended: a consumer checks nothing, it only gets more).
#define WXL_VK_REPLAY_TERRAIN_ONLY 0x100u
/// With WXL_VK_REPLAY_DYNAMIC_ONLY: also the world's STATIC M2 doodads (a Diffuse_ model draw that is no unit's
/// and no tree-wind draw: lamps, posts, fences, crates, signs, wagons, rocks, trunks), their opaque draws alone --
/// an alpha-tested doodad draw (a crown's leaves, a card) stays out even with WXL_VK_REPLAY_ALPHA_TESTED: a
/// consumer that has the crowns from a bake does not want them twice. The terrain and the map objects stay out.
/// A device without this flag ignores it (the flag is appended): the units alone, as DYNAMIC_ONLY says.
#define WXL_VK_REPLAY_WITH_DOODADS 0x200u
/// Culling (appended): a captured draw whose bounding sphere lies wholly outside the light's frustum --
/// lightViewProj's four sides and its far plane; never its near plane, so a caster between the light and the near plane
/// is kept as the depth clip would treat it -- is left out. The bounds are taken at capture time: a model's (a unit's, a
/// doodad's) origin in view space from its model-view rows and a generous radius from its scale and its bones' spread;
/// a draw without one (the terrain; a map object's batch drawn outside the engine's group renderers) is always kept. A map
/// object's batch has one: its group's box (MOGP's) through the batch's own transform. A device without the
/// flag ignores it (every draw replayed, as before); WXL_VkReplayStats::culled says how many it left out.
#define WXL_VK_REPLAY_CULL_FRUSTUM 0x400u
/// With WXL_VK_REPLAY_DYNAMIC_ONLY: also the MAP OBJECTS' draws (the WMOs: every `MapObj` vertex family, exterior and
/// interior groups), opaque, and alpha-tested with WXL_VK_REPLAY_ALPHA_TESTED (a railing, a grate) -- a near map that holds
/// the buildings round the player (the roofs' shadows at that map's texel rather than a bake's), or a local light's
/// map. Culled by their group's box with WXL_VK_REPLAY_CULL_FRUSTUM. The terrain stays out. A device without
/// this flag ignores it (the flag is appended): WXL_VkReplayStats::mapObjects then stays unwritten (its structSize says so).
#define WXL_VK_REPLAY_WITH_MAPOBJECTS 0x800u
/// With WXL_VkReplayDesc::excludeBoxes (appended): the boxes KEEP instead of leaving out -- a captured draw that is
/// no unit's and no map object's is replayed only when its bound's centre lies inside one of them (the units' own draws and
/// the map objects' are unaffected). A local light's live faces: the units, and the weapons and sheaths an unskinned model
/// draws (which the mover test misses), drawn where the units stand. A device without the boxes ignores the flag.
#define WXL_VK_REPLAY_BOXES_KEEP 0x1000u

/// One replay into one cascade (ReplayWorld).
typedef struct WXL_VkReplayDesc
{
    uint32_t        structSize;
    VkCommandBuffer cmd;            ///< outside any pass: a MidPass's ctx->cmd at AfterOpaqueWorld
    VkFramebuffer   framebuffer;    ///< against WXL_VkShadowPass::renderPass
    VkRect2D        area;           ///< the cascade's rectangle in it (cleared, viewport, scissor)
    /// world -> the light's clip (D3D depth 0 .. 1, row vectors, as the engine's matrices).
    float           lightViewProj[16];
    float           depthBias;      ///< constant, in the depth format's units (vkCmdSetDepthBias)
    float           slopeBias;
    uint32_t        cascade;        ///< handed to the SHADOW_CASTERS passes as ctx->invocation
    uint32_t        flags;          ///< WXL_VK_REPLAY_*
    /// (Appended: read when structSize covers it.) Captured draws of fewer indices (vertices when not
    /// indexed) are left out: the small clutter a far cascade cannot resolve. 0 keeps every draw.
    uint32_t        minIndexCount;
    /// (Appended: read when structSize covers it.) A sphere in WORLD space (xyz, radius; radius <= 0: none):
    /// a captured draw with a bound (WXL_VK_REPLAY_CULL_FRUSTUM's) lying wholly outside it is left out -- a local
    /// light's reach. Tested whether or not the flag is set.
    float           cullSphere[4];
    /// (Appended: read when structSize covers it.) `excludeBoxCount` WORLD boxes, six floats each (min xyz, then
    /// max xyz): a captured draw that is NO UNIT'S AND NO MAP OBJECT'S (a static M2 doodad's, a game object's) whose bound's
    /// CENTRE (WXL_VK_REPLAY_CULL_FRUSTUM's: a static model's origin) lies inside one of them is left out (counted in
    /// WXL_VkReplayStats::excluded) -- or, with WXL_VK_REPLAY_BOXES_KEEP, is the only kind kept. What a CACHED map must not
    /// hold: a local light's own model (the brazier's bowl, the lantern's cage, the bonfire's logs: the light inside them),
    /// and what is a unit's but was not drawn as one (a weapon or a sheath an unskinned model draws: the mover test misses it,
    /// and it trailed through a cube's cache as the unit moved). A draw without a bound is never left out. NULL or 0: none.
    const float*    excludeBoxes;
    uint32_t        excludeBoxCount;
} WXL_VkReplayDesc;

/// What ReplayWorld did.
typedef struct WXL_VkReplayStats
{
    uint32_t structSize;
    uint32_t captured;     ///< draws captured this frame (the opaque world so far)
    uint32_t replayed;     ///< drawn into this cascade
    uint32_t skipped;      ///< captured draws without a shadow pipeline (no variant, a failed build)
    uint32_t alphaTested;  ///< of replayed, with the pixel module's alpha test
    uint32_t casters;      ///< SHADOW_CASTERS passes called
    uint32_t culled;       ///< (appended) left out by WXL_VK_REPLAY_CULL_FRUSTUM or cullSphere: their bounds outside
    uint32_t mapObjects;   ///< (appended) of replayed, map objects' draws (WXL_VK_REPLAY_WITH_MAPOBJECTS); a device
                           ///< that writes it (its structSize covers it) knows the flag
    uint32_t excluded;     ///< (appended) left out by WXL_VkReplayDesc::excludeBoxes (or not kept, with
                           ///< WXL_VK_REPLAY_BOXES_KEEP); a device that writes it knows the boxes
} WXL_VkReplayStats;

/**
 * SEVERAL VIEWS OF ONE REPLAY (ReplayWorldViews, appended: a local light's shadow cube, its six faces
 * in one pass). What ReplayWorld does once per view -- the same draws kept by the same flags and boxes, each
 * view culled by its own matrix and sphere, drawn in the same order with the same bias -- in ONE walk of the capture and ONE
 * render pass: the views' depth is exactly what one ReplayWorld per view would have written. The pass is the LOAD class
 * (WXL_VkShadowPass::loadPass): the image must be in SHADER_READ_ONLY_OPTIMAL, nothing outside the views' areas is touched, and
 * a view with `clear` set has its area cleared to the far plane inside the pass before anything is drawn (no render pass of its
 * own). The SHADOW_CASTERS passes are called per view after its draws unless WXL_VK_REPLAY_NO_CASTERS.
 */
#define WXL_VK_REPLAY_MAX_VIEWS 32

/// One view of ReplayWorldViews.
typedef struct WXL_VkReplayView
{
    VkRect2D area;               ///< its rectangle of the framebuffer (viewport, scissor; cleared when `clear`)
    float    lightViewProj[16];  ///< world -> its clip (D3D depth 0 .. 1, row vectors), as WXL_VkReplayDesc's
    float    cullSphere[4];      ///< a WORLD sphere (xyz, radius; radius <= 0: none), as WXL_VkReplayDesc's
    uint32_t cascade;            ///< handed to the SHADOW_CASTERS passes as ctx->invocation
    uint32_t clear;              ///< non-zero: its area cleared to the far plane first
} WXL_VkReplayView;

/// ReplayWorldViews' description: WXL_VkReplayDesc's shared fields, and the views.
typedef struct WXL_VkReplayViewsDesc
{
    uint32_t                structSize;
    VkCommandBuffer         cmd;              ///< outside any pass: a MidPass's ctx->cmd at AfterOpaqueWorld
    VkFramebuffer           framebuffer;      ///< against WXL_VkShadowPass::renderPass (its LOAD class is compatible)
    float                   depthBias;
    float                   slopeBias;
    uint32_t                flags;            ///< WXL_VK_REPLAY_* (WXL_VK_REPLAY_LOAD is implied)
    uint32_t                minIndexCount;
    const float*            excludeBoxes;     ///< as WXL_VkReplayDesc's, for every view
    uint32_t                excludeBoxCount;
    const WXL_VkReplayView* views;
    uint32_t                viewCount;        ///< 1 .. WXL_VK_REPLAY_MAX_VIEWS
    uint32_t                viewStride;       ///< the caller's sizeof(WXL_VkReplayView)
} WXL_VkReplayViewsDesc;

/* --- a consumer's screen-space surface (appended to version 1) ---------------------------------- */

/**
 * A screen-space surface a consumer drew this frame and hands to other consumers (PublishSurface /
 * GetSurface): a grass renderer publishes its grass's, from its MidPass at AfterOpaqueWorld, before its
 * composite and before any stage of the point (an ambient occlusion pass) runs.
 *
 * The grass surface (`view`, RGBA8, one sample, the frame's extent) is premultiplied by the
 * grass's coverage, A being the coverage itself: R the share of the screen-space AO the grass keeps
 * there (most at a blade's root, nearly none on its tips), G its own baked AO, B the height along the
 * blade (0 root .. 1 tip). Where A is 0 there is no grass. An ambient occlusion reader keeps mix(1, R / A, A)
 * of its occlusion on such a pixel. `velocityView` (RG16F, VK_NULL_HANDLE when not written) is the grass's
 * screen motion this frame in pixels (+y down), premultiplied by the coverage likewise.
 *
 * Valid for the frame `serial` only, in `layout` (SHADER_READ_ONLY_OPTIMAL), readable by the fragment
 * and compute stages from the publisher's MidPass on; never written by a reader.
 */
typedef struct WXL_VkSurface
{
    uint32_t      structSize;
    uint32_t      flags;          ///< 0 (reserved)
    uint64_t      serial;         ///< the frame it describes
    VkImageView   view;
    VkImageLayout layout;
    VkFormat      format;
    VkExtent2D    extent;
    VkImageView   velocityView;   ///< VK_NULL_HANDLE: no motion vectors
    VkFormat      velocityFormat;
} WXL_VkSurface;

/* --- the trees' canopy density, for volumetric consumers (appended to version 1) ------------------ */

/**
 * One tree instance's canopy volume (WXL_VkCanopyVolumes::instances): its baked leaf density
 * (an offline bake), placed in the world. `volumeFromWorld` (row-major 3 x 4) maps a world point to the
 * volume's uvw, 0..1 inside it (cell-centred: a slice's texels are centred at (i + 0.5) / edge); outside
 * 0..1 there are no leaves of it. In the atlas, slice k (0..edge-1, along w) of this volume is the
 * rectangle x in [k * edge, (k + 1) * edge), y in [slab[0], slab[0] + slab[1]) of the normalised atlas
 * (so the texel at uv is x = (k + u) * edge / atlasWidth, y = slab[0] + v * slab[1]): a consumer filters
 * along w itself, mixing two slices. A texel of value t (R8_UNORM) is an extinction of t * slab[3]
 * per world yard. `boundsMin` / `boundsMax` (xyz, w 0) are the volume's world box, for culling.
 */
typedef struct WXL_VkCanopyInstance
{
    float volumeFromWorld[12];
    float slab[4];        ///< the volume's first row (normalised atlas y), its height (normalised), its edge (slices), extinction per yard at 1.0
    float boundsMin[4];
    float boundsMax[4];
} WXL_VkCanopyInstance;

/**
 * The trees' canopy density this frame (PublishCanopy / GetCanopy): a tree renderer publishes the tree
 * instances drawn near the eye that have a baked canopy, from its Prepare, before any recording of the
 * frame. `atlas` (2D, `format` R8_UNORM, `extent`) holds every published volume's slices side by side
 * (WXL_VkCanopyInstance); it is in `layout` (SHADER_READ_ONLY_OPTIMAL) and readable by the fragment and
 * compute stages from AfterOpaqueWorld on, for the frame `serial`. `instances` (`count` of them) is the
 * device's copy, valid until the next publish.
 */
typedef struct WXL_VkCanopyVolumes
{
    uint32_t                    structSize;
    uint32_t                    flags;      ///< 0 (reserved)
    uint64_t                    serial;     ///< the frame it describes
    VkImageView                 atlas;
    VkImageLayout               layout;
    VkFormat                    format;
    VkExtent2D                  extent;
    uint32_t                    count;
    const WXL_VkCanopyInstance* instances;
} WXL_VkCanopyVolumes;

/* --- the G-buffer: the opaque world's surfaces, from the device's `gbuf` module variants (appended) ---- */

/**
 * While a consumer asks for it (RequestGBuffer, counted), the opaque world's pass on the back buffer --
 * from the world scene pass's first draw to AfterOpaqueWorld -- carries three more colour attachments,
 * and every opaque or alpha-keyed world draw whose translated modules have a `gbuf` variant
 * (the offline translator's, the manifest's `gbuf` lines) writes them beside the frame:
 *
 *   albedo  R8G8B8A8_UNORM     rgb: the textures' colour before any light and fog, in the engine's own
 *                              (gamma) space -- the colour the engine's light term multiplies. a: the
 *                              material class (WXL_VK_GBUFFER_CLASS_*) in bits 0..4; bit 5 set for a batch
 *                              of an interior map object group (WXL_VK_GBUFFER_INTERIOR_BIT, the row p[14].y
 *                              the device writes inside the engine's group renderers); bit 6 set for a unit's
 *                              draw (a player, a creature, what they carry: WXL_VK_GBUFFER_MOVER_BIT, from
 *                              the row the draw reader writes for every draw); bit 7 set when the
 *                              vertex stage found the engine's light term plain -- the direct and
 *                              ambient bands alone, no tint, no vertex colour, no added term, no point
 *                              light -- so a consumer may relight the pixel from its albedo exactly.
 *   normal  R16G16B16A16_SNORM xyz: the surface normal in VIEW space (through the engine's world x view
 *                              rows), unit. w: the DEPTH TAG (below), positive when the module
 *                              had a normal, negative when it had none (an unlit permutation; xyz are
 *                              zero) -- `w > 0.5` still reads "has a normal". Another 16-bit format when
 *                              the device cannot attach this one: read normalFormat.
 *   light   R8G8B8A8_UNORM     rgb: the engine's light term as it multiplies the albedo -- the vertex
 *                              stage's color0: tint x saturate(ambient + direct x N.L) + the added term,
 *                              or the vertex colour x that, saturated (the terrain's and the map objects'
 *                              doubled, as their pixel stages double it -- clipped at 1, so a map object's
 *                              vertex colour brighter than the engine's neutral half saturates here, never
 *                              in the albedo). a: the terrain's baked shadow factor (MCSH, 0.7 .. 1), 1
 *                              elsewhere; the albedo is written WITHOUT it.
 *
 * So for a pixel of a class above 0: engine colour = (albedo x light + extra) fogged, `extra` what an
 * additive combiner adds (a glow texture; zero for the modulate families) and the fog the scene block's.
 * Everything else -- a draw without a variant, the transparents, liquids, weather, the interface --
 * leaves the attachments as the frame's first world draw cleared them: class 0, normal.w 0.
 *
 * The depth tag. A depth-writing draw WITHOUT a variant (a fixed-function draw, a hand-written row,
 * a module the tool has no family for, a blended draw that writes depth) cannot be trusted to clear the record
 * under it: its pipeline blends zeros into the albedo, but Vulkan leaves an attachment the fragment shader has no
 * output for undefined, and the usual hardware skips the write -- the pixel keeps the record of the surface drawn
 * BEHIND it (a house's wall relit as the terrain behind it). So every variant writes, in the normal's w,
 * its fragment's clip w (the view z the engine's projection puts in w) on a log2 scale:
 *
 *     |w| = WXL_VK_GBUFFER_TAG_MIN + WXL_VK_GBUFFER_TAG_SPAN x clamp((log2(max(clip w, 1/16)) + 4) / 16, 0, 1)
 *
 * (1/16 .. 4096 yd; the sign says whether there is a normal). A consumer decodes it, compares it with the clip w
 * of the final depth at the pixel (z_view x projection[11] + projection[15], the frame's own projection) and
 * leaves the pixel alone when the two differ by more than WXL_VK_GBUFFER_TAG_TOLERANCE in log2 units: the record
 * is then some other surface's. |w| exactly 1 is a variant older than the tag (no tag: trusted as before); 0 a
 * pixel no variant wrote. The SNORM normal resolves the tag to 0.1 %, the SFLOAT fallback to 1.1 %.
 *
 * The images are the device's, sized with the back buffer, in GENERAL layout for the device's life,
 * cleared by the frame's first world draw. From AfterOpaqueWorld on (a MidPass, a stage) they hold the
 * opaque world and their writes are visible to the vertex, fragment and compute stages; the engine's
 * transparents, drawn after the point, never touch them. A reader never writes them.
 */
/*
 * The interior split (appended). The albedo's alpha carries the class in bits 0..4 (every class is
 * below 32: read it with `& 31u`, not `& 63u`) and bit 5 (WXL_VK_GBUFFER_INTERIOR_BIT) set when the draw was a
 * batch of a map object group the engine lights as an INTERIOR -- MOGP flag 0x2000 (SMOGroup::INTERIOR) without
 * 0x40 (EXTERIOR_LIT) -- whatever the shader family: the device brackets the engine's group renderers
 * (Wmo.ExtRender, Wmo.IntRender) and tells the variants through the draw's row p[14].y. Such a pixel's light
 * term is the group's own interior light (baked vertex colours, the WMO ambient), never the sun's bands.
 *
 * The models and units standing in an interior (lit by the engine's interior light too) are not marked: their
 * draws are batched, instanced, and know nothing of the group. For them the G-buffer hands over the interior
 * groups this frame drew (WXL_VkGBuffer::interiors): a consumer tests a pixel's view-space position against
 * their boxes.
 *
 * The list holds every INDOOR group the frame drew (MOGP 0x2000), the exterior-lit ones (0x40:
 * a mine's or a cave's tunnels, an open hall) flagged by boundsMin[3] = WXL_VK_INTERIOR_EXTERIOR_LIT. Their air
 * is enclosed (a fog keeps the outdoor air out of every box), their light the sun's bands (a relight takes only
 * the boxes whose boundsMin[3] is 0, "lit as an interior"; their batches carry no bit 5).
 */
#define WXL_VK_GBUFFER_CLASS_MASK     0x1Fu   /**< the class bits of the albedo's alpha */
#define WXL_VK_GBUFFER_TAG_MIN        0.505f  /**< the depth tag's |normal.w| at a clip w of 1/16 yd (and nearer) */
#define WXL_VK_GBUFFER_TAG_SPAN       0.49f   /**< its span: |normal.w| 0.995 at 4096 yd (and farther) */
#define WXL_VK_GBUFFER_TAG_TOLERANCE  0.03f   /**< log2 units (about 2 %): a tag further from the depth is another surface's */
#define WXL_VK_GBUFFER_INTERIOR_BIT   0x20u   /**< a batch of an interior map object group (MOGP 0x2000 without 0x40) */

/**
 * One interior map object group the frame drew (WXL_VkGBuffer::interiors): its box in the map object's own
 * space and the transform from the frame's VIEW space into it (the space of the G-buffer's normals: the one the
 * engine's projection takes). local = localFromView (row-major 3 x 4) x (view xyz, 1); inside when every
 * component lies within [boundsMin, boundsMax]. `sphere` (view xyz, radius) bounds the box for a quick reject.
 * Six vec4 rows: uploaded as is, it is an std430 array element.
 */
typedef struct WXL_VkInteriorVolume
{
    float localFromView[12];
    float boundsMin[4];   ///< xyz, w the flags as a float: 0 lit as an interior, WXL_VK_INTERIOR_EXTERIOR_LIT
    float boundsMax[4];   ///< xyz, w 0
    float sphere[4];      ///< view-space centre xyz, radius w
} WXL_VkInteriorVolume;

/// boundsMin[3] of an indoor group the engine lights with the exterior's bands (MOGP 0x2000 with 0x40).
#define WXL_VK_INTERIOR_EXTERIOR_LIT 1u

#define WXL_VK_GBUFFER_MAX_INTERIORS 128u   /**< indoor groups a frame hands over at most (the later past it are dropped) */

/**
 * Which map object group an interior box is (appended, WXL_VkGBuffer::interiorGroups, parallel to `interiors`): the
 * WMO's model key -- CRC-32 of its root's file name lower-cased with '\\' separators, which an offline tool can compute
 * from the name alone -- and the group's index in its root (MOGI order, the `_NNN` of its file). The PLACEMENT is not
 * named: two instances of one WMO hand the same key; a consumer that knows an instance's transform tells them apart by
 * the box's localFromView. groupIndex is WXL_VK_INTERIOR_GROUP_UNKNOWN when the root's group array did not list it.
 */
typedef struct WXL_VkInteriorGroup
{
    uint32_t modelKey;
    uint32_t groupIndex;
    uint32_t groupFlags;   ///< the group's MOGP flags
    uint32_t reserved;     ///< 0
} WXL_VkInteriorGroup;

#define WXL_VK_INTERIOR_GROUP_UNKNOWN 0xFFFFFFFFu

#define WXL_VK_GBUFFER_CLASS_NONE     0u   /**< no variant wrote the pixel */
#define WXL_VK_GBUFFER_CLASS_TERRAIN  1u
#define WXL_VK_GBUFFER_CLASS_MAPOBJ   2u   /**< a map object's exterior families */
#define WXL_VK_GBUFFER_CLASS_INTERIOR 3u   /**< a map object's interior families (`MapObjU*`): baked vertex light */
#define WXL_VK_GBUFFER_CLASS_MODEL    4u   /**< a model (M2) lit by its vertex stage */
#define WXL_VK_GBUFFER_CLASS_UNLIT    5u   /**< a model's unlit permutation: no normal, lit by nothing */
#define WXL_VK_GBUFFER_MOVER_BIT      0x40u   /**< a unit's draw: a player, a creature, what they carry */
#define WXL_VK_GBUFFER_PLAIN_BIT      0x80u

typedef struct WXL_VkGBuffer
{
    uint32_t      structSize;    ///< the caller sets it; the device fills only what it covers
    /// 1 when the frame being recorded drew its opaque world with the attachments (they describe it);
    /// 0 when nobody asked for them, the device could not make them, the setting refuses them, or the
    /// world's pass has not run yet this frame.
    uint32_t      valid;
    uint64_t      serial;        ///< the frame they describe
    VkImageView   albedoView;
    VkImageView   normalView;
    VkImageView   lightView;
    VkFormat      albedoFormat;
    VkFormat      normalFormat;
    VkFormat      lightFormat;
    VkImageLayout layout;        ///< VK_IMAGE_LAYOUT_GENERAL
    VkExtent2D    extent;        ///< the back buffer's
    // Appended (check structSize): the indoor groups the frame drew, in its view space (the
    // exterior-lit ones flagged in boundsMin[3]). Valid for the frame `serial`, until the next frame opens;
    // count 0 with a null pointer when none was drawn.
    uint32_t                    interiorCount;
    const WXL_VkInteriorVolume* interiors;
    // Appended (check structSize): which group each of `interiors` is, the same count and order; null on a
    // device that does not know (an older one).
    const WXL_VkInteriorGroup*  interiorGroups;
} WXL_VkGBuffer;

/* --- shared frame services (appended): the device's capabilities, the memory budget, the shared
 *     depth products and the frame's figures. Each entry is appended to the table: a consumer checks structSize. ---------- */

/**
 * The device's OPTIONAL CAPABILITIES (GetDeviceCaps). The device is created at the loader's and the
 * adapter's version up to 1.2 and every optional feature below that the adapter reports is enabled (WXL_VK_CAPS=0 in
 * WarcraftXL.cfg: a plain Vulkan 1.0 device, `enabled` 0). The device's own drawing uses none of them: a consumer
 * builds a variant (FP16 maths, a subgroup reduction, an R16F storage image, a multiview pass) only where `enabled` has
 * the bit, and keeps its plain path for the rest. Shaders using the subgroup operations need SPIR-V 1.3
 * (glslang --target-env vulkan1.1): WXL_VK_CAP_SPIRV_1_3 says the device takes it.
 */
#define WXL_VK_CAP_FLOAT16                  0x00000001u   /**< shaderFloat16: float16_t arithmetic */
#define WXL_VK_CAP_INT8                     0x00000002u   /**< shaderInt8 */
#define WXL_VK_CAP_STORAGE_16BIT            0x00000004u   /**< storageBuffer16BitAccess */
#define WXL_VK_CAP_UNIFORM_16BIT            0x00000008u   /**< uniformAndStorageBuffer16BitAccess */
#define WXL_VK_CAP_PUSH_16BIT               0x00000010u   /**< storagePushConstant16 */
#define WXL_VK_CAP_IO_16BIT                 0x00000020u   /**< storageInputOutput16 (16-bit varyings) */
#define WXL_VK_CAP_INT16                    0x00000040u   /**< shaderInt16 */
#define WXL_VK_CAP_STORAGE_IMAGE_EXTENDED   0x00000080u   /**< shaderStorageImageExtendedFormats: r16f, rg16f, r11f_g11f_b10f ... storage images */
#define WXL_VK_CAP_MULTIVIEW                0x00000100u   /**< multiview: a render pass with a view mask, gl_ViewIndex (maxMultiviewViews) */
#define WXL_VK_CAP_LAYER_FROM_VERTEX        0x00000200u   /**< VK_EXT_shader_viewport_index_layer: gl_Layer / gl_ViewportIndex from a vertex shader */
#define WXL_VK_CAP_MEMORY_BUDGET            0x00000400u   /**< VK_EXT_memory_budget: GetMemoryBudget's figures are the driver's */
#define WXL_VK_CAP_MULTI_DRAW_INDIRECT      0x00000800u   /**< multiDrawIndirect */
#define WXL_VK_CAP_INDIRECT_FIRST_INSTANCE  0x00001000u   /**< drawIndirectFirstInstance */
#define WXL_VK_CAP_SPIRV_1_3                0x00002000u   /**< a 1.1 device: SPIR-V 1.3 modules, the subgroup operations (subgroupOps) */
#define WXL_VK_CAP_SPIRV_1_5                0x00004000u   /**< a 1.2 device: SPIR-V 1.5 modules */
#define WXL_VK_CAP_MULTI_VIEWPORT           0x00008000u   /**< multiViewport */

typedef struct WXL_VkDeviceCaps
{
    uint32_t structSize;        ///< the caller sets it; the device fills only what it covers
    uint32_t apiVersion;        ///< what the device may be used as (VK_MAKE_API_VERSION, patch 0): 1.0, 1.1 or 1.2
    uint32_t instanceVersion;   ///< the instance's
    uint32_t adapterVersion;    ///< the adapter's own (VkPhysicalDeviceProperties::apiVersion)
    uint32_t enabled;           ///< WXL_VK_CAP_*: enabled on the device -- what a consumer may use
    uint32_t offered;           ///< WXL_VK_CAP_*: what the adapter reports (enabled is all of it unless WXL_VK_CAPS is off)
    uint32_t subgroupSize;      ///< lanes a subgroup (32 on NVIDIA); 0 on a 1.0 device
    uint32_t subgroupStages;    ///< VkShaderStageFlags where subgroup operations are allowed
    uint32_t subgroupOps;       ///< VkSubgroupFeatureFlags
    uint32_t maxMultiviewViews; ///< 0 without WXL_VK_CAP_MULTIVIEW
    /// The adapter's queue families beside the device's (graphicsFamily, the one queue it takes). INFORMATION ONLY: the
    /// device owns every queue (rule 2), nothing submits on these yet; an async-compute lane would be the device's service.
    uint32_t graphicsFamily;
    uint32_t graphicsQueues;
    uint32_t computeFamily;     ///< a family with compute and no graphics (async compute), 0xFFFFFFFF none
    uint32_t computeQueues;
    uint32_t transferFamily;    ///< a transfer-only family (a copy engine), 0xFFFFFFFF none
    uint32_t transferQueues;
} WXL_VkDeviceCaps;

/**
 * THE MEMORY BUDGET (GetMemoryBudget, MayKeepResident). With WXL_VK_CAP_MEMORY_BUDGET the heaps' budget and usage are the
 * driver's (VK_EXT_memory_budget: every allocation of the process, the modules' own vkAllocateMemory included, against
 * what the OS lets it have now); without it the budget is 80 % of the heap and the usage the device allocator's reserved
 * bytes (WXL_VK_BUDGET_DRIVER clear). Refreshed every 120 presents (a driver query is not free) and logged: at the device's
 * first frames, every stats block (WXL_VK_STATS) and in the session's summary line. The 32-bit process's address space is
 * part of it: a host-visible block costs its size there, and the largest free run predicts the next failure.
 */
#define WXL_VK_BUDGET_DRIVER 0x1u   /**< budget and usage come from the driver (VK_EXT_memory_budget) */
#define WXL_VK_RESIDENT_MARGIN 0.10f /**< the share of the device-local budget MayKeepResident keeps free (256 MiB at least) */

typedef struct WXL_VkMemoryHeapBudget
{
    uint64_t size;       ///< the heap's size (0: no such heap)
    uint64_t budget;     ///< what the process may hold in it now (the driver's, or 80 % of size)
    uint64_t usage;      ///< what the process holds in it (the driver's count, or the allocator's reserved bytes)
    uint64_t reserved;   ///< of that, the device's allocator (its blocks in this heap's types)
} WXL_VkMemoryHeapBudget;

typedef struct WXL_VkMemoryBudget
{
    uint32_t               structSize;    ///< the caller sets it; the device fills only what it covers
    uint32_t               flags;         ///< WXL_VK_BUDGET_*
    uint64_t               serial;        ///< the frame the figures were taken at
    WXL_VkMemoryHeapBudget deviceLocal;   ///< the largest DEVICE_LOCAL heap: the card's memory
    WXL_VkMemoryHeapBudget hostVisible;   ///< the largest heap without DEVICE_LOCAL: system memory the GPU reads
    /// A DEVICE_LOCAL heap of its own holding the DEVICE_LOCAL | HOST_VISIBLE types (the 256-MiB BAR window without
    /// Resizable BAR; size 0 when those types live in the main heap): where the device's per-frame rings prefer to live.
    WXL_VkMemoryHeapBudget bar;
    uint64_t               allocatorReserved;  ///< the device's allocator: VkDeviceMemory held, every heap
    uint64_t               allocatorInUse;     ///< handed out of it
    uint64_t               allocatorPeak;      ///< the session's highest reserved figure
    uint64_t               mapped;             ///< bytes the allocator keeps mapped: address space spent
    uint64_t               addressFree;        ///< the process's free virtual address space
    uint64_t               addressLargestFree; ///< its largest free run (a request above it fails)
    uint64_t               addressTotal;       ///< the process's user address space (4 GiB with /LARGEADDRESSAWARE on 64-bit Windows)
    /// What MayKeepResident(n, WXL_VK_MEMORY_DEVICE_LOCAL) grants now: the device-local budget minus the usage minus the
    /// safety margin (WXL_VK_RESIDENT_MARGIN of the budget, 256 MiB at least); 0 when nothing more should stay resident.
    uint64_t               residentHeadroom;
} WXL_VkMemoryBudget;

/**
 * THIS FRAME'S OPAQUE DEPTH (RequestOpaqueDepth, GetOpaqueDepth): the copy GetPreviousDepth hands over the NEXT frame,
 * handed over in the frame that made it, from its MidPass on -- the same R32F image, the same moment (AfterOpaqueWorld,
 * after the Prepares, before every consumer's MidPass and Record), the same raw depth values. A consumer that would keep
 * its own copy of the opaque world's depth in a MidPass reads this one instead: one copy a frame for
 * every consumer, the previous depth's included. Described by a WXL_VkPreviousDepth: `valid` 1 when it is the frame being
 * recorded's (`serial` the frame), its camera the frame's; GENERAL layout, readable by the compute, vertex and fragment
 * stages of every command recorded after the copy (a MidPass, a stage at any point of the frame). Never written by a
 * consumer; re-made at a size change (take the view from every call).
 */

/**
 * THE DEPTH PYRAMID (RequestDepthPyramid, GetDepthPyramid): the opaque depth (as GetOpaqueDepth's: the same moment, the
 * same values) reduced to its nearest and farthest raw depth per texel, level by level: level 0 is half the back buffer
 * (rounded up), each level halves the last (rounded up) down to 1 x 1, and a texel's footprint is the 2 x 2 texels under
 * it one level down (the back buffer's pixels at level 0), clamped at the edges -- so level L's texel (x, y) holds the
 * min and max over the pixels [x * 2^(L+1), (x + 1) * 2^(L+1)) of each axis: level 3 is the 16 x 16 tiles. Built in the
 * MidPass break at AfterOpaqueWorld only while a consumer asks (one dispatch a level, ~0.02 ms at 1080p). For a Hi-Z
 * occlusion test (g, the farthest: D3D depth, 0 near), a tile's depth bounds (light culling, the air's empty space), a
 * half-resolution depth (level 0) for screen-space effects.
 */
typedef struct WXL_VkDepthPyramid
{
    uint32_t      structSize;   ///< the caller sets it; the device fills only what it covers
    uint32_t      valid;        ///< 1 when it describes the frame being recorded (`serial`)
    uint64_t      serial;
    VkImage       image;
    VkImageView   view;         ///< every level (textureLod / texelFetch with the level; a nearest sampler)
    VkFormat      format;       ///< VK_FORMAT_R32G32_SFLOAT: r the nearest raw depth of the footprint, g the farthest
    VkImageLayout layout;       ///< VK_IMAGE_LAYOUT_GENERAL; never changed nor written by a consumer
    VkExtent2D    extent;       ///< level 0's: (back buffer + 1) / 2
    uint32_t      levels;
    float         depthMin;     ///< the world's viewport depth range (WXL_VkPreviousDepth's): z' = (d - min) / (max - min)
    float         depthMax;
} WXL_VkDepthPyramid;

/**
 * THE FRAME'S FIGURES (GetFrameStats), for a consumer's panel or an in-game overlay: the CPU frame time
 * (present to present), the GPU time of the frame's submission (timestamps at the start of its first command buffer and
 * after its last pass: everything the frame drew, the modules' work included, the present copy and any wait for the
 * swapchain image left out), the device's own shared work (the opaque depth copy, the depth pyramid), the draws. Windows
 * are the last 128 frames. The draw slot's CPU split is measured only with WXL_VK_STATS (-1 otherwise).
 */
#define WXL_VK_FRAME_GPU_TIMED   0x1u   /**< the gpu* figures are measured (the queue has timestamps) */
#define WXL_VK_FRAME_CPU_SPLIT   0x2u   /**< drawCpuMs / recordCpuMs are measured (WXL_VK_STATS) */

typedef struct WXL_VkFrameStats
{
    uint32_t structSize;        ///< the caller sets it; the device fills only what it covers
    uint32_t flags;             ///< WXL_VK_FRAME_*
    uint64_t frames;            ///< presented so far
    float    cpuFrameMs;        ///< the latest present-to-present interval
    float    cpuFrameAvgMs;     ///< over the window
    float    cpuFrameMaxMs;
    float    gpuFrameMs;        ///< the latest harvested frame's submission on the GPU; -1 unknown
    float    gpuFrameAvgMs;
    float    gpuFrameMaxMs;
    float    depthCopyMs;       ///< the opaque depth copy (GetOpaqueDepth / GetPreviousDepth), window average; -1 not made
    float    pyramidMs;         ///< the depth pyramid, window average; -1 not built
    uint32_t draws;             ///< engine draws received, the latest frame
    uint32_t drawn;             ///< recorded, the latest frame
    float    drawCpuMs;         ///< the draw slot's CPU per frame, window average (WXL_VK_FRAME_CPU_SPLIT), else -1
    float    recordCpuMs;       ///< of that, recording into the command buffer
    uint32_t passBreaks;        ///< the latest frame's pass breaks
    uint32_t reserved;          ///< 0
} WXL_VkFrameStats;

typedef struct WXL_VkRenderApi
{
    uint32_t structSize;
    uint32_t apiVersion;

    /**
     * @brief The device's handles.
     * @param out  filled on success.
     * @return 1 once the Vulkan device exists, 0 before (the device comes up at the engine's first
     *         format set, after the extensions load).
     */
    int (__cdecl* GetDeviceInfo)(WXL_VkDeviceInfo* out);

    /**
     * @brief Registers a pass at an insertion point.
     *
     * Passes at one point run in registration order. Costs nothing on frames before the point
     * fires; the point itself costs nothing while no pass is registered there.
     * @param point  a WXL_VkRenderPoint.
     * @param desc   the callbacks, copied.
     * @param name   for the log; a literal or storage that outlives the registration.
     * @return a handle for UnregisterPass, or NULL (an unknown point, a full table).
     */
    void* (__cdecl* RegisterPass)(uint32_t point, const WXL_VkRenderPassDesc* desc, const char* name);

    /// Withdraws a pass. Not from inside one of its own callbacks.
    void (__cdecl* UnregisterPass)(void* pass);

    /**
     * @brief Suppresses every engine draw whose shader carries this name.
     *
     * A shader matches when its name is `shaderName` or begins with `shaderName` followed by ':' (the
     * engine's permutation suffix): "DetailDoodad" covers "DetailDoodad:0" .. "DetailDoodad:5". The
     * draw is dropped after the engine's own state sync, so the engine sees it as drawn.
     * @param shaderName  the shader family name.
     * @param stages      bit 0 the vertex shader's name, bit 1 the pixel shader's.
     * @param owner       for the log.
     * @return a handle for RemoveDrawFilter, or NULL (a full table).
     */
    void* (__cdecl* AddDrawFilter)(const char* shaderName, uint32_t stages, const char* owner);

    /// Lifts a draw filter.
    void (__cdecl* RemoveDrawFilter)(void* filter);

    /// How many draws a filter has suppressed so far.
    uint64_t (__cdecl* FilteredDraws)(void* filter);

    /**
     * @brief A buffer from the device's allocator.
     * @param desc  size, usage, memory class.
     * @param out   the buffer's handle, size and mapping.
     * @return a handle for RetireBuffer / FlushBuffer, or NULL (no device, no memory).
     */
    void* (__cdecl* CreateBuffer)(const WXL_VkBufferDesc* desc, WXL_VkBufferInfo* out);

    /// Destroys a buffer once the frame with serial `lastSerial` has completed (at once when it has).
    void (__cdecl* RetireBuffer)(void* buffer, uint64_t lastSerial);

    /// Makes CPU writes to a mapped buffer visible to the device (a no-op on coherent memory).
    void (__cdecl* FlushBuffer)(void* buffer);

    /// An image and its view from the device's allocator; NULL on failure.
    void* (__cdecl* CreateImage)(const WXL_VkImageDesc* desc, WXL_VkImageInfo* out);

    /// Destroys an image and its view once the frame with serial `lastSerial` has completed.
    void (__cdecl* RetireImage)(void* image, uint64_t lastSerial);

    /**
     * @brief Pauses (0) or resumes (non-zero) a draw filter without withdrawing it, silently: for a
     *        consumer that hides the engine's content only on the frames it replaces it. Takes effect
     *        from the next engine draw. Appended to version 1: check structSize before calling.
     */
    void (__cdecl* SetDrawFilterActive)(void* filter, int active);

    /**
     * @brief Asks for (non-zero) or stops asking for (0) the previous frame's depth. Counted: each
     *        consumer that asks stops asking once. Nothing is copied while nobody asks, and the first
     *        copy is made in the frame after the first request. Appended to version 1: check
     *        structSize before calling.
     */
    void (__cdecl* RequestPreviousDepth)(int wanted);

    /**
     * @brief The previous frame's depth (WXL_VkPreviousDepth), from Prepare or Record.
     * @param out  set structSize first; filled up to it.
     * @return 1 when a copy exists (check out->valid before using it), 0 when none does. Appended to
     *         version 1: check structSize before calling.
     */
    int (__cdecl* GetPreviousDepth)(WXL_VkPreviousDepth* out);

    /**
     * @brief Tree wind: the stage-0 texture path substrings (case-insensitive, `\` or `/`) that make
     *        an M2 draw's model a candidate. Replaces any earlier list; an empty list stops watching.
     *        Appended to version 1: check structSize before calling.
     */
    void (__cdecl* WatchTextures)(const char* const* substrings, uint32_t count);

    /**
     * @brief Tree wind: the next candidate model whose mesh is read back, or 0 when none is.
     * @param out  set structSize first. Its pointers stay valid until the next call of this or
     *             SetModelWind. Every model handed out must be answered with SetModelWind.
     */
    int (__cdecl* TakeWindModel)(WXL_VkWindModel* out);

    /**
     * @brief Tree wind: one vec4 per vertex for a model TakeWindModel handed out (vertexCount of
     *        them), or null to leave it unwound for the session. Copied.
     * @return 1 when taken, 0 when the id is unknown or the device has no room left
     */
    int (__cdecl* SetModelWind)(uint32_t id, const float* perVertex, uint32_t vertexCount);

    /// Tree wind: this frame's wind (WXL_VkWindParams), copied; the next frame's draws read it.
    void (__cdecl* SetWind)(const WXL_VkWindParams* params);

    /// Tree wind: its counters (WXL_VkWindStats); set structSize first. 1 when filled.
    int (__cdecl* GetWindStats)(WXL_VkWindStats* out);

    /**
     * @brief Tree wind: the windy tree instances drawn so far this frame (one per instance, however
     *        many batches it draws), at most `max`. From a pass's Prepare at AfterOpaqueWorld, the
     *        opaque world's trees are all there.
     * @return the count written
     */
    uint32_t (__cdecl* GetWindInstances)(WXL_VkWindInstance* out, uint32_t max);

    /**
     * @brief A stage at a point (WXL_VkStageDesc). Appended to version 1: check structSize.
     * @return a handle for UnregisterStage, or NULL (a bad point, a full chain)
     */
    void* (__cdecl* RegisterStage)(const WXL_VkStageDesc* desc, const char* name);

    /// Withdraws a stage (it stops being run; its slot in the chain is kept).
    void (__cdecl* UnregisterStage)(void* stage);

    /// The replay's depth-only pass class (WXL_VkShadowPass); set structSize first. 1 when filled.
    int (__cdecl* GetShadowPass)(WXL_VkShadowPass* out);

    /**
     * @brief Asks for (non-zero) or stops asking for (0) the capture of the opaque world's draws.
     *        Counted. While nobody asks the draw path tests one counter.
     */
    void (__cdecl* RequestDrawCapture)(int wanted);

    /**
     * @brief Replays the frame's captured opaque world into one cascade, depth only, then calls the
     *        SHADOW_CASTERS passes inside the same pass. From a MidPass at AfterOpaqueWorld (the
     *        capture is complete there), outside any pass. Begins and ends the render pass itself.
     * @param stats  set structSize first; may be NULL
     * @return 1 when the pass was recorded, 0 when it could not be (no capture, no shadow pass)
     */
    int (__cdecl* ReplayWorld)(const WXL_VkReplayDesc* desc, WXL_VkReplayStats* stats);

    /**
     * @brief A consumer's screen-space surface for this frame (WXL_VkSurface), copied: the latest
     *        published wins. From a MidPass at AfterOpaqueWorld. Appended to version 1: check
     *        structSize before calling.
     */
    void (__cdecl* PublishSurface)(const WXL_VkSurface* surface);

    /**
     * @brief The surface published this frame, if any.
     * @param out  set structSize first; filled up to it.
     * @return 1 when one was published for the frame being recorded, 0 otherwise.
     */
    int (__cdecl* GetSurface)(WXL_VkSurface* out);

    /**
     * @brief Tree wind: SetModelWind with several vec4s per vertex. `vec4sPerVertex` 1 is SetModelWind's
     *        layout (trunk, branch, leaf weights and the height); 4 is a Pivot Painter skeleton the wind
     *        variant bends hierarchically, each element about its own base (wind_m2.glsl documents the
     *        four). Copied. Appended to version 1: check structSize before calling.
     * @return 1 when taken, 0 when the id is unknown, the layout is not one of these or there is no room
     */
    int (__cdecl* SetModelWindEx)(uint32_t id, const float* data, uint32_t vertexCount, uint32_t vec4sPerVertex);

    /**
     * @brief The trees' canopy density for this frame (WXL_VkCanopyVolumes), copied with its instances:
     *        the latest published wins. From a Prepare. Appended to version 1: check structSize.
     */
    void (__cdecl* PublishCanopy)(const WXL_VkCanopyVolumes* volumes);

    /**
     * @brief The canopy density published for the frame being recorded, if any.
     * @param out  set structSize first; filled up to it (`instances` points at the device's copy).
     * @return 1 when there is one for this frame, 0 otherwise.
     */
    int (__cdecl* GetCanopy)(WXL_VkCanopyVolumes* out);

    /**
     * @brief A stage's figures (WXL_VkStageStats): its runs, its GPU time, its point's status.
     * @param stage  a RegisterStage handle.
     * @param out    set structSize first; filled up to it.
     * @return 1 when filled, 0 for an unknown handle. Appended to version 1: check structSize.
     */
    int (__cdecl* GetStageStats)(void* stage, WXL_VkStageStats* out);

    /**
     * @brief Asks for (non-zero) or stops asking for (0) the world's sub-pixel jitter: while any consumer
     *        asks, every world draw on the back buffer (its viewport's depth range ending below 1) is
     *        offset by the frame's sample of an 8-step Halton 2/3 sequence, for a temporal resolve.
     *        Counted; applies from the next world pass. The offset reaches the stages as
     *        WXL_VkStageContext::jitter and the previous depth as WXL_VkPreviousDepth::jitter.
     *        Appended to version 1: check structSize before calling.
     */
    void (__cdecl* RequestJitter)(int wanted);

    /**
     * @brief Asks for (non-zero) or stops asking for (0) the G-buffer (WXL_VkGBuffer): while any
     *        consumer asks, the opaque world's pass carries its three attachments and the `gbuf`
     *        module variants draw. Counted; applies from the next pass the device begins (asked for
     *        in the middle of a world, that frame's G-buffer is partial), and nothing is paid while
     *        nobody asks. Appended to version 1: check structSize before calling.
     */
    void (__cdecl* RequestGBuffer)(int wanted);

    /**
     * @brief The G-buffer of the frame being recorded (WXL_VkGBuffer), from a MidPass or a stage at
     *        AfterOpaqueWorld (the opaque world is complete there).
     * @param out  set structSize first; filled up to it.
     * @return 1 when the attachments exist (read out->valid before using them), 0 when none do.
     */
    int (__cdecl* GetGBuffer)(WXL_VkGBuffer* out);

    /**
     * @brief ReplayWorld for several views in one walk of the capture and one render pass (WXL_VkReplayViewsDesc): the
     *        depth each view gets is what one ReplayWorld per view would have drawn. From a MidPass at AfterOpaqueWorld,
     *        outside any pass. Appended to version 1: check structSize before calling.
     * @param stats  set structSize first; may be NULL. Summed over the views (`captured` once): replayed, culled, skipped,
     *               excluded, mapObjects as the views' ReplayWorld calls would have added them up.
     * @return 1 when the pass was recorded, 0 when it could not be (no capture, no LOAD class, no view)
     */
    int (__cdecl* ReplayWorldViews)(const WXL_VkReplayViewsDesc* desc, WXL_VkReplayStats* stats);

    /* --- appended to version 1: check structSize before calling each ------------------------------------------- */

    /**
     * @brief The device's optional capabilities (WXL_VkDeviceCaps): its version, what it enabled, the subgroup and
     *        multiview limits, the queue families. Fixed once the device is up.
     * @param out  set structSize first; filled up to it.
     * @return 1 once the device exists, 0 before.
     */
    int (__cdecl* GetDeviceCaps)(WXL_VkDeviceCaps* out);

    /**
     * @brief The memory budget and usage (WXL_VkMemoryBudget), as last refreshed (every 120 presents).
     * @param out  set structSize first; filled up to it.
     * @return 1 once the device exists, 0 before.
     */
    int (__cdecl* GetMemoryBudget)(WXL_VkMemoryBudget* out);

    /**
     * @brief Whether `extraBytes` more may stay resident in a memory class (WXL_VkMemoryClass) without crowding the
     *        budget: the device-local heap's usage plus `extraBytes` stays under its budget less the safety margin;
     *        for the mapped classes the address space keeps 512 MiB free and a run larger than the request. A
     *        module that keeps bakes resident (more on a big card) asks before keeping more and frees when refused;
     *        the answer moves with the driver's budget (another application, a zone's textures). Cheap: the cached
     *        figures, no driver call.
     * @return 1 granted, 0 refused (or no device).
     */
    int (__cdecl* MayKeepResident)(uint64_t extraBytes, uint32_t memoryClass);

    /**
     * @brief Asks for (non-zero) or stops asking for (0) this frame's opaque depth (GetOpaqueDepth). Counted. The copy
     *        is the one RequestPreviousDepth makes: asking for either makes it, once a frame.
     */
    void (__cdecl* RequestOpaqueDepth)(int wanted);

    /**
     * @brief This frame's opaque depth (see THIS FRAME'S OPAQUE DEPTH above), from the MidPass at AfterOpaqueWorld on.
     * @param out  set structSize first; filled up to it.
     * @return 1 when this frame's copy exists (out->valid 1, out->serial the frame), 0 when none was made this frame.
     */
    int (__cdecl* GetOpaqueDepth)(WXL_VkPreviousDepth* out);

    /**
     * @brief Asks for (non-zero) or stops asking for (0) the depth pyramid (WXL_VkDepthPyramid). Counted; nothing is
     *        built while nobody asks.
     */
    void (__cdecl* RequestDepthPyramid)(int wanted);

    /**
     * @brief This frame's depth pyramid, from the MidPass at AfterOpaqueWorld on.
     * @param out  set structSize first; filled up to it.
     * @return 1 when this frame's pyramid was built (out->valid 1), 0 otherwise.
     */
    int (__cdecl* GetDepthPyramid)(WXL_VkDepthPyramid* out);

    /**
     * @brief The frame's figures (WXL_VkFrameStats): CPU and GPU frame times, the device's shared work, the draws.
     * @param out  set structSize first; filled up to it.
     * @return 1 once the device presents, 0 before.
     */
    int (__cdecl* GetFrameStats)(WXL_VkFrameStats* out);
} WXL_VkRenderApi;

#ifdef __cplusplus
}
#endif

#endif // WXL_VK_RENDER_API_H
