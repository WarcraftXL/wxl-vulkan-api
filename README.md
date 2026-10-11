# wxl-vulkan-api

**A native Vulkan graphics device for the 3.3.5a (12340) client, in place of its D3D9 backend.**

The engine keeps its own `CGxDevice` object, built by its own factory, with its window, render
states and caches. The extension replaces the code behind the 42 vtable slots that reach D3D9: 39
are reimplemented on Vulkan 1.0, 3 are stubs. The client's DX9 shaders run as SPIR-V translated
offline from the client's own archives; the UI, fixed-function, particle, present and overlay
shaders are hand-written GLSL.

## Enabling it

In `WarcraftXL.cfg` beside the client (or as environment variables):

```
# The native Vulkan device. Unset, the extension stays inert and the stock D3D9 device is used.
WXL_VK_DEVICE=native

# Optional:
#WXL_VK_STATS=1          detailed counters in the log, about once a minute and at exit
#WXL_VK_VALIDATION=1     the Khronos validation layer, when installed (32-bit)
#WXL_VK_TRANSLATED=1     0 hand-written pipelines only, 1 translated modules fill in, 2 prefer them
#WXL_VK_ANISOTROPY=16    anisotropic filtering of the world's mipmapped textures (0 or 1: the engine's own)
#WXL_VK_MIP_BIAS=-0.3    their mip LOD bias, -1 .. 1 (below 0: sharper)
```

Requirements: a Vulkan 1.0 driver and the 32-bit loader (`C:\Windows\SysWOW64\vulkan-1.dll`). Without
them the device is declined and the log says why.

## What the log shows

By default, errors, warnings and one line when the device is destroyed:

```
wxl-vulkan-api: <n> frame(s) presented, frame time avg <a> / max <m> ms; <d> draw(s) received,
<r> drawn, <p> pipeline(s); memory <x> MiB reserved, <y> MiB in use (<z> allocation(s))
```

`WXL_VK_STATS=1` adds per-module counters (draws, frame passes, constants, pools, textures, shaders,
targets, queries, CPU time, memory). `WXL_LOG_LEVEL=debug` (a core key) shows the bring-up details.

## Shipping

```
Extensions/wxl-vulkan-api/
  wxl-vulkan-api.dll
  shaders/*.spv              hand-written modules
  shaders/translated/        manifest.txt and the translated modules
  pipeline-cache.bin         written at runtime
```

Building needs neither the Vulkan SDK nor a shader compiler: the Khronos headers are vendored under
`third_party/vulkan-headers`, every Vulkan entry point is resolved at runtime, and the compiled
`.spv` are versioned. With glslang available (`WXL_VK_GLSLANG`), the `.spv` are rebuilt from
`src/shaders/glsl/`. The translated modules under `shaders/translated/` are produced offline from the
client's own shader archives by a separate translator, which is not part of this repository.
