<div align="center">

# wxl-vulkan-api

*A native Vulkan graphics device for the World of Warcraft 3.3.5a client*

[![WarcraftXL](https://img.shields.io/badge/WarcraftXL-v1.2-1f6feb?style=for-the-badge)](https://github.com/WarcraftXL/wxl-core/tree/v1.2)
[![Vulkan](https://img.shields.io/badge/Vulkan-1.0%20%E2%80%93%201.2-AC162C?style=for-the-badge&logo=vulkan&logoColor=white)](https://www.vulkan.org/)
[![Client](https://img.shields.io/badge/Client-3.3.5a%20(12340)-8a6d3b?style=for-the-badge)](#-requirements)
[![Status](https://img.shields.io/badge/Status-Alpha-orange?style=for-the-badge)](#%EF%B8%8F-alpha-notice)
[![License](https://img.shields.io/badge/License-GPL--3.0-blue?style=for-the-badge)](LICENSE)

---
</div>

> [!IMPORTANT]
> **wxl-vulkan-api is an extension of [WarcraftXL](https://github.com/WarcraftXL/wxl-core) and requires wxl-core v1.2.**
> It does not build or load against wxl-core v1.1.

## ⚠️ Alpha notice

> [!WARNING]
> **wxl-core v1.2 and wxl-vulkan-api are both in alpha.** APIs, configuration keys and file layouts can still change
> between commits, and you should expect rendering bugs, missing features and the occasional crash. Keep a copy of
> your client and its `WTF` folder, and do not use it on a client you cannot afford to break.

## 🚀 Overview

wxl-vulkan-api replaces the client's Direct3D 9 backend with a native Vulkan device, from inside the game.

The engine keeps its own `CGxDevice` object, built by its own factory, with its window, render states and caches.
The extension replaces the code behind the 42 vtable slots that reach D3D9: 39 are reimplemented on Vulkan, 3 are
stubs. The client's DX9 shaders run as SPIR-V translated from **your own client's** shader archives; the UI,
fixed-function, particle, present and overlay shaders are hand-written GLSL.

### Key features

- **A drop-in device**: one line in `WarcraftXL.cfg` switches it on; without it the stock D3D9 device runs as before.
- **The client's own shaders, on Vulkan**: every DX9 shader record the engine loads (1578 of 1578) translated to
  SPIR-V, with bone-palette, tree-wind, shadow and G-buffer variants.
- **A rendering API for other extensions** ([`include/wxl/VkRenderApi.h`](include/wxl/VkRenderApi.h)): insertion
  points in the frame, compute and full-screen stages, the previous and current depth, a depth pyramid, a G-buffer
  of the opaque world, shadow replays of the captured world, device capabilities, memory budget and frame figures.
- **No SDK to install**: the Khronos headers are vendored and every Vulkan entry point is resolved at runtime.

## 📋 Table of Contents

- [Requirements](#-requirements)
- [Installation](#-installation)
- [Translating the shaders](#-translating-the-shaders)
- [Enabling it](#%EF%B8%8F-enabling-it)
- [What the log shows](#-what-the-log-shows)
- [For extension authors](#-for-extension-authors)
- [Contributing](#-contributing)
- [Acknowledgements](#-acknowledgements)
- [License](#-license)

## 📦 Requirements

**To run**
- A legally obtained World of Warcraft **3.3.5a (build 12340)** client
- [WarcraftXL **v1.2**](https://github.com/WarcraftXL/wxl-core/tree/v1.2), installed in that client
- A GPU and driver with **Vulkan 1.0** or later, and the **32-bit** Vulkan loader
  (`C:\Windows\SysWOW64\vulkan-1.dll`, installed by every current NVIDIA, AMD and Intel driver)

**To build**
- What wxl-core v1.2 needs: CMake ≥ 3.20 and a Win32 C++20 toolchain (Visual Studio 2022 or later, or clang
  mingw-w64 on Linux)
- **Python 3.8+** and **[glslang](https://github.com/KhronosGroup/glslang/releases)** to translate the shaders (once)

## ⚡ Installation

wxl-vulkan-api builds as part of wxl-core: clone it into wxl-core's `extensions/` folder.

```bash
# 1. wxl-core, on its v1.2 branch (see its README for the client setup and the patcher)
git clone -b v1.2 https://github.com/WarcraftXL/wxl-core.git
cd wxl-core

# 2. this repository, as an extension
git clone https://github.com/WarcraftXL/wxl-vulkan-api.git extensions/wxl-vulkan-api

# 3. the translated shaders, from your own client (see below)
python extensions/wxl-vulkan-api/tools/translate_shaders.py --client "C:\Games\World of Warcraft 3.3.5a"

# 4. build and deploy into the client
./build.ps1 -ClientPath "C:\Games\World of Warcraft 3.3.5a"
```

The build deploys:

```
Extensions/wxl-vulkan-api/
  wxl-vulkan-api.dll
  shaders/*.spv              hand-written modules (versioned in this repository)
  shaders/translated/        the translated modules and their manifest (made by step 3)
  pipeline-cache.bin         written at runtime
```

## 🔁 Translating the shaders

The translated modules are **not** part of this repository: they are derived from Blizzard's shader archives, so
each user makes them from their own copy of the game. The translator ships in [`tools/`](tools/) and runs on plain
Python, no package to install:

```bash
python tools/translate_shaders.py --client "C:\Games\World of Warcraft 3.3.5a"
python tools/translate_shaders.py --client <client> --glslang "C:\glslang\bin\glslang.exe"
python tools/translate_shaders.py --client <client> --deploy     # also copy the result into the client now
```

It reads the `.bls` shader containers from the client's MPQ archives (in the client's own load order, your custom
patches included), translates every record to GLSL, compiles it to SPIR-V and writes `shaders/translated/` with
its `manifest.txt`. glslang is found from `--glslang`, `$GLSLANG`, a wxl-core build tree, `$VULKAN_SDK`, then
`PATH`. A full run takes a few minutes.

Run it again whenever your client's shaders change (a patch that adds or replaces `.bls` files). Details, and the
smaller tools beside it (disassembler, census, contract checker), are in [`tools/README.md`](tools/README.md).

## ⚙️ Enabling it

In `WarcraftXL.cfg` beside the client (or as environment variables):

```ini
# The native Vulkan device. Unset, the extension stays inert and the stock D3D9 device is used.
WXL_VK_DEVICE=native

# Optional:
#WXL_VK_STATS=1          detailed counters in the log, about once a minute and at exit
#WXL_VK_VALIDATION=1     the Khronos validation layer, when installed (32-bit)
#WXL_VK_TRANSLATED=1     0 hand-written pipelines only, 1 translated modules fill in, 2 prefer them
#WXL_VK_ANISOTROPY=16    anisotropic filtering of the world's mipmapped textures (0 or 1: the engine's own)
#WXL_VK_MIP_BIAS=-0.3    their mip LOD bias, -1 .. 1 (below 0: sharper)
#WXL_VK_CAPS=1           0: a plain Vulkan 1.0 device, without the optional 1.1 / 1.2 features
```

Without a usable Vulkan driver, 32-bit loader or translated shaders, the device is declined and the log says why.

## 📈 What the log shows

By default: errors, warnings and one line when the device is destroyed.

```
wxl-vulkan-api: <n> frame(s) presented, frame time avg <a> / max <m> ms; <d> draw(s) received,
<r> drawn, <p> pipeline(s); memory <x> MiB reserved, <y> MiB in use (<z> allocation(s))
```

`WXL_VK_STATS=1` adds per-module counters (draws, frame passes, constants, pools, textures, shaders, targets,
queries, CPU time, memory). `WXL_LOG_LEVEL=debug` (a core key) shows the bring-up details.

## 🧩 For extension authors

Another WarcraftXL extension can record its own Vulkan work into the frame. The interface is published as
`wxl.vk.render` while the native device runs; resolve it after load, not in `WXL_Load`:

```cpp
#include "wxl/VkRenderApi.h"

const auto* vk = wxl::ScriptMgr::GetInterface<const WXL_VkRenderApi>(WXL_VK_RENDER_API_NAME, WXL_VK_RENDER_API_VERSION);
if (vk) { /* RegisterPass, RegisterStage, RequestGBuffer, ReplayWorld, ... */ }
```

The header is the reference: every entry point, structure and rule (threading, render passes, resource lifetimes)
is documented in [`include/wxl/VkRenderApi.h`](include/wxl/VkRenderApi.h). Fields and functions are only ever
appended; check `structSize` before reading or calling anything marked as appended.

## 🤝 Contributing

Issues and pull requests are welcome, keeping in mind that the project is in alpha and moves fast.

1. **Fork** the repository and create a branch from `main`
2. **Follow** the existing code style: English comments, one idea per file, no monolithic files
3. **Build** against wxl-core v1.2 and test in game with `WXL_VK_DEVICE=native`
4. **Never commit** anything extracted or derived from the client (`shaders/translated/` is ignored for that reason)
5. **Open** a pull request with a clear description of what changed and how you tested it

## 🌟 Acknowledgements

- **[WarcraftXL](https://github.com/WarcraftXL/wxl-core)**: the modding framework this extension is built on
- **[Khronos Group](https://www.khronos.org/)**: Vulkan, the [Vulkan headers](https://github.com/KhronosGroup/Vulkan-Headers)
  vendored in `third_party/` and [glslang](https://github.com/KhronosGroup/glslang)
- **The [wowdev.wiki](https://wowdev.wiki/) contributors**: years of documented file formats

World of Warcraft is a trademark of Blizzard Entertainment. This project is not affiliated with or endorsed by
Blizzard Entertainment and ships none of its files.

## 📄 License

This project is licensed under the GNU General Public License v3.0. See [LICENSE](LICENSE) for details.

---

<div align="center">
<sub>Part of the <a href="https://warcraftxl.github.io/">WarcraftXL</a> project</sub>

[⬆ Back to Top](#wxl-vulkan-api)
</div>
