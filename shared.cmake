# Build settings wxl_add_extensions() reads for this extension.
#
# The extension's own src/ has to be named here: the loop in cmake/WxlExtensions.cmake puts the
# CORE's include/ and src/ on the include path, not the extension's, so without this every header
# of ours would have to be reached by a relative path from the file including it -- and the
# "device/..." / "native/..." / "vulkan/..." includes that keep the folder layout legible would not
# resolve at all.
#
# WXL_EXT_SHARED_SRC is used for exactly one thing below: it is what the loop appends to its
# GLOB_RECURSE result, and therefore the only way a GENERATED file (a .spv) becomes part of the
# build, since a glob taken at configure time cannot see a file the build has not produced yet.

list(APPEND WXL_EXT_INCLUDE_DIRS "${CMAKE_CURRENT_LIST_DIR}/src")

# include/ holds the PUBLIC headers: wxl/VkRenderApi.h, the rendering API another extension consumes
# (its shared.cmake adds this folder and third_party/vulkan-headers, and nothing of src/).
list(APPEND WXL_EXT_INCLUDE_DIRS "${CMAKE_CURRENT_LIST_DIR}/include")

# The loop's GLOB_RECURSE takes every *.cpp under the extension, tests/ included; this file is
# included in the loop's own scope, after the glob, so the standalone test programs are taken out of
# the DLL's sources here (tests/texformat/CMakeLists.txt builds them on their own).
list(FILTER WXL_EXT_SRC EXCLUDE REGEX "/tests/")

# Link-time code generation for the Release build (cmake/WxlExtensions.cmake reads the flag): the
# draw path runs through a dozen files -- the slot, the reader, the resolver, the pipelines' phases
# -- and without it every call across a file boundary is a real call. Off to A/B: the perf lines
# (WXL_VK_STATS) say what it buys on a machine.
set(WXL_EXT_IPO ON)

# --- Vulkan, with nothing for the developer to install --------------------------------------------
#
# The Khronos headers are vendored under third_party/vulkan-headers, pinned to the tag in its
# VERSION file. Headers only: there is no vulkan-1.lib on the link line and no Vulkan SDK
# prerequisite, because every entry point is resolved at runtime through the loader (VK_NO_PROTOTYPES
# plus vkGetInstanceProcAddr). That is also the only correct shape here -- vulkan-1.dll is a system
# component the client may or may not have, so it has to be a LoadLibrary that can fail and be
# reported, not an import that stops the DLL loading at all.
#
# The 32-bit loader is what matters for this client: C:\Windows\SysWOW64\vulkan-1.dll.
list(APPEND WXL_EXT_INCLUDE_DIRS "${CMAKE_CURRENT_LIST_DIR}/third_party/vulkan-headers")

# --- SPIR-V, compiled at build time when a compiler is present ---------------------------------------
#
# The .vert/.frag under src/shaders/glsl/ are the source of truth; glslang turns each into a .spv that
# is VERSIONED under shaders/ and deployed next to the DLL as Extensions/wxl-vulkan-api/shaders/, where
# src/vulkan/Shaders.cpp loads it module-relative. Versioning the output is the whole point: a developer
# who clones gets working shaders with no tool installed, and the tool is only needed to change one.
#
# glslang is not on any default Windows install (the Windows Kit's dxc is built without SPIR-V code
# generation, and there is no glslc without the Vulkan SDK). Khronos publishes prebuilt binaries on the
# glslang GitHub releases page; point WXL_VK_GLSLANG at one, or at the SDK's.
#
# The translated modules under shaders/translated/ are not built here: they are the output of the
# offline translator (not part of this repository), versioned and deployed as they are.
find_program(WXL_VK_GLSLANG
    NAMES glslang glslangValidator
    HINTS "${WXL_VK_GLSLANG_HINT}" ENV VULKAN_SDK
    PATH_SUFFIXES bin Bin)

set(wxl_vk_glsl "${CMAKE_CURRENT_LIST_DIR}/src/shaders/glsl")
set(wxl_vk_spv  "${CMAKE_CURRENT_LIST_DIR}/shaders")

# wxl_vk_add_spirv(<stem> <source under src/shaders/glsl/> [<extra glslang args>...]) -> shaders/<stem>.spv
#
# A macro so the list(APPEND) lands in the caller's scope. The stage comes from the source's
# extension (.vert/.frag), which is what glslang does with it anyway. One source compiled under
# different -D flags is several stems (the fixed-function shaders below).
macro(wxl_vk_add_spirv wxl_vk_stem wxl_vk_src)
    set(wxl_vk_spv_out "${wxl_vk_spv}/${wxl_vk_stem}.spv")
    if(WXL_VK_GLSLANG)
        # -V      Vulkan semantics (not OpenGL): descriptor sets, push constants, gl_Position rules
        # -Os     size; these are tiny and nothing glslang does to them is a surprise
        # --target-env vulkan1.0   the vendored headers may be newer than the runtime; 1.0 SPIR-V
        #                          (1.0) runs on every loader this extension can meet
        add_custom_command(
            OUTPUT "${wxl_vk_spv_out}"
            COMMAND "${CMAKE_COMMAND}" -E make_directory "${wxl_vk_spv}"
            COMMAND "${WXL_VK_GLSLANG}" -V -Os --target-env vulkan1.0
                    ${ARGN}
                    "${wxl_vk_glsl}/${wxl_vk_src}" -o "${wxl_vk_spv_out}"
            DEPENDS "${wxl_vk_glsl}/${wxl_vk_src}"
            COMMENT "glslang ${wxl_vk_src} -> shaders/${wxl_vk_stem}.spv"
            VERBATIM)
        # Listed as a source so the custom command is part of the build; MSVC does not compile a
        # .spv.
        list(APPEND WXL_EXT_SHARED_SRC "${wxl_vk_spv_out}")
    elseif(NOT EXISTS "${wxl_vk_spv_out}")
        message(WARNING
            "wxl-vulkan-api: glslang not found and shaders/${wxl_vk_stem}.spv is not checked in; "
            "the draw path will decline that shader at runtime. Pass -DWXL_VK_GLSLANG=<path>.")
    endif()
endmacro()

# The UI's two image-mode pixel shaders (CSimpleTexture::Init picks "UI" or "Desaturate") on the UI:0
# vertex shader and its push range: src/vulkan/Pipelines.cpp's hand-written rows name these stems.
wxl_vk_add_spirv(ui_vert ui.vert)
wxl_vk_add_spirv(ui_frag ui.frag)
wxl_vk_add_spirv(desaturate_frag desaturate.frag)

# Fixed function on both halves (vs "" / ps "", cgxdevice section 6): one vertex source per layout
# it is compiled for -- the WXL_FF_FORMAT value is the EGxVertexBufferFormat the row declares -- and
# one fragment source compiled with and without the sampler, because the format-7 (PC) draws have
# no texture and a fragment shader that declares none leaves set 0's descriptor unused. All read the
# reader's fixed-function register file from the ring (draw::Cmd::kFf*).
wxl_vk_add_spirv(ff_pc_vert    ff.vert -DWXL_FF_FORMAT=7)
wxl_vk_add_spirv(ff_pct_vert   ff.vert -DWXL_FF_FORMAT=8)
wxl_vk_add_spirv(ff_pt_vert    ff.vert -DWXL_FF_FORMAT=10)
wxl_vk_add_spirv(ff_frag       ff.frag -DWXL_FF_TEXTURED=1)
wxl_vk_add_spirv(ff_color_frag ff.frag -DWXL_FF_TEXTURED=0)
# The layouts with a normal (4 PNCT, 6 PNCT2) are built lit -- D3D's vertex lighting from the
# device's light slots, cgxdevice 6.4 -- and the two-UV layouts (6, 9 PCT2) get the fragment build
# with a second sampler for texture stage 1. The lit fragment builds add the specular input.
wxl_vk_add_spirv(ff_pnct_vert  ff.vert -DWXL_FF_FORMAT=4 -DWXL_FF_LIT=1)
wxl_vk_add_spirv(ff_pnct2_vert ff.vert -DWXL_FF_FORMAT=6 -DWXL_FF_LIT=1)
wxl_vk_add_spirv(ff_pct2_vert  ff.vert -DWXL_FF_FORMAT=9)
wxl_vk_add_spirv(ff_frag2      ff.frag -DWXL_FF_TEXTURED=2)
wxl_vk_add_spirv(ff_lit_frag   ff.frag -DWXL_FF_TEXTURED=1 -DWXL_FF_LIT=1)
wxl_vk_add_spirv(ff_lit2_frag  ff.frag -DWXL_FF_TEXTURED=2 -DWXL_FF_LIT=1)
# Format 7 (PC) drawn WITH textures -- the ground decals (selection circle, spell markers) generate
# both stages' coordinates from the camera-space position -- pairs this vertex build, which writes
# stage 1's coordinate, with the two-sampler ff_frag2. The untextured format-7 draws keep ff_pc_vert +
# ff_color_frag; src/vulkan/Pipelines.cpp chooses per draw.
wxl_vk_add_spirv(ff_pc2_vert   ff.vert -DWXL_FF_FORMAT=7 -DWXL_FF_STAGE1=1)
# The fixed-function vertex stage that feeds a TRANSLATED pixel module (vs "" with a known ps, the
# engine's full-screen FFX* passes): inputs by semantic location over any layout, the custom ones
# included, outputs in the translated varying layout. One build serves every layout;
# src/vulkan/Pipelines.cpp names the stem (kFfSemanticVertStem).
wxl_vk_add_spirv(ff_sem_vert   ff.vert -DWXL_FF_SEMANTIC_VARYINGS=1)

# The particle emitters: Effect(Particle_Unlit) = Color_T1:0 + Combiners_Mod:4 on format 8 PCT,
# Effect(Particle) = CDiffuse_T1:1 + Combiners_Mod:4 on format 4 PNCT. Both pairs read their constants
# from the uniform ring (set 1); the row table in src/vulkan/Pipelines.cpp names these stems.
wxl_vk_add_spirv(particle_unlit_vert particle_unlit.vert)
wxl_vk_add_spirv(particle_lit_vert   particle_lit.vert)
wxl_vk_add_spirv(combiners_mod_frag  combiners_mod.frag)

# The present pass (the back buffer scaled onto the swapchain, the fullscreen gamma LUT) and the
# core's ImGui overlay drawn natively (src/native/Present.cpp, src/native/Overlay.cpp).
wxl_vk_add_spirv(present_vert present.vert)
wxl_vk_add_spirv(present_frag present.frag)
wxl_vk_add_spirv(overlay_vert overlay.vert)
wxl_vk_add_spirv(overlay_frag overlay.frag)

# The opaque depth kept for the rendering API's consumers (src/native/render/DepthHistory.cpp).
wxl_vk_add_spirv(depthcopy_comp depthcopy.comp)
# Its min / max pyramid, on request (src/native/render/DepthPyramid.cpp): rg32f where the device has the extended
# storage formats, rgba32f where it has not.
wxl_vk_add_spirv(depthpyramid_comp  depthpyramid.comp)
wxl_vk_add_spirv(depthpyramid4_comp depthpyramid.comp -DWXL_PYRAMID_RGBA32F=1)
# The shadow replay's alpha cutout (src/capture/ShadowPipelines.cpp).
wxl_vk_add_spirv(shadow_cutout_frag shadow_cutout.frag)


# --- the shader tree, deployed -----------------------------------------------------------------------
#
# Extensions/wxl-vulkan-api/shaders/ in the client is this folder: the .spv built above, and the offline
# translator's shaders/translated/ (its modules, their `bones` and `wind` variants, manifest.txt and
# index.txt). src/vulkan/Shaders.cpp and Manifest.cpp load them module-relative, so a DLL deployed without
# them would decline the shaders a new manifest names. One copy of the whole folder, only its changed
# files, whenever any of them changed -- after the .spv above are built (they are in the DEPENDS).
if(CLIENT_PATH)
    file(GLOB_RECURSE wxl_vk_shader_files CONFIGURE_DEPENDS
         "${wxl_vk_spv}/*.spv" "${wxl_vk_spv}/translated/manifest.txt" "${wxl_vk_spv}/translated/index.txt")
    set(wxl_vk_shaders_stamp "${CMAKE_CURRENT_BINARY_DIR}/wxl-vulkan-api-shaders.stamp")
    add_custom_command(
        OUTPUT "${wxl_vk_shaders_stamp}"
        COMMAND "${CMAKE_COMMAND}" -E make_directory "${CLIENT_PATH}/Extensions/wxl-vulkan-api/shaders"
        COMMAND "${CMAKE_COMMAND}" -E copy_directory_if_different "${wxl_vk_spv}"
                "${CLIENT_PATH}/Extensions/wxl-vulkan-api/shaders"
        COMMAND "${CMAKE_COMMAND}" -E touch "${wxl_vk_shaders_stamp}"
        DEPENDS ${wxl_vk_shader_files} ${WXL_EXT_SHARED_SRC}
        COMMENT "Deploy Extensions/wxl-vulkan-api/shaders (built and translated modules, manifest)"
        VERBATIM)
    list(APPEND WXL_EXT_SHARED_SRC "${wxl_vk_shaders_stamp}")
endif()
