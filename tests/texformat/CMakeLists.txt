# Standalone build of the TexFormat decoder test. Not wired into the extension: the extension is a
# DLL that loads inside the client, and a test that runs on the desktop has no place in its target.
#
#   cmake -S extensions/wxl-vulkan-api/tests/texformat -B <scratch> -A Win32
#   cmake --build <scratch> --config Release
#   <scratch>/Release/texformat_test.exe
#
# Win32 is required, not merely matched: vulkan/Vk.hpp static_asserts that the Vulkan handles were
# parsed as 32-bit, and TexFormat.hpp includes it the way every sibling in src/vulkan/ does.
#
# C++17 and /W4 /WX here, where the extension itself builds as C++20 without /WX: the module is
# promised to be clean under both, and this is the build that enforces it.
# Copyright (C) 2026 WarcraftXL. GPLv3 (see source headers).

cmake_minimum_required(VERSION 3.20)
project(texformat_test LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_SIZEOF_VOID_P EQUAL 4)
    message(FATAL_ERROR "configure with -A Win32: vulkan/Vk.hpp asserts a 32-bit parse")
endif()

get_filename_component(ext_root "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
get_filename_component(core_root "${ext_root}/../.." ABSOLUTE)

add_executable(texformat_test
    "${CMAKE_CURRENT_SOURCE_DIR}/texformat_test.cpp"
    "${ext_root}/src/vulkan/TexFormat.cpp")

target_include_directories(texformat_test PRIVATE
    "${ext_root}/src"
    "${ext_root}/third_party/vulkan-headers"
    "${core_root}/include")

# The same defines the extension builds with (cmake/WxlDefaults.cmake), so windows.h is parsed the
# same way; the test adds nothing of its own.
target_compile_definitions(texformat_test PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX _CRT_SECURE_NO_WARNINGS)

if(MSVC)
    target_compile_options(texformat_test PRIVATE /W4 /WX /permissive-)
else()
    target_compile_options(texformat_test PRIVATE -Wall -Wextra -Werror)
endif()
