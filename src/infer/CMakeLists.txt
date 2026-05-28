# hand_pipeline — board-side RKNN inference library (RTMDet + RTMPose).
#
# Cross-compile via the main repo toolchain:
#   cmake -DCMAKE_TOOLCHAIN_FILE=<repo>/cmake/toolchain-rk3588.cmake -B build .
#   cmake --build build -j
#
# Standalone configure builds tests by default; downstream consumers integrating
# via add_subdirectory get the lib only unless they explicitly set
# -DHAND_PIPELINE_BUILD_TESTS=ON.

# 仅 standalone 配置（直接 cmake 本目录）时建立 project 上下文与全局编译设置；
# 被 add_subdirectory 集成时复用父项目设置，避免 project() 同名嵌套与 flag 重复。
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    cmake_minimum_required(VERSION 3.16)
    project(hand_pipeline_infer CXX)
    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    set(CMAKE_CXX_EXTENSIONS OFF)
    if(NOT CMAKE_BUILD_TYPE)
        set(CMAKE_BUILD_TYPE Release)
    endif()
    add_compile_options(-Wall -Wextra -O2)
endif()

# 路径来源：toolchain-rk3588.cmake include 了 local-config.cmake，
# DAS_EGO_SDK_PATH 在 toolchain 加载完成后必然就绪。
if(NOT DEFINED DAS_EGO_SDK_PATH OR DAS_EGO_SDK_PATH STREQUAL "")
    message(FATAL_ERROR
        "DAS_EGO_SDK_PATH not set. Pass -DCMAKE_TOOLCHAIN_FILE pointing to "
        "<repo>/cmake/toolchain-rk3588.cmake")
endif()

set(SYSROOT
    "${DAS_EGO_SDK_PATH}/buildroot/output/rockchip_genrobot_rk3588/host/aarch64-buildroot-linux-gnu/sysroot")

# ── RKNN runtime (sysroot) ───────────────────────────────────────────────
set(RKNN_INCLUDE_DIR "${SYSROOT}/usr/include/rknn")
set(RKNN_LIB         "${SYSROOT}/usr/lib/librknnrt.so")
if(NOT EXISTS "${RKNN_INCLUDE_DIR}/rknn_api.h")
    message(FATAL_ERROR "rknn_api.h not found at ${RKNN_INCLUDE_DIR}")
endif()
if(NOT EXISTS "${RKNN_LIB}")
    message(FATAL_ERROR "librknnrt.so not found at ${RKNN_LIB}")
endif()
message(STATUS "RKNN include : ${RKNN_INCLUDE_DIR}")
message(STATUS "RKNN library : ${RKNN_LIB}")

# ── OpenCV 3.4.5 (vendored under rknpu2 examples; static) ────────────────
set(OPENCV_ROOT
    "${DAS_EGO_SDK_PATH}/buildroot/output/rockchip_genrobot_rk3588/build/rknpu2-1.0.0/examples/3rdparty/opencv/opencv-linux-aarch64")
set(OPENCV_INCLUDE_DIR "${OPENCV_ROOT}/include")
set(OPENCV_LIB_DIR     "${OPENCV_ROOT}/lib")
set(OPENCV_3RD_DIR     "${OPENCV_ROOT}/share/OpenCV/3rdparty/lib")
if(NOT EXISTS "${OPENCV_INCLUDE_DIR}/opencv2/opencv.hpp")
    message(FATAL_ERROR "OpenCV headers not found at ${OPENCV_INCLUDE_DIR}")
endif()
# 静态链接顺序参考 opencv.pc Libs / Libs.private（高层模块在前，3rdparty 在后）
set(OPENCV_LIBS
    "${OPENCV_LIB_DIR}/libopencv_imgcodecs.a"
    "${OPENCV_LIB_DIR}/libopencv_imgproc.a"
    "${OPENCV_LIB_DIR}/libopencv_core.a"
    "${OPENCV_3RD_DIR}/liblibjpeg-turbo.a"
    "${OPENCV_3RD_DIR}/liblibpng.a"
    "${OPENCV_3RD_DIR}/liblibtiff.a"
    "${OPENCV_3RD_DIR}/liblibwebp.a"
    "${OPENCV_3RD_DIR}/liblibjasper.a"
    "${OPENCV_3RD_DIR}/libIlmImf.a"
    "${OPENCV_3RD_DIR}/libtegra_hal.a"
    "${OPENCV_3RD_DIR}/libzlib.a"
)
message(STATUS "OpenCV root  : ${OPENCV_ROOT}")

# ── library ──────────────────────────────────────────────────────────────
add_library(hand_pipeline STATIC
    src/rknn_session.cpp
    src/rtmdet.cpp
    src/rtmpose.cpp
    src/hand_pipeline.cpp
)

# PUBLIC: 公共头直接 #include <opencv2/...> 与 <rknn_api.h>，
# 任何 link 此库的目标都需要这些路径与库。
target_include_directories(hand_pipeline
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${RKNN_INCLUDE_DIR}
        ${OPENCV_INCLUDE_DIR}
)

target_link_libraries(hand_pipeline
    PUBLIC
        ${RKNN_LIB}
        ${OPENCV_LIBS}
        pthread
        dl
        rt
        m
)

# ── tests ────────────────────────────────────────────────────────────────
# 独立配置 src/infer/ 时默认开启；被主仓库 add_subdirectory 集成时默认关闭
# （避免拖入 nlohmann/json 与 bench 可执行）。
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    set(_HAND_PIPELINE_TESTS_DEFAULT ON)
else()
    set(_HAND_PIPELINE_TESTS_DEFAULT OFF)
endif()
option(HAND_PIPELINE_BUILD_TESTS "Build hand_pipeline tests"
       ${_HAND_PIPELINE_TESTS_DEFAULT})
if(HAND_PIPELINE_BUILD_TESTS)
    add_subdirectory(tests)
endif()
