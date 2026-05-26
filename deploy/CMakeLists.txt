# bench_e2e_rknn — board-side end-to-end validation of converted RKNN models.
#
# Cross-compile with the main repo toolchain:
#   cmake -DCMAKE_TOOLCHAIN_FILE=<repo>/cmake/toolchain-rk3588.cmake -B build .
#   cmake --build build -j
cmake_minimum_required(VERSION 3.16)
project(hand_model_rknn_pipeline_deploy CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

add_compile_options(-Wall -Wextra -O2)

# 路径来源：toolchain-rk3588.cmake include 了 local-config.cmake，
# 因此 DAS_EGO_SDK_PATH 在此处一定已就绪。
if(NOT DEFINED DAS_EGO_SDK_PATH OR DAS_EGO_SDK_PATH STREQUAL "")
    message(FATAL_ERROR
        "DAS_EGO_SDK_PATH not set. Pass -DCMAKE_TOOLCHAIN_FILE pointing to "
        "<repo>/cmake/toolchain-rk3588.cmake")
endif()

set(SYSROOT
    "${DAS_EGO_SDK_PATH}/buildroot/output/rockchip_genrobot_rk3588/host/aarch64-buildroot-linux-gnu/sysroot")

# ── RKNN runtime ─────────────────────────────────────────────────────────
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

# ── nlohmann/json (already vendored at repo's external/) ─────────────────
# external/nlohmann_json relative path from this CMakeLists is ../../../
get_filename_component(REPO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../.." ABSOLUTE)
set(NLOHMANN_INCLUDE_DIR "${REPO_ROOT}/external/nlohmann_json/single_include")
if(NOT EXISTS "${NLOHMANN_INCLUDE_DIR}/nlohmann/json.hpp")
    message(FATAL_ERROR "nlohmann/json.hpp not found at ${NLOHMANN_INCLUDE_DIR}")
endif()

# ── target ───────────────────────────────────────────────────────────────
add_executable(bench_e2e_rknn
    src/main.cpp
    src/rknn_session.cpp
    src/rtmdet.cpp
    src/rtmpose.cpp
    src/eval.cpp
)
target_include_directories(bench_e2e_rknn PRIVATE
    src
    ${RKNN_INCLUDE_DIR}
    ${OPENCV_INCLUDE_DIR}
    ${NLOHMANN_INCLUDE_DIR}
)
target_link_libraries(bench_e2e_rknn PRIVATE
    ${RKNN_LIB}
    ${OPENCV_LIBS}
    pthread
    dl
    rt
    m
)

set_target_properties(bench_e2e_rknn PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)
