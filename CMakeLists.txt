# hand_model_rknn_pipeline —— 板端手部 / LED-box GT 仓伞顶点
#
# 包含：
#   - tools/        主机侧 ONNX→RKNN 转换 + bench（Python，与板端隔离）
#   - assets/       模型 / 配置 / box-mesh（部署单点）
#   - vendor/       二进制交付 SDK（led_box_pose_sdk，无源码）
#   - infer/        板端开源 RTMDet+RTMPose 推理库（PR2 后迁到 src/infer）
#   - src/{facade,adapter}/  上层门面与算法输入适配（PR3+ 落地）
#
# 业务工程集成方式：
#   add_subdirectory(external/hand_model_rknn_pipeline)
#   target_link_libraries(<your_service> PRIVATE hand_pipeline::vendor_sdk)
# 等 PR3 之后将切换为：
#   target_link_libraries(<your_service> PRIVATE hand_pipeline::facade)

cmake_minimum_required(VERSION 3.16)
project(hand_pipeline LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ── 选项 ────────────────────────────────────────────────────────────────
option(HAND_PIPELINE_BUILD_INFER     "Build open-source infer library"          ON)
option(HAND_PIPELINE_BUILD_FACADE    "Build facade (PR3+)"                      OFF)
option(HAND_PIPELINE_BUILD_ADAPTER   "Build algo_input adapter (PR4+)"          OFF)
option(HAND_PIPELINE_BUILD_TESTS     "Build tests"                              OFF)
option(HAND_PIPELINE_BUILD_EXAMPLES  "Build examples"                           OFF)
set(HAND_PIPELINE_SDK_PATH "" CACHE PATH
    "Override path to vendor/led_box_pose_sdk (debug only)")

# ── vendor SDK (单点 IMPORTED) ─────────────────────────────────────────
include(${CMAKE_CURRENT_LIST_DIR}/cmake/ImportSdk.cmake)

# ── L1 infer (现 infer/，PR2 迁到 src/infer/) ──────────────────────────
if(HAND_PIPELINE_BUILD_INFER AND EXISTS "${CMAKE_CURRENT_LIST_DIR}/infer/CMakeLists.txt")
    add_subdirectory(infer)
endif()

# ── L3 facade（占位；PR3 落地） ────────────────────────────────────────
# if(HAND_PIPELINE_BUILD_FACADE)
#     add_subdirectory(src/facade)
# endif()

# ── L4 adapter（占位；PR4 落地） ───────────────────────────────────────
# if(HAND_PIPELINE_BUILD_ADAPTER)
#     add_subdirectory(src/adapter)
# endif()

if(HAND_PIPELINE_BUILD_TESTS)
    enable_testing()
    # tests/CMakeLists.txt 在 PR3+ 引入
endif()

message(STATUS "hand_pipeline configured: infer=${HAND_PIPELINE_BUILD_INFER} "
               "facade=${HAND_PIPELINE_BUILD_FACADE} "
               "adapter=${HAND_PIPELINE_BUILD_ADAPTER}")
