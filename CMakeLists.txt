# hand_model_rknn_pipeline —— 板端手部 GT 仓伞顶点
#
# 子目录：
#   - vendor/     二进制交付 SDK（vitgloves_vis_sdk_pkg，自带 assets/{config,led,model}）
#   - src/infer/  板端 RTMDet+RTMPose 推理库（L1 开源）
#
# 业务工程集成：
#   add_subdirectory(external/hand_model_rknn_pipeline)
#   target_link_libraries(<your_service> PRIVATE hand_pipeline::vendor_sdk)

cmake_minimum_required(VERSION 3.16)
project(hand_pipeline LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(HAND_PIPELINE_BUILD_INFER     "Build open-source infer library"  ON)
option(HAND_PIPELINE_BUILD_TESTS     "Build tests"                      OFF)
set(HAND_PIPELINE_SDK_PATH "" CACHE PATH
    "Override path to vendor/vitgloves_vis_sdk_pkg (debug only)")

include(${CMAKE_CURRENT_LIST_DIR}/cmake/ImportSdk.cmake)

if(HAND_PIPELINE_BUILD_INFER AND EXISTS "${CMAKE_CURRENT_LIST_DIR}/src/infer/CMakeLists.txt")
    add_subdirectory(src/infer)
endif()

if(HAND_PIPELINE_BUILD_TESTS)
    enable_testing()
endif()

message(STATUS "hand_pipeline configured: infer=${HAND_PIPELINE_BUILD_INFER}")
