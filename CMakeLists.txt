# hand_model_rknn_pipeline —— 板端手部 / LED-box GT 仓伞顶点
#
# 子目录：
#   - scripts/    主机侧 ONNX→RKNN 转换 + bench（Python，与板端隔离）
#   - assets/     模型 / 配置 / box-mesh
#   - vendor/     二进制交付 SDK（led_box_pose_sdk）
#   - src/infer/  板端 RTMDet+RTMPose 推理库（L1 开源）
#   - src/facade/ 业务唯一对外门面 hand_pipeline::hand_gt（L3）
#
# 业务工程集成：
#   add_subdirectory(external/hand_model_rknn_pipeline)
#   target_link_libraries(<your_service> PRIVATE hand_pipeline::hand_gt)

cmake_minimum_required(VERSION 3.16)
project(hand_pipeline LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(HAND_PIPELINE_BUILD_INFER     "Build open-source infer library"  ON)
option(HAND_PIPELINE_BUILD_FACADE    "Build hand_gt facade (requires vendor SDK)" ON)
option(HAND_PIPELINE_BUILD_TESTS     "Build tests"                      OFF)
option(HAND_PIPELINE_BUILD_EXAMPLES  "Build examples"                   OFF)
set(HAND_PIPELINE_SDK_PATH "" CACHE PATH
    "Override path to vendor/led_box_pose_sdk (debug only)")

include(${CMAKE_CURRENT_LIST_DIR}/cmake/ImportSdk.cmake)

if(HAND_PIPELINE_BUILD_INFER AND EXISTS "${CMAKE_CURRENT_LIST_DIR}/src/infer/CMakeLists.txt")
    add_subdirectory(src/infer)
endif()

if(HAND_PIPELINE_BUILD_FACADE AND EXISTS "${CMAKE_CURRENT_LIST_DIR}/src/facade/CMakeLists.txt")
    add_subdirectory(src/facade)
endif()

if(HAND_PIPELINE_BUILD_TESTS)
    enable_testing()
endif()

message(STATUS "hand_pipeline configured: infer=${HAND_PIPELINE_BUILD_INFER} "
               "facade=${HAND_PIPELINE_BUILD_FACADE}")
