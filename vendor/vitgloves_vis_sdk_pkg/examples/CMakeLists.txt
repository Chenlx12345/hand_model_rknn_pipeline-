# 集成方的最小 demo:链 vitgloves_vis_sdk 静态库 + 集成方自己的 libalgo_input。
# 你已有的 OpenCV4 + Eigen3 + librknnrt.so 会自动从系统 sysroot 找。
cmake_minimum_required(VERSION 3.16)
project(demo_integrator LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)

set(SDK_DIR "${CMAKE_CURRENT_SOURCE_DIR}/..")
find_package(OpenCV REQUIRED COMPONENTS core imgproc video)
find_library(RKNN_LIB rknnrt PATHS /usr/lib /usr/lib/aarch64-linux-gnu)

add_executable(demo_integrator demo_integrator.cpp)
target_include_directories(demo_integrator PRIVATE
    ${SDK_DIR}/include
    ${SDK_DIR}/third_party/algo_input_api
    ${SDK_DIR}/third_party/rknn
    ${OpenCV_INCLUDE_DIRS})
target_link_libraries(demo_integrator PRIVATE
    ${SDK_DIR}/lib/libvitgloves_vis_sdk.a
    ${OpenCV_LIBS}
    ${RKNN_LIB}
    # 由集成方提供:libalgo_input 实现 services::algo 接口(initializeAlgoInput
    # / setAlgoBatchCallback / setAlgoImuCallback / getAlgoCameraCalibration ...)
    algo_input
    pthread dl)
