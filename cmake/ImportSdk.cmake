# 单点 IMPORTED：板端二进制 SDK（vitgloves_vis_sdk，无源码交付）
# 任何业务工程通过 hand_pipeline::vendor_sdk 链入即可，无需关心路径细节。
#
# 路径优先级：
#   1. HAND_PIPELINE_SDK_PATH（外部覆盖，调试用）
#   2. <submodule>/vendor/vitgloves_vis_sdk_pkg（默认）

set(_VV_DIR "${HAND_PIPELINE_SDK_PATH}")
if(_VV_DIR STREQUAL "")
    set(_VV_DIR "${CMAKE_CURRENT_LIST_DIR}/../vendor/vitgloves_vis_sdk_pkg")
endif()

set(_VV_LIB "${_VV_DIR}/lib/libvitgloves_vis_sdk.a")
set(_VV_INC "${_VV_DIR}/include")
set(_VV_ALGO_INC "${_VV_DIR}/third_party/algo_input_api")
set(_VV_RKNN_INC "${_VV_DIR}/third_party/rknn")

if(NOT EXISTS "${_VV_LIB}")
    message(FATAL_ERROR
        "vitgloves_vis_sdk 静态库缺失：${_VV_LIB}\n"
        "请确认 vendor/vitgloves_vis_sdk_pkg 交付包完整。")
endif()
if(NOT EXISTS "${_VV_INC}/vitgloves_vis_sdk.hpp")
    message(FATAL_ERROR
        "vitgloves_vis_sdk 公开头缺失：${_VV_INC}/vitgloves_vis_sdk.hpp")
endif()

set(_VV_OPENCV_ROOT
    "${DAS_EGO_SDK_PATH}/buildroot/output/rockchip_genrobot_rk3588/build/rknpu2-1.0.0/examples/3rdparty/opencv/opencv-linux-aarch64")
set(_VV_OPENCV_LIBS
    "${_VV_OPENCV_ROOT}/lib/libopencv_video.a"
    "${_VV_OPENCV_ROOT}/lib/libopencv_imgproc.a"
    "${_VV_OPENCV_ROOT}/lib/libopencv_core.a"
    "${_VV_OPENCV_ROOT}/share/OpenCV/3rdparty/lib/libtegra_hal.a"
    "${_VV_OPENCV_ROOT}/share/OpenCV/3rdparty/lib/libzlib.a")

if(NOT TARGET hand_pipeline_vendor_sdk)
    add_library(hand_pipeline_vendor_sdk STATIC IMPORTED GLOBAL)
    set_target_properties(hand_pipeline_vendor_sdk PROPERTIES
        IMPORTED_LOCATION             "${_VV_LIB}"
        INTERFACE_INCLUDE_DIRECTORIES "${_VV_INC};${_VV_ALGO_INC};${_VV_RKNN_INC}"
        INTERFACE_LINK_LIBRARIES      "rknnrt;${_VV_OPENCV_LIBS};pthread;dl"
    )
    add_library(hand_pipeline::vendor_sdk ALIAS hand_pipeline_vendor_sdk)
endif()

message(STATUS "hand_pipeline::vendor_sdk : ${_VV_LIB}")
