# 单点 IMPORTED：板端二进制 SDK（无源码交付）
# 任何业务工程通过 hand_pipeline::vendor_sdk 链入即可，无需关心路径细节。
#
# 路径优先级：
#   1. HAND_PIPELINE_SDK_PATH（外部覆盖，调试用）
#   2. <submodule>/vendor/led_box_pose_sdk（默认）

set(_LBP_DIR "${HAND_PIPELINE_SDK_PATH}")
if(_LBP_DIR STREQUAL "")
    set(_LBP_DIR "${CMAKE_CURRENT_LIST_DIR}/../vendor/led_box_pose_sdk")
endif()

set(_LBP_LIB "${_LBP_DIR}/lib/aarch64/libled_box_pose_sdk.a")
set(_LBP_INC "${_LBP_DIR}/include")

if(NOT EXISTS "${_LBP_LIB}")
    message(FATAL_ERROR
        "led_box_pose_sdk 静态库缺失：${_LBP_LIB}\n"
        "请确认 vendor/led_box_pose_sdk 交付包完整。")
endif()
if(NOT EXISTS "${_LBP_INC}/led_box_pose_sdk.hpp")
    message(FATAL_ERROR
        "led_box_pose_sdk 公开头缺失：${_LBP_INC}/led_box_pose_sdk.hpp")
endif()

if(NOT TARGET hand_pipeline_vendor_sdk)
    add_library(hand_pipeline_vendor_sdk STATIC IMPORTED GLOBAL)
    set_target_properties(hand_pipeline_vendor_sdk PROPERTIES
        IMPORTED_LOCATION             "${_LBP_LIB}"
        INTERFACE_INCLUDE_DIRECTORIES "${_LBP_INC}"
        INTERFACE_LINK_LIBRARIES      "rknnrt"
    )
    add_library(hand_pipeline::vendor_sdk ALIAS hand_pipeline_vendor_sdk)
endif()

message(STATUS "hand_pipeline::vendor_sdk : ${_LBP_LIB}")
