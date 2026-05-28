# hand_pipeline::hand_gt —— 业务唯一对外门面。
#
# 链接传播策略：
#   - PRIVATE 链 vendor_sdk + OpenCV：facade.cpp 自身能 #include SDK 头并解析符号；
#   - $<LINK_ONLY:…>：把这两组依赖挂到 INTERFACE，仅在最终 exe 链阶段贡献库 / -l 标志，
#     不向下游传播 include 路径。业务工程头视野只看到 hand_pipeline/*.hpp。

set(_LBP_OPENCV_ROOT
    "${DAS_EGO_SDK_PATH}/buildroot/output/rockchip_genrobot_rk3588/build/rknpu2-1.0.0/examples/3rdparty/opencv/opencv-linux-aarch64")
set(_LBP_OPENCV_LIBS
    ${_LBP_OPENCV_ROOT}/lib/libopencv_imgproc.a
    ${_LBP_OPENCV_ROOT}/lib/libopencv_core.a
    ${_LBP_OPENCV_ROOT}/share/OpenCV/3rdparty/lib/libtegra_hal.a
    ${_LBP_OPENCV_ROOT}/share/OpenCV/3rdparty/lib/libzlib.a)

add_library(hand_pipeline_facade STATIC
    hand_gt_pipeline.cpp
)

target_include_directories(hand_pipeline_facade PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
)

target_compile_features(hand_pipeline_facade PUBLIC cxx_std_17)

target_link_libraries(hand_pipeline_facade
    PRIVATE
        hand_pipeline::vendor_sdk
        ${_LBP_OPENCV_LIBS}
    INTERFACE
        $<LINK_ONLY:hand_pipeline::vendor_sdk>
        $<LINK_ONLY:${_LBP_OPENCV_LIBS}>
)

add_library(hand_pipeline::hand_gt ALIAS hand_pipeline_facade)
