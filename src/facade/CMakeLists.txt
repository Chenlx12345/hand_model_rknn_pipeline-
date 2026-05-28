# hand_pipeline::hand_gt —— 业务唯一对外门面。

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
)

add_library(hand_pipeline::hand_gt ALIAS hand_pipeline_facade)
