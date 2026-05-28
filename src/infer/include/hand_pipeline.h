// HandPipeline: one-shot RTMDet -> RTMPose wrapper for board-side consumers.
// Use this when business code only wants "give me hands in the frame";
// drop down to RtmDet / RtmPose directly when granular timing or partial
// pipelines are required (e.g. detection-only).
#pragma once

#include "rtmdet.h"
#include "rtmpose.h"

#include <opencv2/core.hpp>

#include <array>
#include <string>
#include <vector>

namespace hand_pipeline {

struct HandResult {
    Detection                       bbox;   // 原图坐标
    std::array<Keypoint, kNumKpts>  kpts;   // 原图坐标
};

struct HandTiming {
    double              det_ms   = 0.0;   // 一次检测推理耗时
    std::vector<double> pose_ms;          // 每只手一次姿态推理耗时
    double              total_ms = 0.0;   // 端到端 wall-clock（含前后处理）
};

class HandPipeline {
public:
    HandPipeline(const std::string& det_rknn,
                 const std::string& pose_rknn,
                 const DetConfig&  dcfg = {},
                 const PoseConfig& pcfg = {});

    // 单帧推理。返回每只手的 bbox + 21 关键点（全部在原图坐标系）。
    // timing 可为 nullptr。
    std::vector<HandResult> Run(const cv::Mat& bgr_img,
                                HandTiming* timing = nullptr);

private:
    RtmDet  det_;
    RtmPose pose_;
};

}  // namespace hand_pipeline
