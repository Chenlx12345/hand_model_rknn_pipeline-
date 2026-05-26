// RTMPose hand keypoint estimator. Strictly mirrors pipeline_lib.py
// topdown_crop_rgb + decode_simcc + affine_kpts. Input model: 256x256 RGB.
#pragma once

#include "rknn_session.h"
#include "rtmdet.h"

#include <opencv2/core.hpp>

#include <array>
#include <vector>

namespace hand_deploy {

struct Keypoint {
    float x, y;       // ORIGINAL image space
    float conf;       // min(max_logit_x, max_logit_y)
};

constexpr int kNumKpts = 21;

struct PoseConfig {
    int input_size = 256;
    float crop_pad = 1.25f;
    float simcc_split = 2.0f;  // [0,511] / 2.0 -> [0,255.5]
};

class RtmPose {
public:
    RtmPose(const std::string& model_path, const PoseConfig& cfg);

    // Run pose for a single bbox; returns 21 keypoints in original image space.
    std::array<Keypoint, kNumKpts> Estimate(const cv::Mat& bgr_img,
                                            const Detection& det,
                                            double* infer_ms = nullptr);

private:
    RknnSession session_;
    PoseConfig cfg_;

    // sx / sy output indices, identified by dims (we expect 2 outputs both
    // shape (1,21, simcc_w)). Python code returns them in model output order;
    // we preserve session index order: 0 = sx, 1 = sy.
    int simcc_w_ = 0;  // ~ 256 * split
    int sx_index_ = 0;
    int sy_index_ = 1;
};

}  // namespace hand_deploy
