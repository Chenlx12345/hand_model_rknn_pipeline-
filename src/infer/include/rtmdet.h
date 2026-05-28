// RTMDet hand detector wrapper. Strictly mirrors pipeline_lib.py decode_rtmdet
// + letterbox + unletterbox_bboxes. Input model expected: 640x640, BGR.
#pragma once

#include "rknn_session.h"

#include <opencv2/core.hpp>

#include <cstdint>
#include <vector>

namespace hand_pipeline {

struct Detection {
    float x1, y1, x2, y2;  // xyxy in ORIGINAL image space
    float score;
};

struct DetConfig {
    int input_size = 640;       // letterbox target
    int pad_val = 114;          // gray pad
    float score_thr = 0.4f;
    float nms_iou_thr = 0.6f;
    int top_k = 100;
    // strides 8/16/32 are implicit via output spatial dims vs input_size
};

class RtmDet {
public:
    RtmDet(const std::string& model_path, const DetConfig& cfg);

    // Returns detections in ORIGINAL image coordinates.
    // Sets *latency_ms to inference time of the RKNN call (excludes pre/post).
    std::vector<Detection> Detect(const cv::Mat& bgr_img,
                                  double* infer_ms = nullptr);

private:
    RknnSession session_;
    DetConfig cfg_;

    // Output channel layout — discovered once at construction.
    // 6 outputs: 3 cls (C=1) and 3 reg (C=4). We sort each set by stride asc.
    struct Branch {
        int out_index;  // index into session.outputs()
        int H;
        int W;
        int stride;
    };
    std::vector<Branch> cls_branches_;  // ascending stride
    std::vector<Branch> reg_branches_;  // ascending stride
};

}  // namespace hand_pipeline
