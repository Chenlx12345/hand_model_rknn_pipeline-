#include "hand_pipeline.h"

#include <time.h>

namespace hand_deploy {

namespace {

double now_ms() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1.0e6;
}

}  // namespace

HandPipeline::HandPipeline(const std::string& det_rknn,
                           const std::string& pose_rknn,
                           const DetConfig& dcfg,
                           const PoseConfig& pcfg)
    : det_(det_rknn, dcfg), pose_(pose_rknn, pcfg) {}

std::vector<HandResult> HandPipeline::Run(const cv::Mat& bgr_img,
                                          HandTiming* timing) {
    const double t0 = timing ? now_ms() : 0.0;

    double det_ms = 0.0;
    auto dets = det_.Detect(bgr_img, &det_ms);

    std::vector<HandResult> out;
    out.reserve(dets.size());
    if (timing) timing->pose_ms.reserve(dets.size());

    for (const auto& d : dets) {
        double pose_ms = 0.0;
        auto kpts = pose_.Estimate(bgr_img, d, &pose_ms);
        out.push_back(HandResult{d, kpts});
        if (timing) timing->pose_ms.push_back(pose_ms);
    }

    if (timing) {
        timing->det_ms   = det_ms;
        timing->total_ms = now_ms() - t0;
    }
    return out;
}

}  // namespace hand_deploy
