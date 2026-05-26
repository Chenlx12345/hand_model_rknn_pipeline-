#include "rtmpose.h"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <time.h>

namespace hand_deploy {

namespace {

double now_ms() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1.0e6;
}

}  // namespace

RtmPose::RtmPose(const std::string& model_path, const PoseConfig& cfg)
    : session_(model_path), cfg_(cfg) {
    if (session_.input_h() != cfg.input_size ||
        session_.input_w() != cfg.input_size) {
        throw std::runtime_error(
            "rtmpose model input size mismatch; got " +
            std::to_string(session_.input_h()) + "x" +
            std::to_string(session_.input_w()));
    }
    const auto& outs = session_.outputs();
    if (outs.size() != 2) {
        throw std::runtime_error("rtmpose expected 2 outputs, got " +
                                 std::to_string(outs.size()));
    }
    // 期望两路输出 dims=(1, 21, simcc_w)；以最后一维为 simcc_w。
    for (const auto& o : outs) {
        if (o.dims.size() != 3 || o.dims[1] != kNumKpts) {
            throw std::runtime_error("rtmpose output unexpected dims");
        }
    }
    simcc_w_ = outs[0].dims[2];
    if (outs[1].dims[2] != simcc_w_) {
        throw std::runtime_error("rtmpose sx/sy simcc_w mismatch");
    }
    const int expect_w = static_cast<int>(std::round(cfg.input_size * cfg.simcc_split));
    if (simcc_w_ != expect_w) {
        // 不是致命错——告警即可；仍按模型给出的 W 使用。
        std::fprintf(stderr,
                     "[rtmpose] warn: simcc_w=%d, expected %d "
                     "(input=%d * split=%.1f)\n",
                     simcc_w_, expect_w, cfg.input_size, cfg.simcc_split);
    }
}

std::array<Keypoint, kNumKpts> RtmPose::Estimate(const cv::Mat& bgr_img,
                                                 const Detection& det,
                                                 double* infer_ms) {
    const float cx = (det.x1 + det.x2) * 0.5f;
    const float cy = (det.y1 + det.y2) * 0.5f;
    const float w = det.x2 - det.x1;
    const float h = det.y2 - det.y1;
    const float bw = std::max(w, h) * cfg_.crop_pad;  // bw == bh
    const float half_b = bw * 0.5f;
    const float size_f = static_cast<float>(cfg_.input_size);

    cv::Point2f src[3] = {
        cv::Point2f(cx,          cy),
        cv::Point2f(cx,          cy - half_b),
        cv::Point2f(cx + half_b, cy),
    };
    cv::Point2f dst[3] = {
        cv::Point2f(size_f * 0.5f, size_f * 0.5f),
        cv::Point2f(size_f * 0.5f, 0.0f),
        cv::Point2f(size_f,        size_f * 0.5f),
    };
    cv::Mat M = cv::getAffineTransform(src, dst);

    cv::Mat patch_bgr;
    cv::warpAffine(bgr_img, patch_bgr, M, cv::Size(cfg_.input_size, cfg_.input_size),
                   cv::INTER_LINEAR);
    cv::Mat patch_rgb;
    cv::cvtColor(patch_bgr, patch_rgb, cv::COLOR_BGR2RGB);
    if (!patch_rgb.isContinuous()) patch_rgb = patch_rgb.clone();

    const double t0 = now_ms();
    auto outputs = session_.Run(patch_rgb.ptr<std::uint8_t>(),
                                static_cast<std::size_t>(patch_rgb.total() * 3));
    if (infer_ms) *infer_ms = now_ms() - t0;

    const float* sx = outputs[sx_index_].data.data();  // (1, 21, W)
    const float* sy = outputs[sy_index_].data.data();
    const int W = simcc_w_;

    // 反 affine: M 是 src->dst 的 2x3 仿射；用 invertAffineTransform 得到 dst->src。
    cv::Mat M_inv;
    cv::invertAffineTransform(M, M_inv);  // 2x3 double
    const double* mi = M_inv.ptr<double>();

    std::array<Keypoint, kNumKpts> out{};
    for (int k = 0; k < kNumKpts; ++k) {
        const float* row_x = sx + k * W;
        const float* row_y = sy + k * W;
        int ax = 0;
        float vx = row_x[0];
        for (int i = 1; i < W; ++i) {
            if (row_x[i] > vx) { vx = row_x[i]; ax = i; }
        }
        int ay = 0;
        float vy = row_y[0];
        for (int i = 1; i < W; ++i) {
            if (row_y[i] > vy) { vy = row_y[i]; ay = i; }
        }
        const float kx = ax / cfg_.simcc_split;  // -> 256-crop space
        const float ky = ay / cfg_.simcc_split;
        const double X = mi[0] * kx + mi[1] * ky + mi[2];
        const double Y = mi[3] * kx + mi[4] * ky + mi[5];
        out[k].x = static_cast<float>(X);
        out[k].y = static_cast<float>(Y);
        out[k].conf = std::min(vx, vy);
    }
    return out;
}

}  // namespace hand_deploy
