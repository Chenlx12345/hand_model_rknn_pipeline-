// 板端测试可视化实现; 字面对齐 scripts/viz_lib.py:draw_hands。
#include "viz_lib.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>

namespace hand_pipeline::viz {

namespace {

// Per-finger BGR. cv::Scalar 构造不是 constexpr, 用 const 即可。
const cv::Scalar kThumb (0,   165, 255);   // orange
const cv::Scalar kIndex (0,   255, 255);   // yellow
const cv::Scalar kMiddle(0,   255, 0);     // green
const cv::Scalar kRing  (255, 128, 0);     // cyan-ish blue
const cv::Scalar kPinky (255, 0,   255);   // magenta

inline int iround(float v) {
    return static_cast<int>(std::lround(v));
}

}  // namespace

const std::array<std::pair<int, int>, 21> kHandSkeleton = {{
    {0, 1},  {1, 2},   {2, 3},   {3, 4},
    {0, 5},  {5, 6},   {6, 7},   {7, 8},
    {5, 9},  {9, 10},  {10, 11}, {11, 12},
    {9, 13}, {13, 14}, {14, 15}, {15, 16},
    {13, 17}, {0, 17}, {17, 18}, {18, 19}, {19, 20},
}};

const std::array<cv::Scalar, kNumKpts> kKptColors = {{
    kThumb,                                    // 0 wrist
    kThumb,  kThumb,  kThumb,  kThumb,         // 1-4
    kIndex,  kIndex,  kIndex,  kIndex,         // 5-8
    kMiddle, kMiddle, kMiddle, kMiddle,        // 9-12
    kRing,   kRing,   kRing,   kRing,          // 13-16
    kPinky,  kPinky,  kPinky,  kPinky,         // 17-20
}};

const cv::Scalar kBboxColor(0, 255, 0);

cv::Mat DrawHands(
    const cv::Mat& img_bgr,
    const std::vector<Detection>& boxes,
    const std::vector<std::array<Keypoint, kNumKpts>>& kpts_per_pred,
    const std::string& label) {

    // Clone, mirrors Python np.copy(); 调用方共享 LoadedImage::img 安全。
    cv::Mat vis = img_bgr.clone();

    for (std::size_t pi = 0; pi < boxes.size(); ++pi) {
        const Detection& b = boxes[pi];
        const cv::Point p1(iround(b.x1), iround(b.y1));
        const cv::Point p2(iround(b.x2), iround(b.y2));
        cv::rectangle(vis, p1, p2, kBboxColor, 2);

        char buf[32];
        std::snprintf(buf, sizeof(buf), "det=%.2f", b.score);
        const cv::Point txt_org(p1.x, std::max(p1.y - 4, 12));
        cv::putText(vis, buf, txt_org,
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, kBboxColor, 1,
                    cv::LINE_AA);

        if (pi >= kpts_per_pred.size()) continue;
        const auto& kp = kpts_per_pred[pi];

        // 骨架先画, 圆点后画 — 与 Python 顺序一致, 点压在线上更清晰。
        for (const auto& [i, j] : kHandSkeleton) {
            if (i >= kNumKpts || j >= kNumKpts) continue;
            const cv::Point a(iround(kp[i].x), iround(kp[i].y));
            const cv::Point b2(iround(kp[j].x), iround(kp[j].y));
            cv::line(vis, a, b2, kKptColors[j], 1, cv::LINE_AA);
        }
        for (int k = 0; k < kNumKpts; ++k) {
            cv::circle(vis, cv::Point(iround(kp[k].x), iround(kp[k].y)),
                       3, kKptColors[k], -1, cv::LINE_AA);
        }
    }

    if (!label.empty()) {
        const int font = cv::FONT_HERSHEY_SIMPLEX;
        const double scale_ = 0.7;
        const int thick = 2;
        int baseline = 0;
        const cv::Size ts = cv::getTextSize(label, font, scale_, thick,
                                            &baseline);
        const int pad = 4;
        cv::rectangle(vis,
                      cv::Point(0, 0),
                      cv::Point(ts.width + 2 * pad,
                                ts.height + 2 * pad + baseline),
                      cv::Scalar(0, 0, 0), cv::FILLED);
        cv::putText(vis, label, cv::Point(pad, ts.height + pad),
                    font, scale_, cv::Scalar(255, 255, 255), thick,
                    cv::LINE_AA);
    }

    return vis;
}

}  // namespace hand_pipeline::viz
