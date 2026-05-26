// 板端测试可视化: bbox + 21 关键点 + 骨架 + 标签。
// 字面对齐主机端 scripts/viz_lib.py:draw_hands; 配色、骨架、字体、
// 顺序均与 Python 一致, 便于 eval/viz/ ↔ out/viz/ 直接 diff。
//
// 仅供 unit_bench_e2e_test 消费, 不入 libhand_pipeline。
#pragma once

#include "rtmdet.h"   // Detection
#include "rtmpose.h"  // Keypoint, kNumKpts

#include <opencv2/core.hpp>

#include <array>
#include <string>
#include <utility>
#include <vector>

namespace hand_deploy::viz {

// 21 条骨架线 — 与 scripts/viz_lib.py:HAND_SKELETON 完全一致, 顺序勿动。
extern const std::array<std::pair<int, int>, 21> kHandSkeleton;

// 21 关节颜色 (BGR) — 与 scripts/viz_lib.py:KPT_COLORS 完全一致。
extern const std::array<cv::Scalar, kNumKpts> kKptColors;

// bbox 颜色 (BGR) — 绿色, 与 Python 端 BBOX_COLOR 对齐。
extern const cv::Scalar kBboxColor;

// 在原图上画 bbox + 21 关键点 + 骨架 + 左上角标签, 返回 NEW Mat (clone)。
// boxes[i] 与 kpts_per_pred[i] 按索引一一对应; kpts_per_pred 长度可短于 boxes,
// 此时多出的 box 只画框, 不画骨架。
//
// label 非空时左上角加黑底白字标签 (例如 "RKNN-NPU" / "ONNX")。
cv::Mat DrawHands(
    const cv::Mat& img_bgr,
    const std::vector<Detection>& boxes,
    const std::vector<std::array<Keypoint, kNumKpts>>& kpts_per_pred,
    const std::string& label = "");

}  // namespace hand_deploy::viz
