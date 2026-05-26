// Evaluation: load COCO-format val.json, greedy IoU match, accumulate
// detection recall / precision and per-keypoint distance for PCK.
// Mirrors bench_e2e.py:253-344.
#pragma once

#include "rtmdet.h"
#include "rtmpose.h"

#include <array>
#include <string>
#include <vector>

namespace hand_deploy {

struct GtAnnotation {
    float bx1, by1, bx2, by2;
    // 21 (x, y, v) keypoints; v > 0 == visible
    std::array<float, kNumKpts * 3> kpts;
};

struct ImageRecord {
    std::string file_name;
    std::vector<GtAnnotation> gts;
};

// Parse a COCO-format val.json into per-image records.
// `img_dir` is used to filter out images that don't exist on disk.
std::vector<ImageRecord> LoadCocoEval(const std::string& json_path,
                                      const std::string& img_dir);

// Greedy IoU matching: return list of (pred_idx, gt_idx) pairs, IoU >= thr.
std::vector<std::pair<int, int>> MatchPredsToGt(
    const std::vector<Detection>& preds,
    const std::vector<GtAnnotation>& gts,
    float iou_thr = 0.5f);

}  // namespace hand_deploy
