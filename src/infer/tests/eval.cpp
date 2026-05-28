#include "eval.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <fstream>
#include <stdexcept>
#include <sys/stat.h>
#include <unordered_map>

namespace hand_pipeline {

namespace {

bool path_exists(const std::string& p) {
    struct stat st{};
    return stat(p.c_str(), &st) == 0;
}

float iou_xyxy(float ax1, float ay1, float ax2, float ay2,
               float bx1, float by1, float bx2, float by2) {
    const float x1 = std::max(ax1, bx1);
    const float y1 = std::max(ay1, by1);
    const float x2 = std::min(ax2, bx2);
    const float y2 = std::min(ay2, by2);
    const float inter = std::max(0.0f, x2 - x1) * std::max(0.0f, y2 - y1);
    const float aa = (ax2 - ax1) * (ay2 - ay1);
    const float bb = (bx2 - bx1) * (by2 - by1);
    return inter / (aa + bb - inter + 1e-9f);
}

}  // namespace

std::vector<ImageRecord> LoadCocoEval(const std::string& json_path,
                                      const std::string& img_dir) {
    std::ifstream fin(json_path);
    if (!fin) {
        throw std::runtime_error("cannot open " + json_path);
    }
    nlohmann::json j;
    fin >> j;

    std::unordered_map<int, std::string> id_to_name;
    for (const auto& im : j["images"]) {
        id_to_name[im["id"].get<int>()] = im["file_name"].get<std::string>();
    }

    std::unordered_map<std::string, ImageRecord> by_name;
    for (const auto& ann : j["annotations"]) {
        const int img_id = ann["image_id"].get<int>();
        auto it = id_to_name.find(img_id);
        if (it == id_to_name.end()) continue;
        const std::string& fname = it->second;

        GtAnnotation g{};
        // bbox: [x, y, w, h]
        const auto& bbox = ann["bbox"];
        const float bx = bbox[0].get<float>();
        const float by = bbox[1].get<float>();
        const float bw = bbox[2].get<float>();
        const float bh = bbox[3].get<float>();
        g.bx1 = bx; g.by1 = by; g.bx2 = bx + bw; g.by2 = by + bh;

        // keypoints: flat list of 63 floats (21 * 3)
        const auto& kp = ann["keypoints"];
        const std::size_t n = std::min<std::size_t>(kp.size(), g.kpts.size());
        for (std::size_t i = 0; i < n; ++i) {
            g.kpts[i] = kp[i].get<float>();
        }
        by_name[fname].file_name = fname;
        by_name[fname].gts.push_back(g);
    }

    std::vector<ImageRecord> result;
    result.reserve(by_name.size());
    for (auto& [name, rec] : by_name) {
        const std::string full = img_dir + "/" + name;
        if (!path_exists(full)) continue;
        result.push_back(std::move(rec));
    }
    // 稳定顺序：按文件名排序，便于多次运行对比。
    std::sort(result.begin(), result.end(),
              [](const ImageRecord& a, const ImageRecord& b) {
                  return a.file_name < b.file_name;
              });
    return result;
}

std::vector<std::pair<int, int>> MatchPredsToGt(
    const std::vector<Detection>& preds,
    const std::vector<GtAnnotation>& gts,
    float iou_thr) {
    struct Cand { float iou; int pi, gi; };
    std::vector<Cand> cands;
    cands.reserve(preds.size() * gts.size());
    for (int pi = 0; pi < static_cast<int>(preds.size()); ++pi) {
        const auto& p = preds[pi];
        for (int gi = 0; gi < static_cast<int>(gts.size()); ++gi) {
            const auto& g = gts[gi];
            const float i = iou_xyxy(p.x1, p.y1, p.x2, p.y2,
                                     g.bx1, g.by1, g.bx2, g.by2);
            if (i >= iou_thr) cands.push_back({i, pi, gi});
        }
    }
    std::sort(cands.begin(), cands.end(),
              [](const Cand& a, const Cand& b) { return a.iou > b.iou; });

    std::vector<char> used_p(preds.size(), 0), used_g(gts.size(), 0);
    std::vector<std::pair<int, int>> pairs;
    for (const auto& c : cands) {
        if (used_p[c.pi] || used_g[c.gi]) continue;
        used_p[c.pi] = used_g[c.gi] = 1;
        pairs.emplace_back(c.pi, c.gi);
    }
    return pairs;
}

}  // namespace hand_pipeline
