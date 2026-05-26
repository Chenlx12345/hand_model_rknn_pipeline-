// unit_bench_e2e_test: board-side end-to-end validation of the on-disk .rknn
// pair against the Python pipeline. Single source of truth for the
// pre/post-processing pipeline is scripts/pipeline_lib.py + bench_e2e.py.
#include "eval.h"
#include "rtmdet.h"
#include "rtmpose.h"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>
#include <time.h>

namespace {

using hand_deploy::Detection;
using hand_deploy::GtAnnotation;
using hand_deploy::ImageRecord;
using hand_deploy::Keypoint;
using hand_deploy::kNumKpts;

// 与 Python bench_e2e.py 的 backend 标签风格一致（小写横线），
// 用 "rknn-npu" 区分于主机模拟器的 "rknn-sim"，便于 diff 时一眼识别来源。
constexpr const char* kBackendLabel = "rknn-npu";

// 板端约定布局：单一 --eval 目录下固定子路径，与主机 datasets/eval/ 同名，
// 让"打包整目录拷到板端"成为唯一部署动作，无需额外重命名。
constexpr const char* kEvalAnnName    = "val.json";
constexpr const char* kEvalImagesName = "val_images";

std::string basename(const std::string& p) {
    const auto pos = p.find_last_of('/');
    return pos == std::string::npos ? p : p.substr(pos + 1);
}

double now_ms() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1.0e6;
}

double mean_of(const std::vector<double>& v) {
    if (v.empty()) return 0.0;
    double s = 0.0;
    for (double x : v) s += x;
    return s / v.size();
}

double percentile(std::vector<double> v, double q) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    const double pos = q * (v.size() - 1) / 100.0;
    const std::size_t lo = static_cast<std::size_t>(std::floor(pos));
    const std::size_t hi = static_cast<std::size_t>(std::ceil(pos));
    return v[lo] + (v[hi] - v[lo]) * (pos - lo);
}

void print_help(const char* argv0) {
    std::fprintf(stderr,
        "Usage: %s --det DET.rknn --pose POSE.rknn --eval EVAL_DIR\n"
        "           [--score-thr 0.4] [--nms 0.6] [--warmup 3] [--n 0]\n"
        "  EVAL_DIR must contain '%s' and '%s/'.\n",
        argv0, kEvalAnnName, kEvalImagesName);
}

struct Args {
    std::string det_path;
    std::string pose_path;
    std::string eval_dir;
    float score_thr = 0.4f;
    float nms_thr = 0.6f;
    int warmup = 3;
    int cap_n = 0;  // 0 = all
};

bool parse_args(int argc, char** argv, Args& out) {
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto need = [&](const char* opt) -> const char* {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "FAIL: %s requires an argument\n", opt);
                std::exit(1);
            }
            return argv[++i];
        };
        if (a == "--det") out.det_path = need("--det");
        else if (a == "--pose") out.pose_path = need("--pose");
        else if (a == "--eval") out.eval_dir = need("--eval");
        else if (a == "--score-thr") out.score_thr = std::atof(need("--score-thr"));
        else if (a == "--nms") out.nms_thr = std::atof(need("--nms"));
        else if (a == "--warmup") out.warmup = std::atoi(need("--warmup"));
        else if (a == "--n") out.cap_n = std::atoi(need("--n"));
        else if (a == "-h" || a == "--help") { print_help(argv[0]); std::exit(0); }
        else { std::fprintf(stderr, "FAIL: unknown arg: %s\n", a.c_str()); return false; }
    }
    if (out.det_path.empty() || out.pose_path.empty() || out.eval_dir.empty()) {
        print_help(argv[0]);
        return false;
    }
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    Args args;
    if (!parse_args(argc, argv, args)) return 1;

    std::printf("[load] backend=%s  det=%s  pose=%s\n",
                kBackendLabel,
                basename(args.det_path).c_str(),
                basename(args.pose_path).c_str());

    hand_deploy::DetConfig dcfg;
    dcfg.score_thr = args.score_thr;
    dcfg.nms_iou_thr = args.nms_thr;
    hand_deploy::PoseConfig pcfg;

    hand_deploy::RtmDet det(args.det_path, dcfg);
    hand_deploy::RtmPose pose(args.pose_path, pcfg);

    const std::string ann_path = args.eval_dir + "/" + kEvalAnnName;
    const std::string img_dir  = args.eval_dir + "/" + kEvalImagesName;

    auto records = hand_deploy::LoadCocoEval(ann_path, img_dir);
    if (records.empty()) {
        std::fprintf(stderr, "FAIL: no images under %s match %s\n",
                     img_dir.c_str(), ann_path.c_str());
        return 2;
    }
    if (args.cap_n > 0 && args.cap_n < static_cast<int>(records.size())) {
        records.resize(args.cap_n);
    }

    int n_gt_total = 0;
    for (const auto& r : records) n_gt_total += static_cast<int>(r.gts.size());
    std::printf("[prep] %zu images, %d GT hands\n", records.size(), n_gt_total);

    // ── pre-decode all images (separate disk I/O from inference) ──
    struct LoadedImage { std::string fname; cv::Mat img; const ImageRecord* rec; };
    std::vector<LoadedImage> imgs;
    imgs.reserve(records.size());
    for (const auto& r : records) {
        cv::Mat img = cv::imread(img_dir + "/" + r.file_name, cv::IMREAD_COLOR);
        if (img.empty()) {
            std::fprintf(stderr, "WARN: imread failed: %s\n", r.file_name.c_str());
            continue;
        }
        imgs.push_back({r.file_name, std::move(img), &r});
    }
    if (imgs.empty()) {
        std::fprintf(stderr, "FAIL: no images decoded\n");
        return 3;
    }

    // ── warmup on first image ──
    for (int i = 0; i < args.warmup; ++i) {
        double tmp;
        auto dets = det.Detect(imgs[0].img, &tmp);
        if (!dets.empty()) {
            pose.Estimate(imgs[0].img, dets[0], &tmp);
        }
    }

    // ── timed pass ──
    std::vector<double> det_lats, pose_lats, e2e_lats;
    std::vector<float> pck_diffs;
    int n_pred = 0, n_match = 0, n_pose_calls = 0;

    for (const auto& li : imgs) {
        const double t_start = now_ms();
        double det_ms = 0.0;
        auto preds = det.Detect(li.img, &det_ms);
        det_lats.push_back(det_ms);
        n_pred += static_cast<int>(preds.size());

        std::vector<std::array<Keypoint, kNumKpts>> kpts_per_pred(preds.size());
        for (std::size_t pi = 0; pi < preds.size(); ++pi) {
            double pose_ms = 0.0;
            kpts_per_pred[pi] = pose.Estimate(li.img, preds[pi], &pose_ms);
            pose_lats.push_back(pose_ms);
            ++n_pose_calls;
        }
        e2e_lats.push_back(now_ms() - t_start);

        // 匹配 + PCK
        const auto pairs = hand_deploy::MatchPredsToGt(preds, li.rec->gts);
        n_match += static_cast<int>(pairs.size());
        for (const auto& [pi, gi] : pairs) {
            const auto& gk = li.rec->gts[gi].kpts;
            const auto& pk = kpts_per_pred[pi];
            for (int k = 0; k < kNumKpts; ++k) {
                const float gx = gk[k * 3 + 0];
                const float gy = gk[k * 3 + 1];
                const float gv = gk[k * 3 + 2];
                if (gv <= 0.0f) continue;
                const float dx = pk[k].x - gx;
                const float dy = pk[k].y - gy;
                pck_diffs.push_back(std::sqrt(dx * dx + dy * dy));
            }
        }
    }

    if (e2e_lats.empty()) {
        std::fprintf(stderr, "FAIL: no images evaluated\n");
        return 4;
    }

    const double recall = 100.0 * n_match / std::max(1, n_gt_total);
    const double prec = 100.0 * n_match / std::max(1, n_pred);

    auto pck = [&](float th) {
        if (pck_diffs.empty()) return 0.0f;
        std::size_t hits = 0;
        for (float d : pck_diffs) if (d <= th) ++hits;
        return static_cast<float>(100.0 * hits / pck_diffs.size());
    };

    const double det_mean = mean_of(det_lats);
    const double pose_mean = pose_lats.empty() ? 0.0 : mean_of(pose_lats);
    const double e2e_mean = mean_of(e2e_lats);
    const double e2e_fps = e2e_mean > 0.0 ? 1000.0 / e2e_mean : 0.0;

    std::printf("\n=== accuracy (single pass over %zu images) ===\n", imgs.size());
    std::printf("  GT hands       : %d\n", n_gt_total);
    std::printf("  pred hands     : %d\n", n_pred);
    std::printf("  matched IoU>=0.5: %d  recall=%.1f%%  prec=%.1f%%\n",
                n_match, recall, prec);
    if (!pck_diffs.empty()) {
        double sum = 0.0;
        for (float d : pck_diffs) sum += d;
        std::vector<float> sorted_pck = pck_diffs;
        std::sort(sorted_pck.begin(), sorted_pck.end());
        // np.median 兼容：偶数长度取中间两元素均值，与主机端 Python 输出对齐。
        const std::size_t nk = sorted_pck.size();
        const float median = (nk % 2 == 0)
            ? 0.5f * (sorted_pck[nk / 2 - 1] + sorted_pck[nk / 2])
            : sorted_pck[nk / 2];
        std::printf("  kpts evaluated : %zu\n", pck_diffs.size());
        std::printf("  mean / median  : %.2f / %.2f px\n",
                    sum / pck_diffs.size(), median);
        std::printf("  PCK@5/10/20    : %.1f%% / %.1f%% / %.1f%%\n",
                    pck(5.0f), pck(10.0f), pck(20.0f));
    } else {
        std::printf("  (no matched hands; PCK unavailable)\n");
    }

    std::printf("\n=== latency (%s, board NPU) ===\n", kBackendLabel);
    std::printf("  det /img       : mean=%.2fms p50=%.2f p95=%.2f\n",
                det_mean, percentile(det_lats, 50), percentile(det_lats, 95));
    std::printf("  pose/hand      : mean=%.2fms p50=%.2f p95=%.2f  (%d hands)\n",
                pose_mean,
                pose_lats.empty() ? 0.0 : percentile(pose_lats, 50),
                pose_lats.empty() ? 0.0 : percentile(pose_lats, 95),
                n_pose_calls);
    std::printf("  e2e /img       : mean=%.2fms p50=%.2f p95=%.2f   E2E_FPS=%.2f\n",
                e2e_mean, percentile(e2e_lats, 50), percentile(e2e_lats, 95),
                e2e_fps);

    std::printf("\nPASS: BACKEND=%s DET=%s POSE=%s IMG=%zu RECALL=%.1f%% PCK@5=%.1f%% "
                "E2E_MEAN=%.2fms E2E_FPS=%.2f\n",
                kBackendLabel,
                basename(args.det_path).c_str(),
                basename(args.pose_path).c_str(),
                imgs.size(), recall, pck(5.0f), e2e_mean, e2e_fps);
    return 0;
}
