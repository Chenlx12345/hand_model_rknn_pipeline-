#include "rtmdet.h"

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

inline float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }

// letterbox: resize keeping aspect ratio, pad to new_size×new_size with pad_val.
cv::Mat letterbox_bgr(const cv::Mat& img, int new_size, int pad_val,
                      float* scale_out, int* x0_out, int* y0_out) {
    const int H = img.rows;
    const int W = img.cols;
    const float scale = static_cast<float>(new_size) / std::max(H, W);
    const int nW = static_cast<int>(std::round(W * scale));
    const int nH = static_cast<int>(std::round(H * scale));
    cv::Mat resized;
    cv::resize(img, resized, cv::Size(nW, nH), 0, 0, cv::INTER_LINEAR);
    cv::Mat out(new_size, new_size, CV_8UC3,
                cv::Scalar(pad_val, pad_val, pad_val));
    const int x0 = (new_size - nW) / 2;
    const int y0 = (new_size - nH) / 2;
    resized.copyTo(out(cv::Rect(x0, y0, nW, nH)));
    *scale_out = scale;
    *x0_out = x0;
    *y0_out = y0;
    return out;
}

struct IndexedDet {
    float x1, y1, x2, y2, score;
};

std::vector<int> greedy_nms(const std::vector<IndexedDet>& dets, float iou_thr) {
    const int N = static_cast<int>(dets.size());
    std::vector<int> order(N);
    for (int i = 0; i < N; ++i) order[i] = i;
    std::sort(order.begin(), order.end(),
              [&](int a, int b) { return dets[a].score > dets[b].score; });

    std::vector<float> areas(N);
    for (int i = 0; i < N; ++i) {
        areas[i] = (dets[i].x2 - dets[i].x1) * (dets[i].y2 - dets[i].y1);
    }

    std::vector<char> removed(N, 0);
    std::vector<int> keep;
    keep.reserve(N);
    for (int oi = 0; oi < N; ++oi) {
        const int i = order[oi];
        if (removed[i]) continue;
        keep.push_back(i);
        for (int oj = oi + 1; oj < N; ++oj) {
            const int j = order[oj];
            if (removed[j]) continue;
            const float xx1 = std::max(dets[i].x1, dets[j].x1);
            const float yy1 = std::max(dets[i].y1, dets[j].y1);
            const float xx2 = std::min(dets[i].x2, dets[j].x2);
            const float yy2 = std::min(dets[i].y2, dets[j].y2);
            const float w = std::max(0.0f, xx2 - xx1);
            const float h = std::max(0.0f, yy2 - yy1);
            const float inter = w * h;
            const float iou = inter / (areas[i] + areas[j] - inter + 1e-9f);
            if (iou > iou_thr) removed[j] = 1;
        }
    }
    return keep;
}

}  // namespace

RtmDet::RtmDet(const std::string& model_path, const DetConfig& cfg)
    : session_(model_path), cfg_(cfg) {
    if (session_.input_h() != cfg.input_size ||
        session_.input_w() != cfg.input_size) {
        throw std::runtime_error(
            "rtmdet model input size != cfg.input_size; model=" +
            std::to_string(session_.input_h()) + "x" +
            std::to_string(session_.input_w()));
    }
    // Sort outputs into cls (C=1) and reg (C=4) branches, by spatial stride asc.
    // RKNN exports NCHW dims by default for ONNX-NCHW models.
    const auto& outs = session_.outputs();
    if (outs.size() != 6) {
        throw std::runtime_error("rtmdet expected 6 outputs, got " +
                                 std::to_string(outs.size()));
    }
    for (std::size_t i = 0; i < outs.size(); ++i) {
        const auto& d = outs[i].dims;
        if (d.size() != 4) {
            throw std::runtime_error("rtmdet output not 4-D");
        }
        // NCHW: (N=1, C, H, W)
        const int C = d[1];
        const int H = d[2];
        const int W = d[3];
        const int stride = cfg.input_size / H;
        Branch br{static_cast<int>(i), H, W, stride};
        if (C == 1) cls_branches_.push_back(br);
        else if (C == 4) reg_branches_.push_back(br);
        else throw std::runtime_error("rtmdet output unexpected C=" +
                                      std::to_string(C));
    }
    if (cls_branches_.size() != 3 || reg_branches_.size() != 3) {
        throw std::runtime_error("rtmdet expected 3 cls + 3 reg branches");
    }
    auto by_stride = [](const Branch& a, const Branch& b) {
        return a.stride < b.stride;
    };
    std::sort(cls_branches_.begin(), cls_branches_.end(), by_stride);
    std::sort(reg_branches_.begin(), reg_branches_.end(), by_stride);
}

std::vector<Detection> RtmDet::Detect(const cv::Mat& bgr_img, double* infer_ms) {
    float scale = 1.0f;
    int x0 = 0, y0 = 0;
    cv::Mat lb = letterbox_bgr(bgr_img, cfg_.input_size, cfg_.pad_val,
                               &scale, &x0, &y0);

    // RKNN takes uint8 NHWC; cv::Mat is BGR uint8 HWC already contiguous.
    if (!lb.isContinuous()) lb = lb.clone();

    const double t0 = now_ms();
    auto outputs = session_.Run(lb.ptr<std::uint8_t>(),
                                static_cast<std::size_t>(lb.total() * 3));
    if (infer_ms) *infer_ms = now_ms() - t0;

    std::vector<IndexedDet> cands;
    cands.reserve(256);

    for (int bi = 0; bi < 3; ++bi) {
        const Branch& cb = cls_branches_[bi];
        const Branch& rb = reg_branches_[bi];
        if (cb.H != rb.H || cb.W != rb.W || cb.stride != rb.stride) {
            throw std::runtime_error("rtmdet cls/reg stride mismatch");
        }
        const int H = cb.H, W = cb.W, S = cb.stride;
        const float* cls = outputs[cb.out_index].data.data();  // shape (1,1,H,W)
        const float* reg = outputs[rb.out_index].data.data();  // shape (1,4,H,W)
        const int HW = H * W;
        for (int y = 0; y < H; ++y) {
            for (int x = 0; x < W; ++x) {
                const int idx = y * W + x;
                const float logit = cls[idx];
                const float p = sigmoid(logit);
                if (p <= cfg_.score_thr) continue;
                const float cx = (x + 0.5f) * S;
                const float cy = (y + 0.5f) * S;
                const float l = reg[0 * HW + idx];
                const float t = reg[1 * HW + idx];
                const float r = reg[2 * HW + idx];
                const float b = reg[3 * HW + idx];
                IndexedDet d;
                d.x1 = cx - l;
                d.y1 = cy - t;
                d.x2 = cx + r;
                d.y2 = cy + b;
                d.score = p;
                cands.push_back(d);
            }
        }
    }

    auto keep = greedy_nms(cands, cfg_.nms_iou_thr);
    if (static_cast<int>(keep.size()) > cfg_.top_k) {
        keep.resize(cfg_.top_k);
    }

    std::vector<Detection> result;
    result.reserve(keep.size());
    // 反 letterbox: (b - pad) / scale
    for (int k : keep) {
        const auto& d = cands[k];
        Detection out;
        out.x1 = (d.x1 - x0) / scale;
        out.y1 = (d.y1 - y0) / scale;
        out.x2 = (d.x2 - x0) / scale;
        out.y2 = (d.y2 - y0) / scale;
        out.score = d.score;
        result.push_back(out);
    }
    return result;
}

}  // namespace hand_deploy
