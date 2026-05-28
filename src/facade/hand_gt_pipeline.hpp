// hand_pipeline::HandGtPipeline —— 业务唯一对外门面。
//
// 包 led_box_pose_sdk + 单槽 drop-oldest + 深拷 + worker；公共头不暴露
// cv::Mat / Eigen / OpenSSL / rknn_* / led_box_pose 命名空间。
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace hand_pipeline {

// 借引图像视图：像素 buffer 由调用方持有，仅在 feed() 调用期内被读取，
// feed 返回时 facade 已完成深拷。
struct ImageView {
    const uint8_t* data         = nullptr;
    int            width        = 0;
    int            height       = 0;
    int            channels     = 0;       // 3 = BGR, 1 = gray
    int            stride_bytes = 0;       // 0 表示连续 (= width*channels)
    int            cam_id       = -1;      // 0..5
};

// 单相机标定（与 services::algo::CameraCalibration 同形）。Double Sphere 模型：
// K = [fx,0,cx, 0,fy,cy, 0,0,1]；dist = [xi, alpha]；T_b_c = body→camera (tx,ty,tz, qx,qy,qz,qw)。
struct CameraCalib {
    int    cam_id   = -1;
    int    width    = 0;
    int    height   = 0;
    double K[9]     = {};
    double dist[8]  = {};
    int    n_dist   = 0;
    double T_b_c[7] = {};
};

struct HandGtConfig {
    std::vector<CameraCalib> calib;
    std::string assets_dir;
    std::string output_path;
    int  cam_left  = 1, cam_right = 4;     // IR 立体对
    int  cam_rgb_a = 2, cam_rgb_b = 3;     // RGB 对
    bool encrypt   = false;                // false = 明文 JSON；true = AES-256-GCM
};

class HandGtPipeline {
public:
    explicit HandGtPipeline(HandGtConfig cfg);
    ~HandGtPipeline();

    HandGtPipeline(const HandGtPipeline&)            = delete;
    HandGtPipeline& operator=(const HandGtPipeline&) = delete;

    bool ok() const;                                                      // 构造期是否成功
    bool feed(const ImageView* views, int n, int64_t pts_us);             // 单槽 drop-oldest
    bool finalize();                                                      // 阻塞排空 + SDK finalize；幂等
    std::string last_error() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace hand_pipeline
