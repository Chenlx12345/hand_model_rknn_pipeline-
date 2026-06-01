#ifndef ALGO_INPUT_API_HPP
#define ALGO_INPUT_API_HPP

// =============================================================================
// AlgoInput API — 算法数据输入接口
//
// 提供：6 路相机图像 / 头部 IMU / 手部 IMU / 相机标定。
//
// 约束：回调在采集线程触发，禁止长时间阻塞；每类数据仅一个消费者。
// =============================================================================

#include <array>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

namespace services {
namespace algo {

// =============================================================================
// CameraCalibration
// =============================================================================

struct CameraCalibration {
    uint8_t                 cam_id = 0xFF;
    uint32_t                width  = 0;
    uint32_t                height = 0;
    std::string             distortion_model;
    std::string             frame_id;
    std::vector<double>     D;
    std::array<double, 9>   K{};
    std::array<double, 9>   R{};
    std::array<double, 12>  P{};
    std::array<double, 7>   T_b_c{};
    bool                    valid = false;
};

// =============================================================================
// 相机 Batch
// =============================================================================

constexpr uint8_t kAlgoCameraCount = 6;

struct AlgoInputConfig {
    std::string calib_dir    = "/userdata/das_ego/calibration";
    std::string filename_fmt = "camera_calibration_cam%u.bin";
    bool        require_all  = false;
};

struct AlgoFrame {
    uint8_t  cam_id   = 0;       // 摄像头编号 0~5
    uint32_t seq      = 0;       // 该路图像序号
    uint64_t timestamp_ns = 0;   // CLOCK_REALTIME (ns)
    cv::Mat  bgr;                // CV_8UC3 BGR
};

struct AlgoBatch {
    uint64_t batch_timestamp_ns = 0;  // CLOCK_REALTIME (ns)
    std::array<AlgoFrame, kAlgoCameraCount> cams;
};

using OnAlgoBatchCallback = std::function<void(const AlgoBatch&)>;

// =============================================================================
// 头部 IMU (200Hz, Forsense 614EB, 单节点)
// =============================================================================

struct AlgoHeadImuSample {
    float    accel_x;
    float    accel_y;
    float    accel_z;
    float    gyro_x;
    float    gyro_y;
    float    gyro_z;
    float    roll;
    float    pitch;
    float    yaw;
    float    temperature;
    int64_t  timestamp_ns;   // CLOCK_REALTIME (ns)
    uint32_t sequence;
};

using OnAlgoHeadImuCallback = std::function<void(const AlgoHeadImuSample&)>;

bool setAlgoHeadImuCallback(OnAlgoHeadImuCallback callback);
void clearAlgoHeadImuCallback();
bool isAlgoHeadImuEnabled();
void pushAlgoHeadImuSample(const AlgoHeadImuSample& sample);

// =============================================================================
// 手部 IMU (5Hz, PeerLink 手套, 左右手各 14 节点)
// =============================================================================

constexpr uint8_t kAlgoHandImuNodeCount = 14;

struct AlgoHandImuNode {
    uint8_t  node_id;       // 硬件地址: 左手 0x11..0x1D + 0x05, 右手 0x21..0x2D + 0x08
    uint16_t status;
    float    accel[3];      // g
    float    gyro[3];       // deg/s
    float    mag[3];        // uT
    float    quat[4];       // w,x,y,z (calibrated)
    float    quat_raw[4];   // w,x,y,z (raw)
};

struct AlgoHandImuFrame {
    uint8_t  handedness;    // 0=left, 1=right
    uint32_t seq;
    int64_t  timestamp_ns;  // CLOCK_REALTIME (ns)
    std::array<AlgoHandImuNode, kAlgoHandImuNodeCount> nodes;  // 顺序不固定，按 node_id 识别
    uint8_t  active_count;
};

using OnAlgoHandImuCallback = std::function<void(const AlgoHandImuFrame&)>;

bool setAlgoHandImuCallback(OnAlgoHandImuCallback callback);
void clearAlgoHandImuCallback();
bool isAlgoHandImuEnabled();
void pushAlgoHandImuFrame(const AlgoHandImuFrame& frame);

// =============================================================================
// 初始化 / 关闭 / 相机 Batch / 标定
// =============================================================================

bool initializeAlgoInput();
bool initializeAlgoInput(const AlgoInputConfig& config);
void shutdownAlgoInput();
bool isAlgoInputEnabled();

bool setAlgoBatchCallback(OnAlgoBatchCallback callback);
void clearAlgoBatchCallback();

const CameraCalibration& getAlgoCameraCalibration(uint8_t cam_id);
bool hasAlgoCameraCalibration(uint8_t cam_id);

}  // namespace algo
}  // namespace services

#endif  // ALGO_INPUT_API_HPP
