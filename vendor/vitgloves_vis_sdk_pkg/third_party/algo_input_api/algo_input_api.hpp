#ifndef ALGO_INPUT_API_HPP
#define ALGO_INPUT_API_HPP

// =============================================================================
// AlgoInput API — 算法数据输入接口
//
// 提供三类数据：6 路同步相机图像、IMU 样本、相机标定参数。
//
// 快速接入：
//
//   // 1. 初始化
//   services::algo::initializeAlgoInput();
//
//   // 2. 读取标定（初始化后即可用，只读线程安全）
//   const auto& calib = services::algo::getAlgoCameraCalibration(cam_id);
//   if (calib.valid) { /* 使用 calib.K / calib.D / calib.T_b_c */ }
//
//   // 3. 注册相机 batch 回调（30fps，6 路始终齐全）
//   services::algo::setAlgoBatchCallback([](const services::algo::AlgoBatch& batch) {
//       for (const auto& frame : batch.cams) {
//           // frame.bgr — CV_8UC3 BGR 图像
//       }
//       // batch.batch_timestamp_ns — 纳秒级时间戳，与 IMU 对齐
//   });
//
//   // 4. 注册 IMU 回调（200Hz）
//   services::algo::setAlgoImuCallback([](const services::algo::AlgoImuSample& s) {
//       // s.accel_x/y/z, s.gyro_x/y/z, s.roll, s.pitch, s.yaw
//       // s.timestamp_ns — 纳秒级时间戳，与相机 batch_timestamp_ns 对齐
//   });
//
//   // 5. 关闭
//   services::algo::clearAlgoBatchCallback();
//   services::algo::clearAlgoImuCallback();
//   services::algo::shutdownAlgoInput();
//
// 约束：
//   - 回调在采集线程触发，禁止在回调中长时间阻塞
//   - 每类数据只支持一个消费者，重复注册覆盖旧回调
//   - 标定加载失败时 CameraCalibration::valid = false，不影响图像和 IMU 流
// =============================================================================

#include <array>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

namespace services {
namespace algo {

constexpr uint8_t kAlgoCameraCount = 6;

struct CameraCalibration {
    uint8_t                cam_id = 0xFF;  // 相机编号 0~5；0xFF 表示无效
    uint32_t               width  = 0;     // 标定图像宽度（像素）
    uint32_t               height = 0;     // 标定图像高度（像素）
    std::string            distortion_model;
    std::string            frame_id;
    std::vector<double>    D;              // 畸变系数
    std::array<double, 9>  K{};            // 3x3 内参矩阵，行主序
    std::array<double, 9>  R{};            // 3x3 校正旋转矩阵，行主序
    std::array<double, 12> P{};            // 3x4 投影矩阵，行主序
    std::array<double, 7>  T_b_c{};        // body->camera 变换：tx,ty,tz,qx,qy,qz,qw
    bool                   valid = false;  // true 表示标定数据已成功加载
};

struct AlgoInputConfig {
    std::string calib_dir    = "/userdata/das_ego/calibration";
    std::string filename_fmt = "camera_calibration_cam%u.bin";
    bool        require_all  = false;  // true：任一路标定失败则初始化失败
};

struct AlgoFrame {
    uint8_t  cam_id       = 0;  // 相机编号 0~5
    uint32_t seq          = 0;  // 该路帧序号
    uint64_t timestamp_ns = 0;  // CLOCK_REALTIME (ns)
    cv::Mat  bgr;               // CV_8UC3 BGR 图像；丢帧时为空 (bgr.empty())
};

struct AlgoBatch {
    uint64_t batch_timestamp_ns = 0;  // CLOCK_REALTIME (ns)
    std::array<AlgoFrame, kAlgoCameraCount> cams;  // 6 路始终齐全
};

struct AlgoImuSample {
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
    int64_t  timestamp_ns;  // CLOCK_REALTIME (ns)
    uint32_t sequence;
};

using OnAlgoBatchCallback = std::function<void(const AlgoBatch&)>;
using OnAlgoImuCallback   = std::function<void(const AlgoImuSample&)>;

bool initializeAlgoInput();
bool initializeAlgoInput(const AlgoInputConfig& config);
void shutdownAlgoInput();
bool isAlgoInputEnabled();

bool setAlgoBatchCallback(OnAlgoBatchCallback callback);
void clearAlgoBatchCallback();

const CameraCalibration& getAlgoCameraCalibration(uint8_t cam_id);
bool hasAlgoCameraCalibration(uint8_t cam_id);

bool setAlgoImuCallback(OnAlgoImuCallback callback);
void clearAlgoImuCallback();
bool isAlgoImuEnabled();

}  // namespace algo
}  // namespace services

#endif  // ALGO_INPUT_API_HPP
