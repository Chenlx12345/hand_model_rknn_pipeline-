#ifndef ALGO_INPUT_API_HPP
#define ALGO_INPUT_API_HPP

// =============================================================================
// AlgoInput API — 算法数据输入接口
//
// 提供三类数据：6 路同步相机图像、IMU 样本、相机标定参数。
//
// 典型用法：
//
//   // 1. 初始化（在模块 initialize() 中调用）
//   services::algo::initializeAlgoInput();
//
//   // 2. 读取标定（初始化后即可使用）
//   const auto& calib = services::algo::getAlgoCameraCalibration(cam_id);
//   if (calib.valid) { /* 使用 calib.K / calib.D / calib.T_b_c */ }
//
//   // 3. 注册相机 batch 回调（6 路图像凑齐后触发，默认 1Hz）
//   services::algo::setAlgoBatchCallback([](const services::algo::AlgoBatch& batch) {
//       for (const auto& frame : batch.cams) {
//           // frame.bgr     CV_8UC3 图像
//           // frame.mono_ns 采集时刻，与 IMU mono_ns 同钟可直接对齐
//       }
//   });
//
//   // 4. 注册 IMU 回调（200Hz）
//   services::algo::setAlgoImuCallback([](const services::algo::AlgoImuSample& s) {
//       // s.accel_*/gyro_*/roll/pitch/yaw
//       // s.mono_ns 与 AlgoBatch::batch_mono_ns 同为 CLOCK_MONOTONIC，可直接对齐
//   });
//
//   // 5. 关闭（在模块 shutdown() 中调用）
//   services::algo::clearAlgoBatchCallback();
//   services::algo::clearAlgoImuCallback();
//   services::algo::shutdownAlgoInput();
//
// 注意：
//   - 回调在采集线程触发，禁止在回调中长时间阻塞；
//   - 每类数据只支持一个消费者，重复注册覆盖旧回调；
//   - 标定加载失败时 CameraCalibration::valid = false，不影响图像和 IMU 流。
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
    uint32_t               width  = 0;    // 标定图像宽度（像素）
    uint32_t               height = 0;    // 标定图像高度（像素）
    std::string            distortion_model;
    std::string            frame_id;
    std::vector<double>    D;             // 畸变系数
    std::array<double, 9>  K{};          // 3×3 内参矩阵，行主序
    std::array<double, 9>  R{};          // 3×3 校正旋转矩阵，行主序
    std::array<double, 12> P{};          // 3×4 投影矩阵，行主序
    std::array<double, 7>  T_b_c{};      // body→camera 变换：tx,ty,tz,qx,qy,qz,qw
    bool                   valid = false; // true 表示标定数据已成功加载
};

// 标定文件路径配置，通常使用默认值即可
struct AlgoInputConfig {
    std::string calib_dir    = "/userdata/das_ego/calibration";
    std::string filename_fmt = "camera_calibration_cam%u.bin";
    bool        require_all  = false; // true：任一路标定失败则初始化失败
};

struct AlgoFrame {
    uint8_t  cam_id  = 0;   // 相机编号 0~5
    uint32_t seq     = 0;   // 该路帧序号
    uint64_t mono_ns = 0;   // 采集时刻，CLOCK_MONOTONIC（纳秒）
    cv::Mat  bgr;           // CV_8UC3 BGR 图像
};

struct AlgoBatch {
    uint64_t pts_us        = 0;  // 硬件同步触发时间戳（微秒）
    uint64_t batch_mono_ns = 0;  // 6 路凑齐时刻，CLOCK_MONOTONIC（纳秒）
    std::array<AlgoFrame, kAlgoCameraCount> cams;
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
    int64_t  imu_ts_us;  // IMU 内部时间戳（微秒）
    int64_t  mono_ns;    // CLOCK_MONOTONIC（纳秒），与 AlgoBatch::batch_mono_ns 同钟
    uint32_t sequence;
};

using OnAlgoBatchCallback = std::function<void(const AlgoBatch&)>;
using OnAlgoImuCallback   = std::function<void(const AlgoImuSample&)>;

// 生命周期
bool initializeAlgoInput();
bool initializeAlgoInput(const AlgoInputConfig& config);
void shutdownAlgoInput();
bool isAlgoInputEnabled();

// 相机 batch 回调（6 路同步图像）
bool setAlgoBatchCallback(OnAlgoBatchCallback callback);
void clearAlgoBatchCallback();

// 相机标定（初始化后只读，线程安全）
const CameraCalibration& getAlgoCameraCalibration(uint8_t cam_id);
bool hasAlgoCameraCalibration(uint8_t cam_id);

// IMU 回调（200Hz）
bool setAlgoImuCallback(OnAlgoImuCallback callback);
void clearAlgoImuCallback();
bool isAlgoImuEnabled();

}  // namespace algo
}  // namespace services

#endif  // ALGO_INPUT_API_HPP


