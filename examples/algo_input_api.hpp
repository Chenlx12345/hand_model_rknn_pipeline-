#ifndef ALGO_INPUT_API_HPP
#define ALGO_INPUT_API_HPP

// =============================================================================
// AlgoInput API（单文件接口快照）
//
// 这个头给 hand_model_rknn_pipeline 的外部集成方使用，用于从 das_ego_app
// 获取 6 路同步 BGR 图像 batch 和相机标定数据。
//
// 说明：
//   - 本文件只有接口声明，实际实现由 das_ego_app 的 algo_input_service 提供；
//   - 调用 initializeAlgoInput() 后，再注册 batch 回调；
//   - 当前只支持一个消费者，重复 setAlgoBatchCallback() 会覆盖旧回调；
//   - 回调在内部采集线程触发，请不要在回调中长时间阻塞；
//   - AlgoFrame::bgr 为 CV_8UC3 BGR，数据可异步保存；
//   - 标定缺失或解析失败时，对应 CameraCalibration.valid=false。
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
    uint8_t                 cam_id = 0xFF;   // 摄像头编号，0~5；无效值为 0xFF
    uint32_t                width  = 0;      // 标定对应图像宽度
    uint32_t                height = 0;      // 标定对应图像高度
    std::string             distortion_model; // 畸变模型名
    std::string             frame_id;        // 标定坐标系名
    std::vector<double>     D;               // 畸变参数
    std::array<double, 9>   K{};             // 3x3 内参矩阵，行主序
    std::array<double, 9>   R{};             // 3x3 校正旋转矩阵，行主序
    std::array<double, 12>  P{};             // 3x4 投影矩阵，行主序
    std::array<double, 7>   T_b_c{};         // body->camera: tx,ty,tz,qx,qy,qz,qw
    bool                    valid = false;   // true 表示该路标定成功加载
};

struct AlgoInputConfig {
    std::string calib_dir    = "/userdata/das_ego/calibration";
    std::string filename_fmt = "camera_calibration_cam%u.bin";
    bool        require_all  = false;        // true: 任一路标定失败则初始化失败
};

struct AlgoFrame {
    uint8_t  cam_id = 0;     // 摄像头编号，0~5
    uint32_t seq = 0;        // 该路图像序号
    uint64_t mono_ns = 0;    // 采集时刻，CLOCK_MONOTONIC 纳秒
    cv::Mat  bgr;           // CV_8UC3 BGR 图像，可异步保存或送入模型
};

struct AlgoBatch {
    uint64_t pts_us = 0;        // 同步触发时间戳，微秒
    uint64_t batch_mono_ns = 0; // batch 集齐时刻，CLOCK_MONOTONIC 纳秒
    std::array<AlgoFrame, kAlgoCameraCount> cams; // 固定 6 路图像
};

using OnAlgoBatchCallback = std::function<void(const AlgoBatch&)>;

// 初始化算法输入服务；成功后可注册 batch 回调。
bool initializeAlgoInput();
bool initializeAlgoInput(const AlgoInputConfig& config);

// 关闭算法输入服务，并清除已注册回调。
void shutdownAlgoInput();
bool isAlgoInputEnabled();

// 注册/清除 6 路图像 batch 回调；重复注册会覆盖旧回调。
bool setAlgoBatchCallback(OnAlgoBatchCallback callback);
void clearAlgoBatchCallback();

// 获取指定相机标定；越界或缺失时返回 valid=false 的对象。
const CameraCalibration& getAlgoCameraCalibration(uint8_t cam_id);
bool hasAlgoCameraCalibration(uint8_t cam_id);

}  // namespace algo
}  // namespace services

#endif  // ALGO_INPUT_API_HPP
