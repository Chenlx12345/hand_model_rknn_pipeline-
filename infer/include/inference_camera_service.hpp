#ifndef INFERENCE_CAMERA_SERVICE_HPP
#define INFERENCE_CAMERA_SERVICE_HPP

// =============================================================================
// InferenceCameraService （接口快照）
//
// 来源：本头文件由上游业务工程 das_ego_app 的
// 拷贝而来，作为 hand_model_rknn_pipeline 调用方对接此服务时的接口参考。
//
// 注意：
//   - 这是 **接口快照**，不参与本仓库（hand_model_rknn_pipeline）的编译；
//   - 实际链接与实现在 das_ego_app 中（编译产物 libinference_camera_service.a）；
//   - 任何字段 / 函数签名变化以上游 das_ego_app 版本为准，本快照仅作阅读用途。
//
// 用途：把 6 路摄像头同一瞬间的图像，作为一个 batch 交给上层用于模型推理。
//
// 使用方式（典型流程）：
//
//   #include "inference_camera_service.hpp"
//
//   auto* svc = services::sensor::InferenceCameraService::getInstance();
//   svc->setOnInferenceCameraBatchCallback(
//       [](const services::sensor::InferenceCameraBatch& batch) {
//           for (const auto& v : batch.cams) {
//               // v.bgr 是 1600x1300 CV_8UC3 BGR，可直接喂给手部模型
//               // 若要异步使用，请先 v.bgr.clone()
//           }
//       });
//
// 关键约定：
//   - 回调里 cams[] 一定有 6 路、cam_id 覆盖 0~5，且 6 路时间严格对齐；
//   - 回调频率由 INFERENCE_CAMERA_BATCH_INTERVAL_MS 决定，默认 1Hz；
//   - InferenceCameraView::bgr 的内存只在回调期内有效，跨线程异步使用必须 clone()；
//   - 回调在内部工作线程触发——回调函数本身请不要长时间阻塞。
// =============================================================================

#include <array>
#include <chrono>
#include <cstdint>
#include <functional>

#include <opencv2/core.hpp>

// 手部模型 RKNN 日志文件路径（供本仓库的推理实现自行打开写入）
#ifndef HAND_INFER_LOG_PATH
#define HAND_INFER_LOG_PATH  "/mnt/sdcard/hand_model_rknn.log"
#endif

// Batch 回调发出间隔（毫秒）。默认 1000ms = 1Hz；如需更高速率，请联系
// das_ego_app 侧在编译时覆盖：
//   -DINFERENCE_CAMERA_BATCH_INTERVAL_MS=200   // 5Hz
#ifndef INFERENCE_CAMERA_BATCH_INTERVAL_MS
#define INFERENCE_CAMERA_BATCH_INTERVAL_MS  1000
#endif

namespace services {
namespace sensor {

// batch 中固定包含的摄像头路数
constexpr uint8_t kInferenceCameraCount = 6;

// 单路图像视图（一个 batch 包含 kInferenceCameraCount 个）
struct InferenceCameraView {
    uint8_t  cam_id;      // 摄像头编号 0 ~ kInferenceCameraCount-1
    uint32_t seq;         // 该路自启动起的帧序号
    uint64_t mono_ns;     // 抓帧时间（CLOCK_MONOTONIC，纳秒）
    cv::Mat  bgr;         // 1600x1300 CV_8UC3 BGR，可直接喂给模型
                          // 注意：内存仅回调期内有效，异步使用须 clone()
};

// 一组同步的 6 路图像
struct InferenceCameraBatch {
    uint64_t pts_us;          // 该 batch 的硬件触发时间戳（6 路相同），单位微秒
    uint64_t batch_mono_ns;   // 本 batch 集齐 6 路时的本机时间（CLOCK_MONOTONIC，纳秒）
    std::array<InferenceCameraView, kInferenceCameraCount> cams;
};

// Batch 回调签名
using OnInferenceCameraBatchCallback = std::function<void(const InferenceCameraBatch&)>;

// 6 路摄像头同步图像分发服务（单例，进程内全局）
class InferenceCameraService {
public:
    // 获取单例。首次调用时自动接管底层数据流，无需额外初始化。
    static InferenceCameraService* getInstance();

    // 注册 batch 回调。设置后即开始接收 batch，回调频率见
    // INFERENCE_CAMERA_BATCH_INTERVAL_MS。重复设置以最新一次为准。
    bool setOnInferenceCameraBatchCallback(OnInferenceCameraBatchCallback callback);

    // 清除回调，停止接收 batch。
    void clearOnInferenceCameraBatchCallback();

    // 查询某一路摄像头是否仍在出帧。max_staleness 内仍有帧到达视为可用。
    bool isCameraAvailable(uint32_t camera_id,
                           std::chrono::seconds max_staleness = std::chrono::seconds(5)) const;

private:
    InferenceCameraService();
    ~InferenceCameraService();
    InferenceCameraService(const InferenceCameraService&) = delete;
    InferenceCameraService& operator=(const InferenceCameraService&) = delete;
    // 其余私有成员省略；实现位于 das_ego_app 端的 .cpp 中
};

}  // namespace sensor
}  // namespace services

#endif  // INFERENCE_CAMERA_SERVICE_HPP
