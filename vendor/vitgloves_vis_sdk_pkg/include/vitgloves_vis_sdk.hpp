// vitgloves_vis_sdk — 戴手套手部 GT 提取 SDK(头部 IMU 辅助 + 6 路相机)。
//
// 接入方式:
//   vitgloves::vis::InitConfig cfg;
//   cfg.assets_dir  = "/opt/vitgloves_vis_sdk/assets";   // model/ + config/ + led/
//   cfg.output_path = "/mnt/sdcard/result.json";
//   vitgloves::vis::VitglovesVisSdk sdk(cfg);
//   if (!sdk.start()) { /* sdk.last_error() */ return 1; }
//   ... 应用运行,services::algo 推 6 路图像与 200Hz IMU 进 SDK ...
//   sdk.stop();
//
// SDK 自己向 services::algo 注册两路回调;调用方不需要碰 algo_input_api。
// 公共头不依赖任何第三方类型(只 <cstdint>/<string>),PImpl + 隐藏符号。
#pragma once

#include "vitgloves_vis_result.hpp"   // 下游 SDK pull 用的 VvResult + POD 类型

#include <cstdint>
#include <string>

namespace vitgloves {
namespace vis {

/* ── Version ─────────────────────────────────────────────────── */
#define VITGLOVES_VIS_SDK_VERSION_MAJOR 0
#define VITGLOVES_VIS_SDK_VERSION_MINOR 1
#define VITGLOVES_VIS_SDK_VERSION_PATCH 0

/* ── Visibility ──────────────────────────────────────────────── */
#if defined(VITGLOVES_VIS_SDK_BUILD)
  #define VITGLOVES_VIS_API __attribute__((visibility("default")))
#else
  #define VITGLOVES_VIS_API
#endif

/* ── Error codes ─────────────────────────────────────────────── */
constexpr int VV_OK              =  0;
constexpr int VV_ERR_NOT_READY   = -1;   ///< start() 没调,或失败
constexpr int VV_ERR_INVALID_ARG = -2;
constexpr int VV_ERR_IO          = -3;   ///< 资源/模型文件读不到
constexpr int VV_ERR_INTERNAL    = -4;

/* ── Init config ─────────────────────────────────────────────── */
/// 资源目录(模型/配置/LED 模型)固化在 SDK 内部,不再作参数;
/// 默认是编译期 VV_ASSETS_DIR_DEFAULT,运行期可用 env VV_ASSETS_DIR 覆盖。
struct InitConfig {
    std::string output_path;  ///< NDJSON 落盘路径;**留空 = 不写文件**,下游用 get_latest 拉
};

/* ── SDK class(PImpl,不可拷贝) ─────────────────────────────── */
class VITGLOVES_VIS_API VitglovesVisSdk {
public:
    explicit VitglovesVisSdk(const InitConfig& cfg);
    ~VitglovesVisSdk();

    VitglovesVisSdk(const VitglovesVisSdk&)            = delete;
    VitglovesVisSdk& operator=(const VitglovesVisSdk&) = delete;

    /// 初始化全套:读资源 + 启动 algo 服务 + 查 6 路标定 + 加载 RKNN 模型 + 注册两路回调。
    /// 成功后服务即开始推送数据进 SDK。
    bool start();

    /// 注销回调、关闭 algo 服务、释放模型。可重复调用(空操作)。
    void stop();

    /// start() 是否成功(且尚未 stop)。
    bool        ok() const;
    std::string last_error() const;

    /// 运行期统计(回调里更新)。
    int frames_received() const;
    int imu_received() const;
    int hand_imu_received() const;

    /// 拉最新一个 5Hz tick 的结果(给下游 SDK 用;线程安全,非阻塞)。
    /// 语义:**drop-stale**(只给最新一份;来不及取的中间帧覆盖丢弃)。
    /// @param out 调用方提供的输出对象;成功时被 SDK 写入。
    /// @return true  — 自上次 get 以来,SDK 又产出了新的 tick,*out 填好;
    ///         false — 还没有新结果(*out 不动);可以稍后再问。
    bool get_latest(VvResult* out);

private:
    struct Impl;
    Impl* pimpl_;
    friend void step234_worker_func(Impl*);
};

} // namespace vis
} // namespace vitgloves
