// demo_integrator — 给集成方的最小启动示例。
//
// 与 demo_init 的区别:demo_init 链 mp4_algo_service 当 `services::algo` 实现(只主机仿真用);
// demo_integrator **不依赖 mp4**,直接调 SDK 的 start/stop,services::algo 由集成方自己的
// libalgo_input(das_ego_app 自带)提供。SIGINT 后优雅 stop。
//
// 用法:
//   ./demo_integrator [output_path]    # 默认 /tmp/vv_result.jsonl
//
// 注意:运行时需要找到 librknnrt.so(板上一般 /usr/lib/),不需要额外参数。
#include "vitgloves_vis_sdk.hpp"

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <thread>

namespace {
std::atomic<bool> g_stop{false};
void on_sigint(int) { g_stop.store(true); }
} // namespace

int main(int argc, char** argv) {
    vitgloves::vis::InitConfig cfg;
    cfg.output_path = argc > 1 ? argv[1] : "/tmp/vv_result.jsonl";

    std::printf("=== vitgloves_vis_sdk integrator demo ===\n");
    std::printf("output_path = %s\n", cfg.output_path.c_str());
    std::printf("press Ctrl-C to stop\n\n");

    std::signal(SIGINT,  on_sigint);
    std::signal(SIGTERM, on_sigint);

    vitgloves::vis::VitglovesVisSdk sdk(cfg);
    if (!sdk.start()) {
        std::fprintf(stderr, "start failed: %s\n", sdk.last_error().c_str());
        return 1;
    }

    // 主线程睡循环;真业务里这里可以做别的(SDK 自己开了回调线程)。
    while (!g_stop.load()) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
        std::printf("[demo] frames=%d  imu=%d\n",
                    sdk.frames_received(), sdk.imu_received());
    }

    sdk.stop();
    return 0;
}
