// demo_consumer — 演示下游 SDK 怎么 pull 我们的结果。
//
// 主程序在 start() 之后,自己 100 ms 轮询一次 get_latest;有新一帧就打印。
// 跟 demo_init 共用 mp4 仿真 services::algo,所以宿主上也能跑通整条链。
//
// 用法:
//   ./demo_consumer [output_path]    # 默认 /tmp/vv_result.jsonl;不想落盘传 ""
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

static void print_tick(const vitgloves::vis::VvResult& r) {
    std::printf("[tick] ts=%llu  frame=%d  boxes=%zu  finger=%zu  step2=%zu  step3=%zu\n",
                (unsigned long long)r.timestamp_ns, r.frame_id,
                r.boxes.size(), r.fingers.size(),
                r.step2_points.size(), r.step3_points.size());
    for (const auto& b : r.boxes) {
        std::printf("    box   side=%d imu=0x%02X  t=[%6.3f %6.3f %6.3f]  reproj=%.2fpx\n",
                    b.side, b.imu_id, b.T[3], b.T[7], b.T[11], b.reproj_px);
    }
    for (const auto& f : r.fingers) {
        std::printf("    fingr side=%d imu=0x%02X xyz=[%6.3f %6.3f %6.3f]  reproj_l/r=%.2f/%.2f\n",
                    f.side, f.imu_id,
                    f.xyz_body[0], f.xyz_body[1], f.xyz_body[2],
                    f.reproj_l, f.reproj_r);
    }
}

int main(int argc, char** argv) {
    vitgloves::vis::InitConfig cfg;
    cfg.output_path = argc > 1 ? argv[1] : "/tmp/vv_result.jsonl";

    std::printf("=== demo_consumer:pull 模式 ===\n");
    std::printf("output_path = %s\n", cfg.output_path.c_str());
    std::printf("press Ctrl-C to stop\n\n");

    std::signal(SIGINT,  on_sigint);
    std::signal(SIGTERM, on_sigint);

    vitgloves::vis::VitglovesVisSdk sdk(cfg);
    if (!sdk.start()) {
        std::fprintf(stderr, "start failed: %s\n", sdk.last_error().c_str());
        return 1;
    }

    int n_ticks = 0, n_polls = 0;
    vitgloves::vis::VvResult r;
    while (!g_stop.load()) {
        if (sdk.get_latest(&r)) { ++n_ticks; print_tick(r); }
        ++n_polls;
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    sdk.stop();
    std::printf("\n[done] polls=%d ticks_got=%d\n", n_polls, n_ticks);
    return 0;
}
