#include "hand_gt_pipeline.hpp"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstring>
#include <exception>
#include <mutex>
#include <optional>
#include <thread>
#include <utility>

#include "led_box_pose_sdk.hpp"

// vendor SDK .a 由 GCC11.4 编出，本工程 toolchain GCC10.4 的 libstdc++.a 不含
// __throw_bad_array_new_length。补一份等价定义让 ld 解析；GCC>=11 时跳过避免冲突。
// 必须放在本 TU 内：HandGtPipeline 构造被 service 引用 → ld 必拉本 .o → 该符号
// 一并入全局表，后续扫 vendor SDK 时即可解析。拆到独立 .cpp 会因单遍 ld
// pull-in 失败导致 undefined reference。
#if defined(__GNUC__) && !defined(__clang__) && __GNUC__ < 11
#include <new>
namespace std {
[[noreturn]] void __throw_bad_array_new_length() {
    throw std::bad_array_new_length();
}
}  // namespace std
#endif

namespace hand_pipeline {

namespace {

// 单帧深拷：按行 memcpy 到紧凑布局（stride = width*channels），切断 AlgoInput
// 回调期 buffer 生命周期依赖。
struct OwnedFrame {
    std::vector<uint8_t> pixels;
    int width = 0, height = 0, channels = 0, cam_id = -1;
};

struct OwnedBatch {
    int64_t                 pts_us = 0;
    std::vector<OwnedFrame> frames;
};

OwnedFrame deepCopy(const ImageView& v) {
    OwnedFrame f{ {}, v.width, v.height, v.channels, v.cam_id };
    const int row_dst = v.width * v.channels;
    const int row_src = (v.stride_bytes > 0) ? v.stride_bytes : row_dst;
    f.pixels.resize(static_cast<size_t>(row_dst) * v.height);
    for (int y = 0; y < v.height; ++y) {
        std::memcpy(f.pixels.data() + static_cast<size_t>(y) * row_dst,
                    v.data + static_cast<size_t>(y) * row_src,
                    static_cast<size_t>(row_dst));
    }
    return f;
}

}  // namespace

struct HandGtPipeline::Impl {
    std::unique_ptr<::led_box_pose::LedBoxPoseSdk>  sdk;
    std::thread                                     worker;
    mutable std::mutex                              mu;          // 保护 latest + last_error
    std::condition_variable                         cv;
    std::optional<OwnedBatch>                       latest;      // 单槽 drop-oldest
    std::string                                     last_error;
    std::atomic<bool>                               stop_flag{false};
    std::atomic<bool>                               finalized{false};

    void setError(const std::string& s) {
        std::lock_guard<std::mutex> lk(mu);
        last_error = s;
    }

    void workerLoop() {
        while (true) {
            OwnedBatch batch;
            {
                std::unique_lock<std::mutex> lk(mu);
                cv.wait(lk, [this]() {
                    return stop_flag.load(std::memory_order_acquire) || latest.has_value();
                });
                if (stop_flag.load(std::memory_order_acquire) && !latest.has_value()) return;
                batch = std::move(*latest);
                latest.reset();
            }

            ::led_box_pose::ImageView views[6];
            int n = 0;
            for (auto& f : batch.frames) {
                if (n >= 6 || f.pixels.empty()) continue;
                views[n++] = ::led_box_pose::ImageView{
                    f.pixels.data(), f.width, f.height, f.channels,
                    f.width * f.channels, f.cam_id,
                };
            }

            int ret = ::led_box_pose::LBP_OK;
            try {
                ret = sdk->process_batch(views, n, batch.pts_us);
            } catch (const std::exception& e) {
                setError(std::string("process_batch threw: ") + e.what());
                continue;
            } catch (...) {
                setError("process_batch threw (unknown)");
                continue;
            }
            if (ret != ::led_box_pose::LBP_OK) {
                setError("process_batch ret=" + std::to_string(ret) +
                         " sdk='" + sdk->last_error() + "'");
            }
        }
    }
};

HandGtPipeline::HandGtPipeline(HandGtConfig cfg) : impl_(std::make_unique<Impl>()) {
    ::led_box_pose::InitConfig sc;
    sc.assets_dir  = std::move(cfg.assets_dir);
    sc.output_path = std::move(cfg.output_path);
    sc.cam_left    = cfg.cam_left;
    sc.cam_right   = cfg.cam_right;
    sc.cam_rgb_a   = cfg.cam_rgb_a;
    sc.cam_rgb_b   = cfg.cam_rgb_b;
    sc.encrypt     = cfg.encrypt;
    sc.calib.reserve(cfg.calib.size());
    for (const auto& c : cfg.calib) {
        ::led_box_pose::CameraCalib out;
        out.cam_id = c.cam_id;
        out.width  = c.width;
        out.height = c.height;
        for (int i = 0; i < 9; ++i) out.K[i] = c.K[i];
        out.n_dist = std::min(8, c.n_dist);
        for (int i = 0; i < out.n_dist; ++i) out.dist[i] = c.dist[i];
        for (int i = 0; i < 7; ++i) out.T_b_c[i] = c.T_b_c[i];
        sc.calib.push_back(out);
    }

    try {
        impl_->sdk = std::make_unique<::led_box_pose::LedBoxPoseSdk>(sc);
    } catch (const std::exception& e) {
        impl_->setError(std::string("SDK ctor threw: ") + e.what());
        impl_->sdk.reset();
        return;
    } catch (...) {
        impl_->setError("SDK ctor threw (unknown)");
        impl_->sdk.reset();
        return;
    }
    if (!impl_->sdk->ok()) {
        impl_->setError("SDK init failed: " + impl_->sdk->last_error());
        impl_->sdk.reset();
        return;
    }

    try {
        impl_->worker = std::thread([this]() { impl_->workerLoop(); });
    } catch (const std::exception& e) {
        impl_->setError(std::string("worker spawn failed: ") + e.what());
        impl_->sdk.reset();
    }
}

HandGtPipeline::~HandGtPipeline() {
    if (impl_->worker.joinable()) {
        impl_->stop_flag.store(true, std::memory_order_release);
        impl_->cv.notify_all();
        impl_->worker.join();
    }
    impl_->sdk.reset();
}

bool HandGtPipeline::ok() const {
    return impl_->sdk != nullptr && !impl_->finalized.load(std::memory_order_acquire);
}

bool HandGtPipeline::feed(const ImageView* views, int n, int64_t pts_us) {
    if (!impl_->sdk || impl_->finalized.load(std::memory_order_acquire)) return false;
    if (views == nullptr || n <= 0) return false;

    OwnedBatch ob;
    ob.pts_us = pts_us;
    ob.frames.reserve(static_cast<size_t>(std::min(n, 6)));
    for (int i = 0; i < n && i < 6; ++i) {
        const auto& v = views[i];
        if (v.data == nullptr || v.width <= 0 || v.height <= 0 || v.channels <= 0) continue;
        ob.frames.push_back(deepCopy(v));
    }
    if (ob.frames.empty()) return false;

    {
        std::lock_guard<std::mutex> lk(impl_->mu);
        if (impl_->stop_flag.load(std::memory_order_acquire)) return false;
        impl_->latest = std::move(ob);    // drop-oldest
    }
    impl_->cv.notify_one();
    return true;
}

bool HandGtPipeline::finalize() {
    if (!impl_->sdk) return false;
    if (impl_->finalized.exchange(true, std::memory_order_acq_rel)) return false;

    if (impl_->worker.joinable()) {
        impl_->stop_flag.store(true, std::memory_order_release);
        impl_->cv.notify_all();
        impl_->worker.join();
    }

    int ret = ::led_box_pose::LBP_OK;
    try {
        ret = impl_->sdk->finalize();
    } catch (const std::exception& e) {
        impl_->setError(std::string("finalize threw: ") + e.what());
        return false;
    } catch (...) {
        impl_->setError("finalize threw (unknown)");
        return false;
    }
    if (ret != ::led_box_pose::LBP_OK) {
        impl_->setError("finalize ret=" + std::to_string(ret) +
                        " sdk='" + impl_->sdk->last_error() + "'");
        return false;
    }
    return true;
}

std::string HandGtPipeline::last_error() const {
    std::lock_guard<std::mutex> lk(impl_->mu);
    return impl_->last_error;
}

}  // namespace hand_pipeline
