# 下游 SDK 怎么消费 `vitgloves_vis_sdk` 的结果

`vitgloves_vis_sdk` 在 5 Hz 节拍上算视觉结果。下游 SDK 通过 **pull 模式**(`get_latest`)
按自己的循环节奏拉走数据。

> 上游(`services::algo`)是 **push**:cam/IMU 来一帧调 SDK 一次。
> 下游(本文)是 **pull**:下游想要就 `get_latest()` 一次。
> 两条都不阻塞 SDK 内部 5 Hz 处理。

---

## 1. 一句话使用

```cpp
#include "vitgloves_vis_sdk.hpp"

// integrator's main app 已经持有这个对象、已经 sdk.start() 了
extern vitgloves::vis::VitglovesVisSdk& shared_sdk;

void downstream_loop() {
    vitgloves::vis::VvResult r;
    while (running) {
        if (shared_sdk.get_latest(&r)) {
            // r 是新的:用就完了
            consume(r);
        }
        // 没拿到也没事,SDK 没出新 tick 而已;睡一会儿再问
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
}
```

---

## 2. `get_latest()` 行为契约

```cpp
bool sdk.get_latest(VvResult* out);
```

| 情况 | 返回值 | `*out` |
|---|---|---|
| **自上次调用以来**,SDK 又跑过 ≥ 1 个 5 Hz tick | `true` | 被覆写为最新 tick 的全部内容 |
| 没有新 tick 产生 | `false` | **不动**(你上轮自己留下的值还在) |
| `out == nullptr` | `false` | — |

**drop-stale 语义**:如果下游 200 ms 内没问,SDK 中间又跑了几个 tick → 中间的全部覆盖丢弃,
`get_latest` 只给最新一份。这适合"我要当前真实状态"的下游;若需要"一帧不漏",
得额外开个 FIFO 队列(目前不提供;告诉我们就加)。

**线程**:任意线程调都可以;SDK 内部 mutex 保证拷贝原子。SDK 自己的 5 Hz 算线程跟你
完全独立,你不调它也照样跑;调它也不会拖慢它(锁窗口只是一次 vector 拷贝,微秒级)。

**幂等**:连续调两次 `get_latest`,第二次会返回 `false`(SDK 没产新东西)。

---

## 3. `VvResult` 结构体

定义在 `include/vitgloves_vis_result.hpp`。

```cpp
namespace vitgloves::vis {

struct VvResult {
    uint64_t timestamp_ns;   // batch_mono_ns,CLOCK_MONOTONIC ns;与 IMU 同钟
    int      frame_id;       // SDK 内部 30 Hz batch 计数器

    std::vector<Box6DofResult>     boxes;          // 0~2 只手
    std::vector<FingerPointResult> fingers;        // 0~26 个 IMU 点(每手 13)
    std::vector<StereoPoint3D>     step2_points;   // 全部三角化 IR 光斑 3D
    std::vector<HandKpt3DResult>   step3_points;   // 全部三角化 hand 关键点 3D
};

struct Box6DofResult {
    int     side;        // 0=left, 1=right
    uint8_t imu_id;      // 灯盒 IMU hex:left=0x11, right=0x21
    double  T[16];       // 4×4 行主序 X_body = R·X_box + t
    double  reproj_px;   // 7 LED 联合 cam1/cam4 平均反投残差(像素)
};

struct FingerPointResult {
    int     side;        // 0=left, 1=right
    uint8_t imu_id;      // 0x12..0x1D / 0x22..0x2D / 0x05 (左腕) / 0x08 (右腕)
    double  xyz_body[3]; // body 系 3D(米)
    double  reproj_l;    // 源 IR 点在 cam1 的 step2 重投影残差
    double  reproj_r;    //                  cam4
};

struct StereoPoint3D {
    double  xyz_body[3];
    double  uv1[2];      // 在 cam1 的像素中心
    double  uv4[2];      // 在 cam4
    double  reproj_l;
    double  reproj_r;
};

struct HandKpt3DResult {
    int     side;        // 0=left, 1=right
    int     kpt;         // 0..20(RTMPose 21 joint;0=wrist, 5=index_MCP, 等)
    int     cam_a, cam_b;// 用了哪两路 RGB
    double  xyz_body[3];
    double  uv_a[2], uv_b[2];
    double  reproj_a, reproj_b;
};

} // namespace
```

字段语义、坐标系约定、单位 = **跟 NDJSON 输出严格一一对应**;详见
[RESULT_FORMAT.md](RESULT_FORMAT.md)。

IMU hex 含义见 [IMU_ID.md](IMU_ID.md)。

`Vv*Result` 都是简单 POD(数组 + 标量);`VvResult` 持有 `std::vector` 自带的所有权,
拿到就完全归你 — 想拷贝就拷贝、想移动就 `std::move`、想跨线程就直接传走、想 cache 就
塞 std::deque 都行。

---

## 4. 谁创建 `VitglovesVisSdk`?

**只有 1 个**。集成方的主程序(das_ego_app)创建并 `start()`,然后把引用/指针交给下游 SDK:

```cpp
// 集成方主程序
int main() {
    vitgloves::vis::InitConfig cfg;
    cfg.output_path = "/data/recordings/vv.jsonl";   // 可选,空 = 不落盘
    static vitgloves::vis::VitglovesVisSdk vv(cfg);
    if (!vv.start()) { /* 错误 */ return 1; }

    DownstreamSdk down(&vv);     // ← 下游 SDK 拿到指针
    down.run();                   // ← 下游里面调 vv.get_latest(...)
    return 0;
}
```

下游 SDK 不需要自己 `start()` / `stop()`,也不能创建第二个 `VitglovesVisSdk`
(`services::algo` 只允许一对回调)。

---

## 5. 链接 / 编译

下游 SDK 用到的只有头文件 + 链接 .a:

```cmake
# 下游 SDK 的 CMakeLists
target_include_directories(my_downstream_sdk PRIVATE
    ${VITGLOVES_VIS_SDK_DIR}/include      # 拿到 vitgloves_vis_sdk.hpp + vitgloves_vis_result.hpp
)
target_link_libraries(my_downstream_sdk PRIVATE
    ${VITGLOVES_VIS_SDK_DIR}/lib/libvitgloves_vis_sdk.a
    # 还需要(集成方主程序通常已经链):
    #   集成方 OpenCV4(cv::* 符号)
    #   librknnrt(rknn_*)
    #   libalgo_input(services::algo::*)
    #   pthread
)
```

公共头**不引** OpenCV / Eigen / nlohmann,所以下游 SDK 不会被这些依赖污染。

---

## 6. 完整最小 demo(放进 tarball 的 `examples/demo_consumer.cpp`)

```cpp
#include "vitgloves_vis_sdk.hpp"

#include <chrono>
#include <cstdio>
#include <thread>

static void print_tick(const vitgloves::vis::VvResult& r) {
    std::printf("[tick] ts=%llu frame=%d  boxes=%zu finger=%zu  step2=%zu step3=%zu\n",
                (unsigned long long)r.timestamp_ns, r.frame_id,
                r.boxes.size(), r.fingers.size(),
                r.step2_points.size(), r.step3_points.size());
    for (const auto& b : r.boxes) {
        // T 的平移 = T[3], T[7], T[11]
        std::printf("    box side=%d imu=0x%02X t=[%.3f,%.3f,%.3f] reproj=%.2fpx\n",
                    b.side, b.imu_id, b.T[3], b.T[7], b.T[11], b.reproj_px);
    }
    for (const auto& f : r.fingers) {
        std::printf("    finger imu=0x%02X xyz=[%.3f,%.3f,%.3f] reproj_l/r=%.2f/%.2f\n",
                    f.imu_id,
                    f.xyz_body[0], f.xyz_body[1], f.xyz_body[2],
                    f.reproj_l, f.reproj_r);
    }
}

int main() {
    vitgloves::vis::InitConfig cfg;
    cfg.output_path = "/tmp/vv_result.jsonl";       // 同时也落 NDJSON
    vitgloves::vis::VitglovesVisSdk sdk(cfg);
    if (!sdk.start()) {
        std::fprintf(stderr, "start: %s\n", sdk.last_error().c_str());
        return 1;
    }

    vitgloves::vis::VvResult r;
    for (int i = 0; i < 60; ++i) {                  // 跑 ~6 秒,~30 个 tick
        if (sdk.get_latest(&r)) print_tick(r);
        std::this_thread::sleep_for(std::chrono::milliseconds(100));   // 10 Hz 轮询
    }
    sdk.stop();
    return 0;
}
```

输出大概长这样:
```
[tick] ts=1779790769720964000 frame=0   boxes=2 finger=4  step2=32 step3=42
    box side=0 imu=0x11 t=[0.174,0.166,-0.322] reproj=0.42px
    box side=1 imu=0x21 t=[-0.043,0.130,-0.376] reproj=1.45px
    finger imu=0x1C xyz=[0.151,0.188,-0.331] reproj_l/r=0.43/0.48
    ...
```

---

## 7. 常见问题

| 问题 | 答 |
|---|---|
| `get_latest` 一直返回 `false`? | SDK 没产新结果。可能 `start()` 没成功;或者 `services::algo` 没在推 batch;或者每个 batch 都被 step2/3/4 拒了。看 `sdk.frames_received()` 和 `last_error()`。 |
| 想知道总共 published 了多少 tick? | 现在没暴露这个统计;`get_latest` 的 true 次数自己加;真要 API 上加 `n_published()` 告诉我们补。 |
| `VvResult` 能跨线程传吗? | 能。`std::vector` 拷贝/移动都安全;但同一个 `VvResult` 别多个线程同时改。 |
| `VvResult` 可以一直 reuse 同一份吗? | 可以。`get_latest` 的实现是 `*out = latest_`,直接覆盖,不分配也行(`std::vector` 会按需 grow / shrink_to_fit 不会触发)。 |
| 怎么判断"这只手现在有 box"? | `r.boxes` 里找 `side == 0`(左)或 `1`(右);找不到说明这只手这帧没 box。 |
| `T_body_box` 怎么用? | `R = T[0..2,0..2]`,`t = (T[3], T[7], T[11])`;`X_body = R · X_box + t`。X_box 是 `box_{left,right}.txt` 里 7 个 LED 的 mm → m 之后的坐标。 |
| `step3_points` 一只手 21 条凑得齐吗? | 不一定。`HandKpts3d.kpt_valid` 过滤过(conf < 0.3 或射线距离 > 3 cm 的 kpt 不会出);**短一两条很正常**,按 `kpt` 字段查。 |
| `step2_points` 跟 `fingers/box` 的 3D 点有重复吗? | **有**。`fingers` 里的 `xyz_body` 复用自某条 `step2_points` 的 `xyz`(同一个 IR 点);`box` 7 个 LED 也来自 step2。`step2_points` 是全集。 |
