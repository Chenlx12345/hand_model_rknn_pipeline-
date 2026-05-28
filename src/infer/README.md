# src/infer — 板端 RKNN 推理库 + 板上自验证

`libhand_pipeline.a` 提供 RTMDet + RTMPose 在 RK3588 NPU 上的推理 API；
同目录 `tests/` 提供 `unit_bench_e2e_test`：用 COCO `val.json` 在板上跑端到端精度 / 延迟。

## 目录结构

```text
src/infer/
├── CMakeLists.txt        # 库 + (默认开启的) tests
├── build.sh              # 一键交叉编译
├── include/
│   ├── rknn_session.h
│   ├── rtmdet.h
│   ├── rtmpose.h
│   └── hand_pipeline.h   # 高层一站式封装
├── src/
│   ├── rknn_session.cpp
│   ├── rtmdet.cpp
│   ├── rtmpose.cpp
│   └── hand_pipeline.cpp
└── tests/
    ├── CMakeLists.txt
    ├── unit_bench_e2e_test.cpp
    ├── eval.h                   # COCO / IoU / PCK 辅助
    ├── eval.cpp
    ├── viz_lib.hpp              # 板端绘图（与主机 viz_lib.py 对齐）
    └── viz_lib.cpp
```

## 分层设计

三层职责（自底向上）：

| 层 | 类 / 文件 | 职责 |
|----|-----------|------|
| L1 | `RknnSession` (`rknn_session.{h,cpp}`) | NPU transport：模型加载 / 推理 / 释放 |
| L2 | `RtmDet` (`rtmdet.{h,cpp}`) / `RtmPose` (`rtmpose.{h,cpp}`) | 单模型管线：letterbox / affine / SimCC 解码 |
| L3 | `HandPipeline` (`hand_pipeline.{h,cpp}`) | 编排 facade：det → 循环 pose + 可选 timing，业务推荐入口 |

L2 直接组合面向「需要 per-stage 计时 / 跳过 pose / 在两阶段之间插桩」的特殊消费者
（如本目录 `tests/unit_bench_e2e_test.cpp` 需拿 `det_ms` / `pose_ms` 分段 bench）。

## 工具链

复用主仓库 `cmake/toolchain-rk3588.cmake`（由 `cmake/local-config.cmake` 注入 `DAS_EGO_SDK_PATH`）。

## 编译

```sh
cd external/hand_model_rknn_pipeline/src/infer
./build.sh
# 产物：
#   build/libhand_pipeline.a   (aarch64 静态库)
#   build/unit_bench_e2e_test  (aarch64 ELF，板上跑)
```

只想构建库、不要测试（不拉 `nlohmann/json`）：

```sh
cmake -B build_lib_only \
      -DCMAKE_TOOLCHAIN_FILE=../../../cmake/toolchain-rk3588.cmake \
      -DHAND_PIPELINE_BUILD_TESTS=OFF .
cmake --build build_lib_only -j
```

## 在其它 CMake 工程里使用

```cmake
add_subdirectory(<...>/external/hand_model_rknn_pipeline/src/infer hand_pipeline_build)
target_link_libraries(<your_app> PRIVATE hand_pipeline)
# include 路径 / OpenCV / RKNN 都通过 PUBLIC 自动传递过来。
# tests 默认 OFF（add_subdirectory 模式），不会污染你的构建。
```

## API 用法

### 高层（推荐）：`HandPipeline::Run`

业务侧只想要「这一帧里的所有手」时，直接用 facade，一个调用拿到 bbox + 21 关键点
（全部已映射回原图坐标系），可选填 `HandTiming*` 拿到各阶段延迟：

```cpp
#include "hand_pipeline.h"

hand_pipeline::HandPipeline hp("rtmdet.rknn",
                             "rtmpose.rknn");

hand_pipeline::HandTiming t;
auto hands = hp.Run(frame, &t);
// t.det_ms / t.pose_ms[i] / t.total_ms
for (const auto& h : hands) {
    // h.bbox  -> Detection（原图坐标）
    // h.kpts  -> std::array<Keypoint, 21>（原图坐标）
}
```

### 低层：直接组合 `RtmDet` + `RtmPose`

需要自己控制串/并行、跳过 pose、或在两阶段之间插入过滤逻辑时，绕开 facade：

```cpp
#include "rtmdet.h"
#include "rtmpose.h"

hand_pipeline::RtmDet  det ("rtmdet.rknn", {});
hand_pipeline::RtmPose pose("rtmpose.rknn", {});

double det_ms = 0.0;
auto dets = det.Detect(frame, &det_ms);
for (const auto& d : dets) {
    double pose_ms = 0.0;
    auto kpts = pose.Estimate(frame, d, &pose_ms);
}
```

bbox 与关键点都已经映射到原图坐标系，调用方无需再做坐标变换。
低层路径适合自己控制串行 / 并行、跳过 pose、或在 det 与 pose 之间插桩；
正常业务路径请优先使用 `HandPipeline`，避免重复编排管线拓扑。

## 板端跑 unit_bench_e2e_test

把以下文件拷到板端同一目录（示例 `/usr/das_ego/ant/`，也可以是 `/userdata/hand_eval/`）。
`eval/` 子目录直接从主机 `datasets/eval/` 整体拷过来，**保持原文件名**：

```text
/usr/das_ego/ant/
├── unit_bench_e2e_test         (src/infer/build/)
├── rtmdet.rknn                 (assets/models/)
├── rtmpose.rknn                (assets/models/)
└── eval/                       (= datasets/eval/)
    ├── val.json
    └── val_images/
```

板端执行：

```sh
cd /usr/das_ego/ant
./unit_bench_e2e_test \
    --det  rtmdet.rknn \
    --pose rtmpose.rknn \
    --eval eval
```

可选参数：`--score-thr 0.4`、`--nms 0.6`、`--warmup 3`、`--n N`（限制图片张数）。

执行完会在 `eval/viz/` 下生成 N 张与 `val_images/` 同名的 `${fname}.jpg`，
每张含绿色 bbox + 21 关节圆点 + 骨架连线，左上角标 `RKNN-NPU`。配色 / 骨架 /
字体与主机端 `scripts/viz_lib.py:draw_hands` 字面一致，可直接和主机 `out/viz/`
（`ONNX` 基线 / `RKNN-SIM` 模拟器）同名 jpg 并排 diff，肉眼核对量化漂移与
NPU 数值差异。

## 与 Python 基线比对

```sh
# 主机参考：一次跑出 ONNX 基线 + RKNN 模拟器对比，与板端 unit_bench_e2e_test
# 的 rknn-npu 数字形成三段对照（ONNX / RKNN-SIM / RKNN-NPU）。
python scripts/bench_e2e.py --backend both \
    --det  onnx/rtmdet.onnx --pose onnx/rtmpose.onnx \
    --det-model rtmdet --pose-model rtmpose \
    --calib-dir datasets/calib/images \
    --ann datasets/eval/val.json --img-dir datasets/eval/val_images
```

板端与主机模拟器允许差异：

- `recall` / `precision`：≤ 1pp
- `PCK@10` / `PCK@20`：≤ 2pp
- `PCK@5`：可能略大（像素级敏感），但同数量级

差异超出此阈值通常说明：letterbox 像素中心约定、affine 仿射变换源/目标点、SimCC 解码
顺序、或 NMS 边界条件出现偏移，参考 `scripts/pipeline_lib.py` 逐行对齐。

## 依赖来源

| 库 | 来源 |
| --- | --- |
| `librknnrt.so` + `rknn_api.h` | SDK buildroot sysroot：`usr/{lib,include/rknn}` |
| OpenCV 3.4.5（静态） | `rknpu2-1.0.0/examples/3rdparty/opencv/opencv-linux-aarch64` |
| `nlohmann/json` | 仓库 `external/nlohmann_json/single_include/`（仅 tests 用） |
