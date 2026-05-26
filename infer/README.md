# infer/ — 板端 RKNN 推理库 + 板上自验证

`libhand_pipeline.a` 提供 RTMDet + RTMPose 在 RK3588 NPU 上的推理 API；
同目录 `tests/` 提供 `unit_bench_e2e_test`：用 COCO `val.json` 在板上跑端到端精度 / 延迟。

## 目录结构

```text
infer/
├── CMakeLists.txt        # 库 + (默认开启的) tests
├── build.sh              # 一键交叉编译
├── include/              # 公共头（无额外子目录）
│   ├── rknn_session.h
│   ├── rtmdet.h
│   ├── rtmpose.h
│   └── hand_pipeline.h   # 高层一站式封装
├── src/                  # 库实现
│   ├── rknn_session.cpp
│   ├── rtmdet.cpp
│   ├── rtmpose.cpp
│   └── hand_pipeline.cpp
└── tests/                # 板上测试
    ├── CMakeLists.txt
    ├── unit_bench_e2e_test.cpp  # 测试入口
    ├── eval.h                   # COCO / IoU / PCK 辅助
    └── eval.cpp
```

## 工具链

复用主仓库 `cmake/toolchain-rk3588.cmake`（由 `cmake/local-config.cmake` 注入 `DAS_EGO_SDK_PATH`）。

## 编译

```sh
cd external/hand_model_rknn_pipeline/infer
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
add_subdirectory(<...>/external/hand_model_rknn_pipeline/infer hand_pipeline_build)
target_link_libraries(<your_app> PRIVATE hand_pipeline)
# include 路径 / OpenCV / RKNN 都通过 PUBLIC 自动传递过来。
# tests 默认 OFF（add_subdirectory 模式），不会污染你的构建。
```

## API 用法

### 高层（推荐）：`HandPipeline::Run`

```cpp
#include "hand_pipeline.h"

hand_deploy::HandPipeline hp("rtmdet_s_hand_640.rknn",
                             "rtmpose_hand_256.rknn");
cv::Mat frame = cv::imread("hand.jpg");

hand_deploy::HandTiming t;
auto hands = hp.Run(frame, &t);
for (const auto& h : hands) {
    // h.bbox.{x1,y1,x2,y2,score}, h.kpts[k].{x,y,conf}
}
```

### 低层：直接组合 `RtmDet` + `RtmPose`

```cpp
#include "rtmdet.h"
#include "rtmpose.h"

hand_deploy::RtmDet  det ("rtmdet_s_hand_640.rknn", {});
hand_deploy::RtmPose pose("rtmpose_hand_256.rknn",  {});

double det_ms = 0.0;
auto dets = det.Detect(frame, &det_ms);
for (const auto& d : dets) {
    double pose_ms = 0.0;
    auto kpts = pose.Estimate(frame, d, &pose_ms);
}
```

低层路径适合自己控制串行 / 并行、单独跑检测、或需要精细化分段计时的场景。

## 板端跑 unit_bench_e2e_test

把以下文件拷到板端同一目录（示例 `/usr/das_ego/ant/`，也可以是 `/userdata/hand_eval/`）。
`eval/` 子目录直接从主机 `datasets/eval/` 整体拷过来，**保持原文件名**：

```text
/usr/das_ego/ant/
├── unit_bench_e2e_test         (infer/build/)
├── rtmdet_s_hand_640.rknn      (../out/)
├── rtmpose_hand_256.rknn       (../out/)
└── eval/                       (= datasets/eval/)
    ├── val.json
    └── val_images/
```

板端执行：

```sh
cd /usr/das_ego/ant
./unit_bench_e2e_test \
    --det  rtmdet_s_hand_640.rknn \
    --pose rtmpose_hand_256.rknn  \
    --eval eval
```

可选参数：`--score-thr 0.4`、`--nms 0.6`、`--warmup 3`、`--n N`（限制图片张数）。

## 与 Python 基线比对

```sh
# 主机参考
python scripts/bench_e2e.py --backend rknn \
    --det  onnx/rtmdet_s_hand_640.onnx --pose onnx/rtmpose_hand_256.onnx \
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
