# deploy/ — 板端 C++ 端到端推理 + 评测

主机侧 `scripts/bench_e2e.py` 跑的是 toolkit2 模拟器；本目录提供一个**真正在 RK3588
NPU 上**消费 `out/*.rknn` 的 C++ 版本，复现同样的指标：detection recall/precision、
PCK@5/10/20、以及板端实测延迟。

## 工具链

复用主仓库 `cmake/toolchain-rk3588.cmake`（由 `cmake/local-config.cmake` 注入
`DAS_EGO_SDK_PATH`）。

## 编译

```sh
cd external/ant_algorithm/hand_det_pose_convert/deploy
./build.sh
# 产物：build/bench_e2e_rknn  (ELF 64-bit LSB, aarch64)
```

## 部署 + 运行

把以下内容拷到板端同一目录（示例放 `/userdata/hand_eval/`）：

```
/userdata/hand_eval/
├── bench_e2e_rknn               (./deploy/build/)
├── rtmdet_s_hand_640.rknn       (../out/)
├── rtmpose_hand_256.rknn        (../out/)
├── val.json                     (../datasets/eval/)
└── val_images/                  (../datasets/eval/)
```

板端执行：

```sh
cd /userdata/hand_eval
./bench_e2e_rknn \
    --det  rtmdet_s_hand_640.rknn \
    --pose rtmpose_hand_256.rknn  \
    --ann  val.json \
    --img-dir val_images
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
|---|---|
| librknnrt.so + rknn_api.h | SDK buildroot sysroot：`usr/{lib,include/rknn}` |
| OpenCV 3.4.5（静态） | `rknpu2-1.0.0/examples/3rdparty/opencv/opencv-linux-aarch64` |
| nlohmann/json | 仓库 `external/nlohmann_json/single_include/` |
