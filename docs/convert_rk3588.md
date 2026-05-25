# RK3588 转换说明

## 版本绑定

板端驱动版本（来源 `dmesg | grep RKNPU`）：

```text
[    5.436644] [drm] Initialized rknpu 0.9.8 20240828 for fdab0000.npu on minor 1
```

`rknpu 0.9.8 20240828` 对应用户态 runtime（`librknnrt.so`）2.3.x 系列 → 主机
toolkit 锁 **rknn-toolkit2 2.3.2**。

onnx 锁 **1.16.1**：1.17+ 移除了 `onnx.mapping`，会让 `from rknn.api import RKNN`
直接 `ImportError`。

## Calibration policy

- INT8 量化使用 `quantized_dtype="w8a8"`（`scripts/onnx2rknn.py`）。
- 校准集 = `datasets/calib/images/` 下**全部图像**（按文件名排序、不抽样、无 RNG）。
  扩 / 缩校准集直接增删该目录下的图。
- 校准集与 `datasets/eval/val_images/` 物理分离。

## RTMDet 通道顺序陷阱

`scripts/onnx2rknn.py` 中 RTMDet preset 设 `quant_img_RGB2BGR=False`，直接把 BGR
喂进网络。toolkit 2.3.2 中此 RTMDet 走 swap 路径会 misquantize。

下游消费此 .rknn 的运行时必须按同样方式喂图：直接 BGR，不在外部预 swap，也不在
外部预 normalize（mean/std 已烘焙进 `.rknn`）。

## PC 模拟器 vs 板端 NPU

`scripts/bench_e2e.py --backend rknn` 跑在 toolkit2 的 PC 模拟器上
（`init_runtime(target=None)`）：

- **精度数据有效**：模拟器跑的就是板端会跑的量化图。
- **延迟数据无效**：模拟器走 CPU，与板端 NPU 吞吐无对应关系。
