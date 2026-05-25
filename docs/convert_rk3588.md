# RK3588 转换说明

## 为什么必须用 rknn-toolkit2 2.3.2

板端启动日志（`dmesg | grep RKNPU`）：

```
[    5.434852] RKNPU fdab0000.npu: Adding to iommu group 0
[    5.436644] [drm] Initialized rknpu 0.9.8 20240828 for fdab0000.npu on minor 1
```

`rknpu 0.9.8 20240828` 是 kernel driver 的版本标签，与之匹配的用户态
runtime（`librknnrt.so`）位于 2.3.x 系列。为了保证主机生成的 `.rknn`
能在板端按字节兼容运行，主机侧 toolkit 必须**锁定 rknn-toolkit2 2.3.2**。

板端此前部署的 `.rknn` 文件，经文件头检查同样是 toolkit 2.3.2 产出。
因此沿用同一版本可以彻底排除"版本漂移导致的转换回归"这一类排查方向。

## 为什么 onnx 必须用 1.16.1

`rknn-toolkit2`（含 2.3.2）的多个内部模块顶部都会
`import onnx.mapping`。上游 onnx 从 1.17.0 起删除了 `onnx.mapping`，
任何更高版本会让 `from rknn.api import RKNN` 直接抛 `ImportError`。
1.16.1 是仍保留该模块的最后一个发行版。

## Calibration policy

- INT8 量化使用 `quantized_dtype="w8a8"`
  （在 `scripts/onnx2rknn.py` 内设定）。
- 校准集放在 `calib/images/` —— 956 张手部 crop，已随仓库入库。
  `scripts/onnx2rknn.py` 的 `build_calib_list` 在转换时按固定种子
  （默认 0）从中采 `--calib-n` 张，按 build 写出 `_calib_*.txt`
  （gitignored）作为 toolkit 的 dataset.txt。
- 校准集和 `eval/val_images/`（34 张留出验证集）**物理分离**，避免
  对验证集的改动悄悄影响量化行为。
- 默认采样数 50；可通过 `CALIB_N=...` 环境变量或 `onnx2rknn.py` 的
  `--calib-n` 参数加大，以更紧致的量化换取更长的 build 时间。

## RTMDet 通道顺序陷阱

`scripts/onnx2rknn.py` 中的 RTMDet preset 设
`quant_img_RGB2BGR=False`，直接把 BGR 喂进网络。在 toolkit 2.3.2 中，
针对这一特定 RTMDet 变体走 swap 路径会引起 misquantize，因此**不要**
轻易翻转该 swap 标志；如需调整，必须先重新跑
`scripts/bench_e2e.py --backend rknn`，确认留出验证集上 recall 与 PCK
仍在预算内。

任何下游消费此 .rknn 的运行时必须按同样方式喂图：直接 BGR，不在外部
预 swap，也不在外部预 normalize（mean/std 已经烘焙进 `.rknn`）。

## PC 端 bench 与板端 NPU 的区别

`scripts/bench_e2e.py --backend rknn` 是把转换后的 .rknn 跑在
rknn-toolkit2 的 PC 模拟器上（`init_runtime(target=None)`）。模拟器
是 NPU op graph 的软件仿真：

- **精度数据是有效的** —— 模拟器跑的就是板端会跑的量化图。这里出现
  recall / PCK 回归，就是真实的量化问题。
- **延迟数据无效** —— 模拟器走 CPU，与板端 NPU 吞吐没有任何对应
  关系。**不要**把模拟器的 `E2E_FPS` 拿来当板端数字引用。

板端真实 FPS 需要单独的板端 benchmark，它不在本子模块的范围内。

## 把转换产物送去部署

`out/*.rknn` 已 gitignore。要部署时，把它们拷到下游运行时期望的位置；
具体目标目录、文件命名、C++ loader 路径，由板端项目自行管理，这里
有意不收口：

```sh
cp out/rtmdet_s_hand_640.rknn <deploy_dir>/
cp out/rtmpose_hand_256.rknn  <deploy_dir>/
```

检测器文件名从旧的 `rtmdet_nano_hand_320.rknn` 变为
`rtmdet_s_hand_640.rknn`。下游所有消费方都需要同步更新 —— 包括 loader
路径，以及把硬编码的 letterbox 尺寸从 320 改为 640。
