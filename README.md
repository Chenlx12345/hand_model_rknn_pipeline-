# hand_det_pose_convert

主机侧 ONNX → RKNN 转换流水线，用于手部检测 / 姿态估计算法；同时附带
PC 端的端到端精度 / 速度评测，可在不接触开发板的前提下，对转换后的 RKNN
与原始 ONNX 做对齐校验。目标平台为 **RK3588**，板端驱动版本
`RKNPU 0.9.8 20240828`（对应 librknnrt 2.3.x 系列）。

**本仓库范围**：仅做转换 + PC 端验证。不含板端运行时，不含部署脚本。

每轮共两个模型：

| 阶段 | ONNX                       | 输入     | preset  | 通道顺序 |
| ---- | -------------------------- | -------- | ------- | -------- |
| det  | rtmdet_s_hand_640.onnx     | 640x640  | rtmdet  | BGR      |
| pose | rtmpose_hand_256.onnx      | 256x256  | rtmpose | RGB      |

输出的 `.rknn` 落到 `out/`；自行复制到下游运行时 / 部署目录。

## 仓库结构

```
.
├── onnx/         输入 ONNX 模型（已入库）
├── calib/
│   └── images/      在库校准集（956 张手部 crop，约 200 MB）
├── eval/         留出的精度对齐验证集（已入库）
│   ├── val.json          COCO 标注（184 张图 / 296 个手部实例）
│   └── val_images/       34 张验证图（val.json 的子集，bench 自动取交集）
├── out/          转换产物（gitignored）
├── scripts/      Python 工具（onnx2rknn / bench_e2e / pipeline_lib）
├── docs/         补充说明
├── third_party/  rknn-toolkit2 submodule
└── convert_all.sh
```

本仓库**自包含**：`./convert_all.sh` 跑全流程不依赖任何外部数据 ——
校准集（`calib/images/`）、输入 ONNX（`onnx/`）、验证集（`eval/`）全部
已入库随仓库一起走。

`scripts/onnx2rknn.py` 在转换时用固定种子从 `calib/images/` 抽 `--calib-n`
张作为校准集，因此校准过程**只**依赖仓库内数据即可复现。

## 版本绑定（必须对齐）

| 组件                  | 版本     | 理由                                            |
| --------------------- | -------- | ----------------------------------------------- |
| 板端 librknnrt        | 2.3.x    | 来自 `dmesg \| grep RKNPU` -> `0.9.8 20240828` |
| rknn-toolkit2 （主机） | 2.3.2    | 与板端已部署的 `.rknn` 文件头声明的版本一致     |
| onnx                  | 1.16.1   | 1.17+ 移除了 `onnx.mapping`，会破坏 toolkit     |
| python                | 3.10     | toolkit wheel 的目标 Python 版本                |
| numpy                 | 1.26.4   | onnx / toolkit 的兼容窗口                       |

## 首次环境搭建

```sh
python3.10 -m venv .venv_rknn
. .venv_rknn/bin/activate

# rknn-toolkit2 不在 PyPI，从子模块本地 wheel 安装
git submodule update --init --recursive
pip install third_party/rknn-toolkit2/rknn-toolkit2/packages/x86_64/rknn_toolkit2-2.3.2-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl

pip install -r requirements.txt
```

环境自检：

```sh
python -c "from importlib.metadata import version; print(version('rknn-toolkit2'))"   # 期望 2.3.2
python -c "import onnx; print(onnx.__version__)"                          # 期望 1.16.1
```

## 一键流水线

```sh
./convert_all.sh         # 默认 CALIB_N=50
CALIB_N=100 ./convert_all.sh
```

## 手动分步

```sh
# 转换
python scripts/onnx2rknn.py --model rtmdet  \
    --onnx onnx/rtmdet_s_hand_640.onnx \
    --out  out/rtmdet_s_hand_640.rknn  \
    --quantize --calib-dir calib/images --calib-n 50

python scripts/onnx2rknn.py --model rtmpose \
    --onnx onnx/rtmpose_hand_256.onnx  \
    --out  out/rtmpose_hand_256.rknn   \
    --quantize --calib-dir calib/images --calib-n 50

# 端到端 PC 评测：ONNX 基线
python scripts/bench_e2e.py --backend onnx \
    --det  onnx/rtmdet_s_hand_640.onnx \
    --pose onnx/rtmpose_hand_256.onnx  \
    --ann  eval/val.json --img-dir eval/val_images

# 端到端 PC 评测：转换后的 RKNN（toolkit2 模拟器）
# 注意：toolkit 2.3.2 的 PC 模拟器 **不能** 加载 .rknn 文件
# （init_runtime(target=None) 拒绝 load_rknn 出来的图）。
# 这里 --det / --pose 传 ONNX 路径，bench 内部用同一份 preset + calib
# 重建 INT8 图，再喂给 simulator。calib 列表用 random.seed(0) 固定，
# 与 onnx2rknn.py 落盘的 .rknn 是同一份量化图。
python scripts/bench_e2e.py --backend rknn \
    --det  onnx/rtmdet_s_hand_640.onnx \
    --pose onnx/rtmpose_hand_256.onnx  \
    --det-model rtmdet --pose-model rtmpose \
    --calib-dir calib/images --calib-n 50 \
    --ann  eval/val.json --img-dir eval/val_images
```

每次评测会打印 `PASS:` 行，包含检测 recall、PCK@5、端到端延迟 / FPS。
比对 ONNX 与 RKNN 即可发现量化退化：recall 下降 ≤ 5pp，且 PCK@5 下降
≤ 5pp 是预期阈值。

> **延迟说明**：RKNN 评测跑在 rknn-toolkit2 的 PC 模拟器上
> （`target=None`），并非板端 NPU。该数字仅用于精度回归排查，**不能**
> 当作板端 FPS 参考。

## 部署到开发板

`out/*.rknn` 已 gitignore。把转换产物复制到下游运行时 / 部署目录即可；
具体路径由项目侧决定，与本仓库无关。

```sh
cp out/rtmdet_s_hand_640.rknn <deploy_dir>/
cp out/rtmpose_hand_256.rknn  <deploy_dir>/
```

检测器文件名从旧的 `rtmdet_nano_hand_320.rknn` 变为
`rtmdet_s_hand_640.rknn`。下游消费方需同步更新文件路径以及 letterbox
输入尺寸（320 → 640）。

## 注意事项

- RTMDet preset 直接以 **BGR** 喂图（`quant_img_RGB2BGR=False`）。
  toolkit 2.3.2 内部的 RGB→BGR swap 路径对此模型有 bug，因此任何消费
  此 `.rknn` 的运行时也必须直接喂 BGR。
- 量化模式 `w8a8`，硬编码在 `scripts/onnx2rknn.py`。
- `out/` 已 gitignore；自行把转换产物拷贝到下游运行时 / 部署目录。
