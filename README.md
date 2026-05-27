# hand_model_rknn_pipeline

主机侧 ONNX → RKNN 转换流水线 + PC 端精度 / 速度评测。目标平台 RK3588
（librknnrt 2.3.x）。

作为主仓库子模块使用时，默认路径为
`external/hand_model_rknn_pipeline`。

| 阶段 | ONNX                       | 输入     | preset  | 通道顺序 |
| ---- | -------------------------- | -------- | ------- | -------- |
| det  | rtmdet_tiny_hand_640.onnx     | 640x640  | rtmdet  | BGR      |
| pose | rtmpose_hand_256.onnx      | 256x256  | rtmpose | RGB      |

## 仓库结构

```
.
├── onnx/             输入 ONNX 模型
├── datasets/         校准 + 验证数据
│   ├── calib/images/     956 张手部 crop
│   └── eval/
│       ├── val.json          COCO 标注（184 张图 / 296 个手部实例）
│       └── val_images/       34 张验证图（val.json 的子集）
├── out/              转换产物（gitignored）
├── scripts/          onnx2rknn / bench_e2e / pipeline_lib / viz_lib
├── third_party/      rknn-toolkit2 submodule
└── convert_all.sh
```

## 版本绑定（必须对齐）

| 组件                  | 版本     | 理由                                            |
| --------------------- | -------- | ----------------------------------------------- |
| 板端 librknnrt        | 2.3.x    | `dmesg \| grep RKNPU` -> `0.9.8 20240828`       |
| rknn-toolkit2 （主机） | 2.3.2    | 与板端 `.rknn` 文件头声明一致                   |
| onnx                  | 1.16.1   | 1.17+ 移除了 `onnx.mapping`                     |
| python                | 3.10     | toolkit wheel 目标版本                          |
| numpy                 | 1.26.4   |                                                 |

## 首次环境搭建

```sh
python3.10 -m venv .venv_rknn
. .venv_rknn/bin/activate

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
./convert_all.sh
```

## 手动分步

```sh
# 转换
python scripts/onnx2rknn.py --model rtmdet  \
    --onnx onnx/rtmdet_tiny_hand_640.onnx \
    --out  out/rtmdet_tiny_hand_640.rknn  \
    --quantize --calib-dir datasets/calib/images

python scripts/onnx2rknn.py --model rtmpose \
    --onnx onnx/rtmpose_hand_256.onnx  \
    --out  out/rtmpose_hand_256.rknn   \
    --quantize --calib-dir datasets/calib/images

# 端到端 PC 评测：ONNX 基线 + RKNN 模拟器一次跑完，并列输出 recall / PCK / 延迟，
# 同时在 out/viz/ 下生成 ONNX | RKNN 左右拼接图，直接看 INT8 量化漂移。
python scripts/bench_e2e.py --backend both \
    --det  onnx/rtmdet_tiny_hand_640.onnx \
    --pose onnx/rtmpose_hand_256.onnx  \
    --det-model rtmdet --pose-model rtmpose \
    --calib-dir datasets/calib/images \
    --ann  datasets/eval/val.json --img-dir datasets/eval/val_images
```

ONNX vs RKNN：recall / PCK@5 差 ≤ 5pp 为预期（`--backend both` 表格列出两者对比）。

## 部署到开发板

```sh
cp out/rtmdet_tiny_hand_640.rknn <deploy_dir>/
cp out/rtmpose_hand_256.rknn  <deploy_dir>/
```
