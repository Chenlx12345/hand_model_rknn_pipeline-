# hand_det_pose_convert

Host-side ONNX -> RKNN conversion pipeline for the hand detection / pose
estimation algorithm, plus a PC-side end-to-end accuracy/speed bench so
the converted RKNN can be sanity-checked against the original ONNX
without touching a board. Targets **RK3588** with the on-board
`RKNPU 0.9.8 20240828` driver (librknnrt 2.3.x line).

**Scope of this repo**: convert + PC-side validation only. No board-side
runtime, no deployment scripts.

Two models per turn:

| stage | ONNX                       | input    | preset  | channel |
| ----- | -------------------------- | -------- | ------- | ------- |
| det   | rtmdet_s_hand_640.onnx     | 640x640  | rtmdet  | BGR     |
| pose  | rtmpose_hand_256.onnx      | 256x256  | rtmpose | RGB     |

Output `.rknn` files land in `out/`; copy them out to your runtime/deployment dir.

## Repository layout

```
.
├── onnx/         input ONNX models (provided)
├── calib/
│   └── images/      in-repo calibration set (956 hand crops, ~200 MB)
├── eval/         held-out accuracy-check set (bundled in-repo)
│   ├── val.json          COCO annotations (34 images)
│   └── val_images/       34 validation images
├── out/          conversion artefacts (gitignored)
├── scripts/      python tools (onnx2rknn, bench_e2e, pipeline_lib)
├── docs/         additional notes
├── third_party/  rknn-toolkit2 submodule
└── convert_all.sh
```

The repo is **self-contained**: `./convert_all.sh` runs the full pipeline
without needing any external data — the calibration set (`calib/images/`),
the input ONNX models (`onnx/`), and the eval set (`eval/`) are all bundled
in-tree.

`scripts/onnx2rknn.py` samples `--calib-n` images out of `calib/images/`
at conversion time with a fixed seed, so calibration is reproducible from
the in-repo data alone.

## Version pinning (must match)

| component         | version  | reason                                        |
| ----------------- | -------- | --------------------------------------------- |
| board librknnrt   | 2.3.x    | from `dmesg | grep RKNPU` -> `0.9.8 20240828` |
| rknn-toolkit2 (host) | 2.3.2 | matches the `.rknn` already on the board      |
| onnx              | 1.16.1   | 1.17+ removed `onnx.mapping`, breaks toolkit  |
| python            | 3.10     | toolkit wheel target                          |
| numpy             | 1.26.4   | onnx/toolkit compatibility window             |

## First-time setup

```sh
python3.10 -m venv .venv_rknn
. .venv_rknn/bin/activate

# rknn-toolkit2 wheel comes from the submodule (not PyPI)
git submodule update --init --recursive
pip install third_party/rknn-toolkit2/rknn-toolkit2/packages/x86_64/cp310/rknn_toolkit2-2.3.2-cp310-cp310-linux_x86_64.whl

pip install -r requirements.txt
```

Sanity check:

```sh
python -c "from rknn.api import RKNN; print(RKNN().get_sdk_version())"   # expect 2.3.2
python -c "import onnx; print(onnx.__version__)"                          # expect 1.16.1
```

## One-shot pipeline

```sh
./convert_all.sh         # default CALIB_N=50
CALIB_N=100 ./convert_all.sh
```

Manual step-by-step:

```sh
# convert
python scripts/onnx2rknn.py --model rtmdet  \
    --onnx onnx/rtmdet_s_hand_640.onnx \
    --out  out/rtmdet_s_hand_640.rknn  \
    --input-size 640 640 \
    --quantize --calib-dir calib/images --calib-n 50

python scripts/onnx2rknn.py --model rtmpose \
    --onnx onnx/rtmpose_hand_256.onnx  \
    --out  out/rtmpose_hand_256.rknn   \
    --input-size 256 256 \
    --quantize --calib-dir calib/images --calib-n 50

# end-to-end PC bench: ONNX baseline
python scripts/bench_e2e.py --backend onnx \
    --det  onnx/rtmdet_s_hand_640.onnx \
    --pose onnx/rtmpose_hand_256.onnx  \
    --ann  eval/val.json --img-dir eval/val_images

# end-to-end PC bench: converted RKNN (toolkit2 simulator)
python scripts/bench_e2e.py --backend rknn \
    --det  out/rtmdet_s_hand_640.rknn \
    --pose out/rtmpose_hand_256.rknn  \
    --ann  eval/val.json --img-dir eval/val_images
```

Each bench prints a `PASS:` line with detector recall, PCK@5, and
end-to-end latency / FPS. Compare ONNX vs RKNN to catch quantization
regressions: recall drop ≤ 5pp and PCK@5 drop ≤ 5pp is the expected
budget.

> **Latency caveat**: the RKNN bench runs on the rknn-toolkit2 PC
> simulator (`target=None`), not on the board NPU. Use it for accuracy
> regression checks, not as a proxy for on-board FPS.

## Deploying to the board

`out/*.rknn` is gitignored. Copy the converted artefacts to whatever
runtime / deployment directory your project uses; that wiring lives
outside this submodule.

```sh
cp out/rtmdet_s_hand_640.rknn <deploy_dir>/
cp out/rtmpose_hand_256.rknn  <deploy_dir>/
```

Detector file name changes from the legacy `rtmdet_nano_hand_320.rknn`
to `rtmdet_s_hand_640.rknn`. Update consumer paths and letterbox input
size (320 -> 640) accordingly on the runtime side.

## Notes / gotchas

- The RTMDet preset feeds **BGR** directly (`quant_img_RGB2BGR=False`). The
  internal RGB->BGR swap path in toolkit 2.3.2 is buggy for this model, so
  any runtime consuming the .rknn must also feed BGR.
- Quantization is `w8a8`; this is hard-coded in `scripts/onnx2rknn.py`.
- `out/` is gitignored; promote the artefacts to your runtime/deployment dir manually.
