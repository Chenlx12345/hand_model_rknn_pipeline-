# hand_det_pose_convert

Host-side ONNX -> RKNN conversion pipeline for the hand detection / pose
estimation algorithm shipped in `external/ant_algorithm/`. Targets **RK3588**
with the on-board `RKNPU 0.9.8 20240828` driver (librknnrt 2.3.x line).

Two models per turn:

| stage | ONNX                       | input    | preset  | channel |
| ----- | -------------------------- | -------- | ------- | ------- |
| det   | rtmdet_s_hand_640.onnx     | 640x640  | rtmdet  | BGR     |
| pose  | rtmpose_hand_256.onnx      | 256x256  | rtmpose | RGB     |

Output `.rknn` files land in `out/`; promote them to `../rknn/` for deployment.

## Repository layout

```
.
├── onnx/         input ONNX models (provided)
├── calib/
│   ├── source_pool/  in-repo calibration pool (956 hand crops, ~200MB)
│   └── images/       per-build sampled subset (gitignored)
├── eval/         held-out artefacts produced by this pipeline only
│   └── onnx_ref_kpts.npz   golden reference from compare_pose
├── out/          conversion artefacts (gitignored)
├── scripts/      python tools (see below)
├── docs/         additional notes
├── third_party/  rknn-toolkit2 submodule
└── convert_all.sh
```

Evaluation data (val.json + val_images/) is **not** kept here — it lives at
`../rknn/` in the parent superproject (`external/ant_algorithm/rknn/`). The
compare scripts reference it through that relative path; override at runtime
with `EVAL_ANN=... EVAL_IMG_DIR=... ./convert_all.sh` if the parent layout
changes.

`calib/source_pool/` is a fixed-population pool: `prepare_calib.py` samples
from it into `calib/images/` (gitignored) for each build, ensuring that
calibration is reproducible from the in-repo data alone.

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
python scripts/prepare_calib.py --src calib/source_pool --dst calib/images --n 50

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

python scripts/compare_det.py  --onnx onnx/rtmdet_s_hand_640.onnx \
    --ann ../rknn/val.json --img-dir ../rknn/val_images
python scripts/compare_pose.py --onnx onnx/rtmpose_hand_256.onnx \
    --ann ../rknn/val.json --img-dir ../rknn/val_images \
    --save-ref-kpts eval/onnx_ref_kpts.npz
```

## Deploying to the board

```sh
cp out/rtmdet_s_hand_640.rknn ../rknn/
cp out/rtmpose_hand_256.rknn  ../rknn/
# scp ../rknn/*.rknn <board>:/path/to/runtime/
```

Detector model name changes from `rtmdet_nano_hand_320.rknn` (the legacy
one shipped under `../rknn/`) to `rtmdet_s_hand_640.rknn`. Update the C++
loader paths and the letterbox input size (320 -> 640) accordingly.

## Notes / gotchas

- The RTMDet preset feeds **BGR** directly (`quant_img_RGB2BGR=False`). The
  internal RGB->BGR swap path in toolkit 2.3.2 is buggy for this model, so
  the runtime side must also feed BGR.
- `onnx_ref_kpts.npz` under `eval/` is a cached set of ONNX-predicted
  keypoints. It is the golden reference for pose accuracy checks; it is
  **not** the INT8 calibration set.
- Quantization is `w8a8`; this is hard-coded in `scripts/onnx2rknn.py`.
- `out/` is gitignored; the artefacts live under `../rknn/` for deployment.
