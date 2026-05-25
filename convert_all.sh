#!/bin/sh
# One-shot pipeline: ONNX->RKNN x2 -> end-to-end bench (ONNX, then RKNN).
# Run from the repo root, after the .venv_rknn environment is set up.
# See README.md for env setup.

set -e

ROOT=$(cd "$(dirname "$0")" && pwd)
cd "$ROOT"

if [ -f ".venv_rknn/bin/activate" ]; then
    # POSIX-compatible source
    . ./.venv_rknn/bin/activate
fi

CALIB_N=${CALIB_N:-50}

# Eval data (val.json + val_images) is bundled into this submodule under
# eval/ — the pipeline is fully self-contained. Override via env if you
# point at a different held-out set.
EVAL_ANN=${EVAL_ANN:-eval/val.json}
EVAL_IMG_DIR=${EVAL_IMG_DIR:-eval/val_images}

echo "[1/4] convert rtmdet_s_hand_640.onnx -> rknn (calib n=$CALIB_N from calib/images)"
python scripts/onnx2rknn.py --model rtmdet \
    --onnx onnx/rtmdet_s_hand_640.onnx \
    --out  out/rtmdet_s_hand_640.rknn \
    --quantize --calib-dir calib/images --calib-n "$CALIB_N"

echo "[2/4] convert rtmpose_hand_256.onnx -> rknn (calib n=$CALIB_N from calib/images)"
python scripts/onnx2rknn.py --model rtmpose \
    --onnx onnx/rtmpose_hand_256.onnx \
    --out  out/rtmpose_hand_256.rknn \
    --quantize --calib-dir calib/images --calib-n "$CALIB_N"

echo "[3/4] end-to-end bench: ONNX (det+pose)"
python scripts/bench_e2e.py --backend onnx \
    --det  onnx/rtmdet_s_hand_640.onnx \
    --pose onnx/rtmpose_hand_256.onnx \
    --ann  "$EVAL_ANN" \
    --img-dir "$EVAL_IMG_DIR"

echo "[4/4] end-to-end bench: RKNN (PC simulator — rebuilds INT8 from ONNX)"
# toolkit 2.3.2 init_runtime(target=None) refuses load_rknn() graphs, so we
# re-run load_onnx + build(do_quantization) with the same preset/calib as
# step [1]/[2]. Determinism comes from random.seed(0) in build_calib_list.
python scripts/bench_e2e.py --backend rknn \
    --det  onnx/rtmdet_s_hand_640.onnx \
    --pose onnx/rtmpose_hand_256.onnx \
    --det-model rtmdet --pose-model rtmpose \
    --calib-dir calib/images --calib-n "$CALIB_N" \
    --ann  "$EVAL_ANN" \
    --img-dir "$EVAL_IMG_DIR"

echo "PASS: rknn ready under out/"
echo "  RKNN latency above is PC simulator, not on-board NPU."
echo "  next: copy out/*.rknn to wherever your runtime expects them."
