#!/bin/sh
# One-shot pipeline: prepare calib -> ONNX->RKNN x2 -> accuracy compare x2.
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
    --input-size 640 640 \
    --quantize --calib-dir calib/images --calib-n "$CALIB_N"

echo "[2/4] convert rtmpose_hand_256.onnx -> rknn (calib n=$CALIB_N from calib/images)"
python scripts/onnx2rknn.py --model rtmpose \
    --onnx onnx/rtmpose_hand_256.onnx \
    --out  out/rtmpose_hand_256.rknn \
    --input-size 256 256 \
    --quantize --calib-dir calib/images --calib-n "$CALIB_N"

echo "[3/4] accuracy compare: detector (ONNX vs RKNN-fp16 simulator)"
python scripts/compare_det.py \
    --onnx onnx/rtmdet_s_hand_640.onnx \
    --ann  "$EVAL_ANN" \
    --img-dir "$EVAL_IMG_DIR"

echo "[4/4] accuracy compare: pose (ONNX vs RKNN-fp16 simulator)"
python scripts/compare_pose.py \
    --onnx onnx/rtmpose_hand_256.onnx \
    --ann  "$EVAL_ANN" \
    --img-dir "$EVAL_IMG_DIR" \
    --save-ref-kpts eval/onnx_ref_kpts.npz

echo "PASS: rknn ready under out/"
echo "  next: cp out/*.rknn <deploy_dir>/   # promote to your runtime/deployment dir"
