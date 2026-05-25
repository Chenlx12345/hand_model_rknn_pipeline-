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

# Eval data (val.json + val_images) is owned by the parent superproject at
# ../rknn/, so reference it there instead of duplicating into this submodule.
EVAL_ANN=${EVAL_ANN:-../rknn/val.json}
EVAL_IMG_DIR=${EVAL_IMG_DIR:-../rknn/val_images}

echo "[1/5] prepare calibration subset (n=$CALIB_N from calib/source_pool)"
python scripts/prepare_calib.py --src calib/source_pool --dst calib/images --n "$CALIB_N"

echo "[2/5] convert rtmdet_s_hand_640.onnx -> rknn"
python scripts/onnx2rknn.py --model rtmdet \
    --onnx onnx/rtmdet_s_hand_640.onnx \
    --out  out/rtmdet_s_hand_640.rknn \
    --input-size 640 640 \
    --quantize --calib-dir calib/images --calib-n "$CALIB_N"

echo "[3/5] convert rtmpose_hand_256.onnx -> rknn"
python scripts/onnx2rknn.py --model rtmpose \
    --onnx onnx/rtmpose_hand_256.onnx \
    --out  out/rtmpose_hand_256.rknn \
    --input-size 256 256 \
    --quantize --calib-dir calib/images --calib-n "$CALIB_N"

echo "[4/5] accuracy compare: detector (ONNX vs RKNN-fp16 simulator)"
python scripts/compare_det.py \
    --onnx onnx/rtmdet_s_hand_640.onnx \
    --ann  "$EVAL_ANN" \
    --img-dir "$EVAL_IMG_DIR"

echo "[5/5] accuracy compare: pose (ONNX vs RKNN-fp16 simulator)"
python scripts/compare_pose.py \
    --onnx onnx/rtmpose_hand_256.onnx \
    --ann  "$EVAL_ANN" \
    --img-dir "$EVAL_IMG_DIR" \
    --save-ref-kpts eval/onnx_ref_kpts.npz

echo "PASS: rknn ready under out/"
echo "  next: cp out/*.rknn ../rknn/   # promote to deployment dir"
