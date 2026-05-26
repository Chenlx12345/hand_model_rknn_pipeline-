#!/bin/sh
set -e

ROOT=$(cd "$(dirname "$0")" && pwd)
cd "$ROOT"

if [ -f ".venv_rknn/bin/activate" ]; then
    . ./.venv_rknn/bin/activate
fi

EVAL_ANN=${EVAL_ANN:-datasets/eval/val.json}
EVAL_IMG_DIR=${EVAL_IMG_DIR:-datasets/eval/val_images}

echo "[1/3] convert rtmdet_s_hand_640.onnx -> rknn"
python scripts/onnx2rknn.py --model rtmdet \
    --onnx onnx/rtmdet_s_hand_640.onnx \
    --out  out/rtmdet_s_hand_640.rknn \
    --quantize --calib-dir datasets/calib/images

echo "[2/3] convert rtmpose_hand_256.onnx -> rknn"
python scripts/onnx2rknn.py --model rtmpose \
    --onnx onnx/rtmpose_hand_256.onnx \
    --out  out/rtmpose_hand_256.rknn \
    --quantize --calib-dir datasets/calib/images

echo "[3/3] end-to-end bench: ONNX vs RKNN side-by-side"
python scripts/bench_e2e.py --backend both \
    --det  onnx/rtmdet_s_hand_640.onnx \
    --pose onnx/rtmpose_hand_256.onnx \
    --det-model rtmdet --pose-model rtmpose \
    --calib-dir datasets/calib/images \
    --ann  "$EVAL_ANN" \
    --img-dir "$EVAL_IMG_DIR"

echo "PASS: rknn ready under out/"
