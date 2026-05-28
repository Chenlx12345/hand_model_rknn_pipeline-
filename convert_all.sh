#!/bin/sh
set -e

ROOT=$(cd "$(dirname "$0")" && pwd)
cd "$ROOT"

if [ -f ".venv_rknn/bin/activate" ]; then
    . ./.venv_rknn/bin/activate
fi

EVAL_ANN=${EVAL_ANN:-datasets/eval/val.json}
EVAL_IMG_DIR=${EVAL_IMG_DIR:-datasets/eval/val_images}

mkdir -p assets/models

echo "[1/3] convert rtmdet.onnx -> rknn"
python scripts/onnx2rknn.py --model rtmdet \
    --onnx onnx/rtmdet.onnx \
    --out  assets/models/rtmdet.rknn \
    --quantize --calib-dir datasets/calib/images

echo "[2/3] convert rtmpose.onnx -> rknn"
python scripts/onnx2rknn.py --model rtmpose \
    --onnx onnx/rtmpose.onnx \
    --out  assets/models/rtmpose.rknn \
    --quantize --calib-dir datasets/calib/images

echo "[3/3] end-to-end bench: ONNX vs RKNN side-by-side"
python scripts/bench_e2e.py --backend both \
    --det  onnx/rtmdet.onnx \
    --pose onnx/rtmpose.onnx \
    --det-model rtmdet --pose-model rtmpose \
    --calib-dir datasets/calib/images \
    --ann  "$EVAL_ANN" \
    --img-dir "$EVAL_IMG_DIR"

echo "PASS: rknn ready under assets/models/"
