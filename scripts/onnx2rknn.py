"""Convert an exported ONNX to RKNN for RK3588.

Target board NPU: RKNPU 0.9.8 20240828 (librknnrt 2.3.x) -> pin
rknn-toolkit2==2.3.2 on the host. See docs/convert_rk3588.md.

Per-model defaults (--model) preset mean/std and channel order to match each
model's data preprocessor. Override individually with --mean / --std /
--rgb-to-bgr if needed.

Presets:
  rtmpose   mean=123.675,116.28,103.53   std=58.395,57.12,57.375   RGB
  rtmdet    mean=103.53,116.28,123.675   std=57.375,57.12,58.395   BGR

Recommended invocation (from repo root):
  python scripts/onnx2rknn.py --model rtmdet  --onnx onnx/rtmdet_s_hand_640.onnx \\
      --out out/rtmdet_s_hand_640.rknn --input-size 640 640 \\
      --quantize --calib-dir calib/images --calib-n 50
  python scripts/onnx2rknn.py --model rtmpose --onnx onnx/rtmpose_hand_256.onnx \\
      --out out/rtmpose_hand_256.rknn --input-size 256 256 \\
      --quantize --calib-dir calib/images --calib-n 50
"""
from __future__ import annotations
import argparse
import random
import sys
from pathlib import Path

PRESETS = {
    "rtmpose": dict(mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    rgb_to_bgr=False),
    "rtmdet":  dict(mean=[103.53, 116.28, 123.675],
                    std=[57.375, 57.12, 58.395],
                    rgb_to_bgr=False),  # feed BGR directly, swap path is buggy
}


def build_calib_list(calib_dir: Path, n: int, out_txt: Path) -> int:
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    images = sorted(p for p in calib_dir.rglob("*") if p.suffix.lower() in exts)
    if not images:
        raise FileNotFoundError(f"No images found under {calib_dir}")
    random.seed(0)
    if n and n < len(images):
        images = random.sample(images, n)
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    with open(out_txt, "w") as f:
        for p in images:
            f.write(str(p) + "\n")
    return len(images)


def _csv_floats(s: str):
    return [float(x) for x in s.split(",")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx",  type=Path, required=True)
    ap.add_argument("--out",   type=Path, required=True)
    ap.add_argument("--target", default="rk3588")
    ap.add_argument("--model", choices=list(PRESETS), default="rtmpose",
                    help="preset for mean/std/channel order")
    ap.add_argument("--mean", type=_csv_floats,
                    help="override mean (comma-separated, in model's input channel order)")
    ap.add_argument("--std",  type=_csv_floats,
                    help="override std")
    ap.add_argument("--rgb-to-bgr", action="store_true",
                    help="set if model wants BGR but you'll feed RGB at runtime")
    ap.add_argument("--no-rgb-to-bgr", action="store_true")
    ap.add_argument("--quantize", action="store_true")
    ap.add_argument("--calib-dir", type=Path)
    ap.add_argument("--calib-n",   type=int, default=50)
    ap.add_argument("--input-size", type=int, nargs=2, default=[256, 256])
    args = ap.parse_args()

    preset = PRESETS[args.model]
    mean = args.mean if args.mean is not None else preset["mean"]
    std  = args.std  if args.std  is not None else preset["std"]
    if args.rgb_to_bgr:
        rgb_to_bgr = True
    elif args.no_rgb_to_bgr:
        rgb_to_bgr = False
    else:
        rgb_to_bgr = preset["rgb_to_bgr"]
    print(f"model={args.model}  mean={mean}  std={std}  rgb_to_bgr={rgb_to_bgr}")

    try:
        from rknn.api import RKNN
    except ImportError:
        sys.stderr.write("rknn-toolkit2 not importable; see notes.txt.\n")
        sys.exit(2)

    if not args.onnx.exists():
        sys.exit(f"ONNX not found: {args.onnx}")
    args.out.parent.mkdir(parents=True, exist_ok=True)

    rknn = RKNN(verbose=True)
    print(f"[1/4] config (target={args.target})")
    rknn.config(
        mean_values=[mean],
        std_values=[std],
        target_platform=args.target,
        quantized_dtype="w8a8",
        quant_img_RGB2BGR=rgb_to_bgr,
    )

    print(f"[2/4] load_onnx: {args.onnx}")
    if rknn.load_onnx(model=str(args.onnx)) != 0:
        sys.exit("load_onnx failed")

    if args.quantize:
        if not args.calib_dir:
            sys.exit("--quantize requires --calib-dir")
        calib_txt = args.out.parent / f"_calib_{args.out.stem}.txt"
        n = build_calib_list(args.calib_dir, args.calib_n, calib_txt)
        print(f"[3/4] build INT8  (calib n={n})")
        ret = rknn.build(do_quantization=True, dataset=str(calib_txt))
    else:
        print(f"[3/4] build (no quantization)")
        ret = rknn.build(do_quantization=False)
    if ret != 0:
        sys.exit("rknn.build failed")

    print(f"[4/4] export_rknn -> {args.out}")
    if rknn.export_rknn(str(args.out)) != 0:
        sys.exit("export_rknn failed")
    sz = args.out.stat().st_size / 1024 / 1024
    print(f"saved {args.out}  ({sz:.1f} MB)")
    rknn.release()


if __name__ == "__main__":
    main()
