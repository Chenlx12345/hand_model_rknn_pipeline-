"""ONNX -> RKNN for RK3588.

Presets:
  rtmpose   mean=123.675,116.28,103.53   std=58.395,57.12,57.375   RGB
  rtmdet    mean=103.53,116.28,123.675   std=57.375,57.12,58.395   BGR

  python scripts/onnx2rknn.py --model rtmdet --onnx onnx/rtmdet_s_hand_640.onnx \\
      --out out/rtmdet_s_hand_640.rknn --quantize --calib-dir datasets/calib/images
"""
from __future__ import annotations
import argparse
import os
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


def build_calib_list(calib_dir: Path, out_txt: Path) -> int:
    """Write every image under calib_dir (sorted, recursive) to out_txt."""
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    images = sorted(p for p in calib_dir.rglob("*") if p.suffix.lower() in exts)
    if not images:
        raise FileNotFoundError(f"No images found under {calib_dir}")
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
    ap.add_argument("--calib-dir", type=Path,
                    help="root of calibration images (entire tree is used)")
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

    # absolute paths so chdir below doesn't break load_onnx/dataset/export
    onnx_abs      = args.onnx.resolve()
    out_abs       = args.out.resolve()
    out_dir       = out_abs.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    calib_dir_abs = args.calib_dir.resolve() if args.calib_dir else None

    orig_cwd = Path.cwd()
    os.chdir(out_dir)
    try:
        rknn = RKNN(verbose=True)
        print(f"[1/4] config (target={args.target})")
        rknn.config(
            mean_values=[mean],
            std_values=[std],
            target_platform=args.target,
            quantized_dtype="w8a8",
            quant_img_RGB2BGR=rgb_to_bgr,
        )

        print(f"[2/4] load_onnx: {onnx_abs}")
        if rknn.load_onnx(model=str(onnx_abs)) != 0:
            sys.exit("load_onnx failed")

        if args.quantize:
            if not calib_dir_abs:
                sys.exit("--quantize requires --calib-dir")
            calib_txt = (out_dir / f"_calib_{out_abs.stem}.txt").resolve()
            n = build_calib_list(calib_dir_abs, calib_txt)
            print(f"[3/4] build INT8  (calib n={n}, full tree)")
            ret = rknn.build(do_quantization=True, dataset=str(calib_txt))
        else:
            print(f"[3/4] build (no quantization)")
            ret = rknn.build(do_quantization=False)
        if ret != 0:
            sys.exit("rknn.build failed")

        print(f"[4/4] export_rknn -> {out_abs}")
        if rknn.export_rknn(str(out_abs)) != 0:
            sys.exit("export_rknn failed")
        sz = out_abs.stat().st_size / 1024 / 1024
        print(f"saved {out_abs}  ({sz:.1f} MB)")
        rknn.release()
    finally:
        os.chdir(orig_cwd)


if __name__ == "__main__":
    main()
