"""Sample a fixed subset of images into calib/images/ for INT8 quantization.

Calibration data must live in its own directory, physically separate from the
evaluation set, so that later changes to eval/val_images/ never silently shift
the calibration result. The chosen filenames are also recorded into
calib/sampled.txt for reproducibility.

Usage (from repo root):
  python scripts/prepare_calib.py --src calib/source_pool --dst calib/images --n 50
"""
from __future__ import annotations
import argparse
import random
import shutil
import sys
from pathlib import Path

EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def list_images(src: Path) -> list[Path]:
    return sorted(p for p in src.rglob("*") if p.suffix.lower() in EXTS)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True,
                    help="source image directory (will be recursed)")
    ap.add_argument("--dst", type=Path, required=True,
                    help="destination directory for the calibration subset")
    ap.add_argument("--n", type=int, default=50,
                    help="number of images to sample (0 = all)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--clear", action="store_true",
                    help="wipe --dst before sampling")
    args = ap.parse_args()

    if not args.src.is_dir():
        sys.exit(f"src not a directory: {args.src}")

    images = list_images(args.src)
    if not images:
        sys.exit(f"no images under {args.src}")

    random.seed(args.seed)
    if args.n and args.n < len(images):
        images = random.sample(images, args.n)

    if args.clear and args.dst.exists():
        for p in args.dst.iterdir():
            if p.is_file():
                p.unlink()
    args.dst.mkdir(parents=True, exist_ok=True)

    manifest = args.dst.parent / "sampled.txt"
    with open(manifest, "w") as f:
        for src_path in images:
            dst_path = args.dst / src_path.name
            shutil.copy2(src_path, dst_path)
            f.write(f"{src_path.name}\n")

    print(f"copied {len(images)} images -> {args.dst}")
    print(f"manifest -> {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
