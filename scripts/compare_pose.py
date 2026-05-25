"""Compare ONNX vs RKNN-fp16 (simulator) on properly cropped hand patches.

Crops come from val.json bboxes via a 3-point TopdownAffine warp that
mirrors mmpose's pre-processing pipeline (square pad, 1.25x context).
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import cv2

RGB_MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
RGB_STD  = np.array([58.395, 57.12, 57.375], dtype=np.float32)
SIMCC_SPLIT = 2.0
INPUT = 256


def topdown_crop_uint8_rgb(img_bgr, bbox_xywh, pad=1.25, size=INPUT):
    """Returns (1, size, size, 3) uint8 RGB cropped patch using mmpose-style
    3-point affine (square pad, 1.25x context)."""
    x, y, w, h = bbox_xywh
    cx, cy = x + w / 2, y + h / 2
    bw = bh = max(w, h) * pad
    src = np.array([[cx, cy],
                    [cx, cy - bh / 2],
                    [cx + bw / 2, cy]], dtype=np.float32)
    dst = np.array([[size / 2, size / 2],
                    [size / 2, 0],
                    [size,    size / 2]], dtype=np.float32)
    M = cv2.getAffineTransform(src, dst)
    patch_bgr = cv2.warpAffine(img_bgr, M, (size, size), flags=cv2.INTER_LINEAR)
    patch_rgb = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2RGB)
    return patch_rgb[None]


def decode_simcc(sx, sy):
    sx, sy = sx[0], sy[0]
    kx = sx.argmax(axis=-1).astype(np.float32) / SIMCC_SPLIT
    ky = sy.argmax(axis=-1).astype(np.float32) / SIMCC_SPLIT
    conf = np.minimum(sx.max(axis=-1), sy.max(axis=-1))
    return np.stack([kx, ky], axis=-1), conf


def onnx_infer(sess, patch_uint8_rgb):
    x = patch_uint8_rgb[0].astype(np.float32)
    x = (x - RGB_MEAN) / RGB_STD
    x = x.transpose(2, 0, 1)[None]
    sx, sy = sess.run(None, {"input": x})
    return decode_simcc(sx, sy)


def rknn_infer(rknn, patch_uint8_rgb):
    out = rknn.inference(inputs=[patch_uint8_rgb])
    return decode_simcc(out[0], out[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", type=Path, required=True)
    ap.add_argument("--ann",  type=Path, required=True,
                    help="COCO-style val.json with hand bboxes")
    ap.add_argument("--img-dir", type=Path, required=True)
    ap.add_argument("--n", type=int, default=30, help="limit annotations")
    ap.add_argument("--save-ref-kpts", type=Path)
    args = ap.parse_args()

    coco = json.load(open(args.ann))
    img_by_id = {im["id"]: im for im in coco["images"]}
    anns = coco["annotations"][:args.n] if args.n else coco["annotations"]

    import onnxruntime as ort
    from rknn.api import RKNN

    print("building RKNN simulator (load_onnx + build, no export)...")
    rknn = RKNN(verbose=False)
    rknn.config(
        mean_values=[[123.675, 116.28, 103.53]],
        std_values=[[58.395, 57.12, 57.375]],
        target_platform="rk3588",
        quant_img_RGB2BGR=False,
    )
    if rknn.load_onnx(model=str(args.onnx)) != 0:
        raise SystemExit("load_onnx failed")
    if rknn.build(do_quantization=False) != 0:
        raise SystemExit("build failed")
    if rknn.init_runtime() != 0:
        raise SystemExit("init_runtime simulator failed")

    sess = ort.InferenceSession(str(args.onnx), providers=["CPUExecutionProvider"])

    diffs, conf_o, conf_r, ref = [], [], [], {}
    print(f"comparing {len(anns)} hand instances")
    for i, ann in enumerate(anns):
        meta = img_by_id[ann["image_id"]]
        img_path = args.img_dir / meta["file_name"]
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        patch = topdown_crop_uint8_rgb(img, ann["bbox"])
        kpts_o, c_o = onnx_infer(sess, patch)
        kpts_r, c_r = rknn_infer(rknn, patch)
        d = np.linalg.norm(kpts_o - kpts_r, axis=-1)
        diffs.append(d); conf_o.append(c_o); conf_r.append(c_r)
        key = f"{meta['file_name']}_ann{ann['id']}"
        ref[key] = kpts_o
        print(f"  [{i+1:3d}] {meta['file_name']} ann{ann['id']}: "
              f"mean={d.mean():.2f}px  max={d.max():.2f}px  conf_o={c_o.mean():.2f} conf_r={c_r.mean():.2f}")

    rknn.release()
    if not diffs:
        raise SystemExit("no annotations evaluated")
    D = np.stack(diffs)
    print()
    print("=== overall (ONNX vs RKNN-fp16 simulator, on hand crops) ===")
    print(f"  hands evaluated: {len(diffs)}")
    print(f"  mean joint diff: {D.mean():.3f} px")
    print(f"  median:          {np.median(D):.3f} px")
    print(f"  p90 / p95 / max: "
          f"{np.percentile(D,90):.3f} / {np.percentile(D,95):.3f} / {D.max():.3f}")
    print(f"  conf mean: onnx={np.stack(conf_o).mean():.3f}  "
          f"rknn={np.stack(conf_r).mean():.3f}")
    pck = lambda th: (D <= th).mean() * 100
    print(f"  PCK@1/2/3/5 px:  {pck(1):.1f} / {pck(2):.1f} / {pck(3):.1f} / {pck(5):.1f}")

    if args.save_ref_kpts:
        np.savez(args.save_ref_kpts, **ref)
        print(f"saved ref kpts: {args.save_ref_kpts}")


if __name__ == "__main__":
    main()
