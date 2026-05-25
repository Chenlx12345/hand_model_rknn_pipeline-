"""End-to-end PC-side benchmark: rtmdet -> rtmpose, single backend.

Same image set, same code paths — switch `--backend` to compare ONNX
end-to-end against the converted RKNN end-to-end on the PC.

Outputs per run:
  - detection recall @ IoU>=0.5
  - keypoint PCK@5/10/20 px (on matched hands, in original image space)
  - per-stage latency (detector / pose-per-hand) + end-to-end ms/img + FPS

WARNING: the RKNN path uses rknn-toolkit2 *PC simulator* (target=None),
NOT the on-board NPU. Its latency is a software-emulation number and
MUST NOT be quoted as board FPS. Recall/PCK numbers, however, ARE valid:
the simulator runs the actual quantized graph.

Usage (from repo root):
  python scripts/bench_e2e.py --backend onnx \
      --det  onnx/rtmdet_s_hand_640.onnx \
      --pose onnx/rtmpose_hand_256.onnx  \
      --ann  eval/val.json --img-dir eval/val_images

  python scripts/bench_e2e.py --backend rknn \
      --det  out/rtmdet_s_hand_640.rknn \
      --pose out/rtmpose_hand_256.rknn  \
      --ann  eval/val.json --img-dir eval/val_images
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path
import numpy as np
import cv2

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from pipeline_lib import (
    decode_rtmdet, letterbox, unletterbox_bboxes,
    topdown_crop_rgb, decode_simcc, affine_kpts,
    match_preds_to_gt,
)

# RTMDet preprocess (BGR; matches onnx2rknn rtmdet preset).
DET_MEAN = np.array([103.53, 116.28, 123.675], dtype=np.float32)
DET_STD  = np.array([57.375, 57.12,  58.395 ], dtype=np.float32)
# RTMPose preprocess (RGB; matches onnx2rknn rtmpose preset).
POSE_MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
POSE_STD  = np.array([58.395,  57.12,  57.375], dtype=np.float32)


# ─── backends ─────────────────────────────────────────────────────────────

class OnnxBackend:
    """Pure ONNX runtime, CPU. Mean/std normalize in Python, NCHW float32."""
    label = "onnx"

    def __init__(self, det_path: Path, pose_path: Path):
        import onnxruntime as ort
        self.det = ort.InferenceSession(
            str(det_path), providers=["CPUExecutionProvider"])
        self.pose = ort.InferenceSession(
            str(pose_path), providers=["CPUExecutionProvider"])
        self.det_in  = self.det.get_inputs()[0].name
        self.pose_in = self.pose.get_inputs()[0].name

    def infer_det(self, lb_bgr_uint8: np.ndarray):
        x = lb_bgr_uint8.astype(np.float32)
        x = (x - DET_MEAN) / DET_STD
        x = x.transpose(2, 0, 1)[None]
        t = time.perf_counter()
        out = self.det.run(None, {self.det_in: x})
        return out[:3], out[3:], (time.perf_counter() - t) * 1000.0

    def infer_pose(self, crop_rgb_uint8_nhwc: np.ndarray):
        x = crop_rgb_uint8_nhwc[0].astype(np.float32)
        x = (x - POSE_MEAN) / POSE_STD
        x = x.transpose(2, 0, 1)[None]
        t = time.perf_counter()
        sx, sy = self.pose.run(None, {self.pose_in: x})
        return sx, sy, (time.perf_counter() - t) * 1000.0

    def release(self):
        pass


class RknnBackend:
    """rknn-toolkit2 simulator on PC. Mean/std + (optional) RGB-BGR swap
    are baked into the .rknn, so we hand uint8 NHWC straight in.

    Per onnx2rknn presets:
      - rtmdet  : quant_img_RGB2BGR=False -> feed BGR
      - rtmpose : quant_img_RGB2BGR=False -> feed RGB
    """
    label = "rknn-sim"

    def __init__(self, det_path: Path, pose_path: Path):
        from rknn.api import RKNN
        self.det  = RKNN(verbose=False)
        if self.det.load_rknn(str(det_path)) != 0:
            raise SystemExit(f"load_rknn failed: {det_path}")
        if self.det.init_runtime(target=None) != 0:
            raise SystemExit("det init_runtime simulator failed")
        self.pose = RKNN(verbose=False)
        if self.pose.load_rknn(str(pose_path)) != 0:
            raise SystemExit(f"load_rknn failed: {pose_path}")
        if self.pose.init_runtime(target=None) != 0:
            raise SystemExit("pose init_runtime simulator failed")

    def infer_det(self, lb_bgr_uint8: np.ndarray):
        t = time.perf_counter()
        out = self.det.inference(inputs=[lb_bgr_uint8[None]])
        return out[:3], out[3:], (time.perf_counter() - t) * 1000.0

    def infer_pose(self, crop_rgb_uint8_nhwc: np.ndarray):
        t = time.perf_counter()
        out = self.pose.inference(inputs=[crop_rgb_uint8_nhwc])
        return out[0], out[1], (time.perf_counter() - t) * 1000.0

    def release(self):
        self.det.release(); self.pose.release()


BACKENDS = {"onnx": OnnxBackend, "rknn": RknnBackend}


# ─── main ─────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=list(BACKENDS), required=True)
    ap.add_argument("--det",  type=Path, required=True,
                    help="detector model (onnx/*.onnx or out/*.rknn)")
    ap.add_argument("--pose", type=Path, required=True,
                    help="pose model     (onnx/*.onnx or out/*.rknn)")
    ap.add_argument("--ann",     type=Path, required=True)
    ap.add_argument("--img-dir", type=Path, required=True)
    ap.add_argument("--n",       type=int, default=0,
                    help="image cap (0 = all images with GT)")
    ap.add_argument("--warmup",  type=int, default=3)
    ap.add_argument("--det-input", type=int, default=640,
                    help="letterbox target (rtmdet_s_hand_640 -> 640)")
    ap.add_argument("--det-score-thr", type=float, default=0.4)
    ap.add_argument("--det-nms-thr",   type=float, default=0.6)
    args = ap.parse_args()

    for p in (args.det, args.pose, args.ann, args.img_dir):
        if not p.exists():
            sys.exit(f"FAIL: missing: {p}")

    print(f"[load] backend={args.backend}  det={args.det.name}  pose={args.pose.name}")
    backend = BACKENDS[args.backend](args.det, args.pose)

    coco = json.load(open(args.ann))
    img_meta = {im["id"]: im for im in coco["images"]}
    gt_by_image: dict[str, list] = {}
    for ann in coco["annotations"]:
        meta = img_meta[ann["image_id"]]
        gt_by_image.setdefault(meta["file_name"], []).append(ann)
    files = [f for f in gt_by_image if (args.img_dir / f).exists()]
    if args.n:
        files = files[:args.n]
    if not files:
        sys.exit("FAIL: no images under --img-dir match --ann")
    n_gt_total = sum(len(gt_by_image[f]) for f in files)
    print(f"[prep] {len(files)} images, {n_gt_total} GT hands")

    # Pre-letterbox once per image; we reuse the same buffer across warmup
    # and timed pass so disk I/O isn't billed against the model.
    records = []
    for f in files:
        img = cv2.imread(str(args.img_dir / f))
        if img is None:
            continue
        lb, scale, x0, y0 = letterbox(img, args.det_input)
        records.append({
            "fname": f, "img": img, "lb": lb,
            "scale": scale, "x0": x0, "y0": y0,
            "gts": gt_by_image[f],
        })

    # ── warmup on the first record ──
    r0 = records[0]
    for _ in range(args.warmup):
        cls_o, reg_o, _ = backend.infer_det(r0["lb"])
        bx, _ = decode_rtmdet(
            cls_o, reg_o,
            score_thr=args.det_score_thr, nms_thr=args.det_nms_thr)
        bx = unletterbox_bboxes(bx, r0["scale"], r0["x0"], r0["y0"])
        for b in bx[:1]:  # at least one pose call to warm pose net
            crop, _ = topdown_crop_rgb(r0["img"], b)
            backend.infer_pose(crop)

    # ── timed end-to-end pass with accuracy collection ──
    det_lats, pose_lats, e2e_lats = [], [], []
    n_pred = n_match = 0
    pck_diffs: list[float] = []
    n_hands_total = 0

    for rec in records:
        t_start = time.perf_counter()

        cls_o, reg_o, det_ms = backend.infer_det(rec["lb"])
        det_lats.append(det_ms)
        boxes_lb, _ = decode_rtmdet(
            cls_o, reg_o,
            score_thr=args.det_score_thr, nms_thr=args.det_nms_thr)
        boxes = unletterbox_bboxes(
            boxes_lb, rec["scale"], rec["x0"], rec["y0"])
        n_pred += len(boxes)

        # GT for matching + PCK
        gt_xyxy_list, gt_kpts_list = [], []
        for ann in rec["gts"]:
            x, y, w, h = ann["bbox"]
            gt_xyxy_list.append([x, y, x + w, y + h])
            gt_kpts_list.append(
                np.array(ann["keypoints"], dtype=np.float32).reshape(-1, 3))
        gt_xyxy = np.array(gt_xyxy_list, dtype=np.float32)

        # pose for every detection (real e2e workload)
        kpts_full_per_pred = {}
        for pi, box in enumerate(boxes):
            crop, M_inv = topdown_crop_rgb(rec["img"], box)
            sx, sy, pose_ms = backend.infer_pose(crop)
            pose_lats.append(pose_ms)
            n_hands_total += 1
            kpts_256, _ = decode_simcc(sx, sy)
            kpts_full_per_pred[pi] = affine_kpts(kpts_256, M_inv)

        e2e_lats.append((time.perf_counter() - t_start) * 1000.0)

        # accuracy on matched pairs only
        pairs = match_preds_to_gt(boxes, gt_xyxy, iou_thr=0.5)
        n_match += len(pairs)
        for pi, gi in pairs:
            gk = gt_kpts_list[gi]
            kf = kpts_full_per_pred[pi]
            for k in range(min(21, len(gk))):
                if gk[k, 2] > 0:  # visible
                    pck_diffs.append(float(np.linalg.norm(kf[k] - gk[k, :2])))

    backend.release()

    # ── report ──
    if not e2e_lats:
        sys.exit("FAIL: no images evaluated")
    E   = np.array(e2e_lats)
    D   = np.array(det_lats)
    P   = np.array(pose_lats) if pose_lats else np.zeros(1)
    PCK = np.array(pck_diffs) if pck_diffs else np.zeros(0)
    recall = n_match / max(n_gt_total, 1) * 100.0
    prec   = n_match / max(n_pred, 1) * 100.0
    e2e_fps = 1000.0 / E.mean()

    print()
    print(f"=== accuracy (single pass over {len(records)} images) ===")
    print(f"  GT hands       : {n_gt_total}")
    print(f"  pred hands     : {n_pred}")
    print(f"  matched IoU>=0.5: {n_match}  recall={recall:.1f}%  prec={prec:.1f}%")
    if PCK.size:
        pck = lambda th: (PCK <= th).mean() * 100.0
        print(f"  kpts evaluated : {PCK.size}")
        print(f"  mean / median  : {PCK.mean():.2f} / {np.median(PCK):.2f} px")
        print(f"  PCK@5/10/20    : {pck(5):.1f}% / {pck(10):.1f}% / {pck(20):.1f}%")
    else:
        print("  (no matched hands; PCK unavailable)")

    print()
    print(f"=== latency ({backend.label}, PC-side; not board NPU) ===")
    print(f"  det /img       : mean={D.mean():.2f}ms p50={np.percentile(D,50):.2f} "
          f"p95={np.percentile(D,95):.2f}")
    print(f"  pose/hand      : mean={P.mean():.2f}ms p50={np.percentile(P,50):.2f} "
          f"p95={np.percentile(P,95):.2f}  ({n_hands_total} hands)")
    print(f"  e2e /img       : mean={E.mean():.2f}ms p50={np.percentile(E,50):.2f} "
          f"p95={np.percentile(E,95):.2f}   E2E_FPS={e2e_fps:.2f}")

    pck5 = (PCK <= 5).mean() * 100.0 if PCK.size else float("nan")
    print()
    print(
        f"PASS: BACKEND={backend.label} DET={args.det.name} POSE={args.pose.name} "
        f"IMG={len(records)} RECALL={recall:.1f}% PCK@5={pck5:.1f}% "
        f"E2E_MEAN={E.mean():.2f}ms E2E_FPS={e2e_fps:.2f}"
    )


if __name__ == "__main__":
    main()
