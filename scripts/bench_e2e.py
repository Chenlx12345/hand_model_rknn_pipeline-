"""End-to-end PC bench: RTMDet -> RTMPose on ONNX runtime or RKNN toolkit2 simulator.

See README for CLI. Toolkit 2.3.2 simulator rebuilds INT8 from ONNX every run
(init_runtime(target=None) rejects load_rknn outputs), so RKNN latency here is
not representative of board NPU — only recall/PCK are."""
from __future__ import annotations
import argparse
import json
import os
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
from viz_lib import draw_hands
# Imported from onnx2rknn so the in-memory rebuild matches the on-disk .rknn.
from onnx2rknn import PRESETS, build_calib_list

DET_MEAN = np.array([103.53, 116.28, 123.675], dtype=np.float32)
DET_STD  = np.array([57.375, 57.12,  58.395 ], dtype=np.float32)
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
    """rknn-toolkit2 simulator on PC. Rebuilds the INT8 graph from ONNX every run
    because toolkit 2.3.2's init_runtime(target=None) rejects load_rknn outputs."""
    label = "rknn-sim"

    def __init__(self, det_onnx: Path, pose_onnx: Path,
                 det_preset: dict, pose_preset: dict,
                 calib_dir: Path,
                 target: str, scratch_dir: Path):
        self.det  = self._build(det_onnx,  det_preset,
                                calib_dir, target, scratch_dir,
                                tag="det")
        self.pose = self._build(pose_onnx, pose_preset,
                                calib_dir, target, scratch_dir,
                                tag="pose")

    @staticmethod
    def _build(onnx_path: Path, preset: dict,
               calib_dir: Path,
               target: str, scratch_dir: Path, tag: str):
        # chdir into scratch_dir so toolkit's hardcoded cwd dumps
        # (check0_base_optimize.onnx / check3_fuse_ops.onnx) land in
        # out/ rather than polluting the repo root. Same trick as
        # onnx2rknn.py — there is no API knob for this in 2.3.2.
        from rknn.api import RKNN
        onnx_abs = onnx_path.resolve()
        calib_txt = (scratch_dir /
                     f"_calib_bench_{onnx_path.stem}.txt").resolve()
        n = build_calib_list(calib_dir, calib_txt)
        print(f"[{tag}] rebuild INT8 from {onnx_path.name} "
              f"(calib n={n}, full tree from {calib_dir})")
        orig_cwd = Path.cwd()
        os.chdir(scratch_dir)
        try:
            rk = RKNN(verbose=False)
            rk.config(
                mean_values=[preset["mean"]],
                std_values=[preset["std"]],
                target_platform=target,
                quantized_dtype="w8a8",
                quant_img_RGB2BGR=preset["rgb_to_bgr"],
            )
            if rk.load_onnx(model=str(onnx_abs)) != 0:
                raise SystemExit(f"load_onnx failed: {onnx_abs}")
            if rk.build(do_quantization=True, dataset=str(calib_txt)) != 0:
                raise SystemExit(f"rknn.build failed: {onnx_abs}")
            if rk.init_runtime(target=None) != 0:
                raise SystemExit(f"{tag} init_runtime simulator failed")
        finally:
            os.chdir(orig_cwd)
        return rk

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


BACKENDS = ("onnx", "rknn", "both")


# ─── per-image inference helper ───────────────────────────────────────────

def run_one(backend, rec, score_thr: float, nms_thr: float):
    """Run det + pose for every detection on a single pre-letterboxed record.

    Returns:
        boxes_full      : (N,4) xyxy in original image coords
        scores          : (N,) detector confidences, aligned with boxes
        kpts_full       : dict[pi] -> (21,2) keypoints in original image coords
        det_ms          : float, detector latency
        pose_ms_list    : list[float], one entry per detected hand
    """
    cls_o, reg_o, det_ms = backend.infer_det(rec["lb"])
    boxes_lb, scores = decode_rtmdet(
        cls_o, reg_o, score_thr=score_thr, nms_thr=nms_thr)
    boxes_full = unletterbox_bboxes(
        boxes_lb, rec["scale"], rec["x0"], rec["y0"])
    kpts_full: dict[int, np.ndarray] = {}
    pose_ms_list: list[float] = []
    for pi, box in enumerate(boxes_full):
        crop, M_inv = topdown_crop_rgb(rec["img"], box)
        sx, sy, pose_ms = backend.infer_pose(crop)
        pose_ms_list.append(pose_ms)
        kpts_256 = decode_simcc(sx, sy)
        kpts_full[pi] = affine_kpts(kpts_256, M_inv)
    return boxes_full, scores, kpts_full, det_ms, pose_ms_list


# ─── `both` comparison reporter ───────────────────────────────────────────

def _new_stats():
    # Per-backend accumulators; mirrors the single-backend timed pass
    # at the bottom of main() so columns are computed identically.
    return {"det": [], "pose": [], "e2e": [], "pck": [],
            "n_pred": 0, "n_match": 0}


def _aggregate(backend, rec, gt_xyxy, gt_kpts_list, s,
               score_thr, nms_thr, viz_label: str | None = None):
    """Run one backend over one image; fold latencies + matches + kpt
    errors into stats dict `s`. Returns the annotated viz image (BGR) so
    `_run_compare` can hstack ONNX|RKNN side-by-side."""
    t0 = time.perf_counter()
    boxes, scores, kpts, det_ms, pose_ms_list = run_one(
        backend, rec, score_thr, nms_thr)
    e2e_ms = (time.perf_counter() - t0) * 1000.0

    s["det"].append(det_ms)
    s["pose"].extend(pose_ms_list)
    s["e2e"].append(e2e_ms)
    s["n_pred"] += len(boxes)

    for pi, gi in match_preds_to_gt(boxes, gt_xyxy, 0.5):
        s["n_match"] += 1
        gk = gt_kpts_list[gi]
        kf = kpts[pi]
        for k in range(min(21, len(gk))):
            if gk[k, 2] > 0:  # visible
                s["pck"].append(float(np.linalg.norm(kf[k] - gk[k, :2])))

    return draw_hands(rec["img"], boxes, kpts,
                      scores=scores, label=viz_label)


def _print_summary(stats, n_gt_total, n_images):
    print()
    print(f"=== bench_e2e: ONNX vs RKNN ({n_images} images) ===")
    print(f"| {'backend':<7} | {'recall':>6} | {'PCK@5':>6} | {'PCK@10':>6} | "
          f"{'mean_err':>8} | {'det_ms':>7} | {'pose_ms':>7} | "
          f"{'e2e_ms':>6} | {'FPS':>5} |")
    print(f"|{'-'*9}|{'-'*8}|{'-'*8}|{'-'*8}|{'-'*10}|"
          f"{'-'*9}|{'-'*9}|{'-'*8}|{'-'*7}|")
    for label, s in stats.items():
        E = np.array(s["e2e"])
        D = np.array(s["det"])
        P = np.array(s["pose"]) if s["pose"] else np.zeros(1)
        PCK = np.array(s["pck"]) if s["pck"] else np.zeros(0)
        recall = s["n_match"] / max(n_gt_total, 1) * 100.0
        pck5 = (PCK <= 5).mean() * 100.0 if PCK.size else float("nan")
        pck10 = (PCK <= 10).mean() * 100.0 if PCK.size else float("nan")
        mean_err = PCK.mean() if PCK.size else float("nan")
        fps = 1000.0 / E.mean() if E.size else float("nan")
        print(
            f"| {label:<7} | {recall:5.1f}% | {pck5:5.1f}% | {pck10:5.1f}% | "
            f"{mean_err:5.2f} px | {D.mean():7.2f} | {P.mean():7.2f} | "
            f"{E.mean():6.2f} | {fps:5.2f} |"
        )


def _run_compare(b_onnx, b_rknn, records, *,
                 score_thr: float, nms_thr: float, warmup: int,
                 viz_dir):
    # warmup both — first record only, mirrors single-backend path
    r0 = records[0]
    for _ in range(warmup):
        for bk in (b_onnx, b_rknn):
            cls_o, reg_o, _ = bk.infer_det(r0["lb"])
            bx, _ = decode_rtmdet(cls_o, reg_o,
                                  score_thr=score_thr, nms_thr=nms_thr)
            bx = unletterbox_bboxes(bx, r0["scale"], r0["x0"], r0["y0"])
            for b in bx[:1]:
                crop, _ = topdown_crop_rgb(r0["img"], b)
                bk.infer_pose(crop)

    stats = {"onnx": _new_stats(), "rknn": _new_stats()}
    n_gt_total = 0
    for rec in records:
        gt_xyxy_list, gt_kpts_list = [], []
        for ann in rec["gts"]:
            x, y, w, h = ann["bbox"]
            gt_xyxy_list.append([x, y, x + w, y + h])
            gt_kpts_list.append(
                np.array(ann["keypoints"], dtype=np.float32).reshape(-1, 3))
        gt_xyxy = np.array(gt_xyxy_list, dtype=np.float32)
        n_gt_total += len(gt_kpts_list)

        vis_onnx = _aggregate(b_onnx, rec, gt_xyxy, gt_kpts_list,
                              stats["onnx"], score_thr, nms_thr,
                              viz_label="ONNX")
        vis_rknn = _aggregate(b_rknn, rec, gt_xyxy, gt_kpts_list,
                              stats["rknn"], score_thr, nms_thr,
                              viz_label="RKNN")
        # Same source `rec["img"]` -> identical H/W; hstack safe.
        cv2.imwrite(str(viz_dir / rec["fname"]),
                    np.hstack([vis_onnx, vis_rknn]))

    b_onnx.release(); b_rknn.release()
    _print_summary(stats, n_gt_total, len(records))


# ─── main ─────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=BACKENDS, required=True)
    ap.add_argument("--det",  type=Path, required=True,
                    help="detector ONNX path (both backends — RKNN backend "
                         "rebuilds INT8 from ONNX, see module docstring)")
    ap.add_argument("--pose", type=Path, required=True,
                    help="pose ONNX path (both backends)")
    ap.add_argument("--ann",     type=Path, required=True)
    ap.add_argument("--img-dir", type=Path, required=True)
    ap.add_argument("--n",       type=int, default=0,
                    help="image cap (0 = all images with GT)")
    ap.add_argument("--warmup",  type=int, default=3)
    ap.add_argument("--det-input", type=int, default=640,
                    help="letterbox target (rtmdet_s_hand_640 -> 640)")
    ap.add_argument("--det-score-thr", type=float, default=0.4)
    ap.add_argument("--det-nms-thr",   type=float, default=0.6)
    # rknn-only
    ap.add_argument("--det-model",  choices=list(PRESETS), default="rtmdet",
                    help="rknn-only: preset key for detector")
    ap.add_argument("--pose-model", choices=list(PRESETS), default="rtmpose",
                    help="rknn-only: preset key for pose")
    ap.add_argument("--calib-dir", type=Path,
                    help="rknn-only: calibration image root (entire tree "
                         "is used, must match what onnx2rknn used)")
    ap.add_argument("--target",    default="rk3588",
                    help="rknn-only: target_platform for quant config")
    args = ap.parse_args()

    for p in (args.det, args.pose, args.ann, args.img_dir):
        if not p.exists():
            sys.exit(f"FAIL: missing: {p}")

    print(f"[load] backend={args.backend}  det={args.det.name}  pose={args.pose.name}")

    def _make_rknn():
        # RKNN backend constructor — also used by `both`. Validates calib args
        # and pins toolkit's cwd-relative dump artefacts under out/ regardless
        # of where the user invoked the script from.
        if args.calib_dir is None:
            sys.exit("FAIL: --backend rknn/both requires --calib-dir")
        if not args.calib_dir.exists():
            sys.exit(f"FAIL: missing: {args.calib_dir}")
        scratch = (HERE.parent / "out").resolve()
        scratch.mkdir(parents=True, exist_ok=True)
        return RknnBackend(
            det_onnx=args.det, pose_onnx=args.pose,
            det_preset=PRESETS[args.det_model],
            pose_preset=PRESETS[args.pose_model],
            calib_dir=args.calib_dir.resolve(),
            target=args.target,
            scratch_dir=scratch,
        )

    if args.backend == "onnx":
        backend = OnnxBackend(args.det, args.pose)
    elif args.backend == "rknn":
        backend = _make_rknn()
    else:  # both — build ONNX then RKNN so log order matches diff direction
        backend_onnx = OnnxBackend(args.det, args.pose)
        backend_rknn = _make_rknn()
        backend = None

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

    # Visualization sink — always on. Lives next to .rknn / calib txt but
    # in its own subdir so it doesn't get mixed up with model artifacts.
    viz_dir = (HERE.parent / "out" / "viz").resolve()
    viz_dir.mkdir(parents=True, exist_ok=True)

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

    # ── `both` mode: per-image ONNX vs RKNN comparison table, no aggregate ──
    if args.backend == "both":
        _run_compare(
            backend_onnx, backend_rknn, records,
            score_thr=args.det_score_thr, nms_thr=args.det_nms_thr,
            warmup=args.warmup, viz_dir=viz_dir,
        )
        return

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
        boxes_lb, scores = decode_rtmdet(
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
            kpts_256 = decode_simcc(sx, sy)
            kpts_full_per_pred[pi] = affine_kpts(kpts_256, M_inv)

        e2e_lats.append((time.perf_counter() - t_start) * 1000.0)

        # Draw bbox + 21 kpts + skeleton onto original image, dump to
        # out/viz/. Done AFTER e2e timer to keep latency numbers clean.
        cv2.imwrite(
            str(viz_dir / rec["fname"]),
            draw_hands(rec["img"], boxes, kpts_full_per_pred,
                       scores=scores, label=backend.label.upper()),
        )

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
