"""Shared post-processing for the RTMDet+RTMPose hand pipeline."""
from __future__ import annotations
import numpy as np
import cv2


# ─── RTMDet decode ─────────────────────────────────────────────────────────

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def nms_np(boxes, scores, iou_thr):
    if len(boxes) == 0:
        return np.array([], dtype=np.int64)
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = np.argsort(-scores)
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0., xx2 - xx1); h = np.maximum(0., yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        idx = np.where(iou <= iou_thr)[0]
        order = order[idx + 1]
    return np.array(keep, dtype=np.int64)


def decode_rtmdet(cls_outs, reg_outs, strides=(8, 16, 32),
                  score_thr=0.4, nms_thr=0.6, top_k=100):
    """Anchor-free decode for RTMDet (single class).
    cls_outs: list of 3 arrays shape (1, 1, H, W) — sigmoid logits
    reg_outs: list of 3 arrays shape (1, 4, H, W) — pixel offsets (l,t,r,b)
              already multiplied by stride.
    Returns boxes (N,4) xyxy in input-image space, scores (N,).
    """
    all_b, all_s = [], []
    for cls, reg, stride in zip(cls_outs, reg_outs, strides):
        _, _, H, W = cls.shape
        scr = sigmoid(cls[0, 0])
        mask = scr > score_thr
        if not mask.any():
            continue
        ys, xs = np.where(mask)
        cx = (xs + 0.5) * stride
        cy = (ys + 0.5) * stride
        l = reg[0, 0, ys, xs]
        t = reg[0, 1, ys, xs]
        r = reg[0, 2, ys, xs]
        b = reg[0, 3, ys, xs]
        boxes = np.stack([cx - l, cy - t, cx + r, cy + b], axis=-1)
        all_b.append(boxes)
        all_s.append(scr[ys, xs])
    if not all_b:
        return np.empty((0, 4)), np.empty((0,))
    boxes = np.concatenate(all_b).astype(np.float32)
    scores = np.concatenate(all_s).astype(np.float32)
    keep = nms_np(boxes, scores, nms_thr)[:top_k]
    return boxes[keep], scores[keep]


# ─── letterbox (RTMDet test_pipeline equivalent) ───────────────────────────

def letterbox(img_bgr, new_size=320, pad_val=114):
    """Resize keeping aspect, pad to new_size×new_size with pad_val."""
    H, W = img_bgr.shape[:2]
    scale = new_size / max(H, W)
    nW, nH = int(round(W * scale)), int(round(H * scale))
    resized = cv2.resize(img_bgr, (nW, nH), interpolation=cv2.INTER_LINEAR)
    out = np.full((new_size, new_size, 3), pad_val, dtype=np.uint8)
    x0 = (new_size - nW) // 2
    y0 = (new_size - nH) // 2
    out[y0:y0 + nH, x0:x0 + nW] = resized
    return out, scale, x0, y0


def unletterbox_bboxes(boxes, scale, x0, y0):
    """Map (N,4) xyxy from letterboxed input space back to original image."""
    if len(boxes) == 0:
        return boxes
    out = boxes.copy().astype(np.float32)
    out[:, [0, 2]] = (out[:, [0, 2]] - x0) / scale
    out[:, [1, 3]] = (out[:, [1, 3]] - y0) / scale
    return out


# ─── RTMPose pre / post (same as standalone benchmark) ─────────────────────

INPUT_POSE = 256
SIMCC_SPLIT = 2.0


def topdown_crop_rgb(img_bgr, bbox_xyxy, pad=1.25, size=INPUT_POSE):
    """Returns (uint8 RGB NHWC patch, inverse affine matrix)."""
    x1, y1, x2, y2 = bbox_xyxy
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    w, h = x2 - x1, y2 - y1
    bw = bh = max(w, h) * pad
    src = np.array([[cx, cy], [cx, cy - bh / 2], [cx + bw / 2, cy]],
                   dtype=np.float32)
    dst = np.array([[size / 2, size / 2], [size / 2, 0], [size, size / 2]],
                   dtype=np.float32)
    M = cv2.getAffineTransform(src, dst)
    patch_bgr = cv2.warpAffine(img_bgr, M, (size, size), flags=cv2.INTER_LINEAR)
    patch_rgb = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2RGB)
    # inverse: dst -> src
    M_inv = cv2.invertAffineTransform(M)
    return patch_rgb[None], M_inv


def decode_simcc(sx, sy):
    """sx, sy: (1,21,512) raw logits. Returns (21,2) kpts in 256 input space."""
    sx, sy = sx[0], sy[0]
    kx = sx.argmax(axis=-1).astype(np.float32) / SIMCC_SPLIT
    ky = sy.argmax(axis=-1).astype(np.float32) / SIMCC_SPLIT
    return np.stack([kx, ky], axis=-1)


def affine_kpts(kpts_256, M_inv):
    """Map 21 kpts from 256-crop space back to original image space."""
    if len(kpts_256) == 0:
        return kpts_256
    ones = np.ones((len(kpts_256), 1), dtype=np.float32)
    homog = np.concatenate([kpts_256, ones], axis=-1)
    return (M_inv @ homog.T).T


# ─── GT match for accuracy ────────────────────────────────────────────────

def iou_xyxy(a, b):
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter = max(0., x2 - x1) * max(0., y2 - y1)
    aa = (a[2] - a[0]) * (a[3] - a[1])
    bb = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (aa + bb - inter + 1e-9)


def match_preds_to_gt(pred_boxes, gt_boxes, iou_thr=0.5):
    """Greedy match. Returns list of (pred_idx, gt_idx) pairs."""
    pairs = []
    used_pred = set()
    used_gt = set()
    # sort pairs by IoU desc
    cand = []
    for pi, pb in enumerate(pred_boxes):
        for gi, gb in enumerate(gt_boxes):
            i = iou_xyxy(pb, gb)
            if i >= iou_thr:
                cand.append((i, pi, gi))
    cand.sort(reverse=True)
    for i, pi, gi in cand:
        if pi in used_pred or gi in used_gt:
            continue
        used_pred.add(pi); used_gt.add(gi)
        pairs.append((pi, gi))
    return pairs
