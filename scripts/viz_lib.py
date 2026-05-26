"""Hand detection + 21-keypoint visualization helpers.

Pure drawing; no model/IO logic. Consumes results already mapped back to
original-image pixel coordinates (i.e. post `unletterbox_bboxes` /
`affine_kpts`). Reused by bench_e2e.py for single-backend dumps and for
left|right hstack in `--backend both`.
"""
from __future__ import annotations
import numpy as np
import cv2


# MediaPipe 21-keypoint hand topology (wrist=0, thumb 1-4, index 5-8,
# middle 9-12, ring 13-16, pinky 17-20). 20 bones total.
HAND_SKELETON: tuple[tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
)

# BGR — one color per finger; wrist shares thumb color.
_THUMB  = (0, 165, 255)   # orange
_INDEX  = (0, 255, 255)   # yellow
_MIDDLE = (0, 255, 0)     # green
_RING   = (255, 128, 0)   # cyan-ish blue
_PINKY  = (255, 0, 255)   # magenta

# index -> color, length 21
KPT_COLORS: tuple[tuple[int, int, int], ...] = (
    _THUMB,                                  # 0 wrist
    _THUMB,  _THUMB,  _THUMB,  _THUMB,       # 1-4
    _INDEX,  _INDEX,  _INDEX,  _INDEX,       # 5-8
    _MIDDLE, _MIDDLE, _MIDDLE, _MIDDLE,      # 9-12
    _RING,   _RING,   _RING,   _RING,        # 13-16
    _PINKY,  _PINKY,  _PINKY,  _PINKY,       # 17-20
)

BBOX_COLOR = (0, 255, 0)   # green


def draw_hands(
    img_bgr: np.ndarray,
    boxes: np.ndarray,
    kpts_per_pred: dict,
    *,
    scores: np.ndarray | None = None,
    label: str | None = None,
) -> np.ndarray:
    """Return a BGR copy of `img_bgr` with bboxes + 21 keypoints + skeleton.

    Args:
        img_bgr        : (H,W,3) uint8 BGR original-image frame.
        boxes          : (N,4) float xyxy in original image coords.
        kpts_per_pred  : {pi: (21,2) float} kpts in original image coords.
        scores         : (N,) optional detection confidence. If given, drawn
                         next to each bbox as 'det=0.87'.
        label          : optional top-left tag, e.g. 'ONNX' / 'RKNN'. Drawn
                         on a black filled rect for legibility under hstack.

    Always returns a NEW array (np.copy) so callers can freely hstack the
    result without aliasing back into the shared rec['img'].
    """
    vis = img_bgr.copy()

    for pi, box in enumerate(boxes):
        x1, y1, x2, y2 = (int(round(v)) for v in box[:4])
        cv2.rectangle(vis, (x1, y1), (x2, y2), BBOX_COLOR, 2)
        if scores is not None and pi < len(scores):
            txt = f"det={float(scores[pi]):.2f}"
            cv2.putText(vis, txt, (x1, max(y1 - 4, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, BBOX_COLOR, 1,
                        cv2.LINE_AA)

        kp = kpts_per_pred.get(pi)
        if kp is None:
            continue
        # skeleton first so dots overlay the lines cleanly
        for i, j in HAND_SKELETON:
            if i >= len(kp) or j >= len(kp):
                continue
            p1 = (int(round(kp[i, 0])), int(round(kp[i, 1])))
            p2 = (int(round(kp[j, 0])), int(round(kp[j, 1])))
            cv2.line(vis, p1, p2, KPT_COLORS[j], 1, cv2.LINE_AA)
        for k in range(min(21, len(kp))):
            cx, cy = int(round(kp[k, 0])), int(round(kp[k, 1]))
            cv2.circle(vis, (cx, cy), 3, KPT_COLORS[k], -1, cv2.LINE_AA)

    if label:
        # black filled rect + white text — survives white backgrounds
        font, scale_, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
        (tw, th), base = cv2.getTextSize(label, font, scale_, thick)
        pad = 4
        cv2.rectangle(vis, (0, 0), (tw + 2 * pad, th + 2 * pad + base),
                      (0, 0, 0), -1)
        cv2.putText(vis, label, (pad, th + pad), font, scale_,
                    (255, 255, 255), thick, cv2.LINE_AA)

    return vis
