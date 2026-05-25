"""Quick detector accuracy compare: ONNX vs RKNN-fp16 simulator on hand val."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import cv2

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from pipeline_lib import decode_rtmdet, letterbox, unletterbox_bboxes, iou_xyxy, match_preds_to_gt

DET_MEAN = np.array([103.53, 116.28, 123.675], dtype=np.float32)   # BGR order
DET_STD  = np.array([57.375, 57.12, 58.395], dtype=np.float32)


def onnx_infer(sess, img_bgr):
    x = img_bgr.astype(np.float32)
    x = (x - DET_MEAN) / DET_STD
    x = x.transpose(2, 0, 1)[None]
    out = sess.run(None, {"input": x})
    # ONNX outputs: cls_p3, cls_p4, cls_p5, reg_p3, reg_p4, reg_p5
    return out[:3], out[3:]


def rknn_infer(rknn, img_bgr_letterbox):
    # RKNN config does (x - mean) / std on uint8 input.
    # We exported with rgb_to_bgr=True, so feed RGB; RKNN will swap.
    img_rgb = cv2.cvtColor(img_bgr_letterbox, cv2.COLOR_BGR2RGB)
    out = rknn.inference(inputs=[img_rgb[None]])
    return out[:3], out[3:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", type=Path, required=True)
    ap.add_argument("--ann",  type=Path, required=True)
    ap.add_argument("--img-dir", type=Path, required=True)
    ap.add_argument("--n", type=int, default=50)
    args = ap.parse_args()

    import onnxruntime as ort
    from rknn.api import RKNN
    sess = ort.InferenceSession(str(args.onnx), providers=["CPUExecutionProvider"])

    rknn = RKNN(verbose=False)
    rknn.config(mean_values=[[103.53, 116.28, 123.675]],
                std_values=[[57.375, 57.12, 58.395]],
                target_platform="rk3588",
                quant_img_RGB2BGR=True)
    rknn.load_onnx(model=str(args.onnx))
    rknn.build(do_quantization=False)
    rknn.init_runtime()

    coco = json.load(open(args.ann))
    img_meta = {im["id"]: im for im in coco["images"]}
    gt_by_image = {}
    for ann in coco["annotations"]:
        meta = img_meta[ann["image_id"]]
        gt_by_image.setdefault(meta["file_name"], []).append(ann)
    files = [f for f in list(gt_by_image)[:args.n] if (args.img_dir / f).exists()]
    print(f"comparing on {len(files)} images")

    gt_n = onnx_match = rknn_match = 0
    iou_diffs = []  # for matched pairs onnx vs rknn predictions
    for f in files:
        img = cv2.imread(str(args.img_dir / f))
        lb, scale, x0, y0 = letterbox(img, 320)
        cls_o, reg_o = onnx_infer(sess, lb)
        cls_r, reg_r = rknn_infer(rknn, lb)
        b_o, _ = decode_rtmdet(cls_o, reg_o)
        b_r, _ = decode_rtmdet(cls_r, reg_r)
        b_o = unletterbox_bboxes(b_o, scale, x0, y0)
        b_r = unletterbox_bboxes(b_r, scale, x0, y0)
        gt_xyxy = np.array([[a["bbox"][0], a["bbox"][1],
                              a["bbox"][0]+a["bbox"][2], a["bbox"][1]+a["bbox"][3]]
                             for a in gt_by_image[f]], dtype=np.float32)
        gt_n += len(gt_xyxy)
        onnx_match += len(match_preds_to_gt(b_o, gt_xyxy))
        rknn_match += len(match_preds_to_gt(b_r, gt_xyxy))
        # box-level onnx vs rknn IoU
        for bo in b_o:
            best = max((iou_xyxy(bo, br) for br in b_r), default=0.0)
            iou_diffs.append(best)
    rknn.release()
    print()
    print(f"GT hands:        {gt_n}")
    print(f"ONNX recall:     {onnx_match}/{gt_n} ({onnx_match/gt_n*100:.1f}%)")
    print(f"RKNN recall:     {rknn_match}/{gt_n} ({rknn_match/gt_n*100:.1f}%)")
    if iou_diffs:
        I = np.array(iou_diffs)
        print(f"ONNX-to-RKNN bbox best-match IoU: "
              f"mean={I.mean():.3f}  median={np.median(I):.3f}  "
              f"min={I.min():.3f}")


if __name__ == "__main__":
    main()
