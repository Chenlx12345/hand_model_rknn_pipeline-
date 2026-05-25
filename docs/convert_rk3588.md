# RK3588 conversion notes

## Why rknn-toolkit2 2.3.2

Board boot log (`dmesg | grep RKNPU`):

```
[    5.434852] RKNPU fdab0000.npu: Adding to iommu group 0
[    5.436644] [drm] Initialized rknpu 0.9.8 20240828 for fdab0000.npu on minor 1
```

`rknpu 0.9.8 20240828` is the kernel driver tag. The matching user-space
runtime (`librknnrt.so`) sits on the 2.3.x line. To stay byte-compatible
with `.rknn` produced on the host, the host toolkit must therefore be
pinned to **rknn-toolkit2 2.3.2**.

The existing `external/ant_algorithm/rknn/*.rknn` files were also produced
by toolkit 2.3.2 (verified by inspecting the RKNN file header), so any
file conversion regression chasing version drift is ruled out by sticking
with the same toolkit version.

## Why onnx must be 1.16.1

`rknn-toolkit2` (up to and including 2.3.2) imports `onnx.mapping` at the
top of several internal modules. Upstream `onnx` removed
`onnx.mapping` starting with 1.17.0, so any newer onnx breaks
`from rknn.api import RKNN` with an `ImportError`. 1.16.1 is the last
release that still ships the module.

## Calibration policy

- INT8 quantization is performed with `quantized_dtype="w8a8"`
  (set in `scripts/onnx2rknn.py`).
- The calibration subset comes from `calib/images/`, populated by
  `scripts/prepare_calib.py` from `eval/val_images/`.
- `prepare_calib.py` **copies** images (does not symlink) and records the
  chosen filenames into `calib/sampled.txt`. This freezes the calibration
  population even if `eval/val_images/` grows or shrinks later.
- Default sample count is 50; raise via `CALIB_N=...` env var or the
  `--calib-n` flag on `onnx2rknn.py` for tighter quantization at the cost
  of build time.

## RTMDet channel-order trap

The RTMDet preset in `scripts/onnx2rknn.py` sets
`quant_img_RGB2BGR=False` and feeds BGR straight into the network. The
swap path inside toolkit 2.3.2 has been observed to mis-quantize this
specific RTMDet variant, so do **not** flip the swap flag without first
re-running `compare_det.py` and checking PCK/IoU on the held-out set.

The C++ inference side on the board must mirror the host: feed BGR
directly, do not pre-swap, do not pre-normalize (mean/std bake-in is
already part of the `.rknn`).

## Promoting an artefact to deployment

`out/*.rknn` is gitignored. To deploy, copy them to wherever the C++
runtime / on-board benchmark expects them (e.g. an `external/ant_algorithm/rknn/`
directory in the parent superproject, if/when that dir exists):

```sh
cp out/rtmdet_s_hand_640.rknn <deploy_dir>/
cp out/rtmpose_hand_256.rknn  <deploy_dir>/
```

Then commit the new files in the parent superproject (not in this
sub-submodule), so the board-side `serial_benchmark.py` and the C++
runtime pick them up.

The detector model changes from `rtmdet_nano_hand_320.rknn` (the legacy
on-board file) to `rtmdet_s_hand_640.rknn`. Update every consumer (Python
bench, C++ loader, any hard-coded path strings).
