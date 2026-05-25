# INT8 calibration set

`images/` holds the in-repo calibration set: 956 hand crops, ~200 MB,
committed so that calibration is reproducible from this repo alone with no
external dataset.

It is deliberately physically separated from the held-out evaluation set
(`../eval/val_images/`, 34 images) so that changes to the evaluation set
never silently shift quantization behaviour.

`scripts/onnx2rknn.py` (function `build_calib_list`) samples `--calib-n`
images out of `images/` at conversion time with a fixed seed, and writes
the per-build dataset.txt as `_calib_*.txt` (gitignored). No separate
prep step is needed — just point `--calib-dir` at `calib/images`.
