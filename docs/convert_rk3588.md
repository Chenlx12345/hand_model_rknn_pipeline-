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

The existing `.rknn` files previously deployed to the board were also
produced by toolkit 2.3.2 (verified by inspecting the RKNN file header),
so any file conversion regression chasing version drift is ruled out by
sticking with the same toolkit version.

## Why onnx must be 1.16.1

`rknn-toolkit2` (up to and including 2.3.2) imports `onnx.mapping` at the
top of several internal modules. Upstream `onnx` removed
`onnx.mapping` starting with 1.17.0, so any newer onnx breaks
`from rknn.api import RKNN` with an `ImportError`. 1.16.1 is the last
release that still ships the module.

## Calibration policy

- INT8 quantization is performed with `quantized_dtype="w8a8"`
  (set in `scripts/onnx2rknn.py`).
- The calibration set lives in `calib/images/` — 956 hand crops bundled
  in-repo. `scripts/onnx2rknn.py` (`build_calib_list`) samples `--calib-n`
  of them at conversion time with a fixed seed (default 0), writes the
  per-build dataset.txt as `_calib_*.txt` (gitignored), then feeds it to
  the toolkit.
- The calibration set is physically separate from `eval/val_images/`
  (34-image held-out evaluation set) so that changes to the eval set never
  silently shift quantization behaviour.
- Default sample count is 50; raise via `CALIB_N=...` env var or the
  `--calib-n` flag on `onnx2rknn.py` for tighter quantization at the cost
  of build time.

## RTMDet channel-order trap

The RTMDet preset in `scripts/onnx2rknn.py` sets
`quant_img_RGB2BGR=False` and feeds BGR straight into the network. The
swap path inside toolkit 2.3.2 has been observed to mis-quantize this
specific RTMDet variant, so do **not** flip the swap flag without first
re-running `scripts/bench_e2e.py --backend rknn` and confirming recall +
PCK on the held-out set are still within budget.

Any downstream runtime consuming the .rknn must mirror this: feed BGR
directly, do not pre-swap, do not pre-normalize (mean/std bake-in is
already part of the `.rknn`).

## PC-side bench vs board NPU

`scripts/bench_e2e.py --backend rknn` runs the converted .rknn through
the rknn-toolkit2 PC simulator (`init_runtime(target=None)`). This is a
software emulation of the NPU op graph:

- **Accuracy numbers are valid** — the simulator runs the same quantized
  graph the board will. Recall / PCK regressions here mean a real
  quantization problem.
- **Latency numbers are NOT** — simulator cycles are CPU-bound and have
  no relationship to on-board NPU throughput. Do not quote `E2E_FPS`
  from the simulator as a board figure.

For real board FPS you need a separate on-device benchmark; that lives
outside this submodule.

## Promoting an artefact to deployment

`out/*.rknn` is gitignored. To deploy, copy them to wherever the
downstream runtime expects them. The wiring (target directory, file
names, C++ loader paths) lives in whatever project owns the board side
and is intentionally out of scope here:

```sh
cp out/rtmdet_s_hand_640.rknn <deploy_dir>/
cp out/rtmpose_hand_256.rknn  <deploy_dir>/
```

Detector file name changes from the legacy `rtmdet_nano_hand_320.rknn`
to `rtmdet_s_hand_640.rknn`. Update every downstream consumer (loader
paths, hard-coded letterbox size 320 -> 640).
