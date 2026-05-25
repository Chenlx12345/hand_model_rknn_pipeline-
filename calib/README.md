# INT8 calibration directory

Contents under `images/` and the sibling `sampled.txt` are produced by
`scripts/prepare_calib.py` and are **not** tracked in git. They are
deliberately physically separated from the held-out evaluation set
(`../../rknn/val_images/`, owned by the parent superproject) so that
changes to the evaluation set never silently shift quantization
behaviour.

`source_pool/` is the in-repo fixed-population calibration source
(956 hand crops, ~200 MB) — committed so that calibration is
reproducible from this repo alone, with no external dataset needed.

Re-populate with:

```sh
python ../scripts/prepare_calib.py \
    --src ./source_pool \
    --dst images \
    --n 50 --seed 0 --clear
```

`sampled.txt` records the filenames actually used during the last
calibration build, enabling exact-reproducible quantization across
machines / time.
