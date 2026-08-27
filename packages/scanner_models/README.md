# scanner-models

The neural half of [TheScanner](https://github.com/itspiku/thescanner).

One model, three heads:

```
RGB 48x192 --> conv stem --> 48 x 256 --> BiLSTM --> 48 x 256
                                                       |
                     +---------------------------------+------------------+
               CTC head (71)                  colour head (7)     quality head (1)
              per-timestep glyphs           whole-plate colour    crop reliability
```

**1.93 M parameters.** Measured at batch 64 on an RTX 4050: 11.2 ms per forward
pass (~5,700 crops/second) in 518 MB of VRAM.

## Why one model instead of three

All three tasks read the same pixels, so sharing the trunk costs almost nothing
and saves two-thirds of the edge memory budget and two of the three forward
passes.

They also reinforce each other. Predicting plate colour forces the trunk to keep
chromatic information a pure-OCR objective would discard in its first layer —
and that colour output is exactly what feeds the decoding prior in
`nepal_plate.decode`. A greyscale-first recogniser throws away the signal the
decoder most depends on.

The quality head trains against the degradation severity `synthplate` records
for every sample: free, perfectly calibrated supervision that would otherwise
need human annotation. Its output drives frame weighting in `nepal_plate.fuse`.

## Two-row plates

CTC reads a horizontal sequence, but Nepali motorcycle plates stack
`zone lot class` above `serial`. Flattening them presents two overlaid streams
that no amount of training can disentangle, so `preprocess.py` **unwraps** them —
splitting at the row boundary and laying the halves side by side.

Deciding one row from two needs no classifier: single-row plates run 3.9–4.8 in
aspect ratio and two-row plates 1.55–1.95, with a wide empty gap between them.
A threshold in that gap settles it.

## Usage

```bash
pip install -e "packages/scanner_models[dev]"
```

```bash
python -m scanner_models.train --data data/synth --epochs 12 --out models/platenet
```

```bash
python -m scanner_models.evaluate --data data/synth --checkpoint models/platenet/best.pt --fuse
```

```bash
python -m scanner_models.export --checkpoint models/platenet/best.pt --out models/platenet.onnx
```

At runtime always go through `PlateReader`, so preprocessing and decoding cannot
drift between training and deployment:

```python
from scanner_models.infer import from_onnx

reader = from_onnx("models/platenet.onnx")
fused = reader.read_track(crops)   # a read is a track estimate, not a frame guess
```

## The evaluation harness is the point

`evaluate.py` runs an ablation on one fixed model — greedy CTC, then
grammar-constrained decoding, then grammar plus the colour prior, then track
fusion — and reports every result **per stratum** (plate system, pixel width,
degradation band, layout) plus a calibration table of accuracy by confidence
band.

A single headline accuracy number hides the thing that matters most
operationally: the false-positive rate at HIGH confidence. A wrong plate
asserted confidently is what puts the wrong person in front of a magistrate.

Licence: Apache-2.0.
