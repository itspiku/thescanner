"""``scanner_models`` -- the neural half of TheScanner.

One multi-task recogniser: a shared convolutional + recurrent trunk with three
heads -- CTC over a unified 71-token dual-script vocabulary, a 7-way plate
colour classifier, and a crop-quality regressor.

The three heads are not independent conveniences. The colour head's output
becomes the decoding prior in ``nepal_plate.decode``; the quality head's output
becomes the frame weight in ``nepal_plate.fuse``. Training them jointly with
recognition also forces the trunk to retain chromatic and degradation
information that a pure-OCR objective would discard in its first layer.

Entry points::

    python -m scanner_models.train    --data data/synth --out models/platenet
    python -m scanner_models.evaluate --data data/synth --checkpoint models/platenet/best.pt
    python -m scanner_models.export   --checkpoint models/platenet/best.pt

At runtime use ``PlateReader``. It is the only supported inference path, so
preprocessing and decoding cannot drift between training and deployment::

    from scanner_models.infer import from_onnx

    reader = from_onnx("models/platenet.onnx")
    plate = reader.read(crop)
    fused = reader.read_track(crops)   # prefer this: a read is a track estimate
"""

from __future__ import annotations

from .preprocess import INPUT_H, INPUT_W, prepare
from .vocab import COLOUR_CLASSES, VOCAB, VOCAB_SIZE

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "VOCAB",
    "VOCAB_SIZE",
    "COLOUR_CLASSES",
    "INPUT_H",
    "INPUT_W",
    "prepare",
]
