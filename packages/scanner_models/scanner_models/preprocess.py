"""Turning a plate crop into a recogniser input.

The interesting problem here is **two-row plates**.

CTC reads a horizontal sequence. A Nepali motorcycle plate stacks
``zone lot class`` above ``serial``, so squashing it into one strip presents the
model with two independent left-to-right streams overlaid on each other. No
amount of training fixes that -- the information is genuinely ambiguous once
flattened, because CTC has no way to express "read the top line, then the
bottom".

The standard fix is to *unwrap*: split the crop horizontally at the row boundary
and lay the two halves side by side, producing one true left-to-right sequence
that matches the grammar's token order.

That requires knowing whether a plate is one row or two, and the cheapest
reliable signal turns out to be **aspect ratio**. From the published dimension
tables, single-row plates run 3.9-4.8 and two-row plates 1.55-1.95 -- the ranges
do not overlap, and there is a wide empty gap between them. So no layout
classifier is needed: a threshold in the middle of that gap decides it, using
the plate quad the detector already produces. One less model to train, quantise
and version.
"""

from __future__ import annotations

from typing import Final, Sequence

import numpy as np
from PIL import Image

#: Recogniser input size. 192 wide gives 48 CTC timesteps after the stem, which
#: is comfortably more than twice the longest plate (9 tokens) -- CTC needs
#: headroom to place blanks between repeated glyphs.
INPUT_H: Final[int] = 48
INPUT_W: Final[int] = 192

#: Aspect ratio below which a plate is treated as two-row. Sits in the empty
#: gap between the two populations (two-row tops out near 1.95, single-row
#: starts near 3.9), so it is insensitive to where exactly it is placed.
TWO_ROW_ASPECT: Final[float] = 2.7

#: ImageNet-ish normalisation. Values are conventional; what matters is that
#: training and inference agree, so they live here rather than in either.
MEAN: Final[tuple[float, float, float]] = (0.485, 0.456, 0.406)
STD: Final[tuple[float, float, float]] = (0.229, 0.224, 0.225)


def looks_two_row(width: int, height: int) -> bool:
    """Whether a crop of this shape is a two-row plate."""
    return height > 0 and (width / height) < TWO_ROW_ASPECT


def unwrap_two_row(img: Image.Image, *, split: float = 0.5) -> Image.Image:
    """Lay the two rows of a stacked plate side by side.

    ``split`` is where the row boundary sits as a fraction of height. Half is
    right for Nepali plates, where both rows carry similar glyph heights.

    A small vertical overlap is deliberately kept on each half: the boundary is
    never exactly at the midpoint on a real plate, and clipping a matra off the
    top of the second row costs far more than including a few pixels of the
    first.
    """
    w, h = img.size
    if h < 4:
        return img
    cut = int(round(h * split))
    pad = max(1, h // 16)
    top = img.crop((0, 0, w, min(h, cut + pad)))
    bottom = img.crop((0, max(0, cut - pad), w, h))

    th = max(top.height, bottom.height)
    top = top.resize((w, th), Image.Resampling.BILINEAR)
    bottom = bottom.resize((w, th), Image.Resampling.BILINEAR)

    out = Image.new("RGB", (w * 2, th))
    out.paste(top, (0, 0))
    out.paste(bottom, (w, 0))
    return out


def fit(img: Image.Image, *, two_row: bool | None = None) -> Image.Image:
    """Unwrap if needed, then resize to the recogniser input size.

    Resizing is a plain stretch rather than aspect-preserving letterbox. Plates
    have a fixed glyph count in a fixed layout, so horizontal scale carries no
    information the model needs -- and letterboxing would spend a third of the
    input on padding for two-row plates.
    """
    if img.mode != "RGB":
        img = img.convert("RGB")
    if two_row is None:
        two_row = looks_two_row(*img.size)
    if two_row:
        img = unwrap_two_row(img)
    return img.resize((INPUT_W, INPUT_H), Image.Resampling.BILINEAR)


def to_array(img: Image.Image) -> np.ndarray:
    """Normalised CHW float32 array."""
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - np.asarray(MEAN, np.float32)) / np.asarray(STD, np.float32)
    return np.ascontiguousarray(arr.transpose(2, 0, 1))


def prepare(img: Image.Image, *, two_row: bool | None = None) -> np.ndarray:
    """Crop -> model input array. The single entry point; use it everywhere.

    Training, evaluation, ONNX export and the edge agent all call this, so a
    preprocessing change can never silently desynchronise them.
    """
    return to_array(fit(img, two_row=two_row))


def quad_aspect(corners: Sequence[Sequence[float]]) -> float:
    """Aspect ratio of a plate quad, for deciding layout after a warp.

    A perspective-warped plate's bounding box is a poor guide to its true shape,
    so the detector's quad is used instead: the mean of the two long edges over
    the mean of the two short ones.
    """
    if len(corners) != 4:
        return 0.0
    pts = np.asarray(corners, dtype=float)

    def edge(a: int, b: int) -> float:
        return float(np.hypot(*(pts[a] - pts[b])))

    top, bottom = edge(0, 1), edge(2, 3)
    left, right = edge(0, 3), edge(1, 2)
    h = (left + right) / 2.0
    return ((top + bottom) / 2.0) / h if h > 1e-6 else 0.0


__all__ = [
    "INPUT_H",
    "INPUT_W",
    "TWO_ROW_ASPECT",
    "MEAN",
    "STD",
    "looks_two_row",
    "unwrap_two_row",
    "fit",
    "to_array",
    "prepare",
    "quad_aspect",
]
