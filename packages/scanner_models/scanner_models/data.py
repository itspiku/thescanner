"""Dataset over a ``synthplate`` corpus.

Reads the JSONL index written by ``synthplate.cli`` and yields the three
supervision signals the model needs: the CTC token sequence, the colour class,
and the ground-truth degradation quality.

Two things worth noting.

**The layout flag comes from the label, not from a guess.** The corpus records
whether each plate is one row or two, so training uses the true value rather
than the aspect-ratio heuristic. Inference has to use the heuristic, and the
gap between them is a real source of train/test skew -- so
:class:`PlateDataset` can be told to use the heuristic instead, and the
evaluation harness does exactly that. Measuring with the same signal you will
have at inference time is the only honest way to do it.

**Augmentation here is deliberately light.** The corpus is already degraded by a
physically-ordered pipeline that models the real imaging chain. Piling generic
augmentations on top would stack artefacts no camera produces, which is a subtle
way of teaching the model something false. Only geometry-preserving photometric
jitter is applied, to cover sensor variation the generator does not model.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from nepal_plate import PlateColour

from .preprocess import prepare
from .vocab import colour_index, encode


@dataclass(frozen=True)
class Sample:
    """One row of the corpus index, parsed."""

    path: str
    tokens: tuple[str, ...]
    canonical: str
    colour: PlateColour
    quality: float
    two_row: bool
    system: str
    width_px: int
    track_id: str | None = None
    frame: int | None = None


def read_index(root: Path | str, *, limit: int | None = None) -> list[Sample]:
    """Load a corpus index. Rows that fail to parse are skipped loudly."""
    root = Path(root)
    index = root / "index.jsonl"
    if not index.is_file():
        raise FileNotFoundError(f"no index.jsonl in {root}")

    out: list[Sample] = []
    bad = 0
    with index.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                out.append(
                    Sample(
                        path=r["path"],
                        tokens=tuple(r["tokens"]),
                        canonical=r["canonical"],
                        colour=PlateColour(r["colour"]),
                        quality=float(r["quality"]),
                        two_row=bool(r["two_row"]),
                        system=r["system"],
                        width_px=int(r["width_px"]),
                        track_id=r.get("track_id"),
                        frame=r.get("frame"),
                    )
                )
            except (KeyError, ValueError, TypeError):
                bad += 1
            if limit is not None and len(out) >= limit:
                break
    if bad:
        print(f"warning: skipped {bad} malformed rows in {index}")
    return out


class PlateDataset(Dataset):
    """Plate crops with CTC, colour and quality targets."""

    def __init__(
        self,
        root: Path | str,
        samples: Sequence[Sample] | None = None,
        *,
        limit: int | None = None,
        train: bool = True,
        use_layout_label: bool = True,
        jitter: float = 0.12,
        seed: int = 0,
    ) -> None:
        self.root = Path(root)
        self.samples = list(samples if samples is not None else read_index(self.root, limit=limit))
        self.train = train
        #: When False, the aspect-ratio heuristic decides layout, matching what
        #: inference will do. Evaluation should always set this False.
        self.use_layout_label = use_layout_label
        self.jitter = jitter
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int):
        s = self.samples[i]
        img = Image.open(self.root / s.path).convert("RGB")

        if self.train and self.jitter > 0:
            img = self._photometric(img)

        arr = prepare(img, two_row=s.two_row if self.use_layout_label else None)
        target = encode(s.tokens)

        return {
            "image": torch.from_numpy(arr),
            "target": torch.tensor(target, dtype=torch.long),
            "target_len": torch.tensor(len(target), dtype=torch.long),
            "colour": torch.tensor(colour_index(s.colour), dtype=torch.long),
            "quality": torch.tensor(s.quality, dtype=torch.float32),
            "index": i,
        }

    def _photometric(self, img: Image.Image) -> Image.Image:
        """Mild brightness/contrast/saturation jitter.

        Saturation is jittered only gently: the colour head's output feeds the
        decoding prior, so desaturating the training data teaches the model to
        be unsure about exactly the signal the decoder relies on most.
        """
        from PIL import ImageEnhance

        j = self.jitter
        for enhancer, amount in (
            (ImageEnhance.Brightness, j),
            (ImageEnhance.Contrast, j),
            (ImageEnhance.Color, j * 0.5),
        ):
            f = 1.0 + self.rng.uniform(-amount, amount)
            img = enhancer(img).enhance(f)
        return img


def collate(batch: list[dict]) -> dict[str, torch.Tensor]:
    """Pack a batch. CTC targets are concatenated rather than padded, which is
    the form ``nn.CTCLoss`` wants and avoids a padding token entirely."""
    images = torch.stack([b["image"] for b in batch])
    targets = torch.cat([b["target"] for b in batch])
    target_lens = torch.stack([b["target_len"] for b in batch])
    return {
        "image": images,
        "target": targets,
        "target_len": target_lens,
        "colour": torch.stack([b["colour"] for b in batch]),
        "quality": torch.stack([b["quality"] for b in batch]),
        "index": torch.tensor([b["index"] for b in batch], dtype=torch.long),
    }


def split_samples(
    samples: Sequence[Sample], *, val_fraction: float = 0.05, seed: int = 0
) -> tuple[list[Sample], list[Sample]]:
    """Train/validation split.

    Split by **track** where tracks exist. Frames of one vehicle passage are
    near-duplicates, so splitting by frame would put near-identical images on
    both sides and report a validation score that is really a training score.
    """
    rng = random.Random(seed)
    tracked = [s for s in samples if s.track_id]
    if tracked and len(tracked) == len(samples):
        ids = sorted({s.track_id for s in samples if s.track_id})
        rng.shuffle(ids)
        n_val = max(1, int(len(ids) * val_fraction))
        val_ids = set(ids[:n_val])
        return (
            [s for s in samples if s.track_id not in val_ids],
            [s for s in samples if s.track_id in val_ids],
        )

    idx = list(range(len(samples)))
    rng.shuffle(idx)
    n_val = max(1, int(len(idx) * val_fraction))
    val = {i for i in idx[:n_val]}
    return (
        [s for i, s in enumerate(samples) if i not in val],
        [s for i, s in enumerate(samples) if i in val],
    )


def group_by_track(samples: Sequence[Sample]) -> dict[str, list[Sample]]:
    """Group corpus rows into tracks, frames in order."""
    out: dict[str, list[Sample]] = {}
    for s in samples:
        if s.track_id:
            out.setdefault(s.track_id, []).append(s)
    for v in out.values():
        v.sort(key=lambda s: s.frame if s.frame is not None else 0)
    return out


__all__ = [
    "Sample",
    "PlateDataset",
    "read_index",
    "collate",
    "split_samples",
    "group_by_track",
]
