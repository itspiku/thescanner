"""Choosing which crops of a track to actually recognise.

A vehicle crossing the frame yields ten to forty crops. Recognising all of them
wastes most of the edge compute budget on near-duplicates: consecutive frames
of a track are highly correlated, so the tenth frame adds far less information
than the first. Crop selection is what makes four camera streams fit on one
Jetson.

The naive policies are both wrong in the same way:

* **Last N frames** -- the vehicle is closest and largest at the end, but also
  sweeping past fastest, so those frames carry the worst motion blur and the
  most extreme yaw. Synthetic track statistics bear this out: the best look is
  usually mid-pass.
* **Top N by quality** -- picks a run of adjacent frames from the single best
  moment. Fusion gains come from *independent* looks, and adjacent frames share
  their blur and their pose, so this throws away exactly the diversity that
  makes fusion work.

So selection is greedy over quality *with a temporal-diversity penalty*: take
the best crop, then repeatedly take the best remaining crop discounted by how
close it sits to something already chosen. That yields a spread of good frames
rather than a cluster.

Quality is estimated from the image directly here -- sharpness, size, position,
detector confidence -- and is separate from the model's learned quality head.
Both exist on purpose: this one is cheap enough to run on every candidate
before deciding what to spend a forward pass on, while the learned head is more
accurate and is used afterwards to weight fusion.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .tracker import Box


@dataclass(frozen=True, slots=True)
class Candidate:
    """One crop of a tracked vehicle, with everything needed to rank it."""

    frame_index: int
    image: np.ndarray          # BGR crop of the plate region
    box: Box                   # plate box in frame coordinates
    detector_score: float
    #: Frame dimensions, for the edge-proximity penalty.
    frame_size: tuple[int, int]
    quad: tuple[tuple[float, float], ...] | None = None


@dataclass(frozen=True, slots=True)
class ScoredCandidate:
    candidate: Candidate
    quality: float
    parts: dict[str, float]


def sharpness(image: np.ndarray) -> float:
    """Variance of the Laplacian, normalised to roughly [0, 1].

    A blurred image has little high-frequency energy, so Laplacian variance is
    a decent no-reference sharpness proxy. It is scale-dependent -- a larger
    crop scores higher for the same blur -- which is why it is combined with an
    explicit size term rather than used alone.
    """
    import cv2

    if image.size == 0:
        return 0.0
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    v = float(cv2.Laplacian(grey, cv2.CV_64F).var())
    # 500 is empirically about where crops stop looking blurred; the tanh keeps
    # very sharp crops from dominating the combined score.
    return float(np.tanh(v / 500.0))


def _size_score(box: Box, min_px: float = 24.0, good_px: float = 110.0) -> float:
    """Plate width in pixels, the single strongest predictor of readability."""
    w = max(box[2] - box[0], 0.0)
    if w <= min_px:
        return 0.0
    return float(np.clip((w - min_px) / (good_px - min_px), 0.0, 1.0))


def _edge_penalty(box: Box, frame_size: tuple[int, int], margin: float = 0.04) -> float:
    """Penalise plates near the frame border.

    A partially out-of-frame plate is unreadable no matter how sharp, and a
    plate at the very edge of a wide lens carries the worst optical distortion.
    """
    w, h = frame_size
    m_x, m_y = w * margin, h * margin
    x1, y1, x2, y2 = box
    if x1 < m_x or y1 < m_y or x2 > w - m_x or y2 > h - m_y:
        return 0.55
    return 1.0


#: Below this plate width nothing is recoverable at any sharpness.
HARD_MIN_WIDTH_PX = 18.0
#: Above this, width stops being a gate and becomes a graded preference.
LEGIBLE_WIDTH_PX = 28.0


def _legibility_gate(box: Box) -> float:
    """Hard gate on plate width, ramping from 0 to 1 across the legible floor.

    Width has to *gate* the score, not merely contribute to it. Laplacian
    variance measures high-frequency energy, and pure sensor noise is maximally
    high-frequency -- so a 12-pixel plate full of noise scores as perfectly
    sharp, and a purely weighted sum lets it through on sharpness and detector
    confidence alone. It is unreadable regardless.
    """
    w = max(box[2] - box[0], 0.0)
    if w <= HARD_MIN_WIDTH_PX:
        return 0.0
    return float(np.clip((w - HARD_MIN_WIDTH_PX) / (LEGIBLE_WIDTH_PX - HARD_MIN_WIDTH_PX), 0.0, 1.0))


def score(candidate: Candidate) -> ScoredCandidate:
    """Rank a crop for how likely it is to read correctly."""
    parts = {
        "size": _size_score(candidate.box),
        "sharpness": sharpness(candidate.image),
        "detector": float(np.clip(candidate.detector_score, 0.0, 1.0)),
        "edge": _edge_penalty(candidate.box, candidate.frame_size),
        "legible": _legibility_gate(candidate.box),
    }
    # Size and sharpness dominate the graded part. The edge and legibility terms
    # are multiplicative because both describe necessary conditions: a clipped
    # plate and a sub-20-pixel plate are unreadable no matter how good
    # everything else looks.
    base = 0.45 * parts["size"] + 0.35 * parts["sharpness"] + 0.20 * parts["detector"]
    quality = base * parts["edge"] * parts["legible"]
    return ScoredCandidate(candidate=candidate, quality=float(quality), parts=parts)


def select(
    candidates: Sequence[Candidate],
    *,
    k: int = 8,
    diversity_window: int = 6,
    diversity_penalty: float = 0.6,
    min_quality: float = 0.05,
) -> list[ScoredCandidate]:
    """Pick up to ``k`` informative, temporally spread crops.

    ``diversity_window`` is in frames: crops within this many frames of an
    already-selected one are discounted by ``diversity_penalty``. At 15 fps a
    window of 6 is about 0.4 s, which is roughly how long it takes a vehicle's
    pose and blur to decorrelate.

    Crops below ``min_quality`` are dropped entirely -- not because they carry
    no information, but because a forward pass on an unreadable crop costs the
    same as one on a readable crop, and the budget is better spent elsewhere.
    """
    scored = [s for s in (score(c) for c in candidates) if s.quality >= min_quality]
    if not scored:
        return []
    if len(scored) <= k:
        return sorted(scored, key=lambda s: s.candidate.frame_index)

    chosen: list[ScoredCandidate] = []
    remaining = list(scored)
    while remaining and len(chosen) < k:
        best: ScoredCandidate | None = None
        best_value = -1.0
        for s in remaining:
            value = s.quality
            for c in chosen:
                gap = abs(s.candidate.frame_index - c.candidate.frame_index)
                if gap < diversity_window:
                    # Linear falloff: an adjacent frame is penalised fully, one
                    # at the window edge not at all.
                    value *= 1.0 - diversity_penalty * (1.0 - gap / diversity_window)
            if value > best_value:
                best_value, best = value, s
        if best is None:
            break
        chosen.append(best)
        remaining.remove(best)

    return sorted(chosen, key=lambda s: s.candidate.frame_index)


def crop_plate(
    frame: np.ndarray, box: Box, *, pad: float = 0.06
) -> tuple[np.ndarray, Box]:
    """Cut a padded plate crop out of a frame.

    The padding matters: a crop clipped exactly to the detector's box shaves
    glyph edges, and the recogniser needs a little surrounding plate to find the
    character baseline. It also matches what the synthetic corpus looks like,
    which is rendered with the same margin.
    """
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = box
    px, py = (x2 - x1) * pad, (y2 - y1) * pad
    cx1 = int(max(0, x1 - px))
    cy1 = int(max(0, y1 - py))
    cx2 = int(min(w, x2 + px))
    cy2 = int(min(h, y2 + py))
    if cx2 - cx1 < 4 or cy2 - cy1 < 4:
        return np.zeros((0, 0, 3), dtype=frame.dtype), box
    return frame[cy1:cy2, cx1:cx2].copy(), (float(cx1), float(cy1), float(cx2), float(cy2))


__all__ = [
    "Candidate",
    "ScoredCandidate",
    "score",
    "select",
    "sharpness",
    "crop_plate",
]
