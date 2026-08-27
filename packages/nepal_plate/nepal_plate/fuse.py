"""Multi-frame fusion of plate reads across a vehicle track.

Why this matters more than super-resolution
-------------------------------------------
The instinct when facing a blurred plate is to sharpen the image. But a camera
watching a road does not get *one* look at a vehicle -- it gets ten to forty as
the vehicle crosses the frame, each with independent blur, a different angle,
different motion smear and different lighting. The information needed to read
the plate is usually present across the set even when no single frame carries
it. Recent low-resolution plate-recognition benchmarks are built around exactly
this: they hand competitors a *track* of frames, not an image, because
track-level fusion is what actually moves the numbers.

So this module treats a read as an estimate over a track, and single-frame
recognition as the degenerate case. Image enhancement is still worth doing --
see ``services/edge`` -- but it is a preprocessing step, not the main lever.

Two fusion levels
-----------------
``fuse_posteriors``
    Log-linear pooling of aligned per-frame posteriors, weighted by frame
    quality. Valid when every crop is resized to the recogniser's fixed input
    width, which makes the time axes comparable. Strongest when it applies,
    because it fuses *before* any information is discarded.

``fuse_track``
    Field-level consensus across independently decoded frames. Robust to
    misalignment, and it exploits something whole-string voting cannot: the
    zone may be legible in frame 3 while the serial is only legible in frame 11.
    Voting on complete strings would discard both partial reads; voting per
    field keeps them, then re-validates the assembled plate against the grammar
    so the consensus can never be a plate that could not exist.

``fuse_track`` is the primary API; it uses ``fuse_posteriors`` internally when
the caller supplies aligned posteriors.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Mapping, Sequence

from .decode import decode
from .grammar import GRAMMARS, Grammar, grammars_for
from .parse import plate_from_tokens
from .types import (
    ColourEvidence,
    Confidence,
    ParsedPlate,
    PlateColour,
    PlateSystem,
    ReadCandidate,
)

NEG_INF = float("-inf")


@dataclass(frozen=True, slots=True)
class FrameObservation:
    """One frame's worth of evidence about a tracked vehicle's plate.

    ``quality`` is the edge agent's estimate of how much this crop should be
    trusted, in [0, 1]. It combines plate pixel width, a blur measure, the
    detector's own confidence and how far the plate is from the frame edge.
    Frames are *not* filtered by quality before fusion -- a low-quality frame
    still carries information, it just carries less of it, and the weighting
    handles that continuously.
    """

    log_probs: Sequence[Sequence[float]]
    quality: float = 1.0
    colour: ColourEvidence | None = None
    frame_index: int = 0


@dataclass(frozen=True, slots=True)
class FusedRead:
    """The consensus read for a track, with its provenance."""

    plate: ParsedPlate
    #: Fused log score. Not a probability; comparable only within a track.
    score: float
    #: Number of frames that contributed evidence.
    n_frames: int
    #: Frames whose top candidate agreed with the consensus, over ``n_frames``.
    agreement: float
    #: Per-field support: field name -> fraction of weighted evidence backing
    #: the chosen value. Exposed so an operator can see *which* part of a plate
    #: is weakly supported rather than only a single overall number.
    field_support: Mapping[str, float] = field(default_factory=dict)
    alternatives: tuple[ParsedPlate, ...] = ()


def _weight(quality: float, gamma: float) -> float:
    """Map a quality score in [0, 1] onto a non-negative fusion weight."""
    q = min(max(float(quality), 0.0), 1.0)
    return q ** gamma


def fuse_posteriors(
    frames: Sequence[FrameObservation],
    *,
    gamma: float = 2.0,
) -> list[list[float]] | None:
    """Weighted log-linear pooling of aligned per-frame posteriors.

    Returns renormalised log-probabilities of the same shape, or ``None`` when
    the frames are not alignable (differing ``T``) and so must be fused at the
    field level instead.

    Log-linear pooling (a weighted geometric mean) rather than an arithmetic
    mean is deliberate: it behaves like a product of experts, so agreement
    between frames sharpens the result, whereas averaging would merely blur
    disagreeing frames together and *increase* entropy.
    """
    if not frames:
        return None
    shape = (len(frames[0].log_probs), len(frames[0].log_probs[0]) if frames[0].log_probs else 0)
    if shape[0] == 0 or shape[1] == 0:
        return None
    for f in frames:
        if len(f.log_probs) != shape[0]:
            return None
        if any(len(row) != shape[1] for row in f.log_probs):
            return None

    weights = [_weight(f.quality, gamma) for f in frames]
    total_w = sum(weights)
    if total_w <= 0.0:
        return None

    T, V = shape
    pooled: list[list[float]] = []
    for t in range(T):
        row = [0.0] * V
        for f, w in zip(frames, weights):
            if w == 0.0:
                continue
            src = f.log_probs[t]
            for v in range(V):
                row[v] += w * src[v]
        # Renormalise back to a distribution in log space.
        row = [x / total_w for x in row]
        m = max(row)
        denom = m + math.log(sum(math.exp(x - m) for x in row))
        pooled.append([x - denom for x in row])
    return pooled


def _merge_colour(frames: Sequence[FrameObservation], gamma: float) -> ColourEvidence | None:
    """Pool per-frame colour posteriors into one track-level colour estimate.

    Colour is a property of the physical plate, so it is constant across the
    track -- which makes it exactly the kind of quantity that benefits most from
    averaging over many noisy looks.
    """
    observed = [(f, _weight(f.quality, gamma)) for f in frames if f.colour is not None]
    observed = [(f, w) for f, w in observed if w > 0.0 and f.colour.reliable]
    if not observed:
        return None
    acc: dict[PlateColour, float] = defaultdict(float)
    total = 0.0
    for f, w in observed:
        for colour, p in f.colour.posterior.items():
            acc[colour] += w * float(p)
        total += w
    if total <= 0.0:
        return None
    return ColourEvidence(
        posterior={c: v / total for c, v in acc.items()},
        reliable=True,
    )


def fuse_track(
    frames: Sequence[FrameObservation],
    *,
    gamma: float = 2.0,
    beam_width: int = 16,
    top_k: int = 3,
    colour_weight: float = 1.0,
    prefer_pooled: bool = True,
) -> FusedRead | None:
    """Produce a single consensus plate read from a track of frames.

    Algorithm:

    1. Pool the colour posteriors -- colour is constant across a track, so this
       is nearly free accuracy and it strengthens the decoding prior for every
       frame at once.
    2. If the frames are alignable and ``prefer_pooled``, pool the posteriors
       and decode once. This is the strongest path.
    3. Independently of (2), decode every frame and accumulate weighted evidence
       per ``(field, value)``.
    4. Assemble the consensus from the highest-supported value of each field and
       check it against the grammar. If it is legal, it wins; if not, fall back
       to the best whole-plate hypothesis. The fallback matters: field-level
       voting across frames can in principle assemble a combination that no
       frame ever proposed and that the grammar forbids, and silently emitting
       such a plate would be worse than emitting a slightly weaker legal one.
    """
    if not frames:
        return None

    track_colour = _merge_colour(frames, gamma)

    # --- 2. pooled-posterior decode ------------------------------------
    pooled_candidates: list[ReadCandidate] = []
    if prefer_pooled:
        pooled = fuse_posteriors(frames, gamma=gamma)
        if pooled is not None:
            pooled_candidates = decode(
                pooled,
                colour=track_colour,
                beam_width=beam_width,
                colour_weight=colour_weight,
                top_k=top_k,
            )

    # --- 3. per-frame decode and field-level accumulation ---------------
    field_evidence: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    plate_evidence: dict[str, float] = defaultdict(float)
    plate_objects: dict[str, ParsedPlate] = {}
    system_evidence: dict[PlateSystem, float] = defaultdict(float)
    total_weight = 0.0
    top_choices: list[str] = []

    for f in frames:
        w = _weight(f.quality, gamma)
        if w <= 0.0:
            continue
        cands = decode(
            f.log_probs,
            colour=track_colour or f.colour,
            beam_width=beam_width,
            colour_weight=colour_weight,
            top_k=top_k,
        )
        if not cands:
            continue
        total_weight += w
        top_choices.append(cands[0].plate.canonical)
        for rank, cand in enumerate(cands):
            p = cand.plate
            # Weight by frame quality and by the candidate's own confidence,
            # decayed by rank so a frame's second guess counts for less.
            cw = w * p.score * (0.5 ** rank)
            plate_evidence[p.canonical] += cw
            plate_objects.setdefault(p.canonical, p)
            system_evidence[p.system] += cw
            for fld in p.fields:
                field_evidence[fld.name][fld.value] += cw * max(fld.score, 1e-6)

    for cand in pooled_candidates:
        p = cand.plate
        cw = total_weight * p.score if total_weight else p.score
        plate_evidence[p.canonical] += cw
        plate_objects.setdefault(p.canonical, p)
        system_evidence[p.system] += cw
        for fld in p.fields:
            field_evidence[fld.name][fld.value] += cw * max(p.score, 1e-6)

    if not plate_evidence:
        return None

    # --- 4. assemble consensus -----------------------------------------
    system = max(system_evidence, key=lambda s: system_evidence[s])
    grammar = GRAMMARS.get(system)
    consensus: ParsedPlate | None = None
    support: dict[str, float] = {}

    if grammar is not None:
        tokens: list[str] = []
        ok = True
        for slot in grammar.slots:
            votes = field_evidence.get(slot.name)
            if not votes:
                if slot.min_repeat > 0:
                    ok = False
                    break
                continue
            value = max(votes, key=lambda v: votes[v])
            denom = sum(votes.values())
            support[slot.name] = votes[value] / denom if denom else 0.0
            tokens.extend(_split_field(grammar, slot.name, value))
        if ok and tokens and grammar.segment(tokens) is not None:
            observed_colour = track_colour.best()[0] if track_colour else PlateColour.UNKNOWN
            mean_support = sum(support.values()) / len(support) if support else 0.0
            consensus = plate_from_tokens(
                grammar,
                tokens,
                score=mean_support,
                observed_colour=observed_colour,
            )

    best_whole = max(plate_evidence, key=lambda k: plate_evidence[k])
    if consensus is None or not consensus.is_valid:
        consensus = plate_objects[best_whole]
        support = {}

    agreement = (
        sum(1 for c in top_choices if c == consensus.canonical) / len(top_choices)
        if top_choices
        else 0.0
    )

    alternatives = tuple(
        plate_objects[k]
        for k in sorted(plate_evidence, key=lambda k: plate_evidence[k], reverse=True)
        if k != consensus.canonical
    )[: max(0, top_k - 1)]

    return FusedRead(
        plate=_reband(consensus, agreement, len(top_choices)),
        score=math.log(max(plate_evidence.get(consensus.canonical, 1e-9), 1e-9)),
        n_frames=len(top_choices),
        agreement=agreement,
        field_support=support,
        alternatives=alternatives,
    )


def _split_field(grammar: Grammar, slot_name: str, value: str) -> list[str]:
    """Split a concatenated field value back into grammar tokens."""
    slot = next(s for s in grammar.slots if s.name == slot_name)
    tokens: list[str] = []
    i = 0
    by_len = sorted(slot.tokens, key=len, reverse=True)
    while i < len(value):
        for tok in by_len:
            if value.startswith(tok, i):
                tokens.append(tok)
                i += len(tok)
                break
        else:
            return []
    return tokens


def _reband(plate: ParsedPlate, agreement: float, n_frames: int) -> ParsedPlate:
    """Adjust the confidence band using cross-frame agreement.

    A plate read identically from twelve independent frames deserves more trust
    than the same plate read once, even at the same per-frame score -- and a
    plate that only two of twelve frames agreed on deserves less, however
    confident those two were. Single-frame reads are capped at MEDIUM: without
    corroboration there is no way to distinguish a clean read from a confidently
    wrong one.
    """
    import dataclasses

    if n_frames <= 1:
        band = plate.confidence
        if band is Confidence.HIGH:
            band = Confidence.MEDIUM
        return dataclasses.replace(plate, confidence=band)

    band = plate.confidence
    if agreement >= 0.8 and n_frames >= 3 and band is Confidence.MEDIUM:
        band = Confidence.HIGH
    elif agreement < 0.5 and band is Confidence.HIGH:
        band = Confidence.MEDIUM
    elif agreement < 0.35:
        band = Confidence.LOW
    return dataclasses.replace(plate, confidence=band)


__all__ = ["FrameObservation", "FusedRead", "fuse_posteriors", "fuse_track"]
