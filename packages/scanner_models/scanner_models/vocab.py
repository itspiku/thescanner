"""Vocabulary and label encoding for the recogniser.

The recogniser has a **single output head over a unified 71-token vocabulary**
(CTC blank + 34 Devanagari plate tokens + 36 Latin) rather than one head per
script. That is a deliberate choice:

* One model to train, quantise, version and deploy, instead of two plus a
  router. On an edge box that difference is most of the memory budget.
* The two alphabets are disjoint, so the model learns the script boundary from
  data and cross-script confusion is essentially impossible.
* Script routing happens at *decode* time, in the grammar, where it is a lookup
  rather than a second forward pass.

Devanagari tokens are **atomic visual units**, not Unicode codepoints -- ``बा``
is one class, not ``ब`` + ``ा``. Zone codes are a closed set of fourteen and are
rendered as single visual units on the plate, so modelling them atomically
matches both the typography and the grammar the decoder will apply.
"""

from __future__ import annotations

from typing import Final, Iterable, Sequence

from nepal_plate import PlateColour, spec

#: Index 0 is the CTC blank, as ``nepal_plate.spec.UNIFIED_VOCAB`` defines it.
VOCAB: Final[tuple[str, ...]] = spec.UNIFIED_VOCAB
VOCAB_SIZE: Final[int] = len(VOCAB)
BLANK: Final[int] = 0

TOKEN_TO_INDEX: Final[dict[str, int]] = {t: i for i, t in enumerate(VOCAB)}

#: Colour-classification classes. Ordered, and this order is baked into exported
#: models -- never reorder without retraining.
COLOUR_CLASSES: Final[tuple[PlateColour, ...]] = (
    PlateColour.RED_WHITE,
    PlateColour.BLACK_WHITE,
    PlateColour.WHITE_RED,
    PlateColour.YELLOW_BLACK,
    PlateColour.GREEN_WHITE,
    PlateColour.BLUE_WHITE,
    PlateColour.WHITE_BLACK,
)
COLOUR_TO_INDEX: Final[dict[PlateColour, int]] = {
    c: i for i, c in enumerate(COLOUR_CLASSES)
}
N_COLOURS: Final[int] = len(COLOUR_CLASSES)


def encode(tokens: Sequence[str]) -> list[int]:
    """Token strings -> vocabulary indices, for a CTC target."""
    try:
        return [TOKEN_TO_INDEX[t] for t in tokens]
    except KeyError as exc:  # pragma: no cover - indicates a corpus bug
        raise ValueError(f"token {exc.args[0]!r} is not in the plate vocabulary") from exc


def decode_indices(indices: Iterable[int]) -> list[str]:
    """Vocabulary indices -> token strings, blanks dropped."""
    return [VOCAB[i] for i in indices if i != BLANK]


def collapse(indices: Sequence[int]) -> list[str]:
    """Greedy CTC collapse: drop repeats, then blanks.

    The unconstrained baseline. ``nepal_plate.decode`` does the constrained
    version; the evaluation harness reports both so the delta is visible.
    """
    out: list[str] = []
    prev = -1
    for i in indices:
        if i != prev and i != BLANK:
            out.append(VOCAB[i])
        prev = i
    return out


def colour_index(colour: PlateColour) -> int:
    """Colour -> class index, defaulting to the embossed scheme.

    Anything not in the six legacy schemes is black-on-white by definition, so
    ``WHITE_BLACK`` is the right fallback rather than an "unknown" class that
    would dilute the head.
    """
    return COLOUR_TO_INDEX.get(colour, COLOUR_TO_INDEX[PlateColour.WHITE_BLACK])


__all__ = [
    "VOCAB",
    "VOCAB_SIZE",
    "BLANK",
    "TOKEN_TO_INDEX",
    "COLOUR_CLASSES",
    "COLOUR_TO_INDEX",
    "N_COLOURS",
    "encode",
    "decode_indices",
    "collapse",
    "colour_index",
]
