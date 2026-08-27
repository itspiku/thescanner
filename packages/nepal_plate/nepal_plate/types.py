"""Core value types for Nepali number plates.

Everything in ``nepal_plate`` is deliberately dependency-free (stdlib only) so it
can be imported by the edge agent, the API service, the training pipeline and
the synthetic data generator without dragging in a deep-learning runtime.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping, Sequence


class PlateSystem(str, Enum):
    """Which of Nepal's two coexisting plate systems a plate belongs to."""

    #: Legacy zonal plates: Devanagari script, colour-coded by ownership.
    #: Issued until the embossed rollout; still the majority of vehicles on the
    #: road and legal to use, so the system must read them indefinitely.
    DEVANAGARI = "devanagari"

    #: Post-2020 embossed plates: Latin FE-Schrift, black-on-white, RFID chip.
    EMBOSSED = "embossed"

    UNKNOWN = "unknown"


class Ownership(str, Enum):
    """Ownership category. On legacy plates this is encoded *twice* -- in the
    plate colour and in the vehicle-class letter -- which lets us cross-check."""

    PRIVATE = "private"
    PUBLIC = "public"
    GOVERNMENT = "government"
    CORPORATION = "corporation"
    TOURIST = "tourist"
    DIPLOMATIC = "diplomatic"
    UNKNOWN = "unknown"


class SizeClass(str, Enum):
    """Coarse vehicle size band encoded by the legacy class letter."""

    HEAVY = "heavy"
    LIGHT = "light"
    MOTORCYCLE = "motorcycle"
    ANY = "any"
    UNKNOWN = "unknown"


class PlateColour(str, Enum):
    """Background/foreground colour scheme of the physical plate.

    Named ``<BACKGROUND>_<TEXT>``. Legacy plates use six schemes that map 1:1 to
    ownership; embossed plates are uniformly black-on-white, which is itself a
    useful signal (it tells you which grammar to apply).
    """

    RED_WHITE = "red_white"        # legacy private
    BLACK_WHITE = "black_white"    # legacy public / commercial
    WHITE_RED = "white_red"        # legacy government
    YELLOW_BLACK = "yellow_black"  # legacy national corporation
    GREEN_WHITE = "green_white"    # legacy tourist
    BLUE_WHITE = "blue_white"      # legacy diplomatic
    WHITE_BLACK = "white_black"    # embossed (all categories)
    UNKNOWN = "unknown"


class Confidence(str, Enum):
    """How much operational trust to place in a read.

    Deliberately coarse: operators act on bands, not on float scores, and the
    band is what gets written to the evidence log.
    """

    HIGH = "high"        # grammar-valid, strong posteriors, colour agrees
    MEDIUM = "medium"    # grammar-valid but weak evidence somewhere
    LOW = "low"          # grammar repair was needed, or evidence is thin
    REJECT = "reject"    # not usable as a read; queue for human review


@dataclass(frozen=True, slots=True)
class PlateField:
    """One decoded field of a plate, with the evidence behind it."""

    name: str
    value: str
    #: Mean per-token posterior for the glyphs making up this field, in [0, 1].
    score: float = 0.0
    #: True when grammar repair overrode the raw argmax for this field.
    repaired: bool = False


@dataclass(frozen=True, slots=True)
class ParsedPlate:
    """A fully parsed, validated Nepali plate.

    ``canonical`` is the stable machine key used for indexing, hashing and
    hotlist matching. Two reads of the same physical plate must always produce
    the same ``canonical`` regardless of spacing, script variant or separators.
    """

    system: PlateSystem
    canonical: str
    display: str

    # --- Legacy (Devanagari) fields -------------------------------------
    zone: str | None = None          # romanised zone code, e.g. "BA"
    zone_deva: str | None = None     # Devanagari zone token, e.g. "बा"
    lot: str | None = None           # 1-2 digit lot/batch number, ASCII
    vehicle_class: str | None = None  # romanised class, e.g. "CHA"
    vehicle_class_deva: str | None = None

    # --- Embossed fields -------------------------------------------------
    province: int | None = None      # 1..7
    class_letter: str | None = None  # "A".."K", optionally with subclass digit
    series: str | None = None        # two-letter series, e.g. "PA"

    # --- Common ----------------------------------------------------------
    serial: str | None = None        # 4-digit serial, ASCII digits
    ownership: Ownership = Ownership.UNKNOWN
    size_class: SizeClass = SizeClass.UNKNOWN
    expected_colour: tuple[PlateColour, ...] = ()

    is_valid: bool = False
    confidence: Confidence = Confidence.REJECT
    score: float = 0.0
    fields: tuple[PlateField, ...] = ()
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.display or self.canonical


@dataclass(frozen=True, slots=True)
class ColourEvidence:
    """Output of the plate-colour classifier, used as a decoding prior.

    ``posterior`` maps each :class:`PlateColour` to a probability. It is kept as
    a distribution rather than a hard label so that a genuinely ambiguous colour
    (a red plate at night under sodium light) degrades the prior smoothly
    instead of forcing a wrong constraint.
    """

    posterior: Mapping[PlateColour, float]
    #: Set when the classifier itself is unsure; suppresses the prior entirely.
    reliable: bool = True

    def best(self) -> tuple[PlateColour, float]:
        if not self.posterior:
            return PlateColour.UNKNOWN, 0.0
        colour = max(self.posterior, key=lambda k: self.posterior[k])
        return colour, float(self.posterior[colour])


@dataclass(frozen=True, slots=True)
class GlyphPosterior:
    """Per-position distribution over the recogniser's token vocabulary.

    This is the interface between the neural recogniser and the grammar decoder.
    Keeping it as a distribution (not a string) is what makes grammar-constrained
    decoding possible -- a hard argmax throws away exactly the information the
    grammar needs to repair a blurred glyph.
    """

    #: ``tokens[i]`` is the vocabulary entry, ``probs[i]`` its probability.
    tokens: Sequence[str]
    probs: Sequence[float]

    def as_dict(self) -> dict[str, float]:
        return {t: float(p) for t, p in zip(self.tokens, self.probs)}

    def top(self) -> tuple[str, float]:
        if not self.tokens:
            return "", 0.0
        i = max(range(len(self.probs)), key=lambda j: self.probs[j])
        return self.tokens[i], float(self.probs[i])


@dataclass(frozen=True, slots=True)
class ReadCandidate:
    """A scored hypothesis produced by the decoder, before final selection."""

    plate: ParsedPlate
    log_prob: float
    #: Contribution of the colour prior to ``log_prob``; reported separately so
    #: an operator can see *why* a plate was chosen.
    colour_bonus: float = 0.0
    notes: tuple[str, ...] = field(default_factory=tuple)
