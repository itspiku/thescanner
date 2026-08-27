"""Sampling legal Nepali plates.

The point of synthetic data is *coverage*, not realism alone. Real-world data is
dominated by private motorcycles in Bagmati; government, diplomatic, tourist and
corporation plates barely appear at all, and those are exactly the plates where
a misread costs the most. So the default sampler here is deliberately **uniform
over the structural dimensions** — every zone, every class, every ownership
category equally represented — rather than matching the road-frequency
distribution.

A frequency-weighted sampler is also provided, for the fine-tuning stage where
matching the deployment distribution is what you want. The two get mixed
according to the schedule in ``docs/research/datasets.md``.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterator, Literal, Sequence

from nepal_plate import PlateColour, PlateSystem, spec
from nepal_plate.grammar import EMBOSSED_GRAMMAR, LEGACY_GRAMMAR
from nepal_plate.parse import plate_from_tokens
from nepal_plate.types import ParsedPlate

Strategy = Literal["uniform", "realistic"]


@dataclass(frozen=True, slots=True)
class PlateSample:
    """A sampled plate plus everything the renderer needs to draw it."""

    plate: ParsedPlate
    colour: PlateColour
    #: Token sequence, so the renderer does not have to re-parse.
    tokens: tuple[str, ...]
    #: Two-row layout is standard for motorcycles and for most rear plates;
    #: single-row appears on car front plates.
    two_row: bool


# Rough road-frequency weights for the "realistic" strategy. Nepal's fleet is
# overwhelmingly two-wheeled and overwhelmingly private.
_REALISTIC_CLASS_WEIGHTS: dict[str, float] = {
    "प": 55.0,   # private motorcycle
    "च": 22.0,   # private light
    "ज": 6.0,    # public light
    "फ": 4.0,    # public motorcycle
    "ख": 3.0,    # public heavy
    "क": 2.5,    # private heavy
    "झ": 2.0,    # government light
    "ब": 1.5,    # government motorcycle
    "ग": 1.2,    # government heavy
    "य": 1.0,    # tourist
    "ञ": 0.8,    # corporation light
    "घ": 0.7,    # corporation heavy
    "सी.डी.": 0.3,  # diplomatic
}

# Registration volume is concentrated in Bagmati; Karnali and the far-western
# zones issue comparatively few plates.
_REALISTIC_ZONE_WEIGHTS: dict[str, float] = {
    "बा": 34.0, "ना": 10.0, "लु": 9.0, "को": 8.0, "ग": 7.0,
    "ज": 6.0, "स": 5.5, "रा": 4.5, "मे": 4.0, "भे": 4.0,
    "से": 3.0, "ध": 2.5, "म": 2.0, "क": 1.5,
}


class PlateSampler:
    """Draws legal plates from the grammar.

    Sampling from the *grammar* rather than from a hand-written format string
    means the generator cannot drift out of sync with the specification: adding
    a zone code or a vehicle class to ``nepal_plate.spec`` immediately changes
    what gets rendered, with no second place to update.
    """

    def __init__(
        self,
        *,
        seed: int | None = None,
        strategy: Strategy = "uniform",
        embossed_fraction: float = 0.5,
    ) -> None:
        self.rng = random.Random(seed)
        self.strategy = strategy
        #: Fraction of samples drawn from the embossed system. 0.5 by default
        #: because the two systems are both live and the model must not develop
        #: a prior toward whichever happens to dominate a training set.
        self.embossed_fraction = embossed_fraction

    # -- helpers --------------------------------------------------------

    def _choice(self, options: Sequence[str], weights: dict[str, float] | None) -> str:
        if self.strategy == "uniform" or weights is None:
            return self.rng.choice(list(options))
        w = [weights.get(o, 1.0) for o in options]
        return self.rng.choices(list(options), weights=w, k=1)[0]

    def _serial_tokens(self, digits: Sequence[str]) -> list[str]:
        """Sample a serial. Four digits overwhelmingly, occasionally shorter.

        Leading zeros are permitted and common on real plates, so they are
        sampled rather than avoided -- a model that never sees ``०१२३`` will
        misread it.
        """
        n = self.rng.choices([4, 3, 2, 1], weights=[88, 7, 3, 2], k=1)[0]
        return [self.rng.choice(list(digits)) for _ in range(n)]

    # -- samplers -------------------------------------------------------

    def sample_legacy(self) -> PlateSample:
        zone = self._choice(spec.ZONE_TOKENS, _REALISTIC_ZONE_WEIGHTS)
        klass = self._choice(spec.CLASS_TOKENS, _REALISTIC_CLASS_WEIGHTS)

        # Lot numbers appear only once a zone's four-digit series is exhausted,
        # which in practice means only in high-volume zones.
        lot: list[str] = []
        if self.rng.random() < 0.25:
            n_lot = self.rng.choices([1, 2], weights=[85, 15], k=1)[0]
            lot = [self.rng.choice(list(spec.DEVA_DIGITS)) for _ in range(n_lot)]
            # A leading-zero lot would not be issued.
            if lot[0] == "०":
                lot[0] = self.rng.choice(list(spec.DEVA_DIGITS[1:]))

        tokens = [zone, *lot, klass, *self._serial_tokens(spec.DEVA_DIGITS)]
        plate = plate_from_tokens(LEGACY_GRAMMAR, tokens, score=1.0)

        colours = spec.OWNERSHIP_COLOURS.get(plate.ownership, ())
        legacy_colours = [c for c in colours if c in spec.LEGACY_COLOUR_OWNERSHIP]
        colour = legacy_colours[0] if legacy_colours else PlateColour.UNKNOWN

        # Motorcycles are essentially always two-row; larger vehicles vary by
        # whether it is a front or rear plate.
        two_row = (
            True
            if plate.size_class.value == "motorcycle"
            else self.rng.random() < 0.45
        )
        return PlateSample(plate=plate, colour=colour, tokens=tuple(tokens), two_row=two_row)

    def sample_embossed(self) -> PlateSample:
        province = self.rng.choice("1234567")
        letter = self.rng.choice(spec.EMBOSSED_BASE_LETTERS)

        subclass: list[str] = []
        options = spec.EMBOSSED_SUBCLASSES.get(letter, ())
        if options:
            # J is only ever issued with a subclass; the others take one
            # sometimes.
            if letter == "J" or self.rng.random() < 0.5:
                subclass = [self.rng.choice(list(options))]

        series = [self.rng.choice(spec.LATIN_LETTERS) for _ in range(2)]
        tokens = [province, letter, *subclass, *series, *self._serial_tokens(spec.LATIN_DIGITS)]
        plate = plate_from_tokens(EMBOSSED_GRAMMAR, tokens, score=1.0)

        two_row = (
            True
            if plate.size_class.value == "motorcycle"
            else self.rng.random() < 0.4
        )
        return PlateSample(
            plate=plate,
            colour=PlateColour.WHITE_BLACK,
            tokens=tuple(tokens),
            two_row=two_row,
        )

    def sample(self) -> PlateSample:
        if self.rng.random() < self.embossed_fraction:
            return self.sample_embossed()
        return self.sample_legacy()

    def stream(self, n: int | None = None) -> Iterator[PlateSample]:
        """Yield ``n`` samples, or an unbounded stream when ``n`` is None."""
        i = 0
        while n is None or i < n:
            yield self.sample()
            i += 1


def sample_balanced_grid(seed: int | None = None) -> Iterator[PlateSample]:
    """Exhaustive sweep of every (zone, class) and (province, class) pair.

    Guarantees that no structural combination is absent from the corpus, which
    uniform random sampling only achieves in expectation. Used to seed the
    training set before random sampling fills it out.
    """
    rng = random.Random(seed)
    sampler = PlateSampler(seed=seed)

    for zone in spec.ZONE_TOKENS:
        for klass in spec.CLASS_TOKENS:
            tokens = [zone, klass] + [rng.choice(list(spec.DEVA_DIGITS)) for _ in range(4)]
            plate = plate_from_tokens(LEGACY_GRAMMAR, tokens, score=1.0)
            colours = [
                c
                for c in spec.OWNERSHIP_COLOURS.get(plate.ownership, ())
                if c in spec.LEGACY_COLOUR_OWNERSHIP
            ]
            yield PlateSample(
                plate=plate,
                colour=colours[0] if colours else PlateColour.UNKNOWN,
                tokens=tuple(tokens),
                two_row=plate.size_class.value == "motorcycle",
            )

    for province in "1234567":
        for letter in spec.EMBOSSED_BASE_LETTERS:
            subs = spec.EMBOSSED_SUBCLASSES.get(letter, ())
            variants: list[list[str]] = [[]] if letter != "J" else []
            variants += [[s] for s in subs]
            for sub in variants:
                tokens = (
                    [province, letter, *sub]
                    + [rng.choice(list(spec.LATIN_LETTERS)) for _ in range(2)]
                    + [rng.choice(list(spec.LATIN_DIGITS)) for _ in range(4)]
                )
                plate = plate_from_tokens(EMBOSSED_GRAMMAR, tokens, score=1.0)
                yield PlateSample(
                    plate=plate,
                    colour=PlateColour.WHITE_BLACK,
                    tokens=tuple(tokens),
                    two_row=plate.size_class.value == "motorcycle",
                )


__all__ = ["PlateSample", "PlateSampler", "sample_balanced_grid", "Strategy"]
