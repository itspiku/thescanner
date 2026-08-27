"""``nepal_plate`` -- the domain core of TheScanner.

A dependency-free (standard library only) model of Nepali vehicle registration
plates: the reference data, the layout grammars for both of Nepal's coexisting
plate systems, a grammar-constrained CTC decoder, and multi-frame fusion.

Everything else in the system -- edge agent, training pipeline, synthetic data
generator, API, UI -- depends on this package and not the other way round.

Typical use::

    from nepal_plate import parse, decode, ColourEvidence, PlateColour

    plate = parse("बा १ च १२३४")
    plate.canonical        # 'NP-L:BA-1-CHA-1234'
    plate.ownership        # Ownership.PRIVATE

    candidates = decode(log_probs, colour=ColourEvidence({PlateColour.RED_WHITE: 0.9}))
    candidates[0].plate.display
"""

from __future__ import annotations

from . import spec
from .decode import best, colour_slot_bonus, ctc_grammar_beam_search, decode
from .fuse import fuse_track, fuse_posteriors
from .grammar import (
    EMBOSSED_GRAMMAR,
    GRAMMARS,
    LEGACY_GRAMMAR,
    Grammar,
    Slot,
    grammars_for,
    language_size,
)
from .parse import canonicalise, parse, plate_from_tokens, tokenize_devanagari, tokenize_latin
from .types import (
    ColourEvidence,
    Confidence,
    GlyphPosterior,
    Ownership,
    ParsedPlate,
    PlateColour,
    PlateField,
    PlateSystem,
    ReadCandidate,
    SizeClass,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "spec",
    # types
    "ColourEvidence",
    "Confidence",
    "GlyphPosterior",
    "Ownership",
    "ParsedPlate",
    "PlateColour",
    "PlateField",
    "PlateSystem",
    "ReadCandidate",
    "SizeClass",
    # grammar
    "Grammar",
    "Slot",
    "LEGACY_GRAMMAR",
    "EMBOSSED_GRAMMAR",
    "GRAMMARS",
    "grammars_for",
    "language_size",
    # parsing
    "parse",
    "canonicalise",
    "plate_from_tokens",
    "tokenize_devanagari",
    "tokenize_latin",
    # decoding
    "decode",
    "best",
    "ctc_grammar_beam_search",
    "colour_slot_bonus",
    "fuse_posteriors",
    "fuse_track",
]
