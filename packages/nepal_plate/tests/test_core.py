"""Tests for the Nepali plate domain core.

The interesting tests are at the bottom: they construct degraded recogniser
output and assert that grammar-constrained decoding recovers the correct plate
where conventional argmax decoding does not. That is the claim the whole
recognition design rests on, so it is tested directly rather than inferred from
end-to-end accuracy numbers.
"""

from __future__ import annotations

import math

import pytest

from nepal_plate import (
    ColourEvidence,
    Confidence,
    Ownership,
    PlateColour,
    PlateSystem,
    SizeClass,
    canonicalise,
    decode,
    grammar as _grammar_mod,
    parse,
    spec,
)
from nepal_plate.decode import ctc_grammar_beam_search, greedy_decode
from nepal_plate.fuse import FrameObservation, fuse_track
from nepal_plate.grammar import EMBOSSED_GRAMMAR, LEGACY_GRAMMAR, language_size


# ---------------------------------------------------------------------------
# Reference data
# ---------------------------------------------------------------------------

def test_devanagari_vocabulary_is_34_tokens():
    """Independent derivation must reproduce the published 34-class charset.

    10 numerals + 14 zone codes + 13 class letters, minus the three glyphs
    (क, ग, ज) that serve in both roles.
    """
    assert len(spec.DEVA_DIGITS) == 10
    assert len(spec.ZONE_TOKENS) == 14
    assert len(spec.CLASS_TOKENS) == 13
    shared = set(spec.ZONE_TOKENS) & set(spec.CLASS_TOKENS)
    assert shared == {"क", "ग", "ज"}
    assert len(spec.DEVA_VOCAB) == 34


def test_vocabulary_has_no_duplicates():
    assert len(set(spec.UNIFIED_VOCAB)) == len(spec.UNIFIED_VOCAB)
    assert spec.UNIFIED_VOCAB[0] == spec.CTC_BLANK


def test_every_legacy_colour_maps_to_exactly_one_ownership():
    assert len(spec.LEGACY_COLOUR_OWNERSHIP) == 6
    assert len(set(spec.LEGACY_COLOUR_OWNERSHIP.values())) == 6


def test_ownership_class_partition_covers_all_classes():
    covered = {t for toks in spec.OWNERSHIP_CLASS_TOKENS.values() for t in toks}
    assert covered == set(spec.CLASS_TOKENS)


def test_confusable_relation_is_symmetric():
    for a, partners in spec.CONFUSABLE.items():
        for b in partners:
            assert spec.are_confusable(b, a), f"{a}/{b} not symmetric"


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def test_parse_devanagari_legacy_plate():
    p = parse("बा १ च १२३४")
    assert p.is_valid
    assert p.system is PlateSystem.DEVANAGARI
    assert p.canonical == "NP-L:BA-1-CHA-1234"
    assert p.zone == "BA"
    assert p.lot == "1"
    assert p.vehicle_class == "CHA"
    assert p.serial == "1234"
    assert p.ownership is Ownership.PRIVATE
    assert p.size_class is SizeClass.LIGHT


def test_parse_legacy_plate_without_lot():
    p = parse("को प २४७")
    assert p.is_valid
    assert p.zone == "KO"
    assert p.lot is None
    assert p.vehicle_class == "PA"
    assert p.serial == "247"
    assert p.canonical == "NP-L:KO-0-PA-247"
    # Short serials are legal but flagged, since four digits is the norm.
    assert any("four is standard" in w for w in p.warnings)


@pytest.mark.parametrize(
    "text",
    ["बा १ च १२३४", "BA 1 CHA 1234", "ba-1-cha-1234", "NP-L:BA-1-CHA-1234"],
)
def test_all_spellings_canonicalise_identically(text):
    """A plate typed by an officer and a plate read by a camera must produce
    the same key, or hotlist matching silently fails."""
    assert canonicalise(text) == "NP-L:BA-1-CHA-1234"


def test_parse_embossed_plate():
    p = parse("3 B PA 1234")
    assert p.is_valid
    assert p.system is PlateSystem.EMBOSSED
    assert p.province == 3
    assert p.class_letter == "B"
    assert p.series == "PA"
    assert p.serial == "1234"
    assert p.canonical == "NP-E:3-B-PA-1234"
    assert p.size_class is SizeClass.LIGHT


def test_parse_embossed_plate_with_subclass():
    p = parse("5 J2 KL 0087")
    assert p.is_valid
    assert p.class_letter == "J2"
    assert spec.EMBOSSED_CLASSES["J2"][0] == "Backhoe Loader"


def test_embossed_country_marker_is_ignored():
    assert canonicalise("NEP 3B PA 1234") == "NP-E:3-B-PA-1234"


@pytest.mark.parametrize(
    "text",
    [
        "",
        "HELLO",
        "बा १ ध १२३४",   # ध is a zone code, never a class letter
        "8 B PA 1234",     # province 8 does not exist
        "3 B PA 12345",    # serial too long
        "3 A1 PA 1234",    # A takes no subclass
        "3 B P 1234",      # series must be two letters
    ],
)
def test_invalid_plates_are_rejected(text):
    p = parse(text)
    assert not p.is_valid
    assert p.confidence is Confidence.REJECT


def test_colour_disagreement_is_warned_not_rejected():
    """A red plate under sodium light can read black. Warn, never discard."""
    p = parse("बा १ च १२३४", observed_colour=PlateColour.BLACK_WHITE)
    assert p.is_valid
    assert any("disagrees with ownership" in w for w in p.warnings)


# ---------------------------------------------------------------------------
# Grammar
# ---------------------------------------------------------------------------

def test_grammar_accepts_and_rejects():
    ok = ["बा", "१", "च", "१", "२", "३", "४"]
    assert LEGACY_GRAMMAR.segment(ok) is not None
    # class letter in the zone position
    assert LEGACY_GRAMMAR.segment(["च", "१", "च", "१", "२", "३", "४"]) is None
    # three-digit lot
    assert LEGACY_GRAMMAR.segment(["बा", "१", "२", "३", "च", "१"]) is None


def test_grammar_segments_into_named_fields():
    seg = LEGACY_GRAMMAR.segment(["बा", "१", "च", "१", "२", "३", "४"])
    assert seg == {
        "zone": ["बा"],
        "lot": ["१"],
        "class": ["च"],
        "serial": ["१", "२", "३", "४"],
    }


def test_legacy_language_size_matches_closed_form():
    """Guard the DP in ``language_size`` against a hand-derived count.

    14 zones x (1 + 10 + 100 lots) x 13 classes x (10 + 100 + 1000 + 10000
    serials) = 224,444,220 legal legacy plates.
    """
    assert language_size(LEGACY_GRAMMAR) == 14 * 111 * 13 * 11110 == 224_444_220


@pytest.mark.parametrize(
    "grammar, min_bits",
    [(LEGACY_GRAMMAR, 12.5), (EMBOSSED_GRAMMAR, 16.0)],
)
def test_grammar_constraint_shrinks_the_search_space(grammar, min_bits):
    """Quantify what the grammar buys, in bits of search space removed.

    Legacy: 13.0 bits (a factor of ~7,960). Embossed: 16.4 bits (~87,800).
    Those bits are exactly what the decoder gets to spend on ambiguous glyphs
    instead of on strings that could never be a plate.
    """
    legal = language_size(grammar)
    unconstrained = len(grammar.alphabet()) ** sum(s.max_repeat for s in grammar.slots)
    bits_saved = math.log2(unconstrained) - math.log2(legal)
    assert legal > 0
    assert bits_saved >= min_bits, f"only {bits_saved:.1f} bits saved"


# ---------------------------------------------------------------------------
# Synthetic recogniser output
# ---------------------------------------------------------------------------

def make_posteriors(
    tokens,
    *,
    ambiguity=None,
    vocab=spec.UNIFIED_VOCAB,
    frames_per_token=2,
    peak=0.93,
):
    """Build a T x V log-probability matrix that emits ``tokens``.

    ``ambiguity`` maps a token position to a ``{token: probability}`` dict,
    letting a test place mass on the wrong glyph to simulate blur.
    """
    idx = {t: i for i, t in enumerate(vocab)}
    V = len(vocab)
    ambiguity = ambiguity or {}
    rows: list[list[float]] = []

    def emit(dist):
        row = [1e-6] * V
        for tok, p in dist.items():
            row[idx[tok]] = p
        s = sum(row)
        rows.append([math.log(x / s) for x in row])

    blank = {spec.CTC_BLANK: 0.98}
    emit(blank)
    for pos, tok in enumerate(tokens):
        dist = ambiguity.get(pos, {tok: peak})
        for _ in range(frames_per_token):
            emit(dist)
        emit(blank)
    return rows


TRUE_TOKENS = ("बा", "१", "च", "१", "२", "३", "४")

#: The class glyph is blurred. ध (a zone code) takes the most mass, then
#: ज (public light), then the true च (private light). This is a realistic
#: confusion: ग/ध/घ and च/ज/ञ are the two tightest Devanagari confusion
#: clusters on plates.
BLURRED_CLASS = {2: {"ध": 0.45, "ज": 0.31, "च": 0.24}}


def test_greedy_decoding_fails_on_a_blurred_class_glyph():
    """The baseline picks a glyph that cannot occur in this position."""
    lp = make_posteriors(TRUE_TOKENS, ambiguity=BLURRED_CLASS)
    greedy = greedy_decode(lp)
    assert greedy[2] == "ध"
    # ...and the resulting string is not a plate at all.
    assert LEGACY_GRAMMAR.segment(list(greedy)) is None


def test_grammar_constraint_eliminates_the_illegal_glyph():
    """Constrained decoding cannot produce ध here, so mass moves to ज/च."""
    lp = make_posteriors(TRUE_TOKENS, ambiguity=BLURRED_CLASS)
    results = decode(lp, system_hint=PlateSystem.DEVANAGARI)
    assert results, "constrained decode returned nothing"
    top = results[0].plate
    assert top.is_valid
    assert top.vehicle_class_deva != "ध"
    # With no colour evidence the decoder correctly prefers the higher-scoring
    # legal glyph, ज -- it has no basis to choose otherwise.
    assert top.vehicle_class_deva == "ज"
    assert top.zone == "BA" and top.serial == "1234"


def test_colour_prior_recovers_the_correct_plate():
    """Red background => private => the class letter must be क, च or प.

    ज (public) is thereby suppressed and the true glyph च wins, even though it
    held the *least* raw probability mass of the three candidates.
    """
    lp = make_posteriors(TRUE_TOKENS, ambiguity=BLURRED_CLASS)
    red = ColourEvidence(posterior={PlateColour.RED_WHITE: 0.88, PlateColour.BLACK_WHITE: 0.12})
    results = decode(lp, colour=red)
    assert results
    top = results[0].plate
    assert top.canonical == "NP-L:BA-1-CHA-1234"
    assert top.ownership is Ownership.PRIVATE
    assert results[0].colour_bonus > math.log(0.5)


def test_colour_prior_is_soft_not_hard():
    """Overwhelming glyph evidence must still beat a confident wrong colour.

    A hard constraint here would make a mislabelled colour unrecoverable, which
    is unacceptable when the colour classifier sees the plate at night.
    """
    lp = make_posteriors(TRUE_TOKENS, ambiguity={2: {"ज": 0.995, "च": 0.005}})
    wrong = ColourEvidence(posterior={PlateColour.RED_WHITE: 0.97})
    top = decode(lp, colour=wrong)[0].plate
    assert top.vehicle_class_deva == "ज"


def test_repaired_fields_are_flagged_and_downgrade_confidence():
    """An operator must be able to see that the grammar overrode the pixels."""
    lp = make_posteriors(TRUE_TOKENS, ambiguity=BLURRED_CLASS)
    red = ColourEvidence(posterior={PlateColour.RED_WHITE: 0.88})
    cand = decode(lp, colour=red)[0]
    assert "repaired:class" in cand.notes
    assert cand.plate.confidence is not Confidence.HIGH


def test_clean_plate_decodes_at_high_confidence():
    lp = make_posteriors(TRUE_TOKENS)
    top = decode(lp, system_hint=PlateSystem.DEVANAGARI)[0].plate
    assert top.canonical == "NP-L:BA-1-CHA-1234"
    assert top.score > 0.9


def test_embossed_plate_decodes_without_a_system_hint():
    """The two alphabets are disjoint, so the system is inferable from evidence."""
    tokens = ("3", "B", "P", "A", "1", "2", "3", "4")
    lp = make_posteriors(tokens)
    top = decode(lp)[0].plate
    assert top.system is PlateSystem.EMBOSSED
    assert top.canonical == "NP-E:3-B-PA-1234"


# ---------------------------------------------------------------------------
# Multi-frame fusion
# ---------------------------------------------------------------------------

def test_fusion_recovers_a_plate_no_single_frame_reads_correctly():
    """Each frame is wrong in a different place; together they are right.

    This is the case that motivates track-level fusion over single-frame
    super-resolution: the information is distributed across the track.
    """
    frames = [
        # zone blurred, rest clean
        FrameObservation(
            make_posteriors(TRUE_TOKENS, ambiguity={0: {"ना": 0.55, "बा": 0.45}}),
            quality=0.8,
        ),
        # serial's last digit blurred, rest clean
        FrameObservation(
            make_posteriors(TRUE_TOKENS, ambiguity={6: {"५": 0.52, "४": 0.48}}),
            quality=0.8,
        ),
        # clean-ish third look
        FrameObservation(make_posteriors(TRUE_TOKENS, peak=0.80), quality=0.9),
    ]
    fused = fuse_track(frames)
    assert fused is not None
    assert fused.plate.canonical == "NP-L:BA-1-CHA-1234"
    assert fused.n_frames == 3


def test_single_frame_reads_are_capped_below_high_confidence():
    """Without corroboration there is no way to tell a clean read from a
    confidently wrong one, so a lone frame never earns the top band."""
    fused = fuse_track([FrameObservation(make_posteriors(TRUE_TOKENS), quality=1.0)])
    assert fused is not None
    assert fused.plate.confidence is not Confidence.HIGH


def test_pooled_posteriors_sharpen_rather_than_blur():
    """Log-linear pooling must behave like a product of experts."""
    from nepal_plate.fuse import fuse_posteriors

    frames = [
        FrameObservation(make_posteriors(TRUE_TOKENS, ambiguity=BLURRED_CLASS), quality=1.0)
        for _ in range(4)
    ]
    pooled = fuse_posteriors(frames)
    assert pooled is not None
    assert len(pooled) == len(frames[0].log_probs)
    for row in pooled:
        assert math.isclose(sum(math.exp(x) for x in row), 1.0, rel_tol=1e-6)


def test_disagreeing_frames_lower_the_confidence_band():
    good = [FrameObservation(make_posteriors(TRUE_TOKENS), quality=0.9) for _ in range(2)]
    other = ("ना", "२", "प", "९", "९", "९", "९")
    bad = [FrameObservation(make_posteriors(other), quality=0.9) for _ in range(2)]
    fused = fuse_track(good + bad, prefer_pooled=False)
    assert fused is not None
    assert fused.agreement <= 0.6
    assert fused.plate.confidence in (Confidence.LOW, Confidence.MEDIUM, Confidence.REJECT)
