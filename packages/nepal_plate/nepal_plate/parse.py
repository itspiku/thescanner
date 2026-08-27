"""Parsing, normalisation and canonicalisation of Nepali plate strings.

Two entry points matter:

``parse(text)``
    Tolerantly read a plate written by a human or emitted by a third-party OCR
    -- in Devanagari or romanised, with any spacing or separators -- and return
    a validated :class:`ParsedPlate`.

``plate_from_tokens(grammar, tokens, ...)``
    Assemble a :class:`ParsedPlate` from a token sequence the decoder produced.

Both funnel into the same validation and canonicalisation logic so that a plate
typed into a hotlist by an officer and a plate read off a camera produce the
*identical* ``canonical`` key. That property is what makes hotlist matching
work at all, and it is the single most common place national ANPR deployments
get subtly wrong.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Iterable, Mapping, Sequence

from . import spec
from .grammar import EMBOSSED_GRAMMAR, LEGACY_GRAMMAR, Grammar
from .types import (
    Confidence,
    Ownership,
    ParsedPlate,
    PlateColour,
    PlateField,
    PlateSystem,
    SizeClass,
)

_SEPARATORS = re.compile(r"[\s\-_.·•/|,]+")
_CANON_PREFIX = {PlateSystem.DEVANAGARI: "NP-L", PlateSystem.EMBOSSED: "NP-E"}

#: Longest-first so that greedy matching consumes ``बा`` before ``ब``.
_DEVA_TOKENS_BY_LENGTH: tuple[str, ...] = tuple(
    sorted(spec.DEVA_VOCAB, key=len, reverse=True)
)


# ---------------------------------------------------------------------------
# Tokenisation
# ---------------------------------------------------------------------------

def tokenize_devanagari(text: str) -> list[str] | None:
    """Split a Devanagari plate string into atomic vocabulary tokens.

    Greedy longest-match. Returns ``None`` if any part of the string is not in
    the plate vocabulary -- being strict here is deliberate, because silently
    dropping an unrecognised glyph is how you end up matching the wrong vehicle.
    """
    s = unicodedata.normalize("NFC", text)
    s = _SEPARATORS.sub("", s)
    out: list[str] = []
    i = 0
    while i < len(s):
        for tok in _DEVA_TOKENS_BY_LENGTH:
            if s.startswith(tok, i):
                out.append(tok)
                i += len(tok)
                break
        else:
            return None
    return out


def tokenize_latin(text: str) -> list[str] | None:
    """Split an embossed plate string into single-character tokens."""
    s = _SEPARATORS.sub("", text.upper())
    # A leading country marker is cosmetic and carries no registration data.
    if s.startswith("NEP"):
        s = s[3:]
    if not s or any(c not in spec.LATIN_VOCAB for c in s):
        return None
    return list(s)


# ---------------------------------------------------------------------------
# Assembly from tokens
# ---------------------------------------------------------------------------

def plate_from_tokens(
    grammar: Grammar,
    tokens: Sequence[str],
    *,
    score: float = 0.0,
    token_scores: Sequence[float] | None = None,
    repaired_slots: Iterable[str] = (),
    observed_colour: PlateColour = PlateColour.UNKNOWN,
) -> ParsedPlate:
    """Build a validated :class:`ParsedPlate` from an accepted token sequence.

    ``observed_colour`` is optional; when supplied it is checked against the
    ownership implied by the class letter. Disagreement does not invalidate the
    plate -- a genuinely red plate can look black at night -- but it is recorded
    as a warning and downgrades confidence, and it is the primary signal for
    detecting a plate whose colour has been altered.
    """
    fields_map = grammar.segment(tokens)
    if fields_map is None:
        return _rejected(
            "".join(tokens),
            errors=(f"token sequence is not accepted by grammar {grammar.name!r}",),
        )

    repaired = set(repaired_slots)
    scores = list(token_scores or ())

    def _field(name: str) -> PlateField:
        toks = fields_map.get(name, [])
        # Recover this field's slice of the per-token scores by walking the
        # token list in order, so the reported per-field score reflects the
        # actual glyphs behind it rather than a plate-wide average.
        start = 0
        for slot in grammar.slots:
            if slot.name == name:
                break
            start += len(fields_map.get(slot.name, []))
        sl = scores[start : start + len(toks)] if scores else []
        mean = sum(sl) / len(sl) if sl else score
        return PlateField(name=name, value="".join(toks), score=float(mean), repaired=name in repaired)

    if grammar.system is PlateSystem.DEVANAGARI:
        return _assemble_legacy(fields_map, _field, score, observed_colour)
    return _assemble_embossed(fields_map, _field, score, observed_colour)


def _assemble_legacy(
    fields_map: Mapping[str, list[str]],
    field_of,
    score: float,
    observed_colour: PlateColour,
) -> ParsedPlate:
    zone_tok = "".join(fields_map.get("zone", []))
    class_tok = "".join(fields_map.get("class", []))
    lot_deva = "".join(fields_map.get("lot", []))
    serial_deva = "".join(fields_map.get("serial", []))

    zone = spec.ZONE_BY_DEVA.get(zone_tok)
    vclass = spec.CLASS_BY_DEVA.get(class_tok)
    errors: list[str] = []
    warnings: list[str] = []
    if zone is None:
        errors.append(f"unknown zone token {zone_tok!r}")
    if vclass is None:
        errors.append(f"unknown class token {class_tok!r}")
    if errors:
        return _rejected(zone_tok + class_tok, errors=tuple(errors))

    lot_ascii = _deva_to_ascii(lot_deva)
    serial_ascii = _deva_to_ascii(serial_deva)

    ownership = vclass.ownership
    expected = spec.OWNERSHIP_COLOURS.get(ownership, ())
    if observed_colour is not PlateColour.UNKNOWN and observed_colour not in expected:
        warnings.append(
            f"plate colour {observed_colour.value!r} disagrees with ownership "
            f"{ownership.value!r} implied by class {vclass.roman!r}"
        )

    if len(serial_ascii) < 4:
        warnings.append(f"serial has {len(serial_ascii)} digits; four is standard")

    fields = tuple(
        field_of(n) for n in ("zone", "lot", "class", "serial") if fields_map.get(n)
    )
    lot_canon = lot_ascii or "0"
    canonical = f"NP-L:{zone.roman}-{lot_canon}-{vclass.roman}-{serial_ascii}"
    display_parts = [zone_tok] + ([lot_deva] if lot_deva else []) + [class_tok, serial_deva]

    return ParsedPlate(
        system=PlateSystem.DEVANAGARI,
        canonical=canonical,
        display=" ".join(display_parts),
        zone=zone.roman,
        zone_deva=zone_tok,
        lot=lot_ascii or None,
        vehicle_class=vclass.roman,
        vehicle_class_deva=class_tok,
        serial=serial_ascii,
        ownership=ownership,
        size_class=vclass.size,
        expected_colour=tuple(expected),
        is_valid=True,
        confidence=_confidence(score, warnings, fields),
        score=float(score),
        fields=fields,
        warnings=tuple(warnings),
    )


def _assemble_embossed(
    fields_map: Mapping[str, list[str]],
    field_of,
    score: float,
    observed_colour: PlateColour,
) -> ParsedPlate:
    province_s = "".join(fields_map.get("province", []))
    letter = "".join(fields_map.get("class_letter", []))
    subclass = "".join(fields_map.get("subclass", []))
    series = "".join(fields_map.get("series", []))
    serial = "".join(fields_map.get("serial", []))

    warnings: list[str] = []
    errors: list[str] = []

    try:
        province = int(province_s)
    except ValueError:
        return _rejected(province_s, errors=(f"bad province {province_s!r}",))
    if province not in spec.PROVINCE_BY_NUMBER:
        errors.append(f"province {province} out of range 1-7")

    class_letter = letter + subclass
    if class_letter not in spec.EMBOSSED_CLASSES:
        errors.append(f"unknown embossed class {class_letter!r}")
    if errors:
        return _rejected(class_letter, errors=tuple(errors))

    _, size = spec.EMBOSSED_CLASSES[class_letter]

    if observed_colour not in (PlateColour.UNKNOWN, PlateColour.WHITE_BLACK):
        warnings.append(
            f"embossed plates are black-on-white; observed {observed_colour.value!r}"
        )
    if len(serial) < 4:
        warnings.append(f"serial has {len(serial)} digits; four is standard")

    fields = tuple(
        field_of(n)
        for n in ("province", "class_letter", "subclass", "series", "serial")
        if fields_map.get(n)
    )
    canonical = f"NP-E:{province}-{class_letter}-{series}-{serial}"

    return ParsedPlate(
        system=PlateSystem.EMBOSSED,
        canonical=canonical,
        display=f"{province} {class_letter} {series} {serial}",
        province=province,
        class_letter=class_letter,
        series=series,
        serial=serial,
        # Embossed plates no longer encode ownership in colour, and the class
        # letter encodes vehicle type rather than ownership, so ownership must
        # come from the registry rather than from the plate itself.
        ownership=Ownership.UNKNOWN,
        size_class=size,
        expected_colour=(PlateColour.WHITE_BLACK,),
        is_valid=True,
        confidence=_confidence(score, warnings, fields),
        score=float(score),
        fields=fields,
        warnings=tuple(warnings),
    )


# ---------------------------------------------------------------------------
# String parsing
# ---------------------------------------------------------------------------

def parse(text: str, *, observed_colour: PlateColour = PlateColour.UNKNOWN) -> ParsedPlate:
    """Parse a plate string in either script, romanised or native.

    Accepts, among others::

        बा १ च १२३४        BA 1 CHA 1234       ba-1-cha-1234
        NP-L:BA-1-CHA-1234  3 B PA 1234         NEP 3B PA 1234
    """
    if not text or not text.strip():
        return _rejected(text, errors=("empty input",))

    raw = text.strip()
    for prefix in ("NP-L:", "NP-E:"):
        if raw.upper().startswith(prefix):
            raw = raw[len(prefix) :]

    # Native Devanagari.
    toks = tokenize_devanagari(raw)
    if toks is not None and LEGACY_GRAMMAR.segment(toks) is not None:
        return plate_from_tokens(
            LEGACY_GRAMMAR, toks, score=1.0, observed_colour=observed_colour
        )

    # Romanised legacy, e.g. "BA 1 CHA 1234".
    romanised = _parse_romanised_legacy(raw, observed_colour)
    if romanised is not None:
        return romanised

    # Embossed.
    toks = tokenize_latin(raw)
    if toks is not None and EMBOSSED_GRAMMAR.segment(toks) is not None:
        return plate_from_tokens(
            EMBOSSED_GRAMMAR, toks, score=1.0, observed_colour=observed_colour
        )

    return _rejected(text, errors=("string does not match any Nepali plate layout",))


def _parse_romanised_legacy(
    raw: str, observed_colour: PlateColour
) -> ParsedPlate | None:
    parts = [p for p in _SEPARATORS.split(raw.upper()) if p]
    if len(parts) == 3:
        zone_r, class_r, serial = parts
        lot = ""
    elif len(parts) == 4:
        zone_r, lot, class_r, serial = parts
    else:
        return None

    zone = spec.ZONE_BY_ROMAN.get(zone_r)
    vclass = spec.CLASS_BY_ROMAN.get(class_r)
    if zone is None or vclass is None:
        return None
    if lot and not (lot.isdigit() and len(lot) <= 2):
        return None
    if not (serial.isdigit() and 1 <= len(serial) <= 4):
        return None

    toks = [zone.deva]
    toks += [spec.ASCII_TO_DEVA_DIGIT[c] for c in lot]
    toks.append(vclass.deva)
    toks += [spec.ASCII_TO_DEVA_DIGIT[c] for c in serial]
    if LEGACY_GRAMMAR.segment(toks) is None:
        return None
    return plate_from_tokens(
        LEGACY_GRAMMAR, toks, score=1.0, observed_colour=observed_colour
    )


def canonicalise(text: str) -> str | None:
    """Return the canonical key for ``text``, or ``None`` if it is not a plate."""
    p = parse(text)
    return p.canonical if p.is_valid else None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _deva_to_ascii(s: str) -> str:
    return "".join(spec.DEVA_TO_ASCII_DIGIT.get(c, c) for c in s)


def _confidence(
    score: float, warnings: Sequence[str], fields: Sequence[PlateField]
) -> Confidence:
    """Map a path score plus structural warnings onto an operational band.

    Thresholds are on mean per-token posterior. They are intentionally
    conservative: in a law-enforcement setting a false ``HIGH`` is far more
    costly than an extra item in the human review queue.
    """
    weakest = min((f.score for f in fields), default=score)
    repaired = any(f.repaired for f in fields)

    if score >= 0.90 and weakest >= 0.70 and not warnings and not repaired:
        return Confidence.HIGH
    if score >= 0.75 and weakest >= 0.45:
        return Confidence.MEDIUM if not repaired else Confidence.LOW
    if score >= 0.50:
        return Confidence.LOW
    return Confidence.REJECT


def _rejected(raw: str, *, errors: tuple[str, ...]) -> ParsedPlate:
    return ParsedPlate(
        system=PlateSystem.UNKNOWN,
        canonical="",
        display=raw,
        is_valid=False,
        confidence=Confidence.REJECT,
        errors=errors,
    )


__all__ = [
    "parse",
    "canonicalise",
    "plate_from_tokens",
    "tokenize_devanagari",
    "tokenize_latin",
]
