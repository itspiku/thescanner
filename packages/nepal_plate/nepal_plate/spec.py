"""Authoritative reference tables for Nepali vehicle registration plates.

This module is the single source of truth for the *domain* -- every other part
of the system (synthetic renderer, recogniser vocabulary, grammar decoder,
database enums, UI labels) derives from here. Change a fact once, and it
propagates.

Nepal currently operates **two incompatible plate systems side by side**:

1. **Legacy zonal plates** -- Devanagari script, laid out as
   ``<zone> <lot> <class> <serial>`` e.g. ``बा १ च १२३४`` ("Ba 1 Cha 1234").
   The *background colour* encodes ownership (red = private, black = public,
   ...) and so does the class letter. That redundancy is the single most useful
   structural prior available to a recogniser and this system exploits it.

2. **Embossed plates** (rollout from 2020) -- Latin FE-Schrift, uniformly
   black-on-white, laid out as ``<province> <class> <series> <serial>`` with a
   left-hand strip carrying the Nepal flag and a blue ``NEP``. An RFID chip is
   embedded. Because every embossed plate is black-on-white, colour no longer
   carries ownership -- the class letter does.

Both systems are legal on the road, so a production ANPR for Nepal must read
both and must never confuse one grammar for the other.

References are recorded in ``docs/research/plate-specification.md``.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Final, Mapping

from .types import Ownership, PlateColour, PlateSystem, SizeClass

# ---------------------------------------------------------------------------
# Devanagari numerals
# ---------------------------------------------------------------------------

#: Devanagari digit glyphs, index == numeric value (U+0966..U+096F).
DEVA_DIGITS: Final[tuple[str, ...]] = (
    "०", "१", "२", "३", "४",
    "५", "६", "७", "८", "९",
)

DEVA_TO_ASCII_DIGIT: Final[Mapping[str, str]] = MappingProxyType(
    {d: str(i) for i, d in enumerate(DEVA_DIGITS)}
)
ASCII_TO_DEVA_DIGIT: Final[Mapping[str, str]] = MappingProxyType(
    {str(i): d for i, d in enumerate(DEVA_DIGITS)}
)


# ---------------------------------------------------------------------------
# Legacy system: zones
# ---------------------------------------------------------------------------

class Zone:
    """A legacy administrative zone that issued plates.

    Nepal's 14 zones were abolished as administrative units in 2015 when the
    country moved to 7 provinces, but zone-coded plates remain in circulation
    and remain legal, so the codes stay operationally meaningful.
    """

    __slots__ = ("deva", "roman", "name", "province")

    def __init__(self, deva: str, roman: str, name: str, province: int) -> None:
        self.deva = deva
        self.roman = roman
        self.name = name
        #: Province the zone's territory now mostly falls under. Approximate --
        #: several zones straddle the new provincial boundaries. Used only for
        #: coarse geographic analytics, never for validation.
        self.province = province

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"Zone({self.roman!r}, {self.deva!r})"


ZONES: Final[tuple[Zone, ...]] = (
    Zone("मे", "ME", "Mechi", 1),           # मे
    Zone("को", "KO", "Koshi", 1),           # को
    Zone("स", "SA", "Sagarmatha", 1),            # स
    Zone("ज", "JA", "Janakpur", 2),              # ज
    Zone("बा", "BA", "Bagmati", 3),         # बा
    Zone("ना", "NA", "Narayani", 3),        # ना
    Zone("ग", "GA", "Gandaki", 4),               # ग
    Zone("लु", "LU", "Lumbini", 5),         # लु
    Zone("ध", "DHA", "Dhawalagiri", 4),          # ध
    Zone("रा", "RA", "Rapti", 5),           # रा
    Zone("भे", "BHE", "Bheri", 6),          # भे
    Zone("क", "KA", "Karnali", 6),               # क
    Zone("से", "SE", "Seti", 7),            # से
    Zone("म", "MA", "Mahakali", 7),              # म
)

ZONE_BY_DEVA: Final[Mapping[str, Zone]] = MappingProxyType({z.deva: z for z in ZONES})
ZONE_BY_ROMAN: Final[Mapping[str, Zone]] = MappingProxyType({z.roman: z for z in ZONES})
ZONE_TOKENS: Final[tuple[str, ...]] = tuple(z.deva for z in ZONES)


# ---------------------------------------------------------------------------
# Legacy system: vehicle class letters
# ---------------------------------------------------------------------------

class VehicleClass:
    """A legacy Devanagari class letter.

    The letter jointly encodes *ownership* and *size band*. Combined with the
    plate colour (which encodes ownership alone) this gives two independent
    observations of the same latent variable -- the decoder uses the
    disagreement between them as a fraud/ misread signal.
    """

    __slots__ = ("deva", "roman", "ownership", "size")

    def __init__(self, deva: str, roman: str, ownership: Ownership, size: SizeClass) -> None:
        self.deva = deva
        self.roman = roman
        self.ownership = ownership
        self.size = size

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"VehicleClass({self.roman!r}, {self.ownership.value}, {self.size.value})"


#: Sentinel token for the diplomatic ``सी.डी.`` marking. It is rendered as a
#: multi-glyph cluster with full stops and is treated as one atomic vocabulary
#: entry rather than four, because that is how it appears visually.
CD_TOKEN: Final[str] = "सी.डी."  # सी.डी.

VEHICLE_CLASSES: Final[tuple[VehicleClass, ...]] = (
    # Private -- red plate, white text
    VehicleClass("क", "KA", Ownership.PRIVATE, SizeClass.HEAVY),        # क
    VehicleClass("च", "CHA", Ownership.PRIVATE, SizeClass.LIGHT),       # च
    VehicleClass("प", "PA", Ownership.PRIVATE, SizeClass.MOTORCYCLE),   # प
    # Public / commercial -- black plate, white text
    VehicleClass("ख", "KHA", Ownership.PUBLIC, SizeClass.HEAVY),        # ख
    VehicleClass("ज", "JA", Ownership.PUBLIC, SizeClass.LIGHT),         # ज
    VehicleClass("फ", "PHA", Ownership.PUBLIC, SizeClass.MOTORCYCLE),   # फ
    # Government -- white plate, red text
    VehicleClass("ग", "GA", Ownership.GOVERNMENT, SizeClass.HEAVY),     # ग
    VehicleClass("झ", "JHA", Ownership.GOVERNMENT, SizeClass.LIGHT),    # झ
    VehicleClass("ब", "BA", Ownership.GOVERNMENT, SizeClass.MOTORCYCLE),  # ब
    # National corporation -- yellow plate, black text
    VehicleClass("घ", "GHA", Ownership.CORPORATION, SizeClass.HEAVY),   # घ
    VehicleClass("ञ", "NYA", Ownership.CORPORATION, SizeClass.LIGHT),   # ञ
    # Tourist -- green plate, white text
    VehicleClass("य", "YA", Ownership.TOURIST, SizeClass.ANY),          # य
    # Diplomatic -- blue plate, white text
    VehicleClass(CD_TOKEN, "CD", Ownership.DIPLOMATIC, SizeClass.ANY),
)

CLASS_BY_DEVA: Final[Mapping[str, VehicleClass]] = MappingProxyType(
    {c.deva: c for c in VEHICLE_CLASSES}
)
CLASS_BY_ROMAN: Final[Mapping[str, VehicleClass]] = MappingProxyType(
    {c.roman: c for c in VEHICLE_CLASSES}
)
CLASS_TOKENS: Final[tuple[str, ...]] = tuple(c.deva for c in VEHICLE_CLASSES)


# ---------------------------------------------------------------------------
# Legacy system: colour <-> ownership
# ---------------------------------------------------------------------------

#: Colour scheme -> ownership it denotes on a *legacy* plate.
LEGACY_COLOUR_OWNERSHIP: Final[Mapping[PlateColour, Ownership]] = MappingProxyType({
    PlateColour.RED_WHITE: Ownership.PRIVATE,
    PlateColour.BLACK_WHITE: Ownership.PUBLIC,
    PlateColour.WHITE_RED: Ownership.GOVERNMENT,
    PlateColour.YELLOW_BLACK: Ownership.CORPORATION,
    PlateColour.GREEN_WHITE: Ownership.TOURIST,
    PlateColour.BLUE_WHITE: Ownership.DIPLOMATIC,
})

#: Ownership -> the colour schemes that may legally carry it.
OWNERSHIP_COLOURS: Final[Mapping[Ownership, tuple[PlateColour, ...]]] = MappingProxyType({
    Ownership.PRIVATE: (PlateColour.RED_WHITE, PlateColour.WHITE_BLACK),
    Ownership.PUBLIC: (PlateColour.BLACK_WHITE, PlateColour.WHITE_BLACK),
    Ownership.GOVERNMENT: (PlateColour.WHITE_RED, PlateColour.WHITE_BLACK),
    Ownership.CORPORATION: (PlateColour.YELLOW_BLACK, PlateColour.WHITE_BLACK),
    Ownership.TOURIST: (PlateColour.GREEN_WHITE, PlateColour.WHITE_BLACK),
    Ownership.DIPLOMATIC: (PlateColour.BLUE_WHITE, PlateColour.WHITE_BLACK),
    Ownership.UNKNOWN: tuple(PlateColour),
})

#: Ownership -> the legacy class letters consistent with it. This is the table
#: the colour prior actually constrains against.
OWNERSHIP_CLASS_TOKENS: Final[Mapping[Ownership, tuple[str, ...]]] = MappingProxyType({
    own: tuple(c.deva for c in VEHICLE_CLASSES if c.ownership is own)
    for own in Ownership
    if own is not Ownership.UNKNOWN
})


# ---------------------------------------------------------------------------
# Embossed system: provinces
# ---------------------------------------------------------------------------

class Province:
    __slots__ = ("number", "name", "deva", "abbr")

    def __init__(self, number: int, name: str, deva: str, abbr: str) -> None:
        self.number = number
        self.name = name
        self.deva = deva
        self.abbr = abbr

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"Province({self.number}, {self.name!r})"


PROVINCES: Final[tuple[Province, ...]] = (
    Province(1, "Koshi", "कोशी", "KOS"),
    Province(2, "Madhesh", "मधेश", "MAD"),
    Province(3, "Bagmati", "बागमती", "BAG"),
    Province(4, "Gandaki", "गण्डकी", "GAN"),
    Province(5, "Lumbini", "लुम्बिनी", "LUM"),
    Province(6, "Karnali", "कर्णाली", "KAR"),
    Province(7, "Sudurpashchim", "सुदूरपश्चिम", "SUD"),
)

PROVINCE_BY_NUMBER: Final[Mapping[int, Province]] = MappingProxyType(
    {p.number: p for p in PROVINCES}
)


# ---------------------------------------------------------------------------
# Embossed system: class letters
# ---------------------------------------------------------------------------

#: Base class letter -> human description and size band.
EMBOSSED_CLASSES: Final[Mapping[str, tuple[str, SizeClass]]] = MappingProxyType({
    "A":  ("Motorcycle, Scooter, Moped", SizeClass.MOTORCYCLE),
    "B":  ("Car, Jeep, Cargo/Delivery Van", SizeClass.LIGHT),
    "C":  ("Tempo, Auto Rickshaw", SizeClass.LIGHT),
    "C1": ("E-Rickshaw", SizeClass.LIGHT),
    "D":  ("Power Tiller", SizeClass.HEAVY),
    "E":  ("Tractor", SizeClass.HEAVY),
    "F":  ("Minibus, Mini Truck", SizeClass.HEAVY),
    "G":  ("Truck, Bus, Lorry", SizeClass.HEAVY),
    "H":  ("Road Roller, Dozer", SizeClass.HEAVY),
    "H1": ("Dozer", SizeClass.HEAVY),
    "H2": ("Road Roller", SizeClass.HEAVY),
    "I":  ("Crane, Fire Brigade, Loader", SizeClass.HEAVY),
    "I1": ("Crane", SizeClass.HEAVY),
    "I2": ("Fire Brigade", SizeClass.HEAVY),
    "I3": ("Loader", SizeClass.HEAVY),
    "J":  ("Heavy Equipment", SizeClass.HEAVY),
    "J1": ("Excavator", SizeClass.HEAVY),
    "J2": ("Backhoe Loader", SizeClass.HEAVY),
    "J3": ("Grader", SizeClass.HEAVY),
    "J4": ("Forklift", SizeClass.HEAVY),
    "J5": ("Other Heavy Equipment", SizeClass.HEAVY),
    "K":  ("Scooter, Moped", SizeClass.MOTORCYCLE),
})

EMBOSSED_BASE_LETTERS: Final[tuple[str, ...]] = tuple("ABCDEFGHIJK")

#: Which base letters admit a numeric subclass, and which digits are valid.
EMBOSSED_SUBCLASSES: Final[Mapping[str, tuple[str, ...]]] = MappingProxyType({
    "C": ("1",),
    "H": ("1", "2"),
    "I": ("1", "2", "3"),
    "J": ("1", "2", "3", "4", "5"),
})


# ---------------------------------------------------------------------------
# Recogniser vocabularies
# ---------------------------------------------------------------------------

#: The 34-token Devanagari plate vocabulary: 10 numerals plus 24 distinct
#: letter tokens (zone codes and class letters, deduplicated -- exactly three
#: glyphs, क / ग / ज, serve as both a zone code and a class letter and are
#: the same glyph in both roles; position in the grammar disambiguates them).
#:
#: This count independently reproduces the 34-class charset used by published
#: Nepali plate-OCR datasets, which is a useful sanity check on the derivation.
DEVA_LETTER_TOKENS: Final[tuple[str, ...]] = tuple(
    dict.fromkeys(ZONE_TOKENS + CLASS_TOKENS)
)
DEVA_VOCAB: Final[tuple[str, ...]] = DEVA_DIGITS + DEVA_LETTER_TOKENS

#: Latin vocabulary for embossed plates. FE-Schrift renders all 26 letters and
#: 10 digits; series letters are unconstrained so the full set is needed.
LATIN_LETTERS: Final[tuple[str, ...]] = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
LATIN_DIGITS: Final[tuple[str, ...]] = tuple("0123456789")
LATIN_VOCAB: Final[tuple[str, ...]] = LATIN_DIGITS + LATIN_LETTERS

#: Unified vocabulary used by the single dual-script recogniser head. The blank
#: symbol for CTC is index 0 and is *not* part of the vocabulary proper.
CTC_BLANK: Final[str] = "∅"  # ∅
UNIFIED_VOCAB: Final[tuple[str, ...]] = (CTC_BLANK,) + DEVA_VOCAB + LATIN_VOCAB
VOCAB_INDEX: Final[Mapping[str, int]] = MappingProxyType(
    {tok: i for i, tok in enumerate(UNIFIED_VOCAB)}
)


# ---------------------------------------------------------------------------
# Confusability graph
# ---------------------------------------------------------------------------
#
# Under motion blur, low resolution and IR illumination, certain glyph pairs
# collapse toward each other. The decoder uses these sets to decide *where* it
# is cheap to apply a grammar repair: substituting a glyph for a known
# confusable costs far less than substituting an unrelated one. Without this,
# grammar repair happily invents plates.

#: Symmetric confusion groups for Devanagari plate glyphs.
DEVA_CONFUSION_GROUPS: Final[tuple[tuple[str, ...], ...]] = (
    ("ग", "ध", "घ"),              # ग ध घ  -- shared bowl + stem
    ("च", "ज", "ञ"),              # च ज ञ  -- shared left hook
    ("प", "फ", "य"),              # प फ य  -- differ by one stroke
    ("ब", "भ", "क"),              # ब भ क
    ("क", "ख"),                        # क ख
    ("म", "भ"),                        # म भ
    ("स", "ग"),                        # स ग
    ("२", "३"),                        # २ ३
    ("४", "५"),                        # ४ ५
    ("६", "७", "९"),              # ६ ७ ९
    ("०", "९"),                        # ० ९
    ("१", "३"),                        # १ ३
    ("८", "६"),                        # ८ ६
)

#: Latin/FE-Schrift confusion groups.
#:
#: FE-Schrift ("fälschungserschwerende Schrift" -- forgery-impeding script) was
#: specifically engineered so that no glyph can be altered into another, which
#: also makes it unusually robust to blur. The residual groups below are the
#: ones that survive heavy degradation, and they are notably fewer than for a
#: normal typeface -- embossed plates are the easier half of this problem.
LATIN_CONFUSION_GROUPS: Final[tuple[tuple[str, ...], ...]] = (
    ("0", "O", "D", "Q"),
    ("1", "I", "L"),
    ("8", "B"),
    ("5", "S"),
    ("2", "Z"),
    ("6", "G"),
    ("U", "V"),
    ("7", "T"),
    ("E", "F"),
    ("C", "G"),
)


def _build_confusion_map(
    groups: tuple[tuple[str, ...], ...]
) -> Mapping[str, frozenset[str]]:
    out: dict[str, set[str]] = {}
    for group in groups:
        for tok in group:
            out.setdefault(tok, set()).update(t for t in group if t != tok)
    return MappingProxyType({k: frozenset(v) for k, v in out.items()})


CONFUSABLE: Final[Mapping[str, frozenset[str]]] = MappingProxyType({
    **_build_confusion_map(DEVA_CONFUSION_GROUPS),
    **_build_confusion_map(LATIN_CONFUSION_GROUPS),
})


def are_confusable(a: str, b: str) -> bool:
    """True when ``a`` and ``b`` are known to collapse under degradation."""
    return b in CONFUSABLE.get(a, frozenset())


# ---------------------------------------------------------------------------
# Derived convenience lookups
# ---------------------------------------------------------------------------

def system_for_colour(colour: PlateColour) -> PlateSystem:
    """Infer which plate grammar a colour scheme implies.

    White-on-black is the embossed scheme; the six legacy schemes are all
    distinct from it. This is a cheap, highly reliable router that runs before
    OCR and halves the decoder's search space.
    """
    if colour is PlateColour.WHITE_BLACK:
        return PlateSystem.EMBOSSED
    if colour in LEGACY_COLOUR_OWNERSHIP:
        return PlateSystem.DEVANAGARI
    return PlateSystem.UNKNOWN


def ownership_for_class_token(token: str) -> Ownership:
    cls = CLASS_BY_DEVA.get(token)
    return cls.ownership if cls else Ownership.UNKNOWN


def size_for_class_token(token: str) -> SizeClass:
    cls = CLASS_BY_DEVA.get(token)
    return cls.size if cls else SizeClass.UNKNOWN
