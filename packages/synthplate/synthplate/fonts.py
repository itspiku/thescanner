"""Font resolution for plate rendering.

This is a small module with a large failure mode attached, so it is explicit
about what it is doing.

Nepali embossed plates use **FE-Schrift**, a typeface engineered so that no
character can be physically altered into another. Legacy plates use a
Devanagari plate face. If the generator renders with the wrong typeface, the
recogniser learns the wrong glyph shapes, and it will do so *silently* -- the
synthetic data will look plausible and the model will underperform on real
plates for reasons that are very hard to diagnose after the fact.

So: the correct fonts are resolved if present, a clearly-labelled fallback is
used if not, and **the font actually used is recorded in every sample's
metadata** and surfaced as a warning. Never ship a model trained on fallback
fonts.

Run ``python -m synthplate.fetch_fonts`` to install the real ones.
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Final

from PIL import ImageFont

#: Where bundled fonts live, relative to the repository root.
ASSET_DIR: Final[Path] = Path(__file__).resolve().parents[3] / "assets" / "fonts"

#: Overridable via environment for CI and containers.
ENV_LATIN: Final[str] = "SYNTHPLATE_LATIN_FONT"
ENV_DEVANAGARI: Final[str] = "SYNTHPLATE_DEVANAGARI_FONT"

#: Filenames we look for, in preference order.
_LATIN_CANDIDATES: Final[tuple[str, ...]] = (
    "FE-FONT.TTF",
    "FE-Schrift.ttf",
    "EuroPlate.ttf",
)
_DEVANAGARI_CANDIDATES: Final[tuple[str, ...]] = (
    "NotoSansDevanagari-Bold.ttf",
    "NotoSansDevanagari-SemiBold.ttf",
    "NotoSansDevanagari-Regular.ttf",
    "Mangal.ttf",
)

#: Last-resort system fonts, by platform. Arial Narrow Bold is a poor stand-in
#: for FE-Schrift -- narrower, different terminals, different digit shapes --
#: but it is at least condensed and bold. Nirmala UI is a genuine Devanagari
#: face and a reasonable stand-in.
_SYSTEM_FALLBACKS: Final[dict[str, tuple[str, ...]]] = {
    "latin": (
        "C:/Windows/Fonts/ARIALNB.TTF",
        "C:/Windows/Fonts/arialbd.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ),
    "devanagari": (
        "C:/Windows/Fonts/Nirmala.ttc",
        "C:/Windows/Fonts/mangal.ttf",
        "/usr/share/fonts/truetype/noto/NotoSansDevanagari-Regular.ttf",
        "/usr/share/fonts/truetype/lohit-devanagari/Lohit-Devanagari.ttf",
    ),
}


@dataclass(frozen=True, slots=True)
class ResolvedFont:
    """A font file plus whether it is the real thing."""

    path: Path
    #: False when we fell back to a system font. Recorded in sample metadata.
    authentic: bool
    script: str

    @property
    def name(self) -> str:
        return self.path.name


def _first_existing(paths) -> Path | None:
    for p in paths:
        path = Path(p)
        if path.is_file():
            return path
    return None


@lru_cache(maxsize=8)
def resolve(script: str) -> ResolvedFont:
    """Resolve the font for ``script`` ("latin" or "devanagari")."""
    if script == "latin":
        env, candidates = ENV_LATIN, _LATIN_CANDIDATES
    elif script == "devanagari":
        env, candidates = ENV_DEVANAGARI, _DEVANAGARI_CANDIDATES
    else:
        raise ValueError(f"unknown script {script!r}")

    override = os.environ.get(env)
    if override and Path(override).is_file():
        return ResolvedFont(Path(override), authentic=True, script=script)

    bundled = _first_existing(ASSET_DIR / c for c in candidates)
    if bundled is not None:
        return ResolvedFont(bundled, authentic=True, script=script)

    fallback = _first_existing(_SYSTEM_FALLBACKS[script])
    if fallback is None:
        raise RuntimeError(
            f"No usable {script} font found. Run 'python -m synthplate.fetch_fonts' "
            f"or set ${env} to a TrueType file."
        )
    warnings.warn(
        f"synthplate: using fallback {script} font {fallback.name!r} instead of the "
        f"authentic plate typeface. Synthetic data rendered this way is fine for "
        f"development but must NOT be used to train a deployed model -- run "
        f"'python -m synthplate.fetch_fonts' first.",
        stacklevel=2,
    )
    return ResolvedFont(fallback, authentic=False, script=script)


@lru_cache(maxsize=256)
def load(script: str, size: int) -> ImageFont.FreeTypeFont:
    """Load a sized PIL font for ``script``. Cached -- sizing is hot."""
    font = resolve(script)
    # Nirmala.ttc and other collections need an explicit face index.
    if font.path.suffix.lower() == ".ttc":
        return ImageFont.truetype(str(font.path), size=size, index=0)
    return ImageFont.truetype(str(font.path), size=size)


def fonts_are_authentic() -> bool:
    """True only when both scripts resolved to real plate typefaces.

    The corpus generator writes this into the dataset manifest, so a training
    run can refuse to proceed on fallback-rendered data.
    """
    try:
        return resolve("latin").authentic and resolve("devanagari").authentic
    except RuntimeError:
        return False


__all__ = ["ResolvedFont", "resolve", "load", "fonts_are_authentic", "ASSET_DIR"]
