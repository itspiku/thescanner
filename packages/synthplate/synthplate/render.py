"""Render Nepali plates from the specification.

Renders a clean, geometrically correct plate image. Degradation -- blur, noise,
perspective, weather -- is applied separately in :mod:`synthplate.degrade`, so
that the clean render can also serve as the ground-truth target for
super-resolution training and as a visual check that the specification is being
followed.

Every render returns per-character bounding boxes alongside the image. That is
free supervision: it gives character-level detection labels and attention
targets that would otherwise cost a great deal to annotate by hand, and it lets
the degradation stage track where the glyphs went under a perspective warp.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Final, Sequence

from PIL import Image, ImageDraw, ImageFilter

from nepal_plate import PlateColour, PlateSystem, spec

from . import fonts
from .sampling import PlateSample

# ---------------------------------------------------------------------------
# Colour schemes
# ---------------------------------------------------------------------------

#: (background, foreground) RGB. Legacy plates are hand-painted in practice, so
#: these are centres of a distribution rather than exact values -- see
#: ``_jitter_colour``.
COLOUR_RGB: Final[dict[PlateColour, tuple[tuple[int, int, int], tuple[int, int, int]]]] = {
    PlateColour.RED_WHITE:    ((178, 32, 38),  (245, 245, 245)),
    PlateColour.BLACK_WHITE:  ((28, 28, 30),   (242, 242, 242)),
    PlateColour.WHITE_RED:    ((243, 241, 236), (176, 26, 32)),
    PlateColour.YELLOW_BLACK: ((228, 178, 26), (24, 24, 24)),
    PlateColour.GREEN_WHITE:  ((18, 104, 58),  (243, 243, 243)),
    PlateColour.BLUE_WHITE:   ((26, 58, 142),  (243, 243, 243)),
    PlateColour.WHITE_BLACK:  ((246, 246, 242), (22, 22, 22)),
}

#: Nepal flag colours, for the embossed plate's left strip.
FLAG_CRIMSON: Final[tuple[int, int, int]] = (220, 20, 60)
FLAG_BLUE: Final[tuple[int, int, int]] = (0, 56, 147)
NEP_BLUE: Final[tuple[int, int, int]] = (0, 56, 147)

#: Plate aspect ratios (width / height), from the published dimension tables.
#: Single-row plates are long and shallow (45x11 cm cars, 52x11 heavy);
#: two-row plates are much squarer (30x18.5 cm car rear, 24x13 three-wheeler).
ASPECT_SINGLE_ROW: Final[tuple[float, float]] = (3.9, 4.8)
ASPECT_TWO_ROW: Final[tuple[float, float]] = (1.55, 1.95)

#: Supersampling factor. Rendering large and downsampling gives proper
#: antialiasing, which matters because the recogniser must not learn to key off
#: aliasing artefacts that real cameras do not produce.
SUPERSAMPLE: Final[int] = 4


@dataclass(frozen=True, slots=True)
class CharBox:
    """One rendered glyph and where it landed, in final image pixels."""

    token: str
    x0: float
    y0: float
    x1: float
    y1: float
    row: int


@dataclass(slots=True)
class RenderedPlate:
    """A clean plate render plus its ground truth."""

    image: Image.Image
    sample: PlateSample
    char_boxes: tuple[CharBox, ...]
    #: Plate corners, clockwise from top-left, in image pixels. Tracked through
    #: the perspective warp so the degraded image still has a correct polygon.
    corners: tuple[tuple[float, float], ...]
    font_authentic: bool
    meta: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

def _rows_for(sample: PlateSample) -> list[list[str]]:
    """Split a plate's tokens into physical rows.

    Legacy two-row plates put zone/lot/class on top and the serial below --
    that is how a Nepali motorcycle plate reads. Embossed two-row plates put
    province and class on top, series and serial below.
    """
    plate = sample.plate
    toks = list(sample.tokens)

    if not sample.two_row:
        return [toks]

    if plate.system is PlateSystem.DEVANAGARI:
        # Everything up to and including the class letter goes on row 1.
        cls_index = max(
            i for i, t in enumerate(toks) if t in spec.CLASS_BY_DEVA
        )
        return [toks[: cls_index + 1], toks[cls_index + 1 :]]

    # Embossed: province + class letter (+ subclass) on top.
    head = 2
    if len(toks) > 2 and toks[2].isdigit():
        head = 3
    return [toks[:head], toks[head:]]


def _jitter_colour(rgb: tuple[int, int, int], rng: random.Random, amount: int) -> tuple[int, int, int]:
    """Perturb a colour.

    Legacy plates are frequently hand-painted and fade in the sun, so the six
    schemes are families rather than exact values. A model trained on the exact
    RGB triples would key off values that real plates rarely hit -- and the
    colour classifier feeds the decoding prior, so getting this wrong is
    expensive.
    """
    return tuple(  # type: ignore[return-value]
        max(0, min(255, c + rng.randint(-amount, amount))) for c in rgb
    )


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def render(
    sample: PlateSample,
    *,
    height: int = 96,
    rng: random.Random | None = None,
    colour_jitter: int = 14,
) -> RenderedPlate:
    """Render ``sample`` as a clean plate image of the given pixel height."""
    rng = rng or random.Random()
    plate = sample.plate
    scheme = sample.colour if sample.colour in COLOUR_RGB else PlateColour.WHITE_BLACK
    bg, fg = COLOUR_RGB[scheme]
    bg = _jitter_colour(bg, rng, colour_jitter)
    fg = _jitter_colour(fg, rng, colour_jitter // 2)

    lo, hi = ASPECT_TWO_ROW if sample.two_row else ASPECT_SINGLE_ROW
    aspect = rng.uniform(lo, hi)

    H = height * SUPERSAMPLE
    W = int(round(H * aspect))
    embossed = plate.system is PlateSystem.EMBOSSED
    script = "latin" if embossed else "devanagari"

    img = Image.new("RGB", (W, H), bg)
    draw = ImageDraw.Draw(img)

    # Border: a painted rim on legacy plates, an embossed rim on new ones.
    rim = max(2, H // 32)
    draw.rectangle([rim, rim, W - rim - 1, H - rim - 1], outline=fg, width=max(1, rim // 2))

    content_x0 = rim * 3
    content_x1 = W - rim * 3

    # Embossed plates carry a left strip with the Nepal flag and a blue "NEP".
    if embossed:
        # Proportional to height, not width: the strip is a fixed physical
        # width on a real plate, so on a squarer two-row plate it takes up a
        # larger fraction of the width.
        strip_w = int(min(W * 0.16, max(H * 0.30, W * 0.085)))
        _draw_nep_strip(img, draw, strip_w, H, rng)
        content_x0 = strip_w + rim * 2

    rows = _rows_for(sample)
    boxes = _draw_rows(
        img, draw, rows, content_x0, content_x1, H, fg, script, embossed, rng,
        breaks=_field_breaks(sample, rows),
    )

    if embossed:
        img = _apply_emboss_relief(img, rng)

    # Downsample to the requested size with a good filter.
    final = img.resize((max(1, W // SUPERSAMPLE), height), Image.Resampling.LANCZOS)
    scale = 1.0 / SUPERSAMPLE
    scaled_boxes = tuple(
        CharBox(b.token, b.x0 * scale, b.y0 * scale, b.x1 * scale, b.y1 * scale, b.row)
        for b in boxes
    )
    w, h = final.size
    corners = ((0.0, 0.0), (float(w), 0.0), (float(w), float(h)), (0.0, float(h)))

    latin_font = fonts.resolve("latin")
    deva_font = fonts.resolve("devanagari")
    return RenderedPlate(
        image=final,
        sample=sample,
        char_boxes=scaled_boxes,
        corners=corners,
        font_authentic=(latin_font if embossed else deva_font).authentic,
        meta={
            "colour_scheme": scheme.value,
            "aspect": round(aspect, 3),
            "two_row": sample.two_row,
            "system": plate.system.value,
            "font": (latin_font if embossed else deva_font).name,
            "bg_rgb": bg,
            "fg_rgb": fg,
        },
    )


def _field_breaks(sample: PlateSample, rows: Sequence[Sequence[str]]) -> list[list[bool]]:
    """Mark, per row, which glyphs start a new plate field.

    Real plates put visible space between zone, lot, class and serial -- the
    groups read as separate units, not as one string. Reproducing that matters
    for more than looks: the recogniser sees those gaps as segmentation cues,
    and a generator that renders everything evenly spaced trains a model that
    has never seen the real layout.
    """
    from nepal_plate.grammar import GRAMMARS

    grammar = GRAMMARS.get(sample.plate.system)
    seg = grammar.segment(list(sample.tokens)) if grammar else None
    if seg is None:
        return [[False] * len(r) for r in rows]

    # Field name for each token, in order.
    per_token: list[str] = []
    for slot in grammar.slots:
        per_token.extend([slot.name] * len(seg.get(slot.name, [])))

    breaks: list[list[bool]] = []
    i = 0
    prev = None
    for row in rows:
        row_breaks = []
        for j in range(len(row)):
            name = per_token[i] if i < len(per_token) else None
            # The embossed subclass digit is part of its class letter -- "J2"
            # reads as one unit on a real plate, not as "J" then "2".
            if name == "subclass":
                name = "class_letter"
            row_breaks.append(j > 0 and name != prev)
            prev = name
            i += 1
        breaks.append(row_breaks)
    return breaks


def _draw_rows(
    img: Image.Image,
    draw: ImageDraw.ImageDraw,
    rows: Sequence[Sequence[str]],
    x0: int,
    x1: int,
    H: int,
    fg: tuple[int, int, int],
    script: str,
    embossed: bool,
    rng: random.Random,
    breaks: Sequence[Sequence[bool]] | None = None,
) -> list[CharBox]:
    """Lay out and draw each row, returning per-glyph boxes."""
    boxes: list[CharBox] = []
    n_rows = len(rows)
    row_h = H // n_rows
    # Devanagari needs more vertical headroom than Latin: matras sit above the
    # headline and descenders below it, so the nominal em box under-reports.
    fill = 0.60 if script == "devanagari" else 0.72
    if n_rows == 1:
        fill *= 0.95

    for r, row in enumerate(rows):
        if not row:
            continue
        target_h = int(row_h * fill)
        font_size = max(8, target_h)
        font = fonts.load(script, font_size)

        row_breaks = list(breaks[r]) if breaks and r < len(breaks) else [False] * len(row)
        avail = x1 - x0

        def _layout(fsize: int):
            """Widths and gaps at a given font size. Returns (font, widths, gaps)."""
            f = fonts.load(script, fsize)
            # Advance width, not ink extent: spacing by ink makes narrow glyphs
            # (Devanagari १, Latin 1) sit too close to their neighbours.
            w = [max(1.0, draw.textlength(t, font=f)) for t in row]
            base = max(3.0, fsize * 0.16)
            g = [base * (2.4 if brk else 1.0) for brk in row_breaks]
            g[0] = 0.0
            return f, w, g

        # Shrink to fit the available width.
        for _ in range(30):
            font, widths, gaps = _layout(font_size)
            if sum(widths) + sum(gaps) <= avail or font_size <= 8:
                break
            font_size = int(font_size * 0.94)

        font, widths, gaps = _layout(font_size)
        total = sum(widths) + sum(gaps)
        cx = x0 + (avail - total) / 2
        cy = r * row_h + row_h / 2

        for tok, w, g in zip(row, widths, gaps):
            cx += g
            bbox = draw.textbbox((0, 0), tok, font=font, anchor="lm")
            draw.text((cx, cy), tok, font=font, fill=fg, anchor="lm")
            boxes.append(
                CharBox(
                    token=tok,
                    x0=cx + bbox[0],
                    y0=cy + bbox[1],
                    x1=cx + bbox[2],
                    y1=cy + bbox[3],
                    row=r,
                )
            )
            cx += w
    return boxes


def _draw_nep_strip(
    img: Image.Image,
    draw: ImageDraw.ImageDraw,
    strip_w: int,
    H: int,
    rng: random.Random,
) -> None:
    """The blue left strip: Nepal flag above, ``NEP`` below.

    The flag is drawn as its distinctive double-pennon silhouette rather than a
    rectangle -- at plate resolution it reduces to a crimson triangle-pair with
    a blue edge, and that shape is what a detector will actually key off.
    """
    pad = strip_w // 6
    flag_h = int(H * 0.5)
    top = pad

    # Double pennon: two stacked triangles, blue border, crimson field.
    w = strip_w - 2 * pad
    upper = [(pad, top), (pad + w, top + flag_h * 0.30), (pad, top + flag_h * 0.52)]
    lower = [(pad, top + flag_h * 0.40), (pad + w, top + flag_h * 0.78), (pad, top + flag_h)]
    for poly in (upper, lower):
        draw.polygon(poly, fill=FLAG_CRIMSON, outline=FLAG_BLUE)

    # "NEP" beneath, sized to fit the strip's *width*. Sizing it from the plate
    # height overflows the strip on two-row plates, where the plate is much
    # narrower for the same height, and the text then collides with the
    # registration rows.
    size = max(6, int(H * 0.20))
    target = strip_w * 0.86
    for _ in range(24):
        font = fonts.load("latin", size)
        if draw.textlength("NEP", font=font) <= target or size <= 6:
            break
        size = int(size * 0.9)
    draw.text(
        (strip_w / 2, top + flag_h + (H - top - flag_h) * 0.45),
        "NEP",
        font=fonts.load("latin", size),
        fill=NEP_BLUE,
        anchor="mm",
    )


def _apply_emboss_relief(img: Image.Image, rng: random.Random) -> Image.Image:
    """Fake the raised-character relief of an embossed plate.

    Characters on an embossed plate stand proud of the surface, so under
    overhead illumination they carry a bright edge on one side and a shadow on
    the other. This is not cosmetic: the relief is a large part of what a plate
    looks like under a headlight or an IR illuminator at night, and a model
    trained on flat printed characters will not have seen it.

    Implemented as a cheap emboss convolution blended over the original, which
    is enough to produce the edge highlight/shadow pair without a full
    lighting model.
    """
    relief = img.filter(ImageFilter.EMBOSS)
    return Image.blend(img, relief.convert("RGB"), alpha=rng.uniform(0.10, 0.22))


__all__ = ["CharBox", "RenderedPlate", "render", "COLOUR_RGB"]
