"""Physically-grounded degradation of rendered plates.

A clean render teaches a model nothing useful. What determines whether
synthetic data transfers is the *degradation model* -- and specifically whether
it composes degradations the way a real imaging chain does.

Three principles.

**Random composition, not a fixed pipeline.** LPSRGAN's "n-stage random
combination degradation" observation is that a fixed sequence of degradations
teaches the model to invert that exact sequence. Randomly selecting *and
ordering* a subset from a pool produces artefacts the model cannot memorise.

**Order follows physics where it must.** Some stages commute and some do not.
Motion blur happens in the lens, before the sensor samples; noise is added by
the sensor; JPEG is applied last. Shuffling those produces artefacts no camera
can make -- a different way of teaching the model something false. So the pool
is partitioned into optical, sensor and codec phases, shuffled *within* each
phase and applied in physical order.

**One difficulty knob, not thirty independent ones.** This is the part that is
easy to get wrong. If every parameter is sampled independently across its full
range, the *product* is almost always hard: with ten degradations each
independently at 50% severity, virtually no sample comes out easy. The first
version of this module did exactly that and produced a corpus with a median
quality of 0.33, most of it unreadable noise -- useless for training, because a
model needs legible examples to learn glyph shapes at all before hard examples
can teach it anything about robustness.

So a single ``difficulty`` in [0, 1] is sampled per plate, and every parameter
range is interpolated from its benign end toward its severe end by that amount.
Difficulty is drawn in bands chosen to give a useful curriculum spread. The
resulting ``quality`` score is then a genuine, well-calibrated ground truth --
and it is the training target for the crop-quality estimator that drives fusion
weighting in ``nepal_plate.fuse``.
"""

from __future__ import annotations

import io
import math
import random
from dataclasses import dataclass, field, replace
from typing import Callable, Sequence

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

from .render import CharBox, RenderedPlate


@dataclass(frozen=True, slots=True)
class DegradationConfig:
    """Parameter ranges, expressed as ``(benign, severe)``.

    The value used for a plate at difficulty ``d`` is drawn from
    ``[benign, benign + d * (severe - benign)]``, so ``d = 0`` gives a clean
    image and ``d = 1`` opens the full range. Several ranges are *inverted*
    (severe is the smaller number) -- plate width and JPEG quality -- and
    :func:`_r` handles that transparently.

    The plate-width window below is the single most important setting: it
    decides whether the corpus teaches the model to read plates at the sizes it
    will actually see. A junction camera sees plates roughly 20-170 px wide,
    and nearly all of the difficulty lives below 60.
    """

    #: Sampling *window* for final plate width, as (at difficulty 0, at
    #: difficulty 1). Both ends slide downward with difficulty rather than the
    #: window merely widening -- widening alone leaves the median width high
    #: even at maximum difficulty, so the corpus never really visits the small
    #: plates where nearly all the difficulty lives.
    plate_width_hi: tuple[int, int] = (170, 68)
    plate_width_lo: tuple[int, int] = (115, 24)

    # Optical -- these are maximum magnitudes, scaled by difficulty.
    yaw_deg: float = 40.0
    pitch_deg: float = 26.0
    roll_deg: float = 11.0
    motion_blur_px: tuple[float, float] = (0.0, 7.0)
    defocus_px: tuple[float, float] = (0.0, 2.2)
    rolling_shutter_shear: tuple[float, float] = (0.0, 0.07)

    # Illumination / atmosphere
    brightness_spread: tuple[float, float] = (0.06, 0.45)
    contrast_spread: tuple[float, float] = (0.06, 0.35)
    haze: tuple[float, float] = (0.0, 0.34)
    glare_prob: tuple[float, float] = (0.04, 0.32)
    shadow_prob: tuple[float, float] = (0.05, 0.38)
    night_prob: tuple[float, float] = (0.10, 0.36)

    # Sensor
    gauss_noise: tuple[float, float] = (0.6, 6.5)
    #: Full-well photon count for shot noise, as (benign, severe). Shot noise
    #: is Poisson in the *photon* count, so this is photons at full scale:
    #: a bright daylight exposure collects thousands, a night frame at high
    #: gain collects tens. Fewer photons means more noise, hence the inversion.
    poisson_photons: tuple[float, float] = (6000.0, 90.0)
    chroma_shift_px: tuple[float, float] = (0.0, 1.1)

    # Codec
    jpeg_quality: tuple[int, int] = (95, 32)

    #: Optical-phase effects applied, at difficulty 0 and difficulty 1.
    n_optical: tuple[int, int] = (1, 4)

    #: Weights over the difficulty bands [0, 0.35), [0.35, 0.7), [0.7, 1.0].
    #: Deliberately front-loaded: a model needs enough legible examples to
    #: learn glyph shapes before hard ones teach it robustness.
    difficulty_weights: tuple[float, float, float] = (0.34, 0.44, 0.22)

    #: Random padding around the plate quad, as a fraction of plate width. Real
    #: detector crops include a little surrounding scene, never a hard edge.
    crop_pad: tuple[float, float] = (0.01, 0.09)


@dataclass(frozen=True, slots=True)
class Scene:
    """Conditions that hold constant while a vehicle crosses the frame.

    A camera sees a vehicle for ten to forty frames over a second or two.
    Whether it is night, how hazy the air is, where the pole shadow falls and
    what colour the bodywork behind the plate is do **not** change over that
    interval -- but per-frame sensor noise, exact motion smear and plate scale
    do.

    Sampling those independently per frame, as a naive track generator would,
    makes synthetic tracks trivially easy: averaging twelve frames with
    independent lighting recovers the plate immediately, whereas averaging
    twelve frames that are *all* dark and hazy does not. A fusion model trained
    on the former learns nothing that transfers.
    """

    night: bool
    haze: float
    surround: tuple[int, int, int]
    brightness: float
    contrast: float
    #: (angle, edge, softness, depth), or None for no shadow.
    shadow: tuple[float, float, float, float] | None
    #: (cx_frac, cy_frac, radius_frac, strength), or None for no glare.
    glare: tuple[float, float, float, float] | None
    night_desat: float = 0.15
    night_gain: float = 0.75


def sample_scene(rng: random.Random, cfg: "DegradationConfig", d: float) -> "Scene":
    """Draw one set of scene conditions, to be shared across a track."""
    shadow = None
    if rng.random() < _r(rng, cfg.shadow_prob, d):
        shadow = (
            rng.uniform(0, math.pi),
            rng.uniform(0.2, 0.8),
            rng.uniform(0.03, 0.25),
            rng.uniform(0.15, 0.30 + 0.30 * d),
        )
    glare = None
    if rng.random() < _r(rng, cfg.glare_prob, d):
        glare = (
            rng.uniform(0, 1),
            rng.uniform(0, 1),
            rng.uniform(0.15, 0.5),
            rng.uniform(30, 60 + 90 * d),
        )
    return Scene(
        night=rng.random() < _r(rng, cfg.night_prob, d),
        haze=_r(rng, cfg.haze, d),
        surround=tuple(rng.randint(40, 150) for _ in range(3)),
        brightness=1.0 + rng.uniform(-1, 1) * _r(rng, cfg.brightness_spread, d),
        contrast=1.0 + rng.uniform(-1, 1) * _r(rng, cfg.contrast_spread, d),
        shadow=shadow,
        glare=glare,
        night_desat=rng.uniform(0.0, 0.3),
        night_gain=rng.uniform(0.6, 0.9),
    )


@dataclass(slots=True)
class DegradedPlate:
    """A degraded plate plus the ground truth that survived the transformation."""

    image: Image.Image
    source: RenderedPlate
    char_boxes: tuple[CharBox, ...]
    corners: tuple[tuple[float, float], ...]
    #: 0 = unreadable, 1 = pristine. Training target for the quality estimator.
    quality: float
    #: The difficulty this sample was drawn at, in [0, 1].
    difficulty: float
    #: Degradations applied, in order, with their severities.
    applied: tuple[tuple[str, float], ...] = ()
    meta: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Difficulty scaling
# ---------------------------------------------------------------------------

def _r(rng: random.Random, span: tuple[float, float], d: float) -> float:
    """Sample a parameter from a range scaled by difficulty.

    Handles inverted ranges (where "severe" is the smaller number, as for JPEG
    quality and plate width) transparently.
    """
    benign, severe = span
    edge = benign + d * (severe - benign)
    return rng.uniform(min(benign, edge), max(benign, edge))


def sample_difficulty(rng: random.Random, weights: tuple[float, float, float]) -> float:
    """Draw a difficulty, banded so the corpus has a usable curriculum spread."""
    bands = ((0.0, 0.35), (0.35, 0.70), (0.70, 1.0))
    lo, hi = rng.choices(bands, weights=list(weights), k=1)[0]
    return rng.uniform(lo, hi)


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _homography(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Solve the 8-parameter homography taking ``src`` corners to ``dst``."""
    a: list[list[float]] = []
    b: list[float] = []
    for (x, y), (u, v) in zip(src, dst):
        a.append([x, y, 1, 0, 0, 0, -u * x, -u * y])
        a.append([0, 0, 0, x, y, 1, -v * x, -v * y])
        b.extend([u, v])
    h, *_ = np.linalg.lstsq(np.asarray(a, float), np.asarray(b, float), rcond=None)
    return np.append(h, 1.0).reshape(3, 3)


def _apply_h(h: np.ndarray, pts: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
    arr = np.asarray([[x, y, 1.0] for x, y in pts], float).T
    out = h @ arr
    out = out[:2] / np.where(np.abs(out[2]) < 1e-9, 1e-9, out[2])
    return [(float(x), float(y)) for x, y in out.T]


def _shift(arr: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """Translate an image array, replicating edges rather than wrapping.

    Wrapping (``np.roll``) smears the right edge of the plate onto the left --
    an artefact no camera produces, and one a model would happily learn.
    """
    if dy == 0 and dx == 0:
        return arr
    py, px = abs(dy), abs(dx)
    padded = np.pad(arr, ((py, py), (px, px), (0, 0)), mode="edge")
    return padded[py - dy : py - dy + arr.shape[0], px - dx : px - dx + arr.shape[1]]


def _perspective(
    img: Image.Image,
    boxes: Sequence[CharBox],
    corners: Sequence[tuple[float, float]],
    rng: random.Random,
    cfg: DegradationConfig,
    d: float,
    scene: Scene,
) -> tuple[Image.Image, tuple[CharBox, ...], tuple[tuple[float, float], ...], float]:
    """Warp the plate to a plausible viewing pose, then crop to the plate quad.

    Roadside cameras are mounted above the road and off to one side, so plates
    are almost never seen fronto-parallel.

    Two details that matter more than they look. The plate is first pasted onto
    a larger neutral surround, so the warp has something to rotate *into*;
    warping the bare plate leaves hard black wedges in the corners, which no
    real crop has and which a model learns to expect. And the result is cropped
    back to the plate's own quad plus a little padding, matching what a detector
    would actually hand the recogniser.
    """
    w, h = img.size
    canvas = Image.new("RGB", (int(w * 1.6), int(h * 1.6)), scene.surround)
    ox, oy = (canvas.width - w) // 2, (canvas.height - h) // 2
    canvas.paste(img, (ox, oy))
    W, H = canvas.size

    yaw = math.radians(rng.uniform(-1, 1) * cfg.yaw_deg * d)
    pitch = math.radians(rng.uniform(-1, 1) * cfg.pitch_deg * d)
    roll = math.radians(rng.uniform(-1, 1) * cfg.roll_deg * d)

    half_w, half_h = W / 2.0, H / 2.0
    plane = np.array(
        [[-half_w, -half_h, 0], [half_w, -half_h, 0], [half_w, half_h, 0], [-half_w, half_h, 0]],
        float,
    )
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cr, sr = math.cos(roll), math.sin(roll)
    rot = (
        np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]])
        @ np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
        @ np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    )

    f = W * 2.2
    cam = (rot @ plane.T).T + np.array([0.0, 0.0, f])
    proj = cam[:, :2] * (f / cam[:, 2:3])
    proj -= proj.min(axis=0)
    span = proj.max(axis=0)
    s = min(W / max(span[0], 1e-6), H / max(span[1], 1e-6)) * 0.98
    dst = proj * s
    dst += (np.array([W, H]) - dst.max(axis=0)) / 2.0

    src = np.array([[0, 0], [W, 0], [W, H], [0, H]], float)
    h_inv = _homography(dst, src)
    warped = canvas.transform(
        (W, H),
        Image.Transform.PERSPECTIVE,
        tuple((h_inv / h_inv[2, 2]).ravel()[:8]),
        Image.Resampling.BICUBIC,
    )

    h_fwd = _homography(src, dst)
    quad = _apply_h(h_fwd, [(ox, oy), (ox + w, oy), (ox + w, oy + h), (ox, oy + h)])

    xs = [p[0] for p in quad]
    ys = [p[1] for p in quad]
    pad = _r(rng, cfg.crop_pad, 1.0) * (max(xs) - min(xs))
    cx0 = max(0, int(min(xs) - pad))
    cy0 = max(0, int(min(ys) - pad))
    cx1 = min(W, int(max(xs) + pad))
    cy1 = min(H, int(max(ys) + pad))
    if cx1 - cx0 < 8 or cy1 - cy0 < 4:
        cx0, cy0, cx1, cy1 = 0, 0, W, H
    cropped = warped.crop((cx0, cy0, cx1, cy1))

    def _rebase(pts):
        return [(px - cx0, py - cy0) for px, py in pts]

    new_boxes = []
    for b in boxes:
        pts = _rebase(
            _apply_h(
                h_fwd,
                [
                    (b.x0 + ox, b.y0 + oy),
                    (b.x1 + ox, b.y0 + oy),
                    (b.x1 + ox, b.y1 + oy),
                    (b.x0 + ox, b.y1 + oy),
                ],
            )
        )
        bxs = [p[0] for p in pts]
        bys = [p[1] for p in pts]
        new_boxes.append(replace(b, x0=min(bxs), y0=min(bys), x1=max(bxs), y1=max(bys)))

    severity = min(1.0, (abs(yaw) + abs(pitch)) / math.radians(66))
    return cropped, tuple(new_boxes), tuple(_rebase(quad)), severity


# ---------------------------------------------------------------------------
# Optical stages
# ---------------------------------------------------------------------------

def _motion_blur(img, rng, cfg, d, scene):
    """Linear motion blur along the direction of travel.

    The mean of copies shifted along the motion vector, which is exactly what a
    linear motion blur is. Done in numpy because PIL's ``ImageFilter.Kernel``
    supports only 3x3 and 5x5, and realistic smear runs longer than that.
    """
    length = _r(rng, cfg.motion_blur_px, d)
    if length < 0.7:
        return img, 0.0
    n = max(3, int(round(length)) | 1)
    # Vehicles cross the frame roughly horizontally; the spread allows for
    # camera tilt and for vehicles crossing at an angle.
    angle = math.radians(rng.uniform(-25, 25))
    arr = np.asarray(img).astype(np.float32)
    acc = np.zeros_like(arr)
    half = n // 2
    for i in range(n):
        t = i - half
        acc += _shift(arr, int(round(t * math.sin(angle))), int(round(t * math.cos(angle))))
    acc /= n
    return (
        Image.fromarray(np.clip(acc, 0, 255).astype(np.uint8)),
        min(1.0, length / cfg.motion_blur_px[1]),
    )


def _defocus(img, rng, cfg, d, scene):
    r = _r(rng, cfg.defocus_px, d)
    if r < 0.3:
        return img, 0.0
    return img.filter(ImageFilter.GaussianBlur(radius=r)), min(1.0, r / cfg.defocus_px[1])


def _rolling_shutter(img, rng, cfg, d, scene):
    """Row-dependent horizontal shear.

    Cheap surveillance cameras use rolling-shutter CMOS sensors: each row is
    exposed at a slightly different instant, so a fast plate is sheared. Models
    trained without it underperform on footage from budget cameras -- which is
    most of them.
    """
    amount = _r(rng, cfg.rolling_shutter_shear, d)
    if amount < 0.008:
        return img, 0.0
    arr = np.asarray(img)
    h, w = arr.shape[:2]
    out = np.empty_like(arr)
    max_shift = amount * w
    for y in range(h):
        dx = int(round((y / max(h - 1, 1) - 0.5) * max_shift))
        if dx == 0:
            out[y] = arr[y]
            continue
        pad = abs(dx)
        row = np.pad(arr[y], ((pad, pad), (0, 0)), mode="edge")
        out[y] = row[pad - dx : pad - dx + w]
    return Image.fromarray(out), min(1.0, amount / cfg.rolling_shutter_shear[1])


def _haze(img, rng, cfg, d, scene):
    """Atmospheric veiling: dust and monsoon haze.

    Kathmandu's dry-season dust is a first-order image degradation, not a
    nicety. Modelled as the standard scattering blend toward an airlight colour.
    """
    t = scene.haze
    if t < 0.02:
        return img, 0.0
    airlight = rng.choice([(200, 200, 195), (185, 178, 160), (210, 208, 205)])
    return (
        Image.blend(img, Image.new("RGB", img.size, airlight), alpha=t),
        min(1.0, t / cfg.haze[1]),
    )


def _illumination(img, rng, cfg, d, scene):
    """Brightness, contrast, directional shadow, specular glare, night IR."""
    sev = 0.0
    b, c = scene.brightness, scene.contrast
    img = ImageEnhance.Brightness(img).enhance(max(0.15, b))
    img = ImageEnhance.Contrast(img).enhance(max(0.25, c))
    sev += min(1.0, abs(b - 1.0) + abs(c - 1.0)) * 0.3

    arr = np.asarray(img).astype(np.float32)
    h, w = arr.shape[:2]

    if scene.shadow is not None:
        # Hard-edged shadow: pole shadows, bodywork, a partially shaded junction.
        ang, edge, soft, depth = scene.shadow
        gx = np.linspace(0, 1, w, dtype=np.float32)
        gy = np.linspace(0, 1, h, dtype=np.float32)[:, None]
        ramp = gx * math.cos(ang) + gy * math.sin(ang)
        mask = 1.0 / (1.0 + np.exp(-(ramp - edge) / soft))
        arr *= (1.0 - depth * mask)[..., None]
        sev += depth * 0.6

    if scene.glare is not None:
        # Retroreflective sheeting throwing back a headlight or the sun.
        fx, fy, frad, strength = scene.glare
        cx, cy = fx * w, fy * h
        radius = frad * max(w, h)
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        blob = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * radius**2))
        arr += (blob * strength)[..., None]
        sev += strength / 150 * 0.5

    if scene.night:
        # IR-illuminated night: the IR-cut filter lifts, colour collapses to
        # near-monochrome and contrast drops. This destroys the plate-colour
        # signal -- which is exactly why the decoder's colour prior is soft
        # rather than a hard constraint.
        grey = arr.mean(axis=2, keepdims=True)
        arr = grey + (arr - grey) * scene.night_desat
        arr *= scene.night_gain
        sev += 0.35

    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)), min(1.0, sev)


# ---------------------------------------------------------------------------
# Sensor stages
# ---------------------------------------------------------------------------

def _noise(img, rng, cfg, d, scene):
    arr = np.asarray(img).astype(np.float32)
    gen = np.random.default_rng(rng.randrange(1 << 31))
    sev = 0.0

    photons = _r(rng, cfg.poisson_photons, d)
    if photons < cfg.poisson_photons[0]:
        # Shot noise: Poisson in photon count, so variance grows with signal --
        # bright areas get proportionally *less* relative noise, which is the
        # signature that distinguishes real sensor noise from added Gaussian.
        #
        # Note this must normalise to [0, 1] before drawing: sampling
        # poisson(intensity * k) with k below 1 collapses the whole intensity
        # range onto a handful of integers and produces pure colour hash, not
        # noise.
        norm = np.clip(arr, 0.0, 255.0) / 255.0
        arr = gen.poisson(norm * photons).astype(np.float32) / photons * 255.0
        lo, hi = cfg.poisson_photons
        sev += min(1.0, max(0.0, (math.log(lo) - math.log(photons)) / (math.log(lo) - math.log(hi)))) * 0.5

    sigma = _r(rng, cfg.gauss_noise, d)
    if sigma > 0.5:
        arr += gen.normal(0, sigma, arr.shape)
        sev += min(1.0, sigma / cfg.gauss_noise[1]) * 0.5

    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)), min(1.0, sev)


def _chromatic_aberration(img, rng, cfg, d, scene):
    shift = _r(rng, cfg.chroma_shift_px, d)
    s = int(round(shift))
    if s < 1:
        return img, 0.0
    arr = np.asarray(img)
    out = arr.copy()
    out[..., 0] = _shift(arr, 0, s)[..., 0]
    out[..., 2] = _shift(arr, 0, -s)[..., 2]
    return Image.fromarray(out), min(1.0, shift / cfg.chroma_shift_px[1])


# ---------------------------------------------------------------------------
# Codec
# ---------------------------------------------------------------------------

def _jpeg(img, rng, cfg, d, scene):
    """Re-encode at the bitrates real RTSP streams actually use.

    Almost every real plate crop has been through lossy compression at least
    once, often twice. Blocking at 8x8 boundaries is a strong learnable cue and
    a model that has never seen it is at a disadvantage.
    """
    q = int(round(_r(rng, cfg.jpeg_quality, d)))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=max(5, min(100, q)))
    buf.seek(0)
    hi, lo = cfg.jpeg_quality
    return Image.open(buf).convert("RGB"), min(1.0, max(0.0, (hi - q) / max(hi - lo, 1)))


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

Stage = Callable[
    [Image.Image, random.Random, DegradationConfig, float, "Scene"],
    "tuple[Image.Image, float]",
]

_OPTICAL: dict[str, Stage] = {
    "motion_blur": _motion_blur,
    "defocus": _defocus,
    "rolling_shutter": _rolling_shutter,
    "haze": _haze,
    "illumination": _illumination,
}
_SENSOR: dict[str, Stage] = {
    "noise": _noise,
    "chromatic_aberration": _chromatic_aberration,
}


def degrade(
    rendered: RenderedPlate,
    *,
    rng: random.Random | None = None,
    cfg: DegradationConfig | None = None,
    difficulty: float | None = None,
    scene: Scene | None = None,
    width_override: int | None = None,
) -> DegradedPlate:
    """Apply a randomly composed degradation chain at a sampled difficulty.

    Pass ``difficulty`` explicitly to generate a controlled evaluation sweep;
    leave it ``None`` for corpus generation.
    """
    rng = rng or random.Random()
    cfg = cfg or DegradationConfig()
    d = sample_difficulty(rng, cfg.difficulty_weights) if difficulty is None else float(difficulty)
    d = min(max(d, 0.0), 1.0)
    scene = scene if scene is not None else sample_scene(rng, cfg, d)

    img, boxes, corners = rendered.image, rendered.char_boxes, rendered.corners
    applied: list[tuple[str, float]] = []
    severities: list[float] = []

    # 1. Geometry first: the plate is posed before anything images it.
    img, boxes, corners, sev = _perspective(img, boxes, corners, rng, cfg, d, scene)
    applied.append(("perspective", sev))
    severities.append(sev * 0.6)

    # 2. Optical phase -- a random subset, random order within the phase.
    lo_n, hi_n = cfg.n_optical
    k = max(0, min(len(_OPTICAL), int(round(lo_n + d * (hi_n - lo_n)))))
    names = rng.sample(list(_OPTICAL), k)
    # Illumination is a scene property, so it precedes lens effects.
    names.sort(key=lambda n: 0 if n == "illumination" else 1)
    for name in names:
        img, sev = _OPTICAL[name](img, rng, cfg, d, scene)
        applied.append((name, sev))
        severities.append(sev)

    # 3. Resolution: sample down to the width a camera would see. Log-uniform,
    #    because difficulty concentrates at small sizes and a uniform draw would
    #    barely visit them.
    hi0, hi1 = cfg.plate_width_hi
    lo0, lo1 = cfg.plate_width_lo
    ceil_w = max(9.0, hi0 + d * (hi1 - hi0))
    floor_w = max(8.0, min(lo0 + d * (lo1 - lo0), ceil_w - 1.0))
    # A track generator overrides this directly, so that a vehicle's plate
    # grows monotonically as it approaches instead of jumping about.
    target_w = (
        int(width_override)
        if width_override is not None
        else int(round(math.exp(rng.uniform(math.log(floor_w), math.log(ceil_w)))))
    )
    scale = target_w / img.width
    img = img.resize(
        (max(8, target_w), max(4, int(round(img.height * scale)))),
        # Cameras and ISPs resample differently, and the kernel leaves a
        # signature. Randomising stops the corpus teaching one downscaler's
        # artefacts. BOX is area-averaging -- the closest analogue to a sensor
        # binning pixels.
        rng.choice(
            [
                Image.Resampling.BOX,
                Image.Resampling.BILINEAR,
                Image.Resampling.LANCZOS,
                Image.Resampling.BICUBIC,
            ]
        ),
    )
    boxes = tuple(
        replace(b, x0=b.x0 * scale, y0=b.y0 * scale, x1=b.x1 * scale, y1=b.y1 * scale)
        for b in boxes
    )
    corners = tuple((x * scale, y * scale) for x, y in corners)
    # Severity relative to the full width range the config can produce.
    res_sev = min(
        1.0,
        max(0.0, (math.log(hi0) - math.log(max(target_w, 8))) / (math.log(hi0) - math.log(lo1))),
    )
    applied.append(("resolution", res_sev))
    severities.append(res_sev * 1.5)  # resolution dominates readability

    # 4. Sensor phase.
    n_sensor = 1 if d < 0.5 else rng.randint(1, len(_SENSOR))
    for name in rng.sample(list(_SENSOR), n_sensor):
        img, sev = _SENSOR[name](img, rng, cfg, d, scene)
        applied.append((name, sev))
        severities.append(sev)

    # 5. Codec last -- compression is always the final step in a real chain.
    img, sev = _jpeg(img, rng, cfg, d, scene)
    applied.append(("jpeg", sev))
    severities.append(sev)

    # Quality anchors on the difficulty that was requested, corrected by what
    # the sampled parameters actually turned out to be. Anchoring purely on
    # measured severities makes the score drift with how many stages happened
    # to fire; anchoring purely on difficulty ignores the sampling.
    measured = float(np.mean(severities)) if severities else 0.0
    quality = float(np.clip(1.0 - (0.55 * d + 0.45 * min(1.0, measured * 1.2)), 0.0, 1.0))

    return DegradedPlate(
        image=img,
        source=rendered,
        char_boxes=boxes,
        corners=corners,
        quality=quality,
        difficulty=d,
        applied=tuple(applied),
        meta={
            "plate_width_px": target_w,
            "n_stages": len(applied),
            "night": scene.night,
            **rendered.meta,
        },
    )


__all__ = [
    "DegradationConfig",
    "DegradedPlate",
    "Scene",
    "degrade",
    "sample_difficulty",
    "sample_scene",
]
