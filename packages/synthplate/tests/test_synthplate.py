"""Tests for the synthetic plate generator.

The generator's job is to produce a corpus that is (a) correctly labelled,
(b) structurally exhaustive, and (c) spread across a usable range of difficulty.
Each of those is checked directly, because all three fail silently: a generator
with a wrong label, a missing vehicle class, or a corpus that is uniformly
unreadable all produce plausible-looking images and a model that quietly
underperforms.
"""

from __future__ import annotations

import random
import statistics
import warnings

import numpy as np
import pytest

from nepal_plate import Ownership, PlateColour, PlateSystem, parse, spec
from nepal_plate.grammar import EMBOSSED_GRAMMAR, LEGACY_GRAMMAR

from synthplate.degrade import (
    DegradationConfig,
    Scene,
    degrade,
    sample_difficulty,
    sample_scene,
)
from synthplate.render import COLOUR_RGB, render
from synthplate.sampling import PlateSampler, sample_balanced_grid
from synthplate.tracks import TrackConfig, synth_track

# The generator warns loudly when authentic plate fonts are missing. That is
# the correct behaviour and is asserted below; silence it elsewhere.
# NB: pytest warning filters use ":" as a field separator, so the pattern
# cannot contain the colon in "synthplate: using fallback".
pytestmark = pytest.mark.filterwarnings(r"ignore:.*using fallback.*:UserWarning")


@pytest.fixture
def rng():
    return random.Random(1234)


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def test_every_sampled_plate_is_grammatically_valid():
    """A corpus with a single illegal label would poison the constrained
    decoder's notion of what a plate looks like."""
    sampler = PlateSampler(seed=7)
    for _ in range(500):
        s = sampler.sample()
        assert s.plate.is_valid, (s.tokens, s.plate.errors)
        assert s.plate.canonical


def test_sampled_label_round_trips_through_the_parser():
    """The image label and the parser must agree, or training targets and
    inference outputs are in different alphabets."""
    sampler = PlateSampler(seed=8)
    for _ in range(300):
        s = sampler.sample()
        assert parse(s.plate.canonical).canonical == s.plate.canonical


def test_legacy_colour_always_matches_ownership():
    """The colour prior in the decoder is only learnable if the corpus is
    self-consistent: a red plate must always be a private one."""
    sampler = PlateSampler(seed=9, embossed_fraction=0.0)
    for _ in range(300):
        s = sampler.sample()
        assert s.colour in spec.LEGACY_COLOUR_OWNERSHIP
        assert spec.LEGACY_COLOUR_OWNERSHIP[s.colour] is s.plate.ownership


def test_embossed_plates_are_always_black_on_white():
    sampler = PlateSampler(seed=10, embossed_fraction=1.0)
    for _ in range(100):
        assert sampler.sample().colour is PlateColour.WHITE_BLACK


def test_balanced_grid_covers_every_structural_combination():
    """Uniform random sampling only covers the space in expectation. The rare
    categories -- diplomatic, corporation, tourist -- are exactly the ones a
    random draw under-serves and where a misread costs most."""
    grid = list(sample_balanced_grid(seed=1))
    legacy = [g for g in grid if g.plate.system is PlateSystem.DEVANAGARI]

    seen = {(g.plate.zone_deva, g.plate.vehicle_class_deva) for g in legacy}
    assert len(seen) == len(spec.ZONE_TOKENS) * len(spec.CLASS_TOKENS)

    # Every ownership category present, including the rare ones.
    owners = {g.plate.ownership for g in legacy}
    assert owners == {o for o in Ownership if o is not Ownership.UNKNOWN}

    embossed = [g for g in grid if g.plate.system is PlateSystem.EMBOSSED]
    assert {g.plate.province for g in embossed} == set(range(1, 8))
    # Every embossed class except a bare "J". The grammar accepts bare "J" --
    # be liberal in what you accept, in case such a plate exists in the field --
    # but heavy equipment is only ever *issued* as J1-J5, so the generator does
    # not manufacture a category that does not exist on the road.
    assert {g.plate.class_letter for g in embossed} == set(spec.EMBOSSED_CLASSES) - {"J"}


def test_uniform_and_realistic_strategies_differ():
    uni = PlateSampler(seed=3, strategy="uniform", embossed_fraction=0.0)
    real = PlateSampler(seed=3, strategy="realistic", embossed_fraction=0.0)
    n = 600
    uni_moto = sum(uni.sample().plate.vehicle_class_deva == "प" for _ in range(n))
    real_moto = sum(real.sample().plate.vehicle_class_deva == "प" for _ in range(n))
    # Nepal's fleet is overwhelmingly two-wheeled, so the realistic sampler
    # should produce far more private motorcycles than a uniform draw.
    assert real_moto > uni_moto * 2


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def test_render_produces_one_box_per_token(rng):
    sampler = PlateSampler(seed=11)
    for _ in range(60):
        s = sampler.sample()
        r = render(s, height=96, rng=rng)
        assert len(r.char_boxes) == len(s.tokens)
        assert [b.token for b in r.char_boxes] == list(s.tokens)


def test_char_boxes_lie_inside_the_image(rng):
    sampler = PlateSampler(seed=12)
    for _ in range(60):
        r = render(sampler.sample(), height=96, rng=rng)
        w, h = r.image.size
        for b in r.char_boxes:
            assert -2 <= b.x0 < b.x1 <= w + 2, b
            assert -2 <= b.y0 < b.y1 <= h + 2, b


def test_render_uses_the_right_colour_scheme(rng):
    """Sanity-check the rendered pixels actually match the declared scheme --
    a mislabelled colour would train the colour classifier backwards."""
    sampler = PlateSampler(seed=13, embossed_fraction=0.0)
    for _ in range(40):
        s = sampler.sample()
        r = render(s, height=96, rng=rng, colour_jitter=0)
        expected_bg = np.array(COLOUR_RGB[s.colour][0], dtype=float)
        # The painted rim starts about 3 px in at this render height, so sample
        # strictly outside it -- the outermost two pixels are pure background.
        arr = np.asarray(r.image).astype(float)
        corner = arr[0:2, 0:2].reshape(-1, 3).mean(axis=0)
        assert np.linalg.norm(corner - expected_bg) < 30, (s.colour, corner, expected_bg)


def test_two_row_plates_are_squarer_than_single_row(rng):
    sampler = PlateSampler(seed=14)
    singles, doubles = [], []
    for _ in range(120):
        s = sampler.sample()
        r = render(s, height=96, rng=rng)
        (doubles if s.two_row else singles).append(r.image.width / r.image.height)
    assert statistics.mean(doubles) < statistics.mean(singles)


def test_fallback_fonts_warn():
    """Silent use of the wrong typeface is the failure mode this guards."""
    from synthplate import fonts

    fonts.resolve.cache_clear()
    if fonts.fonts_are_authentic():
        pytest.skip("authentic fonts installed; nothing to warn about")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fonts.resolve("devanagari")
    assert any("fallback" in str(w.message) for w in caught)


# ---------------------------------------------------------------------------
# Degradation
# ---------------------------------------------------------------------------

def test_difficulty_zero_is_essentially_clean(rng):
    sampler = PlateSampler(seed=15)
    d = degrade(render(sampler.sample(), height=110, rng=rng), rng=rng, difficulty=0.0)
    assert d.quality > 0.85
    assert d.meta["plate_width_px"] >= 90


def test_quality_decreases_monotonically_with_difficulty(rng):
    """``quality`` is the training target for the crop-quality estimator that
    drives fusion weighting, so it has to be a real signal, not noise."""
    sampler = PlateSampler(seed=16)
    means = []
    for d in (0.0, 0.25, 0.5, 0.75, 1.0):
        qs = [
            degrade(render(sampler.sample(), height=110, rng=rng), rng=rng, difficulty=d).quality
            for _ in range(25)
        ]
        means.append(statistics.mean(qs))
    assert means == sorted(means, reverse=True), means
    assert means[0] - means[-1] > 0.4


def test_corpus_has_a_usable_difficulty_spread(rng):
    """A corpus that is uniformly hard trains nothing: the model needs legible
    examples to learn glyph shapes before hard ones teach it robustness.

    This is a regression test for a real defect -- the first version of the
    degradation pipeline sampled every parameter independently at full range,
    and the product came out with a median quality of 0.33, mostly unreadable.
    """
    sampler = PlateSampler(seed=17)
    qs = [
        degrade(render(sampler.sample(), height=110, rng=rng), rng=rng).quality
        for _ in range(200)
    ]
    assert statistics.median(qs) > 0.5, statistics.median(qs)
    assert sum(q >= 0.7 for q in qs) / len(qs) > 0.25, "not enough easy samples"
    assert sum(q <= 0.4 for q in qs) / len(qs) > 0.10, "not enough hard samples"


def _horizontal_autocorrelation(img) -> float:
    """Lag-1 correlation between horizontally adjacent pixels.

    Natural images are spatially smooth and score high; uncorrelated noise
    scores near zero. This discriminates structure from hash, which variance
    cannot -- a clean high-contrast plate and a field of random pixels have
    almost the same standard deviation.
    """
    a = np.asarray(img).astype(float).mean(axis=2)
    x, y = a[:, :-1].ravel(), a[:, 1:].ravel()
    if x.std() < 1e-6 or y.std() < 1e-6:
        return 1.0
    return float(np.corrcoef(x, y)[0, 1])


def test_degradation_never_produces_pure_noise(rng):
    """Regression test for a mis-parameterised Poisson shot-noise model.

    Sampling ``poisson(intensity * k)`` with ``k`` below 1 collapses the whole
    intensity range onto a handful of integers, producing colour hash rather
    than noise. It went unnoticed at first because such an image has a
    perfectly ordinary standard deviation -- what it lacks is *spatial
    structure*, so that is what this checks.
    """
    sampler = PlateSampler(seed=18)
    cors = [
        _horizontal_autocorrelation(
            degrade(render(sampler.sample(), height=110, rng=rng), rng=rng).image
        )
        for _ in range(120)
    ]
    # Asserted on the population, not the worst single sample: a genuinely hard
    # sample (small, noisy, heavily compressed) reaches ~0.41, which overlaps
    # the top of the broken model's range. The distributions are nonetheless
    # far apart -- measured medians are 0.85 for the correct model against 0.27
    # for the broken one -- and the bug affected essentially every sample that
    # hit the noise stage, so a population statistic is the right instrument.
    assert statistics.median(cors) > 0.70, statistics.median(cors)
    assert sum(c < 0.35 for c in cors) / len(cors) < 0.02


def test_plate_width_tracks_difficulty(rng):
    sampler = PlateSampler(seed=19)
    easy = [
        degrade(render(sampler.sample(), height=110, rng=rng), rng=rng, difficulty=0.1).meta["plate_width_px"]
        for _ in range(30)
    ]
    hard = [
        degrade(render(sampler.sample(), height=110, rng=rng), rng=rng, difficulty=0.95).meta["plate_width_px"]
        for _ in range(30)
    ]
    assert statistics.median(hard) < statistics.median(easy) / 1.8


def test_degraded_image_has_no_hard_black_border(rng):
    """The perspective warp used to leave black wedges in the corners. Real
    detector crops contain surrounding scene, never a hard black frame, and a
    model would learn to expect one."""
    sampler = PlateSampler(seed=20)
    for _ in range(40):
        d = degrade(render(sampler.sample(), height=110, rng=rng), rng=rng, difficulty=0.8)
        arr = np.asarray(d.image).astype(float)
        corners = np.concatenate([
            arr[:3, :3].reshape(-1, 3), arr[:3, -3:].reshape(-1, 3),
            arr[-3:, :3].reshape(-1, 3), arr[-3:, -3:].reshape(-1, 3),
        ])
        assert corners.mean() > 12.0


def test_char_boxes_survive_the_warp(rng):
    sampler = PlateSampler(seed=21)
    for _ in range(40):
        d = degrade(render(sampler.sample(), height=110, rng=rng), rng=rng, difficulty=0.6)
        assert len(d.char_boxes) == len(d.source.sample.tokens)
        w, h = d.image.size
        # Boxes should overlap the image; a warp bug sends them far outside.
        for b in d.char_boxes:
            assert b.x1 > -w and b.x0 < 2 * w, b
        assert len(d.corners) == 4


def test_generation_is_reproducible_from_a_seed():
    """An auditor must be able to regenerate a government model's training
    corpus exactly."""
    def run():
        rng = random.Random(999)
        sampler = PlateSampler(seed=999)
        d = degrade(render(sampler.sample(), height=96, rng=rng), rng=rng)
        return d.source.sample.plate.canonical, round(d.quality, 6), d.image.size

    assert run() == run()


# ---------------------------------------------------------------------------
# Tracks
# ---------------------------------------------------------------------------

def test_track_frames_share_one_scene(rng):
    """If it is night in frame 1 it is night in frame 12. Independent lighting
    per frame makes fusion trivially effective and teaches nothing."""
    sampler = PlateSampler(seed=22)
    for _ in range(20):
        t = synth_track(sampler.sample(), rng=rng)
        assert len({f.meta["night"] for f in t.frames}) == 1


def test_track_frames_all_carry_the_same_plate(rng):
    sampler = PlateSampler(seed=23)
    t = synth_track(sampler.sample(), rng=rng)
    assert len({f.source.sample.plate.canonical for f in t.frames}) == 1
    assert t.canonical == t.frames[0].source.sample.plate.canonical


def test_track_width_evolves_smoothly_not_randomly(rng):
    """A vehicle approaches or recedes; plate width follows a trajectory. Frame
    to frame the change should be gradual, not a random walk."""
    sampler = PlateSampler(seed=24)
    tcfg = TrackConfig(dropout=0.0)
    for _ in range(15):
        t = synth_track(sampler.sample(), rng=rng, track_cfg=tcfg)
        widths = [f.meta["plate_width_px"] for f in t.frames]
        if len(widths) < 3:
            continue
        # Monotone in one direction (approaching or receding).
        deltas = [b - a for a, b in zip(widths, widths[1:])]
        assert all(d >= 0 for d in deltas) or all(d <= 0 for d in deltas), widths


def test_track_has_frames_of_varying_quality(rng):
    """Fusion is only worth doing if frames differ. A track where every frame is
    equally good or equally bad is not a useful training example."""
    sampler = PlateSampler(seed=25)
    spreads = []
    for _ in range(20):
        t = synth_track(sampler.sample(), rng=rng)
        qs = t.qualities()
        if len(qs) >= 3:
            spreads.append(max(qs) - min(qs))
    assert statistics.median(spreads) > 0.05


def test_best_frame_is_usually_not_the_last(rng):
    """Far frames are small, near frames are blurred and sweeping fastest -- so
    the best look is typically mid-pass. A generator whose last frame is always
    best would teach a crop-selection policy exactly the wrong rule."""
    sampler = PlateSampler(seed=26)
    last_is_best = 0
    n = 60
    for _ in range(n):
        t = synth_track(sampler.sample(), rng=rng)
        if len(t.frames) < 3:
            n -= 1
            continue
        if t.best_frame() is t.frames[-1]:
            last_is_best += 1
    assert last_is_best / max(n, 1) < 0.5


def test_track_respects_frame_count_bounds(rng):
    sampler = PlateSampler(seed=27)
    tcfg = TrackConfig(n_frames=(6, 10), dropout=0.0)
    for _ in range(20):
        t = synth_track(sampler.sample(), rng=rng, track_cfg=tcfg)
        assert 6 <= len(t.frames) <= 10
