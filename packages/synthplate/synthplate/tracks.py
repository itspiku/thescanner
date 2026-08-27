"""Synthesising vehicle tracks -- multi-frame sequences of one plate.

The recognition design treats a read as an estimate over a *track*, not a guess
from a frame (see ``nepal_plate.fuse``). That only works if the fusion model has
tracks to learn from, and no public Nepali dataset contains any. So we generate
them.

What makes a track different from N independent degraded images
---------------------------------------------------------------
Getting this wrong produces data that trains a model to do nothing useful.

* **Scene conditions are shared.** If it is night in frame 1 it is night in
  frame 12. Sampling lighting independently per frame makes fusion trivially
  effective -- averaging twelve independently-lit frames recovers the plate at
  once, which is not a skill that transfers to a real track where every frame is
  equally dark.
* **Scale evolves smoothly.** A vehicle approaches or recedes, so plate width
  follows a trajectory, not a random walk. Width goes as ``1/distance``, so the
  trajectory is modelled in distance and inverted -- which is why a plate grows
  slowly at first and then rapidly.
* **Difficulty is not monotone in time.** Far frames are hard because the plate
  is small; near frames are hard because the vehicle is sweeping past fastest,
  so motion blur and yaw are worst. The best frames are usually in the middle.
  That non-monotonicity is exactly what a crop-selection policy has to learn,
  and a generator that makes the last frame always the best would teach it the
  wrong lesson.
* **Per-frame noise stays independent.** Sensor noise is what fusion can
  actually average away, so it must not be shared.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Sequence

from .degrade import DegradationConfig, DegradedPlate, Scene, degrade, sample_difficulty, sample_scene
from .render import RenderedPlate, render
from .sampling import PlateSample


@dataclass(frozen=True, slots=True)
class TrackConfig:
    """How a vehicle crosses the camera's field of view."""

    n_frames: tuple[int, int] = (6, 24)

    #: Plate width in pixels at the far and near ends of the pass. The camera
    #: does not necessarily see either extreme -- the track is a window onto
    #: part of the approach.
    width_far: tuple[int, int] = (18, 45)
    width_near: tuple[int, int] = (55, 175)

    #: Fraction of tracks where the vehicle is receding (a rear plate driving
    #: away) rather than approaching. Both occur in practice and they have
    #: opposite quality trajectories.
    receding_fraction: float = 0.35

    #: How much extra difficulty the closest frames take on from motion blur
    #: and yaw sweep. This is what makes the quality peak mid-track.
    proximity_penalty: float = 0.30

    #: Per-frame difficulty jitter, on top of the track's base difficulty.
    frame_jitter: float = 0.07

    #: Fraction of frames dropped, modelling a detector that misses a vehicle
    #: on some frames. Real tracks have gaps.
    dropout: float = 0.08


@dataclass(slots=True)
class Track:
    """One vehicle passage: the same plate seen repeatedly."""

    sample: PlateSample
    frames: tuple[DegradedPlate, ...]
    scene: Scene
    base_difficulty: float
    receding: bool
    clean: RenderedPlate

    @property
    def canonical(self) -> str:
        return self.sample.plate.canonical

    def qualities(self) -> tuple[float, ...]:
        return tuple(f.quality for f in self.frames)

    def best_frame(self) -> DegradedPlate:
        return max(self.frames, key=lambda f: f.quality)


def _width_trajectory(
    n: int, w_far: float, w_near: float, receding: bool
) -> list[int]:
    """Plate widths across the pass.

    Apparent width goes as ``1/distance``, so we walk distance linearly and
    invert. That produces the characteristic non-linear growth of an
    approaching vehicle -- slow, then sudden -- rather than the even ramp a
    naive linear interpolation in width would give.
    """
    if n == 1:
        return [int(round((w_far + w_near) / 2))]
    z_far, z_near = 1.0 / max(w_far, 1e-6), 1.0 / max(w_near, 1e-6)
    widths = []
    for i in range(n):
        t = i / (n - 1)
        z = z_far + (z_near - z_far) * t
        widths.append(max(8, int(round(1.0 / max(z, 1e-6)))))
    if receding:
        widths.reverse()
    return widths


def synth_track(
    sample: PlateSample,
    *,
    rng: random.Random | None = None,
    cfg: DegradationConfig | None = None,
    track_cfg: TrackConfig | None = None,
    render_height: int = 110,
    clean: RenderedPlate | None = None,
) -> Track:
    """Generate one vehicle passage for ``sample``."""
    rng = rng or random.Random()
    cfg = cfg or DegradationConfig()
    tcfg = track_cfg or TrackConfig()

    clean = clean if clean is not None else render(sample, height=render_height, rng=rng)

    # One scene, one base difficulty, for the whole pass.
    base = sample_difficulty(rng, cfg.difficulty_weights)
    scene = sample_scene(rng, cfg, base)
    receding = rng.random() < tcfg.receding_fraction

    n = rng.randint(*tcfg.n_frames)
    w_far = rng.uniform(*tcfg.width_far)
    w_near = rng.uniform(*tcfg.width_near)
    if w_near <= w_far:
        w_near = w_far * 1.8
    widths = _width_trajectory(n, w_far, w_near, receding)

    frames: list[DegradedPlate] = []
    for i, w in enumerate(widths):
        if rng.random() < tcfg.dropout and 0 < i < n - 1:
            continue  # detector missed this frame

        # Proximity in [0, 1], independent of travel direction: 1 == closest.
        proximity = (w - min(widths)) / max(max(widths) - min(widths), 1e-6)
        # Close frames pay a blur/yaw penalty, which is what puts the quality
        # peak in the middle of the pass rather than at its end.
        d = base + tcfg.proximity_penalty * (proximity - 0.5)
        d += rng.uniform(-tcfg.frame_jitter, tcfg.frame_jitter)
        d = min(max(d, 0.0), 1.0)

        frames.append(
            degrade(
                clean,
                rng=rng,
                cfg=cfg,
                difficulty=d,
                scene=scene,       # shared: lighting, haze, shadow, surround
                width_override=w,  # smooth trajectory, not a random draw
            )
        )

    return Track(
        sample=sample,
        frames=tuple(frames),
        scene=scene,
        base_difficulty=base,
        receding=receding,
        clean=clean,
    )


def synth_tracks(
    samples: Sequence[PlateSample],
    *,
    rng: random.Random | None = None,
    cfg: DegradationConfig | None = None,
    track_cfg: TrackConfig | None = None,
    render_height: int = 110,
):
    """Yield one :class:`Track` per sample."""
    rng = rng or random.Random()
    for s in samples:
        yield synth_track(
            s, rng=rng, cfg=cfg, track_cfg=track_cfg, render_height=render_height
        )


__all__ = ["TrackConfig", "Track", "synth_track", "synth_tracks"]
