"""Multi-object tracking, following the BYTE association strategy.

Why tracking is load-bearing here
---------------------------------
A read in this system is an estimate over a vehicle *passage*, not a guess from
a frame (see ``nepal_plate.fuse``). Tracking is what defines the passage. It
also collapses volume by an order of magnitude: one vehicle crossing the frame
produces one read event, not forty, which is the difference between a few
million reads a day nationally and a few hundred million.

The BYTE idea
-------------
Conventional trackers discard low-confidence detections before association.
BYTE's observation is that a low-confidence box is usually a *real object that
got harder to see* -- occluded, motion-blurred, or half out of frame -- and
those are precisely the frames during which a naive tracker drops the track and
then re-acquires it as a new one. A vehicle that fragments into three tracks
produces three reads of the same passage, three zone sessions, and three
watch-list alerts.

So association runs twice: high-confidence detections first, then the
*remaining* tracks are matched against the low-confidence leftovers. Only
high-confidence detections may start a new track.

Implementation notes
--------------------
Motion is a constant-velocity Kalman filter over ``(cx, cy, aspect, height)``,
the SORT formulation, which is well suited to vehicles crossing a fixed camera.

Association uses greedy IoU matching rather than the Hungarian algorithm. At a
junction the frame holds tens of vehicles at most, where greedy and optimal
assignment agree almost always, and greedy avoids a SciPy dependency on an edge
image. If a deployment ever sees dense enough traffic for the difference to
matter, swapping in ``scipy.optimize.linear_sum_assignment`` is a change to one
function.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Sequence

import numpy as np

Box = tuple[float, float, float, float]  # x1, y1, x2, y2


class TrackState(str, Enum):
    TENTATIVE = "tentative"   # seen once; not yet trusted
    CONFIRMED = "confirmed"
    LOST = "lost"             # unmatched, still predicted forward
    REMOVED = "removed"


@dataclass(slots=True)
class Detection:
    box: Box
    score: float
    label: str = "vehicle"
    #: Optional plate box associated with this vehicle by the detector.
    plate_box: Box | None = None


def iou_matrix(a: Sequence[Box], b: Sequence[Box]) -> np.ndarray:
    if not a or not b:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    A = np.asarray(a, dtype=np.float32)[:, None, :]
    B = np.asarray(b, dtype=np.float32)[None, :, :]
    x1 = np.maximum(A[..., 0], B[..., 0])
    y1 = np.maximum(A[..., 1], B[..., 1])
    x2 = np.minimum(A[..., 2], B[..., 2])
    y2 = np.minimum(A[..., 3], B[..., 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (A[..., 2] - A[..., 0]) * (A[..., 3] - A[..., 1])
    area_b = (B[..., 2] - B[..., 0]) * (B[..., 3] - B[..., 1])
    return inter / np.maximum(area_a + area_b - inter, 1e-6)


def greedy_match(cost: np.ndarray, threshold: float) -> list[tuple[int, int]]:
    """Greedy assignment on an IoU matrix, best pair first."""
    matches: list[tuple[int, int]] = []
    if cost.size == 0:
        return matches
    work = cost.copy()
    while True:
        i, j = np.unravel_index(np.argmax(work), work.shape)
        if work[i, j] < threshold:
            break
        matches.append((int(i), int(j)))
        work[i, :] = -1.0
        work[:, j] = -1.0
    return matches


class KalmanBox:
    """Constant-velocity filter over ``(cx, cy, aspect, height)``.

    Deliberately the plain SORT model rather than anything richer. Vehicles
    crossing a fixed camera move smoothly and predictably over the one-to-two
    seconds a track lasts, and the filter's job here is only to bridge a handful
    of missed frames -- not to extrapolate far into the future, where a
    constant-velocity assumption would start to hurt.
    """

    __slots__ = ("mean", "cov")

    def __init__(self, box: Box) -> None:
        self.mean = np.zeros(8, dtype=np.float32)
        self.mean[:4] = self._to_xyah(box)
        self.cov = np.eye(8, dtype=np.float32)
        self.cov[4:, 4:] *= 100.0   # velocity is unknown at birth
        self.cov *= 10.0

    @staticmethod
    def _to_xyah(box: Box) -> np.ndarray:
        x1, y1, x2, y2 = box
        w, h = max(x2 - x1, 1e-3), max(y2 - y1, 1e-3)
        return np.array([x1 + w / 2, y1 + h / 2, w / h, h], dtype=np.float32)

    @staticmethod
    def _to_box(state: np.ndarray) -> Box:
        cx, cy, a, h = state[:4]
        w = max(a * h, 1e-3)
        return (float(cx - w / 2), float(cy - h / 2), float(cx + w / 2), float(cy + h / 2))

    def predict(self) -> Box:
        f = np.eye(8, dtype=np.float32)
        f[:4, 4:] = np.eye(4, dtype=np.float32)
        self.mean = f @ self.mean
        q = np.eye(8, dtype=np.float32)
        q[:4] *= 1.0
        q[4:] *= 0.01
        self.cov = f @ self.cov @ f.T + q
        return self._to_box(self.mean)

    def update(self, box: Box) -> None:
        h = np.zeros((4, 8), dtype=np.float32)
        h[:, :4] = np.eye(4, dtype=np.float32)
        r = np.eye(4, dtype=np.float32) * 1.0
        z = self._to_xyah(box)
        y = z - h @ self.mean
        s = h @ self.cov @ h.T + r
        k = self.cov @ h.T @ np.linalg.inv(s)
        self.mean = self.mean + k @ y
        self.cov = (np.eye(8, dtype=np.float32) - k @ h) @ self.cov

    @property
    def box(self) -> Box:
        return self._to_box(self.mean)


@dataclass(slots=True)
class Track:
    track_id: int
    kalman: KalmanBox
    state: TrackState
    score: float
    label: str
    first_frame: int
    last_frame: int
    hits: int = 1
    age: int = 0
    time_since_update: int = 0
    plate_box: Box | None = None
    #: Box history, for zone crossing and direction of travel.
    trail: list[Box] = field(default_factory=list)

    @property
    def box(self) -> Box:
        return self.kalman.box

    def centroid(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.box
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    def direction(self) -> tuple[float, float] | None:
        """Unit vector of travel, from the trail. ``None`` until it has moved.

        Used by the zone engine to distinguish entering from leaving when a
        vehicle crosses a boundary, and by crop selection to prefer frames where
        the plate faces the camera.
        """
        if len(self.trail) < 2:
            return None
        (ax1, ay1, ax2, ay2), (bx1, by1, bx2, by2) = self.trail[0], self.trail[-1]
        dx = (bx1 + bx2) / 2 - (ax1 + ax2) / 2
        dy = (by1 + by2) / 2 - (ay1 + ay2) / 2
        n = (dx * dx + dy * dy) ** 0.5
        return (dx / n, dy / n) if n > 1e-3 else None


class ByteTracker:
    """BYTE-style multi-object tracker."""

    def __init__(
        self,
        *,
        high_threshold: float = 0.55,
        low_threshold: float = 0.15,
        match_threshold: float = 0.30,
        second_match_threshold: float = 0.45,
        max_age: int = 30,
        min_hits: int = 3,
        trail_length: int = 64,
    ) -> None:
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        self.match_threshold = match_threshold
        #: Second-round matching is stricter. Low-confidence boxes are noisier,
        #: so a loose threshold there attaches junk to real tracks -- which is
        #: worse than briefly losing one.
        self.second_match_threshold = second_match_threshold
        self.max_age = max_age
        #: A track must be seen this many times before it is trusted. Prevents a
        #: single spurious detection from opening a zone session.
        self.min_hits = min_hits
        self.trail_length = trail_length

        self.tracks: list[Track] = []
        self._next_id = 1
        self.frame_index = 0

    def update(self, detections: Sequence[Detection]) -> list[Track]:
        """Advance one frame. Returns the currently confirmed tracks."""
        self.frame_index += 1

        for t in self.tracks:
            t.kalman.predict()
            t.age += 1
            t.time_since_update += 1

        high = [d for d in detections if d.score >= self.high_threshold]
        low = [d for d in detections if self.low_threshold <= d.score < self.high_threshold]

        active = [t for t in self.tracks if t.state is not TrackState.REMOVED]

        # Round 1: confident detections against every live track.
        matched, unmatched_tracks, unmatched_high = self._associate(
            active, high, self.match_threshold
        )
        # Round 2: the BYTE step. Tracks still unmatched get a chance at the
        # low-confidence boxes -- usually the same vehicle, just harder to see.
        matched2, still_unmatched, _ = self._associate(
            unmatched_tracks, low, self.second_match_threshold
        )

        for track, det in matched + matched2:
            track.kalman.update(det.box)
            track.score = det.score
            track.hits += 1
            track.time_since_update = 0
            track.last_frame = self.frame_index
            track.plate_box = det.plate_box or track.plate_box
            track.trail.append(track.box)
            del track.trail[: max(0, len(track.trail) - self.trail_length)]
            if track.state is TrackState.LOST:
                track.state = TrackState.CONFIRMED
            elif track.state is TrackState.TENTATIVE and track.hits >= self.min_hits:
                track.state = TrackState.CONFIRMED

        for track in still_unmatched:
            track.state = (
                TrackState.REMOVED
                if track.time_since_update > self.max_age
                # A tentative track that misses even one frame is discarded --
                # it was probably never real.
                or (track.state is TrackState.TENTATIVE and track.time_since_update > 1)
                else TrackState.LOST
            )

        # Only confident detections may start a track.
        for det in unmatched_high:
            self.tracks.append(
                Track(
                    track_id=self._next_id,
                    kalman=KalmanBox(det.box),
                    state=TrackState.TENTATIVE,
                    score=det.score,
                    label=det.label,
                    first_frame=self.frame_index,
                    last_frame=self.frame_index,
                    plate_box=det.plate_box,
                    trail=[det.box],
                )
            )
            self._next_id += 1

        self.tracks = [t for t in self.tracks if t.state is not TrackState.REMOVED]
        return [t for t in self.tracks if t.state is TrackState.CONFIRMED]

    def _associate(
        self, tracks: Sequence[Track], detections: Sequence[Detection], threshold: float
    ) -> tuple[list[tuple[Track, Detection]], list[Track], list[Detection]]:
        if not tracks or not detections:
            return [], list(tracks), list(detections)
        ious = iou_matrix([t.box for t in tracks], [d.box for d in detections])
        pairs = greedy_match(ious, threshold)
        matched = [(tracks[i], detections[j]) for i, j in pairs]
        used_t = {i for i, _ in pairs}
        used_d = {j for _, j in pairs}
        return (
            matched,
            [t for i, t in enumerate(tracks) if i not in used_t],
            [d for j, d in enumerate(detections) if j not in used_d],
        )

    def finished(self) -> list[Track]:
        """Tracks that have just left the scene, for end-of-passage processing."""
        return [
            t for t in self.tracks
            if t.state is TrackState.LOST and t.time_since_update >= self.max_age
        ]

    def reset(self) -> None:
        self.tracks.clear()
        self.frame_index = 0


__all__ = [
    "Box",
    "Detection",
    "Track",
    "TrackState",
    "ByteTracker",
    "KalmanBox",
    "iou_matrix",
    "greedy_match",
]
