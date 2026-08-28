"""The edge pipeline: camera in, signed read events out.

::

    frames --> detect --> track --> zone sessions --> events
                            |
                        accumulate crops
                            |
                     (track ends / stabilises)
                            |
                    select informative crops
                            |
                     recognise + fuse  --> one read per passage
                            |
                    redact --> sign --> durable queue

Design decisions worth stating
------------------------------
**One read per passage, emitted at the end.** A vehicle crossing the frame is
recognised once, from a fused set of crops, not once per frame. This is what
keeps national volume in the millions rather than the hundreds of millions, and
it is why fusion lives at the edge rather than centrally -- fusing after
transmission would mean shipping forty crops per vehicle over a link that is
the scarcest resource in the whole system.

**Zone events are emitted immediately; the plate arrives later.** Entry happens
the moment a vehicle crosses the boundary, but recognition completes when the
passage ends. Holding the entry event back until the plate is known would delay
every alert by the length of a passage. So sessions are emitted plate-less and
back-filled by ``session_plate`` events, which the platform reconciles on
``session_id``. A session that never receives a plate is reported as an
unidentified passage rather than dropped -- an unread vehicle still entered.

**Recognition reads unredacted pixels; only stored imagery is redacted.** The
plate must be read from the original frame, but the evidence image written to
the queue has faces and cabins blurred first, so an identifiable face never
reaches durable storage. See ``privacy.py``.

**Everything is signed before it is queued.** The queue is the first durable
surface, and anything that reaches it is already part of the tamper-evident
chain.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Iterable, Sequence

import numpy as np

from nepal_plate import Confidence

from .detect import Detector
from scanner_evidence import NodeIdentity
from .privacy import PrivacyConfig, redact_frame
from .queue import EventQueue
from .select import Candidate, crop_plate, select
from .sources import Frame, FrameSource
from .tracker import ByteTracker, Detection, Track, TrackState
from .zones import ZoneEngine, ZoneEvent, ZoneEventKind


@dataclass(frozen=True)
class PipelineConfig:
    camera_id: str
    site_id: str = "unknown"
    #: Crops recognised per passage. Eight is enough for fusion to converge and
    #: cheap enough for four streams on one Jetson.
    crops_per_track: int = 8
    #: Frames a track must be idle before it counts as finished.
    finish_after_idle: int = 12
    #: Emit a provisional read this many frames into a passage, so an alert can
    #: fire while the vehicle is still in view. The final fused read follows and
    #: supersedes it.
    provisional_after: int = 8
    #: Store a plate crop with each read. Small; the context frame is not stored
    #: by default because it is large and far more privacy-sensitive.
    store_plate_image: bool = True
    store_context_image: bool = False
    jpeg_quality: int = 88
    privacy: PrivacyConfig = field(default_factory=PrivacyConfig)
    #: Reads below this band are still recorded but flagged for human review
    #: rather than acted on automatically.
    auto_action_min_confidence: Confidence = Confidence.HIGH


@dataclass(slots=True)
class _TrackBuffer:
    """Crops accumulated for one vehicle passage."""

    track_id: int
    candidates: list[Candidate] = field(default_factory=list)
    first_seen: float = 0.0
    last_seen: float = 0.0
    provisional_emitted: bool = False
    frames_seen: int = 0


class EdgePipeline:
    """Runs one camera end to end."""

    def __init__(
        self,
        *,
        cfg: PipelineConfig,
        detector: Detector,
        reader,                     # scanner_models.infer.PlateReader
        queue: EventQueue,
        zones: ZoneEngine | None = None,
        tracker: ByteTracker | None = None,
        on_event: Callable[[str, dict], None] | None = None,
    ) -> None:
        self.cfg = cfg
        self.detector = detector
        self.reader = reader
        self.queue = queue
        self.zones = zones or ZoneEngine([])
        self.tracker = tracker or ByteTracker()
        self.on_event = on_event

        self._buffers: dict[int, _TrackBuffer] = {}
        self._session_of_track: dict[int, list[str]] = {}
        self.stats = {
            "frames": 0, "detections": 0, "tracks_started": 0,
            "reads": 0, "provisional": 0, "zone_entries": 0, "zone_exits": 0,
            "unreadable": 0, "faces_redacted": 0,
        }

    # -- main loop ------------------------------------------------------

    def run(self, source: FrameSource, *, max_frames: int | None = None) -> dict:
        try:
            for frame in source.frames():
                self.process(frame)
                if max_frames is not None and self.stats["frames"] >= max_frames:
                    break
        finally:
            self.flush()
            source.close()
        return dict(self.stats)

    def process(self, frame: Frame) -> list[dict]:
        """Handle one frame. Returns the event payloads emitted."""
        self.stats["frames"] += 1
        emitted: list[dict] = []

        detections = self.detector.detect(frame.image)
        self.stats["detections"] += len(detections)
        tracks = self.tracker.update(detections)

        for t in tracks:
            if t.track_id not in self._buffers:
                self._buffers[t.track_id] = _TrackBuffer(
                    track_id=t.track_id, first_seen=frame.timestamp
                )
                self.stats["tracks_started"] += 1
            self._accumulate(frame, t)

        for ev in self.zones.update(tracks, frame_index=frame.index,
                                    now=_dt(frame.timestamp)):
            emitted.append(self._emit_zone_event(ev, frame))

        # A provisional read while the vehicle is still in view, so a watch-list
        # alert can fire in time to matter.
        for t in tracks:
            buf = self._buffers.get(t.track_id)
            if (
                buf is not None
                and not buf.provisional_emitted
                and buf.frames_seen >= self.cfg.provisional_after
                and len(buf.candidates) >= 3
            ):
                payload = self._recognise_and_emit(buf, frame, provisional=True)
                if payload:
                    emitted.append(payload)
                buf.provisional_emitted = True

        emitted.extend(self._finish_idle_tracks(frame))
        return emitted

    # -- accumulation ---------------------------------------------------

    def _accumulate(self, frame: Frame, track: Track) -> None:
        buf = self._buffers[track.track_id]
        buf.last_seen = frame.timestamp
        buf.frames_seen += 1

        # Prefer the detector's plate box; fall back to the lower portion of the
        # vehicle box, which is where a plate sits on essentially every vehicle.
        box = track.plate_box or _plate_region(track.box)
        crop, actual = crop_plate(frame.image, box)
        if crop.size == 0:
            return
        buf.candidates.append(
            Candidate(
                frame_index=frame.index,
                image=crop,
                box=actual,
                detector_score=track.score,
                frame_size=frame.size,
            )
        )

    def _finish_idle_tracks(self, frame: Frame) -> list[dict]:
        out: list[dict] = []
        for tid, buf in list(self._buffers.items()):
            track = next((t for t in self.tracker.tracks if t.track_id == tid), None)
            idle = track is None or track.time_since_update >= self.cfg.finish_after_idle
            if not idle:
                continue
            payload = self._recognise_and_emit(buf, frame, provisional=False)
            if payload:
                out.append(payload)
            del self._buffers[tid]
        return out

    # -- recognition ----------------------------------------------------

    def _recognise_and_emit(
        self, buf: _TrackBuffer, frame: Frame, *, provisional: bool
    ) -> dict | None:
        chosen = select(buf.candidates, k=self.cfg.crops_per_track)
        if not chosen:
            self.stats["unreadable"] += 1
            return None

        images = [_to_pil(s.candidate.image) for s in chosen]
        fused = self.reader.read_track(images)
        if fused is None or not fused.plate.is_valid:
            self.stats["unreadable"] += 1
            # An unreadable passage is still recorded. A vehicle that could not
            # be identified nonetheless passed, and pretending otherwise leaves
            # a hole in the record that looks identical to a missed detection.
            return self._emit(
                "unreadable_passage",
                {
                    "camera_id": self.cfg.camera_id,
                    "site_id": self.cfg.site_id,
                    "track_id": buf.track_id,
                    "first_seen": _iso(buf.first_seen),
                    "last_seen": _iso(buf.last_seen),
                    "n_crops": len(chosen),
                    "best_crop_quality": round(max(s.quality for s in chosen), 4),
                },
            )

        plate = fused.plate
        best = max(chosen, key=lambda s: s.quality)

        payload: dict = {
            "camera_id": self.cfg.camera_id,
            "site_id": self.cfg.site_id,
            "track_id": buf.track_id,
            "provisional": provisional,
            "plate": plate.canonical,
            "plate_display": plate.display,
            "system": plate.system.value,
            "ownership": plate.ownership.value,
            "size_class": plate.size_class.value,
            "confidence": plate.confidence.value,
            "score": round(plate.score, 4),
            "actionable": plate.confidence is self.cfg.auto_action_min_confidence,
            "n_frames": fused.n_frames,
            "agreement": round(fused.agreement, 4),
            "field_support": {k: round(v, 4) for k, v in fused.field_support.items()},
            "repaired_fields": [f.name for f in plate.fields if f.repaired],
            "warnings": list(plate.warnings),
            "alternatives": [p.canonical for p in fused.alternatives],
            "first_seen": _iso(buf.first_seen),
            "last_seen": _iso(buf.last_seen),
            "plate_box": [round(v, 1) for v in best.candidate.box],
        }

        if self.cfg.store_plate_image:
            # The plate crop itself carries no faces, so it is stored as-is;
            # the surrounding frame is what needs redaction.
            sha = self.queue.put_blob(_encode_jpeg(best.candidate.image, self.cfg.jpeg_quality))
            payload["plate_image_sha256"] = sha

        if self.cfg.store_context_image:
            redacted, n_faces = redact_frame(
                frame.image,
                vehicle_boxes=[t.box for t in self.tracker.tracks],
                cfg=self.cfg.privacy,
            )
            self.stats["faces_redacted"] += n_faces
            payload["context_image_sha256"] = self.queue.put_blob(
                _encode_jpeg(redacted, self.cfg.jpeg_quality)
            )
            payload["faces_redacted"] = n_faces

        if not provisional:
            self.stats["reads"] += 1
            touched = self.zones.attach_plate(buf.track_id, plate.canonical)
            if touched:
                payload["session_ids"] = [s.session_id for s in touched]
                # Back-fill sessions that were emitted before the plate was
                # known. The platform reconciles on session_id.
                self._emit(
                    "session_plate",
                    {
                        "camera_id": self.cfg.camera_id,
                        "plate": plate.canonical,
                        "confidence": plate.confidence.value,
                        "sessions": [s.session_id for s in touched],
                    },
                )
        else:
            self.stats["provisional"] += 1

        return self._emit("plate_read", payload)

    # -- zone events ----------------------------------------------------

    def _emit_zone_event(self, ev: ZoneEvent, frame: Frame) -> dict:
        if ev.kind is ZoneEventKind.ENTRY:
            self.stats["zone_entries"] += 1
            self._session_of_track.setdefault(ev.session.track_id, []).append(
                ev.session.session_id
            )
        else:
            self.stats["zone_exits"] += 1
        return self._emit(
            f"zone_{ev.kind.value}",
            {
                "camera_id": self.cfg.camera_id,
                "site_id": self.cfg.site_id,
                "zone_id": ev.session.zone_id,
                "frame_index": ev.frame_index,
                **ev.session.to_dict(),
            },
        )

    # -- output ---------------------------------------------------------

    def _emit(self, kind: str, payload: dict) -> dict:
        self.queue.append(kind, payload)
        if self.on_event is not None:
            self.on_event(kind, payload)
        return payload

    def flush(self) -> list[dict]:
        """Finish every open passage. Called at shutdown.

        Without this, a clean stop silently discards every vehicle currently in
        frame -- which on a busy junction is a meaningful number of reads, all
        of them lost precisely when an operator was doing something deliberate.
        """
        out: list[dict] = []
        now = Frame(index=self.tracker.frame_index, image=np.zeros((1, 1, 3), np.uint8),
                    timestamp=time.time())
        for tid, buf in list(self._buffers.items()):
            payload = self._recognise_and_emit(buf, now, provisional=False)
            if payload:
                out.append(payload)
            del self._buffers[tid]
        return out

    def snapshot(self) -> dict:
        """Live status, for telemetry and the operator console."""
        return {
            "camera_id": self.cfg.camera_id,
            "site_id": self.cfg.site_id,
            **self.stats,
            "active_tracks": sum(
                1 for t in self.tracker.tracks if t.state is TrackState.CONFIRMED
            ),
            "open_sessions": len(self.zones.open_sessions()),
            "occupancy": self.zones.occupancy(),
            "queue": self.queue.stats().__dict__,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _plate_region(vehicle_box) -> tuple[float, float, float, float]:
    """Where a plate sits on a vehicle box, when the detector did not say.

    Lower-middle: the bottom third vertically, the middle 70% horizontally.
    Crude, but it is only a fallback for a detector that reports vehicles
    without plates, and it narrows the search enough for crop selection to work.
    """
    x1, y1, x2, y2 = vehicle_box
    w, h = x2 - x1, y2 - y1
    return (x1 + w * 0.15, y1 + h * 0.62, x2 - w * 0.15, y2 - h * 0.04)


def _to_pil(bgr: np.ndarray):
    from PIL import Image

    return Image.fromarray(bgr[:, :, ::-1])


def _encode_jpeg(bgr: np.ndarray, quality: int) -> bytes:
    import cv2

    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return buf.tobytes()


def _dt(ts: float) -> datetime:
    return datetime.fromtimestamp(ts, tz=timezone.utc)


def _iso(ts: float) -> str:
    return _dt(ts).isoformat().replace("+00:00", "Z")


__all__ = ["PipelineConfig", "EdgePipeline"]
