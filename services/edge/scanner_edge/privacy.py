"""On-device privacy: blur faces before any image leaves the node.

Nepal's Privacy Act 2075 governs the processing of personal data, and a
roadside camera unavoidably captures drivers, passengers and pedestrians
alongside the vehicles the system is authorised to identify. The design answer
is data minimisation at the point of capture: **redaction happens on the edge
node, before an image is written to the queue**, so an identifiable face never
reaches the platform, the network, or a backup.

That ordering is the entire point. Blurring at the API would mean unredacted
imagery crossed the network and sat in a queue; blurring in the console would
mean it was stored. Doing it here means the only copy that ever exists outside
volatile memory is already redacted.

Explicitly **not** face recognition
-----------------------------------
Faces are detected in order to destroy them, and no face descriptor, embedding
or identity is computed, stored or transmitted. The distance from a national
ANPR to a national face-surveillance network is short, and refusing to take
that step has to be structural rather than a line in a policy document. See
``docs/security-and-privacy.md``.

On the detector used
--------------------
OpenCV's bundled Haar cascade (BSD-licensed, ships with the wheel, no extra
model to procure or licence). It misses faces at profile angles and small
scales, which for a *redaction* task is the failure direction that matters --
so the pipeline compensates by dilating every detection generously and by
offering :func:`redact_occupant_area`, which blurs the whole cabin region
irrespective of whether a face was found. Relying on detector recall alone for
a privacy control would be a mistake.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Sequence

import numpy as np

from .tracker import Box


@dataclass(frozen=True)
class PrivacyConfig:
    enabled: bool = True
    #: Expand each detected face box by this fraction before blurring. Generous
    #: on purpose: a face half-covered by a blur is not redacted.
    dilate: float = 0.35
    #: Gaussian kernel as a fraction of the region's smaller side. Large enough
    #: that the result cannot be sharpened back.
    blur_fraction: float = 0.45
    min_face_px: int = 18
    #: Blur the upper cabin area of every detected vehicle regardless of face
    #: detection. Costs a little utility, buys independence from detector recall.
    redact_cabin: bool = True
    #: Fraction of a vehicle box, measured from the top, treated as cabin.
    cabin_fraction: float = 0.45


@lru_cache(maxsize=1)
def _cascade():
    import cv2

    path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    clf = cv2.CascadeClassifier(path)
    if clf.empty():
        raise RuntimeError(f"could not load face cascade from {path}")
    return clf


def detect_faces(frame: np.ndarray, cfg: PrivacyConfig | None = None) -> list[Box]:
    """Locate faces, solely so they can be destroyed."""
    import cv2

    cfg = cfg or PrivacyConfig()
    if frame.size == 0:
        return []
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    found = _cascade().detectMultiScale(
        grey, scaleFactor=1.12, minNeighbors=4,
        minSize=(cfg.min_face_px, cfg.min_face_px),
    )
    return [(float(x), float(y), float(x + w), float(y + h)) for x, y, w, h in found]


def blur_region(frame: np.ndarray, box: Box, cfg: PrivacyConfig | None = None) -> None:
    """Irreversibly blur one region, in place.

    Gaussian with a kernel scaled to the region rather than a fixed size: a
    fixed kernel that adequately obscures a 20 px face leaves a 200 px face
    perfectly legible.
    """
    import cv2

    cfg = cfg or PrivacyConfig()
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = box
    dw, dh = (x2 - x1) * cfg.dilate, (y2 - y1) * cfg.dilate
    x1 = int(max(0, x1 - dw))
    y1 = int(max(0, y1 - dh))
    x2 = int(min(w, x2 + dw))
    y2 = int(min(h, y2 + dh))
    if x2 - x1 < 2 or y2 - y1 < 2:
        return
    region = frame[y1:y2, x1:x2]
    k = max(5, int(min(region.shape[:2]) * cfg.blur_fraction) | 1)
    frame[y1:y2, x1:x2] = cv2.GaussianBlur(region, (k, k), 0)


def redact_frame(
    frame: np.ndarray,
    *,
    vehicle_boxes: Sequence[Box] = (),
    cfg: PrivacyConfig | None = None,
) -> tuple[np.ndarray, int]:
    """Return a redacted copy of ``frame`` and the number of regions blurred.

    A *copy* is returned rather than mutating in place, because the caller still
    needs the original for recognition -- the plate must be read from unredacted
    pixels, and only the stored evidence image is redacted.
    """
    cfg = cfg or PrivacyConfig()
    if not cfg.enabled or frame.size == 0:
        return frame.copy(), 0

    out = frame.copy()
    n = 0
    for box in detect_faces(frame, cfg):
        blur_region(out, box, cfg)
        n += 1

    if cfg.redact_cabin:
        for x1, y1, x2, y2 in vehicle_boxes:
            cabin = (x1, y1, x2, y1 + (y2 - y1) * cfg.cabin_fraction)
            blur_region(out, cabin, cfg)
            n += 1
    return out, n


def redact_occupant_area(frame: np.ndarray, vehicle_box: Box, cfg: PrivacyConfig | None = None) -> np.ndarray:
    """Blur a vehicle's cabin without relying on face detection at all.

    Use this when the imagery is being retained as evidence and detector recall
    is not something you are willing to bet a privacy obligation on.
    """
    cfg = cfg or PrivacyConfig()
    out = frame.copy()
    x1, y1, x2, y2 = vehicle_box
    blur_region(out, (x1, y1, x2, y1 + (y2 - y1) * cfg.cabin_fraction), cfg)
    return out


__all__ = [
    "PrivacyConfig",
    "detect_faces",
    "blur_region",
    "redact_frame",
    "redact_occupant_area",
]
