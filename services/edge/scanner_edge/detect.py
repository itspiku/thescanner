"""Detection: finding vehicles and plates in a frame.

Deliberately a **pluggable interface with a permissive-licence default**, not a
single bundled model.

The licensing matters more than it might seem. Ultralytics YOLO is AGPL-3.0;
embedding it in a system a government operates creates a copyleft obligation
over derivative work, which is a procurement blocker (see
``docs/architecture.md``). So the production path is :class:`OnnxDetector`,
which consumes any exported ONNX detector -- D-FINE, RT-DETRv2, DEIM, all
Apache-2.0 -- and the choice of weights stays a deployment decision rather than
something baked into the source tree.

Honest status
-------------
A *trained* Nepali vehicle/plate detector is Phase 2.1 and does not exist yet,
because training one needs annotated road scenes and the public Nepali data is
plate crops, not scenes (see ``docs/research/datasets.md``). Until that lands,
:class:`MotionDetector` makes the rest of the pipeline runnable end to end
today: on a fixed camera, frame differencing finds moving vehicles perfectly
adequately. It is a bootstrap, not the answer, and it says so.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np

from .tracker import Box, Detection


class Detector(Protocol):
    """Anything that turns a BGR frame into detections."""

    def detect(self, frame: np.ndarray) -> list[Detection]: ...


# ---------------------------------------------------------------------------
# ONNX -- the production path
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class OnnxDetectorConfig:
    input_size: tuple[int, int] = (640, 640)
    conf_threshold: float = 0.25
    iou_threshold: float = 0.55
    #: Class index -> label. Defaults suit a two-class vehicle/plate model.
    labels: tuple[str, ...] = ("vehicle", "plate")
    #: Some exports emit xywh (YOLO), others xyxy (DETR family). Autodetected
    #: when None, but can be pinned when autodetection guesses wrong.
    box_format: str | None = None


class OnnxDetector:
    """Runs any exported ONNX detector.

    Output layouts differ between model families and even between export
    scripts, so the parser sniffs the shape rather than assuming one. That is
    less elegant than committing to a single model, and much more useful: a
    deployment can swap detectors without touching this code.
    """

    def __init__(
        self,
        model_path: str | Path,
        cfg: OnnxDetectorConfig | None = None,
        providers: Sequence[str] | None = None,
    ) -> None:
        import onnxruntime as ort

        self.cfg = cfg or OnnxDetectorConfig()
        available = set(ort.get_available_providers())
        wanted = providers or (
            "TensorrtExecutionProvider", "CUDAExecutionProvider",
            "OpenVINOExecutionProvider", "CPUExecutionProvider",
        )
        chosen = [p for p in wanted if p in available] or ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(str(model_path), providers=chosen)
        self.provider = self.session.get_providers()[0]
        self.input_name = self.session.get_inputs()[0].name

    def detect(self, frame: np.ndarray) -> list[Detection]:
        h, w = frame.shape[:2]
        tw, th = self.cfg.input_size
        # Letterbox rather than stretch: a distorted aspect ratio measurably
        # hurts detectors trained on letterboxed data, which is most of them.
        scale = min(tw / w, th / h)
        nw, nh = int(w * scale), int(h * scale)
        canvas = np.full((th, tw, 3), 114, dtype=np.uint8)
        import cv2

        canvas[:nh, :nw] = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
        blob = canvas[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0

        raw = self.session.run(None, {self.input_name: blob})[0]
        boxes, scores, classes = self._parse(np.asarray(raw))
        keep = _nms(boxes, scores, self.cfg.iou_threshold)

        out: list[Detection] = []
        for i in keep:
            if scores[i] < self.cfg.conf_threshold:
                continue
            x1, y1, x2, y2 = boxes[i] / scale
            cls = int(classes[i])
            label = self.cfg.labels[cls] if cls < len(self.cfg.labels) else str(cls)
            out.append(
                Detection(
                    box=(
                        float(np.clip(x1, 0, w)), float(np.clip(y1, 0, h)),
                        float(np.clip(x2, 0, w)), float(np.clip(y2, 0, h)),
                    ),
                    score=float(scores[i]),
                    label=label,
                )
            )
        return out

    def _parse(self, raw: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if raw.ndim == 3:
            raw = raw[0]
        # Some exports are (channels, anchors); transpose to (anchors, channels).
        if raw.shape[0] < raw.shape[1] and raw.shape[0] <= 100:
            raw = raw.T

        n_extra = raw.shape[1] - 4
        if n_extra == 1:
            scores = raw[:, 4]
            classes = np.zeros(len(raw))
        elif n_extra >= 2 and self._has_objectness(raw):
            scores = raw[:, 4] * raw[:, 5:].max(axis=1)
            classes = raw[:, 5:].argmax(axis=1)
        else:
            scores = raw[:, 4:].max(axis=1)
            classes = raw[:, 4:].argmax(axis=1)

        boxes = raw[:, :4].astype(np.float32)
        fmt = self.cfg.box_format or ("xywh" if self._looks_xywh(boxes) else "xyxy")
        if fmt == "xywh":
            cx, cy, bw, bh = boxes.T
            boxes = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], axis=1)
        return boxes, scores.astype(np.float32), classes.astype(np.int32)

    @staticmethod
    def _has_objectness(raw: np.ndarray) -> bool:
        """YOLOv5/v7 carry a separate objectness column; v8+ and DETR do not."""
        col = raw[:, 4]
        return bool((col >= 0).all() and (col <= 1).all() and raw.shape[1] > 6)

    @staticmethod
    def _looks_xywh(boxes: np.ndarray) -> bool:
        """xyxy boxes have x2 > x1 almost everywhere; xywh usually do not."""
        if len(boxes) == 0:
            return False
        return bool((boxes[:, 2] > boxes[:, 0]).mean() < 0.9)


def _nms(boxes: np.ndarray, scores: np.ndarray, threshold: float) -> list[int]:
    if len(boxes) == 0:
        return []
    order = scores.argsort()[::-1]
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    keep: list[int] = []
    while order.size:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(boxes[i, 0], boxes[rest, 0])
        yy1 = np.maximum(boxes[i, 1], boxes[rest, 1])
        xx2 = np.minimum(boxes[i, 2], boxes[rest, 2])
        yy2 = np.minimum(boxes[i, 3], boxes[rest, 3])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        iou = inter / np.maximum(areas[i] + areas[rest] - inter, 1e-6)
        order = rest[iou <= threshold]
    return keep


# ---------------------------------------------------------------------------
# Motion -- the bootstrap path
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MotionDetectorConfig:
    min_area_fraction: float = 0.004
    max_area_fraction: float = 0.60
    #: Vehicles are wider than they are tall from a roadside camera; this
    #: rejects poles, pedestrians and shadows cast along the kerb.
    min_aspect: float = 0.6
    max_aspect: float = 5.0
    history: int = 300
    var_threshold: float = 24.0
    learning_rate: float = 0.004


class MotionDetector:
    """Background-subtraction detector for a *fixed* camera.

    Not a substitute for a trained detector, and not presented as one. But on a
    static pole-mounted camera -- which is every camera in this deployment --
    moving foreground blobs are vehicles with high reliability, and that is
    enough to exercise tracking, zone sessions, recognition and the evidence
    chain end to end before Phase 2.1 delivers real weights.

    It also has a genuine production use as a **gate**: running it ahead of a
    neural detector and skipping frames with no motion cuts inference cost
    dramatically on a road that is empty most of the night.
    """

    def __init__(self, cfg: MotionDetectorConfig | None = None) -> None:
        import cv2

        self.cfg = cfg or MotionDetectorConfig()
        self._bg = cv2.createBackgroundSubtractorMOG2(
            history=self.cfg.history,
            varThreshold=self.cfg.var_threshold,
            detectShadows=True,
        )
        self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    def detect(self, frame: np.ndarray) -> list[Detection]:
        import cv2

        h, w = frame.shape[:2]
        mask = self._bg.apply(frame, learningRate=self.cfg.learning_rate)
        # MOG2 marks shadows as 127; treating them as foreground merges a
        # vehicle with the shadow it casts and doubles the apparent box.
        mask[mask == 127] = 0
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._kernel, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        frame_area = float(h * w)
        out: list[Detection] = []
        for c in contours:
            x, y, bw, bh = cv2.boundingRect(c)
            area = bw * bh
            frac = area / frame_area
            if not (self.cfg.min_area_fraction <= frac <= self.cfg.max_area_fraction):
                continue
            aspect = bw / max(bh, 1)
            if not (self.cfg.min_aspect <= aspect <= self.cfg.max_aspect):
                continue
            # Fill ratio separates solid vehicles from the ragged blobs that
            # foliage movement and rain produce.
            fill = cv2.contourArea(c) / max(area, 1)
            if fill < 0.35:
                continue
            out.append(
                Detection(
                    box=(float(x), float(y), float(x + bw), float(y + bh)),
                    score=float(min(0.95, 0.45 + fill * 0.5)),
                    label="vehicle",
                )
            )
        return out


class StaticDetector:
    """Replays a fixed script of detections. For tests and replay harnesses."""

    def __init__(self, per_frame: Sequence[Sequence[Detection]]) -> None:
        self.per_frame = list(per_frame)
        self.i = 0

    def detect(self, frame: np.ndarray) -> list[Detection]:
        out = list(self.per_frame[self.i]) if self.i < len(self.per_frame) else []
        self.i += 1
        return out


__all__ = [
    "Detector",
    "OnnxDetector",
    "OnnxDetectorConfig",
    "MotionDetector",
    "MotionDetectorConfig",
    "StaticDetector",
]
