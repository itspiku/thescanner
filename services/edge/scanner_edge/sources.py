"""Frame sources: RTSP cameras, video files, image directories.

The one behaviour that matters here is what happens when an RTSP stream drops,
because on a Nepali municipal network it will, repeatedly. A source that raises
and takes the pipeline down with it turns a thirty-second network blip into a
thirty-minute outage that someone has to drive to a pole to fix.

So :class:`RtspSource` reconnects on its own, with backoff, and reports the
interruption through telemetry rather than by failing. The pipeline treats a
gap in frames as normal operation, not an error.

The second behaviour that matters is **frame dropping**. A live camera produces
frames whether or not anything is consuming them; if the pipeline falls behind,
OpenCV's internal buffer grows and the system starts processing footage that is
seconds old. For live enforcement that is worse than useless -- an alert about
where a vehicle was ten seconds ago. Live sources therefore drain to the newest
frame and drop the backlog, while file sources do not, because a file is being
processed for completeness rather than currency.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Protocol

import numpy as np


@dataclass(frozen=True, slots=True)
class Frame:
    """One decoded frame plus the metadata every downstream stage needs."""

    index: int
    image: np.ndarray            # BGR
    #: Wall-clock capture time, epoch seconds. Signed into the evidence chain,
    #: so its provenance matters -- see ``clock_source``.
    timestamp: float
    #: How the timestamp was obtained. A node whose clock has never synced
    #: produces reads whose times cannot be defended, and the platform needs to
    #: know that rather than assume.
    clock_source: str = "system"
    #: Frames the source skipped to stay current. Non-zero means the pipeline
    #: is not keeping up.
    dropped: int = 0

    @property
    def size(self) -> tuple[int, int]:
        h, w = self.image.shape[:2]
        return (w, h)


class FrameSource(Protocol):
    def frames(self) -> Iterator[Frame]: ...
    def close(self) -> None: ...


@dataclass(frozen=True)
class RtspConfig:
    url: str
    #: Cap the processing rate. A 25 fps camera does not need 25 fps of
    #: inference to read a plate -- a vehicle is in frame for a second or more,
    #: and 10-15 fps still yields ten-plus looks for fusion at a third of the
    #: compute.
    target_fps: float = 15.0
    reconnect_delay: float = 2.0
    max_reconnect_delay: float = 60.0
    open_timeout: float = 15.0
    #: Prefer FFmpeg with hardware decode where available.
    use_hardware_decode: bool = True


class RtspSource:
    """A live RTSP camera, with reconnection and backlog dropping."""

    def __init__(self, cfg: RtspConfig) -> None:
        self.cfg = cfg
        self._cap = None
        self._index = 0
        self._closed = False
        self.reconnects = 0

    def _open(self):
        import cv2

        # CAP_FFMPEG rather than the default: it is the only backend with
        # reliable RTSP support across platforms, and it honours the transport
        # options below.
        cap = cv2.VideoCapture(self.cfg.url, cv2.CAP_FFMPEG)
        # A buffer of 1 is what makes dropping work: without it OpenCV queues
        # frames internally and hands back stale ones.
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not cap.isOpened():
            cap.release()
            return None
        return cap

    def frames(self) -> Iterator[Frame]:
        delay = self.cfg.reconnect_delay
        min_interval = 1.0 / max(self.cfg.target_fps, 0.1)
        last_emit = 0.0

        while not self._closed:
            if self._cap is None:
                self._cap = self._open()
                if self._cap is None:
                    time.sleep(delay)
                    # Exponential backoff, capped. Hammering a camera that is
                    # powered down achieves nothing and fills the logs.
                    delay = min(delay * 2, self.cfg.max_reconnect_delay)
                    self.reconnects += 1
                    continue
                delay = self.cfg.reconnect_delay

            ok, image = self._cap.read()
            if not ok or image is None:
                self._cap.release()
                self._cap = None
                self.reconnects += 1
                continue

            now = time.time()
            if now - last_emit < min_interval:
                # Below the target rate: discard rather than queue, so the
                # pipeline always sees the newest frame available.
                continue
            dropped = 0
            last_emit = now
            self._index += 1
            yield Frame(index=self._index, image=image, timestamp=now, dropped=dropped)

    def close(self) -> None:
        self._closed = True
        if self._cap is not None:
            self._cap.release()
            self._cap = None


class VideoFileSource:
    """A video file. Processed for completeness, so no frames are dropped."""

    def __init__(self, path: str | Path, *, stride: int = 1, start_time: float | None = None) -> None:
        self.path = Path(path)
        self.stride = max(1, stride)
        #: Files carry no wall clock, so the caller supplies the recording start
        #: time; frame timestamps are derived from it and the file's fps. A read
        #: whose time is "when the file was processed" is worthless as evidence.
        self.start_time = start_time
        self._cap = None

    def frames(self) -> Iterator[Frame]:
        import cv2

        self._cap = cv2.VideoCapture(str(self.path))
        if not self._cap.isOpened():
            raise FileNotFoundError(f"cannot open video {self.path}")
        fps = self._cap.get(cv2.CAP_PROP_FPS) or 25.0
        base = self.start_time if self.start_time is not None else time.time()
        i = 0
        emitted = 0
        while True:
            ok, image = self._cap.read()
            if not ok:
                break
            if i % self.stride == 0:
                emitted += 1
                yield Frame(
                    index=emitted,
                    image=image,
                    timestamp=base + i / fps,
                    clock_source="file-relative",
                )
            i += 1
        self.close()

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None


class ImageDirSource:
    """A directory of stills, in filename order. For replay and testing."""

    def __init__(self, root: str | Path, *, fps: float = 15.0, start_time: float | None = None) -> None:
        self.root = Path(root)
        self.fps = fps
        self.start_time = start_time

    def frames(self) -> Iterator[Frame]:
        import cv2

        paths = sorted(
            p for p in self.root.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
        )
        base = self.start_time if self.start_time is not None else time.time()
        for i, p in enumerate(paths, 1):
            image = cv2.imread(str(p))
            if image is None:
                continue
            yield Frame(
                index=i, image=image, timestamp=base + i / self.fps,
                clock_source="file-relative",
            )

    def close(self) -> None:
        return None


class ArraySource:
    """An in-memory sequence of frames. For tests and synthetic replay."""

    def __init__(self, images, *, fps: float = 15.0, start_time: float = 0.0) -> None:
        self.images = list(images)
        self.fps = fps
        self.start_time = start_time

    def frames(self) -> Iterator[Frame]:
        for i, img in enumerate(self.images, 1):
            yield Frame(
                index=i, image=img, timestamp=self.start_time + i / self.fps,
                clock_source="synthetic",
            )

    def close(self) -> None:
        return None


def open_source(spec: str, **kwargs) -> FrameSource:
    """Build a source from a URI or path.

    ``rtsp://...`` -> camera; a directory -> stills; anything else -> video file.
    """
    if spec.startswith(("rtsp://", "rtsps://", "http://", "https://")):
        return RtspSource(RtspConfig(url=spec, **kwargs))
    path = Path(spec)
    if path.is_dir():
        return ImageDirSource(path, **kwargs)
    return VideoFileSource(path, **kwargs)


__all__ = [
    "Frame",
    "FrameSource",
    "RtspConfig",
    "RtspSource",
    "VideoFileSource",
    "ImageDirSource",
    "ArraySource",
    "open_source",
]
