"""Edge node configuration.

One file describes a site: which cameras, which zones, where the model is, where
to send events. It is what a field technician edits, so it is YAML with sane
defaults and it validates loudly.

Validation is strict on purpose. A zone polygon with two points, a camera with
no URL or a retention period of zero are all silently survivable at startup and
catastrophic later -- the node runs, appears healthy, and quietly records
nothing useful. Every one of those is a hard failure here, with a message
naming the field.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from .privacy import PrivacyConfig
from .zones import Zone, zones_from_config


@dataclass(frozen=True)
class CameraConfig:
    camera_id: str
    source: str                  # rtsp:// URL, video file, or image directory
    site_id: str = "unknown"
    name: str = ""
    target_fps: float = 15.0
    zones: tuple[Zone, ...] = ()
    #: Free-text siting record captured at commissioning. Not decoration: a
    #: read's evidential value depends on knowing where the camera pointed, and
    #: that is not recoverable after the fact.
    mounting_notes: str = ""


@dataclass(frozen=True)
class NodeConfig:
    node_id: str
    cameras: tuple[CameraConfig, ...]
    data_dir: Path = Path("/var/lib/thescanner")
    key_path: Path | None = None
    model_path: Path | None = None       # recogniser ONNX
    detector_path: Path | None = None    # detector ONNX; motion fallback if absent
    platform_url: str = ""
    platform_token: str = ""
    edge_retention_days: int = 7
    privacy: PrivacyConfig = field(default_factory=PrivacyConfig)
    beam_width: int = 12

    @property
    def resolved_key_path(self) -> Path:
        return self.key_path or (self.data_dir / "node.key")

    def queue_dir(self, camera_id: str) -> Path:
        return self.data_dir / "queue" / camera_id


def _load_mapping(path: Path) -> Mapping[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover - deployment issue
            raise RuntimeError(
                f"{path} is YAML but PyYAML is not installed; "
                f"install it or use JSON"
            ) from exc
        return yaml.safe_load(text) or {}
    return json.loads(text)


def load(path: str | Path) -> NodeConfig:
    """Load and validate a node configuration."""
    path = Path(path)
    raw = _load_mapping(path)

    def require(mapping: Mapping[str, Any], key: str, where: str) -> Any:
        if key not in mapping or mapping[key] in (None, ""):
            raise ValueError(f"{path}: {where} is missing required field {key!r}")
        return mapping[key]

    node_id = str(require(raw, "node_id", "config"))
    cameras_raw = require(raw, "cameras", "config")
    if not isinstance(cameras_raw, Sequence) or not cameras_raw:
        raise ValueError(f"{path}: 'cameras' must be a non-empty list")

    cameras: list[CameraConfig] = []
    seen: set[str] = set()
    for i, c in enumerate(cameras_raw):
        where = f"cameras[{i}]"
        cam_id = str(require(c, "camera_id", where))
        if cam_id in seen:
            # Duplicate ids would interleave two cameras' reads into one hash
            # chain, which makes both unverifiable.
            raise ValueError(f"{path}: duplicate camera_id {cam_id!r}")
        seen.add(cam_id)
        cameras.append(
            CameraConfig(
                camera_id=cam_id,
                source=str(require(c, "source", where)),
                site_id=str(c.get("site_id", raw.get("site_id", "unknown"))),
                name=str(c.get("name", cam_id)),
                target_fps=float(c.get("target_fps", 15.0)),
                zones=tuple(zones_from_config(c.get("zones", []))),
                mounting_notes=str(c.get("mounting_notes", "")),
            )
        )

    retention = int(raw.get("edge_retention_days", 7))
    if retention < 1:
        raise ValueError(
            f"{path}: edge_retention_days must be at least 1 -- a node with no "
            f"local retention loses every read the moment the uplink is down, "
            f"which defeats the point of store-and-forward"
        )

    p = raw.get("privacy", {}) or {}
    privacy = PrivacyConfig(
        enabled=bool(p.get("enabled", True)),
        dilate=float(p.get("dilate", 0.35)),
        blur_fraction=float(p.get("blur_fraction", 0.45)),
        min_face_px=int(p.get("min_face_px", 18)),
        redact_cabin=bool(p.get("redact_cabin", True)),
        cabin_fraction=float(p.get("cabin_fraction", 0.45)),
    )

    return NodeConfig(
        node_id=node_id,
        cameras=tuple(cameras),
        data_dir=Path(raw.get("data_dir", "/var/lib/thescanner")),
        key_path=Path(raw["key_path"]) if raw.get("key_path") else None,
        model_path=Path(raw["model_path"]) if raw.get("model_path") else None,
        detector_path=Path(raw["detector_path"]) if raw.get("detector_path") else None,
        platform_url=str(raw.get("platform_url", "")),
        platform_token=str(raw.get("platform_token", "")),
        edge_retention_days=retention,
        privacy=privacy,
        beam_width=int(raw.get("beam_width", 12)),
    )


EXAMPLE_CONFIG = """\
# TheScanner edge node configuration
#
# One file per node. A node runs one or more cameras; each camera may define
# zones whose entry and exit are recorded.

node_id: KTM-BAL-01
site_id: balkumari-junction
data_dir: /var/lib/thescanner

# Recogniser, exported with `python -m scanner_models.export`.
model_path: /opt/thescanner/models/platenet.onnx

# Optional. Any Apache-2.0 detector exported to ONNX (D-FINE, RT-DETRv2, DEIM).
# If omitted the node falls back to motion detection, which is adequate for a
# fixed camera but is a bootstrap rather than the production path.
# detector_path: /opt/thescanner/models/detector.onnx

platform_url: https://scanner.example.gov.np
platform_token: ${SCANNER_NODE_TOKEN}

# Short by design. The node is a buffer; the platform is the archive.
edge_retention_days: 7

privacy:
  enabled: true
  redact_cabin: true    # do not rely on face-detector recall alone

cameras:
  - camera_id: KTM-BAL-01-N
    name: Balkumari north approach
    source: rtsp://10.20.0.11:554/stream1
    target_fps: 15
    mounting_notes: >
      Pole 4, 6.2 m, facing south along the northbound carriageway,
      commissioned 2026-08-01.
    zones:
      - zone_id: junction-box
        name: Junction box
        # Frame pixel coordinates, clockwise.
        polygon: [[420, 300], [1500, 300], [1620, 900], [300, 900]]
        max_dwell_seconds: 600
"""


__all__ = ["CameraConfig", "NodeConfig", "load", "EXAMPLE_CONFIG"]
