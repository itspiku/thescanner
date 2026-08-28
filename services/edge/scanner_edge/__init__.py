"""``scanner_edge`` -- the edge agent.

Runs on a box at the roadside: decode camera frames, detect and track vehicles,
select the informative crops of each passage, recognise the plate by fusing
across them, record zone entry and exit, redact faces, sign everything, and hold
it in a durable local queue until the platform acknowledges it.

The shape of this service follows from two facts about the deployment. Nepal has
load-shedding and unreliable connectivity, so the node must keep working with no
uplink for days -- hence store-and-forward as the primary write path rather than
a fallback. And the reads may become evidence, so origin and integrity have to
be provable after the fact -- hence per-node Ed25519 signing and a hash-chained
log, applied before anything reaches durable storage.

::

    scanner-edge init   --out node.yaml
    scanner-edge enrol  --config node.yaml
    scanner-edge run    --config node.yaml
    scanner-edge verify --config node.yaml     # re-check the evidence chain
"""

from __future__ import annotations

from .config import CameraConfig, NodeConfig, load
from scanner_evidence import NodeIdentity, SignedEvent, verify, verify_chain
from .pipeline import EdgePipeline, PipelineConfig
from .queue import EventQueue
from .tracker import ByteTracker, Detection, Track
from .zones import Zone, ZoneEngine, ZoneEvent, ZoneSession

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "CameraConfig",
    "NodeConfig",
    "load",
    "NodeIdentity",
    "SignedEvent",
    "verify",
    "verify_chain",
    "EventQueue",
    "ByteTracker",
    "Detection",
    "Track",
    "Zone",
    "ZoneEngine",
    "ZoneEvent",
    "ZoneSession",
    "EdgePipeline",
    "PipelineConfig",
]
