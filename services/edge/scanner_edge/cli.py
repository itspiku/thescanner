"""Edge node command line.

    scanner-edge init      --out node.yaml
    scanner-edge enrol     --config node.yaml
    scanner-edge run       --config node.yaml
    scanner-edge sync      --config node.yaml
    scanner-edge verify    --config node.yaml --camera KTM-BAL-01-N
    scanner-edge status    --config node.yaml

``verify`` is the one worth knowing about: it re-checks a node's entire hash
chain -- every signature, every link, every sequence number -- and reports the
exact point of any break. It is what an investigator runs before relying on a
read, and what an auditor runs on a seized node.
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
import threading
import time
from pathlib import Path

from . import config as config_mod
from .detect import MotionDetector, OnnxDetector, OnnxDetectorConfig
from scanner_evidence import NodeIdentity, verify_chain
from .pipeline import EdgePipeline, PipelineConfig
from .queue import EventQueue
from .sources import open_source
from .uplink import Uplink, UplinkConfig, UplinkRejected
from .zones import ZoneEngine


def _identity(cfg: config_mod.NodeConfig, camera_id: str) -> NodeIdentity:
    # One chain per camera, not per node. Interleaving two cameras' reads into
    # a single chain would make each unverifiable in isolation, and a read is
    # usually presented as evidence on its own.
    return NodeIdentity.load_or_create(cfg.resolved_key_path, f"{cfg.node_id}:{camera_id}")


def _queue(cfg: config_mod.NodeConfig, camera_id: str) -> EventQueue:
    root = cfg.queue_dir(camera_id)
    root.mkdir(parents=True, exist_ok=True)
    return EventQueue(root, _identity(cfg, camera_id))


def _reader(cfg: config_mod.NodeConfig):
    if cfg.model_path is None or not Path(cfg.model_path).is_file():
        raise SystemExit(
            f"recogniser model not found at {cfg.model_path}. "
            f"Export one with: python -m scanner_models.export "
            f"--checkpoint models/platenet/best.pt --out models/platenet.onnx"
        )
    from scanner_models.infer import from_onnx

    return from_onnx(cfg.model_path, beam_width=cfg.beam_width)


def _detector(cfg: config_mod.NodeConfig):
    if cfg.detector_path and Path(cfg.detector_path).is_file():
        return OnnxDetector(cfg.detector_path, OnnxDetectorConfig())
    print(
        "note: no detector model configured; falling back to motion detection. "
        "Adequate for a fixed camera, but a trained detector is the production "
        "path (see docs/PLAN.md, Phase 2.1).",
        file=sys.stderr,
    )
    return MotionDetector()


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_init(args) -> int:
    out = Path(args.out)
    if out.exists() and not args.force:
        print(f"{out} already exists; pass --force to overwrite", file=sys.stderr)
        return 1
    out.write_text(config_mod.EXAMPLE_CONFIG, encoding="utf-8")
    print(f"wrote {out}")
    print("Edit it, then run: scanner-edge enrol --config", out)
    return 0


def cmd_enrol(args) -> int:
    cfg = config_mod.load(args.config)
    if not cfg.platform_url:
        print("platform_url is not set; nothing to enrol with", file=sys.stderr)
        return 1
    for cam in cfg.cameras:
        with _queue(cfg, cam.camera_id) as q:
            up = Uplink(
                UplinkConfig(base_url=cfg.platform_url, node_id=q.identity.node_id,
                             token=cfg.platform_token),
                q,
            )
            record = q.identity.enrolment_record()
            print(f"{record['node_id']}  {record['public_key']}")
            if not args.print_only:
                try:
                    up.enrol()
                    print("  enrolled")
                except Exception as e:  # noqa: BLE001 - surfaced to the operator
                    print(f"  enrolment failed: {e}", file=sys.stderr)
                    return 1
    return 0


def cmd_run(args) -> int:
    cfg = config_mod.load(args.config)
    cam = next((c for c in cfg.cameras if c.camera_id == args.camera), None) if args.camera else cfg.cameras[0]
    if cam is None:
        print(f"no camera {args.camera!r} in config", file=sys.stderr)
        return 1

    reader = _reader(cfg)
    detector = _detector(cfg)
    queue = _queue(cfg, cam.camera_id)
    pipeline = EdgePipeline(
        cfg=PipelineConfig(
            camera_id=cam.camera_id, site_id=cam.site_id, privacy=cfg.privacy
        ),
        detector=detector,
        reader=reader,
        queue=queue,
        zones=ZoneEngine(cam.zones),
    )

    stop = threading.Event()

    def _sync_loop() -> None:
        """Drain the queue in the background, so a slow uplink never stalls
        recognition. The queue is the buffer that makes this safe."""
        if not cfg.platform_url:
            return
        up = Uplink(
            UplinkConfig(base_url=cfg.platform_url, node_id=queue.identity.node_id,
                         token=cfg.platform_token),
            queue,
        )
        while not stop.is_set():
            try:
                up.drain()
            except UplinkRejected as e:
                print(f"uplink rejected: {e}", file=sys.stderr)
            stop.wait(up.backoff)

    def _retention_loop() -> None:
        while not stop.is_set():
            queue.prune(keep_days=cfg.edge_retention_days)
            stop.wait(3600)

    threads = [
        threading.Thread(target=_sync_loop, daemon=True, name="uplink"),
        threading.Thread(target=_retention_loop, daemon=True, name="retention"),
    ]
    for t in threads:
        t.start()

    def _handle(signum, _frame):
        # Flush on shutdown: without it a clean stop discards every vehicle
        # currently in frame, which on a busy junction is a real number of reads.
        print("\nstopping; flushing in-flight passages...", file=sys.stderr)
        stop.set()

    signal.signal(signal.SIGINT, _handle)
    try:
        signal.signal(signal.SIGTERM, _handle)
    except (AttributeError, ValueError):
        pass

    source = open_source(cam.source, **({"target_fps": cam.target_fps} if cam.source.startswith("rtsp") else {}))
    print(f"running {cam.camera_id} <- {cam.source}")
    t0 = time.time()
    try:
        stats = pipeline.run(source, max_frames=args.max_frames)
    finally:
        stop.set()

    elapsed = time.time() - t0
    stats["elapsed_s"] = round(elapsed, 1)
    stats["fps"] = round(stats["frames"] / max(elapsed, 1e-6), 2)
    print(json.dumps(stats, indent=2))
    queue.close()
    return 0


def cmd_sync(args) -> int:
    cfg = config_mod.load(args.config)
    if not cfg.platform_url:
        print("platform_url is not set", file=sys.stderr)
        return 1
    total = 0
    for cam in cfg.cameras:
        with _queue(cfg, cam.camera_id) as q:
            up = Uplink(
                UplinkConfig(base_url=cfg.platform_url, node_id=q.identity.node_id,
                             token=cfg.platform_token),
                q,
            )
            try:
                n = up.drain()
            except UplinkRejected as e:
                print(f"{cam.camera_id}: rejected: {e}", file=sys.stderr)
                return 1
            total += n
            print(f"{cam.camera_id}: sent {n}, {q.stats().unsent} remaining")
    print(f"total sent: {total}")
    return 0


def cmd_verify(args) -> int:
    """Re-verify a node's evidence chain end to end."""
    cfg = config_mod.load(args.config)
    cams = [c for c in cfg.cameras if not args.camera or c.camera_id == args.camera]
    failed = False
    for cam in cams:
        with _queue(cfg, cam.camera_id) as q:
            events = list(q.iter_all())
            pub = q.identity.public_key_b64()
            if not events:
                print(f"{cam.camera_id}: empty chain")
                continue
            # A chain that does not start at sequence 1 has been pruned; verify
            # from the first surviving link rather than reporting a false break.
            start = events[0].prev_hash
            ok, reason = verify_chain(events, pub, start_hash=start)
            status = "OK" if ok else f"BROKEN -- {reason}"
            print(
                f"{cam.camera_id}: {len(events):,} events, "
                f"sequence {events[0].sequence}-{events[-1].sequence}: {status}"
            )
            failed |= not ok
    return 1 if failed else 0


def cmd_status(args) -> int:
    cfg = config_mod.load(args.config)
    out = {"node_id": cfg.node_id, "cameras": []}
    for cam in cfg.cameras:
        with _queue(cfg, cam.camera_id) as q:
            s = q.stats()
            out["cameras"].append({
                "camera_id": cam.camera_id,
                "source": cam.source,
                "zones": [z.zone_id for z in cam.zones],
                "queue": {
                    "total": s.total, "unsent": s.unsent,
                    "oldest_unsent": s.oldest_unsent, "disk_bytes": s.disk_bytes,
                },
                "public_key": q.identity.public_key_b64(),
            })
    print(json.dumps(out, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="scanner-edge", description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="command", required=True)

    i = sub.add_parser("init", help="write an example configuration")
    i.add_argument("--out", default="node.yaml")
    i.add_argument("--force", action="store_true")
    i.set_defaults(func=cmd_init)

    for name, fn, helptext in (
        ("enrol", cmd_enrol, "register this node's public key with the platform"),
        ("sync", cmd_sync, "drain the local queue to the platform"),
        ("status", cmd_status, "show queue and node status"),
    ):
        s = sub.add_parser(name, help=helptext)
        s.add_argument("--config", required=True)
        if name == "enrol":
            s.add_argument("--print-only", action="store_true",
                           help="print the public key without contacting the platform")
        s.set_defaults(func=fn)

    r = sub.add_parser("run", help="run a camera")
    r.add_argument("--config", required=True)
    r.add_argument("--camera", default=None, help="camera_id; defaults to the first")
    r.add_argument("--max-frames", type=int, default=None)
    r.set_defaults(func=cmd_run)

    v = sub.add_parser("verify", help="re-verify the evidence chain")
    v.add_argument("--config", required=True)
    v.add_argument("--camera", default=None)
    v.set_defaults(func=cmd_verify)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
