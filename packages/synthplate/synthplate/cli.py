"""Corpus generation CLI.

    synthplate generate --out data/synth --count 200000
    synthplate tracks   --out data/tracks --count 20000
    synthplate preview  --out preview.png

Output is a directory of images plus a JSONL manifest -- one JSON object per
sample, holding the label, the field decomposition, the character boxes and the
full degradation record. JSONL rather than a packed format on purpose: it is
streamable, appendable, diffable, and readable with nothing but a text editor,
which matters when someone has to audit what a government model was trained on.

Every run writes a ``manifest.json`` recording the seed, the config, the package
versions and **whether authentic plate fonts were used**. A training run can
refuse to proceed on fallback-rendered data, and an auditor can reproduce the
corpus exactly from the seed.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import random
import sys
import time
from pathlib import Path

from PIL import Image

from nepal_plate import __version__ as nepal_plate_version

from . import __version__, fonts
from .degrade import DegradationConfig, DegradedPlate, degrade
from .render import render
from .sampling import PlateSampler, sample_balanced_grid
from .tracks import TrackConfig, synth_track


def _record(d: DegradedPlate, path: str, *, track_id: str | None = None, frame: int | None = None) -> dict:
    p = d.source.sample.plate
    rec = {
        "path": path,
        "canonical": p.canonical,
        "display": p.display,
        "system": p.system.value,
        "tokens": list(d.source.sample.tokens),
        "colour": d.source.sample.colour.value,
        "ownership": p.ownership.value,
        "size_class": p.size_class.value,
        "fields": {
            "zone": p.zone, "lot": p.lot, "vehicle_class": p.vehicle_class,
            "province": p.province, "class_letter": p.class_letter,
            "series": p.series, "serial": p.serial,
        },
        "two_row": d.source.sample.two_row,
        "width_px": d.image.width,
        "height_px": d.image.height,
        "quality": round(d.quality, 4),
        "difficulty": round(d.difficulty, 4),
        "degradations": [[k, round(v, 4)] for k, v in d.applied],
        "char_boxes": [
            {"t": b.token, "box": [round(b.x0, 2), round(b.y0, 2), round(b.x1, 2), round(b.y1, 2)], "row": b.row}
            for b in d.char_boxes
        ],
        "plate_quad": [[round(x, 2), round(y, 2)] for x, y in d.corners],
        "font_authentic": d.source.font_authentic,
        "meta": {k: v for k, v in d.meta.items() if k not in ("bg_rgb", "fg_rgb")},
    }
    if track_id is not None:
        rec["track_id"] = track_id
        rec["frame"] = frame
    return rec


def _write_manifest(out: Path, args, extra: dict) -> None:
    manifest = {
        "generator": "synthplate",
        "synthplate_version": __version__,
        "nepal_plate_version": nepal_plate_version,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seed": args.seed,
        "args": {k: v for k, v in vars(args).items() if k != "func"},
        "fonts_authentic": fonts.fonts_are_authentic(),
        "fonts": {
            s: {"file": fonts.resolve(s).name, "authentic": fonts.resolve(s).authentic}
            for s in ("latin", "devanagari")
        },
        "degradation_config": dataclasses.asdict(DegradationConfig()),
        **extra,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if not manifest["fonts_authentic"]:
        print(
            "\nWARNING: rendered with fallback fonts, not the authentic plate typefaces.\n"
            "         Fine for development; do NOT train a deployed model on this corpus.\n"
            "         Run 'python -m synthplate.fetch_fonts' first.",
            file=sys.stderr,
        )


def cmd_generate(args: argparse.Namespace) -> int:
    out = Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    sampler = PlateSampler(
        seed=args.seed, strategy=args.strategy, embossed_fraction=args.embossed_fraction
    )
    cfg = DegradationConfig()

    # Seed the corpus with an exhaustive structural sweep, so no (zone, class)
    # or (province, class) combination is missing -- random sampling only
    # achieves that in expectation, and the rare classes are the ones that
    # matter most.
    grid = list(sample_balanced_grid(seed=args.seed)) if args.grid else []
    n_grid = min(len(grid), args.count)

    t0 = time.time()
    written = 0
    with (out / "index.jsonl").open("w", encoding="utf-8") as fh:
        for i in range(args.count):
            s = grid[i] if i < n_grid else sampler.sample()
            d = degrade(render(s, height=args.render_height, rng=rng), rng=rng, cfg=cfg)
            # Shard into subdirectories: 200k files in one directory is painful
            # on every filesystem and unusable on some.
            shard = f"{i // 1000:05d}"
            (out / "images" / shard).mkdir(exist_ok=True)
            rel = f"images/{shard}/{i:08d}.jpg"
            d.image.save(out / rel, quality=args.save_quality)
            fh.write(json.dumps(_record(d, rel), ensure_ascii=False) + "\n")
            written += 1
            if written % 2000 == 0:
                rate = written / (time.time() - t0)
                print(f"  {written:,}/{args.count:,}  ({rate:.0f}/s)", file=sys.stderr)

    _write_manifest(out, args, {"kind": "images", "n_samples": written, "grid_seeded": n_grid})
    print(f"wrote {written:,} samples to {out} in {time.time()-t0:.1f}s")
    return 0


def cmd_tracks(args: argparse.Namespace) -> int:
    out = Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    sampler = PlateSampler(
        seed=args.seed, strategy=args.strategy, embossed_fraction=args.embossed_fraction
    )
    cfg = DegradationConfig()
    tcfg = TrackConfig()

    t0 = time.time()
    n_frames = 0
    with (out / "index.jsonl").open("w", encoding="utf-8") as fh:
        for i in range(args.count):
            track = synth_track(
                sampler.sample(), rng=rng, cfg=cfg, track_cfg=tcfg,
                render_height=args.render_height,
            )
            tid = f"t{i:07d}"
            shard = f"{i // 1000:05d}"
            (out / "images" / shard).mkdir(exist_ok=True)
            for j, frame in enumerate(track.frames):
                rel = f"images/{shard}/{tid}_{j:02d}.jpg"
                frame.image.save(out / rel, quality=args.save_quality)
                fh.write(
                    json.dumps(_record(frame, rel, track_id=tid, frame=j), ensure_ascii=False) + "\n"
                )
                n_frames += 1
            if (i + 1) % 500 == 0:
                print(f"  {i+1:,}/{args.count:,} tracks ({n_frames:,} frames)", file=sys.stderr)

    _write_manifest(out, args, {"kind": "tracks", "n_tracks": args.count, "n_frames": n_frames})
    print(f"wrote {args.count:,} tracks / {n_frames:,} frames to {out} in {time.time()-t0:.1f}s")
    return 0


def cmd_preview(args: argparse.Namespace) -> int:
    """Contact sheet, ordered by difficulty. The fastest way to see whether the
    generator is producing something a human could read."""
    rng = random.Random(args.seed)
    sampler = PlateSampler(seed=args.seed, embossed_fraction=args.embossed_fraction)
    cfg = DegradationConfig()

    items = [
        degrade(render(sampler.sample(), height=args.render_height, rng=rng), rng=rng, cfg=cfg)
        for _ in range(args.rows * args.cols)
    ]
    items.sort(key=lambda d: d.difficulty)

    cw = max(i.image.width for i in items) + 12
    ch = max(i.image.height for i in items) + 12
    sheet = Image.new("RGB", (args.cols * cw, args.rows * ch), (24, 24, 28))
    for i, d in enumerate(items):
        x = (i % args.cols) * cw + (cw - d.image.width) // 2
        y = (i // args.cols) * ch + (ch - d.image.height) // 2
        sheet.paste(d.image, (x, y))
    sheet.save(args.out)
    qs = sorted(d.quality for d in items)
    print(
        f"{args.out}  n={len(items)}  quality min={qs[0]:.2f} "
        f"med={qs[len(qs)//2]:.2f} max={qs[-1]:.2f}"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    # Common options live on a parent parser so they are accepted both before
    # and after the subcommand. Attaching them only to the top-level parser
    # makes "synthplate generate --seed 5" an error, which is exactly how
    # anyone would naturally type it.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--seed", type=int, default=0,
                        help="RNG seed; makes a corpus reproducible")
    common.add_argument("--render-height", type=int, default=110,
                        help="clean-render height before degradation")
    common.add_argument("--embossed-fraction", type=float, default=0.5,
                        help="fraction of samples using the embossed system")
    common.add_argument("--strategy", choices=("uniform", "realistic"), default="uniform",
                        help="uniform maximises structural coverage; "
                             "realistic matches road frequency")

    p = argparse.ArgumentParser(
        prog="synthplate", description=__doc__.split("\n")[0], parents=[common]
    )
    sub = p.add_subparsers(dest="command", required=True)

    g = sub.add_parser("generate", help="single-image corpus", parents=[common])
    g.add_argument("--out", required=True)
    g.add_argument("--count", type=int, default=1000)
    g.add_argument("--save-quality", type=int, default=92)
    g.add_argument("--grid", action="store_true", default=True,
                   help="seed with an exhaustive structural sweep first")
    g.add_argument("--no-grid", dest="grid", action="store_false")
    g.set_defaults(func=cmd_generate)

    t = sub.add_parser("tracks", help="multi-frame track corpus", parents=[common])
    t.add_argument("--out", required=True)
    t.add_argument("--count", type=int, default=200, help="number of tracks")
    t.add_argument("--save-quality", type=int, default=92)
    t.set_defaults(func=cmd_tracks)

    v = sub.add_parser("preview", help="contact sheet for eyeballing", parents=[common])
    v.add_argument("--out", default="preview.png")
    v.add_argument("--rows", type=int, default=6)
    v.add_argument("--cols", type=int, default=6)
    v.set_defaults(func=cmd_preview)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
