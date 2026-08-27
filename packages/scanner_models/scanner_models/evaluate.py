"""Evaluation harness: does the grammar actually earn its place?

    python -m scanner_models.evaluate --data data/synth --checkpoint models/platenet/best.pt

The whole recognition design rests on two claims that are easy to assert and
easy to get wrong, so this harness measures both directly, as an ablation on
one fixed model:

1. **greedy** -- argmax CTC, collapse, done. What a conventional ANPR produces.
2. **grammar** -- constrained beam search, no colour prior.
3. **grammar + colour** -- constrained beam search with the plate-colour prior,
   using the colour the model itself predicted (not the ground-truth colour --
   that would be cheating, and would hide colour-head errors).
4. **track fusion** -- where the corpus contains tracks, per-field consensus
   across frames.

Results are reported **per stratum**, not just as a headline number. An ANPR
that is 96% accurate overall but 70% on motorcycles disproportionately
penalises the vehicles poorer people drive, and a single average hides that
completely. Strata: plate system, pixel width, night/day, ownership class, and
degradation band.

The harness also reports **calibration** -- accuracy within each confidence
band. The success criterion that matters most operationally is not accuracy but
the false-positive rate at HIGH confidence: a wrong plate asserted confidently
is the failure mode that puts the wrong person in front of a magistrate.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Sequence

import torch
from torch.utils.data import DataLoader

from nepal_plate import ColourEvidence, Confidence, PlateSystem, decode
from nepal_plate.fuse import FrameObservation, fuse_track
from nepal_plate.grammar import GRAMMARS

from .data import PlateDataset, Sample, collate, group_by_track, read_index
from .model import ModelConfig, PlateNet
from .vocab import COLOUR_CLASSES, collapse

METHODS = ("greedy", "grammar", "grammar+colour")


def load_model(path: Path | str, device: torch.device) -> PlateNet:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ModelConfig(**ckpt["config"]) if "config" in ckpt else ModelConfig()
    model = PlateNet(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def _width_band(px: int) -> str:
    for lo, hi in ((0, 40), (40, 60), (60, 90), (90, 130)):
        if lo <= px < hi:
            return f"{lo}-{hi}px"
    return "130px+"


def _quality_band(q: float) -> str:
    if q >= 0.7:
        return "clean"
    if q >= 0.4:
        return "moderate"
    return "degraded"


class Tally:
    """Correct/total counters, sliced by stratum."""

    def __init__(self) -> None:
        self.total: dict[str, int] = defaultdict(int)
        self.correct: dict[str, dict[str, int]] = {m: defaultdict(int) for m in METHODS}

    def add(self, strata: Sequence[str], results: dict[str, bool]) -> None:
        for s in strata:
            self.total[s] += 1
            for m, ok in results.items():
                if ok:
                    self.correct[m][s] += 1

    def rows(self) -> list[tuple[str, int, dict[str, float]]]:
        out = []
        for s in sorted(self.total, key=lambda k: (-self.total[k], k)):
            n = self.total[s]
            out.append((s, n, {m: self.correct[m][s] / n for m in METHODS}))
        return out


@torch.no_grad()
def run(args: argparse.Namespace) -> int:
    device = torch.device(args.device if args.device != "auto"
                          else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = load_model(args.checkpoint, device)

    samples = read_index(args.data, limit=args.limit)
    if not samples:
        print(f"no samples in {args.data}")
        return 1
    # Inference-time layout decision, not the ground-truth flag: measuring with
    # a signal you will not have in production is self-deception.
    ds = PlateDataset(args.data, samples, train=False, use_layout_label=False)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate,
                    num_workers=args.workers)

    tally = Tally()
    # confidence band -> [correct, total], for the calibration table
    calib: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    colour_correct = 0
    # Cached per-sample model output, reused by the track-fusion pass so the
    # network is not run twice over the same corpus.
    cache: dict[int, tuple[list[list[float]], dict, float]] = {}

    t0 = time.time()
    n = 0
    for batch in dl:
        out = model(batch["image"].to(device))
        lp = out["ctc_log_probs"].cpu()
        colour_probs = torch.softmax(out["colour_logits"], dim=-1).cpu()
        quality = out["quality"].cpu()

        for k, idx in enumerate(batch["index"].tolist()):
            s = ds.samples[idx]
            frame_lp = lp[k].tolist()
            posterior = {COLOUR_CLASSES[j]: float(colour_probs[k, j]) for j in range(len(COLOUR_CLASSES))}
            evidence = ColourEvidence(posterior=posterior)
            q = float(quality[k])
            if args.fuse:
                cache[idx] = (frame_lp, posterior, q)

            truth = list(s.tokens)
            colour_correct += int(evidence.best()[0] == s.colour)

            results = {"greedy": collapse(lp[k].argmax(dim=-1).tolist()) == truth}

            plain = decode(frame_lp, beam_width=args.beam, top_k=1)
            results["grammar"] = bool(plain) and list(_tokens_of(plain[0])) == truth

            withc = decode(frame_lp, colour=evidence, beam_width=args.beam, top_k=1)
            ok_c = bool(withc) and list(_tokens_of(withc[0])) == truth
            results["grammar+colour"] = ok_c

            if withc:
                band = withc[0].plate.confidence.value
                calib[band][1] += 1
                calib[band][0] += int(ok_c)

            tally.add(
                [
                    "ALL",
                    f"system:{s.system}",
                    f"width:{_width_band(s.width_px)}",
                    f"quality:{_quality_band(s.quality)}",
                    f"layout:{'two-row' if s.two_row else 'single-row'}",
                ],
                results,
            )
            n += 1

    elapsed = time.time() - t0
    report = {
        "n_samples": n,
        "checkpoint": str(args.checkpoint),
        "corpus": str(args.data),
        "beam_width": args.beam,
        "seconds": round(elapsed, 1),
        "colour_accuracy": colour_correct / max(n, 1),
        "strata": {
            name: {"n": cnt, **{m: round(v, 4) for m, v in accs.items()}}
            for name, cnt, accs in tally.rows()
        },
        "calibration": {
            band: {"n": tot, "accuracy": round(ok / tot, 4) if tot else None}
            for band, (ok, tot) in sorted(calib.items())
        },
    }

    if args.fuse:
        report["track_fusion"] = _evaluate_tracks(ds.samples, cache, args)

    _print_report(report)
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nwrote {args.out}")
    return 0


def _tokens_of(candidate) -> Iterable[str]:
    """Recover the token sequence from a decoded candidate.

    Segmenting the plate back through its grammar is the reliable way to do
    this -- the canonical string is romanised and lossy, so comparing against it
    would silently compare the wrong things.
    """
    plate = candidate.plate
    grammar = GRAMMARS.get(plate.system)
    if grammar is None:
        return []
    toks: list[str] = []
    for slot in grammar.slots:
        for f in plate.fields:
            if f.name == slot.name:
                toks.extend(_split(slot, f.value))
    return toks


def _split(slot, value: str) -> list[str]:
    out: list[str] = []
    i = 0
    by_len = sorted(slot.tokens, key=len, reverse=True)
    while i < len(value):
        for t in by_len:
            if value.startswith(t, i):
                out.append(t)
                i += len(t)
                break
        else:
            return []
    return out


def _evaluate_tracks(samples: Sequence[Sample], cache: dict, args) -> dict:
    """Track-level fusion against the best and the median single frame.

    The comparison that matters is fusion vs. *the best frame in the track*, not
    fusion vs. a random frame. Beating a random frame is trivial; beating the
    best available look is the claim being made.
    """
    tracks = group_by_track(samples)
    if not tracks:
        return {"n_tracks": 0, "note": "corpus contains no tracks"}

    index_of = {id(s): i for i, s in enumerate(samples)}
    fused_ok = best_ok = first_ok = 0
    counted = 0

    for _, frames in tracks.items():
        obs: list[FrameObservation] = []
        per_frame_ok: list[bool] = []
        truth = list(frames[0].tokens)
        for s in frames:
            i = index_of.get(id(s))
            if i is None or i not in cache:
                continue
            lp, posterior, q = cache[i]
            obs.append(FrameObservation(log_probs=lp, quality=q,
                                        colour=ColourEvidence(posterior=posterior)))
            got = decode(lp, colour=ColourEvidence(posterior=posterior),
                         beam_width=args.beam, top_k=1)
            per_frame_ok.append(bool(got) and list(_tokens_of(got[0])) == truth)
        if not obs:
            continue
        counted += 1
        fused = fuse_track(obs, beam_width=args.beam)
        if fused is not None:
            fused_ok += int(list(_tokens_of(_Wrap(fused.plate))) == truth)
        best_ok += int(any(per_frame_ok))
        first_ok += int(per_frame_ok[0])

    return {
        "n_tracks": counted,
        "fused": round(fused_ok / max(counted, 1), 4),
        "oracle_best_frame": round(best_ok / max(counted, 1), 4),
        "first_frame": round(first_ok / max(counted, 1), 4),
    }


class _Wrap:
    """Adapter so a bare ParsedPlate can go through ``_tokens_of``."""

    def __init__(self, plate):
        self.plate = plate


def _print_report(report: dict) -> None:
    print(f"\n{'='*74}")
    print(f"  {report['n_samples']:,} samples   colour head {report['colour_accuracy']:.1%}"
          f"   {report['seconds']:.0f}s")
    print(f"{'='*74}")
    print(f"{'stratum':24s} {'n':>7s} {'greedy':>9s} {'grammar':>9s} {'+colour':>9s} {'delta':>8s}")
    print("-" * 74)
    for name, row in report["strata"].items():
        d = row["grammar+colour"] - row["greedy"]
        print(f"{name:24s} {row['n']:7,d} {row['greedy']:9.3f} {row['grammar']:9.3f} "
              f"{row['grammar+colour']:9.3f} {d:+8.3f}")

    if report.get("calibration"):
        print(f"\n{'confidence band':24s} {'n':>7s} {'accuracy':>9s}")
        print("-" * 44)
        for band, row in report["calibration"].items():
            acc = "n/a" if row["accuracy"] is None else f"{row['accuracy']:.3f}"
            print(f"{band:24s} {row['n']:7,d} {acc:>9s}")

    tf = report.get("track_fusion")
    if tf and tf.get("n_tracks"):
        print(f"\ntrack fusion over {tf['n_tracks']:,} tracks")
        print("-" * 44)
        print(f"  {'first frame only':22s} {tf['first_frame']:.3f}")
        print(f"  {'oracle best frame':22s} {tf['oracle_best_frame']:.3f}")
        print(f"  {'fused (per-field)':22s} {tf['fused']:.3f}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="scanner_models.evaluate",
                                description=__doc__.split("\n")[0])
    p.add_argument("--data", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", default=None, help="write the full report as JSON")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--beam", type=int, default=12)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--device", default="auto")
    p.add_argument("--fuse", action="store_true", help="also evaluate track-level fusion")
    return p


def main(argv: list[str] | None = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
