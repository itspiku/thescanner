"""Training the recogniser.

    python -m scanner_models.train --data data/synth --epochs 12 --out models/platenet

Loss is the sum of three terms, weighted:

* **CTC** over the 71-token vocabulary -- the main objective.
* **Colour** cross-entropy -- 7 classes. Weighted meaningfully rather than as an
  afterthought, because its output feeds the decoding prior in
  ``nepal_plate.decode``, and a colour error there costs more than a glyph
  error: it steers the whole class-letter decision.
* **Quality** MSE against the generator's recorded degradation severity. Free,
  perfectly calibrated supervision, and its output drives frame weighting in
  ``nepal_plate.fuse``.

Notes on the training setup, all of which are consequences of a 6 GB card:

* Mixed precision throughout, except the CTC loss itself -- cuDNN's CTC is
  numerically fragile in fp16 and will silently produce NaNs on long targets.
  It runs in fp32 inside an autocast-disabled block.
* Gradient accumulation lets the effective batch stay large while the resident
  batch stays small.
* Validation reports **exact-match plate accuracy**, not character accuracy.
  Character accuracy flatters an ANPR badly: a plate read with one wrong digit
  is 87% correct by character and 100% useless operationally.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .data import PlateDataset, collate, read_index, split_samples
from .model import ModelConfig, PlateNet
from .vocab import BLANK, collapse


def _device(pref: str) -> torch.device:
    if pref != "auto":
        return torch.device(pref)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ctc_loss_fp32(
    log_probs: torch.Tensor,
    targets: torch.Tensor,
    target_lens: torch.Tensor,
) -> torch.Tensor:
    """CTC loss, forced to fp32.

    cuDNN's CTC implementation is numerically unstable in half precision and
    produces NaN losses on longer targets without any error. Running just this
    op in fp32 costs almost nothing and removes an entire class of
    silent-failure debugging.
    """
    b, t, _ = log_probs.shape
    input_lens = torch.full((b,), t, dtype=torch.long, device=log_probs.device)
    with torch.autocast(device_type=log_probs.device.type, enabled=False):
        return nn.functional.ctc_loss(
            log_probs.float().permute(1, 0, 2),  # CTC wants (T, B, C)
            targets,
            input_lens,
            target_lens,
            blank=BLANK,
            reduction="mean",
            zero_infinity=True,
        )


@torch.no_grad()
def evaluate_greedy(
    model: PlateNet, loader: DataLoader, device: torch.device, dataset: PlateDataset
) -> dict[str, float]:
    """Fast in-loop validation using greedy CTC.

    Deliberately *not* the grammar-constrained decoder: this measures the raw
    model, so that improvements from the decoder are never confused with
    improvements in the network. ``scanner_models.evaluate`` reports both.
    """
    model.eval()
    exact = total = 0
    colour_ok = 0
    q_err = 0.0
    token_err = token_total = 0

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        out = model(images)
        pred = out["ctc_log_probs"].argmax(dim=-1).cpu()

        colour_ok += (out["colour_logits"].argmax(dim=-1).cpu() == batch["colour"]).sum().item()
        q_err += (out["quality"].cpu() - batch["quality"]).abs().sum().item()

        for row, idx in zip(pred, batch["index"].tolist()):
            truth = list(dataset.samples[idx].tokens)
            got = collapse(row.tolist())
            total += 1
            exact += int(got == truth)
            token_err += _levenshtein(got, truth)
            token_total += len(truth)

    return {
        "exact_match": exact / max(total, 1),
        "token_error_rate": token_err / max(token_total, 1),
        "colour_accuracy": colour_ok / max(total, 1),
        "quality_mae": q_err / max(total, 1),
        "n": float(total),
    }


def _levenshtein(a: list[str], b: list[str]) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def train(args: argparse.Namespace) -> int:
    device = _device(args.device)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    samples = read_index(args.data, limit=args.limit)
    if not samples:
        print(f"no samples found in {args.data}")
        return 1
    tr, va = split_samples(samples, val_fraction=args.val_fraction, seed=args.seed)
    print(f"{len(tr):,} train / {len(va):,} val samples")

    train_ds = PlateDataset(args.data, tr, train=True, seed=args.seed)
    # Validation uses the aspect-ratio layout heuristic, not the ground-truth
    # layout flag, because that is the only signal available at inference.
    val_ds = PlateDataset(args.data, va, train=False, use_layout_label=False)

    train_dl = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate,
        num_workers=args.workers, pin_memory=device.type == "cuda", drop_last=True,
        persistent_workers=args.workers > 0,
    )
    val_dl = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate,
        num_workers=args.workers, pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
    )

    cfg = ModelConfig()
    model = PlateNet(cfg).to(device)
    print(f"PlateNet: {model.n_parameters():,} parameters on {device}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = max(1, len(train_dl) // args.accum)
    total_steps = steps_per_epoch * args.epochs
    warmup = min(args.warmup, max(1, total_steps // 10))

    def lr_at(step: int) -> float:
        if step < warmup:
            return step / max(warmup, 1)
        p = (step - warmup) / max(total_steps - warmup, 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(p, 1.0)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    use_amp = device.type == "cuda" and not args.no_amp
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    colour_loss = nn.CrossEntropyLoss()
    quality_loss = nn.MSELoss()

    history: list[dict] = []
    best = -1.0
    step = 0
    t_start = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = {"ctc": 0.0, "colour": 0.0, "quality": 0.0}
        seen = 0
        opt.zero_grad(set_to_none=True)

        for i, batch in enumerate(train_dl):
            images = batch["image"].to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                out = model(images)
                l_col = colour_loss(out["colour_logits"], batch["colour"].to(device))
                l_qual = quality_loss(out["quality"], batch["quality"].to(device))
            l_ctc = ctc_loss_fp32(
                out["ctc_log_probs"], batch["target"].to(device), batch["target_len"].to(device)
            )
            loss = l_ctc + args.colour_weight * l_col + args.quality_weight * l_qual
            scaler.scale(loss / args.accum).backward()

            running["ctc"] += float(l_ctc.detach())
            running["colour"] += float(l_col.detach())
            running["quality"] += float(l_qual.detach())
            seen += 1

            if (i + 1) % args.accum == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                sched.step()
                step += 1

            if args.log_every and seen % args.log_every == 0:
                print(
                    f"  e{epoch} {seen}/{len(train_dl)}  "
                    f"ctc={running['ctc']/seen:.3f} col={running['colour']/seen:.3f} "
                    f"qual={running['quality']/seen:.4f} lr={sched.get_last_lr()[0]:.2e}"
                )

        metrics = evaluate_greedy(model, val_dl, device, val_ds)
        rec = {
            "epoch": epoch,
            "ctc_loss": running["ctc"] / max(seen, 1),
            "colour_loss": running["colour"] / max(seen, 1),
            "quality_loss": running["quality"] / max(seen, 1),
            "elapsed_s": round(time.time() - t_start, 1),
            **metrics,
        }
        history.append(rec)
        print(
            f"epoch {epoch:2d}  ctc={rec['ctc_loss']:.3f}  "
            f"exact={metrics['exact_match']:.3f}  TER={metrics['token_error_rate']:.3f}  "
            f"colour={metrics['colour_accuracy']:.3f}  qMAE={metrics['quality_mae']:.3f}  "
            f"({rec['elapsed_s']:.0f}s)"
        )

        ckpt = {
            "model": model.state_dict(),
            "config": asdict(cfg),
            "epoch": epoch,
            "metrics": metrics,
            "args": {k: v for k, v in vars(args).items() if k != "func"},
        }
        torch.save(ckpt, out_dir / "last.pt")
        if metrics["exact_match"] > best:
            best = metrics["exact_match"]
            torch.save(ckpt, out_dir / "best.pt")
        (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    print(f"\nbest exact-match {best:.4f} -> {out_dir/'best.pt'}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="scanner_models.train", description=__doc__.split("\n")[0])
    p.add_argument("--data", required=True, help="corpus directory containing index.jsonl")
    p.add_argument("--out", default="models/platenet")
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--accum", type=int, default=1, help="gradient accumulation steps")
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--warmup", type=int, default=500)
    p.add_argument("--clip", type=float, default=5.0)
    p.add_argument("--colour-weight", type=float, default=0.3)
    p.add_argument("--quality-weight", type=float, default=1.0)
    p.add_argument("--val-fraction", type=float, default=0.04)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--log-every", type=int, default=200)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    torch.manual_seed(args.seed)
    return train(args)


if __name__ == "__main__":
    raise SystemExit(main())
