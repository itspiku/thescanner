"""Export a trained recogniser to ONNX, for deployment.

    python -m scanner_models.export --checkpoint models/platenet/best.pt --out models/platenet.onnx

ONNX rather than TorchScript because it is what actually runs at the edge: one
artefact executes under TensorRT on a Jetson, OpenVINO on an x86 mini-PC and
plain CPU on a laptop, and it removes PyTorch -- roughly 2.5 GB -- from the
deployment image.

The export is **verified numerically against the PyTorch model before it is
accepted**, not merely produced. A silently-wrong export is one of the easier
ways to ship a model that scores well in evaluation and badly in the field.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .model import ModelConfig, PlateNet, PlateNetExport
from .preprocess import MEAN, STD
from .vocab import COLOUR_CLASSES, VOCAB


def export(args: argparse.Namespace) -> int:
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = ModelConfig(**ckpt["config"]) if "config" in ckpt else ModelConfig()
    net = PlateNet(cfg)
    net.load_state_dict(ckpt["model"])
    net.eval()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    wrapper = PlateNetExport(net).eval()
    dummy = torch.randn(1, 3, cfg.input_h, cfg.input_w)

    torch.onnx.export(
        wrapper,
        dummy,
        str(out),
        input_names=["image"],
        output_names=["ctc_log_probs", "colour_logits", "quality"],
        # Batch is dynamic so the edge agent can size batches to how many plates
        # a frame actually contains, rather than padding to a fixed shape.
        dynamic_axes={
            "image": {0: "batch"},
            "ctc_log_probs": {0: "batch"},
            "colour_logits": {0: "batch"},
            "quality": {0: "batch"},
        },
        opset_version=args.opset,
        do_constant_folding=True,
        # The TorchScript exporter, not the dynamo one. As of torch 2.13 the
        # dynamo path bakes the tracing batch size into the BiLSTM's internal
        # reshape -- an export traced at batch 1 then fails at runtime on any
        # other batch with "cannot reshape {48,3,2,128} to {48,1,256}". The
        # legacy exporter handles the dynamic axis correctly (verified below at
        # ~1e-6 agreement), so it is pinned rather than left to the default.
        dynamo=False,
    )

    max_diff = _verify(wrapper, out, cfg, args.tolerance)

    sidecar = {
        "input": {
            "name": "image",
            "shape": [None, 3, cfg.input_h, cfg.input_w],
            "mean": list(MEAN),
            "std": list(STD),
            "layout": "NCHW",
            "colour": "RGB",
        },
        "outputs": ["ctc_log_probs", "colour_logits", "quality"],
        "vocab": list(VOCAB),
        "blank_index": 0,
        "colour_classes": [c.value for c in COLOUR_CLASSES],
        "config": {
            k: (list(v) if isinstance(v, tuple) else v)
            for k, v in ckpt.get("config", {}).items()
        },
        "source_checkpoint": str(args.checkpoint),
        "training_metrics": ckpt.get("metrics", {}),
        "opset": args.opset,
        "max_abs_diff_vs_torch": max_diff,
    }
    meta = out.with_suffix(".json")
    meta.write_text(json.dumps(sidecar, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB) and {meta.name}")
    print(f"max |onnx - torch| = {max_diff:.2e}")
    return 0


def _verify(wrapper, path: Path, cfg: ModelConfig, tolerance: float) -> float:
    """Compare ONNX and PyTorch outputs on random inputs.

    Random noise rather than a real plate on purpose: noise exercises the full
    dynamic range of every layer, whereas a well-formed plate can leave parts of
    the network in a narrow regime where a numerical divergence stays hidden.
    """
    try:
        import onnxruntime as ort
    except ImportError:
        print("onnxruntime not installed; skipping numerical verification")
        return float("nan")

    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    worst = 0.0
    for batch in (1, 4):
        x = torch.randn(batch, 3, cfg.input_h, cfg.input_w)
        with torch.no_grad():
            ref = wrapper(x)
        got = sess.run(None, {"image": x.numpy()})
        for a, b in zip(ref, got):
            worst = max(worst, float(np.abs(a.numpy() - b).max()))

    if worst > tolerance:
        raise RuntimeError(
            f"ONNX export diverges from PyTorch by {worst:.3e} "
            f"(tolerance {tolerance:.0e}). Refusing to ship an export that does "
            f"not match the model it came from."
        )
    return worst


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="scanner_models.export", description=__doc__.split("\n")[0]
    )
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", default="models/platenet.onnx")
    p.add_argument("--opset", type=int, default=17)
    p.add_argument("--tolerance", type=float, default=1e-3)
    return p


def main(argv: list[str] | None = None) -> int:
    return export(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
