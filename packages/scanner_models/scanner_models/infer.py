"""Runtime inference: a plate crop in, a validated :class:`ParsedPlate` out.

This is the seam between the neural network and the domain logic, and it is
deliberately the *only* place the two meet. The edge agent, the evaluation
harness and any offline tool all go through :class:`PlateReader`, so
preprocessing, colour handling and decoding can never drift apart between
training and deployment.

Two backends behind one interface:

* :class:`TorchBackend` -- for development, evaluation and training-time checks.
* :class:`OnnxBackend` -- for deployment. ONNX Runtime with the TensorRT or
  OpenVINO execution provider is what actually runs on an edge node, and it
  removes PyTorch (about 2.5 GB) from the deployment image entirely.

The class also exposes :meth:`PlateReader.read_track`, which is the API the
edge agent should prefer: a read is an estimate over a vehicle passage, not a
guess from one frame.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np
from PIL import Image

from nepal_plate import ColourEvidence, ParsedPlate, PlateColour, decode
from nepal_plate.fuse import FrameObservation, FusedRead, fuse_track

from .preprocess import prepare, quad_aspect, TWO_ROW_ASPECT
from .vocab import COLOUR_CLASSES


@dataclass(frozen=True)
class RawOutput:
    """One forward pass, before any domain logic is applied."""

    #: ``T x V`` natural-log probabilities.
    log_probs: list[list[float]]
    colour: ColourEvidence
    quality: float


class Backend(Protocol):
    def infer(self, batch: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(B,3,H,W)`` -> ``(ctc_log_probs, colour_logits, quality)``."""


class TorchBackend:
    """PyTorch backend, for development and evaluation."""

    def __init__(self, checkpoint: str | Path, device: str = "auto") -> None:
        import torch

        from .model import ModelConfig, PlateNet

        self.torch = torch
        self.device = torch.device(
            device if device != "auto"
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        ckpt = torch.load(checkpoint, map_location=self.device, weights_only=False)
        cfg = ModelConfig(**ckpt["config"]) if "config" in ckpt else ModelConfig()
        self.model = PlateNet(cfg).to(self.device).eval()
        self.model.load_state_dict(ckpt["model"])

    def infer(self, batch: np.ndarray):
        t = self.torch
        with t.no_grad():
            out = self.model(t.from_numpy(batch).to(self.device))
            return (
                out["ctc_log_probs"].cpu().numpy(),
                out["colour_logits"].cpu().numpy(),
                out["quality"].cpu().numpy(),
            )


class OnnxBackend:
    """ONNX Runtime backend, for deployment.

    Providers are tried in order and ONNX Runtime silently falls back, so the
    same artefact runs on a Jetson (TensorRT), an x86 mini-PC (OpenVINO) or a
    developer laptop (CPU) with no code change. Which provider was actually
    selected is exposed as :attr:`provider`, because a node that quietly fell
    back to CPU will miss its throughput target and should say so in telemetry
    rather than just running slowly.
    """

    def __init__(self, model_path: str | Path, providers: Sequence[str] | None = None) -> None:
        import onnxruntime as ort

        available = set(ort.get_available_providers())
        wanted = providers or (
            "TensorrtExecutionProvider",
            "CUDAExecutionProvider",
            "OpenVINOExecutionProvider",
            "CPUExecutionProvider",
        )
        chosen = [p for p in wanted if p in available] or ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(str(model_path), providers=chosen)
        self.provider = self.session.get_providers()[0]
        self.input_name = self.session.get_inputs()[0].name

    def infer(self, batch: np.ndarray):
        out = self.session.run(None, {self.input_name: batch.astype(np.float32)})
        return out[0], out[1], out[2]


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


class PlateReader:
    """Crop -> plate. The single supported inference path."""

    def __init__(
        self,
        backend: Backend,
        *,
        beam_width: int = 12,
        colour_weight: float = 1.0,
        colour_min_confidence: float = 0.35,
    ) -> None:
        self.backend = backend
        self.beam_width = beam_width
        self.colour_weight = colour_weight
        #: Below this, the colour head's output is marked unreliable and the
        #: decoding prior is suppressed rather than applied weakly. A prior
        #: built from an unsure classifier is worse than no prior: it adds
        #: variance without adding information.
        self.colour_min_confidence = colour_min_confidence

    # -- raw ------------------------------------------------------------

    def infer_batch(
        self, crops: Sequence[Image.Image], quads: Sequence[Sequence[Sequence[float]]] | None = None
    ) -> list[RawOutput]:
        if not crops:
            return []
        arrays = []
        for i, crop in enumerate(crops):
            two_row = None
            if quads is not None and i < len(quads):
                a = quad_aspect(quads[i])
                # The detector's quad is a better shape estimate than the
                # bounding box after a perspective warp, so prefer it when the
                # caller has one.
                if a > 0:
                    two_row = a < TWO_ROW_ASPECT
            arrays.append(prepare(crop, two_row=two_row))

        lp, colour_logits, quality = self.backend.infer(np.stack(arrays))
        probs = _softmax(np.asarray(colour_logits, dtype=np.float64))

        out: list[RawOutput] = []
        for i in range(len(crops)):
            posterior = {c: float(probs[i, j]) for j, c in enumerate(COLOUR_CLASSES)}
            top = max(posterior.values()) if posterior else 0.0
            out.append(
                RawOutput(
                    log_probs=np.asarray(lp[i], dtype=np.float64).tolist(),
                    colour=ColourEvidence(
                        posterior=posterior,
                        reliable=top >= self.colour_min_confidence,
                    ),
                    quality=float(quality[i]),
                )
            )
        return out

    # -- single frame ---------------------------------------------------

    def read(
        self, crop: Image.Image, quad: Sequence[Sequence[float]] | None = None
    ) -> ParsedPlate | None:
        """Read one crop. Returns ``None`` when nothing legal decodes.

        Returning ``None`` rather than a best-effort string is deliberate: a
        crop of a damaged, obscured or non-Nepali plate has no valid reading,
        and inventing one is worse than reporting the failure.
        """
        raw = self.infer_batch([crop], [quad] if quad is not None else None)
        if not raw:
            return None
        cands = decode(
            raw[0].log_probs,
            colour=raw[0].colour,
            beam_width=self.beam_width,
            colour_weight=self.colour_weight,
            top_k=1,
        )
        return cands[0].plate if cands else None

    # -- track ----------------------------------------------------------

    def read_track(
        self,
        crops: Sequence[Image.Image],
        quads: Sequence[Sequence[Sequence[float]]] | None = None,
    ) -> FusedRead | None:
        """Read a vehicle passage. Prefer this over :meth:`read`.

        The model's own quality head weights each frame, so a badly blurred look
        contributes proportionally less without being discarded -- a poor frame
        still carries information, just less of it.
        """
        raw = self.infer_batch(crops, quads)
        if not raw:
            return None
        return fuse_track(
            [
                FrameObservation(log_probs=r.log_probs, quality=r.quality, colour=r.colour, frame_index=i)
                for i, r in enumerate(raw)
            ],
            beam_width=self.beam_width,
            colour_weight=self.colour_weight,
        )


def from_checkpoint(path: str | Path, **kwargs) -> PlateReader:
    return PlateReader(TorchBackend(path), **kwargs)


def from_onnx(path: str | Path, **kwargs) -> PlateReader:
    return PlateReader(OnnxBackend(path), **kwargs)


__all__ = [
    "RawOutput",
    "Backend",
    "TorchBackend",
    "OnnxBackend",
    "PlateReader",
    "from_checkpoint",
    "from_onnx",
]
