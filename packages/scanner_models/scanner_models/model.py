"""The recogniser: one trunk, three heads.

Architecture
------------
A convolutional stem collapses the image height to 1 while preserving
horizontal resolution, giving a 48-step feature sequence; a two-layer BiLSTM
adds context; three heads read off it.

::

    RGB 48x192 --> conv stem --> 48 x 256 --> BiLSTM --> 48 x 256
                                                          |
                            +-----------------------------+------------------+
                            |                     |                          |
                      CTC head (71)        colour head (7)          quality head (1)
                     per-timestep          pooled, whole-plate      pooled, scalar

Why three heads on one trunk rather than three models
-----------------------------------------------------
All three tasks need the same evidence -- glyph shape, background colour, and
how badly the crop is degraded are all read off the same pixels -- so sharing
the trunk costs almost nothing and saves two-thirds of the edge memory budget
and two of the three forward passes. An edge node running four camera streams
on 8 GB has no room for the alternative.

They are also mutually reinforcing. Predicting colour forces the trunk to keep
chromatic information that a pure-OCR objective would happily discard in the
first layer -- and that colour output is exactly what feeds the decoding prior
in ``nepal_plate.decode``. A greyscale-first recogniser would throw away the
signal the decoder most depends on.

The quality head is trained against the ground-truth degradation severity that
``synthplate`` records for every sample. That is a free, perfectly calibrated
target that would otherwise require human annotation, and its output drives
frame weighting in ``nepal_plate.fuse``.

Size
----
1.93 M parameters. Measured at batch 64 on an RTX 4050: 11.2 ms per forward
pass -- about 5,700 crops/second -- in 518 MB of VRAM. That leaves ample room
to train on a 6 GB card, which is a useful forcing function: a model that
trains on a laptop GPU will run on a Jetson.

The 48 CTC timesteps are deliberate headroom. The longest legal plate is nine
tokens, and CTC needs a blank between repeated glyphs, so the worst case needs
17 steps.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from .preprocess import INPUT_H, INPUT_W
from .vocab import N_COLOURS, VOCAB_SIZE


@dataclass(frozen=True)
class ModelConfig:
    """Shapes and widths. Recorded in every checkpoint so a saved model can
    always be rebuilt without guessing."""

    vocab_size: int = VOCAB_SIZE
    n_colours: int = N_COLOURS
    input_h: int = INPUT_H
    input_w: int = INPUT_W
    channels: tuple[int, ...] = (32, 64, 128, 256, 256)
    rnn_hidden: int = 128
    rnn_layers: int = 2
    dropout: float = 0.1

    @property
    def seq_len(self) -> int:
        """CTC timesteps: width is halved twice in the stem."""
        return self.input_w // 4

    @property
    def feature_dim(self) -> int:
        return self.rnn_hidden * 2


def _conv_bn(in_ch: int, out_ch: int, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class PlateNet(nn.Module):
    """Dual-script plate recogniser with colour and quality heads."""

    def __init__(self, cfg: ModelConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or ModelConfig()
        c = self.cfg.channels

        # Height collapses 48 -> 1; width halves only twice, to 48. Keeping
        # horizontal resolution is what gives CTC enough timesteps to separate
        # adjacent glyphs and to place blanks between repeated digits.
        self.stem = nn.Sequential(
            _conv_bn(3, c[0]),
            _conv_bn(c[0], c[0]),
            nn.MaxPool2d(2, 2),                    # 24 x 96
            _conv_bn(c[0], c[1]),
            _conv_bn(c[1], c[1]),
            nn.MaxPool2d(2, 2),                    # 12 x 48
            _conv_bn(c[1], c[2]),
            nn.MaxPool2d((2, 1), (2, 1)),          # 6 x 48
            _conv_bn(c[2], c[3]),
            nn.MaxPool2d((2, 1), (2, 1)),          # 3 x 48
            _conv_bn(c[3], c[4]),
            nn.MaxPool2d((3, 1), (3, 1)),          # 1 x 48
        )

        self.rnn = nn.LSTM(
            input_size=c[4],
            hidden_size=self.cfg.rnn_hidden,
            num_layers=self.cfg.rnn_layers,
            batch_first=True,
            bidirectional=True,
            dropout=self.cfg.dropout if self.cfg.rnn_layers > 1 else 0.0,
        )

        d = self.cfg.feature_dim
        self.dropout = nn.Dropout(self.cfg.dropout)
        self.ctc_head = nn.Linear(d, self.cfg.vocab_size)

        # Colour and quality are whole-plate properties, so they read a pooled
        # summary rather than per-timestep features. Mean and max concatenated:
        # mean captures the background that dominates the plate area, max
        # captures the brightest glyph strokes.
        self.colour_head = nn.Sequential(
            nn.Linear(d * 2, 128), nn.ReLU(inplace=True), nn.Linear(128, self.cfg.n_colours)
        )
        self.quality_head = nn.Sequential(
            nn.Linear(d * 2, 64), nn.ReLU(inplace=True), nn.Linear(64, 1)
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """``x``: ``(B, 3, 48, 192)``.

        Returns logits for all three heads. Log-softmax is applied to the CTC
        output here rather than in the loss, so the exported ONNX graph emits
        exactly what ``nepal_plate.decode`` expects and no post-processing has
        to be duplicated in the edge agent.
        """
        f = self.stem(x)                       # (B, C, 1, W/4)
        f = f.squeeze(2).permute(0, 2, 1)      # (B, T, C)
        seq, _ = self.rnn(f)                   # (B, T, 2H)
        seq = self.dropout(seq)

        pooled = torch.cat([seq.mean(dim=1), seq.max(dim=1).values], dim=1)

        return {
            "ctc_log_probs": F.log_softmax(self.ctc_head(seq), dim=-1),
            "colour_logits": self.colour_head(pooled),
            # Quality is a proportion in [0, 1]; a sigmoid keeps it there
            # without the loss having to police the range.
            "quality": torch.sigmoid(self.quality_head(pooled)).squeeze(-1),
        }

    @torch.no_grad()
    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


class PlateNetExport(nn.Module):
    """Tuple-output wrapper for ONNX export.

    ONNX Runtime handles dict outputs inconsistently across versions and
    language bindings; a fixed tuple is portable everywhere the edge agent
    might run.
    """

    def __init__(self, net: PlateNet) -> None:
        super().__init__()
        self.net = net

    def forward(self, x: torch.Tensor):
        out = self.net(x)
        return out["ctc_log_probs"], out["colour_logits"], out["quality"]


__all__ = ["ModelConfig", "PlateNet", "PlateNetExport"]
