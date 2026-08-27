"""Tests for the recogniser.

The load-bearing test here is ``test_model_can_overfit_a_tiny_batch``. Shape
assertions prove a network is wired up; only a training run proves it can
learn. A model with a subtly wrong loss, a detached graph or a mis-ordered CTC
tensor will pass every shape check and then train to nothing, and that failure
is expensive to find later.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from nepal_plate import PlateColour, spec

from scanner_models.model import ModelConfig, PlateNet, PlateNetExport
from scanner_models.preprocess import (
    INPUT_H,
    INPUT_W,
    TWO_ROW_ASPECT,
    fit,
    looks_two_row,
    prepare,
    quad_aspect,
    unwrap_two_row,
)
from scanner_models.train import ctc_loss_fp32, _levenshtein
from scanner_models.vocab import (
    BLANK,
    COLOUR_CLASSES,
    N_COLOURS,
    VOCAB,
    VOCAB_SIZE,
    collapse,
    colour_index,
    encode,
)


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

def test_vocab_matches_the_domain_spec():
    """The model's output layer and the decoder's vocabulary must be the same
    object, or the decoder indexes into the wrong classes."""
    assert VOCAB == spec.UNIFIED_VOCAB
    assert VOCAB_SIZE == 71
    assert VOCAB[BLANK] == spec.CTC_BLANK


def test_encode_round_trips():
    toks = ["बा", "१", "च", "१", "२", "३", "४"]
    idx = encode(toks)
    assert [VOCAB[i] for i in idx] == toks


def test_encode_rejects_unknown_tokens():
    """Silently dropping an unknown token would corrupt a CTC target and be
    invisible until accuracy came out wrong."""
    with pytest.raises(ValueError):
        encode(["बा", "NOT_A_TOKEN"])


def test_collapse_implements_ctc_semantics():
    idx = encode(["बा", "च"])
    # repeat with an intervening blank -> two tokens; without -> one
    assert collapse([idx[0], idx[0], BLANK, idx[1]]) == ["बा", "च"]
    assert collapse([idx[0], BLANK, BLANK, idx[0]]) == ["बा", "बा"]
    assert collapse([idx[0], idx[0], idx[0]]) == ["बा"]


def test_colour_classes_cover_every_scheme_the_renderer_produces():
    assert set(COLOUR_CLASSES) == set(spec.LEGACY_COLOUR_OWNERSHIP) | {PlateColour.WHITE_BLACK}
    assert N_COLOURS == 7


def test_unknown_colour_falls_back_to_embossed():
    """Anything not one of the six legacy schemes is black-on-white by
    definition, so there is no need for an 'unknown' class to dilute the head."""
    assert colour_index(PlateColour.UNKNOWN) == COLOUR_CLASSES.index(PlateColour.WHITE_BLACK)


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def test_layout_threshold_separates_the_two_populations():
    """Single-row plates are 3.9-4.8, two-row 1.55-1.95. The threshold sits in
    the empty gap, so it needs no classifier and no tuning."""
    from synthplate.render import ASPECT_SINGLE_ROW, ASPECT_TWO_ROW

    assert ASPECT_TWO_ROW[1] < TWO_ROW_ASPECT < ASPECT_SINGLE_ROW[0]
    assert looks_two_row(100, 60)        # 1.67
    assert not looks_two_row(400, 100)   # 4.0


def test_unwrap_doubles_width_and_preserves_content():
    img = Image.new("RGB", (100, 60), (10, 20, 30))
    out = unwrap_two_row(img)
    assert out.width == 200
    assert out.height >= 30


def test_prepare_always_returns_the_model_input_shape():
    for size in [(100, 60), (400, 100), (37, 21), (900, 190)]:
        arr = prepare(Image.new("RGB", size, (128, 128, 128)))
        assert arr.shape == (3, INPUT_H, INPUT_W), size
        assert arr.dtype == np.float32


def test_quad_aspect_survives_a_perspective_warp():
    """A warped plate's bounding box is a poor shape estimate; its quad is not."""
    # A wide plate sheared into a parallelogram: the box is nearly square,
    # but the quad's edges still say it is wide.
    quad = [(0, 20), (400, 0), (400, 100), (0, 120)]
    assert quad_aspect(quad) > 3.0
    square = [(0, 0), (100, 0), (100, 60), (0, 60)]
    assert quad_aspect(square) < TWO_ROW_ASPECT


def test_two_row_plate_unwraps_to_a_left_to_right_sequence():
    """The whole point of unwrapping: after it, glyphs read in grammar order.

    Checked by construction -- the top half must land in the left half of the
    output and the bottom half in the right.
    """
    img = Image.new("RGB", (100, 60))
    img.paste(Image.new("RGB", (100, 30), (255, 0, 0)), (0, 0))    # top: red
    img.paste(Image.new("RGB", (100, 30), (0, 0, 255)), (0, 30))   # bottom: blue
    out = np.asarray(unwrap_two_row(img))
    left = out[:, : out.shape[1] // 4].reshape(-1, 3).mean(axis=0)
    right = out[:, -out.shape[1] // 4 :].reshape(-1, 3).mean(axis=0)
    assert left[0] > left[2], "top row should be on the left"
    assert right[2] > right[0], "bottom row should be on the right"


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def net():
    torch.manual_seed(0)
    return PlateNet(ModelConfig())


def test_forward_shapes(net):
    out = net(torch.randn(3, 3, INPUT_H, INPUT_W))
    cfg = net.cfg
    assert out["ctc_log_probs"].shape == (3, cfg.seq_len, VOCAB_SIZE)
    assert out["colour_logits"].shape == (3, N_COLOURS)
    assert out["quality"].shape == (3,)


def test_ctc_output_is_already_log_softmax(net):
    """The exported graph must emit exactly what ``nepal_plate.decode`` expects,
    so no post-processing has to be duplicated in the edge agent."""
    out = net(torch.randn(2, 3, INPUT_H, INPUT_W))
    total = out["ctc_log_probs"].exp().sum(-1)
    assert torch.allclose(total, torch.ones_like(total), atol=1e-4)


def test_quality_head_is_bounded(net):
    q = net(torch.randn(8, 3, INPUT_H, INPUT_W))["quality"]
    assert (q >= 0).all() and (q <= 1).all()


def test_enough_timesteps_for_the_longest_plate(net):
    """CTC needs a blank between repeated glyphs, so a 9-token plate needs 17
    steps in the worst case."""
    assert net.cfg.seq_len >= 2 * 9 - 1


def test_model_is_small_enough_for_the_edge(net):
    assert net.n_parameters() < 3_000_000, net.n_parameters()


def test_ctc_loss_is_finite_on_realistic_targets(net):
    out = net(torch.randn(4, 3, INPUT_H, INPUT_W))
    toks = ["बा", "१", "च", "१", "२", "३", "४"]
    targets = torch.tensor(encode(toks) * 4, dtype=torch.long)
    lens = torch.tensor([len(toks)] * 4, dtype=torch.long)
    loss = ctc_loss_fp32(out["ctc_log_probs"], targets, lens)
    assert torch.isfinite(loss) and loss > 0


def test_model_can_overfit_a_tiny_batch():
    """The test that actually proves the training path works.

    Four fixed images, four fixed labels, 500 steps. A correctly wired model
    memorises them; one with a detached graph, a mis-permuted CTC tensor or a
    wrong blank index will not, while still passing every shape assertion above.

    500 steps is chosen with headroom: the loss reaches ~0.07 by step 200 and
    ~0.002 by step 500, and full memorisation only lands once it is well under
    0.01. A tighter budget makes the test flaky for reasons that have nothing
    to do with the defects it is meant to catch.
    """
    torch.manual_seed(0)
    model = PlateNet(ModelConfig(dropout=0.0))
    plates = [
        ["बा", "१", "च", "१", "२", "३", "४"],
        ["को", "प", "२", "४", "७", "९"],
        ["3", "B", "P", "A", "1", "2", "3", "4"],
        ["7", "G", "K", "L", "0", "0", "8", "1"],
    ]
    x = torch.randn(len(plates), 3, INPUT_H, INPUT_W)
    targets = torch.tensor([i for p in plates for i in encode(p)], dtype=torch.long)
    lens = torch.tensor([len(p) for p in plates], dtype=torch.long)

    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    model.train()
    first = last = None
    for step in range(500):
        opt.zero_grad()
        loss = ctc_loss_fp32(model(x)["ctc_log_probs"], targets, lens)
        loss.backward()
        opt.step()
        last = float(loss.detach())
        if first is None:
            first = last

    # Report the loss trajectory separately: if this assertion fails the model
    # is not learning at all, which is a different bug from learning but not
    # converging far enough.
    assert last < first / 100, f"loss barely moved: {first:.2f} -> {last:.2f}"

    model.eval()
    with torch.no_grad():
        pred = model(x)["ctc_log_probs"].argmax(dim=-1)
    got = [collapse(row.tolist()) for row in pred]
    assert got == plates, f"failed to memorise 4 samples (final loss {last:.4f}): {got}"


def test_levenshtein():
    assert _levenshtein(list("abc"), list("abc")) == 0
    assert _levenshtein(list("abc"), list("abd")) == 1
    assert _levenshtein([], list("ab")) == 2


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def tiny_corpus(tmp_path_factory) -> Path:
    """A miniature synthplate corpus, generated on the fly."""
    import warnings

    from synthplate.degrade import degrade
    from synthplate.render import render
    from synthplate.sampling import PlateSampler

    root = tmp_path_factory.mktemp("corpus")
    (root / "images").mkdir()
    rng = random.Random(0)
    sampler = PlateSampler(seed=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with (root / "index.jsonl").open("w", encoding="utf-8") as fh:
            for i in range(24):
                s = sampler.sample()
                d = degrade(render(s, height=96, rng=rng), rng=rng)
                rel = f"images/{i:04d}.jpg"
                d.image.save(root / rel, quality=90)
                fh.write(json.dumps({
                    "path": rel,
                    "tokens": list(s.tokens),
                    "canonical": s.plate.canonical,
                    "colour": s.colour.value,
                    "quality": d.quality,
                    "two_row": s.two_row,
                    "system": s.plate.system.value,
                    "width_px": d.image.width,
                }, ensure_ascii=False) + "\n")
    return root


def test_dataset_yields_all_three_supervision_signals(tiny_corpus):
    from scanner_models.data import PlateDataset, collate

    ds = PlateDataset(tiny_corpus, train=False)
    assert len(ds) == 24
    item = ds[0]
    assert item["image"].shape == (3, INPUT_H, INPUT_W)
    assert item["target_len"].item() == len(ds.samples[0].tokens)
    assert 0 <= item["colour"].item() < N_COLOURS
    assert 0.0 <= item["quality"].item() <= 1.0

    batch = collate([ds[i] for i in range(4)])
    assert batch["image"].shape == (4, 3, INPUT_H, INPUT_W)
    # CTC targets are concatenated, not padded -- so no padding token exists.
    assert batch["target"].numel() == int(batch["target_len"].sum())


def test_validation_split_never_splits_a_track():
    """Frames of one passage are near-duplicates. Splitting by frame would put
    near-identical images on both sides and report a training score as if it
    were a validation score."""
    from scanner_models.data import Sample, split_samples

    samples = [
        Sample(f"i{t}_{f}.jpg", ("बा",), "c", PlateColour.RED_WHITE, 0.5, False, "devanagari", 60,
               track_id=f"t{t}", frame=f)
        for t in range(20) for f in range(5)
    ]
    tr, va = split_samples(samples, val_fraction=0.2, seed=0)
    assert {s.track_id for s in tr}.isdisjoint({s.track_id for s in va})
    assert va


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def test_onnx_export_matches_pytorch(tmp_path, net):
    ort = pytest.importorskip("onnxruntime")

    path = tmp_path / "m.onnx"
    wrapper = PlateNetExport(net).eval()
    torch.onnx.export(
        wrapper, torch.randn(1, 3, INPUT_H, INPUT_W), str(path),
        input_names=["image"],
        output_names=["ctc_log_probs", "colour_logits", "quality"],
        dynamic_axes={"image": {0: "batch"}, "ctc_log_probs": {0: "batch"},
                      "colour_logits": {0: "batch"}, "quality": {0: "batch"}},
        opset_version=17, do_constant_folding=True, dynamo=False,
    )
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    # Batch 3 against a graph exported at batch 1. This is the assertion that
    # catches the torch 2.13 dynamo exporter baking the tracing batch size into
    # the BiLSTM reshape -- without a dynamic batch the edge agent could only
    # ever submit one crop at a time.
    x = torch.randn(3, 3, INPUT_H, INPUT_W)
    with torch.no_grad():
        ref = wrapper(x)
    got = sess.run(None, {"image": x.numpy()})
    for a, b in zip(ref, got):
        assert np.abs(a.numpy() - b).max() < 1e-3
