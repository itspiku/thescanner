# TheScanner

**Vehicle movement intelligence for Nepal.** Cameras on the road; every vehicle
identified, its plate read, its entry and exit from a zone recorded — on
hardware a municipality can afford, under Nepali privacy law, with the data
staying in the country.

> **Status: early.** Two things are built and tested: the **domain core** (plate
> specification, layout grammars, grammar-constrained decoder, multi-frame
> fusion) and the **synthetic data generator** (renders every legal plate in
> both systems, degrades it through a physically-ordered pipeline, and
> synthesises multi-frame vehicle tracks). Models, edge pipeline and platform
> are not implemented yet, and no real Nepali imagery has been evaluated.
> [`docs/PLAN.md`](docs/PLAN.md) has the honest status of every component.

---

## The problem, stated precisely

Nepal runs **two incompatible plate systems at the same time**, and will for
years:

- **Legacy zonal plates** — Devanagari script, `बा १ च १२३४`, colour-coded by
  ownership (red = private, black = public, green = tourist, …). Still the
  majority of vehicles.
- **Embossed plates** — Latin FE-Schrift, `3 B PA 1234`, uniformly
  black-on-white, RFID chip. Rolling out since 2020.

Nearly every published Nepali ANPR handles one or the other. A system meant for
real roads has to handle both, and must never apply one system's rules to the
other's plate.

---

## What makes this different

### 1. The decoder emits plates, not strings

Conventional ANPR takes the argmax of an OCR model and cleans it up with a
regex. TheScanner runs CTC beam search where every beam carries a state in a
finite-state grammar of Nepali plate layouts, so the decoder can only produce
a well-formed plate — decomposed into zone, lot, class and serial, with
ownership derived — or an explicit refusal.

**This does not improve accuracy, and we measured that carefully rather than
assuming it.** Against greedy decoding it is worth +0.001, and no more than
+0.002 even under heavy distribution shift. The reason is instructive: a model
trained only on legal plates has already internalised the grammar, leaving just
3.1 × 10⁻³ of its probability mass on illegal tokens. There is nothing for the
constraint to redistribute. The full write-up is in
[`docs/research/findings-phase2.md`](docs/research/findings-phase2.md).

What it *is* worth is well-formedness. **10.0% of greedy reads are not legal
plates at all** — they cannot be a watch-list key, a typed database column, or a
registry lookup. And the grammar is what makes confidence bands meaningful:
HIGH reads are 97.8% accurate, REJECT reads 0%. A flat string cannot tell you
which of those it is.

### 2. Plate colour is used as a decoding prior — with an honest caveat

Legacy Nepali plates encode ownership **twice**: in the background colour *and*
in the class letter. Colour is a large-area, low-frequency cue that survives
blur far better than glyph shape, so a red plate restricts the class letter to
क / च / प. The mechanism works and is unit-tested directly.

**On a trained model it changes 0.5% of reads and nets nothing.** The
recogniser's mean top-1-minus-top-2 margin per glyph is 0.927 — it is already
near-certain, and a prior can only break ties. The colour head is retained at
97.9% accuracy because ownership class is useful in its own right, and because
colour–class disagreement is a genuine signal for an altered plate.

### 3. A read is an estimate over a track, not a guess from a frame

A camera gets ten to forty looks at a vehicle, each with independent blur,
angle and lighting. The information needed to read a plate is usually present
across the set even when no single frame carries it — which is why the
[ICPR 2026 low-resolution benchmark](https://arxiv.org/abs/2604.22506) is built
from tracks rather than images.

Fusion here is **per-field**, not per-string: if the zone is legible in frame 3
and the serial in frame 11, whole-string voting discards both partial reads
while field-level voting keeps them. The assembled consensus is then
re-validated against the grammar, so it can never be a plate that could not
exist.

### 4. Built for the deployment that actually exists

Kathmandu Valley Traffic Police already run ANPR — six proprietary cameras
feeding a 297-camera control room, with expansion to ~170 sites planned. The gap
is not "can it be done" but "can it be done at a hundred sites on a Nepali
public budget". Hence: edge-first, commodity hardware, permissive licences only,
one database instead of three, and store-and-forward queues so a site keeps
working through a power cut.

### 5. Government-grade means auditable, not just encrypted

Hash-chained append-only read log with per-node Ed25519 signatures, so a read
can be defended in court. Mandatory reason-for-access logging. Automatic
retention expiry and erasure under Nepal's Privacy Act 2075. Face recognition is
deliberately **out of scope** — see
[`docs/security-and-privacy.md`](docs/security-and-privacy.md).

---

## Repository layout

```
packages/
  nepal_plate/     domain core — spec, grammars, decoder, fusion (zero deps) ✅
  synthplate/      synthetic plate renderer + degradation + track synthesis ✅
services/
  edge/            RTSP → detect → track → recognise → fuse → queue         ⬜
  api/             ingest, screening, search, evidence chain                ⬜
  web/             operator console                                         ⬜
docs/
  PLAN.md          phased delivery plan with acceptance criteria
  architecture.md  system design
  security-and-privacy.md
  research/        plate specification, prior art, dataset survey
```

---

## Quickstart

```bash
pip install -e "packages/nepal_plate[dev]" -e "packages/synthplate[dev]"
```

```bash
python -m pytest packages/nepal_plate/tests packages/synthplate/tests -q
```

Generate synthetic plates and eyeball them:

```bash
python -m synthplate.cli preview --out preview.png --rows 6 --cols 6
```

Build a corpus (images + JSONL labels + a reproducibility manifest):

```bash
python -m synthplate.cli generate --out data/synth --count 200000 --seed 1
```

```python
from nepal_plate import parse, decode, ColourEvidence, PlateColour

p = parse("बा १ च १२३४")
p.canonical    # 'NP-L:BA-1-CHA-1234'
p.ownership    # Ownership.PRIVATE
p.size_class   # SizeClass.LIGHT

# Every spelling of a plate must produce the same key, or watch-list
# matching silently fails.
parse("BA 1 CHA 1234").canonical == p.canonical   # True

# Decode from recogniser output, with a colour prior.
red = ColourEvidence({PlateColour.RED_WHITE: 0.88})
decode(log_probs, colour=red)[0].plate.display
```

---

## Documentation

| | |
|---|---|
| [Delivery plan](docs/PLAN.md) | Phases, deliverables, acceptance criteria, risks, open questions |
| [Architecture](docs/architecture.md) | Edge → Ingest → Screen → Exploit |
| [Plate specification](docs/research/plate-specification.md) | Both systems, in full, with sources |
| [Prior art](docs/research/prior-art.md) | What exists in Nepal and globally, and where the gap is |
| [Dataset survey](docs/research/datasets.md) | What public data exists (not much) and the strategy |
| [Phase 2 findings](docs/research/findings-phase2.md) | The ablation that disproved this project's headline claim, and what the mechanisms are actually worth |
| [Security & privacy](docs/security-and-privacy.md) | Threat model, Privacy Act 2075 compliance |

---

## Measured results

A 1.93 M-parameter recogniser trained for 20 epochs on 50,000 synthetic plates,
scored on a 2,000-sample held-out split:

| | |
|---|---|
| Full-plate exact match | **83.6%** |
| Plate colour (7-way) | **97.9%** |
| Crop-quality MAE | **0.070** |
| Clean crops (quality ≥ 0.7) | 98.3% |
| Degraded crops (quality < 0.4) | 51.4% |
| Plates ≥ 130 px wide | 98.5% |
| Plates 40–60 px wide | 54.9% |

Two results that matter more than the headline:

- **Single-row plates score 70.2% against two-row plates' 94.9%** — an artefact
  of preprocessing, not of the plates. Two-row plates get unwrapped to double
  horizontal resolution; single-row plates are resolution-starved in the same
  fixed input. Fixable, and tracked.
- **The HIGH-confidence false-positive rate is 2.2%, against a 0.5% target.**
  That criterion is not met. It is the most operationally important number here,
  because a wrong plate asserted confidently is what puts the wrong person in
  front of a magistrate.

## Honesty about what is and isn't proven

Everything above is measured on **synthetic data from the same generator the
model trained on**. It should be read as an upper bound on real-world
performance, not a forecast of it. No claim here has been validated against real
Nepali road footage, because the evaluation set to do that with does not exist
and has to be built (Phase 1.7). The recogniser also currently renders training
data in fallback typefaces rather than FE-Schrift (Phase 1.9).

For calibration: the winning entry in the ICPR 2026 low-resolution plate
competition scored **82.13%**. Treat any ANPR claiming 99% on degraded imagery
with suspicion — including this one.

---

## Licence

Apache-2.0. Dependencies are permissive-licence-only by policy — AGPL components
such as Ultralytics YOLO are excluded from release builds, because a copyleft
obligation is a procurement blocker for a government deployment.
