# Phase 2 findings — what the measurements actually said

> **Headline: the grammar-constrained decoder does not improve accuracy, and
> the plate-colour prior does not either.** Both were central claims of this
> project's design. Neither survived measurement. This document records what
> was measured, why the result came out that way, and what the mechanisms are
> actually worth — because they are worth something, just not what was claimed.

Setup: `PlateNet` (1.93 M params) trained for 20 epochs on 50,000 synthetic
plates; evaluated on the 2,000-sample held-out split, decoded with beam width
12. Reproduce with:

```bash
python -m scanner_models.evaluate --data data/synth --checkpoint models/platenet/best.pt
```

---

## 1. The ablation

| Stratum | n | greedy | grammar | grammar+colour | delta |
|---|---|---|---|---|---|
| **ALL** | 2,000 | **0.836** | 0.837 | **0.837** | **+0.001** |
| system: devanagari | 1,011 | 0.837 | 0.836 | 0.836 | −0.001 |
| system: embossed | 989 | 0.834 | 0.837 | 0.837 | +0.003 |
| layout: two-row | 1,080 | 0.949 | 0.950 | 0.950 | +0.001 |
| layout: single-row | 920 | 0.702 | 0.703 | 0.703 | +0.001 |
| quality: clean | 759 | 0.980 | 0.983 | 0.983 | +0.003 |
| quality: moderate | 850 | 0.855 | 0.854 | 0.854 | −0.001 |
| quality: degraded | 391 | 0.511 | 0.514 | 0.514 | +0.003 |
| width: 130px+ | 271 | 0.985 | 0.985 | 0.985 | +0.000 |
| width: 90–130px | 865 | 0.924 | 0.926 | 0.926 | +0.002 |
| width: 60–90px | 580 | 0.816 | 0.814 | 0.814 | −0.002 |
| width: 40–60px | 224 | 0.545 | 0.549 | 0.549 | +0.005 |
| width: 0–40px | 60 | 0.167 | 0.167 | 0.167 | +0.000 |

Colour head: **97.9%**. Quality head MAE: **0.070**.

### Under distribution shift

The obvious defence is that the constraint should matter once the model is
pushed off its training distribution. So the same evaluation was rerun with a
corruption the generator never produced (uniform downscale-then-upscale plus
additive noise plus aggressive JPEG), applied at inference only:

| shift | greedy | grammar | +colour | grammar gain |
|---|---|---|---|---|
| 0.00 | 0.851 | 0.850 | 0.850 | −0.001 |
| 0.30 | 0.730 | 0.731 | 0.731 | +0.001 |
| 0.50 | 0.678 | 0.680 | 0.680 | +0.002 |
| 0.70 | 0.590 | 0.590 | 0.590 | +0.000 |
| 0.85 | 0.522 | 0.523 | 0.523 | +0.001 |

The defence fails. Accuracy falls from 85% to 52% and the constraint still adds
nothing.

---

## 2. Why

Two measurements explain it completely.

**The model has already internalised the grammar.** Mean per-timestep
probability mass on grammar-illegal tokens: **3.1 × 10⁻³**. The recogniser was
trained exclusively on legal plates, so it learned that `ध` never appears in the
class position just as surely as the FSA encodes it. There is essentially no
illegal mass left for the constraint to redistribute.

**The recogniser is not ambiguous enough for a prior to help.** Mean top-1 minus
top-2 margin per emitted glyph: **0.927**. The colour prior changed the decoded
output for **2 of 400** legacy plates (0.5%). A prior can only break ties, and
there are almost no ties.

This is not a bug and the mechanisms are not broken — the unit tests in
`packages/nepal_plate/tests/` demonstrate both working exactly as designed. But
those tests construct posteriors by hand, deliberately placing mass on an
illegal glyph. **A trained model does not produce posteriors like that.** The
tests prove the code is correct; they do not prove the mechanism is useful, and
the earlier documentation conflated the two.

### Where this reasoning *would* hold

The grammar-constraint argument is sound in the regime it was borrowed from:
recognisers trained on generic text, or with far less capacity, or where the
label space at training time is broader than the language at inference time.
None of those describe this system. The lesson generalises: a structural prior
is worth what the model has *not* already learned, and a model trained on
in-language data has learned it.

---

## 3. What the grammar is actually worth

Not nothing. Just not accuracy.

Breaking down greedy's 2,000 outputs:

| | count | share |
|---|---|---|
| correct | 1,671 | 83.5% |
| wrong, but still a legal plate | 130 | 6.5% |
| **not a legal plate at all** | **199** | **10.0%** |

**One read in ten from greedy decoding is not a plate.** It cannot be a
watch-list key, cannot be stored in a typed column, cannot be matched against a
registry. The grammar decoder guarantees a well-formed, field-decomposed result
— zone, lot, class, serial, ownership — or an explicit refusal. That is a real
and necessary property for a system whose output feeds a database and an alert
queue, and it is the honest justification for keeping it.

It also produces **calibrated confidence bands**, which the flat string from
greedy cannot:

| band | n | accuracy |
|---|---|---|
| high | 1,434 | 0.978 |
| medium | 315 | 0.844 |
| low | 154 | 0.026 |
| reject | 96 | 0.000 |

The separation is sharp and monotone: the bands genuinely sort good reads from
bad ones.

### The risk it introduces

Grammar repair converts *self-evident garbage* into *plausible but wrong
plates*. Greedy's 10% invalid outputs are trivially detectable; a well-formed
wrong plate is not. In a law-enforcement context that is the more dangerous
failure, and it is created by the mechanism, not mitigated by it.

The `repaired` flags and confidence bands exist for exactly this, and the
calibration table shows they mostly work. But:

> **The HIGH-confidence false-positive rate is 2.2% (32 of 1,434), against a
> target of ≤0.5% in [`../PLAN.md`](../PLAN.md). That criterion is not met.**

This is the most operationally important number in the whole evaluation and it
currently fails. Tightening the HIGH band is Phase 2 rework, not a Phase 6
polish item.

---

## 4. Other findings

**Single-row plates are much harder than two-row: 70.2% vs 94.9%.** This is an
artefact of the preprocessing, not of the plates. Two-row plates are *unwrapped*
— split at the row boundary and laid side by side — which doubles their
effective horizontal resolution in a fixed 192-pixel input. Single-row plates
get the full plate squeezed into 192 pixels with no such bonus, so they are
resolution-starved. Widening the recogniser input, or scaling it by layout, is
the obvious fix and is tracked as Phase 2 rework.

**Accuracy is dominated by plate width**, as expected: 98.5% above 130 px,
54.9% at 40–60 px, 16.7% below 40 px. This is the axis worth engineering
against — lens choice and camera siting will move the numbers far more than
decoder cleverness.

**The auxiliary heads work well.** Colour at 97.9% and quality MAE at 0.070 both
comfortably support their downstream uses, and the quality head cost nothing to
supervise because the generator records the ground-truth degradation severity.

---

## 5. Consequences

1. Claims in `README.md` and `docs/research/prior-art.md` that grammar-constrained
   decoding and the colour prior "do the heavy lifting on degraded imagery" have
   been corrected. They are retained as correctness and calibration machinery,
   described as such.
2. `docs/PLAN.md` gains Phase 2 rework items: widen the recogniser input for
   single-row plates, and tighten the HIGH confidence band to meet the ≤0.5%
   false-positive criterion.
3. The 82.13% winning score at the ICPR 2026 low-resolution benchmark remains
   the right calibration for expectations. 83.6% here is on *synthetic* data
   from the same generator the model trained on, and should be read as an upper
   bound on what real Nepali footage will give, not a forecast of it.
4. Everything above is measured on synthetic data. `NepalPlate-Bench`
   (Phase 1.7) remains the only thing that can turn these into real numbers.
