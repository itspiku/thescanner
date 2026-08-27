# nepal-plate

The domain core of [TheScanner](https://github.com/itspiku/thescanner): a
dependency-free model of Nepali vehicle registration plates.

Nepal runs two incompatible plate systems side by side — legacy Devanagari
zonal plates (`बा १ च १२३४`, colour-coded by ownership) and post-2020 embossed
Latin plates (`3 B PA 1234`, FE-Schrift, black-on-white). This package models
both.

## What's in it

| Module | Purpose |
|---|---|
| `spec` | Authoritative reference tables — zones, class letters, provinces, colour schemes, vocabularies, confusion groups |
| `grammar` | Finite-state layout grammars for both systems |
| `decode` | **Grammar-constrained CTC beam search** with a plate-colour decoding prior |
| `fuse` | Multi-frame track fusion with per-field consensus |
| `parse` | Tolerant parsing and canonicalisation of plate strings |
| `types` | Value types shared across the system |

## Why it has no dependencies

This package is imported by the edge agent, the training pipeline, the
synthetic renderer and the API. Any dependency here would be a dependency
everywhere. It is pure standard library, and stays that way.

## Install

```bash
pip install -e ".[dev]"
python -m pytest tests -q
```

## Use

```python
from nepal_plate import parse, decode, ColourEvidence, PlateColour

p = parse("बा १ च १२३४")
p.canonical      # 'NP-L:BA-1-CHA-1234'
p.ownership      # Ownership.PRIVATE

# Grammar-constrained decode with a colour prior: a red background means
# private ownership, which restricts the class letter to क / च / प.
decode(log_probs, colour=ColourEvidence({PlateColour.RED_WHITE: 0.9}))
```

Licence: Apache-2.0.
