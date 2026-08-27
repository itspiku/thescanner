# synthplate

Synthetic Nepali number plate generation for [TheScanner](https://github.com/itspiku/thescanner).

All public Nepali plate data amounts to roughly 5,000 images, heavily skewed to
daylight photographs of private motorcycles in one zone. That is not enough to
train a production recogniser. This package generates the rest.

Because [`nepal_plate.spec`](../nepal_plate) is a complete machine-readable
model of both of Nepal's plate systems, the generator produces a **balanced,
exhaustively labelled** corpus — including the government, diplomatic, tourist
and corporation plates that barely appear in public data and where a misread
costs the most.

| Module | Purpose |
|---|---|
| `sampling` | Draw legal plates from the grammar — uniform (coverage) or frequency-weighted (deployment-matched) |
| `fonts` | Font resolution, with loud warnings when falling back from the real plate typefaces |
| `render` | Clean plate rendering with per-character bounding boxes |
| `degrade` | Physically-grounded n-stage degradation |
| `tracks` | Coherent multi-frame sequences for the fusion path |

## Fonts matter

Embossed plates use **FE-Schrift**; legacy plates use a Devanagari plate face.
Rendering with the wrong typeface teaches the recogniser the wrong glyph shapes,
and it fails *silently*. Run:

```bash
python -m synthplate.fetch_fonts
```

The font actually used is recorded in every sample's metadata, and
`fonts_are_authentic()` gates production training runs.

Licence: Apache-2.0.
