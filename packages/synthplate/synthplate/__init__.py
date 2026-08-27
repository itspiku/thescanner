"""``synthplate`` -- synthetic Nepali number plate generation.

All public Nepali plate data amounts to roughly five thousand images, heavily
skewed toward daylight photographs of private motorcycles in one zone. That is
not enough to train a production recogniser, and no amount of searching changes
it. So the data strategy is generation: render plates from the specification,
degrade them realistically, and use real data for fine-tuning and evaluation
only.

Because :mod:`nepal_plate.spec` is a complete machine-readable model of both
plate systems, the generator can produce a perfectly balanced, exhaustively
labelled corpus -- including the government, diplomatic, tourist and corporation
plates that barely exist in public data and where a misread costs the most.

See ``docs/research/datasets.md`` for the full strategy.
"""

from __future__ import annotations

from .fonts import fonts_are_authentic
from .render import CharBox, RenderedPlate, render
from .sampling import PlateSample, PlateSampler, sample_balanced_grid

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "PlateSample",
    "PlateSampler",
    "sample_balanced_grid",
    "CharBox",
    "RenderedPlate",
    "render",
    "fonts_are_authentic",
]
