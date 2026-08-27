"""Fetch the authentic plate typefaces into ``assets/fonts/``.

    python -m synthplate.fetch_fonts

Why this is a separate step rather than a bundled asset: the two faces have
different licensing situations and neither should be vendored into the
repository without a deliberate decision.

* **Noto Sans Devanagari** is SIL Open Font License 1.1 -- freely
  redistributable, downloaded automatically here.
* **FE-Schrift** is the German *fälschungserschwerende Schrift*, the typeface
  Nepali embossed plates use. It is published by the German government and
  widely mirrored, but the redistribution terms vary by mirror. This script
  will fetch it if a source is configured and otherwise tells you what to do,
  rather than silently pulling a file of uncertain provenance into a codebase
  destined for government deployment.

Rendering with the wrong typeface fails *silently*: the corpus looks plausible
and the model underperforms on real plates for reasons that are very hard to
diagnose afterwards. So ``synthplate`` records the font used in every sample's
metadata and in the corpus manifest, and :func:`synthplate.fonts_are_authentic`
lets a training run refuse to start on fallback-rendered data.
"""

from __future__ import annotations

import sys
import urllib.error
import urllib.request
from pathlib import Path

from .fonts import ASSET_DIR

#: (filename, url, licence). Only sources with clear redistribution terms.
SOURCES: tuple[tuple[str, str, str], ...] = (
    (
        "NotoSansDevanagari-Bold.ttf",
        "https://github.com/google/fonts/raw/main/ofl/notosansdevanagari/"
        "NotoSansDevanagari%5Bwdth%2Cwght%5D.ttf",
        "SIL Open Font License 1.1",
    ),
)

#: Faces we cannot fetch automatically, with instructions.
MANUAL: tuple[tuple[str, str], ...] = (
    (
        "FE-FONT.TTF",
        "FE-Schrift (embossed plates). Obtain a copy whose licence permits your "
        "use and place it at assets/fonts/FE-FONT.TTF, or point $SYNTHPLATE_LATIN_FONT "
        "at it. Until then embossed plates render in a condensed fallback face, "
        "which is adequate for development only.",
    ),
)

TIMEOUT = 60


def fetch(dest: Path | None = None) -> int:
    dest = dest or ASSET_DIR
    dest.mkdir(parents=True, exist_ok=True)
    failures = 0

    for name, url, licence in SOURCES:
        target = dest / name
        if target.is_file():
            print(f"  [have] {name}")
            continue
        print(f"  [get ] {name}  ({licence})")
        try:
            with urllib.request.urlopen(url, timeout=TIMEOUT) as r:
                data = r.read()
            if len(data) < 10_000:
                raise ValueError(f"suspiciously small response ({len(data)} bytes)")
            target.write_bytes(data)
            print(f"         -> {target} ({len(data):,} bytes)")
        except (urllib.error.URLError, ValueError, OSError) as exc:
            print(f"         FAILED: {exc}", file=sys.stderr)
            failures += 1

    for name, note in MANUAL:
        if (dest / name).is_file():
            print(f"  [have] {name}")
        else:
            print(f"  [todo] {name}\n         {note}")

    return failures


def main() -> int:
    print(f"Fetching plate fonts into {ASSET_DIR}")
    failures = fetch()
    from .fonts import resolve

    resolve.cache_clear()
    print()
    for script in ("latin", "devanagari"):
        f = resolve(script)
        mark = "OK  " if f.authentic else "WARN"
        print(f"  {mark} {script:11s} {f.name}" + ("" if f.authentic else "   (fallback)"))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
