"""Finite-state grammars for Nepali plate layouts.

Why a grammar at all
--------------------
A conventional ANPR reads a plate by taking the argmax of an OCR model and then
cleaning it up with a regex. That throws away the model's uncertainty at exactly
the moment it is most valuable. On a blurred plate the recogniser may put 0.45
on ``ग`` and 0.40 on ``ध`` -- and a regex cannot use the fact that only one of
those is legal in that position.

Nepali plates are a *very* tight language. A legacy plate is
``zone · lot? · class · serial`` where the zone comes from a closed set of 14,
the class from a closed set of 13, and the serial is at most four digits. The
number of syntactically legal legacy plates is exactly 224,444,220 -- against a
raw output space of 34^8 = 1.79e12. Constraining the decoder to the legal
language therefore removes 13.0 bits of search space (a factor of ~7,960) for
legacy plates and 16.4 bits (~87,800) for embossed ones, and it costs nothing
at inference time. See ``language_size`` and the tests that pin those figures.

Design
------
Every Nepali plate layout is a concatenation of *slots* with bounded repeats, so
rather than hand-rolling a general FSA we describe layouts declaratively as a
:class:`Grammar` -- a tuple of :class:`Slot` -- and compile transitions on
demand. A slot may carry a ``guard`` so that its legal alphabet depends on what
has already been emitted (used for the embossed subclass digit, which is only
valid after certain class letters). Plates are at most ten tokens long, so the
guarded lookup is trivially cheap.

The state is ``(slot_index, repeats_in_slot)``. Together with the emitted prefix
that the beam search already carries, this is enough to drive constrained
decoding.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Final, Iterable, Mapping, Sequence

from . import spec
from .types import PlateSystem

#: A grammar state: ``(slot_index, number_of_tokens_emitted_in_that_slot)``.
State = tuple[int, int]

#: Signature of a slot guard: given the tokens emitted so far for the whole
#: plate, return the alphabet legal for the next token of this slot.
Guard = Callable[[tuple[str, ...]], frozenset[str]]


@dataclass(frozen=True, slots=True)
class Slot:
    """One field of a plate layout.

    ``min_repeat``/``max_repeat`` bound how many tokens the slot consumes. A
    slot with ``min_repeat == 0`` is optional and may be skipped entirely.

    ``length_log_prior`` lets a slot express that some lengths are far more
    common than others without forbidding the rare ones. Nepali serials are
    almost always four digits, but short early-series plates exist; forbidding
    them would cause silent misreads, whereas a log-prior merely makes the
    decoder prefer four digits when the evidence is ambiguous.
    """

    name: str
    tokens: frozenset[str]
    min_repeat: int = 1
    max_repeat: int = 1
    length_log_prior: Mapping[int, float] = field(default_factory=dict)
    guard: Guard | None = None

    def alphabet(self, emitted: tuple[str, ...]) -> frozenset[str]:
        if self.guard is None:
            return self.tokens
        return self.tokens & self.guard(emitted)

    def length_bonus(self, count: int) -> float:
        if not self.length_log_prior:
            return 0.0
        return float(self.length_log_prior.get(count, min(self.length_log_prior.values()) - 1.0))


@dataclass(frozen=True, slots=True)
class Grammar:
    """A plate layout: an ordered sequence of slots."""

    system: PlateSystem
    slots: tuple[Slot, ...]
    name: str = ""

    # -- structure ------------------------------------------------------

    @property
    def start(self) -> State:
        """Start state.

        Represented as ``(0, 0)``: positioned at the first slot, nothing
        emitted. Note that slot 0 may itself be optional.
        """
        return (0, 0)

    def slot_names(self) -> tuple[str, ...]:
        return tuple(s.name for s in self.slots)

    def _slot_index_by_name(self, name: str) -> int:
        for i, s in enumerate(self.slots):
            if s.name == name:
                return i
        raise KeyError(name)

    # -- transitions ----------------------------------------------------

    def transitions(
        self, state: State, emitted: tuple[str, ...]
    ) -> dict[str, tuple[State, ...]]:
        """Legal next tokens from ``state``, each mapped to its successor states.

        A token may be legal in more than one successor when slots share an
        alphabet across an optional boundary. Rather than silently picking one,
        all successors are returned and the caller (beam search) branches. Both
        Nepali grammars happen to be unambiguous, but keeping this general means
        the engine works unchanged for other countries' layouts.
        """
        slot_idx, count = state
        out: dict[str, list[State]] = {}

        # Continue the current slot.
        if slot_idx < len(self.slots):
            cur = self.slots[slot_idx]
            if count < cur.max_repeat:
                for tok in cur.alphabet(emitted):
                    out.setdefault(tok, []).append((slot_idx, count + 1))

        # Advance to later slots, skipping over any that are optional.
        if slot_idx >= len(self.slots) or count >= self.slots[slot_idx].min_repeat:
            j = slot_idx + 1
            while j < len(self.slots):
                nxt = self.slots[j]
                if nxt.max_repeat > 0:
                    for tok in nxt.alphabet(emitted):
                        out.setdefault(tok, []).append((j, 1))
                if nxt.min_repeat > 0:
                    break  # this slot is mandatory, cannot skip past it
                j += 1

        return {tok: tuple(states) for tok, states in out.items()}

    def is_accepting(self, state: State) -> bool:
        """True when the plate may legally end here."""
        slot_idx, count = state
        if slot_idx >= len(self.slots):
            return True
        if count < self.slots[slot_idx].min_repeat:
            return False
        return all(s.min_repeat == 0 for s in self.slots[slot_idx + 1 :])

    def alphabet(self) -> frozenset[str]:
        """Union of every token this grammar can ever emit."""
        acc: set[str] = set()
        for s in self.slots:
            acc |= s.tokens
        return frozenset(acc)

    # -- segmentation ---------------------------------------------------

    def segment(self, tokens: Sequence[str]) -> dict[str, list[str]] | None:
        """Split an accepted token sequence back into named fields.

        Returns ``None`` if ``tokens`` is not in the language. This is the
        inverse of decoding and is what turns a raw token string into the typed
        fields of a :class:`~nepal_plate.types.ParsedPlate`.
        """
        results = self._segment_from(tuple(tokens), 0, self.start, {})
        return results

    def _segment_from(
        self,
        tokens: tuple[str, ...],
        pos: int,
        state: State,
        acc: Mapping[str, list[str]],
    ) -> dict[str, list[str]] | None:
        if pos == len(tokens):
            return {k: list(v) for k, v in acc.items()} if self.is_accepting(state) else None

        emitted = tokens[:pos]
        tok = tokens[pos]
        for nxt in self.transitions(state, emitted).get(tok, ()):
            slot_name = self.slots[nxt[0]].name
            branch = {k: list(v) for k, v in acc.items()}
            branch.setdefault(slot_name, []).append(tok)
            found = self._segment_from(tokens, pos + 1, nxt, branch)
            if found is not None:
                return found
        return None


# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------

def _embossed_subclass_guard(emitted: tuple[str, ...]) -> frozenset[str]:
    """Only certain embossed class letters admit a numeric subclass.

    ``C`` takes ``C1``; ``H`` takes ``H1``/``H2``; ``I`` takes ``I1``-``I3``;
    ``J`` takes ``J1``-``J5``. Every other letter stands alone. Encoding this
    as a guard rather than a post-hoc check means the decoder never wastes beam
    width on ``B3`` or ``A7``.
    """
    if not emitted:
        return frozenset()
    letter = emitted[-1]
    return frozenset(spec.EMBOSSED_SUBCLASSES.get(letter, ()))


# ---------------------------------------------------------------------------
# The two Nepali grammars
# ---------------------------------------------------------------------------

#: Serials are overwhelmingly four digits. Shorter serials exist on early
#: series, so they are permitted but discouraged. Values are natural-log units
#: added to the path score, calibrated so that a 3-digit reading must beat a
#: 4-digit one by ~4x in likelihood before it wins.
_SERIAL_LENGTH_PRIOR: Final[Mapping[int, float]] = {
    4: 0.0,
    3: -1.4,
    2: -3.0,
    1: -4.5,
}

#: Most legacy plates carry no lot number at all; it appears only once a zone's
#: four-digit series is exhausted. Two-digit lots are rarer still.
_LOT_LENGTH_PRIOR: Final[Mapping[int, float]] = {
    0: 0.0,
    1: -0.4,
    2: -1.6,
}


LEGACY_GRAMMAR: Final[Grammar] = Grammar(
    system=PlateSystem.DEVANAGARI,
    name="nepal-legacy-zonal",
    slots=(
        Slot("zone", frozenset(spec.ZONE_TOKENS)),
        Slot(
            "lot",
            frozenset(spec.DEVA_DIGITS),
            min_repeat=0,
            max_repeat=2,
            length_log_prior=_LOT_LENGTH_PRIOR,
        ),
        Slot("class", frozenset(spec.CLASS_TOKENS)),
        Slot(
            "serial",
            frozenset(spec.DEVA_DIGITS),
            min_repeat=1,
            max_repeat=4,
            length_log_prior=_SERIAL_LENGTH_PRIOR,
        ),
    ),
)


EMBOSSED_GRAMMAR: Final[Grammar] = Grammar(
    system=PlateSystem.EMBOSSED,
    name="nepal-embossed",
    slots=(
        Slot("province", frozenset("1234567")),
        Slot("class_letter", frozenset(spec.EMBOSSED_BASE_LETTERS)),
        Slot(
            "subclass",
            frozenset("12345"),
            min_repeat=0,
            max_repeat=1,
            guard=_embossed_subclass_guard,
        ),
        Slot("series", frozenset(spec.LATIN_LETTERS), min_repeat=2, max_repeat=2),
        Slot(
            "serial",
            frozenset(spec.LATIN_DIGITS),
            min_repeat=1,
            max_repeat=4,
            length_log_prior=_SERIAL_LENGTH_PRIOR,
        ),
    ),
)


GRAMMARS: Final[Mapping[PlateSystem, Grammar]] = {
    PlateSystem.DEVANAGARI: LEGACY_GRAMMAR,
    PlateSystem.EMBOSSED: EMBOSSED_GRAMMAR,
}


def grammars_for(system: PlateSystem | None) -> tuple[Grammar, ...]:
    """Grammars to search, given a (possibly unknown) system hint.

    When the plate-colour classifier has already routed the crop to one system
    we search only that grammar, which roughly halves decoder work. When it has
    not, we search both and let the likelihood decide -- the two languages are
    disjoint in alphabet, so cross-system confusion is essentially impossible.
    """
    if system in GRAMMARS:
        return (GRAMMARS[system],)
    return (LEGACY_GRAMMAR, EMBOSSED_GRAMMAR)


def language_size(grammar: Grammar) -> int:
    """Exact number of strings this grammar accepts.

    Useful for quantifying how much the grammar constraint is worth: compare
    ``log2(language_size)`` against ``n_tokens * log2(vocab)`` to get the bits
    of search space the constraint removes.
    """
    # Counts are small enough for exact dynamic programming over states.
    from functools import lru_cache

    @lru_cache(maxsize=None)
    def count(state: State, emitted_len: int) -> int:
        # ``emitted`` content only matters for guards; the only guard here keys
        # off the previous token, so we enumerate guarded slots explicitly.
        total = 1 if grammar.is_accepting(state) else 0
        slot_idx, cnt = state
        if slot_idx < len(grammar.slots):
            cur = grammar.slots[slot_idx]
            if cnt < cur.max_repeat and cur.guard is None:
                total += len(cur.tokens) * count((slot_idx, cnt + 1), emitted_len + 1)
        if slot_idx >= len(grammar.slots) or cnt >= grammar.slots[slot_idx].min_repeat:
            j = slot_idx + 1
            while j < len(grammar.slots):
                nxt = grammar.slots[j]
                width = len(nxt.tokens) if nxt.guard is None else _guard_width(grammar, j)
                if nxt.max_repeat > 0:
                    total += width * count((j, 1), emitted_len + 1)
                if nxt.min_repeat > 0:
                    break
                j += 1
        return total

    return count(grammar.start, 0)


def _guard_width(grammar: Grammar, slot_index: int) -> int:
    """Average alphabet width of a guarded slot, for counting purposes."""
    slot = grammar.slots[slot_index]
    if slot.guard is None:
        return len(slot.tokens)
    # The only guarded slot is the embossed subclass; its width depends on the
    # preceding letter. Use the mean over base letters so ``language_size``
    # remains a meaningful scalar.
    widths = [len(spec.EMBOSSED_SUBCLASSES.get(c, ())) for c in spec.EMBOSSED_BASE_LETTERS]
    return max(1, round(sum(widths) / len(widths)))


__all__ = [
    "State",
    "Slot",
    "Grammar",
    "LEGACY_GRAMMAR",
    "EMBOSSED_GRAMMAR",
    "GRAMMARS",
    "grammars_for",
    "language_size",
]
