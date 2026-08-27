"""Grammar-constrained CTC decoding for Nepali plates.

This is the part of the system that does the heavy lifting on degraded imagery.

The idea
--------
A recogniser emits, for every timestep, a distribution over the 71-token
vocabulary. Standard practice is to take the argmax, collapse repeats, drop
blanks, and then sanity-check the string with a regex. That is wasteful: the
regex arrives *after* the information has been destroyed.

Instead we run CTC prefix beam search where every beam additionally carries a
state in the plate grammar, and beams may only be extended by tokens the grammar
permits from that state. The search therefore explores only well-formed plates,
and the probability mass that argmax decoding would have spent on impossible
strings is redistributed onto plausible ones.

Concretely: on a blurred legacy plate where the class glyph sits at 0.45 ``ग``
(government heavy) and 0.40 ``ध``, argmax picks ``ग``. But ``ध`` is a *zone*
code and is not legal in the class position at all -- so the grammar removes it
outright and the remaining mass goes to the legal alternatives. Combine that
with the colour prior below and ambiguous glyphs frequently resolve exactly.

The colour prior
----------------
Legacy Nepali plates encode ownership twice: once in the class letter and once
in the background colour (red = private, black = public, white/red text =
government, yellow = corporation, green = tourist, blue = diplomatic). Colour
survives blur far better than glyph shape does -- it is a low-frequency,
large-area cue. So a colour classifier that is confident the plate is red
contributes ``log P(red | ownership)`` to every path whose class letter implies
private ownership, and effectively eliminates the rest.

This is, as far as the survey in ``docs/research/prior-art.md`` found, not done
by any existing ANPR: elsewhere plate colour is used at most to pick a country
profile, never as a per-path decoding constraint. It is available here only
because Nepal's legacy scheme happens to carry redundant ownership information.

Cost
----
Pure standard library, no NumPy. With T=32 frames, a 16-wide beam and ~35 legal
transitions per state, a decode is on the order of 20k float operations -- tens
of microseconds to a couple of milliseconds in CPython, against ~10 ms for the
recogniser forward pass it follows. Keeping it dependency-free means the same
code runs in the edge agent, the training loop and the API without divergence.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Mapping, Sequence

from . import spec
from .grammar import Grammar, State, grammars_for
from .parse import plate_from_tokens
from .types import (
    ColourEvidence,
    Ownership,
    ParsedPlate,
    PlateColour,
    PlateSystem,
    ReadCandidate,
)

NEG_INF = float("-inf")

#: Slot-level log bonus: ``(slot_name, token) -> log bonus``.
SlotBonus = Callable[[str, str], float]


def _logaddexp(a: float, b: float) -> float:
    if a == NEG_INF:
        return b
    if b == NEG_INF:
        return a
    if a > b:
        return a + math.log1p(math.exp(b - a))
    return b + math.log1p(math.exp(a - b))


@dataclass(slots=True)
class _Beam:
    """Accumulated probability for one ``(prefix, grammar state)`` pair."""

    lp_blank: float = NEG_INF
    lp_nonblank: float = NEG_INF
    #: Per-emitted-token log posterior along the single best path reaching here.
    #: Used only for reporting per-field confidence, never for search.
    token_lps: tuple[float, ...] = ()
    #: Score of the path that produced ``token_lps``, so a better path can
    #: replace it.
    best_path_lp: float = NEG_INF
    #: Slots where the grammar overrode the frame's best non-blank token.
    repaired: frozenset[str] = frozenset()

    def total(self) -> float:
        return _logaddexp(self.lp_blank, self.lp_nonblank)


def _merge(
    table: dict[tuple[tuple[str, ...], State], _Beam],
    key: tuple[tuple[str, ...], State],
    *,
    blank: float = NEG_INF,
    nonblank: float = NEG_INF,
    path_lp: float = NEG_INF,
    token_lps: tuple[float, ...] = (),
    repaired: frozenset[str] = frozenset(),
) -> None:
    entry = table.get(key)
    if entry is None:
        entry = _Beam()
        table[key] = entry
    entry.lp_blank = _logaddexp(entry.lp_blank, blank)
    entry.lp_nonblank = _logaddexp(entry.lp_nonblank, nonblank)
    if path_lp > entry.best_path_lp:
        entry.best_path_lp = path_lp
        entry.token_lps = token_lps
        entry.repaired = repaired


def ctc_grammar_beam_search(
    log_probs: Sequence[Sequence[float]],
    grammar: Grammar,
    *,
    vocab: Sequence[str] = spec.UNIFIED_VOCAB,
    blank_index: int = 0,
    beam_width: int = 16,
    prune_log_gap: float = -9.0,
    slot_bonus: SlotBonus | None = None,
) -> list[tuple[tuple[str, ...], float, tuple[float, ...], frozenset[str]]]:
    """Constrained CTC prefix beam search.

    Args:
        log_probs: ``T x V`` natural-log probabilities from the recogniser.
        grammar: the layout to constrain to.
        vocab: index -> token, matching the recogniser's output layer.
        blank_index: index of the CTC blank in ``vocab``.
        beam_width: beams retained per timestep.
        prune_log_gap: at each frame, ignore tokens whose log probability is
            more than this far below the frame maximum. Bounds the branching
            factor without measurably affecting accuracy.
        slot_bonus: optional per-slot log bonus, used for the colour prior.

    Returns:
        Accepted hypotheses as ``(tokens, log_prob, per_token_log_probs,
        repaired_slot_names)``, best first.
    """
    index: Mapping[str, int] = {tok: i for i, tok in enumerate(vocab)}
    beams: dict[tuple[tuple[str, ...], State], _Beam] = {
        ((), grammar.start): _Beam(lp_blank=0.0, best_path_lp=0.0)
    }
    # A plate cannot be longer than the sum of every slot's maximum repeat.
    max_emit = sum(s.max_repeat for s in grammar.slots)

    for frame in log_probs:
        frame_max = max(frame)
        cutoff = frame_max + prune_log_gap
        # Best non-blank token this frame -- the reference point for deciding
        # whether the grammar had to override the raw evidence.
        best_nonblank = ""
        best_nonblank_lp = NEG_INF
        for i, lp in enumerate(frame):
            if i != blank_index and lp > best_nonblank_lp:
                best_nonblank_lp = lp
                best_nonblank = vocab[i]

        nxt: dict[tuple[tuple[str, ...], State], _Beam] = {}
        blank_lp = frame[blank_index]

        for (prefix, state), b in beams.items():
            ptot = b.total()

            # 1. Emit blank: prefix and grammar state are unchanged.
            _merge(
                nxt,
                (prefix, state),
                blank=ptot + blank_lp,
                path_lp=b.best_path_lp + blank_lp,
                token_lps=b.token_lps,
                repaired=b.repaired,
            )

            # 2. Repeat the last token without a separating blank: CTC collapses
            #    this, so the prefix is unchanged.
            if prefix:
                li = index.get(prefix[-1])
                if li is not None and b.lp_nonblank != NEG_INF:
                    _merge(
                        nxt,
                        (prefix, state),
                        nonblank=b.lp_nonblank + frame[li],
                        path_lp=b.best_path_lp + frame[li],
                        token_lps=b.token_lps,
                        repaired=b.repaired,
                    )

            # 3. Extend, but only along grammar-legal transitions.
            if len(prefix) >= max_emit:
                continue
            for tok, states in grammar.transitions(state, prefix).items():
                ti = index.get(tok)
                if ti is None:
                    continue
                lp_tok = frame[ti]
                if lp_tok < cutoff:
                    continue
                # Emitting a token identical to the previous one requires an
                # intervening blank, so it may only extend the blank path.
                src = b.lp_blank if (prefix and tok == prefix[-1]) else ptot
                if src == NEG_INF:
                    continue
                for st2 in states:
                    slot = grammar.slots[st2[0]]
                    bonus = slot_bonus(slot.name, tok) if slot_bonus is not None else 0.0
                    if bonus == NEG_INF:
                        continue
                    repaired = b.repaired
                    if tok != best_nonblank:
                        repaired = repaired | {slot.name}
                    _merge(
                        nxt,
                        (prefix + (tok,), st2),
                        nonblank=src + lp_tok + bonus,
                        path_lp=b.best_path_lp + lp_tok + bonus,
                        token_lps=b.token_lps + (lp_tok,),
                        repaired=repaired,
                    )

        beams = dict(
            sorted(nxt.items(), key=lambda kv: kv[1].total(), reverse=True)[:beam_width]
        )

    results: list[tuple[tuple[str, ...], float, tuple[float, ...], frozenset[str]]] = []
    for (prefix, state), b in beams.items():
        if not prefix or not grammar.is_accepting(state):
            continue
        seg = grammar.segment(prefix)
        if seg is None:
            continue
        lp = b.total()
        # Length priors: applied once, at the end, over the realised field
        # lengths. Doing it here rather than incrementally keeps the search
        # itself a clean likelihood and makes the prior easy to audit.
        for slot in grammar.slots:
            lp += slot.length_bonus(len(seg.get(slot.name, [])))
        results.append((prefix, lp, b.token_lps, b.repaired))

    results.sort(key=lambda r: r[1], reverse=True)
    return results


# ---------------------------------------------------------------------------
# Colour prior
# ---------------------------------------------------------------------------

def colour_slot_bonus(
    evidence: ColourEvidence | None,
    *,
    weight: float = 1.0,
    floor: float = 1e-3,
) -> SlotBonus | None:
    """Build a slot bonus that scores the class letter against plate colour.

    Returns ``None`` when there is no usable colour evidence, so callers can
    pass the result straight through to the decoder.

    ``floor`` prevents a confident-but-wrong colour classifier from making the
    correct plate unreachable: the worst penalty any path can take is
    ``weight * log(floor)``, roughly -6.9 nats at the default, which strong
    glyph evidence can still overcome. Hard constraints here would be a
    mistake -- a red plate under sodium street lighting genuinely reads orange.
    """
    if evidence is None or not evidence.reliable or not evidence.posterior:
        return None

    # Restrict to the six legacy schemes: on an embossed plate colour carries no
    # ownership information, so scoring against it would be noise.
    legacy_mass = sum(
        evidence.posterior.get(c, 0.0) for c in spec.LEGACY_COLOUR_OWNERSHIP
    )
    if legacy_mass <= 0.0:
        return None

    def bonus(slot_name: str, token: str) -> float:
        if slot_name != "class":
            return 0.0
        own = spec.ownership_for_class_token(token)
        if own is Ownership.UNKNOWN:
            return 0.0
        mass = sum(
            evidence.posterior.get(colour, 0.0)
            for colour, owner in spec.LEGACY_COLOUR_OWNERSHIP.items()
            if owner is own
        )
        return weight * math.log(max(mass / legacy_mass, floor))

    return bonus


# ---------------------------------------------------------------------------
# High-level entry point
# ---------------------------------------------------------------------------

def decode(
    log_probs: Sequence[Sequence[float]],
    *,
    colour: ColourEvidence | None = None,
    system_hint: PlateSystem | None = None,
    vocab: Sequence[str] = spec.UNIFIED_VOCAB,
    blank_index: int = 0,
    beam_width: int = 16,
    colour_weight: float = 1.0,
    top_k: int = 3,
) -> list[ReadCandidate]:
    """Decode recogniser output into ranked, validated plate hypotheses.

    ``system_hint`` normally comes from the colour classifier: black-on-white
    means embossed, anything else means legacy. When it is absent both grammars
    are searched. Their alphabets are disjoint (Devanagari vs Latin), so the
    likelihoods are directly comparable and cross-system confusion is
    essentially impossible.
    """
    if system_hint is None and colour is not None and colour.reliable:
        best_colour, p = colour.best()
        if p >= 0.6:
            inferred = spec.system_for_colour(best_colour)
            if inferred is not PlateSystem.UNKNOWN:
                system_hint = inferred

    bonus = colour_slot_bonus(colour, weight=colour_weight)
    observed_colour = colour.best()[0] if colour is not None else PlateColour.UNKNOWN

    candidates: list[ReadCandidate] = []
    for grammar in grammars_for(system_hint):
        hyps = ctc_grammar_beam_search(
            log_probs,
            grammar,
            vocab=vocab,
            blank_index=blank_index,
            beam_width=beam_width,
            slot_bonus=bonus if grammar.system is PlateSystem.DEVANAGARI else None,
        )
        for tokens, lp, token_lps, repaired in hyps[:top_k]:
            # Geometric mean of per-token posteriors: an interpretable 0-1
            # confidence that does not shrink simply because a plate is long.
            mean_score = (
                math.exp(sum(token_lps) / len(token_lps)) if token_lps else 0.0
            )
            plate = plate_from_tokens(
                grammar,
                tokens,
                score=mean_score,
                token_scores=[math.exp(x) for x in token_lps],
                repaired_slots=repaired,
                observed_colour=observed_colour,
            )
            if not plate.is_valid:
                continue
            colour_contrib = 0.0
            if bonus is not None and plate.vehicle_class_deva:
                colour_contrib = bonus("class", plate.vehicle_class_deva)
            candidates.append(
                ReadCandidate(
                    plate=plate,
                    log_prob=lp,
                    colour_bonus=colour_contrib,
                    notes=tuple(f"repaired:{s}" for s in sorted(repaired)),
                )
            )

    candidates.sort(key=lambda c: c.log_prob, reverse=True)
    return candidates[:top_k]


def best(
    log_probs: Sequence[Sequence[float]], **kwargs
) -> ParsedPlate | None:
    """Convenience wrapper returning only the top hypothesis."""
    results = decode(log_probs, **kwargs)
    return results[0].plate if results else None


def greedy_decode(
    log_probs: Sequence[Sequence[float]],
    *,
    vocab: Sequence[str] = spec.UNIFIED_VOCAB,
    blank_index: int = 0,
) -> tuple[str, ...]:
    """Unconstrained greedy CTC decode: argmax, collapse repeats, drop blanks.

    This is the conventional baseline -- what an ANPR does before a regex gets
    involved. It is kept in the library rather than the tests because the
    evaluation harness reports constrained-vs-greedy side by side, and that
    delta is the headline number for whether the grammar is earning its place.
    """
    out: list[str] = []
    prev = -1
    for frame in log_probs:
        i = max(range(len(frame)), key=lambda j: frame[j])
        if i != prev and i != blank_index:
            out.append(vocab[i])
        prev = i
    return tuple(out)


__all__ = [
    "decode",
    "greedy_decode",
    "best",
    "ctc_grammar_beam_search",
    "colour_slot_bonus",
    "SlotBonus",
]
