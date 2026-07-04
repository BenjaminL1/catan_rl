"""Colonist.io game-log OCR + grammar → ordered event stream + winner.

The Stage-1 ``logparse`` slice (build brief §4). It crops the top-right game-log
panel of a decoded 1080p frame (:func:`crop_log`), OCRs it with easyocr
(:func:`ocr_log_crop`), and parses the raw noisy OCR lines into a typed,
ordered :class:`LogEvent` stream plus the per-game **winner**
(:func:`parse_log`).

Correctness constraints this module is built around (build brief §5):

- **Winner = the victory LOG line ONLY** (``§5.1``). The only signal that yields
  a winner is a ``"<player> won the game!"`` log line. The top-left ``"X - Y"``
  counter is a cumulative match tally (NOT the score); the centre
  ``"Victory!!!"`` banner is unreliable (showed on a POV-lost game) — neither is
  read here. A resign (``"<player> has left the game"``) or a video cutoff leaves
  the winner ``None`` (the game is dropped from the scoreboard but its board +
  openings may still be seeds). :func:`parse_log` NEVER infers a winner from a
  ``"built a Settlement (+1 VP)"`` precursor or a ``"gg"``.

- **OCR noise is real** (``§2`` / the committed ``ocr_*.txt`` fixtures). easyocr
  routinely renders a trailing ``!`` as ``l`` (``"Happy settlingl"``,
  ``"won the gamel"``) and mangles a handle mid-word (``"rayman|47"`` /
  ``"raymani47"``). The grammar therefore matches the *victory* verb
  (``"won the game"``) on a normalised line and resolves the winner **handle
  against the two known player handles** (fuzzy, since the handle glyphs are the
  noisiest token), rather than trusting the raw OCR handle verbatim.

- **Games are segmented by the reset marker** (``§3`` / ``§5.3``): the
  ``"Happy settling! … List of commands: /help"`` new-game placeholder starts a
  game; the victory line / end-screen ends it. :func:`parse_log` flags each reset
  as a :class:`LogEvent` of kind ``"game_reset"`` so ``segment.py`` can slice the
  corpus-wide event stream into per-game windows.

CPU-only. Never imports ``gui/`` or the training path (build brief §6). easyocr
is imported lazily inside :func:`ocr_log_crop` so the pure grammar
(:func:`parse_log`) — the unit-tested core — has no heavy/optional dependency.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:  # pragma: no cover - import only for type-checking
    import numpy as np

#: Top-right log-panel crop as ``(x0, y0, x1, y1)`` fractions of the frame
#: (build brief §2 / M0 spike ``crop_ocr.py``). Applied to a frame's native
#: geometry, so board/log crops are never anisotropically distorted.
LOG_CROP_FRAC: tuple[float, float, float, float] = (0.645, 0.0, 1.0, 0.3)

#: The kinds of parsed log event. ``"game_reset"`` marks the new-game placeholder
#: (segment boundary); ``"victory"`` marks the sole winner-bearing line. The rest
#: are the ordered gameplay events the log grammar recognises. ``"unknown"`` is a
#: line the grammar could not classify (kept for the §5.6 rejection/bias audit).
LogEventKind = Literal[
    "game_reset",
    "setup_settlement",
    "setup_road",
    "starting_resources",
    "roll",
    "got_resources",
    "built_settlement",
    "built_city",
    "built_road",
    "bought_dev",
    "used_dev",
    "moved_robber",
    "stole",
    "bank_trade",
    "resign",
    "victory",
    "unknown",
]


@dataclass(frozen=True, slots=True)
class LogEvent:
    """One ordered, classified game-log line.

    ``kind`` is the grammar classification; ``actor`` is the resolved player
    handle when the line names one (``None`` for actor-less lines like the reset
    placeholder); ``text`` is the normalised source line (kept verbatim for the
    §5.6 bias audit and for debugging OCR noise).
    """

    kind: LogEventKind
    actor: str | None
    text: str


@dataclass(frozen=True, slots=True)
class ParsedLog:
    """The result of parsing a batch of OCR log lines.

    ``events`` is the ordered classified stream (source order preserved).
    ``winner`` is the per-game winner read from the victory LOG line **only**
    (build brief §5.1) — one of the two supplied player handles, or ``None`` on a
    resign / cutoff / no-victory-line-seen game. ``winner`` is NEVER inferred from
    a non-victory line.
    """

    events: tuple[LogEvent, ...]
    winner: str | None


# --- OCR normalisation -------------------------------------------------------

#: Common easyocr glyph confusions, applied only for *matching* (never mutating
#: the retained ``text``). ``|`` inside a handle is a mis-split ``1``/``l``; a
#: trailing ``l`` on a ``!``-terminated word is the "settlingl"/"gamel" typo.
_WS = re.compile(r"\s+")


def _normalise(line: str) -> str:
    """Collapse whitespace + lowercase a raw OCR line for grammar matching.

    The retained :attr:`LogEvent.text` is the collapsed-but-cased line; matching
    is done against this lowercased form so ``"ThePhantom"`` and a noisier
    ``"thephantom"`` classify identically.
    """
    return _WS.sub(" ", line).strip()


def _tokens(line: str) -> list[str]:
    """Alphanumeric tokens of a lowercased line (handle-fuzzy comparison unit)."""
    return re.findall(r"[a-z0-9]+", line.lower())


def _handle_similarity(candidate: str, handle: str) -> float:
    """Cheap 0..1 similarity of an OCR token to a known handle (bigram overlap).

    Used ONLY to resolve which of the two *known* player handles a noisy OCR
    handle refers to — never to accept an arbitrary out-of-set name. A pure-Python
    Sorensen-Dice bigram coefficient (no external dependency), case-insensitive.
    """
    a = candidate.lower()
    b = handle.lower()
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    if len(a) < 2 or len(b) < 2:
        return 1.0 if a == b else 0.0
    ab = [a[i : i + 2] for i in range(len(a) - 1)]
    bb = [b[i : i + 2] for i in range(len(b) - 1)]
    bb_pool = list(bb)
    hits = 0
    for gram in ab:
        if gram in bb_pool:
            bb_pool.remove(gram)
            hits += 1
    return 2.0 * hits / (len(ab) + len(bb))


#: Minimum bigram similarity for an OCR handle token to bind to a known handle.
#: 0.5 accepts ``"raymani47"``/``"rayman|47"``→``"rayman147"`` while rejecting
#: the other player's handle.
_HANDLE_MATCH_THRESHOLD = 0.5


def _resolve_actor(line: str, handles: Sequence[str]) -> str | None:
    """Bind the leading OCR handle token(s) of a line to a known player handle.

    Colonist log lines lead with the acting player's handle. This scans the line
    for the token most similar to one of the two supplied ``handles`` (fuzzy, to
    absorb OCR handle noise) and returns that canonical handle, or ``None`` if no
    token clears :data:`_HANDLE_MATCH_THRESHOLD`.
    """
    best_handle: str | None = None
    best_score = _HANDLE_MATCH_THRESHOLD
    for token in _tokens(line):
        for handle in handles:
            score = _handle_similarity(token, handle)
            if score >= best_score:
                best_score = score
                best_handle = handle
    return best_handle


# --- grammar -----------------------------------------------------------------

#: The new-game reset placeholder (build brief §3 / §5.3). Matches the "Happy
#: settling(!/l)" marker OR the "List of commands: /help" partner line (either
#: alone reliably marks a reset — easyocr sometimes splits them across lines).
_RESET = re.compile(r"happy settling|list of commands")

#: The sole winner-bearing verb (build brief §5.1). ``game[!l]?`` absorbs the
#: "gamel" OCR typo and a dropped terminal punctuation; the trophy glyphs are not
#: OCR'd reliably and are not required.
_VICTORY = re.compile(r"won the game")

#: A resign / player-left line — implies the game ended without a 15-VP victory.
#: Per §5.1 this leaves ``winner=None`` (NOT the other player, since that inference
#: is exactly the confidently-wrong outcome the brief forbids).
_RESIGN = re.compile(r"has left the game|left the game|forfeit|resign")

# Ordered (verb-regex, kind) table. First hit on the normalised, lowercased line
# wins. Actor-less lines (reset) are handled before this. Ordering matters:
# "built a Settlement (+1 VP)" must classify as built_settlement, and the victory
# verb is checked separately (winner extraction), never as a generic event here.
_GRAMMAR: tuple[tuple[re.Pattern[str], LogEventKind], ...] = (
    (re.compile(r"placed a settlement"), "setup_settlement"),
    (re.compile(r"placed a road"), "setup_road"),
    (re.compile(r"received starting resources"), "starting_resources"),
    (re.compile(r"built a city"), "built_city"),
    (re.compile(r"built a settlement"), "built_settlement"),
    (re.compile(r"built a road"), "built_road"),
    (re.compile(r"\brolled\b"), "roll"),
    (re.compile(r"moved robber"), "moved_robber"),
    (re.compile(r"used (knight|monopoly|year of plenty|road building)"), "used_dev"),
    (re.compile(r"\bbought\b"), "bought_dev"),
    (re.compile(r"gave bank|took from bank"), "bank_trade"),
    (re.compile(r"\bstole\b|you stole"), "stole"),
    (re.compile(r"\bgot\b"), "got_resources"),
)


def parse_log(lines: Iterable[str], handles: Sequence[str]) -> ParsedLog:
    """Parse raw OCR log lines into an ordered event stream + the winner.

    ``lines`` are raw easyocr crop lines (comment lines beginning with ``#`` — the
    fixture provenance headers — are skipped). ``handles`` are the two known
    player handles (from the HUD seat row / setup lines); the winner and each
    event's ``actor`` are resolved **against these two handles only** (build brief
    §5.5, §14 — never an arbitrary OCR name).

    Winner rule (build brief §5.1): the winner is the handle on the FIRST
    ``"<player> won the game!"`` line (OCR-noise-tolerant), resolved to one of
    ``handles``. If no victory line is seen — resign, cutoff, or simply not
    sampled — the winner is ``None``. A victory line whose handle cannot be
    resolved to either known handle also yields ``None`` (fail closed rather than
    fabricate an outcome).
    """
    if len(handles) != 2 or len(set(handles)) != 2:
        raise ValueError(f"expected exactly two distinct player handles, got {handles!r}")

    events: list[LogEvent] = []
    winner: str | None = None

    for raw in lines:
        line = _normalise(raw)
        if not line or line.startswith("#"):
            continue
        low = line.lower()

        # Reset placeholder — actor-less segment boundary (checked first).
        if _RESET.search(low):
            events.append(LogEvent(kind="game_reset", actor=None, text=line))
            continue

        # Victory line — the ONLY winner signal (§5.1). Record the event AND, on
        # the first resolvable victory line, latch the winner.
        if _VICTORY.search(low):
            actor = _resolve_actor(line, handles)
            events.append(LogEvent(kind="victory", actor=actor, text=line))
            if winner is None and actor is not None:
                winner = actor
            continue

        # Resign / left — ends the game with NO winner (§5.1: never infer the
        # other player won).
        if _RESIGN.search(low):
            actor = _resolve_actor(line, handles)
            events.append(LogEvent(kind="resign", actor=actor, text=line))
            continue

        kind: LogEventKind = "unknown"
        for pattern, candidate_kind in _GRAMMAR:
            if pattern.search(low):
                kind = candidate_kind
                break
        actor = _resolve_actor(line, handles)
        events.append(LogEvent(kind=kind, actor=actor, text=line))

    return ParsedLog(events=tuple(events), winner=winner)


# --- OCR (optional, lazy easyocr) -------------------------------------------


def crop_log(frame: np.ndarray) -> np.ndarray:
    """Crop the top-right log panel from a native-geometry RGB frame.

    ``frame`` is an ``(H, W, 3)`` ``uint8`` RGB array (an
    :class:`~catan_rl.human_data.ingest.DecodedFrame` ``.frame``). Returns the
    :data:`LOG_CROP_FRAC` sub-image (a view/slice — no copy, no disk write, brief
    §5.12).
    """
    h, w = frame.shape[:2]
    x0f, y0f, x1f, y1f = LOG_CROP_FRAC
    x0, x1 = int(x0f * w), int(x1f * w)
    y0, y1 = int(y0f * h), int(y1f * h)
    return frame[y0:y1, x0:x1]


def ocr_log_crop(crop: np.ndarray) -> list[str]:
    """OCR a log crop into raw text lines (lazy easyocr, CPU-only).

    easyocr is imported *inside* this function so the pure grammar
    (:func:`parse_log`) — the unit-tested core — has no dependency on the heavy,
    optional OCR runtime. ``gpu=False`` pins it to CPU (build brief: CPU-only). A
    single shared reader is cached across calls.
    """
    reader = _easyocr_reader()
    result = reader.readtext(crop, detail=0, paragraph=False)
    return [str(line) for line in result]


class _EasyOcrReader:
    """Structural protocol for the one easyocr method we call (keeps mypy strict)."""

    def readtext(
        self, image: np.ndarray, *, detail: int, paragraph: bool
    ) -> list[object]:  # pragma: no cover - satisfied by the real easyocr.Reader
        raise NotImplementedError


_READER: _EasyOcrReader | None = None


def _easyocr_reader() -> _EasyOcrReader:
    """Lazily build + cache a CPU easyocr reader (English)."""
    global _READER
    if _READER is None:
        import easyocr

        reader: _EasyOcrReader = easyocr.Reader(["en"], gpu=False, verbose=False)
        _READER = reader
    return _READER
