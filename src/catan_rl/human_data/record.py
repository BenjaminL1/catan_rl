"""The frozen ``GameRecord`` data contract — one JSON record per parsed game.

This is the *only* contract between the video-parsing pipeline and downstream
consumers (the opening scoreboard + the human-seed loader), so it is frozen
**first**, before any module code (build brief §3). One record per game, one per
line in the emitted JSONL.

Conventions baked in here (build brief §5, §6):

- ``schema_version`` mirrors ``conformance.recorder.CONFORMANCE_SCHEMA_VERSION``
  (both are ``1``); bump on any breaking shape change.
- All board coordinates are engine integer IDs (19 hex / 54 vertex / 72 edge).
- **Resources are string literals**, never an enum — the only stable resource
  ordering in the codebase is ``RESOURCES_CW`` at the RL boundary; the engine has
  3+ inconsistent ad-hoc orderings. Permitted literals: ``WOOD``, ``BRICK``,
  ``WHEAT``, ``ORE``, ``SHEEP``, ``DESERT`` (desert hexes carry ``number=None``).
- ``episode_source`` is load-bearing: eval / anchor consumers must see **only**
  ``"natural"`` episodes; ``"human_seed"`` episodes are seeds and must never
  re-import the human cap.
- ``opponent_strength`` is a **required** field (never null); games whose strength
  can't be established from the reference are excluded from the scoreboard (they
  may still be seeds).
- ``rejection_reason`` is kept on rejected records for the rejection-bias audit.
- Ports are **omitted in v1** (never extracted in any spike).

.. warning::

   **``opponent_strength`` is a MATCH-strength proxy, NOT the adversary's rank
   (review finding §1).** :func:`derive_opponent_strength` reads the committed
   strength manifest (``data/human/strength_manifest.json``), which measures the
   **channel owner** ThePhantom's world rank off HIS leaderboard row (see
   ``scripts/build_strength_manifest.py`` — ``_phantom_rank_from_toks`` /
   ``_is_phantom`` match only ``"thephantom"``; ``source="ranked_rank"`` carries
   HIS rank, ``source="tournament"`` means HE entered a tournament). Colonist
   ranked matchmaking pairs by rating, so a high ThePhantom rank is only a **weak
   proxy** for the *opponent's* strength — a rank-10 ThePhantom vs a rank-9000
   beginner is still labelled ``tier="high"``. Because ThePhantom is near-always
   high-rank, the ``ranked_rank`` signal is near-constant on the agent seat and
   carries almost no information about the adversary, so pooling all 204 ``high``
   games into one "vs humans" calibration number is exactly the mixed-opponent
   pooling the build brief §5.5 forbids.

   Consequences baked into this module:

   - The **only defensibly strong-vs-strong bucket** is the ``tournament`` subset
     (both seats entered a 1v1 tournament). It is exposed as the *primary*
     scoreboard predicate :meth:`GameRecord.is_strong_opponent_scoreboard_eligible`.
   - :meth:`GameRecord.is_scoreboard_eligible` remains the broad
     "ThePhantom-high-or-tournament" predicate, but its result is
     **opponent-uncontrolled** for the ``rank_badge`` games; a downstream
     scoreboard builder MUST split its ``n`` by ``opponent_strength.source`` and
     report the ``tournament`` (strong-vs-strong) games separately from the
     ``rank_badge`` (ThePhantom-high, opponent-uncontrolled) games — it must never
     collapse the two provenances into a single vs-humans number.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, get_args

#: Schema version of the ``GameRecord`` contract.
#:
#: v1 mirrored ``catan_rl.conformance.recorder.CONFORMANCE_SCHEMA_VERSION``.
#: **v2** adds *provenance orientation-binding* (the board-orientation firewall):
#: ``provenance.board_desert_hex`` and ``provenance.openings_desert_hex`` are now
#: **required** and :meth:`GameRecord.validate` rejects a record whose two
#: artifacts were locked under different D6 orientations. This is the only gate
#: that catches the desert=17/desert=11 weld bug — a D6 flip preserves the
#: resource/number multisets, so every other structural gate passes the wrong
#: record. The conformance recorder schema is unrelated and stays at its own
#: version.
SCHEMA_VERSION = 2

#: Resource string literals permitted on a hex (build brief §5.8). There is no
#: ``RESOURCES`` enum in ``engine/``; these are the only stable values.
RESOURCE_LITERALS: frozenset[str] = frozenset({"WOOD", "BRICK", "WHEAT", "ORE", "SHEEP", "DESERT"})

#: Standard-board structural counts (mirrors :mod:`catan_rl.human_data.topology`,
#: kept local so :meth:`GameRecord.validate` stays a pure value check with no
#: engine / topology import — scope-lock, build brief §6).
NUM_HEXES = 19
NUM_VERTICES = 54
NUM_EDGES = 72

#: Minimum capture resolution the record firewall admits (brief §5.10 / FIX-5).
#: Sub-1080p footage OCRs number tokens and log glyphs to garbage, so a record
#: sourced below this is *confidently* wrong (not merely noisy). Mirrors
#: :data:`catan_rl.human_data.orientation.MIN_RESOLUTION`, kept local so
#: :meth:`GameRecord.validate` stays a pure value check with no orientation
#: import (the same scope-lock pattern as :data:`NUM_HEXES`). The batch path also
#: gates on it in ``assert_scale_up_orientation_gates``, but that gate is
#: bypassable by a single-game / resumed-shard path that builds records directly —
#: so the contract, "the single firewall between the noisy CV/OCR pipeline and the
#: RL stack", must enforce it too (review finding: sub-1080p provenance).
MIN_RESOLUTION = 1080

#: Number tokens a non-desert hex may carry (2..12 minus the robber-only 7).
VALID_HEX_NUMBERS: frozenset[int] = frozenset(set(range(2, 13)) - {7})

#: Legal 2d6 roll outcomes for ``dice_log`` (the load-bearing dice-luck covariate,
#: brief §5.4). Unlike a hex token, the **7 IS a legal roll** (it triggers the
#: robber), so the range is the full 2..12 inclusive — the 7 is NOT excluded here.
VALID_DICE_VALUES: frozenset[int] = frozenset(range(2, 13))

#: Standard 19-tile board resource multiset (engine ``_random_resource_type_list``:
#: 1 DESERT, 3 ORE, 3 BRICK, 4 WHEAT, 4 WOOD, 4 SHEEP). A board-CV resource
#: misclassification yields a structurally-valid-but-wrong board with the wrong
#: counts (brief §5.6) — the multiset gate is the cheapest CV-relevant defense.
#: Kept a local constant (no engine import) to preserve the scope-lock (brief §6).
#:
#: **NECESSARY, NOT SUFFICIENT (review findings §1/§2) — do NOT rely on this gate
#: as the CV-correctness net**, exactly the way the glyph anchor in
#: :mod:`catan_rl.human_data.orientation` is documented as necessary-not-sufficient.
#: The gate only fires when the resource *counts* deviate; it is structurally
#: blind to a multiset-*preserving* swap. Two concrete Stage-2 failure modes it
#: cannot catch (they land here, in the contract, undetected):
#:
#: 1. A mis-tuned per-hex hue classifier that lands game-B's WOOD in the SHEEP bin
#:    AND its SHEEP in the WOOD bin: the 4/4 counts are preserved, so the board is
#:    confidently wrong yet the gate passes. The only CV classifier committed so
#:    far (``scripts/dev/human_data_spikes/board_cv/spike2.py`` ``classify()``) is a
#:    FIXED HSV-threshold table calibrated on one render — its docstring claims
#:    "palette-agnostic" but the code is hardcoded thresholds, contradicting brief
#:    §5.13 ("per-game palette must be generalized") and §2 ("median hex colour,
#:    calibrated per-frame, not hardcoded"). The production ``board_cv.py`` MUST
#:    generalize it: cluster the 19 sampled hex colours PER GAME and assign
#:    cluster→resource from the frame's OWN samples, then corroborate each hex with
#:    an orientation-independent signal (number-token adjacency / port hints) so a
#:    multiset-preserving swap surfaces. Add a per-game calibration-residual gate.
#: 2. The gate becomes **tautological** if ``board_cv.py`` assigns resources by
#:    forcing the standard multiset (Hungarian cluster→resource assignment): the CV
#:    has already forced the counts to match, so this gate can NEVER fire for the
#:    resource dimension — it silently degrades to a no-op. The contract to pick
#:    (document the choice in ``board_cv.py`` when built): (a) PREFERRED — classify
#:    each hex INDEPENDENTLY (raw per-hex label) so this gate is a genuine
#:    cross-check that can fail; or (b) if multiset-forced assignment is used, this
#:    gate must be REPLACED for the resource dimension by an INDEPENDENT signal
#:    (per-hex pip-count vs OCR-number agreement; cross-frame stability of each
#:    hex's raw colour cluster) rather than the now-tautological count check.
STANDARD_RESOURCE_COUNTS: dict[str, int] = {
    "DESERT": 1,
    "ORE": 3,
    "BRICK": 3,
    "WHEAT": 4,
    "WOOD": 4,
    "SHEEP": 4,
}

#: Standard 18-token number bag (engine ``SPIRAL_CHIP_SEQUENCE``): one each of 2
#: and 12, two each of 3..6 and 8..11 (no 7 — that is the robber).
STANDARD_NUMBER_BAG: dict[int, int] = {
    2: 1,
    3: 2,
    4: 2,
    5: 2,
    6: 2,
    8: 2,
    9: 2,
    10: 2,
    11: 2,
    12: 1,
}


def _as_int(value: Any, label: str) -> int:
    """Return ``value`` iff it is a true (non-bool) ``int``; else raise.

    The CV/OCR upstream stages are float-producing (pip-count number reader,
    affine-snap vertex/edge mapper). Coerce-and-range-check (``int(8.5) == 8``)
    silently admits out-of-domain floats that then survive into the emitted JSONL
    row (brief §3 guarantees engine *integer* IDs). Reject non-integers up front;
    ``bool`` is an ``int`` subclass and must not masquerade as a 0/1 token.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an int, got {value!r} ({type(value).__name__})")
    return value


#: The 1v1 Colonist ruleset is hard-locked (CLAUDE.md / brief §1): exactly two
#: players, 15 VP to win. A future 4-player generalization must consciously break
#: the schema — it can never round-trip silently.
RULESET_1V1: dict[str, int] = {"num_players": 2, "win_vp": 15}

#: Provenance keys that bind each parsed artifact to the D6 board orientation it
#: was locked under (schema v2). The board (hex resources/numbers) and the
#: openings (vertex/edge IDs) are produced by *separate* CV stages; if they are
#: snapped under different orientations the record is confidently wrong but
#: passes every multiset/structural gate (a D6 flip permutes IDs while preserving
#: the resource/number multisets). These two ints (the engine ``hex_id`` that the
#: stage placed the desert at) must agree, and :meth:`GameRecord.validate`
#: rejects on mismatch. This is the cross-orientation firewall.
PROVENANCE_BOARD_DESERT = "board_desert_hex"
PROVENANCE_OPENINGS_DESERT = "openings_desert_hex"

#: Provenance flag (step6 §3.1, harvest-blocking placement-order contract): whether
#: the record's :class:`PlayerOpening` tuples are in established LOG PLACEMENT ORDER
#: (``settlements[1]`` = the 2nd/resource-granting settlement). ``True`` when each
#: player's 1st-vs-2nd settlement is disambiguated (see
#: :func:`catan_rl.human_data.orientation.establish_placement_order`). Absent or
#: ``False`` ⟹ **ORDER-UNESTABLISHED**: the tuple order is arbitrary (CV-detection
#: order), so the record is EVAL-excluded — :meth:`GameRecord.is_scoreboard_eligible`
#: requires this flag — but remains :meth:`GameRecord.is_seed_eligible` (the seed
#: loader samples the grant hypothesis per episode). It is **not** enforced by
#: :meth:`GameRecord.validate` (a missing flag is a valid, order-unestablished
#: record), so pre-contract records stay loadable and default to seed-only.
#:
#: **Two establishing regimes (audit Decision 1, ``require_log_ordinal``).** The
#: harvest / vlm-spike log-gates support an opt-in: by DEFAULT the flag needs BOTH
#: the granted-glyph adjacency (VERTEX-side) AND the log setup-event ordinal (the
#: grant follows each player's 2nd settlement) — the two-signal path. Under the
#: explicit opt-in (``require_log_ordinal=False``) the flag may be established by
#: the granted-glyph adjacency ALONE (``glyph_only``): re-OCR duplication makes the
#: log ordinal unavailable on real footage, but the glyph anchor pins the granting
#: settlement uniquely (or fails closed on any grant collision/ambiguity → flag
#: ``False``). WHICH signal established the order is recorded additively in
#: :data:`PROVENANCE_ORDER_SOURCE`; the flag's meaning (established vs not) is
#: identical in both regimes, and eligibility keys on the flag alone.
PROVENANCE_PLACEMENT_ORDER_ESTABLISHED = "placement_order_established"

#: Provenance key (audit Decision 1): WHICH signal established the placement order —
#: ``"log+glyph"`` (both the log setup-event ordinal and the granted-glyph adjacency
#: agreed — the two-signal default), ``"glyph_only"`` (the granted-glyph adjacency
#: alone, under the ``require_log_ordinal=False`` opt-in — the log ordinal was
#: unavailable or unconfirmed but the flag was NOT downgraded), or ``None`` (order
#: UNESTABLISHED — no signal pinned it). Purely INFORMATIONAL provenance: it is
#: additive, never renames an existing key, is NOT enforced by
#: :meth:`GameRecord.validate`, and does NOT gate eligibility —
#: :meth:`GameRecord.is_scoreboard_eligible` keys on
#: :data:`PROVENANCE_PLACEMENT_ORDER_ESTABLISHED` alone. It exists so a downstream
#: scoreboard builder can split / audit its ``n`` by ordering provenance.
PROVENANCE_ORDER_SOURCE = "order_source"

#: Provenance flag: whether the per-roll DICE VALUES were readable for this game.
#: Colonist renders the rolled dice as FACE GLYPHS, not text — the log line OCRs as
#: ``"ThePhantom rolled"`` with no number — so on real video footage the roll VALUES
#: are unreadable even though the roll EVENTS parse fine. A game whose ``roll`` events
#: exist but whose values never OCR'd carries ``False`` here and an empty ``dice_log``.
#: This is an honest "unread", NOT a fabricated luck series, and it is the ONLY case in
#: which :meth:`GameRecord.validate` permits an empty ``dice_log`` on a completed
#: (winner-set) game — a completed game with NO roll events at all is still a parse
#: failure. Consequence: the §5.4 dice-luck covariate is unavailable for such games
#: (reading it back needs dice-face glyph classification, not OCR).
PROVENANCE_DICE_VALUES_READABLE = "dice_values_readable"

#: ``episode_source`` values. ``"natural"`` = a real parsed game (scoreboard
#: + eval/anchor eligible). ``"human_seed"`` = an opening used to seed exploration
#: (never re-imports the human cap; eval/anchor must filter these out).
EpisodeSource = Literal["natural", "human_seed"]

#: Coarse opponent-strength tier. The scoreboard never pools across mixed tiers.
StrengthTier = Literal["high", "unknown"]

#: How opponent strength was established (build brief §5.5). Values:
#:
#: - ``"rank_badge"`` — an objective on-screen rank/elo badge was read. This is
#:   the reconciled name for the committed strength manifest's
#:   (``data/human/strength_manifest.json``) ``source="ranked_rank"`` entries
#:   (top-N world rank read off a leaderboard frame). ``segment.py`` maps
#:   manifest ``ranked_rank`` → ``rank_badge`` when it derives
#:   :class:`OpponentStrength` from the manifest.
#: - ``"tournament"`` — the game is a 1v1 tournament match (the manifest's
#:   ``source="tournament"``); the opponent is high by tournament participation
#:   rather than a read rank badge. Carried through 1:1 from the manifest.
#: - ``"known_window"`` — the game falls in a known high-rank window of the
#:   channel. **PLACEHOLDER — no backing window committed (review finding §1);
#:   the committed manifest is now the source of truth and uses ``ranked_rank`` /
#:   ``tournament`` / ``none``, not a window.** Retained in the literal for
#:   backwards compatibility (never remove an existing source value), but the
#:   manifest-driven ``segment.py`` never emits it. A hand-set
#:   ``tier="high", source="known_window"`` remains unfalsifiable, so it must not
#:   feed the scoreboard's mixed-strength-pooling filter (§5.5).
#:
#: The manifest's third source, ``"none"`` (strength unknown), is NOT a valid
#: :class:`OpponentStrength.source`: an unknown-strength game gets
#: ``tier="unknown"`` (with whatever source the record carries — never a fake
#: ``high``), and an ``excluded`` manifest video never becomes a record at all.
#:
#: **``segment.py`` MUST key the excluded-vs-high decision on the manifest's
#: ``strength`` field, NOT its ``source`` field.** The committed manifest holds
#: three distinct ``strength`` values — ``high`` (204), ``unknown`` (574),
#: ``excluded`` (36) — and ``excluded`` and ``unknown`` are BOTH distinct from
#: ``high`` (they map away from a record / to ``tier="unknown"``, never to a high
#: record). Crucially, ``source`` alone is ambiguous: 36 ``excluded`` videos
#: carry ``source="ranked_rank"`` (a real rank badge read, but rank
#: > ``rank_high_max=200``, so excluded). A naive ``ranked_rank → rank_badge,
#: tier="high"`` mapping keyed on ``source`` would turn a genuinely rank-200+
#: opponent into a confidently-wrong top-tier record. So the derivation gates on
#: ``strength`` first (``high`` → build a high record with the source mapped;
#: ``unknown`` → ``tier="unknown"``; ``excluded`` → emit NO record) and only then
#: maps ``source``. This is enforced where ``segment.py`` is built, not in this
#: pure value contract (the record carries no manifest linkage to check).
StrengthSource = Literal["rank_badge", "known_window", "tournament"]


@dataclass(frozen=True, slots=True)
class OpponentStrength:
    """Objective opponent-strength signal (build brief §5.5).

    Never a handle guess. ``confidence`` is a coarse 0..1 self-assessment of the
    signal, not a calibrated probability.

    **Provenance (review finding §1, now resolved for the manifest sources):**
    ``segment.py`` derives this from the committed strength manifest
    (``data/human/strength_manifest.json``), mapping manifest ``ranked_rank`` →
    ``source="rank_badge"`` and ``tournament`` → ``source="tournament"`` (both
    ``tier="high"``), and manifest ``none`` → ``tier="unknown"``. The dataclass
    still enforces the label's *shape* only; ``source="known_window"`` remains a
    placeholder with no committed backing window (see :data:`StrengthSource`) and
    the manifest-driven path never emits it.
    """

    tier: StrengthTier
    source: StrengthSource
    confidence: float

    def __post_init__(self) -> None:
        # ``Literal`` is mypy-only — enforce the declared sets at runtime so a
        # mislabelled tier/source can't deserialize silently (finding §1).
        if self.tier not in get_args(StrengthTier):
            raise ValueError(f"OpponentStrength.tier {self.tier!r} not in {get_args(StrengthTier)}")
        if self.source not in get_args(StrengthSource):
            raise ValueError(
                f"OpponentStrength.source {self.source!r} not in {get_args(StrengthSource)}"
            )
        if not 0.0 <= float(self.confidence) <= 1.0:
            raise ValueError(f"OpponentStrength.confidence {self.confidence!r} not in [0, 1]")


#: Confidence carried on a derived :class:`OpponentStrength`, by manifest source.
#: A ``ranked_rank`` high was OCR-rank-verified off a leaderboard frame (high
#: confidence); a ``tournament`` high was labelled from a title keyword with NO
#: leaderboard-frame confirmation (a false-high on the tournament path is
#: unrecoverable — build_strength_manifest.py), so it carries a lower confidence.
#: ``unknown`` strength carries no signal.
_DERIVED_CONFIDENCE: dict[str, float] = {"ranked_rank": 0.95, "tournament": 0.8, "none": 0.0}

#: The manifest ``source`` → :data:`StrengthSource` mapping for a HIGH record.
#: ``ranked_rank`` (a read rank badge ≤ ``rank_high_max``) reconciles to
#: ``rank_badge``; ``tournament`` is carried through 1:1.
_MANIFEST_SOURCE_TO_STRENGTH_SOURCE: dict[str, StrengthSource] = {
    "ranked_rank": "rank_badge",
    "tournament": "tournament",
}


def derive_opponent_strength(manifest_entry: dict[str, Any]) -> OpponentStrength | None:
    """Derive the record's :class:`OpponentStrength` from a strength-manifest video
    entry (``data/human/strength_manifest.json`` — see
    ``scripts/build_strength_manifest.py`` for its shape).

    .. warning::

       **This encodes ThePhantom's-MATCH-strength, not the adversary's rank
       (review finding §1).** The manifest measures the channel owner's
       (ThePhantom's) own leaderboard rank / tournament entry, NOT the opponent's.
       A ``tier="high"`` label therefore means "ThePhantom was high-rank (or in a
       tournament) for this match", which under Colonist rating-based matchmaking
       is only a weak proxy for the opponent being strong. The ``tournament``
       source is the only bucket where both seats are genuinely strong; a
       downstream scoreboard must not pool the ``rank_badge`` (opponent-uncontrolled)
       games into a single vs-humans calibration number. See the module docstring
       and :meth:`GameRecord.is_strong_opponent_scoreboard_eligible`.

    **This is THE high/unknown/excluded gate**, moved out of the §5.5 docstring
    prose into a pure, testable helper (review finding §1). It gates on the
    manifest ``strength`` field FIRST and never on ``source`` — the trap being
    that 36 ``excluded`` videos carry ``source="ranked_rank"`` (a real rank badge,
    but rank > ``rank_high_max=200``), so a naive ``ranked_rank → rank_badge,
    tier="high"`` mapping keyed on ``source`` would turn a genuinely rank-200+
    opponent into a confidently-wrong top-tier record. Returns:

    - ``strength="high"`` → an ``OpponentStrength(tier="high", ...)`` with
      ``source`` mapped (``ranked_rank`` → ``rank_badge``; ``tournament`` →
      ``tournament``).
    - ``strength="unknown"`` → ``OpponentStrength(tier="unknown", source="rank_badge",
      confidence=0.0)`` — an unknown-strength game (still a possible seed; never a
      scoreboard game). ``source`` is a placeholder here: :meth:`is_scoreboard_eligible`
      gates on ``tier`` first, so an ``unknown`` record is scoreboard-ineligible
      regardless of its ``source``.
    - ``strength="excluded"`` → ``None`` — an excluded video NEVER becomes a
      record (the rank-288 case must return ``None``, not a high record).

    Pure value logic (no manifest I/O, no engine/topology import) — the caller
    (``segment.py``, a later stage) reads the manifest and passes one entry.
    """
    strength = manifest_entry.get("strength")
    source = manifest_entry.get("source")
    if strength == "excluded":
        return None
    if strength == "high":
        mapped = (
            _MANIFEST_SOURCE_TO_STRENGTH_SOURCE.get(source) if isinstance(source, str) else None
        )
        if mapped is None:
            raise ValueError(
                f"manifest strength='high' with unmappable source {source!r} — a high video must "
                f"carry a rank/tournament source, not {source!r} (keys on strength then source)"
            )
        confidence = _DERIVED_CONFIDENCE.get(source, 0.8) if isinstance(source, str) else 0.8
        return OpponentStrength(tier="high", source=mapped, confidence=confidence)
    if strength == "unknown":
        # tier drives eligibility; source is an unused placeholder for unknown.
        return OpponentStrength(tier="unknown", source="rank_badge", confidence=0.0)
    raise ValueError(
        f"manifest strength {strength!r} is not one of 'high' / 'unknown' / 'excluded'"
    )


@dataclass(frozen=True, slots=True)
class PlayerOpening:
    """One player's snake-draft opening: 2 settlements + 2 roads as engine IDs.

    **PLACEMENT-ORDER CONTRACT (harvest-blocking, step6 §3.1).** The
    ``settlements`` and ``roads`` tuples are in **LOG PLACEMENT ORDER**, sourced
    from the setup-event sequence — NOT the (order-blind) CV detection order:

    - ``settlements[0]`` = the player's **first-placed** setup settlement;
    - ``settlements[1]`` = the player's **second-placed** settlement, which is the
      **resource-granting** one (Colonist grants starting resources from the 2nd
      settlement; the bridge grants from ``settlements[1]``);
    - ``roads[i]`` is the setup road placed immediately after ``settlements[i]``
      (each setup road is incident to the settlement just placed), so ``roads[0]``
      touches ``settlements[0]`` and ``roads[1]`` touches ``settlements[1]``.

    **How the order is established (the disambiguation rule).** The openings CV
    reads a single order-blind post-setup frame, so the raw tuple order carries no
    placement information. The Colonist game log's setup-event sequence
    (``<player> placed a Settlement / Road`` + ``<player> received starting
    resources`` lines, in draft order) is the only order signal: the
    ``"received starting resources"`` line's position identifies which of the
    player's two settlement placements it followed — that placement is the 2nd
    (granting) one (the grant event follows the 2nd settlement). The specific
    *vertex* is then pinned by the glyph anchor: the granted-card multiset (read
    from the grant line's card icons) equals the adjacent-resource multiset of
    exactly one of the two opening settlements, and that vertex is
    ``settlements[1]``. Both signals are required; when either is missing or
    ambiguous (missing setup lines, or a granted multiset that matches neither or
    *both* settlements) the record is marked **ORDER-UNESTABLISHED** via
    ``provenance[`` :data:`PROVENANCE_PLACEMENT_ORDER_ESTABLISHED` ``] = False``.
    :func:`catan_rl.human_data.orientation.establish_placement_order` implements
    the rule; :meth:`GameRecord.is_scoreboard_eligible` additionally requires the
    order to be established (an order-unestablished record is EVAL-excluded but
    still :meth:`~GameRecord.is_seed_eligible` — the seed loader samples the grant
    hypothesis per episode rather than trusting an arbitrary order).
    """

    settlements: tuple[int, ...]
    roads: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class GameRecord:
    """One parsed 1v1 Colonist.io game (build brief §3).

    Frozen + ``slots`` so records are hashable, immutable, and cheap. Serialize
    with :meth:`to_dict` / :meth:`to_json_line`; deserialize with
    :meth:`from_dict` / :meth:`from_json_line` (round-trip stable).
    """

    video_id: str
    game_index: int
    players: dict[str, str]
    opponent_strength: OpponentStrength
    ruleset: dict[str, int]
    # board.hexes: list of {"hex_id": int, "resource": <literal>, "number": int|None}.
    # board.ports is intentionally absent in v1 (never extracted; brief §5.9).
    hexes: tuple[dict[str, Any], ...]
    draft_order: tuple[str, ...]
    openings: dict[str, PlayerOpening]
    dice_log: tuple[int, ...]
    winner: str | None
    episode_source: EpisodeSource
    passed_crosscheck: bool
    provenance: dict[str, Any]
    rejection_reason: str | None = None
    schema_version: int = field(default=SCHEMA_VERSION)

    def __post_init__(self) -> None:
        # The contract is the single firewall between the noisy CV/OCR pipeline
        # and the RL stack (brief §5). Validate on construction so the pipeline
        # cannot bypass the gate — every code path that makes a ``GameRecord``
        # (including :meth:`from_dict`) runs this. A confidently-wrong record
        # raises rather than silently polluting the scoreboard / seeds.
        self.validate()

    def validate(self) -> None:
        """Assert every structural invariant of the 1v1 contract.

        Pure value checks (no engine / topology import — scope-lock, brief §6).
        Raises :class:`ValueError` on any violation. This is the cheap structural
        pre-gate; brief §5.7 still requires the engine opening-legality re-check
        at seed-load time — this does not replace it.

        Invariants enforced:

        - **Players:** exactly the keys ``{"agent", "opponent"}`` with two
          distinct non-empty handle strings (no name-OCR collision).
        - **Resources / numbers:** every hex resource in :data:`RESOURCE_LITERALS`;
          ``DESERT`` ⟺ ``number is None``, non-desert ⟺ ``number`` in 2..12 \\ {7};
          all hex numbers / IDs are true ``int`` s (no float survives the snap).
        - **Standard-board multisets:** non-desert resource counts ==
          :data:`STANDARD_RESOURCE_COUNTS` and the number-token bag ==
          :data:`STANDARD_NUMBER_BAG` — a CV resource/number misclassification is
          a confidently-wrong board (brief §5.6) and is rejected here.
        - **Board IDs:** ``hex_id`` multiset is exactly ``{0..18}``; all opening
          vertices in ``0..53``, edges in ``0..71``; exactly 2 *distinct*
          settlements + 2 *distinct* roads per player; settlement vertices are
          disjoint across the two players (no double-snap / shared vertex).
        - **Openings completeness:** exactly the two player handles have an
          opening (both present, no extras).
        - **draft_order:** a length-4 snake ``[a, b, b, a]`` with both player
          handles appearing exactly twice (``a != b``).
        - **Ruleset:** exactly ``{num_players: 2, win_vp: 15}`` (1v1-locked).
        - **Literals at runtime:** ``episode_source`` in its declared set.
        - **dice_log:** every roll is a true ``int`` (no float / bool) in
          ``2..12`` **inclusive** — the load-bearing dice-luck covariate (brief
          §5.4); unlike a hex token the 7 IS a legal roll outcome here. An empty
          ``dice_log`` is permitted only when ``winner is None`` (resign /
          cutoff); a completed game with no rolls is a parse failure.
        - **Provenance orientation-binding (schema v2):** ``provenance`` must
          carry both :data:`PROVENANCE_BOARD_DESERT` and
          :data:`PROVENANCE_OPENINGS_DESERT` (the engine ``hex_id`` each CV stage
          placed the desert at, ints in ``0..18``); ``board_desert_hex`` must
          equal the board's **own** unique ``DESERT`` ``hex_id`` (the free anchor
          that rejects an internally-contradictory record and ties the firewall
          to the board), and the two stamps must be **equal** — the board and the
          openings must be locked under the same D6 orientation. This is the
          cross-orientation firewall (see below); it is the only gate that
          catches a welded board/openings record because a D6 flip preserves the
          resource/number multisets.
        - **Provenance resolution (FIX-5):** if ``provenance`` carries a
          ``resolution``, it must be ``>= MIN_RESOLUTION`` (1080). A sub-1080p
          source OCRs number tokens and log glyphs to garbage — a confidently-wrong
          record. The batch path also gates this in
          ``assert_scale_up_orientation_gates``, but a single-game / resumed-shard
          path can bypass that gate, so the contract enforces it too.
        - **Cross-field truth table** (brief §5.6 / §5.7 — see below).

        **Truth table the contract enforces and every consumer must honour:**

        - ``rejection_reason is not None`` ⟹ ``passed_crosscheck is False``. A
          rejected record is **scoreboard-ineligible by definition** but still
          emits its parsed features for the §5.6 rejection-bias audit.
        - **scoreboard-eligible** ⟺ ``winner is not None`` AND ``passed_crosscheck``
          AND ``opponent_strength.tier == "high"`` AND
          ``opponent_strength.source in {"rank_badge", "tournament"}`` AND
          ``rejection_reason is None`` AND
          ``provenance[placement_order_established] is True`` (the §5.4 filter +
          the step6 §3.1 placement-order clause — an order-unestablished record is
          EVAL-excluded, seed-only). Not asserted (eligibility
          is a *property*, not every record must be eligible), but it is
          implemented as the exported :meth:`is_scoreboard_eligible` — **that
          method is the single source of truth**; the scoreboard builder and the
          rejection-bias audit must both call it so the mixed-strength-pooling
          filter (§5.5) can't drift. The ``source`` clause excludes the
          unfalsifiable ``known_window`` placeholder (see :data:`StrengthSource`).
        - **seed-eligible** ⟺ ``passed_crosscheck`` (brief §5.7), exported as
          :meth:`is_seed_eligible`; eval/anchor additionally see only
          ``episode_source == "natural"`` seeds.
        """
        # --- ruleset: 1v1-locked (CLAUDE.md / brief §1) ----------------------
        if self.ruleset != RULESET_1V1:
            raise ValueError(
                f"ruleset {self.ruleset!r} is not the 1v1-locked {RULESET_1V1!r} "
                "(num_players must be 2, win_vp must be 15)"
            )

        # --- episode_source literal (Literal is mypy-only) -------------------
        if self.episode_source not in get_args(EpisodeSource):
            raise ValueError(
                f"episode_source {self.episode_source!r} not in {get_args(EpisodeSource)}"
            )

        # --- players: exactly {agent, opponent}, two distinct non-empty -------
        # The 1v1 ruleset guarantees two distinct seats (brief §1). A name-OCR
        # collision (both seats read as the same string) collapses the handle set
        # to size 1, after which every downstream subset check (openings,
        # draft_order, winner) still passes — silently breaking colour→player
        # labelling and the per-game scoreboard pairing (finding §3).
        if set(self.players.keys()) != {"agent", "opponent"}:
            raise ValueError(
                f"players keys must be exactly {{'agent', 'opponent'}}, got {sorted(self.players)}"
            )
        agent_handle = self.players["agent"]
        opponent_handle = self.players["opponent"]
        for role, handle in (("agent", agent_handle), ("opponent", opponent_handle)):
            if not isinstance(handle, str) or not handle.strip():
                raise ValueError(f"{role} handle {handle!r} must be a non-empty string")
        if agent_handle == opponent_handle:
            raise ValueError(
                f"agent and opponent share the handle {agent_handle!r}; the two 1v1 seats "
                "must be distinct (likely a name-OCR collision)"
            )
        player_handles = {agent_handle, opponent_handle}

        # --- hexes: ids are true ints, multiset exactly {0..18} --------------
        hex_ids = [_as_int(h["hex_id"], f"hex_id {h.get('hex_id')!r}") for h in self.hexes]
        if set(hex_ids) != set(range(NUM_HEXES)) or len(hex_ids) != NUM_HEXES:
            raise ValueError(
                f"hex_ids must be exactly the multiset {{0..{NUM_HEXES - 1}}}, "
                f"got {sorted(hex_ids)}"
            )
        resource_counts: Counter[str] = Counter()
        number_bag: Counter[int] = Counter()
        actual_desert_hex: int | None = None
        for h, hid in zip(self.hexes, hex_ids, strict=True):
            resource = h["resource"]
            number = h.get("number")
            if resource not in RESOURCE_LITERALS:
                raise ValueError(
                    f"hex {h['hex_id']} resource {resource!r} not in {sorted(RESOURCE_LITERALS)}"
                )
            resource_counts[resource] += 1
            if resource == "DESERT":
                actual_desert_hex = hid
                if number is not None:
                    raise ValueError(
                        f"desert hex {h['hex_id']} must have number=None, got {number!r}"
                    )
            else:
                num = _as_int(number, f"hex {h['hex_id']} ({resource}) number {number!r}")
                if num not in VALID_HEX_NUMBERS:
                    raise ValueError(
                        f"hex {h['hex_id']} ({resource}) number {number!r} not in "
                        f"{sorted(VALID_HEX_NUMBERS)}"
                    )
                number_bag[num] += 1

        # --- standard-board multisets (brief §5.2 / §5.6, finding §1/§5) -----
        # NECESSARY, NOT SUFFICIENT (review findings §1/§2): this catches a wrong
        # *count* but is blind to a multiset-PRESERVING resource swap (wood↔sheep)
        # and goes tautological if board_cv forces the standard multiset. It is NOT
        # the CV-correctness net — see STANDARD_RESOURCE_COUNTS docstring for the
        # per-game clustering + orientation-independent corroboration board_cv.py
        # must add. Same "necessary-not-sufficient" caveat as the glyph anchor.
        if dict(resource_counts) != STANDARD_RESOURCE_COUNTS:
            raise ValueError(
                f"board resource counts {dict(sorted(resource_counts.items()))} are not the "
                f"standard 19-tile multiset {STANDARD_RESOURCE_COUNTS} — a CV resource "
                "misclassification yields a structurally-valid-but-wrong board"
            )
        if dict(number_bag) != STANDARD_NUMBER_BAG:
            raise ValueError(
                f"board number-token bag {dict(sorted(number_bag.items()))} is not the standard "
                f"18-token bag {STANDARD_NUMBER_BAG}"
            )

        # --- openings: exactly the two handles, each 2 distinct s + 2 distinct r
        # Completeness (both present, no extras) — a partial parse that detects
        # only one colour's pieces must not validate as complete (finding §4).
        if set(self.openings.keys()) != player_handles:
            raise ValueError(
                f"openings keys {sorted(self.openings)} must be exactly the two player handles "
                f"{sorted(player_handles)} (both present, no extras)"
            )
        settlement_sets: dict[str, set[int]] = {}
        for name, opening in self.openings.items():
            if len(opening.settlements) != 2 or len(opening.roads) != 2:
                raise ValueError(
                    f"opening for {name!r} must have 2 settlements + 2 roads, "
                    f"got {len(opening.settlements)} / {len(opening.roads)}"
                )
            settlements = [
                _as_int(v, f"settlement vertex {v!r} for {name!r}") for v in opening.settlements
            ]
            roads = [_as_int(e, f"road edge {e!r} for {name!r}") for e in opening.roads]
            for v in settlements:
                if not 0 <= v < NUM_VERTICES:
                    raise ValueError(
                        f"settlement vertex {v} for {name!r} out of 0..{NUM_VERTICES - 1}"
                    )
            for e in roads:
                if not 0 <= e < NUM_EDGES:
                    raise ValueError(f"road edge {e} for {name!r} out of 0..{NUM_EDGES - 1}")
            # A double-snap (two pieces rounded to one ID) is the literal output
            # of the §5.7 snap error — duplicate vertices/edges must be rejected.
            if len(set(settlements)) != 2:
                raise ValueError(
                    f"opening for {name!r} has duplicate settlement vertices {settlements} — "
                    "must be 2 distinct vertices (likely a double-snap)"
                )
            if len(set(roads)) != 2:
                raise ValueError(
                    f"opening for {name!r} has duplicate road edges {roads} — "
                    "must be 2 distinct edges (likely a double-snap)"
                )
            settlement_sets[name] = set(settlements)
        # Two players cannot occupy the same settlement vertex (cross-player snap).
        shared = settlement_sets[agent_handle] & settlement_sets[opponent_handle]
        if shared:
            raise ValueError(
                f"settlement vertices {sorted(shared)} are shared across both players; "
                "openings must be disjoint (likely a cross-player snap collision)"
            )

        # --- draft_order: length-4 snake [a, b, b, a], each handle twice ------
        # Load-bearing: colour→player assignment is cross-checked against the
        # snake order (brief §2, §5.14). An OCR-corrupted / partial draft order
        # that silently validates can mislabel which opening belongs to whom
        # (finding §2). The 1v1 snake draft is exactly [a, b, b, a].
        if len(self.draft_order) != 4:
            raise ValueError(
                f"draft_order must be the length-4 snake draft, got {list(self.draft_order)}"
            )
        for name in self.draft_order:
            if name not in player_handles:
                raise ValueError(
                    f"draft_order name {name!r} not in player handles {sorted(player_handles)}"
                )
        draft_counts = Counter(self.draft_order)
        a, b, c, d = self.draft_order
        if dict(draft_counts) != {agent_handle: 2, opponent_handle: 2} or not (
            a == d and b == c and a != b
        ):
            raise ValueError(
                f"draft_order {list(self.draft_order)} is not a valid snake draft [a, b, b, a] "
                "with each of the two players appearing exactly twice"
            )

        # --- winner: null or one of the player handles -----------------------
        if self.winner is not None and self.winner not in player_handles:
            raise ValueError(
                f"winner {self.winner!r} is neither null nor a player handle "
                f"{sorted(player_handles)}"
            )

        # --- dice_log: the load-bearing dice-luck covariate (brief §5.4) -----
        # dice_log is THE dice covariate the calibration scoreboard regresses
        # against (v8 value vs realized result, dice marginalized). A
        # confidently-wrong roll (a float OCR misread, a 13/1/0, or the
        # robber-only sentinel) BIASES the calibration estimate rather than
        # merely adding noise — exactly the §5 failure mode this contract blocks.
        # Every other CV-sourced int is firewalled by ``_as_int``; close the gap.
        # Range is the full 2..12 INCLUSIVE: unlike a hex token the 7 IS a legal
        # roll outcome (it triggers the robber), so it is NOT excluded here.
        for i, roll in enumerate(self.dice_log):
            value = _as_int(roll, f"dice_log[{i}] {roll!r}")
            if value not in VALID_DICE_VALUES:
                raise ValueError(
                    f"dice_log[{i}] {roll!r} not a legal 2d6 roll in "
                    f"{sorted(VALID_DICE_VALUES)} (2..12 inclusive; the 7 is legal)"
                )
        # An empty dice_log is permitted for a non-finished game (resign / video
        # cutoff, winner is None), OR when the roll VALUES were provably unreadable
        # (:data:`PROVENANCE_DICE_VALUES_READABLE` is False — Colonist renders the
        # dice as FACE GLYPHS, so "ThePhantom rolled" OCRs with no number and the
        # values cannot be recovered from text on ANY real-video game). Otherwise a
        # completed game (winner set) must have rolled, so a zero-length log there is
        # a parse failure, not a short game.
        dice_unreadable = self.provenance.get(PROVENANCE_DICE_VALUES_READABLE) is False
        if not self.dice_log and self.winner is not None and not dice_unreadable:
            raise ValueError(
                "dice_log is empty but winner is set — a completed game must carry its rolls "
                "(an empty log is only permitted on a resign / cutoff game, winner=None, or "
                f"when {PROVENANCE_DICE_VALUES_READABLE!r} is False — icon-rendered dice)"
            )

        # --- cross-field truth table (brief §5.6 / §5.7) ---------------------
        # Biconditional: rejection_reason is not None ⟺ not passed_crosscheck.
        # Forward direction: a reason set with passed_crosscheck=True is a rejected
        # record masquerading as accepted.
        if self.rejection_reason is not None and self.passed_crosscheck:
            raise ValueError(
                "rejection_reason set but passed_crosscheck=True — a rejected record is "
                "scoreboard-ineligible by definition and must have passed_crosscheck=False"
            )
        # Converse direction (brief §5.6): a failed cross-check with NO reason is a
        # reasonless rejection — it lands in the bias-audit pool uncategorizable, so
        # the per-archetype acceptance-rate audit (the whole point of §5.6) cannot
        # bucket it. Every rejected game must carry its rejection_reason.
        if not self.passed_crosscheck and self.rejection_reason is None:
            raise ValueError(
                "passed_crosscheck=False but rejection_reason is None — a rejected record "
                "must carry a rejection_reason so the §5.6 per-archetype acceptance-rate "
                "audit can bucket it"
            )

        # --- provenance orientation-binding (the cross-orientation firewall) --
        # THE board-orientation bug: the board (resources/numbers) and the
        # openings (vertex/edge IDs) are locked by *separate* CV stages. A D6
        # flip between them permutes every ID while preserving the resource and
        # number *multisets* — so the standard-board multiset gate, the
        # snake-draft gate, the distinctness gates, and the road-incidence
        # snap-sanity gate ALL pass the welded (board=desert11 / openings=desert17)
        # record. The only signal is that the two stages disagree about which
        # engine hex is the desert. Bind them here and reject on mismatch
        # (schema v2). Pure value check — both are engine ``hex_id`` ints in
        # 0..18 (no topology import; scope-lock brief §6).
        board_desert = self.provenance.get(PROVENANCE_BOARD_DESERT)
        openings_desert = self.provenance.get(PROVENANCE_OPENINGS_DESERT)
        if board_desert is None or openings_desert is None:
            raise ValueError(
                f"provenance must carry both {PROVENANCE_BOARD_DESERT!r} and "
                f"{PROVENANCE_OPENINGS_DESERT!r} (schema v{SCHEMA_VERSION} orientation-binding); "
                f"got {self.provenance.get(PROVENANCE_BOARD_DESERT)!r} / "
                f"{self.provenance.get(PROVENANCE_OPENINGS_DESERT)!r}"
            )
        board_desert = _as_int(board_desert, f"provenance.{PROVENANCE_BOARD_DESERT}")
        openings_desert = _as_int(openings_desert, f"provenance.{PROVENANCE_OPENINGS_DESERT}")
        for label, val in (
            (PROVENANCE_BOARD_DESERT, board_desert),
            (PROVENANCE_OPENINGS_DESERT, openings_desert),
        ):
            if not 0 <= val < NUM_HEXES:
                raise ValueError(
                    f"provenance.{label} {val} out of 0..{NUM_HEXES - 1} (must be an engine hex_id)"
                )
        # FREE ANCHOR: the board's *own* desert position (the unique DESERT in
        # ``self.hexes``, guaranteed to exist by the multiset gate above) is the
        # ground truth the board stage's self-reported ``board_desert_hex`` must
        # match. Asserting this (a) rejects an internally-contradictory record
        # whose provenance disagrees with its own hexes, and (b) closes the
        # welded-openings hole: the two stamps agreeing (11/11) no longer suffices
        # if they disagree with the board — and the moment the openings stage
        # honestly stamps the orientation it actually snapped under, a welded
        # board=desert11 / openings-snapped-as-desert17 record surfaces as a
        # 11-vs-17 mismatch below. Without this, the firewall trusts two
        # self-reported ints with no tie to the board it is meant to bind.
        assert actual_desert_hex is not None  # guaranteed by the multiset gate
        if board_desert != actual_desert_hex:
            raise ValueError(
                f"provenance.{PROVENANCE_BOARD_DESERT}={board_desert} disagrees with the board's "
                f"own desert hex_id {actual_desert_hex} (the unique DESERT in hexes) — the board "
                "provenance must stamp the orientation the board was actually locked under"
            )
        if board_desert != openings_desert:
            raise ValueError(
                f"orientation mismatch: provenance.{PROVENANCE_BOARD_DESERT}={board_desert} but "
                f"provenance.{PROVENANCE_OPENINGS_DESERT}={openings_desert} — the board and the "
                "openings were locked under different D6 orientations (a D6 flip preserves the "
                "resource/number multisets, so this is the only gate that catches the welded "
                "desert17/desert11 record)"
            )

        # --- provenance resolution: sub-1080p is confidently wrong (FIX-5) ----
        # The scale-up gate (assert_scale_up_orientation_gates) enforces 1080p on
        # the BATCH path, but a single-game / resumed-shard path can build a
        # GameRecord directly and bypass it. A sub-1080p source OCRs number tokens
        # and log glyphs to garbage — a confidently-wrong record, exactly what the
        # contract is meant to be the last line against. So if provenance carries a
        # ``resolution`` (it does for every real parse — brief §3), require it to
        # meet the minimum. Kept optional-when-absent to avoid breaking a record
        # that legitimately carries no resolution stamp; present-and-too-low is the
        # confidently-wrong case and is rejected.
        resolution = self.provenance.get("resolution")
        if resolution is not None:
            resolution = _as_int(resolution, "provenance.resolution")
            if resolution < MIN_RESOLUTION:
                raise ValueError(
                    f"provenance.resolution {resolution} < required {MIN_RESOLUTION} — a sub-1080p "
                    "source OCRs number tokens and log glyphs to garbage; the record is "
                    "confidently wrong, not merely noisy (FIX-5)"
                )

    def is_scoreboard_eligible(self) -> bool:
        """Whether this record may enter the opening calibration scoreboard.

        The single source of truth for the §5.4 filter — the scoreboard builder
        and the §5.6 rejection-bias audit must both call this so the predicate
        can't drift between them. The ``tier == "high"`` clause is the
        load-bearing one: dropping it silently pools ``"unknown"``-tier games
        into the calibration number, the §5.5 mixed-strength-pooling bias.

        The ``source in {"rank_badge", "tournament"}`` clause closes the
        ``known_window`` placeholder escape hatch: :data:`StrengthSource` still
        carries ``"known_window"`` for backwards compatibility, but it is an
        unfalsifiable legacy source with no committed backing window (the
        committed strength manifest uses only ``ranked_rank`` / ``tournament`` /
        ``none`` and emits zero ``known_window`` entries). A hand-set or drifted
        ``tier="high", source="known_window"`` record must NOT pool into the
        calibration number — that is exactly the §5.5 confidently-wrong
        mixed-strength failure this predicate is the last gate against. Only the
        two manifest-backed high sources are admitted, so the firewall enforces
        the exclusion its own §5.5 docstring documents rather than leaving it as
        prose (review finding: known_window escape hatch).

        **The two high sources are NOT equal-confidence (review finding:
        tournament frame-unverified).** ``rank_badge`` is OCR-rank-verified off a
        leaderboard frame; ``tournament`` is labelled purely from a title-keyword
        regex (``TOURNAMENT_RE`` + ``NON_OWN_GAME_RE`` in
        ``scripts/build_strength_manifest.py``) with NO leaderboard-frame
        confirmation, so a title false-positive enters the calibration number
        systematically (biased, not noisy) — and the scoreboard is only n≈20-40
        high games (§5.4). This predicate admits both on equal footing, so a
        downstream scoreboard builder MUST treat the two high sources as distinct
        provenance: report the ``rank_badge`` vs ``tournament`` split of its n and
        run the calibration with ``tournament`` games excluded as a robustness
        check. The 15 tournament highs (of 204) are 1v1-tournament title matches,
        NOT frame-verified.

        **This is a MATCH-strength predicate, NOT an opponent-rank one (review
        finding §1).** ``tier == "high"`` is derived from ThePhantom's OWN rank /
        tournament entry, so the ``rank_badge`` games admitted here are
        **opponent-uncontrolled** (Colonist matches by rating, so ThePhantom's
        rank is a weak proxy for the adversary's). A scoreboard builder MUST split
        its ``n`` by ``opponent_strength.source`` and report the ``tournament``
        (strong-vs-strong) games separately from the ``rank_badge`` ones — never
        collapse them into a single vs-humans number. The tournament-only subset
        is the defensible primary scoreboard and is exposed as
        :meth:`is_strong_opponent_scoreboard_eligible`.

        **Placement-order clause (step6 §3.1, harvest-blocking).** A scoreboard
        game additionally requires the openings to be in established LOG PLACEMENT
        ORDER (``provenance[`` :data:`PROVENANCE_PLACEMENT_ORDER_ESTABLISHED` ``]
        is True``): the calibration scoreboard measures v8's opening value, which
        depends on WHICH settlement grants the starting resources
        (``settlements[1]``), so an **ORDER-UNESTABLISHED** record (missing setup
        lines, or a granted multiset that disambiguates neither 1st-vs-2nd
        settlement) must NOT enter the scoreboard — an arbitrary CV-detection order
        would inject a Colonist-unreachable start state into the calibration. Such
        a record stays :meth:`is_seed_eligible` (the seed loader samples the grant
        hypothesis per episode); only the eval/scoreboard side excludes it. A
        missing flag reads as ``False`` (order-unestablished, the safe default).
        Eligibility keys on :data:`PROVENANCE_PLACEMENT_ORDER_ESTABLISHED` ALONE;
        :data:`PROVENANCE_ORDER_SOURCE` (``"log+glyph"`` / ``"glyph_only"`` /
        ``None``) is purely informational provenance for a downstream split/audit
        and never changes this predicate — a ``glyph_only``-established record
        (audit Decision 1 opt-in) is scoreboard-eligible exactly when the flag is
        ``True``, same as a ``log+glyph`` one.

        scoreboard-eligible ⟺ ``winner is not None`` AND ``passed_crosscheck``
        AND ``opponent_strength.tier == "high"`` AND
        ``opponent_strength.source in {"rank_badge", "tournament"}`` AND
        ``rejection_reason is None`` AND
        ``provenance[placement_order_established] is True``.
        """
        return (
            self.winner is not None
            and self.passed_crosscheck
            and self.opponent_strength.tier == "high"
            and self.opponent_strength.source in ("rank_badge", "tournament")
            and self.rejection_reason is None
            and self.provenance.get(PROVENANCE_PLACEMENT_ORDER_ESTABLISHED) is True
        )

    def is_strong_opponent_scoreboard_eligible(self) -> bool:
        """Whether this record may enter the **strong-opponent** calibration
        scoreboard — the primary, defensible scoreboard (review finding §1).

        This is the strict subset of :meth:`is_scoreboard_eligible` whose opponent
        is genuinely strong: the ``tournament`` source (both seats entered a 1v1
        tournament). The broad :meth:`is_scoreboard_eligible` also admits
        ``rank_badge`` games, but those are labelled ``high`` off *ThePhantom's*
        own rank, which under Colonist rating-based matchmaking is only a weak
        proxy for the *opponent's* strength — pooling them into one calibration
        number is the §5.5 mixed-opponent-strength bias the build brief forbids.

        A scoreboard builder should report this tournament-only number as its
        headline and the broader :meth:`is_scoreboard_eligible` number only as a
        source-split robustness check, never collapsed into a single vs-humans
        figure.

        strong-opponent-eligible ⟺ :meth:`is_scoreboard_eligible` AND
        ``opponent_strength.source == "tournament"``.
        """
        return self.is_scoreboard_eligible() and self.opponent_strength.source == "tournament"

    def is_seed_eligible(self) -> bool:
        """Whether this record may seed exploration (brief §5.7).

        seed-eligible ⟺ ``passed_crosscheck``. Eval/anchor consumers must
        *additionally* filter to ``episode_source == "natural"`` (that lives with
        the consumer, not the record).

        **Independent of the placement-order flag (step6 §3.1).** Seed eligibility
        deliberately does NOT require
        ``provenance[`` :data:`PROVENANCE_PLACEMENT_ORDER_ESTABLISHED` ``]``: an
        ORDER-UNESTABLISHED record (the openings are a real, structurally-legal
        human placement but the 1st-vs-2nd order could not be recovered) is still a
        usable exploration seed — the seed loader samples the grant hypothesis per
        episode rather than trusting an arbitrary order. Only
        :meth:`is_scoreboard_eligible` (the eval side) requires the flag.

        **NECESSARY, NOT SUFFICIENT (review finding §2).** ``passed_crosscheck``
        alone must NOT be read as "structurally legal opening". This predicate —
        and :meth:`validate` — check distinctness, disjointness, in-range, and the
        provenance orientation-binding, but they deliberately do **not** check the
        engine's opening-legality rules (settlement 2-away spacing, road↔settlement
        incidence as a hard rule, in-range under the *engine's* lattice). Those are
        topology/engine-aware and live outside this pure value contract
        (scope-lock, brief §6). §5.7 is explicit: every loaded seed MUST be re-run
        through the engine's own opening-legality check at load time and illegal
        ones HARD-rejected before any ``human_seed`` episode is emitted (the spike's
        mid-game snap error was 15-24px - "never trust a snapped piece without the
        legality re-check"). The seed loader (Stage 2/3) owns that mandatory gate;
        a ``passed_crosscheck=True`` record can still encode a spacing-rule
        violation and report ``True`` here. The topology-aware
        :func:`check_road_incidence` helper is likewise a snap-sanity gate, NOT a
        legality/orientation check (see its docstring).
        """
        return self.passed_crosscheck

    def to_dict(self) -> dict[str, Any]:
        """Plain-dict (JSON-serializable) form matching the brief §3 layout."""
        return {
            "schema_version": self.schema_version,
            "video_id": self.video_id,
            "game_index": self.game_index,
            "players": dict(self.players),
            "opponent_strength": asdict(self.opponent_strength),
            "ruleset": dict(self.ruleset),
            "board": {
                "hexes": [dict(h) for h in self.hexes],
                "ports": "OMITTED in v1",
            },
            "draft_order": list(self.draft_order),
            "openings": {
                name: {
                    "settlements": list(opening.settlements),
                    "roads": list(opening.roads),
                }
                for name, opening in self.openings.items()
            },
            "dice_log": list(self.dice_log),
            "winner": self.winner,
            "episode_source": self.episode_source,
            "rejection_reason": self.rejection_reason,
            "passed_crosscheck": self.passed_crosscheck,
            "provenance": dict(self.provenance),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> GameRecord:
        """Inverse of :meth:`to_dict`. Tolerant of a missing ``schema_version``
        (defaults to current) but rejects a newer one we don't understand."""
        version = int(payload.get("schema_version", SCHEMA_VERSION))
        if version > SCHEMA_VERSION:
            raise ValueError(
                f"GameRecord schema_version {version} is newer than supported {SCHEMA_VERSION}"
            )
        board = payload["board"]
        openings = {
            name: PlayerOpening(
                settlements=tuple(opening["settlements"]),
                roads=tuple(opening["roads"]),
            )
            for name, opening in payload["openings"].items()
        }
        return cls(
            schema_version=version,
            video_id=payload["video_id"],
            game_index=int(payload["game_index"]),
            players=dict(payload["players"]),
            opponent_strength=OpponentStrength(**payload["opponent_strength"]),
            ruleset={k: int(v) for k, v in payload["ruleset"].items()},
            hexes=tuple(dict(h) for h in board["hexes"]),
            draft_order=tuple(payload["draft_order"]),
            openings=openings,
            # No ``int(d)`` coercion: dice_log is the load-bearing dice-luck
            # covariate (brief §5.4), so it gets the same reject-don't-coerce
            # firewall as the board IDs — values pass through unchanged and
            # ``validate()`` rejects a float / out-of-range roll up front.
            dice_log=tuple(payload["dice_log"]),
            winner=payload["winner"],
            episode_source=payload["episode_source"],
            rejection_reason=payload.get("rejection_reason"),
            passed_crosscheck=bool(payload["passed_crosscheck"]),
            provenance=dict(payload["provenance"]),
        )

    def to_json_line(self) -> str:
        """One compact JSON line (JSONL row), no trailing newline."""
        return json.dumps(self.to_dict(), separators=(",", ":"), sort_keys=True)

    @classmethod
    def from_json_line(cls, line: str) -> GameRecord:
        """Parse one JSONL row back into a :class:`GameRecord`."""
        return cls.from_dict(json.loads(line))


def check_road_incidence(
    record: GameRecord, edge_vertices: tuple[tuple[int, int], ...]
) -> dict[str, list[int]]:
    """Generic snap-sanity gate: each opening road must touch one of its owner's
    opening settlements. Returns ``{player: [offending edge_ids]}`` (empty values
    ⟹ clean). Topology-aware, so it lives **outside** the pure
    :meth:`GameRecord.validate` (scope-lock, brief §6); call it with
    ``load_topology().edge_vertices``.

    **NOT an orientation check.** A D6 board+openings flip relabels the settlement
    *and* the road IDs by the same lattice permutation, so road↔settlement
    incidence is **D6-invariant**: it passes the welded desert17/desert11 record
    just as readily as the correct one (all 4 wrong-orientation game-1 roads pass
    it — verified in the test below). It only catches an *isolated* snap error (a
    road blob that snapped to an edge nowhere near its settlement). The
    cross-orientation firewall is the provenance orientation-binding in
    :meth:`GameRecord.validate` (board_desert == openings_desert) plus the FIX 4
    glyph anchor — **not** this gate. Do not present this as the orientation
    defense.
    """
    offenders: dict[str, list[int]] = {}
    for name, opening in record.openings.items():
        sset = set(opening.settlements)
        bad: list[int] = []
        for edge_id in opening.roads:
            a, b = edge_vertices[edge_id]
            if a not in sset and b not in sset:
                bad.append(edge_id)
        offenders[name] = bad
    return offenders
