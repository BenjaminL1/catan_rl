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

#: ``episode_source`` values. ``"natural"`` = a real parsed game (scoreboard
#: + eval/anchor eligible). ``"human_seed"`` = an opening used to seed exploration
#: (never re-imports the human cap; eval/anchor must filter these out).
EpisodeSource = Literal["natural", "human_seed"]

#: Coarse opponent-strength tier. The scoreboard never pools across mixed tiers.
StrengthTier = Literal["high", "unknown"]

#: How opponent strength was established (build brief §5.5). ``"known_window"`` =
#: the game falls in a known high-rank window of the channel; ``"rank_badge"`` =
#: an objective on-screen rank/elo badge was read.
StrengthSource = Literal["rank_badge", "known_window"]


@dataclass(frozen=True, slots=True)
class OpponentStrength:
    """Objective opponent-strength signal (build brief §5.5).

    Never a handle guess. ``confidence`` is a coarse 0..1 self-assessment of the
    signal, not a calibrated probability.
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


@dataclass(frozen=True, slots=True)
class PlayerOpening:
    """One player's snake-draft opening: 2 settlements + 2 roads as engine IDs."""

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
        - **Cross-field truth table** (brief §5.6 / §5.7 — see below).

        **Truth table the contract enforces and every consumer must honour:**

        - ``rejection_reason is not None`` ⟹ ``passed_crosscheck is False``. A
          rejected record is **scoreboard-ineligible by definition** but still
          emits its parsed features for the §5.6 rejection-bias audit.
        - **scoreboard-eligible** ⟺ ``winner is not None`` AND ``passed_crosscheck``
          AND ``opponent_strength.tier == "high"`` AND ``rejection_reason is None``
          (the §5.4 filter). Not asserted (eligibility is a *property*, not every
          record must be eligible), but it is implemented as the exported
          :meth:`is_scoreboard_eligible` — **that method is the single source of
          truth**; the scoreboard builder and the rejection-bias audit must both
          call it so the mixed-strength-pooling filter (§5.5) can't drift.
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
        # An empty dice_log is permitted ONLY for a non-finished game (resign /
        # video cutoff, winner is None): a completed game (winner set) must have
        # rolled, so a zero-length log there is a parse failure, not a short game.
        if not self.dice_log and self.winner is not None:
            raise ValueError(
                "dice_log is empty but winner is set — a completed game must carry its rolls "
                "(an empty log is only permitted on a resign / cutoff game, winner=None)"
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

    def is_scoreboard_eligible(self) -> bool:
        """Whether this record may enter the opening calibration scoreboard.

        The single source of truth for the §5.4 filter — the scoreboard builder
        and the §5.6 rejection-bias audit must both call this so the predicate
        can't drift between them. The ``tier == "high"`` clause is the
        load-bearing one: dropping it silently pools ``"unknown"``-tier games
        into the calibration number, the §5.5 mixed-strength-pooling bias.

        scoreboard-eligible ⟺ ``winner is not None`` AND ``passed_crosscheck``
        AND ``opponent_strength.tier == "high"`` AND ``rejection_reason is None``.
        """
        return (
            self.winner is not None
            and self.passed_crosscheck
            and self.opponent_strength.tier == "high"
            and self.rejection_reason is None
        )

    def is_seed_eligible(self) -> bool:
        """Whether this record may seed exploration (brief §5.7).

        seed-eligible ⟺ ``passed_crosscheck``. Eval/anchor consumers must
        *additionally* filter to ``episode_source == "natural"`` (that lives with
        the consumer, not the record).
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
