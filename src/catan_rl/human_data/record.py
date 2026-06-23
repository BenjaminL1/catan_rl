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

#: Schema version of the ``GameRecord`` contract. Mirrors
#: ``catan_rl.conformance.recorder.CONFORMANCE_SCHEMA_VERSION`` (both ``1``).
SCHEMA_VERSION = 1

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
        - **Cross-field truth table** (brief §5.6 / §5.7 — see below).

        **Truth table the contract enforces and every consumer must honour:**

        - ``rejection_reason is not None`` ⟹ ``passed_crosscheck is False``. A
          rejected record is **scoreboard-ineligible by definition** but still
          emits its parsed features for the §5.6 rejection-bias audit.
        - **scoreboard-eligible** ⟺ ``winner is not None`` AND ``passed_crosscheck``
          AND ``opponent_strength.tier == "high"`` AND ``rejection_reason is None``
          (the §5.4 filter). Not asserted (eligibility is a *property*, not every
          record must be eligible), but the predicate is fixed here so the
          scoreboard filter and the audit can't drift.
        - **seed-eligible** ⟺ ``passed_crosscheck`` (brief §5.7); eval/anchor see
          only ``episode_source == "natural"`` seeds.
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
        for h in self.hexes:
            resource = h["resource"]
            number = h.get("number")
            if resource not in RESOURCE_LITERALS:
                raise ValueError(
                    f"hex {h['hex_id']} resource {resource!r} not in {sorted(RESOURCE_LITERALS)}"
                )
            resource_counts[resource] += 1
            if resource == "DESERT":
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

        # --- cross-field truth table (brief §5.6 / §5.7) ---------------------
        if self.rejection_reason is not None and self.passed_crosscheck:
            raise ValueError(
                "rejection_reason set but passed_crosscheck=True — a rejected record is "
                "scoreboard-ineligible by definition and must have passed_crosscheck=False"
            )

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
            dice_log=tuple(int(d) for d in payload["dice_log"]),
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
