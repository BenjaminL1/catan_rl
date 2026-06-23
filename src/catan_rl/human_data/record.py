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

        - **Resources / numbers:** every hex resource in :data:`RESOURCE_LITERALS`;
          ``DESERT`` ⟺ ``number is None``, non-desert ⟺ ``number`` in 2..12 \\ {7}.
        - **Board IDs:** ``hex_id`` multiset is exactly ``{0..18}``; all opening
          vertices in ``0..53``, edges in ``0..71``; exactly 2 settlements + 2
          roads per player.
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

        # --- hexes: multiset exactly {0..18}; resources / numbers ------------
        hex_ids = [int(h["hex_id"]) for h in self.hexes]
        if set(hex_ids) != set(range(NUM_HEXES)) or len(hex_ids) != NUM_HEXES:
            raise ValueError(
                f"hex_ids must be exactly the multiset {{0..{NUM_HEXES - 1}}}, "
                f"got {sorted(hex_ids)}"
            )
        for h in self.hexes:
            resource = h["resource"]
            number = h.get("number")
            if resource not in RESOURCE_LITERALS:
                raise ValueError(
                    f"hex {h['hex_id']} resource {resource!r} not in {sorted(RESOURCE_LITERALS)}"
                )
            if resource == "DESERT":
                if number is not None:
                    raise ValueError(
                        f"desert hex {h['hex_id']} must have number=None, got {number!r}"
                    )
            else:
                if number is None or int(number) not in VALID_HEX_NUMBERS:
                    raise ValueError(
                        f"hex {h['hex_id']} ({resource}) number {number!r} not in "
                        f"{sorted(VALID_HEX_NUMBERS)}"
                    )

        # --- openings: keys ⊆ player handles; 2 settlements + 2 roads in range
        player_handles = set(self.players.values())
        for name, opening in self.openings.items():
            if name not in player_handles:
                raise ValueError(
                    f"opening key {name!r} not in player handles {sorted(player_handles)}"
                )
            if len(opening.settlements) != 2 or len(opening.roads) != 2:
                raise ValueError(
                    f"opening for {name!r} must have 2 settlements + 2 roads, "
                    f"got {len(opening.settlements)} / {len(opening.roads)}"
                )
            for v in opening.settlements:
                if not 0 <= int(v) < NUM_VERTICES:
                    raise ValueError(
                        f"settlement vertex {v} for {name!r} out of 0..{NUM_VERTICES - 1}"
                    )
            for e in opening.roads:
                if not 0 <= int(e) < NUM_EDGES:
                    raise ValueError(f"road edge {e} for {name!r} out of 0..{NUM_EDGES - 1}")

        # --- draft_order: names ⊆ player handles -----------------------------
        for name in self.draft_order:
            if name not in player_handles:
                raise ValueError(
                    f"draft_order name {name!r} not in player handles {sorted(player_handles)}"
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
