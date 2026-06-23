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
from typing import Any, Literal

#: Schema version of the ``GameRecord`` contract. Mirrors
#: ``catan_rl.conformance.recorder.CONFORMANCE_SCHEMA_VERSION`` (both ``1``).
SCHEMA_VERSION = 1

#: Resource string literals permitted on a hex (build brief §5.8). There is no
#: ``RESOURCES`` enum in ``engine/``; these are the only stable values.
RESOURCE_LITERALS: frozenset[str] = frozenset({"WOOD", "BRICK", "WHEAT", "ORE", "SHEEP", "DESERT"})

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
