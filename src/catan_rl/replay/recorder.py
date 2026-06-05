"""Recorder: build replay states from a live ``catanGame``.

Phase 2a of the replay-system build-out. Provides the headless
state-snapshot bridge — the only function in this module is
:func:`snapshot_step_state` which wraps the engine's Phase 0.5
``game.snapshot_state(...)`` accessor and converts the returned
JSON-safe dict into the typed :class:`StepStateSnapshot` dataclass.

Why a separate wrapper:

* The engine accessor lives in :mod:`catan_rl.engine.game` and returns
  a ``dict[str, Any]`` so the engine has zero dependency on the
  replay schema. The replay package owns the schema, so the
  dict→dataclass conversion lives here.
* The wrapper enforces the deep-copy contract one more time
  (``copy.deepcopy`` on each mutable sub-dict) so subsequent engine
  mutations cannot reach back into stored snapshots even if the
  engine accessor regresses on its own deep-copy.

Future phases (2b, 2c, 2d) will live in this same module and produce
:class:`SubAction` / :class:`StepEvent` instances per turn.
"""

from __future__ import annotations

import copy
from typing import Any

from catan_rl.replay.schema import PlayerStateSnapshot, StepStateSnapshot


def snapshot_step_state(
    game: Any,
    *,
    seat_to_actor: dict[str, str],
    vertex_pixel_to_idx: dict[Any, int],
    edge_key_to_idx: dict[tuple[str, str], int],
) -> StepStateSnapshot:
    """Capture ``game``'s state and return a :class:`StepStateSnapshot`.

    Args:
        game: a live :class:`catanGame` instance. Its
            ``snapshot_state(...)`` accessor (added Phase 0.5) is the
            only engine surface this function touches.
        seat_to_actor: maps engine player ``name`` →
            ``"player_a"``/``"player_b"``. Built once at recorder
            reset based on ``agent_seat`` and passed unchanged
            thereafter.
        vertex_pixel_to_idx: the env's ``_vertex_to_idx`` map (engine
            keys vertices by pixel-coord tuples; the recorder needs
            integer indices for the JSON).
        edge_key_to_idx: the env's edge-key → integer index map; the
            keys are ``(s1, s2)`` lex-sorted string tuples per the
            convention in :func:`catan_env._edge_key`.

    Returns a frozen :class:`StepStateSnapshot`. The function is
    deep-copy-safe: subsequent engine mutations on ``game`` do not
    alter the returned snapshot.
    """
    raw = game.snapshot_state(seat_to_actor, vertex_pixel_to_idx, edge_key_to_idx)
    return _state_after_from_dict(raw)


def _state_after_from_dict(raw: dict[str, Any]) -> StepStateSnapshot:
    """Convert the engine's snapshot dict to a frozen
    :class:`StepStateSnapshot`. Defensive: deep-copies every mutable
    sub-container so the schema instance is independent of the source
    dict (the engine already deep-copies internally but the wrapper
    repeats it to guarantee the contract end-to-end)."""
    settlements = {k: tuple(int(i) for i in v) for k, v in raw["settlements"].items()}
    cities = {k: tuple(int(i) for i in v) for k, v in raw["cities"].items()}
    roads = {k: tuple(int(i) for i in v) for k, v in raw["roads"].items()}

    players: dict[str, PlayerStateSnapshot] = {}
    for actor, snap in raw["players"].items():
        players[actor] = PlayerStateSnapshot(
            name=str(snap["name"]),
            vp=int(snap["vp"]),
            resources=copy.deepcopy(dict(snap["resources"])),
            dev_cards_hand=copy.deepcopy(dict(snap["dev_cards_hand"])),
            dev_cards_played=copy.deepcopy(dict(snap["dev_cards_played"])),
        )

    return StepStateSnapshot(
        settlements=settlements,
        cities=cities,
        roads=roads,
        robber_hex=int(raw["robber_hex"]),
        players=players,
        longest_road_holder=raw.get("longest_road_holder"),
        largest_army_holder=raw.get("largest_army_holder"),
        last_seven_roller=raw.get("last_seven_roller"),
    )
