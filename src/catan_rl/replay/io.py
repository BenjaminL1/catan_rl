"""Atomic JSON I/O for replay files.

The pattern is the same as
:func:`catan_rl.checkpoint.manager.save_checkpoint`: write to a
``<dest>.tmp`` sibling, ``fsync`` it, ``os.replace`` into place, then
``fsync`` the parent directory. On any ``BaseException`` the tmp
file is unlinked so a Ctrl-C / OOM mid-write never leaves a stray
``.tmp`` alongside the real artifact.

Reads use the same versioned-schema discipline as the checkpoint
module: :func:`load_replay` rejects forward-incompatible files
(``schema_version > REPLAY_SCHEMA_VERSION``) and walks any registered
backward migrations from :mod:`catan_rl.replay.migrations` to the
current version before instantiating dataclasses.
"""

from __future__ import annotations

import contextlib
import dataclasses
import json
import logging
import os
from pathlib import Path
from typing import Any

from catan_rl.replay.migrations import apply_migrations
from catan_rl.replay.schema import (
    REPLAY_SCHEMA_VERSION,
    BoardStatic,
    EdgeStatic,
    HexStatic,
    Metadata,
    PlayerSpec,
    PlayerStateSnapshot,
    PortStatic,
    Replay,
    ReplaySchemaError,
    ReplayStep,
    StepStateSnapshot,
    SubAction,
    VertexStatic,
    event_from_dict,
    event_to_dict,
)

_LOG = logging.getLogger("catan_rl.replay")


# ---------------------------------------------------------------------------
# Write
# ---------------------------------------------------------------------------


def _fsync_dir(path: Path) -> None:
    """Best-effort fsync of a directory inode.

    Required after ``os.replace`` for the rename to survive a power
    loss on POSIX (the file's data is durable after the file-fsync,
    but the directory entry is only guaranteed durable after the
    parent directory is fsync'd). On Windows the call raises
    ``OSError``; silently no-op there."""
    try:
        dir_fd = os.open(path, os.O_DIRECTORY)
    except OSError:
        return
    try:
        with contextlib.suppress(OSError):
            os.fsync(dir_fd)
    finally:
        os.close(dir_fd)


def _replay_to_dict(replay: Replay) -> dict[str, Any]:
    """Convert a :class:`Replay` to a JSON-safe dict.

    Uses ``dataclasses.asdict`` for the leaf dataclasses but routes
    every :class:`StepEvent` through :func:`event_to_dict` so the
    ``kind`` discriminator is emitted correctly and :class:`UnknownEvent`
    round-trips its original payload bytes."""

    def _step_to_dict(step: ReplayStep) -> dict[str, Any]:
        return {
            "step_idx": step.step_idx,
            "kind": step.kind,
            "actor": step.actor,
            "dice_roll": list(step.dice_roll) if step.dice_roll is not None else None,
            "actions": [{"kind": a.kind, "args": dict(a.args)} for a in step.actions],
            "events": [event_to_dict(e) for e in step.events],
            "log_lines": list(step.log_lines),
            "state_after": _state_after_to_dict(step.state_after),
        }

    def _state_after_to_dict(s: StepStateSnapshot) -> dict[str, Any]:
        return {
            "settlements": {k: list(v) for k, v in s.settlements.items()},
            "cities": {k: list(v) for k, v in s.cities.items()},
            "roads": {k: list(v) for k, v in s.roads.items()},
            "robber_hex": s.robber_hex,
            "players": {k: dataclasses.asdict(v) for k, v in s.players.items()},
            "longest_road_holder": s.longest_road_holder,
            "largest_army_holder": s.largest_army_holder,
            "last_seven_roller": s.last_seven_roller,
        }

    def _board_static_to_dict(b: BoardStatic) -> dict[str, Any]:
        return {
            "hexes": [dataclasses.asdict(h) for h in b.hexes],
            "vertices": [
                {
                    "vertex_idx": v.vertex_idx,
                    "adjacent_hex_indices": list(v.adjacent_hex_indices),
                }
                for v in b.vertices
            ],
            "edges": [dataclasses.asdict(e) for e in b.edges],
            "ports": [
                {
                    "port_idx": p.port_idx,
                    "vertex_idx_pair": list(p.vertex_idx_pair),
                    "ratio": p.ratio,
                    "resource": p.resource,
                }
                for p in b.ports
            ],
        }

    return {
        "schema_version": replay.schema_version,
        "metadata": {
            "player_a": dataclasses.asdict(replay.metadata.player_a),
            "player_b": dataclasses.asdict(replay.metadata.player_b),
            "seed": replay.metadata.seed,
            "max_turns": replay.metadata.max_turns,
            "intended_hex_size": list(replay.metadata.intended_hex_size),
            "recorded_at_utc": replay.metadata.recorded_at_utc,
            "winner": replay.metadata.winner,
            "winner_seat": replay.metadata.winner_seat,
            "final_vp": list(replay.metadata.final_vp),
            "total_steps": replay.metadata.total_steps,
            "partial": replay.metadata.partial,
        },
        "board_static": _board_static_to_dict(replay.board_static),
        "steps": [_step_to_dict(s) for s in replay.steps],
    }


def save_replay(replay: Replay, path: str | Path, *, force: bool = False) -> Path:
    """Write ``replay`` to ``path`` atomically.

    The bytes hit ``<path>.tmp`` first, are fsync'd, and then
    :func:`os.replace`'d into the target. After the rename, the
    parent directory's inode is fsync'd so the new directory entry
    survives a power loss on POSIX. A mid-write crash (any
    ``BaseException``, including ``KeyboardInterrupt``) cleans up the
    tmp file.

    Args:
        replay: the in-memory replay to serialise.
        path: destination JSON file. Parent directories are created
            as needed.
        force: if ``False`` (default) and ``path`` already exists,
            raise ``FileExistsError`` to protect against accidental
            overwrites. ``True`` opts in to overwrite.

    Returns the resolved path that was written.
    """
    dest = Path(path).expanduser().resolve()
    if dest.exists() and not force:
        raise FileExistsError(f"refusing to overwrite {dest}; pass force=True to overwrite")
    dest.parent.mkdir(parents=True, exist_ok=True)
    payload = _replay_to_dict(replay)
    # ``separators=(",",":")`` keeps the file compact (these replays
    # can grow to a few MB at long max_turns).
    serialised = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write(serialised)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, dest)
    except BaseException:
        with contextlib.suppress(OSError):
            tmp.unlink(missing_ok=True)
        raise
    _fsync_dir(dest.parent)
    return dest


# ---------------------------------------------------------------------------
# Read
# ---------------------------------------------------------------------------


def _replay_from_dict(payload: dict[str, Any], *, strict: bool) -> Replay:
    """Reconstruct a :class:`Replay` instance from a v1 payload dict.

    Caller responsibility: ``payload`` is already migrated to the
    current schema version (:func:`apply_migrations` was run upstream
    by :func:`load_replay`).
    """

    def _player_spec(d: dict[str, Any]) -> PlayerSpec:
        return PlayerSpec(
            kind=d["kind"],
            ckpt_path=d.get("ckpt_path"),
            color=d["color"],
            seat_index=int(d["seat_index"]),
        )

    def _metadata(d: dict[str, Any]) -> Metadata:
        return Metadata(
            player_a=_player_spec(d["player_a"]),
            player_b=_player_spec(d["player_b"]),
            seed=int(d["seed"]),
            max_turns=int(d["max_turns"]),
            intended_hex_size=tuple(d["intended_hex_size"]),
            recorded_at_utc=str(d["recorded_at_utc"]),
            winner=d.get("winner"),
            winner_seat=d.get("winner_seat"),
            final_vp=tuple(d["final_vp"]),
            total_steps=int(d["total_steps"]),
            partial=bool(d.get("partial", False)),
        )

    def _board_static(d: dict[str, Any]) -> BoardStatic:
        return BoardStatic(
            hexes=tuple(
                HexStatic(
                    hex_idx=int(h["hex_idx"]),
                    q=int(h["q"]),
                    r=int(h["r"]),
                    resource=h["resource"],
                    number_token=h.get("number_token"),
                    has_robber_initial=bool(h["has_robber_initial"]),
                )
                for h in d["hexes"]
            ),
            vertices=tuple(
                VertexStatic(
                    vertex_idx=int(v["vertex_idx"]),
                    adjacent_hex_indices=tuple(v["adjacent_hex_indices"]),
                )
                for v in d["vertices"]
            ),
            edges=tuple(
                EdgeStatic(
                    edge_idx=int(e["edge_idx"]),
                    v1_idx=int(e["v1_idx"]),
                    v2_idx=int(e["v2_idx"]),
                )
                for e in d["edges"]
            ),
            ports=tuple(
                PortStatic(
                    port_idx=int(p["port_idx"]),
                    vertex_idx_pair=tuple(p["vertex_idx_pair"]),
                    ratio=p["ratio"],
                    resource=p.get("resource"),
                )
                for p in d["ports"]
            ),
        )

    def _player_snapshot(d: dict[str, Any]) -> PlayerStateSnapshot:
        return PlayerStateSnapshot(
            name=str(d["name"]),
            vp=int(d["vp"]),
            resources=dict(d["resources"]),
            dev_cards_hand=dict(d["dev_cards_hand"]),
            dev_cards_played=dict(d["dev_cards_played"]),
        )

    def _state_after(d: dict[str, Any]) -> StepStateSnapshot:
        return StepStateSnapshot(
            settlements={k: tuple(v) for k, v in d["settlements"].items()},
            cities={k: tuple(v) for k, v in d["cities"].items()},
            roads={k: tuple(v) for k, v in d["roads"].items()},
            robber_hex=int(d["robber_hex"]),
            players={k: _player_snapshot(v) for k, v in d["players"].items()},
            longest_road_holder=d.get("longest_road_holder"),
            largest_army_holder=d.get("largest_army_holder"),
            # Optional in v1 for tolerance to older replays — the
            # writer always populates it but a hand-crafted v1 file
            # may omit it. Default ``None`` matches the schema.
            last_seven_roller=d.get("last_seven_roller"),
        )

    def _step(d: dict[str, Any]) -> ReplayStep:
        dice = d.get("dice_roll")
        return ReplayStep(
            step_idx=int(d["step_idx"]),
            kind=d["kind"],
            actor=d["actor"],
            dice_roll=tuple(dice) if dice is not None else None,
            actions=tuple(
                SubAction(kind=a["kind"], args=dict(a.get("args", {}))) for a in d["actions"]
            ),
            events=tuple(event_from_dict(e, strict=strict) for e in d["events"]),
            log_lines=tuple(d.get("log_lines", [])),
            state_after=_state_after(d["state_after"]),
        )

    return Replay(
        schema_version=int(payload["schema_version"]),
        metadata=_metadata(payload["metadata"]),
        board_static=_board_static(payload["board_static"]),
        steps=tuple(_step(s) for s in payload["steps"]),
    )


def load_replay(path: str | Path, *, strict: bool = False) -> Replay:
    """Read ``path`` and return a :class:`Replay` instance.

    **Forward-compat contract**: unknown TOP-LEVEL or nested keys in
    the JSON envelope are silently dropped. A v1 reader on a v2
    file that adds new metadata or per-step fields keeps loading
    successfully; the v2-specific fields are inaccessible to v1
    code but a write-after-load cycle does NOT preserve them
    (round-trip is lossy for unknown fields, intentionally). Use
    :class:`UnknownEvent` for event-kind forward compat that DOES
    round-trip.

    Args:
        path: filesystem path to a replay JSON file.
        strict: forwarded to :func:`event_from_dict`. When ``True``,
            unknown event kinds raise; when ``False`` (default),
            they're rendered as :class:`UnknownEvent` and logged.
            Tests pass ``strict=True`` to assert the recorder only
            emits known kinds.

    Raises:
        ReplaySchemaError: on missing/malformed JSON, missing
            ``schema_version``, forward-incompatible version, or any
            decode failure inside a known dataclass.
    """
    src = Path(path).expanduser().resolve()
    if not src.exists():
        raise ReplaySchemaError(f"replay file not found: {src}")
    try:
        raw_text = src.read_text(encoding="utf-8")
    except OSError as e:
        raise ReplaySchemaError(f"failed to read {src}: {e}") from e
    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError as e:
        raise ReplaySchemaError(f"malformed JSON at {src}: {e}") from e
    if not isinstance(payload, dict):
        raise ReplaySchemaError(f"replay top-level must be a dict, got {type(payload).__name__}")
    if "schema_version" not in payload:
        raise ReplaySchemaError(f"replay at {src} missing 'schema_version' field")
    saved_version = int(payload["schema_version"])
    if saved_version > REPLAY_SCHEMA_VERSION:
        raise ReplaySchemaError(
            f"replay at {src} has schema_version={saved_version}; this "
            f"codebase only supports up to v{REPLAY_SCHEMA_VERSION}. "
            "Upgrade the codebase to read this file."
        )
    # Walk the registered migrations forward to the current version.
    try:
        payload = apply_migrations(payload, target_version=REPLAY_SCHEMA_VERSION)
    except Exception as e:  # MigrationError is the expected raise
        raise ReplaySchemaError(f"migration failed for {src}: {e}") from e
    try:
        return _replay_from_dict(payload, strict=strict)
    except (KeyError, TypeError, ValueError) as e:
        # `KeyError` covers missing required fields; `TypeError` and
        # `ValueError` cover ``null`` in non-optional fields (e.g.,
        # ``"intended_hex_size": null`` raises TypeError from
        # ``tuple(None)``). All three surface as ``ReplaySchemaError``
        # so callers have one umbrella to catch.
        raise ReplaySchemaError(f"replay at {src} malformed: {type(e).__name__}: {e}") from e
