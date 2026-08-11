"""Records → labels adapter (spec ``setup-labeling-and-champion-finetune`` D3).

Exports the owner's setup decisions from a ``scripts/play_vs_model.py`` replay
into the SAME JSONL label store the labeling tool writes, so playtesting grows
the opening corpus as a side effect.

The tool stays the volume path. One played game yields exactly **two** owner
decisions (the human's two snake-draft placements) at maybe ten minutes a game;
the tool yields roughly one per minute AND lets the owner choose which denial
positions to cover. This adapter is a by-product harvester, not a replacement —
its rows are marked ``source="game"`` so any later analysis can separate the
two populations (deliberate labels vs in-game choices under time pressure).

Refusals are deliberate and total. A row is only worth writing if the converter
can regenerate the exact position years later, which requires:

* ``metadata.setup_observed`` — the four setup steps must be OBSERVED. A
  synthesized opening (see :attr:`catan_rl.replay.schema.Metadata.setup_observed`)
  reconstructs placements from action tuples, and a mis-reconstructed placement
  is a label that trains the policy on a position nobody played.
* a non-``partial`` record — an interrupted game is a valid trace but the setup
  steps of a game that died mid-draft are not a complete draft.
* a ``human`` seat — a bot-vs-bot record has no owner decision in it, and
  heuristic openings must NEVER enter the opening corpus (spec non-goal).
* ``metadata.human_authored`` — the seat's ``kind="human"`` is a SEAT
  description, not an attestation that a person sat in it. ``play_vs_model``
  stamps that kind unconditionally, and under ``--self-test`` the base env
  auto-plays the seat with its built-in HEURISTIC opponent, opening included.
  Without this refusal the adapter would convert heuristic openings into owner
  labels — a direct violation of the directive that heuristic openings must not
  influence policy openings.

Board reconstruction: the row stores ``game_seed`` only. ``ScenarioGenerator``
seeds both ``random`` and ``np.random`` before constructing ``catanBoard``,
exactly as ``CatanEnv.reset(seed=...)`` does, so the same seed rebuilds the same
layout + port assignment. That equality is the load-bearing property of the
whole durable-corpus claim and is pinned by
``tests/unit/labeling/test_from_record.py::test_the_recorded_board_is_the_one_the_seed_rebuilds``.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

from catan_rl.env.ruleset import RULESET_R0
from catan_rl.labeling.archetypes import Archetype
from catan_rl.labeling.store import SCHEMA_VERSION, append_scenario, load_scenarios

_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_URL, "https://catan-rl/labeling/from_record")


class RecordNotLabelableError(RuntimeError):
    """The replay cannot yield trustworthy labels. Never downgraded to a warning."""


def human_seat(replay: Any) -> int:
    """Seat index (0/1) of the ``kind="human"`` player.

    Raises:
        RecordNotLabelableError: if the record has no human seat, or two.
    """
    seats = [
        int(spec.seat_index)
        for spec in (replay.metadata.player_a, replay.metadata.player_b)
        if spec.kind == "human"
    ]
    if len(seats) != 1:
        raise RecordNotLabelableError(
            f"record has {len(seats)} human seats, expected exactly 1 — "
            f"only a game an owner actually played carries an owner label"
        )
    return seats[0]


def _actor_to_seat(replay: Any) -> dict[str, int]:
    return {
        "player_a": int(replay.metadata.player_a.seat_index),
        "player_b": int(replay.metadata.player_b.seat_index),
    }


def _check_labelable(replay: Any) -> None:
    if not getattr(replay.metadata, "setup_observed", False):
        raise RecordNotLabelableError(
            "record is not setup_observed — its four setup steps were SYNTHESIZED "
            "from action tuples, so the placements are reconstructed rather than "
            "observed and cannot be trusted as labels"
        )
    if replay.metadata.partial:
        raise RecordNotLabelableError(
            "record is partial — the game was interrupted, so the draft it "
            "contains is not known to be a complete one"
        )
    if not getattr(replay.metadata, "human_authored", False):
        raise RecordNotLabelableError(
            "record is not human_authored — the human seat was auto-played (the "
            "--self-test path drives it with the base env's HEURISTIC opponent, "
            "opening included), or predates the attestation. Its draft is not an "
            "owner decision, and heuristic openings must never enter the label "
            "corpus"
        )


def extract_rows(replay: Any, *, labeler_id: str) -> list[dict[str, Any]]:
    """Build the label rows for the human seat's setup decisions.

    Pure: touches no filesystem. Returns rows in draft order.
    """
    _check_labelable(replay)
    seat = human_seat(replay)
    actor_to_seat = _actor_to_seat(replay)

    setup = [s for s in replay.steps if s.kind == "setup"]
    if len(setup) != 4:
        raise RecordNotLabelableError(
            f"record has {len(setup)} setup steps, expected 4 (the 1v1 snake draft)"
        )

    # A stable per-GAME identity, so re-exporting the same record is a no-op
    # rather than a silent doubling of the corpus. The seed alone is not unique
    # (the owner may replay a seed); the recorded timestamp disambiguates.
    game_key = f"{replay.metadata.seed}/{replay.metadata.recorded_at_utc}"
    session_id = str(uuid.uuid5(_NAMESPACE, game_key))

    picks: list[dict[str, int]] = []
    rows: list[dict[str, Any]] = []
    for position, step in enumerate(setup, start=1):
        acting = actor_to_seat[step.actor]
        vertex = int(step.actions[0].args["vertex_idx"])
        edge = int(step.actions[1].args["edge_idx"])
        if acting == seat:
            rows.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "scenario_id": str(uuid.uuid5(_NAMESPACE, f"{game_key}/{position}")),
                    "session_id": session_id,
                    "labeled_at": replay.metadata.recorded_at_utc,
                    "labeler_id": labeler_id,
                    "game_seed": int(replay.metadata.seed),
                    "draft_position": position,
                    "acting_player": acting,
                    "prior_picks": [dict(p) for p in picks],
                    # A played game declares no strategic intent — the tool's
                    # archetype prompt has no counterpart here. ``other`` is the
                    # honest value; inferring one from the placement would put a
                    # guess in a field the owner is the only authority on.
                    "archetype": Archetype.OTHER.value,
                    "settlement_vertex": vertex,
                    "road_edge": edge,
                    "decision_time_ms": 0,
                    "notes": "",
                    # ``quality_flag`` is the tool's fast-click warning. A live
                    # game is not clicked under the same conditions, so it stays
                    # empty rather than borrowing a threshold that means nothing
                    # here.
                    "quality_flag": "",
                    "source": "game",
                    # play_vs_model's HUMAN seat plays the R1 pre-roll window,
                    # but the SETUP phase is identical across epochs and the
                    # converter rebuilds it with ``RULESET_R0`` masks. Stamping
                    # the epoch the label is replayable under, not the epoch the
                    # rest of the game ran.
                    "ruleset": RULESET_R0,
                }
            )
        picks.append({"player": acting, "settlement_vertex": vertex, "road_edge": edge})
    return rows


def export_replay(replay: Any, path: str | Path, *, labeler_id: str) -> int:
    """Append the human seat's setup decisions to the JSONL store at ``path``.

    Already-exported scenarios are skipped (``scenario_id`` is a deterministic
    uuid5 of the game + draft position), so re-running the exporter over a
    replay directory is safe.

    Returns:
        The number of rows appended.
    """
    path = Path(path)
    rows = extract_rows(replay, labeler_id=labeler_id)
    existing = {r.get("scenario_id") for r in load_scenarios(path)}
    written = 0
    for row in rows:
        if row["scenario_id"] in existing:
            continue
        append_scenario(row, path)
        written += 1
    return written


def export_replay_dir(
    replay_dir: str | Path,
    path: str | Path,
    *,
    labeler_id: str,
) -> tuple[int, list[tuple[Path, str]]]:
    """Export every labelable replay under ``replay_dir``.

    Returns ``(rows_written, skipped)`` where ``skipped`` pairs each refused
    file with the reason. Refusals are REPORTED, not raised: a directory of
    playtest records legitimately mixes observed and older synthesized games,
    and one stale file must not abort the harvest of the rest.

    That promise covers UNREADABLE files too, not just refused ones.
    ``runs/human_playtest/replays`` is append-only and hand-editable; a
    truncated write or an older-schema record makes ``load_replay`` raise
    something that is not a :class:`RecordNotLabelableError`
    (``JSONDecodeError`` / ``KeyError`` / ``TypeError`` / ``ValueError``), and
    letting that propagate would let one bad file block every later harvest.
    Every failure is a skip WITH ITS REASON — nothing is dropped silently.
    """
    from catan_rl.replay.io import load_replay

    written = 0
    skipped: list[tuple[Path, str]] = []
    for file in sorted(Path(replay_dir).glob("*.json")):
        try:
            replay = load_replay(file, strict=True)
            written += export_replay(replay, path, labeler_id=labeler_id)
        except RecordNotLabelableError as exc:
            skipped.append((file, str(exc)))
        except Exception as exc:  # reported, never swallowed
            skipped.append((file, f"unreadable record ({type(exc).__name__}): {exc}"))
    return written, skipped
