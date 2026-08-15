"""D3 — export the owner's real-game setup decisions into the label store.

A played game yields exactly TWO owner decisions (the human's snake-draft
placements). The tool remains the volume path; this adapter makes playtesting
grow the corpus as a side effect, at the same schema, so the converter cannot
tell the two sources apart except by the ``source`` field.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from catan_rl.labeling import from_record as fr
from catan_rl.labeling.store import SCHEMA_VERSION, load_scenarios

# ---------------------------------------------------------------------------
# Store schema v2 — optional fields, old rows keep loading
# ---------------------------------------------------------------------------


def test_schema_version_is_2() -> None:
    assert SCHEMA_VERSION == 2


def test_v1_rows_load_and_get_defaulted_provenance(tmp_path: Path) -> None:
    """The on-disk file is never rewritten; defaults are applied at READ time."""
    p = tmp_path / "scenarios.jsonl"
    row = {
        "schema_version": 1,
        "scenario_id": "s1",
        "session_id": "sess",
        "labeled_at": "2026-06-01T00:00:00Z",
        "labeler_id": "ben",
        "game_seed": 7,
        "draft_position": 1,
        "acting_player": 0,
        "prior_picks": [],
        "settlement_vertex": 3,
        "road_edge": 5,
    }
    p.write_text(json.dumps(row) + "\n")
    (loaded,) = load_scenarios(p)
    assert loaded["source"] == "tool"
    assert loaded["ruleset"] == "R0"
    assert loaded["schema_version"] == 1  # the row is NOT rewritten


# ---------------------------------------------------------------------------
# The adapter
# ---------------------------------------------------------------------------


def _auto_played_replay(tmp_path: Path, seed: int = 11, bot_seat: int = 1):  # type: ignore[no-untyped-def]
    """Play one headless game with the human seat AUTO-PLAYED, return its Replay.

    This is the ``--self-test`` path: the "human" seat is driven by the base
    env's built-in HEURISTIC opponent, opening included. The record is
    structurally identical to a played game — which is exactly why the exporter
    needs an attestation and not just ``PlayerSpec.kind``.
    """
    import importlib.util
    import sys

    import torch

    from catan_rl.policy.board_geometry import build_geometry
    from catan_rl.policy.network import CatanPolicy
    from catan_rl.replay.player_factory import _PolicyActor

    script = Path(__file__).resolve().parents[3] / "scripts" / "play_vs_model.py"
    spec = importlib.util.spec_from_file_location("pvm_from_record", script)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["pvm_from_record"] = mod
    spec.loader.exec_module(mod)

    policy = CatanPolicy()
    policy.set_board_geometry(build_geometry().as_dict_of_tensors())
    policy.eval()
    agent = mod._RawPolicyAgent(
        _PolicyActor(kind="policy", ckpt_path="<fresh>", policy=policy, device=torch.device("cpu"))
    )
    env = mod._build_human_env_class()(opponent_type="heuristic", max_turns=12)
    env._auto_human = True
    env.reset(seed=seed, options={"agent_seat": bot_seat})
    rec = mod._HumanGameRecorder(env, bot_seat=bot_seat)
    terminated = truncated = False
    steps = 0
    while not terminated and not truncated and steps < env.max_turns * 20:
        action = agent.choose_action(env)
        _o, _r, terminated, truncated, _i = env.step(action)
        rec.after_env_step(action, agent.last_internals, terminated=terminated, truncated=truncated)
        steps += 1
    return rec.finish(
        ckpt="<fresh>",
        seed=seed,
        mode="raw_policy",
        sims=None,
        clairvoyant=False,
        reveal_bot=False,
    )


@pytest.fixture(scope="module")
def auto_replay(tmp_path_factory):  # type: ignore[no-untyped-def]
    """The raw ``--self-test`` record: human seat auto-played by the heuristic."""
    return _auto_played_replay(tmp_path_factory.mktemp("g"))


@pytest.fixture(scope="module")
def replay(auto_replay):  # type: ignore[no-untyped-def]
    """A record standing in for one a PERSON played.

    A pytest run cannot click a GUI, so the trace is produced by the auto path
    and the authorship attestation is set by hand — the one thing the harness
    cannot honestly generate. Everything else (board, draft, snapshots) is the
    real recorder's output.
    """
    import dataclasses

    return dataclasses.replace(
        auto_replay,
        metadata=dataclasses.replace(auto_replay.metadata, human_authored=True),
    )


def test_refuses_an_auto_played_human_seat(auto_replay, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    """THE owner-directive refusal: heuristic openings must not influence policy
    openings. ``--self-test`` stamps ``kind="human"`` on a seat the base env
    auto-plays with its heuristic, so ``kind`` alone cannot gate the corpus."""
    assert auto_replay.metadata.human_authored is False
    assert auto_replay.metadata.player_a.kind == "human" or (
        auto_replay.metadata.player_b.kind == "human"
    )
    with pytest.raises(fr.RecordNotLabelableError, match="human_authored"):
        fr.export_replay(auto_replay, tmp_path / "s.jsonl", labeler_id="ben")
    assert not (tmp_path / "s.jsonl").exists()


def test_human_authored_survives_a_json_round_trip(auto_replay, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    """The attestation is only worth anything if it is DURABLE, and an absent
    key must never read as True."""
    import dataclasses
    import json as _json

    from catan_rl.replay.io import load_replay, save_replay

    authored = dataclasses.replace(
        auto_replay,
        metadata=dataclasses.replace(auto_replay.metadata, human_authored=True),
    )
    path = tmp_path / "r.json"
    save_replay(authored, path)
    assert load_replay(path, strict=True).metadata.human_authored is True

    payload = _json.loads(path.read_text())
    del payload["metadata"]["human_authored"]
    stripped = tmp_path / "old.json"
    stripped.write_text(_json.dumps(payload))
    assert load_replay(stripped, strict=True).metadata.human_authored is False


def test_exports_exactly_the_two_human_decisions(replay, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    dest = tmp_path / "scenarios.jsonl"
    n = fr.export_replay(replay, dest, labeler_id="ben")
    assert n == 2
    rows = load_scenarios(dest)
    assert len(rows) == 2
    human_seat = fr.human_seat(replay)
    for row in rows:
        assert row["acting_player"] == human_seat
        assert row["source"] == "game"
        assert row["ruleset"] == "R0"
        assert row["schema_version"] == SCHEMA_VERSION
        assert row["game_seed"] == replay.metadata.seed
        assert 0 <= row["settlement_vertex"] < 54
        assert 0 <= row["road_edge"] < 72
    # Snake draft: the human's two picks are at complementary positions.
    positions = sorted(r["draft_position"] for r in rows)
    assert positions in ([1, 4], [2, 3])


def test_prior_picks_are_the_placements_that_preceded_each_decision(replay, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    dest = tmp_path / "scenarios.jsonl"
    fr.export_replay(replay, dest, labeler_id="ben")
    rows = sorted(load_scenarios(dest), key=lambda r: r["draft_position"])
    for row in rows:
        assert len(row["prior_picks"]) == row["draft_position"] - 1
        for pick in row["prior_picks"]:
            assert set(pick) == {"player", "settlement_vertex", "road_edge"}


def test_the_recorded_board_is_the_one_the_seed_rebuilds(replay) -> None:  # type: ignore[no-untyped-def]
    """LOAD-BEARING. The JSONL stores a seed, not a board; the converter
    regenerates the obs from that seed years later. If the seed did not
    reproduce the board the game was played on, every exported row would train
    the policy on the WRONG position."""
    from catan_rl.labeling.scenario_gen import ScenarioGenerator

    gen = ScenarioGenerator(seed=replay.metadata.seed)
    board = gen._board
    # ``has_robber`` is deliberately NOT compared: ``board_static`` is built at
    # ``finish()`` from the LIVE board, so its robber flag has moved with the
    # game. The reconstructable part is the layout — resources + number tokens.
    rebuilt = [(h.resource_type, h.number_token) for h in board.hexTileDict.values()]
    recorded = [(h.resource, h.number_token) for h in replay.board_static.hexes]
    assert rebuilt == recorded
    rebuilt_ports = {k: sorted(v) for k, v in board.get_port_assignment().items()}
    recorded_ports: dict[str, list[int]] = {}
    for port in replay.board_static.ports:
        key = "3:1 PORT" if port.ratio == "3:1" else f"2:1 {port.resource}"
        recorded_ports.setdefault(key, []).extend(port.vertex_idx_pair)
    assert {k: sorted(v) for k, v in recorded_ports.items()} == rebuilt_ports


def test_refuses_a_record_whose_setup_was_synthesized(replay, tmp_path: Path) -> None:
    """A synthesized opening cannot be trusted as a LABEL — the placements were
    reconstructed rather than observed. Refuse rather than emit a plausible row."""
    import dataclasses

    stale = dataclasses.replace(
        replay, metadata=dataclasses.replace(replay.metadata, setup_observed=False)
    )
    with pytest.raises(fr.RecordNotLabelableError, match="setup_observed"):
        fr.export_replay(stale, tmp_path / "s.jsonl", labeler_id="ben")


def test_refuses_a_partial_record(replay, tmp_path: Path) -> None:
    import dataclasses

    partial = dataclasses.replace(
        replay, metadata=dataclasses.replace(replay.metadata, partial=True)
    )
    with pytest.raises(fr.RecordNotLabelableError, match="partial"):
        fr.export_replay(partial, tmp_path / "s.jsonl", labeler_id="ben")


def test_refuses_a_record_with_no_human_seat(replay, tmp_path: Path) -> None:
    import dataclasses

    bot_only = dataclasses.replace(
        replay,
        metadata=dataclasses.replace(
            replay.metadata,
            player_a=dataclasses.replace(replay.metadata.player_a, kind="policy"),
            player_b=dataclasses.replace(replay.metadata.player_b, kind="policy"),
        ),
    )
    with pytest.raises(fr.RecordNotLabelableError, match="human"):
        fr.export_replay(bot_only, tmp_path / "s.jsonl", labeler_id="ben")


def test_export_is_idempotent_per_scenario_id(replay, tmp_path: Path) -> None:
    """Re-exporting the same game must not silently double-count the corpus."""
    dest = tmp_path / "scenarios.jsonl"
    assert fr.export_replay(replay, dest, labeler_id="ben") == 2
    assert fr.export_replay(replay, dest, labeler_id="ben") == 0
    assert len(load_scenarios(dest)) == 2


def test_a_game_sourced_row_converts_through_the_shard_converter(replay, tmp_path: Path) -> None:
    """D3's producer meets D4's consumer.

    ``from_record`` derives ``draft_position`` / ``acting_player`` by enumerating
    setup steps and reading ``metadata.player_*.seat_index``; ``to_shard``
    re-derives them from ``ScenarioGenerator`` state and HARD-RAISES on a
    mismatch. A seat-convention drift between the env and the generator would
    make every game-sourced label unconvertible, and neither side's own tests
    would notice.
    """
    from catan_rl.labeling.to_shard import convert, rows_for_label

    dest = tmp_path / "scenarios.jsonl"
    assert fr.export_replay(replay, dest, labeler_id="ben") == 2
    rows = load_scenarios(dest)
    assert {r["source"] for r in rows} == {"game"}

    for row in rows:
        settle, road = rows_for_label(row, game_id=0)
        assert settle.action[0] != road.action[0]
        assert int(settle.action[1]) == int(row["settlement_vertex"])
        assert int(road.action[2]) == int(row["road_edge"])

    # EXPLICIT 0.0: one exported game is one ``game_seed``, and ``convert``'s
    # default now withholds a seed — which for a single-seed corpus leaves no
    # training half. This test is about the adapter's rows reaching the shard,
    # so it wants the whole corpus.
    manifest = convert(dest, tmp_path / "shard", held_out_frac=0.0)
    assert manifest["label_source_counts"] == {"game": 2}
    assert manifest["n_scenarios"] == 2


def test_both_seatings_convert(tmp_path_factory, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    """The seat convention must hold whichever seat the bot took."""
    import dataclasses

    from catan_rl.labeling.to_shard import rows_for_label

    for bot_seat in (0, 1):
        raw = _auto_played_replay(
            tmp_path_factory.mktemp(f"seat{bot_seat}"), seed=23 + bot_seat, bot_seat=bot_seat
        )
        rep = dataclasses.replace(
            raw, metadata=dataclasses.replace(raw.metadata, human_authored=True)
        )
        dest = tmp_path / f"s{bot_seat}.jsonl"
        assert fr.export_replay(rep, dest, labeler_id="ben") == 2
        for row in load_scenarios(dest):
            assert row["acting_player"] == fr.human_seat(rep)
            assert len(rows_for_label(row, game_id=0)) == 2
