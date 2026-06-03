"""Tests for the BC dataset generator.

Smoke tests (3-10 games) pin the schema, the discounted-z math, the
forced-move filter, and end-to-end NPZ round-trip. The full 30k-game
run lives in scripts/generate_bc_dataset.py and is exercised separately.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from catan_rl.bc.dataset import generate_dataset, play_game
from catan_rl.policy.obs_schema import (
    CURR_PLAYER_DIM,
    N_DEV_TYPES,
    N_EDGES,
    N_TILES,
    N_VERTICES,
    NEXT_PLAYER_DIM,
    TILE_DIM,
)

# ---------------------------------------------------------------------------
# play_game contract
# ---------------------------------------------------------------------------


def test_play_game_returns_at_least_one_decision() -> None:
    record = play_game(game_id=0, seed=0, perturbation="canonical", max_turns=150)
    assert len(record.decisions) > 0


def test_play_game_both_players_recorded() -> None:
    """Decisions from both seat 0 and seat 1 must appear in the record."""
    record = play_game(game_id=0, seed=0, perturbation="canonical", max_turns=150)
    seats = {d.player_seat for d in record.decisions}
    assert seats == {0, 1}, f"missing seat in records: {seats}"


def test_play_game_winner_xor_truncated() -> None:
    """Exactly one of p1_won / p2_won / truncated must be true."""
    record = play_game(game_id=0, seed=0, perturbation="canonical", max_turns=400)
    assert record.p1_won + record.p2_won + int(record.truncated) == 1


def test_play_game_action_shapes() -> None:
    record = play_game(game_id=0, seed=0, perturbation="canonical", max_turns=80)
    for d in record.decisions:
        assert d.action.shape == (6,)
        assert d.action.dtype == np.int64


def test_play_game_setup_actions_tagged_setup_phase() -> None:
    """The 4 setup builds (2 per player) must have phase='setup'."""
    record = play_game(game_id=0, seed=0, perturbation="canonical", max_turns=50)
    setup_decisions = [d for d in record.decisions if d.phase == "setup"]
    # Each player does settle + road in fwd pass + settle + road in reverse
    # = 4 setup decisions per player, 8 total. Some may have been forced
    # (single-option setup_settlement) but the phase tag still fires.
    assert len(setup_decisions) >= 6  # at minimum a few setup decisions
    for d in setup_decisions:
        assert d.action[0] in (0, 2)  # BUILD_SETTLEMENT or BUILD_ROAD


def test_play_game_z_disc_matches_terminal_outcome() -> None:
    """For a P1 win, seat-0 decisions have z_disc > 0 at the end,
    seat-1 decisions have z_disc < 0 at the end.
    """
    # Try a few seeds until we get a non-truncated game.
    for seed in range(20):
        record = play_game(game_id=0, seed=seed, perturbation="canonical", max_turns=400)
        if record.truncated:
            continue
        last_seat0 = next((d for d in reversed(record.decisions) if d.player_seat == 0), None)
        last_seat1 = next((d for d in reversed(record.decisions) if d.player_seat == 1), None)
        if last_seat0 is None or last_seat1 is None:
            continue
        winner = 0 if record.p1_won else 1
        if winner == 0:
            assert last_seat0.z_disc > 0
            assert last_seat1.z_disc < 0
        else:
            assert last_seat0.z_disc < 0
            assert last_seat1.z_disc > 0
        return  # one non-truncated game suffices
    pytest.skip("no non-truncated games in 20 seeds — adjust max_turns")


def test_z_disc_uses_per_seat_steps_not_flat_index() -> None:
    """Regression for the value-head game-length leak (review #4, 2026-06-03).

    Before the fix, ``z_disc[i] = γ^(n-1-i)`` walked a flat decision index
    that interleaved both seats; P1's last decision was at distance
    ``len(P2's_remaining_decisions)`` from "terminal" rather than 0. With
    the per-seat fix, each seat's last decision is at discount^0
    regardless of how many decisions the other seat made after it.

    Pin: for a non-truncated game, the last decision *of each seat*
    has |z_disc| == 1.0 (γ^0). Pre-fix this only held for the seat that
    made the very last decision; the other seat had |z_disc| ≤ γ^k for
    some k > 0.
    """
    for seed in range(20):
        record = play_game(
            game_id=0, seed=seed, perturbation="canonical", max_turns=400, discount=0.99
        )
        if record.truncated:
            continue
        last_seat0 = next((d for d in reversed(record.decisions) if d.player_seat == 0), None)
        last_seat1 = next((d for d in reversed(record.decisions) if d.player_seat == 1), None)
        if last_seat0 is None or last_seat1 is None:
            continue
        # Both seats' last decisions land at discount^0 == 1.0 under the
        # per-seat scheme — the absolute value tracks z_by_seat[seat]
        # which is ±1 on non-truncated games.
        assert abs(abs(last_seat0.z_disc) - 1.0) < 1e-6, (
            f"seed={seed}: seat 0 last z_disc={last_seat0.z_disc}, expected ±1.0"
        )
        assert abs(abs(last_seat1.z_disc) - 1.0) < 1e-6, (
            f"seed={seed}: seat 1 last z_disc={last_seat1.z_disc}, expected ±1.0"
        )
        return
    pytest.skip("no non-truncated games in 20 seeds")


def test_play_game_z_disc_discounted_back_in_time() -> None:
    """z_disc magnitude must decay with discount γ as we go back in time."""
    for seed in range(20):
        record = play_game(
            game_id=0, seed=seed, perturbation="canonical", max_turns=400, discount=0.99
        )
        if record.truncated:
            continue
        seat0_decisions = [d for d in record.decisions if d.player_seat == 0]
        if len(seat0_decisions) < 5:
            continue
        # |z_disc| should be monotonically non-decreasing as we approach the end.
        magnitudes = [abs(d.z_disc) for d in seat0_decisions]
        # All should be non-zero (winner or loser).
        if all(m == 0 for m in magnitudes):
            continue
        # Strictly monotonic non-decreasing.
        for i in range(len(magnitudes) - 1):
            assert magnitudes[i] <= magnitudes[i + 1] + 1e-6
        return
    pytest.skip("no usable game in 20 seeds")


def test_play_game_perturbation_recorded() -> None:
    for perturb in ("canonical", "epsilon_greedy", "weight_noised"):
        record = play_game(game_id=0, seed=0, perturbation=perturb, max_turns=60)
        assert record.perturbation == perturb


def test_play_game_rejects_unknown_perturbation() -> None:
    with pytest.raises(ValueError):
        play_game(game_id=0, seed=0, perturbation="bogus", max_turns=20)


def test_play_game_forced_flag_set_on_roll_dice() -> None:
    """ROLL_DICE decisions are always forced (mask has only that bit)."""
    record = play_game(game_id=0, seed=0, perturbation="canonical", max_turns=80)
    roll_dice_decisions = [d for d in record.decisions if d.action[0] == 12]  # ROLL_DICE
    if roll_dice_decisions:
        for d in roll_dice_decisions:
            assert d.forced is True


# ---------------------------------------------------------------------------
# generate_dataset end-to-end
# ---------------------------------------------------------------------------


def test_generate_dataset_writes_manifest_and_shards(tmp_path: Path) -> None:
    m = generate_dataset(
        out_dir=tmp_path,
        n_games=4,
        perturb_pct=0.50,
        shard_size=2,
        seed=0,
        max_turns=120,
        progress_every=10**9,  # disable progress prints
    )
    assert (tmp_path / "manifest.json").exists()
    on_disk = json.loads((tmp_path / "manifest.json").read_text())
    assert on_disk == m
    assert m["n_games"] == 4
    # 4 games at shard_size=2 → 2 shards.
    assert len(m["shards"]) == 2
    for shard in m["shards"]:
        assert (tmp_path / shard["shard"]).exists()


def test_generate_dataset_shard_npz_has_v2_schema_keys(tmp_path: Path) -> None:
    generate_dataset(
        out_dir=tmp_path,
        n_games=2,
        perturb_pct=0.0,
        shard_size=2,
        seed=0,
        max_turns=80,
        progress_every=10**9,
    )
    shard = np.load(tmp_path / "shard_0000.npz")
    expected_obs = {
        "tile_representations",
        "current_player_main",
        "next_player_main",
        "current_dev_counts",
        "next_played_dev_counts",
        "hex_features",
        "vertex_features",
        "edge_features",
        "opponent_kind",
        "opponent_policy_id",
    }
    obs_keys = {k[4:] for k in shard.files if k.startswith("obs/")}
    assert obs_keys == expected_obs

    expected_masks = {
        "type",
        "corner_settlement",
        "corner_city",
        "edge",
        "tile",
        "resource1_trade",
        "resource1_discard",
        "resource1_default",
        "resource2_default",
    }
    mask_keys = {k[5:] for k in shard.files if k.startswith("mask/")}
    assert mask_keys == expected_masks

    # Spot-check obs shapes.
    n = shard["action"].shape[0]
    assert shard["obs/tile_representations"].shape == (n, N_TILES, TILE_DIM)
    assert shard["obs/current_player_main"].shape == (n, CURR_PLAYER_DIM)
    assert shard["obs/next_player_main"].shape == (n, NEXT_PLAYER_DIM)
    assert shard["obs/current_dev_counts"].shape == (n, N_DEV_TYPES)
    assert shard["obs/vertex_features"].shape == (n, N_VERTICES, 16)
    assert shard["obs/edge_features"].shape == (n, N_EDGES, 16)
    assert shard["action"].shape == (n, 6)
    assert shard["belief_target"].shape == (n, N_DEV_TYPES)
    assert shard["z_disc"].shape == (n,)


def test_generate_dataset_filters_forced_moves(tmp_path: Path) -> None:
    """include_forced=False (default) must drop pairs where mask_type.sum()==1."""
    generate_dataset(
        out_dir=tmp_path,
        n_games=2,
        perturb_pct=0.0,
        shard_size=2,
        seed=0,
        max_turns=80,
        include_forced=False,
        progress_every=10**9,
    )
    shard = np.load(tmp_path / "shard_0000.npz")
    # Every kept pair has at least 2 legal type-mask bits.
    type_sums = shard["mask/type"].sum(axis=-1)
    assert (type_sums >= 2).all(), (
        f"forced pairs leaked through filter: min type-mask sum = {type_sums.min()}"
    )


def test_generate_dataset_include_forced_keeps_them(tmp_path: Path) -> None:
    generate_dataset(
        out_dir=tmp_path,
        n_games=2,
        perturb_pct=0.0,
        shard_size=2,
        seed=0,
        max_turns=80,
        include_forced=True,
        progress_every=10**9,
    )
    shard = np.load(tmp_path / "shard_0000.npz")
    type_sums = shard["mask/type"].sum(axis=-1)
    # With include_forced=True the dataset DOES contain forced pairs
    # (mask sum == 1, e.g. ROLL_DICE).
    assert (type_sums == 1).any()


def test_generate_dataset_perturbation_mix_respects_pct(tmp_path: Path) -> None:
    """With perturb_pct=1.0, every game must be perturbed."""
    m = generate_dataset(
        out_dir=tmp_path,
        n_games=8,
        perturb_pct=1.0,
        shard_size=8,
        seed=0,
        max_turns=60,
        progress_every=10**9,
    )
    counts = m["perturbation_counts"]
    assert counts["canonical"] == 0
    assert counts["epsilon_greedy"] + counts["weight_noised"] == 8


def test_generate_dataset_rejects_bad_perturb_pct(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        generate_dataset(out_dir=tmp_path, n_games=1, perturb_pct=1.5)
    with pytest.raises(ValueError):
        generate_dataset(out_dir=tmp_path, n_games=1, perturb_pct=-0.1)


def test_generate_dataset_manifest_records_provenance(tmp_path: Path) -> None:
    m = generate_dataset(
        out_dir=tmp_path,
        n_games=3,
        perturb_pct=0.30,
        shard_size=10,
        seed=42,
        max_turns=60,
        progress_every=10**9,
    )
    assert "run_id" in m and len(m["run_id"]) == 12
    assert "git_sha" in m
    assert m["seed"] == 42
    assert "wall_clock_seconds" in m and m["wall_clock_seconds"] > 0
    assert "forced_move_drop_pct" in m
    assert 0.0 <= m["forced_move_drop_pct"] < 1.0


def test_generate_dataset_no_game_id_duplication_across_shards(tmp_path: Path) -> None:
    m = generate_dataset(
        out_dir=tmp_path,
        n_games=6,
        perturb_pct=0.0,
        shard_size=2,
        seed=0,
        max_turns=50,
        progress_every=10**9,
    )
    all_game_ids: list[int] = []
    for shard in m["shards"]:
        all_game_ids.extend(shard["game_ids"])
    assert all_game_ids == list(range(6))  # contiguous, no dupes
