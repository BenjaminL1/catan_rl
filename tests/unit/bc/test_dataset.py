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
    FORCED_RULE_VERSION,
    N_DEV_TYPES,
    N_EDGES,
    N_TILES,
    N_VERTICES,
    NEXT_PLAYER_DIM,
    RESOURCES_CW,
    RULESET_VERSION,
    TILE_DIM,
    ActionType,
    action_masked_legal,
    is_forced_decision,
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
        "global_features",
        "is_setup",
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
        "resource1_yop",
        "resource2_yop",
        "resource2_yop_same",
        "resource2_trade",
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
    """include_forced=False (default) must drop only GENUINELY forced pairs.

    D1: the criterion is the shared relevance-aware
    :func:`is_forced_decision`, not ``mask["type"].sum() == 1``. A singleton
    type mask over a wide-open downstream head (every setup placement, every
    robber placement) is a real decision and must survive.
    """
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
    n = int(shard["action"].shape[0])
    for i in range(n):
        row = {k[5:]: shard[k][i] for k in shard.files if k.startswith("mask/")}
        assert not is_forced_decision(row), f"forced pair leaked through filter at row {i}"
    # ... and the singleton-type rows that DO survive are the restored ones.
    type_sums = shard["mask/type"].sum(axis=-1)
    assert (type_sums == 1).any(), "relevance-aware filter kept no singleton-type decision"


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
    # D5: this used to read ``0.0 <= pct`` and passed VACUOUSLY on the shipped
    # manifest's dead 0.0 (the counters were never restored on resume). Some
    # decisions ARE forced (ROLL_DICE), so the rate is strictly positive, and
    # it must be the real ratio rather than a placeholder.
    pre = m["total_decisions_pre_filter"]
    post = m["total_decisions_post_filter"]
    assert pre > 0 and post > 0
    assert 0.0 < m["forced_move_drop_pct"] < 1.0
    assert m["forced_move_drop_pct"] == pytest.approx(1.0 - post / pre)
    # D5: version stamps must be present so the loader can refuse a stale dir.
    assert m["forced_rule_version"] == FORCED_RULE_VERSION
    assert m["ruleset_version"] == RULESET_VERSION


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


# ---------------------------------------------------------------------------
# Resumable / crash-tolerant generation
# ---------------------------------------------------------------------------


def test_resume_writes_manifest_and_progress_every_flush(tmp_path: Path) -> None:
    """A completed (or mid-run-killed) partial leaves loader-readable files.

    After each shard flush, both manifest.json and progress.json exist and
    cover the flushed shards — so a job that dies after the first shard is
    already usable, not orphaned.
    """
    generate_dataset(
        out_dir=tmp_path,
        n_games=4,
        perturb_pct=0.0,
        shard_size=2,
        seed=0,
        max_turns=60,
        progress_every=10**9,
    )
    assert (tmp_path / "manifest.json").exists()
    assert (tmp_path / "progress.json").exists()
    prog = json.loads((tmp_path / "progress.json").read_text())
    assert prog["games_done"] == 4
    assert prog["seed_base"] == 0
    assert prog["target_games"] == 4
    # The partial is loadable by the training loader.
    from catan_rl.bc.loader import BcDataset

    ds = BcDataset(tmp_path, aug_prob=0.0)
    assert len(ds) > 0


def test_resume_continues_numbering_and_generates_remainder(tmp_path: Path) -> None:
    m1 = generate_dataset(
        out_dir=tmp_path,
        n_games=4,
        perturb_pct=0.0,
        shard_size=2,
        seed=0,
        max_turns=60,
        progress_every=10**9,
    )
    assert len(m1["shards"]) == 2
    b0 = (tmp_path / "shard_0000.npz").read_bytes()
    b1 = (tmp_path / "shard_0001.npz").read_bytes()

    m2 = generate_dataset(
        out_dir=tmp_path,
        n_games=6,
        perturb_pct=0.0,
        shard_size=2,
        seed=0,
        max_turns=60,
        progress_every=10**9,
        resume=True,
    )
    # (b) new shard continues numbering; (b') existing shards never overwritten.
    assert (tmp_path / "shard_0002.npz").exists()
    assert not (tmp_path / "shard_0003.npz").exists()
    assert (tmp_path / "shard_0000.npz").read_bytes() == b0
    assert (tmp_path / "shard_0001.npz").read_bytes() == b1
    # (e) final manifest covers ALL shards; (a/c) only remainder generated.
    assert len(m2["shards"]) == 3
    gids: list[int] = []
    for s in m2["shards"]:
        gids.extend(s["game_ids"])
    assert sorted(gids) == list(range(6))  # contiguous 0..5, no dupes


def test_resume_generates_non_overlapping_games(tmp_path: Path) -> None:
    """Resumed games occupy a disjoint, deterministic slice of the run.

    We can't pin dice byte-for-byte (``StackedDice`` is entropy-seeded from
    the stdlib ``random`` state — a pre-existing property of the engine, not
    of resume), so we pin the two guarantees that actually prevent
    duplication: (1) the resumed shard's game_ids are disjoint from the
    already-done games and continue contiguously, and (2) the perturbation
    assignment — the only RNG stream ``generate_dataset`` itself owns — is
    deterministic in the game index, so the resumed run's final mix equals an
    uninterrupted run's mix rather than restarting the stream from 0.
    """
    d_full = tmp_path / "full"
    d_res = tmp_path / "resume"
    m_full = generate_dataset(
        out_dir=d_full,
        n_games=6,
        perturb_pct=1.0,  # exercise the perturbation RNG stream
        shard_size=2,
        seed=0,
        max_turns=60,
        progress_every=10**9,
    )
    # Simulate an interruption after 4 games, then resume to 6.
    generate_dataset(
        out_dir=d_res,
        n_games=4,
        perturb_pct=1.0,
        shard_size=2,
        seed=0,
        max_turns=60,
        progress_every=10**9,
    )
    m_res = generate_dataset(
        out_dir=d_res,
        n_games=6,
        perturb_pct=1.0,
        shard_size=2,
        seed=0,
        max_turns=60,
        progress_every=10**9,
        resume=True,
    )

    # (1) Non-overlapping game_ids: the newly written shard holds exactly the
    #     remainder, disjoint from the games already persisted.
    done_gids = {g for s in m_res["shards"][:2] for g in s["game_ids"]}
    new_gids = {g for s in m_res["shards"][2:] for g in s["game_ids"]}
    assert done_gids == {0, 1, 2, 3}
    assert new_gids == {4, 5}
    assert done_gids.isdisjoint(new_gids)

    # (2) The perturbation assignment is deterministic in the game index, so a
    #     resumed run reproduces the uninterrupted run's overall mix (it did
    #     NOT restart the stream from game 0 for the resumed games).
    assert m_res["perturbation_counts"] == m_full["perturbation_counts"]


def test_resume_reconstructs_when_progress_missing(tmp_path: Path) -> None:
    """Pre-resume shards that predate progress.json are handled: games_done
    is counted from the shard arrays, and a valid manifest is rebuilt."""
    generate_dataset(
        out_dir=tmp_path,
        n_games=4,
        perturb_pct=0.0,
        shard_size=2,
        seed=0,
        max_turns=60,
        progress_every=10**9,
    )
    # Simulate legacy partial: only the raw shards survive.
    (tmp_path / "progress.json").unlink()
    (tmp_path / "manifest.json").unlink()

    m = generate_dataset(
        out_dir=tmp_path,
        n_games=6,
        perturb_pct=0.0,
        shard_size=2,
        seed=0,
        max_turns=60,
        progress_every=10**9,
        resume=True,
    )
    assert len(m["shards"]) == 3
    gids: list[int] = []
    for s in m["shards"]:
        gids.extend(s["game_ids"])
    assert sorted(gids) == list(range(6))
    # Reconstructed + new shards are together loader-readable.
    from catan_rl.bc.loader import BcDataset

    train, val = BcDataset.train_val_split(tmp_path, val_pct=0.25)
    assert len(train) > 0 and len(val) > 0


def test_resume_restores_decision_counters_when_no_new_games_are_generated(
    tmp_path: Path,
) -> None:
    """D5 / AC-8. A resume that generates ZERO new games used to write
    ``total_decisions_pre_filter = 0`` and ``forced_move_drop_pct = 0.0`` over a
    fully-generated corpus, because the counters live only in memory. They must
    survive the resume boundary."""
    first = generate_dataset(
        out_dir=tmp_path,
        n_games=3,
        perturb_pct=0.0,
        shard_size=3,
        seed=7,
        max_turns=60,
        progress_every=10**9,
    )
    assert first["total_decisions_pre_filter"] > 0

    resumed = generate_dataset(
        out_dir=tmp_path,
        n_games=3,  # nothing left to do
        perturb_pct=0.0,
        shard_size=3,
        seed=7,
        max_turns=60,
        progress_every=10**9,
        resume=True,
    )
    assert resumed["total_decisions_pre_filter"] == first["total_decisions_pre_filter"]
    assert resumed["total_decisions_post_filter"] == first["total_decisions_post_filter"]
    assert resumed["forced_move_drop_pct"] == pytest.approx(first["forced_move_drop_pct"])
    assert resumed["forced_move_drop_pct"] > 0.0


def test_resume_counters_survive_the_reconstruct_path(tmp_path: Path) -> None:
    """The SLOW resume branch (sidecar absent or stale) must not zero the
    counters either. ``n_pairs`` recovers the post-filter total exactly; the
    pre-filter total is floored at it so the drop pct under-reports rather than
    claiming a 100% drop off a zeroed numerator."""
    first = generate_dataset(
        out_dir=tmp_path,
        n_games=4,
        perturb_pct=0.0,
        shard_size=2,
        seed=0,
        max_turns=60,
        progress_every=10**9,
    )
    (tmp_path / "progress.json").unlink()
    (tmp_path / "manifest.json").unlink()

    resumed = generate_dataset(
        out_dir=tmp_path,
        n_games=4,  # nothing left to generate — counters come purely from disk
        perturb_pct=0.0,
        shard_size=2,
        seed=0,
        max_turns=60,
        progress_every=10**9,
        resume=True,
    )
    assert resumed["total_decisions_post_filter"] == first["total_decisions_post_filter"]
    assert resumed["total_decisions_pre_filter"] >= resumed["total_decisions_post_filter"] > 0
    assert 0.0 <= resumed["forced_move_drop_pct"] < 1.0


def test_resume_without_existing_shards_is_fresh_run(tmp_path: Path) -> None:
    """--resume on an empty dir behaves exactly like a fresh run."""
    m = generate_dataset(
        out_dir=tmp_path,
        n_games=4,
        perturb_pct=0.0,
        shard_size=2,
        seed=0,
        max_turns=60,
        progress_every=10**9,
        resume=True,
    )
    assert (tmp_path / "shard_0000.npz").exists()
    assert len(m["shards"]) == 2
    gids: list[int] = []
    for s in m["shards"]:
        gids.extend(s["game_ids"])
    assert sorted(gids) == list(range(4))


# ---------------------------------------------------------------------------
# D7 / AC-3 — SHARD-LEVEL coverage.
#
# The pre-existing phase test (``test_play_game_setup_actions_tagged_setup_phase``)
# asserts at ``play_game`` level, BEFORE ``_flatten_records`` applies the
# write-time ``forced`` filter — so it stayed green while 100% of setup rows
# were being dropped on the way to disk. These assertions read what actually
# LANDED in the shard.
# ---------------------------------------------------------------------------


def _load_shard_columns(tmp_path: Path) -> tuple[np.ndarray, np.ndarray]:
    shard = np.load(tmp_path / "shard_0000.npz")
    return shard["phase"], shard["action"]


def test_shard_contains_setup_and_robber_rows_and_no_discard(tmp_path: Path) -> None:
    generate_dataset(
        out_dir=tmp_path,
        n_games=6,
        perturb_pct=0.0,
        shard_size=6,
        seed=100,
        max_turns=200,
        progress_every=10**9,
    )
    phase, action = _load_shard_columns(tmp_path)
    phases = {str(p) for p in phase.tolist()}
    assert {"setup", "main", "robber"} <= phases, f"missing decision class on disk: {phases}"

    types = set(action[:, 0].tolist())
    # Core placement / turn vocabulary must all be represented.
    assert {0, 1, 2, 3, 4, 5, 10} <= types, f"type histogram gaps: {sorted(types)}"
    # D1: DISCARD rows are deliberately never written — the recorder has no
    # per-card intercept, so every such row would carry a fabricated WOOD label
    # from a teacher that discards uniformly at random.
    assert ActionType.DISCARD not in types
    assert "discard" not in phases


def test_shard_contains_dev_card_plays(tmp_path: Path) -> None:
    """D2: the corpus used to hold ~50k BUY_DEV_CARD rows and ZERO plays."""
    generate_dataset(
        out_dir=tmp_path,
        n_games=6,
        perturb_pct=0.0,
        shard_size=6,
        seed=100,
        max_turns=200,
        progress_every=10**9,
    )
    _phase, action = _load_shard_columns(tmp_path)
    types = set(action[:, 0].tolist())
    play_types = {
        ActionType.PLAY_KNIGHT,
        ActionType.PLAY_YOP,
        ActionType.PLAY_MONOPOLY,
        ActionType.PLAY_ROAD_BUILDER,
    }
    assert types & play_types, "teacher still never plays a development card"


# ---------------------------------------------------------------------------
# D2 — the pre-roll dev-card window must contribute NO rows
# ---------------------------------------------------------------------------


def _phase_flag_indices() -> tuple[int, int]:
    """Locate the ``roll_pending`` / ``robber_placement_pending`` slots inside
    ``current_player_main`` by toggling them on the shared encoder, so this
    test does not hardcode an offset that ``obs_encoder`` may re-pack."""
    import queue as _queue

    from catan_rl.agents.heuristic import heuristicAIPlayer
    from catan_rl.engine.game import catanGame
    from catan_rl.env.hand_tracker import BroadcastHandTracker
    from catan_rl.policy.obs_encoder import EnvObsState, ObsEncoder

    game = catanGame(render_mode=None)
    p1 = heuristicAIPlayer("P1", "black")
    p2 = heuristicAIPlayer("P2", "darkslateblue")
    p1.updateAI()
    p2.updateAI()
    game.playerQueue = _queue.Queue(2)
    game.playerQueue.put(p1)
    game.playerQueue.put(p2)
    encoder = ObsEncoder(game.board)
    tracker = BroadcastHandTracker([p1.name, p2.name])
    tracker.subscribe(game.broadcast)

    def obs_for(**flags: bool) -> np.ndarray:
        state = EnvObsState(initial_placement_phase=False, **flags)
        return np.asarray(
            encoder.build_obs(game, p1, p2, state, hand_tracker=tracker)["current_player_main"]
        )

    base = obs_for()
    # Each toggle moves TWO slots: its own flag and the derived ``in_main``
    # flag. ``in_main`` is the one they share, so subtract the intersection.
    roll_diff = set(np.flatnonzero(obs_for(roll_pending=True) != base).tolist())
    robber_diff = set(np.flatnonzero(obs_for(robber_placement_pending=True) != base).tolist())
    shared = roll_diff & robber_diff  # in_main
    (roll,) = roll_diff - shared
    (robber,) = robber_diff - shared
    return int(roll), int(robber)


def test_no_row_claims_two_phases_at_once() -> None:
    """The pre-roll Knight used to drag a robber placement into the corpus with
    ``roll_pending=1`` AND ``robber_placement_pending=1`` in the same obs — a
    state the learner's env can never occupy (the mask builder checks the
    robber branch BEFORE the roll branch, so the row was mask-legal). Measured
    at 7.2% of robber rows before the explicit ``suppress_recording`` gate."""
    roll_idx, robber_idx = _phase_flag_indices()
    seen_robber = 0
    for seed in range(8):
        record = play_game(game_id=seed, seed=seed, perturbation="canonical", max_turns=200)
        for d in record.decisions:
            main = np.asarray(d.obs["current_player_main"])
            seen_robber += int(main[robber_idx] > 0)
            assert not (main[roll_idx] > 0 and main[robber_idx] > 0), (
                f"seed {seed}: row claims roll_pending AND robber_placement_pending"
            )
    assert seen_robber > 0, "no robber rows generated — the assertion would be vacuous"


# ---------------------------------------------------------------------------
# PLAY_KNIGHT tile label (F-A) + write-time relevance-aware legality (F-B)
# ---------------------------------------------------------------------------


def _knight_rows(n_games: int = 8, seed0: int = 100) -> list:  # type: ignore[type-arg]
    rows = []
    for seed in range(seed0, seed0 + n_games):
        record = play_game(game_id=seed, seed=seed, perturbation="canonical", max_turns=300)
        rows.extend(
            (seed, i, d)
            for i, d in enumerate(record.decisions)
            if int(d.action[0]) == ActionType.PLAY_KNIGHT
        )
    return rows


def test_play_knight_row_carries_a_populated_tile_mask() -> None:
    """A knight row's tile head is RELEVANT (it triggers the robber move), so
    its label must be a real hex under a populated tile mask. The row used to
    be written with ``tile_idx`` defaulted to 0 and an all-False tile mask,
    because the mask was built before ``robber_placement_pending`` was set."""
    rows = _knight_rows()
    assert rows, "no PLAY_KNIGHT rows generated — the assertion would be vacuous"
    for seed, i, d in rows:
        assert d.mask["tile"].any(), f"seed {seed} row {i}: empty tile mask on a knight row"
        assert d.mask["tile"][int(d.action[3])], (
            f"seed {seed} row {i}: knight tile {int(d.action[3])} is off the tile mask"
        )


def test_play_knight_tile_matches_the_robber_move_it_triggers() -> None:
    """The knight's recorded hex is the hex the teacher actually robs — i.e.
    the label agrees with the MOVE_ROBBER row the knight drags in for the same
    seat. Pins the purity of ``choose_player_to_rob``: a future stochastic
    override would break this loudly rather than silently mislabel the corpus."""
    checked = 0
    for seed in range(100, 108):
        record = play_game(game_id=seed, seed=seed, perturbation="canonical", max_turns=300)
        for i, d in enumerate(record.decisions):
            if int(d.action[0]) != ActionType.PLAY_KNIGHT:
                continue
            nxt = next(
                (
                    e
                    for e in record.decisions[i + 1 :]
                    if e.player_seat == d.player_seat and int(e.action[0]) == ActionType.MOVE_ROBBER
                ),
                None,
            )
            assert nxt is not None, f"seed {seed} row {i}: knight recorded no robber move"
            assert int(d.action[3]) == int(nxt.action[3]), (
                f"seed {seed} row {i}: knight tile {int(d.action[3])} != "
                f"robbed tile {int(nxt.action[3])}"
            )
            checked += 1
    assert checked > 0, "no PLAY_KNIGHT rows generated — the assertion would be vacuous"


def test_play_knight_rows_survive_the_relevance_gate() -> None:
    """Regression guard on the F-A -> F-B ordering: the write-time legality
    gate now checks the tile head for PLAY_KNIGHT, so if the tile override were
    applied AFTER the gate (or not at all) every knight row would be dropped."""
    assert _knight_rows(), "the relevance-aware gate swallowed every PLAY_KNIGHT row"


def test_every_recorded_row_is_legal_under_its_own_mask() -> None:
    """No row may carry an index that its own mask forbids on a RELEVANT head.

    Scope note, so this is not over-read: on the canonical teacher this is red
    against the PRE-F-A tree (the fabricated ``tile_idx = 0`` knight rows are
    off their own tile mask) and it is an ongoing corpus-wide invariant — but
    it does NOT discriminate F-B. The canonical teacher rejects zero rows on a
    sub-head, so the type-head-only gate would leave this green too. The
    discriminating F-B pin is
    ``test_bank_trade_with_unsupplyable_receive_is_not_recorded``.
    """
    seen = 0
    for seed in range(200, 206):
        record = play_game(game_id=seed, seed=seed, perturbation="canonical", max_turns=300)
        for i, d in enumerate(record.decisions):
            assert action_masked_legal(d.mask, d.action), (
                f"seed {seed} row {i}: action {d.action.tolist()} is off its own mask"
            )
            seen += 1
    assert seen > 0


def _record_one_bank_trade(give: str, receive: str, drained: str) -> tuple[list, dict]:  # type: ignore[type-arg]
    """Drive ONE ``trade_with_bank`` call through the recorder; return ``(records, mask)``.

    The player holds 6 of ``give`` (a 4:1 trade is affordable) and nothing else;
    the bank is drained of ``drained`` ALONE, so every other resource is still
    supplyable and the TYPE head therefore still offers ``BANK_TRADE``. That is
    what makes the drop discriminating: draining the WHOLE bank would close the
    type head, and the old type-head-only gate would have rejected the row too.
    """
    import queue as _queue

    from catan_rl.agents.heuristic import heuristicAIPlayer
    from catan_rl.bc.dataset import (
        _build_index_maps,
        _instrumented_player,
        _RecorderContext,
    )
    from catan_rl.engine.game import catanGame
    from catan_rl.env.hand_tracker import BroadcastHandTracker
    from catan_rl.env.masks import compute_action_masks
    from catan_rl.policy.obs_encoder import EnvObsState, ObsEncoder

    np.random.seed(0)
    game = catanGame(render_mode=None)
    board = game.board

    p1 = heuristicAIPlayer("P1", None)
    p2 = heuristicAIPlayer("P2", None)
    p1.game = game
    p2.game = game
    game.playerQueue = _queue.Queue(2)
    game.playerQueue.put(p1)
    game.playerQueue.put(p2)
    encoder = ObsEncoder(board)
    tracker = BroadcastHandTracker([p1.name, p2.name])
    tracker.subscribe(game.broadcast)
    vertex_to_idx, edge_to_idx = _build_index_maps(board)

    for r in p1.resources:
        p1.resources[r] = 0
    p1.resources[give] = 6
    board.resourceBank[drained] = 0

    env_state = EnvObsState(initial_placement_phase=False)
    ctx = _RecorderContext(
        game=game,
        agent_player=p1,
        opp_player=p2,
        encoder=encoder,
        hand_tracker=tracker,
        vertex_to_idx=vertex_to_idx,
        edge_to_idx=edge_to_idx,
        records=[],
        seat=0,
        env_state=env_state,
    )
    mask = compute_action_masks(game, p1, env_state, vertex_to_idx, edge_to_idx)
    with _instrumented_player(ctx):
        p1.trade_with_bank(give, receive, board)
    return ctx.records, mask


def test_bank_trade_with_unsupplyable_receive_is_not_recorded() -> None:
    """Drain the bank of the resource the teacher would RECEIVE — and only that
    one — and the row must be DROPPED.

    The single-resource drain is the whole point: ``type[BANK_TRADE]`` stays
    True (the give is still tradeable for the four resources the bank can still
    supply), so ``resource2_trade[BRICK]`` is the ONLY head that says no. A
    type-head-only gate records this row; the relevance-aware gate drops it.
    """
    records, mask = _record_one_bank_trade(give="WOOD", receive="BRICK", drained="BRICK")
    brick = RESOURCES_CW.index("BRICK")
    assert bool(mask["type"][ActionType.BANK_TRADE]), (
        "type head closed — the drain was too broad, so this test no longer "
        "discriminates against the old type-head-only gate"
    )
    assert not bool(mask["resource2_trade"][brick]), "BRICK receive should be off resource2_trade"
    assert records == [], "recorded a bank trade whose receive the bank cannot supply"


def test_bank_trade_with_supplyable_receive_is_still_recorded() -> None:
    """Positive control for the gate above: same drained bank, but a receive the
    bank CAN supply survives. Without this, the drop test could pass by the gate
    rejecting every bank trade."""
    records, mask = _record_one_bank_trade(give="WOOD", receive="WHEAT", drained="BRICK")
    wheat = RESOURCES_CW.index("WHEAT")
    assert bool(mask["resource2_trade"][wheat])
    assert len(records) == 1, "a fully supplyable bank trade must survive the legality gate"
    assert int(records[0].action[0]) == ActionType.BANK_TRADE
