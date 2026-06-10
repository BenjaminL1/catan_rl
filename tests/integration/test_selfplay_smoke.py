"""T029 — self-play integration smoke (US3 end-to-end).

Runs real PPO updates with ``snapshot_weight > 0`` and a snapshot that enters the
pool mid-run, exercising the whole self-play path together:
``League.add_snapshot`` → ``build_env_opponent_assignments`` →
``SerialVecEnv.set_opponents`` → a per-env ``FrozenSnapshotOpponent`` (the
review BLOCKER fix) → the in-env snapshot-opponent turn-driver. Also asserts the
1v1 ruleset is unchanged by the feature (FR-009).
"""

from __future__ import annotations

import logging
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from catan_rl.env.catan_env import CatanEnv
from catan_rl.eval.harness import EvalHarness
from catan_rl.policy.board_geometry import build_geometry
from catan_rl.policy.network import CatanPolicy
from catan_rl.ppo.arguments import TrainConfig
from catan_rl.ppo.training_loop import TrainingState, run_training


def _selfplay_cfg(total_updates: int = 3) -> TrainConfig:
    cfg = TrainConfig.default()
    return replace(
        cfg,
        rollout=replace(cfg.rollout, n_envs=2, n_steps=8, max_turns=25),
        ppo=replace(cfg.ppo, batch_size=4, n_epochs=1, target_kl=0.0),
        checkpoint=replace(cfg.checkpoint, save_every_updates=10_000),
        eval=replace(cfg.eval, eval_every_updates=10_000, eval_games=2),
        league=replace(
            cfg.league,
            add_snapshot_every_n_updates=1,  # snapshot enters the pool at update 1
            maxlen=4,
            heuristic_weight=0.5,  # PG-2 floor kept > 0
            snapshot_weight=0.5,  # self-play ON
        ),
        total_steps=total_updates * 2 * 8,
    )


def test_selfplay_smoke_runs_without_error(tmp_path: Path) -> None:
    log = logging.getLogger("catan_rl.train.selfplay_smoke")
    log.setLevel(logging.CRITICAL)
    state = run_training(
        _selfplay_cfg(total_updates=3),
        run_dir=tmp_path,
        device_label="cpu",
        logger=log,
        max_updates=3,
        open_tb=False,
    )
    assert isinstance(state, TrainingState)
    assert state.update_idx == 3
    # A snapshot was added (update 1) and entered play on later rollouts via the
    # per-rollout refresh + resolver — if that path were broken the run would
    # have raised, not completed.
    assert state.league.n_snapshots() > 0


def _pfsp_cfg(total_updates: int = 3) -> TrainConfig:
    # Tiny max_turns so games actually FINISH within the rollout (multiple per
    # env), exercising the real outcome→opponent attribution + the B2 multi-reset
    # case. Pure-snapshot pool so the snapshot is reliably assigned + recorded.
    cfg = TrainConfig.default()
    return replace(
        cfg,
        rollout=replace(cfg.rollout, n_envs=4, n_steps=64, max_turns=6),
        ppo=replace(cfg.ppo, batch_size=8, n_epochs=1, target_kl=0.0),
        checkpoint=replace(cfg.checkpoint, save_every_updates=10_000),
        eval=replace(cfg.eval, eval_every_updates=10_000, eval_games=2),
        league=replace(
            cfg.league,
            add_snapshot_every_n_updates=1,
            maxlen=4,
            heuristic_weight=0.0,
            snapshot_weight=1.0,
            require_heuristic_floor=False,
            pfsp_enabled=True,
            pfsp_curve="hard",
            pfsp_min_games=1,
        ),
        total_steps=total_updates * 4 * 64,
    )


def test_pfsp_smoke_records_outcomes(tmp_path: Path) -> None:
    """PFSP end-to-end: a run with pfsp_enabled attributes finished games to
    the snapshot they faced, so the league's win-rate store accumulates games."""
    log = logging.getLogger("catan_rl.train.pfsp_smoke")
    log.setLevel(logging.CRITICAL)
    state = run_training(
        _pfsp_cfg(total_updates=3),
        run_dir=tmp_path,
        device_label="cpu",
        logger=log,
        max_updates=3,
        open_tb=False,
    )
    assert isinstance(state, TrainingState)
    # Outcomes were attributed: at least one snapshot accrued recorded games.
    total_games = sum(g for _p, g in state.league.opponent_stats_state().values())
    assert total_games > 0, "PFSP recorded no game outcomes — attribution path broken"


def test_anchor_weight_without_anchor_fails_fast(tmp_path: Path) -> None:
    """FN2: anchor_weight>0 with no anchor installed (no anchor_checkpoint_path,
    no manual set_anchor) must raise at run start, not silently renormalize the
    anchor's 25% away leaving the drift guard inert."""
    base = _selfplay_cfg(total_updates=1)
    cfg = replace(base, league=replace(base.league, anchor_weight=0.25))
    log = logging.getLogger("catan_rl.train.anchor_failfast")
    log.setLevel(logging.CRITICAL)
    with pytest.raises(ValueError, match="anchor"):
        run_training(
            cfg, run_dir=tmp_path, device_label="cpu", logger=log, max_updates=1, open_tb=False
        )


def test_selfplay_preserves_1v1_ruleset() -> None:
    """FR-009: the engine's 1v1 ruleset constants are untouched, and the
    eval/rules_invariants gate finds no violations across played games."""
    env = CatanEnv(opponent_type="heuristic")
    env.reset(seed=0)
    assert env.game is not None
    assert env.game.maxPoints == 15  # 1v1 win condition
    assert env.game.numPlayers == 2

    # The harness audits run_all_invariants on every game it plays.
    policy = CatanPolicy()
    policy.set_board_geometry(build_geometry().as_dict_of_tensors())
    harness = EvalHarness(
        opponent_types=("heuristic",),
        n_games_per_seat=1,
        seed=0,
        device=torch.device("cpu"),
        max_turns=25,
        audit_rules=True,
    )
    report = harness.run(policy)
    assert report.results[0].rules_violations == ()
