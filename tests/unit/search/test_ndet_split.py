"""Fixed-budget n-det split + per-depth visit-concentration (spec 008 FR-003).

Guards:
  (a) both new knobs default OFF and the deployed search is byte-identical (chosen
      action + root visit_counts) to an all-flags-explicitly-off run (FR-008);
  (b) ``split_sims_across_determinizations`` makes K trees each run ``sims//K`` for
      a MATCHED total, vs ``K·sims`` when off;
  (c) ``collect_depth_stats`` adds a well-formed per-depth readout and is
      read-only — toggling it changes neither the chosen action nor visit_counts.
"""

from __future__ import annotations

import random

import numpy as np
import torch

from catan_rl.env.catan_env import CatanEnv
from catan_rl.search.config import SearchConfig
from catan_rl.search.mcts import MCTS

from .conftest import drive_to_decision


def _mcts(policy, cfg: SearchConfig) -> MCTS:  # type: ignore[no-untyped-def]
    from catan_rl.selfplay.snapshot_opponent import FrozenSnapshotOpponent

    device = next(policy.parameters()).device
    opp = FrozenSnapshotOpponent(policy, device=device, seed=cfg.seed)
    return MCTS(policy, cfg, opp, device)


def _seed(mcts: MCTS, cfg: SearchConfig) -> None:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    mcts.opponent.reset_rng(seed=cfg.seed)  # type: ignore[union-attr]


def _decided_env(seed: int = 3) -> CatanEnv:
    env = CatanEnv(opponent_type="heuristic")
    env.reset(seed=seed)
    assert drive_to_decision(env)
    return env


# --- (a) defaults off + all-off byte-identity ------------------------------


def test_new_knobs_default_off() -> None:
    cfg = SearchConfig(sims_per_move=16)
    assert cfg.split_sims_across_determinizations is False
    assert cfg.collect_depth_stats is False


def test_all_flags_off_is_byte_identical(policy) -> None:  # type: ignore[no-untyped-def]
    # The FR-008 all-flags-off regression: the default config and one with EVERY
    # new STAGE-A knob explicitly set to its off value must produce byte-identical
    # search output (chosen action + full root visit counts) at a fixed seed.
    default = SearchConfig(sims_per_move=24, seed=0)
    all_off = SearchConfig(
        sims_per_move=24,
        seed=0,
        final_move_mode="max_visit",
        split_sims_across_determinizations=False,
        collect_depth_stats=False,
    )
    m1 = _mcts(policy, default)
    _seed(m1, default)
    a1, d1 = m1.run(_decided_env())

    m2 = _mcts(policy, all_off)
    _seed(m2, all_off)
    a2, d2 = m2.run(_decided_env())

    assert tuple(a1) == tuple(a2)
    assert d1["visit_counts"] == d2["visit_counts"]
    assert "per_depth_concentration" not in d1


# --- (b) budget split matches the total across K ---------------------------


def test_split_matches_total_budget(policy) -> None:  # type: ignore[no-untyped-def]
    sims, k = 24, 3
    off = SearchConfig(sims_per_move=sims, seed=0, n_determinizations=k)
    on = SearchConfig(
        sims_per_move=sims, seed=0, n_determinizations=k, split_sims_across_determinizations=True
    )

    m_off = _mcts(policy, off)
    _seed(m_off, off)
    _, d_off = m_off.run(_decided_env())

    m_on = _mcts(policy, on)
    _seed(m_on, on)
    _, d_on = m_on.run(_decided_env())

    # OFF: each of K trees runs the FULL sims -> total K*sims.
    assert d_off["sims_run"] == sims * k
    # ON: each tree runs sims//K -> total ~sims (matched total budget).
    assert d_on["sims_run"] == (sims // k) * k


def test_split_noop_at_single_determinization(policy) -> None:  # type: ignore[no-untyped-def]
    # With n_determinizations=1 the split has nothing to split -> identical run.
    base = SearchConfig(sims_per_move=20, seed=0, n_determinizations=1)
    split = SearchConfig(
        sims_per_move=20, seed=0, n_determinizations=1, split_sims_across_determinizations=True
    )
    m1 = _mcts(policy, base)
    _seed(m1, base)
    a1, d1 = m1.run(_decided_env())
    m2 = _mcts(policy, split)
    _seed(m2, split)
    a2, d2 = m2.run(_decided_env())
    assert d1["sims_run"] == d2["sims_run"] == 20
    assert tuple(a1) == tuple(a2)
    assert d1["visit_counts"] == d2["visit_counts"]


# --- (c) per-depth concentration is well-formed + read-only ----------------


def test_depth_stats_well_formed_and_read_only(policy) -> None:  # type: ignore[no-untyped-def]
    off = SearchConfig(sims_per_move=32, seed=0)
    on = SearchConfig(sims_per_move=32, seed=0, collect_depth_stats=True)

    m_off = _mcts(policy, off)
    _seed(m_off, off)
    a_off, d_off = m_off.run(_decided_env())

    m_on = _mcts(policy, on)
    _seed(m_on, on)
    a_on, d_on = m_on.run(_decided_env())

    # Read-only: the diagnostic must not change the search decision or the tree.
    assert tuple(a_off) == tuple(a_on)
    assert d_off["visit_counts"] == d_on["visit_counts"]
    assert "per_depth_concentration" not in d_off

    # Well-formed: depths 0/1/2, collapse_frac in [0,1], n_nodes >= 0, and depth-0
    # has exactly one decision node (the root) when the search landed visits.
    pdc = d_on["per_depth_concentration"]
    assert set(pdc.keys()) == {0, 1, 2}
    for m in pdc.values():
        assert 0.0 <= m["collapse_frac"] <= 1.0
        assert m["n_nodes"] >= 0.0
    assert pdc[0]["n_nodes"] == 1.0
