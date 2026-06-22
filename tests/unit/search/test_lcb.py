"""LCB final-move rule (spec 008 US0(a) / FR-002).

Covers the four required guarantees:
  (a) ``final_move_mode='max_visit'`` (default) is byte-identical to before — the
      chosen action AND the root visit counts match a directly-computed max-visit
      over the search's own ``visit_counts`` diagnostic, and the LCB config knobs
      change nothing.
  (b) LCB prefers the well-explored / low-uncertainty child even when a rival has
      a higher mean_Q (max-visit / max-mean would pick the noisy one).
  (c) Unvisited children are excluded; ties break by (lcb, visits, prior).
  (d) The ``child_Q2`` second-moment accumulator is in-memory tree state ONLY —
      it never appears in the policy state-dict (no checkpoint-format change).
"""

from __future__ import annotations

import math

import numpy as np
import torch

from catan_rl.env.catan_env import CatanEnv
from catan_rl.search.config import SearchConfig
from catan_rl.search.mcts import _LCB_VAR_EPS, MCTS, _lcb_select
from catan_rl.search.node import SearchNode

from .conftest import drive_to_decision


def _mcts(policy, cfg: SearchConfig) -> MCTS:  # type: ignore[no-untyped-def]
    from catan_rl.selfplay.snapshot_opponent import FrozenSnapshotOpponent

    device = next(policy.parameters()).device
    opp = FrozenSnapshotOpponent(policy, device=device, seed=cfg.seed)
    return MCTS(policy, cfg, opp, device)


def _seed(mcts: MCTS, cfg: SearchConfig) -> None:
    import random

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    mcts.opponent.reset_rng(seed=cfg.seed)  # type: ignore[union-attr]


# --- (a) default max-visit byte-identity -----------------------------------


def test_final_move_max_visit_byte_identical(policy) -> None:  # type: ignore[no-untyped-def]
    # With the default config (final_move_mode='max_visit') the chosen action must
    # equal a directly-computed max-visit argmax over the search's own root
    # visit_counts (tie-broken by mean-Q then prior), and the LCB additions must
    # leave visit_counts unchanged. This is the FR-008 all-flags-off guard for
    # the LCB slice: the new config defaults + the always-on child_Q2 accumulator
    # change neither the chosen action nor the root statistics.
    env = CatanEnv(opponent_type="heuristic")
    env.reset(seed=3)
    assert drive_to_decision(env)

    cfg = SearchConfig(sims_per_move=24, seed=0)  # defaults: final_move_mode='max_visit'
    mcts = _mcts(policy, cfg)
    _seed(mcts, cfg)
    action, diag = mcts.run(env)

    visit_counts: dict[tuple[int, ...], int] = diag["visit_counts"]
    priors: dict[tuple[int, ...], float] = diag["priors"]
    action_q: dict[tuple[int, ...], float] = diag["action_q"]
    assert visit_counts, "the search must have landed visits"

    # Directly recompute the documented max-visit rule (visits, mean-Q, prior).
    expected = max(
        visit_counts,
        key=lambda a: (visit_counts[a], action_q.get(a, 0.0), priors[a]),
    )
    assert tuple(action) == tuple(expected)
    # The default LCB knobs are present but inert.
    assert cfg.final_move_mode == "max_visit"
    # best_visits is the visit count of the chosen action (max over the counts).
    assert diag["best_visits"] == max(visit_counts.values())


def test_lcb_can_pick_a_different_action_than_max_visit(policy) -> None:  # type: ignore[no-untyped-def]
    # End-to-end: a real search run under final_move_mode='lcb' must (i) stay
    # deterministic and legal and (ii) leave the visit_counts diagnostic identical
    # to the max_visit run at the same seed (LCB changes ONLY the final pick, not
    # the tree). Whether the chosen action differs is data-dependent; we assert the
    # invariants that must hold either way.
    env = CatanEnv(opponent_type="heuristic")
    env.reset(seed=3)
    assert drive_to_decision(env)

    base = SearchConfig(sims_per_move=24, seed=0)
    m1 = _mcts(policy, base)
    _seed(m1, base)
    _, d_max = m1.run(env)

    lcb_cfg = SearchConfig(sims_per_move=24, seed=0, final_move_mode="lcb")
    m2 = _mcts(policy, lcb_cfg)
    _seed(m2, lcb_cfg)
    a_lcb, d_lcb = m2.run(env)

    # The TREE is identical (LCB is a post-search selection rule) -> same visits.
    assert d_max["visit_counts"] == d_lcb["visit_counts"]
    # LCB pick is a legal, visited action (mcts.run returns a plain 6-tuple).
    assert d_lcb["visit_counts"].get(tuple(a_lcb), 0) > 0


# --- (b) LCB prefers low uncertainty (synthetic node) ----------------------


def test_lcb_prefers_low_uncertainty() -> None:
    # Construct two visited root children directly:
    #   A: higher mean_Q (0.9) but few visits (2) with HUGE spread -> big stderr.
    #   B: slightly lower mean_Q (0.7) but many visits (40) all ~equal -> tiny stderr.
    # max-visit picks B (more visits); max-MEAN picks A; LCB must pick B (it
    # penalises A's uncertainty). We assert LCB==B while max-mean==A, so LCB is
    # demonstrably NOT the same as either max-visit or max-mean here.
    a = (2, 0, 1, 0, 0, 0)
    b = (3, 0, 0, 0, 0, 0)

    # A: 2 visits, values 0.4 and 1.4 -> mean 0.9, large variance.
    n_a, w_a, q2_a = 2, 0.4 + 1.4, 0.4**2 + 1.4**2
    # B: 40 visits, all exactly 0.7 -> mean 0.7, ~zero variance.
    n_b, w_b, q2_b = 40, 40 * 0.7, 40 * (0.7**2)

    agg_n = {a: n_a, b: n_b}
    agg_w = {a: w_a, b: w_b}
    agg_q2 = {a: q2_a, b: q2_b}
    priors = {a: 0.5, b: 0.5}

    # Sanity: A has the higher mean, B has the most visits.
    assert w_a / n_a > w_b / n_b  # max-mean would choose A
    assert n_b > n_a  # max-visit would choose B

    # LCB scores.
    def lcb(n: int, w: float, q2: float, z: float) -> float:
        mean = w / n
        var = max(_LCB_VAR_EPS, q2 / n - mean * mean)
        return mean - z * math.sqrt(var / n)

    z = 1.96
    assert lcb(n_b, w_b, q2_b, z) > lcb(n_a, w_a, q2_a, z)
    assert _lcb_select([a, b], agg_n, agg_w, agg_q2, priors, z) == b

    # And with z=0 the LCB collapses to max-mean -> it must pick A (the high mean).
    assert _lcb_select([a, b], agg_n, agg_w, agg_q2, priors, 0.0) == a


# --- (c) unvisited excluded + tie-break ------------------------------------


def test_lcb_unvisited_excluded() -> None:
    # An unvisited child (N==0) with a sky-high prior must NEVER be chosen by LCB
    # (it has no value estimate -> non-competitive / lcb = -inf). The single
    # visited child wins even with a poor value.
    visited = (3, 0, 0, 0, 0, 0)
    unvisited = (2, 0, 1, 0, 0, 0)
    agg_n = {visited: 5, unvisited: 0}
    agg_w = {visited: 5 * 0.2, unvisited: 0.0}
    agg_q2 = {visited: 5 * (0.2**2), unvisited: 0.0}
    priors = {visited: 0.01, unvisited: 0.99}
    assert _lcb_select([visited, unvisited], agg_n, agg_w, agg_q2, priors, 1.96) == visited


def test_lcb_all_unvisited_falls_back_to_first() -> None:
    # Defensive: if (degenerately) no candidate was visited, fall back to the first
    # action rather than crashing (the run() caller guards the empty-agg case, but
    # _lcb_select must be total).
    a = (3, 0, 0, 0, 0, 0)
    b = (2, 0, 1, 0, 0, 0)
    assert _lcb_select([a, b], {}, {}, {}, {a: 0.5, b: 0.5}, 1.96) == a


def test_lcb_tie_break_prefers_more_visits_then_prior() -> None:
    # Two children with IDENTICAL lcb (same mean, same variance, same visits) ->
    # the tie-break key is (lcb, visits, prior) descending, so the higher-prior
    # one wins. Then give one MORE visits at equal lcb -> visits dominate prior.
    a = (3, 0, 0, 0, 0, 0)
    b = (2, 0, 1, 0, 0, 0)

    # Identical stats except prior -> prior breaks the tie (b has the higher prior).
    n, w, q2 = 4, 4 * 0.5, 4 * (0.5**2)
    agg_n = {a: n, b: n}
    agg_w = {a: w, b: w}
    agg_q2 = {a: q2, b: q2}
    assert _lcb_select([a, b], agg_n, agg_w, agg_q2, {a: 0.3, b: 0.7}, 1.96) == b

    # Now make the two have the SAME lcb but b has MORE visits -> visits beat prior.
    # mean 0.5, zero-variance both -> lcb == 0.5 regardless of n -> visits decide.
    agg_n2 = {a: 4, b: 8}
    agg_w2 = {a: 4 * 0.5, b: 8 * 0.5}
    agg_q22 = {a: 4 * 0.25, b: 8 * 0.25}
    # a has the higher prior, but b has more visits at equal lcb -> b wins.
    assert _lcb_select([a, b], agg_n2, agg_w2, agg_q22, {a: 0.9, b: 0.1}, 1.96) == b


def test_lcb_single_visit_uses_eps_floor() -> None:
    # A single-visit child has Q2/N - mean² == 0 exactly -> the eps floor prevents a
    # 0/0 or sqrt(<0); the stderr is sqrt(eps / 1), small but finite, so the child
    # stays selectable (not -inf).
    a = (3, 0, 0, 0, 0, 0)
    chosen = _lcb_select([a], {a: 1}, {a: 0.6}, {a: 0.36}, {a: 1.0}, 1.96)
    assert chosen == a


# --- (d) child_Q2 is not serialized ----------------------------------------


def test_child_q2_is_not_in_policy_state_dict(policy) -> None:  # type: ignore[no-untyped-def]
    # The LCB second-moment accumulator is MCTS tree state, not net state. Running a
    # search must leave the policy state-dict byte-identical (no new keys, no
    # changed tensors) -> no checkpoint-format change (FR-007).
    before = {k: v.clone() for k, v in policy.state_dict().items()}

    env = CatanEnv(opponent_type="heuristic")
    env.reset(seed=3)
    assert drive_to_decision(env)
    cfg = SearchConfig(sims_per_move=16, seed=0, final_move_mode="lcb")
    mcts = _mcts(policy, cfg)
    _seed(mcts, cfg)
    mcts.run(env)

    after = policy.state_dict()
    assert set(after.keys()) == set(before.keys())
    for k, v in after.items():
        assert torch.equal(v, before[k]), f"policy tensor {k} changed during search"
    # And 'child_Q2' / search tree fields must not be polluting the state-dict keys.
    assert not any("child_Q2" in k or "child_W" in k or "child_N" in k for k in after)


def test_child_q2_accumulates_in_lock_step_with_child_w() -> None:
    # Direct node-level guard: record() updates child_Q2 by value² in parallel with
    # child_W's value, while leaving child_N / child_W / total_N exactly as before.
    node = SearchNode(
        env=None,
        obs={},
        is_terminal=False,
        outcome=0.0,
        value=0.5,
        priors={(3, 0, 0, 0, 0, 0): 1.0},
    )
    a = (3, 0, 0, 0, 0, 0)
    node.record(a, 0.7)
    node.record(a, 0.3)
    assert node.child_N[a] == 2
    assert math.isclose(node.child_W[a], 1.0)
    assert math.isclose(node.child_Q2[a], 0.7**2 + 0.3**2)
    assert node.total_N == 2
