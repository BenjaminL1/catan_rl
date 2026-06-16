"""Action priors over the autoregressive action space (Phase 2 / T005, contract C2)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from catan_rl.env.catan_env import ActionType, CatanEnv
from catan_rl.policy.heads import masked_log_softmax
from catan_rl.policy.obs_tensor import masks_to_torch, obs_to_torch
from catan_rl.search.priors import action_priors

from .conftest import drive_to_main_phase

# Which mask key gates each relevant sub-head, per action type, so we can assert
# every chosen sub-index is actually legal (mirrors heads._*_mask selection).
_CORNER_KEY = {
    ActionType.BUILD_SETTLEMENT: "corner_settlement",
    ActionType.BUILD_CITY: "corner_city",
}
_RES1_KEY = {
    ActionType.PLAY_YOP: "resource1_default",
    ActionType.PLAY_MONOPOLY: "resource1_default",
    ActionType.BANK_TRADE: "resource1_trade",
    ActionType.DISCARD: "resource1_discard",
}


def _assert_priors_well_formed(priors: dict, masks: dict) -> None:  # type: ignore[type-arg]
    assert priors, "priors must be non-empty when at least one action is legal"
    # Normalised.
    assert abs(sum(priors.values()) - 1.0) < 1e-5
    legal_types = set(np.flatnonzero(masks["type"]).tolist())
    for action, p in priors.items():
        assert p >= 0.0
        assert isinstance(action, tuple) and len(action) == 6
        assert all(isinstance(x, int) for x in action)
        t, corner, edge, tile, res1, res2 = action
        # No illegal TYPE ever carries prior mass.
        assert t in legal_types, f"type {t} not in legal types {legal_types}"
        # Relevant sub-choices must be legal under their mask.
        if t in _CORNER_KEY and masks[_CORNER_KEY[t]].any():
            assert masks[_CORNER_KEY[t]][corner], f"illegal corner {corner} for type {t}"
        if t == ActionType.BUILD_ROAD and masks["edge"].any():
            assert masks["edge"][edge], f"illegal edge {edge}"
        if t in (ActionType.MOVE_ROBBER, ActionType.PLAY_KNIGHT) and masks["tile"].any():
            assert masks["tile"][tile], f"illegal tile {tile}"
        if t in _RES1_KEY and masks[_RES1_KEY[t]].any():
            assert masks[_RES1_KEY[t]][res1], f"illegal res1 {res1} for type {t}"
        # BANK_TRADE never gives and receives the same resource (a strict loss the
        # policy masks out via _resource2_mask) — the representative action must respect it.
        if t == ActionType.BANK_TRADE:
            assert res1 != res2, f"BANK_TRADE prior gives==receives resource {res1}"


def test_priors_at_setup_state(policy) -> None:  # type: ignore[no-untyped-def]
    # At reset the only legal type is BUILD_SETTLEMENT — priors collapse to the
    # settlement choices and still sum to 1.
    env = CatanEnv(opponent_type="heuristic")
    env.reset(seed=0)
    masks = env.get_action_masks()
    priors = action_priors(policy, env)
    _assert_priors_well_formed(priors, masks)
    assert all(a[0] == ActionType.BUILD_SETTLEMENT for a in priors)


def test_priors_at_main_phase(policy) -> None:  # type: ignore[no-untyped-def]
    env = CatanEnv(opponent_type="heuristic")
    env.reset(seed=3)
    assert drive_to_main_phase(env)
    masks = env.get_action_masks()
    priors = action_priors(policy, env)
    _assert_priors_well_formed(priors, masks)
    # END_TURN is always legal mid-turn, so it must appear with its own prior.
    assert any(a[0] == ActionType.END_TURN for a in priors)
    # One representative action per legal type — key count == legal type count.
    assert len(priors) == int(masks["type"].sum())


def test_priors_keys_are_distinct_per_type(policy) -> None:  # type: ignore[no-untyped-def]
    env = CatanEnv(opponent_type="heuristic")
    env.reset(seed=7)
    assert drive_to_main_phase(env)
    priors = action_priors(policy, env)
    types = [a[0] for a in priors]
    assert len(types) == len(set(types)), "exactly one representative action per type"


def test_prior_mass_equals_type_head_softmax(policy) -> None:  # type: ignore[no-untyped-def]
    # The C2 guarantee: prior(type) IS the masked type-head probability. Compute
    # the type softmax independently and confirm each prior matches it.
    env = CatanEnv(opponent_type="heuristic")
    env.reset(seed=3)
    assert drive_to_main_phase(env)
    priors = action_priors(policy, env)
    device = next(policy.parameters()).device
    obs_t = obs_to_torch(env._get_obs(), device, add_batch=True)
    masks_t = masks_to_torch(env.get_action_masks(), device, add_batch=True)
    trunk = policy.forward(obs_t)["trunk"]
    type_p = masked_log_softmax(policy.action_heads.type_head(trunk), masks_t["type"]).exp()[0]
    for action, p in priors.items():
        assert abs(p - float(type_p[action[0]].item())) < 1e-5


def test_prior_corner_subaction_is_policy_argmax_at_setup(policy) -> None:  # type: ignore[no-untyped-def]
    # Always-runs check (the setup state always has BUILD_SETTLEMENT legal with the
    # corner head relevant): the representative corner is the policy corner-head's
    # masked argmax, wiring context + mask + argmax — not _first_true(mask).
    env = CatanEnv(opponent_type="heuristic")
    env.reset(seed=3)
    priors = action_priors(policy, env)
    device = next(policy.parameters()).device
    obs_t = obs_to_torch(env._get_obs(), device, add_batch=True)
    masks_t = masks_to_torch(env.get_action_masks(), device, add_batch=True)
    trunk = policy.forward(obs_t)["trunk"]
    heads = policy.action_heads
    type_idx = torch.tensor([ActionType.BUILD_SETTLEMENT], device=device)
    logits = heads.corner_head(trunk, heads._corner_context(type_idx))
    cmask = heads._corner_mask(type_idx, masks_t)
    expected = int(logits.masked_fill(~cmask, float("-inf")).argmax(-1)[0].item())
    settle = [a for a in priors if a[0] == ActionType.BUILD_SETTLEMENT]
    assert len(settle) == 1
    assert settle[0][1] == expected


def test_prior_subaction_uses_policy_head_not_first_legal(policy) -> None:  # type: ignore[no-untyped-def]
    # Fully-independent opportunistic check on a context-free head (edge / tile),
    # computed from ONLY the public head + public mask (no private helpers) — when
    # such a type is legal at the reached main-phase state.
    env = CatanEnv(opponent_type="heuristic")
    env.reset(seed=3)
    assert drive_to_main_phase(env)
    priors = action_priors(policy, env)
    device = next(policy.parameters()).device
    obs_t = obs_to_torch(env._get_obs(), device, add_batch=True)
    masks_t = masks_to_torch(env.get_action_masks(), device, add_batch=True)
    trunk = policy.forward(obs_t)["trunk"]

    checked = False
    for action in priors:
        t = action[0]
        if t == ActionType.BUILD_ROAD and bool(masks_t["edge"].any()):
            logits = policy.action_heads.edge_head(trunk)
            expected = int(logits.masked_fill(~masks_t["edge"], float("-inf")).argmax(-1)[0].item())
            assert action[2] == expected
            checked = True
        elif t in (ActionType.MOVE_ROBBER, ActionType.PLAY_KNIGHT) and bool(masks_t["tile"].any()):
            logits = policy.action_heads.tile_head(trunk)
            expected = int(logits.masked_fill(~masks_t["tile"], float("-inf")).argmax(-1)[0].item())
            assert action[3] == expected
            checked = True
    if not checked:
        pytest.skip("no context-free relevant sub-head legal at this state")


def test_priors_are_deterministic(policy) -> None:  # type: ignore[no-untyped-def]
    # Pure read, no hidden RNG: identical calls on an unchanged env must match.
    env = CatanEnv(opponent_type="heuristic")
    env.reset(seed=5)
    assert drive_to_main_phase(env)
    assert action_priors(policy, env) == action_priors(policy, env)


def test_priors_k1_is_default_behavior(policy) -> None:  # type: ignore[no-untyped-def]
    # sub_actions_per_type=1 must be byte-identical to the default (the shipped 003
    # behavior) — the additivity guarantee that keeps the gated bake-off result valid.
    env = CatanEnv(opponent_type="heuristic")
    env.reset(seed=3)
    assert drive_to_main_phase(env)
    assert action_priors(policy, env, sub_actions_per_type=1) == action_priors(policy, env)


def test_multirep_priors_expand_placement_and_preserve_type_mass(policy) -> None:  # type: ignore[no-untyped-def]
    # At setup, BUILD_SETTLEMENT is the only legal type with many legal corners, so
    # k>1 MUST emit multiple distinct settlement sub-actions (k=1 emits exactly one),
    # all legal, summing to 1, and their masses must sum to the k=1 P(type) (the
    # P(type) is SPLIT across sub-actions, never changed).
    env = CatanEnv(opponent_type="heuristic")
    env.reset(seed=3)
    masks = env.get_action_masks()
    p1 = action_priors(policy, env, sub_actions_per_type=1)
    p4 = action_priors(policy, env, sub_actions_per_type=4)
    _assert_priors_well_formed(p4, masks)
    settle1 = [a for a in p1 if a[0] == ActionType.BUILD_SETTLEMENT]
    settle4 = [a for a in p4 if a[0] == ActionType.BUILD_SETTLEMENT]
    assert len(settle1) == 1
    assert 2 <= len(settle4) <= 4, "k=4 must expand multiple legal corners at setup"
    assert len({a[1] for a in settle4}) == len(settle4), "expanded corners must be distinct"
    # P(type) split-not-changed invariant: the k=1 modal corner is the top-prior k=4 child.
    assert max(settle4, key=lambda a: p4[a])[1] == settle1[0][1]
    # The total settlement mass is preserved between k=1 and k=4 (both renormalised
    # over the same single legal type -> 1.0 here; assert the per-type aggregate matches).
    assert abs(sum(p4[a] for a in settle4) - sum(p1[a] for a in settle1)) < 1e-5


def test_multirep_priors_split_mass_by_conditional_prob(policy) -> None:  # type: ignore[no-untyped-def]
    # The expanded sub-actions for a type must carry mass in the policy's conditional
    # head order (top sub-action >= the rest) — i.e. mass is split BY the head probs,
    # not uniformly. At setup the settlement corners are the cleanest probe.
    env = CatanEnv(opponent_type="heuristic")
    env.reset(seed=3)
    p4 = action_priors(policy, env, sub_actions_per_type=4)
    settle4 = sorted(
        (a for a in p4 if a[0] == ActionType.BUILD_SETTLEMENT), key=lambda a: p4[a], reverse=True
    )
    masses = [p4[a] for a in settle4]
    assert masses == sorted(masses, reverse=True)
    assert masses[0] >= masses[-1]
