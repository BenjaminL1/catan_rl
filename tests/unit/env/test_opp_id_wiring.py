"""T009 — feed the fixed-size _OppIdEmbedding real opponent kind/id (FR-008).

A frozen league snapshot maps to the existing ``OPP_KIND_LEAGUE`` slot and a
concrete ``opponent_policy_id``; these flow into the obs and through the policy
WITHOUT resizing any embedding — so the in-flight ``bootstrap_v1`` checkpoint
stays loadable (Constitution III).
"""

from __future__ import annotations

import numpy as np
import torch

from catan_rl.env.catan_env import CatanEnv, _kind_from_opp_type
from catan_rl.policy.board_geometry import build_geometry
from catan_rl.policy.network import CatanPolicy
from catan_rl.policy.obs_schema import N_OPP_KINDS, N_OPP_POLICY_SLOTS, OPP_KIND_LEAGUE


def test_snapshot_type_maps_to_league_kind() -> None:
    assert _kind_from_opp_type("snapshot") == OPP_KIND_LEAGUE
    assert 0 <= OPP_KIND_LEAGUE < N_OPP_KINDS  # in range -> no resize


def test_reset_options_feed_real_opp_id_into_obs() -> None:
    env = CatanEnv(opponent_type="heuristic")
    env.reset(seed=0, options={"opponent_kind": OPP_KIND_LEAGUE, "opponent_policy_id": 5})
    obs = env._get_obs()
    assert int(obs["opponent_kind"]) == OPP_KIND_LEAGUE
    assert int(obs["opponent_policy_id"]) == 5


def test_real_league_opp_id_runs_through_policy_without_resize() -> None:
    env = CatanEnv(opponent_type="heuristic")
    env.reset(
        seed=0,
        options={
            "opponent_kind": OPP_KIND_LEAGUE,
            "opponent_policy_id": N_OPP_POLICY_SLOTS - 2,
        },
    )
    obs = env._get_obs()
    masks = env.get_action_masks()

    policy = CatanPolicy()
    policy.set_board_geometry(build_geometry().as_dict_of_tensors())
    obs_t = {k: torch.as_tensor(np.expand_dims(v, 0)) for k, v in obs.items()}
    masks_t = {k: torch.as_tensor(np.expand_dims(v, 0), dtype=torch.bool) for k, v in masks.items()}

    out = policy.sample(obs_t, masks_t)  # must not raise: embedding fits the real id
    assert out["action"].shape == (1, 6)
