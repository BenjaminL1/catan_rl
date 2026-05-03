"""Tests for Phase 1.3 obs schema switch (drop bucket8 thermometers)."""

from __future__ import annotations

from catan_rl.env.catan_env import CatanEnv, bucket8, compact_scalar

# ── compact_scalar primitive ────────────────────────────────────────────────


def test_compact_scalar_normalizes_to_unit_interval() -> None:
    """``compact_scalar(v, max_v)`` returns ``[v / max_v]`` clipped to [0, 1]."""
    assert compact_scalar(0, 5) == [0.0]
    assert compact_scalar(5, 5) == [1.0]
    assert compact_scalar(2, 8) == [0.25]
    # Out-of-range values clip to [0, 1]
    assert compact_scalar(-1, 5) == [0.0]
    assert compact_scalar(10, 5) == [1.0]


def test_compact_scalar_dim_is_one() -> None:
    """Whatever the value, the result is a single scalar."""
    assert len(compact_scalar(3, 8)) == 1
    assert len(compact_scalar(0, 0)) == 1  # zero max_v is handled


# ── bucket8 still works (legacy back-compat) ───────────────────────────────


def test_bucket8_dim_is_eight() -> None:
    """The legacy 8-threshold thermometer encoding is unchanged."""
    arr = bucket8(3, 8)
    assert arr.shape == (8,)


# ── Env obs shape switch ────────────────────────────────────────────────────


def test_legacy_env_produces_166_dim() -> None:
    """``use_thermometer_encoding=True`` keeps the 166/173 schema (back-compat)."""
    env = CatanEnv(opponent_type="random", max_turns=20, use_thermometer_encoding=True)
    obs, _ = env.reset(seed=0)
    assert obs["current_player_main"].shape == (166,)
    assert obs["next_player_main"].shape == (173,)


def test_compact_env_produces_54_dim() -> None:
    """``use_thermometer_encoding=False`` drops to the compact 54/61 schema."""
    env = CatanEnv(opponent_type="random", max_turns=20, use_thermometer_encoding=False)
    obs, _ = env.reset(seed=0)
    assert obs["current_player_main"].shape == (54,)
    assert obs["next_player_main"].shape == (61,)


def test_env_observation_space_matches_actual_obs() -> None:
    """``env.observation_space`` advertises the same dims the env emits."""
    for thermometer in (True, False):
        env = CatanEnv(opponent_type="random", max_turns=20, use_thermometer_encoding=thermometer)
        obs, _ = env.reset(seed=0)
        space = env.observation_space.spaces
        assert obs["current_player_main"].shape == space["current_player_main"].shape
        assert obs["next_player_main"].shape == space["next_player_main"].shape


def test_compact_obs_in_unit_interval() -> None:
    """All compact-mode features are bounded in [0, 1] (or one-hot / boolean)."""
    env = CatanEnv(opponent_type="random", max_turns=20, use_thermometer_encoding=False)
    obs, _ = env.reset(seed=0)
    cur = obs["current_player_main"]
    # Some flags can equal 1.0 exactly; this tests the bound, not strict <.
    assert (cur >= 0.0).all()
    assert (cur <= 1.0).all(), f"out-of-range: {cur[cur > 1.0]}"


# ── arguments.py auto-resolves dims when only the flag is flipped ──────────


def test_resolve_config_auto_aligns_compact_dims() -> None:
    """``_maybe_apply_compact_obs_dims`` aligns curr/next dims with the flag."""
    from catan_rl.algorithms.ppo.arguments import _maybe_apply_compact_obs_dims, get_config

    cfg = get_config()
    cfg["use_thermometer_encoding"] = False
    # User did not override the dims — they were the legacy defaults.
    cfg = _maybe_apply_compact_obs_dims(cfg)
    assert cfg["curr_player_main_in_dim"] == 54
    assert cfg["other_player_main_in_dim"] == 61


def test_resolve_config_respects_explicit_dim_override() -> None:
    """If the user explicitly sets dims, the auto-aligner does NOT touch them."""
    from catan_rl.algorithms.ppo.arguments import _maybe_apply_compact_obs_dims, get_config

    cfg = get_config()
    cfg["use_thermometer_encoding"] = False
    cfg["curr_player_main_in_dim"] = 99  # custom override
    cfg["other_player_main_in_dim"] = 99
    cfg = _maybe_apply_compact_obs_dims(cfg)
    assert cfg["curr_player_main_in_dim"] == 99
    assert cfg["other_player_main_in_dim"] == 99


def test_phase1_full_yaml_resolves_to_compact_schema() -> None:
    """The shipped phase1_full.yaml resolves to a self-consistent compact config."""
    from catan_rl.algorithms.ppo.arguments import resolve_config

    cfg = resolve_config("configs/phase1_full.yaml")
    assert cfg["use_thermometer_encoding"] is False
    assert cfg["curr_player_main_in_dim"] == 54
    assert cfg["other_player_main_in_dim"] == 61
    assert cfg["use_value_clipping"] is True
    assert cfg["use_devcard_mha"] is False
    assert cfg["symmetry_aug_prob"] == 0.5
    assert cfg["advantage_norm"] == "rollout"
