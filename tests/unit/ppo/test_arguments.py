"""Tests for the PPO training config (`ppo/arguments.py`).

Pins:
1. Defaults match the audit-derived recipe (n_envs=128, serial,
   torch_compile=False, batch_size=512, etc.). ``vec_env_mode``
   reverted from "subproc" → "serial" on 2026-06-06 because no
   SubprocVecEnv class exists; passing "subproc" emits a
   DeprecationWarning (covered by ``test_subproc_emits_deprecation``).
2. Field validators reject invalid values (negative LR, unknown enums,
   batch_size not dividing rollout buffer).
3. YAML round-trip preserves the config.
4. Env-var overrides layer correctly, including type coercion.
5. Unknown YAML / env keys raise (silent drop is a footgun).
6. The default YAML file matches the dataclass defaults exactly.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from catan_rl.ppo.arguments import (
    CheckpointConfig,
    EvalConfig,
    GAEConfig,
    LossConfig,
    OptimizerConfig,
    PPOConfig,
    RolloutConfig,
    TrainConfig,
    resolve_device,
)

# ---------------------------------------------------------------------------
# Default values pinned (these are the audit conclusions)
# ---------------------------------------------------------------------------


class TestDefaultsMatchAudit:
    def test_rollout_defaults(self) -> None:
        r = RolloutConfig()
        assert r.n_envs == 128, "audit baseline"
        assert r.n_steps == 256
        # 2026-06-06 forensic audit: vec_env_mode default reverted
        # from "subproc" to "serial" because no SubprocVecEnv class
        # exists anywhere in src/catan_rl/. Tracked in
        # docs/plans/rust_engine_actual_state.md.
        assert r.vec_env_mode == "serial"
        assert r.max_turns == 400
        assert r.opponent_type == "heuristic"

    def test_ppo_defaults(self) -> None:
        p = PPOConfig()
        assert p.batch_size == 512
        assert p.n_epochs == 4
        assert p.clip_range == 0.2
        assert p.clip_range_vf == 0.2
        assert p.target_kl == 0.02
        assert p.kl_approx == "k3"
        assert p.advantage_norm == "rollout"
        assert p.use_value_clipping is True

    def test_gae_defaults(self) -> None:
        g = GAEConfig()
        assert g.gamma == 0.995
        assert g.gae_lambda == 0.95

    def test_loss_defaults(self) -> None:
        loss = LossConfig()
        assert loss.value_coef == 0.5
        assert loss.entropy_coef_start == 0.04
        assert loss.entropy_coef_end == 0.005
        assert loss.entropy_anneal_start_update == 50
        assert loss.entropy_anneal_end_update == 200
        assert loss.belief_coef == 0.05
        assert loss.opp_action_coef == 0.03

    def test_optimizer_defaults(self) -> None:
        o = OptimizerConfig()
        assert o.lr_start == 3.0e-4
        assert o.lr_end == 1.0e-5
        assert o.weight_decay == 1.0e-4
        assert o.betas == (0.9, 0.999)
        assert o.eps == 1.0e-5
        assert o.grad_clip_max_norm == 1.0

    def test_top_level_defaults(self) -> None:
        cfg = TrainConfig.default()
        # 50,003,968 = 1526 rollouts at 128 envs * 256 steps; nearest
        # multiple at or above 50M.
        assert cfg.total_steps == 50_003_968
        assert cfg.seed == 42
        assert cfg.device == "auto"
        assert cfg.torch_compile is False, (
            "torch_compile must default False — audit found MPS Inductor harmful"
        )


# ---------------------------------------------------------------------------
# Validators
# ---------------------------------------------------------------------------


class TestValidators:
    def test_negative_lr_rejected(self) -> None:
        with pytest.raises(ValueError, match="lr_start"):
            OptimizerConfig(lr_start=-1.0)

    def test_zero_lr_rejected(self) -> None:
        with pytest.raises(ValueError, match="lr_start"):
            OptimizerConfig(lr_start=0.0)

    def test_gamma_above_one_rejected(self) -> None:
        with pytest.raises(ValueError, match="gamma"):
            GAEConfig(gamma=1.1)

    def test_gamma_zero_rejected(self) -> None:
        with pytest.raises(ValueError, match="gamma"):
            GAEConfig(gamma=0.0)

    def test_gae_lambda_negative_rejected(self) -> None:
        with pytest.raises(ValueError, match="gae_lambda"):
            GAEConfig(gae_lambda=-0.1)

    def test_unknown_vec_env_mode_rejected(self) -> None:
        with pytest.raises(ValueError, match="vec_env_mode"):
            RolloutConfig(vec_env_mode="ray")  # type: ignore[arg-type]

    def test_subproc_emits_deprecation_warning(self) -> None:
        # 2026-06-06 forensic audit: ``vec_env_mode='subproc'`` is
        # config-recognized but has no implementation — no
        # SubprocVecEnv class exists in src/catan_rl/. The default
        # was reverted to 'serial' and 'subproc' now logs a
        # DeprecationWarning so existing YAML configs that pinned
        # 'subproc' surface the lie loudly. See
        # docs/plans/rust_engine_actual_state.md.
        with pytest.warns(DeprecationWarning, match="SubprocVecEnv"):
            RolloutConfig(vec_env_mode="subproc")

    def test_subproc_warning_points_at_actual_state_doc(self) -> None:
        # The warning must point readers at the source-of-truth doc
        # so they can find the remediation plan. Future refactors
        # that drop the doc reference should fail this assertion.
        with pytest.warns(DeprecationWarning, match="rust_engine_actual_state"):
            RolloutConfig(vec_env_mode="subproc")

    def test_unknown_kl_approx_rejected(self) -> None:
        with pytest.raises(ValueError, match="kl_approx"):
            PPOConfig(kl_approx="k4")  # type: ignore[arg-type]

    def test_unknown_advantage_norm_rejected(self) -> None:
        with pytest.raises(ValueError, match="advantage_norm"):
            PPOConfig(advantage_norm="minibatch")  # type: ignore[arg-type]

    def test_unknown_device_rejected(self) -> None:
        with pytest.raises(ValueError, match="device"):
            TrainConfig(device="tpu")  # type: ignore[arg-type]

    def test_batch_size_must_divide_rollout(self) -> None:
        with pytest.raises(ValueError, match=r"batch_size.*divide"):
            TrainConfig(
                rollout=RolloutConfig(n_envs=128, n_steps=256),
                ppo=PPOConfig(batch_size=777),
            )

    def test_entropy_anneal_end_before_start_rejected(self) -> None:
        with pytest.raises(ValueError, match="entropy_anneal_end_update"):
            LossConfig(entropy_anneal_start_update=100, entropy_anneal_end_update=50)

    def test_betas_out_of_range_rejected(self) -> None:
        with pytest.raises(ValueError, match="betas"):
            OptimizerConfig(betas=(0.9, 1.0))

    def test_unknown_opponent_type_rejected(self) -> None:
        with pytest.raises(ValueError, match="opponent_type"):
            RolloutConfig(opponent_type="dqn")  # type: ignore[arg-type]

    def test_lr_anneal_total_updates_negative_rejected(self) -> None:
        # Reviewer fix #1: previously had no validator, so a typo'd
        # CATAN_PPO__OPTIMIZER__LR_ANNEAL_TOTAL_UPDATES=-1 would silently
        # drive the LR schedule into NaN.
        with pytest.raises(ValueError, match="lr_anneal_total_updates"):
            OptimizerConfig(lr_anneal_total_updates=-1)

    def test_clip_range_vf_zero_with_value_clipping_rejected(self) -> None:
        # Reviewer fix #2: clip_range_vf=0 with use_value_clipping=True
        # freezes the value head on the clipped branch (constant w.r.t.
        # parameters). Reject the combination.
        with pytest.raises(ValueError, match="clip_range_vf=0"):
            PPOConfig(clip_range_vf=0.0, use_value_clipping=True)

    def test_clip_range_vf_zero_with_value_clipping_off_ok(self) -> None:
        # When use_value_clipping=False, clip_range_vf is irrelevant; allow 0.
        cfg = PPOConfig(clip_range_vf=0.0, use_value_clipping=False)
        assert cfg.clip_range_vf == 0.0

    def test_total_steps_must_be_multiple_of_rollout(self) -> None:
        # Off-by-one rollouts on cloud-rental hardware = real $$ for no benefit.
        with pytest.raises(ValueError, match=r"total_steps.*multiple"):
            TrainConfig(total_steps=50_000_000)  # not a multiple of 128*256

    def test_total_steps_validator_message_includes_nearest_valid(self) -> None:
        with pytest.raises(ValueError) as exc:
            TrainConfig(total_steps=50_000_000)
        msg = str(exc.value)
        assert "49971200" in msg
        assert "50003968" in msg

    # ----- bool-not-int foot-gun (reviewer HIGH) ------------------------------

    def test_bool_rejected_as_int_field(self) -> None:
        # CATAN_PPO__ROLLOUT__N_ENVS=true would silently become 1 without
        # this guard.
        with pytest.raises(ValueError, match="must be numeric"):
            RolloutConfig(n_envs=True)  # type: ignore[arg-type]

    def test_bool_rejected_as_float_field(self) -> None:
        with pytest.raises(ValueError, match="must be numeric"):
            GAEConfig(gamma=True)  # type: ignore[arg-type]

    def test_string_in_numeric_field_clear_error(self) -> None:
        # YAML can leak a string into a numeric field if a user types a
        # non-number. The error message should mention "numeric", not a
        # cryptic TypeError from a downstream comparison.
        with pytest.raises(ValueError, match="must be numeric"):
            GAEConfig(gamma="not_a_number")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# YAML round-trip
# ---------------------------------------------------------------------------


class TestYAMLRoundTrip:
    def test_default_round_trip_via_dict(self) -> None:
        cfg = TrainConfig.default()
        d = cfg.to_dict()
        assert isinstance(d, dict)
        # Round-trip
        cfg2 = TrainConfig._from_dict(d)
        assert cfg2 == cfg

    def test_default_round_trip_via_yaml(self, tmp_path: Path) -> None:
        cfg = TrainConfig.default()
        path = tmp_path / "ppo.yaml"
        cfg.to_yaml(path)
        cfg2 = TrainConfig.from_yaml(path)
        assert cfg2 == cfg

    def test_partial_yaml_falls_back_to_defaults(self, tmp_path: Path) -> None:
        path = tmp_path / "partial.yaml"
        path.write_text(yaml.safe_dump({"rollout": {"n_envs": 64}}))
        cfg = TrainConfig.from_yaml(path)
        assert cfg.rollout.n_envs == 64
        # Everything else stays at defaults
        assert cfg.rollout.n_steps == 256
        assert cfg.ppo.batch_size == 512
        assert cfg.gae.gamma == 0.995

    def test_unknown_yaml_top_level_field_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.yaml"
        path.write_text(yaml.safe_dump({"made_up_thing": 7}))
        with pytest.raises(KeyError, match="made_up_thing"):
            TrainConfig.from_yaml(path)

    def test_unknown_yaml_section_field_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.yaml"
        path.write_text(yaml.safe_dump({"ppo": {"made_up": 1}}))
        with pytest.raises(KeyError, match="made_up"):
            TrainConfig.from_yaml(path)

    def test_default_yaml_file_matches_dataclass_defaults(self) -> None:
        """The shipped configs/ppo_default.yaml should exactly equal
        TrainConfig.default()'s to_dict output."""
        repo_root = Path(__file__).resolve().parents[3]
        yaml_path = repo_root / "configs" / "ppo_default.yaml"
        assert yaml_path.exists(), f"configs/ppo_default.yaml not found at {yaml_path}"
        cfg = TrainConfig.from_yaml(yaml_path)
        assert cfg == TrainConfig.default()


# ---------------------------------------------------------------------------
# Env-var overrides
# ---------------------------------------------------------------------------


class TestEnvVarOverrides:
    def test_no_env_returns_base(self) -> None:
        base = TrainConfig.default()
        assert TrainConfig.from_env(base=base, env={}) == base

    def test_section_override(self) -> None:
        cfg = TrainConfig.from_env(
            base=TrainConfig.default(),
            env={"CATAN_PPO__ROLLOUT__N_ENVS": "256"},
        )
        assert cfg.rollout.n_envs == 256
        # Other rollout fields unchanged. (Default reverted from
        # "subproc" → "serial" on 2026-06-06; see
        # docs/plans/rust_engine_actual_state.md.)
        assert cfg.rollout.vec_env_mode == "serial"

    def test_root_override(self) -> None:
        cfg = TrainConfig.from_env(
            base=TrainConfig.default(),
            env={"CATAN_PPO__ROOT__SEED": "999"},
        )
        assert cfg.seed == 999

    def test_yaml_scalar_parsing_for_bool(self) -> None:
        cfg = TrainConfig.from_env(
            base=TrainConfig.default(),
            env={"CATAN_PPO__ROOT__TORCH_COMPILE": "true"},
        )
        assert cfg.torch_compile is True

    def test_yaml_scalar_parsing_for_float(self) -> None:
        cfg = TrainConfig.from_env(
            base=TrainConfig.default(),
            env={"CATAN_PPO__OPTIMIZER__LR_START": "1.0e-3"},
        )
        assert cfg.optimizer.lr_start == 1.0e-3

    def test_unknown_section_raises(self) -> None:
        with pytest.raises(KeyError, match="unknown config section"):
            TrainConfig.from_env(env={"CATAN_PPO__NONEXISTENT__X": "1"})

    def test_unknown_field_in_section_raises(self) -> None:
        with pytest.raises(KeyError, match="unknown field"):
            TrainConfig.from_env(env={"CATAN_PPO__ROLLOUT__NOT_REAL": "1"})

    def test_unknown_root_field_raises(self) -> None:
        with pytest.raises(KeyError, match="unknown ROOT field"):
            TrainConfig.from_env(env={"CATAN_PPO__ROOT__NOT_REAL": "1"})

    def test_missing_field_part_raises(self) -> None:
        with pytest.raises(KeyError, match="missing field part"):
            TrainConfig.from_env(env={"CATAN_PPO__ONLYONEPART": "1"})

    def test_non_catan_env_vars_ignored(self) -> None:
        base = TrainConfig.default()
        result = TrainConfig.from_env(base=base, env={"PATH": "/usr/bin", "HOME": "/home/x"})
        assert result == base


# ---------------------------------------------------------------------------
# Chained loader (defaults -> YAML -> env)
# ---------------------------------------------------------------------------


class TestLoadHelper:
    def test_load_with_no_args_returns_defaults(self) -> None:
        assert TrainConfig.load(env={}) == TrainConfig.default()

    def test_load_yaml_then_env(self, tmp_path: Path) -> None:
        # YAML sets rollout.n_envs=64; env overrides to 32. Env wins.
        path = tmp_path / "ppo.yaml"
        path.write_text(yaml.safe_dump({"rollout": {"n_envs": 64}}))
        cfg = TrainConfig.load(yaml_path=path, env={"CATAN_PPO__ROLLOUT__N_ENVS": "32"})
        assert cfg.rollout.n_envs == 32

    def test_load_yaml_no_env_uses_yaml(self, tmp_path: Path) -> None:
        path = tmp_path / "ppo.yaml"
        path.write_text(yaml.safe_dump({"rollout": {"n_envs": 64}}))
        cfg = TrainConfig.load(yaml_path=path, env={})
        assert cfg.rollout.n_envs == 64

    def test_load_only_env(self) -> None:
        cfg = TrainConfig.load(env={"CATAN_PPO__ROOT__SEED": "777"})
        assert cfg.seed == 777
        # Other fields keep their dataclass defaults
        assert cfg.rollout.n_envs == 128


# ---------------------------------------------------------------------------
# Device resolution
# ---------------------------------------------------------------------------


class TestResolveDevice:
    def test_auto_returns_one_of_known(self) -> None:
        d = resolve_device("auto")
        assert d in ("cuda", "mps", "cpu")

    def test_cpu_always_succeeds(self) -> None:
        assert resolve_device("cpu") == "cpu"

    def test_unknown_spec_raises(self) -> None:
        with pytest.raises(ValueError, match="unknown device"):
            resolve_device("tpu")

    def test_cuda_when_unavailable_raises(self) -> None:
        import torch

        if torch.cuda.is_available():
            pytest.skip("CUDA is available on this machine")
        with pytest.raises(RuntimeError, match="CUDA is unavailable"):
            resolve_device("cuda")


# ---------------------------------------------------------------------------
# Reproducibility / immutability
# ---------------------------------------------------------------------------


class TestImmutability:
    def test_train_config_is_frozen(self) -> None:
        cfg = TrainConfig.default()
        with pytest.raises(Exception, match=r"FrozenInstance|cannot assign"):  # FrozenInstanceError
            cfg.seed = 99  # type: ignore[misc]

    def test_section_configs_are_frozen(self) -> None:
        cfg = TrainConfig.default()
        with pytest.raises(Exception, match=r"FrozenInstance|cannot assign"):
            cfg.rollout.n_envs = 99  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Smoke: CheckpointConfig + EvalConfig defaults
# ---------------------------------------------------------------------------


class TestCheckpointEvalDefaults:
    def test_checkpoint_defaults(self) -> None:
        c = CheckpointConfig()
        assert c.save_every_updates == 50
        assert c.keep_last_n == 5
        assert c.save_optimizer_state is True

    def test_eval_defaults(self) -> None:
        e = EvalConfig()
        assert e.eval_every_updates == 20
        assert e.eval_games == 200
        assert e.eval_seeds == (0, 1, 2, 3, 4)
        assert e.eval_opponents == ("random", "heuristic")
