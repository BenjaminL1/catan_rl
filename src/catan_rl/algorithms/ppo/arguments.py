"""
Hyperparameter configuration for PPO training.

Charlesworth-style: single self-play phase from start (no curriculum).
Adapted for M1 Pro (8-core CPU): CPU-only, in-process multi-env,
entropy annealing, linear LR decay, league of past policies.

Two ways to obtain a config:

1. ``get_config()`` — returns the legacy hard-coded dict (default).
2. ``load_config_from_yaml(path)`` — loads a phase-specific YAML config from
   ``configs/`` and returns the same flat dict shape. The YAML may declare
   ``_base: <path>`` to inherit from another config (deep merge).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

# Single phase: Charlesworth-style self-play (League = in-memory pool of past policies)
TRAIN_CONFIG: dict[str, Any] = {
    "learning_rate": 1e-4,  # reset from decayed floor; will decay again to lr_final
    "lr_final": 1e-5,
    "use_linear_lr_decay": True,
    "entropy_coef": 0.01,
    "entropy_coef_start": 0.04,
    "entropy_coef_final": 0.005,
    "entropy_coef_anneal_start": 500,  # ~8.2M steps; keep exploration until league matures
    "entropy_coef_anneal_end": 3000,  # ~49M steps
    "clip_range": 0.2,
    "target_kl": 0.025,  # was 0.015; raised to allow ~6-8 epochs instead of 3
    "n_steps": 4096,  # longer rollout for long-horizon credit assignment
    "n_envs": 8,
    "batch_size": 512,
    "n_epochs": 6,  # reduced from 15; PPO update was 67% of wall time
    "gamma": 0.995,
    "gae_lambda": 0.95,
    "max_grad_norm": 0.5,
    "value_coef": 0.5,
    "weight_decay": 1e-4,
    "recompute_returns": True,
    "normalize_values": True,
    "total_timesteps": 200_000_000,  # extended from 100M; LR decays from 1e-4 over remaining steps
    "checkpoint_freq": 500_000,
    "eval_games": 40,
    "opponent_type": "random",  # start vs random; auto-upgrades to heuristic
    "eval_upgrade_threshold": 0.95,  # switch eval opponent to heuristic at this WR
    "max_turns": 500,
    "win_rate_target": None,  # no early stopping (Charlesworth style)
    # League (Charlesworth-style): add policy every N updates
    "league_maxlen": 100,
    "add_policy_every": 4,
    "league_random_weight": 0.05,  # 5% chaos injection; avoids early self-play echo chamber
    "heuristic_opponent_weight": 0.25,  # 25% heuristic as stable anchor until WR >90%
    # Entropy floor: if policy entropy drops below this, coef is temporarily raised
    # to prevent premature convergence to a deterministic policy.
    "entropy_floor": 0.003,
    "entropy_floor_coef": 0.01,
    # Stagnation detection: warn if win rate doesn't improve over this many evals.
    "stagnation_window": 20,  # was 10; longer window avoids false positives at high WR
    "stagnation_threshold": 0.01,  # was 0.03; smaller delta acceptable at 95%+ win rate
    # torch.compile: disabled — reduce-overhead mode uses CUDA graphs (GPU-only);
    # on CPU it adds per-call overhead that outweighs any kernel fusion gains.
    "torch_compile": False,
    "eval_freq": 100_000,  # evaluate every 100k steps
    # checkpoint_freq controls save interval (500k); eval_freq controls eval interval (100k)
    # ── Phase 0: eval harness + per-head entropy diagnostics ─────────────
    "eval_harness_seeds": list(range(0, 200)),
    "eval_harness_swap_first_player": True,
    "frozen_champion_path": "checkpoints/train/checkpoint_07390040.pt",
    "entropy_collapse_threshold": 0.0005,
    "entropy_collapse_consecutive_updates": 3,
}

# Shared across all phases — defines the neural network shape.
# Keys must match build_agent_model.DEFAULT_MODEL_CONFIG.
MODEL_CONFIG: dict[str, Any] = {
    "obs_output_dim": 512,
    "tile_in_dim": 79,
    "tile_model_dim": 128,
    "curr_player_main_in_dim": 166,
    "other_player_main_in_dim": 173,
    "dev_card_embed_dim": 64,
    "dev_card_model_dim": 64,
    "tile_model_num_heads": 4,
    "proj_dev_card_dim": 25,
    "dev_card_model_num_heads": 4,
    "tile_encoder_num_layers": 2,
    "proj_tile_dim": 25,
    "action_head_hidden_dim": 128,
    "value_hidden_dims": (256, 128),
    "dropout": 0.0,
    # Phase 1.3 obs-encoding flag. True keeps the legacy 166/173 thermometer
    # schema (matches checkpoint_07390040.pt). False switches env + model to
    # the compact 53/60 schema. ``_maybe_apply_compact_obs_dims`` (in
    # ``resolve_config``) auto-aligns ``curr_player_main_in_dim`` /
    # ``other_player_main_in_dim`` when the user only flips this flag.
    "use_thermometer_encoding": True,
    # Phase 1.5 dihedral data augmentation probability per minibatch.
    "symmetry_aug_prob": 0.0,
    # ── Phase 2 architecture flags (all default to legacy/off) ───────────
    # 2.1 Axial positional embedding for tiles.
    "use_axial_pos_emb": False,
    "axial_pos_dim": 24,
    # 2.2 Transformer recipe overrides (None = inherit ``dropout``).
    "transformer_dropout": None,
    "transformer_activation": "relu",
    # 2.4 AdaLN-conditioned action heads (FiLM modulation per head).
    "action_head_film": False,
    # 2.5 Value tower mode: 'shared' (legacy) | 'decoupled' (separate encoder).
    "value_head_mode": "shared",
}


def get_config(phase: str = "train") -> dict[str, Any]:
    """Merge training config with model architecture config.

    Single phase (Charlesworth-style): no curriculum, self-play from start.

    Args:
        phase: Ignored for compatibility; always returns TRAIN_CONFIG.

    Returns:
        Single dict containing everything needed to create CatanPPO + CatanPolicy.
    """
    config: dict[str, Any] = {}
    config.update(MODEL_CONFIG)
    config.update(TRAIN_CONFIG)
    return config


# ─────────────────────────────────────────────────────────────────────────────
# YAML loader
# ─────────────────────────────────────────────────────────────────────────────

# YAML schema → flat-dict key mapping. The flat dict is what CatanPPO consumes.
_YAML_SECTIONS = (
    "model",
    "optim",
    "ppo",
    "entropy",
    "league",
    "eval",
    "stagnation",
    "schedule",
    "misc",
)


def load_config_from_yaml(path: str | Path) -> dict[str, Any]:
    """Load a phase config from a YAML file under ``configs/``.

    Supports ``_base: <relative_path>`` for inheritance: the base config is
    loaded first and then the current file's keys deep-merge over it.

    The YAML is expected to be section-keyed (model / optim / ppo / entropy /
    league / eval / stagnation / schedule / misc). The loader flattens those
    sections into a single dict matching ``get_config()``'s output shape.

    Args:
        path: Path to a YAML file (absolute or relative to CWD).

    Returns:
        Flat config dict ready to pass to ``CatanPPO(config)``.
    """
    try:
        import yaml
    except ImportError as e:  # pragma: no cover - hard dep but guard for clarity
        raise RuntimeError("load_config_from_yaml requires PyYAML; pip install pyyaml") from e

    p = Path(path).resolve()
    raw = _load_yaml_with_inheritance(p, yaml)
    return _flatten_yaml(raw)


def _load_yaml_with_inheritance(path: Path, yaml_module: Any) -> dict[str, Any]:
    with path.open("r") as f:
        data: dict[str, Any] = yaml_module.safe_load(f) or {}
    base = data.pop("_base", None)
    if base is None:
        return data
    base_path = (path.parent / base).resolve()
    base_data = _load_yaml_with_inheritance(base_path, yaml_module)
    return _deep_merge(base_data, data)


def _deep_merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``over`` onto ``base`` (over wins on conflict)."""
    out = dict(base)
    for k, v in over.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _flatten_yaml(raw: dict[str, Any]) -> dict[str, Any]:
    """Flatten a section-keyed YAML config into the flat dict CatanPPO expects."""
    flat: dict[str, Any] = {}
    for section in _YAML_SECTIONS:
        if section in raw and isinstance(raw[section], dict):
            flat.update(raw[section])
    # `value_hidden_dims` round-trips as a list in YAML; convert back to tuple.
    if "value_hidden_dims" in flat and isinstance(flat["value_hidden_dims"], list):
        flat["value_hidden_dims"] = tuple(flat["value_hidden_dims"])
    # Carry through any phase-specific extra keys (e.g. `phase0:`).
    for key, val in raw.items():
        if key in _YAML_SECTIONS:
            continue
        if isinstance(val, dict):
            flat.update(val)
        else:
            flat[key] = val
    return flat


def _maybe_apply_compact_obs_dims(config: dict[str, Any]) -> dict[str, Any]:
    """Phase 1.3: when ``use_thermometer_encoding=False`` and the user hasn't
    overridden the player-feature dims, auto-set them to the compact values.

    Without this helper, a YAML override that sets ``use_thermometer_encoding:
    false`` but inherits the legacy 166/173 dims from ``_base.yaml`` would
    silently produce a dim mismatch between env and model.
    """
    if config.get("use_thermometer_encoding", True):
        return config
    # Compact mode requested. Override only if the user kept the legacy defaults.
    if config.get("curr_player_main_in_dim") == 166:
        config["curr_player_main_in_dim"] = 54
    if config.get("other_player_main_in_dim") == 173:
        config["other_player_main_in_dim"] = 61
    return config


def resolve_config(yaml_path: str | Path | None = None) -> dict[str, Any]:
    """Return the legacy default config, or load from YAML if ``yaml_path`` given.

    Applies ``_maybe_apply_compact_obs_dims`` so opting in to the Phase 1.3
    compact encoding via ``use_thermometer_encoding: false`` automatically
    selects the matching 53/60 input dims unless the user explicitly set
    them otherwise.
    """
    if yaml_path is None:
        return _maybe_apply_compact_obs_dims(get_config())
    return _maybe_apply_compact_obs_dims(load_config_from_yaml(yaml_path))
