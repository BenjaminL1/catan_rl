"""PPO training configuration — typed dataclasses + YAML / env-var loader.

Single source of truth for every hyperparameter that the PPO trainer,
rollout buffer, vec env, and eval harness consume. Defaults below come
from the v2 hardware audit measured on this codebase on an M1 Pro
(documented in commit `0baef...`'s notes + Phase 0 of the build-out):

* ``n_envs=128`` — pushes rollout batch above the MPS crossover (~256)
  on the SGD side and amortises Python orchestration overhead.
* ``vec_env_mode="serial"`` — all envs in-process. ``"subproc"`` is a
  config-recognized literal **with no implementation**: no
  ``SubprocVecEnv`` class exists anywhere in ``src/catan_rl/``. The
  previous default ``"subproc"`` was a documentation lie expressed as
  a config value (2026-06-06 forensic audit, see
  ``docs/plans/rust_engine_actual_state.md``). Passing ``"subproc"``
  explicitly logs a ``DeprecationWarning``; behaviour is identical to
  ``"serial"``. The Rust migration's Phase 6 will either implement
  ``"subproc"`` properly or remove the literal from the union.
* ``torch_compile=False`` — explicitly off. The MPS / Inductor backend
  on macOS silently recompiles every input shape and adds overhead
  without producing a CUDA-graph speedup.
* ``batch_size=512`` — MPS SGD wins 2.3x over CPU at this batch.
* ``device="auto"`` — resolves to CUDA → MPS → CPU at runtime via
  :func:`resolve_device`. A single device drives both rollout and SGD
  for simplicity; the audit confirmed that the dual-device hybrid
  (CPU rollout + MPS SGD) only saves ~5% wall-time for this 1.4M-param
  policy and is not worth the cross-device tensor copies.

Hyperparameters that are NOT in this module:
* Policy architecture flags live in :mod:`catan_rl.policy.network`
  (use_axial_pos_emb, use_graph_encoder, use_belief_head, etc.).
* BC training config lives in :mod:`catan_rl.bc` and `configs/bc.yaml`.
* Eval / league weights move to :mod:`catan_rl.selfplay` and
  :mod:`catan_rl.eval` in later phases.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Final, Literal

import yaml

# ---------------------------------------------------------------------------
# Enum-like literal sets used as the validators below.
# ---------------------------------------------------------------------------

_VEC_ENV_MODES: Final[tuple[str, ...]] = ("serial", "subproc")
_KL_APPROXIMATIONS: Final[tuple[str, ...]] = ("k1", "k2", "k3")
_ADVANTAGE_NORM_MODES: Final[tuple[str, ...]] = ("rollout", "batch", "none")
_DEVICE_VALUES: Final[tuple[str, ...]] = ("auto", "cpu", "mps", "cuda")
_OPPONENT_TYPES: Final[tuple[str, ...]] = (
    "random",
    "heuristic",
    "policy",
    "self",
    "league",
)


# ---------------------------------------------------------------------------
# Per-section configs.
#
# Each section is a frozen dataclass with field validators. Values mutate
# only via :func:`replace`-style construction, never in place — keeps
# checkpoint reproducibility tractable.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RolloutConfig:
    """How rollouts are collected from the env."""

    n_envs: int = 128
    """Number of parallel envs in the vec env."""

    n_steps: int = 256
    """Steps per env per rollout. Total transitions per rollout =
    ``n_envs * n_steps`` (default: 32,768)."""

    vec_env_mode: Literal["serial", "subproc"] = "serial"
    """``serial`` runs all envs in the main process. ``subproc`` is
    config-recognized but has no implementation — passing it logs a
    ``DeprecationWarning`` and falls through to ``"serial"``. The
    forensic audit on 2026-06-06 confirmed no ``SubprocVecEnv`` class
    exists anywhere in ``src/catan_rl/``; the previous default
    ``"subproc"`` was a documentation lie. The Rust migration's Phase 6
    will either implement the literal properly or remove it from the
    union — until then, only ``"serial"`` is meaningful."""

    max_turns: int = 400
    """Per-game truncation cap (Catan games normally end well before this;
    the cap exists to bound the buffer for runaway loops)."""

    opponent_type: Literal["random", "heuristic", "policy", "self", "league"] = "heuristic"
    """Default opponent at rollout time. ``league`` requires the league
    selfplay phase to be wired in."""

    def __post_init__(self) -> None:
        _check_positive("n_envs", self.n_envs)
        _check_positive("n_steps", self.n_steps)
        _check_positive("max_turns", self.max_turns)
        _check_in("vec_env_mode", self.vec_env_mode, _VEC_ENV_MODES)
        _check_in("opponent_type", self.opponent_type, _OPPONENT_TYPES)
        if self.vec_env_mode == "subproc":
            import warnings

            warnings.warn(
                "RolloutConfig(vec_env_mode='subproc') is config-recognized "
                "but has no implementation — no SubprocVecEnv class exists "
                "in src/catan_rl/. The Rust migration's Phase 6 will either "
                "implement it or remove the literal. Behaviour is identical "
                "to vec_env_mode='serial'; please update YAML configs to "
                "'serial' to silence this warning. See "
                "docs/plans/rust_engine_actual_state.md.",
                DeprecationWarning,
                stacklevel=2,
            )


@dataclass(frozen=True)
class PPOConfig:
    """Per-update PPO knobs (clip, KL, batches)."""

    n_epochs: int = 4
    """Number of passes over the rollout buffer per PPO update."""

    batch_size: int = 512
    """SGD minibatch size. Must divide ``n_envs * n_steps``."""

    clip_range: float = 0.2
    """Policy ratio clip range (the canonical PPO2 ε)."""

    clip_range_vf: float = 0.2
    """Value-function clip range. The PPO2 formulation is
    ``loss_v = max(MSE_unclipped, MSE_clipped)``; the clip's reference
    is ``old_v``, not the rolling mean."""

    target_kl: float = 0.02
    """If approximate KL exceeds this between two epochs of the same
    update, the remaining epochs are skipped. ``0.0`` disables."""

    kl_approx: Literal["k1", "k2", "k3"] = "k3"
    """Choice of KL estimator. k1 = ``ratio - 1`` (biased, cheap).
    k2 = ``(ratio - 1)^2 / 2`` (biased, low variance). k3 =
    ``(ratio - 1) - log(ratio)`` (Schulman 2020, unbiased, low
    variance — recommended default)."""

    advantage_norm: Literal["rollout", "batch", "none"] = "rollout"
    """When to z-normalise advantages. ``rollout`` standardises across
    the whole buffer once (default); ``batch`` standardises per SGD
    batch; ``none`` skips. Per-rollout typically gives the lowest loss
    variance for actor-critic with shared backbone."""

    use_value_clipping: bool = True
    """Apply the PPO2-style value clip. Disable to fall back to plain
    MSE — useful for ablations."""

    def __post_init__(self) -> None:
        _check_positive("n_epochs", self.n_epochs)
        _check_positive("batch_size", self.batch_size)
        _check_positive("clip_range", self.clip_range)
        _check_non_negative("clip_range_vf", self.clip_range_vf)
        # ``clip_range_vf == 0`` + ``use_value_clipping=True`` makes the
        # PPO2 clipped MSE branch constant in the parameters, silently
        # zeroing value-head gradients for any sample where the clipped
        # branch is selected. Reject the combination.
        if self.use_value_clipping and self.clip_range_vf == 0:
            raise ValueError(
                "clip_range_vf=0 with use_value_clipping=True freezes the "
                "value head on the clipped branch; set clip_range_vf>0 or "
                "disable use_value_clipping"
            )
        _check_non_negative("target_kl", self.target_kl)
        _check_in("kl_approx", self.kl_approx, _KL_APPROXIMATIONS)
        _check_in("advantage_norm", self.advantage_norm, _ADVANTAGE_NORM_MODES)


@dataclass(frozen=True)
class GAEConfig:
    """Discount + GAE λ."""

    gamma: float = 0.995
    """Per-step discount. 0.995 over ~200 steps gives an effective horizon
    of ~200 steps — about one full Catan game, which matches the
    objective (terminal win/loss)."""

    gae_lambda: float = 0.95
    """Bias-variance trade-off for the advantage estimator. 0.95 is
    standard."""

    def __post_init__(self) -> None:
        _check_numeric("gamma", self.gamma)
        _check_numeric("gae_lambda", self.gae_lambda)
        if not 0 < self.gamma <= 1.0:
            raise ValueError(f"gamma must be in (0, 1], got {self.gamma}")
        if not 0 <= self.gae_lambda <= 1.0:
            raise ValueError(f"gae_lambda must be in [0, 1], got {self.gae_lambda}")


@dataclass(frozen=True)
class LossConfig:
    """Per-term loss weights + entropy anneal schedule."""

    value_coef: float = 0.5
    """Coefficient on the value-function loss in the total loss."""

    entropy_coef_start: float = 0.04
    """Entropy bonus coefficient at the start of training."""

    entropy_coef_end: float = 0.005
    """Entropy bonus coefficient after the anneal completes."""

    entropy_anneal_start_update: int = 50
    """Update index at which entropy begins annealing from
    ``entropy_coef_start`` toward ``entropy_coef_end``."""

    entropy_anneal_end_update: int = 200
    """Update index at which entropy reaches ``entropy_coef_end``."""

    belief_coef: float = 0.05
    """Belief-head auxiliary loss weight (CE against engine ground truth
    for the opponent's hidden dev cards)."""

    opp_action_coef: float = 0.03
    """Opponent-action auxiliary loss weight (CE on the next action type
    the opponent took). Only fires when the opponent is a historical
    league policy; filtered out for random/heuristic/self."""

    def __post_init__(self) -> None:
        for name in (
            "value_coef",
            "entropy_coef_start",
            "entropy_coef_end",
            "belief_coef",
            "opp_action_coef",
        ):
            _check_non_negative(name, getattr(self, name))
        if self.entropy_anneal_end_update < self.entropy_anneal_start_update:
            raise ValueError(
                f"entropy_anneal_end_update ({self.entropy_anneal_end_update}) "
                f"must be >= entropy_anneal_start_update ({self.entropy_anneal_start_update})"
            )


@dataclass(frozen=True)
class OptimizerConfig:
    """AdamW + LR schedule (linear decay over total updates)."""

    lr_start: float = 3.0e-4
    """Initial learning rate."""

    lr_end: float = 1.0e-5
    """Final learning rate after linear decay over
    ``lr_anneal_total_updates`` updates."""

    weight_decay: float = 1.0e-4
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1.0e-5
    grad_clip_max_norm: float = 1.0

    lr_anneal_total_updates: int = 0
    """Number of updates over which LR linearly decays from ``lr_start``
    to ``lr_end``. ``0`` means "infer from total_steps and rollout size"
    at trainer construction time."""

    def __post_init__(self) -> None:
        _check_positive("lr_start", self.lr_start)
        _check_non_negative("lr_end", self.lr_end)
        _check_non_negative("weight_decay", self.weight_decay)
        _check_positive("eps", self.eps)
        _check_positive("grad_clip_max_norm", self.grad_clip_max_norm)
        _check_non_negative("lr_anneal_total_updates", self.lr_anneal_total_updates)
        if not (0 <= self.betas[0] < 1 and 0 <= self.betas[1] < 1):
            raise ValueError(f"betas must each be in [0, 1), got {self.betas}")


@dataclass(frozen=True)
class CheckpointConfig:
    """Checkpoint cadence + retention."""

    save_every_updates: int = 50
    """Save a checkpoint every N PPO updates."""

    keep_last_n: int = 5
    """Retain the N most recent checkpoints in the output directory.
    Older ones are deleted on save. ``0`` disables retention pruning."""

    save_optimizer_state: bool = True
    """If True, optimizer + LR-schedule state is saved alongside the
    policy. Required for clean resume; can be False for size-conscious
    deploys."""

    def __post_init__(self) -> None:
        _check_positive("save_every_updates", self.save_every_updates)
        _check_non_negative("keep_last_n", self.keep_last_n)


@dataclass(frozen=True)
class EvalConfig:
    """Eval harness cadence + per-eval game budget."""

    eval_every_updates: int = 20
    """Run the eval harness every N PPO updates."""

    eval_games: int = 200
    """Games per eval matchup. Binomial SE at p=0.5 is ~0.035 at N=200."""

    eval_seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    """Seeds to run; each seed plays both seats (so N games per matchup
    = ``eval_games * len(eval_seeds) * 2``)."""

    eval_opponents: tuple[str, ...] = ("random", "heuristic")
    """Opponent labels evaluated each eval round."""

    def __post_init__(self) -> None:
        _check_positive("eval_every_updates", self.eval_every_updates)
        _check_positive("eval_games", self.eval_games)
        if not self.eval_seeds:
            raise ValueError("eval_seeds must be non-empty")
        if not self.eval_opponents:
            raise ValueError("eval_opponents must be non-empty")


# ---------------------------------------------------------------------------
# Top-level training config.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LeagueConfig:
    """Opponent-mix weights + snapshot retention for self-play.

    See :class:`catan_rl.selfplay.league.League` for semantics.
    """

    random_weight: float = 0.0
    """Weight on the uniformly-random opponent (default 0 — heuristic
    is the v2 baseline opponent)."""

    heuristic_weight: float = 1.0
    """Weight on the heuristic opponent."""

    snapshot_weight: float = 0.0
    """Weight on past-policy snapshots. Setting >0 with an empty
    pool falls back to the remaining kinds. Phase 6 raises
    NotImplementedError if a snapshot is actually sampled — the
    snapshot opponent path lands in Phase 8+."""

    add_snapshot_every_n_updates: int = 4
    """Append a fresh snapshot to the pool every N PPO updates."""

    maxlen: int = 100
    """Maximum snapshot pool size; oldest entries are evicted FIFO."""

    def __post_init__(self) -> None:
        for name in ("random_weight", "heuristic_weight", "snapshot_weight"):
            _check_non_negative(name, getattr(self, name))
        if self.random_weight + self.heuristic_weight + self.snapshot_weight == 0:
            raise ValueError("at least one league weight must be > 0; got all zeros")
        _check_positive("add_snapshot_every_n_updates", self.add_snapshot_every_n_updates)
        _check_positive("maxlen", self.maxlen)


@dataclass(frozen=True)
class TrainConfig:
    """Top-level training configuration. The full set of knobs needed to
    reproduce a training run from scratch.

    Construct via :func:`TrainConfig.default` for the audit-derived M1
    defaults, :func:`TrainConfig.from_yaml` for file-based overrides, or
    :func:`TrainConfig.from_env` to layer environment-variable overrides
    on top.
    """

    rollout: RolloutConfig = field(default_factory=RolloutConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    gae: GAEConfig = field(default_factory=GAEConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    league: LeagueConfig = field(default_factory=LeagueConfig)

    total_steps: int = 50_003_968
    """Total env steps to train for. Must be a multiple of
    ``rollout.n_envs * rollout.n_steps`` so the PPO loop terminates
    exactly at the boundary (no off-by-one rollout overrun on cloud-
    metered hardware). The default is 1526 rollouts at the audit-derived
    rollout size of 128 * 256 = 32,768 transitions, totaling ~50.0M."""

    seed: int = 42
    """RNG seed for env reset + numpy/torch global RNGs."""

    device: Literal["auto", "cpu", "mps", "cuda"] = "auto"
    """Compute device. ``auto`` resolves to CUDA → MPS → CPU at runtime."""

    torch_compile: bool = False
    """Whether to wrap the policy in ``torch.compile``. Default off:
    audit measured that MPS Inductor recompiles per shape and adds
    overhead. Set True only for CUDA + ``mode='reduce-overhead'``."""

    run_name: str = "ppo"
    """Sub-directory under ``output_dir`` for this run's TB + checkpoints."""

    output_dir: str = "runs/train"
    """Root directory for TB logs + checkpoints."""

    def __post_init__(self) -> None:
        _check_positive("total_steps", self.total_steps)
        _check_in("device", self.device, _DEVICE_VALUES)
        # Cross-section coherence: batch_size must divide n_envs * n_steps.
        transitions_per_rollout = self.rollout.n_envs * self.rollout.n_steps
        if transitions_per_rollout % self.ppo.batch_size != 0:
            raise ValueError(
                f"batch_size ({self.ppo.batch_size}) must divide "
                f"n_envs * n_steps ({transitions_per_rollout})"
            )
        # total_steps must be a multiple of the rollout transition count
        # so the trainer's outer loop hits exactly the requested step
        # budget instead of overshooting by up to ``transitions_per_rollout - 1``
        # transitions (real cloud-rental cost).
        if self.total_steps % transitions_per_rollout != 0:
            lower = (self.total_steps // transitions_per_rollout) * transitions_per_rollout
            upper = lower + transitions_per_rollout
            raise ValueError(
                f"total_steps ({self.total_steps}) must be a multiple of "
                f"n_envs * n_steps ({transitions_per_rollout}). "
                f"Nearest valid values: {lower} or {upper}"
            )

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def default(cls) -> TrainConfig:
        """Return the audit-derived M1 default config."""
        return cls()

    @classmethod
    def from_yaml(cls, path: str | Path) -> TrainConfig:
        """Load from a YAML file, falling back to defaults for any
        section not present in the file."""
        raw = yaml.safe_load(Path(path).read_text()) or {}
        if not isinstance(raw, dict):
            raise TypeError(f"{path}: expected a mapping at top level")
        return cls._from_dict(raw)

    @classmethod
    def load(
        cls,
        yaml_path: str | Path | None = None,
        env: dict[str, str] | None = None,
    ) -> TrainConfig:
        """Resolve the full override chain in one call: defaults → YAML → env.

        Operator-friendly entry point for the train script. The order
        is fixed: dataclass defaults at the bottom, YAML file in the
        middle, environment variables on top. Mis-ordered ad-hoc chains
        (which silently re-apply defaults over YAML) are a real foot-gun
        we don't want to repeat at every call site.
        """
        cfg = cls.from_yaml(yaml_path) if yaml_path is not None else cls.default()
        return cls.from_env(base=cfg, env=env)

    @classmethod
    def from_env(
        cls, base: TrainConfig | None = None, env: dict[str, str] | None = None
    ) -> TrainConfig:
        """Layer environment-variable overrides on top of ``base``.

        Env-var convention: ``CATAN_PPO__SECTION__KEY=value`` (double
        underscores as the separator). ``SECTION`` is the lowercase
        sub-config name (``rollout``, ``ppo``, ``gae``, ``loss``,
        ``optimizer``, ``checkpoint``, ``eval``) OR the literal token
        ``ROOT`` for top-level fields. Values are parsed as YAML scalars
        (so ``true``/``false``/numbers/strings all just work).

        Example::

            CATAN_PPO__ROLLOUT__N_ENVS=256
            CATAN_PPO__PPO__BATCH_SIZE=1024
            CATAN_PPO__ROOT__DEVICE=cuda

        Unknown keys raise ``KeyError`` rather than silently dropping —
        a misspelled env var that nukes a hyperparameter override is the
        kind of silent training-corruption bug that wastes a whole run.
        """
        cfg = base if base is not None else cls.default()
        env = env if env is not None else dict(os.environ)
        prefix = "CATAN_PPO__"
        overrides: dict[str, dict[str, Any]] = {}
        for key, value in env.items():
            if not key.startswith(prefix):
                continue
            rest = key[len(prefix) :]
            section, _, field_name = rest.partition("__")
            if not field_name:
                raise KeyError(
                    f"env override {key!r} missing field part "
                    f"(expected CATAN_PPO__SECTION__KEY=value)"
                )
            section = section.lower()
            field_name = field_name.lower()
            parsed = yaml.safe_load(value)
            overrides.setdefault(section, {})[field_name] = parsed
        if not overrides:
            return cfg

        # Merge into a dict, then re-validate.
        d = _dataclass_to_dict(cfg)
        for section, kvs in overrides.items():
            if section == "root":
                for k, v in kvs.items():
                    if k not in d:
                        raise KeyError(f"unknown ROOT field {k!r} in env override")
                    d[k] = v
            else:
                if section not in d or not isinstance(d[section], dict):
                    raise KeyError(f"unknown config section {section!r} in env override")
                for k, v in kvs.items():
                    if k not in d[section]:
                        raise KeyError(f"unknown field {k!r} in section {section!r}")
                    d[section][k] = v
        return cls._from_dict(d)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Return a plain-dict snapshot of the config. Round-trips through
        :func:`_from_dict`."""
        return _dataclass_to_dict(self)

    def to_yaml(self, path: str | Path) -> None:
        """Persist the config to a YAML file."""
        Path(path).write_text(yaml.safe_dump(self.to_dict(), sort_keys=False))

    @classmethod
    def _from_dict(cls, raw: dict[str, Any]) -> TrainConfig:
        """Construct from a dict whose layout matches :func:`to_dict`.

        Unknown keys raise; missing sections fall back to defaults.
        """
        sections: dict[str, type] = {
            "rollout": RolloutConfig,
            "ppo": PPOConfig,
            "gae": GAEConfig,
            "loss": LossConfig,
            "optimizer": OptimizerConfig,
            "checkpoint": CheckpointConfig,
            "eval": EvalConfig,
            "league": LeagueConfig,
        }
        kwargs: dict[str, Any] = {}
        for section_name, section_cls in sections.items():
            section_raw = raw.pop(section_name, None) or {}
            if not isinstance(section_raw, dict):
                raise TypeError(
                    f"section {section_name!r}: expected mapping, got {type(section_raw).__name__}"
                )
            kwargs[section_name] = _construct_dataclass(section_cls, section_raw)
        # Top-level scalar fields.
        scalar_fields = {f.name for f in fields(cls) if f.name not in sections}
        for k in list(raw.keys()):
            if k not in scalar_fields:
                raise KeyError(f"unknown top-level field {k!r}")
            kwargs[k] = raw.pop(k)
        return cls(**kwargs)


# ---------------------------------------------------------------------------
# Device resolution (called at trainer / eval construction time)
# ---------------------------------------------------------------------------


def resolve_device(spec: str) -> str:
    """Resolve a device spec to a concrete backend name at runtime.

    ``"auto"`` picks the best available: CUDA → MPS → CPU. Explicit
    specs are validated against the actually-available backends; if you
    ask for ``"cuda"`` on a machine without CUDA, this raises rather
    than silently falling back.
    """
    import torch  # local import keeps module-level import fast

    if spec == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if spec == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("device='cuda' requested but CUDA is unavailable")
        return "cuda"
    if spec == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("device='mps' requested but MPS is unavailable")
        return "mps"
    if spec == "cpu":
        return "cpu"
    raise ValueError(f"unknown device spec {spec!r}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _check_numeric(name: str, value: Any) -> None:
    """Reject bool / non-numeric values before downstream comparison.

    ``bool`` is a subclass of ``int`` in Python, so ``True > 0`` is
    silently True. Without this guard a malformed env-var override like
    ``CATAN_PPO__ROLLOUT__N_ENVS=true`` would silently set ``n_envs=1``.
    """
    if isinstance(value, bool):
        raise ValueError(f"{name} must be numeric, got bool ({value!r})")
    if not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric, got {type(value).__name__} ({value!r})")


def _check_positive(name: str, value: Any) -> None:
    _check_numeric(name, value)
    if value <= 0:
        raise ValueError(f"{name} must be > 0, got {value}")


def _check_non_negative(name: str, value: Any) -> None:
    _check_numeric(name, value)
    if value < 0:
        raise ValueError(f"{name} must be >= 0, got {value}")


def _check_in(name: str, value: Any, allowed: tuple[Any, ...]) -> None:
    if value not in allowed:
        raise ValueError(f"{name}={value!r} not in {allowed}")


def _dataclass_to_dict(obj: Any) -> dict[str, Any]:
    """``asdict`` but converts tuples to lists for YAML-friendliness."""

    def _convert(v: Any) -> Any:
        if isinstance(v, tuple):
            return [_convert(x) for x in v]
        if isinstance(v, list):
            return [_convert(x) for x in v]
        if isinstance(v, dict):
            return {k: _convert(x) for k, x in v.items()}
        return v

    return _convert(asdict(obj))


def _is_tuple_annotation(annotation: Any) -> bool:
    """Return True iff ``annotation`` resolves to a ``tuple[...]`` type.

    Robust to both the stringly-typed annotations produced by
    ``from __future__ import annotations`` and real ``typing.get_type_hints``
    output. We try the real-type path first via ``get_origin``; if the
    annotation is a string (no module/globals context available here),
    fall back to a conservative substring check that catches
    ``tuple[...]`` and ``typing.Tuple[...]``.
    """
    import typing

    origin = typing.get_origin(annotation)
    if origin is tuple:
        return True
    if isinstance(annotation, str):
        # Conservative: only match top-level tuple annotations.
        stripped = annotation.strip()
        prefixes = ("tuple[", "Tuple[", "typing.Tuple[")
        return any(stripped.startswith(p) for p in prefixes)
    return False


def _construct_dataclass(cls: type, raw: dict[str, Any]) -> Any:
    """Build a dataclass from a dict, raising on unknown keys.

    Coerces list-valued YAML inputs into ``tuple`` for fields annotated
    as ``tuple[...]`` — required because YAML serialises tuples as lists,
    so ``to_yaml ∘ from_yaml`` would otherwise fail round-trip equality
    on fields like ``optimizer.betas`` or ``eval.eval_seeds``.
    """
    if not is_dataclass(cls):
        raise TypeError(f"{cls} is not a dataclass")
    valid = {f.name for f in fields(cls)}
    extra = set(raw.keys()) - valid
    if extra:
        raise KeyError(f"unknown fields for {cls.__name__}: {sorted(extra)}")
    field_types = {f.name: f.type for f in fields(cls)}
    coerced: dict[str, Any] = {}
    for k, v in raw.items():
        if isinstance(v, list) and _is_tuple_annotation(field_types[k]):
            coerced[k] = tuple(v)
        else:
            coerced[k] = v
    return cls(**coerced)
