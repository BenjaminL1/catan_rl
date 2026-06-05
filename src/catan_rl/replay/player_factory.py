"""Player-spec → Actor factory for the replay recorder.

Three kinds of player:

* ``"random"`` — random-masked-action player. Constructed cheaply
  with no IO.
* ``"heuristic"`` — :class:`catan_rl.agents.heuristic.heuristicAIPlayer`.
  Run-time state is per-game (``updateAI()`` is called after
  construction).
* ``"policy"`` — a trained :class:`catan_rl.policy.network.CatanPolicy`
  loaded from a Phase 8 checkpoint. The factory builds a fresh
  policy, calls ``set_board_geometry``, applies the saved
  ``policy_state_dict`` strictly (raises on shape mismatch), moves
  to ``device``, and switches to ``eval()`` mode.

Returns an :class:`Actor` regardless of kind, exposing a uniform
``select_action(obs, masks) -> np.ndarray`` surface for the recorder.

CLI validation (matchup-level) lives in ``scripts/record_game.py``;
this module only builds individual actors. The factory is fully
general: any of the 9 (kind_a, kind_b) pairs is constructible at this
layer. ``(policy, policy)`` is the only combination the recorder
script rejects (Phase 4), and that check happens upstream.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

import numpy as np

_LOG = logging.getLogger("catan_rl.replay")


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


PlayerKind = Literal["random", "heuristic", "policy"]


@dataclass(frozen=True, slots=True)
class RecorderPlayerSpec:
    """One side of a matchup — the construction-time parameter object
    for :func:`build_actor`. Deliberately named differently from
    :class:`catan_rl.replay.schema.PlayerSpec` (the JSON-serialised
    dataclass) to prevent accidental field-mismatch bugs at the
    recorder/schema boundary: the schema dataclass requires ``color``
    and ``seat_index`` which the recorder doesn't know at construction
    time."""

    kind: PlayerKind
    ckpt_path: str | None = None


#: Back-compat alias retained for the original Phase 1 API. New
#: callers should import :class:`RecorderPlayerSpec` directly.
PlayerSpec = RecorderPlayerSpec


class Actor(Protocol):
    """Uniform action-selection surface the recorder consumes.

    Random / heuristic actors don't actually use this method —
    they're driven by the env's ``opponent_type`` mechanism. Only the
    policy variant runs ``select_action`` on the recorder's side."""

    kind: PlayerKind

    def select_action(
        self,
        obs: dict[str, np.ndarray],
        masks: dict[str, np.ndarray],
    ) -> np.ndarray:
        """Return a (6,) int64 action vector."""
        ...


# ---------------------------------------------------------------------------
# Random + heuristic — engine-driven, no recorder-side action choice
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class _EngineDrivenActor:
    """For ``random`` and ``heuristic``: the env runs the engine's own
    action selection (via ``opponent_type`` for the opponent slot, or
    via a random-masked sample for the agent slot when the recorder
    drives a non-policy seat). ``select_action`` here is a uniformly-
    random legal action — used ONLY when the agent slot is a
    non-policy kind."""

    kind: PlayerKind
    _rng: np.random.Generator

    def select_action(
        self,
        obs: dict[str, np.ndarray],
        masks: dict[str, np.ndarray],
    ) -> np.ndarray:
        # Pick a uniformly-random legal type then random legal sub-args
        # per head. The buffer/policy layers don't care about non-
        # relevant heads (they're zero-weighted by relevance), so we
        # only need to ensure the *type* is legal.
        type_mask = masks["type"]
        legal = np.flatnonzero(type_mask)
        action = np.zeros(6, dtype=np.int64)
        if legal.size:
            action[0] = int(self._rng.choice(legal))
        else:
            action[0] = 3  # END_TURN fallback
        return action


# ---------------------------------------------------------------------------
# Policy — load checkpoint, build CatanPolicy, eval mode
# ---------------------------------------------------------------------------


#: Obs keys that the env returns as discrete `Discrete(...)` values —
#: must be cast to long (int64) before the policy's embedding lookup.
#: All other keys are continuous `Box(float32)` and stay as float32.
_DISCRETE_OBS_KEYS = ("opponent_kind", "opponent_policy_id")


@dataclass(slots=True)
class _PolicyActor:
    """Wraps a loaded :class:`CatanPolicy` and exposes
    ``select_action`` matching the recorder's contract.

    The actual sampling is delegated to ``policy.sample(obs, masks)``
    inside a ``torch.no_grad()`` context — same as the eval harness."""

    kind: PlayerKind  # always "policy"
    ckpt_path: str
    policy: Any  # CatanPolicy — typed Any here so static checks don't force torch
    device: Any  # torch.device

    def select_action(
        self,
        obs: dict[str, np.ndarray],
        masks: dict[str, np.ndarray],
    ) -> np.ndarray:
        # ``torch`` is guaranteed to be importable here because the
        # ``_PolicyActor`` is only ever constructed by ``build_actor``
        # which has already imported torch.
        import torch

        obs_t: dict[str, Any] = {}
        for k, v in obs.items():
            # Cast dtype at the boundary — the env's obs spec is
            # float32 for Box keys + int64 for Discrete keys, but the
            # recorder may receive arrays in arbitrary dtype from
            # debug harnesses / future BC pipelines. Explicit casts
            # protect against silent ``mat1/mat2 dtype mismatch``
            # errors inside the trunk's ``nn.Linear`` layers.
            if k in _DISCRETE_OBS_KEYS:
                arr = np.ascontiguousarray(v, dtype=np.int64)
            else:
                arr = np.ascontiguousarray(v, dtype=np.float32)
            obs_t[k] = torch.as_tensor(arr, device=self.device).unsqueeze(0)
        masks_t = {
            k: torch.as_tensor(
                np.ascontiguousarray(v, dtype=bool),
                device=self.device,
                dtype=torch.bool,
            ).unsqueeze(0)
            for k, v in masks.items()
        }
        with torch.no_grad():
            sample_out = self.policy.sample(obs_t, masks_t)
        return sample_out["action"][0].cpu().numpy().astype(np.int64)


# ---------------------------------------------------------------------------
# Device resolution
# ---------------------------------------------------------------------------


def _resolve_device(requested: str) -> str:
    """Resolve ``requested`` to a concrete device string.

    ``"auto"`` walks ``cuda → mps → cpu`` based on availability. If
    the user explicitly requests ``"cuda"`` or ``"mps"`` and that
    backend is not available, log a WARNING and fall back to CPU —
    NEVER raise. The recorder shouldn't crash on the wrong hardware.
    """
    import torch

    if requested == "auto":
        if torch.cuda.is_available():
            chosen = "cuda"
        elif torch.backends.mps.is_available():
            chosen = "mps"
        else:
            chosen = "cpu"
        _LOG.info("replay: --device auto resolved to %s", chosen)
        return chosen
    if requested == "cuda":
        if torch.cuda.is_available():
            return "cuda"
        _LOG.warning("replay: --device cuda requested but CUDA not available; falling back to CPU")
        return "cpu"
    if requested == "mps":
        if torch.backends.mps.is_available():
            return "mps"
        _LOG.warning("replay: --device mps requested but MPS not available; falling back to CPU")
        return "cpu"
    if requested == "cpu":
        return "cpu"
    raise ValueError(f"unknown device {requested!r}; expected one of auto / cpu / mps / cuda")


# ---------------------------------------------------------------------------
# Factory entry point
# ---------------------------------------------------------------------------


def build_actor(
    spec: PlayerSpec,
    *,
    seed: int,
    device: str = "cpu",
) -> Actor:
    """Construct the right :class:`Actor` for a player spec.

    Args:
        spec: kind + (optional) checkpoint path.
        seed: per-player RNG seed for the random-masked-action
            sampler (random/heuristic kinds only — policy is
            deterministic at fixed weights).
        device: ``"cpu"`` / ``"mps"`` / ``"cuda"`` / ``"auto"``. Only
            consulted for the ``policy`` kind.

    Raises:
        FileNotFoundError: if ``kind="policy"`` and ``ckpt_path``
            doesn't exist on disk.
        ValueError: if ``kind="policy"`` but ``ckpt_path`` is None.

    Returns an :class:`Actor`. The recorder owns the actor for the
    lifetime of one game.
    """
    if spec.kind in ("random", "heuristic"):
        rng = np.random.default_rng(seed)
        return _EngineDrivenActor(kind=spec.kind, _rng=rng)

    if spec.kind == "policy":
        if spec.ckpt_path is None:
            raise ValueError("kind='policy' requires ckpt_path to be set")
        ckpt = Path(spec.ckpt_path).expanduser().resolve()
        if not ckpt.exists():
            raise FileNotFoundError(f"checkpoint not found: {ckpt}")

        # Heavy imports are scoped here so consumers that only build
        # random/heuristic actors don't pay the torch import cost.
        import torch

        from catan_rl.checkpoint import load_checkpoint
        from catan_rl.policy.board_geometry import build_geometry
        from catan_rl.policy.network import CatanPolicy

        device_str = _resolve_device(device)
        torch_device = torch.device(device_str)

        policy = CatanPolicy()
        # Order matters: set the board geometry BEFORE moving to the
        # device. ``set_board_geometry`` writes into registered
        # buffers; the subsequent ``apply_to_policy(strict=True)``
        # will overwrite them with the checkpoint's saved buffers.
        # For the sanity_phase10 checkpoint that's an identity rewrite
        # (geometry doesn't change across training runs), so the
        # setter is redundant but harmless. We keep it so future
        # checkpoints that ship without geometry buffers (e.g., a BC
        # warm-start trained on a config that didn't capture them)
        # still get geometry from the live ``build_geometry`` call.
        geom = build_geometry()
        policy.set_board_geometry(geom.as_dict_of_tensors())
        policy = policy.to(torch_device)
        payload = load_checkpoint(ckpt, map_location=torch_device)
        payload.apply_to_policy(policy, strict=True)
        policy.eval()

        return _PolicyActor(
            kind="policy",
            ckpt_path=str(ckpt),
            policy=policy,
            device=torch_device,
        )

    raise ValueError(f"unknown kind {spec.kind!r}; expected random/heuristic/policy")
