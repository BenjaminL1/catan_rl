"""Cross-architecture head-to-head evaluation (new pointer-arch vs v11-era legacy).

The pointer-arch fork changed BOTH the policy network shape (pointer heads +
aux-value head) AND the obs schema (``CURR_PLAYER_DIM`` 54->67, a new POV-neutral
``global_features`` block, an ``is_setup`` flag). A v11-era checkpoint therefore
cannot be loaded by the current :class:`~catan_rl.policy.network.CatanPolicy`
(strict state-dict load raises), so the in-process
:func:`~catan_rl.eval.harness.evaluate_policy_vs_policy` cannot pit the new arch
against v11 — which is the ratified accept gate (clause a: "h2h vs v11_cand
Wilson-LB > 0.50 at n=600") and the only non-saturated signal that self-play is
surpassing v11.

**Why this works in one process.** The engine + board geometry are BYTE-IDENTICAL
across the fork (verified: ``git diff 9692a79~1 HEAD -- src/catan_rl/engine/`` and
``.../board_geometry.py`` are both empty), and an obs is a pure function of game
state. So both policies play the SAME live game: the new-arch champion drives the
agent seat and reads the env's native (new-schema) obs, while v11 drives the
opponent seat and reads a LEGACY-schema obs built by the vendored legacy encoder
(:mod:`catan_rl.eval.legacy_arch`) from the same shared game state. The action
space + masks are unchanged across the fork, so the env's live masks feed either
policy unmodified.

This module is READ-ONLY w.r.t. training: it touches no engine rule, no obs
schema, no training config. :class:`CrossArchEnv` subclasses :class:`CatanEnv`
and overrides ONLY the opponent's obs-build seam (``_sample_snapshot_action``),
leaving every production code path untouched.
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
import torch

from catan_rl.env.catan_env import CatanEnv
from catan_rl.eval.engine_parity import assert_engine_parity
from catan_rl.eval.harness import (
    EvalHarness,
    EvalMatchupResult,
    _restore_torch_rng,
    _snapshot_torch_rng,
)
from catan_rl.eval.wilson import wilson_interval
from catan_rl.policy.network import CatanPolicy
from catan_rl.selfplay.snapshot_opponent import FrozenSnapshotOpponent

if TYPE_CHECKING:
    from catan_rl.policy.obs_encoder import EnvObsState

__all__ = ["CrossArchEnv", "build_legacy_opponent", "cross_arch_h2h"]

OpponentArch = Literal["legacy", "new"]


# ---------------------------------------------------------------------------
# Policy construction
# ---------------------------------------------------------------------------


def _build_new_policy(ckpt_path: str, *, device: str, seed: int) -> CatanPolicy:
    """Load a current pointer-arch checkpoint into a fresh ``CatanPolicy``.

    Reuses ``replay.player_factory.build_actor`` (the same strict loader the eval
    harness uses) and returns the underlying policy, already on ``device`` and in
    ``eval()`` mode.
    """
    from catan_rl.replay.player_factory import PlayerSpec, _PolicyActor, build_actor

    actor = cast(
        _PolicyActor,
        build_actor(PlayerSpec(kind="policy", ckpt_path=ckpt_path), seed=seed, device=device),
    )
    return cast(CatanPolicy, actor.policy)


def _build_legacy_policy(ckpt_path: str, *, device: str) -> Any:
    """Load a v11-era checkpoint into the vendored legacy ``CatanPolicy``.

    Mirrors ``build_actor``'s load order exactly (construct -> set geometry ->
    move to device -> strict state-dict load -> ``eval()``) but against the
    frozen pre-fork architecture in :mod:`catan_rl.eval.legacy_arch`. The strict
    load is the correctness guard: it raises unless every v11 tensor maps onto a
    legacy parameter of identical shape.
    """
    from catan_rl.checkpoint import load_checkpoint
    from catan_rl.eval.legacy_arch.network import CatanPolicy as LegacyCatanPolicy
    from catan_rl.policy.board_geometry import build_geometry
    from catan_rl.replay.player_factory import _resolve_device

    torch_device = torch.device(_resolve_device(device))
    policy = LegacyCatanPolicy()
    policy.set_board_geometry(build_geometry().as_dict_of_tensors())
    policy = policy.to(torch_device)
    payload = load_checkpoint(ckpt_path, map_location=torch_device)
    payload.apply_to_policy(policy, strict=True)
    policy.eval()
    return policy


def build_legacy_opponent(
    ckpt_path: str, *, device: str = "cpu", seed: int = 0
) -> FrozenSnapshotOpponent:
    """Build a frozen opponent wrapping a v11-era (legacy-arch) checkpoint.

    Returns a :class:`FrozenSnapshotOpponent` (reused verbatim for its RNG
    isolation + per-call reproducibility) around the vendored legacy policy. The
    wrapper is duck-typed — it only calls ``policy.sample`` — so the legacy
    ``CatanPolicy`` slots in via a cast without any behavioural change.
    """
    legacy_policy = _build_legacy_policy(ckpt_path, device=device)
    return FrozenSnapshotOpponent(
        cast(CatanPolicy, legacy_policy),
        device=torch.device(next(legacy_policy.parameters()).device),
        seed=seed,
    )


# ---------------------------------------------------------------------------
# Env: legacy-obs opponent seat
# ---------------------------------------------------------------------------


class CrossArchEnv(CatanEnv):
    """:class:`CatanEnv` whose snapshot opponent reads a LEGACY-schema obs.

    When ``legacy_opponent=True``, the single opponent-decision seam
    (:meth:`_sample_snapshot_action`) builds the opponent-POV obs with the
    vendored legacy encoder instead of the env's native (new-schema) encoder;
    every other code path — the agent's obs, masks, the whole game state machine
    — is the unmodified parent. When ``legacy_opponent=False`` the override is a
    pure pass-through, so a new-arch opponent behaves EXACTLY like a plain
    ``CatanEnv`` (the equivalence guarantee the cross-arch test pins).

    The legacy encoder is built lazily and cached on board identity: the env
    rebuilds its board every ``reset``, so a fresh encoder is constructed
    whenever the board object changes, matching the parent's encoder lifetime.
    """

    def __init__(
        self,
        opponent_type: str = "snapshot",
        max_turns: int = 500,
        vp_margin_bonus: float = 1.0 / 15.0,
        engine_backend: str = "python",
        *,
        legacy_opponent: bool = False,
    ) -> None:
        super().__init__(
            opponent_type=opponent_type,
            max_turns=max_turns,
            vp_margin_bonus=vp_margin_bonus,
            engine_backend=engine_backend,
        )
        self._legacy_opponent = legacy_opponent
        self._legacy_obs_encoder: Any = None
        self._legacy_encoder_board: Any = None

    def _sample_snapshot_action(self, env_state: EnvObsState) -> np.ndarray:
        if not self._legacy_opponent:
            return super()._sample_snapshot_action(env_state)

        # Lazy imports keep torch/legacy-arch off the module-level engine path.
        from catan_rl.eval.legacy_arch.obs_encoder import ObsEncoder as LegacyObsEncoder
        from catan_rl.policy.obs_tensor import masks_to_torch, obs_to_torch

        assert self._snapshot_opponent is not None
        assert self.opponent_player is not None and self.agent_player is not None
        assert self.game is not None

        board = self.game.board
        if self._legacy_encoder_board is not board:
            self._legacy_obs_encoder = LegacyObsEncoder(board)
            self._legacy_encoder_board = board

        # Masks are schema-independent (the action space + mask keys are
        # unchanged across the fork) — the parent's live masks feed v11 as-is.
        masks = self._compute_masks(self.opponent_player, env_state)
        # Opponent-POV LEGACY obs from the SAME shared game state (opponent sees
        # its own hand; the agent contributes only observable info — identical
        # POV contract to the parent's ``_build_obs_for``).
        obs = self._legacy_obs_encoder.build_obs(
            self.game,
            self.opponent_player,
            self.agent_player,
            env_state,
            hand_tracker=self._hand_tracker,
        )
        device = self._snapshot_opponent.device
        obs_t = obs_to_torch(obs, device, add_batch=True)
        masks_t = masks_to_torch(masks, device, add_batch=True)
        action_t = self._snapshot_opponent.sample(obs_t, masks_t)
        return action_t[0].detach().cpu().numpy().astype(np.int64)


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------


def cross_arch_h2h(
    *,
    new_ckpt: str,
    old_ckpt: str,
    old_arch: OpponentArch = "legacy",
    n_games: int = 100,
    seed: int = 0,
    device: str = "cpu",
    max_turns: int = 400,
    strict_engine_parity: bool = True,
) -> EvalMatchupResult:
    """Seat-symmetrized head-to-head: new pointer-arch champion vs an old policy.

    The new-arch policy (``new_ckpt``) always drives the AGENT seat and reads the
    env's native new-schema obs; the old policy (``old_ckpt``) drives the OPPONENT
    seat. With ``old_arch="legacy"`` (the v11 case) the opponent reads a
    legacy-schema obs via :class:`CrossArchEnv`; with ``old_arch="new"`` the
    opponent is a current-arch policy read through the plain snapshot path — used
    by the equivalence test (new-vs-new through this harness == the in-process
    ``evaluate_policy_vs_policy`` result).

    Default ``n_games=100`` (seat-symmetrized) is the fast self-play progress
    check; the ratified accept gate runs the same call at ``n_games=600``.

    Before doing anything, :func:`~catan_rl.eval.engine_parity.assert_engine_parity`
    refuses to run if the live engine has drifted from the pre-fork tree the
    vendored legacy encoder was written against (``strict_engine_parity=False``
    bypasses — untrusted).

    Returns an :class:`EvalMatchupResult`: the NEW arch's seat-symmetrized win
    rate + Wilson CI over ``2 * (n_games // 2)`` games. Bit-for-bit reproducible
    on CPU at a fixed seed; the global numpy / stdlib / every-torch-backend RNG
    state is snapshotted and restored, so it is safe to call from anywhere.
    """
    if old_arch not in ("legacy", "new"):
        raise ValueError(f"old_arch must be 'legacy' or 'new'; got {old_arch!r}")
    # Guard 3: the whole in-process method rests on the engine being byte-
    # identical across the fork — refuse (or warn, if unverifiable) up front.
    assert_engine_parity(strict=strict_engine_parity)

    np_state = np.random.get_state()
    py_state = random.getstate()
    torch_state = _snapshot_torch_rng()
    try:
        torch.manual_seed(seed)  # reproducible champion sampling stream
        champion = _build_new_policy(new_ckpt, device=device, seed=seed)
        if old_arch == "legacy":
            opponent = build_legacy_opponent(old_ckpt, device=device, seed=seed)
        else:
            opp_policy = _build_new_policy(old_ckpt, device=device, seed=seed)
            opponent = FrozenSnapshotOpponent(opp_policy, device=torch.device(device), seed=seed)

        env = CrossArchEnv(
            opponent_type="snapshot",
            max_turns=max_turns,
            legacy_opponent=(old_arch == "legacy"),
        )
        env.set_snapshot_opponent(opponent)
        harness = EvalHarness(
            opponent_types=("snapshot",),
            n_games_per_seat=max(1, n_games // 2),
            seed=seed,
            device=torch.device(device),
            max_turns=max_turns,
        )
        try:
            games = harness._run_matchup_games(env, champion, seed_label=str(old_ckpt))
        finally:
            env.close()

        wins = sum(1 for g in games if g.won)
        ci = wilson_interval(wins=wins, n=len(games), alpha=0.05)
        return EvalMatchupResult(
            opponent_type="snapshot",
            games=tuple(games),
            wins=wins,
            n=len(games),
            ci=ci,
            opponent_ref=str(old_ckpt),
        )
    finally:
        np.random.set_state(np_state)
        random.setstate(py_state)
        _restore_torch_rng(torch_state)
