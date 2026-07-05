"""Symmetric-seat eval harness for champion-vs-opponent matches.

Phase 7 of the v2 training-infra build-out. Each opponent matchup plays
``n_games_per_seat`` games from seat 0 plus the same number from seat
1, so the per-match WR is corrected for first-mover advantage
(snake-draft + 2nd-settlement-grants-starting-resources gives seat 2
~5pp WR in heuristic-vs-heuristic per the C.0 diagnostic). The seat
swap is real — ``CatanEnv.reset(options={"agent_seat": 1})`` flips the
playerQueue order, runs the opponent's first setup placement before
the agent's first action, and lets the opponent take the first main
turn after setup.

The result for one matchup is an :class:`EvalResult` carrying the
sample proportion, the Wilson CI, and per-seat breakdowns. The
aggregator :class:`EvalHarness` runs every configured opponent and
returns an :class:`EvalReport` covering all of them.

Contract for the policy:

* ``policy.sample(obs, masks) -> dict`` is the only required surface
  (matches Phase 5's ``RolloutCollector``). Returned actions are taken
  at the next env step.
* The harness runs the policy under ``torch.no_grad()`` for inference
  efficiency.

Contract for the env:

* :class:`catan_rl.env.catan_env.CatanEnv` with the configured
  ``opponent_type``. The harness owns env construction (one env per
  matchup; sequential games), so a Phase 5b vec env is NOT required
  for eval. Memory footprint is bounded by ``max_turns`` and a single
  policy forward batch.
"""

from __future__ import annotations

import random
import zlib
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from catan_rl.env.catan_env import CatanEnv
from catan_rl.eval.rules_invariants import run_all_invariants
from catan_rl.eval.wilson import WilsonInterval, wilson_interval
from catan_rl.policy.obs_tensor import masks_to_torch, obs_to_torch

# ---------------------------------------------------------------------------
# Torch RNG snapshot/restore (all backends)
# ---------------------------------------------------------------------------


def _snapshot_torch_rng() -> dict[str, Any]:
    """Snapshot every torch backend's RNG state (cpu + cuda + mps).

    ``torch.manual_seed`` reseeds EVERY backend's generator, not just the
    one eval samples on — so restoring only the CPU state would leave a
    learner's MPS/CUDA stream clobbered by the in-loop eval. Mirrors
    ``FrozenSnapshotOpponent._snapshot_rng``.
    """
    state: dict[str, Any] = {"cpu": torch.random.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    if (
        hasattr(torch, "mps")
        and torch.backends.mps.is_available()
        and hasattr(torch.mps, "get_rng_state")
    ):
        state["mps"] = torch.mps.get_rng_state()
    return state


def _restore_torch_rng(state: dict[str, Any]) -> None:
    torch.random.set_rng_state(state["cpu"])
    if "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])
    if "mps" in state:
        torch.mps.set_rng_state(state["mps"])


# ---------------------------------------------------------------------------
# Result schema
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GameOutcome:
    """One game's outcome from the agent's POV."""

    seed: int
    agent_seat: int
    """``0`` if agent went first in the snake draft, ``1`` if second."""

    won: bool
    """``True`` iff agent's VP >= 15 at game end. ``False`` for losses
    and truncations."""

    truncated: bool
    """``True`` iff the game hit ``max_turns`` without a winner."""

    final_vp_agent: int
    final_vp_opp: int
    n_turns: int
    rules_violations: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class EvalResult:
    """Aggregated outcomes for one (champion, opponent) matchup."""

    opponent_type: str
    games: tuple[GameOutcome, ...]
    wins: int
    n: int
    ci: WilsonInterval

    @property
    def wr(self) -> float:
        return self.ci.point

    @property
    def n_truncated(self) -> int:
        return sum(1 for g in self.games if g.truncated)

    @property
    def n_seat0(self) -> int:
        return sum(1 for g in self.games if g.agent_seat == 0)

    @property
    def n_seat1(self) -> int:
        return sum(1 for g in self.games if g.agent_seat == 1)

    @property
    def wr_seat0(self) -> float:
        s0 = [g for g in self.games if g.agent_seat == 0]
        if not s0:
            return float("nan")
        return sum(1 for g in s0 if g.won) / len(s0)

    @property
    def wr_seat1(self) -> float:
        s1 = [g for g in self.games if g.agent_seat == 1]
        if not s1:
            return float("nan")
        return sum(1 for g in s1 if g.won) / len(s1)

    @property
    def rules_violations(self) -> tuple[str, ...]:
        out: list[str] = []
        for g in self.games:
            out.extend(g.rules_violations)
        return tuple(out)


@dataclass(frozen=True)
class EvalMatchupResult(EvalResult):
    """Champion vs a loaded opponent policy (US2).

    EXTENDS :class:`EvalResult` — reuses all of its ``wr`` / Wilson-CI / seat
    breakdown machinery (no parallel type) and adds the opponent reference
    (checkpoint path or snapshot id). ``opponent_type`` is ``"snapshot"`` since
    the opponent is seated via the in-env snapshot-opponent driver.
    """

    opponent_ref: str = ""


@dataclass(frozen=True)
class EvalReport:
    """Aggregated results for all matchups in one eval round."""

    results: tuple[EvalResult, ...]
    n_games_total: int

    def by_opponent(self, opponent_type: str) -> EvalResult | None:
        for r in self.results:
            if r.opponent_type == opponent_type:
                return r
        return None


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------


class EvalHarness:
    """Run symmetric-seat champion-vs-opponent matchups.

    Stateless — construction parameters are config-only; ``run`` is the
    only public method and it does not mutate any harness state. The
    same harness can be reused for as many eval rounds as the trainer
    needs.
    """

    def __init__(
        self,
        *,
        opponent_types: tuple[str, ...] = ("random", "heuristic"),
        n_games_per_seat: int = 100,
        seed: int = 0,
        device: torch.device | str = "cpu",
        max_turns: int = 400,
        alpha: float = 0.05,
        audit_rules: bool = True,
    ) -> None:
        if n_games_per_seat <= 0:
            raise ValueError(f"n_games_per_seat must be > 0, got {n_games_per_seat}")
        if not opponent_types:
            raise ValueError("opponent_types must be non-empty")
        self.opponent_types = opponent_types
        self.n_games_per_seat = n_games_per_seat
        self.seed = seed
        self.device = torch.device(device) if isinstance(device, str) else device
        self.max_turns = max_turns
        self.alpha = alpha
        self.audit_rules = audit_rules

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, policy: Any) -> EvalReport:
        """Run all configured matchups and return the aggregated report.

        The policy is moved to ``self.device`` for the duration of the
        eval and restored to its original device afterwards. Eval runs
        the policy at **batch=1** (one env, sequential games) — the
        single regime where MPS is ~7-8x *slower* than CPU, because
        per-kernel launch overhead dominates a 1.4M-param forward and
        there is no batch dimension to amortise it. Pinning eval to CPU
        (``self.device``) while the learner trains on MPS/CUDA keeps the
        batched SGD update fast without paying the batch=1 device tax.
        The move is a no-op when the policy already lives on
        ``self.device`` (so existing CPU-only callers are unaffected),
        and is skipped entirely for parameter-less policy stubs.

        The policy is also switched to ``eval()`` mode for the round and
        restored to its prior mode afterwards (defensive: the encoder
        dropout is 0.0, but the toggle guards against any future stochastic
        layer perturbing eval WR). Restored in the ``finally``.

        The global numpy + stdlib ``random`` PRNGs — and every torch
        backend's RNG — are snapshotted and restored around the round: eval
        plays full games that consume the process-global streams
        (StackedDice, heuristic, steal/dev-card) and the policy samples
        actions through torch's global generator, and in-loop eval runs
        INSIDE the training loop — without this, whether/when eval runs
        would shift the subsequent rollout, and resume on a different
        eval cadence would diverge. Mirrors ``mask_spec_from_env`` and
        ``FrozenSnapshotOpponent``. The torch generator is additionally
        seeded from ``self.seed`` before the matchup loop so the
        champion's sampling stream — and therefore the whole report — is
        bit-identical across runs at a fixed seed (audit 2026-07: this
        was previously left on whatever entropy the process had, making
        eval results non-reproducible).
        """
        orig_device = self._policy_device(policy)
        moved = orig_device is not None and orig_device != self.device
        was_training = bool(getattr(policy, "training", False))
        np_state = np.random.get_state()
        py_state = random.getstate()
        torch_state = _snapshot_torch_rng()
        if moved:
            policy.to(self.device)
        if was_training:
            policy.eval()
        try:
            torch.manual_seed(self.seed % (2**31 - 1))
            results: list[EvalResult] = []
            n_total = 0
            for opp in self.opponent_types:
                result = self._evaluate_matchup(policy=policy, opponent_type=opp)
                results.append(result)
                n_total += result.n
            return EvalReport(results=tuple(results), n_games_total=n_total)
        finally:
            if moved:
                policy.to(orig_device)
            if was_training:
                policy.train()
            np.random.set_state(np_state)
            random.setstate(py_state)
            _restore_torch_rng(torch_state)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _policy_device(policy: Any) -> torch.device | None:
        """Best-effort current device of the policy's parameters.

        Returns ``None`` for a parameter-less stub (test doubles) or any
        object without a ``parameters()`` iterator, in which case
        :meth:`run` skips the device move entirely.
        """
        try:
            return next(policy.parameters()).device
        except (StopIteration, AttributeError):
            return None

    def _run_matchup_games(
        self, env: CatanEnv, policy: Any, *, seed_label: str
    ) -> list[GameOutcome]:
        """Play ``2 * n_games_per_seat`` seat-symmetrized games on ``env``.

        Seeds derive deterministically from ``self.seed`` + ``seed_label`` so
        two runs at the same seed are bit-identical (even cross-process —
        ``zlib.crc32`` is stable where ``hash()`` is PYTHONHASHSEED-salted).
        """
        games: list[GameOutcome] = []
        opp_hash = zlib.crc32(seed_label.encode("utf-8"))
        base_seed = (self.seed * 1_000_003 + opp_hash) % (2**31 - 1)
        for game_idx in range(self.n_games_per_seat):
            seed0 = (base_seed + game_idx * 2) % (2**31 - 1)
            seed1 = (base_seed + game_idx * 2 + 1) % (2**31 - 1)
            # Seat 0 — agent first; seat 1 — opponent first (true symmetric
            # pairing: same board distribution, both seats represented).
            games.append(self._play_one_game(env=env, policy=policy, seed=seed0, agent_seat=0))
            games.append(self._play_one_game(env=env, policy=policy, seed=seed1, agent_seat=1))
        return games

    def _evaluate_matchup(self, *, policy: Any, opponent_type: str) -> EvalResult:
        """Play ``2 * n_games_per_seat`` games for one opponent kind."""
        env = CatanEnv(opponent_type=opponent_type, max_turns=self.max_turns)
        try:
            games = self._run_matchup_games(env, policy, seed_label=opponent_type)
            wins = sum(1 for g in games if g.won)
            ci = wilson_interval(wins=wins, n=len(games), alpha=self.alpha)
            return EvalResult(
                opponent_type=opponent_type,
                games=tuple(games),
                wins=wins,
                n=len(games),
                ci=ci,
            )
        finally:
            env.close()

    def evaluate_vs_policy(
        self, champion: Any, opponent: Any, *, opponent_ref: str
    ) -> EvalMatchupResult:
        """Champion vs a loaded opponent policy (US2).

        ``opponent`` is a :class:`~catan_rl.selfplay.snapshot_opponent.SnapshotOpponent`
        (e.g. a ``FrozenSnapshotOpponent``); it is seated via the in-env
        snapshot-opponent driver (NOT the recorder actor). Seat-symmetrized;
        returns WR + Wilson CI.
        """
        env = CatanEnv(opponent_type="snapshot", max_turns=self.max_turns)
        env.set_snapshot_opponent(opponent)
        try:
            games = self._run_matchup_games(env, champion, seed_label=opponent_ref)
            wins = sum(1 for g in games if g.won)
            ci = wilson_interval(wins=wins, n=len(games), alpha=self.alpha)
            return EvalMatchupResult(
                opponent_type="snapshot",
                games=tuple(games),
                wins=wins,
                n=len(games),
                ci=ci,
                opponent_ref=opponent_ref,
            )
        finally:
            env.close()

    @torch.no_grad()
    def _play_one_game(
        self,
        *,
        env: CatanEnv,
        policy: Any,
        seed: int,
        agent_seat: int,
    ) -> GameOutcome:
        """Play one game; return the outcome from the agent's POV.

        ``agent_seat`` is passed into ``env.reset(options=...)`` so the
        env honours it as a real seat swap (queue order + setup
        ordering + first main-turn ordering all flip when seat=1).
        """
        obs, _ = env.reset(seed=seed, options={"agent_seat": agent_seat})
        masks = env.get_action_masks()
        # ``n_env_steps`` counts every env.step() call, not agent main
        # turns. A single agent turn can drive ~10-20 env steps
        # (settlements, dev cards, discards, robber sub-loop).
        n_env_steps = 0
        terminated = False
        truncated = False
        # Generous safety bound: max_turns counts agent main turns; the
        # env handles real truncation through ``_turn_count >=
        # max_turns``. The bound below is a wall-clock backstop only,
        # for the case where the env never reaches an END_TURN action.
        # Setting it 50x lets a worst-case ~50-step/turn game still
        # reach env truncation honestly before the safety fires.
        safety_cap = self.max_turns * 50
        hit_safety = False
        while not terminated and not truncated:
            obs_t = obs_to_torch(obs, self.device, add_batch=True)
            masks_t = masks_to_torch(masks, self.device, add_batch=True)
            sample_out = policy.sample(obs_t, masks_t)
            action = sample_out["action"][0].cpu().numpy().astype(np.int64)
            obs, _, terminated, truncated, _ = env.step(action)
            masks = env.get_action_masks()
            n_env_steps += 1
            if n_env_steps > safety_cap:
                # Wall-clock backstop. Treated as a truncation so the
                # accounting in ``EvalResult.n_truncated`` stays
                # honest.
                truncated = True
                hit_safety = True
                break

        # Read VPs directly from the env's tracked players.
        assert env.game is not None
        agent = env.agent_player
        opp = env.opponent_player
        assert agent is not None and opp is not None
        agent_vp = int(getattr(agent, "victoryPoints", 0))
        opp_vp = int(getattr(opp, "victoryPoints", 0))
        # Win iff the agent reached the 1v1 cap and the opponent didn't
        # tie at the same step. Safety-break paths are always losses.
        won = (not hit_safety) and agent_vp >= 15 and agent_vp > opp_vp

        violations: tuple[str, ...] = ()
        if self.audit_rules:
            violations = tuple(run_all_invariants(env.game, truncated=truncated))

        return GameOutcome(
            seed=seed,
            agent_seat=agent_seat,
            won=won,
            truncated=truncated,
            final_vp_agent=agent_vp,
            final_vp_opp=opp_vp,
            n_turns=n_env_steps,
            rules_violations=violations,
        )


# ---------------------------------------------------------------------------
# Policy-vs-policy eval (US2)
# ---------------------------------------------------------------------------


def evaluate_policy_vs_policy(
    champion: Any,
    opponent_ckpt: str,
    *,
    n_games: int = 100,
    seed: int = 0,
    device: str = "cpu",
    max_turns: int = 400,
) -> EvalMatchupResult:
    """Evaluate ``champion`` head-to-head against a loaded opponent checkpoint.

    The opponent checkpoint is loaded into a frozen ``CatanPolicy`` by reusing
    ``replay/player_factory.build_actor`` (no new loader), wrapped as a
    ``FrozenSnapshotOpponent``, and seated via the in-env snapshot-opponent
    driver. Plays ``2 * (n_games // 2)`` seat-symmetrized games; returns WR +
    Wilson CI. Bit-for-bit reproducible on CPU at a fixed seed.

    Note: this seeds the global torch RNG for a reproducible champion sampling
    stream, but **saves and restores** every torch backend's state (cpu + cuda
    + mps — ``manual_seed`` reseeds them all) plus the global numpy/stdlib
    streams the games consume — so it is safe to invoke from inside a training
    loop without clobbering the learner's RNG. Mirrors ``EvalHarness.run``.
    """
    from typing import cast

    from catan_rl.replay.player_factory import PlayerSpec, _PolicyActor, build_actor
    from catan_rl.selfplay.snapshot_opponent import FrozenSnapshotOpponent

    np_state = np.random.get_state()
    py_state = random.getstate()
    torch_state = _snapshot_torch_rng()
    try:
        torch.manual_seed(seed)  # reproducible champion sampling stream
        # kind="policy" always yields a _PolicyActor (.policy / .device typed).
        actor = cast(
            _PolicyActor,
            build_actor(
                PlayerSpec(kind="policy", ckpt_path=str(opponent_ckpt)),
                seed=seed,
                device=device,
            ),
        )
        opponent = FrozenSnapshotOpponent(actor.policy, device=actor.device, seed=seed)
        harness = EvalHarness(
            opponent_types=("snapshot",),
            n_games_per_seat=max(1, n_games // 2),
            seed=seed,
            device=torch.device(device),
            max_turns=max_turns,
        )
        return harness.evaluate_vs_policy(champion, opponent, opponent_ref=str(opponent_ckpt))
    finally:
        np.random.set_state(np_state)
        random.setstate(py_state)
        _restore_torch_rng(torch_state)
