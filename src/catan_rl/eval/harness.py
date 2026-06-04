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

import zlib
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from catan_rl.env.catan_env import CatanEnv
from catan_rl.eval.rules_invariants import run_all_invariants
from catan_rl.eval.wilson import WilsonInterval, wilson_interval

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
        """Run all configured matchups and return the aggregated report."""
        results: list[EvalResult] = []
        n_total = 0
        for opp in self.opponent_types:
            result = self._evaluate_matchup(policy=policy, opponent_type=opp)
            results.append(result)
            n_total += result.n
        return EvalReport(results=tuple(results), n_games_total=n_total)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _evaluate_matchup(self, *, policy: Any, opponent_type: str) -> EvalResult:
        """Play ``2 * n_games_per_seat`` games for one opponent kind."""
        env = CatanEnv(opponent_type=opponent_type, max_turns=self.max_turns)
        try:
            games: list[GameOutcome] = []
            # zlib.crc32 is a stable hash across Python processes;
            # ``hash()`` is salted by PYTHONHASHSEED and would break
            # cross-process reproducibility. base_seed and the per-game
            # offsets stay inside the 31-bit non-negative range expected
            # by ``np.random.seed``.
            opp_hash = zlib.crc32(opponent_type.encode("utf-8"))
            base_seed = (self.seed * 1_000_003 + opp_hash) % (2**31 - 1)
            for game_idx in range(self.n_games_per_seat):
                # Per-game seed derivation is deterministic in
                # ``self.seed`` and ``opponent_type`` so two harness
                # runs at the same ``seed`` produce bit-identical
                # outcomes — even across Python processes.
                seed0 = (base_seed + game_idx * 2) % (2**31 - 1)
                seed1 = (base_seed + game_idx * 2 + 1) % (2**31 - 1)
                # Seat 0 — agent goes first.
                games.append(self._play_one_game(env=env, policy=policy, seed=seed0, agent_seat=0))
                # Seat 1 — opponent goes first. ``CatanEnv`` honours
                # ``options={"agent_seat": 1}`` by flipping the queue
                # order and running the opponent's first setup +
                # first main turn before the agent acts. This is a
                # true symmetric pairing: same board distribution,
                # both seats represented equally.
                games.append(self._play_one_game(env=env, policy=policy, seed=seed1, agent_seat=1))
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
            obs_t = {k: torch.as_tensor(v, device=self.device).unsqueeze(0) for k, v in obs.items()}
            masks_t = {
                k: torch.as_tensor(v, device=self.device, dtype=torch.bool).unsqueeze(0)
                for k, v in masks.items()
            }
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
