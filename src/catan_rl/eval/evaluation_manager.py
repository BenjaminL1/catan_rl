"""Evaluation manager: deterministic eval games, H2H, and exploitability hook.

Phase 0 extends the previous (non-deterministic) ``evaluate`` with:

  - ``seeds=...`` argument that fully seeds NumPy / Python ``random`` /
    ``PYTHONHASHSEED`` / Torch RNGs per game so the same seed list reproduces
    bit-identical games.
  - ``evaluate_h2h(policy_a, policy_b, seeds, swap_first_player=True)`` that
    plays each seed twice — once with each policy first — to cancel
    first-mover bias in the snake-draft Catan variant.
  - ``compute_exploitability_stub`` that documents the contract for the
    Phase 0 exploitability runner; the actual training of the fresh adversary
    is in ``catan_rl.eval.exploitability``.

The legacy ``evaluate(policy, device)`` signature still works (it now seeds
internally with the default range so it is implicitly deterministic too).
"""

from __future__ import annotations

import os
import random
from collections.abc import Iterable, Sequence

import numpy as np
import torch

from catan_rl.env.catan_env import CatanEnv


def _seed_everything(seed: int) -> None:
    """Reset all global RNGs so a single int controls game determinism.

    Five-way seeding required for full reproducibility:
      - NumPy (board layout, dice, port shuffle, robber tie-breaks)
      - Python ``random`` (StackedDice, opponent heuristics)
      - ``PYTHONHASHSEED`` (set-iteration order on some Python versions)
      - PyTorch CPU (policy sampling under deterministic=False)
      - PyTorch CUDA (no-op on CPU; future-proofs CUDA path)
    """
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():  # pragma: no cover - CPU baseline
        torch.cuda.manual_seed_all(seed)


def _obs_to_tensor_dict(obs: dict, device: str) -> dict[str, torch.Tensor]:
    """Convert one env observation dict to a B=1 tensor dict for ``policy.act``."""

    def _dev_seq(key: str) -> list[torch.Tensor]:
        seq = obs.get(key)
        if seq is None:
            return [torch.zeros(1, dtype=torch.long, device=device)]
        arr = np.asarray(seq, dtype=np.int64)
        if arr.size == 0:
            arr = np.zeros(1, dtype=np.int64)
        return [torch.tensor(arr, dtype=torch.long, device=device)]

    return {
        "tile_representations": torch.tensor(
            obs["tile_representations"], dtype=torch.float32, device=device
        ).unsqueeze(0),
        "current_player_main": torch.tensor(
            obs["current_player_main"], dtype=torch.float32, device=device
        ).unsqueeze(0),
        "next_player_main": torch.tensor(
            obs["next_player_main"], dtype=torch.float32, device=device
        ).unsqueeze(0),
        "current_player_hidden_dev": _dev_seq("current_player_hidden_dev"),
        "current_player_played_dev": _dev_seq("current_player_played_dev"),
        "next_player_played_dev": _dev_seq("next_player_played_dev"),
    }


def _masks_to_tensor(masks: dict, device: str) -> dict[str, torch.Tensor]:
    return {
        k: torch.tensor(v, dtype=torch.bool, device=device).unsqueeze(0) for k, v in masks.items()
    }


def _play_one_game(
    env: CatanEnv,
    policy: torch.nn.Module,
    device: str,
    *,
    seed: int | None,
    deterministic: bool = True,
    options: dict | None = None,
    opponent_policy: torch.nn.Module | None = None,
) -> dict:
    """Play one full game with ``policy`` as the agent.

    Handles both kinds of opponents:
      - Random/heuristic opponents (no extra plumbing — env runs them inline).
      - Policy opponents (deferred NN inference): when env.step returns
        ``info['opp_turn_pending']``, this drives ``apply_opponent_action``
        with ``opponent_policy`` until the opponent's turn completes.

    Returns the terminal info dict from the *last* env transition.
    """
    if opponent_policy is None and options is not None:
        opponent_policy = options.get("opponent_policy")

    if seed is not None:
        _seed_everything(seed)
    obs, _ = env.reset(seed=seed, options=options or {})
    terminated = truncated = False
    info: dict = {}
    while not (terminated or truncated):
        masks = env.get_action_masks()
        with torch.no_grad():
            actions, _, _ = policy.act(
                _obs_to_tensor_dict(obs, device),
                _masks_to_tensor(masks, device),
                deterministic=deterministic,
            )
        action_np = actions.squeeze(0).cpu().numpy()
        obs, _, terminated, truncated, info = env.step(action_np)

        # Drive deferred opponent NN turn (if any) before the next agent step.
        # The env returns opp_turn_pending=True after the agent ends turn; the
        # caller must repeatedly call apply_opponent_action until turn_complete.
        while info.get("opp_turn_pending"):
            if opponent_policy is None:
                raise RuntimeError(
                    "env returned opp_turn_pending but no opponent_policy was provided"
                )
            opp_obs, opp_masks = env.get_opponent_obs_masks()
            with torch.no_grad():
                opp_actions, _, _ = opponent_policy.act(
                    _obs_to_tensor_dict(opp_obs, device),
                    _masks_to_tensor(opp_masks, device),
                    deterministic=deterministic,
                )
            opp_action_np = opp_actions.squeeze(0).cpu().numpy()
            (
                turn_complete,
                obs_after,
                _reward_delta,
                terminated,
                truncated,
                info,
            ) = env.apply_opponent_action(opp_action_np)
            if turn_complete:
                obs = obs_after
                break
    return info


class EvaluationManager:
    """Run deterministic evaluation games and report statistics.

    Phase 0 adds optional ``seeds`` and ``evaluate_h2h`` for the eval harness;
    the legacy ``evaluate(policy, device)`` call shape is preserved.
    """

    def __init__(
        self,
        n_games: int = 20,
        opponent_type: str = "random",
        max_turns: int | None = 500,
        use_thermometer_encoding: bool = True,
    ):
        self.n_games = n_games
        self.opponent_type = opponent_type
        self.max_turns = max_turns
        # Phase 1.3: must match the policy's expected obs schema.
        self.use_thermometer_encoding = bool(use_thermometer_encoding)

    # ── Default eval ─────────────────────────────────────────────────────

    def evaluate(
        self,
        policy: torch.nn.Module,
        device: str = "cpu",
        *,
        seeds: Sequence[int] | None = None,
        deterministic: bool = True,
    ) -> dict[str, float]:
        """Play games with ``policy`` (deterministic by default) and return stats.

        Args:
            policy: CatanPolicy instance (set to eval mode externally).
            device: torch device string.
            seeds: Optional list of seeds, one per game. When provided, length
                determines the number of games (``n_games`` is ignored). When
                None, falls back to the legacy non-seeded behavior — but each
                game is still played to completion.
            deterministic: When True (default), policy.act picks argmax;
                when False, it samples (used by exploitability adversary
                training, not by the harness).

        Returns:
            ``{'win_rate', 'avg_vp', 'avg_game_length', 'truncation_rate'}``.
        """
        env = CatanEnv(
            render_mode=None,
            opponent_type=self.opponent_type,
            max_turns=self.max_turns,
            use_thermometer_encoding=self.use_thermometer_encoding,
        )
        n = len(seeds) if seeds is not None else self.n_games
        wins = 0
        truncations = 0
        total_vp = 0
        total_length = 0

        for i in range(n):
            seed = int(seeds[i]) if seeds is not None else None
            info = _play_one_game(env, policy, device, seed=seed, deterministic=deterministic)
            stats = info.get("terminal_stats", {})
            if info.get("is_success"):
                wins += 1
            # Truncation: the env's terminal info has no winner — both VPs < 15.
            agent_vp = stats.get("agent_vp", 0)
            opp_vp = stats.get("opponent_vp", 0)
            if agent_vp < 15 and opp_vp < 15:
                truncations += 1
            total_vp += agent_vp
            total_length += stats.get("game_length", 0)

        return {
            "win_rate": wins / max(n, 1),
            "avg_vp": total_vp / max(n, 1),
            "avg_game_length": total_length / max(n, 1),
            "truncation_rate": truncations / max(n, 1),
        }

    # ── Head-to-head: policy A vs policy B with first-mover swap ─────────

    def evaluate_h2h(
        self,
        policy_a: torch.nn.Module,
        policy_b: torch.nn.Module,
        seeds: Sequence[int],
        device: str = "cpu",
        *,
        swap_first_player: bool = True,
        deterministic: bool = True,
    ) -> dict[str, float]:
        """Play each seed twice (A first, then B first) so first-mover bias cancels.

        Catan's snake-draft setup is asymmetric: P1 picks first and last
        settlements; P2 picks the middle two and starts the main game with the
        better expected-value position. Reporting WR over both orderings makes
        the H2H number a clean estimate of relative skill.

        Args:
            policy_a: Candidate (the policy we want to score).
            policy_b: Reference (e.g., the frozen champion).
            seeds: List of integer seeds; each played twice when
                ``swap_first_player=True``.
            device: torch device string.
            swap_first_player: When True, run each seed in both orderings.
                When False, only ``policy_a`` is the agent in each game.
            deterministic: argmax action selection (default).

        Returns:
            Dict with ``win_rate_a``, ``win_rate_b``, ``draw_rate``,
            ``avg_a_vp``, ``avg_b_vp``, ``avg_length``, ``n_games``.
        """
        env_a = CatanEnv(
            render_mode=None,
            opponent_type="policy",
            max_turns=self.max_turns,
            use_thermometer_encoding=self.use_thermometer_encoding,
        )
        env_b = (
            CatanEnv(
                render_mode=None,
                opponent_type="policy",
                max_turns=self.max_turns,
                use_thermometer_encoding=self.use_thermometer_encoding,
            )
            if swap_first_player
            else None
        )

        a_wins = 0
        b_wins = 0
        draws = 0
        total_a_vp = 0
        total_b_vp = 0
        total_length = 0
        n_games = 0

        for seed in seeds:
            seed = int(seed)
            # Orientation 1: A is the agent (P1), B is the opponent (P2).
            info = _play_one_game(
                env_a,
                policy_a,
                device,
                seed=seed,
                deterministic=deterministic,
                options={"opponent_type": "policy", "opponent_policy": policy_b},
            )
            n_games += 1
            stats = info.get("terminal_stats", {})
            a_vp = stats.get("agent_vp", 0)
            b_vp = stats.get("opponent_vp", 0)
            total_a_vp += a_vp
            total_b_vp += b_vp
            total_length += stats.get("game_length", 0)
            if info.get("is_success"):
                a_wins += 1
            elif a_vp >= 15 or b_vp >= 15:
                b_wins += 1
            else:
                draws += 1

            if env_b is not None:
                # Orientation 2: B is the agent (P1), A is the opponent (P2).
                info2 = _play_one_game(
                    env_b,
                    policy_b,
                    device,
                    seed=seed,
                    deterministic=deterministic,
                    options={"opponent_type": "policy", "opponent_policy": policy_a},
                )
                n_games += 1
                stats2 = info2.get("terminal_stats", {})
                # In env_b, "agent" is policy_b and "opponent" is policy_a.
                b_vp2 = stats2.get("agent_vp", 0)
                a_vp2 = stats2.get("opponent_vp", 0)
                total_a_vp += a_vp2
                total_b_vp += b_vp2
                total_length += stats2.get("game_length", 0)
                if info2.get("is_success"):
                    b_wins += 1
                elif b_vp2 >= 15 or a_vp2 >= 15:
                    a_wins += 1
                else:
                    draws += 1

        denom = max(n_games, 1)
        return {
            "win_rate_a": a_wins / denom,
            "win_rate_b": b_wins / denom,
            "draw_rate": draws / denom,
            "avg_a_vp": total_a_vp / denom,
            "avg_b_vp": total_b_vp / denom,
            "avg_length": total_length / denom,
            "n_games": float(n_games),
        }


def standard_eval_seeds(start: int = 0, end: int = 200) -> list[int]:
    """Default deterministic seed list for champion-bench (frozen at Phase 0)."""
    return list(range(start, end))


def iter_seeded_games(seeds: Iterable[int]):  # pragma: no cover - thin generator
    """Yield each seed after seeding all global RNGs (helper for callers
    that drive the env themselves rather than going through ``EvaluationManager``)."""
    for s in seeds:
        _seed_everything(int(s))
        yield int(s)
