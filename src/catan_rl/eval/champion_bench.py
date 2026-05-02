"""Frozen-champion benchmark: H2H vs ``checkpoint_07390040.pt`` over 200 seeds.

This is success criterion C2 in the roadmap:

  Beats previous best champion >= 70% over 200 deterministic eval games.

The bench is a thin orchestrator:

  1. Load the candidate policy (from checkpoint).
  2. Load the frozen champion (from a fixed path).
  3. Run ``EvaluationManager.evaluate_h2h`` over the deterministic seed list,
     swapping first-mover.
  4. Compute the candidate's win rate plus a 95% confidence interval (Wilson
     score interval — well-calibrated even when WR is near 0 or 1).
  5. Emit a structured JSON record to ``runs/eval_harness/<run_name>/``.

The Wilson interval (rather than a normal-approximation interval) is used
because at 200 games a 99% champion-bench WR has tight tail probability —
normal-approx undershoots tail width and gives misleadingly tight CIs.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from catan_rl.algorithms.ppo.trainer import CatanPPO
from catan_rl.eval.evaluation_manager import EvaluationManager, standard_eval_seeds


@dataclass(frozen=True)
class ChampionBenchResult:
    """Champion-bench summary suitable for JSON serialization."""

    candidate_path: str
    champion_path: str
    n_seeds: int
    swap_first_player: bool
    n_games: int
    win_rate: float
    """Fraction of games the candidate won (truncations counted as draws)."""
    win_rate_ci_low: float
    win_rate_ci_high: float
    """95% Wilson score interval bounds."""
    draw_rate: float
    """Truncation share — neither side reached 15 VP within max_turns."""
    avg_candidate_vp: float
    avg_champion_vp: float
    avg_length: float
    threshold: float
    """Phase-3 success threshold for the candidate WR (default 0.70)."""
    passed: bool
    """``win_rate_ci_low >= threshold`` — pass even the conservative bound."""


def wilson_interval(successes: int, n: int, *, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a binomial proportion.

    More accurate than the normal-approximation Wald interval when the
    proportion is near 0 or 1, which is exactly the regime we care about
    when scoring a candidate against a known weaker / stronger reference.
    """
    if n <= 0:
        return 0.0, 1.0
    p = successes / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    half = (z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n))) / denom
    return max(center - half, 0.0), min(center + half, 1.0)


def run_champion_bench(
    candidate_path: str,
    champion_path: str,
    *,
    n_seeds: int = 200,
    swap_first_player: bool = True,
    threshold: float = 0.70,
    output_dir: str | Path | None = None,
    device: str | None = None,
) -> ChampionBenchResult:
    """Run the full champion-bench and return a ``ChampionBenchResult``.

    If ``output_dir`` is given, a ``champion_bench.json`` file is also
    written there (atomic via tmp+rename) for the harness to consume.
    """
    candidate_trainer = CatanPPO.load(candidate_path)
    champion_trainer = CatanPPO.load(champion_path)
    candidate_policy = candidate_trainer.policy.eval()
    champion_policy = champion_trainer.policy.eval()
    eval_device = device or candidate_trainer.device

    em = EvaluationManager(opponent_type="policy", max_turns=500)
    seeds = standard_eval_seeds(0, n_seeds)
    h2h = em.evaluate_h2h(
        candidate_policy,
        champion_policy,
        seeds,
        device=eval_device,
        swap_first_player=swap_first_player,
    )

    n_games = int(h2h["n_games"])
    candidate_wins = int(round(h2h["win_rate_a"] * n_games))
    win_rate = h2h["win_rate_a"]
    ci_low, ci_high = wilson_interval(candidate_wins, n_games)

    result = ChampionBenchResult(
        candidate_path=str(candidate_path),
        champion_path=str(champion_path),
        n_seeds=int(n_seeds),
        swap_first_player=bool(swap_first_player),
        n_games=n_games,
        win_rate=win_rate,
        win_rate_ci_low=ci_low,
        win_rate_ci_high=ci_high,
        draw_rate=h2h["draw_rate"],
        avg_candidate_vp=h2h["avg_a_vp"],
        avg_champion_vp=h2h["avg_b_vp"],
        avg_length=h2h["avg_length"],
        threshold=threshold,
        passed=ci_low >= threshold,
    )

    if output_dir is not None:
        path = Path(output_dir) / "champion_bench.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        with tmp.open("w") as f:
            json.dump(asdict(result), f, indent=2)
        tmp.replace(path)

    # Free trainer state to avoid holding onto two copies of the optimizer.
    del candidate_trainer
    del champion_trainer
    if torch.cuda.is_available():  # pragma: no cover
        torch.cuda.empty_cache()

    return result
