"""Plan §C.0 plateau-source diagnostic (modified — heuristic baseline).

The plan as written calls for running this against a trained champion to
discriminate three failure modes:
  (a) setup-quality bottleneck → setup-phase tweaks (Phase A/B) carry the load
  (b) value-learning ceiling   → ISMCTS (Phase C.1)
  (c) policy bottleneck        → belief-head reinforcement (Phase C.2)

No v2 champion exists yet (see ``project_setup_strength_calibration.md``).
This script runs the modified version on **heuristic-vs-heuristic** games
that captures only part (a): does setup quality (as scored by the analytic
yield function) discriminate winners from losers?

What we measure:

* For each completed game, compute each player's setup-quality score as
  ``analytic_yield(settle_1) + analytic_yield(settle_2)`` using Charlesworth's
  resource weights.
* Compute per-game ``setup_advantage = quality(winner) − quality(loser)``.
* Build the ROC: across all games, can a threshold on
  ``quality(p1) − quality(p2)`` predict the winner?

Output: a JSON summary with histograms, AUC, sample size, and a verdict
keyed against the plan's thresholds (AUC > 0.65 → strong setup signal;
AUC < 0.55 → setup doesn't move the needle).

Usage::

    PYTHONPATH=src python scripts/diagnose_plateau_source.py --n-games 1000

Writes ``runs/plateau_diagnosis/<timestamp>/summary.json``.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from catan_rl.bc.dataset import play_game
from catan_rl.setup_phase.analytic_value import vertex_yield
from catan_rl.setup_phase.resource_weights import CHARLESWORTH_V0

REPO_ROOT = Path(__file__).resolve().parent.parent


def _setup_picks_per_seat(record) -> dict[int, list[int]]:
    """Extract each seat's setup-phase settlement vertex indices.

    Returns ``{seat: [v1, v2]}`` for seats 0, 1. Each seat picks two
    settlements during the 1v1 snake draft.
    """
    picks: dict[int, list[int]] = {0: [], 1: []}
    for dec in record.decisions:
        if dec.phase != "setup":
            continue
        # Action layout: [type, corner, edge, tile, res1, res2].
        # Type 0 = BUILD_SETTLEMENT.
        if int(dec.action[0]) == 0:
            picks[dec.player_seat].append(int(dec.action[1]))
    return picks


def _game_setup_quality(record, board) -> tuple[float, float] | None:
    """Return ``(quality_p1, quality_p2)`` from analytic yields summed
    over each seat's two setup settlements, or None if the picks aren't
    well-formed."""
    picks = _setup_picks_per_seat(record)
    if len(picks[0]) < 2 or len(picks[1]) < 2:
        return None
    q1 = sum(vertex_yield(board, v, CHARLESWORTH_V0) for v in picks[0][:2])
    q2 = sum(vertex_yield(board, v, CHARLESWORTH_V0) for v in picks[1][:2])
    return q1, q2


def _compute_auc(scores: list[float], labels: list[int]) -> float:
    """Compute ROC AUC: scores predict labels.

    ``labels[i] = 1`` if p1 won, 0 otherwise; scores are
    ``quality(p1) − quality(p2)``. Uses the standard rank-based estimator
    so we don't pull in scikit-learn for one number.
    """
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    arr = np.asarray(scores)
    lab = np.asarray(labels)
    order = np.argsort(arr)
    ranks = np.empty(len(arr), dtype=np.float64)
    ranks[order] = np.arange(1, len(arr) + 1)
    sum_ranks_pos = float(ranks[lab == 1].sum())
    return (sum_ranks_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n-games", type=int, default=1000)
    p.add_argument("--seed-start", type=int, default=0)
    p.add_argument("--max-turns", type=int, default=400)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--perturbation", type=str, default="canonical")
    args = p.parse_args()

    out_dir = args.out_dir or REPO_ROOT / "runs" / "plateau_diagnosis" / time.strftime(
        "%Y%m%d_%H%M%S"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"writing to {out_dir}")

    scores: list[float] = []
    labels: list[int] = []
    n_truncated = 0
    n_setup_malformed = 0
    advantages_winner: list[float] = []
    advantages_loser: list[float] = []
    winner_q_when_p1_wins: list[float] = []
    loser_q_when_p1_wins: list[float] = []
    game_lengths: list[int] = []

    t0 = time.time()
    for i in range(args.n_games):
        seed = args.seed_start + i
        record = play_game(
            game_id=i,
            seed=seed,
            perturbation=args.perturbation,
            max_turns=args.max_turns,
        )
        if record.truncated:
            n_truncated += 1
            continue
        # Reconstruct board for vertex yields. ``play_game`` builds the
        # board with seed=i — same seed reproduces the layout.
        np.random.seed(seed)
        import random

        random.seed(seed)
        from catan_rl.engine.board import catanBoard

        board = catanBoard()
        qualities = _game_setup_quality(record, board)
        if qualities is None:
            n_setup_malformed += 1
            continue
        q1, q2 = qualities
        scores.append(q1 - q2)
        labels.append(int(record.p1_won))
        game_lengths.append(record.total_turns)
        if record.p1_won:
            advantages_winner.append(q1)
            advantages_loser.append(q2)
            winner_q_when_p1_wins.append(q1)
            loser_q_when_p1_wins.append(q2)
        else:
            advantages_winner.append(q2)
            advantages_loser.append(q1)

        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            print(
                f"  {i + 1}/{args.n_games} games | "
                f"{rate:.1f} games/s | usable={len(scores)} | "
                f"truncated={n_truncated} | malformed_setup={n_setup_malformed}"
            )

    if not scores:
        print("FAIL: no usable games. Verdict: NONE.")
        return

    auc = _compute_auc(scores, labels)
    mean_winner_q = float(np.mean(advantages_winner)) if advantages_winner else 0.0
    mean_loser_q = float(np.mean(advantages_loser)) if advantages_loser else 0.0
    mean_diff = mean_winner_q - mean_loser_q
    p1_win_rate = float(np.mean(labels))

    # Verdict thresholds per the plan:
    #   AUC > 0.65 → strong setup-quality discrimination
    #   AUC < 0.55 → setup quality does NOT discriminate winners
    #   in [0.55, 0.65] → weak signal, inconclusive
    if auc > 0.65:
        verdict = "STRONG_SETUP_SIGNAL"
        interpretation = (
            "Setup quality (analytic yield) discriminates winners well — "
            "Phase A/B improvements are well-justified. The plateau is "
            "(at least in part) a setup-quality bottleneck."
        )
    elif auc < 0.55:
        verdict = "WEAK_SETUP_SIGNAL"
        interpretation = (
            "Setup quality barely predicts winners — investing further "
            "in setup-phase tweaks has limited expected lift. Consider "
            "pivoting to value-learning (ISMCTS) or policy (belief) work."
        )
    else:
        verdict = "INCONCLUSIVE"
        interpretation = (
            "Setup quality has a weak/ambiguous signal. The diagnostic "
            "is inconclusive on heuristic-vs-heuristic; needs a stronger "
            "policy to re-run."
        )

    summary = {
        "config": {
            "n_games_requested": args.n_games,
            "seed_start": args.seed_start,
            "max_turns": args.max_turns,
            "perturbation": args.perturbation,
            "opponent": "heuristic_vs_heuristic",
            "resource_weights": "charlesworth_v0",
            "diagnostic_part": "a_only (no champion model for part b)",
        },
        "stats": {
            "n_usable": len(scores),
            "n_truncated": n_truncated,
            "n_setup_malformed": n_setup_malformed,
            "p1_win_rate": p1_win_rate,
            "mean_winner_setup_quality": mean_winner_q,
            "mean_loser_setup_quality": mean_loser_q,
            "mean_winner_minus_loser": mean_diff,
            "mean_game_length_turns": float(np.mean(game_lengths)) if game_lengths else 0.0,
        },
        "auc": auc,
        "verdict": verdict,
        "interpretation": interpretation,
        "p1_win_breakdown_by_setup_advantage_quintile": _quintile_winrate(scores, labels),
    }

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print()
    print(f"=== RESULTS ({len(scores)} usable games) ===")
    print(f"AUC(setup_advantage → p1 wins):  {auc:.4f}")
    print(f"mean winner setup quality:       {mean_winner_q:.3f}")
    print(f"mean loser setup quality:        {mean_loser_q:.3f}")
    print(f"mean (winner - loser):           {mean_diff:.3f}")
    print(f"p1 win rate:                     {p1_win_rate:.3f}")
    print()
    print(f"VERDICT: {verdict}")
    print(interpretation)
    print()
    print(f"summary.json → {out_dir / 'summary.json'}")


def _quintile_winrate(scores: list[float], labels: list[int]) -> list[dict]:
    """Bucket games by quintile of setup advantage and report p1 win
    rate per bucket. Gives a sanity-check on the AUC: if the bucket WR
    monotonically tracks the bucket score, the discrimination is real."""
    if len(scores) < 5:
        return []
    arr = np.asarray(scores)
    lab = np.asarray(labels, dtype=np.float64)
    order = np.argsort(arr)
    sorted_lab = lab[order]
    sorted_score = arr[order]
    out = []
    n = len(sorted_lab)
    for k in range(5):
        lo, hi = k * n // 5, (k + 1) * n // 5
        chunk_lab = sorted_lab[lo:hi]
        chunk_score = sorted_score[lo:hi]
        if not len(chunk_lab):
            continue
        out.append(
            {
                "quintile": k,
                "n": len(chunk_lab),
                "mean_advantage": float(chunk_score.mean()),
                "p1_win_rate": float(chunk_lab.mean()),
            }
        )
    return out


if __name__ == "__main__":
    main()
