"""Eval harness CLI (Phase 0).

Modes:
  --mode rules-invariant    Run the runtime 1v1 rules-invariant test.
  --mode champion-bench     H2H over deterministic seeds vs frozen champion.
  --mode exploitability     Train a fresh adversary and report its WR.
  --mode league-rating      Round-robin a glob of checkpoints; emit ratings.
  --mode all                Run everything (rules → champion → exploitability).

Output JSON is written to ``<output-dir>/<mode>.json`` (default
``runs/eval_harness/<run-name>/``). Exit code is 0 if every requested mode
passes its threshold, else 1 — so the harness can gate CI / promotion.
"""

from __future__ import annotations

import argparse
import datetime
import glob
import json
import os
import sys
import warnings
from pathlib import Path

# Allow `python scripts/eval_harness.py` without `pip install -e .` by adding src/ to path.
_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if os.path.isdir(_SRC) and _SRC not in sys.path:
    sys.path.insert(0, _SRC)

warnings.filterwarnings("ignore", message="enable_nested_tensor.*norm_first")


def _default_run_name() -> str:
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def _ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _write_json(path: Path, obj: dict) -> None:
    tmp = path.with_suffix(".json.tmp")
    with tmp.open("w") as f:
        json.dump(obj, f, indent=2, default=str)
    tmp.replace(path)


# ── Mode handlers ────────────────────────────────────────────────────────────


def cmd_rules_invariant(args: argparse.Namespace, output_dir: Path) -> bool:
    from catan_rl.eval.rules_invariants import run as run_invariants

    fails = run_invariants(include_hand_tracker_drift=args.include_hand_tracker_drift)
    payload = {
        "mode": "rules-invariant",
        "n_failures": len(fails),
        "failures": [{"name": f.name, "rule": f.rule, "detail": f.detail} for f in fails],
        "include_hand_tracker_drift": bool(args.include_hand_tracker_drift),
    }
    _write_json(output_dir / "rules_invariant.json", payload)
    if fails:
        print(f"[rules-invariant] FAIL ({len(fails)} failures):")
        for f in fails:
            print(f"  - {f}")
        return False
    print("[rules-invariant] PASS")
    return True


def cmd_champion_bench(args: argparse.Namespace, output_dir: Path) -> bool:
    from catan_rl.eval.champion_bench import run_champion_bench

    if not args.candidate:
        print("[champion-bench] ERROR: --candidate is required", file=sys.stderr)
        return False

    result = run_champion_bench(
        candidate_path=args.candidate,
        champion_path=args.champion,
        n_seeds=args.n_seeds,
        swap_first_player=not args.no_swap,
        threshold=args.threshold,
        output_dir=output_dir,
    )
    print(
        f"[champion-bench] candidate vs champion: WR={result.win_rate:.3f} "
        f"95% CI=[{result.win_rate_ci_low:.3f}, {result.win_rate_ci_high:.3f}] "
        f"({result.n_games} games, threshold={result.threshold:.2f}) "
        f"{'PASS' if result.passed else 'FAIL'}"
    )
    return result.passed


def cmd_exploitability(args: argparse.Namespace, output_dir: Path) -> bool:
    if args.smoke:
        from catan_rl.eval.exploitability import run_exploitability_smoke

        result = run_exploitability_smoke(
            args.champion,
            output_dir=output_dir,
        )
    else:
        from catan_rl.eval.exploitability import run_exploitability

        result = run_exploitability(
            args.champion,
            n_steps=args.adversary_steps,
            eval_games=args.adversary_eval_games,
            threshold=args.exploitability_threshold,
            workspace=args.exploitability_workspace,
            cleanup_workspace=args.cleanup_workspace,
            output_dir=output_dir,
            n_envs=args.n_envs,
        )
    print(
        f"[exploitability] adversary WR vs champion = {result.adversary_win_rate:.3f} "
        f"(threshold={result.threshold:.2f}, smoke={result.smoke}) "
        f"{'PASS' if result.passed else 'FAIL'}"
    )
    return result.passed


def cmd_league_rating(args: argparse.Namespace, output_dir: Path) -> bool:
    """Round-robin a glob of checkpoints with H2H, build a ratings table.

    Each pair plays ``--league-rating-games`` games (with first-player swap),
    and the binary outcome is fed to the rating system.
    """
    from catan_rl.algorithms.ppo.trainer import CatanPPO
    from catan_rl.eval.evaluation_manager import (
        EvaluationManager,
        standard_eval_seeds,
    )
    from catan_rl.selfplay.ratings import RatingTable

    paths = sorted({p for pattern in args.policies for p in glob.glob(pattern)})
    if not paths:
        print("[league-rating] ERROR: no checkpoints matched", file=sys.stderr)
        return False
    print(f"[league-rating] round-robin over {len(paths)} checkpoints")

    table = RatingTable()
    em = EvaluationManager(opponent_type="policy", max_turns=500)
    seeds = standard_eval_seeds(0, args.league_rating_games)

    # Load all policies once; expensive but bounded.
    trainers = [CatanPPO.load(p) for p in paths]
    for t in trainers:
        t.policy.eval()

    for i, (path_a, ta) in enumerate(zip(paths, trainers, strict=True)):
        for j in range(i + 1, len(paths)):
            path_b = paths[j]
            tb = trainers[j]
            h2h = em.evaluate_h2h(
                ta.policy,
                tb.policy,
                seeds,
                device=ta.device,
                swap_first_player=True,
            )
            n = int(h2h["n_games"])
            a_wins = int(round(h2h["win_rate_a"] * n))
            for g in range(n):
                # Distribute wins evenly across the games as binary outcomes
                # for the rating update — we don't track per-game IDs here.
                won = g < a_wins
                table.record_match(path_a, path_b, a_won=won)
            print(
                f"  {Path(path_a).name} vs {Path(path_b).name}: "
                f"WR_a={h2h['win_rate_a']:.2f} draws={h2h['draw_rate']:.2f}"
            )

    payload = {"backend": table.system.backend, "ratings": table.to_dict()}
    _write_json(output_dir / "league_rating.json", payload)
    print(f"[league-rating] wrote {output_dir / 'league_rating.json'}")
    print(f"  top: {[(Path(k).name, round(v.conservative, 2)) for k, v in table.top_k(5)]}")
    return True


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> int:
    p = argparse.ArgumentParser(description="Catan eval harness (Phase 0)")
    p.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["rules-invariant", "champion-bench", "exploitability", "league-rating", "all"],
    )
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--run-name", type=str, default=None)

    # rules-invariant
    p.add_argument("--include-hand-tracker-drift", action="store_true")

    # champion-bench
    p.add_argument("--candidate", type=str, default=None)
    p.add_argument(
        "--champion",
        type=str,
        default="checkpoints/train/checkpoint_07390040.pt",
    )
    p.add_argument("--n-seeds", type=int, default=200)
    p.add_argument("--no-swap", action="store_true")
    p.add_argument("--threshold", type=float, default=0.70)

    # exploitability
    p.add_argument("--adversary-steps", type=int, default=5_000_000)
    p.add_argument("--adversary-eval-games", type=int, default=200)
    p.add_argument("--exploitability-threshold", type=float, default=0.05)
    p.add_argument("--exploitability-workspace", type=str, default=None)
    p.add_argument("--cleanup-workspace", action="store_true")
    p.add_argument(
        "--smoke",
        action="store_true",
        help="Use smoke-runtime variant of exploitability (fast wiring check)",
    )
    p.add_argument("--n-envs", type=int, default=4)

    # league-rating
    p.add_argument("--policies", nargs="+", default=[])
    p.add_argument("--league-rating-games", type=int, default=20)

    args = p.parse_args()

    run_name = args.run_name or _default_run_name()
    output_dir = _ensure_dir(args.output_dir or f"runs/eval_harness/{run_name}")

    summary = {
        "run_name": run_name,
        "output_dir": str(output_dir),
        "mode": args.mode,
        "modes_run": [],
        "modes_passed": [],
        "modes_failed": [],
    }

    modes = (
        ["rules-invariant", "champion-bench", "exploitability"]
        if args.mode == "all"
        else [args.mode]
    )
    overall_pass = True
    for mode in modes:
        summary["modes_run"].append(mode)
        if mode == "rules-invariant":
            ok = cmd_rules_invariant(args, output_dir)
        elif mode == "champion-bench":
            ok = cmd_champion_bench(args, output_dir)
        elif mode == "exploitability":
            ok = cmd_exploitability(args, output_dir)
        elif mode == "league-rating":
            ok = cmd_league_rating(args, output_dir)
        else:  # pragma: no cover - argparse-guarded
            print(f"unknown mode: {mode}", file=sys.stderr)
            ok = False
        (summary["modes_passed"] if ok else summary["modes_failed"]).append(mode)
        overall_pass = overall_pass and ok

    _write_json(output_dir / "summary.json", summary)
    print(
        f"[harness] mode={args.mode} overall {'PASS' if overall_pass else 'FAIL'} "
        f"(passed={summary['modes_passed']} failed={summary['modes_failed']})"
    )
    return 0 if overall_pass else 1


if __name__ == "__main__":
    sys.exit(main())
