"""Apply the Part A trend-analysis verdict to a TensorBoard run.

Reads the ``eval/win_rate`` (and ``eval/avg_vp``) scalars from a
``runs/train/`` events file, applies a rolling 10-eval window, runs
Mann-Kendall + linear-fit, and emits a mechanical
``satisfactory / unsatisfactory / inconclusive`` verdict per the protocol
in ``docs/plans/setup_pretrain_plan.md``.

Idempotent — reads-only on the events file, writes a fresh JSON each
invocation. Safe to run via cron alongside a live training process.

Usage:
    # Full verdict against the live run (uses largest events file by size,
    # skipping <1KB placeholders left by transient eval subprocesses):
    python scripts/analyze_trend.py

    # Interim verdict — emit "interim" tag if the latest TB step is below
    # the final-verdict gate. Use this in a cron so it doesn't prematurely
    # trigger Part B before enough post-bundle data is collected.
    python scripts/analyze_trend.py --final-at-step 18700000

    # Pin to a specific events file (e.g. when comparing runs):
    python scripts/analyze_trend.py --events-file runs/train/events.out.tfevents.X.Y.Z

Exit code: 0 = satisfactory, 1 = unsatisfactory, 2 = inconclusive,
3 = error (no TB data found / bad path). Shell scripts can gate on it.

Cron suggestion (every 6 hours from now to step 18.7M):
    0 */6 * * * cd /path/to/catan_rl && \\
      python scripts/analyze_trend.py --final-at-step 18700000 \\
      --out runs/trend_analysis/$(date +\\%Y\\%m\\%d_\\%H\\%M).json
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path

_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if os.path.isdir(_SRC) and _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from catan_rl.eval.trend_analysis import (
    linear_fit,
    read_tb_scalar,
    rolling_mean,
    verdict,
)


def find_latest_training_events(logdir: str) -> str | None:
    """Pick the largest events file under ``logdir`` (ignores <1KB placeholders
    written by transient eval subprocesses that import EvaluationManager
    without writing scalars). Falls back to mtime for tie-breaking.
    """
    candidates = glob.glob(os.path.join(logdir, "**", "events.out.tfevents.*"), recursive=True)
    candidates += glob.glob(os.path.join(logdir, "events.out.tfevents.*"))
    candidates = [c for c in candidates if os.path.getsize(c) > 1024]
    if not candidates:
        return None
    # Largest file = most scalars logged = the real training run.
    return max(candidates, key=os.path.getsize)


def maybe_make_plot(
    series: list[tuple[int, float]],
    rolling: list[tuple[int, float]],
    fit: dict,
    out_path: str,
) -> bool:
    """Write a PNG plot if matplotlib is available; return success."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    if not series:
        return False
    steps_raw = [s for s, _ in series]
    vals_raw = [v for _, v in series]
    steps_roll = [s for s, _ in rolling]
    vals_roll = [v for _, v in rolling]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter(steps_raw, vals_raw, color="#1f77b4", alpha=0.6, s=24, label="raw eval WR")
    ax.plot(steps_roll, vals_roll, color="#ff7f0e", lw=2, label="rolling 10-eval mean")
    if fit.get("n", 0) >= 3:
        xs = sorted(steps_raw)
        ys = [fit["slope"] * x + fit["intercept"] for x in xs]
        ax.plot(
            xs, ys, color="#2ca02c", ls="--", lw=1, label=f"linear fit (slope={fit['slope']:.2e})"
        )
    ax.axhline(0.30, color="gray", ls=":", lw=1, label="prior plateau ~0.30")
    ax.axhline(0.95, color="red", ls=":", lw=1, label="superhuman gate")
    ax.set_xlabel("step")
    ax.set_ylabel("eval WR vs heuristic")
    ax.set_title("Eval WR trend (Part A verdict harness)")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Part A trend-analysis verdict for plateau-breaking bundle.",
        epilog=__doc__.split("Cron suggestion:")[1] if "Cron suggestion:" in __doc__ else None,
    )
    parser.add_argument("--logdir", default="runs/train/")
    parser.add_argument(
        "--events-file",
        default=None,
        help="Specific events file; overrides --logdir auto-pick.",
    )
    parser.add_argument(
        "--tag",
        default="eval/win_rate",
        help="TB scalar tag for the eval metric (default: eval/win_rate).",
    )
    parser.add_argument("--window", type=int, default=10, help="Rolling-mean window (default: 10).")
    parser.add_argument(
        "--final-at-step",
        type=int,
        default=None,
        help=(
            "If set, emit 'final' verdict only when the latest TB step is "
            ">= this value. Below it, the verdict is tagged 'interim' but "
            "still computed and written."
        ),
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output JSON path; default: runs/trend_analysis/<timestamp>.json",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Also write a .png plot next to the JSON.",
    )
    # Verdict thresholds — exposed so the user can override without an edit.
    parser.add_argument("--prior-plateau-lo", type=float, default=0.27)
    parser.add_argument("--prior-plateau-hi", type=float, default=0.36)
    parser.add_argument("--satisfactory-mean-threshold", type=float, default=0.40)
    parser.add_argument("--lift-threshold", type=float, default=0.05)
    parser.add_argument("--mk-alpha", type=float, default=0.10)
    parser.add_argument("--delta-band", type=float, default=0.02)
    parser.add_argument("--min-evals", type=int, default=10)
    args = parser.parse_args()

    events_path = args.events_file or find_latest_training_events(args.logdir)
    if events_path is None or not os.path.exists(events_path):
        print(
            f"[analyze_trend] no usable events file under {args.logdir!r}; "
            f"check that training has written TB scalars",
            file=sys.stderr,
        )
        return 3

    series = read_tb_scalar(events_path, args.tag)
    if not series:
        print(
            f"[analyze_trend] tag {args.tag!r} not present in {events_path}; "
            f"check that the trainer has logged any eval scalars yet",
            file=sys.stderr,
        )
        return 3

    verdict_result = verdict(
        series,
        prior_plateau_lo=args.prior_plateau_lo,
        prior_plateau_hi=args.prior_plateau_hi,
        satisfactory_mean_threshold=args.satisfactory_mean_threshold,
        lift_threshold=args.lift_threshold,
        mk_alpha=args.mk_alpha,
        delta_band=args.delta_band,
        min_evals=args.min_evals,
    )

    # Gate "final" vs "interim".
    latest_step = series[-1][0]
    is_interim = args.final_at_step is not None and latest_step < args.final_at_step
    verdict_result["tag"] = "interim" if is_interim else "final"
    verdict_result["latest_step"] = latest_step
    verdict_result["events_file"] = events_path

    # Side stats: rolling mean + linear fit on the full series (for plots).
    rolling = rolling_mean(series, args.window)
    fit = linear_fit([s for s, _ in series], [v for _, v in series])

    # Output path.
    if args.out is None:
        ts = time.strftime("%Y%m%d_%H%M%S")
        args.out = os.path.join("runs", "trend_analysis", f"verdict_{ts}.json")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "verdict": verdict_result["verdict"],
        "tag": verdict_result["tag"],
        "reasoning": verdict_result["reasoning"],
        "latest_step": latest_step,
        "n_evals": len(series),
        "events_file": events_path,
        "final_at_step": args.final_at_step,
        "stats": verdict_result["stats"],
        "linear_fit": {k: v for k, v in fit.items()},
        "series": [{"step": s, "wr": v} for s, v in series],
    }
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)

    plot_path = None
    if args.plot:
        plot_path = args.out.replace(".json", ".png")
        if not maybe_make_plot(series, rolling, fit, plot_path):
            plot_path = None

    # Human-readable summary on stdout.
    v = verdict_result["verdict"].upper()
    tag = verdict_result["tag"].upper()
    print(f"[analyze_trend] {tag} verdict: {v}")
    print(f"  reasoning: {verdict_result['reasoning']}")
    print(f"  latest_step: {latest_step:,}  n_evals: {len(series)}")
    s = verdict_result["stats"]
    if "mean10" in s:
        print(
            f"  mean10={s['mean10']:.3f}  "
            f"last5={s['mean_last5']:.3f}  prior5={s['mean_prior5']:.3f}  "
            f"Δ={s['delta_last5_minus_prior5']:.3f}  "
            f"MK τ={s['mk_tau']:.2f} p={s['mk_p_value']:.3f}"
        )
    print(f"  → wrote {args.out}")
    if plot_path:
        print(f"  → wrote {plot_path}")

    # Exit codes: only emit non-zero on a FINAL verdict, so cron-driven
    # interim runs don't trip downstream Part B triggers prematurely.
    if is_interim:
        return 0
    return {"satisfactory": 0, "unsatisfactory": 1, "inconclusive": 2}[verdict_result["verdict"]]


if __name__ == "__main__":
    raise SystemExit(main())
