"""Print a snapshot of the latest TensorBoard scalars + trend deltas.

Usage:
    python scripts/monitor_training.py
    python scripts/monitor_training.py --logdir runs/train --top-k 8

Designed to be polled (Monitor + grep) by an automated supervisor:
emits a single-line summary plus alert tokens (``ALERT:<reason>``) when
the run looks unhealthy. Exit code 0 always; downstream parses the
output for content rather than exit status.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import time

# Allow ``python scripts/monitor_training.py`` without ``pip install -e .``.
_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if os.path.isdir(_SRC) and _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# Canonical key scalars to surface, ordered roughly by training importance.
KEY_SCALARS = [
    "train/fps",
    "rollout/mean_win_rate",
    "rollout/mean_reward",
    "train/entropy",
    "train/value_loss",
    "train/policy_loss",
    "train/explained_variance",
    "train/approx_kl",
    "train/clip_fraction",
    "train/learning_rate",
    "train/entropy_coef",
    "train/entropy_collapse_flag",
    "train/belief_loss",
    "train/opp_action_loss",
    "eval/win_rate",
    "eval/eval_opponent",
    "eval/trueskill_main_mu",
    "eval/trueskill_main_sigma",
]


def latest_events_file(logdir: str) -> str | None:
    """Pick the most recently-modified TF events file under ``logdir``."""
    candidates = glob.glob(os.path.join(logdir, "**", "events.out.tfevents.*"), recursive=True)
    candidates += glob.glob(os.path.join(logdir, "events.out.tfevents.*"))
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)


def load_scalars(events_path: str) -> dict[str, list[tuple[int, float]]]:
    """Return ``{tag: [(step, value), ...]}`` for every scalar in the file."""
    acc = EventAccumulator(events_path, size_guidance={"scalars": 0})
    acc.Reload()
    out: dict[str, list[tuple[int, float]]] = {}
    for tag in acc.Tags().get("scalars", []):
        events = acc.Scalars(tag)
        out[tag] = [(int(e.step), float(e.value)) for e in events]
    return out


def fmt(v: float) -> str:
    """Compact format that handles large/small magnitudes legibly."""
    if abs(v) >= 1000:
        return f"{v:,.0f}"
    if abs(v) >= 1:
        return f"{v:.3f}"
    if abs(v) >= 1e-3:
        return f"{v:.4f}"
    return f"{v:.2e}"


def trend_arrow(history: list[tuple[int, float]]) -> str:
    """Return ↑ / ↓ / → based on the last 5 vs prior 5 mean."""
    if len(history) < 4:
        return "·"
    recent = [v for _, v in history[-5:]]
    prior = [v for _, v in history[-10:-5]] or recent
    delta = (sum(recent) / len(recent)) - (sum(prior) / len(prior))
    if abs(delta) < 1e-6:
        return "→"
    return "↑" if delta > 0 else "↓"


def emit_alerts(scalars: dict[str, list[tuple[int, float]]], step: int) -> list[str]:
    """Detect unhealthy patterns and return ALERT tokens for the supervisor."""
    alerts: list[str] = []

    def latest(tag: str) -> float | None:
        h = scalars.get(tag)
        return h[-1][1] if h else None

    fps = latest("train/fps")
    if fps is not None and fps < 5.0 and step > 5000:
        alerts.append(f"ALERT:LOW_FPS={fps:.1f}")

    ev = latest("train/explained_variance")
    if ev is not None and ev < -0.5 and step > 50_000:
        # Negative EV means the value head is worse than predicting the mean.
        # Tolerable in early training; alarming after 50k steps.
        alerts.append(f"ALERT:NEG_EV={ev:.2f}")

    ent = latest("train/entropy")
    if ent is not None and ent < 0.05 and step > 100_000:
        alerts.append(f"ALERT:LOW_ENTROPY={ent:.4f}")

    collapse = latest("train/entropy_collapse_flag")
    if collapse is not None and collapse > 0.5:
        alerts.append("ALERT:HEAD_COLLAPSE")

    kl = latest("train/approx_kl")
    if kl is not None and kl > 0.10:
        alerts.append(f"ALERT:HIGH_KL={kl:.3f}")

    return alerts


def main() -> int:
    parser = argparse.ArgumentParser(description="TensorBoard scalar snapshot for a training run")
    parser.add_argument("--logdir", default="runs/train", help="TB log root")
    parser.add_argument("--top-k", type=int, default=12, help="How many scalars to print")
    args = parser.parse_args()

    events_path = latest_events_file(args.logdir)
    if events_path is None:
        print(f"[monitor] no events files under {args.logdir}")
        return 0
    age = time.time() - os.path.getmtime(events_path)

    try:
        scalars = load_scalars(events_path)
    except Exception as e:
        print(f"[monitor] read failed for {events_path}: {e}")
        return 0

    if not scalars:
        print(f"[monitor] events file present but no scalars yet (age={age:.0f}s)")
        return 0

    # Find the latest step number across any scalar.
    latest_step = max((h[-1][0] for h in scalars.values() if h), default=0)
    n_events = sum(len(h) for h in scalars.values())

    line = (
        f"[monitor] step={latest_step:,} events={n_events} "
        f"file_age={age:.0f}s file={os.path.basename(events_path)}"
    )
    print(line)

    # Print key scalars in order, with trend arrows.
    for tag in KEY_SCALARS[: args.top_k]:
        history = scalars.get(tag, [])
        if not history:
            continue
        step, val = history[-1]
        arrow = trend_arrow(history)
        print(f"  {arrow} {tag:<32} step={step:>10,}  {fmt(val)}")

    for alert in emit_alerts(scalars, latest_step):
        print(alert)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
