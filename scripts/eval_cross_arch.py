"""Cross-architecture head-to-head: new pointer-arch checkpoint vs v11-era policy.

The pointer-arch fork changed both the network shape and the obs schema, so v11
can no longer be loaded by the current ``CatanPolicy`` — the in-process
``evaluate_policy_vs_policy`` cannot pit new-arch against v11. This tool bridges
the gap IN ONE PROCESS: the engine + geometry are byte-identical across the fork,
so both policies play the same live game, each reading its own obs schema (the
new arch natively; v11 via the vendored legacy encoder in
``catan_rl.eval.legacy_arch``). See ``catan_rl.eval.cross_arch`` for the design.

This is the yardstick for the pointer-arch fork: it measures the ratified accept
gate (clause a: new-arch h2h vs v11_cand Wilson-LB > 0.50 at n=600) and is the
only non-saturated signal that self-play is surpassing v11.

Usage:
  # Primary: measure a new-arch candidate vs the frozen v11 champion.
  python scripts/eval_cross_arch.py \
      --new runs/anchors/<new_cand>.pt \
      --old runs/anchors/v11_cand_u724.pt --n-games 100 --seed 0   # gate: --n-games 600

  # Equivalence self-check (new-vs-new through this bridge == in-process path):
  python scripts/eval_cross_arch.py --new <ckpt> --old <ckpt> --old-arch new --n-games 8

  # Fast smoke that it runs end-to-end:
  python scripts/eval_cross_arch.py --new <ckpt> --old runs/anchors/v11_cand_u724.pt --smoke
"""

from __future__ import annotations

import argparse
import sys
import time

from catan_rl.eval.cross_arch import cross_arch_h2h
from catan_rl.eval.engine_parity import EngineParityError, assert_engine_parity


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--new", required=True, help="new pointer-arch checkpoint (agent seat)")
    p.add_argument(
        "--old",
        required=True,
        help="old checkpoint (opponent seat); v11-era for --old-arch legacy",
    )
    p.add_argument(
        "--old-arch",
        choices=("legacy", "new"),
        default="legacy",
        help="'legacy' = v11-era vendored arch (default); 'new' = current arch (self-check)",
    )
    p.add_argument(
        "--n-games",
        type=int,
        default=100,
        help="total games (seat-symmetrized; default 100; gate uses 600)",
    )
    p.add_argument(
        "--seed", type=int, default=0, help="master seed (default 0; results are seed-reproducible)"
    )
    p.add_argument(
        "--device", default="cpu", help="cpu / mps / cuda / auto (default cpu — eval is CPU-pinned)"
    )
    p.add_argument(
        "--max-turns",
        type=int,
        default=400,
        help="per-game agent-turn truncation cap (default 400)",
    )
    p.add_argument(
        "--skip-engine-parity-check",
        action="store_true",
        help="bypass the engine-vs-pre-fork parity guard (result is then untrusted)",
    )
    p.add_argument(
        "--smoke", action="store_true", help="override to n-games=4 for a fast end-to-end check"
    )
    args = p.parse_args()

    n_games = 4 if args.smoke else args.n_games
    strict_parity = not args.skip_engine_parity_check

    print(
        f"cross-arch h2h: NEW={args.new}\n"
        f"                OLD={args.old} (arch={args.old_arch})\n"
        f"                n_games={n_games} seed={args.seed} "
        f"device={args.device} max_turns={args.max_turns}",
        file=sys.stderr,
    )
    # Engine-parity guard (surfaces the stamp; fails fast with a clean message).
    try:
        stamp = assert_engine_parity(strict=strict_parity)
    except EngineParityError as exc:
        print(f"\nENGINE-PARITY GUARD FAILED — refusing to run:\n  {exc}", file=sys.stderr)
        return 2
    print(
        f"  engine-parity      : engine={stamp['engine']} board_geometry={stamp['board_geometry']}",
        file=sys.stderr,
    )

    t0 = time.time()
    result = cross_arch_h2h(
        new_ckpt=args.new,
        old_ckpt=args.old,
        old_arch=args.old_arch,
        n_games=n_games,
        seed=args.seed,
        device=args.device,
        max_turns=args.max_turns,
        strict_engine_parity=strict_parity,
    )
    dt = time.time() - t0

    ci = result.ci
    violations = result.rules_violations
    gate_pass = ci.clears_zero_against(0.50)
    print("\n==================== cross-arch h2h result ====================")
    print(f"  NEW arch WR vs OLD : {result.wr:.4f}   ({result.wins}/{result.n} games)")
    print(f"  Wilson 95% CI      : [{ci.lower:.4f}, {ci.upper:.4f}]   (LB={ci.lower:.4f})")
    print(
        f"  per-seat WR        : seat0={result.wr_seat0:.4f} (n={result.n_seat0})  "
        f"seat1={result.wr_seat1:.4f} (n={result.n_seat1})"
    )
    print(f"  truncated games    : {result.n_truncated}")
    print(
        f"  rules violations   : {len(violations)}" + (f"  {violations[:5]}" if violations else "")
    )
    print(f"  wall time          : {dt:.1f}s")
    print(
        f"  accept-gate clause a (LB>0.50): {'PASS' if gate_pass else 'not met'}"
        f"  [informational — the ratified gate uses n=600]"
    )
    print("===============================================================")
    return 0


if __name__ == "__main__":
    sys.exit(main())
