"""``catan-rl-search-eval`` — offline determinized-search evaluation CLI (C5).

Evaluates a determinized-MCTS search agent (wrapping a v2 checkpoint) head-to-head
against a raw policy / the heuristic / random, seat-symmetrized, and reports the
win-rate + Wilson CI (optionally writing JSON). Off the training path; CPU by
default (eval device policy); no GUI import.

Examples::

  # Bake-off: search vs the raw policy it wraps (n=200, 50 sims/move)
  catan-rl-search-eval --ckpt <ckpt> --opponent policy:<ckpt> \\
      --sims 50 --n-games 200 --seed 0 --out runs/search/bakeoff_n200.json

  # Anytime (wall-clock) budget vs the heuristic
  catan-rl-search-eval --ckpt <ckpt> --opponent heuristic --time-budget 1.0 --n-games 100
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from catan_rl.eval.harness import EvalMatchupResult
    from catan_rl.search.config import SearchConfig


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="catan-rl-search-eval",
        description="Evaluate a determinized-search agent vs a policy/heuristic/random opponent.",
    )
    p.add_argument("--ckpt", required=True, help="search agent's v2 checkpoint (.pt)")
    p.add_argument(
        "--opponent",
        default=None,
        help="'policy:PATH' | 'heuristic' | 'random' (default: policy:<ckpt> self-match)",
    )
    budget = p.add_mutually_exclusive_group()
    budget.add_argument("--sims", type=int, default=None, help="fixed simulations per move")
    budget.add_argument(
        "--time-budget", type=float, default=None, help="anytime wall-clock seconds per move"
    )
    p.add_argument("--n-games", type=int, default=200, help="total seat-symmetrized games")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--determinizations", type=int, default=1, help="N-determinization trees per move"
    )
    p.add_argument("--c-puct", type=float, default=None, help="PUCT exploration constant")
    p.add_argument("--device", default="cpu", help="cpu (default) / mps / cuda / auto")
    p.add_argument("--max-turns", type=int, default=400)
    p.add_argument("--out", default=None, help="write the result JSON here")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.sims is None and args.time_budget is None:
        args.sims = 50  # sensible default budget when neither is given

    ckpt = Path(args.ckpt).expanduser()
    if not ckpt.exists():
        raise SystemExit(f"error: --ckpt not found at {ckpt}")
    opponent_spec: str = args.opponent if args.opponent is not None else f"policy:{ckpt}"

    # Heavy imports scoped to main (keep CLI import cheap; no GUI on the path).
    from dataclasses import replace

    from catan_rl.search.config import SearchConfig

    cfg = SearchConfig(
        sims_per_move=args.sims,
        time_budget_s=args.time_budget,
        n_determinizations=args.determinizations,
        seed=args.seed,
    )
    if args.c_puct is not None:
        cfg = replace(cfg, c_puct=args.c_puct)

    result = _run(cfg, str(ckpt), opponent_spec, args)

    summary = {
        "ckpt": str(ckpt),
        "opponent": opponent_spec,
        "sims": args.sims,
        "time_budget_s": args.time_budget,
        "determinizations": args.determinizations,
        "n": result.n,
        "wins": result.wins,
        "wr": result.wr,
        "ci_lower": result.ci.lower,
        "ci_upper": result.ci.upper,
        "rules_violations": len(result.rules_violations),
        "n_seat0": result.n_seat0,
        "n_seat1": result.n_seat1,
    }
    print(
        f"search vs {opponent_spec}: WR {result.wr:.3f} "
        f"(95% Wilson [{result.ci.lower:.3f}, {result.ci.upper:.3f}], n={result.n}); "
        f"rules_violations={len(result.rules_violations)}"
    )
    if args.out is not None:
        out = Path(args.out).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2))
        print(f"wrote {out}")
    return 0


def _run(
    cfg: SearchConfig, ckpt: str, opponent_spec: str, args: argparse.Namespace
) -> EvalMatchupResult:
    from typing import cast

    from catan_rl.replay.player_factory import PlayerSpec, _PolicyActor, build_actor
    from catan_rl.search.agent import SearchAgent
    from catan_rl.search.eval_search import evaluate_search_vs_policy, run_search_matchup

    if opponent_spec.startswith("policy:"):
        opp_ckpt = opponent_spec.split(":", 1)[1]
        return evaluate_search_vs_policy(
            cfg,
            ckpt,
            opp_ckpt,
            n_games=args.n_games,
            seed=args.seed,
            device=args.device,
            max_turns=args.max_turns,
        )

    if opponent_spec not in ("heuristic", "random"):
        raise SystemExit("error: --opponent must be 'policy:PATH', 'heuristic', or 'random'")

    search_actor = cast(
        _PolicyActor,
        build_actor(PlayerSpec(kind="policy", ckpt_path=ckpt), seed=args.seed, device=args.device),
    )
    agent = SearchAgent(search_actor.policy, cfg, device=search_actor.device)
    return run_search_matchup(
        agent,
        opponent_type=opponent_spec,
        opponent=None,
        n_games=args.n_games,
        seed=args.seed,
        max_turns=args.max_turns,
        opponent_ref=opponent_spec,
    )


if __name__ == "__main__":
    raise SystemExit(main())
