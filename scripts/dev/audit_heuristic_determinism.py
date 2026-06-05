"""E0.3 — heuristic determinism audit (v2 design §0).

Catanatron's BC-plan dissent (panel vote D9): "the heuristic has a
deterministic tiebreak on corner index, so D6-rotating the state but not
the action would teach inconsistent labels". The faculty review accepted
this as a real risk and gated the symmetry-aug prob on this audit:

  - deterministic-tiebreaker fraction < 5%   → aug-prob 1.0 is safe
  - 5–20%                                    → fall back to aug-prob 0.5
  - > 20%                                    → STOP and consider replacing
                                                tiebreakers in the heuristic
                                                before generating BC data

This script runs 200 heuristic-vs-heuristic games, instruments every
spatial decision (build_settlement, build_city, build_road, move_robber),
and records both the candidate set (rederived from the board at decision
time) and the chosen action. It then asks: when |candidates| > 1, how
often does the heuristic pick the *lexicographically-first* candidate?

For a heuristic using ``np.random.randint`` (which the v2 ``heuristicAIPlayer``
appears to do), the expected lex-first rate is ``E[1/|candidates|]`` —
substantially less than 1.0 since most decisions have many candidates.
If the observed rate exceeds the expected by ≥ 1.5x, that's evidence of a
deterministic tiebreaker.

Output: ``runs/preflight/e03/determinism.json``.
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

# REPO_ROOT preserved for argparse default-path usage at line ~317.
# The previous ``sys.path.insert(REPO_ROOT/'src')`` shim is dropped
# in the maturin sole-backend cutover — ``catan_rl`` is importable
# via the install path now.
REPO_ROOT = Path(__file__).resolve().parent.parent

import queue

from catan_rl.agents.heuristic import heuristicAIPlayer
from catan_rl.engine.game import catanGame
from catan_rl.engine.tracker import ResourceTracker


@dataclass
class TiebreakerRecord:
    """One spatial decision with its candidate set + chosen value."""

    action_method: str  # "build_settlement" | "build_city" | "build_road" | "move_robber"
    candidate_count: int
    chose_lex_first: bool
    # Only logged when |candidates| > 1; rate is what matters.


@dataclass
class GameStats:
    p1_won: int = 0
    p2_won: int = 0
    truncated: bool = False
    decisions: list[TiebreakerRecord] = field(default_factory=list)


def _sorted_candidates(candidates) -> list:
    """Sort candidate keys by str() representation — the canonical
    'lex-first' ordering. ``np.random.randint`` operating over
    ``list(d.keys())`` would respect Python dict insertion order, which is
    well-defined but board-state-dependent; the str-sort is the natural
    'deterministic across runs' baseline.
    """
    return sorted(candidates, key=str)


@contextmanager
def _instrumented(
    player: heuristicAIPlayer, records: list[TiebreakerRecord], board_ref: dict
) -> Iterator[None]:
    orig_build_settlement = player.build_settlement
    orig_build_city = player.build_city
    orig_build_road = player.build_road
    orig_move_robber = player.move_robber

    def patched_build_settlement(vCoord, board, is_free=False):
        # Re-derive candidate set at decision time.
        if is_free:
            candidates = list(board.get_setup_settlements(player).keys())
        else:
            candidates = list(board.get_potential_settlements(player).keys())
        if len(candidates) > 1:
            sorted_c = _sorted_candidates(candidates)
            records.append(
                TiebreakerRecord(
                    action_method="build_settlement",
                    candidate_count=len(candidates),
                    chose_lex_first=(vCoord == sorted_c[0]),
                )
            )
        return orig_build_settlement(vCoord, board, is_free=is_free)

    def patched_build_city(vCoord, board):
        candidates = list(board.get_potential_cities(player).keys())
        if len(candidates) > 1:
            sorted_c = _sorted_candidates(candidates)
            records.append(
                TiebreakerRecord(
                    action_method="build_city",
                    candidate_count=len(candidates),
                    chose_lex_first=(vCoord == sorted_c[0]),
                )
            )
        return orig_build_city(vCoord, board)

    def patched_build_road(v1, v2, board, is_free=False):
        if is_free:
            candidates = list(board.get_setup_roads(player).keys())
        else:
            candidates = list(board.get_potential_roads(player).keys())
        if len(candidates) > 1:
            sorted_c = _sorted_candidates(candidates)
            records.append(
                TiebreakerRecord(
                    action_method="build_road",
                    candidate_count=len(candidates),
                    chose_lex_first=((v1, v2) == sorted_c[0]),
                )
            )
        return orig_build_road(v1, v2, board, is_free=is_free)

    def patched_move_robber(hexIndex, board, player_robbed):
        candidates = list(board.get_robber_spots().keys())
        if len(candidates) > 1:
            sorted_c = _sorted_candidates(candidates)
            records.append(
                TiebreakerRecord(
                    action_method="move_robber",
                    candidate_count=len(candidates),
                    chose_lex_first=(hexIndex == sorted_c[0]),
                )
            )
        return orig_move_robber(hexIndex, board, player_robbed)

    player.build_settlement = patched_build_settlement  # type: ignore[method-assign]
    player.build_city = patched_build_city  # type: ignore[method-assign]
    player.build_road = patched_build_road  # type: ignore[method-assign]
    player.move_robber = patched_move_robber  # type: ignore[method-assign]

    try:
        yield
    finally:
        player.build_settlement = orig_build_settlement  # type: ignore[method-assign]
        player.build_city = orig_build_city  # type: ignore[method-assign]
        player.build_road = orig_build_road  # type: ignore[method-assign]
        player.move_robber = orig_move_robber  # type: ignore[method-assign]


def _grant_setup_resources(game: catanGame, p: heuristicAIPlayer) -> None:
    if not p.buildGraph["SETTLEMENTS"]:
        return
    last = p.buildGraph["SETTLEMENTS"][-1]
    for adj_hex in game.board.boardGraph[last].adjacent_hex_indices:
        res = game.board.hexTileDict[adj_hex].resource_type
        if res != "DESERT":
            p.resources[res] += 1
            game.broadcast.resource_change(p.name, {res: 1}, "SETUP")


def _play_one_game(seed: int, max_turns: int = 400) -> GameStats:
    np.random.seed(seed)
    game = catanGame(render_mode=None)
    board = game.board

    p1 = heuristicAIPlayer("P1", "black")
    p1.game = game
    p1.isAI = True
    p1.updateAI()
    p2 = heuristicAIPlayer("P2", "darkslateblue")
    p2.game = game
    p2.isAI = True
    p2.updateAI()

    game.playerQueue = queue.Queue(2)
    game.playerQueue.put(p1)
    game.playerQueue.put(p2)
    game.resource_tracker = ResourceTracker([p1.name, p2.name])

    stats = GameStats()
    board_ref = {"board": board}

    with (
        _instrumented(p1, stats.decisions, board_ref),
        _instrumented(p2, stats.decisions, board_ref),
    ):
        for p in (p1, p2):
            game.currentPlayer = p
            p.initial_setup(board)
        for p in (p2, p1):
            game.currentPlayer = p
            p.initial_setup(board)
            _grant_setup_resources(game, p)
        game.gameSetup = False

        turn = 0
        while not game.gameOver and turn < max_turns:
            for p in (p1, p2):
                if game.gameOver:
                    break
                game.currentPlayer = p
                p.updateDevCards()
                p.devCardPlayedThisTurn = False
                dice = game.rollDice()
                if dice == 7:
                    for pp in (p1, p2):
                        if sum(pp.resources.values()) > 9:
                            pp.discardResources(game)
                    p.heuristic_move_robber(board)
                else:
                    game.update_playerResources(dice, p)
                if p.victoryPoints >= game.maxPoints:
                    game.gameOver = True
                    break
                p.move(board)
                game.check_longest_road(p)
                game.check_largest_army(p)
                if p.victoryPoints >= game.maxPoints:
                    game.gameOver = True
                    break
            turn += 1

    stats.p1_won = int(p1.victoryPoints >= game.maxPoints)
    stats.p2_won = int(p2.victoryPoints >= game.maxPoints)
    stats.truncated = not (stats.p1_won or stats.p2_won)
    return stats


def _aggregate(records: list[GameStats]) -> dict[str, object]:
    all_decisions: list[TiebreakerRecord] = []
    for r in records:
        all_decisions.extend(r.decisions)

    n_multi_candidate = len(all_decisions)
    if not n_multi_candidate:
        return {
            "metadata": {"n_games": len(records), "n_multi_candidate_decisions": 0},
            "deterministic_tiebreaker_fraction": 0.0,
            "expected_lex_first_under_uniform_random": 0.0,
            "decision_rule": "INSUFFICIENT_DATA",
        }

    chose_lex_first_count = sum(d.chose_lex_first for d in all_decisions)
    observed_rate = chose_lex_first_count / n_multi_candidate
    # Expected lex-first rate under perfect uniform random over candidates:
    expected_rate = float(np.mean([1.0 / d.candidate_count for d in all_decisions]))

    # Per-method breakdown
    by_method: dict[str, dict[str, float]] = {}
    for method in ("build_settlement", "build_city", "build_road", "move_robber"):
        method_decisions = [d for d in all_decisions if d.action_method == method]
        if not method_decisions:
            continue
        n = len(method_decisions)
        first = sum(d.chose_lex_first for d in method_decisions)
        exp_first = float(np.mean([1.0 / d.candidate_count for d in method_decisions]))
        by_method[method] = {
            "n_decisions": n,
            "observed_lex_first_rate": first / n,
            "expected_lex_first_rate_under_uniform": exp_first,
            "ratio_observed_over_expected": (first / n) / max(exp_first, 1e-12),
        }

    # Apply decision rule. The "deterministic-tiebreaker fraction" we
    # report is the *excess* over uniform-random, expressed as a fraction
    # of the observation. If observed equals expected, no determinism;
    # if observed >> expected, strong determinism.
    excess = max(0.0, observed_rate - expected_rate)
    deterministic_fraction = excess  # interpretation: fraction of decisions
    # that would be lex-first beyond random.

    if deterministic_fraction < 0.05:
        rule = "AUG_PROB_1.0"
    elif deterministic_fraction < 0.20:
        rule = "AUG_PROB_0.5"
    else:
        rule = "STOP_AND_AUDIT"

    return {
        "metadata": {
            "n_games": len(records),
            "n_multi_candidate_decisions": n_multi_candidate,
            "p1_wins": sum(r.p1_won for r in records),
            "p2_wins": sum(r.p2_won for r in records),
            "truncated": sum(r.truncated for r in records),
        },
        "observed_lex_first_rate": observed_rate,
        "expected_lex_first_under_uniform_random": expected_rate,
        "deterministic_tiebreaker_fraction": deterministic_fraction,
        "decision_rule": rule,
        "by_method": by_method,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--n-games", type=int, default=200)
    parser.add_argument("--max-turns", type=int, default=400)
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "runs" / "preflight" / "e03" / "determinism.json",
    )
    parser.add_argument("--save-raw", action="store_true")
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)

    print(f"[E0.3] running {args.n_games} heuristic-vs-heuristic games...", flush=True)
    t0 = time.time()
    games = []
    for i in range(args.n_games):
        games.append(_play_one_game(args.seed_offset + i, max_turns=args.max_turns))
        if (i + 1) % 50 == 0:
            print(f"  ... {i + 1}/{args.n_games} games in {time.time() - t0:.1f}s", flush=True)
    dt = time.time() - t0
    print(f"[E0.3] done in {dt:.1f}s", flush=True)

    summary = _aggregate(games)
    summary["metadata"]["wall_clock_seconds"] = dt
    summary["metadata"]["seed_offset"] = args.seed_offset
    args.out.write_text(json.dumps(summary, indent=2))
    print(f"[E0.3] wrote {args.out}", flush=True)
    print(
        f"[E0.3] observed_lex_first={summary['observed_lex_first_rate']:.4f} "
        f"vs expected_under_uniform={summary['expected_lex_first_under_uniform_random']:.4f}",
        flush=True,
    )
    print(
        f"[E0.3] deterministic_tiebreaker_fraction={summary['deterministic_tiebreaker_fraction']:.4f} "  # noqa: E501
        f"-> decision_rule={summary['decision_rule']}",
        flush=True,
    )

    if args.save_raw:
        raw = [asdict(g) for g in games]
        raw_path = args.out.with_suffix(".raw.json")
        raw_path.write_text(json.dumps(raw, indent=2))
        print(f"[E0.3] wrote raw records to {raw_path}", flush=True)


if __name__ == "__main__":
    main()
