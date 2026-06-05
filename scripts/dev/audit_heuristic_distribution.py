"""E0.2 — heuristic action distribution audit (v2 design §0).

Runs 1000 heuristic-vs-heuristic games and records the heuristic's
6-head action choices at every decision point. Produces the baseline
numbers the BC Gate 2 calibration depends on:

  - Per-head marginal frequency distributions (P(action_head = i)).
  - Conditional entropies H(action_head | type).
  - Frequency-baseline NLL: −log P_marginal(a_h) summed over heads.
  - Top-1 baseline accuracy: ``argmax P_marginal == observed_action``.
  - BASE_WR_HEUR_SELF: P1 / P2 / symmetrised heuristic-vs-heuristic WR.

Output: ``runs/preflight/e02/distribution.json``.

Instrumentation strategy
------------------------
The heuristic's ``move()`` makes ≤ ~6 inline decisions per turn by calling
``self.build_settlement``, ``self.build_city``, ``self.build_road``,
``self.draw_devCard``, ``self.trade_with_bank``. We monkey-patch each of
those *on the player instance* (not on the class) for the duration of
one game; the production source in ``catan_rl/agents/heuristic.py``
is never modified.

Forced single-option decisions (e.g., the dice roll at the start of every
turn, the discard split, the robber placement under Friendly Robber) are
recorded but treated as a separate bucket — they don't carry policy
information and the BC plan §6 Gate 2 explicitly skips them.

Per-turn flow we instrument:

  start-of-turn   →  type=ROLL_DICE (implicit; no head choice)
  dice == 7       →  type=DISCARD ×N if hand > 9 cards (skip — forced)
                  →  type=MOVE_ROBBER + tile choice + steal target
  main turn       →  trade() may call BankTrade (1+)
                  →  BuildSettlement (0/1)
                  →  BuildCity (0/1)
                  →  BuildRoad (0..2)
                  →  BuyDevCard (0/1)
  end of turn     →  type=END_TURN (implicit; no head choice)

Setup actions (``is_free=True`` calls during snake-draft) are recorded
separately. They're decisions but the heuristic only picks among legal
candidates uniformly at random, so the marginal will look uniform.

This script does NOT use ``CatanEnv`` — it drives the engine directly,
because the env's state machine expects external actions but here we
have two heuristic agents making decisions inline. The trade-off: we
lose the env's mask-validation safety net, but gain the ability to
observe every individual heuristic decision.
"""

from __future__ import annotations

import argparse
import json
import queue

# Path setup so the script runs from the repo root without `pip install -e .`.
import sys
import time
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from catan_rl.agents.heuristic import heuristicAIPlayer
from catan_rl.engine.game import catanGame
from catan_rl.engine.tracker import ResourceTracker
from catan_rl.policy.obs_schema import (
    N_ACTION_TYPES,
    N_EDGES,
    N_RESOURCES,
    N_TILES,
    N_VERTICES,
    RESOURCES_CW,
    ActionType,
)

# ---------------------------------------------------------------------------
# Data records
# ---------------------------------------------------------------------------


@dataclass
class DecisionRecord:
    """One observed heuristic decision.

    Fields use the same indexing convention as the v2 6-head action:
      ``(type, corner, edge, tile, resource1, resource2)``.

    For heads not relevant to this action_type, the value is -1.
    ``phase`` distinguishes setup / main / robber so we can compute
    conditional stats.
    """

    action_type: int
    corner_idx: int = -1
    edge_idx: int = -1
    tile_idx: int = -1
    resource1_idx: int = -1
    resource2_idx: int = -1
    phase: str = "main"  # "setup" | "main" | "robber" | "discard"
    player_seat: int = 0  # 0 for P1, 1 for P2


@dataclass
class GameRecord:
    """Aggregate record for one game."""

    seat_p1_won: int = 0
    seat_p2_won: int = 0
    truncated: bool = False
    total_turns: int = 0
    decisions: list[DecisionRecord] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Index lookups
# ---------------------------------------------------------------------------


class _BoardIndexer:
    """Build the canonical vertex / edge index maps once per game.

    These must match ``CatanEnv._build_index_maps`` so the recorded
    action indices align with what the v2 policy network expects.
    """

    def __init__(self, board: object) -> None:
        self.board = board
        self._vertex_to_idx: dict[object, int] = {
            px: idx for idx, px in board.vertex_index_to_pixel_dict.items()
        }
        seen: set[tuple[str, str]] = set()
        self._edge_to_idx: dict[tuple[str, str], int] = {}
        for v_pt, v_obj in board.boardGraph.items():
            for nb_pt in v_obj.neighbors:
                key = _edge_key(v_pt, nb_pt)
                if key in seen:
                    continue
                seen.add(key)
                self._edge_to_idx[key] = len(self._edge_to_idx)
        if len(self._edge_to_idx) != N_EDGES:
            raise RuntimeError(
                f"E0.2 indexer derived {len(self._edge_to_idx)} edges, expected {N_EDGES}"
            )

    def vertex(self, px: object) -> int:
        return self._vertex_to_idx[px]

    def edge(self, v1: object, v2: object) -> int:
        return self._edge_to_idx[_edge_key(v1, v2)]


def _edge_key(v1: object, v2: object) -> tuple[str, str]:
    s1, s2 = str(v1), str(v2)
    return (s1, s2) if s1 < s2 else (s2, s1)


def _res_to_cw_idx(res: str) -> int:
    return RESOURCES_CW.index(res) if res in RESOURCES_CW else -1


# ---------------------------------------------------------------------------
# Monkey-patch context manager
# ---------------------------------------------------------------------------


@contextmanager
def _instrumented_heuristic(
    player: heuristicAIPlayer,
    seat: int,
    indexer: _BoardIndexer,
    out_records: list[DecisionRecord],
    phase_ref: dict[str, str],
) -> Iterator[None]:
    """Wrap a heuristic player's action methods with recorders for the
    lifetime of the context. On exit, the originals are restored.

    ``phase_ref`` is a one-key dict {``"phase": <str>``} that the caller
    mutates as the game state machine advances. This lets the recorders
    annotate each record with the current phase without us threading
    that state through every patched method.
    """

    orig_build_settlement = player.build_settlement
    orig_build_city = player.build_city
    orig_build_road = player.build_road
    orig_draw_dev = player.draw_devCard
    orig_trade_bank = player.trade_with_bank
    orig_move_robber = player.move_robber

    def patched_build_settlement(vCoord, board, is_free=False):
        phase = "setup" if is_free else phase_ref["phase"]
        out_records.append(
            DecisionRecord(
                action_type=ActionType.BUILD_SETTLEMENT,
                corner_idx=indexer.vertex(vCoord),
                phase=phase,
                player_seat=seat,
            )
        )
        return orig_build_settlement(vCoord, board, is_free=is_free)

    def patched_build_city(vCoord, board):
        out_records.append(
            DecisionRecord(
                action_type=ActionType.BUILD_CITY,
                corner_idx=indexer.vertex(vCoord),
                phase=phase_ref["phase"],
                player_seat=seat,
            )
        )
        return orig_build_city(vCoord, board)

    def patched_build_road(v1, v2, board, is_free=False):
        phase = "setup" if is_free else phase_ref["phase"]
        out_records.append(
            DecisionRecord(
                action_type=ActionType.BUILD_ROAD,
                edge_idx=indexer.edge(v1, v2),
                phase=phase,
                player_seat=seat,
            )
        )
        return orig_build_road(v1, v2, board, is_free=is_free)

    def patched_draw_dev(board):
        out_records.append(
            DecisionRecord(
                action_type=ActionType.BUY_DEV_CARD,
                phase=phase_ref["phase"],
                player_seat=seat,
            )
        )
        return orig_draw_dev(board)

    def patched_trade_bank(r1, r2):
        out_records.append(
            DecisionRecord(
                action_type=ActionType.BANK_TRADE,
                resource1_idx=_res_to_cw_idx(r1),
                resource2_idx=_res_to_cw_idx(r2),
                phase=phase_ref["phase"],
                player_seat=seat,
            )
        )
        return orig_trade_bank(r1, r2)

    def patched_move_robber(hexIndex, board, player_robbed):
        out_records.append(
            DecisionRecord(
                action_type=ActionType.MOVE_ROBBER,
                tile_idx=int(hexIndex),
                phase="robber",
                player_seat=seat,
            )
        )
        return orig_move_robber(hexIndex, board, player_robbed)

    player.build_settlement = patched_build_settlement  # type: ignore[method-assign]
    player.build_city = patched_build_city  # type: ignore[method-assign]
    player.build_road = patched_build_road  # type: ignore[method-assign]
    player.draw_devCard = patched_draw_dev  # type: ignore[method-assign]
    player.trade_with_bank = patched_trade_bank  # type: ignore[method-assign]
    player.move_robber = patched_move_robber  # type: ignore[method-assign]

    try:
        yield
    finally:
        # Restore originals so subsequent games don't double-instrument.
        player.build_settlement = orig_build_settlement  # type: ignore[method-assign]
        player.build_city = orig_build_city  # type: ignore[method-assign]
        player.build_road = orig_build_road  # type: ignore[method-assign]
        player.draw_devCard = orig_draw_dev  # type: ignore[method-assign]
        player.trade_with_bank = orig_trade_bank  # type: ignore[method-assign]
        player.move_robber = orig_move_robber  # type: ignore[method-assign]


# ---------------------------------------------------------------------------
# One-game driver
# ---------------------------------------------------------------------------


def _play_one_game(
    seed: int,
    max_turns: int,
) -> GameRecord:
    """Run one heuristic-vs-heuristic game and return its record.

    Engine is driven directly (not via CatanEnv) because both players are
    heuristics and we instrument their inline action methods.
    """
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

    indexer = _BoardIndexer(board)
    record = GameRecord()
    phase_ref = {"phase": "setup"}

    with (
        _instrumented_heuristic(p1, 0, indexer, record.decisions, phase_ref),
        _instrumented_heuristic(p2, 1, indexer, record.decisions, phase_ref),
    ):
        # ---- Setup phase (snake draft) ----
        phase_ref["phase"] = "setup"
        for p in (p1, p2):
            game.currentPlayer = p
            p.initial_setup(board)
        for p in (p2, p1):
            game.currentPlayer = p
            p.initial_setup(board)
            _grant_setup_resources(game, p)
        game.gameSetup = False

        # ---- Main loop ----
        phase_ref["phase"] = "main"
        while not game.gameOver and record.total_turns < max_turns:
            for p in (p1, p2):
                if game.gameOver:
                    break
                game.currentPlayer = p
                p.updateDevCards()
                p.devCardPlayedThisTurn = False

                phase_ref["phase"] = "roll"
                dice = game.rollDice()
                if dice == 7:
                    # Discards (recorded only as type=DISCARD count, no head
                    # choices logged — these are forced per-card discards).
                    for pp in (p1, p2):
                        if sum(pp.resources.values()) > 9:
                            pp.discardResources(game)
                    # Robber placement is itself a heuristic choice → captured
                    # by the patched move_robber inside heuristic_move_robber.
                    p.heuristic_move_robber(board)
                else:
                    game.update_playerResources(dice, p)

                if p.victoryPoints >= game.maxPoints:
                    game.gameOver = True
                    break

                phase_ref["phase"] = "main"
                p.move(board)
                game.check_longest_road(p)
                game.check_largest_army(p)
                if p.victoryPoints >= game.maxPoints:
                    game.gameOver = True
                    break

            record.total_turns += 1

    record.seat_p1_won = int(p1.victoryPoints >= game.maxPoints)
    record.seat_p2_won = int(p2.victoryPoints >= game.maxPoints)
    record.truncated = not (record.seat_p1_won or record.seat_p2_won)
    return record


def _grant_setup_resources(game: catanGame, p: heuristicAIPlayer) -> None:
    """Grant starting resources from the player's last-placed settlement."""
    if not p.buildGraph["SETTLEMENTS"]:
        return
    last = p.buildGraph["SETTLEMENTS"][-1]
    for adj_hex in game.board.boardGraph[last].adjacent_hex_indices:
        res = game.board.hexTileDict[adj_hex].resource_type
        if res != "DESERT":
            p.resources[res] += 1
            game.broadcast.resource_change(p.name, {res: 1}, "SETUP")


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _aggregate(records: list[GameRecord]) -> dict[str, object]:
    """Compute summary statistics across all games."""
    all_decisions: list[DecisionRecord] = []
    for r in records:
        all_decisions.extend(r.decisions)

    # ---- WR by seat ----
    n = len(records)
    p1_wins = sum(r.seat_p1_won for r in records)
    p2_wins = sum(r.seat_p2_won for r in records)
    n_truncated = sum(r.truncated for r in records)
    wr_p1 = p1_wins / n if n else 0.0
    wr_p2 = p2_wins / n if n else 0.0
    # Symmetrised WR for self-play: by symmetry of the policy, the
    # heuristic's WR when it plays either seat is the seat's per-game
    # win rate. The symmetrised metric is the mean of the two seat-WRs;
    # for a non-truncating game it lands at 0.5 (one side wins every
    # game). For our truncating env it's 0.5 × (1 − P(truncated)).  # noqa: RUF003
    #
    # This is the same formulation the BC Gate 3 TOST equivalence test
    # uses: BC's symmetrised WR vs the heuristic should equal the
    # heuristic's own symmetrised self-WR within the test's tolerance.
    wr_symmetrised = 0.5 * (wr_p1 + wr_p2)

    avg_turns = float(np.mean([r.total_turns for r in records])) if records else 0.0

    # ---- Per-head marginals ----
    # For each main-turn decision, count which action type was chosen.
    main_type_counter: Counter[int] = Counter()
    main_corner_counter: Counter[int] = Counter()
    main_edge_counter: Counter[int] = Counter()
    main_tile_counter: Counter[int] = Counter()
    main_res1_counter: Counter[int] = Counter()
    main_res2_counter: Counter[int] = Counter()

    setup_type_counter: Counter[int] = Counter()
    setup_corner_counter: Counter[int] = Counter()
    setup_edge_counter: Counter[int] = Counter()

    for d in all_decisions:
        if d.phase == "main":
            main_type_counter[d.action_type] += 1
            if d.corner_idx >= 0:
                main_corner_counter[d.corner_idx] += 1
            if d.edge_idx >= 0:
                main_edge_counter[d.edge_idx] += 1
            if d.tile_idx >= 0:
                main_tile_counter[d.tile_idx] += 1
            if d.resource1_idx >= 0:
                main_res1_counter[d.resource1_idx] += 1
            if d.resource2_idx >= 0:
                main_res2_counter[d.resource2_idx] += 1
        elif d.phase == "setup":
            setup_type_counter[d.action_type] += 1
            if d.corner_idx >= 0:
                setup_corner_counter[d.corner_idx] += 1
            if d.edge_idx >= 0:
                setup_edge_counter[d.edge_idx] += 1
        elif d.phase == "robber":
            if d.tile_idx >= 0:
                main_tile_counter[d.tile_idx] += 1
            main_type_counter[d.action_type] += 1

    # ---- Frequency-baseline analyses ----
    main_total = sum(main_type_counter.values())
    main_type_marginal = (
        {a: c / main_total for a, c in main_type_counter.items()} if main_total else {}
    )

    # Top-1 baseline: always predict the modal action_type in main-turn.
    if main_type_counter:
        modal_type, modal_count = main_type_counter.most_common(1)[0]
        base_top1_type = modal_count / main_total
    else:
        modal_type = -1
        base_top1_type = 0.0

    # Frequency-baseline NLL on the type head (averaged over decisions).
    if main_total:
        nll_type = 0.0
        for action_type, count in main_type_counter.items():  # noqa: B007
            p = count / main_total
            nll_type += count * (-np.log(p + 1e-12))
        nll_type /= main_total
    else:
        nll_type = 0.0

    # ---- Entropy ----
    def _entropy(counter: Counter[int]) -> float:
        total = sum(counter.values())
        if not total:
            return 0.0
        return float(sum(-c / total * np.log(c / total + 1e-12) for c in counter.values()))

    return {
        "metadata": {
            "n_games": n,
            "p1_wins": p1_wins,
            "p2_wins": p2_wins,
            "n_truncated": n_truncated,
            "avg_turns_per_game": avg_turns,
            "total_decisions": len(all_decisions),
            "main_phase_decisions": main_total,
            "setup_phase_decisions": sum(setup_type_counter.values()),
        },
        "BASE_WR_HEUR_SELF": {
            "p1_seat": wr_p1,
            "p2_seat": wr_p2,
            "symmetrised": wr_symmetrised,
        },
        "BASE_TOP1_FREQ": {
            "type_modal_action": int(modal_type),
            "type_top1_accuracy": float(base_top1_type),
        },
        "BASE_NLL_FREQ": {
            "type": float(nll_type),
            # Per-head conditional NLLs are best computed against the BC
            # dataset directly (Step 3), which knows the legal-action mask
            # at each (state, action) pair. The raw marginals below let
            # downstream code compute them quickly without re-running games.
        },
        "main_type_marginal": main_type_marginal,
        "type_head_entropy_main": _entropy(main_type_counter),
        "corner_head_entropy_main": _entropy(main_corner_counter),
        "edge_head_entropy_main": _entropy(main_edge_counter),
        "tile_head_entropy_main": _entropy(main_tile_counter),
        "resource1_head_entropy_main": _entropy(main_res1_counter),
        "resource2_head_entropy_main": _entropy(main_res2_counter),
        "setup_corner_entropy": _entropy(setup_corner_counter),
        "setup_edge_entropy": _entropy(setup_edge_counter),
        # Per-action-type counts (the raw input to any downstream
        # baseline calculation).
        "main_type_counts": {str(a): c for a, c in sorted(main_type_counter.items())},
        "main_corner_counts": {str(a): c for a, c in sorted(main_corner_counter.items())},
        "main_edge_counts": {str(a): c for a, c in sorted(main_edge_counter.items())},
        "main_tile_counts": {str(a): c for a, c in sorted(main_tile_counter.items())},
        "main_resource1_counts": {str(a): c for a, c in sorted(main_res1_counter.items())},
        "main_resource2_counts": {str(a): c for a, c in sorted(main_res2_counter.items())},
        "setup_corner_counts": {str(a): c for a, c in sorted(setup_corner_counter.items())},
        "setup_edge_counts": {str(a): c for a, c in sorted(setup_edge_counter.items())},
        "schema": {
            "N_ACTION_TYPES": N_ACTION_TYPES,
            "N_VERTICES": N_VERTICES,
            "N_EDGES": N_EDGES,
            "N_TILES": N_TILES,
            "N_RESOURCES": N_RESOURCES,
            "resource_order": list(RESOURCES_CW),
        },
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--n-games", type=int, default=1000, help="Number of games to play")
    parser.add_argument(
        "--max-turns",
        type=int,
        default=400,
        help="Per-game turn cap (truncates uneventful games)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "runs" / "preflight" / "e02" / "distribution.json",
        help="Output JSON path",
    )
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument(
        "--save-raw",
        action="store_true",
        help="Also dump per-game decision records to <out>.raw.json (large)",
    )
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)

    print(f"[E0.2] running {args.n_games} heuristic-vs-heuristic games...", flush=True)
    t0 = time.time()
    records: list[GameRecord] = []
    for i in range(args.n_games):
        rec = _play_one_game(seed=args.seed_offset + i, max_turns=args.max_turns)
        records.append(rec)
        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            print(
                f"  ... {i + 1}/{args.n_games} games in {elapsed:.1f}s "
                f"(P1 wins so far: {sum(r.seat_p1_won for r in records)}, "
                f"truncated: {sum(r.truncated for r in records)})",
                flush=True,
            )
    dt = time.time() - t0
    print(f"[E0.2] {args.n_games} games complete in {dt:.1f}s", flush=True)

    summary = _aggregate(records)
    summary["metadata"]["wall_clock_seconds"] = dt
    summary["metadata"]["seed_offset"] = args.seed_offset

    args.out.write_text(json.dumps(summary, indent=2))
    print(f"[E0.2] wrote {args.out}", flush=True)
    print(
        f"[E0.2] BASE_WR_HEUR_SELF: P1={summary['BASE_WR_HEUR_SELF']['p1_seat']:.3f}, "
        f"P2={summary['BASE_WR_HEUR_SELF']['p2_seat']:.3f}, "
        f"sym={summary['BASE_WR_HEUR_SELF']['symmetrised']:.3f}",
        flush=True,
    )

    if args.save_raw:
        raw_path = args.out.with_suffix(".raw.json")
        raw_data = [
            {
                "seat_p1_won": r.seat_p1_won,
                "seat_p2_won": r.seat_p2_won,
                "truncated": r.truncated,
                "total_turns": r.total_turns,
                "decisions": [asdict(d) for d in r.decisions],
            }
            for r in records
        ]
        raw_path.write_text(json.dumps(raw_data, indent=2))
        print(f"[E0.2] wrote raw records to {raw_path}", flush=True)


if __name__ == "__main__":
    main()
