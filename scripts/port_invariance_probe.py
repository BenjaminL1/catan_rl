#!/usr/bin/env python
"""D7 invariance probe: do ports move the policy's opening choice at all?

The port-harvest spec sets NO per-slot accuracy tolerance, because per-slot
accuracy is not the operative quantity. The operative quantity is the
**decision-flip rate** -- P(the policy's chosen corner changes | the port map
changes) -- and it is what decides whether re-ingesting ~100 videos for the
pixel-less corpus rows is worth a day. This script measures it.

Two legs, both reported:

**real-vs-guessed** (the mandated number). For every harvested board, the
policy's deterministic setup choices under the **real** harvested port map are
compared against its choices under each of **K=8** guessed maps -- the same
"deterministic per (row, k)" convention `human-opening-reference` D2 uses. Needs
`data/human/ports/harvest.jsonl`, i.e. the hand-labelling step must have run.

**guessed-vs-guessed** (a label-free lower bound, additive). The same comparison
run *between* the 8 guessed maps on the real hex layouts. This answers the
headline question -- "is the opening choice port-sensitive AT ALL?" -- with zero
labels, and it lower-bounds the real-vs-guessed flip rate: if the policy does not
move between two arbitrary port maps it will not move for the true one either.
It is a supplement to the mandated number, never a substitute for it.

CPU only. Read-only with respect to training: reads a frozen checkpoint, writes
only under ``data/human/ports/``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import opening_sweep as OS
from catan_rl.env.catan_env import CatanEnv
from catan_rl.human_data import ports as P
from catan_rl.policy.obs_encoder import ObsEncoder
from catan_rl.policy.obs_tensor import masks_to_torch, obs_to_torch

DEFAULT_FRAMES = REPO_ROOT / "data/human/vlm_spike/frames"
DEFAULT_CORPUS = REPO_ROOT / "data/human/corpus/provisional_openings.jsonl"
DEFAULT_OUT = REPO_ROOT / "data/human/ports"

#: The 9 slot contents of every standard board, expanded to a flat list. A
#: "guessed" map is a permutation of exactly this -- guessing is about WHICH slot
#: holds what, never about the composition.
EXPANDED_PORTS: tuple[str, ...] = (
    "2:1 BRICK",
    "2:1 ORE",
    "2:1 SHEEP",
    "2:1 WHEAT",
    "2:1 WOOD",
    "3:1 PORT",
    "3:1 PORT",
    "3:1 PORT",
    "3:1 PORT",
)

K_GUESSES = 8


def guessed_names(video_id: str, game_index: int, k: int) -> tuple[str, ...]:
    """Deterministic per-(row, k) permutation of the fixed composition."""
    digest = hashlib.sha256(f"{video_id}|{game_index}|{k}".encode()).hexdigest()[:16]
    rng = np.random.default_rng(int(digest, 16))
    return tuple(EXPANDED_PORTS[i] for i in rng.permutation(len(EXPANDED_PORTS)))


class PortProbeEnv(CatanEnv):
    """A :class:`CatanEnv` whose board is welded to a fixed layout + port map.

    ``CatanEnv.reset`` builds a random board and, when the agent sits at seat 1,
    immediately plays the opponent's first placement -- before any caller could
    inject. So the injection is done inside ``reset``: the parent always runs as
    seat 0 (nothing is placed), the board is overwritten, the observation encoder
    is rebuilt so the new ports reach the obs, and only then is the seat-1
    opponent placement played.
    """

    def __init__(self, hexes: list[dict[str, Any]], robber_hex: int) -> None:
        super().__init__(opponent_type="heuristic")
        self._probe_hexes = hexes
        self._probe_robber = robber_hex
        self._probe_ports: dict[str, list[int]] = {}

    def set_ports(self, port_assignment: dict[str, list[int]]) -> None:
        self._probe_ports = port_assignment

    def reset(  # type: ignore[override]
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        import queue as _queue

        opts = dict(options or {})
        seat = int(opts.get("agent_seat", 0))
        opts["agent_seat"] = 0
        obs, info = super().reset(seed=seed, options=opts)
        assert self.game is not None
        board = self.game.board
        board.inject_hex_layout(
            [str(h["resource"]) for h in self._probe_hexes],
            [h["number"] for h in self._probe_hexes],
            self._probe_robber,
        )
        board.updatePorts(port_assignment=self._probe_ports)
        self._obs_encoder = ObsEncoder(board)
        self._build_index_maps(board)
        if seat == 1:
            self._agent_seat = 1
            self.game.playerQueue = _queue.Queue(2)
            self.game.playerQueue.put(self.opponent_player)
            self.game.playerQueue.put(self.agent_player)
            self._opponent_setup_placement()
        return self._get_obs(), info


def assert_ports_reach_obs(env: PortProbeEnv, video_id: str, game_index: int) -> float:
    """Fail loudly unless two different port maps produce two different observations.

    Without this the probe is unfalsifiable: an injection that silently did not
    take would report a 0.0 flip rate -- indistinguishable from genuine port
    invariance, and far more likely. Returns the L1 obs delta.
    """
    snapshots = []
    for k in (0, 1):
        env.set_ports(P.build_port_assignment(guessed_names(video_id, game_index, k)))
        obs, _ = env.reset(seed=0, options={"agent_seat": 0})
        snapshots.append({key: np.asarray(val, dtype=float).copy() for key, val in obs.items()})
    delta = sum(float(np.abs(snapshots[0][k] - snapshots[1][k]).sum()) for k in snapshots[0])
    if delta <= 0.0:
        raise SystemExit(
            "port injection did not change the observation -- the probe would be vacuous"
        )
    return delta


def setup_decisions(
    env: PortProbeEnv, policy: Any, device: torch.device, seed: int, seat: int
) -> tuple[tuple[int, int], dict[str, float | None]]:
    """Play the 4 deterministic setup decisions; return the 2 settlement ids + pair metrics."""
    obs, _ = env.reset(seed=seed, options={"agent_seat": seat})
    assert env.game is not None
    scorer = OS.BoardScorer(env.game.board)
    chosen: list[int] = []
    objs: list[Any] = []
    for d_idx in range(4):
        settlement = d_idx in (0, 2)
        masks = env.get_action_masks()
        obs_t = obs_to_torch(obs, device, add_batch=True)
        masks_t = masks_to_torch(masks, device, add_batch=True)
        logp = OS.setup_logp(policy, obs_t, masks_t, settlement)[0]
        chosen_id = int(logp.argmax().item())
        act = np.zeros(6, dtype=np.int64)
        if settlement:
            act[0], act[1] = OS.BUILD_SETTLEMENT, chosen_id
            chosen.append(chosen_id)
            objs.append(env._idx_to_vertex[chosen_id])
        else:
            act[0], act[2] = OS.BUILD_ROAD, chosen_id
        if d_idx == 3:
            break
        obs, _, _, _, _ = env.step(act)
    return (chosen[0], chosen[1]), scorer.pair_metrics(objs[0], objs[1])


#: Bootstrap resamples used for the clustered interval on a flip rate.
BOOTSTRAP_DRAWS = 20_000


def _flip_stats(
    cells: list[list[tuple[tuple[int, int], tuple[int, int]]]],
) -> dict[str, Any]:
    """Flip rates over comparisons grouped BY CELL, with a clustered 95% interval.

    ``cells`` is one list of comparisons per (board, seat) cell. The comparisons
    inside a cell are **not independent** -- real-vs-guessed reuses the same real
    pair 8 times, guessed-vs-guessed reuses each of the 8 guesses 7 times -- so a
    binomial interval on the raw comparison count would be far too narrow. The
    interval here is a **cluster bootstrap**: resample whole cells with replacement
    and recompute the pooled rate, which is the standard treatment for this design
    and is the same "report the uncertainty, not just the point estimate" discipline
    the eval stack applies elsewhere (``src/catan_rl/eval/wilson.py``).
    """
    flat = [p for cell in cells for p in cell]
    n = len(flat)
    if n == 0:
        return {"comparisons": 0}

    def rate(pairs: list[tuple[tuple[int, int], tuple[int, int]]], idx: int | None) -> float:
        if idx is None:
            return sum(1 for a, b in pairs if a != b) / len(pairs)
        return sum(1 for a, b in pairs if a[idx] != b[idx]) / len(pairs)

    rng = np.random.default_rng(0)
    counts = np.array([sum(1 for a, b in cell if a != b) for cell in cells], float)
    sizes = np.array([len(cell) for cell in cells], float)
    draws = rng.integers(0, len(cells), size=(BOOTSTRAP_DRAWS, len(cells)))
    boot = counts[draws].sum(axis=1) / sizes[draws].sum(axis=1)
    lo, hi = (float(v) for v in np.percentile(boot, [2.5, 97.5]))
    return {
        "comparisons": n,
        "cells": len(cells),
        "settlement_1_flip_rate": round(rate(flat, 0), 4),
        "settlement_2_flip_rate": round(rate(flat, 1), 4),
        "realised_pair_flip_rate": round(rate(flat, None), 4),
        "realised_pair_flip_rate_ci95": [round(lo, 4), round(hi, 4)],
        "ci_method": f"cluster bootstrap over {len(cells)} (board, seat) cells, "
        f"{BOOTSTRAP_DRAWS} draws",
        "cells_with_any_flip": int(sum(1 for c in counts if c > 0)),
    }


def _metric_spread(rows: list[dict[str, float | None]]) -> dict[str, Any]:
    """Per-metric spread (max - min) across the port conditions of one board."""
    out: dict[str, Any] = {}
    keys = sorted({k for r in rows for k in r})
    for k in keys:
        vals = [float(r[k]) for r in rows if r.get(k) is not None]
        if len(vals) >= 2:
            out[k] = round(max(vals) - min(vals), 4)
    return out


def load_boards(corpus: Path, frames_root: Path, sidecar: Path) -> list[dict[str, Any]]:
    """Boards to probe: harvested dirs, hex layout from the corpus row or ``meta.json``."""
    corpus_rows: dict[tuple[str, int], dict[str, Any]] = {}
    if corpus.exists():
        for line in corpus.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                corpus_rows[(row["video_id"], int(row["game_index"]))] = row
    real_maps: dict[tuple[str, int], tuple[str, ...]] = {}
    if sidecar.exists():
        for line in sidecar.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                key = (row["video_id"], int(row["game_index"]))
                real_maps[key] = tuple(s["port_type"] for s in row["slots"])

    boards: list[dict[str, Any]] = []
    for d in sorted(p for p in frames_root.iterdir() if p.is_dir()):
        video_id, _, game = d.name.rpartition("__g")
        key = (video_id, int(game))
        meta_path = d / "meta.json"
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        row = corpus_rows.get(key)
        hexes = (row or {}).get("board", {}).get("hexes") or meta.get("board_hexes")
        robber = meta.get("board_desert_hex")
        if robber is None and hexes:
            robber = next((int(h["hex_id"]) for h in hexes if str(h["resource"]) == "DESERT"), None)
        if not hexes or robber is None:
            continue
        boards.append(
            {
                "video_id": video_id,
                "game_index": int(game),
                "hexes": sorted(hexes, key=lambda h: int(h["hex_id"])),
                "robber_hex": int(robber),
                "real_names": real_maps.get(key),
            }
        )
    return boards


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--frames-root", type=Path, default=DEFAULT_FRAMES)
    ap.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    ap.add_argument("--sidecar", type=Path, default=DEFAULT_OUT / "harvest.jsonl")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT / "invariance_probe.json")
    ap.add_argument("--ckpt", type=Path, default=OS.DEFAULT_CKPT)
    ap.add_argument("--max-boards", type=int, default=0, help="0 = all")
    ap.add_argument("--seats", type=int, nargs="+", default=[0, 1])
    args = ap.parse_args()

    out_path: Path = args.out.resolve()
    # Containment, not a string prefix (`data/human/ports_scratch` shares the prefix).
    if not out_path.is_relative_to((REPO_ROOT / "data/human/ports").resolve()):
        raise SystemExit(f"refusing to write outside data/human/ports: {out_path}")

    torch.set_num_threads(2)
    device = torch.device("cpu")
    policy = OS.load_policy(args.ckpt, device)

    boards = load_boards(args.corpus, args.frames_root, args.sidecar)
    if args.max_boards:
        boards = boards[: args.max_boards]
    if not boards:
        raise SystemExit("no probeable board found")

    # One entry per (board, seat) CELL -- the unit of independence (see _flip_stats).
    rvg: list[list[tuple[tuple[int, int], tuple[int, int]]]] = []
    gvg: list[list[tuple[tuple[int, int], tuple[int, int]]]] = []
    spreads: list[dict[str, Any]] = []
    per_board: list[dict[str, Any]] = []
    n_real = 0

    obs_delta: float | None = None
    for board in boards:
        env = PortProbeEnv(board["hexes"], board["robber_hex"])
        if obs_delta is None:
            obs_delta = assert_ports_reach_obs(env, board["video_id"], board["game_index"])
        for seat in args.seats:
            seed = 1000 * board["game_index"] + seat
            guesses: list[tuple[int, int]] = []
            metrics: list[dict[str, float | None]] = []
            for k in range(K_GUESSES):
                names = guessed_names(board["video_id"], board["game_index"], k)
                env.set_ports(P.build_port_assignment(names))
                pair, met = setup_decisions(env, policy, device, seed, seat)
                guesses.append(pair)
                metrics.append(met)
            real_pair: tuple[int, int] | None = None
            if board["real_names"] is not None:
                env.set_ports(P.build_port_assignment(tuple(board["real_names"])))
                real_pair, real_met = setup_decisions(env, policy, device, seed, seat)
                metrics.append(real_met)
                rvg.append([(real_pair, g) for g in guesses])
                n_real += 1
            gvg.append(
                [
                    (guesses[i], guesses[j])
                    for i in range(K_GUESSES)
                    for j in range(i + 1, K_GUESSES)
                ]
            )
            spreads.append(_metric_spread(metrics))
            per_board.append(
                {
                    "video_id": board["video_id"],
                    "game_index": board["game_index"],
                    "seat": seat,
                    "guessed_pairs": [list(g) for g in guesses],
                    "distinct_guessed_pairs": len(set(guesses)),
                    "real_pair": list(real_pair) if real_pair else None,
                }
            )

    metric_keys = sorted({k for s in spreads for k in s})
    spread_summary = {
        k: {
            "mean": round(float(np.mean([s[k] for s in spreads if k in s])), 4),
            "max": round(float(np.max([s[k] for s in spreads if k in s])), 4),
            "zero_spread_boards": sum(1 for s in spreads if s.get(k, 0.0) == 0.0),
        }
        for k in metric_keys
    }
    distinct = Counter(b["distinct_guessed_pairs"] for b in per_board)

    payload = {
        "checkpoint": str(Path(args.ckpt).resolve()),
        "checkpoint_sha256": hashlib.sha256(Path(args.ckpt).read_bytes()).hexdigest(),
        "checkpoint_caveat": (
            "This is opening_sweep.DEFAULT_CKPT, NOT a banked champion "
            "(runs/anchors/*.pt). Port sensitivity is exactly the kind of quantity that can "
            "move as the value function matures, so this flip rate is a proxy for the "
            "champion's, not a measurement of it."
        ),
        "invocation": {
            "argv": sys.argv,
            "frames_root": str(args.frames_root.resolve()),
            "corpus": str(args.corpus.resolve()),
            "sidecar": str(args.sidecar.resolve()),
            "seats": list(args.seats),
        },
        "boards": len(boards),
        "board_seat_cells": len(per_board),
        "cells_with_real_map": n_real,
        "K": K_GUESSES,
        "obs_l1_delta_between_two_port_maps": round(obs_delta or 0.0, 4),
        "real_vs_guessed": _flip_stats(rvg),
        "guessed_vs_guessed": _flip_stats(gvg),
        "distinct_guessed_pairs_histogram": {str(k): v for k, v in sorted(distinct.items())},
        "metric_spread": spread_summary,
        "per_board": per_board,
    }
    payload["reingest_recommendation"] = recommendation(payload)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({k: v for k, v in payload.items() if k != "per_board"}, indent=2))


#: Every recommendation carries this: the measured flip rate is the sensitivity of
#: the policy AND the heuristic opponent, because stepping the env drives the
#: opponent and the heuristic reads ports too.
CONFOUND = (
    " Caveat: the env's heuristic opponent also reads ports, so a flip rate measured this way "
    "bounds the policy's own port sensitivity from ABOVE rather than isolating it."
)


def recommendation(payload: dict[str, Any]) -> str:
    """A PLAIN STATEMENT of what was measured -- deliberately not a verdict.

    D7 says "no port-accuracy tolerance exists" and asks for the flip rate, the
    metric spread, and a plain statement. It pre-registers no decision threshold, so
    this function must not invent one: an earlier version compared the point estimate
    against a hardcoded 0.05 and printed "RE-INGEST IS JUSTIFIED", which the
    clustered interval on that same estimate does not support. The re-ingest is a
    ~1-day spend and the call is the owner's; what this returns is the evidence,
    stated so that it cannot be mistaken for a resolved decision.
    """
    g, r = payload["guessed_vs_guessed"], payload["real_vs_guessed"]
    gvg = g.get("realised_pair_flip_rate")
    rvg = r.get("realised_pair_flip_rate")
    if gvg is None:
        return "INCONCLUSIVE: no comparison ran."
    if gvg == 0.0 and rvg in (0.0, None):
        return (
            "DO NOT re-ingest. The policy's opening pair is IDENTICAL under every one of the "
            "8 guessed port maps on every board probed, so the true map cannot move it either. "
            "Exact ports would buy zero decision change at the opening; the ~100-video "
            "re-ingest is not funded by this evidence." + CONFOUND
        )
    if rvg is None:
        return (
            f"PARTIAL: the opening pair does move between guessed maps "
            f"(rate {gvg}), so ports are not inert. The mandated real-vs-guessed number "
            "needs the hand-labelling step; run harvest_ports.py --labels first."
        )
    lo, hi = r.get("realised_pair_flip_rate_ci95", [None, None])
    flipping = r.get("cells_with_any_flip")
    cells = r.get("cells")
    parts = [
        f"MEASURED, NOT DECIDED. Real-vs-guessed realised-pair flip rate {rvg} "
        f"(95% CI [{lo}, {hi}], {r['ci_method']}); {flipping} of {cells} board-seat cells flip "
        f"at all, so on {cells - flipping if cells and flipping is not None else '?'} cells the "
        "policy's opening is identical under every port map tried.",
        f"The label-free guessed-vs-guessed rate is {gvg} "
        f"(CI {g.get('realised_pair_flip_rate_ci95')}), statistically indistinguishable from the "
        "real-vs-guessed leg -- i.e. the hand-labelled REAL map behaves like just another "
        "guess, and it added no information to this decision.",
        "D7 pre-registers no threshold and this script deliberately asserts none: the ~100-video "
        "re-ingest is an owner call, and the honest input to it is that ports move the opening "
        "on a small minority of boards, with an interval wide enough that 'rarely' and 'often "
        "enough to matter' are both inside it.",
    ]
    return " ".join(parts) + CONFOUND


if __name__ == "__main__":
    main()
