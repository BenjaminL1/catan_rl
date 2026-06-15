"""Record a determinized-SEARCH agent vs the heuristic to a viewable replay.

The standard recorder drives the agent via ``select_action(obs, masks)``, but a
search agent needs the LIVE env (the obs is a lossy encoding). This wraps
``SearchAgent`` in a recorder actor that exposes ``bind_env`` — ``record_game``
hands it the env it drives, and the actor returns ``search.choose_action(env)``.
Everything else (broadcast-event -> ReplayStep reconstruction) is the standard
recorder, so the JSON loads in the normal ``catan-rl-replay`` viewer.

Usage:
  python scripts/record_search_replay.py            # v6 best, sims=50, seed=7
  python scripts/record_search_replay.py --sims 100 --seed 7
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_CKPT = "runs/train/selfplay_v6_20260611_065459/checkpoints/ckpt_000001499.pt"


class _SearchRecorderActor:
    """Recorder actor that drives the agent seat with determinized search."""

    kind = "policy"  # treated as the policy-kind driving actor by the recorder

    def __init__(self, search: Any) -> None:
        self._search = search
        self._env: Any | None = None

    def bind_env(self, env: Any) -> None:
        self._env = env

    def select_action(self, obs: dict[str, np.ndarray], masks: dict[str, np.ndarray]) -> np.ndarray:
        # obs/masks ignored — search needs the live env it was bound to.
        assert self._env is not None, "bind_env must be called before select_action"
        return self._search.choose_action(self._env)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--sims", type=int, default=50)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from catan_rl.replay import record_game, save_replay
    from catan_rl.replay.player_factory import PlayerSpec, _PolicyActor, build_actor
    from catan_rl.search.agent import SearchAgent
    from catan_rl.search.config import SearchConfig

    actor = build_actor(PlayerSpec(kind="policy", ckpt_path=args.ckpt), seed=0, device="cpu")
    assert isinstance(actor, _PolicyActor)
    search = SearchAgent(
        actor.policy, SearchConfig(sims_per_move=args.sims, seed=0), device=actor.device
    )
    search_actor = _SearchRecorderActor(search)

    spec_a = PlayerSpec(kind="policy", ckpt_path=args.ckpt)  # seat 0, overridden by search_actor
    spec_b = PlayerSpec(kind="heuristic")
    print(f"recording search@{args.sims} vs heuristic (seed={args.seed})...", flush=True)
    replay = record_game(spec_a, spec_b, seed=args.seed, agent_actor=search_actor, device="cpu")

    out = (
        Path(args.out)
        if args.out
        else Path(f"runs/replays/v6_search{args.sims}_vs_heuristic_seed{args.seed}.json")
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    save_replay(replay, out)
    print(f"wrote {out} ({len(replay.steps)} steps)")


if __name__ == "__main__":
    main()
