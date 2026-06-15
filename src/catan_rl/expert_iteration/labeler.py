"""Search-labeled data generation (contract C1).

Plays games where the AGENT is the 003 determinized ``SearchAgent`` and records,
at each non-forced agent decision, a BcDataset-compatible row:
``(obs, search_action, mask, belief_target, z_disc)``. The recorded ``action`` is
the *search's chosen 6-tuple* — the improved (expert) target the policy is distilled
toward. Output is NPZ shards + ``manifest.json`` read verbatim by ``catan_rl.bc``.

Reuses the BC shard FORMAT (``_DecisionRecord`` / ``_GameRecord`` / ``_flatten_records``)
so the existing BcDataset + bc_loss consume it unchanged; only the play loop is new
(search drives the env, unlike the BC generator which instruments the heuristic).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, cast

import numpy as np

from catan_rl.bc.dataset import _DecisionRecord, _flatten_records, _GameRecord
from catan_rl.expert_iteration.config import SearchLabelConfig
from catan_rl.policy.obs_encoder import hidden_belief_target


def _phase(env: Any) -> str:
    if env.initial_placement_phase:
        return "setup"
    if env.roll_pending:
        return "roll"
    if env.discard_pending:
        return "discard"
    if env.robber_placement_pending:
        return "robber"
    return "main"


def _play_one_label_game(
    env: Any, agent: Any, *, seed: int, agent_seat: int, game_id: int, discount: float
) -> _GameRecord:
    """Play one search-driven game; record every agent decision (agent-POV)."""
    env.reset(seed=seed, options={"agent_seat": agent_seat})
    decisions: list[_DecisionRecord] = []
    terminated = False
    truncated = False
    n_steps = 0
    safety_cap = env.max_turns * 50
    hit_safety = False
    while not terminated and not truncated:
        obs = env._get_obs()
        masks = env.get_action_masks()
        action = agent.choose_action(env)  # search picks the expert action (clones env internally)
        decisions.append(
            _DecisionRecord(
                obs=cast(dict[str, np.ndarray], obs),
                action=np.asarray(action, dtype=np.int64),
                mask=masks,
                belief_target=hidden_belief_target(env.opponent_player),
                forced=bool(int(masks["type"].sum()) <= 1),
                phase=_phase(env),
                player_seat=agent_seat,
                step_idx=len(decisions),
            )
        )
        _obs, _r, terminated, truncated, _info = env.step(action)
        n_steps += 1
        if n_steps > safety_cap:
            truncated = True
            hit_safety = True
            break

    agent_vp = int(env.agent_player.victoryPoints)
    opp_vp = int(env.opponent_player.victoryPoints)
    won = (not hit_safety) and agent_vp >= env.game.maxPoints and agent_vp > opp_vp
    lost = opp_vp >= env.game.maxPoints
    z = 1.0 if won else (-1.0 if lost else 0.0)  # agent-POV outcome; truncation = 0

    rec = _GameRecord(game_id=game_id, perturbation="search")
    rec.decisions = decisions
    rec.total_turns = n_steps
    rec.truncated = not (won or lost)
    # z_disc: agent decisions only (single seat per game) -> discount by decisions-to-end.
    for steps_to_term, d in enumerate(reversed(decisions)):
        d.z_disc = (discount**steps_to_term) * z
    return rec


def generate_search_labels(cfg: SearchLabelConfig) -> dict[str, Any]:
    """Generate search-labeled BcDataset shards; returns + writes the manifest."""
    from typing import cast as _cast

    from catan_rl.env.catan_env import CatanEnv
    from catan_rl.replay.player_factory import PlayerSpec, _PolicyActor, build_actor
    from catan_rl.search.agent import SearchAgent
    from catan_rl.search.config import SearchConfig
    from catan_rl.selfplay.snapshot_opponent import FrozenSnapshotOpponent

    search_actor = _cast(
        _PolicyActor,
        build_actor(
            PlayerSpec(kind="policy", ckpt_path=cfg.base_ckpt), seed=cfg.seed, device="cpu"
        ),
    )
    agent = SearchAgent(
        search_actor.policy,
        SearchConfig(sims_per_move=cfg.sims_per_move, seed=cfg.seed),
        device=search_actor.device,
    )

    if cfg.opponent.startswith("policy:"):
        opp_ckpt = cfg.opponent.split(":", 1)[1]
        opp_actor = _cast(
            _PolicyActor,
            build_actor(PlayerSpec(kind="policy", ckpt_path=opp_ckpt), seed=cfg.seed, device="cpu"),
        )
        opponent: FrozenSnapshotOpponent | None = FrozenSnapshotOpponent(
            opp_actor.policy, device=opp_actor.device, seed=cfg.seed
        )
        opponent_type = "snapshot"
    else:
        opponent = None
        opponent_type = cfg.opponent

    env = CatanEnv(opponent_type=opponent_type, max_turns=cfg.max_turns)
    if opponent is not None:
        env.set_snapshot_opponent(opponent)

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    games: list[_GameRecord] = []
    total_nonforced = 0
    game_id = 0
    t0 = time.time()
    # Continue until the position target is met AND there are >= min_games games
    # (a by-game val split needs >= 2 games to leave both sides non-empty).
    while total_nonforced < cfg.n_positions or game_id < cfg.min_games:
        game_seed = (cfg.seed * 1_000_003 + game_id) % (2**31 - 1)
        rec = _play_one_label_game(
            env,
            agent,
            seed=game_seed,
            agent_seat=game_id % 2,
            game_id=game_id,
            discount=cfg.discount,
        )
        games.append(rec)
        total_nonforced += sum(1 for d in rec.decisions if not d.forced)
        game_id += 1
        print(
            f"[exit.labeler] game {game_id}: {total_nonforced}/{cfg.n_positions} non-forced "
            f"positions ({time.time() - t0:.0f}s)",
            flush=True,
        )

    flat = _flatten_records(games, include_forced=False)
    shard_path = out_dir / "shard_0000.npz"
    cast(Any, np.savez_compressed)(shard_path, **flat)
    n_pairs = int(flat["action"].shape[0])
    manifest = {
        "shards": [
            {
                "shard": shard_path.name,
                "n_pairs": n_pairs,
                "n_games": len(games),
                # BcDataset.train_val_split splits by game_id -> needs the per-shard list.
                "game_ids": [g.game_id for g in games],
            }
        ],
        "n_pairs_total": n_pairs,
        "n_games": len(games),
        "source": "search-labeled (expert iteration)",
        "base_ckpt": cfg.base_ckpt,
        "sims_per_move": cfg.sims_per_move,
        "opponent": cfg.opponent,
        "discount": cfg.discount,
        "seed": cfg.seed,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[exit.labeler] wrote {n_pairs} positions across 1 shard -> {out_dir}", flush=True)
    return manifest
