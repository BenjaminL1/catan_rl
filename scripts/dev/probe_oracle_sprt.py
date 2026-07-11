"""Oracle root-headroom pre-check via the SPRT gate (spec 008 STAGE-A, FR-004).

THE KILL PROBE. Measures, in Elo and on the pentanomial SPRT gate, how much
win-probability the deployed PUCT-root decision rule leaves on the table AT THE
ROOT — the pre-registered gate on whether STAGE-B (Gumbel root) is worth
building. It complements ``probe_root_headroom.py`` (which measures the same
headroom as a per-move win-prob *regret*): here the same "near-perfect root
chooser" is played head-to-head against current PUCT-root through the SPRT so the
headroom lands as a single Elo number with a decision.

The oracle (documented "near-perfect" chooser)
-----------------------------------------------
``OracleRootAgent`` enumerates the SAME candidate root actions PUCT expands (the
modal sub-action per legal type) and scores each by ``rollouts`` Monte-Carlo
playouts to terminal under the frozen v8 net on both seats (the fresh-seeded
``StackedDice`` open-loop determinization used by ``probe_root_headroom``). It
picks ``argmax`` estimated win-prob.

**Resolution (why the argmax is trustworthy).** The quantity the argmax depends
on is the PAIRWISE difference ``q(a) − q(b)`` between candidate root moves, and
the kill-probe must resolve edges of ~2pp/move (a +15-Elo headroom is only
~2pp/move of decision quality). Scoring each candidate on INDEPENDENT rollout
seeds gives a per-candidate win-prob SE of ``sqrt(p(1−p)/n) ≈ ±10pp`` at n=24 —
coarser than the signal, so the argmax would be dominated by rollout noise (and
winner's-curse would inflate whichever candidate got lucky draws), making the
oracle a NOISY, not near-perfect, chooser. The fix is **common random numbers
(CRN)**: every candidate's k-th rollout shares ONE seed, so the shared dice bag
and opponent line cancel in the difference, collapsing its SE to
``sqrt(2·p(1−p)(1−ρ)/n)`` where ρ = corr(win_a, win_b) is high (~0.8–0.95) for
near-tied root moves in the same position. At ρ≈0.9, n=64 that is ~±2.8pp on the
difference — fine enough to resolve the ~2pp/move signal. So with CRN + a
rollout budget of ~64 the argmax is a faithful upper bound on root-decision
headroom: if even it cannot beat PUCT-root, no decision-rule change (Gumbel
included) will, and STAGE-B is NO-GO. The oracle is INTENTIONALLY given a strong
(rollout) evaluator; the matched-total-sim-budget invariant governs the *real*
search A/Bs (LCB, and the future Gumbel-vs-PUCT bake-off), not this ceiling
probe.

Verdict (pre-registered GO rule, FR-004 — DO NOT ALTER)
-------------------------------------------------------
    GO  iff  headroom_elo > +15  AND  depth-0 visit-collapse is high (>0.70)
             AND  root-child-value Spearman >= 0.60
    else NO-GO (record it; flag a chance-node/belief spec instead).

Writes ``runs/search/008a_verdict.json``.

Usage (smoke — proves the readout is well-formed)::

    python scripts/dev/probe_oracle_sprt.py --rollouts 4 --sims 12 \
        --max-pairs 3 --collapse-games 2 --out runs/search/008a_verdict_smoke.json

Usage (full — run by the human)::

    python scripts/dev/probe_oracle_sprt.py --rollouts 64 --sims 100 --max-pairs 400
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import torch

if TYPE_CHECKING:
    from catan_rl.env.catan_env import CatanEnv
    from catan_rl.policy.network import CatanPolicy
    from catan_rl.selfplay.snapshot_opponent import FrozenSnapshotOpponent

V8_CKPT = "runs/anchors/v8_promobar_u243.pt"

#: Pre-registered GO thresholds (FR-004).
HEADROOM_ELO_GO = 15.0
DEPTH0_COLLAPSE_HIGH = 0.70
ROOT_CHILD_SPEARMAN_GO = 0.60
_SEED_STRIDE = 1_000_003


# ---------------------------------------------------------------------------
# Oracle-root agent — rollout oracle at the root, over PUCT's candidate set
# ---------------------------------------------------------------------------


class OracleRootAgent:
    """A near-perfect root chooser: MC-rollout win-prob argmax over PUCT candidates.

    ``choose_action(env)`` runs the deployed PUCT search once to obtain the root
    candidate set (the modal sub-action per legal type it expands) and the
    depth-0 concentration, then re-scores each candidate by independent rollouts
    to terminal and returns the argmax. Exposes ``last_diagnostics`` so the SPRT
    driver + the collapse pass can read the same surface a ``SearchAgent`` does.
    """

    def __init__(
        self,
        policy: CatanPolicy,
        *,
        device: torch.device,
        sims: int,
        rollouts: int,
        seed: int,
    ) -> None:
        from catan_rl.search.agent import SearchAgent
        from catan_rl.search.config import SearchConfig
        from catan_rl.selfplay.snapshot_opponent import FrozenSnapshotOpponent

        self.policy = policy
        self.device = device
        self.rollouts = rollouts
        self.seed = seed
        self._opponent = FrozenSnapshotOpponent(policy, device=device, seed=seed)
        self._puct = SearchAgent(policy, SearchConfig(sims_per_move=sims, seed=seed), device=device)
        self._call = 0
        self.last_diagnostics: dict[str, Any] = {}

    def choose_action(self, env: CatanEnv) -> np.ndarray:
        # Uphold the same global-RNG snapshot/restore contract ``SearchAgent``
        # does (FR-006): the candidate rollouts below reseed torch/numpy/stdlib
        # and advance the opponent stream, so without this the oracle would
        # perturb the surrounding game's RNG and desync the common-random-numbers
        # the SPRT differential relies on. The inner PUCT call restores its own
        # state; this outer guard covers the rollout scoring.
        from catan_rl.search.agent import _restore_rng, _snapshot_rng

        rng = _snapshot_rng()
        try:
            return self._choose_action(env)
        finally:
            _restore_rng(rng)

    def _choose_action(self, env: CatanEnv) -> np.ndarray:
        from catan_rl.search.mcts import clone_env

        # PUCT once -> candidate set + depth-0 concentration.
        puct_action = self._puct.choose_action(env)
        diag = self._puct.last_diagnostics
        self.last_diagnostics = diag
        if diag.get("forced", False):
            return puct_action
        visit_counts: dict[tuple[int, ...], int] = diag.get("visit_counts", {})
        if not visit_counts:
            return puct_action
        candidates = list(visit_counts.keys())

        self._call += 1
        # COMMON RANDOM NUMBERS across candidates (the resolution fix): the k-th
        # rollout of EVERY candidate shares one seed, so the shared dice bag +
        # opponent line cancel in the pairwise difference q(a)-q(b) the argmax
        # depends on. This collapses the difference SE from ~±14pp (independent)
        # to ~±3pp at n=64, letting the oracle resolve the ~2pp/move edge it must
        # detect. The seed depends only on (position call, rollout index k), NOT
        # the candidate index, so it is identical across candidates.
        common_seeds = [
            (self.seed * 7919 + self._call * 104_729 + k * _SEED_STRIDE) & 0x7FFF_FFFF
            for k in range(self.rollouts)
        ]
        best_action = puct_action
        best_q = -1.0
        for action in candidates:
            base = clone_env(env, self._opponent)
            base.step(np.asarray(action, dtype=np.int64))
            wins = 0.0
            for seed_k in common_seeds:
                wins += _rollout_win(
                    base,
                    self.policy,
                    self._opponent,
                    device=self.device,
                    seed=seed_k,
                )
            base.close()
            q = wins / max(1, self.rollouts)
            if q > best_q:
                best_q = q
                best_action = np.asarray(action, dtype=np.int64)
        return best_action


def _rollout_win(
    start_env: CatanEnv,
    policy: CatanPolicy,
    opponent: FrozenSnapshotOpponent,
    *,
    device: torch.device,
    seed: int,
) -> float:
    """One independent playout to terminal under v8 on both seats; 1.0/0.0."""
    from catan_rl.engine.dice import StackedDice
    from catan_rl.policy.obs_tensor import masks_to_torch, obs_to_torch
    from catan_rl.search.mcts import clone_env
    from catan_rl.search.node import agent_outcome

    s32 = seed & 0x7FFF_FFFF
    env = clone_env(start_env, opponent)
    assert env.game is not None
    env.game.dice = StackedDice(seed=s32)
    np.random.seed(s32)
    random.seed(s32)
    opponent.reset_rng(seed=s32)

    obs = env._get_obs()
    masks = env.get_action_masks()
    terminated = truncated = False
    steps = 0
    cap = env.max_turns * 50
    while not terminated and not truncated:
        obs_t = obs_to_torch(obs, device, add_batch=True)
        masks_t = masks_to_torch(masks, device, add_batch=True)
        with torch.no_grad():
            out = policy.sample(obs_t, masks_t)
        action = out["action"][0].cpu().numpy().astype(np.int64)
        obs, _, terminated, truncated, _ = env.step(action)
        masks = env.get_action_masks()
        steps += 1
        if steps > cap:
            break
    win = agent_outcome(env)
    env.close()
    return win


# ---------------------------------------------------------------------------
# Depth-0 collapse + root-child Spearman (reuse the ndet collector)
# ---------------------------------------------------------------------------


def _collapse_and_spearman(
    policy: CatanPolicy, *, sims: int, n_games: int, seed: int, device: torch.device
) -> dict[str, Any]:
    """Deployed-PUCT depth-0 collapse + root-child-value Spearman on a game sample."""
    import sys

    # The n-det collector lives in the sibling script; make it importable whether
    # this file is run directly (its own dir is sys.path[0]) or imported.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from ndet_diagnostic import _play_collecting, _spearman

    from catan_rl.env.catan_env import CatanEnv
    from catan_rl.search.agent import SearchAgent
    from catan_rl.search.config import SearchConfig
    from catan_rl.selfplay.snapshot_opponent import FrozenSnapshotOpponent

    reference = FrozenSnapshotOpponent(policy, device=device, seed=seed)
    env = CatanEnv(opponent_type="snapshot")
    env.set_snapshot_opponent(reference)
    cfg = SearchConfig(sims_per_move=sims, seed=seed, collect_depth_stats=True)
    agent = SearchAgent(policy, cfg, device=device)

    depth0: list[float] = []
    q_series: list[float] = []
    win_series: list[float] = []
    base = (seed * _SEED_STRIDE + 11) % (2**31 - 1)
    for g in range(n_games):
        reference.reset_rng(seed=(base + g) % (2**31 - 1))
        decisions, won = _play_collecting(
            env, agent, seed=(base + g) % (2**31 - 1), agent_seat=g % 2
        )
        for d in decisions:
            m = d["pdc"].get(0)
            if m is not None and m["n_nodes"] > 0:
                depth0.append(m["collapse_frac"])
            q_series.append(d["q"])
            win_series.append(1.0 if won else 0.0)
    env.close()
    return {
        "n_decisions": len(q_series),
        "depth0_collapse_frac": (sum(depth0) / len(depth0)) if depth0 else 0.0,
        "root_child_value_spearman": _spearman(q_series, win_series),
    }


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run(
    *,
    sims: int,
    rollouts: int,
    max_pairs: int,
    collapse_games: int,
    seed: int,
    out_path: Path,
) -> dict[str, Any]:
    from catan_rl.replay.player_factory import PlayerSpec, _PolicyActor, build_actor
    from catan_rl.search.agent import SearchAgent
    from catan_rl.search.config import SearchConfig
    from catan_rl.search.sprt import (
        SPRTConfig,
        run_sprt_agents_vs_reference,
    )
    from catan_rl.selfplay.snapshot_opponent import FrozenSnapshotOpponent

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    device = torch.device("cpu")
    t0 = time.time()

    actor = cast(
        "_PolicyActor",
        build_actor(PlayerSpec(kind="policy", ckpt_path=V8_CKPT), seed=seed, device="cpu"),
    )
    policy = actor.policy

    # A vs B on the SPRT gate: A = oracle-root, B = deployed PUCT-root; both vs a
    # frozen v8 reference on common seeds (the differential pentanomial design).
    oracle = OracleRootAgent(policy, device=device, sims=sims, rollouts=rollouts, seed=seed)
    puct = SearchAgent(policy, SearchConfig(sims_per_move=sims, seed=seed), device=device)
    reference = FrozenSnapshotOpponent(policy, device=device, seed=seed)

    sprt_res = run_sprt_agents_vs_reference(
        oracle,
        puct,
        reference=reference,
        sprt_cfg=SPRTConfig(elo0=0.0, elo1=10.0, max_pairs=max_pairs),
        seed=seed,
        audit_rules=False,
        on_pair=lambda i, s, sp: print(
            f"[oracle-sprt] pair {i}: score={s} llr={sp.llr():+.3f} "
            f"elo_est={sp.elo_estimate():+.1f} ({time.time() - t0:.0f}s)",
            flush=True,
        ),
    )
    headroom_elo = sprt_res.elo_estimate

    collapse = _collapse_and_spearman(
        policy, sims=sims, n_games=collapse_games, seed=seed, device=device
    )
    depth0 = collapse["depth0_collapse_frac"]
    spearman = collapse["root_child_value_spearman"]

    go = (
        headroom_elo > HEADROOM_ELO_GO
        and depth0 > DEPTH0_COLLAPSE_HIGH
        and spearman >= ROOT_CHILD_SPEARMAN_GO
    )
    reasons = []
    if not headroom_elo > HEADROOM_ELO_GO:
        reasons.append(f"headroom_elo {headroom_elo:+.1f} <= {HEADROOM_ELO_GO}")
    if not depth0 > DEPTH0_COLLAPSE_HIGH:
        reasons.append(f"depth-0 collapse {depth0:.3f} <= {DEPTH0_COLLAPSE_HIGH}")
    if not spearman >= ROOT_CHILD_SPEARMAN_GO:
        reasons.append(f"root-child Spearman {spearman:+.3f} < {ROOT_CHILD_SPEARMAN_GO}")
    reason = (
        "all three pre-registered conditions met -> build STAGE-B (US1)"
        if go
        else "NO-GO: " + "; ".join(reasons) + " -> flag a chance-node/belief spec instead"
    )

    verdict = {
        "probe": "oracle root-headroom pre-check via SPRT (spec 008 STAGE-A, FR-004)",
        "ckpt": V8_CKPT,
        "configs": {
            "agent_a": "oracle-root (rollout-argmax over PUCT candidates)",
            "agent_b": "deployed PUCT-root",
            "reference": "frozen v8 (raw, no search)",
            "sims_per_move": sims,
            "rollouts_per_candidate": rollouts,
            "rollout_pairing": "common-random-numbers (shared per-rollout seed across candidates)",
            "sprt": {
                "elo0": 0.0,
                "elo1": 10.0,
                "alpha": 0.05,
                "beta": 0.05,
                "max_pairs": max_pairs,
            },
        },
        "sprt_results": {
            "decision": sprt_res.decision,
            "llr": sprt_res.llr,
            "n_pairs": sprt_res.n_pairs,
            "n_games": sprt_res.n_games,
            "pentanomial_counts": sprt_res.counts,
            "elo_estimate": sprt_res.elo_estimate,
        },
        "headroom_elo": headroom_elo,
        "per_depth_collapse": {"depth0": depth0, "n_decisions": collapse["n_decisions"]},
        "root_child_spearman": spearman,
        "go_rule": {
            "headroom_elo_go": HEADROOM_ELO_GO,
            "depth0_collapse_high": DEPTH0_COLLAPSE_HIGH,
            "root_child_spearman_go": ROOT_CHILD_SPEARMAN_GO,
        },
        "go": go,
        "reason": reason,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(verdict, indent=2))
    print("\n" + "=" * 80)
    print("008a VERDICT (oracle root-headroom pre-check)")
    print(
        f"  headroom_elo={headroom_elo:+.1f}  depth0_collapse={depth0:.3f}  "
        f"root_child_spearman={spearman:+.3f}"
    )
    print(f"  SPRT: {sprt_res.decision} (llr={sprt_res.llr:+.3f}, {sprt_res.n_games} games)")
    print(f"  GO={go} — {reason}")
    print(f"  wrote {out_path}")
    print("=" * 80)
    return verdict


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sims", type=int, default=100, help="PUCT sim budget/move")
    parser.add_argument(
        "--rollouts",
        type=int,
        default=64,
        help="MC rollouts/candidate (oracle); CRN-paired across candidates for ~±3pp "
        "difference resolution at 64 (default 64)",
    )
    parser.add_argument("--max-pairs", type=int, default=400, help="SPRT pair cap")
    parser.add_argument("--collapse-games", type=int, default=50, help="games: collapse/Spearman")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("runs/search/008a_verdict.json"))
    args = parser.parse_args(argv)
    run(
        sims=args.sims,
        rollouts=args.rollouts,
        max_pairs=args.max_pairs,
        collapse_games=args.collapse_games,
        seed=args.seed,
        out_path=args.out,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
