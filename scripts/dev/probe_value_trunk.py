"""US0 frozen-trunk value-head rank probe (spec 007, FR-000 / SC-000).

Standalone, READ-ONLY diagnostic — imports the v2 stack but mutates nothing on
disk except its own JSON output. Decides whether v8's value-head rank deficit
(aggregate Spearman 0.69 vs outcome; ``specs/003-inference-search/research.md:26``,
``src/catan_rl/search/value.py:11``) is *target-fixable* (a retarget of the head
moves rank) or a *trunk/representation limit* (it does not).

Method
------
1. Load v8 (``runs/anchors/v8_promobar_u243.pt``) via the recorder factory.
2. Play ``--n-games`` v8-vs-v8 RAW-policy games (no search), seat-symmetric,
   fixed seed; record per AGENT decision: the obs dict fed to the policy, the
   eventual game OUTCOME for the acting agent (win=1.0 / loss=0.0), and a PHASE
   label (early ``vp<=4`` / mid ``5<=vp<=9`` / late ``vp>=10``).
3. Freeze the v8 trunk; forward it once on every recorded obs -> 512-d embeddings
   (``torch.no_grad``).
4. Fit THREE value heads on the SAME frozen embeddings, split 80/20 BY GAME:
   * ``v8_original``  — v8's pretrained margin ``value_head`` (baseline + sanity
     anchor; overall held-out Spearman MUST come out ~0.69).
   * ``retrained_margin`` (control) — fresh ``ValueHead`` arch, MSE on the
     discounted MARGIN target ``z_disc = gamma^k * (+-1 + vp_diff/15)`` (the env
     terminal-reward shape, ``catan_env.py:1019``; BC ``L_value`` convention,
     ``bc/loss.py:87``).
   * ``retrained_winprob`` (treatment) — same arch + final sigmoid, BCE on the
     terminal win/loss target (1.0/0.0).
5. On the HELD-OUT split, per head: Spearman(output, outcome) OVERALL + per
   phase, raw Brier, fraction of raw output outside [0,1], bootstrap-CI on
   Spearman. Paired bootstrap on the winprob-vs-margin Spearman delta.
6. Verdict: TARGET-FIXABLE iff retrained_winprob OVERALL Spearman beats BOTH
   the retrained_margin control AND 0.69 by a bootstrap-CI-clean delta (paired
   delta lower bound > 0); else TRUNK-LIMITED. Print + write JSON.

Constraints: CPU only, fixed seeds everywhere, additive (new file), no GUI
import, ruff + ``mypy --strict`` clean.

Usage (smoke)::

    python scripts/dev/probe_value_trunk.py --n-games 8 --out runs/probe_smoke.json

Usage (full — run by the human, NOT here)::

    python scripts/dev/probe_value_trunk.py --n-games 300
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import torch
from scipy.stats import spearmanr
from torch import nn

if TYPE_CHECKING:
    from catan_rl.policy.network import CatanPolicy

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

V8_CKPT = "runs/anchors/v8_promobar_u243.pt"
RECORDED_SPEARMAN = 0.69  # aggregate baseline (search/value.py:11)
DEFAULT_GAMMA = 0.995  # ppo/arguments.py:197 (BC z_disc discount)
DEFAULT_VP_MARGIN_BONUS = 1.0 / 15.0  # catan_env.py:179
SANITY_TOL = 0.12  # |v8_original overall Spearman - 0.69| must be < this
N_BOOTSTRAP = 1000
TRAIN_STEPS = 600  # Adam steps for each retrained head (val-plateau early-stop)
PATIENCE = 60  # early-stop patience (steps) on val loss
TRUNK_DIM = 512

# Phase boundaries on the acting agent's total victoryPoints.
PHASE_EARLY_MAX = 4  # early = vp <= 4
PHASE_MID_MAX = 9  # mid = 5..9 ; late = vp >= 10
PHASES: tuple[str, ...] = ("early", "mid", "late")


# ---------------------------------------------------------------------------
# Recorded decision
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class _Decision:
    """One agent decision: the embedding inputs + targets + bookkeeping."""

    game_id: int
    phase: str
    vp: int
    outcome: float  # win=1.0 / loss=0.0 (terminal, agent-POV)
    margin_z: float  # discounted margin target (env shape), filled post-hoc
    obs: dict[str, np.ndarray]


def _phase_for_vp(vp: int) -> str:
    if vp <= PHASE_EARLY_MAX:
        return "early"
    if vp <= PHASE_MID_MAX:
        return "mid"
    return "late"


# ---------------------------------------------------------------------------
# Data generation — v8 vs v8 raw-policy games
# ---------------------------------------------------------------------------


def _play_one_game(
    env: Any,
    policy: CatanPolicy,
    *,
    device: torch.device,
    seed: int,
    agent_seat: int,
    game_id: int,
    gamma: float,
    vp_margin_bonus: float,
) -> list[_Decision]:
    """Play one v8(agent)-vs-v8(snapshot) RAW game; record every agent decision.

    Mirrors ``eval/harness.py:_play_one_game`` exactly (same loop, same safety
    cap) but records the obs/phase per agent decision and assigns the terminal
    outcome + discounted margin target afterward."""
    from catan_rl.policy.obs_tensor import masks_to_torch, obs_to_torch

    obs, _ = env.reset(seed=seed, options={"agent_seat": agent_seat})
    masks = env.get_action_masks()
    decisions: list[_Decision] = []
    terminated = False
    truncated = False
    n_env_steps = 0
    safety_cap = env.max_turns * 50
    hit_safety = False
    while not terminated and not truncated:
        obs_t = obs_to_torch(obs, device, add_batch=True)
        masks_t = masks_to_torch(masks, device, add_batch=True)
        with torch.no_grad():
            sample_out = policy.sample(obs_t, masks_t)
        action = sample_out["action"][0].cpu().numpy().astype(np.int64)
        # Record only non-forced decisions (>1 legal type) — forced moves carry
        # no value-ranking signal and would bias the phase mix toward roll/discard.
        if int(masks["type"].sum()) > 1:
            vp = int(getattr(env.agent_player, "victoryPoints", 0))
            decisions.append(
                _Decision(
                    game_id=game_id,
                    phase=_phase_for_vp(vp),
                    vp=vp,
                    outcome=0.0,  # filled below
                    margin_z=0.0,  # filled below
                    obs={k: np.asarray(v).copy() for k, v in obs.items()},
                )
            )
        obs, _, terminated, truncated, _ = env.step(action)
        masks = env.get_action_masks()
        n_env_steps += 1
        if n_env_steps > safety_cap:
            truncated = True
            hit_safety = True
            break

    assert env.game is not None
    agent_vp = int(getattr(env.agent_player, "victoryPoints", 0))
    opp_vp = int(getattr(env.opponent_player, "victoryPoints", 0))
    won = (not hit_safety) and agent_vp >= 15 and agent_vp > opp_vp
    lost = opp_vp >= 15
    outcome = 1.0 if won else 0.0
    # PRE-2026-07 margin reward shape: +-1 + vp_diff/15, truncation -> margin
    # only. This is the reward the v6/v7/v8-era checkpoints this probe grades
    # were TRAINED under — kept deliberately. It no longer mirrors the live
    # env: since audit 2026-07, CatanEnv._terminal_reward pays 0.0 on
    # truncation (margin is terminal-only). Probing a post-fix checkpoint
    # against this target would grade it on a stale shape — update the
    # truncation branch to 0.0 when that day comes.
    if won:
        margin_terminal = 1.0 + (agent_vp - opp_vp) * vp_margin_bonus
    elif lost:
        margin_terminal = -1.0 + (agent_vp - opp_vp) * vp_margin_bonus
    else:
        margin_terminal = (agent_vp - opp_vp) * vp_margin_bonus
    # z_disc: discount by decisions-to-end (single agent seat per game), matching
    # labeler.py:85 and BC's z_disc convention.
    for steps_to_term, d in enumerate(reversed(decisions)):
        d.outcome = outcome
        d.margin_z = (gamma**steps_to_term) * margin_terminal
    return decisions


def generate_decisions(
    *,
    n_games: int,
    seed: int,
    device: torch.device,
    gamma: float,
) -> tuple[list[_Decision], CatanPolicy]:
    """Play ``n_games`` seat-symmetric v8-vs-v8 raw games; return all decisions.

    Returns the loaded v8 policy too (its trunk + original value head are reused
    for the embeddings and the baseline)."""
    from catan_rl.env.catan_env import CatanEnv
    from catan_rl.replay.player_factory import PlayerSpec, _PolicyActor, build_actor
    from catan_rl.selfplay.snapshot_opponent import FrozenSnapshotOpponent

    actor = cast(
        _PolicyActor,
        build_actor(PlayerSpec(kind="policy", ckpt_path=V8_CKPT), seed=seed, device="cpu"),
    )
    policy = cast("CatanPolicy", actor.policy)
    vp_margin_bonus = float(
        getattr(CatanEnv(opponent_type="snapshot"), "vp_margin_bonus", DEFAULT_VP_MARGIN_BONUS)
    )

    opponent = FrozenSnapshotOpponent(policy, device=device, seed=seed)
    env = CatanEnv(opponent_type="snapshot")
    env.set_snapshot_opponent(opponent)

    decisions: list[_Decision] = []
    t0 = time.time()
    # Seat-symmetric pairing: each game_id alternates agent_seat, derived seeds
    # are deterministic functions of (seed, game_id) for reproducibility.
    for game_id in range(n_games):
        game_seed = (seed * 1_000_003 + game_id) % (2**31 - 1)
        agent_seat = game_id % 2
        opponent.reset_rng(seed=game_seed)
        game_decisions = _play_one_game(
            env,
            policy,
            device=device,
            seed=game_seed,
            agent_seat=agent_seat,
            game_id=game_id,
            gamma=gamma,
            vp_margin_bonus=vp_margin_bonus,
        )
        decisions.extend(game_decisions)
        print(
            f"[probe] game {game_id + 1}/{n_games}: {len(decisions)} decisions "
            f"({time.time() - t0:.0f}s)",
            flush=True,
        )
    env.close()
    return decisions, policy


# ---------------------------------------------------------------------------
# Embeddings — frozen trunk forward
# ---------------------------------------------------------------------------


_DISCRETE_OBS_KEYS = ("opponent_kind", "opponent_policy_id")


def _obs_batch_to_torch(
    obs_list: list[dict[str, np.ndarray]], device: torch.device
) -> dict[str, torch.Tensor]:
    """Stack a list of single-state obs dicts into a batched tensor dict.

    Casts discrete keys to int64 and continuous keys to float32 (the trunk's
    embedding lookups require long; the linear layers require float)."""
    keys = obs_list[0].keys()
    out: dict[str, torch.Tensor] = {}
    for k in keys:
        stacked = np.stack([o[k] for o in obs_list], axis=0)
        dtype = torch.int64 if k in _DISCRETE_OBS_KEYS else torch.float32
        out[k] = torch.as_tensor(stacked, dtype=dtype, device=device)
    return out


@torch.no_grad()
def compute_embeddings(
    policy: CatanPolicy,
    decisions: list[_Decision],
    *,
    device: torch.device,
    chunk: int = 256,
) -> torch.Tensor:
    """Forward the FROZEN v8 trunk on every recorded obs -> (N, 512)."""
    policy.eval()
    embs: list[torch.Tensor] = []
    for start in range(0, len(decisions), chunk):
        batch = decisions[start : start + chunk]
        obs_t = _obs_batch_to_torch([d.obs for d in batch], device)
        trunk = policy._encode(obs_t)  # frozen read-only trunk forward
        embs.append(trunk.detach().cpu())
    return torch.cat(embs, dim=0)


@torch.no_grad()
def v8_original_raw_value(policy: CatanPolicy, embeddings: torch.Tensor) -> torch.Tensor:
    """Apply v8's existing (pretrained margin) value head to frozen embeddings."""
    policy.eval()
    out: torch.Tensor = policy.value_head(embeddings).detach().cpu()
    return out


# ---------------------------------------------------------------------------
# Fresh value heads
# ---------------------------------------------------------------------------


def _make_value_head(*, sigmoid_out: bool) -> nn.Module:
    """A fresh 512->256->128->1 head (same arch as ``ValueHead``).

    ``sigmoid_out=True`` appends a final sigmoid (win-prob head); False leaves
    the raw scalar (margin control)."""
    layers: list[nn.Module] = [
        nn.Linear(TRUNK_DIM, 256),
        nn.LayerNorm(256),
        nn.GELU(),
        nn.Linear(256, 128),
        nn.LayerNorm(128),
        nn.GELU(),
        nn.Linear(128, 1),
    ]
    if sigmoid_out:
        layers.append(nn.Sigmoid())
    net = nn.Sequential(*layers)
    return net


def _train_head(
    *,
    head: nn.Module,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    loss_kind: str,
    seed: int,
) -> nn.Module:
    """Train a fresh head on frozen embeddings; early-stop on val-loss plateau.

    ``loss_kind`` is ``"mse"`` (margin control) or ``"bce"`` (win-prob). Returns
    the best (lowest-val-loss) head state."""
    torch.manual_seed(seed)
    opt = torch.optim.Adam(head.parameters(), lr=1e-3)
    loss_fn: Any = nn.MSELoss() if loss_kind == "mse" else nn.BCELoss()
    best_val = float("inf")
    best_state: dict[str, torch.Tensor] = {
        k: v.detach().clone() for k, v in head.state_dict().items()
    }
    since_improve = 0
    head.train()
    for _step in range(TRAIN_STEPS):
        opt.zero_grad()
        pred = head(x_train).squeeze(-1)
        loss = loss_fn(pred, y_train)
        loss.backward()
        opt.step()
        head.eval()
        with torch.no_grad():
            val_pred = head(x_val).squeeze(-1)
            val_loss = float(loss_fn(val_pred, y_val).item())
        head.train()
        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {k: v.detach().clone() for k, v in head.state_dict().items()}
            since_improve = 0
        else:
            since_improve += 1
            if since_improve >= PATIENCE:
                break
    head.load_state_dict(best_state)
    head.eval()
    return head


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def _spearman(pred: np.ndarray, outcome: np.ndarray) -> float:
    """Spearman rank-correlation; NaN-safe (degenerate / single-class -> 0.0)."""
    if pred.size < 2 or np.unique(outcome).size < 2 or np.unique(pred).size < 2:
        return 0.0
    rho = spearmanr(pred, outcome).statistic
    return 0.0 if (rho is None or np.isnan(rho)) else float(rho)


def _brier(prob: np.ndarray, outcome: np.ndarray) -> float:
    """Brier score = mean( (prob - outcome)^2 ); prob clipped to [0,1] only for
    this metric (the raw-output-out-of-range fraction is reported separately)."""
    return float(np.mean((np.clip(prob, 0.0, 1.0) - outcome) ** 2))


def _bootstrap_spearman_ci(
    pred: np.ndarray, outcome: np.ndarray, *, rng: np.random.Generator
) -> tuple[float, float]:
    """95% bootstrap CI on Spearman (>= N_BOOTSTRAP resamples)."""
    n = pred.size
    if n < 2:
        return (0.0, 0.0)
    samples = np.empty(N_BOOTSTRAP, dtype=np.float64)
    for i in range(N_BOOTSTRAP):
        idx = rng.integers(0, n, size=n)
        samples[i] = _spearman(pred[idx], outcome[idx])
    lo, hi = np.percentile(samples, [2.5, 97.5])
    return (float(lo), float(hi))


def _paired_delta_ci(
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    outcome: np.ndarray,
    *,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    """Bootstrap CI on the PAIRED Spearman delta (pred_a - pred_b) on the same
    resampled rows. Returns (point_delta, lo, hi)."""
    n = outcome.size
    point = _spearman(pred_a, outcome) - _spearman(pred_b, outcome)
    if n < 2:
        return (point, 0.0, 0.0)
    samples = np.empty(N_BOOTSTRAP, dtype=np.float64)
    for i in range(N_BOOTSTRAP):
        idx = rng.integers(0, n, size=n)
        oc = outcome[idx]
        samples[i] = _spearman(pred_a[idx], oc) - _spearman(pred_b[idx], oc)
    lo, hi = np.percentile(samples, [2.5, 97.5])
    return (point, float(lo), float(hi))


def _eval_head(
    pred: np.ndarray,
    outcome: np.ndarray,
    phases: np.ndarray,
    *,
    rng: np.random.Generator,
) -> dict[str, Any]:
    """Spearman (overall + per-phase + bootstrap CI), Brier, frac-outside-[0,1]."""
    overall_lo, overall_hi = _bootstrap_spearman_ci(pred, outcome, rng=rng)
    per_phase: dict[str, dict[str, Any]] = {}
    for ph in PHASES:
        mask = phases == ph
        per_phase[ph] = {
            "n": int(mask.sum()),
            "spearman": _spearman(pred[mask], outcome[mask]),
        }
    frac_outside = float(np.mean((pred < 0.0) | (pred > 1.0)))
    return {
        "spearman_overall": _spearman(pred, outcome),
        "spearman_overall_ci": [overall_lo, overall_hi],
        "spearman_per_phase": per_phase,
        "brier": _brier(pred, outcome),
        "frac_outside_unit": frac_outside,
        "n": int(pred.size),
    }


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_probe(*, n_games: int, seed: int, out_path: Path, gamma: float) -> dict[str, Any]:
    """Full probe: generate -> embed -> fit 3 heads -> measure -> verdict."""
    from catan_rl.search.value import squash_value

    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cpu")

    # --- 1. data gen ---
    decisions, policy = generate_decisions(n_games=n_games, seed=seed, device=device, gamma=gamma)
    if not decisions:
        raise RuntimeError("no decisions recorded — increase --n-games")
    n_total = len(decisions)
    game_ids = np.array([d.game_id for d in decisions])
    outcomes = np.array([d.outcome for d in decisions], dtype=np.float64)
    margin_z = np.array([d.margin_z for d in decisions], dtype=np.float64)
    phases = np.array([d.phase for d in decisions])

    phase_counts = {ph: int(np.sum(phases == ph)) for ph in PHASES}
    win_rate = float(np.mean(outcomes))
    print(
        f"[probe] {n_total} decisions over {n_games} games | phase counts {phase_counts} "
        f"| decision-level win rate {win_rate:.3f}",
        flush=True,
    )

    # --- 2. embeddings (frozen trunk) ---
    embeddings = compute_embeddings(policy, decisions, device=device)  # (N, 512)
    print(f"[probe] embeddings {tuple(embeddings.shape)} (frozen trunk)", flush=True)

    # --- 3. split 80/20 BY GAME (no state leakage across a game) ---
    rng_split = np.random.default_rng(seed)
    unique_games = np.unique(game_ids)
    rng_split.shuffle(unique_games)
    n_train_games = max(1, round(0.8 * unique_games.size))
    n_train_games = min(n_train_games, unique_games.size - 1) if unique_games.size > 1 else 1
    train_games = set(unique_games[:n_train_games].tolist())
    train_mask = np.array([gid in train_games for gid in game_ids])
    test_mask = ~train_mask
    if test_mask.sum() == 0 or train_mask.sum() == 0:
        raise RuntimeError(
            f"degenerate split: train={int(train_mask.sum())} test={int(test_mask.sum())}; "
            "need >= 2 games"
        )
    print(
        f"[probe] split: {n_train_games}/{unique_games.size} games train "
        f"({int(train_mask.sum())} states), {int(test_mask.sum())} test states",
        flush=True,
    )

    x_train = embeddings[torch.as_tensor(train_mask)]
    x_test = embeddings[torch.as_tensor(test_mask)]
    out_test = outcomes[test_mask]
    phases_test = phases[test_mask]

    # --- 4. fit the two retrained heads on TRAIN, measure on TEST ---
    # margin control (MSE on discounted margin target)
    margin_head = _make_value_head(sigmoid_out=False)
    margin_head = _train_head(
        head=margin_head,
        x_train=x_train,
        y_train=torch.as_tensor(margin_z[train_mask], dtype=torch.float32),
        x_val=x_test,
        y_val=torch.as_tensor(margin_z[test_mask], dtype=torch.float32),
        loss_kind="mse",
        seed=seed + 1,
    )
    # win-prob treatment (BCE on win/loss)
    winprob_head = _make_value_head(sigmoid_out=True)
    winprob_head = _train_head(
        head=winprob_head,
        x_train=x_train,
        y_train=torch.as_tensor(outcomes[train_mask], dtype=torch.float32),
        x_val=x_test,
        y_val=torch.as_tensor(out_test, dtype=torch.float32),
        loss_kind="bce",
        seed=seed + 2,
    )

    # --- 5. predictions on TEST ---
    with torch.no_grad():
        # v8 original: raw margin V (Spearman is rank-invariant under the monotone
        # squash; for Brier/frac-outside we report the SQUASHED [0,1] value, the
        # form search actually consumes).
        v8_raw_test = v8_original_raw_value(policy, x_test).numpy()
        v8_sq_test = cast(torch.Tensor, squash_value(torch.as_tensor(v8_raw_test))).numpy()
        margin_pred = margin_head(x_test).squeeze(-1).numpy()
        winprob_pred = winprob_head(x_test).squeeze(-1).numpy()

    rng_metrics = np.random.default_rng(seed + 7)
    heads: dict[str, dict[str, Any]] = {
        # v8_original measured on the squashed [0,1] output (its deployed form):
        # Spearman is identical raw-or-squashed, Brier/frac use the [0,1] form.
        "v8_original": _eval_head(v8_sq_test, out_test, phases_test, rng=rng_metrics),
        "retrained_margin": _eval_head(margin_pred, out_test, phases_test, rng=rng_metrics),
        "retrained_winprob": _eval_head(winprob_pred, out_test, phases_test, rng=rng_metrics),
    }
    # Attach the v8_original RAW (unsquashed) frac-outside as an extra field —
    # this is the headline "27% outside [-1,1]" diagnostic, distinct from [0,1].
    heads["v8_original"]["frac_outside_unit_raw_margin"] = float(
        np.mean((v8_raw_test < -1.0) | (v8_raw_test > 1.0))
    )

    # --- sanity anchor ---
    v8_overall = heads["v8_original"]["spearman_overall"]
    sanity_ok = abs(v8_overall - RECORDED_SPEARMAN) < SANITY_TOL
    if not sanity_ok:
        print(
            f"\n*** WARNING: v8_original overall Spearman {v8_overall:.3f} is NOT within "
            f"{SANITY_TOL} of the recorded {RECORDED_SPEARMAN} — the probe may be buggy "
            f"(small-n noise is expected at tiny --n-games). ***\n",
            file=sys.stderr,
            flush=True,
        )

    # --- 6. verdict: paired bootstrap on winprob - margin Spearman delta ---
    rng_delta = np.random.default_rng(seed + 11)
    delta_pt, delta_lo, delta_hi = _paired_delta_ci(
        winprob_pred, margin_pred, out_test, rng=rng_delta
    )
    winprob_overall = heads["retrained_winprob"]["spearman_overall"]
    winprob_lo = heads["retrained_winprob"]["spearman_overall_ci"][0]
    beats_margin_clean = delta_lo > 0.0
    beats_baseline_clean = winprob_lo > RECORDED_SPEARMAN
    target_fixable = beats_margin_clean and beats_baseline_clean
    verdict = "TARGET-FIXABLE" if target_fixable else "TRUNK-LIMITED"
    conclusion = (
        "win-prob beats the margin control by a CI-clean delta AND its CI lower "
        "bound clears 0.69 -> retarget the head (build US1)."
        if target_fixable
        else "win-prob does NOT beat both the margin control and 0.69 by a clean "
        "delta -> the rank deficit looks trunk/representation-limited; stop and "
        "escalate to a capacity/representation spec (do NOT build US1)."
    )

    result: dict[str, Any] = {
        "probe": "US0 frozen-trunk value-head rank probe (spec 007)",
        "ckpt": V8_CKPT,
        "config": {
            "n_games": n_games,
            "seed": seed,
            "gamma": gamma,
            "n_bootstrap": N_BOOTSTRAP,
            "train_steps": TRAIN_STEPS,
            "phase_boundaries": {
                "early_max_vp": PHASE_EARLY_MAX,
                "mid_max_vp": PHASE_MID_MAX,
            },
            "recorded_baseline_spearman": RECORDED_SPEARMAN,
        },
        "data": {
            "n_decisions": n_total,
            "phase_counts": phase_counts,
            "decision_win_rate": win_rate,
            "n_games_train": n_train_games,
            "n_games_total": int(unique_games.size),
            "n_states_train": int(train_mask.sum()),
            "n_states_test": int(test_mask.sum()),
        },
        "sanity_anchor": {
            "v8_original_overall_spearman": v8_overall,
            "recorded": RECORDED_SPEARMAN,
            "within_tol": sanity_ok,
            "tol": SANITY_TOL,
        },
        "heads": heads,
        "winprob_vs_margin_delta": {
            "point": delta_pt,
            "ci": [delta_lo, delta_hi],
            "beats_margin_ci_clean": beats_margin_clean,
        },
        "winprob_vs_baseline": {
            "winprob_overall": winprob_overall,
            "winprob_ci_lower": winprob_lo,
            "baseline": RECORDED_SPEARMAN,
            "beats_baseline_ci_clean": beats_baseline_clean,
        },
        "verdict": verdict,
        "conclusion": conclusion,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    _print_report(result)
    print(f"\n[probe] wrote {out_path}", flush=True)
    return result


def _print_report(result: dict[str, Any]) -> None:
    """Human-readable summary to stdout."""
    print("\n" + "=" * 72)
    print("US0 FROZEN-TRUNK VALUE-HEAD RANK PROBE — RESULT")
    print("=" * 72)
    data = result["data"]
    print(
        f"decisions={data['n_decisions']}  phase_counts={data['phase_counts']}  "
        f"win_rate={data['decision_win_rate']:.3f}"
    )
    san = result["sanity_anchor"]
    flag = "OK" if san["within_tol"] else "WARNING (off-anchor)"
    print(
        f"sanity: v8_original overall Spearman={san['v8_original_overall_spearman']:.3f} "
        f"vs recorded {san['recorded']}  [{flag}]"
    )
    print("-" * 72)
    hdr = (
        f"{'head':>20} {'overall':>9} {'CI':>17} {'early':>7} {'mid':>7} "
        f"{'late':>7} {'brier':>7} {'out%':>6}"
    )
    print(hdr)
    for name, h in result["heads"].items():
        ci = h["spearman_overall_ci"]
        pp = h["spearman_per_phase"]
        print(
            f"{name:>20} {h['spearman_overall']:>9.3f} "
            f"[{ci[0]:>6.3f},{ci[1]:>6.3f}] "
            f"{pp['early']['spearman']:>7.3f} {pp['mid']['spearman']:>7.3f} "
            f"{pp['late']['spearman']:>7.3f} {h['brier']:>7.3f} "
            f"{100 * h['frac_outside_unit']:>5.1f}%"
        )
    print("-" * 72)
    d = result["winprob_vs_margin_delta"]
    print(
        f"winprob - margin Spearman delta = {d['point']:+.3f} "
        f"CI[{d['ci'][0]:+.3f},{d['ci'][1]:+.3f}]  "
        f"(CI-clean beat of margin: {d['beats_margin_ci_clean']})"
    )
    b = result["winprob_vs_baseline"]
    print(
        f"winprob CI lower {b['winprob_ci_lower']:.3f} vs baseline {b['baseline']}  "
        f"(CI-clean beat of 0.69: {b['beats_baseline_ci_clean']})"
    )
    print("=" * 72)
    print(f"VERDICT: {result['verdict']}")
    print(result["conclusion"])
    print("=" * 72)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--n-games",
        type=int,
        default=300,
        help="number of v8-vs-v8 raw games to play (default 300; use 8 for smoke)",
    )
    parser.add_argument("--seed", type=int, default=0, help="master seed (default 0)")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("runs/probe_value_trunk.json"),
        help="output JSON path (default runs/probe_value_trunk.json)",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=DEFAULT_GAMMA,
        help=f"discount for the margin z_disc target (default {DEFAULT_GAMMA})",
    )
    args = parser.parse_args(argv)
    run_probe(n_games=args.n_games, seed=args.seed, out_path=args.out, gamma=args.gamma)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
