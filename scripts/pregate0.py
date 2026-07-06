#!/usr/bin/env python3
"""PRE-GATE-0 + M0 runner (step6 §2.2, PRE-CORPUS lane).

Plays ``n`` **natural** (no search) champion self-play games in two matchups and
records, per game, everything PRE-GATE-0 and its in-distribution M0 anchor need:

* both openings (engine vertex IDs) of the post-setup state,
* the archetype bucket of the to-move (champion) seat's opening
  (:mod:`catan_rl.human_data.opening_archetypes`, frozen v5.2 spec),
* the **post-setup ``v̂`` of the to-move seat** — the squashed win-probability
  from the champion's value head on the live env's true-port obs (M0's estimator),
* the draft position of the to-move seat, and
* the game outcome (did the to-move seat win).

Two matchups:

* ``v8_v8`` — champion (``runs/anchors/v8_promobar_u243.pt``) vs a frozen copy of
  itself. This is "v8's own eval games"; **M0 (AUC + M2-analog partial Spearman)
  is computed on this subset only** (plan §2.2: M0's permutation strata =
  ``draft_position`` only, since ``opponent_strength.source`` does not exist for
  self-play games).
* ``v8_anchor`` — champion vs a frozen **prior anchor** checkpoint (default
  ``runs/anchors/v7_final_u399.pt``, the prior champion; the league anchor is the
  same checkpoint class). **The committed per-bucket mass table + the collapse
  verdict are computed from this subset only** (plan §2.2: all downstream
  "PRE-GATE-0 mass" quantities read the ``v8-vs-anchor`` subset).

  Only one *current* champion checkpoint exists, so the anchor is a distinct
  frozen prior anchor on disk (documented in the report). If no prior anchor is
  available the runner falls back to a second frozen copy of v8 and says so.

Outputs (default under ``data/human/``):

* ``pregate0_games.jsonl`` — one JSON record per game (resumable append log).
* ``pregate0_mass.json`` — the committed per-bucket mass table (v8-vs-anchor).
* ``pregate0_report.md`` — the archetype histogram, Shannon entropy,
  ``openings/setup_head_entropy`` (mean setup-decision policy entropy), the
  COLLAPSE VERDICT (≥70% one-bucket mass), and M0 = AUC + M2-analog partial
  Spearman with permutation strata ``draft_position`` only.

**Resumable:** each game is atomically appended (flush + fsync); a re-run skips
game IDs already present, so a kill + restart appends without duplicates. The
mass table + report are recomputed from the full JSONL on every run (idempotent).

**Deterministic:** per-game seeds derive from a SHA-256 of ``(seed, matchup,
game_index)`` (never Python ``hash()`` — PYTHONHASHSEED-salted); the champion's
torch sampling stream is seeded per game. CPU-only (rule 7: eval on CPU).

This is the PRE-CORPUS runner; the long ``n>=400`` run is launched detached. Use
``--smoke`` for an end-to-end n=3/matchup check.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import torch

    from catan_rl.human_data.topology import Topology

REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_V8_CKPT = "runs/anchors/v8_promobar_u243.pt"
DEFAULT_ANCHOR_CKPT = "runs/anchors/v7_final_u399.pt"
DEFAULT_OUT_DIR = "data/human"

#: Collapse verdict threshold (plan §2.2): ≥70% of the v8-vs-anchor mass in one
#: bucket ⇒ COLLAPSED.
COLLAPSE_THRESHOLD = 0.70

#: Names of the matchups written to each JSONL record's ``matchup`` field.
MATCHUP_V8V8 = "v8_v8"
MATCHUP_V8ANCHOR = "v8_anchor"

#: Code artifacts whose SHA-256 the runner records in the mass table (plan §1:
#: "all hashes recorded by the CLI"). Paths are repo-relative.
_FREEZE_CODE_FILES = (
    "docs/plans/v2/step6_human_corpus.md",
    "src/catan_rl/human_data/opening_archetypes.py",
    "src/catan_rl/human_data/engine_bridge.py",
    "scripts/pregate0.py",
)


# ---------------------------------------------------------------------------
# Determinism + IO helpers
# ---------------------------------------------------------------------------
def game_seed(base_seed: int, matchup: str, game_index: int) -> int:
    """Deterministic per-game seed (SHA-256, never PYTHONHASHSEED-salted ``hash``)."""
    digest = hashlib.sha256(f"{base_seed}:{matchup}:{game_index}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % (2**31 - 1)


def _sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _append_line(path: Path, line: str) -> None:
    """Atomically append one JSONL line (flush + fsync so a kill can't tear it).

    Mirrors ``human_data.batch._append_line``: if the file does not end in a
    newline (a torn tail from a hard kill mid-append), emit a leading newline
    first so the torn fragment stays a standalone skippable partial line instead
    of welding onto the new record.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab+") as fh:
        if fh.seek(0, os.SEEK_END) > 0:
            fh.seek(-1, os.SEEK_END)
            if fh.read(1) != b"\n":
                fh.write(b"\n")
        fh.write((line + "\n").encode("utf-8"))
        fh.flush()
        os.fsync(fh.fileno())


def _atomic_write(path: Path, data: str) -> None:
    """Write ``data`` to ``path`` via a temp file + ``os.replace`` (atomic)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def load_records(jsonl_path: Path) -> list[dict[str, Any]]:
    """Parse the JSONL log, tolerating a torn partial line (skipped)."""
    records: list[dict[str, Any]] = []
    if not jsonl_path.exists():
        return records
    for raw in jsonl_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            # Torn tail from a hard kill mid-append — skip it (a re-run replays
            # the missing game, since its game_id is absent from the parsed set).
            continue
    return records


def _done_game_ids(jsonl_path: Path) -> set[str]:
    return {str(r["game_id"]) for r in load_records(jsonl_path) if "game_id" in r}


# ---------------------------------------------------------------------------
# Statistics (M0)
# ---------------------------------------------------------------------------
def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Rank-based ROC AUC (Mann-Whitney U). ``nan`` if a class is empty."""
    from scipy.stats import rankdata

    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores, dtype=float)
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(scores)
    sum_pos = float(ranks[labels == 1].sum())
    return (sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    if np.std(a) == 0.0 or np.std(b) == 0.0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def partial_spearman(r: np.ndarray, x: np.ndarray, z: np.ndarray) -> float:
    """First-order partial Spearman of ``r`` on ``x`` controlling ``z``.

    Rank-transforms all three, then the standard partial-correlation formula on
    the ranks: ``(ρ_rx − ρ_rz·ρ_xz) / sqrt((1−ρ_rz²)(1−ρ_xz²))``. ``nan`` when a
    rank vector is constant or the denominator vanishes.
    """
    from scipy.stats import rankdata

    if len(r) < 3:
        return float("nan")
    rr = rankdata(r)
    rx = rankdata(x)
    rz = rankdata(z)
    rho_rx = _pearson(rr, rx)
    rho_rz = _pearson(rr, rz)
    rho_xz = _pearson(rx, rz)
    if any(np.isnan(v) for v in (rho_rx, rho_rz, rho_xz)):
        # z (port adjacency) constant ⇒ no partialling needed; fall back to the
        # plain rank correlation of r vs x.
        return rho_rx
    denom = float(np.sqrt((1.0 - rho_rz**2) * (1.0 - rho_xz**2)))
    if denom == 0.0:
        return float("nan")
    return (rho_rx - rho_rz * rho_xz) / denom


def partial_spearman_perm_p(
    r: np.ndarray,
    x: np.ndarray,
    z: np.ndarray,
    strata: Sequence[Any],
    *,
    perms: int,
    seed: int,
) -> tuple[float, float]:
    """One-sided (positive-sign) permutation p-value for :func:`partial_spearman`.

    ``r`` is permuted **within each stratum** (M0 strata = ``draft_position``
    only), recomputing the partial statistic each draw. Returns ``(observed,
    p_value)``; ``p`` is ``nan`` when the observed statistic is undefined.
    """
    observed = partial_spearman(r, x, z)
    if np.isnan(observed) or perms <= 0:
        return observed, float("nan")
    rng = np.random.default_rng(seed)
    strata_arr = np.asarray(list(strata))
    groups = [np.flatnonzero(strata_arr == s) for s in np.unique(strata_arr)]
    ge = 0
    valid = 0
    for _ in range(perms):
        r_perm = r.copy()
        for idx in groups:
            r_perm[idx] = r[idx][rng.permutation(len(idx))]
        stat = partial_spearman(r_perm, x, z)
        if np.isnan(stat):
            continue
        valid += 1
        if stat >= observed:
            ge += 1
    p = (ge + 1) / (valid + 1) if valid > 0 else float("nan")
    return observed, p


# ---------------------------------------------------------------------------
# Playing one natural game
# ---------------------------------------------------------------------------
def _capture_post_setup(
    env: Any,
    obs: dict[str, np.ndarray],
    agent_policy: Any,
    device: torch.device,
    topology: Topology,
) -> dict[str, Any]:
    """Serialize BOTH seats' openings + the to-move seat's ``v̂`` at post-setup.

    The agent is seated at seat 0 (``agent_seat=0``), so it is the to-move
    (first-drafter, first-roller) seat and the env hands back a **clean** post-setup
    state (no main turn has run for either seat). ``value_from_obs`` on the agent-POV
    obs is therefore the to-move seat's squashed win-probability — M0's estimator.
    """
    from catan_rl.human_data.engine_bridge import serialize_post_setup
    from catan_rl.human_data.opening_archetypes import featurize_opening
    from catan_rl.search.value import value_from_obs

    state = serialize_post_setup(env)
    v_hat_to_move = value_from_obs(agent_policy, obs, device=device)
    hexes = [dict(h) for h in state.hexes]
    seats: dict[str, Any] = {}
    for seat in (0, 1):
        settlements = [int(v) for v in state.placements[seat].settlements]
        feats = featurize_opening(settlements, hexes, topology)
        # RESOURCE_ORDER_CW = (WOOD, BRICK, WHEAT, ORE, SHEEP): idx 2 = WHEAT, 3 = ORE.
        seats[str(seat)] = {
            "settlements": settlements,
            "archetype": str(feats.archetype),
            "pip_share": [float(p) for p in feats.pip_share],
            "ore_wheat_share": float(feats.pip_share[2] + feats.pip_share[3]),
            "port_adjacent": bool(feats.port_adjacent),
            "total_pips": int(feats.total_pips),
            "draft_position": seat,
        }
    return {"v_hat_to_move": float(v_hat_to_move), "seats": seats}


def play_game(
    *,
    agent_policy: Any,
    opponent: Any,
    device: torch.device,
    seed: int,
    max_turns: int,
    topology: Topology,
    matchup: str,
    game_index: int,
) -> dict[str, Any]:
    """Play one natural game (champion at seat 0); return its PRE-GATE-0 record.

    The champion drives seat 0 (``policy.sample``, entropy captured during setup);
    ``opponent`` (a frozen snapshot) drives seat 1 inside the env. Seat 0 is the
    to-move (first-drafter) seat, so the first ``roll_pending`` the agent observes
    is the CLEAN post-setup state — both seats' openings + the to-move ``v̂`` are
    captured there. The record stores pure per-game facts (both openings, both
    archetypes, per-seat win flags, the to-move ``v̂`` and outcome); M0's focal
    orientation + strata are derived downstream in :func:`compute_m0`.
    """
    import torch

    from catan_rl.env.catan_env import CatanEnv
    from catan_rl.policy.obs_tensor import masks_to_torch, obs_to_torch

    env = CatanEnv(opponent_type="snapshot", max_turns=max_turns)
    env.set_snapshot_opponent(opponent)
    try:
        torch.manual_seed(seed % (2**31 - 1))
        obs, _ = env.reset(seed=seed, options={"agent_seat": 0})
        masks = env.get_action_masks()
        setup_entropies: list[float] = []
        post: dict[str, Any] | None = None
        n_steps = 0
        terminated = truncated = False
        safety_cap = max_turns * 50
        while not terminated and not truncated:
            if post is None and (not env.initial_placement_phase) and env.roll_pending:
                post = _capture_post_setup(env, obs, agent_policy, device, topology)
            obs_t = obs_to_torch(obs, device, add_batch=True)
            masks_t = masks_to_torch(masks, device, add_batch=True)
            with torch.no_grad():
                out = agent_policy.sample(obs_t, masks_t)
            if env.initial_placement_phase:
                setup_entropies.append(float(out["entropy"][0].item()))
            action = out["action"][0].cpu().numpy().astype(np.int64)
            obs, _, terminated, truncated, _ = env.step(action)
            masks = env.get_action_masks()
            n_steps += 1
            if n_steps > safety_cap:
                truncated = True
                break

        if post is None:
            raise RuntimeError(
                f"{matchup}:{game_index} ended before a post-setup roll-pending state"
            )
        agent = env.agent_player
        opp = env.opponent_player
        seat0_vp = int(getattr(agent, "victoryPoints", 0))
        seat1_vp = int(getattr(opp, "victoryPoints", 0))
        seat0_won = seat0_vp >= 15 and seat0_vp > seat1_vp
        seat1_won = seat1_vp >= 15 and seat1_vp > seat0_vp
        seats = post["seats"]
        seats["0"]["won"] = int(bool(seat0_won))
        seats["1"]["won"] = int(bool(seat1_won))
        setup_mean = float(np.mean(setup_entropies)) if setup_entropies else None
        record: dict[str, Any] = {
            "game_id": f"{matchup}:{game_index}",
            "matchup": matchup,
            "game_index": int(game_index),
            "seed": int(seed),
            "to_move_seat": 0,
            "v_hat_to_move": post["v_hat_to_move"],
            "outcome_to_move": int(bool(seat0_won)),
            "seats": seats,
            "draft_positions": {"0": 0, "1": 1},
            "final_vp": {"0": seat0_vp, "1": seat1_vp},
            "n_turns": int(n_steps),
            "truncated": bool(truncated),
            "setup_entropy_mean": setup_mean,
            "setup_entropies": setup_entropies,
        }
        return record
    finally:
        env.close()


# ---------------------------------------------------------------------------
# Aggregation: mass table + M0 + report
# ---------------------------------------------------------------------------
def build_mass_table(
    records: Sequence[dict[str, Any]],
    *,
    v8_ckpt: str,
    anchor_ckpt: str,
    threshold: float = COLLAPSE_THRESHOLD,
    freeze: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """The committed per-bucket mass table from the **v8-vs-anchor subset only**."""
    from catan_rl.human_data.opening_archetypes import (
        OpeningArchetype,
        archetype_entropy,
        archetype_histogram,
    )

    subset = [r for r in records if r.get("matchup") == MATCHUP_V8ANCHOR]
    # v8 is seated at seat 0 (agent) in every game, so seat 0's opening is v8's.
    arch = [OpeningArchetype(str(r["seats"]["0"]["archetype"])) for r in subset]
    hist = archetype_histogram(arch)
    n = len(subset)
    counts = {b.value: int(hist[b]) for b in OpeningArchetype}
    mass = {b.value: (counts[b.value] / n if n else 0.0) for b in OpeningArchetype}
    max_bucket = max(counts, key=lambda b: counts[b]) if n else None
    max_mass = mass[max_bucket] if max_bucket is not None else 0.0
    return {
        "schema": "pregate0_mass_v1",
        "source": "v8-vs-anchor",
        "generated_by": "scripts/pregate0.py",
        "v8_ckpt": v8_ckpt,
        "anchor_ckpt": anchor_ckpt,
        "n_games": n,
        "counts": counts,
        "mass": mass,
        "entropy_bits": archetype_entropy(hist),
        "max_bucket": max_bucket,
        "max_mass": max_mass,
        "collapse_threshold": threshold,
        "collapse_verdict": ("COLLAPSED" if (n > 0 and max_mass >= threshold) else "NO_COLLAPSE"),
        "freeze": freeze or {},
    }


def compute_setup_head_entropy(records: Sequence[dict[str, Any]]) -> dict[str, float | int]:
    """Mean setup-decision policy entropy (``openings/setup_head_entropy``)."""
    all_ent: list[float] = []
    for r in records:
        all_ent.extend(float(e) for e in r.get("setup_entropies", []))
    return {
        "n_decisions": len(all_ent),
        "setup_head_entropy": (float(np.mean(all_ent)) if all_ent else float("nan")),
    }


def compute_m0(
    records: Sequence[dict[str, Any]],
    *,
    perms: int,
    seed: int,
) -> dict[str, Any]:
    """M0 (plan §2.2): AUC + M2-analog partial Spearman on the **v8-vs-v8 subset**.

    * **AUC (M1 statistic):** ranks the to-move seat's post-setup ``v̂`` against
      the to-move seat's outcome (one observation per game, seat 0 = v8).
    * **M2-analog partial Spearman:** to obtain ``draft_position`` variation from
      self-play games (where the to-move seat is always the first drafter), a
      **focal seat** is chosen per game by ``game_index`` parity. The focal seat's
      win-probability is the plan's §4 probability-space seat complement —
      ``p̂ = v̂`` if the focal seat is the to-move seat (draft position 0), else
      ``p̂ = 1 − v̂`` — and the residual ``win_focal − p̂`` is correlated with the
      focal seat's ORE+WHEAT pip-share, partialling port-slot adjacency.
    * **Strata = ``draft_position`` ONLY** (self-play has no
      ``opponent_strength.source``); permutation is within-stratum.
    """
    subset = [r for r in records if r.get("matchup") == MATCHUP_V8V8]
    n = len(subset)
    v_to_move = np.array([float(r["v_hat_to_move"]) for r in subset], dtype=float)
    outcome_to_move = np.array([int(r["outcome_to_move"]) for r in subset], dtype=float)
    auc_stat = auc(v_to_move, outcome_to_move) if n else float("nan")

    residual = np.empty(n, dtype=float)
    ore_wheat = np.empty(n, dtype=float)
    port = np.empty(n, dtype=float)
    draft: list[int] = []
    for i, r in enumerate(subset):
        focal = int(r["game_index"]) % 2  # alternate focal seat → draft variation
        v = float(r["v_hat_to_move"])
        phat = v if focal == 0 else 1.0 - v  # §4 probability-space seat complement
        seat = r["seats"][str(focal)]
        residual[i] = float(seat["won"]) - phat
        ore_wheat[i] = float(seat["ore_wheat_share"])
        port[i] = 1.0 if seat["port_adjacent"] else 0.0
        draft.append(focal)
    observed, p_value = partial_spearman_perm_p(
        residual, ore_wheat, port, draft, perms=perms, seed=seed
    )
    return {
        "subset": "v8-vs-v8",
        "n_games": n,
        "auc": auc_stat,
        "partial_spearman": observed,
        "p_value": p_value,
        "permutations": int(perms),
        "strata": ["draft_position"],
        "predicted_sign": "positive",
    }


def _fmt(x: float | None) -> str:
    if x is None:
        return "n/a"
    if isinstance(x, float) and np.isnan(x):
        return "nan (undefined — underpowered/degenerate at this n)"
    return f"{x:.4f}"


def render_report(
    records: Sequence[dict[str, Any]],
    mass: dict[str, Any],
    *,
    perms: int,
    seed: int,
    v8_ckpt: str,
    anchor_ckpt: str,
    anchor_is_v8_copy: bool,
) -> str:
    """Render the PRE-GATE-0 markdown report (deterministic — no wall clock)."""
    from catan_rl.human_data.opening_archetypes import OpeningArchetype

    n_v8v8 = sum(1 for r in records if r.get("matchup") == MATCHUP_V8V8)
    n_anchor = sum(1 for r in records if r.get("matchup") == MATCHUP_V8ANCHOR)
    setup_all = compute_setup_head_entropy(records)
    setup_v8v8 = compute_setup_head_entropy(
        [r for r in records if r.get("matchup") == MATCHUP_V8V8]
    )
    setup_anchor = compute_setup_head_entropy(
        [r for r in records if r.get("matchup") == MATCHUP_V8ANCHOR]
    )
    m0 = compute_m0(records, perms=perms, seed=seed)

    lines: list[str] = []
    lines.append("# PRE-GATE-0 report (step6 §2.2)")
    lines.append("")
    lines.append("Generated by `scripts/pregate0.py` (PRE-CORPUS lane). Natural (no-search)")
    lines.append("champion self-play; CPU eval. This report is recomputed from the full")
    lines.append("`pregate0_games.jsonl` on every run.")
    lines.append("")
    lines.append("## Conditions")
    lines.append("")
    lines.append(f"- champion (v8): `{v8_ckpt}`")
    if anchor_is_v8_copy:
        lines.append(
            f"- anchor: `{anchor_ckpt}` — **no distinct prior anchor found on disk; "
            "using a frozen copy of v8** (documented fallback, plan §2.2)."
        )
    else:
        lines.append(
            f"- anchor (frozen prior anchor / prior champion): `{anchor_ckpt}` — the league "
            "anchor is the same checkpoint class; only one current champion exists, so a "
            "distinct prior anchor is used for the mass subset."
        )
    lines.append(f"- games: {n_v8v8} x `v8_v8`, {n_anchor} x `v8_anchor`; seed={seed}")
    lines.append("")
    lines.append("## Archetype histogram — v8-vs-anchor subset (mass provenance)")
    lines.append("")
    lines.append("| bucket | count | mass |")
    lines.append("|---|---:|---:|")
    for b in OpeningArchetype:
        lines.append(f"| {b.value} | {mass['counts'][b.value]} | {mass['mass'][b.value]:.4f} |")
    lines.append("")
    lines.append(
        f"- Shannon entropy (bits): {_fmt(mass['entropy_bits'])} (uniform ceiling ~= 2.3219)"
    )
    lines.append(f"- max bucket: {mass['max_bucket']} @ mass {_fmt(mass['max_mass'])}")
    lines.append("")
    lines.append("## COLLAPSE VERDICT")
    lines.append("")
    lines.append(
        f"**{mass['collapse_verdict']}** — threshold ≥{mass['collapse_threshold']:.0%} "
        "one-bucket mass (v8-vs-anchor subset)."
    )
    lines.append("")
    lines.append("## openings/setup_head_entropy (mean setup-decision policy entropy)")
    lines.append("")
    lines.append(
        f"- overall: {_fmt(setup_all['setup_head_entropy'])} "
        f"over {setup_all['n_decisions']} setup decisions"
    )
    lines.append(
        f"- v8-vs-v8: {_fmt(setup_v8v8['setup_head_entropy'])} "
        f"({setup_v8v8['n_decisions']} decisions)"
    )
    lines.append(
        f"- v8-vs-anchor: {_fmt(setup_anchor['setup_head_entropy'])} "
        f"({setup_anchor['n_decisions']} decisions)"
    )
    lines.append("")
    lines.append("## M0 — in-distribution anchor (v8-vs-v8 subset only)")
    lines.append("")
    lines.append(
        "M0 estimator = post-setup `v̂` of the to-move seat (champion value head, "
        "squashed, true ports). Permutation strata = `draft_position` ONLY "
        "(no `opponent_strength.source` for self-play)."
    )
    lines.append("")
    lines.append(f"- n (v8-vs-v8 games): {m0['n_games']}")
    lines.append(f"- M1 statistic — AUC(`v̂`, outcome): {_fmt(m0['auc'])}")
    lines.append(
        f"- M2-analog partial Spearman (residual `outcome - v_hat` vs ORE+WHEAT "
        f"pip-share, partialling port adjacency): {_fmt(m0['partial_spearman'])}"
    )
    lines.append(
        f"- one-sided permutation p (positive sign, B={m0['permutations']}): {_fmt(m0['p_value'])}"
    )
    lines.append("")
    lines.append("## Freeze hashes")
    lines.append("")
    for name, h in (mass.get("freeze") or {}).items():
        lines.append(f"- `{name}`: {h}")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def freeze_manifest(*, v8_ckpt: str, anchor_ckpt: str, hash_ckpts: bool) -> dict[str, Any]:
    """SHA-256 of the frozen code artifacts + checkpoint provenance (plan §1)."""
    manifest: dict[str, Any] = {}
    for rel in _FREEZE_CODE_FILES:
        manifest[rel] = _sha256_file(REPO_ROOT / rel)
    for label, ckpt in (("v8_ckpt", v8_ckpt), ("anchor_ckpt", anchor_ckpt)):
        p = Path(ckpt)
        entry: dict[str, Any] = {
            "path": ckpt,
            "bytes": (p.stat().st_size if p.exists() else None),
        }
        if hash_ckpts:
            entry["sha256"] = _sha256_file(p)
        manifest[label] = entry
    return manifest


def run_pregate0(
    *,
    jsonl_path: Path,
    mass_path: Path,
    report_path: Path,
    n_per_matchup: int,
    seed: int,
    perms: int,
    max_turns: int,
    agent_policy: Any,
    v8_opponent: Any,
    anchor_opponent: Any,
    device: torch.device,
    v8_ckpt: str,
    anchor_ckpt: str,
    anchor_is_v8_copy: bool,
    freeze: dict[str, Any] | None = None,
    collapse_threshold: float = COLLAPSE_THRESHOLD,
    progress: bool = False,
) -> dict[str, Any]:
    """Play the two matchups (resumably), then write the mass table + report."""
    from catan_rl.human_data.topology import load_topology

    topology = load_topology()
    done = _done_game_ids(jsonl_path)
    matchups = ((MATCHUP_V8V8, v8_opponent), (MATCHUP_V8ANCHOR, anchor_opponent))
    for matchup, opponent in matchups:
        for gi in range(n_per_matchup):
            gid = f"{matchup}:{gi}"
            if gid in done:
                if progress:
                    print(f"[pregate0] skip {gid} (already logged)", flush=True)
                continue
            gseed = game_seed(seed, matchup, gi)
            record = play_game(
                agent_policy=agent_policy,
                opponent=opponent,
                device=device,
                seed=gseed,
                max_turns=max_turns,
                topology=topology,
                matchup=matchup,
                game_index=gi,
            )
            _append_line(jsonl_path, json.dumps(record, sort_keys=True))
            if progress:
                print(
                    f"[pregate0] {gid} v8_arch={record['seats']['0']['archetype']} "
                    f"v_hat={record['v_hat_to_move']:.3f} "
                    f"outcome_to_move={record['outcome_to_move']}",
                    flush=True,
                )

    records = load_records(jsonl_path)
    mass = build_mass_table(
        records,
        v8_ckpt=v8_ckpt,
        anchor_ckpt=anchor_ckpt,
        threshold=collapse_threshold,
        freeze=freeze,
    )
    _atomic_write(mass_path, json.dumps(mass, indent=2, sort_keys=True) + "\n")
    report = render_report(
        records,
        mass,
        perms=perms,
        seed=seed,
        v8_ckpt=v8_ckpt,
        anchor_ckpt=anchor_ckpt,
        anchor_is_v8_copy=anchor_is_v8_copy,
    )
    _atomic_write(report_path, report)
    return {
        "n_records": len(records),
        "mass": mass,
        "m0": compute_m0(records, perms=perms, seed=seed),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _load_policies(
    *, v8_ckpt: str, anchor_ckpt: str, seed: int, device_str: str
) -> tuple[Any, Any, Any, torch.device, bool]:
    """Load the champion + opponents. Returns
    ``(agent_policy, v8_opponent, anchor_opponent, device, anchor_is_v8_copy)``."""
    from catan_rl.replay.player_factory import PlayerSpec, build_actor
    from catan_rl.selfplay.snapshot_opponent import FrozenSnapshotOpponent

    v8_actor = build_actor(
        PlayerSpec(kind="policy", ckpt_path=v8_ckpt), seed=seed, device=device_str
    )
    device: torch.device = v8_actor.device  # type: ignore[attr-defined]
    agent_policy = v8_actor.policy  # type: ignore[attr-defined]
    v8_opponent = FrozenSnapshotOpponent(agent_policy, device=device, seed=seed)

    anchor_is_v8_copy = not Path(anchor_ckpt).expanduser().exists()
    if anchor_is_v8_copy:
        print(
            f"[pregate0] anchor {anchor_ckpt} not found on disk; "
            "falling back to a frozen v8 copy for the v8-vs-anchor subset.",
            flush=True,
        )
        anchor_opponent = FrozenSnapshotOpponent(agent_policy, device=device, seed=seed + 1)
    else:
        anchor_actor = build_actor(
            PlayerSpec(kind="policy", ckpt_path=anchor_ckpt), seed=seed, device=device_str
        )
        anchor_opponent = FrozenSnapshotOpponent(
            anchor_actor.policy,  # type: ignore[attr-defined]
            device=anchor_actor.device,  # type: ignore[attr-defined]
            seed=seed + 1,
        )
    return agent_policy, v8_opponent, anchor_opponent, device, anchor_is_v8_copy


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="PRE-GATE-0 + M0 runner (step6 §2.2).")
    parser.add_argument(
        "--n", type=int, default=400, help="Games per matchup (default 400; the real run)."
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="End-to-end smoke: n=3/matchup, fewer permutations, skip ckpt content hashing.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Base RNG seed.")
    parser.add_argument("--v8-ckpt", type=str, default=DEFAULT_V8_CKPT)
    parser.add_argument("--anchor-ckpt", type=str, default=DEFAULT_ANCHOR_CKPT)
    parser.add_argument("--out-dir", type=str, default=DEFAULT_OUT_DIR)
    parser.add_argument("--device", type=str, default="cpu", help="Eval device (default cpu).")
    parser.add_argument("--max-turns", type=int, default=400)
    parser.add_argument("--perms", type=int, default=2000, help="M0 permutation-test resamples.")
    parser.add_argument(
        "--hash-ckpts",
        action="store_true",
        help="SHA-256 the checkpoint files for the freeze manifest (slow; off in --smoke).",
    )
    args = parser.parse_args(argv)

    n_per_matchup = 3 if args.smoke else args.n
    perms = 200 if args.smoke else args.perms
    hash_ckpts = args.hash_ckpts and not args.smoke
    if n_per_matchup <= 0:
        parser.error("--n must be > 0")

    out_dir = Path(args.out_dir)
    jsonl_path = out_dir / "pregate0_games.jsonl"
    mass_path = out_dir / "pregate0_mass.json"
    report_path = out_dir / "pregate0_report.md"

    print(
        f"[pregate0] n={n_per_matchup}/matchup seed={args.seed} device={args.device} "
        f"perms={perms} out={out_dir}",
        flush=True,
    )
    agent_policy, v8_opp, anchor_opp, device, anchor_is_v8_copy = _load_policies(
        v8_ckpt=args.v8_ckpt,
        anchor_ckpt=args.anchor_ckpt,
        seed=args.seed,
        device_str=args.device,
    )
    freeze = freeze_manifest(
        v8_ckpt=args.v8_ckpt, anchor_ckpt=args.anchor_ckpt, hash_ckpts=hash_ckpts
    )
    result = run_pregate0(
        jsonl_path=jsonl_path,
        mass_path=mass_path,
        report_path=report_path,
        n_per_matchup=n_per_matchup,
        seed=args.seed,
        perms=perms,
        max_turns=args.max_turns,
        agent_policy=agent_policy,
        v8_opponent=v8_opp,
        anchor_opponent=anchor_opp,
        device=device,
        v8_ckpt=args.v8_ckpt,
        anchor_ckpt=args.anchor_ckpt,
        anchor_is_v8_copy=anchor_is_v8_copy,
        freeze=freeze,
        progress=True,
    )
    mass = result["mass"]
    m0 = result["m0"]
    print(
        f"[pregate0] DONE: {result['n_records']} records; "
        f"collapse={mass['collapse_verdict']} (max {mass['max_bucket']} "
        f"@ {mass['max_mass']:.3f}); M0 AUC={_fmt(m0['auc'])} "
        f"partial_rho={_fmt(m0['partial_spearman'])} p={_fmt(m0['p_value'])}",
        flush=True,
    )
    print(f"[pregate0] wrote {jsonl_path}, {mass_path}, {report_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
