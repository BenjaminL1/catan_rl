"""Paired policy-vs-ThePhantom opening comparison on HIS boards.

Binding spec: ``.claude/veriloop/specs/human-opening-reference.md``.

--------------------------------------------------------------------------
WHAT THIS IS
--------------------------------------------------------------------------
Replay the parsed ThePhantom corpus (``data/human/corpus/provisional_openings.jsonl``)
on the *real* hex layouts, script the *opponent's* placements from the corpus,
and at each of **ThePhantom's** four setup decisions ask the frozen champion
what it would place. Both choices are then scored by the same battery on the
**same board state** (D1). The deliverable is the resulting **paired metric
gap**.

The pairing is what makes the output readable without a "vs random" baseline —
the exact reference-class error that killed the previous opening direction (a
mid-rank percentile has E[pct]=50 as a MEAN property, so a "below-random
median" on tied discrete metrics is an artifact, not a finding).

--------------------------------------------------------------------------
SCOPE — THIS IS A REFUTATION INSTRUMENT (D4)
--------------------------------------------------------------------------
ThePhantom is a strong human, NOT an oracle, so a divergence is **unsigned**:
the policy differing from him is not evidence of a defect, and matching him is
not evidence of strength. This tool can **refute** "the openings are weak"
(policy matches or exceeds him on his own axes => the thread closes); it can
**never confirm** superiority. It measures **no win-probability** — the ExIt
closure already showed overrides that fire at 7.3% and are individually
win-neutral.

--------------------------------------------------------------------------
PORTS — K=8 MARGINALISATION, NOT A FIX (D2)
--------------------------------------------------------------------------
All 140 corpus rows carry ``board.ports = "OMITTED in v1"``, yet ports ARE in
the obs (per-vertex 7-way one-hot, ``policy/obs_encoder.py``). Harvesting the 9
slot->type labels per board from video is out of scope. Port *slots* are fixed
geometry, so this script instead marginalises: K=8 port assignments per board,
each derived from a deterministic ``sha256(video_id|game_index|k)`` seed, are
welded onto the board and the policy chooses under each. Aggregates are means
over k. This is a real limitation, not a repair; the report states it, and the
**port-marginal churn** (fraction of decisions whose policy choice changes
across the 8 assignments) is reported so the limitation is *quantified* rather
than only disclaimed.

The human's metric values are port-INVARIANT: neither ``BoardScorer`` nor the
D3 metrics below read ``vertex.port``. Ports only perturb the policy's choice.

--------------------------------------------------------------------------
NEW METRICS (D3) — HYPOTHESES, NOT ESTABLISHED QUANTITIES
--------------------------------------------------------------------------
The existing battery (imported wholesale from ``scripts/opening_sweep.py``) has
no jurisdiction over the axes the owner actually named. Three metrics are added
here. They are **hypotheses about what the owner reads**, not validated
quantities, and nothing downstream treats them as ground truth.

Notation: ``legal(occ)`` = vertices where a settlement could later be built
given occupancy ``occ`` (distance-rule); ``d(S -> u)`` = road-distance (edges in
the vertex-adjacency graph) from the nearest vertex of set ``S`` to ``u``, with
vertices owned by the other side reachable but not expanded through (engine
rule, ``BoardScorer.bfs``); ``pip(u)`` = ``BoardScorer.pip_sum[u]``.

``contested_race_count`` / ``contested_race_pips`` (SETTLEMENT decisions)
    Let ``M`` = my settlements including the candidate ``v``, ``O`` = the
    opponent's settlements. Over ``u in legal(occ + v)`` with both
    ``d(M -> u) <= 4`` and ``d(O -> u) <= 4`` (the *contested* set), count the
    ``u`` with ``d(M -> u) < d(O -> u)`` — the sites I reach strictly first —
    and their pip weight. **Undefined when the opponent has no settlement yet**
    (seat-0 decision 0): emitted as ``null`` and EXCLUDED from that decision's
    aggregate, never zero-filled.

``denial`` (SETTLEMENT decisions) — the LITERAL spec D3 definition, PRE-REGISTERED
    ``|legal(occ)| - |legal(occ + v)|``: the reduction in the opponent's legal
    future-settlement set caused by my placement, **unrestricted**, exactly as
    spec D3 words it. **Undefined when the opponent has no settlement yet**
    (there is no opponent expansion to deny); emitted as ``null``.

``denial_reach`` (SETTLEMENT decisions) — a SHARPENING, exploratory, NOT a substitute
    ``|L_before| - |L_after|`` where ``L`` is the opponent's *reachable* legal
    set: ``{u in legal(occ) : d(O -> u) <= 4}``, before and after adding ``v``.
    Rationale for offering it: the unrestricted ``denial`` above is close to "how
    many legal neighbours does v have", i.e. largely a board-supply quantity
    rather than a policy choice — the failure mode (flagship metric measuring
    supply, not choice) that killed the previous direction. But the spec is
    OWNER-DELEGATED and ``denial`` is pre-registered, so this variant is reported
    ALONGSIDE the literal metric and flagged for an owner decision; it does not
    replace the pre-registered row. Both are undefined when the opponent has no
    settlement yet.

    Both denial variants measure only the **distance-rule channel** of denial:
    reachability is computed from the PRE-candidate blocking set for both terms,
    so the candidate's other effect — walling the opponent's road expansion
    *through* that vertex — is deliberately NOT counted. The road-blocking
    channel of "denial of the opponent's expansion" is therefore UNMEASURED by
    both rows.

``cut_vulnerability`` (ROAD decisions)
    The number of articulation points of my own road subgraph (the vertex-induced
    graph of the roads I own, including the candidate road) — vertices whose
    removal disconnects it. It came out **identically 0.0 on every decision in
    this run, for both choosers** — but that column is **half STRUCTURAL and half
    EMPIRICAL, and the two must not be pooled**. Exactly half of the road
    decisions are the FIRST-road decision (``decision_index == 1``), where the
    chooser owns no prior road, the subgraph is a single edge, and 0.0 is forced
    BY CONSTRUCTION (a 1-edge graph has no articulation point — pinned by
    ``test_cut_vulnerability_counts_articulation_points``). Only the SECOND-road
    decisions — 134 of the 268 evaluations in this run — carry information, and
    it is on those, not on all 268, that the null is empirical. There it is *not*
    structurally impossible: the distance rule forbids ADJACENT settlements but
    permits two of mine at graph distance 2 sharing a middle vertex, and if both
    setup roads point at that shared vertex the subgraph is a 3-vertex path whose
    middle IS an articulation point (verified legal against the engine masks:
    corpus row ``b_NXNI-l0kI`` g6 offers exactly that double-back at its second
    road decision). Neither chooser ever built it, which is itself the finding.
    It is also NOT the "plow" the owner described (an *opponent* road severing MY
    longest path): expressing that needs the opponent's road network and a
    longest-trail recomputation, which this definition does not touch.

--------------------------------------------------------------------------
HONESTY CONSTRAINTS CARRIED FROM THE CORPUS (D5 / D6)
--------------------------------------------------------------------------
* ``opponent_strength`` is a hardcoded constant across all 140 rows: **zero
  bits**. Never stratified, weighted, or cited as validation.
* Rows lacking ``provenance.placement_order_established`` are SKIPPED, never
  guessed, and appear in the exclusion ledger.
* Exact-match agreement rate is **descriptive colour only, NOT a gate**
  (step6 §6 lists move-agreement as a non-goal).
* Confidence intervals are **cluster bootstraps over ``video_id``** (118 videos
  carry 140 rows; rows from one video share a session and opponent, so
  row-level resampling would be anti-conservative).
* The battery is a multi-metric family. The PRE-REGISTERED primary is the D3
  paired gap; every other metric is exploratory. The CIs are **UNADJUSTED for
  multiplicity**, and a 95% CI that excludes zero *is* a rejected two-sided test
  at alpha=0.05 — withholding the p-value removes the label, not the
  multiplicity. The report states the expected false-positive count for the
  exploratory family alongside the observed one.
* **Informative missingness is counted, never hidden.** Several battery metrics
  are undefined for some candidates (e.g. ``pair_exp_rolls_to_city`` needs a
  city-self-sufficient pair). Dropping those decisions selects the comparison
  subset on the very outcome being compared, so ``summarise`` records how many
  decisions were dropped because the POLICY's pick was undefined and the report
  prints an explicit warning for every metric where that count is nonzero.
* The **corpus census** required by spec D6 (rows / videos / unique layouts /
  winners known) is recomputed on every run and compared against the pinned
  expectation, so a corpus regeneration cannot silently move every number.

--------------------------------------------------------------------------
READ-ONLY W.R.T. TRAINING (D7 / AC8)
--------------------------------------------------------------------------
CPU only (``--device`` is REJECTED unless it resolves to CPU — an accelerator
would contend with the live training run), single-threaded, reads only an
immutable numbered checkpoint. Both output directories are validated against
``ALLOWED_WRITE_ROOTS`` before anything is created, so ``--out-dir`` /
``--raw-dir`` cannot be pointed at ``runs/train/**`` (or anywhere else outside
``data/human/**`` and ``runs/analysis/**``) even by accident.

The committed deliverable was produced with::

    PYTHONPATH=<repo>/src nice -n 19 python scripts/human_opening_reference.py \\
        --ckpt <repo>/runs/train/selfplay_pointer_arch_v2/checkpoints/ckpt_000000500.pt

``PYTHONPATH`` matters: the env board-injection seam lives in this tree, and an
installed ``catan_rl`` that predates it would silently IGNORE the injected layout
(``options`` keys are not validated by Gymnasium). ``assert_layout_applied``
turns that into a loud per-reset failure rather than a report full of numbers
scored on random boards.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]

# Sibling-script import (the ``scripts/`` directory is not a package). Same
# mechanism ``scripts/ladder_gate.py`` uses to reach ``scripts/elo_ladder.py``.
_SWEEP_PATH = REPO / "scripts" / "opening_sweep.py"
_spec = importlib.util.spec_from_file_location("opening_sweep_module", _SWEEP_PATH)
if _spec is None or _spec.loader is None:  # pragma: no cover - defensive
    raise ImportError(f"cannot load {_SWEEP_PATH}")
opening_sweep = importlib.util.module_from_spec(_spec)
sys.modules["opening_sweep_module"] = opening_sweep
_spec.loader.exec_module(opening_sweep)

from catan_rl.env.catan_env import ActionType, CatanEnv
from catan_rl.policy.obs_tensor import masks_to_torch, obs_to_torch

BUILD_SETTLEMENT = ActionType.BUILD_SETTLEMENT
BUILD_ROAD = ActionType.BUILD_ROAD

BoardScorer = opening_sweep.BoardScorer
PAIR_METRICS: tuple[str, ...] = opening_sweep.PAIR_METRICS
ROAD_METRICS: tuple[str, ...] = opening_sweep.ROAD_METRICS
SETTLEMENT_METRICS: tuple[str, ...] = opening_sweep.SETTLEMENT_METRICS

DEFAULT_CORPUS = REPO / "data" / "human" / "corpus" / "provisional_openings.jsonl"
DEFAULT_OUT_DIR = REPO / "data" / "human" / "opening_reference"
#: The per-decision records are part of the deliverable ("a numbers-first report
#: + raw JSON"), so they default to the TRACKED ``data/human/**`` root — the
#: repo's convention for human-corpus artifacts. ``runs/**`` is gitignored, so a
#: raw file written there would not exist on the pushed branch at all.
DEFAULT_RAW_DIR = REPO / "data" / "human" / "opening_reference"
DEFAULT_CKPT = (
    REPO / "runs" / "train" / "selfplay_pointer_arch_v2" / "checkpoints" / "ckpt_000000500.pt"
)

#: Number of port assignments marginalised over (D2).
N_PORT_SAMPLES = 8

#: Road-distance horizon for the D3 reachability metrics. Matches the
#: ``max_depth=4`` the existing battery's ``exp_d2/3/4`` expansion metrics use,
#: so the two families describe the same neighbourhood.
REACH_DEPTH = 4

#: The D3 additions. Settlement-decision metrics first, then the road metric.
CONTESTED_METRICS: tuple[str, ...] = (
    "contested_race_count",
    "contested_race_pips",
    "denial",
    "denial_reach",
)
CUT_METRICS: tuple[str, ...] = ("cut_vulnerability",)

#: Pre-registered PRIMARY family (D3 on settlement decisions). ``denial`` is the
#: LITERAL spec-D3 quantity; ``denial_reach`` is the road-reach-restricted
#: sharpening and is EXPLORATORY — the spec is owner-delegated, so the primary
#: row is not re-defined unilaterally. Everything else is exploratory too.
PRIMARY_METRICS: tuple[str, ...] = ("contested_race_count", "contested_race_pips", "denial")

#: AC8 / D7: the ONLY directory trees this script may write into. Everything
#: else — above all ``runs/train/**``, where a multi-day self-play run lives — is
#: rejected before a single directory is created.
ALLOWED_WRITE_ROOTS: tuple[Path, ...] = (REPO / "data" / "human", REPO / "runs" / "analysis")

#: Spec D6 census to reproduce. Recomputed on every run; a divergence is printed
#: loudly in the report rather than silently changing every number.
EXPECTED_CENSUS: dict[str, int] = {
    "rows": 140,
    "videos": 118,
    "unique_layouts": 135,
    "winners_known": 117,
}

#: ``video_id`` values that are NOT real videos (``scripts/vlm_spike.py`` writes
#: this as its hardcoded default). Disclosed in the report, not filtered — the
#: row is otherwise well-formed and dropping it silently would be worse.
SYNTHETIC_VIDEO_IDS: frozenset[str] = frozenset({"vlm_spike"})


class HardConstraintError(RuntimeError):
    """A spec hard constraint (D7 / AC8) would have been violated.

    Raised BEFORE any side effect: an output directory outside
    ``ALLOWED_WRITE_ROOTS`` or a non-CPU device while training runs on the M1.
    """


def check_write_target(path: Path) -> Path:
    """Resolve ``path`` and assert it lies inside ``ALLOWED_WRITE_ROOTS`` (AC8)."""
    resolved = path.expanduser().resolve()
    for root in ALLOWED_WRITE_ROOTS:
        if resolved == root.resolve() or resolved.is_relative_to(root.resolve()):
            return resolved
    allowed = ", ".join(str(r) for r in ALLOWED_WRITE_ROOTS)
    raise HardConstraintError(
        f"refusing to write to {resolved}: spec AC8 permits only {allowed} "
        "(never runs/train/**, where the live self-play run lives)"
    )


def check_device(device: torch.device) -> torch.device:
    """Assert the device is CPU (D7: the M1's accelerator is training)."""
    if device.type != "cpu":
        raise HardConstraintError(
            f"refusing to run on device {device!r}: spec D7 pins this instrument to "
            "CPU so it cannot contend with the live selfplay_pointer_arch_v3 run"
        )
    return device


class CorpusReplayError(RuntimeError):
    """A corpus row could not be replayed faithfully.

    Raised (and caught per row) when a recorded placement is illegal in the
    injected engine state, or when the scripted opponent is asked for an action
    it was not given. Never swallowed silently: every occurrence lands in the
    exclusion ledger.
    """


# ---------------------------------------------------------------------------
# Corpus
# ---------------------------------------------------------------------------


def load_rows(path: Path) -> list[dict[str, Any]]:
    """Read the corpus JSONL verbatim (no filtering — the caller decides)."""
    with path.open() as fh:
        return [json.loads(line) for line in fh if line.strip()]


def corpus_census(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Reproduce the spec-D6 corpus census, plus the provenance facts a reader
    needs in order to know how much to trust the placement ORDER.

    D6 names "140 rows, 118 videos, 135 unique layouts, 117 winners known" as an
    integrity check. It is recomputed here and compared against
    ``EXPECTED_CENSUS`` so a corpus regeneration is visible instead of silently
    moving every reported number.
    """
    layouts = {
        tuple(
            sorted((int(h["hex_id"]), str(h["resource"]), h["number"]) for h in r["board"]["hexes"])
        )
        for r in rows
    }
    order_sources: dict[str, int] = defaultdict(int)
    for r in rows:
        order_sources[str(r.get("provenance", {}).get("order_source"))] += 1
    census: dict[str, Any] = {
        "rows": len(rows),
        "videos": len({str(r["video_id"]) for r in rows}),
        "unique_layouts": len(layouts),
        "winners_known": sum(1 for r in rows if r.get("winner")),
    }
    census["expected"] = dict(EXPECTED_CENSUS)
    census["matches_expected"] = all(census[k] == v for k, v in EXPECTED_CENSUS.items())
    census["order_source_counts"] = dict(sorted(order_sources.items()))
    census["synthetic_video_rows"] = sorted(
        f"{r['video_id']} g{r['game_index']}"
        for r in rows
        if str(r["video_id"]) in SYNTHETIC_VIDEO_IDS
    )
    return census


def assert_layout_applied(board: Any, layout: dict[str, Any]) -> None:
    """Verify the injected layout actually reached the engine (AC2, at RUNTIME).

    ``CatanEnv.reset`` takes the layout through ``options``, and an ``options``
    key the running ``catan_rl`` build does not know about is silently IGNORED —
    which would score every decision on a RANDOM board while the run looks
    perfectly healthy. That is not hypothetical: it happens whenever this script
    is launched against an installed ``catan_rl`` that predates the seam. So the
    injection is verified per reset rather than trusted.
    """
    for hex_idx in range(19):
        tile = board.hexTileDict[hex_idx]
        want_res = layout["resources"][hex_idx]
        want_num = layout["numbers"][hex_idx]
        if tile.resource_type != want_res or tile.number_token != want_num:
            raise CorpusReplayError(
                f"board injection did NOT take: hex {hex_idx} is "
                f"{tile.resource_type}/{tile.number_token}, expected "
                f"{want_res}/{want_num}. The running catan_rl build probably lacks "
                'the CatanEnv.reset options["board_layout"] seam — run against this '
                "tree's src/ (PYTHONPATH=<repo>/src)."
            )
    port_assignment = layout.get("port_assignment")
    if port_assignment is not None and board.get_port_assignment() != port_assignment:
        raise CorpusReplayError("port assignment did NOT take (same missing-seam cause)")


def layout_from_row(row: dict[str, Any]) -> dict[str, Any]:
    """Build the ``options["board_layout"]`` payload for one corpus row.

    Raises:
        CorpusReplayError: if the row's hex block is not a well-formed 19-hex
            layout whose single desert coincides with the recorded robber hex.
    """
    hexes = row["board"]["hexes"]
    if len(hexes) != 19:
        raise CorpusReplayError(f"expected 19 hexes, got {len(hexes)}")
    resources: list[str] = [""] * 19
    numbers: list[int | None] = [None] * 19
    for entry in hexes:
        h = int(entry["hex_id"])
        if not 0 <= h < 19:
            raise CorpusReplayError(f"hex_id {h} out of range")
        resources[h] = str(entry["resource"])
        numbers[h] = None if entry["number"] is None else int(entry["number"])
    if any(r == "" for r in resources):
        raise CorpusReplayError("hex_id block does not cover 0..18")
    robber_hex = int(row["provenance"]["board_desert_hex"])
    if resources[robber_hex] != "DESERT":
        raise CorpusReplayError(
            f"board_desert_hex {robber_hex} is {resources[robber_hex]!r}, not DESERT"
        )
    return {"resources": resources, "numbers": numbers, "robber_hex": robber_hex}


def port_seed(video_id: str, game_index: int, k: int) -> int:
    """Deterministic per-(row, k) port seed (D2 / AC5).

    ``sha256("<video_id>|<game_index>|<k>")`` truncated to 64 bits. Stable
    across processes, machines and Python versions (unlike ``hash()``).
    """
    digest = hashlib.sha256(f"{video_id}|{game_index}|{k}".encode()).hexdigest()
    return int(digest[:16], 16)


class PortSampler:
    """Deterministic port assignments drawn from the engine's own shuffle.

    Reuses the ratified ``updatePorts(rng=...)`` path on a single throwaway
    board, so the sampled assignments are exactly the ones the engine can
    generate: the 9 port *slots* stay real fixed geometry and only the
    slot->type mapping permutes.
    """

    def __init__(self) -> None:
        from catan_rl.engine.game import catanGame

        self._board = catanGame(render_mode=None).board

    def assignment(self, seed: int) -> dict[str, list[int]]:
        self._board.updatePorts(rng=np.random.default_rng(seed))
        return self._board.get_port_assignment()


# ---------------------------------------------------------------------------
# Scripted opponent (teacher forcing of the corpus opponent's setup)
# ---------------------------------------------------------------------------


class ScriptedSetupOpponent:
    """Drives the opponent seat from the corpus, one recorded placement at a time.

    Duck-types the ``SnapshotOpponent`` protocol (``.device`` / ``.reset_rng`` /
    ``.sample``) that ``CatanEnv._sample_snapshot_action`` calls. The env reads
    only ``action[1]`` for a setup settlement and ``action[2]`` for a setup road,
    so returning the recorded index in the right slot reproduces the human
    opponent's opening exactly.

    Every returned index is checked against the mask BEFORE it is applied, and
    the script raises once exhausted — so a main-turn leak (which would consume
    an action this object has no legal answer for) is impossible rather than
    silently mis-placed.
    """

    def __init__(self, script: list[tuple[str, int]], device: torch.device) -> None:
        self._script = script
        self._device = device
        self._pos = 0

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def consumed(self) -> int:
        return self._pos

    def reset_rng(self, seed: int | None = None) -> None:
        """No randomness to reset — the script is fixed. Part of the protocol."""

    def sample(self, obs: dict[str, torch.Tensor], masks: dict[str, torch.Tensor]) -> torch.Tensor:
        if self._pos >= len(self._script):
            raise CorpusReplayError(
                "scripted opponent exhausted: the env asked for action "
                f"#{self._pos + 1} but only {len(self._script)} setup placements "
                "were scripted (a main-turn leak)"
            )
        kind, idx = self._script[self._pos]
        self._pos += 1
        mask_key = "corner_settlement" if kind == "settlement" else "edge"
        if not bool(masks[mask_key][0, idx].item()):
            raise CorpusReplayError(
                f"scripted opponent {kind} {idx} is ILLEGAL at placement #{self._pos}"
            )
        action = torch.zeros((1, 6), dtype=torch.long, device=self._device)
        if kind == "settlement":
            action[0, 0] = BUILD_SETTLEMENT
            action[0, 1] = idx
        else:
            action[0, 0] = BUILD_ROAD
            action[0, 2] = idx
        return action


# ---------------------------------------------------------------------------
# D3 metrics
# ---------------------------------------------------------------------------


def _multi_source_bfs(
    scorer: Any, sources: list[Any], blocked: set[Any], max_depth: int
) -> dict[Any, int]:
    """Min road-distance from ANY vertex in ``sources`` (0 for the sources)."""
    best: dict[Any, int] = {}
    for src in sources:
        for u, d in scorer.bfs(src, blocked, max_depth).items():
            if d < best.get(u, math.inf):
                best[u] = d
    return best


def settlement_extra_metrics(
    scorer: Any,
    v_px: Any,
    *,
    base_occupied: set[Any],
    my_settlements: list[Any],
    opp_settlements: list[Any],
    my_blocked: set[Any],
    opp_blocked: set[Any],
) -> dict[str, float | None]:
    """``contested_race_*``, ``denial`` and ``denial_reach`` for one candidate (D3).

    All are ``None`` when the opponent has no settlement yet — the race and the
    denial are simply undefined then, and zero-filling would inject a fake tie
    into the paired gap.
    """
    if not opp_settlements:
        return {name: None for name in CONTESTED_METRICS}

    occ_after = set(base_occupied)
    occ_after.add(v_px)
    legal_after = scorer.legal_sites(occ_after)
    legal_before = scorer.legal_sites(base_occupied)

    mine = [*my_settlements, v_px]
    d_mine = _multi_source_bfs(scorer, mine, my_blocked, REACH_DEPTH)
    # The opponent's expansion is blocked by MY buildings, the candidate included.
    d_opp = _multi_source_bfs(scorer, opp_settlements, opp_blocked | {v_px}, REACH_DEPTH)

    count = 0
    pips = 0
    for u in legal_after:
        dm = d_mine.get(u)
        do = d_opp.get(u)
        if dm is None or do is None or dm > REACH_DEPTH or do > REACH_DEPTH:
            continue
        if dm < do:
            count += 1
            pips += scorer.pip_sum[u]

    # ``denial`` = the LITERAL spec D3 quantity: the unrestricted shrinkage of the
    # opponent's legal future-settlement set. Pre-registered, so it is NOT
    # re-defined here.
    denial = len(legal_before) - len(legal_after)

    # ``denial_reach`` = the same shrinkage restricted to the opponent's
    # ROAD-REACHABLE set, reported ALONGSIDE (never instead of) ``denial``.
    # Reachability is measured on the pre-placement blocking set for both terms,
    # so the difference isolates the distance-rule channel and NOT the
    # road-blocking channel (walling his expansion through v) — that channel is
    # uncounted by both rows.
    d_opp_before = _multi_source_bfs(scorer, opp_settlements, opp_blocked, REACH_DEPTH)
    reach = {u for u, d in d_opp_before.items() if d <= REACH_DEPTH}
    denial_reach = len(legal_before & reach) - len(legal_after & reach)

    return {
        "contested_race_count": float(count),
        "contested_race_pips": float(pips),
        "denial": float(denial),
        "denial_reach": float(denial_reach),
    }


def _articulation_points(adjacency: dict[Any, set[Any]]) -> set[Any]:
    """Articulation points of an undirected graph (iterative Hopcroft-Tarjan)."""
    visited: set[Any] = set()
    disc: dict[Any, int] = {}
    low: dict[Any, int] = {}
    parent: dict[Any, Any] = {}
    arts: set[Any] = set()
    timer = 0
    for root in adjacency:
        if root in visited:
            continue
        root_children = 0
        stack: list[tuple[Any, list[Any]]] = [(root, list(adjacency[root]))]
        visited.add(root)
        disc[root] = low[root] = timer
        timer += 1
        while stack:
            node, pending = stack[-1]
            if pending:
                nb = pending.pop()
                if nb not in visited:
                    parent[nb] = node
                    visited.add(nb)
                    disc[nb] = low[nb] = timer
                    timer += 1
                    if node == root:
                        root_children += 1
                    stack.append((nb, list(adjacency[nb])))
                elif nb != parent.get(node):
                    low[node] = min(low[node], disc[nb])
            else:
                stack.pop()
                if stack:
                    up = stack[-1][0]
                    low[up] = min(low[up], low[node])
                    if up != root and low[node] >= disc[up]:
                        arts.add(up)
        if root_children > 1:
            arts.add(root)
    return arts


def road_extra_metrics(my_roads: list[tuple[Any, Any]]) -> dict[str, float]:
    """``cut_vulnerability`` for a road set (D3).

    Articulation points of the vertex-induced graph of the roads I own. At setup
    this is near-degenerate (<= 2 roads) but NOT identically zero: two of my
    settlements may sit at graph distance 2, and two roads both pointing at the
    shared middle vertex make that vertex an articulation point. The observed
    all-zero column is empirical, not structural; the report says so.
    """
    adjacency: dict[Any, set[Any]] = defaultdict(set)
    for v1, v2 in my_roads:
        adjacency[v1].add(v2)
        adjacency[v2].add(v1)
    return {"cut_vulnerability": float(len(_articulation_points(dict(adjacency))))}


# ---------------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------------


@dataclass
class PairedDecision:
    """One of ThePhantom's four setup decisions, scored for both choosers."""

    video_id: str
    game_index: int
    agent_seat: int
    decision_index: int
    kind: str
    human_id: int
    policy_ids: list[int]
    n_candidates: int
    #: Settlements the OPPONENT already owns at this decision. The contested/
    #: denial metrics are defined relative to that set, so their scale differs
    #: systematically between the (seat, decision) regimes and the report breaks
    #: the primary family down by this value rather than pooling it silently.
    opp_settlements: int
    human_metrics: dict[str, float | None]
    #: Per-metric list over the K port assignments (parallel to ``policy_ids``).
    policy_metrics: dict[str, list[float | None]]

    def gaps(self) -> dict[str, float]:
        """mean_k(policy) - human, dropping undefined (None) terms entirely."""
        out: dict[str, float] = {}
        for name, human in self.human_metrics.items():
            if human is None:
                continue
            vals = [v for v in self.policy_metrics.get(name, []) if v is not None]
            if not vals:
                continue
            out[name] = sum(vals) / len(vals) - human
        return out

    def agreement(self) -> float:
        """Fraction of the K assignments whose argmax equals the human's choice.

        DESCRIPTIVE ONLY — not a gate (D5).
        """
        if not self.policy_ids:
            return float("nan")
        return sum(1 for i in self.policy_ids if i == self.human_id) / len(self.policy_ids)

    def churn(self) -> float:
        """Fraction of the K assignments that do NOT take the modal choice.

        Quantifies the D2 port-marginal limitation: 0.0 means ports never moved
        the policy on this decision; higher means the choice is port-driven.
        """
        if not self.policy_ids:
            return float("nan")
        modal = max(set(self.policy_ids), key=self.policy_ids.count)
        return 1.0 - self.policy_ids.count(modal) / len(self.policy_ids)


def _score_candidates(
    env: CatanEnv,
    scorer: Any,
    *,
    settlement: bool,
    first_settlement: Any | None,
) -> tuple[list[int], dict[str, list[float | None]], list[Any]]:
    """The existing battery plus the D3 additions, for every legal candidate."""
    ids, metrics, objs = opening_sweep._score_decision(
        env, scorer, settlement=settlement, first_settlement=first_settlement
    )
    assert env.game is not None and env.agent_player is not None
    agent = env.agent_player
    opp = env.opponent_player
    assert opp is not None

    if settlement:
        base_occupied = scorer.occupied()
        my_settlements = list(agent.buildGraph["SETTLEMENTS"])
        opp_settlements = list(opp.buildGraph["SETTLEMENTS"])
        my_blocked = scorer.opponent_blocked(agent)
        opp_blocked = scorer.opponent_blocked(opp)
        for name in CONTESTED_METRICS:
            metrics[name] = []
        for v_px in objs:
            extra = settlement_extra_metrics(
                scorer,
                v_px,
                base_occupied=base_occupied,
                my_settlements=my_settlements,
                opp_settlements=opp_settlements,
                my_blocked=my_blocked,
                opp_blocked=opp_blocked,
            )
            for name in CONTESTED_METRICS:
                metrics[name].append(extra[name])
    else:
        existing_roads = [tuple(e) for e in agent.buildGraph["ROADS"]]
        for name in CUT_METRICS:
            metrics[name] = []
        for v1, v2 in objs:
            extra_r = road_extra_metrics([*existing_roads, (v1, v2)])
            for name in CUT_METRICS:
                metrics[name].append(extra_r[name])
    return ids, metrics, objs


def _human_decisions(
    row: dict[str, Any],
) -> tuple[int, list[tuple[str, int]], list[tuple[str, int]]]:
    """Return ``(agent_seat, agent_script, opponent_script)`` for one row.

    ``settlements[i]`` / ``roads[i]`` are in PLACEMENT order (``settlements[1]``
    is the resource-granting second placement, ``scripts/vlm_spike.py``), so the
    interleaved (settlement, road) sequence is read off directly.
    """
    agent_name = row["players"]["agent"]
    order = row["draft_order"]
    if len(order) != 4:
        raise CorpusReplayError(f"draft_order has {len(order)} entries, expected 4")
    # The replay assumes the 1v1 snake draft (1 -> 2 -> 2 -> 1) that the engine
    # implements. An alternating order [A, B, A, B] would be silently replayed as
    # a snake, putting every decision on the wrong board state, so reject it
    # rather than guess (D6: never guess order).
    if not (order[0] == order[3] and order[1] == order[2] and order[0] != order[1]):
        raise CorpusReplayError(f"draft_order {order} is not a snake draft (A,B,B,A)")
    if order[0] == agent_name:
        agent_seat = 0
    elif order[1] == agent_name:
        agent_seat = 1
    else:
        raise CorpusReplayError(f"agent {agent_name!r} not found in draft_order {order}")
    opp_name = row["players"]["opponent"]

    def script(name: str) -> list[tuple[str, int]]:
        opening = row["openings"][name]
        settlements = opening["settlements"]
        roads = opening["roads"]
        if len(settlements) != 2 or len(roads) != 2:
            raise CorpusReplayError(f"{name} opening is not 2 settlements + 2 roads")
        return [
            ("settlement", int(settlements[0])),
            ("road", int(roads[0])),
            ("settlement", int(settlements[1])),
            ("road", int(roads[1])),
        ]

    return agent_seat, script(agent_name), script(opp_name)


def replay_row(
    row: dict[str, Any],
    policy: Any,
    device: torch.device,
    ports: PortSampler,
    *,
    n_port_samples: int = N_PORT_SAMPLES,
) -> list[PairedDecision]:
    """Replay one corpus row under K port assignments; return its 4 decisions.

    Raises:
        CorpusReplayError: on any fidelity violation (malformed layout, illegal
            recorded placement, opponent-script exhaustion). The caller records
            the row in the exclusion ledger; nothing is ever half-applied.
    """
    video_id = str(row["video_id"])
    game_index = int(row["game_index"])
    layout = layout_from_row(row)
    agent_seat, agent_script, opp_script = _human_decisions(row)

    decisions: list[PairedDecision] = []
    #: Candidate ids per decision as seen at k == 0, used to pin the port
    #: invariance the ``human_metrics``-stored-once-at-k==0 shortcut relies on.
    cand_ids_k0: list[list[int]] = []
    for k in range(n_port_samples):
        seed = port_seed(video_id, game_index, k)
        layout_k = dict(layout)
        layout_k["port_assignment"] = ports.assignment(seed)

        opponent = ScriptedSetupOpponent(opp_script, device)
        env = CatanEnv(opponent_type="snapshot")
        env.set_snapshot_opponent(opponent)
        env.reset(
            seed=seed % (2**31),
            options={"agent_seat": agent_seat, "board_layout": layout_k},
        )
        assert env.game is not None and env.opponent_player is not None
        assert_layout_applied(env.game.board, layout_k)
        scorer = BoardScorer(env.game.board)
        first_settlement: Any | None = None

        for d_idx in range(4):
            settlement = d_idx in (0, 2)
            kind, human_id = agent_script[d_idx]
            assert kind == ("settlement" if settlement else "road")
            ids, metrics, objs = _score_candidates(
                env,
                scorer,
                settlement=settlement,
                first_settlement=first_settlement if d_idx == 2 else None,
            )
            if human_id not in ids:
                raise CorpusReplayError(
                    f"recorded {kind} {human_id} is ILLEGAL at decision {d_idx}"
                )
            human_pos = ids.index(human_id)

            obs_now = env._get_obs()
            masks_now = env.get_action_masks()
            obs_t = obs_to_torch(obs_now, device, add_batch=True)
            masks_t = masks_to_torch(masks_now, device, add_batch=True)
            with torch.no_grad():
                logp = opening_sweep.setup_logp(policy, obs_t, masks_t, settlement)[0]
            policy_id = int(logp.argmax().item())
            if policy_id not in ids:  # pragma: no cover - mask invariant
                raise CorpusReplayError(f"policy argmax {policy_id} outside the legal set")
            policy_pos = ids.index(policy_id)

            human_metrics = {n: v[human_pos] for n, v in metrics.items()}
            if k == 0:
                decisions.append(
                    PairedDecision(
                        video_id=video_id,
                        game_index=game_index,
                        agent_seat=agent_seat,
                        decision_index=d_idx,
                        kind="settlement" if settlement else "road",
                        human_id=human_id,
                        policy_ids=[],
                        n_candidates=len(ids),
                        opp_settlements=len(env.opponent_player.buildGraph["SETTLEMENTS"]),
                        human_metrics=human_metrics,
                        policy_metrics={n: [] for n in metrics},
                    )
                )
                cand_ids_k0.append(list(ids))
            rec = decisions[d_idx]
            # AC6 pairing invariant: both choosers are scored on the SAME
            # candidate set at the SAME board state, and the human's metrics
            # are port-invariant (neither ``BoardScorer`` nor the D3 metrics read
            # ``vertex.port``), so BOTH the candidate ids AND the human's metric
            # values must reproduce exactly across k — which is what licenses
            # storing ``human_metrics`` only at k == 0. Check the assertion
            # actually leaned on, not just the candidate-set length.
            if cand_ids_k0[d_idx] != ids:
                raise CorpusReplayError(
                    f"candidate set changed across port samples at decision {d_idx}"
                )
            if rec.human_metrics != human_metrics:
                raise CorpusReplayError(
                    f"human metrics changed across port samples at decision {d_idx} "
                    "(they must be port-invariant)"
                )
            rec.policy_ids.append(policy_id)
            for name, vals in metrics.items():
                rec.policy_metrics.setdefault(name, []).append(vals[policy_pos])

            if d_idx == 3:
                # Scored from the mask, never applied: nothing downstream is
                # measured and applying it would start a main turn (which the
                # scripted opponent cannot answer).
                break
            # Step with the HUMAN's action — the policy's choice never alters
            # the trajectory, which is what makes the pairing exact (D1/AC6).
            act = np.zeros(6, dtype=np.int64)
            if settlement:
                act[0], act[1] = BUILD_SETTLEMENT, human_id
                if d_idx == 0:
                    first_settlement = objs[human_pos]
            else:
                act[0], act[2] = BUILD_ROAD, human_id
            env.step(act)
    return decisions


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


@dataclass
class GapSummary:
    metric: str
    kind: str
    n_decisions: int
    n_videos: int
    mean_gap: float
    ci_lo: float
    ci_hi: float
    mean_human: float
    mean_policy: float
    #: Every decision of this ``kind`` that the metric COULD have covered.
    n_decisions_of_kind: int = 0
    #: Of those, the ones the HUMAN's pick left undefined (structural: the metric
    #: does not apply at that decision, e.g. contested/denial at seat-0 decision
    #: 0, or pair metrics at a first settlement).
    n_human_undefined: int = 0
    #: Human defined but EVERY port sample of the policy's pick undefined, so the
    #: decision is dropped. This is INFORMATIVE MISSINGNESS: the decision is
    #: excluded *because* the policy's pick failed the metric's precondition, so
    #: the surviving subset is selected on the outcome being compared.
    n_dropped_policy_undefined: int = 0
    #: Human defined and only SOME port samples defined — averaged over the
    #: survivors, so those rows are partially selected too.
    n_partial_policy_undefined: int = 0

    @property
    def has_informative_missingness(self) -> bool:
        return (self.n_dropped_policy_undefined + self.n_partial_policy_undefined) > 0


def cluster_bootstrap_ci(
    values: list[float], clusters: list[str], *, n_boot: int = 2000, seed: int = 0
) -> tuple[float, float]:
    """Percentile CI from a bootstrap that resamples ``video_id`` CLUSTERS.

    Rows from one video share a session and an opponent, so resampling rows
    independently would understate the interval.
    """
    if not values:
        return (float("nan"), float("nan"))
    by_cluster: dict[str, list[float]] = defaultdict(list)
    for v, c in zip(values, clusters, strict=True):
        by_cluster[c].append(v)
    keys = sorted(by_cluster)
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        pick = rng.integers(0, len(keys), size=len(keys))
        pooled = [x for i in pick for x in by_cluster[keys[i]]]
        means[b] = float(np.mean(pooled))
    return (float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975)))


def summarise(decisions: list[PairedDecision], metric: str, kind: str) -> GapSummary | None:
    """Paired mean gap for one metric, WITH the definedness bookkeeping.

    The counts are not decoration: a metric that is undefined for the policy's
    pick (``pair_exp_rolls_to_city`` on a pair that produces no ore) is dropped
    here, and dropping on the outcome is exactly the artifact class that killed
    the previous opening direction. The counts let the report say so.
    """
    gaps: list[float] = []
    clusters: list[str] = []
    humans: list[float] = []
    policies: list[float] = []
    n_of_kind = 0
    n_human_undef = 0
    n_dropped = 0
    n_partial = 0
    for d in decisions:
        if d.kind != kind:
            continue
        n_of_kind += 1
        human = d.human_metrics.get(metric)
        if human is None:
            n_human_undef += 1
            continue
        raw = d.policy_metrics.get(metric, [])
        vals = [v for v in raw if v is not None]
        if not vals:
            n_dropped += 1
            continue
        if len(vals) != len(raw):
            n_partial += 1
        gaps.append(sum(vals) / len(vals) - human)
        humans.append(human)
        policies.append(sum(vals) / len(vals))
        clusters.append(d.video_id)
    if not gaps:
        return None
    lo, hi = cluster_bootstrap_ci(gaps, clusters)
    return GapSummary(
        metric=metric,
        kind=kind,
        n_decisions=len(gaps),
        n_videos=len(set(clusters)),
        mean_gap=float(np.mean(gaps)),
        ci_lo=lo,
        ci_hi=hi,
        mean_human=float(np.mean(humans)),
        mean_policy=float(np.mean(policies)),
        n_decisions_of_kind=n_of_kind,
        n_human_undefined=n_human_undef,
        n_dropped_policy_undefined=n_dropped,
        n_partial_policy_undefined=n_partial,
    )


@dataclass
class RunResult:
    decisions: list[PairedDecision]
    excluded: list[dict[str, Any]] = field(default_factory=list)
    n_rows: int = 0
    n_rows_replayed: int = 0
    census: dict[str, Any] = field(default_factory=dict)


def run(
    rows: list[dict[str, Any]],
    policy: Any,
    device: torch.device,
    *,
    n_port_samples: int = N_PORT_SAMPLES,
    limit: int | None = None,
    verbose: bool = False,
) -> RunResult:
    """Replay every eligible corpus row. Never guesses placement order (D6)."""
    ports = PortSampler()
    result = RunResult(decisions=[], census=corpus_census(rows))
    for row in rows[: limit if limit is not None else len(rows)]:
        result.n_rows += 1
        prov = row.get("provenance", {})
        # The ledger carries the provenance flags a reader needs to judge WHY a
        # row failed: a mutually-illegal placement pair that nonetheless carries
        # ``passed_crosscheck: true`` is a finding about the cross-check.
        tag = {
            "video_id": row.get("video_id"),
            "game_index": row.get("game_index"),
            "passed_crosscheck": row.get("passed_crosscheck"),
            "order_source": prov.get("order_source"),
        }
        if not prov.get("placement_order_established", False):
            result.excluded.append({**tag, "reason": "placement_order_not_established"})
            continue
        try:
            result.decisions.extend(
                replay_row(row, policy, device, ports, n_port_samples=n_port_samples)
            )
        except CorpusReplayError as exc:
            result.excluded.append({**tag, "reason": f"replay_error: {exc}"})
            continue
        result.n_rows_replayed += 1
        if verbose:
            print(f"  replayed {tag['video_id']} g{tag['game_index']}", flush=True)
    return result


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

_HEADER = """# Human opening reference — champion vs ThePhantom on HIS boards

**READ THIS BEFORE READING ANY NUMBER.**

* **This is a REFUTATION instrument (spec D4).** ThePhantom is a strong human,
  not an oracle, so every divergence below is **UNSIGNED**: the policy differing
  from him is not evidence of a defect, and matching him is not evidence of
  strength. The instrument can *refute* "the openings are weak"; it can **never
  confirm** superiority. It measures **no win-probability** and runs **no
  rollouts**.
* **Ports are marginalised, not known (spec D2).** All corpus rows carry
  `board.ports = "OMITTED in v1"`. Ports ARE in the policy obs, so the policy
  here chooses under {K} deterministic sampled port assignments per board and the
  results are means over them. This is a real limitation. The **port-marginal
  churn** column quantifies it: it is the fraction of assignments that did not
  take the modal choice.
* **`opponent_strength` is a hardcoded constant across all 140 corpus rows
  (spec D6) — ZERO BITS.** It is not stratified on, not weighted by, and not
  cited as validation anywhere in this report.
* **Agreement rate is descriptive colour, NOT a gate (spec D5).** Move-agreement
  is an explicit non-goal of step6 §6.
* **Multiplicity — the CIs below are UNADJUSTED, and omitting p-values does NOT
  remove the multiplicity.** The pre-registered PRIMARY family is the D3 metric
  gap on settlement decisions ({primary}); every other row is exploratory. A 95%
  CI that excludes zero *is* a rejected two-sided test at alpha=0.05 — dropping
  the p-value drops the label, not the test. See "Multiplicity accounting" below
  for the row count, the number of exploratory CIs excluding zero, and how many
  would be expected to do so by chance alone. Treat each exploratory row as a
  hypothesis to re-test on fresh data, not as a finding.
* **Some metrics are UNDEFINED for some picks, and dropping those decisions
  selects on the outcome.** Every table carries the definedness columns
  (`human undef` / `policy undef` / `partial k`) and any metric with a nonzero
  drop count gets an explicit informative-missingness warning. Do not read a
  wide CI on a shrunken `n` as "no difference".
* **The D3 metrics are hypotheses about the axes the owner named, not
  established quantities.** Their definitions are restated in full under "Metric
  definitions (D3)" below, and live in the module docstring of
  `scripts/human_opening_reference.py`.
"""


def _fmt(x: float) -> str:
    return "nan" if math.isnan(x) else f"{x:+.3f}"


def _gap_table(summaries: list[GapSummary]) -> list[str]:
    out = [
        "| metric | decisions used | of kind | human undef | policy undef "
        "| partial k | videos | human mean | policy mean "
        "| gap (policy - human) | 95% CI (cluster) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for s in summaries:
        out.append(
            f"| `{s.metric}` | {s.n_decisions} | {s.n_decisions_of_kind} | "
            f"{s.n_human_undefined} | {s.n_dropped_policy_undefined} | "
            f"{s.n_partial_policy_undefined} | {s.n_videos} | {s.mean_human:.3f} | "
            f"{s.mean_policy:.3f} | {_fmt(s.mean_gap)} | "
            f"[{_fmt(s.ci_lo)}, {_fmt(s.ci_hi)}] |"
        )
    return out


def _missingness_warnings(summaries: list[GapSummary]) -> list[str]:
    """Name every metric whose ``n`` was cut by the POLICY's pick being undefined.

    This is the artifact class that killed the previous direction: a denominator
    chosen by the outcome. The reader must not be left to notice a silent ``n``
    drop on their own.

    ``build_report`` calls this on three DISJOINT metric groups (primary,
    battery, road), so the ``pair_exp_rolls_to_city`` paragraph below is gated on
    that metric actually being one of the warned rows in *this* group — it is a
    claim about that specific row and its `pair_city_self_sufficient` neighbour
    "in the same table", and it is nonsense attached to any other group.
    """
    hits = [s for s in summaries if s.has_informative_missingness]
    if not hits:
        return []
    out = [
        "",
        "> **INFORMATIVE MISSINGNESS — READ BEFORE THE ROWS ABOVE.** For the "
        "metrics listed here the comparison subset is SELECTED ON THE OUTCOME: a "
        "decision is dropped precisely because the policy's pick left the metric "
        "undefined, so the surviving rows are the ones where the policy did NOT "
        "fail. Their gaps are survivor-biased and must not be read as "
        '"no difference".',
        ">",
    ]
    for s in hits:
        out.append(
            f"> * `{s.metric}`: {s.n_decisions_of_kind} decisions of this kind, "
            f"{s.n_human_undefined} undefined for the human, "
            f"**{s.n_dropped_policy_undefined} dropped because EVERY port sample "
            f"of the policy's pick was undefined**, "
            f"{s.n_partial_policy_undefined} averaged over a partial k -> "
            f"n = {s.n_decisions}."
        )
    if any(s.metric == "pair_exp_rolls_to_city" for s in hits) and any(
        s.metric == "pair_city_self_sufficient" for s in summaries
    ):
        out.append(">")
        out.append(
            "> `pair_exp_rolls_to_city` is the load-bearing case. It is undefined "
            "unless the pair produces both ore and wheat, and "
            "`docs/plans/opening_sweep_results.md:456` already declares it NOT "
            "comparable across slices with different self-sufficiency rates. The "
            "`pair_city_self_sufficient` row in the same table is exactly that "
            "difference (the policy is materially LESS often self-sufficient), so "
            "its gap is computed on the subset where the policy did not fail. The "
            "honest reading of the pair is the `pair_city_self_sufficient` row, "
            "not the `pair_exp_rolls_to_city` row."
        )
    return out


def _ci_excludes_zero(s: GapSummary) -> bool:
    if math.isnan(s.ci_lo) or math.isnan(s.ci_hi):
        return False
    return s.ci_lo > 0.0 or s.ci_hi < 0.0


def build_report(
    result: RunResult,
    *,
    ckpt: Path,
    n_port_samples: int,
    raw_path: Path | None = None,
) -> str:
    dec = result.decisions
    settle = [d for d in dec if d.kind == "settlement"]
    road = [d for d in dec if d.kind == "road"]
    lines = [_HEADER.format(K=n_port_samples, primary=", ".join(f"`{m}`" for m in PRIMARY_METRICS))]
    lines.append("")
    lines.append("## Provenance")
    lines.append("")
    lines.append(f"* checkpoint under test: `{ckpt}`")
    lines.append(f"* corpus rows seen: {result.n_rows}; replayed: {result.n_rows_replayed}")
    lines.append(f"* paired decisions: {len(dec)} ({len(settle)} settlement, {len(road)} road)")
    lines.append(
        f"* distinct videos among the REPLAYED rows (bootstrap clusters): "
        f"{len({d.video_id for d in dec})} — this is a POST-exclusion count and is "
        "smaller than the corpus video census below whenever rows are excluded."
    )
    lines.append(f"* port assignments per board (K): {n_port_samples}")
    if raw_path is not None:
        lines.append(
            f"* per-decision raw records (every number here is re-aggregatable from "
            f"them): `{raw_path}` — one JSON object per paired decision, including "
            "the human's pick, the K policy picks, and every metric value for both."
        )
    lines.append("")
    lines.extend(_census_section(result.census))

    lines.append("## PRIMARY — D3 metric gap on settlement decisions")
    lines.append("")
    lines.append(
        "Pre-registered. A gap of 0 means the policy's pick scores exactly as "
        "ThePhantom's pick did on the same board state; the sign is NOT a "
        "quality ordering (D4)."
    )
    lines.append("")
    primary = [s for m in PRIMARY_METRICS if (s := summarise(dec, m, "settlement"))]
    lines.extend(_gap_table(primary))
    lines.extend(_missingness_warnings(primary))
    lines.append("")
    lines.extend(_primary_regime_section(settle, primary))
    lines.extend(_bottom_line_section(dec, primary))
    lines.extend(_pair_self_sufficiency_section(settle))

    lines.append("## Exploratory — existing battery on settlement decisions")
    lines.append("")
    lines.append(
        "**`denial_reach` is the flagged sharpening of the PRIMARY `denial` row, "
        'not a replacement for it** — see "Metric definitions (D3)": the primary '
        "row implements spec D3 literally, and the restricted variant is reported "
        "here for the owner to rule on."
    )
    lines.append("")
    lines.append(
        "**Every `pair_*` row scores a TEACHER-FORCED HYBRID, not the policy's own "
        "opening pair.** The pair metrics are only defined at the second-settlement "
        "decision, and by D1's pairing the first settlement there is *ThePhantom's* "
        '(the policy never chose it). So a `pair_*` "policy mean" is the score of '
        "(ThePhantom's settlement #1, policy's settlement #2): it measures how the "
        "policy COMPLETES a human first settlement, and it cannot support any claim "
        "about the pairs the policy builds when it opens for itself."
    )
    lines.append("")
    exploratory_settlement = (
        tuple(m for m in CONTESTED_METRICS if m not in PRIMARY_METRICS)
        + SETTLEMENT_METRICS
        + PAIR_METRICS
    )
    battery = [s for m in exploratory_settlement if (s := summarise(dec, m, "settlement"))]
    lines.extend(_gap_table(battery))
    lines.extend(_missingness_warnings(battery))
    lines.append("")

    lines.append("## Exploratory — road decisions")
    lines.append("")
    road_sums = [s for m in ROAD_METRICS + CUT_METRICS if (s := summarise(dec, m, "road"))]
    lines.extend(_cut_vulnerability_note(road_sums, road))
    lines.append("")
    lines.extend(_gap_table(road_sums))
    lines.extend(_missingness_warnings(road_sums))
    lines.append("")
    lines.extend(_multiplicity_section(primary, battery + road_sums))

    lines.append("## Port-marginal churn (quantifies the D2 limitation)")
    lines.append("")
    lines.append("| decision kind | decisions | mean churn | fraction with churn > 0 |")
    lines.append("|---|---:|---:|---:|")
    for name, group in (("settlement", settle), ("road", road)):
        if not group:
            continue
        churns = [d.churn() for d in group]
        lines.append(
            f"| {name} | {len(group)} | {np.mean(churns):.3f} | "
            f"{np.mean([1.0 if c > 0 else 0.0 for c in churns]):.3f} |"
        )
    lines.append("")

    lines.append("## Descriptive only — exact-match agreement (NOT A GATE, D5)")
    lines.append("")
    lines.append("| decision kind | decisions | mean agreement | mean candidates |")
    lines.append("|---|---:|---:|---:|")
    for name, group in (("settlement", settle), ("road", road)):
        if not group:
            continue
        lines.append(
            f"| {name} | {len(group)} | {np.mean([d.agreement() for d in group]):.3f} | "
            f"{np.mean([d.n_candidates for d in group]):.1f} |"
        )
    lines.append("")

    lines.append("## Exclusion ledger")
    lines.append("")
    if not result.excluded:
        lines.append("None.")
    else:
        lines.append("| video_id | game | passed_crosscheck | order_source | reason |")
        lines.append("|---|---:|---|---|---|")
        for e in result.excluded:
            lines.append(
                f"| `{e['video_id']}` | {e['game_index']} | "
                f"{e.get('passed_crosscheck')} | `{e.get('order_source')}` | "
                f"{e['reason']} |"
            )
    lines.extend(_crosscheck_finding(result))
    lines.append("")
    lines.extend(_definitions_section())
    return "\n".join(lines)


def _census_section(census: dict[str, Any]) -> list[str]:
    """The spec-D6 census, recomputed, with the provenance caveats (AC/D6)."""
    if not census:
        return []
    exp = census.get("expected", {})
    lines = ["## Corpus census (spec D6 — recomputed, not quoted)", ""]
    lines.append("| quantity | measured | spec D6 expects |")
    lines.append("|---|---:|---:|")
    for key in ("rows", "videos", "unique_layouts", "winners_known"):
        lines.append(f"| {key} | {census.get(key)} | {exp.get(key)} |")
    lines.append("")
    if census.get("matches_expected"):
        lines.append(
            "The census reproduces spec D6 exactly, so the population behind every "
            "number below is the population the spec was written against."
        )
    else:
        lines.append(
            "**WARNING — the census does NOT match spec D6.** The corpus population "
            "has changed, so every number in this report is computed on a different "
            "population than the spec was written against. Re-ratify before reading."
        )
    lines.append("")
    lines.append(
        "Two provenance facts that bear on how much to trust the placement ORDER "
        "(the load-bearing input to the whole pairing) — disclosed, not filtered:"
    )
    lines.append("")
    counts = census.get("order_source_counts", {})
    lines.append(
        "* `provenance.order_source` across the corpus: "
        + ", ".join(f"`{k}` x{v}" for k, v in counts.items())
        + ". Glyph-anchor-only ordering is an OPEN owner decision, so a result "
        "resting on it inherits that open question."
    )
    synth = census.get("synthetic_video_rows", [])
    lines.append(
        "* rows whose `video_id` is a SYNTHETIC placeholder (the hardcoded default "
        f"in `scripts/vlm_spike.py`): {', '.join(f'`{s}`' for s in synth) or 'none'}"
        + (
            ". Such a row is not a real video: it still contributes decisions and "
            "counts as one bootstrap cluster."
            if synth
            else "."
        )
    )
    lines.append("")
    return lines


def _primary_regime_section(settle: list[PairedDecision], primary: list[GapSummary]) -> list[str]:
    """Disclose the (seat, decision) regimes the PRIMARY family pools.

    ``contested_race_*`` and ``denial`` are defined relative to the OPPONENT's
    settlement set, so their scale differs systematically with how many
    settlements he has already placed. The mean paired gap is still unbiased for
    "average gap over decisions", but the mix must be visible.
    """
    if not settle:
        return []
    buckets: dict[tuple[int, int, int], int] = defaultdict(int)
    for d in settle:
        buckets[(d.agent_seat, d.decision_index, d.opp_settlements)] += 1
    lines = [
        "### Why the PRIMARY `n` is smaller than the settlement-decision count",
        "",
        "The D3 contested/denial metrics are UNDEFINED when the opponent has not "
        "placed yet, which is exactly the seat-0 first settlement — those decisions "
        "are excluded by construction, not lost to failure. The remaining regimes "
        "differ in how many opponent settlements the metrics are measured against, "
        "so the pooled mean is a mean over a MIX:",
        "",
        "| agent seat | decision index | opponent settlements | decisions | in PRIMARY? |",
        "|---:|---:|---:|---:|---|",
    ]
    for (seat, d_idx, opp_n), n in sorted(buckets.items()):
        lines.append(
            f"| {seat} | {d_idx} | {opp_n} | {n} | "
            f"{'yes' if opp_n > 0 else 'NO (undefined by construction)'} |"
        )
    lines.append("")
    lines.extend(_within_regime_primary_gaps(settle, primary))
    return lines


def _within_regime_primary_gaps(
    settle: list[PairedDecision], primary: list[GapSummary]
) -> list[str]:
    """Show the PRIMARY gap WITHIN each pooled regime, not only pooled.

    A pooled null can be the average of offsetting signs — the aggregation error
    the spec blames for the previous direction. Point estimates only (no CIs: the
    per-regime `n` does not support a cluster bootstrap), and they are there to
    check for sign cancellation, not to be interpreted on their own.
    """
    metrics = [s.metric for s in primary]
    if not metrics or not settle:
        return []
    regimes = sorted(
        {
            (d.agent_seat, d.decision_index, d.opp_settlements)
            for d in settle
            if d.opp_settlements > 0
        }
    )
    if len(regimes) < 2:
        return []
    lines = [
        "The pooled gap is only readable if it is not an average of offsetting "
        "signs, so here is the same PRIMARY gap computed WITHIN each pooled "
        "regime. Point estimates only — the per-regime `n` does not support a "
        "cluster bootstrap, so these are a sign-cancellation check, not findings:",
        "",
        "| regime (seat / decision / opp settlements) | n | "
        + " | ".join(f"`{m}` gap" for m in metrics)
        + " |",
        "|---|---:|" + "---:|" * len(metrics),
    ]
    for seat, d_idx, opp_n in regimes:
        group = [
            d
            for d in settle
            if (d.agent_seat, d.decision_index, d.opp_settlements) == (seat, d_idx, opp_n)
        ]
        cells = []
        n_used = 0
        for m in metrics:
            vals = [g[m] for d in group if (g := d.gaps()) and m in g]
            n_used = max(n_used, len(vals))
            cells.append(_fmt(float(np.mean(vals))) if vals else "n/a")
        lines.append(
            f"| seat {seat} / decision {d_idx} / opp {opp_n} | {n_used} | "
            + " | ".join(cells)
            + " |"
        )
    lines.append("")
    return lines


def _bottom_line_section(dec: list[PairedDecision], primary: list[GapSummary]) -> list[str]:
    """State what the pre-registered metric actually returned (D4).

    The instrument exists to REFUTE "the openings are weak". A report that makes
    the reader re-derive its own primary answer is not doing that job.
    """
    if not primary:
        return []
    straddling = [s for s in primary if not _ci_excludes_zero(s)]
    lines = ["### Bottom line on the pre-registered question", ""]
    if len(straddling) == len(primary):
        lines.append(
            "**All "
            f"{len(primary)} primary CIs straddle zero: there is NO DETECTABLE "
            "DIFFERENCE between the policy's picks and ThePhantom's on his own "
            "contested-middle / denial axes, on his own boards, at his own decision "
            "points.** Under D4 that is the refutation direction: this instrument "
            'found no support for "the openings are weak" on the axes the owner '
            "named. It is NOT evidence of superiority, it is not a win-probability, "
            "and 'no detectable difference' at this `n` is not 'identical'."
        )
        lines.append("")
        lines.append(
            "**How tight is the null? (D4's refutation logic rests entirely on "
            "this.)** A null only licenses closing a thread if it EXCLUDES gaps big "
            "enough to matter, so here is the precision, as the **two-sided "
            "exclusion bound** `max(|CI_lo|, |CI_hi|)` expressed against the human's "
            "own level on the same metric. It is deliberately NOT the CI half-width: "
            "these CIs are not centred on zero, and a half-width would understate "
            "the magnitude the interval actually still admits — in the direction "
            "that flatters the refutation."
        )
        lines.append("")
        lines.append(
            "| metric | human mean | exclusion bound `max(\\|CI_lo\\|, \\|CI_hi\\|)` | "
            "excludes gaps larger than |"
        )
        lines.append("|---|---:|---:|---:|")
        fracs: list[float] = []
        for s in primary:
            bound = max(abs(s.ci_lo), abs(s.ci_hi))
            frac = abs(bound / s.mean_human) if s.mean_human else float("nan")
            if not math.isnan(frac):
                fracs.append(frac)
            pct = "n/a" if math.isnan(frac) else f"~{100 * frac:.1f}% of the human level"
            lines.append(f"| `{s.metric}` | {s.mean_human:.3f} | {bound:.3f} | {pct} |")
        lines.append("")
        span = (
            f"here {100 * min(fracs):.1f}% to {100 * max(fracs):.1f}% of the human's "
            "own level on the axis"
            if fracs
            else "see the last column"
        )
        lines.append(
            "So this run rules out gaps whose MAGNITUDE is at or above the bound in "
            f"the last column ({span}); anything SMALLER than that is inside the "
            "noise and would need a bigger corpus (more videos, i.e. more bootstrap "
            "clusters — not more port samples, which do not add independent "
            "information)."
        )
    else:
        named = ", ".join(f"`{s.metric}` {_fmt(s.mean_gap)}" for s in primary)
        lines.append(
            f"**{len(primary) - len(straddling)} of {len(primary)} primary CIs "
            "EXCLUDE zero**, so the primary family does detect a difference on the "
            f"owner's own axes: {named}. Under D4 the SIGN is still unsigned — "
            "ThePhantom is not an oracle — so this is a divergence to explain, not a "
            "verdict on quality."
        )
    lines.append("")
    lines.append(
        "Primary rows verbatim: "
        + "; ".join(
            f"`{s.metric}` {_fmt(s.mean_gap)} [{_fmt(s.ci_lo)}, {_fmt(s.ci_hi)}] "
            f"(n={s.n_decisions})"
            for s in primary
        )
        + "."
    )
    lines.append("")
    return lines


def _pair_self_sufficiency_section(settle: list[PairedDecision]) -> list[str]:
    """Surface the one exploratory row that bears on WHY this was commissioned.

    ``pair_city_self_sufficient`` is the only paired, same-position,
    non-reference-class evidence here about the suspected opening ore/city blind
    spot, so it is stated up front with its MECHANISM rather than left as a
    parenthetical inside a missingness warning. It is still EXPLORATORY (see
    "Multiplicity accounting") and still a TEACHER-FORCED pair (ThePhantom's
    settlement #1 + the chooser's #2).
    """
    s = summarise(settle, "pair_city_self_sufficient", "settlement")
    if s is None or s.n_decisions == 0:
        return []
    group = [d for d in settle if d.human_metrics.get("pair_city_self_sufficient") is not None]

    def _mech(per_decision: list[list[tuple[float, float]]]) -> tuple[float, float]:
        """Expected count of (ore but NO wheat) and (no ore at all) pairs."""
        no_wheat = 0.0
        no_ore = 0.0
        for samples in per_decision:
            if not samples:
                continue
            n = len(samples)
            no_wheat += sum(1 for o, w in samples if o > 0 and w == 0) / n
            no_ore += sum(1 for o, _ in samples if o == 0) / n
        return no_wheat, no_ore

    def _human_samples(d: PairedDecision) -> list[tuple[float, float]]:
        ore = d.human_metrics.get("pair_ore_pips")
        wheat = d.human_metrics.get("pair_wheat_pips")
        return [] if ore is None or wheat is None else [(ore, wheat)]

    def _policy_samples(d: PairedDecision) -> list[tuple[float, float]]:
        ore = d.policy_metrics.get("pair_ore_pips", [])
        wheat = d.policy_metrics.get("pair_wheat_pips", [])
        return [(o, w) for o, w in zip(ore, wheat, strict=True) if o is not None and w is not None]

    human_nw, human_no = _mech([_human_samples(d) for d in group])
    policy_nw, policy_no = _mech([_policy_samples(d) for d in group])
    return [
        "### The exploratory row that bears directly on the owner's original question",
        "",
        "The feature was commissioned because of a suspected **opening ore/city "
        "blind spot**. The only paired, same-position, non-reference-class row here "
        f"that speaks to it is `pair_city_self_sufficient`: gap {_fmt(s.mean_gap)} "
        f"[{_fmt(s.ci_lo)}, {_fmt(s.ci_hi)}] (human {s.mean_human:.3f} vs policy "
        f"{s.mean_policy:.3f}, n={s.n_decisions}). It is EXPLORATORY, not "
        "pre-registered, and it scores a TEACHER-FORCED pair (ThePhantom's "
        "settlement #1 + the chooser's #2), so it is a hypothesis for a fresh "
        "sample — but it is the decision-relevant number this run produced, and it "
        "is stated here rather than buried in a footnote about another metric.",
        "",
        "**The mechanism INVERTS the 'ore-blind' framing.** On the same decisions "
        "the policy takes MORE ore, not less (`pair_ore_pips` in the exploratory "
        "table), yet fails self-sufficiency more often, and the failures are the "
        "wrong KIND: `pair_city_self_sufficient` needs ore > 0 AND wheat > 0.",
        "",
        "| failure mode of the pair | human (expected count) | policy (expected count) |",
        "|---|---:|---:|",
        f"| has ore but ZERO wheat | {human_nw:.1f} | {policy_nw:.1f} |",
        f"| no ore at all | {human_no:.1f} | {policy_no:.1f} |",
        "",
        f"(of {s.n_decisions} paired second-settlement decisions; policy counts are "
        "expectations over the K port assignments.) The ore-absent rates are "
        "comparable — the deficit is concentrated in **ore-bearing pairs with no "
        "wheat**, i.e. a wheat/complementarity omission, NOT an ore deficit. That "
        "is the opposite of the hypothesis that motivated the build, and the "
        "follow-up it suggests is about pairing ore with wheat, not about taking "
        "more ore.",
        "",
    ]


def _cut_vulnerability_note(road_sums: list[GapSummary], road: list[PairedDecision]) -> list[str]:
    """Say what `cut_vulnerability` actually measured, without hedging.

    Half of the zero column is a STRUCTURAL zero, not an empirical one: the setup
    loop is a fixed (settlement, road, settlement, road) sequence, so
    ``decision_index == 1`` IS the first road decision, where the agent owns no
    prior road, the induced subgraph is the single candidate edge, and 0.0 is
    true by construction (pinned by
    ``test_cut_vulnerability_counts_articulation_points``). Reporting the full
    denominator as empirical would overstate the evidence.
    """
    cut = next((s for s in road_sums if s.metric == "cut_vulnerability"), None)
    if cut is None:
        return []
    n_structural = sum(1 for d in road if d.decision_index == 1)
    n_empirical = len(road) - n_structural
    if cut.mean_human == 0.0 and cut.mean_policy == 0.0 and cut.mean_gap == 0.0:
        return [
            "**`cut_vulnerability` came out IDENTICALLY ZERO on every decision in "
            f"this run.** All {cut.n_decisions} human evaluations and all their "
            "policy counterparts are exactly 0.0, so the column carries no variance "
            "to compare here — but that denominator is **not all empirical**: "
            f"{n_structural} of the {cut.n_decisions} are the FIRST-road decision, "
            "where the chooser owns no prior road, the induced subgraph is the "
            "single candidate edge, and 0.0 is true **BY CONSTRUCTION** (a "
            "1-edge graph has no articulation point; the module's own unit test "
            "pins it). The honest empirical denominator is the "
            f"**{n_empirical} SECOND-road decisions**, where a self-cuttable pair "
            "was reachable and neither chooser built one. On those the null is "
            "genuinely EMPIRICAL, not a theorem: the distance rule forbids "
            "ADJACENT settlements, but two of "
            "my settlements may sit at graph distance 2 sharing a middle vertex, and "
            "if both setup roads point at that shared vertex the road subgraph is a "
            "3-vertex PATH whose middle IS an articulation point. That double-back "
            "is legal and available — verified against the engine masks on corpus "
            "row `b_NXNI-l0kI` g6 (agent settlements 15 and 14, shared neighbour 13, "
            "recorded first road 13-15, and edge 13-14 a legal second-road "
            "candidate; scoring that pair returns `cut_vulnerability` = 1.0). So the "
            f"reading is: **on the {n_empirical} decisions where a self-cut was even "
            "expressible, neither ThePhantom nor the policy ever built a "
            "self-cuttable road pair**, which is a finding about both choosers, and "
            "the articulation-point family is NOT shown to be incapable of signal "
            "outside the setup regime. Note separately that this definition "
            "(articulation points of MY OWN roads) cannot express the 'plow' the "
            "owner described — an OPPONENT road severing my longest path — which "
            "would need the opponent's road network and a longest-trail "
            "recomputation.",
        ]
    return [
        "`cut_vulnerability` is measured over setup road decisions only (at most "
        "two owned roads), and it scores MY OWN articulation points — it does not "
        f"express the opponent-road 'plow'. Note that {n_structural} of the "
        f"{cut.n_decisions} evaluations are FIRST-road decisions, where 0.0 is "
        f"forced by construction; only the {n_empirical} second-road decisions "
        "carry information. Read it as descriptive colour."
    ]


def _multiplicity_section(primary: list[GapSummary], exploratory: list[GapSummary]) -> list[str]:
    """Count the family and the CIs excluding zero (the honest version of D4)."""
    n_exp = len(exploratory)
    hits = [s for s in exploratory if _ci_excludes_zero(s)]
    up = [s for s in hits if s.mean_gap > 0]
    down = [s for s in hits if s.mean_gap < 0]
    expected_fp = 0.05 * n_exp
    return [
        "## Multiplicity accounting",
        "",
        f"* rows reported: {len(primary) + n_exp} = {len(primary)} pre-registered "
        f"primary + {n_exp} exploratory.",
        f"* exploratory CIs excluding zero: **{len(hits)}** of {n_exp} "
        f"({', '.join(f'`{s.metric}`' for s in hits) or 'none'}).",
        f"* expected to exclude zero by chance alone at alpha=0.05 if the policy and "
        f"ThePhantom were identical on every axis: ~{expected_fp:.1f}.",
        "* the CIs are **UNADJUSTED**. A 95% CI excluding zero is a rejected "
        "two-sided test; not printing a p-value removes the label, not the "
        "multiplicity. With this many rows, individual exploratory 'findings' are "
        "hypotheses for a fresh sample — the pre-registered family is the only one "
        "this run is entitled to interpret.",
        "* **but the expected-false-positive count is the wrong frame for a "
        "COHERENT block of rows.** "
        f"{len(hits)} of {n_exp} exploratory CIs exclude zero, far above the ~"
        f"{expected_fp:.1f} chance expectation, and they are not scattered — "
        f"{len(up)} point UP ("
        + (", ".join(f"`{s.metric}`" for s in up) or "none")
        + f") and {len(down)} point DOWN ("
        + (", ".join(f"`{s.metric}`" for s in down) or "none")
        + "), which on this battery reads as immediate production up against "
        "expansion room and city complementarity down. A per-row chance count says "
        "nothing about a directional block like that. The honest statement is "
        "therefore TWO-PART: the pre-registered "
        "contested-middle / denial family shows **no detectable difference**, while "
        "the exploratory battery shows a **coherent, unvalidated direction** — the "
        "policy buys immediate pips at the cost of expansion room and city "
        "complementarity. The second half is a hypothesis this run cannot confirm "
        "(exploratory, unadjusted, `pair_*` teacher-forced, human is not an oracle), "
        "but it is the strongest signal here on the owner's original question and "
        "must not be read as refuted by the primary null — the primary family "
        "measures different axes.",
        "",
    ]


def _classify_replay_error(reason: str) -> str:
    """Bucket one ``CorpusReplayError`` message into its CAUSE.

    ``CorpusReplayError`` is raised from several structurally different places
    (malformed hex block, non-snake ``draft_order``, scripted-opponent
    exhaustion / main-turn leak, an illegal recorded placement, a policy argmax
    outside the legal set, a port-invariance violation). Only the ILLEGAL
    RECORDED PLACEMENT bucket says anything about what ``passed_crosscheck``
    does or does not validate, so the report must not publish one hardcoded
    diagnosis for whatever a future corpus happens to fail on.
    """
    if "ILLEGAL" in reason:
        return "illegal_recorded_placement"
    if "hex" in reason or "port assignment" in reason or "missing-seam" in reason:
        return "malformed_board_layout"
    if "draft_order" in reason or "not found in draft_order" in reason:
        return "malformed_draft_order"
    if "opening is not 2 settlements + 2 roads" in reason:
        return "malformed_opening_script"
    if "exhausted" in reason or "main-turn leak" in reason:
        return "opponent_script_exhausted"
    if "policy argmax" in reason:
        return "policy_argmax_outside_legal_set"
    if "across port samples" in reason:
        return "port_invariance_violation"
    return "other"


#: Human-readable gloss per ``_classify_replay_error`` bucket.
_REPLAY_ERROR_CAUSES = {
    "illegal_recorded_placement": (
        "a RECORDED placement (the agent's own or the scripted opponent's) is "
        "geometrically illegal at the point it is replayed"
    ),
    "malformed_board_layout": "the row's hex/port block is not a well-formed layout",
    "malformed_draft_order": "the row's `draft_order` is not a 1v1 snake draft",
    "malformed_opening_script": "the row's opening is not 2 settlements + 2 roads",
    "opponent_script_exhausted": ("the scripted opponent ran out of placements (a main-turn leak)"),
    "policy_argmax_outside_legal_set": (
        "the policy's argmax fell outside the legal set (an INSTRUMENT bug, not a corpus defect)"
    ),
    "port_invariance_violation": (
        "the candidate set or the human's metrics moved across port samples (an "
        "INSTRUMENT bug, not a corpus defect)"
    ),
    "other": "an unclassified replay fidelity violation",
}


def _crosscheck_finding(result: RunResult) -> list[str]:
    """Report the replay-exclusion causes ACTUALLY observed, and only conclude
    about ``passed_crosscheck`` when an observed cause supports it.

    The corpus-integrity conclusion below is specifically about mutual LEGALITY
    of recorded placements, so it is emitted only for the
    ``illegal_recorded_placement`` bucket, and only when such rows carry
    ``passed_crosscheck: true``. Every other observed cause is stated plainly
    with no conclusion attached.
    """
    replay_errors = [
        e for e in result.excluded if str(e.get("reason", "")).startswith("replay_error")
    ]
    if not replay_errors:
        return []
    by_cause: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for e in replay_errors:
        by_cause[_classify_replay_error(str(e.get("reason", "")))].append(e)

    out = [
        "",
        f"**Replay exclusions by CAUSE** ({len(replay_errors)} order-established "
        "rows failed replay). `CorpusReplayError` is raised from several "
        "structurally different places, so the causes are reported as observed "
        "rather than assumed:",
        "",
    ]
    for cause in sorted(by_cause, key=lambda c: (-len(by_cause[c]), c)):
        rows = by_cause[cause]
        n_cc = sum(1 for e in rows if e.get("passed_crosscheck"))
        out.append(
            f"* `{cause}` — {len(rows)} row(s), {n_cc} with "
            f"`passed_crosscheck: true`: {_REPLAY_ERROR_CAUSES[cause]}."
        )
    out.append("")

    illegal = by_cause.get("illegal_recorded_placement", [])
    crosschecked = [e for e in illegal if e.get("passed_crosscheck")]
    if crosschecked:
        out.append(
            f"**Finding, not just bookkeeping:** {len(illegal)} of those rows fail "
            "because a RECORDED placement is geometrically illegal (mutually "
            f"adjacent settlements), and {len(crosschecked)} of them carry "
            "`passed_crosscheck: true`. The corpus cross-check therefore does NOT "
            "validate mutual legality of the recorded placements — "
            "`passed_crosscheck` is a weaker guarantee than its name suggests. "
            "That matters for anything downstream (e.g. an outcome-anchored "
            "scoreboard) that reads the flag as 'this row is consistent'."
        )
    elif illegal:
        out.append(
            f"{len(illegal)} row(s) fail on an illegal recorded placement, but none "
            "of them carry `passed_crosscheck: true`, so this run says nothing "
            "about what the cross-check does or does not validate."
        )
    else:
        out.append(
            "No row failed on an illegal recorded placement in this run, so this "
            "run draws NO conclusion about what `passed_crosscheck` validates."
        )
    return out


def _definitions_section() -> list[str]:
    """Restate the D3 definitions in the deliverable itself.

    A reader must not have to open a source file to learn what a reported number
    means — especially where the implementation deliberately deviates from the
    spec's literal wording.
    """
    return [
        "",
        "## Metric definitions (D3)",
        "",
        "* **`contested_race_count` / `contested_race_pips`** — over the vertices "
        "that would still be legal settlement sites after my candidate is placed "
        "and that are within road-distance 4 of BOTH my settlements (candidate "
        "included) and the opponent's, the count of those I reach strictly first, "
        "and their pip weight. Undefined (`null`, excluded) when the opponent has "
        "no settlement yet.",
        "* **`denial`** (PRE-REGISTERED, literal spec D3) — "
        "`|legal(occ)| - |legal(occ + v)|`: the reduction in the opponent's legal "
        "future-settlement set caused by my placement, **unrestricted**, exactly "
        "as spec D3 words it.",
        "* **`denial_reach`** (EXPLORATORY sharpening — reported ALONGSIDE, not "
        "instead of, `denial`) — the same shrinkage restricted to the opponent's "
        "*road-reachable* set: `|legal(occ) ∩ reach| - |legal(occ + v) ∩ reach|`, "
        "`reach` = within road-distance 4 of his settlements. Why it is offered: "
        "the unrestricted `denial` is close to 'how many legal neighbours does the "
        "candidate have', i.e. largely board supply rather than a policy choice — "
        "the failure mode that killed the previous opening direction. **OWNER "
        "DECISION PENDING:** the spec is owner-delegated and `denial` is "
        "pre-registered, so this variant is flagged rather than substituted; if "
        "the owner prefers it, re-designating the primary row is a one-line "
        "change (`PRIMARY_METRICS`).",
        "* **Both denial rows measure only the DISTANCE-RULE channel.** `denial` "
        "counts sites the candidate removes from the legal set via the distance "
        "rule, and `denial_reach` computes its `reach` from the PRE-candidate "
        "blocking set on both sides — so the other channel, my settlement WALLING "
        "the opponent's road expansion *through* that vertex (plausibly what "
        "'denial of the opponent's expansion' means in 1v1), is **not counted by "
        "either row**. That channel is unmeasured by this instrument.",
        "* **`cut_vulnerability`** — articulation points of the vertex-induced "
        "graph of MY OWN roads including the candidate. See the road section: it "
        "came out identically zero on every decision in this run (an EMPIRICAL "
        "null, with a legal counterexample available), and it does not model the "
        "opponent-road 'plow'.",
        "",
    ]


def summary_json(result: RunResult, *, ckpt: Path, n_port_samples: int) -> dict[str, Any]:
    dec = result.decisions
    regimes: dict[tuple[int, int, int], int] = defaultdict(int)
    for d in dec:
        if d.kind == "settlement":
            regimes[(d.agent_seat, d.decision_index, d.opp_settlements)] += 1
    out: dict[str, Any] = {
        "checkpoint": str(ckpt),
        "n_port_samples": n_port_samples,
        "n_rows": result.n_rows,
        "n_rows_replayed": result.n_rows_replayed,
        "n_decisions": len(dec),
        #: POST-exclusion cluster count (smaller than ``census.videos``).
        "n_videos": len({d.video_id for d in dec}),
        "corpus_census": result.census,
        "excluded": result.excluded,
        "scope": (
            "REFUTATION ONLY — unsigned divergence, no win-probability; ports "
            "marginalised over K deterministic assignments; opponent_strength "
            "is a constant and carries zero bits; agreement rate is descriptive."
        ),
        "primary_metrics": list(PRIMARY_METRICS),
        #: Flagged for the owner rather than decided unilaterally: ``denial``
        #: implements spec D3 literally (unrestricted), and the road-reach
        #: restricted variant ships ALONGSIDE it as the exploratory
        #: ``denial_reach`` row. Re-designating the primary is a one-line change.
        "owner_decisions_pending": [
            "denial: literal spec-D3 (unrestricted) is PRIMARY; denial_reach "
            "(road-distance <= 4 of the opponent's settlements) is reported as "
            "exploratory. Both measure only the distance-rule channel; the "
            "road-blocking channel of denial is unmeasured."
        ],
        #: (agent_seat, decision_index, opponent settlements) -> count, so the
        #: regimes the primary family pools are machine-readable too.
        "settlement_decision_regimes": {
            f"seat{seat}.decision{d_idx}.opp{opp_n}": n
            for (seat, d_idx, opp_n), n in sorted(regimes.items())
        },
        "gaps": {},
    }
    for kind, names in (
        ("settlement", CONTESTED_METRICS + SETTLEMENT_METRICS + PAIR_METRICS),
        ("road", CUT_METRICS + ROAD_METRICS),
    ):
        for name in names:
            s = summarise(dec, name, kind)
            if s is not None:
                out["gaps"][f"{kind}.{name}"] = asdict(s)
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--port-samples", type=int, default=N_PORT_SAMPLES)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device. MUST be cpu (spec D7 — the M1's accelerator is training); "
        "anything else is rejected.",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    # D7 / AC8: never contend with, and never write anywhere near, the live
    # training run. Both checks run BEFORE any directory is created.
    torch.set_num_threads(1)
    device = check_device(torch.device(args.device))
    out_dir = check_write_target(args.out_dir)
    raw_dir = check_write_target(args.raw_dir)

    policy = opening_sweep.load_policy(args.ckpt, device)
    rows = load_rows(args.corpus)
    result = run(
        rows,
        policy,
        device,
        n_port_samples=args.port_samples,
        limit=args.limit,
        verbose=args.verbose,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / "raw.json").write_text(json.dumps([asdict(d) for d in result.decisions], indent=1))
    summary = summary_json(result, ckpt=args.ckpt, n_port_samples=args.port_samples)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    (out_dir / "report.md").write_text(
        build_report(
            result,
            ckpt=args.ckpt,
            n_port_samples=args.port_samples,
            raw_path=(raw_dir / "raw.json").relative_to(REPO),
        )
    )
    if not result.census.get("matches_expected", False):
        print(
            "WARNING: corpus census does not match spec D6 "
            f"{EXPECTED_CENSUS} (measured "
            f"{ {k: result.census.get(k) for k in EXPECTED_CENSUS} })",
            flush=True,
        )
    print(
        f"replayed {result.n_rows_replayed}/{result.n_rows} rows -> "
        f"{len(result.decisions)} paired decisions; "
        f"{len(result.excluded)} excluded. Wrote {out_dir}/report.md"
    )


if __name__ == "__main__":
    main()
