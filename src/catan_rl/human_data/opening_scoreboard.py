"""Read-only paired ΔAUC opening scoreboard (new pointer-arch vs v11).

Step6 (`docs/plans/v2/step6_human_corpus.md`) accept-gate clause (b), spec
`.claude/veriloop/specs/opening-scoreboard.md` (RATIFIED 2026-07-22). This module
answers ONE question, computed **paired** for the new pointer-arch checkpoint and
the frozen v11 baseline over ThePhantom's scoreboard-eligible parsed openings:

    "Does the new arch judge ThePhantom's *winning* openings at least as well as
     v11?" (discrimination, not level).

**The metric is rank-based ΔAUC — never a raw-value comparison (D1).** The two
architectures' value heads are separately trained with their own value-normalizers,
so comparing raw ``v̂`` across arches is INVALID and there is NO squash anywhere in
this module (the only V→common-scale map ``σ(3.22·v̂−1.14)`` was fit to a third,
v8-era policy). AUC is a Mann–Whitney rank statistic → scale-invariant → the
cross-arch scale problem disappears. Primary statistic::

    ΔAUC = AUC(new-arch v̂, realized outcome) − AUC(v11 v̂, realized outcome)

one observation per eligible game (the mean over the K=8 port-marginal layouts).

**Port-marginal reading (D4).** A ``GameRecord`` never records the port assignment,
so each game is scored under ``K = 8`` layouts that *marginalize the unrecorded port
assignment over the 9 fixed port slots* (NOT the 12 D6 board symmetries — ports are
unrecorded, the board is known). One observation per game = the mean ``v̂`` over the
8 layouts, per leg. The per-record-per-layout seed is pinned (step6 §1)::

    seed(video_id, game_index, k) =
        int.from_bytes(sha256(f"{video_id}:{game_index}:{k}").digest()[:8], "big")

**Paired-identical-state (D2, the core correctness guard).** Per eligible game ×
layout k, ONE :func:`~catan_rl.human_data.engine_bridge.rebuild_env` produces a
single shared engine state; its :func:`~catan_rl.human_data.engine_bridge.engine_state_hash`
is captured ONCE and feeds BOTH the new-schema obs (native ``ObsEncoder`` → new
value head) and the legacy-schema obs (vendored ``LegacyObsEncoder`` → v11 value
head, exactly as ``cross_arch.CrossArchEnv._sample_snapshot_action`` builds it). The
run is gated on :func:`~catan_rl.eval.engine_parity.assert_engine_parity` (v11 can
only be loaded through the vendored pre-fork arch; the engine must match the
pre-fork tree that arch was written against). The adapter synthesizes the 5 fields a
``GameRecord`` lacks IDENTICALLY for both legs: ports (the K=8 marginal), robber =
desert hex, hands derived from ``settlements[1]`` (to satisfy the bridge's
grant/post-condition asserts), and ``opp_kind``/``opp_policy_id`` pinned to one
fixed sentinel (they enter the obs → affect ``v̂``). Every ``v̂`` routes through
``rebuild_env`` so the bridge's spiral-consistency gate + finite-bank/conservation/
hand-tracker invariants all fire — there is no bypass path.

**Measurement-first (D3).** This slice defers *enforcement*, not *recording*:
- The by-video EVAL-A/HOLDOUT-B split is committed (``split_by_video.json``) and the
  metric is computed on **EVAL-A only**; HOLDOUT-B stays virgin for the later formal
  GATE-A verdict once the corpus passes the 100-game floor.
- The pre-registered sign + tolerance live in the committed split file's
  ``pre_registration`` block AND mirror :data:`PRE_REGISTRATION` here; the module
  asserts they agree so a below-floor number can never be laundered into a verdict.
- Below the floor (n < 100 per cell) the tool emits the metric + CI but stamps a
  ``BELOW FLOOR`` / ``MEASUREMENT-ONLY`` / ``NO VERDICT`` banner and hard-suppresses
  the GO/NO-GO path (:func:`verdict`).
- Input provenance is recorded as ``sha256`` hashes (recording, not refuse-to-run
  enforcement): this module, ``engine_bridge.py`` + the ``engine/board.py`` injection
  surface, the corpus JSONL, both checkpoints, and the v11 pre-fork worktree commit.

**Read-only w.r.t. the live self-play run (D5).** ``--new-ckpt`` must be a numbered,
immutable ``ckpt_<n>.pt`` under a pointer-arch run dir (``latest``/``tmp``/``promo_``
handles are refused); the device is pinned to CPU; every artifact is written under
``data/human/**`` (never ``runs/train/**``); the global RNG (``random``/``numpy``/
``torch``) is snapshotted and restored around the in-process value calls.

CPU only. Imports no ``gui/``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random as _random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from catan_rl.human_data.engine_bridge import (
    ENGINE_RESOURCES,
    BridgeState,
    SeatPlacement,
    engine_state_hash,
    rebuild_env,
)
from catan_rl.human_data.record import GameRecord
from catan_rl.policy.obs_schema import N_OPP_POLICY_SLOTS, OPP_KIND_UNKNOWN

if TYPE_CHECKING:  # pragma: no cover - typing only
    from catan_rl.engine.board import catanBoard

_LOG = logging.getLogger("catan_rl.human_data.opening_scoreboard")

# ---------------------------------------------------------------------------
# Repo-relative paths (this file is src/catan_rl/human_data/opening_scoreboard.py)
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_HUMAN_ROOT = _REPO_ROOT / "data" / "human"
DEFAULT_CORPUS_PATH = DATA_HUMAN_ROOT / "corpus" / "provisional_openings.jsonl"
DEFAULT_OUTPUT_DIR = DATA_HUMAN_ROOT / "scoreboard"
SPLIT_FILE_NAME = "split_by_video.json"
VHAT_CACHE_NAME = "vhat_cache.json"
REPORT_NAME = "scoreboard_report.json"

#: The frozen v11 baseline checkpoint (loaded through the vendored legacy arch).
V11_CKPT_PATH = _REPO_ROOT / "runs" / "anchors" / "v11_cand_u724.pt"

#: Read-only allowlist for ``--new-ckpt``. D5's ``selfplay_pointer_arch`` name is
#: stale (the live dir is ``_v2``); BOTH are treated as read-only run dirs.
_POINTER_ARCH_DIRS = (
    _REPO_ROOT / "runs" / "train" / "selfplay_pointer_arch",
    _REPO_ROOT / "runs" / "train" / "selfplay_pointer_arch_v2",
)
#: A numbered, atomic, immutable checkpoint basename — ``ckpt_000000399.pt``. A
#: ``latest``/``tmp``/``promo_`` handle does NOT match, so it is refused (D5).
_NUMBERED_CKPT_RE = re.compile(r"^ckpt_\d+\.pt$")

# ---------------------------------------------------------------------------
# Fixed measurement constants (D4 / D5 / step6 §1)
# ---------------------------------------------------------------------------
PORT_MARGINAL_K = 8
FLOOR_N = 100
SPLIT_SEED = 20260722  # spec ratification date; pinned, do not change
EVAL_A_FRACTION = 0.60
BOOTSTRAP_B = 10_000
BOOTSTRAP_ALPHA = 0.05

#: Sentinel opponent identity fed to BOTH legs (they enter the obs → affect v̂, so
#: they must be identical across legs; the eval-time "unknown" defaults, D2).
OPP_KIND_SENTINEL = OPP_KIND_UNKNOWN
OPP_POLICY_ID_SENTINEL = N_OPP_POLICY_SLOTS - 1

#: The 9 fixed port slots as (hex_id, corner_1, corner_2) — identical to
#: ``catanBoard.updatePorts``. The port-type multiset (5 specific 2:1 + 4 generic
#: 3:1) is permuted over these slots per layout k.
_PORT_HEX_CORNERS: tuple[tuple[int, int, int], ...] = (
    (7, 2, 3),
    (8, 1, 2),
    (10, 1, 2),
    (11, 0, 1),
    (12, 5, 0),
    (14, 0, 5),
    (15, 4, 5),
    (16, 3, 4),
    (18, 3, 4),
)
_PORT_TYPES: tuple[str, ...] = (
    "2:1 BRICK",
    "2:1 SHEEP",
    "2:1 WOOD",
    "2:1 WHEAT",
    "2:1 ORE",
    "3:1 PORT",
    "3:1 PORT",
    "3:1 PORT",
    "3:1 PORT",
)


# ---------------------------------------------------------------------------
# Pre-registration (D3.3) — committed BEFORE the first metric, mirrored in the
# committed split file so a below-floor number cannot be laundered into a verdict.
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class PreRegistration:
    """The pre-registered clause-(b) sign + tolerance for the paired ΔAUC gate.

    NOTE: a bare ``ΔAUC ≥ 0`` point check is explicitly NOT the rule — at this n it
    fails an unchanged model ~20% of the time. The rule (borrowing step6 B3(4)) is a
    tolerance-band on the paired video-cluster bootstrap CI: NO-GO only if the CI
    excludes the tolerance in the regression direction (i.e. we are confident the new
    arch is worse than v11 by MORE than ``tolerance_auc``).
    """

    metric: str = "paired_delta_auc"
    #: The regression direction is ΔAUC < 0 (new arch worse than v11 at ranking wins).
    regression_direction: str = "delta_auc_below_zero"
    tolerance_auc: float = 0.02
    #: NO-GO iff the bootstrap CI's UPPER bound < -tolerance_auc (confident regression
    #: beyond tolerance). Suppressed entirely below the floor.
    rule: str = "no_go_iff_ci_high_below_neg_tolerance"
    floor_n: int = FLOOR_N

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "regression_direction": self.regression_direction,
            "tolerance_auc": self.tolerance_auc,
            "rule": self.rule,
            "floor_n": self.floor_n,
        }


PRE_REGISTRATION = PreRegistration()


# ---------------------------------------------------------------------------
# AUC (Mann–Whitney U) — provenance: lifted verbatim from ``scripts/pregate0.py``
# (``auc``) so this module has no ``scripts/`` import dependency (step6 GATE-A's
# "Secondary door" ΔAUC machinery). Rank-based → scale-invariant → NO squash (D1).
# ---------------------------------------------------------------------------
def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Rank-based ROC AUC (Mann–Whitney U). ``nan`` if a class is empty."""
    from scipy.stats import rankdata

    labels_i = np.asarray(labels).astype(int)
    scores_f = np.asarray(scores, dtype=float)
    n_pos = int((labels_i == 1).sum())
    n_neg = int((labels_i == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(scores_f)
    sum_pos = float(ranks[labels_i == 1].sum())
    return (sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


# ---------------------------------------------------------------------------
# Corpus loading + eligibility (SoT predicate — never re-implement the filter)
# ---------------------------------------------------------------------------
def load_records(corpus_path: Path) -> list[GameRecord]:
    """Load every ``GameRecord`` from a JSONL corpus (skips blank lines)."""
    records: list[GameRecord] = []
    with corpus_path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            records.append(GameRecord.from_json_line(line))
    return records


def eligible_records(records: list[GameRecord]) -> list[GameRecord]:
    """Filter to scoreboard-eligible records via the SoT predicate (D6).

    Eligibility keys EXCLUSIVELY off ``record.is_scoreboard_eligible()`` — the
    single source of truth — never a re-implemented filter.
    """
    return [r for r in records if r.is_scoreboard_eligible()]


def record_outcome(record: GameRecord) -> int:
    """1 iff the scoreboard subject (the ``agent`` handle) won the game."""
    return int(record.winner is not None and record.winner == record.players["agent"])


# ---------------------------------------------------------------------------
# By-video split (D3.4 / step6 §1) — deterministic, seeded, committed
# ---------------------------------------------------------------------------
def by_video_split(
    video_ids: list[str], *, seed: int = SPLIT_SEED, eval_a_fraction: float = EVAL_A_FRACTION
) -> tuple[list[str], list[str]]:
    """Deterministic seeded by-video split → (eval_a, holdout_b) video-id lists.

    Splits the *sorted unique* video ids so the result is a pure function of the id
    set + seed; EVAL-A gets ``round(eval_a_fraction · n)`` videos. Both lists are
    returned sorted.
    """
    unique = sorted(set(video_ids))
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(unique))
    n_eval = round(eval_a_fraction * len(unique))
    eval_idx = {int(i) for i in perm[:n_eval]}
    eval_a = sorted(unique[i] for i in range(len(unique)) if i in eval_idx)
    holdout_b = sorted(unique[i] for i in range(len(unique)) if i not in eval_idx)
    return eval_a, holdout_b


def build_split_payload(corpus_path: Path) -> dict[str, Any]:
    """Build the committed ``split_by_video.json`` payload from the corpus.

    Embeds the generation seed, the corpus-snapshot sha256, and the pre-registration
    block (so COMMIT 1 fully contains the pre-registered sign/tolerance BEFORE any
    metric is computed — D3.3/D3.4).
    """
    records = load_records(corpus_path)
    elig = eligible_records(records)
    video_ids = [r.video_id for r in elig]
    eval_a, holdout_b = by_video_split(video_ids)
    return {
        "schema": "opening_scoreboard.split_by_video/v1",
        "generated_from": str(corpus_path.relative_to(_REPO_ROOT)),
        "corpus_sha256": sha256_file(corpus_path),
        "split_seed": SPLIT_SEED,
        "eval_a_fraction": EVAL_A_FRACTION,
        "n_eligible_videos": len(set(video_ids)),
        "eval_a": eval_a,
        "holdout_b": holdout_b,
        "pre_registration": PRE_REGISTRATION.to_dict(),
        "note": (
            "Measure on EVAL-A only; HOLDOUT-B stays virgin for the later formal "
            "GATE-A verdict once the corpus passes the 100-game floor."
        ),
    }


def write_split_file(corpus_path: Path, out_path: Path) -> dict[str, Any]:
    """Write the by-video split file under ``data/human/**`` and return its payload."""
    _validate_write_path(out_path)
    payload = build_split_payload(corpus_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def load_split_file(path: Path) -> dict[str, Any]:
    """Load the committed split file and assert its pre-registration matches this
    module's constants (drift guard — the frozen file must never point at a
    superseding sign/tolerance)."""
    payload: dict[str, Any] = json.loads(path.read_text())
    committed = payload.get("pre_registration", {})
    if committed != PRE_REGISTRATION.to_dict():
        raise ValueError(
            "split file pre_registration block does not match PRE_REGISTRATION — "
            f"committed={committed} vs module={PRE_REGISTRATION.to_dict()}"
        )
    if int(payload.get("split_seed", -1)) != SPLIT_SEED:
        raise ValueError(
            f"split file seed {payload.get('split_seed')} != module SPLIT_SEED {SPLIT_SEED}"
        )
    return payload


# ---------------------------------------------------------------------------
# Adapter: GameRecord → BridgeState (net-new; D2/D4)
# ---------------------------------------------------------------------------
def per_layout_seed(video_id: str, game_index: int, k: int) -> int:
    """Pinned per-record-per-layout seed (step6 §1). Python ``hash()`` is banned."""
    digest = hashlib.sha256(f"{video_id}:{game_index}:{k}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


class RecordAdapter:
    """Builds the K=8 port-marginal ``BridgeState``s for one ``GameRecord``.

    The 9 fixed port-slot vertex pairs are board-invariant (pure geometry), so they
    are derived ONCE from a scratch board and reused across every record + layout.
    Per record the desert (robber) hex and the ``settlements[1]``-derived hands are
    computed once (k-invariant); only the port assignment varies with k.
    """

    def __init__(self) -> None:
        from catan_rl.engine.game import catanGame

        self._scratch_game = catanGame(render_mode=None)
        self._port_slots = _port_slot_pairs(self._scratch_game.board)

    def port_assignment_for(self, seed: int) -> dict[str, list[int]]:
        """Marginalize the unrecorded port assignment: permute the 9 port types over
        the 9 fixed slots with a seeded RNG (consumes NO global randomness)."""
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(self._port_slots))
        assignment: dict[str, list[int]] = {}
        for i, slot_i in enumerate(perm):
            assignment.setdefault(_PORT_TYPES[i], []).extend(
                int(v) for v in self._port_slots[int(slot_i)]
            )
        return {k: sorted(v) for k, v in assignment.items()}

    def bridge_states(self, record: GameRecord) -> list[BridgeState]:
        """Return the K=8 ``BridgeState``s (one per port-marginal layout)."""
        from catan_rl.engine.game import catanGame

        desert_hex = _desert_hex(record)
        resources: list[str] = ["DESERT"] * 19
        numbers: list[int | None] = [None] * 19
        for h in record.hexes:
            hid = int(h["hex_id"])
            resources[hid] = str(h["resource"])
            numbers[hid] = h["number"]

        # Scratch board with the record's injected layout → adjacency for hands.
        hand_game = catanGame(render_mode=None)
        hand_game.board.inject_hex_layout(resources, numbers, desert_hex)

        seat0_handle, seat1_handle = record.draft_order[0], record.draft_order[1]
        op0 = record.openings[seat0_handle]
        op1 = record.openings[seat1_handle]
        placements = {
            0: SeatPlacement(
                settlements=(int(op0.settlements[0]), int(op0.settlements[1])),
                roads=(int(op0.roads[0]), int(op0.roads[1])),
            ),
            1: SeatPlacement(
                settlements=(int(op1.settlements[0]), int(op1.settlements[1])),
                roads=(int(op1.roads[0]), int(op1.roads[1])),
            ),
        }
        hands = {
            0: _hand_from_settlement(hand_game.board, int(op0.settlements[1])),
            1: _hand_from_settlement(hand_game.board, int(op1.settlements[1])),
        }
        agent_seat = 0 if record.draft_order[0] == record.players["agent"] else 1
        hexes = tuple(
            {"hex_id": int(h["hex_id"]), "resource": str(h["resource"]), "number": h["number"]}
            for h in record.hexes
        )

        states: list[BridgeState] = []
        for k in range(PORT_MARGINAL_K):
            seed = per_layout_seed(record.video_id, record.game_index, k)
            states.append(
                BridgeState(
                    hexes=hexes,
                    robber_hex=desert_hex,
                    port_assignment=self.port_assignment_for(seed),
                    placements=placements,
                    hands=hands,
                    agent_seat=agent_seat,
                    # A deterministic engine opponent object; the obs opponent
                    # identity is carried by the sentinel opp_kind/opp_policy_id.
                    opponent_type="heuristic",
                    opp_kind=OPP_KIND_SENTINEL,
                    opp_policy_id=OPP_POLICY_ID_SENTINEL,
                )
            )
        return states


def _port_slot_pairs(board: catanBoard) -> list[list[int]]:
    """The 9 fixed port-slot vertex pairs (same derivation as ``updatePorts``)."""
    pairs: list[list[int]] = []
    for h_idx, c1, c2 in _PORT_HEX_CORNERS:
        v1 = _vertex_index_for_corner(board, h_idx, c1)
        v2 = _vertex_index_for_corner(board, h_idx, c2)
        if v1 is None or v2 is None:
            raise ValueError(f"could not resolve port slot vertices for hex {h_idx} ({c1},{c2})")
        pairs.append([v1, v2])
    return pairs


def _vertex_index_for_corner(board: catanBoard, hex_idx: int, corner_idx: int) -> int | None:
    corners = board.hexTileDict[hex_idx].get_corners(board.flat)
    target = corners[corner_idx]
    for v_pt, v_obj in board.boardGraph.items():
        if board.vertexDistance(target, v_pt) == 0:
            return int(v_obj.vertex_index)
    return None


def _desert_hex(record: GameRecord) -> int:
    for h in record.hexes:
        if str(h["resource"]) == "DESERT":
            return int(h["hex_id"])
    raise ValueError(f"record {record.video_id}:{record.game_index} has no DESERT hex")


def _hand_from_settlement(board: catanBoard, vertex_idx: int) -> dict[str, int]:
    """Starting-resource hand granted by the 2nd (resource-granting) settlement —
    the adjacent non-desert resources, matching the bridge's grant assertion (D2)."""
    hand = {r: 0 for r in ENGINE_RESOURCES}
    px = board.vertex_index_to_pixel_dict[vertex_idx]
    for adj_hex in board.boardGraph[px].adjacent_hex_indices:
        res_type = board.hexTileDict[adj_hex].resource_type
        if res_type != "DESERT":
            hand[res_type] += 1
    return hand


# ---------------------------------------------------------------------------
# Paired value scoring (D2) — ONE rebuild feeds BOTH legs (AC2)
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class LayoutValues:
    """The paired values + shared state hash for one (record, layout) cell."""

    new: float
    legacy: float
    engine_state_hash: str


class PairedScorer:
    """Loads both policies once and scores (record, layout) cells paired.

    Every ``v̂`` routes through ``rebuild_env`` (no bypass path, D2). Both legs read
    obs built from the SAME shared engine state; the state hash is captured once and
    returned so the identical-state pin (AC2) is assertable.
    """

    def __init__(self, new_ckpt: str, v11_ckpt: str, *, device: str = "cpu") -> None:
        import torch

        from catan_rl.eval.cross_arch import _build_legacy_policy, _build_new_policy

        if torch.device(device).type != "cpu":
            raise ValueError(f"opening scoreboard is CPU-pinned (D5); got device={device!r}")
        self._device = device
        self._new_policy = _build_new_policy(new_ckpt, device=device, seed=0)
        self._legacy_policy = _build_legacy_policy(v11_ckpt, device=device)

    def score_state(self, state: BridgeState) -> LayoutValues:
        from typing import cast

        import torch

        from catan_rl.eval.legacy_arch.obs_encoder import EnvObsState as LegacyEnvObsState
        from catan_rl.eval.legacy_arch.obs_encoder import ObsEncoder as LegacyObsEncoder
        from catan_rl.policy.obs_encoder import EnvObsState
        from catan_rl.policy.obs_tensor import obs_to_torch

        env = rebuild_env(state)
        assert env.game is not None
        assert env.agent_player is not None and env.opponent_player is not None
        assert env._obs_encoder is not None
        state_hash = engine_state_hash(env.game)

        # Post-setup, roll-pending phase flags with the pinned sentinel opp identity.
        # The new- and legacy-schema encoders take nominally-distinct (but
        # faithful-copy-identical) EnvObsState classes; both are built from the SAME
        # field values so the phase/POV fed to each leg is identical (AC2). The
        # engine STATE is the one shared ``env.game`` (its hash captured once above).
        new_state = EnvObsState(
            initial_placement_phase=False,
            setup_step=4,
            roll_pending=True,
            discard_pending=False,
            robber_placement_pending=False,
            road_building_roads_left=0,
            last_dice_roll=0,
            opp_kind=state.opp_kind,
            opp_policy_id=state.opp_policy_id,
            opp_id_masked=False,
        )
        legacy_state = LegacyEnvObsState(
            initial_placement_phase=False,
            setup_step=4,
            roll_pending=True,
            discard_pending=False,
            robber_placement_pending=False,
            road_building_roads_left=0,
            last_dice_roll=0,
            opp_kind=state.opp_kind,
            opp_policy_id=state.opp_policy_id,
            opp_id_masked=False,
        )
        new_obs = env._obs_encoder.build_obs(
            env.game,
            env.agent_player,
            env.opponent_player,
            new_state,
            hand_tracker=env._hand_tracker,
        )
        legacy_encoder = LegacyObsEncoder(env.game.board)
        legacy_obs = legacy_encoder.build_obs(
            env.game,
            env.agent_player,
            env.opponent_player,
            legacy_state,
            hand_tracker=env._hand_tracker,
        )
        dev = torch.device(self._device)
        with torch.no_grad():
            new_v = float(
                self._new_policy.forward(
                    obs_to_torch(cast("dict[str, np.ndarray]", new_obs), dev, add_batch=True)
                )["value"].item()
            )
            legacy_v = float(
                self._legacy_policy.forward(
                    obs_to_torch(cast("dict[str, np.ndarray]", legacy_obs), dev, add_batch=True)
                )["value"].item()
            )
        env.close()
        return LayoutValues(new=new_v, legacy=legacy_v, engine_state_hash=state_hash)


def compute_vhat_cache(
    records: list[GameRecord],
    scorer: PairedScorer,
    adapter: RecordAdapter,
) -> dict[str, dict[str, Any]]:
    """Compute the K=8 paired ``v̂`` cache, EXACTLY 8× per game per leg (D4/AC5).

    Keyed by ``video_id:game_index`` → ``{"outcome", "video_id", "source",
    "layouts": [{"k", "new", "legacy", "hash"}, ...]}``. Every later bootstrap /
    resample loop reads THIS cache — it never re-bridges.
    """
    cache: dict[str, dict[str, Any]] = {}
    rng_snapshot = _snapshot_global_rng()
    try:
        for rec in records:
            key = f"{rec.video_id}:{rec.game_index}"
            layouts: list[dict[str, Any]] = []
            for k, state in enumerate(adapter.bridge_states(rec)):
                vals = scorer.score_state(state)
                layouts.append(
                    {
                        "k": k,
                        "new": vals.new,
                        "legacy": vals.legacy,
                        "hash": vals.engine_state_hash,
                    }
                )
            cache[key] = {
                "video_id": rec.video_id,
                "game_index": rec.game_index,
                "outcome": record_outcome(rec),
                "source": rec.opponent_strength.source,
                "layouts": layouts,
            }
    finally:
        _restore_global_rng(rng_snapshot)
    return cache


# ---------------------------------------------------------------------------
# Metric: paired ΔAUC + by-video cluster bootstrap CI (D1)
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class GameObservation:
    """One observation per eligible game: the mean-over-k v̂ per leg + outcome."""

    video_id: str
    new_mean: float
    legacy_mean: float
    outcome: int


def observations_from_cache(cache: dict[str, dict[str, Any]]) -> list[GameObservation]:
    """Collapse the K=8 layouts to one port-marginal observation per game (mean v̂)."""
    obs: list[GameObservation] = []
    for entry in cache.values():
        layouts = entry["layouts"]
        new_mean = float(np.mean([lay["new"] for lay in layouts]))
        legacy_mean = float(np.mean([lay["legacy"] for lay in layouts]))
        obs.append(
            GameObservation(
                video_id=str(entry["video_id"]),
                new_mean=new_mean,
                legacy_mean=legacy_mean,
                outcome=int(entry["outcome"]),
            )
        )
    return obs


def paired_delta_auc(observations: list[GameObservation]) -> tuple[float, float, float]:
    """Return (new_auc, legacy_auc, delta_auc) over the port-marginal observations."""
    new = np.array([o.new_mean for o in observations], dtype=float)
    legacy = np.array([o.legacy_mean for o in observations], dtype=float)
    outcomes = np.array([o.outcome for o in observations], dtype=int)
    new_auc = auc(new, outcomes)
    legacy_auc = auc(legacy, outcomes)
    return new_auc, legacy_auc, new_auc - legacy_auc


def cluster_bootstrap_delta_ci(
    observations: list[GameObservation],
    *,
    n_boot: int = BOOTSTRAP_B,
    alpha: float = BOOTSTRAP_ALPHA,
    seed: int = SPLIT_SEED,
) -> tuple[float, float, list[float]]:
    """Paired by-video-cluster percentile bootstrap CI on ΔAUC (step6 §1).

    Resamples video ids (clusters) with replacement, recomputes BOTH AUCs from the
    resampled games (read from the passed observations = the cache), and takes the
    percentile CI on the ΔAUC. ``(ci_low, ci_high, samples)``.
    """
    by_video: dict[str, list[GameObservation]] = {}
    for o in observations:
        by_video.setdefault(o.video_id, []).append(o)
    video_ids = sorted(by_video)
    rng = np.random.default_rng(seed)
    deltas: list[float] = []
    for _ in range(n_boot):
        picks = rng.integers(0, len(video_ids), size=len(video_ids))
        resampled: list[GameObservation] = []
        for idx in picks:
            resampled.extend(by_video[video_ids[int(idx)]])
        _, _, delta = paired_delta_auc(resampled)
        if not np.isnan(delta):
            deltas.append(delta)
    if not deltas:
        return float("nan"), float("nan"), deltas
    lo = float(np.percentile(deltas, 100 * (alpha / 2)))
    hi = float(np.percentile(deltas, 100 * (1 - alpha / 2)))
    return lo, hi, deltas


# ---------------------------------------------------------------------------
# Base-rate WR (D6) — printed beside the metric, split by strength source
# ---------------------------------------------------------------------------
def base_rate_wr(
    observations: list[GameObservation],
    cache: dict[str, dict[str, Any]],
    *,
    seed: int = SPLIT_SEED,
) -> dict[str, dict[str, Any]]:
    """ThePhantom's eligible-game WR with a by-video cluster CI, split by
    ``opponent_strength.source`` (never hardcode "all tournament", D6)."""
    # Map game → (video_id, outcome, source) from the cache (SoT for source).
    rows: list[tuple[str, int, str]] = [
        (str(e["video_id"]), int(e["outcome"]), str(e["source"])) for e in cache.values()
    ]
    result: dict[str, dict[str, Any]] = {}
    sources = sorted({src for _, _, src in rows})
    for src in [*sources, "__all__"]:
        subset = rows if src == "__all__" else [r for r in rows if r[2] == src]
        if not subset:
            continue
        wins = sum(o for _, o, _ in subset)
        n = len(subset)
        lo, hi = _wr_cluster_ci(subset, seed=seed)
        result[src] = {"wins": wins, "n": n, "wr": wins / n, "ci": [lo, hi]}
    return result


def _wr_cluster_ci(
    rows: list[tuple[str, int, str]],
    *,
    seed: int,
    n_boot: int = BOOTSTRAP_B,
    alpha: float = BOOTSTRAP_ALPHA,
) -> tuple[float, float]:
    by_video: dict[str, list[int]] = {}
    for vid, outcome, _ in rows:
        by_video.setdefault(vid, []).append(outcome)
    video_ids = sorted(by_video)
    rng = np.random.default_rng(seed)
    wrs: list[float] = []
    for _ in range(n_boot):
        picks = rng.integers(0, len(video_ids), size=len(video_ids))
        vals: list[int] = []
        for idx in picks:
            vals.extend(by_video[video_ids[int(idx)]])
        if vals:
            wrs.append(float(np.mean(vals)))
    if not wrs:
        return float("nan"), float("nan")
    lo = float(np.percentile(wrs, 100 * (alpha / 2)))
    hi = float(np.percentile(wrs, 100 * (1 - alpha / 2)))
    return lo, hi


# ---------------------------------------------------------------------------
# Verdict / below-floor banner (D3.1) — verdict HARD-suppressed below the floor
# ---------------------------------------------------------------------------
def verdict_suppressed(n: int) -> bool:
    """Whether the GO/NO-GO verdict path is suppressed (n below the floor, D3.1)."""
    return n < FLOOR_N


def below_floor_banner(n: int) -> str:
    return f"n={n} per cell — BELOW FLOOR (<{FLOOR_N}) — MEASUREMENT-ONLY — NO VERDICT"


def verdict(n: int, ci_low: float, ci_high: float) -> str:
    """Pre-registered clause-(b) verdict. HARD-suppressed below the floor (AC6).

    Above the floor: NO-GO iff the paired video-cluster bootstrap CI excludes the
    tolerance in the regression direction (``ci_high < -tolerance_auc``); GO otherwise.
    """
    if verdict_suppressed(n):
        return "MEASUREMENT-ONLY"
    if not np.isnan(ci_high) and ci_high < -PRE_REGISTRATION.tolerance_auc:
        return "NO-GO"
    return "GO"


# ---------------------------------------------------------------------------
# Provenance recording (D3.2) — sha256 hashes (recording, not enforcement)
# ---------------------------------------------------------------------------
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def provenance_hashes(new_ckpt: Path, corpus_path: Path) -> dict[str, str]:
    """Record the D3.2 input hashes: this module, the engine_bridge + board.py
    injection surface, the corpus JSONL, BOTH checkpoints, and the v11 pre-fork
    worktree commit. Recording is one ``sha256``; enforcement is deferred."""
    from catan_rl.eval.engine_parity import assert_engine_parity
    from catan_rl.eval.legacy_arch import _provenance as legacy_provenance

    engine_dir = _REPO_ROOT / "src" / "catan_rl"
    parity = assert_engine_parity(strict=True)
    hashes = {
        "opening_scoreboard.py": sha256_file(Path(__file__).resolve()),
        "engine_bridge.py": sha256_file(engine_dir / "human_data" / "engine_bridge.py"),
        "engine_board.py": sha256_file(engine_dir / "engine" / "board.py"),
        "corpus_jsonl": sha256_file(corpus_path),
        "new_ckpt": sha256_file(new_ckpt),
        "v11_ckpt": sha256_file(V11_CKPT_PATH),
        "v11_prefork_commit": legacy_provenance.VENDOR_COMMIT,
        "engine_parity_engine_tree": parity["engine"],
        "engine_parity_board_geometry": parity["board_geometry"],
    }
    return hashes


# ---------------------------------------------------------------------------
# Global-RNG snapshot / restore (D5)
# ---------------------------------------------------------------------------
def _snapshot_global_rng() -> dict[str, Any]:
    import torch

    return {
        "py": _random.getstate(),
        "np": np.random.get_state(),
        "torch": torch.random.get_rng_state(),
    }


def _restore_global_rng(snap: dict[str, Any]) -> None:
    import torch

    _random.setstate(snap["py"])
    np.random.set_state(snap["np"])
    torch.random.set_rng_state(snap["torch"])


# ---------------------------------------------------------------------------
# Read-only path validation (D5)
# ---------------------------------------------------------------------------
def validate_new_ckpt(path_str: str) -> Path:
    """Resolve + validate ``--new-ckpt`` as a numbered, immutable checkpoint under a
    pointer-arch run dir. Refuses ``latest``/``tmp``/``promo_`` handles (D5)."""
    path = Path(path_str).resolve()
    if not _NUMBERED_CKPT_RE.match(path.name):
        raise ValueError(
            f"--new-ckpt must be a numbered checkpoint 'ckpt_<n>.pt' (immutable); got {path.name!r}"
        )
    if not any(_is_relative_to(path, d) for d in _POINTER_ARCH_DIRS):
        allow = [str(d) for d in _POINTER_ARCH_DIRS]
        raise ValueError(f"--new-ckpt must live under a pointer-arch run dir {allow}; got {path}")
    if not path.is_file():
        raise FileNotFoundError(f"--new-ckpt not found: {path}")
    return path


def _validate_write_path(path: Path) -> Path:
    """Assert a write target resolves under ``data/human/**`` (never runs/train, D5)."""
    resolved = path.resolve()
    if not _is_relative_to(resolved, DATA_HUMAN_ROOT):
        raise ValueError(f"write path must be under {DATA_HUMAN_ROOT}; got {resolved}")
    return resolved


def _is_relative_to(path: Path, base: Path) -> bool:
    try:
        path.relative_to(base)
        return True
    except ValueError:
        return False


# ---------------------------------------------------------------------------
# Report + CLI
# ---------------------------------------------------------------------------
def run_scoreboard(
    *,
    new_ckpt: str,
    corpus_path: Path = DEFAULT_CORPUS_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    device: str = "cpu",
    split_path: Path | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    """Compute the paired ΔAUC opening scoreboard on EVAL-A and write the report.

    Returns the report dict. Verdict is hard-suppressed below the floor (D3.1).
    """
    new_ckpt_path = validate_new_ckpt(new_ckpt)
    output_dir = _validate_write_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    split_path = split_path or (output_dir / SPLIT_FILE_NAME)
    if not split_path.is_file():
        raise FileNotFoundError(
            f"by-video split file missing: {split_path} — it MUST be committed before the "
            "first metric run (D3.4/AC9). Generate it with --emit-split."
        )
    split = load_split_file(split_path)
    eval_a_videos = set(split["eval_a"])

    records = load_records(corpus_path)
    elig = eligible_records(records)
    eval_a_records = [r for r in elig if r.video_id in eval_a_videos]
    if limit is not None:
        eval_a_records = eval_a_records[:limit]
    n = len(eval_a_records)

    scorer = PairedScorer(str(new_ckpt_path), str(V11_CKPT_PATH), device=device)
    adapter = RecordAdapter()
    cache = compute_vhat_cache(eval_a_records, scorer, adapter)
    _validate_write_path(output_dir / VHAT_CACHE_NAME)
    (output_dir / VHAT_CACHE_NAME).write_text(json.dumps(cache, indent=2, sort_keys=True) + "\n")

    observations = observations_from_cache(cache)
    new_auc, legacy_auc, delta = paired_delta_auc(observations)
    ci_low, ci_high, _ = cluster_bootstrap_delta_ci(observations)
    wr = base_rate_wr(observations, cache)
    the_verdict = verdict(n, ci_low, ci_high)
    banner = below_floor_banner(n) if verdict_suppressed(n) else f"VERDICT: {the_verdict}"

    report: dict[str, Any] = {
        "banner": banner,
        "n_eval_a": n,
        "below_floor": verdict_suppressed(n),
        "verdict": the_verdict,
        "pre_registration": PRE_REGISTRATION.to_dict(),
        "metric": {
            "new_auc": new_auc,
            "v11_auc": legacy_auc,
            "delta_auc": delta,
            "delta_auc_ci": [ci_low, ci_high],
            "ci_flavor": "by_video_cluster_percentile_bootstrap",
            "n_boot": BOOTSTRAP_B,
        },
        "base_rate_wr": wr,
        "provenance": provenance_hashes(new_ckpt_path, corpus_path),
        "split_file": str(split_path.relative_to(_REPO_ROOT)),
        "new_ckpt": str(new_ckpt_path.relative_to(_REPO_ROOT)),
        "corpus": str(corpus_path.relative_to(_REPO_ROOT)),
    }
    _validate_write_path(output_dir / REPORT_NAME)
    (output_dir / REPORT_NAME).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def _print_report(report: dict[str, Any]) -> None:
    print("=" * 78)
    print("OPENING SCOREBOARD — paired ΔAUC (new pointer-arch vs v11), EVAL-A")
    print("=" * 78)
    print(report["banner"])
    m = report["metric"]
    print(
        f"ΔAUC = {m['delta_auc']:+.4f}  (new {m['new_auc']:.4f} − v11 {m['v11_auc']:.4f})  "
        f"CI[{m['delta_auc_ci'][0]:+.4f}, {m['delta_auc_ci'][1]:+.4f}]"
    )
    print("ThePhantom eligible-game WR (by opponent_strength.source):")
    for src, row in report["base_rate_wr"].items():
        print(
            f"  {src:>12}: WR {row['wr']:.3f}  (n={row['n']}, wins={row['wins']}, "
            f"CI[{row['ci'][0]:.3f}, {row['ci'][1]:.3f}])"
        )
    print(f"verdict: {report['verdict']}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--new-ckpt",
        help="Numbered pointer-arch checkpoint (ckpt_<n>.pt under a pointer-arch run dir).",
    )
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="cpu", help="CPU-pinned (D5).")
    parser.add_argument("--limit", type=int, default=None, help="Cap #games (dev only).")
    parser.add_argument(
        "--emit-split",
        action="store_true",
        help="Write the by-video split file and exit (COMMIT 1; run before any metric).",
    )
    args = parser.parse_args(argv)

    if args.emit_split:
        out = _validate_write_path(args.output_dir) / SPLIT_FILE_NAME
        payload = write_split_file(args.corpus, out)
        print(f"wrote split file: {out} ({payload['n_eligible_videos']} eligible videos)")
        return 0

    if not args.new_ckpt:
        parser.error("--new-ckpt is required (unless --emit-split)")
    report = run_scoreboard(
        new_ckpt=args.new_ckpt,
        corpus_path=args.corpus,
        output_dir=args.output_dir,
        device=args.device,
        limit=args.limit,
    )
    _print_report(report)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
