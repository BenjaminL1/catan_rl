"""Label store → BC shard converter (spec ``setup-labeling-and-champion-finetune`` D4).

Turns the owner's hand-labeled openings (``scenarios.jsonl``) into an NPZ shard
+ manifest that :class:`catan_rl.bc.loader.BcDataset` loads unchanged.

Why the JSONL is the durable artefact
-------------------------------------
A label row stores a ``game_seed`` and the picks that preceded the decision —
never an observation. The obs schema has moved several times and will move
again; a stored obs would rot with it. Regenerating from the seed at CONVERSION
time means the corpus is proof against schema change forever, at the cost of
replaying a four-pick snake draft per scenario (milliseconds).

What one scenario becomes
-------------------------
**Two** rows, in the order the policy would take them:

1. the SETTLEMENT decision, whose obs is built on the state BEFORE the
   settlement exists (``setup_step`` 0 or 2), and
2. the ROAD decision, whose obs is built on the state AFTER it
   (``setup_step`` 1 or 3).

The second point is not a detail. The road head's legal set is defined by the
settlement just placed, so a road row carrying the pre-settlement obs would
train the policy to choose a road from a position it will never see. §E of
``docs/plans/v2/setup_labeling.md`` was silent here; the split is pinned by
``tests/unit/labeling/test_to_shard.py``.

Opponent-identity duplication
-----------------------------
The policy conditions on an opponent-id embedding (``opponent_kind`` /
``opponent_policy_id``), so a row stamped with ONE kind teaches the opening only
to that conditional slice. A human opening is opponent-independent — it is a
board judgement made before anyone has moved — so every human row is duplicated
across each kind the gates and the successor self-play use. The duplicates
differ in exactly those two obs entries and share a ``game_id``, which is what
stops :meth:`~catan_rl.bc.loader.BcDataset.train_val_split` from putting a row
in train and its twin in val.

Zero heuristic rows are produced here, ever (spec non-goal): the owner's labels
are the SOLE opening teacher.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np

from catan_rl.engine.player import player as EnginePlayer
from catan_rl.env.hand_tracker import BroadcastHandTracker
from catan_rl.env.masks import compute_action_masks
from catan_rl.env.ruleset import RULESET_R0
from catan_rl.labeling.scenario_gen import Pick, ScenarioGenerator
from catan_rl.labeling.store import held_out_split, load_scenarios
from catan_rl.policy.obs_encoder import EnvObsState, ObsEncoder, hidden_belief_target
from catan_rl.policy.obs_schema import (
    FORCED_RULE_VERSION,
    MASK_KEYS,
    N_OPP_POLICY_SLOTS,
    OPP_KIND_HEURISTIC,
    OPP_KIND_LEAGUE,
    OPP_KIND_SELF_LATEST,
    RULESET_VERSION,
    ActionType,
    is_forced_decision,
)

#: Obs members written per row. Mirrors ``bc.loader._OBS_KEYS_FLOAT`` +
#: ``_OBS_KEYS_INT`` — 12 keys, NOT the 10 the source plan named (it predates
#: the opponent-id embedding). The loader reads exactly these.
OBS_KEYS_FLOAT: tuple[str, ...] = (
    "tile_representations",
    "current_player_main",
    "next_player_main",
    "current_dev_counts",
    "next_played_dev_counts",
    "hex_features",
    "vertex_features",
    "edge_features",
    "global_features",
    "is_setup",
)
OBS_KEYS_INT: tuple[str, ...] = ("opponent_kind", "opponent_policy_id")

#: Opponent kinds every human row is duplicated across. ``HEURISTIC`` is what
#: the D7 gates play; ``SELF_LATEST`` / ``LEAGUE`` are what the successor
#: self-play slice stamps. ``UNKNOWN`` is deliberately excluded — it is the
#: masked value, and teaching the opening under a mask the learner only sees
#: stochastically would dilute the signal rather than broaden it.
DEFAULT_OPPONENT_KINDS: tuple[int, ...] = (
    OPP_KIND_HEURISTIC,
    OPP_KIND_SELF_LATEST,
    OPP_KIND_LEAGUE,
)

#: The "unknown policy slot" index. A human label names no league snapshot, and
#: inventing one would tie the opening to a slot that will hold a different
#: policy next week.
UNKNOWN_POLICY_ID: int = N_OPP_POLICY_SLOTS - 1


class LabelConversionError(RuntimeError):
    """A label row cannot be replayed into a position. Never skipped silently."""


@dataclass
class _Row:
    """One (obs, action, mask) training row, pre-duplication."""

    obs: dict[str, Any]
    mask: dict[str, np.ndarray]
    action: np.ndarray
    belief_target: np.ndarray
    forced: bool
    phase: str
    player_seat: int
    step_idx: int
    game_id: int


def _replay_to_scenario(label: dict[str, Any]) -> tuple[ScenarioGenerator, Any, Any]:
    """Rebuild the position a label row describes.

    Returns ``(generator, scenario, hand_tracker)``. The tracker is subscribed
    BEFORE any pick is applied, so the setup resource grants of the reverse pass
    (which arrive as broadcast events) land in it naturally — exactly as they do
    in the env, rather than being back-filled.
    """
    gen = ScenarioGenerator(seed=int(label["game_seed"]))
    scenario = gen.current()
    if scenario is None:  # pragma: no cover - a fresh generator always has one
        raise LabelConversionError("fresh ScenarioGenerator yielded no scenario")
    tracker = BroadcastHandTracker([p.name for p in gen._players])
    tracker.subscribe(scenario.game.broadcast)

    for pick in label["prior_picks"]:
        p = Pick.from_dict(pick)
        gen.apply(p.settlement_vertex, p.road_edge)
    scenario = gen.current()
    if scenario is None:
        raise LabelConversionError(
            f"scenario {label['scenario_id']}: prior_picks exhausted the draft"
        )
    if scenario.draft_position != int(label["draft_position"]):
        raise LabelConversionError(
            f"scenario {label['scenario_id']}: replayed to draft position "
            f"{scenario.draft_position} but the row claims {label['draft_position']}"
        )
    if scenario.acting_player_idx != int(label["acting_player"]):
        raise LabelConversionError(
            f"scenario {label['scenario_id']}: replayed acting player "
            f"{scenario.acting_player_idx} but the row claims {label['acting_player']}"
        )
    return gen, scenario, tracker


def _obs_and_mask(
    *,
    gen: ScenarioGenerator,
    scenario: Any,
    encoder: ObsEncoder,
    tracker: BroadcastHandTracker,
    setup_step: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    acting = scenario._acting_player
    opponent = next(p for p in gen._players if p is not acting)
    env_state = EnvObsState(initial_placement_phase=True, setup_step=setup_step)
    obs = cast(
        dict[str, Any],
        encoder.build_obs(scenario.game, acting, opponent, env_state, hand_tracker=tracker),
    )
    # R0 is PINNED, not defaulted (D6): ``ptr_v1_u500.pt`` carries no ruleset
    # stamp, so the fine-tune it feeds is an R0 lineage. The setup masks are
    # identical across epochs, but naming the epoch here is what keeps a future
    # default flip from silently re-ruling the corpus.
    mask = compute_action_masks(
        scenario.game,
        acting,
        env_state,
        scenario._vertex_to_idx,
        scenario._edge_to_idx,
        ruleset=RULESET_R0,
    )
    return obs, mask


def rows_for_label(label: dict[str, Any], *, game_id: int) -> list[_Row]:
    """Build the two training rows (settlement, then road) for one label.

    Raises:
        LabelConversionError: if the row cannot be replayed, or if the labeled
            settlement / road is illegal in the reconstructed position. A label
            that no longer type-checks against the engine is a corpus defect,
            not a row to drop quietly.
    """
    gen, scenario, tracker = _replay_to_scenario(label)
    encoder = ObsEncoder(gen._board)
    acting = scenario._acting_player
    opponent = next(p for p in gen._players if p is not acting)
    seat = int(scenario.acting_player_idx)
    vertex = int(label["settlement_vertex"])
    edge = int(label["road_edge"])
    setup_step_settle = scenario._setup_step_road - 1

    if not bool(scenario.legal_settlement_corners[vertex]):
        raise LabelConversionError(
            f"scenario {label['scenario_id']}: labeled settlement vertex {vertex} "
            f"is illegal in the reconstructed position"
        )

    settle_obs, settle_mask = _obs_and_mask(
        gen=gen,
        scenario=scenario,
        encoder=encoder,
        tracker=tracker,
        setup_step=setup_step_settle,
    )
    belief = hidden_belief_target(opponent)
    rows = [
        _Row(
            obs=settle_obs,
            mask=settle_mask,
            action=_action(ActionType.BUILD_SETTLEMENT, corner_idx=vertex),
            belief_target=belief,
            forced=is_forced_decision(settle_mask),
            phase="setup",
            player_seat=seat,
            step_idx=0,
            game_id=game_id,
        )
    ]

    # The ROAD decision is made on the POST-settlement board. Apply it for real
    # (this generator is discarded after the scenario) rather than reverting, so
    # the obs is the one the policy will actually be handed.
    v_px = scenario._idx_to_vertex_pixel[vertex]
    cast(EnginePlayer, acting).build_settlement(v_px, gen._board, is_free=True)
    road_obs, road_mask = _obs_and_mask(
        gen=gen,
        scenario=scenario,
        encoder=encoder,
        tracker=tracker,
        setup_step=scenario._setup_step_road,
    )
    if not bool(np.asarray(road_mask["edge"], dtype=bool)[edge]):
        raise LabelConversionError(
            f"scenario {label['scenario_id']}: labeled road edge {edge} is illegal "
            f"after the labeled settlement at {vertex}"
        )
    rows.append(
        _Row(
            obs=road_obs,
            mask=road_mask,
            action=_action(ActionType.BUILD_ROAD, edge_idx=edge),
            belief_target=belief,
            forced=is_forced_decision(road_mask),
            phase="setup",
            player_seat=seat,
            step_idx=1,
            game_id=game_id,
        )
    )
    return rows


def _action(action_type: int, *, corner_idx: int = 0, edge_idx: int = 0) -> np.ndarray:
    return np.array([action_type, corner_idx, edge_idx, 0, 0, 0], dtype=np.int64)


def _duplicate_across_opponents(rows: list[_Row], opponent_kinds: tuple[int, ...]) -> list[_Row]:
    """One row per (row, opponent_kind). See the module docstring for why."""
    out: list[_Row] = []
    for row in rows:
        for kind in opponent_kinds:
            obs = dict(row.obs)
            obs["opponent_kind"] = np.int64(kind)
            obs["opponent_policy_id"] = np.int64(UNKNOWN_POLICY_ID)
            out.append(
                _Row(
                    obs=obs,
                    mask=row.mask,
                    action=row.action,
                    belief_target=row.belief_target,
                    forced=row.forced,
                    phase=row.phase,
                    player_seat=row.player_seat,
                    step_idx=row.step_idx,
                    game_id=row.game_id,
                )
            )
    return out


def _flatten(rows: list[_Row]) -> dict[str, np.ndarray]:
    flat: dict[str, np.ndarray] = {}
    for key in OBS_KEYS_FLOAT:
        flat[f"obs/{key}"] = np.stack([np.asarray(r.obs[key], dtype=np.float32) for r in rows])
    for key in OBS_KEYS_INT:
        flat[f"obs/{key}"] = np.asarray([int(r.obs[key]) for r in rows], dtype=np.int64)
    for key in MASK_KEYS:
        flat[f"mask/{key}"] = np.stack([np.asarray(r.mask[key], dtype=bool) for r in rows])
    flat["action"] = np.stack([r.action for r in rows]).astype(np.int64)
    flat["belief_target"] = np.stack([np.asarray(r.belief_target, dtype=np.float32) for r in rows])
    # No outcome label exists for a human opening — the game it came from (if
    # any) is not the game this position leads to. Zero here, and the fine-tune
    # ZEROES THE VALUE LOSS on these rows rather than regressing toward it (D5).
    flat["z_disc"] = np.zeros(len(rows), dtype=np.float32)
    flat["game_id"] = np.asarray([r.game_id for r in rows], dtype=np.int64)
    flat["step_idx"] = np.asarray([r.step_idx for r in rows], dtype=np.int64)
    flat["player_seat"] = np.asarray([r.player_seat for r in rows], dtype=np.int64)
    flat["phase"] = np.asarray([r.phase for r in rows], dtype="<U8")
    flat["forced"] = np.asarray([r.forced for r in rows], dtype=bool)
    return flat


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:  # pragma: no cover - provenance is best-effort
        return "unknown"


def convert(
    labels_path: str | Path,
    out_dir: str | Path,
    *,
    opponent_kinds: tuple[int, ...] = DEFAULT_OPPONENT_KINDS,
    shard_name: str = "shard_00000.npz",
    held_out_frac: float = 0.0,
    split_seed: int = 0,
) -> dict[str, Any]:
    """Convert a label store into a single BC shard + manifest in ``out_dir``.

    A human corpus is small by construction (hundreds of scenarios ⇒ thousands
    of rows), so one shard is written rather than the streaming multi-shard
    layout ``bc.dataset`` needs for its millions of heuristic rows.

    The held-out split is decided HERE, not at gate time
    -----------------------------------------------------
    With ``held_out_frac > 0`` the labels are split by ``game_seed``
    (:func:`catan_rl.labeling.store.held_out_split`) and **only the train half
    is written into the shard**. The held-out ``game_seed`` s are stamped into
    the manifest, and ``scripts/eval_setup_agreement.py`` reads them back from
    there rather than re-deriving a split of its own.

    That direction of dependency is the whole point. D7 gate 1 is specified as
    *held-out* top-1 settlement agreement; if the split were computed at eval
    time over the same file the shard was built from, the gate would measure
    agreement on rows the candidate was fine-tuned on — memorisation, read as
    generalisation, in exactly the gate whose stated remedy for a marginal
    result is MORE LABELS. Leaving ``held_out_frac=0`` produces a shard with no
    held-out set at all, and the gate CLI then REFUSES to run against it.

    Returns the manifest dict (also written to ``out_dir/manifest.json``).
    """
    all_labels = load_scenarios(Path(labels_path))
    if not all_labels:
        raise LabelConversionError(f"no label rows found in {labels_path}")

    labels = all_labels
    held: list[dict[str, Any]] = []
    if held_out_frac > 0.0:
        labels, held = held_out_split(all_labels, frac=held_out_frac, seed=split_seed)
        if not labels:
            raise LabelConversionError(
                f"held_out_frac={held_out_frac} left no training rows in "
                f"{labels_path} ({len(all_labels)} rows over "
                f"{len({int(r['game_seed']) for r in all_labels})} seeds)"
            )
    held_out_seeds = sorted({int(r["game_seed"]) for r in held})

    # ``game_id`` is per SCENARIO, assigned in first-seen order so the mapping is
    # stable for a given file. Every duplicate of a scenario shares it, which is
    # what keeps ``train_val_split`` (which splits BY game_id) from straddling.
    scenario_ids: dict[str, int] = {}
    rows: list[_Row] = []
    for label in labels:
        sid = str(label["scenario_id"])
        if sid not in scenario_ids:
            scenario_ids[sid] = len(scenario_ids)
        rows.extend(rows_for_label(label, game_id=scenario_ids[sid]))

    rows = _duplicate_across_opponents(rows, opponent_kinds)
    flat = _flatten(rows)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    shard_path = out_dir / shard_name
    cast(Any, np.savez_compressed)(shard_path, **flat)

    source_counts: dict[str, int] = {}
    for label in labels:
        source_counts[str(label["source"])] = source_counts.get(str(label["source"]), 0) + 1

    manifest: dict[str, Any] = {
        "shards": [
            {
                "shard": shard_name,
                "n_pairs": len(rows),
                "n_games": len(scenario_ids),
                "game_ids": sorted(scenario_ids.values()),
            }
        ],
        # ``bc.loader.load_manifest`` REFUSES an unstamped or stale corpus.
        # Imported, never literal: a schema bump must move this file's output
        # with it or the loader will say so.
        "forced_rule_version": FORCED_RULE_VERSION,
        "ruleset_version": RULESET_VERSION,
        # Provenance. A human corpus is small and irreplaceable; a shard that
        # cannot name the labels and the code it came from is not auditable.
        "source": "human_labels",
        "labels_path": str(labels_path),
        "n_scenarios": len(scenario_ids),
        "n_pairs": len(rows),
        "opponent_kinds": list(opponent_kinds),
        # The D7 gate-1 evaluation set, named by the converter that EXCLUDED it.
        # ``scripts/eval_setup_agreement.py`` fails closed on a missing key.
        "held_out_frac": float(held_out_frac),
        "split_seed": int(split_seed),
        "held_out_game_seeds": held_out_seeds,
        "n_held_out_scenarios": len({str(r["scenario_id"]) for r in held}),
        "label_source_counts": source_counts,
        "git_sha": _git_sha(),
        "ruleset": RULESET_R0,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest
