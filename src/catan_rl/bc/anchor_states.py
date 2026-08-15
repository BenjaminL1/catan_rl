"""Anchor-state sampler for the champion fine-tune (spec D5).

The fine-tune installs the owner's openings into ``runs/anchors/ptr_v1_u500.pt``
and holds the rest of the champion's behaviour in place with a KL term against a
frozen copy of itself. That term needs STATES to be evaluated on, and *which*
states is the whole question.

**The states must come from games whose setups are FORCED to the owner's
openings.** An anchor set sampled from ordinary self-play games is off-
distribution by construction: it measures drift on the midgames the champion
already reaches, while the fine-tune is deliberately moving it toward the
midgames the owner's openings lead to. Anchoring on the distribution being LEFT
constrains the wrong thing — the term stays small while the behaviour that
matters is free to move. So every anchor state here is rolled out from a real
labeled opening, replayed into the engine by
:func:`catan_rl.human_data.engine_bridge.rebuild_env`.

**The walker is the FROZEN CHAMPION, and it is a REQUIRED argument.** Once the
labeled opening is on the board, every subsequent ply decides which midgame the
anchor is actually read on. A uniformly-random legal walker leaves the forced
opening as the only champion-shaped thing about the trajectory: two plies later
the position is one no policy would occupy, so the term bounds drift on a
distribution the deployment never visits — the same failure D5 names for anchor
sets drawn from ordinary self-play, one level down. The walker therefore takes
no default; a defaulted one is precisely how the forbidden random fallback would
come back silently.

*Sampled, not greedy.* The deployed rollout and eval paths sample
(``eval.harness._play_one_game``, the PPO rollout collector), so the sampled
distribution IS the champion's on-policy state distribution — the one the anchor
is meant to hold in place. Greedy would anchor a distribution no deployed path
visits and, over a handful of labeled openings x 2 seats x 3 opponent kinds,
would make every refresh replay the same few trajectories, defeating
``anchor_refresh_every``.

*Frozen, not trainable.* Walking with the policy under optimisation would make
the anchor's state distribution a moving target that drifts with the very update
the term exists to bound, and would sample from a ``train()``-mode network.

Only NON-setup states are collected. Setup contexts are the ones the human rows
are teaching; anchoring them to the champion would cancel the fine-tune out.

There is no fallback. With no usable labeled opening the sampler RAISES —
substituting heuristic or random openings would put exactly the distribution the
spec forbids into the anchor term, silently, under a name that says otherwise.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch

from catan_rl.env.ruleset import RULESET_R0
from catan_rl.human_data.engine_bridge import BridgeState, SeatPlacement, rebuild_env
from catan_rl.labeling.scenario_gen import Pick, ScenarioGenerator
from catan_rl.labeling.store import load_scenarios
from catan_rl.labeling.to_shard import DEFAULT_OPPONENT_KINDS
from catan_rl.policy import CatanPolicy
from catan_rl.policy.obs_schema import (
    HEAD_RELEVANCE,
    MASK_KEY_FOR,
    N_OPP_POLICY_SLOTS,
    OPP_KIND_HEURISTIC,
    action_masked_legal,
)
from catan_rl.policy.obs_tensor import masks_to_torch, obs_to_torch

ENGINE_RESOURCES: tuple[str, ...] = ("ORE", "BRICK", "WHEAT", "WOOD", "SHEEP")

#: Walker device default. A module-level singleton because a ``torch.device``
#: call in an argument default is evaluated at import time (ruff B008); CPU is
#: the repo's eval-side device policy and this is a batch-1 forward per ply.
_CPU_DEVICE = torch.device("cpu")

#: Opponent-id stamps the anchor states are spread across.
#:
#: The human shard duplicates every row across all three kinds
#: (:data:`catan_rl.labeling.to_shard.DEFAULT_OPPONENT_KINDS`), so the
#: fine-tune's gradient reaches all three conditional slices. An anchor term
#: read only under ``HEURISTIC`` would therefore leave non-setup drift in the
#: two slices the successor self-play actually runs under both unmeasured and
#: unpenalised. Only the obs stamp varies — the opponent is the heuristic in
#: every case, because these games must still be playable.
ANCHOR_OPPONENT_KINDS: tuple[int, ...] = DEFAULT_OPPONENT_KINDS


class NoLabeledOpeningsError(RuntimeError):
    """No complete labeled opening is available. Never falls back to heuristics."""


@dataclass(frozen=True)
class LabeledOpening:
    """A complete four-pick snake draft recovered from the label corpus."""

    game_seed: int
    picks: tuple[Pick, ...]  # in draft order: seat0, seat1, seat1, seat0


def complete_openings(
    labels_path: str | Path,
    *,
    exclude_game_seeds: Iterable[int] = (),
) -> list[LabeledOpening]:
    """Recover every FULL draft the corpus contains.

    A row at draft position 4 already carries the three picks that preceded it
    in ``prior_picks``, so one such row IS a complete opening — no cross-row
    joining, and no risk of stitching picks from two different sessions on the
    same seed.

    ``exclude_game_seeds`` drops the drafts the shard converter withheld for the
    D7 gate-1 measurement. Without it a held-out opening still shapes the
    candidate through the anchor rollouts, and "held out" would again be a
    statement about one code path rather than about the checkpoint.
    """
    excluded = {int(s) for s in exclude_game_seeds}
    out: list[LabeledOpening] = []
    for row in load_scenarios(Path(labels_path)):
        if int(row["draft_position"]) != 4:
            continue
        if int(row["game_seed"]) in excluded:
            continue
        picks = [Pick.from_dict(p) for p in row["prior_picks"]]
        picks.append(
            Pick(
                player=int(row["acting_player"]),
                settlement_vertex=int(row["settlement_vertex"]),
                road_edge=int(row["road_edge"]),
            )
        )
        if len(picks) != 4:
            continue
        out.append(LabeledOpening(game_seed=int(row["game_seed"]), picks=tuple(picks)))
    return out


def _bridge_state(
    opening: LabeledOpening, *, agent_seat: int, opp_kind: int = OPP_KIND_HEURISTIC
) -> BridgeState:
    """Describe ``opening`` as a post-setup :class:`BridgeState`.

    The board is regenerated from the seed (the same reconstruction the shard
    converter relies on) and the hands are DERIVED from each seat's second
    settlement rather than carried, because a label row stores placements only.
    ``rebuild_env`` re-derives them independently and asserts the two agree, so
    a wrong derivation here surfaces as a ``BridgeError``, not as a quietly
    mis-stocked opening.
    """
    gen = ScenarioGenerator(seed=opening.game_seed)
    board = gen._board
    hexes = tuple(
        {
            "hex_id": i,
            "resource": str(board.hexTileDict[i].resource_type),
            "number": (
                int(board.hexTileDict[i].number_token)
                if board.hexTileDict[i].number_token is not None
                else None
            ),
        }
        for i in range(19)
    )
    robber = [i for i in range(19) if board.hexTileDict[i].has_robber]
    if len(robber) != 1:  # pragma: no cover - a fresh board always has one
        raise NoLabeledOpeningsError(f"expected one robber hex, found {robber}")

    placements: dict[int, SeatPlacement] = {}
    hands: dict[int, dict[str, int]] = {}
    for seat in (0, 1):
        own = [p for p in opening.picks if p.player == seat]
        if len(own) != 2:
            raise NoLabeledOpeningsError(
                f"seed {opening.game_seed}: seat {seat} has {len(own)} picks, expected 2"
            )
        placements[seat] = SeatPlacement(
            settlements=(own[0].settlement_vertex, own[1].settlement_vertex),
            roads=(own[0].road_edge, own[1].road_edge),
        )
        hand = dict.fromkeys(ENGINE_RESOURCES, 0)
        second_px = board.vertex_index_to_pixel_dict[own[1].settlement_vertex]
        for hex_idx in board.boardGraph[second_px].adjacent_hex_indices:
            res = board.hexTileDict[hex_idx].resource_type
            if res != "DESERT":
                hand[res] += 1
        hands[seat] = hand

    return BridgeState(
        hexes=hexes,
        robber_hex=int(robber[0]),
        port_assignment=board.get_port_assignment(),
        placements=placements,
        hands=hands,
        agent_seat=agent_seat,
        # The seat is DRIVEN by the heuristic regardless — only the obs
        # opponent-id stamp varies (see ``ANCHOR_OPPONENT_KINDS``).
        opponent_type="heuristic",
        opp_kind=int(opp_kind),
        opp_policy_id=N_OPP_POLICY_SLOTS - 1,
        # D6: ``ptr_v1_u500.pt`` carries no ruleset stamp, so the lineage this
        # fine-tune extends is R0. Pinned, not defaulted.
        ruleset=RULESET_R0,
    )


def _has_unsatisfiable_head(masks: dict[str, np.ndarray], action_type: int) -> bool:
    """True iff a head this type needs has NO legal index at all.

    The env can offer a type whose relevant sub-head mask is entirely False —
    ``PLAY_KNIGHT`` under the Friendly Robber is the live case: the type is legal
    (a knight is in hand) while ``tile`` is empty because every hex is protected.
    ``heads.masked_log_softmax`` documents its uniform-placeholder fallback for
    exactly that row, the env's ``PLAY_KNIGHT`` branch ignores ``tile_idx``
    entirely (it sets ``robber_placement_pending`` and the placement is a
    separate decision), and :func:`action_masked_legal` — which reads the shared
    ``HEAD_RELEVANCE`` table — reports the sampled index as illegal regardless.

    That is a pre-existing disagreement between the schema table and the env's
    apply path, not something this sampler introduces or is placed to fix; it is
    named here so the legality pin below stays a pin on DRIFT rather than a
    refusal of shipped behaviour.
    """
    for head in range(1, len(HEAD_RELEVANCE[action_type])):
        if not HEAD_RELEVANCE[action_type][head]:
            continue
        key = MASK_KEY_FOR[action_type][head]
        assert key is not None  # relevance implies a routed mask key
        if not np.asarray(masks[key], dtype=bool).any():
            return True
    return False


def _policy_action(
    policy: CatanPolicy,
    obs: dict[str, np.ndarray],
    masks: dict[str, np.ndarray],
    device: torch.device,
) -> list[int]:
    """One action SAMPLED from ``policy`` at this node.

    The emitted action is checked against :func:`action_masked_legal` — the same
    gate the BC writer uses — whenever that check is answerable (see
    :func:`_has_unsatisfiable_head`). The heads already mask, so this is a cheap
    pin against mask-key / head-schema drift rather than a correction: a failure
    means the policy's mask routing and the env's have diverged, which must
    surface loudly rather than as a silent env no-op.
    """
    with torch.no_grad():
        out = policy.sample(
            obs_to_torch(obs, device, add_batch=True),
            masks_to_torch(masks, device, add_batch=True),
        )
    action = [int(x) for x in out["action"].reshape(-1).tolist()]
    if not action_masked_legal(masks, action) and not _has_unsatisfiable_head(masks, action[0]):
        raise NoLabeledOpeningsError(  # pragma: no cover - defensive
            f"the walker sampled {action}, which is illegal under the env's masks "
            f"even though every relevant head had a legal index — the policy's "
            f"mask routing and the env's disagree"
        )
    return action


def sample_anchor_states(
    labels_path: str | Path,
    *,
    policy: CatanPolicy,
    n_states: int,
    rng: np.random.Generator,
    device: torch.device = _CPU_DEVICE,
    max_steps_per_game: int = 60,
    exclude_game_seeds: Iterable[int] = (),
    opponent_kinds: tuple[int, ...] = ANCHOR_OPPONENT_KINDS,
) -> list[dict[str, Any]]:
    """Collect ``n_states`` NON-setup ``{"obs", "mask", "action"}`` samples.

    Games are started from labeled openings (both seatings, so the anchor is not
    a one-seat view), stamped across ``opponent_kinds`` (so the term covers the
    same id-conditional slices the human shard moves), and walked forward by
    ``policy`` — see the module docstring for why the walker is the frozen
    champion, sampled, and not defaultable.

    ``exclude_game_seeds`` drops the drafts withheld for the D7 gate-1
    measurement, so a held-out opening reaches the candidate through no path at
    all.

    Args:
        policy: the FROZEN champion. Required; ``rng`` still drives the opening /
            seat / opponent-kind choice, but the plies come from the policy's own
            (torch-seeded) sampling stream.

    Raises:
        NoLabeledOpeningsError: if the corpus holds no complete draft. There is
            deliberately no heuristic/random fallback.
    """
    openings = complete_openings(labels_path, exclude_game_seeds=exclude_game_seeds)
    if not openings:
        raise NoLabeledOpeningsError(
            f"{labels_path} contains no COMPLETE labeled opening (a row at draft "
            f"position 4) outside the held-out set. The anchor term must be "
            f"evaluated on the state distribution the owner's openings lead to; "
            f"falling back to heuristic or random openings would anchor the "
            f"wrong distribution."
        )

    out: list[dict[str, Any]] = []
    attempt = 0
    while len(out) < n_states:
        attempt += 1
        if attempt > 8 * n_states:  # pragma: no cover - defensive
            raise NoLabeledOpeningsError(
                f"collected only {len(out)}/{n_states} anchor states from "
                f"{len(openings)} labeled openings"
            )
        opening = openings[int(rng.integers(len(openings)))]
        agent_seat = int(rng.integers(2))
        # CYCLED, not sampled: a pool this small (a handful of games per call)
        # would leave a kind unrepresented often enough for the anchor to be
        # blind to a slice by luck. Round-robin makes the coverage a property of
        # the sampler rather than of the seed.
        opp_kind = int(opponent_kinds[(attempt - 1) % len(opponent_kinds)])
        env = rebuild_env(_bridge_state(opening, agent_seat=agent_seat, opp_kind=opp_kind))
        for _ in range(max_steps_per_game):
            if len(out) >= n_states:
                break
            if env.initial_placement_phase:  # pragma: no cover - rebuild is post-setup
                break
            masks = env.get_action_masks()
            obs = env._get_obs()
            action = _policy_action(policy, obs, masks, device)
            out.append(
                {
                    "obs": {k: np.asarray(v) for k, v in obs.items()},
                    "mask": {k: np.asarray(v, dtype=bool) for k, v in masks.items()},
                    "action": np.asarray(action, dtype=np.int64),
                }
            )
            _o, _r, terminated, truncated, _i = env.step(np.asarray(action, dtype=np.int64))
            if terminated or truncated:
                break
    return cast(list[dict[str, Any]], out[:n_states])
