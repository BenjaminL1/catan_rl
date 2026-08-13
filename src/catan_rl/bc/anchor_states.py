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

from catan_rl.env.ruleset import RULESET_R0
from catan_rl.human_data.engine_bridge import BridgeState, SeatPlacement, rebuild_env
from catan_rl.labeling.scenario_gen import Pick, ScenarioGenerator
from catan_rl.labeling.store import load_scenarios
from catan_rl.labeling.to_shard import DEFAULT_OPPONENT_KINDS
from catan_rl.policy.obs_schema import (
    AUTOREGRESSIVE_HEAD,
    HEAD_RELEVANCE,
    MASK_KEY_FOR,
    N_OPP_POLICY_SLOTS,
    OPP_KIND_HEURISTIC,
    ActionType,
    action_masked_legal,
)

ENGINE_RESOURCES: tuple[str, ...] = ("ORE", "BRICK", "WHEAT", "WOOD", "SHEEP")

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


def _candidates(masks: dict[str, np.ndarray], action_type: int, head: int, res1: int) -> np.ndarray:
    """Legal indices for one RELEVANT head, routed exactly as the mask schema says."""
    if head == AUTOREGRESSIVE_HEAD:
        # Head 5 mirrors ``heads._resource2_mask`` (see ``action_masked_legal``).
        if action_type == ActionType.BANK_TRADE:
            cand = np.flatnonzero(np.asarray(masks["resource2_trade"], dtype=bool))
            return cand[cand != res1]
        same = np.flatnonzero(np.asarray(masks["resource2_yop_same"], dtype=bool))
        diff = np.flatnonzero(np.asarray(masks["resource2_yop"], dtype=bool))
        return np.concatenate([same[same == res1], diff[diff != res1]])
    key = MASK_KEY_FOR[action_type][head]
    assert key is not None  # relevance implies a routed mask key
    return np.flatnonzero(np.asarray(masks[key], dtype=bool))


def _sample_legal_action(masks: dict[str, np.ndarray], rng: np.random.Generator) -> list[int]:
    """A uniformly random LEGAL action under ``masks``.

    The anchor term measures how far the fine-tuned policy has moved from the
    champion AT a state; the policy that walked the game there does not enter
    that measurement, so a cheap legal walker is enough and keeps the sampler
    free of a second forward pass per step.

    Sub-head indices are routed through the shared ``MASK_KEY_FOR`` /
    ``HEAD_RELEVANCE`` tables rather than a hand-written key list, so a mask
    schema change cannot leave this walker emitting actions the env silently
    no-ops on. The result is checked against :func:`action_masked_legal` — the
    same gate the BC writer uses — before it is returned.

    Raises:
        NoLabeledOpeningsError: if no legal action exists at all, which would
            mean the env handed back an empty type mask.
    """
    legal_types = np.flatnonzero(np.asarray(masks["type"], dtype=bool))
    if legal_types.size == 0:  # pragma: no cover - the env always offers one
        raise NoLabeledOpeningsError("env offered no legal action type")
    # Random order, then deterministic exhaustion: a type whose relevant
    # sub-head has an empty mask is skipped rather than emitted illegally.
    order = list(rng.permutation(legal_types))
    for a_type in order:
        action = [int(a_type), 0, 0, 0, 0, 0]
        ok = True
        for head in range(1, 6):
            if not HEAD_RELEVANCE[int(a_type)][head]:
                continue
            cand = _candidates(masks, int(a_type), head, action[4])
            if cand.size == 0:
                ok = False
                break
            action[head] = int(rng.choice(cand))
        if ok and action_masked_legal(masks, action):
            return action
    raise NoLabeledOpeningsError(  # pragma: no cover - defensive
        "no legal action could be assembled from the env's masks"
    )


def sample_anchor_states(
    labels_path: str | Path,
    *,
    n_states: int,
    rng: np.random.Generator,
    max_steps_per_game: int = 60,
    exclude_game_seeds: Iterable[int] = (),
    opponent_kinds: tuple[int, ...] = ANCHOR_OPPONENT_KINDS,
) -> list[dict[str, Any]]:
    """Collect ``n_states`` NON-setup ``{"obs", "mask", "action"}`` samples.

    Games are started from labeled openings (both seatings, so the anchor is not
    a one-seat view), stamped across ``opponent_kinds`` (so the term covers the
    same id-conditional slices the human shard moves), and walked forward with
    legal actions.

    ``exclude_game_seeds`` drops the drafts withheld for the D7 gate-1
    measurement, so a held-out opening reaches the candidate through no path at
    all.

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
            action = _sample_legal_action(masks, rng)
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
