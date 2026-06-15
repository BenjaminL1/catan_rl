"""Phase 2d — recorder driver loop.

Orchestrates a single 1v1 game from ``env.reset`` through terminal,
producing a :class:`Replay`. Hooks Phase 2a/2b/2c pieces (snapshots,
event collector, sub-action extraction, setup-burst splitter) together
with a SETUP_COMPLETE boundary listener and assembles the per-step
:class:`ReplayStep` list + :class:`Metadata`.

The recorder is policy-agnostic — it consumes :class:`Actor` instances
via :func:`catan_rl.replay.player_factory.build_actor`. The env hosts
exactly one "agent" slot (driven by ``env.step``) and one "opp" slot
(driven internally by ``opponent_type``); the matchup-to-seat logic
below decides which spec is the agent vs the opp.

**Currently supported matchups** (6 of the 8 valid pairs):

* ``(policy, *)`` — agent slot is the policy, opp is ``*``.
* ``(*, policy)`` — agent slot is the policy (seat=1), opp is ``*``.
* ``(random, *)`` — agent slot is mask-based random, opp is ``*``.
* ``(*, random)`` — when ``*`` is heuristic, agent slot is random (seat=1).

**Not yet supported** (raise :class:`NotImplementedError`):

* ``(heuristic, random)`` and ``(heuristic, heuristic)`` — would
  require heuristic-as-agent, which means translating
  :class:`heuristicAIPlayer.move` into discrete action vectors. The
  recorder errors out cleanly; Phase 4's matchup-parametrized smoke
  test marks these as xfail.
* ``(policy, policy)`` — the env only hosts one policy; the v2 plan
  defers this to the Phase 8 snapshot-opponent path.
"""

from __future__ import annotations

import datetime as _dt
import logging
from typing import Any

import numpy as np

from catan_rl.engine.broadcast import BroadcastEventType
from catan_rl.replay.player_factory import PlayerSpec as RecorderPlayerSpec
from catan_rl.replay.player_factory import build_actor
from catan_rl.replay.recorder import (
    EventCollector,
    classify_step_events,
    extract_sub_actions,
    snapshot_step_state,
    split_burst_one_placement,
    split_burst_two_placements,
    synthesize_intermediate_setup_snapshot,
)
from catan_rl.replay.schema import (
    REPLAY_SCHEMA_VERSION,
    BoardStatic,
    EdgeStatic,
    HexStatic,
    Metadata,
    PlayerSpec,
    PortStatic,
    Replay,
    ReplayStep,
    StepEvent,
    StepStateSnapshot,
    SubAction,
    VertexStatic,
)

_LOG = logging.getLogger("catan_rl.replay")

_RESOURCE_NAMES = {"WOOD", "BRICK", "WHEAT", "ORE", "SHEEP", "DESERT"}


def _resolve_seat_and_opp(
    spec_a: RecorderPlayerSpec, spec_b: RecorderPlayerSpec
) -> tuple[int, str]:
    """Decide ``(agent_seat, opp_kind)`` for the env.

    Returns ``(agent_seat, opp_kind)`` where ``opp_kind`` is one of
    ``"random" | "heuristic"`` (the env only supports those two
    opponent types). Raises :class:`NotImplementedError` for matchups
    that the current env cannot host.
    """
    if spec_a.kind == "policy" and spec_b.kind == "policy":
        raise NotImplementedError(
            "(policy, policy) recording is not supported in v1. The v2 "
            "env hosts one policy only; the Phase 8 snapshot-opponent "
            "path is the planned home for this matchup."
        )
    if spec_a.kind == "policy":
        return 0, spec_b.kind
    if spec_b.kind == "policy":
        return 1, spec_a.kind
    # Neither side is a policy. Prefer to put non-random as the opp
    # (env-driven) and random as the agent (recorder-driven), so the
    # "harder" engine logic stays in the engine. Specifically: if
    # exactly one side is heuristic, that side becomes the opp; the
    # random side drives env.step.
    if spec_a.kind == "heuristic" and spec_b.kind == "random":
        raise NotImplementedError(
            "(heuristic, random) is not supported until heuristic-as-"
            "agent lands. Workaround: use (random, heuristic) — the "
            "JSON's player_a/player_b seats are symmetrised in a "
            "downstream Phase 4 smoke."
        )
    if spec_a.kind == "heuristic" and spec_b.kind == "heuristic":
        raise NotImplementedError(
            "(heuristic, heuristic) is not supported until heuristic-as-agent lands."
        )
    # Both are random, or (random, heuristic): seat 0, opp = spec_b.
    return 0, spec_b.kind


def _seat_to_actor(agent_seat: int) -> dict[str, str]:
    """Engine player name → JSON actor name. ``"Agent"`` is the env's
    ``agent_player`` (the side that drives ``env.step``);
    ``"Opponent"`` is the env-internal opp."""
    if agent_seat == 0:
        return {"Agent": "player_a", "Opponent": "player_b"}
    return {"Agent": "player_b", "Opponent": "player_a"}


def _build_board_static_from_dict(raw: dict[str, Any]) -> BoardStatic:
    """Convert :meth:`catanBoard.board_static`'s JSON-safe dict into a
    frozen :class:`BoardStatic` dataclass."""
    hexes = tuple(
        HexStatic(
            hex_idx=int(h["hex_idx"]),
            q=int(h["q"]),
            r=int(h["r"]),
            resource=str(h["resource"]),  # type: ignore[arg-type]
            number_token=(int(h["number_token"]) if h["number_token"] is not None else None),
            has_robber_initial=bool(h["has_robber_initial"]),
        )
        for h in raw["hexes"]
    )
    vertices = tuple(
        VertexStatic(
            vertex_idx=int(v["vertex_idx"]),
            adjacent_hex_indices=tuple(int(h) for h in v["adjacent_hex_indices"]),
        )
        for v in raw["vertices"]
    )
    edges = tuple(
        EdgeStatic(
            edge_idx=int(e["edge_idx"]),
            v1_idx=int(e["v1_idx"]),
            v2_idx=int(e["v2_idx"]),
        )
        for e in raw["edges"]
    )
    ports = tuple(
        PortStatic(
            port_idx=int(p["port_idx"]),
            vertex_idx_pair=(int(p["vertex_idx_pair"][0]), int(p["vertex_idx_pair"][1])),
            ratio=str(p["ratio"]),  # type: ignore[arg-type]
            resource=(str(p["resource"]) if p["resource"] is not None else None),
        )
        for p in raw["ports"]
    )
    return BoardStatic(hexes=hexes, vertices=vertices, edges=edges, ports=ports)


# Action-type → mask-key mapping for the resource1 head. The env's
# mask dict carries three separate resource1 masks (trade / discard /
# default); the right one depends on the action type. Keys match
# ``env/masks.py``'s ``_pack`` output.
_RESOURCE1_MASK_BY_TYPE = {
    10: "resource1_trade",  # BANK_TRADE
    11: "resource1_discard",  # DISCARD
}


def _corner_mask_for_type(masks: dict[str, np.ndarray], action_type: int) -> np.ndarray | None:
    """Pick the right corner mask for ``action_type``. ``BUILD_SETTLEMENT``
    (0) uses ``corner_settlement``; ``BUILD_CITY`` (1) uses
    ``corner_city``. Setup-phase settle steps also map to
    ``corner_settlement`` because the env's mask builder packs the
    setup-legal vertices into that mask key."""
    if action_type == 1:
        return masks.get("corner_city")
    return masks.get("corner_settlement")


def _resource1_mask_for_type(masks: dict[str, np.ndarray], action_type: int) -> np.ndarray | None:
    key = _RESOURCE1_MASK_BY_TYPE.get(action_type, "resource1_default")
    return masks.get(key)


class _RandomMaskedActor:
    """Mask-aware random sampler used when the agent slot is the
    ``random`` kind. Picks a legal type, then a legal sub-arg per head
    based on the type-conditioned mask. Falls back to ``EndTurn`` if
    no type is legal.

    Replaces the partial ``_EngineDrivenActor.select_action`` (which
    only sampled action[0]) for the recorder's needs — the player
    factory's stub is fine for BC dataset generation where only the
    type matters, but the recorder actually executes the action in the
    env so every relevant head must be legal.

    Critically: the env's mask dict carries SEPARATE corner masks for
    settlement vs city, and three separate resource1 masks for trade /
    discard / default. Reading a single ``"corner"`` or
    ``"resource1"`` key (as an earlier draft did) silently fell
    through to ``action[*]=0`` and let invalid actions reach the
    engine — which in turn either silently failed or built at
    vertex 0 (whatever is at vertex 0 became a magnet for randomly-
    chosen settles in 1-of-N seeds). The type-conditioned lookup
    below pins legality.
    """

    kind = "random"

    def __init__(self, seed: int) -> None:
        self._rng = np.random.default_rng(seed)

    def _pick_from_mask(self, mask: np.ndarray | None) -> int:
        if mask is None:
            return 0
        legal = np.flatnonzero(mask)
        if not legal.size:
            return 0
        return int(self._rng.choice(legal))

    def select_action(
        self,
        obs: dict[str, np.ndarray],
        masks: dict[str, np.ndarray],
    ) -> np.ndarray:
        del obs  # unused — random doesn't look at obs
        type_mask = masks["type"]
        legal_types = np.flatnonzero(type_mask)
        action = np.zeros(6, dtype=np.int64)
        if not legal_types.size:
            action[0] = 3  # END_TURN fallback
            return action
        action[0] = int(self._rng.choice(legal_types))
        action_type = int(action[0])
        action[1] = self._pick_from_mask(_corner_mask_for_type(masks, action_type))
        action[2] = self._pick_from_mask(masks.get("edge"))
        action[3] = self._pick_from_mask(masks.get("tile"))
        action[4] = self._pick_from_mask(_resource1_mask_for_type(masks, action_type))
        action[5] = self._pick_from_mask(masks.get("resource2_default"))
        return action


def _resolve_agent_actor(
    spec_a: RecorderPlayerSpec,
    spec_b: RecorderPlayerSpec,
    *,
    agent_seat: int,
    seed: int,
    device: str,
) -> Any:
    """Build the actor that drives ``env.step``. For ``policy``, uses
    :func:`build_actor`. For ``random``, uses :class:`_RandomMaskedActor`
    (a full mask sampler — the factory's ``_EngineDrivenActor`` only
    samples ``action[0]`` which is insufficient for the recorder)."""
    agent_spec = spec_a if agent_seat == 0 else spec_b
    if agent_spec.kind == "policy":
        return build_actor(agent_spec, seed=seed, device=device)
    if agent_spec.kind == "random":
        return _RandomMaskedActor(seed=seed)
    # Heuristic-as-agent reaches this branch via _resolve_seat_and_opp's
    # filter only when both sides are heuristic; that case already
    # raised NotImplementedError. Defensive raise here covers future
    # extensions.
    raise NotImplementedError(
        f"agent slot kind {agent_spec.kind!r} not supported as the env's driving actor"
    )


def _dice_roll_from_events(events: list[dict]) -> tuple[int, int] | None:
    """Pull the first ``DICE_ROLL`` event in ``events`` and split its
    ``value`` (the two-dice sum) into a pair ``(d1, d2)``. The engine
    only carries the sum — ``StackedDice`` doesn't expose individual
    die values. We synthesize a plausible split (``(value//2, value - value//2)``)
    so the JSON contract is satisfied; the viewer treats this as
    cosmetic only since the engine's behavior is determined by the
    sum alone."""
    for ev in events:
        if ev.get("type") == BroadcastEventType.DICE_ROLL.value:
            value = int(ev.get("value", 0))
            if value <= 0:
                return None
            d1 = max(1, min(6, value // 2))
            d2 = max(1, min(6, value - d1))
            return (d1, d2)
    return None


def _replay_step(
    *,
    step_idx: int,
    kind: str,
    actor: str,
    dice_roll: tuple[int, int] | None,
    actions: tuple[SubAction, ...],
    events: tuple[StepEvent, ...],
    log_lines: tuple[str, ...],
    state_after: StepStateSnapshot,
) -> ReplayStep:
    """Thin wrapper that pins keyword args to make the
    :class:`ReplayStep` construction sites grep-friendly."""
    return ReplayStep(
        step_idx=step_idx,
        kind=kind,  # type: ignore[arg-type]
        actor=actor,  # type: ignore[arg-type]
        dice_roll=dice_roll,
        actions=actions,
        events=events,
        log_lines=log_lines,
        state_after=state_after,
    )


def _setup_steps_seat_0(
    *,
    agent_actions: list[tuple[int, int]],  # [(s1, r1), (s2, r2)] in idx terms
    snap_after_step1: StepStateSnapshot,  # agent.s1+r1, opp full burst
    setup_complete_snap: StepStateSnapshot,  # everything done, no main
    seat_to_actor: dict[str, str],
) -> tuple[list[ReplayStep], list[str]]:
    """Build the 4 setup ReplaySteps for ``agent_seat=0``.

    Order: ``[a, b, b, a]`` (snake draft from agent's POV).
    """
    log_lines: list[str] = []
    steps: list[ReplayStep] = []

    agent_actor_label = seat_to_actor["Agent"]
    opp_actor_label = seat_to_actor["Opponent"]

    s1_idx, r1_idx = agent_actions[0]
    s2_idx, r2_idx = agent_actions[1]

    # Step 0: agent's first placement (s1 + r1).
    # state_after = "agent.s1+r1 done, opp empty". We synthesize this
    # from snap_after_step1 by zeroing opp's placements and resources.
    step0_state = _zero_actor_in_snapshot(snap_after_step1, opp_actor_label)
    steps.append(
        _replay_step(
            step_idx=0,
            kind="setup",
            actor=agent_actor_label,
            dice_roll=None,
            actions=(
                SubAction(kind="BuildSettlement", args={"vertex_idx": s1_idx}),
                SubAction(kind="BuildRoad", args={"edge_idx": r1_idx}),
            ),
            events=(),
            log_lines=(
                f"{agent_actor_label} placed settlement at {s1_idx}",
                f"{agent_actor_label} placed road at {r1_idx}",
            ),
            state_after=step0_state,
        )
    )
    log_lines.extend(steps[-1].log_lines)

    # Steps 1 + 2: opp's burst (two placements) — split via Phase 2c.
    opp_first, opp_second = split_burst_two_placements(
        actor=opp_actor_label,
        prev_snapshot=step0_state,
        post_snapshot=snap_after_step1,
    )
    opp_mid = synthesize_intermediate_setup_snapshot(
        actor=opp_actor_label,
        post_snapshot=snap_after_step1,
        first_settle_idx=opp_first[0],
        first_road_idx=opp_first[1],
    )
    steps.append(
        _replay_step(
            step_idx=1,
            kind="setup",
            actor=opp_actor_label,
            dice_roll=None,
            actions=(
                SubAction(kind="BuildSettlement", args={"vertex_idx": opp_first[0]}),
                SubAction(kind="BuildRoad", args={"edge_idx": opp_first[1]}),
            ),
            events=(),
            log_lines=(
                f"{opp_actor_label} placed settlement at {opp_first[0]}",
                f"{opp_actor_label} placed road at {opp_first[1]}",
            ),
            state_after=opp_mid,
        )
    )
    log_lines.extend(steps[-1].log_lines)
    steps.append(
        _replay_step(
            step_idx=2,
            kind="setup",
            actor=opp_actor_label,
            dice_roll=None,
            actions=(
                SubAction(kind="BuildSettlement", args={"vertex_idx": opp_second[0]}),
                SubAction(kind="BuildRoad", args={"edge_idx": opp_second[1]}),
            ),
            events=(),
            log_lines=(
                f"{opp_actor_label} placed settlement at {opp_second[0]}",
                f"{opp_actor_label} placed road at {opp_second[1]}",
            ),
            state_after=snap_after_step1,
        )
    )
    log_lines.extend(steps[-1].log_lines)

    # Step 3: agent's second placement (s2 + r2). state_after =
    # everything done = setup_complete_snap.
    steps.append(
        _replay_step(
            step_idx=3,
            kind="setup",
            actor=agent_actor_label,
            dice_roll=None,
            actions=(
                SubAction(kind="BuildSettlement", args={"vertex_idx": s2_idx}),
                SubAction(kind="BuildRoad", args={"edge_idx": r2_idx}),
            ),
            events=(),
            log_lines=(
                f"{agent_actor_label} placed settlement at {s2_idx}",
                f"{agent_actor_label} placed road at {r2_idx}",
            ),
            state_after=setup_complete_snap,
        )
    )
    log_lines.extend(steps[-1].log_lines)

    return steps, log_lines


def _setup_steps_seat_1(
    *,
    agent_actions: list[tuple[int, int]],  # [(s1, r1), (s2, r2)]
    snap_after_reset: StepStateSnapshot,  # opp.s1+r1 already placed
    snap_after_step1: StepStateSnapshot,  # opp.s1+r1, agent.s1+r1
    setup_complete_snap: StepStateSnapshot,  # everything done, no main
    seat_to_actor: dict[str, str],
) -> tuple[list[ReplayStep], list[str]]:
    """Build the 4 setup ReplaySteps for ``agent_seat=1``.

    Order: ``[a, b, b, a]`` (player_a is the opp; player_b is agent).
    """
    log_lines: list[str] = []
    steps: list[ReplayStep] = []

    agent_actor_label = seat_to_actor["Agent"]  # player_b
    opp_actor_label = seat_to_actor["Opponent"]  # player_a

    # Step 0: opp's first placement (already done in env.reset).
    opp_first = split_burst_one_placement(
        actor=opp_actor_label,
        prev_snapshot=_empty_state_like(snap_after_reset),
        post_snapshot=snap_after_reset,
    )
    steps.append(
        _replay_step(
            step_idx=0,
            kind="setup",
            actor=opp_actor_label,
            dice_roll=None,
            actions=(
                SubAction(kind="BuildSettlement", args={"vertex_idx": opp_first[0]}),
                SubAction(kind="BuildRoad", args={"edge_idx": opp_first[1]}),
            ),
            events=(),
            log_lines=(
                f"{opp_actor_label} placed settlement at {opp_first[0]}",
                f"{opp_actor_label} placed road at {opp_first[1]}",
            ),
            state_after=snap_after_reset,
        )
    )
    log_lines.extend(steps[-1].log_lines)

    # Step 1: agent's first placement (s1 + r1).
    s1_idx, r1_idx = agent_actions[0]
    steps.append(
        _replay_step(
            step_idx=1,
            kind="setup",
            actor=agent_actor_label,
            dice_roll=None,
            actions=(
                SubAction(kind="BuildSettlement", args={"vertex_idx": s1_idx}),
                SubAction(kind="BuildRoad", args={"edge_idx": r1_idx}),
            ),
            events=(),
            log_lines=(
                f"{agent_actor_label} placed settlement at {s1_idx}",
                f"{agent_actor_label} placed road at {r1_idx}",
            ),
            state_after=snap_after_step1,
        )
    )
    log_lines.extend(steps[-1].log_lines)

    # Step 2: agent's second placement (s2 + r2). state_after = the
    # setup_complete snap with opp's SECOND placement REVERSED — i.e.,
    # "agent full setup + resources, opp only first placement, no opp
    # second-grant resources".
    s2_idx, r2_idx = agent_actions[1]
    # Diff: opp's second placement = whatever's in setup_complete_snap
    # but not in snap_after_step1.
    opp_second = split_burst_one_placement(
        actor=opp_actor_label,
        prev_snapshot=snap_after_step1,
        post_snapshot=setup_complete_snap,
    )
    step2_state = synthesize_intermediate_setup_snapshot(
        actor=opp_actor_label,
        post_snapshot=setup_complete_snap,
        first_settle_idx=opp_first[0],
        first_road_idx=opp_first[1],
    )
    # synthesize set actor (opp) to vp=1, resources=0. We need to
    # ALSO promote agent's state (the OTHER actor) by adding the
    # second settle+road... but synthesize already takes the OTHER
    # state unchanged from setup_complete_snap, which has agent's
    # full setup. So step2_state already reflects "agent full,
    # opp at first placement only".
    steps.append(
        _replay_step(
            step_idx=2,
            kind="setup",
            actor=agent_actor_label,
            dice_roll=None,
            actions=(
                SubAction(kind="BuildSettlement", args={"vertex_idx": s2_idx}),
                SubAction(kind="BuildRoad", args={"edge_idx": r2_idx}),
            ),
            events=(),
            log_lines=(
                f"{agent_actor_label} placed settlement at {s2_idx}",
                f"{agent_actor_label} placed road at {r2_idx}",
            ),
            state_after=step2_state,
        )
    )
    log_lines.extend(steps[-1].log_lines)

    # Step 3: opp's second placement. state_after = setup_complete_snap.
    steps.append(
        _replay_step(
            step_idx=3,
            kind="setup",
            actor=opp_actor_label,
            dice_roll=None,
            actions=(
                SubAction(kind="BuildSettlement", args={"vertex_idx": opp_second[0]}),
                SubAction(kind="BuildRoad", args={"edge_idx": opp_second[1]}),
            ),
            events=(),
            log_lines=(
                f"{opp_actor_label} placed settlement at {opp_second[0]}",
                f"{opp_actor_label} placed road at {opp_second[1]}",
            ),
            state_after=setup_complete_snap,
        )
    )
    log_lines.extend(steps[-1].log_lines)

    return steps, log_lines


def _empty_state_like(snap: StepStateSnapshot) -> StepStateSnapshot:
    """Return a snapshot with the same actor keys as ``snap`` but all
    placements / resources / VP zeroed. Used as the "prev" snapshot
    for split_burst_* helpers when no earlier snapshot exists (e.g.,
    seat=1's opp first placement happens during env.reset, so there
    is no pre-reset snapshot)."""
    from catan_rl.replay.schema import PlayerStateSnapshot

    empty_players = {}
    for actor, player_snap in snap.players.items():
        empty_players[actor] = PlayerStateSnapshot(
            name=player_snap.name,
            vp=0,
            resources={k: 0 for k in player_snap.resources},
            dev_cards_hand={k: 0 for k in player_snap.dev_cards_hand},
            dev_cards_played={k: 0 for k in player_snap.dev_cards_played},
        )
    return StepStateSnapshot(
        settlements={actor: () for actor in snap.settlements},
        cities={actor: () for actor in snap.cities},
        roads={actor: () for actor in snap.roads},
        robber_hex=snap.robber_hex,
        players=empty_players,
        longest_road_holder=None,
        largest_army_holder=None,
        last_seven_roller=None,
    )


def _zero_actor_in_snapshot(snap: StepStateSnapshot, actor: str) -> StepStateSnapshot:
    """Return a copy of ``snap`` with ``actor``'s state reset to zero
    (no placements, no resources, vp=0). Used to synthesize
    "agent placed first, opp empty" intermediate states for seat=0
    where the engine processes opp's burst atomically inside one
    env.step. Other actor's state is preserved unchanged.
    """
    from catan_rl.replay.schema import PlayerStateSnapshot

    settlements = dict(snap.settlements)
    settlements[actor] = ()
    cities = dict(snap.cities)
    cities[actor] = ()
    roads = dict(snap.roads)
    roads[actor] = ()

    new_actor_snap = PlayerStateSnapshot(
        name=snap.players[actor].name,
        vp=0,
        resources={k: 0 for k in snap.players[actor].resources},
        dev_cards_hand={k: 0 for k in snap.players[actor].dev_cards_hand},
        dev_cards_played={k: 0 for k in snap.players[actor].dev_cards_played},
    )
    # Other-actor snapshot is preserved via shallow alias; PlayerStateSnapshot
    # is frozen so callers cannot rebind fields. The mutable dict
    # fields inside (resources, dev_cards_hand, dev_cards_played) are
    # aliased and treated as read-only by all recorder paths.
    players = dict(snap.players)
    players[actor] = new_actor_snap

    return StepStateSnapshot(
        settlements=settlements,
        cities=cities,
        roads=roads,
        robber_hex=snap.robber_hex,
        players=players,
        # Setup phase invariant: nobody can hold LR (needs 5 roads),
        # nobody can hold LA (needs 3 knights), and no 7 has been
        # rolled. Hardcode None here so future engine drift can't
        # leak a stale post-burst value into the synthesized "agent
        # placed first, opp empty" intermediate state.
        longest_road_holder=None,
        largest_army_holder=None,
        last_seven_roller=None,
    )


def _partition_main_events_by_actor(
    raw_events: list[dict],
    *,
    initial_actor: str,
    seat_to_actor: dict[str, str],
) -> list[tuple[str, list[dict]]]:
    """Split a stream of broadcast events into per-actor groups.

    The boundary marker is ``DICE_ROLL`` — each dice roll starts a
    new player's turn and the event's ``player`` field names them.
    Events before the first dice roll are attributed to
    ``initial_actor`` (the agent or opp, depending on context).

    Returns a list of ``(actor, events)`` tuples in temporal order.
    Empty groups (no events for an actor) are filtered out.
    """
    if not raw_events:
        return []
    from catan_rl.replay.recorder import _actor_for  # local import to keep module slim

    groups: list[tuple[str, list[dict]]] = []
    current_actor = initial_actor
    current_events: list[dict] = []
    for event in raw_events:
        etype = event.get("type")
        if etype == BroadcastEventType.DICE_ROLL.value:
            if current_events:
                groups.append((current_actor, current_events))
            mapped = _actor_for(event.get("player"), seat_to_actor)
            current_actor = mapped if mapped is not None else current_actor
            current_events = [event]
        elif etype == BroadcastEventType.DISCARD.value:
            mapped = _actor_for(event.get("player"), seat_to_actor)
            if mapped is not None and mapped != current_actor:
                # Forced discard by the NON-acting player: on a 7-roll BOTH
                # players over the discard threshold must discard, so the
                # opponent discards during the roller's turn. Attribute it to the
                # discarder's own group, then resume the roller — the DICE_ROLL
                # boundary alone would mis-credit the discard to the roller.
                if current_events:
                    groups.append((current_actor, current_events))
                    current_events = []
                groups.append((mapped, [event]))
            else:
                current_events.append(event)
        else:
            current_events.append(event)
    if current_events:
        groups.append((current_actor, current_events))
    return [(a, evs) for a, evs in groups if evs]


def _split_at_setup_complete(raw_events: list[dict]) -> list[dict]:
    """Return only events that occur AFTER the ``SETUP_COMPLETE``
    marker in ``raw_events``. Used to peel out opp's first main turn
    (seat=1 case) from the trailing portion of env.step #3."""
    for i, event in enumerate(raw_events):
        if event.get("type") == BroadcastEventType.SETUP_COMPLETE.value:
            return raw_events[i + 1 :]
    return []


def _consume_main_event_block(
    *,
    raw_events: list[dict],
    prev_snap: StepStateSnapshot,
    post_snap: StepStateSnapshot,
    initial_actor: str,
    seat_to_actor: dict[str, str],
    start_idx: int,
    terminated: bool,
    truncated: bool,
) -> tuple[list[ReplayStep], StepStateSnapshot, int]:
    """Turn a stream of main-phase broadcast events from a single
    env.step (or a residual buffer) into one or more :class:`ReplayStep`
    instances, partitioned by per-actor DICE_ROLL boundaries.

    The state_after for each emitted ReplayStep is ``post_snap`` — the
    granularity of the engine's snapshot accessor only resolves at
    env.step boundaries. The actor attribution within the same
    env.step is derived from the event ``player`` field; the snapshot
    is shared across all sub-groups. The viewer's step bar shows the
    end-of-env.step board for each per-actor step in this block — a
    minor cosmetic shared-tail, NOT a correctness issue (each step's
    sub_actions and events are correctly attributed via the per-actor
    diff in :func:`extract_sub_actions`).

    Returns ``(new_steps, last_snap, next_idx)``. The terminal flag is
    folded into the LAST emitted ReplayStep's ``kind`` field.
    """
    groups = _partition_main_events_by_actor(
        raw_events, initial_actor=initial_actor, seat_to_actor=seat_to_actor
    )
    if not groups:
        # No events but a single env.step still happened (e.g., agent
        # ROLL_DICE producing only an internal state change without
        # events? unlikely in practice). Emit a "filler" step
        # attributed to the initial actor for symmetry — the viewer
        # then shows no action change.
        return [], prev_snap, start_idx

    new_steps: list[ReplayStep] = []
    next_idx = start_idx
    for i, (actor_label, group_events) in enumerate(groups):
        is_last = i == len(groups) - 1
        dice_roll = _dice_roll_from_events(group_events)
        sub_actions = extract_sub_actions(
            group_events,
            prev_snapshot=prev_snap,
            curr_snapshot=post_snap,
            dice_roll=dice_roll,
            seat_to_actor=seat_to_actor,
        )
        step_events, log_lines = classify_step_events(group_events, seat_to_actor=seat_to_actor)
        kind = "terminal" if (is_last and (terminated or truncated)) else "main"
        new_steps.append(
            _replay_step(
                step_idx=next_idx,
                kind=kind,
                actor=actor_label,
                dice_roll=dice_roll,
                actions=tuple(sub_actions),
                events=tuple(step_events),
                log_lines=tuple(log_lines),
                state_after=post_snap,
            )
        )
        next_idx += 1
    return new_steps, post_snap, next_idx


def record_game(
    spec_a: RecorderPlayerSpec,
    spec_b: RecorderPlayerSpec,
    *,
    seed: int,
    max_turns: int = 400,
    intended_hex_size: tuple[int, int] = (1000, 800),
    device: str = "cpu",
    log: logging.Logger | None = None,
    agent_actor: Any | None = None,
) -> Replay:
    """Record one 1v1 Catan game between ``spec_a`` and ``spec_b``.

    Returns a fully populated :class:`Replay`. The caller is
    responsible for writing it via :func:`save_replay`.

    See module docstring for the supported matchups list.
    """
    log = log or _LOG

    agent_seat, opp_kind = _resolve_seat_and_opp(spec_a, spec_b)
    seat_to_actor = _seat_to_actor(agent_seat)
    # An injected ``agent_actor`` (e.g. a search agent that needs the live env)
    # overrides the spec-built one; the seat/opp resolution above still uses the
    # specs (spec_a a policy-kind placeholder, spec_b the env opponent).
    if agent_actor is None:
        agent_actor = _resolve_agent_actor(
            spec_a, spec_b, agent_seat=agent_seat, seed=seed, device=device
        )

    # Heavy import deferred — keeps the replay module importable from
    # the viewer (which has no torch / gymnasium).
    from catan_rl.env.catan_env import CatanEnv

    env = CatanEnv(opponent_type=opp_kind, max_turns=max_turns)
    obs, _info = env.reset(seed=seed, options={"agent_seat": agent_seat})
    # ``env.reset`` populates these — assertions narrow Optional → concrete
    # for mypy without runtime overhead beyond the None-check itself.
    assert env.game is not None
    assert env.agent_player is not None
    assert env.opponent_player is not None

    # A search agent needs the LIVE env (the obs is a lossy encoding); hand it
    # the env it drives. Plain actors don't expose bind_env -> no-op.
    if hasattr(agent_actor, "bind_env"):
        agent_actor.bind_env(env)

    # Index maps used by snapshot_step_state.
    vertex_pixel_to_idx = dict(env._vertex_to_idx)
    edge_key_to_idx = dict(env._edge_to_idx)

    def _snap_now() -> StepStateSnapshot:
        return snapshot_step_state(
            env.game,
            seat_to_actor=seat_to_actor,
            vertex_pixel_to_idx=vertex_pixel_to_idx,
            edge_key_to_idx=edge_key_to_idx,
        )

    # Subscribers: event buffer + setup-complete latch.
    event_collector = EventCollector()
    event_collector.subscribe(env.game.broadcast)

    setup_complete_snaps: list[StepStateSnapshot] = []

    def _setup_complete_cb(event: dict) -> None:
        if event.get("type") == BroadcastEventType.SETUP_COMPLETE.value:
            setup_complete_snaps.append(_snap_now())

    env.game.broadcast.subscribe(_setup_complete_cb)

    # Snapshot immediately after reset. For seat=1, this already has
    # opp's first placement; for seat=0 it's empty.
    snap_after_reset = _snap_now()
    # Drain any reset-time events (seat=1's opp-first-setup BUILDs
    # and the RESOURCE_CHANGE-from-grant events end up here). They
    # aren't surfaced as sub-actions for any ReplayStep — the seat-1
    # ReplayStep #0 derives its actions from the diff against an
    # empty state via split_burst_one_placement.
    _reset_events = event_collector.drain()

    # ----------------- SETUP PHASE -----------------
    # 4 env.step calls drive the agent through s1, r1, s2, r2.
    setup_action_pairs: list[tuple[int, int]] = []  # [(s1, r1), (s2, r2)]
    snap_after_step1: StepStateSnapshot | None = None  # after agent's r1 env.step (seat=0)

    sub_actions_for_step_index: dict[int, int] = {}  # env.step idx → vertex/edge idx
    for env_step_idx in range(4):
        masks = env.get_action_masks()
        action = agent_actor.select_action(obs, masks)
        sub_idx = (
            int(action[1]) if env_step_idx % 2 == 0 else int(action[2])
        )  # corner for settles (steps 0, 2); edge for roads (steps 1, 3)
        sub_actions_for_step_index[env_step_idx] = sub_idx

        obs, _r, terminated, truncated, _step_info = env.step(action)
        if terminated or truncated:
            # Setup-phase termination would be a bug — no one can hit
            # 15 VP from 2 settlements. Defensive: stop and warn.
            log.warning(
                "recorder: setup-phase env.step #%d returned terminated=%s "
                "truncated=%s — unexpected; aborting",
                env_step_idx,
                terminated,
                truncated,
            )
            break

        if env_step_idx == 1:
            snap_after_step1 = _snap_now()

    # The SETUP_COMPLETE event must have fired during env.step #3.
    if len(setup_complete_snaps) != 1:
        raise RuntimeError(
            f"recorder: expected exactly 1 SETUP_COMPLETE event, got "
            f"{len(setup_complete_snaps)}. Engine drift suspected."
        )
    setup_complete_snap = setup_complete_snaps[0]

    setup_action_pairs.append((sub_actions_for_step_index[0], sub_actions_for_step_index[1]))
    setup_action_pairs.append((sub_actions_for_step_index[2], sub_actions_for_step_index[3]))

    if agent_seat == 0:
        assert snap_after_step1 is not None, "seat=0 needs snap_after_step1"
        setup_steps, _setup_log_lines = _setup_steps_seat_0(
            agent_actions=setup_action_pairs,
            snap_after_step1=snap_after_step1,
            setup_complete_snap=setup_complete_snap,
            seat_to_actor=seat_to_actor,
        )
    else:
        # Seat 1: agent_actions are the agent's two placements (player_b);
        # opp's two placements come from snapshots (one at reset, one
        # from the setup_complete snap minus snap_after_step1).
        assert snap_after_step1 is not None, "seat=1 needs snap_after_step1"
        setup_steps, _setup_log_lines = _setup_steps_seat_1(
            agent_actions=setup_action_pairs,
            snap_after_reset=snap_after_reset,
            snap_after_step1=snap_after_step1,
            setup_complete_snap=setup_complete_snap,
            seat_to_actor=seat_to_actor,
        )

    # Drain residual events. For seat=0 these are all setup-phase
    # (BUILDs, RESOURCE_CHANGEs, SETUP_COMPLETE) and are reconstructed
    # via Phase 2c helpers, so they're cosmetic to drain. For seat=1
    # the env runs opp's FIRST MAIN TURN inside env.step #3 — those
    # events live AFTER SETUP_COMPLETE in this buffer and must NOT
    # be dropped, or the replay silently loses opp's first move.
    setup_residual_events = event_collector.drain()

    replay_steps: list[ReplayStep] = list(setup_steps)
    main_step_idx = len(replay_steps)
    prev_snap = setup_complete_snap

    # If the residual stream contains anything after SETUP_COMPLETE,
    # those are main-phase events (opp's first turn in the seat=1
    # case). Emit ReplaySteps for them now, BEFORE the agent's
    # first main env.step.
    post_setup_residual = _split_at_setup_complete(setup_residual_events)
    if post_setup_residual:
        snap_after_setup_loop = _snap_now()
        # Initial actor for residual: the agent's seat is the one
        # that just acted to finalise setup; the residual events are
        # opp's auto-started turn, so the first DICE_ROLL will tag
        # opp. Fallback initial_actor = seat_to_actor["Opponent"].
        new_steps, last_snap, main_step_idx = _consume_main_event_block(
            raw_events=post_setup_residual,
            prev_snap=prev_snap,
            post_snap=snap_after_setup_loop,
            initial_actor=seat_to_actor["Opponent"],
            seat_to_actor=seat_to_actor,
            start_idx=main_step_idx,
            terminated=False,
            truncated=False,
        )
        replay_steps.extend(new_steps)
        prev_snap = last_snap

    # ----------------- MAIN PHASE -----------------
    terminated = False
    truncated = False

    # Safety cap: each env.step typically produces 1-2 ReplaySteps;
    # max_turns is the per-agent turn cap, so 2 * max_turns + slack
    # comfortably bounds the loop.
    max_replay_steps = max_turns * 4 + 64

    while not terminated and not truncated and main_step_idx < max_replay_steps:
        masks = env.get_action_masks()
        action = agent_actor.select_action(obs, masks)
        obs, _r, terminated, truncated, _step_info = env.step(action)
        post_snap = _snap_now()
        raw_events = event_collector.drain()

        # Partition events by actor (DICE_ROLL boundaries). The
        # agent always acts first, so initial_actor is the agent.
        new_steps, last_snap, main_step_idx = _consume_main_event_block(
            raw_events=raw_events,
            prev_snap=prev_snap,
            post_snap=post_snap,
            initial_actor=seat_to_actor["Agent"],
            seat_to_actor=seat_to_actor,
            start_idx=main_step_idx,
            terminated=terminated,
            truncated=truncated,
        )
        replay_steps.extend(new_steps)
        prev_snap = last_snap

    # ----------------- METADATA -----------------
    # Decide winner_seat: 0 if seat-0 player won, 1 if seat-1 player
    # won, None if truncated without a winner.
    agent_vp = env.agent_player.victoryPoints
    opp_vp = env.opponent_player.victoryPoints
    if agent_vp >= env.game.maxPoints:
        winner_seat = agent_seat
        winner_actor: str | None = seat_to_actor["Agent"]
    elif opp_vp >= env.game.maxPoints:
        winner_seat = 1 - agent_seat
        winner_actor = seat_to_actor["Opponent"]
    else:
        winner_seat = None
        winner_actor = None

    # ``final_vp`` is ordered by seat: (vp_seat_0, vp_seat_1).
    final_vp = (int(agent_vp), int(opp_vp)) if agent_seat == 0 else (int(opp_vp), int(agent_vp))

    metadata = Metadata(
        player_a=PlayerSpec(
            kind=spec_a.kind,
            ckpt_path=spec_a.ckpt_path,
            color="black" if agent_seat == 0 else "darkslateblue",
            seat_index=0,
        ),
        player_b=PlayerSpec(
            kind=spec_b.kind,
            ckpt_path=spec_b.ckpt_path,
            color="darkslateblue" if agent_seat == 0 else "black",
            seat_index=1,
        ),
        seed=seed,
        max_turns=max_turns,
        intended_hex_size=intended_hex_size,
        recorded_at_utc=_dt.datetime.now(_dt.UTC).isoformat(timespec="seconds"),
        winner=winner_actor,
        winner_seat=winner_seat,
        final_vp=final_vp,
        total_steps=len(replay_steps),
        partial=False,
    )

    board_static = _build_board_static_from_dict(env.game.board.board_static())

    return Replay(
        schema_version=REPLAY_SCHEMA_VERSION,
        metadata=metadata,
        board_static=board_static,
        steps=tuple(replay_steps),
    )
