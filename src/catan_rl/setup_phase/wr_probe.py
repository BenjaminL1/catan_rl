"""D5 forced-opening win-rate probe: does the scorer's opening WIN more games?

The shape is ExIt STEP 2's: paired seeds, the SAME policy on both sides, and the
arms differ only in ONE SEAT's four setup picks. In the treatment arm the agent
seat's picks are forced to the scorer's argmax while the opposing seat drafts
with the policy and responds to them; in the control arm both seats draft with
the policy. The reported quantity is ΔWR with a CI.

Holding the counterfactual INSIDE the game is the whole design. Force both
seats from the same source and sum over both agent seats, and each arm is 0.5
by symmetry — ΔWR is then identically zero no matter how good the openings are,
and the spec's "ΔWR ~ 0 is AMBIGUOUS" reading would launder a mechanical
artifact as evidence.

Pre-registered reading, so it cannot be spun after the fact:

* **ΔWR > 0** — the opening's win-value survives this midgame. The scorer is
  teaching something real.
* **ΔWR ≈ 0** — AMBIGUOUS, and recorded as ambiguous. u500 may simply be unable
  to exploit a better opening; the probe cannot separate "the opening is
  worthless" from "this midgame wastes it".
* **large negative** — kills synthetic generation.

This module is the DRIVER. Running it for real costs machine time and needs the
fitted weights, so this slice ships it with fixture-scale tests only.

RNG discipline mirrors :meth:`catan_rl.eval.harness.EvalHarness.run`: the global
numpy/stdlib/torch streams are snapshotted and restored around a round, because
the games consume the process-global streams (StackedDice, heuristic, steal) and
a probe run inside another loop must not shift it.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import torch

from catan_rl.bc.anchor_states import LabeledOpening, bridge_state_for_opening
from catan_rl.env.catan_env import CatanEnv
from catan_rl.eval.harness import _restore_torch_rng, _snapshot_torch_rng
from catan_rl.human_data.engine_bridge import rebuild_env
from catan_rl.labeling.scenario_gen import Pick, Scenario, ScenarioGenerator
from catan_rl.labeling.to_shard import UNKNOWN_POLICY_ID, obs_and_mask_for_scenario
from catan_rl.policy.obs_encoder import ObsEncoder
from catan_rl.policy.obs_schema import OPP_KIND_LEAGUE
from catan_rl.policy.obs_tensor import masks_to_torch, obs_to_torch
from catan_rl.setup_phase.gate import paired_binary_difference
from catan_rl.setup_phase.scorer import SetupScorer

PROBE_OPP_KIND: int = OPP_KIND_LEAGUE
"""The opponent-id stamp the probe's games actually run under.

Both arms are played against a FROZEN SNAPSHOT of the same checkpoint, and
``CatanEnv._OPP_TYPE_TO_KIND`` maps ``"snapshot"`` to ``OPP_KIND_LEAGUE``.
Stamping ``OPP_KIND_HEURISTIC`` (as an earlier draft did, by leaving the
``bridge_state_for_opening`` default in place while overriding
``opponent_type``) tells the policy it is facing an opponent it is not, which
puts the absolute win rates off-distribution even though the paired delta
survives. The same stamp is used when reading the policy's own opening, so the
draft and the game it leads to ask the question under one condition.
"""


class ProbeError(RuntimeError):
    """Raised when an arm's opening cannot be derived or replayed.

    Deliberately fatal: a seed whose scorer opening is illegal must stop the
    probe, never quietly fall back to the other arm's opening — that would put
    the two arms on the same picks and report a ΔWR of exactly zero as if it
    were evidence.
    """


@dataclass(frozen=True)
class ArmOutcome:
    seed: int
    agent_seat: int
    won: bool
    truncated: bool


Chooser = Callable[[ScenarioGenerator, Scenario], tuple[int, int]]
"""Decides one draft pick: ``(gen, scenario) -> (settlement_vertex, road_edge)``."""


def _draft(seed: int, chooser_for_seat: Mapping[int, Chooser]) -> LabeledOpening:
    """Run the four-pick snake draft, each seat using its own chooser.

    A PER-SEAT chooser is what makes the probe's contrast non-degenerate.
    Forcing BOTH seats from the same source and then summing over both agent
    seats makes each arm's win rate 0.5 by symmetry and ΔWR identically zero,
    whatever the openings are. The counterfactual has to be held WITHIN the
    game — exactly ExIt STEP 2's shape: change one actor's decision rule, leave
    everything else in place and let the other actor respond to it.
    """
    gen = ScenarioGenerator(seed=seed)
    first = gen.current()
    if first is None:  # pragma: no cover - a fresh generator always has one
        raise ProbeError(f"seed {seed}: fresh generator yielded no scenario")
    picks: list[Pick] = []
    while (scenario := gen.current()) is not None:
        seat = int(scenario.acting_player_idx)
        try:
            chooser = chooser_for_seat[seat]
        except KeyError as exc:  # pragma: no cover - both seats are always given
            raise ProbeError(f"seed {seed}: no chooser for seat {seat}") from exc
        vertex, edge = chooser(gen, scenario)
        picks.append(Pick(player=seat, settlement_vertex=vertex, road_edge=edge))
        gen.apply(vertex, edge)
    return LabeledOpening(game_seed=seed, picks=tuple(picks))


def _scorer_chooser(scorer: SetupScorer, seed: int) -> Chooser:
    def choose(gen: ScenarioGenerator, scenario: Scenario) -> tuple[int, int]:
        v_scores = scorer.score_vertices(
            gen._board,
            scenario.prior_picks,
            int(scenario.acting_player_idx),
            scenario.legal_settlement_corners,
        )
        vertex = int(np.argmax(v_scores))
        if not np.isfinite(v_scores[vertex]):
            raise ProbeError(f"seed {seed}: no legal settlement at pick {scenario.draft_position}")
        legal_edges = scenario.compute_legal_road_edges(vertex)
        e_scores = scorer.score_edges(
            gen._board,
            scenario.prior_picks,
            int(scenario.acting_player_idx),
            scenario.legal_settlement_corners,
            vertex,
            legal_edges,
        )
        edge = int(np.argmax(e_scores))
        if not np.isfinite(e_scores[edge]):
            raise ProbeError(f"seed {seed}: no legal road after settlement {vertex}")
        return vertex, edge

    return choose


def _policy_chooser(policy: Any, seed: int, *, device: str = "cpu") -> Chooser:
    """The policy's argmax pick, read through the label-corpus encoder path.

    Derived through :func:`catan_rl.labeling.to_shard.obs_and_mask_for_scenario`
    — the SAME encoder + mask path the label corpus was converted with — so the
    two arms differ in the picks, not in how the question was asked.
    """
    from catan_rl.env.hand_tracker import BroadcastHandTracker

    torch_device = torch.device(device)
    state: dict[str, Any] = {"tracker": None}

    def choose(gen: ScenarioGenerator, scenario: Scenario) -> tuple[int, int]:
        if state["tracker"] is None:
            tracker = BroadcastHandTracker([p.name for p in gen._players])
            tracker.subscribe(scenario.game.broadcast)
            state["tracker"] = tracker
        tracker = state["tracker"]
        encoder = ObsEncoder(gen._board)
        vertex = _argmax_head(
            policy,
            gen=gen,
            scenario=scenario,
            encoder=encoder,
            tracker=tracker,
            setup_step=scenario._setup_step_road - 1,
            head="corner",
            device=torch_device,
        )
        if not bool(scenario.legal_settlement_corners[vertex]):
            raise ProbeError(
                f"seed {seed}: policy's top-1 corner {vertex} is illegal at pick "
                f"{scenario.draft_position}"
            )
        legal_edges = np.asarray(scenario.compute_legal_road_edges(vertex), dtype=bool)
        with _settlement_placed(scenario, gen, vertex):
            edge = _argmax_head(
                policy,
                gen=gen,
                scenario=scenario,
                encoder=encoder,
                tracker=tracker,
                setup_step=scenario._setup_step_road,
                head="edge",
                device=torch_device,
            )
        if not bool(legal_edges[edge]):
            raise ProbeError(
                f"seed {seed}: policy's top-1 edge {edge} is illegal after settlement {vertex}"
            )
        return vertex, edge

    return choose


def scorer_opening(scorer: SetupScorer, seed: int) -> LabeledOpening:
    """The four-pick draft the scorer would choose on the board ``seed``, BOTH seats.

    Kept for inspection and tests. The probe itself uses
    :func:`mixed_opening`: a both-seats-scorer opening cannot express the
    "only this actor changes" counterfactual D5 asks for.
    """
    chooser = _scorer_chooser(scorer, seed)
    return _draft(seed, {0: chooser, 1: chooser})


def policy_opening(policy: Any, seed: int, *, device: str = "cpu") -> LabeledOpening:
    """The four-pick draft ``policy`` would choose on the board ``seed``, BOTH seats.

    This is the probe's CONTROL arm: neither seat is overridden.
    """
    chooser = _policy_chooser(policy, seed, device=device)
    return _draft(seed, {0: chooser, 1: chooser})


def mixed_opening(
    scorer: SetupScorer,
    policy: Any,
    seed: int,
    *,
    scorer_seat: int,
    device: str = "cpu",
) -> LabeledOpening:
    """The treatment arm: ``scorer_seat``'s picks forced to the scorer's argmax.

    The other seat keeps choosing with the policy, live, so it RESPONDS to the
    overridden picks exactly as it would in a real draft. Paired against
    :func:`policy_opening` on the same seed and seat, the only thing that
    differs between the two games is which rule drove one seat's four picks.
    """
    if scorer_seat not in (0, 1):
        raise ProbeError(f"scorer_seat must be 0 or 1, got {scorer_seat}")
    return _draft(
        seed,
        {
            scorer_seat: _scorer_chooser(scorer, seed),
            1 - scorer_seat: _policy_chooser(policy, seed, device=device),
        },
    )


def _argmax_head(
    policy: Any,
    *,
    gen: ScenarioGenerator,
    scenario: Scenario,
    encoder: ObsEncoder,
    tracker: Any,
    setup_step: int,
    head: str,
    device: torch.device,
) -> int:
    obs_np, mask_np = obs_and_mask_for_scenario(
        gen=gen, scenario=scenario, encoder=encoder, tracker=tracker, setup_step=setup_step
    )
    obs: dict[str, torch.Tensor] = {}
    for key, value in obs_np.items():
        tensor = torch.as_tensor(np.asarray(value), device=device).unsqueeze(0)
        obs[key] = tensor.reshape(-1) if tensor.dtype == torch.int64 else tensor.float()
    obs["opponent_kind"] = torch.full((1,), PROBE_OPP_KIND, dtype=torch.int64, device=device)
    obs["opponent_policy_id"] = torch.full(
        (1,), UNKNOWN_POLICY_ID, dtype=torch.int64, device=device
    )
    mask = {
        key: torch.as_tensor(np.asarray(value, dtype=bool), device=device).unsqueeze(0)
        for key, value in mask_np.items()
    }
    action = torch.zeros((1, 6), dtype=torch.int64, device=device)
    with torch.no_grad():
        out = policy.evaluate_actions(obs, action, mask)
    return int(out[f"log_dist/{head}"].argmax(dim=-1).item())


@contextmanager
def _settlement_placed(scenario: Scenario, gen: ScenarioGenerator, vertex: int) -> Iterator[None]:
    """Place the probe settlement, then RESTORE the snapshot taken before it.

    The road head must be read against the board as it will look after the
    settlement lands, but ``ScenarioGenerator.apply`` re-places that settlement
    for real immediately afterwards, so leaving it would double-count it (VP,
    settlementsLeft, ports).

    SNAPSHOT-AND-RESTORE, not a hand-written inverse. This is the same six
    attributes ``Scenario.compute_legal_road_edges`` snapshots, by the same
    mechanism, for the same reason. An inverse has to re-derive what
    ``player.build_settlement`` did, and it re-derives it WRONG on ports:
    ``build_settlement`` appends a port only when the seat does not already hold
    that type (``engine/player.py``), while popping ``portList[-1]`` whenever it
    matches would delete a port the seat legitimately owns from its first
    settlement. Restoring a snapshot cannot get that wrong, and it cannot drift
    the day ``build_settlement`` touches a seventh attribute.

    ``build_settlement(is_free=True)`` emits no broadcast event, so there is no
    event-bus rollback to do.
    """
    acting = scenario._acting_player
    v_px = scenario._idx_to_vertex_pixel[vertex]
    vertex_obj = gen._board.boardGraph[v_px]

    prev_owner = getattr(vertex_obj, "owner", None)
    prev_building_type = getattr(vertex_obj, "building_type", None)
    prev_settlements = list(acting.buildGraph["SETTLEMENTS"])
    prev_settlements_left = acting.settlementsLeft
    prev_vp = acting.victoryPoints
    prev_port_list = list(acting.portList)

    acting.build_settlement(v_px, gen._board, is_free=True)
    try:
        yield
    finally:
        vertex_obj.owner = prev_owner
        vertex_obj.building_type = prev_building_type
        acting.buildGraph["SETTLEMENTS"] = prev_settlements
        acting.settlementsLeft = prev_settlements_left
        acting.victoryPoints = prev_vp
        acting.portList = prev_port_list


def _play_from_opening(
    opening: LabeledOpening,
    *,
    policy: Any,
    opponent: Any,
    agent_seat: int,
    max_turns: int,
    device: torch.device,
) -> ArmOutcome:
    state = replace(
        bridge_state_for_opening(opening, agent_seat=agent_seat, opp_kind=PROBE_OPP_KIND),
        opponent_type="snapshot",
    )
    env: CatanEnv = rebuild_env(state)
    env.max_turns = max_turns
    env.set_snapshot_opponent(opponent)
    try:
        obs = env._get_obs()
        masks = env.get_action_masks()
        terminated = truncated = False
        steps = 0
        cap = max_turns * 50
        while not terminated and not truncated:
            action = (
                policy.sample(
                    obs_to_torch(obs, device, add_batch=True),
                    masks_to_torch(masks, device, add_batch=True),
                )["action"][0]
                .cpu()
                .numpy()
                .astype(np.int64)
            )
            obs, _r, terminated, truncated, _i = env.step(action)
            masks = env.get_action_masks()
            steps += 1
            if steps > cap:
                truncated = True
                break
        agent = env.agent_player
        opp = env.opponent_player
        assert agent is not None and opp is not None
        agent_vp = int(getattr(agent, "victoryPoints", 0))
        opp_vp = int(getattr(opp, "victoryPoints", 0))
        return ArmOutcome(
            seed=opening.game_seed,
            agent_seat=agent_seat,
            won=(not truncated) and agent_vp >= 15 and agent_vp > opp_vp,
            truncated=truncated,
        )
    finally:
        env.close()


@dataclass(frozen=True)
class ProbeResult:
    scorer_arm: tuple[ArmOutcome, ...]
    policy_arm: tuple[ArmOutcome, ...]
    report: dict[str, Any]


def run_probe(
    *,
    scorer: SetupScorer,
    policy: Any,
    opponent: Any,
    seeds: Sequence[int],
    device: str = "cpu",
    max_turns: int = 400,
    alpha: float = 0.05,
    rng_seed: int = 0,
) -> ProbeResult:
    """Play both arms over ``seeds`` x both seats and report ΔWR with a CI.

    ``policy`` occupies the agent seat and ``opponent`` (a frozen snapshot of the
    SAME checkpoint, for the spec's "u500 plays both sides") the other. Pairing
    is by ``(seed, agent_seat)``.

    **The contrast is within the game, not between two symmetric worlds.** In
    the treatment arm ONLY the agent seat's four picks are forced to the
    scorer's; the other seat drafts with the policy and responds to them. In the
    control arm both seats draft with the policy. Summing over both agent seats
    then measures "does the scorer's opening make THIS seat win more", which is
    what D5 asks. Forcing both seats from one source — as an earlier draft did —
    makes each arm's win rate exactly 0.5 by symmetry and ΔWR identically zero,
    a mechanical artifact the spec's "ΔWR ~ 0 is AMBIGUOUS" clause would then
    have recorded as a finding.
    """
    torch_device = torch.device(device)
    np_state = np.random.get_state()
    py_state = random.getstate()
    torch_state = _snapshot_torch_rng()
    try:
        scorer_arm: list[ArmOutcome] = []
        policy_arm: list[ArmOutcome] = []
        for seed in seeds:
            p_open = policy_opening(policy, int(seed), device=device)
            for agent_seat in (0, 1):
                s_open = mixed_opening(
                    scorer, policy, int(seed), scorer_seat=agent_seat, device=device
                )
                torch.manual_seed(rng_seed + int(seed))
                np.random.seed((rng_seed + int(seed)) % (2**32))
                random.seed(rng_seed + int(seed))
                scorer_arm.append(
                    _play_from_opening(
                        s_open,
                        policy=policy,
                        opponent=opponent,
                        agent_seat=agent_seat,
                        max_turns=max_turns,
                        device=torch_device,
                    )
                )
                torch.manual_seed(rng_seed + int(seed))
                np.random.seed((rng_seed + int(seed)) % (2**32))
                random.seed(rng_seed + int(seed))
                policy_arm.append(
                    _play_from_opening(
                        p_open,
                        policy=policy,
                        opponent=opponent,
                        agent_seat=agent_seat,
                        max_turns=max_turns,
                        device=torch_device,
                    )
                )
        diff = paired_binary_difference(
            [g.won for g in scorer_arm], [g.won for g in policy_arm], alpha=alpha
        )
        report = {
            "n_pairs": diff.n,
            "wr_scorer_opening": diff.rate_a,
            "wr_policy_opening": diff.rate_b,
            "delta_wr": diff.delta,
            "ci_lower": diff.ci_lower,
            "ci_upper": diff.ci_upper,
            "alpha": alpha,
            # Pre-registered, written by the driver so the reading cannot be
            # chosen after seeing the number.
            "reading": (
                "positive"
                if diff.ci_lower > 0.0
                else ("negative" if diff.ci_upper < 0.0 else "ambiguous")
            ),
            "ambiguity_note": (
                "delta_wr ~ 0 does NOT show the opening is worthless: this midgame "
                "may be unable to exploit a better opening. Recorded as ambiguous."
            ),
            "n_truncated": sum(g.truncated for g in scorer_arm + policy_arm),
        }
        return ProbeResult(
            scorer_arm=tuple(scorer_arm), policy_arm=tuple(policy_arm), report=report
        )
    finally:
        np.random.set_state(np_state)
        random.setstate(py_state)
        _restore_torch_rng(torch_state)
