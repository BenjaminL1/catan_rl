"""D5 forced-opening probe driver (acceptance criterion 6).

Fixture scale: two seeds, a stub policy, tiny ``max_turns``. What is pinned is
the EXPERIMENT SHAPE — same board in both arms, openings the only difference,
the global RNG restored, and a refusal (not a fallback) when an arm's opening
is illegal.
"""

from __future__ import annotations

import random

import numpy as np
import pytest
import torch
from torch import nn

from catan_rl.labeling.scenario_gen import Pick, ScenarioGenerator
from catan_rl.setup_phase.fit import fit_scorer
from catan_rl.setup_phase.scorer import SetupScorer
from catan_rl.setup_phase.wr_probe import (
    ProbeError,
    _scorer_chooser,
    mixed_opening,
    policy_opening,
    run_probe,
    scorer_opening,
)

# Which sub-head each action type actually reads, and from which mask. A stub
# that samples a legal TYPE but leaves the sub-heads at 0 is not a stand-in for
# a policy: it emits e.g. ``BuildCity`` on corner 0 whenever the type is legal
# at all, and the engine raises deep inside ``updateBoardGraph_city``. Which
# board states it happens to survive then depends on the openings under test —
# i.e. the fixture breaks when the thing being measured changes, which is the
# opposite of what a fixture is for.
_TYPE_HEADS: dict[int, tuple[tuple[int, str], ...]] = {
    0: ((1, "corner_settlement"),),
    1: ((1, "corner_city"),),
    2: ((2, "edge"),),
    4: ((3, "tile"),),
    6: ((3, "tile"),),
    7: ((4, "resource1_yop"),),
    8: ((4, "resource1_default"),),
    10: ((4, "resource1_trade"),),
    11: ((4, "resource1_discard"),),
}


def _legal_action(masks, rng: np.random.Generator, i: int) -> np.ndarray:
    """One mask-respecting action row: a legal type plus legal sub-heads."""
    action = np.zeros(6, dtype=np.int64)

    def legal_of(key: str) -> np.ndarray:
        return np.flatnonzero(np.asarray(masks[key][i].cpu().numpy(), dtype=bool))

    types = legal_of("type")
    if types.size == 0:
        action[0] = 3  # EndTurn
        return action
    action[0] = int(rng.choice(types))
    for head, key in _TYPE_HEADS.get(int(action[0]), ()):
        options = legal_of(key)
        if options.size == 0:  # pragma: no cover - the type mask forbids this
            return np.array([3, 0, 0, 0, 0, 0], dtype=np.int64)
        action[head] = int(rng.choice(options))
    if int(action[0]) == 7:  # Year of Plenty needs a second resource too
        same = legal_of("resource2_yop_same")
        others = [j for j in legal_of("resource2_yop") if j != action[4]]
        action[5] = int(action[4]) if action[4] in same else int(others[0])
    elif int(action[0]) == 10:  # BankTrade receives something it did not give
        others = [j for j in legal_of("resource2_trade") if j != action[4]]
        action[5] = int(others[0]) if others else int(legal_of("resource2_trade")[0])
    return action


class _StubPolicy(nn.Module):
    """Random legal action; head distributions are index-descending."""

    def __init__(self) -> None:
        super().__init__()
        self.dummy = nn.Parameter(torch.zeros(1))
        self._rng = np.random.default_rng(0)

    def sample(self, obs, masks):
        batch = next(iter(obs.values())).shape[0]
        action = np.stack([_legal_action(masks, self._rng, i) for i in range(batch)])
        device = next(iter(obs.values())).device
        return {"action": torch.as_tensor(action, device=device)}

    def evaluate_actions(self, obs, action, masks):
        out = {}
        for head, size in (("corner", 54), ("edge", 72)):
            mask = masks["corner_settlement" if head == "corner" else "edge"]
            scores = torch.arange(size, dtype=torch.float32).flip(0).unsqueeze(0)
            out[f"log_dist/{head}"] = scores.masked_fill(~mask.bool(), -float("inf"))
        return out


class _StubOpponent:
    def __init__(self) -> None:
        self._rng = np.random.default_rng(1)

    @property
    def device(self) -> torch.device:
        return torch.device("cpu")

    def reset_rng(self, seed: int | None = None) -> None:
        self._rng = np.random.default_rng(0 if seed is None else seed)

    def sample(self, obs, masks) -> torch.Tensor:
        batch = next(iter(obs.values())).shape[0]
        return torch.as_tensor(np.stack([_legal_action(masks, self._rng, i) for i in range(batch)]))


@pytest.fixture(scope="module")
def scorer() -> SetupScorer:
    import tempfile
    from pathlib import Path

    from catan_rl.labeling.session import LabelingSession
    from catan_rl.labeling.store import load_scenarios

    with tempfile.TemporaryDirectory() as tmp:
        session = LabelingSession(data_dir=Path(tmp), labeler_id="f", session_seed=5)
        session.start()
        for _ in range(4):
            s = session.current_scenario()
            assert s is not None
            corners = np.flatnonzero(s.legal_settlement_corners)
            v = int(corners[0])
            e = int(np.flatnonzero(s.compute_legal_road_edges(v))[0])
            session.submit(settlement_vertex=v, road_edge=e)
        session.quit()
        rows = load_scenarios(Path(tmp) / "scenarios.jsonl")
    return fit_scorer(rows, version="v1", seed=0, iters=30).scorer


class TestOpeningDerivation:
    def test_scorer_opening_is_a_legal_four_pick_draft(self, scorer: SetupScorer) -> None:
        opening = scorer_opening(scorer, 42)
        assert len(opening.picks) == 4
        assert [p.player for p in opening.picks] == [0, 1, 1, 0]
        # It replays cleanly through the engine, which is the legality proof.
        gen = ScenarioGenerator(seed=42)
        for pick in opening.picks:
            gen.apply(pick.settlement_vertex, pick.road_edge)

    def test_policy_opening_is_a_legal_four_pick_draft(self) -> None:
        opening = policy_opening(_StubPolicy(), 42)
        assert len(opening.picks) == 4
        gen = ScenarioGenerator(seed=42)
        for pick in opening.picks:
            gen.apply(pick.settlement_vertex, pick.road_edge)

    def test_both_arms_are_derived_on_the_same_board(self, scorer: SetupScorer) -> None:
        a = scorer_opening(scorer, 42)
        b = policy_opening(_StubPolicy(), 42)
        assert a.game_seed == b.game_seed == 42
        assert a.picks != b.picks  # ...and they really do differ

    def test_an_illegal_arm_opening_raises_rather_than_falling_back(self) -> None:
        """A fallback here would put both arms on the SAME opening and report
        ΔWR = 0 as if it were evidence, so the driver must refuse."""

        class _MaskBlindPolicy(_StubPolicy):
            def evaluate_actions(self, obs, action, masks):
                # Ignores the legality mask entirely -> top-1 is vertex 53,
                # which is not legal at pick 1 on seed 42.
                return {
                    "log_dist/corner": torch.arange(54, dtype=torch.float32).unsqueeze(0),
                    "log_dist/edge": torch.arange(72, dtype=torch.float32).unsqueeze(0),
                }

        with pytest.raises(ProbeError, match="illegal"):
            policy_opening(_MaskBlindPolicy(), 42)


class TestProbeHygiene:
    def test_an_opponent_that_wraps_the_agent_module_is_refused(self, scorer: SetupScorer) -> None:
        """The "frozen snapshot" claim is enforced, not documented."""
        policy = _StubPolicy()

        class _Wrapper:
            def __init__(self, inner: _StubPolicy) -> None:
                self.policy = inner

        with pytest.raises(ProbeError, match="frozen snapshot"):
            run_probe(scorer=scorer, policy=policy, opponent=_Wrapper(policy), seeds=[42])
        with pytest.raises(ProbeError, match="frozen snapshot"):
            run_probe(scorer=scorer, policy=policy, opponent=policy, seeds=[42])

    def test_the_driver_puts_the_policy_in_eval_mode_itself(self, scorer: SetupScorer) -> None:
        policy = _StubPolicy()
        policy.train()
        assert policy.training
        run_probe(
            scorer=scorer,
            policy=policy,
            opponent=_StubOpponent(),
            seeds=[42],
            max_turns=2,
        )
        assert not policy.training


class TestProbeRound:
    def test_paired_report_and_rng_restoration(self, scorer: SetupScorer) -> None:
        policy = _StubPolicy()
        policy.eval()
        np.random.seed(1234)
        random.seed(1234)
        torch.manual_seed(1234)
        np_before = np.random.get_state()[1][:5].copy()
        py_before = random.random()
        random.seed(1234)

        result = run_probe(
            scorer=scorer,
            policy=policy,
            opponent=_StubOpponent(),
            seeds=[42, 43],
            max_turns=3,
        )
        # 2 seeds x 2 seats.
        assert result.report["n_pairs"] == 4
        assert len(result.scorer_arm) == len(result.policy_arm) == 4
        assert [(g.seed, g.agent_seat) for g in result.scorer_arm] == [
            (g.seed, g.agent_seat) for g in result.policy_arm
        ]
        assert result.report["reading"] in ("positive", "negative", "ambiguous")
        assert "ambiguity_note" in result.report

        # The global streams are exactly where they were.
        assert np.array_equal(np.random.get_state()[1][:5], np_before)
        assert random.random() == py_before

    def test_arms_differ_only_in_the_opening(self, scorer: SetupScorer) -> None:
        """Same seeds, same seats, same policy — only the forced picks change."""
        s_open = mixed_opening(scorer, _StubPolicy(), 42, scorer_seat=0)
        p_open = policy_opening(_StubPolicy(), 42)
        assert {p.player for p in s_open.picks} == {p.player for p in p_open.picks}
        assert s_open.game_seed == p_open.game_seed
        assert isinstance(s_open.picks[0], Pick)

    def test_the_treatment_forces_only_the_named_seat(self, scorer: SetupScorer) -> None:
        """The counterfactual is held WITHIN the game: one seat's rule changes,
        the other keeps drafting with the policy and responds to it."""
        policy = _StubPolicy()
        for seat in (0, 1):
            opening = mixed_opening(scorer, policy, 42, scorer_seat=seat)
            gen = ScenarioGenerator(seed=42)
            chooser = _scorer_chooser(scorer, 42)
            for pick in opening.picks:
                scenario = gen.current()
                assert scenario is not None
                want = chooser(gen, scenario)
                got = (pick.settlement_vertex, pick.road_edge)
                if pick.player == seat:
                    assert got == want, f"seat {seat} was not forced to the scorer"
                else:
                    assert got != want, f"seat {1 - seat} was forced too"
                gen.apply(pick.settlement_vertex, pick.road_edge)

    def test_the_contrast_is_not_degenerate_by_symmetry(self, scorer: SetupScorer) -> None:
        """The superseded design forced BOTH seats from one source. Summed over
        ``agent_seat`` that makes each arm's win rate 0.5 by symmetry and ΔWR
        identically zero however good the openings are — a null the spec would
        then have recorded as 'AMBIGUOUS'. The treatment arm must therefore be
        a DIFFERENT draft for each agent seat, and neither arm may be the
        both-seats draft.
        """
        policy = _StubPolicy()
        seat0 = mixed_opening(scorer, policy, 42, scorer_seat=0)
        seat1 = mixed_opening(scorer, policy, 42, scorer_seat=1)
        control = policy_opening(policy, 42)
        both = scorer_opening(scorer, 42)
        assert seat0.picks != seat1.picks
        assert seat0.picks != control.picks
        assert seat1.picks != control.picks
        assert seat0.picks != both.picks
        assert seat1.picks != both.picks

    def test_a_bad_scorer_seat_is_refused(self, scorer: SetupScorer) -> None:
        with pytest.raises(ProbeError, match="scorer_seat"):
            mixed_opening(scorer, _StubPolicy(), 42, scorer_seat=2)


class TestSettlementPlacedRestoresExactly:
    """The probe's temporary settlement must round-trip the engine state.

    ``ScenarioGenerator.apply`` re-places the settlement for real right
    afterwards, so a revert that leaves the object subtly wrong is invisible in
    the probe's own output and shows up only as a silently corrupted opening.
    """

    @staticmethod
    def _first_ported_vertex(scenario, gen):
        for v in np.flatnonzero(scenario.legal_settlement_corners):
            v_px = scenario._idx_to_vertex_pixel[int(v)]
            port = getattr(gen._board.boardGraph[v_px], "port", False)
            if port:
                return int(v), port
        return None, None

    def test_a_port_the_seat_already_owns_survives_the_revert(self) -> None:
        """The hand-written inverse popped it; snapshot/restore cannot.

        ``player.build_settlement`` appends a port only when the seat does NOT
        already hold that type, so an inverse that pops ``portList[-1]``
        whenever it matches DELETES a port the seat legitimately owns from an
        earlier settlement.
        """
        from catan_rl.setup_phase.wr_probe import _settlement_placed

        gen = ScenarioGenerator(seed=4242)
        scenario = gen.current()
        assert scenario is not None
        vertex, port = self._first_ported_vertex(scenario, gen)
        assert vertex is not None, "seed 4242 has a legal ported vertex at pick 1"

        acting = scenario._acting_player
        # Stand in for "this seat already took this port type with its first
        # settlement" — the state the inverse got wrong.
        acting.portList = [port]
        with _settlement_placed(scenario, gen, vertex):
            pass
        assert acting.portList == [port]

    def test_every_touched_attribute_round_trips(self) -> None:
        from catan_rl.setup_phase.wr_probe import _settlement_placed

        gen = ScenarioGenerator(seed=7)
        scenario = gen.current()
        assert scenario is not None
        vertex = int(np.flatnonzero(scenario.legal_settlement_corners)[0])
        acting = scenario._acting_player
        v_px = scenario._idx_to_vertex_pixel[vertex]
        vertex_obj = gen._board.boardGraph[v_px]

        before = (
            getattr(vertex_obj, "owner", None),
            getattr(vertex_obj, "building_type", None),
            list(acting.buildGraph["SETTLEMENTS"]),
            acting.settlementsLeft,
            acting.victoryPoints,
            list(acting.portList),
        )
        with _settlement_placed(scenario, gen, vertex):
            # Inside the block the settlement IS placed — that is the point.
            assert acting.victoryPoints == before[4] + 1
        after = (
            getattr(vertex_obj, "owner", None),
            getattr(vertex_obj, "building_type", None),
            list(acting.buildGraph["SETTLEMENTS"]),
            acting.settlementsLeft,
            acting.victoryPoints,
            list(acting.portList),
        )
        assert after == before

    def test_the_state_is_restored_even_when_the_body_raises(self) -> None:
        from catan_rl.setup_phase.wr_probe import _settlement_placed

        gen = ScenarioGenerator(seed=7)
        scenario = gen.current()
        assert scenario is not None
        vertex = int(np.flatnonzero(scenario.legal_settlement_corners)[0])
        acting = scenario._acting_player
        vp = acting.victoryPoints
        with pytest.raises(RuntimeError), _settlement_placed(scenario, gen, vertex):
            raise RuntimeError("policy blew up mid-read")
        assert acting.victoryPoints == vp
