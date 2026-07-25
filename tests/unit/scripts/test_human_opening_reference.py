"""Correctness pins for ``scripts/human_opening_reference.py`` (spec AC2-AC7).

No checkpoint and no policy forward pass is needed: ``opening_sweep.setup_logp``
is replaced by a deterministic mask-respecting stub, so these tests exercise the
whole replay/pairing/aggregation machinery in seconds.

Pins, one per acceptance criterion:

* **AC2 injection fidelity** — a board injected from a corpus row reproduces
  that row's 19 hex resources/numbers exactly and parks the robber on the stated
  desert; and the DEFAULT (non-injected) reset path is byte-identical with and
  without the new ``options`` key.
* **AC3 index mapping** — corpus vertex/edge ids resolve, through the env's own
  ``_idx_to_vertex`` / ``_idx_to_edge``, to the same engine entities the policy
  heads are indexed by; and every recorded placement in a corpus sample is legal
  when replayed (the real evidence that the ids mean what we think).
* **AC4 determinism** — same inputs ⇒ bit-identical output, plus a negative
  control proving the port seed is actually consumed.
* **AC5 port-seed determinism** — ``port_seed`` is stable per (row, k) and
  distinct across k.
* **AC6 pairing** — the policy is queried at exactly the board state
  ThePhantom faced (his own prior placements, nothing of the policy's).
* **AC7** — the one row lacking ``placement_order_established`` is skipped, and a
  non-snake ``draft_order`` is rejected rather than silently replayed as a snake.
* **AC8 read-only pins** — ``--out-dir`` / ``--raw-dir`` outside
  ``data/human/**`` or ``runs/analysis/**`` are refused BEFORE anything is
  created (``runs/train/**`` above all), and a non-CPU device is refused (D7).
* **AC9 / D6** — the report carries the refutation-scope header, the port
  limitation, the constant-``opponent_strength`` warning, the unadjusted-CI
  multiplicity accounting, and the recomputed corpus census; and the census
  itself reproduces the spec-D6 numbers on the shipped corpus.
* **D3 metric numerics** — the three primary metrics are pinned on hand-computed
  graphs (comparison direction and before/after ordering included), plus an
  engine-wired directional pin, so an inverted comparison cannot stay green.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

_REPO = Path(__file__).resolve().parents[3]
_SCRIPT = _REPO / "scripts" / "human_opening_reference.py"
_spec = importlib.util.spec_from_file_location("human_opening_reference_module", _SCRIPT)
assert _spec is not None and _spec.loader is not None
hor = importlib.util.module_from_spec(_spec)
sys.modules["human_opening_reference_module"] = hor
_spec.loader.exec_module(hor)

from catan_rl.env.catan_env import CatanEnv  # noqa: E402

_CORPUS = _REPO / "data" / "human" / "corpus" / "provisional_openings.jsonl"

#: The single corpus row whose placement ORDER could not be established. Pinned
#: by id so AC7 fails loudly if the corpus is regenerated with a different one.
_UNORDERED_ROW = ("7Ft5ZxIm64Q", 1)


@pytest.fixture(scope="module")
def rows() -> list[dict[str, Any]]:
    return hor.load_rows(_CORPUS)


class _StubPolicy:
    """Placeholder — the stubbed ``setup_logp`` never dereferences it."""


def _stub_setup_logp(
    policy: Any,
    obs_t: dict[str, torch.Tensor],
    masks_t: dict[str, torch.Tensor],
    settlement: bool,
) -> torch.Tensor:
    """Deterministic stand-in: uniform over the legal set, -inf elsewhere.

    ``argmax`` therefore returns the LOWEST legal index, which is always legal —
    so the replay path (and its "argmax must be legal" guard) is exercised
    without a checkpoint.
    """
    key = "corner_settlement" if settlement else "edge"
    mask = masks_t[key].to(torch.bool)
    return torch.where(mask, torch.zeros(mask.shape), torch.full(mask.shape, -float("inf")))


@pytest.fixture
def stub_logp(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(hor.opening_sweep, "setup_logp", _stub_setup_logp)


# ---------------------------------------------------------------------------
# AC2 — injection fidelity + default-path byte identity
# ---------------------------------------------------------------------------


def test_injection_reproduces_the_corpus_layout(rows: list[dict[str, Any]]) -> None:
    row = rows[0]
    layout = hor.layout_from_row(row)
    env = CatanEnv(opponent_type="heuristic")
    env.reset(seed=1234, options={"agent_seat": 0, "board_layout": layout})
    assert env.game is not None
    board = env.game.board
    for entry in row["board"]["hexes"]:
        h = int(entry["hex_id"])
        tile = board.hexTileDict[h]
        assert tile.resource_type == entry["resource"]
        assert tile.number_token == entry["number"]
    robber = row["provenance"]["board_desert_hex"]
    assert [h for h in range(19) if board.hexTileDict[h].has_robber] == [robber]
    # The dependent ``resourcesList`` must stay consistent with the tiles.
    assert [r.type for r in board.resourcesList] == layout["resources"]


def test_layout_from_row_rejects_a_desert_mismatch(rows: list[dict[str, Any]]) -> None:
    row = json.loads(json.dumps(rows[0]))
    desert = row["provenance"]["board_desert_hex"]
    row["provenance"]["board_desert_hex"] = (desert + 1) % 19
    with pytest.raises(hor.CorpusReplayError):
        hor.layout_from_row(row)


def test_default_reset_path_is_byte_identical(rows: list[dict[str, Any]]) -> None:
    """The new seam must be a no-op when ``board_layout`` is absent."""

    def build(options: dict[str, Any]) -> tuple[dict[str, np.ndarray], dict[str, list[int]]]:
        env = CatanEnv(opponent_type="heuristic")
        obs, _ = env.reset(seed=99, options=options)
        assert env.game is not None
        return obs, env.game.board.get_port_assignment()

    obs_a, ports_a = build({"agent_seat": 0})
    obs_b, ports_b = build({"agent_seat": 0, "board_layout": None})
    assert ports_a == ports_b
    assert obs_a.keys() == obs_b.keys()
    for key in obs_a:
        np.testing.assert_array_equal(obs_a[key], obs_b[key])


def test_injection_lands_before_the_obs_encoder_port_cache(
    rows: list[dict[str, Any]],
) -> None:
    """A welded port assignment must reach the obs, not just the engine.

    ``ObsEncoder`` caches ``_vertex_port_static`` at construction, so an
    injection applied after it would be invisible to the policy — the exact
    failure this ordering guards against.
    """
    layout = hor.layout_from_row(rows[0])
    sampler = hor.PortSampler()
    layout["port_assignment"] = sampler.assignment(hor.port_seed("pin", 0, 0))
    env = CatanEnv(opponent_type="heuristic")
    env.reset(seed=7, options={"agent_seat": 0, "board_layout": layout})
    assert env.game is not None
    assert env.game.board.get_port_assignment() == layout["port_assignment"]
    cached = env._obs_encoder._vertex_port_static
    for port_type, vertices in layout["port_assignment"].items():
        for v_idx in vertices:
            assert cached[v_idx, 0] == 0.0, f"vertex {v_idx} ({port_type}) cached as NO-PORT"


# ---------------------------------------------------------------------------
# AC3 — index mapping
# ---------------------------------------------------------------------------


def test_env_index_maps_agree_with_engine_vertex_indices() -> None:
    env = CatanEnv(opponent_type="heuristic")
    env.reset(seed=3, options={"agent_seat": 0})
    assert env.game is not None
    board = env.game.board
    assert len(env._idx_to_vertex) == 54
    for idx, px in env._idx_to_vertex.items():
        assert board.boardGraph[px].vertex_index == idx
    assert len(env._idx_to_edge) == 72
    # AC3 for edges is INDEX AGREEMENT, not merely "some valid edge": the corpus
    # parser resolved edge ids against the committed topology export, so edge
    # index i must be the same pair the export calls i. A re-export or a
    # reordering would silently relocate every recorded road otherwise.
    exported = json.loads(
        (_REPO / "src" / "catan_rl" / "human_data" / "topology.json").read_text()
    )["edgeVertices"]
    assert len(exported) == 72
    for idx, (v1, v2) in env._idx_to_edge.items():
        assert v2 in board.boardGraph[v1].neighbors
        pair = sorted((board.boardGraph[v1].vertex_index, board.boardGraph[v2].vertex_index))
        assert pair == sorted(exported[idx]), f"edge {idx} disagrees with topology.json"


def test_recorded_placements_are_legal_on_the_injected_board(
    rows: list[dict[str, Any]], stub_logp: None
) -> None:
    """The load-bearing AC3 evidence: corpus ids replay without a single
    illegal placement, for BOTH seats and both players."""
    sample = [r for r in rows if r["provenance"]["placement_order_established"]][:12]
    seats = set()
    for row in sample:
        decisions = hor.replay_row(
            row, _StubPolicy(), torch.device("cpu"), hor.PortSampler(), n_port_samples=1
        )
        assert len(decisions) == 4
        assert [d.kind for d in decisions] == ["settlement", "road", "settlement", "road"]
        seats.add(decisions[0].agent_seat)
    assert seats == {0, 1}, f"sample did not cover both draft seats: {seats}"


def test_an_illegal_recorded_placement_is_raised_not_silently_applied(
    rows: list[dict[str, Any]], stub_logp: None
) -> None:
    row = json.loads(
        json.dumps(next(r for r in rows if r["provenance"]["placement_order_established"]))
    )
    agent = row["players"]["agent"]
    # Both of the agent's settlements on the same vertex — illegal by the
    # distance rule at the second placement.
    row["openings"][agent]["settlements"][1] = row["openings"][agent]["settlements"][0]
    with pytest.raises(hor.CorpusReplayError):
        hor.replay_row(row, _StubPolicy(), torch.device("cpu"), hor.PortSampler(), n_port_samples=1)


def test_scripted_opponent_refuses_a_fifth_action() -> None:
    opp = hor.ScriptedSetupOpponent([("settlement", 0)], torch.device("cpu"))
    masks = {"corner_settlement": torch.ones((1, 54), dtype=torch.bool)}
    opp.sample({}, masks)
    with pytest.raises(hor.CorpusReplayError, match="exhausted"):
        opp.sample({}, masks)


# ---------------------------------------------------------------------------
# AC4 — determinism (+ negative control)
# ---------------------------------------------------------------------------


def test_replay_is_bit_identical_across_runs(rows: list[dict[str, Any]], stub_logp: None) -> None:
    sample = [r for r in rows if r["provenance"]["placement_order_established"]][:3]

    def once() -> str:
        result = hor.run(sample, _StubPolicy(), torch.device("cpu"), n_port_samples=2)
        payload = {
            "decisions": [asdict(d) for d in result.decisions],
            "summary": hor.summary_json(result, ckpt=Path("stub.pt"), n_port_samples=2),
        }
        return json.dumps(payload, sort_keys=True)

    assert once() == once()


def test_port_assignment_actually_consumes_its_seed() -> None:
    """Negative control: different seeds ⇒ different assignments (so a
    determinism pass cannot be trivially satisfied by ignoring the seed)."""
    sampler = hor.PortSampler()
    assignments = [json.dumps(sampler.assignment(s), sort_keys=True) for s in range(8)]
    assert len(set(assignments)) > 1
    # ...and the same seed still reproduces exactly.
    assert json.dumps(sampler.assignment(3), sort_keys=True) == assignments[3]


def test_real_policy_path_is_deterministic(rows: list[dict[str, Any]]) -> None:
    """Determinism of the SHIPPED instrument, not of the stub.

    The stub ignores ``obs_t`` entirely, so the stubbed determinism pin proves
    only that mask enumeration is stable. This one runs the real
    ``opening_sweep.setup_logp`` forward pass (on an untrained but
    fixed-seed-initialised policy — no checkpoint needed) and pins that the same
    inputs reproduce the same policy CHOICES and metrics bit-for-bit, and that
    the obs path is actually consumed (different port assignments reach the
    network as different obs).
    """
    from catan_rl.policy.network import CatanPolicy

    torch.manual_seed(0)
    policy = CatanPolicy()
    policy.set_board_geometry(hor.opening_sweep.build_geometry().as_dict_of_tensors())
    policy.eval()
    for param in policy.parameters():
        param.requires_grad_(False)

    row = next(r for r in rows if r["provenance"]["placement_order_established"])

    def once() -> str:
        decisions = hor.replay_row(
            row, policy, torch.device("cpu"), hor.PortSampler(), n_port_samples=2
        )
        return json.dumps([asdict(d) for d in decisions], sort_keys=True)

    first = once()
    assert first == once()

    # The obs really carries the port assignment into the network: the same
    # decision under two port assignments yields different logits.
    layout = hor.layout_from_row(row)
    logps = []
    for k in (0, 1):
        layout_k = dict(layout)
        layout_k["port_assignment"] = hor.PortSampler().assignment(
            hor.port_seed(str(row["video_id"]), int(row["game_index"]), k)
        )
        env = CatanEnv(opponent_type="heuristic")
        env.reset(seed=11, options={"agent_seat": 0, "board_layout": layout_k})
        obs_t = hor.obs_to_torch(env._get_obs(), torch.device("cpu"), add_batch=True)
        masks_t = hor.masks_to_torch(env.get_action_masks(), torch.device("cpu"), add_batch=True)
        with torch.no_grad():
            logps.append(hor.opening_sweep.setup_logp(policy, obs_t, masks_t, True))
    assert not torch.equal(logps[0], logps[1]), "ports never reached the policy obs"


def test_cluster_bootstrap_is_seeded_and_reproducible() -> None:
    values = [float(i % 5) for i in range(40)]
    clusters = [f"v{i // 4}" for i in range(40)]
    assert hor.cluster_bootstrap_ci(values, clusters) == hor.cluster_bootstrap_ci(values, clusters)


# ---------------------------------------------------------------------------
# AC5 — port-seed determinism
# ---------------------------------------------------------------------------


def test_port_seed_is_stable_and_distinct_per_k() -> None:
    seeds = [hor.port_seed("abc", 2, k) for k in range(hor.N_PORT_SAMPLES)]
    assert len(set(seeds)) == hor.N_PORT_SAMPLES
    assert seeds == [hor.port_seed("abc", 2, k) for k in range(hor.N_PORT_SAMPLES)]
    assert hor.port_seed("abc", 2, 0) != hor.port_seed("abc", 3, 0)
    assert hor.port_seed("abc", 2, 0) != hor.port_seed("abd", 2, 0)
    # Pinned literal: the seed derivation is a published contract, so a silent
    # change to it (which would silently change every reported number) fails here.
    assert hor.port_seed("abc", 2, 0) == 11295258372497174435


# ---------------------------------------------------------------------------
# AC6 — pairing
# ---------------------------------------------------------------------------


def test_policy_is_queried_at_the_humans_own_board_state(
    rows: list[dict[str, Any]], stub_logp: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Hand-checked row: at every decision the agent seat holds exactly
    ThePhantom's own prior placements — the policy's picks never enter the
    trajectory, which is what makes the comparison paired (D1)."""
    row = next(r for r in rows if r["provenance"]["placement_order_established"])
    _seat, agent_script, opp_script = hor._human_decisions(row)
    human_settles = [i for kind, i in agent_script if kind == "settlement"]
    human_roads = [i for kind, i in agent_script if kind == "road"]

    opp_settles = [i for kind, i in opp_script if kind == "settlement"]
    opp_roads = [i for kind, i in opp_script if kind == "road"]

    observed: list[tuple[list[int], list[int], int]] = []
    opp_observed: list[tuple[list[int], list[int]]] = []
    original = hor._score_candidates

    def spy(env: Any, scorer: Any, **kwargs: Any) -> Any:
        agent = env.agent_player
        v_ids = [env._vertex_to_idx[v] for v in agent.buildGraph["SETTLEMENTS"]]
        e_ids = [env._edge_to_idx[env._edge_key(v1, v2)] for v1, v2 in agent.buildGraph["ROADS"]]
        opp = env.opponent_player
        opp_observed.append(
            (
                [env._vertex_to_idx[v] for v in opp.buildGraph["SETTLEMENTS"]],
                [env._edge_to_idx[env._edge_key(v1, v2)] for v1, v2 in opp.buildGraph["ROADS"]],
            )
        )
        observed.append((v_ids, e_ids, len(opp.buildGraph["SETTLEMENTS"])))
        return original(env, scorer, **kwargs)

    monkeypatch.setattr(hor, "_score_candidates", spy)
    hor.replay_row(row, _StubPolicy(), torch.device("cpu"), hor.PortSampler(), n_port_samples=1)

    assert len(observed) == 4
    # decision 0: nothing of mine placed yet.
    assert observed[0][0] == [] and observed[0][1] == []
    # decision 1 (first road): my first settlement, the HUMAN's, is down.
    assert observed[1][0] == human_settles[:1] and observed[1][1] == []
    # decision 2 (second settlement): the human's first settlement + road.
    assert observed[2][0] == human_settles[:1] and observed[2][1] == human_roads[:1]
    # decision 3 (second road): both human settlements + the first human road.
    assert observed[3][0] == human_settles and observed[3][1] == human_roads[:1]
    # The scripted opponent really placed (otherwise the "same board state"
    # claim would be vacuous).
    assert observed[3][2] == len(opp_settles) - (1 if _seat == 1 else 0)
    # ...and it placed on the CORPUS ids, not wherever a heuristic body would
    # have chosen. A count-only assertion would pass even if the routing
    # (opponent_type="snapshot" + duck-typed set_snapshot_opponent) silently
    # regressed to the env's own heuristic opponent, and then every number in
    # the report would describe a board a bot shaped. Ids are compared as a
    # PREFIX of the corpus script, which is what "placed so far" means.
    assert opp_observed, "the spy never fired"
    for seen_v, seen_e in opp_observed:
        assert seen_v == opp_settles[: len(seen_v)]
        assert seen_e == opp_roads[: len(seen_e)]
    # By the last agent decision the opponent must actually be on the board.
    assert opp_observed[-1][0] and opp_observed[-1][1]


def test_contested_metrics_are_none_when_the_opponent_has_not_placed(
    rows: list[dict[str, Any]], stub_logp: None
) -> None:
    """Undefined ⇒ ``None``, never zero-filled (which would fake a tie)."""
    row = next(
        r
        for r in rows
        if r["provenance"]["placement_order_established"]
        and r["draft_order"][0] == r["players"]["agent"]
    )
    decisions = hor.replay_row(
        row, _StubPolicy(), torch.device("cpu"), hor.PortSampler(), n_port_samples=1
    )
    first = decisions[0]
    for name in hor.CONTESTED_METRICS:
        assert first.human_metrics[name] is None
        assert first.policy_metrics[name] == [None]
    assert all(name not in first.gaps() for name in hor.CONTESTED_METRICS)
    # ...but they ARE defined at the second settlement.
    assert decisions[2].human_metrics["denial"] is not None


# ---------------------------------------------------------------------------
# AC7 — never guess placement order
# ---------------------------------------------------------------------------


def test_the_unordered_row_is_skipped_by_id(rows: list[dict[str, Any]], stub_logp: None) -> None:
    unordered = [
        (r["video_id"], r["game_index"])
        for r in rows
        if not r["provenance"]["placement_order_established"]
    ]
    assert unordered == [_UNORDERED_ROW]
    row = next(r for r in rows if (r["video_id"], r["game_index"]) == _UNORDERED_ROW)
    result = hor.run([row], _StubPolicy(), torch.device("cpu"), n_port_samples=1)
    assert result.decisions == []
    assert len(result.excluded) == 1
    entry = result.excluded[0]
    assert entry["video_id"] == _UNORDERED_ROW[0]
    assert entry["game_index"] == _UNORDERED_ROW[1]
    assert entry["reason"] == "placement_order_not_established"
    # The ledger carries the provenance flags the report reads back.
    assert "passed_crosscheck" in entry and "order_source" in entry


def test_a_non_snake_draft_order_is_rejected_not_guessed(
    rows: list[dict[str, Any]],
) -> None:
    """An alternating order would be silently replayed as a snake (D6)."""
    row = json.loads(
        json.dumps(next(r for r in rows if r["provenance"]["placement_order_established"]))
    )
    a, b = row["players"]["agent"], row["players"]["opponent"]
    row["draft_order"] = [a, b, a, b]
    with pytest.raises(hor.CorpusReplayError, match="snake"):
        hor._human_decisions(row)
    # ...and the real corpus is all snake, so this pin is not vacuous.
    for real in rows:
        order = real["draft_order"]
        assert order[0] == order[3] and order[1] == order[2] and order[0] != order[1]


# ---------------------------------------------------------------------------
# D3 metric units
# ---------------------------------------------------------------------------


class _LineScorer:
    """Minimal ``BoardScorer`` stand-in over a hand-drawn path graph.

    Vertices ``0..n-1`` in a line; vertex ``v`` is worth ``v + 1`` pips, so the
    pip weight identifies WHICH vertices were counted — a count alone cannot,
    which is exactly how an inverted ``dm < do`` can slip through. Implements
    only what ``settlement_extra_metrics`` calls.
    """

    def __init__(self, n: int = 7) -> None:
        self._n = n
        self.pip_sum = {v: v + 1 for v in range(n)}

    def _neighbors(self, v: int) -> list[int]:
        return [u for u in (v - 1, v + 1) if 0 <= u < self._n]

    def legal_sites(self, occupied: set[int]) -> set[int]:
        return {
            v
            for v in range(self._n)
            if v not in occupied and not any(u in occupied for u in self._neighbors(v))
        }

    def bfs(self, source: int, blocked: set[int], max_depth: int) -> dict[int, int]:
        dist = {source: 0}
        frontier = [source]
        depth = 0
        while frontier and depth < max_depth:
            depth += 1
            nxt = []
            for v in frontier:
                if v != source and v in blocked:
                    continue
                for u in self._neighbors(v):
                    if u not in dist:
                        dist[u] = depth
                        nxt.append(u)
            frontier = nxt
        return dist


def test_contested_race_and_denial_match_a_hand_computation() -> None:
    """Numeric pin on the three PRIMARY metrics (the whole deliverable).

    Path 0-1-2-3-4-5-6. I own vertex 0, the opponent owns vertex 6, my candidate
    is vertex 2.

    * occupancy before = {0, 6}; legal(before) = {2, 3, 4}
      (1 and 5 are adjacent to an owned vertex).
    * legal(after = {0, 2, 6}) = {4} (3 is now adjacent to my candidate).
    * ``contested_race``: only vertex 4 is in the legal-after set;
      d(mine={0,2} -> 4) = 2, d(opp={6} -> 4) = 2, so it is NOT reached strictly
      first ⇒ count 0, pips 0.
    * ``denial`` (literal spec D3, unrestricted): |legal_before| - |legal_after| =
      |{2,3,4}| - |{4}| = 2.
    * ``denial_reach``: the opponent reaches {2,3,4,5} within depth 4, so
      |legal_before ∩ reach| = |{2,3,4}| = 3 and |legal_after ∩ reach| = |{4}| = 1
      ⇒ 2 (it coincides with the literal value here — the whole legal set is
      inside his reach on this tiny path).
    """
    scorer = _LineScorer()
    out = hor.settlement_extra_metrics(
        scorer,
        2,
        base_occupied={0, 6},
        my_settlements=[0],
        opp_settlements=[6],
        my_blocked={6},
        opp_blocked={0},
    )
    assert out == {
        "contested_race_count": 0.0,
        "contested_race_pips": 0.0,
        "denial": 2.0,
        "denial_reach": 2.0,
    }


def test_contested_race_counts_only_sites_i_reach_strictly_first() -> None:
    """Direction pin: the comparison is ``dm < do``, NOT ``dm > do``.

    Path 0-1-...-10 (pips = index + 1). I own vertex 0, the opponent owns vertex
    10, my candidate is vertex 3.

    * legal(before = {0, 10}) = {2,3,4,5,6,7,8};
      legal(after = {0, 3, 10}) = {5,6,7,8}.
    * distances at horizon 4, d(mine={0,3} -> u) vs d(opp={10} -> u):
      5 -> 2 vs (out of horizon, skipped); 6 -> 3 vs 4 (MINE first);
      7 -> 4 vs 3 (his); 8 -> (out of horizon, skipped).
    * ⇒ ``contested_race_count`` = 1 and ``contested_race_pips`` = pip(6) = 7.
      An inverted comparison would count vertex 7 instead — same count, but
      pips = 8, which is why the pip value is the load-bearing assertion.
    * ``denial`` (literal, unrestricted): |{2,3,4,5,6,7,8}| - |{5,6,7,8}| = 3.
    * ``denial_reach``: reach(opp from 10, horizon 4) = {6,7,8,9,10};
      |legal_before ∩ reach| = |{6,7,8}| = 3 = |legal_after ∩ reach| ⇒ 0. This is
      the case that separates the two definitions: the candidate removes three
      sites, NONE of them within the opponent's reach.
    """
    scorer = _LineScorer(n=11)
    out = hor.settlement_extra_metrics(
        scorer,
        3,
        base_occupied={0, 10},
        my_settlements=[0],
        opp_settlements=[10],
        my_blocked={10},
        opp_blocked={0},
    )
    assert out == {
        "contested_race_count": 1.0,
        "contested_race_pips": 7.0,
        "denial": 3.0,
        "denial_reach": 0.0,
    }


def test_denial_is_before_minus_after_on_a_real_engine_board(
    rows: list[dict[str, Any]], stub_logp: None
) -> None:
    """Engine-wired pin: both denial variants on real ``BoardScorer`` state.

    * ``denial`` (literal spec D3) must equal ``|legal_before| - |legal_after|``
      recomputed independently from the scorer, and be non-negative (a swapped
      subtraction would make it <= 0).
    * ``denial_reach`` must be strictly larger for a candidate NEAR the opponent
      than for the one furthest from him — that ordering is the whole point of
      the reach restriction, and it is what the literal metric does NOT express.
    """
    row = next(r for r in rows if r["provenance"]["placement_order_established"])
    layout = hor.layout_from_row(row)
    env = CatanEnv(opponent_type="heuristic")
    env.reset(seed=5, options={"agent_seat": 0, "board_layout": layout})
    assert env.game is not None
    board = env.game.board
    scorer = hor.BoardScorer(board)
    verts = list(board.boardGraph)
    mine, theirs = verts[0], verts[30]
    board.updateBoardGraph_settlement(mine, env.agent_player)
    board.updateBoardGraph_settlement(theirs, env.opponent_player)

    base = scorer.occupied()
    my_blocked = scorer.opponent_blocked(env.agent_player)
    opp_blocked = scorer.opponent_blocked(env.opponent_player)
    d_opp = hor._multi_source_bfs(scorer, [theirs], opp_blocked, hor.REACH_DEPTH)
    legal = scorer.legal_sites(base)
    near = min(
        (v for v in legal if v in d_opp and d_opp[v] > 0),
        key=lambda v: (d_opp[v], scorer.centrality[v]),
    )
    far = max(legal, key=lambda v: (d_opp.get(v, 99), scorer.centrality[v]))

    def denial(cand: Any, name: str = "denial_reach") -> float:
        out = hor.settlement_extra_metrics(
            scorer,
            cand,
            base_occupied=base,
            my_settlements=[mine],
            opp_settlements=[theirs],
            my_blocked=my_blocked,
            opp_blocked=opp_blocked,
        )
        value = out[name]
        assert value is not None
        return value

    assert denial(near) > 0.0
    assert denial(near) > denial(far)
    assert denial(far) >= 0.0

    # The literal spec-D3 row: unrestricted shrinkage, recomputed independently.
    for cand in (near, far):
        expected = len(legal) - len(scorer.legal_sites(base | {cand}))
        assert denial(cand, "denial") == float(expected)
        assert expected >= 0


def test_cut_vulnerability_counts_articulation_points() -> None:
    # A path a-b-c: b disconnects it.
    assert hor.road_extra_metrics([("a", "b"), ("b", "c")])["cut_vulnerability"] == 1.0
    # A single edge has none.
    assert hor.road_extra_metrics([("a", "b")])["cut_vulnerability"] == 0.0
    # A triangle is 2-connected.
    assert hor.road_extra_metrics([("a", "b"), ("b", "c"), ("c", "a")])["cut_vulnerability"] == 0.0
    # Two disjoint edges: separate components, still no cut vertex.
    assert hor.road_extra_metrics([("a", "b"), ("c", "d")])["cut_vulnerability"] == 0.0


def test_a_setup_road_pair_CAN_have_an_articulation_point() -> None:
    """The all-zero column is EMPIRICAL, not structural (report + docstring pin).

    The distance rule forbids ADJACENT settlements, not settlements at graph
    distance 2, so two of mine may share a middle vertex — and two setup roads
    both pointing at that shared vertex form a 3-vertex path whose middle IS an
    articulation point. Corpus row ``b_NXNI-l0kI`` g6 is exactly that geometry
    (agent settlements 15 and 14, shared engine neighbour 13, recorded first road
    13-15), so the double-back is reachable in this instrument's own regime.
    """
    env = CatanEnv(opponent_type="heuristic")
    env.reset(seed=0)
    assert env.game is not None
    board = env.game.board
    by_index = {v.vertex_index: px for px, v in board.boardGraph.items()}
    s1, s2, mid = by_index[15], by_index[14], by_index[13]
    # 15 and 14 are NOT adjacent (so both are legal settlements) but both are
    # adjacent to 13 — the geometry the false "always disjoint" proof denied.
    assert s2 not in board.boardGraph[s1].neighbors
    assert mid in board.boardGraph[s1].neighbors and mid in board.boardGraph[s2].neighbors
    assert hor.road_extra_metrics([(s1, mid), (s2, mid)])["cut_vulnerability"] == 1.0


# ---------------------------------------------------------------------------
# AC8 / D7 — read-only pins
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "candidate",
    [
        "runs/train/selfplay_pointer_arch_v3/checkpoints",
        "runs/train",
        "runs",
        "docs",
        "data",
        "/tmp/anywhere",
    ],
)
def test_write_targets_outside_the_allowed_roots_are_refused(candidate: str) -> None:
    """AC8: the one hard constraint whose violation is unrecoverable."""
    path = Path(candidate) if candidate.startswith("/") else _REPO / candidate
    with pytest.raises(hor.HardConstraintError):
        hor.check_write_target(path)


@pytest.mark.parametrize(
    "candidate",
    [
        "data/human/opening_reference",
        "data/human",
        "runs/analysis/human_opening_reference",
        "runs/analysis",
        "data/human/nested/deeper",
    ],
)
def test_the_two_permitted_write_roots_are_accepted(candidate: str) -> None:
    assert hor.check_write_target(_REPO / candidate) == (_REPO / candidate).resolve()


def test_the_defaults_are_inside_the_permitted_roots() -> None:
    assert hor.check_write_target(hor.DEFAULT_OUT_DIR)
    assert hor.check_write_target(hor.DEFAULT_RAW_DIR)


def test_main_refuses_a_runs_train_out_dir_before_creating_anything(
    tmp_path: Path,
) -> None:
    """The guard must fire before ``mkdir`` / checkpoint load, not after."""
    target = _REPO / "runs" / "train" / "__hor_guard_probe__"
    with pytest.raises(hor.HardConstraintError):
        hor.main(["--out-dir", str(target), "--limit", "1"])
    assert not target.exists()


def test_a_non_cpu_device_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    """D7: an accelerator would contend with the live training run."""
    with pytest.raises(hor.HardConstraintError):
        hor.check_device(torch.device("mps"))
    assert hor.check_device(torch.device("cpu")).type == "cpu"
    with pytest.raises(hor.HardConstraintError):
        hor.main(["--device", "mps", "--limit", "1"])


# ---------------------------------------------------------------------------
# D6 — corpus census + AC9 report contents
# ---------------------------------------------------------------------------


def test_corpus_census_reproduces_the_spec_d6_numbers(rows: list[dict[str, Any]]) -> None:
    census = hor.corpus_census(rows)
    assert {k: census[k] for k in hor.EXPECTED_CENSUS} == hor.EXPECTED_CENSUS
    assert census["matches_expected"] is True
    # The provenance disclosures the report prints must be populated.
    assert census["order_source_counts"]
    assert census["synthetic_video_rows"] == ["vlm_spike g1"]


def test_report_carries_the_required_disclosures(
    rows: list[dict[str, Any]], stub_logp: None
) -> None:
    """AC9 + the multiplicity/missingness/census statements."""
    sample = [r for r in rows if r["provenance"]["placement_order_established"]][:6]
    result = hor.run(sample, _StubPolicy(), torch.device("cpu"), n_port_samples=2)
    text = hor.build_report(result, ckpt=Path("stub.pt"), n_port_samples=2)
    for needle in (
        "REFUTATION instrument",
        "Ports are marginalised, not known",
        "hardcoded constant",
        "UNADJUSTED",
        "Multiplicity accounting",
        "Corpus census",
        "Metric definitions (D3)",
        "Bottom line on the pre-registered question",
        "Why the PRIMARY `n` is smaller",
        # The primary row must be the LITERAL spec-D3 denial, with the restricted
        # variant offered alongside and flagged, never substituted for it.
        "OWNER DECISION PENDING",
        "`denial_reach` is the flagged sharpening",
        # `pair_*` rows score ThePhantom's #1 + the chooser's #2.
        "TEACHER-FORCED HYBRID",
        # Both denial rows miss the road-blocking channel.
        "DISTANCE-RULE channel",
    ):
        assert needle in text, f"report is missing {needle!r}"
    # The false claim this replaced must not come back.
    assert "cannot manufacture a finding by selection" not in text
    # ...nor may the FALSE cut_vulnerability "proof": the distance rule forbids
    # adjacency, not distance 2, so a self-cutting road pair is legal and the
    # all-zero column is empirical, not structural.
    assert "This is structural, not luck" not in text
    assert "provably CONSTANT AT ZERO" not in (hor.__doc__ or "")
    assert hor.PRIMARY_METRICS == ("contested_race_count", "contested_race_pips", "denial")
    assert "denial_reach" in hor.CONTESTED_METRICS
    assert "denial_reach" not in hor.PRIMARY_METRICS
    summary = hor.summary_json(result, ckpt=Path("stub.pt"), n_port_samples=2)
    assert summary["corpus_census"]["rows"] == len(sample)
    assert summary["settlement_decision_regimes"]


def test_summarise_counts_informative_missingness() -> None:
    """A metric dropped because the POLICY's pick was undefined is COUNTED."""

    def dec(human: float | None, policy: list[float | None]) -> hor.PairedDecision:
        return hor.PairedDecision(
            video_id="v",
            game_index=1,
            agent_seat=0,
            decision_index=2,
            kind="settlement",
            human_id=0,
            policy_ids=[0] * len(policy),
            n_candidates=1,
            opp_settlements=1,
            human_metrics={"m": human},
            policy_metrics={"m": policy},
        )

    decisions = [
        dec(1.0, [2.0, 2.0]),  # used
        dec(1.0, [None, None]),  # dropped: policy undefined at every k
        dec(1.0, [None, 3.0]),  # partial
        dec(None, [1.0, 1.0]),  # human undefined (structural)
    ]
    s = hor.summarise(decisions, "m", "settlement")
    assert s is not None
    assert (s.n_decisions, s.n_decisions_of_kind) == (2, 4)
    assert s.n_human_undefined == 1
    assert s.n_dropped_policy_undefined == 1
    assert s.n_partial_policy_undefined == 1
    assert s.has_informative_missingness
    warnings = hor._missingness_warnings([s])
    assert any("dropped because EVERY port sample" in line for line in warnings)


def test_a_silently_ignored_injection_is_caught_at_runtime(
    rows: list[dict[str, Any]],
) -> None:
    """An ``options`` key an older ``catan_rl`` build does not know is IGNORED.

    That would score every decision on a random board while the run looks
    healthy, so the layout is verified after every reset — not trusted.
    """
    layout = hor.layout_from_row(rows[0])
    env = CatanEnv(opponent_type="heuristic")
    env.reset(seed=42, options={"agent_seat": 0, "board_layout": layout})
    assert env.game is not None
    hor.assert_layout_applied(env.game.board, layout)  # the real, injected board

    # A board built WITHOUT the injection must be rejected.
    naive = CatanEnv(opponent_type="heuristic")
    naive.reset(seed=42, options={"agent_seat": 0})
    assert naive.game is not None
    with pytest.raises(hor.CorpusReplayError, match="did NOT take"):
        hor.assert_layout_applied(naive.game.board, layout)
