"""D4 — label store → BC shard converter.

The load-bearing pin is ``test_row_obs_equals_the_encoder_on_the_replayed_state``:
the JSONL stores a seed and a list of picks, never an obs, so the whole
durable-corpus claim rests on the converter regenerating exactly what the
current ``obs_encoder`` would produce for that position.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from catan_rl.labeling import to_shard
from catan_rl.labeling.session import LabelingSession
from catan_rl.policy.obs_schema import MASK_KEYS, ActionType


@pytest.fixture(scope="module")
def labels(tmp_path_factory) -> Path:  # type: ignore[no-untyped-def]
    """Label 4 real scenarios (one full snake draft) with legal picks."""
    data_dir = tmp_path_factory.mktemp("labels")
    session = LabelingSession(data_dir=data_dir, labeler_id="test", session_seed=4242)
    session.start()
    rng = np.random.default_rng(0)
    for _ in range(4):
        scenario = session.current_scenario()
        assert scenario is not None
        legal = np.flatnonzero(scenario.legal_settlement_corners)
        vertex = int(rng.choice(legal))
        edges = np.flatnonzero(scenario.compute_legal_road_edges(vertex))
        session.submit(vertex, int(rng.choice(edges)))
    session.quit()
    return session.scenarios_path


@pytest.fixture(scope="module")
def shard_dir(tmp_path_factory, labels: Path) -> Path:  # type: ignore[no-untyped-def]
    out = tmp_path_factory.mktemp("shard")
    # EXPLICIT 0.0: ``convert`` now withholds 20% by default, and this fixture
    # is the "whole corpus" one every downstream row-count assertion counts on.
    to_shard.convert(labels, out, held_out_frac=0.0)
    return out


def test_shard_loads_through_bcdataset_with_no_stale_corpus_error(shard_dir: Path) -> None:
    from catan_rl.bc.loader import BcDataset

    ds = BcDataset(shard_dir, aug_prob=0.0)
    # 4 scenarios x 2 rows x 3 opponent kinds.
    assert len(ds) == 4 * 2 * len(to_shard.DEFAULT_OPPONENT_KINDS)
    item = ds[0]
    assert set(item["mask"]) == set(MASK_KEYS)
    assert item["action"].shape == (6,)
    assert float(item["z_disc"]) == 0.0


def test_manifest_is_stamped_from_the_schema_not_from_literals(shard_dir: Path) -> None:
    import json

    from catan_rl.policy.obs_schema import FORCED_RULE_VERSION, RULESET_VERSION

    manifest = json.loads((shard_dir / "manifest.json").read_text())
    assert manifest["forced_rule_version"] == FORCED_RULE_VERSION == 2
    assert manifest["ruleset_version"] == RULESET_VERSION == 3
    assert manifest["ruleset"] == "R0"
    assert manifest["label_source_counts"] == {"tool": 4}


def test_two_rows_per_scenario_settlement_then_road(labels: Path) -> None:
    from catan_rl.labeling.store import load_scenarios

    label = load_scenarios(labels)[0]
    rows = to_shard.rows_for_label(label, game_id=0)
    assert len(rows) == 2
    settle, road = rows
    assert int(settle.action[0]) == ActionType.BUILD_SETTLEMENT
    assert int(settle.action[1]) == label["settlement_vertex"]
    assert int(road.action[0]) == ActionType.BUILD_ROAD
    assert int(road.action[2]) == label["road_edge"]
    assert [r.step_idx for r in rows] == [0, 1]


def test_the_road_row_sees_the_board_after_its_own_settlement(labels: Path) -> None:
    """§E was silent here. A road row built on the PRE-settlement obs would
    teach the policy to pick a road from a position it never occupies — and its
    own labeled edge would not even be legal there."""
    from catan_rl.labeling.store import load_scenarios

    label = load_scenarios(labels)[0]
    settle, road = to_shard.rows_for_label(label, game_id=0)
    vertex = int(label["settlement_vertex"])
    edge = int(label["road_edge"])
    # The settlement row's mask offers the labeled vertex; the road row's mask
    # offers the labeled edge, which only becomes legal once it is placed.
    assert bool(settle.mask["corner_settlement"][vertex])
    assert bool(road.mask["edge"][edge])
    assert (
        not bool(settle.mask["edge"][edge]) or settle.mask["edge"].sum() != road.mask["edge"].sum()
    )
    # And the obs differs: the settlement is now on the board.
    assert not np.array_equal(settle.obs["vertex_features"], road.obs["vertex_features"])


def test_row_obs_equals_the_encoder_on_the_replayed_state(labels: Path) -> None:
    """LOAD-BEARING. Rebuild the position independently and compare, key by key."""
    from catan_rl.engine.player import player as EnginePlayer  # noqa: F401
    from catan_rl.env.hand_tracker import BroadcastHandTracker
    from catan_rl.labeling.scenario_gen import Pick, ScenarioGenerator
    from catan_rl.labeling.store import load_scenarios
    from catan_rl.policy.obs_encoder import EnvObsState, ObsEncoder

    label = load_scenarios(labels)[2]
    settle_row, _road_row = to_shard.rows_for_label(label, game_id=0)

    gen = ScenarioGenerator(seed=int(label["game_seed"]))
    scenario = gen.current()
    assert scenario is not None
    tracker = BroadcastHandTracker([p.name for p in gen._players])
    tracker.subscribe(scenario.game.broadcast)
    for pick in label["prior_picks"]:
        p = Pick.from_dict(pick)
        gen.apply(p.settlement_vertex, p.road_edge)
    scenario = gen.current()
    assert scenario is not None
    acting = scenario._acting_player
    opponent = next(p for p in gen._players if p is not acting)
    expected = ObsEncoder(gen._board).build_obs(
        scenario.game,
        acting,
        opponent,
        EnvObsState(initial_placement_phase=True, setup_step=scenario._setup_step_road - 1),
        hand_tracker=tracker,
    )
    assert set(settle_row.obs) == set(expected)
    assert len(expected) == 12, "12 obs keys, not the plan's 10"
    for key, value in expected.items():
        np.testing.assert_array_equal(
            np.asarray(settle_row.obs[key]), np.asarray(value), err_msg=key
        )


def test_rows_are_duplicated_across_every_opponent_kind(shard_dir: Path) -> None:
    """The policy conditions on an opponent-id embedding, so a single-kind stamp
    would leave the learned openings unexpressed under the other kinds."""
    import numpy as np

    with np.load(shard_dir / "shard_00000.npz") as z:
        kinds = z["obs/opponent_kind"]
        gids = z["game_id"]
        actions = z["action"]
        policy_ids = z["obs/opponent_policy_id"]
    assert sorted(set(int(k) for k in kinds)) == sorted(to_shard.DEFAULT_OPPONENT_KINDS)
    assert set(int(p) for p in policy_ids) == {to_shard.UNKNOWN_POLICY_ID}
    # Every (game_id, action) appears once per opponent kind, and duplicates
    # share the game_id so train_val_split cannot straddle a scenario.
    for gid in set(int(g) for g in gids):
        sel = gids == gid
        assert int(sel.sum()) == 2 * len(to_shard.DEFAULT_OPPONENT_KINDS)
        assert len({tuple(a) for a in actions[sel]}) == 2


def test_conversion_is_deterministic(labels: Path, tmp_path: Path) -> None:
    """Byte-identical ZIPs are not claimed — ``np.savez_compressed`` stamps each
    member with the local time. What must be deterministic is every ARRAY."""
    import numpy as np

    a, b = tmp_path / "a", tmp_path / "b"
    to_shard.convert(labels, a, held_out_frac=0.0)
    to_shard.convert(labels, b, held_out_frac=0.0)
    with np.load(a / "shard_00000.npz") as za, np.load(b / "shard_00000.npz") as zb:
        assert sorted(za.files) == sorted(zb.files)
        for key in za.files:
            np.testing.assert_array_equal(za[key], zb[key], err_msg=key)


def test_an_illegal_label_is_refused_not_dropped(labels: Path, tmp_path: Path) -> None:
    from catan_rl.labeling.store import load_scenarios

    # Draft position 2: the first pick has already blocked its neighbourhood, so
    # an illegal vertex exists to point the label at (at position 1 all 54 are
    # legal).
    label = dict(load_scenarios(labels)[1])
    from catan_rl.labeling.scenario_gen import Pick, ScenarioGenerator

    gen = ScenarioGenerator(seed=int(label["game_seed"]))
    for pick in label["prior_picks"]:
        pk = Pick.from_dict(pick)
        gen.apply(pk.settlement_vertex, pk.road_edge)
    scenario = gen.current()
    assert scenario is not None
    illegal = int(np.flatnonzero(~scenario.legal_settlement_corners)[0])
    label["settlement_vertex"] = illegal
    with pytest.raises(to_shard.LabelConversionError, match="labeled settlement vertex"):
        to_shard.rows_for_label(label, game_id=0)


def test_empty_store_is_refused(tmp_path: Path) -> None:
    empty = tmp_path / "scenarios.jsonl"
    empty.write_text("")
    with pytest.raises(to_shard.LabelConversionError, match="no label rows"):
        to_shard.convert(empty, tmp_path / "out", held_out_frac=0.0)


# ---------------------------------------------------------------------------
# D7 gate-1 honesty: the held-out set is decided HERE, not at gate time
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def multi_seed_labels(tmp_path_factory) -> Path:  # type: ignore[no-untyped-def]
    """Two full drafts, i.e. two ``game_seed`` s — the minimum a split needs."""
    from catan_rl.labeling.store import load_scenarios

    data_dir = tmp_path_factory.mktemp("labels_multi")
    session = LabelingSession(data_dir=data_dir, labeler_id="test", session_seed=909)
    session.start()
    rng = np.random.default_rng(1)
    for _ in range(8):
        scenario = session.current_scenario()
        assert scenario is not None
        legal = np.flatnonzero(scenario.legal_settlement_corners)
        vertex = int(rng.choice(legal))
        edges = np.flatnonzero(scenario.compute_legal_road_edges(vertex))
        session.submit(vertex, int(rng.choice(edges)))
    session.quit()
    assert len({int(r["game_seed"]) for r in load_scenarios(session.scenarios_path)}) >= 2
    return session.scenarios_path


def test_a_split_needs_more_than_one_seed_and_says_so(labels: Path, tmp_path: Path) -> None:
    """A single-draft corpus cannot be split — refuse rather than write an empty
    shard, or (worse) a full one whose gate would then be in-sample."""
    with pytest.raises(to_shard.LabelConversionError, match="no training rows"):
        to_shard.convert(labels, tmp_path / "nope", held_out_frac=0.5)


def test_the_default_withholds_a_held_out_set(multi_seed_labels: Path, tmp_path: Path) -> None:
    """The CLI has always defaulted to 0.2; the library defaulted to 0.0, so any
    caller that did not repeat the flag built an ungateable shard. The default is
    the one the gates require."""
    manifest = to_shard.convert(multi_seed_labels, tmp_path / "default")
    assert manifest["held_out_frac"] == 0.2
    assert manifest["held_out_game_seeds"]


def test_a_single_seed_corpus_now_fails_closed_under_the_default(
    labels: Path, tmp_path: Path
) -> None:
    """Owned consequence of the new default: ``held_out_split`` always withholds
    at least one seed, so a one-draft corpus has no training half left. Failing
    closed is right — a shard whose gate is in-sample is worse than no shard."""
    with pytest.raises(to_shard.LabelConversionError, match="no training rows"):
        to_shard.convert(labels, tmp_path / "single")


def test_held_out_seeds_are_excluded_from_the_shard_and_named_in_the_manifest(
    multi_seed_labels: Path, tmp_path: Path
) -> None:
    """The load-bearing anti-memorisation pin.

    If the split were re-derived at eval time over the same JSONL, the gate
    would score the candidate on rows it was fine-tuned on. The converter is the
    only place that can make "held out" true of the CHECKPOINT.
    """
    import json

    from catan_rl.labeling.store import held_out_split, load_scenarios

    out = tmp_path / "split"
    manifest = to_shard.convert(multi_seed_labels, out, held_out_frac=0.5, split_seed=0)
    held_seeds = set(manifest["held_out_game_seeds"])
    assert held_seeds

    all_rows = load_scenarios(multi_seed_labels)
    train, held = held_out_split(all_rows, frac=0.5, seed=0)
    assert {int(r["game_seed"]) for r in held} == held_seeds
    assert manifest["n_held_out_scenarios"] == len(held)
    # Every converted scenario comes from the TRAIN half.
    assert manifest["n_scenarios"] == len({r["scenario_id"] for r in train})
    assert manifest["n_pairs"] == (
        manifest["n_scenarios"] * 2 * len(to_shard.DEFAULT_OPPONENT_KINDS)
    )
    on_disk = json.loads((out / "manifest.json").read_text())
    assert on_disk["held_out_game_seeds"] == manifest["held_out_game_seeds"]
    assert on_disk["split_seed"] == 0


def test_no_split_requested_means_an_empty_held_out_set_not_a_missing_key(
    shard_dir: Path,
) -> None:
    """The gate CLI refuses on the EMPTY set; it must be able to see it."""
    import json

    manifest = json.loads((shard_dir / "manifest.json").read_text())
    assert manifest["held_out_game_seeds"] == []
    assert manifest["held_out_frac"] == 0.0
