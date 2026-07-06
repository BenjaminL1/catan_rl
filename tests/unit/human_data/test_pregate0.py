"""Tests for the PRE-GATE-0 + M0 runner (``scripts/pregate0.py``, step6 §2.2).

Two layers:

* **Pure-stat / aggregation** tests on synthetic JSONL records (fast, no ML):
  the mass table is the v8-vs-anchor subset only; M0 uses the v8-vs-v8 subset with
  ``draft_position``-only strata; the permutation test respects strata.
* **End-to-end** tests that actually play a handful of short natural games with a
  fresh (cheap) ``CatanPolicy`` — well-formed JSONL + report, resumability
  (kill + rerun appends without duplicates), and determinism given a seed.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]


def _load_pregate0() -> Any:
    spec = importlib.util.spec_from_file_location(
        "pregate0_mod", REPO_ROOT / "scripts" / "pregate0.py"
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


pregate0 = _load_pregate0()


# ---------------------------------------------------------------------------
# Synthetic records
# ---------------------------------------------------------------------------
def _seat(
    *,
    archetype: str = "BALANCED_LOW",
    ore_wheat: float = 0.2,
    port: bool = False,
    won: int = 0,
    draft: int = 0,
    settlements: list[int] | None = None,
) -> dict[str, Any]:
    return {
        "settlements": settlements or [0, 5],
        "archetype": archetype,
        "pip_share": [0.2, 0.2, 0.2, 0.2, 0.2],
        "ore_wheat_share": ore_wheat,
        "port_adjacent": port,
        "total_pips": 20,
        "draft_position": draft,
        "won": won,
    }


def _rec(
    *,
    matchup: str,
    gi: int,
    archetype: str = "BALANCED_LOW",
    v_hat_to_move: float = 0.5,
    outcome_to_move: int = 0,
    seat1: dict[str, Any] | None = None,
    setup_entropies: list[float] | None = None,
) -> dict[str, Any]:
    seat0 = _seat(archetype=archetype, won=outcome_to_move, draft=0)
    seat1 = seat1 or _seat(archetype="BALANCED_LOW", won=1 - outcome_to_move, draft=1)
    return {
        "game_id": f"{matchup}:{gi}",
        "matchup": matchup,
        "game_index": gi,
        "seed": 0,
        "to_move_seat": 0,
        "v_hat_to_move": v_hat_to_move,
        "outcome_to_move": outcome_to_move,
        "seats": {"0": seat0, "1": seat1},
        "draft_positions": {"0": 0, "1": 1},
        "final_vp": {"0": 15 if outcome_to_move else 8, "1": 8 if outcome_to_move else 15},
        "n_turns": 30,
        "truncated": False,
        "setup_entropy_mean": (float(np.mean(setup_entropies)) if setup_entropies else None),
        "setup_entropies": setup_entropies or [1.0, 1.2],
    }


# ---------------------------------------------------------------------------
# Mass table = v8-vs-anchor subset only
# ---------------------------------------------------------------------------
def test_mass_table_uses_v8_anchor_subset_only() -> None:
    records = [
        _rec(matchup="v8_anchor", gi=0, archetype="ORE_ENGINE"),
        _rec(matchup="v8_anchor", gi=1, archetype="ORE_ENGINE"),
        _rec(matchup="v8_anchor", gi=2, archetype="WOOD_BRICK"),
        # v8_v8 rows carry a bucket ABSENT from the anchor subset — they must not
        # leak into the mass table.
        _rec(matchup="v8_v8", gi=0, archetype="PORT_LED"),
        _rec(matchup="v8_v8", gi=1, archetype="PORT_LED"),
    ]
    mass = pregate0.build_mass_table(records, v8_ckpt="v8.pt", anchor_ckpt="anchor.pt")

    assert mass["source"] == "v8-vs-anchor"
    assert mass["n_games"] == 3  # only the v8_anchor rows
    assert mass["counts"]["ORE_ENGINE"] == 2
    assert mass["counts"]["WOOD_BRICK"] == 1
    assert mass["counts"]["PORT_LED"] == 0  # v8_v8 rows excluded
    assert mass["mass"]["ORE_ENGINE"] == pytest.approx(2 / 3)
    assert mass["max_bucket"] == "ORE_ENGINE"


def test_mass_table_discloses_draft_position_scope() -> None:
    # Finding: v8 is always seat 0 (draft position 0), so the mass table measures
    # v8's first-drafter openings only. The frozen artifact must disclose this.
    records = [
        _rec(matchup="v8_anchor", gi=0, archetype="ORE_ENGINE"),
        _rec(matchup="v8_anchor", gi=1, archetype="WOOD_BRICK"),
    ]
    mass = pregate0.build_mass_table(records, v8_ckpt="v8.pt", anchor_ckpt="a.pt")
    assert mass["measured_seat"] == 0
    assert mass["measured_draft_position"] == 0
    assert "first-drafter" in mass["measurement_note"]
    assert "draft position 0" in mass["measurement_note"]


def test_mass_table_collapse_verdict() -> None:
    # 4/5 = 0.8 ≥ 0.70 threshold ⇒ COLLAPSED.
    records = [_rec(matchup="v8_anchor", gi=i, archetype="ORE_ENGINE") for i in range(4)] + [
        _rec(matchup="v8_anchor", gi=4, archetype="WOOD_BRICK")
    ]
    mass = pregate0.build_mass_table(records, v8_ckpt="v8.pt", anchor_ckpt="a.pt")
    assert mass["collapse_verdict"] == "COLLAPSED"
    assert mass["max_mass"] == pytest.approx(0.8)

    # 3/5 = 0.6 < 0.70 ⇒ NO_COLLAPSE.
    records2 = [_rec(matchup="v8_anchor", gi=i, archetype="ORE_ENGINE") for i in range(3)] + [
        _rec(matchup="v8_anchor", gi=3 + i, archetype="WOOD_BRICK") for i in range(2)
    ]
    mass2 = pregate0.build_mass_table(records2, v8_ckpt="v8.pt", anchor_ckpt="a.pt")
    assert mass2["collapse_verdict"] == "NO_COLLAPSE"


# ---------------------------------------------------------------------------
# M0 = v8-vs-v8 subset only; strata = draft_position only
# ---------------------------------------------------------------------------
def test_m0_uses_v8v8_subset_and_draft_position_strata() -> None:
    rng = np.random.default_rng(1)
    records: list[dict[str, Any]] = []
    for i in range(30):
        # Focal seat alternates by game_index parity inside compute_m0; give both
        # seats distinct ore_wheat + win flags so the partial statistic is defined.
        v = float(rng.uniform(0.3, 0.7))
        ow0 = float(rng.uniform(0, 0.6))
        ow1 = float(rng.uniform(0, 0.6))
        s0_won = int(rng.uniform(0, 1) < 0.5)
        seat1 = _seat(ore_wheat=ow1, won=1 - s0_won, draft=1)
        records.append(
            _rec(
                matchup="v8_v8",
                gi=i,
                v_hat_to_move=v,
                outcome_to_move=s0_won,
                seat1=seat1,
            )
        )
        records[-1]["seats"]["0"]["ore_wheat_share"] = ow0
    # Anchor rows must not enter M0.
    for i in range(10):
        records.append(_rec(matchup="v8_anchor", gi=i, v_hat_to_move=0.9, outcome_to_move=1))

    m0 = pregate0.compute_m0(records, perms=500, seed=7)
    assert m0["subset"] == "v8-vs-v8"
    assert m0["n_games"] == 30  # anchor rows excluded
    assert m0["strata"] == ["draft_position"]
    assert m0["predicted_sign"] == "positive"
    assert 0.0 <= m0["auc"] <= 1.0 or np.isnan(m0["auc"])
    assert not np.isnan(m0["partial_spearman"])


def test_permutation_respects_strata() -> None:
    # Two strata; within each stratum r is constant, so within-stratum permutation
    # never changes r ⇒ every resample equals the observed statistic ⇒ p == 1.0.
    r = np.array([1.0, 1.0, 1.0, 2.0, 2.0, 2.0])
    x = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    z = np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0])
    strata = [0, 0, 0, 1, 1, 1]
    observed, p = pregate0.partial_spearman_perm_p(r, x, z, strata, perms=200, seed=3)
    assert not np.isnan(observed)
    assert p == pytest.approx(1.0)


def test_auc_ranks_scores() -> None:
    scores = np.array([0.1, 0.4, 0.35, 0.8])
    labels = np.array([0, 0, 1, 1])
    # winners (0.35, 0.8) vs losers (0.1, 0.4): 3 of 4 pairs correctly ordered.
    assert pregate0.auc(scores, labels) == pytest.approx(0.75)
    assert np.isnan(pregate0.auc(np.array([1.0, 2.0]), np.array([1, 1])))


def test_setup_head_entropy_mean() -> None:
    records = [
        _rec(matchup="v8_v8", gi=0, setup_entropies=[1.0, 2.0]),
        _rec(matchup="v8_anchor", gi=0, setup_entropies=[3.0, 4.0]),
    ]
    out = pregate0.compute_setup_head_entropy(records)
    assert out["n_decisions"] == 4
    assert out["setup_head_entropy"] == pytest.approx(2.5)


def test_game_seed_deterministic_and_unsalted() -> None:
    a = pregate0.game_seed(0, "v8_v8", 3)
    b = pregate0.game_seed(0, "v8_v8", 3)
    c = pregate0.game_seed(0, "v8_anchor", 3)
    assert a == b
    assert a != c
    assert 0 <= a < 2**31 - 1


# ---------------------------------------------------------------------------
# End-to-end (fresh cheap policy)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def cheap_policy() -> Any:
    from catan_rl.policy.board_geometry import build_geometry
    from catan_rl.policy.network import CatanPolicy

    policy = CatanPolicy()
    policy.set_board_geometry(build_geometry().as_dict_of_tensors())
    policy.eval()
    for p in policy.parameters():
        p.requires_grad_(False)
    return policy


def _opponents(policy: Any) -> tuple[Any, Any, Any]:
    import torch

    from catan_rl.selfplay.snapshot_opponent import FrozenSnapshotOpponent

    device = torch.device("cpu")
    v8_opp = FrozenSnapshotOpponent(policy, device=device, seed=0)
    anchor_opp = FrozenSnapshotOpponent(policy, device=device, seed=1)
    return v8_opp, anchor_opp, device


_REQUIRED_KEYS = {
    "game_id",
    "matchup",
    "game_index",
    "to_move_seat",
    "v_hat_to_move",
    "outcome_to_move",
    "seats",
    "draft_positions",
    "setup_entropies",
}
_ARCHETYPES = {"ORE_ENGINE", "WOOD_BRICK", "PORT_LED", "BALANCED_HIGH", "BALANCED_LOW"}


def _run(tmp: Path, policy: Any, *, n: int, seed: int = 0) -> dict[str, Any]:
    v8_opp, anchor_opp, device = _opponents(policy)
    return pregate0.run_pregate0(
        jsonl_path=tmp / "games.jsonl",
        mass_path=tmp / "mass.json",
        report_path=tmp / "report.md",
        n_per_matchup=n,
        seed=seed,
        perms=50,
        max_turns=15,
        agent_policy=policy,
        v8_opponent=v8_opp,
        anchor_opponent=anchor_opp,
        device=device,
        v8_ckpt="v8.pt",
        anchor_ckpt="anchor.pt",
        anchor_is_v8_copy=True,
        freeze={"stub": "hash"},
    )


def test_end_to_end_smoke(tmp_path: Path, cheap_policy: Any) -> None:
    result = _run(tmp_path, cheap_policy, n=1)
    jsonl = tmp_path / "games.jsonl"
    mass = tmp_path / "mass.json"
    report = tmp_path / "report.md"
    assert jsonl.exists() and mass.exists() and report.exists()

    records = pregate0.load_records(jsonl)
    assert len(records) == 2  # 1 v8_v8 + 1 v8_anchor
    assert {r["matchup"] for r in records} == {"v8_v8", "v8_anchor"}
    for r in records:
        assert set(r) >= _REQUIRED_KEYS
        assert r["outcome_to_move"] in (0, 1)
        assert set(r["seats"]) == {"0", "1"}
        for s in r["seats"].values():
            assert len(s["settlements"]) == 2  # both openings, 2 settlements each
            assert s["archetype"] in _ARCHETYPES
            assert s["won"] in (0, 1)
        assert 0.0 <= r["v_hat_to_move"] <= 1.0  # squashed win-probability

    mass_obj = json.loads(mass.read_text())
    assert mass_obj["source"] == "v8-vs-anchor"
    assert mass_obj["n_games"] == 1  # only the v8_anchor game
    report_txt = report.read_text()
    assert "COLLAPSE VERDICT" in report_txt
    assert "M0 —" in report_txt
    assert "draft_position" in report_txt
    assert result["n_records"] == 2


def test_resumability_no_duplicates(tmp_path: Path, cheap_policy: Any) -> None:
    _run(tmp_path, cheap_policy, n=1)  # writes v8_v8:0, v8_anchor:0
    records1 = pregate0.load_records(tmp_path / "games.jsonl")
    ids1 = [r["game_id"] for r in records1]
    assert sorted(ids1) == ["v8_anchor:0", "v8_v8:0"]

    # Re-run with a larger n: existing game_ids skipped, new ones appended.
    _run(tmp_path, cheap_policy, n=2)
    records2 = pregate0.load_records(tmp_path / "games.jsonl")
    ids2 = [r["game_id"] for r in records2]
    assert len(ids2) == len(set(ids2)) == 4  # no duplicates
    assert set(ids2) == {"v8_v8:0", "v8_v8:1", "v8_anchor:0", "v8_anchor:1"}
    # The two originally-logged records are byte-preserved (not replayed).
    kept = {r["game_id"]: r for r in records2}
    for r in records1:
        assert kept[r["game_id"]] == r


def test_deterministic_given_seed(tmp_path: Path, cheap_policy: Any) -> None:
    dir_a = tmp_path / "a"
    dir_b = tmp_path / "b"
    dir_a.mkdir()
    dir_b.mkdir()
    _run(dir_a, cheap_policy, n=1, seed=42)
    _run(dir_b, cheap_policy, n=1, seed=42)
    lines_a = (dir_a / "games.jsonl").read_text()
    lines_b = (dir_b / "games.jsonl").read_text()
    assert lines_a == lines_b


def test_opponent_rng_reset_is_structural(cheap_policy: Any) -> None:
    """Finding: a game's record must be independent of the shared opponent's
    incoming RNG state (i.e. of how many games preceded it in the same process).

    The frozen opponent object is reused across games, so its ``_call_count``
    advances game-to-game. ``play_game`` reseeds the opponent per game, so the
    same game index yields a byte-identical record whether the opponent is fresh
    or has been advanced by unrelated sampling first — determinism is structural,
    not incidental to run order / resume boundaries.
    """
    import torch

    from catan_rl.human_data.topology import load_topology
    from catan_rl.selfplay.snapshot_opponent import FrozenSnapshotOpponent

    device = torch.device("cpu")
    topology = load_topology()
    gseed = pregate0.game_seed(0, "v8_v8", 1)

    def _play(opp: Any) -> dict[str, Any]:
        return pregate0.play_game(
            agent_policy=cheap_policy,
            opponent=opp,
            device=device,
            seed=gseed,
            max_turns=15,
            topology=topology,
            matchup="v8_v8",
            game_index=1,
        )

    rec_fresh = _play(FrozenSnapshotOpponent(cheap_policy, device=device, seed=0))

    # A DIRTY opponent: advance its RNG stream with unrelated sampling so its
    # ``_call_count`` / ``_seed`` differ from a fresh opponent's before the game.
    dirty = FrozenSnapshotOpponent(cheap_policy, device=device, seed=999)
    dirty.reset_rng(seed=12345)
    for _ in range(7):
        dirty._call_count += 1  # simulate games having been played on it
    rec_dirty = _play(dirty)

    assert rec_fresh == rec_dirty
