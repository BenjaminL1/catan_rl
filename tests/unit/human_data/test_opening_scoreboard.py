"""Unit tests for the paired ΔAUC opening scoreboard, mapped 1:1 to the spec
acceptance criteria (`.claude/veriloop/specs/opening-scoreboard.md`, AC2–AC9).

Scoring-heavy tests use tiny synthetic/stub policies; a few use real eligible
records loaded from the tracked corpus (cheap value checks, no checkpoint loads).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

import catan_rl.human_data.opening_scoreboard as sb
from catan_rl.human_data.engine_bridge import (
    BridgeState,
    engine_state_hash,
    rebuild_env,
    serialize_post_setup,
)
from catan_rl.human_data.record import GameRecord

_CORPUS = sb.DEFAULT_CORPUS_PATH


def _eligible(n: int) -> list[GameRecord]:
    if not _CORPUS.is_file():
        pytest.skip(f"corpus not present: {_CORPUS}")
    recs = sb.eligible_records(sb.load_records(_CORPUS))
    if len(recs) < n:
        pytest.skip(f"corpus has only {len(recs)} eligible records, need {n}")
    return recs[:n]


class _StubPolicy:
    """A deterministic stand-in value head: v̂ = scale · Σ(obs float tensors).

    Schema-agnostic (works on both new- and legacy-schema obs dicts); distinct
    ``scale`` lets the two legs return different values from the same state.
    """

    def __init__(self, scale: float) -> None:
        self._scale = scale

    def forward(self, obs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        total = 0.0
        for t in obs.values():
            if torch.is_floating_point(t):
                total += float(t.sum().item())
        return {"value": torch.tensor([[self._scale * total]], dtype=torch.float32)}


def _stub_scorer(new_scale: float = 1.0, legacy_scale: float = 0.5) -> sb.PairedScorer:
    scorer = sb.PairedScorer.__new__(sb.PairedScorer)
    scorer._device = "cpu"  # type: ignore[attr-defined]
    scorer._new_policy = _StubPolicy(new_scale)  # type: ignore[attr-defined]
    scorer._legacy_policy = _StubPolicy(legacy_scale)  # type: ignore[attr-defined]
    return scorer


# ---------------------------------------------------------------------------
# AC2 — paired-identical-state pin (both legs' obs from the SAME engine state)
# ---------------------------------------------------------------------------
def test_ac2_both_legs_share_one_engine_state(monkeypatch: pytest.MonkeyPatch) -> None:
    rec = _eligible(1)[0]
    adapter = sb.RecordAdapter()
    state = adapter.bridge_states(rec)[0]

    rebuild_calls = {"n": 0}
    real_rebuild = sb.rebuild_env

    def _counting_rebuild(s: BridgeState) -> Any:
        rebuild_calls["n"] += 1
        return real_rebuild(s)

    monkeypatch.setattr(sb, "rebuild_env", _counting_rebuild)

    # Capture the game object each encoder leg receives.
    seen_games: list[int] = []
    from catan_rl.eval.legacy_arch.obs_encoder import ObsEncoder as LegacyObsEncoder
    from catan_rl.policy.obs_encoder import ObsEncoder as NewObsEncoder

    real_new = NewObsEncoder.build_obs
    real_legacy = LegacyObsEncoder.build_obs

    def _spy_new(self: Any, game: Any, *a: Any, **k: Any) -> Any:
        seen_games.append(id(game))
        return real_new(self, game, *a, **k)

    def _spy_legacy(self: Any, game: Any, *a: Any, **k: Any) -> Any:
        seen_games.append(id(game))
        return real_legacy(self, game, *a, **k)

    monkeypatch.setattr(NewObsEncoder, "build_obs", _spy_new)
    monkeypatch.setattr(LegacyObsEncoder, "build_obs", _spy_legacy)

    scorer = _stub_scorer()
    result = scorer.score_state(state)

    assert rebuild_calls["n"] == 1  # ONE rebuild feeds both legs
    assert len(seen_games) == 2 and seen_games[0] == seen_games[1]  # same game object
    # The returned hash is the canonical hash of that one shared state.
    assert result.engine_state_hash == engine_state_hash(real_rebuild(state).game)


# ---------------------------------------------------------------------------
# AC3 — determinism (same seed → identical layouts / v̂ / ΔAUC; diff seed differs)
# ---------------------------------------------------------------------------
def test_ac3_port_assignment_deterministic() -> None:
    adapter = sb.RecordAdapter()
    seed = sb.per_layout_seed("vid", 3, 2)
    a = adapter.port_assignment_for(seed)
    b = adapter.port_assignment_for(seed)
    assert a == b
    assert adapter.port_assignment_for(seed + 1) != a


def test_ac3_per_layout_seed_pinned_formula() -> None:
    import hashlib

    expected = int.from_bytes(hashlib.sha256(b"abc:5:7").digest()[:8], "big")
    assert sb.per_layout_seed("abc", 5, 7) == expected


def test_ac3_bridge_states_and_vhat_bit_identical() -> None:
    rec = _eligible(1)[0]
    adapter = sb.RecordAdapter()
    scorer = _stub_scorer()
    v1 = [scorer.score_state(s) for s in adapter.bridge_states(rec)]
    v2 = [scorer.score_state(s) for s in adapter.bridge_states(rec)]
    assert [x.engine_state_hash for x in v1] == [x.engine_state_hash for x in v2]
    assert [x.new for x in v1] == [x.new for x in v2]
    assert [x.legacy for x in v1] == [x.legacy for x in v2]


def test_ac3_split_deterministic_and_seed_sensitive() -> None:
    vids = [f"v{i}" for i in range(20)]
    a1, b1 = sb.by_video_split(vids, seed=123)
    a2, b2 = sb.by_video_split(vids, seed=123)
    assert (a1, b1) == (a2, b2)
    a3, _ = sb.by_video_split(vids, seed=124)
    assert a3 != a1
    assert set(a1).isdisjoint(b1) and set(a1) | set(b1) == set(vids)


# ---------------------------------------------------------------------------
# AC4 — adapter round-trip pin (adapter state survives serialize→rebuild, hash + v̂)
# ---------------------------------------------------------------------------
def test_ac4_adapter_state_round_trips() -> None:
    rec = _eligible(1)[0]
    adapter = sb.RecordAdapter()
    state = adapter.bridge_states(rec)[0]
    env = rebuild_env(state)
    h0 = engine_state_hash(env.game)
    # Re-serialize the adapter-built env and rebuild: canonical-hash parity.
    reserialized = serialize_post_setup(env)
    env2 = rebuild_env(reserialized)
    assert engine_state_hash(env2.game) == h0
    # v̂ parity across the round-trip (stub policy is a pure function of state).
    scorer = _stub_scorer()
    assert scorer.score_state(state).new == scorer.score_state(reserialized).new


# ---------------------------------------------------------------------------
# AC5 — K=8 caching pin (v̂ computed exactly 8×/game; bootstrap reads cache)
# ---------------------------------------------------------------------------
def test_ac5_vhat_computed_exactly_8x_per_game(monkeypatch: pytest.MonkeyPatch) -> None:
    recs = _eligible(3)
    adapter = sb.RecordAdapter()
    scorer = _stub_scorer()
    calls = {"n": 0}
    real_score = scorer.score_state

    def _counting(state: BridgeState) -> Any:
        calls["n"] += 1
        return real_score(state)

    monkeypatch.setattr(scorer, "score_state", _counting)
    cache = sb.compute_vhat_cache(recs, scorer, adapter)
    assert calls["n"] == sb.PORT_MARGINAL_K * len(recs)
    for entry in cache.values():
        assert len(entry["layouts"]) == sb.PORT_MARGINAL_K

    # The bootstrap resamples the cache and never re-bridges (no more score calls).
    obs = sb.observations_from_cache(cache)
    before = calls["n"]
    sb.cluster_bootstrap_delta_ci(obs, n_boot=50)
    assert calls["n"] == before


# ---------------------------------------------------------------------------
# AC6 — below-floor pin (n<100 → metric+CI emitted, verdict suppressed, banner)
# ---------------------------------------------------------------------------
def test_ac6_below_floor_suppresses_verdict() -> None:
    assert sb.verdict_suppressed(88) is True
    assert sb.verdict_suppressed(sb.FLOOR_N) is False
    assert sb.verdict(88, -1.0, -0.9) == "MEASUREMENT-ONLY"  # even a big regression
    banner = sb.below_floor_banner(88)
    assert "BELOW FLOOR" in banner and "MEASUREMENT-ONLY" in banner and "NO VERDICT" in banner
    assert "n=88" in banner


def test_ac6_above_floor_verdict_branches() -> None:
    # NO-GO iff CI upper bound is below -tolerance (confident regression).
    assert sb.verdict(120, -0.30, -0.05) == "NO-GO"
    # Otherwise GO (CI straddles / small regression within tolerance).
    assert sb.verdict(120, -0.10, 0.05) == "GO"
    assert sb.verdict(120, -0.03, -0.01) == "GO"


# ---------------------------------------------------------------------------
# AC7 — squash-isolation pin (raw v̂; no (3.22, −1.14), no search.value import)
# ---------------------------------------------------------------------------
def test_ac7_no_squash_imported_or_applied() -> None:
    # The docstring explains WHY the squash is rejected; the CODE must never import
    # or apply it. Assert none of the squash identifiers/imports appear.
    src = Path(sb.__file__).read_text()
    for forbidden in (
        "from catan_rl.search",
        "import catan_rl.search",
        "value_from_obs",
        "squash_value",
        "VALUE_SQUASH",
    ):
        assert forbidden not in src, f"squash isolation violated: {forbidden!r} present"


def test_ac7_scoring_uses_raw_value() -> None:
    # The stub returns a known raw value; the cache stores it unchanged (no squash).
    rec = _eligible(1)[0]
    adapter = sb.RecordAdapter()
    scorer = _stub_scorer(new_scale=1.0, legacy_scale=1.0)
    state = adapter.bridge_states(rec)[0]
    val = scorer.score_state(state)
    env = rebuild_env(state)
    from catan_rl.policy.obs_encoder import EnvObsState
    from catan_rl.policy.obs_tensor import obs_to_torch

    es = EnvObsState(
        initial_placement_phase=False,
        setup_step=4,
        roll_pending=True,
        opp_kind=state.opp_kind,
        opp_policy_id=state.opp_policy_id,
        opp_id_masked=False,
    )
    obs = env._obs_encoder.build_obs(  # type: ignore[union-attr]
        env.game, env.agent_player, env.opponent_player, es, hand_tracker=env._hand_tracker
    )
    obs_t = obs_to_torch(obs, torch.device("cpu"), add_batch=True)
    raw = _StubPolicy(1.0).forward(obs_t)["value"].item()
    assert val.new == pytest.approx(raw)


# ---------------------------------------------------------------------------
# AC8 — read-only pins (numbered ckpt only; writes under data/human; provenance; WR)
# ---------------------------------------------------------------------------
def test_ac8_new_ckpt_must_be_numbered(tmp_path: Path) -> None:
    d = sb._POINTER_ARCH_DIRS[1] / "checkpoints"
    with pytest.raises(ValueError, match="numbered checkpoint"):
        sb.validate_new_ckpt(str(d / "latest.pt"))
    with pytest.raises(ValueError, match="numbered checkpoint"):
        sb.validate_new_ckpt(str(d / "promo_ckpt_000000050.pt"))
    # A numbered basename OUTSIDE a pointer-arch dir is also refused.
    with pytest.raises(ValueError, match="pointer-arch run dir"):
        sb.validate_new_ckpt(str(tmp_path / "ckpt_000000399.pt"))


def test_ac8_write_paths_confined_to_data_human(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="under"):
        sb._validate_write_path(sb._REPO_ROOT / "runs" / "train" / "x.json")
    ok = sb.DATA_HUMAN_ROOT / "scoreboard" / "x.json"
    assert sb._validate_write_path(ok) == ok.resolve()


def test_ac8_provenance_hashes_present(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from catan_rl.eval.legacy_arch import _provenance as legacy_prov

    new_ckpt = tmp_path / "ckpt.pt"
    new_ckpt.write_bytes(b"new")
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_bytes(b"{}")
    v11 = tmp_path / "v11.pt"
    v11.write_bytes(b"v11")
    monkeypatch.setattr(sb, "V11_CKPT_PATH", v11)

    hashes = sb.provenance_hashes(new_ckpt, corpus)
    for key in (
        "opening_scoreboard.py",
        "engine_bridge.py",
        "engine_board.py",
        "corpus_jsonl",
        "new_ckpt",
        "v11_ckpt",
        "v11_prefork_commit",
    ):
        assert hashes.get(key)
    assert hashes["v11_prefork_commit"] == legacy_prov.VENDOR_COMMIT


def test_ac8_base_rate_wr_split_by_source() -> None:
    cache = {
        "a:1": {"video_id": "a", "outcome": 1, "source": "tournament"},
        "a:2": {"video_id": "a", "outcome": 0, "source": "tournament"},
        "b:1": {"video_id": "b", "outcome": 1, "source": "tournament"},
    }
    wr = sb.base_rate_wr([], cache, seed=1)  # type: ignore[arg-type]
    assert "tournament" in wr and "__all__" in wr
    assert wr["__all__"]["n"] == 3 and wr["__all__"]["wins"] == 2
    assert wr["tournament"]["wr"] == pytest.approx(2 / 3)


# ---------------------------------------------------------------------------
# AC9 — split file committed before first metric (existence check + pre-reg block)
# ---------------------------------------------------------------------------
def test_ac9_run_requires_committed_split(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # A valid numbered ckpt under a (patched) pointer-arch dir, so validation passes
    # and we reach the split-file existence guard (the AC9 ordering pin).
    fake_dir = tmp_path / "run"
    fake_dir.mkdir()
    ckpt = fake_dir / "ckpt_000000399.pt"
    ckpt.write_bytes(b"x")
    monkeypatch.setattr(sb, "_POINTER_ARCH_DIRS", (fake_dir,))
    out = sb.DATA_HUMAN_ROOT / "scoreboard_test_missing"
    with pytest.raises(FileNotFoundError, match="split file"):
        sb.run_scoreboard(new_ckpt=str(ckpt), output_dir=out)


def test_ac9_split_payload_embeds_pre_registration() -> None:
    if not _CORPUS.is_file():
        pytest.skip("corpus not present")
    payload = sb.build_split_payload(_CORPUS)
    assert payload["pre_registration"] == sb.PRE_REGISTRATION.to_dict()
    assert payload["split_seed"] == sb.SPLIT_SEED
    assert payload["corpus_sha256"]
    assert set(payload["eval_a"]).isdisjoint(payload["holdout_b"])


def test_ac9_load_split_rejects_pre_registration_drift(tmp_path: Path) -> None:
    import json

    good = {
        "split_seed": sb.SPLIT_SEED,
        "eval_a": [],
        "holdout_b": [],
        "pre_registration": sb.PRE_REGISTRATION.to_dict(),
    }
    p = tmp_path / "split.json"
    p.write_text(json.dumps(good))
    assert sb.load_split_file(p)["split_seed"] == sb.SPLIT_SEED
    drift = dict(good)
    drift["pre_registration"] = {**sb.PRE_REGISTRATION.to_dict(), "tolerance_auc": 0.05}
    p.write_text(json.dumps(drift))
    with pytest.raises(ValueError, match="pre_registration"):
        sb.load_split_file(p)


# ---------------------------------------------------------------------------
# AUC provenance sanity (matches scripts/pregate0.py semantics)
# ---------------------------------------------------------------------------
def test_auc_matches_reference_semantics() -> None:
    scores = np.array([0.1, 0.4, 0.35, 0.8])
    labels = np.array([0, 0, 1, 1])
    assert sb.auc(scores, labels) == pytest.approx(0.75)
    assert np.isnan(sb.auc(scores, np.array([1, 1, 1, 1])))
