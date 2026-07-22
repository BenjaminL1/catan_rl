"""Cross-architecture head-to-head eval (new pointer-arch vs v11-era legacy).

Pins the yardstick for the pointer-arch fork:

* **Provenance** — the vendored legacy arch is a faithful byte-copy of the
  pre-fork ``catan_rl.policy`` modules (modulo three declared import rewrites).
* **The gap is real** — a legacy-schema checkpoint canNOT load into the current
  ``CatanPolicy`` (this is *why* the cross-arch bridge exists).
* **Legacy round-trip** — a legacy checkpoint loads STRICT into the vendored
  arch and the legacy encoder emits the v11 obs schema (``current_player_main``
  = 54, ``next_player_main`` = 61, no ``global_features`` / ``is_setup``).
* **Bridge == in-process** — ``CrossArchEnv(legacy_opponent=False)`` drives a
  game bit-identically to a plain :class:`CatanEnv` (the subclass only diverts
  the opponent's obs-build seam; nothing else moves).
* **Determinism** — ``cross_arch_h2h`` is bit-for-bit reproducible at a fixed
  seed and RNG-contained (a perturbation between runs cannot leak in).

Every test builds FRESH random-init policies (saved to ``tmp_path``) — no
reliance on the gitignored ``runs/`` checkpoints — so the suite runs in CI.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch

from catan_rl.checkpoint.manager import save_checkpoint
from catan_rl.env.catan_env import CatanEnv
from catan_rl.eval.cross_arch import CrossArchEnv, build_legacy_opponent, cross_arch_h2h
from catan_rl.eval.harness import EvalHarness, EvalMatchupResult, EvalResult
from catan_rl.eval.legacy_arch import _provenance as prov
from catan_rl.eval.legacy_arch.network import CatanPolicy as LegacyCatanPolicy
from catan_rl.eval.legacy_arch.obs_encoder import ObsEncoder as LegacyObsEncoder
from catan_rl.policy.board_geometry import build_geometry
from catan_rl.policy.network import CatanPolicy
from catan_rl.policy.obs_encoder import EnvObsState
from catan_rl.selfplay.snapshot_opponent import FrozenSnapshotOpponent

_REPO_ROOT = Path(__file__).resolve().parents[3]


# ---------------------------------------------------------------------------
# Fixtures — fresh random-init policies of each arch
# ---------------------------------------------------------------------------


def _new_policy(seed: int) -> CatanPolicy:
    torch.manual_seed(seed)
    policy = CatanPolicy()
    policy.set_board_geometry(build_geometry().as_dict_of_tensors())
    return policy


def _legacy_policy(seed: int) -> LegacyCatanPolicy:
    torch.manual_seed(seed)
    policy = LegacyCatanPolicy()
    policy.set_board_geometry(build_geometry().as_dict_of_tensors())
    return policy


def _save(path: Path, policy: torch.nn.Module) -> str:
    return str(
        save_checkpoint(
            path,
            config={},
            policy=policy,
            optimizer=None,
            update_idx=0,
            global_step=0,
            capture_rng=False,
        )
    )


# ---------------------------------------------------------------------------
# Provenance — vendored arch is a faithful copy of the pinned pre-fork commit
# ---------------------------------------------------------------------------


@pytest.mark.skipif(shutil.which("git") is None, reason="git required to verify provenance")
def test_legacy_arch_is_faithful_copy_of_pinned_commit() -> None:
    """Every generated module equals ``render_from_original(git show <commit>)``.

    Guarantees the vendored arch is byte-identical to the pre-fork policy code
    (modulo the three declared import rewrites) — so v11-via-vendored-legacy
    makes the SAME decisions v11 made under its original code.
    """
    dest = _REPO_ROOT / "src" / "catan_rl" / "eval" / "legacy_arch"
    for module in prov.VENDORED_MODULES:
        show = subprocess.run(
            [
                "git",
                "-C",
                str(_REPO_ROOT),
                "show",
                f"{prov.VENDOR_COMMIT}:{prov.SRC_PREFIX}/{module}.py",
            ],
            capture_output=True,
            text=True,
        )
        if show.returncode != 0:
            pytest.skip(f"pinned commit {prov.VENDOR_COMMIT} unavailable in this checkout")
        expected = prov.render_from_original(module, show.stdout)
        actual = (dest / f"{module}.py").read_text()
        assert actual == expected, f"{module}.py has drifted from the pinned pre-fork source"


def test_vendored_modules_carry_generated_banner() -> None:
    dest = _REPO_ROOT / "src" / "catan_rl" / "eval" / "legacy_arch"
    for module in prov.VENDORED_MODULES:
        head = (dest / f"{module}.py").read_text().splitlines()[0]
        assert "GENERATED" in head and prov.VENDOR_COMMIT_SHORT in head


# ---------------------------------------------------------------------------
# The gap is real — legacy schema is incompatible with the current arch
# ---------------------------------------------------------------------------


def test_current_arch_cannot_load_legacy_checkpoint(tmp_path: Path) -> None:
    """A legacy-schema state dict must NOT load into the current pointer-arch
    ``CatanPolicy`` — the raison d'être of the whole cross-arch bridge."""
    ckpt = _save(tmp_path / "legacy.pt", _legacy_policy(1))
    from catan_rl.replay.player_factory import PlayerSpec, build_actor

    with pytest.raises(RuntimeError):
        build_actor(PlayerSpec(kind="policy", ckpt_path=ckpt), seed=0, device="cpu")


# ---------------------------------------------------------------------------
# Legacy round-trip — strict load + v11 obs schema
# ---------------------------------------------------------------------------


def test_legacy_obs_encoder_emits_v11_schema() -> None:
    env = CatanEnv(opponent_type="heuristic", max_turns=50)
    obs_new, _ = env.reset(seed=0, options={"agent_seat": 0})
    try:
        # New arch: 67-dim current player + a global block + is_setup.
        assert obs_new["current_player_main"].shape == (67,)
        assert "global_features" in obs_new and "is_setup" in obs_new

        legacy_enc = LegacyObsEncoder(env.game.board)
        env_state = EnvObsState(
            initial_placement_phase=env.initial_placement_phase,
            setup_step=env._setup_step,
            roll_pending=env.roll_pending,
            discard_pending=env.discard_pending,
            robber_placement_pending=env.robber_placement_pending,
            road_building_roads_left=env.road_building_roads_left,
            last_dice_roll=env.last_dice_roll,
        )
        obs_old = legacy_enc.build_obs(
            env.game,
            env.agent_player,
            env.opponent_player,
            env_state,
            hand_tracker=env._hand_tracker,
        )
        # Legacy arch: 54-dim current player, 61-dim opponent, no new blocks.
        assert obs_old["current_player_main"].shape == (54,)
        assert obs_old["next_player_main"].shape == (61,)
        assert "global_features" not in obs_old and "is_setup" not in obs_old
    finally:
        env.close()


def test_build_legacy_opponent_loads_strict_and_samples(tmp_path: Path) -> None:
    ckpt = _save(tmp_path / "legacy.pt", _legacy_policy(3))
    opp = build_legacy_opponent(ckpt, device="cpu", seed=0)
    assert isinstance(opp, FrozenSnapshotOpponent)

    # It samples a legal action from a legacy obs + the env's live masks.
    env = CatanEnv(opponent_type="heuristic", max_turns=50)
    env.reset(seed=0, options={"agent_seat": 0})
    try:
        legacy_enc = LegacyObsEncoder(env.game.board)
        env_state = EnvObsState(
            initial_placement_phase=env.initial_placement_phase,
            setup_step=env._setup_step,
            roll_pending=env.roll_pending,
        )
        obs = legacy_enc.build_obs(
            env.game,
            env.agent_player,
            env.opponent_player,
            env_state,
            hand_tracker=env._hand_tracker,
        )
        masks = env.get_action_masks()
        from catan_rl.policy.obs_tensor import masks_to_torch, obs_to_torch

        action = opp.sample(
            obs_to_torch(obs, torch.device("cpu"), add_batch=True),
            masks_to_torch(masks, torch.device("cpu"), add_batch=True),
        )
        a = action[0].cpu().numpy().astype(np.int64)
        assert a.shape == (6,)
        assert bool(masks["type"][a[0]])  # the chosen action type is legal
    finally:
        env.close()


# ---------------------------------------------------------------------------
# Bridge == in-process — CrossArchEnv passthrough ≡ CatanEnv
# ---------------------------------------------------------------------------


def _run_games_through(env: CatanEnv, champion: CatanPolicy, opp: FrozenSnapshotOpponent) -> list:
    harness = EvalHarness(
        opponent_types=("snapshot",),
        n_games_per_seat=2,
        seed=0,
        device=torch.device("cpu"),
        max_turns=50,
    )
    env.set_snapshot_opponent(opp)
    torch.manual_seed(123)
    return harness._run_matchup_games(env, champion, seed_label="equiv")


def test_crossarchenv_passthrough_equals_catanenv() -> None:
    """With a new-arch opponent, ``CrossArchEnv`` (legacy_opponent=False) drives
    games bit-identically to a plain ``CatanEnv`` — the subclass is a pure
    pass-through except on the legacy opponent's obs-build seam. This is the
    "new-vs-new through the bridge == the in-process result" guarantee."""
    champion = _new_policy(0)
    opp_policy = _new_policy(1)

    plain = CatanEnv(opponent_type="snapshot", max_turns=50)
    games_plain = _run_games_through(
        plain, champion, FrozenSnapshotOpponent(opp_policy, device=torch.device("cpu"), seed=0)
    )
    plain.close()

    bridge = CrossArchEnv(opponent_type="snapshot", max_turns=50, legacy_opponent=False)
    games_bridge = _run_games_through(
        bridge, champion, FrozenSnapshotOpponent(opp_policy, device=torch.device("cpu"), seed=0)
    )
    bridge.close()

    assert [g.won for g in games_plain] == [g.won for g in games_bridge]
    assert [g.n_turns for g in games_plain] == [g.n_turns for g in games_bridge]
    assert [g.final_vp_agent for g in games_plain] == [g.final_vp_agent for g in games_bridge]
    assert [g.final_vp_opp for g in games_plain] == [g.final_vp_opp for g in games_bridge]


def test_bridge_newnew_matches_inprocess_evaluate(tmp_path: Path) -> None:
    """Guard (a), literal form: a new-arch policy vs itself driven through
    ``CrossArchEnv`` reproduces the in-process ``evaluate_policy_vs_policy``
    result bit-for-bit at a fixed seed — proving the bridge does not distort
    play. Replicates ``evaluate_policy_vs_policy``'s construction exactly,
    swapping ONLY ``CatanEnv`` -> ``CrossArchEnv(legacy_opponent=False)``."""
    from catan_rl.eval.harness import evaluate_policy_vs_policy
    from catan_rl.replay.player_factory import PlayerSpec, build_actor

    champion = _new_policy(0)
    opp_ckpt = _save(tmp_path / "opp.pt", _new_policy(1))

    ref = evaluate_policy_vs_policy(
        champion, opp_ckpt, n_games=4, seed=0, device="cpu", max_turns=50
    )

    torch.manual_seed(0)
    actor = build_actor(PlayerSpec(kind="policy", ckpt_path=opp_ckpt), seed=0, device="cpu")
    opponent = FrozenSnapshotOpponent(actor.policy, device=actor.device, seed=0)  # type: ignore[attr-defined]
    harness = EvalHarness(
        opponent_types=("snapshot",),
        n_games_per_seat=2,
        seed=0,
        device=torch.device("cpu"),
        max_turns=50,
    )
    bridge_env = CrossArchEnv(opponent_type="snapshot", max_turns=50, legacy_opponent=False)
    bridge_env.set_snapshot_opponent(opponent)
    try:
        games = harness._run_matchup_games(bridge_env, champion, seed_label=opp_ckpt)
    finally:
        bridge_env.close()

    assert sum(1 for g in games if g.won) == ref.wins
    assert [g.won for g in games] == [g.won for g in ref.games]
    assert [g.n_turns for g in games] == [g.n_turns for g in ref.games]
    assert [g.final_vp_agent for g in games] == [g.final_vp_agent for g in ref.games]


# ---------------------------------------------------------------------------
# Engine-parity guard (guard c)
# ---------------------------------------------------------------------------


def test_engine_parity_holds_at_head() -> None:
    """Tripwire: at HEAD the live engine matches the pinned pre-fork tree, so the
    guard returns a stamp without raising. This test FIRES (correctly) if the
    engine is ever changed without re-vendoring + re-pinning the legacy arch."""
    from catan_rl.eval.engine_parity import assert_engine_parity

    stamp = assert_engine_parity(strict=True)
    assert set(stamp) == {"engine", "board_geometry"}


def test_engine_parity_bypass_is_a_noop() -> None:
    from catan_rl.eval.engine_parity import assert_engine_parity

    assert assert_engine_parity(strict=False) == {
        "engine": "unchecked",
        "board_geometry": "unchecked",
    }


def test_engine_parity_detects_drift(monkeypatch: pytest.MonkeyPatch) -> None:
    from catan_rl.eval import engine_parity

    probe = engine_parity.assert_engine_parity(strict=True)
    if probe["engine"] in ("unverified", "unchecked"):
        pytest.skip("git/repo unavailable — cannot exercise drift detection")
    # A bogus pin must be caught as drift (fail-closed on a detected difference).
    monkeypatch.setattr(engine_parity, "PINNED_ENGINE_TREE", "0" * 40)
    with pytest.raises(engine_parity.EngineParityError, match="ENGINE DRIFT"):
        engine_parity.assert_engine_parity(strict=True)


# ---------------------------------------------------------------------------
# cross_arch_h2h — shape, symmetrization, determinism, negative control
# ---------------------------------------------------------------------------


def test_cross_arch_h2h_legacy_runs_and_symmetrizes(tmp_path: Path) -> None:
    new_ckpt = _save(tmp_path / "new.pt", _new_policy(0))
    old_ckpt = _save(tmp_path / "legacy.pt", _legacy_policy(1))
    result = cross_arch_h2h(
        new_ckpt=new_ckpt, old_ckpt=old_ckpt, old_arch="legacy", n_games=4, seed=0, max_turns=50
    )
    assert isinstance(result, EvalMatchupResult)
    assert isinstance(result, EvalResult)
    assert result.opponent_ref == old_ckpt
    assert result.n == 4
    assert result.n_seat0 == 2 and result.n_seat1 == 2
    assert 0.0 <= result.wr <= 1.0
    assert np.isfinite(result.ci.lower) and np.isfinite(result.ci.upper)
    assert result.rules_violations == ()  # legacy opponent plays only legal moves


def test_cross_arch_h2h_is_deterministic(tmp_path: Path) -> None:
    new_ckpt = _save(tmp_path / "new.pt", _new_policy(0))
    old_ckpt = _save(tmp_path / "legacy.pt", _legacy_policy(1))
    r1 = cross_arch_h2h(new_ckpt=new_ckpt, old_ckpt=old_ckpt, n_games=4, seed=0, max_turns=50)
    _ = torch.rand(2048)  # perturb global RNG — must not leak into r2
    r2 = cross_arch_h2h(new_ckpt=new_ckpt, old_ckpt=old_ckpt, n_games=4, seed=0, max_turns=50)
    assert r1.wins == r2.wins
    assert [g.won for g in r1.games] == [g.won for g in r2.games]
    assert [g.n_turns for g in r1.games] == [g.n_turns for g in r2.games]
    assert [g.final_vp_agent for g in r1.games] == [g.final_vp_agent for g in r2.games]


def test_cross_arch_h2h_different_seed_differs(tmp_path: Path) -> None:
    new_ckpt = _save(tmp_path / "new.pt", _new_policy(0))
    old_ckpt = _save(tmp_path / "legacy.pt", _legacy_policy(1))
    a = cross_arch_h2h(new_ckpt=new_ckpt, old_ckpt=old_ckpt, n_games=4, seed=0, max_turns=50)
    b = cross_arch_h2h(new_ckpt=new_ckpt, old_ckpt=old_ckpt, n_games=4, seed=1, max_turns=50)
    assert [g.n_turns for g in a.games] != [g.n_turns for g in b.games]


def test_cross_arch_h2h_rejects_bad_arch(tmp_path: Path) -> None:
    new_ckpt = _save(tmp_path / "new.pt", _new_policy(0))
    with pytest.raises(ValueError, match="old_arch"):
        cross_arch_h2h(new_ckpt=new_ckpt, old_ckpt=new_ckpt, old_arch="bogus", n_games=2)  # type: ignore[arg-type]
