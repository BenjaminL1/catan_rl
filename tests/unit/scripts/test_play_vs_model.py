"""Pins for ``scripts/play_vs_model.py``.

Two things are pinned here:

* the **rename** (D1) — the old script is gone with no shim, nothing outside
  the spec still names it, and the locked design doc still reserves the new
  path (and is NOT edited to match the build); and
* the **policy-internals extractor** (D4) — it recomputes the six masked
  distributions script-side from a single forward, so nothing on the PPO path
  changes, and it hands the schema plain Python numbers.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[3]
_SCRIPT = _REPO / "scripts" / "play_vs_model.py"
#: Assembled at runtime ON PURPOSE. Spelling the old name literally would make
#: this file its own last surviving reference the moment it is tracked, and the
#: grep pin below would fail on itself.
_OLD_STEM = "play_vs_" + "v8"


def _load_module():  # type: ignore[no-untyped-def]
    spec = importlib.util.spec_from_file_location("play_vs_model_module", _SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["play_vs_model_module"] = mod
    spec.loader.exec_module(mod)
    return mod


class TestRenamePin:
    def test_new_path_exists_and_old_one_does_not(self) -> None:
        assert _SCRIPT.is_file()
        assert not (_REPO / "scripts" / f"{_OLD_STEM}.py").exists(), "no deprecation shim"

    def test_no_old_name_reference_survives(self) -> None:
        # ``--untracked`` so a not-yet-staged file cannot hide an offender (and
        # so this pin does not flip red the moment the change is committed);
        # git still skips venvs / runs/ via .gitignore.
        out = subprocess.run(
            ["git", "grep", "--untracked", "-l", _OLD_STEM],
            cwd=_REPO,
            capture_output=True,
            text=True,
            check=False,
        ).stdout.split()
        # The spec itself documents the rename and is allowed to name the old file.
        offenders = [p for p in out if not p.startswith(".claude/veriloop/specs/")]
        assert offenders == [], f"stale {_OLD_STEM} references: {offenders}"

    def test_locked_design_doc_still_reserves_the_path(self) -> None:
        rel = "docs/plans/v2/design.md"
        design = (_REPO / rel).read_text(encoding="utf-8")
        assert "play_vs_model.py" in design
        # D1 forbids editing the ratified design to match the build, so pin
        # that the file is UNMODIFIED (not merely that it mentions the path).
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--", rel],
            cwd=_REPO,
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()
        assert dirty == "", f"{rel} was modified: {dirty}"
        committed = subprocess.run(
            ["git", "diff", "--name-only", "origin/main...HEAD", "--", rel],
            cwd=_REPO,
            capture_output=True,
            text=True,
            check=False,
        )
        assert committed.stdout.strip() == "", f"{rel} was edited on this branch"


class TestPolicyInternals:
    @pytest.fixture(scope="class")
    def captured(self):  # type: ignore[no-untyped-def]
        torch = pytest.importorskip("torch")
        from catan_rl.env.catan_env import CatanEnv
        from catan_rl.policy.board_geometry import build_geometry
        from catan_rl.policy.network import CatanPolicy
        from catan_rl.replay.player_factory import _PolicyActor

        mod = _load_module()
        policy = CatanPolicy()
        policy.set_board_geometry(build_geometry().as_dict_of_tensors())
        policy.eval()
        actor = _PolicyActor(
            kind="policy", ckpt_path="<fresh>", policy=policy, device=torch.device("cpu")
        )
        env = CatanEnv(opponent_type="heuristic", max_turns=50)
        obs, _info = env.reset(seed=3, options={"agent_seat": 0})
        masks = env.get_action_masks()
        action = actor.select_action(obs, masks)
        return mod.capture_policy_internals(actor, obs, masks, action), masks, action

    def test_shapes(self, captured) -> None:  # type: ignore[no-untyped-def]
        internals, masks, action = captured
        assert len(internals.type_mask) == 13
        assert len(internals.type_probs) == 13
        assert tuple(internals.type_mask) == tuple(bool(b) for b in masks["type"])
        assert internals.chosen_action == tuple(int(x) for x in action)

    def test_type_probs_are_a_masked_distribution(self, captured) -> None:  # type: ignore[no-untyped-def]
        internals, _masks, _action = captured
        assert abs(sum(internals.type_probs) - 1.0) < 1e-4
        for legal, p in zip(internals.type_mask, internals.type_probs, strict=True):
            if not legal:
                assert p == pytest.approx(0.0, abs=1e-6)

    def test_pointer_heads_are_truncated_to_top_8(self, captured) -> None:  # type: ignore[no-untyped-def]
        internals, _masks, _action = captured
        for top in (internals.corner_top, internals.edge_top, internals.tile_top):
            assert len(top) <= 8
            probs = [p for _i, p in top]
            assert probs == sorted(probs, reverse=True)

    def test_everything_is_plain_python(self, captured) -> None:  # type: ignore[no-untyped-def]
        internals, _masks, _action = captured
        flat = [
            *internals.type_probs,
            internals.value,
            *[p for _i, p in internals.corner_top],
            *(internals.belief_logits or ()),
        ]
        for x in flat:
            assert type(x) is float, f"{x!r} is {type(x)}, not a plain float"
        for i in internals.chosen_action:
            assert type(i) is int
        for b in internals.type_mask:
            assert type(b) is bool
        # np scalars are float subclasses in some versions — pin explicitly.
        assert not any(isinstance(x, np.generic) for x in flat)


class TestRelevanceGatedHeads:
    """Only the heads the chosen TYPE consults may be recorded.

    When a head is irrelevant its mask is all-False and
    ``masked_log_softmax`` returns the uniform safe fallback — so recording it
    stores 1/54 across every corner and dresses noise up as an opinion."""

    def test_irrelevant_heads_are_empty_for_end_turn(self) -> None:
        torch = pytest.importorskip("torch")
        from catan_rl.env.catan_env import ActionType, CatanEnv
        from catan_rl.policy.board_geometry import build_geometry
        from catan_rl.policy.network import CatanPolicy
        from catan_rl.replay.player_factory import _PolicyActor

        mod = _load_module()
        policy = CatanPolicy()
        policy.set_board_geometry(build_geometry().as_dict_of_tensors())
        policy.eval()
        actor = _PolicyActor(
            kind="policy", ckpt_path="<fresh>", policy=policy, device=torch.device("cpu")
        )
        env = CatanEnv(opponent_type="heuristic", max_turns=50)
        obs, _info = env.reset(seed=3, options={"agent_seat": 0})
        masks = env.get_action_masks()
        # END_TURN consults NO pointer head and NO resource head.
        action = np.array([int(ActionType.END_TURN), 0, 0, 0, 0, 0], dtype=np.int64)
        internals = mod.capture_policy_internals(actor, obs, masks, action)
        assert internals.corner_top == ()
        assert internals.edge_top == ()
        assert internals.tile_top == ()
        assert internals.res1_probs == ()
        assert internals.res2_probs == ()
        # The type distribution is always relevant and stays dense.
        assert len(internals.type_probs) == 13

    def test_bank_trade_records_both_resource_arguments(self) -> None:
        torch = pytest.importorskip("torch")
        from catan_rl.env.catan_env import ActionType, CatanEnv
        from catan_rl.policy.board_geometry import build_geometry
        from catan_rl.policy.network import CatanPolicy
        from catan_rl.replay.player_factory import _PolicyActor

        mod = _load_module()
        policy = CatanPolicy()
        policy.set_board_geometry(build_geometry().as_dict_of_tensors())
        policy.eval()
        actor = _PolicyActor(
            kind="policy", ckpt_path="<fresh>", policy=policy, device=torch.device("cpu")
        )
        env = CatanEnv(opponent_type="heuristic", max_turns=50)
        obs, _info = env.reset(seed=3, options={"agent_seat": 0})
        masks = env.get_action_masks()
        action = np.array([int(ActionType.BANK_TRADE), 0, 0, 0, 0, 1], dtype=np.int64)
        internals = mod.capture_policy_internals(actor, obs, masks, action)
        # "gave 0.62 to BankTrade" is unreadable without the give/get weights.
        assert len(internals.res1_probs) == 5
        assert len(internals.res2_probs) == 5
        assert all(type(x) is float for x in internals.res1_probs + internals.res2_probs)
        assert internals.corner_top == ()


class TestRecorderIsNeverCloned:
    """``--search`` deep-copies the live env once per MCTS simulation.

    A bound method cannot carry a ``__deepcopy__`` guard (``copy`` rebuilds it
    as the ORIGINAL function bound to a copied instance), so the recorder
    subscribes a standalone callable that returns an inert stand-in. Without
    it the first broadcast inside any clone raised ``AttributeError`` and
    ``python scripts/play_vs_model.py --search`` aborted mid-game."""

    def _env_and_recorder(self):  # type: ignore[no-untyped-def]
        pytest.importorskip("torch")
        from catan_rl.env.catan_env import CatanEnv

        mod = _load_module()
        env = CatanEnv(opponent_type="heuristic", max_turns=50)
        env.reset(seed=5, options={"agent_seat": 0})
        return mod, env, mod._HumanGameRecorder(env, bot_seat=0)

    def test_a_clone_emits_without_touching_the_recorder(self) -> None:
        import copy

        _mod, env, recorder = self._env_and_recorder()
        clone = copy.deepcopy(env)
        before = len(recorder._setup_complete_snaps)
        # Any broadcast inside a simulated world must be a no-op for the record.
        clone.game.broadcast.dice_roll("Agent", 8)
        clone.game.broadcast.emit(recorder._setup_complete_type)
        assert len(recorder._setup_complete_snaps) == before
        assert recorder.steps == []

    def test_the_clone_gets_an_inert_subscriber(self) -> None:
        import copy

        mod, env, recorder = self._env_and_recorder()
        clone = copy.deepcopy(env)
        subs = clone.game.broadcast._subscribers
        assert any(isinstance(s, mod._DetachedSubscriber) for s in subs)
        # The real recorder must not be reachable from the clone at all.
        assert not any(getattr(s, "__self__", None) is recorder for s in subs)

    def test_the_live_recorder_still_sees_events(self) -> None:
        _mod, env, recorder = self._env_and_recorder()
        before = len(recorder._setup_complete_snaps)
        env.game.broadcast.emit(recorder._setup_complete_type)
        assert len(recorder._setup_complete_snaps) == before + 1
