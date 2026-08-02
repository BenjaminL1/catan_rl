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
        # Two exemptions, both for artifacts that RECORD the past rather than
        # describe the present:
        #   * ``specs/`` — a spec documents the rename, so it must name the old
        #     file to explain what it renamed.
        #   * ``history/`` — dated, machine-written attestation records that quote
        #     spec text verbatim at the moment a branch landed. They are immutable
        #     audit snapshots; an old name inside one is a true statement about a
        #     past tree, and rewriting it would falsify the record.
        # Nothing else is exempt: the pin exists so no LIVE code or doc leaves a
        # reader hunting for a script that no longer exists. (Note this comment
        # must not spell the old stem either — the grep would match this file.)
        _RECORDS_THE_PAST = (".claude/veriloop/specs/", ".claude/veriloop/history/")
        offenders = [p for p in out if not p.startswith(_RECORDS_THE_PAST)]
        assert offenders == [], f"stale {_OLD_STEM} references: {offenders}"

    def test_locked_design_doc_still_reserves_the_path(self) -> None:
        rel = "docs/plans/v2/design.md"
        design = (_REPO / rel).read_text(encoding="utf-8")
        assert "play_vs_model.py" in design
        # ``play-vs-model-recorder`` D1 forbids an implementer REWRITING the
        # ratified design to match what it built, and the original pin encoded
        # that as "the file is byte-clean".
        #
        # AUTHORITY FOR THE NARROWER RULE BELOW: the owner-ratified AMENDMENT to
        # that spec's D1 (2026-07-31), which keeps the prohibition intact and
        # exempts ONLY owner-directed docs-sync caveats, insertion-only, no
        # threshold value changed. This test does not create that exemption and
        # must not widen it — it enforces the amendment's limit. If a future
        # slice needs more room, amend the spec, never this comment.
        for rev in ("HEAD", "origin/main...HEAD"):
            diff = subprocess.run(
                ["git", "diff", "-U0", rev, "--", rel],
                cwd=_REPO,
                capture_output=True,
                text=True,
                check=False,
            ).stdout.splitlines()
            added = {ln[1:] for ln in diff if ln.startswith("+") and not ln.startswith("+++")}
            # A pure insertion at EOF against a file whose last line had no
            # trailing newline shows up as a -/+ pair with identical content
            # (git re-emits the line to add the newline). That is not a rewrite.
            removed = [
                ln
                for ln in diff
                if ln.startswith("-") and not ln.startswith("---") and ln[1:] not in added
            ]
            assert removed == [], f"{rel} had lines REWRITTEN, not merely inserted: {removed}"
            touched = [
                ln
                for ln in diff
                if ln.startswith(("+", "-"))
                and not ln.startswith(("+++", "---"))
                and "play_vs_" in ln
            ]
            assert touched == [], f"{rel}'s reserved script path was edited: {touched}"


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


class TestMoveLogLines:
    """The on-screen move log is fed from the ACTION TUPLE, not the broadcast.

    Two broadcast facts force this: ``PLAY_KNIGHT`` / ``PLAY_ROAD_BUILDER`` emit
    NO event at all (the counters are incremented directly in the env), and
    ``BUILD`` events carry ``location=-1`` as a documented sentinel. A
    broadcast-fed log would therefore omit knights entirely and could never name
    a build's index."""

    def test_the_default_label_is_byte_identical(self) -> None:
        """FROZEN: this string is written into ``games.jsonl`` (``bot_action_label``)
        and into the replay's step note. Re-pointing it would silently change the
        meaning of content already on disk."""
        mod = _load_module()
        cases = {
            (0, 7, 0, 0, 0, 0): "Build settlement",
            (1, 7, 0, 0, 0, 0): "Build city",
            (2, 0, 31, 0, 0, 0): "Build road",
            (3, 0, 0, 0, 0, 0): "End turn",
            (4, 0, 0, 12, 0, 0): "Move robber",
            (5, 0, 0, 0, 0, 0): "Buy dev card",
            (6, 0, 0, 12, 0, 0): "Play Knight",
            (12, 0, 0, 0, 0, 0): "Roll dice",
        }
        for action, expected in cases.items():
            assert mod._describe_bot_move(np.array(action, dtype=np.int64)) == expected

    def test_a_knight_play_reaches_the_log(self) -> None:
        mod = _load_module()
        line = mod._describe_bot_move(np.array([6, 0, 0, 12, 0, 0]), with_location=True)
        assert "Knight" in line

    def test_a_knight_play_never_names_a_hex(self) -> None:
        """PLAY_KNIGHT does not consume head 3, so head 3 is MEANINGLESS here.

        The robber destination arrives in the SEPARATE MOVE_ROBBER action on the
        next step; during a main turn `tile_mask` is all-False, so `action[3]`
        is a uniformly random index. Printing it would assert a false public
        fact AND contradict the MOVE_ROBBER line logged one step later.
        """
        mod = _load_module()
        for tile in (0, 7, 12, 18):
            line = mod._describe_bot_move(np.array([6, 0, 0, tile, 0, 0]), with_location=True)
            assert line == "Play Knight"
            assert "hex" not in line
        robber = mod._describe_bot_move(np.array([4, 0, 0, 12, 0, 0]), with_location=True)
        assert robber == "Move robber to hex12"

    def test_a_build_names_its_index_not_the_sentinel(self) -> None:
        mod = _load_module()
        settle = mod._describe_bot_move(np.array([0, 23, 0, 0, 0, 0]), with_location=True)
        road = mod._describe_bot_move(np.array([2, 0, 31, 0, 0, 0]), with_location=True)
        city = mod._describe_bot_move(np.array([1, 5, 0, 0, 0, 0]), with_location=True)
        assert settle.endswith("v23") and "-1" not in settle
        assert road.endswith("e31") and "-1" not in road
        assert city.endswith("v5") and "-1" not in city

    def test_no_bot_hand_content_is_reachable_from_any_line(self) -> None:
        """LEAK PIN: only PUBLIC resource facts may appear.

        A bot BUY_DEV_CARD must never name the card drawn, and no action label
        may enumerate resources the bot merely HOLDS. The only resource names a
        line may carry belong to the acted move itself (bank-trade sides, a YoP
        pick, a Monopoly call) — all public the moment they happen."""
        from catan_rl.env.catan_env import RESOURCES_CW

        mod = _load_module()
        private = ("BUY_DEV_CARD", 5), ("PLAY_KNIGHT", 6), ("BUILD_SETTLEMENT", 0)
        for _name, type_id in private:
            for res_idx in range(5):
                action = np.array([type_id, 3, 4, 12, res_idx, res_idx], dtype=np.int64)
                line = mod._describe_bot_move(action, with_location=True)
                assert not any(r in line for r in RESOURCES_CW), line
        # And no dev-card TYPE name can appear on a buy.
        buy = mod._describe_bot_move(np.array([5, 0, 0, 0, 0, 0]), with_location=True)
        for card in ("KNIGHT", "MONOPOLY", "ROADBUILDER", "YEAROFPLENTY", "VP"):
            assert card not in buy.upper().replace(" ", "")


class TestGameOverLine:
    """D4's ONE gate: ``GAME_END`` carries a VP-card-inclusive total, and the
    terminal move-log line is the only place it could reach the screen.

    The blind default must render the bot's score through ``_visible_vp``. This
    pin exists so that dropping the gate re-opens a hidden-VP leak loudly rather
    than silently — the same leak class already fixed and pinned in the obs."""

    class _Bot:
        victoryPoints = 14
        devCards = {"VP": 3, "KNIGHT": 1}

    def test_blind_by_default_shows_visible_vp_only(self) -> None:
        mod = _load_module()
        line = mod._game_over_log_line(self._Bot(), 9, reveal_bot=False)
        assert line == "GAME OVER — Bot 11 - 9 You"
        assert "14" not in line

    def test_reveal_bot_restores_the_total(self) -> None:
        mod = _load_module()
        assert mod._game_over_log_line(self._Bot(), 9, reveal_bot=True) == (
            "GAME OVER — Bot 14 - 9 You"
        )

    def test_the_harness_routes_the_line_through_the_helper(self) -> None:
        # Guards against the ternary being re-inlined (and then dropped) at the
        # call site, which would leave the helper pinned but unused.
        src = _SCRIPT.read_text(encoding="utf-8")
        assert src.count("GAME OVER") == 1, "the string must live only in the helper"
        assert "_game_over_log_line(env.agent_player" in src


class TestHarnessWiring:
    """The two lines that make the log EXIST are otherwise untested.

    Every other pin in this file targets the pure label helpers, so deleting
    ``built.move_log = hud_log`` (the view factory) or the bot append in
    ``play_interactive`` would leave a permanently-empty strip with the whole
    suite still green. ``_log_move`` is pinned behaviourally; the two
    harness-only statements are pinned at the source, the same technique
    ``TestGameOverLine.test_the_harness_routes_the_line_through_the_helper``
    already uses for the terminal line."""

    def test_log_move_reaches_the_deque_only_through_the_view(self) -> None:
        from collections import deque
        from types import SimpleNamespace

        mod = _load_module()
        cls = mod._build_human_env_class()

        # No view (e.g. an MCTS clone, whose __deepcopy__ drops _human_view):
        # logging must be a silent no-op, never an AttributeError.
        clone = SimpleNamespace(_human_view=None)
        cls._log_move(clone, "You rolled 8")

        # A view with no log attached (engine playCatan) is also a no-op.
        cls._log_move(SimpleNamespace(_human_view=SimpleNamespace(move_log=None)), "x")

        log: deque[str] = deque(maxlen=6)
        env = SimpleNamespace(_human_view=SimpleNamespace(move_log=log))
        cls._log_move(env, "You rolled 8")
        assert list(log) == ["You rolled 8"]
        # Empty / None deltas (a cancelled click) must not add a blank line.
        cls._log_move(env, None)
        cls._log_move(env, "")
        assert list(log) == ["You rolled 8"]

    def test_the_view_factory_attaches_the_harness_deque(self) -> None:
        src = _SCRIPT.read_text(encoding="utf-8")
        assert "built.move_log = hud_log" in src, "the strip would render nothing"
        assert src.count("hud_log: Any = deque(maxlen=MOVE_LOG_LINES)") == 1

    def test_every_bot_move_is_appended_to_the_log(self) -> None:
        src = _SCRIPT.read_text(encoding="utf-8")
        assert 'hud_log.append(f"Bot: {_describe_bot_move(action, with_location=True)}")' in src

    def test_the_human_input_loops_still_log(self) -> None:
        # AC5/D7: the log is BOTH players'. If these call sites vanish the strip
        # silently becomes bot-only, which reads as a working feature.
        #
        # The bar was 4 while the human had a pre-roll dev-card window; removing
        # that window (fairness — the bot has no counterpart, see
        # TestNoPreRollWindow) took its logging site with it. Three remain: the
        # setup-draft loop, the robber loop and the main-turn loop.
        src = _SCRIPT.read_text(encoding="utf-8")
        assert src.count("self._log_human_action(before)") >= 3
        assert 'self._log_move(f"You rolled {dice}")' in src
        assert 'self._log_move(f"You moved the robber to hex{hex_i}")' in src


class TestHumanMoveDelta:
    """The human's own moves are logged by diffing their state across one click.

    A cancelled click (every build button can be backed out of) must produce no
    entry at all, and a build must name its index — the same requirement the
    ``location=-1`` sentinel makes impossible from the broadcast."""

    class _FakeEnv:
        _vertex_to_idx = {"vA": 7, "vB": 9}
        _edge_to_idx = {("vA", "vB"): 31}

    def _snapshot(self, **over):  # type: ignore[no-untyped-def]
        base = {
            "settlements": [],
            "cities": [],
            "roads": [],
            "knights": 0,
            "dev": {"KNIGHT": 1, "MONOPOLY": 1, "VP": 2, "ROADBUILDER": 0, "YEAROFPLENTY": 0},
            "new_dev": 0,
            "cards": 10,
        }
        base.update(over)
        return base

    def test_a_cancelled_click_logs_nothing(self) -> None:
        mod = _load_module()
        snap = self._snapshot()
        assert mod._describe_human_delta(snap, self._snapshot(), self._FakeEnv()) is None

    def test_a_settlement_names_its_vertex(self) -> None:
        mod = _load_module()
        line = mod._describe_human_delta(
            self._snapshot(), self._snapshot(settlements=["vA"]), self._FakeEnv()
        )
        assert line == "You built settlement at v7"

    def test_a_road_names_its_edge(self) -> None:
        mod = _load_module()
        line = mod._describe_human_delta(
            self._snapshot(), self._snapshot(roads=[("vA", "vB")]), self._FakeEnv()
        )
        assert line == "You built road at e31"

    def test_a_knight_play_is_named_once(self) -> None:
        mod = _load_module()
        after = self._snapshot(knights=1, dev={"KNIGHT": 0, "MONOPOLY": 1, "VP": 2})
        line = mod._describe_human_delta(self._snapshot(), after, self._FakeEnv())
        assert line == "You played Knight"

    def test_a_monopoly_play_is_named(self) -> None:
        mod = _load_module()
        after = self._snapshot(dev={"KNIGHT": 1, "MONOPOLY": 0, "VP": 2})
        line = mod._describe_human_delta(self._snapshot(), after, self._FakeEnv())
        assert line == "You played Monopoly"

    def test_a_dev_card_buy_is_named_without_its_type(self) -> None:
        mod = _load_module()
        after = self._snapshot(new_dev=1, cards=7)
        line = mod._describe_human_delta(self._snapshot(), after, self._FakeEnv())
        assert line == "You bought a dev card"

    def test_a_vp_dev_card_buy_is_not_mislabelled_as_a_bank_trade(self) -> None:
        # ``draw_devCard`` applies a VP card IMMEDIATELY: devCards['VP'] goes up
        # and newDevCards does NOT. Roughly one buy in five draws a VP card, so
        # falling through to the bank-trade fallback would make the log lie.
        mod = _load_module()
        after = self._snapshot(dev={"KNIGHT": 1, "MONOPOLY": 1, "VP": 3}, cards=7)
        line = mod._describe_human_delta(self._snapshot(), after, self._FakeEnv())
        assert line == "You bought a dev card"

    def test_a_resource_only_change_reads_as_a_bank_trade(self) -> None:
        mod = _load_module()
        line = mod._describe_human_delta(self._snapshot(), self._snapshot(cards=8), self._FakeEnv())
        assert line == "You traded with the bank"

    def test_a_real_engine_road_resolves_against_the_real_index_map(self) -> None:
        """The hand-made string coords above cannot catch a drift between the
        label's key and ``CatanEnv._edge_key`` — the lookup fails SILENTLY to
        "e?". This exercises real engine ``Point`` objects end to end."""
        from catan_rl.engine.board import catanBoard
        from catan_rl.env.catan_env import CatanEnv

        mod = _load_module()
        board = catanBoard()
        env = CatanEnv.__new__(CatanEnv)
        env._build_index_maps(board)
        v1 = next(iter(board.boardGraph))
        v2 = board.boardGraph[v1].neighbors[0]
        line = mod._describe_human_delta(self._snapshot(), self._snapshot(roads=[(v1, v2)]), env)
        assert line is not None and line.startswith("You built road at e")
        assert "e?" not in line


class TestHudLogIsNotTheRecordersCollector:
    """D5 structural pin (the integration behavioural pin lives in
    ``tests/integration/test_human_recorder_smoke.py``, which ``make test-unit``
    does not run).

    ``EventCollector.drain()`` is destructive and single-consumer and the replay
    recorder already owns the only one. A second consumer would silently strip
    events out of the Replay — the artifact built specifically to be trusted."""

    def test_the_harness_uses_a_bounded_deque_and_never_the_collector(self) -> None:
        import ast

        tree = ast.parse(_SCRIPT.read_text(encoding="utf-8"))
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]
        deques = [
            c
            for c in calls
            if getattr(c.func, "id", None) == "deque" or getattr(c.func, "attr", None) == "deque"
        ]
        assert deques, "the HUD log must be a collections.deque"
        assert all(any(k.arg == "maxlen" for k in c.keywords) for c in deques), "must be bounded"
        # EXACTLY ONE EventCollector exists in the harness — the recorder's. The
        # HUD must neither construct a second one nor share that one.
        collectors = [c for c in calls if getattr(c.func, "id", None) == "EventCollector"]
        assert len(collectors) == 1, "a second EventCollector would race the recorder's drain"

    def test_the_env_holds_no_reference_to_the_hud_log(self) -> None:
        """The log is reached ONLY through the view, which ``__deepcopy__``
        already drops — so an MCTS clone structurally cannot log."""
        pytest.importorskip("torch")
        import copy

        mod = _load_module()
        env = mod._build_human_env_class()(opponent_type="heuristic", max_turns=20)
        env._auto_human = True
        env.reset(seed=5, options={"agent_seat": 0})
        assert not [k for k in vars(env) if "log" in k.lower()]
        clone = copy.deepcopy(env)
        assert clone._human_view is None
        clone._log_move("this must go nowhere")  # no view -> no log, no error


class TestFinalScreenHold:
    """``_hold_final_screen`` must never block a headless run.

    It is a blocking modal on the exit path of EVERY game, so the property that
    actually matters is the early return: with no pygame display surface (CI,
    ``--self-test``, any automated run) it must return immediately rather than
    sit for ``FINAL_SCREEN_HOLD_S``. The interactive dismissal paths (click, any
    key, window close) need a real display and are exercised by playing.
    """

    def test_returns_immediately_with_no_display_surface(self) -> None:
        import time

        pygame = pytest.importorskip("pygame")
        mod = _load_module()
        # Guard the guard: if a surface leaked in from another test, this would
        # block for two minutes instead of asserting anything.
        if pygame.get_init() and pygame.display.get_surface() is not None:
            pygame.display.quit()
        started = time.monotonic()
        mod._hold_final_screen()
        assert time.monotonic() - started < 1.0

    def test_timeout_is_a_liveness_guard_not_a_wait(self) -> None:
        """The hold is dismissible; the timeout only bounds an unattended run."""
        mod = _load_module()
        assert mod.FINAL_SCREEN_HOLD_S > 0


def _first_true(mask) -> int:  # type: ignore[no-untyped-def]
    idx = int(np.argmax(mask))
    assert bool(mask[idx]), "no legal index in mask"
    return idx


def _first_legal_action(env):  # type: ignore[no-untyped-def]
    """First-legal action for ``env`` (mirrors tests/unit/env/test_snapshot_opponent)."""
    from catan_rl.env.catan_env import ActionType

    m = env.get_action_masks()
    action = [0, 0, 0, 0, 0, 0]
    if bool(m["type"][ActionType.END_TURN]):
        return np.asarray([ActionType.END_TURN, 0, 0, 0, 0, 0], dtype=np.int64)
    atype = _first_true(m["type"])
    action[0] = atype
    if atype == ActionType.BUILD_SETTLEMENT:
        action[1] = _first_true(m["corner_settlement"])
    elif atype == ActionType.BUILD_CITY:
        action[1] = _first_true(m["corner_city"])
    elif atype == ActionType.BUILD_ROAD:
        action[2] = _first_true(m["edge"])
    elif atype == ActionType.MOVE_ROBBER:
        action[3] = _first_true(m["tile"])
    elif atype == ActionType.DISCARD:
        action[4] = _first_true(m["resource1_discard"])
    return np.asarray(action, dtype=np.int64)


class TestHumanSeatIsNotAI:
    """The human seat is flagged ``isAI=False`` PERMANENTLY, and search is unaffected.

    ``CatanEnv.reset`` stamps ``opp.isAI = True``, which routed a human Monopoly
    / Year of Plenty through ``np.random.choice`` instead of the GUI picker. The
    harness flips it back on every reset. The flip is only safe because no
    deep-copied MCTS clone can reach the branch it changes: ``play_devCard`` has
    exactly two call sites (the engine's ``playCatan`` human loop — unreachable
    because the env builds ``render_mode=None`` — plus this script's single GUI
    site, the main-turn PLAY DEV button; the pre-roll one was removed with the
    pre-roll window), so a clone rollout never executes it. The pin for that is
    ``test_play_devcard_has_only_the_two_audited_call_sites``: unreachability
    is a STRUCTURAL property, and a runtime sentinel dropped into a scripted
    clone rollout could only ever re-confirm it (nothing under ``src/`` calls the
    method, so the sentinel is unreachable by construction regardless of the
    ``isAI`` value) — it would pass identically on a tree where the flip was
    wrong, so it is not written here.
    """

    def _env(self):  # type: ignore[no-untyped-def]
        pytest.importorskip("torch")
        mod = _load_module()
        env = mod._build_human_env_class()(opponent_type="heuristic", max_turns=50)
        env.reset(seed=11, options={"agent_seat": 0})
        return env

    def test_the_human_seat_is_not_flagged_as_ai(self) -> None:
        env = self._env()
        assert env.opponent_player.isAI is False
        assert env.agent_player.isAI is False

    def test_the_flip_survives_a_second_reset(self) -> None:
        env = self._env()
        env.reset(seed=12, options={"agent_seat": 1})
        assert env.opponent_player.isAI is False

    def test_a_clone_never_drives_the_gui(self) -> None:
        import copy

        env = self._env()
        env.set_view_factory(lambda game: object())
        clone = copy.deepcopy(env)
        assert clone._human_view is None
        assert clone._view_factory is None
        assert clone._use_gui() is False

    def test_play_devcard_has_only_the_two_audited_call_sites(self) -> None:
        out = subprocess.run(
            ["git", "grep", "--untracked", "-n", r"\.play_devCard(", "--", "src", "scripts"],
            cwd=_REPO,
            capture_output=True,
            text=True,
            check=False,
        ).stdout.split("\n")
        sites = sorted(line.split(":")[0] for line in out if line.strip())
        assert sites == [
            "scripts/play_vs_model.py",
            "src/catan_rl/engine/game.py",
        ], sites


class TestNoPreRollWindow:
    """Neither seat may play a dev card before rolling (fairness stopgap).

    ``env/masks.py`` returns ONLY ``ROLL_DICE`` while ``roll_pending``, and
    ``env/catan_env.py`` no-ops anything else, so the POLICY structurally cannot
    play a pre-roll Knight — not here and not anywhere in its training history.
    The harness used to give the human that window anyway, which makes a human
    win unattributable between "the policy is weak here" and "the bot was not
    allowed to play its Knight". Both pins below are load-bearing: the first
    proves the human window is gone, the second proves the bot-side asymmetry
    the removal was justified by is real (delete it and the pin becomes an
    unexamined claim in a docstring). See ``preroll-dev-cards-r1.md``.
    """

    def test_the_human_pre_roll_window_is_gone(self) -> None:
        src = _SCRIPT.read_text(encoding="utf-8")
        assert "_human_pre_roll" not in src, "the pre-roll window (or a call to it) survives"
        assert "[PRE-ROLL]" not in src

    def test_the_auto_played_human_seat_never_pre_rolls(self) -> None:
        """The seat the bot's SEARCH models must match the seat at the keyboard.

        The GUI branch has no pre-roll window, but the headless branch delegates
        to the base env, which plays ``heuristic_pre_roll`` for the opponent
        whenever no snapshot drives it — i.e. headless auto-play such as
        ``--self-test``. Left alone the auto-played human seat pre-rolls a Knight
        while the real human at the keyboard cannot, which is the asymmetry D3
        removed.

        SCOPE: this is NOT about MCTS clones. A clone always carries a snapshot
        opponent (``search/mcts.py`` clone_env → ``set_snapshot_opponent``), and
        ``env/catan_env.py`` gates the pre-roll on ``_snapshot_opponent is None``,
        so it cannot fire there."""
        pytest.importorskip("torch")
        mod = _load_module()
        env = mod._build_human_env_class()(opponent_type="heuristic", max_turns=60)
        env.reset(seed=21, options={"agent_seat": 0})
        assert not env._use_gui()
        calls = [0]
        real = type(env.opponent_player).heuristic_pre_roll

        def counting(self, board):  # type: ignore[no-untyped-def]
            calls[0] += 1
            return real(self, board)

        type(env.opponent_player).heuristic_pre_roll = counting
        try:
            for _ in range(400):
                _o, _r, term, trunc, _i = env.step(_first_legal_action(env))
                if term or trunc:
                    break
        finally:
            type(env.opponent_player).heuristic_pre_roll = real
        assert calls[0] == 0, f"{calls[0]} pre-roll windows on the human seat"

    def test_the_record_marks_the_game_post_roll_only(self) -> None:
        assert '"preroll": False' in _SCRIPT.read_text(encoding="utf-8")

    def test_the_policy_may_only_roll_while_roll_pending(self) -> None:
        pytest.importorskip("torch")
        from catan_rl.env.catan_env import ActionType

        mod = _load_module()
        env = mod._build_human_env_class()(opponent_type="heuristic", max_turns=50)
        env.reset(seed=21, options={"agent_seat": 0})
        # Play out the snake draft until the bot faces its first roll.
        for _ in range(200):
            if env.roll_pending:
                break
            env.step(_first_legal_action(env))
        assert env.roll_pending, "never reached a roll-pending state"
        legal = [i for i, ok in enumerate(env.get_action_masks()["type"]) if ok]
        assert legal == [int(ActionType.ROLL_DICE)], legal


class TestBankConservationGuard:
    """D3: the human driver asserts the finite-bank invariant it never had.

    The guard is the only thing standing between a GUI accounting slip and a
    corrupted global feature in the bot's observation, so its call sites are
    pinned at the source the same way the move-log ones are.
    """

    def test_the_helper_holds_on_a_fresh_game(self) -> None:
        pytest.importorskip("torch")
        mod = _load_module()
        env = mod._build_human_env_class()(opponent_type="heuristic", max_turns=50)
        env.reset(seed=13, options={"agent_seat": 0})
        mod._assert_bank_conservation(env)  # setup grants are metered
        for _ in range(30):
            _o, _r, term, trunc, _i = env.step(_first_legal_action(env))
            mod._assert_bank_conservation(env)
            if term or trunc:
                break

    def test_a_broken_bank_is_caught(self) -> None:
        pytest.importorskip("torch")
        mod = _load_module()
        env = mod._build_human_env_class()(opponent_type="heuristic", max_turns=50)
        env.reset(seed=14, options={"agent_seat": 0})
        env.opponent_player.resources["WOOD"] += 1  # a mint, exactly as the picker did
        with pytest.raises(AssertionError):
            mod._assert_bank_conservation(env)

    def test_the_driver_calls_the_guard_after_every_step(self) -> None:
        """The guard must be a STATEMENT inside every env-stepping loop.

        This was a text grep (``src.count("check_bank(env)") >= 2``), which passes
        if both calls sit in dead code, in a docstring, or before the loop rather
        than inside it — for the one check the spec calls the single thing that
        catches both the original bug and the minting bug. Parse instead.
        """
        import ast

        tree = ast.parse(_SCRIPT.read_text(encoding="utf-8"))
        guards = {"check_bank", "_assert_bank_conservation"}

        def _calls_guard(node: ast.AST) -> bool:
            return any(
                isinstance(n, ast.Expr)
                and isinstance(n.value, ast.Call)
                and isinstance(n.value.func, ast.Name)
                and n.value.func.id in guards
                for n in ast.walk(node)
            )

        def _steps_env(node: ast.AST) -> bool:
            return any(
                isinstance(n, ast.Call)
                and isinstance(n.func, ast.Attribute)
                and n.func.attr == "step"
                for n in ast.walk(node)
            )

        stepping = [n for n in ast.walk(tree) if isinstance(n, ast.While) and _steps_env(n)]
        assert stepping, "no env-stepping loop found — this pin is looking at nothing"
        unguarded = [ln.lineno for ln in stepping if not _calls_guard(ln)]
        assert not unguarded, (
            f"env-stepping loop(s) at line(s) {unguarded} step the env without "
            "calling the bank-conservation guard inside the loop body"
        )

    def test_a_broken_bank_is_recorded_not_just_warned(self) -> None:
        """A corrupt game must be MARKED in the ledger, not only printed.

        The reporter prints once and appends to a ``deque(maxlen=6)`` that evicts
        the warning within six log lines, so without a persisted key a
        conservation-broken game is byte-indistinguishable from a clean one and
        can reach the human scoreboard — the pre-mortem this feature exists to
        prevent, one step further down the pipe.
        """
        from collections import deque

        pytest.importorskip("torch")
        mod = _load_module()
        hud: deque[str] = deque(maxlen=6)
        check = mod._make_bank_conservation_reporter(hud)

        env = mod._build_human_env_class()(opponent_type="heuristic", max_turns=50)
        env.reset(seed=14, options={"agent_seat": 0})

        check(env)
        assert check.ok[0] is True, "a clean game must not be marked corrupt"

        env.opponent_player.resources["WOOD"] += 1  # a mint, exactly as the picker did
        check(env)
        assert check.ok[0] is False, "the reporter must expose its tripped state"

        # And the warning alone is not enough: the HUD strip evicts it.
        assert hud.maxlen is not None and hud.maxlen <= 8
        assert '"bank_ok"' in _SCRIPT.read_text(encoding="utf-8"), (
            "the ledger record must persist a bank_ok marker"
        )

    def test_the_interactive_reporter_does_not_kill_the_game(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A broken bank must NOT raise out of the interactive loop: the replay is
        written only after the loop ends, so raising would destroy an hour of human
        play AND the recording that is the evidence of the break. Report, detach,
        play on — the same side-channel rule the recorder already follows."""
        pytest.importorskip("torch")
        mod = _load_module()
        env = mod._build_human_env_class()(opponent_type="heuristic", max_turns=50)
        env.reset(seed=15, options={"agent_seat": 0})
        hud_log: list[str] = []
        check = mod._make_bank_conservation_reporter(hud_log)

        check(env)  # healthy: silent, nothing logged
        assert hud_log == []

        env.opponent_player.resources["WOOD"] += 1  # a mint
        check(env)  # must not raise
        assert len(hud_log) == 1 and "BANK INVARIANT BROKEN" in hud_log[0]
        assert "finite-bank invariant BROKEN" in capsys.readouterr().out

        check(env)  # detached after the first report — no repeat spam
        assert len(hud_log) == 1


class TestCrashSafeArtifacts:
    """A misclick must not destroy an hour of play.

    ``env.step`` sat in a bare ``while`` loop with BOTH artifact writes after it,
    and five GUI paths call ``sys.exit(0)`` on ``pygame.QUIT``. Closing the
    window mid-game therefore discarded the entire game with no partial write.
    ``SystemExit`` is the realistic interruption AND a ``BaseException``, so it
    is what is injected here — an ``except Exception`` would let it through and
    this pin would go red, which is the point.

    Those quit paths call ``pygame.quit()`` BEFORE ``sys.exit(0)``, so the
    ``tear_down_pygame`` variant is the load-bearing one: catching the
    ``SystemExit`` is not enough if anything drawn afterwards raises on the dead
    display and takes the two writes with it.
    """

    def _run(self, tmp_path, monkeypatch, *, exc, ckpt="unused", tear_down_pygame=False):  # type: ignore[no-untyped-def]
        import json

        pytest.importorskip("torch")
        mod = _load_module()

        class _StubAgent:
            last_internals = None

            def choose_action(self, env):  # type: ignore[no-untyped-def]
                return _first_legal_action(env)

        monkeypatch.setattr(mod, "_load_raw_agent", lambda ckpt, seed: _StubAgent())
        # The hold needs a real display; it is not what is under test.
        monkeypatch.setattr(mod, "_hold_final_screen", lambda: None)

        real_build = mod._build_human_env_class

        def _build_exploding_env_class():  # type: ignore[no-untyped-def]
            cls = real_build()
            n = [0]
            base_step = cls.step

            def step(self, action):  # type: ignore[no-untyped-def]
                n[0] += 1
                if n[0] > 3:
                    if tear_down_pygame:
                        # The REAL window-close path: every GUI quit handler
                        # calls pygame.quit() and THEN sys.exit(0), so anything
                        # drawn after the salvage catch hits a dead display.
                        import pygame

                        pygame.quit()
                    raise exc
                return base_step(self, action)

            cls.step = step
            # Auto-play the human seat: the real GUI windows block on mouse
            # input, which is not what this pin is about. This is the env's own
            # headless branch (MCTS clones + ``--self-test`` take it too).
            cls._use_gui = lambda self: False  # type: ignore[method-assign]
            return cls

        monkeypatch.setattr(mod, "_build_human_env_class", _build_exploding_env_class)

        log_path = tmp_path / "games.jsonl"
        # human_seat=1 -> the BOT drafts first, so reset() needs no human input.
        mod.play_interactive(
            ckpt=ckpt,
            sims=1,
            seed=5,
            human_seat=1,
            log_path=str(log_path),
            replay_dir=str(tmp_path),
        )
        record = json.loads(log_path.read_text(encoding="utf-8").splitlines()[-1])
        replays = sorted(tmp_path.glob("*.json"))
        assert len(replays) == 1, replays
        return record, json.loads(replays[0].read_text(encoding="utf-8"))

    def test_a_mid_game_systemexit_still_writes_both_artifacts(
        self,
        tmp_path,
        monkeypatch,  # type: ignore[no-untyped-def]
    ) -> None:
        record, replay = self._run(tmp_path, monkeypatch, exc=SystemExit(0))
        assert record["partial"] is True
        assert replay["metadata"]["partial"] is True
        assert record["winner"] is None

    def test_a_real_window_close_still_writes_both_artifacts(
        self,
        tmp_path,
        monkeypatch,  # type: ignore[no-untyped-def]
    ) -> None:
        """The scenario the salvage exists for: ``pygame.quit()`` then ``sys.exit(0)``.

        A bare ``SystemExit`` never touches pygame, so it passes even on a tree
        where the post-loop ``displayGameScreen()`` raises
        ``pygame.error: display Surface quit`` and takes BOTH writes with it.
        This variant tears the display down first, exactly like
        ``_human_interactive_main_turn``'s QUIT handler."""
        import pygame

        try:
            record, replay = self._run(
                tmp_path, monkeypatch, exc=SystemExit(0), tear_down_pygame=True
            )
        finally:
            # In production this path ends in sys.exit; in-process it would
            # leave pygame torn down for every later test.
            pygame.init()
        assert record["partial"] is True
        assert replay["metadata"]["partial"] is True

    def test_a_keyboardinterrupt_is_salvaged_too(
        self,
        tmp_path,
        monkeypatch,  # type: ignore[no-untyped-def]
    ) -> None:
        record, replay = self._run(tmp_path, monkeypatch, exc=KeyboardInterrupt())
        assert record["partial"] is True
        assert replay["metadata"]["partial"] is True

    def test_the_except_clause_is_baseexception(self) -> None:
        """``except Exception`` would let every ``sys.exit(0)`` path through."""
        import ast

        tree = ast.parse(_SCRIPT.read_text(encoding="utf-8"))
        fn = next(
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name == "play_interactive"
        )
        handlers = [
            h.type.id
            for n in ast.walk(fn)
            for h in getattr(n, "handlers", [])
            if isinstance(h.type, ast.Name)
        ]
        assert "BaseException" in handlers, handlers


class TestProvenanceKeys:
    """The record must name the exact policy and code it was played against.

    Metadata carried ``ckpt_path`` but no git SHA and no checkpoint content
    hash, and ``ckpt_path`` points into a gitignored ``runs/`` directory under a
    ``keep_last_n: 6`` rotation — so tomorrow it may hold different bytes, or
    nothing. Both keys are ADDITIVE and degrade to ``None``: provenance must
    never be able to stop a game from being played.
    """

    def test_the_git_sha_is_this_tree(self) -> None:
        mod = _load_module()
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=_REPO, capture_output=True, text=True, check=False
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=_REPO,
                capture_output=True,
                text=True,
                check=False,
            ).stdout.strip()
        )
        # A SHA with uncommitted edits on top does not describe the code the
        # game was played on, so it must say so.
        assert mod._git_sha() == (f"{head}-dirty" if dirty else head)

    def test_the_checkpoint_hash_is_the_file_content(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        import hashlib

        mod = _load_module()
        blob = b"not-really-a-checkpoint" * 1000
        f = tmp_path / "ckpt.pt"
        f.write_bytes(blob)
        assert mod._ckpt_sha256(str(f)) == hashlib.sha256(blob).hexdigest()

    def test_an_unreadable_checkpoint_degrades_to_none(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        mod = _load_module()
        assert mod._ckpt_sha256(str(tmp_path / "nope.pt")) is None
        assert mod._ckpt_sha256(None) is None
        assert mod._ckpt_sha256("") is None

    def test_both_keys_reach_both_artifacts(self, tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
        ckpt = tmp_path / "ckpt.pt"
        ckpt.write_bytes(b"bytes")
        record, replay = TestCrashSafeArtifacts()._run(
            tmp_path, monkeypatch, exc=SystemExit(0), ckpt=str(ckpt)
        )
        expected = _load_module()._ckpt_sha256(str(ckpt))
        for got in (record, replay["metadata"]):
            sha = got["git_sha"]
            assert sha is not None
            assert len(sha.removesuffix("-dirty")) == 40, sha
            assert got["ckpt_sha256"] == expected


class TestInteractiveRulesAudit:
    """``run_all_invariants`` runs on every PLAYED game, and only RECORDS.

    It used to run solely under ``--self-test``, consumed as
    ``assert not violations`` — which is exactly why a GUI-only rule-gap class
    (stranded dev-card counters, a self-targetable robber) survived: no game a
    human actually played was ever audited.

    Two properties are load-bearing and pinned separately:

    * it must NOT assert. The driver's whole contract is that both artifacts are
      written even on abort, and an assert here would destroy the very record it
      exists to validate; and
    * an ABORTED game must be passed through as ``truncated``. Otherwise
      ``check_terminal_state`` reports "terminated but no player reached 15 VP"
      for every window-close, and an alarm meaning "you closed the window" is one
      the operator learns to ignore within two sessions.
    """

    def test_an_aborted_game_records_a_clean_audit(self, tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
        record, _replay = TestCrashSafeArtifacts()._run(tmp_path, monkeypatch, exc=SystemExit(0))
        assert record["partial"] is True
        assert record["winner"] is None
        assert record["rules_violations"] == [], record["rules_violations"]
        assert record["rules_ok"] is True, (
            "a window-close produced a spurious violation; the abort state is "
            "not reaching check_terminal_state as `truncated`"
        )

    def test_the_abort_state_is_passed_through_to_the_audit(self) -> None:
        src = _SCRIPT.read_text(encoding="utf-8")
        assert "run_all_invariants(" in src
        assert "truncated or aborted" in src, "the audit must treat an aborted game as truncated"
        assert "assert not violations" not in src.split("--self-test")[0]

    def test_the_regime_stamps_are_bumped(self, tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
        """``hud`` is what the human could SEE; ``rules_epoch`` is what they were
        ALLOWED TO DO. Both changed in this slice, and they are separate keys on
        purpose — a later reader has to be able to screen games played under the
        old (self-robbable, counter-leaking) rules."""
        record, _replay = TestCrashSafeArtifacts()._run(tmp_path, monkeypatch, exc=SystemExit(0))
        assert record["hud"] == 3
        assert record["rules_epoch"] == 2
