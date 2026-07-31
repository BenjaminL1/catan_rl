#!/usr/bin/env python3
"""Play 1v1 Catan (Colonist.io ruleset) as a HUMAN against the CHAMPION.

The bot defaults to the pointer-arch champion (``selfplay_pointer_arch_v2``
``ckpt_000000500``) playing its RAW POLICY — no search. You play through the
existing pygame board GUI (mouse + the on-screen buttons).

**Why raw policy is the default (correctness, not taste).** The deployed
determinized search is CLAIRVOYANT against a human: ``mcts.clone_env`` deep-copies
the live env and ``MCTS._reseed`` reseeds only the opponent model + numpy/stdlib
streams, leaving the Rust ``StackedDice`` bag deep-copied and untouched. Every
simulated world therefore shares the opponent's TRUE hidden dev cards and the TRUE
future dice. In a bot-vs-bot harness both sides sit in that same env so it only
inflates the number; against a HUMAN the tree is literally reading your hand and
your next rolls. ``--search`` still exists for bot-vs-bot work and prints a loud
warning, but any human game played with it is uninterpretable.

**The bot's hand is HIDDEN by default.** Its panel shows hand SIZE, unplayed
dev-card COUNT, VISIBLE VP (``victoryPoints - devCards["VP"]``), KNIGHTS PLAYED
and current LONGEST-ROAD length — real Catan visibility. The last two are
PUBLIC facts (knights are played face-up, roads are on the board), so
withholding them would bias the playtest toward the bot. ``--reveal-bot``
restores the omniscient analysis view and is recorded in BOTH logs, because a
revealed game is not a strength read.

Each finished game is written TWICE (dual-write, deliberately):

* the legacy four-field JSON line in ``runs/human_playtest/games.jsonl`` (its
  ``bot_vp`` / ``human_vp`` keys stay TOTAL VP and are never re-pointed), plus
  the ``reveal_bot`` / ``replay_path`` keys and ``"hud": 2`` — the information
  regime the game was played under. A MISSING ``hud`` key means regime 1 (no
  knights/longest-road HUD, no on-screen move log), the same read-a-missing-key
  convention as ``reveal_bot`` below. It also carries ``"bank_ok"`` (did the
  ``bank[R] + Σ hands[R] == 19`` conservation assert hold at every step),
  ``"preroll": false`` (the pre-roll information regime — see below),
  ``"partial"`` (the game was cut short), and the ``"git_sha"`` /
  ``"ckpt_sha256"`` attribution pair; and
* a full-fidelity ``Replay`` JSON under ``runs/human_playtest/replays/``,
  carrying BOTH players' move streams and the policy's per-decision internals,
  openable in the existing ``replay/viewer``. Its ``Metadata`` carries the same
  ``partial`` / ``git_sha`` / ``ckpt_sha256`` provenance.

BOTH artifacts are written even when the game is INTERRUPTED (window close,
``KeyboardInterrupt``, crash), marked ``partial`` — a cut-short game is a valid
trace of the moves it contains but is NOT an outcome/strength read.

**Both seats are POST-ROLL ONLY.** The human gets no pre-roll dev-card window,
because ``env/masks.py`` emits only ``ROLL_DICE`` while ``roll_pending`` — the
policy structurally cannot use that window, so granting it to the human alone
would be an asymmetry, not a courtesy. The auto-played human seat
(MCTS clones, ``--self-test``) likewise has the base env's
``heuristic_pre_roll`` suppressed, so the bot's search models the seat the human
actually plays. Games record ``"preroll": false``. This is a stopgap until
``preroll-dev-cards-r1`` gives BOTH seats the window (which needs a retrain).

FIDELITY CAVEATS on that replay (do not discover these later):

* The four SETUP steps are **SYNTHESIZED, not observed** — ``setup_steps_seat_0/1``
  reconstruct the placements from action tuples and hardcode
  ``longest_road_holder=None``. The OPENING is the least faithful part of the
  record, which is exactly the phase under suspicion.
* ``state_after`` is **shared** across every sub-step produced by one
  ``env.step`` (the human's whole turn is folded inside the bot's step). Actions
  and events ARE attributed per acting seat; only the board snapshot is shared.
* The recorded per-decision softmax reads roughly ``0.97 END_TURN`` at most
  steps. It is a hypothesis generator for contested decisions, **never a
  verdict** — and it is the policy grading its own homework.
* The binding constraint on this harness is **games played**, not bytes per
  game. One recorded game is not evidence of anything.
* Every line already in ``games.jsonl`` predates the blind default and was
  therefore played with the bot's FULL hand on screen. Those lines carry no
  ``reveal_bot`` key; treat a missing key as ``reveal_bot=true``, not as a
  clean result. They are not rewritten — see the dual-write note above.

Run it (a display is required for the real game)::

    python scripts/play_vs_model.py                 # champion, raw policy, logged

Headless smoke (no display, no pygame window — auto-plays a legal human move so
the full turn flow + win detection are exercised end-to-end)::

    python scripts/play_vs_model.py --self-test

ARCHITECTURE (why this shape)
-----------------------------
``CatanEnv`` is agent-centric: the "agent" is one fixed seat and ``env.step``
folds the OTHER seat's whole turn internally (heuristic / snapshot driver).
``SearchAgent.choose_action(env)`` drives the AGENT seat (it deep-copies the env
inside MCTS). So we make the BOT the agent seat (clean reuse of the search loop
from ``eval_search._play_search_game``) and make the HUMAN the env's internal
opponent by overriding the four ``_opponent_*`` hooks.

The env's game MUST stay headless: MCTS does ``copy.deepcopy(env.game)`` and a
real pygame ``catanGameView`` holds unpicklable surfaces (``catanGame.copy``
rejects a non-headless view). So we keep a SEPARATE pygame view that is never
stored on ``game`` (never deepcopied), and temporarily swap ``game.boardView`` to
it only for the duration of the human's input window — letting the existing,
battle-tested engine human-input code paths (``game.build`` / ``game.robber`` /
``player.play_devCard`` / ``discardResources`` / ``initiate_trade``) run verbatim
— then swap the headless view back before returning control to the bot/MCTS.

What the existing GUI human-input covers (no stubs needed): roll dice, build
road / settlement / city, buy dev card, play dev card (Knight/RoadBuilder/YoP/
Monopoly via menus), bank/port trade, end turn, robber move + steal on a 7, and
the 9-card discard menu. The full game is playable through the existing GUI.

Those menus route through the FINITE resource bank (spec 009): the human's
Year-of-Plenty picks ``bank_draw`` (and an unsuppliable resource is greyed out
and unclickable, matching the AI branch's availability gate), discards
``bank_recirculate``, and cancelling a partly-picked YoP puts back what it drew.
The driver asserts ``board.assert_conservation`` after every step and records
the result as ``bank_ok``.
"""

from __future__ import annotations

import argparse
import dataclasses
import secrets
import sys
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, cast

import numpy as np

if TYPE_CHECKING:
    from catan_rl.search.agent import SearchAgent

#: The CURRENT champion (pointer-arch lineage). The old v8 anchor this script was
#: written for no longer loads on main — the pointer-arch fork changed both the
#: policy shape and the obs schema, so ``build_actor`` raises on a v8/v11 file.
#:
#: BANKED, not live: this used to point straight at
#: ``runs/train/selfplay_pointer_arch_v2/checkpoints/ckpt_000000500.pt``, inside a
#: run directory under ``keep_last_n: 6`` rotation — so the champion every human
#: game was scored against could be deleted out from under the path by the next
#: six saves. ``runs/`` is gitignored, so banking the copy (``bank_anchor``,
#: policy-only slim) is an OPS step; only this repoint is tracked.
DEFAULT_CKPT = "runs/anchors/ptr_v1_u500.pt"
DEFAULT_SIMS = 400


def _git_sha() -> str | None:
    """``git rev-parse HEAD`` for the tree this game is being played on.

    A recorded game that names no code is not attributable: the ruleset, the obs
    schema and this harness all move between games. Degrades to ``None`` rather
    than raising — provenance must never stop a game from being played, and a
    consumer reads ``None`` as "unattributed", not "clean".

    A tree with uncommitted edits gets a ``-dirty`` suffix (same convention as
    ``bench_engine._git_sha``): the SHA alone would name code the game was NOT
    played on, which is exactly the mis-attribution this key exists to prevent.
    The checkpoint side is content-hashed for the same reason.
    """
    import subprocess
    from pathlib import Path

    root = str(Path(__file__).resolve().parent.parent)
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )
        sha = out.stdout.strip()
        if not sha:
            return None
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None
    return f"{sha}-dirty" if status.stdout.strip() else sha


def _ckpt_sha256(path: str | None) -> str | None:
    """SHA-256 of the checkpoint FILE, or ``None`` if it cannot be read.

    The path alone is not attribution: the default points into a gitignored
    ``runs/`` tree under a ``keep_last_n`` rotation, so tomorrow the same path
    may hold different bytes or nothing at all. Streamed in 1 MiB chunks — these
    files are tens of megabytes.
    """
    if not path:
        return None
    import hashlib

    h = hashlib.sha256()
    try:
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
    except OSError:
        return None
    return h.hexdigest()


def _assert_bank_conservation(env: Any) -> None:
    """Pin the finite-bank invariant ``bank[R] + sum(hands[R]) == 19`` (spec 009).

    ALWAYS ON: it is a 5-key sum, and this driver — the one a human actually
    watches, and the one whose games feed the human scoreboard — is the only
    driver that never had a bank guard. Both the GUI discard leak and the GUI
    Year-of-Plenty mint lived here undetected for exactly that reason, and the
    bank is a feature of the bot's observation, so a broken bank is a corrupted
    policy input rather than a cosmetic accounting slip.
    """
    game = getattr(env, "game", None)
    if game is None:
        return
    game.board.assert_conservation(list(game.playerQueue.queue))


def _make_bank_conservation_reporter(hud_log: Any) -> Callable[[Any], None]:
    """Non-fatal front end to :func:`_assert_bank_conservation` for the human loop.

    The check itself raises, which is what the tests and the self-test want. In
    the INTERACTIVE loop it must not: the replay is written only after the loop
    ends, so raising here would both kill an hour of human play and destroy the
    recording that is the evidence of the break. Same side-channel rule the
    recorder already follows — report loudly (console + HUD strip), once, then
    detach and let the game finish so the corrupted record can be inspected.

    The returned callable carries ``.ok`` — a live flag the caller MUST write into
    the game record. Warning alone is not enough: the console line is a single
    flush the human may miss, and the HUD strip is a ``deque(maxlen=6)`` that
    evicts the warning within six log lines. Without a persisted marker a
    conservation-broken game is byte-indistinguishable from a clean one and can
    reach the human scoreboard — which is the pre-mortem this whole feature was
    written to prevent, one step further down the pipe.
    """
    live = [True]

    def _check(env: Any) -> None:
        if not live[0]:
            return
        try:
            _assert_bank_conservation(env)
        except AssertionError as exc:
            live[0] = False
            print(
                f"[ERROR] finite-bank invariant BROKEN: {exc}\n"
                "        The bank is in the bot's observation, so this game is "
                "CORRUPT — finish or quit, but do not score it. Further bank "
                "checks are disabled for this game.",
                flush=True,
            )
            hud_log.append("BANK INVARIANT BROKEN - game is corrupt (see console)")

    _check.ok = live  # type: ignore[attr-defined]
    return _check


# ---------------------------------------------------------------------------
# Human-as-opponent env: the bot is the agent seat; the human is the internal
# opponent, driven via a real pygame view temporarily swapped onto game.boardView.
# ---------------------------------------------------------------------------


def _no_pre_roll(board: Any) -> None:
    """Replacement for ``heuristicAIPlayer.heuristic_pre_roll`` on the human seat.

    Bound onto the human player instance so the base env's pre-roll call
    (``catan_env._run_opponent_turn``) is a no-op: the human at the keyboard has
    no pre-roll window, so neither may the auto-played stand-in the bot's search
    (and ``--self-test``) uses to model that seat."""
    return None


def _build_human_env_class() -> type:
    """Build the ``HumanVsBotEnv`` subclass lazily.

    Defined inside a function so importing this module (e.g. for ``--self-test``,
    or by a test) never forces the ``CatanEnv`` import at module load. The class
    overrides ONLY the four internal opponent hooks; everything else (setup snake
    draft, roll/robber/discard sub-phases, turn folding, obs/masks, the
    engine/RESOURCES_CW mapping) is inherited unchanged.
    """
    from catan_rl.engine.game import _HeadlessView
    from catan_rl.engine.player import player as PlainPlayer
    from catan_rl.env.catan_env import CatanEnv

    class HumanVsBotEnv(CatanEnv):
        """``CatanEnv`` whose internal opponent seat is played by a HUMAN.

        ``self._human_view`` is a live pygame ``catanGameView`` (or ``None`` in
        headless self-test, where the hooks auto-pick a legal move). It is NEVER
        assigned to ``game.boardView`` outside a human window, so the env's game
        stays deep-copy-safe for MCTS.
        """

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self._human_view: Any = None
            # Lazily builds a pygame view on first human-input need (interactive
            # mode). None in headless self-test -> hooks auto-pick a legal action.
            self._view_factory: Any = None
            # In self-test we have no GUI; auto-pick a legal action for the human.
            self._auto_human: bool = False
            # Analysis opt-in: show the BOT's full hand in the GUI panel + the
            # console line. OFF by default — see ``--reveal-bot``.
            self._reveal_bot: bool = False

        def reset(self, *args: Any, **kwargs: Any) -> Any:
            """Reset, then mark the opponent seat as HUMAN-driven.

            ``CatanEnv.reset`` stamps ``opp.isAI = True`` on every reset, which
            sends ``player.play_devCard`` down the ``np.random.choice`` branch —
            so a human Monopoly / Year of Plenty picked its OWN resources at
            random instead of opening the picker that already exists.

            The flip is PERMANENT rather than saved/restored around each
            ``play_devCard`` call: a restore that is ever skipped (exception,
            nested view window) would leave ``isAI=False`` inside the bot's
            deepcopied MCTS clones, where ``_HeadlessView.__getattr__`` returns
            ``None`` and the dev card is silently refunded — i.e. the bot would
            search against an opponent who cannot play YoP or Monopoly, with no
            log line. The blast radius of the permanent flip is smaller and was
            enumerated: ``play_devCard`` has only two call sites (the engine's
            ``playCatan`` human loop, unreachable here because the env builds
            with ``render_mode=None``, plus this script's single GUI site), and
            every other ``isAI`` reader is either behind ``render_mode ==
            'human'`` or behind ``view.human_player is None``, which this
            harness always sets.
            """
            out = super().reset(*args, **kwargs)
            assert self.opponent_player is not None
            self.opponent_player.isAI = False
            return out

        def attach_human_view(self, view: Any) -> None:
            self._human_view = view

        def set_view_factory(self, factory: Any) -> None:
            """Register a ``game -> catanGameView`` builder for interactive play.

            The view is built on FIRST human-input need rather than eagerly, so the
            very-first snake-draft placement — which the env runs inside ``reset()``
            when the human drafts first (``catan_env.py`` agent_seat==1 path) — is
            still driven by the GUI instead of being auto-placed. The board already
            exists by then, so the view can be constructed.
            """
            self._view_factory = factory

        def _ensure_view(self) -> Any:
            """Return the human view, building it via the factory on first need."""
            if self._human_view is None and self._view_factory is not None:
                self._human_view = self._view_factory(self.game)
                human: Any = self.opponent_player
                bot: Any = self.agent_player
                # The human sits in the opponent seat — show THEIR hand always.
                self._human_view.human_player = human
                # The bot sits in the agent seat. Its panel is BLIND by default —
                # hand SIZE, unplayed dev-card COUNT and VISIBLE VP only (real
                # Catan visibility). ``--reveal-bot`` flips it to the omniscient
                # analysis view; view.hand_panel_lines owns that distinction.
                self._human_view.bot_player = bot
                self._human_view.reveal_bot = self._reveal_bot
                # Friendly names in the stats panel + broadcast banner.
                self._human_view.name_display = {human.name: "You", bot.name: "Bot"}
            return self._human_view

        def _use_gui(self) -> bool:
            """True iff this turn is driven by human GUI input (else auto-play)."""
            return not self._auto_human and self._ensure_view() is not None

        def __deepcopy__(self, memo: dict[int, Any]) -> Any:
            # mcts.clone_env deep-copies the WHOLE env. The live pygame view holds
            # unpicklable Surfaces AND must not drive input inside a search clone, so
            # clones carry NEITHER the view NOR the view factory: both -> None makes
            # the _opponent_* hooks auto-play (exactly as in headless self-test).
            # Without nulling the factory, a clone's _ensure_view() would rebuild a
            # real pygame view and block on human input mid-search. Defensively force
            # the cloned game headless too, restoring the live view afterward.
            import copy as _copy

            cls = type(self)
            new = cls.__new__(cls)
            memo[id(self)] = new
            game: Any = self.__dict__.get("game")
            saved_bv = None
            if game is not None and not isinstance(game.boardView, _HeadlessView):
                saved_bv, game.boardView = game.boardView, _HeadlessView()
            try:
                for k, v in self.__dict__.items():
                    drop = k in ("_human_view", "_view_factory")
                    setattr(new, k, None if drop else _copy.deepcopy(v, memo))
            finally:
                if saved_bv is not None:
                    game.boardView = saved_bv
            return new

        # -- helpers -----------------------------------------------------------

        def _human_player(self) -> Any:
            # The HUMAN is the env's "opponent" seat.
            return self.opponent_player

        def _log_move(self, text: str | None) -> None:
            """Append one line to the on-screen move log, if one is attached.

            The log is owned by the harness and reached ONLY through the live
            view (``view.move_log``); the env holds no reference of its own. That
            is what makes an MCTS search clone structurally unable to log:
            ``__deepcopy__`` already drops ``_human_view``, so a clone's turns
            never reach a log. It is also why this is a plain bounded deque and
            NOT the recorder's ``EventCollector`` — that drain is destructive and
            single-consumer, and a second consumer would silently strip events
            out of the Replay."""
            if not text:
                return
            view = self._human_view
            if view is None:
                return
            log = getattr(view, "move_log", None)
            if log is not None:
                log.append(text)

        def _log_human_action(self, before: dict[str, Any]) -> None:
            """Log what the human's last click actually changed (or nothing)."""
            after = _human_snapshot(self._human_player())
            self._log_move(_describe_human_delta(before, after, self))

        class _ViewWindow:
            """Context manager: swap the real pygame view onto ``game.boardView``
            for a human input window, restoring the headless view on exit so the
            env's game stays deepcopy-safe for MCTS."""

            def __init__(self, env: HumanVsBotEnv) -> None:
                self.env = env
                self.saved: Any = None

            def __enter__(self) -> Any:
                game: Any = self.env.game
                assert game is not None
                self.saved = game.boardView
                game.boardView = self.env._human_view
                # Reflect whose turn it is for the stats panel + setup-mode input.
                game.currentPlayer = self.env._human_player()
                self.env._human_view.turn_banner = ("YOUR TURN", "forestgreen")
                self.env._human_view.displayGameScreen()
                return self.env._human_view

            def __exit__(self, *exc: Any) -> None:
                game: Any = self.env.game
                assert game is not None
                game.boardView = self.saved if self.saved is not None else _HeadlessView()
                self.env._human_view.displayGameScreen()

        # -- overridden opponent hooks (these become the HUMAN's turn) ---------

        def _opponent_setup_placement(self) -> None:
            """Human places ONE settlement + adjacent road during the snake draft."""
            game: Any = self.game
            assert game is not None
            human = self._human_player()
            board: Any = game.board
            if not self._use_gui():
                # Not human-driven: delegate to the base hook. In an MCTS search
                # clone the injected v8 snapshot plays this seat (the bot searches
                # against the v8 self-model, identical to the deployed SearchAgent);
                # on the live env / headless self-test it falls back to the
                # heuristic initial_setup. NEVER auto-place a passive no-op here —
                # that would make the bot plan against a do-nothing opponent.
                super()._opponent_setup_placement()
                return
            prev_setup = game.gameSetup
            game.gameSetup = True  # makes the view's setup-mode click loop apply
            before = _human_snapshot(human)
            try:
                with self._ViewWindow(self) as view:
                    print("\n[SETUP] Your move: place a SETTLEMENT (click a circle).", flush=True)
                    v = view.buildSettlement_display(human, board.get_setup_settlements(human))
                    if v is not None:
                        human.build_settlement(v, board, is_free=True)
                    self._log_human_action(before)
                    before = _human_snapshot(human)
                    view.displayGameScreen()
                    print("[SETUP] Now place an adjacent ROAD (click a line).", flush=True)
                    r = view.buildRoad_display(human, board.get_setup_roads(human))
                    if r is not None:
                        human.build_road(r[0], r[1], board, is_free=True)
                    self._log_human_action(before)
                    view.displayGameScreen()
            finally:
                game.gameSetup = prev_setup

        def _opponent_move_robber(self) -> None:
            """Human moves the robber + steals (after rolling a 7 or playing a Knight)."""
            game: Any = self.game
            assert game is not None
            human = self._human_player()
            board: Any = game.board
            if not self._use_gui():
                # Search clone (v8 snapshot) / self-test (heuristic) — see
                # _opponent_setup_placement for why we delegate instead of auto.
                super()._opponent_move_robber()
                return
            with self._ViewWindow(self) as view:
                view.displayDiceRoll(self.last_dice_roll)
                print(
                    "\n[ROBBER] Move the robber: click a hex, then a player to steal from.",
                    flush=True,
                )
                hex_i, victim = view.moveRobber_display(human, board.get_robber_spots())
                human.move_robber(hex_i, board, victim)
                self._log_move(f"You moved the robber to hex{hex_i}")
            game.check_largest_army(human)

        def _opponent_discard(self) -> None:
            """Human discards on a 7-roll (>9 cards) via the existing engine path."""
            game: Any = self.game
            assert game is not None
            human = self._human_player()
            if not self._use_gui():
                # Search clone (v8 snapshot) / self-test (heuristic) — delegate
                # instead of a hardcoded discard so the modelled seat plays its own.
                super()._opponent_discard()
                return
            n_before = sum(human.resources.values())
            print(
                f"\n[DISCARD] You rolled a 7 with {n_before} cards — discard half.",
                flush=True,
            )
            base_player: Any = PlainPlayer
            with self._ViewWindow(self) as view:
                view.displayDiceRoll(self.last_dice_roll)
                # The human is a heuristicAIPlayer whose discardResources auto-discards
                # with no GUI. Call the BASE player.discardResources, which drives
                # game.boardView.get_resource_selection so the discard menu shows.
                base_player.discardResources(human, game)
            n_after = sum(human.resources.values())
            if n_after != n_before:
                self._log_move(f"You discarded {n_before - n_after} cards")

        def _run_opponent_turn(self) -> None:
            """Run the human's whole turn. POST-ROLL dev cards only, like the bot.

            KEEP IN SYNC with catan_env._run_opponent_turn — this override now
            differs only in GUI plumbing (the human's discard / robber / main
            phase go through the pygame windows). Clones (MCTS) + the headless
            self-test never use the GUI, so they delegate to the base method —
            with the base's heuristic pre-roll suppressed (below).

            There is deliberately NO pre-roll dev-card window. ``env/masks.py``
            emits ONLY ``ROLL_DICE`` while ``roll_pending``, so the policy cannot
            play a Knight before rolling — not in this harness and not anywhere
            in its training history. A human window with no bot counterpart is
            the single largest bias in a playtest: it makes every human win
            unattributable between "the policy is weak here" and "the bot was not
            allowed to play its Knight". Both seats are post-roll only in this
            harness, matching the ruleset the champion was trained on. STOPGAP:
            ``preroll-dev-cards-r1.md`` (RATIFIED) gives BOTH seats the window
            and requires a retrain.
            """
            if not self._use_gui():  # clones + self-test
                # The base env plays a heuristic pre-roll Knight for the
                # opponent seat whenever no snapshot drives it — which is EVERY
                # MCTS clone rollout and every --self-test game. Left alone, the
                # bot would search against a human who can pre-roll while the
                # real human at the keyboard cannot: the same asymmetry D3
                # removed, reintroduced inside the search model.
                base_opp: Any = self.opponent_player
                if getattr(base_opp, "heuristic_pre_roll", None) is not None:
                    base_opp.heuristic_pre_roll = _no_pre_roll
                super()._run_opponent_turn()
                return
            game: Any = self.game
            assert game is not None
            opp: Any = self.opponent_player
            agent: Any = self.agent_player
            assert opp is not None and agent is not None
            game.currentPlayer = opp
            opp.updateDevCards()
            opp.devCardPlayedThisTurn = False
            dice = game.rollDice()
            self.last_dice_roll = dice
            self._log_move(f"You rolled {dice}")
            if dice != 7:
                game.update_playerResources(dice, opp)
            else:
                if sum(opp.resources.values()) > 9:
                    self._opponent_discard()
                if sum(agent.resources.values()) > 9:
                    # Agent must discard before the opponent can place the robber.
                    self._cards_to_discard = sum(agent.resources.values()) // 2
                    self.discard_pending = True
                    self._opp_pending_robber = True
                    return
                self._opponent_move_robber()
            if opp.victoryPoints >= game.maxPoints:
                return
            self._run_opponent_main_turn()

        def _run_opponent_main_turn(self) -> None:
            """The human's full main turn (dice already rolled by the env caller)."""
            game: Any = self.game
            assert game is not None
            human = self._human_player()
            board: Any = game.board
            if not self._use_gui():
                # LOAD-BEARING: delegate, never auto-pass. In an MCTS search clone
                # the injected v8 snapshot drives the full opponent turn (so the bot
                # plans against the v8 self-model, identical to the deployed
                # SearchAgent); on the live env / self-test it falls back to the
                # heuristic opp.move. A hardcoded no-op here makes the bot search
                # against a passive seat and play materially weaker.
                super()._run_opponent_main_turn()
                return
            with self._ViewWindow(self) as view:
                view.displayDiceRoll(self.last_dice_roll)
                print(f"\n[YOUR ROLL] You rolled {self.last_dice_roll}.", flush=True)
                self._human_interactive_main_turn(view, human, board)
            game.check_longest_road(human)
            game.check_largest_army(human)

        def _human_interactive_main_turn(self, view: Any, human: Any, board: Any) -> None:
            """Drive the human's main-phase buttons until they click END TURN.

            Mirrors the human branch of ``catanGame.playCatan`` (game.py) but
            scoped to a turn whose dice were already rolled by the env. Reuses the
            engine's own ``build`` / ``initiate_trade`` / ``draw_devCard`` /
            ``play_devCard`` GUI code paths verbatim.
            """
            import pygame  # local import — only needed in the interactive path

            # NOTE: updateDevCards + the SINGLE devCardPlayedThisTurn reset happen
            # once in _run_opponent_turn, before the dice roll. Re-resetting here
            # would let the human play a 2nd dev card in one turn.
            game: Any = self.game
            assert game is not None
            print(
                "\n[YOUR TURN] Buttons: ROAD / SETTLE / CITY / DEV CARD / PLAY DEV / "
                "BANK TRADE / END TURN. Dice are already rolled. Click END TURN when done.",
                flush=True,
            )
            view.displayGameScreen()
            clock = pygame.time.Clock()
            turn_over = False
            while not turn_over:
                clock.tick(60)
                for e in pygame.event.get():
                    if e.type == pygame.QUIT:
                        pygame.quit()
                        sys.exit(0)
                    if e.type != pygame.MOUSEBUTTONDOWN:
                        continue
                    before = _human_snapshot(human)
                    if view.buildRoad_button.collidepoint(e.pos):
                        game.build(human, "ROAD")
                        game.check_longest_road(human)
                    elif view.buildSettlement_button.collidepoint(e.pos):
                        game.build(human, "SETTLE")
                        game.check_longest_road(human)
                    elif view.buildCity_button.collidepoint(e.pos):
                        game.build(human, "CITY")
                    elif view.devCard_button.collidepoint(e.pos):
                        human.draw_devCard(board)
                    elif view.playDevCard_button.collidepoint(e.pos):
                        human.play_devCard(game)
                        game.check_largest_army(human)
                        game.check_longest_road(human)
                    elif view.tradeBank_button.collidepoint(e.pos):
                        human.initiate_trade(game, "BANK")
                    elif view.endTurn_button.collidepoint(e.pos):
                        turn_over = True
                        self._log_move("You ended your turn")
                    self._log_human_action(before)
                    if human.victoryPoints >= game.maxPoints:
                        turn_over = True  # win ends the turn at once (cf. game.py playCatan)
                    view.displayGameScreen()
                    print(
                        f"  You: VP={human.victoryPoints} resources={dict(human.resources)}",
                        flush=True,
                    )
                pygame.display.update()

    return HumanVsBotEnv


# ---------------------------------------------------------------------------
# Bot / search wiring
# ---------------------------------------------------------------------------


def _load_search_agent(ckpt: str, sims: int, seed: int) -> SearchAgent:
    """Load the checkpoint named by ``ckpt`` on CPU, wrapped in determinized PUCT-MCTS.

    Loads whatever ``--ckpt`` names (defaulting to :data:`DEFAULT_CKPT`, the
    pointer-arch champion) — NOT "v8", which is a stale lineage label.
    """
    from catan_rl.replay.player_factory import PlayerSpec, _PolicyActor, build_actor
    from catan_rl.search.agent import SearchAgent
    from catan_rl.search.config import SearchConfig

    actor = cast(
        "_PolicyActor",
        build_actor(PlayerSpec(kind="policy", ckpt_path=ckpt), seed=seed, device="cpu"),
    )
    cfg = SearchConfig(sims_per_move=sims, seed=seed)
    return SearchAgent(actor.policy, cfg, device=actor.device)


class _RawPolicyAgent:
    """Raw-policy bot: ``choose_action(env)`` with NO search.

    Drop-in for :class:`SearchAgent` (same one-method surface) so the game loop
    is identical either way.

    WHY THIS IS THE DEFAULT (correctness, not preference): the deployed
    determinized search is CLAIRVOYANT against a human. ``search.mcts.clone_env``
    deep-copies the live env, and ``MCTS._reseed`` reseeds only the opponent model
    and the numpy/stdlib streams — the Rust ``StackedDice`` bag is deep-copied and
    left untouched (its own docstring: "Dice stay per-line faithful via the env
    clone"). So every simulated world shares (a) the opponent's TRUE hidden dev
    cards and (b) the TRUE future dice. Against a bot-vs-bot harness both sides
    live in that same env so it is merely optimistic; against a HUMAN it is the
    tree reading your hand and your future rolls. A playtest run with search on is
    therefore uninterpretable and flattering — use raw policy until the
    de-clairvoyant determinization lands.
    """

    def __init__(self, actor: Any, *, capture_internals: bool = True) -> None:
        self._actor = actor
        #: Internals of the MOST RECENT decision, for the replay recorder.
        self.last_internals: Any = None
        #: Off when nothing is recording — capture costs a second policy forward.
        self.capture_internals = capture_internals
        self._capture_warned = False

    def choose_action(self, env: Any) -> np.ndarray:
        # Same pairing the eval harness uses: obs + legal-action masks straight
        # from the live env, sampled by the policy (no tree, no env clone).
        obs = env._get_obs()
        masks = env.get_action_masks()
        action = self._actor.select_action(obs, masks)
        # Recording is a SIDE CHANNEL: a capture fault must never kill the game it
        # observes (an hour of human play). capture_policy_internals reaches into
        # private head state (``heads._corner_context``, ``_corner_mask``,
        # ``out["_node_v"]`` ...), so drift in policy/heads.py or a checkpoint with
        # a different head set would otherwise raise mid-game and destroy the
        # session. Skip when nothing records; degrade to None on any fault.
        if not self.capture_internals:
            self.last_internals = None
            return action
        try:
            self.last_internals = capture_policy_internals(self._actor, obs, masks, action)
        except Exception as exc:
            self.last_internals = None
            if not self._capture_warned:
                self._capture_warned = True
                print(f"\n[recorder] policy-internals capture disabled: {exc}", flush=True)
        return action


def _top_k_pairs(probs: Any, k: int = 8) -> tuple[tuple[int, float], ...]:
    """Return the ``k`` largest ``(index, probability)`` pairs of a 1-D tensor.

    The pointer heads are 54 / 72 / 19 wide; storing them densely is ~168 floats
    per decision (roughly DOUBLING the replay file) and adds nothing a human
    reviewer reads. Zero-probability (illegal) entries are dropped."""
    import torch

    n = int(probs.shape[-1])
    values, indices = torch.topk(probs, k=min(k, n))
    pairs = zip(values.tolist(), indices.tolist(), strict=True)
    return tuple((int(i), float(v)) for v, i in pairs if v > 0.0)


def capture_policy_internals(actor: Any, obs: Any, masks: Any, action: np.ndarray) -> Any:
    """Recompute what the policy was weighing at ONE decision, script-side.

    ``MultiActionHeads.sample`` computes all six masked log-softmaxes and then
    DISCARDS everything but the chosen-index log-probs, and ``network.sample``
    already hands back ``trunk`` / ``_node_v`` / ``_node_e`` / ``_node_h`` /
    ``_is_setup`` / ``value`` / ``belief_logits``. Rather than change the PPO
    hot path to keep them, we re-run the (deterministic) forward here and read
    the head modules directly — **zero change to the training path**.

    ``chosen_action`` is the action that was actually applied; the
    distributions are deterministic given the obs + masks, so re-running the
    forward cannot disagree with it.

    **Only the heads the chosen type actually CONSULTS are stored.** The
    per-type relevance table (``MultiActionHeads.head_relevance``) is the same
    one PPO weights the joint log-prob with: on ``END_TURN`` the corner / edge /
    tile / resource heads contribute nothing, their masks are all-False, and
    ``masked_log_softmax`` deliberately returns the UNIFORM safe fallback
    (``heads.py``) — so recording them would attach a confident-looking pointer
    distribution to a decision that never looked at a pointer. An empty tuple
    means "this head was irrelevant here", never "the policy was undecided".

    Every value is converted to a plain Python float/int/bool: the replay
    schema and IO must stay torch-free (the viewer imports them).
    """
    import torch

    from catan_rl.policy.heads import masked_log_softmax
    from catan_rl.replay.player_factory import _DISCRETE_OBS_KEYS
    from catan_rl.replay.schema import PolicyInternals

    policy = actor.policy
    device = actor.device
    # Same dtype discipline as ``_PolicyActor.select_action`` — imported, not
    # re-declared, so a new discrete obs key cannot make this forward differ
    # from the one that produced the action.
    obs_t = {
        k: torch.as_tensor(
            v, dtype=torch.int64 if k in _DISCRETE_OBS_KEYS else torch.float32, device=device
        ).unsqueeze(0)
        for k, v in obs.items()
    }
    masks_t = {
        k: torch.as_tensor(
            np.ascontiguousarray(v, dtype=bool), dtype=torch.bool, device=device
        ).unsqueeze(0)
        for k, v in masks.items()
    }
    heads = policy.action_heads
    with torch.no_grad():
        out = policy.forward(obs_t)
        trunk = out["trunk"]
        type_probs = masked_log_softmax(heads.type_head(trunk), masks_t["type"]).exp()[0]
        # Condition the autoregressive heads on the type that was ACTUALLY
        # chosen — that is the branch whose sub-argument distribution is
        # meaningful for review.
        type_idx = torch.as_tensor([int(action[0])], dtype=torch.int64, device=device)
        # Which of the 6 heads this type actually consults — the SAME table PPO
        # weights the joint log-prob with. Irrelevant heads are skipped entirely
        # (their masks are all-False and would yield a uniform safe fallback).
        relevance = heads.head_relevance[type_idx][0]
        corner_top: tuple[tuple[int, float], ...] = ()
        edge_top: tuple[tuple[int, float], ...] = ()
        tile_top: tuple[tuple[int, float], ...] = ()
        res1_probs: tuple[float, ...] = ()
        res2_probs: tuple[float, ...] = ()
        if relevance[1] > 0:
            corner_ctx = heads._corner_context(type_idx, out.get("_is_setup"))
            corner_logits = heads.corner_head(trunk, out["_node_v"], corner_ctx)
            corner_mask = heads._corner_mask(type_idx, masks_t)
            corner_top = _top_k_pairs(masked_log_softmax(corner_logits, corner_mask).exp()[0])
        if relevance[2] > 0:
            edge_top = _top_k_pairs(
                masked_log_softmax(heads.edge_head(trunk, out["_node_e"]), masks_t["edge"]).exp()[0]
            )
        if relevance[3] > 0:
            tile_top = _top_k_pairs(
                masked_log_softmax(heads.tile_head(trunk, out["_node_h"]), masks_t["tile"]).exp()[0]
            )
        # The resource heads are only 5 wide, so they are stored DENSELY (~10
        # floats) — a BankTrade / Monopoly / YoP / Discard is unreadable without
        # the argument it was weighing.
        res1_idx = torch.as_tensor([int(action[4])], dtype=torch.int64, device=device)
        if relevance[4] > 0:
            res1_logits = heads.resource1_head(trunk, heads._resource1_context(type_idx))
            res1_lp = masked_log_softmax(res1_logits, heads._resource1_mask(type_idx, masks_t))
            res1_probs = tuple(float(x) for x in res1_lp.exp()[0].tolist())
        if relevance[5] > 0:
            res2_logits = heads.resource2_head(trunk, heads._resource2_context(type_idx, res1_idx))
            res2_lp = masked_log_softmax(
                res2_logits, heads._resource2_mask(type_idx, res1_idx, masks_t)
            )
            res2_probs = tuple(float(x) for x in res2_lp.exp()[0].tolist())
        belief = out.get("belief_logits")
        return PolicyInternals(
            type_mask=tuple(bool(b) for b in np.asarray(masks["type"]).reshape(-1)),
            type_probs=tuple(float(x) for x in type_probs.tolist()),
            chosen_action=tuple(int(x) for x in action),
            value=float(out["value"].reshape(-1)[0].item()),
            corner_top=corner_top,
            edge_top=edge_top,
            tile_top=tile_top,
            res1_probs=res1_probs,
            res2_probs=res2_probs,
            belief_logits=(
                None if belief is None else tuple(float(x) for x in belief.reshape(-1).tolist())
            ),
        )


def _load_raw_agent(ckpt: str, seed: int) -> _RawPolicyAgent:
    """Load a checkpoint as a bare policy actor (CPU), no search wrapper."""
    from catan_rl.replay.player_factory import PlayerSpec, build_actor

    actor = build_actor(PlayerSpec(kind="policy", ckpt_path=ckpt), seed=seed, device="cpu")
    return _RawPolicyAgent(actor)


def _visible_vp(player: Any) -> int:
    """Publicly VISIBLE victory points: total minus hidden VP cards.

    Mirrors :func:`catan_rl.gui.view.hand_panel_lines` and
    ``policy/obs_encoder.py``. Deliberately does NOT read
    ``player.visibleVictoryPoints`` — that cache is stale (refreshed only at
    init + VP-card buy)."""
    return int(player.victoryPoints) - int(player.devCards.get("VP", 0))


def _game_over_log_line(bot_player: Any, you_vp: int, *, reveal_bot: bool) -> str:
    """The terminal move-log line — the ONE place a VP-card-inclusive bot total
    could reach the screen (spec D4's single gate).

    Blind by default: the bot's score is rendered through :func:`_visible_vp`
    unless ``--reveal-bot`` was passed. Extracted from ``main`` purely so the
    leak can be pinned by a test."""
    bot_shown = int(bot_player.victoryPoints) if reveal_bot else _visible_vp(bot_player)
    return f"GAME OVER — Bot {bot_shown} - {you_vp} You"


def _safe_display_game_screen(view: Any) -> None:
    """Redraw the final board, but never at the cost of the artifacts.

    This runs BEFORE both artifact writes, and the salvage path it sits on is
    reached by ``pygame.QUIT`` handlers that call ``pygame.quit()`` and THEN
    ``sys.exit(0)`` (e.g. the human's own main turn). Drawing onto a torn-down
    display raises ``pygame.error: display Surface quit``, which would propagate
    out of ``play_interactive`` and skip the replay AND the JSONL line — the
    exact loss the salvage exists to prevent. Same no-surface guard as
    :func:`_hold_final_screen`; the ``pygame.error`` catch covers a display that
    dies between the check and the draw.
    """
    import pygame

    if not pygame.get_init() or pygame.display.get_surface() is None:
        return
    try:
        view.displayGameScreen()
    except pygame.error as exc:  # a dead window must not cost the record
        print(f"[WARN] final board could not be redrawn: {exc!r}", flush=True)


#: Seconds the finished board is held on screen before the window closes itself.
FINAL_SCREEN_HOLD_S = 120.0


def _hold_final_screen(timeout_s: float = FINAL_SCREEN_HOLD_S) -> None:
    """Keep the finished board on screen until the player dismisses it.

    Without this, ``play_interactive`` returns, ``main`` returns and
    ``raise SystemExit(main())`` tears the window down within milliseconds — so
    the terminal result line, the final board and the last log entries are never
    actually readable. This feature is the first to put content on screen
    AFTER the last human interaction, which is what makes the instant close a
    defect rather than a cosmetic quirk.

    Dismissed by a click, any key, or the window's close button. The timeout is
    a liveness guard so an unattended run can never hang forever, and the
    no-surface early return keeps every headless path (self-test, CI) untouched."""
    import time

    import pygame

    if not pygame.get_init() or pygame.display.get_surface() is None:
        return
    print("  (click the window or press any key to close)", flush=True)
    clock = pygame.time.Clock()
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        for e in pygame.event.get():
            if e.type in (pygame.QUIT, pygame.MOUSEBUTTONDOWN, pygame.KEYDOWN):
                return
        pygame.display.update()
        clock.tick(30)


def _describe_bot_move(action: Sequence[int] | np.ndarray, *, with_location: bool = False) -> str:
    """Human-readable label for one bot action tuple.

    The DEFAULT string is frozen: it is written into ``games.jsonl`` as
    ``bot_action_label`` and into the replay's step note, so changing it would
    silently re-point content in the project's most-cited artifact.

    ``with_location=True`` is the on-screen HUD variant and additionally names
    the vertex / edge / hex index the action targets. It reads the index off the
    ACTION TUPLE — the broadcast ``BUILD`` event carries ``location=-1`` as a
    documented sentinel, and PLAY_KNIGHT / PLAY_ROAD_BUILDER emit no event at
    all, which is why the log is fed from actions rather than the broadcast.

    Neither variant can name a bot resource TYPE it holds privately: the only
    resources rendered are the ones the acted move makes public (a bank trade's
    two sides, a Year-of-Plenty pick, a Monopoly call)."""
    from catan_rl.env.catan_env import RESOURCES_CW, ActionType

    names = {
        ActionType.BUILD_SETTLEMENT: "Build settlement",
        ActionType.BUILD_CITY: "Build city",
        ActionType.BUILD_ROAD: "Build road",
        ActionType.END_TURN: "End turn",
        ActionType.MOVE_ROBBER: "Move robber",
        ActionType.BUY_DEV_CARD: "Buy dev card",
        ActionType.PLAY_KNIGHT: "Play Knight",
        ActionType.PLAY_YOP: "Play Year of Plenty",
        ActionType.PLAY_MONOPOLY: "Play Monopoly",
        ActionType.PLAY_ROAD_BUILDER: "Play Road Builder",
        ActionType.BANK_TRADE: "Bank trade",
        ActionType.DISCARD: "Discard",
        ActionType.ROLL_DICE: "Roll dice",
    }
    t = int(action[0])
    label = names.get(t, f"type={t}")
    if t == ActionType.BANK_TRADE:
        label += f" (give {RESOURCES_CW[int(action[4])]} -> get {RESOURCES_CW[int(action[5])]})"
    elif t == ActionType.PLAY_YOP:
        label += f" ({RESOURCES_CW[int(action[4])]} + {RESOURCES_CW[int(action[5])]})"
    elif t == ActionType.PLAY_MONOPOLY:
        label += f" ({RESOURCES_CW[int(action[4])]})"
    if with_location:
        if t in (ActionType.BUILD_SETTLEMENT, ActionType.BUILD_CITY):
            label += f" at v{int(action[1])}"
        elif t == ActionType.BUILD_ROAD:
            label += f" at e{int(action[2])}"
        elif t == ActionType.MOVE_ROBBER:
            label += f" to hex{int(action[3])}"
        # PLAY_KNIGHT deliberately names NO hex. It does not consume head 3:
        # `_apply_main_action` (env/catan_env.py) only decrements the card,
        # bumps `knightsPlayed` and sets `robber_placement_pending`; the
        # destination arrives in a SEPARATE MOVE_ROBBER action on the next
        # step. During a main turn `masks.py` leaves `tile_mask` all-False, so
        # `action[3]` is a uniformly random index — naming it would assert a
        # false public fact and contradict the MOVE_ROBBER line that follows.
    return label


#: Dev-card key -> the name shown for the HUMAN's own play in the move log.
_HUMAN_DEV_LABEL = {
    "KNIGHT": "Knight",
    "MONOPOLY": "Monopoly",
    "ROADBUILDER": "Road Builder",
    "YEAROFPLENTY": "Year of Plenty",
}


def _human_snapshot(player: Any) -> dict[str, Any]:
    """Capture the facts ``_describe_human_delta`` diffs across one button click."""
    build_graph = player.buildGraph
    return {
        "settlements": list(build_graph["SETTLEMENTS"]),
        "cities": list(build_graph["CITIES"]),
        "roads": list(build_graph["ROADS"]),
        "knights": int(player.knightsPlayed),
        "dev": dict(player.devCards),
        "new_dev": len(player.newDevCards),
        "cards": sum(player.resources.values()),
    }


def _describe_human_delta(
    before: dict[str, Any], after: dict[str, Any], env: Any = None
) -> str | None:
    """One move-log line for what the human just did, or ``None`` if nothing did.

    Pure apart from the two index maps it reads off ``env``
    (``_vertex_to_idx`` / ``_edge_to_idx``, both already built by ``CatanEnv``),
    so a cancelled click — the common case, since every build button can be
    backed out of — produces no entry at all.

    This is the HUMAN's own information, so naming the dev card they played
    leaks nothing about the bot."""
    vertex_to_idx = getattr(env, "_vertex_to_idx", None) or {}
    edge_to_idx = getattr(env, "_edge_to_idx", None) or {}

    def vertex_label(coord: Any) -> str:
        return f"v{vertex_to_idx[coord]}" if coord in vertex_to_idx else "v?"

    def edge_label(edge: Any) -> str:
        # Use the env's OWN key builder rather than re-deriving it here: the
        # lookup fails silently ("e?"), so a drift between the two would degrade
        # every human build line without tripping anything.
        from catan_rl.env.catan_env import CatanEnv

        key = CatanEnv._edge_key(edge[0], edge[1])
        return f"e{edge_to_idx[key]}" if key in edge_to_idx else "e?"

    parts: list[str] = []
    for coord in after["settlements"]:
        if coord not in before["settlements"]:
            parts.append(f"built settlement at {vertex_label(coord)}")
    for coord in after["cities"]:
        if coord not in before["cities"]:
            parts.append(f"upgraded to city at {vertex_label(coord)}")
    for edge in after["roads"]:
        if edge not in before["roads"]:
            parts.append(f"built road at {edge_label(edge)}")
    if after["knights"] > before["knights"]:
        parts.append("played Knight")
    for card, count in before["dev"].items():
        if card in ("VP", "KNIGHT"):  # VP is never "played"; Knight is counted above
            continue
        if after["dev"].get(card, 0) < count:
            parts.append(f"played {_HUMAN_DEV_LABEL.get(card, card)}")
    # A VP card is applied IMMEDIATELY by ``draw_devCard`` — it bumps
    # ``devCards['VP']`` and never lands in ``newDevCards`` — so a VP draw
    # (5 of the 25 cards) shows up only here. Without this arm the buy fell
    # through to the bank-trade fallback and was logged as a false statement.
    bought_vp = after["dev"].get("VP", 0) > before["dev"].get("VP", 0)
    if after["new_dev"] > before["new_dev"] or bought_vp:
        parts.append("bought a dev card")
    if not parts and after["cards"] != before["cards"]:
        parts.append("traded with the bank")
    if not parts:
        return None
    return "You " + ", ".join(parts)


# ---------------------------------------------------------------------------
# Replay recording for the interactive game
# ---------------------------------------------------------------------------


class _DetachedSubscriber:
    """Inert broadcast subscriber a search clone gets instead of the real one.

    Callable, and does nothing: events emitted inside a throwaway MCTS world
    land here and never reach the record."""

    def __call__(self, event: dict[str, Any]) -> None:
        return None


class _RecorderSubscriber:
    """Broadcast subscriber that refuses to be cloned.

    A BOUND METHOD cannot carry this guard. ``copy._deepcopy_method`` rebuilds
    it as ``MethodType(x.__func__, deepcopy(x.__self__))`` — it keeps the
    ORIGINAL function and only swaps the instance — so a ``__deepcopy__`` on the
    recorder itself would hand the clone ``_HumanGameRecorder.on_event`` bound
    to a stand-in that has none of its attributes, and the FIRST broadcast
    inside any MCTS simulation would raise ``AttributeError``. A standalone
    callable object IS routed through its own ``__deepcopy__``, so the clone
    gets an inert stand-in instead — and the recorder, unreachable from the env
    by any other path, is never deep-copied (its ``steps`` list would otherwise
    be copied once per simulation, at 400 sims/move)."""

    def __init__(self, recorder: _HumanGameRecorder) -> None:
        self._recorder = recorder

    def __call__(self, event: dict[str, Any]) -> None:
        self._recorder.on_event(event)

    def __deepcopy__(self, memo: dict[int, Any]) -> Any:
        detached = _DetachedSubscriber()
        memo[id(self)] = detached
        return detached


class _HumanGameRecorder:
    """Assemble a real :class:`catan_rl.replay.schema.Replay` from an
    INTERACTIVE human-vs-policy game.

    ``recorder_loop.record_game`` cannot host this game: it builds its OWN
    ``CatanEnv`` and drives a fixed 4-step setup loop against a non-interactive
    actor surface. So only the main-loop primitives are reused —
    ``EventCollector``, ``snapshot_step_state``, ``setup_steps_seat_0/1``,
    ``split_at_setup_complete`` and ``consume_main_event_block`` — wrapped
    around the live interactive loop. The bookkeeping below deliberately
    mirrors ``record_game`` step for step so the two records stay comparable.

    FIDELITY CAVEATS — read these before drawing conclusions from a record:

    * **The four SETUP steps are SYNTHESIZED, not observed.**
      ``setup_steps_seat_0/1`` reconstruct the placements from the action
      tuples and hardcode ``longest_road_holder=None``. The OPENING is
      therefore the LEAST faithful part of the record — which is unfortunate,
      because the opening is exactly the phase under suspicion.
    * **``state_after`` is SHARED across every sub-step produced by one
      ``env.step``.** The human's whole turn is folded inside the bot's
      ``env.step``, so a human step and the bot step that contains it carry the
      same end-of-``env.step`` board. Per-step ``actions`` / ``events`` ARE
      correctly attributed (per-actor diff + acting-seat partition); only the
      board snapshot is a shared tail.
    * **``policy_internals`` is a hypothesis generator, never a verdict.** The
      13-way type softmax reads roughly ``0.97 END_TURN`` at most steps and is
      informative only at contested decisions; and a policy explaining its own
      choices is the agent grading its own homework.
    * **Every bot decision carries its internals, including the ones that
      change nothing on the board.** ``END_TURN`` (and a knight that steals
      nothing) emits no broadcast events, so the event partition yields no
      bot-attributed step for it; those decisions would otherwise vanish
      together with their internals — which is exactly the ``END_TURN``-heavy
      class the reviewer wants to inspect, and the class an offline "legal vs
      chosen" query needs. A DECISION-ONLY step (no actions, no events,
      ``state_after`` = end of this ``env.step``) is emitted for them instead.
    * **An empty pointer/resource distribution means "not consulted".** Only
      the heads the chosen action type actually uses are stored (per
      ``MultiActionHeads.head_relevance``): an ``END_TURN`` carries no
      ``corner_top`` / ``edge_top`` / ``tile_top`` / ``res*_probs`` at all,
      because its masks are all-False and ``masked_log_softmax`` would hand
      back the uniform safe fallback — noise wearing the shape of an opinion.
    * **Under ``--search`` no internals are captured at all.** ``SearchAgent``
      exposes no per-decision distribution, so every step carries an empty
      tuple; ``metadata.mode == "search"`` is what marks that record as
      "not captured", NOT "the policy had no opinion".
    """

    def __init__(self, env: Any, *, bot_seat: int) -> None:
        from catan_rl.engine.broadcast import BroadcastEventType
        from catan_rl.replay.recorder import EventCollector, snapshot_step_state

        # ``_seat_to_actor`` / ``_build_board_static_from_dict`` stay private in
        # recorder_loop; importing them beats duplicating a mapping that must
        # never drift from the one record_game uses.
        from catan_rl.replay.recorder_loop import _seat_to_actor

        assert env.game is not None, "recorder must be attached AFTER env.reset"
        self._env = env
        self._bot_seat = bot_seat
        self._setup_complete_type = BroadcastEventType.SETUP_COMPLETE.value
        self._snapshot_step_state = snapshot_step_state
        self.seat_to_actor = _seat_to_actor(bot_seat)
        self._vertex_pixel_to_idx = dict(env._vertex_to_idx)
        self._edge_key_to_idx = dict(env._edge_to_idx)

        self._collector = EventCollector()
        self._collector.subscribe(env.game.broadcast)
        self._setup_complete_snaps: list[Any] = []
        # NOT ``self.on_event``: a bound method survives ``deepcopy`` as the
        # original function rebound to a copy of this recorder, which would put
        # the whole record inside every MCTS clone. See _RecorderSubscriber.
        env.game.broadcast.subscribe(_RecorderSubscriber(self))

        # Snapshot right after reset. When the HUMAN drafts first the env has
        # already placed their first settlement inside reset(); those events are
        # reconstructed from snapshots, exactly as record_game does.
        self._snap_after_reset = self._snap_now()
        self._collector.drain()

        self.steps: list[Any] = []
        self._step_idx = 0
        self._env_step_idx = 0
        self._setup_sub_idx: dict[int, int] = {}
        self._setup_internals: dict[int, Any] = {}
        self._snap_after_step1: Any = None
        self._prev_snap: Any = self._snap_after_reset
        #: Bot decisions seen vs decisions whose internals reached a step. Equal
        #: whenever the agent supplies internals at all; a gap means silent loss.
        self.n_bot_decisions = 0
        self.n_internals_recorded = 0

    # -- plumbing ----------------------------------------------------------

    def on_event(self, event: dict) -> None:
        """Broadcast callback. Invoked through :class:`_RecorderSubscriber`,
        never subscribed directly — see that class for why."""
        if event.get("type") == self._setup_complete_type:
            self._setup_complete_snaps.append(self._snap_now())

    def _snap_now(self) -> Any:
        return self._snapshot_step_state(
            self._env.game,
            seat_to_actor=self.seat_to_actor,
            vertex_pixel_to_idx=self._vertex_pixel_to_idx,
            edge_key_to_idx=self._edge_key_to_idx,
        )

    # -- driver ------------------------------------------------------------

    def after_env_step(
        self,
        action: np.ndarray,
        internals: Any,
        *,
        terminated: bool,
        truncated: bool,
    ) -> None:
        """Call once after every ``env.step``, in order."""
        i = self._env_step_idx
        self._env_step_idx += 1
        if internals is not None:
            self.n_bot_decisions += 1
        if i < 4:
            # Setup: corner head for the settlements (steps 0, 2), edge head for
            # the roads (steps 1, 3).
            self._setup_sub_idx[i] = int(action[1]) if i % 2 == 0 else int(action[2])
            # The opening is the phase under suspicion: keep the internals for
            # the four setup decisions too (they are attached to the two
            # synthesized agent setup steps in _finish_setup).
            self._setup_internals[i] = internals
            if i == 1:
                self._snap_after_step1 = self._snap_now()
            if i == 3:
                self._finish_setup()
            return
        self._consume_main(internals, terminated=terminated, truncated=truncated)

    def _finish_setup(self) -> None:
        from catan_rl.replay.recorder_loop import (
            consume_main_event_block,
            setup_steps_seat_0,
            setup_steps_seat_1,
            split_at_setup_complete,
        )

        if len(self._setup_complete_snaps) != 1:
            raise RuntimeError(
                f"recorder: expected exactly 1 SETUP_COMPLETE event, got "
                f"{len(self._setup_complete_snaps)}. Engine drift suspected."
            )
        setup_complete_snap = self._setup_complete_snaps[0]
        pairs = [
            (self._setup_sub_idx[0], self._setup_sub_idx[1]),
            (self._setup_sub_idx[2], self._setup_sub_idx[3]),
        ]
        assert self._snap_after_step1 is not None
        if self._bot_seat == 0:
            setup_steps, _lines = setup_steps_seat_0(
                agent_actions=pairs,
                snap_after_step1=self._snap_after_step1,
                setup_complete_snap=setup_complete_snap,
                seat_to_actor=self.seat_to_actor,
            )
        else:
            setup_steps, _lines = setup_steps_seat_1(
                agent_actions=pairs,
                snap_after_reset=self._snap_after_reset,
                snap_after_step1=self._snap_after_step1,
                setup_complete_snap=setup_complete_snap,
                seat_to_actor=self.seat_to_actor,
            )
        # Each agent setup step folds ONE settlement + ONE road decision
        # (env.steps 0/1, then 2/3), in order.
        bot_actor = self.seat_to_actor["Agent"]
        pending = [
            (self._setup_internals.get(0), self._setup_internals.get(1)),
            (self._setup_internals.get(2), self._setup_internals.get(3)),
        ]
        for j, step in enumerate(setup_steps):
            if step.actor != bot_actor or not pending:
                continue
            got = tuple(x for x in pending.pop(0) if x is not None)
            if got:
                setup_steps[j] = dataclasses.replace(step, policy_internals=got)
                self.n_internals_recorded += len(got)
        self.steps.extend(setup_steps)
        self._step_idx = len(self.steps)
        self._prev_snap = setup_complete_snap

        # When the human drafts first the env runs their FIRST MAIN TURN inside
        # env.step #3; those events sit after SETUP_COMPLETE in this buffer and
        # must not be dropped, or the replay silently loses the human's opener.
        residual = split_at_setup_complete(self._collector.drain())
        if residual:
            post_snap = self._snap_now()
            new_steps, last_snap, self._step_idx = consume_main_event_block(
                raw_events=residual,
                prev_snap=self._prev_snap,
                post_snap=post_snap,
                initial_actor=self.seat_to_actor["Opponent"],
                seat_to_actor=self.seat_to_actor,
                start_idx=self._step_idx,
                terminated=False,
                truncated=False,
            )
            self.steps.extend(new_steps)
            self._prev_snap = last_snap

    def _consume_main(self, internals: Any, *, terminated: bool, truncated: bool) -> None:
        from catan_rl.replay.recorder_loop import _replay_step, consume_main_event_block

        post_snap = self._snap_now()
        block_start = self._step_idx
        new_steps, last_snap, self._step_idx = consume_main_event_block(
            raw_events=self._collector.drain(),
            prev_snap=self._prev_snap,
            post_snap=post_snap,
            initial_actor=self.seat_to_actor["Agent"],
            seat_to_actor=self.seat_to_actor,
            start_idx=block_start,
            terminated=terminated,
            truncated=truncated,
        )
        if internals is not None:
            # One env.step = exactly one bot decision, so the internals belong
            # to the FIRST step attributed to the bot in this block. Human steps
            # in the same block keep an empty tuple.
            bot_actor = self.seat_to_actor["Agent"]
            attached = False
            for j, step in enumerate(new_steps):
                if step.actor == bot_actor:
                    new_steps[j] = dataclasses.replace(step, policy_internals=(internals,))
                    attached = True
                    break
            if not attached:
                # The decision produced NO broadcast events (END_TURN, a knight
                # that steals nothing, ...), so the partition emitted no
                # bot-attributed step and both the decision and its internals
                # would be dropped. Emit a DECISION-ONLY step at the head of the
                # block — chronologically the bot acts first inside its own
                # env.step — and renumber the rest.
                decision = dataclasses.replace(
                    _replay_step(
                        step_idx=block_start,
                        # The terminal flag stays on the LAST step of the block;
                        # only an otherwise-empty block makes this one last.
                        kind=(
                            "terminal" if (terminated or truncated) and not new_steps else "main"
                        ),
                        actor=bot_actor,
                        dice_roll=None,
                        actions=(),
                        events=(),
                        log_lines=(
                            f"{_describe_bot_move(internals.chosen_action)}"
                            " (no board change; decision-only step)",
                        ),
                        state_after=post_snap,
                    ),
                    policy_internals=(internals,),
                )
                new_steps.insert(0, decision)
                for j, step in enumerate(new_steps):
                    new_steps[j] = dataclasses.replace(step, step_idx=block_start + j)
                self._step_idx = block_start + len(new_steps)
                last_snap = post_snap
            self.n_internals_recorded += 1
        self.steps.extend(new_steps)
        self._prev_snap = last_snap

    # -- finish ------------------------------------------------------------

    def finish(
        self,
        *,
        ckpt: str,
        seed: int,
        mode: str,
        sims: int | None,
        clairvoyant: bool,
        reveal_bot: bool,
        partial: bool = False,
    ) -> Any:
        """Build the :class:`Replay` for the game observed so far.

        ``partial=True`` means the game did NOT play out — the driver's loop was
        interrupted (window closed, ^C, a crash) and this is a salvage write.
        Keyword-only with a default, so existing callers are unaffected.
        """
        import datetime as _dt

        from catan_rl.replay.recorder_loop import _build_board_static_from_dict
        from catan_rl.replay.schema import REPLAY_SCHEMA_VERSION, Metadata, PlayerSpec, Replay

        env = self._env
        game = env.game
        bot_vp = int(env.agent_player.victoryPoints)
        human_vp = int(env.opponent_player.victoryPoints)
        if bot_vp >= game.maxPoints:
            winner_seat: int | None = self._bot_seat
            winner: str | None = self.seat_to_actor["Agent"]
        elif human_vp >= game.maxPoints:
            winner_seat = 1 - self._bot_seat
            winner = self.seat_to_actor["Opponent"]
        else:
            winner_seat = None
            winner = None

        bot_spec = PlayerSpec(
            kind="policy",
            ckpt_path=ckpt,
            color="black" if self._bot_seat == 0 else "darkslateblue",
            seat_index=self._bot_seat,
        )
        # kind="human" is a first-class record value; labelling the person
        # "heuristic" to fit the old enum would poison every consumer.
        human_spec = PlayerSpec(
            kind="human",
            ckpt_path=None,
            color="darkslateblue" if self._bot_seat == 0 else "black",
            seat_index=1 - self._bot_seat,
        )
        player_a, player_b = (
            (bot_spec, human_spec) if self._bot_seat == 0 else (human_spec, bot_spec)
        )
        final_vp = (bot_vp, human_vp) if self._bot_seat == 0 else (human_vp, bot_vp)

        metadata = Metadata(
            player_a=player_a,
            player_b=player_b,
            seed=seed,
            max_turns=int(env.max_turns),
            intended_hex_size=(1000, 800),
            recorded_at_utc=_dt.datetime.now(_dt.UTC).isoformat(timespec="seconds"),
            winner=winner,
            winner_seat=winner_seat,
            final_vp=final_vp,
            total_steps=len(self.steps),
            partial=bool(partial),
            mode=mode,
            sims=sims,
            clairvoyant=clairvoyant,
            reveal_bot=reveal_bot,
            # Provenance: the record must name the exact policy and code it was
            # played against. ``ckpt`` is a path into a gitignored, rotated
            # ``runs/`` tree, so the FILE is hashed as well.
            git_sha=_git_sha(),
            ckpt_sha256=_ckpt_sha256(ckpt),
        )
        return Replay(
            schema_version=REPLAY_SCHEMA_VERSION,
            metadata=metadata,
            board_static=_build_board_static_from_dict(game.board.board_static()),
            steps=tuple(self.steps),
        )


# ---------------------------------------------------------------------------
# Interactive game (real GUI)
# ---------------------------------------------------------------------------


def play_interactive(
    ckpt: str,
    sims: int,
    seed: int,
    human_seat: int,
    *,
    use_search: bool = False,
    log_path: str | None = None,
    reveal_bot: bool = False,
    replay_dir: str | None = None,
) -> None:
    """Run an interactive human-vs-bot game with the pygame GUI.

    ``use_search=False`` (the DEFAULT) plays the RAW policy — see
    :class:`_RawPolicyAgent` for why search is not valid against a human.

    ``reveal_bot=False`` (the DEFAULT) keeps the bot's hand HIDDEN — see
    :func:`catan_rl.gui.view.hand_panel_lines`. Revealing it makes the game an
    analysis session, not a strength read, so the flag is written into both the
    JSONL record and the replay metadata.
    """
    from collections import deque

    from catan_rl.gui.view import MOVE_LOG_LINES
    from catan_rl.gui.view import catanGameView as _catanGameView

    catanGameView: Any = _catanGameView
    HumanVsBotEnv = _build_human_env_class()
    # The BOT is the env "agent" seat; the HUMAN is the env "opponent" seat.
    # agent_seat selects who acts first in the snake draft: bot_seat = 1 - human_seat.
    bot_seat = 1 - human_seat
    mode = f"PUCT-MCTS, {sims} sims/move" if use_search else "RAW POLICY (no search)"

    print("=" * 70, flush=True)
    print("  1v1 Catan (Colonist.io ruleset) — YOU vs the CHAMPION", flush=True)
    print("=" * 70, flush=True)
    print(f"  Bot: {ckpt}", flush=True)
    print(f"  Mode: {mode}  (CPU)", flush=True)
    if use_search:
        print("  " + "!" * 66, flush=True)
        print("  !! WARNING: search is CLAIRVOYANT against a human. Its simulated", flush=True)
        print("  !! worlds share YOUR true hidden dev cards AND the true future dice", flush=True)
        print("  !! (the dice bag is deep-copied and never reseeded). This game is", flush=True)
        print("  !! NOT a valid strength read — the bot is cheating. Use raw policy.", flush=True)
        print("  " + "!" * 66, flush=True)
    print("  Win = 15 VP. No player-to-player trading (bank/port only).", flush=True)
    print("  Discard threshold = 9 cards. Friendly Robber in effect.", flush=True)
    print(f"  You are {'FIRST' if human_seat == 0 else 'SECOND'} in the snake draft.", flush=True)
    print("-" * 70, flush=True)
    print("  HOW TO PLAY (click with the mouse):", flush=True)
    print("   - Setup: click a highlighted circle (settlement) then a line (road).", flush=True)
    print("   - Your turn: dice auto-roll; use the on-screen buttons. Click END", flush=True)
    print("     TURN to pass to the bot. The bot then moves.", flush=True)
    print("   - On a 7: discard menu pops up (>9 cards); then move robber + steal.", flush=True)
    if reveal_bot:
        print("-" * 70, flush=True)
        print("  --reveal-bot: the bot's FULL hand is shown. This is an ANALYSIS", flush=True)
        print("  session, NOT a strength read — recorded as such in both logs.", flush=True)
    print("=" * 70, flush=True)

    agent: Any = _load_search_agent(ckpt, sims, seed) if use_search else _load_raw_agent(ckpt, seed)
    move_log: list[dict[str, Any]] = []
    env: Any = HumanVsBotEnv(opponent_type="heuristic", max_turns=400)
    # Must be set BEFORE reset: reset() can build the view (human drafts first).
    env._reveal_bot = reveal_bot
    # Register the view builder BEFORE reset. When the human drafts first
    # (bot_seat==1), the env places the human's FIRST settlement inside reset();
    # the lazy factory makes that placement use the GUI instead of auto-picking,
    # so NOTHING is auto-placed for either player (the bot's placements always
    # come from search via the loop). The view is never stored on game.boardView
    # except inside a human input window (deepcopy-safe for MCTS).
    # On-screen recent-move log. OWNED HERE and reached only through the view, so
    # the env stores no reference and an MCTS clone (which drops ``_human_view``)
    # cannot log. Deliberately NOT the recorder's EventCollector: that drain is
    # destructive and single-consumer, and sharing it would silently strip events
    # out of the Replay.
    hud_log: Any = deque(maxlen=MOVE_LOG_LINES)

    def _make_view(game: Any) -> Any:
        built = catanGameView(game.board, game)
        built.move_log = hud_log
        return built

    env.set_view_factory(_make_view)
    check_bank = _make_bank_conservation_reporter(hud_log)
    env.reset(seed=seed, options={"agent_seat": bot_seat})
    check_bank(env)  # setup grants are metered too
    # Ensure a view exists for rendering (already built during reset if the human
    # drafted first; built here otherwise — board is fixed at reset).
    assert env.game is not None
    view: Any = env._ensure_view()
    view.displayGameScreen()

    # Full-fidelity record of BOTH players' streams (see _HumanGameRecorder for
    # the fidelity caveats — the synthesized opening especially).
    recorder = _HumanGameRecorder(env, bot_seat=bot_seat) if replay_dir else None

    terminated = truncated = False
    safety_cap = env.max_turns * 50
    n_steps = 0
    aborted = False
    while not terminated and not truncated:
        try:
            # Show a clear "bot thinking" flag before the (blocking) search so turns
            # don't feel like they pass silently. The window can't animate during the
            # search itself (it's a synchronous call); lower --sims for snappier turns.
            view.turn_banner = ("BOT IS THINKING...", "gray30")
            view.displayGameScreen()
            # Bot (agent seat) decides + applies its action; this also folds the
            # human's whole turn internally via the overridden _opponent_* hooks.
            action = agent.choose_action(env)
            print(f"\n[BOT] {_describe_bot_move(action)}", flush=True)
            # HUD variant names the target index; the stdout/ledger label is frozen.
            #
            # A ROLL is logged AFTER the step, because the value only exists once the
            # step has run (``env.last_dice_roll``). Without this the bot's roll read
            # a bare "Roll dice" while the human's read "You rolled 8" — leaving the
            # persistent log strictly LESS informative than the single overwritten
            # banner it replaces, on the most-consulted public fact in the game.
            # Deferring the append is safe for a roll specifically: only END_TURN
            # folds the human's turn, so nothing else can interleave ahead of it.
            from catan_rl.env.catan_env import ActionType as _AT

            is_roll = int(action[0]) == int(_AT.ROLL_DICE)
            if not is_roll:
                hud_log.append(f"Bot: {_describe_bot_move(action, with_location=True)}")
            _obs, _r, terminated, truncated, _info = env.step(action)
            # A step folds the human's WHOLE turn (picker included), so this is the
            # tightest place the invariant can be checked from the driver.
            check_bank(env)
            if is_roll:
                hud_log.append(f"Bot: Rolled {int(getattr(env, 'last_dice_roll', 0))}")
            if recorder is not None:
                # Recording is a SIDE CHANNEL: a recorder fault must never kill or
                # erase the game it is observing (an hour of human play), so it is
                # detached on the first failure and the game plays on.
                try:
                    recorder.after_env_step(
                        action,
                        getattr(agent, "last_internals", None),
                        terminated=terminated,
                        truncated=truncated,
                    )
                except Exception as exc:  # never abort the game over a recorder fault
                    print(f"[WARN] replay recorder disabled after error: {exc!r}", flush=True)
                    recorder = None
            view.displayGameScreen()
            assert env.agent_player is not None and env.opponent_player is not None
            # The bot's VP-card count is HIDDEN information: print its VISIBLE VP
            # unless --reveal-bot. Your own total is your own information.
            bot_vp_label = "Bot VP" if reveal_bot else "Bot visible VP"
            bot_vp_shown = (
                int(env.agent_player.victoryPoints) if reveal_bot else _visible_vp(env.agent_player)
            )
            print(
                f"  {bot_vp_label}={bot_vp_shown} | You VP={env.opponent_player.victoryPoints}",
                flush=True,
            )
            # Replayable record: the bot's action tuple + the VP state after it. The
            # human's whole turn is folded inside env.step, so this is a bot-move log
            # with score checkpoints, not a full move-by-move transcript.
            move_log.append(
                {
                    "step": n_steps,
                    "bot_action": [int(x) for x in action],
                    "bot_action_label": _describe_bot_move(action),
                    "bot_vp": int(env.agent_player.victoryPoints),
                    "human_vp": int(env.opponent_player.victoryPoints),
                }
            )
            n_steps += 1
            if n_steps > safety_cap:
                print("[WARN] safety cap hit; ending.", flush=True)
                break
        except BaseException as exc:
            # A misclick must not destroy an hour of play. BOTH artifact writes
            # live AFTER this loop, and five GUI paths call ``sys.exit(0)`` on
            # ``pygame.QUIT``, so closing the window mid-game used to discard the
            # whole game silently.
            #
            # BaseException, not Exception, ON PURPOSE: ``SystemExit`` (those quit
            # paths) and ``KeyboardInterrupt`` (^C) are both BaseException and
            # would slip straight through an ``except Exception`` — which is the
            # exact failure being fixed. Nothing is re-raised: falling through to
            # the write block is the point, and the record is marked ``partial``
            # so a half-finished game can never be read as a played-out one.
            import traceback

            aborted = True
            print(
                f"\n[ABORTED] game interrupted after {n_steps} bot moves: {exc!r}\n"
                "          writing PARTIAL artifacts before exit.",
                flush=True,
            )
            traceback.print_exc()
            break

    assert env.agent_player is not None and env.opponent_player is not None
    bot_vp = int(env.agent_player.victoryPoints)
    you_vp = int(env.opponent_player.victoryPoints)
    print("\n" + "=" * 70, flush=True)
    if you_vp >= 15 and you_vp > bot_vp:
        print(f"  YOU WIN!  You {you_vp} - {bot_vp} Bot", flush=True)
    elif bot_vp >= 15 and bot_vp > you_vp:
        print(f"  Bot wins.  Bot {bot_vp} - {you_vp} You", flush=True)
    else:
        print(f"  Game ended (truncated). Bot {bot_vp} - {you_vp} You", flush=True)
    print("=" * 70, flush=True)
    # Terminal HUD line. The bot's VP-card count stays HIDDEN unless --reveal-bot
    # (this is the one place a VP-card-inclusive total could reach the screen).
    hud_log.append(_game_over_log_line(env.agent_player, you_vp, reveal_bot=reveal_bot))
    _safe_display_game_screen(view)

    # ---- full-fidelity replay (new format) --------------------------------
    replay_path: str | None = None
    if recorder is not None:
        import time
        from pathlib import Path

        from catan_rl.replay.io import save_replay

        # Same reason as above: a failed replay write must not take the legacy
        # JSONL line (and with it the whole record of the game) down with it.
        try:
            replay = recorder.finish(
                ckpt=ckpt,
                seed=seed,
                mode="search" if use_search else "raw_policy",
                sims=sims if use_search else None,
                clairvoyant=bool(use_search),
                reveal_bot=bool(reveal_bot),
                partial=aborted,
            )
            stem = f"{time.strftime('%Y%m%dT%H%M%S')}_seed{seed}"
            dest = Path(replay_dir) / f"{stem}.json"
            # save_replay refuses to overwrite; two games can share a
            # (second, seed) stem, and losing the second one is not acceptable.
            n = 1
            while dest.exists():
                dest = Path(replay_dir) / f"{stem}_{n}.json"
                n += 1
            replay_path = str(save_replay(replay, dest))
            print(f"  replay written -> {replay_path}", flush=True)
        except Exception as exc:  # never lose the game record over a write fault
            print(f"[WARN] replay could not be written: {exc!r}", flush=True)

    # ---- legacy four-field JSONL (DUAL-WRITE, never rewritten) ------------
    # ``bot_vp`` / ``human_vp`` stay TOTAL VP: the single most-cited artifact in
    # the project is this file, and re-pointing an existing key would silently
    # change the meaning of every line already written. New facts are NEW keys.
    if log_path:
        import json
        import time
        from pathlib import Path

        if you_vp >= 15 and you_vp > bot_vp:
            winner = "human"
        elif bot_vp >= 15 and bot_vp > you_vp:
            winner = "bot"
        else:
            winner = None
        record = {
            "ckpt": ckpt,
            "mode": "search" if use_search else "raw_policy",
            "sims": sims if use_search else None,
            "clairvoyant": bool(use_search),  # search reads hidden hand + future dice
            "seed": seed,
            "human_seat": human_seat,
            "winner": winner,
            "bot_vp": bot_vp,
            "human_vp": you_vp,
            "n_bot_moves": n_steps,
            "truncated": bool(truncated),
            "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            # New keys (additive). A revealed game is an analysis session, not a
            # strength read; games recorded before this key existed were played
            # with the bot's FULL hand on screen and must be read that way.
            "reveal_bot": bool(reveal_bot),
            # Information-regime marker. Games before this key were played
            # WITHOUT the knights / longest-road panel facts and without the
            # on-screen move log, i.e. with strictly less public information than
            # a real table gives — a missing key means regime 1, not this one.
            "hud": 2,
            # Finite-bank integrity. FALSE means the spec-009 invariant
            # `bank[R] + sum(hands[R]) == 19` broke during this game, so the bank
            # the bot OBSERVED was wrong and the game must never be scored.
            # Persisted because warning is not enough: the console line is one
            # flush and the HUD strip is a deque(maxlen=6) that evicts it. Without
            # this key a corrupt game is byte-indistinguishable from a clean one.
            # A MISSING key means the game predates the check, not that it passed.
            "bank_ok": bool(check_bank.ok[0]),
            # Pre-roll dev-card window. FALSE = neither seat had one, i.e. both
            # played post-roll only, which is the ruleset the champion trained
            # on. A MISSING key means the game predates this stopgap and the
            # HUMAN had a pre-roll window the bot structurally could not use
            # (env/masks.py emits only ROLL_DICE while roll_pending), so its
            # result is not attributable. See preroll-dev-cards-r1.md.
            "preroll": False,
            # TRUE = the driver loop was interrupted (window closed, ^C, crash)
            # and these artifacts are a salvage write, not a played-out game.
            # A MISSING key means the game predates the crash-safe write, when
            # an interrupted game produced NO record at all.
            "partial": bool(aborted),
            # Provenance. "ckpt" above is a path into a gitignored runs/ tree
            # under a keep_last_n rotation, so it may name different bytes
            # tomorrow, or none. NULL means it could not be read — never that
            # the tree or the checkpoint was clean.
            "git_sha": _git_sha(),
            "ckpt_sha256": _ckpt_sha256(ckpt),
            "replay_path": replay_path,
            "moves": move_log,
        }
        out = Path(log_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("a", encoding="utf-8") as fh:  # append: one JSON per game
            fh.write(json.dumps(record) + "\n")
        print(f"  game logged -> {out}", flush=True)

    # Hold the finished board (and the terminal result line) on screen. LAST,
    # after every artifact is on disk, so dismissing the window can never cost
    # the replay or the JSONL line.
    _hold_final_screen()


# ---------------------------------------------------------------------------
# Headless self-test (no display, no pygame window)
# ---------------------------------------------------------------------------


def self_test(sims: int, seed: int, *, ckpt: str | None = None) -> int:
    """Headless smoke: prove v8+search loads (if ckpt given), the bot produces a
    LEGAL move from a fresh AND a mid-game state, and a full game completes with
    the human seat auto-playing legal moves. Never opens pygame.

    Returns a process exit code (0 = pass).
    """
    from catan_rl.policy.board_geometry import build_geometry
    from catan_rl.policy.network import CatanPolicy
    from catan_rl.search.agent import SearchAgent
    from catan_rl.search.config import SearchConfig

    print("[self-test] building env (human seat = auto-play, headless)...", flush=True)
    HumanVsBotEnv = _build_human_env_class()

    # Load the real v8 policy if a ckpt is supplied + exists; else a fresh policy
    # (structural test — legality/flow don't depend on trained weights, and the
    # 361MB load is skipped in CI / the default smoke).
    use_ckpt = ckpt is not None
    from pathlib import Path

    if use_ckpt and not Path(cast(str, ckpt)).expanduser().exists():
        print(f"[self-test] ckpt {ckpt} not found; using a fresh policy.", flush=True)
        use_ckpt = False
    if use_ckpt:
        print(f"[self-test] (1) loading v8+search from {ckpt} ...", flush=True)
        agent = _load_search_agent(cast(str, ckpt), sims, seed)
    else:
        print("[self-test] (1) building a fresh-weights SearchAgent ...", flush=True)
        policy = CatanPolicy()
        policy.set_board_geometry(build_geometry().as_dict_of_tensors())
        policy.eval()
        agent = SearchAgent(policy, SearchConfig(sims_per_move=sims, seed=seed))

    # (2a) fresh-state legal move ------------------------------------------------
    env: Any = HumanVsBotEnv(opponent_type="heuristic", max_turns=400)
    env._auto_human = True
    env.reset(seed=seed, options={"agent_seat": 0})
    masks = env.get_action_masks()
    action = agent.choose_action(env)
    assert action.shape == (6,) and action.dtype == np.int64, "bad action shape/dtype"
    assert bool(masks["type"][int(action[0])]), "fresh-state bot move is ILLEGAL"
    print(f"[self-test] (2a) fresh-state bot move legal: {_describe_bot_move(action)}", flush=True)

    # (2b) mid-game legal move ---------------------------------------------------
    # Step a few bot turns to reach a genuine mid-game decision, then re-check.
    mid_ok = False
    for _ in range(40):
        _o, _r, term, trunc, _i = env.step(action)
        _assert_bank_conservation(env)
        if term or trunc:
            break
        # Re-evaluate the mask BEFORE choosing so it matches the chosen action.
        masks = env.get_action_masks()
        action = agent.choose_action(env)
        assert bool(masks["type"][int(action[0])]), "mid-game bot move is ILLEGAL"
        if not env.initial_placement_phase and not env.roll_pending:
            mid_ok = True
            break
    assert mid_ok, "never reached a mid-game main-phase decision"
    print(f"[self-test] (2b) mid-game bot move legal: {_describe_bot_move(action)}", flush=True)

    # (3) full game to terminal --------------------------------------------------
    env2: Any = HumanVsBotEnv(opponent_type="heuristic", max_turns=200)
    env2._auto_human = True
    env2.reset(seed=seed + 1, options={"agent_seat": 0})
    term = trunc = False
    n = 0
    cap = env2.max_turns * 50
    while not term and not trunc and n < cap:
        a = agent.choose_action(env2)
        _o, _r, term, trunc, _i = env2.step(a)
        _assert_bank_conservation(env2)
        n += 1
    assert env2.agent_player is not None and env2.opponent_player is not None
    bot_vp = int(env2.agent_player.victoryPoints)
    human_vp = int(env2.opponent_player.victoryPoints)
    print(
        f"[self-test] (3) full game completed in {n} bot-steps "
        f"(terminated={term}, truncated={trunc}); "
        f"final VP bot={bot_vp} human={human_vp}",
        flush=True,
    )
    assert term or trunc, "game neither terminated nor truncated within cap"

    # (4) rules-invariants audit on the completed game ---------------------------
    from catan_rl.eval.rules_invariants import run_all_invariants

    assert env2.game is not None
    violations = run_all_invariants(env2.game, truncated=trunc)
    assert not violations, f"rules invariants violated: {violations}"
    print(f"[self-test] (4) rules invariants clean ({len(violations)} violations).", flush=True)
    print("[self-test] PASS", flush=True)
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Play 1v1 Catan (Colonist.io ruleset) as a human against a v2 policy "
            "checkpoint. Raw policy by default; the bot's hand is HIDDEN by default."
        ),
    )
    parser.add_argument(
        "--sims", type=int, default=DEFAULT_SIMS, help="MCTS sims/move (default 400)."
    )
    parser.add_argument(
        "--ckpt", type=str, default=DEFAULT_CKPT, help=f"Bot checkpoint (default: {DEFAULT_CKPT})."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "RNG seed. DEFAULT: a fresh random seed per game, so every session "
            "gets a different board and dice bag. Pass a value to replay an exact "
            "game. The seed actually used is always printed and recorded, so a "
            "random game stays reproducible after the fact."
        ),
    )
    parser.add_argument(
        "--search",
        action="store_true",
        help="Enable PUCT-MCTS. OFF by default and NOT VALID vs a human: the tree "
        "deep-copies the env, so it reads your hidden dev cards AND the true "
        "future dice. Only for bot-vs-bot work.",
    )
    parser.add_argument(
        "--log",
        type=str,
        default="runs/human_playtest/games.jsonl",
        help="Append a JSON record per game here (set empty to disable).",
    )
    parser.add_argument(
        "--human-seat",
        type=int,
        default=0,
        choices=(0, 1),
        help="Snake-draft seat for YOU: 0 = you go first (default), 1 = bot first.",
    )
    parser.add_argument(
        "--reveal-bot",
        action="store_true",
        help="Show the bot's FULL hand (resources + hidden dev cards by type + "
        "VP-card-inclusive VP). OFF by default. A revealed game is an ANALYSIS "
        "session, not a strength read — the flag is recorded in both the JSONL "
        "record and the replay metadata so it can never be misfiled later.",
    )
    parser.add_argument(
        "--replay-dir",
        type=str,
        default="runs/human_playtest/replays",
        help="Write a full-fidelity Replay JSON per game here (set empty to "
        "disable). NOTE: the four SETUP steps are SYNTHESIZED, not observed.",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Headless smoke (no display): verify load + legal bot moves + a full game.",
    )
    args = parser.parse_args(argv)

    if args.sims <= 0:
        parser.error("--sims must be > 0")

    if args.self_test:
        # Keep the smoke fast + display-free: a fresh policy unless --ckpt points
        # at a real file (use a small --sims for speed). The self-test stays
        # DETERMINISTIC by default — it is a smoke check, not a game.
        return self_test(args.sims, 0 if args.seed is None else args.seed, ckpt=args.ckpt)

    if args.seed is None:
        # A fresh seed per game. The old default of 0 meant every session replayed
        # the SAME board and the same StackedDice bag, which makes playtests
        # correlated and hides how the bot handles a different map.
        #
        # `secrets` rather than the clock: two games started in the same second
        # would collide on a time-derived seed, and second-resolution wastes most
        # of the space. This is not a security decision, just the cheapest source
        # of a well-spread integer.
        args.seed = secrets.randbelow(2**31)
        print(f"[seed] {args.seed}  (random; pass --seed {args.seed} to replay this game)")

    play_interactive(
        args.ckpt,
        args.sims,
        args.seed,
        args.human_seat,
        use_search=args.search,
        log_path=args.log or None,
        reveal_bot=args.reveal_bot,
        replay_dir=args.replay_dir or None,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
