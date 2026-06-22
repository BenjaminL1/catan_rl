#!/usr/bin/env python3
"""Play 1v1 Catan (Colonist.io ruleset) as a HUMAN against champion v8 + search.

The bot is the strongest available agent: champion ``v8_promobar_u243`` wrapped in
determinized PUCT-MCTS (``SearchAgent``) at a high simulation budget (default 400).
You play through the existing pygame board GUI (mouse + the on-screen buttons),
exactly as the engine's human-render mode already supports.

Run it (a display is required for the real game)::

    python scripts/play_vs_v8.py --sims 400

Headless smoke (no display, no pygame window — auto-plays a legal human move so
the full turn flow + win detection are exercised end-to-end)::

    python scripts/play_vs_v8.py --self-test

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
"""

from __future__ import annotations

import argparse
import sys
from typing import TYPE_CHECKING, Any, cast

import numpy as np

if TYPE_CHECKING:
    from catan_rl.search.agent import SearchAgent

DEFAULT_CKPT = "runs/anchors/v8_promobar_u243.pt"
DEFAULT_SIMS = 400


# ---------------------------------------------------------------------------
# Human-as-opponent env: the bot is the agent seat; the human is the internal
# opponent, driven via a real pygame view temporarily swapped onto game.boardView.
# ---------------------------------------------------------------------------


def _build_human_env_class() -> type:
    """Build the ``HumanVsBotEnv`` subclass lazily.

    Defined inside a function so importing this module (e.g. for ``--self-test``,
    or by a test) never forces the ``CatanEnv`` import at module load. The class
    overrides ONLY the four internal opponent hooks; everything else (setup snake
    draft, roll/robber/discard sub-phases, turn folding, obs/masks, the
    engine/RESOURCES_CW mapping) is inherited unchanged.
    """
    from catan_rl.engine.game import _HeadlessView
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
                # Auto-play (self-test): pick a legal setup vertex + road.
                v = next(iter(board.get_setup_settlements(human).keys()))
                human.build_settlement(v, board, is_free=True)
                r = next(iter(board.get_setup_roads(human).keys()))
                human.build_road(r[0], r[1], board, is_free=True)
                return
            prev_setup = game.gameSetup
            game.gameSetup = True  # makes the view's setup-mode click loop apply
            try:
                with self._ViewWindow(self) as view:
                    print("\n[SETUP] Your move: place a SETTLEMENT (click a circle).", flush=True)
                    v = view.buildSettlement_display(human, board.get_setup_settlements(human))
                    if v is not None:
                        human.build_settlement(v, board, is_free=True)
                    view.displayGameScreen()
                    print("[SETUP] Now place an adjacent ROAD (click a line).", flush=True)
                    r = view.buildRoad_display(human, board.get_setup_roads(human))
                    if r is not None:
                        human.build_road(r[0], r[1], board, is_free=True)
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
                spots = board.get_robber_spots()
                hex_i = next(iter(spots.keys()))
                victims = board.get_players_to_rob(hex_i)
                victim = next((p for p in victims if p is not human), None)
                human.move_robber(hex_i, board, victim)
                game.check_largest_army(human)
                return
            with self._ViewWindow(self) as view:
                print(
                    "\n[ROBBER] Move the robber: click a hex, then a player to steal from.",
                    flush=True,
                )
                hex_i, victim = view.moveRobber_display(human, board.get_robber_spots())
                human.move_robber(hex_i, board, victim)
            game.check_largest_army(human)

        def _opponent_discard(self) -> None:
            """Human discards on a 7-roll (>9 cards) via the existing engine path."""
            game: Any = self.game
            assert game is not None
            human = self._human_player()
            if not self._use_gui():
                # Engine's heuristic/plain discard works headless (no GUI calls).
                human.discardResources(game)
                return
            n_before = sum(human.resources.values())
            print(
                f"\n[DISCARD] You rolled into a 7 with {n_before} cards — discard half.",
                flush=True,
            )
            with self._ViewWindow(self):
                # discardResources(self) drives game.boardView.get_resource_selection.
                human.discardResources(game)

        def _run_opponent_main_turn(self) -> None:
            """The human's full main turn (dice already rolled by the env caller)."""
            game: Any = self.game
            assert game is not None
            human = self._human_player()
            board: Any = game.board
            if not self._use_gui():
                # Self-test: end the turn immediately (a legal no-op main turn).
                game.check_longest_road(human)
                game.check_largest_army(human)
                return
            with self._ViewWindow(self) as view:
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

            human.updateDevCards()
            human.devCardPlayedThisTurn = False
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
    """Load champion v8 on CPU and wrap it in determinized PUCT-MCTS."""
    from catan_rl.replay.player_factory import PlayerSpec, _PolicyActor, build_actor
    from catan_rl.search.agent import SearchAgent
    from catan_rl.search.config import SearchConfig

    actor = cast(
        "_PolicyActor",
        build_actor(PlayerSpec(kind="policy", ckpt_path=ckpt), seed=seed, device="cpu"),
    )
    cfg = SearchConfig(sims_per_move=sims, seed=seed)
    return SearchAgent(actor.policy, cfg, device=actor.device)


def _describe_bot_move(action: np.ndarray) -> str:
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
    return label


# ---------------------------------------------------------------------------
# Interactive game (real GUI)
# ---------------------------------------------------------------------------


def play_interactive(ckpt: str, sims: int, seed: int, human_seat: int) -> None:
    """Run an interactive human-vs-bot game with the pygame GUI."""
    from catan_rl.gui.view import catanGameView as _catanGameView

    catanGameView: Any = _catanGameView
    HumanVsBotEnv = _build_human_env_class()
    # The BOT is the env "agent" seat; the HUMAN is the env "opponent" seat.
    # agent_seat selects who acts first in the snake draft: bot_seat = 1 - human_seat.
    bot_seat = 1 - human_seat

    print("=" * 70, flush=True)
    print("  1v1 Catan (Colonist.io ruleset) — YOU vs champion v8 + search", flush=True)
    print("=" * 70, flush=True)
    print(f"  Bot: {ckpt}  (PUCT-MCTS, {sims} sims/move, CPU)", flush=True)
    print("  Win = 15 VP. No player-to-player trading (bank/port only).", flush=True)
    print("  Discard threshold = 9 cards. Friendly Robber in effect.", flush=True)
    print(f"  You are {'FIRST' if human_seat == 0 else 'SECOND'} in the snake draft.", flush=True)
    print("-" * 70, flush=True)
    print("  HOW TO PLAY (click with the mouse):", flush=True)
    print("   - Setup: click a highlighted circle (settlement) then a line (road).", flush=True)
    print("   - Your turn: dice auto-roll; use the on-screen buttons. Click END", flush=True)
    print("     TURN to pass to the bot. The bot then 'thinks' (search) and moves.", flush=True)
    print("   - On a 7: discard menu pops up (>9 cards); then move robber + steal.", flush=True)
    print("=" * 70, flush=True)

    agent = _load_search_agent(ckpt, sims, seed)
    env: Any = HumanVsBotEnv(opponent_type="heuristic", max_turns=400)
    # Register the view builder BEFORE reset. When the human drafts first
    # (bot_seat==1), the env places the human's FIRST settlement inside reset();
    # the lazy factory makes that placement use the GUI instead of auto-picking,
    # so NOTHING is auto-placed for either player (the bot's placements always
    # come from search via the loop). The view is never stored on game.boardView
    # except inside a human input window (deepcopy-safe for MCTS).
    env.set_view_factory(lambda game: catanGameView(game.board, game))
    env.reset(seed=seed, options={"agent_seat": bot_seat})
    # Ensure a view exists for rendering (already built during reset if the human
    # drafted first; built here otherwise — board is fixed at reset).
    assert env.game is not None
    view: Any = env._ensure_view()
    view.displayGameScreen()

    terminated = truncated = False
    safety_cap = env.max_turns * 50
    n_steps = 0
    while not terminated and not truncated:
        # Bot (agent seat) decides + applies its action; this also folds the
        # human's whole turn internally via the overridden _opponent_* hooks.
        action = agent.choose_action(env)
        print(f"\n[BOT] {_describe_bot_move(action)}", flush=True)
        _obs, _r, terminated, truncated, _info = env.step(action)
        view.displayGameScreen()
        assert env.agent_player is not None and env.opponent_player is not None
        print(
            f"  Bot VP={env.agent_player.victoryPoints} | "
            f"You VP={env.opponent_player.victoryPoints}",
            flush=True,
        )
        n_steps += 1
        if n_steps > safety_cap:
            print("[WARN] safety cap hit; ending.", flush=True)
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
        description="Play 1v1 Catan as a human vs champion v8 + PUCT-MCTS search.",
    )
    parser.add_argument(
        "--sims", type=int, default=DEFAULT_SIMS, help="MCTS sims/move (default 400)."
    )
    parser.add_argument("--ckpt", type=str, default=DEFAULT_CKPT, help="v8 checkpoint path.")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed (reproducible search).")
    parser.add_argument(
        "--human-seat",
        type=int,
        default=0,
        choices=(0, 1),
        help="Snake-draft seat for YOU: 0 = you go first (default), 1 = bot first.",
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
        # at a real file (use a small --sims for speed).
        return self_test(args.sims, args.seed, ckpt=args.ckpt)

    play_interactive(args.ckpt, args.sims, args.seed, args.human_seat)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
