# Settlers of Catan
# Game view class implementation with pygame

import math
import sys

import pygame

from catan_rl.engine.geometry import *
from catan_rl.gui import render
from catan_rl.gui import render_constants as RC
from catan_rl.policy.obs_schema import DEV_DECK_INITIAL

pygame.init()

#: Leading (px) between hand-panel lines. 18, not 20: the revealed panel is 16
#: lines and at 20px a REVEALED bot panel (drawn at y=460) would run past the
#: 800px window bottom.
HAND_PANEL_LINE_HEIGHT = 18

#: Recent-move log strip along the bottom of the board, as (x, y, w, h): the
#: right rail is fully allocated, so the log goes here. x=115 clears the END TURN
#: button (x 20-100) and 705px of width fits "Bot: Build settlement at v23".
MOVE_LOG_RECT = (115, 695, 705, 100)
MOVE_LOG_LINE_HEIGHT = 15
#: How many log lines the strip can show. The window is 1000x800 and the right
#: rail is fully allocated, so the ~12 lines originally wished for do not fit;
#: 6 wide lines covers a typical bot turn (roll, 1-3 actions, end turn).
MOVE_LOG_LINES = 6

#: Hidden-dev-card key -> short label used by the REVEALED hand panel.
_DEV_CARD_DISPLAY = {
    "KNIGHT": "Knight",
    "VP": "VP",
    "MONOPOLY": "Mono",
    "ROADBUILDER": "RB",
    "YEAROFPLENTY": "YOP",
}


def _played_dev_total(player) -> int:
    """Publicly-played dev cards for ``player`` (the counters the obs exposes)."""
    return (
        int(getattr(player, "knightsPlayed", 0))
        + int(getattr(player, "yopPlayed", 0))
        + int(getattr(player, "monopolyPlayed", 0))
        + int(getattr(player, "roadBuilderPlayed", 0))
    )


def dev_card_button_enabled(player, board) -> bool:
    """Is BUY DEV CARD actually usable right now?

    The FULL predicate the bot's mask uses (``env/masks.py``): a non-empty deck
    AND the 1 ORE / 1 WHEAT / 1 SHEEP cost. ``player.draw_devCard`` no-ops on
    either half without spending anything, so greying the button is a UX fix, not
    a correctness one — but the human should not have to learn the rule by
    clicking a gold button that does nothing.
    """
    deck_total = sum((getattr(board, "devCardStack", {}) or {}).values())
    res = getattr(player, "resources", {}) or {}
    can_afford = all(res.get(r, 0) >= 1 for r in ("ORE", "WHEAT", "SHEEP"))
    return bool(deck_total > 0 and can_afford)


def public_dev_deck_remaining(player, opponent) -> int:
    """Dev cards the human can honestly deduce are still unseen.

    Same public-reveal-derived formula the bot observes
    (``policy/obs_encoder._build_global_features``), summed over types:
    ``25 - own_held - own_played - opponent_played``. Deliberately NOT
    ``board.devCardStack``, which is deck TRUTH and would tell the human how many
    cards sit in the opponent's hidden hand — information the bot does not get.
    """
    total = sum(DEV_DECK_INITIAL)
    dev = getattr(player, "devCards", {}) or {}
    new_dev = getattr(player, "newDevCards", []) or []
    own_held = sum(int(v) for v in dev.values()) + len(new_dev)
    played = _played_dev_total(player)
    if opponent is not None:
        played += _played_dev_total(opponent)
    return max(0, total - own_held - played)


def _public_progress_lines(player, board) -> list[str]:
    """Knights played + current longest-road length — PUBLIC in BOTH panels.

    Knights are played face up and roads sit on the board, so an opponent at a
    real table sees both. Withholding them from the human while the bot reads
    them straight off the state makes the playtest HARDER than real Catan and
    biases every result toward the bot — the same error as over-blinding
    DISCARD/YOP. They are therefore emitted in the blind panel too.

    The road length is computed LIVE from ``board`` when one is supplied
    (``player.get_road_length(board)``) instead of read off the
    ``maxRoadLength`` cache. That cache is in fact fresh on every traced
    mutating path, so this is belt-and-braces rather than a bug fix; it is
    affordable because the panel renders once per pick, not once per frame
    (``_animated_pick`` caches the base surface). ``board=None`` (pure /
    duck-typed callers) falls back to the cache.

    ``Knights played:`` deliberately avoids the ``Knight:`` label used by the
    REVEALED dev-card block, so the blind panel still exposes no dev-card TYPE
    string (pinned in ``tests/unit/gui/test_hand_panel.py``).
    """
    if board is not None:
        road_length = player.get_road_length(board)
    else:
        road_length = getattr(player, "maxRoadLength", 0)
    return [
        f"Knights played: {player.knightsPlayed}",
        f"Longest road: {road_length}",
    ]


def hand_panel_lines(player, *, reveal: bool = True, board=None) -> list[str]:
    """Return the text lines of a hand panel for ``player``.

    Pure (no pygame): the rendering in :meth:`catanGameView._draw_hand_panel`
    only lays these out. An empty string is a blank spacer line.

    ``reveal=True`` is the OMNISCIENT view (every resource type, the
    VP-card-inclusive total, and hidden dev cards by type). It is correct for
    the panel showing a player their OWN hand, and is opt-in analysis output
    for anyone else's.

    ``reveal=False`` is what an opponent may legally see in Catan: the hand
    SIZE, the number of unplayed dev cards, and the VISIBLE victory points
    ``victoryPoints - devCards["VP"]``. There is no ``newDevCards`` term —
    a bought VP card scores immediately and never passes through that bucket
    (``engine/player.py``), which is also how ``policy/obs_encoder.py``
    computes it. ``player.visibleVictoryPoints`` is deliberately NOT used:
    it is a stale cache refreshed only at init + VP-card buy.

    BOTH branches also carry the PUBLIC progress facts (knights played, current
    longest-road length) — see :func:`_public_progress_lines`. ``board`` is an
    OPTIONAL, backwards-compatible kwarg: pass it to compute the road length
    live, omit it to fall back to ``player.maxRoadLength``.
    """
    if not reveal:
        visible_vp = player.victoryPoints - player.devCards.get("VP", 0)
        n_dev = sum(player.devCards.values()) + len(player.newDevCards)
        return [
            f"Cards: {sum(player.resources.values())}",
            "",
            f"Victory Points: {visible_vp}",
            *_public_progress_lines(player, board),
            "",
            f"Dev Cards: {n_dev}",
        ]

    lines = [f"{resource}: {count}" for resource, count in player.resources.items()]
    lines += ["", f"Victory Points: {player.victoryPoints}"]
    lines += _public_progress_lines(player, board)
    lines += ["", "Dev Cards:"]
    total_dev_cards = dict(player.devCards)
    for card in player.newDevCards:
        total_dev_cards[card] = total_dev_cards.get(card, 0) + 1
    lines += [
        f"{display_name}: {total_dev_cards.get(card_type, 0)}"
        for card_type, display_name in _DEV_CARD_DISPLAY.items()
    ]
    return lines


def broadcast_message(event, *, name_display=None, blind_player=None):
    """Return ``(text, rgb)`` for the last broadcast event, or ``None``.

    Pure (no pygame): :meth:`catanGameView.displayBroadcastMessage` only draws
    the result.

    ``blind_player`` is accepted for signature stability but no longer changes any
    banner: **DISCARD and YOP resource TYPES are PUBLIC and are never blinded.**
    The engine says so itself — ``tracker.track_steal``: "It is Public Information
    relative to the two players" — and the bot reads this same broadcast stream
    through a ``BroadcastHandTracker`` that does perfect opponent hand-tracking
    (``env/catan_env.py``). Hiding these from the human while the bot tracks them
    would make the playtest HARDER than the real game and bias every result in the
    bot's favour. The leak the blind hand panel closes is the bot's HAND CONTENTS,
    which are private; public events stay public.
    """
    name_display = name_display or {}
    event_type = event.get("type", "")
    raw_name = event.get("player", "")
    player_name = name_display.get(raw_name, raw_name)

    if event_type == "DICE_ROLL":
        return f"Dice: {player_name} rolled {event.get('value', 0)}", (0, 0, 0)
    if event_type == "DISCARD":
        return f"DISCARD: {player_name} lost {event.get('resources', [])}", (255, 0, 0)
    if event_type == "YOP":
        return f"YOP: {player_name} gained {event.get('resources', [])}", (0, 100, 0)
    return None


# Class to handle catan board display


class catanGameView:
    "Class definition for Catan board display"

    def __init__(self, catanBoardObject, catanGameObject):
        self.board = catanBoardObject
        self.game = catanGameObject

        # #Use pygame to display the board
        self.screen = pygame.display.set_mode(self.board.size)
        pygame.display.set_caption("Settlers of Catan")
        self.font_resource = pygame.font.SysFont("cambria", 15)
        self.font_ports = pygame.font.SysFont("cambria", 10)

        self.font_button = pygame.font.SysFont("cambria", 12)
        self.font_diceRoll = pygame.font.SysFont("cambria", 25)  # dice font
        self.font_Robber = pygame.font.SysFont("arialblack", 50)  # robber font
        self.font_menu = pygame.font.SysFont("cambria", 20)
        self.font_broadcast = pygame.font.SysFont("cambria", 18)  # broadcast font
        self.font_movelog = pygame.font.SysFont("cambria", 13)  # recent-move strip

        self.diceRoll = 0  # Initialize dice roll

        # Optional whose-turn banner: (text, bg_color_name) or None. Drawn by
        # displayGameScreen only when set (additive — engine paths leave it None
        # so rendering is unchanged). Set by interactive harnesses.
        self.turn_banner: tuple[str, str] | None = None

        # Optional: the human's player object. When set, displayPlayerStats always
        # shows THIS player's hand (the human sits in the AI-flagged opponent seat
        # in the vs-bot harness, so the default current-non-AI logic would hide it).
        # Engine playCatan leaves it None -> unchanged behavior.
        self.human_player = None

        # Optional: the bot/opponent player object. When set, displayPlayerStats
        # also shows the OPPONENT's panel. BLIND by default (hand SIZE, unplayed
        # dev-card COUNT and VISIBLE VP only — real Catan visibility); set
        # ``reveal_bot`` to show the full hand for post-hoc analysis. Engine
        # playCatan leaves both untouched.
        self.bot_player = None

        # Optional: reveal the bot's FULL hand (resources by type, hidden dev
        # cards by type, VP-card-inclusive VP). OFF by default — a revealed game
        # is not a valid strength read, so harnesses that flip this must record
        # the fact alongside the game.
        self.reveal_bot: bool = False

        # Optional: friendly display-name overrides keyed by player.name (e.g.
        # {"Opponent": "You", "Agent": "Bot"}). Used by displayPlayerStats and
        # displayBroadcastMessage. Engine playCatan leaves it empty -> raw names.
        self.name_display: dict[str, str] = {}

        # Optional: recent-move log rendered in the bottom strip. A sequence of
        # plain strings, OLDEST FIRST — the harness owns it (see
        # ``scripts/play_vs_model.py``), typically a
        # ``collections.deque(maxlen=MOVE_LOG_LINES)``. Deliberately NOT an
        # ``EventCollector``: that drain is destructive and single-consumer, and
        # the replay recorder already owns the only one. Engine playCatan leaves
        # this None -> nothing is drawn.
        self.move_log = None

        # Optional: restrict which dev cards ``get_dev_card_selection`` will
        # accept, as a set of engine ``devCards`` keys. ``None`` (the default,
        # and what engine playCatan leaves it at) means UNRESTRICTED — the
        # historic behaviour, unchanged. The pre-roll window in
        # ``scripts/play_vs_model.py`` sets it to the human's legal pre-roll
        # options so the menu offers exactly what the bot's action mask offers
        # (Road Builder is excluded pre-roll; see ``env/ruleset.py``).
        #
        # It lives on the VIEW rather than as a ``play_devCard`` argument
        # because ``engine/player.play_devCard`` calls
        # ``boardView.get_dev_card_selection(self)`` with one argument and the
        # engine is out of scope for that slice. It restricts the MENU, never
        # the hand: nothing zeroes ``player.devCards``, so a mid-window quit
        # cannot capture or leak a card and the on-screen count stays truthful.
        self.dev_card_filter = None

        return None

    # Function to display the initial board

    def displayInitialBoard(self):
        # Back-compat: external code may still read ``self.colorDict``.
        self.colorDict = dict(RC.TILE_COLORS)

        render.draw_water(self.screen, (self.board.width, self.board.height))
        hex_centers = [self.board.hexTileDict[i].to_pixel(self.board.flat) for i in range(19)]
        render.draw_island_outline(self.screen, hex_centers)

        for hexTile in self.board.hexTileDict.values():
            render.draw_hex_tile(self.screen, hexTile, self.board, with_bevel=True)
            center = hexTile.to_pixel(self.board.flat)
            cx_int = int(center.x)
            cy_int = int(center.y)
            render.draw_resource_symbol(
                self.screen,
                (cx_int, cy_int + RC.RESOURCE_SYMBOL_VERTICAL_OFFSET),
                hexTile.resource_type,
            )
            num = getattr(hexTile, "number_token", None)
            if num is not None and num != 0 and hexTile.resource_type != "DESERT":
                render.draw_number_token(
                    self.screen,
                    (cx_int, cy_int + RC.NUMBER_TOKEN_VERTICAL_OFFSET),
                    num,
                )

        self.displayPorts()

        pygame.display.update()

        return None

    def displayPorts(self):
        """Draw all nine ports (planks + ship + ratio badge).

        Split out of ``displayInitialBoard`` so ``displayGameScreen`` can repaint
        the ports AFTER the move-log strip, exactly as it repaints the buildings
        loop: two ports anchor inside ``MOVE_LOG_RECT`` (the 2:1 WHEAT ship at
        (541, 759) and a 3:1 generic at (296, 751)), and their planks run to
        vertices at y=720. Port access is a first-order PUBLIC planning fact, so
        the strip must not erase it. Idempotent — the seven ports outside the
        strip redraw identically."""
        board_cx = self.board.width / 2.0
        board_cy = self.board.height / 2.0
        vertex_pixel = self.board.vertex_index_to_pixel_dict
        for v1_idx, v2_idx, ratio, resource in render.collect_port_edges(self.board):
            v1_px = vertex_pixel[v1_idx]
            v2_px = vertex_pixel[v2_idx]
            mid_x = (float(v1_px.x) + float(v2_px.x)) / 2.0
            mid_y = (float(v1_px.y) + float(v2_px.y)) / 2.0
            dx = mid_x - board_cx
            dy = mid_y - board_cy
            # Guard the centroid-degenerate case; ports near the board
            # center are impossible by engine construction but cheap to
            # defend against.
            d = math.hypot(dx, dy) or 1.0
            ax = int(mid_x + dx * RC.PORT_PUSH_DISTANCE / d)
            ay = int(mid_y + dy * RC.PORT_PUSH_DISTANCE / d)
            render.draw_port_planks(
                self.screen,
                (ax, ay),
                (int(v1_px.x), int(v1_px.y)),
                (int(v2_px.x), int(v2_px.y)),
            )
            render.draw_port_ship(self.screen, ratio, resource, (ax, ay))

        return None

    # Function to draw a road on the board

    def draw_road(self, edgeToDraw, roadColor):
        pygame.draw.line(self.screen, pygame.Color(roadColor), edgeToDraw[0], edgeToDraw[1], 10)

    # Function to draw a potential road on the board - thin

    def draw_possible_road(self, edgeToDraw, roadColor):
        roadRect = pygame.draw.line(
            self.screen, pygame.Color(roadColor), edgeToDraw[0], edgeToDraw[1], 5
        )
        return roadRect

    # Function to draw a settlement on the board at vertexToDraw

    def draw_settlement(self, vertexToDraw, color):
        newSettlement = pygame.Rect(vertexToDraw.x - 10, vertexToDraw.y - 10, 25, 25)
        pygame.draw.rect(self.screen, pygame.Color(color), newSettlement)

    # Function to draw a potential settlement on the board - thin

    def draw_possible_settlement(self, vertexToDraw, color):
        possibleSettlement = pygame.draw.circle(
            self.screen, pygame.Color(color), (int(vertexToDraw.x), int(vertexToDraw.y)), 20, 3
        )
        return possibleSettlement

    # Function to draw a settlement on the board at vertexToDraw

    def draw_city(self, vertexToDraw, color):
        pygame.draw.circle(
            self.screen, pygame.Color(color), (int(vertexToDraw.x), int(vertexToDraw.y)), 24
        )

    # Function to draw a potential settlement on the board - thin

    def draw_possible_city(self, vertexToDraw, color):
        possibleCity = pygame.draw.circle(
            self.screen, pygame.Color(color), (int(vertexToDraw.x), int(vertexToDraw.y)), 25, 5
        )
        return possibleCity

    # Function to draw the possible spots for a robber

    def draw_possible_robber(self, vertexToDraw):
        possibleRobber = pygame.draw.circle(
            self.screen, pygame.Color("black"), (int(vertexToDraw.x), int(vertexToDraw.y)), 50, 5
        )
        return possibleRobber

    # Function to draw possible players to rob
    def draw_possible_players_to_rob(self, vertexCoord):
        possiblePlayer = pygame.draw.circle(
            self.screen, pygame.Color("black"), (int(vertexCoord.x), int(vertexCoord.y)), 35, 5
        )
        return possiblePlayer

    # ------------------------------------------------------------------
    # Animated highlight helpers (Colonist-style pulsating glow on the
    # spots the current player can act on). Additive: only used by the
    # interactive picker loops below.
    # ------------------------------------------------------------------

    def _pulse(self) -> float:
        """0..1 sine pulse keyed to the wall clock (≈0.8 Hz)."""
        return (math.sin(pygame.time.get_ticks() * 0.006) + 1.0) / 2.0

    def _blit_halo(self, x: int, y: int, radius: int, color: str, alpha: int) -> None:
        """Blit a translucent filled circle (a soft glow) centered at (x, y)."""
        radius = max(1, radius)
        surf = pygame.Surface((radius * 2 + 4, radius * 2 + 4), pygame.SRCALPHA)
        c = pygame.Color(color)
        c.a = max(0, min(255, alpha))
        pygame.draw.circle(surf, c, (radius + 2, radius + 2), radius)
        self.screen.blit(surf, (x - radius - 2, y - radius - 2))

    # Available-spot highlights use a bright gold glow (high contrast on the
    # board and on the dark player colors) with a player-color accent ring. The
    # returned click Rect is a STABLE size so clicks register regardless of pulse.
    _GLOW = "gold"

    def _glow_settlement(self, vertex, color, pulse):
        x, y = int(vertex.x), int(vertex.y)
        self._blit_halo(x, y, int(22 + 12 * pulse), self._GLOW, int(60 + 150 * pulse))
        pygame.draw.circle(self.screen, pygame.Color(color), (x, y), 18, 4)
        return pygame.draw.circle(self.screen, pygame.Color(self._GLOW), (x, y), 20, 3)

    def _glow_city(self, vertex, color, pulse):
        x, y = int(vertex.x), int(vertex.y)
        self._blit_halo(x, y, int(27 + 13 * pulse), self._GLOW, int(60 + 150 * pulse))
        pygame.draw.circle(self.screen, pygame.Color(color), (x, y), 23, 5)
        return pygame.draw.circle(self.screen, pygame.Color(self._GLOW), (x, y), 25, 3)

    def _glow_robber(self, vertexCoord, pulse):
        x, y = int(vertexCoord.x), int(vertexCoord.y)
        self._blit_halo(x, y, int(48 + 16 * pulse), self._GLOW, int(55 + 130 * pulse))
        return pygame.draw.circle(self.screen, pygame.Color("black"), (x, y), 50, 5)

    def _glow_road(self, edge, color, pulse):
        p0, p1 = edge[0], edge[1]
        glow_w = int(10 + 8 * pulse)  # pulsing bright underlay
        rect = pygame.draw.line(self.screen, pygame.Color(self._GLOW), p0, p1, glow_w)
        pygame.draw.line(self.screen, pygame.Color(color), p0, p1, max(2, glow_w // 2))
        return rect

    def _animated_pick(self, spots, draw_fn, allow_cancel):
        """Render the base board once, then each frame restore it and draw a
        pulsating glow on every ``spots`` entry (``draw_fn(spot, pulse)`` returns
        the click Rect). Blocks until the user clicks a spot (returns it) or — if
        ``allow_cancel`` — clicks empty space (returns None). QUIT exits.
        """
        self.displayGameScreen()
        base = self.screen.copy()  # cache the static board for cheap per-frame redraws
        clock = pygame.time.Clock()
        while True:
            self.screen.blit(base, (0, 0))
            pulse = self._pulse()
            rects = {spot: draw_fn(spot, pulse) for spot in spots}
            pygame.display.update()
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    sys.exit(0)
                if e.type == pygame.MOUSEBUTTONDOWN:
                    for spot, rect in rects.items():
                        if rect.collidepoint(e.pos):
                            return spot
                    if allow_cancel:
                        return None
            clock.tick(30)

    # Function to render basic gameplay buttons

    def displayGameButtons(self):
        # Basic GamePlay Buttons
        diceRollText = self.font_button.render("ROLL DICE", False, (0, 0, 0))
        buildRoadText = self.font_button.render("ROAD", False, (0, 0, 0))
        buildSettleText = self.font_button.render("SETTLE", False, (0, 0, 0))
        buildCityText = self.font_button.render("CITY", False, (0, 0, 0))
        endTurnText = self.font_button.render("END TURN", False, (0, 0, 0))
        devCardText = self.font_button.render("DEV CARD", False, (0, 0, 0))
        playDevCardText = self.font_button.render("PLAY DEV", False, (0, 0, 0))
        tradeBankText = self.font_button.render("BANK TRADE", False, (0, 0, 0))

        self.rollDice_button = pygame.Rect(20, 10, 80, 40)
        self.buildRoad_button = pygame.Rect(20, 70, 80, 40)
        self.buildSettlement_button = pygame.Rect(20, 120, 80, 40)
        self.buildCity_button = pygame.Rect(20, 170, 80, 40)

        self.devCard_button = pygame.Rect(20, 300, 80, 40)
        self.playDevCard_button = pygame.Rect(20, 400, 80, 40)

        self.tradeBank_button = pygame.Rect(self.board.width - 125, 400, 100, 40)

        self.endTurn_button = pygame.Rect(20, 700, 80, 40)

        pygame.draw.rect(self.screen, pygame.Color("darkgreen"), self.rollDice_button)
        pygame.draw.rect(self.screen, pygame.Color("gray33"), self.buildRoad_button)
        pygame.draw.rect(self.screen, pygame.Color("gray33"), self.buildSettlement_button)
        pygame.draw.rect(self.screen, pygame.Color("gray33"), self.buildCity_button)
        # BUY DEV CARD is greyed on the FULL predicate (empty deck OR
        # unaffordable), matching the bot's mask; see dev_card_button_enabled.
        buyer = self._button_player()
        buy_enabled = buyer is not None and dev_card_button_enabled(buyer, self.board)
        pygame.draw.rect(
            self.screen,
            pygame.Color("gold") if buy_enabled else pygame.Color("gray50"),
            self.devCard_button,
        )
        pygame.draw.rect(self.screen, pygame.Color("gold"), self.playDevCard_button)
        pygame.draw.rect(self.screen, pygame.Color("magenta"), self.tradeBank_button)

        pygame.draw.rect(self.screen, pygame.Color("burlywood"), self.endTurn_button)

        self.screen.blit(diceRollText, (30, 20))
        self.screen.blit(buildRoadText, (30, 80))
        self.screen.blit(buildSettleText, (30, 130))
        self.screen.blit(buildCityText, (30, 180))
        self.screen.blit(devCardText, (30, 310))
        self.screen.blit(playDevCardText, (30, 410))
        self.screen.blit(tradeBankText, (self.board.width - 115, 410))

        # Dev-deck remaining, in the human's own units: the same public-reveal-
        # derived figure the bot observes, NOT deck truth. The bot sees this and
        # the human saw nothing.
        if buyer is not None:
            deck_left = public_dev_deck_remaining(buyer, self._other_player(buyer))
            self.screen.blit(
                self.font_resource.render(f"Deck: {deck_left}", False, (0, 0, 0)), (20, 348)
            )

        self.screen.blit(endTurnText, (30, 710))

    def _button_player(self):
        """The seat the left-rail buttons act for, or None if undetermined.

        Same resolution order as ``displayPlayerStats``: the harness pins
        ``human_player`` (the human sits in the AI-flagged opponent seat); the
        engine's own ``playCatan`` loop leaves it None and the current player is
        the one clicking.
        """
        if self.human_player is not None:
            return self.human_player
        return getattr(self.game, "currentPlayer", None)

    def _other_player(self, player):
        """The opponent of ``player`` in the 2-player queue, or None."""
        queue = getattr(getattr(self.game, "playerQueue", None), "queue", None)
        if queue is None:
            return None
        for p in list(queue):
            if p is not player:
                return p
        return None

    # Function to display robber

    def displayRobber(self):
        for hexTile in self.board.hexTileDict.values():
            if hexTile.has_robber:
                render.draw_robber_pawn(self.screen, hexTile, self.board)
                return

    def displayPlayerStats(self):
        # Harness sets human_player so the human (who sits in the AI-flagged
        # opponent seat) always sees their OWN hand. Engine playCatan leaves it
        # None and falls back to the current non-AI player.
        player = self.human_player
        if player is None:
            if not hasattr(self.game, "currentPlayer") or self.game.currentPlayer is None:
                return
            player = self.game.currentPlayer
            if player.isAI:
                return

        label = self.name_display.get(player.name, player.name)
        title = f"YOUR HAND ({label})" if self.human_player is not None else f"Player: {label}"
        self._draw_hand_panel(player, self.board.width - 160, 15, title)

        # vs-bot view: the OPPONENT's panel. BLIND by default (hand size, unplayed
        # dev-card count, VISIBLE VP) so playing against the bot is a fair game;
        # ``reveal_bot`` opts into the omniscient ANALYSIS view. Only shown when
        # the harness sets bot_player; engine playCatan leaves it None (so normal
        # play keeps the opponent's hand hidden either way).
        if self.bot_player is not None:
            bot_label = self.name_display.get(self.bot_player.name, self.bot_player.name)
            suffix = "FULL HAND (REVEALED)" if self.reveal_bot else "HAND"
            self._draw_hand_panel(
                self.bot_player,
                self.board.width - 160,
                460,
                f"{bot_label} — {suffix}",
                reveal=self.reveal_bot,
            )

    def _draw_hand_panel(self, player, x, y, title, *, reveal=True):
        """Render a hand panel for ``player`` at (x, y) with a readable backdrop.

        ``reveal`` is forwarded to :func:`hand_panel_lines`, which owns the
        content (and the reveal/blind distinction); this method only lays it out.
        Defaults to ``True`` so existing call sites (a player's OWN hand) are
        unchanged. ``self.board`` is forwarded so the longest-road figure is
        computed live. Returns the panel ``Rect`` so the layout arithmetic is
        testable; no caller reads it.

        The 18px leading is load-bearing, not cosmetic: the revealed panel is 16
        lines, and at the old 20px a REVEALED BOT panel (``--reveal-bot``, drawn
        at y=460) would extend to y=824 in an 800px window."""
        lines = hand_panel_lines(player, reveal=reveal, board=self.board)
        line_height = HAND_PANEL_LINE_HEIGHT
        panel_lines = len(lines) + 2  # title + a trailing margin line
        panel = pygame.Rect(x - 12, y - 8, 156, line_height * panel_lines + 12)
        backdrop = pygame.Surface((panel.width, panel.height), pygame.SRCALPHA)
        backdrop.fill((245, 245, 235, 222))
        self.screen.blit(backdrop, (panel.x, panel.y))
        pygame.draw.rect(self.screen, (0, 0, 0), panel, 2, border_radius=6)

        self.screen.blit(self.font_resource.render(title, False, (0, 0, 0)), (x, y))
        y += line_height * 1.5
        for line in lines:
            if line:
                self.screen.blit(self.font_resource.render(line, False, (0, 0, 0)), (x, y))
            y += line_height
        return panel

    def displayMoveLog(self):
        """Draw the recent-move strip along the bottom of the board.

        No-op unless a harness populated ``self.move_log`` (engine playCatan
        never does). Entries are plain strings, OLDEST FIRST; only the last
        ``MOVE_LOG_LINES`` are drawn, newest at the bottom. Long lines are
        truncated to the strip width rather than wrapped.

        Called BEFORE ``displayPorts`` and the buildings loop in
        ``displayGameScreen``: the strip covers three real vertices (y=720), the
        bottom of hexes 13/14/15, and two of the nine ports (the 2:1 WHEAT ship
        at (541, 759) and a 3:1 generic at (296, 751)), so drawing it last would
        hide pieces and port access placed there — public facts both."""
        if not self.move_log:
            return
        x, y, w, h = MOVE_LOG_RECT
        backdrop = pygame.Surface((w, h), pygame.SRCALPHA)
        backdrop.fill((245, 245, 235, 222))
        self.screen.blit(backdrop, (x, y))
        panel = pygame.Rect(x, y, w, h)
        pygame.draw.rect(self.screen, (0, 0, 0), panel, 2, border_radius=6)

        lines = list(self.move_log)[-MOVE_LOG_LINES:]
        text_y = y + 5
        for line in lines:
            surf = self.font_movelog.render(str(line), False, (0, 0, 0))
            self.screen.blit(surf, (x + 8, text_y), pygame.Rect(0, 0, w - 16, h))
            text_y += MOVE_LOG_LINE_HEIGHT

    # Function to display the gameState board - use to display intermediate build screens
    # gameScreenState specifies which type of screen is to be shown

    def displayGameScreen(self):
        # First display all initial hexes and regular buttons
        self.displayInitialBoard()
        self.displayGameButtons()
        self.displayRobber()
        self.displayPlayerStats()
        self.displayBroadcastMessage()

        # Display Dice Roll
        if self.diceRoll > 0:
            pygame.draw.rect(
                self.screen, pygame.Color("royalblue2"), (100, 20, 50, 50)
            )  # blue background
            diceNum = self.font_diceRoll.render(str(self.diceRoll), False, (0, 0, 0))
            self.screen.blit(diceNum, (110, 20))

        # Recent-move strip. Drawn BEFORE the buildings loop on purpose: the strip
        # overlaps the board's lower hexes and three real vertices (v42/v44/v47 at
        # y=720), so drawing it last would HIDE placed settlements/cities/roads on
        # the bottom row — withholding a public fact, which is exactly what this
        # feature exists to stop. Losing a few characters of log text behind at
        # most three small markers is the cheaper side of the trade.
        self.displayMoveLog()

        # Repaint the ports for the same reason, and in the same place, as the
        # buildings loop below: the strip's backdrop covers the 2:1 WHEAT ship
        # at (541, 759) and a 3:1 generic at (296, 751), plus the planks running
        # to the y=720 vertices. Port access is a PUBLIC planning fact; hiding
        # two of nine for the whole game biases the playtest toward the bot.
        self.displayPorts()

        # Loop through and display all existing buildings from players build graphs
        # Build Settlements and roads of each player
        for player_i in list(self.game.playerQueue.queue):
            for existingRoad in player_i.buildGraph["ROADS"]:
                self.draw_road(existingRoad, player_i.color)

            for settlementCoord in player_i.buildGraph["SETTLEMENTS"]:
                self.draw_settlement(settlementCoord, player_i.color)

            for cityCoord in player_i.buildGraph["CITIES"]:
                self.draw_city(cityCoord, player_i.color)

        # Whose-turn banner (top-center) — only when an interactive harness set it.
        if self.turn_banner is not None:
            text, bg_color = self.turn_banner
            surf = self.font_menu.render(text, True, (255, 255, 255))
            rect = surf.get_rect(center=(self.board.width // 2, 22))
            bg_rect = rect.inflate(34, 16)
            pygame.draw.rect(self.screen, pygame.Color(bg_color), bg_rect, border_radius=8)
            pygame.draw.rect(self.screen, (0, 0, 0), bg_rect, 2, border_radius=8)
            self.screen.blit(surf, rect)

        pygame.display.update()
        return
        # TO-DO Add screens for trades

    # Function to display dice roll

    def displayDiceRoll(self, diceNums):
        self.diceRoll = diceNums
        self.displayGameScreen()
        return None

    def displayBroadcastMessage(self):
        """Display the last broadcast event on the screen"""
        if not self.game.last_broadcast_event:
            return
        rendered = broadcast_message(
            self.game.last_broadcast_event,
            name_display=self.name_display,
            blind_player=(
                None if (self.bot_player is None or self.reveal_bot) else self.bot_player.name
            ),
        )
        if rendered is not None:
            msg_text, text_color = rendered
            text_surface = self.font_broadcast.render(msg_text, True, text_color)
            text_rect = text_surface.get_rect(center=(self.board.width // 2, 60))
            bg_rect = text_rect.inflate(20, 10)
            pygame.draw.rect(self.screen, (255, 255, 255), bg_rect)
            pygame.draw.rect(self.screen, (0, 0, 0), bg_rect, 2)
            self.screen.blit(text_surface, text_rect)

    def buildRoad_display(self, currentPlayer, roadsPossibleDict, allow_cancel=None):
        """Function to control build-road action with display
        args: player, who is building road; roadsPossibleDict - possible roads
        returns: road edge of road to be built

        ``allow_cancel=None`` (the default, and every historical caller) keeps
        the original behaviour: cancellable outside the setup phase. A caller
        that has ALREADY spent something on this placement — Road Builder, whose
        card is gone by the time we get here — passes ``allow_cancel=False`` so a
        stray click cannot silently forfeit the free road.
        """
        # Pulsating glow on every legal road edge; click one to build. Outside
        # the setup phase a click on empty space cancels (returns None).
        if allow_cancel is None:
            allow_cancel = not self.game.gameSetup
        roads = [edge for edge in roadsPossibleDict if roadsPossibleDict[edge]]
        return self._animated_pick(
            roads,
            lambda edge, pulse: self._glow_road(edge, currentPlayer.color, pulse),
            allow_cancel=allow_cancel,
        )

    def buildSettlement_display(self, currentPlayer, verticesPossibleDict):
        """Function to control build-settlement action with display
        args: player, who is building settlement; verticesPossibleDict - dictionary of possible settlement vertices
        returns: vertex of settlement to be built
        """
        # Pulsating glow on every legal settlement vertex (also reused for city
        # upgrades); click one to build. Outside setup a miss cancels (None).
        vertices = [v for v in verticesPossibleDict if verticesPossibleDict[v]]
        return self._animated_pick(
            vertices,
            lambda v, pulse: self._glow_settlement(v, currentPlayer.color, pulse),
            allow_cancel=not self.game.gameSetup,
        )

    def buildCity_display(self, currentPlayer, verticesPossibleDict):
        """Function to control build-city action with display
        args: player, who is building city; verticesPossibleDict - dictionary of possible city vertices
        returns: city vertex of city to be built
        """
        # Pulsating glow on every legal city vertex; click one to upgrade, or
        # click empty space to cancel (returns None).
        vertices = [c for c in verticesPossibleDict if verticesPossibleDict[c]]
        return self._animated_pick(
            vertices,
            lambda v, pulse: self._glow_city(v, currentPlayer.color, pulse),
            allow_cancel=True,
        )

    # Function to control the move-robber action with display

    def moveRobber_display(self, currentPlayer, possibleRobberDict):
        """Pick a robber hex, then resolve the steal victim.

        Two conformance rules the bot already has and this picker did not:

        * **Fail open on an empty spot set**, mirroring ``env/masks.py`` (which
          sets all 19 tiles when the Friendly-Robber filter leaves nothing). The
          pick below is ``allow_cancel=False`` with no ``pygame.QUIT`` arm, so an
          empty set used to hard-lock the window instead of offering anything.
        * **``currentPlayer`` is filtered out of the victim set.**
          ``board.get_players_to_rob`` is keyed on the hex alone, so the human's
          OWN building was offered as a steal target; ``steal_resource(self)`` is
          a net-zero self-transfer, i.e. a mis-click forfeited the steal. The bot
          filters itself (``env/catan_env._apply_robber_placement``), as does
          ``conformance/recorder.py``.
        """
        # Pulsating glow on every legal robber hex; click one, then pick a victim.
        spots = possibleRobberDict or self.board.hexTileDict
        pix = {R: spots[R].to_pixel(self.board.flat) for R in spots}
        hexIndex = self._animated_pick(
            list(pix.keys()),
            lambda R, pulse: self._glow_robber(pix[R], pulse),
            allow_cancel=False,
        )
        possiblePlayerDict = {
            victim: vertex
            for victim, vertex in self.board.get_players_to_rob(hexIndex).items()
            if victim is not currentPlayer
        }
        playerToRob = self.choosePlayerToRob_display(possiblePlayerDict)
        return hexIndex, playerToRob

    # Function to control the choice of player to rob with display
    # Returns the choice of player to rob

    def choosePlayerToRob_display(self, possiblePlayerDict):
        """Choose a steal victim from the (already self-filtered) candidates.

        Zero candidates -> ``None`` (no steal). Exactly one candidate -> it is
        AUTO-SELECTED with no prompt, matching the bot, which never asks. In 1v1
        those are the only two cases that can occur once the robbing player is
        filtered out, so the click loop below is effectively dead code kept for
        the legacy engine loop rather than a path the harness can enter.
        """
        if not possiblePlayerDict:
            return None

        if len(possiblePlayerDict) == 1:
            return next(iter(possiblePlayerDict))

        # Get all other players the player can move robber to and show circles
        for player, vertex in possiblePlayerDict.items():
            possiblePlayerDict[player] = self.draw_possible_players_to_rob(vertex)

        pygame.display.update()

        mouseClicked = False
        clock = pygame.time.Clock()
        while mouseClicked == False:
            for e in pygame.event.get():
                if e.type == pygame.MOUSEBUTTONDOWN:
                    for playerToRob, playerCircleRect in possiblePlayerDict.items():
                        if playerCircleRect.collidepoint(e.pos):
                            return playerToRob
            clock.tick(30)

    def get_resource_selection(self, player, mode, num_to_select=1):
        """
        Displays a resource selection menu and handles user interaction.
        mode: 'DISCARD', 'YOP', 'MONOPOLY', 'BANK'
        num_to_select: Number of resources to select (for DISCARD/YOP)

        This method MUTATES ``player.resources`` directly, so it must also move
        the finite resource bank (spec 009) or the
        ``bank[R] + sum(hands[R]) == 19`` invariant breaks:
          * 'DISCARD' -> ``bank_recirculate`` per card discarded;
          * 'YOP' -> ``bank_draw`` per card granted, gated on
            ``bank_can_supply`` (an unsuppliable swatch is greyed and its click
            does not register), and ``bank_recirculate`` for every pick already
            made if the player cancels;
          * 'MONOPOLY' / 'BANK' return a choice without granting anything here,
            so the caller (``player.play_devCard`` / ``initiate_trade``) owns
            the bank move.
        """
        resources = ["BRICK", "ORE", "SHEEP", "WHEAT", "WOOD"]

        # Menu Geometry
        menu_width = 500
        menu_height = 150
        menu_x = (self.board.width - menu_width) // 2
        menu_y = (self.board.height - menu_height) // 2

        res_size = 80
        spacing = 15
        start_x = menu_x + (menu_width - (5 * res_size + 4 * spacing)) // 2
        res_y = menu_y + (menu_height - res_size) // 2

        # Create Rects for resources
        res_rects = {}
        for i, res in enumerate(resources):
            res_rects[res] = pygame.Rect(
                start_x + i * (res_size + spacing), res_y, res_size, res_size
            )

        # State variables
        selected_resources = []  # For YOP/Discard
        trade_in_res = None  # For Bank
        receive_res = None  # For Bank
        monopoly_res = None  # For Monopoly
        result = None

        running = True
        clock = pygame.time.Clock()
        while running:
            # Draw Menu Background
            pygame.draw.rect(
                self.screen, (200, 200, 200), (menu_x, menu_y, menu_width, menu_height)
            )
            pygame.draw.rect(self.screen, (0, 0, 0), (menu_x, menu_y, menu_width, menu_height), 2)

            # Draw Title
            title_text = ""
            if mode == "DISCARD":
                title_text = f"Discard {num_to_select - len(selected_resources)} cards"
            elif mode == "YOP":
                title_text = f"Select {num_to_select - len(selected_resources)} resources"
                if any(not self.board.bank_can_supply({r: 1}) for r in resources):
                    title_text += "  (greyed = bank empty)"
            elif mode == "MONOPOLY":
                title_text = "Select resource to monopolize"
            elif mode == "BANK":
                title_text = "Select Trade-In (Red) then Receive (Green)"

            text_surf = self.font_menu.render(title_text, True, (0, 0, 0))
            self.screen.blit(text_surf, (menu_x + 10, menu_y + 10))

            # Draw Resources
            for res in resources:
                rect = res_rects[res]
                color = self.colorDict[res]
                pygame.draw.rect(self.screen, color, rect)

                # Draw Count (Player's current amount)
                count = player.resources[res]
                count_text = self.font_button.render(str(count), True, (0, 0, 0))
                self.screen.blit(count_text, (rect.centerx - 5, rect.centery - 5))

                # Year of Plenty DRAWS from the finite bank, so a resource the
                # bank cannot supply is not grantable below. The count drawn
                # above is the PLAYER's holding, so without this an empty-bank
                # resource looks identically pickable and the click just does
                # nothing — grey it out instead of failing silently.
                if mode == "YOP" and not self.board.bank_can_supply({res: 1}):
                    shade = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
                    shade.fill((40, 40, 40, 190))
                    self.screen.blit(shade, rect.topleft)
                    pygame.draw.rect(self.screen, (90, 90, 90), rect, 4)

                # Draw Outlines based on state
                if mode == "MONOPOLY":
                    if res == monopoly_res:
                        pygame.draw.rect(self.screen, (0, 0, 0), rect, 4)  # Black outline

                elif mode == "BANK":
                    if res == trade_in_res:
                        pygame.draw.rect(self.screen, (255, 0, 0), rect, 4)  # Red outline
                    elif res == receive_res:
                        pygame.draw.rect(self.screen, (0, 255, 0), rect, 4)  # Green outline

            pygame.display.update()

            # Event Handling
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    sys.exit(0)

                if e.type == pygame.MOUSEBUTTONDOWN:
                    # Check for click outside menu to cancel (except DISCARD)
                    menu_rect = pygame.Rect(menu_x, menu_y, menu_width, menu_height)
                    if not menu_rect.collidepoint(e.pos) and mode != "DISCARD":
                        if mode == "YOP":
                            # Revert resources added so far. Each was DRAWN from
                            # the finite bank below, so the revert must put it
                            # back (spec 009) — otherwise cancelling a partly
                            # picked YoP leaks the supply the other way.
                            for res in selected_resources:
                                player.resources[res] -= 1
                                self.board.bank_recirculate({res: 1})

                        result = None
                        running = False
                        break

                    clicked_res = None
                    for res, rect in res_rects.items():
                        if rect.collidepoint(e.pos):
                            clicked_res = res
                            break

                    if mode == "DISCARD":
                        if clicked_res and player.resources[clicked_res] > 0:
                            player.resources[clicked_res] -= 1
                            # Discards RECIRCULATE into the finite bank (spec
                            # 009), exactly as every non-GUI discard path does
                            # (heuristic.py, random_ai.py, env/catan_env.py).
                            self.board.bank_recirculate({clicked_res: 1})
                            selected_resources.append(clicked_res)
                            if len(selected_resources) >= num_to_select:
                                result = selected_resources
                                running = False

                    elif mode == "YOP":
                        # Year of Plenty DRAWS from the finite bank and is gated
                        # on availability, matching the AI branch in
                        # player.play_devCard: a resource the bank cannot supply
                        # is simply not grantable (the click does not register;
                        # the swatch is greyed above so that is visible). The AI
                        # branch instead burns the pick on an unsupplied choice,
                        # because it picks blind at random — a human reading a
                        # greyed swatch is not making that choice, so declining
                        # the click is the honest equivalent, not a favour.
                        if clicked_res and self.board.bank_can_supply({clicked_res: 1}):
                            player.resources[clicked_res] += 1
                            self.board.bank_draw({clicked_res: 1})
                            selected_resources.append(clicked_res)
                            if len(selected_resources) >= num_to_select:
                                result = selected_resources
                                running = False

                    elif mode == "MONOPOLY":
                        if clicked_res:
                            if clicked_res == monopoly_res:
                                result = clicked_res  # Confirm
                                running = False
                            else:
                                monopoly_res = clicked_res

                    elif mode == "BANK":
                        if trade_in_res is None:
                            if clicked_res and player.resources[clicked_res] > 0:
                                trade_in_res = clicked_res
                        else:
                            # Trade-in is selected
                            if receive_res is None:
                                if clicked_res == trade_in_res:
                                    trade_in_res = None  # Deselect
                                elif clicked_res:
                                    receive_res = clicked_res
                                else:
                                    # Clicked elsewhere
                                    trade_in_res = None
                            else:
                                # Receive is selected
                                if clicked_res == receive_res:
                                    result = (trade_in_res, receive_res)  # Confirm
                                    running = False
                                elif clicked_res == trade_in_res:
                                    receive_res = None  # Deselect receive
                                elif clicked_res:
                                    receive_res = clicked_res  # Change receive
                                else:
                                    receive_res = None  # Deselect receive

            clock.tick(30)  # Limit redraw rate to prevent flickering
        self.displayGameScreen()
        return result

    def get_dev_card_selection(self, player):
        """
        Displays a dev card selection menu and handles user interaction.

        ``self.dev_card_filter`` (default ``None`` = unrestricted) narrows what
        the menu will accept: a card outside it is drawn DIMMED and its clicks
        are ignored, exactly as an unsuppliable resource is in
        ``get_resource_selection`` — and, as there, the title gains a
        ``(greyed = not now)`` hint whenever a filter is in force, so a
        swallowed click reads as a restriction rather than a freeze. Used by
        the pre-roll window in
        ``scripts/play_vs_model.py`` so the human is offered the same card set
        the bot's action mask offers.
        """
        allowed = self.dev_card_filter
        dev_cards = ["KNIGHT", "VP", "MONOPOLY", "ROADBUILDER", "YEAROFPLENTY"]
        abbreviations = {
            "KNIGHT": "K",
            "VP": "VP",
            "MONOPOLY": "M",
            "ROADBUILDER": "RB",
            "YEAROFPLENTY": "YOP",
        }

        # Menu Geometry
        menu_width = 500
        menu_height = 150
        menu_x = (self.board.width - menu_width) // 2
        menu_y = (self.board.height - menu_height) // 2

        card_size = 80
        spacing = 15
        start_x = menu_x + (menu_width - (5 * card_size + 4 * spacing)) // 2
        card_y = menu_y + (menu_height - card_size) // 2

        # Create Rects for dev cards
        card_rects = {}
        for i, card in enumerate(dev_cards):
            card_rects[card] = pygame.Rect(
                start_x + i * (card_size + spacing), card_y, card_size, card_size
            )

        selected_card = None
        result = None
        running = True

        while running:
            # Draw Menu Background
            pygame.draw.rect(
                self.screen, (200, 200, 200), (menu_x, menu_y, menu_width, menu_height)
            )
            pygame.draw.rect(self.screen, (0, 0, 0), (menu_x, menu_y, menu_width, menu_height), 2)

            # Draw Title
            title_text = "Select Development Card to Play"
            if allowed is not None:
                # Same contract as get_resource_selection's "(greyed = bank
                # empty)": a swallowed click must be EXPLAINED, not read as a
                # freeze. Kept short — the menu is 500px wide at font size 20.
                title_text += "  (greyed = not now)"
            text_surf = self.font_menu.render(title_text, True, (0, 0, 0))
            self.screen.blit(text_surf, (menu_x + 10, menu_y + 10))

            # Draw Dev Cards
            for card in dev_cards:
                rect = card_rects[card]
                blocked = allowed is not None and card not in allowed
                # Dimmed = not offered here (mirrors the unsuppliable-resource
                # treatment in get_resource_selection). The COUNT below is still
                # drawn truthfully — the hand is never altered.
                shade = (90, 90, 90) if blocked else (128, 128, 128)
                pygame.draw.rect(self.screen, shade, rect)  # Gray square

                # Draw Abbreviation (Top)
                abbr = abbreviations[card]
                abbr_text = self.font_button.render(abbr, True, (0, 0, 0))
                self.screen.blit(
                    abbr_text, (rect.centerx - abbr_text.get_width() // 2, rect.top + 5)
                )

                # Draw Count (Middle)
                count = player.devCards[card]
                count_text = self.font_button.render(str(count), True, (0, 0, 0))
                self.screen.blit(
                    count_text,
                    (
                        rect.centerx - count_text.get_width() // 2,
                        rect.centery - count_text.get_height() // 2,
                    ),
                )

                # Draw Selection Outline
                if card == selected_card:
                    pygame.draw.rect(self.screen, (0, 255, 0), rect, 4)  # Green outline

            pygame.display.update()

            # Event Handling
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    sys.exit(0)

                if e.type == pygame.MOUSEBUTTONDOWN:
                    # Check for click outside menu to cancel
                    menu_rect = pygame.Rect(menu_x, menu_y, menu_width, menu_height)
                    if not menu_rect.collidepoint(e.pos):
                        result = None
                        running = False
                        break

                    clicked_card = None
                    for card, rect in card_rects.items():
                        if rect.collidepoint(e.pos):
                            clicked_card = card
                            break

                    if clicked_card:
                        if clicked_card == "VP":
                            continue  # Do nothing for VP

                        if allowed is not None and clicked_card not in allowed:
                            continue  # Not offered in this window

                        if player.devCards[clicked_card] > 0:
                            if clicked_card == selected_card:
                                # Double click (click on already selected) -> Confirm
                                result = clicked_card
                                running = False
                            else:
                                # Select new card
                                selected_card = clicked_card
                        else:
                            # Player doesn't have this card
                            pass

        self.displayGameScreen()
        return result
