"""Leak pins for the opponent hand panel (``gui/view.hand_panel_lines``) and
for the on-screen broadcast banner (``gui/view.broadcast_message``).

Every human-vs-policy game played while the bot's FULL hand was rendered on
screen is uninterpretable as a strength read, so the blind default is pinned
here rather than left to the harness.

``hand_panel_lines`` and ``broadcast_message`` are deliberately pygame-free and
duck-typed, so most of these tests need neither a display nor a real engine
``player``. ``TestPanelRendering`` is the exception: it boots a headless dummy
surface to exercise the box-sizing path in ``_draw_hand_panel``, which the pure
line tests cannot reach.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from catan_rl.gui.view import (
    broadcast_message,
    catanGameView,
    hand_panel_lines,
)

_RESOURCE_NAMES = ("WOOD", "BRICK", "WHEAT", "ORE", "SHEEP")
_DEV_TYPE_LABELS = ("Knight", "VP", "Mono", "RB", "YOP")


@dataclass
class _FakePlayer:
    """The four attributes ``hand_panel_lines`` reads."""

    resources: dict[str, int] = field(
        default_factory=lambda: {"WOOD": 2, "BRICK": 1, "WHEAT": 0, "ORE": 3, "SHEEP": 1}
    )
    # 9 total VP of which 2 come from hidden VP cards -> 7 visible.
    victoryPoints: int = 9
    devCards: dict[str, int] = field(
        default_factory=lambda: {
            "KNIGHT": 1,
            "VP": 2,
            "MONOPOLY": 0,
            "ROADBUILDER": 0,
            "YEAROFPLENTY": 1,
        }
    )
    newDevCards: list[str] = field(default_factory=lambda: ["KNIGHT"])


class TestBlindPanel:
    def test_no_resource_type_is_reachable(self) -> None:
        blob = "\n".join(hand_panel_lines(_FakePlayer(), reveal=False))
        for name in _RESOURCE_NAMES:
            assert name not in blob, f"blind panel leaks resource type {name}"

    def test_no_dev_card_type_is_reachable(self) -> None:
        blob = "\n".join(hand_panel_lines(_FakePlayer(), reveal=False))
        for label in _DEV_TYPE_LABELS:
            assert f"{label}:" not in blob, f"blind panel leaks dev-card type {label}"

    def test_shows_hand_size_and_unplayed_dev_count(self) -> None:
        lines = hand_panel_lines(_FakePlayer(), reveal=False)
        assert "Cards: 7" in lines  # 2+1+0+3+1
        assert "Dev Cards: 5" in lines  # sum(devCards)=4 + 1 newDevCard

    def test_vp_shown_is_visible_vp_not_the_total(self) -> None:
        # 9 total - 2 VP cards = 7 visible. The VP-inclusive total must NOT
        # appear anywhere in the blind panel.
        lines = hand_panel_lines(_FakePlayer(), reveal=False)
        assert "Victory Points: 7" in lines
        assert "Victory Points: 9" not in lines

    def test_newdevcards_do_not_reduce_visible_vp(self) -> None:
        # A VP card scores immediately and never lands in ``newDevCards``
        # (engine/player.py), so a pending KNIGHT in that bucket must not be
        # subtracted. Regression guard against a ``- len(newDevCards)`` term.
        player = _FakePlayer()
        player.newDevCards = ["KNIGHT", "MONOPOLY"]
        assert "Victory Points: 7" in hand_panel_lines(player, reveal=False)

    def test_visible_vp_ignores_the_stale_cache(self) -> None:
        # ``player.visibleVictoryPoints`` is refreshed only at init + VP-card
        # buy, so it goes stale. A deliberately wrong cache must be ignored.
        player = _FakePlayer()
        player.visibleVictoryPoints = 999  # type: ignore[attr-defined]
        assert "Victory Points: 7" in hand_panel_lines(player, reveal=False)


class TestRevealedPanel:
    def test_reveal_restores_the_full_lines(self) -> None:
        lines = hand_panel_lines(_FakePlayer(), reveal=True)
        assert "WOOD: 2" in lines
        assert "ORE: 3" in lines
        assert "Victory Points: 9" in lines  # VP-card-inclusive total
        assert "Knight: 2" in lines  # 1 held + 1 just-bought
        assert "VP: 2" in lines

    def test_reveal_is_the_default(self) -> None:
        # ``_draw_hand_panel`` relies on this default for a player's OWN hand.
        assert hand_panel_lines(_FakePlayer()) == hand_panel_lines(_FakePlayer(), reveal=True)


class TestBroadcastBanner:
    """The banner carries resource types — and for DISCARD/YOP that is CORRECT.

    These events are PUBLIC. The engine says so itself (``tracker.track_steal``:
    "It is Public Information relative to the two players"), and the bot reads
    this same broadcast stream through a ``BroadcastHandTracker`` that does
    perfect opponent hand-tracking (``env/catan_env.py``). Blinding the human to
    the bot's discards while the bot tracks the human's would make the playtest
    HARDER than the real game and bias every result in the bot's favour.

    What the blind gate closes is the bot's HAND CONTENTS (``hand_panel_lines``),
    which are private. Public events stay public."""

    def test_bot_discard_types_stay_public_even_when_blinded(self) -> None:
        text, _color = broadcast_message(
            {"type": "DISCARD", "player": "Agent", "resources": ["ORE", "WHEAT"]},
            blind_player="Agent",
        )
        assert "ORE" in text and "WHEAT" in text

    def test_bot_yop_types_stay_public_even_when_blinded(self) -> None:
        """Year-of-Plenty picks are taken openly from the bank — public, like discards."""
        text, _color = broadcast_message(
            {"type": "YOP", "player": "Agent", "resources": ["ORE", "ORE"]},
            blind_player="Agent",
        )
        assert "ORE" in text

    def test_the_humans_own_discard_is_unchanged(self) -> None:
        text, _color = broadcast_message(
            {"type": "DISCARD", "player": "Opponent", "resources": ["ORE"]},
            blind_player="Agent",
        )
        assert "ORE" in text

    def test_no_blind_player_keeps_the_engine_message(self) -> None:
        # engine playCatan sets no bot_player -> nothing is hidden from anyone.
        text, _color = broadcast_message(
            {"type": "DISCARD", "player": "Agent", "resources": ["ORE"]}
        )
        assert "ORE" in text

    def test_display_names_and_dice_still_work(self) -> None:
        text, _color = broadcast_message(
            {"type": "DICE_ROLL", "player": "Agent", "value": 8},
            name_display={"Agent": "Bot"},
            blind_player="Agent",
        )
        assert text == "Dice: Bot rolled 8"

    def test_unknown_event_renders_nothing(self) -> None:
        assert broadcast_message({"type": "BUILD", "player": "Agent"}) is None


class TestPanelRendering:
    """The box-sizing path (``panel_lines = len(lines) + 2``) is the only part
    of the blind panel the pure line tests cannot reach: the blind panel has 5
    lines where the revealed one has 12, so the backdrop must shrink with it."""

    def _view(self):  # type: ignore[no-untyped-def]
        import pygame

        pygame.init()
        pygame.display.set_mode((320, 240))
        view = catanGameView.__new__(catanGameView)
        view.screen = pygame.display.get_surface()
        view.font_resource = pygame.font.SysFont("cambria", 15)
        return view

    def test_blind_and_revealed_panels_both_render(self) -> None:
        view = self._view()
        player = _FakePlayer()
        # No assertion on pixels (font rendering is platform-dependent); this
        # pins that the sizing arithmetic and the blit calls actually run.
        view._draw_hand_panel(player, 20, 20, "Bot — HAND", reveal=False)
        view._draw_hand_panel(player, 20, 140, "YOUR HAND", reveal=True)

    def test_the_blind_backdrop_is_shorter_than_the_revealed_one(self) -> None:
        blind = len(hand_panel_lines(_FakePlayer(), reveal=False)) + 2
        revealed = len(hand_panel_lines(_FakePlayer(), reveal=True)) + 2
        assert blind < revealed
