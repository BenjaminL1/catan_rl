"""Leak pins for the opponent hand panel (``gui/view.hand_panel_lines``).

Every human-vs-policy game played while the bot's FULL hand was rendered on
screen is uninterpretable as a strength read, so the blind default is pinned
here rather than left to the harness.

``hand_panel_lines`` is deliberately pygame-free and duck-typed, so these tests
need neither a display nor a real engine ``player``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from catan_rl.gui.view import hand_panel_lines

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
