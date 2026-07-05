"""Terminal-reward semantics (audit 2026-07).

The VP-margin bonus is paid only on TRUE termination (someone reached
``maxPoints``). Truncated episodes (``max_turns`` cap) must return 0.0
reward: GAE keeps the ``V(s_T)`` bootstrap on truncated-not-terminated
steps, so any margin payment there would double-count credit for the
same event.
"""

from __future__ import annotations

import numpy as np
import pytest

from catan_rl.env.catan_env import CatanEnv


def _first_legal_action(masks: dict[str, np.ndarray]) -> np.ndarray:
    """Build a 6-head action from the first legal index of each mask.

    Prefers RollDice/EndTurn when legal so games race to the turn cap.
    """

    def first(key: str) -> int:
        idx = np.flatnonzero(masks[key])
        return int(idx[0]) if idx.size else 0

    legal_types = np.flatnonzero(masks["type"])
    assert legal_types.size, "no legal action type"
    t = int(legal_types[0])
    for preferred in (12, 3):  # ROLL_DICE, END_TURN
        if masks["type"][preferred]:
            t = preferred
            break
    corner = first("corner_city") if t == 1 else first("corner_settlement")
    res1_key = {10: "resource1_trade", 11: "resource1_discard"}.get(t, "resource1_default")
    return np.array(
        [t, corner, first("edge"), first("tile"), first(res1_key), first("resource2_default")],
        dtype=np.int64,
    )


@pytest.fixture
def env() -> CatanEnv:
    e = CatanEnv(opponent_type="random", max_turns=5)
    e.reset(seed=0)
    return e


class TestTerminalRewardBranches:
    def test_truncation_pays_zero_despite_vp_lead(self, env: CatanEnv) -> None:
        assert env.agent_player is not None and env.opponent_player is not None
        env.agent_player.victoryPoints = 9
        env.opponent_player.victoryPoints = 3
        # Nobody at maxPoints -> the truncation branch: no margin payment.
        assert env._terminal_reward() == 0.0

    def test_truncation_pays_zero_despite_vp_deficit(self, env: CatanEnv) -> None:
        assert env.agent_player is not None and env.opponent_player is not None
        env.agent_player.victoryPoints = 3
        env.opponent_player.victoryPoints = 9
        assert env._terminal_reward() == 0.0

    def test_win_pays_one_plus_margin(self, env: CatanEnv) -> None:
        assert env.agent_player is not None and env.opponent_player is not None
        env.agent_player.victoryPoints = 15
        env.opponent_player.victoryPoints = 8
        expected = 1.0 + (15 - 8) * env.vp_margin_bonus
        assert env._terminal_reward() == pytest.approx(expected)

    def test_loss_pays_minus_one_plus_negative_margin(self, env: CatanEnv) -> None:
        assert env.agent_player is not None and env.opponent_player is not None
        env.agent_player.victoryPoints = 8
        env.opponent_player.victoryPoints = 15
        expected = -1.0 + (8 - 15) * env.vp_margin_bonus
        assert env._terminal_reward() == pytest.approx(expected)


class TestStepTruncation:
    def test_step_truncation_returns_zero_reward(self) -> None:
        """Drive a real game to the max_turns cap with a forced VP lead.

        Pre-fix, the truncated step paid ``(a_vp - o_vp) * bonus > 0``;
        post-fix it must pay exactly 0.0 while still reporting
        ``truncated=True, terminated=False``.
        """
        env = CatanEnv(opponent_type="random", max_turns=2)
        env.reset(seed=3)
        bumped = False
        reward = 0.0
        terminated = truncated = False
        for _ in range(400):
            if not env.initial_placement_phase and not bumped:
                # Force a nonzero VP margin at the cap (stays far below
                # maxPoints=15, so it cannot flip the game to terminated).
                assert env.agent_player is not None
                env.agent_player.victoryPoints += 5
                bumped = True
            action = _first_legal_action(env.get_action_masks())
            _obs, reward, terminated, truncated, _info = env.step(action)
            if terminated or truncated:
                break
        assert truncated and not terminated
        assert bumped
        assert reward == 0.0
