"""Integration smoke for :class:`EvalHarness`.

Uses a tiny stub policy + a real CatanEnv. The stub picks legal type-0
actions randomly. The test only verifies the harness's plumbing:
matchup iteration, symmetrisation tagging, Wilson CI computation,
rules-invariant audit hook.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from catan_rl.eval.harness import EvalHarness, EvalReport, EvalResult, GameOutcome

# ---------------------------------------------------------------------------
# Stub policy
# ---------------------------------------------------------------------------


class _StubPolicy(nn.Module):
    """Picks a uniformly-random legal type and zeros for other heads."""

    def __init__(self) -> None:
        super().__init__()
        self.dummy = nn.Parameter(torch.zeros(1))
        self._rng = np.random.default_rng(0)

    def sample(
        self, obs: dict[str, torch.Tensor], masks: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        B = next(iter(obs.values())).shape[0]
        type_mask = masks["type"].cpu().numpy()
        action = np.zeros((B, 6), dtype=np.int64)
        for i in range(B):
            legal = np.flatnonzero(type_mask[i])
            if legal.size:
                action[i, 0] = int(self._rng.choice(legal))
            else:
                action[i, 0] = 3
        device = next(iter(obs.values())).device
        return {
            "action": torch.as_tensor(action, device=device),
            "log_prob": torch.zeros(B, device=device),
            "value": torch.zeros(B, device=device),
            "per_head_log_prob": torch.zeros((B, 6), device=device),
            "entropy": torch.zeros(B, device=device) + 1.5,
        }


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_default_construction(self) -> None:
        h = EvalHarness()
        assert h.opponent_types == ("random", "heuristic")
        assert h.n_games_per_seat == 100
        assert h.audit_rules is True

    def test_zero_games_rejected(self) -> None:
        with pytest.raises(ValueError, match="n_games_per_seat"):
            EvalHarness(n_games_per_seat=0)

    def test_empty_opponents_rejected(self) -> None:
        with pytest.raises(ValueError, match="opponent_types"):
            EvalHarness(opponent_types=())


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------


class TestRun:
    def test_smoke_with_random_opponent(self) -> None:
        # Tiny game count for a quick smoke.
        harness = EvalHarness(
            opponent_types=("random",),
            n_games_per_seat=2,
            max_turns=30,
            seed=42,
            device="cpu",
        )
        report = harness.run(_StubPolicy())
        assert isinstance(report, EvalReport)
        assert report.n_games_total == 4  # 2 per seat × 2 seats
        res = report.by_opponent("random")
        assert isinstance(res, EvalResult)
        assert res.n == 4
        # Wilson CI populated.
        assert 0.0 <= res.ci.lower <= res.ci.upper <= 1.0
        # Seat split: 2 games per seat.
        assert res.n_seat0 == 2
        assert res.n_seat1 == 2

    def test_smoke_with_two_opponents(self) -> None:
        harness = EvalHarness(
            opponent_types=("random", "heuristic"),
            n_games_per_seat=1,
            max_turns=20,
            seed=0,
            device="cpu",
        )
        report = harness.run(_StubPolicy())
        assert report.n_games_total == 4  # 1 per seat × 2 seats × 2 opps
        assert len(report.results) == 2
        assert report.by_opponent("random") is not None
        assert report.by_opponent("heuristic") is not None
        assert report.by_opponent("nonexistent") is None

    def test_rules_audit_attached(self) -> None:
        # With audit_rules=True the harness should populate the
        # per-game violations tuple. For a healthy game it should be
        # empty. For an unhealthy one (artificially broken), the
        # harness surfaces the violations.
        harness = EvalHarness(
            opponent_types=("random",),
            n_games_per_seat=1,
            max_turns=20,
            seed=0,
            device="cpu",
            audit_rules=True,
        )
        report = harness.run(_StubPolicy())
        res = report.by_opponent("random")
        assert res is not None
        # Empty violations tuple per game.
        for g in res.games:
            assert isinstance(g.rules_violations, tuple)

    def test_audit_skipped_when_off(self) -> None:
        harness = EvalHarness(
            opponent_types=("random",),
            n_games_per_seat=1,
            max_turns=20,
            seed=0,
            device="cpu",
            audit_rules=False,
        )
        report = harness.run(_StubPolicy())
        res = report.by_opponent("random")
        assert res is not None
        for g in res.games:
            assert g.rules_violations == ()


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


class TestReproducibility:
    def test_same_seed_yields_same_outcomes(self) -> None:
        # Two independent harness runs at the same seed must produce
        # identical per-game outcomes (seeds, won, agent_vp, opp_vp).
        # Sensitive to the seed-derivation algorithm in
        # _evaluate_matchup.
        def _run() -> tuple[GameOutcome, ...]:
            import random as _r

            _r.seed(0)
            np.random.seed(0)
            harness = EvalHarness(
                opponent_types=("random",),
                n_games_per_seat=2,
                max_turns=20,
                seed=42,
                device="cpu",
                audit_rules=False,
            )
            return harness.run(_StubPolicy()).results[0].games

        a = _run()
        b = _run()
        assert len(a) == len(b)
        # Seed sequence is deterministic in harness.seed.
        for ga, gb in zip(a, b, strict=True):
            assert ga.seed == gb.seed


# ---------------------------------------------------------------------------
# Symmetrisation tagging
# ---------------------------------------------------------------------------


class TestSymmetrisation:
    def test_seat_tags_alternate(self) -> None:
        harness = EvalHarness(
            opponent_types=("random",),
            n_games_per_seat=3,
            max_turns=20,
            seed=0,
            device="cpu",
            audit_rules=False,
        )
        report = harness.run(_StubPolicy())
        res = report.by_opponent("random")
        assert res is not None
        # Each game_idx generates exactly one seat=0 and one seat=1 game.
        # The interleaving in _evaluate_matchup is (seat0, seat1) per
        # game_idx, so the tags alternate.
        seats = [g.agent_seat for g in res.games]
        assert seats == [0, 1, 0, 1, 0, 1]

    def test_seat_1_actually_swaps_in_env(self) -> None:
        # Real seat swap: at seat=1 the opponent makes a settlement
        # BEFORE the agent's first action. The agent's first obs in a
        # seat=1 game therefore shows >=1 settlement on the board (the
        # opponent's first setup), while a seat=0 reset has 0.
        from catan_rl.env.catan_env import CatanEnv

        env = CatanEnv(opponent_type="random", max_turns=20)
        try:
            env.reset(seed=123, options={"agent_seat": 0})
            assert env.opponent_player is not None
            # Opponent hasn't placed anything yet.
            opp_settles_seat0 = len(env.opponent_player.buildGraph["SETTLEMENTS"])

            env.reset(seed=123, options={"agent_seat": 1})
            assert env.opponent_player is not None
            opp_settles_seat1 = len(env.opponent_player.buildGraph["SETTLEMENTS"])

            assert opp_settles_seat0 == 0
            assert opp_settles_seat1 == 1
        finally:
            env.close()


# ---------------------------------------------------------------------------
# Cross-process seed determinism
# ---------------------------------------------------------------------------


class TestCrossProcessDeterminism:
    def test_seed_is_stable_across_python_hash_salts(self) -> None:
        # ``hash()`` of a Python string is salted per-process via
        # PYTHONHASHSEED. The harness must derive game seeds from a
        # stable hash (zlib.crc32) instead. Spawn two subprocesses with
        # different PYTHONHASHSEED values and verify the per-game
        # seed list is identical.
        import os
        import subprocess
        import sys
        import textwrap

        prog = textwrap.dedent(
            """
            import sys
            sys.path.insert(0, %r)
            import numpy as np
            import torch
            from torch import nn
            from catan_rl.eval.harness import EvalHarness

            class _StubPolicy(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.dummy = nn.Parameter(torch.zeros(1))
                    self._rng = np.random.default_rng(0)
                def sample(self, obs, masks):
                    B = next(iter(obs.values())).shape[0]
                    type_mask = masks["type"].cpu().numpy()
                    action = np.zeros((B, 6), dtype=np.int64)
                    for i in range(B):
                        legal = np.flatnonzero(type_mask[i])
                        action[i, 0] = int(self._rng.choice(legal)) if legal.size else 3
                    device = next(iter(obs.values())).device
                    return {
                        "action": torch.as_tensor(action, device=device),
                        "log_prob": torch.zeros(B, device=device),
                        "value": torch.zeros(B, device=device),
                        "per_head_log_prob": torch.zeros((B, 6), device=device),
                        "entropy": torch.zeros(B, device=device) + 1.5,
                    }

            h = EvalHarness(
                opponent_types=("random", "heuristic"),
                n_games_per_seat=2,
                max_turns=10,
                seed=42,
                device="cpu",
                audit_rules=False,
            )
            report = h.run(_StubPolicy())
            for r in report.results:
                for g in r.games:
                    print(r.opponent_type, g.seed)
            """
        ).strip() % (os.path.join(os.getcwd(), "src"),)

        env_a = {**os.environ, "PYTHONHASHSEED": "0", "SDL_VIDEODRIVER": "dummy"}
        env_b = {**os.environ, "PYTHONHASHSEED": "987654321", "SDL_VIDEODRIVER": "dummy"}
        out_a = subprocess.check_output([sys.executable, "-c", prog], env=env_a, text=True)
        out_b = subprocess.check_output([sys.executable, "-c", prog], env=env_b, text=True)
        assert out_a == out_b, (
            "EvalHarness game seeds drift across PYTHONHASHSEED values — "
            "use a stable hash for the seed derivation"
        )
