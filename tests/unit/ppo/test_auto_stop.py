"""Unit tests for the automated plateau-stop (`auto_stop`) gate.

Pins the pure decision function :func:`evaluate_auto_stop` (hard/soft thresholds)
and the stateful :class:`AutoStopTracker` (counter reset + window-mean recording on
promotion). The loop-level wiring (never-invoked-when-disabled, terminal-save-on-stop)
is exercised in ``tests/integration/test_train_loop_smoke.py``.
"""

from __future__ import annotations

import pytest

from catan_rl.ppo.arguments import AutoStopConfig, TrainConfig
from catan_rl.ppo.training_loop import (
    AutoStopTracker,
    _resume_updates_since_promotion,
    evaluate_auto_stop,
)


def _cfg(**kw: object) -> AutoStopConfig:
    base = {
        "enabled": True,
        "hard_updates_since_promotion": 400,
        "soft_updates_since_promotion": 250,
        "soft_window_bar": 0.55,
        "soft_window_checks": 8,
    }
    base.update(kw)
    return AutoStopConfig(**base)  # type: ignore[arg-type]


class TestEvaluateAutoStopHard:
    def test_hard_fires_at_exactly_400(self) -> None:
        # No window means recorded, so only the hard clause can fire.
        assert evaluate_auto_stop(_cfg(), updates_since_promotion=400, window_means=[]) == "hard"

    def test_hard_does_not_fire_at_399(self) -> None:
        assert evaluate_auto_stop(_cfg(), updates_since_promotion=399, window_means=[]) is None

    def test_disabled_never_fires(self) -> None:
        assert (
            evaluate_auto_stop(
                _cfg(enabled=False), updates_since_promotion=10_000, window_means=[0.1] * 8
            )
            is None
        )


class TestEvaluateAutoStopSoft:
    def test_soft_no_when_below_update_threshold(self) -> None:
        # 247 < 250 → soft cannot fire even with a clearly-dead median.
        assert (
            evaluate_auto_stop(_cfg(), updates_since_promotion=247, window_means=[0.40] * 8) is None
        )

    def test_soft_no_when_median_above_bar(self) -> None:
        # 251 >= 250 and enough checks, but median 0.56 >= bar 0.55 → no.
        assert (
            evaluate_auto_stop(_cfg(), updates_since_promotion=251, window_means=[0.56] * 8) is None
        )

    def test_soft_yes_when_both_clauses_hold(self) -> None:
        # 251 >= 250 AND median 0.54 < bar 0.55 AND 8 recorded checks → soft.
        assert (
            evaluate_auto_stop(_cfg(), updates_since_promotion=251, window_means=[0.54] * 8)
            == "soft"
        )

    def test_soft_no_when_too_few_recorded_checks(self) -> None:
        # median low + updates high, but only 7 recorded checks (< 8) → no.
        assert (
            evaluate_auto_stop(_cfg(), updates_since_promotion=300, window_means=[0.40] * 7) is None
        )

    def test_soft_uses_only_the_last_window_checks(self) -> None:
        # A stale run of high means must not keep the run alive once the recent
        # window drops below the bar (median is over the LAST soft_window_checks).
        means = [0.90] * 8 + [0.50] * 8
        assert evaluate_auto_stop(_cfg(), updates_since_promotion=300, window_means=means) == "soft"


class TestAutoStopTracker:
    def test_promotion_resets_counter_and_clears_means(self) -> None:
        tr = AutoStopTracker(_cfg(), min_games=150)
        for _ in range(300):
            tr.tick()
        tr.record_check(window_wr=0.50, window_n=200)
        tr.record_check(window_wr=0.50, window_n=200)
        assert tr.updates_since_promotion == 300
        assert len(tr.window_means) == 2
        tr.on_promotion()
        assert tr.updates_since_promotion == 0
        assert len(tr.window_means) == 0
        # A promotion mid-hover restarts the clock → no immediate stop.
        assert tr.evaluate() is None

    def test_initial_updates_since_promotion_seeds_counter(self) -> None:
        # Per-lineage (not per-session) plateau clock: a resumed tracker starts
        # at the seeded count, not 0, so the hard-400 clock keeps counting.
        tr = AutoStopTracker(_cfg(), min_games=150, initial_updates_since_promotion=350)
        assert tr.updates_since_promotion == 350
        # 50 more ticks → 400 → hard stop fires (would NOT fire if it reset to 0).
        for _ in range(50):
            tr.tick()
        assert tr.updates_since_promotion == 400
        assert tr.evaluate() == "hard"

    def test_initial_updates_since_promotion_clamped_nonneg(self) -> None:
        tr = AutoStopTracker(_cfg(), min_games=150, initial_updates_since_promotion=-5)
        assert tr.updates_since_promotion == 0

    def test_default_initial_is_zero(self) -> None:
        tr = AutoStopTracker(_cfg(), min_games=150)
        assert tr.updates_since_promotion == 0


class TestResumeUpdatesSincePromotion:
    def test_continues_from_last_promotion(self) -> None:
        # Resumed at update 500, last promoted at 200 → 300 updates since.
        assert _resume_updates_since_promotion(500, 200) == 300

    def test_never_promoted_counts_from_lineage_start(self) -> None:
        # last_promote_update == -1 (never) → full update_idx.
        assert _resume_updates_since_promotion(500, -1) == 500

    def test_clamped_non_negative(self) -> None:
        # A checkpoint written a tick before the promotion counter caught up.
        assert _resume_updates_since_promotion(200, 250) == 0

    def test_record_check_gates_on_full_window(self) -> None:
        tr = AutoStopTracker(_cfg(), min_games=150)
        tr.record_check(window_wr=0.50, window_n=149)  # window short → not recorded
        assert len(tr.window_means) == 0
        tr.record_check(window_wr=0.50, window_n=150)  # full → recorded
        assert len(tr.window_means) == 1

    def test_window_means_is_bounded_to_soft_window_checks(self) -> None:
        tr = AutoStopTracker(_cfg(soft_window_checks=3), min_games=1)
        for i in range(10):
            tr.record_check(window_wr=float(i), window_n=1)
        assert list(tr.window_means) == [7.0, 8.0, 9.0]

    def test_tracker_evaluate_hard(self) -> None:
        tr = AutoStopTracker(_cfg(), min_games=150)
        for _ in range(400):
            tr.tick()
        assert tr.evaluate() == "hard"

    def test_tracker_evaluate_soft(self) -> None:
        tr = AutoStopTracker(_cfg(), min_games=1)
        for _ in range(251):
            tr.tick()
        for _ in range(8):
            tr.record_check(window_wr=0.54, window_n=1)
        assert tr.evaluate() == "soft"


class TestAutoStopConfig:
    def test_default_is_off(self) -> None:
        # Additive: the default TrainConfig carries a disabled auto_stop block.
        cfg = TrainConfig.default()
        assert cfg.auto_stop.enabled is False

    def test_yaml_without_auto_stop_still_loads_off(self, tmp_path) -> None:
        p = tmp_path / "cfg.yaml"
        p.write_text("seed: 3\n")
        cfg = TrainConfig.from_yaml(p)
        assert cfg.auto_stop.enabled is False

    def test_round_trips_through_dict(self) -> None:
        cfg = TrainConfig.default()
        d = cfg.to_dict()
        assert "auto_stop" in d
        assert TrainConfig._from_dict(d).auto_stop == cfg.auto_stop

    def test_validation_rejects_bad_bar_when_enabled(self) -> None:
        with pytest.raises(ValueError, match="soft_window_bar"):
            AutoStopConfig(enabled=True, soft_window_bar=1.5)

    def test_validation_rejects_soft_above_hard(self) -> None:
        with pytest.raises(ValueError, match="soft_updates_since_promotion"):
            AutoStopConfig(
                enabled=True,
                soft_updates_since_promotion=500,
                hard_updates_since_promotion=400,
            )

    def test_disabled_skips_validation(self) -> None:
        # Byte-identical guarantee: a disabled block never rejects (defaults valid).
        AutoStopConfig(enabled=False, soft_window_bar=9.9, soft_window_checks=-3)
