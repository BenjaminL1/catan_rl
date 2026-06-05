"""Tests for `scripts/record_game.py` argparse + validation.

The recorder simulation loop hasn't landed yet (Phases 2a-2d). This
test set covers the CLI validation contract only:

1. Missing ``--ckpt-a`` when ``--player-a=policy`` exits with a clear
   message.
2. ``(policy, policy)`` raises ``NotImplementedError`` at validation
   time (i.e., before the ckpt files are even touched).
3. ``--out`` exists without ``--force`` refuses.
4. Args parse correctly otherwise.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

# Load ``scripts/record_game.py`` directly via importlib so the test
# doesn't permanently inject ``scripts/`` into ``sys.path`` (which
# would shadow ``train``, ``evaluate``, ``migrate_checkpoint``, etc.
# for any other test in the session that imports those names).
_SCRIPT_PATH = Path(__file__).resolve().parents[3] / "scripts" / "record_game.py"
_spec = importlib.util.spec_from_file_location("record_game", _SCRIPT_PATH)
assert _spec is not None and _spec.loader is not None
record_cli = importlib.util.module_from_spec(_spec)
sys.modules["record_game"] = record_cli
_spec.loader.exec_module(record_cli)


class TestCLIValidation:
    def test_policy_without_ckpt_a_raises(self, tmp_path: Path) -> None:
        with pytest.raises(SystemExit, match="--ckpt-a"):
            record_cli.main(
                [
                    "--player-a",
                    "policy",
                    "--player-b",
                    "heuristic",
                    "--out",
                    str(tmp_path / "r.json"),
                ]
            )

    def test_policy_without_ckpt_b_raises(self, tmp_path: Path) -> None:
        with pytest.raises(SystemExit, match="--ckpt-b"):
            record_cli.main(
                [
                    "--player-a",
                    "heuristic",
                    "--player-b",
                    "policy",
                    "--out",
                    str(tmp_path / "r.json"),
                ]
            )

    def test_policy_vs_policy_not_implemented(self, tmp_path: Path) -> None:
        # Even with both --ckpt paths supplied, (policy, policy) is
        # out of v1 scope.
        ckpt = tmp_path / "fake.pt"
        ckpt.write_bytes(b"placeholder")
        with pytest.raises(NotImplementedError, match="policy, policy"):
            record_cli.main(
                [
                    "--player-a",
                    "policy",
                    "--ckpt-a",
                    str(ckpt),
                    "--player-b",
                    "policy",
                    "--ckpt-b",
                    str(ckpt),
                    "--out",
                    str(tmp_path / "r.json"),
                ]
            )

    def test_out_exists_without_force_refuses(self, tmp_path: Path) -> None:
        existing = tmp_path / "r.json"
        existing.write_text("{}")
        with pytest.raises(SystemExit, match="--force"):
            record_cli.main(
                [
                    "--player-a",
                    "heuristic",
                    "--player-b",
                    "heuristic",
                    "--out",
                    str(existing),
                ]
            )

    def test_heuristic_heuristic_currently_unsupported(self, tmp_path: Path) -> None:
        # ``(heuristic, heuristic)`` requires heuristic-as-agent in the
        # recorder driver loop, which is deferred to a follow-up phase.
        # The CLI args parse cleanly and validation passes, but
        # ``record_game`` raises :class:`NotImplementedError` from
        # :func:`_resolve_seat_and_opp`. Tracks Phase 2d's known
        # limitation; remove this test when heuristic-as-agent lands.
        with pytest.raises(NotImplementedError, match="heuristic"):
            record_cli.main(
                [
                    "--player-a",
                    "heuristic",
                    "--player-b",
                    "heuristic",
                    "--out",
                    str(tmp_path / "r.json"),
                ]
            )
