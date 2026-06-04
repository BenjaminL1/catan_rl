"""Tests for `scripts/migrate_checkpoint.py`.

Pins:
1. Already-at-current → exit 0, prints "already at vN", does not write.
2. --force re-saves even if already current.
3. Pre-versioned input → exit 1.
4. Non-dict input → exit 1.
5. Chain of registered migrations is applied end-to-end.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any

import pytest
import torch

from catan_rl.checkpoint.manager import SCHEMA_VERSION
from catan_rl.checkpoint.migrations import (
    register_migration,
    registered_versions,
    unregister_migration,
)

# Re-import the CLI module so we can call ``main()`` directly without a
# subprocess (faster + easier to inspect).
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
migrate_cli = importlib.import_module("migrate_checkpoint")


@pytest.fixture(autouse=True)
def _clean_registry():
    before = registered_versions()
    yield
    after = registered_versions()
    for v in after:
        if v not in before:
            unregister_migration(v)


class TestCli:
    def test_already_current_no_write(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        src = tmp_path / "in.pt"
        torch.save({"schema_version": SCHEMA_VERSION, "foo": 1}, src)
        out = tmp_path / "out.pt"
        rc = migrate_cli.main(["--in", str(src), "--out", str(out)])
        assert rc == 0
        cap = capsys.readouterr()
        assert f"already at v{SCHEMA_VERSION}" in cap.out
        assert not out.exists()

    def test_force_resaves_when_current(self, tmp_path: Path) -> None:
        src = tmp_path / "in.pt"
        torch.save({"schema_version": SCHEMA_VERSION, "foo": 1}, src)
        out = tmp_path / "out.pt"
        rc = migrate_cli.main(["--in", str(src), "--out", str(out), "--force"])
        assert rc == 0
        assert out.exists()
        re_loaded = torch.load(out, map_location="cpu", weights_only=False)
        assert re_loaded["schema_version"] == SCHEMA_VERSION

    def test_missing_schema_version_exits_1(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        src = tmp_path / "in.pt"
        torch.save({"no_version": True}, src)
        rc = migrate_cli.main(["--in", str(src), "--out", str(tmp_path / "out.pt")])
        assert rc == 1
        cap = capsys.readouterr()
        assert "missing 'schema_version'" in cap.err

    def test_chain_applied(self, tmp_path: Path) -> None:
        # Register two fake migrations covering v(SCHEMA-2) -> SCHEMA.
        target = SCHEMA_VERSION

        def step(p: dict[str, Any]) -> dict[str, Any]:
            return {**p, "schema_version": p["schema_version"] + 1}

        if target >= 2:
            register_migration(target - 2, step)
            register_migration(target - 1, step)
            src = tmp_path / "in.pt"
            torch.save({"schema_version": target - 2}, src)
            out = tmp_path / "out.pt"
            rc = migrate_cli.main(["--in", str(src), "--out", str(out)])
            assert rc == 0
            re_loaded = torch.load(out, map_location="cpu", weights_only=False)
            assert re_loaded["schema_version"] == target
        else:
            pytest.skip(
                "SCHEMA_VERSION too low to exercise the multi-step chain "
                "(needs >=2 to register two stages)."
            )

    def test_missing_input_exits_1(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        rc = migrate_cli.main(
            [
                "--in",
                str(tmp_path / "nope.pt"),
                "--out",
                str(tmp_path / "out.pt"),
            ]
        )
        assert rc == 1
        cap = capsys.readouterr()
        assert "not found" in cap.err

    def test_non_dict_input_exits_1(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        src = tmp_path / "in.pt"
        torch.save([1, 2, 3], src)
        rc = migrate_cli.main(["--in", str(src), "--out", str(tmp_path / "out.pt")])
        assert rc == 1
        cap = capsys.readouterr()
        assert "not a dict" in cap.err
