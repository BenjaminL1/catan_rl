"""Tests for the Colonist player-colour survey (``scripts/color_survey.py`` +
``data/human/color_survey.json``).

The survey widens the openings colour palette beyond GREEN+BLACK (the Tier-5
``hud_unreadable`` NO-GO) with HSV ranges MEASURED from real footage. The
fail-closed abstention guarantee (a seat colour must be either read correctly or
typed-rejected, never mislabelled) rests on the derived ranges being **pairwise
non-overlapping** — two seats in a game must always be separable. These tests pin
that the committed artifact is well-formed and that its ring AND piece ranges are
pairwise non-overlapping 3-D HSV boxes, using the script's own overlap logic (so
the artifact and the derivation code cannot silently drift apart).
"""

from __future__ import annotations

import importlib.util
import itertools
import json
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[3]
_SURVEY = _REPO / "data" / "human" / "color_survey.json"

_SPEC = importlib.util.spec_from_file_location(
    "color_survey", _REPO / "scripts" / "color_survey.py"
)
assert _SPEC is not None and _SPEC.loader is not None
cs = importlib.util.module_from_spec(_SPEC)
sys.modules["color_survey"] = cs
_SPEC.loader.exec_module(cs)


@pytest.fixture(scope="module")
def survey() -> dict:
    return json.loads(_SURVEY.read_text())


def test_survey_well_formed(survey: dict) -> None:
    assert survey["identities"], "no colour identities in the survey"
    # RED + BLACK are the corpus-dominant seats and must always be present + calibrated.
    for required in ("RED", "BLACK"):
        assert required in survey["identities"], f"missing {required}"
        assert survey["identities"][required]["harvest_exclude"] is False
    for cid, e in survey["identities"].items():
        for field in ("ring", "piece"):
            box = e[field]
            for axis in ("hue", "sat", "val"):
                lo, hi = box[axis]
                assert 0 <= lo <= 255 and 0 <= hi <= 255, f"{cid}.{field}.{axis} out of range"
                if axis != "hue":  # sat/val never wrap; hue may (RED)
                    assert lo <= hi, f"{cid}.{field}.{axis} lo>hi but does not wrap"
        assert isinstance(e["n_videos"], int)
        assert e["kind"] in ("chromatic", "achromatic")
        assert e["harvest_exclude"] == (e["n_videos"] < survey["calibrated_min_videos"])


def test_red_hue_wraps(survey: dict) -> None:
    # RED straddles the 0/180 seam, so its hue range must be stored as lo > hi.
    red_ring = survey["identities"]["RED"]["ring"]["hue"]
    assert red_ring[0] > red_ring[1], f"RED ring hue should wrap, got {red_ring}"


@pytest.mark.parametrize("field", ["ring", "piece"])
def test_ranges_pairwise_non_overlapping(survey: dict, field: str) -> None:
    """The core guarantee: any two colour identities' HSV boxes are disjoint, so a
    two-seat game is always separable and a fail-closed reader can never mislabel one
    seat's colour as the other's. Enforced for BOTH the HUD ring and the piece body,
    across ALL identities (including the low-sample/harvest-excluded ones — an
    excluded colour must still never *collide* with an included one)."""
    ids = survey["identities"]
    for a, b in itertools.combinations(sorted(ids), 2):
        box_a, box_b = ids[a][field], ids[b][field]
        assert not cs._box_overlap(box_a, box_b), (
            f"{field} HSV boxes overlap: {a}={box_a} vs {b}={box_b} — two seats using "
            "these colours would not be separable (fail-closed abstention broken)"
        )


def test_hue_overlap_helper_handles_wraparound() -> None:
    # A wraparound range [170, 10] overlaps a low-hue range but not a mid-hue one.
    assert cs._hue_overlap([170, 10], [5, 15]) is True
    assert cs._hue_overlap([170, 10], [175, 179]) is True
    assert cs._hue_overlap([170, 10], [40, 60]) is False
    assert cs._hue_overlap([20, 30], [40, 60]) is False
    assert cs._hue_overlap([20, 45], [40, 60]) is True


def test_derive_reproduces_committed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The artifact is regenerable: re-running ``derive`` on the committed raw
    measurements reproduces the committed ``color_survey.json`` identities/ranges
    byte-for-byte (the ranges are a deterministic function of the measurements, not
    hand-edited)."""
    if not cs.RAW_OUT.exists():
        pytest.skip("raw measurements not present")
    out_json = tmp_path / "survey.json"
    out_md = tmp_path / "survey.md"
    monkeypatch.setattr(cs, "SURVEY_JSON", out_json)
    monkeypatch.setattr(cs, "SURVEY_MD", out_md)
    assert cs.derive() == 0  # also runs the internal _assert_non_overlap
    regenerated = json.loads(out_json.read_text())
    committed = json.loads(_SURVEY.read_text())
    assert regenerated["identities"] == committed["identities"]
