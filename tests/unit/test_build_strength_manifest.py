"""Unit tests for the strength-manifest classifier's pure logic (no OCR/video needed).

Covers the review's BLOCKER surfaces: impostor '...phantom...' rows, tournament-title
over-match, the Global/1v1 gates, and the '#'-sigil / conf / digit-clip rank rules.
"""

from __future__ import annotations

import pathlib
import sys
from typing import Any

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "scripts"))

import build_strength_manifest as m


def _tok(text: str, fx0: float, fy: float, conf: float = 1.0) -> dict[str, Any]:
    """Synthetic OCR token. x0 is derived from fx0 (arbitrary width 1000)."""
    return {"text": text, "conf": conf, "x0": fx0 * 1000, "fx0": fx0, "y": fy * 1000, "fy": fy}


# --- _is_phantom: only the exact handle, never an impostor -----------------------
def test_is_phantom_accepts_canonical() -> None:
    assert m._is_phantom("ThePhantom")
    assert m._is_phantom("thephantom")
    assert m._is_phantom("Th3Phantom")  # digit->cleaned 'thphantom', prefix slip, suffix intact


def test_is_phantom_rejects_impostors() -> None:
    for bad in [
        "PhantomKing",
        "xXPhantomXx",
        "PhantomCatan2",
        "iamphantom",
        "Phantom_GG",
        "ThePhantoms",
        "Phantom99",
    ]:
        assert not m._is_phantom(bad), bad


# --- is_tournament: real 1v1 series only, no metaphor/reaction -------------------
def test_is_tournament_accepts_real_series() -> None:
    for good in [
        "1v1 TITANS TOURNAMENT ROUND 2!!!",
        "THE ELITE 8!!! - Catan World Cup Finals",
        "THE SWEET 16!!! - 1v1 Catan WORLD CUP",
        "Catan - GRAND FINAL 1v1 Invitational Tournament!!!",
    ]:
        assert m.is_tournament(good), good


def test_is_tournament_rejects_metaphor_and_reaction() -> None:
    for bad in [
        "Reacting to a TOURNAMENT game",
        "PLAYOFF PUSH ranked grind",
        "Catan World Cup GUIDE - how to",
        "1v1 CATAN - Climbing The Ranks!!!",
        "Semifinal boardgame arena tier list",
    ]:
        assert not m.is_tournament(bad), bad


# --- 1v1-context gate ------------------------------------------------------------
def test_has_1v1_cue() -> None:
    assert m._has_1v1_cue([_tok("1v1", 0.3, 0.15)])
    assert m._has_1v1_cue([_tok("Iv1", 0.3, 0.15)])  # easyocr variant
    assert not m._has_1v1_cue([_tok("Division", 0.3, 0.1)])  # 'ivi' must not match
    assert not m._has_1v1_cue([_tok("Rating", 0.3, 0.1)])


# --- Global-tab gate -------------------------------------------------------------
def test_global_active() -> None:
    assert m._is_global_active({"global": 192, "oceania": 209, "australia": 210})
    assert m._is_global_active({"global": 192, "friends": 208})  # >=1 other is enough
    assert not m._is_global_active({"global": 201, "australia": 187})  # regional active
    assert not m._is_global_active({"global": 192})  # can't confirm (no other tab)


# --- rank from tokens: the '#'-sigil / conf / ambiguity / digit-clip rules -------
def _phantom_row(fy: float = 0.5) -> dict[str, Any]:
    return _tok("ThePhantom", 0.40, fy)


def test_rank_clean_read() -> None:
    toks = [_phantom_row(), _tok("#8", 0.29, 0.5)]
    assert m._phantom_rank_from_toks(toks) == 8


def test_rank_requires_hash_sigil() -> None:
    # a bare number further left must NOT win over... nothing: no '#' => None
    toks = [_phantom_row(), _tok("2375", 0.29, 0.5), _tok("87", 0.25, 0.5)]
    assert m._phantom_rank_from_toks(toks) is None


def test_rank_hash_beats_stray_bare_number() -> None:
    # a stray bare '0' further left is ignored; the '#32' is the rank
    toks = [_phantom_row(), _tok("0", 0.20, 0.5, conf=0.42), _tok("#32", 0.29, 0.5)]
    assert m._phantom_rank_from_toks(toks) == 32


def test_rank_low_confidence_rejected() -> None:
    toks = [_phantom_row(), _tok("#8", 0.29, 0.5, conf=0.3)]
    assert m._phantom_rank_from_toks(toks) is None


def test_rank_ambiguous_double_phantom_rejected() -> None:
    toks = [_phantom_row(0.5), _tok("#8", 0.29, 0.5), _phantom_row(0.7), _tok("#9", 0.29, 0.7)]
    assert m._phantom_rank_from_toks(toks) is None


def test_rank_profile_name_only_rejected() -> None:
    # phantom only in the header zone (fy<0.13) => not a body row => None
    toks = [_tok("ThePhantom", 0.40, 0.02), _tok("#8", 0.29, 0.02)]
    assert m._phantom_rank_from_toks(toks) is None


def test_rank_three_digit_needs_high_conf() -> None:
    assert m._phantom_rank_from_toks([_phantom_row(), _tok("#150", 0.29, 0.5, conf=0.5)]) is None
    assert m._phantom_rank_from_toks([_phantom_row(), _tok("#150", 0.29, 0.5, conf=0.7)]) == 150
