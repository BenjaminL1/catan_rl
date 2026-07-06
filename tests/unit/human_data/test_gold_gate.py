"""Gold-gate tooling tests (the 30-game blind-labeling exam apparatus, step6 §3).

Covers the three deliverables of ``catan_rl.human_data.gold_gate``:

- ``prepare`` on 2 games writes COMPLETE blind packets and the blindness verifier
  confirms no pipeline-derived content leaked into them (+ the answer key lands in
  the sibling ``_answers/`` tree, not the packet);
- ``score`` on a synthetic hand-made label computes the correct per-field
  accuracies + verdict (a perfect label is READY; a label with known errors falls
  below the bars);
- the D6 orientation-flip detector fires on a rotated board and not on the truth;
- the reference-grid renderer writes a real PNG (headless matplotlib).

The matplotlib-backed paths (frame PNGs + reference grid) ``importorskip`` the dep;
CI installs it via ``.[dev]`` so they run rather than skip.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from catan_rl.human_data.gold_gate import (
    ANSWERS_DIRNAME,
    BOARD_BAR,
    OPENINGS_BAR,
    GoldFrames,
    PacketNotBlindError,
    assert_packet_blind,
    blank_label_template,
    d6_hex_permutations,
    game_id,
    is_orientation_flip,
    prepare_packets,
    render_score_report,
    score_gold,
    select_gold_games,
    write_answer_key,
    write_blind_packet,
    write_score_report,
)
from catan_rl.human_data.record import GameRecord, OpponentStrength, PlayerOpening

# --- a legal standard game-1 board + openings (mirrors test_harvest/test_batch) ---

_HEXES: tuple[dict[str, Any], ...] = (
    {"hex_id": 0, "resource": "SHEEP", "number": 11},
    {"hex_id": 1, "resource": "BRICK", "number": 9},
    {"hex_id": 2, "resource": "WHEAT", "number": 10},
    {"hex_id": 3, "resource": "WHEAT", "number": 3},
    {"hex_id": 4, "resource": "BRICK", "number": 6},
    {"hex_id": 5, "resource": "SHEEP", "number": 5},
    {"hex_id": 6, "resource": "ORE", "number": 4},
    {"hex_id": 7, "resource": "WOOD", "number": 6},
    {"hex_id": 8, "resource": "SHEEP", "number": 2},
    {"hex_id": 9, "resource": "WOOD", "number": 5},
    {"hex_id": 10, "resource": "BRICK", "number": 8},
    {"hex_id": 11, "resource": "DESERT", "number": None},
    {"hex_id": 12, "resource": "SHEEP", "number": 4},
    {"hex_id": 13, "resource": "WOOD", "number": 11},
    {"hex_id": 14, "resource": "WHEAT", "number": 12},
    {"hex_id": 15, "resource": "ORE", "number": 9},
    {"hex_id": 16, "resource": "ORE", "number": 10},
    {"hex_id": 17, "resource": "WOOD", "number": 8},
    {"hex_id": 18, "resource": "WHEAT", "number": 3},
)


def _make_record(
    *,
    video_id: str = "vidAAAAAAAA",
    game_index: int = 1,
    winner: str | None = "ThePhantom",
    passed: bool = True,
    order_established: bool = True,
) -> GameRecord:
    provenance: dict[str, Any] = {
        "board_desert_hex": 11,
        "openings_desert_hex": 11,
        "resolution": 1080,
        "placement_order_established": order_established,
        "ts": 247,
    }
    return GameRecord(
        video_id=video_id,
        game_index=game_index,
        players={"agent": "ThePhantom", "opponent": "rayman147"},
        opponent_strength=OpponentStrength(tier="high", source="rank_badge", confidence=0.95),
        ruleset={"num_players": 2, "win_vp": 15},
        hexes=_HEXES,
        draft_order=("ThePhantom", "rayman147", "rayman147", "ThePhantom"),
        openings={
            "ThePhantom": PlayerOpening(settlements=(1, 19), roads=(0, 35)),
            "rayman147": PlayerOpening(settlements=(11, 3), roads=(19, 8)),
        },
        dice_log=(8, 6, 5) if winner is not None else (),
        winner=winner,
        episode_source="natural",
        passed_crosscheck=passed,
        provenance=provenance,
        rejection_reason=None if passed else "hud_unreadable",
    )


def _frames() -> GoldFrames:
    """Tiny synthetic RGB frames (the raw pixels, not pipeline output)."""
    rng = np.random.default_rng(0)
    frame = rng.integers(0, 256, size=(12, 16, 3), dtype=np.uint8)
    return GoldFrames(
        post_setup=frame,
        mid_setup=(frame.copy(), frame.copy()),
        setup_log_crop=frame[:6],
        terminal_log_crop=frame[6:],
    )


def _perfect_label(record: GameRecord) -> dict[str, Any]:
    """A hand-made label that matches the record exactly (a perfect exam)."""
    return {
        "game_id": game_id(record),
        "board": {
            str(h["hex_id"]): {"resource": h["resource"], "number": h.get("number")}
            for h in record.hexes
        },
        "openings": {
            "pov": {
                "settlements": list(record.openings["ThePhantom"].settlements),
                "roads": list(record.openings["ThePhantom"].roads),
            },
            "opponent": {
                "settlements": list(record.openings["rayman147"].settlements),
                "roads": list(record.openings["rayman147"].roads),
            },
        },
        "winner": "pov",
    }


# --- selection ---------------------------------------------------------------


def test_select_gold_games_accepted_only_deduped_capped() -> None:
    r_accept = _make_record(video_id="vidAAAAAAAA", game_index=1, passed=True)
    r_dupe = _make_record(video_id="vidAAAAAAAA", game_index=1, passed=True)  # same game_id
    r_reject = _make_record(video_id="vidBBBBBBBB", game_index=1, passed=False, winner=None)
    r_accept2 = _make_record(video_id="vidCCCCCCCC", game_index=1, passed=True)

    selected = select_gold_games([r_accept, r_dupe, r_reject, r_accept2], count=30)
    ids = [game_id(r) for r in selected]
    assert ids == ["vidAAAAAAAA__g1", "vidCCCCCCCC__g1"]  # dupe + reject dropped

    capped = select_gold_games([r_accept, r_accept2], count=1)
    assert len(capped) == 1


# --- reference grid + D6 ------------------------------------------------------


def test_d6_permutations_are_the_full_group() -> None:
    perms = d6_hex_permutations()
    assert len(perms) == 12
    identity = tuple(range(19))
    assert identity in perms
    for perm in perms:
        assert sorted(perm) == list(range(19))  # every element is a bijection


def test_render_reference_grid_writes_png(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    from catan_rl.human_data.gold_gate import render_reference_grid

    out = tmp_path / "grid.png"
    render_reference_grid(out)
    assert out.exists()
    assert out.stat().st_size > 0
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"  # PNG magic


# --- prepare: blind packets --------------------------------------------------


def test_prepare_two_games_writes_complete_blind_packets(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    records = [
        _make_record(video_id="vidAAAAAAAA", game_index=1),
        _make_record(video_id="vidBBBBBBBB", game_index=1),
    ]

    def provider(_r: GameRecord) -> GoldFrames:
        return _frames()

    packets = prepare_packets(records, provider, tmp_path, count=30)
    assert len(packets) == 2

    for record, packet in zip(records, packets, strict=True):
        # every required blind artifact is present
        for name in (
            "post_setup.png",
            "mid_setup_1.png",
            "mid_setup_2.png",
            "log_setup.png",
            "log_terminal.png",
            "reference_grid.png",
            "label_template.json",
            "README.md",
        ):
            assert (packet / name).exists(), f"missing {name} in {packet}"
        # the packet is BLIND: nothing derived from the pipeline's parse
        assert_packet_blind(packet, record)
        # the answer key lives OUTSIDE the packet, in the sibling _answers tree
        answer = tmp_path / ANSWERS_DIRNAME / f"{game_id(record)}.json"
        assert answer.exists()
        assert not (packet / ANSWERS_DIRNAME).exists()


def test_assert_packet_blind_catches_leaks(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    record = _make_record()
    packet = write_blind_packet(record, _frames(), tmp_path)
    assert_packet_blind(packet, record)  # clean packet passes

    # (1) a stray answer-shaped file trips the allowlist
    stray = packet / "answers.json"
    stray.write_text("{}", encoding="utf-8")
    with pytest.raises(PacketNotBlindError):
        assert_packet_blind(packet, record)
    stray.unlink()

    # (2) a FILLED label (answer leaked into the template) trips the blank check
    label_path = packet / "label_template.json"
    label = json.loads(label_path.read_text(encoding="utf-8"))
    label["winner"] = "pov"
    label_path.write_text(json.dumps(label), encoding="utf-8")
    with pytest.raises(PacketNotBlindError):
        assert_packet_blind(packet, record)

    # (3) a text file embedding the record's board answer trips the token scan
    label_path.write_text(json.dumps(blank_label_template(game_id(record))), encoding="utf-8")
    assert_packet_blind(packet, record)  # restored to blank → passes again
    board_answer = json.dumps(
        [{"resource": h["resource"], "number": h.get("number")} for h in record.hexes],
        sort_keys=True,
    )
    (packet / "README.md").write_text(board_answer, encoding="utf-8")
    with pytest.raises(PacketNotBlindError):
        assert_packet_blind(packet, record)


# --- score -------------------------------------------------------------------


def test_score_perfect_label_is_ready(tmp_path: Path) -> None:
    record = _make_record()
    write_answer_key(record, tmp_path / ANSWERS_DIRNAME)
    # labeler fills the label in a separate labels dir
    labels = tmp_path / "labels"
    labels.mkdir()
    (labels / f"{game_id(record)}.json").write_text(
        json.dumps(_perfect_label(record)), encoding="utf-8"
    )

    report = score_gold(tmp_path, labels_dir=labels)
    assert report.n_games == 1
    assert report.board.correct == 19 and report.board.total == 19
    assert report.openings.correct == 8 and report.openings.total == 8
    assert report.winner.correct == 1 and report.winner.total == 1
    assert report.orientation_flips == 0
    assert report.ready
    assert report.failures() == []


def test_score_label_with_errors_computes_accuracies_and_fails(tmp_path: Path) -> None:
    record = _make_record()
    write_answer_key(record, tmp_path / ANSWERS_DIRNAME)
    labels = tmp_path / "labels"
    labels.mkdir()

    label = _perfect_label(record)
    # one wrong hex (flip resource on hex 0) → board 18/19
    label["board"]["0"]["resource"] = "ORE"
    # one wrong opening placement (pov first settlement) → openings 7/8
    label["openings"]["pov"]["settlements"][0] = 42
    # wrong winner → winner 0/1
    label["winner"] = "opponent"
    (labels / f"{game_id(record)}.json").write_text(json.dumps(label), encoding="utf-8")

    report = score_gold(tmp_path, labels_dir=labels)
    assert report.board.correct == 18 and report.board.total == 19
    assert report.openings.correct == 7 and report.openings.total == 8
    assert report.winner.correct == 0 and report.winner.total == 1
    assert not report.board.passed  # 18/19 = 0.947 < 0.98
    assert not report.openings.passed  # 7/8 = 0.875 < 0.95
    assert not report.winner.passed
    assert not report.ready
    assert any("board" in f for f in report.failures())


def test_score_skips_unlabeled_games(tmp_path: Path) -> None:
    record = _make_record()
    write_answer_key(record, tmp_path / ANSWERS_DIRNAME)
    labels = tmp_path / "labels"
    labels.mkdir()
    # a BLANK (unfilled) label must be skipped, not scored as all-wrong
    (labels / f"{game_id(record)}.json").write_text(
        json.dumps(blank_label_template(game_id(record))), encoding="utf-8"
    )
    report = score_gold(tmp_path, labels_dir=labels)
    assert report.n_games == 0
    assert game_id(record) in report.skipped_unlabeled
    assert not report.ready  # no labeled games → not ready


def test_score_report_writes_markdown(tmp_path: Path) -> None:
    record = _make_record()
    write_answer_key(record, tmp_path / ANSWERS_DIRNAME)
    labels = tmp_path / "labels"
    labels.mkdir()
    (labels / f"{game_id(record)}.json").write_text(
        json.dumps(_perfect_label(record)), encoding="utf-8"
    )
    report = score_gold(tmp_path, labels_dir=labels)
    text = render_score_report(report)
    assert "Gold-Gate Report" in text
    assert "READY" in text
    assert f"{BOARD_BAR}" in text and f"{OPENINGS_BAR}" in text

    report_path = tmp_path / "gold_gate_report.md"
    written = write_score_report(report, report_path)
    assert written.exists()
    assert "orientation_flips" in written.read_text(encoding="utf-8")


# --- orientation flip --------------------------------------------------------


def test_orientation_flip_detected_on_rotated_board() -> None:
    record = _make_record()
    perms = d6_hex_permutations()
    identity = tuple(range(19))
    nonident = next(p for p in perms if tuple(p) != identity)

    record_cells = {int(h["hex_id"]): (h["resource"], h.get("number")) for h in record.hexes}
    # build a label whose board is the record RELABELED by a non-identity D6 element
    flipped_label = {
        "board": {
            str(h): {
                "resource": record_cells[nonident[h]][0],
                "number": record_cells[nonident[h]][1],
            }
            for h in range(19)
        },
        "openings": {},
        "winner": None,
    }
    assert is_orientation_flip(flipped_label, record, perms)

    # the truthful (identity) board is NOT a flip
    truthful = {
        "board": {
            str(h): {"resource": record_cells[h][0], "number": record_cells[h][1]}
            for h in range(19)
        },
        "openings": {},
        "winner": None,
    }
    assert not is_orientation_flip(truthful, record, perms)


def test_score_counts_orientation_flip(tmp_path: Path) -> None:
    record = _make_record()
    write_answer_key(record, tmp_path / ANSWERS_DIRNAME)
    labels = tmp_path / "labels"
    labels.mkdir()

    perms = d6_hex_permutations()
    nonident = next(p for p in perms if tuple(p) != tuple(range(19)))
    record_cells = {int(h["hex_id"]): (h["resource"], h.get("number")) for h in record.hexes}
    label = _perfect_label(record)
    label["board"] = {
        str(h): {"resource": record_cells[nonident[h]][0], "number": record_cells[nonident[h]][1]}
        for h in range(19)
    }
    (labels / f"{game_id(record)}.json").write_text(json.dumps(label), encoding="utf-8")

    report = score_gold(tmp_path, labels_dir=labels)
    assert report.orientation_flips == 1
    assert not report.flips_ok
    assert not report.ready
