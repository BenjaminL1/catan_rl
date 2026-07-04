"""Scaffold tests for the human-data pipeline (build brief §3, §6).

Test-first guarantees for the ``scaffold`` slice:

- ``GameRecord`` (de)serializes round-trip and always carries ``schema_version``.
- the committed ``topology.json`` package fixture has the standard-board
  19 / 54 / 72 / 9 counts.
- the committed game-1 golden CI fixtures exist and load.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from catan_rl.human_data import (
    SCHEMA_VERSION,
    FFmpegNotFoundError,
    GameRecord,
    OpponentStrength,
    PlayerOpening,
    check_road_incidence,
    derive_opponent_strength,
    load_topology,
    resolve_ffmpeg,
)
from catan_rl.human_data.topology import (
    NUM_EDGES,
    NUM_HEXES,
    NUM_PORTS,
    NUM_VERTICES,
)

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "human_data"


# The real game-1 board (spike artifact
# ``blockers/board/locked_board_240.json``): a legal standard 19-tile board, so
# ``_sample_record()`` is itself a valid board (not the old all-ORE stub which the
# multiset gate now correctly rejects). Desert is hex 11.
_GAME1_HEXES: tuple[dict[str, object], ...] = (
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


def _sample_record() -> GameRecord:
    """A representative ``GameRecord`` built from the game-1 spike result."""
    return GameRecord(
        video_id="9Sm86ml04aI",
        game_index=1,
        players={"agent": "ThePhantom", "opponent": "rayman147"},
        # Faithful to the committed strength manifest: video 9Sm86ml04aI IS the
        # tournament entry (strength="high", source="tournament"). Using a real
        # manifest-backed high source (not the unfalsifiable known_window
        # placeholder) keeps the golden scoreboard-eligible under the tightened
        # is_scoreboard_eligible() source clause.
        opponent_strength=OpponentStrength(tier="high", source="tournament", confidence=0.8),
        ruleset={"num_players": 2, "win_vp": 15},
        hexes=_GAME1_HEXES,
        draft_order=("rayman147", "ThePhantom", "ThePhantom", "rayman147"),
        # Openings RE-SNAPPED under the locked desert=11 affine (orient_lock2 PART A
        # screen rule). The earlier desert=17 IDs (TP s[4,10] r[7,20]; ray s[20,0]
        # r[34,2]) were snapped under a REJECTED D6 orientation and were physically
        # wrong; see ``tests/fixtures/human_data/game1_resnap_overlay.png``.
        openings={
            "ThePhantom": PlayerOpening(settlements=(1, 19), roads=(0, 35)),
            "rayman147": PlayerOpening(settlements=(11, 3), roads=(19, 8)),
        },
        dice_log=(8, 6, 11, 4),
        # Game 1 (the t=80..620 segment) was won by rayman147, NOT ThePhantom:
        # inside game 1's window the terminal LOG (spike scan_60_1200.json t=600)
        # reads "rayman147 built a Settlement (+1 VP)" then "ThePhantom: gg"
        # (ThePhantom conceding). The only "ThePhantom won the game" line (t=1160)
        # belongs to a LATER game. Per spec §5.1 winner = the victory LOG line
        # ONLY, so ThePhantom (the POV) LOST game 1. The earlier winner label was
        # inverted (a later game's win line welded onto game 1).
        winner="rayman147",
        episode_source="natural",
        passed_crosscheck=True,
        # board + openings both locked under the desert=11 orientation (schema v2
        # orientation-binding). board_desert_hex == openings_desert_hex == 11.
        provenance={
            "resolution": 1080,
            "ts": 247,
            "board_desert_hex": 11,
            "openings_desert_hex": 11,
        },
        rejection_reason=None,
    )


def test_game_record_roundtrip_via_dict() -> None:
    rec = _sample_record()
    restored = GameRecord.from_dict(rec.to_dict())
    assert restored == rec


def test_game_record_roundtrip_via_json_line() -> None:
    rec = _sample_record()
    line = rec.to_json_line()
    assert "\n" not in line
    restored = GameRecord.from_json_line(line)
    assert restored == rec


def test_game_record_schema_version_present() -> None:
    rec = _sample_record()
    assert rec.schema_version == SCHEMA_VERSION == 2
    payload = rec.to_dict()
    assert payload["schema_version"] == 2
    assert json.loads(rec.to_json_line())["schema_version"] == 2


def test_game_record_resources_are_string_literals() -> None:
    """Resources are bare strings (no enum) — desert carries number=None."""
    payload = _sample_record().to_dict()
    desert = next(h for h in payload["board"]["hexes"] if h["resource"] == "DESERT")
    assert desert["resource"] == "DESERT"
    assert desert["number"] is None
    assert isinstance(payload["board"]["hexes"][0]["resource"], str)


def test_game_record_ports_omitted_v1() -> None:
    assert _sample_record().to_dict()["board"]["ports"] == "OMITTED in v1"


def test_game_record_rejects_newer_schema() -> None:
    payload = _sample_record().to_dict()
    payload["schema_version"] = SCHEMA_VERSION + 1
    with pytest.raises(ValueError, match="newer than supported"):
        GameRecord.from_dict(payload)


def test_topology_fixture_counts() -> None:
    topo = load_topology()
    assert len(topo.hex_corner_to_vertex) == NUM_HEXES == 19
    assert len(topo.vertex_adjacent_hexes) == NUM_VERTICES == 54
    assert len(topo.edge_vertices) == NUM_EDGES == 72
    assert len(topo.port_slots) == NUM_PORTS == 9
    # Every edge is an ascending vertex pair within range.
    for a, b in topo.edge_vertices:
        assert 0 <= a < b < NUM_VERTICES


def test_golden_fixtures_exist_and_load() -> None:
    postsetup = FIXTURES / "game1_postsetup_t247.png"
    baseline = FIXTURES / "game1_empty_baseline_t105.png"
    log_crop = FIXTURES / "game1_log_crop_t120.png"
    openings = FIXTURES / "game1_openings.json"
    ocr = FIXTURES / "ocr_f1080_120.txt"
    for png in (postsetup, baseline, log_crop):
        assert png.is_file(), png
        # PNG magic bytes.
        assert png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n", png

    parsed = json.loads(openings.read_text(encoding="utf-8"))
    # Golden IDs are the desert=11 re-snap (the desert=17 IDs were wrong).
    assert parsed["fit"]["desert_hex"] == 11
    # Game 1's winner is rayman147 (POV ThePhantom LOST) — the earlier
    # winner="ThePhantom" label welded a LATER game's victory line onto game 1.
    assert parsed["winner"] == "rayman147"
    assert parsed["openings"]["ThePhantom"]["settlements"] == [1, 19]
    assert parsed["openings"]["ThePhantom"]["roads"] == [0, 35]
    assert parsed["openings"]["rayman147"]["settlements"] == [11, 3]
    assert parsed["openings"]["rayman147"]["roads"] == [19, 8]

    text = ocr.read_text(encoding="utf-8")
    # The real noisy OCR carries the "Happy settlingl" typo and setup events.
    assert "Happy settling" in text
    assert "placed a Settlement" in text


# --- contract firewall: GameRecord.validate (resolve pass) ------------------


def test_validate_accepts_the_sample_record() -> None:
    # The representative record must pass the firewall unchanged.
    _sample_record().validate()


def test_validate_rejects_illegal_resource() -> None:
    payload = _sample_record().to_dict()
    payload["board"]["hexes"][0]["resource"] = "GOLD"
    with pytest.raises(ValueError, match="GOLD"):
        GameRecord.from_dict(payload)


def test_validate_rejects_desert_with_number() -> None:
    payload = _sample_record().to_dict()
    # Hex 11 is the game-1 desert; a desert must carry number=None. Drop the
    # paired non-desert token so only the desert-with-number rule fires (the
    # multiset gate would otherwise mask it).
    desert = next(h for h in payload["board"]["hexes"] if h["resource"] == "DESERT")
    desert["number"] = 7  # robber-only token, never a real chip → unambiguous
    with pytest.raises(ValueError, match="desert"):
        GameRecord.from_dict(payload)


def test_validate_rejects_nondesert_with_bad_number() -> None:
    payload = _sample_record().to_dict()
    payload["board"]["hexes"][0]["number"] = 7  # robber-only, never a token
    with pytest.raises(ValueError, match="number"):
        GameRecord.from_dict(payload)


def test_validate_rejects_hex_id_out_of_range() -> None:
    payload = _sample_record().to_dict()
    payload["board"]["hexes"][0]["hex_id"] = 99
    with pytest.raises(ValueError, match="hex_ids"):
        GameRecord.from_dict(payload)


def test_validate_rejects_duplicate_hex_ids() -> None:
    payload = _sample_record().to_dict()
    payload["board"]["hexes"][1]["hex_id"] = 0  # multiset no longer 0..18
    with pytest.raises(ValueError, match="hex_ids"):
        GameRecord.from_dict(payload)


def test_validate_rejects_settlement_vertex_out_of_range() -> None:
    payload = _sample_record().to_dict()
    payload["openings"]["ThePhantom"]["settlements"] = [1, 54]  # 54 == NUM_VERTICES
    with pytest.raises(ValueError, match="settlement vertex"):
        GameRecord.from_dict(payload)


def test_validate_rejects_road_edge_out_of_range() -> None:
    payload = _sample_record().to_dict()
    payload["openings"]["ThePhantom"]["roads"] = [0, 72]  # 72 == NUM_EDGES
    with pytest.raises(ValueError, match="road edge"):
        GameRecord.from_dict(payload)


def test_validate_rejects_non_1v1_ruleset() -> None:
    for bad in ({"num_players": 4, "win_vp": 15}, {"num_players": 2, "win_vp": 10}):
        payload = _sample_record().to_dict()
        payload["ruleset"] = bad
        with pytest.raises(ValueError, match="1v1-locked"):
            GameRecord.from_dict(payload)


def test_validate_rejects_unknown_winner() -> None:
    payload = _sample_record().to_dict()
    payload["winner"] = "somebody_else"
    with pytest.raises(ValueError, match="winner"):
        GameRecord.from_dict(payload)


def test_validate_allows_null_winner() -> None:
    payload = _sample_record().to_dict()
    payload["winner"] = None  # resign / cutoff — valid, just scoreboard-ineligible
    assert GameRecord.from_dict(payload).winner is None


def test_validate_rejects_opening_key_not_a_player() -> None:
    payload = _sample_record().to_dict()
    payload["openings"]["ghost"] = {"settlements": [1, 2], "roads": [3, 4]}
    with pytest.raises(ValueError, match="openings keys"):
        GameRecord.from_dict(payload)


def test_validate_rejects_bad_episode_source() -> None:
    payload = _sample_record().to_dict()
    payload["episode_source"] = "synthetic"
    with pytest.raises(ValueError, match="episode_source"):
        GameRecord.from_dict(payload)


def test_validate_rejects_bad_opponent_strength_tier() -> None:
    payload = _sample_record().to_dict()
    payload["opponent_strength"]["tier"] = "medium"
    with pytest.raises(ValueError, match="tier"):
        GameRecord.from_dict(payload)


def test_opponent_strength_accepts_tournament_source() -> None:
    # Manifest reconciliation: the strength manifest's ``source="tournament"``
    # is carried through 1:1 as an OpponentStrength.source (additive to the
    # existing rank_badge/known_window set). A tournament game is tier="high".
    rec = _sample_record()
    payload = rec.to_dict()
    payload["opponent_strength"] = {"tier": "high", "source": "tournament", "confidence": 0.9}
    restored = GameRecord.from_dict(payload)
    assert restored.opponent_strength.source == "tournament"
    assert restored.opponent_strength.tier == "high"


def test_opponent_strength_accepts_rank_badge_source() -> None:
    # Manifest reconciliation: manifest ``source="ranked_rank"`` maps to
    # ``source="rank_badge"`` (segment.py owns that mapping); the reconciled
    # literal must validate.
    payload = _sample_record().to_dict()
    payload["opponent_strength"] = {"tier": "high", "source": "rank_badge", "confidence": 0.85}
    restored = GameRecord.from_dict(payload)
    assert restored.opponent_strength.source == "rank_badge"


def test_validate_rejects_manifest_none_source() -> None:
    # The manifest's third source, "none" (unknown/excluded), is NOT a valid
    # OpponentStrength.source — an unknown-strength game gets tier="unknown", it
    # must not carry a raw "none" source string through the firewall.
    payload = _sample_record().to_dict()
    payload["opponent_strength"] = {"tier": "unknown", "source": "none", "confidence": 0.0}
    with pytest.raises(ValueError, match="source"):
        GameRecord.from_dict(payload)


# --- manifest -> OpponentStrength derivation (the high/unknown/excluded gate) -
# The single most dangerous §5.5 trap: the derivation MUST key on the manifest
# `strength` field, NOT `source` — 36 excluded videos carry source="ranked_rank"
# (rank > 200), and a naive source-keyed mapping would turn a rank-288 opponent
# into a confidently-wrong top-tier record. These pin ALL FOUR observed manifest
# combos (cross-tab: ('high','tournament'), ('high','ranked_rank'),
# ('excluded','ranked_rank'), ('unknown','none')).


def test_derive_strength_high_tournament() -> None:
    got = derive_opponent_strength({"strength": "high", "source": "tournament"})
    assert got is not None
    assert got.tier == "high"
    assert got.source == "tournament"


def test_derive_strength_high_ranked_rank_maps_to_rank_badge() -> None:
    got = derive_opponent_strength({"strength": "high", "source": "ranked_rank"})
    assert got is not None
    assert got.tier == "high"
    # segment.py's ranked_rank -> rank_badge reconciliation lives in the helper.
    assert got.source == "rank_badge"


def test_derive_strength_excluded_ranked_rank_returns_none() -> None:
    # THE trap: rank > 200 -> excluded despite source="ranked_rank". Must be NO
    # record (None), NOT a high record. This is the rank-288 confidently-wrong case.
    got = derive_opponent_strength({"strength": "excluded", "source": "ranked_rank"})
    assert got is None


def test_derive_strength_unknown_none() -> None:
    got = derive_opponent_strength({"strength": "unknown", "source": "none"})
    assert got is not None
    assert got.tier == "unknown"
    # An unknown-strength game is never scoreboard-eligible regardless of source.


def test_derive_strength_keys_on_strength_not_source() -> None:
    # Same source ("ranked_rank"), opposite strength -> opposite outcome. This is
    # the assertion that the derivation gates on `strength` first, never `source`.
    high = derive_opponent_strength({"strength": "high", "source": "ranked_rank"})
    excluded = derive_opponent_strength({"strength": "excluded", "source": "ranked_rank"})
    assert high is not None and high.tier == "high"
    assert excluded is None


def test_derive_strength_matches_committed_manifest_combos() -> None:
    # Exercise the derivation against the ACTUAL committed manifest: every entry's
    # (strength, source) must derive without error, excluded -> None, high -> high,
    # unknown -> unknown, and the rank-288-style excluded videos really exist there.
    manifest_path = (
        Path(__file__).resolve().parents[3] / "data" / "human" / "strength_manifest.json"
    )
    if not manifest_path.is_file():
        pytest.skip("committed strength manifest not present")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    videos = manifest["videos"]
    seen_excluded_ranked = False
    for entry in videos:
        got = derive_opponent_strength(entry)
        if entry["strength"] == "excluded":
            assert got is None
            if entry["source"] == "ranked_rank":
                seen_excluded_ranked = True
        elif entry["strength"] == "high":
            assert got is not None and got.tier == "high"
            assert got.source in ("rank_badge", "tournament")
        else:
            assert entry["strength"] == "unknown"
            assert got is not None and got.tier == "unknown"
    # The dangerous combo the derivation must firewall really is in the manifest.
    assert seen_excluded_ranked


def test_derive_strength_high_with_unmappable_source_raises() -> None:
    # A malformed manifest entry: strength=high but source=none is contradictory
    # (a high video must carry a rank/tournament source). Fail loud, don't fabricate.
    with pytest.raises(ValueError, match="unmappable source"):
        derive_opponent_strength({"strength": "high", "source": "none"})


def test_validate_enforces_rejection_truth_table() -> None:
    # rejection_reason set ⟹ passed_crosscheck must be False.
    payload = _sample_record().to_dict()
    payload["rejection_reason"] = "green_tile_subtraction_failed"
    payload["passed_crosscheck"] = True
    with pytest.raises(ValueError, match="rejection_reason set but passed_crosscheck"):
        GameRecord.from_dict(payload)


def test_validate_enforces_rejection_truth_table_converse() -> None:
    # Symmetric partner: passed_crosscheck=False ⟹ rejection_reason must be set.
    # A reasonless rejection corrupts the §5.6 per-archetype acceptance-rate audit.
    payload = _sample_record().to_dict()
    payload["passed_crosscheck"] = False
    payload["rejection_reason"] = None
    with pytest.raises(ValueError, match="passed_crosscheck=False but rejection_reason is None"):
        GameRecord.from_dict(payload)


def test_validate_allows_rejected_record_for_bias_audit() -> None:
    # A rejected record still emits features + reason (brief §5.6) — must load.
    payload = _sample_record().to_dict()
    payload["rejection_reason"] = "green_tile_subtraction_failed"
    payload["passed_crosscheck"] = False
    rec = GameRecord.from_dict(payload)
    assert rec.rejection_reason == "green_tile_subtraction_failed"
    assert rec.passed_crosscheck is False


# --- provenance orientation-binding: the cross-orientation firewall (FIX 2) -


def test_validate_rejects_welded_desert17_desert11_record() -> None:
    """THE board-orientation bug: a correct desert=11 board welded with openings
    snapped under the REJECTED desert=17 orientation. A D6 flip preserves the
    resource/number multisets, so every other gate passes — only the
    provenance-binding catches it."""
    payload = _sample_record().to_dict()
    payload["provenance"]["board_desert_hex"] = 11
    payload["provenance"]["openings_desert_hex"] = 17
    with pytest.raises(ValueError, match="orientation mismatch"):
        GameRecord.from_dict(payload)


def test_validate_requires_board_desert_provenance() -> None:
    payload = _sample_record().to_dict()
    del payload["provenance"]["board_desert_hex"]
    with pytest.raises(ValueError, match="provenance must carry"):
        GameRecord.from_dict(payload)


def test_validate_requires_openings_desert_provenance() -> None:
    payload = _sample_record().to_dict()
    del payload["provenance"]["openings_desert_hex"]
    with pytest.raises(ValueError, match="provenance must carry"):
        GameRecord.from_dict(payload)


def test_validate_rejects_desert_provenance_out_of_range() -> None:
    payload = _sample_record().to_dict()
    payload["provenance"]["board_desert_hex"] = 19  # 19 == NUM_HEXES, out of 0..18
    payload["provenance"]["openings_desert_hex"] = 19
    with pytest.raises(ValueError, match="out of 0"):
        GameRecord.from_dict(payload)


def test_validate_rejects_float_desert_provenance() -> None:
    payload = _sample_record().to_dict()
    payload["provenance"]["board_desert_hex"] = 11.0  # must be a true int
    with pytest.raises(ValueError, match="board_desert_hex"):
        GameRecord.from_dict(payload)


# --- provenance resolution: sub-1080p is confidently-wrong (SHOULD-FIX) ------


def test_validate_rejects_sub_1080p_provenance() -> None:
    # A record sourced below 1080p OCRs number tokens + log glyphs to garbage —
    # confidently wrong, not noisy. The contract firewall must reject it even
    # though the scale-up gate (a bypassable batch-path function) also would.
    payload = _sample_record().to_dict()
    payload["provenance"]["resolution"] = 360
    with pytest.raises(ValueError, match="resolution 360"):
        GameRecord.from_dict(payload)


def test_validate_rejects_720p_provenance() -> None:
    # 720p is still below the 1080 minimum — must reject.
    payload = _sample_record().to_dict()
    payload["provenance"]["resolution"] = 720
    with pytest.raises(ValueError, match="< required 1080"):
        GameRecord.from_dict(payload)


def test_validate_allows_absent_resolution_provenance() -> None:
    # Optional-when-absent: a record with no resolution stamp still loads (present-
    # and-too-low is the confidently-wrong case; absent is not asserted-against).
    payload = _sample_record().to_dict()
    del payload["provenance"]["resolution"]
    assert GameRecord.from_dict(payload).provenance.get("resolution") is None


def test_record_min_resolution_mirrors_orientation() -> None:
    # record.py mirrors orientation.MIN_RESOLUTION locally (scope-lock: the pure
    # value contract stays orientation-import-free). Pin the two in sync.
    from catan_rl.human_data import MIN_RESOLUTION as ORIENTATION_MIN
    from catan_rl.human_data.record import MIN_RESOLUTION as RECORD_MIN

    assert RECORD_MIN == ORIENTATION_MIN == 1080


# --- road-incidence: snap-sanity ONLY, NOT an orientation check (FIX 3) -----


def test_road_incidence_clean_for_the_resnapped_record() -> None:
    # The re-snapped game-1 record: every road touches an owner settlement.
    topo = load_topology()
    offenders = check_road_incidence(_sample_record(), topo.edge_vertices)
    assert offenders == {"ThePhantom": [], "rayman147": []}


def test_road_incidence_catches_an_isolated_snap_error() -> None:
    # An isolated bad snap (a road nowhere near its settlement) IS caught — this
    # is all the gate is for.
    topo = load_topology()
    payload = _sample_record().to_dict()
    payload["openings"]["ThePhantom"]["roads"] = [0, 50]  # 50 touches neither v1 nor v19
    rec = GameRecord.from_dict(payload)
    offenders = check_road_incidence(rec, topo.edge_vertices)
    assert offenders["ThePhantom"] == [50]


def test_road_incidence_is_NOT_an_orientation_check() -> None:
    """The load-bearing negative: road-incidence is D6-invariant, so all 4
    wrong-orientation (desert=17) game-1 roads PASS it. This is why it is NOT the
    cross-orientation firewall — the provenance-binding (FIX 2) is. We assert the
    wrong-orientation openings are clean under road-incidence, so nobody mistakes
    this gate for the orientation defense."""
    topo = load_topology()
    # The REJECTED desert=17 IDs — physically wrong, but self-consistent under the
    # flipped lattice, so road↔settlement incidence holds for all 4 roads.
    wrong_openings = {
        "ThePhantom": PlayerOpening(settlements=(4, 10), roads=(7, 20)),
        "rayman147": PlayerOpening(settlements=(20, 0), roads=(34, 2)),
    }
    ev = topo.edge_vertices
    for name, opening in wrong_openings.items():
        sset = set(opening.settlements)
        for edge_id in opening.roads:
            a, b = ev[edge_id]
            assert a in sset or b in sset, (
                f"{name} wrong-orientation road e{edge_id} unexpectedly fails incidence"
            )


# --- standard-board multiset gate (findings #1, #5) -------------------------


def test_validate_rejects_wrong_resource_multiset() -> None:
    # An all-ORE board is structurally valid per-hex (every literal legal, ids
    # 0..18) but is NOT the standard 19-tile board — the CV "confidently wrong"
    # failure mode. The multiset gate must reject it.
    hexes = [{"hex_id": i, "resource": "ORE", "number": 8} for i in range(18)]
    hexes.append({"hex_id": 18, "resource": "DESERT", "number": None})
    payload = _sample_record().to_dict()
    payload["board"]["hexes"] = hexes
    with pytest.raises(ValueError, match="resource"):
        GameRecord.from_dict(payload)


def test_validate_rejects_wrong_number_token_bag() -> None:
    # Keep the standard resource multiset but corrupt the number-token bag
    # (swap a 6 to a 10 → three 10s, one 6): not the standard board.
    payload = _sample_record().to_dict()
    for h in payload["board"]["hexes"]:
        if h["resource"] != "DESERT" and h["number"] == 6:
            h["number"] = 10
            break
    with pytest.raises(ValueError, match="number-token"):
        GameRecord.from_dict(payload)


# --- players dict: exactly two distinct seats (finding #3) ------------------


def test_validate_rejects_players_collision() -> None:
    payload = _sample_record().to_dict()
    payload["players"] = {"agent": "ThePhantom", "opponent": "ThePhantom"}
    with pytest.raises(ValueError, match="distinct"):
        GameRecord.from_dict(payload)


def test_validate_rejects_players_wrong_keys() -> None:
    payload = _sample_record().to_dict()
    payload["players"] = {"p1": "ThePhantom", "p2": "rayman147"}
    with pytest.raises(ValueError, match="players"):
        GameRecord.from_dict(payload)


def test_validate_rejects_empty_handle() -> None:
    payload = _sample_record().to_dict()
    payload["players"] = {"agent": "ThePhantom", "opponent": ""}
    with pytest.raises(ValueError, match="handle"):
        GameRecord.from_dict(payload)


# --- draft_order: snake-draft of length 4 (finding #2) ----------------------


def test_validate_rejects_short_draft_order() -> None:
    payload = _sample_record().to_dict()
    payload["draft_order"] = ["ThePhantom", "rayman147"]
    with pytest.raises(ValueError, match="draft_order"):
        GameRecord.from_dict(payload)


def test_validate_rejects_one_player_draft_order() -> None:
    payload = _sample_record().to_dict()
    payload["draft_order"] = ["ThePhantom"] * 4
    with pytest.raises(ValueError, match="draft_order"):
        GameRecord.from_dict(payload)


def test_validate_rejects_non_snake_draft_order() -> None:
    # [a, a, b, b] is length-4 with each handle twice but not a snake [a,b,b,a].
    payload = _sample_record().to_dict()
    payload["draft_order"] = ["rayman147", "rayman147", "ThePhantom", "ThePhantom"]
    with pytest.raises(ValueError, match="snake"):
        GameRecord.from_dict(payload)


# --- openings: completeness + distinctness + cross-player (findings #4, #6) -


def test_validate_rejects_missing_player_opening() -> None:
    payload = _sample_record().to_dict()
    del payload["openings"]["rayman147"]
    with pytest.raises(ValueError, match="openings"):
        GameRecord.from_dict(payload)


def test_validate_rejects_duplicate_settlement_vertex() -> None:
    payload = _sample_record().to_dict()
    payload["openings"]["ThePhantom"]["settlements"] = [1, 1]  # double-snap
    with pytest.raises(ValueError, match="distinct"):
        GameRecord.from_dict(payload)


def test_validate_rejects_duplicate_road_edge() -> None:
    payload = _sample_record().to_dict()
    payload["openings"]["ThePhantom"]["roads"] = [0, 0]
    with pytest.raises(ValueError, match="distinct"):
        GameRecord.from_dict(payload)


def test_validate_rejects_shared_settlement_across_players() -> None:
    payload = _sample_record().to_dict()
    payload["openings"]["rayman147"]["settlements"] = [1, 9]  # 1 is ThePhantom's
    with pytest.raises(ValueError, match="disjoint"):
        GameRecord.from_dict(payload)


# --- float hole (BLOCKER): coerce-and-check must reject non-integers --------


def test_validate_rejects_float_hex_number() -> None:
    payload = _sample_record().to_dict()
    payload["board"]["hexes"][0]["number"] = 8.5  # int(8.5)==8 used to pass
    with pytest.raises(ValueError, match="number"):
        GameRecord.from_dict(payload)


def test_validate_rejects_float_hex_id() -> None:
    payload = _sample_record().to_dict()
    payload["board"]["hexes"][0]["hex_id"] = 0.0  # must be a true int
    with pytest.raises(ValueError, match="hex_id"):
        GameRecord.from_dict(payload)


def test_validate_rejects_float_settlement_vertex() -> None:
    payload = _sample_record().to_dict()
    payload["openings"]["ThePhantom"]["settlements"] = [1.7, 19]  # int(1.7)==1
    with pytest.raises(ValueError, match="vertex"):
        GameRecord.from_dict(payload)


def test_validate_rejects_float_road_edge() -> None:
    payload = _sample_record().to_dict()
    payload["openings"]["ThePhantom"]["roads"] = [0.2, 35]
    with pytest.raises(ValueError, match="edge"):
        GameRecord.from_dict(payload)


def test_validate_rejects_bool_hex_number() -> None:
    # bool is an int subclass — True must not masquerade as the token 1/0.
    payload = _sample_record().to_dict()
    payload["board"]["hexes"][0]["number"] = True
    with pytest.raises(ValueError, match="number"):
        GameRecord.from_dict(payload)


def test_emitted_jsonl_has_no_float_ids() -> None:
    # End-to-end: a float that slipped through used to survive into the JSONL
    # row. Now construction rejects it, so a valid record's row is all-int.
    line = _sample_record().to_json_line()
    row = json.loads(line)
    for h in row["board"]["hexes"]:
        assert isinstance(h["hex_id"], int) and not isinstance(h["hex_id"], bool)
        assert h["number"] is None or (
            isinstance(h["number"], int) and not isinstance(h["number"], bool)
        )
    for op in row["openings"].values():
        for v in op["settlements"] + op["roads"]:
            assert isinstance(v, int) and not isinstance(v, bool)


# --- dice_log: the load-bearing dice-luck covariate (SHOULD-FIX) ------------


def test_validate_rejects_float_dice_value() -> None:
    # int(6.0)==6 used to silently coerce a float roll; reject like a hex token.
    payload = _sample_record().to_dict()
    payload["dice_log"] = [8, 6.0, 11, 4]
    with pytest.raises(ValueError, match="dice_log"):
        GameRecord.from_dict(payload)


def test_validate_rejects_truncating_float_dice_value() -> None:
    # int(6.9)==6 would silently truncate a misread roll into a valid one.
    payload = _sample_record().to_dict()
    payload["dice_log"] = [8, 6.9, 11, 4]
    with pytest.raises(ValueError, match="dice_log"):
        GameRecord.from_dict(payload)


def test_validate_rejects_bool_dice_value() -> None:
    # bool is an int subclass — True must not masquerade as the roll 1.
    payload = _sample_record().to_dict()
    payload["dice_log"] = [8, True, 11, 4]
    with pytest.raises(ValueError, match="dice_log"):
        GameRecord.from_dict(payload)


def test_validate_rejects_out_of_range_dice_value() -> None:
    for bad in (1, 13, 0, 99, -3):
        payload = _sample_record().to_dict()
        payload["dice_log"] = [8, bad, 11, 4]
        with pytest.raises(ValueError, match="dice_log"):
            GameRecord.from_dict(payload)


def test_validate_allows_seven_in_dice_log() -> None:
    # Unlike a hex token, 7 IS a legal roll outcome (it triggers the robber).
    payload = _sample_record().to_dict()
    payload["dice_log"] = [8, 7, 11, 4]
    assert GameRecord.from_dict(payload).dice_log == (8, 7, 11, 4)


def test_validate_rejects_empty_dice_log_on_finished_game() -> None:
    # A completed game (winner set) must carry its rolls.
    payload = _sample_record().to_dict()
    payload["dice_log"] = []
    with pytest.raises(ValueError, match="dice_log is empty"):
        GameRecord.from_dict(payload)


def test_validate_allows_empty_dice_log_on_cutoff_game() -> None:
    # A resign / cutoff game (winner=None) may have a zero-length log.
    payload = _sample_record().to_dict()
    payload["dice_log"] = []
    payload["winner"] = None
    assert GameRecord.from_dict(payload).dice_log == ()


# --- free anchor: board provenance must match the board's own desert --------


def test_validate_rejects_provenance_desert_disagreeing_with_board() -> None:
    """Probe B: provenance board/openings desert both agree (=5) but the board's
    actual unique DESERT is hex 11 — an internally-contradictory record that the
    two-stamp comparison alone would accept. The free anchor rejects it."""
    payload = _sample_record().to_dict()
    payload["provenance"]["board_desert_hex"] = 5
    payload["provenance"]["openings_desert_hex"] = 5
    with pytest.raises(ValueError, match="disagrees with the board's own desert"):
        GameRecord.from_dict(payload)


# --- scoreboard-eligibility predicate exported as code (SHOULD-FIX) ---------


def test_game1_winner_is_rayman147_pov_lost() -> None:
    # Regression for the inverted-winner BLOCKER: game 1 was won by rayman147, so
    # the record's winner must be rayman147 (the ThePhantom POV LOST). It is still
    # a valid, scoreboard-eligible game (a correctly-labelled human loss) — the bug
    # was the label, not eligibility. If this ever reads "ThePhantom" again, a
    # later game's victory line has been welded onto game 1 and the calibration
    # signal for this opening is inverted.
    rec = _sample_record()
    assert rec.winner == "rayman147"
    assert rec.winner != "ThePhantom"
    assert rec.is_scoreboard_eligible() is True


def test_is_scoreboard_eligible_true_for_sample() -> None:
    assert _sample_record().is_scoreboard_eligible() is True


def test_is_scoreboard_eligible_false_when_tier_not_high() -> None:
    payload = _sample_record().to_dict()
    payload["opponent_strength"]["tier"] = "unknown"
    assert GameRecord.from_dict(payload).is_scoreboard_eligible() is False


def test_is_scoreboard_eligible_false_for_known_window_high() -> None:
    # The known_window placeholder is unfalsifiable (no committed backing window;
    # the manifest emits zero known_window entries). Even with tier="high", a
    # known_window record must NOT feed the scoreboard's §5.5 mixed-strength
    # filter — is_scoreboard_eligible() gates on source, not tier alone.
    payload = _sample_record().to_dict()
    payload["opponent_strength"] = {"tier": "high", "source": "known_window", "confidence": 0.8}
    rec = GameRecord.from_dict(payload)
    assert rec.opponent_strength.source == "known_window"
    assert rec.opponent_strength.tier == "high"
    assert rec.is_scoreboard_eligible() is False


def test_is_scoreboard_eligible_true_for_rank_badge_high() -> None:
    # The two manifest-backed high sources (rank_badge, tournament) ARE eligible.
    payload = _sample_record().to_dict()
    payload["opponent_strength"] = {"tier": "high", "source": "rank_badge", "confidence": 0.9}
    assert GameRecord.from_dict(payload).is_scoreboard_eligible() is True


def test_is_scoreboard_eligible_false_when_winner_null() -> None:
    payload = _sample_record().to_dict()
    payload["winner"] = None
    assert GameRecord.from_dict(payload).is_scoreboard_eligible() is False


def test_is_scoreboard_eligible_false_when_rejected() -> None:
    payload = _sample_record().to_dict()
    payload["rejection_reason"] = "green_tile_subtraction_failed"
    payload["passed_crosscheck"] = False
    rec = GameRecord.from_dict(payload)
    assert rec.is_scoreboard_eligible() is False
    assert rec.is_seed_eligible() is False


def test_is_seed_eligible_tracks_passed_crosscheck() -> None:
    assert _sample_record().is_seed_eligible() is True


def test_resolve_ffmpeg_returns_a_usable_binary() -> None:
    """Resolves to a system or imageio-ffmpeg binary (both available in CI)."""
    path = resolve_ffmpeg()
    assert Path(path).exists()


def test_resolve_ffmpeg_fail_fast(monkeypatch: pytest.MonkeyPatch) -> None:
    """With no system ffmpeg and no imageio-ffmpeg, raise with install guidance."""
    import builtins
    from collections.abc import Mapping, Sequence

    monkeypatch.setattr("catan_rl.human_data.ffmpeg.shutil.which", lambda _: None)
    real_import = builtins.__import__

    def _no_imageio(
        name: str,
        globals: Mapping[str, object] | None = None,
        locals: Mapping[str, object] | None = None,
        fromlist: Sequence[str] = (),
        level: int = 0,
    ) -> object:
        if name == "imageio_ffmpeg":
            raise ImportError("simulated absence")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _no_imageio)
    with pytest.raises(FFmpegNotFoundError, match="ffmpeg not found"):
        resolve_ffmpeg()
