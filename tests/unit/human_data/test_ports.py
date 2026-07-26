"""Port-harvest tests (spec ``port-harvest``, acceptance criteria 2/3/5/6/7/9).

Everything here is CI-safe: the retained Colonist frames are gitignored, so the
real-pixel pins live in ``tests/integration/test_port_harvest_frames.py`` (which
skips when the frames are absent). What is pinned here is the *mechanism*:

- **Geometry** -- the D1 reflection is affine-equivariant and the constant offset
  is applied exactly once (AC2's math, independent of any frame).
- **HUD** -- a synthetic ocean frame with a white HUD rectangle parked beside two
  slots: the tight window ignores it, a loose window locks onto it (AC3).
- **Dual decode** -- a single-slot swap is caught by D3 and NOT by the multiset
  check; a global transposition defeats D3 and is asserted to do so (AC5).
- **Fail-closed** -- no 9-slot map can be built from 8 read slots (AC6/D4).
- **Determinism** -- same input, same clusters and same decode (AC9).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from catan_rl.human_data import ports as P
from catan_rl.human_data.board_cv import load_engine_template
from catan_rl.human_data.topology import NUM_PORTS, load_topology

cv2 = pytest.importorskip("cv2", reason="port geometry needs opencv")

OCEAN = (20, 70, 140)
EDGE = 90.0


# ------------------------------------------------------------------ D1 geometry


def test_slot_centres_are_the_reflection_plus_one_constant_offset() -> None:
    """``2*mid - hex_centre`` in ENGINE space, then exactly one frame-space offset."""
    template, topology = load_engine_template(), load_topology()
    affine = np.array([[1.15, 0.0, 300.0], [0.0, 1.15, 120.0]])
    centres = P.predicted_slot_centres(affine, template, topology)
    assert centres.shape == (NUM_PORTS, 2)
    for i, slot in enumerate(topology.port_slots):
        hexc = template.hex_centers[int(slot["hex"])]
        v0, v1 = (int(v) for v in slot["vertices"])
        mid = 0.5 * (template.vertex_px[v0] + template.vertex_px[v1])
        engine_pt = 2.0 * mid - hexc
        expected = affine[:, :2] @ engine_pt + affine[:, 2] + np.asarray(P.PORT_SLOT_OFFSET_PX)
        assert np.allclose(centres[i], expected)


def test_slot_centres_are_affine_equivariant() -> None:
    """A pure rescale of the affine rescales the slot centres, offset aside.

    The reflection is linear in the projected points, so nothing about it may
    depend on where the board happens to sit in the frame.
    """
    template, topology = load_engine_template(), load_topology()
    a1 = np.array([[1.2, 0.0, 0.0], [0.0, 1.2, 0.0]])
    a2 = np.array([[1.2, 0.0, 400.0], [0.0, 1.2, -60.0]])
    c1 = P.predicted_slot_centres(a1, template, topology)
    c2 = P.predicted_slot_centres(a2, template, topology)
    assert np.allclose(c2 - c1, np.array([400.0, -60.0]))


def test_affine_scale_reads_the_isotropic_render_scale() -> None:
    assert P.affine_scale(np.array([[1.25, 0.0, 5.0], [0.0, 1.25, 7.0]])) == pytest.approx(1.25)


# ----------------------------------------------------------------- AC3 HUD pin


def _synthetic_ocean(width: int = 1400, height: int = 900) -> np.ndarray:
    frame = np.zeros((height, width, 3), np.uint8)
    frame[:, :] = OCEAN
    return frame


def _paint_sail(frame: np.ndarray, cx: float, cy: float, radius: int = 8) -> None:
    cv2.circle(frame, (int(cx), int(cy)), radius, (250, 250, 250), -1)


def _synthetic_board(
    hud_slots: tuple[int, ...] = (5, 6), hud_offset_frac: float = 0.6
) -> tuple[np.ndarray, np.ndarray]:
    """9 sprite-sized sails on a 3x3 grid, plus a big white HUD panel by some slots.

    ``hud_offset_frac`` is the panel's distance from the slot centre in units of
    the edge length. At the default 0.6 the panel is outside a +-0.35 window and
    inside a +-0.95 one, which is the arithmetic the window pins below assert. At
    0.2 it is INSIDE the tight window too -- the case the real-pixel ablation in
    ``data/human/ports/report.md`` measured, where the tight window does not save
    the slot and the sprite-area cap and cross-slot consensus do.
    """
    frame = _synthetic_ocean()
    centres = np.array(
        [[200.0 + 320.0 * (i % 3), 150.0 + 300.0 * (i // 3)] for i in range(NUM_PORTS)]
    )
    for cx, cy in centres:
        _paint_sail(frame, cx, cy)
    for slot in hud_slots:
        cx, cy = centres[slot]
        x0, y0 = int(cx + hud_offset_frac * EDGE), int(cy - 60)
        cv2.rectangle(frame, (x0, y0), (x0 + 110, y0 + 120), (255, 255, 255), -1)
    return frame, centres


def test_tight_window_ignores_the_hud_panel() -> None:
    frame, centres = _synthetic_board()
    loc = P.localise_board_slots(
        frame,
        centres,
        edge_px=EDGE,
        window_frac=P.SEARCH_WINDOW_FRAC,
        area_range=(0.0, 1.0),
        naive=True,
    )
    assert not isinstance(loc, P.PortRejection)
    for s in loc.slots:
        assert float(np.hypot(*s.deviation_px)) < 1.0


def test_loose_window_locks_onto_the_hud_panel() -> None:
    """The failure mode D1's tight window exists to avoid, reproduced deterministically."""
    frame, centres = _synthetic_board()
    loc = P.localise_board_slots(
        frame,
        centres,
        edge_px=EDGE,
        window_frac=0.95,
        area_range=(0.0, 1.0),
        naive=True,
    )
    assert not isinstance(loc, P.PortRejection)
    locked = {
        s.slot for s in loc.slots if float(np.hypot(*s.deviation_px)) > P.MAX_SLOT_DEVIATION_PX
    }
    assert locked == {5, 6}


def test_tight_window_alone_does_not_stop_a_hud_panel_inside_it() -> None:
    """The measured reality, pinned: the window is not the defence D1 assumed.

    On the real frames the tight window still leaves 56/297 naive slots locked onto
    HUD panels. Reproduced here by parking the panel INSIDE the tight window: the
    window buys nothing, and only the sprite-area cap (next test) does.
    """
    frame, centres = _synthetic_board(hud_offset_frac=0.2)
    loc = P.localise_board_slots(
        frame,
        centres,
        edge_px=EDGE,
        window_frac=P.SEARCH_WINDOW_FRAC,
        area_range=(0.0, 1.0),
        naive=True,
    )
    assert not isinstance(loc, P.PortRejection)
    locked = {
        s.slot for s in loc.slots if float(np.hypot(*s.deviation_px)) > P.MAX_SLOT_DEVIATION_PX
    }
    assert locked == {5, 6}


def test_sprite_area_cap_is_what_removes_the_intruding_hud_lock_on() -> None:
    """Same frame, same tight window, cap on: the sail wins because the panel is too big."""
    frame, centres = _synthetic_board(hud_offset_frac=0.2)
    loc = P.localise_board_slots(
        frame, centres, edge_px=EDGE, window_frac=P.SEARCH_WINDOW_FRAC, naive=True
    )
    assert not isinstance(loc, P.PortRejection)
    for s in loc.slots:
        assert float(np.hypot(*s.deviation_px)) < 1.0
        assert P.SPRITE_AREA_FRAC_RANGE[0] <= s.area_frac <= P.SPRITE_AREA_FRAC_RANGE[1]


def test_consensus_localiser_rejects_the_board_when_one_slot_is_displaced() -> None:
    """AC6 / D4: a single bad slot rejects the WHOLE board, never 8-of-9."""
    frame, centres = _synthetic_board(hud_slots=())
    shifted = centres.copy()
    shifted[3] += np.array([18.0, 0.0])  # sail no longer where the geometry says
    assert P.localise_board_slots(frame, shifted, edge_px=EDGE) == P.PortRejection.UNLOCALISED_SLOT


def test_blank_slot_rejects_the_board() -> None:
    frame = _synthetic_ocean()
    centres = np.array([[200.0 + 320.0 * (i % 3), 150.0 + 300.0 * (i // 3)] for i in range(9)])
    assert P.localise_board_slots(frame, centres, edge_px=EDGE) == P.PortRejection.BLANK_SLOT


def test_localiser_requires_nine_slots() -> None:
    frame = _synthetic_ocean()
    with pytest.raises(ValueError, match="9, 2"):
        P.localise_board_slots(frame, np.zeros((8, 2)), edge_px=EDGE)


# ------------------------------------------------------------ AC5 dual decode


def _scores_for(names: tuple[str, ...], confusion: dict[str, str] | None = None) -> np.ndarray:
    """A clean one-hot-ish score matrix for ``names``, optionally with a confusion."""
    mapping = confusion or {}
    scores = np.full((NUM_PORTS, len(P.PORT_TYPES)), -1.0)
    for i, name in enumerate(names):
        read = mapping.get(name, name)
        scores[i, P.PORT_TYPES.index(read)] = 0.0
    return scores


TRUTH: tuple[str, ...] = (
    "2:1 BRICK",
    "3:1 PORT",
    "2:1 ORE",
    "3:1 PORT",
    "2:1 SHEEP",
    "3:1 PORT",
    "2:1 WHEAT",
    "3:1 PORT",
    "2:1 WOOD",
)


def test_clean_scores_decode_and_both_legs_agree() -> None:
    decode = P.decode_slots(_scores_for(TRUTH), names=P.PORT_TYPES)
    assert decode.rejection is None
    assert decode.names == TRUTH
    assert P.composition_ok(TRUTH)


def test_single_slot_swap_is_caught_by_dual_decode_not_by_the_multiset() -> None:
    """D3's whole point: an illegal free reading that a multiset check cannot see."""
    # Slot 2's ORE is misread as WHEAT -> two WHEAT, no ORE. The unconstrained
    # argmax says so; Hungarian is forced to emit a legal multiset. They differ.
    scores = _scores_for(TRUTH)
    scores[2, P.PORT_TYPES.index("2:1 ORE")] = -1.0
    scores[2, P.PORT_TYPES.index("2:1 WHEAT")] = 0.0
    decode = P.decode_slots(scores, names=P.PORT_TYPES)
    assert decode.rejection is P.PortRejection.DECODE_DISAGREEMENT
    assert decode.names is None
    # ... and the permutation-invariant check the constrained leg produces is legal,
    # which is exactly why the multiset alone would have waved this through.
    assert P.composition_ok(decode.hungarian_names)


def test_global_transposition_DEFEATS_the_dual_decode() -> None:
    """The named blind spot (D3). This test asserts the failure, it does not fix it.

    Every ORE read as BRICK and every BRICK as ORE yields a legal multiset, so
    both legs agree and both are wrong on every board. Only the hand-labelled
    centroids (D2) stand between the harvest and this.
    """
    confusion = {"2:1 ORE": "2:1 BRICK", "2:1 BRICK": "2:1 ORE"}
    decode = P.decode_slots(_scores_for(TRUTH, confusion), names=P.PORT_TYPES)
    assert decode.rejection is None
    assert decode.names is not None
    assert decode.names != TRUTH
    assert P.composition_ok(decode.names)


def test_low_margin_is_a_typed_rejection() -> None:
    """The floor must fire on an UNCONFIDENT but otherwise legal read.

    An all-zero score matrix does not test this: the unconstrained argmax picks
    class 0 for all 9 slots, which is an illegal multiset, so the board is rejected
    as ``DECODE_DISAGREEMENT`` before the margin is ever consulted. The scores here
    decode to a legal composition on which both legs AGREE, so ``LOW_MARGIN`` is the
    only thing left that can reject it -- and it is asserted exactly, with no
    ``in (...)`` escape hatch. This is the floor's only evidence: it has never fired
    on real pixels (observed min margin 0.051 against a floor of 0.02).
    """
    scores = _scores_for(TRUTH)
    # Lift every slot's runner-up to 0.1 below its winner: still a clean, legal,
    # agreeing decode -- just an unconfident one.
    for i, name in enumerate(TRUTH):
        for c in range(len(P.PORT_TYPES)):
            if c != P.PORT_TYPES.index(name):
                scores[i, c] = -0.1
    assert P.decode_slots(scores, min_margin=0.05, names=P.PORT_TYPES).names == TRUTH

    decode = P.decode_slots(scores, min_margin=0.5, names=P.PORT_TYPES)
    assert decode.rejection is P.PortRejection.LOW_MARGIN
    assert decode.names is None
    assert decode.argmax_names == TRUTH
    assert decode.hungarian_names == TRUTH
    assert max(decode.margins) < 0.5


def test_a_within_board_permutation_defeats_the_dual_decode_too() -> None:
    """``composition_ok``'s docstring used to credit D3 with catching this. It does not.

    Swapping two slots' CONTENTS leaves a legal multiset, so the argmax and the
    Hungarian leg agree on the swapped reading and the multiset check passes. What
    actually rules this out is the measured geometric jitter (D8): each slot is
    cropped at its own lattice point.
    """
    swapped = list(TRUTH)
    swapped[0], swapped[2] = swapped[2], swapped[0]
    decode = P.decode_slots(_scores_for(tuple(swapped)), names=P.PORT_TYPES)
    assert decode.rejection is None
    assert decode.names == tuple(swapped)
    assert P.composition_ok(decode.names)


def test_decode_is_deterministic() -> None:
    scores = _scores_for(TRUTH)
    assert (
        P.decode_slots(scores, names=P.PORT_TYPES).names
        == P.decode_slots(scores.copy(), names=P.PORT_TYPES).names
    )


def test_decode_honours_a_non_identity_binding() -> None:
    """Regression: the D2 hand binding must reach the decoder.

    ``decode_slots`` used to default ``names`` to :data:`PORT_TYPES`, and every
    caller omitted it -- so score column ``c`` was named alphabetically instead of
    by ``PortClassifier.names``, and the shipped harvest was globally transposed
    (ore->brick, sheep->ore, wheat->sheep, brick->wheat). A global transposition is
    a LEGAL multiset, so both decode legs agree and the composition check passes on
    every board; nothing else in the pipeline can catch it. This pins that a
    permuted binding actually changes the output.
    """
    scores = _scores_for(TRUTH)
    identity = P.decode_slots(scores, names=P.PORT_TYPES)
    # Reverse the 2:1 names, leaving the generic column in place.
    permuted_names = (*list(P.PORT_TYPES[:5])[::-1], P.PORT_TYPES[5])
    permuted = P.decode_slots(scores, names=permuted_names)
    assert identity.names is not None and permuted.names is not None
    assert identity.names != permuted.names, (
        "a permuted class->name binding produced identical output -- `names` is "
        "being ignored, which is exactly the bug this test exists to catch"
    )


def test_decode_rejects_a_wrong_row_count() -> None:
    with pytest.raises(ValueError, match="slot rows"):
        P.decode_slots(np.zeros((8, len(P.PORT_TYPES))), names=P.PORT_TYPES)


# ---------------------------------------------------- AC6 fail-closed assembly


def test_build_port_assignment_requires_nine_independently_read_slots() -> None:
    with pytest.raises(P.PortHarvestError, match="banned"):
        P.build_port_assignment(TRUTH[:8])


def test_build_port_assignment_rejects_an_unknown_type() -> None:
    with pytest.raises(P.PortHarvestError, match="unknown port type"):
        P.build_port_assignment(("2:1 GOLD", *TRUTH[1:]))


def test_build_port_assignment_matches_the_engine_slot_vertices() -> None:
    assignment = P.build_port_assignment(TRUTH)
    topology = load_topology()
    assert sum(len(v) for v in assignment.values()) == 2 * NUM_PORTS
    assert len(assignment["3:1 PORT"]) == 8
    for slot, name in zip(topology.port_slots, TRUTH, strict=True):
        for vertex in slot["vertices"]:
            assert int(vertex) in assignment[name]


def test_composition_ok_rejects_an_illegal_multiset() -> None:
    assert not P.composition_ok(("2:1 ORE", *TRUTH[1:]))


# ------------------------------------------------------- AC9 photometry basics


def _blob_features(seed: int, n_per_class: int = 6) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    base = rng.normal(size=(6, 12))
    base /= np.linalg.norm(base, axis=1, keepdims=True)
    truth = np.repeat(np.arange(6), n_per_class)
    feats = base[truth] + 0.01 * rng.normal(size=(len(truth), 12))
    return feats, truth


def test_kmeans_is_deterministic_and_recovers_well_separated_blobs() -> None:
    feats, truth = _blob_features(0)
    c1, l1 = P.kmeans(feats, 6, seed=3)
    c2, l2 = P.kmeans(feats, 6, seed=3)
    assert np.array_equal(l1, l2) and np.allclose(c1, c2)
    # Same blob -> same cluster (label ids themselves are arbitrary).
    for t in range(6):
        assert len(set(l1[truth == t].tolist())) == 1


def test_kmeans_best_is_deterministic() -> None:
    feats, _ = _blob_features(1)
    a = P.kmeans_best(feats, 6, n_init=4)[1]
    b = P.kmeans_best(feats, 6, n_init=4)[1]
    assert np.array_equal(a, b)


def test_cluster_slot_crops_puts_the_generic_class_last() -> None:
    """The 4-of-9 generic port must land on :data:`GENERIC_CLUSTER` by construction."""
    rng = np.random.default_rng(7)
    specific = np.eye(5, 16)
    generic = np.zeros(16)
    generic[15] = 10.0
    truth = [0, 1, 2, 3, 4] + [5] * 4  # one board's 9 slots
    rows = [(specific[t] if t < 5 else generic) for _ in range(4) for t in truth]
    feats = np.asarray(rows) + 0.005 * rng.normal(size=(36, 16))
    _centroids, labels = P.cluster_slot_crops(feats, n_init=4)
    generic_positions = [i for i, t in enumerate(truth * 4) if t == 5]
    assert set(labels[generic_positions].tolist()) == {P.GENERIC_CLUSTER}
    assert int((labels == P.GENERIC_CLUSTER).sum()) == 16


def test_classifier_rejects_a_bad_binding() -> None:
    centroids = np.zeros((6, 4))
    with pytest.raises(ValueError, match="one name is required"):
        P.PortClassifier(centroids=centroids, names=("3:1 PORT",))
    with pytest.raises(ValueError, match="unknown port type"):
        P.PortClassifier(centroids=centroids, names=("nope",) * 6)


def _labels_for(centroids: np.ndarray) -> dict[str, object]:
    return {
        P.LABELS_FINGERPRINT_KEY: P.centroid_fingerprint(centroids),
        P.LABELS_NAMES_KEY: {str(i): name for i, name in enumerate(P.PORT_TYPES)},
    }


def test_cluster_indices_permute_when_the_board_set_changes() -> None:
    """WHY the binding is nailed to a fingerprint and not to a cluster index.

    Clustering the whole set and clustering a subset of the SAME data put the same
    archetype in different cluster indices. A labels file keyed by index alone
    would therefore silently rebind -- the consistent global transposition D3 is
    blind to.
    """
    rng = np.random.default_rng(3)
    archetypes = np.eye(6, 8)
    truth = np.array([0, 1, 2, 3, 4, 5, 5, 5, 5] * 8)
    feats = archetypes[truth] + 0.01 * rng.normal(size=(len(truth), 8))
    full = P.cluster_slot_crops(feats, n_init=4)[1]
    subset_idx = np.arange(len(truth) - 9)  # drop one board
    part = P.cluster_slot_crops(feats[subset_idx], n_init=4)[1]
    map_full = {int(t): int(full[truth == t][0]) for t in range(5)}
    map_part = {int(t): int(part[truth[subset_idx] == t][0]) for t in range(5)}
    assert map_full != map_part


def test_labels_bound_to_other_centroids_are_refused() -> None:
    rng = np.random.default_rng(5)
    centroids = rng.normal(size=(6, 4))
    labels = _labels_for(centroids)
    assert P.classifier_from_labels(centroids, labels).names == P.PORT_TYPES
    # Re-clustered centroids -> a different fingerprint -> hard stop, not a rebind.
    with pytest.raises(P.PortHarvestError, match="DIFFERENT centroid set"):
        P.classifier_from_labels(rng.normal(size=(6, 4)), labels)


def test_labels_file_must_be_complete_and_fingerprinted() -> None:
    centroids = np.random.default_rng(6).normal(size=(6, 4))
    labels = _labels_for(centroids)
    with pytest.raises(P.PortHarvestError, match="no 'centroids_sha256'"):
        P.classifier_from_labels(centroids, {P.LABELS_NAMES_KEY: labels[P.LABELS_NAMES_KEY]})
    partial = dict(labels)
    names = dict(labels[P.LABELS_NAMES_KEY])  # type: ignore[arg-type]
    names.pop("4")
    partial[P.LABELS_NAMES_KEY] = names
    with pytest.raises(P.PortHarvestError, match=r"missing cluster key\(s\) \['4'\]"):
        P.classifier_from_labels(centroids, partial)
    bad = dict(labels)
    bad[P.LABELS_NAMES_KEY] = {**names, "4": "2:1 GOLD"}
    with pytest.raises(P.PortHarvestError, match="unknown port type"):
        P.classifier_from_labels(centroids, bad)


def test_centroid_fingerprint_survives_a_float_round_trip(tmp_path) -> None:  # type: ignore[no-untyped-def]
    centroids = np.random.default_rng(9).normal(size=(6, 7))
    path = tmp_path / "centroids.npz"
    np.savez(path, centroids=centroids)
    with np.load(path) as z:
        assert P.centroid_fingerprint(np.asarray(z["centroids"], float)) == P.centroid_fingerprint(
            centroids
        )


def test_classifier_scores_are_frozen_and_corpus_independent() -> None:
    """AC9: nearest-FROZEN-centroid, so one frame maps to one map either way."""
    rng = np.random.default_rng(11)
    centroids = rng.normal(size=(6, 5))
    clf = P.PortClassifier(centroids=centroids, names=P.PORT_TYPES)
    feats = rng.normal(size=(NUM_PORTS, 5))
    first = clf.scores(feats)
    assert np.allclose(first, clf.scores(feats))
    # Scoring a different, larger batch does not move any individual row.
    bigger = np.concatenate([feats, rng.normal(size=(20, 5))])
    assert np.allclose(clf.scores(bigger)[:NUM_PORTS], first)


# ------------------------------------- committed-artifact pins (CI-safe, real data)


GEOMETRY_JSON = Path(__file__).resolve().parents[3] / "data/human/ports/geometry.json"


def _geometry() -> dict[str, object]:
    if not GEOMETRY_JSON.exists():
        pytest.skip("the harvest has not been run in this checkout")
    return json.loads(GEOMETRY_JSON.read_text())  # type: ignore[no-any-return]


def test_the_rigid_sail_offset_is_constant_across_boards() -> None:
    """The geometry check that actually has teeth, and it runs WITHOUT the pixels.

    ``localise_board_slots`` fits the board's anchor offset from the very candidates
    it then measures deviation against, and rejects any slot beyond
    ``MAX_SLOT_DEVIATION_PX``. So no accepted board can *ever* show a large
    deviation, and the jitter pin in the integration file cannot fail unless the
    "all nine slots localise" pin already failed. What is NOT self-fulfilling is that
    the offset is the SAME on every board: a systematically wrong
    ``PORT_SLOT_OFFSET_PX``, or a board that locked a different D6 element, produces
    internally consistent slots at a displaced offset. That is pinned here, against
    the committed ``geometry.json`` (measured span x 3.119..4.061, y -31.769..-29.779
    over 32 boards).
    """
    geo = _geometry()
    offsets = np.asarray([b["anchor_offset_px"] for b in geo["boards"]], float)  # type: ignore[index,union-attr]
    assert len(offsets) >= 32
    assert offsets[:, 0].min() >= 2.0 and offsets[:, 0].max() <= 5.5
    assert offsets[:, 1].min() >= -33.5 and offsets[:, 1].max() <= -28.0
    # ...and the SPREAD, which is what a displaced board would blow out.
    assert float(np.ptp(offsets[:, 0])) <= 2.0
    assert float(np.ptp(offsets[:, 1])) <= 3.5


def test_every_harvested_board_records_whether_the_desert_check_ran() -> None:
    """A skipped D6 corroboration must be visible, not absent (see report.md).

    A wrong D6 element relabels all 9 slots at once -- a global slot permutation the
    composition check, both decode legs and the jitter envelope all wave through.
    The desert cross-check is the only guard against it, so boards it did not run on
    have to be enumerated rather than silently pooled with the rest.
    """
    geo = _geometry()
    assert "desert_check_skipped" in geo, "regenerate geometry.json with harvest_ports.py"
    for board in geo["boards"]:  # type: ignore[union-attr]
        assert "desert_corroborated" in board
    skipped = {b["board"] for b in geo["boards"] if not b["desert_corroborated"]}  # type: ignore[index,union-attr]
    assert skipped == set(geo["desert_check_skipped"])  # type: ignore[arg-type]
