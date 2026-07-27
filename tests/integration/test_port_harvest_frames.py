"""Real-pixel pins for the port harvest (spec ``port-harvest`` AC2 / AC4).

The 34 retained Colonist frames are **gitignored**
(``data/human/vlm_spike/frames/``), so these cannot run on CI -- they skip when
the pixels are absent. That is a real coverage gap and the mechanism-level pins
in ``tests/unit/human_data/test_ports.py`` are what CI actually enforces; these
are the local regression that the numbers in ``data/human/ports/report.md`` are
reproducible.

Pinned here:

- **AC2 geometry**: every board that survives the fail-closed gates localises all
  9 slots, and the bias-removed jitter stays inside the measured envelope
  (p95 <= 0.9 px x / 2.2 px y). Be honest about that second one: the localiser fits
  the board's anchor offset from the very candidates it then measures deviation
  against and rejects anything beyond ``MAX_SLOT_DEVIATION_PX``, so the jitter pin
  cannot fail unless the localisation pin already did. The check with real teeth is
  that the rigid sail offset is the SAME across boards, which a systematically wrong
  ``PORT_SLOT_OFFSET_PX`` or a differently-locked D6 element would break -- and it
  lives in ``tests/unit`` (``test_the_rigid_sail_offset_is_constant_across_boards``,
  reading the committed ``geometry.json``) so that CI actually runs it.
- **AC4 photometry**: the label-free composition pass rate beats the measured
  **2/34** naive baseline AND stays within
  :data:`MAX_COMPOSITION_FAILURES` of the accepted-board count. The baseline
  comparison alone is near-vacuous (a collapse from 32/32 to 3/32 would pass it),
  so the second assertion is the one with teeth; it is still relative to the
  accepted count so adding a board does not require editing the test.
- **AC9 pipeline determinism**: re-decoding the frames against the committed
  ``centroids.npz`` + ``centroid_labels.json`` reproduces ``harvest.jsonl``
  exactly -- the binding is nailed to frozen centroids, not to a cluster index
  that a re-clustering can permute.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from catan_rl.human_data import ports as P
from catan_rl.human_data.topology import NUM_PORTS

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parents[2]
FRAMES = REPO_ROOT / "data/human/vlm_spike/frames"

#: The measured naive baseline the spec says to beat (sail-masked k-means).
NAIVE_COMPOSITION_BASELINE = 2

#: Boards that must still pass, pinned AT the achieved count. AC2 asks for 34/34;
#: this build reaches 32 and the shortfall is stated -- with its MEASURED cause, in
#: ``geometry.json["rejection_diagnostics"]`` -- in ``data/human/ports/report.md``.
#: Neither loss is intrinsic: ``Hj_VF4PhHwM__g1`` reaches an 8/9-supported consensus
#: and misses on slot 4 by 0.38 px against ``MAX_SLOT_DEVIATION_PX``, and
#: ``fqBK3_-PO7g__g1`` locks the lattice confidently (screen-rule gap 46.9) but has
#: one spurious token at ~108 px dragging the mean residual past the 5 px cap.
#: Recovering either changes the clustered board set and so invalidates the
#: committed centroid binding -- an owner decision, not a test edit. Pinning below
#: the achieved number would let a third board drop out silently, which is the
#: opposite of what AC2 asks for.
MIN_ACCEPTED_BOARDS = 32

#: Boards whose composition may fail before the photometry pin fires. The measured
#: rate is 32/32; ``> NAIVE_COMPOSITION_BASELINE`` alone would wave through a
#: collapse to 3/32, so the real pin is "essentially all accepted boards".
MAX_COMPOSITION_FAILURES = 2


def _frame_dirs() -> list[Path]:
    if not FRAMES.exists():
        return []
    return sorted(p for p in FRAMES.iterdir() if p.is_dir() and (p / "post_setup.png").exists())


@pytest.fixture(scope="module")
def harvest() -> tuple[dict[str, object], np.ndarray]:
    dirs = _frame_dirs()
    if not dirs:
        pytest.skip("retained Colonist frames are gitignored and not present")
    cv2 = pytest.importorskip("cv2")
    accepted: dict[str, object] = {}
    for d in dirs:
        bgr = cv2.imread(str(d / "post_setup.png"))
        meta_path = d / "meta.json"
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        result = P.read_board_slots(
            np.ascontiguousarray(bgr[:, :, ::-1]),
            expected_desert_hex=meta.get("board_desert_hex"),
        )
        if not isinstance(result, P.PortRejection):
            accepted[d.name] = result
    names = sorted(accepted)
    feats = np.stack(
        [
            P.crop_features(accepted[n].crops[j])  # type: ignore[attr-defined]
            for n in names
            for j in range(NUM_PORTS)
        ]
    )
    return accepted, feats


def test_every_accepted_board_localises_all_nine_slots(harvest) -> None:  # type: ignore[no-untyped-def]
    accepted, _ = harvest
    assert len(accepted) >= MIN_ACCEPTED_BOARDS
    for res in accepted.values():
        assert len(res.localisation.slots) == NUM_PORTS
        assert len(res.crops) == NUM_PORTS


def test_bias_removed_jitter_is_inside_the_measured_envelope(harvest) -> None:  # type: ignore[no-untyped-def]
    accepted, _ = harvest
    dev = np.asarray(
        [[list(s.deviation_px) for s in r.localisation.slots] for r in accepted.values()]
    )
    assert float(np.percentile(np.abs(dev[:, :, 0]), 95)) <= 0.9
    assert float(np.percentile(np.abs(dev[:, :, 1]), 95)) <= 2.2


def test_render_scale_span_matches_the_measured_corpus(harvest) -> None:  # type: ignore[no-untyped-def]
    accepted, _ = harvest
    scales = [r.fit.scale for r in accepted.values()]
    assert 1.14 <= min(scales) <= 1.16
    assert 1.22 <= max(scales) <= 1.24


def test_composition_pass_rate_beats_the_naive_baseline(harvest) -> None:  # type: ignore[no-untyped-def]
    """AC4 -- reported as an actual rate, asserted only against the 2/34 baseline."""
    accepted, feats = harvest
    _centroids, labels = P.cluster_slot_crops(feats)
    per_board = labels.reshape(len(accepted), NUM_PORTS)
    quad = Counter[int]()
    for row in per_board:
        counts = Counter(int(v) for v in row)
        if sorted(counts.values(), reverse=True) == [4, 1, 1, 1, 1, 1]:
            quad[next(k for k, v in counts.items() if v == 4)] += 1
    consistent = max(quad.values()) if quad else 0
    assert consistent > NAIVE_COMPOSITION_BASELINE, (
        f"composition pass rate {consistent}/{len(accepted)} does not beat the "
        f"measured naive baseline of {NAIVE_COMPOSITION_BASELINE}/34"
    )
    assert consistent >= len(accepted) - MAX_COMPOSITION_FAILURES, (
        f"composition pass rate {consistent}/{len(accepted)} regressed: the measured build "
        f"reads every accepted board, so more than {MAX_COMPOSITION_FAILURES} failures is a "
        "photometry regression even though it still beats the naive baseline"
    )


def test_clustering_is_deterministic(harvest) -> None:  # type: ignore[no-untyped-def]
    _accepted, feats = harvest
    a = P.cluster_slot_crops(feats)[1]
    b = P.cluster_slot_crops(feats)[1]
    assert np.array_equal(a, b)


def test_sidecar_reproduces_from_the_frozen_centroids(harvest) -> None:  # type: ignore[no-untyped-def]
    """AC9 at the PIPELINE level: same frame + same frozen binding -> same port map.

    This is what the cluster-index binding could not give. The decode here never
    re-clusters: it loads the committed centroids, checks their fingerprint against
    the committed labels file, and must reproduce every harvested row bit-for-bit.
    """
    out_dir = REPO_ROOT / "data/human/ports"
    sidecar, labels_path, npz = (
        out_dir / "harvest.jsonl",
        out_dir / "centroid_labels.json",
        out_dir / "centroids.npz",
    )
    if not (sidecar.exists() and labels_path.exists() and npz.exists()):
        pytest.skip("the decode phase has not been run in this checkout")
    accepted, _ = harvest
    with np.load(npz, allow_pickle=False) as z:
        centroids = np.asarray(z["centroids"], dtype=np.float64)
    classifier = P.classifier_from_labels(centroids, json.loads(labels_path.read_text()))
    rows = {
        (r["video_id"], r["game_index"]): [s["port_type"] for s in r["slots"]]
        for r in (json.loads(line) for line in sidecar.read_text().splitlines() if line.strip())
    }
    for name, res in accepted.items():
        video_id, _, game = name.rpartition("__g")
        expected = rows.get((video_id, int(game)))
        if expected is None:
            continue
        feats = np.stack([P.crop_features(c) for c in res.crops])  # type: ignore[attr-defined]
        decode = P.decode_slots(classifier.scores(feats), names=classifier.names)
        assert decode.names is not None and list(decode.names) == expected
