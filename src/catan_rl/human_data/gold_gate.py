"""Gold-gate tooling — the 30-game blind-labeling EXAM apparatus (step6 §3, build brief §4).

This is the *instruments only* for the human-data pipeline's final gate: the
30-game gold audit that measures the pipeline's field accuracy against a
**human-blind ground truth** before the 204-video harvest is trusted. Labeling
happens in a later phase; this module builds the packets a labeler fills and the
scorer that grades them.

Three deliverables (the CLI ``scripts/gold_gate.py`` is a thin wrapper over these):

- **prepare** (:func:`prepare_packets`) — select ~30 gold games (the accepted
  Tier-5 games plus additional harvested ``high`` games to reach the target) and,
  for each, write a **BLIND-LABELING packet** under ``<gold_dir>/<game_id>/``: the
  post-setup full frame, 1-2 extra mid-setup frames, the setup + terminal log-crop
  PNGs, a static engine-id **reference grid**, and a **blank label template**. The
  packet contains **NOTHING derived from the pipeline's parse of that game** —
  blindness is the point (:func:`assert_packet_blind` proves it). The pipeline's
  own record (the answer key ``score`` grades against) is written **outside** the
  blind packet, under ``<gold_dir>/_answers/``.

- **score** (:func:`score_gold`) — compare completed label files against the
  pipeline records **field-by-field**, report per-field accuracy against the
  PRE-REGISTERED bars (board ≥98% of hexes, openings ≥95% of placements, winner
  ~100%, orientation flips 0) with Wilson CIs and a verdict line, written to
  ``docs/plans/gold_gate_report.md``.

- **reference grid** (:func:`render_reference_grid`) — a canonical image mapping
  engine hex / vertex / edge integer IDs to positions, so a labeler can name IDs
  WITHOUT any pipeline output. Reuses the committed engine template geometry
  (:func:`~catan_rl.human_data.board_cv.load_engine_template` +
  :func:`~catan_rl.human_data.topology.load_topology`); headless matplotlib (Agg),
  **no ``gui/`` import**.

CPU-only; never imports ``gui/`` or the training path.

Blindness model
---------------
A labeler must read the game **only** from the raw frame pixels — never from any
pipeline decision. So the packet directory holds exactly:

* the frame / log-crop PNGs (raw pixels — the ground truth, not pipeline output);
* ``reference_grid.png`` (STATIC engine geometry — identical for every game, so it
  encodes no game-specific answer);
* ``label_template.json`` (all answer fields ``null``);
* ``README.md`` (static instructions + the orientation convention).

The pipeline's parse of the game (:meth:`GameRecord.to_dict`) lives in a sibling
``_answers/`` tree the labeler never opens. :func:`assert_packet_blind` enforces
this: the label template is all-null, the answer key is not inside the packet, and
no text file in the packet embeds the record's board / winner answers.

Orientation convention (stated to the labeler)
----------------------------------------------
The pipeline locks the D6 board orientation with the screen-space rule (engine
hex 8 → top-centre, hex 11 → rightmost). The reference grid is rendered under that
SAME convention, so a labeler overlays it on the on-screen board and reads each
hex's engine ID directly. Openings are labeled by SEAT ROLE — ``"pov"`` (the
bottom self-seat, always ThePhantom) and ``"opponent"`` (the top seat) — so no
parsed handle leaks into the packet, and ``score`` maps roles back to the record's
``agent`` / ``opponent`` seats.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from catan_rl.eval.wilson import WilsonInterval, wilson_interval
from catan_rl.human_data.board_cv import EngineTemplate, load_engine_template
from catan_rl.human_data.record import NUM_HEXES, GameRecord
from catan_rl.human_data.topology import Topology, load_topology

if TYPE_CHECKING:
    import numpy.typing as npt

# --- pre-registered gold bars (step6 §3 / build brief §4) --------------------

#: Board layout bar: fraction of the 19 hexes (resource AND number both correct)
#: the pipeline must match against the human-blind label. A wrong hex is a
#: confidently-wrong board (brief §5.6).
BOARD_BAR = 0.98

#: Openings bar: fraction of the 8 opening placements (2 settlements + 2 roads per
#: player, in LOG PLACEMENT ORDER) the pipeline must match.
OPENINGS_BAR = 0.95

#: Winner bar: the victory-line read is near-perfect (brief §5.1 — the win LOG line
#: OCRs cleanly), so the bar is ~100%.
WINNER_BAR = 0.99

#: Orientation-flip bar: the number of games whose human-blind board is a
#: NON-IDENTITY D6 relabeling of the pipeline's board (the pipeline locked the
#: wrong orientation — brief §5.2) must be EXACTLY zero.
ORIENTATION_FLIPS_BAR = 0

#: Default target gold-game count (the "30-game exam").
DEFAULT_GOLD_COUNT = 30

#: The stated orientation convention the reference grid is rendered under and the
#: labeler names hex IDs by (brief §5.2 screen-space lock).
ORIENTATION_CONVENTION = (
    "Engine orientation lock (screen-space rule): identify engine hex 8 as the "
    "TOP-CENTRE hex and engine hex 11 as the RIGHTMOST hex, then read every other "
    "hex's engine id off reference_grid.png. Do NOT anchor on the desert (it moves "
    "per game). Vertex ids (0..53) and edge ids (0..71) are shown on the same grid."
)

#: The seat roles openings are labeled by (blind: no parsed handle in the packet).
#: ``score`` maps ``"pov"`` → ``record.players["agent"]`` (ThePhantom, bottom
#: self-seat) and ``"opponent"`` → ``record.players["opponent"]`` (top seat).
POV_ROLE = "pov"
OPPONENT_ROLE = "opponent"

#: Winner label values (the blind label never carries a handle).
WINNER_LABELS: tuple[str, str, str] = (POV_ROLE, OPPONENT_ROLE, "none")

#: The complete set of files a blind packet may contain (blindness allowlist).
_LOG_SETUP_PNG = "log_setup.png"
_LOG_TERMINAL_PNG = "log_terminal.png"
_POST_SETUP_PNG = "post_setup.png"
_GRID_PNG = "reference_grid.png"
_LABEL_JSON = "label_template.json"
_README = "README.md"

#: Subdir (a SIBLING of the packets, never inside one) holding the pipeline answer
#: keys ``score`` grades against.
ANSWERS_DIRNAME = "_answers"


def game_id(record: GameRecord) -> str:
    """The stable gold ``game_id`` for a record: ``"<video_id>__g<game_index>"``."""
    return f"{record.video_id}__g{record.game_index}"


# --- frame bundle (the raw pixels the labeler reads) -------------------------


@dataclass(frozen=True, slots=True)
class GoldFrames:
    """The raw frame pixels one gold packet exposes to a labeler.

    All arrays are ``(H, W, 3)`` ``uint8`` RGB (native geometry, brief §5.12). None
    of these is pipeline output — they are the ground truth the labeler reads:

    - ``post_setup`` — the 8-pieces-down full frame (board + openings source).
    - ``mid_setup`` — 1-2 extra mid-setup frames (cross-frame disambiguation).
    - ``setup_log_crop`` — the log panel over the setup region (draft order + the
      "received starting resources" grant lines).
    - ``terminal_log_crop`` — the log panel over the victory/terminal region (the
      ``🏆 <player> won the game! 🏆`` line the winner is read from, brief §5.1).
    """

    post_setup: npt.NDArray[np.uint8]
    mid_setup: tuple[npt.NDArray[np.uint8], ...]
    setup_log_crop: npt.NDArray[np.uint8]
    terminal_log_crop: npt.NDArray[np.uint8]


#: A frame provider: ``record -> GoldFrames``. The real provider re-ingests the
#: video and routes frames to the game (a real-run CV path); tests inject a stub.
FrameProvider = Callable[[GameRecord], GoldFrames]

#: A grid renderer: ``out_path -> written path``. Defaults to
#: :func:`render_reference_grid`; injectable so a test can stub the (matplotlib)
#: render without the heavy dependency.
GridRenderer = Callable[[Path], Path]


# --- selection ---------------------------------------------------------------


def load_records(paths: Sequence[str | Path]) -> list[GameRecord]:
    """Load :class:`GameRecord` rows from one or more JSONL files (torn lines skipped).

    Tolerant of a torn/partial trailing line (a hard-killed batch, ``batch.py``) and
    of a demoted partial line in the middle — the same tolerance
    :func:`catan_rl.human_data.harvest._read_records` uses.
    """
    out: list[GameRecord] = []
    for path in paths:
        p = Path(path)
        if not p.exists():
            continue
        for line in p.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                out.append(GameRecord.from_json_line(line))
            except (json.JSONDecodeError, KeyError, ValueError):
                continue
    return out


def select_gold_games(
    records: Sequence[GameRecord], count: int = DEFAULT_GOLD_COUNT
) -> list[GameRecord]:
    """Select up to ``count`` gold games — the ACCEPTED (``passed_crosscheck``) games.

    The gold exam grades the pipeline's *accepted* output (the records that enter
    the corpus), so only ``passed_crosscheck`` records are eligible. Deduped by
    :func:`game_id` preserving first-seen order (a resumed corpus can list a game
    twice across ``corpus.jsonl`` reads), so the selection is stable and
    reproducible. Rejected records are intentionally excluded — they carry no
    accepted board/openings to grade.
    """
    seen: set[str] = set()
    selected: list[GameRecord] = []
    for record in records:
        if not record.passed_crosscheck:
            continue
        gid = game_id(record)
        if gid in seen:
            continue
        seen.add(gid)
        selected.append(record)
        if len(selected) >= count:
            break
    return selected


# --- blank label template ----------------------------------------------------


def blank_label_template(record_game_id: str) -> dict[str, Any]:
    """A blank (all-answer-fields-``null``) label template for one gold game.

    The labeler fills ``board`` (resource + number by engine hex id, under
    :data:`ORIENTATION_CONVENTION`), ``openings`` (2 settlements + 2 roads per SEAT
    ROLE, in LOG PLACEMENT ORDER — ``settlements[0]`` = first placed,
    ``settlements[1]`` = second/resource-granting), and ``winner`` (one of
    :data:`WINNER_LABELS`). Nothing here is derived from the pipeline's parse.
    """
    return {
        "game_id": record_game_id,
        "instructions": f"See {_README}. Fill nulls. Do not open the _answers tree.",
        "orientation_convention": ORIENTATION_CONVENTION,
        "board": {str(h): {"resource": None, "number": None} for h in range(NUM_HEXES)},
        "openings": {
            POV_ROLE: {
                "settlements": [None, None],
                "roads": [None, None],
                "_order": "settlements[0]=first placed; settlements[1]=second (grants resources)",
            },
            OPPONENT_ROLE: {
                "settlements": [None, None],
                "roads": [None, None],
                "_order": "settlements[0]=first placed; settlements[1]=second (grants resources)",
            },
        },
        "winner": None,
        "_winner_choices": list(WINNER_LABELS),
    }


def _packet_readme(record_game_id: str) -> str:
    """The static per-packet instruction sheet (no pipeline-derived content)."""
    return (
        f"# Gold blind-labeling packet — `{record_game_id}`\n\n"
        "Fill `label_template.json` **only** from the frame PNGs in this folder. "
        "Do not open the pipeline's answer key (it lives outside this packet, under "
        f"`{ANSWERS_DIRNAME}/`).\n\n"
        "## Frames\n"
        f"- `{_POST_SETUP_PNG}` — the post-setup board (8 pieces down): read the board "
        "layout and both players' openings here.\n"
        "- `mid_setup_*.png` — extra mid-setup frames for disambiguation.\n"
        f"- `{_LOG_SETUP_PNG}` — the setup log panel (draft order + starting-resource grants).\n"
        f"- `{_LOG_TERMINAL_PNG}` — the terminal log panel: read the winner from the "
        "`won the game!` line ONLY (not the score counter, not any banner).\n"
        f"- `{_GRID_PNG}` — the engine-id reference grid (static; maps hex/vertex/edge ids).\n\n"
        "## Orientation convention\n"
        f"{ORIENTATION_CONVENTION}\n\n"
        "## Fields\n"
        "- `board.<hex_id>` — the resource (WOOD/BRICK/WHEAT/ORE/SHEEP/DESERT) and number "
        "(2..12, no 7; DESERT has number null) at that engine hex id.\n"
        "- `openings.pov` / `openings.opponent` — `pov` is the bottom self-seat "
        "(ThePhantom); `opponent` the top seat. Give 2 settlement vertex ids + 2 road edge "
        "ids per seat, in placement order (`[0]`=first, `[1]`=second/granting).\n"
        f"- `winner` — one of {list(WINNER_LABELS)} "
        "(`none` = resign / video cutoff before 15 VP).\n"
    )


# --- blind packet writer -----------------------------------------------------


def _save_png(path: Path, array: npt.NDArray[np.uint8]) -> None:
    """Write an RGB ``uint8`` array to a PNG (headless matplotlib, no ``gui/``)."""
    from matplotlib import image as mpimg  # lazy heavy dep; keeps import-light

    mpimg.imsave(str(path), np.ascontiguousarray(array))


def write_answer_key(record: GameRecord, answers_dir: Path) -> Path:
    """Write the pipeline's record for one game to the SIBLING answer-key tree.

    Kept OUTSIDE the blind packet (``score`` reads it, the labeler never does), so
    the packet stays blind (:func:`assert_packet_blind`).
    """
    answers_dir.mkdir(parents=True, exist_ok=True)
    path = answers_dir / f"{game_id(record)}.json"
    path.write_text(json.dumps(record.to_dict(), indent=1, sort_keys=True), encoding="utf-8")
    return path


def write_blind_packet(
    record: GameRecord,
    frames: GoldFrames,
    gold_dir: str | Path,
    *,
    render_grid: GridRenderer | None = None,
) -> Path:
    """Write one game's BLIND-LABELING packet (+ its sibling answer key).

    Creates ``<gold_dir>/<game_id>/`` with the frame PNGs, the static reference
    grid, a blank label template, and a README — and writes the pipeline's record
    to ``<gold_dir>/_answers/<game_id>.json`` (OUTSIDE the packet). Returns the
    packet directory. ``render_grid`` defaults to :func:`render_reference_grid`; a
    test may inject a stub to avoid the matplotlib render.
    """
    gold_dir = Path(gold_dir)
    gid = game_id(record)
    packet = gold_dir / gid
    packet.mkdir(parents=True, exist_ok=True)

    _save_png(packet / _POST_SETUP_PNG, frames.post_setup)
    for i, mid in enumerate(frames.mid_setup, start=1):
        _save_png(packet / f"mid_setup_{i}.png", mid)
    _save_png(packet / _LOG_SETUP_PNG, frames.setup_log_crop)
    _save_png(packet / _LOG_TERMINAL_PNG, frames.terminal_log_crop)

    grid = render_grid if render_grid is not None else render_reference_grid
    grid(packet / _GRID_PNG)

    (packet / _LABEL_JSON).write_text(
        json.dumps(blank_label_template(gid), indent=2, sort_keys=True), encoding="utf-8"
    )
    (packet / _README).write_text(_packet_readme(gid), encoding="utf-8")

    write_answer_key(record, gold_dir / ANSWERS_DIRNAME)
    return packet


def prepare_packets(
    records: Sequence[GameRecord],
    frame_provider: FrameProvider,
    gold_dir: str | Path,
    *,
    count: int = DEFAULT_GOLD_COUNT,
    render_grid: GridRenderer | None = None,
) -> list[Path]:
    """Select the gold games and write a blind packet for each (the ``prepare`` core).

    Selection is :func:`select_gold_games` (accepted games, deduped, up to
    ``count``); frames come from the injected ``frame_provider`` (the real provider
    re-ingests the video, a real-run CV path; tests inject a stub). Returns the list
    of written packet directories.
    """
    selected = select_gold_games(records, count)
    packets: list[Path] = []
    for record in selected:
        frames = frame_provider(record)
        packets.append(write_blind_packet(record, frames, gold_dir, render_grid=render_grid))
    return packets


# --- blindness verifier ------------------------------------------------------


class PacketNotBlindError(AssertionError):
    """A gold packet leaked pipeline-derived content (blindness violated)."""


def _is_blank_label(label: dict[str, Any]) -> bool:
    """Whether ``label`` has every answer field ``null`` (a blank template)."""
    board = label.get("board", {})
    for cell in board.values():
        if cell.get("resource") is not None or cell.get("number") is not None:
            return False
    for role in (POV_ROLE, OPPONENT_ROLE):
        opening = label.get("openings", {}).get(role, {})
        if any(v is not None for v in opening.get("settlements", [])):
            return False
        if any(v is not None for v in opening.get("roads", [])):
            return False
    return label.get("winner") is None


def assert_packet_blind(packet_dir: str | Path, record: GameRecord) -> None:
    """Prove a packet contains NOTHING derived from the pipeline's parse of the game.

    Raises :class:`PacketNotBlindError` on any leak. Checks (build brief §4):

    1. Only allowlisted files are present (frame PNGs / grid / blank template /
       README) — no stray answer/record/overlay file.
    2. The label template is a BLANK template (every answer field ``null``).
    3. The answer key is NOT inside the packet (it lives in the sibling
       ``_answers/`` tree).
    4. No text file in the packet embeds the record's board or winner answers (the
       serialized ``board`` object, or the winner handle).
    """
    packet = Path(packet_dir)
    gid = game_id(record)

    # (1) allowlist — only blind artifacts, plus the numbered mid-setup frames.
    allowed_exact = {
        _POST_SETUP_PNG,
        _LOG_SETUP_PNG,
        _LOG_TERMINAL_PNG,
        _GRID_PNG,
        _LABEL_JSON,
        _README,
    }
    for entry in packet.iterdir():
        name = entry.name
        if name in allowed_exact:
            continue
        if name.startswith("mid_setup_") and name.endswith(".png"):
            continue
        raise PacketNotBlindError(f"{gid}: unexpected file in blind packet: {name!r}")

    # (2) label template is blank.
    label_path = packet / _LABEL_JSON
    if not label_path.exists():
        raise PacketNotBlindError(f"{gid}: missing {_LABEL_JSON}")
    label = json.loads(label_path.read_text(encoding="utf-8"))
    if not _is_blank_label(label):
        raise PacketNotBlindError(f"{gid}: label template is not blank (answer fields leaked)")

    # (3) answer key not inside the packet.
    if (packet / f"{gid}.json").exists() or (packet / ANSWERS_DIRNAME).exists():
        raise PacketNotBlindError(f"{gid}: pipeline answer key is inside the blind packet")

    # (4) no text file embeds the record's answers.
    board_answer = json.dumps(
        [{"resource": h["resource"], "number": h.get("number")} for h in record.hexes],
        sort_keys=True,
    )
    forbidden = [board_answer]
    if record.winner is not None:
        forbidden.append(f'"winner": {json.dumps(record.winner)}')
        forbidden.append(f'"winner":{json.dumps(record.winner)}')
    for entry in packet.iterdir():
        if entry.suffix.lower() not in (".json", ".md", ".txt"):
            continue
        text = entry.read_text(encoding="utf-8")
        for token in forbidden:
            if token in text:
                raise PacketNotBlindError(
                    f"{gid}: {entry.name!r} embeds a pipeline answer ({token[:40]!r}…)"
                )


# --- D6 orientation-flip detection -------------------------------------------


def d6_hex_permutations(template: EngineTemplate | None = None) -> tuple[tuple[int, ...], ...]:
    """The 12 D6 hex-id permutations of the standard board, derived from geometry.

    The 19-hex board is D6-symmetric (brief §5.2). Each of the 6 rotations x 2
    reflections about the board centroid maps the hex-centre lattice onto itself;
    the induced permutation on engine hex ids is recovered by nearest-centre
    matching. ``perm[h]`` is the hex id that engine hex ``h`` lands on under the
    transform. Returns the deduped group (12 elements, identity included). Derived
    purely from the committed :class:`~catan_rl.human_data.board_cv.EngineTemplate`
    geometry — no engine import.
    """
    tmpl = template if template is not None else load_engine_template()
    centers = tmpl.hex_centers
    centroid = centers.mean(axis=0)
    rel = centers - centroid
    perms: set[tuple[int, ...]] = set()
    for reflect in (1.0, -1.0):
        for k in range(6):
            ang = math.radians(60 * k)
            rot = np.array([[math.cos(ang), -math.sin(ang)], [math.sin(ang), math.cos(ang)]], float)
            transform = rot @ np.array([[1.0, 0.0], [0.0, reflect]], float)
            moved = rel @ transform.T
            perm: list[int] = []
            for i in range(NUM_HEXES):
                dists = np.linalg.norm(rel - moved[i], axis=1)
                perm.append(int(dists.argmin()))
            if sorted(perm) == list(range(NUM_HEXES)):
                perms.add(tuple(perm))
    return tuple(sorted(perms))


def _board_cells(hexes: Sequence[dict[str, Any]]) -> dict[int, tuple[Any, Any]]:
    """``{hex_id: (resource, number)}`` for a record's or label's board."""
    return {int(h): (h_res, h_num) for h, h_res, h_num in _iter_hex_cells(hexes)}


def _iter_hex_cells(hexes: Sequence[dict[str, Any]]) -> list[tuple[int, Any, Any]]:
    out: list[tuple[int, Any, Any]] = []
    for h in hexes:
        out.append((int(h["hex_id"]), h.get("resource"), h.get("number")))
    return out


def _label_board_cells(label: dict[str, Any]) -> dict[int, tuple[Any, Any]]:
    """``{hex_id: (resource, number)}`` from a filled label's ``board`` object."""
    board = label.get("board", {})
    cells: dict[int, tuple[Any, Any]] = {}
    for key, cell in board.items():
        cells[int(key)] = (cell.get("resource"), cell.get("number"))
    return cells


def is_orientation_flip(
    label: dict[str, Any],
    record: GameRecord,
    permutations: Sequence[Sequence[int]] | None = None,
) -> bool:
    """Whether the labeled board is a NON-IDENTITY D6 relabeling of the record board.

    An orientation flip means the pipeline locked the wrong D6 orientation (brief
    §5.2): the human-blind board matches the record's under some non-identity D6
    permutation but NOT under the identity. Formally, a flip iff the identity
    match is imperfect (< 19 hexes) AND some non-identity permutation ``p`` yields a
    PERFECT match (``label[h] == record[p[h]]`` for every hex ``h``). A partially
    filled label (unfilled hexes are ``None``) can never match perfectly, so it
    never spuriously flags.
    """
    perms = permutations if permutations is not None else d6_hex_permutations()
    label_cells = _label_board_cells(label)
    record_cells = _board_cells(record.hexes)
    identity = tuple(range(NUM_HEXES))

    identity_matches = sum(1 for h in range(NUM_HEXES) if label_cells.get(h) == record_cells.get(h))
    if identity_matches == NUM_HEXES:
        return False
    for perm in perms:
        if tuple(perm) == identity:
            continue
        if all(label_cells.get(h) == record_cells.get(int(perm[h])) for h in range(NUM_HEXES)):
            return True
    return False


# --- field-by-field scoring --------------------------------------------------


@dataclass(frozen=True, slots=True)
class FieldScore:
    """Per-field accuracy against a pre-registered bar, with a Wilson CI."""

    name: str
    correct: int
    total: int
    bar: float
    ci: WilsonInterval

    @property
    def accuracy(self) -> float:
        return (self.correct / self.total) if self.total else 0.0

    @property
    def passed(self) -> bool:
        """Bar is met iff the POINT accuracy clears it (the primary gate); the CI is
        reported for context on the achievable ``n`` (brief §5.4)."""
        return self.total > 0 and self.accuracy >= self.bar


@dataclass(frozen=True, slots=True)
class GoldScoreReport:
    """The full gold-audit result (per-field scores + orientation flips + verdict)."""

    board: FieldScore
    openings: FieldScore
    winner: FieldScore
    orientation_flips: int
    n_games: int
    scored_game_ids: tuple[str, ...]
    skipped_unlabeled: tuple[str, ...] = field(default=())

    @property
    def flips_ok(self) -> bool:
        return self.orientation_flips <= ORIENTATION_FLIPS_BAR

    @property
    def ready(self) -> bool:
        """READY iff every bar is met and there are zero orientation flips."""
        return (
            self.n_games > 0
            and self.board.passed
            and self.openings.passed
            and self.winner.passed
            and self.flips_ok
        )

    def failures(self) -> list[str]:
        out: list[str] = []
        if self.n_games == 0:
            out.append("no labeled gold games to score")
            return out
        if not self.board.passed:
            out.append(f"board {self.board.accuracy:.4f} < bar {self.board.bar}")
        if not self.openings.passed:
            out.append(f"openings {self.openings.accuracy:.4f} < bar {self.openings.bar}")
        if not self.winner.passed:
            out.append(f"winner {self.winner.accuracy:.4f} < bar {self.winner.bar}")
        if not self.flips_ok:
            out.append(f"orientation_flips {self.orientation_flips} > bar {ORIENTATION_FLIPS_BAR}")
        return out


def _score_board(label: dict[str, Any], record: GameRecord) -> tuple[int, int]:
    """(correct hexes, 19): a hex is correct iff BOTH resource and number match."""
    label_cells = _label_board_cells(label)
    record_cells = _board_cells(record.hexes)
    correct = sum(1 for h in range(NUM_HEXES) if label_cells.get(h) == record_cells.get(h))
    return correct, NUM_HEXES


def _score_openings(label: dict[str, Any], record: GameRecord) -> tuple[int, int]:
    """(correct placements, 8): 2 settlements + 2 roads per seat, in placement order.

    Seat roles map to record seats: ``pov`` → ``players["agent"]`` (bottom self-seat,
    ThePhantom), ``opponent`` → ``players["opponent"]`` (top seat). A placement is
    correct iff the labeled id equals the record id at the SAME order index.
    """
    role_to_handle = {
        POV_ROLE: record.players["agent"],
        OPPONENT_ROLE: record.players["opponent"],
    }
    label_openings = label.get("openings", {})
    correct = 0
    total = 0
    for role, handle in role_to_handle.items():
        rec_opening = record.openings[handle]
        lab = label_openings.get(role, {})
        lab_settlements = list(lab.get("settlements", [None, None]))
        lab_roads = list(lab.get("roads", [None, None]))
        for i in range(2):
            total += 1
            if i < len(lab_settlements) and lab_settlements[i] == rec_opening.settlements[i]:
                correct += 1
        for i in range(2):
            total += 1
            if i < len(lab_roads) and lab_roads[i] == rec_opening.roads[i]:
                correct += 1
    return correct, total


def _record_winner_label(record: GameRecord) -> str:
    """The record's winner as a seat-role label (``pov`` / ``opponent`` / ``none``)."""
    if record.winner is None:
        return "none"
    if record.winner == record.players["agent"]:
        return POV_ROLE
    return OPPONENT_ROLE


def _score_winner(label: dict[str, Any], record: GameRecord) -> tuple[int, int]:
    """(1 or 0, 1): correct iff the labeled winner role equals the record's."""
    labeled = label.get("winner")
    return (1 if labeled == _record_winner_label(record) else 0), 1


def _label_is_filled(label: dict[str, Any]) -> bool:
    """Whether a label has been filled (any answer field set) — else it is skipped."""
    return not _is_blank_label(label)


def score_gold(
    gold_dir: str | Path,
    *,
    labels_dir: str | Path | None = None,
    answers_dir: str | Path | None = None,
    alpha: float = 0.05,
) -> GoldScoreReport:
    """Grade completed gold labels against the pipeline answer keys (the ``score`` core).

    For each game with BOTH a filled label and an answer key, compares board /
    openings / winner field-by-field and detects orientation flips, then aggregates
    to a :class:`GoldScoreReport` with per-field Wilson CIs.

    Labels are read from ``<labels_dir>/<game_id>.json`` when ``labels_dir`` is
    given, else from the in-packet ``<gold_dir>/<game_id>/label_template.json``
    (the labeler fills it in place). Answer keys are read from ``answers_dir`` (else
    ``<gold_dir>/_answers``). A game whose label is still blank is SKIPPED (not
    scored) and reported in ``skipped_unlabeled``.
    """
    gold_dir = Path(gold_dir)
    ans_dir = Path(answers_dir) if answers_dir is not None else gold_dir / ANSWERS_DIRNAME
    lab_dir = Path(labels_dir) if labels_dir is not None else None

    perms = d6_hex_permutations()

    board_c = board_t = 0
    open_c = open_t = 0
    win_c = win_t = 0
    flips = 0
    scored: list[str] = []
    skipped: list[str] = []

    if not ans_dir.exists():
        return _empty_report(alpha)

    for ans_path in sorted(ans_dir.glob("*.json")):
        gid = ans_path.stem
        record = GameRecord.from_dict(json.loads(ans_path.read_text(encoding="utf-8")))
        label_path = (
            (lab_dir / f"{gid}.json") if lab_dir is not None else (gold_dir / gid / _LABEL_JSON)
        )
        if not label_path.exists():
            skipped.append(gid)
            continue
        label = json.loads(label_path.read_text(encoding="utf-8"))
        if not _label_is_filled(label):
            skipped.append(gid)
            continue

        bc, bt = _score_board(label, record)
        oc, ot = _score_openings(label, record)
        wc, wt = _score_winner(label, record)
        board_c += bc
        board_t += bt
        open_c += oc
        open_t += ot
        win_c += wc
        win_t += wt
        if is_orientation_flip(label, record, perms):
            flips += 1
        scored.append(gid)

    return GoldScoreReport(
        board=_field_score("board", board_c, board_t, BOARD_BAR, alpha),
        openings=_field_score("openings", open_c, open_t, OPENINGS_BAR, alpha),
        winner=_field_score("winner", win_c, win_t, WINNER_BAR, alpha),
        orientation_flips=flips,
        n_games=len(scored),
        scored_game_ids=tuple(scored),
        skipped_unlabeled=tuple(skipped),
    )


def _field_score(name: str, correct: int, total: int, bar: float, alpha: float) -> FieldScore:
    ci = (
        wilson_interval(wins=correct, n=total, alpha=alpha)
        if total > 0
        else WilsonInterval(point=0.0, lower=0.0, upper=0.0, n=0, alpha=alpha)
    )
    return FieldScore(name=name, correct=correct, total=total, bar=bar, ci=ci)


def _empty_report(alpha: float) -> GoldScoreReport:
    return GoldScoreReport(
        board=_field_score("board", 0, 0, BOARD_BAR, alpha),
        openings=_field_score("openings", 0, 0, OPENINGS_BAR, alpha),
        winner=_field_score("winner", 0, 0, WINNER_BAR, alpha),
        orientation_flips=0,
        n_games=0,
        scored_game_ids=(),
        skipped_unlabeled=(),
    )


# --- report rendering --------------------------------------------------------


def render_score_report(report: GoldScoreReport) -> str:
    """Render the gold-audit report as markdown (written to ``gold_gate_report.md``)."""
    verdict = "READY" if report.ready else "NOT READY"
    lines: list[str] = []
    lines.append("# Gold-Gate Report — 30-game blind-labeling exam")
    lines.append("")
    lines.append(
        f"**Verdict:** {verdict} — {report.n_games} labeled game(s) scored "
        f"({len(report.skipped_unlabeled)} unlabeled skipped)."
    )
    lines.append("")
    lines.append("| field | correct / total | accuracy | 95% CI | bar | pass |")
    lines.append("|---|---|---|---|---|---|")
    for fs in (report.board, report.openings, report.winner):
        ci = f"[{fs.ci.lower:.3f}, {fs.ci.upper:.3f}]" if fs.total else "—"
        mark = "✅" if fs.passed else "❌"
        lines.append(
            f"| {fs.name} | {fs.correct} / {fs.total} | {fs.accuracy:.4f} | {ci} "
            f"| {fs.bar} | {mark} |"
        )
    flips_mark = "✅" if report.flips_ok else "❌"
    lines.append(
        f"| orientation_flips | {report.orientation_flips} | — | — "
        f"| {ORIENTATION_FLIPS_BAR} | {flips_mark} |"
    )
    lines.append("")
    if report.failures():
        lines.append("**Open failures:** " + "; ".join(report.failures()))
    else:
        lines.append("**Open failures:** none — every pre-registered bar is met.")
    lines.append("")
    lines.append(
        "Bars (pre-registered, step6 §3): board ≥ 98% of hexes, openings ≥ 95% of "
        "placements, winner ~100%, orientation flips = 0. Accuracies are the primary "
        "gate; Wilson CIs report the achievable n (brief §5.4 — the gold set is small, "
        "so a bar is graded on the point estimate with the CI for context)."
    )
    if report.scored_game_ids:
        lines.append("")
        lines.append("Scored games: " + ", ".join(f"`{g}`" for g in report.scored_game_ids) + ".")
    return "\n".join(lines) + "\n"


def write_score_report(report: GoldScoreReport, report_path: str | Path) -> Path:
    """Render + write the gold report to ``report_path`` (docs/plans/gold_gate_report.md)."""
    path = Path(report_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_score_report(report), encoding="utf-8")
    return path


# --- reference-grid renderer -------------------------------------------------


def render_reference_grid(
    out_path: str | Path,
    *,
    template: EngineTemplate | None = None,
    topology: Topology | None = None,
    dpi: int = 150,
) -> Path:
    """Render the canonical engine-id reference grid to ``out_path`` (headless PNG).

    Maps every engine hex (0..18), vertex (0..53) and edge (0..71) integer id to its
    canonical position so a labeler can name ids WITHOUT any pipeline output. Reuses
    the committed engine template geometry (:func:`load_engine_template` +
    :func:`load_topology`); rendered under :data:`ORIENTATION_CONVENTION` (screen
    y-down, so engine hex 8 sits at the top and hex 11 at the right). Headless
    matplotlib (Agg) — **no ``gui/`` import**. Returns the written path.
    """
    import matplotlib

    matplotlib.use("Agg")  # headless; never a GUI backend
    import matplotlib.pyplot as plt
    from matplotlib.patches import RegularPolygon

    tmpl = template if template is not None else load_engine_template()
    topo = topology if topology is not None else load_topology()
    hex_centers = tmpl.hex_centers
    vertex_px = tmpl.vertex_px

    fig, ax = plt.subplots(figsize=(11, 11))
    ax.set_aspect("equal")

    # Hex tiles: pointy-top RegularPolygon (a vertex points up), radius = centre→vertex.
    radius = float(np.linalg.norm(vertex_px[2] - hex_centers[0]))
    for hid in range(NUM_HEXES):
        cx, cy = float(hex_centers[hid, 0]), float(hex_centers[hid, 1])
        ax.add_patch(
            RegularPolygon(
                (cx, cy),
                numVertices=6,
                radius=radius,
                orientation=0.0,
                facecolor="none",
                edgecolor="#888888",
                linewidth=1.0,
            )
        )
        ax.text(
            cx,
            cy,
            f"H{hid}",
            ha="center",
            va="center",
            fontsize=13,
            fontweight="bold",
            color="#111111",
        )

    # Edges: line segment between the two engine vertices, id at the midpoint.
    for eid, (va, vb) in enumerate(topo.edge_vertices):
        ax_, ay = float(vertex_px[va, 0]), float(vertex_px[va, 1])
        bx, by = float(vertex_px[vb, 0]), float(vertex_px[vb, 1])
        ax.plot([ax_, bx], [ay, by], color="#cccccc", linewidth=0.8, zorder=1)
        mx, my = (ax_ + bx) / 2.0, (ay + by) / 2.0
        ax.text(mx, my, str(eid), ha="center", va="center", fontsize=6, color="#c0392b", zorder=3)

    # Vertices: a dot + id.
    for vid in range(vertex_px.shape[0]):
        vx, vy = float(vertex_px[vid, 0]), float(vertex_px[vid, 1])
        ax.plot(vx, vy, marker="o", markersize=4, color="#2471a3", zorder=4)
        ax.text(
            vx,
            vy + 9,
            str(vid),
            ha="center",
            va="center",
            fontsize=6,
            color="#1a5276",
            zorder=5,
        )

    ax.invert_yaxis()  # screen y-down: hex 8 (y≈160) at top, hex 11 (rightmost) at right
    ax.margins(0.05)
    ax.axis("off")
    ax.set_title(
        "Engine-id reference grid — H# = hex id, blue = vertex id, red = edge id\n"
        "Convention: hex 8 top-centre, hex 11 rightmost (do not anchor on the desert)",
        fontsize=11,
    )

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out), dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out
