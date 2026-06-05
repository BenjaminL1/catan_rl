# v2 Colonist Render Upgrade — Shared pygame primitives for setup labeling + live human-vs-bot GUI

**Status**: design draft (single author, not yet panel-reviewed); implementation gated on the §0 preflight checks and on `v2_setup_labeling.md` having shipped the labeling tool (the first consumer of `render.py`).

**Revision history**:
- 2026-06-01 — Original draft. Shared rendering module (`src/catan_rl/gui/render.py`) plus a separated constants file (`render_constants.py`). Both the just-shipped setup-labeling tool (`src/catan_rl/labeling/ui.py`) and the existing live human-vs-AI GUI (`src/catan_rl/gui/view.py`, 761 LOC) refactor onto it. Brings 8 Colonist-inspired visual changes: blue water canvas, sandy island outline, hex bevel shading, square white number tokens with red 6/8 + pip dots, unicode resource symbols with drawn-polygon fallback, ship-shaped port markers with wooden-plank connectors, yellow/gold vertex markers, robber pawn. Three-commit delivery: primitives + tests, labeling-tool refactor, live-GUI refactor. CLAUDE.md rule 8 unchanged — RL paths still cannot import `gui/`; `labeling/` keeps the existing carve-out.

**Preflight gate** (per `v2_design.md` §0 + carry-forward from `v2_setup_labeling.md` §0): the visual upgrade does **not** start until:
  - The labeling tool (`v2_setup_labeling.md`) has shipped end-to-end. `render.py` is *layered on top of* the existing pygame consumers; without a working `labeling/ui.py` the smoke-test consumer for commit 1 doesn't exist.
  - **All 485 v2 unit tests green at the tip of the branch this work forks from.** The new render module's tests add to this baseline, not replace it. Any pre-existing red is fixed before commit 1 lands.
  - **Pygame headless availability**: `SDL_VIDEODRIVER=dummy pytest tests/unit/labeling/test_ui.py` passes locally on M1 Pro. This was already pinned by the labeling tool work; restated here because the new tests use the same driver.

This doc is the planning equivalent of `v2_step3_bc.md` / `v2_step4_ppo.md` / `v2_setup_labeling.md` — it specifies what gets built, how it's tested, what numbers count as success, and where the risks are. The motivating reference is the Colonist.io game-board screenshot the user supplied this session; the goal is to bring both pygame surfaces (the labeling tool and the live game GUI) up to that visual quality bar without changing any game logic, action space, or engine state.

## Inputs

- v2 engine (read-only at the rendering layer): `catanGame(render_mode=None|"human")` from `src/catan_rl/engine/{game.py, board.py}`. The engine's `__init__` constructs `catanGameView(self.board, self)` when `render_mode="human"` (`engine/game.py:64-66`); that public API stays stable through this work.
- Existing GUI surfaces:
  - `src/catan_rl/gui/view.py` — live human-vs-AI game UI, 761 LOC, legacy.
  - `src/catan_rl/labeling/ui.py` — labeling tool (just shipped per `v2_setup_labeling.md`).
  - `src/catan_rl/viz/debug_board.py` — debug index visualizer; **OUT OF SCOPE** for this upgrade. Stays on whatever rendering path it already uses. (No reason to entangle a debug-only tool with the player-facing visual upgrade.)
- Engine board API used by the renderer (read-only):
  - `hex_tile.to_pixel(board.flat)` — per-hex pixel center.
  - `hex_tile.get_corners(board.flat)` — six corner pixels per hex.
  - `hex_tile.has_robber: bool` — robber location flag.
  - `board.vertex_index_to_pixel_dict` — 54-entry vertex pixel table.
  - Each port-edge vertex has a `.port: str` attribute (both endpoints of a port edge carry the same string, e.g. `"2:1 BRICK"` or `"3:1 PORT"`).
- 1v1 Colonist.io ruleset (`docs/1v1_rules.md`): the renderer is **read-only** w.r.t. rules; it does not encode any rule (the engine does). Two visual touches are rule-derived: the 6/8 red-number convention and pip-dot counts as the 2d6 probability mass. Both are conventions, not rule changes.
- CLAUDE.md rule 8: RL paths (`bc/`, `policy/`, `env/`, `ppo/`, `selfplay/`, `eval/`) cannot import from `gui/`. `labeling/` is explicitly allowed (existing carve-out). The new `render.py` lives under `gui/`, so the ban applies automatically.
- Visual reference: Colonist.io game-board screenshot supplied by the user this session (canonical reference for "what does the upgrade aim at"). Stored mentally / verbally — not committed as an asset file (no asset files ship with this work; see §1 decision).

## Outputs

- `src/catan_rl/gui/render.py` — new module, ~250-350 LOC of pure pygame-primitive draw functions. No state. No file IO. No asset loads. Consumed by both `labeling/ui.py` and `gui/view.py`.
- `src/catan_rl/gui/render_constants.py` — new ~50 LOC of color / font-size / layout constants. Separated so `render.py` stays focused on draw logic and tunable values are grep-discoverable.
- `src/catan_rl/labeling/ui.py` — modified. Inline tile / port / vertex / robber rendering replaced with `render.py` calls. State machine + click handling + persistence layer **unchanged**.
- `src/catan_rl/gui/view.py` — modified. Surgical replacement of the tile / port / vertex / robber rendering call sites with `render.py` calls. Public method signatures (`displayGameScreen()`, `displayDiceRoll(num)`, `buildRoad_display(...)`, etc.) **unchanged**. Font setup, screen size, and event loop preserved.
- `tests/unit/gui/test_render.py` — new, ~20 per-primitive tests.
- `tests/unit/gui/test_unicode_fallback.py` — new, ~4 fallback-decision tests.
- `tests/integration/test_render_full_board_smoke.py` — new, end-to-end full-board render + perf canary.
- Acceptance criterion (§6) gates the visual upgrade as a single deliverable. The compound gate is human visual approval + all-tests-green + perf canary.

---

## 0. Preflight checks (block kickoff)

Three lightweight checks. Each pass-or-fix-before-coding. These mirror the §0 spirit of `v2_setup_labeling.md` — calibrate against measurement, not rhetoric.

### 0.1 — Pygame system-font emoji coverage probe

**Question**: does the M1 Pro target system font render the seven resource emoji glyphs (`🌲 🧱 🐑 🌾 ⛰ 🌵`) with non-blank, non-degenerate bitmaps? The unicode-symbol render path of decision #3 in §1 only carries its design intent if this is yes; otherwise the *fallback* path is the load-bearing path and we should know that before committing primitive code.

**Method**: in a 30-line ad-hoc script (no committed code), instantiate `pygame.font.SysFont(None, 28)`, render each candidate emoji into a `Surface`, compute the bounding box of non-transparent pixels, and report (a) the bbox width, (b) the number of non-background-color pixels. Decision rule: pass iff every emoji has bbox width ≥ 12 px AND ≥ 30 non-background pixels. Repeat once with `pygame.font.SysFont("Apple Color Emoji", 28)` if the default font fails — record which font name to wire into `render_constants.py`.

**Outcome usage**: the probe's result becomes the default value of the runtime fallback decision in `render.py`'s `_should_use_emoji(resource_type)` cached-decision logic. If the probe reports total failure on the default + Apple Color Emoji + any other system font, switch the plan to **fallback-only** mode — `render.py` ships with no emoji path at all, just drawn-polygon icons. The §1 decision tree adapts to the probe; it does not pretend emoji rendering will work without proof.

**Decision rule**: pass iff at least one tested font passes for ALL seven resource glyphs. **Fail**: commit only the drawn-polygon fallback path (delete the emoji branch from §1), update `test_unicode_fallback.py` accordingly (the fallback-only path means the cache is pre-populated with `True` for every resource), and proceed.

### 0.2 — Pixel-color spot-check tolerance calibration

**Question**: under `SDL_VIDEODRIVER=dummy` on M1 Pro, when we fill a region with RGB `(35, 90, 150)` (water blue) and read back a single pixel via `screen.get_at((x, y))`, how far does the read-back drift from the source RGB? Anti-aliasing, alpha-channel compositing, and surface format conversion can perturb pixel values.

**Method**: ad-hoc script. Fill a 100×100 surface with each of the six core colors used by the upgrade (water `(35, 90, 150)`, sand `(235, 215, 165)`, P1 blue, P2 red, gold `(255, 200, 50)`, gray robber `(100, 100, 100)`). For each color, sample 100 random interior pixels via `surface.get_at(...)`. Report the max per-channel drift across all 600 samples.

**Decision rule**: pass iff max drift ≤ 5 per channel. If ≤ 5, the `test_render.py` per-primitive tests use `tolerance = 10` (2× the measured drift) for headroom. **Fail (drift > 5)**: investigate — most likely an alpha-blending bug in the test harness. Tolerance is widened to `2 × measured_drift` and the calibration value is recorded as a constant in `render_constants.py` (`PIXEL_SPOT_CHECK_TOL`).

### 0.3 — Refactor-blast-radius pin on `gui/view.py`

**Question**: when we refactor `view.py` (761 LOC of legacy code) to call `render.py`, which legacy methods are touched and which are left alone? Without an upfront pin, the commit 3 refactor risks ballooning past the 500-LOC commit cap (§13).

**Method**: read `gui/view.py` end-to-end. Build a table mapping each `_draw_*` / `display*` / `buildRoad_display` / etc. method to one of three categories: (a) internal `_draw_*` helper, refactor to `render.py` call; (b) public method signature on `catanGameView`, **do not change** (preserve API); (c) dice / top-bar / button rendering, **out of scope** (the upgrade is about the board, not the chrome).

**Decision rule**: pin the table in the §F.2 commit 3 sub-section before any code is written. If the (a) category exceeds 12 methods, split commit 3 into commits 3a + 3b per the risk register (§9). Hard-cap: no commit > 500 LOC.

---

## 1. Renderer-location and function-shape decision

**Decision**: single module at `src/catan_rl/gui/render.py` exposing **pure functions** taking a `pygame.Surface` plus scene data; a sibling `src/catan_rl/gui/render_constants.py` for color / font / layout constants. Locked.

**Function surface** (per §A below, abbreviated here):

```python
def draw_water(screen: pygame.Surface, size: tuple[int, int]) -> None: ...
def draw_island_outline(screen: pygame.Surface, hex_centers: list[tuple[int, int]]) -> None: ...
def draw_hex_tile(screen: pygame.Surface, hex_tile, board, base_color: tuple[int, int, int], with_bevel: bool = True) -> None: ...
def draw_number_token(screen: pygame.Surface, center: tuple[int, int], number: int) -> None: ...
def draw_resource_symbol(screen: pygame.Surface, center: tuple[int, int], resource_type: str) -> None: ...
def draw_port_ship(screen: pygame.Surface, label: str, anchor: tuple[int, int]) -> None: ...
def draw_port_planks(screen: pygame.Surface, anchor: tuple[int, int], v1_pixel: tuple[int, int], v2_pixel: tuple[int, int]) -> None: ...
def draw_vertex_marker(screen: pygame.Surface, pixel: tuple[int, int], state: VertexState) -> None: ...
def draw_robber_pawn(screen: pygame.Surface, hex_tile, board) -> None: ...
```

Each function ≤ 30 LOC. No state. No `self`. No module-level mutable globals except the unicode-fallback decision cache (§3 below; populated once per process at module import).

**Rationale**:

| Option | Build cost | Sharing between consumers | RL-path import ban (CLAUDE.md rule 8) |
|---|---|---|---|
| Inline rendering in each consumer (status quo) | 0 | None — divergence between `labeling/ui.py` and `gui/view.py` over time | N/A — already in `gui/` |
| Class-based `BoardRenderer(state)` | Medium — constructor state, instance lifetime | Shared but consumers must construct an instance and keep it alive | N/A |
| **Pure functions in `gui/render.py`** (chosen) | **Low** — module + functions, no instances | **Shared with zero coupling**; both consumers pass scene data per call | N/A — under `gui/`, RL ban applies automatically |
| Separate package `src/catan_rl/render/` | Medium — new package, new import paths | Same as pure-functions option | Would require new CLAUDE.md rule 8 carve-out OR risk RL paths accidentally importing |

The decisive factor: pure functions under `gui/` are the cheapest path that (a) keeps both consumers in sync, (b) inherits the existing CLAUDE.md rule 8 ban without adding new exceptions, and (c) lets `tests/unit/gui/test_render.py` test each primitive in isolation with no fixture state.

**Carry-forward**: if a future consumer (e.g., a web-based viewer porting the same visuals to SVG) emerges, this module becomes the contract. Until then, no abstraction layer is built — YAGNI.

## 2. Constants-vs-logic separation decision

**Decision**: split colors / font sizes / layout constants into `render_constants.py`; keep `render.py` for draw logic. Locked.

**Rationale**:
- Tuning the visual look (per the §6 Gate 1 human spot-check) is expected to be iterative. A separate constants file means tuning is a one-line PR in commit-ready form, not a hunt through 300 LOC of draw code.
- Grep-discoverability: `grep -n WATER_COLOR src/catan_rl/gui/render_constants.py` returns one line; the same grep in `render.py` would return many call sites.
- Test isolation: `test_render.py` imports constants directly; tests can pin specific RGB values without coupling to draw-call structure.

**Carry-forward**: if `render_constants.py` exceeds 100 LOC, consider further splitting into `render_colors.py` + `render_layout.py`. Not anticipated at the §A scope.

## 3. Unicode-symbol fallback decision

**Decision**: at module init, render each emoji once into a probe surface; cache a per-resource boolean `_USE_EMOJI[resource_type]`. If the probe fails for a resource, that resource falls through to a drawn-polygon fallback icon. Cache is process-lifetime; the probe runs exactly once per process per resource. Locked.

**Probe specification** (per §0.1 calibration):
- Render the candidate emoji into a `Surface` via `font.render(emoji, True, FG_COLOR)`.
- Compute the non-background-color bounding box.
- Decision rule (matches §0.1): emoji OK iff bbox width ≥ 12 px AND ≥ 30 non-background pixels.
- On fail, `_USE_EMOJI[resource_type] = False` and the resource gets the drawn-polygon path for the rest of the process lifetime.

**Mapping**:
| Resource | Emoji | Drawn-polygon fallback |
|---|---|---|
| WOOD | 🌲 | 3 stacked dark-green triangles |
| BRICK | 🧱 | 2×3 grid of small brick-orange rectangles |
| SHEEP | 🐑 | White rounded blob (3-arc connected curves) on small brown legs |
| WHEAT | 🌾 | 5 short yellow vertical lines bundled at a base |
| ORE | ⛰ | 3 stacked gray quadrilaterals (rocks) |
| DESERT | 🌵 | Single green cactus shape (3 polygon segments) |

Each fallback ≤ 10 LOC inside `render.py`. No image assets.

**Rationale**: macOS ships Apple Color Emoji, so the §0.1 preflight is expected to pass. The fallback exists as a safety net for unknown environments (CI runners, locked-down corporate systems) — the upgrade should not silently degrade to blank squares. The cache is per-process because (a) the probe is mildly expensive (~5 ms × 6 resources at module init), (b) the answer cannot change within a process.

**Test**: `test_unicode_fallback.py` mocks `pygame.font.render` to return crafted surfaces and asserts the cache decision is right. See §8.2.

## 4. Asset-files decision

**Decision**: zero asset files. All visuals via pygame primitives + system fonts. Locked.

**Rationale**:
- No file IO at runtime (renderer remains pure-functions).
- No licensing audit (Colonist.io's actual graphics are proprietary; we're inspired by them, not copying them).
- No path-resolution bugs (asset paths break under different installs, editable vs wheel, conda vs system Python).
- The drawn-polygon fallback icons (§3) are already in-code; no incremental complexity to keep the entire renderer asset-free.

**Carry-forward**: if a future visual revision genuinely needs image assets (e.g., higher-fidelity port ships), add a `src/catan_rl/gui/assets/` directory and a single-line asset-loader helper in `render.py`. Not anticipated.

## 5. Refactor strategy for `gui/view.py`

**Decision**: surgical replacement of internal `_draw_*` helpers with `render.py` calls. Public `catanGameView` methods (`displayGameScreen`, `displayDiceRoll`, `buildRoad_display`, etc.) keep their names + signatures. Font setup, screen size, and event loop stay. Locked.

**Rationale**:
- The engine constructs `catanGameView(self.board, self)` from `engine/game.py:64-66` when `render_mode="human"`; any signature change there would also change `engine/game.py`. The engine's API is out of scope for a visual upgrade.
- 761 LOC of legacy code is too large to rewrite in one PR safely. Surgical replacement keeps the diff small + reviewable + revertible.
- §0.3 preflight pins which methods are touched. Hard cap: no commit > 500 LOC. If the diff would exceed, split into commits 3a + 3b (§9 risk register).

**Method-touch pin** (filled in at §0.3 time; placeholder here):
| Method | Category | Action |
|---|---|---|
| `_draw_tile(...)` | internal `_draw_*` | replace with `render.draw_hex_tile(...)` |
| `_draw_port_label(...)` | internal `_draw_*` | replace with `render.draw_port_ship(...)` + `render.draw_port_planks(...)` |
| `_draw_robber(...)` (if exists) | internal `_draw_*` | replace with `render.draw_robber_pawn(...)` |
| `_draw_settlements(...)` | internal `_draw_*` | wrap legal-pick vertices with `render.draw_vertex_marker(...)` during human turns |
| `displayGameScreen()` | public | preserve signature; internal body calls the new helpers |
| `displayDiceRoll(num)` | public | preserve signature; dice rendering is **out of scope** (chrome, not board) |
| `buildRoad_display(...)` | public | preserve signature; road rendering is unchanged for this upgrade (already drawn as line segments, not letter labels) |

**Carry-forward**: any method whose signature changes must be flagged at PR-review time. The user wants the `engine/game.py` integration to keep working unchanged.

## 6. Refactor strategy for `labeling/ui.py`

**Decision**: replace the inline `_render_board` tile / port / vertex logic with `render.py` calls. State machine, click handling, persistence, and session manager **unchanged**. Locked.

**Why this is simpler than `gui/view.py`**:
- `labeling/ui.py` just shipped this session; it is small, structured, and already organized around a single `_render_board` method.
- No engine integration. The labeling tool consumes `catanGame(render_mode=None)` (headless); the UI is its own surface.
- The state-machine + click handling are orthogonal to rendering — no API surface changes are needed.

**Touch points**:
- `_render_board` loop body: replace inline `pygame.draw.polygon(...)` hex code with `render.draw_hex_tile(...)`.
- Port label rendering: replace inline `font.render(port_label)` + text placement with `render.draw_port_ship(...)` + `render.draw_port_planks(...)`.
- Legal-vertex green-dot loop: replace `pygame.draw.circle(...)` calls with `render.draw_vertex_marker(...)` with the appropriate `VertexState`.
- Desert hex: add a `render.draw_robber_pawn(...)` call (the desert always has the robber at game start; the labeling tool was previously not rendering it).

**Carry-forward**: the labeling tool's UI is the smoke-test surface for commit 1. After commit 2, launching `python scripts/label_setup.py --seed 2026` is the human spot-check that all 8 visual elements look right before commit 3 touches the live game GUI.

## 7. Backward-compatibility commitments

The upgrade is **visual-only**. Concrete commitments:

- `engine/game.py:64-66`'s `catanGameView(self.board, self)` construction must keep working with no changes to `engine/game.py`.
- All public methods on `catanGameView` keep their names + signatures.
- All public methods on `labeling/ui.LabelingUI` (or equivalent) keep their names + signatures.
- The `scripts/label_setup.py` CLI works unchanged after commit 2.
- The engine's `render_mode="human"` path renders a playable game after commit 3.
- The `tests/integration/test_labeling_smoke.py` integration test passes unchanged through all three commits.

Any deviation requires an explicit STOP at the relevant gate (§10) and user approval before continuing.

---

## A. Renderer module (`src/catan_rl/gui/render.py`)

**Target LOC**: ~250-350. Hard cap 400 (refactor before exceeding).

**Module structure**:

```python
"""Pure pygame-primitive draw functions for the Colonist-inspired board view.

Layer order (caller's responsibility to call in this order):
  water → island → hex tiles (with bevel) → number tokens → resource symbols →
  port planks → port ships → vertex markers → prior picks → robber → top-bar overlay.

No state. No file IO. No asset loads. The only module-level state is the
unicode-fallback decision cache, populated once per process at module import.
"""

import pygame
from .render_constants import (
    WATER_COLOR, SAND_COLOR, GOLD_COLOR, ROBBER_COLOR, ...
)

# Populated once at module import; see §3 decision + §8.2 tests.
_USE_EMOJI: dict[str, bool] = {}

def _init_unicode_fallback_cache() -> None:
    """Probe each resource emoji once; cache the boolean decision."""
    ...

_init_unicode_fallback_cache()


def draw_water(screen: pygame.Surface, size: tuple[int, int]) -> None:
    """Fill the canvas with deep ocean blue."""
    screen.fill(WATER_COLOR)


def draw_island_outline(screen: pygame.Surface, hex_centers: list[tuple[int, int]]) -> None:
    """Draw a sandy tan blob behind the hexes.

    Convex hull of the 19 hex pixel centers, buffered outward by ~50 px,
    jittered for an irregular coastline look. Deterministic given the centers
    (uses a fixed seed for the jitter).
    """
    ...


def draw_hex_tile(
    screen: pygame.Surface,
    hex_tile,
    board,
    base_color: tuple[int, int, int],
    with_bevel: bool = True,
) -> None:
    """Draw a hex polygon at hex_tile.to_pixel(board.flat) with optional bevel.

    Bevel = lighter polygon on top half + darker polygon on bottom half;
    ~15% lightness shift over base_color. Disabled by with_bevel=False.
    """
    ...


def draw_number_token(
    screen: pygame.Surface,
    center: tuple[int, int],
    number: int,
) -> None:
    """Square white token with number text + pip dots beneath.

    Number color: red for 6 or 8, dark green/black otherwise.
    Pip count = 2d6 ways to roll: 2/12=1, 3/11=2, 4/10=3, 5/9=4, 6/8=5.
    Pips drawn as small filled circles in a single horizontal row centered
    beneath the number.
    """
    ...


def draw_resource_symbol(
    screen: pygame.Surface,
    center: tuple[int, int],
    resource_type: str,
) -> None:
    """Draw a unicode emoji centered on the hex if available, else fallback polygon.

    Decision per resource_type comes from _USE_EMOJI (populated at module init).
    Fallback polygons defined inline per §3 mapping:
      WOOD = 3 dark-green triangles
      BRICK = 2x3 grid of small rectangles
      SHEEP = white blob on brown legs
      WHEAT = 5 short yellow vertical lines
      ORE = 3 stacked gray quadrilaterals
      DESERT = green cactus shape
    """
    ...


def draw_port_ship(
    screen: pygame.Surface,
    label: str,
    anchor: tuple[int, int],
) -> None:
    """Trapezoid hull + thin mast rectangle + triangular sail.

    Sail carries the port label text rendered via font_small. If label
    overflows the sail polygon, drop to abbreviated form ("2:1" only) +
    color the sail per port resource.
    """
    ...


def draw_port_planks(
    screen: pygame.Surface,
    anchor: tuple[int, int],
    v1_pixel: tuple[int, int],
    v2_pixel: tuple[int, int],
) -> None:
    """Two light-tan wood lines from anchor to each coastal vertex.

    Drawn before the ship (so the ship sits on top), but after the island
    outline (so the planks are visible against the sand).
    """
    ...


class VertexState(enum.Enum):
    LEGAL = "legal"          # gold dot, settle-legal during current pick
    SELECTED = "selected"    # brighter yellow + thicker border, currently selected
    SETTLED_P1 = "settled_p1"   # P1 color (blue)
    SETTLED_P2 = "settled_p2"   # P2 color (red)
    IDLE = "idle"            # no marker rendered (default for non-legal vertices)


def draw_vertex_marker(
    screen: pygame.Surface,
    pixel: tuple[int, int],
    state: VertexState,
) -> None:
    """Draw a vertex circle per state. IDLE state is a no-op."""
    ...


def draw_robber_pawn(
    screen: pygame.Surface,
    hex_tile,
    board,
) -> None:
    """Small gray pyramid/triangle on the hex with has_robber=True.

    No-op if hex_tile.has_robber is False — callers can pass any hex
    without a precondition check.
    """
    ...
```

**Performance budget**: per the §6 Gate 3 perf canary, a full-board render (19 hex tiles + 9 ports + 54 vertex markers + 1 robber + water + island outline) must complete in < 50 ms on M1 Pro CPU under `SDL_VIDEODRIVER=dummy`. The labeling tool runs at 30 FPS (33 ms budget) so 50 ms gives headroom for the rest of the UI (text rendering, top-bar, etc.).

## B. Constants module (`src/catan_rl/gui/render_constants.py`)

**Target LOC**: ~50. Hard cap 100.

**Categories**:

```python
# --- Colors (RGB triples) ---
WATER_COLOR: tuple[int, int, int] = (35, 90, 150)
SAND_COLOR: tuple[int, int, int] = (235, 215, 165)
GOLD_COLOR: tuple[int, int, int] = (255, 200, 50)
SELECTED_YELLOW: tuple[int, int, int] = (255, 240, 100)
ROBBER_COLOR: tuple[int, int, int] = (100, 100, 100)
PLANK_COLOR: tuple[int, int, int] = (180, 140, 80)
SAIL_COLOR: tuple[int, int, int] = (245, 240, 220)
HULL_COLOR: tuple[int, int, int] = (110, 70, 40)
P1_COLOR: tuple[int, int, int] = (60, 90, 180)
P2_COLOR: tuple[int, int, int] = (180, 60, 60)
NUMBER_TOKEN_BG: tuple[int, int, int] = (250, 245, 230)
NUMBER_RED: tuple[int, int, int] = (180, 30, 30)
NUMBER_DARK: tuple[int, int, int] = (40, 40, 40)

# Resource base colors for hex fills (Charlesworth resource order)
RESOURCE_BASE_COLORS: dict[str, tuple[int, int, int]] = {
    "WOOD": (90, 130, 70),
    "BRICK": (190, 100, 60),
    "WHEAT": (240, 200, 80),
    "ORE": (130, 130, 150),
    "SHEEP": (170, 210, 130),
    "DESERT": (220, 200, 150),
}

# --- Layout / sizes ---
VERTEX_MARKER_RADIUS: int = 9
SELECTED_MARKER_RADIUS: int = 12
SELECTED_MARKER_BORDER: int = 3
PORT_SHIP_HULL_WIDTH: int = 24
PORT_SHIP_HULL_HEIGHT: int = 8
PORT_SHIP_MAST_HEIGHT: int = 20
PIP_DOT_RADIUS: int = 2
PIP_DOT_SPACING: int = 5
NUMBER_TOKEN_SIDE: int = 28
ISLAND_BUFFER_PX: int = 50
ISLAND_JITTER_PX: int = 8
BEVEL_LIGHTNESS_DELTA: float = 0.15

# --- Font sizes ---
FONT_NUMBER_SIZE: int = 18
FONT_PORT_LABEL_SIZE: int = 10
FONT_RESOURCE_EMOJI_SIZE: int = 32

# --- Calibration constants (from §0 preflight) ---
PIXEL_SPOT_CHECK_TOL: int = 10  # set by §0.2; adjust if drift > 5
EMOJI_PROBE_MIN_BBOX_WIDTH: int = 12
EMOJI_PROBE_MIN_NON_BG_PIXELS: int = 30
```

All values are tunable in commit 1 follow-ups without touching draw logic.

---

## 8. Testing (TDD discipline, tests-first per `v2_step3_bc.md` + `v2_step4_ppo.md` + `v2_setup_labeling.md`)

Tests are written **before** implementation, per the user's established preference. The patterns below target the failure modes that bit `bc/dataset.py` (silent action filtering), the hex-board UI risk class (`v2_setup_labeling.md` §0.2), and the labeling-tool tests-first discipline already in `tests/unit/labeling/`.

### 8.1 `tests/unit/gui/test_render.py` (~20 tests)

Each test runs under `SDL_VIDEODRIVER=dummy`. Pixel reads use the `PIXEL_SPOT_CHECK_TOL` calibrated in §0.2.

- `test_draw_water_fills_canvas`: pixel at each of the 4 canvas corners + the center is water blue ± tolerance.
- `test_draw_island_outline_covers_hex_centers`: for each of 19 hex pixel centers, the pixel is sand color (not water).
- `test_draw_island_outline_outside_hull_is_water`: pixels far from the hex hull remain water blue.
- `test_draw_hex_tile_centroid_matches_resource_color`: for each resource type, pixel at hex pixel center is the resource base color ± tolerance.
- `test_draw_hex_tile_bevel_top_lighter_than_bottom`: pixel above center has higher RGB sum than pixel below (with bevel enabled).
- `test_draw_hex_tile_no_bevel_is_uniform`: with `with_bevel=False`, top + bottom pixel RGB sums differ by ≤ `PIXEL_SPOT_CHECK_TOL`.
- `test_draw_number_token_red_for_six`: render number 6, sample the center text region, assert red channel dominates (R > G + 50, R > B + 50).
- `test_draw_number_token_red_for_eight`: same for 8.
- `test_draw_number_token_non_red_for_three`: render number 3, text region red channel does NOT dominate.
- `test_pip_dots_count_matches_2d6_probability`: for each of {2,3,4,5,6,8,9,10,11,12}, render the token + count discrete dot blobs along the pip row. Expected: 1, 2, 3, 4, 5, 5, 4, 3, 2, 1.
- `test_draw_resource_symbol_emoji_path`: with `_USE_EMOJI[res] = True` (forced), centroid pixel is non-background (non-white-ish).
- `test_draw_resource_symbol_fallback_path`: with `_USE_EMOJI[res] = False` (forced), centroid pixel signature matches the fallback polygon (a per-resource pixel-signature fixture).
- `test_draw_port_ship_renders_hull_mast_sail`: bounding box of non-water pixels matches the expected ship footprint ± 2 px on each edge.
- `test_draw_port_ship_label_fits_or_abbreviates`: render label `"2:1 BRICK"`; if the text overflows the sail (`font.size > sail_polygon_width`), assert the sail is colored per the port resource (BRICK orange) and the abbreviated label `"2:1"` is rendered.
- `test_draw_port_planks_connects_anchor_to_two_vertices`: sample pixels along the lines from anchor to v1 and v2 at 10 evenly-spaced points each; assert each is plank tan ± tolerance.
- `test_draw_vertex_marker_state_legal`: pixel at marker center is gold ± tolerance.
- `test_draw_vertex_marker_state_selected`: pixel at marker center is brighter yellow ± tolerance AND marker radius > legal radius.
- `test_draw_vertex_marker_state_settled_p1`: pixel is P1 blue ± tolerance.
- `test_draw_vertex_marker_state_settled_p2`: pixel is P2 red ± tolerance.
- `test_draw_vertex_marker_state_idle_is_noop`: pixel at marker location is unchanged from the prior-rendered background.
- `test_draw_robber_pawn_appears_on_correct_hex`: render a board where exactly one hex has `has_robber=True`; assert that hex's center pixel has gray, and the 18 other hex centers do NOT have gray.
- `test_render_full_board_no_exceptions`: integration smoke calling all primitives end-to-end on a seeded `catanBoard()` build. No assertions on pixels — only that nothing raises.

### 8.2 `tests/unit/gui/test_unicode_fallback.py` (~4 tests)

- `test_fallback_when_glyph_renders_blank`: monkey-patch `pygame.font.Font.render` to return a fully-transparent surface; reload the `render` module; assert every entry in `_USE_EMOJI` is `False`.
- `test_fallback_when_bbox_too_narrow`: monkey-patch render to return a surface with non-zero pixels but bbox width < `EMOJI_PROBE_MIN_BBOX_WIDTH`; assert every entry is `False`.
- `test_no_fallback_when_glyph_renders_normally`: monkey-patch render to return a surface with bbox width ≥ threshold AND non-bg pixel count ≥ threshold; assert every entry is `True`.
- `test_fallback_decision_cached_per_resource`: instrument `pygame.font.Font.render`; reload the module; assert render was called exactly N times (N = number of distinct resources probed), not N×K (K = number of draw calls later in the test).

### 8.3 `tests/integration/test_render_full_board_smoke.py` (new)

- `test_render_full_board_end_to_end`: boot a `catanGame(render_mode=None)` with seed 2026, build all hex / port / vertex / robber data, call every primitive in the documented layer order. Assert no exceptions. Sample 30 random pixels and assert each is one of the expected color families (water, sand, hex resource colors, gold, P1, P2, gray robber).
- `test_perf_under_50ms`: time a single full-board render. Assert wall-clock ≤ 50 ms on M1 Pro CPU (skip on CI if `os.environ.get("CI")` since timing varies). **Load-bearing for Gate 3.**

### 8.4 Regression tests (pre-existing, must remain green)

- `tests/integration/test_labeling_smoke.py` — must remain green through all three commits.
- All 290 pre-existing unit tests + 195 labeling-tool tests = 485 baseline tests, all green.

### 8.5 Test-budget commentary

Targets the patterns most likely to bite:

- **Pixel-color brittleness under SDL_dummy**: §0.2 preflight calibrates tolerance; all per-primitive tests use it.
- **Unicode glyph variance across systems**: §3 fallback decision + `test_unicode_fallback.py` pins the cache logic; `test_render.py` forces the cache via monkey-patch when needed.
- **Layer-order regressions**: `test_render_full_board_no_exceptions` + the docstring-pinned layer order in `render.py`. If a future change reorders the layers (e.g., ports drawn before island), the smoke test catches obvious overlap bugs.
- **Refactor blast radius on `view.py`**: §0.3 preflight pin + the §F.2 method-touch table + the manual launch at §10 STOP/RESUME after commit 3.
- **Perf regression**: `test_perf_under_50ms` is the canary; Gate 3 reads it.

---

## 9. Acceptance gate

Compound gate. **All three sub-gates pass or the upgrade is not promoted.** Until promotion, the existing pre-upgrade rendering remains in place.

### Gate 1 — Visual smoke (human spot-check)

After all three commits land, launch:
  - `python scripts/label_setup.py --seed 2026` for the labeling tool.
  - A live human-vs-AI game via the engine's `render_mode="human"` path for the live GUI.

The user (Benjamin) confirms both surfaces look "Catan-flavored and clearly readable" relative to the supplied Colonist reference screenshot. This is a subjective human gate — pass iff the user approves.

**Calibration**: visual quality cannot be pre-numerated. The Gate 1 owner is the user; the plan does not predict pass/fail probability. If the user rejects on subjective grounds, the §9 diagnosis ladder gives a tuning path.

### Gate 2 — All v2 tests green

- 485 pre-existing tests (290 baseline + 195 labeling-tool) green.
- ~20 new `test_render.py` tests green.
- ~4 new `test_unicode_fallback.py` tests green.
- 2 new `test_render_full_board_smoke.py` tests green.
- Total: ~511 tests green.
- Zero regressions in pre-existing tests.

### Gate 3 — Performance canary

A single full-board render under M1 Pro CPU completes in < 50 ms (the labeling tool runs at 30 FPS = 33 ms budget; 50 ms gives headroom for the rest of the UI loop).

**Measured by**: `tests/integration/test_render_full_board_smoke.py::test_perf_under_50ms` over 10 runs; report median. Pass iff median ≤ 50 ms.

**Fail diagnosis ladder**: see below.

### Diagnosis ladder when a gate fails

- **Gate 1 fails (visuals look wrong)**:
  - Identify the worst element from user feedback (e.g., "the ship sails look like blobs", "the pip dots are invisible at this zoom").
  - Cheap path: tune values in `render_constants.py` only. Re-launch + re-check.
  - Medium path: rework the offending primitive (≤ 30 LOC change in `render.py`).
  - Expensive path: rethink the primitive's shape (e.g., redesign the ship as a 2D icon instead of a polygon).
- **Gate 2 fails on new tests**: bug in primitives or in the constants. Read the failing assertion's pixel coordinates + expected RGB; fix the draw code or tune the constant.
- **Gate 2 fails on pre-existing tests**: regression introduced by the commit 2 or commit 3 refactor. Revert the refactor commit. Re-apply incrementally, running the full suite after each method's replacement.
- **Gate 3 fails (>50 ms / frame)**:
  - Profile with `cProfile` to identify the hot primitive.
  - Most likely culprits: (a) island outline (~1000-vertex polygon), (b) bevel shading (38 polygons per board: 19 tops + 19 bottoms), (c) unicode glyph rendering (one per hex).
  - Optimization order:
    1. **Cache the island outline as a Surface blit**. Render once at first call, store on a module-level cache keyed on `(hex_centers, canvas_size)`, blit on subsequent calls.
    2. **Pre-render emoji glyphs once at module init**. Already done by §3 cache, but extend to also cache the rendered Surface, not just the boolean decision.
    3. **Combine the per-hex bevel into a single Surface blit**. Render once per resource color × bevel state at module init.
  - Last resort: drop the bevel + island outline by flipping defaults in `render_constants.py` to `BEVEL_LIGHTNESS_DELTA = 0.0` and short-circuit `draw_island_outline` to a no-op. Re-test Gate 1 (visuals downgrade); user re-approves.

---

## 10. STOP/RESUME points

| Where | What to verify | Human decision |
|---|---|---|
| **Pre-commit-1** (after §0 preflight + before any code) | §0.1 emoji probe passed (or fallback-only mode chosen). §0.2 spot-check tolerance calibrated. §0.3 `view.py` method-touch table pinned. | **Approve commit-1 kickoff.** |
| **After commit 1 (`render.py` + `render_constants.py` + tests)** | All 20 render tests + 4 fallback tests green. Boot a smoke screen via an ad-hoc script (not committed) showing each primitive on seed 2026. Visuals look plausible. | **PASS** → approve commit 2. **FAIL on tests** → fix primitive bug. **FAIL on visuals** → tune constants. |
| **After commit 2 (`labeling/ui.py` refactor + smoke)** | All 195 labeling tests + 4 integration smoke tests still green. Launch `python scripts/label_setup.py --seed 2026` for the human spot-check. | **PASS** → approve commit 3. **FAIL on visuals** → tune constants. **FAIL on tests** → diagnose; if regression, revert the commit and re-apply incrementally. |
| **After commit 3 (`gui/view.py` refactor)** | All v2 tests green. Launch the engine's `render_mode="human"` path manually to confirm the live game GUI works (drive a few turns, build a settlement, end a turn). | **PASS** → finalize. **FAIL on tests** → split commit 3 per the §11 risk register into 3a/3b. **FAIL on visuals** → tune constants OR roll back the offending primitive. |
| **Final acceptance** | Gates 1, 2, 3 all pass per §6. | **Approve** → mark this plan COMPLETE; document the upgrade delta in `MEMORY.md` (one line). |

---

## 11. Risk register

| Risk | Likelihood | Severity | Mitigation |
|---|---|---|---|
| Unicode glyph absence on user's system → blank squares for resources | Low (macOS has emoji; §0.1 probes) | Medium | §0.1 preflight; §3 fallback decision cache; `test_unicode_fallback.py` pins the decision logic. If §0.1 fails, plan shifts to fallback-only mode and the §3 decision tree degrades gracefully. |
| `gui/view.py` refactor breaks `engine/game.py`'s human-vs-AI startup | Medium | High | §0.3 preflight pins the method-touch table. §5 commits to preserving public method signatures on `catanGameView`. §10 STOP/RESUME requires a manual launch at commit 3. Manual launch test at Gate 1. |
| Pixel-color spot-check tests brittle under anti-aliasing | Medium | Low | §0.2 preflight calibrates `PIXEL_SPOT_CHECK_TOL`. Tolerance windows (`abs(r - 35) ≤ TOL`) instead of exact match. Run tests on the same SDL driver every time (`SDL_VIDEODRIVER=dummy`). |
| Port ship doesn't read on sand background (color clash) | Low | Medium | Ship hull gets a dark outline (HULL_COLOR); sail color uses high contrast. Test pin: `test_draw_port_ship_renders_hull_mast_sail` asserts ≥ 50-RGB-unit separation from the sand background at the hull centroid. |
| Perf > 50 ms per frame on M1 Pro | Medium | Medium | Gate 3 measures. §9 diagnosis ladder optimizes in a documented order: cache island outline → pre-render emoji surfaces → combine bevel polygons. Last resort: disable bevel + island outline and re-run Gate 1. |
| Bevel shading looks ugly (cheap-3D effect, may look worse than flat) | Medium | Low | Toggle via `with_bevel=False` parameter on `draw_hex_tile`. User can disable per Gate 1 feedback without rolling back; `BEVEL_LIGHTNESS_DELTA = 0.0` in `render_constants.py` is a one-line override. |
| Drawn-polygon resource icons (fallback path) look unrecognisable | Medium | Medium | Design fallback icons once with extra care (3-triangle tree, sheep tuft as connected curves, 2×3 brick grid, 3 stacked ore quads, bundled wheat lines, cactus shape). User smoke at Gate 1 is the final check. If the §0.1 probe passes for all resources, this path is dormant in production. |
| Port label text doesn't fit on the sail polygon | Medium | Low | `test_draw_port_ship_label_fits_or_abbreviates` pins the abbreviation fallback: drop "BRICK" suffix; color the sail per port resource as the disambiguation. |
| Refactor of `gui/view.py` exceeds 500 LOC commit cap | Medium | Low | §0.3 preflight pins the method-touch table size upfront. If > 12 internal `_draw_*` methods are touched, split into commits 3a (island + water + tile bevel) and 3b (ports + vertex markers + robber). |
| Pygame Surface ordering causes draw primitives to overlap incorrectly | Low | Medium | Layer order documented in `render.py` module docstring: water → island → hex tiles (with bevel) → number tokens → resource symbols → port planks → port ships → vertex markers → prior picks → robber → top-bar overlay. `test_render_full_board_no_exceptions` catches gross order bugs. |
| `labeling/ui.py` refactor breaks the just-shipped tool | Low | High | All 78 labeling unit tests + 4 integration tests must stay green (Gate 2). §10 STOP/RESUME at commit 2 includes a manual launch + spot-check. The refactor scope is bounded to `_render_board`; the state machine and click handler don't change. |
| Constants file grows beyond 100 LOC | Low | Low | §2 hard cap. If exceeded, split into `render_colors.py` + `render_layout.py`. No effect on consumers. |
| Test count inflation slows CI | Low | Low | New tests use `SDL_VIDEODRIVER=dummy` which is fast (~5 ms per draw call). Total new test budget: ~24 unit tests × ~50 ms each + 2 integration tests × ~500 ms each = ~2 seconds added to suite wall-clock. Negligible. |
| Perf canary timing varies on CI | Medium | Low | `test_perf_under_50ms` skips on CI via `os.environ.get("CI")`. Local-only assertion. Gate 3 is run by Benjamin on M1 Pro at acceptance time. |
| Robber rendering on `view.py` was already present and now double-renders | Low | Low | §0.3 preflight's method-touch table identifies whether `_draw_robber` exists. If yes, replace; if no, add. No double-render path exists. |

---

## 12. Compute budget

**Engineering**: ≤ **4 days** hard cap.

| Phase | Days |
|---|---|
| Commit 1 — `render.py` + `render_constants.py` + 20 unit tests + 4 fallback tests | 1.5-2 |
| Commit 2 — `labeling/ui.py` refactor + smoke launch | 0.5 |
| Commit 3 — `gui/view.py` refactor + smoke launch | 1-1.5 |

**Runtime**: per Gate 3, < 50 ms per full-board render on M1 Pro CPU. No GPU. No new compute dependency.

**Test wall-clock impact**: ~2 seconds added to the full unit suite per §11 row "test count inflation". Negligible.

**Asset budget**: zero (§4 decision).

**Compute fallback** (`v2_step5_mcts.md` §7 pattern, restated for completeness): the upgrade is M1 Pro CPU-feasible end-to-end. No A100 fallback needed. The bottleneck is engineering time, not compute.

---

## 13. Carry-forward from project conventions

Decisions inherited from `CLAUDE.md` + `v2_step3_bc.md` + `v2_step4_ppo.md` + `v2_setup_labeling.md`:

- **TDD discipline**: tests-first per primitive. The §8 test plan is written before any commit code lands. Step-3 BC's tests-first surfaced two real bugs (silent action filtering, setup-phase `setup_step` inference); this plan inherits the same discipline.
- **Pre-commit green** (ruff + mypy) is a per-PR requirement on each of the three commits.
- **Commit size cap ~500 LOC**; the file layout in §14 naturally chunks into 3 commits within the cap. §0.3 preflight pins the `view.py` refactor scope to ensure commit 3 stays within the cap.
- **No `Co-Authored-By` AI trailers** in commits or PRs.
- **Branch convention**: `feat/colonist-style-render` (this work); follow-up commits use the same parent slug per CLAUDE.md §12. PR title: `feat: colonist-style render upgrade for setup labeling + live GUI`.
- **CLAUDE.md rule 8** (no `gui/` imports from RL paths): `gui/render.py` lives under `gui/` so the RL ban applies automatically. `labeling/` is outside the RL ring and is **explicitly allowed** to import `gui/` per the existing carve-out. **This plan does not change rule 8.**
- **No new .md docs unless asked** (CLAUDE.md global rule 14): this plan is the only new doc. `MEMORY.md` gets a one-line entry after Gate 1+2+3 pass; `README.md` is not touched until the user asks.
- **1v1 ruleset preservation** (CLAUDE.md project goal): the renderer is **read-only** w.r.t. the engine + ruleset. It uses `catanGame(render_mode=None|"human")`, `hex_tile.to_pixel(board.flat)`, `board.vertex_index_to_pixel_dict`, and `.port: str` attributes without modification. Zero changes to game logic, action space, obs schema, or engine state. The 6/8 red-number convention + pip-dot counts are visual conventions, not rule changes.
- **Backward compatibility** (CLAUDE.md §8): every public method on `catanGameView` keeps its name + signature. Every public method on the labeling-tool UI keeps its name + signature. The engine's `engine/game.py:64-66` construction of `catanGameView` is unchanged.
- **Documentation & schema sync** (global rule 14): no schema changes (the upgrade is visual-only). `MEMORY.md` gets a one-line entry on completion. `CLAUDE.md` is not touched (no convention change). No JSDoc/TSDoc adjustments (Python project; module docstring on `render.py` carries the layer-order documentation).

---

## 14. File layout (new + modified code)

```
src/catan_rl/gui/
├── render.py                                 NEW. ~250-350 LOC of pure draw functions.
├── render_constants.py                       NEW. ~50 LOC of color/font/size constants.
├── view.py                                   MODIFIED. Surgically refactored to use render.py.
└── __init__.py                               UNCHANGED.

src/catan_rl/labeling/
├── ui.py                                     MODIFIED. Inline rendering replaced with render.py calls.
├── scenario_gen.py                           UNCHANGED.
├── session.py                                UNCHANGED.
├── store.py                                  UNCHANGED.
├── archetypes.py                             UNCHANGED.
└── __init__.py                               UNCHANGED.

src/catan_rl/viz/
└── debug_board.py                            UNCHANGED. Out of scope per §Inputs.

src/catan_rl/engine/
└── game.py                                   UNCHANGED. Public catanGameView API preserved.

tests/unit/gui/                               NEW directory.
├── __init__.py                               NEW.
├── test_render.py                            NEW. ~20 per-primitive tests.
└── test_unicode_fallback.py                  NEW. ~4 fallback decision tests.

tests/integration/
├── test_labeling_smoke.py                    UNCHANGED (regression coverage).
├── test_render_full_board_smoke.py           NEW. End-to-end full-board render + perf canary.
└── (other integration tests)                 UNCHANGED.

docs/plans/
└── v2_colonist_render_upgrade.md             NEW. This plan doc.
```

---

## Provenance

- Visual reference: Colonist.io game-board screenshot provided by the user this session (2026-06-01). Canonical mental reference; not committed as an asset.
- First consumer of `render.py`: `docs/plans/v2_setup_labeling.md` (labeling tool, shipped this session). The labeling-tool refactor in commit 2 is the smoke-test surface for the new primitives.
- Existing legacy GUI being refactored: `src/catan_rl/gui/view.py` (761 LOC of legacy code).
- Engine integration point that must stay stable: `src/catan_rl/engine/game.py:64-66` (constructs `catanGameView(self.board, self)` when `render_mode="human"`).
- Out-of-scope consumer: `src/catan_rl/viz/debug_board.py` (debug-only index visualizer).
- Pre-existing 6/8-adjacency engine fix shipped this session: `src/catan_rl/engine/board.py:178-191`. Motivates the §A `draw_number_token` design (red text for 6/8).
- Same-number-adjacency engine fix shipped this session: same file, same line range.
- 1v1 ruleset reference (load-bearing for the read-only commitment): `docs/1v1_rules.md`.
- BC plan that established the tests-first discipline this upgrade inherits: `docs/plans/v2_step3_bc.md`.
- PPO plan that established the §0 preflight + §10 STOP/RESUME pattern: `docs/plans/v2_step4_ppo.md`.
- Setup-labeling plan that established the `labeling/` carve-out from CLAUDE.md rule 8 + the §A/§B/§C section layout this plan follows: `docs/plans/v2_setup_labeling.md`.

---

**WAITING FOR CONFIRMATION**
