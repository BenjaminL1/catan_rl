# v2 Setup Labeling — Interactive human-label tool for setup-phase BC fine-tune

**Status**: design draft (single author, not yet panel-reviewed); implementation gated on the §0 preflight gates and on Step 3 BC having converged with Gates 1+2+3 green.

**Revision history**:
- 2026-06-01 — Original draft. Standalone interactive labeling tool that simulates 1v1 Colonist.io setup phases, captures the user's settlement-vertex + road-edge picks per snake-draft position, persists to JSONL, and exposes a converter producing `bc/loader.py`-compatible NPZ shards for a setup-only BC fine-tune of the v2 anchor.

**Preflight gate** (per `v2_design.md` §0 + carry-forward from Step 3): the labeling tool's *scaffolding* (scenario generator, persistence, UI) can ship before Step 3 BC clears its gates — the labels live forever as JSONL and are useful with or without a fine-tune. The downstream **BC fine-tune integration (§F, `bc/finetune.py`)** does *not* start until:
  - Step 3 BC has converged with Gates 1 + 2 + 3 green (per `v2_step3_bc.md` §6). The BC checkpoint becomes the warm-start anchor for the fine-tune.
  - **Minimum yield gate**: ≥ 200 labeled scenarios in `data/labels/setup/v1/scenarios.jsonl`. Below this, the labels are used **validation-only** (held-out top-1 setup-agreement metric); BC fine-tune is not invoked. See §6 acceptance gate + §STOP/RESUME.

This doc is the planning equivalent of `v2_step3_bc.md` / `v2_step4_ppo.md` / `v2_step5_mcts.md` — it specifies what gets built, how it's tested, what numbers count as success, and where the risks are. The motivating problem and scope come from the (removed) v1 Monte-Carlo setup-pretrain plan (the v1 Monte-Carlo-only ancestor of this idea): the setup phase has high-SNR labels available cheaply if a strong human can supply them, and the v2 BC anchor benefits disproportionately from setup-phase signal because all subsequent gameplay is downstream of those four picks.

## Inputs

- v2 engine: `catanGame(render_mode=None)` from `src/catan_rl/engine/{game.py, board.py, player.py, dice.py}`. Headless constructor, used by both the scenario generator and the JSONL→NPZ converter.
- Obs encoder: `src/catan_rl/policy/obs_encoder.py` — the canonical 10-key obs producer that the BC fine-tune will consume. The converter re-runs this on each reconstructed state.
- Action-mask module: `src/catan_rl/env/masks.py:compute_action_masks(game, acting_player, env_state, vertex_to_idx, edge_to_idx)` — the standalone function the BC pipeline already uses. Reused at click-handling time (UI legality check) AND at NPZ conversion time (per-row mask field). This is the single source of truth for "what is a legal pick at this state"; the UI must not duplicate the logic.
- `BcDataset` interface from `src/catan_rl/bc/loader.py` — the 10 obs keys (`_OBS_KEYS_FLOAT` + `_OBS_KEYS_INT`) and 9 mask keys (`_MASK_KEYS`) the converter must emit, per `loader.py:94-115`.
- 1v1 Colonist.io ruleset (15 VP, no P2P trade, 9-card discard, Friendly Robber, StackedDice persistent Karma) — see `docs/1v1_rules.md`. The labeling tool is **read-only** w.r.t. these rules; it never overrides them.
- Snake draft order: P1 settle+road → P2 settle+road → P2 settle+road → P1 settle+road. Standard Colonist.io.
- v1 setup-pretrain ancestor (informational, not load-bearing): `/Users/benjaminli/my_projects/catan_rl/the removed v1 setup-pretrain plan` and `/Users/benjaminli/my_projects/catan_rl/src/catan_rl/rl/setup/`. The Monte-Carlo rollout heuristic is **not** ported — humans are the label source here, not MC.

## Outputs

- `data/labels/setup/v1/scenarios.jsonl` — append-only raw labels. One JSON object per line. **The durable artefact**; schema-versioned so future obs-schema changes do not invalidate the data.
- `data/labels/setup/v1/sessions/<uuid>/manifest.json` — per-session metadata (start/end time, labeler id, scenario count, archetype histogram). Per-scenario `state.pkl` checkpoints for resume-on-crash.
- `data/bc/human_setup_finetune/{manifest.json, shard_*.npz}` — converter output, `BcDataset`-compatible shards.
- `runs/bc/v1_human_finetune/<run_id>/{best.pt, scalars/}` — fine-tune outputs (TensorBoard scalars + best checkpoint).
- `runs/eval_setup_agreement/<run_id>/agreement.json` — held-out 20% val: BC top-1 match against the user's labels per snake-draft position.
- Acceptance criterion (§6) gates promoting the fine-tuned checkpoint to the Step-4 PPO anchor's BC warm-start.

---

## 0. Preflight gates (block labeling kickoff)

Two checks. Both pass or labeling does not start. This is the analogue of `v2_design.md` §0 for the labeling effort; it calibrates the plan against measurement, not rhetoric.

### 0.1 — Scenario-determinism probe

**Question**: given the same `(game_seed, draft_position, prior_picks)`, does `scenario_gen.py` produce a bit-for-bit identical board state and identical `(legal_settlement_corners, legal_road_edges)` masks? If not, the JSONL is unreplayable and the converter cannot reconstruct the obs the user actually saw.

**Method**: generate 100 scenarios with seed 42, draft_position uniformly drawn from {1,2,3,4}. Pickle each scenario. Regenerate the same 100 scenarios from the same seeds. Assert byte-identity on the pickled state and assert mask equality via `np.array_equal`.

**Decision rule**: pass iff all 100 byte-identical. **Fail**: audit `catanGame.__init__` for any uncontrolled RNG (`time.time()`, `os.urandom`, default-arg-mutable). The v2 engine's `catanGame(render_mode=None)` is intended to be deterministic given the seed argument; this gate proves it before we commit any labels.

### 0.2 — UI click → vertex/edge mapping pin

**Question**: does the UI's click handler map screen coordinates to the same vertex/edge IDs the engine uses? Off-by-one or coordinate-system mismatch silently produces wrong-label data and is the #1 historical bug for hex-board UIs.

**Method**: hand-write a fixture that lists all 54 vertex centroids and all 72 edge midpoints in the rendered board's pixel coordinates. For each, simulate a click at that pixel. Assert the UI returns the corresponding vertex/edge index. Then jitter each click by `±15 px` (smaller than the visual radius of a vertex) and assert the same ID is returned. **Catches**: coordinate origin (top-left vs center), Y-axis flip, hex-orientation (pointy-top vs flat-top), pixel scale.

**Decision rule**: pass iff all 54 + 72 = 126 centroid clicks map to the correct IDs AND all jittered clicks map to the same ID as the centroid. **Fail**: fix the coordinate transform; rerun.

---

## 1. UI tech decision

**Decision**: **pygame**. Locked.

**Rationale** (the user wants this shipped fast; the comparison was explicit):

| Option | Build cost | Visual quality | Labeling throughput target (1-2 min/scenario) | Click handling |
|---|---|---|---|---|
| ASCII / CLI | 0.5-1 day | Poor — text grid for a hex board makes "evaluating the layout" slow and error-prone | **Fails the throughput target** — typing vertex IDs from a printed list is slower than clicking | Type vertex ID; no spatial intuition |
| HTML + Flask + SVG/Canvas | 3-5 days | Best | Meets target | Mouse-click on SVG; well-supported |
| **Pygame** | **2-3 days** | **Acceptable** — existing `src/catan_rl/gui/` renderer can be adapted | **Meets target** | Mouse-click; existing renderer has the vertex/edge centroid math |

The decisive factor: `src/catan_rl/gui/` (v2-ported, currently used for the human-playable smoke) already has a working hex renderer with the 54-vertex / 72-edge pixel-coordinate math. Reusing that math via a thin adapter is ~1 day of work; rebuilding the same math in SVG/Canvas is ~2-3 days. Pygame wins on build cost without losing the visual quality required for the throughput target. The CLAUDE.md rule 8 ban on `gui/` imports applies only to RL paths (`bc/`, `policy/`, `env/`, `ppo/`, ...); `labeling/` is outside that ring and is *explicitly allowed* to import from `gui/`.

**Carry-forward**: if the user labels > 5 scenarios at consistently > 3 min each (the §0 throughput sanity), reassess. The most likely simplification at that point is dropping the archetype dropdown to a single-key shortcut (`b/o/h/r/x` for balanced/OWS/OWS-hybrid/road-builder/other).

---

## 2. Opponent's prior picks source

**Decision**: **(c) user labels both sides of every draft**. Locked.

**Rationale**:
- (a) heuristic prior picks: the heuristic's setup pip+diversity score is the v1 `heuristicAIPlayer`'s opening logic. It's the *baseline we're trying to beat with this data*. Training fine-tune labels conditioned on prior heuristic picks would teach the policy "given a heuristic opener, here is the human best-response" — which is the wrong distribution for self-play.
- (b) user-supplied "human-style opponent library": requires the user to author a stable set of opponent setup patterns. Real effort, plus introduces a second labeling decision (which pattern to use this round) before the labeling can even start.
- (c) user labels both sides: the user *is* the only top-level human player available. By labeling both sides, the captured data is top-vs-top play across the full snake draft. The pick 2 / pick 3 scenarios let the user demonstrate **denial logic against their own prior pick** — which is the strongest possible supervision signal for setup adversarial reasoning.

Side effect of (c): every "scenario" is one of the four snake-draft positions, and a full draft produces four scenarios. The user labels them in order — pick 1, pick 2 (denial against own pick 1), pick 3 (own pick 2 just-placed), pick 4 (own pick 3 just-placed). The session manager (§3.B) tracks this so the user never sees a pick out of order.

**Carry-forward to dataset semantics**: every row in `scenarios.jsonl` is from the perspective of one player at one draft position. The `acting_player` field (1 or 2) is recorded explicitly so the converter knows which obs to encode.

---

## 3. Storage format

**Decision**: **JSONL at the labeling layer, NPZ via a converter for BC training**. Locked.

**Rationale**:
- JSONL is human-readable, atomic-append-friendly (no schema migration at write time), trivially `git diff`-able for spot-checking, and survives schema drift (every row has `schema_version: int`).
- NPZ is what `bc/loader.py` consumes. Going JSONL → NPZ at conversion time means the obs schema can change later (Step 4 might add an obs key) and we re-run the converter without re-labeling.
- The cost of the indirection is one extra script (`scripts/convert_labels_to_bc_shard.py`); the benefit is decoupling label-time from training-time schema.

**JSONL row schema (v1)**:

```json
{
  "schema_version": 1,
  "scenario_id": "<uuid>",
  "session_id": "<uuid>",
  "labeled_at": "2026-06-01T15:30:00Z",
  "labeler_id": "<env LABELER_ID>",
  "game_seed": 12345,
  "draft_position": 1,                    // 1..4 in snake-draft order
  "acting_player": 1,                     // 1 or 2
  "prior_picks": [                        // empty for draft_position==1
    {"player": 1, "settlement_vertex": 17, "road_edge": 30}
  ],
  "archetype": "balanced",                // from labeling/archetypes.py enum
  "settlement_vertex": 23,                // user's pick
  "road_edge": 45,                        // user's pick
  "decision_time_ms": 47200,              // wall-clock between scenario shown and submit
  "notes": ""                             // optional, ≤ 200 chars
}
```

Schema versioning: bumping `schema_version` triggers a migration step in `labeling/store.py:load_scenarios`. v1 → v2 migrations are append-only field additions; field removals require a major-version bump and an explicit migration script.

---

## 4. Data directory layout

```
data/labels/setup/v1/
├── scenarios.jsonl                       (raw labels, append-only)
└── sessions/
    └── <session-uuid>/
        ├── manifest.json                 (start/end time, labeler, count, archetype histogram)
        └── inflight_state.pkl            (current scenario state, for crash-resume; cleared on session end)

data/bc/human_setup_finetune/
├── manifest.json                         (BcDataset-compatible)
└── shard_*.npz                           (256 MB shards per bc/dataset.py convention)

runs/bc/v1_human_finetune/<run_id>/
├── best.pt                               (fine-tune checkpoint)
└── scalars/                              (TensorBoard)

runs/eval_setup_agreement/<run_id>/
└── agreement.json                        (top-1 match metrics)
```

The `v1/` segment in the `data/labels/setup/v1/` path is the schema family, not a session version. If a major schema change ships, it becomes `data/labels/setup/v2/`; the old `v1/` directory stays valid and the converter dispatches by directory.

---

## 5. Mask reuse policy

**Single source of truth**: `catan_rl.env.masks.compute_action_masks(game, acting_player, env_state, vertex_to_idx, edge_to_idx)`. The UI calls this at click time; the converter calls this at NPZ-write time. They cannot diverge — both code paths import the same function.

**UI-side check**: when the user clicks a vertex, the UI calls `compute_action_masks(...)`, reads `corner_settlement` (during settle phase), and accepts the click iff the corresponding index is non-zero in the mask. Illegal clicks are rejected with a visual "denied" cue (red flash on the clicked vertex), not a popup; popups break flow at 1-2 min/scenario throughput.

**Converter-side check**: the NPZ row's per-row mask dict is exactly what `compute_action_masks` returned at that state. This means the BC fine-tune's loss-relevance buffer (`CatanActionHeads.head_relevance`) sees the same mask shape as the rest of the BC pipeline; no special-casing setup-phase rows downstream.

---

## 6. Optional justification field

**Decision**: include `notes: str` (default `""`, length-capped at 200 chars). UI does **not** require it (zero-friction default). Locked.

**Rationale**:
- The user has noted in past sessions that uncertain decisions are worth flagging for re-review. A free-text field at zero default-friction is the right shape.
- 200-char cap is a small enough limit that it never expands JSONL row size meaningfully (max ~250 bytes per row even with the note populated).
- The cap also caps the implied review cost: a 200-char note is ~30 seconds to read on later spot-check; longer would invite labeling-style essays the user will not actually re-read.

---

## A. Scenario generation (`src/catan_rl/labeling/scenario_gen.py`)

**Surface**:

```python
def generate_scenarios(
    n_scenarios: int,
    seed: int,
) -> Iterator[Scenario]:
    """Yield n_scenarios snake-draft scenarios, deterministic given seed."""

@dataclass(frozen=True)
class Scenario:
    scenario_id: str           # uuid
    game_seed: int             # the catanGame seed; reproduces board state
    board_state: bytes         # pickled catanGame at the moment of decision
    draft_position: int        # 1..4
    acting_player: int         # 1 or 2
    prior_picks: list[Pick]    # length == draft_position - 1
    legal_settlement_corners: np.ndarray  # bool (54,)
    legal_road_edges: np.ndarray          # bool (72,) — after settle decided
```

**Behavior**:
- One `catanGame` instance per scenario seed. Walk the snake draft (P1→P2→P2→P1) advancing the engine state at each step. At each user-decision point, yield a `Scenario` capturing the *current* state.
- `legal_settlement_corners` comes from `compute_action_masks(...)`'s `corner_settlement` field.
- `legal_road_edges` is computed *after* the user picks the settlement — the converter and the UI both re-call `compute_action_masks(...)` with the post-settlement state. The yielded scenario carries the pre-settlement mask only; the post-settlement edge mask is computed at click time.
- Determinism invariant: `list(generate_scenarios(N, seed=k)) == list(generate_scenarios(N, seed=k))` byte-for-byte. Pinned by `tests/unit/labeling/test_scenario_gen.py::test_determinism`.

**Implementation note**: snake-draft state-advancement happens by *replaying* the prior picks back through the engine via `player.build_settlement(vertex)` + `player.build_road(edge)` — the same path the BC dataset uses, NOT a special "setup-only" code path. This guarantees the obs the user sees at pick 2 is identical to the obs the policy will see at deployment.

---

## B. Session manager (`src/catan_rl/labeling/session.py`)

**Surface**:

```python
class LabelingSession:
    def __init__(self, session_dir: Path, labeler_id: str) -> None: ...
    def start(self) -> None: ...
    def current_scenario(self) -> Scenario | None: ...
    def submit(self, settlement_vertex: int, road_edge: int,
               archetype: Archetype, notes: str = "",
               decision_time_ms: int = 0) -> None: ...
    def skip(self) -> None: ...
    def quit(self) -> None: ...
    def resume(self) -> bool:
        """If session_dir has an inflight scenario, restore it. Return True if resumed."""
```

**Behavior**:
- On `start()`: writes `manifest.json` with `session_id, start_time, labeler_id, scenarios_completed=0`. Lazily generates scenarios via `scenario_gen.generate_scenarios(seed=<derived from session_id>)` — no upper bound; the generator yields indefinitely.
- On `submit(...)`: writes the scenario row to `scenarios.jsonl` via `store.append_scenario` (atomic), bumps `scenarios_completed`, clears `inflight_state.pkl`, advances to next scenario.
- On `skip()`: same as submit but writes nothing to JSONL (skipped scenarios are not labeled, not failures).
- On `quit()`: writes `end_time` to manifest, removes `inflight_state.pkl`, returns. The session ends cleanly.
- On `resume()`: if `inflight_state.pkl` exists, unpickles, restores current scenario index. Used for crash recovery — the user re-launches and sees the same scenario they were on.

**Session sizing**: **unlimited; session runs until the user explicitly quits.** The user has indicated they label faster than the 1-2 min/scenario design target and wants self-paced sessions. The UI displays scenario count + session elapsed time so the user can self-monitor; there is no hard cap. Configurable via `configs/labeling.yaml` if a per-session cap is ever wanted (default: no cap).

**Labeler identity**: `LABELER_ID` env var, default `"unknown"`. Recorded per scenario so multi-labeler datasets can be sliced later (out of scope for v1 — single user).

---

## C. UI layer (`src/catan_rl/labeling/ui.py`)

**Pygame implementation**. Layout:

```
+------------------------------------------------------------+
|  Scenario #X     |  Pick 2 of 4  |  Session timer: 04:31  |  Top bar
+------------------------------------------------------------+
|                                                            |
|              [Hex board, ~600x600 px]                     |  Center
|              Legal vertices: solid green dots             |
|              Illegal vertices: faint gray                 |
|              Prior picks: solid blue (P1) / red (P2)      |
|                                                            |
+------------------------------------------------------------+
|  Archetype: [▼ balanced]  Notes: [_________________]     |  Bottom bar
|  [Submit (S)]  [Skip (K)]  [Undo (U)]  [Quit (Q)]        |
+------------------------------------------------------------+
```

**State machine per scenario**:

```
SHOW_SETTLEMENT_PICK
  ↓ (click on legal vertex)
SETTLEMENT_PICKED → recompute edge mask via compute_action_masks(...)
  ↓
SHOW_ROAD_PICK (settlement highlighted, adjacent legal edges highlighted)
  ↓ (click on legal edge)
ROAD_PICKED
  ↓ (user clicks Submit OR presses S)
SUBMITTED → write to JSONL, advance scenario
```

**Interactions**:
- Click on illegal vertex/edge: red flash, no state change. The mask is recomputed via the standalone module — no UI-side legality logic.
- Undo (U): revert from `SETTLEMENT_PICKED` to `SHOW_SETTLEMENT_PICK`, or from `ROAD_PICKED` to `SHOW_ROAD_PICK`. Cannot undo a submitted scenario (use Skip on the next one if it was a mistake).
- Submit (S): only enabled in `ROAD_PICKED`. Calls `session.submit(...)`.
- Skip (K): always enabled. Calls `session.skip()`.
- Quit (Q): always enabled. Calls `session.quit()` and exits.

**Quality flag**: if `decision_time_ms < 15000` (15 seconds), the row carries `quality_flag = "fast"` in the JSONL. Spot-checking these rows later catches labeling fatigue. Not a rejection — the user might legitimately recognize a position quickly.

**Existing gui/ reuse**: import `src/catan_rl/gui/board_renderer.py:draw_board` for the hex tile + vertex + edge rendering. Wrap with the labeling-specific overlay (legal/illegal highlight, prior-pick coloring). The renderer's vertex/edge pixel-centroid tables are the same ones the click handler maps against — single source of truth for coordinates.

---

## D. Persistence layer (`src/catan_rl/labeling/store.py`)

**Surface**:

```python
def append_scenario(scenario: dict, path: Path) -> None:
    """Atomic append. Write to temp file, fsync, os.replace."""

def load_scenarios(path: Path) -> list[dict]:
    """Read all rows. Migrate older schema_versions to current."""

def count_scenarios(path: Path) -> int:
    """Fast row count for progress display; does not parse JSON."""

def repair_jsonl(path: Path) -> int:
    """Detect + remove malformed trailing line (crash recovery).
    Returns number of bytes truncated."""
```

**Atomicity** (`append_scenario`):
1. Read existing file size (`os.stat(path).st_size`).
2. Open in append mode (`"ab"`).
3. Serialize the scenario dict to a single line ending in `\n`.
4. `write(...)` + `flush()` + `os.fsync(...)`.
5. Close.

Using append mode + a single `write()` for a < 1 KB row is atomic on POSIX (`write(2)` to an `O_APPEND` fd is atomic for writes ≤ PIPE_BUF, and ext4/HFS+/APFS honor this for small writes). Tested under simulated SIGKILL in `tests/unit/labeling/test_store.py::test_atomic_append_under_sigkill`. For paranoia, `repair_jsonl(...)` is called on every session start: if the last line doesn't parse as JSON, it's truncated.

**Schema migration**: `load_scenarios` reads `schema_version` per row. v1 is the current target. Future migrations (e.g., v1→v2 adds `belief_target_at_pick`) populate missing fields with defaults at read time. **The on-disk file is never rewritten by the migration** — JSONL stays in its as-written state forever, migrations happen in memory at read time. This is non-negotiable: the raw labels are the durable artefact.

---

## E. JSONL → NPZ converter (`scripts/convert_labels_to_bc_shard.py`)

**Surface**:

```bash
python scripts/convert_labels_to_bc_shard.py \
  --jsonl data/labels/setup/v1/scenarios.jsonl \
  --output-dir data/bc/human_setup_finetune/ \
  --shard-size 5000 \
  --seed 0
```

**Behavior**:
1. Read every JSONL row via `store.load_scenarios`.
2. For each row:
   - Reconstruct `catanGame(seed=row["game_seed"])`.
   - Replay `row["prior_picks"]` through the engine (snake-draft order).
   - At the user's decision point, run `obs_encoder.compute_obs(...)` on the current state → 10-key obs dict.
   - Build the 9-key mask dict via `compute_action_masks(...)` (must match what the UI showed at label time; pinned by `tests/integration/test_labels_to_npz_pipeline.py::test_mask_roundtrip`).
   - Build the 6-head action tensor: `[type=BuildSettlement, corner=row["settlement_vertex"], edge=irrelevant, tile=irrelevant, res1=irrelevant, res2=irrelevant]` for the settlement step; a *separate* row for the road step: `[type=BuildRoad, corner=irrelevant, edge=row["road_edge"], ...]`. **Two NPZ rows per JSONL scenario** — the BC loss is relevance-weighted so this is the natural shape.
   - Set `belief_target` to the env's GT at the decision state (typically uniform-zero at setup phase since no dev cards drawn yet — the obs encoder handles this).
   - Set `z_disc = 0.0` (no game-outcome label; the value-loss weight zeros out per §F.2).
3. Pack rows into NPZ shards of `--shard-size` rows each (default 5000 — smaller than the BC pipeline's 5000-games-per-shard since human labels are precious).
4. Write `manifest.json` matching `bc/dataset.py:generate_dataset`'s output format so `BcDataset(data_dir=...)` loads it identically.

**Key invariant**: the obs emitted by the converter MUST equal what `obs_encoder.py` emits on the reconstructed state at the moment of decision. This is the load-bearing claim of the entire pipeline — if it fails, the BC fine-tune trains on out-of-distribution obs. Pinned by `tests/integration/test_labels_to_npz_pipeline.py::test_per_row_obs_equals_encoder_output`.

**Determinism**: the converter must produce bit-identical shards across runs given the same JSONL + seed. Pinned by `test_labels_to_npz_pipeline.py::test_converter_determinism`.

---

## F. BC fine-tune integration (`src/catan_rl/bc/finetune.py`)

**Surface**:

```python
def finetune(
    base_checkpoint: Path,                    # path to checkpoints/bc/best.pt
    heuristic_dataset_dir: Path,              # data/bc/v1/
    human_dataset_dir: Path,                  # data/bc/human_setup_finetune/
    human_weight: float = 50.0,               # WeightedConcatDataset weight
    n_steps: int = 5000,
    val_split: float = 0.20,
    output_dir: Path = Path("runs/bc/v1_human_finetune"),
    seed: int = 0,
) -> Path:
    """Returns path to best.pt."""
```

### F.1 Dataset mixing

`WeightedConcatDataset(datasets=[heuristic_ds, human_ds], weights=[1.0, 50.0])`. The 50× weight is calibrated to roughly match per-batch sample frequency between the two datasets: with ~30k heuristic games × ~50 (s,a) per game = 1.5M heuristic samples and ~500 human-setup samples, a 50× weight on the human side makes them appear ~1/60 of batches instead of ~1/3000. Configurable in `configs/bc_human_finetune.yaml`.

**Test**: `tests/unit/bc/test_finetune_dataset.py::test_human_sample_frequency` — given mock datasets of size 1500 and 500 with weights 1.0 and 50.0, draw 10k batches of size 1024, assert human-sample fraction is within ±5% of the analytic prediction `(500*50) / (1500*1 + 500*50) ≈ 94.3%` (high — the weight is intentionally strong; the heuristic is the majority opener but human labels are the focus).

### F.2 Loss

Same as `bc/loss.py` — relevance-weighted CE per head, value MSE @ 0.1, belief soft-CE @ 0.05. **For human-label rows**, the value-loss weight is multiplied by 0 (the `z_disc` field is 0 by construction because we don't have a game outcome). This zero-multiplier is implemented via a per-row `value_loss_mask` boolean in the batch dict; the loss is `mse * value_loss_mask.float()`. No special-casing in the loss code path; the mask handles it.

**Test**: `tests/unit/bc/test_finetune_loss.py::test_value_loss_zero_for_human_rows` — construct a batch with 50% heuristic + 50% human; assert the value-loss term equals the value-loss term computed on the heuristic-only half (the human half contributes exactly zero).

### F.3 Optimizer + schedule

| Param | Value | Rationale |
|---|---|---|
| Optimizer | AdamW (β=(0.9, 0.999), eps=1e-5, weight_decay=1e-4) | Match Step 3 BC |
| Peak LR | **5e-5** (lower than Step 3's 3e-4) | Fine-tune — don't blow away the BC anchor |
| LR schedule | Constant after 200-step warmup | Match Step 3 spirit; shorter warmup since the model is already trained |
| Batch size | 1024 | Match Step 3 |
| Max steps | 5000 | Roughly 5 epochs over the human + heuristic mix at default weights |
| Early stop | Held-out NLL on **human-only val split**, patience 3 evals (every 200 steps) | The whole point is human-setup agreement; gate the early-stop on that |
| Grad clip | max_norm=1.0 | Standard |

### F.4 Validation

20% of the human labels (stratified by `(draft_position, archetype)`) are held out. The training loop logs every 200 steps:
- `finetune/val_nll_human_only` — NLL on the human-only val split.
- `finetune/val_top1_settlement` — top-1 corner-head match for setup-phase rows. **This is the load-bearing metric** for §6 Gate 1.
- `finetune/val_top1_road` — top-1 edge-head match.
- `finetune/val_nll_heuristic_only` — NLL on a heuristic-only val sample (canary: the model should not regress on heuristic gameplay).

---

## 7. File layout (new code)

```
src/catan_rl/labeling/             (new package; allowed to import gui/)
├── __init__.py
├── scenario_gen.py                Per §A. Generator yielding scenarios from a board+seed.
├── session.py                     Per §B. Session manager + manifest + resume.
├── store.py                       Per §D. JSONL persistence + atomic append + repair.
├── ui.py                          Per §C. Pygame UI layer.
└── archetypes.py                  Enum: {balanced, OWS, OWS_hybrid, road_builder, other}.

src/catan_rl/bc/
└── finetune.py                    Per §F. BC fine-tune (~150 LOC).

scripts/
├── label_setup.py                 CLI entry: `python scripts/label_setup.py [--session-size 20]`.
├── convert_labels_to_bc_shard.py  Per §E. JSONL → NPZ.
└── eval_setup_agreement.py        Held-out 20%: BC vs human top-1 agreement per draft position.

configs/
├── labeling.yaml                  Defaults: data dir, archetype list, session size, time-flag threshold.
└── bc_human_finetune.yaml         BC fine-tune config (extends bc.yaml).

tests/
├── unit/labeling/
│   ├── test_scenario_gen.py       Per §8.1.
│   ├── test_session.py            Per §8.2.
│   ├── test_store.py              Per §8.3.
│   ├── test_ui.py                 Per §8.4 (SDL_VIDEODRIVER=dummy).
│   └── test_archetypes.py         Enum sanity.
├── unit/bc/
│   ├── test_finetune_dataset.py   §F.1 WeightedConcatDataset frequency.
│   └── test_finetune_loss.py      §F.2 value-loss masking.
└── integration/
    ├── test_labeling_smoke.py     End-to-end with mocked user input.
    ├── test_labels_to_npz_pipeline.py    JSONL → NPZ → BcDataset.
    └── test_bc_finetune_smoke.py  Tiny ckpt + 50 human samples → 100 fine-tune steps.

data/labels/setup/v1/scenarios.jsonl       (created at runtime)
data/labels/setup/v1/sessions/<uuid>/      (created per session)
data/bc/human_setup_finetune/              (generated by converter)
runs/bc/v1_human_finetune/                 (BC fine-tune outputs)
```

---

## 8. Testing (TDD discipline, tests-first per Step 3 + Step 4 convention)

Tests are written **before** implementation, per the user's established preference. The patterns below target the failure modes that bit `bc/dataset.py` (silent action filtering) and the historical hex-board UI bugs.

### 8.1 `tests/unit/labeling/test_scenario_gen.py` (~10 tests)

- `test_determinism_per_seed`: byte-identity across two runs with the same seed (the §0.1 preflight, also pinned as a unit test).
- `test_pick1_has_empty_prior_picks`: draft_position == 1 implies `prior_picks == []`.
- `test_pick2_has_one_prior_pick`: draft_position == 2 implies `len(prior_picks) == 1` with the P1 settle+road.
- `test_pick3_has_two_prior_picks`: P1's pick + P2's first pick.
- `test_pick4_has_three_prior_picks`: full chain.
- `test_legal_mask_nonempty`: every yielded scenario has at least one legal corner.
- `test_legal_mask_matches_compute_action_masks`: exact equality vs `compute_action_masks(...)`.
- `test_acting_player_matches_snake_draft`: at positions 1+4 acting_player==1, at 2+3 acting_player==2.
- `test_prior_picks_replayable`: applying prior_picks to a fresh engine reproduces the yielded board_state.
- `test_scenario_id_unique`: 1000 scenarios have 1000 distinct UUIDs.

### 8.2 `tests/unit/labeling/test_session.py` (~6 tests)

- `test_manifest_fields_present`: session_id, start_time, labeler_id, scenarios_completed.
- `test_submit_appends_jsonl`: after `submit(...)`, scenarios.jsonl has +1 row.
- `test_resume_restores_scenario_index`: simulate quit-mid-scenario, resume, verify the same scenario is shown.
- `test_skip_does_not_append`: `skip()` advances scenarios_completed but does not write to JSONL.
- `test_quit_finalizes_manifest`: manifest gets end_time populated.
- `test_concurrent_append_does_not_corrupt`: two sessions writing concurrently to the same JSONL (pathological — single-user app, but worth pinning). Use file-locking via `fcntl.flock` if needed.

### 8.3 `tests/unit/labeling/test_store.py` (~5 tests)

- `test_round_trip`: write a scenario, read it back, assert byte-equality after `json.loads`.
- `test_schema_version_field_present`: every written row has `schema_version` populated.
- `test_atomic_append_under_sigkill`: spawn a subprocess that writes 100 rows, SIGKILL it mid-write, parent reads back: every line that exists is valid JSON.
- `test_repair_jsonl_truncates_malformed_trailing_line`: pre-corrupt last line, `repair_jsonl` removes it, file remains parseable.
- `test_count_scenarios_no_full_parse`: 10k rows; `count_scenarios` returns in < 100 ms (use `wc -l`-equivalent).

### 8.4 `tests/unit/labeling/test_ui.py` (~4 tests, headless via `SDL_VIDEODRIVER=dummy`)

- `test_click_on_legal_vertex_registers`: simulate click on a known legal centroid, assert UI transitions to `SETTLEMENT_PICKED` state.
- `test_click_on_illegal_vertex_rejected`: simulate click on a non-legal vertex centroid, assert state unchanged + visual flash flag set.
- `test_undo_reverts_to_show_settlement_pick`: from `ROAD_PICKED`, press U, assert state == `SHOW_SETTLEMENT_PICK` (note: full undo from `ROAD_PICKED` goes back two states — verified explicitly).
- `test_submit_appends_scenario`: simulate full pick + submit, verify `session.submit(...)` was called with the expected args.

### 8.5 `tests/integration/test_labeling_smoke.py`

End-to-end: spawn a `LabelingSession`, mock `pygame.event.get` to emit a hardcoded sequence of 5 (vertex_click, edge_click, submit) tuples. After the sequence, assert `scenarios.jsonl` has 5 well-formed rows matching the expected (vertex, edge) per scenario.

### 8.6 `tests/integration/test_labels_to_npz_pipeline.py`

- `test_per_row_obs_equals_encoder_output`: write 10 JSONL rows, run converter, load via `BcDataset`, for each row assert `dataset[i]["obs"]` equals `obs_encoder.compute_obs(reconstructed_state)`. **The load-bearing test of the entire pipeline.**
- `test_mask_roundtrip`: same setup, assert `dataset[i]["mask"]` equals `compute_action_masks(reconstructed_state)`.
- `test_converter_determinism`: run converter twice on the same JSONL, assert NPZ shards are byte-identical.
- `test_two_rows_per_scenario`: 5 JSONL rows produce 10 NPZ rows (one settle + one road per scenario).

### 8.7 `tests/integration/test_bc_finetune_smoke.py`

- Load a tiny BC checkpoint (`tests/fixtures/bc_tiny.pt` — 4-layer scaled-down policy, ~10k params).
- Generate 50 mock human-label samples + 100 mock heuristic samples.
- Fine-tune for 100 steps.
- Assert: (a) `val_nll_human_only` decreases by ≥ 20% over the 100 steps, (b) the heuristic-canary `val_nll_heuristic_only` does not increase by > 10% (no catastrophic forgetting), (c) wall-clock ≤ 2 min on M1 Pro CPU (perf canary, same convention as Step-4 §5 preamble).

### 8.8 Test-budget commentary

Targets the patterns that bit `bc/dataset.py` and the hex-board UI risk class:
- **Silent action-filtering after state transition**: `test_labels_to_npz_pipeline.py::test_per_row_obs_equals_encoder_output` — if the converter caches stale env_state, this fails.
- **Click coordinate mismatch**: §0.2 preflight + `test_ui.py::test_click_on_legal_vertex_registers` (the 126-centroid fixture is the durable form).
- **JSONL corruption on crash**: `test_store.py::test_atomic_append_under_sigkill` + `test_repair_jsonl_truncates_malformed_trailing_line`.
- **Schema drift across labeling and training**: schema_version field + read-time migration; pinned by `test_store.py::test_schema_version_field_present`.

---

## 9. Acceptance gate

Compound gate. **The fine-tuned BC checkpoint is promoted to the Step-4 anchor warm-start iff all three sub-gates pass.** Until promotion, the existing pure-heuristic BC checkpoint remains the Step-4 anchor.

### Gate 1 — Setup-agreement with human labels (≥ 30% top-1)

On the 20% held-out validation split of `data/labels/setup/v1/scenarios.jsonl`, the fine-tuned BC's `corner_settlement` head must achieve ≥ **30%** top-1 match against the user's chosen settlement vertex at the held-out scenarios.

Calibration: the pre-fine-tune BC (heuristic-only) is expected to score ~5-15% on this metric (the heuristic and the user share some priors but not many). 30% is **substantially higher** than the pre-fine-tune baseline AND ≥ 6× chance (chance = 1/avg_legal_count ≈ 1/30 ≈ 3.3%). Both halves matter: clearing chance + clearing the heuristic baseline = "the fine-tune is learning something specifically from the human data."

**If the pre-fine-tune baseline measures > 25%**, the 30% gate is too lax; revise upward to `baseline + 0.10` at the next plan revision. This is recorded as a measurement-driven adjustment, not a guess; see §6 of `v2_step3_bc.md` for the same pattern applied to NLL gates.

### Gate 2 — Heuristic WR not regressed by > 5pp

Symmetrised WR vs the heuristic over 200 games per seat × N=3 seeds. **Pass**: WR ≥ `WR_BC_pre_finetune - 0.05`. The fine-tune touches the setup heads but the model is shared with gameplay heads via the encoder; we verify no catastrophic forgetting of gameplay competence.

If WR regresses by > 5pp: **diagnosis**: (a) `human_weight=50.0` over-amplified the human signal; lower to 10-20× and retry. (b) The fine-tune ran too long; reduce `n_steps`. (c) The pre-existing BC anchor was already fragile; this is a Step-3 issue surfaced by the fine-tune and goes back to Step 3.

### Gate 3 — WR vs heuristic ≥ pre-fine-tune BC's WR

Symmetrised. Same 200×3 setup as Gate 2. **Pass**: `WR_BC_finetuned ≥ WR_BC_pre_finetune` (no minimum margin — equivalence is acceptable since the fine-tune's primary metric is Gate 1).

This is a **floor**, not a ceiling. The fine-tune doesn't need to win more games — it needs to play better setups. Subsequent PPO will exploit the better setups.

### Diagnosis ladder when a gate fails

- **Gate 1 fails (< 30% top-1)**: either the human dataset is too small (< 200 scenarios — minimum-yield gate fails), the `human_weight` is too low (raise from 50 → 100 and retry), OR the user's labels are inconsistent (audit via §STOP/RESUME spot-check at 100 scenarios).
- **Gate 2 fails (heuristic WR regressed > 5pp)**: as above (over-amplified human signal or over-trained).
- **Gate 3 fails (WR collapsed)**: the fine-tune broke gameplay heads via shared-encoder updates. Switch to a **head-only fine-tune** (freeze the encoder, train only the `corner_settlement` + `edge` heads) for the next pass.

---

## 10. Compute budget

**Labeling**: user-paced; throughput is determined by the labeler, not the plan. Earlier 1.5 min/scenario estimate was conservative — the labeler has indicated faster pace is achievable. Distribute across however many sessions the user finds comfortable; STOP/RESUME gates at 50, 100, 200, 500 scenarios drive when the next decision happens, not session count. For reference: at 30 sec/scenario, 200 scenarios = ~1.7 hours; 500 scenarios = ~4.2 hours.

**Engineering**: ≤ **5 days** hard cap (per §11 risk register: UI complexity creep is the biggest engineering risk).

| Phase | Days |
|---|---|
| Scaffolding (scenario_gen, store, session, archetypes, configs) | 1 |
| Pygame UI (renderer adapter + click handler + state machine) | 2 |
| Converter (JSONL → NPZ) | 0.5 |
| BC fine-tune integration | 1 |
| Tests (unit + integration; tests-first) | 0.5 (interleaved) |

**BC fine-tune compute**: 5000 steps × 1024 batch × ~80 ms/step = ~7 minutes on M1 Pro CPU. Trivial compared to Step 3 or Step 4. Per-seed cost is negligible; run N=3 seeds for the §9 acceptance gate.

**Eval (Gate 2 + Gate 3)**: 200 games × 3 seeds × 2 seats × ~3 sec/game = ~1 hour. Run after each fine-tune.

---

## 11. Risk register

| Risk | Likelihood | Severity | Mitigation |
|---|---|---|---|
| Opponent prior-pick quality (if heuristic chosen) | **N/A — §2 picks user-labels-both-sides** | High | User-labels-both-sides is the chosen path; no heuristic opponent in labeled data. |
| UI complexity creep | Medium | Medium | Hard 5-day engineering cap. If pygame UI takes > 3 days for the click handler alone, simplify the archetype dropdown to a single-key shortcut. No multi-session, no auth, single-user. |
| Schema drift between labeling and BC training | High | Medium | Store `game_seed` + `prior_picks` so the obs is **regenerable** via the converter at any future obs-schema version. JSONL on-disk never rewritten by migrations. |
| User commitment threshold | High | High | §STOP/RESUME minimum-yield gate (200 scenarios) before BC fine-tune; below that, validation-only use. The data is durable regardless of final use. |
| Labeling fatigue → late-session data quality drops | Medium | Medium | Sessions are user-paced (no auto-cap); UI shows session timer + scenario count so the user can self-monitor. `quality_flag = "fast"` for sub-15-second decisions surfaces individual rows for spot-check audit. If fatigue is observed in spot-checks, the user can adopt their own per-session cadence; the plan does not enforce one. |
| Click-handler legal-mask drift from env masks | Low | High | Single source of truth: UI calls `compute_action_masks` at click time; cannot diverge. §0.2 preflight pins all 126 centroids. |
| JSONL append corruption on crash | Low | Medium | Atomic POSIX append for sub-PIPE_BUF rows + `repair_jsonl` on every session start. `test_store.py::test_atomic_append_under_sigkill` pinning. |
| User loses interest, abandons after 50 scenarios | High | Medium | Plan acceptable degraded use (validation-only at 50-100 scenarios); 200 is the full-integration gate. Labels durable regardless. |
| Per-scenario throughput slower than user expectation | Low | Low | §0 preflight throughput spike with 5 throwaway scenarios; if labeling feels slow, simplify archetype dropdown to single-key shortcut (`b/o/h/r/x`). UI quality flag surfaces individual fast-decision rows passively. |
| Pygame click coordinate mapping bugs | Medium | High | §0.2 preflight + 126-centroid pinning test. The `src/catan_rl/gui/` renderer's vertex/edge centroid tables are the single source of truth. |
| Fine-tune over-amplifies human signal (Gate 2 regression) | Medium | Medium | `human_weight=50.0` is configurable; diagnosis ladder § Gate 2 sweeps to 10-20×. Head-only fine-tune is the fallback if encoder updates are the issue. |
| User labels are inconsistent (e.g. preference shifts mid-labeling) | Medium | Medium | `quality_flag` + decision-time logging; spot-check at 100 scenarios; archetype field lets us slice by intent if a preference shift is identified. |
| Fine-tune catastrophically forgets gameplay (Gate 3 collapse) | Low | High | `WeightedConcatDataset` keeps heuristic samples in every batch; `n_steps=5000` is conservative; LR=5e-5 (1/6 of Step 3); head-only fine-tune is the fallback. |
| Converter produces obs that diverges from training-time obs | Low | High | `test_labels_to_npz_pipeline.py::test_per_row_obs_equals_encoder_output` is the load-bearing pin — converter and `obs_encoder.py` share code paths; any drift fails the test. |
| Setup-agreement metric is too easy to game | Medium | Low | Top-1 corner match is computed on a held-out 20% of scenarios the model never trained on; gaming requires memorizing the held-out split, which the train/val split prevents by construction (stratified by scenario_id). |
| §9 30% top-1 calibration is wrong | Medium | Low | If pre-fine-tune BC baseline measures > 25%, plan revises the gate upward to `baseline + 0.10`. Measurement-driven, not a fixed number. |

---

## 12. STOP/RESUME points

| Where | What to verify | Human decision |
|---|---|---|
| **Pre-labeling smoke** (after scaffolding lands) | Run 3-5 throwaway scenarios end-to-end. JSONL written correctly. No crashes. UI clicks land on correct vertices/edges (the §0.2 preflight asserts this; the smoke is the human-in-the-loop confirmation). | **Approve labeling kickoff.** |
| **After 50 labeled scenarios** | Hand-inspect 5 random rows. Check: recorded settlement+road match the user's intent? Archetype field sensibly populated? Decision times in the 30s-180s range? | OK → continue. NOT OK → fix bugs before more data is collected (the bad 50 may be salvageable or may need to be discarded; decide per-row). |
| **After 100 labeled scenarios** | Spot-check 10 rows manually for data quality + schema correctness. Run `convert_labels_to_bc_shard.py` end-to-end. Confirm `BcDataset(data_dir=...)` loads without exceptions. | OK → continue to 200. NOT OK → fix converter; possibly re-label. |
| **After 200 labeled scenarios** | Validation-only use is unlocked. The plan can commit to BC fine-tune integration. | Decide: continue to 500 (production fine-tune dataset) OR stop at 200 for validation-only use. |
| **After 500 labeled scenarios** | Run end-to-end BC fine-tune pilot. Measure pre-fine-tune vs post-fine-tune top-1 settlement match on the 20% val split. Run heuristic-WR eval (Gates 2 + 3). | **PASS Gate 1+2+3 → promote fine-tuned BC to Step-4 anchor warm-start.** **FAIL Gate 1 → diagnose per §9 ladder.** **FAIL Gate 2 or 3 → drop `human_weight` or head-only fine-tune; retry.** |
| **Final gate evaluation** | Gates 1, 2, 3 green with N=3 seeds | Approve promotion; archive fine-tune checkpoint; document setup-agreement delta in `MEMORY.md`. |

---

## 13. Carry-forward from project conventions

Decisions inherited from `CLAUDE.md` + `v2_step3_bc.md` + `v2_step4_ppo.md` + `v2_step5_mcts.md`:

- **TDD discipline**: tests-first per module (the Step-3 BC convention; Step 4 + Step 5 carried it forward). Step-3 BC's tests-first surfaced two real bugs (silent action filtering, setup-phase `setup_step` inference); this plan inherits the same risk class and the same discipline.
- **Pre-commit green** (ruff + mypy) is a per-PR requirement.
- **Commit size cap ~500 LOC** where possible; the file layout in §7 naturally chunks into 4-5 commits per the package boundaries.
- **No `Co-Authored-By` AI trailers** in commits or PRs.
- **Branch convention**: `feat/setup-labeling-tool` (this work); follow-up commits use the same parent slug per CLAUDE.md §12.
- **CLAUDE.md rule 8** (no `gui/` imports from RL paths): `labeling/` is outside the RL ring and is **explicitly allowed** to import `gui/`. The rule prohibits `bc/`, `policy/`, `env/`, `ppo/`, `selfplay/`, `eval/` from importing `gui/`; `labeling/` is in scope for the carve-out. This plan does not change rule 8 — it relies on the existing scope boundary.
- **No new .md docs unless asked** (CLAUDE.md global rule 14): this plan is the only new doc; `MEMORY.md` gets a one-line entry after promotion; `README.md` is not touched until the user asks.
- **1v1 ruleset preservation** (CLAUDE.md project goal): the labeling tool is **read-only** w.r.t. the engine and ruleset. It uses `catanGame(render_mode=None)` and `compute_action_masks(...)` without modification. Any future change to the engine that affects setup-phase legality (vertex adjacency, port placement, snake draft order) requires a schema_version bump in the JSONL and a re-validation pass on the converter.
- **Phase B compute fallback** (Step 5 §7 pattern): this plan is M1 Pro CPU-feasible end-to-end; no A100 fallback needed. The bottleneck is human labeling time, not compute.

---

## Provenance

- Base motivation: the (removed) v1 Monte-Carlo setup-pretrain plan (v1 Monte-Carlo-only ancestor of this idea). The v2 design swaps MC rollouts for human labels; the rest of the structure (setup-phase focus, encoder-shared head-targeted fine-tune, KL-regularised merge back to the live policy) carries forward in spirit.
- BC plan that this fine-tune layers on top of: `docs/plans/v2_step3_bc.md`. The fine-tune is gated on Step 3's Gates 1+2+3 green per §0.
- PPO plan that consumes the fine-tuned BC checkpoint: `docs/plans/v2_step4_ppo.md`. The fine-tuned `checkpoints/bc/best.pt` becomes the Step-4 piKL anchor + warm-start, replacing the heuristic-only BC checkpoint after promotion.
- MCTS plan that downstream consumes the Step-4 anchor: `docs/plans/v2_step5_mcts.md`. Not directly affected by this work, but the setup-pretrain improvement propagates through Step 4 → Step 5.
- 1v1 ruleset reference (load-bearing for all setup-phase invariants): `docs/1v1_rules.md`.
- Existing GUI renderer reused via thin adapter: `src/catan_rl/gui/board_renderer.py` (allowed import per CLAUDE.md rule 8 scope carve-out).
- Mask single source of truth: `src/catan_rl/env/masks.py`.
- Obs encoder single source of truth: `src/catan_rl/policy/obs_encoder.py`.
- BC dataset format target: `src/catan_rl/bc/loader.py:_OBS_KEYS_FLOAT/_OBS_KEYS_INT/_MASK_KEYS` (lines 94-115).
- v1 setup-pretrain artefacts (informational, not load-bearing): `/Users/benjaminli/my_projects/catan_rl/src/catan_rl/rl/setup/` and `/Users/benjaminli/my_projects/catan_rl/scripts/train_setup.py`. The Monte-Carlo rollout heuristic is **not** ported; humans are the label source here.

---

**WAITING FOR CONFIRMATION**
