# Spec: setup-labeling-and-champion-finetune — the owner's openings become the sole setup teacher

**Status: RATIFIED (2026-08-10) — BINDING. Owner ratified in full view of the open RISKS below (successor-anchor dependency; head-only-first alternative).**

**Feature in one line.** Verify and productionize the owner's setup-labeling pipeline (tool →
JSONL → converter → shards), fix the `play_vs_model` recorder to record *observed* setup steps,
and build a champion fine-tune that installs the owner's hand-labeled openings into
`runs/anchors/ptr_v1_u500.pt` without damaging its midgame.

## Provenance

Revives `docs/plans/v2/setup_labeling.md` (the owner's 2026-06-01 design). The scaffolding
already landed — `src/catan_rl/labeling/` (scenario_gen, session, store, ui, archetypes) with 74
green unit tests plus `scripts/label_setup.py` — but §E (converter) and §F (fine-tune) never
shipped, and §F's anti-forgetting strategy (heuristic-corpus mixing) is invalidated by the lean
opening program (adopted 2026-08-10): owner labels are the SOLE opening teacher, the fine-tune
target is the champion directly, and there is NO corpus regeneration. Owner directives carried
verbatim: heuristic openings must NOT influence policy openings; RL branches off from the
owner's openings. Interview decisions: anti-forgetting = self-distillation mix; scope = full
slice (tool smoke + recorder + converter + fine-tune code; the fine-tune RUN waits for labels).

---

## Decisions (binding)

### D1 — Tool verification, not rebuild
Run the §12 pre-labeling smoke (3–5 throwaway scenarios end-to-end via `scripts/label_setup.py`):
JSONL rows well-formed, clicks map to the correct vertex/edge, crash-resume restores the
in-flight scenario. Fix any pointer-arch drift surfaced; the existing 74-test suite stays green.
No redesign of the tool.

### D2 — Recorder records OBSERVED setup
`scripts/play_vs_model.py` records the four observed setup steps per seat **at placement time**
(state snapshotted then — the `longest_road_holder=None` hardcode and shared-`state_after`
shortcuts must not survive in disguise), replacing post-hoc synthesis via
`setup_steps_seat_0/1`. Records carry `"setup_observed": true`; the SYNTHESIZED caveats in the
module doc and CLI help are updated. Old records stay distinguishable (absent key = synthesized).

### D3 — Records→labels adapter
A small adapter exports the owner's real-game setup decisions from `play_vs_model` records into
the same JSONL label store (`source: "game"` vs `"tool"`), so playtesting grows the corpus as a
side effect. The tool remains the volume path (a full game yields 2 owner setup decisions; the
tool yields ~1 per minute and controls denial-position coverage).

### D4 — Converter to the CURRENT schema
`scripts/convert_labels_to_bc_shard.py` per §E, updated: **12 obs keys** (10 float keys plus
`opponent_kind` / `opponent_policy_id` — see `bc/loader.py:97`; the plan's earlier "10 keys" was
wrong) and the **12 `MASK_KEYS`**; manifest stamped `FORCED_RULE_VERSION = 2` and
`RULESET_VERSION = 3` (the loader refuses anything else); obs reconstructed from
`game_seed + prior_picks` through the current `obs_encoder` at conversion time; two NPZ rows per
scenario (settlement, road). **Human rows are duplicated across every `opponent_kind` value used
by the gates and the successor self-play** — the policy conditions on an opponent-id embedding,
so a single-kind stamp can leave the learned openings unexpressed under the other kind.
Converter determinism pinned by test.

### D5 — Champion fine-tune with self-distillation anchoring
`src/catan_rl/bc/finetune.py` loads `runs/anchors/ptr_v1_u500.pt` via
`catan_rl.checkpoint.load_checkpoint`. Anti-forgetting mechanism (owner-decided): anchor
non-setup behavior to the champion itself. Implementation MAY be offline (pre-generated
pseudo-label set carrying per-head distributions) or **online — KL(fine-tuned ∥ frozen u500)
computed on freshly sampled non-setup states during training; preferred, because it is the same
regularizer with zero new dataset infrastructure** (no per-head-distribution storage exists in
`bc/` today, and building it is the largest hidden cost in the offline form). **Anchor-state
coverage:** the states for the anchor term are sampled from games whose setups are FORCED to
openings drawn from the owner's label corpus, so the midgame anchor covers the state
distribution the fine-tune moves toward, not the one it leaves. Human rows are the only
supervision for setup contexts; zero heuristic setup rows anywhere; value loss zeroed on human
rows (no outcome label). Silent simplification of the anchor to hard-label self-BC is a spec
violation, not an implementation detail.

### D6 — Epoch pins
`ptr_v1_u500.pt` carries no ruleset stamp → **R0**. Anchor-state games and all gate evals run
R0. The candidate checkpoint is stamped with its epoch explicitly. Any R1 transition is the
successor slice's explicit decision, never a side effect of a default config.

### D7 — Gates with honest statistics (run only once ≥ 200 labeled scenarios exist)
1. **Setup agreement**: held-out top-1 settlement agreement; bar =
   `max(0.30, pre-fine-tune baseline + 0.10)` (restores §9's calibration clause the plan had
   dropped), reported WITH a binomial CI; at ~40 held-out scenarios the point estimate carries
   ±14pp noise, so a marginal result's remedy is MORE LABELS, not a lower bar.
2. **Non-inferiority on full-game WR vs the heuristic** (there is no midgame-position eval;
   the metric is full-game WR): paired-seed `WR_ft − WR_pre` with a CI; PASS iff the CI lower
   bound > −0.05. This REPLACES the source doc's Gate-2/Gate-3 pair — a point-estimate
   `WR_ft ≥ WR_pre` fails a perfectly neutral candidate ~50% of the time at n=600
   (SE ≈ 2pp), and the −5pp gate was vacuous beneath it.

### D8 — Deliverables and the successor contract
Outputs: (i) the label corpus (durable, schema-proof — JSONL stores seeds + picks, obs
regenerable forever), (ii) the fine-tuned candidate checkpoint, (iii) a frozen copy of the
candidate designated the **human-opening prior**. **Binding note for the successor self-play
slice:** it MUST implement agreement-with-labels monitoring and a ready-to-enable KL anchor to
the human-opening prior at setup nodes — without that anchor available, the fine-tuned openings
are expected to decay on contact with PPO (the original §Provenance design made this checkpoint
a piKL ANCHOR + warm-start; piKL is unwired; this spec pins what the successor must not omit,
but does not build it).

---

## Non-goals

- No self-play / exploration changes (successor slice).
- No BC corpus regeneration; no heuristic setup rows in any training data, ever.
- No engine changes; `PINNED_ENGINE_TREE` untouched.
- No new docs beyond this spec; `docs/plans/v2/setup_labeling.md` is updated only where this
  spec supersedes it (doc-sync per CLAUDE.md §6).

## Acceptance criteria (the `/dev-loop` gate is the authority)

1. Full gate green on real exit codes.
2. §12 smoke evidenced: throwaway scenarios end-to-end; JSONL rows verified by hand-inspection.
3. Recorder: a played game's record contains observed setup steps for both seats with
   placement-time state; `"setup_observed": true`; unit pins for both seats.
4. Converter: a shard built from ≥3 real labeled scenarios loads via `BcDataset` with no
   `StaleCorpusError`; per-row obs equals the current `obs_encoder` output on the reconstructed
   state (the load-bearing pin); determinism pin; `opponent_kind` duplication pinned.
5. `finetune.py`: smoke fine-tune on mock labels runs end-to-end; the anchor term demonstrably
   bounds non-setup drift (test: KL on held-out non-setup states stays under a pinned bound);
   value-loss-zero on human rows pinned.
6. Gate arithmetic implemented per D7 (CI-based non-inferiority), not point estimates.
7. The fine-tune RUN and its gates are deferred until ≥200 scenarios; only the code and gate
   implementations land now.
8. TDD per repo convention: failing test observed first for each behavioral change.

---

## RISKS — open, NOT cleared (premise-rider findings, carried in substance)

**Pre-mortem (most likely failure story).** The owner labels 200+ scenarios, the fine-tune
passes a noisy gate, the candidate warm-starts self-play — and six weeks later the policy's
openings are indistinguishable from a run that never saw the labels, because the successor
carried no anchor: the original design made the fine-tuned checkpoint a piKL ANCHOR + warm-start,
piKL is unwired, and this plan kept the warm-start half while deferring the anchor half to a
slice nobody has built. Contributing threads the rider grounded in the repo: the
self-distillation bullet conceals the slice's largest build (no per-head-distribution storage,
loader, or KL loss exists in `bc/` — inviting silent simplification to hard-label self-BC); the
anchor set is off-distribution by construction unless anchor states come from forced-human-
opening games; the source doc's gates were arithmetic theater (point-estimate Gate 3 fails a
neutral candidate ~50% of the time; Gate 1 carries ±14pp noise at 40 held-out scenarios); and
two silent mismatches (obs keys are 12 not 10, with the opponent-id embedding deciding which
conditional slice of the policy learns the openings; u500 is R0 while the shipped default
config is R1). D4–D8 respond to each thread — but a response is design, not proof; these risks
stand until the gates and the successor slice demonstrate otherwise.

**Argue the other side (the rider judged this NOT clearly weaker than the plan).**
(1) **Head-only fine-tune first**: the corner head has a setup-specific context path and the
source doc's own §9 diagnosis ladder names head-only fine-tune as the remedy for shared-weight
damage — the plan starts at the escalation instead of the bottom rung. The owner chose
self-distillation over freeze-heads in the interview; D5's online-KL latitude narrows the cost
gap, but the ladder inversion stands as an alternative.
(2) **The slice's value is hostage to the unbuilt successor**: if self-play will not carry a
human-opening anchor, the fine-tune decays on contact; if it will, the anchor consumes the
labels directly and the fine-tune step may be optional. D8 pins the contract but does not build
it. The durable products — the label corpus and the human-opening prior — are arguably the
load-bearing artifacts of this slice.
(3) **Real-game labels**: the recorder fix could grow the corpus as a playtesting side effect;
D3 adopts this.
