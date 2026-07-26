# Spec: port-harvest — recover real Colonist port maps for the human-opening corpus

**Status: OWNER-DELEGATED DRAFT (2026-07-26) — written under explicit instruction to
proceed overnight ("get this done the best way you see fit ... have it ready for me in the
morning"), on a machine-local scope ("run solely on this machine"). NOT owner-ratified.
The premise challenges in §RISKS are OPEN, not cleared.**

**Feature in one line.** Read the port TYPE sitting in each of the 9 fixed board slots from
retained Colonist frames, emit it as a standalone sidecar keyed `(video_id, game_index)`,
and measure whether ports move the policy's opening choice at all — so the far more
expensive re-ingest of ~100 videos is funded by evidence rather than assumption.

## Why this exists

Every corpus row carries `board.ports == "OMITTED in v1"` (140/140). Ports **are** in the
observation (`src/catan_rl/policy/obs_encoder.py:280`, per-vertex 7-way one-hot: none /
3:1 generic / 2:1 x5). The current convention marginalises over K=8 guessed port
assignments. For a **paired** comparison that is defensible and is in fact the
lower-variance estimator; for **unpaired** consumption (training on ThePhantom's openings)
a guessed port map is simply a false statement about the board he faced.

## Decisions (binding for the build)

### D1 — Slot geometry: outward PERPENDICULAR to the slot edge
Slot positions come from `load_topology().port_slots` (9 entries `{slot, hex, corners,
vertices}`; positions never vary between games — only the type does). The port icon sits
outward along the **normal to the segment joining the slot's two vertices**, sign-fixed to
point away from the board centroid. A radial-from-centroid offset was tried first and
mis-centred 3 of 9 slots; the perpendicular normal at ~78 px with a 190 px tile centred all
9 on the pilot board.

### D2 — Sprite match is load-bearing; the ratio read is a CHECK that can only reject
Each port renders a sprite (5 resource icons, or "?") plus a ratio string ("2:1"/"3:1").
These are **not independent** — the sprite determines the ratio — so agreement validates
only the generic-vs-specific bit. Colonist renders as GLYPHS and OCR-on-rendered-text
already broke this pipeline once (the setup-parse wall). Therefore: the **sprite** decides;
the ratio read may only **reject** (disagreement ⇒ typed rejection), never break a tie and
never supply a label.

### D3 — Dual decode; disagreement is the error signal
The composition is fixed (exactly one 2:1 per resource + four 3:1). Decode twice:
(a) **unconstrained** per-slot argmax over sprite match scores, and
(b) **constrained** assignment (Hungarian) maximising total score subject to the
composition.
Using the composition during inference would consume it as information and destroy it as a
check — the generalised form of the banned "deduce the 9th slot" shortcut. Running both and
comparing preserves both. **Agreement is required**; disagreement ⇒ typed rejection. This
matters because the multiset validator is **permutation-invariant** and cannot see a slot
swap, which is the dominant expected error.

### D4 — Fail-closed, and NEVER deduce a slot
If any of the 9 slots is unreadable, reject the whole board. Deducing the 9th slot from the
other 8 is BANNED even though the fixed composition always permits it: it would let a
classifier that never actually reads a slot report success.

### D5 — SIDECAR, not a corpus field
Ports are written to `data/human/ports/harvest.jsonl`, keyed `(video_id, game_index)`,
hashed independently. They are **NOT** written into `row["board"]["ports"]`. Rationale: any
adapter that reads `board` would otherwise inherit ports for free, and the
"training-only, measurement keeps K=8" fence would be a convention rather than a mechanism.
The corpus JSONL is in the step6 §1 freeze set and its hash moved once already today
(the `opponent_strength` restamp); this build must not move it again.

### D6 — Scope tonight is PIXELS-ON-DISK ONLY. Re-ingest is NOT authorised.
Only **34** frame dirs exist (`data/human/vlm_spike/frames/<video_id>__g<N>/`), of which
**24** join an accepted corpus row. The other **116** corpus rows have **no pixels** —
`data/human/vlm_spike/localized/*.json` stores only `video_id`, `game_index`, `vlm_note`,
`players`. Harvesting them requires re-acquiring ~100 videos, which is the real cost pole
and is **explicitly out of scope for this build**. It may only be funded by the D7 result.

### D7 — The invariance probe is a DELIVERABLE, not a nice-to-have
Nobody has stated a port accuracy tolerance, because the operative quantity is not per-slot
accuracy — it is **decision-flip rate**: P(the policy's chosen corner changes | port map
changes). Measure it directly on the boards that now have real ports: compare the policy's
setup choice under the **real** harvested port map against its choice under each of the K=8
guessed assignments. Report the flip rate and the metric spread. This is what decides
whether the ~100-video re-ingest is worth a day.

### D8 — Extrapolation error must be MEASURED, not assumed
`MAX_AFFINE_RESIDUAL_PX = 5.0` (`src/catan_rl/human_data/orientation.py:88`) is the enforced
budget; the 0.86–1.23 px figures observed in `meta.json` are achieved values on hex centres.
Ports render in the sea ring **outside the fitted hull**, so slot positions are an
**extrapolation with unmeasured error**. The pilot centred 9/9 on ONE board; that is not a
measurement. Report per-slot centring error across all 34 boards.

## Non-goals
- No re-ingest / no video downloads / no `yt-dlp` (D6).
- No write to `data/human/corpus/provisional_openings.jsonl` and no corpus re-hash (D5).
- No change to the engine, obs schema, action space, or training config.
- No use of harvested ports in any paired measurement path; K=8 stays the scoreboard
  convention.
- No claim that harvested ports improve any metric — D7 measures, it does not assert.
- Does not fix the 5 known illegal-settlement rows (4 of which have no pixels and cannot be
  re-verified without re-ingest).

## RISKS — open challenges, carried from the premise review (NOT cleared)

**Pre-mortem (most likely failure).** July 2027: the harvest shipped at high per-slot
accuracy and nobody used it. The labels landed in `human-opening-reference`, which was never
ratified because ratification required answering D4 of that spec — the instrument is
*unsigned* and can refute "openings are weak" but never confirm strength. Exact ports made a
refutation instrument more precise without letting it conclude anything new. The opening
battery in `scripts/opening_sweep.py` is **port-blind** (pips, lumps, contested_race, denial,
cut_vulnerability), so ports entered only through the policy's per-vertex one-hot on 18 of 54
vertices, where the K=8 draws had already been near-invariant. Meanwhile the agent still
scored 4 VP with zero cities in 58 turns against a human and `src/catan_rl/eval/harness.py`
still reported no cities, no VP-by-turn, no bank-trade composition. *We raised the precision
of a measurement we could not act on, while the failure we had actually observed stayed
unmeasured.* D7 exists specifically to make this failure detectable before it is paid for.

**The strongest opposite case.** This reverses a RATIFIED non-goal: step6 §288 —
*"no harvest-v1 port extraction"*. The harvest is orphaned at both ends: the upstream
consumer (BC/imitation training) is banned by step6 §6 and was killed by four councils; the
downstream consumer (`opening_scoreboard.py`) **does not exist in the tree**, and
seed-injection's mechanism (`reset_source`/`seed_prob`) returns nothing under `grep`. For a
paired instrument K=8 is the lower-variance estimator, so real ports are arguably a
measurement *downgrade*. The counter-case is weaker only on **ordering**, not merit: do the
harvest after a consumer is ratified and it is defensible; do it now and it is speculative
inventory. D6 confines the spend to pixels already on disk precisely to bound this.

**Known corpus rot this build does not fix.** 5 rows carry adjacent (illegal) settlements;
`human-opening-reference.md:77` is now stale (it calls `opponent_strength` a hardcoded
constant, which today's restamp changed to 124 rank_badge / 13 tournament / 3 unknown), and
that restamp incidentally satisfied step6 §0.2's ≥10-tournament sign-guard at 13.

## Acceptance criteria (the /dev-loop gate is the authority on checks)
1. Full gate green (`make typecheck` · `make lint` · `make test-unit`).
2. **Geometry pin**: slot crop centring error reported per slot across all 34 boards (D8);
   the pilot's 9/9 result is reproduced on the pilot board bit-for-bit.
3. **Dual-decode pin**: a synthetic slot swap is DETECTED by the D3 disagreement check and
   NOT by the multiset validator — proving the check adds what the validator cannot see.
4. **Fail-closed pin**: an occluded/blank slot yields a typed rejection for the whole board,
   and no code path can emit a 9-slot map with fewer than 9 independently read slots (D4).
5. **Sidecar pin**: the run writes only under `data/human/ports/**`; the corpus JSONL hash is
   byte-identical before and after (D5).
6. **Invariance deliverable**: D7's decision-flip rate and metric spread are reported with a
   plain statement of whether the ~100-video re-ingest is justified.
7. Determinism: same frame ⇒ identical port map, bit-for-bit.
8. Read-only w.r.t. training: nothing written under `runs/train/**`.

## Follow-on (explicitly NOT this build)
Re-ingest of ~100 videos for the remaining 116 rows — gated on D7 showing ports actually move
opening choices. Independent of that, both prior premise reviews ranked **aggregate midgame
instrumentation** (cities built, bank-trade composition, VP-by-turn over existing self-play
corpora — n in the thousands, zero new data, no ports) above any further opening-side work.
