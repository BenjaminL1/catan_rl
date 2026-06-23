# Plan Review Resolution (009-finite-resource-bank)

The expert plan review (3 lenses + synthesis, grounded in real code) returned **NOT-READY** with 2 BLOCKERs + 5 SHOULD-FIX + 3 NIT. This file records how each is resolved and supersedes the affected parts of `research.md` / `contracts/internal-interfaces.md`.

## BLOCKER 1 — mutation surface is ~20 sites across 8 files, not "3 paths"

Full enumeration (verified by grep). Every **draw** and **recirc** must hit the bank or the global conservation invariant breaks:

| Bank op | Sites |
|---|---|
| Production draw | `game.py` update_playerResources (two-pass) |
| Setup-grant draw (flat) | `game.py:196`, `bc/dataset.py:324`, `labeling/scenario_gen.py:342`, `env:708` |
| Build recirc | `player.py` build_road(72-73, paid only)/settlement(133-136)/city(187-188) |
| Dev-buy recirc | `player.py` draw_devCard(358-360) |
| Bank-trade recirc+draw | `player.py` trade_with_bank(535-552) — callers env:628, recorder:687, initiate_trade:577 |
| Discard recirc | `player.py:583`, `heuristic.py:256`, `random_ai.py:65`, `recorder.py:483`, `env:436`+`633`, `gui/view.py:625`+`639` |
| YOP draw | `player.py:447`, `recorder.py:702`, `env:590`, `gui/view.py:647` |
| Bank-neutral (NO change) | Monopoly (`player.py:486`, `recorder.py:715`, `env:604`), Steal (`player.py:253`) |

**Resolution**: centralize on `catanBoard` — `resourceBank` + `bank_recirculate(delta)` + `bank_draw(delta)` + `assert_conservation(players)`. Route every site above through them. RL+conformance-critical paths (engine, env, recorder, heuristic, random_ai, bc.dataset, labeling.scenario_gen) are wired and conservation-gated; the GUI paths (`view.py`) are wired for human-play correctness (the play_vs_v8 harness) but are off the conformance gate. The **C7 cross-driver conservation test** drives a depleting line through engine + env + recorder + heuristic-vs-heuristic + bc-dataset and asserts `Σhands+Σbank==95` after every action — the only guard that catches a missed site (seeds 7/8/15 cannot, since they never deplete).

## BLOCKER 2 — depletion fixture buildability

The Python recorder has **no dice-injection seam** (StackedDice is Rust/seed-only; the fixture's `dice_roll` is produced, not injected). Natural 19-card depletion in 1v1 is rare/possibly-unreachable in random play.

**Resolution (staged)**:
1. **Reachability probe** — instrument the engine to flag any production roll that hits a depletion branch; run a large heuristic-vs-heuristic + random search over seeds with a high turn cap. If a depleting seed is found → pin it, record the fixture naturally (no new capability).
2. **If unreachable** → add a minimal **forced-roll capability** to the recorder (a `forced_rolls`/scripted-line option that injects an ordered dice sequence; the fixture records those rolls, TS replays them — the TS side already consumes `dice_roll` from the fixture). Construct a deterministic depletion line (both seats with buildings on one resource's number; force that number repeatedly until combined demand > remaining bank).
3. **Always**, regardless of (1)/(2): prove the depletion *rule* directly via **unit tests in both engines** that set a near-empty bank and assert the outcome equals TS `resolveBankProduction` / `bank.test.ts` cases. This makes SC-003/SC-005 robust even if a full-game fixture proves marginal.

If neither (1) nor (2) yields a clean cross-engine depletion fixture within scope, the spec is honestly amended: cross-engine proof = byte-identical 7/8/15 (no-op) + the two-engine depletion unit tests; the full-game depletion *fixture* is documented as deferred. (Decided empirically by the probe, not assumed.)

**RESOLVED via option (1) — natural depleting seed found.** A reachability probe over 1500 seeds (max_main_turns=250) found seed **1128** hits a depletion branch (sole-claimant: `avail=2, d0=3, d1=0 → seat 0 takes 2, bank→0`; min bank seen across all games = 2). Recorded as `../Torevan/packages/engine/src/conformance/game-seed-1128.json` (768 steps) and the TS conformance suite passes with it included (9 tests). No forced-roll capability needed. Natural 1v1 depletion is extremely rare (1/1500 games, only after ~turn 200) — confirming the obs-out / no-flag decisions impose negligible MDP change. The fixture is written into the Torevan working tree (uncommitted) per FR-015; the Torevan session commits it on `feat/finite-resource-bank`.

## SHOULD-FIX resolutions

- **S1 — gating not expressible in the flat mask + apply paths lack underflow guard.** Two change-sites, both explicit: (a) **apply-time guard** on every YOP/bank-trade *draw* path (env YOP, `player.trade_with_bank`, `player` YOP, recorder picks) — Python failure mode = early-return no-op (no exception contract), mirroring TS throw; pin a unit test that a bank-empty YOP/bank-trade leaves the hand unchanged. (b) **mask gating**: descope to apply-time rejection for the first cut (the flat per-head `resource2_default` cannot express the YOP doubled-pick `bank[first]>=2 iff first==second` constraint without new mask keys + head context). Mask-level gating (new `resource2_yop`/`resource2_trade` keys) is recorded as a follow-up; FR-008 is satisfied by apply-time rejection + recorder-pick gating now, not by the mask layer. **Do not claim mask-level gating.**
- **S2 — player methods have no guaranteed bank ref.** Pass `board` explicitly: `trade_with_bank(self, r1, r2, board)` and route `discardResources` recirc through `game.board` (it already has `game`). No reliance on `getattr(self,'game',None)` for bank access — a missing wire must be a hard error (assert), not a silent no-op. Update callers (env:628, recorder:687, initiate_trade:577).
- **S3 — recorder random YOP/bank-trade pick not gated.** The recorder filters its random YOP/bank-trade pick by current bank availability before emitting (mirroring C6) so no recorded fixture is TS-illegal.
- **S4 — seat d0/d1 binding.** Two-pass tally binds `enumerate(playerQueue.queue)` index 0→d0, 1→d1, iterate resources in fixed order, call `resolve_production` once per resource. Unit test distinguishes seat0-sole-claimant vs seat1-sole-claimant (asymmetric) to catch a transposition.
- **S5 — RNG-keystream identity.** Add an identity test: the two-pass distributor yields byte-identical `player.resources` AND identical broadcast RESOURCE_CHANGE stream vs the pre-change behavior whenever `bank>=demand` over seeded rolls. Bank helpers are RNG-free (asserted).

## NIT resolutions

- **N1 — Rust event reshape** is an internal-consumer concern, not a parity hazard (consumers read canonical state). Keep Rust grant aggregation per-(player) to match Python/TS; fix the stale `state.rs:1180` comment.
- **N2 — snapshot 8th key** (`last_seven_roller`) is dropped by `_normalise_snapshot` (recorder.py:279-287) — the no-bank guarantee rests there. Pin a test that `_normalise_snapshot`'s keyset == the TS LoggedSnapshot keyset.
- **N3 — dict typing**: bank helpers document their key domain; a unit test asserts a Charlesworth-keyed delta recirculated into the engine-keyed bank does not silently drop a resource.

## Implementation order (dependency-gated)

1. `board.py` bank state + helpers (+ conservation assert).  2. `game.py` two-pass production + setup draw.  3. `player.py` recirc/draw + `trade_with_bank(board)` + YOP/discard.  4. env + recorder + heuristic + random_ai + bc.dataset + labeling + gui sites.  5. Unit tests (conservation across all drivers; depletion branches; gating no-op; RNG identity; key-domain).  6. ruff+mypy+pytest green → commit.  7. Build Rust ext in worktree; re-record 7/8/15 → byte-identical gate → commit.  8. Depletion fixture (probe → forced-roll/unit-proof).  9. Phase B Rust mirror.  10. Phase C docs.
