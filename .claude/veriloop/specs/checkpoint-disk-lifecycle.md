# Spec: checkpoint disk lifecycle — strip league pool, slim fallback (audit #2) — RATIFIED 2026-07-18

**Intent.** Every training checkpoint is ~577 MB of which ~560.9 MB (97%) is the embedded
100-snapshot league pool (~99/100 snapshots byte-identical across consecutive saves); the
optimizer is only ~11 MB. This chronic bloat on a 16 GB box is what tripped the disk guard and
killed v11 at u749 mid-climb — and the guard also skipped the terminal save, so v11's final
weights were never written. Fix the lifecycle before any multi-day run (the arch re-bootstrap).

**Binding requirements.**
1. **League sidecar:** stop embedding the league pool in rolling/promo/terminal checkpoint
   files (`src/catan_rl/checkpoint/manager.py`, `_capture_league_state` et al.). Persist league
   snapshots to a content-addressed sidecar store keyed by snapshot id/hash (dedups identical
   snapshots across saves), with the checkpoint holding references. Resume must reconstruct the
   league exactly (pin: save→load round-trip equality on league state, RNG, optimizer, policy).
2. **Backward compatibility:** loading an existing FAT checkpoint (embedded league) must still
   work unchanged — v11_cand_u724.pt, promo ckpts, and all `runs/anchors/*` must stay loadable.
   Migration is one-way on save, opt-out not required.
3. **Slim disk-trip fallback:** when the free-disk guard trips ANY save (rolling/promo/terminal),
   always attempt a policy-only slim save (~5.6 MB) before exiting, write a
   `kind="disk_abort"` marker (log + a small JSON/exit-code signal, nonzero process exit) so a
   stranded run is distinguishable from a clean auto-stop. Test with monkeypatched disk_usage.
4. **`bank_anchor()` + `scripts/reclaim_disk.py`:** a utility that re-saves a checkpoint as
   policy-only (for `runs/anchors/*`) and a script that reports/reclaims (dry-run default,
   `--execute` to act) — dedupe byte-identical files (e.g. v10_cand/v10_chain), re-save anchors
   slim. The loop ships and TESTS the script on synthetic files only; it must NOT touch the real
   `runs/` of the owner's checkout (worktree has no runs/ anyway). Owner executes post-merge
   (expected ~4.5 GB reclaimed).
5. **Persist `updates_since_promotion`** across resume (reconstructable from
   `_last_promote_update` in league/checkpoint state) so auto_stop's hard-400 clock is
   per-lineage, not per-session. Pin: resume mid-run → counter continues, not resets.

**Hard constraints.** Additive/compatible: every existing config trains byte-identically except
file layout; TB scalars never renamed; no change to promotion/ratchet logic or any game rule.
Never touch `src/catan_rl/engine/` game rules or `glyph_anchor.py`.

**Acceptance.** ruff + mypy --strict + full pytest green; round-trip resume pin; fat-checkpoint
load pin; disk-trip slim-fallback pin; reclaim script dry-run pin on synthetic tree. Rolling
checkpoint size measured < 25 MB in the round-trip test (policy+optimizer+refs, no pool).
