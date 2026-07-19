# Spec: glyph-anchor-only scoreboard ordering (audit #1) — RATIFIED 2026-07-18

**Intent.** The human scoreboard (the outcome-anchored eval over parsed ThePhantom games) is
empty — 0 of 20 corpus records eligible — solely because `placement_order_established` is
stamped `False` whenever the log-side ordinal (`_log_grant_ordinal`) cannot fire, which on real
footage is ALWAYS (re-OCR duplication). The glyph-anchor path alone was verified to establish
order on 20/20 real games with zero grant collisions. Owner decision (audit Decision 1): adopt
glyph-anchor-only ordering.

**Binding requirements.**
1. Add an explicit opt-in mode (`require_log_ordinal: bool = True` parameter or equivalent,
   default preserving today's behaviour byte-identically) on the cross-check/stamping path in
   `src/catan_rl/human_data/harvest.py` (see `cross_check` → `placement_order_established`
   stamping, ~L1295/L1365) and its consumers (`record.py`, `validate.py`, `scripts/vlm_spike.py`,
   `scripts/dev/collect_corpus.py`). With `require_log_ordinal=False`, ordering established by
   `order_openings_by_grant` + `identify_granting_settlement` ALONE (unique-or-None; any grant
   collision or ambiguity still fails closed → `placement_order_established=False`).
2. Provenance must record WHICH signal established order (e.g.
   `provenance["order_source"] = "log+glyph" | "glyph_only" | None`) — additive key, never
   renaming existing keys.
3. Existing two-signal path stays the default; every existing test stays green unmodified.
   New tests: glyph-only ordering accepted on a collision-free fixture; grant-collision fixture
   still rejected under `require_log_ordinal=False`; default path unchanged (pin).
4. Wire `--require-log-ordinal/--no-require-log-ordinal` (or equivalent) through
   `scripts/dev/collect_corpus.py` so the corpus rebuild can opt in.
5. Do NOT recompute the real 20-record corpus inside the loop (worktree lacks gitignored data);
   ship the code + a one-command recompute invocation documented in the PR/preview notes. The
   owner's main checkout runs the recompute post-merge (expected result: ~15 winner-bearing
   eligible games).

**Hard constraints.** Never edit `src/catan_rl/human_data/glyph_anchor.py` (SHA-256 fingerprint
gate; validation artifacts not fully in repo). Fail-closed invariants sacred: no accept/reject
decision on any existing fixture may change under the DEFAULT mode. Additive only.

**Acceptance.** ruff + mypy --strict + full pytest green; new pins above; default-mode
byte-identical behaviour proven by the existing suite passing untouched.
