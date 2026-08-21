Run the **catan-rl-v2 overnight queue**: ratified spec slices through the dev-loop
back-to-back, unattended, with **owner-ratified governance (2026-08-21)**: SHOULD-FIX
autonomy (one bounded gate-green fix pass per slice) and **auto-merge to main on clean
gates** (0 blockers). A red gate, a merge conflict, or an unverifiable tip **stops the
queue** — nothing red or unverified ever merges.

> $ARGUMENTS

## Contract (assistant steps)

1. **Build the queue — RATIFIED specs only.** Each queue item needs a spec under
   `.claude/veriloop/specs/` whose Status line contains **RATIFIED**. Read each spec file
   and construct per-item: `title` (kebab slug), `feature` (the same hard-constraints
   framing a normal `/dev-loop` invocation would carry, incl. "spec=<path> (RATIFIED,
   BINDING)"), `spec` (the full spec text, verbatim), `specPath`. If the owner named work
   with **no ratified spec, refuse that item** and point to `/dev-plan` — no interviews,
   no confirm-and-go, at night.
2. **Order the queue** so independent slices come first and dependent slices after the
   slices they build on (auto-merge means each slice sees its predecessors on main).
3. **Keep the machine awake:** start `caffeinate -dims` as a background Bash task before
   launching, and tell the owner the session window must stay open (the queue runs inside
   this session).
4. **Launch** the `catan-rl-v2-overnight` workflow (script:
   `.claude/workflows/catan-rl-v2-overnight.js`) with args:
   `{ repoRoot: <absolute main-checkout path>, devLoopScript: <absolute path of
   .claude/workflows/catan-rl-v2-dev-loop.js>, queue: [...], posture?: ... }`.
5. **Morning report:** when the workflow completes, present its `morning` line plus the
   per-slice table (verdict, merged sha, SHOULD-FIXes applied/skipped, remaining
   concerns) and, if the queue stopped, exactly where and why, with the surviving branch
   named for review. Kill the caffeinate task.

## Hard rules

- The orchestrator merges **only a tip whose sha was verified against the gated,
  fix-pass-re-gated state**; a mismatch stops the queue.
- Merge conflicts are never resolved unattended — abort + stop.
- Preflight refuses to start on: dirty tracked files in the main checkout, diverged
  main vs origin, or any non-ratified spec.
- Commits and merges follow repo convention: conventional, lowercase, **no AI
  attribution or co-author trailers of any kind**.
- This mode does NOT change `/dev-loop` itself — interactive runs still stop before
  merge for owner sign-off.

## Provenance

Owner-ratified 2026-08-21 (auto-merge-on-clean-gates chosen explicitly over
stack-never-merge, with the tradeoff surfaced: findings on merged slices are reviewed
the morning after, not before merge). Hand-owned, session-authored — the veriloop
generator does not regenerate this file or the orchestrator script.
