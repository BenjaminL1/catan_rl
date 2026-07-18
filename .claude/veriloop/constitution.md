# catan-rl-v2 constitution — invariants the dev-loop checks every plan against

These are non-negotiables. The `/dev-loop` gate checks the **plan** against this list *before*
any code is written, and the review lenses check the **diff** against it. A plan or diff that
violates one is a **BLOCKER**. Keep this list short and true.

> **veriloop STARTER** — a scaffold from detected facts. Replace the TODOs with real, code-cited
> invariants (veriloop phase 4 mines these from the code + git history). This file is
> three-way-merged on re-run: your edits are preserved.

## Build & correctness

1. **The gate runs on real exit codes.** `make typecheck` · `make lint` · `cargo fmt --all -- --check` · `make test-unit` must pass; a
   red check is a BLOCKER, never waved through on "looks right". _(owner: `code-review`)_
2. _TODO: the core correctness invariant of this repo (the rule a change must never break)._ _(owner: assign — usually `code-review` or `drift`)_

## Boundaries & safety

3. _TODO: the trust/boundary invariant (what must never leak, what stays server-authoritative)._ _(owner: the `security` expert; if this roster has none, delete this rule or revisit the roster)_

## Conventions

4. **Honor `CLAUDE.md`** code standards (types, exports, imports, secrets via env only). _(owner: `code-review`)_

## Landing (owner-reserved)

5. **Branch + preview only.** Work lands on a branch; **never** merge to the default branch or
   deploy without explicit owner sign-off. Conventional commits, no AI co-author trailer, never
   stage `.env*`. _(owner: `code-review`)_

---

### Rule ownership — target state
Every rule must be owned by exactly ONE expert, and every expert must own at least a
few rules (no orphan rules, no jobless experts). The starter rules are pre-assigned
below; assign each TODO as you replace it — if a rule has no plausible owner in this
roster, either the roster is missing an expert or the rule doesn't belong here.

- **Baseline Reviewer** (`code-review`) — always included — correctness, conventions, type-safety, test integrity
- **Drift Sentinel** (`drift`) — parity/golden files: scripts/dev/probe_oracle_sprt.py, scripts/record_conformance.py, src/catan_rl/selfplay/snapshot_opponent.py
