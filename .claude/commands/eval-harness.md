---
description: Run the evaluation harness against the frozen champion
---

Run the eval harness for `$ARGUMENTS` (a checkpoint path).

Required modes:
1. **Heuristic benchmark** — `python scripts/evaluate.py $ARGUMENTS --opponent heuristic --n-games 200`. Target: ≥99% win rate.
2. **Frozen-champion benchmark** — H2H vs `checkpoint_07390040.pt` over 200 deterministic seeds with first-player swap. Target: ≥70% win rate.
3. **Per-head entropy** — read recent TB logs and report any head with entropy < 0.001.

If `scripts/eval_harness.py` exists (Phase 0), prefer it with `--mode all`. Otherwise fall back to `evaluate.py` for the heuristic benchmark only and note the harness is not yet implemented.

Output a markdown summary table with columns: Mode | Result | Pass/Fail | Threshold.
