"""Self-play infrastructure for the v2 PPO training loop.

Phase 6 ships :class:`catan_rl.selfplay.league.League` — a thin
opponent-sampling container with extension points for past-policy
snapshots. Phase 3-style features (PFSP-hard, latest-policy reg, duo
exploiter, TrueSkill, Nash pruning) are deliberately deferred until
the diagnostic in `docs/plans/v2_setup_strength_roadmap.md` §C.0
indicates they're worth the implementation complexity.
"""
