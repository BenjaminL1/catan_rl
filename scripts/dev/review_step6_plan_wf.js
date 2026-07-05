export const meta = {
  name: 'review-step6-plan',
  description: 'Senior-RL expert review of docs/plans/v2/step6_human_corpus.md (post-corpus implementation plan: opening scoreboard + human-seeded self-play) -> severity-tagged findings + READY/NOT-READY verdict',
  phases: [
    { title: 'Review' },
    { title: 'RedTeam' },
    { title: 'Synthesis' },
  ],
}

const rawAgent = agent
async function safeAgent(prompt, opts) {
  for (let a = 0; a < 3; a++) {
    try { const r = await rawAgent(prompt, opts); if (r) return r }
    catch (e) { log(`retry ${opts && opts.label}: ${String(e).slice(0, 80)}`) }
  }
  return null
}

const OUTFMT = `
OUTPUT: MARKDOWN only. Findings as "### <BLOCKER|SHOULD-FIX|NIT>: title" with
**where** / **issue** / **fix** bullets. End with VERDICT: READY or NOT-READY.`

const CONTEXT = `
You are the senior RL-game-dev reviewer for catan_rl_v2 (custom PPO, 6-head autoregressive
action space, ~1.38M-param net, champion v8 at runs/anchors/v8_promobar_u243.pt, determinized
PUCT-MCTS inference search, league self-play with frozen anchor + promotion bar 0.63).
UNDER REVIEW: docs/plans/v2/step6_human_corpus.md — the implementation plan for USING the
human-openings corpus once the video-parsing harvest completes. Read the plan file fully, plus
the context it builds on: docs/plans/human_data_pipeline.md (§3 schema, §5 correctness traps
esp. 5.4 scoreboard re-scope / 5.7 seed legality / episode_source), src/catan_rl/human_data/record.py
(GameRecord schema v2, is_scoreboard_eligible/is_seed_eligible), docs/plans/v2/setup_strength_roadmap.md
(the pivot lever the plan references), src/catan_rl/eval/harness.py (the eval the plan constrains),
src/catan_rl/ppo/arguments.py (config SoT), src/catan_rl/selfplay/league.py, CLAUDE.md (invariants:
1v1 ruleset sacred, obs schema, additive TB, checkpoint lineage, review-and-resolve convention).
ESTABLISHED EMPIRICAL PRIORS you must hold the plan against: the human playtest judged v8+search
subhuman with an opening/ore blind spot; the Tier-0 probe showed opening value Spearman 0.43-0.50
vs mid/late 0.88 UNDER V8 SELF-ROLLOUTS (self-graded); spec 007 closed value-retargeting as low-EV
(frozen-trunk probe: retarget does not improve rank; MC probe: ~0.08 Spearman headroom, residual is
irreducible StackedDice variance); v7/v8 self-play plateaus were promotion-bar artifacts; search
gives +55-90 Elo. The corpus: ~350-800 games, nearly all featuring ONE human (ThePhantom).`

phase('Review')
const lenses = await parallel([
  () => safeAgent(`${CONTEXT}
LENS A — RL-EXPERIMENT CORRECTNESS. Judge the plan as an experiment design: is GATE-A actually
capable of confirming/denying the blind spot (M1/M2/M3 definitions, the pre-registration
discipline, the single-human + seat + luck controls — are they sufficient, are any circular)?
Is workstream B's training design sound (seed_prob mechanics, both-seats seeding, warm-start
choice, control arm, interference tripwire) and are its gates measuring what they claim
(esp. GATE-B3's "calibration gap shrinks" — is that well-defined and non-gameable)? Does
anything contradict the spec-007 finding that the VALUE head is near its representation
ceiling — i.e., can seeded self-play actually fix an opening-value gap, or does the plan
conflate policy-prior exposure with value-head capacity? Where does the plan risk fooling us?
${OUTFMT}`, { label: 'lens:rl-correctness', phase: 'Review', effort: 'high' }),
  () => safeAgent(`${CONTEXT}
LENS B — STATISTICS + SWE/INTEGRATION. (1) Statistics: n≈350-800 games with ONE human in every
game — walk through the actual power of M1 (Spearman/AUC CIs at that n), M2 (residual permutation
test given archetype base rates), M3 (bottom-quartile WR: what n lands in that cell?); is the
GATE-A GO/NO-GO decision rule crisp enough to be non-gameable, and is the NO-GO branch actually
actionable? Sensitivity of conclusions to the ore_heavy threshold choice. (2) SWE: the engine
bridge (does the engine expose the placement/grant APIs the plan assumes — check
src/catan_rl/engine/game.py + player.py setup paths), obs-encoder reuse at post-setup states,
eval-harness natural-only assertion feasibility, seed_prob wiring surface in
ppo/vec_env/game_manager, TB additivity, effort estimates realism. Flag anything the plan
hand-waves that will bite during implementation. ${OUTFMT}`,
    { label: 'lens:stats-swe', phase: 'Review', effort: 'high' }),
])

phase('RedTeam')
const red = await safeAgent(`${CONTEXT}
RED-TEAM the plan: find the ways it produces a CONFIDENTLY-WRONG conclusion or a wasted
multi-week run. Attack at least: (1) GATE-A passes for a reason OTHER than an opening-value
blind spot (e.g. distribution shift — v8's value head has never seen human-style boards ANYWHERE
in the game, so low external Spearman may reflect global OOD, not opening-specific failure —
does the plan's "midgame external calibration" comparison actually control for this, and can it
even be computed from an openings-only corpus with no midgame states?); (2) seeded self-play
"passes" GATE-B3 by degrading the measurement rather than improving play; (3) the seeds
quietly narrow diversity instead of widening it (dedupe, pool composition, seed_prob choice);
(4) the one-human confound surviving the A3 controls; (5) bridge-induced artifacts (starting
resources, robber, player-to-move) creating states subtly off-distribution for the obs encoder.
For each: is the failure caught by an existing gate, and if not, what cheap addition catches it?
${OUTFMT}`, { label: 'redteam', phase: 'RedTeam', effort: 'high' })

phase('Synthesis')
const synth = await safeAgent(`${CONTEXT}
Synthesize into ONE deduplicated severity-tagged list (BLOCKER first) + a final
VERDICT: READY or NOT-READY for the plan document itself (a plan is READY when a competent
implementer following it would not produce a confidently-wrong conclusion or an unrecoverable
run; SHOULD-FIXes that are wording-level can ship as edits). Be concrete — every finding must
name the plan section and the exact fix.

=== LENS A ===
${lenses[0] || '(none)'}

=== LENS B ===
${lenses[1] || '(none)'}

=== RED-TEAM ===
${red || '(none)'}
${OUTFMT}`, { label: 'synthesis', phase: 'Synthesis', effort: 'high' })

return { synth }
