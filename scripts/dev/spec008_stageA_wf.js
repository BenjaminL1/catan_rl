export const meta = {
  name: 'spec008-stageA',
  description: 'Run spec 008 STAGE-A (the search kill-gate) all on OPUS: audit what STAGE-A code already exists (LCB final-move + variance accumulator appear present), build the MISSING STAGE-A pieces (SPRT infra, fixed-budget n-det diagnostic, oracle root-headroom pre-check), then EXECUTE the diagnostic + oracle probe on frozen v8 (CPU) and emit the PRE-REGISTERED GO/NO-GO verdict that decides whether the Gumbel (US1) build is funded. Inference-only, additive/default-off, matched-sim-budget. Ends at the verdict — building US1 is the user’s call.',
  phases: [
    { title: 'audit' },
    { title: 'build' },
    { title: 'gate' },
  ],
}

const MAX_ITERS = 4
const MODEL = 'opus'
const rawAgent = agent
async function safeAgent(prompt, opts) {
  for (let a = 0; a < 3; a++) {
    try {
      const r = await rawAgent(prompt, opts)
      if (r !== null && r !== undefined) return r
    } catch (e) { log(`retry ${opts && opts.label}: ${String(e).slice(0, 80)}`) }
  }
  return null
}

const COMMON = `You are executing STAGE-A of specs/008-gumbel-search-decision/spec.md (catan_rl_v2).
READ THE SPEC FULLY — it is the design authority (5 review rounds). GOAL: cheaply decide GO/NO-GO on
whether to build Gumbel-root search (US1), by (i) shipping the near-free LCB final-move win, (ii) building
the SPRT confirmation infra everything uses, (iii) running the fixed-budget n-determinization diagnostic,
and (iv) THE KILL PROBE — the oracle root-headroom pre-check. CENTRAL SUBTLETY (spec's #1 finding): a flat
sim-scaling curve looks IDENTICAL whether "visit-collapse is fixable by Gumbel" OR "the v8 root is already
near-optimal (value ceiling), so nothing raises strength" — so the n-det diagnostic ALONE cannot decide;
the ORACLE ROOT-HEADROOM pre-check must establish there is real strength to gain at the root BEFORE Gumbel
is funded. PRE-REGISTERED GO RULE (do not alter): build US1 ONLY IF (oracle root headroom > +15 Elo) AND
(depth-0 visit-collapse is high) AND (root-child-value Spearman >= 0.60); otherwise STOP, record NO-GO,
flag a chance-node/belief spec instead. INVARIANTS (sacred): inference-only on the FROZEN v8 checkpoint
(byte-identical; no retrain, no state-dict change); search-side only (src/catan_rl/search/ + eval/harness.py);
ADDITIVE + DEFAULT-OFF (new flags off => byte-identical to today, prove with a no-op smoke); MATCHED total
sim budget on EVERY comparison (the load-bearing control); CPU-pinned eval; fixed sims_per_move (not time);
append-only TB/JSON; no gui/ import. A training run is LIVE on MPS — do NOT touch src/catan_rl/ppo/ or the
training loop, and keep eval game-counts modest (the spec's diagnostics use ~50-game samples) so you do not
starve it of CPU. What already exists (verify, don't assume): node.py has a second-moment accumulator;
mcts.py has a final_move_mode=='lcb' path; config.py has fpu_mode/n_determinizations/sims_per_move. Champion:
runs/anchors/v8_promobar_u243.pt. HARD RULES: ruff + mypy (make lint/typecheck) + pytest green before every
commit; conventional commits; push origin/main; never loosen a committed test.`

const VERIFY_FMT = `Return MARKDOWN only. Sections: TOOLS (commands RUN + key lines), CHECKS (each
invariant incl. the DEFAULT-OFF no-op byte-identity + MATCHED-BUDGET control: pass/fail + evidence),
DATA (for the gate: open the readout, recompute a number), TAMPER (git diff on tests), BLOCKERS. End with
GREEN or RED on its own line.`
const REVIEW_FMT = `Return MARKDOWN only: "### <BLOCKER|SHOULD-FIX|NIT>: title" with where/issue/fix,
then VERDICT: READY or NOT-READY.`

const journal = []
function note(m) { log(m); journal.push(m) }

async function buildSlice(slice) {
  let findings = []
  for (let iter = 0; iter < MAX_ITERS; iter++) {
    const tag = `${slice.name}#${iter}`
    const impl = await safeAgent(
      `${COMMON}\n\nSLICE ${slice.name}: ${slice.what}\nTESTS/EVIDENCE REQUIRED: ${slice.tests}\n` +
      (iter > 0 ? `RESOLVE PASS — fix EXACTLY these open findings, nothing else:\n${JSON.stringify(findings, null, 2)}\n` : '') +
      `Re-green lint/typecheck/pytest, commit, push. Report commit sha + summary + key numbers.`,
      { label: `impl:${tag}`, phase: slice.name, effort: 'high', model: MODEL })
    if (!impl) { note(`HALT ${slice.name}: impl died`); return { name: slice.name, ready: false } }

    const ver = await safeAgent(
      `${COMMON}\n\nINDEPENDENTLY VERIFY slice ${slice.name} (do NOT trust the implementer). RUN the tools;
check invariants: ${slice.invariants}. Explicitly confirm the DEFAULT-OFF no-op byte-identity and that every
strength comparison is at MATCHED total sim budget. ${VERIFY_FMT}`,
      { label: `verify:${tag}`, phase: slice.name, effort: 'high', model: MODEL })
    const verText = ver || ''
    if (/\bRED\b\s*$/.test(verText.trim()) || !/\bGREEN\b/.test(verText)) {
      findings = [{ severity: 'BLOCKER', title: 'verify RED', issue: verText.slice(-1500), fix: 'make it genuinely green' }]
      note(`${slice.name} iter ${iter}: verify RED -> resolve`); continue
    }

    const review = await safeAgent(
      `${COMMON}\n\nREVIEW slice ${slice.name} against intent: ${slice.what}\nRead the actual diff/files.
Hunt: a sim-budget mismatch that invalidates a comparison; the frozen net being mutated; a default-on flag;
the oracle probe being weaker than "near-perfect root chooser"; the GO rule being softened; a diagnostic
readout that conflates the two hypotheses. ${REVIEW_FMT}`,
      { label: `review:${tag}`, phase: slice.name, effort: 'high', model: MODEL })
    const blockers = []
    const rtext = review || ''
    for (const m of rtext.matchAll(/###\s*(BLOCKER|SHOULD-FIX):\s*([^\n]+)/g)) {
      if (m[1] === 'BLOCKER' || iter === 0) blockers.push({ severity: m[1], title: m[2], issue: 'see review', fix: 'per review' })
    }
    if (blockers.length === 0 || /VERDICT:\s*READY/.test(rtext)) {
      note(`READY ${slice.name} (iter ${iter})`); return { name: slice.name, ready: true }
    }
    findings = blockers
    note(`${slice.name} iter ${iter}: ${blockers.length} open -> resolve`)
  }
  note(`HALT ${slice.name}: stuck after ${MAX_ITERS} iters`)
  return { name: slice.name, ready: false }
}

// ---------- slice 1: audit what STAGE-A already exists ----------
phase('audit')
const audit = await safeAgent(
  `${COMMON}\n\nAUDIT what STAGE-A machinery already exists vs is missing. Read specs/008-gumbel-search-decision/spec.md
(US0 acceptance criteria), then src/catan_rl/search/{config,node,mcts,eval_search,bakeoff}.py and
src/catan_rl/eval/harness.py. For each STAGE-A deliverable — (a) LCB final-move rule + per-child second-moment
accumulator, (b) SPRT/LLR/pentanomial gate wrapping the harness, (c) fixed-budget n-determinization diagnostic
(sims/K per world, K in {2,4,8}, total matched, with per-depth visit-concentration + root-child-value Spearman
readout), (d) oracle root-headroom pre-check (near-perfect root chooser vs current PUCT-root at matched budget)
— report EXISTS (with file:line + whether tested) / PARTIAL / MISSING. Also confirm the frozen-v8 default-off
no-op holds today. Return MARKDOWN: a per-deliverable status table + the concrete build list for the next slice.`,
  { label: 'audit', phase: 'audit', effort: 'high', model: MODEL })
note(`audit done`)

// ---------- slice 2: build the MISSING STAGE-A pieces ----------
const build = await buildSlice({
  name: 'build',
  what: `Build/complete the MISSING STAGE-A deliverables identified by this audit:\n${(audit || '').slice(0, 4000)}\n
Implement per the spec, additive + default-off + inference-only + CPU + matched-budget:
- LCB final-move (if not already): argmax(mean_Q - z*stderr), z=1.96 default, from the existing second-moment
  accumulator; flag off => byte-identical max-visit.
- SPRT gate: paired seat-swapped common-seed pentanomial LLR loop wrapping eval/harness.py; H0 elo<=0,
  H1 elo>=elo1 (5-10); alpha=beta=0.05; bounds +/-2.94; max-games cap. Reused by the probes below.
- Fixed-budget n-det diagnostic: sims/K per world, K in {2,4,8}, TOTAL sims matched; readout = per-K WR +
  per-depth (0,1,2) visit-concentration (% nodes >50% visits on one action) + Spearman(root-child value,
  ex-post terminal outcome) on a ~50-game both-seats sample.
- Oracle root-headroom pre-check: a near-perfect root chooser (high-budget / strong-rollout oracle at the
  ROOT only, PUCT below) vs current PUCT-root at MATCHED total budget, scored on the SPRT gate, to measure
  root headroom in Elo. This is the kill probe — make the oracle genuinely strong (document what makes it
  "near-perfect") or the GO/NO-GO is meaningless.
All flags default-off; ship a no-op smoke proving byte-identity to today with everything off.`,
  tests: `default-off no-op: search+eval byte-identical to today with new flags off (regression on a fixed
seed/board); LCB flag on changes only the final-move selection; SPRT loop reaches a decision on a synthetic
paired stream in fewer games than a fixed n; the n-det diagnostic + oracle probe run end-to-end at a smoke
game-count and emit well-formed readouts; matched-budget asserted in code (a mismatch raises). Suite green.`,
  invariants: `frozen v8 byte-identical; additive/default-off (no-op proven); every comparison matched total
sim budget (asserted, not assumed); CPU-pinned; append-only scalars; no ppo/ or training-loop import.`,
})

// ---------- slice 3: RUN the gate + pre-registered verdict ----------
let verdict = null
if (build.ready) {
  phase('gate')
  verdict = await safeAgent(
    `${COMMON}\n\nEXECUTE STAGE-A and emit the verdict (CPU, modest game-counts to spare the live training run).
Run on frozen v8 (runs/anchors/v8_promobar_u243.pt):
(1) the fixed-budget n-det diagnostic (K in {2,4,8}, matched total budget) -> per-K WR + per-depth
visit-concentration + root-child-value Spearman;
(2) the LCB-vs-max-visit A/B at matched budget on the SPRT gate (no-regression check);
(3) THE KILL PROBE — the oracle root-headroom pre-check at matched budget on the SPRT gate -> root headroom
in Elo (with CI).
Then apply the PRE-REGISTERED GO RULE verbatim: build US1 (Gumbel) ONLY IF (oracle root headroom > +15 Elo)
AND (depth-0 visit-collapse high) AND (root-child-value Spearman >= 0.60); else NO-GO -> flag a
chance-node/belief spec. Write docs/plans/spec008_stageA_verdict.md with all three readouts, the three
GO-rule clauses each marked pass/fail with the measured number, and the resulting GO or NO-GO. Commit + push.
Return MARKDOWN: the readouts, the per-clause pass/fail, and the GO/NO-GO verdict. If a full-power oracle
probe would need more games than is safe alongside training, run a smoke-power version and clearly label the
verdict PROVISIONAL with the game-count, rather than overclaiming.`,
    { label: 'gate-run', phase: 'gate', effort: 'high', model: MODEL })
  note(`gate verdict produced`)
}

phase('gate')
const summary = await safeAgent(
  `Write the completion summary (markdown) for the developer waking up: audit outcome, build slice verdict +
commit shas (git log), and THE STAGE-A GO/NO-GO on funding the Gumbel (US1) build — with the three measured
GO-rule clauses (oracle root headroom Elo, depth-0 visit-collapse, root-child Spearman) each shown pass/fail,
and whether it is full-power or PROVISIONAL (smoke game-count). If NO-GO, state plainly that the search lever
is likely capped by the value ceiling and the next search idea is chance-nodes/belief, not Gumbel.
Journal:\n${JSON.stringify(journal, null, 2)}\nVerdict:\n${verdict || '(did not run)'}`,
  { label: 'summary', phase: 'gate', effort: 'medium', model: MODEL })

return {
  results: [{ name: 'build', ready: build.ready }],
  halted: journal.filter(m => m.includes('HALT')),
  audit,
  verdict,
  summary,
}
