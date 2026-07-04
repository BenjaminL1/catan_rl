export const meta = {
  name: 'human-data-overnight-build',
  description: 'Autonomous review-and-resolve loop building the human_data video-parsing pipeline overnight: frozen-oracle + independent-verify + red-team guards, per-slice loop, stage gates, tiered 5->30 execution, gated full harvest, morning report. Scope-locked to src/catan_rl/human_data + scripts/mine_phantom.py; never touches engine/training/gui. Opponent-strength now comes from the COMMITTED strength manifest.',
  phases: [
    { title: 'Stage0 scaffold' },
    { title: 'Stage1 log-spine' },
    { title: 'Stage2 board+openings' },
    { title: 'Tiers+gate' },
    { title: 'Morning report' },
  ],
}

// ============================ shared config ============================
const SPEC = 'docs/plans/human_data_pipeline.md'          // the build spec the agents follow
const MAX_ITERS = 4                                       // stuck-detector per slice
const COMMON = `You are building the human_data video-parsing pipeline. THE SPEC is ${SPEC}
(Read it fully — esp. the §5 correctness constraints, the GameRecord schema §3, conventions §6).
The de-risk spike code was banked into the repo in Stage 0 (look under scripts/dev/human_data_spikes/
and scripts/export_topology.py); reuse it, do NOT rebuild CV from scratch.
HARD RULES: scope-locked to src/catan_rl/human_data/ + scripts/mine_phantom.py + tests + the named
fixtures — NEVER import or modify gui/, the training path, or engine rules. CPU only. Resources =
string literals. Generalize from the spike (palette/orientation/baseline/color->player must be
per-game, NOT hardcoded).
OPPONENT-STRENGTH IS NOW SOLVED: the committed strength manifest at data/human/strength_manifest.json
(built this session: 814 videos labeled high/unknown/excluded; 204 high = top-200 world rank or a 1v1
tournament) is THE source of truth. The segment/record layer derives opponent_strength.tier from the
video's manifest entry. is_scoreboard_eligible requires tier=='high'. Do NOT invent a rank-OCR scheme
and do NOT use the old "known-high-rank-window" placeholder — read the manifest. See
scripts/build_strength_manifest.py + scripts/render_strength_manifest_md.py for its exact shape (fields:
strength, source in {tournament,ranked_rank,none}, evidence.rank). SCHEMA RECONCILIATION: record.py's
OpponentStrength currently has source Literal["rank_badge","known_window"] and tier Literal["high",
"unknown"] — reconcile it ADDITIVELY with the manifest vocabulary (map ranked_rank->rank_badge,
tournament->a new "tournament" literal; excluded videos NEVER become records so tier stays high|unknown),
updating record.py + its tests additively (never loosen/delete a test). Only 'high' feeds the scoreboard;
'high'+'unknown' feed the seed corpus; 'excluded' is dropped.
NEVER edit a committed golden fixture or loosen/delete a test to go green — if a test fails, FIX THE CODE.`

// Schemas: NO additionalProperties:false and minimal `required` (a prior run died when an agent
// malformed a strict/large StructuredOutput tool call; leniency avoids retry-cap deaths). Ask agents
// to INCLUDE KEY OUTPUT LINES rather than PASTE full logs (smaller payloads malform less).
const SLICE_OUT = { type: 'object', properties: {
  ready_claim: { type: 'boolean' }, summary: { type: 'string' },
  files_touched: { type: 'array', items: { type: 'string' } }, commit: { type: 'string' },
  tests_summary: { type: 'string' }, notes: { type: 'string' } },
  required: ['ready_claim', 'summary', 'commit'] }

const VERIFY = { type: 'object', properties: {
  tools_green: { type: 'boolean', description: 'ruff + mypy --strict + pytest all pass — RAN, not assumed' },
  gold_reproduced: { type: 'boolean', description: 'the committed game-1 golden fixture is reproduced exactly' },
  invariants_pass: { type: 'boolean', description: 'deterministic guards hold (multiset, 4+4, road-incidence, orientation stability, winner-in-handles-or-null, schema)' },
  fixture_untampered: { type: 'boolean', description: 'no committed golden fixture or test was edited/loosened/deleted to pass (check git diff)' },
  evidence: { type: 'string', description: 'the KEY pass/fail lines from ruff/mypy/pytest/gold-diff (not the full logs)' },
  blockers: { type: 'array', items: { type: 'string' } } },
  required: ['tools_green', 'gold_reproduced', 'invariants_pass', 'fixture_untampered'] }

const REVIEW = { type: 'object', properties: {
  findings: { type: 'array', items: { type: 'object', properties: {
    severity: { type: 'string', enum: ['BLOCKER', 'SHOULD-FIX', 'NIT'] }, title: { type: 'string' },
    where: { type: 'string' }, issue: { type: 'string' }, fix: { type: 'string' } },
    required: ['severity', 'title', 'issue'] } }, notes: { type: 'string' } },
  required: ['findings'] }

const REDTEAM = { type: 'object', properties: {
  broke: { type: 'boolean' }, checked: { type: 'string', description: 'what/how many games or cases checked' },
  counterexample: { type: 'string' } }, required: ['broke', 'checked'] }

const journal = []
function note(m) { log(m); journal.push(m) }

// A schema agent() can intermittently THROW (StructuredOutput tool-call malformation, ~5-10%/agent)
// which would otherwise kill the whole run. Retry up to 3x (the flake is per-invocation, so a fresh
// attempt almost always succeeds); return null only if all fail (callers already handle null).
const rawAgent = agent
async function safeAgent(prompt, opts) {
  for (let a = 0; a < 3; a++) {
    try {
      const r = await rawAgent(prompt, opts)
      if (r !== null && r !== undefined) return r
      note(`safeAgent ${opts && opts.label ? opts.label : '?'} attempt ${a}: null (agent died) -> retry`)
    } catch (e) {
      note(`safeAgent ${opts && opts.label ? opts.label : '?'} attempt ${a} threw: ${String(e).slice(0, 90)} -> retry`)
    }
  }
  return null
}

// ============================ the per-slice loop ============================
async function buildSlice(slice, phase) {
  let findings = []
  for (let iter = 0; iter < MAX_ITERS; iter++) {
    const tag = `${slice.module}#${iter}`
    const impl = await safeAgent(
      `${COMMON}\n\nSLICE: implement \`${slice.module}\`. ${slice.what}\nTEST-FIRST requirements: ${slice.tests}\n` +
      (iter > 0 ? `This is a RESOLVE pass — fix EXACTLY these open findings and nothing else:\n${JSON.stringify(findings, null, 2)}\n` : '') +
      `Re-green ruff + mypy --strict + pytest, then commit (conventional message, push origin/main). Report the commit sha.`,
      { label: `impl:${tag}`, phase, schema: SLICE_OUT, effort: 'high' })
    if (!impl) { note(`HALT ${slice.module}: implement agent died iter ${iter}`); return { module: slice.module, ready: false, halted: 'impl-died' } }

    const ver = await safeAgent(
      `${COMMON}\n\nINDEPENDENTLY VERIFY slice \`${slice.module}\` (do NOT trust the implementer). Actually RUN: \`ruff check\`, \`mypy --strict\` (project config), \`pytest\` on the new tests; reproduce the committed game-1 golden fixture exactly; run the deterministic invariants for this slice (${slice.invariants}). Check \`git diff\` for any edited/loosened/deleted golden fixture or test (that = fixture_untampered:false). Be adversarial; green only if truly green. Put the KEY pass/fail lines in evidence.`,
      { label: `verify:${tag}`, phase, schema: VERIFY, effort: 'high' })
    if (!ver) { note(`HALT ${slice.module}: verify agent died iter ${iter}`); return { module: slice.module, ready: false, halted: 'verify-died' } }
    if (ver.fixture_untampered === false) { note(`HALT ${slice.module}: CHEATING DETECTED (fixture/test tampered). evidence: ${ver.evidence}`); return { module: slice.module, ready: false, halted: 'cheating' } }
    if (!ver.tools_green || !ver.gold_reproduced || !ver.invariants_pass) {
      findings = (ver.blockers || ['verify failed']).map((b) => ({ severity: 'BLOCKER', title: 'verify failure', where: slice.module, issue: b, fix: 'make it pass for real' }))
      note(`${slice.module} iter ${iter}: verify RED -> resolve`); continue
    }

    // verify-only slices (e.g. the already-reviewed scaffold) are READY once green — no
    // lens/red-team loop (that thrashes on design trade-offs, not code bugs).
    if (slice.verifyOnly) { note(`READY ${slice.module} (verify-only, iter ${iter}, commit ${impl.commit})`); return { module: slice.module, ready: true, iters: iter + 1, commit: impl.commit, notes: impl.notes, shouldfix: [] } }

    const lenses = (await parallel([
      () => safeAgent(`${COMMON}\n\nREVIEW \`${slice.module}\` — LENS A DATA/MEASUREMENT CORRECTNESS. Read the actual files. Does it avoid every §5 trap relevant to this slice? Would it yield a dataset that is CONFIDENTLY WRONG (biased/mislabeled) rather than merely noisy? Severity-tag findings (BLOCKER/SHOULD-FIX/NIT) with concrete fixes.`,
        { label: `revA:${tag}`, phase, schema: REVIEW, effort: 'high' }),
      () => safeAgent(`${COMMON}\n\nREVIEW \`${slice.module}\` — LENS B CV/OCR ROBUSTNESS + SWE/ADDITIVITY. Read the actual files. Does it GENERALIZE beyond the single spike frame (per-game calibration, orientation/residual gates, cross-frame stability)? Compute/ETA realism, schema/fixture additivity, scope-lock (no engine/gui/training touch). Severity-tag findings with fixes.`,
        { label: `revB:${tag}`, phase, schema: REVIEW, effort: 'high' }),
    ])).filter(Boolean)
    const rt = await safeAgent(
      `${COMMON}\n\nRED-TEAM \`${slice.module}\`: your ONLY job is to find a case where its output is WRONG. Sample real games/frames, reconstruct the engine board from the record and compare to the source frame, probe the deterministic invariants and edge cases (occlusion, non-default palette, animation frames, video cutoff). Report broke=true + a concrete counterexample if you find one.`,
      { label: `redteam:${tag}`, phase, schema: REDTEAM, effort: 'high' })

    const blockers = lenses.flatMap((r) => r.findings || []).filter((f) => f.severity === 'BLOCKER')
    if (rt && rt.broke) blockers.push({ severity: 'BLOCKER', title: 'red-team counterexample', where: slice.module, issue: rt.counterexample, fix: 'handle this case' })
    const shouldfix = lenses.flatMap((r) => r.findings || []).filter((f) => f.severity === 'SHOULD-FIX')
    // BLOCKER always blocks; SHOULD-FIX gets ONE resolve attempt (iter 0) then becomes
    // ADVISORY (carried to the report, not looped) — prevents thrash on design trade-offs
    // that a resolve agent can never "fix" (they are choices, not bugs).
    const open = iter === 0 ? [...blockers, ...shouldfix] : blockers
    if (open.length === 0) { note(`READY ${slice.module} (iter ${iter}, commit ${impl.commit}, ${shouldfix.length} advisory should-fix)`); return { module: slice.module, ready: true, iters: iter + 1, commit: impl.commit, notes: impl.notes, shouldfix } }
    findings = open
    note(`${slice.module} iter ${iter}: ${blockers.length} BLOCKER + ${shouldfix.length} SHOULD-FIX -> resolve`)
  }
  note(`HALT ${slice.module}: BLOCKER still open after ${MAX_ITERS} iters (stuck)`); return { module: slice.module, ready: false, halted: 'stuck', remaining: findings }
}

async function runStage(phase, slices) {
  // BEST-EFFORT: build every module even if one breaks (each is tested against committed
  // fixtures, so they are independent). A blocked module is committed-skipped, never shipped;
  // the integration GATES below run only on the ready subset.
  const out = []
  for (const s of slices) out.push(await buildSlice(s, phase))
  const blocked = out.filter((r) => !r.ready).map((r) => r.module)
  return { allReady: blocked.length === 0, blocked, results: out }
}

// ============================ Stage 0: scaffold ============================
phase('Stage0 scaffold')
const scaffold = await buildSlice({
  module: 'scaffold',
  verifyOnly: true,  // a reviewed, green foundation — verify it's present + green, then proceed (no lens loop)
  what: `ACCEPTED DESIGN DECISIONS (do NOT re-litigate — a prior run thrashed 4 iters on these; they are choices, not bugs): (1) opponent_strength is deliberately ThePhantom's OWN top-200 matchmaking rank from the committed manifest (opponents are Elo-matched) — this is the user's explicit definition, not a gap; (2) the 15 tournament highs are title-keyword-labeled by design (his tournament titles are unambiguous) — acceptable; (3) the standard-board multiset gate is one of SEVERAL cross-checks (necessary-not-sufficient by design) — board_cv/validate add orientation + residual + cross-frame gates. Your job: VERIFY the foundation is present + green, nothing more.
The scaffold AND the orientation-bug fixes are ALREADY BUILT AND COMMITTED on main: the src/catan_rl/human_data/ package (record.py with the schema-v2 GameRecord + provenance ORIENTATION-BINDING validate(), topology.py/json, ffmpeg.py, orientation.py with the glyph-anchor check + scale-up gates), banked spike code at scripts/dev/human_data_spikes/, the CORRECTED game-1 golden fixtures (tests/fixtures/human_data/game1_openings.json is desert=11 with re-snapped IDs — NOT the old desert=17), and tests/unit/human_data/test_scaffold.py (59 passing). Also NEW this session: data/human/strength_manifest.json (the committed opponent-strength source). Your job is NOT to rebuild — VERIFY the foundation is present + correct (schema_version==2; validate() REJECTS a welded board=desert11/openings=desert17 record; the corrected fixture loads; the tests pass; ruff + mypy --strict clean; the strength manifest loads and has 814 videos incl. 'high' entries) and commit only anything genuinely missing. Then READY. Do NOT touch /tmp; do NOT revert the desert=11 fixture to desert=17.`,
  tests: `the existing tests/unit/human_data/ suite passes; schema_version==2; validate() rejects a welded desert11/desert17 record.`,
  invariants: `package imports without gui/torch; schema v2; provenance-binding firewall present; game-1 fixture is desert=11; strength_manifest.json present`,
}, 'Stage0 scaffold')

let report
if (!scaffold.ready) {
  note('Stage 0 failed — cannot scaffold; halting before any pipeline work.')
} else {
  // ===================== Stage 1: log spine =====================
  phase('Stage1 log-spine')
  const s1 = await runStage('Stage1 log-spine', [
    { module: 'ingest', what: 'yt-dlp 1080p download-then-delete per video (+retries/sleep/format-fallback, cap net concurrency 1-2, node satisfies nsig); two-pass frame sampling (sparse 1/3-5s + dense setup-window) piping frames in-memory (no PNG accumulation). Reuse scripts/build_strength_manifest.py ffmpeg/yt-dlp patterns.', tests: 'unit-test the sampling schedule math + a mocked-yt-dlp ingest path.', invariants: 'no PNG files written; ETA formula present' },
    { module: 'logparse', what: 'crop+easyocr the log + a Colonist log-grammar parser -> ordered event stream + winner (from the victory LOG line ONLY; null on resign/cutoff; the top-left counter is NOT the score).', tests: 'unit-test the grammar against the committed REAL noisy ocr_*.txt incl. the "Happy settlingl" typo; a test that winner is one-of-two-handles-or-null.', invariants: 'winner provenance = victory line; counter never used as score' },
    { module: 'segment', what: 'game-boundary detection ("Happy settling" reset / victory line) for the MANY back-to-back games per video; ruleset filter (exactly 2 actors, 15VP terminal, no P2P); opponent_strength.tier read from data/human/strength_manifest.json by video_id (high|unknown|excluded) — NOT a placeholder.', tests: 'unit-test boundary + ruleset filter on a synthetic multi-game event log; a test that opponent_strength.tier comes from the manifest.', invariants: '>2 actors rejected; 10VP/P2P rejected; tier sourced from manifest' },
  ])
  // Stage 1 gate: run on 5 real games, check log/winner accuracy
  let g1 = { ok: false, skipped: true }
  if (s1.allReady) {
    const gate = await safeAgent(`${COMMON}\n\nSTAGE-1 GATE: run the Stage-1 pipeline (ingest+logparse+segment) on 5 real high-rank ThePhantom games end-to-end (pick 'high' videos from data/human/strength_manifest.json). Report whether games segment correctly, the winner reads from the victory line, ruleset filter fires, opponent_strength.tier matches the manifest, and log accuracy looks right. Put evidence in the evidence field. Return tools_green=overall-pass, gold_reproduced (winner correct on a known game), invariants_pass, fixture_untampered, evidence, blockers.`, { label: 'gate:stage1', phase: 'Stage1 log-spine', schema: VERIFY, effort: 'high' })
    g1 = { ok: !!gate && gate.tools_green && gate.invariants_pass && gate.fixture_untampered, gate }
    note(g1.ok ? 'Stage-1 GATE PASSED' : `Stage-1 GATE FAILED: ${gate ? gate.blockers : 'gate died'}`)
  } else note(`Stage-1 GATE SKIPPED — blocked modules: ${s1.blocked.join(',')}`)

  // Stage 2 builds REGARDLESS of the Stage-1 gate (modules are fixture-independent);
  // only the integration tiers below are gated on the ready subset.
  {
    // ===================== Stage 2: board + openings =====================
    phase('Stage2 board+openings')
    const s2 = await runStage('Stage2 board+openings', [
      { module: 'board_cv', what: 'orientation-LOCKED lattice fit (screen-space rule + >=2 OCR anchors, NOT desert; residual gate; cross-frame stability) + per-game-calibrated resource/number/pip. Generalize palettes. Reuse the banked board-CV spike.', tests: 'reproduce the game-1 board map (19/19) byte-identical across >=2 committed frames; reject a deliberately mis-oriented fit.', invariants: 'standard resource+number multiset; cross-frame board-map stability; desert never used to orient' },
      { module: 'openings', what: 'post-setup frame select + empty-baseline (from the log, with an empty-board assertion) green-subtraction + piece detect + snap to vertex/edge + road tiebreak + player->color from per-game HUD. Reuse the banked opening-CV spike.', tests: 'reproduce the committed game1_openings.json EXACTLY (desert=11 with re-snapped IDs — do NOT regress to old desert=17 IDs).', invariants: '4+4 placements; 2+2 per player; road snap-incident to owner settlement; colors match HUD; openings_desert_hex recorded' },
      { module: 'validate', what: 'cross-check gate building on record.py v2: the ORIENTATION firewall is the provenance-binding board_desert_hex==openings_desert_hex (D6-invariant checks like road-incidence are SANITY-ONLY, never the orientation gate); plus standard resource/number multisets, cross-frame board stability, winner-in-handles, resolution>=1080, residual<=5px; reject-on-disagreement; EMIT rejected games + rejection_reason for the bias audit.', tests: 'gate accepts game-1; REJECTS a welded desert11/desert17 record; road-incidence proven D6-invariant (sanity-only).', invariants: 'orientation caught by provenance-binding NOT road-incidence; rejected records carry reason+features' },
      { module: 'glyph_anchor', what: 'build the orientation-INDEPENDENT firewall that unblocks scale-up (orientation.py already has the CHECK + a scale-up gate that currently raises GlyphClassifierNotValidated): grab a frame right AFTER each player "received starting resources" log event, color-classify the granted resource CARD-GLYPHS, and assert the 2nd-settlement adjacent-hex resource multiset (under the chosen affine) == the granted cards. BEST-EFFORT: if the glyphs cannot be classified reliably, leave the scale-up gate engaged (harvest stays blocked) + report exactly why — never fake it validated.', tests: 'glyph classify accept/reject on a real post-grant frame; the scale-up gate flips to allowed ONLY when validated.', invariants: 'orientation-independent; scale-up gate stays engaged unless truly validated' },
      { module: 'batch', what: 'parallel-per-video orchestration reading data/human/strength_manifest.json (harvest high + unknown for seeds; scoreboard = high only), per-(video_id,game_index) ledger {status,error,ts}, atomic appends, resumable; writes the JSONL corpus + a rejected.jsonl.', tests: 'unit-test resume skips done + retries transient; kill-and-resume leaves no dup/corruption.', invariants: 'idempotent; atomic appends; ledger consulted; manifest tier attached' },
    ])
    let g2 = { ok: false, skipped: true }
    if (s1.allReady && g1.ok && s2.allReady) {
      // tiered run: 5-game integration then 30-game gold gate
      const t5 = await safeAgent(`${COMMON}\n\nTIER-5 INTEGRATION: run the FULL pipeline (download->log->board->openings->validate->batch) on 5 real high-rank games end-to-end. Confirm it produces well-formed GameRecords + a rejected.jsonl, no crashes, invariants hold, opponent_strength.tier attached from the manifest. Put evidence in the evidence field. Return the VERIFY schema.`, { label: 'tier:5', phase: 'Stage2 board+openings', schema: VERIFY, effort: 'high' })
      if (t5 && t5.tools_green && t5.invariants_pass && t5.fixture_untampered) {
        const gold = await safeAgent(`${COMMON}\n\nGOLD GATE: hand-verify the pipeline on ~30 games against the PRE-REGISTERED bars (board layout >=98%, openings >=95%, winner ~100%) using the committed gold set + spot-rendering record-vs-frame. ALSO run the rejection-bias audit (per-archetype acceptance rate) and report it. Return VERIFY (tools_green=bars-met) + put the audit + per-field accuracy in evidence.`, { label: 'gate:gold', phase: 'Stage2 board+openings', schema: VERIFY, effort: 'high' })
        g2 = { ok: !!gold && gold.tools_green && gold.invariants_pass && gold.fixture_untampered, t5, gold }
        note(g2.ok ? 'Stage-2 GOLD GATE PASSED' : `Stage-2 GOLD GATE FAILED: ${gold ? gold.blockers : 'gate died'}`)
      } else { note(`Tier-5 integration FAILED: ${t5 ? t5.blockers : 'died'}`); g2 = { ok: false, t5 } }
    } else note(`Stage-2 GATES SKIPPED — s1Ready=${s1.allReady} s1Gate=${g1.ok} s2blocked=[${s2.blocked.join(',')}]`)
    // FULL HARVEST gated on BOTH the gold gate AND the glyph firewall (the orientation-
    // independent joint-flip catch). Both green -> harvest overnight (resumable, firewall-
    // protected). Else halt + report — never scale unprotected.
    let harvest = { ran: false }
    const glyphReady = s2.results.find((r) => r.module === 'glyph_anchor')?.ready === true
    if (g2.ok && glyphReady) {
      phase('Stage3 harvest')
      const h = await safeAgent(`${COMMON}\n\nFULL HARVEST: the gold gate PASSED and the glyph firewall is validated, so scaling is safe. Run scripts/mine_phantom.py over the manifest 'high' (scoreboard) + 'unknown' (seed) 1v1 corpus via batch.py — resumable (per-(video,game) ledger, atomic appends), firewall-protected (provenance-binding + cross-check + glyph anchor reject bad games). Process as many games as you can (resumable; partial is fine). Report #games harvested into the JSONL, #rejected + the rejection-bias audit (per-archetype acceptance rate), and the resume command. Return tools_green=ran-without-crash, gold_reproduced=true, invariants_pass, fixture_untampered, evidence(counts+audit), blockers.`, { label: 'harvest:full', phase: 'Stage3 harvest', schema: VERIFY, effort: 'high' })
      harvest = { ran: !!h, result: h }
      note(h ? 'FULL HARVEST ran (gold + glyph green) — counts in evidence' : 'FULL HARVEST agent died')
    } else {
      note(`FULL HARVEST GATED — gold=${g2.ok} glyph=${glyphReady}. Pipeline built+validated; corpus harvest needs BOTH the gold gate AND the glyph firewall green. Not scaling unprotected.`)
    }
    var stage2 = { s2, g2, harvest }
  }
  var stage1 = { s1, g1 }
}

// ===================== Morning report =====================
phase('Morning report')
report = await safeAgent(
  `Write the MORNING REPORT for an autonomous overnight build of the human_data opening-extraction pipeline (the human reads this at 8am; they did NOT watch, and they explicitly asked me to build this phase overnight while they slept). Be honest and concrete. Use the run journal + stage results below. State: what shipped (which modules are READY + commits), every GATE result with numbers, the rejection-bias audit if reached, exactly WHERE it halted and WHY, the single-command resume state, the HARVEST status (full corpus harvested — with #games + #rejected + the rejection-bias audit — OR gated, saying exactly which gate and what is needed to unblock it), and the human-decision items (esp. the glyph-classifier status, which hard-gates the safe full harvest). Also write this report to docs/plans/human_data_overnight_report.md and commit+push it. Do NOT claim success that the gates did not show.\n\nJOURNAL:\n${JSON.stringify(journal, null, 2)}\n\nSTAGE RESULTS:\n${JSON.stringify({ scaffold, stage1: typeof stage1 !== 'undefined' ? stage1 : null, stage2: typeof stage2 !== 'undefined' ? stage2 : null }, null, 2)}`,
  { label: 'morning-report', phase: 'Morning report', effort: 'high' })

return { halted_summary: journal.filter((m) => m.includes('HALT')), report }
