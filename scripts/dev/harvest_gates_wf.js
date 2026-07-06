export const meta = {
  name: 'harvest-gates',
  description: 'The parser final exams before the 204-video harvest, all agents on OPUS: wire the e2e harvest driver (mine_phantom harvest), Tier-5 integration on 5 real videos + joint-D6 flip sweep, old-UI probe + glyph valset extension with independent double-labeling, then the 30-game GOLD GATE with blind labeling. Ends at the gold-gate report — the harvest go/no-go decision belongs to the user.',
  phases: [
    { title: 'harvest_driver' },
    { title: 'tier5' },
    { title: 'olduiprobe' },
    { title: 'goldgate' },
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

const COMMON = `You are completing the human_data pipeline's FINAL GATES before the 204-video
harvest (catan_rl_v2). Design authority: docs/plans/human_data_pipeline.md (§5 correctness traps,
the gold bars) + docs/plans/v2/step6_human_corpus.md v5.2 (§3.1 placement-order contract — ALREADY
IMPLEMENTED in record.py/commit fc83e6d). ALL pipeline modules exist and are hardened:
src/catan_rl/human_data/{ingest,logparse,segment,board_cv,openings,validate,orientation,
glyph_anchor,engine_bridge,opening_archetypes,record,batch,topology,ffmpeg}.py. The glyph
classifier is VALIDATED (data/human/glyph_validation.json, fingerprint-gated via
load_glyph_validation) and the joint-flip firewall is fail-closed (unreadable grant => typed
reject; anchor must run for acceptance). The strength manifest (data/human/strength_manifest.json,
human-verified) lists the 204 'high' videos. batch.py has the ledger/sink primitives;
scripts/mine_phantom.py currently has only ingest/batch-plan subcommands — the e2e driver is the
missing piece. HARD RULES: ruff + mypy (PROJECT config, i.e. \`make typecheck\` / \`mypy
src/catan_rl\` — not ad-hoc --strict) + pytest green before every commit; conventional commits;
push origin/main. NEVER loosen/delete a committed test or golden fixture. CPU only; no gui/ import
in any pipeline path; yt-dlp + imageio-ffmpeg are the network/frame tools (see
scripts/build_strength_manifest.py + scripts/glyph_valset.py for working patterns; cap network
concurrency at 1-2; download-then-delete per video, no PNG accumulation beyond committed
artifacts). Rejected games carry typed rejection_reason — never silently dropped.`

const VERIFY_FMT = `Return MARKDOWN only. Sections: TOOLS (commands you RAN + key lines), CHECKS
(each invariant: pass/fail + evidence), TAMPER (git diff on tests/fixtures), BLOCKERS. End with
GREEN or RED on its own line.`
const REVIEW_FMT = `Return MARKDOWN only: "### <BLOCKER|SHOULD-FIX|NIT>: title" findings with
where/issue/fix, then VERDICT: READY or NOT-READY.`

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
      `${COMMON}\n\nINDEPENDENTLY VERIFY slice ${slice.name} (do NOT trust the implementer). RUN
the tools; check the slice invariants: ${slice.invariants}. For slices that produced DATA
(tier5/olduiprobe/goldgate), spot-check the data itself (open records/frames, recompute a number).
${VERIFY_FMT}`,
      { label: `verify:${tag}`, phase: slice.name, effort: 'high', model: MODEL })
    const verText = ver || ''
    if (/\bRED\b\s*$/.test(verText.trim()) || !/\bGREEN\b/.test(verText)) {
      findings = [{ severity: 'BLOCKER', title: 'verify RED', issue: verText.slice(-1500), fix: 'make it genuinely green' }]
      note(`${slice.name} iter ${iter}: verify RED -> resolve`); continue
    }

    const review = await safeAgent(
      `${COMMON}\n\nREVIEW slice ${slice.name} against its intent: ${slice.what}\nRead the actual
diff/files/artifacts. Hunt confidently-wrong paths: silent accepts, untyped rejections, order/
consensus/firewall bypasses, biased sampling, blindness violations (gold labels seeing pipeline
output). ${REVIEW_FMT}`,
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

// ---------- slice 1+2: driver, then tier-5 ----------
phase('harvest_driver')
const driver = await buildSlice({
  name: 'harvest_driver',
  what: `Wire the E2E HARVEST DRIVER: a 'harvest' subcommand in scripts/mine_phantom.py that,
given video ids (or --manifest high|high+unknown, --limit), runs the FULL per-game chain:
ingest (download-then-delete, two-pass frame sampling) -> logparse (events + winner) -> segment
(game boundaries + ruleset filter + manifest strength tier) -> board_cv (orientation-locked board,
cross-frame stable) -> openings (piece detect + snap + HUD colors) -> PLACEMENT-ORDER establishment
from the log setup sequence (record.py contract, commit fc83e6d; unestablished => flag per
contract) -> glyph anchor with MULTI-FRAME CONSENSUS (group grant-line reads by (handle,
normalised line text) across the line's ~30s on-screen lifetime, re-detect boxes per frame, feed
>=2 frames to consensus_granted_glyphs; consult the fingerprint-gated validation via
load_glyph_validation + assert_scale_up_orientation_gates ONCE per run) -> validate/cross-check
(typed rejects) -> GameRecord -> batch ledger (resumable, atomic appends, corpus JSONL +
rejected.jsonl). Telemetry counters printed + written per run: {games_seen, accepted, rejected
by reason, anchor_ran, anchor_unreadable, anchor_mismatch, order_unestablished, grant_read
coverage}. Respect all fail-closed semantics — never accept without the anchor having run.`,
  tests: `unit tests for the driver's game-loop glue with mocked stages (accept path, each typed
reject path, anchor-unreadable => reject, order-unestablished => flagged-not-scoreboard);
ledger resume test; the full existing suite stays green.`,
  invariants: `no acceptance without anchor_ran; every rejection typed; scale-up gate consulted
via the fingerprinted loader (a tampered validation JSON must hard-block); ledger idempotent.`,
})

let tier5 = { name: 'tier5', ready: false }
if (driver.ready) {
  phase('tier5')
  tier5 = await buildSlice({
    name: 'tier5',
    what: `TIER-5 INTEGRATION: run the harvest driver end-to-end on 5 REAL 'high' manifest videos
(pick modern-UI ones with known-good frames, e.g. from data/human/glyph_valset/meta.jsonl video
ids). Produce the corpus JSONL + rejected.jsonl + telemetry for those videos. THEN run the
JOINT-D6 FLIP SWEEP on every ACCEPTED game: apply each of the 11 non-identity D6 elements jointly
to (board, openings) and count how many assert_glyph_anchor calls reject, under (i) the current
either-settlement matching and (ii) 2nd-settlement-only matching (the record now knows order) —
this measures the joint-flip leakage residual the plan's §joint-flip question needs. Also assert
>=2 readable grant frames per granting player (consensus supply) and report per-game grant
coverage. Write docs/plans/tier5_report.md with: per-video accept/reject table (typed reasons),
telemetry, the flip-sweep leakage numbers (x/55 per matching mode), consensus-supply stats,
wall-clock per video (the harvest ETA input). Commit the report + the 5-video corpus sample
(data/human/tier5_sample.jsonl if <500KB, else gitignore + report counts).`,
    tests: `the run itself is the evidence — the report must contain real numbers; spot-verify one
accepted record's openings against its post-setup frame (Read the image) and its winner against
the log; flip-sweep numbers recomputed by the verifier on 2 games.`,
    invariants: `0 records accepted without anchor_ran; all rejects typed; flip-sweep leakage
reported for BOTH matching modes; ETA formula stated with measured per-video wall-clock.`,
  })
}

// ---------- slice 3: old-UI probe + valset extension ----------
phase('olduiprobe')
const oldui = await buildSlice({
  name: 'olduiprobe',
  what: `OLD-UI PROBE + GLYPH VALSET EXTENSION (expert round-3 item 6). (a) Using
scripts/glyph_valset.py extract (extend it if needed to accept explicit video ids), extract grant
events from 2-4 OLD-UI 'high' videos (known old-UI ids: EkmCZkOb2yM, A5atIV8ty9g; find 1-2 more
via data/human/frames/*_rank.png thumbnails if needed) PLUS ~30-40 additional frames from other
videos deliberately oversampling games whose grants include ORE and BRICK (the thin confusion
cells: currently only 9 ORE boxes). (b) LABELING with independence: TWO passes over the new
contact sheets by reading the sheet images and labeling each crop by CARD ART (tree/brick-rows/
sheaf/sheep/stones) — the implementer does pass 1; the VERIFIER agent independently does pass 2
without seeing pass-1 labels; crops where the passes disagree => SKIP. Merge into
data/human/glyph_valset/labels.json (additive). (c) Re-run 'python scripts/glyph_valset.py score'
— the fingerprint-gated validation must still PASS the user bar (>=0.98, zero ORE<->BRICK) on the
EXTENDED set, with the old-UI subset's per-box results broken out in the report. If old-UI frames
FAIL (icons differ across eras), do NOT tune constants blindly: report the failure with the
measured old-UI hue/sat values and mark the old-UI videos for exclusion-from-harvest pending a
palette decision — an honest negative is a pass for this slice.`,
  tests: `score output committed (data/human/glyph_validation.json regenerated + report md);
old-UI subset explicitly broken out; n_ORE and n_BRICK boxes materially increased (report before/
after counts); disagreement-SKIP count reported.`,
  invariants: `labels double-passed independently; no classifier constant changed in this slice
(measurement only — constant changes would void the fingerprint and need their own re-validation);
validation JSON regenerated through the frozen scorer.`,
})

// ---------- slice 4: gold-gate tooling ----------
phase('goldgate')
const goldTooling = await buildSlice({
  name: 'goldgate_tooling',
  what: `GOLD-GATE TOOLING (the 30-game exam apparatus; labeling happens in the next phase —
build the instruments only). (a) scripts/gold_gate.py 'prepare': select ~30 games — the Tier-5
accepted games plus enough additional 'high' videos run through the harvest driver to reach 30
(run them; this also grows the corpus sample). For each gold game, save a BLIND-LABELING packet
under data/human/gold/<game_id>/: the post-setup full frame PNG, 1-2 extra mid-setup frames, the
log-crop PNG(s) covering setup + the victory/terminal region, and a blank label template
(board: 19x resource+number by engine hex id under a stated orientation convention rendered as a
reference grid image WITHOUT the pipeline's answers; openings: settlements+roads per player with
first/second order; winner). The packet must contain NOTHING derived from the pipeline's parse of
that game (no record fields, no overlays) — blindness is the point. (b) 'score': compare
completed label files against the pipeline's records field-by-field -> per-field accuracy vs the
PRE-REGISTERED bars (board >=98% of hexes, openings >=95% of placements, winner ~100%,
orientation flips 0) + Wilson CIs + a verdict line, written to docs/plans/gold_gate_report.md.
(c) A reference-grid renderer that maps engine hex/vertex ids to a canonical image so labelers
can name ids without pipeline output (reuse the committed engine template geometry; matplotlib or
PIL, headless, NO gui/ import).`,
  tests: `prepare on 2 games produces complete blind packets (verifier confirms no pipeline-derived
content in them); score on a synthetic hand-made label file computes correct accuracies + verdict;
suite green.`,
  invariants: `packets blind; bars hard-coded to the pre-registered values; score handles partial/
skipped labels honestly (excluded from denominator, counted + reported).`,
})

// ---------- parallel blind labeling + final scoring ----------
let goldResult = null
if (goldTooling.ready) {
  const packs = await safeAgent(
    `${COMMON}\n\nList the prepared gold-game packet directories under data/human/gold/ (run
'ls -d data/human/gold/*/'). Return ONLY a JSON array of directory paths, nothing else.`,
    { label: 'gold:list', phase: 'goldgate', effort: 'low', model: MODEL })
  let dirs = []
  try { dirs = JSON.parse((packs || '[]').replace(/```json|```/g, '').trim()) } catch { dirs = [] }
  note(`gold packets: ${dirs.length}`)
  if (dirs.length >= 10) {
    const CH = 5 // games per labeler
    const groups = []
    for (let i = 0; i < dirs.length; i += CH) groups.push(dirs.slice(i, i + CH))
    const labeled = await parallel(groups.map((g, gi) => () => safeAgent(
      `You are a BLIND LABELER for a Catan video-parsing gold set. For each packet directory in
${JSON.stringify(g)}: Read the frame PNGs (the post-setup frame shows the full board; the log
crops show the setup sequence + terminal lines; the reference grid image maps positions to engine
ids). Fill the label template EXACTLY (board: resource+number per hex id; openings: each player's
settlements+roads as engine ids with first/second placement order from the log sequence; winner
from the victory log line only, null if none visible). Write each completed file as
<packet_dir>/labels_filled.json. You must NOT open, read, or grep any pipeline output (no
*.jsonl corpora, no src/ parsing code, no record files) — your only inputs are the packet images
+ template + reference grid. If a field is genuinely unreadable, write "SKIP" for it. Work
carefully; your labels are the exam's answer key. Report which files you wrote.`,
      { label: `gold:label:${gi}`, phase: 'goldgate', effort: 'high', model: MODEL })))
    note(`labelers done: ${labeled.filter(Boolean).length}/${groups.length}`)
    goldResult = await safeAgent(
      `${COMMON}\n\nRun the gold-gate scorer: 'python scripts/gold_gate.py score' (labels are in
data/human/gold/*/labels_filled.json). Then Read docs/plans/gold_gate_report.md and verify its
numbers by recomputing accuracy for 3 games by hand from the labels + records. Commit + push the
report and the label files. Return MARKDOWN: the per-field accuracy table, the bars, the VERDICT
(PASS/FAIL per bar), flip/orientation findings, and any caveats (skipped fields, packets without
labels).`,
      { label: 'gold:score', phase: 'goldgate', effort: 'high', model: MODEL })
  } else {
    note('HALT goldgate: fewer than 10 packets prepared — not enough for the gate')
  }
}

phase('goldgate')
const summary = await safeAgent(
  `Write the completion summary (markdown) for the developer: slice verdicts + commits (check git
log), Tier-5 numbers (accept/reject, flip-sweep leakage both modes, per-video wall-clock + the
204-video harvest ETA), old-UI probe verdict (does the glyph validation still pass on the extended
set; old-UI included or excluded), and THE GOLD-GATE RESULT vs the pre-registered bars with an
explicit "harvest go/no-go recommendation" line — the launch decision belongs to the user.
Journal:\n${JSON.stringify(journal, null, 2)}\nGold scoring result:\n${goldResult || '(gold gate did not complete)'}`,
  { label: 'summary', phase: 'goldgate', effort: 'medium', model: MODEL })

return {
  results: [driver, tier5, oldui, goldTooling].map(r => ({ name: r.name, ready: r.ready })),
  halted: journal.filter(m => m.includes('HALT')),
  gold: goldResult,
  summary,
}
