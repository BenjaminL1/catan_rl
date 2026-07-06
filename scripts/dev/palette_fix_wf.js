export const meta = {
  name: 'palette-fix',
  description: 'Fix the Tier-5 NO-GO blocker (color reader knows GREEN+BLACK only) all on OPUS: a DATA-DRIVEN color survey measuring real Colonist seat/piece HSV from actual footage, wire the measured ranges into PALETTE/_HUD_RING fail-closed, harden green-piece suppression + Stage-1 segmentation welding, then RE-RUN Tier-5 to test acceptance vs the 0/3 baseline and run the gold gate if >=30 games are reached. Ends at a report — the harvest go/no-go stays the user’s.',
  phases: [
    { title: 'color_survey' },
    { title: 'palette_wire' },
    { title: 'green_seg' },
    { title: 'tier5_rerun' },
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

const COMMON = `You are fixing the DOMINANT blocker that made the Tier-5 harvest a NO-GO
(catan_rl_v2 human_data pipeline). CONTEXT: the e2e harvest driver runs, but Tier-5 accepted
0/3 real ThePhantom games — ALL rejected \`hud_unreadable\` because the opponent-colour reader
(\`src/catan_rl/human_data/openings.py\`: the \`PALETTE\` piece-profile table + the \`_HUD_RING\`
HUD seat-ring table) only has GREEN + BLACK calibrated (the one spike game's seats). Real
Colonist opponents are overwhelmingly other colours (red/blue/orange/white/etc.), so
\`read_hud_seat_colors\` fails the two-distinct-colours vote and the game rejects before anything
downstream. Design authority: docs/plans/human_data_pipeline.md (§5 correctness traps, esp.
§5.14 HUD binding, §5.6 rejection-bias audit, §5.7 piece/road) + docs/plans/tier5_report.md
(the NO-GO evidence). Full pipeline exists + is hardened; the glyph firewall is validated +
fingerprint-gated (do NOT touch it — glyph_anchor.py constants are frozen). HARD RULES: ruff +
mypy (PROJECT config: \`make lint\` / \`make typecheck\`, i.e. \`mypy src/catan_rl\` — NOT ad-hoc
--strict) + pytest green before every commit; conventional commits; push origin/main. NEVER
loosen/delete a committed test or golden fixture. CPU only; no \`gui/\` import; yt-dlp +
imageio-ffmpeg are the frame tools (see scripts/build_strength_manifest.py + scripts/glyph_valset.py
+ src/catan_rl/human_data/ingest.py|ffmpeg.py for working patterns; net concurrency 1-2;
download-then-delete, no PNG accumulation beyond committed artifacts). FAIL-CLOSED IS SACRED: a
seat colour NOT in the (extended) palette must still emit a TYPED rejection, never a silent
mislabel — widening coverage must not weaken the abstention guarantee. Manifest of 204 'high'
video ids: data/human/strength_manifest.json (m['videos'], each has video_id + label=='high').
Reusable cached frames: data/human/glyph_valset/*.png (+ meta.jsonl: video_id/t/boxes) show the
HUD during play; data/human/frames/*_rank.png are leaderboard thumbnails (NOT gameplay).`

const VERIFY_FMT = `Return MARKDOWN only. Sections: TOOLS (commands RUN + key lines), CHECKS
(each invariant: pass/fail + evidence), DATA (for measurement/run slices: open the artifact,
recompute a number), TAMPER (git diff on tests/fixtures), BLOCKERS. End with GREEN or RED on
its own line.`
const REVIEW_FMT = `Return MARKDOWN only: "### <BLOCKER|SHOULD-FIX|NIT>: title" with
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
      `Re-green lint/typecheck/pytest, commit, push. Report commit sha + summary + the key numbers.`,
      { label: `impl:${tag}`, phase: slice.name, effort: 'high', model: MODEL })
    if (!impl) { note(`HALT ${slice.name}: impl died`); return { name: slice.name, ready: false } }

    const ver = await safeAgent(
      `${COMMON}\n\nINDEPENDENTLY VERIFY slice ${slice.name} (do NOT trust the implementer). RUN
the tools; check the slice invariants: ${slice.invariants}. ${VERIFY_FMT}`,
      { label: `verify:${tag}`, phase: slice.name, effort: 'high', model: MODEL })
    const verText = ver || ''
    if (/\bRED\b\s*$/.test(verText.trim()) || !/\bGREEN\b/.test(verText)) {
      findings = [{ severity: 'BLOCKER', title: 'verify RED', issue: verText.slice(-1500), fix: 'make it genuinely green' }]
      note(`${slice.name} iter ${iter}: verify RED -> resolve`); continue
    }

    const review = await safeAgent(
      `${COMMON}\n\nREVIEW slice ${slice.name} against its intent: ${slice.what}\nRead the actual
diff/files/artifacts. Hunt confidently-wrong paths: INVENTED (not measured-from-footage) HSV
ranges; OVERLAPPING colour ranges that make two seats indistinguishable; a widened palette that
silently accepts an out-of-gamut colour instead of rejecting typed; a green-hardening change that
now over-accepts green tiles as pieces; a segmentation change that drops real games. ${REVIEW_FMT}`,
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

// ---------- slice 1: data-driven colour survey (keystone) ----------
phase('color_survey')
const survey = await buildSlice({
  name: 'color_survey',
  what: `MEASURE the real Colonist colour palette from ACTUAL footage — do NOT invent HSV values.
(a) Sample gameplay frames across a SPREAD of >=15 'high' manifest videos (reuse
data/human/glyph_valset/*.png where they show the HUD; extract additional HUD+board frames from
other high videos via the ingest/ffmpeg path). Aim to observe the full opponent-colour set that
actually appears (the report notes non-green dominates; ThePhantom self-seats one fixed colour).
(b) For each seat in each sampled game measure BOTH (i) the HUD seat-avatar RING dominant
saturated HSV (what read_hud_seat_colors keys on) and (ii) a placed PIECE (settlement/road) body
HSV — these can differ and BOTH feed the reader. (c) Cluster the measured samples into distinct
colour identities; derive per-identity NON-OVERLAPPING HSV ranges (hue/sat/val lo-hi) for ring
AND piece, each range justified by its measured sample spread (report n_samples + spread per
colour). (d) Identify which colours collide with a same-hue board TILE (green vs forest/pasture;
check others) — these need the tile_subtract flag. (e) Flag any colour seen in too few games to
calibrate confidently (mark for harvest-exclusion, not a guessed range). COMMIT
data/human/color_survey.json (raw per-video/per-seat measurements + derived ranges + tile-collision
flags + low-sample flags) and data/human/color_survey.md (colour histogram, per-colour n + spread,
the non-overlap proof, excluded colours).`,
  tests: `a small unit test that the derived ranges in color_survey.json are pairwise
NON-OVERLAPPING for ring and for piece (two seats always separable); the artifact is
well-formed + regenerable; existing suite green. (The measurement run itself is the evidence —
the .md must contain real per-colour n and HSV numbers, not placeholders.)`,
  invariants: `every range traces to measured pixels (raw samples in the json, not hand-typed);
ranges pairwise non-overlapping; low-sample colours flagged not guessed; tile-collision colours
identified from data; no classifier constant changed yet (this slice only PRODUCES the artifact).`,
})

// ---------- slice 2: wire the measured ranges ----------
let wire = { name: 'palette_wire', ready: false }
if (survey.ready) {
  phase('palette_wire')
  wire = await buildSlice({
    name: 'palette_wire',
    what: `Wire the color_survey.json measured ranges into openings.py: extend PALETTE (piece
ColorProfile per colour) and _HUD_RING (ring profile per colour) to cover every calibrated
(non-low-sample) survey colour, setting tile_subtract from the survey's tile-collision flags
(only same-hue-tile colours get it — do NOT blanket-enable it). Keep GREEN + BLACK exactly as
they are unless the survey shows their committed ranges are wrong (if so, justify in the commit).
Preserve fail-closed: a seat colour outside the extended palette still returns the typed
player_colors_invalid / hud_unreadable reason (add a unit test proving an out-of-gamut colour
still rejects). Update the module docstring's 'Palette precondition' paragraph to reflect the
widened coverage (doc-sync).`,
    tests: `each newly-added colour round-trips through detect_openings_result on a synthetic
frame painted at that colour's measured HSV (settlement+road detected, correct handle binding);
GREEN still handled incl. the green-tile-subtraction path; an out-of-gamut colour STILL rejects
with a typed reason (fail-closed regression); PALETTE/_HUD_RING key sets match the calibrated
colours in color_survey.json.`,
    invariants: `no silent mislabel (fail-closed intact); tile_subtract set only for surveyed
tile-collision colours; ranges byte-match color_survey.json (single source of truth); docstring
updated; GREEN/BLACK behaviour unchanged unless justified.`,
  })
}

// ---------- slice 3: secondary blockers ----------
let greenseg = { name: 'green_seg', ready: false }
if (wire.ready) {
  phase('green_seg')
  greenseg = await buildSlice({
    name: 'green_seg',
    what: `Harden the two SECONDARY blockers the Tier-5 report named. (a) GREEN-PIECE
SUPPRESSION (§5.6): the empty-baseline green-tile subtraction can eat a REAL green settlement
sitting on a green tile. Tighten the subtraction (smaller/targeted kernel or a piece-vs-tile
saturation/shape discriminator) so real green pieces survive while green TILES are still killed;
keep the :green_tile_suppressed typed reason for genuine, unrecoverable suppression (do not paper
over it). (b) STAGE-1 SEGMENTATION welding on noisy OCR (segment.py): distinct games get welded
when the setup-boundary OCR is noisy. Harden boundary detection so noisy-OCR games SPLIT
correctly, WITHOUT reintroducing the ambiguous-reset over-split the user chose to merge-away
earlier (respect test_missing_terminal_then_identical_reset_merges_not_split). Scope tightly to
these two — do not refactor unrelated CV.`,
    tests: `regression: a synthetic green settlement on a green tile is now DETECTED (was
suppressed); a synthetic noisy-OCR two-game log SPLITS into two games; the existing
merge-not-split test still passes (no over-split regression); suite green.`,
    invariants: `green fix does not now over-accept green tiles as pieces (add the false-positive
guard test); segmentation fix preserves the merge-favouring decision on ambiguous resets; no
change to accepted-game semantics beyond recovering the two blocked cases.`,
  })
}

// ---------- slice 4: re-run Tier-5 + conditional gold gate ----------
phase('tier5_rerun')
let rerun = null, goldResult = null
if (greenseg.ready) {
  rerun = await safeAgent(
    `${COMMON}\n\nRE-RUN TIER-5 with the palette+hardening fixes in place. (a) Run the harvest
driver first on the ORIGINAL 5 Tier-5 videos (from docs/plans/tier5_report.md) for a direct
comparison to the 0/3 baseline. (b) Then EXPAND: run additional 'high' manifest videos until you
reach >=30 ACCEPTED games OR you have run 25 videos total (whichever first) — time is not a
constraint, yield is. Serial or few workers, CPU. (c) Write docs/plans/tier5_rerun_report.md:
per-video accept/reject table (typed reasons), the NEW telemetry (games_seen/accepted/rejected-
by-reason/anchor_ran/order_unestablished/grant coverage), the ACCEPTANCE YIELD (accepted /
games_seen) with a Wilson CI, the NEW dominant rejection reason if any (what's the next blocker
now that colour is fixed?), measured wall-clock, and the updated 204-video harvest ETA + expected
accepted-corpus size at this yield. Commit the report + the accepted-games corpus sample
(data/human/tier5_rerun_corpus.jsonl if <500KB else gitignore + report the counts). Return
MARKDOWN: the accept/reject table, yield+CI, the new-blocker finding, and how many accepted games
you reached.`,
    { label: 'tier5:rerun', phase: 'tier5_rerun', effort: 'high', model: MODEL })
  note(`tier5 rerun done: ${(rerun || '').slice(0, 120)}`)

  // conditional gold gate — only if the re-run reached enough accepted games
  const packs = await safeAgent(
    `${COMMON}\n\nHow many ACCEPTED games did the Tier-5 re-run produce (count records in
data/human/tier5_rerun_corpus.jsonl or the ledger corpus that are accepted, not rejected)?
Return ONLY that integer, nothing else.`,
    { label: 'gold:count', phase: 'tier5_rerun', effort: 'low', model: MODEL })
  const nAcc = parseInt((packs || '0').replace(/[^0-9]/g, '') || '0', 10)
  note(`accepted games available for gold gate: ${nAcc}`)

  if (nAcc >= 30) {
    // prepare blind packets, then parallel blind labelers, then score (reuse gold_gate.py)
    const prep = await safeAgent(
      `${COMMON}\n\nRun 'python scripts/gold_gate.py prepare' to build BLIND-LABELING packets for
30 of the accepted games under data/human/gold/<game_id>/ (frames + reference grid + blank
template; NOTHING derived from the pipeline's parse of that game). Then run
'ls -d data/human/gold/*/' and return ONLY a JSON array of the packet directory paths.`,
      { label: 'gold:prep', phase: 'tier5_rerun', effort: 'high', model: MODEL })
    let dirs = []
    try { dirs = JSON.parse((prep || '[]').replace(/```json|```/g, '').trim()) } catch { dirs = [] }
    note(`gold packets prepared: ${dirs.length}`)
    if (dirs.length >= 30) {
      const CH = 5
      const groups = []
      for (let i = 0; i < dirs.length; i += CH) groups.push(dirs.slice(i, i + CH))
      const labeled = await parallel(groups.map((g, gi) => () => safeAgent(
        `You are a BLIND LABELER for a Catan video-parsing gold set. For each packet dir in
${JSON.stringify(g)}: Read ONLY the packet images (post-setup board frame, log crops, reference
grid) + the blank template. Fill it EXACTLY: board resource+number per engine hex id; each
player's 2 settlements + 2 roads as engine ids in first/second placement order (from the log
sequence in the crops); winner from the victory log line only (null if none). Write
<packet_dir>/labels_filled.json. You must NOT open any pipeline output — no *.jsonl corpora, no
src/ parsing code, no record files. Unreadable field => "SKIP". Your labels are the answer key.
Report which files you wrote.`,
        { label: `gold:label:${gi}`, phase: 'tier5_rerun', effort: 'high', model: MODEL })))
      note(`labelers done: ${labeled.filter(Boolean).length}/${groups.length}`)
      goldResult = await safeAgent(
        `${COMMON}\n\nRun 'python scripts/gold_gate.py score' (labels in
data/human/gold/*/labels_filled.json). Read docs/plans/gold_gate_report.md, recompute accuracy
by hand for 3 games to confirm, commit + push the report + labels. Return MARKDOWN: the per-field
accuracy table vs the pre-registered bars (board >=98%, openings >=95%, winner ~100%, flips=0),
the PASS/FAIL verdict per bar, and any caveats.`,
        { label: 'gold:score', phase: 'tier5_rerun', effort: 'high', model: MODEL })
    } else {
      note(`gold gate skipped: only ${dirs.length} packets prepared`)
    }
  } else {
    note(`gold gate NOT run: ${nAcc} accepted games (< 30 floor) — re-run report stands alone`)
  }
}

phase('tier5_rerun')
const summary = await safeAgent(
  `Write the completion summary (markdown) for the developer: slice verdicts + commit shas (check
git log), then THE HEADLINE — did the palette fix unblock acceptance? (Tier-5 re-run: accepted/
games_seen + yield CI vs the 0/3 baseline), the NEW dominant blocker if any, the gold-gate result
if it ran (vs the pre-registered bars) or why it didn't, the updated harvest ETA + expected
accepted-corpus size, and an explicit HARVEST GO/NO-GO recommendation line (the launch call is the
user's). Journal:\n${JSON.stringify(journal, null, 2)}\nRe-run result:\n${rerun || '(did not run)'}\nGold result:\n${goldResult || '(did not run)'}`,
  { label: 'summary', phase: 'tier5_rerun', effort: 'medium', model: MODEL })

return {
  results: [survey, wire, greenseg].map(r => ({ name: r.name, ready: r.ready })),
  halted: journal.filter(m => m.includes('HALT')),
  rerun,
  gold: goldResult,
  summary,
}
