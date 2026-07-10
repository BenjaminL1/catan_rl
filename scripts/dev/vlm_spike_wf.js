export const meta = {
  name: 'vlm-spike',
  description: 'Timeboxed decision experiment (Track 2) all on OPUS: does a VLM localizer beat the brittle blob detector on the SAME frames the pipeline already extracts? Build scripts/vlm_spike.py (post-setup + empty-baseline frame extraction reuse → pluggable VLM localizer → deterministic adjacency→vertex/edge snap → existing fail-closed validators), run REAL VLM localization (Opus vision as the VLM proxy) on game-1 (hard ground truth) + the Tier-5 videos (yield where classical CV got 0%), then re-run the gating-cell arithmetic with a PRE-REGISTERED continue/archive rule. Ends at a report + a GO/NO-GO recommendation — the decision is the user’s.',
  phases: [
    { title: 'harness' },
    { title: 'localize' },
    { title: 'score' },
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

const COMMON = `You are running a TIMEBOXED DECISION SPIKE for the catan_rl_v2 human-corpus program.
BACKGROUND: the classical-CV opening detector (openings.py blob detection) accepted 0/31 real games
in the Tier-5 re-run — the new wall is opening piece/road detection on real footage (settlement_ambiguous,
blob_shortfall, road_unresolved across colours). HYPOTHESIS: a VLM (vision-language model) can localize
settlements/roads from the SAME frames the pipeline already extracts, where blob detection fails.
THE HYBRID BEING TESTED (do not deviate): reuse the EXISTING frame extraction (the post-setup frame with
all 8 opening pieces down + the empty-baseline frame — board_cv/openings already produce both) → a VLM
localizes each settlement+road and describes its TILE ADJACENCY (which numbered/resource hexes each
settlement corner touches; which two vertices each road connects) → DETERMINISTIC geometry
(topology.load_topology(): vertex_adjacent_hexes, hex_corner_to_vertex, vertex_neighbors) snaps the
adjacency description to the exact engine vertex id (0-53) / edge id (0-71) → PLACEMENT ORDER comes from
the LOG (logparse setup_settlement/setup_road event order + the record.py placement-order contract, commit
fc83e6d), NOT from the VLM → the EXISTING fail-closed validators still guard everything (OpeningResult
invariants: exactly 2 settlements + 2 roads/player; the joint-flip glyph firewall; HUD binding). The VLM
does ONLY perception (localize + adjacency); IDs, order, and flip-safety stay deterministic.
THE VLM PROXY: in this spike the "VLM" is an Opus vision subagent that Reads the frame PNGs — a faithful
frontier-VLM proxy for the question "can a VLM do this localization?"; production would pin a specific API
(Gemini/Claude). Document this clearly; do not pretend it is Gemini.
GROUND TRUTH: tests/fixtures/human_data/game1_openings.json is a hand-verified game (openings + placement_order
+ granted_resources + log_setup_sequence; its own video, NOT one of the Tier-5 five). This is the HARD
accuracy anchor. The Tier-5 videos (33KR75rhTgo, AoOXWyxaTkA, 5RLq1NX4nAo, sG05DoaOmM4, EdMnUD-eZ6A) +
more 'high' manifest videos are the YIELD test (no ground truth — count complete-valid openings via the
fail-closed validators; classical CV got 0%). HARD RULES: ruff + mypy (PROJECT config: make lint/typecheck)
+ pytest green before every commit; conventional commits; push origin/main. CPU only; no gui/ import; do NOT
touch src/catan_rl/ppo/ or the training loop (a training run is LIVE on MPS — stay in human_data/ + a new
scripts/vlm_spike.py + data/human/vlm_spike/). yt-dlp + imageio-ffmpeg for frames (see ingest.py/ffmpeg.py;
net concurrency 1-2, download-then-delete). NEVER loosen a committed test/fixture. Fail-closed is sacred:
an unlocalizable piece is a TYPED rejection, never a guess.`

const VERIFY_FMT = `Return MARKDOWN only. Sections: TOOLS (commands RUN + key lines), CHECKS (each
invariant: pass/fail + evidence), DATA (open the artifact, recompute a number), TAMPER (git diff on
tests/fixtures), BLOCKERS. End with GREEN or RED on its own line.`
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
      `${COMMON}\n\nINDEPENDENTLY VERIFY slice ${slice.name} (do NOT trust the implementer). RUN the
tools; check invariants: ${slice.invariants}. ${VERIFY_FMT}`,
      { label: `verify:${tag}`, phase: slice.name, effort: 'high', model: MODEL })
    const verText = ver || ''
    if (/\bRED\b\s*$/.test(verText.trim()) || !/\bGREEN\b/.test(verText)) {
      findings = [{ severity: 'BLOCKER', title: 'verify RED', issue: verText.slice(-1500), fix: 'make it genuinely green' }]
      note(`${slice.name} iter ${iter}: verify RED -> resolve`); continue
    }

    const review = await safeAgent(
      `${COMMON}\n\nREVIEW slice ${slice.name} against intent: ${slice.what}\nRead the actual diff/files.
Hunt: the VLM being asked for exact IDs instead of adjacency (must be geometry-snapped); order coming from
the VLM instead of the log; a validator being bypassed; the mock localizer leaking into the real path;
ground-truth (game1 fixture) being read during localization (must be blind). ${REVIEW_FMT}`,
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

// ---------- slice 1: the spike harness (code, mock-tested) ----------
phase('harness')
const harness = await buildSlice({
  name: 'harness',
  what: `Build scripts/vlm_spike.py: the end-to-end spike harness with a PLUGGABLE localizer. Components:
(a) FRAME EXTRACTION reuse — for a given video+game, produce the post-setup frame + empty-baseline frame
using the EXISTING board_cv/openings/ingest frame logic (do not reinvent; import + reuse). Also save both
PNGs to data/human/vlm_spike/frames/<video>__g<idx>/ so a vision agent can Read them. (b) A Localizer
PROTOCOL: input = the frame PNG paths + the board_cv BoardRead (so the localizer knows the tile layout);
output = per player, a list of {piece: settlement|road, adjacency: [hex ids] for settlements OR
[vertexA-ish, vertexB-ish] adjacency hint for roads}. Provide TWO implementations behind the protocol: a
MockLocalizer (returns scripted adjacency, for deterministic tests) and a FileLocalizer (reads a per-game
JSON that the VLM phase will write to data/human/vlm_spike/localized/<video>__g<idx>.json). (c) SNAP:
deterministic adjacency→id using topology.load_topology() — a settlement's [hex ids] set uniquely
identifies its vertex via vertex_adjacent_hexes (a corner = the vertex adjacent to exactly those hexes);
a road snaps to the edge between its two endpoint vertices via vertex_neighbors. Fail-closed: adjacency
that matches 0 or >1 vertex => typed reject (ambiguous_snap), never a guess. (d) ORDER: attach placement
order from the log (reuse logparse setup event order + the record.py contract) — NOT from the localizer.
(e) VALIDATE: run the result through the existing OpeningResult invariants (exactly 2 settlements + 2 roads
per player) + the joint-flip glyph firewall; emit a GameRecord on success or a typed rejection. (f) SCORE
helpers: exact vertex/edge match vs a ground-truth openings dict (for game-1). CLI: prepare-frames,
localize (via FileLocalizer), score.`,
  tests: `with MockLocalizer: a scripted correct adjacency snaps to the expected vertex/edge ids and
passes validators (use the game-1 fixture's known openings as the expected answer — the MOCK feeds game-1's
true adjacency, proving the snap+validate math is correct); an ambiguous adjacency (0 or 2 matching
vertices) yields typed ambiguous_snap; order is taken from a supplied log sequence not the localizer;
suite green. (No network in unit tests — mock frames/paths.)`,
  invariants: `VLM/localizer output is adjacency ONLY (never a raw id); ids come exclusively from the
topology snap; order comes exclusively from the log; fail-closed on ambiguous snap; no training-path import;
game-1 fixture used only as the expected answer in scoring, never read by the localizer path.`,
})

// ---------- phase 2: REAL VLM localization (vision fan-out per game) ----------
let localized = []
if (harness.ready) {
  phase('localize')
  // 2a: prepare frames for game-1 (ground truth) + the Tier-5 five + expansion toward ~15 videos
  const prep = await safeAgent(
    `${COMMON}\n\nUsing scripts/vlm_spike.py prepare-frames: download + extract the post-setup + empty-baseline
frames for (1) the game-1 fixture's video [tests/fixtures/human_data/game1_openings.json -> its 'video' id],
and (2) as many 'high' manifest videos as needed to cover the 5 Tier-5 videos plus expansion to ~15 videos
total / ~18 games. Save frames under data/human/vlm_spike/frames/<video>__g<idx>/. Return ONLY a JSON array
of the prepared game dirs (each an object {dir, video, game_idx, is_ground_truth}).`,
    { label: 'prep-frames', phase: 'localize', effort: 'high', model: MODEL })
  let games = []
  try { games = JSON.parse((prep || '[]').replace(/```json|```/g, '').trim()) } catch { games = [] }
  note(`frame sets prepared: ${games.length}`)

  // 2b: per-game VLM localization fan-out (Opus vision reads the frames; blind to ground truth)
  const CH = 3
  const groups = []
  for (let i = 0; i < games.length; i += CH) groups.push(games.slice(i, i + CH))
  const results = await parallel(groups.map((g, gi) => () => safeAgent(
    `You are the VLM LOCALIZER for a Catan opening-parse spike. For each game dir in ${JSON.stringify(g)}:
READ the post-setup frame PNG and the empty-baseline frame PNG (the empty baseline shows the board with NO
pieces — use it to tell a piece from a same-coloured tile). Identify EACH player's opening pieces by colour:
2 settlements + 2 roads per player. For each SETTLEMENT, report the set of board hexes its corner touches
(read the resource + number token of each adjacent hex, e.g. "ore-8, wheat-5, forest-11"). For each ROAD,
report the two board corners (vertices) it connects, described by their adjacent hexes. Output ONLY tile
ADJACENCY — do NOT output engine vertex/edge numbers (the deterministic snap does that). You must NOT open
any pipeline output, the game1 fixture, any *.jsonl corpus, or src/ parsing code — ONLY the frame images.
If a piece genuinely cannot be localized, mark it "unreadable" (fail-closed — do not guess). Write each
game's result to data/human/vlm_spike/localized/<video>__g<idx>.json in the FileLocalizer schema
scripts/vlm_spike.py expects. Report which files you wrote + any unreadable pieces.`,
    { label: `vlm:${gi}`, phase: 'localize', effort: 'high', model: MODEL })))
  localized = results.filter(Boolean)
  note(`VLM localization groups done: ${localized.length}/${groups.length}`)
}

// ---------- slice 3: score + yield + gating arithmetic + decision ----------
let decision = null
if (harness.ready) {
  phase('score')
  decision = await safeAgent(
    `${COMMON}\n\nSCORE THE SPIKE and produce the decision. Run scripts/vlm_spike.py score / the full
snap+validate over every localized game in data/human/vlm_spike/localized/. Produce THREE numbers +
a report at docs/plans/vlm_spike_report.md:
(1) HARD ACCURACY on game-1: snap the VLM localization and compare exact vertex/edge ids to the game1
fixture's openings (8 placements: 4 settlements + 4 roads). Report per-placement correct / total + which missed.
(2) YIELD: over the Tier-5 videos + expansion, the fraction of games that produce a COMPLETE VALID opening
(passes OpeningResult invariants + the joint-flip firewall) — this is the number classical CV got 0/31 on.
Report accepted/seen + Wilson-95 CI + the typed-rejection breakdown (what still fails: ambiguous_snap,
unreadable, flip, hud, board).
(3) RECOVERED GAMES/VIDEO from the expansion set.
Then RE-RUN THE GATING-CELL ARITHMETIC with measured inputs: cell = (games/video × 204) × yield × 0.6.
Apply the PRE-REGISTERED RULE (decide the verdict from the numbers, state it plainly):
  projected cell >= 147  -> CONTINUE (build the VLM front-end for real + integration + gold gate);
  100 <= cell < 147      -> USER-CALL ("large-effects-only study", honest label);
  cell < 147 but > 0 and yield clearly high enough that more videos could reach it -> note the extra-videos path;
  cell < 100             -> ARCHIVE the corpus program (bank PRE-GATE-0/M0/injection-API/manifest; not a failure).
Also save the provisional parsed openings for the non-ground-truth videos to data/human/vlm_spike/provisional_openings.jsonl
FLAGGED for user hand-verification (do not treat as verified). Commit + push the report + artifacts (gitignore
any frame PNGs / large files; commit the localized JSONs + report + provisional openings if < 500KB).
Return MARKDOWN: the three numbers, the gating arithmetic, and the CONTINUE/USER-CALL/ARCHIVE recommendation
with the pre-registered rule shown — the decision is the user's.`,
    { label: 'score-decide', phase: 'score', effort: 'high', model: MODEL })
  note(`score+decision done`)
}

phase('score')
const summary = await safeAgent(
  `Write the completion summary (markdown) for the developer waking up: harness verdict + commit shas
(git log), the THREE spike numbers (game-1 hard accuracy, yield vs the 0/31 classical baseline, games/video),
the re-run gating-cell arithmetic, and the explicit CONTINUE / USER-CALL / ARCHIVE recommendation per the
pre-registered rule. Note clearly that the "VLM" was an Opus-vision proxy and production would pin an API.
Journal:\n${JSON.stringify(journal, null, 2)}\nDecision:\n${decision || '(did not run)'}`,
  { label: 'summary', phase: 'score', effort: 'medium', model: MODEL })

return {
  results: [harness].map(r => ({ name: r.name, ready: r.ready })),
  halted: journal.filter(m => m.includes('HALT')),
  decision,
  summary,
}
