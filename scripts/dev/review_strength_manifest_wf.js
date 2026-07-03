export const meta = {
  name: 'review-strength-manifest',
  description: 'Review-and-resolve the ThePhantom strength-manifest classifier: independent frame verify + 2 review lenses + red-team -> markdown severity-tagged issues + verdict',
  phases: [
    { title: 'Verify',  detail: 'independent agent Reads saved leaderboard frames and confirms the manifest rank matches the visible ThePhantom row' },
    { title: 'Review',  detail: 'data-labeling-correctness + CV/OCR/SWE-robustness lenses read the classifier + sample manifest' },
    { title: 'RedTeam', detail: 'adversarial edge cases that could produce a FALSE-HIGH (worst error: pollutes the scoreboard)' },
    { title: 'Synthesis', detail: 'merge into one severity-tagged issue list (BLOCKER/SHOULD-FIX/NIT) + READY verdict' },
  ],
}

// NOTE: schema-free by design. A prior run died when a lens agent malformed its own
// StructuredOutput tool call (stray tag inside a large payload). Free-form MARKDOWN
// cannot malform a tool call; the main loop consumes the text directly.

const SCRIPT = 'scripts/build_strength_manifest.py'
const MANIFEST = 'data/human/strength_manifest.sample.json'  // stable 10-video snapshot
const FRAMES = 'data/human/frames/'

const OUTFMT = `
OUTPUT FORMAT — return MARKDOWN only (no tool calls for output). For each finding use
exactly this block, one after another:

### <SEVERITY>: <short title>
- **where:** <file:line or component>
- **issue:** <what is wrong / the concrete failing scenario>
- **fix:** <concrete change>

SEVERITY is one of BLOCKER / SHOULD-FIX / NIT. End with a line: \`VERDICT: READY\` or
\`VERDICT: NOT-READY\`. If you find nothing, say so and give VERDICT: READY.`

const CONTEXT = `
CONTEXT — ThePhantom strength manifest (1v1 Catan RL project).
GOAL: label each of ~814 YouTube videos on @ThePhantomcatan as high / excluded / unknown, to gate which
human games feed (A) a strength "scoreboard" (verified top-200/tournament only) and (B) a diverse-opening
seed corpus (anything not confirmed-low). "high-rank" is DEFINED by the user as: ThePhantom is ranked top-200
in the world (shown at start/end of ranked videos on the 1v1 Global leaderboard) OR the game is a 1v1
tournament. Two deterministic LOCAL signals (no API/vision-model):
  1. tournament: title keyword regex -> high.
  2. ranked: OCR (easyocr, CPU) a leaderboard frame (scanned last ~110s then first ~50s), find ThePhantom's
     highlighted row, read the '#N' world rank; N<=200 -> high, N>200 -> excluded, not found -> unknown.
     Robustness already built: reads the LEFTMOST '#N' token on his row (works across both Colonist UI eras);
     a GLOBAL-tab gate (active tab = lowest blue-channel among tab labels; regional tabs rejected) so an
     "Australia #1" frame can't be mistaken for a world rank; an fy>0.13 guard so the top-corner profile
     name can't match.
Classifier: ${SCRIPT}. Sample output: ${MANIFEST}. Saved winning frames: ${FRAMES}<id>_rank.png.
THE WORST ERROR is a FALSE-HIGH: a low-rank or non-ThePhantom game mislabeled 'high' pollutes the scoreboard
(the strength reference the whole project trusts). A false-unknown is cheap (game still usable as a seed).
INVARIANTS: deterministic, local, resumable/crash-safe, CPU-only, no GUI import, additive (new files only).
`

phase('Verify')
const verify = await agent(
  `${CONTEXT}
You are an INDEPENDENT verifier. Do NOT trust the OCR. For each ranked-labeled row in ${MANIFEST} that has a
saved frame (${FRAMES}<id>_rank.png), Read the PNG image yourself and visually confirm: is ThePhantom's row
highlighted, is the Global tab the active one, and does the '#N' rank you SEE match the rank the manifest
recorded? Also sanity-check one 'tournament' row (title) and one 'unknown' row. Use Read on the image files
and on ${MANIFEST}. Return MARKDOWN: a per-video table (video_id | manifest says | frame shows | match?) then
a one-line ALL-MATCH: yes/no and notes. No tool call for output — just markdown text.`,
  { label: 'verify:frames', phase: 'Verify' },
)

phase('Review')
const lenses = [
  {
    key: 'data-correctness',
    prompt: `${CONTEXT}
LENS A — DATA-LABELING CORRECTNESS. Read ${SCRIPT} and ${MANIFEST}. Judge the LABEL LOGIC, hunting FALSE-HIGH
risks above all: (1) the tournament regex — does it over-match (a ranked video mislabeled tournament) or
under-match a real tournament? (2) the rank rule — is the same-row leftmost-'#N' + Global-gate robust against
reading a DIFFERENT row's rank, or a non-ThePhantom row? (3) the N<=200 boundary and the "not found ->
unknown" vs ">200 -> excluded" split — conservative in the right direction? (4) is the high/unknown/excluded
-> scoreboard/seed mapping safe (never lets an unverified game into the scoreboard)?
${OUTFMT}`,
  },
  {
    key: 'cv-swe-robustness',
    prompt: `${CONTEXT}
LENS B — CV/OCR/SWE ROBUSTNESS. Read ${SCRIPT}. Judge robustness/correctness of the implementation: the
Global-tab min-blue heuristic (thresholds, resolution/UI-era dependence, could it pick wrong tab?); the
leftmost-'#N' rank extraction (fx band [0.12,0.48], digit-clip like '#18'->'8', '100' misreads); _is_phantom
fuzzy match (could it match another player's name containing 'phantom', or miss OCR slips?); easyocr reader
reuse; frame-URL expiry across a multi-hour run; yt-dlp/ffmpeg timeouts -> 'unknown' fallback; resumability/
crash-safety (incremental save, skip-done); the END_OFFSETS/START_TIMES sampling (could it MISS a leaderboard
shown only mid-video?).
${OUTFMT}`,
  },
]
const reviews = await parallel(lenses.map(l => () =>
  agent(l.prompt, { label: `review:${l.key}`, phase: 'Review' })))

phase('RedTeam')
const red = await agent(
  `${CONTEXT}
You are a RED-TEAM adversary. Your only goal: construct concrete scenarios where this classifier emits a
FALSE-HIGH (worst error) or a systematic mislabel, then check the code (${SCRIPT}) to see if each is actually
prevented. Consider: an opponent whose username contains "phantom"; a 4-player leaderboard; a frame where
ThePhantom appears twice (profile header + row); a tournament that is actually a casual/for-fun stream; a
title with "#1" or "Rank" that is NOT a tournament; a leaderboard shown for an OLD season; a rank read as '8'
when it was '#18' (digit clipped); a regional tab that narrowly beats Global on the blue heuristic. For each,
state whether the code catches it and, if not, the fix. A real uncaught false-high = BLOCKER.
${OUTFMT}`,
  { label: 'redteam:false-high', phase: 'RedTeam' },
)

phase('Synthesis')
const synth = await agent(
  `${CONTEXT}
Synthesize the independent verification, two review lenses, and the red-team pass into ONE deduplicated,
severity-tagged issue list (markdown) for the strength-manifest classifier. Order BLOCKER first, then
SHOULD-FIX, then NIT. State a final VERDICT: READY (zero open BLOCKER/SHOULD-FIX and no unresolved false-high
risk) or NOT-READY. Be concrete and cite ${SCRIPT} lines where possible.

=== INDEPENDENT FRAME VERIFY ===
${verify || '(verify agent produced no output)'}

=== REVIEW LENS A + B ===
${reviews.filter(Boolean).join('\n\n---\n\n') || '(no lens output)'}

=== RED-TEAM ===
${red || '(no red-team output)'}
${OUTFMT}`,
  { label: 'synthesis', phase: 'Synthesis' },
)

return { verify, reviews: reviews.filter(Boolean), red, synth }
