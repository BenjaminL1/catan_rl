export const meta = {
  name: 'precorpus-lane',
  description: 'Implement the step6 pre-corpus lane (plan v5.2 §2) via the per-slice review-and-resolve loop, all agents on OPUS: placement-order pin (harvest-blocking), measurement-only archetype featurizer, engine injection API + bridge null test (i), setup-phase entropy wiring, PRE-GATE-0/M0 runner.',
  phases: [
    { title: 'order_pin' },
    { title: 'archetypes' },
    { title: 'injection_api' },
    { title: 'setup_entropy' },
    { title: 'pregate0' },
  ],
}

const MAX_ITERS = 4
const MODEL = 'opus' // user model-routing policy: implementation runs on Opus
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

const COMMON = `You are implementing the PRE-CORPUS LANE of the ratified plan
docs/plans/v2/step6_human_corpus.md (v5.2 — READ §0, §1, §2, §3.1 fully; it is the design
authority; it survived 5 expert review rounds + a unanimous 4-persona council, so implement
what it says — do not re-design). Project conventions (CLAUDE.md): 1v1 ruleset sacred; obs
schema/action space/checkpoint lineage unchanged; additive & default-off for anything touching
training; TB scalars additive, never renamed; CPU eval; never import gui/ in RL paths; engine
changes must be flagged as such in the commit message (rule 2); resource-literal vs engine vs
RESOURCES_CW ordering trap (rule 6); ruff + mypy --strict + pytest green before EVERY commit;
conventional commits; push origin/main. NEVER loosen/delete a committed test or golden fixture
to go green — if a change to the game-1 golden fixture is genuinely required by a slice spec,
it must be additive (new fields/metadata), justified in the commit message, and the existing
IDs/values must not change. Relevant existing surfaces: src/catan_rl/human_data/{record,
logparse,openings,validate,orientation,glyph_anchor,board_cv,topology,ffmpeg,batch}.py,
src/catan_rl/engine/{board,player,game}.py, src/catan_rl/eval/harness.py,
src/catan_rl/ppo/arguments.py, tests/unit/human_data/, tests/fixtures/human_data/
(game1_openings.json is the hand-verified golden record: draft_order
[rayman147, ThePhantom, ThePhantom, rayman147], log_setup_sequence included).`

const VERIFY_FMT = `Return MARKDOWN only. Sections: TOOLS (ruff/mypy/pytest key lines you RAN),
CHECKS (each slice invariant: pass/fail + evidence), TAMPER (git diff on committed tests/fixtures:
clean or justified-additive), BLOCKERS (empty if none). End with GREEN or RED on its own line.`
const REVIEW_FMT = `Return MARKDOWN only: findings as "### <BLOCKER|SHOULD-FIX|NIT>: title" with
where/issue/fix bullets, then VERDICT: READY or NOT-READY.`

const journal = []
function note(m) { log(m); journal.push(m) }

async function buildSlice(slice) {
  let findings = []
  for (let iter = 0; iter < MAX_ITERS; iter++) {
    const tag = `${slice.name}#${iter}`
    const impl = await safeAgent(
      `${COMMON}\n\nSLICE ${slice.name}: ${slice.what}\nTESTS REQUIRED: ${slice.tests}\n` +
      (iter > 0 ? `RESOLVE PASS — fix EXACTLY these open findings, nothing else:\n${JSON.stringify(findings, null, 2)}\n` : '') +
      `Re-green ruff + mypy --strict + pytest, commit, push. Report the commit sha + summary.`,
      { label: `impl:${tag}`, phase: slice.name, effort: 'high', model: MODEL })
    if (!impl) { note(`HALT ${slice.name}: impl died`); return { name: slice.name, ready: false } }

    const ver = await safeAgent(
      `${COMMON}\n\nINDEPENDENTLY VERIFY slice ${slice.name} (do NOT trust the implementer). RUN
ruff check, mypy --strict (project config), pytest (the new tests + the full tests/unit/human_data
suite + any touched module's tests). Verify the slice invariants: ${slice.invariants}. Check git
diff/log for test/fixture tampering (additive fixture changes are OK only if the slice spec
authorizes them and old values are unchanged). ${VERIFY_FMT}`,
      { label: `verify:${tag}`, phase: slice.name, effort: 'high', model: MODEL })
    const verText = ver || ''
    if (/\bRED\b\s*$/.test(verText.trim()) || !/\bGREEN\b/.test(verText)) {
      findings = [{ severity: 'BLOCKER', title: 'verify RED', issue: verText.slice(-1500), fix: 'make it genuinely green' }]
      note(`${slice.name} iter ${iter}: verify RED -> resolve`); continue
    }

    const review = await safeAgent(
      `${COMMON}\n\nREVIEW slice ${slice.name} against its intent: ${slice.what}\nRead the actual
diff/files. Hunt: does it ACTUALLY satisfy the plan section it implements (cite the section)?
plan-contract violations? silent behavior changes to training/eval? invariant breaks? regression
risk? ${REVIEW_FMT}`,
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

const slices = [
  {
    name: 'order_pin',
    what: `THE HARVEST-BLOCKING PLACEMENT-ORDER CONTRACT (plan §3.1). (a) record.py: pin in
PlayerOpening's docstring that settlements/roads tuples are in LOG PLACEMENT ORDER
(settlements[0]=first-placed, settlements[1]=second-placed/resource-granting), sourced from the
setup-event sequence. (b) Implement the order-establishment helper: given the parsed setup events
(logparse's ordered stream: '<player> placed a Settlement/Road' lines in draft order) and the
frame-detected UNORDERED opening sets, produce log-ordered tuples. NOTE the detection is
order-blind — the log gives WHO placed WHEN (the snake draft), and each player's 1st vs 2nd
settlement must be disambiguated by an explicit, documented rule; if the two cannot be
distinguished for a player (e.g. missing setup lines), mark the record ORDER-UNESTABLISHED: add
an additive provenance flag (e.g. provenance["placement_order_established"]: bool) — such records
are EVAL-excluded, seed-eligible only (wire the predicate: is_scoreboard_eligible must also
require it; is_seed_eligible must NOT). (c) The game-1 golden fixture: verify the existing
settlement IDs against its committed log_setup_sequence and ADD the order metadata additively
(do not change existing IDs). Think carefully about HOW order is established from available
signals (the grant event follows the 2nd settlement; the 'received starting resources' line
position in the sequence identifies which placement it followed) — document the chosen rule in
the docstring.`,
    tests: `order helper: synthetic setup sequences (normal snake, missing lines -> unestablished);
game-1 fixture round-trips with order established and matches its log; is_scoreboard_eligible
false when order unestablished while is_seed_eligible unaffected; existing 337-test suite green.`,
    invariants: `no existing fixture VALUES changed (additive only); schema stays v2-compatible
(from_dict of old records without the flag still loads, defaulting to unestablished=False? NO —
old records predate the contract: default must be conservative, i.e. NOT established); the
grant-source rule (settlements[1] grants) documented.`,
  },
  {
    name: 'archetypes',
    what: `The measurement-only archetype featurizer (plan §2.1, v5.2 scope restriction).
src/catan_rl/human_data/opening_archetypes.py implementing EXACTLY the frozen spec: features per
(seat, opening) = 5-vector pip-share by resource over the two settlements (adjacent-hex pips via
the committed topology; desert=0; normalized; total-pips=0 => BALANCED_LOW directly), total pips,
port-slot adjacency (either settlement on one of the 9 fixed port-slot vertex pairs from
topology.port_slots). "Share" = the named PAIR-SUM only. Buckets, fixed precedence: ORE_ENGINE
(ORE+WHEAT >= 0.45), WOOD_BRICK (WOOD+BRICK >= 0.45), PORT_LED (port-adjacent AND neither
pair-share >= 0.45), BALANCED_HIGH (neither >= 0.45 AND total pips >= 26), BALANCED_LOW (else).
Also: histogram + Shannon-entropy helpers over the 5 buckets. Module docstring MUST carry the
v5.2 scope restriction verbatim in spirit: measurement-only (PRE-GATE-0 collapse verdict +
GATE-B3(3) + dashboards), never training, never seed selection.`,
    tests: `boundary tests each side of 0.45 and 26; precedence order; zero-pip; pair-share (a
0.5-single-resource opening does NOT trigger a pair bucket unless the pair sum clears 0.45);
port-slot adjacency against the committed topology; entropy of a degenerate vs uniform histogram;
determinism.`,
    invariants: `pure function of (record board + one seat's opening) — no engine, no gui, no
randomness; no other module imports it into a training path.`,
  },
  {
    name: 'injection_api',
    what: `The engine board/port INJECTION API + v8-state re-bridge + bridge null test (i)
(plan §2.3, §3.2). (a) engine/board.py (ENGINE CHANGE — flag rule 2 in the commit message):
additive API to construct/overwrite a catanBoard with a SPECIFIC hex layout (resources+numbers),
robber position, and a DETERMINISTIC port assignment from an explicit RNG or explicit port list
(updatePorts currently draws from global np.random — do not change its default behavior; add the
injectable path alongside). Human boards need arbitrary number placement (the spiral generator
cannot produce them) => post-construction overwrite of hexTileDict/resourcesList + dependent
index maps, kept consistent. (b) A minimal engine_bridge round-trip for V8 SELF-PLAY states:
serialize a live post-setup catanGame (board + placements + hands + ports + robber + to-move)
to a GameRecord-shaped dict + out-of-band port list, rebuild via the injection API + placement
replay with post-condition assertions (engine build_* silently no-op — assert piece counts,
occupancy, distance rule, road incidence via env masks/geometry), grants via spec-009 bank_draw,
then assert_conservation + rules_invariants + hand-tracker parity. (c) BRIDGE NULL TEST (i):
for N=50 v8 self-play post-setup states (generate quickly with the existing env + a fixed seed;
random or v8 policy actions through setup is fine — the point is REAL env-produced states),
round-trip and assert v_hat parity (|delta| < 1e-5) between the live env obs and the re-bridged
obs, plus engine state-hash equality; the fixture set MUST include at least one state whose
opening settlement sits on a port-slot vertex (catches a portList weld). CPU only.`,
    tests: `the null test itself (runs in CI at N reduced, e.g. 8 states, marked slow for N=50);
injection determinism (same explicit ports -> identical board twice); default board generation
byte-unchanged when the injectable path is unused (existing engine tests + conformance stay
green).`,
    invariants: `engine default behavior unchanged (injectable path additive + unused by
training); no obs-schema change; bank conservation holds on every bridged state.`,
  },
  {
    name: 'setup_entropy',
    what: `The setup-phase-only entropy bonus (plan §2.2, §5-mechanism). ppo/arguments.py gains
setup_entropy_coef (default 0.0 — a run must opt in); the PPO loss applies an ADDITIONAL entropy
bonus with this coefficient ONLY to decisions taken during the initial-placement phase (identify
setup steps via the env/buffer signal that already distinguishes the placement phase — find the
cleanest existing signal; if none exists in the buffer, thread a minimal additive flag).
TB: openings/setup_head_entropy scalar (mean policy entropy over setup decisions per update) —
additive. Config plumbing through configs/ppo_default.yaml (default 0.0, documented).`,
    tests: `coef=0.0 => training loss BYTE-IDENTICAL to before the change (regression test on a
fixed synthetic batch); coef>0 changes loss ONLY via setup-step entropy (non-setup steps'
contribution unchanged); TB scalar emitted; arguments round-trip.`,
    invariants: `default-off = exact no-op; no obs/action/checkpoint shape change; TB additive;
existing training tests green.`,
  },
  {
    name: 'pregate0',
    what: `The PRE-GATE-0 + M0 runner (plan §2.2). scripts/pregate0.py: plays n natural v8-vs-v8
AND v8-vs-anchor games via the existing eval machinery (CPU, the frozen champion
runs/anchors/v8_promobar_u243.pt; anchor = the same checkpoint class the league uses — read the
plan; if only one checkpoint is available use v8-vs-v8 plus v8-vs-<frozen prior anchor> if one
exists on disk, else document), and for each game records: both openings (vertex ids), archetype
bucket (via opening_archetypes), post-setup v_hat of the to-move seat (live env, true ports —
this is M0's estimator), draft positions, outcome. Outputs: (a) the committed per-bucket mass
table artifact data/human/pregate0_mass.json (v8-vs-anchor subset only — plan §2.2) + a report
data/human/pregate0_report.md with the archetype histogram, Shannon entropy,
openings/setup_head_entropy (mean setup-decision policy entropy), the COLLAPSE VERDICT
(>=70% one-bucket mass), and M0 = AUC + the M2-analog partial Spearman with permutation strata
(draft_position) ONLY (plan pin). Resumable (append JSONL per game, atomic), deterministic given
seeds, --n and --smoke flags. DO NOT run the full n>=400 inside this workflow — implement +
smoke-test (n=6) only; the long run is launched detached afterward.`,
    tests: `smoke n=2-6 end-to-end produces well-formed JSONL + report; mass table = v8-vs-anchor
subset only; M0 strata = draft_position only; resumability (kill + rerun appends, no dupes);
deterministic given seed.`,
    invariants: `CPU-pinned eval; no training-path import of gui; artifacts additive under
data/human/; the collapse threshold and entropy definitions match opening_archetypes + the plan.`,
  },
]

const results = []
for (const s of slices) { phase(s.name); results.push(await buildSlice(s)) }

phase('pregate0')
const summary = await safeAgent(
  `Write a concise completion summary (markdown) for the developer: which pre-corpus-lane slices
landed (READY) with commit shas (check git log), which halted and why, what is now unblocked
(the harvest-blocking order pin status; whether PRE-GATE-0's full n>=400 run can be launched),
and the exact command to launch the full PRE-GATE-0 run detached. Journal:\n${JSON.stringify(journal, null, 2)}\nResults:\n${JSON.stringify(results, null, 2)}`,
  { label: 'summary', phase: 'pregate0', effort: 'medium', model: MODEL })

return { results, halted: journal.filter(m => m.includes('HALT')), summary }
