export const meta = {
  name: 'catan-rl-v2-overnight',
  description: 'Overnight queue: run ratified slices through the dev-loop back-to-back, auto-fix SHOULD-FIXes gate-green, auto-merge clean slices to main, stop on any red',
  whenToUse: 'Invoked by /overnight with a queue of RATIFIED specs. Owner-ratified governance (2026-08-21): auto-merge on 0 blockers; SHOULD-FIX autonomy with gate re-run; queue stops on FAIL/blockers/conflict.',
  phases: [
    { title: 'Preflight', detail: 'clean main, specs ratified' },
    { title: 'Slices', detail: 'dev-loop -> should-fix -> merge, per queue item' },
    { title: 'Report', detail: 'consolidated morning report' },
  ],
}

// args: {
//   repoRoot: absolute path of the owner's main checkout
//   devLoopScript: absolute path of .claude/workflows/catan-rl-v2-dev-loop.js
//   queue: [{ title, feature, spec, specPath }]  (feature/spec passed VERBATIM to the child)
//   posture?: forwarded to each child
// }
const repo = args.repoRoot
const devLoop = args.devLoopScript
const queue = Array.isArray(args.queue) ? args.queue : []
if (!repo || !devLoop || queue.length === 0) {
  throw new Error('overnight: need args.repoRoot, args.devLoopScript, non-empty args.queue')
}

const GATE_CMDS = 'make typecheck && make lint && cargo fmt --all -- --check && make test-unit'

phase('Preflight')
const pre = await agent(
  `Preflight for an unattended overnight dev queue on the repo at ${repo}. READ-ONLY except nothing at all — do not modify anything.\n` +
  `Check and report:\n` +
  `1. git -C ${repo} status --porcelain — staged or modified TRACKED files present? (untracked files are fine)\n` +
  `2. git -C ${repo} rev-parse HEAD vs origin/main after git -C ${repo} fetch origin — is local main in sync (or ahead only)?\n` +
  `3. For each spec path below: does the file exist and does its Status line contain "RATIFIED"?\n` +
  queue.map((q, i) => `   [${i}] ${q.specPath}`).join('\n') + '\n' +
  `4. Print the current HEAD sha.\n` +
  `Be strict: any staged/modified tracked file, diverged main, or non-ratified spec is a problem.`,
  {
    label: 'preflight', phase: 'Preflight', effort: 'low',
    schema: {
      type: 'object', additionalProperties: false,
      properties: {
        ok: { type: 'boolean' },
        headSha: { type: 'string' },
        problems: { type: 'array', items: { type: 'string' } },
      },
      required: ['ok', 'headSha', 'problems'],
    },
  },
)
if (!pre || !pre.ok) {
  return { stopped: 'preflight', problems: pre ? pre.problems : ['preflight agent failed'], merged: [], perSlice: [] }
}

phase('Slices')
const perSlice = []
const merged = []
let stopReason = null

for (let i = 0; i < queue.length; i++) {
  const item = queue[i]
  const tag = item.title || `slice-${i + 1}`
  log(`[overnight] ${tag}: dev-loop starting (${i + 1}/${queue.length})`)

  let res = null
  try {
    res = await workflow({ scriptPath: devLoop }, {
      feature: item.feature,
      spec: item.spec,
      posture: args.posture,
    })
  } catch (e) {
    stopReason = `${tag}: dev-loop threw — ${e && e.message ? e.message : 'unknown'}`
    perSlice.push({ slice: tag, outcome: 'ERROR', detail: stopReason })
    break
  }

  const verdict = res && res.finalVerdict
  const blockers = (res && res.blockers) || []
  const branch = res && res.branch
  const worktree = res && res.worktree
  const landSha = res && res.land && res.land.commitSha
  if (!branch || verdict === 'FAIL' || blockers.length > 0) {
    stopReason = `${tag}: gate ${verdict}, ${blockers.length} blocker(s) — queue stopped, branch ${branch || 'none'} left for morning review`
    perSlice.push({ slice: tag, outcome: 'STOPPED_RED', verdict, blockers, branch, worktree })
    break
  }

  // SHOULD-FIX autonomy pass (owner-ratified): apply the lens SHOULD-FIXes, keep the gate green,
  // merge only a verified state. One bounded pass.
  const findings = ((res.brief && res.brief.findings) || []).filter(
    (f) => f && (f.severity === 'SHOULD-FIX' || f.severity === 'should-fix'),
  )
  let fixReport = { applied: [], skipped: findings.length ? ['fix agent did not run'] : [], tipSha: landSha }
  if (findings.length > 0) {
    const fr = await agent(
      `Bounded SHOULD-FIX pass on branch ${branch} in the worktree ${worktree} (base sha ${landSha}). ` +
      `Work ONLY in that worktree. Never touch ${repo} directly, never src/catan_rl/engine/, never data/labels/**.\n` +
      `Apply these review SHOULD-FIX findings surgically (skip any that is risky, ambiguous, or spec-contradicting — skipping is fine):\n` +
      findings.map((f, n) => `${n + 1}. [${f.location || '?'}] ${f.issue}`).join('\n') + '\n' +
      `Then run the FULL gate from the worktree root: ${GATE_CMDS}\n` +
      `ALL commands must exit 0. If green: commit (conventional, lowercase, <72 chars, NO AI attribution/trailers of any kind) and push the branch. ` +
      `If you cannot get green: git reset --hard ${landSha} and push nothing. ` +
      `Report exactly what you applied, what you skipped and why, and the final pushed tip sha (git rev-parse HEAD).`,
      {
        label: `should-fix:${tag}`, phase: 'Slices',
        schema: {
          type: 'object', additionalProperties: false,
          properties: {
            applied: { type: 'array', items: { type: 'string' } },
            skipped: { type: 'array', items: { type: 'string' } },
            gateGreen: { type: 'boolean' },
            tipSha: { type: 'string' },
          },
          required: ['applied', 'skipped', 'gateGreen', 'tipSha'],
        },
      },
    )
    if (fr && fr.gateGreen) fixReport = { applied: fr.applied, skipped: fr.skipped, tipSha: fr.tipSha }
    else fixReport = { applied: [], skipped: fr ? fr.skipped : ['fix agent failed — merging the gated base state'], tipSha: landSha }
  }

  // Merge the VERIFIED tip to main. Conflict => abort merge, stop queue.
  const mg = await agent(
    `Merge an overnight-gated slice into main for the repo at ${repo}.\n` +
    `1. Verify git -C ${repo} rev-parse ${branch} equals ${fixReport.tipSha} (the verified tip). If it does not, STOP and report mismatch.\n` +
    `2. git -C ${repo} merge --no-ff ${branch} -m "merge: ${tag} (overnight)" — conventional, lowercase, NO AI attribution or Co-Authored-By of any kind.\n` +
    `3. On ANY conflict: git -C ${repo} merge --abort and report conflict=true. Do not resolve conflicts unattended.\n` +
    `4. On success: git -C ${repo} push origin main. Report the merge sha.`,
    {
      label: `merge:${tag}`, phase: 'Slices', effort: 'low',
      schema: {
        type: 'object', additionalProperties: false,
        properties: {
          mergedSha: { type: 'string' },
          conflict: { type: 'boolean' },
          detail: { type: 'string' },
        },
        required: ['mergedSha', 'conflict', 'detail'],
      },
    },
  )
  if (!mg || mg.conflict || !mg.mergedSha) {
    stopReason = `${tag}: merge did not complete (${mg ? mg.detail : 'merge agent failed'}) — queue stopped, branch ${branch} intact`
    perSlice.push({ slice: tag, outcome: 'STOPPED_MERGE', verdict, branch, fixReport })
    break
  }

  merged.push({ slice: tag, branch, mergedSha: mg.mergedSha, autoFixed: fixReport.applied, fixSkipped: fixReport.skipped })
  perSlice.push({ slice: tag, outcome: 'MERGED', verdict, concernsRemaining: (res.concerns || []).length, branch, mergedSha: mg.mergedSha })
  log(`[overnight] ${tag}: MERGED ${mg.mergedSha} (${fixReport.applied.length} should-fixes applied)`)
}

phase('Report')
return {
  queueLength: queue.length,
  completed: merged.length,
  merged,
  stopReason,
  perSlice,
  morning: stopReason
    ? `Queue stopped after ${merged.length}/${queue.length}: ${stopReason}`
    : `All ${merged.length} slice(s) merged to main. Review the per-slice concerns lists at leisure.`,
}
