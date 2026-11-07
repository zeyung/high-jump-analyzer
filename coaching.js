// Coaching: turns an analysis into feedback for this jump, plus approach
// fundamentals, drills for the weakest phase, and a practice checklist.
// Pure functions, no DOM.

const STRENGTHS = {
  approachSpeed: 'Good approach speed going into the plant.',
  curveLean: 'Good lean into the curve.',
  plantKnee: 'Firm, nearly straight takeoff leg at the plant.',
  plantHip: 'Shoulders stay back over the hips at the plant.',
  takeoffAngle: 'Good takeoff angle: you go up, not through.',
  contactTime: 'Quick takeoff: the foot is down only briefly.',
  rise: 'Strong vertical lift for your height.',
  barMargin: 'Your hips peak above the bar.',
  clearance: 'Every body part cleared the bar.',
  torsoAtPeak: 'Flat back over the bar at the top.',
  landing: 'Safe landing on the upper back.',
};

export const FUNDAMENTALS = [
  { title: 'J-shaped approach', text: 'Run 4–5 straight steps, then 4–5 steps on a curve (8–10 in all). Start the curve at the same mark every time.' },
  { title: 'Build speed gradually', text: 'Accelerate through the run so the last three steps are the fastest, without overstriding.' },
  { title: 'Lean from the ankles', text: 'In the curve, tilt the whole body away from the bar like a bike in a turn. Do not drop the inside shoulder or bend at the waist.' },
  { title: 'Penultimate long, last step quick', text: 'The second-to-last step is slightly longer and flatter, which lowers the hips. The last step is short and fast into the plant.' },
  { title: 'Plant', text: 'Land heel first on a firm, nearly straight leg, foot pointing toward the far standard, about an arm’s length (60–100 cm) out from the bar near the near standard.' },
  { title: 'Drive up', text: 'Drive the free knee up and across and swing the arms up, stopping at head height. Stay tall; do not lean into the bar.' },
  { title: 'Over the bar', text: 'Head back and hips up as you rise; once the hips pass the bar, chin to chest to lift the legs over. Land on the upper back.' },
];

export const DRILLS = {
  approach: [
    'Mark check: a coach watches where your 4th-to-last step lands and adjusts the start mark by foot lengths.',
    'Approach run-throughs at full speed without jumping, hitting the same marks each time.',
  ],
  curve: [
    'Circle runs on a 5–6 m radius, leaning from the ankles, both directions.',
    'Curve run-throughs: the last 5 steps of the curve at full speed, finishing in a plant without jumping.',
  ],
  takeoff: [
    'Walk-in plants: heel first, firm leg, foot toward the far standard, knee and arms drive up.',
    'Penultimate rhythm: "long-flat, short-quick" over the last two steps, then pop up.',
    '3- and 5-step pop-ups onto the mat with no bar, focusing on going straight up.',
  ],
  clearance: [
    'Back-overs: standing two-foot jump backward onto the mat, head back, hips up.',
    'Bridges and back extensions to build the arch.',
    'Short-approach jumps over a low bar focusing on hips up, then chin to chest.',
  ],
  landing: [
    'Back-overs finishing with chin to chest so you land on the shoulders, not the head or hips.',
  ],
};

export const CHECKLIST = [
  'Film every jump from the same spot: takeoff side, bar across the picture, both standards in view, phone on a tripod.',
  'Measure your approach marks (start and start of the curve) from the near standard in foot lengths, and write them down.',
  'Log every jump: bar height, cleared or knocked, approach marks and this rating, so you can see trends.',
  'Change one thing at a time and judge it over several jumps, not just one.',
  'Use the same warm-up every session, with a few short-approach jumps before the full approach.',
  'Rest 3–5 minutes between full-approach jumps so speed and quality stay high.',
];

/** Step rhythm feedback from the last contacts: [{foot, length, duration}]. */
export function stepFeedback(steps) {
  const out = { good: [], fix: [] };
  if (!steps || steps.length < 3) return out;
  const last = steps[steps.length - 1];
  const pen = steps[steps.length - 2];
  if (last.length && pen.length) {
    const r = last.length / pen.length;
    if (r > 1.1) out.fix.push('Your last step is longer than the one before it. Make the second-to-last step the long, flat one and the last step short and quick.');
    else if (r < 0.97) out.good.push('Good rhythm: your last step is shorter than the one before it.');
  }
  if (last.duration && pen.duration) {
    if (last.duration > pen.duration * 1.08) out.fix.push('Your last step takes longer than the one before it, so you slow down into the plant. Keep the last step the quickest.');
    else if (last.duration < pen.duration * 0.97) out.good.push('Good rhythm: your last step is your quickest.');
  }
  return out;
}

/**
 * analysis: result of analyzeJump. Returns {strengths, fixes, drills,
 * weakestPhase, fundamentals, checklist}. fixes are ordered most important
 * first (lowest score, heaviest phase).
 */
export function coachJump(analysis) {
  const m = analysis.metrics;
  const strengths = [];
  const fixes = [];
  const phaseOf = {};
  for (const p of analysis.phases) for (const k of (PHASE_KEYS[p.key] || [])) phaseOf[k] = p;
  for (const [key, metric] of Object.entries(m)) {
    if (metric.score === null || metric.score === undefined) continue;
    if (metric.score >= 100 && STRENGTHS[key]) strengths.push(STRENGTHS[key]);
    else if (metric.tip) fixes.push({ key, text: metric.tip, label: metric.label, score: metric.score, weight: phaseOf[key]?.weight ?? 10 });
  }
  fixes.sort((a, b) => (a.score - b.score) || (b.weight - a.weight));
  const steps = stepFeedback(analysis.info.steps);
  strengths.push(...steps.good);
  const stepFixes = steps.fix.map((text) => ({ key: 'steps', text, label: 'Last steps', score: 50, weight: 20 }));
  const allFixes = [...fixes, ...stepFixes].sort((a, b) => (a.score - b.score) || (b.weight - a.weight));
  const scored = analysis.phases.filter((p) => p.score !== null);
  const weakest = scored.length ? scored.reduce((a, b) => (b.score < a.score ? b : a)) : null;
  return {
    strengths,
    fixes: allFixes,
    weakestPhase: weakest,
    drills: weakest ? DRILLS[weakest.key] || [] : [],
    fundamentals: FUNDAMENTALS,
    checklist: CHECKLIST,
  };
}

const PHASE_KEYS = {
  approach: ['approachSpeed'],
  curve: ['curveLean'],
  takeoff: ['plantKnee', 'plantHip', 'takeoffAngle', 'contactTime'],
  clearance: ['rise', 'barMargin', 'clearance', 'torsoAtPeak'],
  landing: ['landing'],
};
