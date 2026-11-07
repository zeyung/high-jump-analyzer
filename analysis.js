// Pure jump analysis: no DOM, no video. Takes pose landmarks per frame and
// returns metrics, personalized targets, phase scores and a 0-100 rating.
// Landmark indices follow MediaPipe Pose (33 points).

export const LM = {
  nose: 0,
  lShoulder: 11, rShoulder: 12,
  lHip: 23, rHip: 24,
  lKnee: 25, rKnee: 26,
  lAnkle: 27, rAnkle: 28,
};

const G_CM = 981; // gravity, cm/s^2
// Nose sits at about 89% of standing height above the ankle joint.
const NOSE_TO_ANKLE_RATIO = 0.89;

export const SOURCES = [
  'Dapena, J. (1980). Mechanics of translation in the Fosbury-flop. Medicine and Science in Sports and Exercise, 12(1), 37-44.',
  'Dapena, J. (1980). Mechanics of rotation in the Fosbury-flop. Medicine and Science in Sports and Exercise, 12(1), 45-53.',
  'Isolehto, J., Virmavirta, M., Kyröläinen, H., & Komi, P. V. (2007). Biomechanical analysis of the high jump at the 2005 IAAF World Championships. New Studies in Athletics, 22(2), 17-27.',
];

export function angleAt(a, b, c) {
  const v1x = a.x - b.x, v1y = a.y - b.y;
  const v2x = c.x - b.x, v2y = c.y - b.y;
  const d = Math.hypot(v1x, v1y) * Math.hypot(v2x, v2y);
  if (!d) return NaN;
  const cos = Math.max(-1, Math.min(1, (v1x * v2x + v1y * v2y) / d));
  return (Math.acos(cos) * 180) / Math.PI;
}

export const mid = (a, b) => ({ x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 });
const dist = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);
const deg = (r) => (r * 180) / Math.PI;

export function toPx(lms, width, height) {
  return lms.map((p) => ({ x: p.x * width, y: p.y * height, v: p.visibility ?? 1 }));
}

export function percentile(arr, p) {
  const s = arr.filter(Number.isFinite).sort((a, b) => a - b);
  if (!s.length) return NaN;
  const k = (s.length - 1) * p;
  const f = Math.floor(k), c = Math.ceil(k);
  return s[f] + (s[c] - s[f]) * (k - f);
}

function smooth(ys, w) {
  const h = Math.floor(w / 2);
  return ys.map((_, i) => {
    let s = 0, n = 0;
    for (let j = i - h; j <= i + h; j++) if (j >= 0 && j < ys.length) { s += ys[j]; n++; }
    return s / n;
  });
}

export function linearSlope(ts, ys) {
  const n = ts.length;
  if (n < 2) return NaN;
  const mt = ts.reduce((a, b) => a + b, 0) / n;
  const my = ys.reduce((a, b) => a + b, 0) / n;
  let num = 0, den = 0;
  for (let i = 0; i < n; i++) { num += (ts[i] - mt) * (ys[i] - my); den += (ts[i] - mt) ** 2; }
  return den ? num / den : NaN;
}

// Least-squares fit of y = a + b t + c t^2. Returns {a, b, c} or null.
export function fitQuadratic(ts, ys) {
  if (ts.length < 3) return null;
  let s0 = 0, s1 = 0, s2 = 0, s3 = 0, s4 = 0, y0 = 0, y1 = 0, y2 = 0;
  for (let i = 0; i < ts.length; i++) {
    const t = ts[i], y = ys[i], t2 = t * t;
    s0 += 1; s1 += t; s2 += t2; s3 += t2 * t; s4 += t2 * t2;
    y0 += y; y1 += y * t; y2 += y * t2;
  }
  // Solve the 3x3 normal equations with Cramer's rule.
  const det3 = (m) =>
    m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) -
    m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) +
    m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
  const M = [[s0, s1, s2], [s1, s2, s3], [s2, s3, s4]];
  const D = det3(M);
  if (!D) return null;
  const col = (k, v) => M.map((row, r) => row.map((x, c) => (c === k ? v[r] : x)));
  const Y = [y0, y1, y2];
  return { a: det3(col(0, Y)) / D, b: det3(col(1, Y)) / D, c: det3(col(2, Y)) / D };
}

// Targets adjusted for the jumper. Ranges are rounded from the elite values in
// SOURCES and widened for non-elite jumpers; they are guides, not lab norms.
// Height enters twice: approach speed scales with sqrt(height / reference
// height) (taller athletes run and jump at a larger scale), and vertical lift
// is a fraction of the jumper's own height.
export function targetsFor(gender, heightCm) {
  const female = gender === 'female';
  const ref = female ? 168 : 180;
  const k = Math.min(1.1, Math.max(0.9, Math.sqrt(heightCm / ref)));
  const r1 = (x) => Math.round(x * 10) / 10;
  return {
    approachSpeed: { range: (female ? [5.0, 7.0] : [5.5, 8.0]).map((v) => r1(v * k)), tol: 2.5 },
    curveLean: { range: [10, 20], tol: 10 },
    plantKnee: { range: [160, 180], tol: 20 },
    plantHip: { range: [160, 180], tol: 25 },
    takeoffAngle: { range: female ? [44, 56] : [42, 52], tol: 15 },
    rise: { range: (female ? [0.3, 0.48] : [0.35, 0.55]).map((r) => Math.round(r * heightCm)), tol: 0.2 * heightCm },
    torsoAtPeak: { range: [0, 25], tol: 30 },
    // Peak hip height minus bar height. Hips at or above the bar is the aim;
    // in a good flop the hips pass close over the bar, so there is no upper limit.
    barMargin: { range: [0, Infinity], tol: 30 },
    // Lowest body part's height over the bar at its highest point.
    clearance: { range: [0, Infinity], tol: 20 },
  };
}

export function scoreRange(v, [lo, hi], tol) {
  if (!Number.isFinite(v)) return null;
  if (v >= lo && v <= hi) return 100;
  const d = v < lo ? lo - v : v - hi;
  return Math.max(0, Math.round(100 - (100 * d) / tol));
}

const METRIC_INFO = {
  approachSpeed: {
    label: 'Approach speed (last 0.5 s)', unit: 'm/s', digits: 1,
    low: 'Build more speed over the last steps. A faster approach gives you more energy to turn into height.',
    high: 'Your approach may be too fast to control. Use the fastest speed you can still plant cleanly from.',
  },
  curveLean: {
    label: 'Lean into the curve', unit: '°', digits: 0,
    low: 'Lean more into the curve, away from the bar, over the last steps. It lowers your hips before takeoff and sets up the rotation over the bar.',
    high: 'You lean a lot in the curve. Too much lean can make the takeoff drift into the bar.',
  },
  plantKnee: {
    label: 'Takeoff knee angle at plant', unit: '°', digits: 0,
    low: 'Your takeoff leg is bent when it lands. Plant on a nearly straight leg so it can act like a vaulting pole.',
    high: '',
  },
  plantHip: {
    label: 'Hip angle at plant', unit: '°', digits: 0,
    low: 'You fold forward at the hips at plant. Keep your shoulders back over or behind your hips at touchdown.',
    high: '',
  },
  takeoffAngle: {
    label: 'Takeoff angle', unit: '°', digits: 0,
    low: 'Your takeoff is flat, so you run through the jump. Drive up with the free knee and both arms.',
    high: 'Your takeoff is very steep. You may be jumping straight up instead of travelling over the bar.',
  },
  rise: {
    label: 'Vertical jump (hip rise after takeoff)', unit: 'cm', digits: 0,
    low: 'Not much vertical lift. Build takeoff power with single-leg bounds, hops and plyometrics.',
    high: '',
  },
  barMargin: {
    label: 'Peak hip height vs bar', unit: 'cm', digits: 0,
    low: 'Your hips peaked below the bar. You need more lift at takeoff, or a takeoff spot that puts your peak over the bar rather than before or after it.',
    high: '',
  },
  clearance: {
    label: 'Bar clearance (lowest body part)', unit: 'cm', digits: 0,
    low: '', // filled in with the body part that was lowest
    high: '',
  },
  torsoAtPeak: {
    label: 'Back angle at peak (0° = flat)', unit: '°', digits: 0,
    low: '',
    high: 'Your back is not flat over the bar at the top. Work on the arch with bridges and back-overs.',
  },
  landing: {
    label: 'Lands on upper back', unit: '', digits: 0,
    low: 'You land feet or hips first. Finish the rotation so you land on your upper back and shoulders.',
    high: '',
  },
};

const PHASES = [
  { key: 'approach', label: 'Approach', weight: 15, metrics: ['approachSpeed'] },
  { key: 'curve', label: 'Curve lean', weight: 20, metrics: ['curveLean'] },
  { key: 'takeoff', label: 'Takeoff', weight: 30, metrics: ['plantKnee', 'plantHip', 'takeoffAngle'] },
  { key: 'clearance', label: 'Bar clearance', weight: 25, metrics: ['rise', 'barMargin', 'clearance', 'torsoAtPeak'] },
  { key: 'landing', label: 'Landing', weight: 10, metrics: ['landing'] },
];

export function ratingLabel(score) {
  if (score >= 90) return 'Excellent';
  if (score >= 75) return 'Good';
  if (score >= 60) return 'Fair';
  return 'Needs work';
}

// Body parts checked against the bar. Pose points are joint centres, but the
// bar touches the skin on the back side of the body (a flop goes over face
// up), so each part has an allowance from joint centre to the back surface,
// for a 175 cm athlete; it scales with height.
export const CLEARANCE_PARTS = [
  { key: 'head', label: 'Head', points: [[0]], allowance: 18 },
  { key: 'shoulders', label: 'Shoulders', points: [[11, 12]], allowance: 8 },
  { key: 'hips', label: 'Hips', points: [[23, 24]], allowance: 10 },
  { key: 'knees', label: 'Knees', points: [[25], [26]], allowance: 5 },
  { key: 'feet', label: 'Feet', points: [[27], [28]], allowance: 5 },
];

const CLEARANCE_TIPS = {
  head: 'Your head was lowest over the bar. Tuck your chin and look along the bar so your head drops behind it as your hips rise.',
  shoulders: 'Your shoulders were lowest over the bar. Lay back sooner after takeoff so your shoulders drop and your hips rise.',
  hips: 'Your hips were lowest over the bar. Push your hips up into an arch at the top (think "belly button to the sky").',
  knees: 'Your knees were lowest over the bar. Lift your knees by tucking your chin to your chest once your hips pass the bar.',
  feet: 'Your feet were lowest over the bar. Kick your heels up and tuck your chin as your hips pass the bar.',
};

// Index of the earliest frame no more than dt seconds before frame k.
function backBy(V, k, dt) {
  let j = k;
  while (j > 0 && V[k].t - V[j - 1].t <= dt + 1e-9) j--;
  return j;
}

/**
 * frames: [{t: seconds, lms: [{x, y, visibility}] x33 normalized, or null}]
 * opts: {width, height, heightCm, gender: 'male'|'female', barCm?, bar?}
 * bar (optional, needs barCm): the bar marked on the video, normalized
 *   {a: {x, y}, b: {x, y}} = the two ends of the bar, and
 *   ground: {x, y} = the floor under one standard. It gives the exact bar
 *   line and a better scale than the athlete's height.
 */
export function analyzeJump(frames, { width, height, heightCm, gender, barCm = null, bar = null }) {
  if (barCm !== null && barCm !== undefined && !(barCm >= 50 && barCm <= 260)) {
    throw new Error('Bar height must be between 50 and 260 cm.');
  }
  const hasBar = barCm !== null && barCm !== undefined;
  if (!(heightCm >= 100 && heightCm <= 230)) throw new Error('Height must be between 100 and 230 cm.');
  if (gender !== 'male' && gender !== 'female') throw new Error('Gender must be male or female.');

  const V = [];
  frames.forEach((f, i) => {
    if (f && f.lms && f.lms.length >= 33) V.push({ i, t: f.t, P: toPx(f.lms, width, height) });
  });
  if (V.length < 15) {
    throw new Error(`A person was found in only ${V.length} frames. Use a clip where the whole jumper is visible from the last steps through the landing.`);
  }

  const hip = V.map((v) => mid(v.P[LM.lHip], v.P[LM.rHip]));
  const sh = V.map((v) => mid(v.P[LM.lShoulder], v.P[LM.rShoulder]));
  const ank = V.map((v) => mid(v.P[LM.lAnkle], v.P[LM.rAnkle]));
  const lowAnkY = V.map((v) => Math.max(v.P[LM.lAnkle].y, v.P[LM.rAnkle].y));
  const hipYs = smooth(hip.map((p) => p.y), 3);

  // Peak: highest hip position (smallest y, image y points down).
  let peak = 0;
  hipYs.forEach((y, k) => { if (y < hipYs[peak]) peak = k; });
  if (peak < 5) throw new Error('The clip seems to start at or after the peak of the jump. Include the last approach steps.');

  const pre = [...Array(peak).keys()];
  const bodyPx = percentile(pre.map((k) => dist(V[k].P[LM.nose], ank[k])), 0.9) / NOSE_TO_ANKLE_RATIO;
  const cmPerPxFromHeight = heightCm / bodyPx;
  const groundY = percentile(pre.map((k) => lowAnkY[k]), 0.95);

  // Bar line in pixels, and the scale it gives when marked.
  let barLine = null, cmPerPx = cmPerPxFromHeight, scaleSource = 'athlete height';
  if (hasBar && bar && bar.a && bar.b && bar.ground) {
    const A = { x: bar.a.x * width, y: bar.a.y * height };
    const B = { x: bar.b.x * width, y: bar.b.y * height };
    const Gp = { x: bar.ground.x * width, y: bar.ground.y * height };
    const yAt = (x) => (B.x === A.x ? (A.y + B.y) / 2 : A.y + ((B.y - A.y) * (x - A.x)) / (B.x - A.x));
    const barPx = Gp.y - yAt(Gp.x);
    if (barPx > 10) {
      cmPerPx = barCm / barPx;
      scaleSource = 'marked bar';
      barLine = { x1: A.x, y1: A.y, x2: B.x, y2: B.y, yAt, marked: true };
    }
  }
  if (hasBar && !barLine) {
    const y = groundY - barCm / cmPerPx;
    barLine = { x1: 0, y1: y, x2: width, y2: y, yAt: () => y, marked: false };
  }
  const tol = 0.03 * bodyPx;

  // Takeoff (toe-off): last frame before the peak with a foot on the ground.
  let takeoff = -1;
  for (let k = peak - 1; k >= 0; k--) if (lowAnkY[k] >= groundY - tol) { takeoff = k; break; }
  if (takeoff < 0) throw new Error('Could not find the takeoff. Make sure the takeoff foot is visible.');

  let landing = V.length - 1;
  for (let k = peak + 1; k < V.length; k++) if (hipYs[k] >= hipYs[takeoff]) { landing = k; break; }

  // Plant (touchdown of the takeoff foot) is about 0.18 s before toe-off;
  // the curve phase covers the 0.25 s before the plant.
  const plant = backBy(V, takeoff, 0.18);
  const curve = backBy(V, plant, 0.25);

  const leanAt = (k) => deg(Math.atan2(Math.abs(sh[k].x - ank[k].x), ank[k].y - sh[k].y));
  const leanFrames = [];
  for (let k = curve; k <= plant; k++) leanFrames.push(leanAt(k));
  const curveLean = leanFrames.reduce((a, b) => a + b, 0) / leanFrames.length;

  // Takeoff leg: the foot still lowest at toe-off.
  const P0 = V[takeoff].P;
  const side = P0[LM.lAnkle].y >= P0[LM.rAnkle].y ? 'left' : 'right';
  const J = side === 'left'
    ? { sh: LM.lShoulder, hip: LM.lHip, knee: LM.lKnee, ankle: LM.lAnkle }
    : { sh: LM.rShoulder, hip: LM.rHip, knee: LM.rKnee, ankle: LM.rAnkle };
  const Pp = V[plant].P;
  const plantKnee = angleAt(Pp[J.hip], Pp[J.knee], Pp[J.ankle]);
  const plantHip = angleAt(Pp[J.sh], Pp[J.hip], Pp[J.knee]);

  // Flight: fit a parabola to the hip path to get launch angle and rise.
  let end = landing;
  if (end - takeoff < 3) end = Math.min(V.length - 1, takeoff + 6);
  const fts = [], fys = [], fxs = [];
  for (let k = takeoff; k <= end; k++) { fts.push(V[k].t - V[takeoff].t); fys.push(hip[k].y); fxs.push(hip[k].x); }
  const q = fitQuadratic(fts, fys);
  const vx = Math.abs(linearSlope(fts, fxs));
  let takeoffAngle = NaN, risePx = hip[takeoff].y - Math.min(...hip.map((p) => p.y));
  if (q && q.c > 0) {
    takeoffAngle = deg(Math.atan2(-q.b, vx));
    const tv = -q.b / (2 * q.c);
    risePx = hip[takeoff].y - (q.a + q.b * tv + q.c * tv * tv);
  }
  const rise = risePx * cmPerPx;
  const hipAtTakeoff = (groundY - hip[takeoff].y) * cmPerPx;
  const flightTime = V[landing].t - V[takeoff].t;

  const a0 = backBy(V, plant, 0.5);
  const ats = [], axs = [];
  for (let k = a0; k <= plant; k++) { ats.push(V[k].t); axs.push(hip[k].x); }
  const approachSpeed = (Math.abs(linearSlope(ats, axs)) * cmPerPx) / 100;

  const tp = { dx: sh[peak].x - hip[peak].x, dy: sh[peak].y - hip[peak].y };
  const torsoAtPeak = deg(Math.atan2(Math.abs(tp.dy), Math.abs(tp.dx)));
  const torsoDir = (k) => deg(Math.atan2(sh[k].x - hip[k].x, hip[k].y - sh[k].y));
  let rotation = torsoDir(peak) - torsoDir(plant);
  rotation = Math.abs(((rotation + 540) % 360) - 180);
  const landsOnBack = sh[landing].y >= hip[landing].y - 0.05 * bodyPx;

  const targets = targetsFor(gender, heightCm);
  const peakHipHeight = hipAtTakeoff + rise;

  // Clearance: every body part has to be above the bar at some moment in the
  // air, so compare each part's best height over the bar line during flight.
  // This does not need to know when each part crossed the bar plane.
  let clearance = null;
  if (barLine) {
    const scaleAllow = heightCm / 175;
    const flightEnd = Math.max(landing, Math.min(V.length - 1, takeoff + 3));
    const parts = CLEARANCE_PARTS.map((part) => {
      let worst = null; // worst side for left/right parts
      for (const pts of part.points) {
        let best = -Infinity, bestK = takeoff;
        for (let k = takeoff; k <= flightEnd; k++) {
          const P = V[k].P;
          const x = pts.reduce((s, j) => s + P[j].x, 0) / pts.length;
          const y = pts.reduce((s, j) => s + P[j].y, 0) / pts.length;
          const above = barLine.yAt(x) - y;
          if (above > best) { best = above; bestK = k; }
        }
        const cm = best * cmPerPx - part.allowance * scaleAllow;
        if (!worst || cm < worst.cm) worst = { cm, frame: V[bestK].i };
      }
      return { key: part.key, label: part.label, cm: worst.cm, frame: worst.frame };
    });
    const lowest = parts.reduce((a, b) => (b.cm < a.cm ? b : a));
    const verdict = lowest.cm >= 2 ? 'cleared' : lowest.cm >= -3 ? 'close' : 'knocked';
    clearance = { parts, lowest: lowest.key, lowestLabel: lowest.label, lowestCm: lowest.cm, verdict, method: barLine.marked ? 'marked bar' : 'bar height' };
  }

  const values = { approachSpeed, curveLean, plantKnee, plantHip, takeoffAngle, rise, torsoAtPeak };
  if (hasBar) values.barMargin = peakHipHeight - barCm;
  if (clearance) values.clearance = clearance.lowestCm;
  const metrics = {};
  for (const [key, value] of Object.entries(values)) {
    const tgt = targets[key];
    const score = scoreRange(value, tgt.range, tgt.tol);
    const info = METRIC_INFO[key];
    let tip = '';
    if (score !== null && score < 100) tip = value < tgt.range[0] ? info.low : info.high;
    if (key === 'clearance' && score < 100) tip = CLEARANCE_TIPS[clearance.lowest];
    metrics[key] = { key, label: info.label, unit: info.unit, digits: info.digits, value, target: tgt.range, score, tip };
  }
  metrics.landing = {
    key: 'landing', label: METRIC_INFO.landing.label, unit: '', digits: 0,
    value: landsOnBack ? 'Yes' : 'No', target: null, score: landsOnBack ? 100 : 40,
    tip: landsOnBack ? '' : METRIC_INFO.landing.low,
  };

  const phases = PHASES.map((p) => {
    const scores = p.metrics.filter((m) => metrics[m]).map((m) => metrics[m].score).filter((s) => s !== null);
    const score = scores.length ? Math.round(scores.reduce((a, b) => a + b, 0) / scores.length) : null;
    return { key: p.key, label: p.label, weight: p.weight, score };
  });
  const scored = phases.filter((p) => p.score !== null);
  const wsum = scored.reduce((a, p) => a + p.weight, 0);
  const overall = wsum ? Math.round(scored.reduce((a, p) => a + p.weight * p.score, 0) / wsum) : 0;

  return {
    overall,
    rating: ratingLabel(overall),
    phases,
    metrics,
    info: {
      takeoffLeg: side,
      hipHeightAtTakeoff: hipAtTakeoff,
      peakHipHeight,
      barCm: hasBar ? barCm : null,
      barPercentOfHeight: hasBar ? (100 * barCm) / heightCm : null,
      liftNeededForBar: hasBar ? barCm - hipAtTakeoff : null,
      flightTime,
      riseFromFlightTime: (G_CM * flightTime * flightTime) / 8,
      rotation,
      cmPerPx,
      cmPerPxFromHeight,
      scaleSource,
      verticalJump: rise,
      clearance,
      barLine: barLine ? { x1: barLine.x1, y1: barLine.y1, x2: barLine.x2, y2: barLine.y2, marked: barLine.marked } : null,
      groundY,
    },
    keyFrames: {
      curve: V[curve].i, plant: V[plant].i, takeoff: V[takeoff].i, peak: V[peak].i, landing: V[landing].i,
    },
    sources: SOURCES,
  };
}

// Phase name for a frame index, using the key frames from analyzeJump.
export function phaseAt(i, kf) {
  if (i < kf.curve) return 'Approach';
  if (i < kf.plant) return 'Curve';
  if (i <= kf.takeoff) return 'Plant & takeoff';
  if (i <= kf.peak) return 'Flight: rising';
  if (i < kf.landing) return 'Flight: clearing';
  return 'Landing';
}
