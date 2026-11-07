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
// Body proportions as a fraction of standing height: shoulder-to-hip torso
// about 0.29 and hip-knee-ankle leg about 0.49, together 0.78. The ankle
// joint is about 0.04 above the floor.
const SEGMENT_RATIO = 0.78;
const ANKLE_HEIGHT_RATIO = 0.04;

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

/**
 * Flight path with known gravity. In the air the hips follow
 * y = a + b t + g2 t^2, where g2 = half of gravity in pixels per second
 * squared (known once the scale is known). Only a and b are fitted, using
 * the pair of frames that the most other frames agree with (RANSAC), so a
 * few wrong frames (a different person, a bad pose) are ignored.
 * ts: seconds from takeoff, ys: hip y (pixels, down is +), tol: pixels.
 * Returns {a, b, inliers: boolean[]} or null.
 */
export function fitFlight(ts, ys, g2, tol) {
  const n = ts.length;
  if (n < 3) return null;
  const z = ys.map((y, i) => y - g2 * ts[i] * ts[i]); // = a + b t
  let best = null;
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      if (Math.abs(ts[j] - ts[i]) < 1e-6) continue;
      const b = (z[j] - z[i]) / (ts[j] - ts[i]);
      const a = z[i] - b * ts[i];
      let cnt = 0, err = 0;
      for (let k = 0; k < n; k++) { const r = Math.abs(z[k] - (a + b * ts[k])); if (r < tol) { cnt++; err += r; } }
      if (!best || cnt > best.cnt || (cnt === best.cnt && err < best.err)) best = { a, b, cnt, err };
    }
  }
  if (!best || best.cnt < 3) return null;
  // Refine on the inliers by least squares.
  const inl = ts.map((_, k) => Math.abs(z[k] - (best.a + best.b * ts[k])) < tol);
  const it = ts.filter((_, k) => inl[k]), iz = z.filter((_, k) => inl[k]);
  const b = linearSlope(it, iz);
  const a = iz.reduce((s, v) => s + v, 0) / iz.length - b * (it.reduce((s, v) => s + v, 0) / it.length);
  return { a, b, inliers: ts.map((_, k) => Math.abs(z[k] - (a + b * ts[k])) < tol) };
}

/**
 * Flight curve without assuming a scale: fit y = a + b t + c t^2 by RANSAC
 * over triples of frames, so frames of the wrong person are ignored. The
 * time to the top (-b / 2c) does not depend on the camera's distance or
 * angle, and gravity turns it into a real height; the curve's bend (c) also
 * gives the scale at the jumper's distance mid-flight.
 */
export function fitFlightFree(ts, ys, tol) {
  const n = ts.length;
  if (n < 5) return null;
  let best = null;
  const solve = (i, j, k) => fitQuadratic([ts[i], ts[j], ts[k]], [ys[i], ys[j], ys[k]]);
  for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) for (let k = j + 1; k < n; k++) {
    const q = solve(i, j, k);
    if (!q || !(q.c > 0)) continue;
    let cnt = 0, err = 0;
    for (let m = 0; m < n; m++) { const r = Math.abs(ys[m] - (q.a + q.b * ts[m] + q.c * ts[m] * ts[m])); if (r < tol) { cnt++; err += r; } }
    if (!best || cnt > best.cnt || (cnt === best.cnt && err < best.err)) best = { q, cnt, err };
  }
  if (!best || best.cnt < 5) return null;
  const inl = ts.map((tt, m) => Math.abs(ys[m] - (best.q.a + best.q.b * tt + best.q.c * tt * tt)) < tol);
  const q = fitQuadratic(ts.filter((_, m) => inl[m]), ys.filter((_, m) => inl[m])) || best.q;
  if (!(q.c > 0)) return null;
  return { a: q.a, b: q.b, c: q.c, inliers: ts.map((tt, m) => Math.abs(ys[m] - (q.a + q.b * tt + q.c * tt * tt)) < tol) };
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
    // Time the takeoff foot is on the ground, plant to toe-off. Elite flop
    // takeoffs last roughly 0.15-0.20 s; longer usually means a sinking,
    // braking plant.
    contactTime: { range: [0.13, 0.22], tol: 0.12 },
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
  contactTime: {
    label: 'Takeoff contact time', unit: 's', digits: 2,
    low: '',
    high: 'Your takeoff foot stays down a long time. Plant quickly and actively (think "fast feet" on the last two steps) so the takeoff is a quick pop, not a sink.',
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
  { key: 'takeoff', label: 'Takeoff', weight: 30, metrics: ['plantKnee', 'plantHip', 'takeoffAngle', 'contactTime'] },
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
 * opts: {width, height, heightCm, gender: 'male'|'female', barCm?, bar?, overrides?}
 * bar (optional, needs barCm): the bar in the video, normalized
 *   {a: {x, y}, b: {x, y}} = its two ends, and optionally ground: {x, y} =
 *   the floor under the bar. Without ground, the floor at the takeoff spot is
 *   used. The bar is taken to be exactly barCm high, which sets the scale.
 * overrides (optional): {plant, takeoff, peak, landing} frame indices set by hand.
 * takeoffFoot (optional): 'left' | 'right', the foot the athlete jumps off.
 *   Only that foot is searched for the takeoff contact.
 */
export function analyzeJump(frames, { width, height, heightCm, gender, barCm = null, bar = null, overrides = null, takeoffFoot = null }) {
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
  const ankles = [V.map((v) => v.P[LM.lAnkle]), V.map((v) => v.P[LM.rAnkle])];
  const hipYs = smooth(hip.map((p) => p.y), 3);

  // Standing height in pixels from rigid body segments, so posture (lean,
  // bent knees, lying flat in the air) does not change it.
  const segPx = V.map((v) => {
    const P = v.P;
    const torso = dist(mid(P[11], P[12]), mid(P[23], P[24]));
    const legL = dist(P[23], P[25]) + dist(P[25], P[27]);
    const legR = dist(P[24], P[26]) + dist(P[26], P[28]);
    return (torso + Math.max(legL, legR)) / SEGMENT_RATIO;
  });
  let bodyPx = percentile(segPx, 0.75);

  const idxOf = (orig) => {
    if (orig === null || orig === undefined) return null;
    let best = 0;
    V.forEach((v, k) => { if (Math.abs(v.i - orig) < Math.abs(V[best].i - orig)) best = k; });
    return best;
  };
  const ov = overrides || {};

  // The jumper's size in the picture changes as they come toward or away
  // from the camera, so heights and speeds are judged against their size
  // in that part of the clip.
  const sizeAt = smooth(segPx, 9);
  const localFloor = (k) => {
    let m = -Infinity;
    for (let j = k; j >= 0 && V[k].t - V[j].t <= 0.6; j--) m = Math.max(m, ankles[0][j].y, ankles[1][j].y);
    return m;
  };

  // Peak: where the hips are highest above the jumper's own recent floor,
  // in body heights. (Not simply the highest point in the picture: a
  // jumper far away near the horizon is high in the picture too.)
  const elev = smooth(V.map((_, k) => (localFloor(k) - hip[k].y) / sizeAt[k]), 3);
  let peak = 0;
  elev.forEach((e, k) => { if (e > elev[peak]) peak = k; });
  if (ov.peak !== undefined && ov.peak !== null) peak = idxOf(ov.peak);
  if (peak < 5) throw new Error('The clip seems to start at or after the peak of the jump. Include the last approach steps.');

  // Foot contact: a planted foot stays still (camera pans are already
  // removed). A foot is in contact when it barely moves into or out of the
  // frame and is near the local floor.
  const speedAt = (s, k, dk) => {
    const j = k + dk;
    if (j < 0 || j >= V.length) return Infinity;
    return dist(ankles[s][k], ankles[s][j]) / Math.abs(V[j].t - V[k].t || 1e-6);
  };
  const isStill = (s, k) => Math.min(speedAt(s, k, -1), speedAt(s, k, 1)) < 0.45 * sizeAt[k];
  const onGround = (s, k) => isStill(s, k) && ankles[s][k].y >= localFloor(k) - 0.12 * sizeAt[k];

  let takeoff = -1, side = 0, contactFound = false;
  const known = takeoffFoot === 'left' ? 0 : takeoffFoot === 'right' ? 1 : null;
  for (const sides of known === null ? [[0, 1]] : [[known], [0, 1]]) {
    for (let k = peak - 1; k >= 0 && takeoff < 0; k--) {
      for (const s of sides) if (onGround(s, k)) { takeoff = k; side = s; contactFound = true; break; }
    }
    if (takeoff >= 0) break;
  }
  if (takeoff < 0) {
    // Fallback: last frame before the peak with a foot near the floor.
    const lowAnkY = V.map((v) => Math.max(v.P[LM.lAnkle].y, v.P[LM.rAnkle].y));
    const floor = percentile(lowAnkY.slice(0, peak), 0.95);
    for (let k = peak - 1; k >= 0; k--) if (lowAnkY[k] >= floor - 0.03 * bodyPx) { takeoff = k; break; }
    if (takeoff < 0) throw new Error('Could not find the takeoff. Make sure the takeoff foot is visible.');
    side = V[takeoff].P[LM.lAnkle].y >= V[takeoff].P[LM.rAnkle].y ? 0 : 1;
  }
  if (ov.takeoff !== undefined && ov.takeoff !== null) {
    takeoff = idxOf(ov.takeoff);
    side = V[takeoff].P[LM.lAnkle].y >= V[takeoff].P[LM.rAnkle].y ? 0 : 1;
  }

  // Plant: the first frame of the takeoff foot's last contact.
  let plant = takeoff;
  if (contactFound || (ov.takeoff !== undefined && ov.takeoff !== null)) {
    while (plant > 0 && onGround(side, plant - 1) && V[takeoff].t - V[plant - 1].t <= 0.35) plant--;
  }
  // Smoothing blurs the exact frames where the foot lands and leaves, so
  // widen the contact while the foot is still within a few cm of its spot.
  if (contactFound && takeoff > plant) {
    const xs = [], ys = [];
    for (let k = plant; k <= takeoff; k++) { xs.push(ankles[side][k].x); ys.push(ankles[side][k].y); }
    const spot = { x: percentile(xs, 0.5), y: percentile(ys, 0.5) };
    const near = (k) => dist(ankles[side][k], spot) < 0.05 * bodyPx;
    while (takeoff + 1 < peak && near(takeoff + 1)) takeoff++;
    while (plant > 0 && near(plant - 1) && V[takeoff].t - V[plant - 1].t <= 0.35) plant--;
  }
  if (plant === takeoff) plant = backBy(V, takeoff, 0.15);
  if (ov.plant !== undefined && ov.plant !== null) plant = Math.min(idxOf(ov.plant), takeoff);
  if (peak <= takeoff) peak = Math.min(V.length - 1, takeoff + 1);

  // Landing: the fall stops. After the peak the hips speed up downward,
  // then decelerate sharply on the mat.
  let landing = -1;
  const vy = (k) => {
    const a = Math.max(0, k - 1), b = Math.min(V.length - 1, k + 1);
    return (hipYs[b] - hipYs[a]) / Math.max(V[b].t - V[a].t, 1e-6);
  };
  let vmax = 0;
  for (let k = peak + 1; k < V.length && V[k].t - V[peak].t <= 1.2; k++) {
    const v = vy(k);
    if (v > vmax) vmax = v;
    if (vmax > 0.6 * sizeAt[k] && v <= 0.25 * vmax) { landing = k; break; }
  }
  if (landing < 0) {
    for (let k = peak + 1; k < V.length; k++) if (hipYs[k] >= hipYs[takeoff]) { landing = k; break; }
  }
  if (landing < 0) landing = V.length - 1;
  if (ov.landing !== undefined && ov.landing !== null) landing = Math.max(idxOf(ov.landing), peak);

  const curve = backBy(V, plant, 0.25);

  // Approach steps: every ground contact before the plant, oldest first, as
  // {foot, start, end, spot}. Gives the rhythm and length of the last steps.
  const contacts = [];
  for (const s of [0, 1]) {
    let k = 0;
    while (k < plant) {
      if (!onGround(s, k)) { k++; continue; }
      const k0 = k;
      while (k < plant && onGround(s, k)) k++;
      const xs = [], ys = [];
      for (let j = k0; j < k; j++) { xs.push(ankles[s][j].x); ys.push(ankles[s][j].y); }
      contacts.push({ foot: s === 0 ? 'left' : 'right', start: k0, end: k - 1, spot: { x: percentile(xs, 0.5), y: percentile(ys, 0.5) } });
    }
  }
  contacts.sort((p, q) => p.start - q.start);
  // Walk back from the plant, taking contacts that alternate feet and come
  // one after another, within the last 2 s.
  const steps = [];
  {
    const xs = [], ys = [];
    for (let j = plant; j <= takeoff; j++) { xs.push(ankles[side][j].x); ys.push(ankles[side][j].y); }
    steps.push({ foot: side === 0 ? 'left' : 'right', start: plant, end: takeoff, spot: { x: percentile(xs, 0.5), y: percentile(ys, 0.5) } });
  }
  for (let c = contacts.length - 1; c >= 0; c--) {
    const next = steps[0];
    const cand = contacts[c];
    if (cand.foot === next.foot) continue;
    if (cand.end >= next.start) continue;
    if (V[next.start].t - V[cand.start].t > 0.45) break; // gap too long: missed steps
    if (V[plant].t - V[cand.start].t > 2) break;
    steps.unshift(cand);
  }

  // Scale near takeoff, where the measurements matter (the jumper's size in
  // the picture changes as they come toward or away from the camera).
  const near = [];
  V.forEach((v, k) => { if (Math.abs(v.t - V[takeoff].t) <= 0.5) near.push(segPx[k]); });
  if (near.length >= 3) bodyPx = percentile(near, 0.75);
  const cmPerPxFromHeight = heightCm / bodyPx;

  // Floor at the takeoff spot: the planted ankle plus ankle height.
  const plantFoot = ankles[side][plant];
  const groundY = plantFoot.y + ANKLE_HEIGHT_RATIO * bodyPx;

  // Two scales. Near the takeoff the athlete's own height sets it; over
  // the bar, the bar does (it is exactly barCm high, at the depth where the
  // jumper clears it). They differ when the jumper moves toward or away
  // from the camera in the air.
  const cmPerPx = cmPerPxFromHeight;
  let barLine = null, barScale = null, scaleSource = 'athlete height';
  if (hasBar && bar && bar.a && bar.b) {
    const A = { x: bar.a.x * width, y: bar.a.y * height };
    const B = { x: bar.b.x * width, y: bar.b.y * height };
    const yAt = (x) => (Math.abs(B.x - A.x) < 1e-9 ? (A.y + B.y) / 2 : A.y + ((B.y - A.y) * (x - A.x)) / (B.x - A.x));
    const Gp = bar.ground ? { x: bar.ground.x * width, y: bar.ground.y * height } : { x: plantFoot.x, y: groundY };
    const barPx = Gp.y - yAt(Gp.x);
    if (barPx > 10) {
      barScale = barCm / barPx;
      scaleSource = bar.ground ? 'athlete height at takeoff, bar over the bar' : 'athlete height at takeoff, bar over the bar (floor under the bar not marked)';
      barLine = { x1: A.x, y1: A.y, x2: B.x, y2: B.y, yAt, marked: true, ground: Gp };
    }
  }
  if (hasBar && !barLine) {
    // No bar found in the video: compare heights in cm only (nothing drawn).
    const y = groundY - barCm / cmPerPx;
    barLine = { x1: 0, y1: y, x2: width, y2: y, yAt: () => y, marked: false };
  }

  // Lean in the curve: from the supporting (stiller) foot to the shoulders.
  const leanFrames = [];
  for (let k = curve; k <= plant; k++) {
    const s = Math.min(speedAt(0, k, -1), speedAt(0, k, 1)) <= Math.min(speedAt(1, k, -1), speedAt(1, k, 1)) ? 0 : 1;
    const A = ankles[s][k];
    leanFrames.push(deg(Math.atan2(Math.abs(sh[k].x - A.x), A.y - sh[k].y)));
  }
  const curveLean = leanFrames.reduce((a, b) => a + b, 0) / leanFrames.length;

  const takeoffSide = side === 0 ? 'left' : 'right';
  const J = side === 0
    ? { sh: LM.lShoulder, hip: LM.lHip, knee: LM.lKnee, ankle: LM.lAnkle }
    : { sh: LM.rShoulder, hip: LM.rHip, knee: LM.rKnee, ankle: LM.rAnkle };
  const Pp = V[plant].P;
  const plantKnee = angleAt(Pp[J.hip], Pp[J.knee], Pp[J.ankle]);
  const plantHip = angleAt(Pp[J.sh], Pp[J.hip], Pp[J.knee]);

  // Flight. With the scale known, gravity in pixels is known too, so fit
  // the hips to a true flight curve and ignore frames that do not fit it.
  const g2 = (0.5 * G_CM) / cmPerPx;
  const fIdx = [];
  for (let k = takeoff; k < V.length && V[k].t - V[takeoff].t <= 1.2; k++) fIdx.push(k);
  const fts = fIdx.map((k) => V[k].t - V[takeoff].t);
  const fys = fIdx.map((k) => hip[k].y);
  const tolPx = Math.max(4, 0.06 * bodyPx);
  // Prefer the scale-free fit; fall back to known gravity at the takeoff
  // scale. The free fit is only trusted if its implied scale is sensible.
  let flightFit = null, flightCmPerPx = cmPerPx;
  const free = fitFlightFree(fts.slice(0, 40), fys.slice(0, 40), Math.max(3, 0.03 * bodyPx));
  if (free) {
    const sf = (0.5 * G_CM) / free.c;
    const tv = -free.b / (2 * free.c);
    if (sf / cmPerPx > 0.25 && sf / cmPerPx < 2.5 && tv > 0.15 && tv < 0.7) {
      flightFit = { a: free.a, b: free.b, inliers: [...free.inliers, ...new Array(Math.max(0, fts.length - free.inliers.length)).fill(false)], c: free.c };
      flightCmPerPx = sf;
    }
  }
  if (!flightFit) {
    const fixed = fitFlight(fts, fys, g2, tolPx);
    if (fixed) flightFit = { ...fixed, c: g2 };
  }
  let inFlight = new Set();
  let takeoffAngle = NaN, risePx = NaN, q = null, fitGood = false;
  if (flightFit) {
    const flightKs = fIdx.filter((k, j) => flightFit.inliers[j]);
    // Trust the curve to filter out wrong frames only if it explains most
    // of the flight. When the jumper moves toward or away from the camera
    // the path in the picture is not one clean curve, and the curve would
    // throw away good frames.
    // Good = the curve fits most frames up to its last frame AND has frames
    // after its top (it saw the whole arc, not just the rise).
    const lastK = Math.max(...flightKs);
    const span = fIdx.filter((k) => k <= lastK).length;
    const tvx = -flightFit.b / (2 * flightFit.c);
    const afterTop = flightKs.filter((k) => V[k].t - V[takeoff].t > tvx + 0.02).length;
    fitGood = flightKs.length >= 0.7 * span && afterTop >= 2;
    if (fitGood) {
      flightKs.forEach((k) => inFlight.add(k));
      if (ov.landing === undefined || ov.landing === null) landing = Math.max(peak, Math.min(V.length - 1, lastK + 1));
    } else for (let k = takeoff; k <= landing; k++) inFlight.add(k);
    const early = flightKs.filter((k) => V[k].t - V[takeoff].t <= 0.25);
    const use = early.length >= 3 ? early : flightKs;
    const its = use.map((k) => V[k].t - V[takeoff].t), ixs = use.map((k) => hip[k].x);
    const vx = Math.abs(linearSlope(its, ixs));
    takeoffAngle = deg(Math.atan2(-flightFit.b, vx));
    const tv = -flightFit.b / (2 * flightFit.c);
    risePx = tv > 0 ? (flightFit.b * flightFit.b) / (4 * flightFit.c) : 0;
    q = { a: flightFit.a, b: flightFit.b, c: flightFit.c };
    if (!fitGood) {
      // Rough rise straight from the picture.
      risePx = hip[takeoff].y - Math.min(...[...inFlight].map((k) => hip[k].y));
      flightCmPerPx = cmPerPx;
    }
  } else {
    let end = landing;
    if (end - takeoff < 3) end = Math.min(V.length - 1, takeoff + 6);
    const qts = [], qys = [], qxs = [];
    for (let k = takeoff; k <= end; k++) { qts.push(V[k].t - V[takeoff].t); qys.push(hip[k].y); qxs.push(hip[k].x); }
    q = fitQuadratic(qts, qys);
    const vx = Math.abs(linearSlope(qts, qxs));
    risePx = hip[takeoff].y - Math.min(...hip.slice(takeoff, end + 1).map((p) => p.y));
    if (q && q.c > 0) {
      takeoffAngle = deg(Math.atan2(-q.b, vx));
      const tv = -q.b / (2 * q.c);
      if (tv > 0) risePx = hip[takeoff].y - (q.a + q.b * tv + q.c * tv * tv);
    }
    for (let k = takeoff; k <= landing; k++) inFlight.add(k);
  }
  // If the top of the flight was not tracked (a gap right after the highest
  // tracked frame), use the parabola's vertex for the peak instead.
  let peakFromFit = false;
  if (q && q.c > 0) {
    const tv = -q.b / (2 * q.c);
    const tVertex = V[takeoff].t + tv;
    if (tv > 0 && fitGood && (ov.peak === undefined || ov.peak === null)) {
      let best = peak;
      V.forEach((v, k) => { if (k > takeoff && Math.abs(v.t - tVertex) < Math.abs(V[best].t - tVertex)) best = k; });
      peak = best;
      peakFromFit = true;
    }
  }
  const hipAtTakeoff = (groundY - hip[takeoff].y) * cmPerPx;
  // Peak hip height. With the bar marked: the bar height plus how far the
  // hips get above the bar line, in the bar's scale (same depth as the
  // jumper over the bar). Otherwise: from the flight curve.
  let rise = risePx * flightCmPerPx, verticalFrom = 'flight curve';
  if (barLine && barLine.marked) {
    let topK = -1;
    for (const k of inFlight) if (topK < 0 || hip[k].y < hip[topK].y) topK = k;
    if (topK >= 0) {
      const peakH = barCm + (barLine.yAt(hip[topK].x) - hip[topK].y) * barScale;
      rise = peakH - hipAtTakeoff;
      verticalFrom = 'bar';
    }
  }
  // How much the jumper's size in the picture changes in the air (moving
  // toward or away from the camera makes heights from the picture unreliable).
  const sizeTO = sizeAt[takeoff];
  let sizeChange = 0;
  for (const k of inFlight) sizeChange = Math.max(sizeChange, Math.abs(Math.log(sizeAt[k] / sizeTO)));
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
          if (inFlight.size && !inFlight.has(k)) continue; // skip frames off the flight curve
          const P = V[k].P;
          const x = pts.reduce((s, j) => s + P[j].x, 0) / pts.length;
          const y = pts.reduce((s, j) => s + P[j].y, 0) / pts.length;
          const above = barLine.yAt(x) - y;
          if (above > best) { best = above; bestK = k; }
        }
        const cm = best * (barScale || cmPerPx) - part.allowance * scaleAllow;
        if (!worst || cm < worst.cm) worst = { cm, frame: V[bestK].i };
      }
      return { key: part.key, label: part.label, cm: worst.cm, frame: worst.frame };
    });
    const lowest = parts.reduce((a, b) => (b.cm < a.cm ? b : a));
    const verdict = lowest.cm >= 2 ? 'cleared' : lowest.cm >= -3 ? 'close' : 'knocked';
    clearance = { parts, lowest: lowest.key, lowestLabel: lowest.label, lowestCm: lowest.cm, verdict, method: barLine.marked ? 'marked bar' : 'bar height' };
  }

  const contactTime = contactFound ? V[takeoff].t - V[plant].t : NaN;
  const values = { approachSpeed, curveLean, plantKnee, plantHip, takeoffAngle, contactTime, rise, torsoAtPeak };
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
      takeoffLeg: takeoffSide,
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
      contactTime,
      steps: steps.slice(-4).map((s2, j, arr) => ({
        foot: s2.foot,
        time: V[s2.start].t,
        // length and time from the previous contact (cm and s)
        length: j > 0 ? dist(s2.spot, arr[j - 1].spot) * cmPerPx : null,
        duration: j > 0 ? V[s2.start].t - V[arr[j - 1].start].t : null,
      })),
      stepsSeen: steps.length,
      clearance,
      barLine: barLine && barLine.marked ? { x1: barLine.x1, y1: barLine.y1, x2: barLine.x2, y2: barLine.y2, ground: barLine.ground } : null,
      groundY,
      groundPx: { x: plantFoot.x, y: groundY },
      peakHipPx: { x: hip[peak].x, y: hip[peak].y },
      bodyPx,
      contactFound,
      peakFromFit,
      flightFrames: inFlight.size,
      flightFitGood: fitGood,
      timeToPeak: q && q.c > 0 ? -q.b / (2 * q.c) : null,
      flightCmPerPx,
      barScale,
      verticalFrom,
      depthChange: Math.exp(sizeChange) - 1,
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
