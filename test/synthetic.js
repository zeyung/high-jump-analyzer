// Synthetic side-on high jump used by the tests and the in-browser self-test.
// A 300 px tall jumper (180 cm -> 0.6 cm/px) runs with alternating steps:
// the stance foot stays still on the ground while the other foot swings.
// Body proportions match the constants in analysis.js (torso 0.29, leg 0.49
// of standing height, ankle 0.04 above the floor), so the scale is exact.

export const W = 1000, H = 600, FPS = 30, BODY = 300, GROUND = 560;
export const CM_PER_PX = 180 / BODY;
const G_PX = 981 / CM_PER_PX;
const rad = (d) => (d * Math.PI) / 180;
const LEG = 0.49 * BODY, TORSO = 0.29 * BODY, ANKLE_Y = GROUND - 0.04 * BODY;
const HIP_MAX = 0.43 * BODY;

// Knee for a two-segment leg (thigh = shank = LEG/2), bent forward (+x).
function knee(hip, ank) {
  const dx = ank.x - hip.x, dy = ank.y - hip.y, d = Math.hypot(dx, dy) || 1;
  const half = d / 2, seg = LEG / 2;
  const h = Math.sqrt(Math.max(0, seg * seg - half * half));
  let nx = -dy / d, ny = dx / d;
  if (nx < 0) { nx = -nx; ny = -ny; }
  return { x: (hip.x + ank.x) / 2 + nx * h, y: (hip.y + ank.y) / 2 + ny * h };
}

// Keep a foot within leg length of the hip.
function reach(hip, ank) {
  const dx = ank.x - hip.x, dy = ank.y - hip.y, d = Math.hypot(dx, dy);
  if (d <= LEG) return ank;
  return { x: hip.x + (dx * LEG) / d, y: hip.y + (dy * LEG) / d };
}

function build({ hip, tilt, ankL, ankR }) {
  const lms = Array.from({ length: 33 }, () => ({ x: 0, y: 0, visibility: 1 }));
  const put = (i, p) => { lms[i] = { x: p.x / W, y: p.y / H, visibility: 1 }; };
  const s = Math.sin(rad(tilt)), c = Math.cos(rad(tilt));
  const sh = { x: hip.x + TORSO * s, y: hip.y - TORSO * c };
  const nose = { x: hip.x + 0.4 * BODY * s, y: hip.y - 0.4 * BODY * c };
  const elbow = { x: (sh.x * 2 + hip.x) / 3, y: (sh.y * 2 + hip.y) / 3 };
  const wrist = { x: (elbow.x + hip.x) / 2 + 8, y: (elbow.y + hip.y) / 2 };
  put(0, nose);
  for (const i of [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) put(i, nose);
  put(11, sh); put(12, { x: sh.x + 1, y: sh.y });
  put(13, elbow); put(14, elbow); put(15, wrist); put(16, wrist);
  for (const i of [17, 19, 21]) put(i, wrist);
  for (const i of [18, 20, 22]) put(i, wrist);
  put(23, hip); put(24, { x: hip.x + 1, y: hip.y });
  put(25, knee(hip, ankL)); put(26, knee(hip, ankR));
  put(27, ankL); put(28, ankR);
  put(29, { x: ankL.x - 6, y: ankL.y }); put(31, { x: ankL.x + 12, y: ankL.y });
  put(30, { x: ankR.x - 6, y: ankR.y }); put(32, { x: ankR.x + 12, y: ankR.y });
  return lms;
}

/**
 * Frames 0..39 run-up, frame 39 is toe-off from the left foot, whose last
 * stance is frames [40 - stance, 40). Then a parabolic flight and landing.
 */
export function syntheticJump({ lean = 15, launch = 48, risePx = 100, approachPxPerFrame = 10, stance = 6 } = {}) {
  const T0 = 40;
  const v = approachPxPerFrame;
  const hipX = (i) => 100 + v * i;
  // Stance blocks end at T0; block 0 is the left foot's final stance.
  const blockOf = (i) => Math.floor((T0 - 1 - i) / stance); // 0 for the last block
  const footOf = (b) => (((b % 2) + 2) % 2 === 0 ? 'L' : 'R');
  // Where each block's foot lands: under the lean line at mid-stance.
  const footX = (b) => {
    const midI = T0 - stance * b - (stance + 1) / 2;
    const hx = hipX(midI);
    const shX = hx + TORSO * Math.sin(rad(lean));
    const shY = ANKLE_Y - HIP_MAX - TORSO * Math.cos(rad(lean));
    return shX - (ANKLE_Y - shY) * Math.tan(rad(lean));
  };
  const frames = [];
  for (let i = 0; i < T0; i++) {
    const b = blockOf(i);
    const support = footOf(b);
    const sx = footX(b);
    const hx = hipX(i);
    const dx = sx - hx;
    const hipH = Math.min(HIP_MAX, Math.sqrt(Math.max(0, LEG * LEG - dx * dx)));
    const hip = { x: hx, y: ANKLE_Y - hipH };
    // Swing foot travels from its last stance spot to its next one.
    const blockStart = T0 - stance * (b + 1);
    const u = (i - blockStart + 1) / (stance + 1); // lands on the next block's first frame
    const from = footX(b + 1), to = footX(b - 1);
    const swing = reach(hip, { x: from + (to - from) * u, y: ANKLE_Y - 0.1 * BODY * Math.sin(Math.PI * u) });
    const stanceFoot = { x: sx, y: ANKLE_Y };
    const ankL = support === 'L' ? stanceFoot : swing;
    const ankR = support === 'R' ? stanceFoot : swing;
    frames.push({ t: i / FPS, lms: build({ hip, tilt: lean, ankL, ankR }) });
  }
  const last = frames[T0 - 1].lms;
  const hx0 = last[23].x * W, hy0 = last[23].y * H;
  const vy = Math.sqrt(2 * G_PX * risePx);
  const vx = vy / Math.tan(rad(launch));
  const tPeak = vy / G_PX;
  for (let i = T0; i < T0 + 30; i++) {
    let t = (i - (T0 - 1)) / FPS;
    if (t > 2 * tPeak) t = 2 * tPeak;
    const hip = { x: hx0 + vx * t, y: hy0 - (vy * t - 0.5 * G_PX * t * t) };
    const phi = Math.min(130, (90 * t) / tPeak);
    const leg = { x: -LEG * Math.sin(rad(phi)), y: LEG * Math.cos(rad(phi)) };
    const ankL = { x: hip.x + leg.x, y: hip.y + leg.y };
    const ankR = { x: hip.x - LEG * Math.sin(rad(phi + 3)), y: hip.y + LEG * Math.cos(rad(phi + 3)) };
    frames.push({ t: i / FPS, lms: build({ hip, tilt: phi, ankL, ankR }) });
  }
  return frames;
}
