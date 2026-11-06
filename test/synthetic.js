// Synthetic side-on high jump used by the tests and the in-browser self-test.
// Synthetic side-on jump. 300 px tall jumper, 180 cm -> 0.6 cm/px.
export const W = 1000, H = 600, FPS = 30, BODY = 300, GROUND = 560;
export const CM_PER_PX = 180 / BODY;
const G_PX = 981 / CM_PER_PX;
const rad = (d) => (d * Math.PI) / 180;

function pose({ ankleX, ankleY, hipX, hipY, phi }) {
  // phi: torso angle from vertical (0 = upright, 90 = flat). Legs point opposite.
  const lms = Array.from({ length: 33 }, () => ({ x: 0, y: 0, visibility: 1 }));
  const put = (i, x, y) => { lms[i] = { x: x / W, y: y / H, visibility: 1 }; };
  const s = Math.sin(rad(phi)), c = Math.cos(rad(phi));
  const sh = { x: hipX + 0.23 * BODY * s, y: hipY - 0.23 * BODY * c };
  const nose = { x: hipX + 0.42 * BODY * s, y: hipY - 0.42 * BODY * c };
  const ank = ankleX === undefined ? { x: hipX - 0.47 * BODY * s, y: hipY + 0.47 * BODY * c } : { x: ankleX, y: ankleY };
  const knee = { x: (hipX + ank.x) / 2, y: (hipY + ank.y) / 2 };
  put(0, nose.x, nose.y);
  for (const i of [11, 12]) put(i, sh.x, sh.y);
  for (const i of [23, 24]) put(i, hipX, hipY);
  for (const i of [25, 26]) put(i, knee.x, knee.y);
  put(27, ank.x, ank.y); put(28, ank.x, ank.y - 2); // left foot lowest
  // Arms hang from the shoulders toward the hips; feet sit at the ankles.
  const elbow = { x: (sh.x * 2 + hipX) / 3, y: (sh.y * 2 + hipY) / 3 };
  for (const i of [13, 14]) put(i, elbow.x, elbow.y);
  for (const i of [15, 16]) put(i, (elbow.x + hipX) / 2 + 8, (elbow.y + hipY) / 2);
  for (const i of [29, 31]) put(i, ank.x + (i === 31 ? 12 : -6), ank.y);
  for (const i of [30, 32]) put(i, ank.x + (i === 32 ? 12 : -6), ank.y - 2);
  return lms;
}

export function syntheticJump({ lean = 15, launch = 48, risePx = 100, approachPxPerFrame = 10 } = {}) {
  const frames = [];
  const T0 = 40; // frames 0..39 on the ground, 39 = toe-off
  for (let i = 0; i < T0; i++) {
    const ax = 100 + approachPxPerFrame * i;
    const hipX = ax + 0.47 * BODY * Math.sin(rad(lean));
    const hipY = GROUND - 0.47 * BODY * Math.cos(rad(lean));
    frames.push({ t: i / FPS, lms: pose({ ankleX: ax, ankleY: GROUND, hipX, hipY, phi: lean }) });
  }
  const last = frames[T0 - 1].lms;
  const hx0 = last[23].x * W, hy0 = last[23].y * H;
  const vy = Math.sqrt(2 * G_PX * risePx);
  const vx = vy / Math.tan(rad(launch));
  const tPeak = vy / G_PX;
  for (let i = T0; i < T0 + 30; i++) {
    let t = (i - (T0 - 1)) / FPS;
    const flying = t <= 2 * tPeak;
    if (!flying) t = 2 * tPeak;
    const hipX = hx0 + vx * t;
    const hipY = hy0 - (vy * t - 0.5 * G_PX * t * t);
    const phi = Math.min(130, (90 * t) / tPeak);
    frames.push({ t: i / FPS, lms: pose({ hipX, hipY, phi }) });
  }
  return frames;
}

