import { test } from 'node:test';
import assert from 'node:assert/strict';
import { detectBarCandidates, chooseBar, lineYAt, medianBackground } from '../bar.js';

const W = 480, H = 270;
// Seeded noise so the test is repeatable.
let seed = 7;
const rand = () => ((seed = (seed * 1103515245 + 12345) % 2147483648) / 2147483648);

// Scene: gray sky, a dark landing mat (big step edge at y=180), a thin bright
// bar from x=90 to x=400 sloping 2 px, a standard pole, and noise.
function scene({ jumperX = null } = {}) {
  const g = new Float32Array(W * H);
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    let v = y < 180 ? 150 : 60;
    const by = 120 + (2 * (x - 90)) / 310;
    if (x >= 90 && x <= 400 && Math.abs(y - by) <= 1.2) v = 235;
    if ((x === 88 || x === 89) && y > 100 && y < 180) v = 90; // standard
    if (jumperX !== null && Math.abs(x - jumperX) < 25 && y > 70 && y < 175) v = 40; // person blob
    g[y * W + x] = v + (rand() - 0.5) * 12;
  }
  return g;
}

test('median background removes a moving person', () => {
  const frames = [50, 150, 250, 350, 430].map((jx) => scene({ jumperX: jx }));
  const bg = medianBackground(frames);
  // A pixel the person covered once is back to sky.
  assert.ok(Math.abs(bg[100 * W + 150] - 150) < 12);
});

test('finds the thin bar, not the mat edge', () => {
  const frames = [50, 150, 250, 350, 430].map((jx) => scene({ jumperX: jx }));
  const cands = detectBarCandidates(frames, W, H);
  assert.ok(cands.length >= 1);
  const top = cands[0];
  assert.ok(Math.abs(lineYAt(top, 0.5) * H - (120 + (2 * (240 - 90)) / 310)) < 2, `y ${lineYAt(top, 0.5) * H}`);
  assert.ok(top.a.x * W < 110 && top.b.x * W > 380, `ends ${top.a.x * W} ${top.b.x * W}`);
  // Nothing at the mat edge (y=180) should outrank it.
  for (const c of cands) if (Math.abs(lineYAt(c, 0.5) * H - 180) < 3) assert.ok(c.score < top.score);
});

test('chooseBar keeps the line that agrees with the athlete scale', () => {
  const bar = { a: { x: 0.2, y: 120 / H }, b: { x: 0.85, y: 120 / H }, score: 100 };
  const fence = { a: { x: 0.1, y: 30 / H }, b: { x: 0.9, y: 30 / H }, score: 300 }; // stronger but too high
  // Floor at y=220 px under the takeoff spot. Bar 100 px above it = 150 cm -> 1.5 cm/px.
  const info = { groundPx: { x: 200, y: 220 }, peakHipPx: { x: 250, y: 110 }, cmPerPxFromHeight: 1.45 };
  const pick = chooseBar([fence, bar], info, 150, W, H);
  assert.equal(pick.a.x, 0.2);
  // With a bar height that matches neither line, nothing is chosen.
  assert.equal(chooseBar([fence, bar], info, 600, W, H), null);
});
