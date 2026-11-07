import { test } from 'node:test';
import assert from 'node:assert/strict';
import { pickJumper, fixSwaps, fillGaps, smoothFrames, applyCorrections, cleanTracking } from '../tracking.js';
import { syntheticJump } from './synthetic.js';
import { analyzeJump } from '../analysis.js';

const opts = { width: 1000, height: 600, heightCm: 180, gender: 'male' };
const shift = (lms, dx, dy, scale = 1) => lms.map((p) => ({ ...p, x: p.x * scale + dx, y: p.y * scale + dy }));
const swapLegs = (lms) => {
  const c = lms.map((p) => ({ ...p }));
  for (const [l, r] of [[23, 24], [25, 26], [27, 28], [29, 30], [31, 32]]) [c[l], c[r]] = [c[r], c[l]];
  return c;
};

test('pickJumper follows the jumper, not a bystander', () => {
  const jump = syntheticJump();
  // A smaller bystander standing still at the left edge, listed first.
  const bystander = shift(jump[0].lms, -0.05, 0.1, 0.6);
  const ppf = jump.map((f) => ({ t: f.t, poses: [bystander, f.lms] }));
  const picked = pickJumper(ppf);
  assert.ok(picked.every((f, i) => f.lms[23].x === jump[i].lms[23].x));
});

test('pickJumper drops a frame that jumps to someone far away', () => {
  const jump = syntheticJump();
  const ppf = jump.map((f) => ({ t: f.t, poses: [f.lms] }));
  ppf[20] = { t: ppf[20].t, poses: [shift(jump[20].lms, 0.6, 0)] };
  const picked = pickJumper(ppf);
  assert.equal(picked[20].lms, null);
  assert.ok(picked[21].lms);
});

test('fixSwaps undoes a left/right leg mix-up', () => {
  const jump = syntheticJump();
  // Make the legs distinct so a swap is visible: move the right ankle back.
  const frames = jump.map((f, i) => {
    const l = f.lms.map((p) => ({ ...p }));
    l[28] = { ...l[28], x: l[28].x - 0.05 }; l[26] = { ...l[26], x: l[26].x - 0.025 };
    return { t: f.t, lms: l };
  });
  const broken = frames.map((f, i) => (i === 15 ? { ...f, lms: swapLegs(f.lms) } : f));
  const fixed = fixSwaps(broken);
  assert.equal(fixed[15].swapped, true);
  assert.deepEqual(fixed[15].lms[28], frames[15].lms[28]);
  assert.equal(fixed.filter((f) => f.swapped).length, 1);
});

test('fillGaps fills short gaps and leaves long ones', () => {
  const jump = syntheticJump();
  const holes = jump.map((f, i) => ((i >= 10 && i <= 12) || (i >= 30 && i <= 39) ? { t: f.t, lms: null } : f));
  const filled = fillGaps(holes, 5);
  for (const i of [10, 11, 12]) {
    assert.equal(filled[i].filled, true);
    assert.ok(Math.abs(filled[i].lms[23].x - jump[i].lms[23].x) < 1e-6); // straight-line run, so exact
  }
  assert.equal(filled[35].lms, null);
});

test('smoothFrames reduces jitter', () => {
  const jump = syntheticJump();
  const noisy = jump.map((f, i) => ({ t: f.t, lms: f.lms.map((p) => ({ ...p, y: p.y + (i % 2 ? 0.01 : -0.01) })) }));
  const err = (fr) => fr.slice(2, 30).reduce((s, f, i) => s + Math.abs(f.lms[23].y - jump[i + 2].lms[23].y), 0);
  assert.ok(err(smoothFrames(noisy, 1)) < err(noisy) * 0.6);
});

test('applyCorrections overrides a dragged point', () => {
  const jump = syntheticJump();
  const out = applyCorrections(jump, { 5: { 25: { x: 0.5, y: 0.5 } } });
  assert.deepEqual(out[5].lms[25], { x: 0.5, y: 0.5, visibility: 1 });
  assert.equal(out[5].corrected, true);
  assert.notEqual(jump[5].lms[25].x, 0.5); // input untouched
});

test('the cleaned pipeline still analyzes a jump with noise, a bystander and gaps', () => {
  const jump = syntheticJump({ lean: 15, launch: 48, risePx: 100 });
  const bystander = shift(jump[0].lms, -0.05, 0.1, 0.6);
  const ppf = jump.map((f, i) => ({
    t: f.t,
    poses: i === 25 ? [] : [bystander, f.lms.map((p, j) => ({ ...p, x: p.x + ((i * 7 + j) % 3 - 1) * 0.001 }))],
  }));
  const a = analyzeJump(cleanTracking(ppf), opts);
  assert.ok(Math.abs(a.metrics.curveLean.value - 15) < 2, `lean ${a.metrics.curveLean.value}`);
  assert.ok(Math.abs(a.metrics.rise.value - 60) < 6, `rise ${a.metrics.rise.value}`);
});
