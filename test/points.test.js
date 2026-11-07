import test from 'node:test';
import assert from 'node:assert/strict';
import { followPoint, followAnchored, cameraFromPoints, appearanceOf, appearanceSim } from '../points.js';

// A textured scene panned by (vx, vy) pixels a frame.
const W = 160, H = 120;
function scene(x, y) { return 128 + 60 * Math.sin(x * 0.31) * Math.cos(y * 0.23) + 40 * Math.sin((x + 2 * y) * 0.11); }
function frames(n, vx, vy) {
  return Array.from({ length: n }, (_, k) => {
    const g = new Uint8Array(W * H);
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) g[y * W + x] = Math.max(0, Math.min(255, Math.round(scene(x + k * vx, y + k * vy))));
    return g;
  });
}

test('followPoint stays on a feature while the camera pans', () => {
  const F = frames(20, 1.5, -0.5);
  const tr = followPoint(F, W, H, 0, 19, { x: 80, y: 60 });
  const p = tr.get(19);
  assert.ok(p.ok);
  assert.ok(Math.abs(p.x - (80 - 19 * 1.5)) < 0.8, `x ${p.x}`);
  assert.ok(Math.abs(p.y - (60 + 19 * 0.5)) < 0.8, `y ${p.y}`);
});

test('followAnchored passes through every anchor and tracks backward', () => {
  const F = frames(30, 1, 0);
  const out = followAnchored(F, W, H, [{ index: 10, x: 90, y: 50 }, { index: 25, x: 75.5, y: 50 }]);
  assert.deepEqual([out[10].x, out[25].x], [90, 75.5]);
  assert.ok(Math.abs(out[0].x - 100) < 0.8, `back ${out[0].x}`);
  assert.ok(Math.abs(out[18].x - 82) < 1, `between ${out[18].x}`);
});

test('cameraFromPoints uses still points and ignores one that moves', () => {
  const n = 5;
  const cum = Array.from({ length: n }, () => ({ x: 0, y: 0 }));
  const still = (x0) => Array.from({ length: n }, (_, k) => ({ x: x0 + 0.01 * k, y: 0.5, ok: true }));
  const falling = Array.from({ length: n }, (_, k) => ({ x: 0.9 + 0.01 * k, y: 0.5 + 0.05 * k, ok: true }));
  const out = cameraFromPoints(cum, [still(0.2), still(0.4), falling], [{ x: 0.2, y: 0.5 }, { x: 0.4, y: 0.5 }, { x: 0.9, y: 0.5 }], 0);
  assert.ok(Math.abs(out[4].x - 0.04) < 1e-9 && Math.abs(out[4].y) < 1e-9);
});

test('appearance tells two outfits apart', () => {
  const w = 40, h = 80;
  const img = (top, legs) => {
    const d = new Uint8ClampedArray(w * h * 4);
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) { const c = y < 40 ? top : legs; d.set([...c, 255], (y * w + x) * 4); }
    return d;
  };
  const lms = Array.from({ length: 33 }, () => ({ x: 0.5, y: 0.5 }));
  Object.assign(lms, { 11: { x: 0.4, y: 0.1 }, 12: { x: 0.6, y: 0.1 }, 13: { x: 0.3, y: 0.3 }, 14: { x: 0.7, y: 0.3 }, 23: { x: 0.45, y: 0.45 }, 24: { x: 0.55, y: 0.45 }, 25: { x: 0.45, y: 0.7 }, 26: { x: 0.55, y: 0.7 }, 27: { x: 0.45, y: 0.95 }, 28: { x: 0.55, y: 0.95 } });
  const red = appearanceOf(img([220, 30, 30], [20, 20, 20]), w, h, lms);
  const red2 = appearanceOf(img([200, 40, 35], [25, 25, 25]), w, h, lms);
  const blue = appearanceOf(img([30, 60, 220], [230, 230, 230]), w, h, lms);
  assert.ok(appearanceSim(red, red2) > 0.9);
  assert.ok(appearanceSim(red, blue) < 0.4);
});
