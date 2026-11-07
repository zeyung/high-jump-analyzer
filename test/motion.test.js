import { test } from 'node:test';
import assert from 'node:assert/strict';
import { estimateShift, cameraPath } from '../motion.js';

const W = 120, H = 200;
let seed = 3;
const rand = () => ((seed = (seed * 1103515245 + 12345) % 2147483648) / 2147483648);
// A big textured "world"; each frame is a window into it (a panning camera).
const WW = 400, WH = 300;
const world = new Float32Array(WW * WH);
for (let i = 0; i < world.length; i++) world[i] = rand() * 255;
// Blur a little so it looks like a scene, not pure noise.
for (let pass = 0; pass < 2; pass++) for (let y = 1; y < WH - 1; y++) for (let x = 1; x < WW - 1; x++) {
  const i = y * WW + x;
  world[i] = (world[i] * 4 + world[i - 1] + world[i + 1] + world[i - WW] + world[i + WW]) / 8;
}
function view(ox, oy, blob = null) {
  const g = new Float32Array(W * H);
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    let v = world[(y + oy) * WW + (x + ox)];
    if (blob && Math.abs(x - blob.x) < 10 && Math.abs(y - blob.y) < 25) v = 20; // a moving person
    g[y * W + x] = v;
  }
  return g;
}

test('measures a pan between two frames', () => {
  // Camera moves right by 7 and down by 3 -> content moves left 7, up 3.
  const s = estimateShift(view(50, 40), view(57, 43), W, H);
  assert.ok(Math.abs(s.dx + 7) < 0.6, `dx ${s.dx}`);
  assert.ok(Math.abs(s.dy + 3) < 0.6, `dy ${s.dy}`);
});

test('follows a camera that pans across many frames, ignoring the jumper', () => {
  const grays = [], boxes = [];
  for (let k = 0; k < 20; k++) {
    const ox = 20 + k * 6, oy = 50 + Math.round(Math.sin(k / 3) * 3);
    const blob = { x: 60 + (k % 5), y: 120 };
    grays.push(view(ox, oy, blob));
    boxes.push({ x: blob.x - 10, y: blob.y - 25, w: 20, h: 50 });
  }
  const cum = cameraPath(grays, W, H, boxes);
  // Content shifts by -(camera move) since frame 0.
  assert.ok(Math.abs(cum[19].x - -(6 * 19)) < 3, `x ${cum[19].x}`);
  const dy = -(Math.round(Math.sin(19 / 3) * 3) - 0);
  assert.ok(Math.abs(cum[19].y - dy) < 3, `y ${cum[19].y}`);
});
