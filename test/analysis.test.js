import { test } from 'node:test';
import assert from 'node:assert/strict';
import { analyzeJump, angleAt, fitQuadratic, scoreRange, targetsFor, phaseAt } from '../analysis.js';

import { syntheticJump, W, H, FPS, CM_PER_PX } from './synthetic.js';

const opts = { width: W, height: H, heightCm: 180, gender: 'male' };

test('angleAt measures the angle at the middle point', () => {
  assert.equal(Math.round(angleAt({ x: 0, y: 0 }, { x: 1, y: 0 }, { x: 1, y: 1 })), 90);
  assert.equal(Math.round(angleAt({ x: 0, y: 0 }, { x: 1, y: 0 }, { x: 2, y: 0 })), 180);
});

test('fitQuadratic recovers parabola coefficients', () => {
  const ts = [0, 0.1, 0.2, 0.3, 0.4];
  const q = fitQuadratic(ts, ts.map((t) => 5 - 3 * t + 2 * t * t));
  assert.ok(Math.abs(q.a - 5) < 1e-9 && Math.abs(q.b + 3) < 1e-9 && Math.abs(q.c - 2) < 1e-9);
});

test('scoreRange is 100 inside the range and falls off linearly', () => {
  assert.equal(scoreRange(15, [10, 20], 10), 100);
  assert.equal(scoreRange(5, [10, 20], 10), 50);
  assert.equal(scoreRange(40, [10, 20], 10), 0);
  assert.equal(scoreRange(NaN, [10, 20], 10), null);
});

test('targets depend on gender and height', () => {
  const m = targetsFor('male', 180), f = targetsFor('female', 168);
  assert.notDeepEqual(m.takeoffAngle.range, f.takeoffAngle.range);
  assert.ok(targetsFor('male', 195).rise.range[0] > targetsFor('male', 170).rise.range[0]);
  assert.ok(targetsFor('male', 195).approachSpeed.range[1] > targetsFor('male', 170).approachSpeed.range[1]);
});

test('finds the key frames of a synthetic jump', () => {
  const a = analyzeJump(syntheticJump(), opts);
  assert.equal(a.keyFrames.takeoff, 39);
  assert.equal(a.keyFrames.plant, 34);
  assert.ok(a.keyFrames.peak >= 49 && a.keyFrames.peak <= 51, `peak ${a.keyFrames.peak}`);
  assert.ok(a.keyFrames.landing > a.keyFrames.peak);
  assert.equal(phaseAt(10, a.keyFrames), 'Approach');
  assert.equal(phaseAt(a.keyFrames.peak, a.keyFrames), 'Flight: rising');
});

test('measures lean, launch angle, rise, speed and landing', () => {
  const a = analyzeJump(syntheticJump({ lean: 15, launch: 48, risePx: 100 }), opts);
  const m = a.metrics;
  assert.ok(Math.abs(m.curveLean.value - 15) < 1, `lean ${m.curveLean.value}`);
  assert.ok(Math.abs(m.takeoffAngle.value - 48) < 1.5, `launch ${m.takeoffAngle.value}`);
  assert.ok(Math.abs(m.rise.value - 60) < 3, `rise ${m.rise.value}`); // 100 px * 0.6
  assert.ok(Math.abs(m.approachSpeed.value - 1.8) < 0.1, `speed ${m.approachSpeed.value}`); // 300 px/s
  assert.ok(Math.abs(a.info.cmPerPx - CM_PER_PX) < 0.02, `scale ${a.info.cmPerPx}`);
  assert.ok(m.torsoAtPeak.value < 15);
  assert.equal(m.plantKnee.value > 175, true);
  assert.equal(m.landing.value, 'Yes');
});

test('rating is 0-100 and better technique scores higher', () => {
  const good = analyzeJump(syntheticJump({ lean: 15, launch: 47, risePx: 140, approachPxPerFrame: 37 }), opts);
  const poor = analyzeJump(syntheticJump({ lean: 2, launch: 25, risePx: 40, approachPxPerFrame: 10 }), opts);
  for (const a of [good, poor]) assert.ok(a.overall >= 0 && a.overall <= 100);
  assert.ok(good.overall > poor.overall + 20, `good ${good.overall} poor ${poor.overall}`);
  assert.ok(good.overall >= 90, `good ${good.overall}`);
  assert.ok(poor.metrics.curveLean.tip.length > 0);
});

test('rejects clips with no person or bad inputs', () => {
  const empty = Array.from({ length: 60 }, (_, i) => ({ t: i / FPS, lms: null }));
  assert.throws(() => analyzeJump(empty, opts), /only 0 frames/);
  assert.throws(() => analyzeJump(syntheticJump(), { ...opts, heightCm: 40 }), /Height/);
  assert.throws(() => analyzeJump(syntheticJump(), { ...opts, gender: 'x' }), /Gender/);
});

test('bar height compares peak hip height with the bar', () => {
  // Synthetic hip height at takeoff is about 82 cm and the rise 60 cm, so the hips peak near 142 cm.
  const frames = syntheticJump({ lean: 15, launch: 48, risePx: 100 });
  const none = analyzeJump(frames, opts);
  assert.equal(none.metrics.barMargin, undefined);
  assert.equal(none.info.barCm, null);

  const low = analyzeJump(frames, { ...opts, barCm: 130 });
  assert.ok(Math.abs(low.metrics.barMargin.value - (low.info.peakHipHeight - 130)) < 1e-9);
  assert.ok(Math.abs(low.info.peakHipHeight - 142) < 4, `peak ${low.info.peakHipHeight}`);
  assert.equal(low.metrics.barMargin.score, 100);
  assert.equal(low.metrics.barMargin.tip, '');
  assert.ok(Math.abs(low.info.barPercentOfHeight - (100 * 130) / 180) < 1e-9);

  const high = analyzeJump(frames, { ...opts, barCm: 170 });
  assert.ok(high.metrics.barMargin.value < -20);
  assert.ok(high.metrics.barMargin.score < 30);
  assert.ok(high.metrics.barMargin.tip.length > 0);
  assert.ok(high.overall < low.overall);

  assert.throws(() => analyzeJump(frames, { ...opts, barCm: 400 }), /Bar height/);
});
