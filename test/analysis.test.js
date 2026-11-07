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
  // Knee angle is read from the takeoff (left) leg at the plant frame.
  const P = syntheticJump({ lean: 15, launch: 48, risePx: 100 })[a.keyFrames.plant].lms.map((p) => ({ x: p.x * W, y: p.y * H }));
  assert.ok(Math.abs(m.plantKnee.value - angleAt(P[23], P[25], P[27])) < 1e-6);
  assert.equal(m.landing.value, 'Yes');
});

test('rating is 0-100 and better technique scores higher', () => {
  const good = analyzeJump(syntheticJump({ lean: 15, launch: 47, risePx: 140, approachPxPerFrame: 37, stance: 4 }), opts);
  const poor = analyzeJump(syntheticJump({ lean: 2, launch: 25, risePx: 40, approachPxPerFrame: 10 }), opts);
  for (const a of [good, poor]) assert.ok(a.overall >= 0 && a.overall <= 100);
  assert.ok(good.overall > poor.overall + 20, `good ${good.overall} poor ${poor.overall}`);
  assert.ok(good.overall >= 70, `good ${good.overall}`);
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

test('clearance checks each body part against the bar', () => {
  const frames = syntheticJump({ lean: 15, launch: 48, risePx: 100 });
  const none = analyzeJump(frames, opts);
  assert.equal(none.info.clearance, null);
  assert.ok(Math.abs(none.info.verticalJump - 60) < 3);

  const low = analyzeJump(frames, { ...opts, barCm: 100 });
  const cl = low.info.clearance;
  assert.deepEqual(cl.parts.map((p) => p.key), ['head', 'shoulders', 'hips', 'knees', 'feet']);
  assert.equal(cl.verdict, 'cleared');
  assert.equal(cl.method, 'bar height');
  assert.equal(low.metrics.clearance.score, 100);

  const high = analyzeJump(frames, { ...opts, barCm: 160 });
  assert.equal(high.info.clearance.verdict, 'knocked');
  assert.ok(high.metrics.clearance.score < 50);
  assert.ok(high.metrics.clearance.tip.includes(high.info.clearance.lowestLabel.toLowerCase().replace(/s$/, '')));
});

test('a marked bar sets the bar line and the scale', () => {
  const frames = syntheticJump({ lean: 15, launch: 48, risePx: 100 });
  // Bar at 120 cm = 200 px above the ground line at y=560 (0.6 cm/px), drawn level.
  const y = (560 - 200) / 600;
  const bar = { a: { x: 0.2, y }, b: { x: 0.9, y }, ground: { x: 0.9, y: 560 / 600 } };
  const a = analyzeJump(frames, { ...opts, barCm: 120, bar });
  assert.match(a.info.scaleSource, /bar over the bar/);
  assert.equal(a.info.clearance.method, 'marked bar');
  assert.ok(Math.abs(a.info.barScale - 0.6) < 1e-9);
  assert.equal(a.info.verticalFrom, 'bar');
  // Peak hip from the bar agrees with the true rise (100 px * 0.6 = 60 cm).
  assert.ok(Math.abs(a.info.verticalJump - 60) < 4, `vertical ${a.info.verticalJump}`);
  // Same bar via bar height alone should give nearly the same hip clearance.
  const b = analyzeJump(frames, { ...opts, barCm: 120 });
  const hip = (r) => r.info.clearance.parts.find((p) => p.key === 'hips').cm;
  assert.ok(Math.abs(hip(a) - hip(b)) < 4, `${hip(a)} vs ${hip(b)}`);
});

test('a known takeoff foot is searched first', () => {
  const frames = syntheticJump({ lean: 15, launch: 48, risePx: 100 });
  const a = analyzeJump(frames, { ...opts, takeoffFoot: 'left' });
  assert.equal(a.info.takeoffLeg, 'left');
  assert.equal(a.keyFrames.takeoff, 39);
  // Asking for the right foot finds the right foot's last contact instead (earlier).
  const b = analyzeJump(frames, { ...opts, takeoffFoot: 'right' });
  assert.equal(b.info.takeoffLeg, 'right');
  assert.ok(b.keyFrames.takeoff < 39);
});

test('flight numbers survive frames that latch onto someone else', () => {
  const clean = syntheticJump({ lean: 15, launch: 48, risePx: 100 });
  // Frames 47-52 (near the top of the flight) show a person standing lower down.
  const bystander = clean[20].lms.map((p) => ({ ...p, x: p.x + 0.05 }));
  const dirty = clean.map((f, i) => (i >= 47 && i <= 52 ? { ...f, lms: bystander } : f));
  const a = analyzeJump(dirty, opts);
  assert.ok(Math.abs(a.metrics.rise.value - 60) < 4, `rise ${a.metrics.rise.value}`);
  assert.ok(Math.abs(a.metrics.takeoffAngle.value - 48) < 3, `launch ${a.metrics.takeoffAngle.value}`);
  assert.ok(a.info.flightFrames >= 10);
  assert.ok(a.keyFrames.peak >= 48 && a.keyFrames.peak <= 52, `peak ${a.keyFrames.peak}`);
});

test('jumpNear keeps the peak near the jump', async () => {
  const { syntheticJump, W, H } = await import('./synthetic.js');
  const frames = syntheticJump();
  const opts = { width: W, height: H, heightCm: 180, gender: 'male' };
  const free = analyzeJump(frames, opts);
  const hinted = analyzeJump(frames, { ...opts, jumpNear: free.keyFrames.peak });
  assert.equal(hinted.keyFrames.peak, free.keyFrames.peak);
});
