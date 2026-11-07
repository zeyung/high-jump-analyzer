import { test } from 'node:test';
import assert from 'node:assert/strict';
import { coachJump, stepFeedback } from '../coaching.js';
import { analyzeJump } from '../analysis.js';
import { syntheticJump } from './synthetic.js';

const opts = { width: 1000, height: 600, heightCm: 180, gender: 'male' };

test('step rhythm: long penultimate and quick last step is praised', () => {
  const good = stepFeedback([{ length: null }, { length: 200, duration: 0.24 }, { length: 220, duration: 0.25 }, { length: 180, duration: 0.2 }]);
  assert.equal(good.fix.length, 0);
  assert.equal(good.good.length, 2);
  const bad = stepFeedback([{ length: null }, { length: 200, duration: 0.24 }, { length: 180, duration: 0.2 }, { length: 230, duration: 0.26 }]);
  assert.equal(bad.fix.length, 2);
});

test('coaching ranks fixes and picks drills for the weakest phase', () => {
  const poor = analyzeJump(syntheticJump({ lean: 2, launch: 25, risePx: 40 }), { ...opts, barCm: 150 });
  const c = coachJump(poor);
  assert.ok(c.fixes.length >= 2);
  for (let i = 1; i < c.fixes.length; i++) assert.ok(c.fixes[i - 1].score <= c.fixes[i].score);
  assert.ok(c.weakestPhase);
  assert.ok(c.drills.length >= 1);
  assert.ok(c.fixes.some((f) => f.key === 'curveLean'));
  assert.ok(c.fundamentals.length >= 5 && c.checklist.length >= 5);
});
