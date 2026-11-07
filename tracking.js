// Tracking cleanup between the pose model and the jump analysis.
// Pure functions on frames: [{t, lms: [{x, y, visibility}] x33 normalized | null}].
//
// The pose model looks at each frame on its own, so raw output has four
// common faults this module fixes:
//   1. It can lock onto a coach or spectator  -> pickJumper
//   2. In side views it swaps left and right  -> fixSwaps
//   3. It loses the jumper for a few frames   -> fillGaps
//   4. Points jitter from frame to frame      -> smoothFrames

const dist = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);
const hipMid = (l) => ({ x: (l[23].x + l[24].x) / 2, y: (l[23].y + l[24].y) / 2 });
const ankleMid = (l) => ({ x: (l[27].x + l[28].x) / 2, y: (l[27].y + l[28].y) / 2 });
const bodySize = (l) => dist(l[0], ankleMid(l));
const copy = (lms) => lms.map((p) => ({ ...p }));

/**
 * posesPerFrame: [{t, poses: [lms, ...]}]. Picks one person per frame: the
 * largest person to start, then whoever is nearest to where the jumper was.
 * Returns frames [{t, lms|null}].
 */
export function pickJumper(posesPerFrame, maxJump = 0.25) {
  let last = null; // last chosen hip position
  let lastT = null;
  return posesPerFrame.map(({ t, poses }) => {
    if (!poses || !poses.length) return { t, lms: null };
    let pick;
    if (!last) {
      pick = poses.reduce((a, b) => (bodySize(b) > bodySize(a) ? b : a));
    } else {
      pick = poses.reduce((a, b) => (dist(hipMid(b), last) < dist(hipMid(a), last) ? b : a));
      // Allow more movement the longer the jumper has been lost.
      const allowed = maxJump * Math.max(1, (t - lastT) / 0.1);
      if (dist(hipMid(pick), last) > allowed) return { t, lms: null };
    }
    last = hipMid(pick);
    lastT = t;
    return { t, lms: copy(pick) };
  });
}

// Left/right landmark pairs, grouped by limb so the legs and arms are fixed
// separately (legs cross while running, arms swing on their own).
const GROUPS = {
  legs: [[23, 24], [25, 26], [27, 28], [29, 30], [31, 32]],
  arms: [[11, 12], [13, 14], [15, 16], [17, 18], [19, 20], [21, 22]],
  face: [[1, 4], [2, 5], [3, 6], [7, 8], [9, 10]],
};

/**
 * Swap left/right labels when that keeps each limb closer to where it was
 * heading (predicted from the previous two frames).
 */
export function fixSwaps(frames) {
  const out = frames.map((f) => ({ ...f, lms: f.lms ? copy(f.lms) : null }));
  let prev = null, prev2 = null;
  for (const f of out) {
    if (!f.lms) continue;
    if (prev) {
      const predict = (k) => (prev2
        ? { x: 2 * prev[k].x - prev2[k].x, y: 2 * prev[k].y - prev2[k].y }
        : prev[k]);
      for (const pairs of Object.values(GROUPS)) {
        let keep = 0, swap = 0;
        for (const [l, r] of pairs) {
          const pl = predict(l), pr = predict(r);
          keep += dist(f.lms[l], pl) + dist(f.lms[r], pr);
          swap += dist(f.lms[r], pl) + dist(f.lms[l], pr);
        }
        // Require a clear win so near-ties (limbs crossing) are left alone.
        if (swap < keep * 0.7) {
          for (const [l, r] of pairs) [f.lms[l], f.lms[r]] = [f.lms[r], f.lms[l]];
          f.swapped = true;
        }
      }
    }
    prev2 = prev;
    prev = f.lms;
  }
  return out;
}

/** Linearly fill runs of up to maxGap missing frames between two found frames. */
export function fillGaps(frames, maxGap = 5) {
  const out = frames.map((f) => ({ ...f }));
  let i = 0;
  while (i < out.length) {
    if (out[i].lms) { i++; continue; }
    const s = i - 1;
    let e = i;
    while (e < out.length && !out[e].lms) e++;
    if (s >= 0 && e < out.length && e - s - 1 <= maxGap) {
      const A = out[s].lms, B = out[e].lms;
      for (let k = s + 1; k < e; k++) {
        const u = (out[k].t - out[s].t) / (out[e].t - out[s].t);
        out[k] = {
          ...out[k],
          lms: A.map((p, j) => ({
            x: p.x + (B[j].x - p.x) * u,
            y: p.y + (B[j].y - p.y) * u,
            visibility: Math.min(p.visibility ?? 1, B[j].visibility ?? 1),
          })),
          filled: true,
        };
      }
    }
    i = e;
  }
  return out;
}

/**
 * Centered weighted moving average of x and y, weighted by visibility so
 * uncertain points move toward their confident neighbours. radius 1 uses
 * weights 1-2-1, radius 2 uses 1-2-3-2-1. Never smooths across a missing frame.
 */
export function smoothFrames(frames, radius = 1) {
  if (radius < 1) return frames;
  const kernel = [];
  for (let d = -radius; d <= radius; d++) kernel.push(radius + 1 - Math.abs(d));
  return frames.map((f, i) => {
    if (!f.lms) return f;
    const lms = f.lms.map((p, j) => {
      let sx = 0, sy = 0, sw = 0;
      for (let d = -radius; d <= radius; d++) {
        const g = frames[i + d];
        if (!g || !g.lms) continue;
        const q = g.lms[j];
        const w = kernel[d + radius] * Math.max(0.05, q.visibility ?? 1);
        sx += q.x * w; sy += q.y * w; sw += w;
      }
      return { x: sx / sw, y: sy / sw, visibility: p.visibility };
    });
    return { ...f, lms };
  });
}

/** Overwrite points the user dragged by hand. corrections: {frameIndex: {landmark: {x, y}}} */
export function applyCorrections(frames, corrections) {
  return frames.map((f, i) => {
    const c = corrections[i];
    if (!c || !f.lms) return f;
    const lms = copy(f.lms);
    for (const [k, p] of Object.entries(c)) lms[k] = { x: p.x, y: p.y, visibility: 1 };
    return { ...f, lms, corrected: true };
  });
}

/** Full cleanup pipeline used by the app. */
export function cleanTracking(posesPerFrame, { smoothRadius = 1, maxGap = 5 } = {}) {
  const picked = pickJumper(posesPerFrame);
  return smoothFrames(fillGaps(fixSwaps(picked), maxGap), smoothRadius);
}

// ---------- Multi-person tracks and choosing the jumper ----------
// The pose model sometimes "sees" people in the landing mat, crowd or
// equipment. Those ghosts barely move. The jumper runs across the frame and
// rises, so tracks are scored by motion, not by size.

const KEY_POINTS = [0, 11, 12, 23, 24, 25, 26, 27, 28];

export function meanVisibility(lms, idx = KEY_POINTS) {
  return idx.reduce((s, k) => s + (lms[k].visibility ?? 1), 0) / idx.length;
}

/** Bounding box of the main body points, normalized: {cx, cy, w, h, size}. */
export function poseBox(lms, aspect = 1) {
  const idx = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28];
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  for (const k of idx) {
    const p = lms[k];
    x0 = Math.min(x0, p.x); x1 = Math.max(x1, p.x); y0 = Math.min(y0, p.y); y1 = Math.max(y1, p.y);
  }
  const w = x1 - x0, h = y1 - y0;
  // size in "height units" so a horizontal body (in flight) counts the same.
  return { cx: (x0 + x1) / 2, cy: (y0 + y1) / 2, w, h, size: Math.max(w * aspect, h) };
}

/**
 * Is this a believable human pose? Rejects ghosts with impossible limb
 * proportions or low confidence.
 */
export function plausiblePose(lms, aspect = 1) {
  if (!lms || lms.length < 33) return false;
  if (meanVisibility(lms) < 0.35) return false;
  const d = (a, b) => Math.hypot((lms[a].x - lms[b].x) * aspect, lms[a].y - lms[b].y);
  const thighL = d(23, 25), shankL = d(25, 27), thighR = d(24, 26), shankR = d(26, 28);
  const torso = Math.hypot(((lms[11].x + lms[12].x) / 2 - (lms[23].x + lms[24].x) / 2) * aspect, (lms[11].y + lms[12].y) / 2 - (lms[23].y + lms[24].y) / 2);
  const leg = (thighL + shankL + thighR + shankR) / 2;
  if (leg <= 0 || torso <= 0) return false;
  const ratio = (a, b) => a / Math.max(b, 1e-6);
  if (ratio(thighL, shankL) > 2.6 || ratio(shankL, thighL) > 2.6) return false;
  if (ratio(thighR, shankR) > 2.6 || ratio(shankR, thighR) > 2.6) return false;
  if (ratio(torso, leg) > 1.6 || ratio(leg, torso) > 4) return false;
  return true;
}

/**
 * Link detections across frames into tracks by nearest hip position.
 * detections: [{t, poses: [lms]}]. Returns [{id, items: [{frame, lms}]}].
 */
export function buildTracks(detections, { maxStep = 0.12, maxMissing = 0.5, aspect = 1 } = {}) {
  const tracks = [];
  let nextId = 0;
  detections.forEach(({ t, poses }, frame) => {
    const cands = (poses || []).filter((p) => plausiblePose(p, aspect));
    const taken = new Set();
    // Match existing live tracks, closest first.
    const pairs = [];
    for (const tr of tracks) {
      const last = tr.items[tr.items.length - 1];
      const dt = t - tr.lastT;
      if (dt > maxMissing) continue;
      const lh = hipMid(last.lms);
      cands.forEach((c, ci) => {
        const ch = hipMid(c);
        const dd = Math.hypot((ch.x - lh.x) * aspect, ch.y - lh.y);
        const allowed = maxStep * Math.max(1, dt / (1 / 30));
        if (dd <= allowed) pairs.push({ tr, ci, dd });
      });
    }
    pairs.sort((a, b) => a.dd - b.dd);
    const used = new Set();
    for (const { tr, ci } of pairs) {
      if (used.has(tr.id) || taken.has(ci)) continue;
      tr.items.push({ frame, lms: cands[ci] });
      tr.lastT = t;
      used.add(tr.id); taken.add(ci);
    }
    cands.forEach((c, ci) => {
      if (taken.has(ci)) return;
      tracks.push({ id: nextId++, items: [{ frame, lms: c }], lastT: t });
    });
  });
  return tracks;
}

/** How much a track looks like the jumper: long, confident and moving. */
export function scoreTrack(track, frameCount, aspect = 1) {
  if (track.items.length < 3) return 0;
  const hips = track.items.map((it) => hipMid(it.lms));
  const xs = hips.map((p) => p.x * aspect), ys = hips.map((p) => p.y);
  const travel = (Math.max(...xs) - Math.min(...xs)) + (Math.max(...ys) - Math.min(...ys));
  const sizes = track.items.map((it) => poseBox(it.lms, aspect).size).sort((a, b) => a - b);
  const size = sizes[sizes.length >> 1];
  const vis = track.items.reduce((s, it) => s + meanVisibility(it.lms), 0) / track.items.length;
  const coverage = track.items.length / Math.max(1, frameCount);
  // Travel measured in body sizes: a jumper covers several, a ghost almost none.
  const travelBodies = travel / Math.max(size, 1e-6);
  return Math.min(travelBodies, 8) * vis * Math.sqrt(coverage);
}

export function pickTrack(tracks, frameCount, aspect = 1) {
  let best = null, bestS = 0;
  for (const tr of tracks) {
    const s = scoreTrack(tr, frameCount, aspect);
    if (s > bestS) { bestS = s; best = tr; }
  }
  return best;
}

/** The track whose pose on this frame is under (or nearest) a tapped point. */
export function trackAt(tracks, frame, point, aspect = 1) {
  let best = null, bestD = Infinity;
  for (const tr of tracks) {
    const it = tr.items.find((x) => x.frame === frame);
    if (!it) continue;
    const b = poseBox(it.lms, aspect);
    const inside = Math.abs(point.x - b.cx) <= b.w / 2 + 0.02 && Math.abs(point.y - b.cy) <= b.h / 2 + 0.02;
    const d = Math.hypot((point.x - b.cx) * aspect, point.y - b.cy) - (inside ? 10 : 0);
    if (d < bestD) { bestD = d; best = tr; }
  }
  return best;
}
