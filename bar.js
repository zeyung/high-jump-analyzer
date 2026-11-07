// Finds the high jump bar in the video. Pure functions on grayscale images,
// so they run in the browser and in Node tests.
//
// 1. medianBackground: per-pixel median of frames sampled across the clip.
//    The jumper moves and disappears; the bar stays.
// 2. ridgeMap: the bar is a thin line, so it differs from the pixels a few
//    rows above AND below in the same direction. A step edge (the top of the
//    landing mat, the horizon) only differs on one side and scores zero.
// 3. findLines: a Hough transform for near-level lines, then the longest
//    supported run along each line gives the bar's two ends.
// 4. chooseBar: among the line candidates, the bar is the one that makes the
//    entered bar height agree with the athlete's size at the takeoff spot.

export function toGray(rgba, w, h) {
  const g = new Float32Array(w * h);
  for (let i = 0, j = 0; i < g.length; i++, j += 4) g[i] = 0.299 * rgba[j] + 0.587 * rgba[j + 1] + 0.114 * rgba[j + 2];
  return g;
}

export function medianBackground(frames) {
  const n = frames.length;
  if (!n) return null;
  const len = frames[0].length;
  const out = new Float32Array(len);
  const buf = new Float32Array(n);
  for (let p = 0; p < len; p++) {
    for (let k = 0; k < n; k++) buf[k] = frames[k][p];
    // insertion sort: n is small (about a dozen frames)
    for (let a = 1; a < n; a++) {
      const v = buf[a];
      let b = a - 1;
      while (b >= 0 && buf[b] > v) { buf[b + 1] = buf[b]; b--; }
      buf[b + 1] = v;
    }
    out[p] = buf[n >> 1];
  }
  return out;
}

export function ridgeMap(gray, w, h, dists = [2, 3, 4, 5]) {
  const r = new Float32Array(w * h);
  const dmax = Math.max(...dists);
  for (let y = dmax; y < h - dmax; y++) {
    for (let x = 0; x < w; x++) {
      const c = gray[y * w + x];
      let best = 0;
      for (const d of dists) {
        const up = gray[(y - d) * w + x], dn = gray[(y + d) * w + x];
        const bright = Math.min(c - up, c - dn);
        const dark = Math.min(up - c, dn - c);
        const v = bright > dark ? bright : dark;
        if (v > best) best = v;
      }
      r[y * w + x] = best;
    }
  }
  return r;
}

function percentileOf(arr, p) {
  const s = Array.from(arr).filter((v) => v > 0).sort((a, b) => a - b);
  if (!s.length) return 0;
  return s[Math.min(s.length - 1, Math.floor(p * (s.length - 1)))];
}

/**
 * Near-level line segments in a ridge map. Returns candidates with
 * normalized ends {a, b}, a score, and the angle in degrees.
 */
export function findLines(ridge, w, h, { maxAngle = 15, angleStep = 0.5, minLengthFrac = 0.12, maxCandidates = 6 } = {}) {
  const thr = Math.max(8, percentileOf(ridge, 0.97));
  const angles = [];
  for (let a = -maxAngle; a <= maxAngle + 1e-9; a += angleStep) angles.push(a);
  const tans = angles.map((a) => Math.tan((a * Math.PI) / 180));
  const offset = h; // intercept rows from -h to 2h
  const rows = 3 * h;
  const acc = new Float32Array(angles.length * rows);
  const cx = w / 2;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const v = ridge[y * w + x];
      if (v < thr) continue;
      for (let ai = 0; ai < angles.length; ai++) {
        const yc = Math.round(y - tans[ai] * (x - cx)) + offset;
        if (yc >= 0 && yc < rows) acc[ai * rows + yc] += v;
      }
    }
  }
  // Peaks with non-maximum suppression.
  const peaks = [];
  const used = new Uint8Array(acc.length);
  for (let n = 0; n < maxCandidates * 4; n++) {
    let bi = -1, bv = 0;
    for (let i = 0; i < acc.length; i++) if (!used[i] && acc[i] > bv) { bv = acc[i]; bi = i; }
    if (bi < 0 || bv <= 0) break;
    const ai = Math.floor(bi / rows), yc = bi % rows;
    peaks.push({ ai, yc: yc - offset });
    for (let da = -6; da <= 6; da++) for (let dy = -5; dy <= 5; dy++) {
      const a2 = ai + da, y2 = yc + dy;
      if (a2 >= 0 && a2 < angles.length && y2 >= 0 && y2 < rows) used[a2 * rows + y2] = 1;
    }
  }
  // Longest supported run along each line.
  const maxGap = Math.max(6, Math.round(w * 0.02));
  const out = [];
  for (const { ai, yc } of peaks) {
    const t = tans[ai];
    let runStart = -1, lastHit = -1, sum = 0, best = null, runSum = 0, hits = 0, runHits = 0;
    for (let x = 0; x < w; x++) {
      const y = Math.round(yc + t * (x - cx));
      let v = 0;
      for (let dy = -1; dy <= 1; dy++) {
        const yy = y + dy;
        if (yy >= 0 && yy < h) v = Math.max(v, ridge[yy * w + x]);
      }
      if (v >= thr * 0.6) {
        if (runStart < 0 || x - lastHit > maxGap) { runStart = x; runSum = 0; runHits = 0; }
        lastHit = x; runSum += v; runHits++;
        const len = lastHit - runStart;
        if (!best || len > best.len) best = { x1: runStart, x2: lastHit, len, mean: runSum / runHits, fill: runHits / (len + 1) };
      }
    }
    void sum; void hits;
    if (!best || best.len < minLengthFrac * w) continue;
    const yAt = (x) => yc + t * (x - cx);
    out.push({
      a: { x: best.x1 / w, y: yAt(best.x1) / h },
      b: { x: best.x2 / w, y: yAt(best.x2) / h },
      angle: angles[ai],
      score: best.len * best.mean * best.fill,
    });
  }
  out.sort((p, q) => q.score - p.score);
  return out.slice(0, maxCandidates);
}

/** Full detection on sampled grayscale frames (all w x h). */
export function detectBarCandidates(grayFrames, w, h, opts) {
  const bg = medianBackground(grayFrames);
  if (!bg) return [];
  return findLines(ridgeMap(bg, w, h), w, h, opts);
}

/** y of a normalized line at normalized x (extends past the ends). */
export function lineYAt(line, x) {
  const { a, b } = line;
  if (Math.abs(b.x - a.x) < 1e-9) return (a.y + b.y) / 2;
  return a.y + ((b.y - a.y) * (x - a.x)) / (b.x - a.x);
}

/**
 * Pick the bar among candidates using the athlete:
 * - the bar must be above the floor at the takeoff spot;
 * - the bar height entered, divided by the bar's height in pixels there,
 *   gives a scale that should agree with the scale from the athlete's own
 *   height (within a factor of about 1.6 either way);
 * - the bar should reach over where the jumper peaked.
 * info: {groundPx: {x, y}, peakHipPx: {x, y}, cmPerPxFromHeight}, in pixels.
 */
export function chooseBar(candidates, info, barCm, w, h) {
  let best = null;
  for (const c of candidates) {
    const gx = info.groundPx.x / w;
    const barY = lineYAt(c, gx) * h;
    const barPx = info.groundPx.y - barY;
    if (barPx <= 10) continue;
    const ratio = barCm / barPx / info.cmPerPxFromHeight;
    if (!(ratio > 0.62 && ratio < 1.6)) continue;
    const minX = Math.min(c.a.x, c.b.x) * w, maxX = Math.max(c.a.x, c.b.x) * w;
    const reach = (maxX - minX) * 0.15;
    const overPeak = info.peakHipPx.x >= minX - reach && info.peakHipPx.x <= maxX + reach;
    const fit = Math.exp(-Math.abs(Math.log(ratio)) * 3) * (overPeak ? 1 : 0.4);
    const s = c.score * fit;
    if (!best || s > best.s) best = { s, c, ratio };
  }
  return best ? { a: best.c.a, b: best.c.b, scaleRatio: best.ratio, auto: true } : null;
}
