// Following points through the video by what they look like.
//
// The bar ends, the floor under the bar and any joint you fixed by hand are
// followed from frame to frame by matching a small patch of the picture
// around them (normalized cross-correlation), so they stay stuck to the real
// thing when the camera pans. Points you place on several frames are all
// kept: between two of your marks the two tracks are blended, so the point
// passes exactly through each mark.
//
// Also: a colour "fingerprint" of a person (what their top and their legs
// look like), used to find the jumper again after they leave the picture.
// Pure functions, no DOM.

const PATCH = 6; // patch radius in small-frame pixels (13 x 13)

function patchAt(g, w, h, x, y, r = PATCH) {
  const xi = Math.round(x), yi = Math.round(y);
  if (xi - r < 0 || yi - r < 0 || xi + r >= w || yi + r >= h) return null;
  const n = (2 * r + 1) ** 2;
  const v = new Float32Array(n);
  let s = 0, k = 0;
  for (let dy = -r; dy <= r; dy++) for (let dx = -r; dx <= r; dx++) { const q = g[(yi + dy) * w + xi + dx]; v[k++] = q; s += q; }
  const m = s / n;
  let ss = 0;
  for (let i = 0; i < n; i++) { v[i] -= m; ss += v[i] * v[i]; }
  const sd = Math.sqrt(ss);
  if (sd < 1e-6 * n || sd / Math.sqrt(n) < 2) return { v, flat: true };
  for (let i = 0; i < n; i++) v[i] /= sd;
  return { v, flat: false };
}

// NCC of a normalized template against frame g, around integer (cx, cy).
function scoreAt(g, w, h, tpl, cx, cy, r = PATCH) {
  if (cx - r < 0 || cy - r < 0 || cx + r >= w || cy + r >= h) return -1;
  const n = (2 * r + 1) ** 2;
  let s = 0, ss = 0;
  for (let dy = -r; dy <= r; dy++) { const o = (cy + dy) * w + cx; for (let dx = -r; dx <= r; dx++) { const q = g[o + dx]; s += q; ss += q * q; } }
  const m = s / n, sd = Math.sqrt(Math.max(0, ss - n * m * m));
  if (sd < 1e-6) return -1;
  let dot = 0, k = 0;
  for (let dy = -r; dy <= r; dy++) { const o = (cy + dy) * w + cx; for (let dx = -r; dx <= r; dx++) dot += tpl[k++] * (g[o + dx] - m); }
  return dot / sd;
}

/**
 * Best match of the templates near (px, py) in frame g. templates: [{v,
 * weight}]. Returns {x, y, score} with sub-pixel x and y, or null.
 */
export function matchNear(g, w, h, templates, px, py, search = 8) {
  const cx = Math.round(px), cy = Math.round(py);
  const size = 2 * search + 1;
  const grid = new Float32Array(size * size).fill(-2);
  let best = -2, bx = 0, by = 0;
  const wsum = templates.reduce((a, t) => a + t.weight, 0) || 1;
  for (let dy = -search; dy <= search; dy++) {
    for (let dx = -search; dx <= search; dx++) {
      let s = 0;
      for (const t of templates) s += t.weight * scoreAt(g, w, h, t.v, cx + dx, cy + dy);
      s /= wsum;
      // A slight pull toward the prediction breaks ties on plain surfaces.
      const sc = s - 0.002 * Math.hypot(dx, dy);
      grid[(dy + search) * size + dx + search] = sc;
      if (sc > best) { best = sc; bx = dx; by = dy; }
    }
  }
  if (best <= -1) return null;
  const at = (dx, dy) => grid[(dy + search) * size + dx + search];
  const sub = (a, b, c) => { const d = a - 2 * b + c; return d < 0 ? Math.max(-0.5, Math.min(0.5, (0.5 * (a - c)) / d)) : 0; };
  const sx = Math.abs(bx) < search ? sub(at(bx - 1, by), best, at(bx + 1, by)) : 0;
  const sy = Math.abs(by) < search ? sub(at(bx, by - 1), best, at(bx, by + 1)) : 0;
  return { x: cx + bx + sx, y: cy + by + sy, score: best + 0.002 * Math.hypot(bx, by) };
}

/**
 * Follow one point from frame `from` to frame `to` (either direction).
 * frames: grayscale Uint8Arrays (w x h, entries may be null). start: {x, y}
 * in small-frame pixels on frame `from`. predict(k, prev, kPrev): where the
 * point should be on frame k given where it was on kPrev (default: same
 * place). Matches against both the patch from the start frame (no drift)
 * and the previous frame (handles slow changes in look).
 * Returns positions indexed by frame: {x, y, score, ok}.
 */
export function followPoint(frames, w, h, from, to, start, { predict = null, search = 8, minScore = 0.55 } = {}) {
  const out = new Map();
  out.set(from, { x: start.x, y: start.y, score: 1, ok: true });
  const g0 = frames[from];
  const anchor = g0 ? patchAt(g0, w, h, start.x, start.y) : null;
  const step = to >= from ? 1 : -1;
  let prev = { x: start.x, y: start.y }, prevTpl = anchor, lostFor = 0;
  for (let k = from + step; step > 0 ? k <= to : k >= to; k += step) {
    const p = predict ? predict(k, prev, k - step) : prev;
    const g = frames[k];
    let res = null;
    if (g && anchor && !anchor.flat) {
      const tpls = [{ v: anchor.v, weight: 1 }];
      if (prevTpl && !prevTpl.flat && prevTpl !== anchor) tpls.push({ v: prevTpl.v, weight: 1 });
      res = matchNear(g, w, h, tpls, p.x, p.y, search + Math.min(8, lostFor * 2));
    }
    if (res && res.score >= minScore) {
      out.set(k, { x: res.x, y: res.y, score: res.score, ok: true });
      prev = { x: res.x, y: res.y };
      prevTpl = patchAt(g, w, h, res.x, res.y) || prevTpl;
      lostFor = 0;
    } else {
      // Not found (hidden behind the jumper, out of the picture, blurred):
      // carry on with the prediction and keep the last good patch.
      out.set(k, { x: p.x, y: p.y, score: res ? res.score : -1, ok: false });
      prev = p;
      lostFor++;
    }
  }
  return out;
}

/**
 * A point placed by hand on one or more frames, followed through all frames.
 * anchors: [{index, x, y}] in small-frame pixels. Between two anchors the
 * track from each is blended, so the result passes through every anchor.
 * Returns an array (one per frame) of {x, y, ok, score}.
 */
export function followAnchored(frames, w, h, anchors, opts = {}) {
  const n = frames.length;
  const A = anchors.slice().sort((a, b) => a.index - b.index);
  const out = new Array(n).fill(null);
  if (!A.length) return out;
  const lo = opts.from ?? 0, hi = opts.to ?? n - 1;
  const first = A[0], last = A[A.length - 1];
  const back = followPoint(frames, w, h, first.index, Math.max(lo, 0), first, opts);
  for (const [k, p] of back) out[k] = p;
  const fwd = followPoint(frames, w, h, last.index, Math.min(hi, n - 1), last, opts);
  for (const [k, p] of fwd) out[k] = p;
  for (let j = 0; j + 1 < A.length; j++) {
    const a = A[j], b = A[j + 1];
    if (b.index === a.index) continue;
    const fa = followPoint(frames, w, h, a.index, b.index, a, opts);
    const fb = followPoint(frames, w, h, b.index, a.index, b, opts);
    for (let k = a.index; k <= b.index; k++) {
      const pa = fa.get(k), pb = fb.get(k);
      const u = (k - a.index) / (b.index - a.index);
      // Linear blend by time, leaning on whichever track still sees the point.
      let wa = (1 - u) * (pa.ok ? 1 : 0.2), wb = u * (pb.ok ? 1 : 0.2);
      if (k === a.index) { wa = 1; wb = 0; }
      if (k === b.index) { wa = 0; wb = 1; }
      const s = wa + wb || 1;
      out[k] = { x: (pa.x * wa + pb.x * wb) / s, y: (pa.y * wa + pb.y * wb) / s, ok: pa.ok || pb.ok, score: Math.max(pa.score, pb.score) };
    }
  }
  return out;
}

const median = (v) => { const s = v.slice().sort((a, b) => a - b); const m = s.length >> 1; return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2; };

/**
 * Better camera offsets from points that do not move in the world (the bar
 * ends, the floor under the bar). tracks: per point, an array of {x, y, ok}
 * in normalized coordinates. ref: those points' normalized positions on frame
 * refIndex. cum: the existing offsets from the whole-picture estimate.
 * On frames where the points are seen, the offset is the median of how far
 * they moved from ref (points that disagree with the rest are ignored);
 * elsewhere the whole-picture estimate is used, shifted to join up smoothly.
 */
export function cameraFromPoints(cum, tracks, ref, refIndex) {
  const n = cum.length;
  const base = cum[refIndex] || { x: 0, y: 0 };
  const fromPts = new Array(n).fill(null);
  for (let k = 0; k < n; k++) {
    const dx = [], dy = [];
    tracks.forEach((t, j) => { const p = t && t[k]; if (p && p.ok && ref[j]) { dx.push(p.x - ref[j].x); dy.push(p.y - ref[j].y); } });
    if (!dx.length) continue;
    const mx = median(dx), my = median(dy);
    // Drop points that moved differently from the others (knocked bar,
    // hidden end), then take the median of the rest.
    const keep = dx.map((v, i) => Math.hypot(v - mx, dy[i] - my) < 0.02);
    const kx = dx.filter((_, i) => keep[i]), ky = dy.filter((_, i) => keep[i]);
    if (!kx.length || (dx.length >= 2 && kx.length < 2 && tracks.length > 2)) continue;
    fromPts[k] = { x: base.x + median(kx), y: base.y + median(ky) };
  }
  // Difference between the point-based and whole-picture offsets, filled in
  // across frames without points by joining the nearest known differences.
  const diff = fromPts.map((p, k) => (p ? { x: p.x - cum[k].x, y: p.y - cum[k].y } : null));
  const known = diff.map((d, k) => (d ? k : -1)).filter((k) => k >= 0);
  if (!known.length) return cum.map((c) => ({ ...c }));
  const out = new Array(n);
  for (let k = 0; k < n; k++) {
    if (diff[k]) { out[k] = { x: cum[k].x + diff[k].x, y: cum[k].y + diff[k].y }; continue; }
    let p = -1, q = -1;
    for (const kk of known) { if (kk < k) p = kk; else { q = kk; break; } }
    let d;
    if (p < 0) d = diff[q];
    else if (q < 0) d = diff[p];
    else { const u = (k - p) / (q - p); d = { x: diff[p].x + (diff[q].x - diff[p].x) * u, y: diff[p].y + (diff[q].y - diff[p].y) * u }; }
    out[k] = { x: cum[k].x + d.x, y: cum[k].y + d.y };
  }
  return out;
}

// ---------- appearance ----------
// Points along the body where colour is sampled: [from, to, steps].
const UPPER = [[11, 23, 4], [12, 24, 4], [11, 13, 2], [12, 14, 2], [11, 12, 3]];
const LOWER = [[23, 25, 3], [24, 26, 3], [25, 27, 3], [26, 28, 3]];
const BINS = 20; // 8 hues x 2 brightness + 4 greys

function bin(r, g, b) {
  const mx = Math.max(r, g, b), mn = Math.min(r, g, b);
  const v = mx / 255, s = mx ? (mx - mn) / mx : 0;
  if (s < 0.22 || mx < 40) return 16 + Math.min(3, Math.floor(v * 4));
  let hh;
  const d = mx - mn;
  if (mx === r) hh = ((g - b) / d + 6) % 6;
  else if (mx === g) hh = (b - r) / d + 2;
  else hh = (r - g) / d + 4;
  return Math.min(7, Math.floor((hh / 6) * 8)) * 2 + (v > 0.6 ? 1 : 0);
}

function sampleHist(rgba, w, h, lms, segs) {
  const hist = new Float32Array(BINS);
  let n = 0;
  for (const [a, b, steps] of segs) {
    const A = lms[a], B = lms[b];
    if (!A || !B) continue;
    for (let s = 1; s <= steps; s++) {
      const u = s / (steps + 1);
      const x = Math.round((A.x + (B.x - A.x) * u) * w), y = Math.round((A.y + (B.y - A.y) * u) * h);
      for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
        const xx = x + dx, yy = y + dy;
        if (xx < 0 || yy < 0 || xx >= w || yy >= h) continue;
        const i = (yy * w + xx) * 4;
        hist[bin(rgba[i], rgba[i + 1], rgba[i + 2])]++;
        n++;
      }
    }
  }
  if (n < 9) return null;
  for (let i = 0; i < BINS; i++) hist[i] /= n;
  return hist;
}

/** Colour fingerprint of a pose in an RGBA image: {upper, lower} or null. */
export function appearanceOf(rgba, w, h, lms) {
  const upper = sampleHist(rgba, w, h, lms, UPPER), lower = sampleHist(rgba, w, h, lms, LOWER);
  return upper || lower ? { upper, lower } : null;
}

const bhatta = (a, b) => { let s = 0; for (let i = 0; i < a.length; i++) s += Math.sqrt(a[i] * b[i]); return s; };

/** 0 (nothing alike) to 1 (same colours). */
export function appearanceSim(a, b) {
  if (!a || !b) return null;
  const parts = [];
  if (a.upper && b.upper) parts.push(bhatta(a.upper, b.upper));
  if (a.lower && b.lower) parts.push(bhatta(a.lower, b.lower));
  return parts.length ? parts.reduce((x, y) => x + y, 0) / parts.length : null;
}

/** Running average of fingerprints (rate = weight of the new one). */
export function blendAppearance(model, next, rate = 0.15) {
  if (!next) return model;
  if (!model) return { upper: next.upper && Float32Array.from(next.upper), lower: next.lower && Float32Array.from(next.lower) };
  const mix = (m, x) => { if (!x) return m; if (!m) return Float32Array.from(x); const o = new Float32Array(m.length); for (let i = 0; i < m.length; i++) o[i] = m[i] * (1 - rate) + x[i] * rate; return o; };
  return { upper: mix(model.upper, next.upper), lower: mix(model.lower, next.lower) };
}
