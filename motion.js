// Camera motion. Phone clips are often handheld and pan to follow the
// jumper, which makes a planted foot look like it slides and bends the
// flight path. We estimate how far the whole picture shifts between frames
// and undo it, so measurements happen in "ground" coordinates.
//
// Method: edge strength (gradient magnitude) is summed down each column and
// along each row, giving two 1D profiles per frame. The shift that best lines
// up the profiles of two frames is the camera pan/tilt. The jumper's box is
// masked out so their own movement does not count as camera movement.

export function gradientMagnitude(gray, w, h, mask = null) {
  const g = new Float32Array(w * h);
  for (let y = 1; y < h - 1; y++) {
    for (let x = 1; x < w - 1; x++) {
      const i = y * w + x;
      if (mask && mask[i]) continue;
      g[i] = Math.abs(gray[i + 1] - gray[i - 1]) + Math.abs(gray[i + w] - gray[i - w]);
    }
  }
  return g;
}

// Mean edge strength per column and per row, over unmasked pixels only, so a
// masked area does not show up as a fixed feature that pins the match at 0.
function profiles(g, w, h, mask) {
  const col = new Float32Array(w), row = new Float32Array(h);
  const cc = new Float32Array(w), rc = new Float32Array(h);
  for (let y = 1; y < h - 1; y++) for (let x = 1; x < w - 1; x++) {
    const i = y * w + x;
    if (mask && mask[i]) continue;
    const v = g[i];
    col[x] += v; row[y] += v; cc[x]++; rc[y]++;
  }
  for (let x = 0; x < w; x++) col[x] = cc[x] ? col[x] / cc[x] : 0;
  for (let y = 0; y < h; y++) row[y] = rc[y] ? row[y] / rc[y] : 0;
  const norm = (p) => { const m = p.reduce((a, b) => a + b, 0) / p.length; for (let i = 0; i < p.length; i++) p[i] -= m; return p; };
  return { col: norm(col), row: norm(row) };
}

// Best shift s so that b[i + s] matches a[i]; sub-pixel by parabola fit.
function bestShift(a, b, maxShift) {
  const n = a.length;
  const costs = [];
  for (let s = -maxShift; s <= maxShift; s++) {
    let c = 0, cnt = 0;
    for (let i = Math.max(0, -s); i < Math.min(n, n - s); i++) { c += Math.abs(a[i] - b[i + s]); cnt++; }
    costs.push(cnt > n * 0.4 ? c / cnt : Infinity);
  }
  let k = 0;
  for (let i = 1; i < costs.length; i++) if (costs[i] < costs[k]) k = i;
  let sub = 0;
  if (k > 0 && k < costs.length - 1 && Number.isFinite(costs[k - 1]) && Number.isFinite(costs[k + 1])) {
    const d = costs[k - 1] - 2 * costs[k] + costs[k + 1];
    if (d > 0) sub = (0.5 * (costs[k - 1] - costs[k + 1])) / d;
  }
  return k - maxShift + Math.max(-0.5, Math.min(0.5, sub));
}

/** How far the picture content moved from frame a to frame b, in pixels. */
export function estimateShift(a, b, w, h, { mask = null, maxShift = null } = {}) {
  const pa = profiles(gradientMagnitude(a, w, h), w, h, mask);
  const pb = profiles(gradientMagnitude(b, w, h), w, h, mask);
  const mx = maxShift ?? Math.round(w * 0.2);
  const my = maxShift ?? Math.round(h * 0.12);
  return { dx: bestShift(pa.col, pb.col, mx), dy: bestShift(pa.row, pb.row, my) };
}

/** Mask (1 = ignore) for a box in small-image pixels, padded a little. */
export function boxMask(w, h, box, pad = 0.15) {
  const m = new Uint8Array(w * h);
  if (!box) return m;
  const px = box.w * pad, py = box.h * pad;
  const x0 = Math.max(0, Math.floor(box.x - px)), x1 = Math.min(w - 1, Math.ceil(box.x + box.w + px));
  const y0 = Math.max(0, Math.floor(box.y - py)), y1 = Math.min(h - 1, Math.ceil(box.y + box.h + py));
  for (let y = y0; y <= y1; y++) m.fill(1, y * w + x0, y * w + x1 + 1);
  return m;
}

/**
 * Cumulative camera offset per frame (in small-image pixels): where a fixed
 * point from frame 0 appears in frame k is (x0 + cum[k].x, y0 + cum[k].y).
 * grays: small grayscale frames; boxes: jumper box per frame (or null).
 */
export function cameraPath(grays, w, h, boxes = []) {
  const cum = [{ x: 0, y: 0 }];
  for (let k = 1; k < grays.length; k++) {
    const m1 = boxMask(w, h, boxes[k - 1]), m2 = boxMask(w, h, boxes[k]);
    const mask = new Uint8Array(w * h);
    for (let i = 0; i < mask.length; i++) mask[i] = m1[i] | m2[i];
    const s = grays[k - 1] && grays[k] ? estimateShift(grays[k - 1], grays[k], w, h, { mask }) : { dx: 0, dy: 0 };
    cum.push({ x: cum[k - 1].x + s.dx, y: cum[k - 1].y + s.dy });
  }
  return cum;
}

/** Undo camera motion: frame landmarks -> ground coordinates (normalized). */
export function stabilize(frames, cumNorm) {
  return frames.map((f, k) => {
    if (!f.lms || !cumNorm[k]) return f;
    const { x, y } = cumNorm[k];
    return { ...f, lms: f.lms.map((p) => ({ ...p, x: p.x - x, y: p.y - y })) };
  });
}
