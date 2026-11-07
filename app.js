import { analyzeJump, angleAt, mid, toPx, phaseAt, LM } from './analysis.js';
import { cleanTracking, applyCorrections, plausiblePose, poseBox, flipSides } from './tracking.js';
import { cameraPath, stabilize } from './motion.js';
import { coachJump } from './coaching.js';

const TV = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14';
const MODELS = {
  standard: 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task',
  high: 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task',
};
// Standard: full model, 30 frames a second. High: heavy model, 60 a second.
const QUALITY = {
  standard: { model: 'standard', fps: 30, smoothRadius: 1 },
  high: { model: 'high', fps: 60, smoothRadius: 2 },
};
const MAX_SECONDS = 15;
const CROP = 512; // the model runs on a 512 x 512 crop that follows the jumper
const MOTION_W = 120; // small frames for measuring camera pans

const $ = (id) => document.getElementById(id);
const video = $('video');
const canvas = $('canvas');
const ctx = canvas.getContext('2d');
const setupCanvas = $('setupCanvas');
const sctx = setupCanvas.getContext('2d');
const cropCanvas = document.createElement('canvas');
cropCanvas.width = cropCanvas.height = CROP;
const cctx = cropCanvas.getContext('2d');
const motionCanvas = document.createElement('canvas');
const mctx = motionCanvas.getContext('2d', { willReadFrequently: true });

const landmarkers = {};
let state = null; // results
let setup = null; // the pick-the-jumper step
// For checking a run from the browser console.
window.hjState = () => state;
window.hjDebug = { detectCrop: (...a) => detectCrop(...a), detectFull: (...a) => detectFull(...a), seekTo: (t) => seekTo(t), getLandmarker: (k) => getLandmarker(k) };

// ---------- markers, colors, what to show ----------
// Lighter shade + solid dot = left side; deeper shade + ring = right side.
const MARKERS = [
  { k: 0, name: 'Head', color: '#ffffff', side: 'C' },
  { k: 11, name: 'Left shoulder', color: '#ff9fd0', side: 'L' },
  { k: 12, name: 'Right shoulder', color: '#e0218a', side: 'R' },
  { k: 13, name: 'Left elbow', color: '#cdb2ff', side: 'L' },
  { k: 14, name: 'Right elbow', color: '#7b3dff', side: 'R' },
  { k: 15, name: 'Left wrist', color: '#a8d8ff', side: 'L' },
  { k: 16, name: 'Right wrist', color: '#1f6fff', side: 'R' },
  { k: 23, name: 'Left hip', color: '#ffe680', side: 'L' },
  { k: 24, name: 'Right hip', color: '#e6a800', side: 'R' },
  { k: 25, name: 'Left knee', color: '#ffbd80', side: 'L' },
  { k: 26, name: 'Right knee', color: '#ff5f00', side: 'R' },
  { k: 27, name: 'Left ankle', color: '#9ff5df', side: 'L' },
  { k: 28, name: 'Right ankle', color: '#00a383', side: 'R' },
];
const MK = Object.fromEntries(MARKERS.map((m) => [m.k, m]));
const LAYERS = [
  { key: 'skeleton', name: 'Skeleton lines' },
  { key: 'lean', name: 'Lean line (to takeoff)' },
  { key: 'bar', name: 'Bar' },
  { key: 'clearance', name: 'Clearance check' },
  { key: 'phase', name: 'Phase and time' },
  { key: 'legend', name: 'Legend on video (for downloads)' },
];
const BONES_L = [[11, 13], [13, 15], [11, 23], [23, 25], [25, 27], [27, 29], [29, 31], [27, 31]];
const BONES_R = [[12, 14], [14, 16], [12, 24], [24, 26], [26, 28], [28, 30], [30, 32], [28, 32]];
const BONES_C = [[11, 12], [23, 24]];
const PART_POINTS = { head: [0], shoulders: [11, 12], hips: [23, 24], knees: [25, 26], feet: [27, 28] };

const VIEW_KEY = 'hj-view-2';
const view = loadView();
function loadView() {
  const v = { markers: Object.fromEntries(MARKERS.map((m) => [m.k, true])), layers: { skeleton: true, lean: false, bar: true, clearance: true, phase: true, legend: false } };
  try {
    const saved = JSON.parse(localStorage.getItem(VIEW_KEY) || 'null');
    if (saved) { Object.assign(v.markers, saved.markers || {}); Object.assign(v.layers, saved.layers || {}); }
  } catch { /* storage blocked: use defaults */ }
  return v;
}
function saveView() { try { localStorage.setItem(VIEW_KEY, JSON.stringify(view)); } catch { /* ignore */ } }

function swatchSvg(m, size = 18) {
  const c = size / 2;
  if (m.side === 'R') return `<svg width="${size}" height="${size}" aria-hidden="true"><circle cx="${c}" cy="${c}" r="${c - 2.5}" fill="none" stroke="#000" stroke-width="5"/><circle cx="${c}" cy="${c}" r="${c - 2.5}" fill="none" stroke="${m.color}" stroke-width="3"/></svg>`;
  return `<svg width="${size}" height="${size}" aria-hidden="true"><circle cx="${c}" cy="${c}" r="${c - 4}" fill="${m.color}" stroke="#000" stroke-width="1.5"/></svg>`;
}

function buildLegend() {
  const items = MARKERS.map((m) => `
    <label class="lg-item"><input type="checkbox" data-marker="${m.k}" ${view.markers[m.k] ? 'checked' : ''}>${swatchSvg(m)}<span>${m.name}</span></label>`).join('');
  const layers = LAYERS.map((l) => `
    <label class="lg-item"><input type="checkbox" data-layer="${l.key}" ${view.layers[l.key] ? 'checked' : ''}><span>${l.name}</span></label>`).join('');
  $('legend').innerHTML = `
    <div class="lg-head"><strong>Points</strong>
      <span class="lg-quick"><button type="button" class="link" data-all="1">Show all</button> · <button type="button" class="link" data-all="0">Hide all</button></span></div>
    <div class="lg-grid">${items}</div>
    <p class="lg-note">Solid dot = left side, ring = right side. Blue lines = left limbs, red lines = right limbs.</p>
    <div class="lg-head"><strong>On the video</strong></div>
    <div class="lg-grid">${layers}</div>`;
}
$('legend').addEventListener('change', (e) => {
  const t = e.target;
  if (t.dataset.marker !== undefined) view.markers[t.dataset.marker] = t.checked;
  if (t.dataset.layer !== undefined) view.layers[t.dataset.layer] = t.checked;
  saveView();
  if (state && state.frames) drawFrame(currentIndex());
});
$('legend').addEventListener('click', (e) => {
  const all = e.target.dataset?.all;
  if (all === undefined) return;
  for (const m of MARKERS) view.markers[m.k] = all === '1';
  saveView();
  buildLegend();
  if (state && state.frames) drawFrame(currentIndex());
});
buildLegend();

// ---------- inputs ----------
$('units').addEventListener('change', () => {
  const ft = $('units').value === 'ftin';
  for (const id of ['heightCm', 'barCm']) $(id).hidden = ft;
  for (const id of ['heightFt', 'heightIn', 'barFt', 'barIn']) $(id).hidden = !ft;
});

function readHeightCm() {
  if ($('units').value === 'cm') return Number($('heightCm').value);
  const ft = Number($('heightFt').value || 0), inch = Number($('heightIn').value || 0);
  return Math.round((ft * 12 + inch) * 2.54);
}

// Bar height is optional: returns null when left empty.
function readBarCm() {
  if ($('units').value === 'cm') {
    const v = $('barCm').value.trim();
    return v === '' ? null : Number(v);
  }
  const f = $('barFt').value.trim(), i = $('barIn').value.trim();
  if (f === '' && i === '') return null;
  return Math.round((Number(f || 0) * 12 + Number(i || 0)) * 2.54);
}

function setStatus(msg, isError = false) {
  $('status').textContent = msg;
  $('status').classList.toggle('error', isError);
}
function setHint(msg) { $('hint').textContent = msg; }

// ---------- pose model ----------
function getLandmarker(kind) {
  if (!landmarkers[kind]) {
    landmarkers[kind] = (async () => {
      const { PoseLandmarker, FilesetResolver } = await import(`${TV}/vision_bundle.mjs`);
      const fileset = await FilesetResolver.forVisionTasks(`${TV}/wasm`);
      // IMAGE mode: every crop is judged on its own, so a wrong lock-on
      // cannot carry over between frames. Our own tracker links frames.
      const opts = (delegate) => ({
        baseOptions: { modelAssetPath: MODELS[kind], delegate },
        runningMode: 'IMAGE',
        numPoses: 3,
        minPoseDetectionConfidence: 0.3,
        minPosePresenceConfidence: 0.3,
        minTrackingConfidence: 0.3,
      });
      try {
        return await PoseLandmarker.createFromOptions(fileset, opts('GPU'));
      } catch {
        return await PoseLandmarker.createFromOptions(fileset, opts('CPU'));
      }
    })();
    landmarkers[kind].catch(() => { delete landmarkers[kind]; });
  }
  return landmarkers[kind];
}

function seekTo(t) {
  return new Promise((resolve) => {
    if (Math.abs(video.currentTime - t) < 1e-4 && video.readyState >= 2) return resolve();
    const done = () => { video.removeEventListener('seeked', done); clearTimeout(timer); resolve(); };
    const timer = setTimeout(done, 3000);
    video.addEventListener('seeked', done);
    video.currentTime = t;
  });
}

function loadVideo(file) {
  return new Promise((resolve, reject) => {
    video.onloadeddata = () => resolve();
    video.onerror = () => reject(new Error('This video could not be opened in this browser. Try Safari or Chrome, or export it as MP4 (H.264).'));
    video.src = URL.createObjectURL(file);
    video.load();
  });
}

const breathe = () => new Promise((r) => setTimeout(r, 0));
const hipPx = (lms, VW, VH) => ({ x: ((lms[23].x + lms[24].x) / 2) * VW, y: ((lms[23].y + lms[24].y) / 2) * VH });

// Run the model on a square crop around (cx, cy) in video pixels, optionally
// rotated (degrees). Rotating helps when the body is sideways or upside down
// in the air: the model finds upright people best.
function detectCrop(lm, box, angle = 0) {
  const VW = video.videoWidth, VH = video.videoHeight;
  const side = box.side, sx = box.cx - side / 2, sy = box.cy - side / 2, k = CROP / side;
  cctx.save();
  cctx.fillStyle = '#000';
  cctx.fillRect(0, 0, CROP, CROP);
  const x0 = Math.max(0, sx), y0 = Math.max(0, sy), x1 = Math.min(VW, sx + side), y1 = Math.min(VH, sy + side);
  if (x1 <= x0 || y1 <= y0) { cctx.restore(); return []; }
  const a = (angle * Math.PI) / 180;
  if (angle) {
    cctx.translate(CROP / 2, CROP / 2);
    cctx.rotate(a);
    cctx.translate(-CROP / 2, -CROP / 2);
  }
  cctx.drawImage(video, x0, y0, x1 - x0, y1 - y0, (x0 - sx) * k, (y0 - sy) * k, (x1 - x0) * k, (y1 - y0) * k);
  cctx.restore();
  const r = lm.detect(cropCanvas);
  const cos = Math.cos(-a), sin = Math.sin(-a);
  return (r.landmarks || []).map((pose) => pose.map((p) => {
    // Undo the rotation about the crop centre, then map back to the frame.
    let u = p.x - 0.5, v = p.y - 0.5;
    if (angle) { const u2 = u * cos - v * sin, v2 = u * sin + v * cos; u = u2; v = v2; }
    return { x: (box.cx + u * side) / VW, y: (box.cy + v * side) / VH, visibility: p.visibility };
  }));
}

function detectFull(lm) {
  const r = lm.detect(video);
  return (r.landmarks || []).map((pose) => pose.map((p) => ({ x: p.x, y: p.y, visibility: p.visibility })));
}

// Find the person at a tapped point (video pixels): try crops of growing
// size around the tap, then the whole frame.
function personAt(lm, pt) {
  const VW = video.videoWidth, VH = video.videoHeight, aspect = VW / VH;
  const m = Math.min(VW, VH);
  // Upright crops of growing size, then the same turned sideways and upside
  // down (a jumper over the bar is inverted), then the whole frame.
  const tries = [];
  for (const ang of [0, -90, 90, 180]) for (const side of [m * 0.3, m * 0.55, m * 0.9]) tries.push(() => detectCrop(lm, { cx: pt.x, cy: pt.y, side }, ang));
  tries.push(() => detectFull(lm));
  for (const run of tries) {
    let best = null, bestD = Infinity;
    for (const p of run()) {
      if (!plausiblePose(p, aspect)) continue;
      const b = poseBox(p, aspect);
      const inside = Math.abs(pt.x / VW - b.cx) <= b.w / 2 + 0.03 && Math.abs(pt.y / VH - b.cy) <= b.h / 2 + 0.03;
      const hp = hipPx(p, VW, VH);
      const d = Math.hypot(hp.x - pt.x, hp.y - pt.y) - (inside ? 1e6 : 0);
      if (d < bestD) { bestD = d; best = p; }
    }
    if (best && (bestD < 0 || bestD < m * 0.15)) return best;
  }
  return null;
}

/**
 * Follow one person with a zoomed crop, frame by frame, forward and backward
 * from the frame where they were picked. With two picks (say one in the
 * run-up and one in the air), it also tracks back from the second pick and
 * joins the two runs at the last frame where they agree, which keeps the
 * tracker on the jumper through the hard part over the bar. Also keeps a
 * small grayscale copy of every frame to measure camera pans afterwards.
 */
async function followJumper(lm, times, seedOrSeeds, onProgress) {
  const VW = video.videoWidth, VH = video.videoHeight, aspect = VW / VH;
  const n = times.length;
  const seeds = (Array.isArray(seedOrSeeds) ? seedOrSeeds : [seedOrSeeds]).slice().sort((a, b) => a.frame - b.frame).slice(0, 2);
  const total = n + (seeds.length > 1 ? n - seeds[0].frame : 0);
  const grays = new Array(n).fill(null);
  const boxesSmall = new Array(n).fill(null);
  motionCanvas.width = MOTION_W;
  motionCanvas.height = Math.round((MOTION_W * VH) / VW);
  const mw = motionCanvas.width, mh = motionCanvas.height, ms = mw / VW;
  const minSide = Math.min(VW, VH) * 0.16, maxSide = Math.max(VW, VH) * 1.1;
  const sideFor = (lms) => Math.min(maxSide, Math.max(minSide, poseBox(lms, aspect).size * VH * 2.2));
  let done = 0;

  const grabGray = (k) => {
    mctx.drawImage(video, 0, 0, mw, mh);
    const d = mctx.getImageData(0, 0, mw, mh).data;
    const g = new Uint8Array(mw * mh);
    for (let i = 0, j = 0; i < g.length; i++, j += 4) g[i] = (d[j] * 77 + d[j + 1] * 150 + d[j + 2] * 29) >> 8;
    grays[k] = g;
  };

  // Recent body sizes (segment length, which does not change with pose).
  let sizes = [];
  const segSize = (lms) => {
    const P = toPx(lms, VW, VH);
    const torso = Math.hypot((P[11].x + P[12].x - P[23].x - P[24].x) / 2, (P[11].y + P[12].y - P[23].y - P[24].y) / 2);
    const legL = Math.hypot(P[23].x - P[25].x, P[23].y - P[25].y) + Math.hypot(P[25].x - P[27].x, P[25].y - P[27].y);
    const legR = Math.hypot(P[24].x - P[26].x, P[24].y - P[26].y) + Math.hypot(P[26].x - P[28].x, P[26].y - P[28].y);
    return torso + Math.max(legL, legR);
  };
  const refSize = () => {
    if (!sizes.length) return null;
    const s2 = sizes.slice(-8).sort((a, b) => a - b);
    return s2[s2.length >> 1];
  };
  // Cost of a candidate: how far from where the jumper should be, how
  // different in size (in body sizes), and how unsure the model is. A coach
  // or athlete standing nearby is usually further off and a different size.
  const cost = (c, guess) => {
    const ref = refSize() || segSize(c);
    const ratio = segSize(c) / ref;
    // Body size changes slowly; a pose far smaller or bigger is a different
    // person or a broken detection.
    if (ratio < 0.6 || ratio > 1.66) return Infinity;
    const hp = hipPx(c, VW, VH);
    const d = Math.hypot(hp.x - guess.cx, hp.y - guess.cy) / ref;
    const sz = Math.abs(Math.log(ratio));
    const vis = c.reduce((a, q) => a + (q.visibility ?? 1), 0) / c.length;
    return d + 1.5 * sz + (1 - vis);
  };
  const best = (cands, guess) => {
    let pick = null, bc = Infinity;
    for (const c of cands) {
      if (!plausiblePose(c, aspect)) continue;
      const v = cost(c, guess);
      if (v < bc) { bc = v; pick = c; }
    }
    return { pick, c: bc };
  };
  const GOOD = 0.5, OK = 1.3;
  // Try the upright crop first; if nothing fits well, also try the crop
  // turned a quarter each way and upside down (the body is sideways, then
  // inverted over the bar), then the whole frame. Keep the best overall.
  const findIn = (guess, allowFull, misses = 0) => {
    // The longer the jumper has been lost, the further away we accept them.
    const ok = Math.min(3, OK + 0.25 * misses);
    let r = best(detectCrop(lm, guess), guess);
    if (r.c > GOOD) {
      for (const ang of [-90, 90, 180]) {
        const r2 = best(detectCrop(lm, guess, ang), guess);
        if (r2.c < r.c) r = r2;
        if (r.c <= GOOD) break;
      }
    }
    if (r.c > ok && allowFull) {
      const r3 = best(detectFull(lm), guess);
      if (r3.c < r.c) r = r3;
    }
    return r.c <= ok ? r.pick : null;
  };

  async function run(from, to, step, startBox, first, out) {
    sizes = [];
    let box = { ...startBox };
    let vel = { x: 0, y: 0 };
    let last = null, misses = 0;
    for (let k = from; step > 0 ? k <= to : k >= to; k += step) {
      await seekTo(times[k]);
      grabGray(k);
      // Trust the recent speed only partly: at the plant the run-up speed
      // turns into upward speed, so a straight-line guess overshoots.
      const guess = { cx: box.cx + 0.5 * vel.x, cy: box.cy + 0.5 * vel.y, side: box.side };
      const pick = k === from && first ? first : findIn(guess, misses >= 1, misses);
      out[k] = pick;
      if (pick) {
        sizes.push(segSize(pick));
        if (sizes.length > 15) sizes.shift();
        const hp = hipPx(pick, VW, VH);
        if (last) vel = { x: 0.6 * (hp.x - last.x) + 0.4 * vel.x, y: 0.6 * (hp.y - last.y) + 0.4 * vel.y };
        last = hp;
        box = { cx: hp.x, cy: hp.y, side: 0.6 * box.side + 0.4 * sideFor(pick) };
        misses = 0;
      } else {
        // Lost: hold the last position (do not run off along the old speed)
        // and look a little wider each frame.
        misses++;
        vel = { x: 0, y: 0 };
        box = { cx: box.cx, cy: box.cy, side: Math.min(maxSide, box.side * 1.1) };
        lostBoxes[k] = { x: (guess.cx - guess.side / 2) * ms, y: (guess.cy - guess.side / 2) * ms, w: guess.side * ms, h: guess.side * ms };
      }
      done++;
      if (done % 3 === 0) onProgress(Math.min(1, done / total));
      if (done % 6 === 0) await breathe();
    }
  }

  const startOf = (sd) => (sd.pose ? { ...(() => { const h = hipPx(sd.pose, VW, VH); return { cx: h.x, cy: h.y }; })(), side: sideFor(sd.pose) } : sd.box);
  const lostBoxes = new Array(n).fill(null);
  const s0 = seeds[0];
  const A = new Array(n).fill(null);
  await run(s0.frame, n - 1, 1, startOf(s0), s0.pose, A);
  if (s0.frame > 0) await run(s0.frame, 0, -1, startOf(s0), s0.pose, A);
  let out = A;
  if (seeds.length > 1) {
    const s1 = seeds[1];
    const B = new Array(n).fill(null);
    await run(s1.frame, s0.frame, -1, startOf(s1), s1.pose, B);
    if (s1.frame < n - 1) await run(s1.frame, n - 1, 1, startOf(s1), s1.pose, B);
    // Join: the first run up to the last frame where both runs agree, then
    // the run from the second pick.
    let m = s0.frame;
    for (let k = s0.frame; k <= s1.frame; k++) {
      if (!A[k] || !B[k]) continue;
      const ha = hipPx(A[k], VW, VH), hb = hipPx(B[k], VW, VH);
      if (Math.hypot(ha.x - hb.x, ha.y - hb.y) < 0.25 * segSize(B[k])) m = k;
    }
    out = A.map((p, k) => (k <= m ? p : B[k]));
  }
  onProgress(1);
  for (let k = 0; k < n; k++) {
    if (out[k]) {
      const b = poseBox(out[k], aspect);
      boxesSmall[k] = { x: (b.cx - b.w / 2) * mw, y: (b.cy - b.h / 2) * mh, w: b.w * mw, h: b.h * mh };
    } else boxesSmall[k] = lostBoxes[k];
  }

  // Camera pans, measured with the jumper masked out.
  const cum = cameraPath(grays, mw, mh, boxesSmall);
  const cumNorm = cum.map((c) => ({ x: c.x / mw, y: c.y / mh }));
  return { frames: times.map((t, k) => ({ t, lms: out[k] })), cumNorm };
}

// ---------- step 2: pick the jumper and the bar ----------
function setupTimes() {
  const end = Math.min(video.duration, setup.start + MAX_SECONDS);
  const n = Math.max(1, Math.floor((end - setup.start) * setup.fps));
  // Seek to the middle of each frame, not its edge, so every seek lands on
  // a distinct frame (edge times can round to the same frame twice).
  return Array.from({ length: n }, (_, i) => setup.start + (i + 0.5) / setup.fps);
}

function setSetupHint(msg) { $('setupHint').textContent = msg; }
function setSetupStatus(msg, isError = false) {
  $('setupStatus').textContent = msg;
  $('setupStatus').classList.toggle('error', isError);
}

async function setupShow(i) {
  setup.index = i;
  $('setupScrub').value = String(i);
  await seekTo(setup.times[i]);
  drawSetup();
}

function drawSetup() {
  const W = setupCanvas.width, H = setupCanvas.height;
  const s = Math.max(1, W / Math.max(1, setupCanvas.clientWidth || W));
  sctx.drawImage(video, 0, 0, W, H);
  // The picked person, on the frame where they were picked.
  const j = setup.jumpers.find((x) => x.index === setup.index);
  if (j) {
    if (j.pose) {
      const P = toPx(j.pose, W, H);
      sctx.lineCap = 'round';
      sctx.lineWidth = 4 * s; sctx.strokeStyle = '#3ddc84';
      for (const [a, b] of [...BONES_L, ...BONES_R, ...BONES_C]) { sctx.beginPath(); sctx.moveTo(P[a].x, P[a].y); sctx.lineTo(P[b].x, P[b].y); sctx.stroke(); }
    }
    sctx.lineWidth = 3 * s; sctx.strokeStyle = '#3ddc84';
    sctx.beginPath(); sctx.arc(j.x * W, j.y * H, 14 * s, 0, Math.PI * 2); sctx.stroke();
  }
  // The bar, if marked on this frame.
  const bar = setup.bar;
  const pts = bar ? [bar.a, bar.b] : setup.barTaps;
  if (bar && bar.index === setup.index) {
    sctx.lineWidth = 6 * s; sctx.strokeStyle = 'rgba(0,0,0,0.75)';
    sctx.beginPath(); sctx.moveTo(bar.a.x * W, bar.a.y * H); sctx.lineTo(bar.b.x * W, bar.b.y * H); sctx.stroke();
    sctx.lineWidth = 3 * s; sctx.strokeStyle = '#ffffff';
    sctx.beginPath(); sctx.moveTo(bar.a.x * W, bar.a.y * H); sctx.lineTo(bar.b.x * W, bar.b.y * H); sctx.stroke();
  }
  if (!bar || bar.index === setup.index) {
    for (const p of pts) {
      sctx.beginPath(); sctx.arc(p.x * W, p.y * H, 9 * s, 0, Math.PI * 2);
      sctx.fillStyle = '#fff'; sctx.fill(); sctx.lineWidth = 3 * s; sctx.strokeStyle = '#000'; sctx.stroke();
    }
    if (bar && bar.ground) {
      const g = bar.ground;
      // Dashed line from the bar down to the floor point: the bar's height.
      sctx.setLineDash([8 * s, 6 * s]); sctx.lineWidth = 2 * s; sctx.strokeStyle = '#fff';
      const by = bar.a.y + ((bar.b.y - bar.a.y) * (g.x - bar.a.x)) / ((bar.b.x - bar.a.x) || 1e-9);
      sctx.beginPath(); sctx.moveTo(g.x * W, by * H); sctx.lineTo(g.x * W, g.y * H); sctx.stroke();
      sctx.setLineDash([]);
      sctx.fillStyle = '#fff'; sctx.strokeStyle = '#000'; sctx.lineWidth = 3 * s;
      sctx.fillRect(g.x * W - 8 * s, g.y * H - 8 * s, 16 * s, 16 * s);
      sctx.strokeRect(g.x * W - 8 * s, g.y * H - 8 * s, 16 * s, 16 * s);
    }
  }
}

function updateSetupUi() {
  const needBar = setup.inputs.barCm !== null;
  $('pickBarStep').hidden = !needBar;
  $('tapBar').hidden = !needBar;
  $('tapJumper').classList.toggle('active', setup.mode === 'jumper');
  $('tapBar').classList.toggle('active', setup.mode === 'bar');
  $('clearSetupBar').hidden = !setup.bar && !setup.barTaps.length;
  $('analyze').disabled = !setup.jumpers.length;
  const bits = [];
  const js = setup.jumpers;
  if (!js.length) bits.push('Tap the jumper on the video.');
  else {
    bits.push(js.every((x) => x.pose) ? `Jumper picked on ${js.length === 1 ? 'one frame' : 'two frames'} (green outline).` : 'Jumper marked; the model will lock on when tracking starts.');
    if (js.length === 1) bits.push('Optional: also tap the jumper on another frame, ideally in the air over the bar. It keeps tracking on the jumper over the bar.');
  }
  if (needBar) {
    if (setup.bar && setup.bar.ground) bits.push('Bar and floor marked. Drag the handles to adjust.');
    else if (setup.bar) bits.push('Now tap the floor directly under the bar (the bottom of a standard). This makes the bar the measuring stick.');
    else if (setup.barTaps.length === 1) bits.push('Now tap the other end of the bar.');
    else bits.push('Recommended: tap the two ends of the bar, then the floor under it. The bar height then measures your jump.');
  }
  setSetupHint(bits.join(' '));
}

async function openSetup(file, inputs) {
  setStatus('Opening video…');
  await loadVideo(file);
  if (inputs.startSeconds >= video.duration) throw new Error(`The video is only ${video.duration.toFixed(1)} s long.`);
  const q = QUALITY[inputs.quality];
  setup = {
    inputs, start: inputs.startSeconds, fps: q.fps, slow: inputs.slow,
    index: 0, jumpers: [], bar: null, barTaps: [], mode: 'jumper', modelReady: false,
  };
  setup.times = setupTimes();
  setupCanvas.width = video.videoWidth;
  setupCanvas.height = video.videoHeight;
  $('setupScrub').max = String(setup.times.length - 1);
  $('pick').hidden = false;
  $('viewer').hidden = true;
  $('report').innerHTML = '';
  setStatus('');
  updateSetupUi();
  // Start with a frame in the middle of the clip.
  await setupShow(Math.floor(setup.times.length * 0.6));
  $('pick').scrollIntoView({ behavior: 'smooth', block: 'start' });
  // Load the model while the user picks.
  setSetupStatus('Loading the pose model in the background…');
  getLandmarker(q.model).then(async (lm) => {
    if (!setup) return;
    setup.modelReady = true;
    setSetupStatus('Pose model ready.');
    for (const j of setup.jumpers) if (!j.pose) await lockOn(lm, j);
  }).catch(() => setSetupStatus('The pose model could not load. Check your connection and try again.', true));
}

async function lockOn(lm, j) {
  await seekTo(setup.times[j.index]);
  j.pose = personAt(lm, { x: j.x * video.videoWidth, y: j.y * video.videoHeight });
  if (setup.index !== j.index) await seekTo(setup.times[setup.index]);
  drawSetup();
  updateSetupUi();
  if (!j.pose) setSetupHint('Could not find a person there. Try a frame where the jumper is bigger or clearer, and tap their hips.');
}

function setupPoint(e) {
  const r = setupCanvas.getBoundingClientRect();
  return { x: (e.clientX - r.left) / r.width, y: (e.clientY - r.top) / r.height };
}
let setupDrag = null;
setupCanvas.addEventListener('pointerdown', async (e) => {
  if (!setup) return;
  const p = setupPoint(e);
  const W = setupCanvas.width, H = setupCanvas.height;
  const s = Math.max(1, W / Math.max(1, setupCanvas.clientWidth));
  // Drag an existing bar end or floor point on its frame.
  if (setup.bar && setup.bar.index === setup.index) {
    for (const end of ['a', 'b', 'ground']) {
      const q = setup.bar[end];
      if (!q) continue;
      if (Math.hypot((q.x - p.x) * W, (q.y - p.y) * H) < 28 * s) {
        setupDrag = end;
        setupCanvas.setPointerCapture(e.pointerId);
        e.preventDefault();
        return;
      }
    }
  }
  if (setup.mode === 'jumper') {
    // Up to two picks on different frames; a tap on a frame that already has
    // a pick replaces it, and a third pick replaces the nearer one in time.
    const j = { index: setup.index, x: p.x, y: p.y, pose: null };
    const js = setup.jumpers;
    const same = js.findIndex((x) => x.index === setup.index);
    if (same >= 0) js[same] = j;
    else if (js.length < 2) js.push(j);
    else {
      const near = Math.abs(js[0].index - j.index) <= Math.abs(js[1].index - j.index) ? 0 : 1;
      js[near] = j;
    }
    drawSetup();
    updateSetupUi();
    if (setup.modelReady) await lockOn(await getLandmarker(QUALITY[setup.inputs.quality].model), j);
    // After the first pick, the next taps go to the bar.
    if (js.length === 1 && setup.inputs.barCm !== null && !setup.bar) { setup.mode = 'bar'; updateSetupUi(); }
    return;
  }
  if (setup.mode === 'bar') {
    if (setup.bar && !setup.bar.ground && setup.bar.index === setup.index) {
      setup.bar.ground = p; // third tap: the floor under the bar
    } else {
      if (setup.bar) { setup.bar = null; setup.barTaps = []; }
      setup.barTaps.push(p);
      if (setup.barTaps.length === 2) {
        const [a, b] = setup.barTaps;
        setup.bar = { index: setup.index, a, b };
        setup.barTaps = [];
      }
    }
    drawSetup();
    updateSetupUi();
  }
});
setupCanvas.addEventListener('pointermove', (e) => {
  if (!setupDrag) return;
  const p = setupPoint(e);
  setup.bar[setupDrag] = { x: Math.min(1, Math.max(0, p.x)), y: Math.min(1, Math.max(0, p.y)) };
  drawSetup();
});
const endSetupDrag = () => { setupDrag = null; };
setupCanvas.addEventListener('pointerup', endSetupDrag);
setupCanvas.addEventListener('pointercancel', endSetupDrag);

$('setupScrub').addEventListener('input', () => setupShow(Number($('setupScrub').value)));
$('setupPrev').addEventListener('click', () => setupShow(Math.max(0, setup.index - 1)));
$('setupNext').addEventListener('click', () => setupShow(Math.min(setup.times.length - 1, setup.index + 1)));
$('tapJumper').addEventListener('click', () => { setup.mode = 'jumper'; updateSetupUi(); });
$('tapBar').addEventListener('click', () => { setup.mode = 'bar'; setup.barTaps = []; updateSetupUi(); drawSetup(); });
$('clearSetupBar').addEventListener('click', () => { setup.bar = null; setup.barTaps = []; setup.mode = 'bar'; updateSetupUi(); drawSetup(); });
$('backToForm').addEventListener('click', () => { $('pick').hidden = true; setup = null; $('setup').scrollIntoView({ behavior: 'smooth' }); });

$('analyze').addEventListener('click', async () => {
  if (!setup || !setup.jumpers.length) return;
  const btn = $('analyze');
  btn.disabled = true;
  try {
    const q = QUALITY[setup.inputs.quality];
    setSetupStatus('Waiting for the pose model…');
    const lm = await getLandmarker(q.model);
    for (const j of setup.jumpers) if (!j.pose) await lockOn(lm, j);
    const VW = video.videoWidth, VH = video.videoHeight;
    const seeds = setup.jumpers.map((j) => ({ frame: j.index, pose: j.pose, box: { cx: j.x * VW, cy: j.y * VH, side: Math.min(VW, VH) * 0.45 } }));
    $('progress').hidden = false;
    const { frames, cumNorm } = await followJumper(lm, setup.times, seeds, (p) => {
      $('progress').value = p;
      setSetupStatus(`Following the jumper up close… ${Math.round(p * 100)}%`);
    });
    $('progress').hidden = true;
    setSetupStatus('');
    state = {
      start: setup.start, fps: setup.fps, slow: setup.slow, times: setup.times,
      inputs: setup.inputs, cum: cumNorm, rawFollowed: frames,
      corrections: {}, overrides: {}, frames: null, analysis: null, images: {},
      barMarks: setup.bar ? { index: setup.bar.index, a: setup.bar.a, b: setup.bar.b, ground: setup.bar.ground } : null,
      barSource: setup.bar ? 'marked by you' : null,
      mode: 'view', barTaps: [], flipped: false, flipChecked: false,
    };
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    state.cleaned = cleanFrames(frames);
    runAnalysis();
    $('pick').hidden = true;
    await present();
  } catch (err) {
    console.error(err);
    $('progress').hidden = true;
    setSetupStatus(err.message || String(err), true);
  } finally {
    btn.disabled = false;
  }
});

// ---------- camera offsets and the bar ----------
// Analysis works in ground coordinates (camera pans removed). Drawing works
// in the coordinates of the frame on screen.
const camAt = (i) => (state.cum && state.cum[i]) || { x: 0, y: 0 };

// Bar ends in ground coordinates (normalized), or null.
function barGround() {
  const m = state.barMarks;
  if (!m) return null;
  const c = camAt(m.index);
  const g = (p) => ({ x: p.x - c.x, y: p.y - c.y });
  return { a: g(m.a), b: g(m.b), ground: m.ground ? g(m.ground) : undefined };
}
// Bar ends on frame i, in pixels.
function barOnFrame(i) {
  const b = barGround();
  if (!b) return null;
  const c = camAt(i), W = canvas.width, H = canvas.height;
  const f = (p) => ({ x: (p.x + c.x) * W, y: (p.y + c.y) * H });
  return { a: f(b.a), b: f(b.b), ground: b.ground ? f(b.ground) : null };
}
// Re-express the bar marks on frame i (so they can be dragged there).
function barMarksOnFrame(i) {
  const b = barGround();
  if (!b) return null;
  const c = camAt(i);
  const f = (p) => ({ x: p.x + c.x, y: p.y + c.y });
  return { index: i, a: f(b.a), b: f(b.b), ground: b.ground ? f(b.ground) : undefined };
}

// ---------- drawing ----------
const dispScale = () => Math.max(1, canvas.width / Math.max(1, canvas.clientWidth || canvas.width));

function pill(text, x, y, s, color = '#ffffff') {
  ctx.font = `600 ${Math.round(13 * s)}px system-ui, sans-serif`;
  const w = ctx.measureText(text).width + 10 * s, h = 20 * s;
  const bx = Math.min(Math.max(2, x), canvas.width - w - 2), by = Math.min(Math.max(2, y - h / 2), canvas.height - h - 2);
  ctx.fillStyle = 'rgba(0,0,0,0.72)';
  ctx.fillRect(bx, by, w, h);
  ctx.fillStyle = color;
  ctx.textBaseline = 'middle';
  ctx.fillText(text, bx + 5 * s, by + h / 2);
}

function drawMarker(m, x, y, r, s, c = ctx) {
  if (m.side === 'R') {
    // Ring: bigger than the dots, so a left dot underneath stays visible.
    const rr = r * 1.6;
    c.lineWidth = r * 0.75 + 2 * s; c.strokeStyle = '#000';
    c.beginPath(); c.arc(x, y, rr, 0, Math.PI * 2); c.stroke();
    c.lineWidth = r * 0.75; c.strokeStyle = m.color;
    c.beginPath(); c.arc(x, y, rr, 0, Math.PI * 2); c.stroke();
  } else {
    c.beginPath(); c.arc(x, y, r, 0, Math.PI * 2);
    c.fillStyle = m.color; c.fill();
    c.lineWidth = 1.5 * s; c.strokeStyle = '#000'; c.stroke();
  }
}

function drawBarLayer(i, s) {
  const b = barOnFrame(i);
  if (!b) return;
  ctx.lineCap = 'round';
  ctx.lineWidth = 6 * s; ctx.strokeStyle = 'rgba(0,0,0,0.75)';
  ctx.beginPath(); ctx.moveTo(b.a.x, b.a.y); ctx.lineTo(b.b.x, b.b.y); ctx.stroke();
  ctx.lineWidth = 3 * s; ctx.strokeStyle = '#ffffff';
  ctx.beginPath(); ctx.moveTo(b.a.x, b.a.y); ctx.lineTo(b.b.x, b.b.y); ctx.stroke();
}

// Floor point used for the bar's scale, on frame i, in pixels.
function groundOnFrame(i) {
  const b = barOnFrame(i);
  if (b && b.ground) return b.ground;
  const g = state.analysis?.info.groundPx;
  if (!g) return null;
  const c = camAt(i);
  return { x: g.x + c.x * canvas.width, y: g.y + c.y * canvas.height };
}

function drawHandles(i, s) {
  const W = canvas.width, H = canvas.height;
  const circle = (x, y) => {
    ctx.beginPath(); ctx.arc(x, y, 9 * s, 0, Math.PI * 2);
    ctx.fillStyle = '#ffffff'; ctx.fill(); ctx.lineWidth = 3 * s; ctx.strokeStyle = '#000'; ctx.stroke();
  };
  const b = barOnFrame(i);
  if (b) {
    circle(b.a.x, b.a.y); circle(b.b.x, b.b.y);
    const g = groundOnFrame(i);
    if (g) {
      ctx.fillStyle = '#ffffff'; ctx.strokeStyle = '#000'; ctx.lineWidth = 3 * s;
      ctx.fillRect(g.x - 8 * s, g.y - 8 * s, 16 * s, 16 * s);
      ctx.strokeRect(g.x - 8 * s, g.y - 8 * s, 16 * s, 16 * s);
    }
  } else {
    state.barTaps.forEach((p) => circle(p.x * W, p.y * H));
  }
}

function drawLegendOnVideo(s) {
  const shown = MARKERS.filter((m) => view.markers[m.k]);
  if (!shown.length) return;
  const rowH = 18 * s, pad = 8 * s, w = 150 * s, h = pad * 2 + rowH * shown.length;
  const x = canvas.width - w - 8 * s, y = canvas.height - h - 8 * s;
  ctx.fillStyle = 'rgba(0,0,0,0.65)';
  ctx.fillRect(x, y, w, h);
  ctx.font = `500 ${Math.round(12 * s)}px system-ui, sans-serif`;
  ctx.textBaseline = 'middle';
  shown.forEach((m, j) => {
    const cy = y + pad + rowH * (j + 0.5);
    drawMarker(m, x + pad + 6 * s, cy, 4.5 * s, s);
    ctx.fillStyle = '#fff';
    ctx.fillText(m.name, x + pad + 20 * s, cy);
  });
}

function drawFrame(i) {
  const W = canvas.width, H = canvas.height;
  ctx.drawImage(video, 0, 0, W, H);
  if (!state || !state.frames) return;
  const s = dispScale();
  const { frames, analysis } = state;
  const f = frames[i];
  const kf = analysis ? analysis.keyFrames : null;
  const L = view.layers;

  if (state.mode === 'bar') { drawBarLayer(i, s); drawHandles(i, s); } else if (L.bar) drawBarLayer(i, s);

  if (f && f.lms) {
    const P = toPx(f.lms, W, H);
    const bodyPx = analysis?.info.bodyPx || H * 0.3;
    const r = Math.min(8 * s, Math.max(3 * s, bodyPx * 0.02));

    if (L.skeleton) {
      ctx.lineCap = 'round';
      ctx.lineWidth = Math.max(2 * s, r * 0.55);
      const line = (pairs, color) => {
        ctx.strokeStyle = color;
        for (const [a, b] of pairs) { ctx.beginPath(); ctx.moveTo(P[a].x, P[a].y); ctx.lineTo(P[b].x, P[b].y); ctx.stroke(); }
      };
      line(BONES_C, 'rgba(255,255,255,0.75)');
      line(BONES_L, 'rgba(110,200,255,0.9)');
      line(BONES_R, 'rgba(255,120,120,0.9)');
    }

    if (L.lean && kf && i <= kf.takeoff && state.mode === 'view') {
      const A = P[LM.lAnkle].y >= P[LM.rAnkle].y ? P[LM.lAnkle] : P[LM.rAnkle];
      const S = mid(P[LM.lShoulder], P[LM.rShoulder]);
      const len = Math.hypot(S.x - A.x, S.y - A.y);
      ctx.setLineDash([6 * s, 6 * s]);
      ctx.lineWidth = 2 * s; ctx.strokeStyle = 'rgba(255,255,255,0.7)';
      ctx.beginPath(); ctx.moveTo(A.x, A.y); ctx.lineTo(A.x, A.y - len); ctx.stroke();
      ctx.setLineDash([]);
      ctx.lineWidth = 3 * s; ctx.strokeStyle = '#3fd0ff';
      ctx.beginPath(); ctx.moveTo(A.x, A.y); ctx.lineTo(S.x, S.y); ctx.stroke();
    }

    // Clearance: a short line from each body part to the bar at the moment
    // that part was highest; green above the bar, red below.
    const cl = analysis?.info.clearance;
    const b = barOnFrame(i);
    if (L.clearance && cl && b && kf && i > kf.takeoff && i <= kf.landing + 2 && state.mode === 'view') {
      const near = Math.max(1, Math.round(state.fps / 15));
      const yAt = (x) => (Math.abs(b.b.x - b.a.x) < 1e-9 ? b.a.y : b.a.y + ((b.b.y - b.a.y) * (x - b.a.x)) / (b.b.x - b.a.x));
      for (const part of cl.parts) {
        if (Math.abs(i - part.frame) > near) continue;
        const pts = PART_POINTS[part.key].map((k) => P[k]);
        const x = pts.reduce((a, p) => a + p.x, 0) / pts.length, y = pts.reduce((a, p) => a + p.y, 0) / pts.length;
        ctx.lineWidth = 3 * s; ctx.strokeStyle = part.cm >= 0 ? '#3ddc84' : '#ff4d4d';
        ctx.beginPath(); ctx.moveTo(x, y); ctx.lineTo(x, yAt(x)); ctx.stroke();
      }
    }

    const fixing = state.mode === 'fix';
    const corr = state.corrections[i] || {};
    // Left dots first, then right rings on top, so overlapping pairs both show.
    const order = [...MARKERS.filter((m) => m.side !== 'R'), ...MARKERS.filter((m) => m.side === 'R')];
    for (const m of order) {
      if (!view.markers[m.k] && !fixing) continue;
      drawMarker(corr[m.k] ? { ...m, color: '#5dff8f' } : m, P[m.k].x, P[m.k].y, fixing ? r * 1.4 : r, s);
    }
  }

  if (L.phase && f) {
    const phase = kf ? phaseAt(i, kf) : '';
    const realT = (f.t - state.start) / state.slow;
    pill(`${phase}${phase ? '  ·  ' : ''}${realT.toFixed(2)} s${f.corrected ? '  ·  fixed' : ''}`, 8 * s, 16 * s, s);
  }
  if (L.legend) drawLegendOnVideo(s);
  updateReadout(i);
}

function updateReadout(i) {
  const f = state.frames[i];
  const a = state.analysis;
  const rows = [];
  const kf = a?.keyFrames;
  rows.push(['Phase', kf ? phaseAt(i, kf) : '–']);
  rows.push(['Time', f ? `${((f.t - state.start) / state.slow).toFixed(2)} s` : '–']);
  if (f && f.lms) {
    const P = toPx(f.lms, canvas.width, canvas.height);
    const ang = (a1, b, c) => { const v = angleAt(P[a1], P[b], P[c]); return Number.isFinite(v) ? `${Math.round(v)}°` : '–'; };
    rows.push([`${swatchSvg(MK[25], 14)} Left knee`, ang(23, 25, 27)]);
    rows.push([`${swatchSvg(MK[26], 14)} Right knee`, ang(24, 26, 28)]);
    rows.push([`${swatchSvg(MK[23], 14)} Left hip`, ang(11, 23, 25)]);
    rows.push([`${swatchSvg(MK[24], 14)} Right hip`, ang(12, 24, 26)]);
    if (kf && i <= kf.takeoff) {
      const A = P[27].y >= P[28].y ? P[27] : P[28];
      const S = mid(P[11], P[12]);
      rows.push(['Lean', `${Math.round((Math.atan2(Math.abs(S.x - A.x), A.y - S.y) * 180) / Math.PI)}°`]);
    }
    if (a) {
      const hp = mid(P[23], P[24]);
      const b = barOnFrame(i);
      let cm;
      if (b && a.info.barScale && kf && i > kf.takeoff) {
        // In the air: measured from the bar, which is at the jumper's depth.
        const by = Math.abs(b.b.x - b.a.x) < 1e-9 ? b.a.y : b.a.y + ((b.b.y - b.a.y) * (hp.x - b.a.x)) / (b.b.x - b.a.x);
        cm = a.info.barCm + (by - hp.y) * a.info.barScale;
      } else {
        const groundY = a.info.groundY + camAt(i).y * canvas.height;
        cm = (groundY - hp.y) * a.info.cmPerPx;
      }
      rows.push(['Hip height', fmtLen(cm, state.inputs.units)]);
    }
  }
  $('readout').innerHTML = rows.map(([k, v]) => `<div class="ro-row"><span>${k}</span><strong>${v}</strong></div>`).join('');
}

function frameIndexAt(t) {
  const i = Math.floor((t - state.start) * state.fps);
  return Math.max(0, Math.min(state.frames.length - 1, i));
}

async function showFrame(i) {
  await seekTo(state.frames[i].t);
  drawFrame(i);
  $('scrub').value = String(i);
}
const currentIndex = () => Number($('scrub').value);

// ---------- playback & recording ----------
let playing = false;
function playThrough(onDone) {
  const endT = state.frames[state.frames.length - 1].t;
  playing = true;
  $('play').textContent = 'Pause';
  video.play();
  const tick = () => {
    if (!playing) return;
    const i = frameIndexAt(video.currentTime);
    drawFrame(i);
    $('scrub').value = String(i);
    if (video.currentTime >= endT || video.ended) { stopPlay(); onDone && onDone(); return; }
    requestAnimationFrame(tick);
  };
  requestAnimationFrame(tick);
}
function stopPlay() {
  playing = false;
  video.pause();
  $('play').textContent = 'Play';
}

$('play').addEventListener('click', async () => {
  if (playing) return stopPlay();
  setMode('view');
  let i = currentIndex();
  if (i >= state.frames.length - 1) i = 0;
  await seekTo(state.frames[i].t);
  playThrough();
});
$('scrub').addEventListener('input', () => { stopPlay(); showFrame(currentIndex()); });
$('prevFrame').addEventListener('click', () => { stopPlay(); showFrame(Math.max(0, currentIndex() - 1)); });
$('nextFrame').addEventListener('click', () => { stopPlay(); showFrame(Math.min(state.frames.length - 1, currentIndex() + 1)); });

function download(blob, name) {
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = name;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(a.href), 5000);
}

$('dlVideo').addEventListener('click', async () => {
  const types = ['video/mp4;codecs=avc1', 'video/mp4', 'video/webm;codecs=vp9', 'video/webm'];
  const mime = window.MediaRecorder && types.find((m) => MediaRecorder.isTypeSupported(m));
  if (!mime || !canvas.captureStream) {
    setStatus('This browser cannot record video. Try Chrome, Edge or a recent Safari.', true);
    return;
  }
  setMode('view');
  const btn = $('dlVideo');
  btn.disabled = true; btn.textContent = 'Recording… (plays once)';
  stopPlay();
  await showFrame(0);
  const rec = new MediaRecorder(canvas.captureStream(30), { mimeType: mime, videoBitsPerSecond: 6e6 });
  const chunks = [];
  rec.ondataavailable = (e) => e.data.size && chunks.push(e.data);
  const stopped = new Promise((r) => (rec.onstop = r));
  rec.start();
  await new Promise((r) => playThrough(r));
  rec.stop();
  await stopped;
  download(new Blob(chunks, { type: mime }), `high-jump-annotated.${mime.includes('mp4') ? 'mp4' : 'webm'}`);
  btn.disabled = false; btn.textContent = 'Download annotated video';
});

$('dlJson').addEventListener('click', () => {
  const { analysis, inputs, tracking } = state;
  const out = { inputs, tracking, bar: barGround(), overall: analysis.overall, rating: analysis.rating, phases: analysis.phases, metrics: analysis.metrics, info: analysis.info, keyFrames: analysis.keyFrames, coaching: coachJump(analysis), sources: analysis.sources };
  download(new Blob([JSON.stringify(out, null, 2)], { type: 'application/json' }), 'metrics.json');
});

$('dlReport').addEventListener('click', async () => {
  const css = await fetch('style.css').then((r) => r.text()).catch(() => '');
  const html = `<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>High jump report</title><style>${css}</style></head><body><main style="padding-top:24px">${reportHtml()}</main></body></html>`;
  download(new Blob([html], { type: 'text/html' }), 'high-jump-report.html');
});

// ---------- tools ----------
function setMode(mode) {
  if (!state) return;
  state.mode = mode;
  $('pickJumper').classList.toggle('active', mode === 'pick');
  $('markBar').classList.toggle('active', mode === 'bar');
  $('fixPoints').classList.toggle('active', mode === 'fix');
  $('pickJumper').textContent = mode === 'pick' ? 'Cancel' : 'Wrong person? Pick the jumper';
  $('markBar').textContent = mode === 'bar' ? 'Done with the bar' : (state.barMarks ? 'Adjust the bar' : 'Mark the bar');
  $('fixPoints').textContent = mode === 'fix' ? 'Done fixing' : 'Fix points';
  $('keyTools').hidden = mode !== 'fix';
  canvas.classList.toggle('tapping', mode !== 'view');
  if (mode === 'pick') setHint('Tap the jumper on the video. Use a frame where they are clearly visible.');
  else if (mode === 'bar') setHint(state.barMarks
    ? 'Drag the round handles onto the two ends of the bar. Optional: drag the square onto the floor directly under the bar.'
    : 'Find a frame where both bar ends show, then tap the left end and the right end.');
  else if (mode === 'fix') setHint('Drag any point to where the joint really is (fixed points turn green). You can also set the current frame as a key moment.');
  else setHint('');
  updateToolState();
  drawFrame(currentIndex());
}

function updateToolState() {
  const n = Object.keys(state.corrections).filter((k) => Object.keys(state.corrections[k]).length).length;
  const k = Object.keys(state.overrides).length;
  $('applyFixes').hidden = n === 0 && k === 0;
  $('applyFixes').textContent = `Re-analyze with fixes${n ? ` (${n} frame${n === 1 ? '' : 's'})` : ''}`;
  $('clearBar').hidden = !state.barMarks;
}

$('pickJumper').addEventListener('click', () => {
  if (!state) return;
  if (!state.times) { setHint('Picking the jumper needs a real clip.'); return; }
  stopPlay();
  setMode(state.mode === 'pick' ? 'view' : 'pick');
});

$('markBar').addEventListener('click', async () => {
  if (!state) return;
  if (state.mode === 'bar') {
    setMode('view');
    if (state.barEdited) { state.barEdited = false; await reanalyze('Bar updated.'); }
    return;
  }
  if (state.inputs.barCm === null) {
    setHint('Enter the bar height above and start again, then mark the bar.');
    return;
  }
  stopPlay();
  state.barTaps = [];
  setMode('bar');
});
$('clearBar').addEventListener('click', async () => {
  state.barMarks = null;
  state.barSource = null;
  setMode('view');
  await reanalyze('Bar removed.');
});
$('fixPoints').addEventListener('click', () => {
  if (!state) return;
  stopPlay();
  setMode(state.mode === 'fix' ? 'view' : 'fix');
});
$('applyFixes').addEventListener('click', () => reanalyze('Re-analyzed with your fixes.'));
for (const key of ['plant', 'takeoff', 'peak', 'landing']) {
  $(`set-${key}`).addEventListener('click', () => {
    state.overrides[key] = currentIndex();
    updateToolState();
    setHint(`${key[0].toUpperCase() + key.slice(1)} set to this frame. Press Re-analyze with fixes when you are done.`);
  });
}
$('resetKeys').addEventListener('click', () => {
  state.overrides = {};
  updateToolState();
  setHint('Key moments are back to automatic. Press Re-analyze with fixes to apply.');
});

function canvasPoint(e) {
  const r = canvas.getBoundingClientRect();
  return { x: ((e.clientX - r.left) / r.width) * canvas.width, y: ((e.clientY - r.top) / r.height) * canvas.height };
}

let drag = null; // {type: 'point' | 'bar', ...}
canvas.addEventListener('pointerdown', async (e) => {
  if (!state || state.mode === 'view') return;
  const p = canvasPoint(e);
  const W = canvas.width, H = canvas.height, s = dispScale();
  const norm = { x: p.x / W, y: p.y / H };
  const i = currentIndex();

  if (state.mode === 'pick') {
    setMode('view');
    await retrack(i, norm);
    return;
  }

  if (state.mode === 'bar') {
    if (!state.barMarks) {
      state.barTaps.push(norm);
      if (state.barTaps.length === 1) setHint('Now tap the right end of the bar.');
      if (state.barTaps.length === 2) {
        const [a, b] = state.barTaps;
        state.barMarks = { index: i, a, b };
        state.barSource = 'marked by you';
        state.barEdited = true;
        setMode('bar');
      }
      drawFrame(i);
      return;
    }
    const b = barOnFrame(i);
    const g = groundOnFrame(i);
    const handles = [['a', b.a], ['b', b.b], ['ground', g]];
    let best = null, bestD = 28 * s;
    for (const [name, h] of handles) {
      if (!h) continue;
      const d = Math.hypot(h.x - p.x, h.y - p.y);
      if (d < bestD) { bestD = d; best = name; }
    }
    if (best) {
      // Edit the bar as it appears on this frame.
      const marks = barMarksOnFrame(i);
      if (best === 'ground' && !marks.ground) marks.ground = { x: g.x / W, y: g.y / H };
      state.barMarks = marks;
      drag = { type: 'bar', handle: best };
      canvas.setPointerCapture(e.pointerId);
      e.preventDefault();
    }
    return;
  }

  // Fix mode: grab the nearest point.
  const f = state.frames[i];
  if (!f || !f.lms) return;
  const P = toPx(f.lms, W, H);
  let best = null, bestD = 30 * s;
  for (const m of MARKERS) {
    const d = Math.hypot(P[m.k].x - p.x, P[m.k].y - p.y);
    if (d < bestD) { best = m.k; bestD = d; }
  }
  if (best === null) return;
  drag = { type: 'point', i, k: best };
  canvas.setPointerCapture(e.pointerId);
  e.preventDefault();
});
canvas.addEventListener('pointermove', (e) => {
  if (!drag) return;
  const p = canvasPoint(e);
  const nx = Math.min(1, Math.max(0, p.x / canvas.width)), ny = Math.min(1, Math.max(0, p.y / canvas.height));
  if (drag.type === 'bar') {
    state.barMarks = { ...state.barMarks, [drag.handle]: { x: nx, y: ny } };
    state.barSource = 'marked by you';
    state.barEdited = true;
    drawFrame(currentIndex());
    return;
  }
  (state.corrections[drag.i] ||= {})[drag.k] = { x: nx, y: ny };
  state.frames[drag.i].lms[drag.k] = { x: nx, y: ny, visibility: 1 };
  drawFrame(drag.i);
});
const endDrag = () => { if (drag) { drag = null; updateToolState(); } };
canvas.addEventListener('pointerup', endDrag);
canvas.addEventListener('pointercancel', endDrag);

// ---------- labeled frames for training ----------
$('exportData').addEventListener('click', async () => {
  stopPlay();
  const btn = $('exportData');
  btn.disabled = true; btn.textContent = 'Exporting…';
  const keep = new Set([
    ...Object.keys(state.corrections).filter((k) => Object.keys(state.corrections[k]).length).map(Number),
    ...Object.values(state.analysis.keyFrames),
  ]);
  const items = [];
  for (const i of [...keep].sort((a, b) => a - b)) {
    const f = state.frames[i];
    if (!f || !f.lms) continue;
    await seekTo(f.t);
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height); // the plain frame
    items.push({
      frame: i,
      videoTime: f.t,
      source: f.corrected ? 'corrected by hand' : 'model',
      image: canvas.toDataURL('image/jpeg', 0.9),
      keypoints: f.lms.map((p) => [Number(p.x.toFixed(5)), Number(p.y.toFixed(5)), Number((p.visibility ?? 1).toFixed(3))]),
    });
  }
  const data = {
    format: 'mediapipe-pose-33',
    note: 'keypoints are [x, y, visibility], x and y as a fraction of image width and height, in MediaPipe Pose landmark order.',
    video: state.inputs.file,
    imageSize: [canvas.width, canvas.height],
    athlete: { heightCm: state.inputs.heightCm, gender: state.inputs.gender, takeoffFoot: state.inputs.takeoffFoot },
    barCm: state.inputs.barCm,
    bar: state.barMarks,
    keyFrames: state.analysis.keyFrames,
    result: state.inputs.result,
    frames: items,
  };
  download(new Blob([JSON.stringify(data)], { type: 'application/json' }), 'high-jump-labeled-frames.json');
  await showFrame(currentIndex());
  btn.disabled = false; btn.textContent = 'Export labeled frames';
});

// ---------- report ----------
const color = (s) => (s >= 85 ? 'var(--good)' : s >= 60 ? 'var(--mid)' : 'var(--bad)');
const fmt = (v, d) => (typeof v === 'number' ? (Number.isFinite(v) ? v.toFixed(d) : 'n/a') : v);
const ftIn = (cm) => {
  const total = cm / 2.54;
  let ft = Math.floor(total / 12), inch = Math.round(total - ft * 12);
  if (inch === 12) { ft += 1; inch = 0; }
  return `${ft} ft ${inch} in`;
};
const inches = (cm) => `${(cm / 2.54).toFixed(1)} in`;
// Short lengths in inches, longer ones in ft/in, when ft/in is picked.
const fmtLen = (cm, units) => (units === 'ftin' ? (Math.abs(cm) >= 100 ? ftIn(cm) : inches(cm)) : `${Math.round(cm)} cm`);
const showHeight = (cm, units) => (units === 'ftin' ? `${ftIn(cm)} (${Math.round(cm)} cm)` : `${Math.round(cm)} cm`);
const plural = (n, word) => `${n} ${word}${n === 1 ? '' : 's'}`;
const signedLen = (cm, units) => `${cm >= 0 ? '+' : '−'}${fmtLen(Math.abs(cm), units)}`;
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));

function clearanceHtml(a, inputs) {
  const i = a.info, u = inputs.units;
  const approx = i.verticalFrom !== 'bar';
  const vertical = `<div class="stats">
      <div><div class="l">Vertical jump (hip rise)${approx ? ', approximate' : ''}</div><div class="n">${fmtLen(i.verticalJump, u)}</div></div>
      <div><div class="l">Peak hip height</div><div class="n">${fmtLen(i.peakHipHeight, u)}</div></div>
      ${i.barCm !== null ? `<div><div class="l">Bar</div><div class="n">${u === 'ftin' ? ftIn(i.barCm) : `${i.barCm} cm`}</div></div>
      <div><div class="l">Hips vs bar</div><div class="n">${signedLen(i.peakHipHeight - i.barCm, u)}</div></div>` : ''}
    </div>`;
  const depthNote = approx && i.depthChange > 0.15
    ? '<p class="note">You moved a lot toward or away from the camera while in the air, so heights read from the picture are approximate. Mark the bar (both ends and the floor under it) for an accurate vertical.</p>'
    : '';
  const barNote = state.barMarks
    ? 'Uses the bar you marked, taken to be exactly the height you entered, which sets the scale for every measurement.'
    : 'The bar is not marked, so heights use your height for scale and the bar is compared by height only. Mark the bar for a more accurate result.';
  if (!i.clearance) {
    return `<section class="card"><h2>Vertical jump and bar clearance</h2>${vertical}
      ${depthNote}<p class="note">Enter the bar height to check whether each body part cleared the bar.</p></section>`;
  }
  const cl = i.clearance;
  const verdictText = {
    cleared: 'Cleared the bar (estimate)',
    close: 'Very close: may have brushed the bar',
    knocked: `Likely hit the bar with the ${cl.lowestLabel.toLowerCase()}`,
  }[cl.verdict];
  const verdictColor = cl.verdict === 'cleared' ? 'var(--good)' : cl.verdict === 'close' ? 'var(--mid)' : 'var(--bad)';
  const said = { cleared: 'cleared', knocked: 'knocked the bar' }[inputs.result];
  const agrees = said && ((inputs.result === 'cleared') === (cl.verdict !== 'knocked'));
  const rows = cl.parts.map((p) => `<tr><td>${p.label}</td><td class="num" style="color:${p.cm >= 0 ? 'var(--good)' : 'var(--bad)'}">${signedLen(p.cm, u)}</td></tr>`).join('');
  return `<section class="card">
    <h2>Vertical jump and bar clearance</h2>
    ${vertical}
    ${depthNote}
    <p class="verdict" style="color:${verdictColor}">${verdictText}</p>
    ${said ? `<p class="note">You said the jump ${said}. ${agrees ? 'That matches the estimate.' : 'The estimate disagrees, so the tracking or the bar line may be off. Check the bar line, or fix points on the frames over the bar.'}</p>` : ''}
    <div class="table-wrap"><table>
      <thead><tr><th>Body part</th><th>Best height over the bar</th></tr></thead>
      <tbody>${rows}</tbody></table></div>
    <p class="note">Each body part has to be above the bar at some point in the air, so this compares each part's highest point with the bar. Joint points sit inside the body, so a little is taken off for the back of the body. ${barNote}</p>
  </section>`;
}

function stepsHtml(i, u) {
  const steps = i.steps || [];
  if (steps.length < 2) return '';
  const rows = steps.slice(1).map((s2, j) => {
    const name = j === steps.length - 2 ? 'Last step (into the plant)' : j === steps.length - 3 ? 'Second-to-last step' : 'Earlier step';
    return `<tr><td>${name}</td><td>${s2.foot} foot</td><td class="num">${s2.length !== null ? fmtLen(s2.length, u) : 'n/a'}</td><td class="num">${s2.duration !== null ? s2.duration.toFixed(2) + ' s' : 'n/a'}</td></tr>`;
  }).join('');
  return `<h3>Last steps</h3>
    <div class="table-wrap"><table>
      <thead><tr><th>Step</th><th>Lands on</th><th>Length</th><th>Time</th></tr></thead>
      <tbody>${rows}</tbody></table></div>
    <p class="note">Step lengths are measured in the picture, so steps toward or away from the camera look shorter.</p>`;
}

function coachingHtml(a) {
  const c = coachJump(a);
  const fixes = c.fixes.slice(0, 4);
  return `<section class="card coach">
    <h2>Coaching</h2>
    ${c.strengths.length ? `<h3>What you did well</h3><ul class="good">${c.strengths.map((s) => `<li>${esc(s)}</li>`).join('')}</ul>` : ''}
    <h3>Fix first</h3>
    ${fixes.length ? `<ol>${fixes.map((f) => `<li><strong>${esc(f.label)}.</strong> ${esc(f.text)}</li>`).join('')}</ol>` : '<p>Nothing stands out: every measure is in your target range.</p>'}
    ${c.weakestPhase && c.drills.length ? `<h3>Drills for your ${esc(c.weakestPhase.label.toLowerCase())}</h3><ul>${c.drills.map((d) => `<li>${esc(d)}</li>`).join('')}</ul>` : ''}
    <details><summary><strong>Approach and technique fundamentals</strong></summary>
      <ul>${c.fundamentals.map((f) => `<li><strong>${esc(f.title)}.</strong> ${esc(f.text)}</li>`).join('')}</ul>
    </details>
    <details><summary><strong>Practice checklist</strong></summary>
      <ul>${c.checklist.map((x) => `<li>${esc(x)}</li>`).join('')}</ul>
    </details>
  </section>`;
}

function reportHtml() {
  const { analysis: a, inputs, images, tracking } = state;
  const u = inputs.units;
  const phases = a.phases.map((p) => `
    <div class="phase"><div class="l">${p.label} · ${p.weight}%</div>
      <div class="n">${p.score ?? 'n/a'}</div>
      <div class="bar"><i style="width:${p.score ?? 0}%;background:${color(p.score ?? 0)}"></i></div></div>`).join('');
  const lenKeys = new Set(['rise', 'barMargin', 'clearance']);
  const rows = Object.values(a.metrics).map((m) => {
    const isLen = lenKeys.has(m.key) && typeof m.value === 'number' && u === 'ftin';
    const val = isLen ? (m.key === 'rise' ? fmtLen(m.value, u) : signedLen(m.value, u)) : `${fmt(m.value, m.digits)}${m.unit && typeof m.value === 'number' ? ' ' + m.unit : ''}`;
    let tgt = 'Yes';
    if (m.target) {
      if (isLen) tgt = m.target[1] === Infinity ? `${fmtLen(m.target[0], u)} or more` : `${fmtLen(m.target[0], u)}–${fmtLen(m.target[1], u)}`;
      else tgt = m.target[1] === Infinity ? `${m.target[0]} ${m.unit} or more` : `${m.target[0]}–${m.target[1]} ${m.unit}`;
    }
    return `<tr><td>${m.label}</td><td class="num">${val}</td><td class="num">${tgt}</td>
      <td class="num" style="color:${color(m.score ?? 0)}">${m.score ?? 'n/a'}</td></tr>`;
  }).join('');
  const i = a.info;
  const caption = { curve: 'Curve', plant: 'Plant', takeoff: 'Takeoff', peak: 'Peak', landing: 'Landing' };
  const frames = Object.entries(images).map(([k, src]) => `<figure><img src="${src}" alt="${caption[k]} frame"><figcaption>${caption[k]}</figcaption></figure>`).join('');
  const slowNote = state.slow > 1 ? `, slow motion ${state.slow}×` : '';
  const footNote = inputs.takeoffFoot
    ? (state.flipped ? ` The model had your left and right the wrong way round; corrected using your takeoff foot (${inputs.takeoffFoot}).` : '')
    : '';
  return `
  <section class="card">
    <h2>Jump rating</h2>
    <div class="score-head">
      <div class="score-big" style="color:${color(a.overall)}">${a.overall}<small> / 100</small></div>
      <div><div class="rating">${a.rating}</div>
        <div class="note">${inputs.gender === 'female' ? 'Female' : 'Male'}, ${showHeight(inputs.heightCm, u)}${i.barCm !== null ? ` · bar ${showHeight(i.barCm, u)}` : ''} · takeoff leg: ${i.takeoffLeg}</div></div>
    </div>
    <div class="phases">${phases}</div>
  </section>
  ${coachingHtml(a)}
  ${clearanceHtml(a, inputs)}
  <section class="card">
    <h2>Key moments</h2>
    <div class="frames">${frames}</div>
    <p class="note">Plant and takeoff come from when the takeoff foot lands and leaves the ground; landing from when the fall stops. If one is off, use Fix points to set it.</p>
  </section>
  <section class="card">
    <h2>Your numbers against your targets</h2>
    <div class="table-wrap"><table>
      <thead><tr><th>Measure</th><th>You</th><th>Target for you</th><th>Score</th></tr></thead>
      <tbody>${rows}</tbody></table></div>
    ${stepsHtml(i, u)}
    <h3>Also measured</h3>
    <div class="table-wrap"><table><tbody>
      <tr><td>Hip height at takeoff</td><td class="num">${fmtLen(i.hipHeightAtTakeoff, u)}</td></tr>
      ${i.barCm !== null ? `
      <tr><td>Bar height as % of your height</td><td class="num">${fmt(i.barPercentOfHeight, 0)}%</td></tr>
      <tr><td>Lift needed from takeoff hip height to bar</td><td class="num">${fmtLen(i.liftNeededForBar, u)}</td></tr>` : ''}
      <tr><td>Time in the air</td><td class="num">${fmt(i.flightTime, 2)} s</td></tr>
      <tr><td>Vertical jump implied by time in the air (cross-check)</td><td class="num">${fmtLen(i.riseFromFlightTime, u)}</td></tr>
      <tr><td>Body rotation, plant to peak</td><td class="num">${fmt(i.rotation, 0)}°</td></tr>
    </tbody></table></div>
  </section>
  <section class="card">
    <h2>About these numbers</h2>
    <p class="note">Jumper found in ${tracking.foundPct}% of ${tracking.frames} frames (${tracking.quality} quality, ${state.fps} frames a second${slowNote}). ${plural(tracking.filled, 'frame')} filled in from neighbours, ${plural(tracking.swapped, 'left/right mix-up')} fixed, ${plural(tracking.corrected, 'frame')} fixed by hand${tracking.overrides ? `, ${plural(tracking.overrides, 'key moment')} set by hand` : ''}. Camera movement ${tracking.panPct > 2 ? `corrected (the camera panned about ${tracking.panPct}% of the picture width)` : 'was small'}. Scale from ${i.scaleSource}.${i.contactFound ? '' : ' The takeoff foot contact was not clear, so plant and takeoff are approximate.'}${footNote}</p>
    <p class="note">Takeoff angle, approach speed and lean are read off the picture, so they are accurate only when the camera films the takeoff side-on. Filmed from the front or at an angle, motion toward the camera is invisible: the takeoff angle reads too steep and speeds too low.</p>
    <p class="note">Hip height stands in for your center of mass. These are estimates from a single 2D video, not lab measurements. Targets are adjusted for your height and gender and are rounded from published elite high jump data, then widened for non-elite jumpers.</p>
    <p class="sources">Sources:<br>${a.sources.join('<br>')}</p>
  </section>`;
}

async function captureKeyFrames() {
  const images = {};
  const prevMode = state.mode;
  state.mode = 'view';
  for (const [k, idx] of Object.entries(state.analysis.keyFrames)) {
    await seekTo(state.frames[idx].t);
    drawFrame(idx);
    images[k] = canvas.toDataURL('image/jpeg', 0.82);
  }
  state.mode = prevMode;
  return images;
}

function buildKeyButtons() {
  const labels = { curve: 'Curve', plant: 'Plant', takeoff: 'Takeoff', peak: 'Peak', landing: 'Landing' };
  $('keys').innerHTML = '';
  for (const [k, idx] of Object.entries(state.analysis.keyFrames)) {
    const b = document.createElement('button');
    b.type = 'button';
    b.textContent = labels[k];
    b.addEventListener('click', () => { stopPlay(); showFrame(idx); });
    $('keys').appendChild(b);
  }
}

// ---------- analysis flow ----------
function analyzeNow(foot) {
  // Ground coordinates (camera pans removed), real seconds from the start.
  const ground = stabilize(state.frames, state.cum || []);
  const timed = ground.map((f) => ({ ...f, t: (f.t - state.start) / state.slow }));
  return analyzeJump(timed, {
    width: canvas.width, height: canvas.height,
    heightCm: state.inputs.heightCm, gender: state.inputs.gender,
    barCm: state.inputs.barCm, bar: barGround(), overrides: state.overrides,
    takeoffFoot: foot,
  });
}

function runAnalysis() {
  state.frames = applyCorrections(state.cleaned, state.corrections);
  const foot = state.inputs.takeoffFoot;
  // If the athlete told us their takeoff foot and the model has left and
  // right the wrong way round, flip every frame once.
  if (foot && !state.flipChecked) {
    state.flipChecked = true;
    const probe = analyzeNow(null);
    if (probe.info.contactFound && probe.info.takeoffLeg !== foot) {
      state.cleaned = flipSides(state.cleaned);
      state.flipped = true;
      state.frames = applyCorrections(state.cleaned, state.corrections);
    }
  }
  state.analysis = analyzeNow(foot);
  const found = state.frames.filter((f) => f.lms).length;
  const xs = (state.cum || []).map((c) => c.x);
  state.tracking = {
    quality: state.inputs.quality,
    frames: state.frames.length,
    foundPct: Math.round((100 * found) / Math.max(1, state.frames.length)),
    filled: state.frames.filter((f) => f.filled).length,
    swapped: state.frames.filter((f) => f.swapped).length,
    corrected: state.frames.filter((f) => f.corrected).length,
    overrides: Object.keys(state.overrides).length,
    panPct: xs.length ? Math.round((Math.max(...xs) - Math.min(...xs)) * 100) : 0,
  };
}

async function reanalyze(message) {
  try {
    setStatus('Measuring the jump…');
    runAnalysis();
    buildKeyButtons();
    state.images = await captureKeyFrames();
    $('report').innerHTML = reportHtml();
    updateToolState();
    setMode(state.mode);
    await showFrame(Math.min(currentIndex(), state.frames.length - 1));
    const a = state.analysis;
    setStatus(`${message ? message + ' ' : ''}Rating ${a.overall}/100 (${a.rating}).`);
  } catch (err) {
    console.error(err);
    setStatus(err.message || String(err), true);
  }
}

function cleanFrames(followed) {
  const q = QUALITY[state.inputs.quality];
  return cleanTracking(followed.map((f) => ({ t: f.t, poses: f.lms ? [f.lms] : [] })), { smoothRadius: q.smoothRadius, maxGap: Math.round(state.fps / 5) });
}

async function present() {
  if (!state.synthetic) {
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
  }
  $('viewer').hidden = false;
  $('scrub').max = String(state.frames.length - 1);
  buildKeyButtons();
  state.images = await captureKeyFrames();
  $('report').innerHTML = reportHtml();
  setMode('view');
  await showFrame(state.analysis.keyFrames.plant);
  $('viewer').scrollIntoView({ behavior: 'smooth', block: 'start' });
  const a = state.analysis;
  const cl = a.info.clearance;
  const clText = cl ? ` ${cl.verdict === 'cleared' ? 'Cleared the bar (estimate).' : cl.verdict === 'close' ? 'Very close to the bar.' : 'Likely hit the bar.'}` : '';
  setStatus(`Done. Rating ${a.overall}/100 (${a.rating}).${clText} Check that the dots follow the jumper; if not, use Pick the jumper or Fix points.`);
}

// Re-track from a tapped person on the results screen.
async function retrack(frame, point) {
  try {
    const q = QUALITY[state.inputs.quality];
    const lm = await getLandmarker(q.model);
    await seekTo(state.times[frame]);
    const VW = video.videoWidth, VH = video.videoHeight;
    const pose = personAt(lm, { x: point.x * VW, y: point.y * VH });
    if (!pose) { setStatus('Could not find a person there. Try a frame where the jumper is clearer.', true); return; }
    $('progress').hidden = false;
    const { frames, cumNorm } = await followJumper(lm, state.times, { frame, pose }, (p) => {
      $('progress').value = p;
      setStatus(`Following the person you picked… ${Math.round(p * 100)}%`);
    });
    $('progress').hidden = true;
    state.cum = cumNorm;
    state.cleaned = cleanFrames(frames);
    state.corrections = {};
    state.overrides = {};
    state.flipChecked = false;
    state.flipped = false;
    runAnalysis();
    await present();
  } catch (err) {
    console.error(err);
    $('progress').hidden = true;
    setStatus(err.message || String(err), true);
  }
}

// Self-test: open the page with #selftest, then call window.hjSelfTest(videoFile)
// to run a synthetic jump through the cleanup, drawing and report.
if (location.hash === '#selftest') {
  window.hjSelfTest = async (file, { barCm = 170, bar = null, result = 'cleared', units = 'ftin', takeoffFoot = null } = {}) => {
    const { syntheticJump, W, H } = await import('./test/synthetic.js');
    await loadVideo(file);
    const frames = syntheticJump({ lean: 15, launch: 47, risePx: 140, approachPxPerFrame: 20, stance: 5 })
      .map((f) => ({ t: Math.min(f.t, video.duration - 0.05), lms: f.lms }));
    canvas.width = W; canvas.height = H;
    state = {
      synthetic: true, start: 0, fps: 30, slow: 1, times: null, cum: null,
      inputs: { heightCm: 180, gender: 'male', barCm, result, quality: 'standard', units, takeoffFoot, startSeconds: 0, file: file.name },
      corrections: {}, overrides: {}, frames: null, analysis: null, images: {},
      barMarks: bar ? { index: 0, ...bar } : null, barSource: bar ? 'marked by you' : null,
      mode: 'view', barTaps: [], flipped: false, flipChecked: false,
    };
    state.cleaned = cleanFrames(frames);
    runAnalysis();
    await present();
    return state.analysis;
  };
}

$('form').addEventListener('submit', async (e) => {
  e.preventDefault();
  const file = $('file').files[0];
  const heightCm = readHeightCm();
  const gender = $('gender').value;
  const barCm = readBarCm();
  const inputs = {
    heightCm, gender, barCm,
    startSeconds: Math.max(0, Number($('start').value) || 0),
    quality: $('quality').value,
    slow: Number($('slowmo').value) || 1,
    result: $('result').value || null,
    takeoffFoot: $('takeoffFoot').value || null,
    units: $('units').value,
    file: file?.name,
  };
  if (!file) return setStatus('Choose a video first.', true);
  if (!(heightCm >= 100 && heightCm <= 230)) return setStatus('Enter the athlete height: between 3 ft 4 in and 7 ft 6 in (100 to 230 cm).', true);
  if (!gender) return setStatus('Choose a gender so the targets can be adjusted.', true);
  if (barCm !== null && !(barCm >= 50 && barCm <= 260)) return setStatus('Enter a bar height between 50 and 260 cm (1 ft 8 in to 8 ft 6 in), or leave it empty.', true);

  $('go').disabled = true;
  stopPlay();
  state = null;
  try {
    await openSetup(file, inputs);
  } catch (err) {
    console.error(err);
    setStatus(err.message || String(err), true);
  } finally {
    $('go').disabled = false;
  }
});
