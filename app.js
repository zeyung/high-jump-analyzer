import { analyzeJump, angleAt, mid, toPx, phaseAt, LM } from './analysis.js';
import { cleanTracking, applyCorrections, buildTracks, pickTrack, trackAt, plausiblePose, poseBox, meanVisibility } from './tracking.js';
import { toGray, detectBarCandidates, chooseBar } from './bar.js';

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
const CROP = 512; // the zoomed second pass runs the model on a 512 x 512 crop
const BAR_SAMPLES = 12;
const BAR_W = 480;

const $ = (id) => document.getElementById(id);
const video = $('video');
const canvas = $('canvas');
const ctx = canvas.getContext('2d');
const cropCanvas = document.createElement('canvas');
cropCanvas.width = cropCanvas.height = CROP;
const cctx = cropCanvas.getContext('2d');
const barCanvas = document.createElement('canvas');
const bctx = barCanvas.getContext('2d', { willReadFrequently: true });

const landmarkers = {};
let state = null;

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

const view = loadView();
function loadView() {
  const v = { markers: Object.fromEntries(MARKERS.map((m) => [m.k, true])), layers: { skeleton: true, lean: true, bar: true, clearance: true, phase: true, legend: false } };
  try {
    const saved = JSON.parse(localStorage.getItem('hj-view') || 'null');
    if (saved) { Object.assign(v.markers, saved.markers || {}); Object.assign(v.layers, saved.layers || {}); }
  } catch { /* storage blocked: use defaults */ }
  return v;
}
function saveView() { try { localStorage.setItem('hj-view', JSON.stringify(view)); } catch { /* ignore */ } }

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
async function getLandmarker(kind) {
  if (landmarkers[kind]) return landmarkers[kind];
  const { PoseLandmarker, FilesetResolver } = await import(`${TV}/vision_bundle.mjs`);
  const fileset = await FilesetResolver.forVisionTasks(`${TV}/wasm`);
  // IMAGE mode: every frame and every crop is judged on its own, so a wrong
  // lock-on (for example onto the landing mat) cannot carry over between
  // frames. Our own tracker links the frames instead.
  const opts = (delegate) => ({
    baseOptions: { modelAssetPath: MODELS[kind], delegate },
    runningMode: 'IMAGE',
    numPoses: 4,
    minPoseDetectionConfidence: 0.3,
    minPosePresenceConfidence: 0.3,
    minTrackingConfidence: 0.3,
  });
  try {
    landmarkers[kind] = await PoseLandmarker.createFromOptions(fileset, opts('GPU'));
  } catch {
    landmarkers[kind] = await PoseLandmarker.createFromOptions(fileset, opts('CPU'));
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
    video.onerror = () => reject(new Error('This video could not be opened. Try an MP4 or MOV file.'));
    video.src = URL.createObjectURL(file);
    video.load();
  });
}

const toLms = (pose) => pose.map((p) => ({ x: p.x, y: p.y, visibility: p.visibility }));
const breathe = () => new Promise((r) => setTimeout(r, 0));

// Pass 1: whole frame, everyone the model can see. Also samples frames for
// finding the bar.
async function detectAll(start, fps, lm, onProgress) {
  const VW = video.videoWidth, VH = video.videoHeight;
  const end = Math.min(video.duration, start + MAX_SECONDS);
  const n = Math.max(0, Math.floor((end - start) * fps));
  const sampleAt = new Set(Array.from({ length: BAR_SAMPLES }, (_, j) => Math.round((j * (n - 1)) / Math.max(1, BAR_SAMPLES - 1))));
  barCanvas.width = BAR_W;
  barCanvas.height = Math.round((BAR_W * VH) / VW);
  const grays = [];
  const detections = [];
  for (let i = 0; i < n; i++) {
    const t = start + i / fps;
    await seekTo(t);
    const r = lm.detect(video);
    detections.push({ t, poses: (r.landmarks || []).map(toLms) });
    if (sampleAt.has(i)) {
      bctx.drawImage(video, 0, 0, barCanvas.width, barCanvas.height);
      grays.push(toGray(bctx.getImageData(0, 0, barCanvas.width, barCanvas.height).data, barCanvas.width, barCanvas.height));
    }
    if (i % 3 === 0) onProgress((i + 1) / n);
    if (i % 8 === 0) await breathe();
  }
  return { detections, grays, barW: barCanvas.width, barH: barCanvas.height };
}

// Run the model on a square crop around (cx, cy) in video pixels.
function detectCrop(lm, box) {
  const VW = video.videoWidth, VH = video.videoHeight;
  const side = box.side, sx = box.cx - side / 2, sy = box.cy - side / 2, k = CROP / side;
  cctx.fillStyle = '#000';
  cctx.fillRect(0, 0, CROP, CROP);
  const x0 = Math.max(0, sx), y0 = Math.max(0, sy), x1 = Math.min(VW, sx + side), y1 = Math.min(VH, sy + side);
  if (x1 <= x0 || y1 <= y0) return [];
  cctx.drawImage(video, x0, y0, x1 - x0, y1 - y0, (x0 - sx) * k, (y0 - sy) * k, (x1 - x0) * k, (y1 - y0) * k);
  const r = lm.detect(cropCanvas);
  return (r.landmarks || []).map((pose) => pose.map((p) => ({ x: (sx + p.x * side) / VW, y: (sy + p.y * side) / VH, visibility: p.visibility })));
}

const hipPx = (lms, VW, VH) => ({ x: ((lms[23].x + lms[24].x) / 2) * VW, y: ((lms[23].y + lms[24].y) / 2) * VH });

// Pass 2: follow one person with a zoomed crop, frame by frame, forward and
// backward from the frame where they are seen most clearly. The crop gives
// the model several times more pixels of the jumper than the whole frame.
async function followJumper(lm, detections, seed, onProgress) {
  const VW = video.videoWidth, VH = video.videoHeight, aspect = VW / VH;
  const n = detections.length;
  const hint = seed.track ? new Map(seed.track.items.map((it) => [it.frame, it.lms])) : new Map();
  const out = new Array(n).fill(null);
  const minSide = Math.min(VW, VH) * 0.18, maxSide = Math.max(VW, VH) * 1.1;
  const sideFor = (lms) => Math.min(maxSide, Math.max(minSide, poseBox(lms, aspect).size * VH * 1.7));
  let done = 0;

  async function run(from, to, step, startBox) {
    let box = { ...startBox };
    let vel = { x: 0, y: 0 };
    let last = null, misses = 0;
    for (let k = from; step > 0 ? k <= to : k >= to; k += step) {
      await seekTo(detections[k].t);
      const guess = { cx: box.cx + vel.x, cy: box.cy + vel.y, side: box.side };
      // If pass 1 saw this person here, re-centre on it.
      const h = hint.get(k);
      if (h) {
        const hp = hipPx(h, VW, VH);
        if (Math.hypot(hp.x - guess.cx, hp.y - guess.cy) < guess.side * 0.6) { guess.cx = hp.x; guess.cy = hp.y; }
      }
      const cands = detectCrop(lm, guess).filter((p) => plausiblePose(p, aspect));
      let pick = null, bestD = guess.side * 0.55;
      for (const c of cands) {
        const hp = hipPx(c, VW, VH);
        const d = Math.hypot(hp.x - guess.cx, hp.y - guess.cy);
        if (d < bestD) { bestD = d; pick = c; }
      }
      if (!pick && h) pick = h;
      out[k] = pick;
      if (pick) {
        const hp = hipPx(pick, VW, VH);
        if (last) vel = { x: 0.6 * (hp.x - last.x) + 0.4 * vel.x, y: 0.6 * (hp.y - last.y) + 0.4 * vel.y };
        last = hp;
        box = { cx: hp.x, cy: hp.y, side: 0.7 * box.side + 0.3 * sideFor(pick) };
        misses = 0;
      } else {
        misses++;
        box = guess;
        if (misses > 6) vel = { x: vel.x * 0.5, y: vel.y * 0.5 };
      }
      done++;
      if (done % 3 === 0) onProgress(done / n);
      if (done % 6 === 0) await breathe();
    }
  }

  const s = seed.frame;
  await run(s, n - 1, 1, seed.box);
  // The backward run starts from what the forward run found at the seed frame.
  const sp = out[s];
  const back = sp ? { cx: hipPx(sp, VW, VH).x, cy: hipPx(sp, VW, VH).y, side: sideFor(sp) } : seed.box;
  if (s > 0) await run(s - 1, 0, -1, back);
  onProgress(1);
  return detections.map((d, k) => ({ t: d.t, lms: out[k] }));
}

function seedFromTrack(track) {
  const VW = video.videoWidth, VH = video.videoHeight, aspect = VW / VH;
  let best = track.items[0];
  for (const it of track.items) if (meanVisibility(it.lms) > meanVisibility(best.lms)) best = it;
  const hp = hipPx(best.lms, VW, VH);
  const side = Math.max(Math.min(VW, VH) * 0.18, poseBox(best.lms, aspect).size * VH * 1.7);
  return { frame: best.frame, box: { cx: hp.x, cy: hp.y, side }, track };
}

// ---------- drawing ----------
// On-screen sizes stay readable whatever the video resolution.
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

function drawMarker(m, x, y, r, s) {
  if (m.side === 'R') {
    // Ring: bigger than the dots, so a left dot underneath stays visible.
    const rr = r * 1.6;
    ctx.lineWidth = r * 0.75 + 2 * s; ctx.strokeStyle = '#000';
    ctx.beginPath(); ctx.arc(x, y, rr, 0, Math.PI * 2); ctx.stroke();
    ctx.lineWidth = r * 0.75; ctx.strokeStyle = m.color;
    ctx.beginPath(); ctx.arc(x, y, rr, 0, Math.PI * 2); ctx.stroke();
  } else {
    ctx.beginPath(); ctx.arc(x, y, r, 0, Math.PI * 2);
    ctx.fillStyle = m.color; ctx.fill();
    ctx.lineWidth = 1.5 * s; ctx.strokeStyle = '#000'; ctx.stroke();
  }
}

function barPx() {
  if (!state.bar) return null;
  const W = canvas.width, H = canvas.height;
  return { a: { x: state.bar.a.x * W, y: state.bar.a.y * H }, b: { x: state.bar.b.x * W, y: state.bar.b.y * H } };
}

function drawBarLayer(s) {
  const b = barPx();
  if (!b) return;
  ctx.lineCap = 'round';
  ctx.lineWidth = 6 * s; ctx.strokeStyle = 'rgba(0,0,0,0.75)';
  ctx.beginPath(); ctx.moveTo(b.a.x, b.a.y); ctx.lineTo(b.b.x, b.b.y); ctx.stroke();
  ctx.lineWidth = 3 * s; ctx.strokeStyle = '#ffffff';
  ctx.beginPath(); ctx.moveTo(b.a.x, b.a.y); ctx.lineTo(b.b.x, b.b.y); ctx.stroke();
}

function groundPoint() {
  if (state.bar?.ground) return state.bar.ground;
  const g = state.analysis?.info.groundPx;
  return g ? { x: g.x / canvas.width, y: g.y / canvas.height } : null;
}

function drawHandles(s) {
  const W = canvas.width, H = canvas.height;
  const circle = (p) => {
    ctx.beginPath(); ctx.arc(p.x * W, p.y * H, 9 * s, 0, Math.PI * 2);
    ctx.fillStyle = '#ffffff'; ctx.fill(); ctx.lineWidth = 3 * s; ctx.strokeStyle = '#000'; ctx.stroke();
  };
  if (state.bar) {
    circle(state.bar.a); circle(state.bar.b);
    const g = groundPoint();
    if (g) {
      ctx.fillStyle = '#ffffff'; ctx.strokeStyle = '#000'; ctx.lineWidth = 3 * s;
      ctx.fillRect(g.x * W - 8 * s, g.y * H - 8 * s, 16 * s, 16 * s);
      ctx.strokeRect(g.x * W - 8 * s, g.y * H - 8 * s, 16 * s, 16 * s);
    }
  } else {
    state.barTaps.forEach(circle);
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

  if (state.mode === 'bar') { drawBarLayer(s); drawHandles(s); } else if (L.bar) drawBarLayer(s);

  if (f && f.lms) {
    const P = toPx(f.lms, W, H);
    const bodyPx = analysis?.info.bodyPx || H * 0.4;
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
      // From the lower foot (the one on the ground) to the shoulders.
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
    const b = barPx();
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
      rows.push(['Hip height', fmtLen((a.info.groundY - hp.y) * a.info.cmPerPx, state.inputs.units)]);
    }
  }
  $('readout').innerHTML = rows.map(([k, v]) => `<div class="ro-row"><span>${k}</span><strong>${v}</strong></div>`).join('');
}

function frameIndexAt(t) {
  const i = Math.round((t - state.start) * state.fps);
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
  const { analysis, inputs, tracking, bar } = state;
  const out = { inputs, tracking, bar, overall: analysis.overall, rating: analysis.rating, phases: analysis.phases, metrics: analysis.metrics, info: analysis.info, keyFrames: analysis.keyFrames, sources: analysis.sources };
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
  $('markBar').textContent = mode === 'bar' ? 'Done with the bar' : (state.bar ? 'Adjust the bar' : 'Mark the bar');
  $('fixPoints').textContent = mode === 'fix' ? 'Done fixing' : 'Fix points';
  $('keyTools').hidden = mode !== 'fix';
  canvas.classList.toggle('tapping', mode !== 'view');
  if (mode === 'pick') setHint('Tap the jumper on the video. Use a frame where they are clearly visible.');
  else if (mode === 'bar') setHint(state.bar
    ? 'Drag the round handles onto the two ends of the bar. Optional: drag the square onto the floor directly under the bar.'
    : 'Tap the left end of the bar, then the right end.');
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
  $('clearBar').hidden = !state.bar;
}

$('pickJumper').addEventListener('click', () => {
  if (!state) return;
  if (!state.detections) { setHint('Picking the jumper needs a real clip.'); return; }
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
    setHint('Enter the bar height above and press Analyze again, then mark the bar.');
    return;
  }
  stopPlay();
  state.barTaps = [];
  setMode('bar');
});
$('clearBar').addEventListener('click', async () => {
  state.bar = null;
  state.barSource = 'removed';
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

  if (state.mode === 'pick') {
    const i = currentIndex();
    setMode('view');
    await retrack(i, norm);
    return;
  }

  if (state.mode === 'bar') {
    if (!state.bar) {
      state.barTaps.push(norm);
      if (state.barTaps.length === 1) setHint('Now tap the right end of the bar.');
      if (state.barTaps.length === 2) {
        const [a, b] = state.barTaps;
        state.bar = { a, b };
        state.barSource = 'marked by you';
        state.barEdited = true;
        setMode('bar');
      }
      drawFrame(currentIndex());
      return;
    }
    const handles = [['a', state.bar.a], ['b', state.bar.b], ['ground', groundPoint()]];
    let best = null, bestD = 28 * s;
    for (const [name, h] of handles) {
      if (!h) continue;
      const d = Math.hypot(h.x * W - p.x, h.y * H - p.y);
      if (d < bestD) { bestD = d; best = name; }
    }
    if (best) {
      drag = { type: 'bar', handle: best };
      canvas.setPointerCapture(e.pointerId);
      e.preventDefault();
    }
    return;
  }

  // Fix mode: grab the nearest point.
  const i = currentIndex();
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
    state.bar = { ...state.bar, [drag.handle]: { x: nx, y: ny } };
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
    athlete: { heightCm: state.inputs.heightCm, gender: state.inputs.gender },
    barCm: state.inputs.barCm,
    bar: state.bar,
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

function clearanceHtml(a, inputs) {
  const i = a.info, u = inputs.units;
  const vertical = `<div class="stats">
      <div><div class="l">Vertical jump (hip rise)</div><div class="n">${fmtLen(i.verticalJump, u)}</div></div>
      <div><div class="l">Peak hip height</div><div class="n">${fmtLen(i.peakHipHeight, u)}</div></div>
      ${i.barCm !== null ? `<div><div class="l">Bar</div><div class="n">${u === 'ftin' ? ftIn(i.barCm) : `${i.barCm} cm`}</div></div>
      <div><div class="l">Hips vs bar</div><div class="n">${signedLen(i.peakHipHeight - i.barCm, u)}</div></div>` : ''}
    </div>`;
  const barNote = {
    'found automatically': 'The bar was found in the video automatically and is taken to be exactly the height you entered, which sets the scale for every measurement. If the white line is not on the bar, use Adjust the bar.',
    'marked by you': 'Uses the bar you marked, taken to be exactly the height you entered, which sets the scale for every measurement.',
    'not found': 'The bar could not be found in the video, so heights use your height for scale. Use Mark the bar for a more accurate result.',
    removed: 'No bar is marked, so heights use your height for scale.',
  }[state.barSource] || '';
  if (!i.clearance) {
    return `<section class="card"><h2>Vertical jump and bar clearance</h2>${vertical}
      <p class="note">Enter the bar height and analyze again to check whether each body part cleared the bar.</p></section>`;
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
    <p class="verdict" style="color:${verdictColor}">${verdictText}</p>
    ${said ? `<p class="note">You said the jump ${said}. ${agrees ? 'That matches the estimate.' : 'The estimate disagrees, so the tracking or the bar line may be off. Check the bar line, or fix points on the frames over the bar.'}</p>` : ''}
    <div class="table-wrap"><table>
      <thead><tr><th>Body part</th><th>Best height over the bar</th></tr></thead>
      <tbody>${rows}</tbody></table></div>
    <p class="note">Each body part has to be above the bar at some point in the air, so this compares each part's highest point with the bar. Joint points sit inside the body, so a little is taken off for the back of the body. ${barNote}</p>
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
  const tips = Object.values(a.metrics).filter((m) => m.tip).sort((x, y) => (x.score ?? 0) - (y.score ?? 0));
  const i = a.info;
  const caption = { curve: 'Curve', plant: 'Plant', takeoff: 'Takeoff', peak: 'Peak', landing: 'Landing' };
  const frames = Object.entries(images).map(([k, src]) => `<figure><img src="${src}" alt="${caption[k]} frame"><figcaption>${caption[k]}</figcaption></figure>`).join('');
  const slowNote = state.slow > 1 ? `, slow motion ${state.slow}×` : '';
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
    <h2>What to work on</h2>
    ${tips.length ? `<ol class="tiplist">${tips.map((m) => `<li><strong>${m.label}.</strong> ${m.tip}</li>`).join('')}</ol>` : '<p>Every measure is inside your target range. Nice jump.</p>'}
    <h3>Tracking quality</h3>
    <p class="note">Jumper found in ${tracking.foundPct}% of ${tracking.frames} frames (${tracking.quality} quality, ${state.fps} frames a second${slowNote}, zoomed second pass ${tracking.zoomed ? 'on' : 'off'}). ${plural(tracking.filled, 'frame')} filled in from neighbours, ${plural(tracking.swapped, 'left/right mix-up')} fixed, ${plural(tracking.corrected, 'frame')} fixed by hand${tracking.overrides ? `, ${plural(tracking.overrides, 'key moment')} set by hand` : ''}. Scale from ${i.scaleSource}.${i.contactFound ? '' : ' The takeoff foot contact was not clear, so plant and takeoff are approximate.'}</p>
    <p class="note">Hip height stands in for your center of mass. These are estimates from a single 2D video, not lab measurements, and assume a still camera. Targets are adjusted for your height and gender and are rounded from published elite high jump data, then widened for non-elite jumpers.</p>
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
function analyzeNow() {
  const timed = state.frames.map((f) => ({ ...f, t: (f.t - state.start) / state.slow }));
  return analyzeJump(timed, {
    width: canvas.width, height: canvas.height,
    heightCm: state.inputs.heightCm, gender: state.inputs.gender,
    barCm: state.inputs.barCm, bar: state.bar, overrides: state.overrides,
  });
}

function runAnalysis() {
  state.frames = applyCorrections(state.cleaned, state.corrections);
  // Find the bar automatically the first time, using the athlete to tell the
  // bar apart from other lines (fences, mat edges, the horizon).
  if (state.inputs.barCm !== null && !state.bar && state.barSource === 'auto-pending') {
    const first = analyzeNow();
    const pick = chooseBar(state.barCandidates || [], first.info, state.inputs.barCm, canvas.width, canvas.height);
    if (pick) { state.bar = { a: pick.a, b: pick.b }; state.barSource = 'found automatically'; } else state.barSource = 'not found';
  }
  state.analysis = analyzeNow();
  const found = state.frames.filter((f) => f.lms).length;
  state.tracking = {
    quality: state.inputs.quality,
    frames: state.frames.length,
    foundPct: Math.round((100 * found) / Math.max(1, state.frames.length)),
    filled: state.frames.filter((f) => f.filled).length,
    swapped: state.frames.filter((f) => f.swapped).length,
    corrected: state.frames.filter((f) => f.corrected).length,
    overrides: Object.keys(state.overrides).length,
    zoomed: !!state.zoomed,
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
  $('viewer').hidden = false;
  $('scrub').max = String(state.frames.length - 1);
  buildKeyButtons();
  state.images = await captureKeyFrames();
  $('report').innerHTML = reportHtml();
  setMode('view');
  await showFrame(state.analysis.keyFrames.plant);
  const a = state.analysis;
  const cl = a.info.clearance;
  const clText = cl ? ` ${cl.verdict === 'cleared' ? 'Cleared the bar (estimate).' : cl.verdict === 'close' ? 'Very close to the bar.' : 'Likely hit the bar.'}` : '';
  const barText = state.inputs.barCm === null ? '' : state.barSource === 'found automatically' ? ' Bar found in the video.' : state.barSource === 'not found' ? ' Could not find the bar: use Mark the bar.' : '';
  setStatus(`Done. Rating ${a.overall}/100 (${a.rating}).${clText}${barText} Check that the dots follow the jumper; if not, use Pick the jumper or Fix points.`);
}

// Re-track from a tapped person.
async function retrack(frame, point) {
  try {
    const VW = video.videoWidth, VH = video.videoHeight, aspect = VW / VH;
    const lm = await getLandmarker(QUALITY[state.inputs.quality].model);
    const tr = trackAt(state.tracks, frame, point, aspect);
    const it = tr && tr.items.find((x) => x.frame === frame);
    const seed = it
      ? { ...seedFromTrack({ items: [it] }), track: tr }
      : { frame, box: { cx: point.x * VW, cy: point.y * VH, side: Math.min(VW, VH) * 0.45 }, track: null };
    $('progress').hidden = false;
    const followed = await followJumper(lm, state.detections, seed, (p) => {
      $('progress').value = p;
      setStatus(`Following the person you picked… ${Math.round(p * 100)}%`);
    });
    $('progress').hidden = true;
    state.cleaned = cleanFrames(followed);
    state.corrections = {};
    state.overrides = {};
    if (state.barSource === 'found automatically' || state.barSource === 'not found') { state.bar = null; state.barSource = 'auto-pending'; }
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
  window.hjSelfTest = async (file, { barCm = 170, bar = null, result = 'cleared', units = 'ftin' } = {}) => {
    const { syntheticJump, W, H } = await import('./test/synthetic.js');
    await loadVideo(file);
    canvas.width = W; canvas.height = H;
    const frames = syntheticJump({ lean: 15, launch: 47, risePx: 140, approachPxPerFrame: 20, stance: 5 })
      .map((f) => ({ t: Math.min(f.t, video.duration - 0.05), lms: f.lms }));
    state = {
      start: 0, fps: 30, slow: 1, detections: null, tracks: [], zoomed: false,
      inputs: { heightCm: 180, gender: 'male', barCm, result, quality: 'standard', units, startSeconds: 0, file: file.name },
      corrections: {}, overrides: {}, frames: null, analysis: null, images: {}, bar,
      barSource: bar ? 'marked by you' : 'not found', mode: 'view', barTaps: [], barCandidates: [],
    };
    state.cleaned = cleanFrames(frames);
    runAnalysis();
    await present();
    return state.analysis;
  };
  window.hjState = () => state;
}

$('form').addEventListener('submit', async (e) => {
  e.preventDefault();
  const file = $('file').files[0];
  const heightCm = readHeightCm();
  const gender = $('gender').value;
  const barCm = readBarCm();
  const start = Math.max(0, Number($('start').value) || 0);
  const quality = $('quality').value;
  const slow = Number($('slowmo').value) || 1;
  const result = $('result').value || null;
  if (!file) return setStatus('Choose a video first.', true);
  if (!(heightCm >= 100 && heightCm <= 230)) return setStatus('Enter the athlete height: between 3 ft 4 in and 7 ft 6 in (100 to 230 cm).', true);
  if (!gender) return setStatus('Choose a gender so the targets can be adjusted.', true);
  if (barCm !== null && !(barCm >= 50 && barCm <= 260)) return setStatus('Enter a bar height between 50 and 260 cm (1 ft 8 in to 8 ft 6 in), or leave it empty.', true);

  $('go').disabled = true;
  $('viewer').hidden = true;
  $('report').innerHTML = '';
  stopPlay();
  state = null;
  try {
    setStatus('Opening video…');
    await loadVideo(file);
    if (start >= video.duration) throw new Error(`The video is only ${video.duration.toFixed(1)} s long.`);
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const aspect = video.videoWidth / video.videoHeight;

    const q = QUALITY[quality];
    setStatus(`Loading the ${quality === 'high' ? 'high-accuracy ' : ''}pose model (first time takes a few seconds)…`);
    const lm = await getLandmarker(q.model);

    $('progress').hidden = false;
    const { detections, grays, barW, barH } = await detectAll(start, q.fps, lm, (p) => {
      $('progress').value = p * 0.4;
      setStatus(`Step 1 of 3: finding people in each frame… ${Math.round(p * 100)}%`);
    });

    setStatus('Step 2 of 3: picking out the jumper…');
    const tracks = buildTracks(detections, { aspect });
    const track = pickTrack(tracks, detections.length, aspect);
    if (!track) throw new Error('No person was found in this clip. Make sure the whole jumper is in the picture and well lit.');

    const followed = await followJumper(lm, detections, seedFromTrack(track), (p) => {
      $('progress').value = 0.4 + p * 0.55;
      setStatus(`Step 2 of 3: following the jumper up close… ${Math.round(p * 100)}%`);
    });

    setStatus('Step 3 of 3: finding the bar and measuring the jump…');
    $('progress').value = 0.97;
    await breathe();
    const barCandidates = barCm !== null ? detectBarCandidates(grays, barW, barH) : [];
    $('progress').hidden = true;

    state = {
      start, fps: q.fps, slow, detections, tracks, zoomed: true,
      inputs: { heightCm, gender, barCm, result, quality, units: $('units').value, startSeconds: start, file: file.name },
      corrections: {}, overrides: {}, frames: null, analysis: null, images: {},
      bar: null, barSource: barCm !== null ? 'auto-pending' : null, barCandidates, mode: 'view', barTaps: [],
    };
    state.cleaned = cleanFrames(followed);
    runAnalysis();
    await present();
  } catch (err) {
    console.error(err);
    $('progress').hidden = true;
    setStatus(err.message || String(err), true);
  } finally {
    $('go').disabled = false;
  }
});
