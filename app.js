import { analyzeJump, angleAt, mid, toPx, phaseAt, LM } from './analysis.js';
import { cleanTracking, applyCorrections } from './tracking.js';

const TV = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14';
const MODELS = {
  standard: 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task',
  high: 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task',
};
// Standard: full model, 30 frames a second, light smoothing.
// High: heavy model, 60 frames a second, a little more smoothing. Slower.
const QUALITY = {
  standard: { model: 'standard', fps: 30, smoothRadius: 1 },
  high: { model: 'high', fps: 60, smoothRadius: 2 },
};
const MAX_SECONDS = 15;

const $ = (id) => document.getElementById(id);
const video = $('video');
const canvas = $('canvas');
const ctx = canvas.getContext('2d');

const landmarkers = {};
let lastTs = 0; // VIDEO mode needs ever-increasing timestamps, across runs too
let state = null;
// state: {posesPerFrame, cleaned, corrections, frames, analysis, start, fps,
//         slow, inputs, images, bar, mode, barTaps, tracking}

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
  const opts = (delegate) => ({
    baseOptions: { modelAssetPath: MODELS[kind], delegate },
    runningMode: 'VIDEO',
    numPoses: 3, // find bystanders too, so the cleanup can follow the jumper
    minPoseDetectionConfidence: 0.4,
    minPosePresenceConfidence: 0.4,
    minTrackingConfidence: 0.4,
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

async function extractPoses(start, fps, kind, onProgress) {
  const lm = await getLandmarker(kind);
  const end = Math.min(video.duration, start + MAX_SECONDS);
  const n = Math.max(0, Math.floor((end - start) * fps));
  const out = [];
  for (let i = 0; i < n; i++) {
    const t = start + i / fps;
    await seekTo(t);
    lastTs = Math.max(lastTs + 1, Math.round(performance.now()));
    const r = lm.detectForVideo(video, lastTs);
    const poses = (r.landmarks || []).map((pose) => pose.map((p) => ({ x: p.x, y: p.y, visibility: p.visibility })));
    out.push({ t, poses });
    if (i % 3 === 0) onProgress((i + 1) / n);
    if (i % 10 === 0) await new Promise((r2) => setTimeout(r2, 0));
  }
  onProgress(1);
  return out;
}

// ---------- drawing ----------
const BONES = [
  [11, 12], [11, 13], [13, 15], [12, 14], [14, 16], [11, 23], [12, 24], [23, 24],
  [23, 25], [25, 27], [24, 26], [26, 28], [27, 29], [29, 31], [27, 31], [28, 30], [30, 32], [28, 32],
];
const ANCHORS = [11, 12, 23, 24, 25, 26, 27, 28];
const FIXABLE = [0, ...ANCHORS];
const PART_POINTS = { head: [0], shoulders: [11, 12], hips: [23, 24], knees: [25, 26], feet: [27, 28] };

const scaleOf = () => Math.max(canvas.width, canvas.height) / 900;

function pill(text, x, y, s, color = '#ffd23f') {
  ctx.font = `600 ${Math.round(14 * s)}px system-ui, sans-serif`;
  const w = ctx.measureText(text).width + 10 * s, h = 20 * s;
  const bx = Math.min(Math.max(2, x), canvas.width - w - 2), by = Math.min(Math.max(2, y - h / 2), canvas.height - h - 2);
  ctx.fillStyle = 'rgba(0,0,0,0.72)';
  ctx.fillRect(bx, by, w, h);
  ctx.fillStyle = color;
  ctx.textBaseline = 'middle';
  ctx.fillText(text, bx + 5 * s, by + h / 2);
}

function dot(x, y, r, fill, s) {
  ctx.beginPath();
  ctx.arc(x, y, r, 0, Math.PI * 2);
  ctx.fillStyle = fill; ctx.fill();
  ctx.lineWidth = 2 * s; ctx.strokeStyle = '#000'; ctx.stroke();
}

function drawBar(s) {
  const W = canvas.width, H = canvas.height;
  if (state.mode === 'bar') {
    state.barTaps.forEach((p) => dot(p.x * W, p.y * H, 7 * s, '#ff4d4d', s));
  }
  const bl = state.analysis?.info.barLine;
  if (!bl) return;
  ctx.lineWidth = 3 * s;
  ctx.strokeStyle = '#ff4d4d';
  ctx.setLineDash(bl.marked ? [] : [10 * s, 8 * s]);
  // Extend the marked line a little past the ends so it reads as the bar.
  ctx.beginPath(); ctx.moveTo(bl.x1, bl.y1); ctx.lineTo(bl.x2, bl.y2); ctx.stroke();
  ctx.setLineDash([]);
  const lx = Math.min(bl.x1, bl.x2), ly = bl.x1 <= bl.x2 ? bl.y1 : bl.y2;
  pill(`Bar ${state.inputs.barCm} cm${bl.marked ? '' : ' (estimated line)'}`, lx + 6 * s, ly - 16 * s, s, '#ff8080');
}

function drawFrame(i) {
  const W = canvas.width, H = canvas.height;
  ctx.drawImage(video, 0, 0, W, H);
  if (!state || !state.frames) return;
  const s = scaleOf();
  const { frames, analysis } = state;
  const f = frames[i];
  const kf = analysis ? analysis.keyFrames : null;
  const phase = kf ? phaseAt(i, kf) : '';
  const realT = f ? (f.t - state.start) / state.slow : 0;
  pill(`${phase}${phase ? '  ·  ' : ''}${realT.toFixed(2)} s${f?.corrected ? '  ·  fixed by hand' : f?.filled ? '  ·  filled in' : ''}`, 8 * s, 18 * s, s, '#ffffff');
  if (analysis) drawBar(s);

  if (!f || !f.lms) return;
  const P = toPx(f.lms, W, H);

  ctx.lineCap = 'round';
  ctx.lineWidth = 4 * s;
  ctx.strokeStyle = 'rgba(255,255,255,0.9)';
  for (const [a, b] of BONES) {
    ctx.beginPath(); ctx.moveTo(P[a].x, P[a].y); ctx.lineTo(P[b].x, P[b].y); ctx.stroke();
  }
  const fixing = state.mode === 'fix';
  const corr = state.corrections[i] || {};
  for (const k of fixing ? FIXABLE : ANCHORS) {
    const fill = corr[k] ? '#5dff8f' : '#ffd23f';
    dot(P[k].x, P[k].y, (fixing ? 10 : 6.5) * s, fill, s);
  }
  if (fixing) return; // keep the picture clean while dragging points

  const joints = [
    ['L knee', LM.lHip, LM.lKnee, LM.lAnkle], ['R knee', LM.rHip, LM.rKnee, LM.rAnkle],
    ['L hip', LM.lShoulder, LM.lHip, LM.lKnee], ['R hip', LM.rShoulder, LM.rHip, LM.rKnee],
  ];
  for (const [name, a, b, c] of joints) {
    const ang = angleAt(P[a], P[b], P[c]);
    if (Number.isFinite(ang)) pill(`${name} ${Math.round(ang)}°`, P[b].x + 10 * s, P[b].y, s);
  }

  if (kf && i <= kf.takeoff) {
    const A = mid(P[LM.lAnkle], P[LM.rAnkle]), S = mid(P[LM.lShoulder], P[LM.rShoulder]);
    const len = Math.hypot(S.x - A.x, S.y - A.y);
    ctx.setLineDash([6 * s, 6 * s]);
    ctx.lineWidth = 2 * s; ctx.strokeStyle = 'rgba(255,255,255,0.7)';
    ctx.beginPath(); ctx.moveTo(A.x, A.y); ctx.lineTo(A.x, A.y - len); ctx.stroke();
    ctx.setLineDash([]);
    ctx.lineWidth = 3.5 * s; ctx.strokeStyle = '#3fd0ff';
    ctx.beginPath(); ctx.moveTo(A.x, A.y); ctx.lineTo(S.x, S.y); ctx.stroke();
    const lean = Math.atan2(Math.abs(S.x - A.x), A.y - S.y) * 180 / Math.PI;
    pill(`Lean ${Math.round(lean)}°`, S.x + 12 * s, S.y - 24 * s, s, '#3fd0ff');
  }

  // Clearance: label each body part at the frame where it was highest over the bar.
  const cl = analysis?.info.clearance;
  if (cl && kf && i > kf.takeoff && i <= kf.landing + 2) {
    const near = Math.max(1, Math.round(state.fps / 15));
    for (const part of cl.parts) {
      if (Math.abs(i - part.frame) > near) continue;
      const pts = PART_POINTS[part.key].map((k) => P[k]);
      const x = pts.reduce((a, p) => a + p.x, 0) / pts.length, y = pts.reduce((a, p) => a + p.y, 0) / pts.length;
      const good = part.cm >= 0;
      pill(`${part.label} ${part.cm >= 0 ? '+' : ''}${Math.round(part.cm)} cm`, x - 40 * s, y - 30 * s, s, good ? '#5dff8f' : '#ff8080');
    }
    const verdict = { cleared: 'Cleared (estimate)', close: 'Very close to the bar', knocked: `Likely hit the bar: ${cl.lowestLabel.toLowerCase()}` }[cl.verdict];
    pill(verdict, W - 260 * s, 18 * s, s, cl.verdict === 'cleared' ? '#5dff8f' : cl.verdict === 'close' ? '#ffd23f' : '#ff8080');
  }
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
  const out = { inputs, tracking, overall: analysis.overall, rating: analysis.rating, phases: analysis.phases, metrics: analysis.metrics, info: analysis.info, keyFrames: analysis.keyFrames, sources: analysis.sources };
  download(new Blob([JSON.stringify(out, null, 2)], { type: 'application/json' }), 'metrics.json');
});

$('dlReport').addEventListener('click', async () => {
  const css = await fetch('style.css').then((r) => r.text()).catch(() => '');
  const html = `<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>High jump report</title><style>${css}</style></head><body><main style="padding-top:24px">${reportHtml()}</main></body></html>`;
  download(new Blob([html], { type: 'text/html' }), 'high-jump-report.html');
});

// ---------- tools: mark the bar, fix points ----------
function setMode(mode) {
  if (!state) return;
  state.mode = mode;
  $('markBar').classList.toggle('active', mode === 'bar');
  $('fixPoints').classList.toggle('active', mode === 'fix');
  $('markBar').textContent = mode === 'bar' ? 'Cancel marking' : (state.bar ? 'Re-mark the bar' : 'Mark the bar');
  $('fixPoints').textContent = mode === 'fix' ? 'Done fixing' : 'Fix points';
  canvas.classList.toggle('tapping', mode !== 'view');
  if (mode === 'bar') {
    state.barTaps = [];
    setHint('Tap 1 of 3: the left end of the bar.');
  } else if (mode === 'fix') {
    setHint('Drag a point to where the joint really is. Use the slider or the key moment buttons to change frames. Fixed points turn green.');
  } else {
    setHint('');
  }
  updateToolState();
  drawFrame(currentIndex());
}

function updateToolState() {
  const n = Object.keys(state.corrections).filter((k) => Object.keys(state.corrections[k]).length).length;
  $('applyFixes').hidden = n === 0;
  $('applyFixes').textContent = `Re-analyze with fixes (${n} frame${n === 1 ? '' : 's'})`;
  $('clearBar').hidden = !state.bar;
}

$('markBar').addEventListener('click', () => {
  if (!state) return;
  if (state.mode === 'bar') return setMode('view');
  if (state.inputs.barCm === null) {
    setHint('Enter the bar height above and press Analyze again before marking the bar.');
    return;
  }
  stopPlay();
  setMode('bar');
});
$('clearBar').addEventListener('click', async () => {
  state.bar = null;
  await reanalyze('Bar marks removed. The bar line is estimated from the bar height again.');
});
$('fixPoints').addEventListener('click', () => {
  if (!state) return;
  stopPlay();
  setMode(state.mode === 'fix' ? 'view' : 'fix');
});
$('applyFixes').addEventListener('click', () => reanalyze('Re-analyzed with your fixes.'));

function canvasPoint(e) {
  const r = canvas.getBoundingClientRect();
  return { x: ((e.clientX - r.left) / r.width) * canvas.width, y: ((e.clientY - r.top) / r.height) * canvas.height };
}

let drag = null; // {i, k}
canvas.addEventListener('pointerdown', async (e) => {
  if (!state || state.mode === 'view') return;
  const p = canvasPoint(e);
  if (state.mode === 'bar') {
    state.barTaps.push({ x: p.x / canvas.width, y: p.y / canvas.height });
    const n = state.barTaps.length;
    if (n === 1) setHint('Tap 2 of 3: the right end of the bar.');
    if (n === 2) setHint('Tap 3 of 3: the floor directly under one of the standards.');
    drawFrame(currentIndex());
    if (n === 3) {
      const [a, b, ground] = state.barTaps;
      state.bar = { a, b, ground };
      setMode('view');
      await reanalyze('Bar marked. Heights now use the bar for scale.');
    }
    return;
  }
  // Fix mode: grab the nearest point.
  const i = currentIndex();
  const f = state.frames[i];
  if (!f || !f.lms) return;
  const P = toPx(f.lms, canvas.width, canvas.height);
  let best = null, bestD = 30 * scaleOf();
  for (const k of FIXABLE) {
    const d = Math.hypot(P[k].x - p.x, P[k].y - p.y);
    if (d < bestD) { best = k; bestD = d; }
  }
  if (best === null) return;
  drag = { i, k: best };
  canvas.setPointerCapture(e.pointerId);
  e.preventDefault();
});
canvas.addEventListener('pointermove', (e) => {
  if (!drag) return;
  const p = canvasPoint(e);
  const nx = Math.min(1, Math.max(0, p.x / canvas.width)), ny = Math.min(1, Math.max(0, p.y / canvas.height));
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
  const prevMode = state.mode;
  state.mode = 'view';
  const items = [];
  for (const i of [...keep].sort((a, b) => a - b)) {
    const f = state.frames[i];
    if (!f || !f.lms) continue;
    await seekTo(f.t);
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height); // the plain frame, no drawing on it
    items.push({
      frame: i,
      videoTime: f.t,
      source: f.corrected ? 'corrected by hand' : 'model',
      image: canvas.toDataURL('image/jpeg', 0.9),
      keypoints: f.lms.map((p) => [Number(p.x.toFixed(5)), Number(p.y.toFixed(5)), Number((p.visibility ?? 1).toFixed(3))]),
    });
  }
  state.mode = prevMode;
  const data = {
    format: 'mediapipe-pose-33',
    note: 'keypoints are [x, y, visibility], x and y as a fraction of image width and height, in MediaPipe Pose landmark order.',
    video: state.inputs.file,
    imageSize: [canvas.width, canvas.height],
    athlete: { heightCm: state.inputs.heightCm, gender: state.inputs.gender },
    barCm: state.inputs.barCm,
    bar: state.bar,
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
const showHeight = (cm, units) => (units === 'ftin' ? `${ftIn(cm)} (${Math.round(cm)} cm)` : `${Math.round(cm)} cm`);
const plural = (n, word) => `${n} ${word}${n === 1 ? '' : 's'}`;
const signed = (v) => `${v >= 0 ? '+' : ''}${Math.round(v)} cm`;

function clearanceHtml(a, inputs) {
  const i = a.info;
  const vertical = `<div class="stats">
      <div><div class="l">Vertical jump (hip rise)</div><div class="n">${fmt(i.verticalJump, 0)} cm</div></div>
      <div><div class="l">Peak hip height</div><div class="n">${fmt(i.peakHipHeight, 0)} cm</div></div>
      ${i.barCm !== null ? `<div><div class="l">Bar</div><div class="n">${inputs.units === 'ftin' ? ftIn(i.barCm) : `${i.barCm} cm`}</div></div>
      <div><div class="l">Hips vs bar</div><div class="n">${signed(i.peakHipHeight - i.barCm)}</div></div>` : ''}
    </div>`;
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
  const rows = cl.parts.map((p) => `<tr><td>${p.label}</td><td class="num" style="color:${p.cm >= 0 ? 'var(--good)' : 'var(--bad)'}">${signed(p.cm)}</td></tr>`).join('');
  return `<section class="card">
    <h2>Vertical jump and bar clearance</h2>
    ${vertical}
    <p class="verdict" style="color:${verdictColor}">${verdictText}</p>
    ${said ? `<p class="note">You said the jump ${said}. ${agrees ? 'That matches the estimate.' : 'The estimate disagrees, so the tracking or bar line may be off. Try marking the bar or fixing points on the peak frames.'}</p>` : ''}
    <div class="table-wrap"><table>
      <thead><tr><th>Body part</th><th>Best height over the bar</th></tr></thead>
      <tbody>${rows}</tbody></table></div>
    <p class="note">Each body part has to be above the bar at some point in the air, so this compares each part's highest point with the bar. Joint points sit inside the body, so a few cm are taken off for the back of the body. ${cl.method === 'marked bar' ? 'Uses the bar you marked on the video.' : 'The bar line is estimated from the bar height. Mark the bar on the video for a more accurate check.'}</p>
  </section>`;
}

function reportHtml() {
  const { analysis: a, inputs, images, tracking } = state;
  const phases = a.phases.map((p) => `
    <div class="phase"><div class="l">${p.label} · ${p.weight}%</div>
      <div class="n">${p.score ?? 'n/a'}</div>
      <div class="bar"><i style="width:${p.score ?? 0}%;background:${color(p.score ?? 0)}"></i></div></div>`).join('');
  const rows = Object.values(a.metrics).map((m) => `
    <tr><td>${m.label}</td>
      <td class="num">${fmt(m.value, m.digits)}${m.unit && typeof m.value === 'number' ? ' ' + m.unit : ''}</td>
      <td class="num">${m.target ? (m.target[1] === Infinity ? `${m.target[0]} ${m.unit} or more` : `${m.target[0]}–${m.target[1]} ${m.unit}`) : 'Yes'}</td>
      <td class="num" style="color:${color(m.score ?? 0)}">${m.score ?? 'n/a'}</td></tr>`).join('');
  const tips = Object.values(a.metrics).filter((m) => m.tip).sort((x, y) => (x.score ?? 0) - (y.score ?? 0));
  const i = a.info;
  const caption = { curve: 'Curve', plant: 'Plant', takeoff: 'Takeoff', peak: 'Peak', landing: 'Landing' };
  const frames = Object.entries(images).map(([k, src]) => `<figure><img src="${src}" alt="${caption[k]} frame"><figcaption>${caption[k]}</figcaption></figure>`).join('');
  const slowNote = state.slow > 1 ? ` · slow motion ${state.slow}×` : '';
  return `
  <section class="card">
    <h2>Jump rating</h2>
    <div class="score-head">
      <div class="score-big" style="color:${color(a.overall)}">${a.overall}<small> / 100</small></div>
      <div><div class="rating">${a.rating}</div>
        <div class="note">${inputs.gender === 'female' ? 'Female' : 'Male'}, ${showHeight(inputs.heightCm, inputs.units)}${i.barCm !== null ? ` · bar ${showHeight(i.barCm, inputs.units)}` : ''} · takeoff leg: ${i.takeoffLeg}</div></div>
    </div>
    <div class="phases">${phases}</div>
  </section>
  ${clearanceHtml(a, inputs)}
  <section class="card">
    <h2>Key moments</h2>
    <div class="frames">${frames}</div>
  </section>
  <section class="card">
    <h2>Your numbers against your targets</h2>
    <div class="table-wrap"><table>
      <thead><tr><th>Measure</th><th>You</th><th>Target for you</th><th>Score</th></tr></thead>
      <tbody>${rows}</tbody></table></div>
    <h3>Also measured</h3>
    <div class="table-wrap"><table><tbody>
      <tr><td>Hip height at takeoff</td><td class="num">${fmt(i.hipHeightAtTakeoff, 0)} cm</td></tr>
      ${i.barCm !== null ? `
      <tr><td>Bar height as % of your height</td><td class="num">${fmt(i.barPercentOfHeight, 0)}%</td></tr>
      <tr><td>Lift needed from takeoff hip height to bar</td><td class="num">${fmt(i.liftNeededForBar, 0)} cm</td></tr>` : ''}
      <tr><td>Time in the air</td><td class="num">${fmt(i.flightTime, 2)} s</td></tr>
      <tr><td>Vertical jump implied by time in the air (cross-check)</td><td class="num">${fmt(i.riseFromFlightTime, 0)} cm</td></tr>
      <tr><td>Body rotation, plant to peak</td><td class="num">${fmt(i.rotation, 0)}°</td></tr>
    </tbody></table></div>
  </section>
  <section class="card">
    <h2>What to work on</h2>
    ${tips.length ? `<ol class="tiplist">${tips.map((m) => `<li><strong>${m.label}.</strong> ${m.tip}</li>`).join('')}</ol>` : '<p>Every measure is inside your target range. Nice jump.</p>'}
    <h3>Tracking quality</h3>
    <p class="note">Jumper found in ${tracking.foundPct}% of ${tracking.frames} frames (${tracking.quality} quality, ${state.fps} frames a second${slowNote}). ${plural(tracking.filled, 'frame')} filled in from their neighbours, ${plural(tracking.swapped, 'left/right mix-up')} fixed, ${plural(tracking.corrected, 'frame')} fixed by hand. Scale from ${i.scaleSource}.</p>
    <p class="note">Hip height stands in for your center of mass. These numbers are estimates from a single 2D video, not lab measurements. They assume a still camera. Targets are adjusted for your height and gender and are rounded from published elite high jump data, then widened for non-elite jumpers.</p>
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
function runAnalysis() {
  state.frames = applyCorrections(state.cleaned, state.corrections);
  // The analysis works in real seconds from the start of the clip.
  const timed = state.frames.map((f) => ({ ...f, t: (f.t - state.start) / state.slow }));
  state.analysis = analyzeJump(timed, {
    width: canvas.width, height: canvas.height,
    heightCm: state.inputs.heightCm, gender: state.inputs.gender,
    barCm: state.inputs.barCm, bar: state.bar,
  });
  const found = state.frames.filter((f) => f.lms).length;
  state.tracking = {
    quality: state.inputs.quality,
    frames: state.frames.length,
    foundPct: Math.round((100 * found) / Math.max(1, state.frames.length)),
    filled: state.frames.filter((f) => f.filled).length,
    swapped: state.frames.filter((f) => f.swapped).length,
    corrected: state.frames.filter((f) => f.corrected).length,
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
    setMode(state.mode === 'bar' ? 'view' : state.mode);
    await showFrame(Math.min(currentIndex(), state.frames.length - 1));
    const a = state.analysis;
    setStatus(`${message ? message + ' ' : ''}Rating ${a.overall}/100 (${a.rating}).`);
  } catch (err) {
    console.error(err);
    setStatus(err.message || String(err), true);
  }
}

async function showResults(posesPerFrame, start, fps, slow, inputs) {
  const q = QUALITY[inputs.quality];
  state = {
    posesPerFrame, start, fps, slow, inputs,
    cleaned: cleanTracking(posesPerFrame, { smoothRadius: q.smoothRadius, maxGap: Math.round(fps / 6) }),
    corrections: {}, frames: null, analysis: null, images: {}, bar: null, mode: 'view', barTaps: [],
  };
  setStatus('Measuring the jump…');
  runAnalysis();
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
  setStatus(`Done. Rating ${a.overall}/100 (${a.rating}).${clText} Press Play to watch the annotated video.`);
}

// Self-test: open the page with #selftest, then call window.hjSelfTest(videoFile)
// to run a synthetic jump through the tracking cleanup, drawing and report.
if (location.hash === '#selftest') {
  window.hjSelfTest = async (file, { barCm = 170, bar = null, result = 'cleared' } = {}) => {
    const { syntheticJump, W, H } = await import('./test/synthetic.js');
    await loadVideo(file);
    canvas.width = W; canvas.height = H;
    const fps = 30;
    const posesPerFrame = syntheticJump({ lean: 15, launch: 47, risePx: 140, approachPxPerFrame: 37 })
      .map((f) => ({ t: Math.min(f.t, video.duration - 0.05), poses: [f.lms] }));
    await showResults(posesPerFrame, 0, fps, 1, { heightCm: 180, gender: 'male', barCm, result, quality: 'standard', startSeconds: 0, file: file.name });
    if (bar) { state.bar = bar; await reanalyze('Bar marked.'); }
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

    const q = QUALITY[quality];
    setStatus(`Loading the ${quality === 'high' ? 'high-accuracy ' : ''}pose model (first time takes a few seconds)…`);
    await getLandmarker(q.model);

    $('progress').hidden = false;
    const posesPerFrame = await extractPoses(start, q.fps, q.model, (p) => {
      $('progress').value = p;
      setStatus(`Finding joints in each frame… ${Math.round(p * 100)}%`);
    });
    $('progress').hidden = true;

    await showResults(posesPerFrame, start, q.fps, slow, { heightCm, gender, barCm, result, quality, units: $('units').value, startSeconds: start, file: file.name });
  } catch (err) {
    console.error(err);
    $('progress').hidden = true;
    setStatus(err.message || String(err), true);
  } finally {
    $('go').disabled = false;
  }
});
