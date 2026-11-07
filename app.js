import { analyzeJump, angleAt, mid, toPx, phaseAt, LM } from './analysis.js';

const TV = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14';
const MODEL = 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task';
const FPS = 30;
const MAX_SECONDS = 15;

const $ = (id) => document.getElementById(id);
const video = $('video');
const canvas = $('canvas');
const ctx = canvas.getContext('2d');

let landmarker = null;
let lastTs = 0; // VIDEO mode needs ever-increasing timestamps, across runs too
let state = null; // {frames, analysis, start, inputs, images}

// ---------- inputs ----------
$('units').addEventListener('change', () => {
  const ft = $('units').value === 'ftin';
  $('heightCm').hidden = ft;
  $('heightFt').hidden = !ft;
  $('heightIn').hidden = !ft;
  $('barCm').hidden = ft;
  $('barFt').hidden = !ft;
  $('barIn').hidden = !ft;
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

// ---------- pose model ----------
async function getLandmarker() {
  if (landmarker) return landmarker;
  const { PoseLandmarker, FilesetResolver } = await import(`${TV}/vision_bundle.mjs`);
  const fileset = await FilesetResolver.forVisionTasks(`${TV}/wasm`);
  const opts = (delegate) => ({
    baseOptions: { modelAssetPath: MODEL, delegate },
    runningMode: 'VIDEO',
    numPoses: 1,
    minPoseDetectionConfidence: 0.5,
    minTrackingConfidence: 0.5,
  });
  try {
    landmarker = await PoseLandmarker.createFromOptions(fileset, opts('GPU'));
  } catch {
    landmarker = await PoseLandmarker.createFromOptions(fileset, opts('CPU'));
  }
  return landmarker;
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

async function extractPoses(start, onProgress) {
  const lm = await getLandmarker();
  const end = Math.min(video.duration, start + MAX_SECONDS);
  const n = Math.max(0, Math.floor((end - start) * FPS));
  const frames = [];
  for (let i = 0; i < n; i++) {
    const t = start + i / FPS;
    await seekTo(t);
    lastTs = Math.max(lastTs + 1, Math.round(performance.now()));
    const r = lm.detectForVideo(video, lastTs);
    const pose = r.landmarks && r.landmarks[0];
    frames.push({ t, lms: pose ? pose.map((p) => ({ x: p.x, y: p.y, visibility: p.visibility })) : null });
    if (i % 3 === 0) onProgress((i + 1) / n);
    // Let the page breathe so the progress bar updates.
    if (i % 10 === 0) await new Promise((r2) => setTimeout(r2, 0));
  }
  onProgress(1);
  return frames;
}

// ---------- drawing ----------
const BONES = [
  [11, 12], [11, 13], [13, 15], [12, 14], [14, 16], [11, 23], [12, 24], [23, 24],
  [23, 25], [25, 27], [24, 26], [26, 28], [27, 29], [29, 31], [27, 31], [28, 30], [30, 32], [28, 32],
];
const ANCHORS = [11, 12, 23, 24, 25, 26, 27, 28];

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

function drawFrame(i) {
  const W = canvas.width, H = canvas.height;
  ctx.drawImage(video, 0, 0, W, H);
  if (!state) return;
  const s = Math.max(W, H) / 900;
  const { frames, analysis } = state;
  const kf = analysis.keyFrames;
  const phase = phaseAt(i, kf);
  pill(`${phase}  ·  ${(frames[i]?.t ?? 0).toFixed(2)} s`, 8 * s, 18 * s, s, '#ffffff');

  const f = frames[i];
  if (!f || !f.lms) return;
  const P = toPx(f.lms, W, H);

  ctx.lineCap = 'round';
  ctx.lineWidth = 4 * s;
  ctx.strokeStyle = 'rgba(255,255,255,0.9)';
  for (const [a, b] of BONES) {
    ctx.beginPath(); ctx.moveTo(P[a].x, P[a].y); ctx.lineTo(P[b].x, P[b].y); ctx.stroke();
  }
  for (const k of ANCHORS) {
    ctx.beginPath();
    ctx.arc(P[k].x, P[k].y, 6.5 * s, 0, Math.PI * 2);
    ctx.fillStyle = '#ffd23f'; ctx.fill();
    ctx.lineWidth = 2 * s; ctx.strokeStyle = '#000'; ctx.stroke();
  }

  // Joint angles: knee = hip-knee-ankle, hip = shoulder-hip-knee.
  const joints = [
    ['L knee', LM.lHip, LM.lKnee, LM.lAnkle], ['R knee', LM.rHip, LM.rKnee, LM.rAnkle],
    ['L hip', LM.lShoulder, LM.lHip, LM.lKnee], ['R hip', LM.rShoulder, LM.rHip, LM.rKnee],
  ];
  for (const [name, a, b, c] of joints) {
    const ang = angleAt(P[a], P[b], P[c]);
    if (Number.isFinite(ang)) pill(`${name} ${Math.round(ang)}°`, P[b].x + 10 * s, P[b].y, s);
  }

  // Body lean line (ankles to shoulders) with a vertical reference, up to takeoff.
  if (i <= kf.takeoff) {
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
}

function frameIndexAt(t) {
  const i = Math.round((t - state.start) * FPS);
  return Math.max(0, Math.min(state.frames.length - 1, i));
}

async function showFrame(i) {
  await seekTo(state.frames[i].t);
  drawFrame(i);
  $('scrub').value = String(i);
}

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
  let i = Number($('scrub').value);
  if (i >= state.frames.length - 1) i = 0;
  await seekTo(state.frames[i].t);
  playThrough();
});
$('scrub').addEventListener('input', () => { stopPlay(); showFrame(Number($('scrub').value)); });

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
  const btn = $('dlVideo');
  btn.disabled = true; btn.textContent = 'Recording… (plays once)';
  stopPlay();
  await showFrame(0);
  const rec = new MediaRecorder(canvas.captureStream(FPS), { mimeType: mime, videoBitsPerSecond: 6e6 });
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
  const { analysis, inputs } = state;
  const out = { inputs, overall: analysis.overall, rating: analysis.rating, phases: analysis.phases, metrics: analysis.metrics, info: analysis.info, keyFrames: analysis.keyFrames, sources: analysis.sources };
  download(new Blob([JSON.stringify(out, null, 2)], { type: 'application/json' }), 'metrics.json');
});

$('dlReport').addEventListener('click', async () => {
  const css = await fetch('style.css').then((r) => r.text()).catch(() => '');
  const html = `<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>High jump report</title><style>${css}</style></head><body><main style="padding-top:24px">${reportHtml()}</main></body></html>`;
  download(new Blob([html], { type: 'text/html' }), 'high-jump-report.html');
});

// ---------- report ----------
const color = (s) => (s >= 85 ? 'var(--good)' : s >= 60 ? 'var(--mid)' : 'var(--bad)');
const fmt = (v, d) => (typeof v === 'number' ? (Number.isFinite(v) ? v.toFixed(d) : 'n/a') : v);

function reportHtml() {
  const { analysis: a, inputs, images } = state;
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
  return `
  <section class="card">
    <h2>Jump rating</h2>
    <div class="score-head">
      <div class="score-big" style="color:${color(a.overall)}">${a.overall}<small> / 100</small></div>
      <div><div class="rating">${a.rating}</div>
        <div class="note">${inputs.gender === 'female' ? 'Female' : 'Male'}, ${inputs.heightCm} cm${i.barCm !== null ? ` · bar ${i.barCm} cm` : ''} · takeoff leg: ${i.takeoffLeg}</div></div>
    </div>
    <div class="phases">${phases}</div>
  </section>
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
      <tr><td>Peak hip height (estimate)</td><td class="num">${fmt(i.peakHipHeight, 0)} cm</td></tr>
      ${i.barCm !== null ? `
      <tr><td>Bar height as % of your height</td><td class="num">${fmt(i.barPercentOfHeight, 0)}%</td></tr>
      <tr><td>Lift needed from takeoff hip height to bar</td><td class="num">${fmt(i.liftNeededForBar, 0)} cm</td></tr>` : ''}
      <tr><td>Time in the air</td><td class="num">${fmt(i.flightTime, 2)} s</td></tr>
      <tr><td>Rise implied by time in the air</td><td class="num">${fmt(i.riseFromFlightTime, 0)} cm</td></tr>
      <tr><td>Body rotation, plant to peak</td><td class="num">${fmt(i.rotation, 0)}°</td></tr>
    </tbody></table></div>
  </section>
  <section class="card">
    <h2>What to work on</h2>
    ${tips.length ? `<ol class="tiplist">${tips.map((m) => `<li><strong>${m.label}.</strong> ${m.tip}</li>`).join('')}</ol>` : '<p>Every measure is inside your target range. Nice jump.</p>'}
    ${i.barCm === null ? '<p class="note">Add the bar height to see whether your hips peaked over the bar.</p>' : ''}
    <p class="note">Hip height stands in for your center of mass. In a good flop the body drapes around the bar, so a small negative hip-vs-bar number can still be a clearance.</p>
    <p class="note">These numbers are estimates from a single 2D video, not lab measurements. They assume the camera is roughly side-on to the bar and still. Your height sets the scale for distances; targets are adjusted for your height and gender and are rounded from published elite high jump data, then widened for non-elite jumpers.</p>
    <p class="sources">Sources:<br>${a.sources.join('<br>')}</p>
  </section>`;
}

async function captureKeyFrames() {
  const images = {};
  for (const [k, idx] of Object.entries(state.analysis.keyFrames)) {
    await seekTo(state.frames[idx].t);
    drawFrame(idx);
    images[k] = canvas.toDataURL('image/jpeg', 0.82);
  }
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

// ---------- main flow ----------
async function showResults(frames, start, inputs) {
  setStatus('Measuring the jump…');
  const analysis = analyzeJump(frames, { width: canvas.width, height: canvas.height, heightCm: inputs.heightCm, gender: inputs.gender, barCm: inputs.barCm });
  state = { frames, analysis, start, inputs, images: {} };
  $('viewer').hidden = false;
  $('scrub').max = String(frames.length - 1);
  buildKeyButtons();
  state.images = await captureKeyFrames();
  $('report').innerHTML = reportHtml();
  await showFrame(analysis.keyFrames.plant);
  setStatus(`Done. Rating ${analysis.overall}/100 (${analysis.rating}). Press Play to watch the annotated video.`);
}

// Self-test: open the page with #selftest, then call window.hjSelfTest(videoFile)
// to run a synthetic jump through the drawing and report screens.
if (location.hash === '#selftest') {
  window.hjSelfTest = async (file) => {
    const { syntheticJump, W, H } = await import('./test/synthetic.js');
    await loadVideo(file);
    canvas.width = W; canvas.height = H;
    const frames = syntheticJump({ lean: 15, launch: 47, risePx: 140, approachPxPerFrame: 37 })
      .map((f) => ({ ...f, t: Math.min(f.t, video.duration - 0.05) }));
    await showResults(frames, 0, { heightCm: 180, gender: 'male', barCm: 170, startSeconds: 0, file: file.name });
    return state.analysis.overall;
  };
}

$('form').addEventListener('submit', async (e) => {
  e.preventDefault();
  const file = $('file').files[0];
  const heightCm = readHeightCm();
  const gender = $('gender').value;
  const barCm = readBarCm();
  const start = Math.max(0, Number($('start').value) || 0);
  if (!file) return setStatus('Choose a video first.', true);
  if (!(heightCm >= 100 && heightCm <= 230)) return setStatus('Enter a height between 100 and 230 cm (3 ft 4 in to 7 ft 6 in).', true);
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

    setStatus('Loading the pose model (first time takes a few seconds)…');
    await getLandmarker();

    $('progress').hidden = false;
    const frames = await extractPoses(start, (p) => {
      $('progress').value = p;
      setStatus(`Finding joints in each frame… ${Math.round(p * 100)}%`);
    });
    $('progress').hidden = true;

    await showResults(frames, start, { heightCm, gender, barCm, startSeconds: start, file: file.name });
  } catch (err) {
    console.error(err);
    $('progress').hidden = true;
    setStatus(err.message || String(err), true);
  } finally {
    $('go').disabled = false;
  }
});
