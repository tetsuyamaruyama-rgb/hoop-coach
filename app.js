/* バスケコーチ Kids — on-device shooting / dribble / defense analysis.
 * Pose runs locally via MoveNet (TensorFlow.js). No video ever leaves the phone.
 * Feedback is rule-based (offline) and tuned gently for ~7-year-olds.
 * Practice log + weekly graph stored in localStorage (on-device only).
 */
'use strict';

if ('serviceWorker' in navigator) {
  window.addEventListener('load', () => navigator.serviceWorker.register('sw.js').catch(() => {}));
}
// warm up the pose model early so picking a video isn't blocked by a cold load
window.addEventListener('load', () => { setTimeout(() => getDetector().catch(() => {}), 300); });

// ---- helpers ----
const $ = (id) => document.getElementById(id);
const PROC_W = 432;

function showScreen(name) {
  document.querySelectorAll('.screen').forEach((s) => s.classList.remove('active'));
  $('screen-' + name).classList.add('active');
}
function angle(a, b, c) {
  if (!a || !b || !c) return null;
  const ab = { x: a.x - b.x, y: a.y - b.y }, cb = { x: c.x - b.x, y: c.y - b.y };
  const dot = ab.x * cb.x + ab.y * cb.y, cross = ab.x * cb.y - ab.y * cb.x;
  return Math.abs((Math.atan2(cross, dot) * 180) / Math.PI);
}
const dist = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);
const mid = (a, b) => ({ x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 });
function kp(f, name, m = 0.3) { const k = f.byName[name]; return k && k.score >= m ? k : null; }
const avg = (a) => (a.length ? a.reduce((s, x) => s + x, 0) / a.length : null);
function std(a) { if (a.length < 2) return 0; const m = avg(a); return Math.sqrt(avg(a.map((x) => (x - m) ** 2))); }
function series(frames, fn) { return frames.map(fn).filter((v) => v != null); }
function torsoLen(f) {
  const ls = kp(f, 'left_shoulder'), rs = kp(f, 'right_shoulder'), lh = kp(f, 'left_hip'), rh = kp(f, 'right_hip');
  if (!ls || !rs || !lh || !rh) return null;
  return dist(mid(ls, rs), mid(lh, rh));
}

// ---- detector ----
let detector = null;
async function getDetector() {
  if (detector) return detector;
  await tf.setBackend('webgl'); await tf.ready();
  detector = await poseDetection.createDetector(poseDetection.SupportedModels.MoveNet, {
    modelType: poseDetection.movenet.modelType.SINGLEPOSE_THUNDER,
  });
  return detector;
}

// ---- frame sampling ----
// Create a muted, inline, on-DOM video. iOS only decodes/plays video that is
// attached and muted — an off-screen detached element never fires its events.
function makeScanVideo(file) {
  return new Promise((resolve, reject) => {
    const v = document.createElement('video');
    v.muted = true; v.defaultMuted = true; v.playsInline = true;
    v.setAttribute('muted', ''); v.setAttribute('playsinline', ''); v.setAttribute('webkit-playsinline', '');
    v.preload = 'auto';
    v.style.cssText = 'position:fixed;left:0;top:0;width:2px;height:2px;opacity:0.01;pointer-events:none;z-index:-1';
    v.src = URL.createObjectURL(file);
    document.body.appendChild(v);
    let done = false; const ok = () => { if (!done) { done = true; resolve(v); } };
    v.onloadedmetadata = ok;
    v.onerror = () => { if (!done) { done = true; reject(new Error('video-load')); } };
    setTimeout(ok, 4000); // proceed even if the event is flaky on iOS
  });
}
// Scan by PLAYING the clip and sampling frames as they're presented (iOS-safe),
// storing ONLY keypoints (tiny memory). Never hangs: hard safety timeout.
async function analyzeFile(file) {
  const det = await getDetector();
  const video = await makeScanVideo(file);
  let dur = video.duration;
  if (!dur || !isFinite(dur)) {
    await new Promise((r) => { video.ondurationchange = r; setTimeout(r, 1500); });
    dur = video.duration;
  }
  if (!isFinite(dur) || dur <= 0) dur = 8;
  const vw = video.videoWidth || 432, vh = video.videoHeight || 768;
  const w = PROC_W, h = Math.round(PROC_W * vh / vw);
  const cv = document.createElement('canvas'); cv.width = w; cv.height = h;
  const ctx = cv.getContext('2d');
  const targetCount = Math.max(8, Math.min(45, Math.round(dur * 8)));
  const minGap = dur / targetCount;
  const frames = [];
  const sampled = new Set(); // fixed time-grid bins → consistent samples across runs
  let busy = false, finished = false;

  await new Promise((resolve) => {
    const finish = () => { if (finished) return; finished = true; $('progressBar').style.width = '100%'; resolve(); };
    const sample = async () => {
      if (finished || busy) return;
      const t = video.currentTime;
      const bin = Math.floor(t / minGap); // map to a fixed time slot; one sample per slot
      if (sampled.has(bin)) return;
      sampled.add(bin);
      busy = true;
      try {
        ctx.drawImage(video, 0, 0, w, h);
        const poses = await det.estimatePoses(cv, { maxPoses: 1, flipHorizontal: false });
        if (poses[0] && poses[0].keypoints.filter((k) => k.score > 0.3).length >= 6) {
          const byName = {}; poses[0].keypoints.forEach((k) => (byName[k.name] = k));
          frames.push({ t: bin * minGap, byName });
        }
      } catch (e) {}
      busy = false;
      $('progressBar').style.width = Math.min(99, Math.round((t / dur) * 100)) + '%';
    };
    const useRVFC = typeof video.requestVideoFrameCallback === 'function';
    const loop = async () => { await sample(); if (!finished && useRVFC) video.requestVideoFrameCallback(loop); };
    video.onended = finish;
    setTimeout(finish, dur * 1000 + 9000); // absolute safety: never hang
    video.play().then(() => {
      if (useRVFC) video.requestVideoFrameCallback(loop);
      else video.ontimeupdate = sample; // fallback for old iOS
    }).catch(() => finish());
  });

  const url = video.src; // keep the object URL alive for result playback
  try { video.pause(); } catch (e) {}
  try { document.body.removeChild(video); } catch (e) {}
  if (frames.length < 4) { URL.revokeObjectURL(url); throw new Error('no-pose'); }
  return { frames, w, h, url };
}

// ---- side / key-frame pickers ----
function sideByConfidence(frames) {
  let l = 0, r = 0;
  const names = ['shoulder', 'elbow', 'wrist', 'hip', 'knee', 'ankle'];
  for (const f of frames) for (const n of names) {
    l += (f.byName['left_' + n]?.score || 0); r += (f.byName['right_' + n]?.score || 0);
  }
  return r >= l ? 'right' : 'left';
}
function deepestKneeIdx(frames, side) {
  let idx = 0, m = Infinity;
  frames.forEach((f, i) => {
    const a = angle(kp(f, side + '_hip'), kp(f, side + '_knee'), kp(f, side + '_ankle'));
    if (a != null && a < m) { m = a; idx = i; }
  });
  return idx;
}

// ---- analyzers: each returns {keyIdx, side, metrics:[{good,value,label,praise,tip}]} ----
function analyzeShooting({ frames }) {
  // shooting hand = wrist that reaches the highest point
  let side = 'right', minY = Infinity;
  for (const f of frames) for (const s of ['left', 'right']) {
    const wr = kp(f, s + '_wrist'); if (wr && wr.y < minY) { minY = wr.y; side = s; }
  }
  const S = (n) => side + '_' + n;
  let relIdx = 0, mY = Infinity;
  frames.forEach((f, i) => { const wr = kp(f, S('wrist')); if (wr && wr.y < mY) { mY = wr.y; relIdx = i; } });
  let loadIdx = 0, mK = Infinity;
  for (let i = 0; i <= relIdx; i++) {
    const a = angle(kp(frames[i], S('hip')), kp(frames[i], S('knee')), kp(frames[i], S('ankle')));
    if (a != null && a < mK) { mK = a; loadIdx = i; }
  }
  const rel = frames[relIdx], load = frames[loadIdx], metrics = [];
  if (isFinite(mK)) metrics.push({ good: mK < 162, value: Math.round(mK) + '°', label: 'ひざのまげ',
    praise: ['🦵', 'ひざが よくまがってる！', 'あしの ちからで とおくまで とばせるよ'],
    tip: ['🦵', 'ひざを もうすこし まげよう', 'とぶまえに ひざをまげると ちからが でるよ'] });
  // peak elbow extension across the shooting phase (a brief moment the coarse
  // scan can miss at the exact release frame) — answers "did he straighten it?"
  let elb = null;
  for (let i = loadIdx; i < frames.length; i++) {
    const a = angle(kp(frames[i], S('shoulder')), kp(frames[i], S('elbow')), kp(frames[i], S('wrist')));
    if (a != null && (elb == null || a > elb)) elb = a;
  }
  if (elb != null) metrics.push({ good: elb > 150, value: Math.round(elb) + '°', label: 'うでの のび',
    praise: ['💪', 'うでが まっすぐ のびてる！', 'リリースが きれいだよ'],
    tip: ['💪', 'うでを うえに ピンと のばそう', 'ボールを おすとき ひじを のばしきろう'] });
  const wr = kp(rel, S('wrist')), nose = kp(rel, 'nose') || kp(rel, S('shoulder'));
  if (wr && nose) metrics.push({ good: wr.y < nose.y, value: wr.y < nose.y ? 'たかい' : 'ひくい', label: 'フォロースルー',
    praise: ['🙌', 'てが たかく あがってる！', 'バイバイの かたちが できてるね'],
    tip: ['🙌', 'なげたあと てを たかく！', 'ゴールに「バイバイ」してみよう'] });
  const la = kp(load, 'left_ankle'), ra = kp(load, 'right_ankle'), ls = kp(load, 'left_shoulder'), rs = kp(load, 'right_shoulder');
  if (la && ra && ls && rs) { const ratio = dist(la, ra) / (dist(ls, rs) || 1); const good = ratio > 0.55 && ratio < 1.9;
    metrics.push({ good, value: good ? 'バッチリ' : 'なおそう', label: 'あしのはば',
      praise: ['🦶', 'あしの はばが ちょうどいい！', 'バランスが とれてるよ'],
      tip: ['🦶', 'あしを かたはばに ひらこう', 'あしを ひらくと ぐらぐら しないよ'] }); }
  return { keyIdx: relIdx, side, metrics };
}

function analyzeDribble({ frames }) {
  const side = sideByConfidence(frames);
  const S = (n) => side + '_' + n;
  const keyIdx = deepestKneeIdx(frames, side);
  const metrics = [];
  // 1) low athletic stance (knee bend, averaged)
  const knees = series(frames, (f) => angle(kp(f, S('hip')), kp(f, S('knee')), kp(f, S('ankle'))));
  if (knees.length) { const a = avg(knees); metrics.push({ good: a < 168, value: Math.round(a) + '°', label: 'ひくいしせい',
    praise: ['🔽', 'こしが ひくくて じょうず！', 'ひくいと あいてに とられにくいよ'],
    tip: ['🔽', 'もうすこし こしを おとそう', 'ひざをまげて ひくく かまえよう'] }); }
  // 2) head up (upright torso, not hunched over the ball)
  const leans = series(frames, (f) => { const t = torsoLen(f); if (!t) return null;
    const ls = kp(f, 'left_shoulder'), rs = kp(f, 'right_shoulder'), lh = kp(f, 'left_hip'), rh = kp(f, 'right_hip');
    if (!ls || !rs || !lh || !rh) return null; return Math.abs(mid(ls, rs).x - mid(lh, rh).x) / t; });
  if (leans.length) { const lean = avg(leans); metrics.push({ good: lean < 0.5, value: lean < 0.5 ? 'まえむき' : 'うつむき', label: 'あたまアップ',
    praise: ['👀', 'まえを みれてる！', 'ボールを みないで ドリブルできてるね'],
    tip: ['👀', 'かおを あげよう', 'ボールを みないで まえを みる れんしゅうを'] }); }
  // 3) rhythm — active hand moves up & down (normalized vertical motion)
  const t0 = avg(series(frames, torsoLen)) || 1;
  const wy = series(frames, (f) => kp(f, S('wrist'))?.y);
  if (wy.length >= 4) { const amp = std(wy) / t0; metrics.push({ good: amp > 0.08, value: amp > 0.08 ? 'いいリズム' : 'よわい', label: 'リズム',
    praise: ['🥁', 'いい リズムで ついてる！', 'てを つよく うごかせてるね'],
    tip: ['🥁', 'もっと つよく つこう', 'ボールを ゆかに つよく たたきつけよう'] }); }
  return { keyIdx, side, metrics };
}

function analyzeDefense({ frames }) {
  const side = sideByConfidence(frames);
  const S = (n) => side + '_' + n;
  const keyIdx = deepestKneeIdx(frames, side);
  const metrics = [];
  // 1) low stance (most important for defense)
  const knees = series(frames, (f) => angle(kp(f, S('hip')), kp(f, S('knee')), kp(f, S('ankle'))));
  if (knees.length) { const a = avg(knees); metrics.push({ good: a < 160, value: Math.round(a) + '°', label: 'こしを おとす',
    praise: ['🔽', 'こしが ひくい！ いいかまえ！', 'ひくいと はやく うごけるよ'],
    tip: ['🔽', 'もっと こしを おとそう', 'ひざを ぐっと まげて ひくく かまえよう'] }); }
  // 2) wide base
  const widths = series(frames, (f) => { const la = kp(f, 'left_ankle'), ra = kp(f, 'right_ankle'), ls = kp(f, 'left_shoulder'), rs = kp(f, 'right_shoulder');
    if (!la || !ra || !ls || !rs) return null; return dist(la, ra) / (dist(ls, rs) || 1); });
  if (widths.length) { const r = avg(widths); metrics.push({ good: r > 1.05, value: r > 1.05 ? 'ひろい' : 'せまい', label: 'あしのはば',
    praise: ['🦶', 'あしが ひろくて あんてい！', 'バランスばっちりだね'],
    tip: ['🦶', 'あしを もっと ひろげよう', 'かたはばより ひろく すると ぐらつかないよ'] }); }
  // 3) hands up / active
  const handsUp = frames.filter((f) => { const h = kp(f, S('hip')); const w = kp(f, S('wrist')); return h && w && w.y < h.y; }).length;
  const ratio = frames.length ? handsUp / frames.length : 0;
  metrics.push({ good: ratio > 0.4, value: ratio > 0.4 ? 'あげてる' : 'さげてる', label: 'てを あげる',
    praise: ['🙌', 'てを あげて まもれてる！', 'てを あげると パスを とめやすいよ'],
    tip: ['🙌', 'てを あげよう', 'りょうてを ひろげて あいてを とめよう'] });
  return { keyIdx, side, metrics };
}

const ANALYZERS = { shoot: analyzeShooting, dribble: analyzeDribble, defense: analyzeDefense };
const MODE_LABEL = { shoot: '🎯 シュート', dribble: '⛹️ ドリブル', defense: '🛡️ ディフェンス' };
const MODE_ERR = {
  shoot: 'からだが よく うつってなかったみたい。よこ から、ぜんしんが うつるように とってね。',
  dribble: 'からだが よく うつってなかったみたい。ぜんしんが うつるように とってね。',
  defense: 'からだが よく うつってなかったみたい。ぜんしんが うつるように とってね。',
};

// ---- render ----
// full-body skeleton drawn as an overlay on the result video, at the key frame
const SKELETON = [
  ['left_shoulder', 'right_shoulder'], ['left_hip', 'right_hip'],
  ['left_shoulder', 'left_hip'], ['right_shoulder', 'right_hip'],
  ['left_shoulder', 'left_elbow'], ['left_elbow', 'left_wrist'],
  ['right_shoulder', 'right_elbow'], ['right_elbow', 'right_wrist'],
  ['left_hip', 'left_knee'], ['left_knee', 'left_ankle'],
  ['right_hip', 'right_knee'], ['right_knee', 'right_ankle'],
];
const POINTS = ['nose', 'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
  'left_wrist', 'right_wrist', 'left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle'];
function drawOverlayMarks(ov, frame, srcW, srcH) {
  const ctx = ov.getContext('2d');
  ov.width = ov.clientWidth; ov.height = ov.clientHeight;
  ctx.clearRect(0, 0, ov.width, ov.height);
  if (!ov.width || !ov.height) return;
  const sx = ov.width / srcW, sy = ov.height / srcH;
  const P = (n) => kp(frame, n, 0.3);
  ctx.lineWidth = Math.max(3, ov.width / 110); ctx.strokeStyle = '#ff7a18'; ctx.lineCap = 'round';
  for (const [a, b] of SKELETON) {
    const ka = P(a), kb = P(b);
    if (ka && kb) { ctx.beginPath(); ctx.moveTo(ka.x * sx, ka.y * sy); ctx.lineTo(kb.x * sx, kb.y * sy); ctx.stroke(); }
  }
  const r = Math.max(5, ov.width / 55);
  for (const n of POINTS) {
    const k = P(n);
    if (k) { ctx.fillStyle = '#fff'; ctx.strokeStyle = '#e85d04'; ctx.lineWidth = Math.max(2, ov.width / 220);
      ctx.beginPath(); ctx.arc(k.x * sx, k.y * sy, r, 0, 7); ctx.fill(); ctx.stroke(); }
  }
}
// play the clip, then freeze at the key frame with the marks drawn on each joint
function setupResultVideo(data, res) {
  const vid = $('resultVideo'), ov = $('overlayCanvas');
  const keyFrame = data.frames[res.keyIdx], keyTime = keyFrame.t;
  let frozen = false;
  const draw = () => drawOverlayMarks(ov, keyFrame, data.w, data.h);
  const clear = () => { ov.width = ov.clientWidth; ov.height = ov.clientHeight; ov.getContext('2d').clearRect(0, 0, ov.width, ov.height); };
  const freeze = () => { if (frozen) return; frozen = true; try { vid.pause(); } catch (e) {}
    if (Math.abs(vid.currentTime - keyTime) > 0.05) vid.currentTime = keyTime; else draw(); };
  vid.ontimeupdate = () => { if (!frozen && vid.currentTime >= keyTime) freeze(); };
  vid.onseeked = () => { if (frozen) draw(); };
  vid.onended = () => freeze();
  $('replayBtn').onclick = () => { frozen = false; clear(); try { vid.currentTime = 0; } catch (e) {}
    vid.play().catch(() => { frozen = true; vid.currentTime = keyTime; }); };
  vid.src = data.url;
  try { vid.currentTime = 0; } catch (e) {}
  clear();
  // autoplay through once; if blocked, just show the key frame with marks
  vid.play().catch(() => { frozen = true; vid.currentTime = keyTime; });
}
function fbCard(kind, [ico, title, sub]) {
  const d = document.createElement('div'); d.className = 'fb ' + kind;
  d.innerHTML = `<div class="ico">${ico}</div><div class="txt"><b>${title}</b><span>${sub}</span></div>`;
  return d;
}
function showResult(mode, res, data) {
  const good = res.metrics.filter((m) => m.good).length;
  const stars = good >= 3 ? 3 : good === 2 ? 2 : 1;
  $('modeTag').textContent = MODE_LABEL[mode];
  $('scoreStars').textContent = '⭐'.repeat(stars);
  const goods = res.metrics.filter((m) => m.good), bads = res.metrics.filter((m) => !m.good);
  const fb = $('feedback'); fb.innerHTML = '';
  if (goods[0]) fb.appendChild(fbCard('good', goods[0].praise));
  else if (res.metrics[0]) fb.appendChild(fbCard('good', ['🏀', 'チャレンジ えらい！', 'やってみる ことが だいじだよ']));
  if (bads[0]) fb.appendChild(fbCard('tip', bads[0].tip));
  else fb.appendChild(fbCard('good', ['🎉', 'かんぺき！', 'この ちょうしで れんしゅう しよう！']));
  const mWrap = $('metrics'); mWrap.innerHTML = '';
  res.metrics.forEach((m) => { const d = document.createElement('div'); d.className = 'metric ' + (m.good ? 'ok' : 'no');
    d.innerHTML = `<div class="v">${m.good ? '◎' : '△'} ${m.value}</div><div class="l">${m.label}</div>`; mWrap.appendChild(d); });
  saveSession(mode, stars, good, res.metrics.length);
  showScreen('result');
  setupResultVideo(data, res); // after the screen is visible so the video has a size
}

// ---- practice log (localStorage, on-device) ----
const LOG_KEY = 'hoopLog_v1';
function loadLog() { try { return JSON.parse(localStorage.getItem(LOG_KEY)) || []; } catch { return []; } }
function saveSession(mode, stars, good, total) {
  const log = loadLog();
  log.push({ date: new Date().toISOString(), mode, stars, good, total });
  try { localStorage.setItem(LOG_KEY, JSON.stringify(log.slice(-500))); } catch {}
}
function dayKey(d) { return `${d.getFullYear()}-${d.getMonth() + 1}-${d.getDate()}`; }
function drawWeek(canvas, log) {
  const cssW = canvas.clientWidth || 320; const dpr = window.devicePixelRatio || 1;
  canvas.width = cssW * dpr; canvas.height = 160 * dpr;
  const ctx = canvas.getContext('2d'); ctx.scale(dpr, dpr);
  const W = cssW, H = 160, days = [];
  const today = new Date(); today.setHours(0, 0, 0, 0);
  for (let i = 6; i >= 0; i--) { const d = new Date(today); d.setDate(today.getDate() - i); days.push(d); }
  const counts = days.map((d) => log.filter((r) => dayKey(new Date(r.date)) === dayKey(d)).length);
  const maxC = Math.max(1, ...counts);
  const padB = 26, padT = 10, n = 7, gap = 12;
  const bw = (W - gap * (n + 1)) / n;
  const names = ['日', '月', '火', '水', '木', '金', '土'];
  ctx.clearRect(0, 0, W, H);
  days.forEach((d, i) => {
    const x = gap + i * (bw + gap);
    const bh = (counts[i] / maxC) * (H - padB - padT);
    const y = H - padB - bh;
    ctx.fillStyle = counts[i] ? '#ff7a18' : '#ffe0c2';
    const r = 8; ctx.beginPath();
    ctx.moveTo(x, H - padB); ctx.lineTo(x, y + r); ctx.arcTo(x, y, x + r, y, r);
    ctx.lineTo(x + bw - r, y); ctx.arcTo(x + bw, y, x + bw, y + r, r);
    ctx.lineTo(x + bw, H - padB); ctx.closePath(); ctx.fill();
    if (counts[i]) { ctx.fillStyle = '#e85d04'; ctx.font = 'bold 13px sans-serif'; ctx.textAlign = 'center';
      ctx.fillText(String(counts[i]), x + bw / 2, y - 4); }
    ctx.fillStyle = '#9ca3af'; ctx.font = '12px sans-serif'; ctx.textAlign = 'center';
    ctx.fillText(names[d.getDay()], x + bw / 2, H - 8);
  });
}
function showHistory() {
  const log = loadLog();
  drawWeek($('weekCanvas'), log);
  const week = log.filter((r) => (Date.now() - new Date(r.date).getTime()) < 7 * 864e5);
  const avgStars = week.length ? (week.reduce((s, r) => s + r.stars, 0) / week.length).toFixed(1) : '–';
  $('histStats').innerHTML =
    `<div class="stat"><div class="v">${log.length}</div><div class="l">ぜんぶの かいすう</div></div>` +
    `<div class="stat"><div class="v">${week.length}</div><div class="l">こん週</div></div>` +
    `<div class="stat"><div class="v">${avgStars}</div><div class="l">こん週 へいきん⭐</div></div>`;
  const list = $('histList');
  if (!log.length) { list.innerHTML = '<div class="empty">まだ きろくが ないよ。<br>れんしゅうを はじめよう！🏀</div>'; return; }
  list.innerHTML = '';
  log.slice(-8).reverse().forEach((r) => {
    const d = new Date(r.date);
    const row = document.createElement('div'); row.className = 'row';
    row.innerHTML = `<div class="d">${d.getMonth() + 1}/${d.getDate()} ${String(d.getHours()).padStart(2, '0')}:${String(d.getMinutes()).padStart(2, '0')}</div>` +
      `<div class="m">${MODE_LABEL[r.mode] || r.mode}</div><div class="s">${'⭐'.repeat(r.stars)}</div>`;
    list.appendChild(row);
  });
}

// ---- flow ----
let currentMode = 'shoot';
let currentUrl = null;
function leaveResult() {
  const vid = $('resultVideo');
  try { vid.pause(); vid.removeAttribute('src'); vid.load(); } catch (e) {}
  if (currentUrl) { try { URL.revokeObjectURL(currentUrl); } catch (e) {} currentUrl = null; }
}
async function handleFile(file) {
  if (!file) return;
  leaveResult();
  showScreen('loading'); $('progressBar').style.width = '0%';
  $('loadingMsg').textContent = 'AIコーチが よみこみ中…';
  try {
    await getDetector();
    $('loadingMsg').textContent = 'AIコーチが みているよ…';
    const data = await analyzeFile(file);
    const res = ANALYZERS[currentMode](data);
    if (!res.metrics.length) { URL.revokeObjectURL(data.url); throw new Error('no-pose'); }
    currentUrl = data.url; // kept alive for result playback; revoked on leave
    showResult(currentMode, res, data);
  } catch (e) {
    const map = {
      'no-pose': MODE_ERR[currentMode],
      'video-load': 'どうがを よみこめなかったよ。べつの どうがで ためしてね。',
      'no-duration': 'どうがの ながさが わからなかったよ。べつの どうがで ためしてね。',
    };
    $('errorMsg').textContent = map[e.message] || 'うまく いかなかったよ。もういちど ためしてね。';
    showScreen('error');
  }
}

document.querySelectorAll('.mode-btn').forEach((btn) =>
  btn.addEventListener('click', () => { currentMode = btn.dataset.mode; $('videoInput').value = ''; $('videoInput').click(); }));
$('videoInput').addEventListener('change', (e) => handleFile(e.target.files[0]));
$('againBtn').addEventListener('click', () => { leaveResult(); $('videoInput').value = ''; showScreen('home'); });
$('errorBackBtn').addEventListener('click', () => { $('videoInput').value = ''; showScreen('home'); });
$('historyBtn').addEventListener('click', () => { showHistory(); showScreen('history'); });
$('histBackBtn').addEventListener('click', () => showScreen('home'));
$('clearBtn').addEventListener('click', () => {
  if (confirm('れんしゅうの きろくを ぜんぶ けします。いいですか？')) { localStorage.removeItem(LOG_KEY); showHistory(); }
});

// ---- team roster + player cards (sample / demo, on-device) ----
const ROSTER_KEY = 'hoopRoster_v1';
const PARAMS = [
  { k: 'speed', l: 'スピード' }, { k: 'power', l: 'パワー' }, { k: 'dribble', l: 'ドリブル' },
  { k: 'shoot', l: 'シュート' }, { k: 'defense', l: 'ディフェンス' }, { k: 'quick', l: 'クイックネス' },
];
const DRILLS = {
  speed: [['ダッシュ＆敏捷ラダー', '一歩目を速く', 'ふつう']],
  power: [['パワーステップ＆ジャンプ反復', 'ひざを深く曲げて爆発的に', 'ふつう'], ['スクワットジャンプ', 'まっすぐ高く', 'やさしい']],
  dribble: [['クロスオーバー／レッグスルー反復', '低く速く、体に近く', 'ふつう'], ['低いドリブル制御', 'こしを落として', 'やさしい']],
  shoot: [['シュートフォーム 10本×3', 'リリースで腕を伸ばしきる', 'やさしい'], ['フォロースルー意識', 'なげた後、手を高く', 'やさしい']],
  defense: [['ディフェンススライド', '低い姿勢を保つ', 'ふつう'], ['1on1 止める反復', '相手とゴールの間に', 'むずかしい']],
  quick: [['方向転換ドリル', '切り返しを鋭く', 'ふつう'], ['第一歩クイックネス', '反応して素早く', 'ふつう']],
};
function seedRoster() {
  return [
    { id: 'a', name: '選手A', pos: 'ガード', num: '4', stats: { speed: 78, power: 62, dribble: 71, shoot: 65, defense: 80, quick: 74 } },
    { id: 'b', name: '選手B', pos: 'フォワード', num: '7', stats: { speed: 65, power: 82, dribble: 58, shoot: 70, defense: 68, quick: 60 } },
    { id: 'c', name: '選手C', pos: 'センター', num: '10', stats: { speed: 55, power: 88, dribble: 50, shoot: 60, defense: 75, quick: 52 } },
  ];
}
function loadRoster() {
  try { const r = JSON.parse(localStorage.getItem(ROSTER_KEY)); if (r && r.length) return r; } catch (e) {}
  const s = seedRoster(); try { localStorage.setItem(ROSTER_KEY, JSON.stringify(s)); } catch (e) {} return s;
}
const overall = (s) => Math.round(PARAMS.reduce((a, p) => a + s[p.k], 0) / PARAMS.length);

function renderRoster() {
  const list = $('rosterList'); list.innerHTML = '';
  loadRoster().forEach((p) => {
    const row = document.createElement('button'); row.className = 'player-row';
    row.innerHTML = `<span class="pr-ava">${p.num}</span>`
      + `<span class="pr-body"><b>${p.name}</b><span>${p.pos}</span></span>`
      + `<span class="pr-ovr"><span class="l">そうごう</span><span class="v">${overall(p.stats)}</span></span>`;
    row.onclick = () => openPlayer(p.id);
    list.appendChild(row);
  });
}
function radarSVG(stats) {
  const cx = 120, cy = 122, R = 76, n = PARAMS.length;
  const ang = (i) => (-90 + i * 360 / n) * Math.PI / 180;
  const pt = (i, rad) => [cx + Math.cos(ang(i)) * rad, cy + Math.sin(ang(i)) * rad];
  let grid = '';
  [0.25, 0.5, 0.75, 1].forEach((f) => {
    const pts = PARAMS.map((_, i) => pt(i, R * f).map((v) => v.toFixed(1)).join(',')).join(' ');
    grid += `<polygon points="${pts}" fill="none" stroke="#2b3647" stroke-width="1"/>`;
  });
  let axes = '', labels = '';
  PARAMS.forEach((p, i) => {
    const [x, y] = pt(i, R); axes += `<line x1="${cx}" y1="${cy}" x2="${x.toFixed(1)}" y2="${y.toFixed(1)}" stroke="#2b3647" stroke-width="1"/>`;
    const [lx, ly] = pt(i, R + 15); const anc = lx < cx - 5 ? 'end' : (lx > cx + 5 ? 'start' : 'middle');
    labels += `<text x="${lx.toFixed(1)}" y="${(ly + 4).toFixed(1)}" text-anchor="${anc}" font-size="11" fill="#9fb2cc">${p.l}</text>`;
  });
  const clamp = (v) => Math.max(0, Math.min(100, v));
  const vpts = PARAMS.map((p, i) => pt(i, R * clamp(stats[p.k]) / 100).map((v) => v.toFixed(1)).join(',')).join(' ');
  const dots = PARAMS.map((p, i) => { const [x, y] = pt(i, R * clamp(stats[p.k]) / 100); return `<circle cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="3.2" fill="#ff9a3d"/>`; }).join('');
  return `<svg viewBox="0 0 240 248" width="100%" style="max-width:300px;display:block;margin:4px auto 0" aria-label="能力レーダー">`
    + `${grid}${axes}<polygon points="${vpts}" fill="rgba(255,138,43,0.22)" stroke="#ff8a2b" stroke-width="2"/>${dots}${labels}</svg>`;
}
function suggestMenu(stats) {
  const weak = PARAMS.slice().sort((a, b) => stats[a.k] - stats[b.k]).slice(0, 2);
  const drills = [];
  weak.forEach((p) => (DRILLS[p.k] || []).forEach((d) => { if (drills.length < 3) drills.push(d); }));
  return { weak, drills };
}
function openPlayer(id) {
  const p = loadRoster().find((x) => x.id === id); if (!p) return;
  const chips = PARAMS.map((pr) => `<span class="chip">${pr.l} ${p.stats[pr.k]}</span>`).join('');
  const { weak, drills } = suggestMenu(p.stats);
  const drillHTML = drills.map((d) => `<div class="drill"><div class="chk">✓</div><div class="body"><b>${d[0]}</b><span>${d[1]}</span></div><div class="lv">${d[2]}</div></div>`).join('');
  $('playerCard').innerHTML =
    `<div class="card pcard">`
    + `<div class="pchead"><div class="ava">${p.num}</div>`
    + `<div><div style="font-weight:700;font-size:16px">${p.name} <span style="color:#9ca3af;font-weight:400">#${p.num}</span></div>`
    + `<div style="font-size:13px;color:#9fb2cc">${p.pos} ・ サンプル選手</div></div>`
    + `<div class="ovr"><div class="l">そうごう</div><div class="v">${overall(p.stats)}</div></div></div>`
    + radarSVG(p.stats)
    + `<div class="chips">${chips}</div>`
    + `<p class="note" style="text-align:center;margin-top:8px">推定値（サンプル）・チーム内の相対と伸びの推移で見る</p></div>`
    + `<div class="card"><div style="font-weight:700;margin-bottom:4px">おすすめ練習メニュー `
    + `<span style="color:#9fb2cc;font-weight:400;font-size:13px">（弱点：${weak.map((w) => w.l).join('・')}・あくまで提案）</span></div>`
    + drillHTML
    + `<div class="coachbar"><span class="who">👈 コーチが最終判断</span><span style="margin-left:auto"></span>`
    + `<button class="btn btn-edit" id="menuEdit">修正する</button><button class="btn btn-ok" id="menuOk">承認する</button></div>`
    + `<p id="coachMsg" class="note" style="margin-top:8px;display:none"></p></div>`;
  $('menuOk').onclick = () => { const m = $('coachMsg'); m.style.display = 'block'; m.textContent = '✓ コーチが承認しました（デモ）'; m.style.color = '#3fe0a2'; };
  $('menuEdit').onclick = () => { const m = $('coachMsg'); m.style.display = 'block'; m.textContent = '（デモ）ここでメニューの入れ替え・追加・削除ができます'; m.style.color = '#9fb2cc'; };
  showScreen('player');
}
$('rosterBtn').addEventListener('click', () => { renderRoster(); showScreen('roster'); });
$('rosterBackBtn').addEventListener('click', () => showScreen('home'));
$('playerBackBtn').addEventListener('click', () => { renderRoster(); showScreen('roster'); });

// expose a couple of helpers for quick verification in the preview
window.__hoop = { saveSession, loadLog, showHistory, analyzeDribble, analyzeDefense, analyzeFile, analyzeShooting, ANALYZERS, getDetector, loadRoster, renderRoster, openPlayer, suggestMenu };
