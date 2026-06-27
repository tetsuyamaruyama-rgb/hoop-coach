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
function loadVideo(file) {
  return new Promise((resolve, reject) => {
    const v = document.createElement('video');
    v.preload = 'auto'; v.muted = true; v.playsInline = true;
    v.src = URL.createObjectURL(file);
    v.onloadeddata = () => resolve(v);
    v.onerror = () => reject(new Error('video-load'));
  });
}
// iOS-robust seek: wait for 'seeked' AND two animation frames so the frame is
// actually painted before we draw it; hard timeout so we never hang.
function seek(video, t) {
  return new Promise((resolve) => {
    let done = false; const ok = () => { if (!done) { done = true; resolve(); } };
    video.onseeked = () => requestAnimationFrame(() => requestAnimationFrame(ok));
    video.currentTime = Math.min(t, Math.max(0, video.duration - 0.05));
    setTimeout(ok, 900);
  });
}
// Scan the clip storing ONLY keypoints (no per-frame images) — keeps memory tiny
// so iPhones don't stall. Frame count is bounded for speed/reliability.
async function analyzeFile(file) {
  const det = await getDetector();
  const video = await loadVideo(file);
  const dur = video.duration;
  if (!dur || !isFinite(dur)) throw new Error('no-duration');
  const vw = video.videoWidth || PROC_W, vh = video.videoHeight || PROC_W;
  const w = PROC_W, h = Math.round(PROC_W * vh / vw);
  const cv = document.createElement('canvas'); cv.width = w; cv.height = h;
  const ctx = cv.getContext('2d');
  const total = Math.max(6, Math.min(30, Math.round(dur * 6)));
  const frames = [];
  for (let i = 0; i < total; i++) {
    const t = (i + 0.5) * dur / total;
    await seek(video, t);
    if (video.readyState >= 2) {
      ctx.drawImage(video, 0, 0, w, h);
      const poses = await det.estimatePoses(cv, { maxPoses: 1, flipHorizontal: false });
      if (poses[0] && poses[0].keypoints.filter((k) => k.score > 0.3).length >= 6) {
        const byName = {}; poses[0].keypoints.forEach((k) => (byName[k.name] = k));
        frames.push({ t, byName });
      }
    }
    $('progressBar').style.width = Math.round(((i + 1) / total) * 100) + '%';
  }
  if (frames.length < 4) { URL.revokeObjectURL(video.src); throw new Error('no-pose'); }
  return { frames, w, h, video };
}
// grab a single frame's pixels at time t (for drawing the result skeleton)
async function grabFrame(video, t, w, h) {
  await seek(video, t);
  const cv = document.createElement('canvas'); cv.width = w; cv.height = h;
  cv.getContext('2d').drawImage(video, 0, 0, w, h);
  return cv;
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
  const elb = angle(kp(rel, S('shoulder')), kp(rel, S('elbow')), kp(rel, S('wrist')));
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
function drawSkeleton(canvas, srcCanvas, frame, w, h, side) {
  canvas.width = w; canvas.height = h;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(srcCanvas, 0, 0, w, h);
  const pairs = [['shoulder', 'elbow'], ['elbow', 'wrist'], ['shoulder', 'hip'], ['hip', 'knee'], ['knee', 'ankle']];
  ctx.lineWidth = Math.max(3, w / 120); ctx.strokeStyle = '#ff7a18'; ctx.lineCap = 'round';
  for (const [a, b] of pairs) { const ka = kp(frame, side + '_' + a), kb = kp(frame, side + '_' + b);
    if (ka && kb) { ctx.beginPath(); ctx.moveTo(ka.x, ka.y); ctx.lineTo(kb.x, kb.y); ctx.stroke(); } }
  ['shoulder', 'elbow', 'wrist', 'hip', 'knee', 'ankle'].forEach((n) => { const k = kp(frame, side + '_' + n);
    if (k) { ctx.fillStyle = '#fff'; ctx.beginPath(); ctx.arc(k.x, k.y, Math.max(4, w / 90), 0, 7); ctx.fill(); } });
}
function fbCard(kind, [ico, title, sub]) {
  const d = document.createElement('div'); d.className = 'fb ' + kind;
  d.innerHTML = `<div class="ico">${ico}</div><div class="txt"><b>${title}</b><span>${sub}</span></div>`;
  return d;
}
function showResult(mode, res, data, keyCanvas) {
  const good = res.metrics.filter((m) => m.good).length;
  const stars = good >= 3 ? 3 : good === 2 ? 2 : 1;
  $('modeTag').textContent = MODE_LABEL[mode];
  $('scoreStars').textContent = '⭐'.repeat(stars);
  drawSkeleton($('resultCanvas'), keyCanvas, data.frames[res.keyIdx], data.w, data.h, res.side);
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
async function handleFile(file) {
  if (!file) return;
  showScreen('loading'); $('progressBar').style.width = '0%';
  $('loadingMsg').textContent = 'AIコーチが よみこみ中…';
  try {
    await getDetector();
    $('loadingMsg').textContent = 'AIコーチが みているよ…';
    const data = await analyzeFile(file);
    const res = ANALYZERS[currentMode](data);
    if (!res.metrics.length) { URL.revokeObjectURL(data.video.src); throw new Error('no-pose'); }
    const keyCanvas = await grabFrame(data.video, data.frames[res.keyIdx].t, data.w, data.h);
    URL.revokeObjectURL(data.video.src);
    showResult(currentMode, res, data, keyCanvas);
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
$('againBtn').addEventListener('click', () => { $('videoInput').value = ''; showScreen('home'); });
$('errorBackBtn').addEventListener('click', () => { $('videoInput').value = ''; showScreen('home'); });
$('historyBtn').addEventListener('click', () => { showHistory(); showScreen('history'); });
$('histBackBtn').addEventListener('click', () => showScreen('home'));
$('clearBtn').addEventListener('click', () => {
  if (confirm('れんしゅうの きろくを ぜんぶ けします。いいですか？')) { localStorage.removeItem(LOG_KEY); showHistory(); }
});

// expose a couple of helpers for quick verification in the preview
window.__hoop = { saveSession, loadLog, showHistory, analyzeDribble, analyzeDefense };
