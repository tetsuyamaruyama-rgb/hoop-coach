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

// ---- grading: 3 tiers (2=◎ / 1=○ / 0=△). Graded vs the coach's お手本 if set,
//      otherwise vs demanding default bars. ----
const lvlLow = (v, good, ok) => (v <= good ? 2 : v <= ok ? 1 : 0);  // smaller is better
const lvlHigh = (v, good, ok) => (v >= good ? 2 : v >= ok ? 1 : 0); // larger is better
const lvlBand = (v, gLo, gHi, oLo, oHi) => (v >= gLo && v <= gHi ? 2 : v >= oLo && v <= oHi ? 1 : 0);
function metric(level, value, label, praise, tip) { return { level, good: level === 2, value, label, praise, tip }; }
// richer metric: carries the raw measurement + direction so it can be graded
// against EITHER the お手本 or the default bars.
function mtr(key, raw, dir, good, ok, valNum, valWords, label, praise, tip) {
  return { key, raw, dir, good, ok, valNum, valWords, label, praise, tip };
}
function gradeDefault(m) {
  if (m.dir === 'low') return lvlLow(m.raw, m.good, m.ok);
  if (m.dir === 'high') return lvlHigh(m.raw, m.good, m.ok);
  return lvlBand(m.raw, m.good[0], m.good[1], m.ok[0], m.ok[1]); // band
}
function gradeVsModel(m, target) { // how close to the お手本's measured value
  if (m.dir === 'low') return m.raw <= target * 1.04 ? 2 : m.raw <= target * 1.12 ? 1 : 0;
  if (m.dir === 'high') return m.raw >= target * 0.96 ? 2 : m.raw >= target * 0.88 ? 1 : 0;
  const d = Math.abs(m.raw - target) / (target || 1); return d <= 0.15 ? 2 : d <= 0.30 ? 1 : 0;
}
// ---- お手本 (coach reference) storage: measured target values per skill, on-device ----
const MODEL_KEY = 'hoopModel_v1';
function loadModel() { try { return JSON.parse(localStorage.getItem(MODEL_KEY)) || {}; } catch (e) { return {}; } }
function saveModel(mode, vals) { const m = loadModel(); m[mode] = vals; try { localStorage.setItem(MODEL_KEY, JSON.stringify(m)); } catch (e) {} }
function clearModel(mode) { const m = loadModel(); delete m[mode]; try { localStorage.setItem(MODEL_KEY, JSON.stringify(m)); } catch (e) {} }
function modelVals(metrics) { const o = {}; metrics.forEach((x) => { if (x.raw != null) o[x.key] = x.raw; }); return o; }
function gradeAll(mode, metrics, angle) {
  const ref = loadModel()[mode];
  metrics.forEach((m) => {
    if (m.level != null || m.skip) return; // already graded / skipped
    if (angle && ANGLE_OK[m.key] && ANGLE_OK[m.key] !== 'both' && ANGLE_OK[m.key] !== angle) {
      m.skip = true; m.value = '—'; return; // this angle can't measure it
    }
    const L = (ref && ref[m.key] != null) ? gradeVsModel(m, ref[m.key]) : gradeDefault(m);
    m.level = L; m.good = L === 2;
    m.value = (m.valNum != null) ? m.valNum : (m.valWords ? m.valWords[L] : '');
  });
  return metrics;
}

// ---- analyzers: each returns {keyIdx, side, metrics:[{level,value,label,praise,tip}]} ----
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
  if (isFinite(mK)) metrics.push(mtr('knee', mK, 'low', 142, 155, Math.round(mK) + '°', null, 'ひざのため',
    ['🦵', 'ひざが ぐっと まがってる！', 'ためが できてて パワーが でる'],
    ['🦵', 'ひざの ためが あさい', 'もっと 深く しずんでから とぼう（目安 140°台まで）']));
  // peak elbow extension across the shooting phase (a brief moment the coarse
  // scan can miss at the exact release frame) — answers "did he straighten it?"
  let elb = null;
  for (let i = loadIdx; i < frames.length; i++) {
    const a = angle(kp(frames[i], S('shoulder')), kp(frames[i], S('elbow')), kp(frames[i], S('wrist')));
    if (a != null && (elb == null || a > elb)) elb = a;
  }
  if (elb != null) metrics.push(mtr('elbow', elb, 'high', 165, 150, Math.round(elb) + '°', null, 'うでの のび',
    ['💪', 'うでが 完全に のびてる！', 'リリースが するどい'],
    ['💪', 'のびきりが あまい', 'ひじを 最後まで のばしきろう（目安 165°以上）']));
  const wr = kp(rel, S('wrist')), nose = kp(rel, 'nose') || kp(rel, S('shoulder'));
  if (wr && nose) {
    const tl = torsoLen(rel) || 1; const fhr = (nose.y - wr.y) / tl;
    metrics.push(mtr('follow', fhr, 'high', 0.18, 0.02, null, ['ひくい', 'ふつう', 'たかい'], 'フォロースルー',
      ['🙌', 'てが あたまの うえまで のびてる！', 'きれいな フォロースルー'],
      ['🙌', 'フォロースルーが ひくい', 'なげた後、手を あたまより 高く のこそう']));
  }
  const la = kp(load, 'left_ankle'), ra = kp(load, 'right_ankle'), ls = kp(load, 'left_shoulder'), rs = kp(load, 'right_shoulder');
  if (la && ra && ls && rs) {
    const ratio = dist(la, ra) / (dist(ls, rs) || 1);
    metrics.push(mtr('foot', ratio, 'band', [0.85, 1.35], [0.65, 1.6], null, ['なおそう', 'すこし', 'バッチリ'], 'あしのはば',
      ['🦶', 'あしの はばが 理想的！', 'どっしり 安定してる'],
      ['🦶', 'あしの はばが いまいち', 'かたはばに そろえよう（広すぎ・せますぎ 注意）']));
  }
  return { keyIdx: relIdx, side, metrics };
}

function analyzeDribble({ frames }) {
  const side = sideByConfidence(frames);
  const S = (n) => side + '_' + n;
  const keyIdx = deepestKneeIdx(frames, side);
  const metrics = [];
  // 1) low athletic stance (knee bend, averaged)
  const knees = series(frames, (f) => angle(kp(f, S('hip')), kp(f, S('knee')), kp(f, S('ankle'))));
  if (knees.length) { const a = avg(knees); metrics.push(mtr('stance', a, 'low', 148, 160, Math.round(a) + '°', null, 'ひくいしせい',
    ['🔽', 'こしが しっかり ひくい！', 'これなら 簡単に とられない'],
    ['🔽', 'しせいが 高い', 'もっと こしを 落として（目安 平均145°台）']));
  }
  // 2) head up (upright torso, not hunched over the ball)
  const leans = series(frames, (f) => { const t = torsoLen(f); if (!t) return null;
    const ls = kp(f, 'left_shoulder'), rs = kp(f, 'right_shoulder'), lh = kp(f, 'left_hip'), rh = kp(f, 'right_hip');
    if (!ls || !rs || !lh || !rh) return null; return Math.abs(mid(ls, rs).x - mid(lh, rh).x) / t; });
  if (leans.length) { const lean = avg(leans); metrics.push(mtr('headup', lean, 'low', 0.30, 0.45, null, ['うつむき', 'ややうつむき', 'まえむき'], 'あたまアップ',
    ['👀', 'しっかり 前を みれてる！', 'ボールを 見ないで つけてる'],
    ['👀', '前傾して ボールを 見がち', 'かおを 上げて 前を 見たまま つこう']));
  }
  // 3) rhythm — active hand moves up & down (normalized vertical motion)
  const t0 = avg(series(frames, torsoLen)) || 1;
  const wy = series(frames, (f) => kp(f, S('wrist'))?.y);
  if (wy.length >= 4) { const amp = std(wy) / t0; metrics.push(mtr('rhythm', amp, 'high', 0.13, 0.07, null, ['よわい', 'ふつう', 'するどい'], 'リズム・強さ',
    ['🥁', 'つよく するどく つけてる！', 'ボールが 走ってる'],
    ['🥁', 'つきが よわい', 'ゆかへ もっと つよく・速く たたきつけよう']));
  }
  return { keyIdx, side, metrics };
}

function analyzeDefense({ frames }) {
  const side = sideByConfidence(frames);
  const S = (n) => side + '_' + n;
  const keyIdx = deepestKneeIdx(frames, side);
  const metrics = [];
  // 1) low stance (most important for defense)
  const knees = series(frames, (f) => angle(kp(f, S('hip')), kp(f, S('knee')), kp(f, S('ankle'))));
  if (knees.length) { const a = avg(knees); metrics.push(mtr('stance', a, 'low', 140, 153, Math.round(a) + '°', null, 'こしを おとす',
    ['🔽', 'こしが かなり ひくい！ 良い構え', 'これなら 速く 反応できる'],
    ['🔽', '構えが 高い', 'もっと 腰を 落として（目安 平均140°前後）']));
  }
  // 2) wide base
  const widths = series(frames, (f) => { const la = kp(f, 'left_ankle'), ra = kp(f, 'right_ankle'), ls = kp(f, 'left_shoulder'), rs = kp(f, 'right_shoulder');
    if (!la || !ra || !ls || !rs) return null; return dist(la, ra) / (dist(ls, rs) || 1); });
  if (widths.length) { const r = avg(widths); metrics.push(mtr('base', r, 'high', 1.25, 1.05, null, ['せまい', 'ふつう', 'ひろい'], 'あしのはば',
    ['🦶', 'スタンスが ひろくて 安定！', '左右に すばやく 動ける'],
    ['🦶', 'スタンスが せまい', '肩幅より はっきり 広く 構えよう']));
  }
  // 3) hands up / active
  const handsUp = frames.filter((f) => { const h = kp(f, S('hip')); const w = kp(f, S('wrist')); return h && w && w.y < h.y; }).length;
  const ratio = frames.length ? handsUp / frames.length : 0;
  metrics.push(mtr('hands', ratio, 'high', 0.6, 0.35, null, ['さげてる', 'ときどき', 'あげてる'], 'てを あげる',
    ['🙌', 'ずっと 手を あげて 守れてる！', 'パスも シュートも 邪魔できる'],
    ['🙌', '手が 下がりがち', '両手を 上げ続けて コースを 消そう']));
  return { keyIdx, side, metrics };
}

const ANALYZERS = { shoot: analyzeShooting, dribble: analyzeDribble, defense: analyzeDefense };
// dribble move types — each gets its own お手本 + grading (same measurements, move-specific reference)
const DRIBBLE_SUBS = [['dribble_normal', 'ノーマル'], ['dribble_front', 'フロントチェンジ'], ['dribble_back', 'バックチェンジ'], ['dribble_legthru', 'レッグスルー']];
const baseOf = (m) => (m && m.indexOf('dribble') === 0 ? 'dribble' : m); // analyzer key for any dribble sub-mode
const MODE_LABEL = {
  shoot: '🎯 シュート', dribble: '⛹️ ドリブル', defense: '🛡️ ディフェンス',
  dribble_normal: '⛹️ ドリブル：ノーマル', dribble_front: '⛹️ ドリブル：フロントチェンジ',
  dribble_back: '⛹️ ドリブル：バックチェンジ', dribble_legthru: '⛹️ ドリブル：レッグスルー',
};
// which camera angle each measurement is reliable from ('side' | 'front' | 'both')
const ANGLE_OK = {
  knee: 'side', elbow: 'side', follow: 'both', foot: 'front',
  stance: 'side', headup: 'side', rhythm: 'both', base: 'front', hands: 'both',
};
// recommended angle per (sub)mode
const ANGLE_RECO = {
  shoot: 'side', defense: 'front',
  dribble_normal: 'side', dribble_front: 'front', dribble_back: 'front', dribble_legthru: 'front',
};
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
  gradeAll(mode, res.metrics, currentAngle); // ensure levels/skip set (idempotent)
  const lvl = (m) => (m.level == null ? (m.good ? 2 : 0) : m.level);
  const graded = res.metrics.filter((m) => !m.skip); // only angle-measurable metrics count
  const sorted = graded.slice().sort((a, b) => lvl(a) - lvl(b)); // worst first
  const good = graded.filter((m) => lvl(m) === 2).length;
  const total = graded.length;
  const stars = total === 0 ? 1 : good >= total ? 3 : good >= Math.ceil(total / 2) ? 2 : 1;
  const angleTxt = currentAngle === 'front' ? '正面' : '横';
  $('modeTag').textContent = MODE_LABEL[mode] + ' ・ ' + angleTxt + (loadModel()[mode] ? ' ・ お手本基準' : '');
  $('scoreStars').textContent = '⭐'.repeat(stars);
  const best = graded.filter((m) => lvl(m) === 2)[0];
  const worst = sorted[0];
  const fb = $('feedback'); fb.innerHTML = '';
  if (best) fb.appendChild(fbCard('good', best.praise));
  else fb.appendChild(fbCard('good', ['🏀', 'チャレンジ えらい！', 'むずかしい 基準だよ。1つずつ よくしよう']));
  if (worst && lvl(worst) < 2) fb.appendChild(fbCard('tip', worst.tip));
  else fb.appendChild(fbCard('good', ['🎉', 'パーフェクト！', 'きびしい 基準を 全部 クリア！']));
  const SYM = { 2: '◎', 1: '○', 0: '△' }, CLS = { 2: 'ok', 1: 'mid', 0: 'no' };
  const mWrap = $('metrics'); mWrap.innerHTML = '';
  res.metrics.forEach((m) => {
    const d = document.createElement('div');
    if (m.skip) { d.className = 'metric skip';
      d.innerHTML = `<div class="v">—</div><div class="l">${m.label}<br><span style="font-size:10px">この角度では はかれません</span></div>`; }
    else { const L = lvl(m); d.className = 'metric ' + CLS[L];
      d.innerHTML = `<div class="v">${SYM[L]} ${m.value}</div><div class="l">${m.label}</div>`; }
    mWrap.appendChild(d);
  });
  saveSession(mode, stars, good, total);
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
let currentAngle = 'side'; // 'front' | 'side' — chosen on the angle screen
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
    const res = ANALYZERS[baseOf(currentMode)](data);
    if (!res.metrics.length) { URL.revokeObjectURL(data.url); throw new Error('no-pose'); }
    gradeAll(currentMode, res.metrics, currentAngle); // grade vs お手本/default, skipping what this angle can't measure
    currentUrl = data.url; // kept alive for result playback; revoked on leave
    showResult(currentMode, res, data);
  } catch (e) {
    const map = {
      'no-pose': MODE_ERR[baseOf(currentMode)],
      'video-load': 'どうがを よみこめなかったよ。べつの どうがで ためしてね。',
      'no-duration': 'どうがの ながさが わからなかったよ。べつの どうがで ためしてね。',
    };
    $('errorMsg').textContent = map[e.message] || 'うまく いかなかったよ。もういちど ためしてね。';
    showScreen('error');
  }
}

function showAngleScreen() {
  const reco = ANGLE_RECO[currentMode] || 'side';
  $('angleReco').innerHTML = `この技は <b>${reco === 'front' ? '正面' : '横（側面）'}</b> が おすすめ。<br>えらんだ 角度で <b>はかれる項目だけ</b> 採点します。`;
  showScreen('angle');
}
document.querySelectorAll('.mode-btn[data-mode]').forEach((btn) =>
  btn.addEventListener('click', () => {
    const mode = btn.dataset.mode;
    if (mode === 'dribble') { showScreen('dribsub'); return; } // pick a move type first
    currentMode = mode; showAngleScreen();
  }));
document.querySelectorAll('.mode-btn[data-sub]').forEach((btn) =>
  btn.addEventListener('click', () => { currentMode = btn.dataset.sub; showAngleScreen(); }));
document.querySelectorAll('#screen-angle [data-angle]').forEach((btn) =>
  btn.addEventListener('click', () => { currentAngle = btn.dataset.angle; $('videoInput').value = ''; $('videoInput').click(); }));
$('angleBack').addEventListener('click', () => showScreen(currentMode && currentMode.indexOf('dribble') === 0 ? 'dribsub' : 'home'));
$('dribsubBack').addEventListener('click', () => showScreen('home'));
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

// ---- 1on1 record + review (MAIN entry; sample/demo output) ----
const OFF_GOOD = {
  speed: ['⚡', 'スピードで 抜き去れている', '一歩目の速さが 武器'],
  power: ['💪', '体の強さで 押し込めている', 'コンタクトに 強い'],
  dribble: ['🏀', 'ドリブルで かわせている', 'ハンドリングが 効いてる'],
  shoot: ['🎯', 'シュートまで 持ち込めている', '得点力がある'],
  defense: ['🧠', '落ち着いて 攻められている', '焦らない'],
  quick: ['💨', '第一歩で 出し抜けている', '反応が速い'],
};
const OFF_TIP = {
  shoot: ['🎯', 'フィニッシュ（シュート）の精度を上げよう', '最後の一本を 落ち着いて'],
  dribble: ['🏀', '仕掛ける回数を 増やそう', '低く速いドリブルで'],
  power: ['💪', 'コンタクトに 負けないように', '体の使い方を意識'],
  quick: ['💨', '切り返しを もっと鋭く', ''],
};
const DEF_GOOD = {
  defense: ['🛡️', '構えが よく止まれている', '低い姿勢を 保てている'],
  power: ['💪', '体を張って 守れている', '当たりに 強い'],
  quick: ['💨', '反応が 速い', '抜かれにくい'],
  speed: ['⚡', '戻り・寄せが 速い', ''],
  dribble: ['👀', 'よく ボールを 見れている', ''],
  shoot: ['🧱', 'プレッシャーを かけられている', ''],
};
const DEF_TIP = {
  quick: ['💨', '抜かれ際の 反応を 速く', '一歩目に ついていこう'],
  defense: ['🔽', 'もっと 腰を 低く', 'スタンスを 広く'],
  speed: ['⚡', 'スライドの 足を 速く', ''],
  power: ['💪', '当たり負けに 注意', ''],
};
const OFF_PT = {
  speed: ['🏃', 'スピードの ミスマッチ', '外から アタックして 一気に 抜き去ろう'],
  power: ['🪨', 'フィジカルで 優位', 'ゴール下で 体を ぶつけて 強気に'],
  dribble: ['🌀', 'ハンドリングで 優位', '緩急と チェンジで ズレを 作ろう'],
  shoot: ['🎯', 'シュート力で 警戒させる', 'ドライブと シュートの 二択で 揺さぶる'],
  quick: ['⚡', 'クイックネスで 優位', 'タメ→爆発の 緩急で 仕掛けよう'],
  defense: ['🧊', '冷静さが 武器', '相手の 重心を 見て 逆を つこう'],
};
const DEF_PT = {
  defense: ['🧱', '守備の 安定感', '相手を ベースライン／サイドへ 追い込もう'],
  power: ['🪨', '体の強さ', 'コースに 体を 入れて シュートを 難しく'],
  quick: ['⚡', '反応の 速さ', '抜かれても もう一歩 前に 出て やり直す'],
  speed: ['🏃', 'スピード', '戻りと 寄せで カバーを 続けよう'],
  dribble: ['👁️', '観察力', '相手の 腰と ボールを 見て 反応'],
  shoot: ['🙅', 'プレッシャー', 'シュートに 必ず 手を 伸ばそう'],
};
const labelOf = (k) => (PARAMS.find((p) => p.k === k) || { l: k }).l;
function fbHTML(kind, a) { return `<div class="fb ${kind}"><div class="ico">${a[0]}</div><div class="txt"><b>${a[1]}</b><span>${a[2] || ''}</span></div></div>`; }
const strongestK = (s) => PARAMS.slice().sort((a, b) => s[b.k] - s[a.k])[0].k;
const weakestAmong = (s, keys) => keys.slice().sort((a, b) => s[a] - s[b])[0];
function fillPlayerSelect(sel, selId) {
  sel.innerHTML = loadRoster().map((p) => `<option value="${p.id}">${p.name}（#${p.num}・${p.pos}）</option>`).join('');
  if (selId) sel.value = selId;
}
function buildOneon(offId, defId, clipUrl) {
  const r = loadRoster(); const off = r.find((p) => p.id === offId) || r[0]; const def = r.find((p) => p.id === defId) || r[1];
  const oS = strongestK(off.stats), dS = strongestK(def.stats);
  const offGood = OFF_GOOD[oS] || OFF_GOOD.speed;
  const offTip = OFF_TIP[weakestAmong(off.stats, ['shoot', 'dribble', 'power', 'quick'])] || OFF_TIP.shoot;
  const defGood = DEF_GOOD[dS] || DEF_GOOD.defense;
  const defTip = DEF_TIP[weakestAmong(def.stats, ['quick', 'defense', 'speed', 'power'])] || DEF_TIP.quick;
  const offPt = OFF_PT[oS] || OFF_PT.speed, defPt = DEF_PT[dS] || DEF_PT.defense;
  const offFocus = (suggestMenu(off.stats).drills[0] || ['—'])[0];
  const defFocus = (suggestMenu(def.stats).drills[0] || ['—'])[0];
  // positioning lines vary a little with the attacker's profile
  const offPos = (off.stats.speed >= 70 || off.stats.quick >= 70)
    ? 'ドライブの 角度を 作れています。一歩目で 抜き切りたい。'
    : '相手を 引きつけて、味方の スペースを 作れています。';
  const defPos = (def.stats.quick >= 70)
    ? '相手とゴールの 間を キープし、抜かれても 戻れています。'
    : '相手とゴールの 間は キープ。一歩目の 反応を 上げたい。';
  $('oneonResult').innerHTML =
    (clipUrl ? `<div class="video-wrap" style="margin-bottom:12px"><video src="${clipUrl}" playsinline muted controls style="width:100%;display:block"></video></div>` : '')
    + `<div class="vs-tag">🆚 1on1 レビュー（サンプル）</div>`
    + `<div style="font-weight:800;font-size:17px;margin-bottom:6px">${off.name} <span style="color:var(--muted);font-weight:400">vs</span> ${def.name}</div>`
    + `<div class="card" style="background:#10243a;border-color:#1d3a57"><div style="font-size:13px;color:#9fc4ee">🔍 マッチアップ</div>`
    + `<div style="font-weight:700;margin-top:2px">${off.name}の【${labelOf(oS)}】 と ${def.name}の【${labelOf(dS)}】 の 勝負。</div></div>`
    + `<div class="card"><div class="role-ttl"><span class="dot" style="background:#ff8a2b"></span>オフェンス：${off.name}</div>`
    + fbHTML('good', offGood) + fbHTML('tip', offTip)
    + `<div class="pos">💡 ポイント：<b>${offPt[1]}</b> — ${offPt[2]}</div>`
    + `<div class="pos">📍 ポジショニング：${offPos}</div>`
    + `<div class="pos" style="color:var(--orange-l)">🎯 今日のフォーカス：<b>${offFocus}</b></div></div>`
    + `<div class="card"><div class="role-ttl"><span class="dot" style="background:#3fe0a2"></span>ディフェンス：${def.name}</div>`
    + fbHTML('good', defGood) + fbHTML('tip', defTip)
    + `<div class="pos">💡 ポイント：<b>${defPt[1]}</b> — ${defPt[2]}</div>`
    + `<div class="pos">📍 ポジショニング：${defPos}</div>`
    + `<div class="pos" style="color:var(--good)">🎯 今日のフォーカス：<b>${defFocus}</b></div></div>`
    + `<p class="note" style="text-align:center">✓ 結果は 各選手のカードに 反映されます（サンプル）。<br>実解析（複数人・攻守 両者の自動評価・ポジショニング）は 開発中です。</p>`;
}
function runOneon() {
  showScreen('loading'); $('progressBar').style.width = '0%'; $('loadingMsg').textContent = 'AIが 1on1を 解析中…（サンプル）';
  let pct = 0; const off = $('offSel').value, def = $('defSel').value;
  const iv = setInterval(() => {
    pct = Math.min(100, pct + 12); $('progressBar').style.width = pct + '%';
    if (pct >= 100) { clearInterval(iv); buildOneon(off, def); showScreen('1on1-result'); }
  }, 140);
}
$('oneonBtn').addEventListener('click', () => { fillPlayerSelect($('offSel'), 'a'); fillPlayerSelect($('defSel'), 'b'); showScreen('1on1'); });
$('oneonBack').addEventListener('click', () => showScreen('home'));
$('oneonPick').addEventListener('click', () => { $('oneonInput').value = ''; $('oneonInput').click(); });
$('oneonInput').addEventListener('change', (e) => { if (e.target.files[0]) runOneon(); });
$('oneonAgain').addEventListener('click', () => showScreen('home'));

// ---- in-app recorder: continuous record + one-tap offense/defense swap ----
let recStream = null, recRecorder = null, recChunks = [], recMime = '', recSegRoles = null,
  recSegStart = 0, recEnding = false, recClips = [], recOff = 'a', recDef = 'b', recN = 0,
  recTimerIv = null, recSessionStart = 0;
const nameOf = (id) => { const p = loadRoster().find((x) => x.id === id); return p ? `${p.name}（#${p.num}）` : id; };
function recUpdateRoles() { $('recOffV').textContent = nameOf(recOff); $('recDefV').textContent = nameOf(recDef); }
function recTick() { const s = Math.floor((Date.now() - recSessionStart) / 1000); $('recTimer').textContent = Math.floor(s / 60) + ':' + String(s % 60).padStart(2, '0'); }
function recStartTimer() { recSessionStart = Date.now(); recTick(); recTimerIv = setInterval(recTick, 500); }
function recStopTimer() { if (recTimerIv) { clearInterval(recTimerIv); recTimerIv = null; } }
function recShowOverlays(on) { $('recBadge').hidden = !on; $('recPoss').hidden = !on; $('recHint').hidden = on; }
function recPickMime() {
  const c = ['video/mp4', 'video/mp4;codecs=h264', 'video/webm;codecs=vp9', 'video/webm;codecs=vp8', 'video/webm'];
  for (const m of c) { try { if (window.MediaRecorder && MediaRecorder.isTypeSupported(m)) return m; } catch (e) {} } return '';
}
async function recOpenCamera() {
  $('recStatus').textContent = 'カメラを ひらいています…';
  $('recStart').disabled = true; $('recSwitch').disabled = true; $('recStop').disabled = true;
  recShowOverlays(false); recStopTimer(); $('recTimer').textContent = '0:00';
  try {
    try { recStream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment', width: { ideal: 1280 }, height: { ideal: 720 } }, audio: false }); }
    catch (e) { recStream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false }); }
    $('recPrev').srcObject = recStream; await $('recPrev').play().catch(() => {});
    recMime = recPickMime();
    $('recStart').disabled = false;
    $('recStatus').textContent = '役割を確認して「録画スタート」。';
  } catch (e) { $('recStatus').textContent = 'カメラを ひらけませんでした：' + e.name + '（カメラの許可を確認してください）'; }
}
function recStopCamera() { if (recStream) { recStream.getTracks().forEach((t) => t.stop()); recStream = null; } }
function recStartSeg() {
  recChunks = []; recSegRoles = { off: recOff, def: recDef }; recSegStart = Date.now();
  try { recRecorder = recMime ? new MediaRecorder(recStream, { mimeType: recMime }) : new MediaRecorder(recStream); }
  catch (e) { $('recStatus').textContent = '録画を 開始できません：' + e.message; return; }
  recRecorder.ondataavailable = (e) => { if (e.data && e.data.size) recChunks.push(e.data); };
  recRecorder.onstop = recOnSegStop;
  recRecorder.start();
  $('recPossN').textContent = recClips.length + 1; // possession currently being recorded
}
async function saveClip(blob, name) {
  if (!blob) return;
  const file = new File([blob], name, { type: blob.type || 'video/mp4' });
  if (navigator.canShare && navigator.canShare({ files: [file] })) {
    try { await navigator.share({ files: [file] }); return; } catch (e) { if (e && e.name === 'AbortError') return; }
  }
  const url = URL.createObjectURL(blob); const a = document.createElement('a');
  a.href = url; a.download = name; a.target = '_blank'; document.body.appendChild(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 15000);
}
function recOnSegStop() {
  const blob = new Blob(recChunks, { type: (recChunks[0] && recChunks[0].type) || recMime || 'video/mp4' });
  recN++; recClips.push({ n: recN, off: recSegRoles.off, def: recSegRoles.def, dur: Math.max(0, (Date.now() - recSegStart) / 1000), url: URL.createObjectURL(blob), blob });
  if (!recEnding) { recStartSeg(); }
  else { recStopCamera(); renderClips(); showScreen('1on1-clips'); }
}
function renderClips() {
  const list = $('clipList');
  if (!recClips.length) { list.innerHTML = '<div class="empty">クリップが ありません。</div>'; return; }
  list.innerHTML = '';
  recClips.forEach((c) => {
    const row = document.createElement('div'); row.className = 'clip-row';
    row.innerHTML = `<video src="${c.url}" playsinline muted></video>`
      + `<div class="ci"><b>ポゼッション${c.n}</b><span>攻 ${nameOf(c.off)} / 守 ${nameOf(c.def)} ・ 約${c.dur.toFixed(1)}秒</span></div>`
      + `<div style="display:flex;flex-direction:column;gap:6px">`
      + `<button class="rev">レビュー</button>`
      + `<button class="clip-save">💾 保存</button></div>`;
    row.querySelector('.rev').onclick = () => { buildOneon(c.off, c.def, c.url); showScreen('1on1-result'); };
    row.querySelector('.clip-save').onclick = () => saveClip(c.blob, `1on1_p${c.n}.mp4`);
    list.appendChild(row);
  });
}
$('oneonRec').addEventListener('click', () => {
  recOff = $('offSel').value; recDef = $('defSel').value; recClips = []; recN = 0; recEnding = false;
  recUpdateRoles(); showScreen('1on1-rec'); recOpenCamera();
});
$('recStart').addEventListener('click', () => {
  if (!recStream) return; recEnding = false; recStartSeg();
  $('recStart').disabled = true; $('recSwitch').disabled = false; $('recStop').disabled = false;
  recShowOverlays(true); recStartTimer();
  $('recStatus').textContent = '● 録画中… 攻撃が終わったら「攻守交替」。';
});
$('recSwitch').addEventListener('click', () => {
  if (!recRecorder || recRecorder.state === 'inactive') return;
  recEnding = false; const o = recOff; recOff = recDef; recDef = o; recUpdateRoles();
  recRecorder.stop(); // saves current possession (with its own roles) then auto-restarts with swapped roles
  $('recStatus').textContent = '● 録画中… 攻守交替！ 次の攻撃へ。';
});
$('recStop').addEventListener('click', () => {
  if (!recRecorder || recRecorder.state === 'inactive') { recStopCamera(); showScreen('1on1'); return; }
  recEnding = true; recRecorder.stop();
  $('recSwitch').disabled = true; $('recStop').disabled = true;
  recShowOverlays(false); recStopTimer();
});
$('recBack').addEventListener('click', () => {
  recEnding = true; try { if (recRecorder && recRecorder.state !== 'inactive') recRecorder.stop(); } catch (e) {}
  recStopCamera(); recShowOverlays(false); recStopTimer(); showScreen('1on1');
});
$('clipsBack').addEventListener('click', () => showScreen('1on1'));

// ---- お手本 (coach reference) registration ----
let pendingModelMode = null;
const MODEL_MODES = [['shoot', '🎯 シュート'],
  ['dribble_normal', '⛹️ ドリブル：ノーマル'], ['dribble_front', '⛹️ ：フロントチェンジ'],
  ['dribble_back', '⛹️ ：バックチェンジ'], ['dribble_legthru', '⛹️ ：レッグスルー'],
  ['defense', '🛡️ ディフェンス']];
function renderModelScreen() {
  const ref = loadModel(); const list = $('modelList'); list.innerHTML = '';
  MODEL_MODES.forEach(([mode, label]) => {
    const has = !!ref[mode];
    const badge = has
      ? '<span style="font-size:12px;font-weight:700;color:#3fe0a2;background:rgba(63,224,162,.12);border-radius:999px;padding:3px 10px">お手本 登録済み ✓</span>'
      : '<span style="font-size:12px;font-weight:700;color:#9fb2cc;background:#1a2233;border-radius:999px;padding:3px 10px">標準基準</span>';
    const vals = has ? `<div class="note" style="margin-top:8px">きじゅん値：${Object.entries(ref[mode]).map(([k, v]) => k + ' ' + (Math.round(v * 100) / 100)).join(' / ')}</div>` : '';
    const row = document.createElement('div'); row.className = 'card';
    row.innerHTML = `<div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap"><b style="font-size:16px">${label}</b>${badge}`
      + `<span style="margin-left:auto"></span><button class="btn btn-ok" data-reg="${mode}">📹 お手本を登録</button>`
      + (has ? `<button class="btn btn-edit" data-clr="${mode}">標準にもどす</button>` : '') + `</div>${vals}`;
    list.appendChild(row);
  });
  list.querySelectorAll('[data-reg]').forEach((b) => b.onclick = () => { pendingModelMode = b.dataset.reg; $('modelInput').value = ''; $('modelInput').click(); });
  list.querySelectorAll('[data-clr]').forEach((b) => b.onclick = () => { clearModel(b.dataset.clr); renderModelScreen(); });
}
async function handleModelFile(file) {
  if (!file || !pendingModelMode) return;
  const mode = pendingModelMode;
  showScreen('loading'); $('progressBar').style.width = '0%'; $('loadingMsg').textContent = 'お手本を 解析中…';
  try {
    await getDetector();
    const data = await analyzeFile(file);
    const res = ANALYZERS[baseOf(mode)](data); URL.revokeObjectURL(data.url);
    if (!res.metrics.length) throw new Error('no-pose');
    saveModel(mode, modelVals(res.metrics));
    renderModelScreen(); showScreen('model');
  } catch (e) {
    $('errorMsg').textContent = e.message === 'no-pose'
      ? 'お手本の からだが よく うつってませんでした。よこ から、ぜんしんが うつるように とってね。'
      : 'うまく いかなかったよ。べつの どうがで ためしてね。';
    showScreen('error');
  }
}
$('modelBtn').addEventListener('click', () => { renderModelScreen(); showScreen('model'); });
$('modelBackBtn').addEventListener('click', () => showScreen('home'));
$('modelInput').addEventListener('change', (e) => handleModelFile(e.target.files[0]));

// expose a couple of helpers for quick verification in the preview
window.__hoop = { saveSession, loadLog, showHistory, analyzeDribble, analyzeDefense, analyzeFile, analyzeShooting, ANALYZERS, getDetector, loadRoster, renderRoster, openPlayer, suggestMenu, buildOneon, gradeAll, loadModel, saveModel, clearModel, modelVals };
