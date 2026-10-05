"""Dashboard HTML -- pure string template, no framework/persona dependency,
so this module stays importable in isolation. Tabs: Agent loop
(wake/heard/reply events + sensors and service health), Mind (its inner life),
Dreams & wishes (nightly dream notes, wishes, self-written rules), Camera (live video with the object detector's
boxes, via /api/camera.mjpg and /api/objects, + what Rocky made of its last look, via /api/vision), Jev (every question asked of the Jev decision
model and its answer, via /api/jev), Stats (is it getting better, per day, via
/api/stats), and Files (browse the raw state/ files
every service generates). Fed by /api/stream (Server-Sent Events)
plus /api/files* for the Files tab.
"""

PAGE = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>OpenBot dashboard</title>
<style>
  body { font-family: -apple-system, sans-serif; background: #111; color: #eee; margin: 0; padding: 16px; }
  h1 { font-size: 18px; margin-bottom: 12px; }
  .tabs { display: flex; gap: 4px; margin-bottom: 12px; border-bottom: 1px solid #333; }
  .tab { padding: 8px 16px; cursor: pointer; color: #999; border-bottom: 2px solid transparent; }
  .tab.active { color: #eee; border-bottom-color: #7cf; }
  .panel { display: none; }
  .panel.active { display: block; }
  .cols { display: flex; gap: 16px; flex-wrap: wrap; }
  .col { flex: 1; min-width: 320px; }
  .card { background: #1c1c1c; border-radius: 8px; padding: 12px; margin-bottom: 12px; }
  .row { display: flex; justify-content: space-between; padding: 2px 0; border-bottom: 1px solid #2a2a2a; }
  .log { height: 480px; overflow-y: auto; font-size: 13px; }
  .log div { padding: 3px 0; border-bottom: 1px solid #222; }
  .kind-wake { color: #7cf; } .kind-reply { color: #9f9; }
  .kind-safety { color: #fc6; } .kind-mic { color: #888; }
  .ok { color: #7f7; } .degraded { color: #fc6; } .stale { color: #fa6; }
  .failing, .missing { color: #f77; }
  .files-layout { display: flex; gap: 16px; }
  .file-list { width: 260px; flex-shrink: 0; }
  .file-list div { padding: 4px 6px; cursor: pointer; border-radius: 4px; font-size: 13px; word-break: break-all; }
  .file-list div:hover { background: #2a2a2a; }
  .file-list div.selected { background: #2f3f4f; color: #7cf; }
  .file-content { flex: 1; min-width: 0; }
  .file-content pre { background: #1c1c1c; border-radius: 8px; padding: 12px; overflow: auto;
                       max-height: 560px; font-size: 12px; white-space: pre-wrap; word-break: break-all; }
  .jev pre { background: #151515; border-radius: 6px; padding: 8px; overflow: auto; max-height: 360px;
             font-size: 12px; white-space: pre-wrap; word-break: break-all; }
  .jev summary { cursor: pointer; color: #7cf; font-size: 13px; margin-top: 6px; }
  .stats-wrap { overflow-x: auto; }
  .stats td, .stats th { padding: 4px 8px; text-align: right; border-bottom: 1px solid #2a2a2a; white-space: nowrap; }
  .stats th { color: #999; font-weight: normal; font-size: 12px; }
  .stats td:first-child, .stats th:first-child { text-align: left; }
</style>
</head>
<body>
<h1>OpenBot</h1>
<div class="tabs">
  <div class="tab active" data-tab="live">Agent loop</div>
  <div class="tab" data-tab="mind">Mind</div>
  <div class="tab" data-tab="dreams">Dreams &amp; wishes</div>
  <div class="tab" data-tab="camera">Camera</div>
  <div class="tab" data-tab="jev">Jev</div>
  <div class="tab" data-tab="stats">Stats</div>
  <div class="tab" data-tab="files">Files</div>
</div>

<div class="panel active" id="panel-live">
  <div class="cols">
    <div class="col">
      <div class="card"><div class="log" id="log"></div></div>
    </div>
    <div class="col">
      <div class="card"><b>Sensors</b><div id="sensors"></div></div>
      <div class="card"><b>Services</b><div id="health"></div></div>
    </div>
  </div>
</div>

<div class="panel" id="panel-mind">
  <div class="cols">
    <div class="col">
      <div class="card"><b>Right now</b><div id="mind-now" style="margin-top:6px"></div></div>
      <div class="card"><b>Today so far</b><div id="mind-summary" style="margin-top:6px;line-height:1.4"></div></div>
      <div class="card"><b>Goals</b><div id="mind-goals"></div></div>
      <div class="card"><b>Reminders &amp; watches</b><div id="mind-plans"></div></div>
      <div class="card"><b>How people reacted</b><div id="mind-reactions" style="font-size:13px"></div></div>
    </div>
    <div class="col" style="flex:1.4">
      <div class="card"><b>Journal</b> <span style="color:#999;font-size:12px">-- one line per thing heard, said, noticed, thought, found, did</span>
        <div class="log" id="mind-journal" style="margin-top:6px"></div></div>
    </div>
  </div>
</div>

<div class="panel" id="panel-dreams">
  <div class="cols">
    <div class="col" style="flex:1.4">
      <div class="card"><b>Dreams</b> <span style="color:#999;font-size:12px">-- each night it goes over the day and keeps what matters</span>
        <div id="dreams-list" style="margin-top:8px"></div></div>
    </div>
    <div class="col">
      <div class="card"><b>Wishes</b> <span style="color:#999;font-size:12px">-- things it wishes it could do or have; for you to read</span>
        <div id="dreams-wishes" style="margin-top:6px"></div></div>
      <div class="card"><b>Rules it wrote for itself</b> <span style="color:#999;font-size:12px">-- what works with people, rewritten nightly</span>
        <div id="dreams-rules" style="margin-top:6px;font-size:13px"></div></div>
    </div>
  </div>
</div>

<div class="panel" id="panel-camera">
  <div class="cols">
    <div class="col" style="flex:2">
      <div class="card"><div style="position:relative;max-width:640px">
        <img id="cam-img" alt="Camera stream unavailable (openbot-camera down?)" style="width:100%;border-radius:6px;display:block">
        <canvas id="cam-boxes" style="position:absolute;inset:0;width:100%;height:100%;pointer-events:none"></canvas>
      </div></div>
    </div>
    <div class="col">
      <div class="card"><b>What it sees</b><div id="cam-scene" style="margin-top:8px"></div></div>
      <div class="card"><b>Last look each way</b><div id="cam-dirs"></div></div>
      <div class="card" style="color:#999;font-size:13px">Live video (~10 fps), with boxes around what the object
        detector sees (green: people). The description is from the bot's
        last look -- it thinks about a frame every ~90s, or whenever it decides to <code>look</code>.</div>
    </div>
  </div>
</div>

<div class="panel jev" id="panel-jev">
  <div class="card" style="color:#999;font-size:13px">Every question asked of Jev (TypeSafe's decision model) --
    after each of the bot's thoughts: is this worth texting you on WhatsApp? -- with everything sent and Jev's
    whole answer. Newest first; refreshes while open. All of it is kept, one file a day, in state/jev/ --
    the training data for a classifier of our own.</div>
  <div id="jev-list"></div>
</div>

<div class="panel" id="panel-stats">
  <div class="card stats-wrap"><b>Is he getting better?</b> <span style="color:#999;font-size:12px">-- per day,
    newest first. Texts: the ones he started and what came back. Confusion: thoughts that the room changed,
    "something is moving" notices, and WhatsApp replies that may claim a move a text can't make.</span>
    <table class="stats" id="stats-table" style="margin-top:8px;border-collapse:collapse;font-size:13px"></table></div>
</div>

<div class="panel" id="panel-files">
  <div class="files-layout">
    <div class="card file-list" id="file-list"></div>
    <div class="card file-content"><pre id="file-content">Select a file.</pre></div>
  </div>
</div>

<script>
const log = document.getElementById('log');
const sensorsEl = document.getElementById('sensors');
const healthEl = document.getElementById('health');

function render(data) {
  log.innerHTML = data.events.map(e =>
    `<div class="kind-${e.kind}">[${new Date(e.ts*1000).toLocaleTimeString()}] ${e.text}</div>`
  ).reverse().join('');
  const s = data.sensors || {};
  sensorsEl.innerHTML = `
    <div class="row"><span>Distance</span><span>${s.distance ?? '?'} cm</span></div>
    <div class="row"><span>Grayscale</span><span>${(s.grayscale || []).join(', ')}</span></div>
    <div class="row"><span>Battery</span><span>${s.battery_pct != null ? s.battery_pct.toFixed(0) + '% (' + s.battery_v.toFixed(2) + 'V)' : '?'}</span></div>
    <div class="row"><span>Cliff latch</span><span>${s.latches?.cliff ?? '-'}</span></div>
    <div class="row"><span>Danger latch</span><span>${s.latches?.danger ?? '-'}</span></div>
    <div class="row"><span>Caution latch</span><span>${s.latches?.caution ?? '-'}</span></div>
    <div class="row"><span>Mood</span><span>${data.session?.mood ?? '-'}</span></div>
    <div class="row"><span>In session</span><span>${data.session?.in_session ?? '-'}</span></div>
  `;
  healthEl.innerHTML = Object.entries(data.health || {}).map(([name, status]) =>
    `<div class="row"><span>${name}</span><span class="${status}">${status}</span></div>`
  ).join('');
}

const source = new EventSource('/api/stream');
source.onmessage = (e) => render(JSON.parse(e.data));
fetch('/api/state').then(r => r.json()).then(render);

// --- Tabs ---
document.querySelectorAll('.tab').forEach(tab => {
  tab.addEventListener('click', () => {
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
    tab.classList.add('active');
    document.getElementById('panel-' + tab.dataset.tab).classList.add('active');
    if (tab.dataset.tab === 'files') loadFileList();
    setStream(tab.dataset.tab === 'camera');
    if (tab.dataset.tab === 'camera') loadCamera();
    if (tab.dataset.tab === 'mind') loadMind();
    if (tab.dataset.tab === 'dreams') loadDreams();
    if (tab.dataset.tab === 'jev') loadJev();
    if (tab.dataset.tab === 'stats') loadStats();
  });
});

// --- Stats tab ---
const STAT_COLS = [['day', 'Day'], ['thoughts', 'Thoughts'], ['said', 'Said'], ['heard', 'Heard'],
  ['texts_started', 'Texts he started'], ['replied', 'Replied'], ['reactions', 'Reactions'], ['no_reply', 'No reply'],
  ['reply_rate', 'Reply rate'], ['jev_calls', 'Jev calls'], ['jev_said_text', 'Jev said text'],
  ['room_changed_thoughts', '"Room changed"'], ['moving_notices', '"Something moving"'],
  ['possible_move_claims', 'Possible false moves'], ['held_back', 'Held back']];
function loadStats() {
  fetch('/api/stats').then(r => r.json()).then(days => {
    const cell = (k, v) => k === 'reply_rate' ? (v === null ? '-' : Math.round(v * 100) + '%') : esc(String(v ?? '-'));
    document.getElementById('stats-table').innerHTML =
      '<tr>' + STAT_COLS.map(([, h]) => `<th>${h}</th>`).join('') + '</tr>' +
      days.map(d => '<tr>' + STAT_COLS.map(([k]) => `<td>${cell(k, d[k])}</td>`).join('') + '</tr>').join('');
  });
}

// --- Jev tab ---
function loadJev() {
  fetch('/api/jev').then(r => r.json()).then(calls => {
    document.getElementById('jev-list').innerHTML = calls.map(c => {
      const q = ((c.request || {}).questions || {}).pick || {};
      const a = ((c.response || {}).answers || {}).pick;
      const verdict = a
        ? `<b style="color:${a.choice === 'text' ? '#9f9' : '#ccc'}">${esc(a.choice)}</b> (confidence ${Number(a.confidence).toFixed(2)})`
        : `<span style="color:#f77">${esc(c.error || 'no answer')}</span>`;
      return `<div class="card">` +
        `<div class="row"><span>${new Date(c.ts * 1000).toLocaleString()}</span>` +
        `<span>${verdict} &middot; ${c.ms ?? '-'} ms &middot; HTTP ${c.status ?? '-'}</span></div>` +
        `<div style="margin:6px 0">${esc(q.instructions)}</div>` +
        Object.entries(q.criteria || {}).map(([k, v]) =>
          `<div style="font-size:12px;color:#999"><b>${esc(k)}</b>: ${esc(v)}</div>`).join('') +
        `<details><summary>State sent</summary><pre>${esc(JSON.stringify((c.request || {}).state, null, 2))}</pre></details>` +
        `<details><summary>Jev's whole answer</summary><pre>${esc(JSON.stringify(c.response ?? null, null, 2))}</pre></details>` +
        `</div>`;
    }).join('') || `<div class="card" style="color:#999">No questions yet. Jev is asked after each thought once its
      API key is set: OPENBOT_JEV_API_KEY in /etc/openbot/jev.env on the robot.</div>`;
  });
}
setInterval(() => {
  if (document.getElementById('panel-jev').classList.contains('active')) loadJev();
}, 5000);

// --- Dreams & wishes tab ---
function loadDreams() {
  fetch('/api/dreams').then(r => r.json()).then(d => {
    document.getElementById('dreams-list').innerHTML = (d.dreams || []).map(n =>
      `<div style="margin-bottom:12px"><div style="color:#c9f;margin-bottom:4px">🌙 ${esc(n.night)}</div>` +
      `<div style="white-space:pre-wrap;font-size:13px;line-height:1.5">${esc(n.text)}</div></div>`
    ).join('') || '<div style="color:#999">No dreams yet -- it dreams once a night, after midnight.</div>';
    document.getElementById('dreams-wishes').innerHTML = (d.wishes || []).map(w => `<div class="row"><span>✨ ${esc(w)}</span></div>`).join('')
      || '<div style="color:#999">No wishes yet.</div>';
    document.getElementById('dreams-rules').innerHTML = (d.rules || []).map(r => `<div class="row"><span>${esc(r)}</span></div>`).join('')
      || '<div style="color:#999">No rules yet -- written after a few days of reactions.</div>';
  });
}

// --- Mind tab ---
const KIND_COLOR = {heard: '#7cf', said: '#9f9', thought: '#ccc', noticed: '#fc6', found: '#c9f', planned: '#f9c',
  goal: '#f9c', resolved: '#9f9', remembered: '#c9f', reflex: '#fa6', reaction: '#7fd', held_back: '#888'};
function timeOf(ts) { return new Date(ts * 1000).toLocaleTimeString([], {hour: 'numeric', minute: '2-digit'}); }
function loadMind() {
  fetch('/api/mind').then(r => r.json()).then(m => {
    document.getElementById('mind-now').innerHTML =
      `<div class="row"><span>Mode</span><span>${{awake: '☀️ Awake', asleep: '😴 Asleep'}[m.mode] || '-'}${m.in_session ? ' -- in a conversation' : ''}</span></div>` +
      `<div class="row"><span>Mood</span><span>${esc(m.mood || '-')}</span></div>` +
      `<div class="row"><span>Sees</span><span>${esc(m.who || 'nobody')}</span></div>` +
      `<div class="row"><span>Resting</span><span>${m.resting_until ? 'until ' + timeOf(m.resting_until) : 'no'}</span></div>`;
    document.getElementById('mind-summary').textContent = m.summary || 'Nothing summarized yet today.';
    document.getElementById('mind-goals').innerHTML = (m.goals || []).map((g, i) =>
      `<div class="row"><span>${i + 1}. ${esc(g.text)}</span><span style="color:#999;font-size:12px">${esc(g.since)}</span></div>`
    ).join('') || '<div style="color:#999">No open goals.</div>';
    document.getElementById('mind-plans').innerHTML =
      (m.reminders || []).map(r => `<div class="row"><span>⏰ ${esc(r.about)}</span><span>${timeOf(r.at)}</span></div>`).join('') +
      (m.watches || []).map(w => `<div class="row"><span>👁 ${esc(w.for)}: ${esc(w.about)}</span></div>`).join('') +
      (m.tasks || []).map(t => `<div class="row"><span>☐ ${esc(t.item)}</span><span>${esc(t.who || '')}</span></div>`).join('') ||
      '<div style="color:#999">Nothing scheduled.</div>';
    document.getElementById('mind-reactions').innerHTML = (m.reactions || []).map(r => `<div>${esc(r)}</div>`).join('') ||
      '<div style="color:#999">No reactions recorded yet.</div>';
    document.getElementById('mind-journal').innerHTML = (m.journal || []).slice().reverse().map(line => {
      const k = (line.match(/\\[(\\w+)\\]/) || [])[1];
      return `<div style="color:${KIND_COLOR[k] || '#eee'}">${esc(line)}</div>`;
    }).join('');
  });
}
setInterval(() => {
  if (document.getElementById('panel-mind').classList.contains('active')) loadMind();
}, 4000);

// --- Camera tab ---
function esc(t) { const d = document.createElement('div'); d.textContent = t ?? ''; return d.innerHTML; }
function loadCamera() {
  fetch('/api/vision').then(r => r.json()).then(v => {
    const ago = v.ts ? Math.round((Date.now() / 1000 - v.ts)) : null;
    document.getElementById('cam-scene').innerHTML = v.scene
      ? `<div>${esc(v.scene)}</div><div style="color:#999;font-size:13px;margin-top:6px">looking ${esc(v.direction)}, ${ago}s ago${v.changed ? ' -- changed: ' + esc(v.what_changed) : ''}</div>`
      : 'No look yet.';
    document.getElementById('cam-dirs').innerHTML = Object.entries(v.by_direction || {}).map(([d, s]) =>
      `<div class="row"><span>${esc(d)}</span><span style="text-align:right;max-width:75%">${esc(s)}</span></div>`).join('');
  });
}
// Live MJPEG straight from openbot-camera; only connected while the tab is
// open, so a background dashboard tab doesn't hold a stream.
const camImg = document.getElementById('cam-img');
// Live boxes over the video, ~4x a second while the tab is open (asking keeps the
// detector running on every frame -- /api/objects).
const camBoxes = document.getElementById('cam-boxes');
let boxTimer = null;
function drawBoxes(found) {
  const r = camBoxes.getBoundingClientRect();
  camBoxes.width = r.width; camBoxes.height = r.height;
  const g = camBoxes.getContext('2d');
  g.lineWidth = 2; g.font = '13px sans-serif';
  for (const o of found) {
    const [l, t, rt, b] = o.box;
    g.strokeStyle = g.fillStyle = o.name === 'person' ? '#38a169' : '#3182ce';
    g.strokeRect(l * r.width, t * r.height, (rt - l) * r.width, (b - t) * r.height);
    g.fillText(`${o.name} ${Math.round(o.score * 100)}%`, l * r.width + 3, Math.max(13, t * r.height - 4));
  }
}
function setStream(on) {
  if (on) camImg.src = '/api/camera.mjpg';
  else camImg.removeAttribute('src');  // closes the stream
  clearInterval(boxTimer);
  boxTimer = on ? setInterval(() => fetch('/api/objects').then(r => r.json()).then(drawBoxes).catch(() => {}), 250) : null;
  if (!on) drawBoxes([]);
}
setInterval(() => {
  if (document.getElementById('panel-camera').classList.contains('active')) loadCamera();
}, 5000);

// --- Files tab ---
const fileListEl = document.getElementById('file-list');
const fileContentEl = document.getElementById('file-content');
let selectedFile = null;

function loadFileList() {
  fetch('/api/files').then(r => r.json()).then(names => {
    fileListEl.innerHTML = names.map(n =>
      `<div data-name="${n}" class="${n === selectedFile ? 'selected' : ''}">${n}</div>`
    ).join('');
    fileListEl.querySelectorAll('div[data-name]').forEach(el => {
      el.addEventListener('click', () => selectFile(el.dataset.name));
    });
  });
}

function selectFile(name) {
  selectedFile = name;
  fileListEl.querySelectorAll('div[data-name]').forEach(el => {
    el.classList.toggle('selected', el.dataset.name === name);
  });
  fileContentEl.textContent = 'Loading...';
  fetch('/api/files/' + name).then(r => r.json()).then(data => {
    if (data.error) { fileContentEl.textContent = 'Error: ' + data.error; return; }
    try {
      fileContentEl.textContent = JSON.stringify(JSON.parse(data.content), null, 2);
    } catch {
      fileContentEl.textContent = data.content;
    }
  });
}

// Refresh the file list periodically while the tab is open (new health
// files, etc. appear over time) -- cheap, and simpler than a second
// SSE stream just for a file listing.
setInterval(() => {
  if (document.getElementById('panel-files').classList.contains('active')) loadFileList();
}, 5000);
</script>
</body>
</html>"""
