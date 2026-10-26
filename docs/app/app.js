'use strict';
/* DOLL-E in-browser app: form + JSON editing of the full config, pre-flight checks, simulation runs,
   sticking calculator and the verification suite. The physics runs in worker.js (Pyodide). */

const $ = (id) => document.getElementById(id);
const COLORS = { truth: '#5b9dff', stereo: '#3ccf7e', '2d_relative': '#ff8a3d', '2d_normal': '#d36bd0' };
const LABELS = { truth: 'truth (3D normal speed)', stereo: 'stereo cameras', '2d_relative': '1 camera: 2D relative speed',
                 '2d_normal': '1 camera: 2D normal component' };

/* ------------------------------------------------------------------ worker RPC */
const worker = new Worker('worker.js');
let nextId = 1;
const pending = new Map();
let INFO = null, CFG = null, CH2 = null, TEMPLATE = null, BUSY = false;
// state of the lazily loaded tabs (declared here: showTab() runs before the code further down)
let SOURCES_LOADED = false, MISSION = null, FLIGHT_LOADED = false;

worker.onmessage = (ev) => {
  const m = ev.data;
  if (m.type === 'status') return loaderStep(m.step);
  if (m.type === 'fatal') return loaderError(m.msg);
  if (m.type === 'ready') return onReady(m.version);
  const p = pending.get(m.id);
  if (!p) return;
  if (m.type === 'progress') { if (p.onProgress) p.onProgress(JSON.parse(m.data)); return; }
  pending.delete(m.id);
  if (m.type === 'result') p.resolve(JSON.parse(m.data));
  else p.reject(new Error(m.msg));
};

function call(fn, arg, onProgress) {
  return new Promise((resolve, reject) => {
    const id = nextId++;
    pending.set(id, { resolve, reject, onProgress });
    worker.postMessage({ id, fn, arg: typeof arg === 'string' ? arg : JSON.stringify(arg ?? null) });
  });
}

/* ------------------------------------------------------------------ loading screen */
const LOAD_T0 = performance.now();
let LOAD_STEP = 0;
const loadTimer = setInterval(() => {
  const s = (performance.now() - LOAD_T0) / 1000;
  $('load-time').textContent = `${s.toFixed(0)} s` + (s > 45 ? ' · still working (a slow connection can take a minute)' : '');
}, 500);

function loaderStep(step) {
  LOAD_STEP = step;
  document.querySelectorAll('#load-steps li').forEach((li) => {
    const k = +li.dataset.step;
    li.className = k < step ? 'done' : k === step ? 'active' : '';
  });
  $('load-bar').style.width = `${Math.min(100, (step - 1) * 25 + 8)}%`;
}

function loaderError(msg) {
  clearInterval(loadTimer);
  $('loader').classList.add('error');
  $('load-title').textContent = 'The simulator could not start';
  $('load-sub').innerHTML = 'Your browser needs WebAssembly and an internet connection to download the Python runtime. ' +
    'Try reloading the page, or use a recent Chrome, Edge, Firefox or Safari.<br><span class="fail small">' + esc(msg) + '</span>';
}

async function onReady(version) {
  INFO = await call('info', null);
  TEMPLATE = await (await fetch('py/dolle_sim/viewer_template.html')).text();
  fillSelects();
  fillQuick();
  if (!CFG) setConfig(INFO.default_config);
  buildVerifyTable();
  ['btn-run', 'btn-preflight', 'btn-stick', 'btn-inv', 'btn-verify', 'btn-loads'].forEach((b) => ($(b).disabled = false));
  document.querySelectorAll('#load-steps li').forEach((li) => (li.className = 'done'));
  $('load-bar').style.width = '100%';
  clearInterval(loadTimer);
  setTimeout(() => { $('loader').style.display = 'none'; $('app').style.display = 'block'; showTab(); }, 350);
}

/* ------------------------------------------------------------------ tabs */
function showTab() {
  const t = (location.hash || '#simulate').slice(1);
  const tab = ['simulate', 'sticking', 'flight', 'verify', 'sources'].includes(t) ? t : 'simulate';
  if (tab === 'sources') loadSources();
  if (tab === 'flight') loadFlight();
  document.querySelectorAll('.tabs a').forEach((a) => a.classList.toggle('on', a.dataset.tab === tab));
  document.querySelectorAll('.pane').forEach((p) => p.classList.toggle('on', p.id === 'pane-' + tab));
}
window.addEventListener('hashchange', showTab);
showTab();

/* ------------------------------------------------------------------ config paths */
const key = (k) => (/^\d+$/.test(k) ? +k : k);
function getp(o, path) { return path.split('.').reduce((a, k) => (a == null ? undefined : a[key(k)]), o); }
function setp(o, path, v) {
  const ks = path.split('.');
  let a = o;
  for (let i = 0; i < ks.length - 1; i++) {
    const k = key(ks[i]);
    if (a[k] == null) a[k] = /^\d+$/.test(ks[i + 1]) ? [] : {};
    a = a[k];
  }
  a[key(ks[ks.length - 1])] = v;
}
function fmt(v) {
  if (v === null || v === undefined || Number.isNaN(v)) return '';
  if (typeof v !== 'number') return String(v);
  if (v === 0) return '0';
  const a = Math.abs(v);
  if (a < 1e-3 || a >= 1e6) return v.toExponential(3).replace(/\.?0+e/, 'e');
  return String(+v.toPrecision(5));
}

/* ------------------------------------------------------------------ form schema */
const GROUPS = [
  { title: 'Run', fields: [
    ['duration_s', 'Duration', 's'], ['dt_s', 'Time step', 'ms', 1e3], ['seed', 'Random seed', ''],
    ['contact.steps_per_contact', 'Contact sub-steps', 'per impact'], ['output_fps', 'Replay frame rate', 'fps']] },
  { title: 'Flight environment', fields: [
    ['environment.pressure_pa', 'Gas pressure', 'Pa'], ['environment.temperature_k', 'Gas temperature', 'K'],
    ['environment.gas', 'Gas', '', 1, 'sel', ['air', 'nitrogen', 'argon', 'helium']],
    ['environment.residual_accel_g.2', 'Residual acceleration (z)', 'g'], ['environment.jitter_rms_g', 'g-jitter rms', 'g'],
    ['environment.spin_rate_hz', 'Payload spin rate', 'Hz'], ['environment.temperature_gradient_k_m.2', 'Temperature gradient (z)', 'K/m'],
    ['environment.illumination.intensity_w_m2', 'Camera illumination', 'W/m²'], ['environment.altitude_m', 'Altitude', 'km', 1e-3]] },
  { title: 'Contacts', fields: [
    ['contact.wall_sticking', 'Wall sticking', '', 1, 'sel', ['jkr', 'never', 'always']],
    ['contact.friction_coefficient', 'Friction coefficient', ''], ['contact.tangential_restitution', 'Tangential restitution', ''],
    ['contact.contact_charging', 'Contact (tribo-)charging', '', 1, 'chk'], ['contact.fragmentation', 'Aggregate fragmentation', '', 1, 'chk']] },
  { title: 'Camera', fields: [
    ['camera.fps', 'Frame rate', 'fps'], ['camera.position_noise_m', 'Centroid noise', 'µm', 1e6],
    ['camera.track_frames', 'Frames tracked per side', ''], ['camera.depth_of_field_m', 'Depth of field', 'mm', 1e3]] },
  { title: 'Forces', checks: [
    ['forces.coulomb', 'Coulomb'], ['forces.space_charge', 'Space-charge field'], ['forces.polarization', 'Polarisation'],
    ['forces.image_charges', 'Image charges'], ['forces.self_gravity', 'Mutual gravity'], ['forces.lorentz', 'Lorentz (q v×B)'],
    ['forces.external_e_field', 'External E field'], ['forces.thermophoresis', 'Thermophoresis'], ['forces.photophoresis', 'Photophoresis'],
    ['forces.radiation_pressure', 'Radiation pressure'], ['forces.brownian', 'Brownian motion'], ['environment.gravity_gradient', 'Gravity gradient']] },
];
const CH_FIELDS = [
  ['name', 'Name', '', 1, 'text'], ['size_m', 'Chamber size (cube)', 'mm', 1e3, 'cube'],
  ['wall_material', 'Wall material', '', 1, 'selwall'], ['wall_adhesion_factor', 'Wall adhesion factor', '×'],
  ['wall_vibration.amplitude_m', 'Wall vibration amplitude', 'µm', 1e6], ['wall_vibration.frequency_hz', 'Wall vibration frequency', 'Hz'],
  ['particles.0.material', 'Dust material', '', 1, 'seldust'], ['particles.0.count', 'Number of grains', ''],
  ['particles.0.radius_m', 'Grain diameter', 'µm', 2e6], ['particles.0.radius_rel_sigma', 'Size scatter', 'rel.'],
  ['particles.0.surface_energy_log_sigma', 'Surface-energy scatter', 'ln σ'],
  ['particles.0.velocity.mode', 'Launch pattern', '', 1, 'sel', ['isotropic', 'jet']],
  ['particles.0.velocity.speed_mean_m_s', 'Launch speed mean', 'mm/s', 1e3], ['particles.0.velocity.speed_sigma_m_s', 'Launch speed σ', 'mm/s', 1e3],
  ['particles.0.spin.rate_hz_mean', 'Initial grain spin', 'Hz'], ['particles.0.charge_e_sigma', 'Initial charge σ', 'e'],
];

function inputFor(path, type, opts, scale) {
  let el;
  if (type === 'sel' || type === 'selwall' || type === 'seldust') {
    el = document.createElement('select');
    const list = type === 'selwall' ? INFO.wall_materials : type === 'seldust' ? INFO.dust_materials : opts;
    for (const o of list) el.add(new Option(o, o));
  } else if (type === 'chk') {
    el = document.createElement('input'); el.type = 'checkbox';
  } else {
    el = document.createElement('input'); el.type = type === 'text' ? 'text' : 'number'; el.step = 'any';
  }
  el.dataset.path = path; el.dataset.type = type || 'num'; el.dataset.scale = scale || 1;
  el.addEventListener('change', () => { readField(el); onConfigEdited(); });
  return el;
}

function readField(el) {
  const path = el.dataset.path, type = el.dataset.type, sc = +el.dataset.scale;
  if (path.startsWith('chambers.1') && !CFG.chambers[1]) return;
  let v;
  if (type === 'chk') v = el.checked;
  else if (type === 'num' || type === 'cube') { v = parseFloat(el.value) / sc; if (!Number.isFinite(v)) return; }
  else v = el.value;
  if (type === 'cube') v = [v, v, v];
  if (/(count|seed|track_frames|steps_per_contact)$/.test(path)) v = Math.round(v);
  setp(CFG, path, v);
}

function writeField(el) {
  const path = el.dataset.path, type = el.dataset.type, sc = +el.dataset.scale;
  const holder = path.startsWith('chambers.1') ? (CFG.chambers[1] ? CFG : { chambers: [null, CH2] }) : CFG;
  let v = getp(holder, path);
  if (type === 'cube' && Array.isArray(v)) v = v[0];
  if (type === 'chk') el.checked = !!v;
  else if (type === 'num' || type === 'cube') el.value = v === undefined ? '' : fmt(v * sc);
  else el.value = v ?? '';
  el.disabled = path.startsWith('chambers.1') && !CFG.chambers[1];
}

function buildForm() {
  const form = $('form');
  form.innerHTML = '';
  // chambers first (two columns)
  const fs = document.createElement('fieldset');
  fs.className = 'wide';
  fs.innerHTML = '<legend>Chambers (one per dust material)</legend>';
  const hdr = document.createElement('div'); hdr.className = 'row two';
  hdr.innerHTML = '<span></span><b>Chamber 1</b>';
  const lab = document.createElement('label'); lab.className = 'flex';
  const en = document.createElement('input'); en.type = 'checkbox'; en.id = 'ch2-on'; en.checked = !!CFG.chambers[1];
  en.addEventListener('change', () => {
    if (en.checked) { CFG.chambers[1] = CH2 || JSON.parse(JSON.stringify(CFG.chambers[0])); }
    else { CH2 = CFG.chambers[1]; CFG.chambers.length = 1; }
    refreshForm(); onConfigEdited();
  });
  lab.append(en, document.createTextNode(' Chamber 2'));
  hdr.append(lab); fs.append(hdr);
  for (const [p, label, unit, scale, type, opts] of CH_FIELDS) {
    const row = document.createElement('div'); row.className = 'row two';
    row.innerHTML = `<label>${label} <span class="u">${unit}</span></label>`;
    row.append(inputFor('chambers.0.' + p, type, opts, scale), inputFor('chambers.1.' + p, type, opts, scale));
    fs.append(row);
  }
  form.append(fs);
  for (const g of GROUPS) {
    const f = document.createElement('fieldset');
    f.innerHTML = `<legend>${g.title}</legend>`;
    if (g.fields) for (const [p, label, unit, scale, type, opts] of g.fields) {
      const row = document.createElement('div'); row.className = 'row';
      row.innerHTML = `<label>${label} <span class="u">${unit}</span></label>`;
      row.append(inputFor(p, type || 'num', opts, scale)); f.append(row);
    }
    if (g.checks) {
      const box = document.createElement('div'); box.className = 'checks';
      for (const [p, label] of g.checks) {
        const l = document.createElement('label'); l.append(inputFor(p, 'chk'), document.createTextNode(label)); box.append(l);
      }
      f.append(box);
    }
    form.append(f);
  }
}

function refreshForm() {
  if (!$('ch2-on')) buildForm();
  $('ch2-on').checked = !!CFG.chambers[1];
  document.querySelectorAll('#form [data-path]').forEach(writeField);
  buildMaterialTable();
  $('cfgjson').value = JSON.stringify(CFG, null, 2);
}

function setConfig(cfg) {
  CFG = JSON.parse(JSON.stringify(cfg));
  CFG.materials = CFG.materials || {};
  CH2 = CFG.chambers[1] || CH2 || JSON.parse(JSON.stringify(INFO.default_config.chambers[1]));
  buildForm();
  refreshForm();
  if ($('qs-a').options.length) {
    $('qs-a').value = CFG.chambers[0].particles[0].material;
    $('qs-b').value = CFG.chambers[1] ? CFG.chambers[1].particles[0].material : '';
    $('qs-d').value = fmt(CFG.chambers[0].particles[0].radius_m * 2e6);
    $('qs-n').value = CFG.chambers[0].particles[0].count;
    $('qs-t').value = CFG.duration_s;
    $('qs-p').value = CFG.environment.pressure_pa;
  }
  quickEstimate();
}

function onConfigEdited() {
  $('cfgjson').value = JSON.stringify(CFG, null, 2);
}


/* ------------------------------------------------------------------ quick setup */
const PRETTY = { graphite: 'Graphite (carbon)', amorphous_carbon: 'Amorphous carbon', silica: 'Silica (glass)',
  forsterite: 'Forsterite (olivine)', alumina: 'Alumina', carbon_silica_50_50: 'Carbon-silica mix', water_ice: 'Water ice' };
const pretty = (m) => PRETTY[m] || m;

function fillQuick() {
  const a = $('qs-a'), b = $('qs-b');
  a.innerHTML = ''; b.innerHTML = '';
  b.add(new Option('nothing (one dust only)', ''));
  for (const m of INFO.dust_materials) { a.add(new Option(pretty(m), m)); b.add(new Option(pretty(m), m)); }
  a.value = 'graphite'; b.value = 'silica';
  for (const id of ['qs-a', 'qs-b', 'qs-d', 'qs-mode', 'qs-speed', 'qs-n', 'qs-t', 'qs-p'])
    $(id).addEventListener('change', () => { $('qs-speed-box').style.display = $('qs-mode').value === 'custom' ? '' : 'none'; quickEstimate(); });
}

function quickEstimate() {
  const n = (+$('qs-n').value || 0) * ($('qs-b').value ? 2 : 1);
  const secs = 0.05 * n * (+$('qs-t').value || 0) * (CFG?.contact?.steps_per_contact ?? 100) / 100;
  $('estimate').textContent = `takes about ${secs < 90 ? Math.max(5, Math.round(secs / 5) * 5) + ' seconds' : (secs / 60).toFixed(1) + ' minutes'}`;
}

async function applyQuick() {
  const mats = [$('qs-a').value, $('qs-b').value].filter(Boolean);
  const d = +$('qs-d').value, n = Math.round(+$('qs-n').value), T = +$('qs-t').value, P = +$('qs-p').value;
  let speeds;
  if ($('qs-mode').value === 'auto') {
    const th = await call('thresholds', { materials_list: mats, diameter_um: d, materials: CFG.materials });
    speeds = mats.map((m) => 0.9 * th[m]);             // launch around each material's sticking speed
  } else {
    speeds = mats.map(() => +$('qs-speed').value / 1e3);
  }
  const base = CFG.chambers[0];
  const chambers = mats.map((m, i) => {
    const ch = JSON.parse(JSON.stringify(CFG.chambers[i] || (i === 1 ? CH2 : null) || base));
    ch.name = m;
    const g = ch.particles[0];
    g.material = m; g.count = n; g.radius_m = d * 0.5e-6;
    g.velocity = g.velocity || {};
    g.velocity.speed_mean_m_s = speeds[i]; g.velocity.speed_sigma_m_s = 0.55 * speeds[i];
    ch.wall_vibration = ch.wall_vibration || { frequency_hz: 200, axes: [0, 1, 2] };
    const f = ch.wall_vibration.frequency_hz || 200;
    ch.wall_vibration.amplitude_m = 0.8 * speeds[i] / (2 * Math.PI * f);   // walls re-energise grains at ~ the same speed
    return ch;
  });
  CFG.chambers = chambers;
  CFG.duration_s = T;
  CFG.environment.pressure_pa = P;
  refreshForm();
  onConfigEdited();
}

/* ------------------------------------------------------------------ materials editor */
const MAT_COLS = [['rho', 'ρ', 1], ['E', 'E', 1e-9], ['nu', 'ν', 1], ['gamma', 'γ', 1], ['e_dry', 'e_dry', 1],
  ['work_function', 'φ', 1], ['permittivity', 'ε', 1], ['yield_pressure', 'p_y', 1e-9], ['thermal_conductivity', 'k', 1],
  ['absorptivity', 'absorb', 1]];
function buildMaterialTable() {
  const names = [...INFO.dust_materials, ...INFO.wall_materials];
  let h = '<table><thead><tr><th>material</th>' + MAT_COLS.map((c) => `<th>${c[1]}</th>`).join('') + '<th></th></tr></thead><tbody>';
  for (const n of names) {
    h += `<tr><td>${n}${INFO.wall_materials.includes(n) ? ' <span class="muted small">(wall)</span>' : ''}</td>`;
    for (const [f, , sc] of MAT_COLS) {
      const base = INFO.materials[n][f];
      const ov = CFG.materials?.[n]?.[f];
      const v = ov ?? base;
      h += `<td><input data-mat="${n}" data-f="${f}" data-sc="${sc}" value="${v === null ? 'inf' : fmt(v * sc)}"${ov !== undefined ? ' style="border-color:#ffb547"' : ''}></td>`;
    }
    h += `<td><a href="#" data-reset="${n}" class="small">reset</a></td></tr>`;
  }
  $('mattable').innerHTML = h + '</tbody></table><p class="small muted">Edited values are outlined in orange.</p>';
  $('mattable').querySelectorAll('input').forEach((el) => el.addEventListener('change', () => {
    const n = el.dataset.mat, f = el.dataset.f, sc = +el.dataset.sc;
    const raw = el.value.trim().toLowerCase();
    const v = raw === 'inf' ? Infinity : parseFloat(raw) / sc;
    if (Number.isNaN(v)) return;
    const base = INFO.materials[n][f] ?? Infinity;
    CFG.materials[n] = CFG.materials[n] || {};
    if (Math.abs(v - base) <= 1e-12 * Math.abs(base)) delete CFG.materials[n][f];
    else CFG.materials[n][f] = Number.isFinite(v) ? v : 1e300;
    if (!Object.keys(CFG.materials[n]).length) delete CFG.materials[n];
    buildMaterialTable(); onConfigEdited();
  }));
  $('mattable').querySelectorAll('[data-reset]').forEach((a) => a.addEventListener('click', (e) => {
    e.preventDefault(); delete CFG.materials[a.dataset.reset]; buildMaterialTable(); onConfigEdited();
  }));
}

/* ------------------------------------------------------------------ buttons */
$('btn-apply-json').addEventListener('click', () => {
  try {
    const c = JSON.parse($('cfgjson').value);
    if (!Array.isArray(c.chambers) || !c.chambers.length) throw new Error('config needs a non-empty "chambers" list');
    setConfig(c); $('jsonmsg').textContent = 'Applied.'; $('jsonmsg').className = 'status pass';
  } catch (e) { $('jsonmsg').textContent = 'Invalid JSON: ' + e.message; $('jsonmsg').className = 'status fail'; }
});
$('btn-reset').addEventListener('click', () => { CH2 = null; setConfig(INFO.default_config); });
$('btn-download-cfg').addEventListener('click', () => download('dolle_config.json', JSON.stringify(CFG, null, 2), 'application/json'));
$('file-cfg').addEventListener('change', async (e) => {
  const f = e.target.files[0]; if (!f) return;
  try { setConfig(JSON.parse(await f.text())); } catch (err) { alert('Could not read config: ' + err.message); }
});

function download(name, text, type) {
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([text], { type: type || 'text/plain' }));
  a.download = name; a.click(); setTimeout(() => URL.revokeObjectURL(a.href), 5000);
}

function setBusy(b) {
  BUSY = b;
  ['btn-run', 'btn-preflight', 'btn-stick', 'btn-inv', 'btn-verify', 'btn-loads'].forEach((id) => ($(id).disabled = b));
}

/* ------------------------------------------------------------------ pre-flight */
$('btn-preflight').addEventListener('click', async () => {
  if (BUSY) return;
  setBusy(true);
  $('results').innerHTML = '';
  try { renderPreflight(await call('preflight', CFG)); }
  catch (e) { $('preflight-out').innerHTML = `<p class="fail">Error: ${esc(e.message)}</p>`; }
  setBusy(false);
});

function esc(s) { return String(s).replace(/[&<>]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c])); }
function mm(v) { return v == null ? '–' : fmt(v * 1e3); }
function mm3(v) { return v == null ? '–' : String(+(v * 1e3).toPrecision(3)); }

function renderPreflight(pfs) {
  let h = '<h2>Pre-flight check</h2><div class="grid c2">';
  pfs.forEach((pf, k) => {
    const g = pf.groups[0];
    h += `<div class="card"><h3>${esc(pf.chamber)}: ${esc(g.material)}</h3><table>
      <tr><td>Predicted sticking speed, grain–grain (DEM)</td><td class="num"><b>${mm(g.v_stick_pp_DEM_m_s)} mm/s</b></td></tr>
      <tr><td>Simple energy estimate (no damping)</td><td class="num">${mm(g.v_stick_pp_energy_estimate_m_s)} mm/s</td></tr>
      <tr><td>Sticking speed on the wall</td><td class="num">${mm(g.v_stick_wall_DEM_m_s)} mm/s</td></tr>
      <tr><td>Wall hits expected to stick</td><td class="num">${fmt(100 * g.wall_capture_probability_per_hit)} %</td></tr>
      <tr><td>Gas stopping time / Knudsen number</td><td class="num">${fmt(g.stopping_time_s)} s / ${fmt(g.knudsen_number)}</td></tr>
      <tr><td>Expected grain–grain collisions</td><td class="num">~${Math.round(g.expected_pp_collisions_total)}</td></tr>
      <tr><td>Spin-down time / Brownian speed</td><td class="num">${fmt(g.spin_down_time_s)} s / ${fmt(g.brownian_speed_m_s * 1e6)} µm/s</td></tr>
      </table><canvas class="chart" id="budget${k}" style="height:360px;margin-top:10px"></canvas>`;
    if (pf.warnings.length) h += '<ul class="warnings">' + pf.warnings.map((w) => `<li>${esc(w)}</li>`).join('') + '</ul>';
    h += '</div>';
  });
  $('preflight-out').innerHTML = h + '</div>';
  pfs.forEach((pf, k) => {
    const b = pf.groups[0].acceleration_budget_m_s2;
    const items = Object.entries(b).filter(([, v]) => v > 0).map(([n, v]) => ({ label: n.replace(/_/g, ' '), value: v / 9.80665 }))
      .sort((a, c) => c.value - a.value);
    barChart($('budget' + k), items, 'typical acceleration [g]');
  });
}

/* ------------------------------------------------------------------ run */
$('btn-run').addEventListener('click', async () => {
  if (BUSY) return;
  setBusy(true);
  $('preflight-out').innerHTML = ''; $('results').innerHTML = '';
  $('run-progress').style.display = 'block'; $('bar').style.width = '0%';
  $('runpct').textContent = '0%'; $('runmsg').textContent = 'Setting up the chambers…';
  const t0 = performance.now();
  try {
    await applyQuick();
    const res = await call('run', CFG, (p) => {
      const frac = (p.chamber + p.t / p.T) / p.n_chambers;
      $('bar').style.width = (100 * frac).toFixed(1) + '%';
      $('runpct').textContent = Math.floor(100 * frac) + '%';
      const el = (performance.now() - t0) / 1000;
      const eta = frac > 0.02 ? Math.max(0, el / frac - el) : NaN;
      const e = p.events;
      $('runmsg').textContent = `Simulating ${pretty(p.name)}: ${p.t.toFixed(1)} of ${p.T} s · ${e.stick + e.bounce} collisions so far` +
        (Number.isFinite(eta) ? ` · about ${Math.max(1, Math.round(eta))} s left` : '');
    });
    $('bar').style.width = '100%'; $('runpct').textContent = '100%';
    $('runmsg').textContent = `Done in ${((performance.now() - t0) / 1000).toFixed(0)} s.`;
    setTimeout(() => ($('run-progress').style.display = 'none'), 1500);
    renderResults(res);
  } catch (e) {
    $('results').innerHTML = `<p class="fail">Simulation error: ${esc(e.message)}</p>`;
  }
  setBusy(false);
});

function fitTxt(f) { return f ? `<b>${mm(f.v50_m_s)}</b> <span class="muted">[${mm(f.v50_lo_1sigma)}–${mm(f.v50_hi_1sigma)}]</span>` : '<span class="muted">no fit</span>'; }

function renderResults(res) {
  let h = '<h2>Result</h2><div class="grid c2">';
  res.chambers.forEach((c) => {
    const m = c.measurement, g = c.preflight.groups[0], s = c.stats;
    const tr = m.threshold_truth_normal_speed, st = m.threshold_stereo_normal, one = m.threshold_2d_relative;
    const v = tr ? tr.v50_m_s : g.v_stick_pp_DEM_m_s;
    h += `<div class="card answer"><div class="muted">${esc(pretty(g.material))} grains, ${fmt(2e6 * g.radius_m)} µm</div>
      <div class="headline">stick below ${mm3(v)} <small>mm/s</small></div>
      <p>${tr ? `Found from ${s.monomer_monomer_collisions} simulated collisions` +
                (s.monomer_monomer_collisions < 50 ? ' (a rough number: run longer or with more grains to sharpen it)' : '') +
                `. The theory predicts ${mm3(g.v_stick_pp_DEM_m_s)} mm/s.`
              : `Too few collisions to measure: run longer or with more grains. The theory predicts ${mm3(g.v_stick_pp_DEM_m_s)} mm/s.`}</p>
      <p>Two cameras (stereo) would measure <b>${st ? mm3(st.v50_m_s) + ' mm/s' : 'not enough data'}</b>;
        a single camera ${one ? mm3(one.v50_m_s) + ' mm/s' : 'not enough data'}${one && tr && Math.abs(one.v50_m_s / tr.v50_m_s - 1) > 0.2 ? ' (unreliable)' : ''}.</p>
      <p class="small muted">${s.grain_grain_collisions} collisions, ${s.grain_grain_sticks} stuck · ${s.free_monomers} of ${s.n_grains} grains still floating free at the end.</p></div>`;
  });
  h += '</div><div class="grid c2" style="margin-top:16px">';
  res.chambers.forEach((c, k) => { h += `<canvas class="chart" id="stk${k}"></canvas>`; });
  h += '</div><h2>Watch it</h2><iframe class="viewer" id="viewer"></iframe>';
  h += '<details><summary>Detailed numbers, charts and downloads</summary><div class="inner"><div class="grid c2">';
  res.chambers.forEach((c, k) => {
    const s = c.stats, m = c.measurement, g = c.preflight.groups[0];
    h += `<div class="card"><h3>${esc(c.name)}: ${esc(g.material)}</h3><table>
      <tr><td>Grain–grain impacts (stuck)</td><td class="num">${s.grain_grain_collisions} (${s.grain_grain_sticks})</td></tr>
      <tr><td>Free single-grain impacts</td><td class="num">${s.monomer_monomer_collisions}</td></tr>
      <tr><td>Wall impacts / stuck</td><td class="num">${s.wall_collisions} / ${s.wall_sticks}</td></tr>
      <tr><td>Grains still free at the end</td><td class="num">${s.free_monomers} of ${s.n_grains}</td></tr>
      <tr><td>Largest aggregate</td><td class="num">${s.max_aggregate_size} grains</td></tr></table>
      <h3>Sticking speed [mm/s] (50% point, 1σ range)</h3><table>
      <tr><td>Model prediction (pre-flight)</td><td class="num">${mm(g.v_stick_pp_DEM_m_s)}</td></tr>
      <tr><td>Truth: 3D normal speed</td><td class="num">${fitTxt(m.threshold_truth_normal_speed)}</td></tr>
      <tr><td>Stereo cameras</td><td class="num">${fitTxt(m.threshold_stereo_normal)}</td></tr>
      <tr><td>1 camera, 2D relative speed</td><td class="num">${fitTxt(m.threshold_2d_relative)}</td></tr>
      <tr><td>1 camera, 2D normal component</td><td class="num">${fitTxt(m.threshold_2d_normal)}</td></tr>
      <tr><td>Impacts onto wall-bound grains</td><td class="num">${fitTxt(m.threshold_truth_on_wall_bound_grains)}</td></tr></table>
      <canvas class="chart" id="ts${k}" style="margin-top:12px;height:240px"></canvas>
      <div class="btns"><button class="btn ghost" data-dl="${k}" data-kind="collisions">collisions.csv</button>
      <button class="btn ghost" data-dl="${k}" data-kind="camera">camera_measurements.csv</button></div></div>`;
  });
  h += '</div><div class="btns"><button class="btn ghost" id="dl-summary">Download summary JSON</button></div></div></details>';
  $('results').innerHTML = h;
  res.chambers.forEach((c, k) => {
    const series = [];
    for (const key of ['truth', 'stereo', '2d_relative', '2d_normal']) {
      const pts = c.measurement_points[key].map((p) => [p[0] * 1e3, p[1]]);
      series.push({ pts, color: COLORS[key], type: 'scatter', label: null });
      const f = { truth: c.measurement.threshold_truth_normal_speed, stereo: c.measurement.threshold_stereo_normal,
        '2d_relative': c.measurement.threshold_2d_relative, '2d_normal': c.measurement.threshold_2d_normal }[key];
      if (f && pts.length) {
        const xs = pts.map((p) => p[0]); const lo = Math.min(...xs), hi = Math.max(...xs);
        const line = [];
        for (let i = 0; i <= 120; i++) {
          const x = lo * Math.pow(hi / lo, i / 120);
          line.push([x, 1 / (1 + Math.exp((Math.log(x / 1e3) - Math.log(f.v50_m_s)) / f.log_width))]);
        }
        series.push({ pts: line, color: COLORS[key], type: 'line', label: `${LABELS[key]}: ${mm3(f.v50_m_s)} mm/s` });
      } else series.push({ pts: [], color: COLORS[key], type: 'line', label: `${LABELS[key]}: no fit` });
    }
    plot($('stk' + k), { series, xlog: true, ylim: [-0.03, 1.03], xlabel: 'collision speed [mm/s]', ylabel: 'fraction that stick',
      vlines: [{ x: c.preflight.groups[0].v_stick_pp_DEM_m_s * 1e3, color: '#cfd6e2', label: 'theory' }], title: `${pretty(c.name)}: which collisions stick` });
    const ts = c.timeseries, n = c.stats.n_grains;
    plot($('ts' + k), { series: [
      { pts: ts.map((r) => [r.t, r.free_monomers / n]), color: '#5b9dff', type: 'line', label: 'free grains (fraction)' },
      { pts: ts.map((r) => [r.t, r.wall_monomers / n]), color: '#8f98a8', type: 'line', label: 'on walls (fraction)', dash: [5, 4] },
      { pts: ts.map((r) => [r.t, r.stick / Math.max(1, ts[ts.length - 1].stick + ts[ts.length - 1].bounce)]), color: '#3ccf7e', type: 'line', label: 'grain–grain sticks (norm.)' },
      { pts: ts.map((r) => [r.t, r.bounce / Math.max(1, ts[ts.length - 1].stick + ts[ts.length - 1].bounce)]), color: '#ff8a3d', type: 'line', label: 'grain–grain bounces (norm.)' },
    ], xlabel: 'time [s]', ylabel: 'fraction', ylim: [0, 1.02], title: 'cloud evolution' });
  });
  $('results').querySelectorAll('[data-dl]').forEach((b) => b.addEventListener('click', () => {
    const c = res.chambers[+b.dataset.dl];
    if (b.dataset.kind === 'collisions') download(`${c.name}_collisions.csv`, c.collisions_csv, 'text/csv');
    else download(`${c.name}_camera_measurements.csv`, c.camera_csv, 'text/csv');
  }));
  $('dl-summary').addEventListener('click', () => {
    const slim = { config: CFG, chambers: res.chambers.map((c) => ({ name: c.name, stats: c.stats, measurement: c.measurement, preflight: c.preflight })) };
    download('dolle_summary.json', JSON.stringify(slim, null, 2), 'application/json');
  });
  const html = TEMPLATE.split('__TITLE__').join('Browser run: ' + CFG.name).replace('/*__DATA__*/null', JSON.stringify(res.viewer));
  $('viewer').src = URL.createObjectURL(new Blob([html], { type: 'text/html' }));
}

/* ------------------------------------------------------------------ sticking calculator */
function fillSelects() {
  for (const id of ['st-a', 'st-b', 'inv-m', 'inv-p']) {
    const s = $(id); s.innerHTML = '';
    if (id === 'inv-p') s.add(new Option('(same material)', ''));
    for (const m of [...INFO.dust_materials, ...INFO.wall_materials]) s.add(new Option(m, m));
  }
  $('st-a').value = 'graphite'; $('st-b').value = 'graphite'; $('inv-m').value = 'graphite';
}

$('btn-stick').addEventListener('click', async () => {
  if (BUSY) return;
  setBusy(true);
  $('st-out').innerHTML = '<p class="status">Computing…</p>';
  try {
    const r = await call('sticking', { mat_a: $('st-a').value, mat_b: $('st-b').value, diameter_um: +$('st-d').value,
      gamma_log_sigma: +$('st-gs').value, radius_rel_sigma: +$('st-rs').value, samples: +$('st-n').value, materials: CFG.materials });
    $('st-out').innerHTML = `<div class="grid c2"><div class="card"><table>
      <tr><td>Sticking speed (DEM, identical spheres)</td><td class="num"><b>${mm(r.v_stick_dem)} mm/s</b></td></tr>
      <tr><td>JKR hysteresis energy estimate</td><td class="num">${mm(r.v_stick_jkr_hysteresis)} mm/s</td></tr>
      <tr><td>Simple energy estimate (no damping)</td><td class="num">${mm(r.v_stick_energy_estimate)} mm/s</td></tr>
      <tr><td>Plastic yield above</td><td class="num">${r.v_yield == null ? '∞' : fmt(r.v_yield)} m/s</td></tr>
      <tr><td>Pull-off force / break energy</td><td class="num">${fmt(r.pull_off_force_N)} N / ${fmt(r.break_energy_J)} J</td></tr>
      </table><p class="small muted">Material edits from the Simulate tab are applied here too.</p></div>
      <div><canvas class="chart" id="stc1"></canvas><canvas class="chart" id="stc2" style="margin-top:12px;height:240px"></canvas></div></div>`;
    const v = r.velocities.map((x) => x * 1e3);
    plot($('stc1'), { series: [{ pts: v.map((x, i) => [x, r.stick_probability[i]]), color: '#3ccf7e', type: 'line', label: 'sticking probability (with scatter)' }],
      xlog: true, ylim: [-0.03, 1.03], xlabel: 'impact speed [mm/s]', ylabel: 'P(stick)',
      vlines: [{ x: r.v_stick_dem * 1e3, color: '#cfd6e2', label: 'DEM threshold' }], title: `${$('st-a').value} on ${$('st-b').value}` });
    plot($('stc2'), { series: [{ pts: v.map((x, i) => [x, r.restitution[i]]), color: '#ff8a3d', type: 'line', label: 'normal restitution (0 = stuck)' }],
      xlog: true, ylim: [-0.03, 1.03], xlabel: 'impact speed [mm/s]', ylabel: 'e_n', title: 'coefficient of restitution' });
  } catch (e) { $('st-out').innerHTML = `<p class="fail">${esc(e.message)}</p>`; }
  setBusy(false);
});

$('btn-inv').addEventListener('click', async () => {
  if (BUSY) return;
  setBusy(true);
  try {
    const r = await call('invert', { material: $('inv-m').value, partner: $('inv-p').value, diameter_um: +$('inv-d').value,
      v_stick_m_s: +$('inv-v').value / 1e3, materials: CFG.materials });
    $('inv-out').innerHTML = `Effective surface energy <b>γ = ${fmt(r.gamma_eff)} J/m²</b> (database ${fmt(r.gamma_database)}, ratio ${fmt(r.ratio)}).`;
  } catch (e) { $('inv-out').innerHTML = `<span class="fail">${esc(e.message)}</span>`; }
  setBusy(false);
});

/* ------------------------------------------------------------------ flight (mission) */
const escA = (s) => esc(s).replace(/"/g, '&quot;');
const g3 = (x) => (x === 0 ? '0' : Math.abs(x) >= 1e-3 && Math.abs(x) < 1e4 ? String(+x.toPrecision(3)) : x.toExponential(1));
const SCENARIOS = {
  nominal: 'Shaker: 50 g at 200 Hz from T+90 s for 10 s (vacuum, tubules standing)',
  no_release: 'No release mechanism',
  kick: 'Three single-cycle kicks: 5000 g at 5 kHz, at T+90, 150 and 210 s',
  strong_shaker: 'Strong shaker: 500 g at 1 kHz from T+90 s for 5 s',
  rough_grains: 'Nominal shaker, wall stickiness × 0.02 (rough grains or an anti-stick coating)',
  lying: 'Nominal shaker, tubules lying flat on the deck',
  spin_1hz: 'Nominal shaker, 1 Hz residual spin after de-spin (the guide’s upper value)',
  gas_100pa: 'Nominal shaker, 100 Pa of air in the tubules',
  gas_1atm: 'Nominal shaker, 1 atm of air in the tubules',
  no_drag: 'Nominal shaker, no drag deceleration in the coasts',
  'kick+lying': 'Kicks, tubules lying flat on the deck',
  'kick+spin_1hz': 'Kicks, 1 Hz residual spin after de-spin (the guide’s upper value)',
  'kick+gas_100pa': 'Kicks, 100 Pa of air in the tubules',
  'kick+gas_1atm': 'Kicks, 1 atm of air in the tubules',
  'rough_grains+shock_oblique': 'Weakly bound dust; parachute and splash shocks at 45 degrees (half sideways)',
  'rough_grains+hang_nose_up': 'Weakly bound dust; hanging nose up under the parachute',
};
const FACE = { '-z': 'bottom (aft)', '+z': 'top (nose end)', '-x': 'side −x', '+x': 'side +x', '-y': 'side −y', '+y': 'side +y' };

async function loadFlight() {
  if (FLIGHT_LOADED || !INFO) return;
  FLIGHT_LOADED = true;
  try {
    MISSION = await (await fetch('../data/mission_config.json', { cache: 'no-cache' })).json();
    renderTubeEditor();
  } catch (e) { $('fl-tubes').innerHTML = '<p class="fail">Could not load the mission configuration.</p>'; }
  fetch('../data/mission_results.json', { cache: 'no-cache' }).then((r) => r.json()).then(renderMissionResults)
    .catch(() => { $('fl-results').innerHTML = '<p class="muted small">No stored runs yet.</p>'; });
}

function renderTubeEditor() {
  const opt = (list, v) => list.map((m) => `<option${m === v ? ' selected' : ''}>${esc(m)}</option>`).join('');
  let h = '<div style="overflow-x:auto"><table class="small"><tr><th>Tubule</th><th>Dust</th><th>Sizes (µm)</th><th>Glass</th>' +
    '<th title="Multiplies the work of adhesion: roughness or coatings lower it">Stickiness ×</th><th>Orientation</th><th>Deck position x, y (cm)</th></tr>';
  MISSION.tubules.forEach((t, i) => {
    const sd = t.dust.size_distribution || {};
    h += `<tr><td><input id="fl-n${i}" value="${escA(t.name)}" style="min-width:150px"></td>` +
      `<td><select id="fl-d${i}">${opt(INFO.dust_materials, t.dust.material)}</select></td>` +
      `<td><input id="fl-lo${i}" type="number" step="any" value="${sd.min_m * 1e6}" style="width:62px"> – ` +
      `<input id="fl-hi${i}" type="number" step="any" value="${sd.max_m * 1e6}" style="width:62px"></td>` +
      `<td><select id="fl-w${i}">${opt(INFO.wall_materials, t.wall_material)}</select></td>` +
      `<td><input id="fl-f${i}" type="number" step="any" value="${t.wall_adhesion_factor ?? 1}" style="width:64px"></td>` +
      `<td><select id="fl-o${i}">${opt(['standing', 'lying'], t.orientation || 'standing')}</select></td>` +
      `<td><input id="fl-x${i}" type="number" step="any" value="${(t.deck_position_m[0] * 100).toFixed(1)}" style="width:62px"> ` +
      `<input id="fl-y${i}" type="number" step="any" value="${(t.deck_position_m[1] * 100).toFixed(1)}" style="width:62px"></td></tr>`;
  });
  $('fl-tubes').innerHTML = h + '</table></div><p class="small muted">Sizes: a sieve range with 90% of grains inside it ' +
    '(assumed log-normal). Stickiness scatter: log-normal, σ = 0.6 in ln w.</p>';
}

function readTubes() {
  const m = JSON.parse(JSON.stringify(MISSION));
  m.tubules.forEach((t, i) => {
    t.name = $('fl-n' + i).value || `T${i + 1}`;
    t.dust.material = $('fl-d' + i).value;
    t.dust.size_distribution = { type: 'sieve', min_m: +$('fl-lo' + i).value * 1e-6, max_m: +$('fl-hi' + i).value * 1e-6,
      fraction_in_range: t.dust.size_distribution?.fraction_in_range ?? 0.9 };
    t.wall_material = $('fl-w' + i).value;
    t.wall_adhesion_factor = +$('fl-f' + i).value;
    t.orientation = $('fl-o' + i).value;
    t.deck_position_m = [+$('fl-x' + i).value / 100, +$('fl-y' + i).value / 100];
  });
  return m;
}

$('btn-loads').addEventListener('click', async () => {
  if (BUSY) return;
  setBusy(true);
  $('fl-msg').textContent = 'Computing…';
  try {
    const r = await call('launch_loads', readTubes());
    let h = '';
    for (const t of r.tubules) {
      const th = t.threshold_g;
      h += `<h3 style="margin:22px 0 6px">${esc(t.name)}</h3><p class="small muted" style="margin:0 0 8px">${esc(t.dust)} on ` +
        `${esc(t.wall)}, ${esc(t.orientation)}, ${g3(t.r_far_m * 100)} cm from the spin axis. Pull-off thresholds: ` +
        `<b>10% of grains below ${g3(th.p10)} g, half below ${g3(th.median)} g</b>, 90% below ${g3(th.p90)} g.</p>` +
        '<div style="overflow-x:auto"><table class="small"><tr><th>Phase</th><th>T+ (s)</th><th>Pressing onto the bottom, up to (g)</th><th>Pulling off the bottom, up to (g)</th>' +
        '<th title="Spin (centrifugal + Euler), plus the guide’s ~10 g lateral load during the ascent (worst case). May roll grains (not modelled).">Load along the face (g)</th><th>Grains pulled off</th></tr>';
      for (const p of t.phases) {
        h += `<tr><td>${esc(p.phase)}</td><td class="num">${p.t0.toFixed(1)}–${p.t1.toFixed(1)}</td>` +
          `<td class="num">${g3(p.max_press_onto_aft_g)}</td><td class="num">${p.max_pull_off_aft_g > 0 ? g3(p.max_pull_off_aft_g) : '0'}</td>` +
          `<td class="num">${g3(p.max_tangential_g)}</td>` +
          `<td class="num">${(100 * p.fraction_pulled_off_quasi_static).toFixed(1)}%</td></tr>`;
      }
      const v = t.vibration_estimate;
      h += '</table></div>' + `<p class="small muted" style="margin:6px 0 0">Random-vibration estimate (test level, ` +
        `Gaussian): ${(100 * v.ground_random_test).toFixed(1)}% freed in the ground vibration test, ` +
        `${(100 * v.flight_lift_off_to_burnout).toFixed(1)}% from lift-off to burnout.</p>`;
    }
    $('fl-out').innerHTML = h + '<p class="small muted">"Pulled off" counts pull-off straight away from the face only. ' +
      'Rolling along the face is not modelled; it may start at ~0.1–1 g for ~100 µm grains (an uncertain estimate), ' +
      'below the spin loads.</p>';
    $('fl-msg').textContent = '';
  } catch (e) { $('fl-msg').innerHTML = `<span class="fail">${esc(e.message)}</span>`; }
  setBusy(false);
});

function renderMissionResults(d) {
  const order = Object.keys(SCENARIOS).filter((k) => d[k]).concat(Object.keys(d).filter((k) => !SCENARIOS[k]));
  if (!order.length) { $('fl-results').innerHTML = '<p class="muted small">No stored runs yet.</p>'; return; }
  const names = d[order[0]].tubules.map((t) => t.tubule);
  let h = '<div style="overflow-x:auto"><table class="small"><tr><th>Design option</th>' +
    names.map((n) => `<th>${esc(n.replace(/_/g, ' '))}</th>`).join('') + '</tr>';
  for (const k of order) {
    h += `<tr><td><b>${esc(k.replace(/_/g, ' '))}</b><br><span class="muted">${esc(SCENARIOS[k] || '')}</span></td>`;
    for (const t of d[k].tubules) {
      const med = t.ug_median_impact_speed_m_s;
      h += `<td class="num">${t.grains_released} of ${t.n_grains} grains released (${t.detachment_events} release events), ` +
        `${t.free_at_power_off ?? '?'} free at power-off (T+332 s), ${t.free_at_end} at the end<br>` +
        `${t.ug_grain_grain_impacts} grain–grain impacts (${t.ug_monomer_sticks} of ${t.ug_monomer_impacts} single-grain stuck` +
        `${med != null ? `, median ${g3(med * 1e3)} mm/s` : ''})<br><span class="muted">5 s after splashdown: ` +
        Object.entries(t.wall_bound_by_face).sort((a, b) => b[1] - a[1]).map(([f, n]) => `${n} on the ${FACE[f] || f}`).join(', ') +
        `</span></td>`;
    }
    h += '</tr>';
  }
  $('fl-results').innerHTML = h + '</table></div><p class="small muted">Grains released: distinct grains pulled off a ' +
    'wall at least once; a grain that sticks again and is freed again counts once, but each release is an event. ' +
    'Impacts are counted in the microgravity window only (T+88 to T+313 s, above ~100 km); each run continues ' +
    'through re-entry and the parachute to splashdown. 800 grains per tubule is an example amount. Faces are for ' +
    'standing tubules (bottom = aft end).</p>';
}

/* ------------------------------------------------------------------ verification */
function buildVerifyTable() {
  const tb = document.querySelector('#vtable tbody');
  $('btn-verify').textContent = `Run all ${INFO.checks.length} tests`;
  tb.innerHTML = INFO.checks.map((c) => `<tr id="v-${c}"><td>${c.replace(/^check_/, '').replace(/_/g, ' ')}</td><td class="muted small">not run yet</td><td></td></tr>`).join('');
  fetch('../data/rebound_check.json').then((r) => r.json()).then(renderRebound).catch(() => {});
  fetch('../data/lammps_check.json').then((r) => r.json()).then(renderLammps).catch(() => {});
}

$('btn-verify').addEventListener('click', async () => {
  if (BUSY) return;
  setBusy(true);
  let ok = 0;
  const t0 = performance.now();
  for (const [i, c] of INFO.checks.entries()) {
    $('vmsg').textContent = `Running ${i + 1}/${INFO.checks.length}: ${c}…`;
    const row = $('v-' + c);
    row.children[2].innerHTML = '<span class="muted">running…</span>';
    try {
      const r = await call('verify', c);
      row.innerHTML = `<td><b>${esc(r.name)}</b><br><span class="small muted">${esc(r.what)}</span></td>
        <td class="small">${esc(r.value)}<br><span class="muted">required: ${esc(r.criterion)}</span></td>
        <td class="${r.passed ? 'pass' : 'fail'}">${r.passed ? 'PASS' : 'FAIL'}</td>`;
      if (r.passed) ok++;
    } catch (e) { row.children[2].innerHTML = '<span class="fail">ERROR</span>'; row.children[1].textContent = e.message; }
  }
  const tot = INFO.checks.length;
  $('vmsg').innerHTML = `<span class="${ok === tot ? 'pass' : 'fail'}">${ok} of ${tot} tests passed</span> (${((performance.now() - t0) / 1000).toFixed(0)} s, in your browser).`;
  setBusy(false);
});

function renderRebound(d) {
  const keys = [['grain_grain_impacts', 'grain–grain impacts'], ['grain_grain_sticks', 'grain–grain sticks'],
    ['free_monomer_stick_fraction', 'free single-grain stick fraction'],
    ['free_monomer_median_speed_m_s', 'median single-grain impact speed (m/s)'], ['wall_impacts', 'wall impacts'], ['wall_sticks', 'wall sticks']];
  let h = '<div class="grid c2">';
  for (const [name, r] of Object.entries(d)) {
    h += `<div class="card"><h3>${esc(name)} (${r.n_grains} grains, ${r.duration_s} s)</h3><table><tr><th></th><th>DOLL-E</th><th>REBOUND</th></tr>`;
    for (const [k, lab] of keys) h += `<tr><td>${lab}</td><td class="num">${fmt(r.dolle_engine[k])}</td><td class="num">${fmt(r.rebound_engine[k])}</td></tr>`;
    h += `<tr><td>ballistic trajectories (vs IAS15)</td><td colspan="2" class="num">max difference ${fmt(r.ballistic_check.max_position_difference_m * 1e9)} nm</td></tr></table></div>`;
  }
  $('rebound').innerHTML = h + '</div>';
}

function renderLammps(d) {
  const pct = (x) => `${(100 * x).toFixed(2)}%`;
  let h = '<div class="grid c2"><div class="card"><h3>Collisions (undamped, 100 µm grains)</h3><table>' +
    '<tr><th></th><th>DOLL-E</th><th>LAMMPS</th><th>LAMMPS − DOLL-E</th></tr>';
  for (const c of d.collisions) {
    h += `<tr><td>${esc(c.material)}: sticking threshold (mm/s)</td><td class="num">${fmt(c.dolle_threshold_force_break_m_s * 1e3)}</td>` +
      `<td class="num">${fmt(c.lammps_threshold_m_s * 1e3)}</td><td class="num">${pct(c.rel_diff_same_criterion)}</td></tr>`;
    for (const e of c.restitution) {
      h += `<tr><td class="small">restitution at ${fmt(e.v_m_s * 1e3)} mm/s</td><td class="num">${e.dolle_e.toFixed(4)}</td>` +
        `<td class="num">${e.lammps_e.toFixed(4)}</td><td class="num">${(e.lammps_e - e.dolle_e >= 0 ? '+' : '') + (e.lammps_e - e.dolle_e).toFixed(4)}</td></tr>`;
    }
  }
  h += '</table></div><div class="card"><h3>Pull-off from a glass wall</h3><table><tr><th></th><th>1.5πwR (N)</th><th>LAMMPS (N)</th><th>LAMMPS − 1.5πwR</th></tr>';
  for (const p of d.pulloff) {
    h += `<tr><td>${esc(p.grain)} on ${esc(p.wall)}</td><td class="num">${p.jkr_pulloff_N.toExponential(4)}</td>` +
      `<td class="num">${p.lammps_detach_force_N.toExponential(4)}</td><td class="num">${pct(p.rel_diff)}</td></tr>`;
  }
  h += '</table><p class="small muted" style="margin-bottom:0">A grain resting on a flat wall is pulled by a slowly rising ' +
    'force until it comes off. DOLL-E uses 1.5πwR (JKR) to decide when wall-bound grains are released.</p></div></div>';
  $('lammps').innerHTML = h;
}

/* ------------------------------------------------------------------ sources (provenance) */
async function loadSources() {
  if (SOURCES_LOADED) return;
  SOURCES_LOADED = true;
  try {
    const d = await (await fetch('../data/provenance.json', { cache: 'no-cache' })).json();
    $('src-summary').innerHTML = d.summary.map((p) => `<p style="margin:0 0 10px">${p}</p>`).join('') +
      '<p class="muted small" style="margin:0">Origin: <span class="org published">published</span> equation or value ' +
      'from the cited paper · <span class="org standard">standard</span> textbook physics or numerics · ' +
      '<span class="org derived">derived</span> derived for DOLL-E · <span class="org assumed">assumed</span> ' +
      'a choice we made, not from a source.</p>';
    let h = '';
    for (const sec of d.sections) {
      h += `<h3 style="margin-top:28px">${esc(sec.title)}</h3><table class="src"><thead><tr><th style="width:23%">Component</th>` +
        `<th style="width:33%">Based on</th><th style="width:22%">How it is checked</th><th>Assumptions and limits</th></tr></thead><tbody>`;
      for (const c of sec.items) {
        const reb = c.rebound === 'none' ? '' : `<br><span class="org rebound">REBOUND: ${esc(c.rebound)}</span>`;
        const chk = c.checks.length ? `<br><span class="muted small">Verify: ${c.checks.map(esc).join(', ')}</span>` : '';
        h += `<tr><td><b>${esc(c.component)}</b><br><span class="tag">${esc(c.code)}</span><br>` +
          `<span class="org ${esc(c.origin)}">${esc(c.origin)}</span>${reb}</td><td>${esc(c.basis)}` +
          (c.refs ? `<br><span class="muted small">${esc(c.refs)}</span>` : '') + `</td><td>${esc(c.checked)}${chk}</td>` +
          `<td class="muted">${esc(c.limits)}</td></tr>`;
      }
      h += '</tbody></table>';
    }
    if (d.sensitivity) {
      h += '<h3 style="margin-top:28px">How much the inputs matter (100 µm grains)</h3><table class="src"><thead><tr>' +
        '<th style="width:40%">Change</th><th>Effect on the 50% sticking speed</th></tr></thead><tbody>' +
        d.sensitivity.map((s) => `<tr><td>${esc(s.change)}</td><td>${esc(s.effect)}</td></tr>`).join('') + '</tbody></table>';
    }
    $('src-table').innerHTML = h;
  } catch (e) {
    SOURCES_LOADED = false;
    $('src-summary').innerHTML = '<p class="fail">Could not load the source list.</p>';
  }
}

/* ------------------------------------------------------------------ charts (small canvas plotting) */
function setupCanvas(cv) {
  const dpr = window.devicePixelRatio || 1;
  const W = cv.clientWidth || 600, H = cv.clientHeight || 300;
  cv.width = W * dpr; cv.height = H * dpr;
  const ctx = cv.getContext('2d'); ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, W, H);
  ctx.font = '12px system-ui, sans-serif';
  return { ctx, W, H };
}

function niceTicks(lo, hi, n = 6) {
  const span = hi - lo || 1, step0 = span / n, mag = Math.pow(10, Math.floor(Math.log10(step0)));
  const step = [1, 2, 2.5, 5, 10].map((f) => f * mag).find((s) => span / s <= n) || 10 * mag;
  const out = []; for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-12 * span; v += step) out.push(+v.toPrecision(10));
  return out;
}

function plot(cv, o) {
  const { ctx, W, H } = setupCanvas(cv);
  const leg = o.series.filter((s) => s.label);
  const cols = W > 560 ? 2 : 1;
  const m = { l: 56, r: 16, t: 28 + 16 * Math.ceil(leg.length / cols), b: 40 };
  const pts = o.series.flatMap((s) => s.pts).filter((p) => Number.isFinite(p[0]) && Number.isFinite(p[1]) && (!o.xlog || p[0] > 0));
  let [x0, x1] = o.xlim || [Math.min(...pts.map((p) => p[0])), Math.max(...pts.map((p) => p[0]))];
  for (const v of o.vlines || []) if (Number.isFinite(v.x) && (!o.xlog || v.x > 0)) { x0 = Math.min(x0, v.x); x1 = Math.max(x1, v.x); }
  if (!pts.length || !Number.isFinite(x0)) { ctx.fillStyle = '#8f98a8'; ctx.fillText('not enough collisions to plot yet: run longer or with more grains', 20, H / 2); return; }
  let [y0, y1] = o.ylim || [Math.min(...pts.map((p) => p[1])), Math.max(...pts.map((p) => p[1]))];
  if (x0 === x1) { x0 *= 0.9; x1 *= 1.1; } if (y0 === y1) { y0 -= 1; y1 += 1; }
  if (o.xlog) { x0 = Math.log10(x0) - 0.05; x1 = Math.log10(x1) + 0.05; }
  const X = (x) => m.l + ((o.xlog ? Math.log10(x) : x) - x0) / (x1 - x0) * (W - m.l - m.r);
  const Y = (y) => H - m.b - (y - y0) / (y1 - y0) * (H - m.t - m.b);
  ctx.strokeStyle = '#2a303c'; ctx.fillStyle = '#8f98a8'; ctx.lineWidth = 1;
  const xt = o.xlog ? (() => { const t = []; for (let k = Math.floor(x0); k <= Math.ceil(x1); k++) for (const f of [1, 2, 5]) { const v = Math.log10(f) + k; if (v >= x0 && v <= x1) t.push(v); } return t; })() : niceTicks(x0, x1);
  for (const v of xt) {
    const px = m.l + (v - x0) / (x1 - x0) * (W - m.l - m.r);
    ctx.beginPath(); ctx.moveTo(px, m.t); ctx.lineTo(px, H - m.b); ctx.stroke();
    const val = o.xlog ? Math.pow(10, v) : v;
    ctx.fillText(fmt(+val.toPrecision(3)), px - 10, H - m.b + 15);
  }
  for (const v of niceTicks(y0, y1, 5)) {
    const py = Y(v); ctx.beginPath(); ctx.moveTo(m.l, py); ctx.lineTo(W - m.r, py); ctx.stroke();
    ctx.fillText(fmt(v), 6, py + 4);
  }
  ctx.fillStyle = '#d8dde6';
  if (o.xlabel) ctx.fillText(o.xlabel, (W - ctx.measureText(o.xlabel).width) / 2, H - 8);
  if (o.title) { ctx.font = '600 13px system-ui, sans-serif'; ctx.fillText(o.title, m.l, 16); ctx.font = '12px system-ui, sans-serif'; }
  if (o.ylabel) { ctx.save(); ctx.translate(14, (H - m.b + m.t) / 2 + ctx.measureText(o.ylabel).width / 2); ctx.rotate(-Math.PI / 2); ctx.fillText(o.ylabel, 0, 0); ctx.restore(); }
  ctx.save(); ctx.beginPath(); ctx.rect(m.l, m.t, W - m.l - m.r, H - m.t - m.b); ctx.clip();
  for (const v of o.vlines || []) {
    if (!Number.isFinite(v.x)) continue;
    ctx.strokeStyle = v.color; ctx.setLineDash([4, 4]); ctx.beginPath(); ctx.moveTo(X(v.x), m.t); ctx.lineTo(X(v.x), H - m.b); ctx.stroke(); ctx.setLineDash([]);
    ctx.fillStyle = v.color; ctx.fillText(v.label || '', X(v.x) + 4, m.t + 12);
  }
  for (const s of o.series) {
    ctx.strokeStyle = s.color; ctx.fillStyle = s.color; ctx.lineWidth = 2; ctx.setLineDash(s.dash || []);
    const p = s.pts.filter((q) => Number.isFinite(q[1]) && (!o.xlog || q[0] > 0));
    if (s.type === 'scatter') for (const q of p) { ctx.beginPath(); ctx.arc(X(q[0]), Y(q[1]), 3.2, 0, 6.283); ctx.fill(); }
    else { ctx.beginPath(); p.forEach((q, i) => (i ? ctx.lineTo(X(q[0]), Y(q[1])) : ctx.moveTo(X(q[0]), Y(q[1])))); ctx.stroke(); }
    ctx.setLineDash([]);
  }
  ctx.restore();
  // legend: rows above the plot area
  const colW = (W - m.l - m.r) / cols;
  leg.forEach((s, i) => {
    const x = m.l + (i % cols) * colW, y = 32 + 16 * Math.floor(i / cols);
    ctx.fillStyle = s.color; ctx.fillRect(x, y - 4, 14, 3);
    ctx.fillStyle = '#d8dde6';
    let txt = s.label;
    while (ctx.measureText(txt).width > colW - 24 && txt.length > 4) txt = txt.slice(0, -2);
    ctx.fillText(txt === s.label ? txt : txt + '…', x + 18, y + 1);
  });
}

function barChart(cv, items, xlabel) {
  const { ctx, W, H } = setupCanvas(cv);
  const m = { l: 190, r: 16, t: 10, b: 34 };
  const vals = items.map((i) => Math.log10(i.value));
  const x0 = Math.floor(Math.min(...vals)), x1 = Math.ceil(Math.max(...vals));
  const X = (v) => m.l + (v - x0) / (x1 - x0) * (W - m.l - m.r);
  const bh = (H - m.t - m.b) / items.length;
  ctx.strokeStyle = '#2a303c'; ctx.fillStyle = '#8f98a8';
  for (let k = x0; k <= x1; k++) { ctx.beginPath(); ctx.moveTo(X(k), m.t); ctx.lineTo(X(k), H - m.b); ctx.stroke(); if ((k - x0) % 2 === 0 || x1 - x0 < 10) ctx.fillText('1e' + k, X(k) - 10, H - m.b + 14); }
  items.forEach((it, i) => {
    const y = m.t + i * bh;
    ctx.fillStyle = it.label.includes('residual') ? '#ffb547' : '#5b9dff';
    ctx.fillRect(m.l, y + 2, Math.max(1, X(Math.log10(it.value)) - m.l), bh - 4);
    ctx.fillStyle = '#d8dde6'; ctx.fillText(it.label, 6, y + bh / 2 + 4);
  });
  ctx.fillStyle = '#d8dde6'; ctx.fillText(xlabel, (W - ctx.measureText(xlabel).width) / 2 + m.l / 2, H - 6);
}

