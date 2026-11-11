// DOLL-E 3-D page: draws what python computed (dolle_core/sim3d.py). No physics is computed here: the recorded
// replays come from tools/make_sim3d_replays.py (data/*.json) and the live run from the same python files running
// in the browser (worker.js). This file only plays frames back, counts events and plots the momentum ledger.
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const $ = (id) => document.getElementById(id);
const C = { stick: "#41d18b", bounce: "#ff8a3d", breakup: "#ff6b77", accent: "#6aa5ff", muted: "#8b93a3",
  line: "#2b3342", text: "#e6e9ef", warn: "#f5b54a", attract: "#5fd4ff", bg: "#121722" };
const SLOW_UNTIL = 0.5;         // s: the cloud phase, saved every 4 ms step (frames_note of the replay)
const SLOW_FACTOR = 10;         // shown 10x slower when "slow motion" is ticked
const FLASH_S = 0.9;            // how long a stick/bounce marker shows, in seconds on screen
const GAP_MM = 9;               // gap between the two tubules in the picture
const GRAPHITE_LIGHT = "#9a9a9a"; // palette index 1 (single graphite grain, #4b4b4b) lightened for the dark background
const reduceMotion = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;

// ------------------------------------------------------------------ small helpers
function fmt(x, d = 3) { return Number.isFinite(x) ? x.toFixed(d) : "-"; }
function sci(x) {
  if (!Number.isFinite(x)) return "-";
  if (x === 0) return "0";
  const e = Math.floor(Math.log10(Math.abs(x)));
  const m = x / 10 ** e;
  return `${m.toFixed(1)}×10<sup>${e}</sup>`;
}
function niceFloor(x) {               // largest 1, 2 or 5 x 10^k not above x
  const e = 10 ** Math.floor(Math.log10(x));
  for (const m of [5, 2, 1]) if (m * e <= x) return m * e;
  return e;
}
function bisectRight(arr, x) {        // number of entries <= x in the sorted array arr
  let lo = 0, hi = arr.length;
  while (lo < hi) { const mid = (lo + hi) >> 1; if (arr[mid] <= x) lo = mid + 1; else hi = mid; }
  return lo;
}
function norm3(a, b, c) { return Math.hypot(a, b, c); }
function setupCanvas2d(cv) {
  const dpr = window.devicePixelRatio || 1;
  const w = cv.clientWidth, h = cv.clientHeight;
  if (cv.width !== Math.round(w * dpr) || cv.height !== Math.round(h * dpr)) {
    cv.width = Math.round(w * dpr); cv.height = Math.round(h * dpr);
  }
  const g = cv.getContext("2d");
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  g.clearRect(0, 0, w, h);
  return { g, w, h };
}

// A small line plot: series [{x[], y[], color, width, dash, label}], drawn up to x <= xCut.
function linePlot(g, box, series, xmax, ymin, ymax, opts = {}) {
  const { x0, y0, w, h } = box;
  const X = (x) => x0 + (x / xmax) * w;
  const Y = (y) => y0 + h - ((y - ymin) / (ymax - ymin)) * h;
  g.strokeStyle = C.line; g.lineWidth = 1; g.setLineDash([]);
  g.strokeRect(x0, y0, w, h);
  if (ymin < 0 && ymax > 0) { g.beginPath(); g.moveTo(x0, Y(0)); g.lineTo(x0 + w, Y(0)); g.stroke(); }
  for (const hl of opts.hlines || []) {
    g.strokeStyle = hl.color; g.setLineDash([4, 4]); g.beginPath(); g.moveTo(x0, Y(hl.y)); g.lineTo(x0 + w, Y(hl.y));
    g.stroke(); g.setLineDash([]); g.fillStyle = hl.color; g.font = "11px system-ui"; g.textAlign = "right";
    g.textAlign = "left"; g.fillText(hl.label, x0 + 4, Y(hl.y) + (hl.below ? 12 : -3));
  }
  for (const s of series) {
    const n = s.n ?? s.x.length;
    if (n < 1) continue;
    g.strokeStyle = s.color; g.lineWidth = s.width || 1.5; g.setLineDash(s.dash || []);
    g.beginPath();
    for (let i = 0; i < n; i++) { const px = X(s.x[i]), py = Y(s.y[i]); if (i) g.lineTo(px, py); else g.moveTo(px, py); }
    g.stroke();
  }
  g.setLineDash([]);
  if (opts.cursor != null) {
    g.strokeStyle = "rgba(230,233,239,.55)"; g.beginPath(); g.moveTo(X(opts.cursor), y0); g.lineTo(X(opts.cursor), y0 + h); g.stroke();
  }
  g.fillStyle = C.muted; g.font = "11px system-ui";
  if (opts.ylabel) g.fillText(opts.ylabel, x0 + 4, y0 + 12);
  if (opts.xlabel) { g.textAlign = "right"; g.fillText(opts.xlabel, x0 + w, y0 + h + 13); g.textAlign = "left"; }
  if (opts.ticks) {
    g.textAlign = "right";
    for (const t of opts.ticks) g.fillText(t.label, x0 - 3, Y(t.y) + 4);
    g.textAlign = "left";
  }
  let lx = x0 + w;
  g.textAlign = "right";
  for (const s of [...series].reverse()) {
    if (!s.label) continue;
    g.fillStyle = s.color; g.fillText(s.label, lx - 4, y0 - 4);
    lx -= g.measureText(s.label).width + 14;
  }
  g.textAlign = "left";
}

// ------------------------------------------------------------------ runs: what the viewer plays
// A run holds, per tubule, the frames (t, flat integer-um positions, clump ids, wall flags, colour indices), the
// events (rows of event_columns), the momentum rows (momentum_columns, one per frame) and p_scale = sum M|V| at
// the start [1e-12 kg m/s].
function newTubule(name, material) {
  return { name, material, radii: [], t: [], pos: [], clump: [], wall: [], colour: [], events: [], mom: [],
    p_scale: null, v_stick: null, done: false, summary: null };
}
function eventIndex(cols) {
  const k = {}; cols.forEach((c, i) => (k[c] = i)); return k;
}
function appendFrames(tub, fr) {
  for (let i = 0; i < fr.t.length; i++) {
    tub.t.push(fr.t[i]); tub.pos.push(fr.pos[i]); tub.clump.push(fr.clump[i]); tub.wall.push(fr.wall[i]);
    tub.colour.push(fr.colour[i]);
  }
}
function runFromReplay(rep) {
  const run = { kind: "replay", size: rep.tubules[0].size_um, palette: rep.palette.slice(), done: true,
    evcols: eventIndex(rep.tubules[0].event_columns), tEnd: 0, tubules: [], meta: rep };
  for (const t of rep.tubules) {
    const tub = newTubule(t.name, t.material);
    tub.radii = t.radii_um; appendFrames(tub, t.frames); tub.events = t.events; tub.mom = t.momentum || [];
    tub.p_scale = t.p_scale; tub.v_stick = t.sticking_speeds_mm_s; tub.done = true; tub.summary = t.summary;
    run.tubules.push(tub);
    run.tEnd = Math.max(run.tEnd, tub.t[tub.t.length - 1]);
  }
  return run;
}
function runFromSandbox(first, params, palette) {
  const run = { kind: "sandbox", size: first.size_um, palette, done: false, evcols: eventIndex(first.event_columns),
    tEnd: params.t_end, tubules: [], params };
  for (const t of first.tubules) run.tubules.push(newTubule(t.material, t.material));
  return run;
}
function appendSandbox(run, data) {
  data.tubules.forEach((t, k) => {
    const tub = run.tubules[k];
    if (t.radii_um) tub.radii = t.radii_um;
    appendFrames(tub, t.frames);
    for (const e of t.events) tub.events.push(e);
    for (const m of t.momentum) tub.mom.push(m);
    tub.p_scale = t.p_scale; tub.done = t.done; tub.tNow = t.t;
  });
  run.done = data.tubules.every((t) => t.done);
}
// Simulated time up to which every tubule has frames (or has finished).
function available(run) {
  if (run.done) return Math.max(...run.tubules.map((t) => t.t[t.t.length - 1] ?? 0));
  return Math.min(...run.tubules.map((t) => (t.done ? Infinity : (t.t[t.t.length - 1] ?? 0))));
}

// What happened in tubule `tub` up to time t: the counters of the cards.
function countsAt(run, tub, t, fi) {
  const k = run.evcols, n = bisectRight(tub.events.map((e) => e[k.t_s]), t);
  const c = { impacts: 0, sticks: 0, bounces: 0, breakups: 0, wallStick: 0, wallBounce: 0, last: null };
  for (let i = 0; i < n; i++) {
    const e = tub.events[i];
    if (e[k.kind] === 0) { c.impacts++; if (e[k.outcome] === 1) c.sticks++; else c.bounces++; }
    else if (e[k.outcome] === 1) c.wallStick++; else c.wallBounce++;
    if (e[k.mode]) c.breakups++;
  }
  c.last = n ? tub.events[n - 1] : null;
  const ids = tub.clump[fi] || [], members = new Map();
  for (const id of ids) members.set(id, (members.get(id) || 0) + 1);
  let clumps = 0, inClumps = 0, largest = ids.length ? 1 : 0;
  for (const m of members.values()) if (m > 1) { clumps++; inClumps += m; largest = Math.max(largest, m); }
  const wall = tub.wall[fi] || "";
  let flying = 0; for (let i = 0; i < wall.length; i++) if (wall[i] === "0") flying++;
  return Object.assign(c, { clumps, inClumps, largest, flying });
}

// ------------------------------------------------------------------ the 3-D viewer of the tubules
class TubeViewer {
  constructor(canvas, labelsEl) {
    this.canvas = canvas; this.labelsEl = labelsEl;
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(C.bg);
    this.camera = new THREE.PerspectiveCamera(24, 1, 0.05, 4000);
    this.controls = new OrbitControls(this.camera, canvas);
    this.controls.enableDamping = true;
    this.scene.add(new THREE.HemisphereLight(0xdde6ff, 0x30343c, 1.6));
    const sun = new THREE.DirectionalLight(0xffffff, 1.6); sun.position.set(30, 60, 50); this.scene.add(sun);
    this.root = new THREE.Group(); this.scene.add(this.root);
    this.sphere = new THREE.SphereGeometry(1, 18, 12);
    this.flashTex = this.makeRingTexture();
    this.flashes = [];
    for (let i = 0; i < 90; i++) {
      const s = new THREE.Sprite(new THREE.SpriteMaterial({ map: this.flashTex, transparent: true, depthTest: false,
        depthWrite: false }));
      s.visible = false; s.renderOrder = 10; this.scene.add(s); this.flashes.push(s);
    }
    this.mag = 5; this.showWall = true; this.run = null; this.groups = [];
    this.tmpM = new THREE.Matrix4(); this.tmpC = new THREE.Color(); this.bgC = new THREE.Color(C.bg);
  }
  makeRingTexture() {
    const cv = document.createElement("canvas"); cv.width = cv.height = 64;
    const g = cv.getContext("2d"); g.strokeStyle = "#fff"; g.lineWidth = 6;
    g.beginPath(); g.arc(32, 32, 26, 0, 2 * Math.PI); g.stroke();
    const tex = new THREE.CanvasTexture(cv); tex.colorSpace = THREE.SRGBColorSpace; return tex;
  }
  // World coordinates [mm]: the tubule's long axis z points up the screen; (x, y, z) -> (x, z, -y) keeps it
  // right-handed. Tubule k is shifted sideways by k (width + gap); the pair is centred on the origin.
  offsetX(k) { const n = this.run.tubules.length, sx = this.run.size[0] / 1000; return (k - (n - 1) / 2) * (sx + GAP_MM); }
  toWorld(k, xu, yu, zu, out) {
    const s = this.run.size;
    out.set((xu - s[0] / 2) / 1000 + this.offsetX(k), (zu - s[2] / 2) / 1000, -(yu - s[1] / 2) / 1000);
    return out;
  }
  setRun(run) {
    this.run = run;
    for (const g of this.groups) { this.root.remove(g.group); g.mesh.dispose(); }
    this.groups = [];
    this.labelsEl.innerHTML = "";
    const [sx, sy, sz] = run.size.map((v) => v / 1000);
    run.tubules.forEach((tub, k) => {
      const group = new THREE.Group();
      const box = new THREE.BoxGeometry(sx, sz, sy);
      const glass = new THREE.Mesh(box, new THREE.MeshBasicMaterial({ color: 0x6aa5ff, transparent: true, opacity: 0.05,
        depthWrite: false, side: THREE.DoubleSide }));
      const edges = new THREE.LineSegments(new THREE.EdgesGeometry(box),
        new THREE.LineBasicMaterial({ color: 0x6aa5ff, transparent: true, opacity: 0.55 }));
      glass.position.x = edges.position.x = this.offsetX(k);
      group.add(glass, edges);
      const N = Math.max(tub.radii.length, 1);
      const mesh = new THREE.InstancedMesh(this.sphere, new THREE.MeshStandardMaterial({ roughness: 0.55, metalness: 0.05 }), N);
      mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
      mesh.setColorAt(0, new THREE.Color(1, 1, 1));
      mesh.count = tub.radii.length;
      group.add(mesh);
      this.root.add(group);
      const lbl = document.createElement("div"); lbl.className = "lbl";
      lbl.textContent = tub.material[0].toUpperCase() + tub.material.slice(1);
      lbl.style.color = tub.material === "graphite" ? "#c8c8c8" : "#a9cfe8";
      this.labelsEl.appendChild(lbl);
      this.groups.push({ group, mesh, lbl, lastFi: -1 });
    });
    this.viewAll();
  }
  ensureCapacity(k) {                     // a sandbox run learns its radii with the first chunk
    const g = this.groups[k], tub = this.run.tubules[k];
    if (g.mesh.count === tub.radii.length) return;
    if (tub.radii.length > g.mesh.instanceMatrix.count) {
      g.group.remove(g.mesh); g.mesh.dispose();
      g.mesh = new THREE.InstancedMesh(this.sphere, new THREE.MeshStandardMaterial({ roughness: 0.55, metalness: 0.05 }), tub.radii.length);
      g.mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
      g.mesh.setColorAt(0, new THREE.Color(1, 1, 1));
      g.group.add(g.mesh);
    }
    g.mesh.count = tub.radii.length;
  }
  frameAt(tub, t) {                       // index i with t[i] <= t (clamped) and the interpolation weight
    const n = tub.t.length;
    if (!n) return { i: -1, a: 0 };
    let i = bisectRight(tub.t, t) - 1;
    if (i < 0) return { i: 0, a: 0 };
    if (i >= n - 1) return { i: n - 1, a: 0 };
    return { i, a: (t - tub.t[i]) / (tub.t[i + 1] - tub.t[i]) };
  }
  colourOf(idx) {
    if (idx === 1) return GRAPHITE_LIGHT;
    return this.run.palette[idx] || "#ccc";
  }
  update(t, wallAge) {
    if (!this.run) return;
    const v = new THREE.Vector3();
    this.run.tubules.forEach((tub, k) => {
      this.ensureCapacity(k);
      const g = this.groups[k], { i, a } = this.frameAt(tub, t);
      if (i < 0) { g.mesh.count = 0; return; }
      const P0 = tub.pos[i], P1 = a > 0 ? tub.pos[i + 1] : P0, wall = tub.wall[i], col = tub.colour[i];
      for (let j = 0; j < tub.radii.length; j++) {
        const x = P0[3 * j] + a * (P1[3 * j] - P0[3 * j]);
        const y = P0[3 * j + 1] + a * (P1[3 * j + 1] - P0[3 * j + 1]);
        const z = P0[3 * j + 2] + a * (P1[3 * j + 2] - P0[3 * j + 2]);
        this.toWorld(k, x, y, z, v);
        const r = (tub.radii[j] / 1000) * this.mag;
        this.tmpM.makeScale(r, r, r); this.tmpM.setPosition(v);
        g.mesh.setMatrixAt(j, this.tmpM);
        if (g.lastFi !== i) {
          this.tmpC.set(this.colourOf(col[j]));
          if (wall[j] === "1") this.tmpC.lerp(this.bgC, 0.55);
          g.mesh.setColorAt(j, this.tmpC);
        }
      }
      g.mesh.instanceMatrix.needsUpdate = true;
      if (g.lastFi !== i && g.mesh.instanceColor) { g.mesh.instanceColor.needsUpdate = true; g.lastFi = i; }
    });
    this.updateFlashes(t, wallAge);
  }
  updateFlashes(t, wallAge) {
    let used = 0;
    const k = this.run.evcols, v = new THREE.Vector3();
    this.run.tubules.forEach((tub, ti) => {
      const ts = tub.events;
      let n = 0; { let lo = 0, hi = ts.length; while (lo < hi) { const m = (lo + hi) >> 1; if (ts[m][k.t_s] <= t) lo = m + 1; else hi = m; } n = lo; }
      for (let i = n - 1; i >= 0 && used < this.flashes.length; i--) {
        const e = ts[i], age = wallAge(e[k.t_s]);
        if (age > FLASH_S) break;
        const isWall = e[k.kind] === 1;
        if (isWall && !this.showWall) continue;
        const s = this.flashes[used++];
        const f = age / FLASH_S;
        this.toWorld(ti, e[k.x_um], e[k.y_um], e[k.z_um], v);
        s.position.copy(v);
        const base = isWall ? 0.5 : 1.0 + 0.06 * this.mag;
        s.scale.setScalar(base * (0.6 + 1.4 * f));
        s.material.color.set(e[k.mode] ? C.breakup : (e[k.outcome] === 1 ? C.stick : C.bounce));
        s.material.opacity = (isWall ? 0.55 : 1) * (1 - f);
        s.visible = true;
      }
    });
    for (let i = used; i < this.flashes.length; i++) this.flashes[i].visible = false;
  }
  frame(box) {                             // put the camera so that box {w, h} [mm] around the origin fits
    const fov = (this.camera.fov * Math.PI) / 180, asp = this.camera.aspect || 1;
    const d = 1.12 * Math.max(box.h / 2 / Math.tan(fov / 2), box.w / 2 / (Math.tan(fov / 2) * asp)) + box.depth / 2;
    const dir = new THREE.Vector3(0.18, 0.22, 1).normalize();
    this.controls.target.set(0, 0, 0);
    this.camera.position.copy(dir.multiplyScalar(d));
    this.controls.update();
  }
  viewAll() {
    if (!this.run) return;
    const n = this.run.tubules.length, [sx, sy, sz] = this.run.size.map((v) => v / 1000);
    this.frame({ w: n * sx + (n - 1) * GAP_MM, h: sz, depth: sy });
  }
  viewCloud() {
    if (!this.run) return;
    const n = this.run.tubules.length, sx = this.run.size[0] / 1000;
    this.frame({ w: (n - 1) * (sx + GAP_MM) + 7, h: 7, depth: 2 });
  }
  resize() {
    const w = this.canvas.clientWidth, h = this.canvas.clientHeight;
    if (!w || !h) return;
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h; this.camera.updateProjectionMatrix();
  }
  render() {
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
    if (!this.run) return;
    const w = this.canvas.clientWidth, h = this.canvas.clientHeight, v = new THREE.Vector3();
    const sz = this.run.size[2] / 1000;
    this.groups.forEach((g, k) => {
      v.set(this.offsetX(k), sz / 2 + 0.6, 0).project(this.camera);
      g.lbl.style.left = `${((v.x + 1) / 2) * w}px`; g.lbl.style.top = `${((1 - v.y) / 2) * h}px`;
      g.lbl.style.display = v.z < 1 ? "" : "none";
    });
  }
}

// ------------------------------------------------------------------ the cards: counters and momentum plot
function buildCards(run) {
  const host = $("cards"); host.innerHTML = "";
  return run.tubules.map((tub) => {
    const el = document.createElement("div"); el.className = "card";
    const vs = tub.v_stick ? `sticking speed ${fmt(tub.v_stick.grain_grain, 1)} mm/s grain-grain, ${fmt(tub.v_stick.grain_wall, 1)} mm/s on the wall (50 µm radius)`
      : (run.params ? `${run.params.n} grains, diameters ${run.params.d_min_um}–${run.params.d_max_um} µm` : "");
    el.innerHTML = `<h3><span>${tub.material[0].toUpperCase() + tub.material.slice(1)}</span><span class="muted">${vs}</span></h3>
      <div class="counts">
        <div><b data-k="impacts">0</b><span>grain-grain impacts</span></div>
        <div><b data-k="sticks" style="color:${C.stick}">0</b><span>sticks</span></div>
        <div><b data-k="bounces" style="color:${C.bounce}">0</b><span>bounces</span></div>
        <div><b data-k="breakups" style="color:${C.breakup}">0</b><span>clump break-ups</span></div>
        <div><b data-k="clumps">0</b><span>clumps (2+ grains)</span></div>
        <div><b data-k="inClumps">0</b><span>grains in clumps</span></div>
        <div><b data-k="largest">1</b><span>largest clump, grains</span></div>
        <div><b data-k="walls">0</b><span>grains on walls / flying</span></div>
      </div>
      <canvas class="plot" aria-label="Momentum of this tubule against time"></canvas>
      <div class="lastev" data-k="mom"></div>
      <div class="lastev" data-k="last"></div>`;
    host.appendChild(el);
    const q = (k) => el.querySelector(`[data-k="${k}"]`);
    return { el, tub, plot: el.querySelector("canvas"), q };
  });
}
function describeEvent(run, e) {
  const k = run.evcols;
  const who = (n) => (n > 1 ? `a ${n}-grain clump` : "a grain");
  const out = e[k.outcome] === 1 ? `<span style="color:${C.stick}">sticks</span>` : `<span style="color:${C.bounce}">bounces</span>`;
  const extra = e[k.mode] ? `, <span style="color:${C.breakup}">${e[k.mode]}: ${e[k.fragments]} fragment(s)</span>` : "";
  const ee = e[k.outcome] === 1 ? "" : ` (e = ${fmt(e[k.e_n], 2)})`;
  if (e[k.kind] === 1) return `t = ${fmt(e[k.t_s], 4)} s: ${who(e[k.n_i])} hits a wall at ${fmt(e[k.v_n_mm_s], 1)} mm/s and ${out}${ee}${extra}`;
  return `t = ${fmt(e[k.t_s], 4)} s: ${who(e[k.n_i])} meets ${who(e[k.n_j])} at ${fmt(e[k.v_n_mm_s], 1)} mm/s (normal) and ${out}${ee}${extra}`;
}
function updateCards(run, cards, t, viewer) {
  cards.forEach((cd) => {
    const tub = cd.tub, { i } = viewer.frameAt(tub, t);
    if (i < 0) return;
    const c = countsAt(run, tub, t, i);
    cd.q("impacts").textContent = c.impacts; cd.q("sticks").textContent = c.sticks; cd.q("bounces").textContent = c.bounces;
    cd.q("breakups").textContent = c.breakups;
    cd.q("clumps").textContent = c.clumps; cd.q("largest").textContent = c.largest; cd.q("inClumps").textContent = c.inClumps;
    cd.q("walls").textContent = `${tub.radii.length - c.flying} / ${c.flying}`;
    let last = `Wall impacts: ${c.wallStick} captured, ${c.wallBounce} bounced off.<br>`
      + (c.last ? "Latest: " + describeEvent(run, c.last) : "No impact yet.");
    if (tub.done && i === tub.t.length - 1 && c.flying === 0) last += `<br>All grains are on the walls since t = ${fmt(tub.t[i], 2)} s.`;
    cd.q("last").innerHTML = last;
    drawMomentum(cd, run, i, t);
  });
}
// Momentum ledger (MOMENTUM_COLUMNS of dolle_core/scenes.py): |grains flying|, |walls received|,
// |residual acceleration gave| and |total = flying + walls - acceleration|, all divided by S0 = sum M|V| at t = 0.
function drawMomentum(cd, run, fi, t) {
  const tub = cd.tub, rows = tub.mom, S = tub.p_scale;
  const { g, w, h } = setupCanvas2d(cd.plot);
  if (!rows.length || !S) { g.fillStyle = C.muted; g.fillText("no momentum data", 10, 20); return; }
  if (!cd.series || cd.series.n !== rows.length) {
    const x = [], fly = [], wal = [], acc = [], tot = [];
    let ymax = 0;
    for (const r of rows) {
      x.push(r[0]);
      fly.push(norm3(r[1], r[2], r[3]) / S); wal.push(norm3(r[4], r[5], r[6]) / S); acc.push(norm3(r[7], r[8], r[9]) / S);
      tot.push(norm3(r[1] + r[4] - r[7], r[2] + r[5] - r[8], r[3] + r[6] - r[9]) / S);
      ymax = Math.max(ymax, fly.at(-1), wal.at(-1), acc.at(-1), tot.at(-1));
    }
    cd.series = { n: rows.length, x, fly, wal, acc, tot, ymax };
  }
  const s = cd.series, n = Math.min(fi + 1, s.n);
  const xmax = Math.max(run.tEnd, s.x.at(-1), 1e-3), ymax = Math.max(s.ymax * 1.15, 0.05);
  linePlot(g, { x0: 34, y0: 18, w: w - 42, h: h - 36 }, [
    { x: s.x, y: s.fly, n, color: C.accent, width: 1.3, label: "grains flying" },
    { x: s.x, y: s.wal, n, color: C.warn, width: 1.3, label: "walls received" },
    { x: s.x, y: s.acc, n, color: C.muted, width: 1.3, dash: [3, 3], label: "acceleration gave" },
    { x: s.x, y: s.tot, n, color: C.stick, width: 2.4, label: "total" },
  ], xmax, 0, ymax, { cursor: Math.min(t, xmax), xlabel: `t [s], 0 to ${fmt(xmax, 1)}`,
    ticks: [{ y: 0, label: "0" }, { y: ymax / 1.15, label: fmt(ymax / 1.15, 2) }] });
  let dev = 0; for (let i = 0; i < n; i++) dev = Math.max(dev, rows[i][10]);
  cd.q("mom").innerHTML = `Momentum, in units of the grains' Σ M|V| at the start: the <b style="color:${C.stick}">total</b> `
    + `stays at ${fmt(s.tot[n - 1], 5)}; its largest change so far is ${sci(dev)} (rounding error).`;
}

// ------------------------------------------------------------------ playback (shared by replay and live run)
const state = { runs: { replay: null, sandbox: null }, which: "replay", playing: !reduceMotion, w: 0,
  cards: [], speed: 1, slowmo: true, last: null, viewer: null };

function wallOf(t) { return state.slowmo ? (t < SLOW_UNTIL ? t * SLOW_FACTOR : SLOW_UNTIL * SLOW_FACTOR + (t - SLOW_UNTIL)) : t; }
function simOf(w) { return state.slowmo ? (w < SLOW_UNTIL * SLOW_FACTOR ? w / SLOW_FACTOR : SLOW_UNTIL + (w - SLOW_UNTIL * SLOW_FACTOR)) : w; }
function curRun() { return state.runs[state.which]; }
function curT() { return simOf(state.w); }

function selectRun(which) {
  const r = state.runs[which]; if (!r) return;
  if (state.runs[state.which]) state.runs[state.which].savedW = state.w;
  state.which = which;
  state.w = r.savedW ?? 0;
  state.viewer.setRun(r);
  state.cards = buildCards(r);
  for (const [id, k] of [["src-replay", "replay"], ["src-sandbox", "sandbox"]]) {
    $(id).classList.toggle("on", k === which); $(id).setAttribute("aria-selected", String(k === which));
  }
  $("src-note").textContent = which === "replay"
    ? "computed by python tools/make_sim3d_replays.py"
    : `computed in your browser: ${r.tubules.length} tubule(s), ${r.params.n} grains each`;
  refresh();
}
function refresh() {
  const run = curRun(); if (!run) return;
  const t = curT();
  const slowNow = state.slowmo && t < SLOW_UNTIL;
  $("tread").innerHTML = `t = ${fmt(t, 3)} s of ${fmt(run.tEnd, 1)} s${slowNow ? " · slow motion ×10" : ""}`
    + (run.kind === "sandbox" && !run.done ? ` · computed to ${fmt(available(run), 2)} s` : "");
  $("scrub").value = String(Math.round((1000 * state.w) / Math.max(wallOf(run.tEnd), 1e-9)));
  const age = (te) => (state.w - wallOf(te)) / state.speed;
  state.viewer.update(t, age);
  updateCards(run, state.cards, t, state.viewer);
  $("play").textContent = state.playing ? "Pause" : "Play";
}
function tick(dt) {
  const run = curRun(); if (!run) return;
  if (state.playing) {
    const wEnd = wallOf(run.tEnd), wAvail = wallOf(available(run));
    state.w = Math.min(state.w + dt * state.speed, wAvail);
    if (state.w >= wEnd - 1e-9 || (run.done && state.w >= wAvail - 1e-9)) { state.w = Math.min(wEnd, wAvail); state.playing = false; }
  }
  refresh();
}

// ------------------------------------------------------------------ the impact up close
class Closeup {
  constructor(canvas, plot) {
    this.canvas = canvas; this.plot = plot;
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    this.scene = new THREE.Scene(); this.scene.background = new THREE.Color(C.bg);
    this.camera = new THREE.PerspectiveCamera(30, 1, 0.1, 100);
    this.camera.position.set(3.4, 1.9, 7.4); this.camera.lookAt(0, 0, 0);
    this.scene.add(new THREE.HemisphereLight(0xdde6ff, 0x30343c, 1.5));
    const sun = new THREE.DirectionalLight(0xffffff, 1.8); sun.position.set(3, 5, 6); this.scene.add(sun);
    const geo = new THREE.SphereGeometry(1, 48, 32);
    this.s1 = new THREE.Mesh(geo, new THREE.MeshStandardMaterial({ color: 0xa9cfe8, roughness: 0.5, transparent: true, opacity: 0.88 }));
    this.s2 = new THREE.Mesh(geo, this.s1.material.clone());
    this.scene.add(this.s1, this.s2);
    this.patch = new THREE.Mesh(new THREE.CylinderGeometry(1, 1, 1, 48, 1, true),
      new THREE.MeshBasicMaterial({ color: 0xf5b54a, transparent: true, opacity: 0.9, side: THREE.DoubleSide, depthTest: false }));
    this.patch.rotation.z = Math.PI / 2; this.patch.renderOrder = 5; this.scene.add(this.patch);
    this.a1 = new THREE.ArrowHelper(new THREE.Vector3(-1, 0, 0), new THREE.Vector3(), 1, 0xff8a3d, 0.25, 0.16);
    this.a2 = new THREE.ArrowHelper(new THREE.Vector3(1, 0, 0), new THREE.Vector3(), 1, 0xff8a3d, 0.25, 0.16);
    for (const a of [this.a1, this.a2]) { a.renderOrder = 6; a.line.material.depthTest = false; a.cone.material.depthTest = false; this.scene.add(a); }
    // Timeline on screen [s]: approach, contact (slow motion), after, hold.
    this.T = { A: 1.8, B: 6.0, C: 2.2, D: 1.0 };
    this.w = 0; this.playing = !reduceMotion; this.case = null;
  }
  setCase(cs) {
    const k = {}; cs.columns.forEach((c, i) => (k[c] = i));
    this.case = cs; this.k = k;
    const R = cs.radius_um;
    this.X = niceFloor((0.15 * R * 1000) / cs.max_overlap_nm);                     // overlap exaggeration
    const aMax = Math.max(...cs.rows.map((r) => r[k.a_um]));
    this.Y = niceFloor((0.35 * R) / aMax);                                         // patch exaggeration
    this.Fmax = Math.max(...cs.rows.map((r) => Math.abs(r[k.F_el_uN] + r[k.F_damp_uN])));
    this.delta0 = ((cs.pair.a0_um ** 2) / (3 * cs.pair.R_star_um)) * 1000;           // nm, JKR equilibrium overlap
    this.tc = cs.rows.at(-1)[k.t_us];
    const col = cs.material === "graphite" ? 0x9a9a9a : 0xa9cfe8;
    this.s1.material.color.set(col); this.s2.material.color.set(col);
    this.w = 0;
    $("c-note").innerHTML = `Drawn to scale except: the overlap is drawn ${this.X.toLocaleString("en")}× larger than true (at most `
      + `${fmt(cs.max_overlap_nm, 1)} nm, ${Math.round((R * 1000) / cs.max_overlap_nm).toLocaleString("en")} times smaller than the radius), `
      + `and the contact patch ${this.Y}× larger (its radius is at most ${fmt(aMax, 2)} µm). The approach starts `
      + `${fmt(cs.approach.gap_start_um, 0)} µm apart, to scale. Before touch there is no force: in this model the sticky contact snaps on `
      + `at touch. The arrows above the grains are the contact force on each grain (equal and opposite: Newton's third law; length proportional to the force); orange pushes apart, blue pulls together. `
      + (cs.outcome === "stick"
        ? `A stick is decided as soon as the code proves the pair can never separate (dolle_core/collide.py, tests 3 and 4), while the grains still vibrate; damping then stops them. That last part is not computed: the picture eases them to the resting overlap δ₀ = a₀²/(3R*) = ${fmt(this.delta0, 2)} nm (JKR).`
        : `The bounce happens when the stretched neck reaches the break overlap ${fmt(cs.pair.delta_break_nm, 2)} nm (negative: the grains are already slightly apart). They leave at e = ${fmt(cs.e, 3)} of the approach speed.`)
      + ` Source: dolle_core.scenes.impact_closeup (the trace of dolle_core.collide.impact).`;
    this.drawPlotCache = null;
  }
  total() { return this.T.A + this.T.B + this.T.C + this.T.D; }
  // State at screen time w: phase, physical time, centre distance [R], patch radius [R], force [uN], row.
  stateAt(w) {
    const cs = this.case, k = this.k, R = cs.radius_um, T = this.T, Ta = cs.approach.duration_s;
    const rows = cs.rows;
    if (w < T.A) {
      const f = w / T.A, tPhys = -Ta * (1 - f);
      return { phase: "approach", tPhys, D: 2 + (cs.approach.gap_start_um * (1 - f)) / R, a: 0, F: null, row: null,
        slow: Ta / T.A, v1: cs.u1_mm_s, v2: cs.u2_mm_s };
    }
    if (w < T.A + T.B) {
      const tus = ((w - T.A) / T.B) * this.tc;
      let i = 0; { let lo = 0, hi = rows.length - 1; while (lo < hi) { const m = (lo + hi + 1) >> 1; if (rows[m][k.t_us] <= tus) lo = m; else hi = m - 1; } i = lo; }
      const r0 = rows[i], r1 = rows[Math.min(i + 1, rows.length - 1)];
      const span = r1[k.t_us] - r0[k.t_us], a = span > 0 ? (tus - r0[k.t_us]) / span : 0;
      const lerp = (c) => r0[c] + a * (r1[c] - r0[c]);
      const delta = lerp(k.overlap_nm);
      return { phase: "contact", tPhys: tus * 1e-6, D: 2 - (delta * 1e-3 * this.X) / R, a: (lerp(k.a_um) * this.Y) / R,
        F: lerp(k.F_el_uN) + lerp(k.F_damp_uN), Fel: lerp(k.F_el_uN), Fd: lerp(k.F_damp_uN), delta, aum: lerp(k.a_um),
        row: i, slow: (this.tc * 1e-6) / T.B, v1: lerp(k.v1_mm_s), v2: lerp(k.v2_mm_s), tus };
    }
    const f = Math.min((w - T.A - T.B) / T.C, 1);
    const last = rows.at(-1);
    if (cs.outcome === "bounce") {
      const tPhys = f * Ta;
      const gap = (Math.abs(last[k.overlap_nm]) * 1e-3 * this.X) / R + ((cs.v2_mm_s - cs.v1_mm_s) * 1e-3 * tPhys * 1e6) / R;
      return { phase: "after", tPhys: this.tc * 1e-6 + tPhys, D: 2 + gap, a: 0, F: 0, slow: Ta / T.C, v1: cs.v1_mm_s, v2: cs.v2_mm_s, done: w >= T.A + T.B };
    }
    const e = 1 - (1 - f) ** 3, delta = last[k.overlap_nm] + e * (this.delta0 - last[k.overlap_nm]);
    const a0 = cs.pair.a0_um, aum = last[k.a_um] + e * (a0 - last[k.a_um]);
    return { phase: "after", tPhys: null, D: 2 - (delta * 1e-3 * this.X) / R, a: (aum * this.Y) / R, F: 0, delta, aum,
      v1: 0, v2: 0, done: true };
  }
  update(dt) {
    if (!this.case) return;
    if (this.playing) { this.w += dt; if (this.w > this.total()) this.w = 0; }
    const s = this.stateAt(this.w), cs = this.case;
    this.s1.position.set(-s.D / 2, 0, 0); this.s2.position.set(s.D / 2, 0, 0);
    // patch (contact) or neck (stretched, overlap < 0): a short cylinder between the spheres
    const show = s.a > 0;
    this.patch.visible = show;
    if (show) {
      const len = Math.max(2 - s.D, 0) > 0 ? 0.012 : Math.max(s.D - 2, 0.012);
      this.patch.scale.set(s.a, len, s.a);
    }
    const Fn = s.F == null ? 0 : s.F;
    const on = s.phase === "contact" && Math.abs(Fn) > 1e-9;
    for (const [arr, sign, sph] of [[this.a1, -1, this.s1], [this.a2, 1, this.s2]]) {
      arr.visible = on;
      if (!on) continue;
      const dir = new THREE.Vector3(sign * Math.sign(Fn), 0, 0);
      arr.position.set(sph.position.x, 1.35, 0); arr.setDirection(dir);          // drawn just above each grain
      arr.setLength(0.25 + 1.35 * (Math.abs(Fn) / this.Fmax), 0.25, 0.16);
      arr.setColor(Fn > 0 ? 0xff8a3d : 0x5fd4ff);
    }
    // labels and readout
    const ph = { approach: "Approach (no force before touch)", contact: "Contact: squash and rebound", after: cs.outcome === "stick" ? "Stuck together" : "Separating" }[s.phase];
    const slowTxt = s.slow ? ` · slowed down ${Math.round(1 / s.slow).toLocaleString("en")}×` : "";
    $("c-phase").textContent = ph + slowTxt;
    const out = $("c-outcome");
    if (s.phase === "after") {
      out.className = `outcome ${cs.outcome}`;
      out.textContent = cs.outcome === "stick" ? "STICKS" : `BOUNCES · e = ${fmt(cs.e, 2)}`;
    } else { out.className = "outcome"; out.textContent = ""; }
    let tt;
    if (s.phase === "approach") tt = `t = ${fmt(s.tPhys * 1e3, 2)} ms (before touch)`;
    else if (s.phase === "contact") tt = `t = ${fmt(s.tus, 3)} µs after touch`;
    else tt = s.tPhys == null ? `after ${fmt(this.tc, 2)} µs: bound` : `t = ${fmt(s.tPhys * 1e3, 3)} ms after touch`;
    $("c-tread").textContent = tt;
    $("c-scrub").value = String(Math.round((1000 * this.w) / this.total()));
    $("c-play").textContent = this.playing ? "Pause" : "Play";
    const rk = this.k, row = s.row != null ? cs.rows[s.row] : null;
    let txt = `<b>${cs.material}</b>, two grains of ${cs.radius_um} µm radius, approach speed <b>${fmt(cs.v_in_mm_s, 2)} mm/s</b> `
      + `(${cs.factor_of_v_stick}× the sticking speed ${fmt(cs.v_stick_mm_s, 2)} mm/s)\n`
      + `velocities (centre-of-mass frame): ${fmt(s.v1, 2)} and ${fmt(s.v2, 2)} mm/s\n`;
    if (s.phase === "contact" && row) {
      txt += `overlap δ = <b>${fmt(s.delta, 2)} nm</b> · contact patch radius a = ${fmt(s.aum, 3)} µm\n`
        + `contact force F = <b>${fmt(s.F, 2)} µN</b> (JKR ${fmt(s.Fel, 2)}, damping ${fmt(s.Fd, 2)}); pull-off force F_c = ${fmt(cs.pair.F_c_uN, 2)} µN\n`
        + `energy [fJ]: relative motion ${fmt(row[rk.kinetic_fJ], 2)}, stored in the contact (JKR potential, negative when the adhesion dominates) ${fmt(row[rk.stored_fJ], 2)}, lost to damping ${fmt(row[rk.damping_fJ], 2)}, `
        + `to adhesion hysteresis ${fmt(row[rk.hysteresis_fJ], 2)}, to yield ${fmt(row[rk.plastic_fJ], 2)} (in: ${fmt(cs.E_in_fJ, 2)})`;
    } else if (s.phase === "approach") {
      txt += `incoming energy of the relative motion ½ m* v² = ${fmt(cs.E_in_fJ, 2)} fJ (1 fJ = 10⁻¹⁵ J); adhesion energy at touch |U₀| = ${fmt(-cs.pair.U0_fJ, 2)} fJ`;
    } else {
      txt += cs.outcome === "stick" ? `contact lasted ${fmt(this.tc, 3)} µs until the pair was proved bound; largest force ${fmt(cs.max_force_uN, 1)} µN`
        : `contact lasted ${fmt(this.tc, 3)} µs; largest force ${fmt(cs.max_force_uN, 1)} µN; restitution e = ${fmt(cs.e, 3)}`;
    }
    $("c-read").innerHTML = txt;
    this.drawPlot(s);
  }
  drawPlot(s) {
    const cs = this.case, k = this.k, rows = cs.rows;
    const { g, w, h } = setupCanvas2d(this.plot);
    const x = rows.map((r) => r[k.t_us]);
    const F = rows.map((r) => r[k.F_el_uN] + r[k.F_damp_uN]), Fel = rows.map((r) => r[k.F_el_uN]), Fd = rows.map((r) => r[k.F_damp_uN]);
    const d = rows.map((r) => r[k.overlap_nm]);
    const cur = s.phase === "contact" ? s.tus : s.phase === "approach" ? 0 : this.tc;
    const fmin = Math.min(...F, ...Fel, -cs.pair.F_c_uN) * 1.15, fmax = Math.max(...F, ...Fel) * 1.15;
    const h1 = (h - 56) * 0.58, h2 = (h - 56) - h1;
    linePlot(g, { x0: 40, y0: 16, w: w - 48, h: h1 }, [
      { x, y: Fel, color: C.accent, width: 1.2, label: "JKR" },
      { x, y: Fd, color: C.muted, width: 1.2, dash: [3, 3], label: "damping" },
      { x, y: F, color: C.bounce, width: 2.4, label: "force F [µN]" },
    ], this.tc, fmin, fmax, { cursor: cur, hlines: [{ y: -cs.pair.F_c_uN, color: C.attract, label: "−F_c (pull-off force)", below: true }],
      ticks: [{ y: 0, label: "0" }, { y: fmax / 1.15, label: fmt(fmax / 1.15, 0) }, { y: fmin / 1.15, label: fmt(fmin / 1.15, 0) }] });
    const dmin = Math.min(...d, cs.pair.delta_break_nm) * 1.2, dmax = Math.max(...d) * 1.15;
    linePlot(g, { x0: 40, y0: 16 + h1 + 22, w: w - 48, h: h2 }, [
      { x, y: d, color: C.warn, width: 2, label: "overlap δ [nm]" },
    ], this.tc, Math.min(dmin, -0.05 * dmax), dmax, { cursor: cur, xlabel: `time after touch [µs], 0 to ${fmt(this.tc, 2)}`,
      hlines: [{ y: cs.pair.delta_break_nm, color: C.muted, label: "break overlap", below: true }],
      ticks: [{ y: 0, label: "0" }, { y: dmax / 1.15, label: fmt(dmax / 1.15, 1) }] });
  }
  resize() {
    const w = this.canvas.clientWidth, h = this.canvas.clientHeight;
    if (!w || !h) return;
    this.renderer.setSize(w, h, false); this.camera.aspect = w / h;
    const d = w / h < 1.2 ? 1.35 : 1; this.camera.position.set(3.4 * d, 1.9 * d, 7.4 * d); this.camera.lookAt(0, 0, 0);
    this.camera.updateProjectionMatrix();
  }
  render() { this.renderer.render(this.scene, this.camera); }
}

// ------------------------------------------------------------------ the live sandbox (python in a web worker)
const sb = { worker: null, running: false, run: null, params: null, wallSum: 0 };
function sbStatus(text, cls = "") { const el = $("sb-status"); el.className = "status " + cls; el.innerHTML = text; }
function readForm() {
  const f = $("sbform"), v = (n) => f.elements[n].value;
  const [dmin, dmax] = v("dband").split(",").map(Number);
  return { materials: v("materials").split(","), n: Number(v("n")), d_min_um: dmin, d_max_um: dmax,
    speed_mm_s: Number(v("speed_mm_s")), sigma_mm_s: Number(v("sigma_mm_s")), preset: v("preset"),
    box_mm: Number(v("box_mm")), t_end: Number(v("t_end")), seed: Math.max(1, Math.round(Number(v("seed")) || 1)) };
}
function startSandbox() {
  if (sb.running) return;
  if (!sb.worker) {
    sb.worker = new Worker("worker.js");
    sb.worker.onmessage = onWorker;
    sb.worker.onerror = (e) => { sb.running = false; setSbButtons(); sbStatus("The python worker failed: " + (e.message || e), "err"); };
  }
  sb.params = readForm(); sb.run = null; sb.wallSum = 0; sb.running = true; setSbButtons();
  $("sb-bar").style.width = "0";
  sbStatus("Starting python...", "busy");
  sb.worker.postMessage({ cmd: "run", params: sb.params });
}
function setSbButtons() { $("sb-run").disabled = sb.running; $("sb-stop").disabled = !sb.running; }
function onWorker(ev) {
  const m = ev.data;
  if (m.type === "status") sbStatus(m.text, "busy");
  else if (m.type === "ready") sbStatus("Python is ready. Computing...", "busy");
  else if (m.type === "chunk") {
    const palette = state.runs.replay ? state.runs.replay.palette : ["#a9cfe8", "#4b4b4b", "#e6194b", "#3cb44b", "#ffb000", "#4363d8", "#f58231", "#911eb4", "#46c0c0", "#f032e6", "#9a6324", "#008080", "#800000", "#808000"];
    const first = !sb.run;
    if (first) { sb.run = runFromSandbox(m.data, sb.params, palette); state.runs.sandbox = sb.run; }
    appendSandbox(sb.run, m.data);
    sb.wallSum += m.wall_s;
    if (first) {
      sb.run.savedW = 0;
      $("src-sandbox").disabled = false;
      selectRun("sandbox");
      state.playing = true;
      $("viewer").scrollIntoView({ behavior: reduceMotion ? "auto" : "smooth", block: "start" });
    }
    if (state.which === "sandbox") refresh();
    const tAv = available(sb.run);
    $("sb-bar").style.width = `${Math.min(100, (100 * Math.min(tAv, sb.params.t_end)) / sb.params.t_end)}%`;
    sbStatus(`Computing... ${fmt(Math.min(tAv, sb.params.t_end), 2)} s of ${sb.params.t_end} s simulated, in ${fmt(sb.wallSum, 1)} s of computer time `
      + `(python in your browser). The viewer plays what is ready.`, "busy");
  } else if (m.type === "done") {
    sb.running = false; setSbButtons();
    if (sb.run) { sb.run.done = true; sb.run.tEnd = Math.min(sb.run.tEnd, available(sb.run)); if (m.stopped) sb.run.tEnd = available(sb.run); }
    const parts = (sb.run ? sb.run.tubules : []).map((t) => {
      const c = countsAt(sb.run, t, Infinity, t.t.length - 1);
      const pl = (n, one, many) => `${n} ${n === 1 ? one : many}`;
      return `${t.material}: ${pl(c.impacts, "grain-grain impact", "grain-grain impacts")}, ${pl(c.sticks, "stick", "sticks")}, `
        + `${pl(c.bounces, "bounce", "bounces")}, ${pl(c.clumps, "clump", "clumps")}`;
    });
    $("sb-bar").style.width = "100%";
    if (state.which === "sandbox") refresh();
    sbStatus(`${m.stopped ? "Stopped" : "Finished"} after ${fmt(sb.wallSum, 1)} s of computer time. ${parts.join("; ")}.`);
  } else if (m.type === "error") {
    sb.running = false; setSbButtons();
    sbStatus("Error in the live run: " + m.msg, "err");
  }
}

// ------------------------------------------------------------------ start-up
function bindControls(closeup) {
  $("play").onclick = () => {
    const run = curRun(); if (!run) return;
    if (!state.playing && state.w >= wallOf(run.tEnd) - 1e-6) state.w = 0;
    state.playing = !state.playing; refresh();
  };
  $("scrub").oninput = (e) => {
    const run = curRun(); if (!run) return;
    state.w = Math.min((Number(e.target.value) / 1000) * wallOf(run.tEnd), wallOf(available(run)));
    state.playing = false; refresh();
  };
  $("speed").onchange = (e) => { state.speed = Number(e.target.value); };
  $("slowmo").onchange = (e) => { const t = curT(); state.slowmo = e.target.checked; state.w = wallOf(t); refresh(); };
  $("mag").onchange = (e) => {
    state.viewer.mag = Number(e.target.value);
    state.viewer.groups.forEach((g) => (g.lastFi = -1));
    $("magnote").textContent = state.viewer.mag === 1
      ? "Grains are drawn at their true size (about 0.1 mm across in tubules 12.7 mm wide): zoom in to see them."
      : `Grains are about 0.1 mm across in tubules 12.7 mm wide, so they are drawn ${state.viewer.mag}× larger than true size to be visible (choose "true size" and zoom in to see them as they are).`;
    refresh();
  };
  $("walls").onchange = (e) => { state.viewer.showWall = e.target.checked; refresh(); };
  $("view-all").onclick = () => state.viewer.viewAll();
  $("view-cloud").onclick = () => state.viewer.viewCloud();
  $("src-replay").onclick = () => selectRun("replay");
  $("src-sandbox").onclick = () => selectRun("sandbox");
  $("sb-run").onclick = startSandbox;
  $("sb-stop").onclick = () => { if (sb.worker) sb.worker.postMessage({ cmd: "stop" }); };
  const f = $("sbform");
  for (const [name, out] of [["n", "o-n"], ["speed_mm_s", "o-speed"], ["sigma_mm_s", "o-sigma"], ["box_mm", "o-box"]]) {
    f.elements[name].oninput = (e) => ($(out).textContent = e.target.value);
  }
  f.elements.preset.onchange = (e) => { f.elements.box_mm.disabled = e.target.value !== "cloud"; };
  $("c-play").onclick = () => { closeup.playing = !closeup.playing; };
  $("c-scrub").oninput = (e) => { closeup.playing = false; closeup.w = (Number(e.target.value) / 1000) * closeup.total(); };
}

async function main() {
  let viewer, closeup;
  try {
    viewer = new TubeViewer($("c3d"), $("labels"));
    closeup = new Closeup($("cclose"), $("cplot"));
  } catch (e) {
    $("loading").textContent = "This page needs WebGL, which this browser does not provide (" + e.message + ").";
    return;
  }
  state.viewer = viewer;
  window.DOLLE3D = { state, closeup, tick, refresh };   // for inspecting the page from the browser console
  bindControls(closeup);
  const ro = new ResizeObserver(() => { viewer.resize(); closeup.resize(); });
  ro.observe($("c3d")); ro.observe($("cclose"));
  viewer.resize(); closeup.resize();

  const visible = { viewer: true, closeup: false };
  const io = new IntersectionObserver((es) => es.forEach((e) => {
    if (e.target.id === "stage") visible.viewer = e.isIntersecting; else visible.closeup = e.isIntersecting;
  }));
  io.observe($("stage")); io.observe($("cclose").parentElement);

  let tPrev = performance.now();
  const loop = (now) => {
    const dt = Math.min((now - tPrev) / 1000, 0.1); tPrev = now;
    if (visible.viewer && curRun()) { tick(dt); viewer.render(); }
    if (visible.closeup && closeup.case) { closeup.update(dt); closeup.render(); }
    requestAnimationFrame(loop);
  };
  requestAnimationFrame(loop);

  const [rep, close] = await Promise.all([
    fetch("data/two_tubules.json").then((r) => { if (!r.ok) throw new Error("data/two_tubules.json: " + r.status); return r.json(); }),
    fetch("data/impact_closeup.json").then((r) => { if (!r.ok) throw new Error("data/impact_closeup.json: " + r.status); return r.json(); }),
  ]).catch((e) => { $("loading").textContent = "Could not load the replay data (" + e.message + ")."; return [null, null]; });
  if (!rep) return;
  state.runs.replay = runFromReplay(rep);
  selectRun("replay");
  $("loading").classList.add("hidden");
  $("provenance").textContent = `Replay data: ${rep.made_by}${rep.commit ? ", commit " + rep.commit : ""}; engine ${rep.engine}.`;

  const host = $("cases");
  close.cases.forEach((cs, i) => {
    const b = document.createElement("button"); b.className = "tab" + (i === 0 ? " on" : "");
    b.innerHTML = `${cs.material[0].toUpperCase() + cs.material.slice(1)} · ${fmt(cs.v_in_mm_s, 1)} mm/s · ${cs.outcome === "stick" ? "sticks" : "bounces"}`;
    b.onclick = () => { host.querySelectorAll(".tab").forEach((x) => x.classList.remove("on")); b.classList.add("on"); closeup.setCase(cs); closeup.playing = true; };
    host.appendChild(b);
  });
  closeup.setCase(close.cases[0]);
  closeup.update(0); closeup.render();
}
main();
