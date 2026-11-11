// The live sandbox of the DOLL-E 3-D page: runs the repository's python (dolle_core) in the browser with
// Pyodide, off the main thread so the page stays responsive.
//
// Messages in:  {cmd: "run", params}  start a run (dolle_core.scenes.sandbox_start(params_json))
//               {cmd: "stop"}         stop after the current chunk
// Messages out: {type: "status", text}  {type: "ready"}  {type: "chunk", data, wall_s}  {type: "done", stopped}
//               {type: "error", msg}
// The python files are byte-for-byte copies of dolle_core/*.py made by tools/build_sim3d.py (py/manifest.json).
const PYODIDE = "https://cdn.jsdelivr.net/pyodide/v0.28.3/full/";   // pinned version (same as the main site's app)

let py = null;
let stopRequested = false;
let busy = false;

async function boot() {
  postMessage({ type: "status", text: "Downloading python (Pyodide; with numpy about 15 MB the first time, then cached)..." });
  importScripts(PYODIDE + "pyodide.js");
  py = await loadPyodide({ indexURL: PYODIDE });
  postMessage({ type: "status", text: "Loading numpy..." });
  await py.loadPackage(["numpy"]);
  postMessage({ type: "status", text: "Loading dolle_core..." });
  const manifest = await (await fetch("py/manifest.json", { cache: "no-cache" })).json();
  py.FS.mkdirTree("/home/pyodide/dolle_core");
  for (const f of manifest.files) {
    const r = await fetch("py/dolle_core/" + f, { cache: "no-cache" });
    if (!r.ok) throw new Error("could not load py/dolle_core/" + f + " (" + r.status + ")");
    py.FS.writeFile("/home/pyodide/dolle_core/" + f, await r.text());
  }
  py.runPython("import sys\nsys.path.insert(0, '/home/pyodide')\nfrom dolle_core import scenes");
  postMessage({ type: "ready", files: manifest.files.length });
}

let ready = null;
const pause = () => new Promise((res) => setTimeout(res, 0));   // lets a "stop" message in between chunks

async function run(params) {
  busy = true;
  stopRequested = false;
  py.globals.set("params_json", JSON.stringify(params));
  py.runPython("state = scenes.sandbox_start(params_json)");
  let simT = 0;
  for (;;) {
    // Short chunks while the cloud is dense (frames every 4 ms step), longer ones after.
    const chunk = simT < 0.5 ? 0.05 : 0.25;
    const t0 = performance.now();
    const text = py.runPython(`scenes.sandbox_advance(state, ${chunk})`);
    const data = JSON.parse(text);
    postMessage({ type: "chunk", data, wall_s: (performance.now() - t0) / 1000 });
    simT = Math.min(...data.tubules.map((t) => (t.done ? Infinity : t.t)));
    if (data.tubules.every((t) => t.done)) { postMessage({ type: "done", stopped: false }); break; }
    await pause();
    if (stopRequested) { postMessage({ type: "done", stopped: true }); break; }
  }
  py.runPython("del state");
  busy = false;
}

onmessage = async (ev) => {
  const msg = ev.data;
  if (msg.cmd === "stop") { stopRequested = true; return; }
  if (msg.cmd !== "run" || busy) return;
  try {
    if (!ready) ready = boot();
    await ready;
    await run(msg.params);
  } catch (e) {
    busy = false;
    if (!py || !py.globals.has("scenes")) ready = null;     // python did not finish loading: try again next time
    postMessage({ type: "error", msg: String(e && e.message ? e.message : e) });
  }
};
