// Runs the DOLL-E Python simulator in the browser (Pyodide / WebAssembly), off the main thread.
const PYODIDE = "https://cdn.jsdelivr.net/pyodide/v0.28.3/full/";
importScripts(PYODIDE + "pyodide.js");

let pyodide = null;
let W = null;

async function boot() {
  postMessage({ type: "status", msg: "Downloading the Python runtime (Pyodide, ~10 MB)…" });
  pyodide = await loadPyodide({ indexURL: PYODIDE });
  postMessage({ type: "status", msg: "Loading numpy and scipy…" });
  await pyodide.loadPackage(["numpy", "scipy"]);
  postMessage({ type: "status", msg: "Loading the DOLL-E simulator…" });
  const manifest = await (await fetch("py/manifest.json", { cache: "no-cache" })).json();
  pyodide.FS.mkdirTree("/home/pyodide/dolle_sim");
  for (const f of manifest.files) {
    const txt = await (await fetch("py/dolle_sim/" + f, { cache: "no-cache" })).text();
    pyodide.FS.writeFile("/home/pyodide/dolle_sim/" + f, txt);
  }
  pyodide.runPython("import sys; sys.path.insert(0, '/home/pyodide')\nimport dolle_sim.webapi as W");
  W = pyodide.globals.get("W");
  postMessage({ type: "ready", version: manifest.version });
}

const ready = boot().catch((e) => postMessage({ type: "fatal", msg: String(e) }));

onmessage = async (ev) => {
  await ready;
  const { id, fn, arg } = ev.data;
  try {
    let res;
    if (fn === "run") {
      const progress = (s) => postMessage({ type: "progress", id, data: s });
      res = W.run(arg, progress);
    } else {
      res = W[fn](arg);
    }
    postMessage({ type: "result", id, data: res });
  } catch (e) {
    postMessage({ type: "error", id, msg: String(e) });
  }
};
