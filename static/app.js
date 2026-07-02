"use strict";

const $ = (id) => document.getElementById(id);
const WINDOW = 30;        // points kept on each chart
const POLL_MS = 500;      // stats poll interval

// ---- consistent per-class colors (mirrors the server's md5 scheme loosely) ----
function colorFor(name) {
  let h = 0;
  for (let i = 0; i < name.length; i++) h = (h * 31 + name.charCodeAt(i)) >>> 0;
  const r = 60 + (h & 0xff) % 180;
  const g = 60 + ((h >> 8) & 0xff) % 180;
  const b = 60 + ((h >> 16) & 0xff) % 180;
  return `rgb(${r},${g},${b})`;
}

// ---- charts (instrument palette) ----
function makeChart(ctx, label, color, suggestedMax) {
  return new Chart(ctx, {
    type: "line",
    data: {
      labels: Array(WINDOW).fill(""),
      datasets: [{
        label, data: Array(WINDOW).fill(null),
        borderColor: color, backgroundColor: color + "1f",
        borderWidth: 1.5, pointRadius: 0, tension: 0.32, fill: true, spanGaps: true,
      }],
    },
    options: {
      animation: false, responsive: true, maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        title: { display: true, text: label, color: "#687a89", align: "start",
                 font: { family: "'IBM Plex Mono', monospace", size: 10 } },
      },
      scales: {
        x: { display: false },
        y: { beginAtZero: true, suggestedMax,
             ticks: { color: "#455463", maxTicksLimit: 4, font: { size: 9 } },
             grid: { color: "rgba(110,150,175,.08)" }, border: { display: false } },
      },
    },
  });
}

const fpsChart = makeChart($("fpsChart"), "FPS — 15s", "#ffb02e", 30);
const ramChart = makeChart($("ramChart"), "RAM % — 15s", "#4fd6c6", 100);

function pushPoint(chart, value) {
  const d = chart.data.datasets[0].data;
  d.push(value); d.shift();
  chart.update("none");
}

// ---- config controls ----
let configInit = false;

function postConfig(payload) {
  return fetch("/config", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  }).then((r) => r.json());
}

function postTrack(body) {
  return fetch("/track", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  }).then((r) => r.json());
}

function fmtDur(s) {
  return s >= 60 ? `${Math.floor(s / 60)}m${String(Math.round(s % 60)).padStart(2, "0")}s`
                 : `${(s || 0).toFixed(0)}s`;
}

function setFill(el) {
  const min = +el.min, max = +el.max;
  el.style.setProperty("--fill", ((el.value - min) / (max - min) * 100) + "%");
}

$("maxd").addEventListener("input", (e) => { $("maxd-val").textContent = e.target.value; setFill(e.target); });
$("maxd").addEventListener("change", (e) => postConfig({ max_detections: +e.target.value }));

$("thr").addEventListener("input", (e) => { $("thr-val").textContent = (e.target.value / 100).toFixed(2); setFill(e.target); });
$("thr").addEventListener("change", (e) => postConfig({ threshold: e.target.value / 100 }));

$("model").addEventListener("change", (e) => postConfig({ model: e.target.value }));

// ---- camera source / stream selection ----
function updateStreamEnabled() {
  $("stream-ctl").classList.toggle("dim", $("camera-source").value !== "rtsp");
}
$("camera-source").addEventListener("change", (e) => {
  postConfig({ camera_source: e.target.value });
  updateStreamEnabled();
});
$("rtsp-stream").addEventListener("change", (e) => postConfig({ rtsp_stream: e.target.value }));

// ---- tracking + enrollment controls ----
let enrollMode = false;
function setEnrollMode(on) {
  enrollMode = on;
  $("enroll-btn").classList.toggle("on", on);
  $("enroll-hint").textContent = "click a cat on the video…";
  $("enroll-hint").classList.toggle("hidden", !on);
}
$("enroll-btn").addEventListener("click", () => setEnrollMode(!enrollMode));
$("cats-list").addEventListener("click", (e) => {
  const b = e.target.closest("button[data-clear]");
  if (b) fetch("/enroll", { method: "POST", headers: { "Content-Type": "application/json" },
                            body: JSON.stringify({ action: "clear", name: b.dataset.clear }) });
});
$("stream").addEventListener("click", (e) => {
  const r = e.currentTarget.getBoundingClientRect();
  const nx = (e.clientX - r.left) / r.width, ny = (e.clientY - r.top) / r.height;
  if (enrollMode) {
    const name = (prompt("Name this cat:") || "").trim();
    if (name) fetch("/enroll", { method: "POST", headers: { "Content-Type": "application/json" },
                                 body: JSON.stringify({ action: "enroll", name, x: nx, y: ny }) });
    setEnrollMode(false);
    return;
  }
  postTrack({ action: "select", x: nx, y: ny });
});
$("trk-stop").addEventListener("click", () => postTrack({ action: "stop" }));
$("trk-list").addEventListener("click", (e) => {           // event delegation
  const li = e.target.closest("li[data-id]");
  if (li) postTrack({ action: "select", id: +li.dataset.id });
});
function wireToggle(id, key) {
  $(id).addEventListener("click", () => {
    const on = !$(id).classList.contains("on");
    $(id).classList.toggle("on", on);
    postConfig({ [key]: on });
  });
}
wireToggle("tg-trail", "track_trail");
wireToggle("tg-heatmap", "track_heatmap");
wireToggle("tg-zones", "track_zones");

// ---- training progress ----
const mapChart = new Chart($("mapChart"), {
  type: "line",
  data: { labels: [], datasets: [{ label: "mAP50", data: [], borderColor: "#4fd6c6",
          backgroundColor: "#4fd6c61f", borderWidth: 1.5, pointRadius: 0, tension: 0.3, fill: true }] },
  options: { animation: false, responsive: true, maintainAspectRatio: false,
    plugins: { legend: { display: false },
               title: { display: true, text: "mAP50 vs epoch", color: "#687a89", align: "start",
                        font: { family: "'IBM Plex Mono', monospace", size: 10 } } },
    scales: { x: { display: false },
              y: { beginAtZero: true, suggestedMax: 1, ticks: { color: "#455463", maxTicksLimit: 4, font: { size: 9 } },
                   grid: { color: "rgba(110,150,175,.08)" }, border: { display: false } } } },
});
let trainLastEpoch = -1;
let identEnabled = true;
$("ident-toggle").addEventListener("click", () =>
  postConfig({ identity_enabled: !identEnabled }));
$("train-btn").addEventListener("click", () =>
  fetch("/train", { method: "POST", headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ action: "start" }) }));
$("train-cancel").addEventListener("click", () =>
  fetch("/train", { method: "POST", headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ action: "cancel" }) }));

function trainPoll() {
  fetch("/train/status").then((r) => r.json()).then((t) => {
    const st = t.state || "idle";
    $("train-state").textContent = t.running ? st : st;
    $("train-prog").classList.toggle("hidden",
      !(t.running || ["training", "starting", "exporting", "building", "done", "error"].includes(st)));
    $("train-cancel").classList.toggle("hidden", !t.running);
    $("train-btn").disabled = !!t.running;
    $("train-overlay").classList.toggle("hidden", !t.running);
    if (t.running) {
      $("train-overlay-tx").textContent =
        st === "exporting" ? "⚙ EXPORTING MODEL — FEED PAUSED"
        : st === "building" ? "⚙ BUILDING TENSORRT ENGINE — FEED PAUSED (a few minutes)"
        : `⚙ TRAINING ${t.epoch || 0}/${t.epochs || "?"} — FEED PAUSED`;
    }
    if (st === "error") {
      $("tp-epoch").textContent = "—";
      $("tp-map").textContent = String(t.msg || t.err || "error").slice(0, 40);
      return;
    }
    const ep = t.epoch || 0, eps = t.epochs || 0;
    $("tp-epoch").textContent = ep + "/" + eps;
    $("tp-loss").textContent = (t.loss != null ? t.loss : "—");
    $("tp-map").textContent = (t.mAP50 != null ? t.mAP50 : "—");
    $("tp-bar").style.width = (eps ? Math.min(100, 100 * ep / eps) : 0) + "%";
    if (st === "starting" || ep < trainLastEpoch) {
      mapChart.data.labels = []; mapChart.data.datasets[0].data = []; trainLastEpoch = -1;
    }
    if (t.mAP50 != null && ep > trainLastEpoch && ep > 0) {
      mapChart.data.labels.push(ep); mapChart.data.datasets[0].data.push(t.mAP50);
      mapChart.update("none"); trainLastEpoch = ep;
    }
  }).catch(() => {});
}
setInterval(trainPoll, 1000); trainPoll();

$("rotate").addEventListener("click", () => {
  const cur = +($("rotate").dataset.rot || 0);
  postConfig({ rotation: (cur + 90) % 360 });
});

let flipH = false, flipV = false;
$("flip-h").addEventListener("click", () => postConfig({ flip_h: !flipH }));
$("flip-v").addEventListener("click", () => postConfig({ flip_v: !flipV }));

let paused = false;
$("pause").addEventListener("click", () => { paused = !paused; postConfig({ paused }); });

$("snap").addEventListener("click", () => {
  $("snap").textContent = "SAVING…";
  fetch("/snapshot", { method: "POST" })
    .then((r) => r.json())
    .then((j) => {
      $("snap").textContent = "SNAPSHOT";
      if (j.path) {
        const link = $("snap-link");
        link.href = "/snapshot/latest?t=" + Date.now();
        link.classList.remove("hidden");
      }
    })
    .catch(() => { $("snap").textContent = "SNAPSHOT"; });
});

// ---- fullscreen ----
const videoWrap = $("video-wrap");
const fsExit = $("fs-exit");
function enterFullscreen() {
  const fn = videoWrap.requestFullscreen || videoWrap.webkitRequestFullscreen || videoWrap.msRequestFullscreen;
  if (fn) fn.call(videoWrap);
}
function exitFullscreen() {
  const fn = document.exitFullscreen || document.webkitExitFullscreen || document.msExitFullscreen;
  if (fn) fn.call(document);
}
function onFsChange() {
  const fs = document.fullscreenElement || document.webkitFullscreenElement;
  fsExit.classList.toggle("hidden", !fs);
}
$("fs-btn").addEventListener("click", enterFullscreen);
fsExit.addEventListener("click", exitFullscreen);
document.addEventListener("fullscreenchange", onFsChange);
document.addEventListener("webkitfullscreenchange", onFsChange);

// ---- clock ----
function tick() {
  $("clock").textContent = new Date().toLocaleTimeString("en-GB");
}
setInterval(tick, 1000); tick();

// ---- detections ----
function renderDetections(dets) {
  const list = $("det-list");
  if (!dets.length) { list.innerHTML = '<li class="muted">no objects detected</li>'; return; }
  dets = dets.slice().sort((a, b) =>            // stable order: no row jumping
    a.name < b.name ? -1 : a.name > b.name ? 1 : b.score - a.score);
  list.innerHTML = dets.map((d) =>
    `<li><span class="swatch" style="background:${colorFor(d.name)}"></span>` +
    `<span class="nm">${d.name}</span>` +
    `<span class="sc">${(d.score * 100).toFixed(0)}%</span></li>`
  ).join("");
}

// ---- one-time config hydration ----
function applyConfigOnce(cfg) {
  if (configInit) return;
  configInit = true;

  $("model").innerHTML = cfg.models.map((m) =>
    `<option value="${m}"${m === cfg.model ? " selected" : ""}>${m}</option>`).join("");

  $("camera-source").innerHTML = (cfg.camera_sources || ["rtsp"]).map((s) =>
    `<option value="${s}"${s === cfg.camera_source ? " selected" : ""}>${s.toUpperCase()}</option>`).join("");
  $("rtsp-stream").innerHTML = (cfg.rtsp_streams || ["main", "sub"]).map((s) =>
    `<option value="${s}"${s === cfg.rtsp_stream ? " selected" : ""}>${s.toUpperCase()}</option>`).join("");
  updateStreamEnabled();

  $("maxd").value = cfg.max_detections; $("maxd-val").textContent = cfg.max_detections; setFill($("maxd"));
  $("thr").value = Math.round(cfg.threshold * 100); $("thr-val").textContent = cfg.threshold.toFixed(2); setFill($("thr"));
  paused = cfg.paused;
  $("tg-trail").classList.toggle("on", cfg.track_trail);
  $("tg-heatmap").classList.toggle("on", cfg.track_heatmap);
  $("tg-zones").classList.toggle("on", cfg.track_zones);
}

// ---- stats poll ----
function poll() {
  fetch("/stats").then((r) => r.json()).then((s) => {
    const cfg = s.config;
    $("status-dot").classList.remove("stale");

    $("m-fps").textContent = s.fps.toFixed(1);
    $("m-infer").textContent = s.infer_ms.toFixed(0);
    $("m-count").textContent = s.count;
    $("m-model").textContent = cfg.model;
    $("backend").textContent = cfg.backend || "—";

    const srcLabel = cfg.camera_source === "rtsp"
      ? `RTSP / ${(cfg.rtsp_stream || "").toUpperCase()}` : (cfg.camera_source || "").toUpperCase();
    $("vp-source").textContent = srcLabel;

    $("rotate").dataset.rot = cfg.rotation;
    $("rot-val").textContent = cfg.rotation + "°";
    flipH = cfg.flip_h; flipV = cfg.flip_v;
    $("flip-h").classList.toggle("on", flipH);
    $("flip-v").classList.toggle("on", flipV);

    const cpu = s.cpu_pct, ram = s.ram_pct;
    $("s-cpu").textContent = cpu.toFixed(0);
    $("s-ram").textContent = ram.toFixed(0);
    $("s-rammb").textContent = s.ram_used_mb;
    $("s-temp").textContent = s.cpu_temp == null ? "—" : s.cpu_temp;
    $("bar-cpu").style.width = Math.min(100, cpu) + "%";
    $("bar-ram").style.width = Math.min(100, ram) + "%";
    $("bar-cpu").classList.toggle("hot", cpu >= 85);
    $("bar-ram").classList.toggle("hot", ram >= 85);

    $("det-count").textContent = s.count;
    renderDetections(s.detections);

    // ---- cats + tracking ----
    const cats = s.cats || [];
    $("cats-list").innerHTML = cats.map((n) =>
      `<li><span class="swatch" style="background:${colorFor(n)}"></span>`
      + `<span class="nm">${n}</span>`
      + `<button class="x" data-clear="${n}" title="forget">✕</button></li>`).join("");
    if (s.enrolling) {
      $("enroll-hint").textContent = "learning " + s.enrolling + "…";
      $("enroll-hint").classList.remove("hidden");
    } else if (!enrollMode) {
      $("enroll-hint").classList.add("hidden");
    }

    const ds = s.dataset || {};
    const dsKeys = Object.keys(ds);
    $("ds-counts").innerHTML = dsKeys.length
      ? dsKeys.map((n) => `<li><span class="swatch" style="background:${colorFor(n)}"></span>`
          + `<span class="nm">${n}</span><span class="sc">${ds[n]} frames</span></li>`).join("")
      : '<li class="muted">no samples yet — name a cat & keep it in view</li>';

    const tracks = (s.tracks || []).slice().sort((a, b) => a.id - b.id);  // stable rows
    $("trk-list").innerHTML = tracks.length
      ? tracks.map((t) =>
          `<li data-id="${t.id}" class="${t.selected ? "sel" : ""}">`
          + `<span class="swatch" style="background:${colorFor(t.cls)}"></span>`
          + `<span class="nm">${t.label}</span>`
          + `<span class="sc">${(t.score * 100).toFixed(0)}%</span></li>`).join("")
      : '<li class="muted">no objects yet</li>';

    const tk = s.track;
    $("trk-stop").classList.toggle("invis", !tk || tk.state === "stopped");
    $("trk-state").textContent = tk ? tk.state : "idle";
    if (tk) {
      $("trk-elapsed").textContent = fmtDur(tk.elapsed_s);
      $("trk-moving").textContent = fmtDur(tk.moving_s);
      $("trk-still").textContent = fmtDur(tk.still_s);
      $("trk-dist").textContent = `${tk.dist_px.toFixed(0)}px · ${tk.dist_frames}×`;
      $("trk-place").textContent = tk.current_zone || "—";
      $("trk-zones").innerHTML = (tk.zones || []).map((z) =>
        `<li><span class="nm">${z.label}</span>`
        + `<span class="sc">${fmtDur(z.dwell_s)}</span></li>`).join("");
    } else {                                       // keep size: dashes, not hiding
      ["trk-elapsed", "trk-moving", "trk-still", "trk-dist", "trk-place"]
        .forEach((id) => { $(id).textContent = "—"; });
      $("trk-zones").innerHTML = "";
    }

    const idNames = cfg.identity_names || [];
    $("ident-names").textContent = idNames.length
      ? idNames.join(", ") + (cfg.identity_active ? "" : " (loading…)")
      : "not trained yet";
    identEnabled = !!cfg.identity_enabled;
    $("ident-toggle").textContent = identEnabled ? "ON" : "OFF";
    $("ident-toggle").classList.toggle("on", identEnabled);

    pushPoint(fpsChart, s.fps);
    pushPoint(ramChart, ram);

    applyConfigOnce(cfg);

    paused = cfg.paused;
    $("pause").textContent = paused ? "RESUME" : "PAUSE";
    $("pause").classList.toggle("paused", paused);
    $("paused-overlay").classList.toggle("hidden", !paused);
  }).catch(() => {
    $("status-dot").classList.add("stale");
  });
}

setInterval(poll, POLL_MS);
poll();
