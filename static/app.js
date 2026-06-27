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
