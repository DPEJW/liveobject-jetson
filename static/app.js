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

// ---- charts ----
function makeChart(ctx, label, color, suggestedMax) {
  return new Chart(ctx, {
    type: "line",
    data: {
      labels: Array(WINDOW).fill(""),
      datasets: [{
        label, data: Array(WINDOW).fill(null),
        borderColor: color, backgroundColor: color + "22",
        borderWidth: 2, pointRadius: 0, tension: 0.3, fill: true, spanGaps: true,
      }],
    },
    options: {
      animation: false, responsive: true, maintainAspectRatio: false,
      plugins: { legend: { display: false }, title: { display: true, text: label, color: "#8b97a6", font: { size: 11 } } },
      scales: {
        x: { display: false },
        y: { beginAtZero: true, suggestedMax, ticks: { color: "#8b97a6", maxTicksLimit: 4 }, grid: { color: "#2a323d" } },
      },
    },
  });
}

const ramChart = makeChart($("ramChart"), "RAM %", "#4aa8ff", 100);
const fpsChart = makeChart($("fpsChart"), "FPS", "#36d399", 30);

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

$("maxd").addEventListener("input", (e) => { $("maxd-val").textContent = e.target.value; });
$("maxd").addEventListener("change", (e) => postConfig({ max_detections: +e.target.value }));

$("thr").addEventListener("input", (e) => { $("thr-val").textContent = (e.target.value / 100).toFixed(2); });
$("thr").addEventListener("change", (e) => postConfig({ threshold: e.target.value / 100 }));

$("model").addEventListener("change", (e) => postConfig({ model: e.target.value }));

let paused = false;
$("pause").addEventListener("click", () => {
  paused = !paused;
  postConfig({ paused });
});

$("snap").addEventListener("click", () => {
  $("snap").textContent = "Saving…";
  fetch("/snapshot", { method: "POST" })
    .then((r) => r.json())
    .then((j) => {
      $("snap").textContent = "Snapshot";
      if (j.path) {
        const link = $("snap-link");
        link.href = "/snapshot/latest?t=" + Date.now();
        link.classList.remove("hidden");
      }
    })
    .catch(() => { $("snap").textContent = "Snapshot"; });
});

// ---- detections list ----
function renderDetections(dets) {
  const list = $("det-list");
  if (!dets.length) {
    list.innerHTML = '<li class="muted">no objects detected</li>';
    return;
  }
  list.innerHTML = dets.map((d) =>
    `<li><span class="swatch" style="background:${colorFor(d.name)}"></span>` +
    `<span class="nm">${d.name}</span>` +
    `<span class="sc">${(d.score * 100).toFixed(0)}%</span></li>`
  ).join("");
}

// ---- stats poll ----
function applyConfigOnce(cfg) {
  if (configInit) return;
  configInit = true;
  const sel = $("model");
  sel.innerHTML = cfg.models.map((m) =>
    `<option value="${m}"${m === cfg.model ? " selected" : ""}>${m}</option>`).join("");
  $("maxd").value = cfg.max_detections; $("maxd-val").textContent = cfg.max_detections;
  $("thr").value = Math.round(cfg.threshold * 100); $("thr-val").textContent = cfg.threshold.toFixed(2);
  paused = cfg.paused;
}

function poll() {
  fetch("/stats").then((r) => r.json()).then((s) => {
    $("status-dot").classList.remove("stale");
    $("m-fps").textContent = s.fps.toFixed(1);
    $("m-infer").textContent = s.infer_ms.toFixed(1);
    $("m-count").textContent = s.count;
    $("m-model").textContent = s.config.model;

    $("s-ram").textContent = s.ram_pct.toFixed(0);
    $("s-rammb").textContent = s.ram_used_mb + " / " + s.ram_total_mb;
    $("s-cpu").textContent = s.cpu_pct.toFixed(0);
    $("s-temp").textContent = s.cpu_temp == null ? "—" : s.cpu_temp;

    $("det-count").textContent = s.count;
    renderDetections(s.detections);

    pushPoint(ramChart, s.ram_pct);
    pushPoint(fpsChart, s.fps);

    applyConfigOnce(s.config);

    paused = s.config.paused;
    $("pause").textContent = paused ? "Resume" : "Pause";
    $("pause").classList.toggle("paused", paused);
    $("paused-overlay").classList.toggle("hidden", !paused);
  }).catch(() => {
    $("status-dot").classList.add("stale");
  });
}

setInterval(poll, POLL_MS);
poll();
