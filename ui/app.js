// aiVibe Studio window logic. All work happens in Python (window.pywebview.api); this file only draws.
"use strict";
const $ = id => document.getElementById(id);
let api, settings, models = [], modelById = {}, lastSeq = 0, selectedPath = null, galleryKey = "";
let picked = [], galleryPaths = [], anchor = null, upChoice = "4x";
let subjects = [], subjectPicks = [], reference = null;
let progress = {step: 0, steps: 0, trainEta: null, text: ""};   // latest step/tile/train event for the busy window   // subjectPicks: [{id, weight}]
const ASPECTS = {"1:1": [1, 1], "4:5": [4, 5], "16:9": [16, 9]};
const SHORT_PROMPT_FAMILIES = new Set(["sd15", "sdxl", "playground"]);   // CLIP stops reading at 77 tokens

function el(tag, props = {}, ...kids) {
  const e = document.createElement(tag);
  Object.entries(props).forEach(([k, v]) => {
    if (v == null || v === false) return;       // no attribute at all: selected="null" would still select
    k === "class" ? e.className = v : k.startsWith("on") ? e[k] = v : e.setAttribute(k, v);
  });
  kids.forEach(k => e.append(k instanceof Node ? k : document.createTextNode(String(k))));
  return e;
}
const time = t => new Date(t * 1000).toLocaleTimeString([], {hour12: false});

// ---------- diagnostics: window errors and clicks go to studio.log ----------
const early = [];      // before pywebview is ready
function flog(level, msg) {
  try { if (api) api.log_client(level, msg); else early.push([level, msg]); } catch (e) { /* logging must never break the app */ }
}
window.addEventListener("error", e => flog("error", `window error: ${e.message} at ${e.filename}:${e.lineno}:${e.colno}\n${e.error && e.error.stack || ""}`));
window.addEventListener("unhandledrejection", e => flog("error", `unhandled promise: ${e.reason && (e.reason.stack || e.reason.message) || e.reason}`));
document.addEventListener("click", e => {
  const t = e.target.closest("button, a, input[type=checkbox], select, .thumb, #previewImg");
  if (!t) return;
  const what = t.id ? `#${t.id}` : t.classList.contains("thumb") ? `thumb ${t.title.slice(0, 60)}` : t.tagName.toLowerCase();
  const label = (t.textContent || t.value || "").trim().replace(/\s+/g, " ").slice(0, 50);
  flog("info", `click ${what}${label ? ` "${label}"` : ""}${t.type === "checkbox" ? ` -> ${t.checked}` : ""}`);
}, true);

async function reportOpen() {
  if (!$("report").hidden) return;
  try { await api.report_start(); } catch (e) { flog("error", `report screenshot: ${e}`); }
  $("reportNote").value = ""; $("report").hidden = false; $("reportNote").focus();
}
async function reportSave() {
  const note = $("reportNote").value.trim();
  if (!note) { $("reportNote").focus(); return; }
  const r = await api.report_problem(note, selectedPath);
  $("report").hidden = true;
  if (r.error) return log(r.error, "em");
  log(`Problem report saved (${r.name}). Thanks, it is on this PC for Claude to read.`, "b");
}

function log(text, cls) {
  if (cls === "em") flog("warn", `shown in Log: ${text}`);
  const line = el("div", {}, `[${time(Date.now() / 1000)}] `);
  line.append(cls ? el(cls, {}, text) : text);
  $("log").append(line);
  $("log").scrollTop = $("log").scrollHeight;
}

// ---------- settings + controls ----------
function current() {
  return {
    model: $("model").value, prompt: $("prompt").value, negative: $("negative").value, name: $("name").value,
    width: +$("width").value, height: +$("height").value, steps: +$("steps").value, guidance: +$("guidance").value,
    count: +$("count").value, seed: $("randomSeed").checked ? -1 : +$("seed").value,
    subjects: subjectPicks, reference: reference ? reference.path : null, reference_strength: +$("refStrength").value,
    face_fix: $("faceFix").checked,
  };
}
let saveTimer;
function remember() {
  clearTimeout(saveTimer);
  saveTimer = setTimeout(() => {
    const c = current();
    api.save_settings({model: c.model, prompt: c.prompt, negative: c.negative, width: c.width, height: c.height,
      steps: c.steps, guidance: c.guidance, count: c.count, random_seed: $("randomSeed").checked, seed: +$("seed").value});
  }, 600);
}
function setSlider(id, v) { $(id).value = v; $(id + "Out").textContent = (id === "guidance" || id === "vidGuidance") ? (+v).toFixed(1) : v; }
const imageModels = () => models.filter(m => m.kind !== "video");
const videoModels = () => models.filter(m => m.kind === "video");
function summary() {
  const m = modelById[$("model").value];
  const n = +$("count").value;
  $("summary").textContent = `${$("width").value} × ${$("height").value} · ${n} render${n > 1 ? "s" : ""} · ${m ? m.name : ""}`;
}
function tokenCount() {
  // rough CLIP estimate: words plus punctuation marks
  const t = $("prompt").value.match(/[A-Za-z0-9']+|[^\sA-Za-z0-9]/g) || [];
  const m = modelById[$("model").value];
  const short = m && SHORT_PROMPT_FAMILIES.has(m.family);
  $("promptCount").textContent = short ? `~${t.length} / 75 tokens` : `~${t.length} tokens`;
  $("promptCount").classList.toggle("warn", short && t.length > 75);
  $("promptCount").title = short && t.length > 75 ? "This model stops reading after about 75 tokens; the end of the prompt is ignored." : "";
}
function showModel() {
  const m = modelById[$("model").value];
  if (!m) return;
  $("modelDesc").textContent = m.description || "";
  $("modelLicense").textContent = {commercial: "commercial use OK", conditions: "free with conditions", nc: "personal use only"}[m.license];
  $("modelLicense").className = "tag" + (m.license === "nc" ? " nc" : "");
  const fit = {ok: "", tight: " · may be slow on this card", too_big: ` · needs ${m.min_vram_gb} GB of video memory`}[m.fit] || "";
  $("modelState").textContent = (m.downloaded ? "on disk" : `${m.size_gb} GB to download`) + fit;
  $("modelState").style.color = m.fit === "too_big" ? "var(--bad)" : m.fit === "tight" ? "var(--warn)" : "";
  $("btnGuide").disabled = !m.guide;
  likenessAvailability();
  summary(); tokenCount();
}
function recommended() {
  const d = modelById[$("model").value].defaults;
  setSlider("steps", d.steps); setSlider("guidance", d.guidance);
  $("width").value = d.width; $("height").value = d.height;
  document.querySelectorAll("#aspect button").forEach(b => b.classList.toggle("on", b.dataset.a === "1:1"));
  summary(); remember();
}
function fillModels(keep, keepVid) {
  const imgs = imageModels();
  $("model").replaceChildren(...imgs.map(m => el("option", {value: m.id}, m.name + (m.downloaded ? "" : "  (download)")
    + (m.fit === "too_big" ? "  (too big for this card)" : m.fit === "tight" ? "  (may be slow)" : ""))));
  if (keep && modelById[keep] && modelById[keep].kind !== "video") $("model").value = keep;
  showModel();
  const vids = videoModels();
  if ($("vidModel")) {
    const prevVid = keepVid || $("vidModel").value || (settings && settings.last.vid_model) || "ltx-video";
    $("vidModel").replaceChildren(...vids.map(m => el("option", {value: m.id}, m.name + (m.downloaded ? "" : "  (download)")
      + (m.fit === "too_big" ? "  (too big for this card)" : m.fit === "tight" ? "  (may be slow)" : ""))));
    if (prevVid && modelById[prevVid] && modelById[prevVid].kind === "video") $("vidModel").value = prevVid;
    showVidModel();
  }
}
async function loadModels(keep) {
  models = await api.models();
  modelById = Object.fromEntries(models.map(m => [m.id, m]));
  fillModels(keep || $("model").value, $("vidModel") && $("vidModel").value);
  if (!$("batchMaker").hidden) bmRender();
}
function fillChips() {
  const chips = $("chips"); chips.replaceChildren();
  (settings.last.styles || []).forEach((s, i) => {
    const c = el("span", {class: "chip", title: s.text + "\n(right-click to remove)"}, s.name);
    c.onclick = () => { const p = $("prompt"); p.value = p.value.trim() ? `${p.value.trim().replace(/,$/, "")}, ${s.text}` : s.text; tokenCount(); remember(); };
    c.oncontextmenu = e => { e.preventDefault(); if (confirm(`Remove the "${s.name}" style?`)) { settings.last.styles.splice(i, 1); api.save_settings({styles: settings.last.styles}); fillChips(); } };
    chips.append(c);
  });
  const add = el("span", {class: "chip add"}, "+ style");
  add.onclick = () => {
    const name = prompt("Style button name:"); if (!name) return;
    const text = prompt(`Text "${name}" adds to the prompt:`); if (!text) return;
    settings.last.styles.push({name, text}); api.save_settings({styles: settings.last.styles}); fillChips();
  };
  chips.append(add);
}
function fillNegatives() {
  const sel = $("negPreset");
  sel.replaceChildren(el("option", {value: ""}, "Presets…"),
    ...(settings.last.negatives || []).map((n, i) => el("option", {value: i}, n.name)), el("option", {value: "save"}, "Save current as preset…"));
  sel.onchange = () => {
    if (sel.value === "save") {
      const name = prompt("Preset name:");
      if (name) { settings.last.negatives.push({name, text: $("negative").value}); api.save_settings({negatives: settings.last.negatives}); fillNegatives(); }
    } else if (sel.value !== "") { $("negative").value = settings.last.negatives[+sel.value].text; remember(); }
    sel.value = "";
  };
  bmFillNegPresets();
  vidFillNegPresets();
}
const PANELS = {model: "Model", subjects: "Subjects", reference: "Reference image", canvas: "Canvas", quality: "Quality",
  variations: "Variations", device: "Processing device", upscale: "Upscale", assistant: "Assistant", log: "Log"};
// ---------- workspace layout: columns of movable panels ----------
const DEFAULT_LAYOUT = [["model", "subjects", "reference", "canvas", "quality"], ["studio"], ["variations", "device", "upscale", "assistant", "log"]];
const PRESETS = {
  "Default": DEFAULT_LAYOUT,
  "Wide canvas": [["model", "subjects", "reference", "canvas", "quality", "variations", "device", "upscale", "log"], ["studio"]],
  "Four columns": [["model", "subjects", "reference"], ["canvas", "quality", "variations", "device"], ["studio"], ["upscale", "log"]],
  "Canvas left": [["studio"], ["model", "subjects", "reference", "canvas"], ["quality", "variations", "device", "upscale", "log"]],
};
function panelEls() {
  const m = {};
  document.querySelectorAll("#work [data-panel]").forEach(p => { m[p.dataset.panel] = p; });
  return m;
}
function currentLayout() {
  return [...document.querySelectorAll("#work > .col")].map(c => [...c.querySelectorAll(":scope > [data-panel]")].map(p => p.dataset.panel));
}
function buildLayout(layout) {
  const els = panelEls(), work = $("work");
  const placed = new Set();
  const cols = (layout && layout.length ? layout : DEFAULT_LAYOUT).map(ids => {
    const col = el("div", {class: "col"});
    ids.forEach(id => { if (els[id] && !placed.has(id)) { col.append(els[id]); placed.add(id); } });
    return col;
  });
  if (!placed.has("studio")) cols.splice(Math.min(1, cols.length), 0, el("div", {class: "col"}, els.studio)), placed.add("studio");
  // panels the saved layout does not mention (new ones) go to the last column without the picture area
  const spare = [...cols].reverse().find(c => !c.querySelector('[data-panel="studio"]')) || cols[cols.length - 1];
  const logPanel = spare.querySelector(':scope > [data-panel="log"]');      // new panels go above the Log
  Object.keys(els).forEach(id => { if (!placed.has(id)) logPanel ? spare.insertBefore(els[id], logPanel) : spare.append(els[id]); });
  work.replaceChildren(...cols);
  cols.forEach(markColumn);
  cols.forEach(bindDrop);
}
function markColumn(col) {
  col.classList.toggle("wide", !!col.querySelector('[data-panel="studio"]'));
  const empty = !col.querySelector("[data-panel]");
  col.classList.toggle("empty", empty);
  if (empty && !col.textContent.trim()) col.append(el("span", {class: "hint"}, "Drag panels here"));
  if (!empty) col.querySelectorAll(":scope > .hint").forEach(h => h.remove());
}
async function saveLayout() {
  settings.last.layout = currentLayout();
  document.querySelectorAll("#work > .col").forEach(markColumn);
  await api.save_settings({layout: settings.last.layout});
}
let dragged = null, mark = el("div", {class: "drop-mark"});
function bindDrag() {
  document.querySelectorAll("#work [data-panel]").forEach(p => {
    const handle = p.matches(".studio") ? p.querySelector(".summary .grip") : p.querySelector(":scope > h3");
    if (!handle) return;
    handle.style.cursor = "grab";
    handle.onmousedown = () => { p.draggable = true; };
    p.ondragstart = e => { dragged = p; p.classList.add("dragging"); e.dataTransfer.effectAllowed = "move"; e.dataTransfer.setData("text/plain", p.dataset.panel); };
    p.ondragend = () => { p.draggable = false; p.classList.remove("dragging"); mark.remove(); dragged = null; };
  });
}
function bindDrop(col) {
  col.ondragover = e => {
    if (!dragged) return;
    e.preventDefault();
    const after = [...col.querySelectorAll(":scope > [data-panel]")].filter(p => p !== dragged)
      .find(p => e.clientY < p.getBoundingClientRect().top + p.getBoundingClientRect().height / 2);
    after ? col.insertBefore(mark, after) : col.append(mark);
  };
  col.ondrop = e => {
    if (!dragged) return;
    e.preventDefault();
    mark.replaceWith(dragged);
    saveLayout();
  };
}
function setColumns(n) {
  const layout = currentLayout();
  while (layout.length < n) layout.push([]);
  while (layout.length > n) {
    const gone = layout.pop();
    const target = layout.findLastIndex(c => !c.includes("studio"));
    (layout[target >= 0 ? target : layout.length - 1]).push(...gone);
  }
  buildLayout(layout); bindDrag(); saveLayout();
}
function applyLayoutPreset(name) { buildLayout(PRESETS[name]); bindDrag(); saveLayout(); }

function placeRecent(mode) {
  // recent pictures along the bottom, the top, as a movable panel, or not at all
  mode = mode || "bottom";
  const app = document.querySelector(".app");
  app.classList.remove("recent-top", "recent-off", "recent-column");
  if (mode === "top") app.classList.add("recent-top");
  if (mode === "off") app.classList.add("recent-off");
  if (mode === "column") app.classList.add("recent-column");
  (mode === "column" ? $("recentSlot") : $("dock")).append($("gallery"));
}

function applyPanels() {
  const hidden = settings.last.hidden_panels || [];
  document.querySelectorAll("section.panel[data-panel]").forEach(p =>
    p.classList.toggle("hidden-by-user", hidden.includes(p.dataset.panel)));
}
function settingsModal() {
  const reportsList = el("div", {class: "desc"}, "Loading reports…");
  api.reports().then(rs => reportsList.replaceChildren(rs.length ? el("ul", {}, ...rs.map(r => el("li", {}, `${r.name}: ${r.note}`))) : "No reports yet."));
  const hidden = new Set(settings.last.hidden_panels || []);
  const folder = el("input", {value: settings.output_root, readonly: "readonly"});
  const warn = el("div", {class: "desc", style: "color:var(--warn)"},
    settings.backed_up ? "" : "This folder is not in Pictures, Documents or Downloads, so backup apps that watch those folders may not copy it.");
  const batchSub = el("input", {type: "checkbox", style: "width:auto"});
  batchSub.checked = settings.last.batch_subfolder !== false;
  batchSub.onchange = async () => { settings.last.batch_subfolder = batchSub.checked; await api.save_settings({batch_subfolder: batchSub.checked}); };
  const idle = el("input", {type: "number", min: "1", max: "60", value: String(settings.idle_minutes), style: "width:80px"});
  idle.onchange = async () => { settings.idle_minutes = +idle.value; await api.save_settings({idle_minutes: +idle.value}); };
  const quality = el("input", {type: "number", min: "1", max: "100", value: String(settings.last.output_quality || 92), style: "width:80px"});
  quality.onchange = async () => {
    const n = Math.min(100, Math.max(1, Math.round(+quality.value) || 92)); quality.value = n;
    settings.last.output_quality = n; await api.save_settings({output_quality: n});
  };
  const assistantBox = el("div", {class: "stack"}, el("span", {class: "desc"}, "Loading…"));
  const hfBox = el("div", {class: "stack"}, el("span", {class: "desc"}, "Loading…"));
  hfSettings(hfBox);
  assistantSettings(assistantBox);
  const recentLimit = el("input", {type: "number", min: "1", max: "500", value: String(settings.last.recent_limit || 60), style: "width:80px"});
  recentLimit.onchange = async () => {
    const n = Math.min(500, Math.max(1, Math.round(+recentLimit.value) || 60)); recentLimit.value = n;
    settings.last.recent_limit = n; await api.save_settings({recent_limit: n}); refreshGallery(true);
  };
  // Advanced: folder and GPU-sharing settings are read when the app starts
  const restartNote = el("div", {class: "desc", style: "color:var(--warn)"}, "");
  const saveAdv = async d => { const r = await api.save_settings(d); settings = await api.get_settings();
    if (r.restart) restartNote.textContent = "Saved. Restart aiVibe Studio for this to take effect."; };
  const extra = el("textarea", {rows: "2", placeholder: "One folder per line"}, (settings.extra_model_folders || []).join("\n"));
  extra.onchange = () => saveAdv({extra_model_folders: extra.value.split("\n")});
  const advInput = (key, placeholder) => {
    const i = el("input", {value: settings[key] || "", placeholder});
    i.onchange = () => saveAdv({[key]: i.value});
    return i;
  };
  const updates = el("input", {type: "checkbox", style: "width:auto"});
  updates.checked = settings.update_check !== false;
  updates.onchange = () => saveAdv({update_check: updates.checked});
  const checks = Object.entries(PANELS).map(([id, label]) => {
    const c = el("input", {type: "checkbox", style: "width:auto"});
    c.checked = !hidden.has(id);
    c.onchange = async () => {
      c.checked ? hidden.delete(id) : hidden.add(id);
      settings.last.hidden_panels = [...hidden];
      applyPanels();
      await api.save_settings({hidden_panels: settings.last.hidden_panels});
    };
    return el("label", {}, c, ` ${label}`);
  });
  modal("Settings", el("div", {class: "stack"},
    el("h3", {}, "Output folder"),
    el("div", {class: "row"}, folder, el("button", {onclick: async () => {
      const r = await api.choose_folder();
      if (r.ok) { settings = await api.get_settings(); refreshGallery(true); settingsModal(); }
    }}, "Choose folder")),
    warn,
    el("label", {}, batchSub, " Put each batch's pictures in their own folder (named after the batch file)"),
    el("div", {class: "row"}, el("span", {}, "Save pictures as:"), el("div", {class: "layoutbtns"},
      ...[["png", "PNG (exact, larger files)"], ["jpg", "JPG (small, any app opens it)"], ["webp", "WebP (smallest)"]]
        .map(([f, label]) => el("button", {class: (settings.last.output_format || "png") === f ? "primary" : "", onclick: async () => {
          settings.last.output_format = f; await api.save_settings({output_format: f}); settingsModal();
        }}, label)))),
    (settings.last.output_format || "png") !== "png" ? el("label", {}, "Quality ", quality, " (1 to 100; 92 looks the same as PNG to the eye)") : "",
    el("p", {class: "desc"}, "Applies to new pictures, fixes and upscales. The recipe (.json) is saved beside each one either way."),
    el("h3", {}, "Layout"),
    el("p", {class: "desc"}, "Drag any panel by its title (or the picture area by ⋮⋮ at its top left) into any column. Empty columns show a drop area."),
    el("div", {class: "row"}, el("span", {}, "Columns:"), el("div", {class: "layoutbtns"},
      ...[2, 3, 4, 5].map(n => el("button", {class: currentLayout().length === n ? "primary" : "", onclick: () => { setColumns(n); settingsModal(); }}, String(n))))),
    el("div", {class: "row"}, el("span", {}, "Ready layouts:"), el("div", {class: "layoutbtns"},
      ...Object.keys(PRESETS).map(name => el("button", {onclick: () => { applyLayoutPreset(name); settingsModal(); }}, name)))),
    el("h3", {}, "Recent pictures"),
    el("div", {class: "layoutbtns"}, ...[["bottom", "Along the bottom"], ["top", "Along the top"], ["column", "As a panel (drag it to any column)"], ["off", "Hidden"]]
      .map(([m, label]) => el("button", {class: (settings.last.recent || "bottom") === m ? "primary" : "", onclick: async () => {
        settings.last.recent = m; placeRecent(m); await api.save_settings({recent: m}); settingsModal();
      }}, label))),
    el("div", {class: "row"}, el("span", {}, "Show:"),
      el("span", {class: "desc"}, settings.last.recent_folder ? `${settings.last.recent_folder} (this folder only, no subfolders)` : "The newest pictures from the output folder"),
      el("button", {onclick: async () => {
        const r = await api.choose_recent_folder();
        if (r.ok) { settings = await api.get_settings(); refreshGallery(true); settingsModal(); }
      }}, "Choose folder"),
      settings.last.recent_folder ? el("button", {onclick: async () => {
        settings.last.recent_folder = ""; await api.save_settings({recent_folder: ""}); refreshGallery(true); settingsModal();
      }}, "Back to newest") : ""),
    el("label", {}, "Show at most ", recentLimit, " pictures"),
    el("h3", {}, "Assistant"),
    assistantBox,
    el("h3", {}, "Hugging Face"),
    hfBox,
    el("h3", {}, "Workspace panels"),
    el("p", {class: "desc"}, "Untick a panel to hide it from the workspace. Prompt, preview and Generate always stay."),
    el("div", {class: "settings-grid"}, ...checks),
    el("h3", {}, "GPU"),
    el("label", {}, "Free the GPU after ", idle, " minutes without work"),
    el("h3", {}, "Advanced"),
    el("label", {}, updates, " Check for a new version when the app starts"),
    el("div", {}, "Extra model folders (read-only; diffusers model folders another app already downloaded)"), extra,
    el("div", {}, "Model download folder (blank = inside the aiVibe Studio data folder)"), advInput("hf_cache_dir", "Default"),
    el("div", {}, "Pause while another GPU program runs (optional; leave blank if you do not know you need it)"),
    advInput("gpu_marker_file", "Busy marker file the other program reads"),
    advInput("gpu_jobs_folder", "That program's jobs folder"),
    restartNote,
    el("h3", {}, "Problem reports and logs"),
    el("p", {class: "desc"}, "Report a problem from the ⚑ button on the left, the busy window, or Ctrl+Shift+B. Logs are kept 14 days."),
    reportsList,
    el("div", {class: "row"}, el("button", {onclick: () => api.open_reports()}, "Open reports folder"),
      el("button", {onclick: () => api.open_logs()}, "Open logs folder")),
    el("p", {class: "desc"}, `aiVibe Studio ${settings.version}. Everything runs on this PC. Made in collaboration with AI agents (Claude and Gemini) and a human. MIT license.`)));
}
function applyTheme() {
  document.documentElement.dataset.theme = settings.theme;
  $("themeLabel").textContent = settings.theme === "dark" ? "Light" : "Dark";
}

// ---------- subjects + reference ----------
const KIND_CLASSES = {person: ["man", "woman", "person"], pet: ["dog", "cat"], car: ["car"], thing: ["object"]};
function likenessAvailability() {
  const m = modelById[$("model").value];
  const ok = m && m.family === "sdxl";
  document.querySelector('[data-panel="subjects"]').classList.toggle("disabled", !ok);
  $("subjectNote").textContent = ok ? (subjects.some(s => s.trained) ? "Click to add (up to 3) and use the name in the prompt. Solo picture? Add \"alone\". Click a face in the preview to fix it. More in Guide." : "No trained subjects yet.")
    : "Subjects and reference images need an SDXL model (RealVisXL or Juggernaut).";
}
async function loadSubjects() {
  subjects = await api.subjects();
  subjectPicks = subjectPicks.filter(p => subjects.some(s => s.id === p.id && s.trained));
  drawSubjects();
  if (!$("batchMaker").hidden) bmRender();
}
function drawSubjects() {
  $("subjectChips").replaceChildren(...subjects.map(s => {
    const on = subjectPicks.some(p => p.id === s.id);
    const c = el("span", {class: "chip" + (on ? " on" : "") + (s.trained ? "" : " untrained"),
      title: s.trained ? `${s.name} (${s.class_word}), trained ${s.trained_at || ""}` : `${s.name}: not trained yet (${s.photos} photos)`}, s.name);
    c.onclick = () => {
      if (!s.trained) return subjectModal(s.id);
      if (on) subjectPicks = subjectPicks.filter(p => p.id !== s.id);
      else if (subjectPicks.length >= 3) return log("Up to 3 subjects in one picture.", "em");
      else subjectPicks.push({id: s.id, weight: s.default_weight || 1.0});
      drawSubjects();
    };
    return c;
  }));
  $("subjectPicks").replaceChildren(...subjectPicks.map(p => {
    const s = subjects.find(x => x.id === p.id);
    const out = el("output", {}, p.weight.toFixed(2));
    const r = el("input", {type: "range", min: "0", max: "1.3", step: "0.05", value: String(p.weight), title: "Likeness: low = loosely inspired, high = clearly them"});
    r.oninput = () => { p.weight = +r.value; out.textContent = p.weight.toFixed(2); };
    return el("div", {class: "pick"}, el("span", {}, s ? s.name : p.id), r, out);
  }));
  likenessAvailability();
}
function newSubjectModal() {
  const kind = el("select", {id: "nsKind"}, ...Object.keys(KIND_CLASSES).map(k => el("option", {value: k}, k)));
  const cls = el("select", {id: "nsClass"});
  const fillCls = () => cls.replaceChildren(...KIND_CLASSES[kind.value].map(c => el("option", {value: c}, c)));
  kind.onchange = fillCls; fillCls();
  const name = el("input", {placeholder: "Name, e.g. Alex, Mom, Rex, the truck"});
  modal("New subject", el("div", {class: "stack"},
    el("p", {}, "A subject is a person, pet, car or thing the app learns from your photos. It is saved for good and appears in the Subjects panel."),
    el("div", {class: "row"}, name, kind, cls),
    el("p", {class: "desc"}, "Next you pick a folder of 12-30 photos (5 is the minimum): only that subject in each, sharp, different days, angles and light. Faces clearly visible for people."),
    el("button", {class: "primary", onclick: async () => {
      const r = await api.new_subject(name.value, kind.value, cls.value);
      if (r.error) return alert(r.error);
      await loadSubjects(); subjectModal(r.id);
    }}, "Create and pick photos")));
}
async function subjectModal(id) {
  await loadSubjects();
  const s = subjects.find(x => x.id === id);
  if (!s) return;
  const status = el("p", {}, s.trained ? `Trained ${s.trained_at}. ${s.photos} photos.` : `${s.photos} photos, not trained yet.`);
  const photoGrid = el("div", {class: "photogrid"});
  const drawPhotos = async () => {
    const photos = await api.subject_photos(s.id);
    photoGrid.replaceChildren(...photos.map(p => {
      const cell = el("div", {class: "pg", title: p.name});
      cell.style.backgroundImage = `url("${p.thumb}")`;
      cell.append(el("button", {title: "Remove this photo", onclick: async () => {
        const r = await api.remove_subject_photo(s.id, p.name);
        if (r.error) return alert(r.error);
        status.textContent = `${r.total} photos${s.trained ? " (retrain to apply the change)" : ""}.`;
        drawPhotos(); loadSubjects();
      }}, "✕"));
      return cell;
    }));
  };
  drawPhotos();
  const report = el("div", {class: "desc"});
  modal(`Subject: ${s.name}`, el("div", {class: "stack"}, status,
    el("div", {class: "row"},
      el("button", {onclick: async () => {
        const r = await api.import_photos(s.id, null);
        if (r.cancelled) return;
        if (r.error) return alert(r.error);
        report.replaceChildren(el("div", {}, `Added ${r.added} photos, ${r.total} in total.`),
          ...(r.warning ? [el("div", {style: "color:var(--warn)"}, r.warning)] : []),
          ...r.skipped.map(x => el("div", {}, `skipped ${x}`)));
        await loadSubjects(); status.textContent = `${r.total} photos, ${s.trained ? "trained (retrain to use the new photos)" : "not trained yet"}.`;
      }}, "Add photos from a folder"),
      el("button", {class: "primary", onclick: async () => {
        const hw = settings.hardware || {};
        if (!hw.train_ok && !confirm(`Training needs a card with 12 GB of video memory; this PC has ${hw.vram_gb ? hw.vram_gb + " GB" : "no supported card"}. It may fail or take many hours. Train anyway?`)) return;
        const r = await api.train(s.id);
        if (r.error) return alert(r.error);
        if (r.needs_download) return askDownload(r.needs_download);
        log(`Training ${s.name}: ${r.steps} steps, about ${r.minutes} minutes. The GPU is busy until it finishes.`, "b");
        $("modal").hidden = true;
      }}, s.trained ? "Retrain" : "Train")),
    report,
    el("h3", {}, "Training photos"),
    el("p", {class: "desc"}, "The cropped copies used for training. Click ✕ to remove one (your original photo is not touched), then Retrain."),
    photoGrid,
    el("div", {class: "row", style: "margin-top:12px"},
      el("button", {onclick: async () => { const n = prompt("New name:", s.name); if (n) { await api.rename_subject(s.id, n); await loadSubjects(); subjectModal(s.id); } }}, "Rename"),
      el("button", {onclick: async () => { const w = prompt("Default likeness for this subject (0 to 1.3):", s.default_weight); if (w) { const r = await api.set_default_weight(s.id, w); if (r.error) alert(r.error); await loadSubjects(); subjectModal(s.id); } }}, `Default likeness: ${s.default_weight}`),
      el("button", {onclick: async () => { if (confirm(`Delete ${s.name}, its photos copy and its training? Your original photos are not touched.`)) { await api.delete_subject(s.id); await loadSubjects(); $("modal").hidden = true; } }}, "Delete"))));
}
function manageSubjectsModal() {
  modal("Subjects", el("table", {}, el("tr", {}, el("th", {}, "Name"), el("th", {}, "Kind"), el("th", {}, "Photos"), el("th", {}, "Status"), el("th", {}, "")),
    ...subjects.map(s => el("tr", {}, el("td", {}, s.name), el("td", {}, `${s.kind} (${s.class_word})`), el("td", {}, s.photos),
      el("td", {}, s.trained ? `trained ${s.trained_at}` : "not trained"), el("td", {}, el("button", {onclick: () => subjectModal(s.id)}, "Open"))))));
}
function previewClick(ev) {
  // click-to-fix: map the click to image pixels (object-fit: contain leaves bars on two sides)
  const img = $("previewImg"), box = img.getBoundingClientRect();
  const scale = Math.min(box.width / img.naturalWidth, box.height / img.naturalHeight);
  const w = img.naturalWidth * scale, h = img.naturalHeight * scale;
  const ox = box.left + (box.width - w) / 2, oy = box.top + (box.height - h) / 2;
  const x = Math.round((ev.clientX - ox) / scale), y = Math.round((ev.clientY - oy) / scale);
  if (x < 0 || y < 0 || x > img.naturalWidth || y > img.naturalHeight || !selectedPath) return;
  const trained = subjects.filter(s => s.trained);
  if (!trained.length) return;
  const menu = $("moreMenu");
  menu.replaceChildren(el("div", {class: "desc", style: "padding:4px 8px"}, "Redraw this spot as:"),
    ...trained.map(s => el("button", {onclick: async e => {
      e.stopPropagation(); menu.hidden = true;
      const r = await api.fix_face(selectedPath, x, y, s.id);
      if (r.error) log(r.error, "em"); else log(`Fixing that spot as ${s.name}…`);
    }}, s.name)));
  menu.style.top = `${ev.clientY}px`; menu.style.left = `${ev.clientX}px`; menu.style.right = "auto";
  menu.hidden = false;
  ev.stopPropagation();
}
function drawReference() {
  const t = $("refThumb");
  if (reference) { t.style.backgroundImage = `url("${reference.thumb}")`; t.textContent = ""; t.title = reference.name; }
  else { t.style.backgroundImage = ""; t.textContent = "none"; t.title = ""; }
}

// ---------- updates: offered, never installed without a click ----------
async function checkUpdate(manual) {
  const r = await api.check_update(manual);
  if (r.error) return manual && alert(r.message);
  if (!r.update) { if (manual) alert(`aiVibe Studio ${settings.version} is the newest version.`); return; }
  log(`aiVibe Studio ${r.version} is available. Click Updates on the left to get it.`, "b");
  if (!manual) return;
  if (!confirm(`aiVibe Studio ${r.version} is available (you have ${settings.version}).\n\n${(r.notes || "").slice(0, 800)}\n\nDownload and install it now? The app closes while the installer runs; your pictures, subjects and settings are kept.`)) return;
  log("Downloading the new version…");
  const g = await api.get_update();
  if (g.error) { $("statusLine").textContent = ""; log(g.error, "em"); }
}
// ---------- generate / batch ----------
// ---------- model downloads: a window with a bar; what was asked for starts by itself when they finish ----------
let dlWatch = {pending: new Map(), then: null};       // model id -> {name, size_gb}
function showDownloads() {
  const items = [...dlWatch.pending.values()];
  if (!items.length) { $("dl").hidden = true; return; }
  $("dlTitle").textContent = `Downloading ${items.map(i => i.name).join(", ")}`;
  $("dlNext").textContent = dlWatch.then ? "Your picture starts by itself when the download finishes." :
    "You can keep working; press the button again when the download finishes.";
  $("dl").hidden = false;
}
function downloadEvent(e) {
  const it = dlWatch.pending.get(e.model);
  if (!it) return false;
  if (e.kind === "download") {
    $("dlBar").style.width = `${e.percent ?? 0}%`;
    $("dlCount").textContent = `${e.gb} of about ${it.size_gb} GB${e.percent != null ? ` (${e.percent}%)` : ""}`;
    return true;
  }
  dlWatch.pending.delete(e.model);
  if (e.kind !== "download_done") dlWatch.then = null;      // failed or cancelled: do not start anything
  if (!dlWatch.pending.size) {
    $("dl").hidden = true; $("dlBar").style.width = "0%"; $("dlCount").textContent = "";
    const then = dlWatch.then; dlWatch.then = null;
    if (then) setTimeout(then, 300);
  } else showDownloads();
  return false;
}
async function askDownload(need, then = null) {
  const list = (Array.isArray(need) ? need : [need]);
  const infos = await Promise.all(list.map(n => api.download_info(n.id)));
  const fitText = {ok: "", tight: "\n   May be slow on this graphics card.", too_big: "\n   Probably too big for this graphics card."};
  const text = infos.map(i => `${i.name}: about ${i.size_gb} GB (${i.free_gb} GB free)\n   License: ${i.license_text}`
    + (i.gated && !i.has_token ? "\n   Needs a free Hugging Face account and token (Settings > Hugging Face)." : "")
    + (fitText[i.fit] || "") + (i.card_url ? `\n   Model page: ${i.card_url}` : "")).join("\n\n");
  if (!confirm(`This needs a model that is not downloaded yet:\n\n${text}\n\nBy downloading you accept the model's license. Download now? You can keep working; the log shows progress.`)) return;
  let started = 0;
  for (const i of infos) {
    const r = await api.download(i.id);
    if (r.error === "needs_token" || r.error === "no_space") { alert(r.message); continue; }
    if (r.error && r.error !== "already downloading") { log(`${i.name}: ${r.error}`, "em"); continue; }
    dlWatch.pending.set(i.id, {name: i.name, size_gb: i.size_gb}); started++;
    log(`Downloading ${i.name}…`);
  }
  if (started) { dlWatch.then = then; showDownloads(); }
}
async function generate() {
  if (!$("busy").hidden) return;      // Ctrl+Enter while something runs
  const r = await api.generate(current());
  if (r.error) return log(r.error, "em");
  if (r.needs_download) return askDownload(r.needs_download, generate);
  remember();
}
async function loadBatch() {
  const r = await api.load_batch(null);
  if (r.cancelled) return;
  if (r.problems) { r.problems.forEach(p => log(p, "em")); return modal("Batch file problems", el("div", {}, el("p", {}, "Nothing was started. Fix these and load it again:"), el("ul", {}, ...r.problems.map(p => el("li", {}, p))))); }
  if (r.needs_download) return askDownload(r.needs_download);
  log(`Batch started: ${r.count} prompts`, "b");
}

// ---------- batch maker UI: build, open, modify, save and run batches ----------
const BM_STORAGE_KEY = "aivibe_bm_draft_v1";
const STD_UPSCALES = new Set(["", "2x", "4x", "1080p", "4k"]);
let bm = {
  path: null,
  filename: null,
  name: "my-batch",
  dirty: false,
  jsonMode: false,
  expandAll: false,
  openOpts: new Set(),
  problems: [],
  defRefThumb: null,
  batch: {defaults: {}, items: [{prompt: ""}]},
};

function bmLoadDraft() {
  try {
    const raw = localStorage.getItem(BM_STORAGE_KEY);
    if (!raw) return;
    const saved = JSON.parse(raw);
    if (saved && isObj(saved.batch)) {
      bm.batch = tidyBatch(saved.batch);
      if (!bm.batch.items.length) bm.batch.items = [{prompt: ""}];
      bm.name = saved.name || "my-batch";
      bm.path = saved.path || null;
      bm.filename = saved.filename || null;
      bm.dirty = !!saved.dirty;
    }
  } catch (e) { /* ignore corrupt draft */ }
}

function bmSaveDraft() {
  try {
    localStorage.setItem(BM_STORAGE_KEY, JSON.stringify({
      name: bm.name, path: bm.path, filename: bm.filename, dirty: bm.dirty, batch: bm.batch,
    }));
  } catch (e) { /* ignore quota errors */ }
}

function bmNonEmptyCount() {
  return (bm.batch.items || []).filter(it => isObj(it) && String(it.prompt || "").trim()).length;
}

function bmTotalPictures() {
  const d = bm.batch.defaults || {};
  return (bm.batch.items || []).reduce((sum, it) => {
    const c = +((it && it.count) ?? d.count ?? 1) || 1;
    return sum + Math.max(1, c);
  }, 0);
}

function bmUpdateHeader() {
  const n = (bm.batch.items || []).length;
  const active = bmNonEmptyCount();
  const pics = bmTotalPictures();
  $("navBatchLabel").textContent = active > 0 ? `Batch (${active})` : "Batch";
  $("bmSummary").textContent = `${n} item${n === 1 ? "" : "s"} · ${pics} picture${pics === 1 ? "" : "s"}`;
  const info = $("bmFileInfo");
  if (bm.path) {
    info.textContent = `File: ${bm.filename || bm.path.split(/[\\/]/).pop()}${bm.dirty ? " (modified)" : " (saved)"}`;
    info.title = bm.path;
  } else {
    info.textContent = bm.dirty ? "Unsaved batch (modified)" : "New unsaved batch";
    info.title = "";
  }
  const canRun = active > 0 && bm.batch.items.every(it => String(it.prompt || "").trim());
  $("bmRun").disabled = $("bmSaveRun").disabled = !canRun && !bm.jsonMode;
}

function bmTouch(clearProblems = true) {
  bm.dirty = true;
  if (clearProblems && bm.problems.length) {
    bm.problems = [];
    $("bmProblems").replaceChildren();
    $("bmStatus").textContent = "Modified. Click Check batch or Run batch to validate.";
  }
  bmSaveDraft();
  bmUpdateHeader();
}

function showView(mode) {
  $("modal").hidden = true;
  $("work").hidden = mode !== "create";
  $("videoMaker").hidden = mode !== "video";
  $("batchMaker").hidden = mode !== "batch";
  $("navCreate").classList.toggle("on", mode === "create");
  $("navVideo").classList.toggle("on", mode === "video");
  $("navBatch").classList.toggle("on", mode === "batch");
  if (mode === "batch") bmRender();
  if (mode === "video") vidRender();
}

function bmFillNegPresets() {
  const sel = $("bmDefNegPreset");
  if (!sel || !settings) return;
  sel.replaceChildren(el("option", {value: ""}, "Presets…"),
    ...(settings.last.negatives || []).map((n, i) => el("option", {value: String(i)}, n.name)));
  sel.onchange = () => {
    if (sel.value !== "") {
      const txt = settings.last.negatives[+sel.value].text;
      $("bmDefNegative").value = txt;
      if (txt.trim()) bm.batch.defaults.negative = txt; else delete bm.batch.defaults.negative;
      bmTouch();
    }
    sel.value = "";
  };
}

function bmFindSubject(nameOrId) {
  const q = String(nameOrId || "").trim().toLowerCase();
  return subjects.find(s => s.id.toLowerCase() === q || s.name.toLowerCase() === q) || null;
}

function bmNormalizeSubList(raw) {
  return asList(raw).map(r => {
    const rawName = isObj(r) ? (r.name || r.id || "") : String(r || "");
    const s = bmFindSubject(rawName);
    const hasW = isObj(r) && r.weight != null && r.weight !== "";
    const w = hasW ? +r.weight : (s ? (s.default_weight ?? 1.0) : 1.0);
    return {name: s ? s.name : String(rawName).trim(), weight: isFinite(w) ? w : 1.0, explicit: hasW};
  }).filter(x => x.name);
}

function bmSerializeSubList(entries) {
  return entries.map(e => e.explicit ? {name: e.name, weight: +(+e.weight).toFixed(2)} : e.name);
}

function bmRenderSubjectEditor(chipsBox, picksBox, rawList, onChange) {
  const entries = bmNormalizeSubList(rawList);
  const knownNames = new Set(subjects.map(s => s.name.toLowerCase()));
  const extraEntries = entries.filter(e => !knownNames.has(e.name.toLowerCase()));
  const allChoices = [
    ...subjects.map(s => ({name: s.name, trained: s.trained, kind: s.kind, class_word: s.class_word, default_weight: s.default_weight || 1.0})),
    ...extraEntries.map(e => ({name: e.name, trained: true, kind: "in file", class_word: "subject", default_weight: e.weight || 1.0})),
  ];
  chipsBox.replaceChildren(...(allChoices.length ? allChoices.map(s => {
    const idx = entries.findIndex(e => e.name.toLowerCase() === s.name.toLowerCase());
    const on = idx >= 0;
    const chip = el("span", {
      class: "chip" + (on ? " on" : "") + (s.trained ? "" : " untrained"),
      title: s.trained ? `${s.name} (${s.class_word})` : `${s.name}: not trained yet`,
    }, s.name);
    chip.onclick = () => {
      if (on) {
        entries.splice(idx, 1);
      } else {
        if (entries.length >= 3) return log("Up to 3 subjects in one picture.", "em");
        entries.push({name: s.name, weight: s.default_weight || 1.0, explicit: false});
      }
      onChange(entries.length ? bmSerializeSubList(entries) : null);
    };
    return chip;
  }) : [el("span", {class: "desc"}, "No trained subjects yet.")]));

  picksBox.replaceChildren(...entries.map((e, idx) => {
    const num = el("input", {type: "number", min: "0", max: "1.3", step: "0.05", value: (+e.weight).toFixed(2), title: "Likeness weight (0 to 1.3)"});
    const range = el("input", {type: "range", min: "0", max: "1.3", step: "0.05", value: String(e.weight), title: "Drag to adjust likeness strength"});
    const updateW = val => {
      const w = Math.min(1.3, Math.max(0, +val || 0));
      e.weight = w; e.explicit = true;
      range.value = String(w); num.value = w.toFixed(2);
      onChange(bmSerializeSubList(entries));
    };
    range.oninput = () => { num.value = (+range.value).toFixed(2); };
    range.onchange = () => updateW(range.value);
    num.onchange = () => updateW(num.value);
    const rm = el("button", {class: "tiny ghost", title: "Remove this subject", onclick: () => {
      entries.splice(idx, 1);
      onChange(entries.length ? bmSerializeSubList(entries) : null);
    }}, "×");
    return el("div", {class: "bm-sub-row"}, el("span", {}, e.name), range, num, rm);
  }));
}

function bmRenderDefaults() {
  const d = bm.batch.defaults || (bm.batch.defaults = {});
  const imgs = imageModels();
  const curModel = modelById[settings.last.model] || imgs[0];
  const modelSel = $("bmDefModel");
  modelSel.replaceChildren(
    el("option", {value: ""}, `(studio model: ${curModel ? curModel.name : "default"})`),
    ...imgs.map(m => el("option", {value: m.id}, m.name + (m.downloaded ? "" : " (download)"))),
    ...(d.model && !modelById[d.model] ? [el("option", {value: String(d.model)}, `${d.model} (unknown)`)] : [])
  );
  modelSel.value = d.model || "";

  const effModel = modelById[d.model] || curModel;
  const md = effModel ? effModel.defaults : {width: 1024, height: 1024, steps: 30, guidance: 6.0};
  $("bmDefWidth").placeholder = String(md.width || 1024);
  $("bmDefHeight").placeholder = String(md.height || 1024);
  $("bmDefSteps").placeholder = String(md.steps ?? 30);
  $("bmDefGuidance").placeholder = String(md.guidance ?? 6.0);
  $("bmDefWidth").value = d.width ?? "";
  $("bmDefHeight").value = d.height ?? "";
  $("bmDefCount").value = d.count ?? "";
  $("bmDefSteps").value = d.steps ?? "";
  $("bmDefGuidance").value = d.guidance ?? "";
  $("bmDefSeed").value = d.seed != null && d.seed !== -1 ? d.seed : "";

  const upVal = d.upscale == null ? "" : String(d.upscale).toLowerCase();
  if (STD_UPSCALES.has(upVal)) {
    $("bmDefUpscale").value = upVal;
    $("bmDefUpPxWrap").hidden = true;
    $("bmDefUpPx").value = "";
  } else {
    $("bmDefUpscale").value = "custom";
    $("bmDefUpPxWrap").hidden = false;
    $("bmDefUpPx").value = String(d.upscale).replace(/px$/i, "");
  }

  $("bmDefNegative").value = d.negative || "";
  $("bmDefFaceFix").checked = d.face_fix !== false;
  $("bmDefRefPath").value = d.reference || "";
  $("bmDefRefStrength").value = d.reference_strength ?? 0.5;
  const rt = $("bmDefRefThumb");
  if (d.reference && bm.defRefThumb && bm.defRefThumb.path === d.reference) {
    rt.style.backgroundImage = `url("${bm.defRefThumb.thumb}")`; rt.textContent = ""; rt.title = d.reference;
  } else {
    rt.style.backgroundImage = ""; rt.textContent = d.reference ? "file" : "none"; rt.title = d.reference || "";
  }

  bmRenderSubjectEditor($("bmDefSubChips"), $("bmDefSubPicks"), d.subjects, next => {
    if (next && next.length) d.subjects = next; else delete d.subjects;
    bmTouch(); bmRenderDefaults(); bmRenderItems();
  });
}

function bmItemOverrideCount(it) {
  let n = 0;
  for (const k of ["negative", "subjects", "steps", "guidance", "face_fix", "reference"]) {
    if (it[k] !== undefined && it[k] !== null && it[k] !== "" && !(Array.isArray(it[k]) && !it[k].length)) n++;
  }
  return n;
}

function bmRenderItems() {
  const d = bm.batch.defaults || {};
  const items = bm.batch.items || (bm.batch.items = []);
  const box = $("bmItems");
  if (!items.length) {
    box.replaceChildren(el("div", {class: "bm-empty"}, "No items in this batch yet. Click + Add item or + Paste lines above."));
    return;
  }
  const imgs = imageModels();
  const defModel = modelById[d.model] || modelById[settings.last.model] || imgs[0];
  const defModelLabel = defModel ? defModel.name : "default";
  const defW = d.width || (defModel && defModel.defaults.width) || 1024;
  const defH = d.height || (defModel && defModel.defaults.height) || 1024;

  box.replaceChildren(...items.map((it, i) => {
    const ovrCount = bmItemOverrideCount(it);
    const isOpen = bm.expandAll || bm.openOpts.has(i) || ovrCount > 0;

    // Prompt textarea
    const promptIn = el("textarea", {class: "bitem-prompt", rows: "2", placeholder: "Describe this image…"}, it.prompt || "");
    promptIn.oninput = () => { it.prompt = promptIn.value; bmTouch(); };

    // Name input
    const nameIn = el("input", {class: "bitem-name", placeholder: "name (optional)", value: it.name || "", title: "Short label used in the saved file name"});
    nameIn.onchange = () => { const v = nameIn.value.trim(); if (v) it.name = v; else delete it.name; bmTouch(); };

    // Model select
    const modelSel = el("select", {class: "bitem-model", title: "Model for this item (or use shared default)"},
      el("option", {value: ""}, `(default: ${defModelLabel})`),
      ...imgs.map(m => el("option", {value: m.id, selected: it.model === m.id ? "selected" : null}, m.name + (m.downloaded ? "" : " (dl)"))),
      ...(it.model && !modelById[it.model] ? [el("option", {value: String(it.model), selected: "selected"}, `${it.model} (unknown)`)] : [])
    );
    modelSel.onchange = () => {
      if (modelSel.value) it.model = modelSel.value; else delete it.model;
      bmTouch(); bmRenderItems();
    };

    // Size input (WxH)
    const sizeVal = it.width && it.height ? `${it.width}x${it.height}` : "";
    const sizeIn = el("input", {class: "bitem-size", list: "bmSizePresets", value: sizeVal, placeholder: `${defW}x${defH}`, title: "Size as WidthxHeight (e.g. 1344x768), or blank for default"});
    sizeIn.onchange = () => {
      const m = sizeIn.value.trim().match(/^(\d+)\s*[x×,]\s*(\d+)$/i);
      if (m) { it.width = +m[1]; it.height = +m[2]; sizeIn.value = `${it.width}x${it.height}`; }
      else { delete it.width; delete it.height; sizeIn.value = ""; }
      bmTouch();
    };

    // Count input
    const countIn = el("input", {class: "bitem-count", type: "number", min: "1", max: "8", value: it.count ?? "", placeholder: `×${d.count || 1}`, title: "Images for this item (1 to 8)"});
    countIn.onchange = () => {
      if (countIn.value !== "") it.count = +countIn.value; else delete it.count;
      bmTouch();
    };

    // Upscale select
    const rawUp = it.upscale === undefined ? "__def__" : String(it.upscale).toLowerCase();
    const isCustomUp = rawUp !== "__def__" && !STD_UPSCALES.has(rawUp);
    const upSel = el("select", {class: "bitem-up", title: "Upscale for this item"},
      el("option", {value: "__def__"}, `(default${d.upscale ? `: ${d.upscale}` : ": off"})`),
      ...(d.upscale ? [el("option", {value: ""}, "Off (no upscale)")] : []),
      el("option", {value: "2x"}, "Upscale 2x"),
      el("option", {value: "4x"}, "Upscale 4x"),
      el("option", {value: "1080p"}, "Upscale 1080p"),
      el("option", {value: "4k"}, "Upscale 4K"),
      ...(isCustomUp ? [el("option", {value: String(it.upscale)}, `Upscale ${it.upscale}`)] : []),
      el("option", {value: "__custom__"}, "Custom px…")
    );
    upSel.value = isCustomUp ? String(it.upscale) : rawUp;
    upSel.onchange = () => {
      if (upSel.value === "__def__") {
        delete it.upscale;
      } else if (upSel.value === "__custom__") {
        const px = prompt("Target long edge in pixels (512 to 8192):", isCustomUp ? String(it.upscale).replace(/px$/i, "") : "2400");
        if (px && /^\d+/.test(px.trim())) it.upscale = `${parseInt(px.trim(), 10)}px`;
      } else {
        it.upscale = upSel.value;
      }
      bmTouch(); bmRenderItems();
    };

    // Seed input
    const seedIn = el("input", {class: "bitem-seed", type: "number", value: it.seed ?? "", placeholder: d.seed != null && d.seed !== -1 ? `seed ${d.seed}` : "random seed", title: "Seed for this item (-1 for random, or blank for default)"});
    seedIn.onchange = () => {
      if (seedIn.value.trim() !== "") it.seed = +seedIn.value; else delete it.seed;
      bmTouch();
    };

    // Options drawer toggle
    const moreBox = el("div", {class: "bitem-more", hidden: !isOpen});
    const optBtn = el("button", {
      class: "ghost bitem-opt-btn" + (ovrCount > 0 ? " has-ovr" : ""),
      title: "Show or hide per-item overrides (Negative prompt, Subjects, Steps, Guidance, Face fix, Reference)",
      onclick: () => {
        if (!moreBox.hidden) { moreBox.hidden = true; bm.openOpts.delete(i); }
        else { moreBox.hidden = false; bm.openOpts.add(i); }
        optBtn.textContent = `${ovrCount ? `Options (${ovrCount})` : "Options"} ${moreBox.hidden ? "▾" : "▴"}`;
      },
    }, `${ovrCount ? `Options (${ovrCount})` : "Options"} ${isOpen ? "▴" : "▾"}`);

    // ---- Advanced Overrides Drawer Content ----
    const effItemModel = modelById[it.model || d.model] || defModel;
    const imd = effItemModel ? effItemModel.defaults : {steps: 30, guidance: 6.0};
    const stepsIn = el("input", {type: "number", min: "1", max: "150", value: it.steps ?? "", placeholder: String(d.steps ?? imd.steps ?? 30)});
    stepsIn.onchange = () => { if (stepsIn.value !== "") it.steps = +stepsIn.value; else delete it.steps; bmTouch(); bmRenderItems(); };

    const guidIn = el("input", {type: "number", min: "0", max: "30", step: "0.5", value: it.guidance ?? "", placeholder: String(d.guidance ?? imd.guidance ?? 6.0)});
    guidIn.onchange = () => { if (guidIn.value !== "") it.guidance = +guidIn.value; else delete it.guidance; bmTouch(); bmRenderItems(); };

    const ffSel = el("select", {},
      el("option", {value: ""}, `(default: ${d.face_fix === false ? "Off" : "On"})`),
      el("option", {value: "true"}, "On"),
      el("option", {value: "false"}, "Off")
    );
    ffSel.value = it.face_fix === undefined ? "" : String(!!it.face_fix);
    ffSel.onchange = () => {
      if (ffSel.value === "") delete it.face_fix; else it.face_fix = ffSel.value === "true";
      bmTouch(); bmRenderItems();
    };

    const refIn = el("input", {value: it.reference || "", placeholder: d.reference || "None", style: "flex:1"});
    refIn.onchange = () => {
      const v = refIn.value.trim();
      if (v) it.reference = v; else { delete it.reference; delete it.reference_strength; }
      bmTouch(); bmRenderItems();
    };
    const refStrIn = el("input", {type: "number", min: "0", max: "1", step: "0.05", value: it.reference_strength ?? (d.reference_strength ?? 0.5), style: "width:68px", title: "Reference strength (0 to 1)"});
    refStrIn.onchange = () => { it.reference_strength = +refStrIn.value; bmTouch(); };

    const negArea = el("textarea", {rows: "2", placeholder: d.negative ? `(default: ${d.negative})` : "(no negative prompt)"}, it.negative ?? "");
    negArea.oninput = () => {
      if (negArea.value.trim()) it.negative = negArea.value; else delete it.negative;
      bmTouch();
    };

    const subChips = el("div", {class: "chips"});
    const subPicks = el("div", {class: "stack"});
    const hasItemSubs = it.subjects !== undefined;
    bmRenderSubjectEditor(subChips, subPicks, hasItemSubs ? it.subjects : [], next => {
      if (next && next.length) it.subjects = next; else delete it.subjects;
      bmTouch(); bmRenderItems();
    });

    moreBox.append(
      el("div", {class: "bitem-grid"},
        el("div", {}, el("label", {}, "Steps override"), stepsIn),
        el("div", {}, el("label", {}, "Guidance override"), guidIn),
        el("div", {}, el("label", {}, "Face fix"), ffSel),
        el("div", {}, el("label", {}, "Reference strength"), refStrIn)
      ),
      el("div", {class: "row"},
        el("label", {style: "white-space:nowrap"}, "Reference image:"),
        refIn,
        el("button", {class: "tiny", onclick: async () => {
          const r = await api.pick_reference();
          if (r.error) return log(r.error, "em");
          if (!r.cancelled) { it.reference = r.path; bmTouch(); bmRenderItems(); }
        }}, "Pick…"),
        it.reference ? el("button", {class: "tiny ghost", onclick: () => { delete it.reference; delete it.reference_strength; bmTouch(); bmRenderItems(); }}, "Clear") : ""
      ),
      el("div", {},
        el("div", {class: "row"},
          el("label", {style: "flex:1"}, "Negative prompt override"),
          d.negative ? el("button", {class: "tiny ghost", onclick: () => { it.negative = d.negative; bmTouch(); bmRenderItems(); }}, "Copy shared default") : "",
          it.negative !== undefined ? el("button", {class: "tiny ghost", onclick: () => { delete it.negative; bmTouch(); bmRenderItems(); }}, "Use shared default") : ""
        ),
        negArea
      ),
      el("div", {},
        el("div", {class: "row"},
          el("label", {style: "flex:1"}, hasItemSubs ? "Subjects override for this item:" : `Subjects: using shared default (${asList(d.subjects).map(x => isObj(x) ? `${x.name || x.id} (${x.weight ?? 1})` : x).join(", ") || "none"})`),
          (d.subjects && d.subjects.length) ? el("button", {class: "tiny ghost", onclick: () => {
            it.subjects = JSON.parse(JSON.stringify(d.subjects));
            bmTouch(); bmRenderItems();
          }}, "Copy shared subjects") : "",
          hasItemSubs ? el("button", {class: "tiny ghost", onclick: () => { delete it.subjects; bmTouch(); bmRenderItems(); }}, "Use shared default") : ""
        ),
        subChips,
        subPicks
      )
    );

    const moveUp = el("button", {class: "tiny ghost", title: "Move item up", disabled: i === 0 ? "disabled" : null, onclick: () => bmMoveItem(i, -1)}, "↑");
    const moveDown = el("button", {class: "tiny ghost", title: "Move item down", disabled: i === items.length - 1 ? "disabled" : null, onclick: () => bmMoveItem(i, 1)}, "↓");
    const dupBtn = el("button", {class: "ghost", title: "Duplicate this item", onclick: () => bmDuplicateItem(i)}, "⧉");
    const delBtn = el("button", {class: "ghost", title: "Remove this item", onclick: () => bmRemoveItem(i)}, "×");

    return el("div", {class: "bitem"},
      el("div", {class: "bitem-main"},
        el("div", {class: "bitem-side"},
          el("span", {class: "bitem-idx"}, `#${i + 1}`),
          el("div", {class: "bitem-move"}, moveUp, moveDown)
        ),
        el("div", {class: "bitem-body"},
          promptIn,
          el("div", {class: "bitem-bar"}, nameIn, modelSel, sizeIn, countIn, upSel, seedIn, optBtn)
        ),
        el("div", {class: "bitem-acts"}, dupBtn, delBtn)
      ),
      moreBox
    );
  }));
}

function bmRender() {
  $("bmName").value = bm.name || "my-batch";
  $("bmProblems").replaceChildren(...(bm.problems || []).map(p => el("li", {}, p)));
  $("bmJsonText").hidden = !bm.jsonMode;
  $("bmItems").hidden = bm.jsonMode;
  $("bmJsonToggle").textContent = bm.jsonMode ? "Edit as form" : "Edit as JSON";
  $("bmExpandAll").textContent = bm.expandAll ? "Options ▴" : "Options ▾";
  bmFillNegPresets();
  bmRenderDefaults();
  if (bm.jsonMode) {
    $("bmJsonText").value = JSON.stringify(bm.batch, null, 2);
  } else {
    bmRenderItems();
  }
  bmUpdateHeader();
}

function bmSyncFromJson() {
  if (!bm.jsonMode) return true;
  try {
    const parsed = JSON.parse($("bmJsonText").value);
    bm.batch = tidyBatch(Array.isArray(parsed) ? {items: parsed} : parsed);
    return true;
  } catch (e) {
    $("bmStatus").textContent = `Invalid JSON: ${e.message}`;
    return false;
  }
}

function bmMoveItem(index, delta) {
  const items = bm.batch.items;
  const target = index + delta;
  if (target < 0 || target >= items.length) return;
  [items[index], items[target]] = [items[target], items[index]];
  bmTouch(); bmRenderItems();
}

function bmDuplicateItem(index) {
  const copy = JSON.parse(JSON.stringify(bm.batch.items[index] || {prompt: ""}));
  if (copy.name) copy.name = `${copy.name}-copy`;
  bm.batch.items.splice(index + 1, 0, copy);
  bmTouch(); bmRenderItems();
}

function bmRemoveItem(index) {
  bm.batch.items.splice(index, 1);
  if (!bm.batch.items.length) bm.batch.items.push({prompt: ""});
  bmTouch(); bmRenderItems();
}

function bmAddItem() {
  if (!bmSyncFromJson()) return;
  bm.jsonMode = false;
  bm.batch.items.push({prompt: ""});
  bmTouch(); bmRender();
  const areas = document.querySelectorAll("#bmItems .bitem-prompt");
  if (areas.length) areas[areas.length - 1].focus();
}

function bmCopyFromStudio() {
  const c = current();
  const d = bm.batch.defaults;
  d.model = c.model;
  d.width = c.width;
  d.height = c.height;
  d.steps = c.steps;
  d.guidance = c.guidance;
  if (c.count > 1) d.count = c.count; else delete d.count;
  if (c.seed >= 0) d.seed = c.seed; else delete d.seed;
  if (c.negative.trim()) d.negative = c.negative.trim(); else delete d.negative;
  if (c.subjects && c.subjects.length) {
    d.subjects = c.subjects.map(p => {
      const s = subjects.find(x => x.id === p.id);
      const nm = s ? s.name : p.id;
      return s && p.weight === (s.default_weight || 1.0) ? nm : {name: nm, weight: p.weight};
    });
  } else delete d.subjects;
  if (c.reference) {
    d.reference = c.reference;
    d.reference_strength = c.reference_strength;
    bm.defRefThumb = reference ? {path: reference.path, thumb: reference.thumb} : null;
  } else {
    delete d.reference; delete d.reference_strength; bm.defRefThumb = null;
  }
  if (c.face_fix === false) d.face_fix = false; else delete d.face_fix;
  bmTouch(); bmRender();
  $("bmStatus").textContent = "Copied current settings from the Create tab into Shared defaults.";
}

function bmAddFromCreate() {
  const c = current();
  if (!c.prompt.trim()) {
    showView("batch");
    return;
  }
  if (!bmSyncFromJson()) {
    showView("batch");
    return;
  }
  const d = bm.batch.defaults;
  const isFresh = Object.keys(d).length === 0 && bm.batch.items.every(x => !String(x.prompt || "").trim());
  if (isFresh) {
    bmCopyFromStudio();
    delete bm.batch.defaults.seed;
    const item = {prompt: c.prompt.trim()};
    if (c.name.trim()) item.name = c.name.trim();
    if (c.seed >= 0) item.seed = c.seed;
    bm.batch.items = [item];
  } else {
    const item = {prompt: c.prompt.trim()};
    if (c.name.trim()) item.name = c.name.trim();
    const defModelId = d.model || settings.last.model;
    const defModel = modelById[defModelId] || models[0];
    const md = defModel ? defModel.defaults : {width: 1024, height: 1024, steps: 30, guidance: 6.0};
    if (c.model !== defModelId) item.model = c.model;
    if (c.width !== (d.width || md.width) || c.height !== (d.height || md.height)) {
      item.width = c.width; item.height = c.height;
    }
    if (c.count !== (d.count || 1)) item.count = c.count;
    if (c.seed >= 0 && c.seed !== d.seed) item.seed = c.seed;
    if (c.negative.trim() !== (d.negative || "").trim()) item.negative = c.negative.trim();
    if (bm.batch.items.length === 1 && !String(bm.batch.items[0].prompt || "").trim()) {
      bm.batch.items[0] = item;
    } else {
      bm.batch.items.push(item);
    }
  }
  bmTouch();
  const n = bmNonEmptyCount();
  log(`Added to batch (${n} item${n === 1 ? "" : "s"}). Click Batch on the left to edit, save or run.`, "b");
}

function bmNewBatch() {
  if (bm.dirty && (bmNonEmptyCount() > 0 || Object.keys(bm.batch.defaults || {}).length > 0)) {
    if (!confirm("Start a new blank batch? Unsaved changes in the current batch will be cleared.")) return;
  }
  bm.path = null;
  bm.filename = null;
  bm.name = "my-batch";
  bm.dirty = false;
  bm.jsonMode = false;
  bm.openOpts.clear();
  bm.problems = [];
  bm.defRefThumb = null;
  bm.batch = {defaults: {}, items: [{prompt: ""}]};
  bmSaveDraft();
  bmRender();
  $("bmStatus").textContent = "Started a new blank batch.";
}

async function bmOpenFile() {
  const r = await api.open_batch_file(null);
  if (r.cancelled) return;
  if (r.error) {
    $("bmStatus").textContent = r.error;
    return log(r.error, "em");
  }
  bm.path = r.path;
  bm.filename = r.filename;
  bm.name = r.name || "batch";
  bm.dirty = false;
  bm.jsonMode = false;
  bm.openOpts.clear();
  bm.problems = r.problems || [];
  bm.defRefThumb = null;
  bm.batch = tidyBatch(r.batch);
  if (!bm.batch.items.length) bm.batch.items = [{prompt: ""}];
  bmSaveDraft();
  bmRender();
  if (bm.problems.length) {
    $("bmStatus").textContent = `Loaded ${r.filename} (${bm.batch.items.length} items) with ${bm.problems.length} issue(s) to fix.`;
  } else {
    $("bmStatus").textContent = `Loaded ${r.filename} (${bm.batch.items.length} items).`;
  }
  log(`Opened batch file ${r.filename} (${bm.batch.items.length} items) in Batch Maker.`);
}

async function bmSaveFile(saveAs = false) {
  if (!bmSyncFromJson()) return false;
  const sname = (bm.filename && !saveAs) ? bm.filename : `${(bm.name || "my-batch").replace(/\.(json|txt)$/i, "")}.json`;
  const r = await api.save_batch_file(bm.batch, saveAs ? null : bm.path, sname);
  if (r.cancelled) return false;
  if (r.error) {
    $("bmStatus").textContent = r.error;
    log(r.error, "em");
    return false;
  }
  bm.path = r.path;
  bm.filename = r.filename;
  bm.name = r.name || bm.name;
  bm.dirty = false;
  bm.problems = r.problems || [];
  bmSaveDraft();
  bmRender();
  $("bmStatus").textContent = bm.problems.length
    ? `Saved to ${r.filename} (note: ${bm.problems.length} issue(s) to fix before running).`
    : `Saved to ${r.filename}.`;
  log(`Saved batch to ${r.filename}.`, "b");
  return true;
}

async function bmCheck() {
  if (!bmSyncFromJson()) return;
  const r = await api.check_batch(bm.batch);
  bm.problems = r.problems || [];
  $("bmProblems").replaceChildren(...bm.problems.map(p => el("li", {}, p)));
  if (r.ok) {
    $("bmStatus").textContent = `Batch is valid: ${r.count} item${r.count === 1 ? "" : "s"}, ${r.pictures} picture${r.pictures === 1 ? "" : "s"} ready to run.`;
  } else {
    $("bmStatus").textContent = `Found ${bm.problems.length} problem${bm.problems.length === 1 ? "" : "s"}. Fix the items listed above before running.`;
  }
}

async function bmRun(saveFirst = false) {
  if (!bmSyncFromJson()) return;
  if (saveFirst) {
    const ok = await bmSaveFile(false);
    if (!ok) return;
  }
  $("bmRun").disabled = $("bmSaveRun").disabled = true;
  let r;
  try {
    r = await api.run_batch(bm.batch, bm.filename || bm.name || "batch");
  } catch (e) {
    r = {error: `Something went wrong: ${e && e.message || e}`};
  } finally {
    bmUpdateHeader();
  }
  if (r.error) {
    $("bmStatus").textContent = r.error;
    return log(r.error, "em");
  }
  if (r.problems) {
    bm.problems = r.problems;
    $("bmProblems").replaceChildren(...bm.problems.map(p => el("li", {}, p)));
    $("bmStatus").textContent = "Nothing was started. Fix the problems listed above first.";
    return;
  }
  if (r.needs_download) {
    return askDownload(r.needs_download, () => bmRun(false));
  }
  $("bmStatus").textContent = `Batch started: ${r.count} items (${r.pictures} pictures).`;
  log(`Batch started (${r.name}): ${r.count} items, ${r.pictures} pictures`, "b");
}

function reviewToBatchMaker() {
  if (!readReviewText()) return;
  bm.path = null;
  bm.filename = null;
  bm.name = ($("assistantRequest").value.trim().toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "").slice(0, 32)) || "assistant-batch";
  bm.dirty = true;
  bm.jsonMode = false;
  bm.openOpts.clear();
  bm.problems = review.problems || [];
  bm.batch = JSON.parse(JSON.stringify(review.batch));
  $("assistantReview").hidden = true;
  bmSaveDraft();
  showView("batch");
  $("bmStatus").textContent = `Loaded ${bm.batch.items.length} items from Assistant. You can edit, save to a file or run.`;
}

function bindBatchMaker() {
  bmLoadDraft();
  bmUpdateHeader();
  const d = () => bm.batch.defaults;
  $("bmName").onchange = () => { bm.name = $("bmName").value.trim() || "my-batch"; bmSaveDraft(); };
  $("bmNew").onclick = bmNewBatch;
  $("bmOpen").onclick = bmOpenFile;
  $("bmQuickRun").onclick = loadBatch;
  $("bmSave").onclick = () => bmSaveFile(false);
  $("bmSaveAs").onclick = () => bmSaveFile(true);
  $("bmFromStudio").onclick = bmCopyFromStudio;
  $("bmClearDefaults").onclick = () => {
    bm.batch.defaults = {}; bm.defRefThumb = null;
    bmTouch(); bmRender();
    $("bmStatus").textContent = "Cleared shared defaults.";
  };

  $("bmDefModel").onchange = () => {
    if ($("bmDefModel").value) d().model = $("bmDefModel").value; else delete d().model;
    bmTouch(); bmRenderDefaults(); bmRenderItems();
  };
  document.querySelectorAll("#bmDefAspects button").forEach(b => b.onclick = () => {
    const [w, h] = b.dataset.wh.split("x").map(Number);
    d().width = w; d().height = h;
    bmTouch(); bmRenderDefaults(); bmRenderItems();
  });
  const bindDefNum = (id, key, isFloat = false) => {
    $(id).onchange = () => {
      const v = $(id).value.trim();
      if (v !== "") d()[key] = isFloat ? parseFloat(v) : parseInt(v, 10);
      else delete d()[key];
      bmTouch(); bmRenderItems();
    };
  };
  bindDefNum("bmDefWidth", "width");
  bindDefNum("bmDefHeight", "height");
  bindDefNum("bmDefCount", "count");
  bindDefNum("bmDefSteps", "steps");
  bindDefNum("bmDefGuidance", "guidance", true);
  bindDefNum("bmDefSeed", "seed");

  $("bmDefUpscale").onchange = () => {
    const v = $("bmDefUpscale").value;
    if (v === "custom") {
      $("bmDefUpPxWrap").hidden = false;
      const px = parseInt($("bmDefUpPx").value, 10) || 2400;
      $("bmDefUpPx").value = String(px);
      d().upscale = `${px}px`;
    } else if (v) {
      $("bmDefUpPxWrap").hidden = true;
      d().upscale = v;
    } else {
      $("bmDefUpPxWrap").hidden = true;
      delete d().upscale;
    }
    bmTouch(); bmRenderItems();
  };
  $("bmDefUpPx").onchange = () => {
    const px = parseInt($("bmDefUpPx").value, 10);
    if (px) d().upscale = `${px}px`; else delete d().upscale;
    bmTouch(); bmRenderItems();
  };
  $("bmDefNegative").oninput = () => {
    const v = $("bmDefNegative").value;
    if (v.trim()) d().negative = v; else delete d().negative;
    bmTouch();
  };
  $("bmDefFaceFix").onchange = () => {
    if ($("bmDefFaceFix").checked) delete d().face_fix; else d().face_fix = false;
    bmTouch(); bmRenderItems();
  };
  $("bmDefRefPath").onchange = () => {
    const v = $("bmDefRefPath").value.trim();
    if (v) d().reference = v; else { delete d().reference; delete d().reference_strength; bm.defRefThumb = null; }
    bmTouch(); bmRenderDefaults(); bmRenderItems();
  };
  $("bmDefPickRef").onclick = async () => {
    const r = await api.pick_reference();
    if (r.error) return log(r.error, "em");
    if (!r.cancelled) {
      d().reference = r.path;
      d().reference_strength = +$("bmDefRefStrength").value || 0.5;
      bm.defRefThumb = {path: r.path, thumb: r.thumb};
      bmTouch(); bmRenderDefaults(); bmRenderItems();
    }
  };
  $("bmDefClearRef").onclick = () => {
    delete d().reference; delete d().reference_strength; bm.defRefThumb = null;
    bmTouch(); bmRenderDefaults(); bmRenderItems();
  };
  $("bmDefRefStrength").onchange = () => {
    if (d().reference) d().reference_strength = +$("bmDefRefStrength").value;
    bmTouch(); bmRenderItems();
  };

  $("bmAddItem").onclick = bmAddItem;
  $("bmBulkToggle").onclick = () => {
    $("bmBulkBox").hidden = !$("bmBulkBox").hidden;
    if (!$("bmBulkBox").hidden) $("bmBulkText").focus();
  };
  $("bmBulkCancel").onclick = () => { $("bmBulkBox").hidden = true; };
  $("bmBulkAdd").onclick = () => {
    if (!bmSyncFromJson()) return;
    const lines = $("bmBulkText").value.split(/\r?\n/).map(l => l.trim()).filter(l => l && !l.startsWith("#"));
    if (!lines.length) return;
    if (bm.batch.items.length === 1 && !String(bm.batch.items[0].prompt || "").trim()) {
      bm.batch.items = lines.map(p => ({prompt: p}));
    } else {
      bm.batch.items.push(...lines.map(p => ({prompt: p})));
    }
    $("bmBulkText").value = "";
    $("bmBulkBox").hidden = true;
    bm.jsonMode = false;
    bmTouch(); bmRender();
    $("bmStatus").textContent = `Added ${lines.length} prompt line${lines.length === 1 ? "" : "s"}.`;
  };
  $("bmExpandAll").onclick = () => {
    bm.expandAll = !bm.expandAll;
    if (!bm.expandAll) bm.openOpts.clear();
    bmRender();
  };
  $("bmJsonToggle").onclick = () => {
    if (bm.jsonMode && !bmSyncFromJson()) return;
    bm.jsonMode = !bm.jsonMode;
    bmRender();
  };
  $("bmJsonText").oninput = () => bmTouch();
  $("bmCheck").onclick = bmCheck;
  $("bmSaveRun").onclick = () => bmRun(true);
  $("bmRun").onclick = () => bmRun(false);
}

// ---------- video maker (text-to-video and image-to-video) ----------
let vidImage = null;   // {path, thumb, name}
const VID_ASPECTS = {"16:9": [16, 9], "9:16": [9, 16], "1:1": [1, 1]};

function vidFillNegPresets() {
  const sel = $("vidNegPreset");
  if (!sel || !settings) return;
  sel.replaceChildren(el("option", {value: ""}, "Presets…"),
    ...(settings.last.negatives || []).map((n, i) => el("option", {value: String(i)}, n.name)));
  sel.onchange = () => {
    if (sel.value !== "") {
      $("vidNegative").value = settings.last.negatives[+sel.value].text;
    }
    sel.value = "";
  };
}

function drawVidImage() {
  const t = $("vidImgThumb");
  if (!t) return;
  if (vidImage) {
    t.style.backgroundImage = `url("${vidImage.thumb}")`;
    t.textContent = "";
    t.title = vidImage.name || vidImage.path;
  } else {
    t.style.backgroundImage = "";
    t.textContent = "none";
    t.title = "";
  }
}

function vidTokenCount() {
  const t = ($("vidPrompt").value.match(/[A-Za-z0-9']+|[^\sA-Za-z0-9]/g) || []);
  $("vidPromptCount").textContent = `~${t.length} tokens`;
}

function vidSummary() {
  const m = modelById[$("vidModel").value];
  const w = +$("vidWidth").value || 704, h = +$("vidHeight").value || 480;
  const f = +$("vidFrames").value || 49, fps = Math.max(1, +$("vidFps").value || 24);
  const secs = (f / fps).toFixed(1);
  $("vidDurationInfo").textContent = `${f} frames at ${fps} fps (~${secs} s clip)`;
  $("vidSummary").textContent = `${w} × ${h} · ${f} frames @ ${fps} fps (~${secs} s) · ${m ? m.name : ""}`;
}

function showVidModel() {
  const m = modelById[$("vidModel") && $("vidModel").value];
  if (!m) return;
  $("vidModelDesc").textContent = m.description || "";
  $("vidModelMode").textContent = m.t2v && m.i2v ? "Text & Image → Video" : m.i2v ? "Image → Video only" : "Text → Video";
  $("vidModelLicense").textContent = {commercial: "commercial use OK", conditions: "free with conditions", nc: "personal use only"}[m.license] || m.license;
  $("vidModelLicense").className = "tag" + (m.license === "nc" ? " nc" : "");
  const fit = {ok: "", tight: " · may be slow on this card", too_big: ` · needs ${m.min_vram_gb} GB of video memory`}[m.fit] || "";
  $("vidModelState").textContent = (m.downloaded ? "on disk" : `${m.size_gb} GB to download`) + fit;
  $("vidModelState").style.color = m.fit === "too_big" ? "var(--bad)" : m.fit === "tight" ? "var(--warn)" : "";
  $("vidBtnGuide").disabled = !m.guide;
  $("vidPickImg").disabled = $("vidUseCurrent").disabled = !m.i2v;
  if (m.i2v && m.t2v) {
    $("vidImgNote").textContent = "Optional starting frame. Leave empty for Text-to-Video, or pick an image to animate it.";
  } else if (m.i2v) {
    $("vidImgNote").textContent = "Required starting frame for this model (Image-to-Video).";
  } else {
    $("vidImgNote").textContent = "This model generates video from text only (no starting image).";
  }
  vidSummary();
  vidTokenCount();
}

function vidRecommended() {
  const m = modelById[$("vidModel").value];
  if (!m) return;
  const d = m.defaults || {};
  $("vidWidth").value = d.width || 704;
  $("vidHeight").value = d.height || 480;
  $("vidFrames").value = d.frames || 49;
  $("vidFps").value = d.fps || 24;
  setSlider("vidSteps", d.steps ?? 30);
  setSlider("vidGuidance", d.guidance ?? 3.0);
  document.querySelectorAll("#vidAspect button").forEach(b => b.classList.toggle("on", b.dataset.va === "16:9"));
  vidSummary();
}

function vidRender() {
  vidFillNegPresets();
  drawVidImage();
  showVidModel();
}

async function animateImage(path) {
  if (!path) return;
  const r = await api.image(path);
  if (r.error || r.media === "video") return log(r.error || "Pick a still image to animate.", "em");
  vidImage = {path, thumb: r.data, name: r.name};
  const curVid = modelById[$("vidModel").value];
  if (!curVid || !curVid.i2v) {
    const i2vModel = videoModels().find(m => m.i2v && m.downloaded) || videoModels().find(m => m.i2v);
    if (i2vModel) {
      $("vidModel").value = i2vModel.id;
      vidRecommended();
    }
  }
  if (r.recipe && r.recipe.prompt && !$("vidPrompt").value.trim()) {
    $("vidPrompt").value = r.recipe.prompt;
  }
  drawVidImage();
  showView("video");
  log(`Loaded ${r.name} as starting image in Video tab.`);
}

async function generateVideo() {
  const payload = {
    model: $("vidModel").value,
    prompt: $("vidPrompt").value,
    negative: $("vidNegative").value,
    name: $("vidName").value,
    width: +$("vidWidth").value,
    height: +$("vidHeight").value,
    frames: +$("vidFrames").value,
    fps: +$("vidFps").value,
    steps: +$("vidSteps").value,
    guidance: +$("vidGuidance").value,
    seed: $("vidRandomSeed").checked ? -1 : +$("vidSeed").value,
    image: vidImage && modelById[$("vidModel").value] && modelById[$("vidModel").value].i2v ? vidImage.path : null,
  };
  api.save_settings({vid_model: payload.model});
  const r = await api.generate_video(payload);
  if (r.error) return log(r.error, "em");
  if (r.needs_download) return askDownload(r.needs_download, generateVideo);
  if (r.width) {
    $("vidWidth").value = r.width;
    $("vidHeight").value = r.height;
    $("vidFrames").value = r.frames;
    $("vidFps").value = r.fps;
    vidSummary();
  }
  log(`Generating video (${r.width}×${r.height}, ${r.frames} frames @ ${r.fps} fps)…`, "b");
}

function bindVideoMaker() {
  vidRecommended();
  $("vidModel").onchange = () => { showVidModel(); vidRecommended(); api.save_settings({vid_model: $("vidModel").value}); };
  $("vidBtnGuide").onclick = () => api.model_guide($("vidModel").value);
  $("vidBtnModels").onclick = modelsModal;
  $("vidPickImg").onclick = async () => {
    const r = await api.pick_reference();
    if (r.error) return log(r.error, "em");
    if (!r.cancelled) { vidImage = r; drawVidImage(); }
  };
  $("vidUseCurrent").onclick = () => {
    if (!selectedPath) return log("Select a picture in the Recent strip first.", "em");
    animateImage(selectedPath);
  };
  $("vidClearImg").onclick = () => { vidImage = null; drawVidImage(); };
  document.querySelectorAll("#vidAspect button").forEach(b => b.onclick = () => {
    document.querySelectorAll("#vidAspect button").forEach(x => x.classList.toggle("on", x === b));
    const m = modelById[$("vidModel").value];
    const [aw, ah] = VID_ASPECTS[b.dataset.va];
    const longEdge = m ? Math.max(m.defaults.width || 704, m.defaults.height || 480) : 704;
    const step = m && m.family === "svd" ? 64 : m && m.family === "ltx" ? 32 : 16;
    const scale = longEdge / Math.max(aw, ah);
    $("vidWidth").value = Math.max(256, Math.round(aw * scale / step) * step);
    $("vidHeight").value = Math.max(256, Math.round(ah * scale / step) * step);
    vidSummary();
  });
  ["vidWidth", "vidHeight", "vidFrames", "vidFps"].forEach(id => $(id).onchange = vidSummary);
  $("vidRecommended").onclick = vidRecommended;
  ["vidSteps", "vidGuidance"].forEach(id => $(id).oninput = () => setSlider(id, $(id).value));
  $("vidRandomSeed").onchange = () => { $("vidSeed").disabled = $("vidRandomSeed").checked; };
  $("vidPrompt").oninput = vidTokenCount;
  $("vidClearPrompt").onclick = () => { $("vidPrompt").value = ""; vidTokenCount(); };
  $("vidClearNeg").onclick = () => { $("vidNegative").value = ""; };
  $("btnGenerateVideo").onclick = generateVideo;
}

// ---------- assistant (phase 4): a cloud AI writes a batch, the user checks it, Go runs it ----------
async function hfSettings(box) {
  const st = await api.hf_token_status();
  const input = el("input", {type: "password", placeholder: "hf_… (a Read token)", autocomplete: "off"});
  box.replaceChildren(
    el("p", {class: "desc"}, "Some models (Stable Diffusion 3.5, FLUX.1 schnell) need a free Hugging Face account: make one at huggingface.co, accept the model's terms on its page, then make a Read token under Settings, Access Tokens and paste it here. It stays on this PC, encrypted, and is only sent to Hugging Face."),
    el("div", {class: "row"}, input, el("button", {onclick: async () => {
      const r = await api.set_hf_token(input.value);
      if (r.error) return alert(r.error);
      hfSettings(box);
    }}, "Save token"), st.key ? el("button", {onclick: async () => { await api.remove_hf_token(); hfSettings(box); }}, "Remove token") : ""),
    el("div", {class: "desc"}, st.key ? `Token ${st.key}.` : "No token saved."));
}
async function assistantSettings(box) {
  const st = await api.assistant_status();
  const cur = st.providers.find(p => p.id === st.provider);
  const keyIn = el("input", {type: "password", placeholder: cur.key ? cur.key : `Paste your ${cur.label} API key`, autocomplete: "off", style: "flex:1"});
  const keyMsg = el("span", {class: "desc"}, cur.key || "No key saved yet.");
  const list = el("datalist", {id: "assistantModels"});
  const modelIn = el("input", {value: cur.model, list: "assistantModels", style: "flex:1"});
  modelIn.onchange = async () => { await api.save_settings({[`assistant_model_${cur.id}`]: modelIn.value.trim()}); settings = await api.get_settings(); };
  const check = el("span", {class: "desc"});
  box.replaceChildren(
    el("div", {class: "layoutbtns"}, ...st.providers.map(p => el("button", {class: p.id === st.provider ? "primary" : "", onclick: async () => {
      await api.save_settings({assistant_provider: p.id}); settings = await api.get_settings(); assistantSettings(box);
    }}, p.label + (p.key ? " ✓" : "")))),
    el("div", {class: "row"}, keyIn, el("button", {onclick: async () => {
      const r = await api.set_assistant_key(cur.id, keyIn.value);
      keyIn.value = "";
      if (r.error) return (keyMsg.textContent = r.error);
      assistantSettings(box);
    }}, "Save key"), cur.key ? el("button", {onclick: async () => { await api.remove_assistant_key(cur.id); assistantSettings(box); }}, "Remove key") : ""),
    keyMsg,
    el("div", {class: "row"}, el("span", {}, "Model:"), modelIn, list, el("button", {onclick: async () => {
      check.textContent = "Checking…";
      const r = await api.assistant_models(cur.id);
      if (r.error) return (check.textContent = r.error);
      list.replaceChildren(...r.models.map(m => el("option", {value: m})));
      check.textContent = `Key works: ${r.models.length} models. Click the Model box to pick one.`;
    }}, "Check key")),
    check,
    el("p", {class: "desc"}, "Only your typed request goes to the provider, never pictures or subject photos. API use is billed by the provider, separately from any subscription."));
}

let review = null;      // {request, batch, problems, provider, label, model, tokens, text: false}
const asList = x => (x == null || x === "") ? [] : [].concat(x);       // the AI may send "Alex" for ["Alex"]
const isObj = x => x && typeof x === "object" && !Array.isArray(x);
function tidyBatch(b) {
  // odd lines become empty ones (Go stays off until they get a prompt); the server lists what was wrong
  b = isObj(b) ? b : {};
  return {defaults: isObj(b.defaults) ? b.defaults : {}, items: (Array.isArray(b.items) ? b.items : []).map(it => isObj(it) ? it : {prompt: ""})};
}
async function assistantWrite() {
  const request = $("assistantRequest").value.trim();
  const st = await api.assistant_status();
  const p = st.providers.find(x => x.id === st.provider);
  const t0 = Date.now();
  $("btnAssistant").disabled = true;
  const tick = setInterval(() => { $("assistantStatus").textContent = `Asking ${p.label} (${st.model})… ${Math.round((Date.now() - t0) / 1000)} s`; }, 500);
  let r;
  try { r = await api.assistant_write(request); }
  catch (e) { r = {error: `Something went wrong: ${e && e.message || e}`}; }
  finally { clearInterval(tick); $("btnAssistant").disabled = false; }
  if (r.error) { $("assistantStatus").textContent = r.error; return log(r.error, "em"); }
  r.batch = tidyBatch(r.batch);
  $("assistantStatus").textContent = `${p.label} wrote ${r.batch.items.length} lines in ${r.seconds} s.`;
  review = {request, ...r, text: false};
  renderReview();
  $("assistantReview").hidden = false;
}
function reviewPictures() {
  const d = review.batch.defaults || {};
  return review.batch.items.reduce((n, it) => n + (+((it && it.count) ?? d.count ?? 1) || 1), 0);
}
function setField(it, key, value) {
  if (value === "" || value == null) delete it[key]; else it[key] = value;
  reviewEdited();
}
function reviewEdited() {
  // an edit may fix (or cause) a problem: the list goes, and Go checks the whole batch again
  if (review.problems.length) { review.problems = []; $("reviewProblems").replaceChildren(); $("reviewStatus").textContent = "Changed: checked again when you press Go."; }
  reviewGoState();
}
function reviewGoState() {
  const items = review.batch.items;
  $("reviewGo").disabled = review.problems.length > 0 || !items.length || items.some(it => !String(it.prompt || "").trim());
  $("reviewInfo").textContent = `${items.length} lines, ${reviewPictures()} pictures · written by ${review.label} (${review.model}) · ${review.tokens} tokens`;
}
function renderReview() {
  const d = review.batch.defaults || {};
  $("reviewProblems").replaceChildren(...review.problems.map(p => el("li", {}, p)));
  const shared = Object.entries(d).map(([k, v]) => `${k} ${Array.isArray(v) ? v.join(", ") : v}`).join(", ");
  $("reviewDefaults").textContent = shared ? `For every line unless the line says otherwise: ${shared}` : "";
  $("reviewText").hidden = !review.text; $("reviewRows").hidden = review.text;
  $("reviewTextToggle").textContent = review.text ? "Edit as lines" : "Edit as text";
  if (review.text) { $("reviewText").value = JSON.stringify(review.batch, null, 2); return reviewGoState(); }
  const fallback = modelById[d.model] || modelById[settings.last.model];
  const usable = imageModels().filter(m => m.downloaded);
  const modelOpts = (it) => el("select", {onchange: e => setField(it, "model", e.target.value)},
    el("option", {value: ""}, `(${fallback ? fallback.name : "current model"})`),
    ...usable.map(m => el("option", {value: m.id, selected: it.model === m.id ? "selected" : null}, m.name)),
    ...(it.model && !usable.some(m => m.id === it.model)      // show what the AI picked, even when it cannot run
      ? [el("option", {value: String(it.model), selected: "selected"}, `${it.model} (not downloaded)`)] : []));
  const head = el("div", {class: "rline head"}, el("span"), el("span", {}, "Prompt"), el("span", {}, "Model"), el("span", {}, "Size"),
    el("span", {}, "Count"), el("span", {}, "Upscale"), el("span", {}, "Subjects"), el("span"));
  $("reviewRows").replaceChildren(head, ...review.batch.items.map((it, i) => {
    const prompt = el("textarea", {}, it.prompt || "");
    prompt.oninput = () => setField(it, "prompt", prompt.value);
    const size = el("input", {value: it.width && it.height ? `${it.width}x${it.height}` : "", placeholder: d.width ? `${d.width}x${d.height}` : "model's"});
    size.onchange = () => {
      const m = size.value.trim().match(/^(\d+)\s*[x×]\s*(\d+)$/i);
      if (m) { it.width = +m[1]; it.height = +m[2]; } else { delete it.width; delete it.height; size.value = ""; }
      reviewEdited();
    };
    const count = el("input", {type: "number", min: "1", max: "8", value: it.count ?? "", placeholder: String(d.count ?? 1)});
    count.onchange = () => setField(it, "count", count.value ? +count.value : "");
    const up = el("input", {value: it.upscale ?? "", placeholder: d.upscale || "none"});
    up.onchange = () => setField(it, "upscale", up.value.trim());
    const subs = el("input", {value: asList(it.subjects).map(x => isObj(x) ? x.name : String(x)).join(", "), placeholder: asList(d.subjects).map(x => isObj(x) ? x.name : String(x)).join(", ") || "none"});
    subs.onchange = () => { const names = subs.value.split(",").map(x => x.trim()).filter(Boolean); setField(it, "subjects", names.length ? names : ""); };
    return el("div", {class: "rline"}, el("span", {class: "n"}, String(i + 1)), prompt, modelOpts(it), size, count, up, subs,
      el("button", {class: "ghost", title: "Remove this line", onclick: () => { review.batch.items.splice(i, 1); reviewEdited(); renderReview(); }}, "×"));
  }));
  reviewGoState();
}
function readReviewText() {
  if (!review.text) return true;
  try {
    const b = JSON.parse($("reviewText").value);
    review.batch = tidyBatch(Array.isArray(b) ? {items: b} : b);
    return true;
  } catch (e) { $("reviewStatus").textContent = `That text is not valid JSON: ${e.message}`; return false; }
}
async function reviewRevise() {
  const instruction = $("reviewInstruction").value.trim();
  if (!readReviewText()) return;
  $("reviewRevise").disabled = $("reviewGo").disabled = true; $("reviewStatus").textContent = `Asking ${review.label}…`;
  let r;
  try { r = await api.assistant_revise(review.request, review.batch, instruction); }
  catch (e) { r = {error: `Something went wrong: ${e && e.message || e}`}; }
  finally { $("reviewRevise").disabled = false; reviewGoState(); }
  if (r.error) return ($("reviewStatus").textContent = r.error);
  review = {...review, ...r, batch: tidyBatch(r.batch), text: review.text};
  $("reviewInstruction").value = ""; $("reviewStatus").textContent = `Revised in ${r.seconds} s.`;
  renderReview();
}
async function reviewGo() {
  if (!readReviewText()) return;
  $("reviewGo").disabled = $("reviewRevise").disabled = true;       // one click, one batch
  let r;
  try { r = await api.assistant_run(review.request, review.batch, review.provider, review.model); }
  catch (e) { r = {error: `Something went wrong: ${e && e.message || e}`}; }
  finally { $("reviewRevise").disabled = false; reviewGoState(); }
  if (r.problems) { review.problems = r.problems; renderReview(); $("reviewStatus").textContent = "Nothing was started. Fix these first."; return; }
  if (r.error) return ($("reviewStatus").textContent = r.error);
  if (r.needs_download) { $("assistantReview").hidden = true; return askDownload(r.needs_download); }
  $("assistantReview").hidden = true;
  log(`Running assistant batch: ${reviewPictures()} pictures`, "b");
}

// ---------- events, queue, gallery ----------
async function poll() {
  try {
    const evs = await api.events(lastSeq);
    for (const e of evs) {
      lastSeq = e.seq;
      if (e.kind === "image") { log(`saved ${e.path.split(/[\\/]/).pop()}`, "b"); show(e.path); refreshGallery(true); }
      else if (e.kind === "video") { log(`saved video ${e.path.split(/[\\/]/).pop()}`, "b"); show(e.path); refreshGallery(true); }
      else if (e.kind === "item_failed") log(`item ${e.index} failed: ${e.error}`, "em");
      else if (e.kind === "model_failed") log(`item ${e.index} skipped: ${e.error}`, "em");
      else if (e.kind === "model_loading") log(`Loading ${modelById[e.model] ? modelById[e.model].name : e.model}…`);
      else if (e.kind === "done") log("Done.");
      else if (e.kind === "cancelled") log("Cancelled.");
      else if (e.kind === "update_progress") $("statusLine").textContent = `Downloading the new version${e.percent != null ? `: ${e.percent}%` : "…"}`;
      if (["download", "download_done", "download_failed", "download_cancelled"].includes(e.kind) && downloadEvent(e)) continue;
      if (e.kind === "download") $("statusLine").replaceChildren(`Downloading ${e.name}: ${e.gb} GB${e.percent != null ? ` (~${e.percent}%)` : ""} `,
        el("button", {onclick: () => api.cancel_download(e.model)}, "Cancel"));
      else if (e.kind === "download_cancelled") { log(`${e.name} download cancelled.`); $("statusLine").textContent = ""; }
      else if (e.kind === "download_done") { log(`${e.name} downloaded.`, "b"); loadModels(); $("statusLine").textContent = ""; }
      else if (e.kind === "download_failed") { log(`${e.name} download failed: ${e.error}`, "em"); $("statusLine").textContent = ""; }
      if (["step", "upscale_step", "train_step"].includes(e.kind)) {
        progress.step = e.step; progress.steps = e.steps;
        if (e.kind === "train_step") progress.trainEta = e.eta_s;
      }
      if (e.kind === "image" || e.kind === "video" || e.kind === "upscaled") { progress.step = 0; progress.steps = 0; }
      if (e.kind === "step") $("statusLine").textContent = `step ${e.step} of ${e.steps}`;
      else if (e.kind === "train_step") $("statusLine").textContent = `training ${e.subject}: step ${e.step} of ${e.steps}` + (e.eta_s != null ? `, about ${Math.ceil(e.eta_s / 60)} min left` : "");
      else if (e.kind === "training") log(`Training ${e.subject}…`);
      else if (e.kind === "trained") { log(`${e.subject} is trained and ready in the Subjects panel.`, "b"); loadSubjects(); }
      else if (e.kind === "train_failed") { log(`Training ${e.subject} failed: ${e.error}`, "em"); loadSubjects(); }
      else if (e.kind === "fixed") { log(`${e.subject} fixed: ${e.path.split(/[\\/]/).pop()}`, "b"); show(e.path); refreshGallery(true); }
      else if (e.kind === "upscale_step") $("statusLine").textContent = `upscaling ${e.name}: tile ${e.step} of ${e.steps}`;
      else if (e.kind === "upscaled") { log(`upscaled ${e.path.split(/[\\/]/).pop()} (${e.seconds} s)`, "b"); show(e.path); refreshGallery(true); }
    }
    const s = await api.state();
    drawBusy(s);
    if (!s.running && /^(image|upscaling|training)/.test($("statusLine").textContent)) $("statusLine").textContent = "";
  } catch (err) { console.error(err); flog("error", `poll: ${err && (err.stack || err.message) || err}`); }
  setTimeout(poll, 700);
}
function fmtTime(sec) {
  if (sec == null || !isFinite(sec)) return "estimating…";
  sec = Math.max(0, Math.round(sec));
  const m = Math.floor(sec / 60), r = sec % 60;
  return m >= 60 ? `about ${Math.floor(m / 60)} h ${m % 60} min left` : m ? `about ${m} min ${r.toString().padStart(2, "0")} s left` : `about ${r} s left`;
}
function drawBusy(s) {
  const busy = s.running || s.pending > 0;
  $("busy").hidden = !busy;
  if (!busy) { progress = {step: 0, steps: 0, trainEta: null, text: ""}; return; }
  $("busyTitle").textContent = s.label || "Working";
  const frac = progress.steps ? progress.step / progress.steps : 0;
  let pct, count, eta = null;
  if (s.kind === "train") {
    pct = frac * 100;
    count = progress.steps ? `step ${progress.step} of ${progress.steps}` : "starting (loading the model takes a minute)";
    eta = progress.trainEta;
  } else {
    const units = (s.done || 0) + frac;
    pct = s.total ? units / s.total * 100 : 0;
    count = s.total ? `${Math.min(s.done + 1, s.total)} of ${s.total}` : "";
    if (units > 0.15 && s.started) {
      const elapsed = Date.now() / 1000 - s.started;
      eta = elapsed / units * (s.total - units);
    }
  }
  $("busyBar").style.width = `${Math.min(100, pct).toFixed(1)}%`;
  $("busyCount").textContent = count + (s.pending ? `  (+${s.pending} more queued)` : "");
  $("busyEta").textContent = fmtTime(eta);
  $("busyDetail").textContent = $("statusLine").textContent;
}
async function gpu() {
  try {
    const g = await api.gpu();
    const dot = $("gpuDot");
    dot.className = "dot" + (g.state === "working" ? " work" : "");
    $("gpuText").textContent = g.state === "working" ? "GPU working" : "GPU ready";
    $("loadedText").textContent = g.loaded ? (modelById[g.loaded] ? modelById[g.loaded].name : g.loaded) : "nothing";
    if (g.used_mb != null) {
      $("vramBar").style.width = `${Math.round(g.used_mb / g.total_mb * 100)}%`;
      $("vramText").textContent = `${(g.used_mb / 1024).toFixed(1)} / ${Math.round(g.total_mb / 1024)} GB`;
      $("utilBar").style.width = `${Math.round(g.util)}%`;
      $("utilText").textContent = `${Math.round(g.util)}%`;
      $("tempText").textContent = g.temp_c != null ? `${Math.round(g.temp_c)} °C` : "-";
    }
  } catch (err) { console.error(err); }
  setTimeout(gpu, 3000);
}
async function show(path) {
  const r = await api.image(path);
  if (r.error) return;
  selectedPath = path;
  $("previewEmpty").hidden = true;
  if (r.media === "video") {
    $("previewImg").hidden = true;
    $("previewVideo").src = r.data; $("previewVideo").hidden = false;
    $("btnPreviewUp").hidden = true;
    $("btnPreviewAnim").hidden = true;
    $("vidPreviewEmpty").hidden = true;
    $("vidPreviewVideo").src = r.data; $("vidPreviewVideo").hidden = false;
    $("vidPreviewCaption").textContent = r.name; $("vidPreviewCaption").hidden = false;
    $("vidLastInfo").textContent = r.recipe ? `${r.recipe.model_name} · seed ${r.recipe.seed} · ${r.recipe.steps} steps` : "";
  } else {
    $("previewVideo").pause(); $("previewVideo").removeAttribute("src"); $("previewVideo").hidden = true;
    $("previewImg").src = r.data; $("previewImg").hidden = false;
    $("btnPreviewUp").hidden = false;
    $("btnPreviewAnim").hidden = false;
  }
  $("previewCaption").textContent = r.name; $("previewCaption").hidden = false;
  $("lastInfo").textContent = r.recipe ? `${r.recipe.model_name} · seed ${r.recipe.seed} · ${r.recipe.steps} steps` : "";
  document.querySelectorAll(".thumb").forEach(t => t.classList.toggle("sel", t.dataset.path === path));
}
async function refreshGallery(force) {
  const items = await api.gallery(null, null);      // folder and how many come from Settings
  galleryPaths = items.map(i => i.path);
  picked = picked.filter(p => galleryPaths.includes(p)); pickedChanged();
  const key = items.map(i => i.path).join("|");
  if (!force && key === galleryKey) return;
  galleryKey = key;
  const folder = settings.last.recent_folder;
  const g = $("gallery"); g.replaceChildren(el("span", {class: "day", title: folder || ""}, folder ? folder.split(/[\\/]/).filter(Boolean).pop() : "Recent"));
  if (!items.length) g.append(el("span", {class: "desc"}, "No images yet."));
  for (const it of items) {
    const t = el("div", {class: "thumb" + (it.width > it.height ? " wide" : "") + (it.path === selectedPath ? " sel" : ""), title: it.prompt || it.name});
    t.dataset.path = it.path;
    t.style.backgroundImage = `url("${it.thumb}")`;
    if (it.width && it.height) t.style.aspectRatio = `${it.width} / ${it.height}`;
    if (it.upscaled) t.append(el("span", {class: "badge"}, it.upscaled.toUpperCase()));
    if (it.media === "video") t.append(el("span", {class: "vbadge"}, "▶ MP4"));
    if (picked.includes(it.path)) t.classList.add("multi");
    t.onclick = e => pick(it.path, e);
    const acts = el("div", {class: "acts"},
      el("button", {title: "Same recipe", onclick: e => { e.stopPropagation(); remake(it.path, false); }}, "Re-make"),
      ...(it.media === "video" ? [] : [
        el("button", {title: "Upscale with the Upscale panel's size", onclick: e => { e.stopPropagation(); upscale([it.path]); }}, "Up"),
        el("button", {title: "Animate this picture in the Video tab", onclick: e => { e.stopPropagation(); animateImage(it.path); }}, "Anim"),
      ]),
      el("button", {title: "Same recipe, new seed", onclick: e => { e.stopPropagation(); remake(it.path, true); }}, "Vary"),
      el("button", {title: "Copy prompt, open folder, load settings", onclick: e => { e.stopPropagation(); itemMenu(it); }}, "⋯"));
    t.append(acts);
    g.append(t);
  }
}
function pick(path, e) {
  if (e.shiftKey && anchor && galleryPaths.includes(anchor)) {
    const [a, b] = [galleryPaths.indexOf(anchor), galleryPaths.indexOf(path)].sort((x, y) => x - y);
    picked = [...new Set([...picked, ...galleryPaths.slice(a, b + 1)])];
  } else if (e.ctrlKey) {
    picked = picked.includes(path) ? picked.filter(p => p !== path) : [...picked, path];
    anchor = path;
  } else {
    picked = [path]; anchor = path;
  }
  document.querySelectorAll(".thumb").forEach(t => t.classList.toggle("multi", picked.includes(t.dataset.path)));
  pickedChanged();
  show(path);
}
function pickedChanged() {
  $("btnUpscale").textContent = `Upscale selected (${picked.length})`;
  $("btnUpscale").disabled = !picked.length;
}
function currentUpChoice() { const px = $("upPx").value.trim(); return px ? px : upChoice; }
async function upscale(paths) {
  const r = await api.upscale(paths, currentUpChoice());
  if (r.error) return log(r.error, "em");
  log(`Upscaling ${r.count} image${r.count > 1 ? "s" : ""} to ${currentUpChoice()}…`);
}
async function remake(path, newSeed) {
  const r = await api.remake(path, newSeed);
  if (r.error) log(r.error, "em"); else if (r.needs_download) askDownload(r.needs_download);
}
async function itemMenu(it) {
  const r = await api.image(it.path);
  const rec = r.recipe;
  modal(it.name, el("div", {},
    rec ? el("table", {}, ...[["Prompt", rec.prompt], ["Negative", rec.negative || "-"], ["Model", `${rec.model_name} (${rec.license})`],
      ["Seed", rec.seed], ["Steps / guidance", `${rec.steps} / ${rec.guidance}`], ["Size", `${rec.width} × ${rec.height}`], ["Made", rec.created]]
      .map(([k, v]) => el("tr", {}, el("th", {}, k), el("td", {}, v)))) : el("p", {}, "No recipe file for this image."),
    el("div", {class: "row", style: "margin-top:12px"},
      el("button", {onclick: async () => { await navigator.clipboard.writeText(await api.copy_prompt(it.path)); log("Prompt copied."); }}, "Copy prompt"),
      el("button", {onclick: () => rec && loadRecipe(rec)}, "Load these settings"),
      el("button", {onclick: () => api.open_output(it.path)}, "Open folder"))));
}
function loadRecipe(r) {
  const m = modelById[r.model_id];
  if (m && m.kind === "video") {
    $("vidModel").value = r.model_id;
    showVidModel();
    $("vidPrompt").value = r.prompt || "";
    $("vidNegative").value = r.negative || "";
    $("vidWidth").value = r.width || m.defaults.width || 704;
    $("vidHeight").value = r.height || m.defaults.height || 480;
    if (r.extra && r.extra.frames) $("vidFrames").value = r.extra.frames;
    if (r.extra && r.extra.fps) $("vidFps").value = r.extra.fps;
    setSlider("vidSteps", r.steps || m.defaults.steps || 30);
    setSlider("vidGuidance", r.guidance ?? m.defaults.guidance ?? 3.0);
    $("vidRandomSeed").checked = false; $("vidSeed").disabled = false; $("vidSeed").value = r.seed;
    $("modal").hidden = true;
    showView("video");
    vidSummary(); vidTokenCount();
    return;
  }
  if (m) $("model").value = r.model_id;
  showModel();
  $("prompt").value = r.prompt; $("negative").value = r.negative;
  $("width").value = r.width; $("height").value = r.height;
  setSlider("steps", r.steps); setSlider("guidance", r.guidance);
  $("randomSeed").checked = false; $("seed").disabled = false; $("seed").value = r.seed;
  $("modal").hidden = true; showView("create"); tokenCount(); summary(); remember();
}

// ---------- modals: models, storage, help ----------
function modal(title, body) { $("modalTitle").textContent = title; $("modalBody").replaceChildren(body); $("modal").hidden = false; }
async function modelsModal() {
  await loadModels();
  const fams = ["sdxl", "sd15", "sd3", "flux", "flux2", "playground", "zimage", "ltx", "wan", "cogvideo", "svd"];
  const form = el("div", {class: "stack", style: "margin-top:14px"},
    el("h3", {}, "Add a model"),
    el("div", {class: "row"}, el("select", {id: "addKind"}, el("option", {value: "hf"}, "Hugging Face id (owner/name)"), el("option", {value: "file"}, ".safetensors file path")),
      el("input", {id: "addValue", placeholder: "e.g. SG161222/RealVisXL_V5.0 or C:\\models\\mine.safetensors"})),
    el("div", {class: "row"}, el("input", {id: "addName", placeholder: "Name shown in the menu"}),
      el("select", {id: "addFamily"}, ...fams.map(f => el("option", {value: f}, f))),
      el("select", {id: "addLicense"}, el("option", {value: "nc"}, "personal use"), el("option", {value: "conditions"}, "conditions"), el("option", {value: "commercial"}, "commercial OK")),
      el("button", {class: "primary", onclick: async () => {
        const r = await api.add_model($("addKind").value, $("addValue").value, $("addName").value || $("addValue").value, $("addFamily").value, $("addLicense").value);
        if (r.error) return alert(r.error);
        log(`Added model ${$("addName").value}`, "b"); await loadModels(r.id); modelsModal();
      }}, "Add")),
    el("p", {class: "desc"}, "Family: SDXL for most community photo models, SD 1.5 for older 512-pixel ones, or ltx / wan / cogvideo / svd for video models. Check the model's page if unsure."));
  const table = el("table", {}, el("tr", {}, el("th", {}, "Model"), el("th", {}, "Licence"), el("th", {}, "Status"), el("th", {}, "")),
    ...models.map(m => el("tr", {}, el("td", {}, el("b", {}, m.name), el("div", {class: "desc"}, m.description || "")),
      el("td", {}, m.license), el("td", {}, m.downloaded ? "on disk" : `${m.size_gb} GB`),
      el("td", {}, m.guide ? el("button", {onclick: () => api.model_guide(m.id)}, "Guide") : "",
        m.user ? el("button", {onclick: async () => { await api.remove_model(m.id); await loadModels(); modelsModal(); }}, "Remove") : ""))));
  modal("Models", el("div", {}, table, form));
}
async function storageModal() {
  const rows = await api.storage();
  const total = rows.reduce((a, r) => a + r.size_gb, 0).toFixed(1);
  modal(`Model storage (${total} GB)`, el("table", {}, el("tr", {}, el("th", {}, "Model"), el("th", {}, "Size"), el("th", {}, "Where"), el("th", {}, "")),
    ...rows.map(r => el("tr", {}, el("td", {}, r.name), el("td", {}, `${r.size_gb} GB`),
      el("td", {}, {app: "extra model folder (read-only)", cache: "aiVibe Studio downloads", file: "your file"}[r.where], el("div", {class: "desc"}, r.path)),
      el("td", {}, r.deletable ? el("button", {onclick: async () => {
        if (!confirm(`Delete ${r.name} (${r.size_gb} GB)? It can be downloaded again later.`)) return;
        const x = await api.delete_model(r.id); if (x.error) alert(x.error); else { await loadModels(); storageModal(); }
      }}, "Delete") : "")))));
}
function helpModal() {
  modal("Guide", el("div", {},
    el("p", {}, "Write a prompt, pick a model, press Generate or Ctrl+Enter. Every image is saved to the output folder with a matching .json recipe; Re-make and Vary in the strip below use it."),
    el("p", {}, "Thinking (steps): more steps = more detail, slower. Turbo models need only 4 to 8. Guidance: how strictly it follows the prompt. Recommended settings resets both for the chosen model."),
    el("h3", {}, "Video (Text-to-Video and Image-to-Video)"),
    el("p", {}, "Click ▶ Video on the left to generate .mp4 video clips locally on your GPU using Hugging Face video models (LTX-Video 2B, Wan 2.1 1.3B, CogVideoX 2B, and Stable Video Diffusion XT / XT 1.1). Click Animate this on any picture in the preview (or Anim on a thumbnail) to send that picture straight to the Video tab as a starting frame."),
    el("h3", {}, "Assistant"),
    el("p", {}, "Type what you want in the Assistant panel (\"20 space sci-fi pictures, upscaled to 1080p\") and press Write batch. The AI you picked in Settings > Assistant writes the batch; you check every line, change or remove lines, or type how to change it and press Revise. Nothing runs until you press Go. Only your typed text goes to the AI, never pictures or photos."),
    el("h3", {}, "Subjects (you, family, pets, the car)"),
    el("p", {}, "Click a trained subject in the Subjects panel, then write its name in the prompt (\"Alex hiking at sunset\"). Likeness 1.0 is the default; lower is looser, higher is stronger."),
    el("p", {}, "Tip for pictures of one person: add \"alone\" or \"solo\", or describe a closer shot (\"waist-up photo of Alex at a wedding\"). Wide scenes make the model invent extra people and it may give the main role to a stranger."),
    el("p", {}, "Click to fix: click any face (or a pet) in the preview and pick who it should be. That spot is redrawn as them and saved as a new file beside the original (..._fix)."),
    el("h3", {}, "Batch Maker and batch files"),
    el("p", {}, "Click Batch on the left to open the in-app Batch Maker: set shared defaults (model, size, negative prompt, subjects, upscale), add or reorder prompt items, or click + Batch next to Generate on the Create tab to send prompts over. Click Open file… to load and edit any existing .json or .txt batch file, Save / Save as… to write it to disk, and Run batch (or Save & Run) to execute it."),
    el("p", {}, "A batch file can be a .txt file (one prompt per line, # for comments) or a .json file:"),
    el("pre", {}, `{"defaults": {"model": "realvis-v5", "width": 1344, "height": 768},
 "items": [
  {"prompt": "a dinosaur in a swamp", "name": "dino"},
  {"prompt": "a lighthouse in a storm", "model": "z-image-turbo", "seed": 42, "count": 2}
 ]}`),
    el("p", {}, "Item keys: prompt (required), name, negative, model, width, height, count, seed, steps, guidance, upscale (2x, 4x, 1080p, 4k or a long edge in px), subjects (list of names, or {name, weight} objects; up to 3), reference (image path), reference_strength, face_fix (true/false). Model ids are listed under Models. The whole file is checked before anything runs."),
    el("p", {class: "desc"}, "Examples: the examples folder next to the app")));
}

// ---------- start ----------
async function start() {
  api = window.pywebview.api;
  early.splice(0).forEach(([l, m]) => flog(l, m));
  settings = await api.get_settings();
  $("version").textContent = settings.version;
  applyTheme(); buildLayout(settings.last.layout); bindDrag(); applyPanels(); placeRecent(settings.last.recent);
  await loadModels(settings.last.model || "z-image-turbo");
  const L = settings.last;
  if (L.prompt) $("prompt").value = L.prompt;
  if (L.negative != null) $("negative").value = L.negative;
  if (L.width) { $("width").value = L.width; $("height").value = L.height; setSlider("steps", L.steps); setSlider("guidance", L.guidance); $("count").value = L.count || 1; }
  else recommended();
  if (L.random_seed === false) { $("randomSeed").checked = false; $("seed").disabled = false; $("seed").value = L.seed; }
  document.querySelectorAll("#device button").forEach(b => b.classList.toggle("on", b.dataset.d === (L.device || "Auto")));
  fillChips(); fillNegatives(); summary(); tokenCount();

  $("model").onchange = () => { showModel(); recommended(); };
  ["steps", "guidance"].forEach(id => $(id).oninput = () => { setSlider(id, $(id).value); remember(); });
  ["width", "height", "count", "seed", "name"].forEach(id => $(id).onchange = () => { summary(); remember(); });
  $("prompt").oninput = () => { tokenCount(); remember(); };
  $("negative").oninput = remember;
  document.querySelectorAll("[data-clear]").forEach(b => b.onclick = () => {
    const t = $(b.dataset.clear);
    t.focus(); t.select();
    // deleting through the editor keeps Ctrl+Z, and its input event updates the word count and saved settings
    if (!document.execCommand("delete")) { t.value = ""; t.dispatchEvent(new Event("input")); }
  });
  $("randomSeed").onchange = () => { $("seed").disabled = $("randomSeed").checked; remember(); };
  document.querySelectorAll("#aspect button").forEach(b => b.onclick = () => {
    document.querySelectorAll("#aspect button").forEach(x => x.classList.toggle("on", x === b));
    const [aw, ah] = ASPECTS[b.dataset.a], longEdge = Math.max(modelById[$("model").value].defaults.width, 512);
    const scale = longEdge / Math.max(aw, ah);
    $("width").value = Math.round(aw * scale / 64) * 64; $("height").value = Math.round(ah * scale / 64) * 64;
    if (b.dataset.a === "16:9" && longEdge >= 1024) { $("width").value = 1344; $("height").value = 768; }
    summary(); remember();
  });
  document.querySelectorAll("#device button").forEach(b => b.onclick = () => {
    document.querySelectorAll("#device button").forEach(x => x.classList.toggle("on", x === b));
    api.save_settings({device: b.dataset.d});
    if (b.dataset.d === "CPU") log("CPU mode: very slow except for small models such as SD Compact.");
  });
  $("btnRecommended").onclick = recommended;
  upChoice = L.upscale || "4x";
  if (/^\d+$/.test(String(upChoice).replace("px", "")) && !["2x", "4x"].includes(upChoice)) { $("upPx").value = String(upChoice).replace("px", ""); }
  document.querySelectorAll("#upSize button").forEach(b => {
    b.classList.toggle("on", b.dataset.u === upChoice);
    b.onclick = () => { upChoice = b.dataset.u; $("upPx").value = ""; document.querySelectorAll("#upSize button").forEach(x => x.classList.toggle("on", x === b)); api.save_settings({upscale: upChoice}); };
  });
  $("upPx").onchange = () => { if ($("upPx").value) { document.querySelectorAll("#upSize button").forEach(x => x.classList.remove("on")); api.save_settings({upscale: $("upPx").value + "px"}); } };
  $("btnUpscale").onclick = () => upscale(picked);
  $("btnImportUp").onclick = async () => {
    const r = await api.import_and_upscale(null, currentUpChoice());
    if (r.cancelled) return;
    if (r.error) return log(r.error, "em");
    log(`Imported ${r.imported.split(/[\\/]/).pop()}, upscaling to ${currentUpChoice()}…`);
  };
  $("btnNewSubject").onclick = newSubjectModal;
  $("btnManageSubjects").onclick = manageSubjectsModal;
  $("btnPickRef").onclick = async () => { const r = await api.pick_reference(); if (r.error) return log(r.error, "em"); if (!r.cancelled) { reference = r; drawReference(); } };
  $("btnClearRef").onclick = () => { reference = null; drawReference(); };
  $("refStrength").oninput = () => { $("refStrengthOut").textContent = (+$("refStrength").value).toFixed(2); };
  await loadSubjects();
  $("btnPreviewUp").onclick = () => selectedPath && upscale([selectedPath]);
  $("btnPreviewAnim").onclick = () => selectedPath && animateImage(selectedPath);
  $("btnGenerate").onclick = generate;
  $("btnAddBatch").onclick = bmAddFromCreate;
  $("btnAssistant").onclick = assistantWrite;
  $("reviewClose").onclick = () => { $("assistantReview").hidden = true; };
  $("reviewToBatch").onclick = reviewToBatchMaker;
  $("reviewGo").onclick = reviewGo;
  $("reviewRevise").onclick = reviewRevise;
  $("reviewAdd").onclick = () => { if (!readReviewText()) return; review.text = false; review.batch.items.push({prompt: ""}); renderReview(); };
  $("reviewText").oninput = reviewEdited;      // a fix typed as text must be able to turn Go back on
  $("reviewTextToggle").onclick = () => { if (!readReviewText()) return; review.text = !review.text; renderReview(); };
  bindBatchMaker();
  bindVideoMaker();
  document.addEventListener("keydown", e => {
    if (e.ctrlKey && e.shiftKey && (e.key === "B" || e.key === "b")) { e.preventDefault(); reportOpen(); return; }
    if (e.ctrlKey && (e.key === "s" || e.key === "S") && !$("batchMaker").hidden && $("report").hidden && $("assistantReview").hidden) {
      e.preventDefault(); bmSaveFile(e.shiftKey); return;
    }
    if (e.ctrlKey && e.key === "Enter" && $("report").hidden && $("assistantReview").hidden) {
      if (!$("work").hidden) { e.preventDefault(); generate(); }
      else if (!$("videoMaker").hidden) { e.preventDefault(); generateVideo(); }
    }
    if (e.key === "Escape") { $("modal").hidden = true; $("moreMenu").hidden = true; $("report").hidden = true; $("assistantReview").hidden = true; }
  });
  $("navReport").onclick = reportOpen;
  $("busyReport").onclick = reportOpen;
  $("reportSave").onclick = reportSave;
  $("reportClose").onclick = () => { $("report").hidden = true; };
  $("navVideo").onclick = () => showView("video");
  $("navBatch").onclick = () => showView("batch");
  $("navFiles").onclick = () => api.open_output(null);
  $("navTheme").onclick = () => { settings.theme = settings.theme === "dark" ? "light" : "dark"; applyTheme(); api.save_settings({theme: settings.theme}); };
  $("navSettings").onclick = settingsModal;
  $("navUpdate").onclick = () => checkUpdate(true);
  $("dlHide").onclick = () => { $("dl").hidden = true; };
  $("dlCancel").onclick = () => { for (const id of dlWatch.pending.keys()) api.cancel_download(id); };
  $("busyCancel").onclick = () => { api.cancel(); $("busyCancel").textContent = "Cancelling after the current step…"; setTimeout(() => { $("busyCancel").textContent = "Cancel"; }, 8000); };
  $("btnGuide").onclick = () => api.model_guide($("model").value);
  $("btnAddModel").onclick = modelsModal;
  $("navModels").onclick = modelsModal;
  $("navStorage").onclick = storageModal;
  $("navGuide").onclick = helpModal;
  $("navCreate").onclick = () => showView("create");
  $("btnExpand").onclick = () => { const p = document.querySelector(".prompts"); p.classList.toggle("big"); $("btnExpand").textContent = p.classList.contains("big") ? "Shrink" : "Expand"; };
  $("btnClear").onclick = () => $("log").replaceChildren();
  $("modalClose").onclick = () => { $("modal").hidden = true; };
  $("previewImg").onclick = previewClick;
  $("previewImg").title = "Click a face (or a pet) to redraw it as one of your subjects";
  document.addEventListener("click", () => { $("moreMenu").hidden = true; });
  log(`aiVibe Studio ${settings.version} ready.`);
  const hw = settings.hardware || {};
  if (hw.support === "no_gpu") log("No supported NVIDIA card found. Pictures will be made on the CPU and be very slow.", "em");
  else if (hw.support === "unsupported_old") log(`${hw.name} is not supported (GTX 16-series and older can give black pictures). Pictures may fail or be very slow.`, "em");
  refreshGallery(true); poll(); gpu();
  if (settings.update_check !== false) setTimeout(() => checkUpdate(false), 5000);
  setInterval(() => refreshGallery(false), 15000);
}
window.addEventListener("pywebviewready", start);
