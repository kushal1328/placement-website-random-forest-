/* Placement Outlook - frontend logic (plain JavaScript, no libraries) */
(() => {
  "use strict";

  const boot = JSON.parse(document.getElementById("boot").textContent);
  const $ = (sel, root = document) => root.querySelector(sel);
  const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];

  const form = $("#student-form");
  const slipBody = $("#slip-body");
  const submitBtn = $("#submit-btn");
  const NUMERIC = boot.numeric_features;
  const CATEGORICAL = boot.categorical_features;
  const LABEL = boot.labels;

  /* ---------- small helpers ---------- */
  const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const num = (v) => String(Number(Number(v).toFixed(2)));            // 7.50 -> "7.5"
  // Never print a probability as exactly 100% or 0%: no model is that certain.
  const pct = (p) => (p >= 0.995 ? "99%+" : p <= 0.005 ? "<1%" : Math.round(p * 100) + "%");

  const SAMPLES = {
    strong: { CGPA: 8.4, Aptitude_Test_Score: 85, Backlogs: 0, Internships: 2, Projects: 5, Certifications: 3, Coding_Skills: 8, Communication_Skills: 8, Soft_Skills_Rating: 7, Degree: "B.Tech", Branch: "CSE" },
    onegap: { CGPA: 7.5, Aptitude_Test_Score: 75, Backlogs: 0, Internships: 1, Projects: 4, Certifications: 2, Coding_Skills: 6, Communication_Skills: 4, Soft_Skills_Rating: 6, Degree: "B.Tech", Branch: "IT" },
    atrisk: { CGPA: 6.1, Aptitude_Test_Score: 52, Backlogs: 2, Internships: 0, Projects: 2, Certifications: 1, Coding_Skills: 4, Communication_Skills: 4, Soft_Skills_Rating: 5, Degree: "B.Tech", Branch: "ECE" },
  };

  /* ---------- keep each slider and number box in sync ---------- */
  function paintRange(range) {
    const pctFill = ((range.value - range.min) / (range.max - range.min)) * 100;
    range.style.setProperty("--pct", pctFill + "%");
  }
  function setField(name, value) {
    const box = $("#n-" + name);
    box.value = value;
    const range = $(`.field[data-field="${name}"] .range`);
    if (range) { range.value = value; paintRange(range); }
    clearError(name);
  }
  $$(".field").forEach((field) => {
    const range = $(".range", field), box = $(".num", field);
    if (!range) return;
    paintRange(range);
    range.addEventListener("input", () => { box.value = range.value; paintRange(range); clearError(field.dataset.field); });
    box.addEventListener("input", () => {
      if (box.value !== "" && !Number.isNaN(Number(box.value))) { range.value = box.value; paintRange(range); }
      clearError(field.dataset.field);
    });
  });
  $$("select", form).forEach((s) => s.addEventListener("change", () => clearError(s.name)));

  /* ---------- errors ---------- */
  function showError(name, msg) {
    const field = $(`.field[data-field="${name}"]`);
    if (!field) return;
    field.classList.add("invalid");
    const el = $("#e-" + name);
    el.textContent = msg; el.hidden = false;
  }
  function clearError(name) {
    const field = $(`.field[data-field="${name}"]`);
    if (!field) return;
    field.classList.remove("invalid");
    const el = $("#e-" + name);
    el.textContent = ""; el.hidden = true;
  }

  /* ---------- read + validate the form ---------- */
  function readForm() {
    const data = {}; let ok = true;
    NUMERIC.forEach((f) => {
      const box = $("#n-" + f), v = box.value.trim();
      const [lo, hi] = boot.ranges[f];
      const n = Number(v);
      if (v === "" || Number.isNaN(n)) { showError(f, "Enter a number."); ok = false; }
      else if (n < lo || n > hi) { showError(f, `Use a value from ${num(lo)} to ${num(hi)}, the range the model was trained on.`); ok = false; }
      else { data[f] = n; clearError(f); }
    });
    CATEGORICAL.forEach((f) => { data[f] = $("#n-" + f).value; });
    return ok ? data : null;
  }

  /* ---------- render the result slip ---------- */
  const TICK = '<svg width="12" height="12" viewBox="0 0 12 12" aria-hidden="true"><path d="M2 6.5l2.8 2.8L10 3.5" fill="none" stroke="#fff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>';
  const CROSS = '<svg width="12" height="12" viewBox="0 0 12 12" aria-hidden="true"><path d="M3 3l6 6M9 3l-6 6" fill="none" stroke="#fff" stroke-width="2" stroke-linecap="round"/></svg>';

  function renderResult(r) {
    const p = r.prediction, s = r.second_opinion;
    const placed = p.class === 1;
    const failed = r.checklist.filter((c) => !c.passed);

    const secondLine = r.models_agree
      ? `<p class="second">Logistic Regression agrees: it estimates a <strong>${pct(s.probability_placed)}</strong> chance of being placed.</p>`
      : `<p class="second differ">The two models <strong>disagree</strong>. Random Forest says ${esc(p.label)}; Logistic Regression says ${esc(s.label)} (${pct(s.probability_placed)} chance). Treat this estimate with extra caution and review it with a counsellor.</p>`;

    const checkItem = (c) => {
      const need = (c.direction === "min" ? "needs at least " : "needs at most ") + num(c.target);
      return `<li class="${c.passed ? "pass" : "fail"}">
        <span class="mark">${c.passed ? TICK : CROSS}</span>
        <span><strong>${esc(c.label)}</strong> ${num(c.value)}, <span class="need">${c.passed ? "meets the minimum (" + need + ")" : need}</span><span class="sr-only">${c.passed ? " (met)" : " (not met)"}</span></span>
      </li>`;
    };
    const metItems = r.checklist.filter((c) => c.passed);
    const checks = (failed.length ? `<ul class="checks">${failed.map(checkItem).join("")}</ul>` : "") +
      (metItems.length ? `<details class="met"><summary>${failed.length ? "Show the " + metItems.length + " minimums you meet" : "All " + metItems.length + " minimum levels are met. Show details"}</summary><ul class="checks">${metItems.map(checkItem).join("")}</ul></details>` : "");

    let gapsBlock = "";
    if (r.probability_if_gaps_closed !== null) {
      gapsBlock = `<p class="closed">Reaching every minimum would lift your estimate from <strong>${pct(p.probability_placed)}</strong> to about <strong>${pct(r.probability_if_gaps_closed)}</strong>. Minimums are necessary but not enough, so keep building above them.</p>`;
    }

    let gainsBlock;
    if (r.fastest_gains.length) {
      gainsBlock = `<h3>Fastest ways to improve</h3>
        <p class="sub">Small changes tried one at a time, ranked by effect on the probability.</p>
        <ul class="gains">${r.fastest_gains.map((g) => `<li>
          <span><strong>${esc(g.label)}</strong> ${num(g.from)} to ${num(g.to)}</span>
          <span class="change">${pct(p.probability_placed)} to ${pct(g.new_probability)}</span></li>`).join("")}</ul>`;
    } else if (!failed.length) {
      gainsBlock = `<p class="all-clear">Every minimum level is met and no single small change moves the estimate much. Keep these strengths up.</p>`;
    } else {
      gainsBlock = `<p class="sub">No single change is enough on its own. Work through the unmet minimums above together.</p>`;
    }

    slipBody.innerHTML = `
      <div class="${placed ? "is-placed" : "is-not"}">
        <div class="verdict">
          <div class="stamp land">${esc(p.label)}</div>
          <div class="prob">
            <div class="prob-num">${pct(p.probability_placed)}</div>
            <div class="prob-cap">estimated chance of being placed</div>
          </div>
        </div>
        <div class="splitbar" role="img" aria-label="Placed ${pct(p.probability_placed)}, not placed ${pct(p.probability_not_placed)}">
          <span class="seg-placed" style="width:0%"></span>
        </div>
        <div class="splitlabels"><span class="l-placed">Placed ${pct(p.probability_placed)}</span><span class="l-not">Not placed ${pct(p.probability_not_placed)}</span></div>
      </div>
      <dl class="facts">
        <div><dt>Predicted class</dt><dd>${esc(p.label)} (class ${p.class})</dd></div>
        <div><dt>Probability of Placed</dt><dd>${p.probability_placed.toFixed(3)}</dd></div>
        <div><dt>Decision rule</dt><dd>Placed when the probability is ${p.threshold.toFixed(2)} or higher</dd></div>
        <div><dt>Placement outlook</dt><dd>${esc(p.outlook)}</dd></div>
      </dl>
      ${secondLine}
      <hr class="tear">
      <h3>Minimum levels</h3>
      <p class="sub">${r.checklist.length - failed.length} of ${r.checklist.length} met. Taken from the lowest levels seen among placed students.</p>
      ${checks}
      ${gapsBlock}
      ${gainsBlock}
      <p class="disclaimer">An estimate from patterns in past data, not a guarantee. Use it to plan your preparation. It should never be used to select or reject a student.</p>`;

    $("#slip").scrollTop = 0;
    requestAnimationFrame(() => requestAnimationFrame(() => {   // let the bar grow from 0
      const bar = $(".seg-placed", slipBody);
      if (bar) bar.style.width = (p.probability_placed * 100) + "%";
    }));
  }

  function renderProblem(message) {
    slipBody.innerHTML = `<p class="slip-error">${esc(message)}</p>`;
  }

  /* ---------- submit ---------- */
  async function predict() {
    const data = readForm();
    if (!data) {
      const first = $(".field.invalid .num");
      if (first) first.focus();
      renderProblem("Some values need attention. Fix the highlighted fields and predict again.");
      return;
    }
    submitBtn.disabled = true; submitBtn.textContent = "Predicting...";
    try {
      const res = await fetch("/api/predict", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(data) });
      const out = await res.json();
      if (!out.ok) {
        Object.entries(out.errors || {}).forEach(([k, msg]) => showError(k, msg));
        renderProblem("The server could not accept some values. Fix the highlighted fields and predict again.");
      } else {
        renderResult(out);
        if (window.innerWidth <= 1020) $("#slip").scrollIntoView({ behavior: "smooth", block: "start" });
      }
    } catch (err) {
      renderProblem("Could not reach the prediction service. Check that the server is still running, then try again.");
    } finally {
      submitBtn.disabled = false; submitBtn.textContent = "Predict my result";
    }
  }

  form.addEventListener("submit", (e) => { e.preventDefault(); predict(); });

  $$("[data-sample]").forEach((btn) => btn.addEventListener("click", () => {
    const sample = SAMPLES[btn.dataset.sample];
    Object.entries(sample).forEach(([k, v]) => setField(k, v));
    predict();
  }));

  $("#reset-btn").addEventListener("click", () => {
    Object.entries(boot.defaults).forEach(([k, v]) => setField(k, v));
    slipBody.innerHTML = `<div class="pending"><div class="stamp-ghost">Awaiting result</div><p>Fill in the form or pick a sample profile, then choose <strong>Predict my result</strong>. The class and probability appear here.</p></div>`;
  });

  /* ---------- ROC chart (drawn from the real test-set curve points) ---------- */
  function drawRoc() {
    const svg = $("#roc");
    if (!svg) return;
    const L = 46, T = 14, S = 250, B = T + S;                    // plot box: left, top, size, bottom
    const X = (v) => L + v * S, Y = (v) => B - v * S;
    const path = (pts) => pts.map(([x, y], i) => (i ? "L" : "M") + X(x).toFixed(1) + " " + Y(y).toFixed(1)).join(" ");
    let out = "";
    [0, 0.25, 0.5, 0.75, 1].forEach((t) => {
      out += `<line class="roc-grid" x1="${L}" x2="${L + S}" y1="${Y(t)}" y2="${Y(t)}"/><line class="roc-grid" y1="${T}" y2="${B}" x1="${X(t)}" x2="${X(t)}"/>`;
      out += `<text class="roc-text" x="${L - 8}" y="${Y(t) + 3.5}" text-anchor="end">${t}</text><text class="roc-text" x="${X(t)}" y="${B + 15}" text-anchor="middle">${t}</text>`;
    });
    out += `<line class="roc-diag" x1="${X(0)}" y1="${Y(0)}" x2="${X(1)}" y2="${Y(1)}"/>`;
    out += `<path class="roc-line" stroke="#96600A" d="${path(boot.models.logistic_regression.roc_curve)}"/>`;
    out += `<path class="roc-line" stroke="#2547C7" d="${path(boot.models.random_forest.roc_curve)}"/>`;
    out += `<rect class="roc-axis" x="${L}" y="${T}" width="${S}" height="${S}"/>`;
    out += `<text class="roc-title" x="${L + S / 2}" y="${B + 34}" text-anchor="middle">False positive rate</text>`;
    out += `<text class="roc-title" transform="translate(11 ${T + S / 2}) rotate(-90)" text-anchor="middle">True positive rate (recall)</text>`;
    svg.innerHTML = out;
  }
  drawRoc();
})();
