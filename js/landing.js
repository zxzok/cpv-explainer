/* Landing page: language toggle, copy actions, the four-point counterexample (live), the cohort diagram,
 * the resolution chart, the target-aware design demo, citation box and metadata.  All numbers that matter are
 * also present as HTML; the canvases illustrate. */
(function () {
  const M = window.CPV.M, DATA = window.CPV_DATA, LINKS = window.CPV_LINKS || {}, CITE = window.CPV_CITATION || {};
  const C = { bg: "#0F1722", panel: "#121A26", ink: "#E6ECF4", muted: "#8A97AC", dim: "#55627A", grid: "#1C2634",
              latent: "#4ECDC4", target: "#F5B841", A: "#5B8FF9", B: "#E15C9C", expl: "#5AD469", alert: "#FF6B6B", gold: "#FFD166", worldM: "#F08A4B" };
  const FONT = { sans: "'IBM Plex Sans', 'PingFang SC', 'Hiragino Sans GB', system-ui, sans-serif", mono: "'IBM Plex Mono', Menlo, monospace", display: "'Fraunces', Georgia, serif" };
  const rgba = (hex, a) => { const n = parseInt(hex.slice(1), 16); return `rgba(${(n >> 16) & 255},${(n >> 8) & 255},${n & 255},${a})`; };
  const reduced = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const lang = () => document.documentElement.getAttribute("data-lang") || "en";
  const t = (en, zh) => lang() === "zh" ? zh : en;
  const $ = (s, r = document) => r.querySelector(s), $$ = (s, r = document) => Array.from(r.querySelectorAll(s));

  // ------------------------------------------------------------ canvas helpers (CSS-pixel coordinates, HiDPI aware)
  function setupCanvas(cv) {
    // the logical size comes from the original width/height attributes; remember them, because the
    // backing store is resized for HiDPI and a second call must not read the enlarged size back
    if (!cv.dataset.w) { cv.dataset.w = cv.width; cv.dataset.h = cv.height; }
    const dpr = Math.min(2, window.devicePixelRatio || 1), W = +cv.dataset.w, H = +cv.dataset.h;
    const rect = cv.getBoundingClientRect(), cssW = rect.width || W, cssH = cssW * H / W;
    cv.width = Math.round(W * dpr); cv.height = Math.round(H * dpr); cv.style.height = cssH + "px";
    const ctx = cv.getContext("2d"); ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return { ctx, W, H };
  }
  const text = (ctx, s, x, y, o = {}) => { ctx.save(); ctx.font = `${o.weight || 400} ${o.size || 16}px ${FONT[o.font || "sans"]}`; ctx.fillStyle = o.color || C.ink; ctx.textAlign = o.align || "left"; ctx.textBaseline = o.baseline || "alphabetic"; ctx.globalAlpha = o.alpha === undefined ? 1 : o.alpha; ctx.fillText(s, x, y); ctx.restore(); };
  const line = (ctx, x1, y1, x2, y2, o = {}) => { ctx.save(); ctx.strokeStyle = o.color || C.muted; ctx.lineWidth = o.width || 2; ctx.lineCap = "round"; if (o.dash) ctx.setLineDash(o.dash); ctx.globalAlpha = o.alpha === undefined ? 1 : o.alpha; ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke(); ctx.restore(); };
  const poly = (ctx, pts, o = {}) => { if (pts.length < 2) return; ctx.save(); ctx.strokeStyle = o.color || C.ink; ctx.lineWidth = o.width || 2.5; ctx.lineJoin = "round"; ctx.lineCap = "round"; if (o.dash) ctx.setLineDash(o.dash); ctx.globalAlpha = o.alpha === undefined ? 1 : o.alpha; ctx.beginPath(); pts.forEach((p, i) => i ? ctx.lineTo(p[0], p[1]) : ctx.moveTo(p[0], p[1])); ctx.stroke(); ctx.restore(); };
  const dot = (ctx, x, y, r, color, o = {}) => { ctx.save(); ctx.globalAlpha = o.alpha === undefined ? 1 : o.alpha; ctx.fillStyle = color; ctx.beginPath(); ctx.arc(x, y, r, 0, Math.PI * 2); ctx.fill(); if (o.stroke) { ctx.strokeStyle = o.stroke; ctx.lineWidth = 1.5; ctx.stroke(); } ctx.restore(); };
  const rect = (ctx, x, y, w, h, o = {}) => { ctx.save(); ctx.globalAlpha = o.alpha === undefined ? 1 : o.alpha; if (o.fill) { ctx.fillStyle = o.fill; ctx.fillRect(x, y, w, h); } if (o.stroke) { ctx.strokeStyle = o.stroke; ctx.lineWidth = o.width || 1.5; ctx.strokeRect(x, y, w, h); } ctx.restore(); };
  const mark = (ctx, x, y1, y2, color, o = {}) => { ctx.save(); ctx.globalAlpha = o.alpha === undefined ? 1 : o.alpha; ctx.shadowColor = color; ctx.shadowBlur = o.blur === undefined ? 12 : o.blur; ctx.strokeStyle = color; ctx.lineWidth = o.width || 4; ctx.lineCap = "round"; if (o.dash) ctx.setLineDash(o.dash); ctx.beginPath(); ctx.moveTo(x, y1); ctx.lineTo(x, y2); ctx.stroke(); ctx.restore(); };

  // ------------------------------------------------------------ language
  function setLang(l) {
    document.documentElement.setAttribute("data-lang", l); document.documentElement.lang = l === "zh" ? "zh-CN" : "en";
    $$("#lang button").forEach(b => b.classList.toggle("active", b.dataset.lang === l));
    try { localStorage.setItem("cpv-lang", l); } catch (e) {}
    redrawAll();
  }
  const FULL = !!$("#tw-fig");   // the landing page; figure/social pages only borrow the drawing functions
  $$("#lang button").forEach(b => b.addEventListener("click", () => setLang(b.dataset.lang)));
  document.addEventListener("keydown", e => { if ((e.key === "l" || e.key === "L") && !/input|textarea/i.test(e.target.tagName)) setLang(lang() === "en" ? "zh" : "en"); });
  (function initLang() {
    const q = new URLSearchParams(location.search).get("lang");
    let saved = null; try { saved = localStorage.getItem("cpv-lang"); } catch (e) {}
    const nav = (navigator.language || "").toLowerCase().startsWith("zh") ? "zh" : "en";
    document.documentElement.setAttribute("data-lang", q || saved || nav);
    document.documentElement.lang = lang() === "zh" ? "zh-CN" : "en";
    $$("#lang button").forEach(b => b.classList.toggle("active", b.dataset.lang === lang()));
  })();

  // ------------------------------------------------------------ toast, copy, links, tracking
  const toastEl = $("#toast"); let toastTimer = null;
  function toast(msg) { if (!toastEl) return; toastEl.textContent = msg; toastEl.classList.add("show"); clearTimeout(toastTimer); toastTimer = setTimeout(() => toastEl.classList.remove("show"), 1800); }
  function copy(s, msg) { (navigator.clipboard ? navigator.clipboard.writeText(s) : Promise.reject()).then(() => toast(msg), () => { const ta = document.createElement("textarea"); ta.value = s; document.body.appendChild(ta); ta.select(); try { document.execCommand("copy"); toast(msg); } catch (e) { toast(t("Copy failed — select the text manually", "复制失败，请手动选择")); } ta.remove(); }); }
  function track(ev) {
    document.dispatchEvent(new CustomEvent("cpv:track", { detail: ev }));
    if (LINKS.analytics && navigator.sendBeacon) { try { navigator.sendBeacon(LINKS.analytics, JSON.stringify({ event: ev, page: location.pathname, lang: lang(), t: Date.now() })); } catch (e) {} }
  }
  $$("[data-track]").forEach(el => el.addEventListener("click", () => track(el.dataset.track)));
  if ($("#bibtex")) { $("#bibtex").textContent = CITE.bibtex || ""; $("#plain-cite").textContent = CITE.plain || ""; }
  $$("[data-copy]").forEach(b => b.addEventListener("click", () => copy(b.dataset.copy === "bibtex" ? CITE.bibtex : CITE.plain, t("Citation copied", "引用已复制"))));
  $$(".copy-link").forEach(b => b.addEventListener("click", () => { const u = location.href.split("#")[0] + "#" + b.dataset.anchor; copy(u, t("Link copied", "链接已复制")); }));
  if ($("#code-link")) {
    if (LINKS.code) { $$("[data-link=code]").forEach(a => { a.href = LINKS.code; a.target = "_blank"; a.rel = "noopener"; }); $("#code-link").href = LINKS.code; $("#code-link").target = "_blank"; $("#code-note").textContent = LINKS.code.replace(/^https?:\/\//, ""); }
    else { $("#code-link").addEventListener("click", e => e.preventDefault()); }
  }
  if (LINKS.contact && $("#contact-link")) $("#contact-link").href = "mailto:" + LINKS.contact;
  // structured data for scholarly indexing (kept in config.js so URLs can be filled in later)
  if (FULL) (function jsonld() {
    const base = LINKS.site ? LINKS.site.replace(/\/?$/, "/") : "";
    const d = { "@context": "https://schema.org", "@type": "ScholarlyArticle", headline: "Counterfactual Evaluation of Temporal Observation Protocols",
      author: { "@type": "Person", name: "Xizhe Zhang", email: LINKS.contact || undefined, affiliation: { "@type": "Organization", name: "School of Biomedical Engineering and Informatics, Nanjing Medical University" } },
      inLanguage: "en", dateCreated: "2026-08", keywords: "identifiability, observational equivalence, observation design, temporal aggregates, calibration",
      abstract: "We study counterfactual protocol evaluation: whether data collected under a realised observation protocol determine the predictive value of alternatives that were never deployed. Even infinite benchmark data need not determine this value; a value-specific identification theory, uniform calibration bounds and exact marginal gains connect identification, calibration resolution and observation design.",
      url: base || undefined, encoding: { "@type": "MediaObject", contentUrl: base + (LINKS.paper || "paper/main.pdf"), encodingFormat: "application/pdf" } };
    if (LINKS.doi) d.identifier = LINKS.doi; if (LINKS.code) d.codeRepository = LINKS.code; if (LINKS.journal) d.sameAs = LINKS.journal;
    const s = document.createElement("script"); s.type = "application/ld+json"; s.textContent = JSON.stringify(d); document.head.appendChild(s);
    if (base) { const c = document.createElement("link"); c.rel = "canonical"; c.href = base; document.head.appendChild(c); $$('meta[property="og:image"], meta[name="twitter:image"]').forEach(m => m.content = base + "assets/social-card.png"); }
  })();

  // ------------------------------------------------------------ the two-worlds figure (hero and interactive)
  const EPS0 = 0.1321;
  let epsMax = 0; { let e = 0; while (e < 0.5 && M.fourPoint(e + 0.001).pd) e += 0.001; epsMax = Math.floor(e * 0.97 * 1000) / 1000; }
  function drawTwoWorlds(cv, eps, o = {}) {
    const { ctx, W, H } = setupCanvas(cv), fp = M.fourPoint(eps), fp0 = M.fourPoint(0);
    ctx.fillStyle = C.bg; ctx.fillRect(0, 0, W, H);
    const compact = !!o.compact, pad = compact ? 36 : 48;
    // left: the four points and the correlation profiles
    const L = { x: pad + 40, y: compact ? 120 : 150, w: compact ? W * 0.48 : W * 0.46, h: compact ? H - 220 : H - 280 };
    // points row
    const py = compact ? 60 : 70, px = i => L.x + L.w * (0.06 + 0.88 * i / 3);
    line(ctx, L.x, py, L.x + L.w, py, { color: C.dim, width: 1.5 });
    const MT = window.CPV.mathText, mt = (s, x, y, o) => MT ? MT.draw(ctx, s, x, y, o) : text(ctx, s, x, y, { ...o, font: "mono" });
    for (let i = 0; i < 4; i++) { dot(ctx, px(i), py, 6, C.latent); mt("Z_" + i, px(i), py + 24, { size: 15, color: C.latent, align: "center" }); }
    mark(ctx, px(0), py - 34, py - 12, C.A, { width: 3, blur: 8 }); text(ctx, "A", px(0), py - 40, { size: 13, color: C.A, align: "center", weight: 500 });
    [1, 2].forEach(i => mark(ctx, px(i), py - 34, py - 12, C.B, { width: 2.5, dash: [5, 4], blur: 8 })); text(ctx, "B", px(1.5), py - 40, { size: 13, color: C.B, align: "center", weight: 500 });
    mt("Θ = \\frac{1}{4}(Z_0 + Z_1 + Z_2 + Z_3)", L.x + L.w, py - 36, { size: 13, color: C.target, align: "right" });
    // profile chart
    const X = v => L.x + (v + 0.2) / 3.4 * L.w, Y = v => L.y + L.h - (v + 0.3) / 1.35 * L.h;
    [0, 0.5, 1].forEach(v => { line(ctx, L.x, Y(v), L.x + L.w, Y(v), { color: C.grid, width: 1 }); text(ctx, String(v), L.x - 8, Y(v) + 4, { size: 11, color: C.dim, align: "right", font: "mono" }); });
    [0, 1, 2, 3].forEach(v => text(ctx, "ρ(" + v + ")", X(v), L.y + L.h + 20, { size: 12, color: C.muted, align: "center", font: "mono" }));
    const series = (ys, color, dash, label) => { const pts = ys.map((v, i) => [X(i), Y(v)]); poly(ctx, pts, { color, dash, width: dash ? 1.5 : 3 }); if (!dash) pts.forEach(p => dot(ctx, p[0], p[1], 4, color)); if (label) text(ctx, label, pts[3][0] + 10, pts[3][1] + 4, { size: 13, color, font: "mono", weight: 500 }); };
    series(fp0.rho0, C.dim, [4, 4]);
    series(fp.rhoPlus, C.latent, null, "K₊"); series(fp.rhoMinus, C.worldM, null, "K₋");
    text(ctx, t("correlation profile ρ(lag)", "相关函数 ρ(滞后)"), L.x, L.y - 10, { size: 13, color: C.muted });
    mt("ε = " + eps.toFixed(4), L.x + L.w, L.y - 10, { size: 13, color: C.gold, align: "right" });
    // right: what A sees (identical) and what B is worth (different)
    const R = { x: L.x + L.w + (compact ? 50 : 70), y: compact ? 56 : 70, w: W - pad - (L.x + L.w + (compact ? 50 : 70)) };
    text(ctx, t("what the benchmark (protocol A) can measure", "基准数据（协议 A）能测到的量"), R.x, R.y, { size: 13, color: C.A, weight: 500 });
    const rows = [["Var(Y_A)", "varY"], ["Cov(Y_A, Θ)", "covYT"], ["Var(Θ)", "varT"]];   // typeset by mathtext
    rows.forEach((r, i) => { const y = R.y + 34 + i * 28; mt(r[0], R.x, y, { size: 14, color: C.ink }); text(ctx, fp.obsPlus[r[1]].toFixed(6), R.x + R.w * 0.48, y, { size: 13, color: C.latent, font: "mono", align: "right" }); text(ctx, fp.obsMinus[r[1]].toFixed(6), R.x + R.w, y, { size: 13, color: C.worldM, font: "mono", align: "right" }); });
    const disc = Math.max(...rows.map(r => Math.abs(fp.obsPlus[r[1]] - fp.obsMinus[r[1]])));
    text(ctx, t("identical — discrepancy ", "完全相同——差异 ") + (disc < 1e-12 ? disc.toExponential(0) : disc.toExponential(1)), R.x, R.y + 34 + 3 * 28 + 2, { size: 12, color: C.muted, font: "mono" });
    const gy = R.y + (compact ? 170 : 190);
    mt(t("\\text{value of protocol } B = \\{Z_1, Z_2\\}", "\\text{协议 } B = \\{Z_1, Z_2\\} \\text{ 的价值}"), R.x, gy, { size: 13, color: C.B });
    const gauge = (y, v, color, label) => { const bx = R.x + 78, bw = R.w - 78 - 58; rect(ctx, bx, y - 13, bw, 18, { stroke: C.dim, width: 1 }); rect(ctx, bx + 1, y - 12, (bw - 2) * Math.max(0, Math.min(1, v)), 16, { fill: color }); mt(label, R.x, y + 3, { size: 13, color }); text(ctx, v.toFixed(3), bx + bw + 10, y + 2, { size: 14, color, font: "mono", weight: 500 }); };
    gauge(gy + 34, fp.valuePlus, C.latent, "I(B;K_+)"); gauge(gy + 68, fp.valueMinus, C.worldM, "I(B;K_−)");
    text(ctx, t("difference ", "相差 ") + Math.abs(fp.valueMinus - fp.valuePlus).toFixed(4), R.x + R.w, gy + 98, { size: 13, color: C.alert, font: "mono", align: "right" });
    if (!compact) { text(ctx, t("Same data. Two answers.", "同样的数据，两个答案。"), R.x, H - 40, { size: 20, color: C.gold, font: "display", weight: 500 }); }
    return fp;
  }
  const heroCv = $("#hero-fig"), twCv = $("#tw-fig"), slider = $("#tw-eps");
  let twEps = EPS0, heroEps = EPS0;
  function updateTwoWorlds(eps) {
    twEps = eps; if (!twCv) return;
    const fp = drawTwoWorlds(twCv, eps);
    const f6 = v => v.toFixed(6);
    $("#tw-vy-p").textContent = f6(fp.obsPlus.varY); $("#tw-vy-m").textContent = f6(fp.obsMinus.varY);
    $("#tw-cv-p").textContent = f6(fp.obsPlus.covYT); $("#tw-cv-m").textContent = f6(fp.obsMinus.covYT);
    $("#tw-vt-p").textContent = f6(fp.obsPlus.varT); $("#tw-vt-m").textContent = f6(fp.obsMinus.varT);
    const disc = Math.max(Math.abs(fp.obsPlus.varY - fp.obsMinus.varY), Math.abs(fp.obsPlus.covYT - fp.obsMinus.covYT), Math.abs(fp.obsPlus.varT - fp.obsMinus.varT));
    $("#tw-disc").textContent = disc < 1e-12 ? disc.toExponential(0) : disc.toExponential(1);
    $("#tw-vp").textContent = fp.valuePlus.toFixed(3); $("#tw-vm").textContent = fp.valueMinus.toFixed(3);
    $("#tw-bar-p").style.width = (fp.valuePlus * 100).toFixed(1) + "%"; $("#tw-bar-m").style.width = (fp.valueMinus * 100).toFixed(1) + "%";
    $("#tw-gap").textContent = Math.abs(fp.valueMinus - fp.valuePlus).toFixed(4);
    $("#tw-eps-val").textContent = "ε = " + eps.toFixed(4);
    slider.value = Math.round(eps / epsMax * 1000);
  }
  if (slider) slider.addEventListener("input", () => { cancelAnim(); updateTwoWorlds(slider.value / 1000 * epsMax); });
  let animId = null;
  function cancelAnim() { if (animId) cancelAnimationFrame(animId); animId = null; }
  function animateEps(to, ms, done) {
    cancelAnim(); if (reduced) { updateTwoWorlds(to); done && done(); return; }
    const from = twEps, t0 = performance.now();
    const step = now => { const u = Math.min(1, (now - t0) / ms), e = u < 0.5 ? 2 * u * u : 1 - Math.pow(-2 * u + 2, 2) / 2; updateTwoWorlds(from + (to - from) * e); if (u < 1) animId = requestAnimationFrame(step); else { animId = null; done && done(); } };
    animId = requestAnimationFrame(step);
  }
  function replay() { updateTwoWorlds(0); setTimeout(() => animateEps(EPS0, 2800), 400); }
  if ($("#tw-replay")) $("#tw-replay").addEventListener("click", replay);
  if ($("#see-60")) $("#see-60").addEventListener("click", () => { setTimeout(replay, 500); });
  // 60-second narration: the four beats of explainer chapter 1, with the cards lit in turn
  const NARR = window.CPV_NARRATION, twEl = $("#tw"), cap = $("#tw-cap"), playBtn = $("#tw-play");
  const audio = new Audio(); let playing = false, beatIdx = -1;
  function beatsOf(id) { const ch = (NARR && NARR.chapters || []).find(c => c.id === id); return ch ? ch.beats : []; }
  function stopNarration() { playing = false; try { audio.pause(); } catch (e) {} audio.removeAttribute("src"); audio.load(); if (!twEl) return; twEl.classList.remove("playing"); $$(".tw-card", twEl).forEach(c => c.classList.remove("on")); cap.textContent = ""; playBtn.querySelector(".en").textContent = "Play the 60-second narration"; playBtn.querySelector(".zh").textContent = "播放 60 秒解说"; }
  function playBeat(j) {
    const beats = beatsOf("e1"); if (j >= beats.length) { stopNarration(); track("core_example_completed"); return; }
    beatIdx = j; const b = beats[j]; cap.textContent = lang() === "zh" ? b.zh : b.en;
    $$(".tw-card", twEl).forEach(c => c.classList.toggle("on", c.dataset.beat === String(j) || (j === 1 && c.dataset.beat === "0")));
    if (j === 0) updateTwoWorlds(0); if (j === 1) animateEps(EPS0, 2600);
    audio.src = `audio/explainer/${lang()}/e1-b${j}.m4a`;
    audio.onended = () => playing && setTimeout(() => playBeat(j + 1), 350);
    audio.onerror = () => { const ms = 1500 + (lang() === "zh" ? b.zh.length * 220 : b.en.split(/\s+/).length * 380); setTimeout(() => playing && playBeat(j + 1), ms); };
    audio.play().catch(() => audio.onerror());
  }
  if (playBtn) playBtn.addEventListener("click", () => { if (playing) { stopNarration(); return; } playing = true; twEl.classList.add("playing"); playBtn.querySelector(".en").textContent = "Stop"; playBtn.querySelector(".zh").textContent = "停止"; playBeat(0); });

  // ------------------------------------------------------------ cohort diagram
  function drawCohort(cv) {
    const { ctx, W, H } = setupCanvas(cv); ctx.fillStyle = C.bg; ctx.fillRect(0, 0, W, H);
    const rand = M.rng(3), pad = 40, x0 = pad + 150, x1 = W - pad;
    text(ctx, t("large routine cohort — one sparse protocol", "大规模常规队列——同一个稀疏协议"), pad, 40, { size: 14, color: C.ink, weight: 500 });
    text(ctx, t("n units, same time points each", "n 个对象，每个都在同样的时刻"), pad, 62, { size: 12, color: C.muted });
    const rows = 16, y0 = 84, rh = 14, times = [0.18, 0.52];
    for (let i = 0; i < rows; i++) { const y = y0 + i * rh; line(ctx, x0, y, x1, y, { color: C.grid, width: 1 }); times.forEach(u => dot(ctx, x0 + (x1 - x0) * u, y, 3.5, C.A)); }
    text(ctx, "…", x0 - 24, y0 + rows * rh + 6, { size: 18, color: C.dim });
    line(ctx, x0, y0 + rows * rh + 26, x1, y0 + rows * rh + 26, { color: C.dim, width: 1.5 });
    times.forEach(u => text(ctx, "A", x0 + (x1 - x0) * u, y0 + rows * rh + 44, { size: 12, color: C.A, align: "center", weight: 500 }));
    text(ctx, t("adds units, repeats the same constraints", "增加对象，只重复同样的约束"), x1, y0 + rows * rh + 44, { size: 12, color: C.muted, align: "right" });
    // dense subset
    const y2 = y0 + rows * rh + 90;
    text(ctx, t("small dense calibration subset", "小规模密集校准子集"), pad, y2, { size: 14, color: C.ink, weight: 500 });
    text(ctx, t("m units, whole trajectory", "m 个对象，整条轨迹"), pad, y2 + 22, { size: 12, color: C.muted });
    const p = 64, K = M.traitState(M.ouKernel(p, 0.12, 1), 0.3), Lc = M.cholesky(K, 1e-9), paths = M.samplePaths(Lc, 6, rand);
    paths.forEach((z, k) => { const yy = y2 + 48 + k * 40; line(ctx, x0, yy, x1, yy, { color: C.grid, width: 1 }); const pts = Array.from(z, (v, j) => [x0 + (x1 - x0) * (j + 0.5) / p, yy - v * 9]); poly(ctx, pts, { color: C.latent, width: 1.8, alpha: 0.9 }); times.forEach(u => dot(ctx, x0 + (x1 - x0) * u, yy, 3, C.A, { alpha: 0.8 })); });
    text(ctx, t("reveals how times relate → identifies the latent covariance → values every candidate protocol", "揭示时刻之间的关系 → 识别潜在协方差 → 为每个候选协议定价"), pad, H - 22, { size: 12.5, color: C.gold });
  }

  // ------------------------------------------------------------ resolution chart (Figure 2c, coarsest vs finest class)
  function drawResolution(cv) {
    const { ctx, W, H } = setupCanvas(cv); ctx.fillStyle = C.bg; ctx.fillRect(0, 0, W, H);
    const res = DATA.resolution, R = { x: 90, y: 50, w: W - 130, h: H - 130 };
    const lx = v => Math.log10(v), X = v => R.x + (lx(v) - lx(20)) / (lx(1300) - lx(20)) * R.w, Y = v => R.y + R.h - (lx(v) - lx(0.004)) / (lx(0.08) - lx(0.004)) * R.h;
    [25, 100, 1000].forEach(v => { line(ctx, X(v), R.y, X(v), R.y + R.h, { color: C.grid, width: 1 }); text(ctx, String(v), X(v), R.y + R.h + 22, { size: 12, color: C.muted, align: "center", font: "mono" }); });
    [0.005, 0.01, 0.02, 0.05].forEach(v => { line(ctx, R.x, Y(v), R.x + R.w, Y(v), { color: C.grid, width: 1 }); text(ctx, String(v), R.x - 8, Y(v) + 4, { size: 12, color: C.muted, align: "right", font: "mono" }); });
    line(ctx, R.x, R.y + R.h, R.x + R.w, R.y + R.h, { color: C.dim }); line(ctx, R.x, R.y, R.x, R.y + R.h, { color: C.dim });
    text(ctx, t("calibration trajectories m", "校准轨迹数 m"), R.x + R.w / 2, R.y + R.h + 46, { size: 13, color: C.muted, align: "center" });
    ctx.save(); ctx.translate(R.x - 58, R.y + R.h / 2); ctx.rotate(-Math.PI / 2); text(ctx, t("true selection regret", "真实选择损失"), 0, 0, { size: 13, color: C.muted, align: "center" }); ctx.restore();
    const draw = (ys, color, label) => { const pts = res.m.map((m, i) => [X(m), Y(ys[i])]); poly(ctx, pts, { color, width: 3 }); pts.forEach(p => dot(ctx, p[0], p[1], 4, color)); text(ctx, label, pts[pts.length - 1][0] - 6, pts[pts.length - 1][1] - 12, { size: 12.5, color, align: "right", weight: 500 }); };
    draw(res.regret[1], C.A, t("coarsest class: 2 layouts", "最粗类：2 种布局")); draw(res.regret[4], C.B, t("finest class: 568 exact supports", "最细类：568 个精确位置"));
    text(ctx, t("≈ restriction regret of the coarse class", "≈ 粗类的限制损失"), X(1000) + 4, Y(res.regret[1][5]) + 4, { size: 11, color: C.A, align: "right", alpha: 0 });
    const f = v => v.toFixed(3);
    if ($("#res-fine-25")) ["", "-zh"].forEach(s => { $("#res-fine-25" + s).textContent = f(res.regret[4][0]); $("#res-coarse-25" + s).textContent = f(res.regret[1][0]); $("#res-coarse-1000" + s).textContent = f(res.regret[1][5]); $("#res-fine-1000" + s).textContent = f(res.regret[4][5]); });
  }

  // ------------------------------------------------------------ target-aware design demo
  const design = (function () {
    const p = 64, T = 20, K = M.traitState(M.ouKernel(p, 2.2, T), 0.25);
    const w = Float64Array.from({ length: p }, (_, i) => i + 1); { const tot = w.reduce((a, b) => a + b, 0); for (let i = 0; i < p; i++) w[i] /= tot; }
    const cands = []; for (let j = 2; j < p; j += 4) { const ell = new Float64Array(p); ell[j] = 1; cands.push({ ell, idx: j, r: 0.3, cost: 1 }); }
    const xs = M.grid(p, T);
    return { p, T, xs, cands, K, w, steps: { mean: M.greedy(K, w, cands, 4, { kind: "mean" }), occ: M.greedy(K, w, cands, 4, { kind: "occ", c: 1 }) } };
  })();
  let target = "mean";
  // the user's own measurement set per target (starts at the greedy choice); marks are draggable and snap to the 16 candidate slots
  const dstate = { sel: { mean: design.steps.mean.map(s => s.idx), occ: design.steps.occ.map(s => s.idx) }, custom: { mean: false, occ: false }, drag: null };
  const targetOf = kind => kind === "mean" ? { kind: "mean" } : { kind: "occ", c: 1 };
  function designValue(sel, kind) { const A = sel.map(c => design.cands[c].ell), R = M.diagR(A.length, design.cands[0].r); return M.value(design.K, A, R, design.w, targetOf(kind)); }
  const greedyValue = { mean: design.steps.mean[design.steps.mean.length - 1].value, occ: design.steps.occ[design.steps.occ.length - 1].value };
  function drawDesign(cv) {
    const { ctx, W, H } = setupCanvas(cv); ctx.fillStyle = C.bg; ctx.fillRect(0, 0, W, H);
    const x0 = 60, x1 = W - 60, X = u => x0 + (x1 - x0) * u / design.T;
    const rowsDef = [["mean", t("mean level  g(z) = z", "均值目标  g(z) = z"), C.expl], ["occ", t("time above a threshold  g(z) = 1{z > 1}", "超过阈值的时间  g(z) = 1{z > 1}"), C.target]];
    rowsDef.forEach(([key, label, color], k) => {
      const y = 120 + k * 170, on = key === target, a = on ? 1 : 0.28;
      text(ctx, label, x0, y - 48, { size: 15, color, font: "mono", alpha: a, weight: on ? 500 : 400 });
      line(ctx, x0, y, x1, y, { color: C.dim, width: 2, alpha: a });
      design.cands.forEach(c => line(ctx, X(design.xs[c.idx]), y - 6, X(design.xs[c.idx]), y + 6, { color: C.muted, width: 1.5, alpha: a * 0.7 }));
      const sel = dstate.sel[key], custom = dstate.custom[key];
      if (custom) design.steps[key].forEach(s => { const x = X(design.xs[s.action.idx]); line(ctx, x, y - 22, x, y + 22, { color, width: 2, alpha: a * 0.35 }); });   // faint: where greedy would measure
      sel.forEach((c, i) => { const x = X(design.xs[design.cands[c].idx]); mark(ctx, x, y - 34, y + 34, color, { width: 5, alpha: a, blur: on ? 12 : 0 }); if (on) { dot(ctx, x, y - 40, 5, color); dot(ctx, x, y + 40, 5, color); } text(ctx, custom ? design.xs[design.cands[c].idx].toFixed(1) + " h" : String(i + 1), x, y + 58, { size: 12, color, align: "center", font: "mono", alpha: a }); });
      if (on) {
        const v = designValue(sel, key);
        text(ctx, t("value of these 4 measurements: ", "这 4 次测量的价值：") + v.toFixed(3), x1, y - 48, { size: 13, color, font: "mono", align: "right" });
        if (custom) text(ctx, t("greedy choice: ", "贪心选择：") + greedyValue[key].toFixed(3) + (v < greedyValue[key] - 1e-9 ? t("  (yours is lower)", "（你的更低）") : v > greedyValue[key] + 1e-9 ? t("  (yours is higher)", "（你的更高）") : ""), x1, y - 30, { size: 12, color: C.muted, font: "mono", align: "right" });
        else text(ctx, t("drag a line to another slot", "拖动竖线换一个时刻"), x1, y - 30, { size: 12, color: C.muted, align: "right" });
      }
    });
    [0, 5, 10, 15, 20].forEach(h => text(ctx, h + " h", X(h), H - 26, { size: 12, color: C.dim, align: "center", font: "mono" }));
    text(ctx, t("same latent process, same 16 candidates, same budget of 4 — the chosen times differ", "同一个潜在过程、同样的 16 个候选、同样的预算 4——选中的时刻不同"), x0, H - 54, { size: 13, color: C.gold });
  }
  function designReset() { dstate.sel[target] = design.steps[target].map(s => s.idx); dstate.custom[target] = false; if ($("#design-reset")) $("#design-reset").hidden = true; drawDesign($("#design-fig")); }
  $$(".target-switch button").forEach(b => b.addEventListener("click", () => { target = b.dataset.target; $$(".target-switch button").forEach(x => x.classList.toggle("active", x === b)); if ($("#design-reset")) $("#design-reset").hidden = !dstate.custom[target]; drawDesign($("#design-fig")); }));
  if ($("#design-reset")) $("#design-reset").addEventListener("click", designReset);
  (function wireDesignDrag() {
    const cv = $("#design-fig"); if (!cv) return;
    const logical = ev => { const r = cv.getBoundingClientRect(), W = +cv.dataset.w || cv.width, H = +cv.dataset.h || cv.height; return { sx: (ev.clientX - r.left) * W / r.width, sy: (ev.clientY - r.top) * H / r.height, W }; };
    const geom = W => { const x0 = 60, x1 = W - 60; return { X: u => x0 + (x1 - x0) * u / design.T, y: 120 + (target === "mean" ? 0 : 1) * 170 }; };
    const slotX = (c, X) => X(design.xs[design.cands[c].idx]);
    cv.addEventListener("pointerdown", ev => {
      const { sx, sy, W } = logical(ev), { X, y } = geom(W); if (Math.abs(sy - y) > 48) return;
      const sel = dstate.sel[target]; let best = -1, bd = 16;
      sel.forEach((c, i) => { const d = Math.abs(sx - slotX(c, X)); if (d < bd) { bd = d; best = i; } });
      if (best < 0) return;
      dstate.drag = { pos: best }; cv.setPointerCapture(ev.pointerId); cv.classList.add("dragging"); ev.preventDefault();
    });
    cv.addEventListener("pointermove", ev => {
      const { sx, sy, W } = logical(ev), { X, y } = geom(W);
      if (!dstate.drag) { const sel = dstate.sel[target]; cv.style.cursor = (Math.abs(sy - y) <= 48 && sel.some(c => Math.abs(sx - slotX(c, X)) < 16)) ? "grab" : "default"; return; }
      const sel = dstate.sel[target]; let nearest = -1, nd = Infinity;
      design.cands.forEach((c, j) => { const d = Math.abs(sx - slotX(j, X)); if (d < nd && (j === sel[dstate.drag.pos] || !sel.includes(j))) { nd = d; nearest = j; } });   // snap to the nearest free slot
      if (nearest >= 0 && nearest !== sel[dstate.drag.pos]) { sel[dstate.drag.pos] = nearest; dstate.custom[target] = true; if ($("#design-reset")) $("#design-reset").hidden = false; drawDesign(cv); }
    });
    const end = () => { if (!dstate.drag) return; dstate.drag = null; cv.classList.remove("dragging"); };
    cv.addEventListener("pointerup", end); cv.addEventListener("pointercancel", end);
  })();

  // ------------------------------------------------------------ draw / redraw
  function redrawAll() {
    if (heroCv) drawTwoWorlds(heroCv, heroEps, { compact: true });
    updateTwoWorlds(twEps);
    if ($("#cohort-fig")) drawCohort($("#cohort-fig")); if ($("#res-fig")) drawResolution($("#res-fig")); if ($("#design-fig")) drawDesign($("#design-fig"));
  }
  window.CPV_LANDING = { drawTwoWorlds, drawCohort, drawResolution, drawDesign, setTarget: k => { target = k; }, EPS0, epsMax, C, FONT, setupCanvas, text, line, rect, dot, mark, poly, t };
  let resizeTimer = null;
  window.addEventListener("resize", () => { clearTimeout(resizeTimer); resizeTimer = setTimeout(redrawAll, 120); });
  if (document.fonts && document.fonts.ready) document.fonts.ready.then(redrawAll);
  redrawAll();
  // the hero figure eases into the paper's ε once on load
  if (!reduced && heroCv) { heroEps = 0; drawTwoWorlds(heroCv, 0, { compact: true }); const t0 = performance.now(); const step = now => { const u = Math.min(1, (now - t0 - 600) / 2400); if (u >= 0) { const e = u < 0.5 ? 2 * u * u : 1 - Math.pow(-2 * u + 2, 2) / 2; heroEps = EPS0 * e; drawTwoWorlds(heroCv, heroEps, { compact: true }); } if (u < 1) requestAnimationFrame(step); }; requestAnimationFrame(step); }
  if (location.hash === "#two-worlds") setTimeout(replay, 600);
})();

// typeset the page's inline formulas (<span class="f">TeX</span>)
if (window.CPV && CPV.mathText) CPV.mathText.typesetAll(document);

// the tour links carry the landing page's current language (explainer/?lang=zh …), kept in sync with the toggle
(function syncTourLinks() {
  const cur = () => document.documentElement.dataset.lang || "en";
  const sync = () => document.querySelectorAll('a[href^="explainer/"], a[href^="technical/"]').forEach(a => { const h = a.getAttribute("href"), [pathAndQuery, hash] = h.split("#"), path = pathAndQuery.split("?")[0]; a.setAttribute("href", path + "?lang=" + cur() + (hash ? "#" + hash : "")); });
  sync();
  new MutationObserver(sync).observe(document.documentElement, { attributes: true, attributeFilter: ["data-lang"] });
})();
