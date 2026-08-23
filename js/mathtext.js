/* mathtext: typeset a small TeX-like subset on a 2D canvas.
 *
 *   CPV.mathText.draw(ctx, "I_g(S;K) = \\frac{F_g(S;K)}{V_g(K)}", x, y, { size, color, align, alpha })
 *
 * Supported: _ and ^ (single token or {group}), \frac{a}{b}, \text{…} / \mathrm{…} (upright, spaces kept),
 * \it{…}, \bf{…}, \sum \int \prod, \top (transpose), \inv (^{-1}), \norm{…}, \abs{…}, \hat \bar \tilde
 * (combining accents), \le \ge \ne \approx \to \Rightarrow \Leftrightarrow \pm \cdot \times \infty \in \subseteq,
 * \, \; \quad spacing, and Unicode Greek typed directly.  Single Latin letters are italic, multi-letter words
 * (Var, Cov, diag, sup, max, …) are upright, digits and punctuation upright, lowercase Greek italic.
 * Relations and binary operators get proper spacing; sub/superscripts are scaled and shifted; fractions are
 * stacked with a rule.  Returns the layout width so callers can align or underline. */
(function () {
  const CPV = window.CPV = window.CPV || {};
  const SANS = "'IBM Plex Sans', 'PingFang SC', 'Hiragino Sans GB', 'Noto Sans CJK SC', system-ui, sans-serif";
  const SYMBOLS = { le: "≤", ge: "≥", ne: "≠", approx: "≈", to: "→", Rightarrow: "⇒", Leftrightarrow: "⇔", pm: "±", cdot: "·", times: "×",
    infty: "∞", in: "∈", subseteq: "⊆", succeq: "⪰", succ: "≻", sim: "∼", propto: "∝", ldots: "…", cdots: "⋯", mid: "|",
    alpha: "α", beta: "β", gamma: "γ", delta: "δ", epsilon: "ε", varepsilon: "ε", eta: "η", theta: "θ", lambda: "λ", mu: "μ", nu: "ν", xi: "ξ",
    rho: "ρ", sigma: "σ", tau: "τ", phi: "φ", varphi: "φ", omega: "ω", Delta: "Δ", Theta: "Θ", Sigma: "Σ", Pi: "Π", Omega: "Ω", Xi: "Ξ", Phi: "Φ",
    sum: "Σ", int: "∫", prod: "Π", partial: "∂", nabla: "∇", ell: "ℓ", hbar: "ℏ", star: "⋆", circ: "∘", equiv: "≡", coloneq: "≔" };
  const BIG = new Set(["Σ", "∫", "Π"]);
  const SCRIPT = { A: "𝒜", B: "ℬ", C: "𝒞", D: "𝒟", E: "ℰ", F: "ℱ", G: "𝒢", H: "ℋ", I: "ℐ", L: "ℒ", M: "ℳ", N: "𝒩", P: "𝒫", R: "ℛ", S: "𝒮", T: "𝒯" };
  const RELATIONS = new Set(["=", "≤", "≥", "≠", "≈", "→", "⇒", "⇔", "∈", "⊆", "⪰", "≻", "∼", "∝", "≡", "≔", ":", "<", ">"]);
  const BINOPS = new Set(["+", "−", "-", "±", "·", "×", "∘"]);
  const ACCENTS = { hat: 1, bar: 1, tilde: 1, vec: 1, dot: 1 };
  const ACCENT_GLYPH = { hat: "ˆ", bar: "¯", tilde: "˜", vec: "→", dot: "˙" };
  const GREEK_LOWER = /[α-ω]/, GREEK_UPPER = /[Α-Ω]/;

  // ------------------------------------------------------------ parse
  function parse(src) {
    let i = 0;
    const peek = () => src[i], next = () => src[i++];
    function group() {                      // after "{"
      const items = []; while (i < src.length && peek() !== "}") items.push(atom()); next(); return items;
    }
    function arg() {                        // one argument: {group} | \cmd | char
      while (peek() === " ") next();
      if (peek() === "{") { next(); return group(); }
      return [atom()];
    }
    function word() { let w = ""; while (i < src.length && /[A-Za-z]/.test(peek())) w += next(); return w; }
    function atom() {
      const c = next();
      if (c === " ") return { t: "space", w: 0.22, implicit: true };
      if (c === "{") return { t: "row", items: group() };
      if (c === "\\") {
        const w = word();
        if (w === "") { const s = next(); return s === "," ? { t: "space", w: 0.17 } : s === ";" ? { t: "space", w: 0.28 } : s === "|" ? { t: "ch", ch: "‖", st: "rm" } : { t: "ch", ch: s, st: "rm" }; }
        if (w === "frac") return { t: "frac", num: arg(), den: arg() };
        if (w === "text" || w === "mathrm" || w === "rm") return { t: "text", str: rawGroup(), st: "rm" };
        if (w === "it" || w === "mathit") return { t: "text", str: rawGroup(), st: "it" };
        if (w === "bf" || w === "mathbf") return { t: "text", str: rawGroup(), st: "bf" };
        if (w === "quad") return { t: "space", w: 1 };
        if (w === "mathcal") { const g = rawGroup(); return { t: "ch", ch: SCRIPT[g] || g, st: "rm" }; }
        if (w === "top") return { t: "sup", base: null, sup: [{ t: "ch", ch: "T", st: "rm" }] };
        if (w === "inv") return { t: "sup", base: null, sup: [{ t: "ch", ch: "−", st: "rm" }, { t: "ch", ch: "1", st: "rm" }] };
        if (w === "norm") return { t: "row", items: [{ t: "ch", ch: "‖", st: "rm" }, ...arg(), { t: "ch", ch: "‖", st: "rm" }] };
        if (w === "abs") return { t: "row", items: [{ t: "ch", ch: "|", st: "rm" }, ...arg(), { t: "ch", ch: "|", st: "rm" }] };
        if (ACCENTS[w]) { const a = arg(); const base = a[0] && a[0].t === "ch" ? a[0] : null; return base ? { t: "ch", ch: base.ch, st: base.st, accent: ACCENT_GLYPH[w] } : { t: "row", items: a }; }
        if (SYMBOLS[w]) { const s = SYMBOLS[w]; return { t: "ch", ch: s, st: GREEK_LOWER.test(s) ? "it" : "rm", big: BIG.has(s) }; }
        return { t: "text", str: w, st: "rm" };
      }
      if (c === "_" || c === "^") { const a = arg(); return { t: c === "_" ? "sub" : "sup", base: null, [c === "_" ? "sub" : "sup"]: a }; }
      if (/[A-Za-z]/.test(c)) {
        let w = c; while (i < src.length && /[A-Za-z]/.test(peek())) w += next();
        if (w.length === 1) return { t: "ch", ch: w, st: "it" };
        return { t: "text", str: w, st: "rm", fn: true };
      }
      if (GREEK_LOWER.test(c)) return { t: "ch", ch: c, st: "it" };
      return { t: "ch", ch: c === "-" ? "−" : c, st: "rm" };
    }
    function rawGroup() { while (peek() === " ") next(); if (peek() !== "{") return next(); next(); let s = ""; let depth = 1; while (i < src.length) { const c = next(); if (c === "{") depth++; if (c === "}") { depth--; if (!depth) break; } s += c; } return s; }
    const items = []; while (i < src.length) items.push(atom());
    return tighten(attach(items));
  }
  /* TeX ignores source spaces: drop implicit spaces next to relations, operators and punctuation, and at row ends. */
  function tighten(items) {
    const spaced = n => n && n.t === "ch" && (RELATIONS.has(n.ch) || BINOPS.has(n.ch) || n.ch === "," || n.ch === ";");
    const out = items.filter((it, i) => {
      if (it.t !== "space" || !it.implicit) return true;
      const prev = items.slice(0, i).reverse().find(n => n.t !== "space"), next = items.slice(i + 1).find(n => n.t !== "space");
      return !(spaced(prev) || (next && RELATIONS.has(next.ch) || (next && BINOPS.has(next.ch))) || !prev || !next);
    });
    for (const it of out) {
      if (it.t === "row") it.items = tighten(it.items);
      if (it.t === "frac") { it.num = tighten(it.num); it.den = tighten(it.den); }
      if (it.t === "script") { if (it.sub) it.sub = tighten(it.sub); if (it.sup) it.sup = tighten(it.sup); if (it.base && it.base.t === "row") it.base.items = tighten(it.base.items); }
    }
    return out;
  }
  /* Attach _ and ^ nodes to the preceding atom; merge sub+sup on the same base. */
  function attach(items) {
    const out = [];
    for (const it of items) {
      if (it.t === "row") it.items = attach(it.items);
      if (it.t === "frac") { it.num = attach(it.num); it.den = attach(it.den); }
      if ((it.t === "sub" || it.t === "sup") && it.base === null) {
        const prev = out[out.length - 1];
        if (prev && (prev.t === "script")) { prev[it.t] = attach(it[it.t]); continue; }
        if (prev && prev.t !== "space" && !(prev.t === "ch" && (RELATIONS.has(prev.ch) || BINOPS.has(prev.ch)))) { out.pop(); out.push({ t: "script", base: prev, sub: it.sub ? attach(it.sub) : null, sup: it.sup ? attach(it.sup) : null }); continue; }
        out.push({ t: "script", base: { t: "space", w: 0 }, sub: it.sub ? attach(it.sub) : null, sup: it.sup ? attach(it.sup) : null }); continue;
      }
      out.push(it);
    }
    return out;
  }

  /* a minus/plus after nothing, an opening bracket, a comma or a relation is unary: no operator spacing */
  function markUnary(items) {
    items.forEach((it, i) => { if (it.t === "ch" && (it.ch === "−" || it.ch === "+")) { const p = items.slice(0, i).reverse().find(n => n.t !== "space"); it.unary = !p || (p.t === "ch" && (/[(\[{,;=]/.test(p.ch) || RELATIONS.has(p.ch) || BINOPS.has(p.ch))); } });
  }

  // ------------------------------------------------------------ layout
  function font(size, st) { return `${st === "it" ? "italic " : ""}${st === "bf" ? "600 " : "400 "}${size}px ${SANS}`; }
  function measure(ctx, node, size) {
    const em = size;
    switch (node.t) {
      case "space": return { w: node.w * em, asc: 0, desc: 0 };
      case "ch": {
        const sz = node.big ? size * 1.3 : size; ctx.font = font(sz, node.st);
        let w = ctx.measureText(node.ch).width; const pad = node.st === "it" ? 0.03 * em : 0;
        const rel = RELATIONS.has(node.ch), bin = BINOPS.has(node.ch) && !node.unary;
        node._sz = sz; node._lead = rel ? 0.28 * em : bin ? 0.2 * em : 0; node._trail = rel ? 0.28 * em : bin ? 0.2 * em : (node.ch === "," || node.ch === ";") ? 0.16 * em : pad;
        return { w: w + node._lead + node._trail, asc: (node.big ? 0.95 : 0.75) * sz, desc: (node.big ? 0.3 : 0.22) * sz };
      }
      case "text": { ctx.font = font(size, node.st); const w = ctx.measureText(node.str).width; node._trail = node.fn ? 0.04 * em : 0; return { w: w + node._trail, asc: 0.75 * size, desc: 0.22 * size }; }
      case "row": return rowMetrics(ctx, node.items, size);
      case "script": {
        const b = measure(ctx, node.base, size), ss = size * 0.72;
        const sub = node.sub ? rowMetrics(ctx, node.sub, ss) : null, sup = node.sup ? rowMetrics(ctx, node.sup, ss) : null;
        const shiftSub = 0.2 * size, shiftSup = 0.36 * size;
        const w = b.w + Math.max(sub ? sub.w : 0, sup ? sup.w : 0) + (sub || sup ? 0.04 * size : 0);
        const asc = Math.max(b.asc, sup ? sup.asc + shiftSup : 0), desc = Math.max(b.desc, sub ? sub.desc + shiftSub : 0);
        node._m = { b, sub, sup, shiftSub, shiftSup }; return { w, asc, desc };
      }
      case "frac": {
        const fs = size * 0.92, n = rowMetrics(ctx, node.num, fs), d = rowMetrics(ctx, node.den, fs);
        const w = Math.max(n.w, d.w) + 0.3 * size, axis = 0.32 * size;
        node._m = { n, d, fs, axis, w };
        return { w, asc: axis + 0.12 * size + n.desc + n.asc, desc: -axis + 0.12 * size + d.asc + d.desc };
      }
    }
    return { w: 0, asc: 0, desc: 0 };
  }
  function rowMetrics(ctx, items, size) {
    let w = 0, asc = 0, desc = 0;
    markUnary(items);
    for (const it of items) { const m = measure(ctx, it, size); it._w = m.w; w += m.w; asc = Math.max(asc, m.asc); desc = Math.max(desc, m.desc); }
    return { w, asc, desc };
  }

  // ------------------------------------------------------------ draw
  function drawNode(ctx, node, x, y, size, color) {
    switch (node.t) {
      case "space": return;
      case "ch": {
        ctx.font = font(node._sz, node.st); ctx.fillStyle = color; ctx.textAlign = "left"; ctx.textBaseline = "alphabetic";
        ctx.fillText(node.ch, x + node._lead, y + (node.big ? 0.08 * size : 0));
        if (node.accent) { const w = ctx.measureText(node.ch).width; ctx.font = font(node._sz * 0.9, "rm"); ctx.textAlign = "center"; ctx.fillText(node.accent, x + node._lead + w * 0.55, y - 0.62 * node._sz); }
        return;
      }
      case "text": { ctx.font = font(size, node.st); ctx.fillStyle = color; ctx.textAlign = "left"; ctx.textBaseline = "alphabetic"; ctx.fillText(node.str, x, y); return; }
      case "row": { drawRow(ctx, node.items, x, y, size, color); return; }
      case "script": {
        const { b, sub, sup, shiftSub, shiftSup } = node._m; drawNode(ctx, node.base, x, y, size, color);
        const sx = x + b.w + 0.02 * size, ss = size * 0.72;
        if (sup) drawRow(ctx, node.sup, sx, y - shiftSup, ss, color);
        if (sub) drawRow(ctx, node.sub, sx, y + shiftSub, ss, color);
        return;
      }
      case "frac": {
        const { n, d, fs, axis, w } = node._m;
        drawRow(ctx, node.num, x + (w - n.w) / 2, y - axis - 0.12 * size - n.desc, fs, color);
        drawRow(ctx, node.den, x + (w - d.w) / 2, y - axis + 0.12 * size + d.asc, fs, color);
        ctx.strokeStyle = color; ctx.lineWidth = Math.max(1, size * 0.05); ctx.beginPath(); ctx.moveTo(x + 0.06 * size, y - axis); ctx.lineTo(x + w - 0.06 * size, y - axis); ctx.stroke();
        return;
      }
    }
  }
  function drawRow(ctx, items, x, y, size, color) { let cx = x; for (const it of items) { drawNode(ctx, it, cx, y, size, color); cx += it._w; } }

  const cache = new Map();
  function layout(ctx, src, size) {
    const key = src + "|" + size; let tree = cache.get(key);
    if (!tree) { tree = parse(src); cache.set(key, tree); if (cache.size > 2000) cache.clear(); }
    const m = rowMetrics(ctx, tree, size); return { tree, ...m };
  }
  /* Draw `src` with its baseline at y; returns the width.  o: size, color, align (left|center|right), alpha. */
  function draw(ctx, src, x, y, o = {}) {
    const size = o.size || 24, a = o.alpha === undefined ? 1 : o.alpha; if (a <= 0) return 0;
    ctx.save(); ctx.globalAlpha *= a;
    const L = layout(ctx, src, size);
    const x0 = o.align === "center" ? x - L.w / 2 : o.align === "right" ? x - L.w : x;
    drawRow(ctx, L.tree, x0, y, size, o.color || "#E6ECF4");
    ctx.restore(); return L.w;
  }
  function width(ctx, src, size) { return layout(ctx, src, size || 24).w; }

  // ------------------------------------------------------------ HTML output (prose formulas)
  const esc = str => str.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  function nodeHTML(node) {
    switch (node.t) {
      case "space": return node.w ? `<span class="mt-sp" style="width:${node.w.toFixed(2)}em"></span>` : "";
      case "ch": {
        const cls = ["mt-" + (node.st === "it" ? "i" : node.st === "bf" ? "b" : "r")];
        if (RELATIONS.has(node.ch)) cls.push("mt-rel"); else if (BINOPS.has(node.ch) && !node.unary) cls.push("mt-bin"); else if (node.ch === "," || node.ch === ";") cls.push("mt-pun");
        if (node.big) cls.push("mt-big");
        const base = `<span class="${cls.join(" ")}">${esc(node.ch)}</span>`;
        return node.accent ? `<span class="mt-acc">${base}<span class="mt-hat">${esc(node.accent)}</span></span>` : base;
      }
      case "text": return `<span class="mt-${node.st === "it" ? "i" : node.st === "bf" ? "b" : "r"} mt-txt">${esc(node.str)}</span>`;
      case "row": return rowHTML(node.items);
      case "script": {
        const b = nodeHTML(node.base), sub = node.sub ? rowHTML(node.sub) : null, sup = node.sup ? rowHTML(node.sup) : null;
        if (sub && sup) return `${b}<span class="mt-sup">${sup}</span><span class="mt-sub mt-after">${sub}</span>`;
        return b + (sup ? `<span class="mt-sup">${sup}</span>` : `<span class="mt-sub">${sub}</span>`);
      }
      case "frac": return `<span class="mt-frac"><span class="mt-num">${rowHTML(node.num)}</span><span class="mt-den">${rowHTML(node.den)}</span></span>`;
    }
    return "";
  }
  function rowHTML(items) { markUnary(items); return items.map(nodeHTML).join(""); }
  /* TeX-subset source → HTML string; o.display renders a centred block */
  function html(src, o) { o = o || {}; return `<span class="mt${o.display ? " mt-d" : ""}">${rowHTML(parse(src))}</span>`; }
  /* Replace every <span class="f">TeX</span> (class "f d" = display) inside root with typeset HTML. */
  function typesetAll(root) {
    (root || document).querySelectorAll("span.f").forEach(el => { const d = el.classList.contains("d"); const out = html(el.textContent, { display: d }); el.outerHTML = d ? out.replace(/^<span/, "<div").replace(/<\/span>$/, "</div>") : out; });
  }

  CPV.mathText = { draw, width, parse, html, typesetAll };
  if (CPV.D) CPV.D.math = draw;
})();
