/* Engine: the Stage (canvas pair + tweens + draw kit) and the Player (chapters,
 * beats, narration audio, language, keyboard).
 *
 * A scene is a plain object registered in CPV.scenes[id]:
 *   { setup(stage)   -- called once; build state, sample paths, etc.
 *     enter(stage)   -- called each time the chapter becomes active
 *     leave(stage)   -- optional
 *     beats: [fn, …] -- fn(stage, player) starts the animation of narration beat j
 *     draw(stage, t, dt) -- paints the 2D overlay each frame (ctx in a 1600x900 frame)
 *     useGL: bool, glRect: {x,y,w,h} -- when the WebGL line field is used
 *     onPointer(stage, type, x, y) -- optional interaction }
 */
(function () {
  const CPV = window.CPV = window.CPV || {};
  CPV.scenes = CPV.scenes || {};

  // ------------------------------------------------------------ palette & fonts (shared with the film)
  CPV.C = {
    bg: "#0B1119", panel: "#121A26", ink: "#E6ECF4", muted: "#8A97AC", dim: "#55627A", grid: "#1C2634",
    latent: "#4ECDC4", target: "#F5B841", A: "#5B8FF9", B: "#E15C9C", expl: "#5AD469", resid: "#4A5568",
    alert: "#FF6B6B", gold: "#FFD166", worldP: "#4ECDC4", worldM: "#F08A4B",
  };
  CPV.FONT = {
    sans: "'IBM Plex Sans', 'PingFang SC', 'Hiragino Sans GB', 'Noto Sans CJK SC', 'Microsoft YaHei', system-ui, sans-serif",
    mono: "'IBM Plex Mono', Menlo, Consolas, monospace",
    display: "'Fraunces', 'Iowan Old Style', Georgia, serif",
  };
  CPV.easing = {
    linear: t => t,
    inOut: t => (t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2),
    out: t => 1 - Math.pow(1 - t, 3),
    in: t => t * t * t,
    back: t => 1 + 2.2 * Math.pow(t - 1, 3) + 1.2 * Math.pow(t - 1, 2),
  };
  CPV.motion = window.CPV_SHEET ? 0.0005 : (window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches ? 0.3 : 1);
  CPV.lerp = (a, b, t) => a + (b - a) * t;
  CPV.clamp = (x, a, b) => Math.max(a, Math.min(b, x));
  CPV.hexToRgb = hex => { const n = parseInt(hex.slice(1), 16); return [(n >> 16) & 255, (n >> 8) & 255, n & 255]; };
  CPV.rgba = (hex, a) => { const [r, g, b] = CPV.hexToRgb(hex); return `rgba(${r},${g},${b},${a})`; };
  CPV.glColor = (hex, a = 1) => { const [r, g, b] = CPV.hexToRgb(hex); return [r / 255, g / 255, b / 255, a]; };
  CPV.mix = (h1, h2, t) => { const a = CPV.hexToRgb(h1), b = CPV.hexToRgb(h2); return `rgb(${a.map((v, i) => Math.round(v + (b[i] - v) * t)).join(",")})`; };
  CPV.fmt = (x, d = 3) => (x === undefined || x === null || Number.isNaN(x)) ? "–" : x.toFixed(d);

  // ------------------------------------------------------------ draw kit (logical 1600 x 900 frame)
  const D = CPV.D = {};
  D.font = (size, family = "sans", weight = 400) => `${weight} ${size}px ${CPV.FONT[family] || family}`;
  /* D.text-compatible signature for typeset labels (used by heatmap/gauge titles) */
  D.mathTitle = function (ctx, src, x, y, o) { o = o || {}; const size = o.size || 15; CPV.mathText.draw(ctx, src, x, y + (o.baseline === "middle" ? size * 0.35 : 0), { size, color: o.color, align: o.align, alpha: o.alpha }); };
  D.text = (ctx, str, x, y, o = {}) => {
    if (o.alpha !== undefined && o.alpha <= 0) return;
    ctx.save();
    ctx.globalAlpha = (o.alpha === undefined ? 1 : o.alpha) * (ctx.globalAlpha || 1);
    ctx.font = D.font(o.size || 24, o.font || "sans", o.weight || 400);
    ctx.fillStyle = o.color || CPV.C.ink;
    ctx.textAlign = o.align || "left";
    ctx.textBaseline = o.baseline || "alphabetic";
    if (o.letterSpacing && "letterSpacing" in ctx) ctx.letterSpacing = o.letterSpacing;
    ctx.fillText(str, x, y);
    ctx.restore();
  };
  D.measure = (ctx, str, o = {}) => { ctx.save(); ctx.font = D.font(o.size || 24, o.font || "sans", o.weight || 400); const w = ctx.measureText(str).width; ctx.restore(); return w; };
  /* Word-wrapped paragraph; returns the number of lines drawn. CJK text wraps per character. */
  D.paragraph = (ctx, str, x, y, maxW, o = {}) => {
    const size = o.size || 24, lh = o.lineHeight || size * 1.4;
    ctx.save(); ctx.font = D.font(size, o.font || "sans", o.weight || 400);
    const cjk = /[　-鿿＀-￯]/.test(str);
    const tokens = cjk ? Array.from(str) : str.split(/(\s+)/);
    const lines = []; let cur = "";
    for (const tk of tokens) {
      const test = cur + tk;
      if (ctx.measureText(test).width > maxW && cur) { lines.push(cur.trimEnd()); cur = cjk ? tk : tk.trimStart(); }
      else cur = test;
    }
    if (cur.trim()) lines.push(cur.trimEnd());
    ctx.restore();
    lines.forEach((ln, i) => D.text(ctx, ln, x, y + i * lh, o));
    return lines.length;
  };
  D.line = (ctx, x1, y1, x2, y2, o = {}) => {
    if (o.alpha !== undefined && o.alpha <= 0) return;
    ctx.save(); if (o.alpha !== undefined) ctx.globalAlpha *= o.alpha;
    ctx.strokeStyle = o.color || CPV.C.muted; ctx.lineWidth = o.width || 2; ctx.lineCap = o.cap || "round";
    if (o.dash) ctx.setLineDash(o.dash);
    ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke(); ctx.restore();
  };
  D.poly = (ctx, pts, o = {}) => {
    if (pts.length < 2 || (o.alpha !== undefined && o.alpha <= 0)) return;
    ctx.save(); if (o.alpha !== undefined) ctx.globalAlpha *= o.alpha;
    ctx.strokeStyle = o.color || CPV.C.ink; ctx.lineWidth = o.width || 2; ctx.lineJoin = "round"; ctx.lineCap = "round";
    if (o.dash) ctx.setLineDash(o.dash);
    ctx.beginPath(); pts.forEach((p, i) => (i ? ctx.lineTo(p[0], p[1]) : ctx.moveTo(p[0], p[1])));
    if (o.fill) { ctx.fillStyle = o.fill; ctx.fill(); }
    if (!o.noStroke) ctx.stroke();
    ctx.restore();
  };
  D.rect = (ctx, x, y, w, h, o = {}) => {
    if (o.alpha !== undefined && o.alpha <= 0) return;
    ctx.save(); if (o.alpha !== undefined) ctx.globalAlpha *= o.alpha;
    const r = o.radius || 0;
    ctx.beginPath();
    if (r) { ctx.moveTo(x + r, y); ctx.arcTo(x + w, y, x + w, y + h, r); ctx.arcTo(x + w, y + h, x, y + h, r); ctx.arcTo(x, y + h, x, y, r); ctx.arcTo(x, y, x + w, y, r); ctx.closePath(); }
    else ctx.rect(x, y, w, h);
    if (o.fill) { ctx.fillStyle = o.fill; ctx.fill(); }
    if (o.stroke) { ctx.strokeStyle = o.stroke; ctx.lineWidth = o.width || 1.5; if (o.dash) ctx.setLineDash(o.dash); ctx.stroke(); }
    ctx.restore();
  };
  D.circle = (ctx, x, y, r, o = {}) => {
    if (o.alpha !== undefined && o.alpha <= 0) return;
    ctx.save(); if (o.alpha !== undefined) ctx.globalAlpha *= o.alpha;
    ctx.beginPath(); ctx.arc(x, y, r, 0, Math.PI * 2);
    if (o.fill) { ctx.fillStyle = o.fill; ctx.fill(); }
    if (o.stroke) { ctx.strokeStyle = o.stroke; ctx.lineWidth = o.width || 1.5; ctx.stroke(); }
    ctx.restore();
  };
  D.glow = (ctx, x, y, r, color, alpha = 0.5) => {
    ctx.save(); const g = ctx.createRadialGradient(x, y, 0, x, y, r);
    g.addColorStop(0, CPV.rgba(color, alpha)); g.addColorStop(1, CPV.rgba(color, 0));
    ctx.fillStyle = g; ctx.beginPath(); ctx.arc(x, y, r, 0, Math.PI * 2); ctx.fill(); ctx.restore();
  };
  D.arrow = (ctx, x1, y1, x2, y2, o = {}) => {
    if (o.alpha !== undefined && o.alpha <= 0) return;
    const head = o.head || 10, ang = Math.atan2(y2 - y1, x2 - x1);
    D.line(ctx, x1, y1, x2, y2, o);
    ctx.save(); if (o.alpha !== undefined) ctx.globalAlpha *= o.alpha;
    ctx.fillStyle = o.color || CPV.C.muted; ctx.beginPath(); ctx.moveTo(x2, y2);
    ctx.lineTo(x2 - head * Math.cos(ang - 0.45), y2 - head * Math.sin(ang - 0.45));
    ctx.lineTo(x2 - head * Math.cos(ang + 0.45), y2 - head * Math.sin(ang + 0.45)); ctx.closePath(); ctx.fill(); ctx.restore();
  };
  /* Value gauge: a 0..1 bar with label and read-out (the film's ValueGauge). */
  D.gauge = (ctx, x, y, w, h, value, o = {}) => {
    const a = o.alpha === undefined ? 1 : o.alpha; if (a <= 0) return;
    const col = o.color || CPV.C.expl;
    D.rect(ctx, x, y, w, h, { stroke: CPV.C.dim, width: 1.5, radius: 4, alpha: a });
    for (let i = 1; i < 4; i++) D.line(ctx, x + w * i / 4, y + h, x + w * i / 4, y + h + 5, { color: CPV.C.dim, width: 1, alpha: a });
    D.rect(ctx, x + 2, y + 2, Math.max(0, (w - 4) * CPV.clamp(value, 0, 1)), h - 4, { fill: col, radius: 3, alpha: a });
    if (o.label && o.labelMath) CPV.mathText.draw(ctx, o.label, x - 14, y + h / 2 + (o.labelSize || 22) * 0.35, { size: o.labelSize || 22, color: o.labelColor || col, align: "right", alpha: a });
    else if (o.label) D.text(ctx, o.label, x - 14, y + h / 2, { size: o.labelSize || 22, color: o.labelColor || col, align: "right", baseline: "middle", font: o.labelFont || "sans", alpha: a });
    if (o.readout !== false) D.text(ctx, CPV.fmt(value, o.digits === undefined ? 3 : o.digits), x + w + 14, y + h / 2, { size: o.readSize || 22, color: col, font: "mono", baseline: "middle", alpha: a });
  };
  /* Heat map of a square matrix; cellAlpha(j,k) optionally dims cells; cellStroke(j,k) outlines. */
  D.heatmap = (ctx, Mtx, x, y, size, o = {}) => {
    const p = Mtx.length, cell = size / p, lo = o.lo === undefined ? 0 : o.lo, hi = o.hi === undefined ? 1 : o.hi;
    const c0 = o.colorLo || "#13202C", c1 = o.colorHi || CPV.C.latent, cn = o.colorNeg || CPV.C.worldM;
    const a = o.alpha === undefined ? 1 : o.alpha; if (a <= 0) return;
    ctx.save(); ctx.globalAlpha *= a;
    for (let j = 0; j < p; j++) for (let k = 0; k < p; k++) {
      const v = Mtx[j][k];
      let col;
      if (v < 0 && o.signed) col = CPV.mix(c0, cn, CPV.clamp(-v / Math.max(1e-9, hi), 0, 1));
      else col = CPV.mix(c0, c1, CPV.clamp((v - lo) / Math.max(1e-9, hi - lo), 0, 1));
      const ca = o.cellAlpha ? o.cellAlpha(j, k) : 1;
      ctx.globalAlpha = a * ca;
      ctx.fillStyle = col; ctx.fillRect(x + k * cell, y + j * cell, cell + 0.5, cell + 0.5);
    }
    ctx.globalAlpha = a;
    if (o.cellStroke) for (let j = 0; j < p; j++) for (let k = 0; k < p; k++) { const s = o.cellStroke(j, k); if (s) { ctx.strokeStyle = s; ctx.lineWidth = 1.5; ctx.strokeRect(x + k * cell + 0.75, y + j * cell + 0.75, cell - 1.5, cell - 1.5); } }
    if (o.frame !== false) { ctx.strokeStyle = o.frameColor || CPV.C.dim; ctx.lineWidth = 1; ctx.strokeRect(x - 0.5, y - 0.5, size + 1, size + 1); }
    ctx.restore();
    if (o.title) (o.titleMath ? D.mathTitle : D.text)(ctx, o.title, x + size / 2, y - 14, { size: o.titleSize || 22, color: o.titleColor || CPV.C.muted, align: "center", alpha: a, font: o.titleFont || "mono" });
  };
  /* Horizontal time axis with optional hour ticks. */
  D.axis = (ctx, x, y, w, o = {}) => {
    const a = o.alpha === undefined ? 1 : o.alpha; if (a <= 0) return;
    D.line(ctx, x, y, x + w, y, { color: o.color || CPV.C.dim, width: o.width || 1.5, alpha: a });
    (o.ticks || []).forEach(tk => {
      const tx = x + w * tk.u;
      D.line(ctx, tx, y, tx, y + 7, { color: o.color || CPV.C.dim, width: 1.5, alpha: a });
      if (tk.label !== undefined) D.text(ctx, tk.label, tx, y + 26, { size: o.tickSize || 18, color: o.tickColor || CPV.C.muted, align: "center", font: "mono", alpha: a });
    });
    if (o.label) D.text(ctx, o.label, x + w, y + (o.labelDy || 52), { size: 18, color: CPV.C.muted, align: "right", alpha: a });
  };
  /* Rounded label chip. */
  D.chip = (ctx, str, x, y, o = {}) => {
    const size = o.size || 18, padX = o.padX || 12, h = size + 14;
    const w = D.measure(ctx, str, { size, font: o.font || "sans", weight: 500 }) + padX * 2;
    const a = o.alpha === undefined ? 1 : o.alpha; if (a <= 0) return 0;
    const col = o.color || CPV.C.muted;
    const left = o.align === "center" ? x - w / 2 : o.align === "right" ? x - w : x;
    D.rect(ctx, left, y - h / 2, w, h, { fill: o.fill || CPV.rgba(col, 0.14), stroke: o.stroke === false ? null : CPV.rgba(col, 0.6), radius: h / 2, alpha: a });
    D.text(ctx, str, left + w / 2, y, { size, color: o.textColor || col, align: "center", baseline: "middle", weight: 500, font: o.font || "sans", alpha: a });
    return w;
  };
  /* Boxed one-line conclusion (the film's `takeaway`). */
  D.takeaway = (ctx, str, cx, cy, o = {}) => {
    const a = o.alpha === undefined ? 1 : o.alpha; if (a <= 0) return;
    const size = o.size || 28, col = o.color || CPV.C.gold, maxW = o.maxW || 1100;
    ctx.save(); ctx.font = D.font(size, "sans", 500);
    let w = Math.min(maxW, ctx.measureText(str).width + 64); ctx.restore();
    const lines = D.paragraph(ctx, str, 0, -9999, maxW - 64, { size, weight: 500, alpha: 0 });
    const h = lines * size * 1.35 + 30;
    D.rect(ctx, cx - w / 2, cy - h / 2, w, h, { stroke: col, width: 2, radius: 10, fill: CPV.rgba(col, 0.06), alpha: a });
    D.paragraph(ctx, str, cx, cy - h / 2 + 15 + size * 1.05, maxW - 64, { size, weight: 500, color: col, align: "center", alpha: a, lineHeight: size * 1.35 });
  };
  /* A small numbered step marker. */
  D.badge = (ctx, n, x, y, o = {}) => {
    const col = o.color || CPV.C.gold, a = o.alpha === undefined ? 1 : o.alpha;
    D.circle(ctx, x, y, o.r || 16, { fill: CPV.rgba(col, 0.15), stroke: col, width: 1.5, alpha: a });
    D.text(ctx, String(n), x, y + 1, { size: (o.r || 16) * 1.1, color: col, align: "center", baseline: "middle", font: "mono", weight: 500, alpha: a });
  };

  // ------------------------------------------------------------ Stage
  class Stage {
    constructor(el) {
      this.el = el; this.W = 1600; this.H = 900;
      this.glCanvas = document.createElement("canvas"); this.glCanvas.className = "gl";
      this.ovCanvas = document.createElement("canvas"); this.ovCanvas.className = "ov";
      el.appendChild(this.glCanvas); el.appendChild(this.ovCanvas);
      this.ctx = this.ovCanvas.getContext("2d");
      try { this.field = new CPV.LineField(this.glCanvas); } catch (e) { console.warn("WebGL unavailable", e); this.field = { ok: false, layers: [], draw() {}, clearLayers() {} }; }
      this.tweens = []; this.timers = []; this.scene = null; this.t = 0; this.last = 0; this.running = true;
      this.pointer = { x: 0, y: 0, down: false };
      this.resize();
      new ResizeObserver(() => this.resize()).observe(el);
      const toLocal = e => { const r = el.getBoundingClientRect(); return [(e.clientX - r.left - this.ox) / this.s, (e.clientY - r.top - this.oy) / this.s]; };
      const fwd = type => e => {
        const [x, y] = toLocal(e); this.pointer.x = x; this.pointer.y = y;
        if (type === "down") { this.pointer.down = true; el.setPointerCapture && el.setPointerCapture(e.pointerId); }
        if (type === "up") this.pointer.down = false;
        if (this.scene && this.scene.onPointer) { const used = this.scene.onPointer(this, type, x, y, e); if (used && type === "down") e.preventDefault(); }
      };
      el.addEventListener("pointerdown", fwd("down")); el.addEventListener("pointermove", fwd("move"));
      el.addEventListener("pointerup", fwd("up")); el.addEventListener("pointercancel", fwd("up")); el.addEventListener("pointerleave", fwd("leave"));
      requestAnimationFrame(ts => this.loop(ts));
    }
    loop(ts) { requestAnimationFrame(t2 => this.loop(t2)); this.frame(ts); }
    resize() {
      const r = this.el.getBoundingClientRect(); if (r.width === 0) return;
      const dpr = Math.min(2, window.devicePixelRatio || 1);
      this.dpr = dpr; this.cssW = r.width; this.cssH = r.height;
      for (const c of [this.glCanvas, this.ovCanvas]) { c.width = Math.round(r.width * dpr); c.height = Math.round(r.height * dpr); c.style.width = r.width + "px"; c.style.height = r.height + "px"; }
      this.s = Math.min(r.width / this.W, r.height / this.H);
      this.ox = (r.width - this.W * this.s) / 2; this.oy = (r.height - this.H * this.s) / 2;
    }
    glViewport(rect) {
      const dpr = this.dpr;
      return { x: Math.round((this.ox + rect.x * this.s) * dpr), y: Math.round((this.cssH - (this.oy + (rect.y + rect.h) * this.s)) * dpr),
               w: Math.round(rect.w * this.s * dpr), h: Math.round(rect.h * this.s * dpr) };
    }
    tween(obj, to, dur = 600, o = {}) {
      const keys = Object.keys(to);
      // a new tween takes over only the keys it shares with running tweens; their other keys keep animating
      this.tweens.forEach(tw => { if (tw.obj === obj) tw.keys = tw.keys.filter(k => !keys.includes(k)); });
      this.tweens = this.tweens.filter(tw => tw.keys.length > 0);
      const tw = { obj, keys, from: {}, to, dur: dur * CPV.motion, delay: (o.delay || 0) * CPV.motion, t: 0,
                   ease: typeof o.ease === "function" ? o.ease : (CPV.easing[o.ease || "inOut"]), onDone: o.onDone, onUpdate: o.onUpdate, done: false };
      keys.forEach(k => tw.from[k] = obj[k]);
      this.tweens.push(tw); return tw;
    }
    delay(ms, fn) {
      if (window.CPV_SHEET) { fn(); return 0; }
      const rec = { fn, left: ms * CPV.motion, id: 0 };
      rec.id = setTimeout(() => { this.timers = this.timers.filter(t => t !== rec); fn(); }, rec.left);
      this.timers.push(rec); return rec;
    }
    clearAnimations() { this.tweens.length = 0; this.timers.forEach(t => clearTimeout(t.id)); this.timers.length = 0; }
    /* Advance the running animation by `ms` of virtual time: tweens jump ahead, pending delays fire in order
     * (whatever they start is advanced by the remainder).  Used when the viewer seeks into the middle of a beat. */
    fastForward(ms) {
      let remaining = Math.max(0, ms);
      const advance = step => { if (step <= 0) return; this.tweens.forEach(tw => tw.t += step); this.timers.forEach(t => t.left -= step); this.t += step / 1000; };
      for (let guard = 0; guard < 200; guard++) {
        const due = this.timers.filter(t => t.left <= remaining).sort((a, b) => a.left - b.left)[0];
        if (!due) break;
        advance(due.left); remaining -= due.left;
        clearTimeout(due.id); this.timers = this.timers.filter(t => t !== due); due.fn();
      }
      advance(remaining);
      this.timers.forEach(t => { clearTimeout(t.id); t.id = setTimeout(() => { this.timers = this.timers.filter(x => x !== t); t.fn(); }, Math.max(0, t.left)); });
      if (this.last) this.frame(this.last);   // apply the advanced tween state immediately
    }
    setScene(scene) {
      if (this.scene === scene) return;
      if (this.scene && this.scene.leave) this.scene.leave(this);
      this.clearAnimations(); this.field.clearLayers();
      this.scene = scene;
      if (scene && !scene._ready) { scene.setup(this); scene._ready = true; }
      if (scene && scene.enter) scene.enter(this);
      this.applySceneGL();
    }
    /* Composite scenes switch their GL use mid-chapter; they call this after changing useGL/glRect. */
    applySceneGL() { this.glCanvas.style.visibility = this.scene && this.scene.useGL ? "visible" : "hidden"; }
    /* One frame of work; `loop` schedules it on requestAnimationFrame, the storyboard drives it manually. */
    frame(ts) {
      const dt = Math.max(0, Math.min(0.1, (ts - (this.last || ts)) / 1000)); this.last = ts; this.t += dt;
      for (const tw of this.tweens) {
        tw.t += dt * 1000; if (tw.t < tw.delay) continue;
        const u = tw.dur > 0 ? CPV.clamp((tw.t - tw.delay) / tw.dur, 0, 1) : 1, e = tw.ease(u);
        tw.keys.forEach(k => tw.obj[k] = tw.from[k] + (tw.to[k] - tw.from[k]) * e);
        if (tw.onUpdate) tw.onUpdate(u);
        if (u >= 1) { tw.done = true; if (tw.onDone) tw.onDone(); }
      }
      if (this.tweens.some(t => t.done)) this.tweens = this.tweens.filter(t => !t.done);
      if (!this.scene) return;
      const ctx = this.ctx;
      ctx.setTransform(1, 0, 0, 1, 0, 0); ctx.clearRect(0, 0, this.ovCanvas.width, this.ovCanvas.height);
      if (this.scene.useGL && this.field.ok) this.field.draw(this.scene.glRect ? this.glViewport(this.scene.glRect) : null);
      ctx.setTransform(this.dpr * this.s, 0, 0, this.dpr * this.s, this.ox * this.dpr, this.oy * this.dpr);
      ctx.globalAlpha = 1;
      this.scene.draw(this, this.t, dt);
    }
  }
  CPV.Stage = Stage;

  // ------------------------------------------------------------ Player
  class Player {
    constructor(o) {
      this.stage = o.stage; this.N = o.narration; this.els = o.els;
      this.lang = o.lang || "en"; this.chapter = -1; this.beat = -1; this.playing = false; this.autoplay = true;
      this.sound = true; this.readTitle = o.readTitle || { en: "Read: this chapter in the paper", zh: "读一读：这一章在论文里" };
      this.audio = new Audio(); this.audio.preload = "auto";
      this.audio.addEventListener("ended", () => this.onBeatEnd());
      this.audio.addEventListener("error", () => { if (this.audio.getAttribute("src")) this.fallbackSpeak(); });
      this._tok = 0;
      this.fallbackTimer = null; this.synthUtter = null;
      this.buildNav();
      this.bindUI();
      this.buildTimeline();
      this.audio.addEventListener("timeupdate", () => this.updateTimeline());
      setInterval(() => { if (this.playing) this.updateTimeline(); }, 200);
      // deep links: #ch3, #ch3-b2 (technical) or #e3, #e3-b1 (explainer) — chapter ids come from the narration set
      const h = location.hash.match(/^#([a-z]+\d+)(?:-b(\d+))?/);
      const idx = h ? this.chapters.findIndex(c => c.id === h[1]) : -1;
      this.gotoChapter(idx >= 0 ? idx : 0, { beat: idx >= 0 ? (+h[2] || 0) : 0, play: false });
    }
    get chapters() { return this.N.chapters; }
    get scene() { return CPV.scenes[this.chapters[this.chapter].id]; }
    t(obj) { return obj[this.lang] || obj.en; }
    buildNav() {
      const ol = this.els.nav; ol.innerHTML = "";
      this.chapters.forEach((ch, i) => {
        const li = document.createElement("li");
        li.innerHTML = `<button type="button" data-ch="${i}"><span class="idx">${i === 0 ? "◦" : i}</span><span class="ttl"></span><span class="dur"></span></button>`;
        li.querySelector("button").addEventListener("click", () => this.gotoChapter(i, { play: this.playing || true }));
        ol.appendChild(li);
      });
      this.refreshNavText();
    }
    refreshNavText() {
      this.chapters.forEach((ch, i) => {
        const li = this.els.nav.children[i];
        li.querySelector(".ttl").textContent = this.t(ch.title);
        const secs = ch.beats.reduce((s, b) => s + ((b.dur && b.dur[this.lang]) || 0), 0);
        li.querySelector(".dur").textContent = secs ? `${Math.round(secs / 60)}:${String(Math.round(secs % 60)).padStart(2, "0")}` : "";
        li.classList.toggle("active", i === this.chapter);
      });
    }
    bindUI() {
      const e = this.els;
      e.play.addEventListener("click", () => this.togglePlay());
      e.prev.addEventListener("click", () => this.prev());
      e.next.addEventListener("click", () => this.next());
      e.lang.querySelectorAll("button").forEach(b => b.addEventListener("click", () => this.setLang(b.dataset.lang)));
      e.autoplay.addEventListener("change", () => { this.autoplay = e.autoplay.checked; });
      if (e.sound) e.sound.addEventListener("click", () => this.setSound(!this.sound));
      document.addEventListener("keydown", ev => {
        if (ev.target && /input|textarea|select/i.test(ev.target.tagName)) return;
        if (ev.code === "Space") { ev.preventDefault(); this.togglePlay(); }
        else if (ev.code === "ArrowRight") { ev.preventDefault(); this.next(); }
        else if (ev.code === "ArrowLeft") { ev.preventDefault(); this.prev(); }
        else if (ev.code === "PageDown" || (ev.code === "ArrowDown" && ev.shiftKey)) { ev.preventDefault(); this.gotoChapter(this.chapter + 1, { play: this.playing }); }
        else if (ev.code === "PageUp" || (ev.code === "ArrowUp" && ev.shiftKey)) { ev.preventDefault(); this.gotoChapter(this.chapter - 1, { play: this.playing }); }
        else if (ev.key === "l" || ev.key === "L") this.setLang(this.lang === "en" ? "zh" : "en");
        else if (ev.key === "m" || ev.key === "M") this.setSound(!this.sound);
      });
    }
    /* Narration can be switched off: beats then advance on a timer (the clip's duration if known). */
    setSound(on) {
      this.sound = on;
      this.updateSoundLabel();
      if (this.playing) this.speak();
    }
    updateSoundLabel() {
      const on = this.sound;
      if (this.els.sound) { this.els.sound.setAttribute("aria-pressed", on ? "true" : "false"); this.els.sound.textContent = on ? (this.lang === "zh" ? "🔊 有解说" : "🔊 Narration on") : (this.lang === "zh" ? "🔇 静音" : "🔇 Narration off"); }
    }
    setLang(lang) {
      if (lang === this.lang) return;
      this.lang = lang; document.documentElement.lang = lang === "zh" ? "zh-CN" : "en";
      this.els.lang.querySelectorAll("button").forEach(b => b.classList.toggle("active", b.dataset.lang === lang));
      this.refreshNavText(); this.renderChapterText(); this.updateSoundLabel(); this.buildTimeline();
      if (this.beat >= 0) { this.renderTranscript(); if (this.playing) this.speak(); }   // one restart in the new language
    }
    gotoChapter(i, o = {}) {
      i = CPV.clamp(i, 0, this.chapters.length - 1);
      this.stopAudio();
      this.chapter = i; this.beat = -1;
      const scene = this.scene;
      this.stage.setScene(scene);
      this.refreshNavText(); this.renderChapterText();
      history.replaceState(null, "", `#${this.chapters[i].id}`);
      const beat = o.beat || 0;
      if (o.play) { this.playing = true; this.playBeat(beat); }
      else { this.playing = false; this.showBeat(beat, { silent: true }); }
      this.updateTransport();
    }
    renderChapterText() {
      const ch = this.chapters[this.chapter];
      this.els.kicker.textContent = this.t(ch.kicker);
      this.els.title.textContent = this.t(ch.title);
      this.els.read.innerHTML = this.t(ch.read);
      if (CPV.mathText) CPV.mathText.typesetAll(this.els.read);
      this.els.readTitle.textContent = this.t(this.readTitle);
    }
    // ------------------------------------------------------------ timeline (video-style progress bar over the whole tour)
    /* Chapter segments with gaps, beat ticks, a draggable play-head, hover tooltip and chapter labels. */
    timelineModel() {
      let t = 0; const chapters = this.chapters.map((ch, i) => {
        const beats = ch.beats.map(b => { const d = (b.dur && b.dur[this.lang]) || 8; const rec = { start: t, dur: d }; t += d; return rec; });
        return { i, id: ch.id, title: this.t(ch.title), start: beats[0].start, dur: t - beats[0].start, beats };
      });
      return { total: t, chapters };
    }
    fmt(s) { s = Math.max(0, Math.round(s)); return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`; }
    buildTimeline() {
      const tl = this.els.timeline; if (!tl) return;
      const m = this.tl = this.timelineModel();
      const track = tl.querySelector(".tl-track"), labels = tl.querySelector(".tl-labels");
      track.querySelectorAll(".tl-seg").forEach(n => n.remove()); labels.innerHTML = "";
      const knob = track.querySelector(".tl-knob"), tip = track.querySelector(".tl-tip");
      m.chapters.forEach(ch => {
        const seg = document.createElement("div"); seg.className = "tl-seg"; seg.style.left = (ch.start / m.total * 100) + "%"; seg.style.width = (ch.dur / m.total * 100) + "%";
        seg.innerHTML = `<div class="tl-fill"></div>` + ch.beats.slice(1).map(b => `<i class="tl-tick" style="left:${((b.start - ch.start) / ch.dur * 100).toFixed(2)}%"></i>`).join("");
        track.insertBefore(seg, knob);
        const lab = document.createElement("button"); lab.type = "button"; lab.className = "tl-label"; lab.style.left = (ch.start / m.total * 100) + "%"; lab.style.width = (ch.dur / m.total * 100) + "%";
        lab.innerHTML = `<span class="n">${ch.i === 0 ? "◦" : ch.i}</span><span class="ttl">${ch.title}</span>`; lab.title = ch.title;
        lab.addEventListener("click", () => this.gotoChapter(ch.i, { play: true }));
        labels.appendChild(lab);
      });
      if (!this._tlBound) {
        this._tlBound = true;
        const frac = ev => { const r = track.getBoundingClientRect(); return CPV.clamp((ev.clientX - r.left) / r.width, 0, 1); };
        const preview = (f, show) => { const t = f * this.tl.total, loc = this.locate(t); tip.style.left = (f * 100) + "%"; tip.innerHTML = `<b>${loc.chapter.title}</b><span>${this.fmt(t)}</span>`; tip.classList.toggle("show", show); };
        track.addEventListener("pointerdown", ev => { ev.preventDefault(); this._scrub = true; track.setPointerCapture(ev.pointerId); tl.classList.add("scrubbing"); this._scrubFrac = frac(ev); preview(this._scrubFrac, true); this.updateTimeline(); });
        track.addEventListener("pointermove", ev => { const f = frac(ev); if (this._scrub) { this._scrubFrac = f; this.updateTimeline(); } preview(f, true); });
        track.addEventListener("pointerleave", () => { if (!this._scrub) tip.classList.remove("show"); });
        const release = ev => { if (!this._scrub) return; this._scrub = false; tl.classList.remove("scrubbing"); tip.classList.remove("show"); this.seekTo(this._scrubFrac * this.tl.total); };
        track.addEventListener("pointerup", release); track.addEventListener("pointercancel", release);
        const ro = new ResizeObserver(() => this.layoutTimelineLabels()); ro.observe(tl);
      }
      this.layoutTimelineLabels(); this.updateTimeline();
    }
    layoutTimelineLabels() { const tl = this.els.timeline; if (!tl) return; tl.querySelectorAll(".tl-label").forEach(l => l.classList.toggle("narrow", l.getBoundingClientRect().width < 72)); }
    /* absolute tour time -> {chapter, beat, offset (s)} */
    locate(t) {
      const m = this.tl || this.timelineModel(); t = CPV.clamp(t, 0, m.total - 0.001);
      const chapter = m.chapters.find(c => t < c.start + c.dur) || m.chapters[m.chapters.length - 1];
      let bi = chapter.beats.findIndex(b => t < b.start + b.dur); if (bi < 0) bi = chapter.beats.length - 1;
      return { chapter, beat: bi, offset: t - chapter.beats[bi].start };
    }
    /* current absolute position in the tour (seconds) */
    currentPos() {
      if (!this.tl || this.chapter < 0 || this.beat < 0) return 0;
      const ch = this.tl.chapters[this.chapter], b = ch.beats[this.beat]; if (!b) return ch.start;
      const within = this.sound ? (this.audio.currentTime || 0) : ((this._beatOff || 0) + (this._beatStart ? (performance.now() - this._beatStart) : 0)) / 1000;
      return Math.min(b.start + b.dur, b.start + (this.playing || this.sound ? within : (this._beatOff || 0) / 1000));
    }
    updateTimeline() {
      const tl = this.els.timeline; if (!tl || !this.tl) return;
      const m = this.tl, pos = this._scrub ? this._scrubFrac * m.total : this.currentPos();
      tl.querySelectorAll(".tl-seg").forEach((seg, i) => { const ch = m.chapters[i]; const f = CPV.clamp((pos - ch.start) / ch.dur, 0, 1); seg.querySelector(".tl-fill").style.width = (f * 100) + "%"; seg.classList.toggle("current", i === this.chapter); });
      tl.querySelector(".tl-knob").style.left = (pos / m.total * 100) + "%";
      tl.querySelectorAll(".tl-label").forEach((l, i) => l.classList.toggle("current", i === this.chapter));
      if (this.els.time) this.els.time.textContent = `${this.fmt(pos)} / ${this.fmt(m.total)}`;
    }
    /* Jump to an absolute time: chapter + beat, then fast-forward the animation and the clip by the offset. */
    seekTo(t) {
      const loc = this.locate(t), wasPlaying = this.playing, off = Math.max(0, loc.offset - 0.25);
      this._seekOff = off > 0.5 ? off * 1000 : 0; this._ffMs = this._seekOff;
      if (loc.chapter.i !== this.chapter) this.gotoChapter(loc.chapter.i, { beat: loc.beat, play: wasPlaying });
      else if (wasPlaying) this.playBeat(loc.beat);
      else { this.stopAudio(); this.showBeat(loc.beat, { silent: true }); this._beatOff = this._seekOff; this.updateTransport(); }
      this.updateTimeline();
    }
    renderTranscript() {
      const ch = this.chapters[this.chapter], b = ch.beats[this.beat];
      this.els.transcript.textContent = b ? this.t(b) : "";
      if (this.els.counter) this.els.counter.textContent = `${this.beat + 1} / ${ch.beats.length}`;
      history.replaceState(null, "", `#${ch.id}-b${this.beat}`);
      this.updateTimeline();
    }
    /* Run the scene's beat animation and (unless silent) the narration. */
    showBeat(j, o = {}) {
      const ch = this.chapters[this.chapter];
      j = CPV.clamp(j, 0, ch.beats.length - 1);
      this.beat = j;
      const scene = this.scene;
      if (scene && scene.beats && scene.beats[j]) {
        try { scene.beats[j].call(scene, this.stage, this); } catch (err) { console.error("beat error", ch.id, j, err); }
      }
      if (this._ffMs > 0) { this.stage.fastForward(this._ffMs); this._ffMs = 0; }
      this.renderTranscript();
      if (!o.silent) this.speak();
    }
    playBeat(j) { this.playing = true; this.showBeat(j); this.updateTransport(); }
    speak() {
      this.stopAudio();
      const ch = this.chapters[this.chapter], b = ch.beats[this.beat], key = `${ch.id}-${b.id}`;
      const off = this._seekOff || 0; this._seekOff = 0;
      this._beatStart = performance.now(); this._beatOff = off;
      if (!this.sound) {   // silent mode: hold each beat for its clip length (or a reading-speed estimate)
        const text = this.t(b), ms = (b.dur && b.dur[this.lang]) ? b.dur[this.lang] * 1000 : 1200 + (this.lang === "zh" ? text.length * 220 : text.split(/\s+/).length * 380);
        this.fallbackTimer = setTimeout(() => this.onBeatEnd(), Math.max(50, ms - off)); return;
      }
      const base = window.CPV_AUDIO_BASE || "audio/";
      const src = (window.CPV_AUDIO && CPV_AUDIO[this.lang] && CPV_AUDIO[this.lang][key]) || `${base}${this.lang}/${key}.m4a`;
      this.audio.src = src;
      const tok = ++this._tok;
      if (off > 0) {
        const onMeta = () => { this.audio.removeEventListener("loadedmetadata", onMeta); try { this.audio.currentTime = off / 1000; } catch (e) {} };
        this.audio.addEventListener("loadedmetadata", onMeta);
        // hosts without HTTP Range cannot seek inside a clip: if the clip comes back at 0, restart the beat from its start so sound and picture stay together
        const onPlaying = () => { this.audio.removeEventListener("playing", onPlaying); if (tok !== this._tok) return; if (this.audio.currentTime < off / 1000 - 1.5) { this._seekOff = 0; this._ffMs = 0; this.showBeat(this.beat); } };
        this.audio.addEventListener("playing", onPlaying);
      }
      const p = this.audio.play();
      if (p && p.catch) p.catch(err => {
        if (tok !== this._tok || (err && err.name === "AbortError")) return;   // superseded by a newer speak(); the new clip is already playing
        if (err && err.name === "NotAllowedError") { this.playing = false; this.updateTransport(); this.els.stage.classList.add("needs-gesture"); }
        else this.fallbackSpeak();
      });
    }
    fallbackSpeak() {
      // A clip failed to load: stay silent (never the browser's synthetic voice), hold the beat for its clip length, move on.
      if (!this.playing) return;
      const ch = this.chapters[this.chapter], b = ch.beats[this.beat], text = this.t(b);
      const ms = (b.dur && b.dur[this.lang]) ? b.dur[this.lang] * 1000 : 1200 + (this.lang === "zh" ? text.length * 220 : text.split(/\s+/).length * 380);
      if (this.fallbackTimer) clearTimeout(this.fallbackTimer);
      this.fallbackTimer = setTimeout(() => this.onBeatEnd(), ms);
    }
    stopAudio() {
      try { this.audio.pause(); } catch (e) {}
      this.audio.removeAttribute("src"); this.audio.load();
      if (window.speechSynthesis) speechSynthesis.cancel();
      if (this.fallbackTimer) { clearTimeout(this.fallbackTimer); this.fallbackTimer = null; }
    }
    onBeatEnd() {
      if (!this.playing) return;
      const ch = this.chapters[this.chapter];
      if (this.beat < ch.beats.length - 1) { this.stage.delay(350, () => this.playing && this.playBeat(this.beat + 1)); }
      else if (this.autoplay && this.chapter < this.chapters.length - 1) { this.stage.delay(900, () => this.playing && this.gotoChapter(this.chapter + 1, { play: true })); }
      else { this.playing = false; this.updateTransport(); }
    }
    togglePlay() {
      if (this.playing) { this.playing = false; this.stopAudio(); }
      else { this.playing = true; this.els.stage.classList.remove("needs-gesture"); this.playBeat(Math.max(0, this.beat)); }
      this.updateTransport();
    }
    next() { const ch = this.chapters[this.chapter]; if (this.beat < ch.beats.length - 1) this.playBeat(this.beat + 1); else this.gotoChapter(this.chapter + 1, { play: true }); }
    prev() { if (this.beat > 0) this.playBeat(this.beat - 1); else if (this.chapter > 0) this.gotoChapter(this.chapter - 1, { play: this.playing }); }
    updateTransport() {
      this.els.play.classList.toggle("playing", this.playing);
      this.els.play.setAttribute("aria-label", this.playing ? "Pause" : "Play");
      this.els.stage.classList.toggle("paused", !this.playing);
    }
  }
  CPV.Player = Player;
})();
