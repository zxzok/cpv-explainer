/* Helpers shared by the chapter scenes. */
(function () {
  const CPV = window.CPV, C = CPV.C, D = CPV.D, M = CPV.M;
  const S = CPV.S = {};

  S.t = (en, zh) => (window.player && window.player.lang === "zh") ? zh : en;
  S.isZh = () => window.player && window.player.lang === "zh";
  S.num = CPV_DATA.num;
  S.numf = k => parseFloat(String(CPV_DATA.num[k]).replace(/[+%]/g, ""));

  S.header = (ctx, en, zh, o = {}) => {
    D.text(ctx, S.t(en, zh), 60, 66, { size: 26, weight: 500, color: C.ink, alpha: o.alpha === undefined ? 1 : o.alpha });
    if (o.sub) D.text(ctx, S.t(o.sub[0], o.sub[1]), 60, 96, { size: 17, color: C.muted, alpha: o.alpha === undefined ? 1 : o.alpha });
  };

  /* Vertical measurement mark with a soft glow. */
  S.mark = (ctx, x, y1, y2, color, o = {}) => {
    const a = o.alpha === undefined ? 1 : o.alpha; if (a <= 0) return;
    ctx.save(); ctx.globalAlpha *= a;
    ctx.shadowColor = color; ctx.shadowBlur = o.blur === undefined ? 14 : o.blur;
    ctx.strokeStyle = color; ctx.lineWidth = o.width || 4; ctx.lineCap = "round";
    if (o.dash) ctx.setLineDash(o.dash);
    ctx.beginPath(); ctx.moveTo(x, y1); ctx.lineTo(x, y2); ctx.stroke();
    ctx.restore();
    if (o.label) D.text(ctx, o.label, x, (o.labelBelow ? y2 + 26 : y1 - 12), { size: o.labelSize || 17, color, align: "center", alpha: a, font: o.labelFont || "sans", weight: 500 });
  };
  /* Horizontal window (averaging action). */
  S.window = (ctx, x1, x2, y1, y2, color, o = {}) => {
    const a = o.alpha === undefined ? 1 : o.alpha; if (a <= 0) return;
    D.rect(ctx, x1, y1, x2 - x1, y2 - y1, { fill: CPV.rgba(color, 0.22), stroke: CPV.rgba(color, 0.8), radius: 4, alpha: a, dash: o.dash });
  };

  /* Toy hypnogram: five sleep cycles, deep sleep early, REM lengthening by dawn (ported from s01_question.py). */
  S.toyHypnogram = (n = 170, seed = 11) => {
    const rand = M.rng(seed), out = [];
    while (out.length < n) {
      const frac = out.length / n;
      const cycle = frac < 0.6 ? ["N1", "N2", "N3", "N2", "R"] : ["N1", "N2", "R"];
      for (const st of cycle) {
        let dur;
        if (st === "R") dur = Math.floor(2 + 6 * frac + Math.floor(rand() * 2));
        else if (st === "N3") dur = Math.floor(10 - 7 * frac + Math.floor(rand() * 3));
        else dur = Math.floor(5 + Math.floor(rand() * 5));
        for (let i = 0; i < Math.max(dur, 1); i++) out.push(st);
      }
      if (rand() < 0.3) { const k = 1 + Math.floor(rand() * 2); for (let i = 0; i < k; i++) out.push("W"); }
    }
    return out.slice(0, n);
  };
  S.STAGES = ["W", "R", "N1", "N2", "N3"];

  /* A log- or linear-scaled chart frame.  Returns {x(v), y(v)} mappers after drawing grid and ticks. */
  S.chart = (ctx, r, o = {}) => {
    const a = o.alpha === undefined ? 1 : o.alpha;
    const xl = o.xlog, yl = o.ylog;
    const fx = v => xl ? Math.log10(v) : v, fy = v => yl ? Math.log10(v) : v;
    const x0 = fx(o.xr[0]), x1 = fx(o.xr[1]), y0 = fy(o.yr[0]), y1 = fy(o.yr[1]);
    const X = v => r.x + (fx(v) - x0) / (x1 - x0) * r.w, Y = v => r.y + r.h - (fy(v) - y0) / (y1 - y0) * r.h;
    if (a <= 0) return { X, Y };
    D.rect(ctx, r.x, r.y, r.w, r.h, { fill: CPV.rgba("#ffffff", 0.015), alpha: a });
    (o.xticks || []).forEach(t => { const v = typeof t === "object" ? t.v : t, lab = typeof t === "object" ? t.label : String(t);
      D.line(ctx, X(v), r.y, X(v), r.y + r.h, { color: C.grid, width: 1, alpha: a }); D.text(ctx, lab, X(v), r.y + r.h + 24, { size: 15, color: C.muted, align: "center", font: "mono", alpha: a }); });
    (o.yticks || []).forEach(t => { const v = typeof t === "object" ? t.v : t, lab = typeof t === "object" ? t.label : String(t);
      D.line(ctx, r.x, Y(v), r.x + r.w, Y(v), { color: C.grid, width: 1, alpha: a }); D.text(ctx, lab, r.x - 10, Y(v) + 5, { size: 15, color: C.muted, align: "right", font: "mono", alpha: a }); });
    D.line(ctx, r.x, r.y + r.h, r.x + r.w, r.y + r.h, { color: C.dim, width: 1.5, alpha: a });
    D.line(ctx, r.x, r.y, r.x, r.y + r.h, { color: C.dim, width: 1.5, alpha: a });
    if (o.xlabel) D.text(ctx, o.xlabel, r.x + r.w / 2, r.y + r.h + 52, { size: 16, color: C.muted, align: "center", alpha: a });
    if (o.ylabel) { ctx.save(); ctx.translate(r.x - 58, r.y + r.h / 2); ctx.rotate(-Math.PI / 2); D.text(ctx, o.ylabel, 0, 0, { size: 16, color: C.muted, align: "center", alpha: a }); ctx.restore(); }
    if (o.title) D.text(ctx, o.title, r.x + r.w / 2, r.y - 14, { size: 18, color: C.ink, align: "center", weight: 500, alpha: a });
    return { X, Y };
  };
  /* Progressive polyline with end marker; progress in [0,1]. */
  S.series = (ctx, map, xs, ys, color, progress, o = {}) => {
    const a = (o.alpha === undefined ? 1 : o.alpha); if (a <= 0 || progress <= 0) return;
    const n = xs.length, total = (n - 1) * progress, k = Math.floor(total), f = total - k;
    const pts = [];
    for (let i = 0; i <= Math.min(k, n - 1); i++) pts.push([map.X(xs[i]), map.Y(ys[i])]);
    if (k < n - 1 && f > 0) pts.push([CPV.lerp(map.X(xs[k]), map.X(xs[k + 1]), f), CPV.lerp(map.Y(ys[k]), map.Y(ys[k + 1]), f)]);
    if (o.band) {
      const lo = o.band[0], hi = o.band[1], poly = [];
      for (let i = 0; i <= Math.min(k, n - 1); i++) poly.push([map.X(xs[i]), map.Y(hi[i])]);
      for (let i = Math.min(k, n - 1); i >= 0; i--) poly.push([map.X(xs[i]), map.Y(lo[i])]);
      D.poly(ctx, poly, { fill: CPV.rgba(color, 0.14), noStroke: true, alpha: a });
    }
    D.poly(ctx, pts, { color, width: o.width || 3, alpha: a, dash: o.dash });
    if (o.markers !== false) for (let i = 0; i <= Math.min(k, n - 1); i++) D.circle(ctx, map.X(xs[i]), map.Y(ys[i]), o.r || 4.5, { fill: color, alpha: a });
    if (o.label) { const last = pts[pts.length - 1]; D.text(ctx, o.label, last[0] + 10, last[1] + (o.labelDy || 5), { size: 15, color, alpha: a, weight: 500 }); }
  };
  S.legend = (ctx, x, y, items, o = {}) => {
    const a = o.alpha === undefined ? 1 : o.alpha;
    let cx = x;
    items.forEach(it => {
      D.line(ctx, cx, y, cx + 26, y, { color: it.color, width: 3.5, alpha: a, dash: it.dash });
      if (it.marker !== false) D.circle(ctx, cx + 13, y, 4, { fill: it.color, alpha: a });
      D.text(ctx, it.label, cx + 34, y + 5, { size: 15, color: C.ink, alpha: a });
      cx += 34 + D.measure(ctx, it.label, { size: 15 }) + (o.gap || 28);
    });
  };
  /* Evidence-type label, top right of the frame: which kind of claim the picture supports. */
  S.SCOPE = { math: ["Minimal mathematical example", "最小数学例子"], sim: ["Simulation under known data-generating laws", "已知生成机制下的模拟"],
              real: ["Retrospective reconstruction from fully annotated data", "基于完整标注数据的回溯重建"], schematic: ["Schematic thought experiment — not the Sleep-EDF analysis reported later", "示意性的思想实验——不是后文报告的 Sleep-EDF 分析"] };
  S.scope = (ctx, kind, alpha = 1) => { const c = { math: C.latent, sim: C.expl, real: C.target, schematic: C.muted }[kind]; D.chip(ctx, S.t(S.SCOPE[kind][0], S.SCOPE[kind][1]), 1540, 62, { color: c, align: "right", alpha, size: 14, padX: 10 }); };
  S.countUp = (v, target, digits = 0) => (Math.round(target * v * Math.pow(10, digits)) / Math.pow(10, digits)).toLocaleString("en-US", { minimumFractionDigits: digits, maximumFractionDigits: digits });
})();
