/* Chapter 6 — Corollary 12: selection regret ≤ 2ε and the resolution of nested candidate classes. */
(function () {
  const CPV = window.CPV, C = CPV.C, D = CPV.D, S = CPV.S;
  const st = { argue: 0, classes: 0, star: 0, chart: 0, prog: 0, take: 0, main: 1 };

  CPV.scenes.ch6 = {
    setup() { this.res = CPV_DATA.resolution; this.reg = CPV_DATA.regret; },
    enter() { Object.assign(st, { argue: 0, classes: 0, star: 0, chart: 0, prog: 0, take: 0, main: 1 }); },
    beats: [
      function (stage) { Object.assign(st, { classes: 0, star: 0, chart: 0, prog: 0, take: 0 }); stage.tween(st, { argue: 1 }, 1400, { ease: "out" }); },
      function (stage) { stage.tween(st, { argue: 0 }, 400); stage.tween(st, { classes: 1 }, 1200, { delay: 400 }); stage.tween(st, { star: 1 }, 700, { delay: 2200, ease: "back" }); },
      function (stage) { stage.tween(st, { classes: 0, star: 0 }, 400); stage.tween(st, { chart: 1 }, 600, { delay: 400 }); stage.tween(st, { prog: 1 }, 3000, { delay: 900, ease: "linear" }); },
      function (stage) { stage.tween(st, { take: 1 }, 800); },
    ],
    draw(stage) {
      const ctx = stage.ctx;
      if (st.argue > 0) {
        const a = st.argue;
        S.header(ctx, "Corollary 12 — from uniform error to a decision guarantee", "推论 12——从一致误差到决策保证", { alpha: a });
        // bars: true values of S* and S-hat, estimated values within ±ε
        const x0 = 260, y0 = 620, w = 150, eps = 0.07, scale = 520;
        const Istar = 0.74, Ihat = 0.66;          // illustrative values, not paper numbers
        const bar = (x, v, col, label, lab2) => { D.rect(ctx, x, y0 - v * scale, w, v * scale, { fill: CPV.rgba(col, 0.25), stroke: col, width: 2, radius: 6, alpha: a }); D.math(ctx, label, x + w / 2, y0 + 30, { size: 19, color: col, align: "center", alpha: a }); if (lab2) D.text(ctx, lab2, x + w / 2, y0 + 56, { size: 15, color: C.muted, align: "center", alpha: a }); };
        bar(x0, Istar, C.gold, "I(S^*)", S.t("true optimum", "真正最优"));
        bar(x0 + 220, Ihat, C.expl, "I(\\hat S)", S.t("true value of the selected", "被选中者的真实价值"));
        // error brackets
        [[x0, Istar], [x0 + 220, Ihat]].forEach(([x, v]) => {
          const yc = y0 - v * scale; D.line(ctx, x + w + 14, yc - eps * scale, x + w + 14, yc + eps * scale, { color: C.alert, width: 2, alpha: a });
          D.line(ctx, x + w + 8, yc - eps * scale, x + w + 20, yc - eps * scale, { color: C.alert, width: 2, alpha: a }); D.line(ctx, x + w + 8, yc + eps * scale, x + w + 20, yc + eps * scale, { color: C.alert, width: 2, alpha: a });
          D.math(ctx, "±ε", x + w + 28, yc + 6, { size: 17, color: C.alert, alpha: a });
        });
        D.line(ctx, x0 - 30, y0, x0 + 420, y0, { color: C.dim, width: 1.5, alpha: a });
        const lines = [
          [S.t("the true optimum may be underestimated by at most ε", "真正最优的协议最多被低估 ε"), C.gold],
          [S.t("the selected protocol may be overestimated by at most ε", "被选中的协议最多被高估 ε"), C.expl],
          [S.t("the selected one won on estimated values, so", "它是按估计值胜出的，所以"), C.muted],
        ];
        lines.forEach((l, i) => D.text(ctx, l[0], 820, 250 + i * 48, { size: 22, color: l[1], alpha: a * CPV.clamp(a * 3 - i, 0, 1) }));
        D.math(ctx, "I(S^*) − I(\\hat S) ≤ 2ε", 820, 440, { size: 40, color: C.ink, alpha: a * CPV.clamp(a * 3 - 2, 0, 1) });
        D.math(ctx, S.t("\\text{with an approximate maximiser: } ≤ 2ε + η \\text{ (Corollary 12)}", "\\text{若只是近似极大化：} ≤ 2ε + η \\text{（推论 12）}"), 820, 488, { size: 17, color: C.muted, alpha: a * CPV.clamp(a * 3 - 2, 0, 1) });
        D.paragraph(ctx, S.t("No probability is used — it is an inequality. An estimated gap larger than 2ε certifies the same ordering in the population; gaps below 2ε cannot be resolved from the calibration data.", "这里没有用到概率——它只是一个不等式。估计出的差距若大于 2ε，则总体上的排序相同；小于 2ε 的差距无法从校准数据中分辨。"), 820, 560, 660, { size: 18, color: C.ink, alpha: a * CPV.clamp(a * 3 - 2, 0, 1), lineHeight: 28 });
      }
      if (st.classes > 0) {
        const a = st.classes;
        S.header(ctx, "Coarse or fine candidate classes?", "候选类该粗还是细？", { alpha: a });
        const sizes = this.res.class_sizes, names = [[S.t("layouts", "布局"), 1], [S.t("phase", "相位"), 2], [S.t("coarse bins", "粗分箱"), 3], [S.t("fine supports", "精确位置"), 4]];
        const cx = 520, cy = 470;
        [4, 3, 2, 1].forEach(L => { const r = 90 + (L - 1) * 95, al = a * CPV.clamp(a * 4 - (4 - L), 0, 1);
          D.rect(ctx, cx - r, cy - r * 0.62, 2 * r, 2 * r * 0.62, { fill: CPV.rgba(C.latent, 0.05), stroke: CPV.rgba(C.latent, 0.7), radius: 20, alpha: al });
          D.text(ctx, "L" + L + " " + names[L - 1][0] + " (" + sizes[L] + ")", cx, cy - r * 0.62 + 26, { size: L === 1 ? 14 : 16, color: C.latent, align: "center", alpha: al }); });
        if (st.star > 0) { const sx = cx + 310, sy = cy + 110; D.glow(ctx, sx, sy, 40, C.gold, 0.5 * st.star); D.text(ctx, "★", sx, sy + 12, { size: 36, color: C.gold, align: "center", alpha: st.star }); D.math(ctx, "S^*", sx + 30, sy + 8, { size: 21, color: C.gold, alpha: st.star }); }
        const x = 1000, y = 260;
        D.text(ctx, S.t("regret of the best estimated protocol in class ℓ", "第 ℓ 类中最优估计协议的损失"), x, y, { size: 19, color: C.ink, weight: 500, alpha: a });
        D.math(ctx, "≤ \\,[\\, I(S^*) − \\max_{Π^{(ℓ)}} I \\,] + 2ε_ℓ", x, y + 50, { size: 24, color: C.ink, alpha: a });
        D.text(ctx, S.t("class restriction", "类别限制"), x + 60, y + 86, { size: 16, color: C.alert, alpha: a });
        D.text(ctx, S.t("calibration error", "校准误差"), x + 380, y + 86, { size: 16, color: C.expl, alpha: a });
        D.paragraph(ctx, S.t("A coarse class may not contain S* — no amount of calibration fixes that. A fine class contains it, but its many close values need more data to resolve: the restriction term falls and the error term rises as the class is refined.", "粗的类可能不包含 S*——再多校准也弥补不了。细的类包含它，但众多相近的值需要更多数据才能分辨：随着类别变细，限制项下降而误差项上升。"), x, y + 140, 520, { size: 17, color: C.muted, alpha: a, lineHeight: 27 });
      }
      if (st.chart > 0) {
        const a = st.chart, res = this.res, cols = [C.A, C.worldM, C.expl, C.B];
        S.header(ctx, "Simulation: regret and uniform error by candidate class (Figure 2c,d)", "模拟：各候选类的选择损失与一致误差（图 2c,d）", { alpha: a });
        S.scope(ctx, "sim", a);
        const r1 = { x: 200, y: 180, w: 560, h: 460 }, r2 = { x: 900, y: 180, w: 560, h: 460 };
        const m1 = S.chart(ctx, r1, { xlog: true, ylog: true, xr: [20, 1300], yr: [0.004, 0.08], xticks: [25, 100, 1000].map(v => ({ v, label: String(v) })), yticks: [0.005, 0.01, 0.02, 0.05].map(v => ({ v, label: String(v) })), xlabel: "m", ylabel: S.t("true selection regret", "真实选择损失"), alpha: a });
        const m2 = S.chart(ctx, r2, { xlog: true, ylog: true, xr: [20, 1300], yr: [0.01, 0.3], xticks: [25, 100, 1000].map(v => ({ v, label: String(v) })), yticks: [0.01, 0.02, 0.05, 0.1, 0.2].map(v => ({ v, label: String(v) })), xlabel: "m", ylabel: S.t("uniform error ε_ℓ", "一致误差 ε_ℓ"), alpha: a });
        [1, 2, 3, 4].forEach((L, i) => { S.series(ctx, m1, res.m, res.regret[L], cols[i], st.prog, { alpha: a, width: 2.5 }); S.series(ctx, m2, res.m, res.eps[L], cols[i], st.prog, { alpha: a, width: 2.5 }); });
        S.legend(ctx, r1.x + 20, r1.y + 26, [1, 2, 3, 4].map((L, i) => ({ color: cols[i], label: "L" + L + " " + res.class_names[L] + " (" + res.class_sizes[L] + ")" })), { alpha: a, gap: 16 });
        const k = st.prog;
        D.text(ctx, S.t("m = 25: finest class regret " + res.regret[4][0].toFixed(3) + " vs coarsest " + res.regret[1][0].toFixed(3) + ", despite ε₄ = " + res.eps[4][0].toFixed(2), "m = 25：最细类损失 " + res.regret[4][0].toFixed(3) + "，最粗类 " + res.regret[1][0].toFixed(3) + "，尽管 ε₄ = " + res.eps[4][0].toFixed(2)), 800, 740, { size: 17, color: C.ink, align: "center", alpha: a * CPV.clamp(k * 3 - 0.5, 0, 1) });
        D.text(ctx, S.t("m = 1000: coarsest class stuck at restriction regret " + res.regret[1][5].toFixed(3) + "; finest reaches " + res.regret[4][5].toFixed(3) + " with ε₄ = " + res.eps[4][5].toFixed(3), "m = 1000：最粗类停在限制损失 " + res.regret[1][5].toFixed(3) + "；最细类降到 " + res.regret[4][5].toFixed(3) + "，此时 ε₄ = " + res.eps[4][5].toFixed(3)), 800, 772, { size: 17, color: C.ink, align: "center", alpha: a * CPV.clamp(k * 3 - 2, 0, 1) });
        D.text(ctx, S.t("ordering the leading candidates is easier than estimating every value uniformly", "排好前几名，比一致地估准每一个值容易"), 800, 808, { size: 17, color: C.gold, align: "center", alpha: a * CPV.clamp(k * 3 - 2, 0, 1) });
      }
      if (st.take > 0) {
        D.rect(ctx, 0, 0, 1600, 900, { fill: CPV.rgba(C.bg, 0.85 * st.take) });
        D.takeaway(ctx, S.t("The granularity at which protocols can be optimised is set by the calibration data, not by the optimiser.", "协议能优化到多细的粒度，由校准数据决定，而不是由优化算法决定。"), 800, 420, { alpha: st.take, size: 34, maxW: 1200 });
        D.text(ctx, S.t("— the reading rule for the real-data results in Chapter 8", "——这也是第 8 章真实数据结果的解读规则"), 800, 540, { size: 20, color: C.muted, align: "center", alpha: st.take });
      }
    },
  };
})();
