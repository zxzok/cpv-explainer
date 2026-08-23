/* Chapter 8 — Sleep-EDF and Long-Term AF: reconstructed protocols, layout contrasts, and the instability of learned supports. */
(function () {
  const CPV = window.CPV, C = CPV.C, D = CPV.D, M = CPV.M, S = CPV.S;
  const st = { cards: 0, sleep: 0, sProg: 0, af: 0, aProg: 0, sweep: 0, wProg: 0 };

  CPV.scenes.ch8 = {
    setup() {
      this.hyp = S.toyHypnogram(240, 5);
      const rand = M.rng(13), runs = []; let state = 0, i = 0; while (i < 300) { const len = 5 + Math.floor(rand() * 40); for (let k = 0; k < len && i < 300; k++, i++) runs.push(state); state = 1 - state; }
      this.af = runs;
      this.sleep = CPV_DATA.sleep.REM; this.afr = CPV_DATA.af_raw; this.sw = CPV_DATA.sweep; this.N = CPV_DATA.num;
    },
    enter() { Object.assign(st, { cards: 0, sleep: 0, sProg: 0, af: 0, aProg: 0, sweep: 0, wProg: 0 }); },
    beats: [
      function (stage) { Object.assign(st, { sleep: 0, sProg: 0, af: 0, aProg: 0, sweep: 0, wProg: 0 }); stage.tween(st, { cards: 1 }, 1000); },
      function (stage) { stage.tween(st, { cards: 0 }, 400); stage.tween(st, { sleep: 1 }, 600, { delay: 400 }); stage.tween(st, { sProg: 1 }, 2200, { delay: 900, ease: "out" }); },
      function (stage) { stage.tween(st, { sleep: 0 }, 400); stage.tween(st, { af: 1 }, 600, { delay: 400 }); stage.tween(st, { aProg: 1 }, 2400, { delay: 900, ease: "linear" }); },
      function (stage) { stage.tween(st, { af: 0 }, 400); stage.tween(st, { sweep: 1 }, 600, { delay: 400 }); stage.tween(st, { wProg: 1 }, 2400, { delay: 900, ease: "linear" }); },
    ],
    draw(stage) {
      const ctx = stage.ctx, N = this.N;
      S.scope(ctx, "real", Math.max(st.cards, st.sleep, st.af, st.sweep));
      if (st.cards > 0) {
        const a = st.cards;
        S.header(ctx, "Two fully annotated data sets: any protocol can be reconstructed from the complete record", "两个完整标注的数据集：任何协议都能从完整记录里重建", { alpha: a });
        const card = (x, title, lines, strip) => {
          D.rect(ctx, x, 160, 680, 560, { fill: CPV.rgba(C.panel, 0.9), stroke: C.grid, radius: 14, alpha: a });
          D.text(ctx, title, x + 30, 210, { size: 26, color: C.ink, font: "display", weight: 500, alpha: a });
          strip(x + 30, 250);
          lines.forEach((l, i) => { D.text(ctx, l[0], x + 30, 400 + i * 40, { size: 17, color: C.muted, alpha: a }); D.text(ctx, l[1], x + 650, 400 + i * 40, { size: 18, color: C.ink, font: "mono", align: "right", alpha: a }); });
        };
        card(100, "Sleep-EDF Expanded", [[S.t("hypnograms", "整夜分期记录"), N.SleepRecordings], [S.t("subjects", "受试者"), N.SleepSubjects], [S.t("valid annotated hours", "有效标注小时数"), N.SleepHours], [S.t("relative-time anchors p", "相对时间锚点 p"), N.SleepGrid], [S.t("targets", "目标"), "REM / N3 / Wake"], [S.t("budgets N (30-s epochs)", "预算 N（30 秒片段）"), "4 · 8 · 16 · 32 · 64"], [S.t("evaluation", "评估"), S.t("5-fold, subject-disjoint, pooled R²", "五折、受试者不交叠、汇总 R²")]],
          (x, y) => { const n = this.hyp.length, w = 620 / n, rows = { W: 0, R: 14, N1: 28, N2: 42, N3: 56 }; this.hyp.forEach((s, i) => D.line(ctx, x + i * w, y + rows[s], x + (i + 1) * w, y + rows[s], { color: s === "R" ? C.target : C.latent, width: 5, alpha: a * (s === "R" ? 1 : 0.5), cap: "butt" })); D.text(ctx, S.t("one night, scored every 30 s (schematic)", "一夜，每 30 秒评分一次（示意）"), x, y + 100, { size: 13, color: C.dim, alpha: a }); });
        card(820, "Long-Term AF Database", [[S.t("records", "记录数"), N.LtafRecords], [S.t("reviewed rhythm hours", "审阅过的心律小时数"), N.LtafHours], [S.t("median record length", "记录长度中位数"), N.LtafMedianHours + " h"], [S.t("relative-time bins p", "相对时间分箱 p"), N.AfGrid], [S.t("target", "目标"), S.t("AF burden (fraction of time in AF)", "房颤负担（房颤时间占比）")], [S.t("budgets N (15-min windows)", "预算 N（15 分钟窗口）"), "1 · 2 · 4 · 8 · 16 · 32"], [S.t("evaluation", "评估"), S.t("5-fold by record, pooled R²", "按记录五折、汇总 R²")]],
          (x, y) => { const n = this.af.length, w = 620 / n; this.af.forEach((s, i) => D.rect(ctx, x + i * w, y + (s ? 0 : 30), w + 0.5, 26, { fill: s ? C.alert : C.latent, alpha: a * (s ? 0.9 : 0.45) })); D.text(ctx, S.t("24 h of rhythm annotation: AF (red) and other rhythm (schematic)", "24 小时心律标注：房颤（红）与其他心律（示意）"), x, y + 100, { size: 13, color: C.dim, alpha: a }); });
        D.text(ctx, S.t("only expert annotation files are used — no raw PSG or ECG signal; the question is which segments to annotate, not end-to-end signal decoding", "只使用专家标注文件——不涉及原始 PSG 或 ECG 信号；问题是“该标注哪些片段”，而不是端到端信号解码"), 800, 780, { size: 16, color: C.muted, align: "center", alpha: a });
      }
      if (st.sleep > 0) {
        const a = st.sleep, k = st.sProg;
        S.header(ctx, "Sleep-EDF, REM fraction: dispersed vs contiguous at matched budgets (Figure 3a)", "Sleep-EDF，REM 比例：相同预算下分散 vs 连续（图 3a）", { alpha: a });
        const budgets = [4, 8, 16, 32, 64], r = { x: 200, y: 170, w: 820, h: 520 };
        const map = S.chart(ctx, r, { xr: [-0.5, 4.5], yr: [0.5, 0.95], xticks: budgets.map((b, i) => ({ v: i, label: "N = " + b })), yticks: [0.5, 0.6, 0.7, 0.8, 0.9].map(v => ({ v, label: v.toFixed(1) })), ylabel: S.t("cross-fitted R²", "交叉拟合 R²"), alpha: a });
        budgets.forEach((b, i) => {
          const c = this.sleep.consecutive[b], u = this.sleep.uniform[b], bw = 52;
          const grow = CPV.clamp(k * 5 - i, 0, 1);
          D.rect(ctx, map.X(i) - bw - 4, map.Y(0.5 + (c - 0.5) * grow), bw, map.Y(0.5) - map.Y(0.5 + (c - 0.5) * grow), { fill: CPV.rgba(C.B, 0.55), stroke: C.B, radius: 4, alpha: a });
          D.rect(ctx, map.X(i) + 4, map.Y(0.5 + (u - 0.5) * grow), bw, map.Y(0.5) - map.Y(0.5 + (u - 0.5) * grow), { fill: CPV.rgba(C.target, 0.55), stroke: C.target, radius: 4, alpha: a });
          if (grow >= 1) { D.text(ctx, c.toFixed(3), map.X(i) - bw / 2 - 4, map.Y(c) - 10, { size: 14, color: C.B, align: "center", font: "mono", alpha: a }); D.text(ctx, u.toFixed(3), map.X(i) + bw / 2 + 4, map.Y(u) - 10, { size: 14, color: C.target, align: "center", font: "mono", alpha: a }); }
        });
        S.legend(ctx, r.x + 20, r.y + 26, [{ color: C.B, label: S.t("centred contiguous block", "居中连续块"), marker: false }, { color: C.target, label: S.t("uniformly dispersed", "均匀分散"), marker: false }], { alpha: a });
        const x = 1080, y = 220;
        D.text(ctx, S.t("matched budget, different layout", "相同预算，不同布局"), x, y, { size: 20, color: C.ink, weight: 500, alpha: a });
        D.paragraph(ctx, S.t("N = 4: " + N.CfContigFour + " → " + N.CfUniformFour + ";  N = 16: " + N.CfContigSixteen + " → " + N.CfUniformSixteen + ";  N = 64: " + N.CfContigSixtyFour + " → " + N.CfUniformSixtyFour + ".", "N = 4：" + N.CfContigFour + " → " + N.CfUniformFour + "；N = 16：" + N.CfContigSixteen + " → " + N.CfUniformSixteen + "；N = 64：" + N.CfContigSixtyFour + " → " + N.CfUniformSixtyFour + "。"), x, y + 44, 440, { size: 16, color: C.muted, font: "mono", alpha: a * k, lineHeight: 26 });
        D.paragraph(ctx, S.t("The dispersed-minus-contiguous percentile range is positive in " + N.SleepAdjustedPositiveRanges + " of " + N.SleepAdjustedCells + " target–budget cells. Analysed by source study, " + N.SleepCohortPositiveRanges + " of " + N.SleepCohortCells + " ranges lie above zero, " + N.SleepCohortNegativeRanges + " below, " + N.SleepCohortUnresolvedRanges + " include zero — the pooled contrast is heterogeneous.", "“分散减连续”的分位数区间在 " + N.SleepAdjustedCells + " 个目标–预算格中有 " + N.SleepAdjustedPositiveRanges + " 个为正。按来源研究分开分析，" + N.SleepCohortCells + " 个区间中 " + N.SleepCohortPositiveRanges + " 个高于零、" + N.SleepCohortNegativeRanges + " 个低于零、" + N.SleepCohortUnresolvedRanges + " 个跨零——合并后的对比并不均匀。"), x, y + 150, 440, { size: 16, color: C.ink, alpha: a * k, lineHeight: 26 });
        D.text(ctx, S.t("target-aware and learned kernel-quadrature supports: next beat but one", "针对目标与学得核求积的位置：见本章末"), x, y + 400, { size: 14, color: C.dim, alpha: a * k });
      }
      if (st.af > 0) {
        const a = st.af, k = st.aProg;
        S.header(ctx, "Long-Term AF burden: dispersed windows vs a contiguous block (Figure 3b)", "长程房颤负担：分散窗口 vs 连续块（图 3b）", { alpha: a });
        const ns = [1, 2, 4, 8, 16, 32], hours = [0.25, 0.5, 1, 2, 4, 8], r = { x: 200, y: 170, w: 820, h: 520 };
        const map = S.chart(ctx, r, { xlog: true, xr: [0.18, 11], yr: [0.55, 1.02], xticks: hours.map((h, i) => ({ v: h, label: h + " h" })), yticks: [0.6, 0.7, 0.8, 0.9, 1.0].map(v => ({ v, label: v.toFixed(1) })), xlabel: S.t("total observed time (N windows of 15 min; 24-h record)", "总观测时长（N 个 15 分钟窗口；24 小时记录）"), ylabel: S.t("cross-fitted R²", "交叉拟合 R²"), alpha: a });
        const cont = ns.map(n => this.afr["contiguous|n=" + n].cross_fitted_r2), disp = ns.map(n => this.afr["dispersed|n=" + n].cross_fitted_r2);
        S.series(ctx, map, hours, cont, C.B, k, { alpha: a, dash: [7, 6], label: S.t("contiguous", "连续") });
        S.series(ctx, map, hours, disp, C.target, k, { alpha: a, label: S.t("dispersed", "分散"), labelDy: -10 });
        if (k >= 1) { ns.forEach((n, i) => { D.text(ctx, disp[i].toFixed(3), map.X(hours[i]), map.Y(disp[i]) - 16, { size: 13, color: C.target, align: "center", font: "mono", alpha: a }); D.text(ctx, cont[i].toFixed(3), map.X(hours[i]), map.Y(cont[i]) + 26, { size: 13, color: C.B, align: "center", font: "mono", alpha: a }); }); }
        const x = 1080, y = 220;
        D.text(ctx, S.t("four dispersed 15-min windows outperformed every contiguous block evaluated", "四个分散的 15 分钟窗口优于所评估的每一个连续块"), x, y, { size: 20, color: C.ink, weight: 500, alpha: a });
        D.paragraph(ctx, S.t("At N = 4 (1 h, " + N.AfObservedPctFour + " of the record): contiguous " + N.CfAfContigOneHour + ", dispersed " + N.CfAfDispOneHour + ". At N = 16 (4 h): " + N.CfAfContigFourHour + " vs " + N.CfAfDispFourHour + ". The paired percentile range is positive at every budget with at least two windows; at N = 1 the comparison is one of window location, not dispersion.", "N = 4（1 小时，占记录的 " + N.AfObservedPctFour + "）：连续 " + N.CfAfContigOneHour + "，分散 " + N.CfAfDispOneHour + "。N = 16（4 小时）：" + N.CfAfContigFourHour + " 对 " + N.CfAfDispFourHour + "。每个至少两个窗口的预算上，配对分位数区间都为正；N = 1 时比较的是窗口位置而非分散程度。"), x, y + 44, 440, { size: 16, color: C.ink, alpha: a * k, lineHeight: 26 });
        D.paragraph(ctx, S.t("AF episodes cluster within the day: one block may land inside or outside a cluster. Both layouts need the same 24-h wear time; what is saved is the signal that must be reviewed.", "房颤发作在一天内成簇：一整块可能恰好落在簇内或簇外。两种布局的佩戴时间相同（24 小时）；省下的是需要人工审阅的信号时长。"), x, y + 260, 440, { size: 15, color: C.muted, alpha: a * k, lineHeight: 24 });
      }
      if (st.sweep > 0) {
        const a = st.sweep, k = st.wProg, sw = this.sw;
        S.header(ctx, "Learned REM supports at N = 16: no stable held-out advantage (Figure 4)", "N = 16 时学得的 REM 位置：没有稳定的留出优势（图 4）", { alpha: a });
        const r = { x: 200, y: 180, w: 640, h: 480 };
        const map = S.chart(ctx, r, { xr: [15, 85], yr: [-0.06, 0.06], xticks: [20, 40, 60, 80].map(v => ({ v, label: String(v) })), yticks: [-0.04, -0.02, 0, 0.02, 0.04].map(v => ({ v, label: (v > 0 ? "+" : "") + v.toFixed(2) })), xlabel: S.t("training subjects used to select the support", "用于选择位置的训练受试者数"), ylabel: S.t("held-out R² advantage of the target-aware support", "针对目标位置的留出 R² 优势"), alpha: a });
        D.line(ctx, r.x, map.Y(0), r.x + r.w, map.Y(0), { color: C.muted, width: 1.5, alpha: a });
        S.series(ctx, map, sw.m, sw.d_kq, C.A, k, { alpha: a, band: [sw.d_kq.map((v, i) => v - sw.d_kq_se[i]), sw.d_kq.map((v, i) => v + sw.d_kq_se[i])], label: S.t("vs learned kernel quadrature", "对比学得核求积") });
        S.series(ctx, map, sw.m, sw.d_uni, C.target, k, { alpha: a, band: [sw.d_uni.map((v, i) => v - sw.d_uni_se[i]), sw.d_uni.map((v, i) => v + sw.d_uni_se[i])], label: S.t("vs fixed dispersion", "对比固定分散"), labelDy: 18 });
        // resampling ranges
        const x = 960, y = 250, w = 520, rs = sw.resample, lo = -0.14, hi = 0.12, Xr = v => x + (v - lo) / (hi - lo) * w;
        D.text(ctx, S.t(N.SelectionSubsampleReps + " study-stratified " + N.SelectionSubsamplePct + " subject subsamples — full pipeline rerun", N.SelectionSubsampleReps + " 次按研究分层的 " + N.SelectionSubsamplePct + " 受试者重抽样——整条流程重跑"), x, y - 30, { size: 16, color: C.ink, weight: 500, alpha: a });
        [["delta_vs_uniform", S.t("vs fixed dispersion", "对比固定分散"), C.target, sw.original.delta_vs_uniform], ["delta_vs_kq", S.t("vs kernel quadrature", "对比核求积"), C.A, sw.original.delta_vs_kq]].forEach((row, i) => {
          const yy = y + 40 + i * 90, d = rs[row[0]];
          D.text(ctx, row[1], x, yy - 20, { size: 15, color: row[2], alpha: a * k });
          D.line(ctx, Xr(lo), yy, Xr(hi), yy, { color: C.grid, width: 1, alpha: a });
          D.line(ctx, Xr(0), yy - 20, Xr(0), yy + 20, { color: C.muted, width: 1.5, alpha: a });
          D.line(ctx, Xr(d.p025), yy, Xr(d.p975), yy, { color: row[2], width: 8, alpha: a * k, cap: "butt" });
          D.circle(ctx, Xr(d.median), yy, 7, { fill: row[2], stroke: C.ink, width: 1.5, alpha: a * k });
          D.line(ctx, Xr(row[3]), yy - 14, Xr(row[3]), yy + 14, { color: C.ink, width: 2.5, alpha: a * k });
          D.text(ctx, S.t("median ", "中位数 ") + (d.median > 0 ? "+" : "") + d.median.toFixed(3) + "   [" + d.p025.toFixed(3) + ", +" + d.p975.toFixed(3) + "]", x, yy + 38, { size: 14, color: C.muted, font: "mono", alpha: a * k });
        });
        D.text(ctx, S.t("tick: original sample · dot: subsample median · bar: 2.5–97.5 percentile range", "竖线：原样本 · 圆点：重抽样中位数 · 横条：2.5–97.5 分位区间"), x, y + 230, { size: 13, color: C.dim, alpha: a * k });
        D.paragraph(ctx, S.t("Both ranges include zero. In this analysis, about " + N.TabTrainSubjects + " training subjects supported clearer broad-layout comparisons than exact anchor selection — the resolution pattern of Chapter 6 in real data.", "两个区间都包含零。在这项分析中，大约 " + N.TabTrainSubjects + " 个训练受试者支持的是对粗布局的清楚比较，而不是对精确锚点的选择——这正是第 6 章的分辨率规律在真实数据上的体现。"), x, y + 290, w, { size: 17, color: C.gold, alpha: a * k, lineHeight: 27 });
      }
    },
  };
})();
