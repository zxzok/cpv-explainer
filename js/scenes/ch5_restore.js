/* Chapter 5 — restoring identification: augmentation (Proposition 8), dense calibration, the estimator, Theorem 10, and the simulated rates. */
(function () {
  const CPV = window.CPV, C = CPV.C, D = CPV.D, M = CPV.M, S = CPV.S;
  const st = { aug: 0, augStep: 0, cal: 0, calAlpha: 0, pipe: 0, pipeStep: 0, thm: 0, chart: 0, prog: 0, main: 1 };
  const R = { x: 100, y: 170, w: 720, h: 400 }, T = 20;
  const DIMS = [4, 2, 0];

  CPV.scenes.ch5 = {
    useGL: true, glRect: R,
    setup(stage) {
      const p = 64, K = M.traitState(M.ouKernel(p, 2.5, T), 0.3), L = M.cholesky(K, 1e-9), rand = M.rng(21);
      this.xs = Float32Array.from(M.grid(p, T));
      const paths = M.samplePaths(L, 60, rand);
      this.ys = paths.map(z => Float32Array.from(z, v => v + 0.35 * rand.gauss()));
      this.colors = paths.map(() => CPV.glColor(C.latent, 0.3));
      this.cal = CPV_DATA.calibration;
    },
    enter(stage) {
      Object.assign(st, { aug: 0, augStep: 0, cal: 0, calAlpha: 0, pipe: 0, pipeStep: 0, thm: 0, chart: 0, prog: 0, main: 1 });
      stage.field.setView(0, T, -3.6, 3.6);
      this.layer = stage.field.addLayer(stage.field.makeLayer(this.xs, this.ys, this.colors, { width: 1.5, alpha: 0 }));
    },
    beats: [
      function (stage) { Object.assign(st, { cal: 0, calAlpha: 0, pipe: 0, pipeStep: 0, thm: 0, chart: 0, prog: 0, main: 1, augStep: 0 }); stage.tween(st, { aug: 1 }, 800);
        stage.delay(3200, () => stage.tween(st, { augStep: 1 }, 900)); stage.delay(6200, () => stage.tween(st, { augStep: 2 }, 900)); },
      function (stage) { stage.tween(st, { aug: 0 }, 500); stage.tween(st, { cal: 1, calAlpha: 1 }, 1200, { delay: 400 }); },
      function (stage) { stage.tween(st, { calAlpha: 0.25 }, 600); stage.tween(st, { pipe: 1 }, 700, { delay: 300 }); [1, 2, 3, 4].forEach(k => stage.delay(1200 + k * 1500, () => { st.pipeStep = k; })); },
      function (stage) { stage.tween(st, { cal: 0, calAlpha: 0, pipe: 0 }, 500); stage.tween(st, { thm: 1 }, 900, { delay: 500 }); },
      function (stage) { stage.tween(st, { thm: 0 }, 400); stage.tween(st, { chart: 1 }, 600, { delay: 400 }); stage.tween(st, { prog: 1 }, 2600, { delay: 1000, ease: "linear" }); },
    ],
    draw(stage, t) {
      const ctx = stage.ctx;
      this.layer.alpha = st.calAlpha;
      if (st.aug > 0) {
        const a = st.aug;
        S.header(ctx, "Proposition 8 — identification by augmentation is value-specific", "命题 8——通过补测恢复识别，且只需针对所评估的价值", { alpha: a });
        const y = 300, x0 = 300, dx = 260, px = i => x0 + dx * i;
        D.line(ctx, x0 - 40, y, x0 + 3 * dx + 40, y, { color: C.dim, width: 2, alpha: a });
        for (let i = 0; i < 4; i++) { D.circle(ctx, px(i), y, 10, { fill: C.latent, alpha: a }); D.math(ctx, "Z_" + i, px(i), y + 42, { size: 22, color: C.latent, align: "center", alpha: a }); }
        S.mark(ctx, px(0), y - 90, y - 22, C.A, { alpha: a, label: "A", labelSize: 20 });
        const s1 = CPV.clamp(st.augStep, 0, 1), s2 = CPV.clamp(st.augStep - 1, 0, 1);
        S.mark(ctx, px(1), y - 90, y - 22, C.A, { alpha: a * s1, label: S.t("add Z₁", "补测 Z₁"), labelSize: 18 });
        S.mark(ctx, px(2), y - 90, y - 22, C.A, { alpha: a * s2, label: S.t("add Z₂", "补测 Z₂"), labelSize: 18 });
        [1, 2].forEach(i => S.mark(ctx, px(i), y + 60, y + 110, C.B, { alpha: a, dash: [8, 6], width: 3.5 }));
        D.math(ctx, "B = \\{Z_1, Z_2\\}", px(1.5), y + 140, { size: 18, color: C.B, align: "center", alpha: a });
        const dim = st.augStep < 1 ? DIMS[0] + (DIMS[1] - DIMS[0]) * s1 : DIMS[1] + (DIMS[2] - DIMS[1]) * s2;
        D.text(ctx, S.t("dimension of the invisible space (no stationarity assumed)", "不可见空间的维数（不假设平稳）"), 800, 520, { size: 19, color: C.muted, align: "center", alpha: a });
        D.text(ctx, Math.round(dim).toString(), 800, 620, { size: 96, color: dim > 0.5 ? C.alert : C.expl, align: "center", font: "mono", weight: 500, alpha: a });
        D.math(ctx, "\\{Z_0\\}: 4 \\quad\\quad \\{Z_0, Z_1\\}: 2 \\quad\\quad \\{Z_0, Z_1, Z_2\\}: 0", 800, 680, { size: 20, color: C.muted, align: "center", alpha: a });
        D.text(ctx, S.t("under stationarity, Z₁ alone already suffices (1 → 0); identification targets the value of B, not all of K", "若假设平稳，仅补测 Z₁ 已足够（1 → 0）；识别的对象是 B 的价值，而不是整个 K"), 800, 760, { size: 17, color: C.gold, align: "center", alpha: a * s2 });
      }
      if (st.cal > 0) {
        const a = st.cal;
        S.header(ctx, "Dense calibration: a small set of units observed over the whole horizon", "密集校准：少量对象，记录整条轨迹", { alpha: a });
        D.axis(ctx, R.x, R.y + R.h + 12, R.w, { ticks: [0, 5, 10, 15, 20].map(h => ({ u: h / T, label: h })), alpha: a });
        D.chip(ctx, S.t("m = 60 calibration units, W = Z + η", "m = 60 个校准对象，W = Z + η"), R.x + 8, R.y + 22, { color: C.latent, alpha: a });
        const x = 900, y = 230;
        D.math(ctx, "W^{(i)} = Z^{(i)} + η^{(i)}, \\quad i = 1, …, m", x, y, { size: 22, color: C.ink, alpha: a });
        D.math(ctx, "Cov(W^{(i)}) = K + R_0", x, y + 48, { size: 22, color: C.latent, alpha: a });
        D.paragraph(ctx, S.t("Their population law identifies K, hence the value of every candidate protocol. The question becomes statistical: how accurately, with finite m?", "它们的总体分布识别 K，从而识别每一个候选协议的价值。问题于是变成统计问题：有限的 m 下能估多准？"), x, y + 110, 600, { size: 19, color: C.muted, alpha: a, lineHeight: 30 });
        if (st.pipe > 0) {
          const steps = [["\\hat Σ = m^{-1} \\sum_i W^{(i)} W^{(i)\\top}", S.t("sample covariance", "样本协方差")], ["\\hat Σ − \\hat R_0", S.t("subtract calibration noise", "扣除校准噪声")], ["proj\\{H ⪰ τ I\\}", S.t("eigenvalue floor (valid covariance)", "特征值下限（合法协方差）")], ["D^{-1/2}(\\,·\\,)\\,D^{-1/2}", S.t("rescale to a correlation matrix", "重新标准化为相关矩阵")], ["\\hat K → \\hat I_g(S) = F_g(S;\\hat K)/V_g(\\hat K)", S.t("plug in, for every candidate S", "代入每一个候选 S")]];
          const bx = R.x, by = 650, bw = 270, gap = 24;
          steps.forEach((s, i) => {
            const on = st.pipeStep >= i ? 1 : 0.35, col = i === 4 ? C.expl : C.ink, x0 = bx + i * (bw + gap);
            D.rect(ctx, x0, by, bw, 110, { fill: CPV.rgba(C.panel, 0.95), stroke: st.pipeStep === i ? C.gold : C.grid, width: st.pipeStep === i ? 2 : 1, radius: 10, alpha: st.pipe * on });
            D.badge(ctx, i + 1, x0 + 22, by + 24, { alpha: st.pipe * on, r: 12 });
            D.math(ctx, s[0], x0 + 14, by + 66, { size: 15, color: col, alpha: st.pipe * on });
            D.text(ctx, s[1], x0 + 14, by + 92, { size: 14, color: C.muted, alpha: st.pipe * on });
            if (i < 4) D.arrow(ctx, x0 + bw + 2, by + 55, x0 + bw + gap - 2, by + 55, { color: C.dim, width: 2, alpha: st.pipe * on, head: 8 });
          });
        }
      }
      if (st.thm > 0) {
        const a = st.thm;
        S.header(ctx, "Theorem 10 — covariance error becomes value error, uniformly over the family", "定理 10——协方差误差一致地转化为整个候选族上的价值误差", { alpha: a });
        D.math(ctx, "\\sup_{S ∈ Π_B} |\\hat I_g(S) − I_g(S)| ≤ C\\,\\norm{\\hat K − K}_{op}^{\\,β}", 800, 250, { size: 30, color: C.ink, align: "center", alpha: a });
        D.text(ctx, S.t("C depends on the target modulus, the target-variance floor, bounds on K, and the family's conditioning — not on S", "C 只依赖目标的连续模、目标方差下界、K 的界和候选族的条件数——与 S 无关"), 800, 296, { size: 16, color: C.muted, align: "center", alpha: a });
        const rows = [[S.t("\\text{smooth target } g ∈ W^{1,2}(φ)", "\\text{平滑目标 } g ∈ W^{1,2}(φ)"), "β = 1", C.expl], [S.t("\\text{threshold target, worst case over all correlation matrices}", "\\text{阈值目标，所有相关矩阵上的最坏情形}"), "β = 1/2", C.alert], [S.t("\\text{threshold target at any fixed interior model } (|r| ≤ r_0 < 1)", "\\text{阈值目标，任何固定的内部模型 }(|r| ≤ r_0 < 1)"), "β = 1", C.expl]];
        rows.forEach((r, i) => { const y = 380 + i * 64; D.rect(ctx, 300, y - 34, 1000, 50, { fill: CPV.rgba(C.panel, 0.9), stroke: C.grid, radius: 8, alpha: a }); D.math(ctx, r[0], 322, y, { size: 19, color: C.ink, alpha: a }); D.math(ctx, r[1], 1270, y, { size: 24, color: r[2], align: "right", alpha: a }); });
        D.text(ctx, S.t("Proposition 9: the square-root envelope is sharp — C_gc(1) − C_gc(1−δ) ~ e^{−c²/2}√(2δ)/2π.  Theorem 11: at a fixed model the plug-in values converge at the root-m rate.", "命题 9：平方根包络不可改进——C_gc(1) − C_gc(1−δ) ~ e^{−c²/2}√(2δ)/2π。定理 11：固定模型下插入估计以根号 m 速率收敛。"), 800, 640, { size: 16, color: C.muted, align: "center", alpha: a });
        D.takeaway(ctx, S.t("Regret and resolution follow from one uniform error number ε_m — next chapter.", "选择损失与分辨率都由同一个一致误差 ε_m 决定——见下一章。"), 800, 760, { alpha: a, size: 24 });
      }
      if (st.chart > 0) {
        const a = st.chart;
        S.header(ctx, "Simulation: family-uniform error over 495 protocols vs calibration size", "模拟：495 个协议上的一致误差随校准规模的变化", { alpha: a });
        S.scope(ctx, "sim", a);
        const r = { x: 220, y: 170, w: 760, h: 520 };
        const map = S.chart(ctx, r, { xlog: true, ylog: true, xr: [20, 1300], yr: [0.04, 0.35], xticks: [25, 50, 100, 250, 500, 1000].map(v => ({ v, label: String(v) })), yticks: [0.05, 0.1, 0.2, 0.3].map(v => ({ v, label: v.toFixed(2) })), xlabel: S.t("calibration trajectories m", "校准轨迹数 m"), ylabel: "ε_m", alpha: a });
        const cm = this.cal.mean, co = this.cal.occ_c0;
        // fitted slope line for the mean target (paper: -0.4134) through the geometric centre
        const lx = cm.m.map(Math.log10), ly = cm.eps.map(Math.log10), mx = lx.reduce((s, v) => s + v, 0) / lx.length, my = ly.reduce((s, v) => s + v, 0) / ly.length;
        const slope = S.numf("SlopeMean"), ref = [20, 1300].map(v => Math.pow(10, my + slope * (Math.log10(v) - mx)));
        S.series(ctx, map, [20, 1300], ref, C.dim, a > 0 ? 1 : 0, { dash: [6, 6], width: 1.5, markers: false, alpha: a * st.prog });
        S.series(ctx, map, cm.m, cm.eps, C.latent, st.prog, { alpha: a });
        S.series(ctx, map, co.m, co.eps, C.target, st.prog, { alpha: a });
        S.legend(ctx, r.x + 30, r.y + 30, [{ color: C.latent, label: S.t("temporal mean", "均值目标") }, { color: C.target, label: S.t("occupation above zero", "零阈值占用时间") }, { color: C.dim, label: S.t("fit, slope " + CPV_DATA.num.SlopeMean, "拟合斜率 " + CPV_DATA.num.SlopeMean), dash: [6, 6], marker: false }], { alpha: a });
        const x = 1060, y = 230;
        D.text(ctx, S.t("log–log slopes", "对数–对数斜率"), x, y, { size: 20, color: C.ink, weight: 500, alpha: a });
        [["mean", "SlopeMean", "TailSlopeMean", C.latent], ["occupation c = 0", "SlopeOccZero", "TailSlopeOccZero", C.target]].forEach((r2, i) => {
          D.text(ctx, r2[0], x, y + 46 + i * 70, { size: 17, color: r2[3], alpha: a * st.prog });
          D.text(ctx, S.t("full range ", "全程 ") + CPV_DATA.num[r2[1]] + S.t("   largest three m: ", "   最大的三个 m：") + CPV_DATA.num[r2[2]], x, y + 72 + i * 70, { size: 16, color: C.muted, font: "mono", alpha: a * st.prog });
        });
        D.text(ctx, S.t("fixed-model exponent: −1/2", "固定模型下的理论指数：−1/2"), x, y + 210, { size: 17, color: C.gold, alpha: a * st.prog });
        D.paragraph(ctx, S.t("Family of " + CPV_DATA.num.EstFamily + " four-action protocols, OU model on p = 128, " + CPV_DATA.num.EstReplications + " replications per m; the two targets behave alike, as Theorem 10's interior case predicts.", CPV_DATA.num.EstFamily + " 个四动作协议，p = 128 的 OU 模型，每个 m 重复 " + CPV_DATA.num.EstReplications + " 次；两种目标表现一致，与定理 10 的内部情形一致。"), x, y + 260, 440, { size: 15, color: C.muted, alpha: a * st.prog, lineHeight: 23 });
      }
    },
  };
})();
