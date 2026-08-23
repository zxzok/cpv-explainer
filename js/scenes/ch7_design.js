/* Chapter 7 — target-aware design: exact rank-one gains (Proposition 13), greedy search with a collapsing posterior, target dependence, and the non-submodularity caution. */
(function () {
  const CPV = window.CPV, C = CPV.C, D = CPV.D, M = CPV.M, S = CPV.S;
  const R = { x: 80, y: 150, w: 1000, h: 400 }, T = 20, p = 64, NP = 150;
  const st = { cat: 0, formula: 0, run: 0, step: -1, collapse: 1, gains: 0, compare: 0, caution: 0, field: 1, main: 1 };

  CPV.scenes.ch7 = {
    useGL: true, glRect: R,
    setup(stage) {
      this.K = M.traitState(M.ouKernel(p, 2.2, T), 0.25);
      // recency-weighted aggregate: omega grows linearly over the horizon (the paper's 'recency' setting), where target-specific supports differ
      this.w = Float64Array.from({ length: p }, (_, i) => i + 1); { const tot = this.w.reduce((a, b) => a + b, 0); for (let i = 0; i < p; i++) this.w[i] /= tot; }
      const L = M.cholesky(this.K, 1e-9); this.rand = M.rng(99);
      this.truth = M.samplePaths(L, 1, this.rand)[0];
      this.priors = M.samplePaths(L, NP, this.rand);
      this.xs = Float32Array.from(M.grid(p, T));
      this.cands = []; for (let j = 2; j < p; j += 4) { const ell = new Float64Array(p); ell[j] = 1; this.cands.push({ ell, idx: j, r: 0.3, cost: 1, label: String(j) }); }
      this.stepsMean = M.greedy(this.K, this.w, this.cands, 4, { kind: "mean" });
      this.stepsOcc = M.greedy(this.K, this.w, this.cands, 4, { kind: "occ", c: 1 });
      // all-candidate gains at each greedy step (mean target), for the bar display
      this.gainTable = [];
      let P = M.copy(this.K);
      for (let s = 0; s <= this.stepsMean.length; s++) {
        const chosen = new Set(this.stepsMean.slice(0, s).map(x => x.idx));
        this.gainTable.push(this.cands.map((a, i) => { if (chosen.has(i)) return 0; const v = M.matvec(P, a.ell), d = M.dot(this.w, v); return d * d / (M.dot(a.ell, v) + a.r); }));
        if (s < this.stepsMean.length) P = this.stepsMean[s].P;
      }
      this.colors = this.priors.map(() => CPV.glColor(C.latent, 0.2));
      this.ysPrior = this.priors.map(z => Float32Array.from(z));
      this.methods = CPV_DATA.design_methods;
    },
    enter(stage) {
      Object.assign(st, { cat: 0, formula: 0, run: 0, step: -1, collapse: 1, gains: 0, compare: 0, caution: 0, field: 1, main: 1 });
      stage.field.setView(0, T, -3.6, 3.6);
      this.layer = stage.field.addLayer(stage.field.makeLayer(this.xs, this.ysPrior, this.colors, { width: 1.5, alpha: 1 }));
      this.obs = []; this.ysFrom = this.ysPrior; this.ysTo = this.ysPrior;
    },
    addObservation(stage, k) {
      const step = this.stepsMean[k]; this.obs.push({ idx: step.action.idx, r: step.action.r });
      this.ysFrom = this.ysTo;
      this.ysTo = M.conditionPaths(this.K, this.obs, this.truth, this.priors, M.rng(5 + k)).map(z => Float32Array.from(z));
      st.collapse = 0; stage.tween(st, { collapse: 1 }, 1000, { ease: "inOut" });
      st.step = k;
    },
    beats: [
      function (stage) { Object.assign(st, { formula: 0, run: 0, step: -1, gains: 0, compare: 0, caution: 0, field: 1, main: 1 }); this.obs = []; this.ysFrom = this.ysPrior; this.ysTo = this.ysPrior; st.collapse = 1; stage.tween(st, { cat: 1 }, 900); },
      function (stage) { stage.tween(st, { formula: 1 }, 900); },
      function (stage) { stage.tween(st, { formula: 0.25, gains: 1 }, 500); this.obs = []; this.ysFrom = this.ysPrior; this.ysTo = this.ysPrior; st.collapse = 1; st.step = -1;
        [0, 1, 2, 3].forEach(k => stage.delay(900 + k * 2300, () => this.addObservation(stage, k))); },
      function (stage) { stage.tween(st, { gains: 0, field: 0.35 }, 500); stage.tween(st, { compare: 1 }, 900, { delay: 400 }); },
      function (stage) { stage.tween(st, { compare: 0, cat: 0, field: 0 }, 400); stage.tween(st, { caution: 1 }, 900, { delay: 400 }); },
    ],
    draw(stage, t) {
      const ctx = stage.ctx, X = u => R.x + R.w * u / T, Y = v => R.y + R.h / 2 - v / 3.6 * R.h / 2;
      // posterior collapse interpolation
      if (st.collapse < 1 || this._lastTo !== this.ysTo) {
        const k = st.collapse, ys = this.ysTo.map((z, i) => Float32Array.from(z, (v, j) => this.ysFrom[i][j] + (v - this.ysFrom[i][j]) * k));
        stage.field.updateYs(this.layer, ys); if (k >= 1) this._lastTo = this.ysTo;
      }
      this.layer.alpha = st.field;
      if (st.caution < 1) {
        const a = st.main * (1 - st.caution);
        S.header(ctx, "Target-aware observation design under a cost budget", "预算约束下针对目标的观测设计", { alpha: a });
        // truth and observations on the overlay
        if (st.field > 0) {
          const pts = Array.from(this.truth, (v, j) => [X(this.xs[j]), Y(v)]);
          D.poly(ctx, pts, { color: C.gold, width: 2, alpha: 0.9 * st.field * a });
          this.obs.forEach((o, i) => { const x = X(this.xs[o.idx]); S.mark(ctx, x, R.y + 10, R.y + R.h - 10, C.A, { alpha: a * (i === this.obs.length - 1 ? st.collapse : 1), width: 3, blur: 10 }); D.circle(ctx, x, Y(this.truth[o.idx]), 7, { fill: C.A, stroke: C.ink, width: 2, alpha: a }); });
          D.chip(ctx, S.t("posterior sample paths given the measurements so far", "给定已有测量的后验样本轨迹"), R.x + 8, R.y + 22, { color: C.latent, alpha: a * st.field });
          D.chip(ctx, S.t("one true trajectory", "一条真实轨迹"), R.x + R.w - 8, R.y + 22, { color: C.gold, alpha: a * st.field, align: "right" });
        }
        if (st.cat > 0) {
          const ay = R.y + R.h + 20;
          D.line(ctx, R.x, ay, R.x + R.w, ay, { color: C.dim, width: 2, alpha: a * st.cat });
          this.cands.forEach((c, i) => { const x = X(this.xs[c.idx]), chosen = this.obs.some(o => o.idx === c.idx); D.line(ctx, x, ay - 8, x, ay + 8, { color: chosen ? C.A : C.muted, width: chosen ? 4 : 2, alpha: a * st.cat }); });
          D.text(ctx, S.t("catalogue: 16 candidate point actions, noise r = 0.3, cost 1 each, budget 4 · target weights ω rise linearly over the horizon (recency-weighted)", "候选目录：16 个点测量动作，噪声 r = 0.3，成本各 1，预算 4 · 目标权重 ω 随时间线性增大（偏重近期）"), R.x, ay + 40, { size: 16, color: C.muted, alpha: a * st.cat });
          const x = 1120, y = 200;
          D.math(ctx, S.t("\\text{an action } a = (ℓ_a, r_a, c_a)", "\\text{一个动作 } a = (ℓ_a, r_a, c_a)"), x, y, { size: 20, color: C.ink, alpha: a * st.cat });
          [[S.t("timing & support", "时刻与支撑"), "ℓ_a"], [S.t("repetition & noise", "重复次数与噪声"), "r_a = ν_a^2 / M_a"], [S.t("cost", "成本"), "c_a"], [S.t("feasible set", "可行集"), "\\sum_{a ∈ S} c_a ≤ B"]].forEach((r, i) => { D.text(ctx, r[0], x, y + 44 + i * 40, { size: 16, color: C.muted, alpha: a * st.cat }); D.math(ctx, r[1], x + 190, y + 44 + i * 40, { size: 17, color: C.latent, alpha: a * st.cat }); });
          D.math(ctx, "\\max_{S ∈ Π_B} F_g(S; \\hat K)", x, y + 230, { size: 22, color: C.expl, alpha: a * st.cat });
        }
        if (st.formula > 0) {
          const x = 1120, y = 470, al = a * st.formula;
          D.text(ctx, S.t("exact rank-one gain (Proposition 13)", "精确的秩一边际收益（命题 13）"), x, y, { size: 19, color: C.ink, weight: 500, alpha: al });
          D.math(ctx, "v = P_S ℓ_a, \\quad s = ℓ_a^\\top P_S ℓ_a + r_a", x, y + 40, { size: 17, color: C.latent, alpha: al });
          D.math(ctx, "Q_{S ∪ a} = Q_S + v v^\\top / s", x, y + 74, { size: 17, color: C.expl, alpha: al });
          D.math(ctx, "Δ_{mean}(a \\mid S) = \\frac{(ω^\\top P_S ℓ_a)^2}{s}", x, y + 116, { size: 19, color: C.gold, alpha: al });
          D.text(ctx, S.t("P_S = K − Q_S is the residual covariance: what is still unexplained", "P_S = K − Q_S 是残差协方差：还没被解释的部分"), x, y + 144, { size: 14, color: C.muted, alpha: al });
        }
        if (st.gains > 0) {
          const gy = R.y + R.h + 218, gh = 100, step = CPV.clamp(st.step + 1, 0, this.gainTable.length - 1);
          const g = this.gainTable[step], gmax = Math.max(...this.gainTable[0]);
          D.text(ctx, S.t("marginal gain of each candidate, given the measurements so far", "给定已有测量，每个候选的边际收益"), R.x, gy - gh - 16, { size: 16, color: C.muted, alpha: a * st.gains });
          this.cands.forEach((c, i) => { const x = X(this.xs[c.idx]), h = gh * g[i] / gmax, best = g[i] === Math.max(...g) && g[i] > 0; D.rect(ctx, x - 10, gy - h, 20, h, { fill: best ? C.gold : CPV.rgba(C.expl, 0.55), radius: 3, alpha: a * st.gains }); });
          if (st.step >= 0) { const s = this.stepsMean[st.step]; D.text(ctx, S.t("step " + (st.step + 1) + ": add t = " + this.xs[s.action.idx].toFixed(1) + ", gain " + s.gain.toFixed(4) + ", value " + s.value.toFixed(3), "第 " + (st.step + 1) + " 步：加入 t = " + this.xs[s.action.idx].toFixed(1) + "，收益 " + s.gain.toFixed(4) + "，价值 " + s.value.toFixed(3)), R.x + R.w, gy - gh - 16, { size: 16, color: C.gold, font: "mono", align: "right", alpha: a * st.gains }); }
        }
        if (st.compare > 0) {
          const al = a * st.compare, y = R.y + R.h + 120;
          [[S.t("\\text{mean target} \\quad g(z) = z", "\\text{均值目标} \\quad g(z) = z"), this.stepsMean, C.expl], [S.t("\\text{occupation target} \\quad g(z) = \\mathbf{1}\\{z > 1\\}", "\\text{占用时间目标} \\quad g(z) = \\mathbf{1}\\{z > 1\\}"), this.stepsOcc, C.target]].forEach((row, k) => {
            const yy = y + k * 90;
            D.math(ctx, row[0], R.x, yy - 18, { size: 17, color: row[2], alpha: al });
            D.line(ctx, R.x, yy + 16, R.x + R.w, yy + 16, { color: C.dim, width: 2, alpha: al });
            row[1].forEach((s, i) => S.mark(ctx, X(this.xs[s.action.idx]), yy, yy + 32, row[2], { alpha: al * CPV.clamp(st.compare * 4 - i, 0, 1), width: 5, blur: 10, label: String(i + 1), labelSize: 14, labelBelow: true }));
          });
          D.text(ctx, S.t("same K, same catalogue, same budget — the two targets select different times", "同一个 K、同一个候选目录、同样的预算——两种目标选出不同的时刻"), R.x + R.w / 2, y + 196, { size: 19, color: C.gold, align: "center", alpha: al });
        }
      }
      if (st.caution > 0) {
        const a = st.caution;
        D.rect(ctx, 0, 0, 1600, 900, { fill: CPV.rgba(C.bg, 0.97 * a) });
        S.header(ctx, "Monotone, but not submodular — and how close greedy gets", "单调但不次模——以及贪心离最优有多近", { alpha: a });
        S.scope(ctx, "sim", a);
        D.paragraph(ctx, S.t("Lemma 14: adding measurements never lowers F_g. But returns need not diminish: over " + CPV_DATA.num.SubmodTriples + " triples per cell, " + CPV_DATA.num.SubmodCellsViol + " of " + CPV_DATA.num.SubmodCells + " kernel–target cells contain an action whose gain increases after another action is added. No approximation ratio is claimed for greedy; instead the search is measured against exhaustive enumeration.", "引理 14：增加测量不会降低 F_g。但收益未必递减：每格 " + CPV_DATA.num.SubmodTriples + " 个三元组中，" + CPV_DATA.num.SubmodCells + " 个核–目标组合里有 " + CPV_DATA.num.SubmodCellsViol + " 个出现“加入另一个动作后收益反而增大”的情形。论文不对贪心声称近似比，而是用穷举搜索来度量。"), 100, 150, 1400, { size: 19, color: C.ink, alpha: a, lineHeight: 30 });
        const names = { mutual_information: S.t("latent-state mutual information (target-free)", "潜在状态互信息（与目标无关）"), imse: S.t("integrated posterior variance (target-free)", "积分后验方差（与目标无关）"), linear_target: S.t("actual-noise linear target", "带真实噪声的线性目标"), kernel_quadrature: S.t("noiseless kernel quadrature", "无噪声核求积"), label_aware_greedy: S.t("target-aware greedy", "针对目标的贪心"), label_aware: S.t("target-aware greedy + one swap", "针对目标的贪心 + 一轮交换"), linear_target: S.t("actual-noise linear target", "带真实噪声的线性目标"), linear: S.t("actual-noise linear target", "带真实噪声的线性目标") };
        const x0 = 560, y0 = 300, bw = 700, rowH = 62;
        D.text(ctx, S.t("relative efficiency I(S)/I(S*) over 25 enumerated instances — minimum (bar), mean, median", "25 个穷举实例上的相对效率 I(S)/I(S*)——最小值（条形）、均值、中位数"), 100, y0 - 30, { size: 16, color: C.muted, alpha: a });
        this.methods.forEach((m, i) => {
          const y = y0 + i * rowH, isA = /label_aware|greedy/.test(m.method), col = isA ? C.expl : (m.min < 0.5 ? C.alert : C.muted);
          D.text(ctx, names[m.method] || m.method, x0 - 20, y + 28, { size: 17, color: isA ? C.ink : C.muted, align: "right", alpha: a });
          D.rect(ctx, x0, y + 8, bw * m.min * CPV.clamp(a * 2 - i * 0.15, 0, 1), 30, { fill: CPV.rgba(col, 0.6), radius: 4, alpha: a });
          D.text(ctx, m.min.toFixed(3), x0 + bw * m.min + 12, y + 30, { size: 18, color: col, font: "mono", weight: 500, alpha: a });
          D.text(ctx, S.t("mean ", "均值 ") + m.mean.toFixed(3) + S.t("  median ", "  中位数 ") + m.median.toFixed(3), x0 + bw + 110, y + 30, { size: 14, color: C.dim, font: "mono", alpha: a });
        });
        D.line(ctx, x0 + bw, y0, x0 + bw, y0 + rowH * this.methods.length, { color: C.gold, width: 1.5, dash: [6, 6], alpha: a });
        D.text(ctx, S.t("exhaustive optimum", "穷举最优"), x0 + bw, y0 + rowH * this.methods.length + 24, { size: 14, color: C.gold, align: "center", alpha: a });
        D.text(ctx, S.t("greedy + one swap evaluates " + CPV_DATA.num.DesignEvalFracHetero + "–" + CPV_DATA.num.DesignEvalFracStat + " of the enumerated sets and reaches the optimum in three of five settings", "贪心 + 一轮交换只评估了穷举集合的 " + CPV_DATA.num.DesignEvalFracHetero + "–" + CPV_DATA.num.DesignEvalFracStat + "，在五种设定中的三种达到最优"), 800, 790, { size: 17, color: C.gold, align: "center", alpha: a });
      }
    },
  };
})();
