/* Chapter 2 — protocol value: Definition 1 and the Gaussian identity (Proposition 2), with a live calculator. */
(function () {
  const CPV = window.CPV, C = CPV.C, D = CPV.D, M = CPV.M, S = CPV.S;
  const p = 32, T = 1;
  const st = { def: 0, gauge: 0, gaugeVal: 0, K: 0, omega: 0, Arows: 0, Q: 0, f2: 0, inter: 0, hint: 0 };
  const HM = { x: 120, y: 190, s: 300 }, HQ = { x: 560, y: 190, s: 300 }, TL = { x: 120, y: 720, w: 740 };
  const NOISE = 0.15;

  CPV.scenes.ch2 = {
    setup() {
      this.K = M.traitState(M.ouKernel(p, 0.16, T), 0.2);
      this.w = M.uniformWeights(p);
      this.idx = [5, 15, 25];
      this.drag = -1;
      this.recompute();
    },
    recompute() {
      const A = M.rowsFromIndices(this.idx, p), R = M.diagR(this.idx.length, NOISE);
      this.Q = M.Q_of(this.K, A, R);
      this.vMean = M.value(this.K, A, R, this.w, { kind: "mean" });
      this.vOcc = M.value(this.K, A, R, this.w, { kind: "occ", c: 0 });
    },
    enter(stage) { Object.assign(st, { def: 0, gauge: 0, gaugeVal: 0, K: 0, omega: 0, Arows: 0, Q: 0, f2: 0, inter: 0, hint: 0 }); },
    beats: [
      function (stage) { Object.assign(st, { K: 0, omega: 0, Arows: 0, Q: 0, f2: 0, inter: 0, hint: 0 }); stage.tween(st, { def: 1 }, 700); stage.tween(st, { gauge: 1 }, 600, { delay: 900 }); stage.tween(st, { gaugeVal: this.vMean }, 1500, { delay: 1500, ease: "out" }); },
      function (stage) { stage.tween(st, { def: 0.35 }, 500); stage.tween(st, { K: 1 }, 900, { delay: 300 }); stage.tween(st, { omega: 1 }, 700, { delay: 1500 }); stage.tween(st, { Arows: 1 }, 800, { delay: 2600 }); },
      function (stage) { stage.tween(st, { Q: 1 }, 1000, { delay: 200 }); stage.tween(st, { f2: 1 }, 700, { delay: 1200 }); stage.tween(st, { gaugeVal: this.vMean }, 800, { delay: 1400 }); },
      function (stage) { stage.tween(st, { omega: 0, Arows: 0 }, 400); stage.tween(st, { inter: 1 }, 700, { delay: 200 }); stage.tween(st, { hint: 1 }, 600, { delay: 1100 }); },
    ],
    onPointer(stage, type, x, y) {
      if (st.inter <= 0.5) return false;
      const u2x = i => TL.x + TL.w * (i + 0.5) / p;
      if (type === "down") {
        let best = -1, bd = 30;
        this.idx.forEach((i, k) => { const d = Math.hypot(x - u2x(i), y - TL.y); if (d < bd) { bd = d; best = k; } });
        this.drag = best; return best >= 0;
      }
      if (type === "move" && this.drag >= 0) {
        let i = Math.round((x - TL.x) / TL.w * p - 0.5); i = CPV.clamp(i, 0, p - 1);
        if (!this.idx.some((j, k) => k !== this.drag && j === i) && i !== this.idx[this.drag]) { this.idx[this.drag] = i; this.recompute(); st.gaugeVal = this.vMean; }
        return true;
      }
      if (type === "up" || type === "leave") { this.drag = -1; }
      return false;
    },
    draw(stage) {
      const ctx = stage.ctx, K = this.K;
      S.header(ctx, "Protocol value: the population R² a protocol can support", "协议价值：一个协议所能支持的总体 R²");
      // definition
      if (st.def > 0) {
        const a = st.def, y = 160;
        D.math(ctx, "I(S) = \\frac{Var(E[Θ \\mid Y_S])}{Var(Θ)}", 1000, y + 6, { size: 28, color: C.ink, alpha: a });
        D.text(ctx, S.t("population R² of the best predictor of Θ from the protocol's measurements", "用该协议的测量预测 Θ 的最优预测器的总体 R²"), 1000, y + 62, { size: 16, color: C.muted, alpha: a });
        D.text(ctx, S.t("a property of the protocol — not of a learner, not of a sample size", "是协议的性质——与模型、样本量无关"), 1000, y + 86, { size: 16, color: C.gold, alpha: a });
      }
      if (st.gauge > 0) D.gauge(ctx, 1060, 300, 360, 30, st.gaugeVal, { color: C.expl, label: "I(S)", labelMath: true, alpha: st.gauge });
      // K heat map + weights + protocol rows
      if (st.K > 0) {
        D.heatmap(ctx, K, HM.x, HM.y, HM.s, { alpha: st.K, title: "K \\quad \\text{(latent correlation, } p = 32)", titleMath: true, cellStroke: st.Arows > 0 ? ((j, k) => (this.idx.includes(j) && this.idx.includes(k)) ? CPV.rgba(C.A, st.Arows) : null) : null });
        if (st.Arows > 0) this.idx.forEach(i => { const cy = HM.y + HM.s * (i + 0.5) / p; D.line(ctx, HM.x - 14, cy, HM.x - 4, cy, { color: C.A, width: 4, alpha: st.Arows }); });
        D.math(ctx, "Z \\sim N(0, K), \\quad diag(K) = 1", HM.x, HM.y + HM.s + 34, { size: 18, color: C.latent, alpha: st.K });
      }
      if (st.omega > 0) {
        const y = HM.y + HM.s + 70;
        D.math(ctx, "ω", HM.x - 26, y + 14, { size: 22, color: C.target, align: "right", alpha: st.omega });
        for (let j = 0; j < p; j++) D.rect(ctx, HM.x + HM.s * j / p + 1, y, HM.s / p - 2, 18, { fill: C.target, alpha: 0.75 * st.omega });
        D.math(ctx, "Θ = \\sum_j ω_j\\, g(Z_j)", HM.x, y + 48, { size: 18, color: C.target, alpha: st.omega });
      }
      if (st.Arows > 0) {
        const y = HM.y + HM.s + 150;
        D.math(ctx, "A_S", HM.x - 26, y + 8, { size: 22, color: C.A, align: "right", alpha: st.Arows });
        D.line(ctx, HM.x, y, HM.x + HM.s, y, { color: C.dim, width: 1.5, alpha: st.Arows });
        this.idx.forEach(i => S.mark(ctx, HM.x + HM.s * (i + 0.5) / p, y - 16, y + 16, C.A, { alpha: st.Arows, width: 4, blur: 8 }));
        D.math(ctx, "Y_S = A_S Z + ε, \\quad ε \\sim N(0, R_S)", HM.x, y + 42, { size: 18, color: C.A, alpha: st.Arows });
      }
      if (st.Q > 0) {
        D.heatmap(ctx, this.Q, HQ.x, HQ.y, HQ.s, { alpha: st.Q, title: "Q_S = Cov(E[Z \\mid Y_S])", titleMath: true, colorHi: C.expl });
        D.math(ctx, "Q_S(K) = K A_S^\\top (A_S K A_S^\\top + R_S)^{-1} A_S K", HQ.x, HQ.y + HQ.s + 36, { size: 17, color: C.expl, alpha: st.Q });
        D.text(ctx, S.t("the part of the trajectory recovered from Y_S", "从 Y_S 中恢复出的那部分轨迹"), HQ.x, HQ.y + HQ.s + 58, { size: 16, color: C.muted, alpha: st.Q });
      }
      if (st.f2 > 0) {
        const a = st.f2, x = 1000, y = 400;
        D.math(ctx, "I_g(S;K) = \\frac{F_g(S;K)}{V_g(K)}", x, y + 4, { size: 24, color: C.ink, alpha: a });
        D.math(ctx, "F_g = \\sum_{j,k} ω_j ω_k\\, C_g(Q_{jk})", x, y + 52, { size: 20, color: C.expl, alpha: a });
        D.math(ctx, "V_g = \\sum_{j,k} ω_j ω_k\\, C_g(K_{jk})", x, y + 84, { size: 20, color: C.target, alpha: a });
        D.math(ctx, "C_g(r) = Cov\\{g(U), g(V_r)\\}", x, y + 122, { size: 18, color: C.muted, alpha: a });
        D.text(ctx, S.t("every nonlinearity of the target enters through C_g alone", "目标的全部非线性只通过 C_g 进入"), x, y + 146, { size: 16, color: C.muted, alpha: a });
        D.math(ctx, S.t("\\text{mean target: } C_g(r) = r", "\\text{均值目标：} C_g(r) = r"), x, y + 176, { size: 15, color: C.dim, alpha: a });
        D.math(ctx, S.t("\\text{occupation above 0: } C_g(r) = \\frac{arcsin(r)}{2π}", "\\text{零阈值占用时间：} C_g(r) = \\frac{arcsin(r)}{2π}"), x, y + 210, { size: 15, color: C.dim, alpha: a });
      }
      if (st.inter > 0) {
        const a = st.inter;
        D.rect(ctx, 80, 660, 1440, 200, { fill: CPV.rgba(C.panel, 0.9), stroke: C.grid, radius: 12, alpha: a });
        D.text(ctx, S.t("Try it: drag the three measurement times", "试一试：拖动三个测量时刻"), TL.x, 700, { size: 19, color: C.gold, weight: 500, alpha: a });
        D.line(ctx, TL.x, TL.y, TL.x + TL.w, TL.y, { color: C.dim, width: 2, alpha: a });
        [0, 0.25, 0.5, 0.75, 1].forEach(u => D.text(ctx, (u * 100).toFixed(0) + "%", TL.x + TL.w * u, TL.y + 34, { size: 14, color: C.dim, align: "center", font: "mono", alpha: a }));
        this.idx.forEach((i, k) => { const x = TL.x + TL.w * (i + 0.5) / p; D.glow(ctx, x, TL.y, 26, C.A, 0.35 * a); D.circle(ctx, x, TL.y, 12, { fill: C.A, stroke: C.ink, width: 2, alpha: a }); });
        D.gauge(ctx, 1090, 712, 300, 24, this.vMean, { color: C.expl, label: S.t("mean target", "均值目标"), labelSize: 17, alpha: a });
        D.gauge(ctx, 1090, 760, 300, 24, this.vOcc, { color: C.target, label: S.t("occupation target", "占用时间目标"), labelSize: 17, alpha: a });
        D.text(ctx, S.t("same K, same three measurements — different targets, different values", "同一个 K、同样三次测量——目标不同，价值不同"), 1090, 828, { size: 15, color: C.muted, alpha: a * st.hint });
      }
    },
  };
})();
