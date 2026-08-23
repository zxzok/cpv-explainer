/* Chapter 3 — Theorem 3: two observationally equivalent worlds that value protocol B differently. */
(function () {
  const CPV = window.CPV, C = CPV.C, D = CPV.D, M = CPV.M, S = CPV.S;
  const st = { grid: 0, blocks: 0, eps: 0, curves: 0, table: 0, B: 0, gauges: 0, gaugeAnim: 0, slider: 0, why: 0, main: 1 };
  const EPS0 = 0.1321;
  const PTS = { x: 120, y: 190, w: 560 }, CH = { x: 150, y: 365, w: 500, h: 240 }, SL = { x: 330, y: 826, w: 900 };

  CPV.scenes.ch3 = {
    setup() {
      // admissible range of epsilon (both worlds positive definite)
      let e = 0; while (e < 0.5 && M.fourPoint(e + 0.001).pd) e += 0.001;
      this.epsMax = Math.floor(e * 0.97 * 1000) / 1000;
      this.drag = false;
    },
    enter() { Object.assign(st, { grid: 0, blocks: 0, eps: 0, curves: 0, table: 0, B: 0, gauges: 0, gaugeAnim: 0, slider: 0, why: 0, main: 1 }); },
    beats: [
      function (stage) { Object.assign(st, { curves: 0, table: 0, B: 0, gauges: 0, gaugeAnim: 0, slider: 0, why: 0, main: 1, eps: 0 }); stage.tween(st, { grid: 1 }, 800); stage.tween(st, { blocks: 1 }, 800, { delay: 1600 }); },
      function (stage) { stage.tween(st, { curves: 1 }, 800); stage.tween(st, { eps: EPS0 }, 2200, { delay: 900, ease: "inOut" }); },
      function (stage) { stage.tween(st, { table: 1 }, 900); },
      function (stage) { stage.tween(st, { B: 1 }, 700); stage.tween(st, { gauges: 1 }, 600, { delay: 600 }); stage.tween(st, { gaugeAnim: 1 }, 1400, { delay: 1100, ease: "out" }); },
      function (stage) { stage.tween(st, { slider: 1 }, 700); },
      function (stage) { stage.tween(st, { why: 1 }, 900, { delay: 200 }); },
    ],
    onPointer(stage, type, x, y) {
      if (st.slider < 0.5) return false;
      const near = Math.abs(y - SL.y) < 40 && x > SL.x - 20 && x < SL.x + SL.w + 20;
      if (type === "down" && near) this.drag = true;
      if ((type === "move" && this.drag) || (type === "down" && near)) { st.eps = CPV.clamp((x - SL.x) / SL.w, 0, 1) * this.epsMax; stage.clearAnimations(); return true; }
      if (type === "up" || type === "leave") this.drag = false;
      return false;
    },
    draw(stage) {
      const ctx = stage.ctx, fp = M.fourPoint(st.eps), fp0 = M.fourPoint(0);
      S.header(ctx, "Theorem 3 — the smallest case where evaluation fails", "定理 3——评估失效的最小例子", { alpha: st.main });
      S.scope(ctx, "math", st.main);
      const px = i => PTS.x + PTS.w * i / 3;
      // the four points, protocol A at Z0, protocol B at Z1, Z2
      if (st.grid > 0) {
        const a = st.grid;
        D.line(ctx, PTS.x - 30, PTS.y, PTS.x + PTS.w + 30, PTS.y, { color: C.dim, width: 2, alpha: a });
        for (let i = 0; i < 4; i++) { D.circle(ctx, px(i), PTS.y, 9, { fill: C.latent, alpha: a }); D.math(ctx, "Z_" + i, px(i), PTS.y + 40, { size: 21, color: C.latent, align: "center", alpha: a }); }
        S.mark(ctx, px(0), PTS.y - 70, PTS.y - 18, C.A, { alpha: a, label: "A", labelSize: 20 });
        D.math(ctx, "Θ = \\frac{1}{4}(Z_0 + Z_1 + Z_2 + Z_3)", PTS.x, PTS.y + 86, { size: 19, color: C.target, alpha: a });
        D.math(ctx, S.t("\\text{stationary, standardised: } Corr(Z_j, Z_k) = ρ(|j − k|)", "\\text{平稳、标准化：} Corr(Z_j, Z_k) = ρ(|j − k|)"), PTS.x, PTS.y + 122, { size: 16, color: C.muted, alpha: a });
      }
      if (st.B > 0) [1, 2].forEach((i, k) => S.mark(ctx, px(i), PTS.y - 70, PTS.y - 18, C.B, { alpha: st.B, dash: [8, 6], width: 3.5, label: k === 0 ? "B" : undefined, labelSize: 20 }));
      // correlation-profile chart
      if (st.curves > 0 || st.grid > 0) {
        const a = Math.max(st.curves, 0.001);
        const map = S.chart(ctx, CH, { xr: [-0.2, 3.2], yr: [-0.3, 1.05], xticks: [0, 1, 2, 3].map(v => ({ v, label: "ρ(" + v + ")" })), yticks: [{ v: 0, label: "0" }, { v: 0.5, label: ".5" }, { v: 1, label: "1" }], alpha: a, title: S.t("correlation profile ρ(lag)", "相关函数 ρ(滞后)") });
        if (st.curves > 0) {
          const lags = [0, 1, 2, 3];
          S.series(ctx, map, lags, fp0.rho0, C.dim, 1, { dash: [5, 5], width: 2, r: 3, alpha: a });
          S.series(ctx, map, lags, fp.rhoPlus, C.worldP, 1, { alpha: a, label: "K₊" });
          S.series(ctx, map, lags, fp.rhoMinus, C.worldM, 1, { alpha: a, label: "K₋" });
          D.math(ctx, "ρ_+ = ρ_0 + ε\\,(0, 1, −2, 1), \\quad ρ_− = ρ_0 − ε\\,(0, 1, −2, 1)", CH.x, CH.y + CH.h + 76, { size: 16, color: C.muted, alpha: a });
          D.math(ctx, "ε = " + st.eps.toFixed(4), CH.x + CH.w, CH.y - 14, { size: 18, color: C.gold, align: "right", alpha: a });
        }
      }
      // what A can measure
      if (st.blocks > 0) {
        const a = st.blocks, x = 760, y = 200;
        D.text(ctx, S.t("Everything the benchmark can ever learn", "基准数据能学到的全部内容"), x, y, { size: 20, color: C.A, weight: 500, alpha: a });
        const rows = [["Var(Y_A)", "varY"], ["Cov(Y_A, Θ)", "covYT"], ["Var(Θ)", "varT"]];
        rows.forEach((r, i) => {
          const yy = y + 48 + i * 44;
          D.math(ctx, r[0], x, yy, { size: 20, color: C.ink, alpha: a });
          if (st.table > 0) {
            D.text(ctx, fp.obsPlus[r[1]].toFixed(6), x + 250, yy, { size: 20, color: C.worldP, font: "mono", alpha: st.table });
            D.text(ctx, fp.obsMinus[r[1]].toFixed(6), x + 440, yy, { size: 20, color: C.worldM, font: "mono", alpha: st.table });
          } else D.text(ctx, fp0.obsPlus[r[1]].toFixed(6), x + 250, yy, { size: 20, color: C.muted, font: "mono", alpha: a });
        });
        if (st.table > 0) {
          D.math(ctx, S.t("\\text{world } K_+", "\\text{世界 } K_+"), x + 250, y + 22, { size: 15, color: C.worldP, alpha: st.table });
          D.math(ctx, S.t("\\text{world } K_−", "\\text{世界 } K_−"), x + 440, y + 22, { size: 15, color: C.worldM, alpha: st.table });
          const disc = Math.max(...["varY", "covYT", "varT"].map(k => Math.abs(fp.obsPlus[k] - fp.obsMinus[k])));
          D.text(ctx, S.t("max discrepancy  ", "最大差异  ") + (disc < 1e-12 ? disc.toExponential(0) : disc.toExponential(2)), x, y + 196, { size: 18, color: C.alert, font: "mono", alpha: st.table });
          D.text(ctx, S.t("= floating-point noise: the two benchmark laws coincide", "= 浮点误差量级：两个基准分布完全相同"), x, y + 224, { size: 16, color: C.alert, alpha: st.table });
        }
      }
      // gauges for protocol B
      if (st.gauges > 0) {
        const a = st.gauges, x = 1060, y = 480;
        D.math(ctx, S.t("\\text{value of protocol } B = \\{Z_1, Z_2\\}", "\\text{协议 } B = \\{Z_1, Z_2\\} \\text{ 的价值}"), 760, y - 10, { size: 20, color: C.B, alpha: a });
        const vp = fp.valuePlus, vm = fp.valueMinus, k = st.gaugeAnim;
        D.gauge(ctx, x, y + 20, 300, 28, vp * k, { color: C.worldP, label: "I(B; K_+)", labelMath: true, alpha: a });
        D.gauge(ctx, x, y + 70, 300, 28, vm * k, { color: C.worldM, label: "I(B; K_−)", labelMath: true, alpha: a });
        D.text(ctx, S.t("difference ", "相差 ") + Math.abs(vm - vp).toFixed(4), x + 300 + 14, y + 130, { size: 20, color: C.alert, font: "mono", align: "right", alpha: a * k });
        D.text(ctx, S.t("same data, two answers → no A-data estimator is consistent in both worlds", "同样的数据，两个答案 → 没有任何只用 A 数据的估计器能在两个世界都相合"), 760, y + 172, { size: 16, color: C.ink, alpha: a * k });
      }
      // epsilon slider
      if (st.slider > 0) {
        const a = st.slider;
        D.line(ctx, SL.x, SL.y, SL.x + SL.w, SL.y, { color: C.dim, width: 4, alpha: a });
        const x0 = SL.x + SL.w * EPS0 / this.epsMax;
        D.line(ctx, x0, SL.y - 10, x0, SL.y + 10, { color: C.gold, width: 2, alpha: a });
        D.text(ctx, S.t("paper's ε", "论文取值"), x0, SL.y - 18, { size: 13, color: C.gold, align: "center", alpha: a });
        const hx = SL.x + SL.w * st.eps / this.epsMax;
        D.glow(ctx, hx, SL.y, 28, C.gold, 0.4 * a); D.circle(ctx, hx, SL.y, 13, { fill: C.gold, stroke: C.ink, width: 2, alpha: a });
        D.math(ctx, "ε = 0", SL.x, SL.y + 34, { size: 14, color: C.muted, alpha: a });
        D.math(ctx, "ε = " + this.epsMax.toFixed(3) + S.t("\\text{ (last admissible)}", "\\text{（可容许的上限）}"), SL.x + SL.w, SL.y + 34, { size: 14, color: C.muted, align: "right", alpha: a });
        D.text(ctx, S.t("drag ε: the three observed numbers stay frozen, B's value slides", "拖动 ε：三个可观测的数纹丝不动，B 的价值在滑动"), 800, SL.y - 46, { size: 18, color: C.gold, align: "center", alpha: a });
      }
      // why: redundancy between B's two measurements
      if (st.why > 0) {
        const a = st.why;
        D.rect(ctx, 740, 160, 800, 520, { fill: CPV.rgba(C.bg, 0.96), alpha: a });
        const cx = 1140, cy = 330;
        D.circle(ctx, cx, cy - 120, 30, { fill: CPV.rgba(C.target, 0.2), stroke: C.target, width: 2, alpha: a }); D.math(ctx, "Θ", cx, cy - 111, { size: 26, color: C.target, align: "center", alpha: a });
        const z1 = [cx - 150, cy + 60], z2 = [cx + 150, cy + 60];
        [z1, z2].forEach((z, i) => { D.circle(ctx, z[0], z[1], 30, { fill: CPV.rgba(C.B, 0.2), stroke: C.B, width: 2, alpha: a }); D.math(ctx, "Z_" + (i + 1), z[0], z[1] + 9, { size: 24, color: C.B, align: "center", alpha: a }); });
        D.line(ctx, z1[0] + 14, z1[1] - 26, cx - 18, cy - 96, { color: C.target, width: 3, alpha: a });
        D.line(ctx, z2[0] - 14, z2[1] - 26, cx + 18, cy - 96, { color: C.target, width: 3, alpha: a });
        D.text(ctx, S.t("fixed", "不变"), cx - 130, cy - 60, { size: 16, color: C.target, align: "center", alpha: a });
        D.text(ctx, S.t("fixed", "不变"), cx + 130, cy - 60, { size: 16, color: C.target, align: "center", alpha: a });
        const ov = fp.rhoPlus[1];
        D.line(ctx, z1[0] + 32, z1[1], z2[0] - 32, z2[1], { color: C.alert, width: 2 + 10 * CPV.clamp(ov, 0, 1), alpha: a });
        D.math(ctx, "Cov(Z_1, Z_2) = ρ(1) ± ε", cx, cy + 110, { size: 19, color: C.alert, align: "center", alpha: a });
        D.text(ctx, S.t("the only thing that moves: the overlap between B's two measurements", "唯一在变的：B 的两个测量之间的重叠"), cx, cy + 140, { size: 16, color: C.alert, align: "center", alpha: a });
        D.math(ctx, "I(B;K_±) = \\frac{2b^2}{(1 + ν_B^2 + ρ(1) ± ε)\\,Var(Θ)}, \\quad b = \\frac{1 + 2ρ(1) + ρ(2)}{4}", cx, cy + 222, { size: 17, color: C.muted, align: "center", alpha: a });
        D.paragraph(ctx, S.t("A protocol's value depends on how much its measurements overlap — and A's data never saw two times together.", "协议的价值取决于它的测量之间重叠多少——而 A 的数据从未同时看到两个时刻。"), cx - 360, cy + 270, 720, { size: 20, color: C.gold, alpha: a, lineHeight: 30 });
      }
    },
  };
})();
