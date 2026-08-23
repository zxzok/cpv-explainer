/* Chapter 0 — prologue.  A WebGL field of latent trajectories; protocols are where we look. */
(function () {
  const CPV = window.CPV, C = CPV.C, D = CPV.D, M = CPV.M, S = CPV.S;
  const R = { x: 60, y: 150, w: 1480, h: 520 }, T = 24;
  const st = { reveal: 0, title: 1, band: 0, ticksA: 0, ticksB: 0, question: 0, brace: 0 };
  const ticksA = [3.5, 9, 14.5, 20], ticksB = [2, 6, 10, 12.5, 17, 22];

  CPV.scenes.ch0 = {
    useGL: true, glRect: R,
    setup(stage) {
      const p = 96, K = M.traitState(M.ouKernel(p, 3.2, T), 0.3), L = M.cholesky(K, 1e-9), rand = M.rng(7);
      this.xs = Float32Array.from(M.grid(p, T));
      const paths = M.samplePaths(L, 260, rand);
      this.ys = paths.map(z => Float32Array.from(z));
      this.colors = paths.map((_, i) => i % 37 === 0 ? CPV.glColor(C.target, 0.55) : CPV.glColor(C.latent, 0.16));
    },
    enter(stage) {
      Object.assign(st, { reveal: 0, title: 1, band: 0, ticksA: 0, ticksB: 0, question: 0, brace: 0 });
      stage.field.setView(0, T, -3.6, 3.6);
      this.layer = stage.field.addLayer(stage.field.makeLayer(this.xs, this.ys, this.colors, { width: 1.6, reveal: 0 }));
      stage.tween(st, { reveal: T }, 4200, { ease: "out" });
    },
    beats: [
      function (stage) { stage.tween(st, { title: 1 }, 400); if (st.reveal >= T) { st.reveal = 0; stage.tween(st, { reveal: T }, 4200, { ease: "out" }); } },
      function (stage) { stage.tween(st, { title: 0 }, 500); stage.tween(st, { band: 1 }, 700, { delay: 300 }); stage.tween(st, { ticksA: 1 }, 900, { delay: 1200 }); stage.tween(st, { brace: 1 }, 700, { delay: 2600 }); },
      function (stage) { stage.tween(st, { ticksB: 1 }, 900, { delay: 200 }); stage.tween(st, { question: 1 }, 700, { delay: 1400 }); },
    ],
    draw(stage, t) {
      const ctx = stage.ctx;
      this.layer.reveal = st.reveal;
      const X = u => R.x + R.w * u / T;
      // horizon axis
      D.axis(ctx, R.x, R.y + R.h + 12, R.w, { ticks: [0, 6, 12, 18, 24].map(h => ({ u: h / T, label: h + " h" })), alpha: 0.9 });
      D.text(ctx, S.t("horizon of one unit (a night, a day, a week)", "一个对象的时间跨度（一夜、一天、一周）"), R.x + R.w, R.y + R.h + 60, { size: 16, color: C.muted, align: "right" });
      // title card over the band
      if (st.title > 0) {
        const a = st.title;
        D.rect(ctx, 0, R.y + 90, 1600, 320, { fill: CPV.rgba(C.bg, 0.72 * a) });
        D.text(ctx, S.t("Counterfactual Evaluation of", "时间观测协议的"), 800, R.y + 200, { size: 54, font: "display", weight: 500, color: C.ink, align: "center", alpha: a });
        D.text(ctx, S.t("Temporal Observation Protocols", "反事实评估"), 800, R.y + 268, { size: 54, font: "display", weight: 500, color: C.ink, align: "center", alpha: a });
        D.text(ctx, S.t("Xizhe Zhang · Nanjing Medical University", "张锡哲 · 南京医科大学"), 800, R.y + 330, { size: 20, color: C.muted, align: "center", alpha: a });
      }
      if (st.band > 0) {
        D.chip(ctx, S.t("latent trajectory Z(t), one unit per line", "潜在轨迹 Z(t)，每条线是一个对象"), R.x + 10, R.y + 24, { color: C.latent, alpha: st.band });
        D.chip(ctx, S.t("target Θ = aggregate over the whole horizon", "目标 Θ = 整个时间跨度上的聚合量"), R.x + R.w - 10, R.y + 24, { color: C.target, alpha: st.band, align: "right" });
      }
      if (st.brace > 0) {
        const y = R.y + R.h - 14;
        D.line(ctx, X(0.2), y, X(T - 0.2), y, { color: C.target, width: 2, alpha: 0.8 * st.brace });
        D.line(ctx, X(0.2), y - 10, X(0.2), y, { color: C.target, width: 2, alpha: 0.8 * st.brace });
        D.line(ctx, X(T - 0.2), y - 10, X(T - 0.2), y, { color: C.target, width: 2, alpha: 0.8 * st.brace });
        D.math(ctx, "Θ = \\int ω(t)\\,g(Z(t))\\,\\text{d}t", X(T / 2), y - 18, { size: 22, color: C.target, align: "center", alpha: st.brace });
      }
      ticksA.forEach((h, i) => S.mark(ctx, X(h), R.y + 70, R.y + R.h - 40, C.A, { alpha: st.ticksA * CPV.clamp((st.ticksA * 4 - i), 0, 1), label: i === 0 ? S.t("protocol A: where the benchmark looked", "协议 A：基准数据实际观测的位置") : undefined, labelSize: 17 }));
      if (st.ticksA > 0) ticksA.forEach(h => D.circle(ctx, X(h), R.y + R.h / 2, 7, { fill: C.A, alpha: st.ticksA }));
      ticksB.forEach((h, i) => S.mark(ctx, X(h), R.y + 70, R.y + R.h - 40, C.B, { alpha: st.ticksB * CPV.clamp((st.ticksB * 6 - i), 0, 1), dash: [10, 8], width: 3, label: i === 3 ? S.t("protocol B: never run", "协议 B：从未实施") : undefined, labelSize: 17 }));
      if (st.question > 0) {
        D.takeaway(ctx, S.t("Do data collected under A determine how well B would predict Θ?", "在 A 下采集的数据，能否确定 B 会把 Θ 预测到多准？"), 800, 800, { alpha: st.question, size: 30 });
      }
    },
  };
})();
