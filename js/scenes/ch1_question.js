/* Chapter 1 — what a benchmark fixes: the sleep example. */
(function () {
  const CPV = window.CPV, C = CPV.C, D = CPV.D, S = CPV.S;
  const st = { hyp: 0, rem: 0, theta: 0, hypDim: 1, protA: 0, pairs: 0, protB: 0, q: 0, chain: 0, no: 0, why: 0, hypShow: 1 };
  const X0 = 140, X1 = 1460, TMAX = 8, ROWY = { W: 190, R: 250, N1: 310, N2: 370, N3: 430 }, AXY = 480;
  const tx = h => X0 + (X1 - X0) * h / TMAX;

  CPV.scenes.ch1 = {
    setup() { this.stages = S.toyHypnogram(170, 11); this.remFrac = this.stages.filter(s => s === "R").length / this.stages.length; },
    enter(stage) {
      Object.assign(st, { hyp: 0, rem: 0, theta: 0, hypDim: 1, protA: 0, pairs: 0, protB: 0, q: 0, chain: 0, no: 0, why: 0, hypShow: 1 });
    },
    beats: [
      function (stage) { Object.assign(st, { hyp: 0, rem: 0, theta: 0, hypDim: 1, protA: 0, pairs: 0, protB: 0, q: 0, chain: 0, no: 0, why: 0, hypShow: 1 });
        stage.tween(st, { hyp: 1 }, 2600, { ease: "linear" }); stage.tween(st, { rem: 1 }, 700, { delay: 3000 }); stage.tween(st, { theta: 1 }, 700, { delay: 3600 }); },
      function (stage) { stage.tween(st, { hypDim: 0.28, theta: 0.35 }, 700); stage.tween(st, { protA: 1 }, 800, { delay: 500 }); stage.tween(st, { pairs: 1 }, 1600, { delay: 1600, ease: "out" }); },
      function (stage) { stage.tween(st, { protB: 1 }, 1200, { delay: 300 }); stage.tween(st, { q: 1 }, 700, { delay: 2000 }); },
      function (stage) { stage.tween(st, { hypShow: 0, q: 0 }, 500); stage.tween(st, { chain: 1 }, 2600, { delay: 500, ease: "linear" }); stage.tween(st, { no: 1 }, 600, { delay: 3600, ease: "back" }); },
      function (stage) { stage.tween(st, { chain: 0, no: 0 }, 400); stage.tween(st, { why: 1 }, 900, { delay: 500 }); },
    ],
    draw(stage, t) {
      const ctx = stage.ctx, n = this.stages.length, segW = (X1 - X0) / n;
      S.header(ctx, "A night of sleep, fully scored", "整夜睡眠：完整轨迹与聚合目标", { alpha: st.hypShow, sub: ["latent trajectory Z(t): one sleep stage every 30 s, about 960 epochs per night", "潜在轨迹 Z(t)：每 30 秒一个睡眠分期，整夜约 960 个片段"] });
      if (st.hypShow > 0) {
        const a = st.hypShow;
        S.scope(ctx, "schematic", a);
        // stage rows + axis
        S.STAGES.forEach(s => D.text(ctx, s, X0 - 18, ROWY[s] + 6, { size: 17, color: C.muted, align: "right", font: "mono", alpha: a }));
        D.axis(ctx, X0, AXY, X1 - X0, { ticks: [0, 2, 4, 6, 8].map(h => ({ u: h / TMAX, label: h + " h" })), alpha: a });
        const shown = Math.floor(n * st.hyp);
        for (let i = 0; i < shown; i++) {
          const s = this.stages[i], isR = s === "R";
          const col = isR ? C.target : C.latent, al = (isR ? 1 : 0.5) * st.hypDim * a;
          D.line(ctx, X0 + i * segW, ROWY[s], X0 + (i + 1) * segW, ROWY[s], { color: col, width: isR && st.rem > 0 ? 7 + 3 * st.rem : 6, alpha: al, cap: "butt" });
          if (i > 0 && this.stages[i - 1] !== s) D.line(ctx, X0 + i * segW, ROWY[this.stages[i - 1]], X0 + i * segW, ROWY[s], { color: C.dim, width: 1, alpha: 0.6 * st.hypDim * a });
        }
        if (st.theta > 0) {
          const y = 604, al = st.theta * a;
          D.math(ctx, "Θ", X0, y, { size: 34, color: C.target, alpha: al });
          D.text(ctx, S.t("= REM epochs / all epochs  =", "= REM 片段数 / 全部片段数  ="), X0 + 44, y - 2, { size: 24, color: C.target, alpha: al });
          D.text(ctx, this.remFrac.toFixed(3), X0 + 44 + D.measure(ctx, S.t("= REM epochs / all epochs  =", "= REM 片段数 / 全部片段数  ="), { size: 24 }) + 16, y, { size: 30, color: C.target, font: "mono", weight: 500, alpha: al });
          D.text(ctx, S.t("The target is not the label of one moment; it is an aggregate of the whole trajectory.", "目标不是某一瞬间的标签，而是整条轨迹的聚合量。"), X0, y + 44, { size: 20, color: C.gold, alpha: al });
        }
        if (st.protA > 0) {
          S.mark(ctx, tx(0.5), 165, AXY - 8, C.A, { alpha: st.protA * a, label: S.t("A: one epoch, 30 min after onset", "A：入睡后 30 分钟的一个片段"), labelSize: 18 });
          D.math(ctx, "Y_A", tx(0.5), AXY + 58, { size: 22, color: C.A, align: "center", alpha: st.protA * a });
        }
        if (st.pairs > 0) {
          const nPairs = Math.round(10000 * st.pairs);
          const nStr = nPairs.toLocaleString("en-US");
          D.math(ctx, S.t("\\text{benchmark = " + nStr + " pairs }(Y_A, Θ)", "\\text{基准数据 = " + nStr + " 个配对样本 }(Y_A, Θ)"), 800, 706, { size: 26, color: C.ink, align: "center", alpha: Math.min(1, st.pairs * 3) * a });
          D.text(ctx, S.t("one measurement per subject · one trajectory-level target per subject", "每人一个测量值 · 每人一个轨迹级目标"), 800, 740, { size: 17, color: C.muted, align: "center", alpha: Math.min(1, st.pairs * 3) * a });
        }
        if (st.protB > 0) {
          [1, 3, 5, 7].forEach((h, i) => S.mark(ctx, tx(h), 165, AXY - 8, C.B, { alpha: CPV.clamp(st.protB * 4 - i, 0, 1) * a, dash: [9, 7], width: 3.5, label: i === 2 ? S.t("B: one epoch at hours 1, 3, 5, 7 — never deployed", "B：第 1、3、5、7 小时各一个片段——从未实施") : undefined, labelSize: 18 }));
        }
        if (st.q > 0) D.takeaway(ctx, S.t("How well would protocol B predict Θ?  Only A's data exist.", "协议 B 能把 Θ 预测到多好？手上只有 A 的数据。"), 800, 820, { alpha: st.q * a, size: 28 });
      }
      if (st.chain > 0) {
        const items = ["n = 100", "10 000", "1 000 000", "n → ∞"], y = 330;
        D.text(ctx, S.t("Intuition: with enough subjects under A, the answer must emerge.", "直觉：协议 A 下样本够多，总能算出来吧？"), 800, 230, { size: 28, color: C.muted, align: "center", alpha: Math.min(1, st.chain * 4) });
        items.forEach((s, i) => {
          const a = CPV.clamp(st.chain * 4 - i, 0, 1), x = 330 + i * 310;
          D.text(ctx, s, x, y, { size: 34, color: C.A, align: "center", font: "mono", alpha: a });
          if (i < 3) D.arrow(ctx, x + 95, y - 11, x + 215, y - 11, { color: C.dim, width: 2.5, alpha: a * CPV.clamp(st.chain * 4 - i - 0.5, 0, 1) });
        });
        if (st.no > 0) {
          D.text(ctx, S.t("No.", "不行。"), 800, 470 + (1 - st.no) * 20, { size: 92, color: C.alert, align: "center", font: "display", weight: 600, alpha: st.no });
          D.text(ctx, S.t("Even infinite data under A need not determine B's value.", "即使 A 下的数据无穷多，也未必能确定 B 的价值。"), 800, 540, { size: 24, color: C.ink, align: "center", alpha: st.no });
        }
      }
      if (st.why > 0) {
        const a = st.why;
        // schematic: A sees one time; B needs the relation between two times
        const y = 330;
        D.axis(ctx, 260, y + 90, 1080, { ticks: [0, 0.25, 0.5, 0.75, 1].map(u => ({ u, label: (u * 8) + " h" })), alpha: a });
        S.mark(ctx, 260 + 1080 * (0.5 / 8), y - 40, y + 80, C.A, { alpha: a, label: "A", labelSize: 20 });
        [1, 3, 5, 7].forEach(h => S.mark(ctx, 260 + 1080 * h / 8, y - 40, y + 80, C.B, { alpha: a, dash: [9, 7], width: 3.5 }));
        const xa = 260 + 1080 * 1 / 8, xb = 260 + 1080 * 3 / 8;
        ctx.save(); ctx.globalAlpha = a; ctx.strokeStyle = C.gold; ctx.lineWidth = 2.5; ctx.setLineDash([6, 6]);
        ctx.beginPath(); ctx.moveTo(xa, y - 50); ctx.quadraticCurveTo((xa + xb) / 2, y - 140, xb, y - 50); ctx.stroke(); ctx.restore();
        D.text(ctx, S.t("how are two times related?", "两个时刻之间如何相关？"), (xa + xb) / 2, y - 118, { size: 19, color: C.gold, align: "center", alpha: a });
        const lines = [
          [S.t("Protocol A never observes two different times of the same night,", "协议 A 从未在同一夜里观测过两个不同时刻，"), C.ink],
          [S.t("so it does not determine the joint dependence among the unobserved times.", "所以它无法确定未观测时刻之间的联合依赖。"), C.ink],
          [S.t("Protocol B's value depends on exactly that.", "而协议 B 的价值恰恰取决于这一点。"), C.gold],
        ];
        lines.forEach((l, i) => D.text(ctx, l[0], 800, 560 + i * 44, { size: 26, color: l[1], align: "center", alpha: a * CPV.clamp(a * 3 - i * 0.8, 0, 1) }));
        D.text(ctx, S.t("The first question is whether the benchmark contains enough information to evaluate a new sampling scheme.", "第一个问题是：基准数据是否包含足够的信息来评估一种新的采样方案。"), 800, 760, { size: 20, color: C.muted, align: "center", alpha: a });
      }
    },
  };
})();
