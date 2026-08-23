/* Chapter 4 — invisible directions (Definition 4, Theorem 5), counting, and the permutation construction (Proposition 6, Example 7). */
(function () {
  const CPV = window.CPV, C = CPV.C, D = CPV.D, M = CPV.M, S = CPV.S;
  const st = { blocks: 0, heat: 0, lit: 0, count: 0, thm: 0, perm: 0, swap: 0, main: 1 };
  const P = 16, OBS = [1, 5, 9, 13];
  const HM = { x: 110, y: 180, s: 400 };

  CPV.scenes.ch4 = {
    setup() { this.K = M.ouKernel(P, 3.2, P); this.Ka = [[1, 0.6, 0], [0.6, 1, 0], [0, 0, 1]].map(r => Float64Array.from(r)); this.Kb = [[1, 0, 0.6], [0, 1, 0], [0.6, 0, 1]].map(r => Float64Array.from(r)); },
    enter() { Object.assign(st, { blocks: 0, heat: 0, lit: 0, count: 0, thm: 0, perm: 0, swap: 0, main: 1 }); },
    beats: [
      function (stage) { Object.assign(st, { count: 0, thm: 0, perm: 0, swap: 0, main: 1 }); stage.tween(st, { heat: 1 }, 900); stage.tween(st, { blocks: 1 }, 800, { delay: 900 }); stage.tween(st, { lit: 1 }, 1000, { delay: 2200 }); },
      function (stage) { stage.tween(st, { count: 1 }, 2400, { ease: "out" }); },
      function (stage) { stage.tween(st, { thm: 1 }, 900); },
      function (stage) { stage.tween(st, { main: 0 }, 500); stage.tween(st, { perm: 1 }, 800, { delay: 500 }); stage.tween(st, { swap: 1 }, 1400, { delay: 2200, ease: "inOut" }); },
    ],
    draw(stage, t) {
      const ctx = stage.ctx;
      if (st.main > 0) {
        const a = st.main;
        S.header(ctx, "Which directions of K can the benchmark see?", "基准数据能看到 K 的哪些方向？", { alpha: a });
        S.scope(ctx, "math", a);
        if (st.heat > 0) {
          const lit = st.lit;
          D.heatmap(ctx, this.K, HM.x, HM.y, HM.s, { alpha: a * st.heat, title: "K, \\quad p = 16", titleMath: true, cellAlpha: (j, k) => (OBS.includes(j) && OBS.includes(k)) ? 1 : 1 - 0.82 * lit, cellStroke: (j, k) => (OBS.includes(j) && OBS.includes(k)) ? CPV.rgba(C.A, lit) : null });
          OBS.forEach(i => { const cy = HM.y + HM.s * (i + 0.5) / P; D.line(ctx, HM.x - 16, cy, HM.x - 5, cy, { color: C.A, width: 4, alpha: a * st.heat }); });
          D.text(ctx, S.t("protocol A observes d = 4 of the 16 grid points", "协议 A 观测 16 个格点中的 d = 4 个"), HM.x, HM.y + HM.s + 36, { size: 17, color: C.A, alpha: a * st.heat });
          if (lit > 0) D.text(ctx, S.t("lit: what AKAᵀ pins down · dimmed: unconstrained by the benchmark", "亮：AKAᵀ 固定的部分 · 暗：基准数据不约束的部分"), HM.x, HM.y + HM.s + 62, { size: 15, color: C.muted, alpha: a * lit });
        }
        if (st.blocks > 0) {
          const x = 600, y = 200, al = a * st.blocks;
          D.text(ctx, S.t("The benchmark law depends on K through three blocks", "基准分布只通过三块依赖于 K"), x, y, { size: 21, color: C.ink, weight: 500, alpha: al });
          [["A K A^\\top", S.t("covariance among realised measurements", "已实施测量之间的协方差")], ["A K h", S.t("their covariance with the target", "它们与目标的协方差")], ["h^\\top K h", S.t("the target variance", "目标的方差")]].forEach((r, i) => {
            D.math(ctx, r[0], x, y + 52 + i * 46, { size: 24, color: C.A, alpha: al });
            D.text(ctx, r[1], x + 150, y + 52 + i * 46, { size: 17, color: C.muted, alpha: al });
          });
          D.math(ctx, S.t("Δ \\text{ is invisible } \\Leftrightarrow A Δ A^\\top = 0, \\; A Δ h = 0, \\; h^\\top Δ h = 0, \\; diag(Δ) = 0", "Δ \\text{ 不可见 } \\Leftrightarrow A Δ A^\\top = 0, \\; A Δ h = 0, \\; h^\\top Δ h = 0, \\; diag(Δ) = 0"), x, y + 212, { size: 18, color: C.gold, alpha: al });
        }
        if (st.count > 0) {
          const x = 600, y = 470, k = st.count;
          const free = P * (P - 1) / 2, cons = 4 * 5 / 2 + 4 + 1;
          D.text(ctx, S.t("free entries of K", "K 的自由参数"), x, y, { size: 18, color: C.latent, alpha: a });
          D.math(ctx, "p(p − 1)/2 = " + S.countUp(k, free), x + 330, y, { size: 26, color: C.latent, alpha: a });
          D.text(ctx, S.t("constraints from the benchmark", "基准数据给出的约束"), x, y + 44, { size: 18, color: C.A, alpha: a });
          D.math(ctx, "d(d + 1)/2 + d + 1 = " + S.countUp(k, cons), x + 330, y + 44, { size: 26, color: C.A, alpha: a });
          D.line(ctx, x, y + 66, x + 700, y + 66, { color: C.dim, width: 1.5, alpha: a });
          D.text(ctx, S.t("invisible directions", "不可见方向"), x, y + 100, { size: 18, color: C.alert, alpha: a });
          D.math(ctx, "≥ " + S.countUp(k, free - cons), x + 330, y + 100, { size: 30, color: C.alert, alpha: a });
          D.text(ctx, S.t("O(d²) constraints against O(p²) parameters: long horizons with sparse protocols leave most of K invisible", "O(d²) 个约束对 O(p²) 个参数：长跨度 + 稀疏采样，K 的大部分方向都看不见"), x, y + 140, { size: 15, color: C.muted, alpha: a * k });
        }
        if (st.thm > 0) {
          const x = 600, y = 680, al = a * st.thm;
          D.rect(ctx, x - 20, y - 40, 920, 160, { fill: CPV.rgba(C.panel, 0.9), stroke: C.grid, radius: 10, alpha: al });
          D.text(ctx, S.t("Theorem 5 — value-changing invisible directions imply non-identification", "定理 5——改变价值的不可见方向意味着不可识别"), x, y, { size: 19, color: C.ink, weight: 500, alpha: al });
          D.math(ctx, "D_B(Δ;K_0) = \\frac{\\text{d}}{\\text{d}ε} I(B; K_0 + εΔ)|_{ε=0} = \\frac{2h^\\top Δ B^\\top W B K_0 h − h^\\top K_0 B^\\top W (B Δ B^\\top) W B K_0 h}{h^\\top K_0 h}, \\quad W = (B K_0 B^\\top + R_B)^{-1}", x, y + 44, { size: 15, color: C.muted, alpha: al });
          D.text(ctx, S.t("D_B ≠ 0  ⇒  K₀ ± εΔ share the benchmark law and value B differently; more units under A cannot help.", "D_B ≠ 0  ⇒  K₀ ± εΔ 有相同的基准分布，却给 B 不同的价值；在 A 下再多收集对象也无济于事。"), x, y + 80, { size: 16, color: C.gold, alpha: al });
        }
      }
      if (st.perm > 0) {
        const a = st.perm, s = st.swap;
        S.header(ctx, "Nonlinear targets: an exact permutation argument (Proposition 6, Example 7)", "非线性目标：精确的置换构造（命题 6、例 7）", { alpha: a });
        S.scope(ctx, "math", a);
        const y = 300, x0 = 300, dx = 220;
        D.line(ctx, x0 - 40, y, x0 + 2 * dx + 40, y, { color: C.dim, width: 2, alpha: a });
        // Z1 and Z2 swap positions along arcs
        const pos = [x0, x0 + dx + dx * s, x0 + 2 * dx - dx * s];
        const lift = [0, -90 * Math.sin(Math.PI * s), 90 * Math.sin(Math.PI * s)];
        [0, 1, 2].forEach(i => { D.circle(ctx, pos[i], y + lift[i], 22, { fill: CPV.rgba(C.latent, 0.25), stroke: C.latent, width: 2, alpha: a }); D.math(ctx, "Z_" + i, pos[i], y + lift[i] + 8, { size: 22, color: C.latent, align: "center", alpha: a }); });
        S.mark(ctx, x0, y - 90, y - 30, C.A, { alpha: a, label: "A", labelSize: 20 });
        S.mark(ctx, x0 + dx, y + 30, y + 90, C.B, { alpha: a, dash: [8, 6], width: 3.5, label: "B", labelSize: 20, labelBelow: true });
        D.math(ctx, S.t("Ξ\\text{: swap coordinates 1 and 2;} \\quad A Ξ = A, \\quad Ξ^\\top ω = ω", "Ξ\\text{：交换坐标 1 与 2；} \\quad A Ξ = A, \\quad Ξ^\\top ω = ω"), x0 - 40, y + 150, { size: 18, color: C.gold, alpha: a });
        D.text(ctx, S.t("the permutation leaves every realised measurement and the aggregate weights unchanged", "置换不改变任何已实施的测量，也不改变聚合权重"), x0 - 40, y + 180, { size: 16, color: C.muted, alpha: a });
        // the two matrices
        const hx = 1000, hy = 180, hs = 150;
        D.heatmap(ctx, this.Ka, hx, hy, hs, { alpha: a, title: "K_a", titleMath: true, hi: 1 });
        D.heatmap(ctx, this.Kb, hx + 230, hy, hs, { alpha: a, title: "K_a' = Ξ K_a Ξ^\\top", titleMath: true, hi: 1 });
        D.math(ctx, "a = 0.6", hx + 190, hy + hs + 28, { size: 15, color: C.muted, align: "center", alpha: a });
        D.text(ctx, S.t("joint law of (Y_A, Θ_g):  identical for every g ∈ L²(φ)", "(Y_A, Θ_g) 的联合分布：对每个 g ∈ L²(φ) 都完全相同"), hx, hy + hs + 70, { size: 17, color: C.A, alpha: a * s });
        D.math(ctx, "I_g(B; K_a) > I_g(B; K_a')", hx, hy + hs + 108, { size: 22, color: C.alert, alpha: a * s });
        D.text(ctx, S.t("for every nonconstant g, every a ∈ (0,1), every noise level", "对每个非常数 g、每个 a ∈ (0,1)、每个噪声水平都成立"), hx, hy + hs + 136, { size: 15, color: C.muted, alpha: a * s });
        D.takeaway(ctx, S.t("Exact, not asymptotic; covers occupation-time targets; needs no stationarity — three grid points suffice.", "精确而非渐近；覆盖占用时间目标；不需要平稳性——三个格点就够。"), 800, 700, { alpha: a * s, size: 24, maxW: 1200 });
      }
    },
  };
})();
