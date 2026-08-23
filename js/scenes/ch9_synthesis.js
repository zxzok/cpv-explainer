/* Chapter 9 — synthesis: identification → calibration → design. */
(function () {
  const CPV = window.CPV, C = CPV.C, D = CPV.D, S = CPV.S;
  const st = { cols: 0, principle: 0, links: 0 };

  CPV.scenes.ch9 = {
    setup() {},
    enter() { Object.assign(st, { cols: 0, principle: 0, links: 0 }); },
    beats: [
      function (stage) { Object.assign(st, { principle: 0, links: 0 }); stage.tween(st, { cols: 1 }, 2400, { ease: "out" }); },
      function (stage) { stage.tween(st, { principle: 1 }, 1000); },
      function (stage) { stage.tween(st, { links: 1 }, 900); },
    ],
    draw(stage, t) {
      const ctx = stage.ctx;
      S.header(ctx, "Three questions, in order", "三个问题，按顺序问");
      const cols = [
        { title: S.t("Identification", "识别"), q: S.t("Is the value in the data?", "答案在数据里吗？"), body: S.t("Theorems 3 & 5, Proposition 6: a single-protocol benchmark need not determine an undeployed protocol's value — even with infinite data. Proposition 8: targeted augmentation restores it, value by value.", "定理 3、5，命题 6：单一协议的基准数据未必能确定未实施协议的价值——即使数据无穷多。命题 8：定向补测可以逐个价值地恢复它。"), color: C.alert, sec: S.t("Section 3", "第 3 节") },
        { title: S.t("Calibration", "校准"), q: S.t("How accurately can it be estimated?", "能估多准？"), body: S.t("Theorems 10–11: dense calibration gives uniform value error C‖K̂−K‖^β, root-m at a fixed model. Corollary 12: regret ≤ 2ε — a resolution below which protocols cannot be told apart.", "定理 10–11：密集校准给出一致的价值误差 C‖K̂−K‖^β，固定模型下为根号 m 速率。推论 12：损失 ≤ 2ε——低于这个分辨率的协议分不开。"), color: C.latent, sec: S.t("Section 4", "第 4 节") },
        { title: S.t("Design", "设计"), q: S.t("Which candidates can be told apart?", "哪些候选分得开？"), body: S.t("Proposition 13: exact rank-one gains for cost-constrained, target-aware search; monotone but not submodular. Real data: coarse layout contrasts were clearer than exact learned anchors.", "命题 13：预算约束下针对目标搜索的精确秩一收益；单调但不次模。真实数据：粗布局的差异比精确学得的锚点更清楚。"), color: C.expl, sec: S.t("Sections 5–6", "第 5–6 节") },
      ];
      cols.forEach((c, i) => {
        const a = CPV.clamp(st.cols * 3 - i, 0, 1), x = 100 + i * 480, y = 150;
        D.rect(ctx, x, y, 440, 470, { fill: CPV.rgba(C.panel, 0.9), stroke: CPV.rgba(c.color, 0.6), radius: 14, alpha: a });
        D.badge(ctx, i + 1, x + 36, y + 40, { color: c.color, alpha: a, r: 18 });
        D.text(ctx, c.title, x + 68, y + 48, { size: 28, color: c.color, font: "display", weight: 500, alpha: a });
        D.text(ctx, c.sec, x + 410, y + 46, { size: 14, color: C.dim, align: "right", font: "mono", alpha: a });
        D.text(ctx, c.q, x + 30, y + 100, { size: 21, color: C.ink, weight: 500, alpha: a });
        D.paragraph(ctx, c.body, x + 30, y + 150, 380, { size: 16.5, color: C.muted, alpha: a, lineHeight: 26 });
        if (i < 2) D.arrow(ctx, x + 446, y + 235, x + 474, y + 235, { color: C.dim, width: 2.5, alpha: a, head: 10 });
      });
      D.text(ctx, S.t("the order cannot be reversed: if the answer is not in the data, no optimiser helps; if it is but ε is large, fine optimisation overfits", "顺序不能颠倒：答案不在数据里，优化算法无济于事；在数据里但 ε 很大，精细优化就会过拟合"), 800, 672, { size: 17, color: C.muted, align: "center", alpha: CPV.clamp(st.cols * 3 - 2.5, 0, 1) });
      if (st.principle > 0) {
        D.rect(ctx, 0, 0, 1600, 900, { fill: CPV.rgba(C.bg, 0.9 * st.principle) });
        D.text(ctx, S.t("A benchmark fixes a statistical experiment, not just a sample.", "一个基准数据集固定的是一个统计实验，而不只是一个样本。"), 800, 330, { size: 40, color: C.ink, font: "display", weight: 500, align: "center", alpha: st.principle });
        D.takeaway(ctx, S.t("Identification before optimisation: evaluate undeployed protocols only once the data can determine their value, and optimise only at the granularity the calibration data support.", "先识别，再优化：只有当数据能够确定协议的价值时才评估未实施的协议；也只在校准数据支持的粒度上做优化。"), 800, 470, { alpha: st.principle, size: 26, maxW: 1250 });
        D.text(ctx, S.t("practical pattern: a large routine cohort + a small, intensively observed calibration subset chosen with future protocol decisions in mind", "实践模式：大规模常规队列 + 为未来协议决策而设计的小规模密集校准子集"), 800, 620, { size: 18, color: C.gold, align: "center", alpha: st.principle });
      }
      if (st.links > 0) {
        const a = st.links;
        D.rect(ctx, 300, 680, 1000, 150, { fill: CPV.rgba(C.panel, 0.95), stroke: C.grid, radius: 12, alpha: a });
        D.text(ctx, "Counterfactual Evaluation of Temporal Observation Protocols", 800, 722, { size: 22, color: C.ink, font: "display", weight: 500, align: "center", alpha: a });
        D.text(ctx, S.t("Xizhe Zhang · manuscript, code and experiment package released together", "张锡哲 · 论文、代码与实验软件包一起发布"), 800, 756, { size: 16, color: C.muted, align: "center", alpha: a });
        D.text(ctx, S.t("every number in this tour is read from paper/numbers.tex and results/ — the same files that generate the paper", "本导览中的每一个数字都读取自 paper/numbers.tex 与 results/——生成论文的同一批文件"), 800, 790, { size: 15, color: C.latent, align: "center", font: "mono", alpha: a });
      }
    },
  };
})();
