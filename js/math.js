/* Small dense linear algebra and the protocol-value calculus of the paper,
 * ported from the protocol_ceiling package so the interactive scenes compute
 * real values (Proposition 2, Theorem 3, Proposition 13) in the browser.
 * Matrices are arrays of Float64Array rows.  Everything here is O(p^3) with
 * p <= 96, which is far below a frame budget. */
(function () {
  const M = {};

  // ------------------------------------------------------------ randomness
  M.rng = function (seed) {
    let a = seed >>> 0;
    const rand = function () {
      a |= 0; a = (a + 0x6D2B79F5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
    rand.gauss = function () {
      let u = 0, v = 0;
      while (u === 0) u = rand();
      v = rand();
      return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
    };
    return rand;
  };

  // ------------------------------------------------------------ matrices
  M.zeros = (n, m) => Array.from({ length: n }, () => new Float64Array(m));
  M.eye = n => { const I = M.zeros(n, n); for (let i = 0; i < n; i++) I[i][i] = 1; return I; };
  M.copy = A => A.map(r => Float64Array.from(r));
  M.transpose = A => {
    const n = A.length, m = A[0].length, T = M.zeros(m, n);
    for (let i = 0; i < n; i++) for (let j = 0; j < m; j++) T[j][i] = A[i][j];
    return T;
  };
  M.matmul = (A, B) => {
    const n = A.length, k = B.length, m = B[0].length, C = M.zeros(n, m);
    for (let i = 0; i < n; i++) {
      const Ai = A[i], Ci = C[i];
      for (let l = 0; l < k; l++) {
        const a = Ai[l]; if (a === 0) continue;
        const Bl = B[l];
        for (let j = 0; j < m; j++) Ci[j] += a * Bl[j];
      }
    }
    return C;
  };
  M.matvec = (A, x) => Float64Array.from(A, row => { let s = 0; for (let j = 0; j < row.length; j++) s += row[j] * x[j]; return s; });
  M.dot = (x, y) => { let s = 0; for (let i = 0; i < x.length; i++) s += x[i] * y[i]; return s; };
  M.quad = (A, x) => M.dot(x, M.matvec(A, x));               // x' A x
  M.add = (A, B, b = 1) => A.map((r, i) => r.map((v, j) => v + b * B[i][j]));
  M.outer = (u, v) => Array.from(u, ui => Float64Array.from(v, vj => ui * vj));   // Array.from: a typed array cannot hold rows
  M.addDiag = (A, d) => { const C = M.copy(A); for (let i = 0; i < C.length; i++) C[i][i] += d[i] !== undefined ? d[i] : d; return C; };

  /* Solve A X = B for square A (Gaussian elimination, partial pivoting).  B may be a vector. */
  M.solve = (A, B) => {
    const n = A.length, vec = !Array.isArray(B) && !(B[0] instanceof Float64Array);
    const Bm = vec ? [Float64Array.from(B)] : B.map(r => Float64Array.from(r));
    const m = vec ? 1 : Bm[0].length;
    const a = A.map(r => Float64Array.from(r));
    const b = vec ? M.transpose(Bm) : Bm;
    for (let c = 0; c < n; c++) {
      let piv = c;
      for (let r = c + 1; r < n; r++) if (Math.abs(a[r][c]) > Math.abs(a[piv][c])) piv = r;
      if (piv !== c) { [a[c], a[piv]] = [a[piv], a[c]]; [b[c], b[piv]] = [b[piv], b[c]]; }
      const d = a[c][c];
      if (Math.abs(d) < 1e-14) throw new Error("singular");
      for (let r = c + 1; r < n; r++) {
        const f = a[r][c] / d; if (f === 0) continue;
        for (let k = c; k < n; k++) a[r][k] -= f * a[c][k];
        for (let k = 0; k < m; k++) b[r][k] -= f * b[c][k];
      }
    }
    const X = M.zeros(n, m);
    for (let r = n - 1; r >= 0; r--) {
      for (let k = 0; k < m; k++) {
        let s = b[r][k];
        for (let c = r + 1; c < n; c++) s -= a[r][c] * X[c][k];
        X[r][k] = s / a[r][r];
      }
    }
    return vec ? Float64Array.from(X, r => r[0]) : X;
  };

  /* Lower Cholesky factor, or null if A is not positive definite. */
  M.cholesky = (A, jitter = 0) => {
    const n = A.length, L = M.zeros(n, n);
    for (let i = 0; i < n; i++) {
      for (let j = 0; j <= i; j++) {
        let s = A[i][j];
        for (let k = 0; k < j; k++) s -= L[i][k] * L[j][k];
        if (i === j) {
          s += jitter;
          if (s <= 0) return null;
          L[i][i] = Math.sqrt(s);
        } else L[i][j] = s / L[j][j];
      }
    }
    return L;
  };
  M.isPD = A => M.cholesky(A) !== null;

  // ------------------------------------------------------------ kernels
  M.grid = (p, T) => Float64Array.from({ length: p }, (_, i) => T * (i + 0.5) / p);
  M.ouKernel = (p, tau, T = 1) => {
    const t = M.grid(p, T), K = M.zeros(p, p);
    for (let i = 0; i < p; i++) for (let j = 0; j < p; j++) K[i][j] = Math.exp(-Math.abs(t[i] - t[j]) / tau);
    return K;
  };
  M.seKernel = (p, tau, T = 1) => {
    const t = M.grid(p, T), K = M.zeros(p, p);
    for (let i = 0; i < p; i++) for (let j = 0; j < p; j++) { const d = (t[i] - t[j]) / tau; K[i][j] = Math.exp(-0.5 * d * d); }
    return K;
  };
  /* Correlation length growing along the horizon (fast early, slow late). */
  M.twoScaleKernel = (p, tau, T = 1) => {
    const t = M.grid(p, T), K = M.zeros(p, p);
    for (let i = 0; i < p; i++) for (let j = 0; j < p; j++) {
      const si = tau * (0.35 + 1.3 * i / (p - 1)), sj = tau * (0.35 + 1.3 * j / (p - 1));
      K[i][j] = Math.exp(-Math.abs(t[i] - t[j]) / Math.sqrt(si * sj));
    }
    return K;
  };
  /* Trait-state correlation: a persistent component of share alpha plus a state process. */
  M.traitState = (K, alpha) => K.map(r => r.map(v => alpha + (1 - alpha) * v));
  M.toeplitz = rho => { const p = rho.length, K = M.zeros(p, p); for (let i = 0; i < p; i++) for (let j = 0; j < p; j++) K[i][j] = rho[Math.abs(i - j)]; return K; };
  M.rowsFromIndices = (idx, p) => idx.map(j => { const r = new Float64Array(p); r[j] = 1; return r; });
  M.uniformWeights = p => new Float64Array(p).fill(1 / p);

  // ------------------------------------------------------------ C_g: Gaussian covariance transform
  /* Plackett: G_c(r) = int_0^r exp(-c^2/(1+s)) / (2 pi sqrt(1-s^2)) ds, with s = sin u. */
  M.occIntegral = (r, c) => {
    const u1 = Math.asin(Math.max(-0.999999, Math.min(0.999999, r)));
    const n = 48, h = u1 / n;
    let s = 0;
    for (let i = 0; i <= n; i++) {
      const u = i * h, w = (i === 0 || i === n) ? 1 : (i % 2 ? 4 : 2);
      s += w * Math.exp(-c * c / (1 + Math.sin(u)));
    }
    return s * h / 3 / (2 * Math.PI);
  };
  const occTables = {};
  M.occTable = c => {
    const key = c.toFixed(6);
    if (!occTables[key]) {
      const n = 801, rs = new Float64Array(n), vs = new Float64Array(n);
      for (let i = 0; i < n; i++) { rs[i] = -1 + 2 * i / (n - 1); vs[i] = M.occIntegral(rs[i], c); }
      occTables[key] = { rs, vs };
    }
    return occTables[key];
  };
  M.C_occ = (r, c = 0) => {
    const { rs, vs } = M.occTable(c), n = rs.length;
    const x = Math.max(-1, Math.min(1, r)), f = (x + 1) / 2 * (n - 1);
    const i = Math.min(n - 2, Math.floor(f)), t = f - i;
    return vs[i] * (1 - t) + vs[i + 1] * t;
  };
  M.C_of = target => target.kind === "mean" ? (r => r) : (r => M.C_occ(r, target.c || 0));

  // ------------------------------------------------------------ protocol value (Proposition 2)
  M.Q_of = (K, A, R) => {
    const AK = M.matmul(A, K);                          // d x p
    const S = M.add(M.matmul(AK, M.transpose(A)), R);   // d x d
    const X = M.solve(S, AK);                           // d x p
    return M.matmul(M.transpose(AK), X);                // p x p
  };
  M.diagR = (d, r) => { const R = M.zeros(d, d); for (let i = 0; i < d; i++) R[i][i] = Array.isArray(r) ? r[i] : r; return R; };
  M.quadC = (Mtx, w, C) => {
    let s = 0; const p = w.length;
    for (let j = 0; j < p; j++) { if (w[j] === 0) continue; const row = Mtx[j]; for (let k = 0; k < p; k++) if (w[k] !== 0) s += w[j] * w[k] * C(row[k]); }
    return s;
  };
  /* I_g(S;K) = F_g / V_g.  `target` is {kind:"mean"} or {kind:"occ", c}. */
  M.value = (K, A, R, w, target = { kind: "mean" }) => {
    if (!A || A.length === 0) return 0;
    const C = M.C_of(target);
    const Q = M.Q_of(K, A, R);
    return M.quadC(Q, w, C) / M.quadC(K, w, C);
  };
  M.valueMean = (K, A, R, w) => M.value(K, A, R, w, { kind: "mean" });

  /* Posterior residual covariance P_S = K - Q_S. */
  M.residual = (K, A, R) => (!A || A.length === 0) ? M.copy(K) : M.add(K, M.Q_of(K, A, R), -1);

  // ------------------------------------------------------------ greedy design (Proposition 13)
  /* candidates: [{ell: Float64Array, r, cost, label}]; returns the chosen steps with exact gains.
   * Each step records P and Q after the addition so callers can animate the collapse. */
  M.greedy = (K, w, candidates, budget, target = { kind: "mean" }) => {
    const C = M.C_of(target), p = K.length;
    let P = M.copy(K), Q = M.zeros(p, p), spent = 0;
    const chosen = new Set(), steps = [];
    const base0 = M.quadC(Q, w, C);
    let base = base0;
    for (;;) {
      let best = null;
      candidates.forEach((a, idx) => {
        if (chosen.has(idx) || spent + a.cost > budget + 1e-12) return;
        const v = M.matvec(P, a.ell), s = M.dot(a.ell, v) + a.r;
        if (s <= 1e-12) return;
        let gain;
        if (target.kind === "mean") { const t = M.dot(w, v); gain = t * t / s; }
        else gain = M.quadC(M.add(Q, M.outer(v, v), 1 / s), w, C) - base;
        const score = gain / a.cost;
        if (!best || score > best.score) best = { idx, gain, score, v, s };
      });
      if (!best) break;
      const { idx, v, s } = best;
      P = M.add(P, M.outer(v, v), -1 / s);
      Q = M.add(Q, M.outer(v, v), 1 / s);
      base = M.quadC(Q, w, C);
      chosen.add(idx); spent += candidates[idx].cost;
      steps.push({ idx, action: candidates[idx], gain: best.gain, P: M.copy(P), Q: M.copy(Q), value: base / M.quadC(K, w, C) });
    }
    return steps;
  };

  // ------------------------------------------------------------ sample paths
  M.samplePaths = (L, n, rand) => {
    const p = L.length, out = [];
    for (let k = 0; k < n; k++) {
      const eta = Float64Array.from({ length: p }, () => rand.gauss());
      const z = new Float64Array(p);
      for (let i = 0; i < p; i++) { let s = 0; const Li = L[i]; for (let j = 0; j <= i; j++) s += Li[j] * eta[j]; z[i] = s; }
      out.push(z);
    }
    return out;
  };
  /* Matheron update: condition prior sample paths on noisy point observations of `truth`.
   * obs: [{idx, r}]. Returns new paths; each prior path z becomes z + W (y - (z_S + e)). */
  M.conditionPaths = (K, obs, truth, priors, rand) => {
    if (obs.length === 0) return priors.map(z => Float64Array.from(z));
    const p = K.length, d = obs.length;
    const A = M.rowsFromIndices(obs.map(o => o.idx), p), R = M.diagR(d, obs.map(o => o.r));
    const AK = M.matmul(A, K), S = M.add(M.matmul(AK, M.transpose(A)), R);
    const W = M.transpose(M.solve(S, AK));             // p x d  = K A' S^-1
    const y = obs.map(o => truth[o.idx] + Math.sqrt(o.r) * rand.gauss());
    return priors.map(z => {
      const resid = obs.map((o, i) => y[i] - (z[o.idx] + Math.sqrt(o.r) * rand.gauss()));
      const out = Float64Array.from(z);
      for (let i = 0; i < p; i++) { let s = 0; for (let j = 0; j < d; j++) s += W[i][j] * resid[j]; out[i] += s; }
      return out;
    });
  };

  // ------------------------------------------------------------ Theorem 3 instance (Figure 1)
  M.fourPoint = (eps, nuB2 = 0) => {
    const rho0 = [1, Math.exp(-1), Math.exp(-2), Math.exp(-3)], delta = [0, 1, -2, 1];
    const rp = rho0.map((r, i) => r + eps * delta[i]), rm = rho0.map((r, i) => r - eps * delta[i]);
    const Kp = M.toeplitz(rp), Km = M.toeplitz(rm), h = M.uniformWeights(4);
    const A = M.rowsFromIndices([0], 4), B = M.rowsFromIndices([1, 2], 4), RB = M.diagR(2, nuB2), RA = M.diagR(1, 0);
    const obs = K => ({ varY: M.quad(K, A[0]), covYT: M.dot(M.matvec(K, A[0]), h), varT: M.quad(K, h) });
    return {
      rho0, rhoPlus: rp, rhoMinus: rm, Kp, Km, h, A, B,
      obsPlus: obs(Kp), obsMinus: obs(Km),
      valuePlus: M.isPD(Kp) ? M.valueMean(Kp, B, RB, h) : NaN,
      valueMinus: M.isPD(Km) ? M.valueMean(Km, B, RB, h) : NaN,
      pd: M.isPD(Kp) && M.isPD(Km),
    };
  };

  window.CPV = window.CPV || {};
  window.CPV.M = M;
})();
