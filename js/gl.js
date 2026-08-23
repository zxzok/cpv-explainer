/* LineField: a WebGL renderer for ensembles of trajectories.
 *
 * Every path is drawn as a screen-space ribbon (two triangles per segment,
 * mitred in the vertex shader) with additive blending, so hundreds of latent
 * trajectories read as a glowing band whose density is the posterior.  The
 * scenes update the y-values every frame while a posterior collapses, so the
 * vertex buffer is rebuilt from typed arrays rather than re-created.
 *
 * Coordinates: data space (x in [x0,x1], y in [y0,y1]) mapped into a logical
 * rectangle of the stage (same 1600x900 frame the 2D overlay uses). */
(function () {
  const VS = `
    attribute vec2 a_pos; attribute vec2 a_prev; attribute vec2 a_next;
    attribute float a_side; attribute vec4 a_color;
    uniform vec2 u_res;          // viewport size in device px
    uniform vec4 u_view;         // x0, x1, y0, y1 of the data window
    uniform float u_width;       // ribbon width in device px
    varying vec4 v_color; varying float v_x;
    vec2 toPx(vec2 p) {
      return vec2((p.x - u_view.x) / (u_view.y - u_view.x), (p.y - u_view.z) / (u_view.w - u_view.z)) * u_res;
    }
    void main() {
      vec2 p = toPx(a_pos), q = toPx(a_prev), r = toPx(a_next);
      vec2 d1 = p - q, d2 = r - p;
      if (length(d1) < 1e-4) d1 = d2;
      if (length(d2) < 1e-4) d2 = d1;
      vec2 dir = normalize(normalize(d1) + normalize(d2));
      vec2 nrm = vec2(-dir.y, dir.x);
      float mitre = 1.0 / max(0.35, dot(nrm, vec2(-normalize(d1).y, normalize(d1).x)));
      vec2 off = nrm * a_side * u_width * 0.5 * min(mitre, 2.5);
      vec2 px = p + off;
      gl_Position = vec4(px / u_res * 2.0 - 1.0, 0.0, 1.0);
      v_color = a_color; v_x = a_pos.x;
    }`;
  const FS = `
    precision mediump float;
    varying vec4 v_color; varying float v_x;
    uniform float u_reveal; uniform float u_alpha;
    void main() {
      if (v_x > u_reveal) discard;
      gl_FragColor = vec4(v_color.rgb, v_color.a * u_alpha);
    }`;

  function compile(gl, type, src) {
    const s = gl.createShader(type); gl.shaderSource(s, src); gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s));
    return s;
  }

  class LineField {
    constructor(canvas) {
      this.canvas = canvas;
      const gl = canvas.getContext("webgl", { alpha: false, antialias: true, preserveDrawingBuffer: !!window.CPV_SHEET });
      this.gl = gl;
      this.ok = !!gl;
      if (!gl) return;
      const prog = gl.createProgram();
      gl.attachShader(prog, compile(gl, gl.VERTEX_SHADER, VS));
      gl.attachShader(prog, compile(gl, gl.FRAGMENT_SHADER, FS));
      gl.linkProgram(prog);
      if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(prog));
      this.prog = prog;
      this.loc = {
        pos: gl.getAttribLocation(prog, "a_pos"), prev: gl.getAttribLocation(prog, "a_prev"),
        next: gl.getAttribLocation(prog, "a_next"), side: gl.getAttribLocation(prog, "a_side"),
        color: gl.getAttribLocation(prog, "a_color"),
        res: gl.getUniformLocation(prog, "u_res"), view: gl.getUniformLocation(prog, "u_view"),
        width: gl.getUniformLocation(prog, "u_width"), reveal: gl.getUniformLocation(prog, "u_reveal"),
        alpha: gl.getUniformLocation(prog, "u_alpha"),
      };
      this.vbo = gl.createBuffer(); this.ibo = gl.createBuffer();
      this.layers = [];            // [{xs, ys[], colors[], width, reveal, alpha, vertices, indices, count}]
      this.clear = [0.043, 0.067, 0.098];
      this.view = [0, 1, -3, 3];
      this.rect = null;            // logical rect {x,y,w,h} in stage units, null = whole canvas
    }
    setClear(r, g, b) { this.clear = [r, g, b]; }
    setView(x0, x1, y0, y1) { this.view = [x0, x1, y0, y1]; }
    setRect(rect) { this.rect = rect; }

    /* Build (or rebuild) the geometry of one layer.  xs: Float32Array(p); ys: array of Float32Array(p);
     * colors: array of [r,g,b,a] per path. */
    makeLayer(xs, ys, colors, opts = {}) {
      const n = ys.length, p = xs.length, stride = 11;
      const verts = new Float32Array(n * p * 2 * stride);
      const idx = new Uint32Array(n * (p - 1) * 6);
      let vi = 0, ii = 0;
      for (let k = 0; k < n; k++) {
        const y = ys[k], c = colors[k];
        for (let i = 0; i < p; i++) {
          const ip = Math.max(0, i - 1), inx = Math.min(p - 1, i + 1);
          for (let side = -1; side <= 1; side += 2) {
            verts[vi++] = xs[i]; verts[vi++] = y[i];
            verts[vi++] = xs[ip]; verts[vi++] = y[ip];
            verts[vi++] = xs[inx]; verts[vi++] = y[inx];
            verts[vi++] = side;
            verts[vi++] = c[0]; verts[vi++] = c[1]; verts[vi++] = c[2]; verts[vi++] = c[3];
          }
        }
        const base = k * p * 2;
        for (let i = 0; i < p - 1; i++) {
          const a = base + 2 * i;
          idx[ii++] = a; idx[ii++] = a + 1; idx[ii++] = a + 2;
          idx[ii++] = a + 1; idx[ii++] = a + 3; idx[ii++] = a + 2;
        }
      }
      const layer = { xs, ys, colors, n, p, stride, verts, idx, width: opts.width || 1.5,
                      reveal: opts.reveal !== undefined ? opts.reveal : Infinity, alpha: opts.alpha !== undefined ? opts.alpha : 1,
                      visible: true, dirty: true };
      return layer;
    }
    addLayer(layer) { this.layers.push(layer); return layer; }
    clearLayers() { this.layers.length = 0; }

    /* Update y-values in place (used for animated posterior collapse). */
    updateYs(layer, ys) {
      const { p, stride, verts } = layer;
      for (let k = 0; k < layer.n; k++) {
        const y = ys[k];
        for (let i = 0; i < p; i++) {
          const ip = Math.max(0, i - 1), inx = Math.min(p - 1, i + 1);
          for (let s = 0; s < 2; s++) {
            const o = ((k * p + i) * 2 + s) * stride;
            verts[o + 1] = y[i]; verts[o + 3] = y[ip]; verts[o + 5] = y[inx];
          }
        }
      }
      layer.ys = ys; layer.dirty = true;
    }
    updateColors(layer, colors) {
      const { p, stride, verts } = layer;
      for (let k = 0; k < layer.n; k++) {
        const c = colors[k];
        for (let i = 0; i < p * 2; i++) {
          const o = (k * p * 2 + i) * stride;
          verts[o + 7] = c[0]; verts[o + 8] = c[1]; verts[o + 9] = c[2]; verts[o + 10] = c[3];
        }
      }
      layer.colors = colors; layer.dirty = true;
    }

    /* Draw all layers.  `vp` = {x, y, w, h} viewport in device px (y from bottom). */
    draw(vp) {
      const gl = this.gl; if (!gl) return;
      const W = this.canvas.width, H = this.canvas.height;
      gl.viewport(0, 0, W, H);
      gl.disable(gl.SCISSOR_TEST);
      gl.clearColor(this.clear[0], this.clear[1], this.clear[2], 1);
      gl.clear(gl.COLOR_BUFFER_BIT);
      if (vp) { gl.viewport(vp.x, vp.y, vp.w, vp.h); gl.enable(gl.SCISSOR_TEST); gl.scissor(vp.x, vp.y, vp.w, vp.h); }
      const res = vp ? [vp.w, vp.h] : [W, H];
      gl.enable(gl.BLEND); gl.blendFunc(gl.SRC_ALPHA, gl.ONE);
      gl.useProgram(this.prog);
      gl.uniform2f(this.loc.res, res[0], res[1]);
      gl.uniform4f(this.loc.view, this.view[0], this.view[1], this.view[2], this.view[3]);
      const ext = gl.getExtension("OES_element_index_uint");
      for (const L of this.layers) {
        if (!L.visible || L.alpha <= 0) continue;
        gl.bindBuffer(gl.ARRAY_BUFFER, this.vbo);
        gl.bufferData(gl.ARRAY_BUFFER, L.verts, gl.DYNAMIC_DRAW);
        gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.ibo);
        gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, ext ? L.idx : Uint16Array.from(L.idx), gl.DYNAMIC_DRAW);
        const s = L.stride * 4;
        gl.enableVertexAttribArray(this.loc.pos); gl.vertexAttribPointer(this.loc.pos, 2, gl.FLOAT, false, s, 0);
        gl.enableVertexAttribArray(this.loc.prev); gl.vertexAttribPointer(this.loc.prev, 2, gl.FLOAT, false, s, 8);
        gl.enableVertexAttribArray(this.loc.next); gl.vertexAttribPointer(this.loc.next, 2, gl.FLOAT, false, s, 16);
        gl.enableVertexAttribArray(this.loc.side); gl.vertexAttribPointer(this.loc.side, 1, gl.FLOAT, false, s, 24);
        gl.enableVertexAttribArray(this.loc.color); gl.vertexAttribPointer(this.loc.color, 4, gl.FLOAT, false, s, 28);
        gl.uniform1f(this.loc.width, L.width * (window.devicePixelRatio || 1));
        gl.uniform1f(this.loc.reveal, L.reveal);
        gl.uniform1f(this.loc.alpha, L.alpha);
        gl.drawElements(gl.TRIANGLES, L.idx.length, ext ? gl.UNSIGNED_INT : gl.UNSIGNED_SHORT, 0);
      }
      gl.disable(gl.SCISSOR_TEST);
    }
  }

  window.CPV = window.CPV || {};
  window.CPV.LineField = LineField;
})();
