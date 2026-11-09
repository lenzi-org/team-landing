/* ==========================================================================
   Algo scene — 03 · Computer science · Arshia Dehghan  (accent #ffb547)

   0.00–0.25  A seeded Poisson-disc k-NN graph fills the viewport; candidate
              s→t routes are probed one after another ("thousands of paths").
   0.25–0.50  A real, precomputed Dijkstra run is revealed as a wavefront: the
              shortest-path tree grows edge by edge, frontier nodes ring, a mono
              HUD reports live counts. Then the optimal path draws back t → s.
   0.50–0.76  The target blooms into a huge rolling "61".
   0.76–1.00  Graph nodes collapse into a halftone of Arshia's face, which
              crossfades into the shared tri-tone portrait; the 61 parks as a
              "#61" chip on the portrait's corner.

   Everything is a pure function of progress; time only drives twinkle and a
   light pulse along the path. All topology is precomputed on resize.
   ========================================================================== */
(function () {
  "use strict";
  const U = window.SceneUtils;
  const TAU = Math.PI * 2;
  const BG = "#060708";
  const NB = 6; // particle luminance buckets (one fill per bucket)
  const DIG = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"];

  window.SceneRegistry = window.SceneRegistry || {};

  // Warm the portrait once the page has loaded, so it is decoded long before
  // the convergence beat even if the reader jumps straight down the page.
  window.addEventListener("load", () => {
    const el = document.querySelector('[data-scene="algo"]');
    const src = el && el.dataset.portrait;
    if (!src || !U) return;
    const go = () => U.loadImage(src).catch(() => {});
    (window.requestIdleCallback || ((fn) => setTimeout(fn, 200)))(go, { timeout: 2000 });
  }, { once: true });

  /* ---------- Poisson-disc sampling (Bridson), seeded ---------- */
  function poisson(x0, y0, w, h, r, rnd) {
    const cell = r / Math.SQRT2;
    const gw = Math.ceil(w / cell), gh = Math.ceil(h / cell);
    const grid = new Int32Array(gw * gh).fill(-1);
    const xs = [], ys = [], active = [];
    const r2 = r * r;
    const add = (x, y) => {
      const i = xs.length;
      xs.push(x); ys.push(y); active.push(i);
      grid[(((y - y0) / cell) | 0) * gw + (((x - x0) / cell) | 0)] = i;
    };
    add(x0 + w * (0.3 + 0.4 * rnd()), y0 + h * (0.3 + 0.4 * rnd()));
    while (active.length) {
      const ai = (rnd() * active.length) | 0;
      const i = active[ai];
      let placed = false;
      for (let t = 0; t < 28; t++) {
        const ang = rnd() * TAU, rad = r * (1 + rnd());
        const x = xs[i] + Math.cos(ang) * rad, y = ys[i] + Math.sin(ang) * rad;
        if (x < x0 || y < y0 || x >= x0 + w || y >= y0 + h) continue;
        const gx = ((x - x0) / cell) | 0, gy = ((y - y0) / cell) | 0;
        let ok = true;
        for (let yy = Math.max(0, gy - 2); yy <= Math.min(gh - 1, gy + 2) && ok; yy++) {
          for (let xx = Math.max(0, gx - 2); xx <= Math.min(gw - 1, gx + 2); xx++) {
            const j = grid[yy * gw + xx];
            if (j >= 0) {
              const dx = xs[j] - x, dy = ys[j] - y;
              if (dx * dx + dy * dy < r2) { ok = false; break; }
            }
          }
        }
        if (ok) { add(x, y); placed = true; break; }
      }
      if (!placed) { active[ai] = active[active.length - 1]; active.pop(); }
    }
    return { xs, ys, n: xs.length };
  }

  /** Loop-erase a walk so it becomes a simple path. */
  function loopErase(seq) {
    const pos = new Map(), out = [];
    for (const v of seq) {
      if (pos.has(v)) {
        const k = pos.get(v);
        for (let i = k + 1; i < out.length; i++) pos.delete(out[i]);
        out.length = k + 1;
      } else { pos.set(v, out.length); out.push(v); }
    }
    return out;
  }

  window.SceneRegistry.algo = function createAlgo(canvas, opts) {
    const ctx = canvas.getContext("2d");
    const reduced = !!opts.reducedMotion;
    const A = opts.accentRgb || [255, 181, 71];
    const WHITE = [255, 255, 255];
    const css = (c) => `rgb(${Math.round(c[0])},${Math.round(c[1])},${Math.round(c[2])})`;
    const C_AMBER = css(A);
    const C_AMBER_HI = css(U.mixRgb(A, WHITE, 0.45));
    const C_AMBER_PALE = css(U.mixRgb(A, WHITE, 0.8));
    const C_NODE = "#c3c8d0";
    const C_INK = "#eef0f3", C_MUTED = "#9aa1ad", C_DIM = "#5f6673";
    const sprAmber = U.glowSprite(A, 64, 0.14);
    const sprBloom = U.glowSprite(A, 256, 0.03);
    const sprWhite = U.glowSprite([232, 238, 246], 64, 0.2);
    const sprPale = U.glowSprite(U.mixRgb(A, WHITE, 0.6), 64, 0.22);
    const hasLS = "letterSpacing" in ctx;

    let alive = true;
    let W = 0, H = 0, DPR = 1, mobile = false;

    /* graph */
    let N = 0, E = 0, R = 40, maxEdge = 1, gx0 = 0, gy0 = 0, gcols = 1, grows = 1;
    let X = null, Y = null, eA = null, eB = null, eLen = null;
    let adjOff = null, adjTo = null, adjE = null, cellHead = null, cellNext = null;
    let edgeShort = null, edgeLong = null, nodeGroups = null;
    /* dijkstra */
    let S = 0, T = 0, count = 0, V = 1, DT = 1;
    let dist = null, pred = null, order = null, orderDist = null, disc = null;
    let hN = null, hK = null, done = null;
    let path = null, pathCum = null, pathLen = 1;
    let alts = [];
    /* layout */
    let ax = 0, ay = 0, gutter = 24, capBottom = 80;
    const cam = { s: 1, ox: 0, oy: 0 };
    let headX = 0, headY = 0;
    /* type */
    let F = 300, asc = 210, colW0 = 150, colW1 = 100, gap = 0, numW = 250, maxDW = 150;
    let fontNum = "", fontMono = "", fontChip = "", fontTiny = "";
    let numGrad = null, glowA = null, glowB = null;
    let hudFs = 12, monoCW = 7.2, chipFs = 15, chipHashW = 9;
    let labelW = 0, labelRankW = 0;
    let scrimCap = null, scrimSide = null;
    /* portrait & particles */
    let img = null, portrait = null, box = null;
    let M = 0, ptx = null, pty = null, prad = null, psrc = null, pdel = null, psw = null, bIdx = null;
    let pjx = null, pjy = null;
    const bOff = new Int32Array(NB + 1);
    const bCol = new Array(NB).fill(C_AMBER);
    const bAlpha = new Float32Array(NB);
    const bRad = new Float32Array(NB);
    const MAXDEL_DIST = 0.19, MAXDEL = 0.07 + MAXDEL_DIST;
    let landed = null;
    const stats = { nodes: 0, edges: 0, visited: 0, particles: 0, buildMs: 0, faceMs: 0 };
    if (/[?&]debug\b/.test(location.search)) canvas.__algoStats = stats;

    /* ---------------- camera ---------------- */
    function camAt(p, out) {
      const a = U.ease.inOutSine(U.range(p, 0, 0.5));
      const b = U.ease.inOutSine(U.range(p, 0.45, 0.8));
      const s = 1.04 + 0.05 * a + 0.045 * b;
      const fx = W * (0.47 + 0.035 * a) + (ax - W * 0.5) * 0.12 * b;
      const fy = H * (0.51 - 0.015 * a) + (ay - H * 0.5) * 0.12 * b;
      out.s = s;
      out.ox = W * 0.5 - fx * s;
      out.oy = H * 0.5 - fy * s;
    }

    /* ---------------- graph ---------------- */
    function cellOf(x, y) {
      const cx = U.clamp(((x - gx0) / R) | 0, 0, gcols - 1);
      const cy = U.clamp(((y - gy0) / R) | 0, 0, grows - 1);
      return cy * gcols + cx;
    }

    function buildGraph() {
      const m = 0.08;
      gx0 = -W * m; gy0 = -H * m;
      const gw = W * (1 + 2 * m), gh = H * (1 + 2 * m), area = gw * gh;
      const target = mobile ? U.clamp(area / 1150, 300, 370) : U.clamp(area / 1850, 720, 880);
      let r = Math.sqrt(area / target / 1.6), pts = null;
      for (let it = 0; it < 5; it++) {
        pts = poisson(gx0, gy0, gw, gh, r, U.rng(61));
        const ratio = pts.n / target;
        if (Math.abs(ratio - 1) < 0.05) break;
        r *= Math.sqrt(ratio);
      }
      R = r; N = pts.n;
      X = Float32Array.from(pts.xs); Y = Float32Array.from(pts.ys);

      // spatial hash, cell = R
      gcols = Math.ceil(gw / R) + 1; grows = Math.ceil(gh / R) + 1;
      cellHead = new Int32Array(gcols * grows).fill(-1);
      cellNext = new Int32Array(N);
      for (let i = 0; i < N; i++) { const c = cellOf(X[i], Y[i]); cellNext[i] = cellHead[c]; cellHead[c] = i; }

      // k nearest neighbours (k = 4), symmetric union
      const K = 4, bd = new Float64Array(K), bi = new Int32Array(K);
      const keys = new Set(), ea = [], eb = [];
      for (let i = 0; i < N; i++) {
        bd.fill(Infinity); bi.fill(-1);
        const cx = ((X[i] - gx0) / R) | 0, cy = ((Y[i] - gy0) / R) | 0;
        for (let yy = cy - 2; yy <= cy + 2; yy++) {
          if (yy < 0 || yy >= grows) continue;
          for (let xx = cx - 2; xx <= cx + 2; xx++) {
            if (xx < 0 || xx >= gcols) continue;
            for (let j = cellHead[yy * gcols + xx]; j >= 0; j = cellNext[j]) {
              if (j === i) continue;
              const dx = X[j] - X[i], dy = Y[j] - Y[i], d = dx * dx + dy * dy;
              if (d >= bd[K - 1]) continue;
              let q = K - 1;
              while (q > 0 && bd[q - 1] > d) { bd[q] = bd[q - 1]; bi[q] = bi[q - 1]; q--; }
              bd[q] = d; bi[q] = j;
            }
          }
        }
        for (let q = 0; q < K; q++) {
          const j = bi[q];
          if (j < 0) continue;
          const a = Math.min(i, j), b = Math.max(i, j), key = a * N + b;
          if (!keys.has(key)) { keys.add(key); ea.push(a); eb.push(b); }
        }
      }
      E = ea.length;
      eA = Int32Array.from(ea); eB = Int32Array.from(eb); eLen = new Float32Array(E);

      // CSR adjacency
      const off = new Int32Array(N + 1);
      maxEdge = 0;
      for (let e = 0; e < E; e++) {
        eLen[e] = Math.hypot(X[eB[e]] - X[eA[e]], Y[eB[e]] - Y[eA[e]]);
        if (eLen[e] > maxEdge) maxEdge = eLen[e];
        off[eA[e] + 1]++; off[eB[e] + 1]++;
      }
      for (let i = 0; i < N; i++) off[i + 1] += off[i];
      adjOff = off; adjTo = new Int32Array(2 * E); adjE = new Int32Array(2 * E);
      const fill = off.slice(0, N);
      for (let e = 0; e < E; e++) {
        const a = eA[e], b = eB[e];
        adjTo[fill[a]] = b; adjE[fill[a]++] = e;
        adjTo[fill[b]] = a; adjE[fill[b]++] = e;
      }

      // Static geometry, world space: two edge buckets, three twinkle groups
      edgeShort = new Path2D(); edgeLong = new Path2D();
      for (let e = 0; e < E; e++) {
        const pth = eLen[e] < R * 1.45 ? edgeShort : edgeLong;
        pth.moveTo(X[eA[e]], Y[eA[e]]); pth.lineTo(X[eB[e]], Y[eB[e]]);
      }
      const rnd = U.rng(611);
      nodeGroups = [new Path2D(), new Path2D(), new Path2D()];
      for (let i = 0; i < N; i++) {
        const g = (rnd() * 3) | 0, rr = rnd() < 0.1 ? 1.55 : 1.05;
        nodeGroups[g].moveTo(X[i] + rr, Y[i]);
        nodeGroups[g].arc(X[i], Y[i], rr, 0, TAU);
      }

      dist = new Float64Array(N); pred = new Int32Array(N);
      order = new Int32Array(N); orderDist = new Float64Array(N); disc = new Int32Array(N);
      hN = new Int32Array(2 * E + N + 2); hK = new Float64Array(2 * E + N + 2); done = new Uint8Array(N);
    }

    /** Dijkstra with a binary heap (lazy deletion). Fills dist/pred and,
     *  optionally, the pop order and each node's discovery step (the pop
     *  index of the first settled neighbour, i.e. when it entered the queue). */
    function dijkstra(src, w, d, pr, ord, dc) {
      d.fill(Infinity); pr.fill(-1); done.fill(0);
      if (dc) dc.fill(-1);
      let hs = 0, cnt = 0;
      const push = (n, k) => {
        let i = hs++;
        while (i > 0) {
          const pa = (i - 1) >> 1;
          if (hK[pa] <= k) break;
          hK[i] = hK[pa]; hN[i] = hN[pa]; i = pa;
        }
        hK[i] = k; hN[i] = n;
      };
      d[src] = 0; push(src, 0);
      while (hs > 0) {
        const u = hN[0], du = hK[0];
        hs--;
        if (hs > 0) {
          const ln = hN[hs], lk = hK[hs];
          let i = 0;
          for (;;) {
            let c = 2 * i + 1;
            if (c >= hs) break;
            if (c + 1 < hs && hK[c + 1] < hK[c]) c++;
            if (hK[c] >= lk) break;
            hK[i] = hK[c]; hN[i] = hN[c]; i = c;
          }
          hK[i] = lk; hN[i] = ln;
        }
        if (done[u] || du > d[u]) continue;
        done[u] = 1;
        if (ord) ord[cnt] = u;
        for (let q = adjOff[u]; q < adjOff[u + 1]; q++) {
          const v = adjTo[q];
          if (done[v]) continue;
          if (dc && dc[v] < 0) dc[v] = cnt;
          const nd = du + w[adjE[q]];
          if (nd < d[v]) { d[v] = nd; pr[v] = u; push(v, nd); }
        }
        cnt++;
      }
      return cnt;
    }

    function nearest(wx, wy, reach) {
      let best = -1, bd = Infinity;
      for (let i = 0; i < N; i++) {
        if (reach && !(reach[i] < Infinity)) continue;
        const dx = X[i] - wx, dy = Y[i] - wy, d = dx * dx + dy * dy;
        if (d < bd) { bd = d; best = i; }
      }
      return best;
    }

    function nearestGrid(wx, wy) {
      const cx = U.clamp(((wx - gx0) / R) | 0, 0, gcols - 1), cy = U.clamp(((wy - gy0) / R) | 0, 0, grows - 1);
      let best = -1, bd = Infinity;
      for (let rad = 2; rad <= 6 && best < 0; rad += 2) {
        for (let yy = Math.max(0, cy - rad); yy <= Math.min(grows - 1, cy + rad); yy++) {
          for (let xx = Math.max(0, cx - rad); xx <= Math.min(gcols - 1, cx + rad); xx++) {
            for (let j = cellHead[yy * gcols + xx]; j >= 0; j = cellNext[j]) {
              const dx = X[j] - wx, dy = Y[j] - wy, d = dx * dx + dy * dy;
              if (d < bd) { bd = d; best = j; }
            }
          }
        }
      }
      return best >= 0 ? best : nearest(wx, wy, null);
    }

    function cumulative(nodes) {
      const c = new Float32Array(nodes.length);
      for (let i = 1; i < nodes.length; i++) {
        c[i] = c[i - 1] + Math.hypot(X[nodes[i]] - X[nodes[i - 1]], Y[nodes[i]] - Y[nodes[i - 1]]);
      }
      return c;
    }

    function buildRoute() {
      if (mobile) { ax = W * 0.5; ay = H * 0.3; }
      else { ax = W * 0.585; ay = H * 0.42; }
      const sx = mobile ? W * 0.1 : W * 0.1, sy = mobile ? H * 0.63 : H * 0.33;
      camAt(0.3, cam);
      S = nearest((sx - cam.ox) / cam.s, (sy - cam.oy) / cam.s, null);
      count = dijkstra(S, eLen, dist, pred, order, disc);
      camAt(0.5, cam);
      T = nearest((ax - cam.ox) / cam.s, (ay - cam.oy) / cam.s, dist);
      V = 1;
      for (let j = 0; j < count; j++) { orderDist[j] = dist[order[j]]; if (order[j] === T) V = j + 1; }
      DT = Math.max(1, dist[T]);
      const pn = [];
      for (let v = T; v >= 0; v = pred[v]) pn.push(v);
      path = Int32Array.from(pn);            // t → s
      pathCum = cumulative(path);
      pathLen = pathCum[path.length - 1] || 1;
      buildAlts();
    }

    /** Alternative s→t routes for beat 1: Dijkstra on jittered weights via
     *  scattered waypoints (two runs per pass give every waypoint for free). */
    function buildAlts() {
      alts = [];
      const dS = new Float64Array(N), pS = new Int32Array(N), dT = new Float64Array(N), pT = new Int32Array(N);
      const wp = new Float32Array(E);
      const sx = X[S], sy = Y[S], tx = X[T], ty = Y[T];
      const L = Math.max(1, Math.hypot(tx - sx, ty - sy)), nx = -(ty - sy) / L, ny = (tx - sx) / L;
      const rnd = U.rng(1061);
      const per = mobile ? 3 : 5;
      for (let pass = 0; pass < 2; pass++) {
        for (let e = 0; e < E; e++) wp[e] = eLen[e] * (0.7 + 1.5 * rnd() * rnd());
        dijkstra(S, wp, dS, pS, null, null);
        dijkstra(T, wp, dT, pT, null, null);
        for (let q = 0; q < per; q++) {
          const side = (q + pass) % 2 ? 1 : -1;
          const lam = 0.3 + 0.4 * rnd(), off = side * L * (0.08 + 0.3 * rnd());
          let vx = sx + (tx - sx) * lam + nx * off, vy = sy + (ty - sy) * lam + ny * off;
          vx = U.clamp(vx, W * 0.05, W * (mobile ? 0.95 : 0.8));
          vy = U.clamp(vy, H * 0.15, H * (mobile ? 0.68 : 0.64));
          const via = nearest(vx, vy, dS);
          if (via < 0 || !(dT[via] < Infinity)) continue;
          const seq = [];
          for (let v = via; v >= 0; v = pS[v]) seq.push(v);
          seq.reverse();
          for (let v = pT[via]; v >= 0; v = pT[v]) seq.push(v);
          const nodes = Int32Array.from(loopErase(seq));
          if (nodes.length < 3) continue;
          const cum = cumulative(nodes);
          alts.push({ nodes, cum, len: cum[nodes.length - 1] });
        }
      }
    }

    /* ---------------- type ---------------- */
    function buildText() {
      if (!W) return;
      F = Math.round(mobile ? Math.min(W * 0.54, H * 0.27) : Math.min(H * 0.5, W * 0.31));
      fontNum = `700 ${F}px ${U.fonts.display}`;
      hudFs = mobile ? 10.5 : 11.5;
      fontMono = `500 ${hudFs}px ${U.fonts.mono}`;
      fontTiny = `500 10px ${U.fonts.mono}`;
      chipFs = mobile ? 14 : 15;
      fontChip = `600 ${chipFs}px ${U.fonts.display}`;
      ctx.save();
      ctx.setTransform(1, 0, 0, 1, 0, 0);
      ctx.font = fontNum;
      maxDW = 0;
      const dw = new Float32Array(10);
      for (let d = 0; d < 10; d++) { dw[d] = ctx.measureText(DIG[d]).width; if (dw[d] > maxDW) maxDW = dw[d]; }
      const m6 = ctx.measureText("6");
      asc = m6.actualBoundingBoxAscent > 0 ? m6.actualBoundingBoxAscent : F * 0.7;
      colW0 = dw[6]; colW1 = dw[1];
      gap = ctx.measureText("61").width - colW0 - colW1 - F * 0.03;
      numW = colW0 + colW1 + gap;
      numGrad = ctx.createLinearGradient(0, -asc * 0.55, 0, asc * 0.55);
      numGrad.addColorStop(0, css(U.mixRgb(A, WHITE, 0.9)));
      numGrad.addColorStop(0.42, css(U.mixRgb(A, WHITE, 0.42)));
      numGrad.addColorStop(1, css(U.mixRgb(A, [222, 112, 24], 0.42)));
      ctx.font = fontMono;
      monoCW = ctx.measureText("0000000000").width / 10;
      ctx.font = fontChip;
      chipHashW = ctx.measureText("#").width;
      ctx.font = fontMono;
      if (hasLS) ctx.letterSpacing = "1px";
      labelW = ctx.measureText("rank · national entrance exam").width;
      labelRankW = ctx.measureText("rank").width;
      if (hasLS) ctx.letterSpacing = "0px";
      ctx.restore();
      buildGlow();
    }

    /** Soft glow in the shape of "61": render tiny, downsample, upscale. */
    function buildGlow() {
      const sc = 1 / 8, pad = F * 0.5;
      const w = Math.ceil((numW + pad * 2) * sc), h = Math.ceil((asc + pad * 2) * sc);
      const a = document.createElement("canvas");
      a.width = w; a.height = h;
      const x = a.getContext("2d");
      x.translate(w / 2, h / 2);
      x.scale(sc, sc);
      x.font = fontNum; x.fillStyle = css(A); x.textAlign = "center"; x.textBaseline = "alphabetic";
      x.fillText("6", -numW / 2 + colW0 / 2, asc / 2);
      x.fillText("1", -numW / 2 + colW0 + gap + colW1 / 2, asc / 2);
      const b = document.createElement("canvas");
      b.width = Math.max(2, Math.ceil(w / 4)); b.height = Math.max(2, Math.ceil(h / 4));
      const y = b.getContext("2d");
      y.imageSmoothingQuality = "high";
      y.drawImage(a, 0, 0, b.width, b.height);
      // two cheap blur passes on the 1/8 layer
      const c = document.createElement("canvas");
      c.width = w; c.height = h;
      const z = c.getContext("2d");
      z.globalAlpha = 0.34;
      for (let i = 0; i < 9; i++) z.drawImage(a, ((i % 3) - 1) * 1.2, (((i / 3) | 0) - 1) * 1.2);
      glowA = c; glowB = b;
      glowA._w = w / sc; glowA._h = h / sc;
    }

    function buildScrims() {
      if (mobile) {
        scrimCap = ctx.createLinearGradient(0, H * 0.58, 0, H);
        scrimCap.addColorStop(0, "rgba(6,7,8,0)");
        scrimCap.addColorStop(0.55, "rgba(6,7,8,0.55)");
        scrimCap.addColorStop(1, "rgba(6,7,8,0.72)");
        scrimSide = ctx.createLinearGradient(0, H * 0.5, 0, H * 0.62);
      } else {
        const cx = gutter + 250, cy = H - capBottom - 64, rr = Math.max(440, W * 0.32);
        scrimCap = ctx.createRadialGradient(cx, cy, 0, cx, cy, rr);
        scrimCap.addColorStop(0, "rgba(6,7,8,0.74)");
        scrimCap.addColorStop(0.5, "rgba(6,7,8,0.5)");
        scrimCap.addColorStop(1, "rgba(6,7,8,0)");
        scrimSide = ctx.createLinearGradient(W * 0.54, 0, W * 0.7, 0);
      }
      scrimSide.addColorStop(0, "rgba(6,7,8,0)");
      scrimSide.addColorStop(1, "rgba(6,7,8,0.82)");
    }

    /* ---------------- portrait particles ---------------- */
    function fallbackPoints(cols, rows) {
      const out = [];
      for (let j = 0; j < rows; j++) {
        for (let i = 0; i < cols; i++) {
          const u = (i + 0.5) / cols, v = (j + 0.5) / rows;
          const head = Math.hypot((u - 0.5) / 0.22, (v - 0.36) / 0.27);
          const body = Math.hypot((u - 0.5) / 0.48, (v - 1.02) / 0.42);
          const lum = Math.max(head < 1 ? 0.8 - head * 0.3 : 0, body < 1 ? 0.5 - body * 0.2 : 0);
          if (lum > 0.07) out.push({ u, v, lum, r: lum * A[0], g: lum * A[1], b: lum * A[2] });
        }
      }
      return out;
    }

    /** The shared tri-tone portrait, with an extra elliptical feather so a
     *  bright photo background never shows the box's rectangular edges. */
    let feathered = null;
    function featherPortrait(src) {
      if (feathered && feathered._src === src) return feathered;
      const w = src.width, h = src.height;
      const c = document.createElement("canvas");
      c.width = w; c.height = h;
      const x = c.getContext("2d");
      x.drawImage(src, 0, 0);
      // inner vignette: keep the face, sink the bright backdrop toward the rim
      x.globalCompositeOperation = "source-atop";
      const v = x.createRadialGradient(w * 0.5, h * 0.42, w * 0.2, w * 0.5, h * 0.46, w * 0.62);
      v.addColorStop(0, "rgba(6,7,8,0)");
      v.addColorStop(0.55, "rgba(6,7,8,0.18)");
      v.addColorStop(1, "rgba(6,7,8,0.62)");
      x.fillStyle = v;
      x.fillRect(0, 0, w, h);
      // elliptical feather that reaches zero before every edge of the box
      x.globalCompositeOperation = "destination-in";
      x.translate(w * 0.5, h * 0.48);
      x.scale(1, (h * 0.48) / (w * 0.49));
      const g = x.createRadialGradient(0, 0, 0, 0, 0, w * 0.49);
      g.addColorStop(0, "rgba(0,0,0,1)");
      g.addColorStop(0.56, "rgba(0,0,0,1)");
      g.addColorStop(0.72, "rgba(0,0,0,0.82)");
      g.addColorStop(0.88, "rgba(0,0,0,0.3)");
      g.addColorStop(1, "rgba(0,0,0,0)");
      x.fillStyle = g;
      x.fillRect(-w, -h * 2, w * 2, h * 4);
      // a soft background-coloured matte, drawn first, so the faint path and
      // any particles never read through the feathered portrait
      const mt = document.createElement("canvas");
      mt.width = Math.ceil(w / 4); mt.height = Math.ceil(h / 4);
      const y = mt.getContext("2d");
      y.translate(mt.width / 2, mt.height * 0.5);
      y.scale(1, mt.height / mt.width);
      const mg = y.createRadialGradient(0, 0, 0, 0, 0, mt.width * 0.5);
      mg.addColorStop(0, "rgba(6,7,8,1)");
      mg.addColorStop(0.72, "rgba(6,7,8,0.96)");
      mg.addColorStop(1, "rgba(6,7,8,0)");
      y.fillStyle = mg;
      y.fillRect(-mt.width, -mt.height, mt.width * 2, mt.height * 2);
      c._matte = mt;
      c._src = src;
      feathered = c;
      return c;
    }

    function buildFace() {
      if (!N) return;
      const t0 = performance.now();
      box = U.portraitBox(W, H);
      const cols = mobile ? 42 : 58, rows = Math.round(cols * 1.25);
      let pts = null;
      if (img) {
        try {
          portrait = featherPortrait(U.portraitCanvas(img, A));
          const c = document.createElement("canvas");
          c.width = cols * 4; c.height = rows * 4;
          const x = c.getContext("2d");
          x.fillStyle = BG; x.fillRect(0, 0, c.width, c.height);
          x.imageSmoothingQuality = "high";
          x.drawImage(portrait, 0, 0, c.width, c.height);
          pts = U.samplePoints(c, cols, rows, 0.075, 13);
        } catch (e) { pts = null; }
      }
      if (!pts || !pts.length) pts = fallbackPoints(cols, rows);

      M = pts.length;
      ptx = new Float32Array(M); pty = new Float32Array(M); prad = new Float32Array(M);
      psrc = new Int32Array(M); pdel = new Float32Array(M); psw = new Float32Array(M);
      pjx = new Float32Array(M); pjy = new Float32Array(M);
      const bucket = new Uint8Array(M);
      const sp = box.w / cols, diag = Math.hypot(W, H);
      const sums = new Float64Array(NB * 4);
      const c76 = { s: 1, ox: 0, oy: 0 };
      camAt(0.76, c76);
      const rnd = U.rng(76);
      for (let i = 0; i < M; i++) {
        const pt = pts[i];
        ptx[i] = box.x + pt.u * box.w;
        pty[i] = box.y + pt.v * box.h;
        const b = Math.min(NB - 1, (pt.lum * NB) | 0);
        bucket[i] = b;
        prad[i] = sp * (0.09 + 0.37 * Math.pow((b + 0.5) / NB, 1.15)); // 6-step halftone
        sums[b * 4] += pt.r; sums[b * 4 + 1] += pt.g; sums[b * 4 + 2] += pt.b; sums[b * 4 + 3]++;
        // The whole viewport folds into the portrait box: each particle starts
        // at the node that sits where its pixel would be at full-screen scale.
        const sx = (pt.u + (rnd() - 0.5) * 0.05) * W, sy = (pt.v + (rnd() - 0.5) * 0.05) * H;
        const n = nearestGrid((sx - c76.ox) / c76.s, (sy - c76.oy) / c76.s);
        psrc[i] = n;
        const nsx = X[n] * c76.s + c76.ox, nsy = Y[n] * c76.s + c76.oy;
        pdel[i] = 0.07 * rnd() + MAXDEL_DIST * Math.min(1, Math.hypot(ptx[i] - nsx, pty[i] - nsy) / (diag * 0.6));
        psw[i] = 0.08 + 0.16 * rnd();
        const ja = rnd() * TAU, jr = R * 0.45 * Math.sqrt(rnd());
        pjx[i] = Math.cos(ja) * jr; pjy[i] = Math.sin(ja) * jr;
      }
      // counting sort by bucket
      bOff.fill(0);
      for (let i = 0; i < M; i++) bOff[bucket[i] + 1]++;
      for (let b = 0; b < NB; b++) bOff[b + 1] += bOff[b];
      bIdx = new Int32Array(M);
      const cur = bOff.slice(0, NB);
      for (let i = 0; i < M; i++) bIdx[cur[bucket[i]]++] = i;
      landed = [];
      for (let b = 0; b < NB; b++) {
        const n = sums[b * 4 + 3] || 1;
        bCol[b] = css([sums[b * 4] / n, sums[b * 4 + 1] / n, sums[b * 4 + 2] / n]);
        bAlpha[b] = 0.42 + 0.58 * ((b + 0.5) / NB);
        bRad[b] = sp * (0.09 + 0.37 * Math.pow((b + 0.5) / NB, 1.15));
        const p2 = new Path2D();
        for (let q = bOff[b]; q < bOff[b + 1]; q++) {
          const i = bIdx[q];
          p2.moveTo(ptx[i] + prad[i], pty[i]);
          p2.arc(ptx[i], pty[i], prad[i], 0, TAU);
        }
        landed.push(p2);
      }
      stats.particles = M;
      stats.faceMs = Math.round((performance.now() - t0) * 10) / 10;
    }

    /* ---------------- drawing helpers ---------------- */
    /** Append the first `upto` px of a node polyline to the current path. */
    function tracePath(nodes, cum, upto) {
      let x = X[nodes[0]], y = Y[nodes[0]];
      ctx.moveTo(x, y);
      for (let i = 1; i < nodes.length; i++) {
        const nx = X[nodes[i]], ny = Y[nodes[i]];
        if (cum[i] <= upto) { ctx.lineTo(nx, ny); x = nx; y = ny; }
        else {
          const f = (upto - cum[i - 1]) / Math.max(1e-6, cum[i] - cum[i - 1]);
          x += (nx - x) * f; y += (ny - y) * f;
          ctx.lineTo(x, y);
          break;
        }
      }
      headX = x; headY = y;
    }

    /** Append the polyline between arc lengths [from, to] to the current path. */
    function tracePathRange(nodes, cum, from, to) {
      if (to <= from) { pointAt(nodes, cum, to); ctx.moveTo(headX, headY); return; }
      pointAt(nodes, cum, from);
      let x = headX, y = headY;
      ctx.moveTo(x, y);
      let i = 1;
      while (i < nodes.length && cum[i] <= from) i++;
      for (; i < nodes.length; i++) {
        const nx = X[nodes[i]], ny = Y[nodes[i]];
        if (cum[i] <= to) { ctx.lineTo(nx, ny); x = nx; y = ny; }
        else {
          const f = (to - cum[i - 1]) / Math.max(1e-6, cum[i] - cum[i - 1]);
          const px = X[nodes[i - 1]], py = Y[nodes[i - 1]];
          x = px + (nx - px) * f; y = py + (ny - py) * f;
          ctx.lineTo(x, y);
          break;
        }
      }
      headX = x; headY = y;
    }

    /** Arc length from t at which the optimal path leaves the number's box. */
    function pathTrim() {
      const s = cam.s, ox = cam.ox, oy = cam.oy;
      const pad = F * 0.05 * NG.kk;
      const x0 = NG.cx - numW * 0.5 * NG.kk - pad, x1 = NG.cx + numW * 0.5 * NG.kk + pad;
      const y0 = NG.cy - asc * 0.5 * NG.kk - pad;
      let y1 = NG.cy + asc * 0.5 * NG.kk + pad;
      if (NG.la > 0.05) y1 = Math.max(y1, NG.ly + 12);
      let px = X[path[0]] * s + ox, py = Y[path[0]] * s + oy;
      if (!(px > x0 && px < x1 && py > y0 && py < y1)) return 0;
      for (let i = 1; i < path.length; i++) {
        const nx = X[path[i]] * s + ox, ny = Y[path[i]] * s + oy;
        if (!(nx > x0 && nx < x1 && ny > y0 && ny < y1)) {
          const dx = nx - px, dy = ny - py;
          let te = 1;
          if (dx > 0) te = Math.min(te, (x1 - px) / dx); else if (dx < 0) te = Math.min(te, (x0 - px) / dx);
          if (dy > 0) te = Math.min(te, (y1 - py) / dy); else if (dy < 0) te = Math.min(te, (y0 - py) / dy);
          return pathCum[i - 1] + (pathCum[i] - pathCum[i - 1]) * U.clamp(te);
        }
        px = nx; py = ny;
      }
      return pathLen;
    }

    function pointAt(nodes, cum, d) {
      let i = 1;
      while (i < nodes.length - 1 && cum[i] < d) i++;
      const a = nodes[i - 1], b = nodes[i];
      const f = U.clamp((d - cum[i - 1]) / Math.max(1e-6, cum[i] - cum[i - 1]));
      headX = X[a] + (X[b] - X[a]) * f; headY = Y[a] + (Y[b] - Y[a]) * f;
    }

    function visitedAt(D) {
      let lo = 0, hi = V;
      while (lo < hi) { const mid = (lo + hi) >> 1; if (orderDist[mid] <= D) lo = mid + 1; else hi = mid; }
      return lo;
    }

    function sprite(img, x, y, size, alpha) {
      ctx.globalAlpha = alpha;
      ctx.drawImage(img, x - size / 2, y - size / 2, size, size);
    }

    /* ---------------- render ---------------- */
    let hudK = 0, hudF = 0, hudD = 0;

    function drawWorld(p, t) {
      const s = cam.s, lw = 1 / s;
      const graphOut = 1 - U.ease.inOutSine(U.range(p, 0.75, 0.84));
      const nodesOut = 1 - U.range(p, 0.765, 0.8);
      const calm3 = 1 - 0.62 * U.ease.inOutSine(U.range(p, 0.47, 0.58));

      ctx.setTransform(DPR * s, 0, 0, DPR * s, DPR * cam.ox, DPR * cam.oy);
      ctx.lineCap = "round"; ctx.lineJoin = "round";

      // 1 · base graph
      if (graphOut > 0) {
        ctx.lineWidth = lw;
        ctx.strokeStyle = "#ffffff";
        ctx.globalAlpha = 0.085 * graphOut * calm3; ctx.stroke(edgeShort);
        ctx.globalAlpha = 0.05 * graphOut * calm3; ctx.stroke(edgeLong);
        ctx.fillStyle = C_NODE;
        for (let g = 0; g < 3; g++) {
          const tw = reduced ? 0.5 : 0.5 + 0.5 * Math.sin(t * 0.0011 + g * 2.094);
          ctx.globalAlpha = (0.24 + 0.3 * tw) * nodesOut * calm3;
          ctx.fill(nodeGroups[g]);
        }
      }

      // 2 · beat 1: candidate routes probed one after another
      const altOut = (1 - U.range(p, 0.22, 0.3)) * graphOut;
      if (altOut > 0 && alts.length) {
        const n = alts.length;
        for (let i = 0; i < n; i++) {
          const a0 = 0.01 + i * (0.165 / n);
          const dr = U.range(p, a0, a0 + 0.065);
          if (dr <= 0) continue;
          const al = alts[i];
          const recent = 1 - U.range(p, a0 + 0.065, a0 + 0.15);
          ctx.beginPath();
          tracePath(al.nodes, al.cum, U.ease.inOutSine(dr) * al.len);
          ctx.lineWidth = lw;
          ctx.strokeStyle = C_INK;
          ctx.globalAlpha = (0.1 + 0.36 * recent) * altOut;
          ctx.stroke();
          if (dr < 1) {
            ctx.globalCompositeOperation = "lighter";
            sprite(sprWhite, headX, headY, 22 * lw, 0.85 * altOut);
            ctx.globalCompositeOperation = "source-over";
          }
        }
      }

      // 3 · beat 2: Dijkstra wavefront (shortest-path tree grows edge by edge)
      hudK = 0; hudF = 0; hudD = 0;
      if (p > 0.25 && graphOut > 0) {
        const wave = U.range(p, 0.25, 0.43);
        const D = DT * (0.55 * wave + 0.45 * U.ease.inOutSine(wave));
        const k = visitedAt(D);
        const after = U.ease.inOutSine(U.range(p, 0.44, 0.52));
        const treeK = (1 - 0.5 * after) * calm3 * graphOut;
        const hot = DT * 0.06, warm = DT * 0.2, lim = D + maxEdge;
        const ageD = D + after * warm * 1.5; // once t is settled the front cools
        const growB = after > 0.3 ? 1 : 0;
        ctx.lineWidth = lw;
        for (let b = 0; b < 3; b++) {
          ctx.beginPath();
          for (let j = 1; j < k; j++) {
            const age = ageD - orderDist[j];
            if ((age < hot ? 0 : age < warm ? 1 : 2) !== b) continue;
            const v = order[j], u = pred[v];
            ctx.moveTo(X[u], Y[u]); ctx.lineTo(X[v], Y[v]);
          }
          if (b === growB) {
            for (let j = k; j < count && orderDist[j] < lim; j++) {
              const v = order[j], u = pred[v], du = dist[u];
              if (du > D) continue;
              const f = (D - du) / Math.max(1e-6, orderDist[j] - du);
              ctx.moveTo(X[u], Y[u]);
              ctx.lineTo(X[u] + (X[v] - X[u]) * f, Y[u] + (Y[v] - Y[u]) * f);
            }
          }
          ctx.strokeStyle = b === 0 ? C_AMBER_HI : C_AMBER;
          ctx.globalAlpha = (b === 0 ? 0.9 : b === 1 ? 0.5 : 0.26) * treeK;
          ctx.stroke();
        }
        // visited nodes
        for (let b = 0; b < 3; b++) {
          ctx.beginPath();
          const rr = b === 0 ? 1.7 : 1.45;
          for (let j = 0; j < k; j++) {
            const age = ageD - orderDist[j];
            if ((age < hot ? 0 : age < warm ? 1 : 2) !== b) continue;
            const v = order[j];
            ctx.moveTo(X[v] + rr, Y[v]); ctx.arc(X[v], Y[v], rr, 0, TAU);
          }
          ctx.fillStyle = b === 0 ? C_AMBER_PALE : C_AMBER;
          ctx.globalAlpha = (b === 0 ? 1 : b === 1 ? 0.8 : 0.5) * treeK;
          ctx.fill();
        }
        // frontier: in the priority queue = discovered, not yet settled
        let front = 0;
        ctx.beginPath();
        const ringR = (4.2 + (reduced ? 0 : 0.6 * Math.sin(t * 0.004))) * lw;
        for (let j = k; j < count && orderDist[j] < lim; j++) {
          const v = order[j];
          if (disc[v] < 0 || disc[v] >= k) continue;
          front++;
          ctx.moveTo(X[v] + ringR, Y[v]); ctx.arc(X[v], Y[v], ringR, 0, TAU);
        }
        const frontA = (1 - 0.65 * after) * treeK;
        ctx.strokeStyle = C_AMBER; ctx.globalAlpha = 0.75 * frontA; ctx.stroke();
        ctx.beginPath();
        for (let j = k; j < count && orderDist[j] < lim; j++) {
          const v = order[j];
          if (disc[v] < 0 || disc[v] >= k) continue;
          ctx.moveTo(X[v] + 1.6, Y[v]); ctx.arc(X[v], Y[v], 1.6, 0, TAU);
        }
        ctx.fillStyle = C_AMBER_PALE; ctx.globalAlpha = frontA; ctx.fill();
        // settle flashes
        ctx.globalCompositeOperation = "lighter";
        for (let j = k - 1; j >= 0; j--) {
          const age = ageD - orderDist[j];
          if (age >= hot) break;
          const v = order[j], f = 1 - age / hot;
          sprite(sprAmber, X[v], Y[v], (8 + 14 * f) * lw, 0.75 * f * treeK);
        }
        ctx.globalCompositeOperation = "source-over";
        hudK = Math.min(k, V); hudF = front; hudD = D;
      }

      // 4 · the optimal path, drawn back from t to s
      const pd = U.range(p, 0.43, 0.495);
      if (pd > 0) {
        const pa = 1 - 0.84 * U.ease.inOutSine(U.range(p, 0.76, 0.86));
        const trim = NG.on ? pathTrim() : 0;
        ctx.globalCompositeOperation = "lighter";
        ctx.beginPath();
        tracePathRange(path, pathCum, trim, U.ease.inOutCubic(pd) * pathLen);
        ctx.strokeStyle = C_AMBER;
        ctx.lineWidth = 8 * lw; ctx.globalAlpha = 0.1 * pa; ctx.stroke();
        ctx.lineWidth = 3 * lw; ctx.globalAlpha = 0.28 * pa; ctx.stroke();
        ctx.strokeStyle = C_AMBER_PALE;
        ctx.lineWidth = 1.3 * lw; ctx.globalAlpha = 0.95 * pa; ctx.stroke();
        if (pd < 1) sprite(sprPale, headX, headY, 40 * lw, 0.95);
        if (trim > 0) {
          const ea = NG.alpha * (1 - NG.toChip);
          pointAt(path, pathCum, trim);
          sprite(sprPale, headX, headY, 26 * lw, 0.9 * ea);
          ctx.beginPath(); ctx.arc(headX, headY, 2.2 * lw, 0, TAU);
          ctx.fillStyle = C_AMBER_PALE; ctx.globalAlpha = ea; ctx.fill();
        }
        // ambient: a light pulse travels s → t, into the number
        const pulseA = U.range(p, 0.5, 0.56) * (1 - U.range(p, 0.72, 0.77));
        if (!reduced && pulseA > 0) {
          const ph = (t * 0.00032) % 1;
          pointAt(path, pathCum, trim + (1 - ph) * (pathLen - trim));
          sprite(sprPale, headX, headY, 30 * lw, 0.9 * pulseA * Math.sin(Math.PI * ph));
        }
        ctx.globalCompositeOperation = "source-over";
      }
    }

    function drawMarkers(p, t) {
      const s = cam.s;
      const mA = 1 - U.ease.inOutSine(U.range(p, 0.75, 0.82));
      if (mA <= 0) return;
      ctx.font = fontTiny; ctx.textBaseline = "middle"; ctx.textAlign = "left";
      ctx.lineWidth = 1;
      // s
      const sx = X[S] * s + cam.ox, sy = Y[S] * s + cam.oy;
      const sA = mA * (1 - 0.5 * U.range(p, 0.5, 0.56));
      ctx.globalCompositeOperation = "lighter";
      sprite(sprWhite, sx, sy, 26, 0.5 * sA);
      ctx.globalCompositeOperation = "source-over";
      ctx.beginPath(); ctx.arc(sx, sy, 6.5, 0, TAU);
      ctx.strokeStyle = C_INK; ctx.globalAlpha = 0.65 * sA; ctx.stroke();
      ctx.beginPath(); ctx.arc(sx, sy, 2.3, 0, TAU);
      ctx.fillStyle = C_INK; ctx.globalAlpha = sA; ctx.fill();
      ctx.fillStyle = C_MUTED; ctx.globalAlpha = 0.9 * sA;
      ctx.fillText("s", sx + 11, sy - 10);
      // t — gives way to the number
      const tA = mA * (1 - U.range(p, 0.49, 0.52));
      if (tA > 0) {
        const tx = X[T] * s + cam.ox, ty = Y[T] * s + cam.oy;
        const pulse = reduced ? 0.5 : 0.5 + 0.5 * Math.sin(t * 0.003);
        ctx.globalCompositeOperation = "lighter";
        sprite(sprAmber, tx, ty, 30 + 8 * pulse, (0.35 + 0.25 * pulse) * tA);
        ctx.globalCompositeOperation = "source-over";
        ctx.beginPath(); ctx.arc(tx, ty, 6.5, 0, TAU);
        ctx.strokeStyle = C_AMBER; ctx.globalAlpha = 0.85 * tA; ctx.stroke();
        ctx.beginPath(); ctx.arc(tx, ty, 2.3, 0, TAU);
        ctx.fillStyle = C_AMBER_PALE; ctx.globalAlpha = tA; ctx.fill();
        ctx.fillStyle = C_AMBER; ctx.globalAlpha = 0.9 * tA;
        ctx.fillText("t", tx + 11, ty - 10);
      }
    }

    /** Particle i at eased flight progress e → (PX, PY), screen space. */
    let PX = 0, PY = 0;
    function particleAt(i, e) {
      const n = psrc[i];
      const sx = (X[n] + pjx[i]) * cam.s + cam.ox, sy = (Y[n] + pjy[i]) * cam.s + cam.oy;
      const dx = ptx[i] - sx, dy = pty[i] - sy, sw = psw[i] * Math.sin(Math.PI * e);
      PX = sx + dx * e - dy * sw; PY = sy + dy * e + dx * sw;
    }
    const ease3 = (t) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2);

    /** Nodes fly into the halftone. In flight each particle is a short
     *  round-capped streak (its own motion blur), one stroke per bucket. */
    function drawParticles(conv, alpha) {
      ctx.globalCompositeOperation = "lighter";
      if (conv >= 1) {
        for (let b = 0; b < NB; b++) {
          ctx.globalAlpha = bAlpha[b] * alpha; ctx.fillStyle = bCol[b];
          ctx.fill(landed[b]);
        }
      } else {
        const span = 1 - MAXDEL, g0 = U.clamp((conv - 0.35) / 0.65), grow = g0 * g0;
        ctx.lineCap = "round";
        for (let b = 0; b < NB; b++) {
          ctx.beginPath();
          for (let q = bOff[b]; q < bOff[b + 1]; q++) {
            const i = bIdx[q];
            const tau = U.clamp((conv - pdel[i]) / span);
            particleAt(i, ease3(Math.max(0, tau - 0.13)));
            ctx.moveTo(PX, PY);
            particleAt(i, ease3(tau));
            ctx.lineTo(PX + 0.01, PY);
          }
          ctx.lineWidth = 2 * (0.65 + (bRad[b] - 0.65) * grow);
          ctx.strokeStyle = bCol[b];
          ctx.globalAlpha = Math.min(1, bAlpha[b] * (1.25 - 0.25 * conv)) * alpha;
          ctx.stroke();
        }
      }
      ctx.globalCompositeOperation = "source-over";
    }

    /* Number + chip geometry, computed once per frame (no allocations). */
    const CH = { x: 0, y: 0, w: 0, h: 0, cx: 0, cy: 0, padX: 12 };
    const NG = { on: false, ap: 0, cx: 0, cy: 0, kk: 1, toChip: 0, alpha: 0, la: 0, ly: 0 };

    function numberGeom(p) {
      const ap = U.range(p, 0.497, 0.565);
      NG.on = ap > 0 && !!box;
      if (!NG.on) return;
      CH.x = box.x + (mobile ? 12 : 18); CH.y = box.y + (mobile ? 12 : 18); CH.h = mobile ? 28 : 30;
      const w61 = numW * chipFs / F;
      CH.w = CH.padX * 2 + chipHashW + 2 + w61;
      CH.cx = CH.x + CH.padX + chipHashW + 2 + w61 / 2; CH.cy = CH.y + CH.h / 2;
      const tsx = X[T] * cam.s + cam.ox, tsy = Y[T] * cam.s + cam.oy;
      const toChip = U.ease.inOutCubic(U.range(p, 0.765, 0.875));
      const ae = U.ease.outExpo(ap), am = U.ease.outCubic(ap);
      let cx = U.lerp(tsx, ax, am), cy = U.lerp(tsy, ay, am), kk = 0.5 + 0.5 * ae;
      if (toChip > 0) {
        cx = U.lerp(cx, CH.cx, toChip); cy = U.lerp(cy, CH.cy, toChip);
        kk *= Math.pow(chipFs / F, toChip);
      }
      NG.ap = ap; NG.cx = cx; NG.cy = cy; NG.kk = kk; NG.toChip = toChip;
      NG.alpha = Math.min(1, ap * 2.4);
      NG.la = U.range(p, 0.565, 0.62) * (1 - U.range(p, 0.735, 0.775));
      NG.ly = ay + asc * 0.5 + (mobile ? 30 : 42);
    }

    function drawDigitColumn(xc, v, settled, alpha) {
      if (settled) {
        ctx.globalAlpha = alpha;
        ctx.fillText(DIG[((Math.round(v) % 10) + 10) % 10], xc, asc / 2);
        return;
      }
      // Ticker cadence: each digit holds crisp for ~65% of its step, then
      // slides; a soft alpha falloff replaces a hard clip window.
      const base = Math.floor(v), fr = v - base, lh = asc * 1.1;
      const st = U.clamp(fr / 0.35), f = st * st * (3 - 2 * st);
      ctx.save();
      for (let q = 0; q < 2; q++) {
        const y = (q === 0 ? -f : 1 - f) * lh;
        const fall = U.clamp(1 - Math.abs(y) / (lh * 0.62));
        if (fall <= 0) continue;
        ctx.globalAlpha = alpha * fall * fall;
        ctx.translate(0, y); // the gradient travels with the glyph
        ctx.fillText(DIG[(((base + q) % 10) + 10) % 10], xc, asc / 2);
        ctx.translate(0, -y);
      }
      ctx.restore();
    }

    function drawNumber(p) {
      if (!NG.on) return;
      const tsx = X[T] * cam.s + cam.ox, tsy = Y[T] * cam.s + cam.oy;

      // bloom + shock ring at the target node
      const fl = U.range(p, 0.49, 0.61);
      if (fl > 0 && fl < 1) {
        const fe = U.ease.outCubic(fl);
        ctx.globalCompositeOperation = "lighter";
        sprite(sprBloom, tsx, tsy, F * (0.3 + 2.4 * fe), 0.5 * Math.sin(Math.PI * Math.min(1, fl * 1.3)));
        ctx.globalCompositeOperation = "source-over";
        ctx.beginPath(); ctx.arc(tsx, tsy, F * 1.05 * fe, 0, TAU);
        ctx.lineWidth = 1; ctx.strokeStyle = C_AMBER; ctx.globalAlpha = 0.5 * (1 - fe); ctx.stroke();
      }

      // label under the number
      const la = NG.la;
      if (la > 0) {
        const ly = NG.ly, lx = ax - labelW / 2;
        ctx.font = fontMono; ctx.textBaseline = "middle"; ctx.textAlign = "left";
        if (hasLS) ctx.letterSpacing = "1px";
        ctx.globalAlpha = la; ctx.fillStyle = C_AMBER;
        ctx.fillText("rank", lx, ly);
        ctx.fillStyle = C_MUTED;
        ctx.fillText(" · national entrance exam", lx + labelRankW, ly);
        if (hasLS) ctx.letterSpacing = "0px";
        const rule = mobile ? 18 : 28, gapR = 14;
        ctx.fillStyle = C_DIM; ctx.globalAlpha = 0.8 * la;
        ctx.fillRect(lx - gapR - rule, Math.round(ly) - 0.5, rule, 1);
        ctx.fillRect(lx + labelW + gapR, Math.round(ly) - 0.5, rule, 1);
      }

      // chip that the number parks in
      const ca = U.range(p, 0.845, 0.885);
      if (ca > 0) {
        U.roundRect(ctx, CH.x + 0.5, CH.y + 0.5, CH.w, CH.h, CH.h / 2);
        ctx.globalAlpha = 0.72 * ca; ctx.fillStyle = BG; ctx.fill();
        ctx.lineWidth = 1; ctx.strokeStyle = C_AMBER; ctx.globalAlpha = 0.5 * ca; ctx.stroke();
        ctx.font = fontChip; ctx.textBaseline = "alphabetic"; ctx.textAlign = "left";
        ctx.fillStyle = C_AMBER; ctx.globalAlpha = 0.65 * ca;
        ctx.fillText("#", CH.x + CH.padX, CH.cy + (asc * chipFs / F) / 2);
      }

      // the number
      const kk = NG.kk, alpha = NG.alpha;
      ctx.setTransform(DPR * kk, 0, 0, DPR * kk, DPR * NG.cx, DPR * NG.cy);
      const roll = U.range(p, 0.5, 0.625);
      const gA = alpha * (1 - NG.toChip) * U.clamp((roll - 0.9) / 0.1);
      if (gA > 0 && glowA) {
        ctx.globalCompositeOperation = "lighter";
        const gw = glowA._w, gh = glowA._h;
        ctx.globalAlpha = 0.5 * gA; ctx.drawImage(glowB, -gw / 2, -gh / 2, gw, gh);
        ctx.globalAlpha = 0.3 * gA; ctx.drawImage(glowA, -gw / 2, -gh / 2, gw, gh);
        ctx.globalCompositeOperation = "source-over";
      }
      ctx.font = fontNum; ctx.textAlign = "center"; ctx.textBaseline = "alphabetic";
      ctx.fillStyle = numGrad;
      const re = 1 - Math.pow(1 - roll, 3);
      const settled = roll >= 1;
      drawDigitColumn(-numW / 2 + colW0 / 2, 6 + 20 * (1 - re), settled, alpha);
      drawDigitColumn(-numW / 2 + colW0 + gap + colW1 / 2, 1 + 40 * (1 - re), settled, alpha);
      ctx.setTransform(DPR, 0, 0, DPR, 0, 0);
    }

    function drawHud(p) {
      const a = U.range(p, 0.255, 0.29) * (1 - U.range(p, 0.49, 0.53));
      if (a <= 0 || hudK <= 0) return;
      const nStr = String(N), nd = nStr.length;
      const kStr = String(hudK).padStart(nd, " ");
      const fStr = String(hudF).padStart(2, " ");
      const dStr = (hudD / R).toFixed(1).padStart(4, " ");
      const chars = 8 + nd + 3 + nd + 12 + fStr.length + 8 + dStr.length;
      const padX = 14, dotW = 14, h = mobile ? 30 : 32;
      const w = padX * 2 + dotW + chars * monoCW;
      const x = mobile ? gutter : W - gutter - w;
      const y = mobile ? 80 : H - capBottom - h;
      U.roundRect(ctx, x + 0.5, y + 0.5, w, h, 8);
      ctx.globalAlpha = 0.72 * a; ctx.fillStyle = BG; ctx.fill();
      ctx.lineWidth = 1; ctx.strokeStyle = "#ffffff"; ctx.globalAlpha = 0.1 * a; ctx.stroke();
      // progress hairline (visited / V)
      const bx = x + padX, bw = w - padX * 2, by = y + h - 0.5;
      ctx.fillStyle = C_AMBER; ctx.globalAlpha = 0.85 * a;
      ctx.fillRect(bx, by - 0.5, bw * (hudK / V), 1);
      // live dot
      const running = U.range(p, 0.25, 0.43) < 1;
      ctx.beginPath(); ctx.arc(x + padX + 3, y + h / 2, 2.6, 0, TAU);
      ctx.fillStyle = running ? C_AMBER : C_AMBER_PALE; ctx.globalAlpha = a; ctx.fill();
      // text
      ctx.font = fontMono; ctx.textBaseline = "middle"; ctx.textAlign = "left";
      let cx = x + padX + dotW;
      const cy = y + h / 2 + 0.5;
      const seg = (str, col) => { ctx.fillStyle = col; ctx.fillText(str, cx, cy); cx += str.length * monoCW; };
      ctx.globalAlpha = a;
      seg("visited ", C_MUTED); seg(kStr, C_INK); seg(" / ", C_DIM); seg(nStr, C_MUTED);
      seg(" · frontier ", C_DIM); seg(fStr, C_INK);
      seg(" · dist ", C_DIM); seg(dStr, C_AMBER_HI);
    }

    function render(p, timeMs) {
      const t = reduced ? 0 : timeMs || 0;
      ctx.setTransform(DPR, 0, 0, DPR, 0, 0);
      ctx.globalCompositeOperation = "source-over";
      ctx.globalAlpha = 1;
      ctx.fillStyle = BG;
      ctx.fillRect(0, 0, W, H);
      if (!N) return;
      camAt(p, cam);
      numberGeom(p);

      drawWorld(p, t);
      ctx.setTransform(DPR, 0, 0, DPR, 0, 0);
      drawMarkers(p, t);

      // keep the caption corner quiet while captions are on
      const capA = 1 - U.range(p, 0.76, 0.82);
      if (capA > 0) {
        ctx.globalAlpha = capA; ctx.fillStyle = scrimCap;
        if (mobile) ctx.fillRect(0, H * 0.58, W, H * 0.42);
        else ctx.fillRect(0, H * 0.25, Math.min(W, gutter + 250 + Math.max(440, W * 0.32)), H * 0.75);
      }

      // convergence: nodes → halftone face → portrait
      const conv = U.range(p, 0.762, 0.88);
      const partA = U.range(p, 0.762, 0.79) * (1 - U.range(p, 0.875, 0.93));
      if (M && conv > 0 && partA > 0) drawParticles(conv, partA);

      // calm the person-card side
      const sideA = U.range(p, 0.74, 0.8);
      if (sideA > 0) {
        ctx.globalAlpha = sideA; ctx.fillStyle = scrimSide;
        if (mobile) ctx.fillRect(0, H * 0.5, W, H * 0.5);
        else ctx.fillRect(W * 0.54, 0, W * 0.46, H);
      }

      const q = U.ease.inOutSine(U.range(p, 0.85, 0.925));
      if (portrait && box && q > 0) {
        if (portrait._matte) {
          ctx.globalAlpha = q;
          ctx.drawImage(portrait._matte, box.x - box.w * 0.04, box.y - box.h * 0.03, box.w * 1.08, box.h * 1.06);
        }
        ctx.globalAlpha = q;
        ctx.drawImage(portrait, box.x, box.y, box.w, box.h);
      }

      drawNumber(p);
      drawHud(p);
      ctx.globalAlpha = 1;
    }

    /* ---------------- lifecycle ---------------- */
    function resize(w, h, dpr) {
      W = w; H = h; DPR = dpr; mobile = U.isMobile(w);
      gutter = U.clamp(w * 0.04, 16, 48);
      capBottom = U.clamp(h * 0.1, 40, 110);
      const t0 = performance.now();
      buildGraph();
      buildRoute();
      stats.nodes = N; stats.edges = E; stats.visited = V;
      stats.buildMs = Math.round((performance.now() - t0) * 10) / 10;
      buildText();
      buildScrims();
      buildFace();
    }

    const idle = window.requestIdleCallback || ((fn) => setTimeout(fn, 60));
    if (opts.portraitSrc) {
      U.loadImage(opts.portraitSrc).then((im) => {
        if (!alive) return;
        img = im;
        // the tri-tone pass is ~a few dozen ms: keep it off the scroll path
        idle(() => { if (alive && W) { buildFace(); if (opts.invalidate) opts.invalidate(); } }, { timeout: 300 });
      }).catch(() => {});
    }
    if (document.fonts && document.fonts.load) {
      Promise.all([
        document.fonts.load(`700 100px "Space Grotesk"`),
        document.fonts.load(`600 15px "Space Grotesk"`),
        document.fonts.load(`500 12px "JetBrains Mono"`),
      ]).then(() => { if (alive && W) { buildText(); if (opts.invalidate) opts.invalidate(); } }).catch(() => {});
    }

    return {
      resize,
      render,
      destroy() { alive = false; },
    };
  };
})();
