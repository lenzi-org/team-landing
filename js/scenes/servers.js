/* ==========================================================================
   Scene 01 · "servers" — Samim Sanaei · Full-stack & DevOps
   --------------------------------------------------------------------------
   0.00–0.20  One server boots on a dot-grid floor; a boot log types itself.
   0.20–0.45  The topology unfolds out of it: edge → lb → app ×3 → redis/pg.
              Packets flow; the HUD shows req/s climbing at a calm p95.
   0.45–0.66  Traffic spikes; deployment/web scales 3 → 12 pods on a grid.
   0.66–0.85  Every node, edge and packet dissolves into particles that
              settle on points sampled from the portrait.
   0.85–1.00  The particle face crossfades into the clean tri-tone portrait.
   State is derived from progress only; time adds ambient motion (LEDs,
   packet flow, marching ants, twinkle) and is 0 under reduced motion.
   ========================================================================== */
(function () {
  "use strict";

  const Registry = (window.SceneRegistry = window.SceneRegistry || {});

  Registry.servers = function createServersScene(canvas, opts) {
    const U = opts.utils || window.SceneUtils;
    const ctx = canvas.getContext("2d");
    const ACC = opts.accentRgb || [56, 217, 245];
    const REDUCED = !!opts.reducedMotion;
    const clamp = U.clamp, lerp = U.lerp, range = U.range, EZ = U.ease;
    const TAU = Math.PI * 2;
    const HAS_LS = "letterSpacing" in ctx;
    let destroyed = false;

    /* ------------------------------------------------------------------ */
    /* Palette — constant strings; alpha always goes through globalAlpha.  */
    /* ------------------------------------------------------------------ */
    const COL = {
      bg: "#060708",
      card: "#0b0e12",
      white: "#ffffff",
      text: "#eef0f3",
      muted: "#9aa1ad",
      dim: "#5f6673",
      acc: U.rgba(ACC, 1),
      green: "#5fe3a1",
      floor: "#c8d2de",
    };
    const DASH = [3, 4], NO_DASH = [];

    /* ------------------------------------------------------------------ */
    /* Pre-rendered sprites                                                */
    /* ------------------------------------------------------------------ */
    function radialSprite(size, stops) {
      const c = document.createElement("canvas");
      c.width = c.height = size;
      const x = c.getContext("2d");
      const g = x.createRadialGradient(size / 2, size / 2, 0, size / 2, size / 2, size / 2);
      for (let i = 0; i < stops.length; i++) g.addColorStop(stops[i][0], stops[i][1]);
      x.fillStyle = g;
      x.fillRect(0, 0, size, size);
      return c;
    }
    const SPR_GLOW = U.glowSprite(ACC, 64, 0.16);
    const SPR_GREEN = U.glowSprite([95, 227, 161], 32, 0.22);
    const SPR_HALO = radialSprite(256, [
      [0, U.rgba(ACC, 1)], [0.3, U.rgba(ACC, 0.4)], [0.66, U.rgba(ACC, 0.07)], [1, U.rgba(ACC, 0)],
    ]);
    // Particle dots: tri-tone ramp accent → near-white (same ramp as U.portraitCanvas)
    const LIGHT = U.mixRgb(ACC, [255, 255, 255], 0.82);
    const NB = 6;
    const SPR_DOT = [];
    for (let i = 0; i < NB; i++) {
      const rgb = U.mixRgb(ACC, LIGHT, i / (NB - 1)).map((v) => Math.round(v));
      SPR_DOT.push(radialSprite(32, [
        [0, U.rgba(rgb, 1)], [0.42, U.rgba(rgb, 0.95)], [0.7, U.rgba(rgb, 0.28)], [1, U.rgba(rgb, 0)],
      ]));
    }
    const DOT_VIS = 0.64; // visible disc diameter ÷ sprite draw size

    /* ------------------------------------------------------------------ */
    /* Cached strings (no string building in the hot path)                 */
    /* ------------------------------------------------------------------ */
    const PCT = [];
    for (let i = 0; i <= 100; i++) PCT.push(i + "%");
    const CMD = "docker compose up";
    const CMD_PREFIX = [];
    for (let i = 0; i <= CMD.length; i++) CMD_PREFIX.push(CMD.slice(0, i));
    const NUM = [];
    for (let i = 0; i <= 12; i++) NUM.push(String(i));

    /* ------------------------------------------------------------------ */
    /* Topology model                                                      */
    /* ------------------------------------------------------------------ */
    const CDN = 0, LB = 1, REDIS = 2, PG = 3, REPL = 4, POD = 5, NPOD = 12, NN = POD + NPOD;
    const LABEL = ["cdn", "nginx", "redis", "postgres", "postgres",
      "django", "next.js", "django", "next.js", "django", "next.js",
      "django", "next.js", "django", "next.js", "django", "next.js"];
    const TAG = ["EDGE", "LOAD BALANCER", "CACHE", "PRIMARY", "REPLICA"];
    const TAG_M = ["EDGE", "LB", "CACHE", "PRIMARY", "REPLICA"]; // short: clear of vertical edges
    // Unfold timing (beat 2). Pod 1 is the original server itself.
    const APPEAR = [0.214, 0.2, 0.222, 0.228, 0.24, 0.206, 0, 0.209];
    // Pod grid slots — minor axis (across the flow) × major axis (along the app column).
    const SLOT_MIN = [0, 0, 0, -1, 1, -1, 1, -1, 1, 0, -1, 1];
    const SLOT_MAJ = [-1.5, -0.5, 0.5, -1.5, -1.5, -0.5, -0.5, 0.5, 0.5, 1.5, 1.5, 1.5];
    const SPAWN_SRC = [0, 1, 2, 0, 0, 1, 1, 2, 2, 2, 2, 2]; // new pods slide out from behind these
    const SPAWN0 = 0.478, SPAWN_STEP = 0.0135, SPAWN_DUR = 0.034;
    const POD_NOISE = new Float32Array(NPOD);
    {
      const r = U.rng(99);
      for (let k = 0; k < NPOD; k++) POD_NOISE[k] = (r() - 0.5) * 0.08;
    }

    const L0x = new Float32Array(NN), L0y = new Float32Array(NN);
    const L1x = new Float32Array(NN), L1y = new Float32Array(NN);
    const nx = new Float32Array(NN), ny = new Float32Array(NN);
    const nw = new Float32Array(NN), nh = new Float32Array(NN);
    const na = new Float32Array(NN), nfl = new Float32Array(NN), nsp = new Float32Array(NN);
    const inX = new Float32Array(NN), inY = new Float32Array(NN);
    const outX = new Float32Array(NN), outY = new Float32Array(NN);
    const pinX = new Float32Array(NPOD), pinY = new Float32Array(NPOD);
    const poutX = new Float32Array(NPOD), poutY = new Float32Array(NPOD);
    let gL = 0, gT = 0, gR = 0, gB = 0, gInX = 0, gInY = 0, gOutX = 0, gOutY = 0;
    let fE = 0, fM = 0, fB = 0, nReady = 3, nReadyF = 3;

    /* ------------------------------------------------------------------ */
    /* Packets — deterministic functions of (time, index)                  */
    /* ------------------------------------------------------------------ */
    const MAXP = 160;
    let PK_N = MAXP; // per-layout packet budget
    const PK_PH = new Float32Array(MAXP), PK_RATE = new Float32Array(MAXP);
    const PK_RAY = new Uint8Array(MAXP);
    {
      const r = U.rng(42);
      for (let j = 0; j < MAXP; j++) {
        PK_PH[j] = r();
        PK_RATE[j] = (0.86 + r() * 0.28) / 3.4; // ~3.4 s per journey
        PK_RAY[j] = (r() * 5) | 0;
      }
    }
    // Journey segments along u ∈ [0,1]
    const S_RAY = 0.14, S_CDN = 0.18, S_E1 = 0.33, S_LB = 0.37, S_E2 = 0.54, S_APP = 0.68, S_E3 = 0.97;
    const hash2 = (a, b) => {
      let h = Math.imul(a ^ 0x9e3779b9, 0x85ebca6b) ^ Math.imul(b + 0x632be5ab, 0xc2b2ae35);
      h ^= h >>> 15; h = Math.imul(h, 0x2c1b3c6d); h ^= h >>> 12;
      return h >>> 0;
    };

    /* ------------------------------------------------------------------ */
    /* Layout state (CSS px)                                               */
    /* ------------------------------------------------------------------ */
    let W = 0, H = 0, DPR = 1, MOB = false, VERT = false, K = 1, HAIR = 1, HALF = 0;
    let CARD_W = 0, CARD_H = 0, POD_W = 0, POD_H = 0, RAD = 7;
    let MAJ0 = 0, MAJ1 = 0, MIN1 = 0, GPAD = 0;
    let G0X = 0, G0Y = 0, G1X = 0, G1Y = 0;
    let SRV_X = 0, SRV_Y = 0, SRV_W = 0, SRV_H = 0, SRV_GRAD = null;
    let LOG_X = 0, LOG_Y = 0, LOG_LH = 0;
    let HUD_X = 0, HUD_Y = 0;
    let FLOOR_Y = 0, FLOOR_SP = 0, FLOOR_CX = 0;
    const FLOOR_CAM = 6; // camera height in grid cells
    let BOX = { x: 0, y: 0, w: 0, h: 0, cx: 0, cy: 0 };
    let RAY_GRAD = null, NRAYS = 5;
    const RAY_X = new Float32Array(5), RAY_Y = new Float32Array(5);
    const F = { label: "", tag: "", log: "", hud: "", hudS: "", group: "", srv: "" };
    let lastP = -1, lastT = 0;

    const snap = (v) => Math.round(v * DPR) / DPR + HALF;
    const px = (v) => Math.round(v * DPR) / DPR;
    const fs = (v) => Math.round(v * 2) / 2;
    const slotX = (k) => (VERT ? SLOT_MAJ[k] * MAJ1 : SLOT_MIN[k] * MIN1);
    const slotY = (k) => (VERT ? SLOT_MIN[k] * MIN1 : SLOT_MAJ[k] * MAJ1);
    const set = (ax, ay, i, x, y) => { ax[i] = x; ay[i] = y; };

    function layout() {
      MOB = U.isMobile(W);
      VERT = MOB;
      const dev = Math.max(1, Math.round(DPR));
      HAIR = dev / DPR;
      HALF = dev % 2 ? 0.5 / DPR : 0;
      BOX = U.portraitBox(W, H);

      if (!MOB) {
        K = Math.min(clamp((W - 120) / 1140, 0.7, 1.15), clamp(H / 820, 0.78, 1.15));
        CARD_W = POD_W = Math.round(112 * K);
        CARD_H = POD_H = Math.round(34 * K);
        RAD = 7 * K;
        const GY = Math.round(H * 0.42);
        // L0 — beat 2: four airy columns
        const gap0 = 150 * K;
        const span0 = 4 * CARD_W + 3 * gap0, left0 = W / 2 - span0 / 2;
        const col0 = (i) => left0 + CARD_W / 2 + i * (CARD_W + gap0);
        MAJ0 = 58 * K; MAJ1 = POD_H + 12 * K; MIN1 = POD_W + 14 * K; GPAD = 16 * K;
        const gridW = 2 * MIN1 + POD_W;
        set(L0x, L0y, CDN, col0(0), GY);
        set(L0x, L0y, LB, col0(1), GY);
        G0X = col0(2); G0Y = GY;
        for (let i = 0; i < 3; i++) set(L0x, L0y, POD + i, G0X, GY + (i - 1) * MAJ0);
        const d0 = col0(3);
        set(L0x, L0y, REDIS, d0, GY - 70 * K);
        set(L0x, L0y, PG, d0, GY + 10 * K);
        set(L0x, L0y, REPL, d0, GY + 80 * K);
        // L1 — beat 3: the deployment grid makes room
        const gapX = 104 * K;
        const span1 = 3 * CARD_W + gridW + 2 * GPAD + 3 * gapX, left1 = W / 2 - span1 / 2;
        set(L1x, L1y, CDN, left1 + CARD_W / 2, GY);
        set(L1x, L1y, LB, left1 + CARD_W * 1.5 + gapX, GY);
        G1X = left1 + 2 * CARD_W + 2 * gapX + GPAD + gridW / 2; G1Y = GY;
        const d1 = G1X + gridW / 2 + GPAD + gapX + CARD_W / 2;
        set(L1x, L1y, REDIS, d1, GY - 70 * K);
        set(L1x, L1y, PG, d1, GY + 10 * K);
        set(L1x, L1y, REPL, d1, GY + 80 * K);
        for (let i = 0; i < 3; i++) set(L1x, L1y, POD + i, G1X + slotX(i), G1Y + slotY(i));
        // Beat 1: server + boot log, centred as a pair
        SRV_W = Math.round(196 * K); SRV_H = Math.round(68 * K);
        LOG_LH = 22 * K;
        const comp = SRV_W + 56 * K + 236 * K;
        SRV_X = W / 2 - comp / 2 + SRV_W / 2; SRV_Y = GY;
        LOG_X = SRV_X + SRV_W / 2 + 56 * K; LOG_Y = GY - 2 * LOG_LH;
        HUD_X = W / 2; HUD_Y = Math.max(96, GY - 172 * K);
        FLOOR_Y = H * 0.33; FLOOR_CX = W / 2;
        NRAYS = 5; PK_N = MAXP;
        for (let r = 0; r < NRAYS; r++) { RAY_X[r] = -12; RAY_Y[r] = GY + (r - 2) * 66 * K; }
        RAY_GRAD = ctx.createLinearGradient(0, 0, W * 0.16, 0);
      } else {
        K = clamp(W / 390, 0.85, 1.2);
        CARD_W = Math.round(100 * K); CARD_H = Math.round(28 * K);
        POD_W = Math.round(74 * K); POD_H = Math.round(26 * K);
        RAD = 6 * K;
        const CX = W / 2;
        // vertical scale so the stack clears the captions (bottom ≈ H − 240)
        const vs = clamp((H - 240 - 98) / 400, 0.72, 1.1);
        const Y = (d) => 98 + (d - 98) * vs;
        const yCdn = Y(152), yLb = Y(214), GY = Y(318), yData = Y(438);
        MAJ0 = 100 * K; MAJ1 = POD_W + 8 * K; MIN1 = POD_H + 10 * K; GPAD = 10 * K;
        set(L0x, L0y, CDN, CX, yCdn); set(L1x, L1y, CDN, CX, yCdn);
        set(L0x, L0y, LB, CX, yLb); set(L1x, L1y, LB, CX, yLb);
        G0X = G1X = CX; G0Y = G1Y = GY;
        for (let i = 0; i < 3; i++) {
          set(L0x, L0y, POD + i, CX + (i - 1) * MAJ0, GY);
          set(L1x, L1y, POD + i, G1X + slotX(i), G1Y + slotY(i));
        }
        const dx = 110 * K;
        set(L0x, L0y, REDIS, CX - dx, yData); set(L1x, L1y, REDIS, CX - dx, yData);
        set(L0x, L0y, PG, CX, yData); set(L1x, L1y, PG, CX, yData);
        set(L0x, L0y, REPL, CX + dx, yData); set(L1x, L1y, REPL, CX + dx, yData);
        SRV_W = Math.round(168 * K); SRV_H = Math.round(58 * K);
        SRV_X = CX; SRV_Y = GY - 44 * vs;
        LOG_LH = 20 * K;
        LOG_X = CX - SRV_W / 2 + 2 * K; LOG_Y = SRV_Y + SRV_H / 2 + 34 * K;
        // HUD sits under the stack so the incoming request rays never cross it
        HUD_X = CX; HUD_Y = yData + CARD_H / 2 + 56 * vs;
        FLOOR_Y = GY - 60; FLOOR_CX = CX;
        NRAYS = 3; PK_N = 96;
        for (let r = 0; r < NRAYS; r++) { RAY_X[r] = CX + (r - 1) * 120 * K; RAY_Y[r] = -12; }
        RAY_GRAD = ctx.createLinearGradient(0, 0, 0, yCdn * 0.8);
      }
      FLOOR_SP = ((H - FLOOR_Y) * 1.1) / FLOOR_CAM; // px per cell at z = 1
      RAY_GRAD.addColorStop(0, "rgba(255,255,255,0)");
      RAY_GRAD.addColorStop(1, "rgba(255,255,255,1)");

      SRV_GRAD = ctx.createLinearGradient(0, 0, 0, SRV_H);
      SRV_GRAD.addColorStop(0, "#13171d");
      SRV_GRAD.addColorStop(1, "#0a0c10");

      const kt = clamp(K, 0.86, 1.1);
      const m = U.fonts.mono;
      F.label = `500 ${fs((MOB ? 9.5 : 11) * kt)}px ${m}`;
      F.tag = `500 ${fs((MOB ? 8 : 8.5) * kt)}px ${m}`;
      F.log = `400 ${fs((MOB ? 11.5 : 12.5) * kt)}px ${m}`;
      F.hud = `500 ${fs((MOB ? 9 : 10) * kt)}px ${m}`;
      F.hudS = `500 ${fs((MOB ? 10.5 : 11.5) * kt)}px ${m}`;
      F.group = `500 ${fs((MOB ? 9 : 10) * kt)}px ${m}`;
      F.srv = `500 ${fs(8.5 * kt)}px ${m}`;
    }

    /* ------------------------------------------------------------------ */
    /* Node state as a function of progress                                */
    /* ------------------------------------------------------------------ */
    function computeNodes(p) {
      const e = (fE = EZ.inOutCubic(range(p, 0.47, 0.605)));
      const m = (fM = EZ.inOutCubic(range(p, 0.185, 0.25)));
      fB = EZ.inOutCubic(range(p, 0.45, 0.49));
      const scx = lerp(SRV_X, L0x[POD + 1], m), scy = lerp(SRV_Y, L0y[POD + 1], m);
      const gcx = lerp(G0X, G1X, e), gcy = lerp(G0Y, G1Y, e);

      for (let i = 0; i < NN; i++) {
        nw[i] = i >= POD ? POD_W : CARD_W;
        nh[i] = i >= POD ? POD_H : CARD_H;
        nfl[i] = 0;
        nsp[i] = 1;
      }
      for (let i = 0; i < POD + 3; i++) {
        let x, y;
        if (i < POD) { x = lerp(L0x[i], L1x[i], e); y = lerp(L0y[i], L1y[i], e); }
        else { const k = i - POD; x = lerp(L0x[i], gcx + slotX(k), e); y = lerp(L0y[i], gcy + slotY(k), e); }
        if (i === POD + 1) {
          if (m < 1) { x = scx; y = scy; nw[i] = lerp(SRV_W, POD_W, m); nh[i] = lerp(SRV_H, POD_H, m); }
          na[i] = m;
        } else {
          const a = EZ.outCubic(range(p, APPEAR[i], APPEAR[i] + 0.05));
          x = lerp(scx, x, a); y = lerp(scy, y, a);
          na[i] = a;
        }
        nx[i] = x; ny[i] = y;
      }
      nReady = 3; nReadyF = 3;
      for (let k = 3; k < NPOD; k++) {
        const i = POD + k, s0 = SPAWN0 + (k - 3) * SPAWN_STEP;
        const s = range(p, s0, s0 + SPAWN_DUR), es = EZ.outCubic(s);
        const src = POD + SPAWN_SRC[k];
        nx[i] = lerp(nx[src], gcx + slotX(k), es);
        ny[i] = lerp(ny[src], gcy + slotY(k), es);
        na[i] = range(s, 0, 0.4);
        nsp[i] = s;
        if (s >= 0.62) nReady++;
        nReadyF += EZ.inOutSine(range(s, 0.3, 1));
      }
      // Deployment boundary = bbox of visible pods
      let x0 = 1e9, y0 = 1e9, x1 = -1e9, y1 = -1e9;
      for (let k = 0; k < NPOD; k++) {
        const i = POD + k;
        if (k >= 3 && na[i] <= 0) continue;
        const hw = nw[i] / 2, hh = nh[i] / 2;
        if (nx[i] - hw < x0) x0 = nx[i] - hw;
        if (nx[i] + hw > x1) x1 = nx[i] + hw;
        if (ny[i] - hh < y0) y0 = ny[i] - hh;
        if (ny[i] + hh > y1) y1 = ny[i] + hh;
      }
      gL = x0 - GPAD; gR = x1 + GPAD; gT = y0 - GPAD; gB = y1 + GPAD;
      // Ports
      for (let i = 0; i < NN; i++) {
        if (VERT) { inX[i] = outX[i] = nx[i]; inY[i] = ny[i] - nh[i] / 2; outY[i] = ny[i] + nh[i] / 2; }
        else { inY[i] = outY[i] = ny[i]; inX[i] = nx[i] - nw[i] / 2; outX[i] = nx[i] + nw[i] / 2; }
      }
      if (VERT) { gInX = gOutX = (gL + gR) / 2; gInY = gT; gOutY = gB; }
      else { gInY = gOutY = (gT + gB) / 2; gInX = gL; gOutX = gR; }
      for (let k = 0; k < NPOD; k++) {
        const bk = k < 3 ? fB : 1, i = POD + k;
        pinX[k] = lerp(inX[i], gInX, bk); pinY[k] = lerp(inY[i], gInY, bk);
        poutX[k] = lerp(outX[i], gOutX, bk); poutY[k] = lerp(outY[i], gOutY, bk);
      }
    }

    /* ------------------------------------------------------------------ */
    /* Load model (HUD + pod meters)                                       */
    /* ------------------------------------------------------------------ */
    function rpsAt(q) {
      if (q < 0.255) return 0;
      return 140 + 4660 * EZ.inOutSine(range(q, 0.255, 0.44)) + 16900 * EZ.outCubic(range(q, 0.452, 0.515));
    }
    const p95At = (q) => 42 + 5 * EZ.inOutSine(range(q, 0.452, 0.49)) * (1 - EZ.inOutSine(range(q, 0.5, 0.6))) - range(q, 0.58, 0.63);

    /* ------------------------------------------------------------------ */
    /* Curves — bezier with tangents along the flow axis                   */
    /* ------------------------------------------------------------------ */
    let PX = 0, PY = 0;
    function bez(x0, y0, x1, y1, t) {
      let c1x, c1y, c2x, c2y;
      if (VERT) { const d = (y1 - y0) * 0.5; c1x = x0; c1y = y0 + d; c2x = x1; c2y = y1 - d; }
      else { const d = (x1 - x0) * 0.5; c1x = x0 + d; c1y = y0; c2x = x1 - d; c2y = y1; }
      const it = 1 - t, a = it * it * it, b = 3 * it * it * t, c = 3 * it * t * t, d = t * t * t;
      PX = a * x0 + b * c1x + c * c2x + d * x1;
      PY = a * y0 + b * c1y + c * c2y + d * y1;
    }
    function bezC(x0, y0, c1x, c1y, c2x, c2y, x1, y1, t) {
      const it = 1 - t, a = it * it * it, b = 3 * it * it * t, c = 3 * it * t * t, d = t * t * t;
      PX = a * x0 + b * c1x + c * c2x + d * x1;
      PY = a * y0 + b * c1y + c * c2y + d * y1;
    }
    function lin(x0, y0, x1, y1, t) { PX = x0 + (x1 - x0) * t; PY = y0 + (y1 - y0) * t; }
    /** Append a (partial, 0..t) flow curve to the current path. */
    function pathEdge(x0, y0, x1, y1, t) {
      if (t <= 0) return;
      let c1x, c1y, c2x, c2y;
      if (VERT) { const d = (y1 - y0) * 0.5; c1x = x0; c1y = y0 + d; c2x = x1; c2y = y1 - d; }
      else { const d = (x1 - x0) * 0.5; c1x = x0 + d; c1y = y0; c2x = x1 - d; c2y = y1; }
      ctx.moveTo(x0, y0);
      if (t >= 1) { ctx.bezierCurveTo(c1x, c1y, c2x, c2y, x1, y1); return; }
      const ax = x0 + (c1x - x0) * t, ay = y0 + (c1y - y0) * t;
      const bx = c1x + (c2x - c1x) * t, by = c1y + (c2y - c1y) * t;
      const cx = c2x + (x1 - c2x) * t, cy = c2y + (y1 - c2y) * t;
      const dx = ax + (bx - ax) * t, dy = ay + (by - ay) * t;
      const ex = bx + (cx - bx) * t, ey = by + (cy - by) * t;
      ctx.bezierCurveTo(ax, ay, dx, dy, dx + (ex - dx) * t, dy + (ey - dy) * t);
    }
    // Replication link primary → replica (desktop: arc on the right; mobile: arc below)
    let RA0x = 0, RA0y = 0, RA1x = 0, RA1y = 0, RA2x = 0, RA2y = 0, RA3x = 0, RA3y = 0;
    function replArc() {
      if (VERT) {
        const yb = ny[PG] + nh[PG] / 2, yr = ny[REPL] + nh[REPL] / 2, d = 26 * K;
        RA0x = nx[PG]; RA0y = yb; RA1x = nx[PG]; RA1y = yb + d;
        RA2x = nx[REPL]; RA2y = yr + d; RA3x = nx[REPL]; RA3y = yr;
      } else {
        const xr = nx[PG] + nw[PG] / 2, xr2 = nx[REPL] + nw[REPL] / 2, d = 36 * K;
        RA0x = xr; RA0y = ny[PG]; RA1x = xr + d; RA1y = ny[PG];
        RA2x = xr2 + d; RA2y = ny[REPL]; RA3x = xr2; RA3y = ny[REPL];
      }
    }

    /** Packet position at journey parameter u; writes PX/PY. */
    function packetAt(u, r, pod, dest) {
      if (u < S_RAY) bez(RAY_X[r], RAY_Y[r], inX[CDN], inY[CDN], u / S_RAY);
      else if (u < S_CDN) lin(inX[CDN], inY[CDN], outX[CDN], outY[CDN], (u - S_RAY) / (S_CDN - S_RAY));
      else if (u < S_E1) bez(outX[CDN], outY[CDN], inX[LB], inY[LB], (u - S_CDN) / (S_E1 - S_CDN));
      else if (u < S_LB) lin(inX[LB], inY[LB], outX[LB], outY[LB], (u - S_E1) / (S_LB - S_E1));
      else if (u < S_E2) bez(outX[LB], outY[LB], pinX[pod], pinY[pod], (u - S_LB) / (S_E2 - S_LB));
      else if (u < S_APP) {
        const q = (u - S_E2) / (S_APP - S_E2), i = POD + pod;
        if (q < 0.5) lin(pinX[pod], pinY[pod], nx[i], ny[i], q * 2);
        else lin(nx[i], ny[i], poutX[pod], poutY[pod], q * 2 - 1);
      } else if (u < S_E3) bez(poutX[pod], poutY[pod], inX[dest], inY[dest], (u - S_APP) / (S_E3 - S_APP));
      else lin(inX[dest], inY[dest], nx[dest], ny[dest], (u - S_E3) / (1 - S_E3));
    }

    /* ------------------------------------------------------------------ */
    /* Particles (convergence → face)                                      */
    /* ------------------------------------------------------------------ */
    const CAP = 2500;
    const pSX = new Float32Array(CAP), pSY = new Float32Array(CAP);
    const pTX = new Float32Array(CAP), pTY = new Float32Array(CAP), pTV = new Float32Array(CAP);
    const pS0 = new Float32Array(CAP), pS1 = new Float32Array(CAP);
    const pA0 = new Float32Array(CAP), pA1 = new Float32Array(CAP);
    const pD = new Float32Array(CAP), pTh = new Float32Array(CAP), pPh = new Float32Array(CAP);
    const pB = new Uint8Array(CAP);
    let NP = 0, targets = null, tCols = 40, targetsKey = "";
    let img = null, portrait = null;

    const smooth = (a, b, x) => { const t = clamp((x - a) / (b - a)); return t * t * (3 - 2 * t); };
    // Shared U.portraitCanvas radial mask × our elliptical feather, in (u, v) of
    // the 4:5 box — so the particle face and the final portrait share one silhouette.
    const FEATHER = [[0, 1], [0.6, 1], [0.72, 0.84], [0.84, 0.45], [0.93, 0.13], [1, 0]];
    function featherAt(d) {
      if (d >= 1) return 0;
      for (let i = 1; i < FEATHER.length; i++) {
        if (d <= FEATHER[i][0]) {
          const a = FEATHER[i - 1], b = FEATHER[i];
          return a[1] + (b[1] - a[1]) * ((d - a[0]) / (b[0] - a[0]));
        }
      }
      return 0;
    }
    function maskAt(u, v) {
      const d = Math.hypot(u - 0.5, (v - 0.43) * 1.25);
      const t = (d - 0.18) / 0.54;
      const shared = t <= 0 ? 1 : t < 0.62 ? 1 - 0.08 * (t / 0.62) : Math.max(0, 0.92 * (1 - (t - 0.62) / 0.38));
      return shared * featherAt(Math.hypot((u - 0.5) / 0.5, (v - 0.47) / 0.47));
    }

    /** High-quality cover-fit downscale so the coarse sampling grid doesn't alias. */
    function prescale(im, w, h) {
      const c = document.createElement("canvas");
      c.width = w; c.height = h;
      const x = c.getContext("2d");
      x.imageSmoothingEnabled = true;
      x.imageSmoothingQuality = "high";
      U.drawCover(x, im, 0, 0, w, h);
      return c;
    }

    function buildTargets() {
      const cap = MOB ? 1200 : 2500;
      // sampling depends only on the image and the particle budget — a plain
      // resize (e.g. a mobile URL bar) just re-maps sources and targets
      const key = (img ? "img" : "sil") + cap;
      if (targets && key === targetsKey) { buildParticles(); return; }
      targetsKey = key;
      const cols = Math.max(24, Math.round(Math.sqrt(cap / 0.66 / 1.25)));
      const rows = Math.round(cols * 1.25);
      let src = null;
      if (img) { try { src = prescale(img, cols * 4, rows * 4); } catch (e) { src = null; } }
      const pts = U.samplePoints(src, cols, rows, 0, 11); // null src → silhouette fallback
      const out = [];
      for (let i = 0; i < pts.length; i++) {
        const q = pts[i];
        const l = clamp((q.lum - 0.5) * 1.18 + 0.5);
        let I = l < 0.5 ? 0.03 + 0.69 * Math.pow(l * 2, 1.25) : 0.72 + 0.26 * (l - 0.5) * 2;
        I *= Math.pow(maskAt(q.u, q.v), 1.3);
        I = smooth(0.2, 0.96, I);
        if (I < 0.05) continue;
        out.push({ u: q.u, v: q.v, I, l, j: q.j });
      }
      if (out.length > cap) { out.sort((a, b) => b.I - a.I); out.length = cap; }
      targets = out;
      tCols = cols;
      buildParticles();
    }

    function buildParticles() {
      if (!targets || !W) return;
      // Source geometry: the topology as it stands at the start of convergence
      computeNodes(0.66);
      const sx = [], sy = [], ss = [];
      const step = MOB ? 4 : 4.5;
      const addSeg = (x0, y0, x1, y1, size) => {
        const n = Math.max(1, Math.round(Math.hypot(x1 - x0, y1 - y0) / step));
        for (let i = 0; i < n; i++) { sx.push(x0 + (x1 - x0) * (i / n)); sy.push(y0 + (y1 - y0) * (i / n)); ss.push(size); }
      };
      const addRect = (cx, cy, w, h, size) => {
        const l = cx - w / 2, t = cy - h / 2;
        addSeg(l, t, l + w, t, size); addSeg(l + w, t, l + w, t + h, size);
        addSeg(l + w, t + h, l, t + h, size); addSeg(l, t + h, l, t, size);
      };
      const addCurve = (x0, y0, x1, y1, size) => {
        const n = Math.max(2, Math.round(Math.hypot(x1 - x0, y1 - y0) / step));
        for (let i = 0; i <= n; i++) { bez(x0, y0, x1, y1, i / n); sx.push(PX); sy.push(PY); ss.push(size); }
      };
      for (let i = 0; i < NN; i++) addRect(nx[i], ny[i], nw[i], nh[i], 1);
      addRect((gL + gR) / 2, (gT + gB) / 2, gR - gL, gB - gT, 0.8);
      addCurve(outX[CDN], outY[CDN], inX[LB], inY[LB], 0.8);
      addCurve(outX[LB], outY[LB], gInX, gInY, 0.8);
      addCurve(gOutX, gOutY, inX[REDIS], inY[REDIS], 0.8);
      addCurve(gOutX, gOutY, inX[PG], inY[PG], 0.8);
      for (let r = 0; r < NRAYS; r++) addCurve(RAY_X[r], RAY_Y[r], inX[CDN], inY[CDN], 0.6);
      // Sample N source points, then map sources → targets coherently: both sets are
      // cut into the same number of x-strips (sorted by x / u) and each strip is
      // sorted by y / v. The diagram folds into the face as one smooth wave
      // instead of a random swarm.
      const pool = sx.length;
      const rnd = U.rng(1234);
      const n = targets.length;
      NP = n;
      const srcIdx = new Array(n), srcK = new Int32Array(n), jx = new Float32Array(n), jy = new Float32Array(n);
      for (let i = 0; i < n; i++) {
        srcK[i] = (rnd() * pool) | 0;
        jx[i] = sx[srcK[i]] + (rnd() - 0.5) * 2;
        jy[i] = sy[srcK[i]] + (rnd() - 0.5) * 2;
        srcIdx[i] = i;
      }
      const tgtIdx = new Array(n);
      for (let i = 0; i < n; i++) tgtIdx[i] = i;
      const strips = MOB ? 26 : 38;
      // vertical flow (mobile) → strips across y instead
      const sMajor = (i) => (VERT ? jy[i] : jx[i]), sMinor = (i) => (VERT ? jx[i] : jy[i]);
      const tMajor = (i) => (VERT ? targets[i].v : targets[i].u), tMinor = (i) => (VERT ? targets[i].u : targets[i].v);
      srcIdx.sort((a, b) => sMajor(a) - sMajor(b));
      tgtIdx.sort((a, b) => tMajor(a) - tMajor(b));
      for (let st = 0; st < strips; st++) {
        const a = Math.floor((st * n) / strips), b = Math.floor(((st + 1) * n) / strips);
        const sl = srcIdx.slice(a, b).sort((x, y) => sMinor(x) - sMinor(y));
        const tl = tgtIdx.slice(a, b).sort((x, y) => tMinor(x) - tMinor(y));
        for (let m = 0; m < sl.length; m++) { srcIdx[a + m] = sl[m]; tgtIdx[a + m] = tl[m]; }
      }
      const cell = BOX.w / tCols;
      for (let m = 0; m < n; m++) {
        const i = m, si = srcIdx[m], q = targets[tgtIdx[m]], k = srcK[si];
        const packet = rnd() < 0.06; // a sprinkle of bright "packet" particles
        pSX[i] = jx[si];
        pSY[i] = jy[si];
        pS0[i] = (packet ? 5.5 : 2.6) * ss[k];
        pA0[i] = packet ? 0.9 : 0.42;
        pTX[i] = BOX.x + q.u * BOX.w;
        pTY[i] = BOX.y + q.v * BOX.h;
        pTV[i] = q.v;
        const dd = cell * (0.2 + 0.8 * q.I);
        pS1[i] = dd / DOT_VIS;
        pA1[i] = 0.3 + 0.7 * q.I;
        pB[i] = Math.round(clamp((q.l - 0.5) / 0.5) * (NB - 1));
        // Right-hand (desktop) / lower (mobile) particles leave first so the
        // person-card side clears early; a little noise keeps it organic.
        const lead = MOB ? pSY[i] / H : pSX[i] / W;
        pD[i] = 0.12 * rnd() + 0.34 * clamp(1 - lead);
        pTh[i] = 0.12 + 0.22 * rnd();
        pPh[i] = rnd() * TAU;
      }
    }

    /* ------------------------------------------------------------------ */
    /* Drawing helpers                                                     */
    /* ------------------------------------------------------------------ */
    const LED_MAX = 64;
    const ledX = new Float32Array(LED_MAX), ledY = new Float32Array(LED_MAX), ledA = new Float32Array(LED_MAX);
    let ledN = 0;
    function led(x, y, on, a, r) {
      ctx.beginPath();
      ctx.arc(x, y, r, 0, TAU);
      if (on > 0.01) {
        ctx.globalAlpha = a * (0.25 + 0.75 * on);
        ctx.fillStyle = COL.acc;
        ctx.fill();
        if (ledN < LED_MAX) { ledX[ledN] = x; ledY[ledN] = y; ledA[ledN] = a * on; ledN++; }
      } else {
        ctx.globalAlpha = a * 0.16;
        ctx.fillStyle = COL.white;
        ctx.fill();
      }
    }
    function flushLeds(size) {
      if (!ledN) return;
      ctx.globalCompositeOperation = "lighter";
      for (let i = 0; i < ledN; i++) {
        ctx.globalAlpha = ledA[i] * 0.55;
        ctx.drawImage(SPR_GLOW, ledX[i] - size / 2, ledY[i] - size / 2, size, size);
      }
      ctx.globalCompositeOperation = "source-over";
      ledN = 0;
    }
    function spaced(on) { if (HAS_LS) ctx.letterSpacing = on ? "0.12em" : "0px"; }

    /* ---------------- Floor: perspective dot grid with a gentle dolly ---------------- */
    // Camera height = 1; rows sit every 1/FLOOR_CAM of that in depth and columns
    // use the same spacing, so cells are square on the ground plane. Rows fade
    // into fog well before the horizon (~1k fillRects, no allocation).
    const FLOOR_ROWS = 26;
    function drawFloor(a, p) {
      if (a <= 0.004) return;
      ctx.fillStyle = COL.floor;
      const yH = FLOOR_Y, D = (H - yH) * 1.1, cx = FLOOR_CX, c = 1 / FLOOR_CAM;
      const zMax = 1 + FLOOR_ROWS * c;
      const shift = (p * 9) % 1; // dolly toward the viewer as you scroll
      const halfW = W * 0.5;
      for (let r = 0; r < FLOOR_ROWS; r++) {
        const z = 1 + (r - shift) * c;
        if (z < 0.9) continue;
        const sy = yH + D / z;
        if (sy > H + 3) continue;
        const far = 1 - (z - 1) / (zMax - 1);
        const ra = a * 0.34 * Math.pow(far, 1.5) * Math.min(1, (z - 0.9) / 0.15);
        if (ra < 0.004) continue;
        const sp = FLOOR_SP / z;
        const size = Math.max(1, 2.1 / Math.sqrt(z));
        const nHalf = Math.min(Math.ceil(halfW / sp) + 1, 300);
        for (let k = -nHalf; k <= nHalf; k++) {
          const sx = cx + k * sp;
          if (sx < -3 || sx > W + 3) continue;
          const dn = (sx - cx) / (halfW * 1.15);
          const ca = ra * (1 - dn * dn);
          if (ca <= 0.004) continue;
          ctx.globalAlpha = ca;
          ctx.fillRect(sx - size * 0.5, sy - size * 0.5, size, size);
        }
      }
    }

    /* ---------------- Beat 1: the server ---------------- */
    function drawServer(p, t) {
      const m = fM;
      if (m >= 1) return;
      const cx = lerp(SRV_X, L0x[POD + 1], m), cy = lerp(SRV_Y, L0y[POD + 1], m);
      const w = lerp(SRV_W, POD_W, m), h = lerp(SRV_H, POD_H, m);
      const boot = EZ.outCubic(range(p, 0.03, 0.085));
      const a = (0.3 + 0.7 * boot) * (1 - m);
      const det = a * (1 - range(m, 0, 0.6));
      const x = snap(cx - w / 2), y = snap(cy - h / 2);

      // aura + floor reflection once healthy
      const live = range(p, 0.15, 0.19) * (1 - m);
      if (live > 0) {
        ctx.globalCompositeOperation = "lighter";
        ctx.globalAlpha = 0.07 * live;
        const s = SRV_W * 2.4;
        ctx.drawImage(SPR_HALO, cx - s / 2, cy - s / 2, s, s);
        ctx.globalAlpha = 0.16 * live;
        ctx.drawImage(SPR_HALO, cx - s * 0.6, cy + h / 2 + 6 * K - s * 0.06, s * 1.2, s * 0.12);
        ctx.globalCompositeOperation = "source-over";
      }

      ctx.save();
      ctx.translate(x, y);
      U.roundRect(ctx, 0, 0, w, h, lerp(8 * K, RAD, m));
      ctx.globalAlpha = a;
      ctx.fillStyle = SRV_GRAD;
      ctx.fill();
      ctx.lineWidth = HAIR;
      ctx.strokeStyle = COL.white;
      ctx.globalAlpha = a * (0.14 + 0.1 * boot);
      ctx.stroke();
      if (live > 0) { ctx.strokeStyle = COL.acc; ctx.globalAlpha = 0.35 * live; ctx.stroke(); }

      if (det > 0.01) {
        const bay = h / 2;
        // bevel + divider
        ctx.fillStyle = COL.white;
        ctx.globalAlpha = det * 0.07;
        ctx.fillRect(8 * K, px(bay), w - 16 * K, HAIR);
        ctx.globalAlpha = det * 0.09;
        ctx.fillRect(RAD, HAIR, w - 2 * RAD, HAIR);
        const ledR = 1.7 * K;
        const tb = REDUCED ? 0 : t;
        for (let b = 0; b < 2; b++) {
          const by = bay * b + bay / 2;
          // three status LEDs
          for (let l = 0; l < 3; l++) {
            let on = 0;
            const lx = 13 * K + l * 8 * K;
            if (b === 0) {
              if (l === 0) on = range(p, 0.075, 0.08);
              else if (l === 1) on = range(p, 0.08, 0.085) * (REDUCED ? 1 : (Math.sin(tb * 9.1) > -0.4 ? 1 : 0.35));
              else on = range(p, 0.085, 0.09) * (p < 0.13 ? (REDUCED ? 1 : (Math.sin(tb * 23 + 1) > 0 ? 1 : 0.3)) : 1);
            } else {
              if (l === 0) on = range(p, 0.137, 0.142);
              else if (l === 1) on = range(p, 0.152, 0.157);
              else on = range(p, 0.165, 0.17);
            }
            led(lx, by, on, det, ledR);
          }
          // label
          ctx.font = F.srv;
          ctx.textBaseline = "middle";
          ctx.textAlign = "left";
          ctx.fillStyle = COL.muted;
          ctx.globalAlpha = det * 0.75;
          ctx.fillText(b === 0 ? "prod-01" : "docker", px(42 * K), px(by + 0.5));
          if (b === 0) {
            // vent grille
            ctx.fillStyle = COL.white;
            ctx.globalAlpha = det * 0.09;
            const vh = bay * 0.42;
            for (let v = 0; v < 11; v++) ctx.fillRect(px(w - 14 * K - v * 4 * K), px(by - vh / 2), HAIR, px(vh));
          } else {
            // drive caddies with activity lights
            for (let d = 0; d < 4; d++) {
              const dx = w - 14 * K - (d + 1) * 19 * K, dw = 16 * K, dh = bay * 0.46;
              ctx.globalAlpha = det * 0.1;
              ctx.strokeStyle = COL.white;
              ctx.strokeRect(snap(dx), snap(by - dh / 2), px(dw), px(dh));
              const act = p > 0.085 ? (REDUCED ? 0.6 : 0.5 + 0.5 * Math.sin(tb * (5 + d * 2.3) + d * 1.7)) : 0;
              if (act > 0) {
                ctx.globalAlpha = det * act * (p < 0.13 ? 0.9 : 0.45);
                ctx.fillStyle = COL.acc;
                ctx.fillRect(px(dx + dw - 4 * K), px(by - 0.75), px(2 * K), px(1.5));
                ctx.fillStyle = COL.white;
              }
            }
          }
        }
      }
      ctx.restore();
      ctx.save();
      flushLeds(11 * K);
      ctx.restore();

      // tag above the chassis
      const ta = det * boot * (1 - range(m, 0, 0.2));
      if (ta > 0.01) {
        ctx.font = F.tag;
        ctx.textAlign = "left";
        ctx.textBaseline = "alphabetic";
        spaced(true);
        ctx.fillStyle = COL.dim;
        ctx.globalAlpha = ta;
        ctx.fillText("PROD-01 · EU-CENTRAL", px(x), px(y - 9 * K));
        const ok = range(p, 0.165, 0.175);
        if (ok > 0) {
          ctx.textAlign = "right";
          ctx.fillStyle = COL.acc;
          ctx.globalAlpha = ta * ok;
          ctx.fillText("HEALTHY", px(x + w), px(y - 9 * K));
        }
        spaced(false);
      }
    }

    function drawLog(p, t) {
      const out = range(p, 0.172, 0.2);
      const a = 1 - out;
      if (a <= 0 || p < 0) return;
      const dy = -12 * K * EZ.inCubic(out);
      ctx.font = F.log;
      ctx.textAlign = "left";
      ctx.textBaseline = "middle";
      const cw = ctx.measureText("0").width;
      const x = px(LOG_X);
      const blink = REDUCED ? 1 : (Math.floor(t * 1.9) % 2 === 0 ? 1 : 0.15);
      let y = LOG_Y + dy;

      // 1 — the command
      const n = Math.min(CMD.length, Math.floor(CMD.length * range(p, 0.02, 0.07) + 1e-4));
      ctx.globalAlpha = a * 0.95;
      ctx.fillStyle = COL.acc;
      ctx.fillText("$", x, px(y));
      ctx.fillStyle = COL.text;
      ctx.fillText(CMD_PREFIX[n], x + 2 * cw, px(y));
      if (p < 0.08) {
        ctx.globalAlpha = a * 0.85 * (n < CMD.length && p > 0.02 ? 1 : blink);
        ctx.fillStyle = COL.acc;
        ctx.fillRect(px(x + (2 + n) * cw + 1), px(y - LOG_LH * 0.32), px(cw * 0.62), px(LOG_LH * 0.64));
      }

      // 2 — build progress
      y += LOG_LH;
      const l2 = range(p, 0.078, 0.09);
      if (l2 > 0) {
        const bp = range(p, 0.085, 0.13);
        ctx.globalAlpha = a * l2;
        ctx.fillStyle = COL.muted;
        ctx.fillText(bp < 1 ? "building" : "built   ", x + 2 * cw, px(y));
        const bx = x + 11 * cw, bw = 9 * cw;
        ctx.fillStyle = COL.white;
        ctx.globalAlpha = a * l2 * 0.12;
        ctx.fillRect(px(bx), px(y - 1), px(bw), px(2));
        ctx.fillStyle = COL.acc;
        ctx.globalAlpha = a * l2 * 0.9;
        ctx.fillRect(px(bx), px(y - 1), px(bw * bp), px(2));
        if (bp > 0 && bp < 1) {
          ctx.globalCompositeOperation = "lighter";
          ctx.globalAlpha = a * 0.8;
          const s = 16 * K;
          ctx.drawImage(SPR_GLOW, bx + bw * bp - s / 2, y - s / 2, s, s);
          ctx.globalCompositeOperation = "source-over";
        }
        ctx.fillStyle = bp < 1 ? COL.muted : COL.text;
        ctx.globalAlpha = a * l2 * 0.85;
        ctx.fillText(PCT[Math.round(bp * 100)], bx + bw + cw, px(y));
      }

      // 3, 4 — services ready
      for (let i = 0; i < 2; i++) {
        y += LOG_LH;
        const r0 = i === 0 ? 0.137 : 0.152;
        const la = range(p, r0, r0 + 0.008);
        if (la <= 0) continue;
        const ox = (1 - EZ.outCubic(la)) * -6 * K;
        ctx.globalAlpha = a * la;
        ctx.fillStyle = COL.acc;
        ctx.beginPath();
        ctx.arc(x + 2 * cw + cw * 0.3 + ox, y, 2.2 * K, 0, TAU);
        ctx.fill();
        ctx.fillStyle = COL.text;
        ctx.fillText(i === 0 ? "db" : "web", x + 4 * cw + ox, px(y));
        ctx.fillStyle = COL.muted;
        ctx.fillText(i === 0 ? "· ready" : "· ready", x + (i === 0 ? 7 : 8) * cw + ox, px(y));
        ctx.fillStyle = COL.dim;
        ctx.fillText(i === 0 ? "1.24s" : "1.87s", x + 21 * cw + ox, px(y));
      }

      // 5 — prompt again
      y += LOG_LH;
      const l5 = range(p, 0.16, 0.165);
      if (l5 > 0) {
        ctx.globalAlpha = a * l5;
        ctx.fillStyle = COL.acc;
        ctx.fillText("$", x, px(y));
        ctx.globalAlpha = a * l5 * 0.85 * blink;
        ctx.fillRect(px(x + 2 * cw + 1), px(y - LOG_LH * 0.32), px(cw * 0.62), px(LOG_LH * 0.64));
      }
    }

    /* ---------------- Beats 2–3: topology ---------------- */
    let hudReqV = -1, hudReqS = "", hudP95V = -1, hudP95S = "", hudPodsKey = -1, hudPodsS = "";
    function drawTopology(p, t, ta) {
      const tt = REDUCED ? 0 : t;
      const rps = rpsAt(p);
      const surge = range(p, 0.452, 0.52);

      /* edges */
      ctx.lineWidth = HAIR;
      ctx.strokeStyle = COL.white;
      const eA = (0.13 + 0.05 * surge) * ta;
      // incoming request rays
      const er = range(p, 0.25, 0.29);
      if (er > 0) {
        ctx.strokeStyle = RAY_GRAD;
        ctx.globalAlpha = eA * 0.7 * na[CDN];
        ctx.beginPath();
        for (let r = 0; r < NRAYS; r++) pathEdge(RAY_X[r], RAY_Y[r], inX[CDN], inY[CDN], er);
        ctx.stroke();
        ctx.strokeStyle = COL.white;
      }
      ctx.globalAlpha = eA;
      ctx.beginPath();
      pathEdge(outX[CDN], outY[CDN], inX[LB], inY[LB], range(p, 0.236, 0.266));
      ctx.stroke();
      // lb → app and app → data, bundling into the deployment's ports
      for (let k = 0; k < 3; k++) {
        const wk = k === 1 ? 1 : 1 - fB;
        if (wk <= 0.002) continue;
        ctx.globalAlpha = eA * wk * Math.min(na[POD + k], 1);
        ctx.beginPath();
        pathEdge(outX[LB], outY[LB], pinX[k], pinY[k], range(p, 0.226 + k * 0.004, 0.256 + k * 0.004));
        pathEdge(poutX[k], poutY[k], inX[REDIS], inY[REDIS], range(p, 0.24, 0.27));
        pathEdge(poutX[k], poutY[k], inX[PG], inY[PG], range(p, 0.246, 0.276));
        ctx.stroke();
      }
      // replication
      const rp = range(p, 0.262, 0.29);
      replArc();
      if (rp > 0) {
        ctx.setLineDash(DASH);
        ctx.lineDashOffset = -tt * 10;
        ctx.globalAlpha = eA * 1.2 * rp;
        ctx.beginPath();
        ctx.moveTo(RA0x, RA0y);
        ctx.bezierCurveTo(RA1x, RA1y, RA2x, RA2y, RA3x, RA3y);
        ctx.stroke();
        ctx.setLineDash(NO_DASH);
      }

      /* deployment boundary fill (under the packets) */
      const gA = EZ.outCubic(range(p, 0.445, 0.48)) * ta;
      if (gA > 0) {
        U.roundRect(ctx, snap(gL), snap(gT), px(gR - gL), px(gB - gT), 11 * K);
        ctx.fillStyle = COL.acc;
        ctx.globalAlpha = 0.035 * gA;
        ctx.fill();
      }

      /* packets */
      const nActF = PK_N * (0.24 * EZ.outCubic(range(p, 0.268, 0.43)) + 0.76 * EZ.outCubic(range(p, 0.452, 0.51)));
      const nAct = Math.min(PK_N, Math.ceil(nActF));
      if (nAct > 0) {
        ctx.globalCompositeOperation = "lighter";
        // dense traffic → smaller, dimmer packets and shorter tails so the
        // stream stays a stream of discrete packets, not a neon tube
        const ps = (12 - 2.5 * surge) * K;
        const headA = 0.95 - 0.3 * surge, tailA = 0.42 * (1 - 0.6 * surge), tail2A = 0.18 * (1 - 0.8 * surge);
        for (let j = 0; j < nAct; j++) {
          const aj = clamp((nActF - j) / 5) * ta;
          if (aj <= 0) continue;
          const cyc = tt * PK_RATE[j] + PK_PH[j];
          const ci = Math.floor(cyc), u = cyc - ci;
          const h = hash2(j, ci);
          const pod = ((h & 1023) * nReady) >> 10;
          const dest = ((h >>> 10) & 1023) < 720 ? REDIS : PG;
          const r = PK_RAY[j] % NRAYS;
          if (u >= S_E2 && u < S_APP) nfl[POD + pod] += 0.4;
          else if (u >= S_RAY && u < S_CDN) nfl[CDN] += 0.06;
          else if (u >= S_E1 && u < S_LB) nfl[LB] += 0.06;
          else if (u >= S_E3) nfl[dest] += 0.2;
          for (let s = 0; s < 3; s++) {
            const us = u - s * 0.009;
            if (us < 0) break;
            packetAt(us, r, pod, dest);
            const vis = us < 0.03 ? us / 0.03 : us > S_E3 ? 1 - (us - S_E3) / (1 - S_E3) : 1;
            const sa = s === 0 ? headA : s === 1 ? tailA : tail2A;
            const sz = ps * (s === 0 ? 1 : s === 1 ? 0.78 : 0.6);
            ctx.globalAlpha = aj * vis * sa;
            ctx.drawImage(SPR_GLOW, PX - sz / 2, PY - sz / 2, sz, sz);
          }
        }
        // replication packets primary → replica
        if (rp > 0 && p > 0.29) {
          for (let j = 0; j < 3; j++) {
            const u = (tt / 2.6 + j / 3 + 0.17) % 1;
            bezC(RA0x, RA0y, RA1x, RA1y, RA2x, RA2y, RA3x, RA3y, u);
            ctx.globalAlpha = 0.6 * ta * Math.sin(Math.PI * u);
            const sz = 10 * K;
            ctx.drawImage(SPR_GLOW, PX - sz / 2, PY - sz / 2, sz, sz);
          }
        }
        ctx.globalCompositeOperation = "source-over";
      }

      /* deployment boundary (dashed) + label */
      if (gA > 0) {
        ctx.setLineDash(DASH);
        ctx.lineDashOffset = -tt * 8;
        U.roundRect(ctx, snap(gL), snap(gT), px(gR - gL), px(gB - gT), 11 * K);
        ctx.strokeStyle = COL.acc;
        ctx.lineWidth = HAIR;
        ctx.globalAlpha = 0.42 * gA;
        ctx.stroke();
        ctx.setLineDash(NO_DASH);
      }

      /* cards — spawned pods first so they slide out from behind the originals */
      ctx.lineWidth = HAIR;
      const baseLoad = Math.min(0.97, rps / (Math.max(3, nReadyF) * 3200));
      for (let k = NPOD - 1; k >= 0; k--) drawCard(POD + k, na[POD + k] * ta, true, baseLoad, k);
      for (let i = 0; i < POD; i++) drawCard(i, na[i] * ta, false, 0, 0);
      ctx.save();
      flushLeds(10 * K);
      ctx.restore();

      /* tags above infra cards */
      ctx.font = F.tag;
      ctx.textBaseline = "alphabetic";
      spaced(true);
      ctx.fillStyle = COL.dim;
      ctx.textAlign = "left";
      const tags = MOB ? TAG_M : TAG;
      for (let i = 0; i < POD; i++) {
        const a = na[i] * ta * range(na[i], 0.6, 1);
        if (a <= 0.01) continue;
        ctx.globalAlpha = a;
        ctx.fillText(tags[i], px(nx[i] - nw[i] / 2), px(ny[i] - nh[i] / 2 - 7 * K));
      }
      // "APP" tag in beat 2 → replaced by the deployment label in beat 3
      const appA = na[POD] * ta * (1 - gA / Math.max(ta, 1e-3)) * range(p, 0.24, 0.26);
      if (appA > 0.01) {
        ctx.globalAlpha = appA;
        ctx.fillText("APP", px(nx[POD] - nw[POD] / 2), px(ny[POD] - nh[POD] / 2 - 7 * K));
      }
      spaced(false);

      if (gA > 0) drawGroupLabel(p, gA);
      const hudA = range(p, 0.255, 0.29) * (1 - range(p, 0.635, 0.672));
      if (hudA > 0) drawHUD(p, tt, hudA, rps);
    }

    function drawCard(i, a, isPod, baseLoad, k) {
      if (a <= 0.004) return;
      const w = nw[i], h = nh[i];
      const L = snap(nx[i] - w / 2), T = snap(ny[i] - h / 2);
      const fl = Math.min(1, nfl[i]);
      U.roundRect(ctx, L, T, w, h, RAD);
      ctx.globalAlpha = a * 0.9;
      ctx.fillStyle = COL.card;
      ctx.fill();
      ctx.strokeStyle = COL.white;
      ctx.globalAlpha = a * (0.15 + 0.08 * fl);
      ctx.stroke();
      if (fl > 0.02) {
        ctx.strokeStyle = COL.acc;
        ctx.globalAlpha = a * fl * 0.45;
        ctx.stroke();
      }
      const cy = ny[i];
      led(L + 10 * K, cy, 0.55 + 0.45 * fl, a, 1.8 * K);
      // text only once the card has mostly arrived (keeps the unfold/spawn clean)
      a *= range(na[i], 0.55, 1);
      if (a <= 0.01) return;
      ctx.font = F.label;
      ctx.textBaseline = "middle";
      ctx.textAlign = "left";
      ctx.fillStyle = COL.text;
      ctx.globalAlpha = a * 0.9;
      ctx.fillText(LABEL[i], px(L + 18 * K), px(cy + 0.5));
      if (isPod) {
        // load meter — newly scheduled pods ramp up as they become ready
        const ld = clamp((baseLoad + POD_NOISE[k]) * EZ.inOutSine(range(nsp[i], 0.4, 1)));
        if (!MOB) {
          ctx.textAlign = "right";
          ctx.fillStyle = COL.muted;
          ctx.globalAlpha = a * 0.75;
          ctx.fillText(PCT[Math.round(ld * 100)], px(L + w - 9 * K), px(cy + 0.5));
        }
        else {
          const bx = L + 7 * K, bw = w - 14 * K, by = T + h - 4 * K;
          ctx.fillStyle = COL.white;
          ctx.globalAlpha = a * 0.1;
          ctx.fillRect(px(bx), px(by), px(bw), HAIR);
          ctx.fillStyle = COL.acc;
          ctx.globalAlpha = a * 0.65;
          ctx.fillRect(px(bx), px(by), px(bw * ld), HAIR);
        }
      }
    }

    function drawGroupLabel(p, a) {
      ctx.font = F.group;
      ctx.textBaseline = "alphabetic";
      ctx.textAlign = "left";
      const cw = ctx.measureText("0").width;
      const x = px(gL + 2 * K), y = px(gT - 8 * K);
      ctx.globalAlpha = a;
      ctx.fillStyle = COL.acc;
      ctx.fillText("k8s", x, y);
      ctx.fillStyle = COL.muted;
      const mid = MOB ? " · replicas " : " · deployment/web · replicas ";
      ctx.fillText(mid, x + 3 * cw, y);
      const nx0 = x + (3 + mid.length) * cw;
      // flash the count as it ticks
      let tick = 0;
      for (let k = 3; k < NPOD; k++) { const s = nsp[POD + k]; if (s >= 0.62 && s < 1) tick = Math.max(tick, 1 - range(s, 0.62, 1)); }
      ctx.fillStyle = tick > 0.05 ? COL.acc : COL.text;
      ctx.fillText(NUM[nReady], nx0, y);
    }

    function fmtReq(v) {
      if (MOB) return v >= 1000 ? (v / 1000).toFixed(1) + "k" : String(v);
      return v.toLocaleString("en-US");
    }

    function drawHUD(p, t, a, rps) {
      // values (strings rebuilt only when they change)
      const jitter = REDUCED ? 1 : 1 + 0.012 * Math.sin(t * 2.3) + 0.008 * Math.sin(t * 5.1 + 1);
      const req = Math.round((rps * jitter) / 10) * 10;
      if (req !== hudReqV) { hudReqV = req; hudReqS = fmtReq(req); }
      const p95 = Math.round(p95At(p) + (REDUCED ? 0 : 0.6 * Math.sin(t * 1.3)));
      if (p95 !== hudP95V) { hudP95V = p95; hudP95S = p95 + "ms"; }
      const desired = p < 0.452 ? 3 : 12;
      const key = nReady * 100 + desired;
      if (key !== hudPodsKey) { hudPodsKey = key; hudPodsS = nReady + "/" + desired; }

      ctx.textBaseline = "middle";
      ctx.textAlign = "left";
      ctx.font = F.hud;
      const cwL = ctx.measureText("0").width;
      ctx.font = F.hudS;
      const cwV = ctx.measureText("0").width;
      const gap = (MOB ? 16 : 30) * K, ig = 7 * K;
      const sparkW = MOB ? 0 : 56 * K;
      // fixed-width slots so nothing jitters
      const wLive = MOB ? 0 : 12 * K + 4 * cwL;
      const wReq = 5 * cwL + ig + 6 * cwV + (sparkW ? 10 * K + sparkW : 0);
      const wP95 = 3 * cwL + ig + 4 * cwV;
      const wPods = 4 * cwL + ig + 5 * cwV;
      const wErr = MOB ? 0 : 3 * cwL + ig + 5 * cwV;
      const total = wLive + wReq + wP95 + wPods + wErr + gap * (MOB ? 2 : 4);
      let x = HUD_X - total / 2;
      const y = px(HUD_Y);

      ctx.globalAlpha = a;
      if (!MOB) {
        ctx.globalCompositeOperation = "lighter";
        const s = 12 * K;
        ctx.globalAlpha = a * (REDUCED ? 0.9 : 0.6 + 0.4 * Math.sin(t * 3));
        ctx.drawImage(SPR_GREEN, x + 3 * K - s / 2, y - s / 2, s, s);
        ctx.globalCompositeOperation = "source-over";
        ctx.globalAlpha = a;
        ctx.font = F.hud;
        spaced(true);
        ctx.fillStyle = COL.muted;
        ctx.fillText("LIVE", px(x + 12 * K), y);
        spaced(false);
        x += wLive + gap;
        sep(x - gap / 2, y, a);
      }
      // req/s
      hudPair("REQ/S", 5, hudReqS, 6, COL.text, x, y, a, cwL, cwV, ig);
      if (sparkW) {
        const sx = x + 5 * cwL + ig + 6 * cwV + 10 * K, sh = 12 * K;
        ctx.strokeStyle = COL.acc;
        ctx.lineWidth = 1;
        ctx.globalAlpha = a * 0.8;
        ctx.beginPath();
        const N = 24;
        for (let i = 0; i < N; i++) {
          const q = p - 0.26 + (0.26 * i) / (N - 1);
          let v = rpsAt(q) / 26000;
          v *= 1 + 0.025 * Math.sin(i * 1.9 + t * 1.7);
          const xx = sx + (sparkW * i) / (N - 1), yy = y + sh / 2 - clamp(v) * sh;
          if (i === 0) ctx.moveTo(xx, yy); else ctx.lineTo(xx, yy);
        }
        ctx.stroke();
      }
      x += wReq + gap; sep(x - gap / 2, y, a);
      hudPair("P95", 3, hudP95S, 4, COL.green, x, y, a, cwL, cwV, ig);
      x += wP95 + gap; sep(x - gap / 2, y, a);
      hudPair("PODS", 4, hudPodsS, 5, COL.text, x, y, a, cwL, cwV, ig);
      if (!MOB) {
        x += wPods + gap; sep(x - gap / 2, y, a);
        hudPair("ERR", 3, "0.00%", 5, COL.text, x, y, a, cwL, cwV, ig);
      }
    }
    function hudPair(label, ll, val, vl, col, x, y, a, cwL, cwV, ig) {
      ctx.font = F.hud;
      ctx.fillStyle = COL.dim;
      ctx.globalAlpha = a;
      ctx.textAlign = "left";
      ctx.fillText(label, px(x), y);
      ctx.font = F.hudS;
      ctx.fillStyle = col;
      ctx.textAlign = "right";
      ctx.fillText(val, px(x + ll * cwL + ig + vl * cwV), y);
      ctx.textAlign = "left";
    }
    function sep(x, y, a) {
      ctx.fillStyle = COL.white;
      ctx.globalAlpha = a * 0.12;
      ctx.fillRect(snap(x), px(y - 6 * K), HAIR, px(12 * K));
    }

    /* ---------------- Beats 4–5: particles → portrait ---------------- */
    function drawParticles(p, t, fadeOut) {
      const fin = range(p, 0.655, 0.685);
      const ga = fin * fadeOut;
      if (ga <= 0.003 || !NP) return;
      const c = range(p, 0.665, 0.8);
      const INV = 1 / 0.54;
      const scan = range(p, 0.785, 0.86);
      const band = -0.15 + 1.3 * scan;
      const scanOn = scan > 0 && scan < 1;
      const tt = REDUCED ? 0 : t;
      ctx.globalCompositeOperation = "lighter";
      for (let i = 0; i < NP; i++) {
        let lt = (c - pD[i]) * INV;
        lt = lt <= 0 ? 0 : lt >= 1 ? 1 : lt;
        const e = lt < 0.5 ? 4 * lt * lt * lt : 1 - Math.pow(-2 * lt + 2, 3) / 2;
        let x, y;
        if (e >= 1) { x = pTX[i]; y = pTY[i]; }
        else {
          const ie = 1 - e, ang = pTh[i] * ie;
          const cs = Math.cos(ang), sn = Math.sin(ang);
          const dx = (pSX[i] - pTX[i]) * ie, dy = (pSY[i] - pTY[i]) * ie;
          x = pTX[i] + dx * cs - dy * sn;
          y = pTY[i] + dx * sn + dy * cs;
        }
        let s = pS0[i] + (pS1[i] - pS0[i]) * e;
        let a = pA0[i] + (pA1[i] - pA0[i]) * e;
        if (e >= 1) a *= 0.86 + 0.14 * Math.sin(tt * 1.7 + pPh[i]);
        if (scanOn) {
          const d = (pTV[i] - band) * 9;
          if (d > -1 && d < 1) { const b = (1 - d * d); s *= 1 + 0.35 * b * b; a += 0.35 * b * b; }
        }
        ctx.globalAlpha = a > 1 ? ga : a * ga;
        ctx.drawImage(SPR_DOT[pB[i]], x - s * 0.5, y - s * 0.5, s, s);
      }
      ctx.globalCompositeOperation = "source-over";
    }

    function drawPortrait(p, t) {
      const pa = EZ.inOutSine(range(p, 0.85, 0.94));
      if (portrait && pa > 0) {
        ctx.globalAlpha = pa;
        ctx.drawImage(portrait, BOX.x, BOX.y, BOX.w, BOX.h);
      }
      // a few packets in slow orbit, kept clear of the face
      const oa = range(p, 0.88, 0.96);
      if (oa <= 0) return;
      const tt = REDUCED ? 0 : t;
      const cx = BOX.cx, cy = BOX.cy + BOX.h * 0.02, rx = BOX.w * 0.64, ry = BOX.h * 0.57, tilt = -0.2;
      const ct = Math.cos(tilt), st = Math.sin(tilt);
      ctx.strokeStyle = COL.acc;
      ctx.lineWidth = HAIR;
      ctx.globalAlpha = 0.09 * oa;
      ctx.beginPath();
      ctx.ellipse(cx, cy, rx, ry, tilt, 0, TAU);
      ctx.stroke();
      ctx.globalCompositeOperation = "lighter";
      const sz = 12 * K;
      for (let j = 0; j < 4; j++) {
        const base = tt * 0.22 + j * (TAU / 4) + 0.6;
        for (let s = 0; s < 6; s++) {
          const th = base - s * 0.035;
          const ex = Math.cos(th) * rx, ey = Math.sin(th) * ry;
          const x = cx + ex * ct - ey * st, y = cy + ex * st + ey * ct;
          ctx.globalAlpha = oa * (s === 0 ? 0.85 : 0.32 * (1 - s / 6));
          const ss = sz * (s === 0 ? 1 : 0.7);
          ctx.drawImage(SPR_GLOW, x - ss / 2, y - ss / 2, ss, ss);
        }
      }
      ctx.globalCompositeOperation = "source-over";
    }

    /* ------------------------------------------------------------------ */
    /* Portrait loading                                                    */
    /* ------------------------------------------------------------------ */
    /**
     * The shared tri-tone mask stays ~0.9 opaque at the box edges, which reads
     * as a hard rectangle on the dark page. Feather it once into an ellipse
     * that reaches 0 at every edge (offscreen, one-time cost).
     */
    function feather(src) {
      const w = src.width, h = src.height;
      const c = document.createElement("canvas");
      c.width = w; c.height = h;
      const x = c.getContext("2d");
      x.drawImage(src, 0, 0);
      x.globalCompositeOperation = "destination-in";
      x.translate(w / 2, h * 0.47);
      x.scale(1, (h * 0.47) / (w * 0.5));
      const g = x.createRadialGradient(0, 0, 0, 0, 0, w * 0.5);
      for (let i = 0; i < FEATHER.length; i++) g.addColorStop(FEATHER[i][0], "rgba(0,0,0," + FEATHER[i][1] + ")");
      x.fillStyle = g;
      x.fillRect(-w, -h * 2, w * 2, h * 4);
      return c;
    }
    if (opts.portraitSrc) {
      U.loadImage(opts.portraitSrc).then((im) => {
        if (destroyed) return;
        img = im;
        if (W) buildTargets();
        // tri-tone portrait off the critical path
        setTimeout(() => {
          if (destroyed) return;
          try { portrait = feather(U.portraitCanvas(img, ACC)); } catch (e) { portrait = null; }
          if (lastP >= 0) render(lastP, lastT, 0);
        }, 60);
        if (lastP >= 0) render(lastP, lastT, 0);
      }).catch(() => {});
    }

    /* ------------------------------------------------------------------ */
    /* Contract                                                            */
    /* ------------------------------------------------------------------ */
    function resize(width, height, dpr) {
      W = width; H = height; DPR = dpr || 1;
      layout();
      buildTargets();
    }

    function render(p, timeMs, dtMs) {
      if (destroyed) return;
      lastP = p; lastT = timeMs;
      const t = REDUCED ? 0 : timeMs / 1000;
      ctx.setTransform(DPR, 0, 0, DPR, 0, 0);
      ctx.globalCompositeOperation = "source-over";
      ctx.globalAlpha = 1;
      ctx.fillStyle = COL.bg;
      ctx.fillRect(0, 0, W, H);
      if (!W || !H) return;

      computeNodes(p);

      // floor: near-dark at entry, recedes as the face forms
      const floorA = (0.35 + 0.65 * EZ.outCubic(range(p, 0, 0.08))) * (1 - 0.75 * range(p, 0.66, 0.82));
      drawFloor(floorA, p);

      // portrait rim glow sits behind everything that follows
      const halo = range(p, 0.78, 0.95);
      if (halo > 0) {
        ctx.globalCompositeOperation = "lighter";
        ctx.globalAlpha = 0.17 * halo;
        const gs = BOX.h * 1.35;
        ctx.drawImage(SPR_HALO, BOX.cx - gs / 2, BOX.cy - BOX.h * 0.06 - gs / 2, gs, gs);
        ctx.globalCompositeOperation = "source-over";
      }

      // topology first: during the unfold its nodes slide out from *behind* the server
      const ta = 1 - range(p, 0.665, 0.705);
      if (p > 0.185 && ta > 0) drawTopology(p, t, ta);
      if (p < 0.25) { drawServer(p, t); drawLog(p, t); }

      const fadeOut = 1 - range(p, 0.86, 0.95);
      // until the portrait is ready, the particle silhouette stays as the final state
      const pf = portrait ? fadeOut : Math.max(fadeOut, 0.6);
      if (p > 0.655 && pf > 0) drawParticles(p, t, pf);
      if (p > 0.84) drawPortrait(p, t);
      ctx.globalAlpha = 1;
    }

    function destroy() { destroyed = true; }

    return { resize, render, destroy };
  };
})();
