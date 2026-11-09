/* ==========================================================================
   Scene · vision — Hooman Khoshbin (AI / ML / computer vision / LLMs)

   "A neural network looking at Hooman." Every derived image is computed from
   his real photo ONCE (mosaics, Sobel edges, feature maps, tri-tone) and then
   composited per frame as a pure function of scroll progress. Time only drives
   ambient motion (sensor static, REC blink, cursor, idle pulses).

   Beats (match the captions in index.html):
     0.00–0.18  raw pixels   mosaic 11×14 → 22×28 → 44×55 inside a viewfinder
     0.18–0.38  conv1        a scanline reveals the Sobel edge map + 3×3 readout
     0.38–0.58  conv2…conv5  feature-map tiles peel off → network lights up L→R
     0.58–0.78  llm          describe(image) streams tokens; cross-attention glows
     0.78–1.00  detection    box snaps to the face, label chip, landmarks flash
   ========================================================================== */
(function () {
  "use strict";

  const U = window.SceneUtils;
  window.SceneRegistry = window.SceneRegistry || {};
  if (!U) return;

  /* ------------------------------------------------------------------------
     Static data — normalized portrait coords (u = x / 880, v = y / 1100),
     traced on assets/team/hooman.webp.
     ------------------------------------------------------------------------ */

  // Subject silhouette (head, neck, shoulders). Used to dim the bright sunset
  // sky so the subject — not the background — is the brightest thing.
  const SILHOUETTE = [
    [0.0, 1.0], [0.0, 0.93], [0.013, 0.87], [0.032, 0.81], [0.057, 0.75], [0.088, 0.69],
    [0.12, 0.65], [0.15, 0.62], [0.2, 0.59], [0.25, 0.57], [0.3, 0.558], [0.323, 0.553],
    [0.354, 0.536], [0.375, 0.52], [0.4, 0.495], [0.417, 0.478], [0.417, 0.465], [0.41, 0.455],
    [0.4, 0.44], [0.39, 0.425], [0.377, 0.41], [0.365, 0.4], [0.357, 0.38], [0.352, 0.36],
    [0.345, 0.335], [0.348, 0.31], [0.345, 0.297], [0.331, 0.27], [0.322, 0.25], [0.319, 0.22],
    [0.322, 0.2], [0.329, 0.19], [0.347, 0.175], [0.366, 0.16], [0.389, 0.145], [0.42, 0.13],
    [0.44, 0.12], [0.47, 0.117], [0.513, 0.12], [0.539, 0.125], [0.564, 0.135], [0.588, 0.15],
    [0.606, 0.17], [0.617, 0.19], [0.626, 0.21], [0.633, 0.24], [0.637, 0.27], [0.638, 0.29],
    [0.64, 0.31], [0.647, 0.312], [0.655, 0.33], [0.655, 0.35], [0.648, 0.37], [0.632, 0.385],
    [0.623, 0.4], [0.615, 0.43], [0.61, 0.45], [0.612, 0.465], [0.64, 0.473], [0.664, 0.483],
    [0.68, 0.5], [0.7, 0.526], [0.757, 0.55], [0.82, 0.576], [0.884, 0.61], [0.934, 0.647],
    [0.972, 0.692], [0.997, 0.753], [1.0, 0.76], [1.0, 1.0],
  ];

  const LANDMARKS = [
    [0.352, 0.345], [0.362, 0.4], [0.39, 0.447], [0.442, 0.478], [0.51, 0.49], //  0-4 jaw L → chin
    [0.574, 0.472], [0.61, 0.432], [0.63, 0.385], [0.636, 0.335],              //  5-8 jaw R
    [0.378, 0.3], [0.462, 0.296], [0.532, 0.286], [0.608, 0.281],              //  9-12 brows
    [0.432, 0.338], [0.563, 0.326], [0.497, 0.33],                             // 13-15 lenses, bridge
    [0.5, 0.386], [0.474, 0.392], [0.528, 0.389],                              // 16-18 nose tip, nostrils
    [0.456, 0.417], [0.556, 0.41], [0.507, 0.424],                             // 19-21 mouth L, R, centre
    [0.497, 0.232],                                                            // 22 forehead
  ];
  const MESH = [
    0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8,
    9, 10, 10, 11, 11, 12, 0, 9, 8, 12,
    9, 22, 10, 22, 11, 22, 12, 22,
    9, 13, 10, 13, 13, 15, 15, 14, 14, 11, 14, 12, 10, 15, 15, 11, 0, 13, 13, 1, 8, 14, 14, 7,
    15, 16, 13, 17, 17, 16, 16, 18, 18, 14,
    17, 19, 18, 20, 19, 21, 21, 20, 16, 21,
    1, 19, 2, 19, 3, 19, 3, 21, 4, 21, 5, 21, 5, 20, 6, 20, 7, 20,
  ];

  // BPE-style tokens (leading spaces belong to the token, sub-words split).
  const TOKENS = ["An", " engineer", " who", " teach", "es", " machines", " to", " see", ",", " read", " and", " reason", "."];
  // Cross-attention per token: image regions [u, v, radius as a fraction of box width].
  const ATTN = [
    [[0.5, 0.4, 0.42]],
    [[0.5, 0.32, 0.26]],
    null,
    [[0.508, 0.42, 0.12]],
    [[0.508, 0.42, 0.1]],
    [[0.49, 0.29, 0.32]],
    null,
    [[0.432, 0.338, 0.11], [0.563, 0.326, 0.11]],
    null,
    [[0.432, 0.338, 0.11], [0.563, 0.326, 0.11]],
    null,
    [[0.49, 0.21, 0.2]],
    null,
  ];
  const PROMPT = "describe(image)";

  // Network columns (base sizes in CSS px at a 1440×900 reference).
  const NET = [
    { label: "conv2", n: 4, tw: 70 },
    { label: "conv3", n: 3, tw: 58 },
    { label: "conv4", n: 3, tw: 46 },
    { label: "conv5", n: 2, tw: 36 },
    { label: "fc", n: 6, dot: true },
    { label: "out", n: 1, dot: true },
  ];
  const N_TILES = 12;

  const STAGE_LABELS = [
    "880×1100 · RGB · frame 0001",
    "220×275 · 1 ch · stride 1",
    "pool ×2 · 12 of 256 maps",
    "196 patches → 13 tokens",
    "1 face · 12 ms · frame 0001",
  ];
  const STAGE_AT = [0, 0.18, 0.38, 0.58, 0.78, 1];

  // Working resolutions
  const PW = 880, PH = 1100; // photo canvases (4:5)
  const EW = 220, EH = 275;  // Sobel / feature maps

  window.SceneRegistry.vision = function createVisionScene(canvas, opts) {
    const ctx = canvas.getContext("2d");
    const ACC = (opts.accentRgb || [124, 242, 154]).map((v) => v | 0);
    const ACC_S = `rgb(${ACC[0]},${ACC[1]},${ACC[2]})`;
    const LIGHT = U.mixRgb(ACC, [255, 255, 255], 0.82).map(Math.round);
    const WHITE = "#eef0f3";
    const BG = "#060708";
    const INK = "#04140a"; // text on a filled accent chip
    const FACE = opts.face && opts.face.length === 4 ? opts.face : [0.34, 0.15, 0.66, 0.5];
    const reduced = !!opts.reducedMotion;
    const R = U.range, EZ = U.ease, clamp = U.clamp, lerp = U.lerp;
    const MONO = U.fonts.mono;
    const hasLS = "letterSpacing" in ctx;

    const FONT_LABEL = `500 10px ${MONO}`;
    const FONT_SMALL = `400 10px ${MONO}`;
    const FONT_KERNEL = `500 10.5px ${MONO}`;
    const FONT_CHIP = `500 11px ${MONO}`;

    // Precomputed strings (no per-frame string building)
    const CONF = []; for (let i = 0; i <= 100; i++) CONF.push((i / 100).toFixed(2));
    const PREFIX = []; for (let i = 0; i <= PROMPT.length; i++) PREFIX.push(PROMPT.slice(0, i));
    const TOKCOUNT = []; for (let i = 0; i <= TOKENS.length; i++) TOKCOUNT.push(String(i).padStart(2, "0") + " / " + TOKENS.length + " tok");
    const TOKSHORT = TOKCOUNT.map((s) => s.replace(" tok", ""));
    const ROWSTR = []; for (let i = 0; i < EH; i++) ROWSTR.push("row " + String(i + 1).padStart(3, "0"));
    const TOK_TXT = TOKENS.map((t) => t.replace(/^ /, ""));
    const TOK_LEAD = TOKENS.map((t) => t.charAt(0) === " ");
    const KERNEL = ["−1", "0", "+1", "−2", "0", "+2", "−1", "0", "+1"];
    const CHIP_PREFIX = "person · Hooman Khoshbin · ";

    /* ---------------- State ---------------- */
    let W = 0, H = 0, DPR = 1, mobile = false, gutter = 24;
    let layoutDirty = true, dead = false;
    const B = { x: 0, y: 0, w: 0, h: 0, cx: 0, cy: 0 };
    const CAP = { left: 0, right: 0, top: 0, bottom: 0 };
    const FR = { x0: 0, y0: 0, x1: 0, y1: 0 };             // face rect (px)
    const LMX = new Float32Array(LANDMARKS.length), LMY = new Float32Array(LANDMARKS.length);
    const TILE = [];                                       // tile node rects
    for (let k = 0; k < N_TILES; k++) TILE.push({ x: 0, y: 0, w: 0, h: 0, col: 0 });
    const DOTS = []; for (let i = 0; i < 6; i++) DOTS.push({ x: 0, y: 0 });
    const OUT = { x: 0, y: 0, r: 6 };
    const COLX = new Float32Array(NET.length);
    let colLabelY = 0, netS = 1, recW = 22, segFit = true, colLabelsFit = true;
    const LK = { n: 0, d: new Float32Array(64 * 9), ph: new Float32Array(64) };
    const TERM = { x: 0, y: 0, w: 0, h: 0, fs: 13, lh: 22, pad: 14, head: 28, font: "", promptY: 0,
      tx: new Float32Array(TOKENS.length), ty: new Float32Array(TOKENS.length), tw: new Float32Array(TOKENS.length),
      charW: 7.8, footY: 0, compact: false };
    const CHIP = { w: 0, h: 20, prefixW: 0 };

    // Assets (filled progressively once the photo loads)
    const A = {
      failed: false, baseReady: false, edgeReady: false, tilesReady: false, triReady: false,
      sil: null, bgDim: null, G: null, levels: null, E: null, Eg: null, T: null, tiles: null, rows: null,
    };

    /* ---------------- Canvas helpers ---------------- */
    const mk = (w, h) => {
      const c = document.createElement("canvas");
      c.width = Math.max(1, Math.round(w)); c.height = Math.max(1, Math.round(h));
      return c;
    };
    const resample = (src, w, h) => {
      const c = mk(w, h), x = c.getContext("2d");
      x.imageSmoothingEnabled = true; x.imageSmoothingQuality = "high";
      x.drawImage(src, 0, 0, c.width, c.height);
      return c;
    };
    /** Progressive halving for area-like downsampling. */
    const downTo = (src, w, h) => {
      let cur = src;
      while (cur.width / 2 >= w * 1.5 && cur.height / 2 >= h * 1.5) cur = resample(cur, cur.width / 2, cur.height / 2);
      return resample(cur, w, h);
    };
    /** Same soft vignette as U.portraitCanvas, so every layer shares one silhouette. */
    const fillVignette = (x, w, h) => {
      const g = x.createRadialGradient(w * 0.5, h * 0.4, w * 0.18, w * 0.5, h * 0.46, w * 0.72);
      g.addColorStop(0, "rgba(0,0,0,1)");
      g.addColorStop(0.62, "rgba(0,0,0,0.92)");
      g.addColorStop(1, "rgba(0,0,0,0)");
      x.fillStyle = g; x.fillRect(0, 0, w, h);
    };
    /** Elliptical vignette so the 4:5 frame dissolves into the page (sides + corners). */
    const fillEllipse = (x, w, h) => {
      x.save();
      x.translate(w * 0.5, h * 0.45); x.scale(1, 1.25);
      const g = x.createRadialGradient(0, 0, 0, 0, 0, w * 0.6);
      g.addColorStop(0, "rgba(0,0,0,1)"); g.addColorStop(0.55, "rgba(0,0,0,1)");
      g.addColorStop(0.7, "rgba(0,0,0,0.78)"); g.addColorStop(0.86, "rgba(0,0,0,0.3)"); g.addColorStop(1, "rgba(0,0,0,0)");
      x.fillStyle = g; x.fillRect(-w, -h, w * 2, h * 2);
      x.restore();
    };
    const readAlpha = (src, w, h) => {
      const c = mk(w, h), x = c.getContext("2d", { willReadFrequently: true });
      x.drawImage(src, 0, 0, w, h);
      const d = x.getImageData(0, 0, w, h).data, out = new Float32Array(w * h);
      for (let i = 0; i < out.length; i++) out[i] = d[i * 4 + 3] / 255;
      return out;
    };
    const setLS = (v) => { if (hasLS) ctx.letterSpacing = v; };
    // Crisp line coordinates
    const hair = () => 1 / DPR;
    const crH = (v) => (Math.round(v * DPR - 0.5) + 0.5) / DPR;                // 1 device-px line
    const cr1 = (v) => { const lw = Math.max(1, Math.round(DPR)), o = (lw % 2) * 0.5; return (Math.round(v * DPR - o) + o) / DPR; };

    /* ---------------- Sprites (no image needed) ---------------- */
    const GLOW = U.glowSprite(ACC, 64, 0.16);
    const SOFT = (() => { // accent bloom without a white core (attention, ambient)
      const s = 128, c = mk(s, s), x = c.getContext("2d");
      const g = x.createRadialGradient(s / 2, s / 2, 0, s / 2, s / 2, s / 2);
      g.addColorStop(0, U.rgba(ACC, 0.55)); g.addColorStop(0.5, U.rgba(ACC, 0.16)); g.addColorStop(1, U.rgba(ACC, 0));
      x.fillStyle = g; x.fillRect(0, 0, s, s); return c;
    })();
    const SHADE = (() => { // dark radial used to seat panels / captions
      const s = 128, c = mk(s, s), x = c.getContext("2d");
      const g = x.createRadialGradient(s / 2, s / 2, 0, s / 2, s / 2, s / 2);
      g.addColorStop(0, "rgba(6,7,8,0.9)"); g.addColorStop(0.55, "rgba(6,7,8,0.6)"); g.addColorStop(1, "rgba(6,7,8,0)");
      x.fillStyle = g; x.fillRect(0, 0, s, s); return c;
    })();
    const SCANBAND = (() => { // light trailing above the scanline
      const c = mk(256, 64), x = c.getContext("2d");
      const g = x.createLinearGradient(0, 0, 0, 64);
      g.addColorStop(0, U.rgba(ACC, 0)); g.addColorStop(0.75, U.rgba(ACC, 0.1)); g.addColorStop(1, U.rgba(ACC, 0.42));
      x.fillStyle = g; x.fillRect(0, 0, 256, 64);
      x.globalCompositeOperation = "destination-in";
      const h = x.createLinearGradient(0, 0, 256, 0);
      h.addColorStop(0, "rgba(0,0,0,0)"); h.addColorStop(0.12, "#000"); h.addColorStop(0.88, "#000"); h.addColorStop(1, "rgba(0,0,0,0)");
      x.fillStyle = h; x.fillRect(0, 0, 256, 64); return c;
    })();
    const SCANLINE = (() => {
      const c = mk(512, 2), x = c.getContext("2d");
      const h = x.createLinearGradient(0, 0, 512, 0);
      h.addColorStop(0, U.rgba(ACC, 0)); h.addColorStop(0.08, U.rgba(ACC, 1)); h.addColorStop(0.92, U.rgba(ACC, 1)); h.addColorStop(1, U.rgba(ACC, 0));
      x.fillStyle = h; x.fillRect(0, 0, 512, 2); return c;
    })();
    const NZW = 150, NZH = 188;
    const NOISE = (() => { // 2×2 quadrants of sparse sensor static, jittered at ~14 fps
      const c = mk(NZW * 2, NZH * 2), x = c.getContext("2d");
      const id = x.createImageData(c.width, c.height), d = id.data, rnd = U.rng(902);
      for (let i = 0; i < d.length; i += 4) {
        const r = rnd(); d[i] = d[i + 1] = d[i + 2] = 255; d[i + 3] = Math.pow(r, 5) * 230;
      }
      x.putImageData(id, 0, 0);
      // Pre-mask every quadrant with the frame vignette so static never outlines the box
      x.globalCompositeOperation = "destination-in";
      for (let q = 0; q < 4; q++) {
        x.save(); x.translate((q & 1) * NZW, (q >> 1) * NZH);
        x.beginPath(); x.rect(0, 0, NZW, NZH); x.clip();
        fillEllipse(x, NZW, NZH);
        x.restore();
      }
      x.globalCompositeOperation = "source-over";
      return c;
    })();

    /* ======================================================================
       Precompute — staged across idle callbacks so loading never janks scroll
       ====================================================================== */
    const idle = window.requestIdleCallback
      ? (fn) => window.requestIdleCallback(fn, { timeout: 160 })
      : (fn) => setTimeout(fn, 24);
    const timings = [];
    // Each stage is isolated: if one fails, the scene keeps rendering with what it has.
    // Cost is exposed on the canvas as data-prep for QA.
    const timed = (name, fn) => {
      const t0 = performance.now();
      let ok = true;
      try { fn(); } catch (e) { ok = false; console.warn("[scene:vision] " + name, e); }
      timings.push(name + " " + (performance.now() - t0).toFixed(1) + "ms");
      if (/[?&]debug\b/.test(location.search)) canvas.dataset.prep = timings.join(" · ");
      if (opts.invalidate) opts.invalidate(); // redraw stills (reduced motion) as stages land
      return ok;
    };

    if (opts.portraitSrc) {
      U.loadImage(opts.portraitSrc).then((img) => {
        if (dead) return;
        // ~10–40 ms slices, each in its own idle callback
        const stages = [
          ["matte", () => { const mt = buildMatte(img); A.sil = mt.mat; A.bgDim = mt.dim; }],
          ["frame", () => buildBase(img)],
          ["edges", () => buildEdges(img)],
          ["tilesA", () => buildTiles(0)],
          ["tilesB", () => buildTiles(1)],
          ["tri", () => buildTri(img)],
        ];
        let si = 0;
        const step = () => {
          if (dead || si >= stages.length) return;
          const [name, fn] = stages[si++];
          if (!timed(name, fn) && si <= 2) { A.failed = true; return; } // no frame → "no signal"
          idle(step);
        };
        step();
      }).catch(() => { A.failed = true; });
    } else {
      A.failed = true;
    }

    /**
     * Subject matte from the traced polygon. Around the hair (v < 0.3) the
     * photo itself refines the edge: dark hair vs. bright sunset sky, so a
     * luminance key trims a slightly dilated band down to the real wisps.
     * (Not on the sides: the sun-lit ear and jaw are as bright as the sky.)
     * Falls back to the polygon alone when pixels can't be read.
     */
    function buildMatte(img) {
      const MW = 440, MH = 550;
      const poly = (x, mode) => {
        x.lineJoin = "round"; x.lineWidth = mode === "erode" ? 14 : 8;
        x.beginPath();
        for (let i = 0; i < SILHOUETTE.length; i++) {
          const u = SILHOUETTE[i][0] * MW, v = SILHOUETTE[i][1] * MH;
          if (i) x.lineTo(u, v); else x.moveTo(u, v);
        }
        x.closePath();
        x.fillStyle = "#fff"; x.strokeStyle = "#fff"; x.fill();
        if (mode === "dilate") x.stroke();
        if (mode === "erode") { x.globalCompositeOperation = "destination-out"; x.stroke(); x.globalCompositeOperation = "source-over"; }
      };
      const pc = mk(MW, MH), dc = mk(MW, MH), ec = mk(MW, MH);
      poly(pc.getContext("2d"), "plain"); poly(dc.getContext("2d"), "dilate"); poly(ec.getContext("2d"), "erode");
      const pl = readAlpha(pc, MW, MH), dil = readAlpha(dc, MW, MH), ero = readAlpha(ec, MW, MH);
      const pix = U.readPixels(img, MW, MH);
      let m = new Float32Array(MW * MH);
      for (let y = 0; y < MH; y++) {
        const v = y / MH, hz = pix ? 1 - smooth(0.27, 0.31, v) : 0; // hair zone weight
        for (let x = 0; x < MW; x++) {
          const i = y * MW + x;
          let keyed = pl[i];
          if (hz > 0) {
            const d = pix.data, k = i * 4;
            const l = (0.2126 * d[k] + 0.7152 * d[k + 1] + 0.0722 * d[k + 2]) / 255;
            keyed = Math.min(1, ero[i] + Math.max(0, dil[i] - ero[i]) * (1 - smooth(0.5, 0.74, l)));
          }
          m[i] = pl[i] + (keyed - pl[i]) * hz;
        }
      }
      m = blur(m, MW, MH, 1, 1);
      const mat = paint(MW, MH, (d, k, i) => { d[k] = d[k + 1] = d[k + 2] = 255; d[k + 3] = m[i] * 255; });
      const dim = paint(MW, MH, (d, k, i) => { d[k] = d[k + 1] = d[k + 2] = 0; d[k + 3] = (1 - m[i]) * 255; });
      return { mat, dim };
    }

    function buildBase(img) {
      // G: the "raw" greyscale frame — S-curve contrast, background dimmed, vignetted
      const G = mk(PW, PH), g = G.getContext("2d");
      U.drawCover(g, img, 0, 0, PW, PH);
      g.globalCompositeOperation = "saturation"; g.fillStyle = "#000"; g.fillRect(0, 0, PW, PH);
      const flat = resample(G, PW, PH);
      g.globalCompositeOperation = "overlay"; g.globalAlpha = 0.7; g.drawImage(flat, 0, 0); g.globalAlpha = 1;
      g.globalCompositeOperation = "multiply"; g.fillStyle = "rgb(212,226,219)"; g.fillRect(0, 0, PW, PH);
      g.globalCompositeOperation = "source-over"; g.globalAlpha = 0.72; g.drawImage(A.bgDim, 0, 0, PW, PH); g.globalAlpha = 1;
      g.globalCompositeOperation = "destination-in"; fillVignette(g, PW, PH); fillEllipse(g, PW, PH);
      g.globalCompositeOperation = "source-over";
      A.G = G;

      // Mosaic levels 11×14, 22×28, 44×55 (area-averaged), with dissolve masks
      const l2 = downTo(G, 44, 55), l1 = resample(l2, 22, 28), l0 = resample(l1, 11, 14);
      const fc = [(FACE[0] + FACE[2]) / 2, (FACE[1] + FACE[3]) / 2 - 0.02];
      const mkLevel = (c, seed) => {
        const cols = c.width, rows = c.height, n = cols * rows, rnd = U.rng(seed);
        const thr = new Float32Array(n);
        let lo = 1e9, hi = -1e9;
        for (let j = 0; j < rows; j++) for (let i = 0; i < cols; i++) {
          const du = (i + 0.5) / cols - fc[0], dv = ((j + 0.5) / rows - fc[1]) * 1.25;
          const v = 0.55 * rnd() + 0.45 * Math.min(1, Math.hypot(du, dv) / 0.7); // the face resolves first
          thr[j * cols + i] = v; if (v < lo) lo = v; if (v > hi) hi = v;
        }
        for (let k = 0; k < n; k++) thr[k] = ((thr[k] - lo) / (hi - lo + 1e-6)) * 0.985;
        const mc = mk(cols, rows), out = mk(cols, rows);
        const mctx = mc.getContext("2d"), mask = mctx.createImageData(cols, rows);
        for (let k = 0; k < n; k++) { mask.data[k * 4] = mask.data[k * 4 + 1] = mask.data[k * 4 + 2] = 255; }
        // Per-cell read-out noise: 4 frames, pre-masked by the level's own alpha
        const nz = [];
        for (let f = 0; f < 4; f++) {
          const cnv = mk(cols, rows), x = cnv.getContext("2d"), id = x.createImageData(cols, rows);
          for (let k = 0; k < n; k++) { const r = rnd(); id.data[k * 4] = id.data[k * 4 + 1] = id.data[k * 4 + 2] = 255; id.data[k * 4 + 3] = r * r * 255; }
          x.putImageData(id, 0, 0);
          x.globalCompositeOperation = "destination-in"; x.drawImage(c, 0, 0);
          nz.push(cnv);
        }
        return { c, cols, rows, thr, mask, mc, mctx, out, octx: out.getContext("2d"), lastM: -1, nz };
      };
      A.levels = [mkLevel(l0, 11), mkLevel(l1, 22), mkLevel(l2, 44)];
      A.baseReady = true;
    }

    /* ---- numeric helpers for the feature maps ---- */
    const blur = (src, w, h, r, passes) => {
      let a = Float32Array.from(src), b = new Float32Array(w * h);
      const inv = 1 / (2 * r + 1);
      for (let p = 0; p < passes; p++) {
        for (let y = 0; y < h; y++) {
          const row = y * w; let acc = 0;
          for (let x = -r; x <= r; x++) acc += a[row + Math.min(w - 1, Math.max(0, x))];
          for (let x = 0; x < w; x++) {
            b[row + x] = acc * inv;
            acc += a[row + Math.min(w - 1, x + r + 1)] - a[row + Math.max(0, x - r)];
          }
        }
        for (let x = 0; x < w; x++) {
          let acc = 0;
          for (let y = -r; y <= r; y++) acc += b[Math.min(h - 1, Math.max(0, y)) * w + x];
          for (let y = 0; y < h; y++) {
            a[y * w + x] = acc * inv;
            acc += b[Math.min(h - 1, y + r + 1) * w + x] - b[Math.max(0, y - r) * w + x];
          }
        }
      }
      return a;
    };
    const pool = (src, sw, sh, dw, dh, useMax) => {
      const out = new Float32Array(dw * dh);
      for (let y = 0; y < dh; y++) {
        const y0 = Math.floor((y * sh) / dh), y1 = Math.max(y0 + 1, Math.floor(((y + 1) * sh) / dh));
        for (let x = 0; x < dw; x++) {
          const x0 = Math.floor((x * sw) / dw), x1 = Math.max(x0 + 1, Math.floor(((x + 1) * sw) / dw));
          let acc = useMax ? -1e9 : 0, n = 0;
          for (let yy = y0; yy < y1; yy++) for (let xx = x0; xx < x1; xx++) {
            const v = src[yy * sw + xx];
            if (useMax) { if (v > acc) acc = v; } else { acc += v; n++; }
          }
          out[y * dw + x] = useMax ? acc : acc / n;
        }
      }
      return out;
    };
    const pct = (arr, q) => {
      const step = Math.max(1, Math.floor(arr.length / 4000)), s = [];
      for (let i = 0; i < arr.length; i += step) s.push(arr[i]);
      s.sort((a, b) => a - b);
      return s[Math.min(s.length - 1, Math.floor(q * s.length))] || 1;
    };
    const norm = (arr, q) => {
      const hi = pct(arr, q) || 1, out = new Float32Array(arr.length);
      for (let i = 0; i < arr.length; i++) out[i] = Math.min(1, Math.max(0, arr[i] / hi));
      return out;
    };
    const smooth = (a, b, x) => { const t = clamp((x - a) / (b - a)); return t * t * (3 - 2 * t); };
    const D0 = [10, 13, 15], NEG = [182, 194, 200];
    const ramp = (d, k, v) => { // dark → accent → near-white (same family as the tri-tone)
      v = v < 0 ? 0 : v > 1 ? 1 : v;
      if (v < 0.5) {
        const t = Math.pow(v * 2, 1.15);
        d[k] = D0[0] + (ACC[0] - D0[0]) * t; d[k + 1] = D0[1] + (ACC[1] - D0[1]) * t; d[k + 2] = D0[2] + (ACC[2] - D0[2]) * t;
      } else {
        const t = (v - 0.5) * 2;
        d[k] = ACC[0] + (LIGHT[0] - ACC[0]) * t; d[k + 1] = ACC[1] + (LIGHT[1] - ACC[1]) * t; d[k + 2] = ACC[2] + (LIGHT[2] - ACC[2]) * t;
      }
      d[k + 3] = 255;
    };
    const diverge = (d, k, s) => { // signed response: accent (+) / cool grey (−)
      const a = Math.pow(Math.min(1, Math.abs(s)), 0.75), T = s >= 0 ? ACC : NEG;
      d[k] = D0[0] + (T[0] - D0[0]) * a; d[k + 1] = D0[1] + (T[1] - D0[1]) * a; d[k + 2] = D0[2] + (T[2] - D0[2]) * a; d[k + 3] = 255;
    };
    const paint = (w, h, fill) => {
      const c = mk(w, h), x = c.getContext("2d"), id = x.createImageData(w, h), d = id.data;
      for (let i = 0, n = w * h; i < n; i++) fill(d, i * 4, i);
      x.putImageData(id, 0, 0);
      return c;
    };

    // Shared numeric products of buildEdges → buildTiles
    let NUM = null;

    function buildEdges(img) {
      const id = U.readPixels(img, EW, EH);
      const N = EW * EH;
      // Weights from my own canvases (never tainted)
      const vig = (() => {
        const c = mk(EW, EH), x = c.getContext("2d");
        fillVignette(x, EW, EH); x.globalCompositeOperation = "destination-in"; fillEllipse(x, EW, EH);
        return readAlpha(c, EW, EH);
      })();
      const sil = readAlpha(A.sil, EW, EH);
      const fcu = (FACE[0] + FACE[2]) / 2, fcv = (FACE[1] + FACE[3]) / 2;
      const focus = new Float32Array(N);
      for (let y = 0; y < EH; y++) for (let x = 0; x < EW; x++) {
        const u = x / EW, v = y / EH, i = y * EW + x;
        const gs = Math.exp(-((u - fcu) ** 2) / (2 * 0.2 * 0.2) - ((v - fcv) ** 2) / (2 * 0.24 * 0.24));
        const bottom = 1 - 0.7 * smooth(0.6, 1.0, v); // calm under the caption zone
        focus[i] = (0.1 + 0.9 * sil[i]) * (0.28 + 0.72 * gs) * vig[i] * bottom;
      }

      if (id) {
        const d = id.data, L = new Float32Array(N);
        for (let i = 0; i < N; i++) L[i] = (0.2126 * d[i * 4] + 0.7152 * d[i * 4 + 1] + 0.0722 * d[i * 4 + 2]) / 255;
        const gx = new Float32Array(N), gy = new Float32Array(N), mag = new Float32Array(N);
        for (let y = 1; y < EH - 1; y++) for (let x = 1; x < EW - 1; x++) {
          const i = y * EW + x;
          const a = L[i - EW - 1], b = L[i - EW], c = L[i - EW + 1], dd = L[i - 1], f = L[i + 1], g = L[i + EW - 1], h = L[i + EW], k = L[i + EW + 1];
          const sx = (c + 2 * f + k) - (a + 2 * dd + g);
          const sy = (g + 2 * h + k) - (a + 2 * b + c);
          gx[i] = sx; gy[i] = sy; mag[i] = Math.hypot(sx, sy);
        }
        const p98 = pct(mag, 0.985) || 1, sc = 1 / p98;
        // Soft non-maximum suppression (Canny-style thinning) → thin wires;
        // the un-thinned map feeds the bloom.
        const ew = new Float32Array(N), mn = new Float32Array(N);
        for (let y = 1; y < EH - 1; y++) for (let x = 1; x < EW - 1; x++) {
          const i = y * EW + x, ax = Math.abs(gx[i]), ay = Math.abs(gy[i]), m0 = mag[i];
          let a, b;
          if (ax > 2.414 * ay) { a = mag[i - 1]; b = mag[i + 1]; }
          else if (ay > 2.414 * ax) { a = mag[i - EW]; b = mag[i + EW]; }
          else if (gx[i] * gy[i] > 0) { a = mag[i - EW - 1]; b = mag[i + EW + 1]; }
          else { a = mag[i - EW + 1]; b = mag[i + EW - 1]; }
          mn[i] = (m0 >= a && m0 >= b ? m0 : m0 * 0.28) * sc;
        }
        for (let i = 0; i < N; i++) ew[i] = smooth(0.1, 0.85, mag[i] * sc) * focus[i];
        const soft = paint(EW, EH, (o, k, i) => { o[k] = ACC[0]; o[k + 1] = ACC[1]; o[k + 2] = ACC[2]; o[k + 3] = Math.min(255, ew[i] * 255); });
        // Edge map rendered at 2×: bilinear magnitude field re-thresholded (SDF-style)
        // so lines stay thin and crisp at display size; accent with white-hot cores.
        const samp = (arr, x, y) => {
          const x0 = Math.max(0, Math.min(EW - 1, Math.floor(x))), y0 = Math.max(0, Math.min(EH - 1, Math.floor(y)));
          const x1 = Math.min(EW - 1, x0 + 1), y1 = Math.min(EH - 1, y0 + 1);
          const fx = clamp(x - x0), fy = clamp(y - y0);
          const a = arr[y0 * EW + x0] + (arr[y0 * EW + x1] - arr[y0 * EW + x0]) * fx;
          const b = arr[y1 * EW + x0] + (arr[y1 * EW + x1] - arr[y1 * EW + x0]) * fx;
          return a + (b - a) * fy;
        };
        const E2W = EW * 2;
        A.E = paint(E2W, EH * 2, (o, k, i) => {
          const X = ((i % E2W) + 0.5) / 2 - 0.5, Y = (Math.floor(i / E2W) + 0.5) / 2 - 0.5;
          const e = smooth(0.14, 0.7, samp(mn, X, Y)) * samp(focus, X, Y);
          const wt = e * e * 0.4;
          o[k] = ACC[0] + (255 - ACC[0]) * wt; o[k + 1] = ACC[1] + (255 - ACC[1]) * wt; o[k + 2] = ACC[2] + (255 - ACC[2]) * wt;
          o[k + 3] = Math.min(255, e * 290);
        });
        A.Eg = downTo(soft, 55, 69);
        // Per-row strongest edge → the readout tracks real values under the scanline
        const rows = new Float32Array(EH * 2);
        let lastX = 0.5;
        for (let y = 0; y < EH; y++) {
          let best = 0, bx = -1;
          for (let x = Math.floor(EW * 0.12); x < Math.ceil(EW * 0.88); x++) { const v = ew[y * EW + x]; if (v > best) { best = v; bx = x; } }
          if (bx >= 0 && best > 0.06) lastX = bx / EW;
          rows[y * 2] = lastX; rows[y * 2 + 1] = Math.min(0.99, 1 - Math.exp(-1.3 * mag[y * EW + Math.round(lastX * EW)] * sc));
        }
        A.rows = rows;
        NUM = { L, gx, gy, mag, sc, sil, vig, focus };
      } else {
        // Tainted canvas (file://): approximate the gradient with offset differences.
        A.E = compositeEdges(A.G, EW, EH, true);
        const rows = new Float32Array(EH * 2);
        for (let y = 0; y < EH; y++) {
          const v = y / EH;
          rows[y * 2] = 0.5 + 0.18 * Math.sin(v * 17.0) * Math.min(1, v * 3);
          rows[y * 2 + 1] = 0.35 + 0.3 * Math.abs(Math.sin(v * 23.0));
        }
        A.rows = rows;
      }
      if (!A.Eg) A.Eg = downTo(A.E, 55, 69);
      A.edgeReady = true;
    }

    /** Gradient-magnitude approximation via 'difference' compositing (works on tainted sources). */
    function compositeEdges(src, w, h, tint) {
      const base = mk(w, h), bx = base.getContext("2d");
      bx.fillStyle = "#000"; bx.fillRect(0, 0, w, h); bx.drawImage(src, 0, 0, w, h);
      const ex = mk(w, h), xx = ex.getContext("2d");
      xx.drawImage(base, 0, 0); xx.globalCompositeOperation = "difference"; xx.drawImage(base, 1, 0);
      const ey = mk(w, h), yx = ey.getContext("2d");
      yx.drawImage(base, 0, 0); yx.globalCompositeOperation = "difference"; yx.drawImage(base, 0, 1);
      const out = mk(w, h), o = out.getContext("2d");
      o.drawImage(ex, 0, 0); o.globalCompositeOperation = "lighter"; o.drawImage(ey, 0, 0);
      const tmp = resample(out, w, h);
      for (let i = 0; i < 2; i++) o.drawImage(tmp, 0, 0); // ×3 gain
      if (tint) { o.globalCompositeOperation = "multiply"; o.fillStyle = ACC_S; o.fillRect(0, 0, w, h); }
      o.globalCompositeOperation = "destination-in"; fillVignette(o, w, h); fillEllipse(o, w, h);
      // Same focus as the Sobel path: subject over background, face over shirt
      const fcx = ((FACE[0] + FACE[2]) / 2) * w, fcy = ((FACE[1] + FACE[3]) / 2) * h;
      const g = o.createRadialGradient(fcx, fcy, w * 0.12, fcx, fcy, w * 0.75);
      g.addColorStop(0, "rgba(0,0,0,1)"); g.addColorStop(1, "rgba(0,0,0,0.25)");
      o.fillStyle = g; o.fillRect(0, 0, w, h);
      if (A.sil) {
        const m = mk(w, h), mx = m.getContext("2d");
        mx.fillStyle = "rgba(0,0,0,0.12)"; mx.fillRect(0, 0, w, h); mx.drawImage(A.sil, 0, 0, w, h);
        o.drawImage(m, 0, 0);
      }
      o.globalCompositeOperation = "source-over";
      return out;
    }

    function buildTiles(part) {
      const tiles = A.tilesWip || (A.tilesWip = []);
      if (NUM) {
        const { L, gx, gy, mag, sc, sil } = NUM;
        const N = EW * EH;
        const m2 = new Float32Array(N); for (let i = 0; i < N; i++) m2[i] = mag[i] * mag[i];
        if (part === 0) {
          // conv2 — oriented edges & inversion (full 220×275)
          tiles.push({ c: paint(EW, EH, (d, k, i) => diverge(d, k, gx[i] * sc * 1.25)), px: false });
          tiles.push({ c: paint(EW, EH, (d, k, i) => diverge(d, k, gy[i] * sc * 1.25)), px: false });
          tiles.push({ c: paint(EW, EH, (d, k, i) => diverge(d, k, (gx[i] - gy[i]) * 0.7071 * sc * 1.25)), px: false });
          tiles.push({ c: paint(EW, EH, (d, k, i) => ramp(d, k, Math.pow(1 - L[i], 1.4) * 0.95)), px: false });
          // conv3 — 110×138: low-pass, texture energy, thresholded edges
          const w3 = 110, h3 = 138;
          const lp = pool(blur(L, EW, EH, 3, 2), EW, EH, w3, h3, false);
          tiles.push({ c: paint(w3, h3, (d, k, i) => ramp(d, k, lp[i] * 0.9)), px: true });
          const tex = norm(pool(blur(m2, EW, EH, 3, 2), EW, EH, w3, h3, false), 0.99);
          tiles.push({ c: paint(w3, h3, (d, k, i) => ramp(d, k, Math.sqrt(tex[i]))), px: true });
          const bin = new Float32Array(N); for (let i = 0; i < N; i++) bin[i] = mag[i] * sc > 0.42 ? 1 : 0;
          const binp = pool(bin, EW, EH, w3, h3, true);
          tiles.push({ c: paint(w3, h3, (d, k, i) => ramp(d, k, binp[i] * 0.86)), px: true });
          return; // conv4/conv5 in the next idle slice
        }
        // conv4 — 55×69: dark blobs (lenses, hair, beard), horizontal bars, mirror symmetry
        const w4 = 55, h4 = 69;
        const dark = new Float32Array(N); for (let i = 0; i < N; i++) dark[i] = 1 - L[i];
        const b1 = blur(dark, EW, EH, 2, 2), b2 = blur(dark, EW, EH, 8, 2);
        const dog = new Float32Array(N); for (let i = 0; i < N; i++) dog[i] = Math.max(0, b1[i] - b2[i]);
        const dogN = norm(pool(dog, EW, EH, w4, h4, true), 0.985);
        tiles.push({ c: paint(w4, h4, (d, k, i) => ramp(d, k, dogN[i])), px: true });
        const ax = blur(Float32Array.from(gx, Math.abs), EW, EH, 3, 2), ay = blur(Float32Array.from(gy, Math.abs), EW, EH, 3, 2);
        const hb = new Float32Array(N); for (let i = 0; i < N; i++) hb[i] = Math.max(0, ay[i] - ax[i] * 0.8);
        const hbN = norm(pool(hb, EW, EH, w4, h4, true), 0.985);
        tiles.push({ c: paint(w4, h4, (d, k, i) => ramp(d, k, hbN[i])), px: true });
        const axis = Math.round(EW * 0.503), sym = new Float32Array(N);
        for (let y = 0; y < EH; y++) for (let x = 0; x < EW; x++) {
          const xm = 2 * axis - x, i = y * EW + x;
          sym[i] = xm < 0 || xm >= EW ? 0 : (1 - Math.min(1, Math.abs(L[i] - L[y * EW + xm]) * 5)) * sil[i];
        }
        const symN = norm(pool(blur(sym, EW, EH, 2, 1), EW, EH, w4, h4, false), 0.99);
        tiles.push({ c: paint(w4, h4, (d, k, i) => ramp(d, k, symN[i] * 0.95)), px: true });
        // conv5 — 22×28: "face" and "person" units
        const w5 = 22, h5 = 28;
        const fcu = (FACE[0] + FACE[2]) / 2, fcv = (FACE[1] + FACE[3]) / 2;
        const dog5 = norm(pool(dog, EW, EH, w5, h5, false), 0.99);
        const face = paint(w5, h5, (d, k, i) => {
          const u = ((i % w5) + 0.5) / w5, v = (Math.floor(i / w5) + 0.5) / h5;
          const prior = Math.exp(-((u - fcu) ** 2) / (2 * 0.11 * 0.11) - ((v - fcv) ** 2) / (2 * 0.13 * 0.13));
          ramp(d, k, Math.pow(prior, 0.8) * (0.55 + 0.45 * dog5[i]));
        });
        tiles.push({ c: face, px: true });
        const sil5 = pool(sil, EW, EH, w5, h5, false), tex5 = norm(pool(m2, EW, EH, w5, h5, false), 0.97);
        tiles.push({ c: paint(w5, h5, (d, k, i) => ramp(d, k, sil5[i] * (0.45 + 0.4 * Math.sqrt(tex5[i])))), px: true });
        NUM = null; // release the big arrays
      } else {
        if (part === 1) return; // the fallback builds everything in one slice
        // Tainted fallback: composite-only variations of the grey frame
        const G = A.G;
        const ex = compositeEdges(G, EW, EH, true);
        const inv = (() => { const c = mk(EW, EH), x = c.getContext("2d"); x.fillStyle = "#000"; x.fillRect(0, 0, EW, EH); x.drawImage(G, 0, 0, EW, EH); x.globalCompositeOperation = "difference"; x.fillStyle = "#fff"; x.fillRect(0, 0, EW, EH); x.globalCompositeOperation = "multiply"; x.fillStyle = ACC_S; x.fillRect(0, 0, EW, EH); return c; })();
        const tintG = (() => { const c = mk(EW, EH), x = c.getContext("2d"); x.fillStyle = "#0a0d0f"; x.fillRect(0, 0, EW, EH); x.drawImage(G, 0, 0, EW, EH); x.globalCompositeOperation = "multiply"; x.fillStyle = ACC_S; x.fillRect(0, 0, EW, EH); return c; })();
        const srcs = [ex, ex, inv, tintG, tintG, ex, inv, ex, tintG, inv, tintG, ex];
        const res = [EW, EW, EW, EW, 110, 110, 110, 55, 55, 55, 22, 22];
        for (let k = 0; k < N_TILES; k++) {
          const w = res[k], h = Math.round(w * 1.25);
          tiles.push({ c: k > 3 ? downTo(srcs[k], w, h) : srcs[k], px: k > 3 });
        }
      }
      A.tiles = tiles; A.tilesWip = null;
      A.tilesReady = tiles.length === N_TILES;
    }

    function buildTri(img) {
      const base = U.portraitCanvas(img, ACC, { contrast: 1.15 });
      const T = mk(base.width, base.height), t = T.getContext("2d");
      t.drawImage(base, 0, 0);
      t.globalCompositeOperation = "source-atop"; t.globalAlpha = 0.76;
      t.drawImage(A.bgDim, 0, 0, T.width, T.height);
      t.globalAlpha = 1;
      t.globalCompositeOperation = "destination-in"; fillEllipse(t, T.width, T.height);
      t.globalCompositeOperation = "source-over";
      A.T = T;
      A.triReady = true;
    }

    /* ======================================================================
       Layout (on resize / font load)
       ====================================================================== */
    function measureCaptions() {
      const wrap = opts.el && opts.el.querySelector(".scene__captions");
      const cr = canvas.getBoundingClientRect();
      if (wrap && cr.width) {
        const r = wrap.getBoundingClientRect();
        let hmax = 0;
        wrap.querySelectorAll(".cap").forEach((c) => { hmax = Math.max(hmax, c.offsetHeight); });
        CAP.left = r.left - cr.left; CAP.right = r.right - cr.left;
        CAP.bottom = r.bottom - cr.top; CAP.top = CAP.bottom - (hmax || 140);
      } else {
        CAP.left = gutter; CAP.right = Math.min(W - gutter, gutter + 560);
        CAP.bottom = H - clamp(H * 0.1, 40, 110); CAP.top = CAP.bottom - 150;
      }
    }

    function addLink(x0, y0, c0x, c0y, c1x, c1y, x1, y1, stage, ph) {
      const o = LK.n * 9, d = LK.d;
      d[o] = x0; d[o + 1] = y0; d[o + 2] = c0x; d[o + 3] = c0y; d[o + 4] = c1x; d[o + 5] = c1y; d[o + 6] = x1; d[o + 7] = y1; d[o + 8] = stage;
      LK.ph[LK.n] = ph; LK.n++;
    }

    function layoutNet() {
      const unit = clamp(B.h / 594, 0.75, 1.3);
      const side = !mobile || W > H * 1.25; // landscape phones: beside the frame, not under it
      const gapX = (mobile ? 54 : 68) * unit, gapY = 12 * unit, dotGap = 22 * unit, outLabel = (mobile ? 54 : 84) * unit;
      let sumW = 0, maxH = 0;
      for (const c of NET) {
        if (!c.dot) { sumW += c.tw * unit; maxH = Math.max(maxH, c.n * c.tw * 1.25 * unit + (c.n - 1) * gapY); }
        else maxH = Math.max(maxH, (c.n - 1) * dotGap);
      }
      const needW = sumW + gapX * (NET.length - 1) + outLabel, needH = maxH + 30 * unit;
      let x0, x1, yc, availH;
      if (side) {
        x0 = B.x + B.w + Math.max(36, W * 0.03); x1 = W - gutter; yc = B.cy; availH = B.h * 0.86;
      } else {
        const y0 = B.y + B.h + 34, y1 = Math.max(y0 + 120, CAP.top - 14);
        x0 = gutter; x1 = W - gutter; yc = (y0 + y1) / 2 - 6; availH = y1 - y0;
      }
      const s = Math.min(1.15, (x1 - x0) / needW, availH / needH);
      netS = s;
      let cx = x0 + Math.max(0, (x1 - x0 - needW * s) / 2);
      const top = yc - (maxH * s) / 2 - 8 * unit * s;
      colLabelY = top + maxH * s + 24 * unit * s;
      const netTop = top;
      let k = 0;
      for (let ci = 0; ci < NET.length; ci++) {
        const c = NET[ci];
        if (!c.dot) {
          const tw = c.tw * unit * s, th = tw * 1.25, gy = gapY * s, colH = c.n * th + (c.n - 1) * gy;
          const y0 = netTop + (maxH * s - colH) / 2;
          for (let i = 0; i < c.n; i++, k++) { const t = TILE[k]; t.x = cx; t.y = y0 + i * (th + gy); t.w = tw; t.h = th; t.col = ci; }
          COLX[ci] = cx + tw / 2;
          cx += tw + gapX * s;
        } else if (c.n > 1) {
          const g = dotGap * s, colH = (c.n - 1) * g, y0 = netTop + (maxH * s - colH) / 2;
          for (let i = 0; i < c.n; i++) { DOTS[i].x = cx; DOTS[i].y = y0 + i * g; }
          COLX[ci] = cx;
          cx += gapX * s;
        } else {
          OUT.x = cx; OUT.y = netTop + (maxH * s) / 2; OUT.r = 6.5 * s * unit;
          COLX[ci] = cx;
        }
      }
      // Links
      LK.n = 0;
      const rnd = U.rng(77);
      const curve = (ax, ay, bx, by, st) => { const dx = (bx - ax) * 0.5; addLink(ax, ay, ax + dx, ay, bx - dx, by, bx, by, st, rnd()); };
      if (side) {
        for (let i = 0; i < 4; i++) {
          const t = TILE[i], yIn = B.y + B.h * (0.3 + i * 0.09);
          curve(B.x + B.w + 6, yIn, t.x, t.y + t.h / 2, 0);
        }
      }
      const colTiles = (ci) => { const out = []; for (let j = 0; j < N_TILES; j++) if (TILE[j].col === ci) out.push(TILE[j]); return out; };
      for (let ci = 0; ci < 3; ci++) {
        const a = colTiles(ci), b = colTiles(ci + 1);
        for (const ta of a) for (const tb of b) curve(ta.x + ta.w, ta.y + ta.h / 2, tb.x, tb.y + tb.h / 2, ci + 1);
      }
      for (const ta of colTiles(3)) for (const d of DOTS) curve(ta.x + ta.w, ta.y + ta.h / 2, d.x - 3 * s, d.y, 4);
      for (const d of DOTS) curve(d.x + 3 * s, d.y, OUT.x - OUT.r, OUT.y, 5);
    }

    function layoutTerm() {
      const fs = mobile || W < 1200 ? 12 : 13;
      TERM.fs = fs; TERM.lh = Math.round(fs * 1.8); TERM.pad = mobile ? 14 : 16; TERM.head = 30;
      TERM.font = `400 ${fs}px ${MONO}`;
      let x, y, w;
      if (!mobile || W > H * 1.25) {
        x = B.x + B.w * 0.665;
        w = clamp(W * 0.6 - x, 230, 400);
        y = B.y + B.h * 0.085;
      } else {
        w = Math.min(W - gutter * 2, 460); x = clamp(B.cx - w / 2, gutter, W - gutter - w); y = B.y + B.h + 34;
      }
      ctx.save();
      ctx.font = TERM.font; setLS("0px");
      const space = ctx.measureText(" ").width;
      TERM.charW = ctx.measureText("M").width;
      const maxW = w - TERM.pad * 2;
      let cx = 0, line = 0;
      for (let i = 0; i < TOKENS.length; i++) {
        const tw = ctx.measureText(TOK_TXT[i]).width;
        let lead = TOK_LEAD[i] ? space : 0;
        if (cx + lead + tw > maxW && cx > 0) { line++; cx = 0; lead = 0; }
        TERM.tx[i] = cx + lead; TERM.ty[i] = line; TERM.tw[i] = tw;
        cx += lead + tw;
      }
      ctx.restore();
      const lines = line + 1;
      const h = TERM.pad + TERM.head + TERM.lh * (1 + lines) + TERM.lh * 0.55 + TERM.pad * 0.5;
      if (mobile && y + h > CAP.top - 12) y = Math.max(B.y + B.h + 8, CAP.top - 12 - h);
      TERM.x = x; TERM.y = y; TERM.w = w; TERM.h = h; TERM.compact = w < 300;
      TERM.promptY = TERM.pad + TERM.head + TERM.lh * 0.5;
      TERM.footY = TERM.promptY + TERM.lh * (1 + lines) - TERM.lh * 0.1;
    }

    function layout() {
      mobile = U.isMobile(W);
      gutter = clamp(W * 0.04, 16, 48);
      const pb = U.portraitBox(W, H);
      B.x = pb.x; B.y = pb.y; B.w = pb.w; B.h = pb.h; B.cx = pb.cx; B.cy = pb.cy;
      measureCaptions();
      FR.x0 = B.x + FACE[0] * B.w; FR.y0 = B.y + FACE[1] * B.h; FR.x1 = B.x + FACE[2] * B.w; FR.y1 = B.y + FACE[3] * B.h;
      for (let i = 0; i < LANDMARKS.length; i++) { LMX[i] = B.x + LANDMARKS[i][0] * B.w; LMY[i] = B.y + LANDMARKS[i][1] * B.h; }
      layoutNet();
      layoutTerm();
      ctx.save(); ctx.font = FONT_CHIP; setLS("0px");
      CHIP.prefixW = ctx.measureText(CHIP_PREFIX).width;
      CHIP.w = Math.ceil(CHIP.prefixW + ctx.measureText("0.99").width + 18);
      ctx.font = FONT_LABEL; setLS("0.16em"); recW = ctx.measureText("REC").width;
      setLS("0.08em"); let lw = 0; for (const l of STAGE_LABELS) lw = Math.max(lw, ctx.measureText(l).width); setLS("0px");
      const segTotal = 5 * (mobile ? 12 : 16) + 16;
      segFit = lw + segTotal + 12 < B.w + (mobile ? 16 : 24);
      setLS("0.12em"); colLabelsFit = true;
      for (let i = 0; i < NET.length - 1; i++) {
        const need = (ctx.measureText(NET[i].label).width + ctx.measureText(NET[i + 1].label).width) / 2 + 6;
        if (COLX[i + 1] - COLX[i] < need) colLabelsFit = false;
      }
      setLS("0px");
      ctx.restore();
      layoutDirty = false;
    }

    if (document.fonts && document.fonts.ready) {
      document.fonts.ready.then(() => { layoutDirty = true; });
      if (document.fonts.addEventListener) document.fonts.addEventListener("loadingdone", () => { layoutDirty = true; });
    }

    /* ======================================================================
       Drawing
       ====================================================================== */
    function drawMosaic(p, alpha, tm) {
      const Lv = A.levels;
      const m1 = R(p, 0.04, 0.085), m2 = R(p, 0.1, 0.145);
      ctx.imageSmoothingEnabled = false;
      ctx.globalAlpha = alpha;
      let gridLv, gridB = null, gridM = 0;
      if (m2 >= 1) { ctx.drawImage(Lv[2].c, B.x, B.y, B.w, B.h); gridLv = Lv[2]; }
      else if (m1 >= 1) {
        ctx.drawImage(Lv[1].c, B.x, B.y, B.w, B.h); gridLv = Lv[1];
        if (m2 > 0) { ctx.drawImage(dissolve(Lv[2], m2), B.x, B.y, B.w, B.h); gridB = Lv[2]; gridM = m2; }
      } else {
        ctx.drawImage(Lv[0].c, B.x, B.y, B.w, B.h); gridLv = Lv[0];
        if (m1 > 0) { ctx.drawImage(dissolve(Lv[1], m1), B.x, B.y, B.w, B.h); gridB = Lv[1]; gridM = m1; }
      }
      // Static: cells flicker like raw sensor read-out (~11 fps, still when reduced)
      const nzA = (0.16 - 0.06 * R(p, 0, 0.18)) * alpha;
      const nl = gridM > 0.5 ? gridB : gridLv;
      ctx.globalCompositeOperation = "lighter"; ctx.globalAlpha = nzA;
      ctx.drawImage(nl.nz[reduced ? 0 : Math.floor(tm / 90) % 4], B.x, B.y, B.w, B.h);
      ctx.globalCompositeOperation = "source-over";
      ctx.imageSmoothingEnabled = true;
      // Cell separators: one device pixel of page colour, like a sensor grid
      ctx.strokeStyle = BG; ctx.lineWidth = hair();
      drawGrid(gridLv, alpha * 0.7 * (1 - gridM));
      if (gridB) drawGrid(gridB, alpha * 0.7 * gridM);
      ctx.globalAlpha = 1;
    }
    function drawGrid(lv, a) {
      if (a <= 0.01) return;
      ctx.globalAlpha = a;
      ctx.beginPath();
      for (let i = 1; i < lv.cols; i++) { const x = crH(B.x + (B.w * i) / lv.cols); ctx.moveTo(x, B.y); ctx.lineTo(x, B.y + B.h); }
      for (let j = 1; j < lv.rows; j++) { const y = crH(B.y + (B.h * j) / lv.rows); ctx.moveTo(B.x, y); ctx.lineTo(B.x + B.w, y); }
      ctx.stroke();
    }
    function dissolve(lv, m) {
      if (lv.lastM !== m) {
        const d = lv.mask.data, thr = lv.thr;
        for (let i = 0, n = thr.length; i < n; i++) d[i * 4 + 3] = thr[i] < m ? 255 : 0;
        lv.mctx.putImageData(lv.mask, 0, 0);
        const o = lv.octx;
        o.globalCompositeOperation = "copy"; o.drawImage(lv.c, 0, 0);
        o.globalCompositeOperation = "destination-in"; o.drawImage(lv.mc, 0, 0);
        o.globalCompositeOperation = "source-over";
        lv.lastM = m;
      }
      return lv.out;
    }

    function drawNoise(alpha, tm) {
      if (alpha <= 0.002) return;
      const f = reduced ? 0 : Math.floor(tm / 70) % 4;
      ctx.globalCompositeOperation = "lighter"; ctx.globalAlpha = alpha;
      ctx.drawImage(NOISE, (f & 1) * NZW, (f >> 1) * NZH, NZW, NZH, B.x, B.y, B.w, B.h);
      ctx.globalCompositeOperation = "source-over"; ctx.globalAlpha = 1;
    }

    function clipBand(y0, y1) {
      ctx.save(); ctx.beginPath(); ctx.rect(B.x - 40, y0, B.w + 80, y1 - y0); ctx.clip();
    }

    // Shared per-frame values
    let scanY = 0, tokVis = 0;

    function drawPhoto(p, tm) {
      // Ambient backlight
      ctx.globalCompositeOperation = "lighter";
      ctx.globalAlpha = 0.22 + 0.18 * R(p, 0.78, 0.95);
      const gs = B.h * 1.5;
      ctx.drawImage(SOFT, B.cx - gs / 2, B.cy - gs * 0.55, gs, gs);
      ctx.globalCompositeOperation = "source-over"; ctx.globalAlpha = 1;

      if (!A.baseReady) {
        // Tasteful "no signal" while the photo loads
        ctx.fillStyle = "rgba(255,255,255,0.018)"; ctx.fillRect(B.x, B.y, B.w, B.h);
        drawNoise(0.15, tm);
        ctx.font = FONT_LABEL; setLS("0.14em"); ctx.fillStyle = WHITE; ctx.globalAlpha = 0.35;
        ctx.textAlign = "center"; ctx.textBaseline = "middle";
        ctx.fillText(A.failed ? "NO SIGNAL" : "AWAITING FRAME", B.cx, B.cy);
        ctx.textAlign = "left"; ctx.textBaseline = "alphabetic"; setLS("0px"); ctx.globalAlpha = 1;
        return;
      }

      const scanT = R(p, 0.185, 0.365);
      scanY = B.y + B.h * EZ.inOutSine(scanT);
      const triW = A.triReady ? EZ.inOutCubic(R(p, 0.555, 0.64)) : 0;

      // 1) Raw mosaic (entire frame before the scan; below the scanline during it)
      if (p < 0.37) {
        const top = p >= 0.185 ? scanY : B.y;
        if (top < B.y + B.h - 0.5) {
          const mosA = 1 - 0.42 * R(p, 0.16, 0.2);
          const clipped = top > B.y;
          if (clipped) clipBand(top, B.y + B.h + 2);
          drawMosaic(p, mosA, tm);
          if (clipped) ctx.restore();
        }
      }

      // 2) Processed: dim grey frame + glowing edges above the scanline
      if (p >= 0.185 && triW < 1) {
        const procA = 1 - triW;
        const bot = scanT < 1 ? scanY : B.y + B.h;
        if (bot > B.y + 0.5) {
          const clipped = scanT < 1;
          if (clipped) clipBand(B.y - 2, bot);
          const gA = (0.36 - 0.1 * R(p, 0.38, 0.47)) * procA;
          ctx.globalAlpha = gA; ctx.drawImage(A.G, B.x, B.y, B.w, B.h);
          if (A.edgeReady) {
            const eA = procA * (1 - 0.55 * R(p, 0.39, 0.47));
            ctx.globalCompositeOperation = "lighter";
            ctx.globalAlpha = 0.55 * eA; ctx.drawImage(A.Eg, B.x, B.y, B.w, B.h);
            ctx.globalAlpha = eA; ctx.drawImage(A.E, B.x, B.y, B.w, B.h);
            ctx.globalCompositeOperation = "source-over";
          }
          ctx.globalAlpha = 1;
          if (clipped) ctx.restore();
          // Freshly-processed band right above the line burns brighter
          if (A.edgeReady && scanT > 0 && scanT < 1) {
            clipBand(Math.max(B.y, scanY - 42), scanY);
            ctx.globalCompositeOperation = "lighter"; ctx.globalAlpha = 0.85;
            ctx.drawImage(A.E, B.x, B.y, B.w, B.h);
            ctx.globalCompositeOperation = "source-over"; ctx.globalAlpha = 1;
            ctx.restore();
          }
        }
      }

      // 3) Tri-tone portrait — the "understood" image
      if (triW > 0) {
        const a = triW * (0.66 + 0.34 * EZ.inOutCubic(R(p, 0.78, 0.9)));
        ctx.globalAlpha = a; ctx.drawImage(A.T, B.x, B.y, B.w, B.h); ctx.globalAlpha = 1;
      }

      // 4) Cross-attention while the LLM writes
      const attA = R(p, 0.62, 0.64) * (1 - R(p, 0.75, 0.79));
      if (attA > 0 && tokVis > 0) {
        ctx.globalCompositeOperation = "lighter";
        const cur = Math.min(TOKENS.length - 1, Math.floor(tokVis));
        for (let back = 0; back < 3; back++) {
          const i = cur - back; if (i < 0) break;
          const regs = ATTN[i]; if (!regs) continue;
          const age = tokVis - (i + 1); // <0 while appearing
          const w = back === 0 ? clamp(tokVis - i) : Math.max(0, 1 - age * 0.9);
          if (w <= 0) continue;
          for (let r = 0; r < regs.length; r++) {
            const rr = regs[r], s = rr[2] * B.w * 2.2;
            ctx.globalAlpha = attA * w * 0.5;
            ctx.drawImage(SOFT, B.x + rr[0] * B.w - s / 2, B.y + rr[1] * B.h - s / 2, s, s);
          }
        }
        ctx.globalCompositeOperation = "source-over"; ctx.globalAlpha = 1;
      }

      // 5) Seat the captions (desktop: they overlap the lower-left of the portrait)
      if (!mobile) {
        const seat = 0.55 * (1 - R(p, 0.78, 0.83));
        if (seat > 0) {
          ctx.globalAlpha = seat;
          const sw = (CAP.right - CAP.left) * 1.5, sh = (CAP.bottom - CAP.top) * 2.0;
          ctx.drawImage(SHADE, CAP.left - sw * 0.12, CAP.top - sh * 0.22, sw, sh);
          ctx.globalAlpha = 1;
        }
      }
    }

    function drawScan(p, tm) {
      const sA = R(p, 0.17, 0.195) * (1 - R(p, 0.36, 0.385));
      if (sA <= 0 || !A.baseReady) return;
      const ext = mobile ? 10 : 22;
      const x0 = B.x - ext, w = B.w + ext * 2;
      ctx.globalCompositeOperation = "lighter";
      ctx.globalAlpha = 0.6 * sA; ctx.drawImage(SCANBAND, x0, scanY - 52, w, 52);
      ctx.globalAlpha = sA; ctx.drawImage(SCANLINE, x0, scanY - 0.5, w, 1);
      ctx.globalCompositeOperation = "source-over";

      // Sample window on the strongest edge of this row
      const row = Math.min(EH - 1, Math.max(0, Math.floor(((scanY - B.y) / B.h) * EH)));
      const rows = A.rows;
      const ru = rows ? rows[row * 2] : 0.5, val = rows ? rows[row * 2 + 1] : 0;
      const mx = B.x + ru * B.w, sz = mobile ? 9 : 11;
      ctx.globalCompositeOperation = "lighter"; ctx.globalAlpha = 0.9 * sA;
      ctx.drawImage(GLOW, mx - 14, scanY - 14, 28, 28);
      ctx.globalCompositeOperation = "source-over";
      ctx.strokeStyle = ACC_S; ctx.lineWidth = 1; ctx.globalAlpha = sA;
      ctx.strokeRect(cr1(mx - sz / 2), cr1(scanY - sz / 2), sz, sz);
      ctx.lineWidth = hair(); ctx.globalAlpha = 0.5 * sA;
      ctx.beginPath();
      ctx.moveTo(crH(mx - sz / 6), scanY - sz / 2); ctx.lineTo(crH(mx - sz / 6), scanY + sz / 2);
      ctx.moveTo(crH(mx + sz / 6), scanY - sz / 2); ctx.lineTo(crH(mx + sz / 6), scanY + sz / 2);
      ctx.stroke();

      // 3×3 Sobel kernel readout that rides with the scanline
      const cw = mobile ? 17 : 22, ch = mobile ? 14 : 17;
      const kw = cw * 3, title = mobile ? 0 : 16, valH = mobile ? 15 : 30;
      const bh = title + ch * 3 + valH;
      let rx, ry;
      if (!mobile) {
        rx = B.x + B.w + 30;
        ry = clamp(scanY - title - ch * 1.5, B.y - 10, B.y + B.h - bh + 10);
        // leader from the sampled pixel to the readout
        ctx.strokeStyle = WHITE; ctx.lineWidth = hair(); ctx.globalAlpha = 0.28 * sA;
        ctx.beginPath(); ctx.moveTo(mx + sz / 2 + 3, crH(scanY)); ctx.lineTo(rx - 8, crH(scanY)); ctx.stroke();
      } else {
        rx = B.x + B.w - kw - 12;
        ry = scanY + 10 + bh < B.y + B.h ? scanY + 10 : scanY - bh - 10;
        ctx.globalAlpha = 0.78 * sA; ctx.fillStyle = "rgba(6,7,8,0.85)";
        U.roundRect(ctx, rx - 6, ry - 6, kw + 12, bh + 10, 6); ctx.fill();
      }
      ctx.globalAlpha = sA;
      ctx.textBaseline = "middle";
      if (title) {
        ctx.font = FONT_LABEL; setLS("0.14em"); ctx.fillStyle = WHITE; ctx.globalAlpha = 0.5 * sA;
        ctx.fillText("SOBEL · Gx", rx, ry + 5); setLS("0px");
      }
      const gy0 = ry + title;
      ctx.strokeStyle = WHITE; ctx.lineWidth = hair(); ctx.globalAlpha = 0.16 * sA;
      ctx.beginPath();
      for (let i = 0; i <= 3; i++) {
        const x = crH(rx + i * cw), y = crH(gy0 + i * ch);
        ctx.moveTo(x, gy0); ctx.lineTo(x, gy0 + ch * 3);
        ctx.moveTo(rx, y); ctx.lineTo(rx + kw, y);
      }
      ctx.stroke();
      ctx.font = mobile ? FONT_SMALL : FONT_KERNEL; ctx.textAlign = "center";
      for (let i = 0; i < 9; i++) {
        const v = KERNEL[i], col = i % 3;
        ctx.fillStyle = col === 2 ? ACC_S : WHITE;
        ctx.globalAlpha = sA * (col === 1 ? 0.3 : col === 2 ? 0.95 : 0.6);
        ctx.fillText(v, rx + (col + 0.5) * cw, gy0 + (Math.floor(i / 3) + 0.5) * ch + 0.5);
      }
      ctx.textAlign = "left";
      const vy = gy0 + ch * 3 + (mobile ? 9 : 10);
      ctx.font = mobile ? FONT_SMALL : FONT_KERNEL;
      ctx.fillStyle = WHITE; ctx.globalAlpha = 0.5 * sA; ctx.fillText("|∇|", rx, vy);
      ctx.fillStyle = ACC_S; ctx.globalAlpha = sA; ctx.fillText(CONF[Math.round(val * 100)], rx + (mobile ? 22 : 26), vy);
      if (!mobile) {
        ctx.font = FONT_SMALL; ctx.fillStyle = WHITE; ctx.globalAlpha = 0.32 * sA;
        ctx.fillText(ROWSTR[row], rx, vy + 15);
      }
      ctx.textBaseline = "alphabetic"; ctx.globalAlpha = 1;
    }

    function drawViewfinder(p, tm) {
      const o = mobile ? 8 : 12, arm = mobile ? 14 : 20;
      const x0 = cr1(B.x - o), y0 = cr1(B.y - o), x1 = cr1(B.x + B.w + o), y1 = cr1(B.y + B.h + o);
      const a = (p < 0.18 ? 1 : lerp(1, 0.5, R(p, 0.18, 0.24))) * (1 - 0.62 * R(p, 0.78, 0.86));
      // Corner brackets
      ctx.strokeStyle = WHITE; ctx.lineWidth = 1; ctx.globalAlpha = 0.72 * a;
      ctx.beginPath();
      ctx.moveTo(x0, y0 + arm); ctx.lineTo(x0, y0); ctx.lineTo(x0 + arm, y0);
      ctx.moveTo(x1 - arm, y0); ctx.lineTo(x1, y0); ctx.lineTo(x1, y0 + arm);
      ctx.moveTo(x1, y1 - arm); ctx.lineTo(x1, y1); ctx.lineTo(x1 - arm, y1);
      ctx.moveTo(x0 + arm, y1); ctx.lineTo(x0, y1); ctx.lineTo(x0, y1 - arm);
      ctx.stroke();
      // Rule of thirds + centre mark (raw-capture beat only)
      const ta = 1 - R(p, 0.15, 0.2);
      if (ta > 0) {
        // Rule of thirds: hairlines in the frame margin + crosses at the intersections
        ctx.lineWidth = hair(); ctx.globalAlpha = 0.07 * ta;
        ctx.beginPath();
        for (let i = 1; i < 3; i++) {
          const x = crH(B.x + (B.w * i) / 3), y = crH(B.y + (B.h * i) / 3);
          ctx.moveTo(x, B.y); ctx.lineTo(x, B.y + B.h); ctx.moveTo(B.x, y); ctx.lineTo(B.x + B.w, y);
        }
        ctx.stroke();
        ctx.globalAlpha = 0.5 * ta; ctx.lineWidth = 1;
        ctx.beginPath();
        for (let i = 1; i < 3; i++) {
          const x = cr1(B.x + (B.w * i) / 3);
          ctx.moveTo(x, y0 - 4); ctx.lineTo(x, y0 + 4); ctx.moveTo(x, y1 - 4); ctx.lineTo(x, y1 + 4);
          const y = cr1(B.y + (B.h * i) / 3);
          ctx.moveTo(x0 - 4, y); ctx.lineTo(x0 + 4, y); ctx.moveTo(x1 - 4, y); ctx.lineTo(x1 + 4, y);
          for (let j = 1; j < 3; j++) {
            const yy = cr1(B.y + (B.h * j) / 3), c = 4;
            ctx.moveTo(x - c, yy); ctx.lineTo(x + c, yy); ctx.moveTo(x, yy - c); ctx.lineTo(x, yy + c);
          }
        }
        ctx.stroke();
        // REC
        const blink = reduced ? 1 : 0.3 + 0.7 * (0.5 + 0.5 * Math.cos(tm * 0.0055));
        const rx = B.x + B.w - 16, ry = B.y + 18;
        ctx.font = FONT_LABEL; setLS("0.16em"); ctx.textAlign = "right"; ctx.textBaseline = "middle";
        ctx.fillStyle = WHITE; ctx.globalAlpha = 0.8 * ta; ctx.fillText("REC", rx, ry);
        const tw = recW;
        ctx.textAlign = "left"; ctx.textBaseline = "alphabetic"; setLS("0px");
        ctx.globalCompositeOperation = "lighter"; ctx.globalAlpha = 0.6 * ta * blink;
        ctx.drawImage(GLOW, rx - tw - 12 - 9, ry - 9, 18, 18);
        ctx.globalCompositeOperation = "source-over";
        ctx.fillStyle = ACC_S; ctx.globalAlpha = ta * blink;
        ctx.beginPath(); ctx.arc(rx - tw - 12, ry, 3, 0, Math.PI * 2); ctx.fill();
      }
      // Stage label + pipeline segments (above the frame on desktop, below on mobile)
      const ly = mobile ? y1 + 17 : y0 - 11;
      let s = 0; while (s < 4 && p >= STAGE_AT[s + 1]) s++;
      const la = R(p, STAGE_AT[s] - 0.0001, STAGE_AT[s] + 0.02) * (s === 4 ? 0.55 : 1);
      ctx.font = FONT_LABEL; setLS("0.08em"); ctx.fillStyle = WHITE;
      ctx.globalAlpha = 0.55 * (s === 0 ? 1 : la);
      ctx.fillText(STAGE_LABELS[s], x0, ly);
      setLS("0px");
      const segW = mobile ? 12 : 16, segG = 4, n = 5, sx0 = x1 - (n * segW + (n - 1) * segG);
      const fa = s === 4 ? 0.5 : 0.9;
      for (let i = 0; i < n && segFit; i++) {
        const sx = sx0 + i * (segW + segG), lp = R(p, STAGE_AT[i], STAGE_AT[i + 1]);
        ctx.fillStyle = WHITE; ctx.globalAlpha = 0.16 * fa; ctx.fillRect(sx, ly - 5, segW, 2);
        if (lp > 0) { ctx.fillStyle = i === s ? ACC_S : WHITE; ctx.globalAlpha = (i === s ? 0.95 : 0.45) * fa; ctx.fillRect(sx, ly - 5, segW * lp, 2); }
      }
      ctx.globalAlpha = 1;
    }

    function bez(o, u, axis) { // cubic bezier coordinate at u for link o (axis 0 = x, 1 = y)
      const d = LK.d, m = 1 - u;
      return m * m * m * d[o + axis] + 3 * m * m * u * d[o + 2 + axis] + 3 * m * u * u * d[o + 4 + axis] + u * u * u * d[o + 6 + axis];
    }

    function drawNetwork(p, tm) {
      if (p < 0.378 || p > 0.605) return;
      const fadeOut = 1 - R(p, 0.568, 0.6);
      const netA = R(p, 0.44, 0.48) * fadeOut;
      const F = R(p, 0.462, 0.55) * 6;
      const d = LK.d;

      if (netA > 0) {
        // Base wiring
        ctx.strokeStyle = WHITE; ctx.lineWidth = hair(); ctx.globalAlpha = 0.13 * netA;
        ctx.beginPath();
        for (let l = 0; l < LK.n; l++) { const o = l * 9; ctx.moveTo(d[o], d[o + 1]); ctx.bezierCurveTo(d[o + 2], d[o + 3], d[o + 4], d[o + 5], d[o + 6], d[o + 7]); }
        ctx.stroke();
        // Lit wiring (partial curves up to the activation front)
        ctx.strokeStyle = ACC_S; ctx.lineWidth = 1; ctx.globalAlpha = 0.34 * netA;
        ctx.beginPath();
        for (let l = 0; l < LK.n; l++) {
          const o = l * 9, u = clamp(F - d[o + 8]);
          if (u <= 0) continue;
          if (u >= 1) { ctx.moveTo(d[o], d[o + 1]); ctx.bezierCurveTo(d[o + 2], d[o + 3], d[o + 4], d[o + 5], d[o + 6], d[o + 7]); continue; }
          // de Casteljau split at u
          const x0 = d[o], y0 = d[o + 1], x1 = d[o + 2], y1 = d[o + 3], x2 = d[o + 4], y2 = d[o + 5], x3 = d[o + 6], y3 = d[o + 7];
          const ax = x0 + (x1 - x0) * u, ay = y0 + (y1 - y0) * u, bx = x1 + (x2 - x1) * u, by = y1 + (y2 - y1) * u;
          const cx = x2 + (x3 - x2) * u, cy = y2 + (y3 - y2) * u, abx = ax + (bx - ax) * u, aby = ay + (by - ay) * u;
          const bcx = bx + (cx - bx) * u, bcy = by + (cy - by) * u, ex = abx + (bcx - abx) * u, ey = aby + (bcy - aby) * u;
          ctx.moveTo(x0, y0); ctx.bezierCurveTo(ax, ay, abx, aby, ex, ey);
        }
        ctx.stroke();
        // Activation pulses at the front + idle pulses on lit wires
        ctx.globalCompositeOperation = "lighter";
        const ps = 16 * Math.max(0.7, netS);
        for (let l = 0; l < LK.n; l++) {
          const o = l * 9, u = clamp(F - d[o + 8]);
          if (u > 0 && u < 1) {
            ctx.globalAlpha = netA; ctx.drawImage(GLOW, bez(o, u, 0) - ps / 2, bez(o, u, 1) - ps / 2, ps, ps);
          } else if (u >= 1 && !reduced) {
            const v = (tm * 0.00032 + LK.ph[l]) % 1;
            ctx.globalAlpha = 0.4 * netA * Math.sin(v * Math.PI);
            ctx.drawImage(GLOW, bez(o, v, 0) - ps * 0.3, bez(o, v, 1) - ps * 0.3, ps * 0.6, ps * 0.6);
          }
        }
        ctx.globalCompositeOperation = "source-over"; ctx.globalAlpha = 1;
      }

      // Feature-map tiles: peel off the portrait into their columns
      if (A.tilesReady || netA > 0) {
        for (let k = 0; k < N_TILES; k++) {
          const nd = TILE[k];
          const t0 = 0.382 + k * 0.005;
          const tk = EZ.inOutCubic(R(p, t0, t0 + 0.046));
          if (tk <= 0 && netA <= 0) continue;
          const x = lerp(B.x, nd.x, tk), y = lerp(B.y, nd.y, tk), w = lerp(B.w, nd.w, tk), h = lerp(B.h, nd.h, tk);
          const lit = R(F, nd.col + 0.8, nd.col + 1.05);
          const a = smooth(0.08, 0.6, tk) * fadeOut; // faint while still portrait-sized
          if (A.tilesReady && tk > 0) {
            const tl = A.tiles[k];
            if (!tl) continue;
            ctx.imageSmoothingEnabled = !tl.px;
            ctx.globalAlpha = a * (0.62 + 0.38 * lit);
            ctx.drawImage(tl.c, x, y, w, h);
            ctx.imageSmoothingEnabled = true;
          }
          const ba = tk > 0 ? a : netA * 0.5;
          ctx.lineWidth = hair();
          ctx.strokeStyle = WHITE; ctx.globalAlpha = ba * 0.22 * (1 - lit);
          ctx.strokeRect(crH(x), crH(y), w, h);
          if (lit > 0) {
            ctx.strokeStyle = ACC_S; ctx.lineWidth = 1; ctx.globalAlpha = ba * 0.85 * lit;
            ctx.strokeRect(cr1(x), cr1(y), w, h);
          }
        }
        ctx.globalAlpha = 1;
      }

      if (netA <= 0) return;
      // fc units + output node
      const litFc = R(F, 4.8, 5.05), litOut = R(F, 5.8, 6);
      for (let i = 0; i < DOTS.length; i++) {
        const dd = DOTS[i], r = 3.2 * Math.max(0.75, netS);
        ctx.fillStyle = WHITE; ctx.globalAlpha = 0.28 * netA * (1 - litFc);
        ctx.beginPath(); ctx.arc(dd.x, dd.y, r, 0, Math.PI * 2); ctx.fill();
        if (litFc > 0) {
          ctx.fillStyle = ACC_S; ctx.globalAlpha = netA * litFc * (0.55 + 0.45 * ((i * 0.37) % 1));
          ctx.beginPath(); ctx.arc(dd.x, dd.y, r, 0, Math.PI * 2); ctx.fill();
        }
      }
      ctx.strokeStyle = litOut > 0 ? ACC_S : WHITE; ctx.lineWidth = 1; ctx.globalAlpha = netA * (0.35 + 0.65 * litOut);
      ctx.beginPath(); ctx.arc(OUT.x, OUT.y, OUT.r, 0, Math.PI * 2); ctx.stroke();
      if (litOut > 0) {
        ctx.globalCompositeOperation = "lighter"; ctx.globalAlpha = netA * litOut;
        const gsz = OUT.r * 7; ctx.drawImage(GLOW, OUT.x - gsz / 2, OUT.y - gsz / 2, gsz, gsz);
        ctx.globalCompositeOperation = "source-over";
        ctx.fillStyle = ACC_S; ctx.beginPath(); ctx.arc(OUT.x, OUT.y, OUT.r * 0.45, 0, Math.PI * 2); ctx.fill();
        // class readout
        ctx.textBaseline = "middle";
        ctx.font = FONT_LABEL; setLS("0.12em"); ctx.fillStyle = WHITE; ctx.globalAlpha = 0.6 * netA * litOut;
        const lx = OUT.x + OUT.r + 10;
        ctx.fillText(mobile ? "PERSON" : "PERSON", lx, OUT.y - 7);
        setLS("0px"); ctx.font = FONT_KERNEL; ctx.fillStyle = ACC_S; ctx.globalAlpha = netA * litOut;
        ctx.fillText("0.99", lx, OUT.y + 8);
        ctx.textBaseline = "alphabetic";
      }
      // column labels
      if (!colLabelsFit) { ctx.globalAlpha = 1; return; }
      ctx.font = FONT_LABEL; setLS("0.12em"); ctx.fillStyle = WHITE; ctx.textAlign = "center";
      for (let ci = 0; ci < NET.length; ci++) {
        const on = R(F, ci + 0.8, ci + 1.05);
        ctx.globalAlpha = netA * (0.32 + 0.38 * on);
        ctx.fillText(NET[ci].label, COLX[ci], colLabelY);
      }
      ctx.textAlign = "left"; setLS("0px"); ctx.globalAlpha = 1;
    }

    function drawTerminal(p, tm) {
      const inA = EZ.outCubic(R(p, 0.588, 0.618)), outA = 1 - R(p, 0.765, 0.795);
      const a = inA * outA;
      if (a <= 0) return;
      const T = TERM, x = T.x, y = T.y + (1 - inA) * 10;
      // Seat + panel
      ctx.globalAlpha = 0.65 * a;
      ctx.drawImage(SHADE, x - T.w * 0.15, y - T.h * 0.3, T.w * 1.3, T.h * 1.6);
      ctx.globalAlpha = a; ctx.fillStyle = "rgba(10,12,14,0.86)";
      U.roundRect(ctx, x, y, T.w, T.h, 12); ctx.fill();
      ctx.strokeStyle = WHITE; ctx.lineWidth = hair(); ctx.globalAlpha = 0.14 * a;
      U.roundRect(ctx, crH(x), crH(y), T.w, T.h, 12); ctx.stroke();
      // Header
      const hx = x + T.pad, hy = y + T.pad + 6;
      ctx.textBaseline = "middle";
      ctx.font = FONT_LABEL; setLS("0.14em");
      ctx.fillStyle = ACC_S; ctx.globalAlpha = 0.9 * a;
      ctx.beginPath(); ctx.arc(hx + 3, hy, 2.5, 0, Math.PI * 2); ctx.fill();
      ctx.fillStyle = WHITE; ctx.globalAlpha = 0.55 * a; ctx.fillText(T.compact ? "LLM" : "LLM · VISION → TEXT", hx + 12, hy);
      setLS("0px"); ctx.font = FONT_SMALL; ctx.textAlign = "right"; ctx.globalAlpha = 0.4 * a;
      ctx.fillText((T.compact ? TOKSHORT : TOKCOUNT)[Math.floor(tokVis)], x + T.w - T.pad, hy);
      ctx.textAlign = "left";
      ctx.globalAlpha = 0.09 * a; ctx.fillStyle = WHITE;
      ctx.fillRect(x + 1, crH(y + T.pad + T.head - 8), T.w - 2, hair());
      // Prompt
      ctx.font = T.font;
      const py = y + T.promptY;
      const typed = Math.floor(R(p, 0.592, 0.618) * PROMPT.length + 0.0001);
      ctx.fillStyle = ACC_S; ctx.globalAlpha = a; ctx.fillText(">", hx, py);
      ctx.fillStyle = WHITE; ctx.globalAlpha = 0.62 * a;
      const px0 = hx + T.charW * 2;
      ctx.fillText(PREFIX[typed], px0, py);
      // Tokens with faint pill boundaries
      const ty0 = py + T.lh;
      const full = Math.floor(tokVis), frac = tokVis - full;
      const pillH = T.fs + 7;
      for (let i = 0; i < TOKENS.length && i <= full; i++) {
        const ta = i < full ? 1 : frac;
        if (ta <= 0) break;
        const tx = hx + T.tx[i], tyy = ty0 + T.ty[i] * T.lh;
        const newest = i === Math.min(full, TOKENS.length - 1) || (i === full - 1 && frac < 0.35);
        ctx.fillStyle = ACC_S; ctx.globalAlpha = a * ta * (newest ? 0.2 : 0.085);
        U.roundRect(ctx, tx - 2.5, tyy - pillH / 2, T.tw[i] + 5, pillH, 4); ctx.fill();
        ctx.fillStyle = WHITE; ctx.globalAlpha = a * ta * 0.95;
        ctx.fillText(TOK_TXT[i], tx, tyy + (1 - ta) * 3);
      }
      // Cursor
      let cx, cy;
      if (tokVis <= 0) { cx = px0 + T.charW * typed + 3; cy = py; }
      else { const li = Math.min(TOKENS.length - 1, full - (frac > 0 ? 0 : 1)); const j = Math.max(0, li); cx = hx + T.tx[j] + T.tw[j] + 4; cy = ty0 + T.ty[j] * T.lh; }
      const on = reduced || Math.floor(tm / 530) % 2 === 0;
      if (on) { ctx.fillStyle = ACC_S; ctx.globalAlpha = 0.85 * a; ctx.fillRect(cx, cy - T.fs * 0.62, Math.max(2, T.charW * 0.55), T.fs * 1.24); }
      // Footer once done
      const done = R(p, 0.742, 0.756);
      if (done > 0) {
        ctx.font = FONT_SMALL; ctx.fillStyle = WHITE; ctx.globalAlpha = 0.38 * a * done;
        ctx.fillText("■ stop · 13 tokens · 0.21 s", hx, y + T.footY);
      }
      ctx.textBaseline = "alphabetic"; ctx.globalAlpha = 1;
    }

    function drawDetection(p, tm) {
      if (p < 0.778) return;
      const t = R(p, 0.785, 0.84), e = EZ.outBack(t);
      const a = R(p, 0.778, 0.795);
      const settle = R(p, 0.86, 0.96);
      const o = mobile ? 8 : 12;
      const x0 = lerp(B.x - o, FR.x0, e), y0 = lerp(B.y - o, FR.y0, e), x1 = lerp(B.x + B.w + o, FR.x1, e), y1 = lerp(B.y + B.h + o, FR.y1, e);
      const w = x1 - x0, h = y1 - y0, arm = Math.min(w, h) * 0.2;
      // Landmarks + mesh flash in, then fade, leaving the face clean
      const show = R(p, 0.832, 0.866) * (1 - R(p, 0.9, 0.948));
      if (show > 0) {
        const n = LANDMARKS.length;
        const jit = reduced ? 0 : 0.35;
        // "analysis mode": the face dims a touch so the mesh reads on bright skin
        const fw = (FR.x1 - FR.x0) * 1.7, fh = (FR.y1 - FR.y0) * 1.45;
        ctx.globalAlpha = 0.5 * show;
        ctx.drawImage(SHADE, (FR.x0 + FR.x1) / 2 - fw / 2, (FR.y0 + FR.y1) / 2 - fh / 2, fw, fh);
        ctx.strokeStyle = WHITE; ctx.lineWidth = hair(); ctx.globalAlpha = 0.5 * show;
        ctx.beginPath();
        for (let i = 0; i < MESH.length; i += 2) {
          const a1 = MESH[i], b1 = MESH[i + 1];
          ctx.moveTo(LMX[a1], LMY[a1]); ctx.lineTo(LMX[b1], LMY[b1]);
        }
        ctx.stroke();
        ctx.globalCompositeOperation = "lighter";
        for (let i = 0; i < n; i++) {
          const pa = clamp(show * 1.8 - (i / n) * 0.8);
          if (pa <= 0) continue;
          const jx = jit * Math.sin(tm * 0.021 + i * 1.7), jy = jit * Math.cos(tm * 0.017 + i * 2.3);
          ctx.globalAlpha = 0.55 * pa; ctx.drawImage(GLOW, LMX[i] + jx - 6, LMY[i] + jy - 6, 12, 12);
        }
        ctx.globalCompositeOperation = "source-over";
        ctx.fillStyle = WHITE;
        for (let i = 0; i < n; i++) {
          const pa = clamp(show * 1.8 - (i / n) * 0.8);
          if (pa <= 0) continue;
          const jx = jit * Math.sin(tm * 0.021 + i * 1.7), jy = jit * Math.cos(tm * 0.017 + i * 2.3);
          ctx.globalAlpha = pa; ctx.beginPath(); ctx.arc(LMX[i] + jx, LMY[i] + jy, 1.5, 0, Math.PI * 2); ctx.fill();
        }
      }
      const flash = t >= 1 ? 1 - R(p, 0.84, 0.88) : 0;
      // thin full rect
      ctx.strokeStyle = ACC_S; ctx.lineWidth = hair(); ctx.globalAlpha = a * (0.3 - 0.12 * settle);
      ctx.strokeRect(crH(x0), crH(y0), w, h);
      // corner brackets
      ctx.lineWidth = 1.5; ctx.globalAlpha = a * (1 - 0.42 * settle);
      const X0 = cr1(x0), Y0 = cr1(y0), X1 = cr1(x1), Y1 = cr1(y1);
      ctx.beginPath();
      ctx.moveTo(X0, Y0 + arm); ctx.lineTo(X0, Y0); ctx.lineTo(X0 + arm, Y0);
      ctx.moveTo(X1 - arm, Y0); ctx.lineTo(X1, Y0); ctx.lineTo(X1, Y0 + arm);
      ctx.moveTo(X1, Y1 - arm); ctx.lineTo(X1, Y1); ctx.lineTo(X1 - arm, Y1);
      ctx.moveTo(X0 + arm, Y1); ctx.lineTo(X0, Y1); ctx.lineTo(X0, Y1 - arm);
      ctx.stroke();
      if (flash > 0) {
        ctx.globalCompositeOperation = "lighter"; ctx.globalAlpha = 0.5 * flash;
        const gz = 26;
        ctx.drawImage(GLOW, X0 - gz / 2, Y0 - gz / 2, gz, gz); ctx.drawImage(GLOW, X1 - gz / 2, Y0 - gz / 2, gz, gz);
        ctx.drawImage(GLOW, X0 - gz / 2, Y1 - gz / 2, gz, gz); ctx.drawImage(GLOW, X1 - gz / 2, Y1 - gz / 2, gz, gz);
        ctx.globalCompositeOperation = "source-over";
      }

      // Label chip: wipes in, then relaxes from filled to outlined
      const rev = EZ.outCubic(R(p, 0.812, 0.845));
      if (rev > 0) {
        const ch = CHIP.h, cw = CHIP.w;
        let cx = FR.x0; if (cx + cw > W - 8) cx = W - 8 - cw;
        const cy = FR.y0 - 8 - ch;
        const conf = CONF[Math.round(99 * EZ.outCubic(R(p, 0.815, 0.86)))];
        const sub = R(p, 0.88, 0.96);
        ctx.save();
        ctx.beginPath(); ctx.rect(cx - 1, cy - 1, cw * rev + 2, ch + 2); ctx.clip();
        ctx.font = FONT_CHIP; ctx.textBaseline = "middle";
        if (sub < 1) {
          ctx.globalAlpha = (1 - sub) * 0.95; ctx.fillStyle = ACC_S;
          U.roundRect(ctx, cx, cy, cw, ch, 4); ctx.fill();
          ctx.fillStyle = INK; ctx.fillText(CHIP_PREFIX, cx + 9, cy + ch / 2 + 0.5); ctx.fillText(conf, cx + 9 + CHIP.prefixW, cy + ch / 2 + 0.5);
        }
        if (sub > 0) {
          ctx.globalAlpha = sub * 0.8; ctx.fillStyle = "rgba(6,7,8,0.7)";
          U.roundRect(ctx, cx, cy, cw, ch, 4); ctx.fill();
          ctx.strokeStyle = ACC_S; ctx.lineWidth = hair(); ctx.globalAlpha = sub * 0.4;
          U.roundRect(ctx, crH(cx), crH(cy), cw, ch, 4); ctx.stroke();
          ctx.fillStyle = WHITE; ctx.globalAlpha = sub * 0.78; ctx.fillText(CHIP_PREFIX, cx + 9, cy + ch / 2 + 0.5);
          ctx.fillStyle = ACC_S; ctx.globalAlpha = sub * 0.95; ctx.fillText(conf, cx + 9 + CHIP.prefixW, cy + ch / 2 + 0.5);
        }
        ctx.restore();
        ctx.textBaseline = "alphabetic";
      }

      ctx.globalAlpha = 1;
    }

    /* ======================================================================
       Contract
       ====================================================================== */
    return {
      resize(width, height, dpr) {
        W = width; H = height; DPR = dpr;
        layout();
      },
      render(progress, timeMs) {
        if (!W || !H) return;
        if (layoutDirty) layout();
        const p = clamp(progress), tm = reduced ? 0 : timeMs;
        ctx.setTransform(DPR, 0, 0, DPR, 0, 0);
        ctx.globalCompositeOperation = "source-over"; ctx.globalAlpha = 1;
        ctx.fillStyle = BG; ctx.fillRect(0, 0, W, H);
        tokVis = R(p, 0.625, 0.74) * TOKENS.length;
        drawPhoto(p, tm);
        drawScan(p, tm);
        drawViewfinder(p, tm);
        drawNetwork(p, tm);
        drawTerminal(p, tm);
        drawDetection(p, tm);
        ctx.globalAlpha = 1; ctx.globalCompositeOperation = "source-over";
      },
      destroy() { dead = true; },
    };
  };
})();
