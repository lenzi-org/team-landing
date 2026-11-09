/* ==========================================================================
   SceneUtils — shared helpers for every canvas scene.
   Loaded first. Scenes register factories on window.SceneRegistry.

   SCENE CONTRACT
   --------------
   window.SceneRegistry.<name> = function createScene(canvas, opts) {
     // opts = {
     //   el:            the <section data-scene> element
     //   accent:        "#38d9f5"           (hex string)
     //   accentRgb:     [56, 217, 245]
     //   portraitSrc:   "assets/team/x.webp" | null
     //   face:          [x0, y0, x1, y1] normalized face box in the portrait | null
     //   reducedMotion: boolean
     //   utils:         SceneUtils
     // }
     return {
       resize(width, height, dpr) {},   // CSS px; canvas backing store already set to width*dpr
       render(progress, timeMs, dtMs) {}, // progress 0..1 across the pinned section
       destroy() {}                     // optional
     };
   };
   The engine (main.js) owns sizing, visibility and the rAF loop. Scenes must:
   - call ctx.setTransform(dpr, 0, 0, dpr, 0, 0) and clear at the start of render()
   - be a pure-ish function of progress (scrolling back must rewind cleanly);
     time may add ambient motion only
   - when reducedMotion is true, ignore time (render a still per progress)
   ========================================================================== */
(function () {
  "use strict";

  const U = {};

  U.clamp = (v, a = 0, b = 1) => Math.min(b, Math.max(a, v));
  U.lerp = (a, b, t) => a + (b - a) * t;
  /** Local 0..1 progress of p inside [a, b]. */
  U.range = (p, a, b) => U.clamp((p - a) / (b - a));
  U.ease = {
    linear: (t) => t,
    inCubic: (t) => t * t * t,
    outCubic: (t) => 1 - Math.pow(1 - t, 3),
    inOutCubic: (t) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2),
    outExpo: (t) => (t === 1 ? 1 : 1 - Math.pow(2, -10 * t)),
    inOutSine: (t) => -(Math.cos(Math.PI * t) - 1) / 2,
    outBack: (t) => { const c1 = 1.70158, c3 = c1 + 1; return 1 + c3 * Math.pow(t - 1, 3) + c1 * Math.pow(t - 1, 2); },
  };

  U.hexToRgb = (hex) => {
    const h = hex.replace("#", "");
    const n = parseInt(h.length === 3 ? h.split("").map((c) => c + c).join("") : h, 16);
    return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
  };
  U.rgba = (rgb, a) => `rgba(${rgb[0]},${rgb[1]},${rgb[2]},${a})`;
  U.mixRgb = (a, b, t) => [U.lerp(a[0], b[0], t), U.lerp(a[1], b[1], t), U.lerp(a[2], b[2], t)];

  /** Deterministic PRNG (mulberry32) so layouts don't jump between frames/reloads. */
  U.rng = (seed = 1) => {
    let s = seed >>> 0;
    return () => {
      s = (s + 0x6d2b79f5) >>> 0;
      let t = s;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  };

  U.isMobile = (w) => w < 900;

  /**
   * Where the portrait lives inside a team scene (CSS px). Matches the CSS
   * layout in main.css: desktop → left-center, person card on the right;
   * mobile → top half, person card in the bottom ~42%.
   * Returns a 4:5 box { x, y, w, h, cx, cy }.
   */
  U.portraitBox = (W, H) => {
    let bw, bh, cx, cy;
    if (!U.isMobile(W)) {
      bh = Math.min(H * 0.66, W * 0.42 * 1.25);
      bw = bh * 0.8;
      cx = W * 0.32;
      cy = H * 0.5;
    } else {
      bw = Math.min(W * 0.72, H * 0.42 * 0.8);
      bh = bw * 1.25;
      cx = W * 0.5;
      cy = 64 + (H * 0.5 - 64) * 0.5 + 8; // centered in the area between nav and the card
      cy = Math.max(cy, 96 + bh / 2);     // keep clear of the fixed nav (~72px)
    }
    return { x: cx - bw / 2, y: cy - bh / 2, w: bw, h: bh, cx, cy };
  };

  const imageCache = new Map();
  U.loadImage = (src) => {
    if (!src) return Promise.reject(new Error("no src"));
    if (imageCache.has(src)) return imageCache.get(src);
    const p = new Promise((resolve, reject) => {
      const img = new Image();
      img.decoding = "async";
      img.onload = () => resolve(img);
      img.onerror = reject;
      img.src = src;
    });
    imageCache.set(src, p);
    return p;
  };

  /**
   * Read pixels of an image scaled to (cols x rows). Returns ImageData or null
   * when the canvas is tainted (e.g. opened via file://). Always handle null.
   */
  U.readPixels = (img, cols, rows) => {
    try {
      const c = document.createElement("canvas");
      c.width = cols; c.height = rows;
      const x = c.getContext("2d", { willReadFrequently: true });
      x.drawImage(img, 0, 0, cols, rows);
      return x.getImageData(0, 0, cols, rows);
    } catch (e) {
      return null;
    }
  };

  /**
   * Sample particle targets from an image on a grid. Returns
   * [{ u, v, lum, r, g, b }] with u,v in 0..1 (image space), skipping
   * near-black cells when minLum > 0. Falls back to a synthetic oval
   * silhouette when pixels can't be read.
   */
  U.samplePoints = (img, cols, rows, minLum = 0.0, seed = 7) => {
    const data = U.readPixels(img, cols, rows);
    const out = [];
    const rnd = U.rng(seed);
    for (let j = 0; j < rows; j++) {
      for (let i = 0; i < cols; i++) {
        const u = (i + 0.5) / cols, v = (j + 0.5) / rows;
        let r, g, b, lum;
        if (data) {
          const k = (j * cols + i) * 4;
          r = data.data[k]; g = data.data[k + 1]; b = data.data[k + 2];
          lum = (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255;
        } else {
          // Fallback silhouette: head + shoulders
          const head = Math.hypot((u - 0.5) / 0.22, (v - 0.36) / 0.27);
          const body = Math.hypot((u - 0.5) / 0.48, (v - 1.02) / 0.42);
          lum = Math.max(head < 1 ? 0.8 - head * 0.3 : 0, body < 1 ? 0.5 - body * 0.2 : 0);
          r = g = b = lum * 255;
        }
        if (lum >= minLum) out.push({ u, v, lum, r, g, b, j: rnd() });
      }
    }
    return out;
  };

  /**
   * Prepare the shared portrait look: tri-tone (shadow → accent → near-white)
   * plus a soft elliptical alpha mask so every photo melts into the dark page,
   * regardless of its original background. Returns a canvas (cached).
   */
  const portraitCache = new Map();
  U.portraitCanvas = (img, accentRgb, opts = {}) => {
    const key = img.src + "|" + accentRgb.join(",") + "|" + JSON.stringify(opts);
    if (portraitCache.has(key)) return portraitCache.get(key);
    const w = opts.width || 880, h = Math.round(w * 1.25);
    const c = document.createElement("canvas");
    c.width = w; c.height = h;
    const x = c.getContext("2d", { willReadFrequently: true });
    // cover-fit the source into 4:5
    U.drawCover(x, img, 0, 0, w, h);
    const dark = [6, 7, 8], light = U.mixRgb(accentRgb, [255, 255, 255], 0.82);
    let ok = true;
    try {
      const id = x.getImageData(0, 0, w, h), d = id.data;
      const contrast = opts.contrast ?? 1.18, lift = opts.lift ?? 0;
      // 256-entry lookup table: same tri-tone curve, no per-pixel allocation
      const lr = new Uint8ClampedArray(256), lg = new Uint8ClampedArray(256), lb = new Uint8ClampedArray(256);
      for (let i = 0; i < 256; i++) {
        const l = U.clamp((i / 255 - 0.5) * contrast + 0.5 + lift);
        let a, b, t;
        if (l < 0.5) { a = dark; b = accentRgb; t = Math.pow(l * 2, 1.25); }
        else { a = accentRgb; b = light; t = (l - 0.5) * 2; }
        lr[i] = a[0] + (b[0] - a[0]) * t;
        lg[i] = a[1] + (b[1] - a[1]) * t;
        lb[i] = a[2] + (b[2] - a[2]) * t;
      }
      for (let k = 0; k < d.length; k += 4) {
        const i = (0.2126 * d[k] + 0.7152 * d[k + 1] + 0.0722 * d[k + 2]) | 0;
        d[k] = lr[i]; d[k + 1] = lg[i]; d[k + 2] = lb[i];
      }
      x.putImageData(id, 0, 0);
    } catch (e) {
      ok = false; // tainted: approximate with blend modes
      x.globalCompositeOperation = "saturation"; x.fillStyle = "#000"; x.fillRect(0, 0, w, h);
      x.globalCompositeOperation = "multiply"; x.fillStyle = U.rgba(accentRgb, 1); x.fillRect(0, 0, w, h);
      x.globalCompositeOperation = "source-over";
    }
    if (opts.mask !== false) {
      // Soft vignette alpha mask centred on the face
      x.globalCompositeOperation = "destination-in";
      const g = x.createRadialGradient(w * 0.5, h * 0.4, w * 0.18, w * 0.5, h * 0.46, w * 0.72);
      g.addColorStop(0, "rgba(0,0,0,1)");
      g.addColorStop(0.62, "rgba(0,0,0,0.92)");
      g.addColorStop(1, "rgba(0,0,0,0)");
      x.fillStyle = g; x.fillRect(0, 0, w, h);
      x.globalCompositeOperation = "source-over";
    }
    c._duotoneOk = ok;
    portraitCache.set(key, c);
    return c;
  };

  /** Draw an image so it covers the box (like object-fit: cover). */
  U.drawCover = (ctx, img, x, y, w, h) => {
    const iw = img.naturalWidth || img.width, ih = img.naturalHeight || img.height;
    const s = Math.max(w / iw, h / ih);
    const sw = w / s, sh = h / s;
    ctx.drawImage(img, (iw - sw) / 2, (ih - sh) / 2, sw, sh, x, y, w, h);
  };

  /** Pre-rendered radial glow sprite (fast alternative to shadowBlur). */
  const spriteCache = new Map();
  U.glowSprite = (rgb, size = 64, core = 0.18) => {
    const key = rgb.join(",") + "|" + size + "|" + core;
    if (spriteCache.has(key)) return spriteCache.get(key);
    const c = document.createElement("canvas");
    c.width = c.height = size;
    const x = c.getContext("2d");
    const g = x.createRadialGradient(size / 2, size / 2, 0, size / 2, size / 2, size / 2);
    g.addColorStop(0, "rgba(255,255,255,1)");
    g.addColorStop(core, U.rgba(rgb, 0.9));
    g.addColorStop(0.45, U.rgba(rgb, 0.25));
    g.addColorStop(1, U.rgba(rgb, 0));
    x.fillStyle = g; x.fillRect(0, 0, size, size);
    spriteCache.set(key, c);
    return c;
  };

  /** Rounded rect path helper. */
  U.roundRect = (ctx, x, y, w, h, r) => {
    r = Math.min(r, w / 2, h / 2);
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.arcTo(x + w, y, x + w, y + h, r);
    ctx.arcTo(x + w, y + h, x, y + h, r);
    ctx.arcTo(x, y + h, x, y, r);
    ctx.arcTo(x, y, x + w, y, r);
    ctx.closePath();
  };

  U.fonts = {
    display: '"Space Grotesk", ui-sans-serif, system-ui, sans-serif',
    body: '"Inter", ui-sans-serif, system-ui, sans-serif',
    mono: '"JetBrains Mono", ui-monospace, SFMono-Regular, Menlo, monospace',
  };

  window.SceneUtils = U;
  window.SceneRegistry = window.SceneRegistry || {};
})();
