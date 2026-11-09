/* ==========================================================================
   FramePlayer — scroll- and hover-scrubbed image sequences drawn to <canvas>.
   Powers every .frame-player element:

     data-frames  base path            "assets/work/obsidian/"
     data-count   number of frames     48
     data-pad     file-name padding    3   → 001.webp
     data-ext     file extension       "webp"
     data-poster  poster image (shown until the first frame is drawn)
     data-step    optional; stride to the original sequence for the HUD (default 5)

   Children: <canvas>, img.frame-player__poster, [data-frame-label].
   Adds .is-ready once the first frame is on the canvas (CSS fades the poster).
   If frames fail to load, the poster simply stays.
   No dependencies. Exposes window.FramePlayer (optional). Loaded with defer.
   ========================================================================== */
(function () {
  "use strict";

  const mqReduce = window.matchMedia("(prefers-reduced-motion: reduce)");
  const mqFine = window.matchMedia("(hover: hover) and (pointer: fine)");

  const DPR_CAP = 2;
  const PRELOAD_MARGIN = "150% 0px 150% 0px"; // start loading ~1.5 viewports away
  const SCRUB_IN = 0.22;  // scroll progress where the sequence starts…
  const SCRUB_OUT = 0.78; // …and where it ends: the element's centre crossing ~90% → ~10% of the viewport
  const MAX_INFLIGHT = 4; // concurrent frame requests per player
  const TAU_SCROLL = 110; // ms smoothing time-constants (ignored under reduced motion)
  const TAU_HOVER = 60;

  const clamp = (v, a = 0, b = 1) => Math.min(b, Math.max(a, v));
  const easeInOutSine = (t) => -(Math.cos(Math.PI * t) - 1) / 2;

  /** 0 when the element's top enters at the bottom, 1 when its bottom leaves at the top. */
  function scrollProgress(el) {
    const r = el.getBoundingClientRect();
    const vh = window.innerHeight || document.documentElement.clientHeight;
    return clamp((vh - r.top) / (vh + r.height));
  }

  /** First and last frame, then coarse-to-fine, so early scrubbing already spans the range. */
  function loadOrder(n) {
    const seen = new Uint8Array(n);
    const order = [];
    const push = (i) => { if (i >= 0 && i < n && !seen[i]) { seen[i] = 1; order.push(i); } };
    push(0);
    push(n - 1);
    for (let s = 16; s >= 1; s >>= 1) for (let i = 0; i < n; i += s) push(i);
    return order;
  }

  const IDLE = 0, LOADING = 1, OK = 2, FAILED = 3;

  class Player {
    constructor(el) {
      const d = el.dataset;
      this.el = el;
      this.base = d.frames || "";
      this.count = Math.max(1, parseInt(d.count, 10) || 1);
      this.pad = parseInt(d.pad, 10) || 0;
      this.ext = (d.ext || "webp").replace(/^\./, "");
      this.step = parseInt(d.step, 10) || 5;

      this.canvas = el.querySelector("canvas");
      this.label = el.querySelector("[data-frame-label]");
      const poster = el.querySelector(".frame-player__poster");
      if (poster && !poster.getAttribute("src") && d.poster) poster.src = d.poster;

      this.ctx = null;
      try { this.ctx = this.canvas ? this.canvas.getContext("2d", { alpha: false }) : null; } catch (e) { /* keep poster */ }
      this.ok = !!(this.ctx && this.base);

      this.frames = new Array(this.count).fill(null);
      this.state = new Uint8Array(this.count);
      this.queue = [];
      this.inflight = 0;
      this.started = false;

      this.near = false;  // within the preload margin
      this.hover = false; // pointer scrub active
      this.hoverT = 0;
      this.cur = -1;      // displayed position 0..1 (smoothed)
      this.target = 0;

      this.w = 0; this.h = 0; this.dpr = 1;
      this.drawnImg = null;
      this.drawnIdx = -1;
      this.ready = false;

      if (this.ok) this.bindPointer();
    }

    url(i) {
      return this.base + String(i + 1).padStart(this.pad, "0") + "." + this.ext;
    }

    /* ---------- Loading ---------- */
    start() {
      if (this.started || !this.ok) return;
      this.started = true;
      // The first frame alone, then the rest without blocking it.
      this.load(0, (ok) => {
        if (!ok) return; // first frame failed: leave the poster in place
        this.queue = loadOrder(this.count).filter((i) => this.state[i] === IDLE);
        this.pump();
      });
    }

    pump() {
      while (this.inflight < MAX_INFLIGHT && this.queue.length) {
        const i = this.queue.shift();
        if (this.state[i] === IDLE) this.load(i, () => this.pump());
      }
    }

    load(i, done) {
      this.state[i] = LOADING;
      this.inflight++;
      const img = new Image();
      img.decoding = "async";
      let settled = false;
      const finish = (ok) => {
        if (settled) return;
        settled = true;
        this.inflight--;
        this.state[i] = ok ? OK : FAILED;
        if (ok) { this.frames[i] = img; kick(); }
        if (done) done(ok);
      };
      img.src = this.url(i);
      if (typeof img.decode === "function") {
        // decode() can reject for a loaded image (e.g. aborted decode); fall back to load state.
        img.decode().then(() => finish(true), () => finish(img.complete && img.naturalWidth > 0));
      } else {
        img.onload = () => finish(true);
        img.onerror = () => finish(false);
      }
    }

    /* ---------- Sizing ---------- */
    measure() {
      if (!this.ok) return;
      const w = this.canvas.clientWidth, h = this.canvas.clientHeight;
      const dpr = Math.min(window.devicePixelRatio || 1, DPR_CAP);
      if (!w || !h || (w === this.w && h === this.h && dpr === this.dpr)) return;
      this.w = w; this.h = h; this.dpr = dpr;
      this.canvas.width = Math.round(w * dpr);
      this.canvas.height = Math.round(h * dpr);
      this.drawnImg = null; // backing store was cleared: force a redraw
    }

    /* ---------- Pointer scrub (fine pointers only) ---------- */
    bindPointer() {
      const el = this.el;
      const fine = (e) => mqFine.matches && e.pointerType !== "touch";
      const setX = (e) => {
        const r = el.getBoundingClientRect();
        this.hoverT = clamp((e.clientX - r.left) / Math.max(1, r.width));
      };
      const onMove = (e) => {
        if (!fine(e)) return;
        this.hover = true;
        setX(e);
        kick();
      };
      el.addEventListener("pointerenter", onMove, { passive: true });
      el.addEventListener("pointermove", onMove, { passive: true });
      el.addEventListener("pointerleave", () => {
        if (!this.hover) return;
        this.hover = false; // eases back to the scroll position
        kick();
      }, { passive: true });
    }

    /* ---------- Per-frame ---------- */
    /** Returns true while still easing toward the target. */
    update(dt) {
      if (this.hover) {
        this.target = this.hoverT;
      } else {
        const p = scrollProgress(this.el);
        this.target = easeInOutSine(clamp((p - SCRUB_IN) / (SCRUB_OUT - SCRUB_IN)));
      }

      let moving = false;
      if (this.cur < 0 || mqReduce.matches) {
        this.cur = this.target;
      } else {
        const k = 1 - Math.exp(-dt / (this.hover ? TAU_HOVER : TAU_SCROLL));
        this.cur += (this.target - this.cur) * k;
        if (Math.abs(this.target - this.cur) < 0.0005) this.cur = this.target;
        else moving = true;
      }

      this.render();
      return moving;
    }

    /** Nearest decoded frame to i, searching outward; -1 if none yet. */
    nearest(i) {
      const n = this.count, s = this.state;
      for (let d = 0; d < n; d++) {
        if (i - d >= 0 && s[i - d] === OK) return i - d;
        if (i + d < n && s[i + d] === OK) return i + d;
      }
      return -1;
    }

    render() {
      if (!this.w) return;
      const idx = this.nearest(Math.round(this.cur * (this.count - 1)));
      if (idx < 0) return;
      const img = this.frames[idx];
      if (img === this.drawnImg) return; // only redraw when the frame changes

      const cw = this.canvas.width, ch = this.canvas.height;
      const iw = img.naturalWidth, ih = img.naturalHeight;
      if (!iw || !ih) return;
      const s = Math.max(cw / iw, ch / ih); // cover
      const dw = iw * s, dh = ih * s;
      try {
        this.ctx.imageSmoothingEnabled = true;
        this.ctx.imageSmoothingQuality = "high";
        this.ctx.drawImage(img, (cw - dw) / 2, (ch - dh) / 2, dw, dh);
      } catch (e) {
        return;
      }
      this.drawnImg = img;

      if (idx !== this.drawnIdx) {
        this.drawnIdx = idx;
        if (this.label) {
          // map 0..count-1 onto the original 1..(count*step) so the last frame reads "240 / 240"
          const total = this.count * this.step;
          const n = this.count > 1 ? Math.round(1 + (idx / (this.count - 1)) * (total - 1)) : 1;
          this.label.textContent = String(n).padStart(3, "0");
        }
      }
      if (!this.ready) {
        this.ready = true;
        this.el.classList.add("is-ready");
      }
    }
  }

  /* ---------- Shared loop ---------- */
  const players = [];
  const byEl = new Map();
  let raf = 0, last = 0;

  function frame(t) {
    raf = 0;
    const dt = Math.min(64, Math.max(0, t - last));
    last = t;
    let again = false;
    for (const p of players) if (p.near && p.update(dt)) again = true;
    if (again) raf = requestAnimationFrame(frame);
  }
  function kick() {
    if (!raf) { last = performance.now(); raf = requestAnimationFrame(frame); }
  }

  const io = "IntersectionObserver" in window
    ? new IntersectionObserver((entries) => {
        for (const e of entries) {
          const p = byEl.get(e.target);
          if (!p) continue;
          p.near = e.isIntersecting;
          if (p.near) { p.measure(); p.start(); }
        }
        kick();
      }, { rootMargin: PRELOAD_MARGIN })
    : null;

  const ro = "ResizeObserver" in window
    ? new ResizeObserver((entries) => {
        for (const e of entries) { const p = byEl.get(e.target); if (p) p.measure(); }
        kick();
      })
    : null;

  function init(root) {
    (root || document).querySelectorAll(".frame-player").forEach((el) => {
      if (byEl.has(el)) return;
      const p = new Player(el);
      if (!p.ok) return; // no canvas/context: the poster stays
      byEl.set(el, p);
      players.push(p);
      if (ro) ro.observe(el);
      if (io) io.observe(el);
      else { p.near = true; p.measure(); p.start(); }
    });
    kick();
  }

  window.addEventListener("scroll", kick, { passive: true });
  window.addEventListener("resize", () => { players.forEach((p) => p.measure()); kick(); }, { passive: true });
  document.addEventListener("visibilitychange", () => { if (!document.hidden) kick(); });

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", () => init());
  else init();

  window.FramePlayer = { init, players };
})();
