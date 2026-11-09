/* ==========================================================================
   Engine: scroll progress, scene lifecycle, reveals, counters, nav.
   No dependencies. Loaded last (defer).
   ========================================================================== */
(function () {
  "use strict";

  const U = window.SceneUtils;
  const Registry = window.SceneRegistry || {};
  const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  window.__engineReady = true; // cancels the no-JS failsafe in <head>

  /* ---------------- Nav state ---------------- */
  const nav = document.querySelector("[data-nav]");
  const onScrollNav = () => nav && nav.classList.toggle("is-scrolled", window.scrollY > 24);
  window.addEventListener("scroll", onScrollNav, { passive: true });
  onScrollNav();

  /* ---------------- Footer year ---------------- */
  document.querySelectorAll("[data-year]").forEach((n) => (n.textContent = new Date().getFullYear()));

  /* ---------------- Reveal on scroll ---------------- */
  const revealIO = new IntersectionObserver(
    (entries) => {
      entries.forEach((e) => {
        if (!e.isIntersecting) return;
        const el = e.target;
        // Stagger siblings that reveal together
        const sibs = el.parentElement ? [...el.parentElement.children].filter((c) => c.classList.contains("reveal")) : [el];
        const i = Math.max(0, sibs.indexOf(el));
        el.style.transitionDelay = reduced ? "0ms" : `${Math.min(i, 6) * 70}ms`;
        el.classList.add("is-in");
        revealIO.unobserve(el);
      });
    },
    { rootMargin: "0px 0px -8% 0px", threshold: 0.08 }
  );
  document.querySelectorAll(".reveal").forEach((el) => revealIO.observe(el));

  /* ---------------- Counters ---------------- */
  const fmt = (n) => Math.round(n).toLocaleString("en-US");
  const counterIO = new IntersectionObserver(
    (entries) => {
      entries.forEach((e) => {
        if (!e.isIntersecting) return;
        const el = e.target;
        counterIO.unobserve(el);
        const to = parseFloat(el.dataset.countTo);
        const suffix = el.dataset.suffix || "";
        if (reduced) { el.textContent = fmt(to) + suffix; return; }
        const dur = to > 1000 ? 2000 : 900;
        const t0 = performance.now();
        const step = (t) => {
          const k = U.clamp((t - t0) / dur);
          el.textContent = fmt(to * U.ease.outExpo(k)) + (k === 1 ? suffix : "");
          if (k < 1) requestAnimationFrame(step);
        };
        el.textContent = "0";
        requestAnimationFrame(step);
      });
    },
    { threshold: 0.6 }
  );
  document.querySelectorAll("[data-count-to]").forEach((el) => counterIO.observe(el));

  /* ---------------- Progress items & scenes ---------------- */
  const items = [...document.querySelectorAll("[data-progress]")].map((el) => ({
    el,
    p: -1,
    target: 0,
    visible: false,
    kind: el.classList.contains("scene") ? "pinned" : el.classList.contains("hero") ? "hero" : "flow",
    caps: [...el.querySelectorAll("[data-from]")].map((c) => ({
      el: c,
      from: parseFloat(c.dataset.from),
      to: parseFloat(c.dataset.to),
      on: false,
    })),
    sceneName: el.dataset.scene || null,
    scene: null,
    canvas: el.querySelector(":scope > canvas.scene__canvas, :scope > .scene__sticky > canvas.scene__canvas"),
    size: { w: 0, h: 0, dpr: 1 },
  }));
  const byEl = new Map(items.map((it) => [it.el, it]));

  function computeTarget(it) {
    const r = it.el.getBoundingClientRect();
    const vh = window.innerHeight;
    if (it.kind === "pinned") return U.clamp(-r.top / Math.max(1, r.height - vh));
    if (it.kind === "hero") return U.clamp(-r.top / Math.max(1, r.height));
    return U.clamp((vh - r.top) / (vh + r.height)); // 0: entering bottom → 1: leaving top
  }

  function sizeCanvas(it) {
    if (!it.canvas || !it.scene) return;
    const w = it.canvas.clientWidth, h = it.canvas.clientHeight;
    if (!w || !h) return;
    const dpr = Math.min(window.devicePixelRatio || 1, U.isMobile(w) ? 1.75 : 2);
    if (w === it.size.w && h === it.size.h && dpr === it.size.dpr) return;
    it.size = { w, h, dpr };
    it.canvas.width = Math.round(w * dpr);
    it.canvas.height = Math.round(h * dpr);
    try { it.scene.resize(w, h, dpr); } catch (e) { console.error(`[scene:${it.sceneName}] resize`, e); }
    it.p = -1; // force a redraw
  }

  function ensureScene(it) {
    if (it.scene || !it.sceneName || !it.canvas) return;
    const factory = Registry[it.sceneName];
    if (typeof factory !== "function") return;
    const accent = it.el.dataset.accent || "#ffffff";
    const face = it.el.dataset.face ? it.el.dataset.face.split(",").map(Number) : null;
    try {
      it.scene = factory(it.canvas, {
        el: it.el,
        accent,
        accentRgb: U.hexToRgb(accent),
        portraitSrc: it.el.dataset.portrait || null,
        face,
        reducedMotion: reduced,
        utils: U,
        // Scenes call this after async work (image decode, idle precompute,
        // font load) so a still frame gets redrawn even when nothing scrolls.
        invalidate: () => { it.p = -1; kick(); },
      });
    } catch (e) {
      console.error(`[scene:${it.sceneName}] init`, e);
      it.scene = null;
      return;
    }
    sizeCanvas(it);
  }

  const visIO = new IntersectionObserver(
    (entries) => {
      entries.forEach((e) => {
        const it = byEl.get(e.target);
        if (!it) return;
        it.visible = e.isIntersecting;
        if (it.visible) ensureScene(it);
      });
      kick();
    },
    { rootMargin: "50% 0px 50% 0px" }
  );
  items.forEach((it) => visIO.observe(it.el));

  if ("ResizeObserver" in window) {
    const ro = new ResizeObserver((entries) => {
      entries.forEach((e) => {
        const it = items.find((i) => i.canvas === e.target);
        if (it) sizeCanvas(it);
      });
      kick();
    });
    items.forEach((it) => it.canvas && ro.observe(it.canvas));
  } else {
    window.addEventListener("resize", () => { items.forEach(sizeCanvas); kick(); });
  }

  /* ---------------- Main loop ----------------
     Runs while something is visible and either animating ambiently (scenes)
     or still easing toward its scroll target. */
  let raf = 0, last = performance.now();
  function frame(t) {
    raf = 0;
    const dt = Math.min(64, t - last);
    last = t;
    let keepGoing = false;

    for (const it of items) {
      if (!it.visible) continue;
      it.target = computeTarget(it);

      // Frame-rate independent smoothing for pinned scenes; direct for the rest
      if (it.p < 0 || reduced || it.kind !== "pinned") it.p = it.target;
      else {
        const k = 1 - Math.exp(-dt / 90);
        it.p += (it.target - it.p) * k;
        if (Math.abs(it.target - it.p) < 0.0004) it.p = it.target;
        else keepGoing = true;
      }

      it.el.style.setProperty("--p", it.p.toFixed(4));

      for (const c of it.caps) {
        const on = it.p >= c.from && it.p < c.to;
        if (on !== c.on) { c.on = on; c.el.classList.toggle("is-on", on); }
      }

      if (it.scene) {
        try { it.scene.render(it.p, reduced ? 0 : t, dt); }
        catch (e) { console.error(`[scene:${it.sceneName}] render`, e); it.scene = null; }
        if (!reduced) keepGoing = true; // ambient motion
      }
    }

    if (keepGoing) raf = requestAnimationFrame(frame);
  }
  function kick() { if (!raf) { last = performance.now(); raf = requestAnimationFrame(frame); } }
  window.addEventListener("scroll", kick, { passive: true });
  window.addEventListener("resize", kick, { passive: true });
  document.addEventListener("visibilitychange", () => { if (!document.hidden) kick(); });

  // Warm all portraits once the page is idle, so a reader who jumps straight
  // to a team scene never sees the placeholder (U.loadImage is cached/shared).
  window.addEventListener("load", () => {
    const warm = () => document.querySelectorAll("[data-portrait]").forEach((el) => U.loadImage(el.dataset.portrait).catch(() => {}));
    (window.requestIdleCallback || ((fn) => setTimeout(fn, 300)))(warm, { timeout: 2500 });
  }, { once: true });

  // Redraw scenes once web fonts are ready (canvas text uses them)
  if (document.fonts && document.fonts.ready) document.fonts.ready.then(() => { items.forEach((it) => (it.p = -1)); kick(); });

  kick();
})();
