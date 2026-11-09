/* ==========================================================================
   Hero scene — three signals (infra · AI · CS) weave in from the left and
   converge into one white core, then continue as a single beam.
   The visual thesis of the page: three disciplines, one output.
   ========================================================================== */
(function () {
  "use strict";
  const U = window.SceneUtils;

  window.SceneRegistry.hero = function createHero(canvas, opts) {
    const ctx = canvas.getContext("2d");
    const reduced = opts.reducedMotion;
    let W = 0, H = 0, DPR = 1;

    const STREAMS = [
      { rgb: [56, 217, 245], y: 0.26, amp: 0.15, freq: 1.35, speed: 0.42, phase: 0.0 },
      { rgb: [124, 242, 154], y: 0.52, amp: 0.12, freq: 1.9, speed: 0.58, phase: 2.2 },
      { rgb: [255, 181, 71], y: 0.80, amp: 0.16, freq: 1.15, speed: 0.36, phase: 4.1 },
    ];
    const HARMONICS = 6;
    const SEG = 140;

    // Particles riding the streams: s in [0, 1.35] (past 1 = merged beam)
    const rnd = U.rng(42);
    const particles = [];
    STREAMS.forEach((st, si) => {
      for (let i = 0; i < 34; i++) {
        particles.push({ si, s: rnd() * 1.35, v: 0.045 + rnd() * 0.07, h: Math.floor(rnd() * 3), size: 0.6 + rnd() * 1.1 });
      }
    });
    const sprites = STREAMS.map((st) => U.glowSprite(st.rgb, 48, 0.12));
    const whiteSprite = U.glowSprite([255, 255, 255], 256, 0.06);

    // Pointer parallax (desktop only)
    let mx = 0.5, my = 0.5, tmx = 0.5, tmy = 0.5;
    const onMove = (e) => { tmx = e.clientX / window.innerWidth; tmy = e.clientY / window.innerHeight; };
    if (!reduced) window.addEventListener("pointermove", onMove, { passive: true });

    // Static dot field, pre-rendered on resize
    let field = null;
    function buildField() {
      field = document.createElement("canvas");
      field.width = Math.round(W * DPR); field.height = Math.round(H * DPR);
      const f = field.getContext("2d");
      f.scale(DPR, DPR);
      const gap = U.isMobile(W) ? 22 : 28;
      const r2 = U.rng(9);
      for (let y = gap / 2; y < H; y += gap) {
        for (let x = gap / 2; x < W; x += gap) {
          const a = 0.035 + r2() * 0.05;
          f.fillStyle = `rgba(255,255,255,${a})`;
          f.fillRect(x, y, 1, 1);
        }
      }
    }

    function focal() {
      // Anchor to the visible viewport, not the (possibly taller) hero box
      const vh = Math.min(H, window.innerHeight || H);
      if (U.isMobile(W)) return { x: W * 0.84, y: Math.min(H * 0.94, vh * 0.9) };
      return { x: W * (0.76 + (mx - 0.5) * 0.04), y: vh * (0.5 + (my - 0.5) * 0.06) };
    }

    // y position of stream si, harmonic h, at normalized x t∈[0,1] (0 = left edge, 1 = focal)
    function streamY(si, h, t, time, F) {
      const st = STREAMS[si];
      const conv = 1 - U.ease.inCubic(t);                 // amplitude collapses at the focal point
      const base = U.lerp(st.y * H, F.y, U.ease.inOutSine(t));
      const hp = h * 0.32;
      const wave = Math.sin(st.freq * t * Math.PI * 2 + time * st.speed + st.phase + hp)
                 + 0.35 * Math.sin(st.freq * 2.7 * t * Math.PI * 2 - time * st.speed * 1.3 + hp * 2);
      return base + wave * st.amp * H * conv * (1 - h * 0.07);
    }

    return {
      resize(w, h, dpr) { W = w; H = h; DPR = dpr; buildField(); },

      render(p, timeMs, dtMs) {
        const time = reduced ? 3.2 : timeMs / 1000;
        const dt = Math.min(0.05, (dtMs || 16) / 1000);
        mx += (tmx - mx) * 0.04; my += (tmy - my) * 0.04;
        const F = focal();
        // dim as the hero scrolls away; quieter on narrow screens where copy spans the width
        const fade = (1 - U.ease.inCubic(U.clamp(p * 1.4))) * (W < 1100 ? 0.6 : 1);
        F.y -= p * H * 0.12;

        ctx.setTransform(1, 0, 0, 1, 0, 0);
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.setTransform(DPR, 0, 0, DPR, 0, 0);
        if (field) { ctx.globalAlpha = 0.9 * fade + 0.1; ctx.drawImage(field, 0, 0, W, H); ctx.globalAlpha = 1; }

        ctx.globalCompositeOperation = "lighter";
        ctx.lineCap = "round";

        // Streams with harmonics (echo lines)
        for (let si = 0; si < STREAMS.length; si++) {
          const rgb = STREAMS[si].rgb;
          for (let h = HARMONICS - 1; h >= 0; h--) {
            const a = (h === 0 ? 0.9 : 0.22 / h) * fade;
            if (a < 0.01) continue;
            ctx.beginPath();
            for (let k = 0; k <= SEG; k++) {
              const t = k / SEG;
              const x = t * F.x;
              const y = streamY(si, h, t, time, F);
              if (k === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
            }
            ctx.strokeStyle = U.rgba(rgb, a);
            ctx.lineWidth = h === 0 ? 1.4 : 1;
            ctx.stroke();
            if (h === 0) { // soft wide glow pass
              ctx.strokeStyle = U.rgba(rgb, 0.08 * fade);
              ctx.lineWidth = 10;
              ctx.stroke();
            }
          }
        }

        // Merged beam to the right edge
        const beamGrad = ctx.createLinearGradient(F.x, 0, W, 0);
        beamGrad.addColorStop(0, `rgba(255,255,255,${0.85 * fade})`);
        beamGrad.addColorStop(1, "rgba(255,255,255,0)");
        ctx.strokeStyle = beamGrad;
        ctx.lineWidth = 1.6;
        ctx.beginPath();
        ctx.moveTo(F.x, F.y);
        for (let k = 1; k <= 40; k++) {
          const x = F.x + (W - F.x) * (k / 40);
          ctx.lineTo(x, F.y + Math.sin(k * 0.5 - time * 2) * (k / 40) * 3);
        }
        ctx.stroke();

        // Particles
        for (const pt of particles) {
          if (!reduced) { pt.s += pt.v * dt; if (pt.s > 1.35) pt.s -= 1.35; }
          let x, y, spr;
          if (pt.s <= 1) {
            x = pt.s * F.x; y = streamY(pt.si, pt.h, pt.s, time, F); spr = sprites[pt.si];
          } else {
            const b = (pt.s - 1) / 0.35;
            x = F.x + (W - F.x) * b; y = F.y + Math.sin(b * 20 - time * 2) * b * 3; spr = whiteSprite;
          }
          const sz = (pt.s > 1 ? 10 : 14) * pt.size;
          ctx.globalAlpha = fade * (pt.s > 1 ? 0.7 : 0.95);
          ctx.drawImage(spr, x - sz / 2, y - sz / 2, sz, sz);
        }
        ctx.globalAlpha = 1;

        // Core: pulsing white star where the three meet
        const pulse = 1 + Math.sin(time * 2.2) * 0.06;
        const core = (U.isMobile(W) ? 120 : 220) * pulse;
        ctx.globalAlpha = 0.95 * fade;
        ctx.drawImage(whiteSprite, F.x - core / 2, F.y - core / 2, core, core);
        // ripples
        for (let r = 0; r < 3; r++) {
          const ph = ((time * 0.45 + r / 3) % 1);
          ctx.beginPath();
          ctx.arc(F.x, F.y, 8 + ph * (U.isMobile(W) ? 60 : 110), 0, Math.PI * 2);
          ctx.strokeStyle = `rgba(255,255,255,${(1 - ph) * 0.25 * fade})`;
          ctx.lineWidth = 1;
          ctx.stroke();
        }
        ctx.globalAlpha = 1;
        ctx.globalCompositeOperation = "source-over";
      },

      destroy() { window.removeEventListener("pointermove", onMove); },
    };
  };
})();
