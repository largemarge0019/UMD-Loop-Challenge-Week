/* autotype sim dashboard — vanilla JS, no build step, no CDN.
 *
 * GET /api/static once, then WS /ws
 * carrying State (text JSON, 20 Hz) and Frame (binary: 0x01 + JPEG, ~10 Hz).
 * Commands go back over the same socket only when StaticInfo.teleop is true.
 *
 * Hot path discipline: ws.onmessage only parses and stores; all drawing and
 * DOM work happens in one requestAnimationFrame loop and only when something
 * changed. No per-frame closures are allocated in the draw functions.
 */
'use strict';
(function () {

  // ------------------------------------------------------------------ utils

  const $ = (id) => document.getElementById(id);
  const DPR = () => (window.devicePixelRatio || 1);
  const clamp = (v, a, b) => (v < a ? a : v > b ? b : v);
  const TRAIL_S = 3.0;
  const FLASH_MS = 600;
  const REASON_CLASS = {
    ACCEPTED: 'ok',
    NO_KEY: 'warn', GLANCING: 'warn', OUT_OF_REACH: 'warn', TOO_CLOSE: 'warn', NO_INTERSECT: 'warn',
    MOVING: 'info', DEBOUNCE: 'info',
  };
  const C = {
    ok: '#3ddc84', bad: '#ff4d4d', warn: '#ffa733', info: '#4da3ff', amber: '#ffbf3f',
    cyan: '#3ee6ff', violet: '#b48cff', text: '#e8eef6', muted: '#8f9cb0', grid: '#1b2230',
  };

  function fitCanvas(canvas) {
    // Match the backing store to CSS size × DPR; returns true if it changed.
    const dpr = DPR();
    const w = Math.max(1, Math.round(canvas.clientWidth * dpr));
    const h = Math.max(1, Math.round(canvas.clientHeight * dpr));
    if (canvas.width !== w || canvas.height !== h) {
      canvas.width = w; canvas.height = h;
      return true;
    }
    return false;
  }
  function beginDraw(ctx) {
    const dpr = DPR();
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return { w: ctx.canvas.width / dpr, h: ctx.canvas.height / dpr };
  }

  // ------------------------------------------------------------------ store

  let S = null;            // StaticInfo
  let staticJSON = null;   // to detect changes on reconnect
  let state = null;        // latest State
  let dirtyState = false;
  let dirtyFrame = false;
  let needLayout = true;
  let lastStateWall = 0;   // performance.now() of last state
  let lastDpr = DPR();     // re-fit canvases when the window moves between 1x/2x displays
  let connStale = false;   // State stopped arriving on an open socket
  const STALE_MS = 1000;   // 20 Hz stream: 1 s silence = 20 missed States
  let keyIndexByName = Object.create(null);

  // stylus trail ring buffer [x, y, t]
  const TRAIL_N = 256;
  const trail = { x: new Float32Array(TRAIL_N), y: new Float32Array(TRAIL_N), t: new Float64Array(TRAIL_N), head: 0, n: 0, lastT: 0 };
  function trailPush(x, y, t) {
    trail.x[trail.head] = x; trail.y[trail.head] = y; trail.t[trail.head] = t;
    trail.head = (trail.head + 1) % TRAIL_N;
    if (trail.n < TRAIL_N) trail.n++;
    trail.lastT = t;
  }

  let flashIdx = -1, flashT = -1e9;

  // ------------------------------------------------------------------ DOM refs

  const el = {
    banner: $('banner'), conn: $('conn'), pill: $('status-pill'), hudT: $('hud-t'), hudSeed: $('hud-seed'),
    panel: $('panel'), panelWrap: $('panel-wrap'), tooltip: $('tooltip'),
    camera: $('camera'), noimg: $('noimg'), fps: $('fps'),
    armSide: $('arm-side'), armTop: $('arm-top'), joints: $('joints'),
    cmdAge: $('cmd-age'), wdLed: $('wd-led'), simT: $('sim-t'), stylusInfo: $('stylus-info'),
    typed: $('typed'), targetText: $('target-text'), log: $('log'),
    teleop: $('teleop'), jog: $('jog'), speed: $('speed'), speedVal: $('speed-val'), seedIn: $('seed-in'),
    result: $('result'),
  };

  // ------------------------------------------------------------------ network

  let ws = null, backoff = 500, reconnectTimer = null, everConnected = false, bannerTimer = null;

  function setConn(cls, text) {
    el.conn.className = 'conn ' + cls;
    el.conn.textContent = text;
  }

  async function loadStatic() {
    try {
      const r = await fetch('/api/static', { cache: 'no-store' });
      if (!r.ok) throw new Error('HTTP ' + r.status);
      const txt = await r.text();
      if (txt !== staticJSON) {
        staticJSON = txt;
        S = JSON.parse(txt);
        initStatic();
      }
      return true;
    } catch (e) {
      showBanner('cannot reach server (' + e.message + ') — retrying…', true);
      return false;
    }
  }

  function showBanner(text, soft) {
    el.banner.textContent = text;
    el.banner.className = 'banner' + (soft ? ' soft' : '');
    el.banner.hidden = false;
  }
  function hideBanner() { el.banner.hidden = true; }

  function connect() {
    if (ws) { try { ws.close(); } catch (e) { /* ignore */ } }
    const proto = location.protocol === 'https:' ? 'wss://' : 'ws://';
    setConn('connecting', everConnected ? 'reconnecting' : 'connecting');
    let sock;
    try {
      sock = new WebSocket(proto + location.host + '/ws');
    } catch (e) {
      scheduleReconnect();
      return;
    }
    ws = sock;
    sock.binaryType = 'arraybuffer';
    sock.onopen = async () => {
      if (everConnected) await loadStatic();  // server may have restarted with other settings
      everConnected = true;
      backoff = 500;
      hideBanner();
      if (bannerTimer) { clearInterval(bannerTimer); bannerTimer = null; }
      connStale = false;
      setConn('live', 'live');
    };
    sock.onmessage = onMessage;
    sock.onclose = () => { if (ws === sock) { ws = null; onDrop(); } };
    sock.onerror = () => { /* onclose follows */ };
  }

  function onDrop() {
    setConn('down', 'disconnected');
    teleopStopAll(false);
    scheduleReconnect();
  }

  function scheduleReconnect() {
    if (reconnectTimer) return;
    const wait = backoff;
    backoff = Math.min(10000, Math.round(backoff * 1.7));
    const due = performance.now() + wait;
    const tick = () => {
      const left = Math.max(0, due - performance.now());
      showBanner('DISCONNECTED — reconnecting in ' + (left / 1000).toFixed(1) + ' s', false);
    };
    tick();
    if (bannerTimer) clearInterval(bannerTimer);
    bannerTimer = setInterval(tick, 200);
    reconnectTimer = setTimeout(async () => {
      reconnectTimer = null;
      if (!S) { if (!(await loadStatic())) { scheduleReconnect(); return; } }
      connect();
    }, wait);
  }

  function onMessage(ev) {
    const d = ev.data;
    if (typeof d === 'string') {
      let st;
      try { st = JSON.parse(d); } catch (e) { return; }
      onState(st);
    } else if (d instanceof ArrayBuffer) {
      if (d.byteLength > 1 && new Uint8Array(d, 0, 1)[0] === 0x01) onFrame(d);
    }
  }

  function send(obj) {
    if (ws && ws.readyState === 1) ws.send(JSON.stringify(obj));
  }

  // ------------------------------------------------------------------ state ingest

  function onState(st) {
    state = st;
    lastStateWall = performance.now();
    if (st.stylus && st.stylus.board_xy) {
      trailPush(st.stylus.board_xy[0], st.stylus.board_xy[1], lastStateWall);
    }
    dirtyState = true;
  }

  // ------------------------------------------------------------------ camera frames

  const cam = { ctx: null, bitmap: null, pending: null, decoding: false, frames: 0, fpsT: 0, fps: 0, useBitmap: typeof createImageBitmap === 'function' };

  function onFrame(buf) {
    cam.frames++;
    const blob = new Blob([new Uint8Array(buf, 1)], { type: 'image/jpeg' });
    if (cam.decoding) { cam.pending = blob; return; }   // keep only the newest
    decodeFrame(blob);
  }

  function decodeFrame(blob) {
    cam.decoding = true;
    if (cam.useBitmap) {
      createImageBitmap(blob).then(onDecoded, onDecodeFail);
    } else {
      const url = URL.createObjectURL(blob);
      const img = new Image();
      img.onload = () => { URL.revokeObjectURL(url); onDecoded(img); };
      img.onerror = () => { URL.revokeObjectURL(url); onDecodeFail(); };
      img.src = url;
    }
  }
  function onDecoded(bmp) {
    if (cam.bitmap && cam.bitmap.close) cam.bitmap.close();
    cam.bitmap = bmp;
    dirtyFrame = true;
    cam.decoding = false;
    if (cam.pending) { const b = cam.pending; cam.pending = null; decodeFrame(b); }
  }
  function onDecodeFail() {
    cam.decoding = false;
    if (cam.pending) { const b = cam.pending; cam.pending = null; decodeFrame(b); }
  }

  function drawCamera() {
    const ctx = cam.ctx;
    const { w, h } = beginDraw(ctx);
    ctx.fillStyle = '#000';
    ctx.fillRect(0, 0, w, h);
    const bmp = cam.bitmap;
    if (!bmp) { el.noimg.hidden = false; return; }
    el.noimg.hidden = true;
    const iw = bmp.width, ih = bmp.height;
    const sc = Math.min(w / iw, h / ih);
    const dw = iw * sc, dh = ih * sc, dx = (w - dw) / 2, dy = (h - dh) / 2;
    ctx.drawImage(bmp, dx, dy, dw, dh);

    const st = state;
    if (st && st.reticle_px) {
      // The JPEG is a downscaled copy of the camera image: image px = reticle_px × (iw / camera.width)
      // (0.5 for today's 640×360 half-resolution frames; derived so a frame-size change cannot mis-aim it).
      const k = (S.camera && S.camera.width > 0) ? iw / S.camera.width : 0.5;
      const u = st.reticle_px[0] * k * sc + dx;
      const v = st.reticle_px[1] * k * sc + dy;
      const inside = u >= dx && u <= dx + dw && v >= dy && v <= dy + dh;
      const wa = st.stylus ? st.stylus.would_accept : null;
      const col = wa === 'ACCEPTED' ? C.ok : (REASON_CLASS[wa] === 'info' ? C.info : C.warn);
      ctx.save();
      ctx.beginPath(); ctx.rect(dx, dy, dw, dh); ctx.clip();
      ctx.lineWidth = 1.5;
      ctx.strokeStyle = inside ? col : C.bad;
      const gap = 6, arm = 20, r = 13;
      ctx.beginPath();
      ctx.moveTo(u - arm, v); ctx.lineTo(u - gap, v);
      ctx.moveTo(u + gap, v); ctx.lineTo(u + arm, v);
      ctx.moveTo(u, v - arm); ctx.lineTo(u, v - gap);
      ctx.moveTo(u, v + gap); ctx.lineTo(u, v + arm);
      ctx.stroke();
      ctx.beginPath(); ctx.arc(u, v, r, 0, Math.PI * 2); ctx.stroke();
      ctx.restore();
      if (!inside) {
        ctx.fillStyle = C.bad; ctx.font = '12px ui-monospace, Menlo, monospace';
        ctx.fillText('reticle off-image', dx + 8, dy + dh - 8);
      }
      if (st.stylus && st.stylus.key) {
        ctx.font = 'bold 13px ui-monospace, Menlo, monospace';
        ctx.fillStyle = col;
        ctx.fillText(st.stylus.key, clamp(u + 18, dx + 4, dx + dw - 40), clamp(v - 16, dy + 14, dy + dh - 4));
      }
    }
    // image size + fps in the corner
    ctx.fillStyle = 'rgba(0,0,0,.55)';
    ctx.fillRect(dx, dy, 140, 17);
    ctx.fillStyle = C.muted; ctx.font = '12px ui-monospace, Menlo, monospace';
    ctx.fillText(iw + '×' + ih + ' · ' + cam.fps.toFixed(1) + ' fps', dx + 4, dy + 13);
  }

  // ------------------------------------------------------------------ top-down panel

  const panel = {
    ctx: null, bg: document.createElement('canvas'), tex: null, texLoaded: false,
    scale: 1, ox: 0, oy: 0, keyPx: null, hoverIdx: -1, mouseX: 0, mouseY: 0,
  };

  function panelLayout() {
    const canvas = el.panel;
    fitCanvas(canvas);
    const dpr = DPR();
    const w = canvas.width / dpr, h = canvas.height / dpr;
    const ml = 36, mt = 18, mr = 8, mb = 8;
    const sc = Math.min((w - ml - mr) / S.panel.w, (h - mt - mb) / S.panel.h);
    panel.scale = sc;
    panel.ox = ml + ((w - ml - mr) - sc * S.panel.w) / 2;
    panel.oy = mt + ((h - mt - mb) - sc * S.panel.h) / 2;
    const n = S.keys.length;
    if (!panel.keyPx || panel.keyPx.length !== n * 4) panel.keyPx = new Float32Array(n * 4);
    for (let i = 0; i < n; i++) {
      const r = S.keys[i].rect;
      panel.keyPx[i * 4] = panel.ox + r[0] * sc;
      panel.keyPx[i * 4 + 1] = panel.oy + r[1] * sc;
      panel.keyPx[i * 4 + 2] = (r[2] - r[0]) * sc;
      panel.keyPx[i * 4 + 3] = (r[3] - r[1]) * sc;
    }
    panelRenderBg(w, h);
  }

  function panelRenderBg(w, h) {
    const bg = panel.bg;
    const dpr = DPR();
    bg.width = Math.round(w * dpr); bg.height = Math.round(h * dpr);
    const ctx = bg.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.fillStyle = '#0a0d12';
    ctx.fillRect(0, 0, w, h);
    const sc = panel.scale, ox = panel.ox, oy = panel.oy;
    const pw = S.panel.w * sc, ph = S.panel.h * sc;
    ctx.fillStyle = '#000'; ctx.fillRect(ox, oy, pw, ph);
    if (panel.texLoaded) {
      // StaticInfo.texture gives the PNG's scale (px_per_m); its metre extent is pixels / px_per_m,
      // anchored at the board origin. It need not equal panel.w × panel.h (a padded or rounded
      // texture would otherwise silently mis-register against the key rectangles), so clip to the panel.
      const tx = S.texture;
      const ppm = tx && tx.px_per_m > 0 ? tx.px_per_m : 0;
      const tpw = ppm ? (panel.tex.naturalWidth || tx.w) / ppm : S.panel.w;
      const tph = ppm ? (panel.tex.naturalHeight || tx.h) / ppm : S.panel.h;
      ctx.save();
      ctx.beginPath(); ctx.rect(ox, oy, pw, ph); ctx.clip();
      ctx.drawImage(panel.tex, ox, oy, tpw * sc, tph * sc);
      ctx.restore();
    }
    ctx.strokeStyle = '#4a5568'; ctx.lineWidth = 1;
    ctx.strokeRect(ox - 0.5, oy - 0.5, pw + 1, ph + 1);

    // metre ticks every 0.05 m along the top and left edges
    ctx.font = '12px ui-monospace, Menlo, monospace';
    ctx.fillStyle = C.muted; ctx.strokeStyle = '#4a5568';
    ctx.textAlign = 'center'; ctx.textBaseline = 'bottom';
    for (let x = 0; x <= S.panel.w + 1e-9; x += 0.05) {
      const px = ox + x * sc;
      ctx.beginPath(); ctx.moveTo(px, oy - 1); ctx.lineTo(px, oy - 5); ctx.stroke();
      ctx.fillText(x.toFixed(2), px, oy - 6);
    }
    ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
    for (let y = 0; y <= S.panel.h + 1e-9; y += 0.05) {
      const py = oy + y * sc;
      ctx.beginPath(); ctx.moveTo(ox - 1, py); ctx.lineTo(ox - 5, py); ctx.stroke();
      ctx.fillText(y.toFixed(2), ox - 7, py);
    }
    // faint minor ticks at 0.01 m
    ctx.strokeStyle = '#2c3543';
    for (let x = 0; x <= S.panel.w + 1e-9; x += 0.01) {
      const px = ox + x * sc; ctx.beginPath(); ctx.moveTo(px, oy - 1); ctx.lineTo(px, oy - 3); ctx.stroke();
    }
    for (let y = 0; y <= S.panel.h + 1e-9; y += 0.01) {
      const py = oy + y * sc; ctx.beginPath(); ctx.moveTo(ox - 1, py); ctx.lineTo(ox - 3, py); ctx.stroke();
    }

    // markers
    ctx.lineWidth = 1.5; ctx.strokeStyle = C.cyan; ctx.fillStyle = C.cyan;
    ctx.font = 'bold 12px ui-monospace, Menlo, monospace'; ctx.textAlign = 'left'; ctx.textBaseline = 'top';
    for (const m of S.markers) {
      const s = m.size * sc;
      const x = ox + m.center[0] * sc - s / 2, y = oy + m.center[1] * sc - s / 2;
      ctx.strokeRect(x, y, s, s);
      ctx.fillText('id' + m.id, x + s + 3, y);
    }

    // axis glyph at the board origin: +x right, +y DOWN
    const ax = ox, ay = oy, L = Math.min(40, 0.06 * sc);
    ctx.strokeStyle = '#ffffff'; ctx.fillStyle = '#ffffff'; ctx.lineWidth = 2;
    arrow(ctx, ax, ay, ax + L, ay);
    arrow(ctx, ax, ay, ax, ay + L);
    ctx.font = 'bold 12px ui-monospace, Menlo, monospace';
    ctx.textAlign = 'left'; ctx.textBaseline = 'top';
    ctx.fillText('+x', ax + L + 4, ay + 2);
    ctx.fillText('+y', ax + 5, ay + L + 1);
    ctx.font = '12px ui-monospace, Menlo, monospace'; ctx.fillStyle = C.muted;
    ctx.fillText('[m]', ax + L + 4, ay + 15);
  }

  function arrow(ctx, x0, y0, x1, y1) {
    const dx = x1 - x0, dy = y1 - y0, len = Math.hypot(dx, dy) || 1;
    const ux = dx / len, uy = dy / len, hs = 6;
    ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x1, y1); ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.lineTo(x1 - hs * ux + hs * 0.5 * uy, y1 - hs * uy - hs * 0.5 * ux);
    ctx.lineTo(x1 - hs * ux - hs * 0.5 * uy, y1 - hs * uy + hs * 0.5 * ux);
    ctx.closePath(); ctx.fill();
  }

  function drawPanel(now) {
    const ctx = panel.ctx;
    const { w, h } = beginDraw(ctx);
    ctx.drawImage(panel.bg, 0, 0, w, h);
    const st = state;
    const kp = panel.keyPx, n = S.keys.length;
    const reach = reachOf(st);

    // key rectangles
    ctx.lineWidth = 1;
    if (reach) {
      for (let pass = 0; pass < 2; pass++) {
        // pass 0 = unreachable (red), pass 1 = reachable (green); grouped to cut style changes
        const want = pass === 1;
        ctx.fillStyle = want ? 'rgba(61,220,132,0.35)' : 'rgba(255,77,77,0.35)';
        ctx.strokeStyle = want ? C.ok : C.bad;
        ctx.beginPath();
        for (let i = 0; i < n; i++) {
          if ((reach[i] === true) !== want) continue;
          ctx.rect(kp[i * 4] + 0.5, kp[i * 4 + 1] + 0.5, kp[i * 4 + 2] - 1, kp[i * 4 + 3] - 1);
        }
        ctx.fill(); ctx.stroke();
      }
    } else {
      // no State yet, or reachable missing / not one entry per key: neutral outline, no red/green fill
      ctx.strokeStyle = '#55607a';
      ctx.beginPath();
      for (let i = 0; i < n; i++) ctx.rect(kp[i * 4] + 0.5, kp[i * 4 + 1] + 0.5, kp[i * 4 + 2] - 1, kp[i * 4 + 3] - 1);
      ctx.stroke();
    }

    // last accepted key flash
    const fage = now - flashT;
    if (flashIdx >= 0 && fage < FLASH_MS) {
      const a = 1 - fage / FLASH_MS;
      const i = flashIdx;
      ctx.fillStyle = 'rgba(255,255,255,' + (0.75 * a).toFixed(3) + ')';
      ctx.fillRect(kp[i * 4], kp[i * 4 + 1], kp[i * 4 + 2], kp[i * 4 + 3]);
      ctx.lineWidth = 3; ctx.strokeStyle = 'rgba(61,220,132,' + a.toFixed(3) + ')';
      const grow = 6 * (1 - a);
      ctx.strokeRect(kp[i * 4] - grow, kp[i * 4 + 1] - grow, kp[i * 4 + 2] + 2 * grow, kp[i * 4 + 3] + 2 * grow);
    }

    // hover highlight
    if (panel.hoverIdx >= 0) {
      const i = panel.hoverIdx;
      ctx.lineWidth = 2; ctx.strokeStyle = C.cyan;
      ctx.strokeRect(kp[i * 4], kp[i * 4 + 1], kp[i * 4 + 2], kp[i * 4 + 3]);
    }

    // stylus trail (3 s fade) + dot
    const sc = panel.scale, ox = panel.ox, oy = panel.oy;
    if (trail.n > 0) {
      const cutoff = now - TRAIL_S * 1000;
      for (let k = 0; k < trail.n; k++) {
        const idx = (trail.head - 1 - k + TRAIL_N * 2) % TRAIL_N;
        const t = trail.t[idx];
        if (t < cutoff) break;
        const a = 1 - (now - t) / (TRAIL_S * 1000);
        ctx.fillStyle = 'rgba(62,230,255,' + (a * 0.8).toFixed(3) + ')';
        const r = 1.5 + 2 * a;
        ctx.beginPath();
        ctx.arc(ox + trail.x[idx] * sc, oy + trail.y[idx] * sc, r, 0, Math.PI * 2);
        ctx.fill();
      }
    }
    if (st && st.stylus && st.stylus.board_xy) {
      const x = ox + st.stylus.board_xy[0] * sc, y = oy + st.stylus.board_xy[1] * sc;
      const wa = st.stylus.would_accept;
      const ring = wa === 'ACCEPTED' ? C.ok : (REASON_CLASS[wa] === 'info' ? C.info : C.warn);
      ctx.beginPath(); ctx.arc(x, y, 9, 0, Math.PI * 2);
      ctx.lineWidth = 2; ctx.strokeStyle = ring; ctx.stroke();
      ctx.beginPath(); ctx.arc(x, y, 4, 0, Math.PI * 2);
      ctx.fillStyle = '#fff'; ctx.fill();

    } else if (st) {
      ctx.fillStyle = C.warn; ctx.font = 'bold 12px ui-monospace, Menlo, monospace';
      ctx.textAlign = 'right'; ctx.textBaseline = 'bottom';
      ctx.fillText('NO INTERSECT', ox + S.panel.w * sc - 6, oy + S.panel.h * sc - 6);
    }
  }

  // State.reachable only when it is one boolean per StaticInfo.keys entry; otherwise unknown (null).
  function reachOf(st) {
    return st && Array.isArray(st.reachable) && st.reachable.length === S.keys.length ? st.reachable : null;
  }

  function panelHitTest(cx, cy) {
    const kp = panel.keyPx; if (!kp) return -1;
    const n = S.keys.length;
    for (let i = 0; i < n; i++) {
      const x = kp[i * 4], y = kp[i * 4 + 1];
      if (cx >= x && cx <= x + kp[i * 4 + 2] && cy >= y && cy <= y + kp[i * 4 + 3]) return i;
    }
    return -1;
  }

  function onPanelMove(ev) {
    const rect = el.panel.getBoundingClientRect();
    const x = ev.clientX - rect.left, y = ev.clientY - rect.top;
    const idx = panelHitTest(x, y);
    if (idx !== panel.hoverIdx) {
      panel.hoverIdx = idx;
      dirtyState = dirtyState || true;
      if (idx < 0) { el.tooltip.hidden = true; }
      else {
        const k = S.keys[idx];
        const cx = (k.rect[0] + k.rect[2]) / 2, cy = (k.rect[1] + k.rect[3]) / 2;
        const reach = reachOf(state);
        el.tooltip.textContent = k.name + '  centre (' + cx.toFixed(4) + ', ' + cy.toFixed(4) + ') m  ' +
          (reach === null ? '' : reach[idx] === true ? '· reachable' : '· unreachable');
        el.tooltip.hidden = false;
      }
      needPanelRedraw = true;
    }
    if (idx >= 0) {
      const tw = el.tooltip.offsetWidth;
      el.tooltip.style.left = Math.min(x + 14, rect.width - tw - 4) + 'px';
      el.tooltip.style.top = (y + 16) + 'px';
    }
  }
  function onPanelLeave() {
    if (panel.hoverIdx !== -1) { panel.hoverIdx = -1; needPanelRedraw = true; }
    el.tooltip.hidden = true;
  }
  let needPanelRedraw = false;

  // ------------------------------------------------------------------ arm views

  const arm = { side: null, top: null };
  const V = { n: [0, 0, 0], hit: [0, 0, 0] }; // scratch

  function planeHit(head, aim, corners, out) {
    // Plane through the board corners; normal oriented toward the head.
    const c0 = corners[0], c1 = corners[1], c3 = corners[3];
    const ax = c1[0] - c0[0], ay = c1[1] - c0[1], az = c1[2] - c0[2];
    const bx = c3[0] - c0[0], by = c3[1] - c0[1], bz = c3[2] - c0[2];
    let nx = ay * bz - az * by, ny = az * bx - ax * bz, nz = ax * by - ay * bx;
    const nl = Math.hypot(nx, ny, nz) || 1; nx /= nl; ny /= nl; nz /= nl;
    const hx = head[0] - c0[0], hy = head[1] - c0[1], hz = head[2] - c0[2];
    if (hx * nx + hy * ny + hz * nz < 0) { nx = -nx; ny = -ny; nz = -nz; }
    const denom = aim[0] * nx + aim[1] * ny + aim[2] * nz;
    if (denom >= -1e-9) return -1;
    const t = -(hx * nx + hy * ny + hz * nz) / denom;
    if (t <= 0) return -1;
    out[0] = head[0] + t * aim[0]; out[1] = head[1] + t * aim[1]; out[2] = head[2] + t * aim[2];
    return t;
  }

  function drawArmSide() {
    const ctx = arm.side;
    const { w, h } = beginDraw(ctx);
    ctx.fillStyle = '#0a0d12'; ctx.fillRect(0, 0, w, h);
    // fixed world window: r in [-0.25, 1.45], z in [-0.05, 1.15]
    const R0 = -0.25, R1 = 1.45, Z0 = -0.05, Z1 = 1.15;
    const sc = Math.min((w - 10) / (R1 - R0), (h - 10) / (Z1 - Z0));
    const X = (r) => 5 + (r - R0) * sc;
    const Y = (z) => h - 5 - (z - Z0) * sc;

    // grid every 0.25 m
    ctx.strokeStyle = C.grid; ctx.lineWidth = 1;
    ctx.beginPath();
    for (let r = 0; r <= R1; r += 0.25) { ctx.moveTo(X(r), Y(Z0)); ctx.lineTo(X(r), Y(Z1)); }
    for (let z = 0; z <= Z1; z += 0.25) { ctx.moveTo(X(R0), Y(z)); ctx.lineTo(X(R1), Y(z)); }
    ctx.stroke();
    ctx.fillStyle = C.muted; ctx.font = '12px ui-monospace, Menlo, monospace';
    ctx.textAlign = 'center'; ctx.textBaseline = 'top';
    for (let r = 0; r <= R1; r += 0.5) ctx.fillText(r.toFixed(1), X(r), Y(Z0) - 13);
    ctx.textAlign = 'left'; ctx.textBaseline = 'middle';
    for (let z = 0.5; z <= Z1; z += 0.5) ctx.fillText(z.toFixed(1) + 'm', X(R0) + 2, Y(z));
    // ground
    ctx.strokeStyle = '#3a4658'; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.moveTo(X(R0), Y(0)); ctx.lineTo(X(R1), Y(0)); ctx.stroke();

    const st = state;
    if (!st || !st.arm) return;
    const yaw = (st.q_deg ? st.q_deg[0] : 0) * Math.PI / 180;
    const c0 = Math.cos(yaw), s0 = Math.sin(yaw);
    const proj = (p) => p[0] * c0 + p[1] * s0;   // signed radial coordinate in the yaw plane

    // board (four corners projected into the r–z plane)
    if (st.board && st.board.corners) {
      const cs = st.board.corners;
      ctx.fillStyle = 'rgba(180,140,255,0.18)'; ctx.strokeStyle = C.violet; ctx.lineWidth = 2;
      ctx.beginPath();
      for (let i = 0; i < 4; i++) { const p = cs[i]; if (i === 0) ctx.moveTo(X(proj(p)), Y(p[2])); else ctx.lineTo(X(proj(p)), Y(p[2])); }
      ctx.closePath(); ctx.fill(); ctx.stroke();
    }

    const sh = st.arm.shoulder, elb = st.arm.elbow, hd = st.arm.head, aim = st.arm.aim;
    // base column
    ctx.fillStyle = '#2a3342';
    ctx.fillRect(X(-0.04), Y(sh[2]), 0.08 * sc, sh[2] * sc);
    // links
    ctx.lineCap = 'round'; ctx.lineWidth = 6; ctx.strokeStyle = '#c3ccd9';
    ctx.beginPath(); ctx.moveTo(X(proj(sh)), Y(sh[2])); ctx.lineTo(X(proj(elb)), Y(elb[2])); ctx.stroke();
    ctx.strokeStyle = '#e8eef6';
    ctx.beginPath(); ctx.moveTo(X(proj(elb)), Y(elb[2])); ctx.lineTo(X(proj(hd)), Y(hd[2])); ctx.stroke();
    ctx.lineCap = 'butt';
    // camera axis (short, dashed)
    if (st.arm.cam_axis) {
      const ca = st.arm.cam_axis;
      ctx.setLineDash([3, 3]); ctx.lineWidth = 1; ctx.strokeStyle = C.muted;
      ctx.beginPath(); ctx.moveTo(X(proj(hd)), Y(hd[2]));
      ctx.lineTo(X(proj(hd) + 0.15 * (ca[0] * c0 + ca[1] * s0)), Y(hd[2] + 0.15 * ca[2])); ctx.stroke();
      ctx.setLineDash([]);
    }
    // aim ray to the panel plane (or stylus max if it misses)
    let t = st.board ? planeHit(hd, aim, st.board.corners, V.hit) : -1;
    const smin = S.stylus.min, smax = S.stylus.max;
    const hit = t > 0;
    if (!hit) t = smax;
    const ex = proj(hd) + t * (aim[0] * c0 + aim[1] * s0), ez = hd[2] + t * aim[2];
    const wa = st.stylus ? st.stylus.would_accept : null;
    ctx.lineWidth = 2;
    ctx.strokeStyle = hit ? (wa === 'ACCEPTED' ? C.ok : (REASON_CLASS[wa] === 'info' ? C.info : C.warn)) : C.bad;
    if (!hit) ctx.setLineDash([4, 4]);
    ctx.beginPath(); ctx.moveTo(X(proj(hd)), Y(hd[2])); ctx.lineTo(X(ex), Y(ez)); ctx.stroke();
    ctx.setLineDash([]);
    // stylus min / max ticks along the ray
    const ar = aim[0] * c0 + aim[1] * s0, az = aim[2];
    const pr = -az, pz = ar; // perpendicular in the plane
    ctx.strokeStyle = C.muted; ctx.lineWidth = 1;
    for (const d of [smin, smax]) {
      const rr = proj(hd) + d * ar, zz = hd[2] + d * az;
      ctx.beginPath();
      ctx.moveTo(X(rr - 0.012 * pr), Y(zz - 0.012 * pz)); ctx.lineTo(X(rr + 0.012 * pr), Y(zz + 0.012 * pz)); ctx.stroke();
    }
    // joints
    ctx.fillStyle = '#fff';
    for (const p of [sh, elb]) { ctx.beginPath(); ctx.arc(X(proj(p)), Y(p[2]), 4, 0, Math.PI * 2); ctx.fill(); }
    ctx.fillStyle = C.cyan;
    ctx.beginPath(); ctx.arc(X(proj(hd)), Y(hd[2]), 5, 0, Math.PI * 2); ctx.fill();
    if (hit) { ctx.fillStyle = '#fff'; ctx.beginPath(); ctx.arc(X(ex), Y(ez), 3, 0, Math.PI * 2); ctx.fill(); }
    // range label
    if (st.stylus) {
      ctx.fillStyle = C.text; ctx.font = '12px ui-monospace, Menlo, monospace'; ctx.textAlign = 'left'; ctx.textBaseline = 'bottom';
      ctx.fillText('r=' + st.stylus.range.toFixed(3) + '/' + smax.toFixed(2) + 'm  inc=' + st.stylus.incidence_deg.toFixed(1) + '/' + maxIncDeg() + '°', X(proj(hd)) + 8, Y(hd[2]) - 8);
    }
  }

  function drawArmTop() {
    const ctx = arm.top;
    const { w, h } = beginDraw(ctx);
    ctx.fillStyle = '#0a0d12'; ctx.fillRect(0, 0, w, h);
    // world x up the screen, world y to the LEFT (right-handed, looking down)
    const X0 = -0.3, X1 = 1.45, Y0 = -0.7, Y1 = 0.7;
    const sc = Math.min((w - 10) / (Y1 - Y0), (h - 10) / (X1 - X0));
    const cxs = w / 2, cys = h - 5;
    const SX = (y) => cxs - y * sc;
    const SY = (x) => cys - (x - X0) * sc;

    ctx.strokeStyle = C.grid; ctx.lineWidth = 1;
    ctx.beginPath();
    for (let x = 0; x <= X1; x += 0.25) { ctx.moveTo(SX(Y0), SY(x)); ctx.lineTo(SX(Y1), SY(x)); }
    for (let y = -0.5; y <= 0.5; y += 0.25) { ctx.moveTo(SX(y), SY(X0)); ctx.lineTo(SX(y), SY(X1)); }
    ctx.stroke();
    ctx.fillStyle = C.muted; ctx.font = '12px ui-monospace, Menlo, monospace';
    ctx.textAlign = 'left'; ctx.textBaseline = 'middle';
    for (let x = 0.5; x <= X1; x += 0.5) ctx.fillText(x.toFixed(1), SX(Y1) + 2, SY(x));
    ctx.textAlign = 'center'; ctx.textBaseline = 'bottom';
    ctx.fillText('+y', SX(0.55), SY(X0) - 2); ctx.fillText('−y', SX(-0.55), SY(X0) - 2);
    ctx.fillText('+x ↑', SX(0), SY(X1) + 12);

    const st = state;
    if (!st || !st.arm) return;
    if (st.board && st.board.corners) {
      const cs = st.board.corners;
      ctx.fillStyle = 'rgba(180,140,255,0.18)'; ctx.strokeStyle = C.violet; ctx.lineWidth = 2;
      ctx.beginPath();
      for (let i = 0; i < 4; i++) { const p = cs[i]; if (i === 0) ctx.moveTo(SX(p[1]), SY(p[0])); else ctx.lineTo(SX(p[1]), SY(p[0])); }
      ctx.closePath(); ctx.fill(); ctx.stroke();
      // mark the board's top-left (origin) corner
      ctx.fillStyle = C.violet; ctx.beginPath(); ctx.arc(SX(cs[0][1]), SY(cs[0][0]), 3, 0, Math.PI * 2); ctx.fill();
    }
    const yaw = (st.q_deg ? st.q_deg[0] : 0) * Math.PI / 180;
    const sh = st.arm.shoulder, elb = st.arm.elbow, hd = st.arm.head, aim = st.arm.aim;
    // base disc + yaw direction
    ctx.fillStyle = '#2a3342'; ctx.beginPath(); ctx.arc(SX(0), SY(0), 0.08 * sc, 0, Math.PI * 2); ctx.fill();
    ctx.strokeStyle = C.amber; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.moveTo(SX(0), SY(0)); ctx.lineTo(SX(0.14 * Math.sin(yaw)), SY(0.14 * Math.cos(yaw))); ctx.stroke();
    // links
    ctx.lineCap = 'round'; ctx.lineWidth = 6; ctx.strokeStyle = '#c3ccd9';
    ctx.beginPath(); ctx.moveTo(SX(sh[1]), SY(sh[0])); ctx.lineTo(SX(elb[1]), SY(elb[0])); ctx.stroke();
    ctx.strokeStyle = '#e8eef6';
    ctx.beginPath(); ctx.moveTo(SX(elb[1]), SY(elb[0])); ctx.lineTo(SX(hd[1]), SY(hd[0])); ctx.stroke();
    ctx.lineCap = 'butt';
    // aim
    let t = st.board ? planeHit(hd, aim, st.board.corners, V.hit) : -1;
    const hit = t > 0; if (!hit) t = S.stylus.max;
    const wa = st.stylus ? st.stylus.would_accept : null;
    ctx.lineWidth = 2;
    ctx.strokeStyle = hit ? (wa === 'ACCEPTED' ? C.ok : (REASON_CLASS[wa] === 'info' ? C.info : C.warn)) : C.bad;
    if (!hit) ctx.setLineDash([4, 4]);
    ctx.beginPath(); ctx.moveTo(SX(hd[1]), SY(hd[0])); ctx.lineTo(SX(hd[1] + t * aim[1]), SY(hd[0] + t * aim[0])); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = C.cyan; ctx.beginPath(); ctx.arc(SX(hd[1]), SY(hd[0]), 5, 0, Math.PI * 2); ctx.fill();
    if (hit) { ctx.fillStyle = '#fff'; ctx.beginPath(); ctx.arc(SX(V.hit[1]), SY(V.hit[0]), 3, 0, Math.PI * 2); ctx.fill(); }
    ctx.fillStyle = C.amber; ctx.font = '12px ui-monospace, Menlo, monospace'; ctx.textAlign = 'left'; ctx.textBaseline = 'top';
    ctx.fillText('yaw ' + (st.q_deg ? st.q_deg[0].toFixed(1) : '—') + '°', 6, 19);
  }

  // ------------------------------------------------------------------ joint bars

  const jointRows = [];

  function buildJoints() {
    el.joints.textContent = '';
    jointRows.length = 0;
    S.joint_names.forEach((name, i) => {
      const row = document.createElement('div'); row.className = 'jrow';
      const nm = document.createElement('span'); nm.className = 'jname'; nm.textContent = name;
      const bar = document.createElement('div'); bar.className = 'jbar';
      const lmin = document.createElement('span'); lmin.className = 'jlim min'; lmin.textContent = S.q_min_deg[i].toFixed(0) + '°';
      const lmax = document.createElement('span'); lmax.className = 'jlim max'; lmax.textContent = S.q_max_deg[i].toFixed(0) + '°';
      const qd = document.createElement('div'); qd.className = 'jqd';
      const mark = document.createElement('div'); mark.className = 'jmark';
      bar.append(lmin, lmax, qd, mark);
      const val = document.createElement('span'); val.className = 'jval'; val.textContent = '—';
      row.append(nm, bar, val);
      el.joints.appendChild(row);
      jointRows.push({ row, qd, mark, val, cls: '' });
    });
  }

  function updateJoints(st) {
    const q = st.q_deg, qd = st.qd_deg_s;
    if (!q) return;
    for (let i = 0; i < jointRows.length; i++) {
      const r = jointRows[i];
      const lo = S.q_min_deg[i], hi = S.q_max_deg[i], span = hi - lo;
      const pct = clamp((q[i] - lo) / span, 0, 1) * 100;
      r.mark.style.left = pct.toFixed(2) + '%';
      const v = qd ? qd[i] : 0;
      const vmax = S.v_max_deg_s[i] || 1;
      const len = clamp(Math.abs(v) / vmax, 0, 1) * 40; // percent of bar
      if (len < 0.5) { r.qd.style.width = '0'; }
      else if (v > 0) { r.qd.className = 'jqd'; r.qd.style.left = pct.toFixed(2) + '%'; r.qd.style.width = Math.min(len, 100 - pct).toFixed(2) + '%'; }
      else { r.qd.className = 'jqd neg'; const l = Math.max(0, pct - len); r.qd.style.left = l.toFixed(2) + '%'; r.qd.style.width = (pct - l).toFixed(2) + '%'; }
      const dmin = q[i] - lo, dmax = hi - q[i];
      const near = Math.min(dmin, dmax);
      const cls = near <= 0.05 ? 'jrow limit' : near <= 5 ? 'jrow near' : 'jrow';
      if (cls !== r.cls) { r.row.className = cls; r.cls = cls; }
      r.val.textContent = q[i].toFixed(1) + '°';
    }
  }

  // ------------------------------------------------------------------ typed strip

  let typedKey = null;
  function updateTyped(st) {
    const target = (st.episode && st.episode.launch_key) || S.launch_key || '';
    const typed = st.typed || '';
    const key = target + '\u0000' + typed;   // NUL separator: cannot occur in either string
    if (key === typedKey) return;
    typedKey = key;
    el.targetText.textContent = target;
    el.typed.textContent = '';
    const n = Math.max(target.length, typed.length);
    for (let i = 0; i < n; i++) {
      const box = document.createElement('div');
      const span = document.createElement('span');
      if (i < typed.length) {
        span.textContent = typed[i] === ' ' ? '␣' : typed[i];
        box.className = 'tbox ' + (i >= target.length ? 'extra' : typed[i] === target[i] ? 'ok' : 'bad');
      } else {
        span.textContent = target[i];
        box.className = 'tbox hollow';
      }
      box.appendChild(span);
      el.typed.appendChild(box);
    }
  }

  // ------------------------------------------------------------------ event log

  const shownSigs = [];
  const evSig = (e) => e.t + '|' + e.kind + '|' + (e.reason || e.text || '') + '|' + (e.key || '');

  function updateLog(st) {
    const evs = st.events || [];
    // Is the shown list a suffix-aligned prefix of the incoming list?
    let appendFrom = 0;
    let rebuild = false;
    if (shownSigs.length === 0) {
      appendFrom = 0;
    } else {
      const last = shownSigs[shownSigs.length - 1];
      let j = -1;
      for (let i = evs.length - 1; i >= 0; i--) { if (evSig(evs[i]) === last) { j = i; break; } }
      if (j < 0) rebuild = true;
      else {
        // verify alignment of the tail we hold
        const m = Math.min(shownSigs.length, j + 1);
        for (let k = 0; k < m; k++) {
          if (shownSigs[shownSigs.length - 1 - k] !== evSig(evs[j - k])) { rebuild = true; break; }
        }
        appendFrom = j + 1;
      }
    }
    // Only rows appended to an already-shown log are "new": the first fill (page load / reconnect)
    // and a rebuild are history and must not flash rows or light the last accepted key.
    const fresh = !rebuild && shownSigs.length > 0;
    if (rebuild) { el.log.textContent = ''; shownSigs.length = 0; appendFrom = 0; }
    if (appendFrom >= evs.length) return;
    const atBottom = el.log.scrollTop + el.log.clientHeight >= el.log.scrollHeight - 12;
    const frag = document.createDocumentFragment();
    const now = performance.now();
    for (let i = appendFrom; i < evs.length; i++) {
      const e = evs[i];
      frag.appendChild(renderEvent(e, fresh));
      shownSigs.push(evSig(e));
      if (fresh && e.kind === 'press' && e.accepted && e.key) {
        const idx = keyIndexByName[e.key];
        if (idx !== undefined) { flashIdx = idx; flashT = now; }
      }
    }
    el.log.appendChild(frag);
    while (el.log.childElementCount > 200) el.log.removeChild(el.log.firstChild);
    while (shownSigs.length > 200) shownSigs.shift();
    if (atBottom) el.log.scrollTop = el.log.scrollHeight;
  }

  function renderEvent(e, fresh) {
    const row = document.createElement('div');
    const t = document.createElement('span'); t.className = 'et'; t.textContent = (e.t == null ? '—' : Number(e.t).toFixed(2));
    row.appendChild(t);
    if (e.kind === 'press') {
      const cls = REASON_CLASS[e.reason] || 'warn';
      row.className = 'ev ' + cls + (fresh ? ' new' : '');
      const r = document.createElement('span'); r.className = 'er'; r.textContent = e.reason || '?';
      const k = document.createElement('span'); k.className = 'ek'; k.textContent = e.key ? 'key ' + e.key : (e.reason === 'NO_INTERSECT' ? '' : 'no key');
      const xy = document.createElement('span'); xy.className = 'exy';
      xy.textContent = e.board_xy ? 'xy ' + (e.board_xy[0] * 1000).toFixed(1) + ', ' + (e.board_xy[1] * 1000).toFixed(1) + ' mm' : '';
      row.append(r, k, xy);
    } else {
      row.className = 'ev grey' + (fresh ? ' new' : '');
      const txt = document.createElement('span'); txt.className = 'etext'; txt.textContent = e.text || '';
      row.appendChild(txt);
    }
    return row;
  }

  // ------------------------------------------------------------------ result overlay

  let resultSig = null;
  function updateResult(st) {
    const r = st.result;
    if (!r) { if (resultSig !== null) { resultSig = null; el.result.hidden = true; } return; }
    const sig = JSON.stringify(r);
    if (sig === resultSig) return;
    resultSig = sig;
    renderResult(r);
    el.result.hidden = false;
  }

  function renderResult(r) {
    const target = r.target || '', typed = r.typed || '';
    const v = $('result-verdict');
    if (r.exact_match) { v.textContent = 'EXACT MATCH'; v.className = 'verdict ok'; }
    else { v.textContent = 'MISMATCH — edit distance ' + r.edit_distance; v.className = 'verdict bad'; }
    const tgt = $('result-target'), typ = $('result-typed');
    tgt.textContent = ''; typ.textContent = '';
    const n = Math.max(target.length, typed.length);
    for (let i = 0; i < n; i++) {
      const a = document.createElement('div');
      a.className = 'rc ' + (i < target.length ? 'plain' : 'miss');
      a.textContent = i < target.length ? target[i] : '·';
      tgt.appendChild(a);
      const b = document.createElement('div');
      if (i < typed.length) {
        b.className = 'rc ' + (i < target.length && typed[i] === target[i] ? 'ok' : 'bad');
        b.textContent = typed[i] === ' ' ? '␣' : typed[i];
      } else { b.className = 'rc miss bad'; b.textContent = '·'; }
      typ.appendChild(b);
    }
    $('result-attempted').textContent = r.presses_attempted;
    $('result-accepted').textContent = r.presses_accepted;
    $('result-elapsed').textContent = (r.elapsed != null ? Number(r.elapsed).toFixed(1) + ' s' : '—');
    $('result-ed').textContent = r.edit_distance;
    const tb = $('result-rej'); tb.textContent = '';
    const rej = r.rejections || {};
    const names = Object.keys(rej).sort((a, b) => rej[b] - rej[a]);
    if (names.length === 0) {
      const tr = document.createElement('tr'); const td = document.createElement('td'); td.colSpan = 2; td.className = 'dim'; td.textContent = 'no rejected presses';
      tr.appendChild(td); tb.appendChild(tr);
    }
    for (const nm of names) {
      const tr = document.createElement('tr');
      const td1 = document.createElement('td'); td1.textContent = nm; td1.className = 'r-' + (REASON_CLASS[nm] || 'warn');
      const td2 = document.createElement('td'); td2.textContent = rej[nm]; td2.className = 'count';
      tr.append(td1, td2); tb.appendChild(tr);
    }
  }

  $('result-close').addEventListener('click', () => { el.result.hidden = true; });
  el.result.addEventListener('click', (ev) => { if (ev.target === el.result) el.result.hidden = true; });

  // ------------------------------------------------------------------ misc DOM per state

  let lastPillCls = '';
  function updateHud(st) {
    const t = st.t != null ? Number(st.t).toFixed(2) : '—';
    el.hudT.textContent = t;
    el.simT.textContent = t;
    if (st.episode) {
      el.hudSeed.textContent = st.episode.seed;
      const cls = 'pill ' + (st.episode.status || 'idle');
      if (cls !== lastPillCls) { el.pill.className = cls; el.pill.textContent = st.episode.status || 'idle'; lastPillCls = cls; }
    }
    const age = st.cmd_age_s;
    if (age == null) { el.cmdAge.textContent = 'never'; el.cmdAge.className = 'mono bad'; }
    else { const ms = age * 1000; el.cmdAge.textContent = ms.toFixed(0) + ' ms'; el.cmdAge.className = 'mono' + (ms > 100 ? ' bad' : ''); }
    el.wdLed.className = 'led ' + (st.watchdog_tripped ? 'trip' : (age == null ? '' : 'ok'));
    const sty = st.stylus;
    el.stylusInfo.textContent = sty
      ? 'stylus: ' + (sty.key || '—') + ' · ' + sty.would_accept +
        ' · r ' + sty.range.toFixed(3) + '/' + S.stylus.max.toFixed(2) + ' m' +
        ' · inc ' + sty.incidence_deg.toFixed(1) + '/' + maxIncDeg() + '°'
      : 'stylus: NO_INTERSECT';
  }
  function maxIncDeg() { return S.press && S.press.max_incidence_deg != null ? Number(S.press.max_incidence_deg).toFixed(0) : '?'; }

  // ------------------------------------------------------------------ teleop

  const jogs = new Map();   // joint -> velocity currently being sent
  let jogTimer = null;
  const keyBind = {
    KeyQ: ['base_yaw', +1], KeyA: ['base_yaw', -1],
    KeyW: ['shoulder_pitch', +1], KeyS: ['shoulder_pitch', -1],
    KeyE: ['elbow_pitch', +1], KeyD: ['elbow_pitch', -1],
    ArrowLeft: ['head_pan', +1], ArrowRight: ['head_pan', -1],
    ArrowUp: ['head_tilt', +1], ArrowDown: ['head_tilt', -1],
  };
  const jogButtons = new Map(); // joint+sign -> button
  // Fallback on ev.key for browsers/automation that do not populate ev.code.
  const keyBindByKey = {
    q: keyBind.KeyQ, a: keyBind.KeyA, w: keyBind.KeyW, s: keyBind.KeyS, e: keyBind.KeyE, d: keyBind.KeyD,
    arrowleft: keyBind.ArrowLeft, arrowright: keyBind.ArrowRight, arrowup: keyBind.ArrowUp, arrowdown: keyBind.ArrowDown,
  };
  function bindFor(ev) {
    return keyBind[ev.code] || (ev.key ? keyBindByKey[ev.key.toLowerCase()] : undefined);
  }
  function actionFor(ev) {
    const k = ev.key ? ev.key.toLowerCase() : '';
    if (ev.code === 'Space' || k === ' ' || k === 'spacebar') return 'press';
    if (ev.code === 'Enter' || k === 'enter') return 'done';
    if (ev.code === 'KeyR' || k === 'r') return 'reset';
    return null;
  }

  function speed() { return Number(el.speed.value) || 10; }

  function jogTick() {
    for (const [joint, v] of jogs) send({ type: 'jog', joint, velocity_deg_s: v });
  }
  function jogStart(joint, sign) {
    if (!S || !S.teleop) return;
    const v = sign * speed();
    jogs.set(joint, v);
    send({ type: 'jog', joint, velocity_deg_s: v });
    if (!jogTimer) jogTimer = setInterval(jogTick, 100);
    const b = jogButtons.get(joint + (sign > 0 ? '+' : '-')); if (b) b.classList.add('held');
  }
  function jogStop(joint) {
    if (!jogs.has(joint)) return;
    jogs.delete(joint);
    send({ type: 'jog', joint, velocity_deg_s: 0 });
    if (jogs.size === 0 && jogTimer) { clearInterval(jogTimer); jogTimer = null; }
    const bp = jogButtons.get(joint + '+'), bm = jogButtons.get(joint + '-');
    if (bp) bp.classList.remove('held'); if (bm) bm.classList.remove('held');
  }
  function teleopStopAll(sendStop) {
    for (const joint of Array.from(jogs.keys())) jogStop(joint);
    if (sendStop) send({ type: 'stop' });
  }

  let teleopBuilt = false;
  const teleopOn = () => !!(S && S.teleop);   // Commands go out only while StaticInfo.teleop is true

  function buildTeleop() {
    document.body.classList.add('has-teleop');
    el.teleop.hidden = false;
    el.jog.textContent = '';
    jogButtons.clear();
    for (const joint of S.joint_names) {
      const item = document.createElement('div'); item.className = 'jog-item';
      const lbl = document.createElement('span'); lbl.className = 'jl'; lbl.textContent = joint;
      const minus = document.createElement('button'); minus.className = 'btn jogb'; minus.textContent = '−';
      const plus = document.createElement('button'); plus.className = 'btn jogb'; plus.textContent = '+';
      bindHold(minus, joint, -1); bindHold(plus, joint, +1);
      jogButtons.set(joint + '-', minus); jogButtons.set(joint + '+', plus);
      item.append(lbl, minus, plus);
      el.jog.appendChild(item);
    }
    // The static buttons and key bindings are bound once per page. buildTeleop can run again
    // after a reconnect (teleop false → true), and re-binding would send every command twice.
    if (teleopBuilt) return;
    teleopBuilt = true;
    el.speed.addEventListener('input', () => {
      el.speedVal.textContent = el.speed.value;
      // live-update velocities being held
      for (const [joint, v] of jogs) jogs.set(joint, Math.sign(v) * speed());
    });
    $('btn-stop').addEventListener('click', () => { if (teleopOn()) teleopStopAll(true); });
    $('btn-press').addEventListener('click', () => { if (teleopOn()) send({ type: 'press' }); });
    $('btn-done').addEventListener('click', () => { if (teleopOn()) send({ type: 'done' }); });
    $('btn-reset').addEventListener('click', sendReset);
    window.addEventListener('keydown', onKeyDown);
    window.addEventListener('keyup', onKeyUp);
    window.addEventListener('blur', () => teleopStopAll(false));
  }

  function hideTeleop() {
    teleopStopAll(false);
    el.teleop.hidden = true;
    document.body.classList.remove('has-teleop');
  }

  function sendReset() {
    if (!teleopOn()) return;
    const raw = el.seedIn.value.trim();
    if (raw === '') send({ type: 'reset' });
    else send({ type: 'reset', seed: Number(raw) });
  }

  function bindHold(btn, joint, sign) {
    btn.addEventListener('pointerdown', (ev) => { ev.preventDefault(); btn.setPointerCapture && btn.setPointerCapture(ev.pointerId); jogStart(joint, sign); });
    const stop = () => jogStop(joint);
    btn.addEventListener('pointerup', stop);
    btn.addEventListener('pointerleave', stop);
    btn.addEventListener('pointercancel', stop);
    btn.addEventListener('contextmenu', (ev) => ev.preventDefault());
  }

  function isTyping(t) {
    if (!t) return false;
    const tag = t.tagName;
    return tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT' || t.isContentEditable;
  }

  function onKeyDown(ev) {
    if (ev.code === 'Escape' || ev.key === 'Escape') { el.result.hidden = true; return; }
    if (isTyping(ev.target) || ev.metaKey || ev.ctrlKey || ev.altKey) return;
    if (!teleopOn()) return;   // bindings stay registered, but never send once teleop is off
    const b = bindFor(ev);
    if (b) { ev.preventDefault(); if (!ev.repeat) jogStart(b[0], b[1]); return; }
    if (ev.repeat) return;
    const act = actionFor(ev);
    if (act === 'press') { ev.preventDefault(); send({ type: 'press' }); }
    else if (act === 'done') { ev.preventDefault(); send({ type: 'done' }); }
    else if (act === 'reset') { ev.preventDefault(); sendReset(); }
  }
  function onKeyUp(ev) {
    const b = bindFor(ev);
    if (b) jogStop(b[0]);
  }
  // Esc dismisses the result even without teleop
  window.addEventListener('keydown', (ev) => { if (ev.code === 'Escape') el.result.hidden = true; });

  // ------------------------------------------------------------------ init

  function initStatic() {
    keyIndexByName = Object.create(null);
    S.keys.forEach((k, i) => { keyIndexByName[k.name] = i; });
    el.hudSeed.textContent = S.seed;
    el.targetText.textContent = S.launch_key || '—';
    typedKey = null;
    buildJoints();
    if (S.teleop) buildTeleop(); else hideTeleop();
    // texture
    const img = new Image();
    panel.texLoaded = false;
    img.onload = () => { panel.tex = img; panel.texLoaded = true; needLayout = true; };
    img.onerror = () => { panel.texLoaded = false; needLayout = true; };
    img.src = '/static/texture.png?ts=' + Date.now();
    needLayout = true;
  }

  function setupCanvases() {
    panel.ctx = el.panel.getContext('2d');
    cam.ctx = el.camera.getContext('2d');
    arm.side = el.armSide.getContext('2d');
    arm.top = el.armTop.getContext('2d');
    el.panel.addEventListener('mousemove', onPanelMove);
    el.panel.addEventListener('mouseleave', onPanelLeave);
    if (typeof ResizeObserver === 'function') {
      const ro = new ResizeObserver(() => { needLayout = true; });
      ro.observe(el.panelWrap); ro.observe($('camera-wrap'));
      ro.observe(el.armSide.parentElement); ro.observe(el.armTop.parentElement);
    } else {
      window.addEventListener('resize', () => { needLayout = true; });
    }
  }

  function layoutAll() {
    needLayout = false;
    if (S) panelLayout();
    fitCanvas(el.camera);
    fitCanvas(el.armSide);
    fitCanvas(el.armTop);
  }

  function frame(now) {
    requestAnimationFrame(frame);
    // fps counter for received frames (once per second)
    if (now - cam.fpsT >= 1000) {
      const dt = (now - cam.fpsT) / 1000;
      cam.fps = cam.fpsT ? cam.frames / dt : 0;
      cam.frames = 0; cam.fpsT = now;
      el.fps.textContent = cam.fps.toFixed(1) + ' fps';
    }
    if (!S) return;
    const dpr = DPR();
    if (dpr !== lastDpr) { lastDpr = dpr; needLayout = true; }   // moved between 1x and 2x displays
    if (needLayout) {
      layoutAll();
      dirtyState = true; dirtyFrame = true;
    }
    // State stalled on an open socket: say so instead of freezing the HUD under a 'live' pill.
    if (ws && ws.readyState === 1 && state) {
      const stale = now - lastStateWall > STALE_MS;
      if (stale !== connStale) { connStale = stale; setConn(stale ? 'stale' : 'live', stale ? 'stale' : 'live'); }
      if (stale) { const s = ((now - lastStateWall) / 1000).toFixed(1) + ' s'; if (el.conn.textContent !== 'stale ' + s) el.conn.textContent = 'stale ' + s; }
    }
    if (dirtyState && state) {
      dirtyState = false; dirtyFrame = false; needPanelRedraw = false;
      updateHud(state);
      updateJoints(state);
      updateTyped(state);
      updateLog(state);
      updateResult(state);
      drawPanel(now);
      drawArmSide();
      drawArmTop();
      drawCamera();
      return;
    }
    if (dirtyFrame) { dirtyFrame = false; drawCamera(); }
    // keep trail fade / flash / hover alive when no state is arriving
    const animating = (flashIdx >= 0 && now - flashT < FLASH_MS) ||
                      (trail.n > 0 && now - trail.lastT < TRAIL_S * 1000 + 100 && now - lastStateWall > 80);
    if (needPanelRedraw || animating) { needPanelRedraw = false; drawPanel(now); }
    if (!state && dirtyState) { dirtyState = false; drawPanel(now); drawArmSide(); drawArmTop(); drawCamera(); }
  }

  async function main() {
    setupCanvases();
    requestAnimationFrame(frame);
    setConn('connecting', 'connecting');
    if (await loadStatic()) { hideBanner(); connect(); }
    else scheduleReconnect();
  }

  main();
})();
