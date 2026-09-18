// Replays recorded evaluation calls: one audio file per call, transcript lines revealed on their timestamps.
// Every number, line and source stamp comes from data/calls.json, which site/build.py and site/annotate.py write.

(function () {
  const $ = (id) => document.getElementById(id);
  const reduce = window.matchMedia("(prefers-reduced-motion: reduce)");
  const audio = new Audio();
  audio.preload = "auto";
  const bargeAudio = new Audio();
  bargeAudio.preload = "none";

  let data = null;
  let current = null;
  let rows = [];
  let nowIndex = -1;
  // A seek before the audio has loaded is ignored by the browser, so the page keeps its own position too.
  let held = null;
  let raf = 0;

  const fmt = (s) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`;
  const ARABIC = /[؀-ۿ]/;
  const LATIN = /[A-Za-z]/;
  const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

  function el(tag, cls, text) {
    const n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text !== undefined && text !== null) n.textContent = text;
    return n;
  }

  // ---- bidi ----------------------------------------------------------------

  // Code-switched lines hold both scripts. Each run is isolated in its own <bdi> so punctuation and digits
  // stay with the run they belong to, and the paragraph takes its direction from its first strong letter.
  const RUN = /[؀-ۿ][؀-ۿً-ْ\d\s،؛؟.,:!-]*/g;

  function scriptText(node, text) {
    const hasAr = ARABIC.test(text);
    const hasLat = LATIN.test(text);
    if (hasAr && !hasLat) {
      node.lang = "ar";
      node.dir = "rtl";
      node.textContent = text;
      return;
    }
    node.dir = hasAr ? "auto" : "ltr";
    if (!hasAr) {
      node.textContent = text;
      return;
    }
    let last = 0;
    for (const m of text.matchAll(RUN)) {
      const run = m[0].replace(/\s+$/, "");
      if (m.index > last) node.append(text.slice(last, m.index));
      const b = el("bdi", "ar", run);
      b.lang = "ar";
      b.dir = "rtl";
      node.append(b);
      last = m.index + run.length;
    }
    if (last < text.length) node.append(text.slice(last));
  }

  // ---- transcript ------------------------------------------------------------

  function groupFacts(facts) {
    const groups = [];
    const key = (f) => `${f.call}|${f.ref || ""}|${f.via || ""}|${f.said_at ?? ""}`;
    for (const f of facts) {
      let g = groups.find((x) => x.key === key(f));
      if (!g) {
        g = { key: key(f), f, items: [] };
        groups.push(g);
      }
      f.group = groups.indexOf(g);
      g.items.push(f);
    }
    return groups;
  }

  function spokenWithFacts(node, turn) {
    const text = turn.text;
    const arabic = ARABIC.test(text) && !LATIN.test(text);
    if (!turn.facts || !turn.facts.length) {
      scriptText(node, text || "(nothing heard)");
      return;
    }
    // Agent replies are one script each, so fact spans never straddle a script boundary.
    node.dir = arabic ? "rtl" : "ltr";
    if (arabic) node.lang = "ar";
    let last = 0;
    for (const f of turn.facts) {
      const [a, b] = f.at;
      if (a > last) node.append(text.slice(last, a));
      const s = el("span", `fact${f.call === "caller" ? " from-caller" : ""}`, text.slice(a, b));
      s.dataset.group = f.group;
      node.append(s);
      last = b;
    }
    if (last < text.length) node.append(text.slice(last));
  }

  function sourceRow(g, i, turn) {
    const f = g.f;
    const li = el("li", `src${f.call === "caller" ? " caller" : ""}`);
    li.style.setProperty("--i", i);
    li.dataset.group = i;
    const code = el("code", null, f.call);
    code.translate = false;
    li.appendChild(code);
    let ref = f.ref || "";
    if (f.call === "caller") ref = f.said_at != null ? `said at ${fmt(f.said_at)}` : "said by the caller";
    else if (f.seq != null) ref += `, audit #${f.seq}`;
    if (f.via) ref += `, ${f.via}`;
    li.appendChild(el("span", "ref", ref));
    const vals = el("span", "vals");
    const seen = new Set();
    g.items.forEach((x) => {
      const t = turn.text.slice(x.at[0], x.at[1]);
      if (seen.has(t)) return;
      seen.add(t);
      if (vals.childNodes.length) vals.append("  ");
      const v = el("bdi", null, t);
      if (ARABIC.test(t)) {
        v.lang = "ar";
        v.dir = "rtl";
      }
      vals.append(v);
    });
    li.appendChild(vals);
    return li;
  }

  function row(turn, index) {
    const li = el("li", `turn ${turn.who}`);
    if (turn.tag === "safe reply") li.classList.add("fault");
    const meta = el("div", "meta");
    meta.appendChild(el("span", "who", turn.who === "caller" ? "Caller, as heard" : "Parley"));
    meta.appendChild(el("span", "at", fmt(turn.t0)));
    if (turn.tag) meta.appendChild(el("span", "tag", turn.tag));
    li.appendChild(meta);

    const groups = turn.facts ? groupFacts(turn.facts) : [];
    const scripts = el("div", "scripts");
    const spoken = el("p", "spoken");
    spokenWithFacts(spoken, turn);
    scripts.appendChild(spoken);
    if (turn.latin) {
      // Only a fully Arabic line gets the side-by-side layout. A mixed line already reads left to right.
      if (!LATIN.test(turn.text)) scripts.classList.add("two");
      const tr = el("p", "translit", turn.latin);
      tr.lang = "ar-Latn";
      tr.dir = "ltr";
      scripts.appendChild(tr);
    }
    li.appendChild(scripts);

    const plain = (x) => (x || "").replace(/[\p{P}ً-ْ]/gu, "").replace(/\s+/g, " ").trim().toLowerCase();
    if (turn.said && plain(turn.said) !== plain(turn.text)) {
      const s = el("p", "script-line");
      s.appendChild(el("b", null, "Script"));
      const t = el("span");
      scriptText(t, turn.said);
      s.appendChild(t);
      li.appendChild(s);
    }
    if (groups.length) {
      const ul = el("ul", "sources");
      ul.setAttribute("aria-label", "Where each fact came from");
      groups.forEach((g, i) => ul.appendChild(sourceRow(g, i, turn)));
      li.appendChild(ul);
      if (turn.checked) {
        const n = turn.checked.ungrounded;
        li.appendChild(el("p", "checked", n === 0 ? "grounding check: every fact matched" : `grounding check: ${n} unknown`));
      }
      li.addEventListener("pointerover", (e) => hot(li, e.target.closest("[data-group]")));
      li.addEventListener("pointerleave", () => hot(li, null));
    }
    li.dataset.index = index;
    return li;
  }

  function hot(li, target) {
    const g = target ? target.dataset.group : null;
    li.querySelectorAll("[data-group]").forEach((n) => n.classList.toggle("hot", n.dataset.group === g));
  }

  // ---- booking card -----------------------------------------------------------

  function auditLine(e) {
    const d = e.detail;
    const li = el("li");
    li.appendChild(el("span", "seq", `#${e.seq}`));
    let text = "";
    let cls = "act";
    if (e.action === "tool_call") {
      const args = Object.entries(d.args).map(([k, v]) => `${k} ${v}`).join(", ");
      text = `${d.tool}(${args})`;
    } else if (e.action === "tool_failed") {
      text = `${d.tool} failed: ${d.error}${d.attempt ? `, try ${d.attempt}` : ""}`;
      cls += " bad";
    } else if (e.action === "callback_requested") {
      text = `callback queued for ${d.phone}`;
      cls += " ok";
    } else if (e.action === "confirm") {
      text = `booked ${d.booking_id}`;
      cls += " ok";
    } else if (e.action === "hold") {
      text = `held ${d.slot_id} as ${d.hold_id}`;
    }
    li.appendChild(el("span", cls, text));
    return li;
  }

  function renderCard(show) {
    const card = $("card");
    const body = $("card-body");
    const c = current;
    const h = card.querySelector("h2");
    body.replaceChildren();
    body.classList.remove("enter");
    if (!show) {
      card.dataset.state = "pending";
      h.textContent = "Booking";
      body.appendChild(el("p", "muted", "Appears when the booking API confirms it."));
      return;
    }
    if (c.booking) {
      card.dataset.state = "booked";
      h.textContent = "Booked";
      const dl = el("dl");
      for (const [k, v] of c.booking.rows) {
        dl.appendChild(el("dt", null, k));
        const dd = el("dd", null, v);
        if (ARABIC.test(v)) {
          dd.dir = "rtl";
          dd.lang = "ar";
        }
        dl.appendChild(dd);
      }
      body.appendChild(dl);
      const confirm = (c.trace || []).find((e) => e.action === "confirm");
      if (confirm) {
        const p = el("p", "src-line");
        p.append("from ");
        p.appendChild(el("code", null, "confirm_booking"));
        p.append(`, ${confirm.detail.booking_id}, audit #${confirm.seq}`);
        body.appendChild(p);
      }
    } else if (c.outage) {
      card.dataset.state = "fault";
      h.textContent = c.outage.title;
      body.appendChild(el("p", null, c.outage.lines[0]));
      const last = Math.max(...(c.trace || []).map((e) => e.turn));
      const ul = el("ol", "audit");
      ul.setAttribute("aria-label", "Audit log for the confirm step");
      (c.trace || []).filter((e) => e.turn === last).forEach((e) => ul.appendChild(auditLine(e)));
      body.appendChild(ul);
      if (c.outage.lines[1]) body.appendChild(el("p", "recovery", c.outage.lines[1]));
    } else {
      card.dataset.state = "pending";
      body.appendChild(el("p", "muted", c.no_booking || "No booking was made."));
    }
    void body.offsetWidth;
    body.classList.add("enter");
  }

  // ---- grounding rejection ----------------------------------------------------

  function renderRejected() {
    const box = $("rejected");
    const g = data.grounding;
    box.replaceChildren();
    if (!g) {
      box.appendChild(el("p", "foot-note", "No rejected reply was recorded."));
      return;
    }
    box.appendChild(el("p", null, g.intro));
    const said = el("div");
    said.appendChild(el("span", "label", "The model wanted to say"));
    const s = el("p", "model");
    scriptText(s, g.model_reply);
    said.appendChild(s);
    box.appendChild(said);

    const why = el("p", "why");
    if (g.violations && g.violations.length && g.truth) {
      const v = g.violations[0];
      why.append(`It read the ${v.kind} as ${v.value}. `);
      why.appendChild(el("code", null, g.truth.call));
      why.append(` ${g.truth.ref} returned ${g.truth.starts_at.slice(11, 16)}. Rejected.`);
    } else {
      why.append(`Flagged: ${g.flagged}. Rejected.`);
    }
    box.appendChild(why);

    const spoken = el("div");
    spoken.appendChild(el("span", "label", "Parley said the checked template instead"));
    const sp = el("p");
    scriptText(sp, g.spoken);
    spoken.appendChild(sp);
    box.appendChild(spoken);
    box.appendChild(el("p", "foot-note", g.footnote));
  }

  const rejSwitch = $("show-rejected");
  rejSwitch.addEventListener("click", () => {
    const on = rejSwitch.getAttribute("aria-checked") !== "true";
    rejSwitch.setAttribute("aria-checked", String(on));
    $("rejected").hidden = !on;
  });

  // ---- call selector thumb: a critically damped spring on transform, so a second click mid-flight re-targets ----

  const dial = document.querySelector(".dial");
  const thumb = document.querySelector(".dial-thumb");
  const spring = { x: 0, y: 0, vx: 0, vy: 0, tx: 0, ty: 0, tw: 0, h: 0, running: false, placed: false };

  function moveThumb(button) {
    spring.tx = button.offsetLeft;
    spring.tw = button.offsetWidth;
    spring.ty = button.offsetTop;
    spring.h = button.offsetHeight;
    thumb.style.opacity = "1";
    if (!spring.placed || reduce.matches) {
      Object.assign(spring, { x: spring.tx, y: spring.ty, vx: 0, vy: 0, placed: true });
      paintThumb();
      return;
    }
    if (!spring.running) {
      spring.running = true;
      let t = performance.now();
      const step = (now) => {
        const dt = Math.min(0.032, (now - t) / 1000);
        t = now;
        // Response 0.35 s, damping ratio 1: arrives without overshoot and keeps its velocity when re-targeted.
        const w0 = (2 * Math.PI) / 0.35;
        for (const [p, v, target] of [["x", "vx", "tx"], ["y", "vy", "ty"]]) {
          const a = -w0 * w0 * (spring[p] - spring[target]) - 2 * w0 * spring[v];
          spring[v] += a * dt;
          spring[p] += spring[v] * dt;
        }
        paintThumb();
        const settled = Math.abs(spring.x - spring.tx) < 0.3 && Math.abs(spring.y - spring.ty) < 0.3 &&
          Math.abs(spring.vx) + Math.abs(spring.vy) < 2;
        if (settled) {
          Object.assign(spring, { x: spring.tx, y: spring.ty, vx: 0, vy: 0, running: false });
          paintThumb();
        } else requestAnimationFrame(step);
      };
      requestAnimationFrame(step);
    }
  }

  function paintThumb() {
    thumb.style.width = `${spring.tw}px`;
    thumb.style.height = `${spring.h}px`;
    thumb.style.transform = `translate(${spring.x}px, ${spring.y}px)`;
  }

  // ---- main waveform ----------------------------------------------------------

  const wave = $("wave");
  const wctx = wave.getContext("2d");
  let colors = {};

  // Canvas sizes are measured only when the layout changes, so the draw loop never reads layout.
  const sizes = new WeakMap();
  function measure(c) {
    const r = c.getBoundingClientRect();
    sizes.set(c, { w: r.width, h: r.height });
  }
  function sizeCanvas(c) {
    if (!sizes.has(c)) measure(c);
    const r = { width: sizes.get(c).w, height: sizes.get(c).h };
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    const w = Math.max(1, Math.round(r.width * dpr));
    const h = Math.max(1, Math.round(r.height * dpr));
    if (c.width !== w || c.height !== h) {
      c.width = w;
      c.height = h;
    }
    return { w: r.width, h: r.height, dpr };
  }

  function whoAt(t) {
    for (const turn of current.turns) if (t >= turn.t0 && t < turn.t1) return turn.who;
    return null;
  }

  function drawWave(now) {
    if (!current || !current.peaks) return;
    const { w, h, dpr } = sizeCanvas(wave);
    wctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    wctx.clearRect(0, 0, w, h);
    const pitch = 3;
    const bars = Math.floor(w / pitch);
    const per = current.peaks_per_s;
    const dur = current.duration;
    const mid = h / 2;
    for (let i = 0; i < bars; i++) {
      const t0 = (i / bars) * dur;
      const t1 = ((i + 1) / bars) * dur;
      let p = 0;
      for (let k = Math.floor(t0 * per); k < Math.min(current.peaks.length, Math.ceil(t1 * per)); k++) p = Math.max(p, current.peaks[k]);
      const who = whoAt((t0 + t1) / 2);
      const bh = who ? Math.max(2, (p / 99) * (h - 6)) : 1.5;
      wctx.globalAlpha = t0 < now ? 1 : 0.3;
      wctx.fillStyle = who === "agent" ? colors.accent : who === "caller" ? colors.caller : colors.line;
      wctx.fillRect(i * pitch, mid - bh / 2, pitch - 1, bh);
    }
    wctx.globalAlpha = 1;
    if (current.card_at != null) {
      const x = (current.card_at / dur) * w;
      wctx.fillStyle = current.outage ? colors.signal : colors.accent;
      wctx.fillRect(Math.round(x) - 1, 0, 2, h);
    }
    const px = Math.min(w - 1, (now / dur) * w);
    wctx.fillStyle = colors.ink;
    wctx.fillRect(Math.round(px) - 1, 0, 2, h);
  }

  // ---- playback -------------------------------------------------------------

  const play = $("play");

  function position() {
    return held !== null ? held : audio.currentTime || 0;
  }

  function setPlayState(state) {
    play.dataset.state = state;
    play.setAttribute("aria-label", state === "playing" ? "Pause" : state === "ended" ? "Replay" : "Play");
  }

  function tick(scroll) {
    if (!current) return;
    const now = position();
    const dur = current.duration;
    drawWave(now);
    $("clock").textContent = fmt(now);
    wave.setAttribute("aria-valuenow", String(Math.round(now)));
    wave.setAttribute("aria-valuetext", `${fmt(now)} of ${fmt(dur)}`);
    let last = -1;
    current.turns.forEach((t, i) => {
      const on = now >= t.t0 - 0.05;
      rows[i].classList.toggle("on", on);
      rows[i].classList.toggle("now", on && now < t.t1);
      if (on) last = i;
    });
    $("log-empty").hidden = last >= 0;
    if (last !== nowIndex) {
      nowIndex = last;
      if (last >= 0 && (scroll || !audio.paused)) follow(rows[last]);
    }
    const done = current.card_at !== null && now >= current.card_at;
    if ($("card").dataset.done !== String(done)) {
      $("card").dataset.done = String(done);
      renderCard(done);
    }
  }

  // Scroll only the transcript box, never the page, so a reader scrolling elsewhere is not pulled back.
  function follow(node) {
    const log = $("log");
    const top = node.offsetTop;
    const bottom = top + node.offsetHeight;
    let target = null;
    if (bottom > log.scrollTop + log.clientHeight) target = Math.min(top, bottom - log.clientHeight + 8);
    else if (top < log.scrollTop) target = top;
    if (target !== null) log.scrollTo({ top: target, behavior: reduce.matches ? "auto" : "smooth" });
  }

  function loop() {
    tick();
    if (!audio.paused) raf = requestAnimationFrame(loop);
  }

  play.addEventListener("click", () => {
    if (!current) return;
    if (audio.paused) {
      if (audio.ended) audio.currentTime = 0;
      if (held !== null) {
        audio.currentTime = held;
        held = null;
      }
      setPlayState("loading");
      bargeAudio.pause();
      audio.play().catch(() => setPlayState("paused"));
    } else {
      audio.pause();
    }
  });
  audio.addEventListener("playing", () => {
    setPlayState("playing");
    cancelAnimationFrame(raf);
    raf = requestAnimationFrame(loop);
  });
  audio.addEventListener("waiting", () => setPlayState("loading"));
  audio.addEventListener("pause", () => {
    if (!audio.ended) setPlayState("paused");
    tick();
  });
  audio.addEventListener("ended", () => {
    setPlayState("ended");
    tick();
  });

  function seek(t) {
    if (!current) return;
    t = Math.max(0, Math.min(current.duration, t));
    held = t;
    audio.currentTime = t;
    if (!audio.paused) held = null;
    if (audio.ended || play.dataset.state === "ended") setPlayState("paused");
    tick(true);
  }

  // Scrubbing tracks the pointer 1:1 and keeps going when it leaves the canvas.
  let scrubbing = false;
  const timeAt = (e) => {
    const r = wave.getBoundingClientRect();
    return ((e.clientX - r.left) / r.width) * current.duration;
  };
  wave.addEventListener("pointerdown", (e) => {
    if (!current) return;
    scrubbing = true;
    wave.setPointerCapture(e.pointerId);
    seek(timeAt(e));
  });
  wave.addEventListener("pointermove", (e) => {
    if (scrubbing) seek(timeAt(e));
  });
  const stop = () => { scrubbing = false; };
  wave.addEventListener("pointerup", stop);
  wave.addEventListener("pointercancel", stop);
  wave.addEventListener("keydown", (e) => {
    if (!current) return;
    const step = { ArrowRight: 5, ArrowLeft: -5, PageUp: 15, PageDown: -15 }[e.key];
    if (step !== undefined) seek(position() + step);
    else if (e.key === "Home") seek(0);
    else if (e.key === "End") seek(current.duration);
    else if (e.key === " ") play.click();
    else return;
    e.preventDefault();
  });

  function select(id) {
    current = data.calls.find((c) => c.id === id);
    document.querySelectorAll(".dial button").forEach((b) => {
      const on = b.dataset.call === id;
      b.setAttribute("aria-pressed", String(on));
      if (on) moveThumb(b);
    });
    if (current.outage) dial.dataset.fault = "";
    else delete dial.dataset.fault;
    audio.pause();
    bargeAudio.pause();
    audio.src = current.audio;
    held = 0;
    nowIndex = -1;
    setPlayState("paused");
    play.disabled = false;
    $("note").textContent = current.note;
    $("total").textContent = fmt(current.duration);
    wave.setAttribute("aria-valuemax", String(Math.round(current.duration)));
    const key = $("event-key");
    key.hidden = current.card_at == null;
    key.textContent = current.outage ? "Booking API stops answering" : "Booking confirmed";
    key.classList.toggle("fault", !!current.outage);
    const log = $("log");
    log.replaceChildren();
    log.scrollTop = 0;
    rows = current.turns.map((t, i) => {
      const r = row(t, i);
      log.appendChild(r);
      return r;
    });
    $("card").dataset.done = "";
    renderCard(false);
    tick();
  }

  document.querySelectorAll(".dial button").forEach((b) => b.addEventListener("click", () => {
    select(b.dataset.call);
    // The chosen call goes in the URL so a link opens on the same call.
    try {
      history.replaceState(null, "", b.dataset.call === "english" ? location.pathname : `?call=${b.dataset.call}`);
    } catch (e) { /* file:// and sandboxed frames refuse, which only loses the deep link */ }
  }));

  // Frame-exact seeking for the committed screen recording, which is rendered frame by frame.
  window.parleySeek = (id, t) => {
    if (!current || current.id !== id) select(id);
    held = t;
    tick(true);
  };

  // ---- barge-in ---------------------------------------------------------------

  const bwave = $("barge-wave");
  const bctx = bwave.getContext("2d");
  const bplay = $("barge-play");
  let braf = 0;

  function drawBarge(now) {
    const b = data && data.barge_in;
    if (!b) return;
    const { w, h, dpr } = sizeCanvas(bwave);
    bctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    bctx.clearRect(0, 0, w, h);
    const span = b.duration + 1;
    const x = (t) => (t / span) * w;
    const pitch = 3;
    const bars = Math.floor(w / pitch);
    const lane = (h - 30) / 2;
    const lanes = [{ peaks: b.agent_peaks, mid: 12 + lane / 2, color: colors.accent, label: "Parley" },
                   { peaks: b.caller_peaks, mid: 22 + lane * 1.5, color: colors.caller, label: "Caller" }];
    lanes.forEach((L, li) => {
      for (let i = 0; i < bars; i++) {
        const t0 = (i / bars) * span;
        const t1 = ((i + 1) / bars) * span;
        let p = 0;
        const k1 = Math.min(L.peaks.length, Math.ceil(t1 * b.peaks_per_s));
        for (let k = Math.floor(t0 * b.peaks_per_s); k < k1; k++) p = Math.max(p, L.peaks[k]);
        if (!p) continue;
        const bh = Math.max(2, (p / 99) * (lane - 4));
        const unsaid = li === 0 && t0 >= b.stopped;
        bctx.globalAlpha = unsaid ? 0.14 : now != null && t0 > now ? 0.4 : 1;
        bctx.fillStyle = unsaid ? colors.ink3 : L.color;
        bctx.fillRect(i * pitch, L.mid - bh / 2, pitch - 1, bh);
      }
      bctx.globalAlpha = 1;
      bctx.fillStyle = colors.ink3;
      bctx.font = `500 11px ${css("--text")}`;
      bctx.fillText(L.label, 0, L.mid - lane / 2 + 2);
    });
    // Caller onset, then the cut where Parley's audio stops.
    const xo = x(b.onset);
    const xs = x(b.stopped);
    bctx.setLineDash([3, 3]);
    bctx.strokeStyle = colors.caller;
    bctx.lineWidth = 1;
    bctx.beginPath();
    bctx.moveTo(Math.round(xo) + 0.5, 8);
    bctx.lineTo(Math.round(xo) + 0.5, h - 4);
    bctx.stroke();
    bctx.setLineDash([]);
    bctx.fillStyle = colors.signal;
    bctx.fillRect(Math.round(xs), 6, 2, lane + 10);
    bctx.font = `500 11px ${css("--mono")}`;
    const label = `stopped +${b.latency.toFixed(2)} s`;
    const lw = bctx.measureText(label).width;
    bctx.fillText(label, Math.min(w - lw, xs + 6), 10);
    if (now != null) {
      bctx.fillStyle = colors.ink;
      bctx.fillRect(Math.round(x(now)) - 1, 0, 2, h);
    }
  }

  function bargeLoop() {
    drawBarge(bargeAudio.currentTime);
    if (!bargeAudio.paused) braf = requestAnimationFrame(bargeLoop);
  }

  bplay.addEventListener("click", () => {
    if (bargeAudio.paused) {
      audio.pause();
      if (bargeAudio.ended) bargeAudio.currentTime = 0;
      bplay.dataset.state = "loading";
      bargeAudio.play().catch(() => { bplay.dataset.state = "paused"; });
    } else bargeAudio.pause();
  });
  bargeAudio.addEventListener("playing", () => {
    bplay.dataset.state = "playing";
    bplay.setAttribute("aria-label", "Pause the interruption");
    cancelAnimationFrame(braf);
    braf = requestAnimationFrame(bargeLoop);
  });
  const bargeRest = () => {
    bplay.dataset.state = "paused";
    bplay.setAttribute("aria-label", "Play the interruption");
    drawBarge(bargeAudio.ended ? null : bargeAudio.currentTime);
  };
  bargeAudio.addEventListener("pause", bargeRest);
  bargeAudio.addEventListener("ended", bargeRest);

  function renderBarge() {
    const b = data.barge_in;
    if (!b) {
      document.querySelector(".barge").hidden = true;
      return;
    }
    bargeAudio.src = b.audio;
    bplay.disabled = false;
    $("barge-text").textContent =
      `A caller can talk over Parley and it stops. Here the caller’s next line (“${b.caller_text}”) was played over ` +
      `Parley's ${Math.round(b.agent_full)} second list of viewings, through the same turn-taking code the ` +
      `evaluation uses. The grey bars are what Parley never got to say.`;
    const stats = $("barge-stats");
    stats.replaceChildren();
    for (const [label, value, cls] of [["Caller starts", `${b.onset.toFixed(2)} s`, ""],
                                       ["Parley stops", `${b.stopped.toFixed(2)} s`, "stop"],
                                       ["Time to stop", `${b.latency.toFixed(2)} s`, ""]]) {
      const d = el("div", cls);
      d.appendChild(el("dt", null, label));
      d.appendChild(el("dd", null, value));
      stats.appendChild(d);
    }
    drawBarge(null);
  }

  // ---- results ------------------------------------------------------------------

  function countUp(node, value) {
    const m = /^(\d+(?:\.\d+)?)(.*)$/.exec(value);
    if (!m || reduce.matches) {
      node.textContent = value;
      return;
    }
    const end = parseFloat(m[1]);
    const places = (m[1].split(".")[1] || "").length;
    const t0 = performance.now();
    const step = (now) => {
      const k = Math.min(1, (now - t0) / 700);
      const e = 1 - Math.pow(1 - k, 3);
      node.textContent = `${(end * e).toFixed(places)}${m[2]}`;
      if (k < 1) requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
  }

  function renderNumbers(d) {
    const dl = $("numbers");
    const cells = [];
    for (const n of d.summary.tiles) {
      const row = el("div");
      row.appendChild(el("dt", null, n.label));
      const dd = el("dd", null, n.value);
      row.appendChild(dd);
      dl.appendChild(row);
      cells.push([dd, n.value]);
    }
    $("numbers-note").textContent = d.summary.note;
    if (!("IntersectionObserver" in window)) return;
    const io = new IntersectionObserver((entries) => {
      if (!entries.some((e) => e.isIntersecting)) return;
      io.disconnect();
      cells.forEach(([dd, v]) => countUp(dd, v));
    }, { threshold: 0.4 });
    io.observe(dl);
  }

  // ---- boot ---------------------------------------------------------------------

  function readColors() {
    colors = { accent: css("--accent"), caller: css("--caller"), line: css("--line-2"), ink: css("--ink"),
               ink3: css("--ink-3"), signal: css("--signal") };
  }

  let resizeFrame = 0;
  new ResizeObserver(() => {
    cancelAnimationFrame(resizeFrame);
    resizeFrame = requestAnimationFrame(() => {
      measure(wave);
      measure(bwave);
      if (current) drawWave(position());
      drawBarge(bargeAudio.paused ? null : bargeAudio.currentTime);
      const on = document.querySelector('.dial button[aria-pressed="true"]');
      if (on) {
        spring.placed = false;
        moveThumb(on);
      }
    });
  }).observe(document.body);

  setPlayState("loading");
  readColors();
  fetch("data/calls.json")
    .then((r) => {
      if (!r.ok) throw new Error(String(r.status));
      return r.json();
    })
    .then((d) => {
      data = d;
      renderNumbers(d);
      renderRejected();
      renderBarge();
      const asked = new URLSearchParams(location.search).get("call");
      select(d.calls.some((c) => c.id === asked) ? asked : "english");
      // Canvas text uses the web fonts, so redraw once they arrive.
      if (document.fonts) document.fonts.ready.then(() => { tick(); drawBarge(null); });
    })
    .catch(() => {
      setPlayState("paused");
      $("load-error").hidden = false;
      document.querySelectorAll(".dial button").forEach((b) => { b.disabled = true; });
    });
})();
