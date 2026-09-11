// Replays recorded evaluation calls: one audio file per call, transcript lines revealed on their timestamps.
// Every number and line comes from data/calls.json, which site/build.py writes from eval/ outputs.

(function () {
  const $ = (id) => document.getElementById(id);
  const audio = new Audio();
  audio.preload = "auto";
  let data = null;
  let current = null;
  let shown = [];
  // A seek before the audio has loaded is ignored by the browser, so the page keeps its own position too.
  let held = null;

  const fmt = (s) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`;
  const isArabic = (t) => /[\u0600-\u06FF]/.test(t || "");

  function el(tag, cls, text) {
    const n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text !== undefined) n.textContent = text;
    return n;
  }

  function bubble(turn) {
    const b = el("div", `msg ${turn.who}`);
    const who = el("div", "who", turn.who === "caller" ? "Caller, as heard" : "Parley");
    if (turn.tag) who.appendChild(el("span", "tag", turn.tag));
    b.appendChild(who);
    const t = el("div", "text", turn.text || "(nothing heard)");
    if (isArabic(turn.text)) t.dir = "rtl";
    b.appendChild(t);
    if (turn.latin) b.appendChild(el("div", "latin", turn.latin));
    const plain = (x) => (x || "").replace(/[\p{P}\u064B-\u0652]/gu, "").replace(/\s+/g, " ").trim().toLowerCase();
    if (turn.said && plain(turn.said) !== plain(turn.text)) b.appendChild(el("div", "said", `Script: ${turn.said}`));
    return b;
  }

  function renderCard(show) {
    const card = $("card");
    card.className = "card empty";
    card.replaceChildren(el("h2", null, "Booking"));
    const c = current;
    if (!show) {
      card.appendChild(el("p", "muted", "Appears when the booking API confirms it."));
      return;
    }
    if (c.booking) {
      card.className = "card ok";
      const dl = el("dl");
      for (const [k, v] of c.booking.rows) {
        dl.appendChild(el("dt", null, k));
        const dd = el("dd", null, v);
        if (isArabic(v)) dd.dir = "rtl";
        dl.appendChild(dd);
      }
      card.appendChild(dl);
    } else if (c.outage) {
      card.className = "card warn";
      card.replaceChildren(el("h2", null, c.outage.title));
      for (const line of c.outage.lines) card.appendChild(el("p", null, line));
    } else {
      card.appendChild(el("p", "muted", c.no_booking || "No booking was made."));
    }
  }

  function renderRejected() {
    const box = $("rejected");
    const g = data.grounding;
    box.replaceChildren();
    if (!g) {
      box.appendChild(el("p", "muted", "No rejected reply was recorded."));
      return;
    }
    box.appendChild(el("p", null, g.intro));
    const model = el("p");
    model.appendChild(el("span", "muted", "Model said: "));
    const s = el("s", null, g.model_reply);
    if (isArabic(g.model_reply)) s.dir = "rtl";
    model.appendChild(s);
    box.appendChild(model);
    box.appendChild(el("p", null, `Flagged: ${g.flagged}`));
    const spoken = el("p");
    spoken.appendChild(el("span", "muted", "Spoken instead: "));
    const sp = el("span", null, g.spoken);
    if (isArabic(g.spoken)) sp.dir = "rtl";
    spoken.appendChild(sp);
    box.appendChild(spoken);
    box.appendChild(el("p", "muted", g.footnote));
  }

  function select(id) {
    current = data.calls.find((c) => c.id === id);
    document.querySelectorAll(".calls button").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.call === id)));
    audio.pause();
    audio.src = current.audio;
    held = 0;
    $("play").textContent = "Play";
    $("note").textContent = current.note;
    const log = $("log");
    log.replaceChildren();
    shown = current.turns.map((t) => {
      const b = bubble(t);
      log.appendChild(b);
      return b;
    });
    renderCard(false);
    tick();
  }

  function tick() {
    if (!current) return;
    const now = held !== null ? held : audio.currentTime || 0;
    const dur = current.duration;
    $("progress").style.width = `${Math.min(100, (100 * now) / dur)}%`;
    $("clock").textContent = fmt(now);
    let last = null;
    current.turns.forEach((t, i) => {
      const on = now >= t.t0 - 0.05;
      shown[i].classList.toggle("on", on);
      shown[i].classList.toggle("now", on && now < t.t1);
      if (on) last = shown[i];
    });
    if (last && !audio.paused) last.scrollIntoView({ block: "nearest", behavior: "smooth" });
    const done = current.card_at !== null && now >= current.card_at;
    if ($("card").dataset.done !== String(done)) {
      $("card").dataset.done = String(done);
      renderCard(done);
    }
  }

  $("play").addEventListener("click", () => {
    if (audio.paused) {
      if (audio.ended) audio.currentTime = 0;
      if (held !== null) { audio.currentTime = held; held = null; }
      audio.play();
      $("play").textContent = "Pause";
    } else {
      audio.pause();
      $("play").textContent = "Play";
    }
  });
  audio.addEventListener("timeupdate", tick);
  audio.addEventListener("ended", () => { $("play").textContent = "Replay"; tick(); });
  document.querySelector(".bar").addEventListener("click", (e) => {
    const r = e.currentTarget.getBoundingClientRect();
    held = ((e.clientX - r.left) / r.width) * current.duration;
    audio.currentTime = held;
    if (!audio.paused) held = null;
    tick();
  });
  document.querySelectorAll(".calls button").forEach((b) => b.addEventListener("click", () => select(b.dataset.call)));
  $("show-rejected").addEventListener("change", (e) => { $("rejected").hidden = !e.target.checked; });

  // Frame-exact seeking for the committed screen recording, which is rendered frame by frame.
  window.parleySeek = (id, t) => {
    if (!current || current.id !== id) select(id);
    held = t;
    tick();
    const now = document.querySelector(".msg.now") || [...document.querySelectorAll(".msg.on")].pop();
    if (now) now.scrollIntoView({ block: "nearest" });
  };

  fetch("data/calls.json").then((r) => r.json()).then((d) => {
    data = d;
    const tiles = $("numbers");
    for (const n of d.summary.tiles) {
      const t = el("div", "tile");
      t.appendChild(el("b", null, n.value));
      t.appendChild(el("span", null, n.label));
      tiles.appendChild(t);
    }
    $("numbers-note").textContent = d.summary.note;
    renderRejected();
    select("english");
  });
})();
