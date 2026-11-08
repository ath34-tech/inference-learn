/* =========================================================
   InferenceLearn: shared site script
   header/footer · theme · progress · roadmap · curriculum
   lesson chrome · AI-tutor box · vocabulary + flashcards
   (calculators live in widgets.js, quiz arcade in practice.js)
   ========================================================= */
(function () {
  "use strict";

  const DATA = window.IE_DATA || { levels: [], lessons: [], pages: [] };
  const LEVELS = DATA.levels, LESSONS = DATA.lessons;
  const ROOT = (document.body.dataset.root || ".").replace(/\/$/, "");
  const PAGE = document.body.dataset.page || "";
  const href = (p) => ROOT + "/" + p;

  /* ---------------- helpers ---------------- */
  const store = {
    get(k, d) { try { const v = localStorage.getItem(k); return v === null ? d : JSON.parse(v); } catch (e) { return d; } },
    set(k, v) { try { localStorage.setItem(k, JSON.stringify(v)); } catch (e) { /* ignore */ } }
  };
  const slug = (s) => s.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "");
  const getDone = () => new Set(store.get("ie-progress", []));
  const setDone = (s) => store.set("ie-progress", [...s]);
  function fmtBytes(b) {
    const u = ["B", "KiB", "MiB", "GiB", "TiB"];
    let i = 0;
    while (Math.abs(b) >= 1024 && i < u.length - 1) { b /= 1024; i++; }
    return (Math.abs(b) >= 100 ? b.toFixed(0) : Math.abs(b) >= 10 ? b.toFixed(1) : b.toFixed(2)) + " " + u[i];
  }
  const fmt = (x, d = 1) => Number.isFinite(x) ? x.toLocaleString("en-US", { maximumFractionDigits: d }) : "–";

  /* ---------------- pixel art ---------------- */
  function pixelSVG(rows, scale, extraClass) {
    const h = rows.length, w = rows[0].length;
    let r = "";
    rows.forEach((row, y) => {
      [...row].forEach((c, x) => {
        if (c !== ".") r += `<rect x="${x}" y="${y}" width="1.02" height="1.02" class="px-${c}"/>`;
      });
    });
    return `<svg class="${extraClass || ""}" viewBox="0 0 ${w} ${h}" width="${w * scale}" height="${h * scale}" shape-rendering="crispEdges" aria-hidden="true">${r}</svg>`;
  }
  const ART = {
    chip: ["...k..k..k..k...", "...k..k..k..k...", ".kkkkkkkkkkkkkk.", ".kppppppppppppk.", "kkpwwwwwwwwwwpkk",
      ".kpwiwwwwwwiwpk.", "kkpwiwwwwwwiwpkk", ".kpwwwwwwwwwwpk.", "kkpwdwwwwwwdwpkk", ".kpwwwiiiiwwwpk.",
      "kkpwwwwwwwwwwpkk", ".kppppppppppppk.", ".kkkkkkkkkkkkkk.", "...k..k..k..k...", "...k..k..k..k...", "................"],
    heart: [".kk.kk.", "kppkppk", "kpppppk", ".kpppk.", "..kpk..", "...k..."],
    heartEmpty: [".kk.kk.", "k..k..k", "k.....k", ".k...k.", "..k.k..", "...k..."],
    star: ["..y..", "..y..", "yyyyy", "..y..", "..y.."],
    sun: ["....y....", ".y..y..y.", "..yyyyy..", "..yyyyy..", "yyyyyyyyy", "..yyyyy..", "..yyyyy..", ".y..y..y.", "....y...."],
    moon: ["..llll...", ".lll.....", "lll......", "lll......", "lll......", "lll......", "llll...l.", ".lllllll.", "..llll..."],
    trophy: ["yyyyyyy", "yyyyyyy", ".yyyyy.", "..yyy..", "...y...", "..yyy..", ".kkkkk."]
  };
  window.IE = { DATA, LESSONS, LEVELS, pixelSVG, ART, store, fmt, fmtBytes, slug, href };

  /* ---------------- theme ---------------- */
  const themeNow = () => document.documentElement.getAttribute("data-theme") || "light";
  const themeIcon = () => pixelSVG(themeNow() === "dark" ? ART.sun : ART.moon, 3);
  function toggleTheme() {
    const next = themeNow() === "dark" ? "light" : "dark";
    document.documentElement.setAttribute("data-theme", next);
    try { localStorage.setItem("ie-theme", next); } catch (e) { /* ignore */ }
    const b = document.getElementById("theme-toggle");
    if (b) { b.innerHTML = themeIcon(); b.setAttribute("aria-label", next === "dark" ? "Switch to light theme" : "Switch to dark theme"); }
    document.dispatchEvent(new Event("ie-theme"));
  }

  /* ---------------- header / footer ---------------- */
  function renderChrome() {
    const nav = [
      ["home", "index.html", "Home"],
      ["roadmap", "roadmap.html", "Roadmap"],
      ["learn", "learn.html", "Lessons"],
      ["practice", "practice.html", "Practice"],
      ["vocabulary", "vocabulary.html", "Vocab"],
      ["resources", "resources.html", "Resources"]
    ];
    const active = document.body.dataset.lesson ? "learn" : PAGE;
    const h = document.getElementById("site-header");
    if (h) {
      h.outerHTML = `
      <header class="site-header">
        <div class="container header-inner">
          <a class="brand" href="${href("index.html")}" aria-label="InferenceLearn home">
            ${pixelSVG(ART.chip, 2.5)}
            <span class="brand-text">Inference<span>Learn</span></span>
          </a>
          <button class="icon-btn nav-toggle" id="nav-toggle" aria-label="Open menu" aria-expanded="false">≡</button>
          <nav class="site-nav" id="site-nav" aria-label="Main">
            ${nav.map(([k, f, t]) => `<a href="${href(f)}" class="${k === active ? "active" : ""}">${t}</a>`).join("")}
          </nav>
          <button class="icon-btn" id="theme-toggle" aria-label="Toggle theme">${themeIcon()}</button>
        </div>
      </header>`;
      document.getElementById("theme-toggle").addEventListener("click", toggleTheme);
      const nt = document.getElementById("nav-toggle"), sn = document.getElementById("site-nav");
      nt.addEventListener("click", () => nt.setAttribute("aria-expanded", sn.classList.toggle("open")));
    }
    const f = document.getElementById("site-footer");
    if (f) {
      f.outerHTML = `
      <footer class="site-footer">
        <div class="container footer-inner">
          <p>${pixelSVG(ART.heart, 3)} &nbsp;InferenceLearn: learn how LLMs are served fast and cheap.</p>
          <div class="footer-links">
            <a href="${href("roadmap.html")}">Roadmap</a>
            <a href="${href("practice.html")}">Quiz</a>
            <a href="${href("interview.html")}">Interview prep</a>
            <a href="${href("workbook.html")}">Workbook</a>
            <a href="${href("vocabulary.html")}">Vocabulary</a>
            <a href="${href("llms.txt")}" title="Plain-text index for AI assistants">llms.txt</a>
          </div>
        </div>
      </footer>`;
    }
    const top = document.createElement("button");
    top.className = "icon-btn to-top";
    top.setAttribute("aria-label", "Back to top");
    top.innerHTML = '<span style="font-family:var(--f-pixel);font-size:12px">▲</span>';
    top.addEventListener("click", () => window.scrollTo({ top: 0 }));
    document.body.appendChild(top);
    window.addEventListener("scroll", () => top.classList.toggle("show", window.scrollY > 700), { passive: true });
  }

  /* ---------------- progress ---------------- */
  function renderProgress() {
    const done = getDone();
    const n = LESSONS.filter((l) => done.has(l.id)).length;
    const pct = Math.round((n / LESSONS.length) * 100);
    document.querySelectorAll("[data-progress]").forEach((el) => {
      el.innerHTML = `<div class="pbar" role="progressbar" aria-valuemin="0" aria-valuemax="100" aria-valuenow="${pct}"><span style="width:${pct}%"></span></div>
        <div class="progress-meta"><span>${n} / ${LESSONS.length} lessons cleared</span><span>${pct}%</span></div>`;
    });
    const next = LESSONS.find((l) => !done.has(l.id));
    document.querySelectorAll("[data-continue]").forEach((a) => {
      if (next) { a.href = href(next.file); a.textContent = n ? `Continue: ${next.n}` : "Start lesson 1-1"; }
      else { a.href = href("practice.html"); a.textContent = "All clear! Play the quiz"; }
    });
  }

  /* ---------------- roadmap & curriculum (re-render static HTML with checkmarks) ---------------- */
  function renderRoadmap() {
    const el = document.getElementById("roadmap");
    if (!el) return;
    const done = getDone();
    el.innerHTML = LEVELS.map((lv) => {
      const ls = LESSONS.filter((l) => l.level === lv.id);
      const all = ls.every((l) => done.has(l.id));
      return `
      <div class="stage ${all ? "complete" : ""}">
        <div class="card">
          <div class="stage-head"><span class="badge ${lv.cls}">${lv.world}</span><span class="stage-world">${ls.length} lessons · ~${ls.reduce((a, l) => a + l.mins, 0)} min</span></div>
          <h3>${lv.name}</h3>
          <p class="muted" style="font-size:13.5px;margin:0">${lv.blurb}</p>
          <ul class="stage-list">
            ${ls.map((l) => `<li><span class="chk ${done.has(l.id) ? "on" : ""}"></span><a href="${href(l.file)}">${l.n} · ${l.title}</a><span class="sub">${l.topics.join(" · ")}</span></li>`).join("")}
          </ul>
        </div>
      </div>`;
    }).join("");
  }
  function renderCurriculum() {
    const el = document.getElementById("curriculum");
    if (!el) return;
    const done = getDone();
    el.innerHTML = LEVELS.map((lv) => `
      <section class="level-block" id="w${lv.id}" style="scroll-margin-top:90px">
        <h2><span class="badge ${lv.cls}" style="font-size:20px">${lv.world}</span> ${lv.name}</h2>
        <p class="muted">${lv.blurb}</p>
        <div class="grid grid-3">
          ${LESSONS.filter((l) => l.level === lv.id).map((l) => `
            <a class="card lesson-card ${done.has(l.id) ? "is-done" : ""}" href="${href(l.file)}">
              ${done.has(l.id) ? '<span class="badge l1 done-flag">✓ cleared</span>' : ""}
              <span class="num">${l.n}</span>
              <h3>${l.title}</h3>
              <p>${l.desc}</p>
              <div class="meta"><span>${l.topics.length} topics</span><span>~${l.mins} min</span></div>
            </a>`).join("")}
        </div>
      </section>`).join("");
  }

  /* ---------------- AI tutor box ---------------- */
  function aiBox(el) {
    const url = location.href.split("#")[0];
    const md = url.replace(/\.html$/, ".md").replace(/\/$/, "/index.md");
    const full = new URL(href("llms-full.txt"), location.href).href;
    const title = (document.querySelector("h1") || {}).textContent || document.title;
    const prompt =
`You are my patient tutor for inference engineering.
Please read this page: ${url}
(plain-text version: ${md} · whole course: ${full})

Teach me "${title.trim()}" step by step:
1. First ask me 2 short questions to find out what I already know.
2. Explain one section at a time, with an analogy and a small worked number example.
3. After each section, ask me one quiz question and wait for my answer.
4. Finish with 3 interview-style questions and give feedback on my answers.`;
    const isFile = location.protocol === "file:";
    el.innerHTML = `
      <div class="ai-box">
        <div class="ai-head">${pixelSVG(ART.chip, 2)}<div><h4>Study this with an AI tutor</h4>
          <p>Give this page's link to ChatGPT, Claude or Gemini, or use the buttons. Every page also has a plain-text <a href="${md}">.md version</a>, and the whole course is in <a href="${full}">llms-full.txt</a>.</p></div></div>
        ${isFile ? `<p class="ai-warn">⚠ You're viewing a local file. An AI can't open <code>file://</code> links. Host the site (e.g. GitHub Pages), or download the .md file and upload it to the chat.</p>` : ""}
        <pre class="ai-prompt"><code>${prompt.replace(/&/g, "&amp;").replace(/</g, "&lt;")}</code></pre>
        <div class="ai-actions">
          <button class="btn small" data-ai="copy">Copy prompt</button>
          <a class="btn small alt" target="_blank" rel="noopener" href="https://chatgpt.com/?q=${encodeURIComponent(prompt)}">Open in ChatGPT</a>
          <a class="btn small alt" target="_blank" rel="noopener" href="https://claude.ai/new?q=${encodeURIComponent(prompt)}">Open in Claude</a>
          <a class="btn small alt" data-ai="gemini" target="_blank" rel="noopener" href="https://gemini.google.com/app">Copy + open Gemini</a>
        </div>
      </div>`;
    const copy = () => navigator.clipboard && navigator.clipboard.writeText(prompt);
    el.querySelector('[data-ai="copy"]').addEventListener("click", (e) => {
      copy(); e.target.textContent = "Copied!"; setTimeout(() => (e.target.textContent = "Copy prompt"), 1300);
    });
    el.querySelector('[data-ai="gemini"]').addEventListener("click", copy);
  }

  /* ---------------- lesson page ---------------- */
  function initLesson() {
    const id = document.body.dataset.lesson;
    if (!id) return;
    const i = LESSONS.findIndex((l) => l.id === id);
    const L = LESSONS[i];
    const lv = LEVELS.find((v) => v.id === L.level);

    const head = document.getElementById("lesson-meta");
    if (head) {
      head.innerHTML = `<div class="crumbs"><a href="${href("learn.html")}">Lessons</a> / ${lv.world} ${lv.name} / ${L.n}</div>
        <span class="badge ${lv.cls}">${lv.world}</span><span class="badge">Lesson ${L.n}</span><span class="badge">~${L.mins} min read</span>`;
    }

    const article = document.querySelector(".lesson");
    const hs = [...article.querySelectorAll("h2")];
    hs.forEach((h, k) => { if (!h.id) h.id = "s-" + (k + 1); });
    const toc = document.getElementById("toc");
    if (toc) {
      toc.innerHTML = `<div class="card"><h4>On this page</h4><ol>${hs.map((h) => `<li><a href="#${h.id}">${h.textContent.replace(/^▶\s*/, "")}</a></li>`).join("")}</ol><div class="toc-progress" data-progress></div></div>`;
      const links = [...toc.querySelectorAll("a")];
      const io = new IntersectionObserver((ents) => {
        ents.forEach((e) => {
          if (e.isIntersecting) links.forEach((a) => a.classList.toggle("active", a.getAttribute("href") === "#" + e.target.id));
        });
      }, { rootMargin: "-90px 0px -70% 0px" });
      hs.forEach((h) => io.observe(h));
    }

    const foot = document.getElementById("lesson-foot");
    if (foot) {
      const prev = LESSONS[i - 1], next = LESSONS[i + 1];
      foot.innerHTML = `
        <div data-ai-box></div>
        <div class="lesson-foot">
          <div><span class="kicker">Finished reading?</span> <a href="${href("practice.html")}#arcade">Test yourself in the Quiz Arcade ▶</a></div>
          <button class="btn" id="mark-done"></button>
        </div>
        <div class="pager">
          ${prev ? `<a class="card prev" href="${href(prev.file)}"><span class="dir">◀ previous · ${prev.n}</span><span class="t">${prev.title}</span></a>` : "<span></span>"}
          ${next ? `<a class="card next" href="${href(next.file)}"><span class="dir">next · ${next.n} ▶</span><span class="t">${next.title}</span></a>`
                 : `<a class="card next" href="${href("interview.html")}"><span class="dir">you made it ▶</span><span class="t">Interview prep</span></a>`}
        </div>`;
      const b = document.getElementById("mark-done");
      const paint = () => {
        const d = getDone().has(id);
        b.textContent = d ? "✓ Cleared!" : "Mark as cleared";
        b.classList.toggle("done", d);
      };
      b.addEventListener("click", () => {
        const s = getDone();
        s.has(id) ? s.delete(id) : s.add(id);
        setDone(s); paint(); renderProgress();
      });
      paint();
    }
  }

  /* ---------------- code copy buttons ---------------- */
  function initCopy() {
    document.querySelectorAll("pre:not(.ai-prompt)").forEach((pre) => {
      const b = document.createElement("button");
      b.className = "copy-btn"; b.type = "button"; b.textContent = "copy";
      b.addEventListener("click", () => {
        const txt = (pre.querySelector("code") || pre).innerText;
        if (navigator.clipboard) navigator.clipboard.writeText(txt).then(() => { b.textContent = "copied!"; setTimeout(() => (b.textContent = "copy"), 1200); }, () => {});
      });
      pre.appendChild(b);
    });
  }

  /* ---------------- resource filters ---------------- */
  function initFilters() {
    const bar = document.getElementById("res-filters");
    if (!bar) return;
    bar.addEventListener("click", (e) => {
      const b = e.target.closest(".chip");
      if (!b) return;
      bar.querySelectorAll(".chip").forEach((c) => c.classList.toggle("on", c === b));
      const t = b.dataset.f;
      document.querySelectorAll(".res").forEach((r) => r.classList.toggle("hide", t !== "all" && r.dataset.type !== t));
      document.querySelectorAll(".res-group").forEach((g) => { g.style.display = g.querySelector(".res:not(.hide)") ? "" : "none"; });
    });
  }

  /* ---------------- interview page filters ---------------- */
  function initInterview() {
    const bar = document.getElementById("iv-filters");
    if (!bar) return;
    bar.addEventListener("click", (e) => {
      const b = e.target.closest(".chip");
      if (!b) return;
      bar.querySelectorAll(".chip").forEach((c) => c.classList.toggle("on", c === b));
      const lv = b.dataset.f;
      document.querySelectorAll(".iv").forEach((q) => { q.style.display = lv === "all" || q.dataset.level === lv ? "" : "none"; });
    });
    const t = document.getElementById("iv-expand");
    if (t) t.addEventListener("click", () => {
      const open = t.dataset.open !== "1";
      document.querySelectorAll(".iv").forEach((d) => (d.open = open));
      t.dataset.open = open ? "1" : "0";
      t.textContent = open ? "Collapse all" : "Expand all";
    });
  }

  /* ---------------- vocabulary: search, categories, flashcards ---------------- */
  function initVocab() {
    const cards = [...document.querySelectorAll(".vocab-card")];
    if (!cards.length) return;
    const q = document.getElementById("vocab-q");
    const cats = document.getElementById("vocab-cats");
    const cnt = document.getElementById("vocab-count");
    let cat = "all";
    const known = new Set(store.get("ie-known", []));

    function filter() {
      const s = (q.value || "").trim().toLowerCase();
      let n = 0;
      cards.forEach((c) => {
        const m = (cat === "all" || c.dataset.cat === cat) && (!s || c.textContent.toLowerCase().includes(s));
        c.classList.toggle("hide", !m); if (m) n++;
      });
      document.querySelectorAll(".vocab-group").forEach((g) => { g.style.display = g.querySelector(".vocab-card:not(.hide)") ? "" : "none"; });
      cnt.textContent = `${n} term${n === 1 ? "" : "s"} · ${known.size} marked known`;
    }
    q.addEventListener("input", filter);
    cats.addEventListener("click", (e) => {
      const b = e.target.closest(".chip"); if (!b) return;
      cats.querySelectorAll(".chip").forEach((c) => c.classList.toggle("on", c === b));
      cat = b.dataset.f; filter(); deal();
    });

    // tabs
    const tabs = document.getElementById("vocab-tabs");
    tabs.addEventListener("click", (e) => {
      const b = e.target.closest(".chip"); if (!b) return;
      tabs.querySelectorAll(".chip").forEach((c) => c.classList.toggle("on", c === b));
      document.getElementById("vocab-list").hidden = b.dataset.tab !== "list";
      document.getElementById("vocab-flash").hidden = b.dataset.tab !== "flash";
      if (b.dataset.tab === "flash") deal();
    });

    // flashcards
    let deck = [], pos = 0;
    const fc = document.getElementById("flashcard");
    const fcFront = document.getElementById("fc-front"), fcBack = document.getElementById("fc-back");
    const fcMeta = document.getElementById("fc-meta");
    const onlyNew = document.getElementById("fc-only-new");
    function deal() {
      deck = cards.filter((c) => cat === "all" || c.dataset.cat === cat)
        .map((c) => ({ id: c.id, term: c.querySelector("dt").textContent, def: c.querySelector("dd").innerHTML, cat: c.dataset.cat }))
        .filter((d) => !onlyNew.checked || !known.has(d.id));
      for (let i = deck.length - 1; i > 0; i--) { const j = Math.floor(Math.random() * (i + 1)); [deck[i], deck[j]] = [deck[j], deck[i]]; }
      pos = 0; show();
    }
    function show() {
      fc.classList.remove("flipped");
      if (!deck.length) { fcFront.innerHTML = "<span>🎉 No cards left!</span>"; fcBack.innerHTML = ""; fcMeta.textContent = "Untick 'only unknown' or pick another category."; return; }
      const d = deck[pos % deck.length];
      fcFront.innerHTML = `<span class="badge">${d.cat}</span><div class="fc-term">${d.term}</div><span class="muted fc-hint">click or press space to flip</span>`;
      fcBack.innerHTML = `<div class="fc-term small">${d.term}</div><div class="fc-def">${d.def}</div>`;
      fcMeta.textContent = `card ${(pos % deck.length) + 1} / ${deck.length} · ${known.size} known in total`;
    }
    fc.addEventListener("click", () => fc.classList.toggle("flipped"));
    document.getElementById("fc-know").addEventListener("click", () => {
      if (!deck.length) return;
      known.add(deck[pos % deck.length].id); store.set("ie-known", [...known]);
      if (onlyNew.checked) { deck.splice(pos % deck.length, 1); if (pos >= deck.length) pos = 0; } else pos++;
      show(); filter();
    });
    document.getElementById("fc-again").addEventListener("click", () => { pos++; show(); });
    document.getElementById("fc-shuffle").addEventListener("click", deal);
    document.getElementById("fc-reset").addEventListener("click", () => { known.clear(); store.set("ie-known", []); deal(); filter(); });
    onlyNew.addEventListener("change", deal);
    document.addEventListener("keydown", (e) => {
      if (document.getElementById("vocab-flash").hidden || e.target.matches("input,textarea")) return;
      if (e.code === "Space") { e.preventDefault(); fc.classList.toggle("flipped"); }
      if (e.key === "ArrowRight") { pos++; show(); }
    });
    if (location.hash === "#flashcards") tabs.querySelector('[data-tab="flash"]').click();
    filter(); deal();
  }

  /* ---------------- decorative art slots ---------------- */
  function initArt() {
    document.querySelectorAll("[data-art]").forEach((el) => {
      const a = ART[el.dataset.art];
      if (a) el.innerHTML = pixelSVG(a, +(el.dataset.scale || 4));
    });
  }

  /* ---------------- boot ---------------- */
  renderChrome();
  initArt();
  renderRoadmap();
  renderCurriculum();
  initLesson();
  document.querySelectorAll("[data-ai-box]").forEach(aiBox);
  renderProgress();
  initCopy();
  initFilters();
  initInterview();
  initVocab();
})();
