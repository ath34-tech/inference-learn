/* =========================================================
   InferenceLearn: Quiz Arcade
   Reads questions from the static #qbank list (so the bank is
   also readable by people and LLMs without JavaScript).
   ========================================================= */
(function () {
  "use strict";
  const IE = window.IE;
  const arcade = document.getElementById("arcade");
  const bank = document.getElementById("qbank");
  if (!IE || !arcade || !bank) return;
  const { pixelSVG, ART, store, href } = IE;

  const QUESTIONS = [...bank.querySelectorAll(":scope > li")].map((li) => ({
    topic: li.dataset.topic,
    lesson: li.dataset.lesson,
    answer: "ABCD".indexOf(li.dataset.a),
    q: li.querySelector(".q").innerHTML,
    opts: [...li.querySelectorAll(".opts > li")].map((o) => o.innerHTML),
    ex: li.querySelector(".ex").innerHTML.replace(/^<strong>Answer:[^<]*<\/strong>\s*/, "")
  }));
  const TOPICS = [
    ["all", "Everything"], ["foundations", "Foundations"], ["memory", "KV & batching"],
    ["attention", "Attention & quant"], ["advanced", "Advanced"], ["systems", "Kernels & prod"]
  ];
  const RANKS = [[0, "Cold Start 🧊"], [40, "Token Intern"], [60, "Batch Wizard"], [80, "Kernel Knight"], [100, "Speed-of-Light Legend ⚡"]];
  const LIVES = 3, ROUND = 10;
  let topic = "all", deck = [], i = 0, lives, score, streak, best, correct, missed, locked;

  const hearts = () => Array.from({ length: LIVES }, (_, k) => pixelSVG(k < lives ? ART.heart : ART.heartEmpty, 3)).join("");
  const lessonOf = (id) => IE.LESSONS.find((l) => l.id === id);

  function start() {
    const pool = QUESTIONS.filter((q) => topic === "all" || q.topic === topic);
    deck = pool.slice().sort(() => Math.random() - 0.5).slice(0, ROUND);
    i = 0; lives = LIVES; score = 0; streak = 0; best = 0; correct = 0; missed = []; locked = false;
    ask();
  }

  function title() {
    const hi = store.get("ie-hiscore", 0);
    arcade.innerHTML = `
      <div class="arcade-screen center">
        <div class="arcade-logo">QUIZ ARCADE</div>
        <p class="blink-text">INSERT BRAIN TO START</p>
        <p class="muted">${ROUND} questions · ${LIVES} lives · streaks multiply your points</p>
        <div class="filters" id="topic-chips">${TOPICS.map(([k, t]) => `<button class="chip ${k === topic ? "on" : ""}" data-t="${k}">${t} (${k === "all" ? QUESTIONS.length : QUESTIONS.filter((q) => q.topic === k).length})</button>`).join("")}</div>
        <button class="btn" id="go">▶ PRESS START</button>
        <p class="hiscore">HI-SCORE ${String(hi).padStart(5, "0")}</p>
      </div>`;
    arcade.querySelector("#topic-chips").addEventListener("click", (e) => {
      const b = e.target.closest(".chip"); if (!b) return;
      topic = b.dataset.t; title();
    });
    arcade.querySelector("#go").addEventListener("click", start);
  }

  function ask() {
    if (i >= deck.length || lives <= 0) return end();
    const Q = deck[i];
    locked = false;
    arcade.innerHTML = `
      <div class="arcade-screen">
        <div class="hud">
          <span class="hearts">${hearts()}</span>
          <span>Q ${i + 1}/${deck.length}</span>
          <span>STREAK ×${streak}</span>
          <span>SCORE ${String(score).padStart(5, "0")}</span>
        </div>
        <div class="q-text">${Q.q}</div>
        <div class="opts-grid">${Q.opts.map((o, k) => `<button class="opt" data-k="${k}"><span class="opt-key">${"ABCD"[k]}</span><span>${o}</span></button>`).join("")}</div>
        <div id="feedback"></div>
      </div>`;
    arcade.querySelectorAll(".opt").forEach((b) => b.addEventListener("click", () => pick(+b.dataset.k)));
  }

  function pick(k) {
    if (locked) return;
    locked = true;
    const Q = deck[i], ok = k === Q.answer;
    const btns = arcade.querySelectorAll(".opt");
    btns.forEach((b, n) => { b.disabled = true; if (n === Q.answer) b.classList.add("right"); if (n === k && !ok) b.classList.add("wrong"); });
    let gained = 0;
    if (ok) { streak++; best = Math.max(best, streak); correct++; gained = 100 + 50 * Math.min(streak - 1, 4); score += gained; }
    else { streak = 0; lives--; missed.push(Q); arcade.querySelector(".arcade-screen").classList.add("shake"); }
    arcade.querySelector(".hud").innerHTML = `<span class="hearts">${hearts()}</span><span>Q ${i + 1}/${deck.length}</span><span>STREAK ×${streak}</span><span>SCORE ${String(score).padStart(5, "0")}</span>`;
    const L = lessonOf(Q.lesson);
    arcade.querySelector("#feedback").innerHTML = `
      <div class="fb ${ok ? "ok" : "bad"}">
        <div class="fb-title">${ok ? `+${gained} NICE!${streak > 1 ? ` ${streak} in a row 🔥` : ""}` : lives > 0 ? "OOF, −1 LIFE" : "GAME OVER"}</div>
        <p>${Q.ex}</p>
        ${L ? `<p class="muted" style="font-size:13px">Review: <a href="${href(L.file)}">${L.n} ${L.title}</a></p>` : ""}
        <button class="btn small" id="next">${i + 1 >= deck.length || lives <= 0 ? "See results ▶" : "Next ▶"}</button>
      </div>`;
    const nx = arcade.querySelector("#next");
    nx.focus({ preventScroll: true });
    nx.addEventListener("click", () => { i++; ask(); });
  }

  function end() {
    const pct = Math.round((correct / deck.length) * 100);
    const rank = RANKS.filter(([p]) => pct >= p).pop()[1];
    const hi = store.get("ie-hiscore", 0);
    const newHi = score > hi;
    if (newHi) store.set("ie-hiscore", score);
    arcade.innerHTML = `
      <div class="arcade-screen center">
        <div>${pixelSVG(ART.trophy, 8)}</div>
        <div class="arcade-logo small">${lives > 0 ? "STAGE CLEAR" : "GAME OVER"}</div>
        <p class="rank">RANK: ${rank}</p>
        <div class="readout" style="max-width:560px;margin:16px auto">
          <div class="r"><div class="k">score</div><span class="v">${score}</span></div>
          <div class="r"><div class="k">correct</div><span class="v">${correct}/${deck.length}</span></div>
          <div class="r"><div class="k">best streak</div><span class="v">×${best}</span></div>
        </div>
        ${newHi ? '<p class="blink-text">★ NEW HI-SCORE ★</p>' : `<p class="hiscore">HI-SCORE ${String(Math.max(hi, score)).padStart(5, "0")}</p>`}
        ${missed.length ? `<div class="missed"><h4>Review what you missed</h4>${missed.map((q) => {
          const L = lessonOf(q.lesson);
          return `<details class="quiz"><summary>${q.q}</summary><div class="ans"><p><strong>Answer: ${"ABCD"[q.answer]}. ${q.opts[q.answer]}</strong></p><p>${q.ex}</p>${L ? `<p><a href="${href(L.file)}">Review ${L.n} ${L.title}</a></p>` : ""}</div></details>`;
        }).join("")}</div>` : ""}
        <button class="btn" id="again">▶ PLAY AGAIN</button>
        <button class="btn alt" id="menu">Change topic</button>
      </div>`;
    arcade.querySelector("#again").addEventListener("click", start);
    arcade.querySelector("#menu").addEventListener("click", title);
  }

  document.addEventListener("keydown", (e) => {
    if (!arcade.querySelector(".opts-grid") || e.target.matches("input,textarea")) return;
    const k = "abcd1234".indexOf(e.key.toLowerCase());
    if (k >= 0 && !locked) pick(k % 4);
    else if (e.key === "Enter" && locked && !e.target.matches("button, a")) { const n = arcade.querySelector("#next"); if (n) n.click(); }
  });

  title();
})();
