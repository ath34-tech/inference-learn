/* =========================================================
   InferenceLearn: interactive widgets
   Each widget shows: inputs → live "show the math" steps → a visual.
   ========================================================= */
(function () {
  "use strict";
  const IE = window.IE;
  if (!IE) return;
  const { fmt, fmtBytes } = IE;
  const $ = (root, s) => root.querySelector(s);
  const B = (x) => `<b>${x}</b>`;
  const math = (el, lines) => { el.innerHTML = `<div class="math-title">▶ show the math</div>` + lines.map((l) => `<div class="ml">${l}</div>`).join(""); };
  const readout = (el, items) => { el.innerHTML = items.map(([k, v, cls]) => `<div class="r ${cls || ""}"><div class="k">${k}</div><span class="v">${v}</span></div>`).join(""); };
  const onInput = (w, fn) => { w.querySelectorAll("input,select").forEach((x) => x.addEventListener("input", fn)); fn(); };
  const ms = (s) => (s * 1e3 >= 100 ? fmt(s * 1e3, 0) : fmt(s * 1e3, 2)) + " ms";
  const GB = (b) => fmt(b / 1e9, 1) + " GB";

  /* ============ 1. KV cache calculator + GPU memory bar ============ */
  function kvCalc() {
    const w = document.getElementById("kv-calc");
    if (!w) return;
    const PRESETS = {
      "llama3-8b": { layers: 32, kvh: 8, hd: 128, p: 8 },
      "llama3-70b": { layers: 80, kvh: 8, hd: 128, p: 70 },
      "llama2-7b": { layers: 32, kvh: 32, hd: 128, p: 6.7 },
      "mistral-7b": { layers: 32, kvh: 8, hd: 128, p: 7.2 },
      "qwen2.5-7b": { layers: 28, kvh: 4, hd: 128, p: 7.6 }
    };
    const v = (s) => +$(w, s).value;
    $(w, "#kv-preset").addEventListener("change", (e) => {
      const p = PRESETS[e.target.value];
      if (p) { $(w, "#kv-layers").value = p.layers; $(w, "#kv-heads").value = p.kvh; $(w, "#kv-dim").value = p.hd; $(w, "#kv-params").value = p.p; }
    });
    onInput(w, () => {
      const L = v("#kv-layers"), H = v("#kv-heads"), D = v("#kv-dim"), S = v("#kv-seq"), N = v("#kv-batch"), by = v("#kv-dtype");
      const gpu = v("#kv-gpu") * 1e9, P = v("#kv-params") * 1e9, wb = v("#kv-wbytes");
      const perTok = 2 * L * H * D * by, perSeq = perTok * S, total = perSeq * N;
      const usable = gpu * 0.9, weights = P * wb, ws = 2e9;
      const free = usable - weights - ws;
      const maxSeqs = Math.max(0, Math.floor(free / perSeq));
      const fits = total <= free;

      readout($(w, "#kv-out"), [
        ["per token", fmtBytes(perTok)],
        [`per sequence (${fmt(S, 0)} tok)`, fmtBytes(perSeq)],
        [`all ${fmt(N, 0)} sequences`, fmtBytes(total)],
        ["max sequences that fit", fmt(maxSeqs, 0), fits ? "good" : "bad"]
      ]);
      math($(w, "#kv-math"), [
        `per token = 2 (K and V) × ${B(L)} layers × ${B(H)} KV heads × ${B(D)} head dim × ${B(by)} bytes`,
        `&nbsp;&nbsp;= ${B(fmt(perTok, 0))} bytes = ${B(fmtBytes(perTok))}`,
        `per sequence = ${fmtBytes(perTok)} × ${B(fmt(S, 0))} tokens = ${B(fmtBytes(perSeq))}`,
        `whole batch = ${fmtBytes(perSeq)} × ${B(fmt(N, 0))} sequences = ${B(fmtBytes(total))}`,
        `KV room = ${GB(gpu)} × 0.9 − ${GB(weights)} weights − 2 GB workspace = ${B(GB(Math.max(0, free)))}`,
        `max sequences = ${GB(Math.max(0, free))} ÷ ${fmtBytes(perSeq)} = ${B(fmt(maxSeqs, 0))}`
      ]);

      // memory bar
      const pct = (x) => Math.max(0, Math.min(100, (x / usable) * 100));
      const kvShown = Math.min(total, Math.max(0, free));
      $(w, "#kv-mem").innerHTML = `
        <div class="membar">
          <span class="seg sw" style="width:${pct(weights)}%" title="weights">${pct(weights) > 9 ? "weights" : ""}</span>
          <span class="seg so" style="width:${pct(ws)}%" title="workspace"></span>
          <span class="seg sk ${fits ? "" : "over"}" style="width:${pct(kvShown)}%" title="KV cache">${pct(kvShown) > 9 ? "KV cache" : ""}</span>
        </div>
        <div class="membar-legend"><span><i class="sw"></i>weights ${GB(weights)}</span><span><i class="so"></i>workspace 2 GB</span><span><i class="sk"></i>KV ${GB(total)}</span><span>usable ${GB(usable)}</span></div>
        <div class="verdict ${fits ? "ok" : "oom"}">${weights + ws > usable ? "💥 The weights alone don't fit. Use a bigger GPU, quantize, or shard with tensor parallelism."
          : fits ? `✓ FITS! ${fmt(N, 0)} sequences use ${fmt((total / free) * 100, 0)}% of the KV room.`
          : `💥 OUT OF MEMORY by ${GB(total - free)}. Only ${fmt(maxSeqs, 0)} of your ${fmt(N, 0)} sequences fit. Try FP8 KV, a shorter context, or a GQA model.`}</div>`;

      // one pixel per sequence (or group of sequences)
      const per = Math.max(1, Math.ceil(N / 96));
      const boxes = Math.ceil(N / per);
      let cells = "";
      for (let i = 0; i < boxes; i++) cells += `<span class="seqpx ${i * per < maxSeqs ? "in" : "out"}"></span>`;
      $(w, "#kv-seqs").innerHTML = `<div class="seqgrid">${cells}</div><div class="cap-s">each square = ${per} sequence${per > 1 ? "s" : ""} · green fits, red doesn't</div>`;
    });
  }

  /* ============ 2. Speed-of-light + roofline ============ */
  function solCalc() {
    const w = document.getElementById("sol-calc");
    if (!w) return;
    const GPUS = {
      h100: { name: "H100 SXM", tflops: 989, bw: 3.35 },
      h200: { name: "H200 SXM", tflops: 989, bw: 4.8 },
      a100: { name: "A100 80GB", tflops: 312, bw: 2.04 },
      l4: { name: "L4", tflops: 121, bw: 0.3 },
      b200: { name: "B200", tflops: 2250, bw: 8.0 }
    };
    onInput(w, () => {
      const g = GPUS[$(w, "#sol-gpu").value];
      const Pb = +$(w, "#sol-params").value, P = Pb * 1e9;
      const bpp = +$(w, "#sol-bytes").value;
      const Bt = Math.pow(2, +$(w, "#sol-batch").value);
      $(w, "#sol-batch-v").textContent = Bt;
      const prompt = +$(w, "#sol-prompt").value;
      const F = g.tflops * 1e12, BW = g.bw * 1e12;
      const wBytes = P * bpp;
      const tMem = wBytes / BW, tCmp = (2 * P * Bt) / F, step = Math.max(tMem, tCmp);
      const ridge = F / BW, ai = (2 * P * Bt) / wBytes;
      const prefill = (2 * P * prompt) / F;
      const memBound = tMem >= tCmp;

      readout($(w, "#sol-out"), [
        ["min decode step", ms(step)],
        ["tok/s per user", fmt(1 / step, 0)],
        [`tok/s whole batch (×${Bt})`, fmt(Bt / step, 0)],
        ["regime", memBound ? "memory-bound" : "compute-bound", memBound ? "mem" : "cmp"],
        [`min prefill (${fmt(prompt, 0)} tok)`, ms(prefill)]
      ]);
      math($(w, "#sol-math"), [
        `weights = ${B(Pb + "B")} params × ${B(bpp)} bytes = ${B(GB(wBytes))}`,
        `memory time = weights ÷ bandwidth = ${GB(wBytes)} ÷ ${B(g.bw + " TB/s")} = ${B(ms(tMem))}`,
        `compute time = 2 × params × batch ÷ FLOP/s = 2 × ${Pb}e9 × ${B(Bt)} ÷ ${B(g.tflops + " TFLOP/s")} = ${B(ms(tCmp))}`,
        `step = max(${ms(tMem)}, ${ms(tCmp)}) = ${B(ms(step))} → ${B(fmt(1 / step, 0))} tok/s per user`,
        `batch intensity = 2·P·B ÷ bytes = ${B(fmt(ai, 0))} FLOP/byte vs ridge = ${g.tflops} T ÷ ${g.bw} T = ${B(fmt(ridge, 0))} → ${memBound ? "below ridge: memory-bound" : "above ridge: compute-bound"}`,
        `prefill ≥ 2 × ${Pb}e9 × ${fmt(prompt, 0)} ÷ ${g.tflops}e12 = ${B(ms(prefill))}`
      ]);

      // tug-of-war bars
      const mx = Math.max(tMem, tCmp);
      $(w, "#sol-tug").innerHTML = `
        <div class="tug"><span class="tug-l">memory</span><div class="tug-track"><span class="tug-bar mem ${memBound ? "win" : ""}" style="width:${(tMem / mx) * 100}%"></span></div><span class="tug-v">${ms(tMem)}${memBound ? " ◀ BOTTLENECK" : ""}</span></div>
        <div class="tug"><span class="tug-l">compute</span><div class="tug-track"><span class="tug-bar cmp ${!memBound ? "win" : ""}" style="width:${(tCmp / mx) * 100}%"></span></div><span class="tug-v">${ms(tCmp)}${!memBound ? " ◀ BOTTLENECK" : ""}</span></div>`;

      // roofline svg (log-log)
      const W = 600, H = 250, padL = 52, padB = 34, padT = 14, padR = 14;
      const xmin = 0, xmax = 13; // log2 AI
      const ymax = Math.log10(F * 2), ymin = Math.log10(BW * 0.5);
      const X = (ai) => padL + ((Math.log2(ai) - xmin) / (xmax - xmin)) * (W - padL - padR);
      const Y = (f) => padT + (1 - (Math.log10(f) - ymin) / (ymax - ymin)) * (H - padT - padB);
      const roof = (a) => Math.min(F, a * BW);
      const pts = [1, ridge, Math.pow(2, xmax)].map((a) => `${X(a)},${Y(roof(a))}`).join(" ");
      const dotAI = Math.min(Math.max(ai, 1), Math.pow(2, xmax));
      const ticks = [1, 4, 16, 64, 256, 1024, 4096];
      $(w, "#sol-roof").innerHTML = `
        <svg viewBox="0 0 ${W} ${H}" class="roof-svg" role="img" aria-label="Roofline chart">
          <rect x="${padL}" y="${padT}" width="${X(ridge) - padL}" height="${H - padT - padB}" style="fill:var(--sky);opacity:.18"/>
          <rect x="${X(ridge)}" y="${padT}" width="${W - padR - X(ridge)}" height="${H - padT - padB}" style="fill:var(--peach);opacity:.18"/>
          <text x="${padL + 8}" y="${padT + 16}" class="rt">memory-bound</text>
          <text x="${W - padR - 8}" y="${padT + 16}" class="rt" text-anchor="end">compute-bound</text>
          <line x1="${padL}" y1="${H - padB}" x2="${W - padR}" y2="${H - padB}" style="stroke:var(--border);stroke-width:2"/>
          <line x1="${padL}" y1="${padT}" x2="${padL}" y2="${H - padB}" style="stroke:var(--border);stroke-width:2"/>
          ${ticks.map((t) => `<text x="${X(t)}" y="${H - padB + 16}" class="rt s" text-anchor="middle">${t}</text>`).join("")}
          <text x="${(W + padL) / 2}" y="${H - 4}" class="rt s" text-anchor="middle">arithmetic intensity (FLOP/byte, log scale)</text>
          <text x="12" y="${(H - padB) / 2}" class="rt s" transform="rotate(-90 12 ${(H - padB) / 2})" text-anchor="middle">FLOP/s</text>
          <polyline points="${pts}" style="fill:none;stroke:var(--pink-strong);stroke-width:4" shape-rendering="crispEdges"/>
          <line x1="${X(ridge)}" y1="${Y(F)}" x2="${X(ridge)}" y2="${H - padB}" style="stroke:var(--muted);stroke-width:1.5;stroke-dasharray:4 4"/>
          <text x="${X(ridge) + 6}" y="${H - padB - 8}" class="rt s">ridge ≈ ${fmt(ridge, 0)}</text>
          <rect x="${X(dotAI) - 7}" y="${Y(roof(dotAI)) - 7}" width="14" height="14" style="fill:var(--butter);stroke:var(--border);stroke-width:3" class="roof-dot"/>
          <text x="${Math.min(X(dotAI) + 12, W - 120)}" y="${Y(roof(dotAI)) + 22}" class="rt">you: batch ${Bt}</text>
        </svg>`;
    });
  }

  /* ============ 3. Latency timeline + cost ============ */
  function latCalc() {
    const w = document.getElementById("lat-calc");
    if (!w) return;
    const v = (s) => +$(w, s).value;
    onInput(w, () => {
      const qm = v("#lc-queue"), pm = v("#lc-prefill"), tp = v("#lc-tpot"), out = Math.max(1, v("#lc-out"));
      const price = v("#lc-price"), gpus = Math.max(1, v("#lc-gpus")), tps = v("#lc-tps"), rps = v("#lc-rps");
      const ttft = qm + pm, dec = tp * (out - 1), e2e = ttft + dec;
      const cap = gpus * tps, need = rps * out, util = need / cap;
      const conc = rps * (e2e / 1000);
      const cost = ((price * gpus) / (Math.min(need, cap) * 3600)) * 1e6;
      readout($(w, "#lc-out-r"), [
        ["TTFT", fmt(ttft, 0) + " ms"],
        ["end-to-end", fmt(e2e / 1000, 2) + " s"],
        ["per-user speed", fmt(1000 / tp, 0) + " tok/s"],
        ["in flight (Little)", fmt(conc, 0) + " requests"],
        ["$ per 1M output tok", "$" + fmt(cost, 2)],
        ["GPU load", fmt(util * 100, 0) + "%", util > 1 ? "bad" : util > 0.85 ? "mem" : "good"]
      ]);
      const pct = (x) => (x / e2e) * 100;
      $(w, "#lc-timeline").innerHTML = `
        <div class="tl">
          <span class="tl-seg q" style="width:${pct(qm)}%" title="queue"></span>
          <span class="tl-seg p" style="width:${pct(pm)}%" title="prefill"></span>
          <span class="tl-seg d" style="width:${pct(dec)}%" title="decode">${pct(dec) > 25 ? `decode ${out - 1} × ${tp} ms` : ""}</span>
        </div>
        <div class="membar-legend"><span><i class="tq"></i>queue ${qm} ms</span><span><i class="tp"></i>prefill ${pm} ms</span><span><i class="td"></i>decode ${fmt(dec, 0)} ms</span><span>▲ first token at ${fmt(ttft, 0)} ms</span></div>
        <div class="verdict ${util > 1 ? "oom" : "ok"}">${util > 1 ? `💥 Overloaded! You need ${fmt(need, 0)} tok/s but have ${fmt(cap, 0)}. Add ~${Math.ceil(need / tps) - gpus} GPU(s) or the queue grows forever.` : `✓ Capacity OK: GPUs ${fmt(util * 100, 0)}% busy. Below ~85% leaves headroom for spikes.`}</div>`;
      math($(w, "#lc-math"), [
        `TTFT = queue + prefill = ${B(qm)} + ${B(pm)} = ${B(fmt(ttft, 0) + " ms")}`,
        `E2E = TTFT + TPOT × (tokens − 1) = ${fmt(ttft, 0)} + ${B(tp)} × ${B(out - 1)} = ${B(fmt(e2e, 0) + " ms")}`,
        `demand = ${B(rps)} req/s × ${B(out)} tokens = ${B(fmt(need, 0))} tok/s · capacity = ${gpus} × ${fmt(tps, 0)} = ${B(fmt(cap, 0))} tok/s`,
        `concurrency (Little's law) = ${rps} req/s × ${fmt(e2e / 1000, 2)} s = ${B(fmt(conc, 0))} requests in flight`,
        `cost = ($${price} × ${gpus} GPUs per hour) ÷ (${fmt(Math.min(need, cap), 0)} tok/s × 3600) × 1M = ${B("$" + fmt(cost, 2))} per 1M tokens`
      ]);
    });
  }

  /* ============ 4. Batching simulator with playback ============ */
  function batchSim() {
    const w = document.getElementById("batch-sim");
    if (!w) return;
    let seed = 7, reqs = [], timer = null;
    const rnd = () => { seed = (seed * 16807) % 2147483647; return seed / 2147483647; };
    const color = (id) => `hsl(${(id * 47 + 330) % 360} 80% 72%)`;
    const slotsEl = $(w, "#sim-slots"), nEl = $(w, "#sim-n");
    const gen = () => { reqs = Array.from({ length: +nEl.value }, (_, i) => ({ id: i, len: 2 + Math.floor(rnd() * 13) })); };

    function simStatic(S) {
      const grid = Array.from({ length: S }, () => []), fin = {}; let t = 0;
      for (let i = 0; i < reqs.length; i += S) {
        const batch = reqs.slice(i, i + S), L = Math.max(...batch.map((r) => r.len));
        for (let s = 0; s < S; s++) {
          const r = batch[s];
          for (let k = 0; k < L; k++) grid[s].push(r && k < r.len ? r.id : null);
          if (r) fin[r.id] = t + L;
        }
        t += L;
      }
      return { grid, steps: t, fin };
    }
    function simCont(S) {
      const grid = Array.from({ length: S }, () => []), free = Array(S).fill(0), fin = {};
      for (const r of reqs) {
        const s = free.indexOf(Math.min(...free)), start = free[s];
        while (grid[s].length < start) grid[s].push(null);
        for (let k = 0; k < r.len; k++) grid[s].push(r.id);
        free[s] = start + r.len; fin[r.id] = free[s];
      }
      const steps = Math.max(...free);
      grid.forEach((g) => { while (g.length < steps) g.push(null); });
      return { grid, steps, fin };
    }
    function draw(el, res, label, S, maxSteps) {
      const used = reqs.reduce((a, r) => a + r.len, 0);
      const util = (used / (S * res.steps)) * 100;
      const avg = Object.values(res.fin).reduce((a, b) => a + b, 0) / reqs.length;
      el.innerHTML = `<div class="sim-head">${label}<span class="sim-live" data-steps="${res.steps}">${res.steps} steps · ${fmt(util, 0)}% busy · avg finish @ ${fmt(avg, 1)}</span></div>
        <div class="sim-grid" style="grid-template-columns:repeat(${maxSteps},14px)">
          ${res.grid.map((row) => row.map((c, t) => c === null ? `<div class="sim-cell idle" data-t="${t}"></div>` : `<div class="sim-cell" data-t="${t}" style="background:${color(c)}" title="request ${c}"></div>`).join("")
            + Array(maxSteps - row.length).fill('<div class="sim-cell none"></div>').join("")).join("")}
        </div>`;
      return { util, steps: res.steps, avg };
    }
    function run() {
      stop();
      const S = +slotsEl.value;
      $(w, "#sim-slots-v").textContent = S; $(w, "#sim-n-v").textContent = nEl.value;
      const a = simStatic(S), b = simCont(S), mx = Math.max(a.steps, b.steps);
      const ra = draw($(w, "#sim-static"), a, "STATIC BATCHING", S, mx);
      const rb = draw($(w, "#sim-cont"), b, "CONTINUOUS BATCHING", S, mx);
      $(w, "#sim-reqs").innerHTML = reqs.map((r) => `<span class="badge" style="background:${color(r.id)};color:#2a0f1d">R${r.id}: ${r.len}</span>`).join("");
      $(w, "#sim-verdict").innerHTML = `<div class="verdict ok">Continuous batching finished in ${B(rb.steps)} steps vs ${B(ra.steps)}: ${B(fmt(ra.steps / rb.steps, 2) + "×")} faster, with ${fmt(rb.util - ra.util, 0)} more points of slot use.</div>`;
    }
    function play() {
      stop();
      const cells = [...w.querySelectorAll(".sim-cell[data-t]")];
      const mx = Math.max(...cells.map((c) => +c.dataset.t)) + 1;
      let t = 0;
      cells.forEach((c) => c.classList.add("future"));
      $(w, "#sim-play").textContent = "■ Stop";
      timer = setInterval(() => {
        cells.forEach((c) => { if (+c.dataset.t === t) c.classList.remove("future"); });
        w.querySelectorAll(".sim-live").forEach((s) => {
          s.textContent = t + 1 >= +s.dataset.steps ? `done in ${s.dataset.steps} steps ✓` : `step ${t + 1} …`;
        });
        if (++t >= mx) { stop(); run(); }
      }, 110);
    }
    function stop() {
      if (timer) { clearInterval(timer); timer = null; }
      $(w, "#sim-play").textContent = "▶ Play";
    }
    $(w, "#sim-shuffle").addEventListener("click", () => { seed = Math.floor(Math.random() * 1e6) + 1; gen(); run(); });
    $(w, "#sim-play").addEventListener("click", () => (timer ? (stop(), run()) : play()));
    slotsEl.addEventListener("input", run);
    nEl.addEventListener("input", () => { gen(); run(); });
    gen(); run();
  }

  /* ============ 5. Quantization playground ============ */
  function quantLab() {
    const w = document.getElementById("quant-lab");
    if (!w) return;
    let seed = 11, base = [];
    const rnd = () => { seed = (seed * 16807) % 2147483647; return seed / 2147483647; };
    const gauss = () => { let u = 0, v = 0; while (!u) u = rnd(); while (!v) v = rnd(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); };
    const gen = () => { base = Array.from({ length: 16 }, () => +(gauss() * 0.5).toFixed(3)); };
    function quant(xs, bits) {
      const qmax = Math.pow(2, bits - 1) - 1;
      const s = Math.max(...xs.map(Math.abs)) / qmax || 1e-9;
      return { s, qmax, q: xs.map((x) => Math.max(-qmax, Math.min(qmax, Math.round(x / s)))) };
    }
    onInput(w, render);
    $(w, "#q-resample").addEventListener("click", () => { seed = Math.floor(Math.random() * 1e6) + 1; gen(); render(); });
    function render() {
      if (!base.length) gen();
      const bits = +$(w, "#q-bits").value, group = $(w, "#q-gran").value === "group";
      const xs = base.slice();
      if ($(w, "#q-outlier").checked) xs[5] = 4.0;
      const G = group ? 4 : 16;
      const deq = [], scales = [], qs = [];
      for (let i = 0; i < 16; i += G) {
        const r = quant(xs.slice(i, i + G), bits);
        scales.push(r.s);
        r.q.forEach((q) => { qs.push(q); deq.push(q * r.s); });
      }
      const err = xs.map((x, i) => Math.abs(x - deq[i]));
      const mae = err.reduce((a, b) => a + b, 0) / 16;
      const zeros = qs.filter((q, i) => q === 0 && Math.abs(xs[i]) > 1e-6).length;
      const qmax = Math.pow(2, bits - 1) - 1;

      // svg
      const W = 600, H = 230, pad = 24, bw = (W - 2 * pad) / 16;
      const M = Math.max(...xs.map(Math.abs)) * 1.1;
      const Y = (x) => H / 2 - (x / M) * (H / 2 - pad);
      let bars = "";
      xs.forEach((x, i) => {
        const x0 = pad + i * bw + 4, ww = bw - 8;
        const yq = Y(deq[i]), yo = Y(x);
        bars += `<rect x="${x0}" y="${Math.min(yq, H / 2)}" width="${ww}" height="${Math.max(2, Math.abs(yq - H / 2))}" style="fill:${qs[i] === 0 && Math.abs(x) > 1e-6 ? "var(--peach)" : "var(--pink)"}"/>`;
        bars += `<rect x="${x0}" y="${Math.min(yo, H / 2)}" width="${ww}" height="${Math.max(1, Math.abs(yo - H / 2))}" style="fill:none;stroke:var(--border);stroke-width:2;stroke-dasharray:3 2"/>`;
      });
      let grid = "";
      if (!group) {
        const s = scales[0], step = Math.abs(Y(s) - Y(0));
        if (step > 4) for (let k = -qmax; k <= qmax; k++) grid += `<line x1="${pad}" x2="${W - pad}" y1="${Y(k * s)}" y2="${Y(k * s)}" style="stroke:var(--surface-3);stroke-width:1"/>`;
      } else {
        for (let g = 1; g < 4; g++) grid += `<line x1="${pad + g * 4 * bw}" x2="${pad + g * 4 * bw}" y1="${pad / 2}" y2="${H - pad / 2}" style="stroke:var(--muted);stroke-width:1.5;stroke-dasharray:5 4"/>`;
      }
      $(w, "#q-svg").innerHTML = `<svg viewBox="0 0 ${W} ${H}" class="roof-svg" role="img" aria-label="Original vs quantized weights">
        ${grid}
        <line x1="${pad}" x2="${W - pad}" y1="${H / 2}" y2="${H / 2}" style="stroke:var(--border);stroke-width:2"/>
        ${bars}
        <text x="${pad}" y="16" class="rt s">dashed = original · filled = quantized · orange = squashed to 0 · ${!group ? "grey lines = allowed levels" : "dividers = groups"}</text>
      </svg>`;
      readout($(w, "#q-out"), [
        ["levels available", fmt(2 * qmax + 1, 0)],
        [group ? "scales (one per group)" : "scale s", group ? scales.map((s) => fmt(s, 3)).join(" · ") : fmt(scales[0], 4)],
        ["mean |error|", fmt(mae, 4), mae > 0.08 ? "bad" : mae > 0.03 ? "mem" : "good"],
        ["weights squashed to 0", zeros + " / 16", zeros > 3 ? "bad" : "good"],
        ["memory vs BF16", fmt((bits / 16) * 100, 0) + "%"]
      ]);
      const i0 = xs.findIndex((x, i) => i !== 5 && Math.abs(x) > 0.15) >= 0 ? xs.findIndex((x, i) => i !== 5 && Math.abs(x) > 0.15) : 0;
      const s0 = scales[Math.floor(i0 / G)];
      math($(w, "#q-math"), [
        `symmetric ${bits}-bit: q_max = 2^(${bits}−1) − 1 = ${B(qmax)}`,
        `scale s = max|x| ÷ q_max = ${fmt(Math.max(...xs.slice(Math.floor(i0 / G) * G, Math.floor(i0 / G) * G + G).map(Math.abs)), 3)} ÷ ${qmax} = ${B(fmt(s0, 4))}${group ? " (this weight's group)" : ""}`,
        `example: x = ${B(fmt(xs[i0], 3))} → q = round(${fmt(xs[i0], 3)} ÷ ${fmt(s0, 4)}) = ${B(qs[i0])} → x̂ = ${qs[i0]} × ${fmt(s0, 4)} = ${B(fmt(deq[i0], 3))} (error ${fmt(err[i0], 3)})`,
        zeros > 3 && !group ? `⚠ ${B("Ouch!")} the big value stretched the scale, so ${zeros} small weights rounded to 0. Switch to ${B("per-group")} and watch the error drop.`
          : group ? `per-group: each group of 4 gets its own scale, so an outlier only hurts its own group.` : `tip: tick "add an outlier" to see why activations with outliers are hard to quantize.`
      ]);
    }
  }

  /* ============ 6. Speculative decoding: formula + dice simulation ============ */
  function specCalc() {
    const w = document.getElementById("spec-calc");
    if (!w) return;
    const v = (s) => +$(w, s).value;
    const calc = () => {
      const a = Math.min(0.99, Math.max(0, v("#sp-alpha"))), k = Math.max(1, Math.round(v("#sp-k"))), c = Math.max(0, v("#sp-cost"));
      const exp = (1 - Math.pow(a, k + 1)) / (1 - a), cost = 1 + k * c;
      readout($(w, "#sp-out"), [
        ["expected tokens / target pass", fmt(exp, 2)],
        ["cost per round (× normal step)", fmt(cost, 2)],
        ["ideal speedup", fmt(exp / cost, 2) + "×", exp / cost >= 1 ? "good" : "bad"]
      ]);
      math($(w, "#sp-math"), [
        `E[tokens] = (1 − α^(k+1)) ÷ (1 − α) = (1 − ${a}^${k + 1}) ÷ (1 − ${a}) = (1 − ${fmt(Math.pow(a, k + 1), 3)}) ÷ ${fmt(1 - a, 2)} = ${B(fmt(exp, 2))}`,
        `cost = 1 target pass + k × draft cost = 1 + ${k} × ${c} = ${B(fmt(cost, 2))}`,
        `speedup = ${fmt(exp, 2)} ÷ ${fmt(cost, 2)} = ${B(fmt(exp / cost, 2) + "×")}${exp / cost < 1 ? " (slower than plain decoding!)" : ""}`
      ]);
    };
    onInput(w, calc);
    $(w, "#sp-run").addEventListener("click", () => {
      const a = Math.min(0.99, Math.max(0, v("#sp-alpha"))), k = Math.max(1, Math.round(v("#sp-k")));
      const rounds = 12; let total = 0, html = "";
      for (let r = 0; r < rounds; r++) {
        let row = "", got = 0, rejected = false;
        for (let i = 0; i < k; i++) {
          if (rejected) { row += `<span class="tok skip" title="discarded">·</span>`; continue; }
          if (Math.random() < a) { row += `<span class="tok ok" title="accepted">✓</span>`; got++; }
          else { row += `<span class="tok bad" title="rejected">✗</span><span class="tok fix" title="target's correction">+1</span>`; got++; rejected = true; }
        }
        if (!rejected) { row += `<span class="tok bonus" title="bonus token">+1</span>`; got++; }
        total += got;
        html += `<div class="spec-row" style="animation-delay:${r * 70}ms"><span class="spec-n">pass ${r + 1}</span>${row}<span class="spec-got">${got} tok</span></div>`;
      }
      const exp = (1 - Math.pow(a, k + 1)) / (1 - a);
      $(w, "#sp-sim").innerHTML = html + `
        <div class="race"><span class="race-l">plain decoding</span><div class="race-track"><span style="width:${(rounds / total) * 100}%" class="rb plain">${rounds}</span></div></div>
        <div class="race"><span class="race-l">speculative</span><div class="race-track"><span style="width:100%" class="rb spec">${total}</span></div></div>
        <div class="verdict ok">In ${rounds} target passes: ${B(total)} tokens vs ${B(rounds)} without speculation. Measured ${B(fmt(total / rounds, 2))} tokens/pass vs ${B(fmt(exp, 2))} expected. Roll again: luck varies!</div>`;
    });
  }

  kvCalc(); solCalc(); latCalc(); batchSim(); quantLab(); specCalc();
})();
