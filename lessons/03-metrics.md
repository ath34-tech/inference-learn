# Metrics: Latency, Throughput & Cost

> Lesson 1-3 · WORLD 1 Foundations · ~20 min · topics: TTFT / TPOT / ITL, Throughput vs latency, Goodput & SLOs, Cost per 1M tokens

> previous: [1-2 GPU Fundamentals & the Roofline](../lessons/02-gpu-fundamentals.md) · next: [2-1 The KV Cache](../lessons/04-kv-cache.md)

> Part of **InferenceLearn**, a free course on LLM inference engineering. Web version: lessons/03-metrics.html. Whole course in one file: llms-full.txt

"Make it faster" means nothing until you say *which* number should move. Inference has several competing metrics, and nearly every optimization trades one for another.

#### By the end you'll be able to

- Define TTFT, TPOT/ITL, end-to-end latency and throughput precisely
- Reason about the latency–throughput trade-off and **goodput**
- Convert GPU cost into **$ per million tokens**
- Avoid the classic benchmarking mistakes

#### Key vocabulary

- **[TTFT](../vocabulary.md#v-ttft)**: Time To First Token: from sending the request to receiving the first token. Includes queueing and prefill.
- **[ITL](../vocabulary.md#v-itl)**: Inter-Token Latency: the gap between consecutive streamed tokens.
- **[TPOT](../vocabulary.md#v-tpot)**: Time Per Output Token: a request's average decode time per token, (E2E − TTFT) ÷ (tokens − 1).
- **[Throughput](../vocabulary.md#v-throughput)**: Work done per second across all users, usually output tokens/s. Higher throughput means lower cost per token.
- **[Goodput](../vocabulary.md#v-goodput)**: Throughput counting only requests that met their latency SLOs.
- **[SLO](../vocabulary.md#v-slo)**: Service Level Objective: a latency or availability target, e.g. "p90 TTFT under 500 ms".
- **[Little's law](../vocabulary.md#v-little-s-law)**: Concurrency = throughput × latency. 20 req/s × 10 s means ~200 requests in flight.
- **[MFU](../vocabulary.md#v-mfu)**: Model FLOPs Utilization: achieved FLOP/s ÷ peak. The efficiency metric for compute-bound work.
- **[MBU](../vocabulary.md#v-mbu)**: Model Bandwidth Utilization: achieved bytes/s ÷ peak bandwidth. The efficiency metric for decode. 60–80% is good.

[Drill all terms as flashcards ▶](../vocabulary.md#flashcards)

## Latency metrics

```
timeline: send □ □ □ □ □ t₁ □ t₂ □ t₃ □ t₄ done
```

*Diagram: striped = queueing · pink = prefill · yellow = token arrives · blue = gap between tokens*

- **TTFT (Time To First Token)**: from sending the request to receiving the first token. It includes queueing, prefill and network. This is what makes a chatbot feel "snappy".
- **ITL (Inter-Token Latency)**: the gap between consecutive streamed tokens. Humans read ~5–10 tokens/s, so streaming faster than ~20–50 ms/token mostly helps agents and long outputs.
- **TPOT (Time Per Output Token)**: average decode time per token for one request: `(E2E − TTFT) / (output_tokens − 1)`. It's basically the mean ITL.
- **E2E latency**: total time for the full response.

> E2E latency ≈ **TTFT** + **TPOT** × (output_tokens − 1)

> **WATCH OUT:** Always report **percentiles** (p50, p90, p99), not just averages. One long prefill landing in a batch can spike every other user's ITL. That shows up in p99, not the mean.

## Throughput metrics

- **Output tokens/s** (whole system): the usual headline number. Count *generated* tokens only.
- **Total tokens/s**: input plus output. It looks much bigger and is often used to inflate results. Know which one you're reading.
- **Requests/s**: only meaningful with a fixed input/output length distribution.
- **Per-user tokens/s** = 1 / TPOT: what a single user experiences.

## The latency–throughput trade-off

Raising concurrency (more sequences per batch) raises system throughput, because weight reads are amortized (lesson 1-2). But each step gets slightly slower and queues grow, so per-user latency gets worse. Plot both and you get a **Pareto frontier**:

```
 system tok/s
 (throughput)
    ▲
    │ ■ ← batch 256: huge throughput, each user slow
    │     ■
    │         ■  ← batch 32
    │             ■
    │                 ■
    │                     ■ ← batch 1: fastest per user, GPU mostly idle
    └──────────────────────────────────────▶ per-user tok/s (interactivity)
```

*Diagram: every engine and config gives a different frontier. Optimization pushes the whole curve outward*

The useful way to compare two systems: *"At the same per-user speed of 50 tok/s, which delivers more total throughput?"* or *"At the same throughput, which has lower p99 latency?"*

## Little's law & goodput

**Little's law** links the three quantities in any stable queueing system:

> concurrency = **throughput** × **latency**

If you need 20 requests/s and each takes 10 s end to end, about 200 requests are in flight at any moment. Your engine must hold the KV cache for all 200.

**Goodput** counts only requests that meet their SLO, for example TTFT < 500 ms *and* TPOT < 50 ms. A system that pushes 10k tok/s but misses the SLO half the time has poor goodput. This is the metric behind disaggregated serving (lesson 3-4).

## Cost per token

> $ per 1M output tokens = **GPU $/hour** ÷ (**output tok/s** × 3600) × 10⁶

**Worked example:** an H100 rented at $3/hour serving 2,500 output tok/s in total gives 2,500 × 3600 = 9M tokens/hour, so $3 / 9 ≈ **$0.33 per 1M output tokens**. Double the throughput at the same latency target and you halve the cost. That's why throughput optimizations are business-critical.

> **Interactive lab: Latency & cost builder.** Build one request's timeline, then your whole service. Raise the request rate until the GPUs overload, then add GPUs. Watch how TPOT dominates long answers. *(Use it on the web page. The formulas it uses are explained in the text.)*

> **TIP:** Real cost includes idle time. If traffic is spiky and your GPUs run at 30% average utilization, your real cost per token is ~3× the benchmark number. Autoscaling and batching across tenants matter as much as kernel speed.

## Utilization metrics: MFU & MBU

- **MFU (Model FLOPs Utilization)** = achieved FLOP/s ÷ peak FLOP/s. Use it for compute-bound phases (prefill, large batches).
- **MBU (Model Bandwidth Utilization)** = achieved bytes/s ÷ peak bandwidth, where bytes = weights + KV read per step. Use it for memory-bound decode. 60–80% MBU is good.

These tell you how close you are to the speed of light, and so whether more optimization is worth it.

## Benchmarking without fooling yourself

- **Warm up** first. The first requests pay for compilation, CUDA graph capture and memory allocation.
- Use **realistic length distributions** (e.g. from production logs or ShareGPT-style datasets), not one fixed length.
- Prefer **open-loop** load (requests arrive at a target rate, e.g. Poisson) over closed-loop (N users who wait for each response). Closed-loop hides queueing collapse.
- Measure at the **client**, with streaming, so network and detokenization are included.
- Fix `max_tokens` and use `ignore_eos` when comparing engines, so all of them generate the same number of tokens.
- Watch prefix caching: repeated identical prompts can make TTFT look amazing.
- Report hardware, engine version, precision, parallelism and all flags.

## Check yourself

**A request has TTFT 400 ms, and 251 output tokens arrive over a total of 5.4 s. What's the TPOT?**

(5.4 − 0.4) / (251 − 1) = 5.0 / 250 = **20 ms/token**, or 50 tok/s per user.

**Your service handles 50 req/s with 4 s average E2E latency. How many concurrent sequences does the engine hold?**

Little's law: 50 × 4 = **200** concurrent requests, each with its own KV cache.

**Engine A: 8,000 tok/s total at p99 TPOT of 120 ms. Engine B: 6,000 tok/s at 40 ms. Which is better?**

It depends on your SLO. If you need TPOT < 50 ms, B wins: A's goodput under that SLO could be near zero. Compare them at the *same* latency target.

## Further reading

- [LLM Inference Performance Engineering: Best Practices](https://www.databricks.com/blog/llm-inference-performance-engineering-best-practices) (Databricks)
- [DistServe](https://arxiv.org/abs/2401.09670), which introduced goodput-centric serving
- [GuideLLM](https://github.com/vllm-project/guidellm), a load-testing tool for LLM servers

## Interview corner

Questions on this lesson that come up in real interviews. Answer out loud first, then open the card.

**[junior] Define TTFT, TPOT and throughput. Which matters most for a chatbot vs a batch summarization job?**

#### Strong answer

TTFT = time to first token (queue + prefill). TPOT = average time per output token after the first. Throughput = total tokens/s across all users. A chatbot cares about TTFT (does it feel responsive?) and a TPOT faster than reading speed. Offline batch jobs care almost only about throughput and cost, so you can use huge batches and accept high latency.

#### What they're testing

That optimization goals depend on the product.

#### Follow-ups

- Why report p99 rather than the mean?
- What is goodput?

**[mid] How would you benchmark two inference engines fairly?**

#### Strong answer

Same model, precision, hardware and parallelism. A realistic input/output length distribution. Fixed output lengths (ignore_eos) so both engines do the same work. Warm up first. Use open-loop arrivals and sweep the request rate. Measure TTFT/ITL percentiles at the client with streaming. Compare throughput **at equal latency SLOs**, not peak throughput. Control prefix caching. Report versions and flags.

#### What they're testing

Experimental rigor. Lots of published benchmarks are misleading.

#### Follow-ups

- Why is closed-loop testing misleading near saturation?

**[mid] Your API serves 40 req/s, each ~10 s end to end. How much KV memory do you need for an 8B model (128 KiB/token) at ~2k tokens per request?**

#### Strong answer

Little's law: 40 × 10 = 400 concurrent requests. Each holds ~2k × 128 KiB = 256 MiB of KV, so ~100 GiB in total. That doesn't fit next to the weights on one 80 GB GPU, so you need at least two replicas, more with headroom. Prefix caching or FP8 KV could cut it.

#### What they're testing

Linking traffic to memory with napkin math.

#### Follow-ups

- What if average output length doubles?

[All interview questions ▶](../interview.md) · [Quiz Arcade ▶](../practice.md#arcade) · [Workbook ▶](../workbook.md)
