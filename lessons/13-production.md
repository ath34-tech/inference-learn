# Serving in Production

> Lesson 4-2 · WORLD 4 Expert / Systems · ~35 min · topics: Choosing an engine, Capacity planning, Benchmarking, Observability & reliability

> previous: [4-1 Kernels, Fusion & Compilation](../lessons/12-kernels.md)

> Part of **InferenceLearn**, a free course on LLM inference engineering. Web version: lessons/13-production.html. Whole course in one file: llms-full.txt

The final boss: turning a fast engine into a dependable service. This lesson brings everything together: picking an engine, sizing hardware, benchmarking honestly, watching the right metrics, and not falling over at 3 a.m.

#### By the end you'll be able to

- Pick an inference engine for a given workload
- Do capacity planning from first principles
- Run a sound benchmark sweep
- Set up observability and reliability guardrails

#### Key vocabulary

- **[OpenAI-compatible API](../vocabulary.md#v-openai-compatible-api)**: The de-facto HTTP interface (/v1/chat/completions) that most engines expose, so any OpenAI client works.
- **[SSE streaming](../vocabulary.md#v-sse-streaming)**: Server-Sent Events: the server pushes each token to the client as soon as it's generated.
- **[Capacity planning](../vocabulary.md#v-capacity-planning)**: Estimating how many GPUs you need from traffic, token lengths, SLOs and benchmarked per-GPU capacity.
- **[Load test (open-loop)](../vocabulary.md#v-load-test-open-loop)**: Sending requests at a target arrival rate whether or not earlier ones have finished, like real traffic. It exposes queueing collapse.
- **[Backpressure](../vocabulary.md#v-backpressure)**: Signalling clients to slow down (HTTP 429/503) instead of queueing work forever.
- **[Graceful drain](../vocabulary.md#v-graceful-drain)**: On deploy or shutdown: stop admitting new requests, let in-flight streams finish, then restart.
- **[Multi-LoRA serving](../vocabulary.md#v-multi-lora-serving)**: Serving many small LoRA fine-tune adapters on one shared base model at the same time.
- **[Structured output](../vocabulary.md#v-structured-output)**: Constraining generation to a JSON schema or grammar by masking invalid tokens at each step.

[Drill all terms as flashcards ▶](../vocabulary.md#flashcards)

## The engine landscape

| Engine | Sweet spot | Highlights |
|---|---|---|
| **vLLM** | General-purpose GPU serving | PagedAttention, huge model coverage, many hardware backends, OpenAI-compatible server, large community |
| **SGLang** | High-throughput serving, structured / agentic workloads | RadixAttention prefix caching, fast structured output, strong large-scale MoE support |
| **TensorRT-LLM** | Peak performance on NVIDIA | Highly optimized kernels, FP8/FP4, in-flight batching. Pairs with Triton Inference Server or Dynamo |
| **llama.cpp / Ollama** | Local, CPU, Apple Silicon, edge | GGUF quantization, runs nearly anywhere, minimal dependencies |
| **MLC LLM** | Mobile, browser (WebGPU), diverse devices | Compiler-based deployment across platforms |
| **LMDeploy, TGI, others** | Varies | Worth benchmarking for your model and hardware |

> **HOW TO CHOOSE:** Benchmark the **2–3 strongest candidates on your model, your hardware and your traffic shape**. Rankings change from release to release and from workload to workload. Also weigh operations: model support speed, quantization formats, observability, community and your team's familiarity.

## A minimal deployment

```
# launch an OpenAI-compatible server
vllm serve Qwen/Qwen2.5-7B-Instruct \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.90 \
  --enable-prefix-caching

# call it with any OpenAI client, streaming tokens via SSE
curl http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d '{
  "model": "Qwen/Qwen2.5-7B-Instruct",
  "messages": [{"role": "user", "content": "Explain the KV cache in one sentence."}],
  "max_tokens": 128, "stream": true
}'
```

## Capacity planning

Work it out on paper before buying GPUs. **Example:** a chat product on an 8B model needs **40 requests/s** at peak. Requests average 1,500 input and 300 output tokens. The SLO is TTFT p90 < 1 s and TPOT p90 < 40 ms.

1. **Token load**: 40 × 300 = 12,000 output tok/s and 40 × 1,500 = 60,000 prefill tok/s.
2. **Concurrency** (Little's law): E2E ≈ 0.5 s + 300 × 0.035 s ≈ 11 s, so ~440 sequences in flight.
3. **KV memory**: 440 × 1,800 tokens × 128 KiB ≈ 97 GiB. That won't fit on one 80 GB GPU next to the weights.
4. **Per-GPU capacity**: *benchmark* one replica and find the highest load that still meets the SLO. Say ~4,000 output tok/s per H100 with prefix caching on.
5. **Replicas**: 12,000 / 4,000 = 3, plus headroom for spikes and failures (N+1, ~30%), so **4–5 GPUs**.

Then validate with a real load test at 1.0× and 1.5× the expected peak.

## Benchmarking sweep

- Use a load generator such as `vllm bench serve`, **GuideLLM** or NVIDIA **GenAI-Perf**, with a realistic dataset.
- **Sweep the request rate** (or concurrency) from low to past saturation. Record throughput and TTFT/TPOT p50/p90/p99 at each point.
- Find the **knee**: the highest load where p90 still meets the SLO. That's your real capacity, not the peak throughput number.
- Change **one variable at a time**: engine, quantization, TP, max-num-seqs, speculative decoding, chunk size.

## Observability

Most engines export Prometheus metrics. Dashboard and alert on:

| Metric | Why it matters |
|---|---|
| TTFT, ITL / TPOT, E2E histograms | Your SLOs. Alert on p90/p99 |
| Running vs waiting requests | Queue growth means you're out of capacity |
| KV cache usage % | Nearing 100% means preemptions are coming |
| Preemptions / recomputes | Too much concurrency for your memory |
| Prefix-cache hit rate | A drop usually means a prompt template changed |
| Prompt & generation tokens/s | Real load against benchmarked capacity |
| Errors, timeouts, client cancels | Reliability and user experience |
| Spec-decode acceptance rate | If speculation is on, whether it's paying off |

## Reliability guardrails

- **Limit inputs**: cap `max_tokens` and prompt length, and set per-tenant rate limits. One 128k-token request can take over a replica.
- **Cancel on disconnect**: stop generating when the client goes away, or you burn GPU time on nobody.
- **Timeouts and backpressure**: return 429/503 early instead of queueing forever.
- **Health checks**: readiness only after weights load and graphs are captured. Liveness that detects a hung GPU.
- **Graceful drain** on deploys: stop admitting, finish in-flight streams, then restart.
- **Determinism caveats**: batching changes floating-point reduction order, so outputs at temperature 0 can still differ slightly between runs. Don't promise bit-exact reproducibility.
- **Structured output** (JSON schema / grammar-guided decoding) and **multi-LoRA** serving have their own costs. Benchmark with them enabled.

## Launch checklist

- Model + precision chosen and evaluated on *your* task suite
- Hardware sized with KV headroom; TP/DP layout decided
- Engine flags tuned: max-model-len, max-num-seqs, token budget, prefix caching
- Load-tested with a realistic dataset up to 1.5× peak; SLO knee recorded
- Dashboards and alerts for latency, queue, KV usage and errors
- Rate limits, max tokens, timeouts and cancellation in place
- Autoscaling signals and cold-start time measured
- Rollback plan for model and engine upgrades

## Check yourself

**Your benchmark peaks at 9,000 tok/s, but p90 TPOT exceeds the SLO above 6,000 tok/s. What's your capacity?**

**~6,000 tok/s**. Capacity is the highest load that still meets the SLO, not the raw peak.

**KV usage hovers at 99% and preemptions are climbing. Name two fixes.**

Lower `max-num-seqs` or the admitted concurrency. Add replicas. Use FP8 KV cache. Reduce `max-model-len` if the workload allows. Use a model with smaller KV (GQA/MLA), or quantize the weights to free memory.

**Why must a server stop generating when the client disconnects?**

Otherwise the abandoned sequence keeps its batch slot and KV memory and keeps using GPU time until it finishes, which takes capacity from real users.

## You cleared the course!

You now have the full mental model: what a token costs, where the bytes go, and the techniques that fix each bottleneck. Next, **build the World 4 capstone** from the [roadmap](../roadmap.md), read the source of a real engine (nano-vllm is a great small one), and keep up with new work via the [resources](../resources.md) page.

## Further reading

- [vLLM documentation](https://docs.vllm.ai/) · [SGLang documentation](https://docs.sglang.ai/)
- [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM)
- [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm), a small, readable vLLM-style engine
- [Machine Learning Engineering](https://github.com/stas00/ml-engineering) by Stas Bekman (the inference and debugging chapters)

## Interview corner

Questions on this lesson that come up in real interviews. Answer out loud first, then open the card.

**[mid] p99 latency doubled after a deploy. Walk me through the investigation.**

#### Strong answer

Is it TTFT or ITL? Check queue depth and running sequences (a capacity problem?), KV usage and preemptions, and prefix-cache hit rate (did a prompt template change?). Diff the deploy: engine version, flags (token budget, max seqs, speculation), model or precision. Check the traffic mix (longer prompts?). Look for hardware issues (a straggler GPU, throttling). Roll back if the SLO is breached, then bisect.

#### What they're testing

Calm, data-driven incident response.

#### Follow-ups

- Which three dashboards do you open first?

**[senior] How do you plan capacity for a new LLM feature?**

#### Strong answer

Gather peak req/s, input and output length distributions, and SLOs. Napkin math: token demand, Little's-law concurrency, KV memory, and whether the model fits with headroom. Benchmark one replica on realistic data and find the knee where p90 still meets the SLO. Replicas = demand ÷ knee capacity, plus headroom for spikes, failures and deploys (N+1, ~30%). Validate with an open-loop load test at 1.5× peak, then set autoscaling signals.

#### What they're testing

Turning product requirements into infrastructure.

#### Follow-ups

- What changes if 60% of the input tokens are a shared system prompt?

**[staff] Design an API that serves a 70B chat model at 500 req/s with p90 TTFT < 1 s and TPOT < 40 ms.**

#### A strong structure

- **Requirements:** lengths (say 2k in / 400 out), streaming, multi-tenancy, availability target.
- **Napkin math:** 500 × 400 = 200k output tok/s. Concurrency ≈ 500 × (~1 + 400×0.035 ≈ 15 s) ≈ 7,500 sequences. KV ≈ 7,500 × 2.4k tokens × 320 KiB ≈ 5.5 TiB, so hundreds of GPUs. FP8 weights and KV halve it.
- **Replica design:** FP8, TP=4 or 8 on H100/H200 nodes, continuous batching with chunked prefill, prefix caching, CUDA graphs. Benchmark the knee per replica.
- **Fleet:** KV/prefix-aware router, autoscaling on queue depth and SLOs, warm pools, multi-region.
- **At this scale, consider:** P/D disaggregation, speculative decoding for the latency tier.
- **Operations:** rate limits, max-token caps, cancellation, observability, canaries, rollback.

#### What they're testing

Driving a design from numbers, and making trade-offs explicit.

[All interview questions ▶](../interview.md) · [Quiz Arcade ▶](../practice.md#arcade) · [Workbook ▶](../workbook.md)
