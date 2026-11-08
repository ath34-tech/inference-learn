# Interview Prep

> Part of **InferenceLearn**, a free course on LLM inference engineering. Web version: interview.html. Whole course in one file: llms-full.txt

// boss rush

39 questions that come up in inference, ML-systems and LLM-infra interviews. Answer out loud first, then open the card and compare.

### How to use this page

- **Say your answer before opening the card.** Recognizing an answer is much easier than producing one.
- Strong answers follow a pattern: **name the bottleneck → do the napkin math → propose a fix → state the trade-off.**
- Levels: [junior] [mid] [senior] [staff]  are rough guides to depth, not hard rules.
- Practise with an AI: paste this page's link into ChatGPT, Claude or Gemini and ask it to interview you, one question at a time, and grade your answers.

## Fundamentals

**[junior] Walk me through what happens when an LLM serves one request.**

#### Strong answer

Tokenize the prompt. Run **prefill**: one forward pass over all prompt tokens in parallel, which fills the KV cache and produces logits for the first token. Sample a token, stream it back, then loop in **decode**: each step feeds one new token, appends its K/V to the cache, and samples the next, until EOS or max_tokens. Detokenize and stream as you go.

#### What they're testing

Whether you know the two phases and that decode is a sequential loop. Mentioning the KV cache unprompted is a good sign.

#### Follow-ups

- Which phase dominates latency for a 2k-in / 500-out chat request? (Decode: 500 sequential steps.)
- Where does the scheduler fit in?

**[mid] Why do prefill and decode stress the GPU so differently?**

#### Strong answer

Prefill multiplies the weights by a matrix of N prompt tokens, so each weight byte loaded is reused N times. Arithmetic intensity is high and it's **compute-bound**. Decode multiplies the weights by one vector per sequence, so it's about 1 FLOP per byte at batch 1 and **memory-bandwidth-bound**. Decode also reads the KV cache, which can't be shared across sequences.

#### What they're testing

The roofline intuition, which most of inference optimization rests on.

#### Follow-ups

- How does batching change decode's position on the roofline?
- What is attention's intensity in decode, and why doesn't batching help it?

**[junior] Greedy vs temperature vs top-p sampling: when would you use each?**

#### Strong answer

Greedy (T=0) for deterministic tasks such as extraction, classification and code where you want the most likely answer. Temperature 0.7–1.0 with top-p ≈ 0.9–0.95 for creative or chat output. Top-p adapts the candidate set to the model's confidence, while top-k is a fixed cutoff. Mention that sampling has a cost in serving (sorting large vocabularies per sequence), so engines use fused GPU samplers.

#### What they're testing

Basic decoding knowledge plus some awareness of the systems side.

#### Follow-ups

- Why can T=0 still be non-deterministic in a batched server?

**[mid] Estimate the max single-user tokens/s for a 70B model in FP8 on one H200.**

#### Strong answer

The weights are ~70 GB in FP8. H200 bandwidth is ~4.8 TB/s, so each step takes ≥ 70 / 4800 s ≈ 14.6 ms, or about **68 tok/s** at the speed of light. Expect ~70–85% of that in practice, and less at long context because KV reads add bytes.

#### What they're testing

Whether you can do speed-of-light math quickly and know that bandwidth is the limit.

#### Follow-ups

- How does this change with TP=2 on two GPUs? (Roughly halves bytes per GPU per step, minus communication overhead.)
- And with batch 64?

**[senior] A new GPU doubles FLOPs but only raises bandwidth 1.3×. What happens to LLM serving performance?**

#### Strong answer

Low-batch decode speeds up by only ~1.3× because it's bandwidth-bound. Prefill and large-batch decode can gain up to ~2× if the kernels use the new compute. The ridge point rises (FLOPs ÷ BW goes up ~1.5×), so you need *bigger* batches to reach compute-bound. That pushes you toward larger batches, lower precision (to cut bytes), speculative decoding (to turn spare FLOPs into tokens), and disaggregation (to put prefill on the compute-rich parts).

#### What they're testing

Applying roofline reasoning to hardware trends.

#### Follow-ups

- Which optimizations become *more* valuable on that GPU?

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

## Memory & batching

**[junior] What is the KV cache and why does it exist?**

#### Strong answer

With causal attention, each past token's key and value vectors never change once computed. Caching them means each decode step computes K/V only for the new token and attends over the cache. That turns quadratic recomputation into linear work. The cost is memory that grows with layers × KV heads × head dim × sequence length × batch.

#### What they're testing

The most fundamental concept in serving.

#### Follow-ups

- Why aren't queries cached?
- Compute the per-token size for Llama-3-8B.

**[mid] Why is serving 128k-token contexts so much more expensive than 4k?**

#### Strong answer

(1) KV memory grows linearly, so 32× more per sequence and far fewer sequences per GPU, which means lower throughput and higher cost. (2) Every decode step reads the whole cache. At long context, KV bytes can exceed weight bytes, and batching doesn't amortize them. (3) Prefill attention is O(N²) compute, so TTFT grows a lot. (4) Scheduling gets harder: big preemptions and big prefills that stall other users.

#### What they're testing

A view of all the dimensions of long-context cost.

#### Follow-ups

- What mitigations exist? (GQA/MLA, FP8 KV, prefix caching, context parallelism, sliding windows.)

**[senior] You're out of KV memory at your target concurrency. List options in order of what you'd try.**

#### Strong answer

(1) Check config: `gpu-memory-utilization`, an over-sized `max-model-len`, CUDA-graph memory. (2) Enable prefix caching if prompts share prefixes. (3) FP8 KV cache, about 2× capacity with small quality risk, so evaluate it. (4) Quantize the weights to free memory for KV. (5) Add TP to pool memory, or add replicas. (6) Offload cold KV to CPU. (7) Long term, pick a model with a smaller KV footprint (GQA/MLA). Each step trades cost, quality or latency.

#### What they're testing

Practical prioritization: cheap, safe changes first.

#### Follow-ups

- How would you validate that FP8 KV didn't hurt quality?

**[junior] Explain continuous batching and why it beats static batching.**

#### Strong answer

Static batching runs a fixed batch until its longest sequence finishes, so short sequences leave idle slots and new requests wait. Continuous (iteration-level) batching re-decides the batch every decode step: finished sequences leave immediately and waiting ones join. The GPU stays full, often giving several times more throughput at similar latency. It needs ragged-batch kernels and paged KV memory.

#### What they're testing

The core scheduling idea behind every modern engine.

#### Follow-ups

- What new problem does it create? (Prefill/decode interference.)

**[mid] What is chunked prefill, and how do you tune it?**

#### Strong answer

Split long prompts into chunks and fill each step's token budget with decode tokens first, then a prefill chunk. This stops long prefills from freezing everyone's token stream, and it mixes compute-heavy and memory-heavy work in the same step. The knob is the per-step token budget (e.g. max-num-batched-tokens): smaller gives smoother ITL but slower TTFT for long prompts; larger gives better throughput and TTFT but spikier ITL.

#### What they're testing

Understanding the latency trade-off, not just the definition.

#### Follow-ups

- How does this compare with disaggregation?

**[senior] Preemption: recompute vs swap. When is each better?**

#### Strong answer

Recompute throws away the victim's KV and re-runs prefill later. Its cost is compute, and prefill is fast and parallel, so it's cheap for short contexts. Swap copies the KV to CPU and back. Its cost is PCIe bandwidth (~tens of GB/s) and CPU memory. For long contexts where recompute is expensive, or on systems with fast CPU–GPU links (e.g. NVLink-C2C), swapping can win. In both cases, frequent preemption means you're over-admitting: fix the admission limits.

#### What they're testing

Reasoning about costs on different resources.

#### Follow-ups

- Which sequence would you choose as the victim, and why?

## Attention & quantization

**[mid] How does FlashAttention work, and is it an approximation?**

#### Strong answer

It's exact. Naive attention writes the N×N score matrix to HBM and reads it back for softmax and the V product, so it's memory-bound. FlashAttention tiles Q/K/V into SRAM, computes partial scores per tile, and combines them with **online softmax** (running max and sum, rescaling). Only the output goes to HBM. Memory traffic drops from O(N²) to O(N), giving big speedups and allowing long sequences.

#### What they're testing

IO-awareness, the most important performance idea of recent years.

#### Follow-ups

- Derive the online-softmax update.
- What did FA2 and FA3 change?

**[mid] What problem does PagedAttention solve, and what else does it enable?**

#### Strong answer

Contiguous per-request KV reservations waste memory through over-reservation and fragmentation; the vLLM paper measured 60–80% waste in earlier systems. PagedAttention stores KV in fixed-size blocks mapped by a per-sequence block table, like virtual memory, so waste is under ~4%. It also enables **sharing** via reference counts and copy-on-write: parallel sampling, beam search and prefix caching.

#### What they're testing

Memory management as a throughput lever.

#### Follow-ups

- How does block size affect performance and waste?

**[senior] Compare MHA, GQA and MLA from a serving perspective.**

#### Strong answer

MHA has one KV head per query head: the largest cache and the lowest decode-attention intensity. GQA shares KV heads across groups of query heads: g× smaller cache, g× more reuse per byte, and near-MHA quality, which is why it's the default. MLA caches a low-rank latent per token and absorbs the projections at inference: a much smaller cache, but different kernels, and some wrinkles with TP sharding and positional embeddings. All three change how many sequences fit, and so throughput.

#### What they're testing

Connecting architecture to memory, bandwidth and parallelism.

#### Follow-ups

- Why does a small number of KV heads limit the tensor-parallel degree?

**[mid] W4A16 vs W8A8 (FP8): when would you choose each?**

#### Strong answer

W4A16 cuts weight bytes 4× and is ideal for memory-bound, low-batch serving or fitting big models on small GPUs. The math still runs in BF16 with dequantization overhead, so it can lose at high batch. W8A8 (FP8 on Hopper) halves bytes *and* uses faster FP8 tensor cores, so it helps prefill and high-batch throughput too, and it's usually near-lossless. Rule of thumb: FP8 for throughput serving on modern GPUs; INT4 weight-only for latency at low concurrency or tight memory.

#### What they're testing

Matching quantization to the bottleneck.

#### Follow-ups

- Where would FP8 KV cache fit in?

**[senior] Why are activations harder to quantize than weights, and how do people deal with it?**

#### Strong answer

Activations vary per input and contain systematic outlier channels 10–100× larger than the rest. Per-tensor INT8 scaling then crushes the normal values. Fixes: mixed precision for outliers (LLM.int8), SmoothQuant (migrate scale into the weights), per-token dynamic scales, rotations (QuaRot/SpinQuant) to spread outliers, and FP8, whose exponent handles range better than INT8.

#### What they're testing

Depth beyond "just use INT8".

#### Follow-ups

- Static vs dynamic activation scales: what's the trade-off?

**[mid] How would you decide a quantized model is safe to ship?**

#### Strong answer

Perplexity as a smoke test only. Then task evals matching the product (knowledge, math, code, long-context retrieval, multilingual if relevant) plus a domain eval set. Side-by-side human or LLM-judge comparisons on real traffic samples. Check the tails: math and code break first. Measure the actual speed and cost gain at the production batch size. Then canary-deploy with quality metrics.

#### What they're testing

Evaluation discipline.

#### Follow-ups

- A 0.5% MMLU drop: ship or not? (Depends on the product and what it saves.)

## Speculative decoding & parallelism

**[mid] Explain speculative decoding and why it doesn't change the output distribution.**

#### Strong answer

A cheap drafter proposes k tokens and the target verifies all of them in one forward pass, which costs about one decode step when memory-bound. Accept draft token x with probability min(1, p(x)/q(x)). On the first rejection, resample from normalize(max(0, p − q)) and stop. This rejection-sampling scheme provably yields samples from p. Expected tokens per pass = (1 − α^(k+1)) / (1 − α).

#### What they're testing

Both the systems intuition and the probability argument.

#### Follow-ups

- What's the greedy special case?
- What happens to the KV cache for rejected tokens?

**[senior] Speculative decoding made p50 latency better but throughput worse under load. Why, and what would you do?**

#### Strong answer

At high batch the engine is compute-bound. Verifying k extra tokens per sequence adds real FLOPs, and rejected drafts are pure waste, so total throughput drops. Options: adapt k (or turn speculation off) based on batch size or load; use cheaper drafters (n-gram, EAGLE heads); only speculate for latency-sensitive tiers; and track acceptance rate per workload.

#### What they're testing

Knowing when an optimization stops paying off.

#### Follow-ups

- Which metrics would you put on a dashboard for this?

**[junior] Name three ways to get draft tokens.**

#### Strong answer

A small draft model from the same family (same tokenizer); n-gram or prompt-lookup copying from the context; extra prediction heads trained on the target (Medusa, EAGLE) or built-in MTP modules (e.g. DeepSeek-V3).

#### What they're testing

Breadth.

#### Follow-ups

- Which would you pick for a code-editing product, and why? (Prompt lookup: edits copy large spans.)

**[mid] Tensor vs pipeline parallelism for inference: trade-offs?**

#### Strong answer

TP splits each layer across GPUs. It cuts per-token latency, since each GPU reads 1/N of the weights, but needs two all-reduces per layer, so it wants NVLink within a node. PP splits layers into stages, with cheap point-to-point communication that works across nodes. It doesn't reduce single-token latency and needs many micro-batches to avoid bubbles. The usual setup is TP inside a node and PP across nodes, then data-parallel replicas for throughput.

#### What they're testing

Understanding which parallelism reduces latency and which only adds capacity.

#### Follow-ups

- Why does TP stop scaling well beyond 8 GPUs?

**[senior] What makes serving large MoE models hard?**

#### Strong answer

Memory scales with total parameters but compute with active ones, so you need lots of memory and little compute per token. At high batch, nearly all experts are touched every step, so weight reads aren't small anymore. Expert parallelism needs all-to-all dispatch and combine, which is latency-sensitive. Load imbalance creates hot experts and stragglers. Fixes include wide EP with attention data parallelism, redundant hot experts, specialized all-to-all kernels, grouped GEMMs, and overlapping communication with compute.

#### What they're testing

Current frontier knowledge.

#### Follow-ups

- Why might you pair attention data parallelism with expert parallelism?

**[staff] Estimate the tensor-parallel all-reduce traffic per decode step for a 70B dense model, batch 64, TP=8.**

#### Strong answer

With hidden size ~8192 and 80 layers, there are 2 all-reduces per layer, each on a [64 × 8192] BF16 tensor = 1 MiB. That's 160 all-reduces of ~1 MiB per step, and a ring all-reduce moves ~2·(N−1)/N × size per GPU ≈ 1.75 MiB each, so ~280 MiB per GPU per step. At ~450 GB/s effective NVLink that's ~0.6 ms of bandwidth time. But 160 small collectives each carry latency (tens of µs), so **latency, not bandwidth,** often dominates. Hence fused all-reduce kernels, overlap, and custom one-shot all-reduce for small messages.

#### What they're testing

Quantitative distributed-systems reasoning.

#### Follow-ups

- How does this change for prefill with 8k tokens?

## Serving systems

**[mid] How would you implement prefix caching in a paged-KV engine?**

#### Strong answer

Give each full block a key = hash(parent key, the block's token IDs, extras like LoRA ID). Keep a hash → physical-block map. On a new request, hash the prompt block by block, reuse the longest run of hits (incrementing ref counts), and prefill from there. When a request finishes, its blocks go to an evictable LRU list rather than being freed. Evict only blocks with ref count 0, leaves first. The alternative is a radix tree (SGLang) for token-level matching.

#### What they're testing

Data-structure design under memory pressure.

#### Follow-ups

- How do you handle a partially filled last block?
- Hash collisions?

**[senior] You run 20 replicas behind a round-robin load balancer and prefix-cache hit rates are poor. What do you change?**

#### Strong answer

Round-robin scatters each conversation across replicas, so caches never warm up. Move to cache-aware routing: session affinity (hash the conversation ID) as a cheap first step, then a KV-aware router that scores replicas on predicted prefix overlap minus current load, to avoid hotspots. Also fix the prompt template so static content comes first, and consider a shared or offloaded KV tier.

#### What they're testing

Seeing the system as a whole beyond one engine.

#### Follow-ups

- How do you measure success? (Hit rate, TTFT and GPU prefill time per request.)

**[senior] When is prefill/decode disaggregation worth it, and what does it cost?**

#### Strong answer

It's worth it at scale with strict, separate TTFT and ITL SLOs, long prompts, or very different prefill and decode loads. Separate pools remove interference and can be scaled and configured independently. Costs: KV transfer bandwidth (e.g. ~640 MB for 2k tokens on a 70B GQA model, needing RDMA or NVLink), extra operational complexity, new failure modes, and possible stranded capacity if the pool ratio is wrong. For small deployments, chunked prefill on one pool is usually enough.

#### What they're testing

Judging when an architecture pays off.

#### Follow-ups

- How would you choose the prefill:decode GPU ratio?

**[mid] What signals would you autoscale an LLM service on?**

#### Strong answer

Queue depth or waiting requests, KV-cache utilization, TTFT/ITL percentiles against the SLO, and tokens/s against benchmarked capacity. Not CPU, and not raw nvidia-smi utilization. Account for slow cold starts (weights, compilation, graph capture) with warm pools, local weight caches and predictive scaling.

#### What they're testing

Production operations knowledge.

#### Follow-ups

- How do you scale down without dropping streams? (Graceful drain.)

**[staff] Design a scheduler that serves interactive and batch traffic on the same GPUs.**

#### Strong answer

Use priority classes: interactive requests are admitted first and their decodes always get budget, with a guaranteed TTFT/ITL target. Batch requests fill leftover token budget and are preemptible (recompute preferred, since they're latency-tolerant). Use admission control per class, fairness across tenants within a class (token-based fair queueing), and maybe a different speculation policy per class. Monitor goodput for interactive traffic and throughput and cost for batch. Scale on interactive SLO signals and let batch traffic soak up idle capacity.

#### What they're testing

Policy design with competing objectives.

#### Follow-ups

- How do you prevent starvation of batch jobs?

## Kernels & production

**[mid] What are CUDA graphs and why do inference engines use them?**

#### Strong answer

They record a sequence of kernel launches with fixed shapes and addresses, then replay it with one launch. Small-batch decode steps are short (milliseconds) but launch hundreds of kernels, so CPU launch overhead can rival GPU time. Graphs remove it. Engines capture one graph per batch-size bucket and pad up, mostly for decode. The costs are capture time, memory, and rigid shapes.

#### What they're testing

Understanding CPU vs GPU overheads.

#### Follow-ups

- Why aren't graphs typically used for prefill?

**[mid] Give an example of kernel fusion in an LLM and explain the speedup.**

#### Strong answer

SwiGLU: silu(gate) × up. Unfused, it's separate kernels that each read and write the full [tokens × 14336] tensors to HBM. Fused, it's one read of each input and one write of the output. Elementwise work is memory-bound, so time drops roughly in proportion to the bytes saved, plus launches. Other examples: RMSNorm+residual(+FP8 quant), rotary fused into attention, dequant fused into GEMM.

#### What they're testing

Bytes-moved reasoning at the kernel level.

#### Follow-ups

- Sketch that kernel in Triton.

**[senior] A decode step takes 2× longer than your speed-of-light estimate. How do you find out why?**

#### Strong answer

First check the estimate itself: include KV bytes at the actual context lengths and the communication. Then take an Nsight Systems trace and look for gaps between kernels (CPU-bound: scheduling, sampling sync, Python, missing CUDA graphs), many tiny kernels (missing fusion), and slow collectives. Then use Nsight Compute on the top kernels for achieved bandwidth vs peak, occupancy and wave quantization. Compute MBU per kernel class and fix the biggest gap first.

#### What they're testing

A systematic performance-debugging method.

#### Follow-ups

- Which GEMM shapes in decode tend to underutilize the GPU?

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

## System design (open-ended)

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

**[staff] Design serving for a RAG product: 30k-token prompts (mostly retrieved docs), short answers.**

#### A strong structure

- **Prefill-dominated:** TTFT and prefill compute are the cost drivers, not decode.
- **Reuse:** cache KV for popular documents. Order prompts as [system][docs sorted by a stable ID][question]. Use a KV offload tier (CPU/SSD) for document caches. Route on document overlap.
- **Compute:** FP8 prefill, high-TP or disaggregated prefill pools, chunked prefill so the few decoders aren't starved.
- **Memory:** long KV per request but short lifetimes. Size the pool for burst concurrency.
- **Product lever:** retrieve fewer and better chunks. The cheapest token is the one you never send.

#### What they're testing

Recognizing the workload shape and optimizing for it.

**[staff] Design a platform where 1,000 customers each have their own fine-tuned variant of one 8B base model.**

#### A strong structure

- Don't deploy 1,000 full models. Use **LoRA adapters** on a shared base with multi-LoRA batching (kernels that apply different adapters per sequence in one batch).
- Keep adapters in GPU memory with LRU caching, a CPU/SSD tier behind it, and preloading for hot tenants.
- Prefix-cache keys must include the adapter ID.
- Routing: send a tenant's requests to replicas with the adapter loaded, balanced against load.
- Per-tenant fairness, rate limits and usage metering. Isolation for noisy neighbours.
- Trade-off: high-rank adapters or full fine-tunes for the top customers might justify dedicated replicas.

#### What they're testing

Multi-tenancy, caching hierarchies and cost awareness.

[[play] Quiz Arcade ▶Rapid-fire multiple choice to lock in the facts.](practice.md#arcade) [[think] Workbook ▶Practise the napkin math these answers rely on.](workbook.md)
