# Scheduling & Disaggregated Serving

> Lesson 3-4 · WORLD 3 Advanced · ~30 min · topics: Scheduler policies, P/D disaggregation, KV-aware routing, Autoscaling

> previous: [3-3 Prefix Caching & KV Memory](../lessons/10-prefix-caching.md) · next: [4-1 Kernels, Fusion & Compilation](../lessons/12-kernels.md)

> Part of **InferenceLearn**, a free course on LLM inference engineering. Web version: lessons/11-disaggregation.html. Whole course in one file: llms-full.txt

Past a single engine, performance depends on *where* and *when* work runs: which requests to admit, which replica serves them, and whether prefill and decode should share GPUs at all.

#### By the end you'll be able to

- Compare scheduling policies and admission control
- Explain prefill/decode disaggregation and when it pays off
- Design KV- and prefix-aware request routing
- Pick the right signals for autoscaling LLM servers

#### Key vocabulary

- **[Admission control](../vocabulary.md#v-admission-control)**: Deciding whether to accept, queue or reject new requests, so the system doesn't overload and collapse.
- **[Disaggregated serving](../vocabulary.md#v-disaggregated-serving)**: Running prefill and decode on separate GPU pools and moving the KV cache between them.
- **[KV-aware routing](../vocabulary.md#v-kv-aware-routing)**: Sending a request to the replica that already caches its longest prefix, balanced against current load.
- **[Session affinity](../vocabulary.md#v-session-affinity)**: Routing a conversation's requests to the same replica (e.g. by hashing the session ID) to keep cache hits.
- **[Autoscaling](../vocabulary.md#v-autoscaling)**: Adding or removing replicas based on load signals such as queue depth, KV usage and latency against SLO.
- **[Cold start](../vocabulary.md#v-cold-start)**: The time for a new replica to become ready: pull the image, load the weights, compile, capture graphs. Often minutes.
- **[Goodput](../vocabulary.md#v-goodput)**: Throughput counting only requests that met their latency SLOs.

[Drill all terms as flashcards ▶](../vocabulary.md#flashcards)

## Scheduling inside an engine

Each step, the scheduler decides which waiting requests to admit and how to fill the token budget. Common policies:

- **FCFS (first come, first served)**: simple and fair in arrival order. The default in most engines.
- **Priority**: interactive traffic goes ahead of batch jobs. Low-priority work can be preempted.
- **Shortest-job-first** (using a predicted output length): lowers average latency, but long requests risk starvation.
- **Fairness across tenants**: stops one heavy customer from monopolizing the batch, e.g. with token-based fair queueing.
- **Cache-aware**: run requests that share a cached prefix close together in time, before the prefix gets evicted.

**Admission control** matters as much as ordering. Accepting more work than the KV cache can hold leads to preemption storms and latency collapse. It's better to queue, or return HTTP 429, early.

## Why disaggregate prefill & decode?

Prefill is compute-bound and bursty. Decode is memory-bound and steady. On the same GPU they interfere: prefills stall decodes (ITL spikes), and decodes slow prefills down (TTFT spikes). Chunked prefill reduces this but doesn't remove it. Both phases also share the *same* parallelism config and batch size, even though their ideal settings differ.

**Disaggregated serving** (DistServe, Splitwise, Mooncake) runs them on **separate GPU pools**:

Flow: request → PREFILL pool compute-optimized (e.g. higher TP) → DECODE pool memory-optimized (big batches) → token stream

*Diagram: the KV cache produced by prefill is transferred to a decode worker over NVLink / RDMA*

- **No interference**: TTFT and ITL can be tuned independently to meet their own SLOs.
- **Independent scaling**: prompt-heavy workloads such as RAG add prefill GPUs. Long-generation workloads add decode GPUs.
- **Different configs**: e.g. more parallelism for prefill and larger batches for decode. Splitwise even explored different GPU types per phase.
- **Costs**: moving KV takes bandwidth. A 2k-token prompt on Llama-3-70B is ~640 MB of KV. Over 50 GB/s RDMA that's ~13 ms, acceptable but not free. It adds complexity and new failure modes.

> **WHEN TO USE:** Disaggregation pays off at **scale**, with **strict, separate TTFT and ITL SLOs**, long prompts, and fast interconnects. For a few GPUs or loose SLOs, a single pool with chunked prefill is simpler and often just as good. Measure goodput both ways.

## Request routing

With many replicas, the load balancer is part of the inference system. Round-robin HTTP balancing ignores everything that matters:

- **Load-aware**: route on queue depth, running sequences or KV utilization, not open connection count.
- **Prefix / KV-aware**: send a request to the replica that already caches the longest matching prefix, such as the same conversation. This can raise hit rates a lot.
- **Balance the two**: always routing to the best cache hit creates hotspots. Practical routers score *cache overlap minus load*.
- **Session affinity**: a cheap approximation that hashes on conversation or user ID.

Open-source projects in this space include **NVIDIA Dynamo** (disaggregation plus a KV-aware router), **llm-d** (Kubernetes-native distributed inference) and SGLang's router.

## Autoscaling

- **Good signals**: queue length or waiting requests, KV cache utilization, TTFT/ITL p90 against the SLO, tokens/s against measured capacity.
- **Bad signals**: CPU utilization (meaningless for GPU work), and raw GPU "utilization" from `nvidia-smi`. That number only says a kernel was running, not how much work it did.
- **Cold starts are slow**: pulling a container plus loading 100+ GB of weights plus compiling and capturing CUDA graphs can take minutes. Mitigations: keep warm spares, cache weights on local NVMe, stream weights quickly, and scale ahead of known traffic patterns.
- **Scale to zero** saves money for rarely used models but forces cold starts. Multi-LoRA serving (many adapters on one base model) is often a better fit for the long tail.

## Check yourself

**Name two ways prefill and decode interfere when they share a GPU.**

(1) A long prefill in a step delays every running decode, which spikes ITL. (2) Many decodes in a step share compute and memory with a prefill, which delays it and raises TTFT. They also have to share the same parallelism and batching configuration.

**Why can pure "route to the best prefix match" be harmful?**

Popular prefixes would all go to one replica, overloading it while others sit idle. Routing must weigh cache benefit against current load.

**Your GPUs show 100% utilization in nvidia-smi. Are they at capacity?**

Not necessarily. That metric means "a kernel was active during the sample window", not "SMs were fully busy". Look at throughput against your benchmarked maximum, queue depth and KV usage instead.

## Further reading

- [DistServe: Disaggregating Prefill and Decoding for Goodput-optimized LLM Serving](https://arxiv.org/abs/2401.09670)
- [Splitwise: Efficient Generative LLM Inference Using Phase Splitting](https://arxiv.org/abs/2311.18677)
- [NVIDIA Dynamo](https://github.com/ai-dynamo/dynamo) · [llm-d](https://github.com/llm-d/llm-d)

## Interview corner

Questions on this lesson that come up in real interviews. Answer out loud first, then open the card.

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

[All interview questions ▶](../interview.md) · [Quiz Arcade ▶](../practice.md#arcade) · [Workbook ▶](../workbook.md)
