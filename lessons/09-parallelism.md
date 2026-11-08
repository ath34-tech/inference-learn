# Parallelism & MoE Inference

> Lesson 3-2 · WORLD 3 Advanced · ~35 min · topics: Tensor parallel, Pipeline parallel, Expert parallel & MoE, Context parallel

> previous: [3-1 Speculative Decoding](../lessons/08-speculative-decoding.md) · next: [3-3 Prefix Caching & KV Memory](../lessons/10-prefix-caching.md)

> Part of **InferenceLearn**, a free course on LLM inference engineering. Web version: lessons/09-parallelism.html. Whole course in one file: llms-full.txt

When a model doesn't fit on one GPU, or one GPU isn't fast enough, you split the work. Each way of splitting trades communication for memory, latency or throughput, and the right mix depends on your model and your interconnect.

#### By the end you'll be able to

- Describe tensor, pipeline, data, expert and context parallelism
- Say which ones reduce latency and which only add throughput
- Reason about communication costs and interconnects
- Explain the special memory/compute profile of Mixture-of-Experts models

#### Key vocabulary

- **[Tensor parallelism (TP)](../vocabulary.md#v-tensor-parallelism-tp)**: Splits each layer's weight matrices across GPUs, with all-reduces per layer. Lowers per-token latency.
- **[Pipeline parallelism (PP)](../vocabulary.md#v-pipeline-parallelism-pp)**: Splits a model by layers across GPUs. Works across nodes, but doesn't make a single token faster.
- **[Pipeline bubble](../vocabulary.md#v-pipeline-bubble)**: Idle time in pipeline stages while waiting for work. It's reduced by keeping several micro-batches in flight.
- **[Data parallelism (DP)](../vocabulary.md#v-data-parallelism-dp)**: Independent model replicas with requests load-balanced across them. Scales throughput.
- **[Expert parallelism (EP)](../vocabulary.md#v-expert-parallelism-ep)**: Places different MoE experts on different GPUs and routes tokens to them with all-to-all communication.
- **[MoE](../vocabulary.md#v-moe)**: Mixture of Experts: a router sends each token to a few of many expert MLPs. Total parameters ≫ active parameters.
- **[Router (gating)](../vocabulary.md#v-router-gating)**: The small network in an MoE layer that picks the top-k experts for each token.
- **[All-reduce](../vocabulary.md#v-all-reduce)**: A collective operation that sums tensors across GPUs and gives everyone the result. TP needs two per transformer layer.
- **[All-to-all](../vocabulary.md#v-all-to-all)**: A collective where every GPU sends a different chunk to every other GPU. Used for MoE token dispatch.
- **[NVLink](../vocabulary.md#v-nvlink)**: NVIDIA's high-bandwidth GPU-to-GPU link (~900 GB/s per GPU on H100). Essential for fast tensor parallelism.

[Drill all terms as flashcards ▶](../vocabulary.md#flashcards)

## Why go multi-GPU?

- **Capacity**: a 405B model in BF16 is ~810 GB, more than any single GPU holds.
- **Latency**: decode speed ≈ bandwidth. Split the weights over 8 GPUs and you have ~8× the aggregate bandwidth for each token.
- **Throughput**: more total memory means more KV cache, bigger batches and more users.

## Tensor parallelism (TP)

Split *each layer's* weight matrices across GPUs (Megatron-LM style). In an MLP block, the first matrix is split by **columns** and the second by **rows**. Each GPU computes its slice independently, and one **all-reduce** sums the partial outputs at the end. Attention is split by heads in the same way.

```
GPU 0: W₁[:, 0:¼] W₂[0:¼, :] partial Y
GPU 1: W₁[:, ¼:½] W₂[¼:½, :] partial Y
…: … … all-reduce → Y
```

*Diagram: 2 all-reduces per transformer layer (one after attention, one after the MLP)*

- **Reduces latency**: each GPU reads only 1/TP of the weights per step.
- **Communication-heavy**: two all-reduces per layer, every step. You need fast links (NVLink/NVSwitch inside a node). TP across nodes over Ethernet is usually a bad idea.
- TP is limited by the number of KV heads. With 8 KV heads, TP=16 has to replicate KV heads.

## Pipeline parallelism (PP)

Split the model *by layers*: GPU 0 runs layers 0–19, GPU 1 runs 20–39, and so on. Activations flow from stage to stage. Communication is small (one activation tensor per stage boundary), so PP works across nodes.

- **Doesn't reduce single-request latency**: one token still passes through every layer in sequence.
- Needs **multiple micro-batches in flight** to keep all stages busy. Otherwise you get "pipeline bubbles".
- Common pattern: **TP inside a node, PP across nodes**.

## Data parallelism (replicas)

Run independent copies of the model and load-balance requests across them. There's no communication during inference, and throughput scales linearly. It's the default way to scale out once one replica meets your latency target. Its "cost" is memory: each replica holds a full copy of the weights.

## Expert parallelism & MoE

**Mixture-of-Experts** models replace each MLP with many "expert" MLPs plus a **router** that sends each token to its top-k experts. Mixtral 8x7B has ~47B total parameters, but each token uses only ~13B of them.

> **KEY IDEA:** MoE splits **memory** from **compute**: you must *store* all parameters but only *compute with* the active ones. At batch 1 you read only the active experts. At large batch, different tokens hit different experts, so nearly all experts get read every step. Large-scale MoE serving is then about using all those weight reads well.

- **Expert parallelism (EP)**: place different experts on different GPUs. Each token is sent to the GPUs holding its chosen experts via **all-to-all** communication (dispatch), then sent back (combine).
- **Load imbalance**: popular experts become hotspots. Fixes include redundant copies of hot experts and capacity limits.
- Large-scale deployments of models such as DeepSeek-V3 combine attention data parallelism with wide EP across many GPUs, using specialized all-to-all kernels such as DeepEP.

## Context / sequence parallelism

For very long contexts (hundreds of thousands to millions of tokens), even a single sequence's prefill and KV cache are too much for one GPU. **Context parallelism** splits the *sequence* across GPUs. Approaches like **Ring Attention** pass K/V blocks around a ring of devices while each GPU computes attention for its own chunk of queries, overlapping communication with compute.

## Choosing a layout

| Strategy | Splits | Lower latency? | Comm pattern | Best for |
|---|---|---|---|---|
| TP | within layers | ✓ yes | all-reduce ×2 / layer | Inside one NVLink node |
| PP | across layers | ✗ no | point-to-point | Across nodes, huge models |
| DP | requests | ✗ no | none | Scaling throughput |
| EP | MoE experts | ~ | all-to-all | Large MoE models |
| CP / SP | sequence | ✓ (long prompts) | ring / all-gather | Very long context |

```
# vLLM examples
vllm serve meta-llama/Llama-3.1-70B-Instruct --tensor-parallel-size 4
# 2 nodes × 8 GPUs: TP inside each node, PP across nodes
vllm serve <huge-model> --tensor-parallel-size 8 --pipeline-parallel-size 2
```

> **RULE OF THUMB:** Use the **smallest TP that fits the model with enough KV room** and meets your latency target, then scale out with data-parallel replicas. Higher TP means more communication overhead per token, and its gains flatten out.

## Check yourself

**You need lower per-token latency for a 70B model that already fits on 2 GPUs. TP=4 or PP=2 with DP=2?**

**TP=4**. It splits each step's weight reads across 4 GPUs, cutting per-token time. PP doesn't make a single token faster, and DP only adds throughput.

**Why does an MoE model with 37B active parameters (out of 671B) still need many GPUs?**

All 671B parameters must be *stored* somewhere, because any token can be routed to any expert. Memory capacity is set by total parameters, while per-token compute is set by active ones.

**Why is TP across nodes over regular Ethernet usually slow?**

TP needs two all-reduces per layer on every step. That's latency-critical communication on the hot path, and Ethernet is far slower than NVLink in both bandwidth and latency.

## Further reading

- [Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/abs/1909.08053)
- [Mixtral of Experts](https://arxiv.org/abs/2401.04088) · [Switch Transformers](https://arxiv.org/abs/2101.03961)
- [Ring Attention with Blockwise Transformers](https://arxiv.org/abs/2310.01889)
- [How to Scale Your Model](https://jax-ml.github.io/scaling-book/) (sharding chapters)

## Interview corner

Questions on this lesson that come up in real interviews. Answer out loud first, then open the card.

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

[All interview questions ▶](../interview.md) · [Quiz Arcade ▶](../practice.md#arcade) · [Workbook ▶](../workbook.md)
