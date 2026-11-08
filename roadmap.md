# The Inference Engineering Roadmap

> Part of **InferenceLearn**, a free course on LLM inference engineering. Web version: roadmap.html. Whole course in one file: llms-full.txt

// world map

Start at World 0 and work down. Each stage lists its lessons and the topics they cover. Cleared lessons get a green check.

[WORLD 0] prerequisites · self-paced

### Before you start

You don't need to be an expert in any of these. Being comfortable with them makes everything after much easier. Each item links to a good free resource.

#### Programming

- Python: classes, generators, async basics
- PyTorch tensors, shapes, `matmul`, device placement ([PyTorch basics](https://pytorch.org/tutorials/beginner/basics/intro.html))
- Reading C++ / CUDA is a plus for World 4

#### Math

- Matrix multiplication and its FLOP count (2·m·n·k)
- Softmax, probability distributions, sampling
- Back-of-envelope estimation with orders of magnitude

#### Transformers

- Self-attention: Q, K, V, heads ([Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/))
- Decoder-only LLMs, embeddings, MLP / FFN blocks
- Build one yourself ([Karpathy: Let's build GPT](https://www.youtube.com/watch?v=kCc8FmEb1nY))

#### Systems

- Processes, memory, latency vs bandwidth
- HTTP APIs, queues, basic load balancing
- Linux and the command line, Docker

[WORLD 1] 3 lessons · ~70 min

### Foundations

How a token is actually produced, what a GPU is good and bad at, and how to measure an inference system.

- [1-1 · How LLM Inference Works](lessons/01-how-inference-works.md)Tokenization & logits · Autoregressive loop · Prefill vs decode · Sampling strategies
- [1-2 · GPU Fundamentals & the Roofline](lessons/02-gpu-fundamentals.md)HBM, SRAM, tensor cores · Arithmetic intensity · Roofline model · Speed-of-light math
- [1-3 · Metrics: Latency, Throughput & Cost](lessons/03-metrics.md)TTFT / TPOT / ITL · Throughput vs latency · Goodput & SLOs · Cost per 1M tokens

[WORLD 2] 4 lessons · ~130 min

### Core Techniques

The four ideas every serving engine is built on: KV caching, batching, fast attention and quantization.

- [2-1 · The KV Cache](lessons/04-kv-cache.md)Why caching works · Sizing formula · Memory budget · Reduction techniques
- [2-2 · Batching: Static → Continuous](lessons/05-batching.md)Why batching helps · Static & dynamic · Continuous batching · Chunked prefill
- [2-3 · Attention Optimizations](lessons/06-attention.md)MQA / GQA / MLA · FlashAttention · PagedAttention · Decode attention kernels
- [2-4 · Quantization](lessons/07-quantization.md)Scales & zero points · Granularity · Outliers · GPTQ / AWQ / FP8

[WORLD 3] 4 lessons · ~120 min

### Advanced

Squeezing more out of the hardware: speculative decoding, multi-GPU parallelism, prefix reuse and smart scheduling.

- [3-1 · Speculative Decoding](lessons/08-speculative-decoding.md)Draft & verify · Rejection sampling · Expected speedup · Medusa / EAGLE / MTP
- [3-2 · Parallelism & MoE Inference](lessons/09-parallelism.md)Tensor parallel · Pipeline parallel · Expert parallel & MoE · Context parallel
- [3-3 · Prefix Caching & KV Memory](lessons/10-prefix-caching.md)Shared prefixes · Hash blocks vs radix tree · Eviction & preemption · KV offloading
- [3-4 · Scheduling & Disaggregated Serving](lessons/11-disaggregation.md)Scheduler policies · P/D disaggregation · KV-aware routing · Autoscaling

[WORLD 4] 2 lessons · ~70 min

### Expert / Systems

Writing and fusing kernels, then running the whole thing reliably in production.

- [4-1 · Kernels, Fusion & Compilation](lessons/12-kernels.md)Kernel fusion · CUDA graphs · Triton & CUTLASS · Profiling (MFU / MBU)
- [4-2 · Serving in Production](lessons/13-production.md)Choosing an engine · Capacity planning · Benchmarking · Observability & reliability

// boss level

## Capstone projects

Reading gets you about halfway. Building gets you the rest. Pick one project per world.

[World 1 project]

### Naive generator + benchmark

Write a token-by-token generation loop for a small Hugging Face model in plain PyTorch. Measure TTFT and per-token latency, then compare them with the speed-of-light number for your GPU.

[World 2 project]

### Add a KV cache & batching

Extend your generator with a KV cache, then a simple continuous-batching scheduler with a fixed slot count. Plot throughput vs concurrency before and after.

[World 3 project]

### Speculative decoding + prefix cache

Use a small draft model to speed up a bigger one, and check that greedy outputs match exactly. Add a hash-based prefix cache and measure TTFT on repeated system prompts.

[World 4 project]

### Mini inference engine

Build a "nano-vLLM": paged KV blocks, a scheduler with chunked prefill, CUDA graphs for decode, and an OpenAI-compatible streaming endpoint. Benchmark it against vLLM and explain the gap.

// suggested pace

## An 8-week study plan

About 5–8 hours a week. Go faster or slower as you like.

| Week | Focus | Do |
|---|---|---|
| 1 | World 0 + lesson 1-1 | Build GPT from scratch (Karpathy). Write the naive generation loop. |
| 2 | Lessons 1-2, 1-3 | Roofline math for your GPU. Benchmark your naive loop. |
| 3 | Lessons 2-1, 2-2 | Add a KV cache and continuous batching to your loop. |
| 4 | Lessons 2-3, 2-4 | Read the FlashAttention and vLLM papers. Quantize a model to INT4/FP8 and run evals. |
| 5 | Lessons 3-1, 3-2 | Implement speculative decoding. Run vLLM with `--tensor-parallel-size 2` if you have the hardware. |
| 6 | Lessons 3-3, 3-4 | Benchmark prefix caching and chunked prefill in vLLM or SGLang. |
| 7 | Lesson 4-1 | Write a fused Triton kernel. Profile a decode step with Nsight Systems. |
| 8 | Lesson 4-2 | Capstone: deploy, load-test and write up your results. |
