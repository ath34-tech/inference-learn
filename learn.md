# All Lessons

> Part of **InferenceLearn**, a free course on LLM inference engineering. Web version: learn.html. Whole course in one file: llms-full.txt

// curriculum

Thirteen lessons in four worlds. Each one has learning goals, diagrams, worked math, code, and a quiz at the end.

## [WORLD 1]  Foundations

How a token is actually produced, what a GPU is good and bad at, and how to measure an inference system.

[1-1How LLM Inference WorksTokens, the autoregressive loop, prefill vs decode, and sampling.4 topics~20 min](lessons/01-how-inference-works.md)[1-2GPU Fundamentals & the RooflineMemory hierarchy, arithmetic intensity, and why decode is memory-bound.4 topics~30 min](lessons/02-gpu-fundamentals.md)[1-3Metrics: Latency, Throughput & CostTTFT, ITL, throughput, goodput, percentiles and cost per token.4 topics~20 min](lessons/03-metrics.md)

## [WORLD 2]  Core Techniques

The four ideas every serving engine is built on: KV caching, batching, fast attention and quantization.

[2-1The KV CacheWhy we cache keys and values, how big the cache gets, and how to shrink it.4 topics~30 min](lessons/04-kv-cache.md)[2-2Batching: Static → ContinuousStatic, dynamic and continuous batching, plus chunked prefill.4 topics~30 min](lessons/05-batching.md)[2-3Attention OptimizationsMQA/GQA/MLA, FlashAttention and PagedAttention.4 topics~35 min](lessons/06-attention.md)[2-4QuantizationINT8, FP8, INT4 and FP4; weight-only vs weight+activation; GPTQ, AWQ, SmoothQuant.4 topics~35 min](lessons/07-quantization.md)

## [WORLD 3]  Advanced

Squeezing more out of the hardware: speculative decoding, multi-GPU parallelism, prefix reuse and smart scheduling.

[3-1Speculative DecodingDraft then verify: get more than one token per forward pass, with the same output distribution.4 topics~30 min](lessons/08-speculative-decoding.md)[3-2Parallelism & MoE InferenceTensor, pipeline, data, expert and context parallelism.4 topics~35 min](lessons/09-parallelism.md)[3-3Prefix Caching & KV MemoryReusing KV across requests, radix trees, eviction and offloading tiers.4 topics~25 min](lessons/10-prefix-caching.md)[3-4Scheduling & Disaggregated ServingSchedulers, prefill/decode disaggregation, routing and autoscaling.4 topics~30 min](lessons/11-disaggregation.md)

## [WORLD 4]  Expert / Systems

Writing and fusing kernels, then running the whole thing reliably in production.

[4-1Kernels, Fusion & CompilationKernel fusion, CUDA graphs, Triton, torch.compile and profiling.4 topics~35 min](lessons/12-kernels.md)[4-2Serving in ProductionEngines, capacity planning, benchmarking, observability and a launch checklist.4 topics~35 min](lessons/13-production.md)
