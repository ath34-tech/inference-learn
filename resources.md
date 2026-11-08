# Resources

> Part of **InferenceLearn**, a free course on LLM inference engineering. Web version: resources.html. Whole course in one file: llms-full.txt

// item shop

Hand-picked papers, blog posts, videos, courses and open-source projects, grouped by topic. If you're new, start with the first group.

## ★ Start here

[blog] [beginner]

### [Large Transformer Model Inference Optimization](https://lilianweng.github.io/posts/2023-01-10-inference-optimization/)

Lilian Weng · 2023

A broad survey of the techniques: distillation, quantization, pruning, sparsity, efficient attention.

[blog] [beginner]

### [Transformer Inference Arithmetic](https://kipp.ly/transformer-inference-arithmetic/)

kipply · 2022

The classic napkin-math walkthrough of KV cache, FLOPs, memory and latency.

[blog] [beginner]

### [Making Deep Learning Go Brrrr From First Principles](https://horace.io/brrr_intro.html)

Horace He · 2022

Compute vs memory vs overhead: the three regimes every kernel lives in.

[book] [intermediate]

### [How to Scale Your Model](https://jax-ml.github.io/scaling-book/)

Google DeepMind · 2025

A free online book on rooflines, sharding and transformer inference on TPUs and GPUs.

[blog] [beginner]

### [Mastering LLM Techniques: Inference Optimization](https://developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization/)

NVIDIA Technical Blog

Batching, KV cache, parallelism, quantization and speculative decoding in one overview.

[blog] [beginner]

### [LLM Inference Performance Engineering: Best Practices](https://www.databricks.com/blog/llm-inference-performance-engineering-best-practices)

Databricks · 2023

Metrics, MBU, batching and hardware choices, written by practitioners.

## Foundations

[paper] [beginner]

### [Attention Is All You Need](https://arxiv.org/abs/1706.03762)

Vaswani et al. · 2017

The transformer paper. Read it once and you'll recognize every block.

[blog] [beginner]

### [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)

Jay Alammar

The best visual introduction to attention, Q/K/V and multi-head attention.

[video] [beginner]

### [Let's build GPT: from scratch, in code, spelled out](https://www.youtube.com/watch?v=kCc8FmEb1nY)

Andrej Karpathy

Build a decoder-only transformer line by line. The best possible prerequisite.

[repo] [beginner]

### [nanoGPT](https://github.com/karpathy/nanoGPT)

Andrej Karpathy

A tiny, readable GPT implementation to hack your first inference experiments on.

[paper] [intermediate]

### [Efficiently Scaling Transformer Inference](https://arxiv.org/abs/2211.05102)

Pope et al. (Google) · 2022

Partitioning strategies, latency/cost trade-offs and MQA at scale. Foundational.

[blog] [beginner]

### [How to generate text: decoding methods](https://huggingface.co/blog/how-to-generate)

Hugging Face

Greedy, beam search, top-k and top-p sampling explained with examples.

[paper] [intermediate]

### [LLM Inference Unveiled: Survey and Roofline Model Insights](https://arxiv.org/abs/2402.16363)

Yuan et al. · 2024

A survey that analyzes inference techniques through the roofline model.

[paper] [intermediate]

### [A Survey on Efficient Inference for Large Language Models](https://arxiv.org/abs/2404.14294)

Zhou et al. · 2024

A taxonomy of data-, model- and system-level optimizations.

## KV cache & attention

[paper] [intermediate]

### [FlashAttention](https://arxiv.org/abs/2205.14135)

Dao et al. · 2022

IO-aware exact attention with tiling and online softmax.

[paper] [advanced]

### [FlashAttention-2](https://arxiv.org/abs/2307.08691)

Tri Dao · 2023

Better parallelism and work partitioning for about 2× more speed.

[paper] [advanced]

### [FlashAttention-3](https://arxiv.org/abs/2407.08608)

Shah et al. · 2024

Hopper-specific: asynchrony, warp specialization and FP8.

[paper] [intermediate]

### [Efficient Memory Management for LLM Serving with PagedAttention](https://arxiv.org/abs/2309.06180)

Kwon et al. (vLLM) · SOSP 2023

Paged KV cache, near-zero fragmentation and copy-on-write sharing.

[blog] [beginner]

### [vLLM: Easy, Fast, and Cheap LLM Serving with PagedAttention](https://blog.vllm.ai/2023/06/20/vllm.html)

vLLM team · 2023

A friendly, illustrated introduction to PagedAttention.

[blog] [intermediate]

### [Flash-Decoding for long-context inference](https://crfm.stanford.edu/2023/10/12/flashdecoding.html)

Dao, Haziza, Massa, Sizov · 2023

Split-KV parallelism for fast decode attention at long context.

[paper] [intermediate]

### [Fast Transformer Decoding: One Write-Head is All You Need](https://arxiv.org/abs/1911.02150)

Noam Shazeer · 2019

Introduces multi-query attention (MQA).

[paper] [intermediate]

### [GQA: Training Generalized Multi-Query Transformer Models](https://arxiv.org/abs/2305.13245)

Ainslie et al. · 2023

Grouped-query attention, today's default KV-head layout.

[paper] [advanced]

### [DeepSeek-V2](https://arxiv.org/abs/2405.04434)

DeepSeek-AI · 2024

Introduces multi-head latent attention (MLA) and a very compact KV cache.

[paper] [intermediate]

### [Efficient Streaming Language Models with Attention Sinks](https://arxiv.org/abs/2309.17453)

Xiao et al. · 2023

StreamingLLM: keep sink tokens and a window to stream indefinitely.

[repo] [advanced]

### [FlashInfer](https://github.com/flashinfer-ai/flashinfer)

flashinfer-ai

A kernel library for LLM serving: paged and ragged attention, sampling and more.

## Batching, scheduling & serving systems

[paper] [intermediate]

### [Orca: A Distributed Serving System for Transformer-Based Generative Models](https://www.usenix.org/conference/osdi22/presentation/yu)

Yu et al. · OSDI 2022

Iteration-level scheduling, the origin of continuous batching.

[blog] [beginner]

### [How continuous batching enables 23× throughput](https://www.anyscale.com/blog/continuous-batching-llm-inference)

Anyscale · 2023

A clear explanation with benchmarks of static vs continuous batching.

[paper] [intermediate]

### [Sarathi-Serve: Taming Throughput-Latency Tradeoff](https://arxiv.org/abs/2403.02310)

Agrawal et al. · OSDI 2024

Chunked prefill and stall-free batching.

[paper] [advanced]

### [DistServe](https://arxiv.org/abs/2401.09670)

Zhong et al. · OSDI 2024

Prefill/decode disaggregation, optimized for goodput.

[paper] [advanced]

### [Splitwise](https://arxiv.org/abs/2311.18677)

Patel et al. · ISCA 2024

Phase splitting across machines, including different hardware per phase.

[paper] [advanced]

### [Mooncake: A KVCache-centric Disaggregated Architecture](https://arxiv.org/abs/2407.00079)

Moonshot AI · 2024

Large-scale disaggregated serving built around a distributed KV store.

[paper] [intermediate]

### [SGLang: Efficient Execution of Structured LM Programs](https://arxiv.org/abs/2312.07104)

Zheng et al. · 2023

RadixAttention prefix caching and fast constrained decoding.

## Quantization

[blog] [beginner]

### [A Visual Guide to Quantization](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-quantization)

Maarten Grootendorst · 2024

An illustrated tour from number formats to GPTQ, AWQ and GGUF.

[paper] [intermediate]

### [LLM.int8()](https://arxiv.org/abs/2208.07339)

Dettmers et al. · 2022

Finds emergent activation outliers. Mixed-precision INT8 matmul.

[paper] [intermediate]

### [SmoothQuant](https://arxiv.org/abs/2211.10438)

Xiao et al. · 2022

Moves quantization difficulty from activations to weights to enable W8A8.

[paper] [advanced]

### [GPTQ](https://arxiv.org/abs/2210.17323)

Frantar et al. · 2022

Second-order, error-compensating post-training weight quantization.

[paper] [intermediate]

### [AWQ: Activation-aware Weight Quantization](https://arxiv.org/abs/2306.00978)

Lin et al. · MLSys 2024

Protects salient channels with scaling. Simple and effective INT4.

[paper] [intermediate]

### [FP8 Formats for Deep Learning](https://arxiv.org/abs/2209.05433)

Micikevicius et al. · 2022

Defines the E4M3 and E5M2 formats used by modern GPUs.

[docs] [beginner]

### [Hugging Face: Quantization overview](https://huggingface.co/docs/transformers/main/en/quantization/overview)

Hugging Face docs

A practical comparison of quantization backends and how to use them.

## Speculative decoding

[paper] [intermediate]

### [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192)

Leviathan, Kalman, Matias · ICML 2023

The draft-and-verify idea with a lossless acceptance rule.

[paper] [intermediate]

### [Accelerating LLM Decoding with Speculative Sampling](https://arxiv.org/abs/2302.01318)

Chen et al. (DeepMind) · 2023

Concurrent discovery, with results at Chinchilla scale.

[paper] [advanced]

### [Medusa](https://arxiv.org/abs/2401.10774)

Cai et al. · 2024

Multiple decoding heads plus tree attention. No separate draft model.

[paper] [advanced]

### [EAGLE](https://arxiv.org/abs/2401.15077)

Li et al. · 2024

Feature-level drafting with high acceptance rates. Followed by EAGLE-2 and EAGLE-3.

[paper] [advanced]

### [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)

DeepSeek-AI · 2024

Multi-token prediction (usable for speculation), FP8 and large-scale MoE deployment.

## Parallelism & MoE

[paper] [intermediate]

### [Megatron-LM](https://arxiv.org/abs/1909.08053)

Shoeybi et al. · 2019

The tensor-parallel layer-splitting scheme used everywhere.

[paper] [intermediate]

### [Switch Transformers](https://arxiv.org/abs/2101.03961)

Fedus, Zoph, Shazeer · 2021

Simple top-1 routing MoE at trillion-parameter scale.

[paper] [intermediate]

### [Mixtral of Experts](https://arxiv.org/abs/2401.04088)

Mistral AI · 2024

A widely used open sparse MoE: 8 experts, top-2 routing.

[paper] [advanced]

### [Ring Attention with Blockwise Transformers](https://arxiv.org/abs/2310.01889)

Liu, Zaharia, Abbeel · 2023

Context parallelism for near-infinite sequence lengths.

[repo] [advanced]

### [DeepEP](https://github.com/deepseek-ai/DeepEP)

DeepSeek-AI

High-performance all-to-all kernels for expert parallelism.

## GPU programming & kernels

[docs] [intermediate]

### [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)

NVIDIA

The official reference for the CUDA programming and memory model.

[tutorial] [intermediate]

### [Triton tutorials](https://triton-lang.org/main/getting-started/tutorials/index.html)

OpenAI / Triton

From vector add to fused softmax, matmul and FlashAttention in Triton.

[video] [intermediate]

### [GPU MODE lectures (YouTube)](https://www.youtube.com/@GPUMODE)

GPU MODE community

Weekly talks on CUDA, Triton, profiling, quantization kernels and more.

[repo] [intermediate]

### [GPU MODE lecture materials](https://github.com/gpu-mode/lectures)

GPU MODE

Slides and code that go with the lecture series.

[blog] [intermediate]

### [Accelerating Generative AI with PyTorch II: GPT, Fast](https://pytorch.org/blog/accelerating-generative-ai-2/)

PyTorch team · 2023

torch.compile, INT8/INT4, speculative decoding and TP in under 1,000 lines.

[repo] [intermediate]

### [gpt-fast](https://github.com/pytorch-labs/gpt-fast)

PyTorch Labs

The code from the blog above. Small, fast and very readable.

[repo] [advanced]

### [CUTLASS / CuTe](https://github.com/NVIDIA/cutlass)

NVIDIA

CUDA templates for high-performance GEMM and attention kernels.

[book] [intermediate]

### Programming Massively Parallel Processors (4th ed.)

Hwu, Kirk, El Hajj · 2022

The standard textbook on CUDA and parallel algorithms. Pairs well with GPU MODE.

## Inference engines & tools

[repo] [engine]

### [vLLM](https://github.com/vllm-project/vllm)

vLLM project

High-throughput serving engine with PagedAttention and broad model and hardware support.

[repo] [engine]

### [SGLang](https://github.com/sgl-project/sglang)

SGLang project

Fast serving with RadixAttention, structured output and large-scale MoE support.

[repo] [engine]

### [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM)

NVIDIA

NVIDIA's optimized LLM inference library: FP8/FP4, in-flight batching and more.

[repo] [engine]

### [llama.cpp](https://github.com/ggml-org/llama.cpp)

ggml-org

C/C++ inference on CPUs, Apple Silicon and GPUs. The home of GGUF.

[repo] [engine]

### [Ollama](https://github.com/ollama/ollama)

Ollama

The easiest way to run models locally. Great for experiments.

[repo] [engine]

### [Text Generation Inference](https://github.com/huggingface/text-generation-inference)

Hugging Face

Hugging Face's production serving toolkit.

[repo] [engine]

### [LMDeploy](https://github.com/InternLM/lmdeploy)

InternLM

A toolkit for compressing, deploying and serving LLMs (TurboMind engine).

[repo] [engine]

### [MLC LLM](https://github.com/mlc-ai/mlc-llm)

MLC AI

A compiler-based engine for phones, browsers (WebGPU) and diverse GPUs.

[repo] [platform]

### [NVIDIA Dynamo](https://github.com/ai-dynamo/dynamo)

NVIDIA

Distributed inference framework: disaggregated serving and a KV-aware router.

[repo] [platform]

### [llm-d](https://github.com/llm-d/llm-d)

llm-d project

Kubernetes-native distributed inference with smart routing and disaggregation.

[repo] [platform]

### [LMCache](https://github.com/LMCache/LMCache)

LMCache

A KV cache layer for storing and reusing KV across GPU, CPU and disk.

[repo] [learn]

### [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm)

GeeeekExplorer

A small vLLM-style engine you can read in an afternoon. Ideal for the capstone.

[repo] [tool]

### [GuideLLM](https://github.com/vllm-project/guidellm)

vLLM project

Load testing and SLO-oriented benchmarking for LLM servers.

[repo] [tool]

### [Triton Inference Server](https://github.com/triton-inference-server/server)

NVIDIA

A general model server with dynamic batching and multi-framework backends.

## Courses & books

[book] [intermediate]

### [Machine Learning Engineering Open Book](https://github.com/stas00/ml-engineering)

Stas Bekman

Hardware, networking, debugging and inference: lessons from training and serving at scale.

[course] [intermediate]

### [Stanford CS336: Language Modeling from Scratch](https://stanford-cs336.github.io/spring2025/)

Stanford · 2025

Build an LLM end to end, including GPU kernels, parallelism and an inference lecture.

[course] [intermediate]

### [MIT 6.5940: TinyML and Efficient Deep Learning](https://hanlab.mit.edu/courses/2024-fall-65940)

Song Han, MIT

Quantization, pruning, distillation and efficient LLM deployment.
