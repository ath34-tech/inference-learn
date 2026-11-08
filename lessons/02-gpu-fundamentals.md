# GPU Fundamentals & the Roofline

> Lesson 1-2 · WORLD 1 Foundations · ~30 min · topics: HBM, SRAM, tensor cores, Arithmetic intensity, Roofline model, Speed-of-light math

> previous: [1-1 How LLM Inference Works](../lessons/01-how-inference-works.md) · next: [1-3 Metrics: Latency, Throughput & Cost](../lessons/03-metrics.md)

> Part of **InferenceLearn**, a free course on LLM inference engineering. Web version: lessons/02-gpu-fundamentals.html. Whole course in one file: llms-full.txt

You can't optimize what you can't estimate. This lesson gives you the mental model and the napkin math to predict, before running any code, how fast an LLM *could* run on a given GPU.

#### By the end you'll be able to

- Name the parts of a GPU that matter for inference
- Compute **arithmetic intensity** and place an operation on the **roofline**
- Explain exactly why decode is memory-bound and prefill is compute-bound
- Estimate the "speed of light" latency and throughput for any model and GPU

#### Key vocabulary

- **[Streaming Multiprocessor (SM)](../vocabulary.md#v-streaming-multiprocessor-sm)**: One of the GPU's many independent compute units (132 on an H100 SXM). Each has cores, tensor cores, registers and shared memory.
- **[Tensor cores](../vocabulary.md#v-tensor-cores)**: Specialized units for very fast matrix multiply-accumulate at low precision (BF16, FP8, FP4). Most of a GPU's FLOPs come from them.
- **[HBM](../vocabulary.md#v-hbm)**: High-Bandwidth Memory: the GPU's main memory (e.g. 80 GB at ~3.35 TB/s on H100). It holds the weights and KV cache.
- **[SRAM / shared memory](../vocabulary.md#v-sram-shared-memory)**: Small, very fast on-chip memory per SM (~228 KB on H100). Kernels stage tiles of data here.
- **[Arithmetic intensity](../vocabulary.md#v-arithmetic-intensity)**: FLOPs done per byte moved from memory. For LLM linear layers in BF16 it's roughly the number of tokens in the batch.
- **[Roofline model](../vocabulary.md#v-roofline-model)**: Attainable performance = min(peak FLOP/s, intensity × bandwidth). It tells you which limit you'll hit first.
- **[Ridge point](../vocabulary.md#v-ridge-point)**: Peak FLOP/s ÷ bandwidth: the intensity where a GPU switches from memory-bound to compute-bound (~295 FLOP/byte on H100).
- **[Memory-bound](../vocabulary.md#v-memory-bound)**: Limited by how fast data moves, not by math. Compute units sit idle waiting for bytes. Typical of decode.
- **[Compute-bound](../vocabulary.md#v-compute-bound)**: Limited by math throughput. Memory keeps up and the tensor cores are the bottleneck. Typical of prefill and large batches.

[Drill all terms as flashcards ▶](../vocabulary.md#flashcards)

## A GPU in one picture

A modern data-center GPU (NVIDIA H100, say) is a big grid of **Streaming Multiprocessors (SMs)**, 132 on an H100 SXM. Each SM has CUDA cores for general math, **tensor cores** for very fast matrix multiply, a register file, and a small fast **shared memory / L1** (SRAM). All SMs share an **L2 cache** and a large but comparatively slow **HBM** (high-bandwidth memory) that holds weights and the KV cache.

| Level | Size (H100 SXM, approx.) | Bandwidth | Holds |
|---|---|---|---|
| Registers | 256 KB per SM | highest | values a thread is working on |
| Shared mem / L1 (SRAM) | up to ~228 KB per SM | tens of TB/s total | tiles of matrices inside a kernel |
| L2 cache | 50 MB | ~several TB/s | recently used data |
| **HBM3** | 80 GB | **~3.35 TB/s** | weights, KV cache, activations |
| Host DRAM (over PCIe 5) | 100s of GB | ~64 GB/s per direction | offloaded data |

The key fact: tensor cores can do **~989 TFLOPS** of dense BF16 math, but HBM can only deliver **~3.35 TB/s**. So the GPU can do about **300 FLOPs in the time it takes to load one byte** from HBM. Any operation that does less work per byte than that leaves the compute idle.

## Arithmetic intensity

> **Arithmetic Intensity (AI)** = FLOPs performed / bytes moved to and from memory

Take a linear layer, `Y = X · W`, with `X` of shape `[B, d]` (B tokens) and `W` of shape `[d, d]` in BF16 (2 bytes per number):

- FLOPs: `2 · B · d · d` (a multiply and an add for each output element times d)
- Bytes: `2·d²` for the weights, plus `2·B·d` in and `2·B·d` out. When B ≪ d, the weights dominate: ≈ `2·d²`

> AI ≈ (2·B·d²) / (2·d²) = **B** FLOPs per byte

> **KEY IDEA:** For LLM linear layers in BF16, **arithmetic intensity ≈ the number of tokens in the batch**. A single decode step (B = 1) does about 1 FLOP per byte, roughly 300× below what an H100 needs to be compute-bound. Prefill with a 2,000-token prompt has AI ≈ 2,000 and is comfortably compute-bound.

## The roofline model

The roofline model says the performance you can reach is capped by whichever of two limits you hit first:

> attainable FLOP/s = **min**( peak FLOP/s , AI × memory bandwidth )

```
  FLOP/s
    ▲
989T┤                    ┌──────────────────────── compute roof
    │                  ╱ │
    │                ╱   │   prefill lives here  ■■■
    │              ╱     │
    │            ╱       │
    │          ╱  memory │
    │        ╱   roof    │
    │      ╱  (slope=BW) │
    │ ■  ╱  ← decode     │
    └──┴─────────────────┴──────────────────────────▶ arithmetic intensity
      ~1                ~295 = ridge point           (FLOP / byte)
```

*Diagram: left of the ridge: memory-bound · right of the ridge: compute-bound*

The **ridge point** is `peak FLOP/s ÷ bandwidth`. For an H100 that's 989e12 / 3.35e12 ≈ **295 FLOP/byte**. In BF16, a batch needs roughly 300 tokens being processed together before the linear layers become compute-bound. (Attention over the KV cache is a separate story. Its AI stays low even with batching, because every sequence has its own KV. More on that in lesson 2-1.)

## Speed-of-light math

At low batch, a decode step must read every weight once, so a hard lower bound is:

> decode step time ≥ **model bytes / HBM bandwidth**

**Worked example:** Llama-3-8B in BF16 is 8×10⁹ × 2 bytes = 16 GB. On an H100: 16 GB / 3.35 TB/s ≈ **4.8 ms per step**, so at most **~209 tokens/s** for a single sequence. That's the ceiling before counting KV-cache reads, kernel overheads or anything else. Real engines often reach 60–85% of it.

Prefill is compute-bound, so its bound uses FLOPs instead. A forward pass costs about `2 × params` FLOPs per token:

> prefill time ≥ **2 · params · prompt_tokens / peak FLOP/s**

For 8B params and 2,000 tokens: 2 × 8e9 × 2000 / 989e12 ≈ **32 ms** at 100% utilization. Real Model FLOPs Utilization (MFU) is often 40–70%, so expect ~50–80 ms.

> **Interactive lab: Speed-of-light lab.** Pick a GPU and a model, then **drag the batch slider**. Watch the yellow dot move along the roofline. When it crosses the ridge, the bottleneck flips from memory to compute. *(Use it on the web page. The formulas it uses are explained in the text.)* Ideal bounds from dense BF16 spec numbers. It ignores KV-cache reads, communication and overheads, and uses BF16 peak FLOPs even for low-bit weights.

## Numeric formats

Fewer bytes per weight means fewer bytes to move and more tokens per second in memory-bound decode. Newer GPUs also have tensor cores that run faster at lower precision.

| Format | Bytes | Notes |
|---|---|---|
| FP32 | 4 | Rarely used for LLM inference weights today |
| BF16 / FP16 | 2 | The default "full precision" for serving. BF16 has FP32's exponent range |
| FP8 (E4M3 / E5M2) | 1 | Native tensor-core support on Hopper / Ada and newer. ~2× BF16 FLOPs |
| INT8 | 1 | Mature, widely supported. Needs good scaling for activations |
| INT4 / FP4 | 0.5 | Mostly weight-only on older GPUs. Blackwell has native FP4 tensor cores |

## Other things that slow you down

- **Kernel launch overhead**: each GPU kernel launch costs a few microseconds of CPU time. A decode step for a 32-layer model can launch hundreds of kernels, which is visible when each step is only ~5 ms. Fixes: CUDA graphs and fusion (lesson 4-1).
- **CPU-side work**: scheduling, tokenization, detokenization and the Python interpreter can all stall the GPU.
- **Interconnect**: with multiple GPUs, NVLink (~900 GB/s per GPU on H100) and PCIe/InfiniBand become new rooflines (lesson 3-2).
- **Wave quantization & tiling**: matrix shapes that don't split evenly across SMs leave hardware idle.

## Check yourself

**A 70B model in FP8 runs on an H200 (4.8 TB/s). What's the best-case single-sequence decode speed?**

Weights = 70 GB. Step ≥ 70 / 4800 s ≈ 14.6 ms, so about **68 tokens/s**. (A 70B model in FP8 fits on one 141 GB H200 with room left for KV cache.)

**Why does doubling batch size from 1 to 2 nearly double decode throughput, while going from 512 to 1024 barely helps?**

At B=1 we're far left of the ridge (memory-bound), so the extra token rides along nearly for free on the same weight reads. At B≈512 we're past the ridge (compute-bound), so twice the tokens takes about twice the time. KV-cache memory also limits large batches in practice.

**What's the ridge point of an A100 (312 TFLOPS BF16, 2.04 TB/s)?**

312 / 2.04 ≈ **153 FLOP/byte**. It's lower than the H100's, so A100s become compute-bound at smaller batches.

## Further reading

- [Making Deep Learning Go Brrrr From First Principles](https://horace.io/brrr_intro.html) by Horace He
- [Transformer Inference Arithmetic](https://kipp.ly/transformer-inference-arithmetic/) by kipply
- [How to Scale Your Model](https://jax-ml.github.io/scaling-book/) by Google DeepMind (the roofline and inference chapters)
- [Roofline model](https://en.wikipedia.org/wiki/Roofline_model) on Wikipedia

## Interview corner

Questions on this lesson that come up in real interviews. Answer out loud first, then open the card.

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

[All interview questions ▶](../interview.md) · [Quiz Arcade ▶](../practice.md#arcade) · [Workbook ▶](../workbook.md)
