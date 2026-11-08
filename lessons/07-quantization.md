# Quantization

> Lesson 2-4 · WORLD 2 Core Techniques · ~35 min · topics: Scales & zero points, Granularity, Outliers, GPTQ / AWQ / FP8

> previous: [2-3 Attention Optimizations](../lessons/06-attention.md) · next: [3-1 Speculative Decoding](../lessons/08-speculative-decoding.md)

> Part of **InferenceLearn**, a free course on LLM inference engineering. Web version: lessons/07-quantization.html. Whole course in one file: llms-full.txt

Decode speed is roughly *bytes moved ÷ bandwidth*. Quantization stores numbers in fewer bits, so it directly cuts the bytes, and on modern GPUs it unlocks faster low-precision tensor cores too. Done well, you lose almost no quality.

#### By the end you'll be able to

- Quantize and dequantize a tensor by hand, with scales and zero points
- Choose between weight-only and weight + activation quantization
- Explain the outlier problem and how SmoothQuant, GPTQ and AWQ handle it
- Evaluate a quantized model properly

#### Key vocabulary

- **[Quantization](../vocabulary.md#v-quantization)**: Storing numbers in fewer bits (8, 4…) to cut memory and bandwidth. Done well, with little quality loss.
- **[Scale & zero point](../vocabulary.md#v-scale-zero-point)**: The mapping between real and integer values: q = round(x/s) + z, x̂ = s·(q − z). Symmetric quantization uses z = 0.
- **[Per-group quantization](../vocabulary.md#v-per-group-quantization)**: One scale per small group of weights (e.g. 128), so an outlier only hurts its own group.
- **[W4A16 / W8A8](../vocabulary.md#v-w4a16-w8a8)**: Notation for weight bits / activation bits. W4A16 = INT4 weights with BF16 math (weight-only). W8A8 = 8-bit weights and activations.
- **[FP8 (E4M3 / E5M2)](../vocabulary.md#v-fp8-e4m3-e5m2)**: 8-bit floating-point formats with native tensor-core support on Hopper and newer. A common near-lossless choice.
- **[Outliers](../vocabulary.md#v-outliers)**: A few activation channels with values 10–100× larger than the rest. They wreck naive INT8 scaling.
- **[SmoothQuant](../vocabulary.md#v-smoothquant)**: Moves activation outliers into the weights with an equivalent per-channel rescaling, enabling W8A8.
- **[GPTQ](../vocabulary.md#v-gptq)**: Post-training quantization that uses second-order (Hessian) information to compensate for each rounding error column by column.
- **[AWQ](../vocabulary.md#v-awq)**: Activation-aware Weight Quantization: scales up salient weight channels (those multiplied by large activations) before quantizing.
- **[Calibration set](../vocabulary.md#v-calibration-set)**: A few hundred sample inputs used to measure activation ranges and statistics during post-training quantization.

[Drill all terms as flashcards ▶](../vocabulary.md#flashcards)

## Why quantize?

- **Faster decode**: INT4 weights are 4× fewer bytes than BF16, so memory-bound steps get up to ~4× faster (less in practice because of dequantization overhead).
- **Fit bigger models**: a 70B model is 140 GB in BF16 but 70 GB in FP8, so it fits on one large GPU.
- **More KV room**: smaller weights leave more memory for the cache, which means bigger batches and higher throughput.
- **Faster compute**: FP8 tensor cores on Hopper do ~2× the FLOPs of BF16, which helps compute-bound prefill.

## The basics: scale & zero point

**Affine (asymmetric)** quantization maps a real value x to a b-bit integer q:

> q = clamp( round(x / **s**) + **z**, q_min, q_max ) · x̂ = **s** · (q − **z**)

**Symmetric** quantization fixes z = 0 and uses s = max|x| / q_max (127 for INT8). It's simpler and faster, and the usual choice for weights.

```
# symmetric per-output-channel INT8 weight quantization
def quantize_int8(W):                       # W: [out_features, in_features]
    scale = W.abs().amax(dim=1, keepdim=True) / 127.0  # one scale per row
    q = torch.clamp(torch.round(W / scale), -127, 127).to(torch.int8)
    return q, scale

def dequantize(q, scale):
    return q.float() * scale

W = torch.randn(4096, 4096)
q, s = quantize_int8(W)
print((W - dequantize(q, s)).abs().mean())     # small reconstruction error
```

> **Interactive lab: Quantization playground.** 16 real-looking weights. Squeeze them into fewer bits and see what survives. Dashed outlines are the originals and filled bars are what comes back after quantize → dequantize. Try this: **4-bit + add an outlier**, then switch to **per-group**. *(Use it on the web page. The formulas it uses are explained in the text.)*

## Granularity

One scale for a whole tensor is cheap but coarse: a single large value blows up the scale and wipes out precision for everything else. Finer granularity means more scales and better accuracy:

| Granularity | One scale per… | Typical use |
|---|---|---|
| Per-tensor | whole matrix | FP8 weights/activations (FP8's exponent handles range) |
| Per-channel | output row of W | INT8 weights |
| Per-token | row of activations | INT8/FP8 activations computed on the fly |
| Per-group / block | e.g. 128 consecutive weights | INT4 weights (GPTQ/AWQ), MX formats (block of 32) |

## What to quantize

| Scheme | Example | Speeds up | Notes |
|---|---|---|---|
| **Weight-only** | W4A16, W8A16 | Memory-bound decode | Weights are dequantized to BF16 inside the GEMM kernel. Doesn't speed up compute-bound prefill |
| **Weight + activation** | W8A8 (INT8 / FP8) | Decode *and* prefill | Uses low-precision tensor cores. Activations need scaling (static or dynamic) |
| **KV cache** | FP8 KV, INT8/INT4 KV | Long-context decode, capacity | Halves KV memory and bandwidth. Usually little quality impact for FP8 |
| **Low-bit floats** | FP4 (NVFP4, MXFP4) | Everything, on Blackwell | Native FP4 tensor cores with block scales |

## The outlier problem

Weights are well behaved, roughly Gaussian. **Activations** in LLMs are not: a few channels carry values 10–100× larger than the rest (LLM.int8() documented this for models of ~6.7B parameters and up). With per-tensor INT8, those outliers force a huge scale and the normal values all round to zero.

- **LLM.int8()**: keep outlier channels in FP16 and do the rest in INT8 (mixed-precision decomposition).
- **SmoothQuant**: divide activations by a per-channel factor and multiply the weights by it, a mathematically equivalent transform. This moves the difficulty from activations, which are hard to quantize, into weights, which are easy, and enables W8A8.
- **FP8**: an exponent gives FP8 a much wider dynamic range than INT8, so it handles outliers better. That's a big reason FP8 W8A8 is a popular "safe default" on Hopper.
- **Rotations** (QuaRot, SpinQuant): multiply by orthogonal matrices to spread outliers across channels before quantizing.

## Smarter weight quantization: GPTQ & AWQ

Rounding every weight to the nearest grid point ("round-to-nearest", RTN) works fine at 8 bits but hurts at 4 bits. Two popular post-training quantization (PTQ) methods do better with a small calibration set:

- **GPTQ** (Frantar et al.): quantizes weights one column at a time. It uses second-order (Hessian) information from calibration activations to *update the remaining weights* and compensate for each rounding error.
- **AWQ** (Lin et al.): observes that a small fraction of weight channels matter most, namely those multiplied by large activations. It scales those salient channels up before quantizing to protect them. No backprop and no reordering needed.

> **ECOSYSTEM:** You'll meet these as formats: **GGUF** (llama.cpp's file format, with "k-quant" types like Q4_K_M), GPTQ/AWQ checkpoints on the Hugging Face Hub, **compressed-tensors** / LLM Compressor for vLLM, and NVIDIA's **ModelOpt** for TensorRT-LLM. Kernels such as Marlin make W4A16 fast on NVIDIA GPUs.

## Evaluating a quantized model

- **Perplexity** on held-out text is a quick sanity check, but it can hide real regressions.
- Run **task benchmarks** that match your use case: MMLU-style knowledge, GSM8K-style math, code, long-context retrieval, plus your own eval set.
- Compare **generations side by side** on real prompts. Math, code and multilingual tasks usually break first.
- Measure **speed for real**. Weight-only INT4 can be *slower* than BF16 at large batch sizes, where you're compute-bound and pay the dequantization cost.

> **RULE OF THUMB:** FP8 (W8A8) is close to lossless for most large models. INT4 weight-only with GPTQ/AWQ is usually acceptable for chat. Below 4 bits, quality drops quickly, especially for smaller models. Always check on *your* task.

## Check yourself

**You serve at batch size 1 on a laptop GPU. W4A16 or W8A8, and why?**

**W4A16**. At batch 1 you're completely memory-bound, so the fewest bytes per weight wins. Faster low-precision compute (W8A8) wouldn't help much.

**Why is per-group quantization (group of 128) more accurate than per-channel?**

Each group of 128 weights gets its own scale, so one large weight only affects the precision of its own group, not the whole row. The cost is storing more scales: one FP16 value per 128 weights adds ~0.125 bits per weight.

**What does SmoothQuant actually change in the model?**

It rescales each input channel: activations are divided by sⱼ and the matching weight rows are multiplied by sⱼ. The product X·W is mathematically unchanged, but both tensors become easier to quantize.

## Further reading

- [LLM.int8()](https://arxiv.org/abs/2208.07339) · [SmoothQuant](https://arxiv.org/abs/2211.10438)
- [GPTQ](https://arxiv.org/abs/2210.17323) · [AWQ](https://arxiv.org/abs/2306.00978)
- [FP8 Formats for Deep Learning](https://arxiv.org/abs/2209.05433)
- [Hugging Face quantization overview](https://huggingface.co/docs/transformers/main/en/quantization/overview)

## Interview corner

Questions on this lesson that come up in real interviews. Answer out loud first, then open the card.

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

[All interview questions ▶](../interview.md) · [Quiz Arcade ▶](../practice.md#arcade) · [Workbook ▶](../workbook.md)
