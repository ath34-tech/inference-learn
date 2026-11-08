# Workbook

> Part of **InferenceLearn**, a free course on LLM inference engineering. Web version: workbook.html. Whole course in one file: llms-full.txt

// grind zone

Deep understanding comes from working things out yourself. Try each problem on paper first, then open the hint, and only then the solution.

### Your cheat sheet

| Forward-pass FLOPs | ≈ 2 × params per token |
|---|---|
| Decode step (memory-bound) | ≥ (weight bytes + KV bytes) ÷ bandwidth |
| Prefill time | ≥ 2 × params × tokens ÷ (peak FLOP/s × MFU) |
| KV per token | 2 × layers × KV heads × head dim × bytes |
| Linear-layer intensity (BF16) | ≈ batch size, in FLOP/byte |
| Little's law | concurrency = throughput × latency |
| Speculative decoding | E[tokens] = (1 − α^(k+1)) ÷ (1 − α) |
| H100 SXM (approx.) | 989 TFLOPS BF16 dense · 3.35 TB/s · 80 GB · ridge ≈ 295 |

## Part 1: Napkin-math problems

[World 1]

[warm-up]

### P1 · How long is that answer?

**Given:** the model writes a 600-word answer. 1 token ≈ 0.75 words. TTFT = 400 ms, TPOT = 25 ms.  
**Find:** output tokens and total response time.

**Hint**

Convert words to tokens first, then use E2E = TTFT + TPOT × (tokens − 1).

**Step-by-step solution**

1. Tokens = 600 ÷ 0.75 = **800 tokens**.
2. Decode time = 25 ms × 799 ≈ 20.0 s.
3. E2E = 0.4 + 20.0 ≈ **20.4 s**.

**Takeaway:** long answers are almost entirely decode time. Shaving TPOT matters far more than TTFT here.

[World 1]

[roofline]

### P2 · Speed of light on an A100

**Given:** Llama-3-8B, A100 80GB (2.04 TB/s), batch 1.  
**Find:** max decode tok/s in BF16, and in INT4 weight-only.

**Hint**

At batch 1, step time ≈ weight bytes ÷ bandwidth. How many bytes per parameter in each format?

**Step-by-step solution**

1. BF16: 8e9 × 2 = 16 GB → 16 ÷ 2040 s ≈ 7.8 ms → **~128 tok/s**.
2. INT4: 8e9 × 0.5 = 4 GB → ≈ 1.96 ms → **~510 tok/s** (in theory).
3. Reality: dequantization overhead, KV reads and launch overheads mean INT4 usually gives ~2–3×, not 4×.

**Takeaway:** at batch 1, bytes per weight is the speed dial.

[World 1]

[roofline]

### P3 · When does decode become compute-bound?

**Given:** H100: 989 TFLOPS BF16, ~1979 TFLOPS FP8 (dense), 3.35 TB/s.  
**Find:** the batch size where linear layers become compute-bound for (a) BF16 weights and compute, (b) FP8 weights and compute.

**Hint**

Intensity = 2·P·B ÷ (P × bytes per param). Set it equal to the ridge point for each precision.

**Step-by-step solution**

1. (a) Ridge = 989 ÷ 3.35 ≈ 295. Intensity = 2B ÷ 2 = B, so **B ≈ 295**.
2. (b) Ridge = 1979 ÷ 3.35 ≈ 591. Intensity = 2B ÷ 1 = 2B, so 2B = 591 and **B ≈ 295** again!
3. FP8 doubles both the FLOPs and the work per byte, so the crossover batch stays the same, but every step is ~2× faster.

**Takeaway:** real batches are usually smaller than ~300 (KV memory limits them), so decode is memory-bound in most deployments.

[World 1]

[prefill]

### P4 · TTFT for a long prompt

**Given:** 70B dense model, 8,192-token prompt, 8×H100 with TP=8, 50% MFU.  
**Find:** approximate prefill time (ignore the extra attention FLOPs).

**Hint**

Total FLOPs = 2 × params × tokens. Divide by the cluster's effective FLOP/s.

**Step-by-step solution**

1. FLOPs = 2 × 70e9 × 8192 ≈ 1.15e15.
2. Effective compute = 8 × 989e12 × 0.5 ≈ 3.96e15 FLOP/s.
3. Time ≈ 1.15e15 ÷ 3.96e15 ≈ **0.29 s**. Attention adds more at long context.

**Takeaway:** prefill scales with prompt length × params. Prefix caching avoids paying it twice.

[World 2]

[KV cache]

### P5 · The price of long context

**Given:** Llama-3-70B: 80 layers, 8 KV heads, head dim 128, BF16 KV.  
**Find:** KV per token, and per 32k-token sequence. What does FP8 KV change?

**Hint**

2 × layers × KV heads × head dim × bytes.

**Step-by-step solution**

1. Per token = 2 × 80 × 8 × 128 × 2 = 327,680 B = **320 KiB**.
2. Per 32k sequence = 320 KiB × 32,768 = **10 GiB**.
3. FP8 KV halves it to 5 GiB, so twice as many concurrent long-context users fit.

**Takeaway:** one long-context user can need as much memory as a small model's entire weights.

[World 2]

[memory budget]

### P6 · Will it fit on a 24 GB card?

**Given:** Qwen2.5-7B (~7.6B params; 28 layers, 4 KV heads, head dim 128) on a 24 GB GPU, 90% usable, ~1.5 GB workspace.  
**Find:** how many KV tokens fit with BF16 weights vs INT4 weights (~0.5 B/param)?

**Hint**

KV room = usable − weights − workspace. Then divide by KV bytes per token.

**Step-by-step solution**

1. Usable = 21.6 GB. KV per token = 2 × 28 × 4 × 128 × 2 = 57,344 B = 56 KiB.
2. BF16 weights ≈ 15.2 GB → room = 21.6 − 15.2 − 1.5 = 4.9 GB → ≈ **85k tokens** (about 10 × 8k-token chats).
3. INT4 weights ≈ 3.8 GB → room ≈ 16.3 GB → ≈ **284k tokens**, about 3.3× more concurrent users.

**Takeaway:** on small GPUs, quantizing the weights mainly buys KV capacity, which means throughput.

[World 2]

[batching]

### P7 · Static vs continuous by hand

**Given:** 2 slots. Requests arrive in order R0..R3 needing 8, 2, 2 and 8 decode steps.  
**Find:** total steps and slot utilization for static batching (pairs in order) vs continuous batching (fill the earliest-free slot).

**Hint**

Draw a 2-row grid and fill it in. Useful work = 8+2+2+8 = 20 slot-steps.

**Step-by-step solution**

1. Static: batch {R0,R1} takes max(8,2) = 8 steps, batch {R2,R3} takes 8 → **16 steps**, utilization 20 ÷ 32 = 62.5%.
2. Continuous: slot A runs R0 (0–8). Slot B runs R1 (0–2), R2 (2–4), R3 (4–12) → **12 steps**, utilization 20 ÷ 24 = 83%.
3. A perfect packing would take 10 steps. Arrival order still matters, which is why some schedulers consider predicted lengths.

**Takeaway:** continuous batching removes the straggler waste. Try more cases in the [simulator](lessons/05-batching.md).

[World 2]

[quantization]

### P8 · Quantize four numbers by hand

**Given:** weights [0.10, −0.30, 0.05, 2.40], symmetric INT4 (q_max = 7).  
**Find:** the dequantized values with (a) one per-tensor scale, (b) per-group scales with groups of 2.

**Hint**

s = max|x| ÷ 7, q = round(x ÷ s), x̂ = q × s.

**Step-by-step solution**

1. (a) s = 2.4 ÷ 7 ≈ 0.343. q = [0, −1, 0, 7], so x̂ = [**0**, −0.343, **0**, 2.40]. Two weights wiped out!
2. (b) Group 1 [0.10, −0.30]: s = 0.3 ÷ 7 ≈ 0.0429, q = [2, −7], x̂ = [0.086, −0.300].
3. Group 2 [0.05, 2.40]: s ≈ 0.343, q = [0, 7], x̂ = [0, 2.40]. Only the weight sharing a group with the outlier suffers.

**Takeaway:** outliers stretch the scale, and smaller groups contain the damage. See it live in the [quantization playground](lessons/07-quantization.md).

[World 3]

[speculative]

### P9 · Choosing the draft length

**Given:** acceptance rate α = 0.7. Each draft token costs 0.1 of a target forward pass.  
**Find:** the ideal speedup for k = 3 and k = 6. Which is better?

**Hint**

Speedup = E[tokens] ÷ (1 + k × 0.1).

**Step-by-step solution**

1. k=3: E = (1 − 0.7⁴) ÷ 0.3 = 0.7599 ÷ 0.3 ≈ 2.53. Cost = 1.3. Speedup ≈ **1.95×**.
2. k=6: E = (1 − 0.7⁷) ÷ 0.3 = 0.9176 ÷ 0.3 ≈ 3.06. Cost = 1.6. Speedup ≈ **1.91×**.
3. k=3 wins slightly. Tokens far down the chain are rarely accepted but always cost drafting time.

**Takeaway:** the best k grows with α and shrinks with draft cost.

[World 3]

[parallelism]

### P10 · Does TP=8 halve latency vs TP=4?

**Given:** 70B BF16, decode at small batch on H100s. 80 layers × 2 all-reduces per layer, ~20 µs each (latency-dominated).  
**Find:** step time at TP=4 and TP=8.

**Hint**

The number of all-reduces per step doesn't depend on the TP size. Only the weight bytes per GPU do.

**Step-by-step solution**

1. Communication = 160 × 20 µs = 3.2 ms per step, in both cases.
2. TP=4: 140 GB ÷ 4 = 35 GB per GPU → 10.4 ms → total ≈ **13.6 ms**.
3. TP=8: 17.5 GB → 5.2 ms → total ≈ **8.4 ms**. That's 1.6× faster, not 2×.

**Takeaway:** fixed communication latency limits TP scaling. Hence fused, low-latency all-reduce kernels.

[World 3]

[prefix caching]

### P11 · What is prefix caching worth?

**Given:** 8B model, 3,000-token shared system prompt + 500-token user part, 1M requests/day, 90% cache hits, H100 at 50% MFU, $3/GPU-hour.  
**Find:** GPU-hours of prefill saved per day.

**Hint**

Saved tokens = hit rate × shared tokens × requests. FLOPs = 2 × params × tokens.

**Step-by-step solution**

1. Saved tokens = 0.9 × 3,000 × 1M = 2.7e9 tokens/day.
2. FLOPs = 2 × 8e9 × 2.7e9 ≈ 4.3e19.
3. Effective H100 = 0.5 × 989e12 ≈ 4.9e14 FLOP/s → ≈ 87,000 s ≈ **24 GPU-hours/day** (≈ $72/day), plus much lower TTFT (prefill 500 tokens instead of 3,500).

**Takeaway:** stable prompt prefixes are free money. Design templates for caching.

[World 3]

[disaggregation]

### P12 · Is KV transfer a deal-breaker?

**Given:** 8B model (128 KiB/token KV), 4,096-token prompt. Network 25 GB/s (≈200 Gb/s) or 100 GB/s. Prefill on one H100 at 50% MFU.  
**Find:** KV transfer time vs prefill time.

**Hint**

KV bytes = tokens × per-token size.

**Step-by-step solution**

1. KV = 4096 × 128 KiB = 512 MiB ≈ 0.54 GB.
2. Transfer: ≈ 21 ms at 25 GB/s, ≈ 5.4 ms at 100 GB/s.
3. Prefill ≈ 2 × 8e9 × 4096 ÷ (0.5 × 989e12) ≈ 132 ms. So transfer adds ~16% at 25 GB/s and ~4% at 100 GB/s, and layer-by-layer streaming can hide much of it.

**Takeaway:** disaggregation needs fast links. Small models with long prompts make the trade-off the tightest.

[World 4]

[profiling]

### P13 · Compute your MBU

**Given:** 8B BF16 model, batch 16, average context 4k tokens, measured decode step 8 ms on H100.  
**Find:** bytes per step, achieved bandwidth, and MBU.

**Hint**

Bytes = weights + KV for every sequence.

**Step-by-step solution**

1. Weights = 16 GB. KV = 16 × 4096 × 128 KiB = 8 GiB ≈ 8.6 GB. Total ≈ 24.6 GB.
2. Achieved = 24.6 GB ÷ 8 ms ≈ 3.07 TB/s.
3. MBU = 3.07 ÷ 3.35 ≈ **92%**. Suspiciously high: double-check the context lengths and timing before celebrating.

**Takeaway:** MBU tells you how much speed is left. Over ~90% usually means a measurement mistake.

[World 4]

[capacity]

### P14 · Fleet size and cost

**Given:** 5M requests/day, 400 output tokens each, peak traffic = 3× average, benchmarked knee = 3,000 output tok/s per GPU, 30% headroom, $3/GPU-hour.  
**Find:** peak GPU count, and $/1M tokens for a static fleet vs perfect autoscaling.

**Hint**

Average tok/s = requests × tokens ÷ 86,400. Size for the peak, then compare with what you actually use.

**Step-by-step solution**

1. Average = 5e6 × 400 ÷ 86,400 ≈ 23,100 tok/s. Peak ≈ 69,400 tok/s.
2. Peak GPUs = 69,400 ÷ 3,000 ≈ 23.1, plus 30% → **~31 GPUs**.
3. Static fleet: 31 × $3 × 24 = $2,232/day for 2,000M tokens → **$1.12 per 1M**.
4. Autoscaling at average load: 23,100 ÷ 3,000 × 1.3 ≈ 10 GPUs → ~$720/day → **~$0.36 per 1M**.

**Takeaway:** utilization can matter as much as kernel speed. Idle GPUs are the most expensive ones.

## Part 2: Explain it back (Feynman checks)

If you can explain something simply, you understand it. Answer each one out loud in under a minute, without notes. Then check against the lesson, or ask an AI tutor to grade you.

1. Explain to a non-engineer why a chatbot's first word takes longer than each word after it.
2. Explain why making the GPU's math twice as fast doesn't make single-user chat twice as fast.
3. Explain the KV cache using the analogy of taking notes in a meeting.
4. Explain why serving 100 users at once can cost almost the same per step as serving 1.
5. Explain continuous batching using a restaurant, a bus or a sushi conveyor belt.
6. Explain how FlashAttention can be faster while doing *more* arithmetic (recomputation) than naive attention.
7. Explain why an outlier ruins INT8 quantization, using a ruler analogy.
8. Explain why speculative decoding can't change the model's answers.
9. Explain why a 600B-parameter MoE model can run faster per token than a 70B dense model, but still need more GPUs.
10. Explain to a product manager why putting the user's name at the start of the system prompt costs money.

## Part 3: Hands-on experiments

Small experiments that turn the theory into numbers. A single consumer or cloud GPU is enough for most of them.

### E1 · Measure prefill vs decode

Using the naive loop from [lesson 1-1](lessons/01-how-inference-works.md) with a small model, time the first token vs the rest for prompts of 100, 1,000 and 4,000 tokens. Plot TTFT against prompt length. **Expect:** TTFT grows with the prompt; per-token decode time stays nearly flat (with a KV cache).

### E2 · Find your batching curve

Run decode at batch 1, 2, 4 … 256 and record step time. Plot system tok/s and per-user tok/s. **Expect:** step time barely moves at first (memory-bound), then rises linearly once you pass the ridge or hit attention costs.

### E3 · Benchmark a real server

```
# terminal 1: start a server
vllm serve Qwen/Qwen2.5-1.5B-Instruct --max-model-len 8192

# terminal 2: load-test it (see `vllm bench serve --help` for dataset options)
vllm bench serve --model Qwen/Qwen2.5-1.5B-Instruct --dataset-name random \
  --random-input-len 1024 --random-output-len 256 --num-prompts 500 --request-rate 8
```

Sweep `--request-rate` and record TTFT/TPOT p50 and p99. Find the knee. *(CLI flags change between versions, so check `--help`.)*

### E4 · Prefix caching A/B

Send requests sharing a 2,000-token system prompt, with prefix caching on and off. Compare TTFT and the hit-rate metric on `/metrics`. Then put a timestamp at the start of the prompt and watch the hit rate collapse.

### E5 · Quantize and evaluate

Serve the same model in BF16, FP8 (if your GPU supports it) and an AWQ/GPTQ INT4 checkpoint. Measure throughput at batch 1 and batch 64, and score a small eval (e.g. 100 GSM8K-style math questions). Fill in a table of speed vs accuracy.

### E6 · Profile a decode step

```
nsys profile -o trace --trace=cuda,nvtx python my_decode_loop.py
# open trace.nsys-rep in Nsight Systems: look for gaps between kernels
```

Compare eager mode, `torch.compile`, and CUDA graphs. Count kernels per step and the idle gaps.

[[play] Quiz Arcade ▶Check what stuck.](practice.md#arcade) [[prepare] Interview Prep ▶Use these numbers in interview answers.](interview.md)
