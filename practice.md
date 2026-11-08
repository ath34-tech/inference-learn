# Practice

> Part of **InferenceLearn**, a free course on LLM inference engineering. Web version: practice.html. Whole course in one file: llms-full.txt

// training grounds

Reading isn't learning until you can use it. Test yourself in the arcade, prepare for interviews, then work through the napkin-math workbook.

[[play] 45Quiz ArcadeMultiple-choice game with lives, streaks and a hi-score. Every answer comes with an explanation.](#arcade) [[prepare] 39Interview PrepReal-style interview questions with model answers, what the interviewer is testing, and follow-ups.](interview.md) [[think] 14WorkbookNapkin-math problems with hints and step-by-step solutions, explain-it-back prompts and experiments.](workbook.md)

*(An interactive quiz game runs here on the web page. The full question bank with answers is below.)*

## Question bank (study mode)

Every arcade question with its answer and explanation. Good for revision, or for an AI tutor to quiz you from.

**Show all 45 questions with answers**

1. Why is LLM decode usually memory-bandwidth-bound?
   A. Softmax is expensive
   B. Each step reads all the weights but does very little math per byte
   C. The tokenizer runs on the GPU
   D. Tensor cores can't do matrix-vector products
   **Answer: B.** One decode step multiplies the weights by a single token vector per sequence. That's about 2 FLOPs per weight read, far below the ~300 FLOP/byte an H100 needs to be compute-bound.
2. Which metric is mostly determined by the prefill phase?
   A. Inter-token latency
   B. TPOT
   C. Time to first token
   D. Output tokens/s per user
   **Answer: C.** Prefill processes the prompt and produces the first token, so it dominates TTFT (along with queueing).
3. What happens to the sampling distribution as temperature approaches 0?
   A. It becomes greedy (argmax)
   B. It becomes uniform
   C. It's unaffected
   D. The model refuses to answer
   **Answer: A.** Dividing logits by a tiny T exaggerates the gaps, so nearly all probability goes to the top token.
4. For a BF16 linear layer with batch B ≪ hidden size, arithmetic intensity is roughly…
   A. 1, always
   B. hidden_size
   C. B²
   D. B FLOPs per byte
   **Answer: D.** FLOPs = 2·B·d², bytes ≈ 2·d² (the weights), so AI ≈ B. More tokens per weight read means more math per byte.
5. Best-case single-user decode speed of an 8B BF16 model on an H100 (3.35 TB/s)?
   A. ~20 tok/s
   B. ~200 tok/s
   C. ~2,000 tok/s
   D. ~20,000 tok/s
   **Answer: B.** 16 GB ÷ 3.35 TB/s ≈ 4.8 ms per step, so ~209 tok/s at most. Real engines get a good fraction of that.
6. The ridge point of a GPU is…
   A. Its maximum clock speed
   B. The largest batch that fits in memory
   C. Peak FLOP/s divided by memory bandwidth
   D. The temperature at which it throttles
   **Answer: C.** Below that arithmetic intensity you're memory-bound; above it, compute-bound. It's ~295 FLOP/byte on an H100.
7. A service handles 30 req/s with 5 s average latency. By Little's law, how many requests are in flight?
   A. 150
   B. 6
   C. 35
   D. 0.17
   **Answer: A.** Concurrency = throughput × latency = 30 × 5 = 150.
8. What is "goodput"?
   A. Throughput including input tokens
   B. Peak tokens/s at maximum batch
   C. GPU utilization from nvidia-smi
   D. Throughput counting only requests that met their SLOs
   **Answer: D.** Goodput measures useful work: requests delivered within latency targets.
9. An H100 costs $3/hour and serves 3,000 output tok/s. Cost per 1M output tokens?
   A. ~$3.00
   B. ~$0.28
   C. ~$0.03
   D. ~$10.80
   **Answer: B.** 3,000 × 3,600 = 10.8M tokens/hour, and $3 ÷ 10.8 ≈ $0.28 per million.
10. Why are queries not stored in the KV cache?
   A. They're too large
   B. They're recomputed by the sampler
   C. Only the current token's query is ever needed
   D. Queries are stored in the embedding table
   **Answer: C.** Each step's new query attends over all cached keys and values. Past queries are never used again.
11. Llama-3-8B: 32 layers, 8 KV heads, head dim 128, BF16. KV cache per token?
   A. 128 KiB
   B. 16 KiB
   C. 512 KiB
   D. 1 MiB
   **Answer: A.** 2 × 32 × 8 × 128 × 2 bytes = 131,072 B = 128 KiB.
12. Batching many sequences amortizes weight reads. Why doesn't it amortize KV-cache reads?
   A. The KV cache lives on the CPU
   B. KV reads are cached in L2
   C. Attention is compute-bound
   D. Each sequence has its own unique KV cache
   **Answer: D.** Weights are shared by every sequence, but every sequence's K/V is different and must be read separately.
13. Which change shrinks the KV cache without retraining the model?
   A. Switching MHA to GQA
   B. Storing the KV cache in FP8
   C. Adding more layers
   D. Using MLA
   **Answer: B.** KV-cache quantization is a serving-time choice. GQA and MLA are architectural and decided at training time.
14. The key idea of continuous batching is…
   A. Padding all sequences to the same length
   B. Waiting 5 ms to collect a full batch
   C. Making scheduling decisions every decode step, so sequences join and leave freely
   D. Running prefill and decode on different GPUs
   **Answer: C.** Iteration-level scheduling (Orca) keeps the batch full: finished sequences leave immediately and new ones join at the next step.
15. Long prompts arriving cause stream "freezes" for other users. What's the standard fix?
   A. Chunked prefill with a per-step token budget
   B. Raise temperature
   C. Disable the KV cache
   D. Use static batching
   **Answer: A.** Chunking spreads a long prefill over many steps, mixed with decodes, so inter-token latency stays smooth.
16. In vLLM, when KV memory runs out mid-generation, the default response is to…
   A. Crash with OOM
   B. Truncate the prompt
   C. Switch to CPU inference
   D. Preempt a sequence and recompute its KV later
   **Answer: D.** Preemption by recompute (or swapping to CPU) frees blocks. Frequent preemptions mean concurrency is set too high.
17. Static batching wastes GPU time mainly because…
   A. It can't use tensor cores
   B. Finished sequences sit idle until the longest one in the batch completes
   C. It requires FP32
   D. It doesn't support sampling
   **Answer: B.** The straggler problem: slots stay blocked by the slowest sequence, and new requests wait outside.
18. 64 query heads and 8 KV heads (GQA). How much smaller is the KV cache than MHA's?
   A. 2×
   B. 4×
   C. 8×
   D. 64×
   **Answer: C.** 64 ÷ 8 = 8× fewer K/V heads, so the cache is 8× smaller.
19. FlashAttention speeds up attention mainly by…
   A. Avoiding HBM reads/writes of the N×N score matrix through tiling in SRAM
   B. Approximating softmax with a polynomial
   C. Skipping distant tokens
   D. Using INT4 math
   **Answer: A.** It's IO-aware and exact. It tiles Q/K/V on-chip and uses online softmax, so the big matrix never touches HBM.
20. PagedAttention mainly solves which problem?
   A. Slow softmax
   B. KV-cache memory fragmentation and over-reservation
   C. Tokenizer speed
   D. Multi-GPU communication
   **Answer: B.** Fixed-size blocks plus a block table, like OS paging, cut KV waste from 60–80% to a few percent and allow block sharing.
21. Online softmax lets you compute softmax over a row…
   A. Only in FP32
   B. Without exponentials
   C. Approximately, with bounded error
   D. Chunk by chunk, keeping a running max and sum and rescaling
   **Answer: D.** When a larger max appears, the earlier partial sums are rescaled by exp(m_old − m_new). The result is exact.
22. At batch size 1, which scheme speeds up decode the most?
   A. W8A8 INT8
   B. W4A16 (4-bit weight-only)
   C. FP32 weights
   D. Quantizing only the LM head
   **Answer: B.** Batch 1 is purely memory-bound, so the fewest bytes per weight wins. Faster low-precision compute barely matters.
23. Why is per-group quantization (e.g. groups of 128) more accurate than per-tensor?
   A. It uses more bits per weight
   B. It skips the outliers
   C. Each group gets its own scale, so one large value only affects its own group
   D. It's done during training
   **Answer: C.** Finer scales follow local ranges. The cost is a little extra storage for the scales.
24. SmoothQuant makes W8A8 work by…
   A. Moving activation outliers into the weights with an equivalent per-channel rescale
   B. Keeping outliers in FP16
   C. Retraining the model
   D. Clipping all activations to ±1
   **Answer: A.** Divide activations and multiply weights by the same per-channel factor. The product is unchanged and both become easier to quantize.
25. Symmetric INT4 quantization has how many usable levels (with q_max = 7)?
   A. 4
   B. 8
   C. 16
   D. 15
   **Answer: D.** −7…+7 = 15 levels. (Full INT4 has 16 values, but the symmetric scheme usually leaves −8 unused.)
26. GPTQ improves on round-to-nearest by…
   A. Using larger groups
   B. Updating not-yet-quantized weights to compensate for each rounding error, using second-order info
   C. Fine-tuning with RL
   D. Quantizing activations first
   **Answer: B.** It quantizes column by column and uses Hessian information from calibration data to fix up the remaining columns.
27. With the proper acceptance rule, speculative decoding's outputs are…
   A. Distributed exactly like the target model's
   B. Slightly lower quality
   C. Identical to the draft model's
   D. Always greedy
   **Answer: A.** Rejection sampling (accept with min(1, p/q), else resample from max(0, p−q)) is lossless.
28. α = 0.8, k = 4 draft tokens. Expected tokens per target pass?
   A. 4.0
   B. 0.8
   C. ~3.36
   D. 5.0
   **Answer: C.** (1 − 0.8⁵) ÷ (1 − 0.8) = 0.672 ÷ 0.2 ≈ 3.36.
29. When does speculative decoding help least?
   A. Batch size 1, latency-sensitive chat
   B. Code editing with lots of repeated text
   C. Low-temperature structured output
   D. Very high batch sizes where the GPU is compute-bound
   **Answer: D.** When compute-bound, verifying extra tokens costs real FLOPs, and rejected drafts are wasted work.
30. Which parallelism reduces per-token latency for a model that already fits on one node?
   A. Data parallelism
   B. Tensor parallelism
   C. Pipeline parallelism
   D. More replicas
   **Answer: B.** TP splits each layer's weights, so every GPU reads only a fraction per step. PP and DP don't make a single token faster.
31. An MoE model has 671B total / 37B active parameters. Memory must hold…
   A. All 671B parameters
   B. Only the 37B active ones
   C. Only the router
   D. Half of the experts
   **Answer: A.** Any token can be routed to any expert, so all experts must be resident. Compute per token scales with the active parameters.
32. Expert parallelism mainly relies on which communication pattern?
   A. All-reduce
   B. Broadcast
   C. All-to-all
   D. No communication
   **Answer: C.** Tokens are dispatched to whichever GPUs hold their chosen experts, then combined back, both via all-to-all.
33. Your prompt template starts with the current timestamp. Effect on prefix caching?
   A. None
   B. Better hit rate
   C. Only TPOT changes
   D. The hit rate collapses, because the first block differs every request
   **Answer: D.** Block keys chain the whole prefix, so a change at the very start breaks every later match. Put dynamic content last.
34. Why does a block's cache key include its parent block's hash?
   A. To save memory
   B. Because KV depends on all preceding tokens, not just the block's own tokens
   C. For security
   D. To support FP8
   **Answer: B.** The same 16 tokens after a different prefix produce different KV. Chaining the hashes encodes the whole prefix.
35. The main benefit of prefill/decode disaggregation is…
   A. Removing interference, so TTFT and ITL can be tuned and scaled independently
   B. Smaller model weights
   C. No need for a KV cache
   D. Lower precision
   **Answer: A.** Separate pools stop prefills from stalling decodes. The cost is transferring the KV cache between them.
36. Which is the WORST autoscaling signal for an LLM server?
   A. Queue depth
   B. KV cache utilization
   C. CPU utilization
   D. p90 TTFT vs SLO
   **Answer: C.** The work happens on the GPU. CPU usage says almost nothing about inference capacity.
37. Why can pure "route to the best prefix-cache match" routing backfire?
   A. Cache hits are slower
   B. Popular prefixes overload one replica while others idle
   C. It breaks streaming
   D. It disables batching
   **Answer: B.** Good routers weigh cache overlap against current load.
38. Fusing RMSNorm with the residual add speeds things up because…
   A. It reduces FLOPs
   B. It enables FP4
   C. It improves accuracy
   D. It removes an HBM round-trip and a kernel launch for memory-bound ops
   **Answer: D.** Elementwise ops are memory-bound. Time follows bytes moved, and fusion moves fewer bytes.
39. CUDA graphs mainly help with…
   A. CPU kernel-launch overhead, especially in small-batch decode
   B. Large prefill FLOPs
   C. Model accuracy
   D. Network latency
   **Answer: A.** Hundreds of launches are replayed as one, so the GPU stops waiting on the CPU.
40. Why do engines capture CUDA graphs only for batch sizes like 1, 2, 4, 8…?
   A. GPUs only support powers of two
   B. Graphs require FP8
   C. Each graph has fixed shapes and costs memory and startup time, so engines pad to buckets
   D. Odd batch sizes are slower
   **Answer: C.** Padding to the next bucket wastes a little compute but keeps the number of graphs manageable.
41. You compute an MBU of 115%. What does that mean?
   A. Excellent kernels
   B. Your measurement or byte accounting is wrong
   C. The GPU is overclocked
   D. Prefix caching is off
   **Answer: B.** You can't beat peak bandwidth. Check the step timing, context lengths, and whether caching skipped work.
42. Your benchmark peaks at 9k tok/s, but p90 TPOT breaks the SLO above 6k tok/s. Your real capacity is…
   A. 9k tok/s
   B. 15k tok/s
   C. 7.5k tok/s
   D. ~6k tok/s
   **Answer: D.** Capacity is the highest load that still meets the SLO: the knee of the curve, not the peak.
43. Why must a server stop generating when a client disconnects?
   A. Otherwise the sequence keeps holding a batch slot, KV memory and GPU time for nobody
   B. HTTP requires it
   C. To keep logs small
   D. Tokenizers can't handle it
   **Answer: A.** Abandoned sequences take capacity away from real users.
44. Which load-testing style best reveals queueing collapse?
   A. Closed-loop with 1 user
   B. A single long request
   C. Open-loop: requests arrive at a target rate regardless of completions
   D. Running the same prompt 1000× with caching on
   **Answer: C.** Closed-loop load slows itself down when the server slows, which hides overload. Open-loop doesn't.
45. KV usage sits at 99% and preemptions are climbing. A reasonable first fix is…
   A. Raise max-num-seqs
   B. Lower max-num-seqs (or add replicas / use FP8 KV)
   C. Raise temperature
   D. Disable chunked prefill
   **Answer: B.** You're admitting more concurrent sequences than memory holds. Admit fewer, or make more room.
46. At temperature 0, an engine sometimes gives slightly different outputs for the same prompt. The most likely cause is…
   A. A bug in the tokenizer
   B. Random seeds
   C. Speculative decoding is lossy
   D. Batch composition changes floating-point reduction order
   **Answer: D.** Different batch shapes pick different kernels and summation orders, and tiny numeric differences can flip an argmax.

[[next] Interview Prep ▶39 questions from junior to staff level, plus system-design prompts.](interview.md) [[next] Workbook ▶Napkin-math problems with full worked solutions.](workbook.md)
