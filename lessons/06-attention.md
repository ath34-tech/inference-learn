# Attention Optimizations

> Lesson 2-3 · WORLD 2 Core Techniques · ~35 min · topics: MQA / GQA / MLA, FlashAttention, PagedAttention, Decode attention kernels

> previous: [2-2 Batching: Static → Continuous](../lessons/05-batching.md) · next: [2-4 Quantization](../lessons/07-quantization.md)

> Part of **InferenceLearn**, a free course on LLM inference engineering. Web version: lessons/06-attention.html. Whole course in one file: llms-full.txt

Attention is the only part of a transformer whose cost grows with context length. Three families of tricks tame it: **architecture** (fewer KV heads), **kernels** (FlashAttention) and **memory management** (PagedAttention).

#### By the end you'll be able to

- Compare MHA, MQA, GQA and MLA and their KV costs
- Explain how FlashAttention avoids materializing the N×N matrix
- Explain how PagedAttention removes KV fragmentation
- Know why decode attention needs its own kernels

#### Key vocabulary

- **[MHA](../vocabulary.md#v-mha)**: Multi-Head Attention: every query head has its own key and value head. Best quality, biggest KV cache.
- **[MQA](../vocabulary.md#v-mqa)**: Multi-Query Attention: all query heads share a single K/V head. The smallest cache, with some quality risk.
- **[GQA](../vocabulary.md#v-gqa)**: Grouped-Query Attention: groups of query heads share a K/V head (e.g. 32 Q heads, 8 KV heads), so the cache is 4× smaller than MHA.
- **[MLA](../vocabulary.md#v-mla)**: Multi-head Latent Attention (DeepSeek): caches a compressed low-rank latent instead of full per-head K/V.
- **[FlashAttention](../vocabulary.md#v-flashattention)**: An IO-aware exact attention kernel. It tiles Q/K/V in on-chip SRAM and never writes the N×N score matrix to HBM.
- **[Online softmax](../vocabulary.md#v-online-softmax)**: Computing softmax chunk by chunk with a running max and sum, rescaling earlier results as you go. This makes tiling possible.
- **[PagedAttention](../vocabulary.md#v-pagedattention)**: Stores the KV cache in fixed-size, non-contiguous blocks, like OS virtual-memory pages, to remove fragmentation.
- **[Block table](../vocabulary.md#v-block-table)**: The per-sequence map from logical KV blocks to physical memory blocks in PagedAttention.
- **[Flash-Decoding](../vocabulary.md#v-flash-decoding)**: Splits the KV sequence across thread blocks during decode, so even small batches with long contexts use the whole GPU.

[Drill all terms as flashcards ▶](../vocabulary.md#flashcards)

## The cost of attention

For a sequence of length `N`, standard attention computes `S = QKᵀ` (an N×N matrix), applies softmax, and multiplies by `V`. In prefill that's **O(N²)** compute and, done naively, **O(N²) memory traffic** for writing and reading S. In decode it's O(N) per token, but the whole KV cache must be read, so it's memory-bound.

## MHA → MQA → GQA → MLA

```
MHA: Q Q Q Q Q Q Q Q K K K K K K K K
GQA (g=2): Q Q Q Q Q Q Q Q K K
MQA: Q Q Q Q Q Q Q Q K
```

*Diagram: 8 query heads. Each K head shown has a matching V head. Fewer KV heads means a smaller KV cache*

- **MHA (multi-head attention)**: every query head has its own K and V head. Highest quality, biggest cache.
- **MQA (multi-query attention)**, Shazeer 2019: all query heads share *one* K/V head. Smallest cache, but quality can drop.
- **GQA (grouped-query attention)**, Ainslie et al. 2023: query heads are split into groups and each group shares a K/V head. Llama 3 uses 32 Q heads and 8 KV heads, so the cache is 4× smaller than MHA with nearly the same quality. This is today's standard.
- **MLA (multi-head latent attention)**, DeepSeek-V2: project K and V down to a small shared **latent vector** per token and cache only that. At inference, the up-projection matrices can be "absorbed" into the query and output projections, so attention runs directly on the latent. The cache is far smaller than MHA and quality holds up well.

> **KEY IDEA:** Fewer KV heads also **raises the arithmetic intensity** of decode attention. Every KV head loaded from HBM is reused by several query heads. With GQA group size g, decode attention does about g× more math per byte.

## FlashAttention

The insight of FlashAttention (Dao et al., 2022) is that attention is slow not because of FLOPs but because of **memory traffic**: writing the N×N score matrix to HBM, reading it back for softmax, writing it again… FlashAttention is **IO-aware**:

1. **Tile** Q, K and V into blocks that fit in on-chip SRAM.
2. For each Q block, loop over K/V blocks, computing partial scores *in SRAM*.
3. Use **online softmax** to combine partial results without ever seeing a full row.
4. Write only the final output to HBM. The N×N matrix never exists in HBM.

The result is **exact** attention (not an approximation) with O(N) memory and several-fold speedups. Later versions refine the idea:

- **FlashAttention-2**: better work partitioning across warps and thread blocks, and parallelism over the sequence length.
- **FlashAttention-3**: built for Hopper (H100). Uses asynchronous TMA loads, warp specialization, softmax overlapped with matmuls, and FP8 support.

### Online softmax in 10 lines

```
# process scores for one query row in chunks, never holding the whole row
m = -inf      # running max
l = 0.0       # running sum of exp(score - m)
acc = zeros(d)  # running weighted sum of V
for K_blk, V_blk in blocks:
    s = q @ K_blk.T / sqrt(d)            # partial scores
    m_new = max(m, s.max())
    scale = exp(m - m_new)               # rescale old stats to the new max
    p = exp(s - m_new)
    l = l * scale + p.sum()
    acc = acc * scale + p @ V_blk
    m = m_new
out = acc / l
```

Subtracting the running max keeps `exp` numerically stable. Rescaling by `exp(m − m_new)` fixes up everything accumulated so far whenever a larger score shows up.

## PagedAttention

Early serving systems reserved a **contiguous** KV buffer per request, sized for the maximum possible length. That caused:

- **Internal fragmentation**: memory reserved for 4k tokens but only 300 used.
- **External fragmentation**: free memory scattered in gaps too small to use.

The vLLM paper (Kwon et al., SOSP 2023) measured that existing systems wasted **60–80%** of KV memory this way. **PagedAttention** borrows the idea of virtual memory from operating systems:

```
seq A table: → blk 7 → blk 2 → blk 9
seq B table: → blk 4 → blk 0
GPU blocks: 0 1 2 3 4 5 6 7 8 9
```

*Diagram: each block holds KV for a fixed number of tokens (e.g. 16). Logical order comes from the block table, not physical position*

- KV memory is split into fixed-size **blocks** (e.g. 16 tokens each).
- Each sequence has a **block table** that maps logical blocks to physical blocks, like a page table.
- Blocks are allocated on demand as the sequence grows, so waste is limited to the last, partly filled block (under 4% in the paper).
- The attention kernel follows the block table to gather K/V.
- Blocks can be **shared** with copy-on-write. That's useful for parallel sampling, beam search and prefix caching (lesson 3-3).

## Decode attention kernels

In decode, each sequence has a single query token. A kernel that parallelizes only over (batch × heads) can leave most SMs idle when the batch is small and the context is long. **Flash-Decoding** (split-KV) also splits the KV sequence across thread blocks, computes partial results in parallel, and merges them using the online-softmax statistics.

Libraries such as **FlashInfer** provide optimized kernels for paged KV, ragged batches, prefix sharing (cascade attention) and many head layouts. Several serving engines use it.

## Beyond full attention

- **Sliding-window attention**: each token attends only to the last W tokens, so KV per layer is capped at W. Some models alternate local and global layers.
- **Sparse / linear attention, state-space models** (e.g. Mamba-style layers): sub-quadratic alternatives. Hybrid models mix them with a few full-attention layers, which changes the KV math completely.

## Check yourself

**A model has 64 query heads and 8 KV heads. How much smaller is its KV cache than MHA's?**

64 / 8 = **8× smaller**.

**Is FlashAttention an approximation of attention?**

No. It computes exactly the same result (up to floating-point rounding). It only changes the order of operations and where intermediate data lives.

**Why does PagedAttention make parallel sampling (n=4 completions of one prompt) cheaper?**

All four sequences point their block tables at the *same* physical blocks for the prompt. New blocks are allocated only when they diverge (copy-on-write), so the prompt's KV is stored once instead of four times.

## Further reading

- [FlashAttention](https://arxiv.org/abs/2205.14135), [FlashAttention-2](https://arxiv.org/abs/2307.08691), [FlashAttention-3](https://arxiv.org/abs/2407.08608)
- [Efficient Memory Management for LLM Serving with PagedAttention](https://arxiv.org/abs/2309.06180) (vLLM)
- [GQA](https://arxiv.org/abs/2305.13245) · [MQA](https://arxiv.org/abs/1911.02150) · [DeepSeek-V2 (MLA)](https://arxiv.org/abs/2405.04434)
- [FlashInfer](https://github.com/flashinfer-ai/flashinfer) kernel library

## Interview corner

Questions on this lesson that come up in real interviews. Answer out loud first, then open the card.

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

[All interview questions ▶](../interview.md) · [Quiz Arcade ▶](../practice.md#arcade) · [Workbook ▶](../workbook.md)
