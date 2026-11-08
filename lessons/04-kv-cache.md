# The KV Cache

> Lesson 2-1 · WORLD 2 Core Techniques · ~30 min · topics: Why caching works, Sizing formula, Memory budget, Reduction techniques

> previous: [1-3 Metrics: Latency, Throughput & Cost](../lessons/03-metrics.md) · next: [2-2 Batching: Static → Continuous](../lessons/05-batching.md)

> Part of **InferenceLearn**, a free course on LLM inference engineering. Web version: lessons/04-kv-cache.html. Whole course in one file: llms-full.txt

The KV cache is the single most important data structure in LLM serving. It turns generation from quadratic recomputation into linear work. It's also the main consumer of GPU memory, and so it limits how many users you can serve at once.

#### By the end you'll be able to

- Explain why keys and values can be cached but queries can't
- Compute KV cache size for any model with one formula
- Work out how many concurrent sequences fit on a GPU
- List the main techniques that shrink or manage the cache

#### Key vocabulary

- **[KV cache](../vocabulary.md#v-kv-cache)**: Stored key and value vectors for all previous tokens, so each decode step only computes the new token's K/V. Size = 2 × layers × KV heads × head dim × bytes, per token.
- **[KV head](../vocabulary.md#v-kv-head)**: A key/value head. With GQA, several query heads share one, so fewer KV heads means a smaller cache.
- **[Context length](../vocabulary.md#v-context-length)**: The number of tokens (prompt plus output) a sequence holds. The KV cache grows linearly with it.
- **[GQA](../vocabulary.md#v-gqa)**: Grouped-Query Attention: groups of query heads share a K/V head (e.g. 32 Q heads, 8 KV heads), so the cache is 4× smaller than MHA.
- **[KV cache quantization](../vocabulary.md#v-kv-cache-quantization)**: Storing K/V in FP8, INT8 or INT4 to halve or quarter cache memory and bandwidth.
- **[Preemption](../vocabulary.md#v-preemption)**: Evicting a running sequence when KV memory runs out, then recomputing or swapping its cache in later.

[Drill all terms as flashcards ▶](../vocabulary.md#flashcards)

## Why caching works

In self-attention, each token `i` produces three vectors from its hidden state: a query `qᵢ`, a key `kᵢ` and a value `vᵢ`. Token `t`'s output is:

> outₜ = Σᵢ≤ₜ softmax( qₜ · kᵢ / √d )ᵢ · vᵢ

With **causal masking**, token `i`'s key and value depend only on tokens ≤ `i`. Once computed, they **never change**, no matter how many tokens come after. So instead of recomputing `k₁…kₜ` and `v₁…vₜ` at every step, we store them, and each decode step computes K and V for *only the new token* and appends them.

```
step t=5: K K K K K □ □
: V V V V V □ □
step t=6: K K K K K K □
: V V V V V V □
```

*Diagram: pink = computed this step · blue/green = read from cache · per layer, per KV head*

The **query** isn't cached because it's only needed for the token currently being produced. Old queries are never used again.

```
# minimal single-layer KV cache (shapes: [batch, heads, seq, head_dim])
class KVCache:
    def __init__(self):
        self.k = None; self.v = None

    def update(self, k_new, v_new):
        if self.k is None:
            self.k, self.v = k_new, v_new
        else:
            self.k = torch.cat([self.k, k_new], dim=2)   # append along seq
            self.v = torch.cat([self.v, v_new], dim=2)
        return self.k, self.v

def attention_step(q, k_new, v_new, cache):
    k, v = cache.update(k_new, v_new)                # q: [B, H, 1, D] during decode
    scores = (q @ k.transpose(-1, -2)) / k.shape[-1] ** 0.5
    return torch.softmax(scores, dim=-1) @ v
```

> **WATCH OUT:** `torch.cat` on every step reallocates and copies the whole cache, which is O(n²) memory traffic. Real engines **preallocate** memory and write in place. The best ones use a **paged** layout (lesson 2-3).

## How big is it?

For every token, every layer stores one key and one value vector per **KV head**:

> KV bytes per token = **2** × n_layers × n_kv_heads × head_dim × bytes_per_element

The leading 2 is for K and V. Multiply by sequence length and batch size for the total.

| Model | Layers | KV heads | Head dim | Per token (BF16) | 8k-token sequence |
|---|---|---|---|---|---|
| Llama-2-7B (MHA) | 32 | 32 | 128 | 512 KiB | 4 GiB |
| Llama-3-8B (GQA) | 32 | 8 | 128 | 128 KiB | 1 GiB |
| Llama-3-70B (GQA) | 80 | 8 | 128 | 320 KiB | 2.5 GiB |

Llama-3-8B's grouped-query attention (8 KV heads shared by 32 query heads) makes its cache **4× smaller** than Llama-2-7B's, even though the two models are about the same size.

> **Interactive lab: KV cache lab: will it fit?** Pick a model and a GPU, then crank up the context length and the number of users. The memory bar fills up, and if you overdo it you get the classic **out-of-memory** 💥. Then try the fixes: FP8 KV, a GQA model, a shorter context. *(Use it on the web page. The formulas it uses are explained in the text.)* Ignores small per-block overhead. Check a model's `config.json` for `num_hidden_layers`, `num_key_value_heads` and `head_dim` (or hidden_size / num_attention_heads).

## The memory budget

On a serving GPU, memory splits roughly like this:

> KV pool ≈ GPU memory × utilization target − **weights** − activation workspace − CUDA graphs/overhead

**Worked example:** Llama-3-8B (BF16) on an 80 GB H100 at a 90% utilization target leaves 72 GB, minus 16 GB of weights and a few GB of workspace, so roughly **~54 GB for KV**. At 128 KiB per token that's about 410k tokens: around 50 concurrent sequences at 8k context, or ~400 at 1k context.

> **KEY IDEA:** KV capacity sets the **maximum batch size**, batch size sets **throughput**, and throughput sets **cost**. Shrinking the KV cache is one of the most effective cost levers you have. This is why vLLM's `--gpu-memory-utilization` and `--max-model-len` flags matter so much.

The KV cache also costs **bandwidth**. Every decode step reads each sequence's entire cache. For long contexts (e.g. 128k tokens), reading KV can take longer than reading the weights, and batching doesn't help here because each sequence's KV is unique.

## Shrinking & managing the cache

| Technique | How it helps | Covered in |
|---|---|---|
| **GQA / MQA** | Fewer KV heads, shared across query heads. 4–8× smaller | [2-3](06-attention.md) |
| **MLA** (DeepSeek) | Cache a compressed low-rank latent instead of full K/V | [2-3](06-attention.md) |
| **KV quantization** | Store K/V in FP8 or INT8/INT4: 2–4× smaller | [2-4](07-quantization.md) |
| **PagedAttention** | Allocate in fixed-size blocks, so almost no fragmentation | [2-3](06-attention.md) |
| **Prefix caching** | Share identical prefixes across requests | [3-3](10-prefix-caching.md) |
| **Sliding window / sinks** | Keep only recent tokens plus a few "attention sink" tokens (e.g. StreamingLLM) | below |
| **Eviction** (H2O, SnapKV…) | Drop tokens that get little attention. Lossy | below |
| **Offloading** | Move cold KV to CPU RAM or SSD and bring it back on demand | [3-3](10-prefix-caching.md) |

**Architectural changes** (GQA, MLA, sliding-window layers) are decided at training time, so as a serving engineer you take them as given. **Systems techniques** (paging, quantization, prefix sharing, offloading) are yours to choose. **Lossy eviction** methods trade quality for memory and need careful evaluation.

## Check yourself

**Why don't we cache the queries?**

Only the current token's query is used. It attends to all past keys. Past queries never take part in any future computation.

**Llama-3-70B in BF16 on 8×H100 (640 GB total). Weights ≈ 140 GB. Roughly how many 32k-token sequences fit?**

At 90% utilization: ~576 GB − 140 GB − workspace ≈ ~420 GB for KV. Each 32k-token sequence needs 32,768 × 320 KiB = 10 GiB ≈ 10.7 GB. That's about **39 sequences**. Long context gets expensive fast.

**Does doubling batch size double the KV bandwidth per step?**

Yes. Weight reads are shared across the batch, but each sequence's KV cache is unique and must be read separately. At long context, attention stays memory-bound however big the batch is.

## Further reading

- [Efficiently Scaling Transformer Inference](https://arxiv.org/abs/2211.05102) by Pope et al. (Google)
- [Efficient Streaming Language Models with Attention Sinks](https://arxiv.org/abs/2309.17453) (StreamingLLM)
- [Optimizing your LLM in production](https://huggingface.co/blog/optimize-llm) on the Hugging Face blog

## Interview corner

Questions on this lesson that come up in real interviews. Answer out loud first, then open the card.

**[junior] What is the KV cache and why does it exist?**

#### Strong answer

With causal attention, each past token's key and value vectors never change once computed. Caching them means each decode step computes K/V only for the new token and attends over the cache. That turns quadratic recomputation into linear work. The cost is memory that grows with layers × KV heads × head dim × sequence length × batch.

#### What they're testing

The most fundamental concept in serving.

#### Follow-ups

- Why aren't queries cached?
- Compute the per-token size for Llama-3-8B.

**[mid] Why is serving 128k-token contexts so much more expensive than 4k?**

#### Strong answer

(1) KV memory grows linearly, so 32× more per sequence and far fewer sequences per GPU, which means lower throughput and higher cost. (2) Every decode step reads the whole cache. At long context, KV bytes can exceed weight bytes, and batching doesn't amortize them. (3) Prefill attention is O(N²) compute, so TTFT grows a lot. (4) Scheduling gets harder: big preemptions and big prefills that stall other users.

#### What they're testing

A view of all the dimensions of long-context cost.

#### Follow-ups

- What mitigations exist? (GQA/MLA, FP8 KV, prefix caching, context parallelism, sliding windows.)

**[senior] You're out of KV memory at your target concurrency. List options in order of what you'd try.**

#### Strong answer

(1) Check config: `gpu-memory-utilization`, an over-sized `max-model-len`, CUDA-graph memory. (2) Enable prefix caching if prompts share prefixes. (3) FP8 KV cache, about 2× capacity with small quality risk, so evaluate it. (4) Quantize the weights to free memory for KV. (5) Add TP to pool memory, or add replicas. (6) Offload cold KV to CPU. (7) Long term, pick a model with a smaller KV footprint (GQA/MLA). Each step trades cost, quality or latency.

#### What they're testing

Practical prioritization: cheap, safe changes first.

#### Follow-ups

- How would you validate that FP8 KV didn't hurt quality?

[All interview questions ▶](../interview.md) · [Quiz Arcade ▶](../practice.md#arcade) · [Workbook ▶](../workbook.md)
