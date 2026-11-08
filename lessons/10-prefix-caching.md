# Prefix Caching & KV Memory

> Lesson 3-3 · WORLD 3 Advanced · ~25 min · topics: Shared prefixes, Hash blocks vs radix tree, Eviction & preemption, KV offloading

> previous: [3-2 Parallelism & MoE Inference](../lessons/09-parallelism.md) · next: [3-4 Scheduling & Disaggregated Serving](../lessons/11-disaggregation.md)

> Part of **InferenceLearn**, a free course on LLM inference engineering. Web version: lessons/10-prefix-caching.html. Whole course in one file: llms-full.txt

Real traffic repeats itself: the same system prompt, the same few-shot examples, the same growing chat history, the same documents. If two requests share a prefix, their KV for that prefix is identical. Compute it once and reuse it.

#### By the end you'll be able to

- Explain why shared prefixes give identical KV entries
- Compare hash-based block caching and radix-tree caching
- Describe eviction, reference counting and offloading tiers
- Design prompts and routing to maximize cache hits

#### Key vocabulary

- **[Prefix caching](../vocabulary.md#v-prefix-caching)**: Reusing KV for prompt prefixes shared across requests (system prompts, chat history), so prefill skips them.
- **[RadixAttention](../vocabulary.md#v-radixattention)**: SGLang's prefix cache, organized as a radix tree over token sequences, with token-level matching and natural branching.
- **[Reference counting](../vocabulary.md#v-reference-counting)**: Tracking how many running sequences use a KV block. Only blocks with count 0 may be evicted.
- **[LRU eviction](../vocabulary.md#v-lru-eviction)**: Least-Recently-Used: when memory is needed, drop the cached block that was used longest ago.
- **[Copy-on-write](../vocabulary.md#v-copy-on-write)**: Sequences share the same physical KV blocks until one of them writes something different. Only then is a private copy made.
- **[KV offloading](../vocabulary.md#v-kv-offloading)**: Moving cold KV blocks from GPU memory to CPU RAM, SSD or remote storage, and loading them back when needed.
- **[Cache hit rate](../vocabulary.md#v-cache-hit-rate)**: The fraction of prompt tokens served from the prefix cache. A sudden drop often means a prompt template changed.

[Drill all terms as flashcards ▶](../vocabulary.md#flashcards)

## Why prefixes repeat

- **System prompts**: every request to a product starts with the same 2,000-token instructions.
- **Multi-turn chat**: turn N's prompt is turn N−1's prompt plus the reply plus the new user message.
- **Agents & tool use**: long, mostly stable contexts that get re-sent on every call.
- **Few-shot & RAG**: the same examples or documents across many queries.

Because of causal attention, the KV for token `i` depends only on tokens 0…i. Two requests with the same first 2,000 tokens have **bit-identical KV** for those tokens, given the same model and numerics.

## Hash-based block caching

With a paged KV cache (lesson 2-3), each block holds a fixed number of tokens. vLLM's **automatic prefix caching** gives every *full* block a key:

> block_hash = hash( **parent_block_hash**, tokens_in_this_block, extra keys such as LoRA id )

Including the parent hash means a block's key identifies its **entire prefix**, not just its own tokens. When a new request arrives, the engine hashes its prompt block by block and looks up each hash. The longest run of hits is reused and prefill starts after it.

```
request A: sys 1 sys 2 sys 3 user A…
request B: sys 1 sys 2 sys 3 user B…
```

*Diagram: green blocks are shared (cache hits). Only the coloured tails need prefill*

## Radix-tree caching

SGLang's **RadixAttention** stores cached prefixes in a **radix tree** keyed by token sequences. Each edge is a run of tokens, and each node points to KV memory. Matching a new prompt means walking the tree. Branching conversations, such as several replies from one point or tree-of-thought agents, share structure naturally.

- Hash blocks: simple, O(1) lookups per block, matching at block granularity.
- Radix tree: matching at token granularity and natural branching. Supports cache-aware scheduling, e.g. running requests that share a prefix together.

## Eviction & reference counting

Cached blocks stay in GPU memory after their request finishes, in case they're useful later. When memory is needed:

- Blocks still used by running requests have **ref count > 0** and can't be evicted.
- Free cached blocks are evicted with a policy like **LRU**, usually leaf-first, so shared ancestors survive longer.
- A cache hit "revives" a block: its ref count goes up and it leaves the free list.

So caching is nearly free: cached-but-idle blocks sit in memory that would otherwise be empty, and they're dropped the moment a live request needs the space.

## Offloading tiers

GPU HBM is small. A growing ecosystem extends the KV cache down a memory hierarchy:

| Tier | Capacity | Speed to GPU | Use |
|---|---|---|---|
| GPU HBM | 10s of GB | — | Active and hot cached blocks |
| CPU DRAM | 100s of GB–TBs | PCIe / NVLink-C2C | Warm prefixes, swapped sequences |
| Local NVMe | TBs | GB/s | Large cold caches |
| Remote / distributed store | Cluster-wide | RDMA / network | Sharing KV across replicas |

Projects like **LMCache** and **Mooncake** (Kimi's KV-centric serving architecture) build on this. Loading KV pays off when it's faster than recomputing it. For long prefixes on fast links, it often is.

## Making it work in practice

- **Put static content first**: system prompt, tool definitions, documents, *then* the variable part. A timestamp or user ID at the top of the prompt breaks every cache hit.
- **Keep formatting byte-stable**: different whitespace or reordered JSON keys produce different tokens and miss the cache.
- **Prefix-aware routing**: with many replicas, send requests that share a prefix (same conversation or tenant) to the same replica. Random load balancing spreads prefixes and wrecks hit rates (lesson 3-4).
- **Monitor the hit rate**: engines export prefix-cache hit metrics, and a sudden drop usually means a prompt-template change.

> **KEY IDEA:** A cache hit skips prefill for the shared tokens, which cuts both **TTFT** and **GPU compute**. That's why API providers charge less for cached input tokens.

## Check yourself

**Why does a block hash include the parent's hash rather than just its own tokens?**

KV depends on *all* preceding tokens. The same 16 tokens after a different prefix produce different KV. Chaining the parent hash makes the key represent the whole prefix.

**Your prompt template starts with "Current time: 14:03:22". What happens to your cache hit rate?**

It falls to almost zero. The first block differs on every request, so no later block can match either. Move dynamic values to the end.

**Does keeping finished requests' blocks around reduce capacity for live requests?**

No. Cached blocks with ref count 0 can be evicted at any moment when a running or new request needs memory. They only occupy space that would otherwise be idle.

## Further reading

- [SGLang: Efficient Execution of Structured Language Model Programs](https://arxiv.org/abs/2312.07104) (RadixAttention)
- [vLLM docs: Automatic Prefix Caching design](https://docs.vllm.ai/en/latest/design/prefix_caching.html)
- [LMCache](https://github.com/LMCache/LMCache) · [Mooncake](https://github.com/kvcache-ai/Mooncake)

## Interview corner

Questions on this lesson that come up in real interviews. Answer out loud first, then open the card.

**[mid] How would you implement prefix caching in a paged-KV engine?**

#### Strong answer

Give each full block a key = hash(parent key, the block's token IDs, extras like LoRA ID). Keep a hash → physical-block map. On a new request, hash the prompt block by block, reuse the longest run of hits (incrementing ref counts), and prefill from there. When a request finishes, its blocks go to an evictable LRU list rather than being freed. Evict only blocks with ref count 0, leaves first. The alternative is a radix tree (SGLang) for token-level matching.

#### What they're testing

Data-structure design under memory pressure.

#### Follow-ups

- How do you handle a partially filled last block?
- Hash collisions?

**[senior] You run 20 replicas behind a round-robin load balancer and prefix-cache hit rates are poor. What do you change?**

#### Strong answer

Round-robin scatters each conversation across replicas, so caches never warm up. Move to cache-aware routing: session affinity (hash the conversation ID) as a cheap first step, then a KV-aware router that scores replicas on predicted prefix overlap minus current load, to avoid hotspots. Also fix the prompt template so static content comes first, and consider a shared or offloaded KV tier.

#### What they're testing

Seeing the system as a whole beyond one engine.

#### Follow-ups

- How do you measure success? (Hit rate, TTFT and GPU prefill time per request.)

**[staff] Design serving for a RAG product: 30k-token prompts (mostly retrieved docs), short answers.**

#### A strong structure

- **Prefill-dominated:** TTFT and prefill compute are the cost drivers, not decode.
- **Reuse:** cache KV for popular documents. Order prompts as [system][docs sorted by a stable ID][question]. Use a KV offload tier (CPU/SSD) for document caches. Route on document overlap.
- **Compute:** FP8 prefill, high-TP or disaggregated prefill pools, chunked prefill so the few decoders aren't starved.
- **Memory:** long KV per request but short lifetimes. Size the pool for burst concurrency.
- **Product lever:** retrieve fewer and better chunks. The cheapest token is the one you never send.

#### What they're testing

Recognizing the workload shape and optimizing for it.

[All interview questions ▶](../interview.md) · [Quiz Arcade ▶](../practice.md#arcade) · [Workbook ▶](../workbook.md)
