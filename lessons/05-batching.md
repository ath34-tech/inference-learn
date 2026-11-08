# Batching: Static → Continuous

> Lesson 2-2 · WORLD 2 Core Techniques · ~30 min · topics: Why batching helps, Static & dynamic, Continuous batching, Chunked prefill

> previous: [2-1 The KV Cache](../lessons/04-kv-cache.md) · next: [2-3 Attention Optimizations](../lessons/06-attention.md)

> Part of **InferenceLearn**, a free course on LLM inference engineering. Web version: lessons/05-batching.html. Whole course in one file: llms-full.txt

Decode is memory-bound: the GPU reads all the weights to produce one token per sequence. Batching makes those reads serve many sequences at once. How you batch makes a 10×+ difference in throughput.

#### By the end you'll be able to

- Explain why batching boosts throughput with little latency cost
- Compare static, dynamic and continuous (iteration-level) batching
- Describe prefill/decode interference and how **chunked prefill** fixes it
- Tune the main scheduler knobs in a real engine

#### Key vocabulary

- **[Static batching](../vocabulary.md#v-static-batching)**: Form a batch, pad it, and run until every sequence finishes. Simple but wasteful.
- **[Dynamic batching](../vocabulary.md#v-dynamic-batching)**: Wait a short window or until the batch is full, then run. Good for fixed-cost models, but still request-level.
- **[Continuous batching](../vocabulary.md#v-continuous-batching)**: Iteration-level scheduling: sequences join and leave the running batch at every decode step. Also called in-flight batching.
- **[Ragged batch](../vocabulary.md#v-ragged-batch)**: A batch whose sequences have different lengths, packed without padding. Kernels get per-sequence offsets.
- **[Head-of-line blocking](../vocabulary.md#v-head-of-line-blocking)**: New work stuck waiting behind slow work already in progress.
- **[Chunked prefill](../vocabulary.md#v-chunked-prefill)**: Splitting a long prompt's prefill into chunks that share engine steps with running decodes, so token streams don't freeze.
- **[Token budget](../vocabulary.md#v-token-budget)**: The maximum number of tokens (prefill chunks plus decodes) processed in one engine step, e.g. vLLM's max-num-batched-tokens.
- **[Preemption](../vocabulary.md#v-preemption)**: Evicting a running sequence when KV memory runs out, then recomputing or swapping its cache in later.

[Drill all terms as flashcards ▶](../vocabulary.md#flashcards)

## Why batching helps

From lesson 1-2: a linear layer's arithmetic intensity ≈ batch size. At B=1, an H100 runs at well under 1% of peak FLOPs during decode. Put 64 sequences in the batch and the GPU reads the weights **once** and does 64× the useful work. The step takes barely longer, because we're still memory-bound.

| Batch | Step time (illustrative, 8B on H100) | System tok/s | Per-user tok/s |
|---|---|---|---|
| 1 | ~6 ms | ~170 | ~170 |
| 16 | ~7 ms | ~2,300 | ~145 |
| 64 | ~9 ms | ~7,000 | ~110 |
| 256 | ~20 ms | ~12,800 | ~50 |

Numbers are illustrative, to show the shape of the curve. Measure your own setup.

## Static batching

The simplest approach: collect N requests, pad them to the same length, and run them together until *all* are done. This is what you get from calling `model.generate()` on a padded batch.

- **Padding waste**: short prompts are padded up to the longest one.
- **Straggler problem**: if one sequence generates 2,000 tokens and the rest stop at 20, those slots sit idle for 1,980 steps.
- **Head-of-line blocking**: new requests wait for the whole batch to finish before they can start.

## Dynamic batching

A request-level improvement used by general model servers (e.g. Triton Inference Server): wait up to a small time window (say 5 ms) or until the batch is full, then run. Great for fixed-cost models like classifiers and embedding models. For generative LLMs it still has the straggler problem, because the batch only "ends" when its slowest member ends.

## Continuous batching

**Continuous batching** (also called iteration-level scheduling or in-flight batching) was introduced by the Orca paper (OSDI 2022). The scheduler decides who's in the batch at **every decode step**:

- When a sequence finishes (EOS or max tokens), it leaves *immediately* and frees its slot and KV memory.
- A waiting request joins at the very next step: its prefill runs and it starts decoding.
- The batch stays full, so the GPU stays busy.

To make this work, the engine needs attention kernels that handle **ragged batches** (different sequence lengths with no padding) and flexible KV memory (paging, lesson 2-3). Every modern engine (vLLM, SGLang, TensorRT-LLM, TGI) does it.

> **Interactive lab: Batching simulator.** Each row is a GPU slot and each column is one decode step. Each colour is one request, labelled with how many tokens it needs. Striped cells are **wasted** slots. Press **▶ Play** to watch both schedulers race, then change the slots and request count. *(Use it on the web page. The formulas it uses are explained in the text.)* Simplified: one token per step, prefill is ignored, and all requests arrive at t=0. In static batching, results return when the whole batch completes.

```
# the heart of a continuous-batching scheduler (simplified)
while True:
    # 1) admit new requests while there's room (slots + KV blocks)
    while waiting and len(running) < MAX_SEQS and kv.can_allocate(waiting[0]):
        req = waiting.popleft()
        kv.allocate(req)
        running.append(req)            # its prefill runs this step

    # 2) one model step over the ragged batch: prefills + one decode token each
    new_tokens = model.step(running)

    # 3) retire finished sequences immediately
    for req, tok in zip(running, new_tokens):
        req.append(tok); stream_to_client(req, tok)
    for req in [r for r in running if r.finished]:
        running.remove(req); kv.free(req)
```

## Prefill/decode interference

Continuous batching adds a new problem. When a request with a 16k-token prompt joins, its prefill takes hundreds of milliseconds, and **every running sequence's decode stalls** for that long. Users see the stream freeze (an ITL spike), while the new request's TTFT is still long.

```
no chunking: □ □ BIG PREFILL (decodes stall) □ □
chunked: □ □ ¼ □ ¼ □ ¼ □ ¼ □
```

*Diagram: blue = decode steps · pink = prefill work. Chunks ride along with decodes, so streams keep flowing*

## Chunked prefill

**Chunked prefill** (popularized by Sarathi-Serve) splits long prompts into chunks of, say, 512–2,048 tokens. Each engine step has a fixed **token budget**. The scheduler first fills it with one decode token per running sequence, then tops it up with a chunk of pending prefill.

- Decodes keep flowing, so ITL stays steady.
- Each step mixes a compute-heavy prefill chunk with memory-bound decodes. That uses *both* the compute and the bandwidth.
- Trade-off: a single long prompt's TTFT goes up a little, because its prefill is spread over several steps.

> **REAL KNOBS:** In vLLM: `--max-num-seqs` (max concurrent sequences), `--max-num-batched-tokens` (per-step token budget, which is the chunk size control), and `--enable-chunked-prefill` (on by default in recent versions). A larger token budget favors throughput and TTFT. A smaller one favors smooth ITL.

## When memory runs out: preemption

Sequences grow one token at a time, so the engine can run out of KV blocks in the middle of generation. Then it has to **preempt** some sequences:

- **Recompute**: drop the victim's KV and re-prefill it later. Cheap when prompts are short, and it's vLLM's default.
- **Swap**: copy the KV to CPU memory and copy it back later. Limited by PCIe bandwidth.

Frequent preemption is a sign that `max-num-seqs` is set too high for your memory. Engines expose a preemption counter metric: watch it.

## Check yourself

**Why doesn't dynamic batching fix the straggler problem for LLMs?**

It decides the batch once, at the request level. Once formed, the batch runs until its longest generation finishes, so finished slots can't be refilled until then.

**What must an attention kernel support for continuous batching to work without padding?**

Variable-length ("ragged") sequences in one batch, typically by passing sequence-start offsets and per-sequence KV block tables instead of a dense padded tensor.

**Your p99 ITL spikes whenever long prompts arrive. What's the first knob to try?**

Make sure chunked prefill is on and **lower the per-step token budget** (e.g. `max-num-batched-tokens`), so prefill chunks are smaller and decodes get scheduled more often.

## Further reading

- [Orca: A Distributed Serving System for Transformer-Based Generative Models](https://www.usenix.org/conference/osdi22/presentation/yu) (OSDI '22)
- [How continuous batching enables 23× throughput in LLM inference](https://www.anyscale.com/blog/continuous-batching-llm-inference) (Anyscale)
- [Sarathi-Serve: Taming Throughput-Latency Tradeoff](https://arxiv.org/abs/2403.02310) (chunked prefill)

## Interview corner

Questions on this lesson that come up in real interviews. Answer out loud first, then open the card.

**[junior] Explain continuous batching and why it beats static batching.**

#### Strong answer

Static batching runs a fixed batch until its longest sequence finishes, so short sequences leave idle slots and new requests wait. Continuous (iteration-level) batching re-decides the batch every decode step: finished sequences leave immediately and waiting ones join. The GPU stays full, often giving several times more throughput at similar latency. It needs ragged-batch kernels and paged KV memory.

#### What they're testing

The core scheduling idea behind every modern engine.

#### Follow-ups

- What new problem does it create? (Prefill/decode interference.)

**[mid] What is chunked prefill, and how do you tune it?**

#### Strong answer

Split long prompts into chunks and fill each step's token budget with decode tokens first, then a prefill chunk. This stops long prefills from freezing everyone's token stream, and it mixes compute-heavy and memory-heavy work in the same step. The knob is the per-step token budget (e.g. max-num-batched-tokens): smaller gives smoother ITL but slower TTFT for long prompts; larger gives better throughput and TTFT but spikier ITL.

#### What they're testing

Understanding the latency trade-off, not just the definition.

#### Follow-ups

- How does this compare with disaggregation?

**[senior] Preemption: recompute vs swap. When is each better?**

#### Strong answer

Recompute throws away the victim's KV and re-runs prefill later. Its cost is compute, and prefill is fast and parallel, so it's cheap for short contexts. Swap copies the KV to CPU and back. Its cost is PCIe bandwidth (~tens of GB/s) and CPU memory. For long contexts where recompute is expensive, or on systems with fast CPU–GPU links (e.g. NVLink-C2C), swapping can win. In both cases, frequent preemption means you're over-admitting: fix the admission limits.

#### What they're testing

Reasoning about costs on different resources.

#### Follow-ups

- Which sequence would you choose as the victim, and why?

[All interview questions ▶](../interview.md) · [Quiz Arcade ▶](../practice.md#arcade) · [Workbook ▶](../workbook.md)
