# How LLM Inference Works

> Lesson 1-1 · WORLD 1 Foundations · ~20 min · topics: Tokenization & logits, Autoregressive loop, Prefill vs decode, Sampling strategies

> next: [1-2 GPU Fundamentals & the Roofline](../lessons/02-gpu-fundamentals.md)

> Part of **InferenceLearn**, a free course on LLM inference engineering. Web version: lessons/01-how-inference-works.html. Whole course in one file: llms-full.txt

Before we optimize anything, we need a clear picture of what happens between "user sends a prompt" and "tokens stream back". Every later lesson is a trick to make some part of this picture faster.

#### By the end you'll be able to

- Trace a prompt through tokenization, the transformer, and sampling
- Explain the autoregressive loop and why it's sequential
- Tell apart the **prefill** and **decode** phases and their very different costs
- Describe the common sampling strategies

#### Key vocabulary

- **[Token](../vocabulary.md#v-token)**: The unit an LLM reads and writes: an integer ID from the tokenizer's vocabulary. An English token is about ¾ of a word.
- **[Tokenizer](../vocabulary.md#v-tokenizer)**: Converts text to token IDs and back (e.g. BPE). Its vocabulary usually has 32k–256k entries.
- **[Logits](../vocabulary.md#v-logits)**: The model's raw, unnormalized scores, one for every vocabulary token. Softmax turns them into probabilities.
- **[Autoregressive decoding](../vocabulary.md#v-autoregressive-decoding)**: Generating one token at a time and feeding each new token back in as input for the next step.
- **[Prefill](../vocabulary.md#v-prefill)**: The first phase of a request: process the whole prompt in parallel, build its KV cache, and produce the first token. Compute-bound.
- **[Decode](../vocabulary.md#v-decode)**: The phase that produces output tokens one per step. Each step reads all the weights for very little math, so it's memory-bandwidth-bound.
- **[Temperature](../vocabulary.md#v-temperature)**: Divides logits before softmax. Below 1 makes outputs more focused, above 1 more random, and 0 means greedy.
- **[Top-p (nucleus) sampling](../vocabulary.md#v-top-p-nucleus-sampling)**: Sample from the smallest set of tokens whose probabilities add up to at least p (e.g. 0.95). The set grows when the model is unsure.

[Drill all terms as flashcards ▶](../vocabulary.md#flashcards)

## Inference vs training

**Training** adjusts a model's weights with gradient descent over huge datasets. It runs once (or a few times), on big clusters, and is measured in weeks. **Inference** uses the frozen weights to produce outputs for real requests. It runs continuously and is measured in milliseconds per request and dollars per million tokens.

Inference has no backward pass, no optimizer state and no gradients. That makes it simpler in one way. But it brings problems training doesn't have: requests arrive at random times, have wildly different lengths, and each user is waiting on a latency target.

|  | Training | Inference |
|---|---|---|
| Goal | Learn weights | Serve outputs fast and cheap |
| Passes | Forward + backward | Forward only |
| Batch shape | Fixed, large, known ahead | Dynamic, arrives over time |
| Main metric | Time to converge, MFU | Latency, throughput, $/token |
| Memory hogs | Activations, optimizer state | Weights, **KV cache** |

## From text to tokens and back

An LLM never sees characters. It sees **tokens**, integer IDs from a fixed vocabulary (often 32k–256k entries) produced by a tokenizer such as BPE. As a rule of thumb, one English token is about ¾ of a word.

Flow: "Hello pixel world" → tokenizer → [9906, 13438, 1917] → transformer (N layers) → logits [vocab] → sample → next token

*Diagram: one forward pass = one step of the pipeline above (token IDs are illustrative)*

Inside the model, each token ID is looked up in an **embedding table** to get a vector of size `d_model` (4096 for an 8B model). That vector passes through `N` identical transformer blocks. Each block has **self-attention**, where tokens look at earlier tokens, and an **MLP**, a per-token feed-forward network. At the end, a final projection (the "LM head") turns the last hidden state into one **logit** per vocabulary entry.

## The autoregressive loop

A decoder-only LLM predicts *one* next token at a time. To generate a sentence, we append the predicted token to the input and run the model again, over and over. This is **autoregressive generation**:

```
# the simplest possible generation loop (no KV cache yet!)
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B", torch_dtype=torch.bfloat16).cuda().eval()

ids = tok("The capital of France is", return_tensors="pt").input_ids.cuda()
with torch.no_grad():
    for _ in range(20):
        logits = model(ids).logits          # [1, seq_len, vocab]: recomputes EVERYTHING each step
        next_id = logits[:, -1].argmax(-1, keepdim=True)   # greedy pick
        ids = torch.cat([ids, next_id], dim=1)
        if next_id.item() == tok.eos_token_id: break
print(tok.decode(ids[0]))
```

This works, but it's very wasteful. Step `t` re-processes all `t` previous tokens, even though their internal states never change. Fixing that waste is the [KV cache](04-kv-cache.md) (lesson 2-1).

> **KEY IDEA:** Generation is **sequential by nature**. Token 51 can't be computed until token 50 is known. You can't parallelize *across time* within one request, so inference engineers parallelize *across requests* (batching) and find ways around the rule (speculative decoding).

## Prefill vs decode

With a KV cache, serving a request splits into two phases that behave very differently on hardware:

```
prefill: □ □ □ □ □ □ □ □ → token 1
decode 1: □ □ □ □ □ □ □ □ □ → token 2
decode 2: □ □ □ □ □ □ □ □ □ □ → token 3
```

*Diagram: pink = processed in parallel · striped = read from KV cache · blue = the one new token*

|  | Prefill | Decode |
|---|---|---|
| Input | The whole prompt (100s–100,000s of tokens) at once | 1 new token per sequence |
| Produces | KV cache for the prompt + the first output token | One more token + its KV entry |
| Bottleneck | **Compute**: big matmuls, tensor cores busy | **Memory bandwidth**: reads all weights + KV for a tiny bit of math |
| User-visible metric | Time To First Token (TTFT) | Inter-Token Latency (ITL) |
| Runs | Once per request | Once per output token |

Why is decode memory-bound? Each decode step must stream *every weight* of the model from GPU memory (16 GB for an 8B model in BF16), only to multiply them by a single token's vector. The math finishes almost instantly, and the GPU spends most of its time waiting on memory. [Lesson 1-2](02-gpu-fundamentals.md) makes this exact with the roofline model.

## Sampling: picking the next token

The model outputs logits, not a token. A **sampler** turns logits into a choice:

- **Greedy**: always pick `argmax`. Deterministic, but can be repetitive.
- **Temperature** `T`: divide logits by `T` before softmax. `T<1` sharpens the distribution, `T>1` flattens it, and `T→0` is greedy.
- **Top-k**: keep only the k most likely tokens, renormalize, then sample.
- **Top-p (nucleus)**: keep the smallest set of tokens whose probabilities add up to ≥ p.
- **Min-p**: keep tokens with probability ≥ `min_p × p_max`. This scales with the model's confidence.
- **Penalties**: repetition, frequency and presence penalties change the logits of tokens already generated.

```
def sample(logits, temperature=0.8, top_p=0.95):
    if temperature == 0:
        return logits.argmax(-1)
    probs = torch.softmax(logits / temperature, dim=-1)
    sorted_p, idx = probs.sort(descending=True)
    cum = sorted_p.cumsum(-1)
    sorted_p[cum - sorted_p > top_p] = 0        # drop the tail beyond the nucleus
    choice = torch.multinomial(sorted_p / sorted_p.sum(), 1)
    return idx.gather(-1, choice)
```

> **ENGINEERING NOTE:** Sampling is cheap in FLOPs, but it's not free in a serving engine. Sorting a 150k-entry vocab for every sequence in a batch of 256, every step, adds up. Engines use fused GPU sampling kernels and avoid copying logits back to the CPU.

## Where the time goes

For a typical chat request (2,000-token prompt, 300-token answer) on one GPU:

- Prefill happens **once** and is fast per token, because tokens are processed in parallel.
- Decode happens **300 times**, and each step costs about the same as reading the whole model from memory.
- So end-to-end latency is usually **dominated by decode**, and decode is dominated by memory bandwidth.

That's why so many optimizations in this course target decode: batching (amortize the weight reads), quantization (fewer bytes to read), speculative decoding (more tokens per read), and smaller KV caches (fewer bytes per token).

## Check yourself

**Why can't we generate all 300 output tokens in one forward pass?**

Each token depends on the ones before it. The model needs token *t* as input to predict token *t+1*, so generation is a chain of dependent steps.

**Prefill of 2,000 tokens vs decode of 1 token: which uses the GPU's tensor cores more efficiently, and why?**

Prefill. It multiplies the weights by a 2,000-row matrix, so each byte of weight loaded is reused 2,000 times. Decode multiplies by a single row, so each weight byte loaded is used for only ~2 FLOPs. That's far too little work to keep tensor cores busy.

**What happens to the output distribution as temperature → 0? As temperature → ∞?**

T → 0 puts all probability on the argmax (greedy). T → ∞ makes the distribution uniform over the vocabulary, so the output is random noise.

## Further reading

- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) by Jay Alammar
- [Let's build GPT: from scratch, in code](https://www.youtube.com/watch?v=kCc8FmEb1nY) by Andrej Karpathy
- [How to generate text: decoding methods](https://huggingface.co/blog/how-to-generate) on the Hugging Face blog

## Interview corner

Questions on this lesson that come up in real interviews. Answer out loud first, then open the card.

**[junior] Walk me through what happens when an LLM serves one request.**

#### Strong answer

Tokenize the prompt. Run **prefill**: one forward pass over all prompt tokens in parallel, which fills the KV cache and produces logits for the first token. Sample a token, stream it back, then loop in **decode**: each step feeds one new token, appends its K/V to the cache, and samples the next, until EOS or max_tokens. Detokenize and stream as you go.

#### What they're testing

Whether you know the two phases and that decode is a sequential loop. Mentioning the KV cache unprompted is a good sign.

#### Follow-ups

- Which phase dominates latency for a 2k-in / 500-out chat request? (Decode: 500 sequential steps.)
- Where does the scheduler fit in?

**[mid] Why do prefill and decode stress the GPU so differently?**

#### Strong answer

Prefill multiplies the weights by a matrix of N prompt tokens, so each weight byte loaded is reused N times. Arithmetic intensity is high and it's **compute-bound**. Decode multiplies the weights by one vector per sequence, so it's about 1 FLOP per byte at batch 1 and **memory-bandwidth-bound**. Decode also reads the KV cache, which can't be shared across sequences.

#### What they're testing

The roofline intuition, which most of inference optimization rests on.

#### Follow-ups

- How does batching change decode's position on the roofline?
- What is attention's intensity in decode, and why doesn't batching help it?

**[junior] Greedy vs temperature vs top-p sampling: when would you use each?**

#### Strong answer

Greedy (T=0) for deterministic tasks such as extraction, classification and code where you want the most likely answer. Temperature 0.7–1.0 with top-p ≈ 0.9–0.95 for creative or chat output. Top-p adapts the candidate set to the model's confidence, while top-k is a fixed cutoff. Mention that sampling has a cost in serving (sorting large vocabularies per sequence), so engines use fused GPU samplers.

#### What they're testing

Basic decoding knowledge plus some awareness of the systems side.

#### Follow-ups

- Why can T=0 still be non-deterministic in a batched server?

[All interview questions ▶](../interview.md) · [Quiz Arcade ▶](../practice.md#arcade) · [Workbook ▶](../workbook.md)
