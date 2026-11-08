# Speculative Decoding

> Lesson 3-1 · WORLD 3 Advanced · ~30 min · topics: Draft & verify, Rejection sampling, Expected speedup, Medusa / EAGLE / MTP

> previous: [2-4 Quantization](../lessons/07-quantization.md) · next: [3-2 Parallelism & MoE Inference](../lessons/09-parallelism.md)

> Part of **InferenceLearn**, a free course on LLM inference engineering. Web version: lessons/08-speculative-decoding.html. Whole course in one file: llms-full.txt

Decode is memory-bound: checking 5 tokens costs about the same as producing 1. Speculative decoding exploits this. Guess several tokens cheaply, then verify them all in one forward pass of the big model, and keep exactly the output distribution you'd have had anyway.

#### By the end you'll be able to

- Walk through the draft → verify → accept loop
- Explain why rejection sampling keeps the output distribution identical
- Estimate speedup from acceptance rate and draft length
- Compare draft methods: draft models, n-gram, Medusa, EAGLE, MTP

#### Key vocabulary

- **[Speculative decoding](../vocabulary.md#v-speculative-decoding)**: Draft several tokens cheaply, then verify them all in one target-model pass. The output distribution is unchanged.
- **[Draft model](../vocabulary.md#v-draft-model)**: The cheap proposer in speculative decoding: a small model, n-gram lookup, or extra heads.
- **[Target model](../vocabulary.md#v-target-model)**: The big model whose output distribution you want. It verifies the drafts.
- **[Acceptance rate](../vocabulary.md#v-acceptance-rate)**: α: the probability that a drafted token is accepted. Expected tokens per pass = (1 − α^(k+1)) ÷ (1 − α).
- **[Rejection sampling](../vocabulary.md#v-rejection-sampling)**: Accept a draft token with probability min(1, p/q), otherwise resample from max(0, p − q). This makes speculation lossless.
- **[EAGLE / Medusa](../vocabulary.md#v-eagle-medusa)**: Speculative methods that draft with lightweight heads attached to the target model instead of a separate model.
- **[MTP](../vocabulary.md#v-mtp)**: Multi-Token Prediction: modules trained to predict several future tokens. They can serve as speculative drafts.
- **[Tree attention](../vocabulary.md#v-tree-attention)**: Verifying a tree of alternative drafts in one pass with a special attention mask.

[Drill all terms as flashcards ▶](../vocabulary.md#flashcards)

## The core idea

A forward pass of the target model over `k+1` tokens costs about the same as over 1 token at low batch, because the weight reads dominate. So if something cheap can *guess* the next k tokens, the big model can check all k in one pass:

```
draft guesses: the cat sat on a
target checks: the ✓ cat ✓ sat ✓ on ✗ → "in" (drop)
```

*Diagram: 3 accepted + 1 corrected token = 4 tokens from a single target forward pass*

1. **Draft**: a cheap method proposes k tokens `x₁…xₖ`, with draft probabilities `q(xᵢ)`.
2. **Verify**: run the target model once on the context plus all k drafts. This gives the target probabilities `p(·)` at each of the k+1 positions.
3. **Accept/reject** left to right. At the first rejection, sample a corrected token and discard the remaining drafts.
4. If all k are accepted, you also get a free **bonus token** from the last position.

## Why the output is identical

Speculative sampling (Leviathan et al.; Chen et al., 2023) uses a rejection-sampling rule:

> accept draft token x with probability **min(1, p(x) / q(x))**  
  
on rejection, sample from **norm( max(0, p − q) )**

You can prove that the accepted-or-resampled token is distributed **exactly according to p**, the target model's distribution. Speculative decoding is **lossless**: it changes speed, not output. With greedy decoding it's even simpler: accept while the draft token equals the target's argmax.

```
# greedy speculative decoding: one verify step
def spec_step(target, draft, ctx, k=4):
    proposal = draft.greedy_generate(ctx, k)          # k cheap tokens
    logits = target(ctx + proposal)                   # ONE big forward pass
    preds = logits[-(k + 1):].argmax(-1)               # target's choice at each position
    out = []
    for i in range(k):
        if proposal[i] == preds[i]:
            out.append(proposal[i])                    # accepted
        else:
            out.append(preds[i]); return out              # corrected, stop
    out.append(preds[k])                               # all accepted: bonus token
    return out                                         # remember to roll back rejected KV entries
```

## Expected speedup

If each draft token is accepted independently with probability **α** (the acceptance rate), the expected number of tokens per verify step with draft length k is:

> E[tokens] = **(1 − α^k+1) / (1 − α)**

With α = 0.8 and k = 4 that's (1 − 0.8⁵) / 0.2 ≈ **3.36 tokens** per target pass. Divide by the cost of a step (target pass + k draft passes) to get the ideal speedup.

> **Interactive lab: Speculative decoding lab.** Set how often the draft is right (α) and how far it guesses ahead (k). Then **roll the dice**: each row is one target-model pass. ✓ accepted · ✗ rejected · +1 the target's own token (a correction or a bonus). *(Use it on the web page. The formulas it uses are explained in the text.)* Assumes the verify pass costs the same as one normal decode step (true at low batch). Try a low α with a large k to see wasted drafting.

## Where drafts come from

| Method | How it drafts | Pros / cons |
|---|---|---|
| **Draft model** | Small model from the same family (e.g. 1B drafting for 70B) | Simple. Needs the same tokenizer and costs extra memory |
| **N-gram / prompt lookup** | Copy spans that already appear in the prompt or context | Free. Great for code edits, RAG and summarization; weak elsewhere |
| **Medusa** | Extra decoding heads predict tokens +2, +3… from the last hidden state | No separate model. Heads need training. Uses tree verification |
| **EAGLE (1/2/3)** | Tiny autoregressive head on the target's own features | High acceptance rates and widely supported. Needs a trained head |
| **MTP** | Multi-token-prediction modules trained with the model (e.g. DeepSeek-V3) | Strong drafts out of the box when the model ships with them |

**Tree speculation**: instead of one chain of guesses, propose a small *tree* of alternatives and verify every branch in one pass with a special tree-attention mask. This raises the expected accepted length.

## When it helps and when it hurts

- **Helps most at low batch**, where you're memory-bound and the extra verify tokens are almost free. It's ideal for latency-sensitive, low-concurrency serving.
- **Gains shrink at high batch.** Once you're compute-bound, verifying k extra tokens per sequence costs real FLOPs, and rejected tokens are wasted compute. Some engines lower k or switch speculation off under load.
- **Depends on the domain.** Acceptance rates are higher for predictable text (code, structured output) and lower for creative, high-temperature sampling.
- **Bookkeeping**: rejected tokens' KV entries must be rolled back. The draft model needs its own KV cache. It interacts with CUDA graphs and batching.

## Check yourself

**Does speculative decoding change what the model outputs?**

No. With the correct rejection-sampling rule, outputs follow exactly the target model's distribution (and match exactly under greedy decoding, up to floating-point non-determinism).

**α = 0.5, k = 8. What is E[tokens]? Is k = 8 a good choice?**

(1 − 0.5⁹)/0.5 ≈ **2.0**. With k=2 you'd already get (1 − 0.5³)/0.5 = 1.75. A long draft wastes effort when α is low, so a shorter k is better here.

**Why does speculative decoding help less when the server is heavily loaded?**

At high batch, the forward pass is compute-bound. Verifying extra tokens is no longer free: it adds real FLOPs, which slows down every sequence in the batch.

## Further reading

- [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192) by Leviathan et al.
- [Accelerating LLM Decoding with Speculative Sampling](https://arxiv.org/abs/2302.01318) by Chen et al. (DeepMind)
- [Medusa](https://arxiv.org/abs/2401.10774) · [EAGLE](https://arxiv.org/abs/2401.15077)

## Interview corner

Questions on this lesson that come up in real interviews. Answer out loud first, then open the card.

**[mid] Explain speculative decoding and why it doesn't change the output distribution.**

#### Strong answer

A cheap drafter proposes k tokens and the target verifies all of them in one forward pass, which costs about one decode step when memory-bound. Accept draft token x with probability min(1, p(x)/q(x)). On the first rejection, resample from normalize(max(0, p − q)) and stop. This rejection-sampling scheme provably yields samples from p. Expected tokens per pass = (1 − α^(k+1)) / (1 − α).

#### What they're testing

Both the systems intuition and the probability argument.

#### Follow-ups

- What's the greedy special case?
- What happens to the KV cache for rejected tokens?

**[senior] Speculative decoding made p50 latency better but throughput worse under load. Why, and what would you do?**

#### Strong answer

At high batch the engine is compute-bound. Verifying k extra tokens per sequence adds real FLOPs, and rejected drafts are pure waste, so total throughput drops. Options: adapt k (or turn speculation off) based on batch size or load; use cheaper drafters (n-gram, EAGLE heads); only speculate for latency-sensitive tiers; and track acceptance rate per workload.

#### What they're testing

Knowing when an optimization stops paying off.

#### Follow-ups

- Which metrics would you put on a dashboard for this?

**[junior] Name three ways to get draft tokens.**

#### Strong answer

A small draft model from the same family (same tokenizer); n-gram or prompt-lookup copying from the context; extra prediction heads trained on the target (Medusa, EAGLE) or built-in MTP modules (e.g. DeepSeek-V3).

#### What they're testing

Breadth.

#### Follow-ups

- Which would you pick for a code-editing product, and why? (Prompt lookup: edits copy large spans.)

[All interview questions ▶](../interview.md) · [Quiz Arcade ▶](../practice.md#arcade) · [Workbook ▶](../workbook.md)
