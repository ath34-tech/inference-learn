# LearnInference Engineering, one pixel at a time.

> Part of **InferenceLearn**, a free course on LLM inference engineering. Web version: index.html. Whole course in one file: llms-full.txt

A free, structured path from *"how does a model produce a token?"* to running LLMs fast and cheap in production. You'll cover KV caches, batching, FlashAttention, quantization, speculative decoding, parallelism and more.

[View the roadmap ▶](roadmap.md) [Start lesson 1-1](lessons/01-how-inference-works.md)

// what is inference engineering?

## Training builds the model. Inference is where you pay for it.

Inference engineering is the craft of **serving trained models** to users with low latency, high throughput and low cost. A lab trains a model once, then runs inference on it billions of times. Small gains add up: a 2× faster engine can halve a GPU bill.

It sits where ML, GPU architecture and distributed systems meet. You need to know what a transformer computes, where the bytes move on the hardware, and how to schedule thousands of requests at once.

4worlds, basic → expert

13in-depth lessons

6interactive labs

60+curated resources

// the journey

## Four worlds to clear

Each world builds on the last. Go in order if you're new, or jump straight to a topic.

[[World 1]  Foundations Prefill vs decode, GPU memory hierarchy, roofline math, and the metrics that matter.](learn.md#w1) [[World 2]  Core Techniques KV cache, continuous batching, FlashAttention, PagedAttention, quantization.](learn.md#w2) [[World 3]  Advanced Speculative decoding, tensor/expert parallelism, prefix caching, disaggregated serving.](learn.md#w3) [[World 4]  Expert / Systems Kernel fusion, CUDA graphs, Triton, profiling, and production operations.](learn.md#w4)

// your save file

## Track your progress

Mark lessons as cleared and the site remembers them in your browser. Nothing leaves your machine: no account, no tracking.

[Start lesson 1-1](lessons/01-how-inference-works.md)

// inventory

## What's inside

[Roadmap A visual path from prerequisites to capstone projects.](roadmap.md) [Lessons Written explanations with diagrams, formulas, code, calculators and quizzes.](learn.md) [Resources Key papers, blogs, videos, books and open-source engines.](resources.md) [Vocabulary 129 terms in plain English with analogies, plus flashcards to drill them.](vocabulary.md) [Quiz Arcade 45 questions, 3 lives, streak bonuses and a hi-score. Every answer is explained.](practice.md) [Interview Prep 39 questions with model answers, what interviewers test, and system-design prompts.](interview.md) [Workbook 14 napkin-math problems with worked solutions, explain-it-back prompts and experiments.](workbook.md) [6 Interactive labs KV memory map, roofline, batching race, quantization playground, speculative dice, cost builder.](lessons/04-kv-cache.md#kv-calc)

// co-op mode

## Learn with your own AI tutor

Every page is written to be easy for AI assistants to read. Paste any page's link into **ChatGPT, Claude or Gemini** and ask it to teach you, quiz you, or interview you. Each lesson has a ready-made tutor prompt at the bottom.

- [llms.txt](llms.txt): an index of the whole site for AI assistants
- [llms-full.txt](llms-full.txt): the entire course as one plain-text file
- Every page has a `.md` twin, e.g. [lessons/04-kv-cache.md](lessons/04-kv-cache.md)
- Try: *"Read [link] and teach me this lesson step by step, quizzing me as we go."*
