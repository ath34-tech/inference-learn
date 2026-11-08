# Vocabulary

> Part of **InferenceLearn**, a free course on LLM inference engineering. Web version: vocabulary.html. Whole course in one file: llms-full.txt

// item descriptions

Every term in the course in plain English. Most come with a "≈" analogy to make them stick. Search the list, or switch to **flashcards** to drill them.

## Basics

- **Inference**: Running a trained model on new inputs to produce outputs. No learning happens, only forward passes. Training is writing the cookbook; inference is cooking from it for every customer.
- **Token**: The unit an LLM reads and writes: an integer ID from the tokenizer's vocabulary. An English token is about ¾ of a word. Lego bricks that text is built from.
- **Tokenizer**: Converts text to token IDs and back (e.g. BPE). Its vocabulary usually has 32k–256k entries.
- **Embedding**: A learned vector (e.g. 4096 numbers) looked up for each token ID. It's the model's internal representation of the token.
- **Logits**: The model's raw, unnormalized scores, one for every vocabulary token. Softmax turns them into probabilities.
- **Softmax**: Turns a list of scores into probabilities that add up to 1: exp(xᵢ) ÷ Σ exp(xⱼ).
- **Autoregressive decoding**: Generating one token at a time and feeding each new token back in as input for the next step. Writing a sentence where you can only see the words you've already written.
- **Prefill**: The first phase of a request: process the whole prompt in parallel, build its KV cache, and produce the first token. Compute-bound. Reading the whole question before you start answering.
- **Decode**: The phase that produces output tokens one per step. Each step reads all the weights for very little math, so it's memory-bandwidth-bound.
- **Greedy decoding**: Always picking the single most likely next token (argmax). Deterministic, but can be repetitive.
- **Temperature**: Divides logits before softmax. Below 1 makes outputs more focused, above 1 more random, and 0 means greedy. A creativity dial.
- **Top-k sampling**: Sample only from the k most likely tokens.
- **Top-p (nucleus) sampling**: Sample from the smallest set of tokens whose probabilities add up to at least p (e.g. 0.95). The set grows when the model is unsure.
- **EOS token**: End-of-sequence token. When the model emits it, generation for that request stops.
- **Context length**: The number of tokens (prompt plus output) a sequence holds. The KV cache grows linearly with it.
- **Self-attention**: The layer where each token computes a query, key and value, and mixes in information from earlier tokens weighted by query·key similarity.
- **Causal mask**: Stops a token from attending to future tokens. It's why past keys and values never change, and so why they can be cached.

## GPU & hardware

- **GPU**: A massively parallel processor with thousands of cores and specialized matrix units. The workhorse of LLM inference.
- **Streaming Multiprocessor (SM)**: One of the GPU's many independent compute units (132 on an H100 SXM). Each has cores, tensor cores, registers and shared memory. One kitchen station in a huge restaurant.
- **Tensor cores**: Specialized units for very fast matrix multiply-accumulate at low precision (BF16, FP8, FP4). Most of a GPU's FLOPs come from them.
- **HBM**: High-Bandwidth Memory: the GPU's main memory (e.g. 80 GB at ~3.35 TB/s on H100). It holds the weights and KV cache. The warehouse: huge, but a walk away.
- **SRAM / shared memory**: Small, very fast on-chip memory per SM (~228 KB on H100). Kernels stage tiles of data here. The cutting board: tiny, but right under your hands.
- **Memory bandwidth**: How many bytes per second can move between HBM and the compute units. It sets decode speed.
- **FLOPs**: Floating-point operations. A forward pass costs about 2 × parameters FLOPs per token. FLOP/s (per second) measures compute speed.
- **Arithmetic intensity**: FLOPs done per byte moved from memory. For LLM linear layers in BF16 it's roughly the number of tokens in the batch. How much cooking you do per trip to the warehouse.
- **Roofline model**: Attainable performance = min(peak FLOP/s, intensity × bandwidth). It tells you which limit you'll hit first.
- **Ridge point**: Peak FLOP/s ÷ bandwidth: the intensity where a GPU switches from memory-bound to compute-bound (~295 FLOP/byte on H100).
- **Memory-bound**: Limited by how fast data moves, not by math. Compute units sit idle waiting for bytes. Typical of decode.
- **Compute-bound**: Limited by math throughput. Memory keeps up and the tensor cores are the bottleneck. Typical of prefill and large batches.
- **BF16**: bfloat16: a 16-bit float with FP32's range (8-bit exponent) but less precision. The default serving precision.
- **NVLink**: NVIDIA's high-bandwidth GPU-to-GPU link (~900 GB/s per GPU on H100). Essential for fast tensor parallelism.
- **PCIe**: The general-purpose bus between GPU and CPU (~64 GB/s each way for PCIe 5 x16). Much slower than NVLink.

## Metrics

- **TTFT**: Time To First Token: from sending the request to receiving the first token. Includes queueing and prefill. How long until the waiter says "coming right up".
- **ITL**: Inter-Token Latency: the gap between consecutive streamed tokens.
- **TPOT**: Time Per Output Token: a request's average decode time per token, (E2E − TTFT) ÷ (tokens − 1).
- **E2E latency**: End-to-end time for the full response ≈ TTFT + TPOT × (output tokens − 1).
- **Throughput**: Work done per second across all users, usually output tokens/s. Higher throughput means lower cost per token.
- **Goodput**: Throughput counting only requests that met their latency SLOs. Pizzas delivered hot, not just pizzas delivered.
- **SLO**: Service Level Objective: a latency or availability target, e.g. "p90 TTFT under 500 ms".
- **Percentile latency (p50/p99)**: p99 = 99% of requests are faster than this. Tail percentiles reveal spikes that averages hide.
- **Little's law**: Concurrency = throughput × latency. 20 req/s × 10 s means ~200 requests in flight.
- **MFU**: Model FLOPs Utilization: achieved FLOP/s ÷ peak. The efficiency metric for compute-bound work.
- **MBU**: Model Bandwidth Utilization: achieved bytes/s ÷ peak bandwidth. The efficiency metric for decode. 60–80% is good.
- **Cost per million tokens**: GPU $/hour ÷ (tokens/s × 3600) × 10⁶. The number that ends up on the invoice.

## Memory & KV cache

- **KV cache**: Stored key and value vectors for all previous tokens, so each decode step only computes the new token's K/V. Size = 2 × layers × KV heads × head dim × bytes, per token. Notes from earlier in the conversation, so you don't reread the whole chat every time.
- **KV head**: A key/value head. With GQA, several query heads share one, so fewer KV heads means a smaller cache.
- **KV cache quantization**: Storing K/V in FP8, INT8 or INT4 to halve or quarter cache memory and bandwidth.
- **Fragmentation**: Wasted memory: reserved-but-unused space (internal) or free gaps too small to use (external). PagedAttention mostly eliminates it.
- **Prefix caching**: Reusing KV for prompt prefixes shared across requests (system prompts, chat history), so prefill skips them.
- **RadixAttention**: SGLang's prefix cache, organized as a radix tree over token sequences, with token-level matching and natural branching.
- **Reference counting**: Tracking how many running sequences use a KV block. Only blocks with count 0 may be evicted.
- **LRU eviction**: Least-Recently-Used: when memory is needed, drop the cached block that was used longest ago.
- **Copy-on-write**: Sequences share the same physical KV blocks until one of them writes something different. Only then is a private copy made.
- **KV offloading**: Moving cold KV blocks from GPU memory to CPU RAM, SSD or remote storage, and loading them back when needed.
- **Cache hit rate**: The fraction of prompt tokens served from the prefix cache. A sudden drop often means a prompt template changed.

## Batching & scheduling

- **Batch size**: How many sequences are processed together in one forward step. Bigger batches amortize weight reads.
- **Static batching**: Form a batch, pad it, and run until every sequence finishes. Simple but wasteful. A bus that won't leave until the last passenger arrives, and won't let anyone on mid-route.
- **Dynamic batching**: Wait a short window or until the batch is full, then run. Good for fixed-cost models, but still request-level.
- **Continuous batching**: Iteration-level scheduling: sequences join and leave the running batch at every decode step. Also called in-flight batching. A conveyor-belt sushi bar: seats free up and fill instantly.
- **Ragged batch**: A batch whose sequences have different lengths, packed without padding. Kernels get per-sequence offsets.
- **Head-of-line blocking**: New work stuck waiting behind slow work already in progress.
- **Chunked prefill**: Splitting a long prompt's prefill into chunks that share engine steps with running decodes, so token streams don't freeze.
- **Token budget**: The maximum number of tokens (prefill chunks plus decodes) processed in one engine step, e.g. vLLM's max-num-batched-tokens.
- **Preemption**: Evicting a running sequence when KV memory runs out, then recomputing or swapping its cache in later.
- **Admission control**: Deciding whether to accept, queue or reject new requests, so the system doesn't overload and collapse.

## Attention

- **MHA**: Multi-Head Attention: every query head has its own key and value head. Best quality, biggest KV cache.
- **MQA**: Multi-Query Attention: all query heads share a single K/V head. The smallest cache, with some quality risk.
- **GQA**: Grouped-Query Attention: groups of query heads share a K/V head (e.g. 32 Q heads, 8 KV heads), so the cache is 4× smaller than MHA. Roommates sharing one fridge instead of each buying their own.
- **MLA**: Multi-head Latent Attention (DeepSeek): caches a compressed low-rank latent instead of full per-head K/V.
- **FlashAttention**: An IO-aware exact attention kernel. It tiles Q/K/V in on-chip SRAM and never writes the N×N score matrix to HBM.
- **Online softmax**: Computing softmax chunk by chunk with a running max and sum, rescaling earlier results as you go. This makes tiling possible.
- **PagedAttention**: Stores the KV cache in fixed-size, non-contiguous blocks, like OS virtual-memory pages, to remove fragmentation. A library storing a book's chapters on any free shelf, with an index card saying where each one is.
- **Block table**: The per-sequence map from logical KV blocks to physical memory blocks in PagedAttention.
- **Flash-Decoding**: Splits the KV sequence across thread blocks during decode, so even small batches with long contexts use the whole GPU.
- **Sliding-window attention**: Each token attends only to the last W tokens, which caps KV size per layer.

## Quantization

- **Quantization**: Storing numbers in fewer bits (8, 4…) to cut memory and bandwidth. Done well, with little quality loss. Saving a photo as a smaller JPEG.
- **Scale & zero point**: The mapping between real and integer values: q = round(x/s) + z, x̂ = s·(q − z). Symmetric quantization uses z = 0.
- **Per-group quantization**: One scale per small group of weights (e.g. 128), so an outlier only hurts its own group.
- **W4A16 / W8A8**: Notation for weight bits / activation bits. W4A16 = INT4 weights with BF16 math (weight-only). W8A8 = 8-bit weights and activations.
- **FP8 (E4M3 / E5M2)**: 8-bit floating-point formats with native tensor-core support on Hopper and newer. A common near-lossless choice.
- **FP4 (NVFP4 / MXFP4)**: 4-bit float formats with small-block scales, natively supported by Blackwell tensor cores.
- **Outliers**: A few activation channels with values 10–100× larger than the rest. They wreck naive INT8 scaling. One giant in a class photo forces everyone else to be tiny.
- **SmoothQuant**: Moves activation outliers into the weights with an equivalent per-channel rescaling, enabling W8A8.
- **GPTQ**: Post-training quantization that uses second-order (Hessian) information to compensate for each rounding error column by column.
- **AWQ**: Activation-aware Weight Quantization: scales up salient weight channels (those multiplied by large activations) before quantizing.
- **Calibration set**: A few hundred sample inputs used to measure activation ranges and statistics during post-training quantization.
- **PTQ**: Post-Training Quantization: quantizing an already-trained model without retraining it.
- **GGUF**: llama.cpp's single-file model format, with many quantization types (e.g. Q4_K_M).

## Decoding tricks

- **Speculative decoding**: Draft several tokens cheaply, then verify them all in one target-model pass. The output distribution is unchanged. An intern drafts the email; the boss approves several sentences at a glance.
- **Draft model**: The cheap proposer in speculative decoding: a small model, n-gram lookup, or extra heads.
- **Target model**: The big model whose output distribution you want. It verifies the drafts.
- **Acceptance rate**: α: the probability that a drafted token is accepted. Expected tokens per pass = (1 − α^(k+1)) ÷ (1 − α).
- **Rejection sampling**: Accept a draft token with probability min(1, p/q), otherwise resample from max(0, p − q). This makes speculation lossless.
- **EAGLE / Medusa**: Speculative methods that draft with lightweight heads attached to the target model instead of a separate model.
- **MTP**: Multi-Token Prediction: modules trained to predict several future tokens. They can serve as speculative drafts.
- **Tree attention**: Verifying a tree of alternative drafts in one pass with a special attention mask.
- **Structured output**: Constraining generation to a JSON schema or grammar by masking invalid tokens at each step.

## Parallelism

- **Tensor parallelism (TP)**: Splits each layer's weight matrices across GPUs, with all-reduces per layer. Lowers per-token latency. Four chefs each chopping a quarter of every vegetable.
- **Pipeline parallelism (PP)**: Splits a model by layers across GPUs. Works across nodes, but doesn't make a single token faster. An assembly line.
- **Pipeline bubble**: Idle time in pipeline stages while waiting for work. It's reduced by keeping several micro-batches in flight.
- **Data parallelism (DP)**: Independent model replicas with requests load-balanced across them. Scales throughput.
- **Expert parallelism (EP)**: Places different MoE experts on different GPUs and routes tokens to them with all-to-all communication.
- **MoE**: Mixture of Experts: a router sends each token to a few of many expert MLPs. Total parameters ≫ active parameters. A hospital: every patient sees only 2 specialists, but you employ 64.
- **Router (gating)**: The small network in an MoE layer that picks the top-k experts for each token.
- **All-reduce**: A collective operation that sums tensors across GPUs and gives everyone the result. TP needs two per transformer layer.
- **All-to-all**: A collective where every GPU sends a different chunk to every other GPU. Used for MoE token dispatch.
- **Context parallelism (CP)**: Splits one long sequence across GPUs (e.g. Ring Attention) for very long contexts.

## Kernels & compilers

- **Kernel**: A function launched on the GPU across a grid of thread blocks. Eager PyTorch runs roughly one kernel per op.
- **Kernel fusion**: Combining several ops into one kernel, so data stays in registers instead of round-tripping through HBM. Doing all the errands in one trip instead of three.
- **Kernel launch overhead**: A few µs of CPU time per kernel launch. Significant when a decode step launches hundreds of small kernels.
- **CUDA graph**: A recorded sequence of kernel launches that is replayed with a single launch. Removes CPU overhead in decode. A macro instead of typing every command.
- **Triton (language)**: A Python-embedded language and compiler for block-level GPU kernels. Not the same thing as Triton Inference Server.
- **torch.compile**: PyTorch's compiler. It captures the model graph and generates fused kernels (via TorchInductor/Triton).
- **CUTLASS**: NVIDIA's C++ template library (with CuTe) for building high-performance GEMM and attention kernels.
- **Nsight Systems**: NVIDIA's whole-system timeline profiler (`nsys`). Shows CPU threads, CUDA calls, kernels and the gaps between them.
- **Nsight Compute**: NVIDIA's per-kernel profiler (`ncu`): achieved bandwidth, occupancy, roofline position.

## Production systems

- **Inference engine**: Software that serves models efficiently: vLLM, SGLang, TensorRT-LLM, llama.cpp and others.
- **Disaggregated serving**: Running prefill and decode on separate GPU pools and moving the KV cache between them. Separate prep kitchen and plating line.
- **KV-aware routing**: Sending a request to the replica that already caches its longest prefix, balanced against current load.
- **Session affinity**: Routing a conversation's requests to the same replica (e.g. by hashing the session ID) to keep cache hits.
- **Autoscaling**: Adding or removing replicas based on load signals such as queue depth, KV usage and latency against SLO.
- **Cold start**: The time for a new replica to become ready: pull the image, load the weights, compile, capture graphs. Often minutes.
- **OpenAI-compatible API**: The de-facto HTTP interface (/v1/chat/completions) that most engines expose, so any OpenAI client works.
- **SSE streaming**: Server-Sent Events: the server pushes each token to the client as soon as it's generated.
- **Capacity planning**: Estimating how many GPUs you need from traffic, token lengths, SLOs and benchmarked per-GPU capacity.
- **Load test (open-loop)**: Sending requests at a target arrival rate whether or not earlier ones have finished, like real traffic. It exposes queueing collapse.
- **Backpressure**: Signalling clients to slow down (HTTP 429/503) instead of queueing work forever.
- **Graceful drain**: On deploy or shutdown: stop admitting new requests, let in-flight streams finish, then restart.
- **Multi-LoRA serving**: Serving many small LoRA fine-tune adapters on one shared base model at the same time.
