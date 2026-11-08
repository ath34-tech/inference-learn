# Kernels, Fusion & Compilation

> Lesson 4-1 · WORLD 4 Expert / Systems · ~35 min · topics: Kernel fusion, CUDA graphs, Triton & CUTLASS, Profiling (MFU / MBU)

> previous: [3-4 Scheduling & Disaggregated Serving](../lessons/11-disaggregation.md) · next: [4-2 Serving in Production](../lessons/13-production.md)

> Part of **InferenceLearn**, a free course on LLM inference engineering. Web version: lessons/12-kernels.html. Whole course in one file: llms-full.txt

Every technique so far eventually runs as GPU kernels. At this level you stop treating PyTorch ops as black boxes and start asking: how many bytes does this kernel move, how many times do we launch it, and what is the CPU doing in the meantime?

#### By the end you'll be able to

- Explain why kernel fusion speeds up memory-bound code
- Use CUDA graphs to remove launch overhead in decode
- Write a simple fused kernel in Triton
- Profile a decode step and compute MBU/MFU

#### Key vocabulary

- **[Kernel](../vocabulary.md#v-kernel)**: A function launched on the GPU across a grid of thread blocks. Eager PyTorch runs roughly one kernel per op.
- **[Kernel fusion](../vocabulary.md#v-kernel-fusion)**: Combining several ops into one kernel, so data stays in registers instead of round-tripping through HBM.
- **[Kernel launch overhead](../vocabulary.md#v-kernel-launch-overhead)**: A few µs of CPU time per kernel launch. Significant when a decode step launches hundreds of small kernels.
- **[CUDA graph](../vocabulary.md#v-cuda-graph)**: A recorded sequence of kernel launches that is replayed with a single launch. Removes CPU overhead in decode.
- **[Triton (language)](../vocabulary.md#v-triton-language)**: A Python-embedded language and compiler for block-level GPU kernels. Not the same thing as Triton Inference Server.
- **[torch.compile](../vocabulary.md#v-torch-compile)**: PyTorch's compiler. It captures the model graph and generates fused kernels (via TorchInductor/Triton).
- **[CUTLASS](../vocabulary.md#v-cutlass)**: NVIDIA's C++ template library (with CuTe) for building high-performance GEMM and attention kernels.
- **[Nsight Systems](../vocabulary.md#v-nsight-systems)**: NVIDIA's whole-system timeline profiler (`nsys`). Shows CPU threads, CUDA calls, kernels and the gaps between them.

[Drill all terms as flashcards ▶](../vocabulary.md#flashcards)

## What a kernel is

A **kernel** is a function launched on the GPU across a grid of thread blocks. Eager PyTorch launches roughly one kernel per op. For an op like `y = silu(gate) * up`, that's three kernels: `silu`, `mul`, and a copy or cast. Each one reads its inputs from HBM and writes its outputs back to HBM.

## Kernel fusion

Elementwise ops do 1 or 2 FLOPs per element, so they're completely memory-bound. Fusing a chain of them into one kernel means data is read from HBM **once**, all the math happens in registers, and the result is written **once**.

```
unfused: read silu write read mul write
fused: read silu·mul write
```

*Diagram: same math, less than half the HBM traffic, and one launch instead of two*

Common fusions in LLM inference:

- **RMSNorm + residual add** (and optionally quantizing the output to FP8)
- **SwiGLU**: `silu(gate) * up`, often with the gate and up projections merged into one GEMM
- **QKV projection** as one GEMM, plus **rotary embeddings** fused into the attention kernel or the KV-cache write
- **Dequantization fused into GEMM** (W4A16 kernels like Marlin)
- **Attention** itself: FlashAttention is a big fusion of matmul, softmax and matmul
- **MoE**: routing, permutation and grouped GEMMs fused together

## Triton: kernels in Python

**Triton** lets you write block-level GPU kernels in a Python-like language. You describe what a block of data does, and the compiler handles thread-level details, memory coalescing and shared memory. Here is a fused SiLU-and-multiply:

```
import torch, triton
import triton.language as tl

@triton.jit
def silu_mul_kernel(gate_ptr, up_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    g = tl.load(gate_ptr + offs, mask=mask).to(tl.float32)
    u = tl.load(up_ptr + offs, mask=mask).to(tl.float32)
    y = g * tl.sigmoid(g) * u                      # all in registers
    tl.store(out_ptr + offs, y.to(out_ptr.dtype.element_ty), mask=mask)

def silu_mul(gate, up):
    out = torch.empty_like(gate)
    n = gate.numel()
    grid = (triton.cdiv(n, 1024),)
    silu_mul_kernel[grid](gate, up, out, n, BLOCK=1024)
    return out

g = torch.randn(64, 14336, device="cuda", dtype=torch.bfloat16)
u = torch.randn_like(g)
torch.testing.assert_close(silu_mul(g, u), torch.nn.functional.silu(g) * u, rtol=1e-2, atol=1e-2)
```

Other tools you'll meet: **CUDA C++** for full control, **CUTLASS / CuTe** (NVIDIA's templates for high-performance GEMMs and attention), and newer tile-based DSLs. For GEMMs you'll almost always call cuBLAS, CUTLASS or a vendor library rather than writing your own.

## torch.compile

`torch.compile` traces your PyTorch model into a graph and uses **TorchInductor** to generate fused Triton kernels automatically. Elementwise and reduction chains get fused with little effort. Engines like vLLM use it for the model's "glue" ops, alongside hand-written kernels for attention and GEMMs.

```
model = torch.compile(model, mode="max-autotune-no-cudagraphs")   # or let the engine manage it
```

> **WATCH OUT:** Dynamic shapes (batch size and sequence length change every step) can cause recompilations. Engines mark dimensions as dynamic or compile for a set of bucketed sizes.

## CUDA graphs

One decode step for a 32-layer model launches hundreds of kernels. If each launch costs ~5 µs of CPU time, that's on the order of a millisecond of pure overhead on a step that might take 5–10 ms. At small batch, the GPU can end up **waiting for the CPU**.

**CUDA graphs** record the whole sequence of kernel launches once and then **replay** it with a single launch:

```
# capture once (static input/output buffers!)
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    static_out = model.decode_step(static_input_ids, static_positions)

# every step: copy new data into the static buffers, then replay
static_input_ids.copy_(next_ids)
static_positions.copy_(next_pos)
g.replay()
```

- Graphs need **fixed shapes and fixed memory addresses**, so engines capture one graph per batch size bucket (1, 2, 4, 8, … 256) and pad up to the nearest one.
- They're mostly used for **decode**. Prefill shapes vary too much, and prefill is compute-bound anyway, so launch overhead matters less there.
- Capturing takes time and memory at startup. That's part of why engine startup is slow.

## Overlapping CPU & GPU

Even with graphs, the CPU must schedule the next step, prepare inputs, sample, detokenize and stream. Modern engines move toward **asynchronous scheduling**: prepare step N+1 on the CPU while the GPU runs step N, keep sampling on the GPU, and push detokenization and HTTP handling into separate processes. vLLM's V1 architecture and SGLang's overlap scheduler are both built around this.

## Profiling

- **PyTorch Profiler**: op-level timeline. A good first look.
- **Nsight Systems** (`nsys`): whole-system timeline across CPU threads, CUDA API calls, kernels and NCCL. Gaps between kernels show CPU overhead.
- **Nsight Compute** (`ncu`): deep dive on one kernel: achieved bandwidth, occupancy, tensor-core use, roofline position.

```
nsys profile -o decode_trace --trace=cuda,nvtx,osrt python bench_decode.py
ncu --set full -k regex:attention -o attn_report python bench_decode.py
```

## Check yourself

**Why does fusing RMSNorm with the residual add help even though it saves no FLOPs?**

Both ops are memory-bound. Fusion removes an HBM round-trip of the hidden-state tensor (one write and one read) plus a kernel launch. Time is set by bytes moved, not by FLOPs.

**Why do engines capture CUDA graphs for batch sizes 1, 2, 4, 8… rather than every size?**

Each graph has fixed shapes and uses memory. Capturing every size would cost too much startup time and memory. Padding to the next bucket wastes a little compute but keeps the number of graphs small.

**An nsys trace shows 2 ms gaps between decode steps where no kernels run. What's likely happening?**

CPU-side work is blocking the GPU: scheduling, input preparation, sampling copied back to the host, detokenization, or Python overhead. Fixes: async scheduling, keep sampling on the GPU, CUDA graphs, and move work off the critical path.

## Further reading

- [Triton tutorials](https://triton-lang.org/main/getting-started/tutorials/index.html) (start with vector add, then fused softmax and matmul)
- [GPU MODE lectures](https://github.com/gpu-mode/lectures), a community series on CUDA, Triton and profiling
- [Accelerating Generative AI with PyTorch II: GPT, Fast](https://pytorch.org/blog/accelerating-generative-ai-2/) (torch.compile, quantization, speculative decoding)
- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)

## Interview corner

Questions on this lesson that come up in real interviews. Answer out loud first, then open the card.

**[mid] What are CUDA graphs and why do inference engines use them?**

#### Strong answer

They record a sequence of kernel launches with fixed shapes and addresses, then replay it with one launch. Small-batch decode steps are short (milliseconds) but launch hundreds of kernels, so CPU launch overhead can rival GPU time. Graphs remove it. Engines capture one graph per batch-size bucket and pad up, mostly for decode. The costs are capture time, memory, and rigid shapes.

#### What they're testing

Understanding CPU vs GPU overheads.

#### Follow-ups

- Why aren't graphs typically used for prefill?

**[mid] Give an example of kernel fusion in an LLM and explain the speedup.**

#### Strong answer

SwiGLU: silu(gate) × up. Unfused, it's separate kernels that each read and write the full [tokens × 14336] tensors to HBM. Fused, it's one read of each input and one write of the output. Elementwise work is memory-bound, so time drops roughly in proportion to the bytes saved, plus launches. Other examples: RMSNorm+residual(+FP8 quant), rotary fused into attention, dequant fused into GEMM.

#### What they're testing

Bytes-moved reasoning at the kernel level.

#### Follow-ups

- Sketch that kernel in Triton.

**[senior] A decode step takes 2× longer than your speed-of-light estimate. How do you find out why?**

#### Strong answer

First check the estimate itself: include KV bytes at the actual context lengths and the communication. Then take an Nsight Systems trace and look for gaps between kernels (CPU-bound: scheduling, sampling sync, Python, missing CUDA graphs), many tiny kernels (missing fusion), and slow collectives. Then use Nsight Compute on the top kernels for achieved bandwidth vs peak, occupancy and wave quantization. Compute MBU per kernel class and fix the biggest gap first.

#### What they're testing

A systematic performance-debugging method.

#### Follow-ups

- Which GEMM shapes in decode tend to underutilize the GPU?

[All interview questions ▶](../interview.md) · [Quiz Arcade ▶](../practice.md#arcade) · [Workbook ▶](../workbook.md)
