# System design

## Why expert streaming works

Nemotron-H checkpoints combine Mamba and attention sequence-mixing layers with
dense or mixture-of-experts (MoE) feed-forward layers. In an MoE model, most
of the weight bytes are in expert feed-forward networks. Each MoE layer has
many experts, but a router selects only the top-K experts needed for each
token. The other experts do no work for that token.

This divides the model into two kinds of data:

- Common weights are needed continuously: embeddings, attention, routers,
  normalization, and output layers. They remain in unified memory.
- Routed expert weights are needed only when selected. They can be kept in a
  bounded memory cache and reloaded from the original checkpoint on demand.

The KV cache, current activations, and MLX working memory also remain in
unified memory. Streaming reduces expert-weight residency; it does not remove
those other memory costs.

## Streaming inference flow

Before inference, `ModelOptIndex` reads the checkpoint configuration and
safetensors headers. It records the file and byte range of every expert tensor
without loading the tensor data. mlx-lm then loads the common model weights.
The index validates each layer's expert layout once and records its projection
formats; loading and computation reuse that metadata.

Each MoE layer receives an `ExpertPool`: a fixed number of memory slots used as
that layer's expert cache. One slot holds one expert's NVFP4 `up_proj` and
`down_proj` weights and scales. Pools are independent because expert 10 in one
layer has different weights from expert 10 in another layer.

Inference then follows this path:

1. Prefill begins by processing a chunk of prompt tokens.
2. At each MoE layer, the router selects top-K expert IDs for those tokens.
3. The layer's pool checks which selected experts are already cached.
4. Missing experts are read from their original safetensors byte ranges and
   copied into cache slots in unified memory.
5. Router expert IDs are remapped to their current slot IDs.
6. The NVFP4 expert computation runs from the cached slots.
7. Decode generates one token at a time and repeats the same lookup at every
   MoE layer.

The cache remains populated across prefill, decode, and repeated `--runs` in
the same process. Later tokens therefore reuse experts selected earlier.

## Cache replacement and reloads

Each pool is a least-recently-used (LRU) cache. A hit marks an expert as most
recently used. A miss takes an unused slot; once the pool is full, it evicts
the least-recently-used expert that is not also required by the current
forward pass. Protecting the current expert set prevents one load from
evicting another expert before the computation uses it.

A load is classified as:

- A cold miss when that layer/expert pair has never been loaded in this model
  process.
- A capacity miss when it was loaded before, evicted for space, and must now
  be reloaded.

Every miss loads an expert, so `misses = cold_misses + capacity_misses`.
Knowledge of previously seen experts persists across runs solely to preserve
that classification. Resetting statistics does not flush the cache.

`ExpertReader` performs the reload directly from the original checkpoint. It
groups nearby tensor ranges into larger reads and uses `--workers` reader
threads for independent ranges. The fill is synchronous: expert computation
waits until the selected experts are in their slots. There is currently no
prefetch or overlap between file I/O and expert computation.

`--expert-budget-gib` is the total expert-cache budget across all MoE layers.
It determines one common slot count per layer, and must provide at least top-K
slots in every layer.

## Resident mode

Resident mode has enough memory for every expert. At startup, it loads every
expert's weights and scales into a complete bank of arrays for each MoE layer
and calls `mx.eval` so those arrays are materialized before inference. The
checkpoint files are then closed.

The arrays are indexed by expert ID: expert 347 is stored at index 347. When
the router selects experts 17 and 347, the NVFP4 computation reads indices 17
and 347 directly. By contrast, a streaming cache might currently store expert
347 in slot 12 and must look up that mapping first.

This direct resident path avoids copying router IDs from Metal to the host,
checking the cache, updating the LRU, remapping IDs, evicting experts, or
reading checkpoint data. Those checks can disrupt the inference hot path even
when every requested expert is already cached.

`auto` selects resident mode when the checkpoint source bytes fit within 75%
of physical memory after a 4 GiB reserve; otherwise it selects streaming.
For checkpoints with routed experts, `--expert-mode resident` and
`--expert-mode stream` override that choice.

## Dense-only checkpoints

A Nemotron-H or Llama checkpoint with no routed experts has nothing to cache or
stream. All weights are loaded into unified memory, so the runtime reports
resident mode and builds no `ExpertPool` objects. The loader keeps NVFP4 weights
packed, converts FP8 weights to BF16, and leaves BF16 weights unchanged.

Expert-cache budgets, workers, and statistics have no work to perform for a
dense-only checkpoint. `--compile` currently compiles only routed-expert
computation, so dense-only runs report compilation as disabled.

## Prefill and compilation

Resident prefill defaults to 2048 tokens. Streaming prefill defaults to
`floor(slots_per_layer / top_k)`, limiting the worst-case selected expert set
to the cache capacity. `--prefill-chunk` overrides either default.

`--compile` compiles only the numerical NVFP4 expert computation. Routing,
cache management, reloads, remapping, and statistics stay outside the compiled
function. Cache slots are fixed MLX tensors, so their contents can change
without rebuilding the compiled function.

The complete Nemotron-H model is not compiled because generation mutates
Mamba state and, in hybrid models, attention KV state. Those cache objects are
not pure MLX array inputs and outputs, so compiling the model call would capture
stale state rather than preserve token-to-token updates.

MLX compilation is lazy. The first inference run includes JIT compilation for
the shapes it encounters, so its timing contains both compilation and
execution. Later `--runs` in the same process reuse those compiled programs.
Running at least twice makes the distinction visible: run 1 reports cold-JIT
performance, while run 2 reports warmed compiled inference for the same
shapes. The model and expert pools remain loaded between those runs.

## Reported data

Every run reports:

- `hits`, `misses`, and `evictions` for the expert pools.
- `bytes_read`, `useful_bytes`, and `reads` for checkpoint I/O. The difference
  between bytes read and useful bytes is the cost of coalescing nearby ranges.
- Prompt and generation token counts and throughput.
- Peak memory, model load time, run time, cumulative process time, finish
  reason, run number, and whether compilation was enabled.

`--expert-stats` resets the cache and I/O counters before each run and adds:

- `cold_misses` and `capacity_misses`.
- `experts_loaded`, the number of occupied layer cache slots at the end of the
  run.
- `unique_experts_accessed`, the number of distinct layer/expert pairs used by
  a streaming run.

Resident mode reports all experts loaded and zero inference-time cache
activity. It omits `unique_experts_accessed` because resident inference does
not copy router results to the host merely to collect statistics.

Dense-only runs report zero expert and `ExpertReader` I/O activity because
neither subsystem exists; those zeros do not mean the checkpoint was not
loaded.

## Scope

The loader supports dense, MoE, and mixed `model_type=nemotron_h` checkpoints,
plus dense `model_type=llama` and `model_type=qwen2` checkpoints. Metadata may
declare `NVFP4`, `MIXED_PRECISION`, or `FP8`; routed experts must use the NVFP4 tensor layout.
Static FP8 weights are expanded to BF16 once while loading, and BF16 weights
retain their checkpoint dtype. The loader reads checkpoint shards in place and
never rewrites the checkpoint.

NVIDIA's `qwen3_5_moe` NVFP4 export is supported for text generation. Its
three expert projections use SwiGLU and the same cache and disk reader.
Per-module `W4A16_NVFP4` metadata disables activation rounding while keeping
weights packed; `NVFP4` retains the activation-rounding path. Qwen's attention,
recurrent state, shared experts, and checkpoint sanitization use MLX-LM.
