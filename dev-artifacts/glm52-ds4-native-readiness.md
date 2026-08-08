# DS4-Native GLM 5.2 TP4 Readiness Inventory

Date: 2026-08-06

Goal: make DS4 a lighter, narrower, faster serving engine for GLM 5.2 on four GX10 ranks. Bird/vLLM remains the topology/correctness oracle, not the implementation model.

## Current Readiness Verdict

BCD contract status: `READY_TO_START_IMPLEMENTATION`

Native implementation status: `NOT_READY`

Native DS4 has GLM 5.2 model knowledge and GLM decode/prefill kernels, but it does not yet have the runtime, checkpoint, collectives, or CUDA GLM TP path required for four-machine TP4+DCP4 serving.

Current BCD proof hash:

- blueprint: `e9e9e06f90600d5fbb1353d1f669ff9270095bb7f9c6eeb1e204cfd9d283d97a`
- semantic: `224e0e716e87f2c8f3f138b5e2fec753ef9f81c01a72adc2b686b5d2a9079dea`
- decode simulation: `dev-artifacts/simulation-decode-current-full-trace-v2.json`, PASS, `b929d1c92edeac66e6c35dedf959a33d9996607a5cf33ee351335ff221d860ce`

The current repo has three relevant execution modes:

- GLM graph backend: real GLM 5.2 decode/prefill implementation over Metal/CUDA/ROCm.
- Mac-to-Mac TP: two-rank lockstep tensor parallelism over Metal, with a 50/50 expert split and pairwise gates.
- CUDA TP: intra-process multi-GPU path for DeepSeek V4 Flash, explicitly not GLM.

The desired target is a fourth mode: four rank processes across four GX10 machines, every rank running every GLM layer in lockstep, with TP4 query-head/expert/vocab ownership and DCP4 sparse-cache ownership.

## Hard Evidence From Current Code

### GLM architecture is modeled

DS4 encodes GLM 5.2 as 79 blocks in the local execution shape, with 78 normal transformer layers plus one nextn/MTP block, hidden size 6144, vocab 154880, 64 query heads, one KV head, q LoRA 2048, KV LoRA 512, 256 routed experts, top-8 expert routing, one shared expert, and top-k 2048 indexer selection.

Code seams:

- `DS4_SHAPE_GLM52` in `/Users/helium/src/ds4-gb10x4-bcd/ds4.c`.
- `config_validate_glm_dsa_model()`.
- `weights_validate_glm_dsa_layout()`.

### GLM MLA representation is present

DS4 validates:

- `attn_q_b`: `[q_lora_rank, n_head * key_mla]`
- `attn_kv_a_mqa`: `[embd, head_dim]`
- `attn_k_b`: `[q_nope, kv_lora, n_head]`
- `attn_v_b`: `[kv_lora, value_mla, n_head]`
- `attn_output`: `[n_head * value_mla, embd]`

This confirms the TP split must be query-head-local reconstruction from shared KV latent, not four KV-head groups.

Code seams:

- `weights_validate_glm_dsa_layout()`
- `layer_kv_projection_normed_one()`
- `glm_k_b_project_f32_ref()`
- GLM GPU graph fields `layer_kv_lora_cache[]` and `layer_k_rope_cache[]`.

### Existing network TP is two-rank Metal only

`ds4_tp.h` and `ds4_tp.c` define a two-rank leader/worker protocol: rank 0 mirrors session commands to rank 1, then pairwise exchanges per-layer partial vectors. Public docs also state it is one 50/50 worker.

Code seams:

- `ds4_tp_options`: only leader/worker roles.
- `ds4_tp_create()`: rank is `0` for leader, `1` for worker.
- `ds4_tp_slab_bytes()`: one out vector and one in vector per gate.
- `ds4_tp_gate_exchange()`: pairwise exchange.
- `ds4_engine_tp_bind()`: compiled only for Apple/Metal and logs "50/50 expert split".

### CUDA TP rejects GLM

`--cuda-tensor-parallel` currently requires an even multi-GPU CUDA placement, but after model validation it rejects non-DeepSeek models.

Code seam:

- `ds4_engine_open_internal()`: `--cuda-tensor-parallel is currently supported only for DeepSeek models`.

This means four GX10 support needs a new GLM CUDA TP path. It cannot use the current DeepSeek CUDA TP path unchanged.

### GLM TP paths are partial and two-way

The GLM graph has some TP-aware logic, but it is all `tp_world == 2`.

Examples:

- GLM graph fields describe "50/50 expert sharding".
- Sparse FFN decode gates add two partials.
- Batch/prefill FFN gates exchange one peer partial.
- Indexed prefill attention can zero unowned heads and exchange one partial, but only when `g->tp_world == 2`.
- Session setup assigns `s->glm_graph.tp_world = 2`.

This is useful prior art, but the ownership and collective model must become rank-count aware.

### DCP4 is missing as a first-class runtime concept

Bird/vLLM uses `--decode-context-parallel-size 4`, `VLLM_DCP_GLOBAL_TOPK=1`, and `VLLM_DCP_SHARD_DRAFT=1`. DS4 has compact KV caches and sparse indexed attention, but no four-rank context-shard ownership, global sparse top-k, or cross-rank selected-row read contract.

Code seams:

- `layer_kv_lora_cache[]`
- `layer_k_rope_cache[]`
- `layer_indexer_key_cache[]`
- `indexer_selected`
- `batch_indexer_selected`
- decode `ds4_gpu_indexer_topk_tensor(...)`
- indexed attention calls consuming selected rows.

## What We Need To Build

### 1. Fixed Four-Rank Process Topology

Build a DS4-native rank group, not a generic distributed framework.

Minimum target:

- Rank 0: API, tokenizer/session owner, sampler, collective coordinator.
- Ranks 1-3: headless workers.
- All ranks: load their local shard, own a TP rank id, execute every GLM layer in the same order.
- One active decode stream first; batching can come later.

New or changed seams:

- Extend `ds4_tp_options` or introduce `ds4_rank_options`.
- Add `--tp-size 4`, `--rank N`, `--master-addr`, `--master-port`, and local
  fabric address options. For the four-GX10 target, the collective endpoint is
  the CRS812 200G data-plane address (`10.100.185.x`), not the `192.168.0.x`
  management SSH address.
- Replace leader/worker-only `ds4_tp_create()` with a four-member rendezvous.
- Keep `ds4_distributed.c` separate; it is layer pipeline, not this target.

### 2. Group Collectives

The current pairwise slab is not enough. TP4 needs named group collectives.

Minimum collectives:

- request/token broadcast from rank 0 to ranks 1-3;
- hidden-state broadcast or replicated checkpoint initialization;
- ATTN all-reduce sum over four rank-local partials;
- FFN all-reduce sum over four routed/expert partials;
- DCP sparse-index global top-k merge;
- selected-row exchange/read for DCP-owned context shards;
- logits gather or distributed top-k to rank 0;
- MTP draft/commit broadcast.

Implementation direction:

- On GX10, use NCCL/RoCE if available.
- If avoiding NCCL to stay lightweight, implement a small TCP/RDMA ring/tree for fixed f32 vectors and top-k records, but treat that as a serious performance risk.
- The first practical milestone should use NCCL for all-reduce/gather and a simple control socket for commands.

### 3. Checkpoint Format Decision

This is a major design choice.

Current DS4 consumes tested GLM GGUF layouts. Bird's validated four-GX10 baseline uses vLLM sharded-state safetensors with 20 base shards plus one MTP shard per rank.

To stay lightweight, the best DS4 route is likely not a full Hugging Face/vLLM loader. Choose one:

- Preferred DS4-native route: write an offline converter that emits four rank-local DS4 GGUF-pack shards plus a manifest. Runtime stays small and validates rank ownership before mapping tensors.
- Alternate route: implement a narrow safetensors sharded-state loader for the Bird artifact naming pattern only. This is faster to align with Bird but pulls DS4 away from its GGUF-native loader design.

Required manifest fields:

- `model_id`, source config hash, tokenizer/config hashes;
- `tp_size = 4`, `pp_size = 1`, `dcp_size = 4`;
- rank id and expected rank-local files;
- tensor ownership spans for Q heads, `k_b`, `v_b`, output/vocab, experts, and MTP/speculator;
- per-file byte length and SHA-256;
- rejection rule for foreign-rank tensors.

### 4. GLM QKV TP4

Required rank ownership:

- Q heads: 16 query heads per rank.
- `q_a`: replicated or reduce-scattered intermediate; simplest first pass replicates `q_a`.
- `q_b`: rank-local columns only for owned 16 heads.
- `kv_a_mqa`: shared latent projection; either replicated on every TP rank or produced once and broadcast. Replication is simpler and likely acceptable because KV latent is only 512 plus RoPE K.
- `k_b`/`v_b`: rank-local head slices for owned query heads.
- KV cache: store `kv_lora[512]` and `k_rope[64]` under DCP ownership.

Code seams:

- `glm_graph_matmul_q8_0_tensor()` around q path and kv path.
- `ds4_gpu_glm_store_compact_kv_tensor()`.
- `ds4_gpu_glm_attention_qk_low_batch_tensor()` or equivalent qk-low kernels.
- `ds4_gpu_glm_attention_indexed_*`.
- `ds4_gpu_glm_value_project_*`.

### 5. DSA/IndexShare + DCP4

This is the highest correctness risk.

Required behavior:

- Preserve `DS4_N_INDEXER_TOP_K = 2048`.
- Respect GLM 5.2 full/shared indexer schedule.
- Compute or reuse selected sparse indices exactly according to the model schedule.
- With DCP4, each rank scores/selects over its context shard, then a global top-k merge determines the final selected rows.
- Attention must read selected rows from the owning DCP shard, or run an equivalent all-gather of selected KV latent/rope rows.

Smallest practical first implementation:

- Store DCP-owned compact rows by token position modulo/range partition.
- Compute local top-k candidates per rank.
- Gather top-k candidates to rank 0 or all ranks.
- Merge to global top-k.
- Broadcast selected row descriptors.
- Fetch/gather selected `kv_lora` and `k_rope` rows needed by local query heads.

Validation needed:

- Compare selected row ids against Bird for small and long prompts.
- Compare attention output per layer before output projection.
- Include prompts where the useful row is outside local top-k unless global merge is correct.

### 6. Attention Output TP4

Per rank:

- compute local attention for 16 query heads;
- project rank-local heads through local output shard/slice;
- produce a rank-local `hidden[6144]` partial;
- all-reduce sum across four ranks;
- apply residual/norm in the same order on every rank.

Code seams:

- GLM indexed attention calls.
- `attn_output` projection call sites.
- Existing two-rank `glm_graph_tp_batch_ffn_combine()` can inspire a generic `glm_graph_tp_allreduce_hidden()`.

Do not infer correctness from local tensor shapes only. The contract must declare expected rank-local head range and expected all-reduce result.

### 7. MoE TP4

Current GLM Metal TP does 50/50 expert ownership. The observed Bird/vLLM
GLM 5.2 TP4 artifact does not split the routed-expert id dimension four ways:
each rank-local shard keeps all 256 expert ids and shards expert matrix
dimensions. The native DS4 contract should therefore treat expert ids as
present on every rank, while FFN computation remains tensor parallel and
requires an all-reduce to restore replicated hidden.

Minimum first split:

- Every rank has expert-id coverage `[0,256)`.
- Each rank owns its local tensor-dimension shard of routed/shared expert
  matrices.
- Router selection is replicated on all ranks.
- Each rank computes its tensor partial for selected experts.
- FFN all-reduce sum restores replicated hidden.

Code seams:

- `glm_graph_routed_moe_one_dispatch()`.
- `glm_graph_routed_moe_batch_dispatch()`.
- Metal kernel helper `ds4_tp_owns_expert()` already supports generic `tp_world`; many C call sites still gate only on `tp_world == 2`.
- CUDA/ROCm MoE launch paths need equivalent ownership-aware selected-expert execution for GLM.

### 8. Vocab/Logits

For the lightweight first target:

- shard vocab/output head four ways;
- each rank computes its logits shard;
- rank 0 gathers logits or gathers top-k candidates;
- rank 0 samples and broadcasts the accepted token.

Do not replicate the full output head if the memory goal depends on rank-local sharding. Current two-Mac GLM explicitly leaves output unsplit; that should change for four-GX10.

Code seams:

- `ds4_engine_tp_vocab_split()`.
- `tp_logits_half`.
- `ds4_tp_send_logits_half()`.
- CUDA output TP helper `engine_cuda_tp_output_shard_span()` already partitions by arbitrary `ways`, but the GLM path cannot reach it today because CUDA TP rejects GLM.

### 9. MTP/Speculation

Bird uses a matched DSpark speculator and DCP draft sharding. DS4 has GLM nextn/MTP logic, but it is not a TP4+DCP4 speculator.

First milestone options:

- Milestone A: disable speculation and validate base decode TP4 first.
- Milestone B: enable GLM nextn/MTP after base decode matches, with rank-local MTP shards and all required collectives.
- Milestone C: match Bird DSpark speculator behavior.

Recommendation: make MTP a separate BCD child graph and do not let it block base TP4 correctness.

### 10. Server Integration

The first server should be narrower than vLLM:

- one served model;
- one rank group;
- one active request/session initially;
- fixed max model length configured at startup;
- no broad scheduling, adapters, LoRA, tool parser framework, or arbitrary model loader.

Code seams:

- `ds4_server.c`
- `ds4_web.c`
- `ds4_session_sync()`
- `ds4_session_eval_internal()`
- rank0 sampler path.

## Suggested Milestones

### M0: Contract Freeze

Update BCD with leaf graphs for:

- rank-group formation;
- rank-shard loading;
- QKV/MLA TP4;
- DSA IndexShare + DCP4;
- attention all-reduce;
- MoE all-reduce;
- logits gather/top-k;
- MTP disabled exit.

### M1: Rank Runtime Skeleton

Start four DS4 processes, rendezvous, verify config hashes, rank ids, tp/dcp sizes, and clean shutdown. No model execution yet.

### M2: Rank-Local Loader

Load rank-local GLM 5.2 shards with manifest validation. Reject missing shards, foreign rank shards, wrong config hash, and unsupported tensor layout.

### M3: Base Decode Without DCP

Run one-token decode at short context with replicated KV cache, TP4 query-head split, MoE 4-way ownership, attention/FFN all-reduce, and logits gather. This proves TP4 math before DCP.

### M4: DCP4 Sparse Attention

Partition KV/cache rows across ranks, add global sparse top-k merge and selected-row gather. Validate against Bird at selected-index and per-layer attention-output boundaries.

### M5: Prefill

Add chunked prefill for the same TP4+DCP4 rules. Keep one active request.

### M6: MTP

Add nextn/speculator path and verify commit/rollback under TP4+DCP4.

### M7: Server Hardening

Add health, error propagation, rank restart policy, long-context smoke, and performance counters.

## Minimal Validation Matrix

- Config/manifest: rank mismatch, tp size mismatch, wrong config hash, missing shard, foreign shard.
- Shape: Q head ranges, `k_b/v_b` slices, vocab shard spans, expert ranges.
- Runtime: four-rank rendezvous, barrier, all-reduce, gather, broadcast timeout.
- Math: per-layer Q, KV latent, selected indices, attention output partial, attention all-reduce, FFN partial, FFN all-reduce, logits.
- End-to-end: greedy continuation against Bird for short prompts, long memory prompts, and code prompts.
- Performance: startup time, resident memory per rank, prefill tok/s, decode tok/s, TPOT, DCP selected-row bytes/token, all-reduce bytes/token.

## Current BCD Hierarchy

The current BCD graph is no longer only conceptual. It now expands `decode token j` into explicit contracts for:

1. `model-load`: validate launch plan, validate shard manifest, map rank-local shards, and publish rank engines.
2. `prefill`: tokenize, chunk, run TP layer prefill, all-reduce hidden, commit DCP-compatible prompt KV, and publish decode handoff.
3. `decode-qkv-mla`: project rank-owned Q heads, produce shared MLA latent/RoPE row, and append position-j cache metadata.
4. `decode-dcp-topk`: merge sparse-index scores across DCP4 shards and return globally selected rows.
5. `decode-sparse-attention`: run rank-local sparse attention for owned query heads and produce an additive hidden partial.
6. `decode-expert-route`: compute replicated top-8 routed expert selection.
7. `decode-ffn-local`: run rank-owned routed/shared FFN work and produce an additive hidden partial.
8. Parent `decode`: keep ATTN all-reduce, FFN all-reduce, vocab shard, logits gather, sampling, and cursor update ordered at the composite level.

Each leaf should have typed ports for rank-local shard, rank-local partial, replicated hidden, leader-only token/logits, and DCP-owned cache rows. No derivation from edges should replace expected typed port declarations; the validator should reject dead-end or accidentally inferred ports.
