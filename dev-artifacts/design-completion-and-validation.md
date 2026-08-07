# GLM 5.2 GX10x4 BCD Design Completion

Date: 2026-08-06

Blueprint: `ds4-arch`

Current digests:

- blueprint: `e9e9e06f90600d5fbb1353d1f669ff9270095bb7f9c6eeb1e204cfd9d283d97a`
- semantic: `224e0e716e87f2c8f3f138b5e2fec753ef9f81c01a72adc2b686b5d2a9079dea`

## Completed Design Scope

The BCD hierarchy now covers the target system from serving lifecycle through model load, prefill, and token-j decode:

- `serve`: root GX10x4 service, rank hosting architecture, request/prefill/decode/stream/checkpoint order, model/TP/KV/cursor/rank-plan state.
- `model-load`: launch-plan validation, TP4 shard manifest validation, local shard mapping, rank-engine readiness.
- `prefill`: request tokenization, bounded chunk planning, TP layer execution, prefill all-reduce, DCP-compatible KV commit, cursor commit, decode handoff.
- `decode`: composite token-j workflow with explicit step order, cursor read/update, ATTN/FFN collectives, logits gather, and child expansions.
- `decode-qkv-mla`: GLM 5.2 MLA/QKV rank-head split with Q heads sharded 16 per rank and K reconstructed from shared `kv_lora[512]` plus `k_rope[64]`.
- `decode-dcp-topk`: DCP4 global sparse top-k merge and selected-row ownership.
- `decode-sparse-attention`: rank-local attention for owned query heads.
- `decode-expert-route`: replicated top-8 routed expert selection.
- `decode-ffn-local`: rank-owned expert/shared FFN partial production.

MTP/speculation is explicitly deferred from the base TP4+DCP4 token-j path. It should be added as its own child graph after base decode and prefill match the Bird/vLLM oracle.

## Mechanical Validation

Current evidence files:

- `dev-artifacts/validation-lint-current.json`: PASS, no warnings.
- `dev-artifacts/validation-serve-l0-current.json`: PASS, no warnings.
- `dev-artifacts/validation-decode-composite-current.json`: PASS, no warnings.
- `dev-artifacts/validation-leaves-current.json`: PASS, no warnings.

The deterministic validator confirms typed ports are authored, not derived from graph edges, and the graph rejects dead-end action structure under its profiles.

## Simulation Validation

Current evidence:

- `dev-artifacts/simulation-decode-current-full-trace-v2.json`: PASS, no warnings.

The passing decode simulation covers the token-j TP4/DCP4 trace, all active decode edges, all parent-conserved decode obligations, all produced DATA field/property flows, terminal state observations, cursor read/update semantics, and the bounded layer loop.

## Semantic Review

No P0/P1 semantic blockers remain in the design itself.

The core architecture is coherent for TP rather than pipeline parallelism: all four ranks execute every layer in the same order, own rank-local Q/head/expert/vocab/DCP spans, and synchronize only at explicit collectives. The Q/K design is feasible for GLM 5.2 because Q is query-head sharded while K is MLA-style latent plus RoPE state, not four independent KV-head groups.

The major implementation risks are now explicit design obligations:

- DS4 CUDA GLM TP4 does not exist today; current CUDA TP is DeepSeek-specific.
- Model loading needs a rank-local shard manifest or narrow Bird-compatible safetensors loader.
- DCP4 global top-k and selected-row exchange are the highest correctness risk.
- Prefill must commit DCP-compatible cache rows and cursor state in the same order decode consumes them.
- Output/logits sharding must become four-way for the memory goal.

## Implementation Readiness

Verdict: `READY_TO_START_IMPLEMENTATION`

Reason: current contract validation and decode simulation now pass. The production codebase still does not contain the GX10x4 GLM TP runtime, model-loader, DCP4 collectives, or CUDA GLM TP kernel path, so this is readiness to begin implementation, not a claim that the target already runs.

Recommended first implementation packets:

1. Rank runtime and launch manifest: four processes, rank ids, TP/DCP sizes, config hashes, shutdown.
2. Rank-local model loader: manifest validation, foreign-rank rejection, local shard mapping.
3. Base decode without DCP: TP4 QKV, rank-local attention, ATTN/FFN all-reduce, vocab gather.
4. DCP4 sparse attention: global top-k, selected-row exchange, cache-row ownership.
5. Prefill: chunked prompt execution, all-reduce, DCP-compatible KV commit, cursor handoff.
