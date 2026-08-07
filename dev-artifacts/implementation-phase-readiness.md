# Implementation Phase Readiness

Date: 2026-08-06

Verdict: `READY_TO_START_IMPLEMENTATION`

This means the BCD design and evidence are sufficient to start production implementation work. It does not mean the GX10x4 GLM 5.2 TP4+DCP4 runtime already exists in DS4.

The concrete implementation and test sequence is recorded in
`dev-artifacts/implementation-and-test-plan.md`.

## Pinned Evidence

- Blueprint hash: `e9e9e06f90600d5fbb1353d1f669ff9270095bb7f9c6eeb1e204cfd9d283d97a`
- Semantic hash: `224e0e716e87f2c8f3f138b5e2fec753ef9f81c01a72adc2b686b5d2a9079dea`
- Lint: `dev-artifacts/validation-lint-current.json`, PASS.
- L0 validation: `dev-artifacts/validation-serve-l0-current.json`, PASS.
- Decode composite validation: `dev-artifacts/validation-decode-composite-current.json`, PASS.
- Leaf validation: `dev-artifacts/validation-leaves-current.json`, PASS.
- Decode simulation: `dev-artifacts/simulation-decode-current-full-trace-v2.json`, PASS.

## Required Implementation Packets

### 1. TP4 Rank Runtime

Target seams:

- `/Users/helium/src/ds4-gb10x4-bcd/ds4_tp.h`
- `/Users/helium/src/ds4-gb10x4-bcd/ds4_tp.c`
- `/Users/helium/src/ds4-gb10x4-bcd/ds4.c`
- `/Users/helium/src/ds4-gb10x4-bcd/ds4_server.c`

Implement four rank processes, rank ids 0..3, TP/DCP size validation, startup barrier, clean shutdown, and command broadcast. Do not reuse `ds4_distributed.*` as the core runtime because that code models pipeline/layer distribution, not lockstep tensor parallel decode.

Validation gates: rank mismatch, missing rank, wrong TP/DCP size, timeout, clean shutdown, and a no-model rendezvous smoke test.

### 2. Rank-Local Model Loading

Target seams:

- `/Users/helium/src/ds4-gb10x4-bcd/ds4.c`
- `/Users/helium/src/ds4-gb10x4-bcd/ds4_layer_pack.c`
- `/Users/helium/src/ds4-gb10x4-bcd/ds4_layer_pack.h`

Implement a narrow rank-shard manifest for GLM 5.2 TP4. Validate config hash, tensor ownership spans, local file hashes, rank id, TP size, DCP size, and foreign-rank rejection before mapping weights.

Validation gates: missing shard, wrong rank shard, wrong config hash, unsupported quant/layout, and complete ownership coverage.

### 3. Base Decode TP4 Without DCP

Target seams:

- `/Users/helium/src/ds4-gb10x4-bcd/ds4.c`
- `/Users/helium/src/ds4-gb10x4-bcd/ds4_cuda.cu`
- `/Users/helium/src/ds4-gb10x4-bcd/ds4_gpu.h`
- `/Users/helium/src/ds4-gb10x4-bcd/ds4_gpu_mgpu.h`

Implement rank-owned Q heads, local attention/output partials, ATTN all-reduce, 4-way expert ownership, FFN all-reduce, vocab/logits sharding, rank0 sampling, and token broadcast. First milestone may use replicated compact KV before DCP4 row partitioning.

Validation gates: per-layer Q shard ranges, attention partial shape, ATTN all-reduce equality, FFN all-reduce equality, logits gather/top-k, and short greedy continuation against the Bird/vLLM oracle.

### 4. DCP4 Sparse Attention

Target seams:

- `/Users/helium/src/ds4-gb10x4-bcd/ds4.c`
- `/Users/helium/src/ds4-gb10x4-bcd/ds4_cuda.cu`
- `/Users/helium/src/ds4-gb10x4-bcd/ds4_kvstore.c`
- `/Users/helium/src/ds4-gb10x4-bcd/ds4_kvstore.h`

Implement DCP-owned compact KV rows, local sparse candidates, global top-k merge, selected-row descriptor broadcast, and selected-row exchange/read before local attention.

Validation gates: selected row ids against Bird/vLLM, global top-k counterexamples, selected-row owner correctness, attention output comparison, and long-context sparse selection smoke.

### 5. Prefill

Target seams:

- `/Users/helium/src/ds4-gb10x4-bcd/ds4.c`
- `/Users/helium/src/ds4-gb10x4-bcd/ds4_cuda.cu`
- `/Users/helium/src/ds4-gb10x4-bcd/ds4_bench.c`
- `/Users/helium/src/ds4-gb10x4-bcd/tests/test_engine_mgpu_runtime.c`

Implement bounded prompt chunks, TP layer execution, prefill all-reduce, DCP-compatible KV row commit, cursor commit, and decode handoff.

Validation gates: chunk cursor monotonicity, prompt KV ownership, prefill-vs-decode handoff, `--tensor-parallel-token-prefill` exact diagnostic, and prefill throughput comparison against Bird/vLLM.

## Deferred Packet

MTP/speculation remains deferred. Add it only after base decode and prefill match the oracle because it adds draft/verify commit semantics and DCP draft sharding.
