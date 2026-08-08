# L1 Real Execution Frontier Evidence

Date: 2026-08-07

Scope: first production-code boundary for the four implementation concerns
after the TP4 mock proof:

1. model-load ready rank engines;
2. real TP4 collective backend boundary;
3. real DCP selected-row exchange boundary;
4. real GLM 5.2 decode-step boundary.

This is not a claim that real GX10 tensor/network execution is complete. It is
the fail-closed production frontier that separates mock proofs from real-path
work.

## Code Mapping

- `model-load/a-ready-rank-engines`
  - Code: `ds4_glm52_l0_stub_action(DS4_GLM52_L0_ACTION_SERVE_OPEN)`.
  - Returns `OK` only when `resident_shards_ready(state)` is true after
    launch-plan validation, rank shard manifest validation, and
    `model-shard-layout` publication.
  - Without a DS4 layout manifest, the path still blocks at
    `model-load/a-map-rank-shards`.

- TP4 collective real boundary
  - Code: `ds4_glm52_tp4_real_collective_allreduce`.
  - Host-buffer execution code:
    `ds4_glm52_tp4_collective_allreduce_f32_host` and
    `ds4_glm52_tp4_logits_gather_topk_f32_host`.
  - Shared frame validator: `ds4_glm52_tp4_collective_frame_validate`.
  - Shared frame wire codec:
    `ds4_glm52_tp4_collective_frame_encode/decode`.
  - Type: `ds4_glm52_tp4_collective_request`.
  - Wire contract: portable 96-byte big-endian collective metadata frame.
  - Validates collective frame version, collective kind, rank in `[0,4)`,
    TP4/DCP4/rank_count=4, supported tensor dtype, full participant mask,
    nonzero sequence/model/session identity, layer, shape hash, element count,
    byte count matching `element_count * dtype_size`, topology readiness,
    tensor transport readiness, rank-local partial readiness, and replicated
    output buffer readiness.
  - Returns `NOT_READY` after validation because real tensor all-reduce
    execution is not implemented behind this boundary yet.
  - Host-buffer ATTN/FFN helper validates four rank-indexed frames, four
    rank-local F32 partial buffers, and four output buffers, then writes
    identical replicated hidden outputs in deterministic rank order.
  - Host-buffer LOGITS helper validates four rank-indexed LOGITS frames and
    contiguous vocab shard coverage, then merges rank-local candidates into a
    deterministic global top-k list. The LOGITS frame element count now names
    the full rank-owned vocab shard width; `candidate_count` is only the
    already-local candidate list consumed by host top-k.
  - Tensor-binding validators:
    `ds4_glm52_tp4_collective_bind_tensor_buffers` and
    `ds4_glm52_tp4_logits_bind_tensor_shards`.
  - Binding evidence validates the real backend handoff metadata before any
    execution: opaque tensor/storage handle presence, rank indexing, readiness,
    dtype, shape hash, byte count, dtype-aligned byte range, capacity, full
    frame identity, and LOGITS contiguous vocab coverage.
  - Bound-host executors:
    `ds4_glm52_tp4_collective_allreduce_f32_bound_host` and
    `ds4_glm52_tp4_logits_gather_topk_f32_bound_host`.
  - Bound-host ATTN/FFN execution first validates the same tensor bindings,
    then reads rank-local partials from bound byte ranges and writes replicated
    outputs to bound output byte ranges.
  - Bound-host LOGITS execution first validates full rank-owned vocab shard
    bindings, then scans the bound shard tensors directly for deterministic
    global top-k. This proves the full-shard handoff, not just preselected
    candidate arrays.

- DCP selected-row real boundary
  - Code: `ds4_glm52_dcp_real_row_exchange`.
  - Host-buffer execution code: `ds4_glm52_dcp_selected_rows_host`.
  - Bound-host execution code: `ds4_glm52_dcp_selected_rows_bound_host`.
  - Type: `ds4_glm52_dcp_exchange_request`.
  - Bound row type: `ds4_glm52_dcp_bound_row_payload`.
  - Validates rank in `[0,4)`, DCP4/rank_count=4, nonzero sequence/model/session
    identity, valid layer and selection count, full owner rank mask, validated
    ownership plan, append-ordered KV, row payload readiness, and transport
    readiness.
  - Returns `NOT_READY` after validation because real selected-row network
    exchange is not implemented behind this boundary yet.
  - Host-buffer selected-row helper validates rank-indexed DCP requests,
    contiguous owner ranges, selected row ids, catalog payload metadata,
    append-ordered KV evidence, and transport readiness, then publishes
    deterministic per-rank replies only after every request passes.
  - Bound-host selected-row helper additionally validates selected row KV and
    K-rope byte bindings before publication: selected rows must be ready,
    capacity-safe, and hash-match the row metadata. Unselected cold rows may
    remain unbound.

- Real decode-step boundary
  - Code: `ds4_glm52_decode_real_step`.
  - Type: `ds4_glm52_decode_real_request`.
  - Validates DS4 GLM5.2 TP4 config, prefill-complete append-ordered KV state,
    replicated cursor, rank identity, nonzero sequence/model/session identity,
    valid input token, resident model readiness, TP4 collective readiness, DCP
    exchange readiness, and GLM 5.2 kernel readiness.
  - Returns `NOT_READY` after validation because real decode execution is not
    wired. It does not mutate `state-kv`, `state-kv-cursor`, or sampled-token
    state.

- DCP row-exchange mock tightening
  - Code: `ds4_glm52_dcp_mock.c`.
  - Rejection paths now clear replies at entry and stage rows before
    publication, so invalid requests leave no partial rows and nothing marked
    complete.
  - Selection counts outside `[0, DS4_GLM52_DCP_MOCK_MAX_SELECTION]` are
    rejected before row resolution.
  - Replies sort by query layer, token position, then row id.

## Validation

- `make -B tests/test_glm52_l0 && ./tests/test_glm52_l0`: PASS.
- `make -B tests/glm52_tp4_fabric_smoke`: PASS.
- BCD lint:
  `dev-artifacts/validation/lint-after-dcp-bound-payload.result.json`: PASS.
- BCD TP4 leaf validation:
  `dev-artifacts/validation/validate-tp4-leaves-after-dcp-bound-payload.result.json`:
  PASS.
- BCD decode full-trace simulation:
  `dev-artifacts/validation/simulate-decode-after-host-collectives.result.json`:
  PASS. The corresponding request explicitly records CRS812 `s-tp`
  `data_plane`, `fabric_addrs`, `fabric_switch`, and `management_addrs` field
  flows from the same TP state provenance used by the other `s-tp` fields.
- Real four-GX10 CRS812 portable-frame payload smoke and bad-frame rejection
  smoke: PASS on commit `91bcec7`.
  - Deployed the current committed tree to `/tmp/ds4-gx10-91bcec7` on all four
    GX10s.
  - Management SSH / fabric rank map used:
    rank0 `192.168.0.40` / `10.100.185.3`,
    rank1 `192.168.0.240` / `10.100.185.1`,
    rank2 `192.168.0.99` / `10.100.185.2`,
    rank3 `192.168.0.39` / `10.100.185.4`.
  - Per-host build and unit checks passed on all four GX10s:
    `tests/glm52_tp4_fabric_smoke`,
    `tests/test_glm52_dcp_row_exchange`, and
    `tests/test_glm52_tp4_allreduce`.
  - Positive CRS812 run: rank0 listened on `10.100.185.3:49121`, workers
    connected from ranks 1..3, the bound-host all-reduce executor verified 1024
    payload floats on all worker ranks, and rank0 completed with
    `ack_mask=0xf`.
  - Bad-frame CRS812 run: rank0 listened on `10.100.185.3:49122`, rank1 sent a
    corrupted typed collective frame, and rank0 rejected it before reduction
    with the frame byte-count/dtype-size validation error.
- `make cpu tests/test_tp4_rank_group tests/test_glm52_l0
  tests/test_glm52_mock tests/test_glm52_tp4_mock
  tests/test_glm52_tp4_mock_serve tests/test_glm52_tp4_allreduce
  tests/test_glm52_dcp_row_exchange tests/test_glm52_tp4_mock_process
  tests/test_engine_mgpu_placement tests/test_gpu_args` plus direct execution
  of those tests and `tests/test_gpu_args_cli.sh`: PASS.
- `tests/test_glm52_tp4_allreduce`: PASS. Covers L0 host-buffer ATTN/FFN
  all-reduce, tensor binding validation for all-reduce partial/output buffers,
  bound-host all-reduce execution from tensor bindings, L0 host-buffer logits
  gather/top-k, tensor binding validation for rank-owned logits shards, and
  bound-host full-shard logits top-k.
- `tests/test_glm52_dcp_row_exchange`: PASS. Covers L0 host-buffer DCP
  selected-row exchange, bound-host selected row payload validation, fail-closed
  hash/capacity rejection, unselected cold rows, and the existing DCP mock
  exchange.
- New `tests/test_glm52_l0.c` coverage:
  - `test_model_load_layout_reaches_ready_rank_engines`;
  - `test_real_collective_frontier_fails_closed`, including frame-version,
    dtype, shape-hash, byte-count, participant, transport, and fail-closed
    execution checks;
  - `test_real_dcp_exchange_frontier_fails_closed`;
  - `test_real_decode_frontier_preserves_cursor`.

## Remaining Work

- Real mmap handles and tensor object publication from model-load.
- Real GPU-resident GLM 5.2 QKV/MLA/MoE/logits kernels.
- GPU/NCCL or fabric-native tensor all-reduce and logits gather/top-k across
  four GX10 ranks using the validated tensor binding descriptors as the real
  payload handoff. The bound-host executor is an executable reference backend
  and transport smoke path, not the final GPU-resident collective backend.
- GPU/fabric-backed DCP selected-row network exchange for compact KV payloads.
  The bound-host executor is the selected-row payload reference backend; it is
  not the final network exchange implementation.
- Cross-GX10 launch, endpoint files, failure deadlines, and deployment smoke.
