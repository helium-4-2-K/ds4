# L1 Real Execution Frontier Evidence

Date: 2026-08-07

Scope: first production-code boundary for the four implementation concerns
after the TP4 mock proof:

1. model-load ready rank engines;
2. real TP4 collective backend boundary;
3. real DCP selected-row exchange boundary;
4. real GLM 5.2 decode-step boundary.

This records the production frontier plus the first executable real GX10
tensor/network slice. The validated real path is a staged-GPU TP4 collective
over CRS812; the full GLM5.2 kernel backend remains a separate implementation
frontier.

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
  - CRS812 f32 collective transport:
    `ds4_glm52_tp4_transport_send_collective_f32` and
    `ds4_glm52_tp4_transport_recv_collective_f32`.
  - Staged GPU tensor execution frontier:
    `ds4_glm52_tp4_gpu_collective_read_f32` and
    `ds4_glm52_tp4_gpu_collective_write_f32`.
  - The staged GPU path validates the same collective frame plus the
    rank-owned `ds4_gpu_tensor` binding before moving bytes. A CUDA backend can
    supply `ds4_gpu_tensor_read/write` callbacks, so the path is:
    rank-local GPU partial -> host staging -> CRS812 typed f32 frame ->
    coordinator deterministic TP4 reduce -> CRS812 typed f32 frame ->
    rank-local GPU replicated output. This is a real inter-machine executable
    TP4 path over CRS812, but it is intentionally staged through host memory and
    is not an NCCL/RDMA device-to-device collective.

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
  - Returns `NOT_READY` without mutation unless the backend also supplies
    explicit logits-ready, sampled-token-ready, and KV-append-committed
    evidence.
  - With complete backend evidence, validates the sampled token against the
    GLM5.2 vocabulary, publishes it at the current token-step index `j`,
    advances `state-kv.length`, `state-kv-cursor.kv_length`, and
    `state-kv-cursor.token_step_j` by one, switches the cursor to decode phase,
    and marks logits coordinator-visible.

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
- Real four-GX10 CRS812 fabric-ready runtime handshake: PASS on working tree
  after `0e3ac0b`.
  - Production API added:
    `ds4_glm52_tp4_fabric_ready_handshake`.
  - The no-payload `tests/glm52_tp4_fabric_smoke` path now calls that API
    directly instead of duplicating coordinator/worker hello-command-ack logic.
  - Deployed the working tree to
    `/tmp/ds4-gx10-ready-handshake-20260807220545` on all four GX10s.
  - Per-host target build passed:
    `make tests/glm52_tp4_fabric_smoke tests/test_tp4_rank_group`.
  - CRS812 run: rank0 `192.168.0.40` listened on `10.100.185.3:49131`;
    ranks 1..3 connected from `192.168.0.240`, `192.168.0.99`, and
    `192.168.0.39`.
  - Rank0 published local TP4/DCP4 fabric-ready state and completed shutdown
    command sequence `1` with `ack_mask=0xf`; all workers received shutdown
    sequence `1` and acked cleanly.
- Decode-bound collective dispatch and real four-GX10 CRS812 payload smoke:
  PASS on working tree after `1090a6b`.
  - Production API added:
    `ds4_glm52_decode_bound_collective_host`.
  - The API validates decode request rank/token/sequence/model/session
    identity and model/TP/DCP/kernel readiness before dispatching typed TP4
    frames to a backend.
  - ATTN/FFN dispatch calls
    `ds4_glm52_tp4_collective_allreduce_f32_bound_host`; LOGITS dispatch calls
    `ds4_glm52_tp4_logits_gather_topk_f32_bound_host`.
  - Local unit gate: `tests/test_glm52_tp4_allreduce` covers decode-bound
    ATTN all-reduce, decode/frame identity mismatch rejection, decode-bound
    LOGITS full-shard top-k, and readiness rejection.
  - Deployed the working tree to
    `/tmp/ds4-gx10-decode-bound-20260807220956` on all four GX10s.
  - Per-host target build and unit check passed:
    `make tests/glm52_tp4_fabric_smoke tests/test_glm52_tp4_allreduce &&
    ./tests/test_glm52_tp4_allreduce`.
  - CRS812 payload run: rank0 `192.168.0.40` listened on
    `10.100.185.3:49132`; ranks 1..3 connected from `192.168.0.240`,
    `192.168.0.99`, and `192.168.0.39`.
  - Rank0 reduced 1024 payload floats through
    `ds4_glm52_decode_bound_collective_host`, observed `first=1000.0` and
    `last=5092.0`, then completed decode command sequence `1` with
    `ack_mask=0xf`; every worker verified the replicated reduced payload and
    acked.
- DCP transport payload and real four-GX10 CRS812 selected-row smoke: PASS on
  working tree after `3d487ef`.
  - Production payload types added:
    `ds4_glm52_dcp_transport_payload` and
    `ds4_glm52_dcp_transport_row`.
  - Production conversion APIs added:
    `ds4_glm52_dcp_transport_payload_from_bound` and
    `ds4_glm52_dcp_transport_payload_to_bound`.
  - Production framed transport APIs added:
    `ds4_glm52_dcp_transport_send_payload` and
    `ds4_glm52_dcp_transport_recv_payload`.
  - Local unit gate: `tests/test_glm52_dcp_row_exchange` covers bound-row
    serialization, received bound-row reconstruction, feeding reconstructed
    rows into `ds4_glm52_dcp_selected_rows_bound_host`, and received-byte hash
    mismatch rejection.
  - Local four-process smoke: `tests/glm52_tp4_fabric_smoke --dcp-rows 8`
    passed with 32 transported owner rows and 4x4 selected-row replies.
  - Deployed the working tree to
    `/tmp/ds4-gx10-dcp-transport-20260807221750` on all four GX10s.
  - Per-host target build and unit check passed:
    `make tests/glm52_tp4_fabric_smoke tests/test_glm52_dcp_row_exchange &&
    ./tests/test_glm52_dcp_row_exchange`.
  - CRS812 DCP run: rank0 `192.168.0.40` listened on
    `10.100.185.3:49133`; ranks 1..3 connected from `192.168.0.240`,
    `192.168.0.99`, and `192.168.0.39`.
  - Rank0 received 8 owner rows from each rank, reconstructed 32 bound
    KV/K-rope row payloads, produced complete selected-row replies for all
    four requester ranks, and completed decode command sequence `1` with
    `ack_mask=0xf`; every worker sent its DCP payload and acked.
- GPU-resident TP4 binding frontier and real GX10 CUDA device check: PARTIAL
  PASS on working tree after `bd8a535`; expanded to staged GPU fabric execution
  after the current change.
  - Production GPU binding type added:
    `ds4_glm52_tp4_gpu_tensor_binding`.
  - Production validator added:
    `ds4_glm52_tp4_collective_bind_gpu_tensors`.
  - The validator upgrades the previous opaque-handle check for TP4
    ATTN/FFN collectives: every rank-local partial and replicated output must
    name a ready `ds4_gpu_tensor`, non-null device pointer, rank-indexed
    binding, expected logical device ownership, matching dtype/shape/byte
    metadata, and dtype-aligned byte range within the device tensor capacity.
  - Local unit gate: `tests/test_glm52_tp4_allreduce` covers accepted
    rank-owned GPU tensor descriptors, wrong-device rejection, and unaligned
    range rejection.
  - Deployed the working tree to
    `/tmp/ds4-gx10-gpu-binding-20260807222117` on all four GX10s.
  - Per-host GLM52 binding gate passed:
    `make tests/test_glm52_tp4_allreduce && ./tests/test_glm52_tp4_allreduce`.
  - Rank0 real CUDA gate passed:
    `make tests/test_gpu_xdev && ./tests/test_gpu_xdev` on `192.168.0.40`.
    The test saw one NVIDIA GB10 CUDA device (`sm_121`) and passed existing
    allocation/copy/top-k/attention/MoE/Q8/F16/attention-output-TP CUDA checks.
  - Added CUDA/GX10 smoke executable:
    `tests/glm52_tp4_gpu_fabric_smoke`.
  - Deployed the working tree to
    `/tmp/ds4-gx10-gpu-fabric-20260807223814` on all four GX10s.
  - All four GX10s built:
    `make tests/test_glm52_l0 tests/test_glm52_tp4_allreduce
    tests/glm52_tp4_gpu_fabric_smoke`, then passed
    `./tests/test_glm52_l0` and `./tests/test_glm52_tp4_allreduce`.
  - CRS812 staged GPU collective run: rank0 `192.168.0.40` listened on
    `10.100.185.3:49150`; ranks 1..3 connected from `192.168.0.240`,
    `192.168.0.99`, and `192.168.0.39`.
  - Each rank allocated CUDA tensors on its local NVIDIA GB10 (`sm_121`),
    wrote its rank-local f32 partial to GPU, staged that partial through
    `ds4_glm52_tp4_gpu_collective_read_f32`, exchanged typed collective frames
    over CRS812, wrote the replicated reduced result back to GPU through
    `ds4_glm52_tp4_gpu_collective_write_f32`, read it back, and verified
    1024 floats. Rank0 completed with `ack_mask=0xf`.
  - Remaining limitation: the CRS812 collective is staged through host memory
    and reduced on rank0 CPU after GPU readback. The repo still needs real
    GLM5.2 QKV/MLA/MoE/logits weight kernels and a production decode backend
    that produces the logits/sample/KV-append evidence consumed by
    `ds4_glm52_decode_real_step`.
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
  - `test_real_decode_frontier_preserves_cursor`, including non-mutating
    missing backend-output evidence, successful sampled-token/KV cursor commit,
    and invalid sampled-token rejection.
- New `tests/glm52_tp4_gpu_fabric_smoke.c` coverage:
  - Four real GX10 processes over CRS812;
  - CUDA tensor allocation/write/read on every rank;
  - production TP4 f32 collective transport;
  - production GPU staged read/write frontier;
  - deterministic TP4 reduction and replicated result verification.

## Remaining Work

- GPU upload from retained model-load mmap handles and DS4-native runtime
  tensor metadata.
- Optional direct production checkpoint header decode for dtype/shape metadata
  instead of relying on the DS4-native layout manifest.
- Real GPU-resident GLM 5.2 QKV/MLA/MoE/logits kernels.
- Optional replacement of the staged host-memory CRS812 collective with
  direct device collective transport if the deployment requires lower latency
  than the current validated fallback.
- Direct GPU/NCCL or fabric-native tensor all-reduce and logits gather/top-k
  across four GX10 ranks. The current validated CRS812 path is a staged-GPU
  fallback: GPU tensors are real on every rank, but network transfer and
  reduction stage through host memory.
- GPU/fabric-backed DCP selected-row network exchange for compact KV payloads.
  The bound-host executor is the selected-row payload reference backend; it is
  not the final network exchange implementation.
- Cross-GX10 launch, endpoint files, failure deadlines, and deployment smoke.
