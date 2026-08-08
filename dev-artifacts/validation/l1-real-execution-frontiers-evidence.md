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
  - Shared frame validator: `ds4_glm52_tp4_collective_frame_validate`.
  - Type: `ds4_glm52_tp4_collective_request`.
  - Validates collective frame version, collective kind, rank in `[0,4)`,
    TP4/DCP4/rank_count=4, supported tensor dtype, full participant mask,
    nonzero sequence/model/session identity, layer, shape hash, element count,
    byte count matching `element_count * dtype_size`, topology readiness,
    tensor transport readiness, rank-local partial readiness, and replicated
    output buffer readiness.
  - Returns `NOT_READY` after validation because real tensor all-reduce
    execution is not implemented behind this boundary yet.

- DCP selected-row real boundary
  - Code: `ds4_glm52_dcp_real_row_exchange`.
  - Type: `ds4_glm52_dcp_exchange_request`.
  - Validates rank in `[0,4)`, DCP4/rank_count=4, nonzero sequence/model/session
    identity, valid layer and selection count, full owner rank mask, validated
    ownership plan, append-ordered KV, row payload readiness, and transport
    readiness.
  - Returns `NOT_READY` after validation because real selected-row network
    exchange is not implemented behind this boundary yet.

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
- Local and CRS812 typed-frame payload smoke and bad-frame rejection smoke:
  PASS.
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
- Real tensor all-reduce and logits gather/top-k across four GX10 ranks.
- Real DCP selected-row network exchange for compact KV payloads.
- Cross-GX10 launch, endpoint files, failure deadlines, and deployment smoke.
