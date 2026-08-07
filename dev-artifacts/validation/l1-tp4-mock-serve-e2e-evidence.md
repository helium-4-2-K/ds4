# L1 GLM 5.2 TP4 Mock Serve E2E Evidence

Date: 2026-08-07

Scope: executable mock proof for one DS4 GLM 5.2 TP4/DCP4 serving request:
initialize four rank descriptors, prefill one prompt, gather a rank-owned
coordinator token, decode one token, and observe contract-visible state.

This evidence binds executable mock behavior to the BCD contract. It does not
claim real GLM kernels, real tensor-network collectives, or cross-GX10
performance.

## Pinned Contract

- Blueprint hash:
  `3466079cab4a6e6d8011d675c3cf429d7cda2bc961c2d5ca006e3087af317b6c`
- Semantic hash:
  `3177dc06506cb955e519157ab2f2125472287223a6e5f60ded7e3a95ecead023`
- Validated graphs:
  - `serve`
  - `decode`
  - `decode-attn-allreduce-sum`
  - `decode-ffn-allreduce-sum`
  - `decode-logits-gather-topk`
  - `decode-dcp-row-exchange`

## Code Mapping

- `serve/action-serve-open`
  - Code: `ds4_glm52_mock_tp4_serve_request` initializes four
    `ds4_glm52_mock_model` rank descriptors through `ds4_glm52_mock_model_init`.
  - Evidence: observation binds `tp_size=4`, `dcp_size=4`, `rank_count=4`, and
    one shared `model_hash`.
- `model-load/a-map-rank-shards`
  - Code: `ds4_glm52_mock_model_init` creates rank-local Q-head and vocab
    ownership descriptors and replicated embedding/MLA metadata.
  - Evidence: observation requires Q heads tile `[0,64)` exactly once and
    vocab shards tile `[0,154880)` exactly once.
- `serve/action-serve-prefill` / `prefill`
  - Code: `ds4_glm52_mock_tp4_prefill` performs rank-local prefill and
    `ds4_glm52_mock_tp4_apply_collectives` publishes post-collective hidden
    state.
  - Evidence: observation requires one logical `session_hash`, prefill
    `token_step_j == prompt_length`, prefill `kv_length == prompt_length`, and
    replicated hidden/cursor state across all four ranks.
- `decode/action-decode-token`
  - Code: `ds4_glm52_mock_tp4_decode` consumes the coordinator-selected token,
    appends one KV row, advances the cursor, and applies DCP/all-reduce mock
    collectives.
  - Evidence: observation requires decode `token_step_j == prefill + 1`,
    decode `kv_length == prefill + 1`, changed hidden checksum, and replicated
    hidden/cursor state across all ranks.
- `decode-logits-gather-topk`
  - Code: `ds4_glm52_mock_tp4_step_add_rank` /
    `ds4_glm52_mock_tp4_step_add_contribution` gather rank-local vocab
    candidates and choose a coordinator-visible token by score and token-id
    tie-break.
  - Evidence: observation requires the prefill-selected token to belong to the
    winning rank's vocab shard and both prefill/decode candidate ranks to be in
    `0..3`.
- `decode-attn-allreduce-sum` and `decode-ffn-allreduce-sum`
  - Code: `ds4_glm52_mock_allreduce_add_contribution` and
    `ds4_glm52_mock_allreduce_finish`, used by
    `ds4_glm52_mock_tp4_apply_collectives`.
  - Evidence: `tests/test_glm52_tp4_allreduce` proves rank-complete
    deterministic sum, duplicate/missing rank rejection, identity mismatch
    rejection, shape/dtype rejection, and non-finite rejection.
- `decode-dcp-row-exchange`
  - Code: standalone lowered primitive in `ds4_glm52_dcp_mock.c` and integrated
    mock DCP selection inside `ds4_glm52_mock_tp4_apply_collectives`.
  - Evidence: `tests/test_glm52_dcp_row_exchange` proves complete/gapless owner
    mapping, missing/duplicate owner rejection, out-of-range rejection,
    conflicting payload rejection, deterministic ordering, deduplication, and
    cross-rank row delivery.

## Validation

- `make -B tests/test_glm52_tp4_mock_serve`: PASS.
- `./tests/test_glm52_tp4_mock_serve`: PASS.
- `make -B tests/test_glm52_tp4_allreduce tests/test_glm52_dcp_row_exchange
  tests/test_glm52_tp4_mock tests/test_glm52_tp4_mock_process
  tests/test_glm52_mock tests/test_glm52_l0 tests/test_tp4_rank_group`: PASS.
- `./tests/test_glm52_tp4_allreduce`: PASS.
- `./tests/test_glm52_dcp_row_exchange`: PASS.
- `./tests/test_glm52_tp4_mock`: PASS.
- `./tests/test_glm52_tp4_mock_process`: PASS.
- `./tests/test_glm52_mock`: PASS.
- `./tests/test_glm52_l0`: PASS.
- `./tests/test_tp4_rank_group`: PASS.
- `make cpu`: PASS.
- `git diff --check`: PASS.

BCD mechanical evidence regenerated:

- `dev-artifacts/validation/status-after-mock-serve.result.json`: PASS.
- `dev-artifacts/validation/lint-mock-serve.result.json`: PASS.
- `dev-artifacts/validation/validate-mock-serve-tp4-leaves.result.json`: PASS.
- `dev-artifacts/validation/validate-mock-serve-decode-composite.result.json`:
  PASS.
- `dev-artifacts/validation/validate-mock-serve-serve-composite.result.json`:
  PASS.
- `dev-artifacts/validation/simulate-mock-serve-decode-full-trace.result.json`:
  PASS.
- `dev-artifacts/ds4-arch-review.html`: PASS render, 14 graphs, 135 nodes,
  179 edges, 0 warnings.

## Real-Component Frontier

The BCD contract now has lowered leaves for the real data-plane components, and
the mock code proves their orchestration shape. The following are not yet real:

- GLM 5.2 QKV/MLA CUDA kernels for TP4.
- Real DCP selected-row network exchange over four GX10 machines.
- Real all-reduce tensor transport and sum over device buffers.
- Real GLM 5.2 sharded checkpoint loading for rank-specific files.
- Bird/vLLM oracle comparison and GX10 performance evidence.

These are implementation leaves, not missing architecture decisions.
