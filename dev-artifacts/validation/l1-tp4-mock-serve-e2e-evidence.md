# L1 GLM 5.2 TP4 Mock Serve E2E Evidence

Date: 2026-08-07

Scope: executable mock proof for the first-version DS4 GLM 5.2 TP4/DCP4
serving cycle without checkpoint: initialize four rank descriptors, bind a
request/session, prefill one prompt, gather rank-owned coordinator tokens,
decode repeatedly, record generated-token stream events, stop cleanly, and
prove injected serving failures do not commit the failed token.

This evidence binds executable mock behavior to the BCD contract. It does not
claim real GLM kernels, real tensor-network collectives, or cross-GX10
performance.

## Pinned Contract

- Blueprint hash:
  `f5090c2f97f06122067f4b3c840bb2e6e581e52bccdbaf370c5a4bec47aa31ed`
- Semantic hash:
  `d90ce8a6de0e834c605c043318e3806cf1b8186cac93a897698438f64da2b538`
- Validated graphs:
  - `serve`
  - `request-session`
  - `stream-token`
  - `decode`
  - `decode-attn-allreduce-sum`
  - `decode-ffn-allreduce-sum`
  - `decode-logits-gather-topk`
  - `decode-dcp-row-exchange`

## Code Mapping

- `serve/action-serve-open`
  - Code: `ds4_glm52_mock_tp4_serve_request_ex` initializes four
    `ds4_glm52_mock_model` rank descriptors through `ds4_glm52_mock_model_init`.
  - Evidence: observation binds `tp_size=4`, `dcp_size=4`, `rank_count=4`, and
    one shared `model_hash`.
- `serve/action-serve-accept` / `request-session`
  - Code: `ds4_glm52_mock_tp4_serve_request_ex` validates the prompt request,
    marks `request_bound`, initializes one session identity, and marks
    `prefill_enqueued` before prefill execution.
  - Evidence: `tests/test_glm52_tp4_mock_serve.c` requires request binding and
    prefill enqueue before generated tokens are observed, and invalid prompt,
    observation, or budget inputs are rejected before request claim.
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
  - Code: `ds4_glm52_mock_tp4_serve_request_ex` stages each
    `ds4_glm52_mock_tp4_decode` result, gathers all rank contributions, and
    commits staged session state only after the TP4 decode step is complete.
  - Evidence: observation requires decode `token_step_j == prefill +
    generated_count`, decode `kv_length == prefill + generated_count`, changed
    hidden checksum, and replicated hidden/cursor state across all ranks.
- `decode-logits-gather-topk`
  - Code: `ds4_glm52_mock_tp4_step_add_rank` /
    `ds4_glm52_mock_tp4_step_add_contribution` gather rank-local vocab
    candidates and choose a coordinator-visible token by score and token-id
    tie-break.
  - Evidence: observation requires the prefill-selected token to belong to the
    winning rank's vocab shard and both prefill/decode candidate ranks to be in
    `0..3`.
- `serve/action-stream-token` / `stream-token`
  - Code: `ds4_glm52_mock_tp4_serve_request_ex` records each committed
    coordinator token in `generated_tokens[]`, records its owner rank in
    `generated_token_ranks[]`, increments `stream_event_count`, and applies
    stop-token/max-token continuation policy.
  - Evidence: tests require one stream event per generated token, valid global
    vocab ids, valid owner ranks, distinct max-token versus stop-token
    termination, and no stream event for a failed decode token.
- Serving failure preservation
  - Code: failure injection in `ds4_glm52_mock_tp4_serve_request_ex` covers
    worker loss, bad rank contribution, cursor divergence, and cancellation.
  - Evidence: tests require each failure to report its kind/index, preserve the
    last committed `token_step_j`/`kv_length`, and set
    `failed_without_cursor_advance`.
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
- `make tests/test_glm52_mock tests/test_glm52_tp4_mock
  tests/test_glm52_tp4_mock_process tests/test_tp4_rank_group
  tests/test_glm52_l0`: PASS.
- `./tests/test_glm52_mock`: PASS.
- `./tests/test_glm52_tp4_mock`: PASS.
- `./tests/test_glm52_tp4_mock_process`: PASS.
- `./tests/test_tp4_rank_group`: PASS.
- `./tests/test_glm52_l0`: PASS.
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
- Generic HTTP `ds4_server` parser/response-loop integration for this TP4 mock
  coordinator. The completed first-version mock serving seam is deterministic
  and server-shaped, but not yet driven through an actual socket-level HTTP
  request.

These are implementation leaves, not missing architecture decisions.
