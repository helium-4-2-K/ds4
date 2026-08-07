# L1 Prefill L0 Implementation Evidence

Date: 2026-08-07

Scope: first L1 realization under `serve/action-serve-prefill`, which expands
to the `prefill` child graph. This slice implements the prefill L0 state
machine: prompt token span ownership, token position cursor, KV length cursor,
append-ordered KV/cache state, same-state identity back to serve, and
rank-plan/model-shard-layout binding. No real GLM 5.2 attention/matmul kernels
or weights are involved.

## Contract Mapping (code -> BCD node/state)

- `prefill/a-tokenize-prompt`
  - Code: `ds4_glm52_l0_prefill_set_prompt` (bind leader-owned prompt token
    span) plus the a-tokenize guard block in `ds4_glm52_l0_prefill_run`.
  - State: `state->prompt` (`ds4_glm52_l0_prompt_tokens`) carries
    `session_id`, `token_ids`, `prompt_length`, and the consumed-position
    cursor. The session id is copied into owned prompt state. Maps to BCD data node `d-prompt-tokens`
    (`blueprint.work.ds4_arch.prompt_tokens`, leader-only).
  - Preconditions block fail-closed in dependency order before any cursor
    mutation: resident rank-local model shards (`state-model-worker`),
    ready TP4/DCP4 collective fabric (`state-tp`), accepted request/session
    (`serve` accept), and a prompt token span bound to the accepted session.
  - The session identity of the prompt must match the accepted request
    session (`same_state_as` back to the serve request).
- `prefill/a-plan-prefill-chunks`
  - Code: chunk-plan block in `ds4_glm52_l0_prefill_run`.
  - State: `state->chunk` (`ds4_glm52_l0_prefill_chunk`) records
    `chunk_index`, `start_position`, `end_position`, `token_count` per
    committed chunk. Maps to BCD data node `d-prefill-chunk`
    (`blueprint.work.ds4_arch.prefill_chunk`, replicated).
  - The default chunk size is the BCD reference 1024 prefill chunk
    (`DS4_GLM52_L0_PREFILL_CHUNK`); callers may override for testing.
  - KV identity is validated here: `state-kv` must be TP-aware and
    append-ordered, not already prefilled, and session-consistent with the
    request (same-state identity to `serve/state-kv`).
- `prefill/a-prefill-layer-tp`
  - Code: `ds4_glm52_l0_prefill_run` skeleton seam (identity gate plus
    explicit mock-only continuation).
  - State read: `state-model-worker` (`resident_shards`) and `state-tp`
    (`tp_fabric`). Rank identity is validated as
    `resident_shards.rank == rank_plan.rank == config.rank ==
    tp_fabric.local_rank` (`prefill_identity_ok`), enforcing the
    rank-plan/model-shard-layout binding; a mismatch is INVALID and mutates
    nothing. Non-mock execution stops here with NOT_READY before any KV/cursor
    commit because real GLM 5.2 TP prefill kernels are not implemented.
- `prefill/a-prefill-allreduce`
  - Code: `ds4_glm52_l0_prefill_run` explicit mock skeleton seam (trace). No
    real allreduce runs; non-mock execution cannot reach this action yet.
- `prefill/a-commit-prefill-kv`
  - Code: append-order validation pass + apply loop in
    `ds4_glm52_l0_prefill_run`.
  - State update: `state-kv` (`ds4_glm52_l0_kv_state`): `kv.length` advances
    to the committed prompt length, `kv.session_id` binds to the request;
    `state-kv-cursor` (`ds4_glm52_l0_kv_cursor`): `kv_length` and
    `token_step_j` advance together (append-ordered monotonic).
  - Append ordering is explicit and validated fail-closed: the whole chunk
    plan must start exactly at the current append-ordered KV cursor and cover
    the prompt contiguously, or the commit is INVALID and no state mutates.
  - Output: `state->commit` (`ds4_glm52_l0_prefill_commit`) maps to BCD
    `d-prefill-kv-commit` / `d-prefill-handoff`
    (`blueprint.work.ds4_arch.prefill_kv_commit`, replicated):
    `prefilled_length`, `tp_degree=4`, `dcp_degree=4`,
    `kv_cache_dtype=fp8_ds_mla`.
- `prefill/a-prefill-ready`
  - Code: finalization block in `ds4_glm52_l0_prefill_run`.
  - State: `kv.prefill_complete = true`, `cursor.replicated = true`,
    `cursor.phase = PREFILL`. After this, `prefill_ready()` admits decode.
  - Exit semantics match `serve/action-serve-prefill`: "prefilled" / decode
    may start at the next position using the committed KV cursor.
- `serve/action-serve-prefill` (root expands to `prefill`)
  - Code: `ds4_glm52_l0_stub_action(DS4_GLM52_L0_ACTION_PREFILL)` now
    delegates to `ds4_glm52_l0_prefill_run`. In explicit mock mode, success
    returns OK and leaves decode-ready state; in non-mock mode, the action
    returns typed NOT_READY at the real TP layer seam before KV/cursor commit.
    Any prerequisite/identity/append failure returns typed NOT_READY/INVALID
    and mutates nothing.

## Shared state identity (same_state_as)

- `state-kv` (prefill) == `serve/state-kv` == `prefill/state-kv`: the same
  `ds4_glm52_l0_kv_state` is updated by prefill and read by
  `serve/action-decode-token`. The KV session identity is bound to the
  request session during prefill.
- `state-kv-cursor` (prefill) == `serve/state-kv-cursor`: the same
  `ds4_glm52_l0_kv_cursor`. New `phase` field (`PREFILL`/`DECODE`) records
  which phase last advanced the cursor, matching BCD `kv_cursor.phase`.
- `state-model-worker` / `state-tp` (prefill) == serve root states: resident
  shards and TP4 fabric produced by model-load/TP-group are read by prefill.

## Decode cursor gate

- `prefill_ready()` (used by `serve/action-decode-token` and
  `serve/action-kv-checkpoint`) now additionally requires
  `cursor.phase == PREFILL || DECODE`. Decode can start only from a
  prefill-produced cursor (phase PREFILL, consistent kv_length) or another
  consistent produced cursor (phase DECODE with matching kv_length); a
  cursor with no producing phase or a diverged kv_length is rejected.

## Production State Evidence

- Header state: `ds4_glm52_l0_prompt_tokens`, `ds4_glm52_l0_prefill_chunk`,
  `ds4_glm52_l0_prefill_commit`, and `phase` on
  `ds4_glm52_l0_kv_cursor`, all in `ds4_glm52_l0.h`.
- `DS4_GLM52_L0_PREFILL_CHUNK` (1024) mirrors the Bird/vLLM
  max-num-batched-tokens/prefill-chunk reference recorded in the BCD.
- `DS4_GLM52_L0_KV_CACHE_DTYPE` (fp8_ds_mla) mirrors BCD
  `distlink.kv_cache_dtype`.

## Validation

- `make tests/test_glm52_l0 && ./tests/test_glm52_l0`: PASS.
- `make tests/test_tp4_rank_group && ./tests/test_tp4_rank_group`: PASS.
- `make tests/test_glm52_mock && ./tests/test_glm52_mock`: PASS.
- `make cpu`: PASS.
- `git diff --check`: PASS.
- BCD prefill leaf validation:
  `dev-artifacts/validation-prefill-current.json`, PASS.
- BCD serve L0 validation: `dev-artifacts/validation-serve-l0-current.json`,
  PASS.

### Proof points added (tests/test_glm52_l0.c)

- prefill child action order matches the BCD prefill graph (6 actions in
  contract order).
- prefill cannot run before model/rank-plan/fabric/request readiness, and a
  blocked prefill does not mutate KV/cursor/commit state.
- non-mock prefill reaches the real TP layer seam and stops NOT_READY with no
  KV/cursor/commit mutation.
- explicit mock prefill advances `token_step_j` and `kv_length`
  deterministically to the prompt length and leaves a decode-ready replicated
  PREFILL cursor.
- KV append ordering is explicit and validated: a chunk plan that does not
  start at the append-ordered cursor is rejected (INVALID) with no mutation.
- multi-chunk prefill (37 tokens / chunk 16) commits exactly once and traces
  all six BCD child actions in order.
- mismatched rank-plan/model/config/fabric identity is rejected with no
  mutation.
- empty prompt and multi-token prompt behavior are both deterministic and
  repeatable.
- re-running prefill on an already-prefilled cursor is rejected (no double
  append).
- decode starts only from a prefill-produced or otherwise valid produced
  cursor (PREFILL or DECODE phase, consistent kv_length); phase=NONE and
  diverged cursors are rejected.

## Remaining Frontier

The next frontier is the real four-rank prefill execution: rank-local GLM 5.2
TP layer kernels (`prefill/a-prefill-layer-tp`), the TP all-reduce for hidden
partials (`prefill/a-prefill-allreduce`), and DCP-aware prompt KV row
materialization inside `prefill/a-commit-prefill-kv`. This slice proves the
state/ordering/identity contract and leaves those kernel seams strict and
fail-closed through the traced skeleton steps.
