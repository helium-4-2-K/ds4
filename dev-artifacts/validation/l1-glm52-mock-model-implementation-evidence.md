# L1 GLM 5.2 Mock-Model Implementation Evidence

Date: 2026-08-07

Scope: explicit mock-model support for the GLM 5.2 TP4/GX10x4 BCD
implementation. This is a validation seam for the production engine/session
surface; it is not a real GLM kernel implementation and does not change the
target blueprint contract.

## Contract Mapping

- `serve/action-serve-open`
  - Code: `ds4_engine_open_internal` in `ds4.c`.
  - Under `--glm52-tp4-mock-model --glm52-tp4-mock-matmul`, validates the same
    GLM 5.2 TP4/DCP4/PP1/rank/rank-plan contract through
    `ds4_glm52_l0_validate_config`.
  - Opens a synthetic engine only after explicit mock opt-in; normal L0 still
    fails closed at unresolved seams.
- `model-load/a-map-rank-shards`
  - Code: `ds4_glm52_mock_model_init` in `ds4_glm52_mock.c`.
  - Produces descriptor evidence for rank-local Q-head/vocab ownership and
    replicated embedding/MLA metadata without mmaping real weights.
- `serve/action-serve-prefill`
  - Code: `ds4_session_sync_internal` mock branch in `ds4.c`, backed by
    `ds4_glm52_mock_prefill`.
  - Updates the session checkpoint and deterministic mock KV cursor/hash state;
    mock `token_step_j` uses the same prompt-length cursor convention as the
    L0 prefill skeleton.
- `serve/action-decode-token`
  - Code: `ds4_session_eval_internal` mock branch in `ds4.c`, backed by
    `ds4_glm52_mock_decode`.
  - Advances the checkpoint by exactly one token and produces deterministic
    full-vocab logits.
- `serve/action-tp-group`
  - Code: `ds4_glm52_mock_tp4_step_add_rank`,
    `ds4_glm52_mock_tp4_step_add_contribution`, and the mock rank-step/token
    transport helpers in `ds4_glm52_mock.c`, composed with the existing
    `ds4_glm52_tp4_rank_group` command/ack state machine.
  - Proves mock ranks share logical model/session identity, preserve
    rank-local Q-head/vocab ownership, agree on command sequence and
    cursor/KV state, round-trip rank-step/token frames, reject malformed mock
    frames, and gather a coordinator-visible token from rank-local vocab
    candidates.

## Mock Boundary

- The mock has full GLM 5.2 shape constants for layers, hidden size, vocab,
  Q-head count, MLA KV dimensions, and TP4 rank spans.
- The mock does not allocate or parse GGUF tensors.
- The mock tokenizer is synthetic and deterministic; it is only for server/API
  exercising without a real tokenizer file.
- Real QKV/MLA, DCP top-k, attention, MoE routing, collectives, and output-head
  kernels remain implementation frontiers.
- The TP4 mock proof includes both in-process rank contributions and a local
  four-process loopback TCP run. The frame helpers are tested for local
  round-trip and malformed-frame rejection, but they do not prove cross-machine
  network transport performance or real tensor all-reduce/gather math.

## Validation

- `make tests/test_glm52_mock`: PASS.
- `./tests/test_glm52_mock`: PASS.
- `make tests/test_glm52_tp4_mock`: PASS.
- `./tests/test_glm52_tp4_mock`: PASS.
- `make tests/test_glm52_tp4_mock_process`: PASS.
- `./tests/test_glm52_tp4_mock_process`: PASS.
- `make cpu`: PASS.
- `./ds4 --help | rg "glm52-tp4-mock|glm52-tp4-l0"`: PASS.
- `./ds4-server --help | rg "glm52-tp4-mock|glm52-tp4-l0"`: PASS.
- CLI smoke with `--model /mock/glm-5.2 --glm52-tp4-mock-model
  --glm52-tp4-mock-matmul --glm52-tp4-rank 2 ... --prompt hello
  --tokens 2`: PASS; opens rank 2 with Q heads `[32,48)` and emits visible
  mock token text.
- Server smoke with the same mock flags on `127.0.0.1:19052`: PASS; process
  opens the mock engine, reaches listen state, and shuts down cleanly when
  signaled.
