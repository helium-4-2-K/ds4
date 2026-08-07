# L1 TP4 Mock Serve Contract-to-Code Review

Date: 2026-08-07

## Verdict

- `SUBJECT_FIDELITY`: PASS
- `PARENT_PROJECTION`: PASS
- `SUFFICIENT_ASSURANCE`: PASS
- Final verdict: PASS

## Subject

Actual work reviewed:

- `ds4_glm52_mock_tp4_serve_request`
- `ds4_glm52_mock_serve_observation`
- `tests/test_glm52_tp4_mock_serve.c`
- Makefile test wiring
- BCD evidence files under `dev-artifacts/validation/*mock-serve*`

Pinned contract:

- Blueprint hash:
  `3466079cab4a6e6d8011d675c3cf429d7cda2bc961c2d5ca006e3087af317b6c`
- Semantic hash:
  `3177dc06506cb955e519157ab2f2125472287223a6e5f60ded7e3a95ecead023`

## Obligation Trace

- TP4/DCP4 topology: code initializes four rank descriptors with
  `DS4_GLM52_L0_TP_SIZE`, `DS4_GLM52_L0_DCP_SIZE`, and
  `DS4_GLM52_L0_RANK_COUNT`; test asserts all three.
- Rank plan ownership: code verifies Q-head tiling and vocab tiling before
  running the request; test asserts both observations.
- Prefill: code calls `ds4_glm52_mock_tp4_prefill`, records prompt-length
  cursor/KV state, and checks replicated hidden/cursor state; test asserts
  `prefill_token_step_j == prompt_length` and
  `prefill_kv_length == prompt_length`.
- Decode: code consumes the coordinator-selected prefill token, calls
  `ds4_glm52_mock_tp4_decode`, and records the next cursor/KV state; test
  asserts both advance by one.
- Logits/top-k gather: code creates complete TP4 rank steps and records
  coordinator token/rank from rank-local vocab candidates; test asserts the
  selected token is rank-owned and candidate ranks are valid.
- Real-component boundary: evidence explicitly states real kernels, real
  network collectives, real checkpoint loading, Bird/vLLM oracle parity, and
  GX10 performance remain future implementation leaves.

## Evidence

- Code gate:
  `make -B tests/test_glm52_tp4_mock_serve tests/test_glm52_tp4_allreduce
  tests/test_glm52_dcp_row_exchange tests/test_glm52_tp4_mock
  tests/test_glm52_tp4_mock_process tests/test_glm52_mock
  tests/test_glm52_l0 tests/test_tp4_rank_group`
- Runtime gate:
  `./tests/test_glm52_tp4_mock_serve`,
  `./tests/test_glm52_tp4_allreduce`,
  `./tests/test_glm52_dcp_row_exchange`,
  `./tests/test_glm52_tp4_mock`,
  `./tests/test_glm52_tp4_mock_process`,
  `./tests/test_glm52_mock`,
  `./tests/test_glm52_l0`,
  `./tests/test_tp4_rank_group`
- Build gate: `make cpu`
- Hygiene gate: `git diff --check`
- BCD mechanics:
  `status-after-mock-serve.result.json`,
  `lint-mock-serve.result.json`,
  `validate-mock-serve-tp4-leaves.result.json`,
  `validate-mock-serve-decode-composite.result.json`,
  `validate-mock-serve-serve-composite.result.json`,
  `simulate-mock-serve-decode-full-trace.result.json`
- Render: `dev-artifacts/ds4-arch-review.html`, PASS with 14 graphs, 135
  nodes, 179 edges, 0 warnings.

## Findings

No blockers or nonblocking defects found.

Follow-ups remain the declared real-component frontier:

- GLM 5.2 TP4 CUDA QKV/MLA kernels.
- Real DCP selected-row exchange over GX10 ranks.
- Real all-reduce/gather device tensor transport.
- Rank-specific GLM 5.2 checkpoint loader.
- Bird/vLLM oracle parity and GX10 performance validation.
