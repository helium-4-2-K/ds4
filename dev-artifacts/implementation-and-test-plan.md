# DS4 GLM 5.2 GX10x4 Implementation and Test Plan

Date: 2026-08-07

Verdict: `READY_TO_IMPLEMENT_PHASED`

This plan follows the BCD implementation-readiness rule: the blueprint is the
contract, and implementation work must preserve the current graph behavior
unless a code discovery proves that the contract itself is wrong or incomplete.

## Pinned Contract

- Blueprint hash: `3466079cab4a6e6d8011d675c3cf429d7cda2bc961c2d5ca006e3087af317b6c`
- Semantic hash: `3177dc06506cb955e519157ab2f2125472287223a6e5f60ded7e3a95ecead023`
- Root graph: `serve`, graph hash `9212a3685de85a5df4e17c1471458e064e4c335988ae220f36f65731baca3c8a`
- Decode graph: `decode`, graph hash `2af92c0420b89ec8189e6c62009a4b879cb65a12c361d794476dd907904b0a1e`
- Decode DCP top-k graph: `decode-dcp-topk`, graph hash `2311682e85d881362a93c0be8eed1d36aad3e87844f612878ff85c816b3c3f24`
- Model-load graph: `model-load`, graph hash `0d3a26146444d194a72ed4f843fabbd3f6275eeefa45873465f5b9e2ac63c215`
- Model-shard-layout graph: `model-shard-layout`, graph hash `eb05725c27670f0c0d9c3b90044c040e3ee5e6130eca1e09e8d1b26a202b911e`
- Prefill graph: registered under current blueprint; validate before prefill implementation starts.
- Lowered data-plane child graphs:
  - `decode-attn-allreduce-sum`: TP4 attention hidden all-reduce invariants.
  - `decode-ffn-allreduce-sum`: TP4 FFN hidden all-reduce invariants.
  - `decode-logits-gather-topk`: TP4 vocab-shard gather/top-k invariants.
  - `decode-dcp-row-exchange`: DCP4 selected-row owner mapping and payload exchange.
- Validation evidence:
  - `dev-artifacts/validation/mutate-lower-tp4-data-plane.result.json`: PASS
  - `dev-artifacts/validation/validate-lowered-tp4-leaves.result.json`: PASS
  - `dev-artifacts/validation/validate-lowered-decode-composite.result.json`: PASS
  - `dev-artifacts/validation/validate-lowered-serve-composite.result.json`: PASS
  - `dev-artifacts/validation/status-after-lowering.result.json`: PASS
  - `dev-artifacts/ds4-arch-review.html`: regenerated, PASS render with 14
    graphs, 135 nodes, 179 edges.
  - `dev-artifacts/validation-lint-current.json`: PASS
  - `dev-artifacts/validation-serve-l0-current.json`: PASS
  - `dev-artifacts/validation-decode-composite-current.json`: PASS
  - `dev-artifacts/validation-model-load-current.json`: PASS
  - `dev-artifacts/validation-model-shard-layout-current.json`: PASS
  - `dev-artifacts/validation/simulate-model-shard-layout-current.result.json`: PASS
  - `dev-artifacts/validation/simulate-decode-current-full-trace.result.json`: PASS

## Current Implementation Status

Completed implementation slices:

- `serve` L0 production skeleton in `ds4_glm52_l0.c` /
  `ds4_glm52_l0.h`.
- Explicit GLM 5.2 TP4 mock-model execution seam:
  - `--glm52-tp4-mock-model` and `--glm52-tp4-mock-matmul` are CLI/server
    options, and both preserve the `--glm52-tp4-l0` rank contract;
  - the mock opens through `ds4_engine_open()` without a GGUF file only when
    the explicit mock flags are set;
  - sessions run through `ds4_session_sync()`, `ds4_session_eval()`, sampling,
    and token rendering with deterministic mocked logits;
  - tensor descriptors preserve GLM 5.2 full-shape metadata and rank-local
    TP4 ownership for Q heads and vocab shards without allocating real weights.
- Model-load validation skeleton:
  - validates launch plan;
  - validates rank-local shard manifest shape;
  - runs the `model-shard-layout` child when a DS4 layout manifest is supplied;
  - publishes `model-load/a-ready-rank-engines` only after validated rank-local
    layout ownership and file/byte-range residency evidence.
- `serve/action-tp-group` topology-binding skeleton:
  - binds TP4/DCP4/PP1;
  - binds local rank, Q-head span, DCP rank, rank count, and fabric address;
  - leaves real four-rank transport readiness unresolved.
- TP4 no-model rank-group smoke:
  - validates ranks 0..3 exactly once;
  - validates TP/DCP/model/config/plan hash compatibility;
  - supports rank0 command broadcast and all-rank acknowledgement;
  - frames hello/command/ack messages over file descriptors;
  - proves a forked local coordinator/worker process handshake over
    `socketpair()`;
  - parses numeric TCP endpoints, binds a coordinator listener, accepts worker
    connections with timeout handling, and connects workers over loopback TCP;
  - clears group readiness when a rank transport failure is recorded;
  - publishes L0 fabric readiness only after the compatible four-rank group is
    complete.
- Focused L0 tests in `tests/test_glm52_l0.c`.
- Focused TP4 rank-group tests in `tests/test_tp4_rank_group.c`.
- Prefill L0 state machine under `serve/action-serve-prefill` (expands to the
  `prefill` child graph):
  - `ds4_glm52_l0_prefill_set_prompt` binds the leader-owned prompt token span
    (`prefill/a-tokenize-prompt`);
  - `ds4_glm52_l0_prefill_run` plans bounded prompt chunks with the BCD 1024
    reference chunk (`prefill/a-plan-prefill-chunks`) and validates rank-plan /
    model-shard / fabric identity and TP-aware append-ordered KV identity;
  - non-mock execution stops `not_ready` before `prefill/a-prefill-layer-tp`
    mutates KV/cursor state because real GLM 5.2 TP prefill kernels and
    all-reduce are still open;
  - explicit mock execution may continue through mock layer/all-reduce seams,
    commit the prompt prefix with explicit append-order validation
    (`prefill/a-commit-prefill-kv`), and publish decode handoff readiness
    (`prefill/a-prefill-ready`);
  - `kv.cursor.phase` records the producing phase (PREFILL/DECODE) so decode can
    start only from a prefill-produced or otherwise valid decoded cursor;
  - still strict `not_ready`/no-commit for the real GLM 5.2 TP layer kernels
    and TP all-reduce seams.
- Production execution frontiers:
  - typed real TP4 all-reduce request validation rejects wrong ranks,
    participant masks, identity, dtype, layer, and tensor shape evidence before
    execution;
  - typed real DCP selected-row exchange validation rejects incomplete owner
    maps, invalid identity, and non-append-ordered KV evidence before
    execution;
  - typed real decode-step validation requires prefill-ready cursor, resident
    model readiness, TP4 collectives, DCP exchange, and GLM 5.2 kernels, and
    preserves KV/cursor/token state while the execution backend is not wired;
  - deterministic DCP row-exchange mock now publishes replies only after all
    requests validate, sorts by query layer/token position, and rejects
    out-of-bound selection counts.

Delegated implementation slice:

- `model-shard-layout`: DS4-native layout parser, tensor ownership validator,
  mmap of rank-local/replicated tensors, and loaded-rank-shard publication.

Open implementation slices before real serving:

- cross-machine deployment policy for rank endpoint files/CLI wiring;
- real collective backend execution for the lowered
  `decode-attn-allreduce-sum`, `decode-ffn-allreduce-sum`, and
  `decode-logits-gather-topk` contracts;
- real DCP selected-row network exchange execution for
  `decode-dcp-row-exchange`;
- request/session binding;
- real TP4/DCP4 prefill kernels/collectives (prefill L0 state machine is done);
- decode skeleton and then real GLM 5.2 TP4/DCP4 kernels/collectives;
- server integration and streaming after token commit.

## Implementation Scope

Implement DS4-native serving for GLM 5.2 on 4x GX10 machines using tensor
parallelism with `tp_size = 4` and distributed context parallel sparse attention
with `dcp_size = 4`.

The first production target is one rank-0 server process plus three worker rank
processes. Rank 0 owns the external API, token selection, and client-visible
streaming. All ranks participate in model load, prefill, decode, per-layer
collectives, compact KV ownership, and cursor advancement.

Initial exclusions and clarifications:

- MTP/speculative decode execution. The model-load path still has to validate
  rank-local MTP/speculator shard presence when the Bird/DwarfStar checkpoint
  layout requires it; DS4 must not silently ignore or misplace those files.
- Multiple concurrent active decode sequences.
- Arbitrary TP/DCP sizes beyond the fixed GLM 5.2 GX10x4 target.
- Pipeline/layer parallelism.
- Silent fallback to existing two-Mac TP behavior.

## Test Ladder

Testing starts immediately and becomes more concrete as each frontier is
realized. A phase may not claim readiness by skipping the lower-rung tests.

### Rung 1: BCD Contract Tests

Purpose: prove the graph contract remains mechanically coherent while the code
changes.

Already active:

- BCD lint for the complete blueprint.
- `serve` L0 validation.
- `model-load` composite validation.
- `model-shard-layout` leaf validation.
- `decode` composite validation.
- `model-shard-layout` simulation.
- `decode` full-trace simulation.

Required after any blueprint-impacting change:

- Re-run the affected graph validation.
- Re-run any ancestor validation whose boundary was touched.
- Re-run simulations covering the changed graph or parent-visible behavior.
- Regenerate `dev-artifacts/ds4-arch-review.html`.

### Rung 2: Production Skeleton Tests

Purpose: prove the real code follows the BCD ordering and fails closed rather
than faking success.

Already active in `tests/test_glm52_l0.c`:

- root action order;
- configuration rejection for wrong TP/DCP/PP/rank/model/mixed legacy TP;
- model-load validation reaches `model-load/a-map-rank-shards`;
- model-load with a valid DS4 layout reaches
  `model-load/a-ready-rank-engines`;
- missing rank shards and foreign rank shards fail;
- TP group waits for resident shards;
- TP group binds topology but leaves real transport unresolved;
- prefill waits for resident shards, TP fabric, and accepted request;
- decode waits for prefill-complete KV cursor;
- stream waits for sampled token;
- unresolved decode does not mutate KV/cursor state.
- real TP4 collective, real DCP row-exchange, and real decode frontiers reject
  invalid inputs and remain `not_ready` without mutating KV/cursor while the
  concrete backend is unwired.

Required while L0/L1 skeleton work continues:

- Every new strict stub must have a test proving its first missing prerequisite.
- Every new partial implementation must have a test proving what state it
  updates and what state it deliberately does not claim.
- `make tests/test_glm52_l0 && ./tests/test_glm52_l0` must pass after every
  skeleton change.

### Rung 2.5: Full-Shape Mock Model Tests

Purpose: let the server/session path run before real GLM 5.2 TP4 kernels and
real GX10 hardware are available, while keeping the mock explicit and
shape-preserving.

Active in `tests/test_glm52_mock.c`:

- mock model init requires both `mock_model` and `mock_matmul`;
- rank 2 owns Q heads `[32,48)` under TP4;
- full GLM 5.2 vocab size is preserved as `154880`;
- descriptors include replicated embedding/norm, replicated MLA KV projection,
  rank-local Q-head tensors, and rank-local vocab output shard;
- `ds4_engine_open()` returns a synthetic GLM 5.2 TP4 engine only under the
  explicit mock flags;
- `ds4_tokenize_rendered_chat()`, `ds4_session_create()`,
  `ds4_session_sync()`, `ds4_session_eval()`, `ds4_session_argmax()`, and
  `ds4_token_text()` work through the normal public API;
- prefill advances the checkpoint to prompt length and decode advances it by
  exactly one token.

Required next:

- HTTP smoke remains useful for server plumbing, but it is not the TP4 proof.
- server-path mock TP4 wiring should reuse the process mock protocol instead
  of adding a second rank contribution path;
- contract-to-code review for the mock seam whenever the model-load or decode
  contracts gain a first-class mock/simulation variant.

### Rung 2.6: TP4 Mock Orchestration Tests

Purpose: prove the TP4 control contract in mock before real tensor kernels are
available. This rung demonstrates that four rank-local model/session states
participate in the same prefill/decode command with one logical model identity,
one session identity, matching cursor/KV state, complete Q-head ownership, and
rank-local vocab candidates gathered to a coordinator-visible token.

Active in `tests/test_glm52_tp4_mock.c`:

- four mock ranks initialize with one shared logical model hash and distinct
  rank-local Q-head/vocab spans;
- the existing TP4 rank-group state machine registers ranks 0..3, broadcasts
  `PREFILL`, and completes only after all four rank contributions acknowledge
  the same command sequence;
- a TP4 mock prefill step rejects duplicate rank contributions, sequence
  mismatches, model identity mismatches, cursor/KV mismatches, and invalid
  rank-local vocab candidates;
- mock rank-step and token-span transport helpers round-trip valid frames and
  reject corrupt frames, truncated frames, and receiver capacity overflow;
- complete prefill proves Q-head coverage `[0,64)` exactly once and contiguous
  vocab-shard coverage `[0,154880)`;
- coordinator token selection is gathered from a rank-local vocab candidate,
  not invented outside rank ownership;
- all ranks consume the gathered token in a mock `DECODE` step, advance to the
  same prompt-length `token_step_j`/`kv_length`, and complete the second
  rank-group command.

Active in `tests/test_glm52_tp4_mock_process.c`:

- rank 0 binds a local TCP rendezvous endpoint and forks three worker ranks;
- each worker initializes a rank-local full-shape GLM 5.2 mock model, connects
  to rank 0, and registers through the TP4 hello/rank-group protocol;
- rank 0 broadcasts `PREFILL` and sends the prompt token span over a mock
  token-span frame, then receives rank-local mock contribution frames and
  command acknowledgements from workers;
- rank 0 validates the complete TP4 prefill step, gathers the coordinator token
  from rank-owned vocab candidates, broadcasts `DECODE`, sends the gathered
  token, and validates that every rank advances to the same cursor/KV state;
- rank 0 broadcasts `SHUTDOWN`, records all worker acknowledgements, and
  verifies every worker exits cleanly.

Active in `tests/test_glm52_tp4_mock_serve.c`:

- `ds4_glm52_mock_tp4_serve_request` initializes four full-shape mock rank
  descriptors and records one serving request observation;
- the observation proves TP4/DCP4/rank-count identity, shared model/session
  identity, Q-head tiling, global vocab tiling, rank-owned top-k selection,
  prefill cursor/KV advancement to prompt length, decode cursor/KV advancement
  by exactly one token, replicated hidden/cursor state, and rank-local logits
  summary publication for top-k gather;
- invalid prompt/observation inputs are rejected before any request is claimed.

This proves TP4 mock orchestration, process/rank identity, local TCP command
transport, shared command/cursor agreement, and rank-local output-gather
semantics. It also now proves the executable mock serving-request seam through
prefill plus one decode token. It still does not prove real QKV/MLA kernels,
real DCP network row exchange, real all-reduce device tensor math,
cross-GX10 deployment, or GX10 transport performance.

### Rung 3: DS4-Native Shard-Layout Fixture Tests

Purpose: prove model loading is architecturally correct before real model
files and transport are involved.

Active in `tests/test_glm52_l0.c`:

- valid TP4 rank-local manifest passes;
- model-load with a valid layout publishes ready resident rank engines;
- wrong `tp_size` fails;
- foreign-rank tensor mapping fails;
- missing required base shard fails;
- byte range outside the file fails;
- replicated tensor is visible on every rank;
- sharded tensor is visible only on its owner rank;
- Q-head ownership covers the current rank's 16-head span exactly once;
- missing Q-head span fails;
- overlapping Q-head span fails;

Still to add:

- full sha256 recomputation failure;
- real mmap handle presence;
- missing required MTP/speculator shard failure derived from manifest roles;
- expert and vocab span coverage completeness;
- `kv_lora[512]` replicated activation semantics and DCP-owned committed row
  semantics in the real kernel path.

Suggested test file:

- `tests/test_glm52_shard_layout.c` or an expanded `tests/test_glm52_l0.c`
  section if the loader API stays in `ds4_glm52_l0.*`.

### Rung 4: Four-Rank No-Model Runtime Smoke

Purpose: prove process/rank orchestration before GLM kernels are involved.

Already active in `tests/test_tp4_rank_group.c`:

- four fake local ranks register and make the rank group ready;
- missing rank keeps the group unready;
- duplicate rank fails;
- wrong `tp_size` / `dcp_size` fails;
- mismatched model/config/plan hash fails;
- only rank 0 can broadcast a command;
- all ranks must acknowledge a command before it is complete;
- framed FD hello rejects mismatched rank metadata;
- wrong frame type and truncated frame reject before registration;
- rank 0 broadcasts a shutdown command to three forked worker ranks over
  `socketpair()`;
- forked workers acknowledge clean shutdown over the same FD transport;
- endpoint parser accepts `host:port` and rejects malformed endpoints;
- coordinator listener binds an ephemeral loopback port and reports the bound
  port;
- missing worker accept times out with a typed error;
- rank0 accepts three forked TCP worker ranks over loopback;
- TCP worker hellos register into the same rank group state machine;
- rank0 broadcasts shutdown and records TCP worker acknowledgements;
- recorded transport loss clears `transport_ready` / `group_ready`;
- publishing fabric readiness unblocks the L0 prefill prerequisite and reaches
  the strict prefill child stub.

Tests still to add with the cross-machine transport handshake:

- rank endpoint config is loaded from the real GX10 launch plan;
- workers retry/back off according to deployment policy;
- multi-host smoke proves the same endpoint protocol across GX10 machines.

Suggested test file:

- `tests/test_tp4_rank_group.c`.

### Rung 5: Collective Semantics Tests

Purpose: prove distributed math primitives cannot consume the wrong rank,
layer, cursor, or shape.

Active in `tests/test_glm52_l0.c` and DCP tests:

- real TP4 collective frontier rejects missing rank participants and missing
  tensor transport before execution;
- real DCP row-exchange frontier rejects incomplete owner maps and missing row
  payload/transport before execution;
- DCP row-exchange mock rejects invalid selection counts, never publishes
  partial replies on failure, and orders replies deterministically.

Tests to add with real collective execution:

- all-reduce sum over four rank-local hidden partials;
- logits gather or distributed top-k to rank 0;
- DCP local top-k candidate merge into global top-k;
- selected-row owner mapping and selected-row byte exchange;
- sequence mismatch by layer/token cursor is rejected;
- shape mismatch is rejected before collective execution.

Suggested test files:

- `tests/test_glm_tp4_collectives.c`;
- `tests/test_glm_dcp_topk.c`.

### Rung 6: Prefill/Decode Correctness Tests

Purpose: prove GLM 5.2 TP4/DCP4 behavior against single-process DS4 and then
Bird/vLLM oracle traces.

Tests to add after collectives exist:

- rank-owned Q-head spans are correct for every layer;
- attention/output partials all-reduce to the single-process reference in a
  dense-equivalent fixture;
- FFN/expert partials all-reduce to the reference;
- prefill chunk cursor is monotonic;
- prefill commits DCP-owned KV rows and a replicated cursor;
- decode advances cursor exactly once per committed sampled token;
- failed collective does not advance cursor;
- selected sparse row ids match Bird/vLLM traces;
- final logits/top-k match Bird/vLLM within agreed tolerance;
- greedy short continuation matches oracle tokens.

Suggested test files:

- `tests/test_glm_tp4_decode_math.c`;
- `tests/test_glm_tp4_prefill.c`;
- `tests/test_engine_mgpu_runtime.c` extensions;
- Bird/vLLM trace harness under `tests/` or `dev-artifacts/oracle/`.

### Rung 7: Real 4x GX10 Smoke and Performance Tests

Purpose: prove the deployed target works and supports DS4's lightweight-serving
claim.

Required hardware tests:

- real 4x GX10 launch with rank-specific shard files;
- one short prompt, one-token decode;
- prompt prefill plus streaming decode;
- long-context sparse selection smoke;
- worker failure during prefill and decode;
- startup/load time, memory per rank, prefill tokens/sec, decode ms/token,
  selected-row bytes/token, collective latency.

Correctness gates must pass before performance claims are recorded.

## Current Code Binding

The existing native TP seam is two-rank focused:

- `ds4_tp.h`: public TP control/exchange API; `ds4_tp_create`,
  `ds4_tp_gate_exchange`, `ds4_tp_worker_run`.
- `ds4_tp.c`: two-rank leader/worker setup and transport; CLI still describes
  one 50/50 worker and rejects explicit layer slices.
- `ds4.c`: `ds4_engine_tp_bind` binds current TP into the engine and hard-codes
  GLM graph `tp_world = 2` in session setup.
- `ds4_server.c`: rank-0 serving integration point for the external API.

The existing GLM compute and cache seams are usable targets:

- `ds4.c`: GLM shape/config/layout validation, including `DS4_SHAPE_GLM52`,
  `config_validate_glm_dsa_model`, `weights_validate_glm_dsa_layout`,
  `attn_k_b`, `attn_v_b`, `DS4_N_KV_LORA`, and `DS4_N_INDEXER_TOP_K`.
- `ds4.c`: GLM execution paths `glm_graph_forward_token`,
  `glm_graph_forward_tokens`, `glm_graph_forward_indexed_tokens`,
  `glm_graph_indexed_prefill_batch_ready`, and
  `glm_graph_maybe_warm_compact_indexer_after_prefill`.
- `ds4_gpu.h` and `ds4_cuda.cu`: compact KV and sparse indexer APIs, including
  `ds4_gpu_glm_store_compact_kv_tensor`,
  `ds4_gpu_glm_indexer_score_one_tensor`,
  `ds4_gpu_glm_indexer_scores_batch_tensor`, and
  `ds4_gpu_indexer_topk_tensor`.
- `tests/test_engine_mgpu_runtime.c`: existing GPU-vs-GPU correctness pattern
  with prefill/decode delta gates.
- `tests/test_engine_mgpu_placement.c`: existing CUDA TP accounting tests.
- `QA_BEFORE_RELEASES.md`: existing release gates mention GLM long-context,
  Metal session oracle, and physical two-machine GLM TP.

## BCD Contract Obligations

Implementation must preserve these user-visible and graph-visible obligations:

- Four ranks load one GLM 5.2 model plan with matching model/config/shard
  identity before serving.
- DwarfStar launch settings match the four-rank Bird/vLLM baseline: TP4, PP1,
  DCP4, rank-specific sharded checkpoint directories, and fabric endpoints.
- Every token step has an explicit token position and KV cursor.
- Prefill commits prompt KV rows and leaves decode at the next cursor position.
- Decode advances the replicated cursor by exactly one accepted token.
- Each layer executes rank-owned Q heads and attention/value work, then combines
  partial hidden contributions at the attention/output gate.
- DCP sparse attention scores local KV row ownership, merges global top-k
  candidates, exchanges selected rows, and computes attention over the selected
  global rows.
- MoE routing is global-top-k over experts, with rank-owned expert execution and
  hidden all-reduce.
- Rank 0 produces the sampled token and broadcasts it to workers before the
  next decode position.
- Optional KV persistence/checkpoint policy can durably checkpoint rank-local
  TP KV shards and replicated cursor metadata.
- Failure must be explicit for rank mismatch, shard mismatch, cursor divergence,
  collective timeout, and unsupported GLM layout.

## Phase L0: Production Skeleton Realization

Implementation:

- Add the actual DS4 production composition root for GLM 5.2 GX10x4 serving,
  behind an explicit experimental mode.
- Bind the root `serve` graph to real target-language seams:
  - request accept / stream seam in `ds4_server.c`.
  - engine/session state in `ds4.c`.
  - TP rank-group state in `ds4_tp.h` / `ds4_tp.c` or a narrow sibling module
    if the existing two-rank API cannot carry four-rank state cleanly.
  - GLM 5.2 TP4 model-load state in a dedicated loader module.
- Define production structs for the L0 state, even before the child actions are
  implemented: model plan, resident rank shards, rank plan, TP fabric, TP-aware
  KV state, replicated KV cursor, request/session, token, logits summary, and
  checkpoint status.
- Wire the L0 action sequence through real code with strict stubs:
  - open/load TP shards.
  - form TP=4 collective group.
  - accept/enqueue request.
  - prefill prompt.
  - decode token.
  - stream token.
  - optional KV checkpoint.
- Every unresolved action stub must preserve the parent boundary: consume the
  same inputs/state, return a typed unsupported/not-ready failure, and avoid
  mutating KV/cursor state unless its contract is actually implemented.
- Add trace/log events with graph/action ids so contract-to-code review can map
  runtime behavior back to `serve` L0.

Tests:

- Build with the experimental mode enabled and disabled.
- Server/engine smoke reaches every GLM 5.2 TP4 root action in L0 review mode,
  preserves KV/cursor state, and returns a typed not-ready error naming the
  first unresolved action.
- Stub mutation test proves failed unresolved actions do not advance
  `state-kv` or `state-kv-cursor`.
- Action-order test verifies the L0 sequence is `open -> form TP group ->
  accept -> prefill -> decode/stream loop -> optional checkpoint`.
- Existing non-TP and current supported TP tests continue to pass.

Exit gate:

- L0 compiles in production code, is reachable through the real outer seam, and
  has strict stubs for every unresolved child action.
- A contract-to-code review can map each `serve` action/state/effect to a real
  code seam or a strict failing stub.
- No L1 implementation phase may be called contract-ready until this skeleton
  passes.

## Phase 0: Baseline Guardrails

Implementation:

- Add a fixed experimental mode for GLM 5.2 GX10x4 TP4+DCP4, behind explicit CLI
  and engine options.
- Make startup reject `tp_size != 4`, `dcp_size != 4`, non-GLM5.2 model shape,
  and mixed two-rank TP options in this mode.
- Add a runtime capability record carrying `rank`, `tp_size`, `dcp_size`,
  model hash, config hash, rank-plan hash, and transport kind.
- Validate the DwarfStar launch plan carries `tp_size = 4`, `pp_size = 1`,
  `dcp_size = 4`, four GX10 ranks, rank-specific checkpoint roots, and the
  expected host/fabric endpoint mapping.
- Keep existing two-rank TP behavior available for current supported flows.

Tests:

- CLI rejects missing `--tp-size 4` or incompatible two-rank flags.
- GLM TP4 mode rejects DeepSeek-only CUDA TP path assumptions.
- Current non-TP and current supported TP tests continue to pass.

Exit gate:

- Code compiles with the feature enabled and disabled.
- Existing tests pass without changing expected behavior.

## Phase 1: TP4 Rank Runtime

Implementation:

- Replace the single-peer TP runtime shape with a rank group abstraction.
- Rank 0 coordinates ranks 1..3; every rank completes a hello/barrier that
  checks rank id, total rank count, DCP size, model/config identity, and plan
  identity.
- Add rank-group broadcast for commands: load, prefill chunk, decode step,
  sample result, shutdown.
- Add collective deadlines and error propagation so rank 0 can fail the external
  request instead of hanging.
- Prefer NCCL/RoCE collectives on GX10 for production performance; keep any TCP
  fallback clearly marked as diagnostic, not the performance target.

Tests:

- `tests/test_tp4_rank_group.c`: local fake transport rendezvous, barriers,
  command broadcast, clean shutdown.
- Mismatch tests: duplicate rank, missing rank, wrong `tp_size`, wrong
  `dcp_size`, wrong model hash, wrong plan hash.
- Timeout test: one worker stalls; rank 0 returns a typed failure.

Exit gate:

- Four local fake ranks can synchronize and shut down deterministically.
- All mismatch cases fail before model execution.

## Phase 2: Collective Substrate

Implementation:

- Add typed collectives needed by the blueprint:
  - `decode-attn-allreduce-sum`: four rank-local attention partials for the
    same session/layer/position/dtype/shape hash reduce to identical hidden
    output on every rank.
  - `decode-ffn-allreduce-sum`: four rank-local FFN partials reduce to the
    next replicated hidden with the same epoch and failure rules.
  - `decode-logits-gather-topk`: rank-local contiguous vocab shards cover the
    vocabulary exactly once and merge deterministic top-k candidates on rank 0.
  - `decode-dcp-row-exchange`: DCP ranks score owned MLA/index rows, merge
    global top-k, and fetch owner-mapped selected row payloads before attention.
- Bind collectives to real GLM dimensions: hidden `6144`, `64` heads,
  `16` Q heads per rank, `kv_lora = 512`, and GLM sparse indexer top-k.
- Add per-collective sequence numbers tied to layer id and token cursor.

Tests:

- `tests/test_glm_tp4_collectives.c`: deterministic all-reduce, gather, top-k
  merge, and selected-row exchange using fake ranks.
- Counterexample: each rank has a high local candidate but only the global top-k
  rows survive.
- Sequence mismatch test: wrong layer or cursor is rejected.

Exit gate:

- Collective tests prove rank-local outputs cannot be consumed at the wrong
  layer or token position.

## Phase 3: Rank-Shard Model Loading

Implementation:

- Define a narrow GLM 5.2 TP4 shard manifest with:
  - global model/config hash.
  - rank id and total ranks.
  - tensor ownership spans.
  - replicated tensors.
  - sharded tensors.
  - expected base and MTP/speculator shard files, where the checkpoint layout
    includes MTP files even though speculative execution is deferred.
  - local file hashes and byte spans.
- Load only tensors allowed for the current rank.
- Validate complete ownership coverage across ranks before serving.
- Keep `kv_lora[512]` projection semantics explicit: `kv_lora` activations are
  replicated per token before DCP row ownership; compact KV rows are DCP-owned
  after commit.

Tests:

- `tests/test_glm_tp4_manifest.c`: happy path, missing shard, foreign shard,
  wrong file hash, wrong rank id, overlapping spans, missing spans.
- GLM layout test validates `attn_k_b` and `attn_v_b` shard ownership matches
  the planned head ownership.

Exit gate:

- No GLM TP4 rank reaches prefill/decode until model identity and shard
  coverage are proven.

## Phase 4: Base Decode TP4 Scaffold Without DCP Row Sharding

Implementation:

- First decode milestone uses replicated compact KV to validate TP math before
  adding DCP row ownership.
- This is an internal scaffold only. It is not a contract-complete serving mode
  and must not be exposed as GLM 5.2 TP4+DCP4 serving readiness.
- Each rank runs its 16 Q heads and matching output-projection partial.
- Apply model position treatment to Q/K for the current cursor.
- All-reduce the attention/output contribution into the replicated hidden
  vector.
- Execute rank-owned routed experts and all-reduce FFN partial hidden output.
- Shard logits or gather rank-local logits to rank 0 for sampling.
- Rank 0 samples/chooses the next token and broadcasts it to all ranks.
- Cursor advances only after every rank acknowledges the accepted token.

Tests:

- `tests/test_glm_tp4_decode_math.c`: synthetic layer partials and all-reduce
  equality against a single-rank reference.
- Per-layer diagnostics compare rank-owned Q/K/V/head ranges.
- Short greedy decode compares TP partial math against a single-process
  DS4/debug reference with DCP disabled or made dense-equivalent. Bird/vLLM
  TP4+DCP4 parity is reserved for Phase 5 after DCP row ownership is enabled.
- Cursor test proves failed collectives do not advance the cursor.

Exit gate:

- A short GLM 5.2 prompt can exercise the TP4 math scaffold without exposing a
  false production-ready path. Contract-complete decode remains blocked until
  Phase 5 passes.

## Phase 5: DCP4 Sparse Attention

Implementation:

- Partition committed compact KV rows by DCP ownership across the four ranks.
- For each decode layer, each rank scores only its local owned rows.
- Merge local candidates into one global top-k set.
- Broadcast selected row descriptors: global row id, owner rank, cache position,
  and score.
- Exchange selected `kv_lora[512]` and rope rows needed by each rank's local
  heads.
- Compute attention over selected global rows, then feed the existing TP
  attention/output all-reduce.
- Add the minimal Bird/vLLM trace adapter needed to compare selected sparse row
  ids and per-layer attention outputs for this phase. Phase 9 expands this into
  the full oracle/performance harness.

Tests:

- `tests/test_glm_dcp_topk.c`: local candidates, global top-k merge,
  deterministic tie break, selected-row owner mapping.
- Selected-row exchange test checks row bytes and row ids, not just counts.
- Compare selected row ids against Bird/vLLM sparse attention traces.
- Long-context smoke with sparse selection enabled.

Exit gate:

- DCP selected rows and attention output match the oracle on short and
  long-context fixtures.

## Phase 6: Prefill

Implementation:

- Add TP4 prefill chunks for prompt tokens.
- Use the same rank-owned Q/head/expert/logit ownership rules as decode.
- Commit compact KV rows in DCP-owned layout as each chunk completes.
- Maintain a first-class cursor state: prompt length after prefill, then decode
  token position `j`.
- Handoff to decode only when every rank has committed the same cursor.

Tests:

- `tests/test_glm_tp4_prefill.c`: chunk cursor monotonicity, prompt KV row
  ownership, prefill/decode cursor handoff.
- Token-by-token diagnostic compares chunked prefill to repeated decode for a
  small prompt.
- Run `tests/glm_long_context_smoke.sh` with DCP enabled on the GLM fixture.
- Extend `tests/test_engine_mgpu_runtime.c` style delta checks for TP4 prefill
  and decode.

Exit gate:

- Prefill and decode agree on cursor, KV ownership, and next-token logits across
  short, medium, and long prompts.

## Phase 7: Server Integration

Implementation:

- Rank 0 exposes the normal DS4 serving API.
- Workers run headless and receive load/prefill/decode/shutdown commands.
- Streaming sends client-visible tokens only after rank 0 commits the token and
  all ranks acknowledge cursor advancement.
- Add status fields: rank readiness, model hash, plan hash, cursor, transport,
  DCP enabled, selected-row bytes/token, collective latency.
- Add typed errors for worker loss, collective timeout, shard mismatch, and
  cursor divergence.

Tests:

- Local server smoke with fake workers.
- Real 4x GX10 server smoke: load, one prompt, streaming decode, clean shutdown.
- Failure injection: kill one worker during prefill and during decode.

Exit gate:

- Rank 0 can serve a real request and return clear errors when the rank group is
  unhealthy.

## Phase 8: KV Checkpoint and Eviction Policy

Implementation:

- Implement the optional serve-graph checkpoint branch for TP-aware KV state.
- Persist each rank's local fp8_ds_mla KV/MLA cache slice plus replicated cursor
  metadata.
- Checkpoint only after a token has been streamed or when an explicit eviction
  policy requires it.
- Restore/validate checkpointed cursor length and per-rank KV ownership before
  reusing a live-prefix checkpoint.

Tests:

- Checkpoint writes all four rank-local KV slices and one replicated cursor
  record.
- Restore rejects missing rank shard, cursor mismatch, stale session id, and
  wrong model/rank-plan hash.
- Eviction policy cannot remove a row needed by the committed cursor.

Exit gate:

- Optional checkpointing preserves the serve contract without becoming required
  for the normal decode path.

## Phase 9: Oracle, Performance, and Release Gates

Implementation:

- Harden the Bird/vLLM oracle trace runner for GLM 5.2 TP4+DCP4 across the full
  serving path.
- Record per-layer selected row ids, token logits, sampled token, cursor,
  communication bytes, and startup/load timing.
- Add a performance harness for startup time, prefill tokens/sec, decode
  ms/token, GPU memory per rank, selected-row bytes/token, and collective
  latency.

Tests:

- Official GLM logprob vectors where available.
- Bird/vLLM trace comparison for:
  - selected sparse row ids.
  - per-layer partial hidden reduce result.
  - final logits.
  - generated token sequence.
- 4x GX10 hardware benchmark with at least:
  - short prompt.
  - long prompt beyond local dense-attention feasibility.
  - repeated decode run for stability.

Exit gate:

- Correctness gates pass before any performance claim.
- Performance report shows DS4's lightweight target with concrete startup,
  memory, prefill, decode, and communication numbers.

## Phase-to-Contract Coverage

| Phase | Primary BCD graph/action coverage |
| --- | --- |
| Phase L0 | Whole `serve` L0 composition: `action-serve-open`, `action-tp-group`, `action-serve-accept`, `action-serve-prefill`, `action-decode-token`, `action-stream-token`, `action-kv-checkpoint`, and root states `state-model-leader`, `state-model-worker`, `state-tp`, `state-rank-plan`, `state-kv`, `state-kv-cursor` |
| Phase 0 | `serve/action-serve-open`, `serve/action-tp-group`, `model-load/a-validate-launch-plan` |
| Phase 1 | `serve/action-tp-group`, `serve/state-tp`, `serve/state-rank-plan` |
| Phase 2 | `decode/a-embed-broadcast`, `decode/a-attn-allreduce` -> `decode-attn-allreduce-sum`, `decode/a-ffn-allreduce` -> `decode-ffn-allreduce-sum`, `decode/a-logits-gather` -> `decode-logits-gather-topk`, `decode-dcp-topk/a-dcp-topk-merge` -> `decode-dcp-row-exchange` |
| Phase 3 | `model-load/a-validate-launch-plan`, `model-load/a-validate-shard-manifest`, `model-load/a-map-rank-shards`, `model-load/a-ready-rank-engines` |
| Phase 4 | Internal scaffold for `decode/a-qkv-shard`, `decode/a-attn-local`, `decode/a-attn-allreduce`, `decode/a-ffn-route`, `decode/a-ffn-local`, `decode/a-ffn-allreduce`, `decode/a-logits-gather`, `decode/a-sample`; not contract-complete without Phase 5 |
| Phase 5 | `decode/a-dcp-topk`, `decode-dcp-topk/a-dcp-topk-merge`, `decode-sparse-attention/a-sparse-attn-kernel`, `decode/state-kv` DCP ownership |
| Phase 6 | `prefill/a-tokenize-prompt`, `prefill/a-plan-prefill-chunks`, `prefill/a-prefill-layer-tp`, `prefill/a-prefill-allreduce`, `prefill/a-commit-prefill-kv`, `prefill/a-prefill-ready` |
| Phase 7 | `serve/action-serve-accept`, `serve/action-serve-prefill`, `serve/action-decode-token`, `serve/action-stream-token` |
| Phase 8 | `serve/action-kv-checkpoint`, `serve/state-kv`, `serve/state-kv-cursor` |
| Phase 9 | Whole `serve` operation and Bird/vLLM oracle evidence |

## Deterministic Validation Commands

Run after every blueprint-impacting change:

```sh
PY=/Users/helium/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3
BT=/Users/helium/.codex/skills/blueprint-centric-dev/scripts/blueprint_tool.py

$PY $BT call --request dev-artifacts/ds4-arch-bp/lint-new-validator.json
$PY $BT call --request dev-artifacts/ds4-arch-bp/vc-serve.json
$PY $BT call --request dev-artifacts/validation/simulate-model-shard-layout-current.request.json
$PY $BT call --request dev-artifacts/validation/simulate-decode-current-full-trace.request.json
```

Run after implementation changes:

```sh
make tests/test_glm52_l0 && ./tests/test_glm52_l0
make tests/test_tp4_rank_group && ./tests/test_tp4_rank_group
make cpu
git diff --check
```

Add the new TP4 tests to the normal test target as they land:

```sh
tests/test_tp4_rank_group
tests/test_glm_tp4_manifest
tests/test_glm_tp4_collectives
tests/test_glm_tp4_decode_math
tests/test_glm_dcp_topk
tests/test_glm_tp4_prefill
```

## Phase Review Rule

At the end of every phase, write a short review record with:

- BCD graph actions covered.
- Code files changed.
- Tests added or updated.
- Contract obligations proven.
- Any contract gap discovered.
- Whether the next phase is unblocked.

Return to BCD design only if implementation proves that an input, output,
state, effect, ordering, authority, failure meaning, or inter-component
obligation must change. Local helper structure, file organization, loop shape,
transport wrappers, and defensive checks are implementation freedom when they
preserve the blueprint contract.
