# L1 Model-Load Implementation Evidence

Date: 2026-08-07

Scope: first L1 realization under `serve/action-serve-open`, preserving the
current BCD contract for `model-load`.

## Contract Mapping

- `model-load/a-validate-launch-plan`
  - Code: `ds4_glm52_l0.c`, `load_plan_and_manifest`.
  - Checks `tp_size=4`, `dcp_size=4`, `pp_size=1`, four ranks/four GX10 hosts,
    local rank match, and GLM 5.2 model identity.
- `model-load/a-validate-shard-manifest`
  - Code: `ds4_glm52_l0.c`, `load_plan_and_manifest`.
  - Checks rank-local shard entries resolve to regular files, exactly 20 base
    shards plus one MTP shard are present, and current-rank entries do not point
    at foreign `rankN` paths.
- `model-load/a-map-rank-shards`
  - Code: `ds4_glm52_l0.c`, `DS4_GLM52_L0_ACTION_SERVE_OPEN`.
  - Still strict `not_ready`. The implementation records validated manifest
    evidence but does not claim tensor mmap/resident engine readiness.
  - Delegated implementation dependency: the separate `model-shard-layout`
    slice must publish mapped rank-local tensor slices before this parent seam
    can proceed to rank engine readiness.
- `model-load/a-ready-rank-engines`
  - Still downstream strict `not_ready`, because rank engine residency depends
    on the unresolved mapper and TP fabric work.
- `serve/action-tp-group`
  - Code: `ds4_glm52_l0.c`, `bind_tp_group`.
  - When resident rank-local shards are present, binds TP4/DCP4/PP1 topology,
    local rank id, Q-head ownership span, DCP rank, rank count, and fabric
    address into production state.
  - Still strict `not_ready` for real four-rank collective transport handshake;
    topology binding does not claim broadcast/all-reduce/logits-gather runtime
    transport is ready.

## Production State Evidence

- Header state: `ds4_glm52_l0_launch_plan`,
  `ds4_glm52_l0_shard_manifest`, and extended
  `ds4_glm52_l0_resident_rank_shards`.
- `serve/action-serve-open` now validates model-load child state before stopping
  at `model-load/a-map-rank-shards`.
- The parent root still reports `serve/action-serve-open` as the blocked action,
  so L0 boundary behavior remains fail-closed.
- KV/cursor state remains unmodified by unresolved decode/prefill seams.

## Orchestration Preconditions

- `serve/action-tp-group` now fails closed until resident rank-local model
  shards are present, then binds topology while leaving real transport
  readiness unresolved.
- `serve/action-serve-accept` now fails closed until resident rank engines and
  the TP4/DCP4 fabric are ready.
- `serve/action-serve-prefill` now fails closed in dependency order: resident
  model shards, TP4/DCP4 fabric, then accepted request/session.
- `serve/action-decode-token` now fails closed until prefill has committed
  append-ordered TP-aware KV state with a replicated cursor.
- `serve/action-stream-token` now fails closed until a sampled token is present.
- `serve/action-kv-checkpoint` now fails closed when checkpointing is requested
  before TP-aware KV/cursor state exists.

The delegated shard-layout implementation remains the only writer of the
resident mapped-shard success condition. The parent skeleton records the
required interface and blocks downstream orchestration until that condition is
true.

## Validation

- `make tests/test_glm52_l0 && ./tests/test_glm52_l0`: PASS.
- Added focused tests for orchestration dependency failures:
  resident-shards-before-TP, TP-before-prefill, request-before-prefill,
  prefill-cursor-before-decode, and token-before-stream.
- Added focused tests for TP-group topology binding: valid rank-local resident
  shards bind rank/Q-head/DCP/fabric topology, missing fabric is rejected, and
  resident-shard rank mismatch is rejected.
- `make cpu`: PASS.
- CLI smoke with a real temporary rank-2 plan and 20+1 shard fixture: expected
  `rc=1`; message names `model-load/a-map-rank-shards` after launch/manifest
  validation.
- `git diff --check`: PASS.
- BCD lint: `dev-artifacts/validation-lint-current.json`, PASS.
- BCD serve L0 validation: `dev-artifacts/validation-serve-l0-current.json`,
  PASS.
- BCD model-load leaf validation:
  `dev-artifacts/validation-model-load-current.json`, PASS.
- BCD decode composite validation:
  `dev-artifacts/validation-decode-composite-current.json`, PASS.
- BCD decode simulation:
  `dev-artifacts/validation/simulate-decode-current-full-trace.result.json`,
  PASS.
- Review HTML:
  `dev-artifacts/ds4-arch-review.html`, PASS render with 10 graphs, 113 nodes,
  161 edges, 25 types, and 0 warnings.

## Remaining Frontier

The delegated implementation frontier is `model-shard-layout`: parse the
DS4-native GLM 5.2 TP4 tensor shard manifest, validate ownership spans and
hashes, mmap only current-rank tensors plus explicitly replicated tensors, and
publish the loaded rank shard state consumed by `model-load/a-map-rank-shards`.

The next non-delegated frontier is the real four-rank collective transport
handshake behind `serve/action-tp-group`. The current code binds topology but
does not claim transport readiness.
