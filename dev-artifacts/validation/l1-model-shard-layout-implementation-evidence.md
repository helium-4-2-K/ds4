# L1 model-shard-layout Implementation Evidence

Date: 2026-08-08

Scope: first L1 realization of the `model-shard-layout` child graph, which
expands `model-load/a-map-rank-shards`. This implements the DS4-native GLM 5.2
TP=4 model shard layout loader for 4x GB10/GX10 machines.

## Contract Mapping

The `model-shard-layout` graph defines four ordered actions. Each maps to a
code unit function in `ds4_glm52_l0.c`:

### a-read-layout-spec -> `ds4_glm52_layout_read_manifest`

- Code: `ds4_glm52_l0.c`, `ds4_glm52_layout_read_manifest`.
- Parses the DS4-native layout manifest (`ds4-shard-layout/v1` format).
- Validates: format version, tp_size=4, dcp_size=4, rank_count=4,
  model_config_sha256 present, at least one tensor entry.
- Produces a `ds4_glm52_layout_spec` with typed entries carrying tensor_name,
  role, distribution_scope, rank, file_path, byte_offset, byte_length, sha256,
  replicated flag, q_head/expert/vocab spans, storage dtype, logical tensor
  shape, element count, and stable dtype/shape hash.
- Validates tensor metadata before ownership: dtype must be supported, shape
  must be non-empty, element_count must match the shape product, byte_length
  must equal element_count times dtype size, and any manifest shape_hash must
  match the runtime derivation.

### a-validate-tensor-ownership -> `ds4_glm52_layout_validate_ownership`

- Code: `ds4_glm52_l0.c`, `ds4_glm52_layout_validate_ownership`.
- Validates: replicated tensors visible on all ranks, rank-local entries only
  visible on owning rank, no foreign-rank entries, no foreign-rank file paths,
  sha256 present for every entry, byte_length non-empty, Q-head coverage
  exactly once per rank (expected span [R*16, (R+1)*16)), expert-id coverage
  `[0,256)` on each rank (Bird/vLLM GLM 5.2 TP4 shards expert matrix
  dimensions, not the expert-id dimension), vocab coverage exactly once per
  rank (expected span based on 154880 / 4), and no overlapping or out-of-rank
  ownership spans.
- Produces a `ds4_glm52_layout_ownership_plan` with coverage_complete,
  overlaps_present, missing_spans_present flags plus Q-head/expert/vocab span
  bindings for the rank plan.

### a-load-rank-tensors -> `ds4_glm52_layout_mmap_rank_tensors`

- Code: `ds4_glm52_l0.c`, `ds4_glm52_layout_mmap_rank_tensors`.
- For each entry in the ownership plan: resolves the file path relative to
  checkpoint_root, verifies the file exists as a regular file, validates the
  byte range [offset, offset+length) is inside the file, checks for
  foreign-rank path references, recomputes sha256 over the exact declared byte
  slice, and rejects mismatches before any resident shard readiness is
  published.
- Produces a `ds4_glm52_layout_mapped_slices` with tensor_count, mapped_bytes,
  no_foreign_rank_shard, hash_verified=true, and one
  `ds4_glm52_layout_mapped_tensor` per accepted visible tensor after all mapped
  slices match their manifest sha256 values. Each mapped tensor record retains
  the validated dtype, shape dimensions, element count, and shape hash required
  by later collective/kernel identity checks.
- Creates real read-only `mmap()` handles for the exact validated file slices.
  Non-page-aligned byte offsets are page-aligned for the OS map while the
  published tensor record points at the declared slice. File descriptors are
  closed after mapping; mappings stay owned by the mapped-slices result until
  publication transfers them to resident state.
- On any late failure after a prior slice was mapped, partial mappings are
  unmapped before returning `INVALID`.

### a-publish-loaded-shard -> `ds4_glm52_layout_publish_loaded_rank_shard`

- Code: `ds4_glm52_l0.c`, `ds4_glm52_layout_publish_loaded_rank_shard`.
- Converts mapped tensor slices into the parent-visible `loaded_rank_shard`
  record by updating `state->resident_shards` (mapped, mapped_bytes,
  tensor_count, mapped tensor records, no_foreign_rank_shard, base/mtp shard
  counts) and `state->rank_plan` (q_head_start, q_head_end, expert_start,
  expert_end, vocab_start, vocab_end, rank, bound).
- Updates `state-model-worker` (same_state_as parent model-load graph) only
  after the complete rank-local layout is mapped.
- Transfers mmap ownership into `state->resident_shards`; callers clear it with
  `ds4_glm52_l0_unmap_resident_rank_shards`.

## Child Integration

The four code units are composed in `run_model_shard_layout()` (static helper
in `ds4_glm52_l0.c`), called from `serve/action-serve-open` when
`resident_shards_ready(state)` is false. The execution order matches the
contract's causal edges:

1. `a-read-layout-spec` (e-read-before-ownership)
2. `a-validate-tensor-ownership` (e-ownership-before-mmap)
3. `a-load-rank-tensors` (e-mmap-before-publish)
4. `a-publish-loaded-shard` (e-publish-out, e-publish-state)

On success, `resident_shards_ready(state)` becomes true and serve-open
projects `model-load/a-ready-rank-engines` as `OK`: resident rank-local shard
readiness is now a real parent-visible model-load output. The next runtime
gate is TP4/DCP4 fabric readiness, not model-load readiness.

## Production State Evidence

- Header types: `ds4_glm52_layout_entry`, `ds4_glm52_layout_spec`,
  `ds4_glm52_layout_ownership_plan`, `ds4_glm52_layout_mapped_tensor`,
  `ds4_glm52_layout_mapped_slices`, and resident mapped tensors in
  `ds4_glm52_l0_resident_rank_shards` in `ds4_glm52_l0.h`.
- Cleanup APIs: `ds4_glm52_layout_unmap_mapped_slices` and
  `ds4_glm52_l0_unmap_resident_rank_shards`.
- Config: `ds4_glm52_l0_config.layout_path` field; `ds4_engine_options.glm52_tp4_layout_path` in `ds4.h`.
- CLI: `--glm52-tp4-layout FILE` flag in `ds4_cli.c` and `ds4_server.c`.
- Help: `--glm52-tp4-layout` documented in `ds4_help.c`.
- Manifest format: `ds4-shard-layout/v1` key=value text format with `[entry]`
  section headers.
- Bird import helper: `misc/glm52_bird_layout.py` reads a per-rank
  `integrity-manifest.json`, emits a DS4 rank plan and DS4 layout manifest,
  verifies all 20 base shards plus one MTP shard by full-file SHA-256, and
  binds rank Q-head, `[0,256)` expert-id, and vocab spans for L0 loading.
- Fabric binding: rank plans now declare `fabric_data_plane=crs812-200g`.
  Model-load and TP-group validation distinguish CRS812 200G fabric addresses
  from management SSH addresses and reject `192.168.0.x` for TP/DCP
  collective traffic.

## Validation

### Code gates

- `make cpu`: PASS (0 errors, 0 warnings).
- `make tests/test_glm52_l0 && ./tests/test_glm52_l0`: PASS.
- Focused GLM 5.2 suite: PASS
  (`tests/test_glm52_l0`, `tests/test_tp4_rank_group`,
  `tests/test_glm52_mock`, `tests/test_glm52_tp4_mock`,
  `tests/test_glm52_tp4_mock_serve`, `tests/test_glm52_tp4_mock_process`,
  `tests/test_glm52_tp4_allreduce`, `tests/test_glm52_dcp_row_exchange`).
- `python3 -m py_compile misc/glm52_bird_layout.py`: PASS.
- Rank-0 GX10 real layout dry run: PASS through `serve/action-serve-open`;
  stopped at expected `serve/action-tp-group` real fabric boundary. Evidence:
  `dev-artifacts/validation/rank0-gx10-real-layout-dry-run.md`.
- Rank-1 and rank-2 GX10 real layout dry runs: PASS through
  `serve/action-serve-open` with `fabric_data_plane=crs812-200g`; stopped at
  expected `serve/action-tp-group` real fabric boundary. Rank 3 is pending SSH
  access restoration. Evidence:
  `dev-artifacts/validation/gx10-real-layout-dry-runs-crs812.md`.
- `git diff --check`: PASS.

### Test cases (12 required scenarios)

| Test | Scenario |
| --- | --- |
| `test_layout_valid_manifest_passes` | Valid TP4 rank-local manifest passes all 4 code units |
| `test_layout_wrong_tp_fails` | tp_size=2 rejected at read_manifest |
| `test_layout_foreign_rank_fails` | Foreign-owned rank-local entry rejected + foreign-rank file path rejected |
| `test_layout_missing_qhead_fails` | Q-head span [32,40) missing coverage for rank 2 |
| `test_layout_overlap_qhead_fails` | Two Q-head entries overlapping [32,40)+[36,48) rejected |
| `test_layout_replicated_visible_all_ranks` | Replicated tensor visible on all 4 ranks |
| `test_layout_sharded_only_on_owning_rank` | Sharded tensor visible on rank 2, invisible on rank 0 |
| `test_layout_missing_shard_file_fails` | Nonexistent file_path fails at mmap |
| `test_layout_sha256_mismatch_fails` | Existing file with wrong slice sha256 fails before publish |
| `test_layout_missing_metadata_fails` | Missing dtype/shape metadata fails at read_manifest |
| `test_layout_metadata_byte_count_mismatch_fails` | dtype/shape element bytes must match byte_length |
| `test_model_load_layout_reaches_ready_rank_engines` | Valid layout lets serve-open publish ready rank-local model engines |

Additional assertion coverage in `test_layout_valid_manifest_passes` now proves
the mapped result contains real readable mmap slice handles, including a
non-page-aligned declared byte offset, retains validated dtype/shape metadata,
and transfers both handles and metadata to resident state before resident
cleanup unmaps them.

GX10 target run: deployed current source to
`/tmp/ds4-gx10-runtime-metadata-20260808062450` on rank0 `192.168.0.40`,
rank1 `192.168.0.240`, rank2 `192.168.0.99`, and rank3 `192.168.0.39`.
`make tests/test_glm52_l0 && ./tests/test_glm52_l0` passed on all four Linux
GX10 hosts.

### GPU upload/binding slice

- Added `ds4_glm52_layout_upload_resident_gpu_tensors`, an explicit
  backend-injected upload boundary that reads the retained resident mmap
  tensor table, requires validated dtype/shape/element_count/shape_hash
  metadata, allocates one rank-owned device tensor per mapped tensor, uploads
  the exact mmap payload bytes, and records ready GPU tensor bindings in
  `state->resident_shards`.
- Cleanup is fail-closed: any allocation, device ownership, metadata, or copy
  failure releases partial GPU allocations and leaves `gpu_resident=false`.
- `test_layout_valid_manifest_passes` now verifies uploaded tensor count,
  device id, byte count, dtype, shape hash, payload bytes, and cleanup. Added
  `test_layout_gpu_upload_missing_metadata_fails` to prove missing runtime
  metadata blocks GPU residency before allocation.
- Added `ds4_glm52_cuda_gpu_tensor_runtime`, the production CUDA upload
  adapter for the same BCD boundary. It wraps `ds4_gpu_tensor_alloc_on` so the
  callback success convention is correct, uses `ds4_gpu_tensor_write` for exact
  host-to-device bytes, and uses `ds4_gpu_tensor_free_in_place` for cleanup.
- Added `tests/test_glm52_cuda_layout_upload`, a CUDA-only checkpoint-free
  smoke that uploads a synthetic resident mapped tensor through
  `ds4_glm52_layout_upload_resident_gpu_tensors`, reads it back with
  `ds4_gpu_tensor_read`, and verifies GPU tensor metadata plus byte-for-byte
  payload retention.
- CUDA adapter GX10 run for this slice: deployed current source to
  `/tmp/ds4-gx10-cuda-layout-upload-20260808080612` on rank0 `192.168.0.40`,
  rank1 `192.168.0.240`, rank2 `192.168.0.99`, and rank3 `192.168.0.39`.
  The build target `tests/test_glm52_cuda_layout_upload` and the executable
  `./tests/test_glm52_cuda_layout_upload` passed on all four Linux GB10 hosts
  with CUDA backend initialization on `NVIDIA GB10 (sm_121)`.
- Added the opt-in startup path `--glm52-tp4-gpu-resident`. CLI/server options
  request GPU residency, the GLM52 L0 config carries the request and injected
  CUDA runtime, and `run_model_shard_layout` now executes
  `a-upload-gpu-resident-tensors` after `a-publish-loaded-shard` when requested.
  The real engine branch initializes CUDA with `ds4_gpu_init_multi`, injects
  `ds4_glm52_cuda_gpu_tensor_runtime`, reports the GPU-resident tensor count
  and bytes, then still fails closed at later unresolved L0 serving seams.
- Expanded `tests/test_glm52_cuda_layout_upload` to drive `SERVE_OPEN` with a
  generated DS4-native rank plan/layout fixture, proving the model-load child
  graph can map layout files, publish rank-local shard spans, upload all mapped
  tensors to CUDA residency, read back payload bytes, and clean up.
- Startup GPU-residency GX10 run for this slice: deployed current source to
  `/tmp/ds4-gx10-serve-open-gpu-resident-20260808083245` on rank0
  `192.168.0.40`, rank1 `192.168.0.240`, rank2 `192.168.0.99`, and rank3
  `192.168.0.39`. `make ds4 tests/test_glm52_cuda_layout_upload &&
  ./tests/test_glm52_cuda_layout_upload` passed warning-clean on all four Linux
  GB10 hosts.
- GX10 target run for this slice: deployed current source to
  `/tmp/ds4-gx10-gpu-upload-binding-20260808073416` on rank0 `192.168.0.40`,
  rank1 `192.168.0.240`, rank2 `192.168.0.99`, and rank3 `192.168.0.39`.
  `make tests/test_glm52_l0 && ./tests/test_glm52_l0` passed on all four
  Linux GX10 hosts. The Linux compiler emitted the known test-fixture
  `snprintf` truncation warnings; the target exited 0 on every host.

### BCD mechanical validation

- Lint (complete): PASS, 18 graphs, 0 errors, 0 warnings
  (`dev-artifacts/validation/lint-after-runtime-tensor-metadata.result.json`).
- model-shard-layout leaf validation: PASS, 0 errors, 0 warnings
  (`dev-artifacts/validation/validate-model-shard-layout-after-runtime-tensor-metadata.result.json`).
- serve L0 validation: PASS, 0 errors, 0 warnings
  (`dev-artifacts/validation/validate-serve-after-crs812-fabric.result.json`).
- model-shard-layout simulation (map-rank-local-ds4-layout-success): PASS,
  simulation_sha256 `0d62c846706f3a5061058c22314313e5503cecbf3689813833267e0bd0e3024e`
  (`dev-artifacts/validation/simulate-model-shard-layout-after-runtime-tensor-metadata.result.json`).
- GPU upload/binding lint: PASS, 18 graphs, 0 errors, 0 warnings
  (`dev-artifacts/validation/lint-after-gpu-upload-binding.result.json`).
- GPU upload/binding model-shard-layout leaf validation: PASS, 0 errors,
  0 warnings
  (`dev-artifacts/validation/validate-model-shard-layout-after-gpu-upload-binding.result.json`).
- GPU upload/binding simulation: PASS for mapped-state publication and GPU
  upload success cases, simulation_sha256
  `d4039d04353210e65ca30cc9a32afdb4c9176a8883f309a2fb0ffa94f6d3c58d`
  (`dev-artifacts/validation/simulate-model-shard-layout-after-gpu-upload-binding.result.json`).
- Startup GPU-residency recheck reused the same current contract digests:
  lint PASS, model-shard-layout leaf validation PASS, and model-shard-layout
  simulation PASS for the GPU upload success case.

### Current digests

- blueprint_sha256: `25cfd9d6bad30a9b9163857f4df866da258fc6ffc09ff2445bfef0cba554669b`
- semantic_sha256: `e8c2bcf10d15c04139b4cc38b6b55b4a35ce7d48140a2dee72f57f8e31792952`
- model-load graph_contract_sha256: `9dbc3e9a385d4fe634eace00c2a1a9745dd571854d8ea38f81af4fcb5de54078`
- model-load type_schema_sha256: `9d632a15b05a694407b465f8d592366a941b035cc99ead4b69626a9e9f520431`
- model-shard-layout graph_contract_sha256: `2cb594a34ade6be64bbb10e147e6f012c3b4e3554d2399dae097288660eae35e`
- model-shard-layout type_schema_sha256: `38e336f9f776ff0f281abe0dac6e02a1989c03c01c6b4fc56c2f15af3595a2ef`
- serve graph_contract_sha256: `e86e9e8efe3ad48fdda966be2e1723b862adfd62176e7a45defabfa67b50720a`
- serve type_schema_sha256: `3f4be1b144e1cee44727789840835951678fede862e9ccde1a2c91d025010a46`

## Remaining Gaps

1. **Base/MTP shard count is simplified.** `publish_loaded_rank_shard` sets
   `base_shard_count` and `mtp_shard_count` to the expected constants (20/1)
   rather than counting from the actual mapped entries. This satisfies the
   `resident_shards_ready` check but should be derived from the ownership
   plan's actual entry roles.

2. **Real full-size rank layout validation remains.** The startup path can now
   opt into CUDA GPU residency and is validated with checkpoint-free fixtures;
   the next data-dependent gate is to run it against the actual GLM 5.2
   per-rank layout manifests and local shard files on the four GX10 machines.

3. **Production checkpoint header decode is still deferred.** The runtime trusts
   the DS4-native layout manifest for dtype/shape metadata after validating it
   against byte_length and dtype size. It does not yet parse a full safetensors
   or GGUF-style tensor header directly from the mapped file bytes.
