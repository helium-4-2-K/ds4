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

### Current digests

- blueprint_sha256: `35a59fbba4347862df3f2f50f269a1f6a8e951d5aecb3210c6403a83352f0ee1`
- semantic_sha256: `3bfdecba885bd5f0bb442195112d5bc5106f663b2933edc2faae689d2098a954`
- model-load graph_contract_sha256: `083ff1b8ab754f0fe11170f32c449b6777b2944c9156de43a105f0836d89b2d5`
- model-load type_schema_sha256: `0aa0c0c7dfc43caabc5f745e1e1d6c0a82aae418e40a141221cb94ffe2b1696f`
- model-shard-layout graph_contract_sha256: `02eac4c71742fc979fd644e595afd791b49dc6c6a50e7faff20da038f61ff9ca`
- model-shard-layout type_schema_sha256: `aa33ec3ae3a40ac468691f0fb786c052d10c1ece1e28b8ab04a80fadc728b2c6`
- serve graph_contract_sha256: `e86e9e8efe3ad48fdda966be2e1723b862adfd62176e7a45defabfa67b50720a`
- serve type_schema_sha256: `3f4be1b144e1cee44727789840835951678fede862e9ccde1a2c91d025010a46`

## Remaining Gaps

1. **Base/MTP shard count is simplified.** `publish_loaded_rank_shard` sets
   `base_shard_count` and `mtp_shard_count` to the expected constants (20/1)
   rather than counting from the actual mapped entries. This satisfies the
   `resident_shards_ready` check but should be derived from the ownership
   plan's actual entry roles.

2. **GPU upload is still deferred.** The parent-visible ready state now retains
   real mmap slice handles plus DS4-native manifest-derived dtype/shape
   metadata for validated rank-local and replicated files. It still does not
   upload those tensors to GPU or bind them to real GLM 5.2 kernels.

3. **Production checkpoint header decode is still deferred.** The runtime trusts
   the DS4-native layout manifest for dtype/shape metadata after validating it
   against byte_length and dtype size. It does not yet parse a full safetensors
   or GGUF-style tensor header directly from the mapped file bytes.
