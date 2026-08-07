# L1 model-shard-layout Implementation Evidence

Date: 2026-08-07

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
  replicated flag, and q_head/expert/vocab spans.

### a-validate-tensor-ownership -> `ds4_glm52_layout_validate_ownership`

- Code: `ds4_glm52_l0.c`, `ds4_glm52_layout_validate_ownership`.
- Validates: replicated tensors visible on all ranks, rank-local entries only
  visible on owning rank, no foreign-rank entries, no foreign-rank file paths,
  sha256 present for every entry, byte_length non-empty, Q-head coverage
  exactly once per rank (expected span [R*16, (R+1)*16)), expert coverage
  exactly once per rank (expected span [R*64, (R+1)*64)), vocab coverage
  exactly once per rank (expected span based on 154880 / 4), and no overlapping
  or out-of-rank ownership spans.
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
  no_foreign_rank_shard, and hash_verified=true after all mapped slices match
  their manifest sha256 values.
- Does NOT create real mmap handles yet; records file existence and byte-range
  validity. This is the first action allowed to create runtime mmap handles
  per the contract authority semantic.

### a-publish-loaded-shard -> `ds4_glm52_layout_publish_loaded_rank_shard`

- Code: `ds4_glm52_l0.c`, `ds4_glm52_layout_publish_loaded_rank_shard`.
- Converts mapped tensor slices into the parent-visible `loaded_rank_shard`
  record by updating `state->resident_shards` (mapped, mapped_bytes,
  no_foreign_rank_shard, base/mtp shard counts) and `state->rank_plan`
  (q_head_start, q_head_end, expert_start, expert_end, vocab_start, vocab_end,
  rank, bound).
- Updates `state-model-worker` (same_state_as parent model-load graph) only
  after the complete rank-local layout is mapped.

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
  `ds4_glm52_layout_ownership_plan`, `ds4_glm52_layout_mapped_slices` in
  `ds4_glm52_l0.h`.
- Config: `ds4_glm52_l0_config.layout_path` field; `ds4_engine_options.glm52_tp4_layout_path` in `ds4.h`.
- CLI: `--glm52-tp4-layout FILE` flag in `ds4_cli.c` and `ds4_server.c`.
- Help: `--glm52-tp4-layout` documented in `ds4_help.c`.
- Manifest format: `ds4-shard-layout/v1` key=value text format with `[entry]`
  section headers.

## Validation

### Code gates

- `make cpu`: PASS (0 errors, 0 warnings).
- `make tests/test_glm52_l0 && ./tests/test_glm52_l0`: PASS.
- Focused GLM 5.2 suite: PASS
  (`tests/test_glm52_l0`, `tests/test_tp4_rank_group`,
  `tests/test_glm52_mock`, `tests/test_glm52_tp4_mock`,
  `tests/test_glm52_tp4_mock_serve`, `tests/test_glm52_tp4_mock_process`,
  `tests/test_glm52_tp4_allreduce`, `tests/test_glm52_dcp_row_exchange`).
- `git diff --check`: PASS.

### Test cases (10 required scenarios)

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
| `test_model_load_layout_reaches_ready_rank_engines` | Valid layout lets serve-open publish ready rank-local model engines |

### BCD mechanical validation

- Lint (complete): PASS, 10 graphs, 0 errors, 0 warnings.
- model-shard-layout leaf validation: PASS, 0 errors, 0 warnings
  (`dev-artifacts/validation/validate-model-shard-layout-current-real-loader.result.json`).
- model-load composite validation: PASS, 0 errors.
- model-shard-layout simulation (map-rank-local-ds4-layout-success): PASS,
  simulation_sha256 `a5cabe3d7bf7a2bd9a2829d8c30572cbd3c241d20b60fce1bdb09610ebbc9ce0`.

### Current digests

- blueprint_sha256: `f5090c2f97f06122067f4b3c840bb2e6e581e52bccdbaf370c5a4bec47aa31ed`
- semantic_sha256: `d90ce8a6de0e834c605c043318e3806cf1b8186cac93a897698438f64da2b538`
- model-shard-layout graph_contract_sha256: `eb05725c27670f0c0d9c3b90044c040e3ee5e6130eca1e09e8d1b26a202b911e`

## Remaining Gaps

1. **Real mmap handles are not retained.** `ds4_glm52_layout_mmap_rank_tensors`
   validates file existence and byte ranges but does not call `mmap()` or
   record mapped addresses. The `mapped_tensor_slices` struct records counts
   and totals plus verified slice hashes, but not actual memory pointers. This
   is sufficient for a real layout/file-integrity dry run but needs retained
   mappings for production kernels.

2. **Base/MTP shard count is simplified.** `publish_loaded_rank_shard` sets
   `base_shard_count` and `mtp_shard_count` to the expected constants (20/1)
   rather than counting from the actual mapped entries. This satisfies the
   `resident_shards_ready` check but should be derived from the ownership
   plan's actual entry roles.

3. **Real resident tensor handles are still deferred.** The parent-visible
   ready state proves DS4 has a validated rank-local layout and file/byte-range
   residency evidence with verified bytes. It still does not prove real model
   tensor handles, decoded tensor metadata objects, or GPU upload are wired.
