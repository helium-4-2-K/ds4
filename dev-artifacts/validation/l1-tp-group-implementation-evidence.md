# L1 TP-Group Implementation Evidence

Date: 2026-08-07

Scope: first realization of `serve/action-tp-group` behind the GLM 5.2 TP4
L0 skeleton. This implements topology binding only; the real four-rank
collective transport handshake remains unresolved.

## Contract Mapping

- `serve/action-tp-group`
  - Code: `ds4_glm52_l0.c`, `bind_tp_group`.
  - Reads resident rank-local model shard state through
    `resident_shards_ready`.
  - Updates rank-plan state with local rank id, Q-head span, and DCP rank.
  - Updates TP fabric state with TP4/DCP4/PP1, rank count, local rank, and
    fabric address.
  - Returns strict `not_ready` after topology binding because broadcast,
    all-reduce, and logits-gather/top-k transport are not implemented yet.

## Production State Evidence

- Header state: `ds4_glm52_l0_rank_plan` and `ds4_glm52_l0_tp_fabric`.
- `ds4_glm52_l0_tp_fabric` distinguishes:
  - `topology_bound`: local descriptor has been validated and recorded.
  - `transport_ready`: real four-rank collective transport handshake is ready.
  - `group_ready`: downstream prefill/decode may use the collective group.
- Current implementation sets `topology_bound=true` but leaves
  `transport_ready=false` and `group_ready=false`.

## Validation

- `make tests/test_glm52_l0 && ./tests/test_glm52_l0`: PASS.
- `make cpu`: PASS.
- `git diff --check`: PASS.
- BCD lint: current digest PASS.
- BCD serve L0 validation: current digest PASS.
- BCD `model-shard-layout` simulation: current digest PASS.

## Remaining Frontier

Implement real four-rank collective transport readiness for TP4/DCP4:
rank registration, compatibility handshake, broadcast path, all-reduce paths,
and logits gather/top-k path. The existing two-rank `ds4_tp` transport is a
reference seam but is not sufficient for the GLM 5.2 four-rank target.
