# L1 TP-Group Implementation Evidence

Date: 2026-08-07

Scope: first realization of `serve/action-tp-group` behind the GLM 5.2 TP4
L0 skeleton. The L0 serve action still stops before production prefill/decode,
but the standalone rank-group transport smoke now proves the four-rank CRS812
hello/register/readiness/command/ack handshake.

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
- `tp-group` transport smoke
  - Code: `tests/glm52_tp4_fabric_smoke.c` plus
    `ds4_glm52_tp4_rank_group_*`, `ds4_glm52_tp4_transport_*`, and
    `ds4_glm52_tp4_tcp_*`.
  - Rank 0 listens on a CRS812 fabric endpoint, receives/validates worker hello
    frames, publishes TP fabric readiness, broadcasts shutdown, and records all
    worker acks.
  - Workers connect to rank 0 over the fabric endpoint, send validated hello,
    receive shutdown, ack, and exit.

## Production State Evidence

- Header state: `ds4_glm52_l0_rank_plan` and `ds4_glm52_l0_tp_fabric`.
- `ds4_glm52_l0_tp_fabric` distinguishes:
  - `topology_bound`: local descriptor has been validated and recorded.
  - `transport_ready`: real four-rank collective transport handshake is ready.
  - `group_ready`: downstream prefill/decode may use the collective group.
- Current implementation sets `topology_bound=true` but leaves
  `transport_ready=false` and `group_ready=false`.
- Standalone smoke publication sets `topology_bound=true`,
  `transport_ready=true`, and `group_ready=true` after all four ranks register.

## Validation

- `make tests/test_glm52_l0 && ./tests/test_glm52_l0`: PASS.
- `make tests/glm52_tp4_fabric_smoke tests/test_tp4_rank_group &&
  ./tests/test_tp4_rank_group`: PASS.
- Local four-process `tests/glm52_tp4_fabric_smoke` loopback: PASS.
- Real CRS812 four-GX10 `tests/glm52_tp4_fabric_smoke`: PASS
  (`ack_mask=0xf`). Evidence:
  `dev-artifacts/validation/l1-tp-group-crs812-smoke-evidence.md`.
- `make cpu`: PASS.
- `git diff --check`: PASS.
- BCD lint: current digest PASS.
- BCD serve L0 validation: current digest PASS.
- BCD `model-shard-layout` simulation: current digest PASS.

## Remaining Frontier

Integrate the CRS812 smoke protocol into the normal DS4 CLI/server rank roles,
then implement real collective payload transport for TP4/DCP4 all-reduce,
logits gather/top-k, and DCP row exchange. The existing two-rank `ds4_tp`
transport is a reference seam but is not sufficient for the GLM 5.2 four-rank
target.
