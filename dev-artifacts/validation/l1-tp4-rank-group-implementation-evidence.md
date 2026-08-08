# L1 TP4 Rank-Group Implementation Evidence

Date: 2026-08-07

Scope: no-model rank-group smoke for the GLM 5.2 TP4/DCP4 serving skeleton.
This started as an in-memory compatibility and command-state seam and now also
has forked local file-descriptor, loopback TCP, and real CRS812 four-GX10
transport smoke coverage. It does not initialize NCCL/RoCE tensor collectives
or execute GLM kernels.

## Contract Mapping

- `serve/action-tp-group`
  - Code: `ds4_glm52_l0.c`, `ds4_glm52_tp4_rank_group_*` and
    `ds4_glm52_tp4_transport_*`.
  - Establishes the state-machine contract for four ranks before real transport
    collectives are wired in.
  - Validates rank ids `0..3`, `tp_size=4`, `dcp_size=4`, model hash, config
    hash, and rank-plan hash.
  - Allows command broadcast only from rank 0.
  - Requires all four ranks to acknowledge a command before it is complete.
  - Frames hello, command, and ack records across an FD boundary with fixed
    magic/version/type checks.
  - Parses numeric TCP `host:port` endpoints, binds a coordinator listener,
    accepts worker connections with timeout handling, and connects workers to
    the coordinator.
  - Provides `tests/glm52_tp4_fabric_smoke`, a coordinator/worker executable
    that runs the same hello/register, readiness publication, command
    broadcast, ack, and shutdown sequence against real CRS812 endpoints.
  - Clears topology/transport/group readiness when a rank transport failure is
    recorded.
  - Publishes TP fabric readiness into `ds4_glm52_l0_state` only after the full
    rank set is registered and compatible.

## Production State Evidence

- Header state/API: `ds4_glm52_tp4_rank_group`,
  `ds4_glm52_tp4_rank_member`, `ds4_glm52_tp4_command`, and
  `ds4_glm52_tp4_transport_hello`.
- TCP endpoint API: `ds4_glm52_tp4_tcp_endpoint`,
  `ds4_glm52_tp4_tcp_parse_endpoint`, `ds4_glm52_tp4_tcp_listen`,
  `ds4_glm52_tp4_tcp_accept`, and `ds4_glm52_tp4_tcp_connect`.
- CRS812 smoke tool: `tests/glm52_tp4_fabric_smoke.c`.
- The group distinguishes topology/transport/group readiness from individual
  command acknowledgement.
- The frame helpers are deliberately below the deployment policy layer. TCP
  loopback proves real socket readiness and timeout behavior, but host byte
  order remains acceptable for the current smoke and is not yet a stable
  multi-version wire contract.
- `ds4_glm52_tp4_rank_group_publish_fabric` is the explicit bridge from the
  rank-group seam to the L0 `state-tp` representation.
- While validating the broader L0 gate, the model-shard-layout ownership rule
  was tightened to match the dedicated per-rank disk layout: rank-local entries
  owned by a non-current rank now fail at ownership validation, while replicated
  entries remain visible in each rank's own valid manifest.

## Validation

- `make tests/test_tp4_rank_group tests/test_glm52_l0 &&
  ./tests/test_tp4_rank_group && ./tests/test_glm52_l0`: PASS.
- `make tests/glm52_tp4_fabric_smoke`: PASS.
- Local four-process loopback smoke with `tests/glm52_tp4_fabric_smoke`: PASS.
- Management endpoint rejection with `192.168.0.40`: PASS.
- Real CRS812 four-GX10 smoke: PASS. Rank 0 listened on `10.100.185.3:49052`;
  ranks 1, 2, and 3 connected from their GX10 hosts; rank 0 registered all
  workers, published fabric readiness, sent shutdown, and collected
  `ack_mask=0xf`. Evidence:
  `dev-artifacts/validation/l1-tp-group-crs812-smoke-evidence.md`.
- `make cpu`: PASS.
- BCD lint at blueprint
  `a0d9cf6b17df5f544ab2d17930aaf58b6f2bf4a3c1ce4034d3f91f1ba202e30a`: PASS.
- BCD `serve` L0 validation at the same blueprint digest: PASS.
- `git diff --check`: PASS.

## Remaining Frontier

Wire this CRS812 endpoint protocol into the normal DS4 CLI/server process
roles, then add the collective payload backend used by prefill/decode. The
remaining real frontier is tensor data movement and math: all-reduce payloads,
logits gather/top-k payloads, DCP selected-row exchange payloads, and GPU/GLM
kernels.
