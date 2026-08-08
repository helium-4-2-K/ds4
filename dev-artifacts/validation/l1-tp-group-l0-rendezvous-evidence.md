# L1 TP Group L0 Rendezvous Evidence

Date: 2026-08-08

Scope: wire the `tp-group` L0 action into the existing four-rank CRS812
fabric-ready handshake and prove that real GX10 rank-local model-load/layout
startup advances past `serve/action-tp-group`.

## Contract

- Updated `bdev/tp-group.graph.textproto` through the BCD writer.
- New supported semantics:
  - rank 0 owns a CRS812 rendezvous endpoint;
  - ranks 1..3 connect to rank 0 and exchange validated hello frames;
  - rank 0 broadcasts a fabric-ready command and collects all rank acks;
  - worker ranks retry rendezvous connection within a bounded startup timeout.
- Current blueprint digest after mutation:
  `fd90a32e376e3824cf35a1c1a6a8281067225eb8d09e0f826a8cfd6ae31b420a`.
- Current semantic digest:
  `6063f389b518c4521c0caf9a2998c42431b3e0005b6edd9e1c0b7955733104eb`.

## BCD Mechanical Gates

- `dev-artifacts/validation/lint-after-tp-group-rendezvous.result.json`:
  PASS.
- `dev-artifacts/validation/validate-tp-group-after-rendezvous.result.json`:
  PASS, leaf profile, `tp-group`, 0 errors.
- `dev-artifacts/validation/simulate-tp-group-after-rendezvous.result.json`:
  PASS, `tp-group-rendezvous-success`, 0 errors.

## Local Code Gates

- `make ds4 tests/test_glm52_l0 tests/test_tp4_rank_group`: PASS on Darwin.
- `./tests/test_glm52_l0`: PASS.
- `./tests/test_tp4_rank_group`: PASS.
- `tests/glm52_tp4_fabric_smoke` four-process loopback:
  coordinator reached `ack_mask=0xf`; workers 1..3 received shutdown sequence
  1 and acked.
- `git diff --check`: PASS.

## Four-GX10 Build/Test Gate

Deployed source to `/tmp/ds4-gx10-rendezvous-20260808092728`.

Rank map:

| Rank | Host | Management IP | Fabric IP |
| --- | --- | --- | --- |
| 0 | `gx10-b38f` | `192.168.0.40` | `10.100.185.3` |
| 1 | `gx10-095d` | `192.168.0.240` | `10.100.185.1` |
| 2 | `gx10-180f` | `192.168.0.99` | `10.100.185.2` |
| 3 | `gx10-9c9a` | `192.168.0.39` | `10.100.185.4` |

On all four hosts:

```text
make ds4 tests/test_glm52_l0 tests/test_tp4_rank_group
./tests/test_glm52_l0
./tests/test_tp4_rank_group
```

Result: PASS on ranks 0, 1, 2, and 3. Linux emitted pre-existing
`-Wformat-truncation` warnings in the test fixture path construction; they did
not fail the build.

## Real L0 CRS812 Rendezvous Run

Rendezvous endpoint: `10.100.185.3:49161`.

Each rank ran `./ds4 --inspect` with its real rank-local model root, rank plan,
layout, local fabric address, and shared rendezvous endpoint.

Observed result on every rank:

```text
serve/action-serve-open status=ok
serve/action-tp-group status=ok
serve/action-serve-accept status=not_ready
not_ready at serve/action-serve-accept: request/session binding is not implemented
```

Interpretation:

- PASS: real rank-local GLM 5.2 layout/model-load completed on all ranks.
- PASS: ranks 0..3 formed the TP4/DCP4 fabric-ready group over CRS812.
- PASS: the L0 blocker moved from `serve/action-tp-group` to the next
  expected seam, `serve/action-serve-accept`.
- Expected nonzero process exit: `ds4 --inspect` still returns failure when
  the first unresolved L0 seam is reached.

## Remaining Limitation

This proves startup rendezvous and readiness publication, not a persistent
production collective session for serving tokens. The current handshake closes
cleanly after command/ack shutdown, matching the implemented L0 readiness
frontier. A real serving path still needs request/session binding and a
persistent fabric-backed execution path for prefill/decode collectives.
