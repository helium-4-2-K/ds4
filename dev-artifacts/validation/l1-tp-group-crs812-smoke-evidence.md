# L1 TP Group CRS812 Smoke Evidence

Date: 2026-08-07

Scope: first executable real TP-group transport smoke over the CRS812 200G
fabric. This validates the `tp-group` child graph frontier without model
kernels: rank registration, hello frame validation, TP fabric readiness
publication, rank-0 command broadcast, all-rank acknowledgement, and clean
shutdown.

## Implementation Mapping

- Tool: `tests/glm52_tp4_fabric_smoke`
- Coordinator path:
  - listens on the rank-0 CRS812 fabric endpoint;
  - registers local rank 0;
  - accepts three worker TCP connections;
  - receives and validates TP4/DCP4 hello frames via
    `ds4_glm52_tp4_transport_recv_hello_and_register`;
  - publishes fabric readiness via
    `ds4_glm52_tp4_rank_group_publish_fabric`;
  - broadcasts `DS4_GLM52_TP4_COMMAND_SHUTDOWN`;
  - records rank 0 local ack and worker acks via
    `ds4_glm52_tp4_transport_recv_ack_and_record`.
- Worker path:
  - connects to rank 0 over the fabric endpoint;
  - sends a validated hello frame;
  - receives the shutdown command;
  - sends an ack frame and exits.
- Safety guard: the tool rejects `192.168.0.x` management endpoints and
  requires callers to use the CRS812 fabric endpoint.

## Local Validation

Built and ran a four-process loopback smoke:

```text
coordinator: listening 127.0.0.1:49052
coordinator: registered rank 3
coordinator: registered rank 1
coordinator: registered rank 2
coordinator: fabric ready tp=4 dcp=4 ranks=4
coordinator: ack rank 1
coordinator: ack rank 2
coordinator: ack rank 3
coordinator: shutdown complete seq=1 ack_mask=0xf
worker1: command shutdown seq=1
worker1: ack sent
worker2: command shutdown seq=1
worker2: ack sent
worker3: command shutdown seq=1
worker3: ack sent
```

Negative guard:

```text
coordinator: refusing management endpoint 192.168.0.40; use CRS812 fabric
```

## Real CRS812 Run

Rank mapping:

| Rank | Host | Management IP | Fabric IP |
| --- | --- | --- | --- |
| 0 | `gx10-b38f` | `192.168.0.40` | `10.100.185.3` |
| 1 | `gx10-095d` | `192.168.0.240` | `10.100.185.1` |
| 2 | `gx10-180f` | `192.168.0.99` | `10.100.185.2` |
| 3 | `gx10-9c9a` | `192.168.0.39` | `10.100.185.4` |

Coordinator command:

```bash
./tests/glm52_tp4_fabric_smoke \
  --role coordinator \
  --rank 0 \
  --listen 10.100.185.3:49052 \
  --timeout-ms 30000
```

Worker commands:

```bash
./tests/glm52_tp4_fabric_smoke --role worker --rank 1 --connect 10.100.185.3:49052 --timeout-ms 30000
./tests/glm52_tp4_fabric_smoke --role worker --rank 2 --connect 10.100.185.3:49052 --timeout-ms 30000
./tests/glm52_tp4_fabric_smoke --role worker --rank 3 --connect 10.100.185.3:49052 --timeout-ms 30000
```

Observed worker output:

```text
worker1: command shutdown seq=1
worker1: ack sent
worker2: command shutdown seq=1
worker2: ack sent
worker3: command shutdown seq=1
worker3: ack sent
```

Observed coordinator output:

```text
coordinator: listening 10.100.185.3:49052
coordinator: registered rank 1
coordinator: registered rank 3
coordinator: registered rank 2
coordinator: fabric ready tp=4 dcp=4 ranks=4
coordinator: ack rank 1
coordinator: ack rank 2
coordinator: ack rank 3
coordinator: shutdown complete seq=1 ack_mask=0xf
```

Interpretation:

- PASS: ranks 1, 2, and 3 connected to rank 0 over the CRS812 fabric endpoint.
- PASS: rank 0 validated all worker hello frames and formed a complete TP4/DCP4 group.
- PASS: the group published fabric readiness in the L0 state seam.
- PASS: rank 0 broadcast a command and collected all acknowledgements.
- PROVEN IN FOLLOW-UP: portable 96-byte big-endian TP4 collective metadata
  frames plus fixed-size CPU float payload exchange, shared L0 host-buffer
  all-reduce smoke, and malformed frame rejection over CRS812. Evidence:
  `dev-artifacts/validation/l1-tp4-allreduce-payload-smoke-evidence.md`.
- PROVEN LOCALLY: host-buffer logits gather/top-k and DCP selected-row exchange
  validation/reply semantics.
- NOT PROVEN: production tensor collectives, decentralized all-reduce topology,
  high-throughput streaming, fabric-backed DCP row payload exchange,
  fabric-backed logits gather/top-k, GPU buffers, or GLM kernels.

## Next Frontier

The next implementation step is to reuse this transport lifecycle for real
collective payloads:

1. portable production wire encoding for typed collective requests if TCP
   remains a backend is complete for collective metadata frames; remaining
   control-frame portability is optional until the smoke transport becomes a
   stable multi-release protocol;
2. TP4 attention/FFN all-reduce backed by actual GLM rank-local buffers and
   the production GPU/fabric backend;
3. logits gather/top-k transport from actual vocab-shard buffers;
4. DCP selected-row exchange payloads over the production backend.
