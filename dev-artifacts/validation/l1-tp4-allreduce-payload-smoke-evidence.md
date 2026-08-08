# L1 TP4 All-Reduce Payload Smoke Evidence

Date: 2026-08-07

Scope: first executable tensor-payload smoke over local loopback and the real
CRS812 four-GX10 fabric. This extends the TP-group hello/command/ack smoke with
a deterministic fixed-size float all-reduce payload. It is still a CPU test
tool, not the final GPU collective backend.

Typed-frame update: the payload smoke now carries a typed TP4 collective
metadata frame before every rank-local partial payload and before every reduced
result payload on local loopback and CRS812. The frame is validated with the
production L0 `ds4_glm52_tp4_collective_frame_validate` helper before payload
bytes are accepted.

## Implementation Mapping

- Tool: `tests/glm52_tp4_fabric_smoke`
- New option: `--payload-floats N`
- Negative-test option: `--bad-payload-frame`
- Command used for payload mode: `DS4_GLM52_TP4_COMMAND_DECODE`
- Typed frame fields: frame version, collective kind, rank topology, dtype,
  element count, byte count, shape hash, model/session identity, sequence,
  token position, participant mask, and readiness bits.
- Worker behavior:
  - connects to rank 0 over CRS812 fabric;
  - sends TP4/DCP4 hello;
  - receives decode command;
  - sends a typed TP4 collective frame;
  - sends deterministic `float[N]` rank-local partial payload;
  - receives and validates the reduced-result typed TP4 collective frame;
  - receives reduced `float[N]`;
  - verifies the reduced vector equals the rank-ordered sum across ranks 0..3;
  - sends ack.
- Coordinator behavior:
  - listens on rank-0 CRS812 fabric address;
  - validates all worker hello frames;
  - publishes fabric readiness;
  - broadcasts decode command;
  - validates each worker's typed TP4 collective frame before payload bytes
    enter the reduction;
  - reads worker payloads;
  - sums rank 0 local partial plus ranks 1..3 payloads;
  - sends a typed TP4 collective frame before every reduced payload;
  - sends reduced vector to every worker;
  - records all acks.

## Local Validation

Command shape:

```bash
./tests/glm52_tp4_fabric_smoke --role coordinator --rank 0 --listen 127.0.0.1:49110 --payload-floats 16
./tests/glm52_tp4_fabric_smoke --role worker --rank 1 --connect 127.0.0.1:49110 --payload-floats 16
./tests/glm52_tp4_fabric_smoke --role worker --rank 2 --connect 127.0.0.1:49110 --payload-floats 16
./tests/glm52_tp4_fabric_smoke --role worker --rank 3 --connect 127.0.0.1:49110 --payload-floats 16
./tests/glm52_tp4_fabric_smoke --role coordinator --rank 0 --listen 127.0.0.1:49128 --payload-floats 1024
./tests/glm52_tp4_fabric_smoke --role worker --rank 1 --connect 127.0.0.1:49128 --payload-floats 1024
./tests/glm52_tp4_fabric_smoke --role worker --rank 2 --connect 127.0.0.1:49128 --payload-floats 1024
./tests/glm52_tp4_fabric_smoke --role worker --rank 3 --connect 127.0.0.1:49128 --payload-floats 1024
```

Observed result:

```text
coordinator: allreduce payload floats=1024 first=1000.0 last=5092.0
coordinator: command decode complete seq=1 ack_mask=0xf
worker1: allreduce payload verified floats=1024
worker2: allreduce payload verified floats=1024
worker3: allreduce payload verified floats=1024
```

Negative command shape:

```bash
./tests/glm52_tp4_fabric_smoke --role worker --rank 1 --connect 127.0.0.1:49129 --payload-floats 1024 --bad-payload-frame
```

Observed negative result:

```text
statuses coord=13 w1=141 w2=141 w3=141
coordinator: recv payload rank 1 failed: real TP4 collective requires nonzero identity, layer, shape hash, element count, and matching dtype byte count
```

## Real CRS812 Run

Source staged on each GX10 from commit `67a47e2` into
`~/ds4-smoke-67a47e2`; all four ranks rebuilt
`tests/glm52_tp4_fabric_smoke` locally before execution.

Rank mapping:

| Rank | Host | Fabric IP |
| --- | --- | --- |
| 0 | `gx10-b38f` | `10.100.185.3` |
| 1 | `gx10-095d` | `10.100.185.1` |
| 2 | `gx10-180f` | `10.100.185.2` |
| 3 | `gx10-9c9a` | `10.100.185.4` |

Coordinator command:

```bash
./tests/glm52_tp4_fabric_smoke \
  --role coordinator \
  --rank 0 \
  --listen 10.100.185.3:49055 \
  --timeout-ms 30000 \
  --payload-floats 1024
```

Worker commands:

```bash
./tests/glm52_tp4_fabric_smoke --role worker --rank 1 --connect 10.100.185.3:49055 --timeout-ms 30000 --payload-floats 1024
./tests/glm52_tp4_fabric_smoke --role worker --rank 2 --connect 10.100.185.3:49055 --timeout-ms 30000 --payload-floats 1024
./tests/glm52_tp4_fabric_smoke --role worker --rank 3 --connect 10.100.185.3:49055 --timeout-ms 30000 --payload-floats 1024
```

Observed worker output:

```text
worker1: command decode seq=1
worker1: allreduce payload verified floats=1024
worker1: ack sent
worker2: command decode seq=1
worker2: allreduce payload verified floats=1024
worker2: ack sent
worker3: command decode seq=1
worker3: allreduce payload verified floats=1024
worker3: ack sent
```

Observed coordinator output:

```text
coordinator: listening 10.100.185.3:49055
coordinator: registered rank 1
coordinator: registered rank 2
coordinator: registered rank 3
coordinator: fabric ready tp=4 dcp=4 ranks=4
coordinator: allreduce payload floats=1024 first=1000.0 last=5092.0
coordinator: ack rank 1
coordinator: ack rank 2
coordinator: ack rank 3
coordinator: command decode complete seq=1 ack_mask=0xf
```

Observed CRS812 negative run:

```text
coordinator: listening 10.100.185.3:49056
coordinator: registered rank 1
coordinator: registered rank 2
coordinator: registered rank 3
coordinator: fabric ready tp=4 dcp=4 ranks=4
coordinator: recv payload rank 1 failed: real TP4 collective requires nonzero identity, layer, shape hash, element count, and matching dtype byte count
```

Worker negative statuses:

```text
rank1_status=7
rank2_status=8
rank3_status=8
```

Interpretation:

- PASS: fixed-size tensor bytes moved from worker ranks to rank 0 over local
  loopback and CRS812.
- PASS: each local and CRS812 payload is preceded by a typed TP4 collective
  metadata frame.
- PASS: malformed local and CRS812 typed metadata is rejected before payload
  reduction.
- PASS: rank 0 performed deterministic all-rank float summation.
- PASS: reduced tensor bytes moved from rank 0 back to workers locally and over
  CRS812.
- PASS: all workers verified the reduced payload before acking in the positive
  path.
- NOT PROVEN: portable endian-stable production wire encoding, decentralized
  all-reduce topology, high-throughput streaming, GPU buffers, BF16/FP8 tensor
  payload execution, NCCL/RDMA semantics, or GLM layer integration.

## Next Frontier

Replace the coordinator-summed smoke with the target production collective
shape:

1. promote the smoke's native fixed-width frame into portable production wire
   encoding if TCP remains a backend;
2. attention/FFN all-reduce payloads backed by the real rank-local output
   buffers;
3. logits gather/top-k payloads;
4. DCP selected-row exchange payloads.
