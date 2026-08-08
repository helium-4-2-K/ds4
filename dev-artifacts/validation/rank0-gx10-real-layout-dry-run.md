# Rank 0 GX10 Real Layout Dry Run

Date: 2026-08-07

Host:

- Management IP: `192.168.0.40`
- Hostname: `gx10-b38f`
- GLM 5.2 rank: `0`
- Fabric IP: `10.100.185.3`
- Model root: `/home/helium_gx/models/glm52-full-tp4-mtp-rank0`

Generated inputs on the host:

- Rank plan: `/tmp/ds4-glm52-rank0.plan`
- DS4 layout: `/tmp/ds4-glm52-rank0.layout`
- Generator: `misc/glm52_bird_layout.py`
- Layout entries: 21 full-file safetensors entries
- Ownership spans: Q heads `[0,16)`, expert ids `[0,256)`, vocab `[0,38720)`
- Data plane: CRS812 200G fabric, declared as
  `fabric_data_plane=crs812-200g`; `192.168.0.40` is management only.

Observed Bird checkpoint fact:

- The routed expert tensors keep all 256 expert ids on each rank and shard
  expert matrix dimensions. This invalidated the earlier 64-expert-per-rank
  assumption and the validator was corrected before this dry run.

Command:

```bash
./ds4 \
  -m /home/helium_gx/models/glm52-full-tp4-mtp-rank0 \
  --glm52-tp4-l0 \
  --glm52-tp4-rank 0 \
  --glm52-tp4-rank-plan /tmp/ds4-glm52-rank0.plan \
  --glm52-tp4-layout /tmp/ds4-glm52-rank0.layout \
  --glm52-tp4-fabric 10.100.185.3 \
  --inspect
```

Result:

```text
serve/action-serve-open status=ok
serve/action-tp-group status=not_ready
...
not_ready at serve/action-tp-group:
TP4/DCP4 topology is bound; real four-rank collective transport handshake is not implemented
real 526.16
user 515.23
sys 10.86
```

Interpretation:

- PASS: DS4 opened the GLM 5.2 TP4 L0 model-load path on the real rank-0 GX10.
- PASS: the generated DS4 layout was accepted.
- PASS: full-file SHA-256 verification of the rank-0 Bird shard files
  completed before resident rank-local shard readiness was published.
- PASS: the first blocker moved from missing layout to the next known
  implementation boundary, real TP4/DCP4 fabric handshake.
- NOT PROVEN: retained mmap handles, GPU tensor handles, real kernels, real
  four-rank collective transport, prefill, decode, or serving output.
