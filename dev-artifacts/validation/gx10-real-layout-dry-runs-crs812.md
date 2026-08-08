# GX10 Real Layout Dry Runs With CRS812 Fabric Guard

Date: 2026-08-07

Scope: real GLM 5.2 TP4 rank-local model-load/layout dry runs on reachable
GX10 ranks after adding explicit CRS812 200G data-plane validation.

## Fabric Rule

- Management/control-plane SSH addresses are `192.168.0.x`.
- TP/DCP collective traffic must use CRS812 200G data-plane addresses
  `10.100.185.x`.
- Generated DS4 rank plans include `fabric_data_plane=crs812-200g`.
- The L0 loader/TP-group path rejects `192.168.0.x` as a collective endpoint.

## Rank 0 Prior Evidence

- Host: `gx10-b38f`
- Management IP: `192.168.0.40`
- Fabric IP: `10.100.185.3`
- Model root: `/home/helium_gx/models/glm52-full-tp4-mtp-rank0`
- Result: PASS through `serve/action-serve-open`, then expected NOT_READY at
  `serve/action-tp-group`.
- Evidence: `dev-artifacts/validation/rank0-gx10-real-layout-dry-run.md`.

## Rank 1

- Host: `gx10-095d`
- Management IP: `192.168.0.240`
- Fabric IP/interface: `10.100.185.1/24` on `enP2p1s0f0np0`
- Model root: `/home/helium_gx/models/glm52-full-tp4-mtp-rank1`
- Generated rank plan: `/tmp/ds4-glm52-rank1.plan`
- Generated layout: `/tmp/ds4-glm52-rank1.layout`
- Plan evidence: `fabric_addr=10.100.185.1`,
  `fabric_data_plane=crs812-200g`
- Representative roles:
  - q_head: `model-rank-1-part-0.safetensors`
  - expert: `model-rank-1-part-1.safetensors`
  - vocab: `model-rank-1-part-19.safetensors`
- Result:

```text
serve/action-serve-open status=ok
serve/action-tp-group status=not_ready
not_ready at serve/action-tp-group:
TP4/DCP4 topology is bound; real four-rank collective transport handshake is not implemented
real 531.42
user 512.54
sys 18.78
```

Interpretation: PASS through real rank-1 model-load/layout verification with
the CRS812 fabric marker accepted; expected blocker is the unresolved real
four-rank collective transport handshake.

## Rank 2

- Host: `gx10-180f`
- Management IP: `192.168.0.99`
- Fabric IP/interfaces: `10.100.185.2/24` on `enp1s0f0np0` and
  `enP2p1s0f1np1`
- Model root: `/home/helium_gx/models/glm52-full-tp4-mtp-rank2`
- Generated rank plan: `/tmp/ds4-glm52-rank2.plan`
- Generated layout: `/tmp/ds4-glm52-rank2.layout`
- Plan evidence: `fabric_addr=10.100.185.2`,
  `fabric_data_plane=crs812-200g`
- Representative roles:
  - q_head: `model-rank-2-part-0.safetensors`
  - expert: `model-rank-2-part-1.safetensors`
  - vocab: `model-rank-2-part-19.safetensors`
- Result:

```text
serve/action-serve-open status=ok
serve/action-tp-group status=not_ready
not_ready at serve/action-tp-group:
TP4/DCP4 topology is bound; real four-rank collective transport handshake is not implemented
real 521.59
user 510.27
sys 11.09
```

Interpretation: PASS through real rank-2 model-load/layout verification with
the CRS812 fabric marker accepted; expected blocker is the unresolved real
four-rank collective transport handshake.

## Rank 3 Access Blocker

- Expected host: `gx10-9c9a`
- Management IP: `192.168.0.39`
- Expected fabric IP: `10.100.185.4`
- Model root: `/home/helium_gx/models/glm52-full-tp4-mtp-rank3`
- Blocker: SSH from this shell with the available NVIDIA Sync identity failed
  with `Permission denied (publickey,password)`.

Rank 3 still needs the same command sequence once SSH access is restored:

```bash
cd /home/helium_gx/src/ds4-gb10x4-bcd-realtest
./misc/glm52_bird_layout.py \
  --model-root /home/helium_gx/models/glm52-full-tp4-mtp-rank3 \
  --rank 3 \
  --fabric-addr 10.100.185.4 \
  --rank-plan-out /tmp/ds4-glm52-rank3.plan \
  --layout-out /tmp/ds4-glm52-rank3.layout
./ds4 \
  -m /home/helium_gx/models/glm52-full-tp4-mtp-rank3 \
  --glm52-tp4-l0 \
  --glm52-tp4-rank 3 \
  --glm52-tp4-rank-plan /tmp/ds4-glm52-rank3.plan \
  --glm52-tp4-layout /tmp/ds4-glm52-rank3.layout \
  --glm52-tp4-fabric 10.100.185.4 \
  --inspect
```

## Not Proven

These dry runs prove real rank-local model-load/layout verification and CRS812
fabric endpoint binding, not the real four-rank collective implementation. The
next implementation frontier remains the TP4/DCP4 transport handshake and
collective data plane over the CRS812 fabric.
