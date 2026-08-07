# Plane bug draft: BCD simulation evidence can go stale without durable request

Title: BCD workflow should persist simulation request inputs and rerun them after digest changes

Priority: High

Type: Bug

Description:
During the DS4 GLM 5.2 TP4 BCD work, the blueprint contract and evidence digests were updated successfully, but the simulation proof initially became stale because only prior simulation output was available. Reconstructing a simulate-contract request from saved output lost the authored `property_flows`, `field_flows`, and `bounded_cycle_entry_ids`, causing the stricter simulator to fail even though contract validation passed.

Impact:
Agents can incorrectly report "simulation not validated" after a contract/evidence refresh, or worse, try to reconstruct missing proof mappings from output. This weakens BCD's intended rule that data/property/field mappings must be authored explicitly and not derived from graph edges.

Expected workflow:
- Persist the exact `simulate_contract` request as a durable artifact next to the result.
- Treat simulation PASS as current only when its artifact digests match current `bdev`.
- After any covered graph or ancestor digest changes, rerun the durable request.
- If graph shape changes, update `property_flows`, `field_flows`, and `bounded_cycle_entry_ids` explicitly against current semantic IDs before rerunning.

Observed in this repo:
- Current durable request: `dev-artifacts/validation/simulate-decode-current-full-trace.request.json`
- Current passing result: `dev-artifacts/validation/simulate-decode-current-full-trace.result.json`
- Current blueprint digest: `c80c8377089a6b917a54b4fd4a4f70941c210de5e4e6df4c636db99ba3ae6b50`
- Current semantic digest: `f240d44c7ee98961759595283598e5a97be5ed17ba6919b03d9c638db6a7e796`
- Current simulation digest: `54e02231ca5ace0ec879e135572c81a15a48f0a3d9fd2d8124eb73949b2d7cc9`

Acceptance criteria:
- BCD validation checklist requires a durable simulation request and result pair.
- Simulation rerun step compares current graph/type/ancestor digests before accepting a prior PASS.
- Guidance warns not to reconstruct authoring proof from saved simulation output.
