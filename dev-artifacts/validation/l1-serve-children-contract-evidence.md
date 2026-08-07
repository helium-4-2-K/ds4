# L1 Serve Child Graph Contract Evidence

Date: 2026-08-07

Scope: complete the remaining first-level `serve` child graphs so the root
graph is no longer a dense list of unexpanded actions. This is a BCD contract
change only; it does not claim the production code for these children is
implemented.

## Completed Graphs

- `request-session`: lowers `serve/action-serve-accept` into request
  validation, TP-aware session/KV binding, and prefill queue publication.
- `stream-token`: lowers `serve/action-stream-token` into client token write
  and continuation/checkpoint guard evaluation.
- `kv-checkpoint`: lowers `serve/action-kv-checkpoint` into checkpoint policy,
  rank-local KV shard write, and replicated marker publication.
- `tp-group`: lowers `serve/action-tp-group` into rank hello validation,
  rank-plan binding, collective fabric establishment, and TP-ready publication.

Parent placeholder `code_unit` semantics were removed from the expanded `serve`
actions. Code-unit obligations now live on the child leaf actions where the
implementation seams are explicit.

## State Terminal Modeling

The first mutation produced valid child graphs, but simulation exposed that the
terminal state nodes were not directly bound to outcome semantics. A follow-up
mutation added explicit state-bound `effect` or `no_effect` entries for:

- request/session KV binding and prefill queue publication;
- stream response-state recording;
- KV checkpoint state publication;
- TP group model-worker read, rank-plan binding, and collective fabric ready
  state.

This keeps the validator strict: state terminals are not treated as data
outputs, and simulation must show the exact state effect being claimed.

## Validation

- `dev-artifacts/validation/mutate-complete-serve-l1-children.result.json`:
  PASS.
- `dev-artifacts/validation/mutate-state-terminal-outcomes-for-serve-l1-children.result.json`:
  PASS.
- `dev-artifacts/validation/status-after-serve-l1-children.result.json`: PASS,
  blueprint hash
  `f5090c2f97f06122067f4b3c840bb2e6e581e52bccdbaf370c5a4bec47aa31ed`.
- `dev-artifacts/validation/lint-after-serve-l1-children.result.json`: PASS.
- `dev-artifacts/validation/validate-serve-after-l1-children.result.json`:
  PASS.
- `dev-artifacts/validation/validate-serve-l1-children-leaves.result.json`:
  PASS.
- `dev-artifacts/validation/validate-serve-l1-expanded-composite.result.json`:
  PASS.
- `dev-artifacts/validation/validate-serve-existing-l1-leaves.result.json`:
  PASS.
- `dev-artifacts/validation/simulate-serve-l1-children-full-trace.result.json`:
  PASS with 0 errors and 0 warnings.
- `dev-artifacts/ds4-arch-review.html`: PASS render with 18 graphs, 161 nodes,
  213 edges, and 28 types.

## Remaining Implementation Work

- Wire `request-session` to the server/session enqueue path.
- Wire `stream-token` to the client-visible streaming path after rank0 token
  selection.
- Wire `kv-checkpoint` to durable rank-local KV shard and cursor metadata
  persistence.
- Replace the remaining `tp-group` strict transport frontier with real
  cross-GX10 collective readiness.
