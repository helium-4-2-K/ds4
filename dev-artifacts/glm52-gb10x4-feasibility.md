# GLM 5.2 TP4 on Four GX10 Machines: Feasibility Note

Date: 2026-08-06

## Verdict

The plan is architecturally feasible, but the current native DS4 implementation is not ready to serve it directly.

The strongest proof point is the local Bird/vLLM baseline: it already served the unpruned `QuantTrio/GLM-5.2-Int4-Int8Mix` checkpoint on four GX10 nodes with TP4, PP1, DCP4, one rank per machine, `B12X_MLA_SPARSE`, `fp8_ds_mla`, per-rank sharded checkpoints, and NCCL/RoCE. That proves the target topology can work on this cluster class.

Native DS4 should not clone Bird/vLLM. The intended DS4 product goal is a lighter and more efficient serving engine with a narrower GLM 5.2 TP4 path: fewer framework layers, less dynamic scheduling machinery, explicit rank ownership, fixed collectives, and deterministic validation against the Bird baseline. Bird/vLLM is the oracle for correctness and topology, not the implementation shape to copy.

Native DS4 still needs substantial work. The current code has many hard-coded two-rank tensor-parallel assumptions: `tp_world = 2`, rank 0/1 expert ownership, half-vocab logits views, and two-way attention head splitting.

## GLM 5.2 Architecture Facts

Primary architecture source: `QuantTrio/GLM-5.2-Int4-Int8Mix/config.json`.

- Model class: `GlmMoeDsaForCausalLM`; model type: `glm_moe_dsa`.
- Shape: hidden size 6144, vocab size 154880, 78 decoder layers plus one next-token/MTP layer in the serving artifact path.
- Attention: 64 query heads, q LoRA rank 2048, KV LoRA rank 512, qk no-RoPE dim 192, qk RoPE dim 64, qk head dim 256, value head dim 256.
- Long context: `max_position_embeddings = 1048576`.
- DSA/indexer: 32 indexer heads, index head dim 128, `index_topk = 2048`, `index_topk_freq = 4`, and an explicit full/shared indexer pattern.
- FFN/MoE: first 3 layers dense, later layers sparse MoE; 256 routed experts, 8 experts per token, 1 shared expert, expert intermediate size 2048, dense intermediate size 12288.
- MTP/speculation: `num_nextn_predict_layers = 1` in the QuantTrio config; the local Bird launch uses a matched DSpark speculator with 3 speculative tokens.

## Q/K Distribution Across Four Ranks

The TP split should be query-head based, not KV-head based.

- Q distribution: rank 0 owns query heads 0-15, rank 1 owns 16-31, rank 2 owns 32-47, rank 3 owns 48-63.
- K distribution: GLM 5.2's DS4-local representation is MLA-like. It has one shared KV latent path, not four independent KV-head groups. DS4 records `n_head_kv = 1`, `n_kv_lora = 512`, `n_key_mla = 256`, and `n_value_mla = 256`.
- Per rank, the owned query heads use rank-owned `k_b` and `v_b` slices to reconstruct the non-RoPE K and V components from the shared `kv_lora` latent. The RoPE K component is the 64-wide shared position component.
- Therefore, a naive "64 KV heads / 4 ranks" interpretation is wrong for this implementation. The achievable split is: shared or DCP-partitioned latent KV cache plus query-head-local reconstruction.

## Current DS4 Evidence

- `ds4.c` defines GLM 5.2 as hidden 6144, vocab 154880, 64 heads, one KV head, 256 routed experts, 8 active experts, 1 shared expert, q LoRA 2048, KV LoRA 512.
- The GLM layout validates `attn_k_b` as `[q_nope, kv_lora, n_head]`, which means every query head has its own reconstruction slice over the shared KV latent.
- The decode/prefill implementation already understands a two-way attention head split in places, but it is guarded by `g->tp_world == 2`.
- Session setup writes `tp_world = 2`, builds half-vocab logits views, and several expert-loading paths explicitly assume rank 0/rank 1 ownership.

## Required Work To Support The Plan In Native DS4

The implementation should keep DS4's serving surface deliberately narrow: one model family, one TP4/DCP4 topology first, one checkpoint shard format, explicit kernel/collective contracts, and minimal runtime indirection. Generality can be added after the TP4 GLM 5.2 path is correct and measured.

1. Generalize TP runtime from 2 ranks to N ranks, with TP4 as the first target.
2. Replace pairwise/partner collectives with group collectives over ranks 0-3: broadcast, all-reduce sum, gather, and distributed top-k.
3. Add a TP4 checkpoint loader contract: per-rank shard manifest, rank coverage validation, foreign-rank rejection, and MTP/speculator shard validation.
4. Implement MLA-aware TP QKV:
   - q_a replicated or reduce-scattered;
   - q_b head-sharded to 16 query heads per rank;
   - kv_a_mqa emits shared 512-wide latent plus 64-wide RoPE K;
   - k_b/v_b kernels operate only on rank-owned query heads.
5. Implement DSA/IndexShare correctly:
   - full/shared indexer schedule from the model config;
   - `index_topk = 2048`;
   - global top-k behavior across DCP shards;
   - stable reuse of selected indices across shared-index layers.
6. Add DCP4 KV-cache semantics:
   - context-prefix partitioning;
   - cross-rank sparse attention reads;
   - `fp8_ds_mla` or an equivalent DS4 cache dtype;
   - explicit treatment for layers skipped by the cache dtype, matching the Bird baseline if retained.
7. Generalize attention/output projection:
   - local attention per rank-owned query head range;
   - local output-projection partials;
   - 4-rank all-reduce sum into replicated hidden.
8. Generalize MoE:
   - 4-way expert ownership or expert parallel routing for 256 routed experts;
   - top-8 expert dispatch and aggregation;
   - shared-expert path;
   - no rank 0/1 50/50 assumptions.
9. Generalize vocab/logits:
   - four-way LM-head/vocab shards;
   - gather or distributed top-k;
   - rank0 sampling output.
10. Integrate MTP/speculation:
    - load matched speculator shards;
    - run draft/verify under TP4 and DCP4;
    - preserve DCP draft sharding semantics.
11. Add validation:
    - per-layer Q/K/KV-latent tensor checks against Bird/vLLM on short prompts;
    - selected sparse-index top-k comparison;
    - logits comparison before sampling;
    - 4-rank failure tests for missing shard, wrong rank shard, stale config, and collective timeout.

## Lightweight DS4 Serving Strategy

To be meaningfully lighter than vLLM/Bird, DS4 should avoid importing broad-serving abstractions unless they are necessary for GLM 5.2 correctness.

- Use fixed TP4 rank topology before supporting arbitrary TP sizes.
- Use a fixed local sharded checkpoint layout instead of a generic model-loader stack.
- Precompute and validate per-rank tensor ownership at load time.
- Use a small set of explicit collectives: request broadcast, hidden all-reduce, FFN all-reduce, logits gather/top-k, and DCP sparse-index exchange.
- Keep scheduling simple initially: one active decode stream, bounded prefill chunking, and explicit MTP verify loop.
- Specialize kernels for GLM 5.2's MLA/DSA dimensions: 64 query heads, 16 heads per rank, KV latent 512, RoPE K 64, top-k 2048.
- Measure the win against Bird on resident memory, startup time, prefill throughput, decode TPOT, draft acceptance, and per-token communication volume.

Detailed implementation readiness inventory: `dev-artifacts/glm52-ds4-native-readiness.md`.

## Current BCD Coverage

The current BCD hierarchy now includes the previously missing model-load, prefill, and decode child contracts:

- `serve`: root lifecycle and four-rank hosting architecture.
- `model-load`: launch-plan validation, shard-manifest validation, rank-local shard mapping, and rank-engine readiness.
- `prefill`: prompt tokenization, bounded chunk planning, TP layer execution, all-reduce, DCP-compatible KV commit, and decode handoff.
- `decode`: composite ordered token-j workflow over TP4/DCP4.
- `decode-qkv-mla`: GLM 5.2 query-head TP split, MLA latent/KV row append, and rank-owned K/V reconstruction.
- `decode-dcp-topk`: DCP4 sparse global top-k merge and selected-row ownership.
- `decode-sparse-attention`: rank-local sparse MLA attention over owned query heads.
- `decode-expert-route`: replicated router/top-8 expert selection.
- `decode-ffn-local`: rank-owned expert/shared FFN partial production.

MTP/speculator support is intentionally not part of the base token-j design. It should be added as a separate child graph after base decode and prefill match the Bird/vLLM oracle.

## Sources

- Hugging Face QuantTrio config: https://huggingface.co/QuantTrio/GLM-5.2-Int4-Int8Mix/blob/main/config.json
- Hugging Face Transformers GLM MoE DSA docs: https://huggingface.co/docs/transformers/en/model_doc/glm_moe_dsa
- Z.ai GLM 5.2 release: https://z.ai/blog/glm-5.2
- vLLM GLM 5.2 recipe: https://recipes.vllm.ai/zai-org/GLM-5.2
- SGLang GLM 5.2 docs: https://lmsysorg.mintlify.app/cookbook/autoregressive/GLM/GLM-5.2
- Local Bird runbook: `/Users/helium/src/locals/gx10-glm52-full-bird-bf16-runbook.md`
- Local Bird launch script: `/Users/helium/src/locals/gx10-bird-bf16-assets/launch-rank.sh`
- Local TP4 converter: `/Users/helium/src/locals/gx10-bird-bf16-assets/convert_glm52_tp4.py`
