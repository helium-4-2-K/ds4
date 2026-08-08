#include "ds4.h"
#include "ds4_glm52_mock.h"

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void check(bool cond, const char *msg) {
    if (!cond) {
        fprintf(stderr, "test_glm52_tp4_allreduce: %s\n", msg);
        exit(1);
    }
}

static ds4_glm52_l0_config mock_cfg(int rank) {
    ds4_glm52_l0_config cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.enabled = true;
    cfg.mock_model = true;
    cfg.mock_matmul = true;
    cfg.rank_set = true;
    cfg.rank = rank;
    cfg.tp_size = DS4_GLM52_L0_TP_SIZE;
    cfg.dcp_size = DS4_GLM52_L0_DCP_SIZE;
    cfg.pp_size = DS4_GLM52_L0_PP_SIZE;
    cfg.model_root = "/mock/glm-5.2";
    cfg.rank_plan = "/mock/rank-plan.textproto";
    cfg.fabric_addr = "127.0.0.1:41000";
    return cfg;
}

static void init_models_and_sessions(
        ds4_glm52_mock_model models[DS4_GLM52_L0_RANK_COUNT],
        ds4_glm52_mock_session sessions[DS4_GLM52_L0_RANK_COUNT]) {
    int prompt[] = {100, 101, 102, 103};
    char err[192] = "";
    for (int rank = 0; rank < DS4_GLM52_L0_RANK_COUNT; rank++) {
        ds4_glm52_l0_config cfg = mock_cfg(rank);
        check(ds4_glm52_mock_model_init(&cfg, &models[rank], err, sizeof(err)),
              "rank mock model should initialize");
        check(ds4_glm52_mock_prefill(&models[rank],
                                     prompt,
                                     sizeof(prompt) / sizeof(prompt[0]),
                                     &sessions[rank],
                                     NULL,
                                     0,
                                     err,
                                     sizeof(err)),
              "rank mock prefill should succeed");
    }
}

static void fill_nonassoc_partials(float *partials[DS4_GLM52_L0_RANK_COUNT],
                                   size_t n) {
    for (size_t i = 0; i < n; i++) {
        partials[0][i] = 1.0e20f;
        partials[1][i] = 1.0f;
        partials[2][i] = -1.0e20f;
        partials[3][i] = 3.0f;
    }
}

static uint32_t full_rank_mask(void) {
    return (UINT32_C(1) << DS4_GLM52_L0_RANK_COUNT) - UINT32_C(1);
}

static void make_l0_requests(
        ds4_glm52_tp4_collective_request requests[DS4_GLM52_L0_RANK_COUNT],
        ds4_glm52_tp4_collective_kind kind,
        uint64_t seq,
        int layer,
        size_t element_count) {
    for (int rank = 0; rank < DS4_GLM52_L0_RANK_COUNT; rank++) {
        ds4_glm52_tp4_collective_request *req = &requests[rank];
        memset(req, 0, sizeof(*req));
        req->frame_version = DS4_GLM52_TP4_COLLECTIVE_FRAME_VERSION;
        req->kind = kind;
        req->rank = rank;
        req->tp_size = DS4_GLM52_L0_TP_SIZE;
        req->dcp_size = DS4_GLM52_L0_DCP_SIZE;
        req->rank_count = DS4_GLM52_L0_RANK_COUNT;
        req->layer_index = layer;
        req->dtype = DS4_GLM52_TP4_TENSOR_DTYPE_F32;
        req->seq = seq;
        req->model_hash = 0x52000000u;
        req->session_hash = 0x52000001u;
        req->token_step_j = 17;
        req->element_count = element_count;
        req->shape_hash = 0x52006144u;
        req->byte_count =
            element_count * ds4_glm52_tp4_tensor_dtype_size(req->dtype);
        req->participant_mask = full_rank_mask();
        req->topology_ready = true;
        req->transport_ready = true;
        req->rank_local_partial_ready = true;
        req->replicated_output_ready = true;
    }
}

static void free_float_buffers(float *buffers[DS4_GLM52_L0_RANK_COUNT]) {
    for (int rank = 0; rank < DS4_GLM52_L0_RANK_COUNT; rank++) {
        free(buffers[rank]);
        buffers[rank] = NULL;
    }
}

static void test_l0_host_allreduce_buffers(void) {
    const size_t n = DS4_GLM52_MOCK_N_EMBD;
    ds4_glm52_tp4_collective_request requests[DS4_GLM52_L0_RANK_COUNT];
    float *partials[DS4_GLM52_L0_RANK_COUNT] = {0};
    float *outputs[DS4_GLM52_L0_RANK_COUNT] = {0};
    const float *partial_inputs[DS4_GLM52_L0_RANK_COUNT] = {0};
    char err[192] = "";

    for (int rank = 0; rank < DS4_GLM52_L0_RANK_COUNT; rank++) {
        partials[rank] = malloc(n * sizeof(partials[rank][0]));
        outputs[rank] = malloc(n * sizeof(outputs[rank][0]));
        check(partials[rank] != NULL && outputs[rank] != NULL,
              "L0 host all-reduce buffers should allocate");
        partial_inputs[rank] = partials[rank];
    }
    fill_nonassoc_partials(partials, n);

    make_l0_requests(requests, DS4_GLM52_TP4_COLLECTIVE_ATTN, 21, 11, n);
    check(ds4_glm52_tp4_collective_allreduce_f32_host(
              requests,
              partial_inputs,
              outputs,
              n,
              err,
              sizeof(err)),
          "L0 host attention all-reduce should execute");
    check(outputs[0][0] == 3.0f,
          "L0 host attention all-reduce should sum in rank order");
    for (int rank = 1; rank < DS4_GLM52_L0_RANK_COUNT; rank++) {
        check(memcmp(outputs[0], outputs[rank], n * sizeof(outputs[0][0])) == 0,
              "L0 host all-reduce output should be replicated to every rank");
    }

    make_l0_requests(requests, DS4_GLM52_TP4_COLLECTIVE_FFN, 22, 12, n);
    check(ds4_glm52_tp4_collective_allreduce_f32_host(
              requests,
              partial_inputs,
              outputs,
              n,
              err,
              sizeof(err)),
          "L0 host FFN all-reduce should execute");

    for (int rank = 0; rank < DS4_GLM52_L0_RANK_COUNT; rank++) {
        for (size_t i = 0; i < n; i++) outputs[rank][i] = 42.0f;
    }
    make_l0_requests(requests, DS4_GLM52_TP4_COLLECTIVE_LOGITS, 23, 13, n);
    check(!ds4_glm52_tp4_collective_allreduce_f32_host(
              requests,
              partial_inputs,
              outputs,
              n,
              err,
              sizeof(err)),
          "L0 host all-reduce should reject logits collectives");
    check(strstr(err, "attention or FFN") != NULL,
          "L0 host all-reduce logits rejection should name supported kinds");
    check(outputs[0][0] == 42.0f,
          "L0 host all-reduce should not publish output after kind rejection");

    make_l0_requests(requests, DS4_GLM52_TP4_COLLECTIVE_ATTN, 24, 14, n);
    requests[3].session_hash++;
    check(!ds4_glm52_tp4_collective_allreduce_f32_host(
              requests,
              partial_inputs,
              outputs,
              n,
              err,
              sizeof(err)),
          "L0 host all-reduce should reject rank identity mismatch");
    check(strstr(err, "identity") != NULL,
          "L0 host all-reduce identity rejection should name identity");

    free_float_buffers(partials);
    free_float_buffers(outputs);
}

static void fill_logits_shards(
        ds4_glm52_tp4_logits_rank_candidates shards[DS4_GLM52_L0_RANK_COUNT],
        int token_ids[DS4_GLM52_L0_RANK_COUNT][3],
        float scores[DS4_GLM52_L0_RANK_COUNT][3]) {
    const int shard_width = DS4_GLM52_L0_VOCAB_SIZE / DS4_GLM52_L0_RANK_COUNT;
    for (int rank = 0; rank < DS4_GLM52_L0_RANK_COUNT; rank++) {
        const int start = rank * shard_width;
        const int end = (rank == DS4_GLM52_L0_RANK_COUNT - 1)
            ? DS4_GLM52_L0_VOCAB_SIZE
            : start + shard_width;
        shards[rank].rank = rank;
        shards[rank].vocab_start = start;
        shards[rank].vocab_end = end;
        shards[rank].token_ids = token_ids[rank];
        shards[rank].scores = scores[rank];
        shards[rank].candidate_count = 3;
        shards[rank].present = true;
        for (int i = 0; i < 3; i++) {
            token_ids[rank][i] = start + 10 + i;
            scores[rank][i] = (float)(rank * 3 + i);
        }
    }
    scores[0][0] = 10.0f;
    scores[0][1] = 8.0f;
    scores[0][2] = 1.0f;
    scores[1][0] = 11.0f;
    scores[1][1] = 9.0f;
    scores[1][2] = 7.0f;
    scores[2][0] = 2.0f;
    scores[2][1] = 11.0f;
    scores[2][2] = 6.0f;
    scores[3][0] = 12.0f;
    scores[3][1] = 5.0f;
    scores[3][2] = 4.0f;
}

static void test_l0_logits_gather_topk(void) {
    ds4_glm52_tp4_collective_request requests[DS4_GLM52_L0_RANK_COUNT];
    ds4_glm52_tp4_logits_rank_candidates shards[DS4_GLM52_L0_RANK_COUNT];
    ds4_glm52_tp4_logits_topk_entry top[5];
    int token_ids[DS4_GLM52_L0_RANK_COUNT][3];
    float scores[DS4_GLM52_L0_RANK_COUNT][3];
    char err[192] = "";

    fill_logits_shards(shards, token_ids, scores);
    make_l0_requests(requests, DS4_GLM52_TP4_COLLECTIVE_LOGITS, 31, 15, 3);
    check(ds4_glm52_tp4_logits_gather_topk_f32_host(
              requests,
              shards,
              5,
              top,
              sizeof(top) / sizeof(top[0]),
              err,
              sizeof(err)),
          "L0 logits gather/top-k should merge rank-local candidates");
    check(top[0].owner_rank == 3 && top[0].score == 12.0f,
          "L0 logits top-k should choose highest score first");
    check(top[1].owner_rank == 1 && top[1].score == 11.0f,
          "L0 logits top-k ties should prefer lower token id");
    check(top[2].owner_rank == 2 && top[2].score == 11.0f,
          "L0 logits top-k should preserve second tied candidate");
    check(top[3].owner_rank == 0 && top[3].score == 10.0f,
          "L0 logits top-k should include rank 0 candidate by score");
    check(top[4].owner_rank == 1 && top[4].score == 9.0f,
          "L0 logits top-k should fill requested k");

    ds4_glm52_tp4_logits_rank_candidates bad_shards[DS4_GLM52_L0_RANK_COUNT];
    memcpy(bad_shards, shards, sizeof(bad_shards));
    bad_shards[2].vocab_start++;
    check(!ds4_glm52_tp4_logits_gather_topk_f32_host(
              requests,
              bad_shards,
              5,
              top,
              sizeof(top) / sizeof(top[0]),
              err,
              sizeof(err)),
          "L0 logits gather/top-k should reject vocab coverage gaps");
    check(strstr(err, "vocab shards") != NULL,
          "L0 logits coverage rejection should name vocab shards");

    memcpy(bad_shards, shards, sizeof(bad_shards));
    token_ids[0][0] = shards[1].vocab_start;
    check(!ds4_glm52_tp4_logits_gather_topk_f32_host(
              requests,
              bad_shards,
              5,
              top,
              sizeof(top) / sizeof(top[0]),
              err,
              sizeof(err)),
          "L0 logits gather/top-k should reject token outside owner shard");
    check(strstr(err, "owner shard") != NULL,
          "L0 logits ownership rejection should name owner shard");
}

static void make_partials(
        const ds4_glm52_mock_model models[DS4_GLM52_L0_RANK_COUNT],
        const ds4_glm52_mock_session sessions[DS4_GLM52_L0_RANK_COUNT],
        float *partial_values[DS4_GLM52_L0_RANK_COUNT],
        ds4_glm52_mock_hidden_partial partials[DS4_GLM52_L0_RANK_COUNT],
        ds4_glm52_mock_allreduce_kind kind,
        uint64_t seq,
        int layer) {
    char err[192] = "";
    for (int rank = 0; rank < DS4_GLM52_L0_RANK_COUNT; rank++) {
        check(ds4_glm52_mock_make_hidden_partial(
                  &models[rank],
                  &sessions[rank],
                  kind,
                  seq,
                  layer,
                  partial_values[rank],
                  DS4_GLM52_MOCK_N_EMBD,
                  &partials[rank],
                  err,
                  sizeof(err)),
              "rank hidden partial should be created");
    }
}

static void add_order(ds4_glm52_mock_allreduce_step *step,
                      const ds4_glm52_mock_hidden_partial partials[DS4_GLM52_L0_RANK_COUNT],
                      const int order[DS4_GLM52_L0_RANK_COUNT]) {
    char err[192] = "";
    for (int i = 0; i < DS4_GLM52_L0_RANK_COUNT; i++) {
        check(ds4_glm52_mock_allreduce_add_contribution(step,
                                                        &partials[order[i]],
                                                        err,
                                                        sizeof(err)),
              "rank hidden partial should enter all-reduce step");
    }
}

static void test_allreduce_sums_in_rank_order(void) {
    ds4_glm52_mock_model models[DS4_GLM52_L0_RANK_COUNT];
    ds4_glm52_mock_session sessions[DS4_GLM52_L0_RANK_COUNT];
    ds4_glm52_mock_hidden_partial partials[DS4_GLM52_L0_RANK_COUNT];
    float *values[DS4_GLM52_L0_RANK_COUNT];
    char err[192] = "";

    init_models_and_sessions(models, sessions);
    for (int rank = 0; rank < DS4_GLM52_L0_RANK_COUNT; rank++) {
        values[rank] = malloc((size_t)DS4_GLM52_MOCK_N_EMBD * sizeof(float));
        check(values[rank] != NULL, "partial allocation should succeed");
    }
    fill_nonassoc_partials(values, DS4_GLM52_MOCK_N_EMBD);
    make_partials(models,
                  sessions,
                  values,
                  partials,
                  DS4_GLM52_MOCK_ALLREDUCE_ATTN,
                  11,
                  7);

    ds4_glm52_mock_allreduce_step arrival_a = {0};
    const int order_a[] = {2, 0, 3, 1};
    add_order(&arrival_a, partials, order_a);
    check(arrival_a.complete, "all-reduce step should complete after ranks 0..3");

    float *out_a = malloc((size_t)DS4_GLM52_MOCK_N_EMBD * sizeof(float));
    check(out_a != NULL, "all-reduce output allocation should succeed");
    check(ds4_glm52_mock_allreduce_finish(&arrival_a,
                                          out_a,
                                          DS4_GLM52_MOCK_N_EMBD,
                                          err,
                                          sizeof(err)),
          "all-reduce finish should produce replicated hidden");
    check(out_a[0] == 3.0f,
          "rank-ordered sum should not depend on arrival order");

    ds4_glm52_mock_allreduce_step arrival_b = {0};
    const int order_b[] = {3, 2, 1, 0};
    add_order(&arrival_b, partials, order_b);

    float *out_b = malloc((size_t)DS4_GLM52_MOCK_N_EMBD * sizeof(float));
    check(out_b != NULL, "second all-reduce output allocation should succeed");
    check(ds4_glm52_mock_allreduce_finish(&arrival_b,
                                          out_b,
                                          DS4_GLM52_MOCK_N_EMBD,
                                          err,
                                          sizeof(err)),
          "second all-reduce finish should produce replicated hidden");
    check(memcmp(out_a,
                 out_b,
                 (size_t)DS4_GLM52_MOCK_N_EMBD * sizeof(float)) == 0,
          "different arrival orders should produce identical replicated hidden");

    for (int rank = 0; rank < DS4_GLM52_L0_RANK_COUNT; rank++) free(values[rank]);
    free(out_a);
    free(out_b);
}

static void test_allreduce_rejects_missing_and_duplicate_ranks(void) {
    ds4_glm52_mock_model models[DS4_GLM52_L0_RANK_COUNT];
    ds4_glm52_mock_session sessions[DS4_GLM52_L0_RANK_COUNT];
    ds4_glm52_mock_hidden_partial partials[DS4_GLM52_L0_RANK_COUNT];
    float *values[DS4_GLM52_L0_RANK_COUNT];
    char err[192] = "";

    init_models_and_sessions(models, sessions);
    for (int rank = 0; rank < DS4_GLM52_L0_RANK_COUNT; rank++) {
        values[rank] = calloc((size_t)DS4_GLM52_MOCK_N_EMBD, sizeof(float));
        check(values[rank] != NULL, "partial allocation should succeed");
    }
    make_partials(models,
                  sessions,
                  values,
                  partials,
                  DS4_GLM52_MOCK_ALLREDUCE_FFN,
                  12,
                  8);

    ds4_glm52_mock_allreduce_step step = {0};
    check(ds4_glm52_mock_allreduce_add_contribution(&step,
                                                    &partials[0],
                                                    err,
                                                    sizeof(err)),
          "first all-reduce contribution should be accepted");
    check(!ds4_glm52_mock_allreduce_add_contribution(&step,
                                                     &partials[0],
                                                     err,
                                                     sizeof(err)),
          "duplicate all-reduce rank should be rejected");

    float *out = calloc((size_t)DS4_GLM52_MOCK_N_EMBD, sizeof(float));
    check(out != NULL, "all-reduce output allocation should succeed");
    check(!ds4_glm52_mock_allreduce_finish(&step,
                                           out,
                                           DS4_GLM52_MOCK_N_EMBD,
                                           err,
                                           sizeof(err)),
          "all-reduce finish should reject missing rank contributions");

    for (int rank = 0; rank < DS4_GLM52_L0_RANK_COUNT; rank++) free(values[rank]);
    free(out);
}

static void test_allreduce_rejects_identity_mismatches(void) {
    ds4_glm52_mock_model models[DS4_GLM52_L0_RANK_COUNT];
    ds4_glm52_mock_session sessions[DS4_GLM52_L0_RANK_COUNT];
    ds4_glm52_mock_hidden_partial partials[DS4_GLM52_L0_RANK_COUNT];
    float *values[DS4_GLM52_L0_RANK_COUNT];
    char err[192] = "";

    init_models_and_sessions(models, sessions);
    for (int rank = 0; rank < DS4_GLM52_L0_RANK_COUNT; rank++) {
        values[rank] = calloc((size_t)DS4_GLM52_MOCK_N_EMBD, sizeof(float));
        check(values[rank] != NULL, "partial allocation should succeed");
    }
    make_partials(models,
                  sessions,
                  values,
                  partials,
                  DS4_GLM52_MOCK_ALLREDUCE_ATTN,
                  13,
                  9);

    ds4_glm52_mock_allreduce_step seq_mismatch = {0};
    check(ds4_glm52_mock_allreduce_add_contribution(&seq_mismatch,
                                                    &partials[0],
                                                    err,
                                                    sizeof(err)),
          "first contribution should establish all-reduce sequence");
    ds4_glm52_mock_hidden_partial bad = partials[1];
    bad.seq++;
    check(!ds4_glm52_mock_allreduce_add_contribution(&seq_mismatch,
                                                     &bad,
                                                     err,
                                                     sizeof(err)),
          "all-reduce sequence mismatch should be rejected");

    ds4_glm52_mock_allreduce_step layer_mismatch = {0};
    check(ds4_glm52_mock_allreduce_add_contribution(&layer_mismatch,
                                                    &partials[0],
                                                    err,
                                                    sizeof(err)),
          "first contribution should establish all-reduce layer");
    bad = partials[1];
    bad.layer++;
    check(!ds4_glm52_mock_allreduce_add_contribution(&layer_mismatch,
                                                     &bad,
                                                     err,
                                                     sizeof(err)),
          "all-reduce layer mismatch should be rejected");

    ds4_glm52_mock_allreduce_step position_mismatch = {0};
    check(ds4_glm52_mock_allreduce_add_contribution(&position_mismatch,
                                                    &partials[0],
                                                    err,
                                                    sizeof(err)),
          "first contribution should establish all-reduce token position");
    bad = partials[1];
    bad.token_step_j++;
    check(!ds4_glm52_mock_allreduce_add_contribution(&position_mismatch,
                                                     &bad,
                                                     err,
                                                     sizeof(err)),
          "all-reduce token position mismatch should be rejected");

    ds4_glm52_mock_allreduce_step session_mismatch = {0};
    check(ds4_glm52_mock_allreduce_add_contribution(&session_mismatch,
                                                    &partials[0],
                                                    err,
                                                    sizeof(err)),
          "first contribution should establish all-reduce session identity");
    bad = partials[1];
    bad.session_hash++;
    check(!ds4_glm52_mock_allreduce_add_contribution(&session_mismatch,
                                                     &bad,
                                                     err,
                                                     sizeof(err)),
          "all-reduce session identity mismatch should be rejected");

    ds4_glm52_mock_allreduce_step model_mismatch = {0};
    check(ds4_glm52_mock_allreduce_add_contribution(&model_mismatch,
                                                    &partials[0],
                                                    err,
                                                    sizeof(err)),
          "first contribution should establish all-reduce model identity");
    bad = partials[1];
    bad.model_hash++;
    check(!ds4_glm52_mock_allreduce_add_contribution(&model_mismatch,
                                                     &bad,
                                                     err,
                                                     sizeof(err)),
          "all-reduce model identity mismatch should be rejected");

    for (int rank = 0; rank < DS4_GLM52_L0_RANK_COUNT; rank++) free(values[rank]);
}

static void test_allreduce_rejects_shape_dtype_and_nonfinite(void) {
    ds4_glm52_mock_model models[DS4_GLM52_L0_RANK_COUNT];
    ds4_glm52_mock_session sessions[DS4_GLM52_L0_RANK_COUNT];
    ds4_glm52_mock_hidden_partial partials[DS4_GLM52_L0_RANK_COUNT];
    float *values[DS4_GLM52_L0_RANK_COUNT];
    char err[192] = "";

    init_models_and_sessions(models, sessions);
    for (int rank = 0; rank < DS4_GLM52_L0_RANK_COUNT; rank++) {
        values[rank] = calloc((size_t)DS4_GLM52_MOCK_N_EMBD, sizeof(float));
        check(values[rank] != NULL, "partial allocation should succeed");
    }
    make_partials(models,
                  sessions,
                  values,
                  partials,
                  DS4_GLM52_MOCK_ALLREDUCE_ATTN,
                  14,
                  10);

    ds4_glm52_mock_allreduce_step dtype_mismatch = {0};
    check(ds4_glm52_mock_allreduce_add_contribution(&dtype_mismatch,
                                                    &partials[0],
                                                    err,
                                                    sizeof(err)),
          "first contribution should establish all-reduce dtype");
    ds4_glm52_mock_hidden_partial bad = partials[1];
    bad.dtype++;
    check(!ds4_glm52_mock_allreduce_add_contribution(&dtype_mismatch,
                                                     &bad,
                                                     err,
                                                     sizeof(err)),
          "all-reduce dtype mismatch should be rejected");

    ds4_glm52_mock_allreduce_step shape_mismatch = {0};
    check(ds4_glm52_mock_allreduce_add_contribution(&shape_mismatch,
                                                    &partials[0],
                                                    err,
                                                    sizeof(err)),
          "first contribution should establish all-reduce shape");
    bad = partials[1];
    bad.shape_hash++;
    check(!ds4_glm52_mock_allreduce_add_contribution(&shape_mismatch,
                                                     &bad,
                                                     err,
                                                     sizeof(err)),
          "all-reduce shape hash mismatch should be rejected");

    uint32_t qnan_bits = UINT32_C(0x7fc00000);
    memcpy(&values[2][0], &qnan_bits, sizeof(qnan_bits));
    check(!ds4_glm52_mock_make_hidden_partial(&models[2],
                                              &sessions[2],
                                              DS4_GLM52_MOCK_ALLREDUCE_ATTN,
                                              14,
                                              10,
                                              values[2],
                                              DS4_GLM52_MOCK_N_EMBD,
                                              &bad,
                                              err,
                                              sizeof(err)),
          "all-reduce contribution creation should reject non-finite partial values");

    for (int rank = 0; rank < DS4_GLM52_L0_RANK_COUNT; rank++) free(values[rank]);
}

int main(void) {
    test_l0_host_allreduce_buffers();
    test_l0_logits_gather_topk();
    test_allreduce_sums_in_rank_order();
    test_allreduce_rejects_missing_and_duplicate_ranks();
    test_allreduce_rejects_identity_mismatches();
    test_allreduce_rejects_shape_dtype_and_nonfinite();
    puts("test_glm52_tp4_allreduce: ok");
    return 0;
}
