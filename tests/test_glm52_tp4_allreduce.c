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
    test_allreduce_sums_in_rank_order();
    test_allreduce_rejects_missing_and_duplicate_ranks();
    test_allreduce_rejects_identity_mismatches();
    test_allreduce_rejects_shape_dtype_and_nonfinite();
    puts("test_glm52_tp4_allreduce: ok");
    return 0;
}
