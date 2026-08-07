#include "ds4.h"
#include "ds4_glm52_mock.h"

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void check(bool cond, const char *msg) {
    if (!cond) {
        fprintf(stderr, "test_glm52_tp4_mock: %s\n", msg);
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

static void init_models(ds4_glm52_mock_model models[DS4_GLM52_L0_RANK_COUNT]) {
    char err[192] = "";
    for (int rank = 0; rank < DS4_GLM52_L0_RANK_COUNT; rank++) {
        ds4_glm52_l0_config cfg = mock_cfg(rank);
        check(ds4_glm52_mock_model_init(&cfg, &models[rank], err, sizeof(err)),
              "rank mock model should initialize");
        check(models[rank].rank == rank, "mock model rank should match config");
        check(models[rank].q_head_start ==
                  rank * DS4_GLM52_L0_Q_HEADS_PER_RANK,
              "rank Q-head start should match TP4 partition");
        check(models[rank].q_head_end ==
                  (rank + 1) * DS4_GLM52_L0_Q_HEADS_PER_RANK,
              "rank Q-head end should match TP4 partition");
        check(models[rank].model_hash == models[0].model_hash,
              "all ranks should share the same logical model hash");
    }
}

static void prefill_sessions(
        const ds4_glm52_mock_model models[DS4_GLM52_L0_RANK_COUNT],
        ds4_glm52_mock_session sessions[DS4_GLM52_L0_RANK_COUNT]) {
    int prompt[] = {100, 101, 102, 103};
    char err[192] = "";
    for (int rank = 0; rank < DS4_GLM52_L0_RANK_COUNT; rank++) {
        check(ds4_glm52_mock_prefill(&models[rank],
                                     prompt,
                                     sizeof(prompt) / sizeof(prompt[0]),
                                     &sessions[rank],
                                     NULL,
                                     0,
                                     err,
                                     sizeof(err)),
              "rank mock prefill should succeed");
        check(sessions[rank].session_hash == sessions[0].session_hash,
              "all ranks should preserve one logical session identity");
        check(sessions[rank].token_step_j == 3,
              "prefill token_step_j should point at last prompt token");
        check(sessions[rank].kv_length == 4,
              "prefill should commit four KV rows");
    }
}

static void register_rank_group(
        const ds4_glm52_mock_model models[DS4_GLM52_L0_RANK_COUNT],
        ds4_glm52_tp4_rank_group *group) {
    char err[192] = "";
    ds4_glm52_tp4_rank_group_init(group, models[0].model_hash, 22, 33);
    for (int rank = 0; rank < DS4_GLM52_L0_RANK_COUNT; rank++) {
        check(ds4_glm52_tp4_rank_group_register(group,
                                                rank,
                                                DS4_GLM52_L0_TP_SIZE,
                                                DS4_GLM52_L0_DCP_SIZE,
                                                models[rank].model_hash,
                                                22,
                                                33,
                                                err,
                                                sizeof(err)),
              "mock rank should register into TP4 group");
    }
    check(ds4_glm52_tp4_rank_group_ready(group),
          "four mock ranks should make TP4 group ready");
}

static void add_all_ranks(
        ds4_glm52_mock_tp4_step *step,
        ds4_glm52_tp4_rank_group *group,
        const ds4_glm52_mock_model models[DS4_GLM52_L0_RANK_COUNT],
        const ds4_glm52_mock_session sessions[DS4_GLM52_L0_RANK_COUNT],
        ds4_glm52_tp4_command command,
        uint64_t seq) {
    char err[192] = "";
    const int order[] = {2, 0, 3, 1};
    for (int i = 0; i < DS4_GLM52_L0_RANK_COUNT; i++) {
        const int rank = order[i];
        check(ds4_glm52_mock_tp4_step_add_rank(step,
                                               &models[rank],
                                               &sessions[rank],
                                               command,
                                               seq,
                                               err,
                                               sizeof(err)),
              "mock rank should contribute to TP4 step");
        check(ds4_glm52_tp4_rank_group_ack(group,
                                           rank,
                                           command,
                                           seq,
                                           err,
                                           sizeof(err)),
              "mock rank should ack TP4 command after contribution");
    }
    check(step->complete, "TP4 mock step should complete after all ranks");
    check(ds4_glm52_tp4_rank_group_command_done(group),
          "rank group command should complete after all mock acks");
}

static void assert_complete_step(const ds4_glm52_mock_tp4_step *step,
                                 uint64_t token_step_j,
                                 uint64_t kv_length) {
    check(step->rank_mask == 0x0fu, "TP4 step should contain ranks 0..3");
    check(step->token_step_j == token_step_j, "TP4 step cursor should match");
    check(step->kv_length == kv_length, "TP4 step KV length should match");
    check(step->coordinator_rank >= 0 &&
              step->coordinator_rank < DS4_GLM52_L0_RANK_COUNT,
          "TP4 step should select a coordinator-visible rank candidate");
    for (int rank = 0; rank < DS4_GLM52_L0_RANK_COUNT; rank++) {
        const ds4_glm52_mock_rank_step *r = &step->ranks[rank];
        check(r->present, "rank contribution should be present");
        check(r->q_head_start == rank * DS4_GLM52_L0_Q_HEADS_PER_RANK,
              "rank contribution should preserve Q-head shard start");
        check(r->q_head_end == (rank + 1) * DS4_GLM52_L0_Q_HEADS_PER_RANK,
              "rank contribution should preserve Q-head shard end");
        check(r->candidate_token >= r->vocab_start &&
                  r->candidate_token < r->vocab_end,
              "rank candidate should come from rank-local vocab shard");
    }
    const ds4_glm52_mock_rank_step *winner =
        &step->ranks[step->coordinator_rank];
    check(step->coordinator_token == winner->candidate_token,
          "coordinator token should be gathered from winning rank shard");
}

static void test_prefill_and_decode_tp4_mock_steps(void) {
    ds4_glm52_mock_model models[DS4_GLM52_L0_RANK_COUNT];
    ds4_glm52_mock_session sessions[DS4_GLM52_L0_RANK_COUNT];
    ds4_glm52_tp4_rank_group group;
    char err[192] = "";

    init_models(models);
    prefill_sessions(models, sessions);
    register_rank_group(models, &group);

    uint64_t seq = 0;
    check(ds4_glm52_tp4_rank_group_broadcast(&group,
                                             0,
                                             DS4_GLM52_TP4_COMMAND_PREFILL,
                                             &seq,
                                             err,
                                             sizeof(err)),
          "rank0 should broadcast mock prefill");
    ds4_glm52_mock_tp4_step prefill = {0};
    add_all_ranks(&prefill,
                  &group,
                  models,
                  sessions,
                  DS4_GLM52_TP4_COMMAND_PREFILL,
                  seq);
    assert_complete_step(&prefill, 3, 4);

    const int gathered_token = prefill.coordinator_token;
    for (int rank = 0; rank < DS4_GLM52_L0_RANK_COUNT; rank++) {
        check(ds4_glm52_mock_decode(&models[rank],
                                    &sessions[rank],
                                    gathered_token,
                                    NULL,
                                    0,
                                    err,
                                    sizeof(err)),
              "rank mock decode should consume gathered token");
    }

    check(ds4_glm52_tp4_rank_group_broadcast(&group,
                                             0,
                                             DS4_GLM52_TP4_COMMAND_DECODE,
                                             &seq,
                                             err,
                                             sizeof(err)),
          "rank0 should broadcast mock decode");
    ds4_glm52_mock_tp4_step decode = {0};
    add_all_ranks(&decode,
                  &group,
                  models,
                  sessions,
                  DS4_GLM52_TP4_COMMAND_DECODE,
                  seq);
    assert_complete_step(&decode, 4, 5);
}

static void test_tp4_mock_rejects_mismatches(void) {
    ds4_glm52_mock_model models[DS4_GLM52_L0_RANK_COUNT];
    ds4_glm52_mock_session sessions[DS4_GLM52_L0_RANK_COUNT];
    char err[192] = "";

    init_models(models);
    prefill_sessions(models, sessions);

    ds4_glm52_mock_tp4_step step = {0};
    check(ds4_glm52_mock_tp4_step_add_rank(&step,
                                           &models[0],
                                           &sessions[0],
                                           DS4_GLM52_TP4_COMMAND_PREFILL,
                                           1,
                                           err,
                                           sizeof(err)),
          "first rank contribution should establish step identity");
    check(!ds4_glm52_mock_tp4_step_add_rank(&step,
                                            &models[0],
                                            &sessions[0],
                                            DS4_GLM52_TP4_COMMAND_PREFILL,
                                            1,
                                            err,
                                            sizeof(err)),
          "duplicate rank contribution should fail");

    ds4_glm52_mock_tp4_step seq_mismatch = {0};
    check(ds4_glm52_mock_tp4_step_add_rank(&seq_mismatch,
                                           &models[0],
                                           &sessions[0],
                                           DS4_GLM52_TP4_COMMAND_PREFILL,
                                           1,
                                           err,
                                           sizeof(err)),
          "first rank contribution should establish sequence");
    check(!ds4_glm52_mock_tp4_step_add_rank(&seq_mismatch,
                                            &models[1],
                                            &sessions[1],
                                            DS4_GLM52_TP4_COMMAND_PREFILL,
                                            2,
                                            err,
                                            sizeof(err)),
          "sequence mismatch should fail");

    ds4_glm52_mock_tp4_step cursor_mismatch = {0};
    check(ds4_glm52_mock_tp4_step_add_rank(&cursor_mismatch,
                                           &models[0],
                                           &sessions[0],
                                           DS4_GLM52_TP4_COMMAND_PREFILL,
                                           1,
                                           err,
                                           sizeof(err)),
          "first rank contribution should establish cursor");
    sessions[1].kv_length++;
    check(!ds4_glm52_mock_tp4_step_add_rank(&cursor_mismatch,
                                            &models[1],
                                            &sessions[1],
                                            DS4_GLM52_TP4_COMMAND_PREFILL,
                                            1,
                                            err,
                                            sizeof(err)),
          "cursor mismatch should fail");
}

int main(void) {
    test_prefill_and_decode_tp4_mock_steps();
    test_tp4_mock_rejects_mismatches();
    puts("test_glm52_tp4_mock: ok");
    return 0;
}
