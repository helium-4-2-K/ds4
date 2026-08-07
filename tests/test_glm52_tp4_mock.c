#include "ds4.h"
#include "ds4_glm52_mock.h"

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>

static void check(bool cond, const char *msg) {
    if (!cond) {
        fprintf(stderr, "test_glm52_tp4_mock: %s\n", msg);
        exit(1);
    }
}

static void make_socket_pair(int fds[2]) {
    check(socketpair(AF_UNIX, SOCK_STREAM, 0, fds) == 0,
          "socketpair should succeed");
}

static void close_pair(int fds[2]) {
    if (fds[0] >= 0) close(fds[0]);
    if (fds[1] >= 0) close(fds[1]);
    fds[0] = -1;
    fds[1] = -1;
}

static pid_t fork_send_tokens(int fd, const int *tokens, size_t token_count) {
    pid_t pid = fork();
    check(pid >= 0, "token transport fork should succeed");
    if (pid == 0) {
        char err[192] = "";
        bool ok = ds4_glm52_mock_transport_send_tokens(fd,
                                                       tokens,
                                                       token_count,
                                                       err,
                                                       sizeof(err));
        close(fd);
        _exit(ok ? 0 : 2);
    }
    return pid;
}

static void wait_sender_ok(pid_t pid) {
    int status = 0;
    check(waitpid(pid, &status, 0) == pid, "token sender waitpid should succeed");
    check(WIFEXITED(status) && WEXITSTATUS(status) == 0,
          "token sender should exit successfully");
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
    check(ds4_glm52_mock_tp4_prefill(models,
                                     prompt,
                                     sizeof(prompt) / sizeof(prompt[0]),
                                     sessions,
                                     err,
                                     sizeof(err)),
          "TP4 mock prefill should run rank-local work and all-reduce collectives");
    for (int rank = 0; rank < DS4_GLM52_L0_RANK_COUNT; rank++) {
        check(sessions[rank].session_hash == sessions[0].session_hash,
              "all ranks should preserve one logical session identity");
        check(sessions[rank].token_step_j == 4,
              "prefill token_step_j should point at prompt-length cursor");
        check(sessions[rank].kv_length == 4,
              "prefill should commit four KV rows");
        check(sessions[rank].hidden_checksum == sessions[0].hidden_checksum,
              "TP4 prefill should publish replicated hidden after all-reduce");
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
    assert_complete_step(&prefill, 4, 4);

    const int gathered_token = prefill.coordinator_token;
    check(ds4_glm52_mock_tp4_decode(models,
                                    sessions,
                                    gathered_token,
                                    err,
                                    sizeof(err)),
          "TP4 mock decode should consume gathered token and all-reduce hidden");
    for (int rank = 0; rank < DS4_GLM52_L0_RANK_COUNT; rank++) {
        check(sessions[rank].hidden_checksum == sessions[0].hidden_checksum,
              "TP4 decode should publish replicated hidden after all-reduce");
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
    assert_complete_step(&decode, 5, 5);
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
    ds4_glm52_mock_session bad_kv_cursor = sessions[1];
    bad_kv_cursor.kv_length++;
    check(!ds4_glm52_mock_tp4_step_add_rank(&cursor_mismatch,
                                            &models[1],
                                            &bad_kv_cursor,
                                            DS4_GLM52_TP4_COMMAND_PREFILL,
                                            1,
                                            err,
                                            sizeof(err)),
          "cursor mismatch should fail");

    ds4_glm52_mock_tp4_step step_cursor_mismatch = {0};
    check(ds4_glm52_mock_tp4_step_add_rank(&step_cursor_mismatch,
                                           &models[0],
                                           &sessions[0],
                                           DS4_GLM52_TP4_COMMAND_PREFILL,
                                           1,
                                           err,
                                           sizeof(err)),
          "first rank contribution should establish token cursor");
    ds4_glm52_mock_session bad_token_cursor = sessions[2];
    bad_token_cursor.token_step_j++;
    check(!ds4_glm52_mock_tp4_step_add_rank(&step_cursor_mismatch,
                                            &models[2],
                                            &bad_token_cursor,
                                            DS4_GLM52_TP4_COMMAND_PREFILL,
                                            1,
                                            err,
                                            sizeof(err)),
          "token step cursor mismatch should fail");

    ds4_glm52_mock_rank_step bad_contribution;
    check(ds4_glm52_mock_make_rank_step(&models[2],
                                        &sessions[2],
                                        DS4_GLM52_TP4_COMMAND_PREFILL,
                                        1,
                                        &bad_contribution,
                                        err,
                                        sizeof(err)),
          "rank step creation should succeed");
    bad_contribution.candidate_token = bad_contribution.vocab_end;
    ds4_glm52_mock_tp4_step bad_step = {0};
    check(!ds4_glm52_mock_tp4_step_add_contribution(&bad_step,
                                                    &bad_contribution,
                                                    err,
                                                    sizeof(err)),
          "candidate outside rank vocab shard should fail");

    ds4_glm52_mock_tp4_step model_mismatch = {0};
    check(ds4_glm52_mock_tp4_step_add_rank(&model_mismatch,
                                           &models[0],
                                           &sessions[0],
                                           DS4_GLM52_TP4_COMMAND_PREFILL,
                                           1,
                                           err,
                                           sizeof(err)),
          "first rank contribution should establish model identity");
    check(ds4_glm52_mock_make_rank_step(&models[1],
                                        &sessions[1],
                                        DS4_GLM52_TP4_COMMAND_PREFILL,
                                        1,
                                        &bad_contribution,
                                        err,
                                        sizeof(err)),
          "second rank step creation should succeed");
    bad_contribution.model_hash++;
    check(!ds4_glm52_mock_tp4_step_add_contribution(&model_mismatch,
                                                    &bad_contribution,
                                                    err,
                                                    sizeof(err)),
          "model identity mismatch should fail");
}

static void test_tp4_mock_tie_breaks_by_token_id(void) {
    ds4_glm52_mock_model models[DS4_GLM52_L0_RANK_COUNT];
    ds4_glm52_mock_session sessions[DS4_GLM52_L0_RANK_COUNT];
    char err[192] = "";

    init_models(models);
    prefill_sessions(models, sessions);

    ds4_glm52_mock_tp4_step step = {0};
    const int order[] = {3, 2, 1, 0};
    for (int i = 0; i < DS4_GLM52_L0_RANK_COUNT; i++) {
        const int rank = order[i];
        ds4_glm52_mock_rank_step contribution;
        check(ds4_glm52_mock_make_rank_step(&models[rank],
                                            &sessions[rank],
                                            DS4_GLM52_TP4_COMMAND_PREFILL,
                                            9,
                                            &contribution,
                                            err,
                                            sizeof(err)),
              "tie-break rank step creation should succeed");
        contribution.candidate_score = 42.0f;
        contribution.candidate_token = contribution.vocab_start + 7;
        check(ds4_glm52_mock_tp4_step_add_contribution(&step,
                                                       &contribution,
                                                       err,
                                                       sizeof(err)),
              "tie-break contribution should be accepted");
    }

    check(step.complete, "tie-break step should complete");
    check(step.coordinator_rank == 0,
          "equal-score top-k merge should choose the lowest token id");
    check(step.coordinator_token == models[0].vocab_start + 7,
          "tie-break coordinator token should be rank 0's lowest token");
}

static void test_tp4_mock_prefill_decode_use_allreduce_hidden(void) {
    ds4_glm52_mock_model models[DS4_GLM52_L0_RANK_COUNT];
    ds4_glm52_mock_session rank_local[DS4_GLM52_L0_RANK_COUNT];
    ds4_glm52_mock_session tp4[DS4_GLM52_L0_RANK_COUNT];
    int prompt[] = {100, 101, 102, 103};
    char err[192] = "";

    init_models(models);
    for (int rank = 0; rank < DS4_GLM52_L0_RANK_COUNT; rank++) {
        check(ds4_glm52_mock_prefill(&models[rank],
                                     prompt,
                                     sizeof(prompt) / sizeof(prompt[0]),
                                     &rank_local[rank],
                                     NULL,
                                     0,
                                     err,
                                     sizeof(err)),
              "rank-local mock prefill should succeed");
    }
    check(ds4_glm52_mock_tp4_prefill(models,
                                     prompt,
                                     sizeof(prompt) / sizeof(prompt[0]),
                                     tp4,
                                     err,
                                     sizeof(err)),
          "TP4 mock prefill should succeed");
    check(tp4[0].hidden_checksum != rank_local[0].hidden_checksum,
          "TP4 prefill hidden should include all-reduce output, not just rank-local checksum");
    for (int rank = 1; rank < DS4_GLM52_L0_RANK_COUNT; rank++) {
        check(tp4[rank].hidden_checksum == tp4[0].hidden_checksum,
              "TP4 prefill hidden should be replicated across ranks");
    }

    const int input_token = 4242;
    for (int rank = 0; rank < DS4_GLM52_L0_RANK_COUNT; rank++) {
        check(ds4_glm52_mock_decode(&models[rank],
                                    &rank_local[rank],
                                    input_token,
                                    NULL,
                                    0,
                                    err,
                                    sizeof(err)),
              "rank-local mock decode should succeed");
    }
    check(ds4_glm52_mock_tp4_decode(models,
                                    tp4,
                                    input_token,
                                    err,
                                    sizeof(err)),
          "TP4 mock decode should succeed");
    check(tp4[0].hidden_checksum != rank_local[0].hidden_checksum,
          "TP4 decode hidden should include all-reduce output, not just rank-local checksum");
    for (int rank = 1; rank < DS4_GLM52_L0_RANK_COUNT; rank++) {
        check(tp4[rank].hidden_checksum == tp4[0].hidden_checksum,
              "TP4 decode hidden should be replicated across ranks");
    }
}

static void test_tp4_mock_short_prompt_dcp_empty_shards(void) {
    ds4_glm52_mock_model models[DS4_GLM52_L0_RANK_COUNT];
    ds4_glm52_mock_session sessions[DS4_GLM52_L0_RANK_COUNT];
    int prompt[] = {777};
    char err[192] = "";

    init_models(models);
    check(ds4_glm52_mock_tp4_prefill(models,
                                     prompt,
                                     sizeof(prompt) / sizeof(prompt[0]),
                                     sessions,
                                     err,
                                     sizeof(err)),
          "TP4 mock prefill should tolerate DCP ranks with empty short-prompt shards");
    for (int rank = 0; rank < DS4_GLM52_L0_RANK_COUNT; rank++) {
        check(sessions[rank].kv_length == 1,
              "short-prompt prefill should commit one KV row");
        check(sessions[rank].hidden_checksum == sessions[0].hidden_checksum,
              "short-prompt TP4 hidden should be replicated");
    }
}

static void test_mock_transport_round_trip_and_rejections(void) {
    ds4_glm52_mock_model models[DS4_GLM52_L0_RANK_COUNT];
    ds4_glm52_mock_session sessions[DS4_GLM52_L0_RANK_COUNT];
    char err[192] = "";

    init_models(models);
    prefill_sessions(models, sessions);

    ds4_glm52_mock_rank_step sent;
    ds4_glm52_mock_rank_step received;
    check(ds4_glm52_mock_make_rank_step(&models[1],
                                        &sessions[1],
                                        DS4_GLM52_TP4_COMMAND_PREFILL,
                                        7,
                                        &sent,
                                        err,
                                        sizeof(err)),
          "rank step creation should succeed for transport");
    int fds[2] = {-1, -1};
    make_socket_pair(fds);
    check(ds4_glm52_mock_transport_send_rank_step(fds[0],
                                                  &sent,
                                                  err,
                                                  sizeof(err)),
          "rank step transport send should succeed");
    check(ds4_glm52_mock_transport_recv_rank_step(fds[1],
                                                  &received,
                                                  err,
                                                  sizeof(err)),
          "rank step transport recv should succeed");
    check(received.present &&
              received.rank == sent.rank &&
              received.model_hash == sent.model_hash &&
              received.session_hash == sent.session_hash &&
              received.command == sent.command &&
              received.seq == sent.seq &&
              received.candidate_token == sent.candidate_token &&
              received.token_step_j == sent.token_step_j &&
              received.kv_length == sent.kv_length,
          "rank step transport should preserve contribution identity");
    close_pair(fds);

    const int tokens[] = {10, 11, 12, 13};
    int token_out[4] = {0};
    size_t token_count = 0;
    make_socket_pair(fds);
    pid_t sender = fork_send_tokens(fds[0],
                                    tokens,
                                    sizeof(tokens) / sizeof(tokens[0]));
    close(fds[0]);
    fds[0] = -1;
    check(ds4_glm52_mock_transport_recv_tokens(fds[1],
                                               token_out,
                                               sizeof(token_out) / sizeof(token_out[0]),
                                               &token_count,
                                               err,
                                               sizeof(err)),
          "token transport recv should succeed");
    check(token_count == sizeof(tokens) / sizeof(tokens[0]) &&
              memcmp(tokens, token_out, sizeof(tokens)) == 0,
          "token transport should preserve the prompt span");
    wait_sender_ok(sender);
    close_pair(fds);

    make_socket_pair(fds);
    sender = fork_send_tokens(fds[0],
                              tokens,
                              sizeof(tokens) / sizeof(tokens[0]));
    close(fds[0]);
    fds[0] = -1;
    check(!ds4_glm52_mock_transport_recv_tokens(fds[1],
                                                token_out,
                                                2,
                                                &token_count,
                                                err,
                                                sizeof(err)),
          "token transport should reject receiver capacity overflow");
    wait_sender_ok(sender);
    close_pair(fds);

    make_socket_pair(fds);
    uint8_t corrupt_header[16] = {0};
    check(write(fds[0], corrupt_header, sizeof(corrupt_header)) ==
              (ssize_t)sizeof(corrupt_header),
          "corrupt frame write should succeed");
    check(!ds4_glm52_mock_transport_recv_rank_step(fds[1],
                                                   &received,
                                                   err,
                                                   sizeof(err)),
          "rank step transport should reject corrupt frames");
    close_pair(fds);

    make_socket_pair(fds);
    const uint8_t truncated[] = {0x35, 0x4d, 0x53};
    check(write(fds[0], truncated, sizeof(truncated)) ==
              (ssize_t)sizeof(truncated),
          "truncated frame write should succeed");
    close(fds[0]);
    fds[0] = -1;
    check(!ds4_glm52_mock_transport_recv_tokens(fds[1],
                                                token_out,
                                                sizeof(token_out) / sizeof(token_out[0]),
                                                &token_count,
                                                err,
                                                sizeof(err)),
          "token transport should reject truncated frames");
    close_pair(fds);
}

static void add_rank_contribution_scored(
        ds4_glm52_mock_tp4_step *step,
        const ds4_glm52_mock_model *model,
        const ds4_glm52_mock_session *session,
        float score,
        int token_offset,
        char *err,
        size_t err_size) {
    ds4_glm52_mock_rank_step contribution;
    check(ds4_glm52_mock_make_rank_step(model, session,
                                        DS4_GLM52_TP4_COMMAND_PREFILL, 11,
                                        &contribution, err, err_size),
          "scored rank step creation should succeed");
    contribution.candidate_score = score;
    contribution.candidate_token = model->vocab_start + token_offset;
    check(ds4_glm52_mock_tp4_step_add_contribution(step, &contribution,
                                                   err, err_size),
          "scored contribution should be accepted");
}

static void test_tp4_mock_shards_tile_vocab_exactly(void) {
    ds4_glm52_mock_model models[DS4_GLM52_L0_RANK_COUNT];
    ds4_glm52_mock_session sessions[DS4_GLM52_L0_RANK_COUNT];
    char err[192] = "";
    init_models(models);
    prefill_sessions(models, sessions);

    ds4_glm52_mock_tp4_step step = {0};
    for (int rank = 0; rank < DS4_GLM52_L0_RANK_COUNT; rank++) {
        check(ds4_glm52_mock_tp4_step_add_rank(&step,
                                               &models[rank],
                                               &sessions[rank],
                                               DS4_GLM52_TP4_COMMAND_PREFILL,
                                               21,
                                               err,
                                               sizeof(err)),
              "rank contribution should be accepted for coverage tiling");
    }
    check(step.complete, "four shards should complete coverage");
    const int span = DS4_GLM52_MOCK_N_VOCAB / DS4_GLM52_L0_TP_SIZE;
    int covered = 0;
    for (int rank = 0; rank < DS4_GLM52_L0_RANK_COUNT; rank++) {
        const ds4_glm52_mock_rank_step *r = &step.ranks[rank];
        check(r->vocab_start == covered,
              "gathered shards should be contiguous with no gap");
        check(r->vocab_end - r->vocab_start == span,
              "gathered shards should be non-empty fixed-span partitions");
        check(r->candidate_token >= r->vocab_start &&
                  r->candidate_token < r->vocab_end,
              "gathered candidate should stay inside its owned shard");
        covered = r->vocab_end;
    }
    check(covered == DS4_GLM52_MOCK_N_VOCAB,
          "four gathered shards should cover the global vocabulary exactly once");
}

static void test_tp4_mock_rejects_invalid_vocab_shards(void) {
    ds4_glm52_mock_model models[DS4_GLM52_L0_RANK_COUNT];
    ds4_glm52_mock_session sessions[DS4_GLM52_L0_RANK_COUNT];
    char err[192] = "";
    init_models(models);
    prefill_sessions(models, sessions);

    ds4_glm52_mock_rank_step contribution;
    check(ds4_glm52_mock_make_rank_step(&models[1],
                                        &sessions[1],
                                        DS4_GLM52_TP4_COMMAND_PREFILL,
                                        5,
                                        &contribution,
                                        err,
                                        sizeof(err)),
          "rank step creation should succeed before vocab mutation");
    contribution.candidate_score = 3.0f;

    {
        ds4_glm52_mock_tp4_step step = {0};
        ds4_glm52_mock_rank_step c = contribution;
        c.vocab_start += 17;
        c.candidate_token = c.vocab_start + 5;
        check(!ds4_glm52_mock_tp4_step_add_contribution(&step, &c,
                                                        err, sizeof(err)),
              "vocab shard gap should be rejected");
    }
    {
        ds4_glm52_mock_tp4_step step = {0};
        ds4_glm52_mock_rank_step c = contribution;
        c.vocab_end -= 17;
        c.candidate_token = c.vocab_end - 5;
        check(!ds4_glm52_mock_tp4_step_add_contribution(&step, &c,
                                                        err, sizeof(err)),
              "vocab shard overlap should be rejected");
    }
    {
        ds4_glm52_mock_tp4_step step = {0};
        ds4_glm52_mock_rank_step c = contribution;
        c.vocab_end = c.vocab_start;
        check(!ds4_glm52_mock_tp4_step_add_contribution(&step, &c,
                                                        err, sizeof(err)),
              "empty vocab shard should be rejected");
    }
    {
        ds4_glm52_mock_tp4_step step = {0};
        ds4_glm52_mock_rank_step c = contribution;
        c.vocab_start = -1;
        c.candidate_token = 0;
        check(!ds4_glm52_mock_tp4_step_add_contribution(&step, &c,
                                                        err, sizeof(err)),
              "negative vocab start should be rejected");
    }
    {
        ds4_glm52_mock_tp4_step step = {0};
        ds4_glm52_mock_rank_step c = contribution;
        c.vocab_end = DS4_GLM52_MOCK_N_VOCAB + 100;
        c.candidate_token = c.vocab_end - 1;
        check(!ds4_glm52_mock_tp4_step_add_contribution(&step, &c,
                                                        err, sizeof(err)),
              "vocab end beyond global vocab should be rejected");
    }
    {
        ds4_glm52_mock_tp4_step step = {0};
        ds4_glm52_mock_rank_step c = contribution;
        c.candidate_token = -1;
        check(!ds4_glm52_mock_tp4_step_add_contribution(&step, &c,
                                                        err, sizeof(err)),
              "negative candidate token should be rejected");
    }
    {
        ds4_glm52_mock_tp4_step step = {0};
        ds4_glm52_mock_rank_step c = contribution;
        c.candidate_token = DS4_GLM52_MOCK_N_VOCAB;
        check(!ds4_glm52_mock_tp4_step_add_contribution(&step, &c,
                                                        err, sizeof(err)),
              "candidate token at global vocab end should be rejected");
    }
}

static void test_tp4_mock_missing_rank_stays_incomplete(void) {
    ds4_glm52_mock_model models[DS4_GLM52_L0_RANK_COUNT];
    ds4_glm52_mock_session sessions[DS4_GLM52_L0_RANK_COUNT];
    char err[192] = "";
    init_models(models);
    prefill_sessions(models, sessions);

    ds4_glm52_mock_tp4_step step = {0};
    for (int rank = 0; rank < DS4_GLM52_L0_RANK_COUNT - 1; rank++) {
        check(ds4_glm52_mock_tp4_step_add_rank(&step,
                                               &models[rank],
                                               &sessions[rank],
                                               DS4_GLM52_TP4_COMMAND_PREFILL,
                                               31,
                                               err,
                                               sizeof(err)),
              "partial rank contribution should be accepted");
    }
    check(!step.complete, "missing rank should leave the gather incomplete");
    check(ds4_glm52_mock_tp4_step_add_rank(&step,
                                           &models[3],
                                           &sessions[3],
                                           DS4_GLM52_TP4_COMMAND_PREFILL,
                                           31,
                                           err,
                                           sizeof(err)),
          "final rank contribution should be accepted");
    check(step.complete, "gather should complete once all ranks contribute");
}

static void test_tp4_mock_merge_deterministic_across_orders(void) {
    ds4_glm52_mock_model models[DS4_GLM52_L0_RANK_COUNT];
    ds4_glm52_mock_session sessions[DS4_GLM52_L0_RANK_COUNT];
    init_models(models);
    prefill_sessions(models, sessions);

    /* Pattern A: rank1 wins on the highest score even though rank3 holds the
     * smallest token offset - proves score dominates the token-id tie-break. */
    const float scores_a[DS4_GLM52_L0_RANK_COUNT] = {5.0f, 9.0f, 5.0f, 7.0f};
    const int offsets_a[DS4_GLM52_L0_RANK_COUNT] = {100, 200, 300, 50};
    /* Pattern B: ranks 1..3 tie at the max score; the globally lowest raw
     * token id (on rank1, across a rank boundary) must win in every order. */
    const float scores_b[DS4_GLM52_L0_RANK_COUNT] = {1.0f, 8.0f, 8.0f, 8.0f};
    const int offsets_b[DS4_GLM52_L0_RANK_COUNT] = {50, 30, 25, 20};

    const struct {
        const float *scores;
        const int *offsets;
    } patterns[] = {
        {scores_a, offsets_a},
        {scores_b, offsets_b},
    };
    for (size_t pat = 0;
         pat < sizeof(patterns) / sizeof(patterns[0]);
         pat++) {
        const float *scores = patterns[pat].scores;
        const int *offsets = patterns[pat].offsets;
        int best_rank = -1;
        int best_token = -1;
        float best_score = 0.0f;
        for (int r = 0; r < DS4_GLM52_L0_RANK_COUNT; r++) {
            const int token = models[r].vocab_start + offsets[r];
            if (best_rank < 0 || scores[r] > best_score ||
                (scores[r] == best_score && token < best_token)) {
                best_rank = r;
                best_score = scores[r];
                best_token = token;
            }
        }
        check(best_rank >= 0, "reference merge should always select a winner");

        for (int i0 = 0; i0 < 4; i0++)
        for (int i1 = 0; i1 < 4; i1++) {
            if (i1 == i0) continue;
            for (int i2 = 0; i2 < 4; i2++) {
                if (i2 == i0 || i2 == i1) continue;
                for (int i3 = 0; i3 < 4; i3++) {
                    if (i3 == i0 || i3 == i1 || i3 == i2) continue;
                    const int order[4] = {i0, i1, i2, i3};
                    ds4_glm52_mock_tp4_step step = {0};
                    char err[192] = "";
                    for (int k = 0; k < DS4_GLM52_L0_RANK_COUNT; k++) {
                        const int rank = order[k];
                        add_rank_contribution_scored(&step,
                                                     &models[rank],
                                                     &sessions[rank],
                                                     scores[rank],
                                                     offsets[rank],
                                                     err,
                                                     sizeof(err));
                    }
                    check(step.complete, "permuted gather should complete");
                    check(step.coordinator_rank == best_rank,
                          "coordinator rank should be independent of arrival order");
                    check(step.coordinator_token == best_token,
                          "coordinator token should be independent of arrival order");
                    check(step.coordinator_score == best_score,
                          "coordinator score should be independent of arrival order");
                }
            }
        }
    }
}

int main(void) {
    test_prefill_and_decode_tp4_mock_steps();
    test_tp4_mock_rejects_mismatches();
    test_tp4_mock_tie_breaks_by_token_id();
    test_tp4_mock_prefill_decode_use_allreduce_hidden();
    test_tp4_mock_short_prompt_dcp_empty_shards();
    test_tp4_mock_shards_tile_vocab_exactly();
    test_tp4_mock_rejects_invalid_vocab_shards();
    test_tp4_mock_missing_rank_stays_incomplete();
    test_tp4_mock_merge_deterministic_across_orders();
    test_mock_transport_round_trip_and_rejections();
    puts("test_glm52_tp4_mock: ok");
    return 0;
}
