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
        check(sessions[rank].token_step_j == 4,
              "prefill token_step_j should point at prompt-length cursor");
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
    assert_complete_step(&prefill, 4, 4);

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
    sessions[1].kv_length++;
    check(!ds4_glm52_mock_tp4_step_add_rank(&cursor_mismatch,
                                            &models[1],
                                            &sessions[1],
                                            DS4_GLM52_TP4_COMMAND_PREFILL,
                                            1,
                                            err,
                                            sizeof(err)),
          "cursor mismatch should fail");

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

int main(void) {
    test_prefill_and_decode_tp4_mock_steps();
    test_tp4_mock_rejects_mismatches();
    test_mock_transport_round_trip_and_rejections();
    puts("test_glm52_tp4_mock: ok");
    return 0;
}
