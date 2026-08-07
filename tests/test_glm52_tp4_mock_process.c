#include "ds4.h"
#include "ds4_glm52_mock.h"

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

static void check(bool cond, const char *msg) {
    if (!cond) {
        fprintf(stderr, "test_glm52_tp4_mock_process: %s\n", msg);
        exit(1);
    }
}

static void close_checked(int fd) {
    if (fd >= 0) close(fd);
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
    cfg.fabric_addr = "127.0.0.1:0";
    return cfg;
}

static void worker_fail(int rank, const char *msg, const char *err) {
    fprintf(stderr,
            "test_glm52_tp4_mock_process: worker %d: %s%s%s\n",
            rank,
            msg,
            err && err[0] ? ": " : "",
            err && err[0] ? err : "");
    _exit(2);
}

static void worker_send_step(int fd,
                             int rank,
                             const ds4_glm52_mock_model *model,
                             const ds4_glm52_mock_session *session,
                             ds4_glm52_tp4_command command,
                             uint64_t seq) {
    char err[192] = "";
    ds4_glm52_mock_rank_step step;
    if (!ds4_glm52_mock_make_rank_step(model, session, command, seq,
                                       &step, err, sizeof(err))) {
        worker_fail(rank, "make rank step failed", err);
    }
    if (!ds4_glm52_mock_transport_send_rank_step(fd, &step,
                                                 err, sizeof(err))) {
        worker_fail(rank, "send rank step failed", err);
    }
    if (!ds4_glm52_tp4_transport_send_ack(fd, rank, command, seq,
                                          err, sizeof(err))) {
        worker_fail(rank, "send ack failed", err);
    }
}

static void worker_recv_session_state(int fd,
                                      int rank,
                                      ds4_glm52_mock_session *session) {
    char err[192] = "";
    ds4_glm52_mock_session_state state;
    if (!ds4_glm52_mock_transport_recv_session_state(fd,
                                                     &state,
                                                     err,
                                                     sizeof(err))) {
        worker_fail(rank, "recv replicated session state failed", err);
    }
    ds4_glm52_mock_session_import(session, &state);
}

static void worker_main(uint16_t port, int rank) {
    char err[192] = "";
    ds4_glm52_l0_config cfg = mock_cfg(rank);
    ds4_glm52_mock_model model;
    ds4_glm52_mock_session session;
    memset(&session, 0, sizeof(session));

    if (!ds4_glm52_mock_model_init(&cfg, &model, err, sizeof(err))) {
        worker_fail(rank, "mock model init failed", err);
    }

    ds4_glm52_tp4_tcp_endpoint endpoint = {
        .host = "127.0.0.1",
        .port = port,
    };
    int fd = -1;
    if (!ds4_glm52_tp4_tcp_connect(&endpoint, 1000, &fd, err, sizeof(err))) {
        worker_fail(rank, "connect failed", err);
    }

    ds4_glm52_tp4_transport_hello hello = {
        .rank = rank,
        .tp_size = DS4_GLM52_L0_TP_SIZE,
        .dcp_size = DS4_GLM52_L0_DCP_SIZE,
        .model_hash = model.model_hash,
        .config_hash = 22,
        .plan_hash = 33,
    };
    if (!ds4_glm52_tp4_transport_send_hello(fd, &hello,
                                            err, sizeof(err))) {
        worker_fail(rank, "send hello failed", err);
    }

    for (;;) {
        ds4_glm52_tp4_command command = DS4_GLM52_TP4_COMMAND_NONE;
        uint64_t seq = 0;
        if (!ds4_glm52_tp4_transport_recv_command(fd, &command, &seq,
                                                  err, sizeof(err))) {
            worker_fail(rank, "recv command failed", err);
        }
        if (command == DS4_GLM52_TP4_COMMAND_SHUTDOWN) {
            if (!ds4_glm52_tp4_transport_send_ack(fd, rank, command, seq,
                                                  err, sizeof(err))) {
                worker_fail(rank, "send shutdown ack failed", err);
            }
            close_checked(fd);
            _exit(0);
        }

        int tokens[DS4_GLM52_L0_PREFILL_MAX_PROMPT_TOKENS];
        size_t token_count = 0;
        if (!ds4_glm52_mock_transport_recv_tokens(fd,
                                                  tokens,
                                                  DS4_GLM52_L0_PREFILL_MAX_PROMPT_TOKENS,
                                                  &token_count,
                                                  err,
                                                  sizeof(err))) {
            worker_fail(rank, "recv tokens failed", err);
        }
        if (command == DS4_GLM52_TP4_COMMAND_PREFILL) {
            if (!ds4_glm52_mock_prefill(&model, tokens, token_count,
                                        &session, NULL, 0,
                                        err, sizeof(err))) {
                worker_fail(rank, "mock prefill failed", err);
            }
        } else if (command == DS4_GLM52_TP4_COMMAND_DECODE) {
            if (token_count != 1) {
                worker_fail(rank, "decode command expected one token", NULL);
            }
            if (!ds4_glm52_mock_decode(&model, &session, tokens[0],
                                       NULL, 0, err, sizeof(err))) {
                worker_fail(rank, "mock decode failed", err);
            }
        } else {
            worker_fail(rank, "unexpected command", NULL);
        }
        worker_send_step(fd, rank, &model, &session, command, seq);
        worker_recv_session_state(fd, rank, &session);
    }
}

static int newly_registered_rank(uint32_t before, uint32_t after) {
    uint32_t delta = after & ~before;
    for (int rank = 0; rank < DS4_GLM52_L0_RANK_COUNT; rank++) {
        if (delta == (UINT32_C(1) << rank)) return rank;
    }
    return -1;
}

static void send_command_and_tokens(int fd,
                                    ds4_glm52_tp4_command command,
                                    uint64_t seq,
                                    const int *tokens,
                                    size_t token_count) {
    char err[192] = "";
    check(ds4_glm52_tp4_transport_send_command(fd, command, seq,
                                               err, sizeof(err)),
          "coordinator should send command");
    check(ds4_glm52_mock_transport_send_tokens(fd, tokens, token_count,
                                               err, sizeof(err)),
          "coordinator should send token payload");
}

static void recv_worker_contribution(int fd,
                                     ds4_glm52_mock_tp4_step *step,
                                     ds4_glm52_tp4_rank_group *group) {
    char err[192] = "";
    ds4_glm52_mock_rank_step rank_step;
    check(ds4_glm52_mock_transport_recv_rank_step(fd, &rank_step,
                                                  err, sizeof(err)),
          "coordinator should receive rank step");
    check(ds4_glm52_mock_tp4_step_add_contribution(step, &rank_step,
                                                   err, sizeof(err)),
          "coordinator should accept rank contribution");
    check(ds4_glm52_tp4_transport_recv_ack_and_record(fd, group,
                                                      err, sizeof(err)),
          "coordinator should record worker ack");
}

static void send_worker_session_state(int fd,
                                      const ds4_glm52_mock_session *session) {
    char err[192] = "";
    ds4_glm52_mock_session_state state;
    ds4_glm52_mock_session_export(session, &state);
    check(ds4_glm52_mock_transport_send_session_state(fd,
                                                      &state,
                                                      err,
                                                      sizeof(err)),
          "coordinator should send replicated session state");
}

static void assert_complete_step(const ds4_glm52_mock_tp4_step *step,
                                 uint64_t token_step_j,
                                 uint64_t kv_length) {
    check(step->complete, "TP4 process mock step should be complete");
    check(step->rank_mask == 0x0fu, "TP4 process mock step should cover ranks 0..3");
    check(step->token_step_j == token_step_j, "step token cursor should match");
    check(step->kv_length == kv_length, "step KV length should match");
    check(step->coordinator_rank >= 0 &&
              step->coordinator_rank < DS4_GLM52_L0_RANK_COUNT,
          "step should select a coordinator rank");
    for (int rank = 0; rank < DS4_GLM52_L0_RANK_COUNT; rank++) {
        const ds4_glm52_mock_rank_step *r = &step->ranks[rank];
        check(r->present, "rank contribution should be present");
        check(r->q_head_start == rank * DS4_GLM52_L0_Q_HEADS_PER_RANK,
              "rank contribution should preserve Q-head start");
        check(r->q_head_end == (rank + 1) * DS4_GLM52_L0_Q_HEADS_PER_RANK,
              "rank contribution should preserve Q-head end");
        check(r->candidate_token >= r->vocab_start &&
                  r->candidate_token < r->vocab_end,
              "rank contribution token should stay inside vocab shard");
    }
}

static void test_tcp_process_prefill_decode_mock(void) {
    char err[192] = "";
    ds4_glm52_mock_model models[DS4_GLM52_L0_RANK_COUNT];
    ds4_glm52_mock_session sessions[DS4_GLM52_L0_RANK_COUNT];
    memset(sessions, 0, sizeof(sessions));
    for (int rank = 0; rank < DS4_GLM52_L0_RANK_COUNT; rank++) {
        ds4_glm52_l0_config cfg = mock_cfg(rank);
        check(ds4_glm52_mock_model_init(&cfg, &models[rank], err, sizeof(err)),
              "coordinator mock model should initialize all rank descriptors");
    }

    ds4_glm52_tp4_tcp_endpoint endpoint;
    check(ds4_glm52_tp4_tcp_parse_endpoint("127.0.0.1:0",
                                           &endpoint,
                                           err,
                                           sizeof(err)),
          "loopback endpoint should parse");
    int listen_fd = -1;
    uint16_t bound_port = 0;
    check(ds4_glm52_tp4_tcp_listen(&endpoint,
                                   &listen_fd,
                                   &bound_port,
                                   err,
                                   sizeof(err)),
          "coordinator listener should bind");

    pid_t pids[DS4_GLM52_L0_RANK_COUNT];
    int worker_fds[DS4_GLM52_L0_RANK_COUNT];
    memset(pids, 0, sizeof(pids));
    for (int rank = 0; rank < DS4_GLM52_L0_RANK_COUNT; rank++) {
        worker_fds[rank] = -1;
    }

    for (int rank = 1; rank < DS4_GLM52_L0_RANK_COUNT; rank++) {
        pid_t pid = fork();
        check(pid >= 0, "worker fork should succeed");
        if (pid == 0) {
            close_checked(listen_fd);
            worker_main(bound_port, rank);
        }
        pids[rank] = pid;
    }

    ds4_glm52_tp4_rank_group group;
    ds4_glm52_tp4_rank_group_init(&group, models[0].model_hash, 22, 33);
    check(ds4_glm52_tp4_rank_group_register(&group,
                                            0,
                                            DS4_GLM52_L0_TP_SIZE,
                                            DS4_GLM52_L0_DCP_SIZE,
                                            models[0].model_hash,
                                            22,
                                            33,
                                            err,
                                            sizeof(err)),
          "coordinator rank should register");
    for (int i = 1; i < DS4_GLM52_L0_RANK_COUNT; i++) {
        int fd = -1;
        check(ds4_glm52_tp4_tcp_accept(listen_fd, 1000, &fd,
                                       err, sizeof(err)),
              "coordinator should accept worker");
        uint32_t before = group.registered_mask;
        check(ds4_glm52_tp4_transport_recv_hello_and_register(fd,
                                                              &group,
                                                              err,
                                                              sizeof(err)),
              "worker hello should register");
        int rank = newly_registered_rank(before, group.registered_mask);
        check(rank > 0 && rank < DS4_GLM52_L0_RANK_COUNT,
              "registered worker rank should be identifiable");
        worker_fds[rank] = fd;
    }
    check(ds4_glm52_tp4_rank_group_ready(&group),
          "all TCP mock ranks should be ready");

    const int prompt[] = {100, 101, 102, 103};
    uint64_t seq = 0;
    check(ds4_glm52_tp4_rank_group_broadcast(&group,
                                             0,
                                             DS4_GLM52_TP4_COMMAND_PREFILL,
                                             &seq,
                                             err,
                                             sizeof(err)),
          "coordinator should broadcast prefill");
    for (int rank = 1; rank < DS4_GLM52_L0_RANK_COUNT; rank++) {
        send_command_and_tokens(worker_fds[rank],
                                DS4_GLM52_TP4_COMMAND_PREFILL,
                                seq,
                                prompt,
                                sizeof(prompt) / sizeof(prompt[0]));
    }
    check(ds4_glm52_mock_prefill(&models[0],
                                 prompt,
                                 sizeof(prompt) / sizeof(prompt[0]),
                                 &sessions[0],
                                 NULL,
                                 0,
                                 err,
                                 sizeof(err)),
          "coordinator mock prefill should run");
    ds4_glm52_mock_tp4_step prefill = {0};
    check(ds4_glm52_mock_tp4_step_add_rank(&prefill,
                                           &models[0],
                                           &sessions[0],
                                           DS4_GLM52_TP4_COMMAND_PREFILL,
                                           seq,
                                           err,
                                           sizeof(err)),
          "coordinator should add local prefill contribution");
    check(ds4_glm52_tp4_rank_group_ack(&group,
                                       0,
                                       DS4_GLM52_TP4_COMMAND_PREFILL,
                                       seq,
                                       err,
                                       sizeof(err)),
          "coordinator should ack local prefill");
    for (int rank = 1; rank < DS4_GLM52_L0_RANK_COUNT; rank++) {
        recv_worker_contribution(worker_fds[rank], &prefill, &group);
    }
    check(ds4_glm52_tp4_rank_group_command_done(&group),
          "prefill command should complete");
    assert_complete_step(&prefill, 4, 4);
    const uint64_t rank_local_prefill_hidden =
        prefill.ranks[0].hidden_checksum;
    check(ds4_glm52_mock_tp4_collective_step(models,
                                             &prefill,
                                             prompt[(sizeof(prompt) /
                                                     sizeof(prompt[0])) - 1u],
                                             sessions,
                                             err,
                                             sizeof(err)),
          "coordinator should run process-level DCP/all-reduce prefill");
    check(sessions[0].hidden_checksum != rank_local_prefill_hidden,
          "process prefill should publish collective hidden, not rank-local hidden");
    for (int rank = 1; rank < DS4_GLM52_L0_RANK_COUNT; rank++) {
        check(sessions[rank].hidden_checksum == sessions[0].hidden_checksum,
              "process prefill hidden should be replicated");
        send_worker_session_state(worker_fds[rank], &sessions[rank]);
    }

    const int gathered_token = prefill.coordinator_token;
    const int decode_tokens[] = {gathered_token};
    check(ds4_glm52_tp4_rank_group_broadcast(&group,
                                             0,
                                             DS4_GLM52_TP4_COMMAND_DECODE,
                                             &seq,
                                             err,
                                             sizeof(err)),
          "coordinator should broadcast decode");
    for (int rank = 1; rank < DS4_GLM52_L0_RANK_COUNT; rank++) {
        send_command_and_tokens(worker_fds[rank],
                                DS4_GLM52_TP4_COMMAND_DECODE,
                                seq,
                                decode_tokens,
                                1);
    }
    check(ds4_glm52_mock_decode(&models[0],
                                &sessions[0],
                                gathered_token,
                                NULL,
                                0,
                                err,
                                sizeof(err)),
          "coordinator mock decode should run");
    ds4_glm52_mock_tp4_step decode = {0};
    check(ds4_glm52_mock_tp4_step_add_rank(&decode,
                                           &models[0],
                                           &sessions[0],
                                           DS4_GLM52_TP4_COMMAND_DECODE,
                                           seq,
                                           err,
                                           sizeof(err)),
          "coordinator should add local decode contribution");
    check(ds4_glm52_tp4_rank_group_ack(&group,
                                       0,
                                       DS4_GLM52_TP4_COMMAND_DECODE,
                                       seq,
                                       err,
                                       sizeof(err)),
          "coordinator should ack local decode");
    for (int rank = 1; rank < DS4_GLM52_L0_RANK_COUNT; rank++) {
        recv_worker_contribution(worker_fds[rank], &decode, &group);
    }
    check(ds4_glm52_tp4_rank_group_command_done(&group),
          "decode command should complete");
    assert_complete_step(&decode, 5, 5);
    const uint64_t rank_local_decode_hidden = decode.ranks[0].hidden_checksum;
    check(ds4_glm52_mock_tp4_collective_step(models,
                                             &decode,
                                             gathered_token,
                                             sessions,
                                             err,
                                             sizeof(err)),
          "coordinator should run process-level DCP/all-reduce decode");
    check(sessions[0].hidden_checksum != rank_local_decode_hidden,
          "process decode should publish collective hidden, not rank-local hidden");
    for (int rank = 1; rank < DS4_GLM52_L0_RANK_COUNT; rank++) {
        check(sessions[rank].hidden_checksum == sessions[0].hidden_checksum,
              "process decode hidden should be replicated");
        send_worker_session_state(worker_fds[rank], &sessions[rank]);
    }

    check(ds4_glm52_tp4_rank_group_broadcast(&group,
                                             0,
                                             DS4_GLM52_TP4_COMMAND_SHUTDOWN,
                                             &seq,
                                             err,
                                             sizeof(err)),
          "coordinator should broadcast shutdown");
    for (int rank = 1; rank < DS4_GLM52_L0_RANK_COUNT; rank++) {
        check(ds4_glm52_tp4_transport_send_command(worker_fds[rank],
                                                   DS4_GLM52_TP4_COMMAND_SHUTDOWN,
                                                   seq,
                                                   err,
                                                   sizeof(err)),
              "coordinator should send shutdown");
    }
    check(ds4_glm52_tp4_rank_group_ack(&group,
                                       0,
                                       DS4_GLM52_TP4_COMMAND_SHUTDOWN,
                                       seq,
                                       err,
                                       sizeof(err)),
          "coordinator should ack local shutdown");
    for (int rank = 1; rank < DS4_GLM52_L0_RANK_COUNT; rank++) {
        check(ds4_glm52_tp4_transport_recv_ack_and_record(worker_fds[rank],
                                                          &group,
                                                          err,
                                                          sizeof(err)),
              "coordinator should receive shutdown ack");
        close_checked(worker_fds[rank]);
        int status = 0;
        check(waitpid(pids[rank], &status, 0) == pids[rank],
              "worker waitpid should succeed");
        check(WIFEXITED(status) && WEXITSTATUS(status) == 0,
              "worker should exit cleanly");
    }
    check(ds4_glm52_tp4_rank_group_command_done(&group),
          "shutdown command should complete");
    close_checked(listen_fd);
}

int main(void) {
    test_tcp_process_prefill_decode_mock();
    puts("test_glm52_tp4_mock_process: ok");
    return 0;
}
