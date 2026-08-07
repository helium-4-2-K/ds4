#include "ds4_glm52_l0.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>

static void fail(const char *msg) {
    fprintf(stderr, "test_tp4_rank_group: %s\n", msg);
    exit(1);
}

static void check(bool ok, const char *msg) {
    if (!ok) fail(msg);
}

static void close_checked(int fd) {
    if (fd >= 0) close(fd);
}

static ds4_glm52_l0_config valid_cfg(void) {
    ds4_glm52_l0_config cfg = {
        .enabled = true,
        .rank_set = true,
        .rank = 0,
        .tp_size = DS4_GLM52_L0_TP_SIZE,
        .dcp_size = DS4_GLM52_L0_DCP_SIZE,
        .pp_size = DS4_GLM52_L0_PP_SIZE,
        .model_root = "/models/glm-5.2",
        .rank_plan = "/plans/rank0.plan",
        .fabric_addr = "10.100.185.3",
    };
    return cfg;
}

static void mark_prefill_inputs_except_fabric(ds4_glm52_l0_state *state) {
    memset(state, 0, sizeof(*state));
    state->model_plan.validated = true;
    state->shard_manifest.validated = true;
    state->resident_shards.rank = 0;
    state->resident_shards.base_shard_count =
        DS4_GLM52_L0_EXPECTED_BASE_SHARDS;
    state->resident_shards.mtp_shard_count =
        DS4_GLM52_L0_EXPECTED_MTP_SHARDS;
    state->resident_shards.mapped_bytes = 21;
    state->resident_shards.no_foreign_rank_shard = true;
    state->resident_shards.mapped = true;
    state->rank_plan.rank = 0;
    state->rank_plan.q_head_start = 0;
    state->rank_plan.q_head_end = DS4_GLM52_L0_Q_HEADS_PER_RANK;
    state->rank_plan.dcp_rank = 0;
    state->rank_plan.bound = true;
    state->request.session_id = "session";
    state->request.accepted = true;
}

static ds4_glm52_tp4_rank_group ready_group(void) {
    ds4_glm52_tp4_rank_group group;
    char err[160] = {0};
    ds4_glm52_tp4_rank_group_init(&group, 11, 22, 33);
    for (int rank = 0; rank < DS4_GLM52_L0_RANK_COUNT; rank++) {
        check(ds4_glm52_tp4_rank_group_register(
                  &group,
                  rank,
                  DS4_GLM52_L0_TP_SIZE,
                  DS4_GLM52_L0_DCP_SIZE,
                  11,
                  22,
                  33,
                  err,
                  sizeof(err)),
              "rank registration should succeed");
    }
    check(ds4_glm52_tp4_rank_group_ready(&group),
          "all four ranks should make the group ready");
    return group;
}

static void worker_rank_main(int fd, int rank) {
    char err[160] = {0};
    ds4_glm52_tp4_transport_hello hello = {
        .rank = rank,
        .tp_size = DS4_GLM52_L0_TP_SIZE,
        .dcp_size = DS4_GLM52_L0_DCP_SIZE,
        .model_hash = 11,
        .config_hash = 22,
        .plan_hash = 33,
    };
    if (!ds4_glm52_tp4_transport_send_hello(
                fd, &hello, err, sizeof(err))) {
        fprintf(stderr, "worker %d send hello failed: %s\n", rank, err);
        _exit(2);
    }

    ds4_glm52_tp4_command command = DS4_GLM52_TP4_COMMAND_NONE;
    uint64_t seq = 0;
    if (!ds4_glm52_tp4_transport_recv_command(
                fd, &command, &seq, err, sizeof(err))) {
        fprintf(stderr, "worker %d recv command failed: %s\n", rank, err);
        _exit(3);
    }
    if (command != DS4_GLM52_TP4_COMMAND_SHUTDOWN || seq != 1) {
        fprintf(stderr, "worker %d saw wrong command\n", rank);
        _exit(4);
    }
    if (!ds4_glm52_tp4_transport_send_ack(
                fd, rank, command, seq, err, sizeof(err))) {
        fprintf(stderr, "worker %d send ack failed: %s\n", rank, err);
        _exit(5);
    }
    close_checked(fd);
    _exit(0);
}

static void tcp_worker_rank_main(uint16_t port, int rank) {
    char err[160] = {0};
    ds4_glm52_tp4_tcp_endpoint endpoint = {
        .host = "127.0.0.1",
        .port = port,
    };
    int fd = -1;
    if (!ds4_glm52_tp4_tcp_connect(&endpoint, 1000, &fd, err, sizeof(err))) {
        fprintf(stderr, "tcp worker %d connect failed: %s\n", rank, err);
        _exit(10);
    }
    worker_rank_main(fd, rank);
}

static void test_registration_happy_path_and_missing_rank(void) {
    ds4_glm52_tp4_rank_group group;
    char err[160] = {0};
    ds4_glm52_tp4_rank_group_init(&group, 11, 22, 33);
    for (int rank = 0; rank < DS4_GLM52_L0_RANK_COUNT - 1; rank++) {
        check(ds4_glm52_tp4_rank_group_register(
                  &group,
                  rank,
                  DS4_GLM52_L0_TP_SIZE,
                  DS4_GLM52_L0_DCP_SIZE,
                  11,
                  22,
                  33,
                  err,
                  sizeof(err)),
              "partial rank registration should succeed");
    }
    check(!ds4_glm52_tp4_rank_group_ready(&group),
          "missing rank should keep group unready");

    check(ds4_glm52_tp4_rank_group_register(
              &group,
              3,
              DS4_GLM52_L0_TP_SIZE,
              DS4_GLM52_L0_DCP_SIZE,
              11,
              22,
              33,
              err,
              sizeof(err)),
          "final rank registration should succeed");
    check(ds4_glm52_tp4_rank_group_ready(&group),
          "complete rank set should make group ready");
}

static void test_registration_rejections(void) {
    ds4_glm52_tp4_rank_group group;
    char err[160] = {0};
    ds4_glm52_tp4_rank_group_init(&group, 11, 22, 33);
    check(ds4_glm52_tp4_rank_group_register(
              &group, 0, 4, 4, 11, 22, 33, err, sizeof(err)),
          "rank 0 should register");
    check(!ds4_glm52_tp4_rank_group_register(
              &group, 0, 4, 4, 11, 22, 33, err, sizeof(err)),
          "duplicate rank should fail");
    check(strstr(err, "duplicate") != NULL,
          "duplicate rejection should be explicit");

    ds4_glm52_tp4_rank_group_init(&group, 11, 22, 33);
    check(!ds4_glm52_tp4_rank_group_register(
              &group, 4, 4, 4, 11, 22, 33, err, sizeof(err)),
          "out-of-range rank should fail");
    check(!ds4_glm52_tp4_rank_group_register(
              &group, 1, 2, 4, 11, 22, 33, err, sizeof(err)),
          "wrong tp size should fail");
    check(!ds4_glm52_tp4_rank_group_register(
              &group, 1, 4, 2, 11, 22, 33, err, sizeof(err)),
          "wrong dcp size should fail");
    check(!ds4_glm52_tp4_rank_group_register(
              &group, 1, 4, 4, 99, 22, 33, err, sizeof(err)),
          "model hash mismatch should fail");
    check(!ds4_glm52_tp4_rank_group_register(
              &group, 1, 4, 4, 11, 99, 33, err, sizeof(err)),
          "config hash mismatch should fail");
    check(!ds4_glm52_tp4_rank_group_register(
              &group, 1, 4, 4, 11, 22, 99, err, sizeof(err)),
          "plan hash mismatch should fail");
}

static void test_command_broadcast_and_ack(void) {
    ds4_glm52_tp4_rank_group group = ready_group();
    char err[160] = {0};
    uint64_t seq = 0;
    check(!ds4_glm52_tp4_rank_group_broadcast(
              &group,
              1,
              DS4_GLM52_TP4_COMMAND_PREFILL,
              &seq,
              err,
              sizeof(err)),
          "non-coordinator broadcast should fail");
    check(strstr(err, "rank 0") != NULL,
          "broadcast failure should name rank 0 authority");

    check(ds4_glm52_tp4_rank_group_broadcast(
              &group,
              0,
              DS4_GLM52_TP4_COMMAND_PREFILL,
              &seq,
              err,
              sizeof(err)),
          "rank 0 broadcast should succeed");
    check(seq == 1, "first command seq should be 1");
    check(!ds4_glm52_tp4_rank_group_command_done(&group),
          "command should wait for all rank acks");

    check(!ds4_glm52_tp4_rank_group_ack(
              &group,
              0,
              DS4_GLM52_TP4_COMMAND_DECODE,
              seq,
              err,
              sizeof(err)),
          "wrong command ack should fail");
    check(!ds4_glm52_tp4_rank_group_ack(
              &group,
              0,
              DS4_GLM52_TP4_COMMAND_PREFILL,
              seq + 1,
              err,
              sizeof(err)),
          "wrong seq ack should fail");

    for (int rank = 0; rank < DS4_GLM52_L0_RANK_COUNT; rank++) {
        check(ds4_glm52_tp4_rank_group_ack(
                  &group,
                  rank,
                  DS4_GLM52_TP4_COMMAND_PREFILL,
                  seq,
                  err,
                  sizeof(err)),
              "matching rank ack should succeed");
    }
    check(ds4_glm52_tp4_rank_group_command_done(&group),
          "all rank acks should complete command");

    check(ds4_glm52_tp4_rank_group_broadcast(
              &group,
              0,
              DS4_GLM52_TP4_COMMAND_SHUTDOWN,
              &seq,
              err,
              sizeof(err)),
          "shutdown broadcast should succeed");
    check(seq == 2, "second command seq should be 2");
    check(!ds4_glm52_tp4_rank_group_command_done(&group),
          "new command should clear prior acks");
}

static void test_publish_fabric_gates_prefill(void) {
    ds4_glm52_l0_config cfg = valid_cfg();
    ds4_glm52_l0_state state;
    ds4_glm52_l0_result result;
    char err[160] = {0};
    mark_prefill_inputs_except_fabric(&state);
    ds4_glm52_l0_status status =
        ds4_glm52_l0_stub_action(DS4_GLM52_L0_ACTION_PREFILL,
                                 &cfg,
                                 &state,
                                 &result);
    check(status == DS4_GLM52_L0_STATUS_NOT_READY,
          "prefill should wait for fabric before publish");
    check(strstr(result.message, "collective fabric") != NULL,
          "prefill block should name fabric");

    ds4_glm52_tp4_rank_group group = ready_group();
    check(ds4_glm52_tp4_rank_group_publish_fabric(
              &group, 0, &state, err, sizeof(err)),
          "ready rank group should publish fabric state");
    status = ds4_glm52_l0_stub_action(DS4_GLM52_L0_ACTION_PREFILL,
                                      &cfg,
                                      &state,
                                      &result);
    check(status == DS4_GLM52_L0_STATUS_NOT_READY,
          "prefill should wait for a prompt token span after fabric publish");
    check(strstr(result.message, "prefill/a-tokenize-prompt") != NULL,
          "prefill should reach the tokenize child action after fabric publish");
    check(strstr(result.message, "prompt token span") != NULL,
          "prefill should name the missing prompt span as the next prerequisite");

    memset(&group, 0, sizeof(group));
    mark_prefill_inputs_except_fabric(&state);
    check(!ds4_glm52_tp4_rank_group_publish_fabric(
              &group, 0, &state, err, sizeof(err)),
          "unready rank group must not publish fabric");
}

static void test_transport_loss_clears_readiness(void) {
    ds4_glm52_tp4_rank_group group = ready_group();
    char err[160] = {0};
    check(ds4_glm52_tp4_rank_group_transport_failed(
              &group, 2, err, sizeof(err)),
          "transport loss should record successfully");
    check(!ds4_glm52_tp4_rank_group_ready(&group),
          "transport loss should clear group readiness");
    check(group.registered_count == DS4_GLM52_L0_RANK_COUNT - 1,
          "transport loss should remove failed rank registration");
    check(!ds4_glm52_tp4_rank_group_publish_fabric(
              &group, 0, &(ds4_glm52_l0_state){0}, err, sizeof(err)),
          "unready group after loss must not publish fabric");

    check(ds4_glm52_tp4_rank_group_register(
              &group,
              2,
              DS4_GLM52_L0_TP_SIZE,
              DS4_GLM52_L0_DCP_SIZE,
              11,
              22,
              33,
              err,
              sizeof(err)),
          "failed rank should be able to re-register");
    check(ds4_glm52_tp4_rank_group_ready(&group),
          "re-registration should restore readiness");
}

static void test_forked_fd_transport_handshake_and_shutdown(void) {
    int fds[DS4_GLM52_L0_RANK_COUNT][2];
    pid_t pids[DS4_GLM52_L0_RANK_COUNT];
    memset(fds, -1, sizeof(fds));
    memset(pids, 0, sizeof(pids));

    for (int rank = 1; rank < DS4_GLM52_L0_RANK_COUNT; rank++) {
        check(socketpair(AF_UNIX, SOCK_STREAM, 0, fds[rank]) == 0,
              "socketpair should succeed");
        pid_t pid = fork();
        check(pid >= 0, "fork should succeed");
        if (pid == 0) {
            close_checked(fds[rank][0]);
            worker_rank_main(fds[rank][1], rank);
        }
        pids[rank] = pid;
        close_checked(fds[rank][1]);
        fds[rank][1] = -1;
    }

    char err[160] = {0};
    ds4_glm52_tp4_rank_group group;
    ds4_glm52_tp4_rank_group_init(&group, 11, 22, 33);
    check(ds4_glm52_tp4_rank_group_register(
              &group,
              0,
              DS4_GLM52_L0_TP_SIZE,
              DS4_GLM52_L0_DCP_SIZE,
              11,
              22,
              33,
              err,
              sizeof(err)),
          "coordinator rank should register locally");
    for (int rank = 1; rank < DS4_GLM52_L0_RANK_COUNT; rank++) {
        check(ds4_glm52_tp4_transport_recv_hello_and_register(
                  fds[rank][0], &group, err, sizeof(err)),
              "worker hello should register over fd transport");
    }
    check(ds4_glm52_tp4_rank_group_ready(&group),
          "fd transport hellos should make group ready");

    uint64_t seq = 0;
    check(ds4_glm52_tp4_rank_group_broadcast(
              &group,
              0,
              DS4_GLM52_TP4_COMMAND_SHUTDOWN,
              &seq,
              err,
              sizeof(err)),
          "coordinator broadcast should update command state");
    check(seq == 1, "transport command seq should be 1");
    for (int rank = 1; rank < DS4_GLM52_L0_RANK_COUNT; rank++) {
        check(ds4_glm52_tp4_transport_send_command(
                  fds[rank][0],
                  DS4_GLM52_TP4_COMMAND_SHUTDOWN,
                  seq,
                  err,
                  sizeof(err)),
              "coordinator should send command over fd transport");
    }
    check(ds4_glm52_tp4_rank_group_ack(
              &group,
              0,
              DS4_GLM52_TP4_COMMAND_SHUTDOWN,
              seq,
              err,
              sizeof(err)),
          "coordinator rank ack should record locally");
    for (int rank = 1; rank < DS4_GLM52_L0_RANK_COUNT; rank++) {
        check(ds4_glm52_tp4_transport_recv_ack_and_record(
                  fds[rank][0], &group, err, sizeof(err)),
              "worker ack should record over fd transport");
    }
    check(ds4_glm52_tp4_rank_group_command_done(&group),
          "all process-bound ranks should complete shutdown command");

    for (int rank = 1; rank < DS4_GLM52_L0_RANK_COUNT; rank++) {
        close_checked(fds[rank][0]);
        int status = 0;
        check(waitpid(pids[rank], &status, 0) == pids[rank],
              "worker waitpid should succeed");
        check(WIFEXITED(status) && WEXITSTATUS(status) == 0,
              "worker should exit cleanly");
    }
}

static void test_fd_transport_rejections(void) {
    int fds[2] = {-1, -1};
    char err[160] = {0};
    ds4_glm52_tp4_rank_group group;

    check(socketpair(AF_UNIX, SOCK_STREAM, 0, fds) == 0,
          "socketpair should succeed");
    ds4_glm52_tp4_transport_hello wrong_hash = {
        .rank = 1,
        .tp_size = DS4_GLM52_L0_TP_SIZE,
        .dcp_size = DS4_GLM52_L0_DCP_SIZE,
        .model_hash = 99,
        .config_hash = 22,
        .plan_hash = 33,
    };
    ds4_glm52_tp4_rank_group_init(&group, 11, 22, 33);
    check(ds4_glm52_tp4_transport_send_hello(
              fds[1], &wrong_hash, err, sizeof(err)),
          "wrong-hash hello should still write as a frame");
    check(!ds4_glm52_tp4_transport_recv_hello_and_register(
              fds[0], &group, err, sizeof(err)),
          "wrong-hash hello should fail registration");
    check(strstr(err, "hash mismatch") != NULL,
          "wrong-hash rejection should name hash mismatch");
    close_checked(fds[0]);
    close_checked(fds[1]);

    fds[0] = -1;
    fds[1] = -1;
    check(socketpair(AF_UNIX, SOCK_STREAM, 0, fds) == 0,
          "socketpair should succeed");
    ds4_glm52_tp4_rank_group_init(&group, 11, 22, 33);
    check(ds4_glm52_tp4_transport_send_command(
              fds[1],
              DS4_GLM52_TP4_COMMAND_SHUTDOWN,
              1,
              err,
              sizeof(err)),
          "command frame should write");
    check(!ds4_glm52_tp4_transport_recv_hello_and_register(
              fds[0], &group, err, sizeof(err)),
          "command frame must not be accepted as hello");
    check(strstr(err, "unexpected frame") != NULL,
          "wrong frame type rejection should be explicit");
    close_checked(fds[0]);
    close_checked(fds[1]);

    fds[0] = -1;
    fds[1] = -1;
    check(socketpair(AF_UNIX, SOCK_STREAM, 0, fds) == 0,
          "socketpair should succeed");
    const char short_frame[] = "bad";
    check(write(fds[1], short_frame, sizeof(short_frame)) ==
              (ssize_t)sizeof(short_frame),
          "short frame write should succeed");
    close_checked(fds[1]);
    fds[1] = -1;
    ds4_glm52_tp4_rank_group_init(&group, 11, 22, 33);
    check(!ds4_glm52_tp4_transport_recv_hello_and_register(
              fds[0], &group, err, sizeof(err)),
          "truncated frame should fail");
    check(strstr(err, "header read failed") != NULL,
          "truncated frame rejection should name header read");
    close_checked(fds[0]);
}

static void test_tcp_endpoint_parsing_and_accept_timeout(void) {
    char err[160] = {0};
    ds4_glm52_tp4_tcp_endpoint endpoint;

    check(ds4_glm52_tp4_tcp_parse_endpoint(
              "127.0.0.1:0", &endpoint, err, sizeof(err)),
          "valid IPv4 endpoint should parse");
    check(!strcmp(endpoint.host, "127.0.0.1"),
          "parsed endpoint should preserve host");
    check(endpoint.port == 0, "parsed endpoint should preserve port");
    check(!ds4_glm52_tp4_tcp_parse_endpoint(
              "127.0.0.1", &endpoint, err, sizeof(err)),
          "endpoint without port should fail");
    check(!ds4_glm52_tp4_tcp_parse_endpoint(
              "127.0.0.1:notaport", &endpoint, err, sizeof(err)),
          "endpoint with invalid port should fail");

    int listen_fd = -1;
    uint16_t bound_port = 0;
    check(ds4_glm52_tp4_tcp_listen(
              &endpoint, &listen_fd, &bound_port, err, sizeof(err)),
          "loopback listener should bind");
    check(bound_port != 0, "ephemeral listener should report bound port");
    int accepted_fd = -1;
    check(!ds4_glm52_tp4_tcp_accept(
              listen_fd, 25, &accepted_fd, err, sizeof(err)),
          "accept with no worker should time out");
    check(strstr(err, "timed out") != NULL,
          "accept timeout should be explicit");
    check(accepted_fd == -1, "failed accept should not publish fd");
    close_checked(listen_fd);
}

static void test_tcp_loopback_rendezvous_and_shutdown(void) {
    char err[160] = {0};
    ds4_glm52_tp4_tcp_endpoint endpoint;
    check(ds4_glm52_tp4_tcp_parse_endpoint(
              "127.0.0.1:0", &endpoint, err, sizeof(err)),
          "loopback endpoint should parse");
    int listen_fd = -1;
    uint16_t bound_port = 0;
    check(ds4_glm52_tp4_tcp_listen(
              &endpoint, &listen_fd, &bound_port, err, sizeof(err)),
          "coordinator TCP listener should bind");

    pid_t pids[DS4_GLM52_L0_RANK_COUNT];
    int worker_fds[DS4_GLM52_L0_RANK_COUNT];
    memset(pids, 0, sizeof(pids));
    memset(worker_fds, -1, sizeof(worker_fds));
    for (int rank = 1; rank < DS4_GLM52_L0_RANK_COUNT; rank++) {
        pid_t pid = fork();
        check(pid >= 0, "tcp worker fork should succeed");
        if (pid == 0) {
            close_checked(listen_fd);
            tcp_worker_rank_main(bound_port, rank);
        }
        pids[rank] = pid;
    }

    ds4_glm52_tp4_rank_group group;
    ds4_glm52_tp4_rank_group_init(&group, 11, 22, 33);
    check(ds4_glm52_tp4_rank_group_register(
              &group,
              0,
              DS4_GLM52_L0_TP_SIZE,
              DS4_GLM52_L0_DCP_SIZE,
              11,
              22,
              33,
              err,
              sizeof(err)),
          "coordinator rank should register locally");
    for (int rank = 1; rank < DS4_GLM52_L0_RANK_COUNT; rank++) {
        check(ds4_glm52_tp4_tcp_accept(
                  listen_fd, 1000, &worker_fds[rank], err, sizeof(err)),
              "coordinator should accept worker TCP connection");
        check(ds4_glm52_tp4_transport_recv_hello_and_register(
                  worker_fds[rank], &group, err, sizeof(err)),
              "worker TCP hello should register");
    }
    check(ds4_glm52_tp4_rank_group_ready(&group),
          "TCP worker hellos should make group ready");

    uint64_t seq = 0;
    check(ds4_glm52_tp4_rank_group_broadcast(
              &group,
              0,
              DS4_GLM52_TP4_COMMAND_SHUTDOWN,
              &seq,
              err,
              sizeof(err)),
          "TCP coordinator broadcast should update command state");
    for (int rank = 1; rank < DS4_GLM52_L0_RANK_COUNT; rank++) {
        check(ds4_glm52_tp4_transport_send_command(
                  worker_fds[rank],
                  DS4_GLM52_TP4_COMMAND_SHUTDOWN,
                  seq,
                  err,
                  sizeof(err)),
              "TCP coordinator should send command");
    }
    check(ds4_glm52_tp4_rank_group_ack(
              &group,
              0,
              DS4_GLM52_TP4_COMMAND_SHUTDOWN,
              seq,
              err,
              sizeof(err)),
          "coordinator ack should record");
    for (int rank = 1; rank < DS4_GLM52_L0_RANK_COUNT; rank++) {
        check(ds4_glm52_tp4_transport_recv_ack_and_record(
                  worker_fds[rank], &group, err, sizeof(err)),
              "TCP worker ack should record");
    }
    check(ds4_glm52_tp4_rank_group_command_done(&group),
          "TCP shutdown should complete on all ranks");

    for (int rank = 1; rank < DS4_GLM52_L0_RANK_COUNT; rank++) {
        close_checked(worker_fds[rank]);
        int status = 0;
        check(waitpid(pids[rank], &status, 0) == pids[rank],
              "tcp worker waitpid should succeed");
        check(WIFEXITED(status) && WEXITSTATUS(status) == 0,
              "tcp worker should exit cleanly");
    }
    close_checked(listen_fd);
}

int main(void) {
    test_registration_happy_path_and_missing_rank();
    test_registration_rejections();
    test_command_broadcast_and_ack();
    test_publish_fabric_gates_prefill();
    test_transport_loss_clears_readiness();
    test_forked_fd_transport_handshake_and_shutdown();
    test_fd_transport_rejections();
    test_tcp_endpoint_parsing_and_accept_timeout();
    test_tcp_loopback_rendezvous_and_shutdown();
    return 0;
}
