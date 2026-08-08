#include "ds4_glm52_l0.h"

#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

typedef enum {
    ROLE_NONE = 0,
    ROLE_COORDINATOR,
    ROLE_WORKER,
} smoke_role;

typedef struct {
    smoke_role role;
    int rank;
    const char *listen;
    const char *connect;
    int timeout_ms;
    uint64_t model_hash;
    uint64_t config_hash;
    uint64_t plan_hash;
} smoke_config;

static void usage(FILE *fp, const char *prog) {
    fprintf(fp,
            "usage:\n"
            "  %s --role coordinator --rank 0 --listen HOST:PORT [hashes]\n"
            "  %s --role worker --rank N --connect HOST:PORT [hashes]\n"
            "\n"
            "options:\n"
            "  --role coordinator|worker\n"
            "  --rank N                 TP rank 0..3\n"
            "  --listen HOST:PORT       coordinator listen endpoint\n"
            "  --connect HOST:PORT      worker connect endpoint\n"
            "  --timeout-ms N           default 10000\n"
            "  --model-hash U64         default 11\n"
            "  --config-hash U64        default 22\n"
            "  --plan-hash U64          default 33\n",
            prog, prog);
}

static bool parse_u64_arg(const char *s, uint64_t *out) {
    if (!s || !s[0] || !out) return false;
    char *end = NULL;
    errno = 0;
    unsigned long long v = strtoull(s, &end, 10);
    if (errno || !end || *end) return false;
    *out = (uint64_t)v;
    return true;
}

static bool parse_int_arg(const char *s, int *out) {
    uint64_t v = 0;
    if (!parse_u64_arg(s, &v) || v > (uint64_t)INT32_MAX) return false;
    *out = (int)v;
    return true;
}

static bool endpoint_is_management(const ds4_glm52_tp4_tcp_endpoint *endpoint) {
    return endpoint &&
           !strncmp(endpoint->host, "192.168.0.", strlen("192.168.0."));
}

static bool parse_args(int argc, char **argv, smoke_config *cfg) {
    memset(cfg, 0, sizeof(*cfg));
    cfg->role = ROLE_NONE;
    cfg->rank = -1;
    cfg->timeout_ms = 10000;
    cfg->model_hash = 11;
    cfg->config_hash = 22;
    cfg->plan_hash = 33;

    for (int i = 1; i < argc; i++) {
        const char *arg = argv[i];
        const char *value = NULL;
        if (!strcmp(arg, "--role") ||
            !strcmp(arg, "--rank") ||
            !strcmp(arg, "--listen") ||
            !strcmp(arg, "--connect") ||
            !strcmp(arg, "--timeout-ms") ||
            !strcmp(arg, "--model-hash") ||
            !strcmp(arg, "--config-hash") ||
            !strcmp(arg, "--plan-hash")) {
            if (++i >= argc) return false;
            value = argv[i];
        }

        if (!strcmp(arg, "--role")) {
            if (!strcmp(value, "coordinator")) {
                cfg->role = ROLE_COORDINATOR;
            } else if (!strcmp(value, "worker")) {
                cfg->role = ROLE_WORKER;
            } else {
                return false;
            }
        } else if (!strcmp(arg, "--rank")) {
            if (!parse_int_arg(value, &cfg->rank)) return false;
        } else if (!strcmp(arg, "--listen")) {
            cfg->listen = value;
        } else if (!strcmp(arg, "--connect")) {
            cfg->connect = value;
        } else if (!strcmp(arg, "--timeout-ms")) {
            if (!parse_int_arg(value, &cfg->timeout_ms)) return false;
        } else if (!strcmp(arg, "--model-hash")) {
            if (!parse_u64_arg(value, &cfg->model_hash)) return false;
        } else if (!strcmp(arg, "--config-hash")) {
            if (!parse_u64_arg(value, &cfg->config_hash)) return false;
        } else if (!strcmp(arg, "--plan-hash")) {
            if (!parse_u64_arg(value, &cfg->plan_hash)) return false;
        } else {
            return false;
        }
    }

    if (cfg->rank < 0 || cfg->rank >= DS4_GLM52_L0_RANK_COUNT) return false;
    if (cfg->role == ROLE_COORDINATOR) {
        return cfg->rank == 0 && cfg->listen && !cfg->connect;
    }
    if (cfg->role == ROLE_WORKER) {
        return cfg->rank > 0 && cfg->connect && !cfg->listen;
    }
    return false;
}

static int rank_from_new_mask(uint32_t before, uint32_t after) {
    uint32_t diff = after & ~before;
    for (int rank = 0; rank < DS4_GLM52_L0_RANK_COUNT; rank++) {
        if (diff == (UINT32_C(1) << rank)) return rank;
    }
    return -1;
}

static int run_coordinator(const smoke_config *cfg) {
    char err[256] = {0};
    ds4_glm52_tp4_tcp_endpoint endpoint;
    if (!ds4_glm52_tp4_tcp_parse_endpoint(
                cfg->listen, &endpoint, err, sizeof(err))) {
        fprintf(stderr, "coordinator: bad listen endpoint: %s\n", err);
        return 2;
    }
    if (endpoint_is_management(&endpoint)) {
        fprintf(stderr,
                "coordinator: refusing management endpoint %s; use CRS812 fabric\n",
                endpoint.host);
        return 2;
    }

    int listen_fd = -1;
    uint16_t bound_port = 0;
    if (!ds4_glm52_tp4_tcp_listen(
                &endpoint, &listen_fd, &bound_port, err, sizeof(err))) {
        fprintf(stderr, "coordinator: listen failed: %s\n", err);
        return 3;
    }
    printf("coordinator: listening %s:%u\n", endpoint.host, bound_port);
    fflush(stdout);

    ds4_glm52_tp4_rank_group group;
    ds4_glm52_tp4_rank_group_init(&group,
                                  cfg->model_hash,
                                  cfg->config_hash,
                                  cfg->plan_hash);
    if (!ds4_glm52_tp4_rank_group_register(
                &group,
                0,
                DS4_GLM52_L0_TP_SIZE,
                DS4_GLM52_L0_DCP_SIZE,
                cfg->model_hash,
                cfg->config_hash,
                cfg->plan_hash,
                err,
                sizeof(err))) {
        fprintf(stderr, "coordinator: local rank register failed: %s\n", err);
        close(listen_fd);
        return 4;
    }

    int fds[DS4_GLM52_L0_RANK_COUNT];
    for (int i = 0; i < DS4_GLM52_L0_RANK_COUNT; i++) fds[i] = -1;
    for (int i = 1; i < DS4_GLM52_L0_RANK_COUNT; i++) {
        int fd = -1;
        if (!ds4_glm52_tp4_tcp_accept(
                    listen_fd, cfg->timeout_ms, &fd, err, sizeof(err))) {
            fprintf(stderr, "coordinator: accept failed: %s\n", err);
            close(listen_fd);
            return 5;
        }
        uint32_t before = group.registered_mask;
        if (!ds4_glm52_tp4_transport_recv_hello_and_register(
                    fd, &group, err, sizeof(err))) {
            fprintf(stderr, "coordinator: hello failed: %s\n", err);
            close(fd);
            close(listen_fd);
            return 6;
        }
        int rank = rank_from_new_mask(before, group.registered_mask);
        if (rank <= 0 || rank >= DS4_GLM52_L0_RANK_COUNT || fds[rank] >= 0) {
            fprintf(stderr, "coordinator: invalid new rank registration\n");
            close(fd);
            close(listen_fd);
            return 7;
        }
        fds[rank] = fd;
        printf("coordinator: registered rank %d\n", rank);
        fflush(stdout);
    }

    if (!ds4_glm52_tp4_rank_group_ready(&group)) {
        fprintf(stderr, "coordinator: rank group did not become ready\n");
        close(listen_fd);
        return 8;
    }

    ds4_glm52_l0_state state;
    memset(&state, 0, sizeof(state));
    if (!ds4_glm52_tp4_rank_group_publish_fabric(
                &group, 0, &state, err, sizeof(err))) {
        fprintf(stderr, "coordinator: publish fabric failed: %s\n", err);
        close(listen_fd);
        return 9;
    }
    printf("coordinator: fabric ready tp=%d dcp=%d ranks=%d\n",
           state.tp_fabric.tp_size,
           state.tp_fabric.dcp_size,
           state.tp_fabric.rank_count);

    uint64_t seq = 0;
    if (!ds4_glm52_tp4_rank_group_broadcast(
                &group,
                0,
                DS4_GLM52_TP4_COMMAND_SHUTDOWN,
                &seq,
                err,
                sizeof(err))) {
        fprintf(stderr, "coordinator: broadcast failed: %s\n", err);
        close(listen_fd);
        return 10;
    }
    for (int rank = 1; rank < DS4_GLM52_L0_RANK_COUNT; rank++) {
        if (!ds4_glm52_tp4_transport_send_command(
                    fds[rank],
                    DS4_GLM52_TP4_COMMAND_SHUTDOWN,
                    seq,
                    err,
                    sizeof(err))) {
            fprintf(stderr, "coordinator: send command rank %d failed: %s\n",
                    rank, err);
            close(listen_fd);
            return 11;
        }
    }
    if (!ds4_glm52_tp4_rank_group_ack(
                &group,
                0,
                DS4_GLM52_TP4_COMMAND_SHUTDOWN,
                seq,
                err,
                sizeof(err))) {
        fprintf(stderr, "coordinator: local ack failed: %s\n", err);
        close(listen_fd);
        return 12;
    }
    for (int rank = 1; rank < DS4_GLM52_L0_RANK_COUNT; rank++) {
        if (!ds4_glm52_tp4_transport_recv_ack_and_record(
                    fds[rank], &group, err, sizeof(err))) {
            fprintf(stderr, "coordinator: recv ack rank %d failed: %s\n",
                    rank, err);
            close(listen_fd);
            return 13;
        }
        printf("coordinator: ack rank %d\n", rank);
        close(fds[rank]);
    }
    close(listen_fd);

    if (!ds4_glm52_tp4_rank_group_command_done(&group)) {
        fprintf(stderr, "coordinator: shutdown command incomplete\n");
        return 14;
    }
    printf("coordinator: shutdown complete seq=%llu ack_mask=0x%x\n",
           (unsigned long long)seq,
           group.ack_mask);
    return 0;
}

static int run_worker(const smoke_config *cfg) {
    char err[256] = {0};
    ds4_glm52_tp4_tcp_endpoint endpoint;
    if (!ds4_glm52_tp4_tcp_parse_endpoint(
                cfg->connect, &endpoint, err, sizeof(err))) {
        fprintf(stderr, "worker%d: bad connect endpoint: %s\n", cfg->rank, err);
        return 2;
    }
    if (endpoint_is_management(&endpoint)) {
        fprintf(stderr,
                "worker%d: refusing management endpoint %s; use CRS812 fabric\n",
                cfg->rank,
                endpoint.host);
        return 2;
    }

    int fd = -1;
    if (!ds4_glm52_tp4_tcp_connect(
                &endpoint, cfg->timeout_ms, &fd, err, sizeof(err))) {
        fprintf(stderr, "worker%d: connect failed: %s\n", cfg->rank, err);
        return 3;
    }

    ds4_glm52_tp4_transport_hello hello = {
        .rank = cfg->rank,
        .tp_size = DS4_GLM52_L0_TP_SIZE,
        .dcp_size = DS4_GLM52_L0_DCP_SIZE,
        .model_hash = cfg->model_hash,
        .config_hash = cfg->config_hash,
        .plan_hash = cfg->plan_hash,
    };
    if (!ds4_glm52_tp4_transport_send_hello(fd, &hello, err, sizeof(err))) {
        fprintf(stderr, "worker%d: send hello failed: %s\n", cfg->rank, err);
        close(fd);
        return 4;
    }

    ds4_glm52_tp4_command command = DS4_GLM52_TP4_COMMAND_NONE;
    uint64_t seq = 0;
    if (!ds4_glm52_tp4_transport_recv_command(
                fd, &command, &seq, err, sizeof(err))) {
        fprintf(stderr, "worker%d: recv command failed: %s\n", cfg->rank, err);
        close(fd);
        return 5;
    }
    printf("worker%d: command %s seq=%llu\n",
           cfg->rank,
           ds4_glm52_tp4_command_name(command),
           (unsigned long long)seq);
    if (!ds4_glm52_tp4_transport_send_ack(
                fd, cfg->rank, command, seq, err, sizeof(err))) {
        fprintf(stderr, "worker%d: send ack failed: %s\n", cfg->rank, err);
        close(fd);
        return 6;
    }
    close(fd);
    printf("worker%d: ack sent\n", cfg->rank);
    return 0;
}

int main(int argc, char **argv) {
    smoke_config cfg;
    if (!parse_args(argc, argv, &cfg)) {
        usage(stderr, argv[0]);
        return 1;
    }
    if (cfg.role == ROLE_COORDINATOR) return run_coordinator(&cfg);
    if (cfg.role == ROLE_WORKER) return run_worker(&cfg);
    usage(stderr, argv[0]);
    return 1;
}
