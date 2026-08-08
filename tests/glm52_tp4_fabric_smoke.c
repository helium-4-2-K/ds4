#include "ds4_glm52_l0.h"

#include <errno.h>
#include <math.h>
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
    int payload_floats;
    bool bad_payload_frame;
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
            "  --payload-floats N       optional fixed-size all-reduce payload\n"
            "  --bad-payload-frame      send a mismatched typed payload frame\n"
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

static bool write_exact_local(int fd, const void *buf, size_t len) {
    const char *p = (const char *)buf;
    while (len > 0) {
        ssize_t n = write(fd, p, len);
        if (n < 0) {
            if (errno == EINTR) continue;
            return false;
        }
        if (n == 0) return false;
        p += (size_t)n;
        len -= (size_t)n;
    }
    return true;
}

static bool read_exact_local(int fd, void *buf, size_t len) {
    char *p = (char *)buf;
    while (len > 0) {
        ssize_t n = read(fd, p, len);
        if (n < 0) {
            if (errno == EINTR) continue;
            return false;
        }
        if (n == 0) return false;
        p += (size_t)n;
        len -= (size_t)n;
    }
    return true;
}

static bool endpoint_is_management(const ds4_glm52_tp4_tcp_endpoint *endpoint) {
    return endpoint &&
           !strncmp(endpoint->host, "192.168.0.", strlen("192.168.0."));
}

static float partial_value(int rank, int index) {
    return (float)(rank + 1) * 100.0f + (float)index;
}

static float reduced_value(int index) {
    float sum = 0.0f;
    for (int rank = 0; rank < DS4_GLM52_L0_RANK_COUNT; rank++) {
        sum += partial_value(rank, index);
    }
    return sum;
}

static uint32_t full_rank_mask_local(void) {
    return (UINT32_C(1) << DS4_GLM52_L0_RANK_COUNT) - UINT32_C(1);
}

static uint64_t shape_hash_for_payload(int n) {
    uint64_t h = 1469598103934665603ull;
    h ^= (uint64_t)DS4_GLM52_TP4_COLLECTIVE_ATTN;
    h *= 1099511628211ull;
    h ^= (uint64_t)DS4_GLM52_TP4_TENSOR_DTYPE_F32;
    h *= 1099511628211ull;
    h ^= (uint64_t)(uint32_t)n;
    h *= 1099511628211ull;
    return h;
}

static uint64_t session_hash_for_smoke(const smoke_config *cfg) {
    return cfg->config_hash ^ (cfg->plan_hash << 1) ^ UINT64_C(0x9e3779b97f4a7c15);
}

static ds4_glm52_tp4_collective_request make_collective_request(
        const smoke_config *cfg,
        int rank,
        uint64_t seq) {
    ds4_glm52_tp4_collective_request req;
    memset(&req, 0, sizeof(req));
    req.frame_version = DS4_GLM52_TP4_COLLECTIVE_FRAME_VERSION;
    req.kind = DS4_GLM52_TP4_COLLECTIVE_ATTN;
    req.rank = rank;
    req.tp_size = DS4_GLM52_L0_TP_SIZE;
    req.dcp_size = DS4_GLM52_L0_DCP_SIZE;
    req.rank_count = DS4_GLM52_L0_RANK_COUNT;
    req.layer_index = 0;
    req.dtype = DS4_GLM52_TP4_TENSOR_DTYPE_F32;
    req.seq = seq;
    req.model_hash = cfg->model_hash;
    req.session_hash = session_hash_for_smoke(cfg);
    req.token_step_j = seq;
    req.element_count = (size_t)cfg->payload_floats;
    req.shape_hash = shape_hash_for_payload(cfg->payload_floats);
    req.byte_count =
        req.element_count * ds4_glm52_tp4_tensor_dtype_size(req.dtype);
    req.participant_mask = full_rank_mask_local();
    req.topology_ready = true;
    req.transport_ready = true;
    req.rank_local_partial_ready = true;
    req.replicated_output_ready = true;
    return req;
}

static bool collective_frames_match(
        const ds4_glm52_tp4_collective_request *a,
        const ds4_glm52_tp4_collective_request *b) {
    return a && b &&
           a->frame_version == b->frame_version &&
           a->kind == b->kind &&
           a->tp_size == b->tp_size &&
           a->dcp_size == b->dcp_size &&
           a->rank_count == b->rank_count &&
           a->layer_index == b->layer_index &&
           a->dtype == b->dtype &&
           a->seq == b->seq &&
           a->model_hash == b->model_hash &&
           a->session_hash == b->session_hash &&
           a->token_step_j == b->token_step_j &&
           a->element_count == b->element_count &&
           a->shape_hash == b->shape_hash &&
           a->byte_count == b->byte_count &&
           a->participant_mask == b->participant_mask &&
           a->topology_ready == b->topology_ready &&
           a->transport_ready == b->transport_ready &&
           a->rank_local_partial_ready == b->rank_local_partial_ready &&
           a->replicated_output_ready == b->replicated_output_ready;
}

static bool send_collective_frame(
        int fd,
        const ds4_glm52_tp4_collective_request *req,
        bool corrupt_byte_count,
        char *err,
        size_t err_size) {
    unsigned char wire[DS4_GLM52_TP4_COLLECTIVE_WIRE_SIZE];
    if (!ds4_glm52_tp4_collective_frame_encode(
                req, wire, sizeof(wire), err, err_size)) {
        return false;
    }
    if (corrupt_byte_count) {
        wire[87] ^= 1u;
    }
    if (!write_exact_local(fd, &wire, sizeof(wire))) {
        snprintf(err, err_size, "collective frame write failed");
        return false;
    }
    return true;
}

static bool recv_collective_frame(
        int fd,
        ds4_glm52_tp4_collective_request *req,
        char *err,
        size_t err_size) {
    unsigned char wire[DS4_GLM52_TP4_COLLECTIVE_WIRE_SIZE];
    if (!read_exact_local(fd, &wire, sizeof(wire))) {
        snprintf(err, err_size, "collective frame read failed");
        return false;
    }
    return ds4_glm52_tp4_collective_frame_decode(
            wire, sizeof(wire), req, err, err_size);
}

static bool payload_count_valid(int n) {
    return n >= 0 && n <= 4096;
}

static bool send_payload(int fd,
                         const ds4_glm52_tp4_collective_request *req,
                         int n,
                         bool corrupt_frame,
                         char *err,
                         size_t err_size) {
    if (!payload_count_valid(n)) {
        snprintf(err, err_size, "invalid payload float count");
        return false;
    }
    if (!send_collective_frame(fd, req, corrupt_frame, err, err_size)) {
        return false;
    }
    for (int i = 0; i < n; i++) {
        float value = partial_value(req->rank, i);
        if (!write_exact_local(fd, &value, sizeof(value))) {
            snprintf(err, err_size, "payload body write failed");
            return false;
        }
    }
    return true;
}

static bool recv_payload_partial(int fd,
                                 int expected_rank,
                                 const ds4_glm52_tp4_collective_request *expected,
                                 float *partial,
                                 char *err,
                                 size_t err_size) {
    ds4_glm52_tp4_collective_request actual;
    if (!recv_collective_frame(fd, &actual, err, err_size)) return false;
    if (actual.rank != expected_rank ||
        actual.element_count != expected->element_count ||
        !collective_frames_match(&actual, expected)) {
        snprintf(err, err_size, "payload collective frame mismatch");
        return false;
    }
    const int n = (int)actual.element_count;
    for (int i = 0; i < n; i++) {
        float value = 0.0f;
        if (!read_exact_local(fd, &value, sizeof(value))) {
            snprintf(err, err_size, "payload body read failed");
            return false;
        }
        if (!isfinite(value)) {
            snprintf(err, err_size, "payload value is non-finite");
            return false;
        }
        partial[i] = value;
    }
    return true;
}

static void free_rank_buffers(float *buffers[DS4_GLM52_L0_RANK_COUNT]) {
    if (!buffers) return;
    for (int rank = 0; rank < DS4_GLM52_L0_RANK_COUNT; rank++) {
        free(buffers[rank]);
        buffers[rank] = NULL;
    }
}

static bool send_reduced_payload(int fd,
                                 const ds4_glm52_tp4_collective_request *req,
                                 const float *sum,
                                 char *err,
                                 size_t err_size) {
    const int n = (int)req->element_count;
    if (!send_collective_frame(fd, req, false, err, err_size)) return false;
    if (n > 0 && !write_exact_local(fd, sum, (size_t)n * sizeof(sum[0]))) {
        snprintf(err, err_size, "reduced payload body write failed");
        return false;
    }
    return true;
}

static bool recv_and_validate_reduced_payload(int fd,
                                              const ds4_glm52_tp4_collective_request *expected,
                                              char *err,
                                              size_t err_size) {
    ds4_glm52_tp4_collective_request actual;
    if (!recv_collective_frame(fd, &actual, err, err_size)) return false;
    if (actual.rank != 0 || !collective_frames_match(&actual, expected)) {
        snprintf(err, err_size, "reduced collective frame mismatch");
        return false;
    }
    const int n = (int)actual.element_count;
    for (int i = 0; i < n; i++) {
        float value = 0.0f;
        if (!read_exact_local(fd, &value, sizeof(value))) {
            snprintf(err, err_size, "reduced payload body read failed");
            return false;
        }
        if (value != reduced_value(i)) {
            snprintf(err, err_size, "reduced payload value mismatch");
            return false;
        }
    }
    return true;
}

static bool parse_args(int argc, char **argv, smoke_config *cfg) {
    memset(cfg, 0, sizeof(*cfg));
    cfg->role = ROLE_NONE;
    cfg->rank = -1;
    cfg->timeout_ms = 10000;
    cfg->payload_floats = 0;
    cfg->bad_payload_frame = false;
    cfg->model_hash = 11;
    cfg->config_hash = 22;
    cfg->plan_hash = 33;

    for (int i = 1; i < argc; i++) {
        const char *arg = argv[i];
        const char *value = NULL;
        if (!strcmp(arg, "--bad-payload-frame")) {
            cfg->bad_payload_frame = true;
            continue;
        }
        if (!strcmp(arg, "--role") ||
            !strcmp(arg, "--rank") ||
            !strcmp(arg, "--listen") ||
            !strcmp(arg, "--connect") ||
            !strcmp(arg, "--timeout-ms") ||
            !strcmp(arg, "--payload-floats") ||
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
        } else if (!strcmp(arg, "--payload-floats")) {
            if (!parse_int_arg(value, &cfg->payload_floats) ||
                !payload_count_valid(cfg->payload_floats)) return false;
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
    fflush(stdout);

    ds4_glm52_tp4_command command = cfg->payload_floats > 0 ?
        DS4_GLM52_TP4_COMMAND_DECODE : DS4_GLM52_TP4_COMMAND_SHUTDOWN;
    uint64_t seq = 0;
    if (!ds4_glm52_tp4_rank_group_broadcast(
                &group,
                0,
                command,
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
                    command,
                    seq,
                    err,
                    sizeof(err))) {
            fprintf(stderr, "coordinator: send command rank %d failed: %s\n",
                    rank, err);
            close(listen_fd);
            return 11;
        }
    }

    float *sum = NULL;
    if (cfg->payload_floats > 0) {
        ds4_glm52_tp4_collective_request requests[DS4_GLM52_L0_RANK_COUNT];
        const float *partial_inputs[DS4_GLM52_L0_RANK_COUNT] = {0};
        float *partials[DS4_GLM52_L0_RANK_COUNT] = {0};
        float *outputs[DS4_GLM52_L0_RANK_COUNT] = {0};
        const size_t n = (size_t)cfg->payload_floats;
        bool allocated = true;
        for (int rank = 0; rank < DS4_GLM52_L0_RANK_COUNT; rank++) {
            requests[rank] = make_collective_request(cfg, rank, seq);
            partials[rank] = calloc(n, sizeof(partials[rank][0]));
            outputs[rank] = calloc(n, sizeof(outputs[rank][0]));
            allocated = allocated && partials[rank] && outputs[rank];
        }
        if (!allocated) {
            fprintf(stderr, "coordinator: payload allocation failed\n");
            free_rank_buffers(partials);
            free_rank_buffers(outputs);
            close(listen_fd);
            return 12;
        }
        for (int i = 0; i < cfg->payload_floats; i++) {
            partials[0][i] = partial_value(0, i);
        }
        partial_inputs[0] = partials[0];
        for (int rank = 1; rank < DS4_GLM52_L0_RANK_COUNT; rank++) {
            if (!recv_payload_partial(fds[rank],
                                      rank,
                                      &requests[rank],
                                      partials[rank],
                                      err,
                                      sizeof(err))) {
                fprintf(stderr, "coordinator: recv payload rank %d failed: %s\n",
                        rank, err);
                free_rank_buffers(partials);
                free_rank_buffers(outputs);
                close(listen_fd);
                return 13;
            }
            partial_inputs[rank] = partials[rank];
        }
        if (!ds4_glm52_tp4_collective_allreduce_f32_host(
                    requests,
                    partial_inputs,
                    outputs,
                    n,
                    err,
                    sizeof(err))) {
            fprintf(stderr, "coordinator: allreduce payload failed: %s\n", err);
            free_rank_buffers(partials);
            free_rank_buffers(outputs);
            close(listen_fd);
            return 14;
        }
        sum = outputs[0];
        for (int i = 0; i < cfg->payload_floats; i++) {
            if (sum[i] != reduced_value(i)) {
                fprintf(stderr, "coordinator: reduced payload mismatch at %d\n",
                        i);
                free_rank_buffers(partials);
                free_rank_buffers(outputs);
                close(listen_fd);
                return 15;
            }
        }
        for (int rank = 1; rank < DS4_GLM52_L0_RANK_COUNT; rank++) {
            if (!send_reduced_payload(fds[rank],
                                      &requests[0],
                                      outputs[rank],
                                      err,
                                      sizeof(err))) {
                fprintf(stderr, "coordinator: send reduced rank %d failed: %s\n",
                        rank, err);
                free_rank_buffers(partials);
                free_rank_buffers(outputs);
                close(listen_fd);
                return 16;
            }
        }
        printf("coordinator: allreduce payload floats=%d first=%.1f last=%.1f\n",
               cfg->payload_floats,
               sum[0],
               sum[cfg->payload_floats - 1]);
        free_rank_buffers(partials);
        free_rank_buffers(outputs);
        sum = NULL;
    }

    if (!ds4_glm52_tp4_rank_group_ack(
                &group,
                0,
                command,
                seq,
                err,
                sizeof(err))) {
        fprintf(stderr, "coordinator: local ack failed: %s\n", err);
        free(sum);
        close(listen_fd);
        return 17;
    }
    for (int rank = 1; rank < DS4_GLM52_L0_RANK_COUNT; rank++) {
        if (!ds4_glm52_tp4_transport_recv_ack_and_record(
                    fds[rank], &group, err, sizeof(err))) {
            fprintf(stderr, "coordinator: recv ack rank %d failed: %s\n",
                    rank, err);
            free(sum);
            close(listen_fd);
            return 18;
        }
        printf("coordinator: ack rank %d\n", rank);
        close(fds[rank]);
    }
    free(sum);
    close(listen_fd);

    if (!ds4_glm52_tp4_rank_group_command_done(&group)) {
        fprintf(stderr, "coordinator: shutdown command incomplete\n");
        return 19;
    }
    printf("coordinator: command %s complete seq=%llu ack_mask=0x%x\n",
           ds4_glm52_tp4_command_name(command),
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
    if (cfg->payload_floats > 0) {
        if (command != DS4_GLM52_TP4_COMMAND_DECODE) {
            fprintf(stderr, "worker%d: expected decode payload command\n",
                    cfg->rank);
            close(fd);
            return 6;
        }
        ds4_glm52_tp4_collective_request req =
            make_collective_request(cfg, cfg->rank, seq);
        bool corrupt_frame = cfg->bad_payload_frame && cfg->rank == 1;
        if (!send_payload(fd,
                          &req,
                          cfg->payload_floats,
                          corrupt_frame,
                          err,
                          sizeof(err))) {
            fprintf(stderr, "worker%d: send payload failed: %s\n",
                    cfg->rank, err);
            close(fd);
            return 7;
        }
        ds4_glm52_tp4_collective_request expected =
            make_collective_request(cfg, 0, seq);
        if (!recv_and_validate_reduced_payload(fd,
                                               &expected,
                                               err,
                                               sizeof(err))) {
            fprintf(stderr, "worker%d: reduced payload failed: %s\n",
                    cfg->rank, err);
            close(fd);
            return 8;
        }
        printf("worker%d: allreduce payload verified floats=%d\n",
               cfg->rank,
               cfg->payload_floats);
    }
    if (!ds4_glm52_tp4_transport_send_ack(
                fd, cfg->rank, command, seq, err, sizeof(err))) {
        fprintf(stderr, "worker%d: send ack failed: %s\n", cfg->rank, err);
        close(fd);
        return 9;
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
