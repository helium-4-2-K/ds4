#include "ds4_gpu.h"
#include "ds4_gpu_mgpu.h"
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
    int payload_floats;
    uint64_t model_hash;
    uint64_t config_hash;
    uint64_t plan_hash;
} smoke_config;

static void usage(FILE *fp, const char *prog) {
    fprintf(fp,
            "usage:\n"
            "  %s --role coordinator --rank 0 --listen HOST:PORT --payload-floats N\n"
            "  %s --role worker --rank N --connect HOST:PORT --payload-floats N\n",
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

static bool parse_args(int argc, char **argv, smoke_config *cfg) {
    memset(cfg, 0, sizeof(*cfg));
    cfg->role = ROLE_NONE;
    cfg->rank = -1;
    cfg->timeout_ms = 10000;
    cfg->payload_floats = 1024;
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
            !strcmp(arg, "--payload-floats") ||
            !strcmp(arg, "--model-hash") ||
            !strcmp(arg, "--config-hash") ||
            !strcmp(arg, "--plan-hash")) {
            if (++i >= argc) return false;
            value = argv[i];
        }
        if (!strcmp(arg, "--role")) {
            if (!strcmp(value, "coordinator")) cfg->role = ROLE_COORDINATOR;
            else if (!strcmp(value, "worker")) cfg->role = ROLE_WORKER;
            else return false;
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
                cfg->payload_floats <= 0 || cfg->payload_floats > 4096) {
                return false;
            }
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

static bool endpoint_is_management(const ds4_glm52_tp4_tcp_endpoint *endpoint) {
    return endpoint &&
           !strncmp(endpoint->host, "192.168.0.", strlen("192.168.0."));
}

static uint32_t full_rank_mask(void) {
    return (UINT32_C(1) << DS4_GLM52_L0_RANK_COUNT) - UINT32_C(1);
}

static uint64_t session_hash_for_smoke(const smoke_config *cfg) {
    return cfg->config_hash ^ (cfg->plan_hash << 1) ^ UINT64_C(0x9e3779b97f4a7c15);
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
    req.participant_mask = full_rank_mask();
    req.topology_ready = true;
    req.transport_ready = true;
    req.rank_local_partial_ready = true;
    req.replicated_output_ready = true;
    return req;
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

static bool collective_frames_match_except_rank(
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

static int gpu_read_cb(const ds4_gpu_tensor *tensor,
                       uint64_t offset,
                       void *data,
                       uint64_t bytes) {
    return ds4_gpu_tensor_read(tensor, offset, data, bytes);
}

static int gpu_write_cb(ds4_gpu_tensor *tensor,
                        uint64_t offset,
                        const void *data,
                        uint64_t bytes) {
    return ds4_gpu_tensor_write(tensor, offset, data, bytes);
}

static ds4_glm52_tp4_gpu_tensor_binding make_gpu_binding(
        int rank,
        const ds4_gpu_tensor *tensor,
        const ds4_glm52_tp4_collective_request *req) {
    ds4_glm52_tp4_gpu_tensor_binding binding;
    memset(&binding, 0, sizeof(binding));
    binding.rank = rank;
    binding.tensor = tensor;
    binding.byte_offset = 0;
    binding.byte_count = req->byte_count;
    binding.dtype = req->dtype;
    binding.shape_hash = req->shape_hash;
    binding.ready = true;
    return binding;
}

static int init_gpu(void) {
    ds4_gpu_config gpu_cfg;
    memset(&gpu_cfg, 0, sizeof(gpu_cfg));
    gpu_cfg.n_gpus = 1;
    gpu_cfg.device_indices[0] = 0;
    return ds4_gpu_init_multi(&gpu_cfg) ? 0 : 1;
}

static int rank_from_new_mask(uint32_t before, uint32_t after) {
    uint32_t diff = after & ~before;
    for (int rank = 0; rank < DS4_GLM52_L0_RANK_COUNT; rank++) {
        if (diff == (UINT32_C(1) << rank)) return rank;
    }
    return -1;
}

static void free_rank_buffers(float *buffers[DS4_GLM52_L0_RANK_COUNT]) {
    for (int rank = 0; rank < DS4_GLM52_L0_RANK_COUNT; rank++) {
        free(buffers[rank]);
        buffers[rank] = NULL;
    }
}

static int run_coordinator(const smoke_config *cfg) {
    char err[256] = "";
    if (init_gpu() != 0) {
        fprintf(stderr, "coordinator: CUDA init failed\n");
        return 2;
    }

    ds4_glm52_tp4_tcp_endpoint endpoint;
    if (!ds4_glm52_tp4_tcp_parse_endpoint(
                cfg->listen, &endpoint, err, sizeof(err))) {
        fprintf(stderr, "coordinator: bad listen endpoint: %s\n", err);
        ds4_gpu_cleanup();
        return 3;
    }
    if (endpoint_is_management(&endpoint)) {
        fprintf(stderr,
                "coordinator: refusing management endpoint %s; use CRS812 fabric\n",
                endpoint.host);
        ds4_gpu_cleanup();
        return 3;
    }

    int listen_fd = -1;
    uint16_t bound_port = 0;
    if (!ds4_glm52_tp4_tcp_listen(
                &endpoint, &listen_fd, &bound_port, err, sizeof(err))) {
        fprintf(stderr, "coordinator: listen failed: %s\n", err);
        ds4_gpu_cleanup();
        return 4;
    }
    printf("coordinator: listening %s:%u\n", endpoint.host, bound_port);
    fflush(stdout);

    ds4_glm52_tp4_rank_group group;
    ds4_glm52_tp4_rank_group_init(
            &group, cfg->model_hash, cfg->config_hash, cfg->plan_hash);
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
        fprintf(stderr, "coordinator: local register failed: %s\n", err);
        close(listen_fd);
        ds4_gpu_cleanup();
        return 5;
    }

    int fds[DS4_GLM52_L0_RANK_COUNT];
    for (int i = 0; i < DS4_GLM52_L0_RANK_COUNT; i++) fds[i] = -1;
    for (int i = 1; i < DS4_GLM52_L0_RANK_COUNT; i++) {
        int fd = -1;
        if (!ds4_glm52_tp4_tcp_accept(
                    listen_fd, cfg->timeout_ms, &fd, err, sizeof(err))) {
            fprintf(stderr, "coordinator: accept failed: %s\n", err);
            close(listen_fd);
            ds4_gpu_cleanup();
            return 6;
        }
        uint32_t before = group.registered_mask;
        if (!ds4_glm52_tp4_transport_recv_hello_and_register(
                    fd, &group, err, sizeof(err))) {
            fprintf(stderr, "coordinator: hello failed: %s\n", err);
            close(fd);
            close(listen_fd);
            ds4_gpu_cleanup();
            return 7;
        }
        int rank = rank_from_new_mask(before, group.registered_mask);
        if (rank <= 0 || rank >= DS4_GLM52_L0_RANK_COUNT || fds[rank] >= 0) {
            fprintf(stderr, "coordinator: invalid new rank registration\n");
            close(fd);
            close(listen_fd);
            ds4_gpu_cleanup();
            return 8;
        }
        fds[rank] = fd;
        printf("coordinator: registered rank %d\n", rank);
        fflush(stdout);
    }
    if (!ds4_glm52_tp4_rank_group_ready(&group)) {
        fprintf(stderr, "coordinator: rank group not ready\n");
        close(listen_fd);
        ds4_gpu_cleanup();
        return 9;
    }

    uint64_t seq = 0;
    if (!ds4_glm52_tp4_rank_group_broadcast(
                &group,
                0,
                DS4_GLM52_TP4_COMMAND_DECODE,
                &seq,
                err,
                sizeof(err))) {
        fprintf(stderr, "coordinator: broadcast failed: %s\n", err);
        close(listen_fd);
        ds4_gpu_cleanup();
        return 10;
    }
    for (int rank = 1; rank < DS4_GLM52_L0_RANK_COUNT; rank++) {
        if (!ds4_glm52_tp4_transport_send_command(
                    fds[rank],
                    DS4_GLM52_TP4_COMMAND_DECODE,
                    seq,
                    err,
                    sizeof(err))) {
            fprintf(stderr, "coordinator: send command rank %d failed: %s\n",
                    rank, err);
            close(listen_fd);
            ds4_gpu_cleanup();
            return 11;
        }
    }

    const size_t n = (size_t)cfg->payload_floats;
    ds4_glm52_tp4_collective_request requests[DS4_GLM52_L0_RANK_COUNT];
    float *partials[DS4_GLM52_L0_RANK_COUNT] = {0};
    float *outputs[DS4_GLM52_L0_RANK_COUNT] = {0};
    const float *partial_inputs[DS4_GLM52_L0_RANK_COUNT] = {0};
    bool allocated = true;
    for (int rank = 0; rank < DS4_GLM52_L0_RANK_COUNT; rank++) {
        requests[rank] = make_collective_request(cfg, rank, seq);
        partials[rank] = calloc(n, sizeof(partials[rank][0]));
        outputs[rank] = calloc(n, sizeof(outputs[rank][0]));
        allocated = allocated && partials[rank] && outputs[rank];
        partial_inputs[rank] = partials[rank];
    }
    if (!allocated) {
        fprintf(stderr, "coordinator: host staging allocation failed\n");
        free_rank_buffers(partials);
        free_rank_buffers(outputs);
        close(listen_fd);
        ds4_gpu_cleanup();
        return 12;
    }

    ds4_gpu_tensor local_partial;
    ds4_gpu_tensor local_output;
    memset(&local_partial, 0, sizeof(local_partial));
    memset(&local_output, 0, sizeof(local_output));
    if (ds4_gpu_tensor_alloc_on(&local_partial, 0, requests[0].byte_count) != 0 ||
        ds4_gpu_tensor_alloc_on(&local_output, 0, requests[0].byte_count) != 0) {
        fprintf(stderr, "coordinator: GPU tensor allocation failed\n");
        free_rank_buffers(partials);
        free_rank_buffers(outputs);
        close(listen_fd);
        ds4_gpu_cleanup();
        return 13;
    }
    for (int i = 0; i < cfg->payload_floats; i++) {
        partials[0][i] = partial_value(0, i);
    }
    if (!ds4_gpu_tensor_write(&local_partial,
                              0,
                              partials[0],
                              requests[0].byte_count)) {
        fprintf(stderr, "coordinator: GPU partial write failed\n");
        ds4_gpu_tensor_free_in_place(&local_partial);
        ds4_gpu_tensor_free_in_place(&local_output);
        free_rank_buffers(partials);
        free_rank_buffers(outputs);
        close(listen_fd);
        ds4_gpu_cleanup();
        return 14;
    }
    const ds4_glm52_gpu_tensor_io io = {
        .read = gpu_read_cb,
        .write = gpu_write_cb,
    };
    ds4_glm52_tp4_gpu_tensor_binding local_partial_binding =
        make_gpu_binding(0, &local_partial, &requests[0]);
    ds4_glm52_tp4_gpu_tensor_binding local_output_binding =
        make_gpu_binding(0, &local_output, &requests[0]);
    if (!ds4_glm52_tp4_gpu_collective_read_f32(
                &requests[0],
                &local_partial_binding,
                0,
                &io,
                partials[0],
                n,
                err,
                sizeof(err))) {
        fprintf(stderr, "coordinator: staged GPU read failed: %s\n", err);
        ds4_gpu_tensor_free_in_place(&local_partial);
        ds4_gpu_tensor_free_in_place(&local_output);
        free_rank_buffers(partials);
        free_rank_buffers(outputs);
        close(listen_fd);
        ds4_gpu_cleanup();
        return 15;
    }

    for (int rank = 1; rank < DS4_GLM52_L0_RANK_COUNT; rank++) {
        ds4_glm52_tp4_collective_request actual;
        size_t got = 0;
        if (!ds4_glm52_tp4_transport_recv_collective_f32(
                    fds[rank],
                    &actual,
                    partials[rank],
                    n,
                    &got,
                    err,
                    sizeof(err)) ||
            actual.rank != rank ||
            got != n ||
            !collective_frames_match_except_rank(&actual, &requests[rank])) {
            fprintf(stderr, "coordinator: recv GPU payload rank %d failed: %s\n",
                    rank, err);
            ds4_gpu_tensor_free_in_place(&local_partial);
            ds4_gpu_tensor_free_in_place(&local_output);
            free_rank_buffers(partials);
            free_rank_buffers(outputs);
            close(listen_fd);
            ds4_gpu_cleanup();
            return 16;
        }
    }

    if (!ds4_glm52_tp4_collective_allreduce_f32_host(
                requests, partial_inputs, outputs, n, err, sizeof(err))) {
        fprintf(stderr, "coordinator: host reduce failed: %s\n", err);
        ds4_gpu_tensor_free_in_place(&local_partial);
        ds4_gpu_tensor_free_in_place(&local_output);
        free_rank_buffers(partials);
        free_rank_buffers(outputs);
        close(listen_fd);
        ds4_gpu_cleanup();
        return 17;
    }
    if (!ds4_glm52_tp4_gpu_collective_write_f32(
                &requests[0],
                &local_output_binding,
                0,
                &io,
                outputs[0],
                n,
                err,
                sizeof(err))) {
        fprintf(stderr, "coordinator: staged GPU write failed: %s\n", err);
        ds4_gpu_tensor_free_in_place(&local_partial);
        ds4_gpu_tensor_free_in_place(&local_output);
        free_rank_buffers(partials);
        free_rank_buffers(outputs);
        close(listen_fd);
        ds4_gpu_cleanup();
        return 18;
    }
    memset(partials[0], 0, requests[0].byte_count);
    if (!ds4_gpu_tensor_read(&local_output, 0, partials[0], requests[0].byte_count)) {
        fprintf(stderr, "coordinator: GPU output readback failed\n");
        ds4_gpu_tensor_free_in_place(&local_partial);
        ds4_gpu_tensor_free_in_place(&local_output);
        free_rank_buffers(partials);
        free_rank_buffers(outputs);
        close(listen_fd);
        ds4_gpu_cleanup();
        return 19;
    }
    for (int i = 0; i < cfg->payload_floats; i++) {
        if (partials[0][i] != reduced_value(i)) {
            fprintf(stderr, "coordinator: GPU reduced mismatch at %d\n", i);
            ds4_gpu_tensor_free_in_place(&local_partial);
            ds4_gpu_tensor_free_in_place(&local_output);
            free_rank_buffers(partials);
            free_rank_buffers(outputs);
            close(listen_fd);
            ds4_gpu_cleanup();
            return 20;
        }
    }
    for (int rank = 1; rank < DS4_GLM52_L0_RANK_COUNT; rank++) {
        if (!ds4_glm52_tp4_transport_send_collective_f32(
                    fds[rank],
                    &requests[rank],
                    outputs[rank],
                    n,
                    err,
                    sizeof(err))) {
            fprintf(stderr, "coordinator: send reduced rank %d failed: %s\n",
                    rank, err);
            ds4_gpu_tensor_free_in_place(&local_partial);
            ds4_gpu_tensor_free_in_place(&local_output);
            free_rank_buffers(partials);
            free_rank_buffers(outputs);
            close(listen_fd);
            ds4_gpu_cleanup();
            return 21;
        }
    }

    if (!ds4_glm52_tp4_rank_group_ack(
                &group,
                0,
                DS4_GLM52_TP4_COMMAND_DECODE,
                seq,
                err,
                sizeof(err))) {
        fprintf(stderr, "coordinator: local ack failed: %s\n", err);
        ds4_gpu_tensor_free_in_place(&local_partial);
        ds4_gpu_tensor_free_in_place(&local_output);
        free_rank_buffers(partials);
        free_rank_buffers(outputs);
        close(listen_fd);
        ds4_gpu_cleanup();
        return 22;
    }
    for (int rank = 1; rank < DS4_GLM52_L0_RANK_COUNT; rank++) {
        if (!ds4_glm52_tp4_transport_recv_ack_and_record(
                    fds[rank], &group, err, sizeof(err))) {
            fprintf(stderr, "coordinator: recv ack rank %d failed: %s\n",
                    rank, err);
            ds4_gpu_tensor_free_in_place(&local_partial);
            ds4_gpu_tensor_free_in_place(&local_output);
            free_rank_buffers(partials);
            free_rank_buffers(outputs);
            close(listen_fd);
            ds4_gpu_cleanup();
            return 23;
        }
        close(fds[rank]);
    }
    printf("coordinator: gpu allreduce payload floats=%d first=%.1f last=%.1f ack_mask=0x%x\n",
           cfg->payload_floats,
           partials[0][0],
           partials[0][cfg->payload_floats - 1],
           group.ack_mask);
    ds4_gpu_tensor_free_in_place(&local_partial);
    ds4_gpu_tensor_free_in_place(&local_output);
    free_rank_buffers(partials);
    free_rank_buffers(outputs);
    close(listen_fd);
    ds4_gpu_cleanup();
    return ds4_glm52_tp4_rank_group_command_done(&group) ? 0 : 24;
}

static int run_worker(const smoke_config *cfg) {
    char err[256] = "";
    if (init_gpu() != 0) {
        fprintf(stderr, "worker%d: CUDA init failed\n", cfg->rank);
        return 2;
    }

    ds4_glm52_tp4_tcp_endpoint endpoint;
    if (!ds4_glm52_tp4_tcp_parse_endpoint(
                cfg->connect, &endpoint, err, sizeof(err))) {
        fprintf(stderr, "worker%d: bad connect endpoint: %s\n", cfg->rank, err);
        ds4_gpu_cleanup();
        return 3;
    }
    if (endpoint_is_management(&endpoint)) {
        fprintf(stderr,
                "worker%d: refusing management endpoint %s; use CRS812 fabric\n",
                cfg->rank,
                endpoint.host);
        ds4_gpu_cleanup();
        return 3;
    }

    int fd = -1;
    if (!ds4_glm52_tp4_tcp_connect(
                &endpoint, cfg->timeout_ms, &fd, err, sizeof(err))) {
        fprintf(stderr, "worker%d: connect failed: %s\n", cfg->rank, err);
        ds4_gpu_cleanup();
        return 4;
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
        ds4_gpu_cleanup();
        return 5;
    }
    ds4_glm52_tp4_command command = DS4_GLM52_TP4_COMMAND_NONE;
    uint64_t seq = 0;
    if (!ds4_glm52_tp4_transport_recv_command(fd, &command, &seq, err, sizeof(err)) ||
        command != DS4_GLM52_TP4_COMMAND_DECODE) {
        fprintf(stderr, "worker%d: recv decode command failed: %s\n",
                cfg->rank, err);
        close(fd);
        ds4_gpu_cleanup();
        return 6;
    }

    ds4_glm52_tp4_collective_request req =
        make_collective_request(cfg, cfg->rank, seq);
    const size_t n = (size_t)cfg->payload_floats;
    float *host = calloc(n, sizeof(host[0]));
    float *reduced = calloc(n, sizeof(reduced[0]));
    if (!host || !reduced) {
        fprintf(stderr, "worker%d: host staging allocation failed\n", cfg->rank);
        free(host);
        free(reduced);
        close(fd);
        ds4_gpu_cleanup();
        return 7;
    }
    for (int i = 0; i < cfg->payload_floats; i++) {
        host[i] = partial_value(cfg->rank, i);
    }

    ds4_gpu_tensor partial;
    ds4_gpu_tensor output;
    memset(&partial, 0, sizeof(partial));
    memset(&output, 0, sizeof(output));
    if (ds4_gpu_tensor_alloc_on(&partial, 0, req.byte_count) != 0 ||
        ds4_gpu_tensor_alloc_on(&output, 0, req.byte_count) != 0) {
        fprintf(stderr, "worker%d: GPU tensor allocation failed\n", cfg->rank);
        free(host);
        free(reduced);
        close(fd);
        ds4_gpu_cleanup();
        return 8;
    }
    if (!ds4_gpu_tensor_write(&partial, 0, host, req.byte_count)) {
        fprintf(stderr, "worker%d: GPU partial write failed\n", cfg->rank);
        ds4_gpu_tensor_free_in_place(&partial);
        ds4_gpu_tensor_free_in_place(&output);
        free(host);
        free(reduced);
        close(fd);
        ds4_gpu_cleanup();
        return 9;
    }
    const ds4_glm52_gpu_tensor_io io = {
        .read = gpu_read_cb,
        .write = gpu_write_cb,
    };
    ds4_glm52_tp4_gpu_tensor_binding partial_binding =
        make_gpu_binding(cfg->rank, &partial, &req);
    ds4_glm52_tp4_gpu_tensor_binding output_binding =
        make_gpu_binding(cfg->rank, &output, &req);
    memset(host, 0, req.byte_count);
    if (!ds4_glm52_tp4_gpu_collective_read_f32(
                &req,
                &partial_binding,
                0,
                &io,
                host,
                n,
                err,
                sizeof(err)) ||
        !ds4_glm52_tp4_transport_send_collective_f32(
                fd, &req, host, n, err, sizeof(err))) {
        fprintf(stderr, "worker%d: staged GPU send failed: %s\n", cfg->rank, err);
        ds4_gpu_tensor_free_in_place(&partial);
        ds4_gpu_tensor_free_in_place(&output);
        free(host);
        free(reduced);
        close(fd);
        ds4_gpu_cleanup();
        return 10;
    }

    ds4_glm52_tp4_collective_request response;
    size_t got = 0;
    if (!ds4_glm52_tp4_transport_recv_collective_f32(
                fd,
                &response,
                reduced,
                n,
                &got,
                err,
                sizeof(err)) ||
        response.rank != cfg->rank ||
        got != n ||
        !collective_frames_match_except_rank(&response, &req) ||
        !ds4_glm52_tp4_gpu_collective_write_f32(
                &response,
                &output_binding,
                0,
                &io,
                reduced,
                n,
                err,
                sizeof(err))) {
        fprintf(stderr, "worker%d: staged GPU recv/write failed: %s\n",
                cfg->rank, err);
        ds4_gpu_tensor_free_in_place(&partial);
        ds4_gpu_tensor_free_in_place(&output);
        free(host);
        free(reduced);
        close(fd);
        ds4_gpu_cleanup();
        return 11;
    }
    memset(host, 0, req.byte_count);
    if (!ds4_gpu_tensor_read(&output, 0, host, req.byte_count)) {
        fprintf(stderr, "worker%d: GPU output readback failed\n", cfg->rank);
        ds4_gpu_tensor_free_in_place(&partial);
        ds4_gpu_tensor_free_in_place(&output);
        free(host);
        free(reduced);
        close(fd);
        ds4_gpu_cleanup();
        return 12;
    }
    for (int i = 0; i < cfg->payload_floats; i++) {
        if (host[i] != reduced_value(i)) {
            fprintf(stderr, "worker%d: GPU reduced mismatch at %d\n",
                    cfg->rank, i);
            ds4_gpu_tensor_free_in_place(&partial);
            ds4_gpu_tensor_free_in_place(&output);
            free(host);
            free(reduced);
            close(fd);
            ds4_gpu_cleanup();
            return 13;
        }
    }
    if (!ds4_glm52_tp4_transport_send_ack(
                fd,
                cfg->rank,
                DS4_GLM52_TP4_COMMAND_DECODE,
                seq,
                err,
                sizeof(err))) {
        fprintf(stderr, "worker%d: send ack failed: %s\n", cfg->rank, err);
        ds4_gpu_tensor_free_in_place(&partial);
        ds4_gpu_tensor_free_in_place(&output);
        free(host);
        free(reduced);
        close(fd);
        ds4_gpu_cleanup();
        return 14;
    }
    printf("worker%d: gpu allreduce verified floats=%d first=%.1f last=%.1f\n",
           cfg->rank,
           cfg->payload_floats,
           host[0],
           host[cfg->payload_floats - 1]);
    ds4_gpu_tensor_free_in_place(&partial);
    ds4_gpu_tensor_free_in_place(&output);
    free(host);
    free(reduced);
    close(fd);
    ds4_gpu_cleanup();
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
