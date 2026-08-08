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
    int dcp_rows;
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
            "  --dcp-rows N             optional per-owner DCP row payload count, max 8\n"
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

static ds4_glm52_decode_real_request make_decode_request(
        const smoke_config *cfg,
        uint64_t seq) {
    ds4_glm52_decode_real_request req;
    memset(&req, 0, sizeof(req));
    req.rank = cfg->rank;
    req.input_token = 123;
    req.seq = seq;
    req.model_hash = cfg->model_hash;
    req.session_hash = session_hash_for_smoke(cfg);
    req.model_ready = true;
    req.tp_collectives_ready = true;
    req.dcp_exchange_ready = true;
    req.glm52_kernels_ready = true;
    return req;
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

#define SMOKE_DCP_PER_RANK_ROWS 8
#define SMOKE_DCP_SELECTIONS 4
#define SMOKE_DCP_KV_BYTES 8
#define SMOKE_DCP_K_ROPE_BYTES 6

static bool dcp_rows_valid(int n) {
    return n >= 0 && n <= SMOKE_DCP_PER_RANK_ROWS;
}

static void fill_dcp_owners(
        ds4_glm52_dcp_owner_range owners[DS4_GLM52_L0_RANK_COUNT]) {
    for (int rank = 0; rank < DS4_GLM52_L0_RANK_COUNT; rank++) {
        owners[rank].rank = rank;
        owners[rank].row_start = (uint64_t)(rank * SMOKE_DCP_PER_RANK_ROWS);
        owners[rank].row_end =
            (uint64_t)((rank + 1) * SMOKE_DCP_PER_RANK_ROWS);
        owners[rank].present = true;
    }
}

static void fill_dcp_selection(int rank,
                               uint64_t selected[SMOKE_DCP_SELECTIONS]) {
    static const uint64_t selections[DS4_GLM52_L0_RANK_COUNT]
                                   [SMOKE_DCP_SELECTIONS] = {
        {28, 5, 20, 2},
        {25, 12, 3, 30},
        {6, 18, 1, 27},
        {23, 4, 15, 8},
    };
    memcpy(selected, selections[rank], sizeof(selections[rank]));
}

static ds4_glm52_dcp_exchange_request make_dcp_request(
        const smoke_config *cfg,
        int rank,
        uint64_t seq,
        int selection_count) {
    ds4_glm52_dcp_exchange_request req;
    memset(&req, 0, sizeof(req));
    req.rank = rank;
    req.dcp_size = DS4_GLM52_L0_DCP_SIZE;
    req.rank_count = DS4_GLM52_L0_RANK_COUNT;
    req.layer_index = 3;
    req.seq = seq;
    req.model_hash = cfg->model_hash;
    req.session_hash = session_hash_for_smoke(cfg);
    req.token_step_j = 29;
    req.selected_row_count = selection_count;
    req.owner_rank_mask = full_rank_mask_local();
    req.ownership_plan_valid = true;
    req.append_ordered_kv = true;
    req.row_payload_ready = true;
    req.transport_ready = true;
    return req;
}

static void build_dcp_owner_rows(
        int owner_rank,
        int row_count,
        ds4_glm52_dcp_bound_row_payload rows[SMOKE_DCP_PER_RANK_ROWS],
        unsigned char kv[SMOKE_DCP_PER_RANK_ROWS][SMOKE_DCP_KV_BYTES],
        unsigned char k_rope[SMOKE_DCP_PER_RANK_ROWS][SMOKE_DCP_K_ROPE_BYTES]) {
    for (int i = 0; i < row_count; i++) {
        const int row_id = owner_rank * SMOKE_DCP_PER_RANK_ROWS + i;
        for (size_t j = 0; j < SMOKE_DCP_KV_BYTES; j++) {
            kv[i][j] = (unsigned char)(row_id * 17 + (int)j);
        }
        for (size_t j = 0; j < SMOKE_DCP_K_ROPE_BYTES; j++) {
            k_rope[i][j] = (unsigned char)(row_id * 31 + owner_rank + (int)j);
        }
        memset(&rows[i], 0, sizeof(rows[i]));
        rows[i].row.row_id = (uint64_t)row_id;
        rows[i].row.owner_rank = owner_rank;
        rows[i].row.layer_index = 3;
        rows[i].row.token_step_j = (uint64_t)row_id;
        rows[i].row.kv_hash =
            ds4_glm52_dcp_payload_hash_host(kv[i], SMOKE_DCP_KV_BYTES);
        rows[i].row.k_rope_hash =
            ds4_glm52_dcp_payload_hash_host(
                    k_rope[i], SMOKE_DCP_K_ROPE_BYTES);
        rows[i].row.present = true;
        rows[i].kv_handle = kv[i];
        rows[i].kv_byte_offset = 0;
        rows[i].kv_byte_count = SMOKE_DCP_KV_BYTES;
        rows[i].kv_capacity_bytes = SMOKE_DCP_KV_BYTES;
        rows[i].k_rope_handle = k_rope[i];
        rows[i].k_rope_byte_offset = 0;
        rows[i].k_rope_byte_count = SMOKE_DCP_K_ROPE_BYTES;
        rows[i].k_rope_capacity_bytes = SMOKE_DCP_K_ROPE_BYTES;
        rows[i].ready = true;
    }
}

static bool make_dcp_transport_payload(
        const smoke_config *cfg,
        int rank,
        uint64_t seq,
        ds4_glm52_dcp_transport_payload *payload,
        char *err,
        size_t err_size) {
    uint64_t selected[SMOKE_DCP_SELECTIONS];
    fill_dcp_selection(rank, selected);
    ds4_glm52_dcp_bound_row_payload rows[SMOKE_DCP_PER_RANK_ROWS];
    unsigned char kv[SMOKE_DCP_PER_RANK_ROWS][SMOKE_DCP_KV_BYTES];
    unsigned char k_rope[SMOKE_DCP_PER_RANK_ROWS][SMOKE_DCP_K_ROPE_BYTES];
    build_dcp_owner_rows(rank, cfg->dcp_rows, rows, kv, k_rope);
    ds4_glm52_dcp_exchange_request req =
        make_dcp_request(cfg, rank, seq, SMOKE_DCP_SELECTIONS);
    return ds4_glm52_dcp_transport_payload_from_bound(
            &req,
            selected,
            SMOKE_DCP_SELECTIONS,
            rows,
            (size_t)cfg->dcp_rows,
            payload,
            err,
            err_size);
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

static ds4_glm52_tp4_tensor_binding payload_binding(
        int rank,
        float *buffer,
        const ds4_glm52_tp4_collective_request *req) {
    ds4_glm52_tp4_tensor_binding binding;
    memset(&binding, 0, sizeof(binding));
    binding.rank = rank;
    binding.handle = buffer;
    binding.byte_offset = 0;
    binding.byte_count = req->byte_count;
    binding.capacity_bytes = req->byte_count;
    binding.dtype = req->dtype;
    binding.shape_hash = req->shape_hash;
    binding.ready = true;
    return binding;
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
    cfg->dcp_rows = 0;
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
            !strcmp(arg, "--dcp-rows") ||
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
        } else if (!strcmp(arg, "--dcp-rows")) {
            if (!parse_int_arg(value, &cfg->dcp_rows) ||
                !dcp_rows_valid(cfg->dcp_rows)) return false;
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
    if (cfg->payload_floats == 0 && cfg->dcp_rows == 0) {
        ds4_glm52_tp4_fabric_ready_config ready_cfg = {
            .rank = cfg->rank,
            .endpoint = cfg->listen,
            .timeout_ms = cfg->timeout_ms,
            .model_hash = cfg->model_hash,
            .config_hash = cfg->config_hash,
            .plan_hash = cfg->plan_hash,
            .command = DS4_GLM52_TP4_COMMAND_SHUTDOWN,
        };
        ds4_glm52_tp4_fabric_ready_result ready;
        if (!ds4_glm52_tp4_fabric_ready_handshake(
                    &ready_cfg, &ready, err, sizeof(err))) {
            fprintf(stderr, "coordinator: fabric ready handshake failed: %s\n",
                    err);
            return 3;
        }
        printf("coordinator: fabric ready tp=%d dcp=%d ranks=%d\n",
               ready.state.tp_fabric.tp_size,
               ready.state.tp_fabric.dcp_size,
               ready.state.tp_fabric.rank_count);
        printf("coordinator: command %s complete seq=%llu ack_mask=0x%x\n",
               ds4_glm52_tp4_command_name(ready.command),
               (unsigned long long)ready.command_seq,
               ready.ack_mask);
        return 0;
    }

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

    ds4_glm52_tp4_command command =
        (cfg->payload_floats > 0 || cfg->dcp_rows > 0) ?
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
    if (cfg->dcp_rows > 0) {
        ds4_glm52_dcp_owner_range owners[DS4_GLM52_L0_RANK_COUNT];
        ds4_glm52_dcp_transport_payload payloads[DS4_GLM52_L0_RANK_COUNT];
        ds4_glm52_dcp_exchange_request requests[DS4_GLM52_L0_RANK_COUNT];
        uint64_t *selected[DS4_GLM52_L0_RANK_COUNT];
        size_t selection_counts[DS4_GLM52_L0_RANK_COUNT];
        ds4_glm52_dcp_bound_row_payload rows[
            DS4_GLM52_L0_RANK_COUNT * SMOKE_DCP_PER_RANK_ROWS];
        size_t row_count = 0;
        ds4_glm52_dcp_row_payload reply_rows[
            DS4_GLM52_L0_RANK_COUNT][SMOKE_DCP_SELECTIONS];
        ds4_glm52_dcp_rank_reply replies[DS4_GLM52_L0_RANK_COUNT];

        fill_dcp_owners(owners);
        if (!make_dcp_transport_payload(
                    cfg, 0, seq, &payloads[0], err, sizeof(err))) {
            fprintf(stderr, "coordinator: make local DCP payload failed: %s\n",
                    err);
            close(listen_fd);
            return 12;
        }
        for (int rank = 1; rank < DS4_GLM52_L0_RANK_COUNT; rank++) {
            if (!ds4_glm52_dcp_transport_recv_payload(
                        fds[rank], &payloads[rank], err, sizeof(err))) {
                fprintf(stderr, "coordinator: recv DCP payload rank %d failed: %s\n",
                        rank, err);
                close(listen_fd);
                return 13;
            }
        }
        for (int rank = 0; rank < DS4_GLM52_L0_RANK_COUNT; rank++) {
            size_t got_rows = 0;
            if (!ds4_glm52_dcp_transport_payload_to_bound(
                        &payloads[rank],
                        &requests[rank],
                        &selected[rank],
                        &selection_counts[rank],
                        &rows[row_count],
                        (DS4_GLM52_L0_RANK_COUNT * SMOKE_DCP_PER_RANK_ROWS) -
                            row_count,
                        &got_rows,
                        err,
                        sizeof(err))) {
                fprintf(stderr, "coordinator: decode DCP payload rank %d failed: %s\n",
                        rank, err);
                close(listen_fd);
                return 14;
            }
            row_count += got_rows;
        }
        for (int rank = 0; rank < DS4_GLM52_L0_RANK_COUNT; rank++) {
            memset(reply_rows[rank], 0, sizeof(reply_rows[rank]));
            memset(&replies[rank], 0, sizeof(replies[rank]));
            replies[rank].requester_rank = rank;
            replies[rank].rows = reply_rows[rank];
            replies[rank].row_capacity = SMOKE_DCP_SELECTIONS;
        }
        if (!ds4_glm52_dcp_selected_rows_bound_host(
                    requests,
                    owners,
                    rows,
                    row_count,
                    (const uint64_t *const *)selected,
                    selection_counts,
                    replies,
                    err,
                    sizeof(err))) {
            fprintf(stderr, "coordinator: bound DCP payload exchange failed: %s\n",
                    err);
            close(listen_fd);
            return 15;
        }
        for (int rank = 0; rank < DS4_GLM52_L0_RANK_COUNT; rank++) {
            if (!replies[rank].complete ||
                replies[rank].row_count != SMOKE_DCP_SELECTIONS) {
                fprintf(stderr, "coordinator: DCP reply rank %d incomplete\n",
                        rank);
                close(listen_fd);
                return 16;
            }
        }
        printf("coordinator: dcp payload rows=%zu replies=%d x %d\n",
               row_count,
               DS4_GLM52_L0_RANK_COUNT,
               SMOKE_DCP_SELECTIONS);
    }
    if (cfg->payload_floats > 0) {
        ds4_glm52_tp4_collective_request requests[DS4_GLM52_L0_RANK_COUNT];
        ds4_glm52_tp4_tensor_binding partial_bindings[DS4_GLM52_L0_RANK_COUNT];
        ds4_glm52_tp4_tensor_binding output_bindings[DS4_GLM52_L0_RANK_COUNT];
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
        }
        for (int rank = 0; rank < DS4_GLM52_L0_RANK_COUNT; rank++) {
            partial_bindings[rank] =
                payload_binding(rank, partials[rank], &requests[rank]);
            output_bindings[rank] =
                payload_binding(rank, outputs[rank], &requests[rank]);
        }
        ds4_glm52_decode_real_request decode = make_decode_request(cfg, seq);
        decode.rank = 0;
        ds4_glm52_decode_bound_collective_call call = {
            .decode = &decode,
            .requests = requests,
            .partials = partial_bindings,
            .outputs = output_bindings,
            .element_count = n,
        };
        if (!ds4_glm52_decode_bound_collective_host(
                    &call, err, sizeof(err))) {
            fprintf(stderr, "coordinator: decode-bound allreduce payload failed: %s\n",
                    err);
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
    if (cfg->payload_floats == 0 && cfg->dcp_rows == 0) {
        ds4_glm52_tp4_fabric_ready_config ready_cfg = {
            .rank = cfg->rank,
            .endpoint = cfg->connect,
            .timeout_ms = cfg->timeout_ms,
            .model_hash = cfg->model_hash,
            .config_hash = cfg->config_hash,
            .plan_hash = cfg->plan_hash,
            .command = DS4_GLM52_TP4_COMMAND_SHUTDOWN,
        };
        ds4_glm52_tp4_fabric_ready_result ready;
        if (!ds4_glm52_tp4_fabric_ready_handshake(
                    &ready_cfg, &ready, err, sizeof(err))) {
            fprintf(stderr, "worker%d: fabric ready handshake failed: %s\n",
                    cfg->rank, err);
            return 3;
        }
        printf("worker%d: command %s seq=%llu\n",
               cfg->rank,
               ds4_glm52_tp4_command_name(ready.command),
               (unsigned long long)ready.command_seq);
        return 0;
    }

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
    if (cfg->dcp_rows > 0) {
        if (command != DS4_GLM52_TP4_COMMAND_DECODE) {
            fprintf(stderr, "worker%d: expected decode DCP command\n",
                    cfg->rank);
            close(fd);
            return 6;
        }
        ds4_glm52_dcp_transport_payload payload;
        if (!make_dcp_transport_payload(
                    cfg, cfg->rank, seq, &payload, err, sizeof(err))) {
            fprintf(stderr, "worker%d: make DCP payload failed: %s\n",
                    cfg->rank, err);
            close(fd);
            return 7;
        }
        if (!ds4_glm52_dcp_transport_send_payload(
                    fd, &payload, err, sizeof(err))) {
            fprintf(stderr, "worker%d: send DCP payload failed: %s\n",
                    cfg->rank, err);
            close(fd);
            return 8;
        }
        printf("worker%d: dcp payload sent rows=%llu\n",
               cfg->rank,
               (unsigned long long)payload.row_count);
    }
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
