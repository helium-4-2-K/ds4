#include "ds4_glm52_l0.h"

#include <limits.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

#ifndef PATH_MAX
#define PATH_MAX 4096
#endif

typedef struct {
    const char *ids[8];
    ds4_glm52_l0_status statuses[8];
    bool prerequisite_blocked[8];
    int n;
} trace_capture;

static bool g_fixture_ready;
static char g_fixture_dir[PATH_MAX];
static char g_model_root[PATH_MAX];
static char g_plan_path[PATH_MAX];
static char g_missing_plan_path[PATH_MAX];
static char g_foreign_plan_path[PATH_MAX];
static int g_fake_gpu_allocs;
static int g_fake_gpu_frees;

static void fail(const char *msg) {
    fprintf(stderr, "test_glm52_l0: %s\n", msg);
    exit(1);
}

static void check(bool ok, const char *msg) {
    if (!ok) fail(msg);
}

static int fake_gpu_alloc(ds4_gpu_tensor *tensor,
                          int device_id,
                          uint64_t bytes) {
    if (!tensor || device_id < 0 || bytes == 0) return 0;
    void *ptr = malloc((size_t)bytes);
    if (!ptr) return 0;
    tensor->ptr = ptr;
    tensor->bytes = bytes;
    tensor->owner = 1;
    tensor->device_id = device_id;
    g_fake_gpu_allocs++;
    return 1;
}

static int fake_gpu_upload(ds4_gpu_tensor *tensor,
                           uint64_t offset,
                           const void *data,
                           uint64_t bytes) {
    if (!tensor || !tensor->ptr || !data ||
        offset > tensor->bytes ||
        bytes > tensor->bytes - offset) {
        return 0;
    }
    memcpy((unsigned char *)tensor->ptr + offset, data, (size_t)bytes);
    return 1;
}

static void fake_gpu_free(ds4_gpu_tensor *tensor) {
    if (!tensor) return;
    free(tensor->ptr);
    memset(tensor, 0, sizeof(*tensor));
    g_fake_gpu_frees++;
}

static const ds4_glm52_gpu_tensor_runtime k_fake_gpu_runtime = {
    .alloc = fake_gpu_alloc,
    .upload = fake_gpu_upload,
    .free = fake_gpu_free,
};

static void create_file(const char *path) {
    FILE *fp = fopen(path, "wb");
    check(fp != NULL, "fixture file open failed");
    fputs("x", fp);
    fclose(fp);
}

static void write_rank_plan(const char *path, int rank,
                            int base_count, bool foreign_first) {
    FILE *fp = fopen(path, "w");
    check(fp != NULL, "fixture rank plan open failed");
    fprintf(fp, "tp_size=4\n");
    fprintf(fp, "dcp_size=4\n");
    fprintf(fp, "pp_size=1\n");
    fprintf(fp, "rank_count=4\n");
    fprintf(fp, "host_count=4\n");
    fprintf(fp, "rank=%d\n", rank);
    fprintf(fp, "model_name=glm-5.2\n");
    fprintf(fp, "checkpoint_root=%s\n", g_model_root);
    fprintf(fp, "fabric_data_plane=%s\n", DS4_GLM52_L0_FABRIC_DATA_PLANE);
    for (int i = 0; i < base_count; i++) {
        if (foreign_first && i == 0) {
            fprintf(fp, "rank%d.base_shard=rank3/base-00.safetensors\n",
                    rank);
        } else {
            fprintf(fp, "rank%d.base_shard=rank%d/base-%02d.safetensors\n",
                    rank, rank, i);
        }
    }
    fprintf(fp, "rank%d.mtp_shard=rank%d/mtp.safetensors\n", rank, rank);
    fclose(fp);
}

static void ensure_fixture(void) {
    if (g_fixture_ready) return;
    char tmpl[] = "/tmp/ds4-glm-5.2-l0-XXXXXX";
    char *dir = mkdtemp(tmpl);
    check(dir != NULL, "mkdtemp failed");
    snprintf(g_fixture_dir, sizeof(g_fixture_dir), "%s", dir);
    snprintf(g_model_root, sizeof(g_model_root), "%s/model", g_fixture_dir);
    check(mkdir(g_model_root, 0700) == 0, "model fixture mkdir failed");
    char rank2_dir[PATH_MAX];
    char rank3_dir[PATH_MAX];
    snprintf(rank2_dir, sizeof(rank2_dir), "%s/rank2", g_model_root);
    snprintf(rank3_dir, sizeof(rank3_dir), "%s/rank3", g_model_root);
    check(mkdir(rank2_dir, 0700) == 0, "rank2 fixture mkdir failed");
    check(mkdir(rank3_dir, 0700) == 0, "rank3 fixture mkdir failed");
    for (int i = 0; i < DS4_GLM52_L0_EXPECTED_BASE_SHARDS; i++) {
        char path[PATH_MAX];
        snprintf(path, sizeof(path), "%s/base-%02d.safetensors",
                 rank2_dir, i);
        create_file(path);
    }
    char mtp_path[PATH_MAX];
    snprintf(mtp_path, sizeof(mtp_path), "%s/mtp.safetensors", rank2_dir);
    create_file(mtp_path);
    char foreign_path[PATH_MAX];
    snprintf(foreign_path, sizeof(foreign_path),
             "%s/base-00.safetensors", rank3_dir);
    create_file(foreign_path);

    snprintf(g_plan_path, sizeof(g_plan_path), "%s/rank2.plan",
             g_fixture_dir);
    snprintf(g_missing_plan_path, sizeof(g_missing_plan_path),
             "%s/rank2-missing.plan", g_fixture_dir);
    snprintf(g_foreign_plan_path, sizeof(g_foreign_plan_path),
             "%s/rank2-foreign.plan", g_fixture_dir);
    write_rank_plan(g_plan_path,
                    2,
                    DS4_GLM52_L0_EXPECTED_BASE_SHARDS,
                    false);
    write_rank_plan(g_missing_plan_path,
                    2,
                    DS4_GLM52_L0_EXPECTED_BASE_SHARDS - 1,
                    false);
    write_rank_plan(g_foreign_plan_path,
                    2,
                    DS4_GLM52_L0_EXPECTED_BASE_SHARDS,
                    true);
    g_fixture_ready = true;
}

static void capture_trace(void *ud,
                          const char *graph_id,
                          const char *action_id,
                          ds4_glm52_l0_status status,
                          bool prerequisite_blocked,
                          const ds4_glm52_l0_state *state) {
    trace_capture *cap = (trace_capture *)ud;
    check(graph_id && strcmp(graph_id, DS4_GLM52_L0_GRAPH_ID) == 0,
          "trace graph id must be serve");
    check(state != NULL, "trace state must be present");
    check(cap->n < (int)(sizeof(cap->ids) / sizeof(cap->ids[0])),
          "trace capture overflow");
    cap->ids[cap->n] = action_id;
    cap->statuses[cap->n] = status;
    cap->prerequisite_blocked[cap->n] = prerequisite_blocked;
    cap->n++;
}

static ds4_glm52_l0_config valid_config(void) {
    ensure_fixture();
    ds4_engine_options opt = {
        .model_path = g_model_root,
        .glm52_tp4_l0 = true,
        .glm52_tp4_rank_set = true,
        .glm52_tp4_rank = 2,
        .glm52_tp4_rank_plan = g_plan_path,
        .glm52_tp4_fabric_addr = "10.100.185.2",
    };
    ds4_glm52_l0_config cfg;
    char err[128] = {0};
    check(ds4_glm52_l0_config_from_engine(&opt, &cfg, err, sizeof(err)),
          "engine config conversion should succeed");
    check(ds4_glm52_l0_validate_config(&cfg, err, sizeof(err)),
          "default TP4/DCP4/PP1 config should validate");
    check(cfg.tp_size == 4, "tp size should default to 4");
    check(cfg.dcp_size == 4, "dcp size should default to 4");
    check(cfg.pp_size == 1, "pp size should default to 1");
    check(cfg.rank == 2, "rank should be copied");
    check(strcmp(cfg.fabric_addr, "10.100.185.2") == 0,
          "fabric address should be copied");
    return cfg;
}

static const char *fabric_addr_for_rank(int rank) {
    static const char *const addrs[DS4_GLM52_L0_RANK_COUNT] = {
        "10.100.185.3",
        "10.100.185.1",
        "10.100.185.2",
        "10.100.185.4",
    };
    check(rank >= 0 && rank < DS4_GLM52_L0_RANK_COUNT,
          "rank fabric lookup requires valid rank");
    return addrs[rank];
}

static ds4_glm52_l0_config mock_config(void) {
    ds4_glm52_l0_config cfg = valid_config();
    cfg.mock_model = true;
    cfg.mock_matmul = true;
    return cfg;
}

static void mark_resident_shards_for_rank(ds4_glm52_l0_state *state,
                                          int rank,
                                          const char *fabric_addr) {
    state->model_plan.validated = true;
    state->shard_manifest.validated = true;
    state->resident_shards.rank = rank;
    state->resident_shards.base_shard_count =
        DS4_GLM52_L0_EXPECTED_BASE_SHARDS;
    state->resident_shards.mtp_shard_count =
        DS4_GLM52_L0_EXPECTED_MTP_SHARDS;
    state->resident_shards.mapped_bytes = 21;
    state->resident_shards.no_foreign_rank_shard = true;
    state->resident_shards.mapped = true;
    snprintf(state->launch_plan.fabric_data_plane,
             sizeof(state->launch_plan.fabric_data_plane),
             "%s",
             DS4_GLM52_L0_FABRIC_DATA_PLANE);
    snprintf(state->launch_plan.fabric_addr,
             sizeof(state->launch_plan.fabric_addr),
             "%s",
             fabric_addr ? fabric_addr : "");
}

static void mark_resident_shards(ds4_glm52_l0_state *state) {
    mark_resident_shards_for_rank(state, 2, "10.100.185.2");
}

static void mark_tp_group(ds4_glm52_l0_state *state) {
    mark_resident_shards(state);
    state->rank_plan.rank = 2;
    state->rank_plan.q_head_start = 32;
    state->rank_plan.q_head_end = 48;
    state->rank_plan.dcp_rank = 2;
    state->rank_plan.bound = true;
    state->tp_fabric.tp_size = DS4_GLM52_L0_TP_SIZE;
    state->tp_fabric.dcp_size = DS4_GLM52_L0_DCP_SIZE;
    state->tp_fabric.pp_size = DS4_GLM52_L0_PP_SIZE;
    state->tp_fabric.rank_count = DS4_GLM52_L0_RANK_COUNT;
    state->tp_fabric.local_rank = 2;
    state->tp_fabric.topology_bound = true;
    state->tp_fabric.transport_ready = true;
    state->tp_fabric.group_ready = true;
}

static void mark_request(ds4_glm52_l0_state *state) {
    mark_tp_group(state);
    state->request.session_id = "test-session";
    state->request.accepted = true;
}

static void mark_prefill(ds4_glm52_l0_state *state) {
    mark_request(state);
    state->kv.tp_aware = true;
    state->kv.append_ordered = true;
    state->kv.prefill_complete = true;
    state->kv.length = 19;
    state->cursor.replicated = true;
    state->cursor.kv_length = 19;
    state->cursor.token_step_j = 19;
    state->cursor.phase = DS4_GLM52_L0_CURSOR_PHASE_PREFILL;
}

/* Pre-prefill readied state: resident shards + TP4/DCP4 fabric + accepted
 * request + TP-aware append-ordered empty KV, ready to accept a prompt. */
static void mark_readied_for_prefill(ds4_glm52_l0_state *state) {
    mark_request(state);
    state->kv.tp_aware = true;
    state->kv.append_ordered = true;
    state->kv.length = 0;
    state->kv.prefill_complete = false;
    state->cursor.replicated = true;
    state->cursor.kv_length = 0;
    state->cursor.token_step_j = 0;
    state->cursor.phase = DS4_GLM52_L0_CURSOR_PHASE_NONE;
}

static void test_action_order(void) {
    const char *expected[] = {
        "serve/action-serve-open",
        "serve/action-tp-group",
        "serve/action-serve-accept",
        "serve/action-serve-prefill",
        "serve/action-decode-token",
        "serve/action-stream-token",
        "serve/action-kv-checkpoint",
    };
    check(ds4_glm52_l0_action_count() ==
              sizeof(expected) / sizeof(expected[0]),
          "action count must match BCD serve root order");
    for (int i = 0; i < (int)ds4_glm52_l0_action_count(); i++) {
        check(strcmp(ds4_glm52_l0_action_id((ds4_glm52_l0_action)i),
                     expected[i]) == 0,
              "action id order mismatch");
    }
}

static void test_validation(void) {
    ds4_glm52_l0_config cfg = valid_config();
    char err[128] = {0};
    cfg.tp_size = 2;
    check(!ds4_glm52_l0_validate_config(&cfg, err, sizeof(err)),
          "tp_size=2 should be rejected");

    cfg = valid_config();
    cfg.rank = 4;
    check(!ds4_glm52_l0_validate_config(&cfg, err, sizeof(err)),
          "rank 4 should be rejected");

    cfg = valid_config();
    cfg.model_root = "";
    check(!ds4_glm52_l0_validate_config(&cfg, err, sizeof(err)),
          "empty model root should be rejected");

    cfg = valid_config();
    cfg.rank_plan = "";
    check(!ds4_glm52_l0_validate_config(&cfg, err, sizeof(err)),
          "empty rank plan should be rejected");

    cfg = valid_config();
    cfg.rank_set = false;
    check(!ds4_glm52_l0_validate_config(&cfg, err, sizeof(err)),
          "missing explicit rank should be rejected");

    ds4_engine_options gpu_opt = {
        .model_path = g_model_root,
        .glm52_tp4_l0 = true,
        .glm52_tp4_rank_set = true,
        .glm52_tp4_rank = 2,
        .glm52_tp4_rank_plan = g_plan_path,
        .glm52_tp4_gpu_resident = true,
    };
    check(ds4_glm52_l0_config_from_engine(&gpu_opt, &cfg, err, sizeof(err)),
          "GPU-resident engine config conversion should succeed");
    check(cfg.gpu_residency_required && cfg.gpu_device_id == 0,
          "GPU-resident engine option should request rank-local device 0");
    check(ds4_glm52_l0_validate_config(&cfg, err, sizeof(err)),
          "GPU-resident real layout config should validate before runtime injection");

    cfg.mock_model = true;
    cfg.mock_matmul = true;
    check(!ds4_glm52_l0_validate_config(&cfg, err, sizeof(err)),
          "GPU residency should reject mock model mode");

    ds4_engine_options opt = {
        .model_path = "/models/glm-5.2",
        .glm52_tp4_l0 = true,
        .glm52_tp4_rank_set = true,
        .glm52_tp4_rank_plan = "/plans/rank0.textproto",
        .cuda_tensor_parallel = true,
    };
    check(ds4_glm52_l0_config_from_engine(&opt, &cfg, err, sizeof(err)),
          "legacy mixed config conversion should succeed");
    check(!ds4_glm52_l0_validate_config(&cfg, err, sizeof(err)),
          "legacy CUDA TP must not mix with GLM52 TP4 L0");

    opt.cuda_tensor_parallel = false;
    opt.model_path = "/models/glm-5.1";
    check(ds4_glm52_l0_config_from_engine(&opt, &cfg, err, sizeof(err)),
          "wrong GLM version config conversion should succeed");
    check(!ds4_glm52_l0_validate_config(&cfg, err, sizeof(err)),
          "non-GLM-5.2 model root should be rejected");

    opt.model_path = "/models/not-glm";
    check(ds4_glm52_l0_config_from_engine(&opt, &cfg, err, sizeof(err)),
          "false-positive GLM config conversion should succeed");
    check(!ds4_glm52_l0_validate_config(&cfg, err, sizeof(err)),
          "path containing glm without 5.2 should be rejected");
}

static void test_stub_does_not_mutate_cursor(void) {
    ds4_glm52_l0_config cfg = valid_config();
    ds4_glm52_l0_state state = {
        .model_plan.validated = true,
        .rank_plan.bound = true,
        .tp_fabric.group_ready = true,
        .cursor.token_step_j = 19,
        .cursor.kv_length = 19,
        .kv.length = 19,
    };
    ds4_glm52_l0_state before = state;
    ds4_glm52_l0_result result;
    ds4_glm52_l0_status status =
        ds4_glm52_l0_stub_action(DS4_GLM52_L0_ACTION_DECODE_TOKEN,
                                 &cfg,
                                 &state,
                                 &result);
    check(status == DS4_GLM52_L0_STATUS_NOT_READY,
          "decode stub should fail closed with not_ready");
    check(result.action == DS4_GLM52_L0_ACTION_DECODE_TOKEN,
          "decode stub should report decode action");
    check(state.cursor.token_step_j == before.cursor.token_step_j,
          "decode stub must not advance token_step_j");
    check(state.cursor.kv_length == before.cursor.kv_length,
          "decode stub must not append KV");
    check(state.kv.length == before.kv.length,
          "decode stub must not mutate KV length");
    check(state.model_plan.validated == before.model_plan.validated &&
              state.rank_plan.bound == before.rank_plan.bound &&
              state.tp_fabric.group_ready == before.tp_fabric.group_ready,
          "decode stub must not mutate bound states");
}

static void test_orchestration_prerequisites(void) {
    ds4_glm52_l0_config cfg = valid_config();
    ds4_glm52_l0_state state = {0};
    ds4_glm52_l0_result result;
    ds4_glm52_l0_status status =
        ds4_glm52_l0_stub_action(DS4_GLM52_L0_ACTION_TP_GROUP,
                                 &cfg,
                                 &state,
                                 &result);
    check(status == DS4_GLM52_L0_STATUS_NOT_READY,
          "TP group should wait for resident shards");
    check(strstr(result.message, "resident rank-local model shards") != NULL,
          "TP group block should name resident shard prerequisite");

    memset(&state, 0, sizeof(state));
    status = ds4_glm52_l0_stub_action(DS4_GLM52_L0_ACTION_PREFILL,
                                      &cfg,
                                      &state,
                                      &result);
    check(status == DS4_GLM52_L0_STATUS_NOT_READY,
          "prefill should wait for resident shards");
    check(strstr(result.message, "prefill requires resident") != NULL,
          "prefill block should name resident shard prerequisite");

    memset(&state, 0, sizeof(state));
    mark_resident_shards(&state);
    status = ds4_glm52_l0_stub_action(DS4_GLM52_L0_ACTION_PREFILL,
                                      &cfg,
                                      &state,
                                      &result);
    check(status == DS4_GLM52_L0_STATUS_NOT_READY,
          "prefill should wait for TP fabric after shards are resident");
    check(strstr(result.message, "collective fabric") != NULL,
          "prefill block should name TP fabric prerequisite");

    memset(&state, 0, sizeof(state));
    mark_tp_group(&state);
    status = ds4_glm52_l0_stub_action(DS4_GLM52_L0_ACTION_PREFILL,
                                      &cfg,
                                      &state,
                                      &result);
    check(status == DS4_GLM52_L0_STATUS_NOT_READY,
          "prefill should wait for accepted request after TP fabric");
    check(strstr(result.message, "accepted request") != NULL,
          "prefill block should name request prerequisite");

    memset(&state, 0, sizeof(state));
    mark_request(&state);
    status = ds4_glm52_l0_stub_action(DS4_GLM52_L0_ACTION_DECODE_TOKEN,
                                      &cfg,
                                      &state,
                                      &result);
    check(status == DS4_GLM52_L0_STATUS_NOT_READY,
          "decode should wait for prefill-complete KV cursor");
    check(strstr(result.message, "prefill-complete") != NULL,
          "decode block should name prefill cursor prerequisite");

    memset(&state, 0, sizeof(state));
    mark_prefill(&state);
    status = ds4_glm52_l0_stub_action(DS4_GLM52_L0_ACTION_DECODE_TOKEN,
                                      &cfg,
                                      &state,
                                      &result);
    check(status == DS4_GLM52_L0_STATUS_NOT_READY,
          "decode should still be a strict unresolved child stub");
    check(strstr(result.message, "decode-token child graph") != NULL,
          "decode should reach the unresolved child only after prerequisites");

    memset(&state, 0, sizeof(state));
    status = ds4_glm52_l0_stub_action(DS4_GLM52_L0_ACTION_STREAM_TOKEN,
                                      &cfg,
                                      &state,
                                      &result);
    check(status == DS4_GLM52_L0_STATUS_NOT_READY,
          "streaming should wait for sampled token");
    check(strstr(result.message, "sampled decode token") != NULL,
          "stream block should name token prerequisite");
}

static void test_tp_group_binding(void) {
    ds4_glm52_l0_config cfg = valid_config();
    ds4_glm52_l0_state state = {0};
    ds4_glm52_l0_result result;
    mark_resident_shards(&state);
    ds4_glm52_l0_status status =
        ds4_glm52_l0_stub_action(DS4_GLM52_L0_ACTION_TP_GROUP,
                                 &cfg,
                                 &state,
                                 &result);
    check(status == DS4_GLM52_L0_STATUS_NOT_READY,
          "TP group should bind topology but keep transport unresolved");
    check(strstr(result.message, "topology is bound") != NULL,
          "TP group result should report topology-bound state");
    check(state.rank_plan.bound,
          "TP group should bind rank ownership state");
    check(state.rank_plan.rank == 2 &&
              state.rank_plan.q_head_start == 32 &&
              state.rank_plan.q_head_end == 48 &&
              state.rank_plan.dcp_rank == 2,
          "TP group should bind rank-local Q-head and DCP spans");
    check(state.tp_fabric.topology_bound,
          "TP fabric topology should be bound");
    check(!state.tp_fabric.transport_ready &&
              !state.tp_fabric.group_ready,
          "TP fabric should not claim real collective transport readiness");
    check(state.tp_fabric.rank_count == DS4_GLM52_L0_RANK_COUNT &&
              state.tp_fabric.local_rank == 2 &&
              strcmp(state.tp_fabric.fabric_addr, "10.100.185.2") == 0,
          "TP fabric should record local rank and fabric address");
    check(state.tp_fabric.fabric_data_plane &&
              strcmp(state.tp_fabric.fabric_data_plane,
                     DS4_GLM52_L0_FABRIC_DATA_PLANE) == 0,
          "TP fabric should record CRS812 data-plane identity");

    memset(&state, 0, sizeof(state));
    mark_resident_shards(&state);
    cfg.fabric_addr = "";
    state.launch_plan.fabric_addr[0] = '\0';
    status = ds4_glm52_l0_stub_action(DS4_GLM52_L0_ACTION_TP_GROUP,
                                      &cfg,
                                      &state,
                                      &result);
    check(status == DS4_GLM52_L0_STATUS_INVALID,
          "TP group should reject missing fabric address");
    check(strstr(result.message, "fabric") != NULL,
          "missing fabric rejection should name fabric address");

    cfg = valid_config();
    cfg.fabric_addr = "192.168.0.99";
    memset(&state, 0, sizeof(state));
    mark_resident_shards(&state);
    snprintf(state.launch_plan.fabric_data_plane,
             sizeof(state.launch_plan.fabric_data_plane),
             "%s",
             DS4_GLM52_L0_FABRIC_DATA_PLANE);
    status = ds4_glm52_l0_stub_action(DS4_GLM52_L0_ACTION_TP_GROUP,
                                      &cfg,
                                      &state,
                                      &result);
    check(status == DS4_GLM52_L0_STATUS_INVALID,
          "TP group should reject management-network fabric addresses");
    check(strstr(result.message, "management network") != NULL,
          "management-network rejection should be explicit");

    cfg = valid_config();
    memset(&state, 0, sizeof(state));
    mark_resident_shards(&state);
    state.launch_plan.fabric_data_plane[0] = '\0';
    status = ds4_glm52_l0_stub_action(DS4_GLM52_L0_ACTION_TP_GROUP,
                                      &cfg,
                                      &state,
                                      &result);
    check(status == DS4_GLM52_L0_STATUS_INVALID,
          "TP group should reject missing fabric data-plane identity");
    check(strstr(result.message, "fabric_data_plane") != NULL,
          "missing data-plane rejection should name rank-plan field");

    cfg = valid_config();
    memset(&state, 0, sizeof(state));
    mark_resident_shards(&state);
    state.resident_shards.rank = 1;
    status = ds4_glm52_l0_stub_action(DS4_GLM52_L0_ACTION_TP_GROUP,
                                      &cfg,
                                      &state,
                                      &result);
    check(status == DS4_GLM52_L0_STATUS_INVALID,
          "TP group should reject resident shard rank mismatch");
    check(strstr(result.message, "local resident shard rank") != NULL,
          "rank mismatch should name local shard ownership");
}

static int reserve_loopback_port(void) {
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    check(fd >= 0, "loopback port socket should open");
    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    addr.sin_port = 0;
    check(bind(fd, (struct sockaddr *)&addr, sizeof(addr)) == 0,
          "loopback ephemeral bind should succeed");
    socklen_t len = sizeof(addr);
    check(getsockname(fd, (struct sockaddr *)&addr, &len) == 0,
          "loopback ephemeral port should be visible");
    int port = ntohs(addr.sin_port);
    close(fd);
    check(port > 0, "loopback ephemeral port should be nonzero");
    return port;
}

static void tp_group_l0_rank_main(int rank, const char *endpoint) {
    ds4_glm52_l0_config cfg = valid_config();
    cfg.rank = rank;
    cfg.fabric_addr = fabric_addr_for_rank(rank);
    cfg.tp_rendezvous = endpoint;

    ds4_glm52_l0_state state;
    memset(&state, 0, sizeof(state));
    mark_resident_shards_for_rank(&state, rank, cfg.fabric_addr);

    ds4_glm52_l0_result result;
    ds4_glm52_l0_status status =
        ds4_glm52_l0_stub_action(DS4_GLM52_L0_ACTION_TP_GROUP,
                                 &cfg,
                                 &state,
                                 &result);
    if (status != DS4_GLM52_L0_STATUS_OK ||
        !state.rank_plan.bound ||
        !state.tp_fabric.topology_bound ||
        !state.tp_fabric.transport_ready ||
        !state.tp_fabric.group_ready ||
        state.tp_fabric.local_rank != rank) {
        fprintf(stderr,
                "rank %d TP_GROUP failed: status=%s message=%s\n",
                rank,
                ds4_glm52_l0_status_name(status),
                result.message);
        _exit(1);
    }
    _exit(0);
}

static void test_tp_group_l0_rendezvous_publish_ready(void) {
    char endpoint[64];
    snprintf(endpoint,
             sizeof(endpoint),
             "127.0.0.1:%d",
             reserve_loopback_port());

    pid_t pids[DS4_GLM52_L0_RANK_COUNT] = {0};
    pids[0] = fork();
    check(pids[0] >= 0, "TP_GROUP coordinator fork should succeed");
    if (pids[0] == 0) {
        tp_group_l0_rank_main(0, endpoint);
    }

    usleep(100000);
    for (int rank = 1; rank < DS4_GLM52_L0_RANK_COUNT; rank++) {
        pids[rank] = fork();
        check(pids[rank] >= 0, "TP_GROUP worker fork should succeed");
        if (pids[rank] == 0) {
            tp_group_l0_rank_main(rank, endpoint);
        }
    }

    for (int rank = 0; rank < DS4_GLM52_L0_RANK_COUNT; rank++) {
        int status = 0;
        check(waitpid(pids[rank], &status, 0) == pids[rank],
              "TP_GROUP rank waitpid should succeed");
        check(WIFEXITED(status) && WEXITSTATUS(status) == 0,
              "TP_GROUP rank should publish fabric readiness");
    }
}

static void test_model_load_child_validation(void) {
    ds4_glm52_l0_config cfg = valid_config();
    ds4_glm52_l0_state state = {0};
    ds4_glm52_l0_result result;
    ds4_glm52_l0_status status =
        ds4_glm52_l0_stub_action(DS4_GLM52_L0_ACTION_SERVE_OPEN,
                                 &cfg,
                                 &state,
                                 &result);
    check(status == DS4_GLM52_L0_STATUS_NOT_READY,
          "serve-open should stop at unresolved model-load mapper");
    check(result.action == DS4_GLM52_L0_ACTION_SERVE_OPEN,
          "model-load result should project to serve-open");
    check(strstr(result.message, "model-load/a-map-rank-shards") != NULL,
          "serve-open result should name the unresolved model-load child");
    check(state.launch_plan.validated,
          "launch plan should be validated before mapper stub");
    check(state.shard_manifest.validated,
          "shard manifest should be validated before mapper stub");
    check(state.model_plan.validated,
          "model plan state should be marked validated");
    check(state.resident_shards.base_shard_count ==
              DS4_GLM52_L0_EXPECTED_BASE_SHARDS,
          "validated mapping should record 20 base shards");
    check(state.resident_shards.mtp_shard_count ==
              DS4_GLM52_L0_EXPECTED_MTP_SHARDS,
          "validated mapping should record one MTP shard");
    check(state.resident_shards.mapped_bytes ==
              (uint64_t)(DS4_GLM52_L0_EXPECTED_BASE_SHARDS +
                         DS4_GLM52_L0_EXPECTED_MTP_SHARDS),
          "validated mapping should record bytes from concrete files");
    check(!state.resident_shards.mapped,
          "validated manifest must not claim tensor mmap is implemented");
    check(state.resident_shards.no_foreign_rank_shard,
          "validated manifest should record no foreign-rank shard");
}

static void test_model_load_manifest_rejections(void) {
    ds4_glm52_l0_config cfg = valid_config();
    ds4_glm52_l0_state state = {0};
    ds4_glm52_l0_result result;

    cfg.rank_plan = g_missing_plan_path;
    ds4_glm52_l0_status status =
        ds4_glm52_l0_stub_action(DS4_GLM52_L0_ACTION_SERVE_OPEN,
                                 &cfg,
                                 &state,
                                 &result);
    check(status == DS4_GLM52_L0_STATUS_INVALID,
          "missing base shard count should be rejected");
    check(strstr(result.message,
                 "model-load/a-validate-shard-manifest") != NULL,
          "missing shard rejection should name shard manifest validation");

    memset(&state, 0, sizeof(state));
    cfg.rank_plan = g_foreign_plan_path;
    status = ds4_glm52_l0_stub_action(DS4_GLM52_L0_ACTION_SERVE_OPEN,
                                      &cfg,
                                      &state,
                                      &result);
    check(status == DS4_GLM52_L0_STATUS_INVALID,
          "foreign rank shard should be rejected");
    check(strstr(result.message, "no foreign-rank shard") != NULL,
          "foreign shard rejection should explain ownership violation");
}

static void test_run_reaches_root_seam(void) {
    ds4_glm52_l0_config cfg = valid_config();
    ds4_glm52_l0_state state;
    ds4_glm52_l0_result result;
    trace_capture cap = {0};
    ds4_glm52_l0_status status =
        ds4_glm52_l0_run_skeleton(&cfg,
                                  &state,
                                  &result,
                                  capture_trace,
                                  &cap);
    check(status == DS4_GLM52_L0_STATUS_NOT_READY,
          "skeleton should stop at first not-ready production seam");
    check(cap.n == 1, "strict skeleton should trace only the first blocked action");
    check(strcmp(cap.ids[0], "serve/action-serve-open") == 0,
          "first traced action should be serve-open");
    check(!cap.prerequisite_blocked[0],
          "first strict trace action should not have prior blocked prerequisite");
    check(result.action == DS4_GLM52_L0_ACTION_SERVE_OPEN,
          "blocked action should be serve-open");
    check(state.cursor.token_step_j == 0 && state.cursor.kv_length == 0,
          "root seam must not mutate cursor state");
    check(state.model_plan.tp_size == 4 &&
              state.tp_fabric.dcp_size == 4 &&
              state.rank_plan.q_head_start == 32 &&
              state.rank_plan.q_head_end == 48,
          "root seam should bind typed L0 placeholder state from config");
}

static void test_review_reaches_all_root_actions(void) {
    const char *expected[] = {
        "serve/action-serve-open",
        "serve/action-tp-group",
        "serve/action-serve-accept",
        "serve/action-serve-prefill",
        "serve/action-decode-token",
        "serve/action-stream-token",
        "serve/action-kv-checkpoint",
    };
    ds4_glm52_l0_config cfg = valid_config();
    ds4_glm52_l0_state state;
    ds4_glm52_l0_result result;
    trace_capture cap = {0};
    ds4_glm52_l0_status status =
        ds4_glm52_l0_review_skeleton(&cfg,
                                     &state,
                                     &result,
                                     capture_trace,
                                     &cap);
    check(status == DS4_GLM52_L0_STATUS_NOT_READY,
          "review skeleton should still fail closed");
    check(cap.n == (int)(sizeof(expected) / sizeof(expected[0])),
          "review skeleton should trace every L0 root action");
    for (int i = 0; i < cap.n; i++) {
        check(strcmp(cap.ids[i], expected[i]) == 0,
              "review action order mismatch");
        check(cap.prerequisite_blocked[i] == (i > 0),
              "review trace should mark actions after first block as review-only enumeration");
    }
    check(result.action == DS4_GLM52_L0_ACTION_SERVE_OPEN,
          "review result should report the first unresolved action");
    check(state.cursor.token_step_j == 0 &&
              state.cursor.kv_length == 0 &&
              state.kv.length == 0,
          "review traversal must not mutate KV/cursor");
}

/* ---- model-shard-layout tests ---- */

static char g_layout_dir[PATH_MAX];
static char g_layout_valid_path[PATH_MAX];
static char g_layout_wrong_tp_path[PATH_MAX];
static char g_layout_foreign_path[PATH_MAX];
static char g_layout_missing_qhead_path[PATH_MAX];
static char g_layout_overlap_qhead_path[PATH_MAX];
static char g_layout_missing_file_path[PATH_MAX];
static char g_layout_bad_sha_path[PATH_MAX];
static char g_layout_missing_metadata_path[PATH_MAX];
static char g_layout_bad_metadata_bytes_path[PATH_MAX];
static bool g_layout_fixture_ready;

static const char *k_sha256_x32 =
    "c62e4615bd39e222572f3a1bf7c2132ea1e65b17ec805047bd6b2842c593493f";

static void write_layout_header(FILE *fp) {
    fprintf(fp, "format_version=ds4-shard-layout/v1\n");
    fprintf(fp, "model_config_sha256=abc123fakehash\n");
    fprintf(fp, "tp_size=4\n");
    fprintf(fp, "dcp_size=4\n");
    fprintf(fp, "rank_count=4\n");
    fprintf(fp, "required_base_shards=20\n");
    fprintf(fp, "required_mtp_shards=1\n");
    fprintf(fp, "has_mtp=true\n");
}

static void write_layout_entry(FILE *fp, const char *name, const char *role,
                               const char *scope, int rank,
                               const char *file_path,
                               uint64_t offset, uint64_t length,
                               const char *sha,
                               int qhs, int qhe,
                               int es, int ee, int vs, int ve) {
    fprintf(fp, "[entry]\n");
    fprintf(fp, "tensor_name=%s\n", name);
    fprintf(fp, "role=%s\n", role);
    fprintf(fp, "distribution_scope=%s\n", scope);
    fprintf(fp, "rank=%d\n", rank);
    fprintf(fp, "file_path=%s\n", file_path);
    fprintf(fp, "byte_offset=%llu\n", (unsigned long long)offset);
    fprintf(fp, "byte_length=%llu\n", (unsigned long long)length);
    fprintf(fp, "sha256=%s\n", sha);
    fprintf(fp, "replicated=%s\n", (rank < 0) ? "true" : "false");
    fprintf(fp, "dtype=bf16\n");
    fprintf(fp, "shape=16\n");
    fprintf(fp, "element_count=16\n");
    if (qhs >= 0) fprintf(fp, "q_head_start=%d\n", qhs);
    if (qhe >= 0) fprintf(fp, "q_head_end=%d\n", qhe);
    if (es >= 0) fprintf(fp, "expert_start=%d\n", es);
    if (ee >= 0) fprintf(fp, "expert_end=%d\n", ee);
    if (vs >= 0) fprintf(fp, "vocab_start=%d\n", vs);
    if (ve >= 0) fprintf(fp, "vocab_end=%d\n", ve);
}

/* Create a shard file of at least 'size' bytes. */
static void create_shard_file(const char *dir, const char *name, int size) {
    char path[PATH_MAX];
    snprintf(path, sizeof(path), "%s/%s", dir, name);
    FILE *fp = fopen(path, "wb");
    check(fp != NULL, "layout fixture shard file open failed");
    for (int i = 0; i < size; i++) fputc('x', fp);
    fclose(fp);
}

static void ensure_layout_fixture(void) {
    if (g_layout_fixture_ready) return;
    ensure_fixture();

    snprintf(g_layout_dir, sizeof(g_layout_dir), "%s/layout", g_fixture_dir);
    check(mkdir(g_layout_dir, 0700) == 0, "layout fixture mkdir failed");

    /* Create shard files for rank 2 (16 bytes each for byte-range tests). */
    char rank2_dir[PATH_MAX];
    snprintf(rank2_dir, sizeof(rank2_dir), "%s/rank2", g_model_root);
    for (int i = 0; i < 5; i++) {
        char name[64];
        snprintf(name, sizeof(name), "shard-%02d.bin", i);
        create_shard_file(rank2_dir, name, 64);
    }
    /* Replicated shard in model root. */
    create_shard_file(g_model_root, "shared.bin", 64);

    /* Valid layout: rank 2, Q heads [32,48), plus replicated tensor. */
    snprintf(g_layout_valid_path, sizeof(g_layout_valid_path),
             "%s/valid.layout", g_layout_dir);
    {
        FILE *fp = fopen(g_layout_valid_path, "w");
        check(fp != NULL, "valid layout open failed");
        write_layout_header(fp);
        write_layout_entry(fp, "q_proj_b", "q_head", "rank_local_shard",
                           2, "rank2/shard-00.bin", 16, 32, k_sha256_x32,
                           32, 48, -1, -1, -1, -1);
        write_layout_entry(fp, "mla_kv_b", "mla_kv", "rank_local_shard",
                           2, "rank2/shard-01.bin", 0, 32, k_sha256_x32,
                           -1, -1, -1, -1, -1, -1);
        write_layout_entry(fp, "expert_0", "expert", "rank_local_shard",
                           2, "rank2/shard-02.bin", 0, 32, k_sha256_x32,
                           -1, -1, 0, 256, -1, -1);
        write_layout_entry(fp, "vocab_w", "vocab", "rank_local_shard",
                           2, "rank2/shard-03.bin", 0, 32, k_sha256_x32,
                           -1, -1, -1, -1, 77440, 116160);
        write_layout_entry(fp, "shared_emb", "replicated", "replicated",
                           -1, "shared.bin", 0, 32, k_sha256_x32,
                           -1, -1, -1, -1, -1, -1);
        fclose(fp);
    }

    /* Wrong tp_size. */
    snprintf(g_layout_wrong_tp_path, sizeof(g_layout_wrong_tp_path),
             "%s/wrong_tp.layout", g_layout_dir);
    {
        FILE *fp = fopen(g_layout_wrong_tp_path, "w");
        check(fp != NULL, "wrong_tp layout open failed");
        write_layout_header(fp);
        /* Override tp_size. */
        fseek(fp, 0, SEEK_SET);
        fprintf(fp, "format_version=ds4-shard-layout/v1\n");
        fprintf(fp, "model_config_sha256=abc123fakehash\n");
        fprintf(fp, "tp_size=2\n");
        fprintf(fp, "dcp_size=4\n");
        fprintf(fp, "rank_count=4\n");
        fprintf(fp, "required_base_shards=20\n");
        fprintf(fp, "required_mtp_shards=1\n");
        fprintf(fp, "has_mtp=true\n");
        write_layout_entry(fp, "q_proj_b", "q_head", "rank_local_shard",
                           2, "rank2/shard-00.bin", 0, 32, "hash0",
                           32, 48, -1, -1, -1, -1);
        fclose(fp);
    }

    /* Foreign rank shard: entry owned by rank 0 but we are rank 2. */
    snprintf(g_layout_foreign_path, sizeof(g_layout_foreign_path),
             "%s/foreign.layout", g_layout_dir);
    {
        FILE *fp = fopen(g_layout_foreign_path, "w");
        check(fp != NULL, "foreign layout open failed");
        write_layout_header(fp);
        write_layout_entry(fp, "q_proj_b", "q_head", "rank_local_shard",
                           2, "rank2/shard-00.bin", 0, 32, "hash0",
                           32, 48, -1, -1, -1, -1);
        write_layout_entry(fp, "expert_0", "expert", "rank_local_shard",
                           0, "rank0/shard-04.bin", 0, 32, "hash5",
                           -1, -1, 0, 64, -1, -1);
        fclose(fp);
    }

    /* Missing Q-head span: Q-head entry exists but only covers [32,40). */
    snprintf(g_layout_missing_qhead_path, sizeof(g_layout_missing_qhead_path),
             "%s/missing_qhead.layout", g_layout_dir);
    {
        FILE *fp = fopen(g_layout_missing_qhead_path, "w");
        check(fp != NULL, "missing_qhead layout open failed");
        write_layout_header(fp);
        write_layout_entry(fp, "q_proj_b", "q_head", "rank_local_shard",
                           2, "rank2/shard-00.bin", 0, 32, "hash0",
                           32, 40, -1, -1, -1, -1);
        fclose(fp);
    }

    /* Overlapping Q-head span: two entries covering [32,40) and [36,48). */
    snprintf(g_layout_overlap_qhead_path, sizeof(g_layout_overlap_qhead_path),
             "%s/overlap_qhead.layout", g_layout_dir);
    {
        FILE *fp = fopen(g_layout_overlap_qhead_path, "w");
        check(fp != NULL, "overlap_qhead layout open failed");
        write_layout_header(fp);
        write_layout_entry(fp, "q_proj_b_a", "q_head", "rank_local_shard",
                           2, "rank2/shard-00.bin", 0, 32, "hash0",
                           32, 40, -1, -1, -1, -1);
        write_layout_entry(fp, "q_proj_b_b", "q_head", "rank_local_shard",
                           2, "rank2/shard-01.bin", 0, 32, "hash1",
                           36, 48, -1, -1, -1, -1);
        fclose(fp);
    }

    /* Missing required shard file: file_path points at nonexistent file. */
    snprintf(g_layout_missing_file_path, sizeof(g_layout_missing_file_path),
             "%s/missing_file.layout", g_layout_dir);
    {
        FILE *fp = fopen(g_layout_missing_file_path, "w");
        check(fp != NULL, "missing_file layout open failed");
        write_layout_header(fp);
        write_layout_entry(fp, "q_proj_b", "q_head", "rank_local_shard",
                           2, "rank2/nonexistent.bin", 0, 32, k_sha256_x32,
                           32, 48, -1, -1, -1, -1);
        write_layout_entry(fp, "mla_kv_b", "mla_kv", "rank_local_shard",
                           2, "rank2/shard-01.bin", 0, 32, k_sha256_x32,
                           -1, -1, -1, -1, -1, -1);
        write_layout_entry(fp, "expert_0", "expert", "rank_local_shard",
                           2, "rank2/shard-02.bin", 0, 32, k_sha256_x32,
                           -1, -1, 0, 256, -1, -1);
        write_layout_entry(fp, "vocab_w", "vocab", "rank_local_shard",
                           2, "rank2/shard-03.bin", 0, 32, k_sha256_x32,
                           -1, -1, -1, -1, 77440, 116160);
        fclose(fp);
    }

    /* Bad sha256: file exists and range is valid, but the digest is wrong. */
    snprintf(g_layout_bad_sha_path, sizeof(g_layout_bad_sha_path),
             "%s/bad_sha.layout", g_layout_dir);
    {
        FILE *fp = fopen(g_layout_bad_sha_path, "w");
        check(fp != NULL, "bad_sha layout open failed");
        write_layout_header(fp);
        write_layout_entry(fp, "q_proj_b", "q_head", "rank_local_shard",
                           2, "rank2/shard-00.bin", 0, 32,
                           "0000000000000000000000000000000000000000000000000000000000000000",
                           32, 48, -1, -1, -1, -1);
        write_layout_entry(fp, "expert_0", "expert", "rank_local_shard",
                           2, "rank2/shard-02.bin", 0, 32, k_sha256_x32,
                           -1, -1, 0, 256, -1, -1);
        write_layout_entry(fp, "vocab_w", "vocab", "rank_local_shard",
                           2, "rank2/shard-03.bin", 0, 32, k_sha256_x32,
                           -1, -1, -1, -1, 77440, 116160);
        fclose(fp);
    }

    /* Missing tensor metadata: read_manifest must reject before ownership. */
    snprintf(g_layout_missing_metadata_path,
             sizeof(g_layout_missing_metadata_path),
             "%s/missing_metadata.layout", g_layout_dir);
    {
        FILE *fp = fopen(g_layout_missing_metadata_path, "w");
        check(fp != NULL, "missing_metadata layout open failed");
        write_layout_header(fp);
        fprintf(fp, "[entry]\n");
        fprintf(fp, "tensor_name=q_proj_b\n");
        fprintf(fp, "role=q_head\n");
        fprintf(fp, "distribution_scope=rank_local_shard\n");
        fprintf(fp, "rank=2\n");
        fprintf(fp, "file_path=rank2/shard-00.bin\n");
        fprintf(fp, "byte_offset=0\n");
        fprintf(fp, "byte_length=32\n");
        fprintf(fp, "sha256=%s\n", k_sha256_x32);
        fprintf(fp, "replicated=false\n");
        fprintf(fp, "q_head_start=32\n");
        fprintf(fp, "q_head_end=48\n");
        fclose(fp);
    }

    /* Bad tensor metadata: bf16 shape=16 requires 32 bytes, not 30. */
    snprintf(g_layout_bad_metadata_bytes_path,
             sizeof(g_layout_bad_metadata_bytes_path),
             "%s/bad_metadata_bytes.layout", g_layout_dir);
    {
        FILE *fp = fopen(g_layout_bad_metadata_bytes_path, "w");
        check(fp != NULL, "bad_metadata_bytes layout open failed");
        write_layout_header(fp);
        write_layout_entry(fp, "q_proj_b", "q_head", "rank_local_shard",
                           2, "rank2/shard-00.bin", 0, 30, k_sha256_x32,
                           32, 48, -1, -1, -1, -1);
        fclose(fp);
    }

    g_layout_fixture_ready = true;
}

static void test_layout_valid_manifest_passes(void) {
    ensure_layout_fixture();
    ds4_glm52_layout_spec spec;
    ds4_glm52_l0_result result;
    ds4_glm52_l0_status st =
        ds4_glm52_layout_read_manifest(g_layout_valid_path, &spec, &result);
    check(st == DS4_GLM52_L0_STATUS_OK,
          "valid layout read should succeed");
    check(spec.validated, "valid layout spec should be validated");
    check(spec.tp_size == 4, "valid layout tp_size should be 4");
    check(spec.entry_count == 5, "valid layout should have 5 entries");
    check(spec.entries[0].dtype == DS4_GLM52_TP4_TENSOR_DTYPE_BF16 &&
              spec.entries[0].shape_count == 1 &&
              spec.entries[0].shape[0] == 16 &&
              spec.entries[0].element_count == 16 &&
              spec.entries[0].shape_hash != 0,
          "valid layout should derive typed tensor metadata");

    ds4_glm52_layout_ownership_plan plan;
    st = ds4_glm52_layout_validate_ownership(&spec, 2, &plan, &result);
    check(st == DS4_GLM52_L0_STATUS_OK,
          "valid layout ownership should succeed");
    check(plan.coverage_complete, "valid layout coverage should be complete");
    check(!plan.overlaps_present, "valid layout should have no overlaps");
    check(!plan.missing_spans_present, "valid layout should have no missing spans");
    check(plan.q_head_start == 32, "rank 2 Q-head start should be 32");
    check(plan.q_head_end == 48, "rank 2 Q-head end should be 48");
    check(plan.expert_start == 0, "rank 2 expert start should be 0");
    check(plan.expert_end == 256, "rank 2 expert end should be 256");
    check(plan.vocab_start == 77440, "rank 2 vocab start should be 77440");
    check(plan.vocab_end == 116160, "rank 2 vocab end should be 116160");
    check(plan.entry_count == 5, "valid plan should have 5 visible entries");

    ds4_glm52_layout_mapped_slices mapped;
    st = ds4_glm52_layout_mmap_rank_tensors(&plan, g_model_root,
                                            &mapped, &result);
    check(st == DS4_GLM52_L0_STATUS_OK,
          "valid layout mmap should succeed");
    check(mapped.mapped, "valid layout should be mapped");
    check(mapped.no_foreign_rank_shard, "valid layout should have no foreign shards");
    check(mapped.tensor_count == 5, "valid layout should map 5 tensors");
    check(mapped.mapped_bytes > 0, "valid layout should map non-zero bytes");
    check(mapped.hash_verified, "valid layout should verify tensor sha256");
    check(strcmp(mapped.source_layout_sha256, "verified-entry-sha256") == 0,
          "valid layout should record verified source identity");
    check(mapped.tensors[0].mapped &&
              mapped.tensors[0].map_base != NULL &&
              mapped.tensors[0].data != NULL &&
              mapped.tensors[0].data_bytes == 32 &&
              mapped.tensors[0].byte_offset == 16,
          "valid layout should publish a real mmap handle for each tensor");
    check(mapped.tensors[0].dtype == DS4_GLM52_TP4_TENSOR_DTYPE_BF16 &&
              mapped.tensors[0].shape_count == 1 &&
              mapped.tensors[0].shape[0] == 16 &&
              mapped.tensors[0].element_count == 16 &&
              mapped.tensors[0].shape_hash == spec.entries[0].shape_hash,
          "mapped tensor should retain validated dtype and shape metadata");
    check(mapped.tensors[0].data[0] == 'x' &&
              mapped.tensors[0].data[31] == 'x',
          "mapped tensor data should be readable at the declared slice");

    ds4_glm52_l0_state state = {0};
    st = ds4_glm52_layout_publish_loaded_rank_shard(&mapped, &plan,
                                                     &state, &result);
    check(st == DS4_GLM52_L0_STATUS_OK,
          "valid layout publish should succeed");
    check(state.resident_shards.mapped, "publish should set mapped flag");
    check(state.resident_shards.no_foreign_rank_shard,
          "publish should set no_foreign_rank_shard");
    check(state.rank_plan.q_head_start == 32, "publish should set q_head_start");
    check(state.rank_plan.q_head_end == 48, "publish should set q_head_end");
    check(state.rank_plan.expert_start == 0,
          "publish should set expert_start");
    check(state.rank_plan.expert_end == 256,
          "publish should set expert_end");
    check(state.rank_plan.vocab_start == 77440,
          "publish should set vocab_start");
    check(state.rank_plan.vocab_end == 116160,
          "publish should set vocab_end");
    check(state.rank_plan.bound, "publish should bind rank plan");
    check(mapped.tensor_count == 0 && !mapped.mapped,
          "publish should transfer mapped slice ownership to resident state");
    check(state.resident_shards.tensor_count == 5,
          "resident state should retain mapped tensor records");
    check(state.resident_shards.tensors[0].mapped &&
              state.resident_shards.tensors[0].data != NULL &&
              state.resident_shards.tensors[0].data[0] == 'x' &&
              state.resident_shards.tensors[0].dtype ==
                  DS4_GLM52_TP4_TENSOR_DTYPE_BF16 &&
              state.resident_shards.tensors[0].shape_hash ==
                  spec.entries[0].shape_hash,
          "resident mapped tensor should retain readable bytes and metadata");

    g_fake_gpu_allocs = 0;
    g_fake_gpu_frees = 0;
    st = ds4_glm52_layout_upload_resident_gpu_tensors(
            &state, 7, &k_fake_gpu_runtime, &result);
    check(st == DS4_GLM52_L0_STATUS_OK,
          "resident GPU tensor upload should succeed");
    check(state.resident_shards.gpu_resident &&
              state.resident_shards.gpu_tensor_count ==
                  state.resident_shards.tensor_count &&
              state.resident_shards.gpu_bytes ==
                  state.resident_shards.mapped_bytes,
          "GPU upload should publish one ready device tensor per mapped tensor");
    check(state.resident_shards.gpu_tensors[0].ready &&
              state.resident_shards.gpu_tensors[0].device_id == 7 &&
              state.resident_shards.gpu_tensors[0].rank == 2 &&
              state.resident_shards.gpu_tensors[0].byte_count == 32 &&
              state.resident_shards.gpu_tensors[0].dtype ==
                  DS4_GLM52_TP4_TENSOR_DTYPE_BF16 &&
              state.resident_shards.gpu_tensors[0].shape_hash ==
                  spec.entries[0].shape_hash,
          "GPU tensor should retain rank, device, byte, dtype, and shape metadata");
    check(((unsigned char *)state.resident_shards.gpu_tensors[0].tensor.ptr)[0] == 'x' &&
              ((unsigned char *)state.resident_shards.gpu_tensors[0].tensor.ptr)[31] == 'x',
          "GPU upload should copy the exact mapped tensor payload");
    check(g_fake_gpu_allocs == state.resident_shards.gpu_tensor_count,
          "GPU upload should allocate one tensor per mapped tensor");
    ds4_glm52_l0_unmap_resident_rank_shards(&state);
    check(!state.resident_shards.mapped &&
              state.resident_shards.tensor_count == 0 &&
              !state.resident_shards.gpu_resident &&
              state.resident_shards.gpu_tensor_count == 0,
          "resident unmap should clear mapped and GPU tensor records");
    check(g_fake_gpu_frees == g_fake_gpu_allocs,
          "resident unmap should release uploaded GPU tensors");
}

static void test_layout_gpu_upload_missing_metadata_fails(void) {
    ensure_layout_fixture();
    ds4_glm52_layout_spec spec;
    ds4_glm52_l0_result result;
    ds4_glm52_l0_status st =
        ds4_glm52_layout_read_manifest(g_layout_valid_path, &spec, &result);
    check(st == DS4_GLM52_L0_STATUS_OK,
          "valid layout manifest should parse");

    ds4_glm52_layout_ownership_plan plan;
    st = ds4_glm52_layout_validate_ownership(&spec, 2, &plan, &result);
    check(st == DS4_GLM52_L0_STATUS_OK,
          "valid ownership should parse before corrupting metadata");
    ds4_glm52_layout_mapped_slices mapped;
    st = ds4_glm52_layout_mmap_rank_tensors(
            &plan, g_model_root, &mapped, &result);
    check(st == DS4_GLM52_L0_STATUS_OK,
          "valid layout mmap should succeed before corrupting metadata");

    ds4_glm52_l0_state state = {0};
    st = ds4_glm52_layout_publish_loaded_rank_shard(
            &mapped, &plan, &state, &result);
    check(st == DS4_GLM52_L0_STATUS_OK,
          "valid layout publish should succeed before corrupting metadata");

    state.resident_shards.tensors[0].shape_hash = 0;
    g_fake_gpu_allocs = 0;
    g_fake_gpu_frees = 0;
    st = ds4_glm52_layout_upload_resident_gpu_tensors(
            &state, 7, &k_fake_gpu_runtime, &result);
    check(st == DS4_GLM52_L0_STATUS_INVALID,
          "GPU upload should reject missing runtime tensor metadata");
    check(strstr(result.message, "shape hash") != NULL ||
              strstr(result.message, "metadata") != NULL,
          "GPU upload metadata rejection should name metadata");
    check(!state.resident_shards.gpu_resident &&
              state.resident_shards.gpu_tensor_count == 0 &&
              g_fake_gpu_allocs == 0 &&
              g_fake_gpu_frees == 0,
          "failed GPU upload should not leave resident device tensors");
    ds4_glm52_l0_unmap_resident_rank_shards(&state);
}

static void test_model_load_layout_reaches_ready_rank_engines(void) {
    ensure_layout_fixture();
    ds4_glm52_l0_config cfg = valid_config();
    cfg.layout_path = g_layout_valid_path;
    ds4_glm52_l0_state state = {0};
    ds4_glm52_l0_result result;
    ds4_glm52_l0_status status =
        ds4_glm52_l0_stub_action(DS4_GLM52_L0_ACTION_SERVE_OPEN,
                                 &cfg,
                                 &state,
                                 &result);
    check(status == DS4_GLM52_L0_STATUS_OK,
          "serve-open should reach ready rank engines after valid layout mapping");
    check(result.action == DS4_GLM52_L0_ACTION_SERVE_OPEN,
          "ready model-load result should project to serve-open");
    check(strstr(result.message, "a-ready-rank-engines") != NULL,
          "ready model-load result should name the ready-rank-engines action");
    check(state.launch_plan.validated &&
              state.shard_manifest.validated &&
              state.model_plan.validated,
          "serve-open should validate launch, manifest, and model plan");
    check(state.resident_shards.mapped &&
              state.resident_shards.no_foreign_rank_shard &&
              state.resident_shards.base_shard_count ==
                  DS4_GLM52_L0_EXPECTED_BASE_SHARDS &&
              state.resident_shards.mtp_shard_count ==
                  DS4_GLM52_L0_EXPECTED_MTP_SHARDS &&
              state.resident_shards.mapped_bytes > 0,
          "serve-open should publish resident rank-local shard readiness");
    check(state.rank_plan.bound &&
              state.rank_plan.rank == 2 &&
              state.rank_plan.q_head_start == 32 &&
              state.rank_plan.q_head_end == 48 &&
              state.rank_plan.expert_start == 0 &&
              state.rank_plan.expert_end == 256 &&
              state.rank_plan.vocab_start == 77440 &&
              state.rank_plan.vocab_end == 116160,
          "serve-open should bind the rank-local Q-head/expert/vocab layout");
}

static void test_model_load_layout_gpu_residency_reaches_ready_rank_engines(void) {
    ensure_layout_fixture();
    ds4_glm52_l0_config cfg = valid_config();
    cfg.layout_path = g_layout_valid_path;
    cfg.gpu_residency_required = true;
    cfg.gpu_runtime = &k_fake_gpu_runtime;
    cfg.gpu_device_id = 7;
    ds4_glm52_l0_state state = {0};
    ds4_glm52_l0_result result;
    g_fake_gpu_allocs = 0;
    g_fake_gpu_frees = 0;
    ds4_glm52_l0_status status =
        ds4_glm52_l0_stub_action(DS4_GLM52_L0_ACTION_SERVE_OPEN,
                                 &cfg,
                                 &state,
                                 &result);
    check(status == DS4_GLM52_L0_STATUS_OK,
          "serve-open should reach ready rank engines after GPU upload");
    check(state.resident_shards.mapped &&
              state.resident_shards.gpu_resident &&
              state.resident_shards.gpu_tensor_count ==
                  state.resident_shards.tensor_count &&
              state.resident_shards.gpu_bytes ==
                  state.resident_shards.mapped_bytes,
          "serve-open should publish GPU-resident rank-local tensor bindings");
    check(state.resident_shards.gpu_tensors[0].ready &&
              state.resident_shards.gpu_tensors[0].device_id == 7 &&
              ((unsigned char *)state.resident_shards.gpu_tensors[0].tensor.ptr)[0] == 'x',
          "serve-open GPU upload should retain binding metadata and payload");
    ds4_glm52_l0_unmap_resident_rank_shards(&state);
    check(g_fake_gpu_frees == g_fake_gpu_allocs,
          "serve-open cleanup should release GPU-resident tensors");
}

static void test_model_load_layout_gpu_residency_requires_runtime(void) {
    ensure_layout_fixture();
    ds4_glm52_l0_config cfg = valid_config();
    cfg.layout_path = g_layout_valid_path;
    cfg.gpu_residency_required = true;
    cfg.gpu_runtime = NULL;
    ds4_glm52_l0_state state = {0};
    ds4_glm52_l0_result result;
    ds4_glm52_l0_status status =
        ds4_glm52_l0_stub_action(DS4_GLM52_L0_ACTION_SERVE_OPEN,
                                 &cfg,
                                 &state,
                                 &result);
    check(status == DS4_GLM52_L0_STATUS_INVALID,
          "serve-open GPU residency should fail closed without runtime");
    check(strstr(result.message, "CUDA tensor runtime") != NULL,
          "missing GPU runtime error should name CUDA tensor runtime");
    check(state.resident_shards.mapped &&
              !state.resident_shards.gpu_resident &&
              state.resident_shards.gpu_tensor_count == 0,
          "missing runtime should not publish GPU-resident tensor bindings");
    ds4_glm52_l0_unmap_resident_rank_shards(&state);
}

static void test_layout_wrong_tp_fails(void) {
    ensure_layout_fixture();
    ds4_glm52_layout_spec spec;
    ds4_glm52_l0_result result;
    ds4_glm52_l0_status st =
        ds4_glm52_layout_read_manifest(g_layout_wrong_tp_path, &spec, &result);
    check(st == DS4_GLM52_L0_STATUS_INVALID,
          "wrong tp_size layout should be rejected");
    check(strstr(result.message, "tp_size must be 4") != NULL,
          "wrong tp_size should be named in error");
}

static void test_layout_foreign_rank_fails(void) {
    ensure_layout_fixture();
    ds4_glm52_layout_spec spec;
    ds4_glm52_l0_result result;
    ds4_glm52_l0_status st =
        ds4_glm52_layout_read_manifest(g_layout_foreign_path, &spec, &result);
    check(st == DS4_GLM52_L0_STATUS_OK,
          "foreign layout header should parse");
    ds4_glm52_layout_ownership_plan plan;
    st = ds4_glm52_layout_validate_ownership(&spec, 2, &plan, &result);
    check(st == DS4_GLM52_L0_STATUS_INVALID,
          "foreign-owned entry should be rejected in per-rank layout");
    check(strstr(result.message, "foreign rank") != NULL,
          "foreign-owned entry rejection should name foreign rank");
    bool found_foreign = false;
    for (int i = 0; i < plan.entry_count; i++) {
        if (!strcmp(plan.entries[i].tensor_name, "expert_0") &&
            plan.entries[i].rank == 0) {
            found_foreign = true;
            break;
        }
    }
    check(!found_foreign,
          "foreign-rank entry must not appear in current-rank plan");

    /* Now test the actual rejection: a current-rank entry with a foreign-rank
     * file path. */
    ds4_glm52_layout_spec spec2 = {0};
    snprintf(spec2.format_version, sizeof(spec2.format_version),
             "%s", DS4_GLM52_LAYOUT_FORMAT_VERSION);
    snprintf(spec2.model_config_sha256,
             sizeof(spec2.model_config_sha256), "abc123");
    spec2.tp_size = 4;
    spec2.dcp_size = 4;
    spec2.rank_count = 4;
    spec2.required_base_shards = 20;
    spec2.required_mtp_shards = 1;
    spec2.validated = true;
    ds4_glm52_layout_entry e = {0};
    snprintf(e.tensor_name, sizeof(e.tensor_name), "bad_entry");
    e.role = DS4_GLM52_LAYOUT_ROLE_Q_HEAD;
    e.scope = DS4_GLM52_LAYOUT_SCOPE_RANK_LOCAL;
    e.rank = 2;
    snprintf(e.file_path, sizeof(e.file_path), "rank0/bad.bin");
    e.byte_length = 32;
    snprintf(e.sha256, sizeof(e.sha256), "hash0");
    e.q_head_start = 32;
    e.q_head_end = 48;
    spec2.entries[spec2.entry_count++] = e;
    ds4_glm52_layout_ownership_plan plan2;
    st = ds4_glm52_layout_validate_ownership(&spec2, 2, &plan2, &result);
    check(st == DS4_GLM52_L0_STATUS_INVALID,
          "current-rank entry with foreign-rank file path should be rejected");
    check(strstr(result.message, "foreign-rank file path") != NULL,
          "foreign-rank file path should be named in error");
}

static void test_layout_missing_qhead_fails(void) {
    ensure_layout_fixture();
    ds4_glm52_layout_spec spec;
    ds4_glm52_l0_result result;
    ds4_glm52_l0_status st =
        ds4_glm52_layout_read_manifest(g_layout_missing_qhead_path, &spec, &result);
    check(st == DS4_GLM52_L0_STATUS_OK,
          "missing-qhead layout header should parse");
    ds4_glm52_layout_ownership_plan plan;
    st = ds4_glm52_layout_validate_ownership(&spec, 2, &plan, &result);
    check(st == DS4_GLM52_L0_STATUS_INVALID,
          "missing Q-head span should be rejected");
    check(strstr(result.message, "missing Q-head") != NULL,
          "missing Q-head should be named in error");
}

static void test_layout_overlap_qhead_fails(void) {
    ensure_layout_fixture();
    ds4_glm52_layout_spec spec;
    ds4_glm52_l0_result result;
    ds4_glm52_l0_status st =
        ds4_glm52_layout_read_manifest(g_layout_overlap_qhead_path, &spec, &result);
    check(st == DS4_GLM52_L0_STATUS_OK,
          "overlap-qhead layout header should parse");
    ds4_glm52_layout_ownership_plan plan;
    st = ds4_glm52_layout_validate_ownership(&spec, 2, &plan, &result);
    check(st == DS4_GLM52_L0_STATUS_INVALID,
          "overlapping Q-head span should be rejected");
    check(strstr(result.message, "overlapping") != NULL,
          "overlapping Q-head should be named in error");
}

static void test_layout_replicated_visible_all_ranks(void) {
    ensure_layout_fixture();
    ds4_glm52_layout_spec spec;
    ds4_glm52_l0_result result;
    ds4_glm52_l0_status st =
        ds4_glm52_layout_read_manifest(g_layout_valid_path, &spec, &result);
    check(st == DS4_GLM52_L0_STATUS_OK, "valid layout should parse");

    /* The replicated 'shared_emb' entry should be visible in each rank's own
     * per-rank layout. Mutate the rank-local entries in memory so this test
     * does not depend on accepting a global manifest with foreign owners. */
    for (int r = 0; r < 4; r++) {
        ds4_glm52_layout_spec rank_spec = spec;
        for (int i = 0; i < rank_spec.entry_count; i++) {
            ds4_glm52_layout_entry *entry = &rank_spec.entries[i];
            if (!entry->replicated &&
                entry->scope == DS4_GLM52_LAYOUT_SCOPE_RANK_LOCAL) {
                entry->rank = r;
                if (entry->role == DS4_GLM52_LAYOUT_ROLE_Q_HEAD) {
                    entry->q_head_start =
                        r * DS4_GLM52_L0_Q_HEADS_PER_RANK;
                    entry->q_head_end =
                        entry->q_head_start +
                        DS4_GLM52_L0_Q_HEADS_PER_RANK;
                } else if (entry->role == DS4_GLM52_LAYOUT_ROLE_EXPERT) {
                    entry->expert_start = 0;
                    entry->expert_end = DS4_GLM52_L0_EXPERTS;
                } else if (entry->role == DS4_GLM52_LAYOUT_ROLE_VOCAB) {
                    entry->vocab_start =
                        (DS4_GLM52_L0_VOCAB_SIZE * r) /
                        DS4_GLM52_L0_TP_SIZE;
                    entry->vocab_end =
                        (DS4_GLM52_L0_VOCAB_SIZE * (r + 1)) /
                        DS4_GLM52_L0_TP_SIZE;
                }
                snprintf(entry->file_path,
                         sizeof(entry->file_path),
                         "rank%d/shard-%02d.bin",
                         r,
                         i);
            }
        }
        ds4_glm52_layout_ownership_plan plan;
        ds4_glm52_l0_result r2;
        st = ds4_glm52_layout_validate_ownership(&rank_spec, r, &plan, &r2);
        check(st == DS4_GLM52_L0_STATUS_OK,
              "per-rank layout ownership should validate");
        bool found_replicated = false;
        for (int i = 0; i < plan.entry_count; i++) {
            if (plan.entries[i].replicated &&
                !strcmp(plan.entries[i].tensor_name, "shared_emb")) {
                found_replicated = true;
                break;
            }
        }
        check(found_replicated,
              "replicated tensor should be visible on all ranks");
    }
}

static void test_layout_sharded_only_on_owning_rank(void) {
    ensure_layout_fixture();
    ds4_glm52_layout_spec spec;
    ds4_glm52_l0_result result;
    ds4_glm52_l0_status st =
        ds4_glm52_layout_read_manifest(g_layout_valid_path, &spec, &result);
    check(st == DS4_GLM52_L0_STATUS_OK, "valid layout should parse");

    /* The rank-local 'expert_0' entry (rank 2) should be visible on rank 2
     * but NOT on rank 0. */
    ds4_glm52_layout_ownership_plan plan_r2;
    ds4_glm52_l0_result r2;
    ds4_glm52_layout_validate_ownership(&spec, 2, &plan_r2, &r2);
    bool found_on_r2 = false;
    for (int i = 0; i < plan_r2.entry_count; i++) {
        if (!strcmp(plan_r2.entries[i].tensor_name, "expert_0")) {
            found_on_r2 = true;
            break;
        }
    }
    check(found_on_r2, "sharded tensor should be visible on owning rank 2");

    ds4_glm52_layout_ownership_plan plan_r0;
    ds4_glm52_l0_result r0;
    st = ds4_glm52_layout_validate_ownership(&spec, 0, &plan_r0, &r0);
    check(st == DS4_GLM52_L0_STATUS_INVALID,
          "rank 0 should reject rank 2's per-rank layout");
    bool found_on_r0 = false;
    for (int i = 0; i < plan_r0.entry_count; i++) {
        if (!strcmp(plan_r0.entries[i].tensor_name, "expert_0")) {
            found_on_r0 = true;
            break;
        }
    }
    check(!found_on_r0,
          "sharded tensor should NOT be visible on non-owning rank 0");
}

static void test_layout_missing_shard_file_fails(void) {
    ensure_layout_fixture();
    ds4_glm52_layout_spec spec;
    ds4_glm52_l0_result result;
    ds4_glm52_l0_status st =
        ds4_glm52_layout_read_manifest(g_layout_missing_file_path, &spec, &result);
    check(st == DS4_GLM52_L0_STATUS_OK,
          "missing-file layout header should parse");
    ds4_glm52_layout_ownership_plan plan;
    st = ds4_glm52_layout_validate_ownership(&spec, 2, &plan, &result);
    check(st == DS4_GLM52_L0_STATUS_OK,
          "missing-file layout ownership should pass (file not checked yet)");
    ds4_glm52_layout_mapped_slices mapped;
    st = ds4_glm52_layout_mmap_rank_tensors(&plan, g_model_root,
                                            &mapped, &result);
    check(st == DS4_GLM52_L0_STATUS_INVALID,
          "missing required shard file should fail at mmap");
    check(strstr(result.message, "missing required shard file") != NULL,
          "missing shard file should be named in error");
}

static void test_layout_sha256_mismatch_fails(void) {
    ensure_layout_fixture();
    ds4_glm52_layout_spec spec;
    ds4_glm52_l0_result result;
    ds4_glm52_l0_status st =
        ds4_glm52_layout_read_manifest(g_layout_bad_sha_path, &spec, &result);
    check(st == DS4_GLM52_L0_STATUS_OK,
          "bad-sha layout header should parse");
    ds4_glm52_layout_ownership_plan plan;
    st = ds4_glm52_layout_validate_ownership(&spec, 2, &plan, &result);
    check(st == DS4_GLM52_L0_STATUS_OK,
          "bad-sha layout ownership should pass before file verification");
    ds4_glm52_layout_mapped_slices mapped;
    st = ds4_glm52_layout_mmap_rank_tensors(&plan, g_model_root,
                                            &mapped, &result);
    check(st == DS4_GLM52_L0_STATUS_INVALID,
          "sha256 mismatch should fail during rank tensor load");
    check(strstr(result.message, "sha256 mismatch") != NULL,
          "sha256 mismatch should be named in error");
    check(!mapped.mapped, "sha256 mismatch must not claim mapped tensors");
    check(!mapped.hash_verified,
          "sha256 mismatch must not claim hash verification");
}

static void test_layout_missing_metadata_fails(void) {
    ensure_layout_fixture();
    ds4_glm52_layout_spec spec;
    ds4_glm52_l0_result result;
    ds4_glm52_l0_status st =
        ds4_glm52_layout_read_manifest(g_layout_missing_metadata_path,
                                       &spec,
                                       &result);
    check(st == DS4_GLM52_L0_STATUS_INVALID,
          "layout with missing dtype/shape metadata should fail");
    check(strstr(result.message, "metadata") != NULL ||
              strstr(result.message, "dtype") != NULL,
          "missing metadata failure should name metadata or dtype");
}

static void test_layout_metadata_byte_count_mismatch_fails(void) {
    ensure_layout_fixture();
    ds4_glm52_layout_spec spec;
    ds4_glm52_l0_result result;
    ds4_glm52_l0_status st =
        ds4_glm52_layout_read_manifest(g_layout_bad_metadata_bytes_path,
                                       &spec,
                                       &result);
    check(st == DS4_GLM52_L0_STATUS_INVALID,
          "layout with dtype/shape byte mismatch should fail");
    check(strstr(result.message, "byte_length") != NULL,
          "metadata byte mismatch should name byte_length");
}

/* ---- prefill child graph tests ---- */

typedef struct {
    const char *graph_ids[8];
    const char *ids[8];
    ds4_glm52_l0_status statuses[8];
    bool prerequisite_blocked[8];
    int n;
} prefill_trace_capture;

static void capture_prefill_trace(void *ud,
                                  const char *graph_id,
                                  const char *action_id,
                                  ds4_glm52_l0_status status,
                                  bool prerequisite_blocked,
                                  const ds4_glm52_l0_state *state) {
    prefill_trace_capture *cap = (prefill_trace_capture *)ud;
    check(graph_id && strcmp(graph_id, DS4_GLM52_PREFILL_GRAPH_ID) == 0,
          "prefill trace graph id must be prefill");
    check(state != NULL, "prefill trace state must be present");
    check(cap->n < (int)(sizeof(cap->ids) / sizeof(cap->ids[0])),
          "prefill trace capture overflow");
    cap->graph_ids[cap->n] = graph_id;
    cap->ids[cap->n] = action_id;
    cap->statuses[cap->n] = status;
    cap->prerequisite_blocked[cap->n] = prerequisite_blocked;
    cap->n++;
}

static void test_prefill_action_order_and_phase_names(void) {
    const char *expected[] = {
        "prefill/a-tokenize-prompt",
        "prefill/a-plan-prefill-chunks",
        "prefill/a-prefill-layer-tp",
        "prefill/a-prefill-allreduce",
        "prefill/a-commit-prefill-kv",
        "prefill/a-prefill-ready",
    };
    check(ds4_glm52_prefill_action_count() ==
              sizeof(expected) / sizeof(expected[0]),
          "prefill child action count must match BCD prefill order");
    for (int i = 0; i < (int)ds4_glm52_prefill_action_count(); i++) {
        check(strcmp(ds4_glm52_prefill_action_id(i), expected[i]) == 0,
              "prefill child action id order mismatch");
    }
    check(strcmp(ds4_glm52_prefill_action_id(-1),
                 "prefill/action-invalid") == 0,
          "invalid prefill action index should be rejected");
    check(strcmp(ds4_glm52_l0_cursor_phase_name(
                     DS4_GLM52_L0_CURSOR_PHASE_NONE), "none") == 0,
          "none cursor phase name mismatch");
    check(strcmp(ds4_glm52_l0_cursor_phase_name(
                     DS4_GLM52_L0_CURSOR_PHASE_PREFILL), "prefill") == 0,
          "prefill cursor phase name mismatch");
    check(strcmp(ds4_glm52_l0_cursor_phase_name(
                     DS4_GLM52_L0_CURSOR_PHASE_DECODE), "decode") == 0,
          "decode cursor phase name mismatch");
}

static void test_prefill_set_prompt_validation(void) {
    ds4_glm52_l0_state state = {0};
    char err[160] = {0};
    int ids[4] = {11, 12, 13, 14};

    check(!ds4_glm52_l0_prefill_set_prompt(&state, "", ids, 4,
                                           err, sizeof(err)),
          "empty session id should be rejected");
    check(!ds4_glm52_l0_prefill_set_prompt(&state, "s", NULL, 4,
                                           err, sizeof(err)),
          "null token span with positive count should be rejected");
    check(!ds4_glm52_l0_prefill_set_prompt(
              &state, "s", ids, DS4_GLM52_L0_PREFILL_MAX_PROMPT_TOKENS + 1,
              err, sizeof(err)),
          "over-bound prompt should be rejected");
    check(!ds4_glm52_l0_prefill_set_prompt(&state, "s", ids, -1,
                                           err, sizeof(err)),
          "negative token count should be rejected");

    check(ds4_glm52_l0_prefill_set_prompt(&state, "s", ids, 4,
                                          err, sizeof(err)),
          "valid prompt span should be accepted");
    check(state.prompt.set && state.prompt.prompt_length == 4,
          "valid prompt should record length and set flag");
    check(strcmp(state.prompt.session_id, "s") == 0,
          "valid prompt should record session id");
    check(state.prompt.token_ids[0] == 11 &&
              state.prompt.token_ids[3] == 14 &&
              state.prompt.consumed == 0,
          "valid prompt should copy the token span");

    memset(&state, 0, sizeof(state));
    check(ds4_glm52_l0_prefill_set_prompt(&state, "s", NULL, 0,
                                          err, sizeof(err)),
          "empty prompt should be accepted");
    check(state.prompt.set && state.prompt.prompt_length == 0,
          "empty prompt should record zero length");
}

static void test_prefill_blocks_before_rank_plan_without_mutation(void) {
    ds4_glm52_l0_config cfg = valid_config();
    ds4_glm52_l0_state state = {0};
    mark_resident_shards(&state);
    const uint64_t kv_before = state.kv.length;
    const uint64_t cursor_before = state.cursor.token_step_j;
    ds4_glm52_l0_result result;
    ds4_glm52_l0_status status =
        ds4_glm52_l0_prefill_run(&cfg, &state, 0, NULL, NULL, &result);
    check(status == DS4_GLM52_L0_STATUS_NOT_READY,
          "prefill should block before rank-plan/topology readiness");
    check(strstr(result.message, "collective fabric") != NULL,
          "prefill block should name TP fabric prerequisite");
    check(state.kv.length == kv_before &&
              state.cursor.token_step_j == cursor_before &&
              state.cursor.kv_length == 0 &&
              !state.commit.committed &&
              !state.kv.prefill_complete,
          "blocked prefill must not mutate KV/cursor/commit state");
}

static void test_prefill_advances_cursors_deterministically(void) {
    ds4_glm52_l0_config cfg = mock_config();
    int ids[7] = {101, 102, 103, 104, 105, 106, 107};
    ds4_glm52_l0_state results[2] = {0};
    char err[160] = {0};

    for (int run = 0; run < 2; run++) {
        ds4_glm52_l0_state *state = &results[run];
        mark_readied_for_prefill(state);
        check(ds4_glm52_l0_prefill_set_prompt(state, "test-session",
                                              ids, 7, err, sizeof(err)),
              "prefill prompt should bind");
        ds4_glm52_l0_result result;
        ds4_glm52_l0_status status =
            ds4_glm52_l0_stub_action(DS4_GLM52_L0_ACTION_PREFILL,
                                     &cfg, state, &result);
        check(status == DS4_GLM52_L0_STATUS_OK,
              "readied prefill should succeed");
        check(result.action == DS4_GLM52_L0_ACTION_PREFILL,
              "prefill result should project to serve prefill action");
        check(state->kv.length == 7 &&
                  state->cursor.kv_length == 7 &&
                  state->cursor.token_step_j == 7,
              "prefill should advance KV and token cursors to prompt length");
        check(state->kv.prefill_complete &&
                  state->cursor.replicated &&
                  state->cursor.phase == DS4_GLM52_L0_CURSOR_PHASE_PREFILL,
              "prefill should leave decode-ready replicated cursor");
        check(state->kv.session_id &&
                  strcmp(state->kv.session_id, "test-session") == 0,
              "prefill should bind KV session identity to the request");
        check(state->commit.prefilled_length == 7 &&
                  state->commit.tp_degree == DS4_GLM52_L0_TP_SIZE &&
                  state->commit.dcp_degree == DS4_GLM52_L0_DCP_SIZE &&
                  state->commit.committed,
              "prefill should produce replicated decode-handoff commit");
        check(state->commit.kv_cache_dtype &&
                  strcmp(state->commit.kv_cache_dtype,
                         DS4_GLM52_L0_KV_CACHE_DTYPE) == 0,
              "prefill commit should record the fp8_ds_mla cache dtype");
        check(state->prompt.consumed == 7 &&
                  state->chunk.end_position == 7,
              "prefill should consume the full prompt span");
    }
    check(results[0].cursor.kv_length == results[1].cursor.kv_length &&
              results[0].cursor.token_step_j ==
                  results[1].cursor.token_step_j &&
              results[0].kv.length == results[1].kv.length,
          "prefill cursor advancement must be deterministic");
}

static void test_prefill_multi_chunk_append_order_and_trace(void) {
    ds4_glm52_l0_config cfg = mock_config();
    ds4_glm52_l0_state state;
    memset(&state, 0, sizeof(state));
    mark_readied_for_prefill(&state);
    int ids[37];
    for (int i = 0; i < 37; i++) ids[i] = 2000 + i;
    char err[160] = {0};
    check(ds4_glm52_l0_prefill_set_prompt(&state, "test-session",
                                          ids, 37, err, sizeof(err)),
          "multi-token prompt should bind");

    prefill_trace_capture cap = {0};
    ds4_glm52_l0_result result;
    ds4_glm52_l0_status status =
        ds4_glm52_l0_prefill_run(&cfg, &state, 16,
                                 capture_prefill_trace, &cap, &result);
    check(status == DS4_GLM52_L0_STATUS_OK,
          "multi-chunk prefill should succeed");
    check(cap.n == 6,
          "prefill child should trace all six BCD actions on success");
    const char *expected[] = {
        "prefill/a-tokenize-prompt",
        "prefill/a-plan-prefill-chunks",
        "prefill/a-prefill-layer-tp",
        "prefill/a-prefill-allreduce",
        "prefill/a-commit-prefill-kv",
        "prefill/a-prefill-ready",
    };
    for (int i = 0; i < cap.n; i++) {
        check(strcmp(cap.ids[i], expected[i]) == 0,
              "prefill trace action order mismatch");
        check(cap.statuses[i] == DS4_GLM52_L0_STATUS_OK,
              "prefill success trace should be all-ok");
        check(!cap.prerequisite_blocked[i],
              "prefill success trace should not report blocked prerequisites");
    }

    /* 37 tokens / 16 chunk -> [0,16) [16,32) [32,37); last chunk is index 2. */
    check(state.chunk.chunk_index == 2 &&
              state.chunk.start_position == 32 &&
              state.chunk.end_position == 37 &&
              state.chunk.token_count == 5,
          "prefill should leave the final committed chunk metadata");
    check(state.kv.length == 37 &&
              state.cursor.kv_length == 37 &&
              state.cursor.token_step_j == 37,
          "multi-chunk prefill should commit the whole prompt once");
    check(state.commit.prefilled_length == 37,
          "prefill commit summary should be the full prompt length");
}

static void test_prefill_append_order_violation_rejected(void) {
    ds4_glm52_l0_config cfg = valid_config();
    ds4_glm52_l0_state state;
    memset(&state, 0, sizeof(state));
    mark_readied_for_prefill(&state);
    /* Corrupt the append-ordered KV cursor so prefill cannot append at 0. */
    state.kv.length = 5;
    state.cursor.kv_length = 5;
    int ids[10] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    char err[160] = {0};
    check(ds4_glm52_l0_prefill_set_prompt(&state, "test-session",
                                          ids, 10, err, sizeof(err)),
          "prompt should bind before append-order rejection");

    ds4_glm52_l0_result result;
    ds4_glm52_l0_status status =
        ds4_glm52_l0_stub_action(DS4_GLM52_L0_ACTION_PREFILL,
                                 &cfg, &state, &result);
    check(status == DS4_GLM52_L0_STATUS_INVALID,
          "prefill should reject an append-order violation");
    check(strstr(result.message, "append-ordered KV cursor") != NULL,
          "append-order rejection should name the KV cursor rule");
    check(strstr(result.message, "prefill/a-commit-prefill-kv") != NULL,
          "append-order rejection should project to the KV commit action");
    check(state.kv.length == 5 &&
              state.cursor.kv_length == 5 &&
              state.cursor.token_step_j == 0 &&
              state.prompt.consumed == 0 &&
              !state.commit.committed &&
              !state.kv.prefill_complete,
          "rejected prefill must leave KV/cursor state untouched");
}

static void test_prefill_rank_identity_mismatch_rejected(void) {
    ds4_glm52_l0_config cfg = valid_config();
    char err[160] = {0};
    int ids[3] = {7, 8, 9};

    /* Case 1: resident shard rank does not match the bound rank plan. */
    ds4_glm52_l0_state a;
    memset(&a, 0, sizeof(a));
    mark_readied_for_prefill(&a);
    a.resident_shards.rank = 1; /* rank_plan.rank and config rank are 2 */
    check(ds4_glm52_l0_prefill_set_prompt(&a, "test-session",
                                          ids, 3, err, sizeof(err)),
          "prompt should bind before identity rejection");
    ds4_glm52_l0_result ra;
    ds4_glm52_l0_status sa =
        ds4_glm52_l0_stub_action(DS4_GLM52_L0_ACTION_PREFILL,
                                 &cfg, &a, &ra);
    check(sa == DS4_GLM52_L0_STATUS_INVALID,
          "resident/model rank mismatch should be rejected");
    check(strstr(ra.message, "identity mismatch") != NULL,
          "rank identity rejection should name identity mismatch");
    check(a.kv.length == 0 && a.cursor.kv_length == 0 &&
              !a.commit.committed && !a.kv.prefill_complete,
          "identity rejection must not mutate KV/cursor state");

    /* Case 2: configured rank does not match the bound rank plan. */
    ds4_glm52_l0_state b;
    memset(&b, 0, sizeof(b));
    mark_readied_for_prefill(&b);
    check(ds4_glm52_l0_prefill_set_prompt(&b, "test-session",
                                          ids, 3, err, sizeof(err)),
          "prompt should bind for config mismatch case");
    ds4_glm52_l0_config cfg2 = valid_config();
    cfg2.rank = 1;
    ds4_glm52_l0_result rb;
    ds4_glm52_l0_status sb =
        ds4_glm52_l0_stub_action(DS4_GLM52_L0_ACTION_PREFILL,
                                 &cfg2, &b, &rb);
    check(sb == DS4_GLM52_L0_STATUS_INVALID,
          "config/rank-plan mismatch should be rejected");
    check(strstr(rb.message, "identity mismatch") != NULL,
          "config identity rejection should name identity mismatch");
}

static void test_prefill_real_kernel_seam_not_ready(void) {
    ds4_glm52_l0_config cfg = valid_config();
    ds4_glm52_l0_state state;
    memset(&state, 0, sizeof(state));
    mark_readied_for_prefill(&state);
    int ids[4] = {31, 32, 33, 34};
    char err[160] = {0};
    check(ds4_glm52_l0_prefill_set_prompt(&state, "test-session",
                                          ids, 4, err, sizeof(err)),
          "prompt should bind before real kernel seam");

    ds4_glm52_l0_result result;
    ds4_glm52_l0_status status =
        ds4_glm52_l0_stub_action(DS4_GLM52_L0_ACTION_PREFILL,
                                 &cfg, &state, &result);
    check(status == DS4_GLM52_L0_STATUS_NOT_READY,
          "non-mock prefill should stop at the real TP layer seam");
    check(strstr(result.message, "prefill/a-prefill-layer-tp") != NULL,
          "non-mock prefill block should name the TP layer action");
    check(strstr(result.message, "not implemented") != NULL,
          "non-mock prefill block should name missing real kernels");
    check(state.kv.length == 0 &&
              state.cursor.kv_length == 0 &&
              state.cursor.token_step_j == 0 &&
              state.prompt.consumed == 0 &&
              !state.commit.committed &&
              !state.kv.prefill_complete,
          "non-mock prefill block must not commit KV/cursor state");
}

static void test_prefill_empty_prompt_deterministic(void) {
    ds4_glm52_l0_config cfg = mock_config();
    char err[160] = {0};
    ds4_glm52_l0_state results[2] = {0};

    for (int run = 0; run < 2; run++) {
        ds4_glm52_l0_state *state = &results[run];
        mark_readied_for_prefill(state);
        check(ds4_glm52_l0_prefill_set_prompt(state, "test-session",
                                              NULL, 0, err, sizeof(err)),
              "empty prompt should bind");
        ds4_glm52_l0_result result;
        ds4_glm52_l0_status status =
            ds4_glm52_l0_stub_action(DS4_GLM52_L0_ACTION_PREFILL,
                                     &cfg, state, &result);
        check(status == DS4_GLM52_L0_STATUS_OK,
              "empty-prompt prefill should succeed deterministically");
        check(state->kv.length == 0 &&
                  state->cursor.kv_length == 0 &&
                  state->cursor.token_step_j == 0,
              "empty-prompt prefill should leave zero cursors");
        check(state->kv.prefill_complete &&
                  state->cursor.phase == DS4_GLM52_L0_CURSOR_PHASE_PREFILL,
              "empty-prompt prefill should still be decode-ready");
        check(state->commit.prefilled_length == 0 &&
                  state->commit.committed &&
                  state->chunk.end_position == 0 &&
                  state->prompt.consumed == 0,
              "empty-prompt prefill should commit an empty summary");
    }
    check(results[0].kv.length == results[1].kv.length &&
              results[0].cursor.token_step_j ==
                  results[1].cursor.token_step_j,
          "empty-prompt prefill must be repeatable");
}

static void test_prefill_rejects_double_append(void) {
    ds4_glm52_l0_config cfg = mock_config();
    char err[160] = {0};
    int ids[3] = {1, 2, 3};
    ds4_glm52_l0_state state;
    memset(&state, 0, sizeof(state));
    mark_readied_for_prefill(&state);
    check(ds4_glm52_l0_prefill_set_prompt(&state, "test-session",
                                          ids, 3, err, sizeof(err)),
          "prompt should bind for the first prefill");
    ds4_glm52_l0_result r1;
    check(ds4_glm52_l0_stub_action(DS4_GLM52_L0_ACTION_PREFILL,
                                   &cfg, &state, &r1) ==
              DS4_GLM52_L0_STATUS_OK,
          "first prefill should succeed");
    const uint64_t kv_after = state.kv.length;
    const uint64_t cursor_after = state.cursor.token_step_j;

    ds4_glm52_l0_result r2;
    ds4_glm52_l0_status s2 =
        ds4_glm52_l0_stub_action(DS4_GLM52_L0_ACTION_PREFILL,
                                 &cfg, &state, &r2);
    check(s2 == DS4_GLM52_L0_STATUS_INVALID,
          "re-running prefill on a completed cursor should be rejected");
    check(strstr(r2.message, "already-prefilled") != NULL,
          "double-append rejection should name the completed cursor");
    check(state.kv.length == kv_after &&
              state.cursor.token_step_j == cursor_after,
          "rejected double prefill must not advance cursors");
}

static void test_decode_requires_prefill_produced_cursor(void) {
    ds4_glm52_l0_config cfg = mock_config();
    char err[160] = {0};
    int ids[3] = {21, 22, 23};

    /* (a) decode from a prefill-produced cursor reaches the decode child. */
    ds4_glm52_l0_state produced;
    memset(&produced, 0, sizeof(produced));
    mark_readied_for_prefill(&produced);
    check(ds4_glm52_l0_prefill_set_prompt(&produced, "test-session",
                                          ids, 3, err, sizeof(err)),
          "prompt should bind for prefill-produced decode case");
    ds4_glm52_l0_result rp;
    check(ds4_glm52_l0_stub_action(DS4_GLM52_L0_ACTION_PREFILL,
                                   &cfg, &produced, &rp) ==
              DS4_GLM52_L0_STATUS_OK,
          "prefill should produce decode-ready state");
    ds4_glm52_l0_result rd;
    ds4_glm52_l0_status sd =
        ds4_glm52_l0_stub_action(DS4_GLM52_L0_ACTION_DECODE_TOKEN,
                                 &cfg, &produced, &rd);
    check(sd == DS4_GLM52_L0_STATUS_NOT_READY,
          "decode from a prefill-produced cursor should pass prerequisites");
    check(strstr(rd.message, "decode-token child graph") != NULL,
          "prefill-produced cursor should reach the decode child stub");

    /* (b) manufactured but consistent decode cursor (phase=decode) passes. */
    ds4_glm52_l0_state man;
    memset(&man, 0, sizeof(man));
    mark_request(&man);
    man.kv.tp_aware = true;
    man.kv.append_ordered = true;
    man.kv.prefill_complete = true;
    man.kv.length = 3;
    man.cursor.replicated = true;
    man.cursor.kv_length = 3;
    man.cursor.token_step_j = 3;
    man.cursor.phase = DS4_GLM52_L0_CURSOR_PHASE_DECODE;
    ds4_glm52_l0_result rm;
    ds4_glm52_l0_status sm =
        ds4_glm52_l0_stub_action(DS4_GLM52_L0_ACTION_DECODE_TOKEN,
                                 &cfg, &man, &rm);
    check(sm == DS4_GLM52_L0_STATUS_NOT_READY,
          "manufactured consistent decode cursor should pass prerequisites");
    check(strstr(rm.message, "decode-token child graph") != NULL,
          "manufactured decode cursor should reach the decode child stub");

    /* (c) phase=NONE (no producing phase) blocks decode. */
    ds4_glm52_l0_state none_s;
    memset(&none_s, 0, sizeof(none_s));
    mark_request(&none_s);
    none_s.kv.tp_aware = true;
    none_s.kv.append_ordered = true;
    none_s.kv.prefill_complete = true;
    none_s.kv.length = 3;
    none_s.cursor.replicated = true;
    none_s.cursor.kv_length = 3;
    none_s.cursor.token_step_j = 3;
    none_s.cursor.phase = DS4_GLM52_L0_CURSOR_PHASE_NONE;
    ds4_glm52_l0_result rn;
    ds4_glm52_l0_status sn =
        ds4_glm52_l0_stub_action(DS4_GLM52_L0_ACTION_DECODE_TOKEN,
                                 &cfg, &none_s, &rn);
    check(sn == DS4_GLM52_L0_STATUS_NOT_READY,
          "decode should require a producing cursor phase");
    check(strstr(rn.message, "prefill-complete") != NULL,
          "decode block should name the prefill-produced cursor prerequisite");

    /* (d) cursor/length divergence blocks decode. */
    ds4_glm52_l0_state div;
    memset(&div, 0, sizeof(div));
    mark_request(&div);
    div.kv.tp_aware = true;
    div.kv.append_ordered = true;
    div.kv.prefill_complete = true;
    div.kv.length = 3;
    div.cursor.replicated = true;
    div.cursor.kv_length = 5;
    div.cursor.token_step_j = 5;
    div.cursor.phase = DS4_GLM52_L0_CURSOR_PHASE_PREFILL;
    ds4_glm52_l0_result rv;
    ds4_glm52_l0_status sv =
        ds4_glm52_l0_stub_action(DS4_GLM52_L0_ACTION_DECODE_TOKEN,
                                 &cfg, &div, &rv);
    check(sv == DS4_GLM52_L0_STATUS_NOT_READY,
          "decode should reject a diverged KV cursor");
    check(strstr(rv.message, "prefill-complete") != NULL,
          "diverged cursor block should name the cursor prerequisite");
}

static uint32_t full_rank_mask(void) {
    return (UINT32_C(1) << DS4_GLM52_L0_RANK_COUNT) - UINT32_C(1);
}

static ds4_glm52_tp4_collective_request valid_real_collective_request(void) {
    ds4_glm52_tp4_collective_request req;
    memset(&req, 0, sizeof(req));
    req.frame_version = DS4_GLM52_TP4_COLLECTIVE_FRAME_VERSION;
    req.kind = DS4_GLM52_TP4_COLLECTIVE_ATTN;
    req.rank = 2;
    req.tp_size = DS4_GLM52_L0_TP_SIZE;
    req.dcp_size = DS4_GLM52_L0_DCP_SIZE;
    req.rank_count = DS4_GLM52_L0_RANK_COUNT;
    req.layer_index = 7;
    req.dtype = DS4_GLM52_TP4_TENSOR_DTYPE_F32;
    req.seq = 44;
    req.model_hash = 55;
    req.session_hash = 66;
    req.token_step_j = 19;
    req.element_count = 6144;
    req.shape_hash = 0x6144f32u;
    req.byte_count =
        req.element_count * ds4_glm52_tp4_tensor_dtype_size(req.dtype);
    req.participant_mask = full_rank_mask();
    req.topology_ready = true;
    req.transport_ready = true;
    req.rank_local_partial_ready = true;
    req.replicated_output_ready = true;
    return req;
}

static void test_real_collective_frontier_fails_closed(void) {
    ds4_glm52_l0_result result;
    ds4_glm52_tp4_collective_request req =
        valid_real_collective_request();
    char err[192] = "";

    check(strcmp(ds4_glm52_tp4_tensor_dtype_name(
                     DS4_GLM52_TP4_TENSOR_DTYPE_BF16), "bf16") == 0,
          "tensor dtype names should be stable");
    check(ds4_glm52_tp4_tensor_dtype_size(
              DS4_GLM52_TP4_TENSOR_DTYPE_FP8_E4M3) == 1,
          "tensor dtype byte sizes should be stable");
    check(strcmp(ds4_glm52_tp4_tensor_dtype_name(
                     DS4_GLM52_TP4_TENSOR_DTYPE_I32), "i32") == 0 &&
              ds4_glm52_tp4_tensor_dtype_size(
                  DS4_GLM52_TP4_TENSOR_DTYPE_I32) == 4,
          "packed checkpoint dtype metadata should support i32");
    check(ds4_glm52_tp4_collective_frame_validate(
              &req, err, sizeof(err)),
          "valid collective frame metadata should pass standalone validation");
    check(DS4_GLM52_TP4_COLLECTIVE_WIRE_SIZE == 96u,
          "collective wire frame size should be stable");
    unsigned char wire[DS4_GLM52_TP4_COLLECTIVE_WIRE_SIZE];
    check(ds4_glm52_tp4_collective_frame_encode(
              &req, wire, sizeof(wire), err, sizeof(err)),
          "valid collective frame should encode");
    check(wire[0] == 0 && wire[1] == 0 && wire[2] == 0 && wire[3] == 1,
          "collective frame version should be big-endian on the wire");
    ds4_glm52_tp4_collective_request decoded;
    check(ds4_glm52_tp4_collective_frame_decode(
              wire, sizeof(wire), &decoded, err, sizeof(err)),
          "valid collective frame should decode");
    check(decoded.kind == req.kind &&
          decoded.rank == req.rank &&
          decoded.dtype == req.dtype &&
          decoded.seq == req.seq &&
          decoded.model_hash == req.model_hash &&
          decoded.session_hash == req.session_hash &&
          decoded.token_step_j == req.token_step_j &&
          decoded.element_count == req.element_count &&
          decoded.shape_hash == req.shape_hash &&
          decoded.byte_count == req.byte_count &&
          decoded.participant_mask == req.participant_mask,
          "decoded collective frame should preserve metadata identity");
    wire[87] ^= 1u;
    check(!ds4_glm52_tp4_collective_frame_decode(
              wire, sizeof(wire), &decoded, err, sizeof(err)),
          "corrupted collective frame byte count should be rejected");
    check(strstr(err, "byte count") != NULL,
          "wire byte-count rejection should name byte count");

    req.frame_version = 99;
    ds4_glm52_l0_status status =
        ds4_glm52_tp4_real_collective_allreduce(&req, &result);
    check(status == DS4_GLM52_L0_STATUS_INVALID,
          "real collective should reject unknown frame versions");
    check(strstr(result.message, "frame version") != NULL,
          "frame version rejection should name the metadata frame");

    req = valid_real_collective_request();
    req.dtype = DS4_GLM52_TP4_TENSOR_DTYPE_INVALID;
    status = ds4_glm52_tp4_real_collective_allreduce(&req, &result);
    check(status == DS4_GLM52_L0_STATUS_INVALID,
          "real collective should reject unsupported dtype");
    check(strstr(result.message, "dtype") != NULL,
          "dtype rejection should name tensor dtype");

    req = valid_real_collective_request();
    req.shape_hash = 0;
    status = ds4_glm52_tp4_real_collective_allreduce(&req, &result);
    check(status == DS4_GLM52_L0_STATUS_INVALID,
          "real collective should reject missing shape hash");
    check(strstr(result.message, "shape hash") != NULL,
          "shape-hash rejection should name tensor shape evidence");

    req = valid_real_collective_request();
    req.byte_count--;
    status = ds4_glm52_tp4_real_collective_allreduce(&req, &result);
    check(status == DS4_GLM52_L0_STATUS_INVALID,
          "real collective should reject mismatched dtype byte count");
    check(strstr(result.message, "byte count") != NULL,
          "byte-count rejection should name byte-count evidence");

    req = valid_real_collective_request();
    req.participant_mask = 0x7u;
    status = ds4_glm52_tp4_real_collective_allreduce(&req, &result);
    check(status == DS4_GLM52_L0_STATUS_INVALID,
          "real collective should reject missing rank participants");
    check(strstr(result.message, "all four rank participants") != NULL,
          "participant rejection should name all-rank collective requirement");

    req = valid_real_collective_request();
    req.transport_ready = false;
    status = ds4_glm52_tp4_real_collective_allreduce(&req, &result);
    check(status == DS4_GLM52_L0_STATUS_NOT_READY,
          "real collective should block without tensor transport");
    check(strstr(result.message, "transport") != NULL,
          "transport block should name collective transport readiness");

    req = valid_real_collective_request();
    status = ds4_glm52_tp4_real_collective_allreduce(&req, &result);
    check(status == DS4_GLM52_L0_STATUS_NOT_READY,
          "real collective should remain not_ready until tensor execution is wired");
    check(strstr(result.message, "not implemented") != NULL,
          "complete frontier should name unimplemented tensor execution");
}

static ds4_glm52_dcp_exchange_request valid_real_dcp_request(void) {
    ds4_glm52_dcp_exchange_request req;
    memset(&req, 0, sizeof(req));
    req.rank = 2;
    req.dcp_size = DS4_GLM52_L0_DCP_SIZE;
    req.rank_count = DS4_GLM52_L0_RANK_COUNT;
    req.layer_index = 7;
    req.seq = 45;
    req.model_hash = 55;
    req.session_hash = 66;
    req.token_step_j = 19;
    req.selected_row_count = 4;
    req.owner_rank_mask = full_rank_mask();
    req.ownership_plan_valid = true;
    req.append_ordered_kv = true;
    req.row_payload_ready = true;
    req.transport_ready = true;
    return req;
}

static void test_real_dcp_exchange_frontier_fails_closed(void) {
    ds4_glm52_l0_result result;
    ds4_glm52_dcp_exchange_request req = valid_real_dcp_request();

    req.owner_rank_mask = 0xbu;
    ds4_glm52_l0_status status =
        ds4_glm52_dcp_real_row_exchange(&req, &result);
    check(status == DS4_GLM52_L0_STATUS_INVALID,
          "real DCP exchange should reject incomplete owner maps");
    check(strstr(result.message, "complete four-rank owner map") != NULL,
          "DCP owner-map rejection should name complete owner map");

    req = valid_real_dcp_request();
    req.row_payload_ready = false;
    status = ds4_glm52_dcp_real_row_exchange(&req, &result);
    check(status == DS4_GLM52_L0_STATUS_NOT_READY,
          "real DCP exchange should block without row payloads");
    check(strstr(result.message, "row payloads") != NULL,
          "DCP payload block should name row payload readiness");

    req = valid_real_dcp_request();
    status = ds4_glm52_dcp_real_row_exchange(&req, &result);
    check(status == DS4_GLM52_L0_STATUS_NOT_READY,
          "real DCP exchange should remain not_ready until execution is wired");
    check(strstr(result.message, "not implemented") != NULL,
          "complete DCP frontier should name unimplemented execution");
}

static ds4_glm52_decode_real_request valid_real_decode_request(void) {
    ds4_glm52_decode_real_request req;
    memset(&req, 0, sizeof(req));
    req.rank = 2;
    req.input_token = 123;
    req.seq = 46;
    req.model_hash = 55;
    req.session_hash = 66;
    req.model_ready = true;
    req.tp_collectives_ready = true;
    req.dcp_exchange_ready = true;
    req.glm52_kernels_ready = true;
    return req;
}

static void test_real_decode_frontier_preserves_cursor(void) {
    ds4_glm52_l0_config cfg = valid_config();
    ds4_glm52_l0_state state;
    memset(&state, 0, sizeof(state));
    mark_prefill(&state);
    ds4_glm52_l0_state before = state;
    ds4_glm52_l0_result result;
    ds4_glm52_decode_real_request req = valid_real_decode_request();

    req.tp_collectives_ready = false;
    ds4_glm52_l0_status status =
        ds4_glm52_decode_real_step(&cfg, &req, &state, &result);
    check(status == DS4_GLM52_L0_STATUS_NOT_READY,
          "real decode should block without TP collectives");
    check(strstr(result.message, "TP4 all-reduce") != NULL,
          "real decode should name the missing collective backend");
    check(state.cursor.token_step_j == before.cursor.token_step_j &&
              state.cursor.kv_length == before.cursor.kv_length &&
              state.kv.length == before.kv.length,
          "blocked real decode must not mutate cursor or KV");

    req = valid_real_decode_request();
    status = ds4_glm52_decode_real_step(&cfg, &req, &state, &result);
    check(status == DS4_GLM52_L0_STATUS_NOT_READY,
          "real decode should remain not_ready until backend outputs are available");
    check(strstr(result.message, "backend logits") != NULL,
          "complete decode frontier should name missing backend output evidence");
    check(state.cursor.token_step_j == before.cursor.token_step_j &&
              state.cursor.kv_length == before.cursor.kv_length &&
              state.kv.length == before.kv.length &&
              !state.token.present,
          "incomplete real decode must preserve cursor, KV, and token state");

    req = valid_real_decode_request();
    req.logits_ready = true;
    req.sampled_token_ready = true;
    req.kv_append_committed = true;
    req.sampled_token = 321;
    status = ds4_glm52_decode_real_step(&cfg, &req, &state, &result);
    check(status == DS4_GLM52_L0_STATUS_OK,
          "real decode should commit complete backend output evidence");
    check(state.token.present &&
              state.token.token_id == 321 &&
              state.token.position == before.cursor.token_step_j,
          "real decode should publish the sampled token at prior position j");
    check(state.cursor.token_step_j == before.cursor.token_step_j + 1 &&
              state.cursor.kv_length == before.cursor.kv_length + 1 &&
              state.kv.length == before.kv.length + 1 &&
              state.cursor.phase == DS4_GLM52_L0_CURSOR_PHASE_DECODE,
          "real decode should advance token cursor and KV length by one");

    before = state;
    req = valid_real_decode_request();
    req.logits_ready = true;
    req.sampled_token_ready = true;
    req.kv_append_committed = true;
    req.sampled_token = DS4_GLM52_L0_VOCAB_SIZE;
    status = ds4_glm52_decode_real_step(&cfg, &req, &state, &result);
    check(status == DS4_GLM52_L0_STATUS_INVALID,
          "real decode should reject sampled token outside vocab");
    check(state.cursor.token_step_j == before.cursor.token_step_j &&
              state.cursor.kv_length == before.cursor.kv_length &&
              state.kv.length == before.kv.length &&
              state.token.token_id == before.token.token_id,
          "invalid sampled token must not mutate committed decode state");
}

int main(void) {
    test_action_order();
    test_validation();
    test_stub_does_not_mutate_cursor();
    test_orchestration_prerequisites();
    test_tp_group_binding();
    test_tp_group_l0_rendezvous_publish_ready();
    test_model_load_child_validation();
    test_model_load_manifest_rejections();
    test_run_reaches_root_seam();
    test_review_reaches_all_root_actions();
    /* model-shard-layout child tests */
    test_layout_valid_manifest_passes();
    test_model_load_layout_reaches_ready_rank_engines();
    test_model_load_layout_gpu_residency_reaches_ready_rank_engines();
    test_model_load_layout_gpu_residency_requires_runtime();
    test_layout_wrong_tp_fails();
    test_layout_foreign_rank_fails();
    test_layout_missing_qhead_fails();
    test_layout_overlap_qhead_fails();
    test_layout_replicated_visible_all_ranks();
    test_layout_sharded_only_on_owning_rank();
    test_layout_missing_shard_file_fails();
    test_layout_sha256_mismatch_fails();
    test_layout_missing_metadata_fails();
    test_layout_metadata_byte_count_mismatch_fails();
    test_layout_gpu_upload_missing_metadata_fails();
    /* prefill child graph tests */
    test_prefill_action_order_and_phase_names();
    test_prefill_set_prompt_validation();
    test_prefill_blocks_before_rank_plan_without_mutation();
    test_prefill_advances_cursors_deterministically();
    test_prefill_multi_chunk_append_order_and_trace();
    test_prefill_append_order_violation_rejected();
    test_prefill_rank_identity_mismatch_rejected();
    test_prefill_real_kernel_seam_not_ready();
    test_prefill_empty_prompt_deterministic();
    test_prefill_rejects_double_append();
    test_decode_requires_prefill_produced_cursor();
    test_real_collective_frontier_fails_closed();
    test_real_dcp_exchange_frontier_fails_closed();
    test_real_decode_frontier_preserves_cursor();
    return 0;
}
